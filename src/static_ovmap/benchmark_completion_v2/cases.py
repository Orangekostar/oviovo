"""Deterministic qualitative cases from already frozen first-scene predictions."""
from pathlib import Path
import numpy as np

from .common import canonical_digest, digest_file, source_identity, verified, write

METHODS = ('U00', 'U10', 'U01', 'U11')
METHOD_NAMES = ('F only', 'Multi-source', 'Calibrated F', 'Full system')


def choose_cases(pairs, before, after):
    selected = dict(corrected=None, harmed=None)
    counts = dict(matched=len(pairs), changed=0, corrected=0, harmed=0,
                  other_changed=0, unchanged=0)
    if len({pair['owner'] for pair in pairs}) != len(pairs):
        raise ValueError('Case selection requires unique matched native owners')
    for pair in sorted(pairs, key=lambda row: row['owner']):
        owner, target = pair['owner'], pair['class_id']
        old, new = before[owner], after[owner]
        if old == new:
            counts['unchanged'] += 1
            continue
        counts['changed'] += 1
        kind = 'corrected' if old != target and new == target else (
            'harmed' if old == target and new != target else 'other_changed')
        counts[kind] += 1
        if kind in selected and selected[kind] is None:
            selected[kind] = dict(pair)
    return dict(cases=selected, counts=counts)


def _sample(points, limit):
    return points if len(points) <= limit else points[np.linspace(0, len(points)-1, limit, dtype=np.int64)]


def _render(data, clouds, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Okabe-Ito accents and direct text labels keep status legible without color.
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig = plt.figure(figsize=(8.8, 5.1), facecolor='white')
        for column, kind in enumerate(('corrected', 'harmed')):
            left = .035 + .5 * column
            case = data['cases'][kind]
            title = 'Correction' if kind == 'corrected' else 'Harm'
            fig.text(left, .95, f"({'ab'[column]}) {title}", fontsize=13, weight='bold')
            if case is None:
                fig.text(left, .8, f'No {kind} instance in this scene.')
                continue
            fig.text(left, .905, f"GT: {case['target_name']}    IoU: {case['IoU']:.3f}", fontsize=11)
            ax = fig.add_axes([left-.025, .385, .46, .5], projection='3d')
            context, obj, width = clouds[kind]
            ax.scatter(*context.T, s=.6, color='#A6A6A6', alpha=.25,
                       linewidths=0, depthshade=False, rasterized=True)
            ax.scatter(*obj.T, s=2.2, color='#0072B2', alpha=.95,
                       linewidths=0, depthshade=False, rasterized=True)
            for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
                setter(-width/2, width/2)
            ax.set_box_aspect((1, 1, 1))
            ax.set_proj_type('ortho')
            ax.view_init(elev=25, azim=-60)
            ax.set_axis_off()
            fig.text(left, .38, f"Owner {case['owner']}; {case['mask_vertices']:,} vertices; crop {width:.2f} m",
                     fontsize=9, color='#666666')
            for row, (method, label) in enumerate(zip(METHODS, METHOD_NAMES, strict=True)):
                value = case['predictions'][method]
                y = .315 - .047 * row
                weight = 'bold' if method == 'U11' else 'normal'
                fig.text(left, y, label, fontsize=10, weight=weight)
                fig.text(left+.185, y, value['name'], fontsize=10, weight=weight)
                fig.text(left+.35, y, 'correct' if value['correct'] else 'incorrect', fontsize=9,
                         color='#00765A' if value['correct'] else '#B34B00')
        counts = data['selection']['counts']
        fig.text(.035, .08, f"All {counts['matched']} geometric matches: {counts['corrected']} corrected, "
                 f"{counts['harmed']} harmed, {counts['other_changed']} other label change(s).", fontsize=9)
        fig.text(.035, .04, f"{data['scene']}; seed 17. Blue: fixed instance. Gray: local context. "
                 'Masks are identical across all four methods.', fontsize=8.5)
        fig.savefig(folder/'figure.png', dpi=220, facecolor='white')
        fig.savefig(folder/'figure.pdf', facecolor='white',
                    metadata={'CreationDate': None, 'ModDate': None, 'Creator': 'OVMap frozen case renderer'})
        plt.close(fig)


def build(binding):
    """Read frozen diagnostics; never acquire GT, rerun models or change recipes."""
    root = Path(binding['output_root'])
    protocol = verified(root/'protocol_lock.json')
    benchmark = 'scannet200_val'
    scene = protocol['scopes'][benchmark]['scenes'][0]
    diagnostic_path = root/'diagnostics'/benchmark/scene/'receipt.json'
    if not diagnostic_path.is_file():
        return None
    diagnostic = verified(diagnostic_path)
    if diagnostic['status'] != 'DONE':
        return None
    prediction = verified(root/'benchmark/predictions'/benchmark/scene/'receipt.json')
    scored = verified(root/'benchmark/evaluation'/benchmark/scene/'receipt.json')
    bundle = verified(root/'benchmark/bundles'/benchmark/scene/'bundle.json')
    if (scored['status'] != 'DONE' or diagnostic['source']['score'] != scored['identity']
            or diagnostic['source']['prediction_lock'] != prediction['identity']
            or scored['prediction_lock'] != prediction['identity']
            or diagnostic['source']['bundle'] != bundle['identity']
            or diagnostic['source']['protocol'] != protocol['identity']):
        raise ValueError('Real cases require matching frozen prediction, score and diagnostic sources')
    sources = dict(protocol=protocol['identity'], prediction=prediction['identity'],
                   score=scored['identity'], diagnostic=diagnostic['identity'], bundle=bundle['identity'],
                   geometry=bundle['arrays']['geometry'])
    identity = canonical_digest(dict(sources=sources, producer=source_identity(('cases',)),
                                     seed=17, selection='FIRST_FORMAL_SCENE_MIN_OWNER_PER_CHANGE_KIND'))
    folder = root/'cases'/benchmark/scene
    receipt_path = folder/'receipt.json'
    if receipt_path.is_file():
        previous = verified(receipt_path)
        if previous['input_identity'] != identity:
            raise ValueError('Frozen real-case inputs changed; preserve the previous case first')
        for artifact in previous['artifacts'].values():
            path = Path(artifact['path'])
            if not path.is_file() or digest_file(path) != artifact['sha256']:
                raise ValueError('Frozen real-case artifact is missing or changed')
        return previous
    geometry_descriptor = bundle['arrays']['geometry']
    if digest_file(geometry_descriptor['path']) != geometry_descriptor['sha256']:
        raise ValueError('Real-case evaluation geometry differs from its frozen source')
    with np.load(geometry_descriptor['path'], allow_pickle=False) as arrays:
        geometry = {key: arrays[key].copy() for key in ('owners', 'indptr', 'indices', 'target_xyz')}
    labels = {}
    for method in METHODS:
        descriptor = prediction['methods'][method+'_seed17']
        manifest = verified(descriptor['path'])
        if manifest['identity'] != descriptor['identity']:
            raise ValueError('Real-case prediction manifest differs from its freeze')
        if digest_file(manifest['arrays']['path']) != manifest['arrays']['sha256']:
            raise ValueError('Real-case prediction arrays differ from their freeze')
        with np.load(manifest['arrays']['path'], allow_pickle=False) as arrays:
            if any(not np.array_equal(arrays[key], geometry[key]) for key in ('owners', 'indptr', 'indices')):
                raise ValueError('Real-case method changed the common frozen owner mask domain')
            labels[method] = {int(owner): int(label) for owner, label in zip(arrays['owners'], arrays['labels'], strict=True)}
    selection = choose_cases(diagnostic['coverage']['pairs'], labels['U10'], labels['U11'])
    expected = diagnostic['interventions']['U11_seed17-U10_seed17']
    if any(selection['counts'][key] != expected[key] for key in ('changed', 'corrected', 'harmed')):
        raise ValueError('Real-case selection disagrees with the frozen all-match diagnostic')
    names = dict(zip(bundle['text_ids'], bundle['text_names'], strict=True))
    names[0] = 'unknown'
    cases, clouds = {}, {}
    xyz = geometry['target_xyz']
    for kind, pair in selection['cases'].items():
        if pair is None:
            cases[kind] = None
            continue
        oi = pair['owner_index']
        if geometry['owners'][oi] != pair['owner']:
            raise ValueError('Case diagnostic owner index differs from frozen mask order')
        indices = geometry['indices'][geometry['indptr'][oi]:geometry['indptr'][oi+1]]
        obj = xyz[indices]
        low, high = obj.min(axis=0)-.5, obj.max(axis=0)+.5
        center = (low+high)/2
        width = float((high-low).max())
        in_context = ((xyz >= low) & (xyz <= high)).all(axis=1)
        in_context[indices] = False
        context = xyz[in_context]
        display_obj, display_context = _sample(obj, 6000), _sample(context, 18000)
        clouds[kind] = (display_context-center, display_obj-center, width)
        cases[kind] = dict(pair, target_name=names[pair['class_id']],
            mask_vertices=len(indices), context_vertices=len(context),
            displayed_object_points=len(display_obj), displayed_context_points=len(display_context),
            crop_width_metres=width, predictions={method: dict(class_id=values[pair['owner']],
                name=names[values[pair['owner']]], correct=values[pair['owner']]==pair['class_id'])
                for method, values in labels.items()})
    folder.mkdir(parents=True, exist_ok=True)
    data = write(folder/'data.json', dict(status='DONE', benchmark=benchmark, scene=scene, seed=17,
        sources=sources, selection=selection, cases=cases,
        selection_rule='FIRST_FORMAL_SCENE_MIN_OWNER_PER_CHANGE_KIND',
        geometric_matching='CLASS_INDEPENDENT_STRICT_IOU_GT_.5',
        same_frozen_mask_domain=True, display_only_subsampling=True,
        display=dict(context_margin_metres=.5, elevation_degrees=25, azimuth_degrees=-60,
                     object_color='#0072B2', context_color='#A6A6A6'),
        raw_geometry_GT_RGB_exported=False, new_model_calls=0, new_image_encodings=0, new_training_updates=0))
    _render(data, clouds, folder)
    counts = selection['counts']
    caption = (f'# Frozen semantic case study: {scene}\n\n'
        'The first formal ScanNet scene and seed17 are fixed. The smallest native owner ID in each '
        'U10-to-U11 correction/harm category is displayed, using the previously frozen strict-IoU > 0.5 '
        'class-independent matches. An absent category stays absent. '
        f"Among all {counts['matched']} matched instances, {counts['corrected']} are corrected, "
        f"{counts['harmed']} are harmed and {counts['other_changed']} undergo another label change.\n\n"
        'Blue marks the frozen instance and gray its local context. All four methods share the same mask; '
        'the panels compare semantic labels. Geometry is in metres with a 0.5 m context margin, equal axis scales, '
        'elevation25 and azimuth-60. At most6000 object and18000 context points are deterministically displayed; '
        'IoU and vertex counts use the complete masks. The examples do not replace full-benchmark AP.\n\n'
        'Exact source identities, all-match counts and the four labels are in `data.json`. '
        'Original point clouds, raw RGB and complete GT remain local.\n')
    (folder/'caption.md').write_text(caption)
    artifacts = {name: dict(path=str(folder/name), sha256=digest_file(folder/name), bytes=(folder/name).stat().st_size)
                 for name in ('data.json', 'caption.md', 'figure.png', 'figure.pdf')}
    return write(receipt_path, dict(status='DONE', input_identity=identity, sources=sources,
        benchmark=benchmark, scene=scene, seed=17, data=data['identity'], artifacts=artifacts,
        primary_visual_inspection='PENDING', synthetic=False, new_model_calls=0, new_training_updates=0))
