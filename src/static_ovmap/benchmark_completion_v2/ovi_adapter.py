"""Full original OVI native readout on the shared paper-200-frame protocol port."""
import ast
from pathlib import Path
import pickle
import numpy as np

from .common import (verified, read, write, save_arrays, digest_file, canonical_digest,
                     ConsumedInputs, source_identity, StageFailure)

PIN = '6304b5e93c14e66f25a01a64d3f650e0f1ccb0b9'


def predict(binding, benchmark, scene):
    """Reuse identical native geometry/N content; never consume F, MV or Q scores."""
    import subprocess
    import torch
    from plyfile import PlyData
    from static_ovmap.module_validation.scannet_study import (
        native_readout, native_surface_readout, project_values)
    root = Path(binding['output_root'])
    repo = root / 'assets/external/OVI-MAP'
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != PIN:
        raise ValueError('OVI origin implementation pin changed')
    bundle_path = root / 'benchmark/bundles' / benchmark / scene / 'bundle.json'
    native_path = root / 'benchmark/NQ' / benchmark / scene / 'native_features.pkl'
    if not bundle_path.is_file() or not native_path.is_file():
        raise StageFailure('Shared original-native capture/readout has not completed for this official scene',
                           missing=[str(p) for p in (bundle_path, native_path) if not p.is_file()])
    bundle = verified(bundle_path)
    scope = verified(root / 'protocol_lock.json')['scopes'][benchmark]
    capture = read(bundle['capture']['path'])
    native = verified(root / 'benchmark/native' / benchmark / scene / 'receipt.json')
    plan = verified(root / 'benchmark/frames' / benchmark / scene / 'frame_plan.json')
    index = ConsumedInputs(root)
    native_input = index.identity(native_path)
    text_path = native_path.with_name('target_text.npz')
    text_input = index.identity(text_path)
    postprocess = repo / 'scripts/utils/mesh_postprocess_utils.py'
    confidence_source = repo / 'scripts/eval_sem_seg.py'
    inputs = [index.identity(postprocess), index.identity(confidence_source), native_input, text_input]
    identity = canonical_digest(dict(bundle=bundle['identity'], native=native['identity'], inputs=inputs,
        frame_plan=plan['identity'], official_pin=PIN, valid_ids=scope['text_ids'],
        producer=source_identity(('ovi_adapter',))))
    folder = root / 'external/OVI-MAP/predictions' / benchmark / scene
    manifest_path = folder / 'manifest.json'
    if manifest_path.is_file():
        previous = verified(manifest_path)
        if previous['input_identity'] != identity:
            raise ValueError('Original OVI protocol-port prediction changed')
        index.identity(previous['arrays']['path'], previous['arrays'])
        return previous
    with native_path.open('rb') as stream:
        saved = pickle.load(stream)
    with np.load(text_path, allow_pickle=False) as arrays:
        text, canonical = arrays['text_embeddings'].copy(), arrays['canonical_embeddings'].copy()
        if not np.array_equal(arrays['valid_ids'], scope['text_ids']):
            raise ValueError('Original OVI text domain differs from the locked protocol')
    readout = native_readout(saved, text, canonical, tuple(scope['text_ids']))
    # Independently execute the pinned author's unchanged classifier function
    # on every available actual feature, rather than comparing two aliases.
    tree = ast.parse(postprocess.read_text())
    classifier = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'match_feature_to_label_embed')
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[classifier], type_ignores=[]), str(postprocess), 'exec'), namespace)
    checked = 0
    for owner, row in readout.items():
        if row['status'] != 'AVAILABLE':
            continue
        actual = namespace['match_feature_to_label_embed'](torch.as_tensor(row['feature']),
            torch.as_tensor(text), torch.as_tensor(canonical), True)
        if scope['text_ids'][actual] != row['class_id']:
            raise ValueError('Native readout differs from the pinned author classifier on a real object')
        checked += 1
    mesh = root / 'benchmark/native' / benchmark / scene / 'mapper/cropformer_inst' / (
        'instance_mesh_' + str(len(plan['frame_ids'])) + '.ply')
    index.identity(mesh)
    original = PlyData.read(str(mesh))
    colors = np.column_stack([original['vertex'][key] for key in ('red', 'green', 'blue')])
    with np.load(bundle['arrays']['geometry']['path'], allow_pickle=False) as arrays:
        faces = arrays['source_faces'].copy()
        xyz = np.column_stack([original['vertex'][key] for key in ('x', 'y', 'z')])
        if not np.array_equal(xyz, arrays['source_xyz']):
            raise ValueError('Original author mesh vertex order changed')
        source_owner, source_label = native_surface_readout(colors, faces, saved, readout)
        owners = project_values(source_owner, arrays['target_nearest'], arrays['target_matched'])
        labels = project_values(source_label, arrays['target_nearest'], arrays['target_matched'])
        point_count = len(arrays['target_xyz'])
    # Preserve the author's map_pred_mesh confidence (projected area divided
    # by the largest same-class instance), including its six-decimal export.
    owner_ids, counts = np.unique(owners, return_counts=True)
    instance_rows, maxima = [], {}
    for owner, count in zip(owner_ids, counts, strict=True):
        if owner == 0 or count < scope['minimum_region_vertices']:
            continue
        mask = owners == owner
        label = int(np.argmax(np.bincount(labels[mask])))
        if label:
            instance_rows.append((int(owner), label, np.flatnonzero(mask)))
        maxima[label] = max(maxima.get(label, 0), int(count))
    confidence = np.array([float(f'{len(mask) / maxima[label]:.6f}')
                           for owner, label, mask in instance_rows], np.float64)
    owner_ids = np.array([owner for owner, _, _ in instance_rows], np.int64)
    class_ids = np.array([label for _, label, _ in instance_rows], np.int64)
    indptr = np.r_[0, np.cumsum([len(mask) for _, _, mask in instance_rows])].astype(np.int64)
    indices = np.concatenate([mask for _, _, mask in instance_rows]) if instance_rows else np.empty(0, np.int64)
    path = folder / 'masks.npz'
    save_arrays(path, owners=owner_ids, indptr=indptr, indices=indices, labels=class_ids, confidence=confidence,
                exported=np.isin(class_ids, scope['instance_ids']))
    record = index.identity(path)
    result = write(manifest_path, dict(status='DONE', method='OVI-MAP', benchmark=benchmark, scene=scene,
        input_identity=identity, prediction_identity=canonical_digest(dict(arrays=record['sha256'], points=point_count)),
        arrays=record, target_points=point_count, confidence='AUTHOR_AREA_CURRENT_CLASS_RATIO_6_DECIMALS',
        variant='ORIGIN_IMPLEMENTATION_STANDARD_PROTOCOL_PORT', official_commit=PIN,
        proposal='AUTHOR_TRIANGLE_FIRST_COLOR_NATIVE_PAINT;ORIGINAL_OBSERVATION_RULES',
        GT_labels_used_for_prediction=False, new_GPU_encodings=0, new_training_updates=0,
        shared_inputs=dict(native_capture=native['identity'], native_features=native_input,
            original_SigLIP_text=text_input, frame_plan=plan['identity']),
        unused_sources=['F', 'MV', 'Q_SCORES'], actual_author_classifier_checked_objects=checked,
        original_N_at_least_two_observations=True, original_top10_then_last8_area_pool=True,
        boundary_adaptations=['paper evenly sampled200 finite-pose frames; explicit iterator',
            'original native reconstruction runtime deployment', 'target text/class IDs and original full mesh point order',
            'common official evaluator; author native confidence retained'],
        sources=inputs, compatible_paper_report_not_assumed=True))
    index.save()
    return result


def score(binding, benchmark, scene):
    """Freeze native prediction before score-only GT access."""
    from .standard_export import official_prediction
    from .standard_score import evaluator, ap_metrics, _json
    root = Path(binding['output_root'])
    manifest = predict(binding, benchmark, scene)
    gt_path = root / 'benchmark/GT' / benchmark / scene / 'receipt.json'
    if not gt_path.is_file():
        raise StageFailure('Shared official full-GT conversion has not completed', missing=[str(gt_path)])
    gt = verified(gt_path)
    inputs = official_prediction(manifest)
    index = ConsumedInputs(root)
    index.identity(gt['GT']['path'], gt['GT'])
    identity = canonical_digest(dict(prediction=manifest['identity'], GT=gt['identity'],
        producer=source_identity(('ovi_adapter', 'standard_score'))))
    folder = root / 'external/OVI-MAP/evaluation' / benchmark / scene
    path = folder / 'receipt.json'
    if path.is_file():
        prior = verified(path)
        if prior['input_identity'] != identity:
            raise ValueError('Completed OVI official score changed')
        return prior
    namespace = evaluator(binding, benchmark)
    left, right = namespace['assign_instances_for_scan'](inputs, gt['GT']['path'])
    matches = {scene: dict(gt=left, pred=right)}
    metrics, classes, _ = ap_metrics(namespace, matches)
    record = write(folder / 'matches.json.gz', dict(status='DONE', matches=_json(matches)))
    result = write(path, dict(status='DONE', input_identity=identity, method='OVI-MAP', benchmark=benchmark,
        scene=scene, metrics=metrics, per_class=classes, prediction=manifest['identity'], GT=gt['identity'],
        matches=dict(path=str(folder / 'matches.json.gz'), identity=record['identity']),
        protocol=verified(root / 'protocol_lock.json')['identity'], locally_executed=True,
        original_GT_instances=sum(len(v) for v in left.values()), full_GT_denominator=True,
        official_classifier_checked=True, confidence=manifest['confidence'],
        variant=manifest['variant'], scope='ONE_FORMAL_SCENE;NOT_FULL_BENCHMARK'))
    index.save()
    return result
