"""Score-only geometry coverage and locked conditional recognition diagnostics."""
from pathlib import Path
import numpy as np

from .common import (SEEDS, verified, write, save_arrays, ConsumedInputs,
                     canonical_digest, source_identity)


def geometry_matches(owners, indptr, indices, gt, class_ids, *, minimum=100):
    """Strict IoU>.5 is unambiguous for two disjoint instance partitions."""
    owners, indptr, indices, gt = [np.asarray(a, np.int64) for a in (owners, indptr, indices, gt)]
    if (gt.ndim != 1 or indptr.shape != (len(owners) + 1,) or indptr[0] != 0
            or indptr[-1] != len(indices) or (np.diff(indptr) < 0).any()
            or (indices < 0).any() or (indices >= len(gt)).any()
            or len(np.unique(indices)) != len(indices)):
        raise ValueError('Diagnostic requires the original disjoint native owner partition')
    ids, sizes = np.unique(gt, return_counts=True)
    valid = {int(i): int(n) for i, n in zip(ids, sizes, strict=True)
             if i >= 1000 and i // 1000 in class_ids and n >= minimum}
    pairs = []
    for oi, owner in enumerate(owners):
        mask = indices[indptr[oi]:indptr[oi + 1]]
        labels, intersections = np.unique(gt[mask], return_counts=True)
        for instance, intersection in zip(labels, intersections, strict=True):
            if int(instance) not in valid:
                continue
            iou = float(intersection / (len(mask) + valid[int(instance)] - intersection))
            if iou > .5:
                pairs.append(dict(owner_index=oi, owner=int(owner), GT_instance=int(instance),
                                  class_id=int(instance // 1000), IoU=iou))
    if (len({p['owner'] for p in pairs}) != len(pairs)
            or len({p['GT_instance'] for p in pairs}) != len(pairs)):
        raise ValueError('Strict half-IoU partition matching ceased to be unique')
    return dict(valid_GT_instances=len(valid), matched_GT_instances=len(pairs),
                proposal_recall=None if not valid else len(pairs) / len(valid), pairs=pairs,
                valid_GT=[dict(instance=i, class_id=i // 1000, vertices=n) for i, n in sorted(valid.items())],
                full_GT_denominator=True, class_independent_matching=True, IoU_rule='STRICT_GT_.5')


def _intervention(target, before, after):
    old, new = np.asarray(before) == target, np.asarray(after) == target
    return dict(corrected=int((~old & new).sum()), harmed=int((old & ~new).sum()),
                changed=int((np.asarray(before) != after).sum()),
                correct_before=int(old.sum()), correct_after=int(new.sum()), originals=len(target))


def scene(binding, benchmark, scene_name):
    """Only consume already frozen/scored predictions; never acquire/select GT."""
    from .standard_export import score_F
    from .calibration import zero_scores
    from static_ovmap.semantic_update_evidence_v1.metrics import recognition
    root = Path(binding['output_root'])
    scored_path = root / 'benchmark/evaluation' / benchmark / scene_name / 'receipt.json'
    scored = verified(scored_path)
    protocol = verified(root / 'protocol_lock.json')
    scope = protocol['scopes'][benchmark]
    bundle = verified(root / 'benchmark/bundles' / benchmark / scene_name / 'bundle.json')
    gt = verified(root / 'benchmark/GT' / benchmark / scene_name / 'receipt.json')
    f = verified(root / 'benchmark/F' / benchmark / scene_name / 'receipt.json')
    recipes = verified(root / 'simple_recipe_lock.json')
    prediction = verified(root / 'benchmark/predictions' / benchmark / scene_name / 'receipt.json')
    identity = canonical_digest(dict(score=scored['identity'], bundle=bundle['identity'], GT=gt['identity'],
        F=f['identity'], recipes=recipes['identity'], protocol=protocol['identity'],
        producer=source_identity(('diagnostics',))))
    folder = root / 'diagnostics' / benchmark / scene_name
    path = folder / 'receipt.json'
    if path.is_file():
        previous = verified(path)
        if previous['input_identity'] != identity:
            raise ValueError('Completed score-only diagnostic changed; preserve affected records first')
        return previous
    index = ConsumedInputs(root)
    index.identity(gt['GT']['path'], gt['GT'])
    index.identity(f['arrays']['path'], f['arrays'])
    with np.load(bundle['arrays']['geometry']['path'], allow_pickle=False) as arrays:
        geometry = {key: arrays[key].copy() for key in ('owners', 'indptr', 'indices')}
    with np.load(f['arrays']['path'], allow_pickle=False) as arrays:
        vectors = {key: arrays[key].copy() for key in arrays.files}
    values = np.loadtxt(gt['GT']['path'], dtype=np.int64)
    coverage = geometry_matches(**geometry, gt=values, class_ids=scope['instance_ids'],
                                minimum=scope['minimum_region_vertices'])
    if not np.array_equal(vectors['owners'], geometry['owners']):
        raise ValueError('Conditional recognition changed the common all-owner order')
    pairs = coverage['pairs']
    matched = np.array([p['owner_index'] for p in pairs], np.int64)
    targets = np.array([p['class_id'] for p in pairs], np.int64)
    coverage['matched_without_F'] = int((~vectors['available'][matched]).sum())
    mapping = scope['mapping']['base_mask']
    if mapping is None:
        return write(path, dict(status='BLOCKED_DEPENDENCY', input_identity=identity, benchmark=benchmark,
            scene=scene_name, coverage=coverage, rows=[], prediction_lock=prediction['identity'],
            missing=scope['mapping']['unresolved'], reason='UNRESOLVED_BASE_ROLE_PREVENTS_FULL_CALIBRATION_DIAGNOSTIC'))
    base_mask = np.asarray(mapping, bool)
    ids = vectors['valid_ids']
    base_ids, novel_ids = ids[base_mask].tolist(), ids[~base_mask].tolist()
    coverage['groups'] = {}
    for group, group_ids in (('Base', base_ids), ('Novel', novel_ids)):
        total = sum(p['class_id'] in group_ids for p in coverage['valid_GT'])
        seen = sum(p['class_id'] in group_ids for p in pairs)
        coverage['groups'][group] = dict(valid_GT_instances=total, matched_GT_instances=seen,
                                        proposal_recall=None if not total else seen / total)
    names = ['RAW', 'OFFSET', 'CAP10', 'CAP10_OFFSET', 'SEARCH_CAP', 'SEARCH_MIX', 'SEARCH_NMIX']
    scores = np.zeros((len(names), len(SEEDS), len(geometry['owners']), len(ids)), vectors['text'].dtype)
    predicted = np.zeros(scores.shape[:-1], np.int64)
    for si, seed in enumerate(SEEDS):
        for oi in np.flatnonzero(vectors['available']):
            b, z, text = vectors['b'][oi], vectors['z'][si, oi], vectors['text']
            raw, cap_post = score_F(b, z, text, base_mask, calibrated=True)
            offset = raw.copy()
            offset[base_mask] -= .025
            # Compute cap-only directly: subtract/add gamma in FP32 would
            # change the frozen operation at a rounding boundary.
            from static_ovmap.semantic_update_evidence_v1.calibration import angle_cap
            capped, _ = angle_cap(b[None], z[None], 10.)
            cap = text @ capped[0]
            row_scores = [raw, offset, cap, cap_post]
            for kind in ('CAP', 'MIX', 'NMIX'):
                recipe = recipes['benchmark_simple_controls'][kind]['recipe']
                value, _ = zero_scores(recipe, (text @ b)[None], raw[None], b[None], z[None], text)
                value = value[0].copy()
                value[base_mask] -= recipe['gamma']
                row_scores.append(value)
            for ri, row in enumerate(row_scores):
                if not np.isfinite(row).all():
                    raise ValueError('A supported locked conditional intervention produced nonfinite scores')
                scores[ri, si, oi] = row
                predicted[ri, si, oi] = ids[int(np.argmax(row))]
    rows, interventions = [], {}
    for ri, name in enumerate(names):
        for si, seed in enumerate(SEEDS):
            row = predicted[ri, si, matched]
            rows.append(dict(method=name, seed=seed, metrics=recognition(targets, row, base_ids, novel_ids),
                selection_role='FIXED_OR_EXPOSED_CONF12_LOCK;NO_BENCHMARK_SELECTION'))
            if ri:
                interventions[name + '_seed' + str(seed)] = _intervention(targets, predicted[0, si, matched], row)
    full_grid = {}
    for method, descriptor in prediction['methods'].items():
        manifest = verified(descriptor['path'])
        index.identity(manifest['arrays']['path'], manifest['arrays'])
        with np.load(manifest['arrays']['path'], allow_pickle=False) as arrays:
            if not np.array_equal(arrays['owners'], geometry['owners']):
                raise ValueError('GLOBAL four-cell owner domains differ in the real artifact')
            full_grid[method] = arrays['labels'][matched].copy()
    for si, seed in enumerate(SEEDS):
        for before, after in (('U00', 'U01'), ('U10', 'U11'), ('U01', 'U11')):
            left, right = before + '_seed' + str(seed), after + '_seed' + str(seed)
            if left in full_grid and right in full_grid:
                interventions[right + '-' + left] = _intervention(targets, full_grid[left], full_grid[right])
        # Exact fixed F rows must agree with actual MFoff exports, including
        # zero-F unknown0 and predictions of stuff classes.
        for label, ri in (('U00', 0), ('U01', 3)):
            if not np.array_equal(full_grid[label + '_seed' + str(seed)], predicted[ri, si, matched]):
                raise ValueError('Diagnostic F operation differs from the actual GLOBAL export')
    arrays_path = folder / 'conditional.npz'
    ranks = np.full((len(names), len(SEEDS), len(matched)), -1, np.int64)
    for pi, (oi, target) in enumerate(zip(matched, targets, strict=True)):
        if not vectors['available'][oi]:
            continue
        column = np.flatnonzero(ids == target)
        if len(column) != 1:
            raise ValueError('A conditional target is outside the frozen text axis')
        ci = int(column[0])
        value = scores[:, :, oi]
        ranks[:, :, pi] = 1 + (value > value[:, :, ci, None]).sum(-1) + (
            (value == value[:, :, ci, None]) & (np.arange(len(ids)) < ci)).sum(-1)
    save_arrays(arrays_path, owners=geometry['owners'], defined=vectors['available'], methods=np.array(names),
        predictions=predicted, scores=scores, valid_ids=ids, matched_owner_indices=matched, targets=targets,
        target_ranks=ranks)
    result = write(path, dict(status='DONE', input_identity=identity, benchmark=benchmark, scene=scene_name,
        coverage=coverage, rows=rows, interventions=interventions, arrays=index.identity(arrays_path),
        source=dict(score=scored['identity'], F=f['identity'], bundle=bundle['identity'], GT=gt['identity'],
                    prediction_lock=prediction['identity'], recipe_lock=recipes['identity'], protocol=protocol['identity']),
        geometry_matching_precedes_label_readout=True, all_geometric_matches_share_same_denominator=True,
        unsupported_matched_proposals_retained=True, condition='REAL_NATIVE_PROPOSAL_IOU_GT_.5',
        benchmark_GT_never_selects_recipe=True, substitutes_for_official_AP=False))
    index.save()
    return result


def collect(binding, benchmark):
    from static_ovmap.semantic_update_evidence_v1.metrics import recognition
    root = Path(binding['output_root'])
    scope = verified(root / 'protocol_lock.json')['scopes'][benchmark]
    records, targets, predictions = {}, [], {}
    for name in scope['scenes']:
        if not (root / 'benchmark/evaluation' / benchmark / name / 'receipt.json').is_file():
            continue
        row = scene(binding, benchmark, name)
        records[name] = row
        if row['status'] != 'DONE':
            continue
        with np.load(row['arrays']['path'], allow_pickle=False) as arrays:
            targets.append(arrays['targets'])
            for ri, method in enumerate(arrays['methods']):
                for si, seed in enumerate(SEEDS):
                    predictions.setdefault((str(method), seed), []).append(
                        arrays['predictions'][ri, si, arrays['matched_owner_indices']])
    rows = {}
    if predictions:
        role = np.asarray(scope['mapping']['base_mask'], bool)
        ids = np.asarray(scope['text_ids'])
        for (method, seed), parts in predictions.items():
            rows[method + '_seed' + str(seed)] = dict(method=method, seed=seed,
                metrics=recognition(np.concatenate(targets), np.concatenate(parts), ids[role], ids[~role]))
    denominator = sum(row['coverage']['valid_GT_instances'] for row in records.values())
    matched = sum(row['coverage']['matched_GT_instances'] for row in records.values())
    return write(root / 'diagnostics' / benchmark / 'summary.json', dict(status='DONE' if
        len(records) == len(scope['scenes']) and all(r['status'] == 'DONE' for r in records.values())
        else 'BLOCKED_DEPENDENCY', rows=rows,
        benchmark=benchmark, scene_receipts={name: row['identity'] for name, row in records.items()},
        full_scene_coverage=len(records) == len(scope['scenes']), completed_scenes=len(records), expected_scenes=len(scope['scenes']),
        scope='FULL_BENCHMARK' if len(records) == len(scope['scenes']) else 'PARTIAL_SUBSET',
        valid_GT_instances=denominator, matched_GT_instances=matched,
        proposal_recall=None if not denominator else matched / denominator,
        mean_scene_accuracy_not_used=True, conditional_recognition_not_official_AP=True,
        benchmark_GT_never_selects_recipe=True))
