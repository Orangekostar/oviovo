"""Independent full-GT Replica scoring shared by frozen external methods."""
from pathlib import Path

import numpy as np

from .common import ConsumedInputs, canonical_digest, source_identity, verified, write

def _folder(binding, adapter):
    return Path(binding['output_root']) / 'external' / adapter['method']

def _evaluator(binding):
    from .standard_score import evaluator
    from .score_compat import adapt
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    namespace = evaluator(binding, 'replica48')
    evidence = adapt(namespace, protocol['official_sources']['evaluator_scan']['path'])
    write(root / 'review/G3_evaluator_auxiliary_compat.json', dict(evidence,
        protocol=protocol['identity'], new_GPU_calls=0, new_training_updates=0))
    return namespace

def _xyz(path):
    from plyfile import PlyData
    points = PlyData.read(str(path))['vertex']
    xyz = np.column_stack([points[key] for key in ('x', 'y', 'z')]).astype(np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Replica author evaluation point domain is malformed')
    return xyz

def _prediction(binding, scene, path, adapter):
    from .standard_export import official_prediction
    manifest = verified(path)
    if (manifest['status'] != 'DONE' or manifest['benchmark'] != 'replica48'
            or manifest['scene'] != scene or manifest['method'] != adapter['method']):
        raise ValueError('A frozen external Replica prediction is required before GT access')
    prediction = official_prediction(manifest)  # Verify bytes before any GT read.
    provenance = manifest['provenance']
    if not provenance.get('synthetic', False):
        expected = adapter['freeze'](binding, scene)
        if expected['identity'] != manifest['identity']:
            raise ValueError('Prediction does not bind the complete actual original pipeline')
    index = ConsumedInputs(binding['output_root'])
    index.identity(provenance['geometry']['path'], provenance['geometry'])
    xyz = _xyz(provenance['geometry']['path'])
    if len(xyz) != manifest['target_points']:
        raise ValueError('Frozen prediction differs from the original evaluation point domain')
    index.save()
    return manifest, prediction, xyz

def _ground_truth(binding, scene, manifest, xyz, adapter):
    """Called only after _prediction has validated the complete frozen export."""
    from .replica_io import _tuple_path, encode_labels
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    source = _tuple_path(binding, scene)
    author = protocol['official_sources']['evaluator_replica']
    index = ConsumedInputs(root)
    inputs = [index.identity(source), index.identity(author['path'], author)]
    identity = canonical_digest(dict(prediction=manifest['identity'], inputs=inputs,
        geometry=manifest['provenance']['geometry'], producer=source_identity((adapter['producer'], 'replica_baseline_score', 'replica_io'))))
    folder = _folder(binding, adapter) / 'GT/replica48' / scene
    path = folder / 'receipt.json'
    if path.is_file():
        prior = verified(path)
        if prior['input_identity'] != identity:
            raise ValueError('Completed external Replica GT changed; preserve before affected repair')
        index.identity(prior['GT']['path'], prior['GT'])
        index.save()
        return prior
    import torch
    value = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError('Replica scoring requires the original author four-field instance tuple')
    coords, _, semantic, instances = [np.asarray(v) for v in value]
    if coords.shape != xyz.shape or not np.array_equal(coords.astype(np.float32), xyz):
        raise ValueError('Replica GT does not preserve the original point order')
    if (semantic.shape != (len(xyz),) or instances.shape != semantic.shape
            or not np.issubdtype(semantic.dtype, np.integer) or not np.issubdtype(instances.dtype, np.integer)):
        raise ValueError('Author Replica semantic/instance label axes are malformed')
    labels = encode_labels(semantic, instances, author['path'])
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / (scene + '.txt')
    np.savetxt(output, labels, fmt='%d')
    result = write(path, dict(status='DONE', input_identity=identity, benchmark='replica48', scene=scene,
        GT=index.identity(output), inputs=inputs, target_points=len(labels),
        prediction_freeze=manifest['identity'], geometry=manifest['provenance']['geometry'],
        full_GT_denominator=True, GT_filtered_by_prediction=False,
        definition='UNCHANGED_AUTHOR_OPEN3DIS_REPLICA_GT_PREFIX;EXACT_FP32_COORDINATE_ORDER;FULL_TUPLE'))
    index.save()
    return result

def score_prediction(binding, scene, manifest_path, adapter):
    from .standard_score import ap_metrics, _json
    from .replica_validation import author_evaluator
    manifest, prediction, xyz = _prediction(binding, scene, manifest_path, adapter)
    author, author_sources = author_evaluator(binding)
    gt = _ground_truth(binding, scene, manifest, xyz, adapter)
    protocol = verified(Path(binding['output_root']) / 'protocol_lock.json')
    identity = canonical_digest(dict(prediction=manifest['identity'], GT=gt['identity'],
        protocol=protocol['identity'], author_sources=author_sources,
        producer=source_identity((adapter['producer'], 'replica_baseline_score', 'standard_score', 'score_compat', 'replica_validation'))))
    folder = _folder(binding, adapter) / 'evaluation/replica48' / scene
    path = folder / 'receipt.json'
    if path.is_file():
        prior = verified(path)
        if prior['input_identity'] != identity:
            raise ValueError('Completed external Replica score changed; preserve before affected repair')
        evidence = verified(prior['matches']['path'])
        if evidence['identity'] != prior['matches']['identity']:
            raise ValueError('Frozen external Replica match statistics changed')
        return prior
    namespace = _evaluator(binding)
    gt2pred, pred2gt = namespace['assign_instances_for_scan'](prediction, gt['GT']['path'])
    matches = {scene: dict(gt=gt2pred, pred=pred2gt)}
    metrics, classes, common_ap = ap_metrics(namespace, matches)
    import torch
    value = torch.load(gt['inputs'][0]['path'], map_location='cpu', weights_only=False)
    semantic, instances = (np.asarray(value[i]) for i in (2, 3))
    native = [dict(scan_id=scene, label_id=int(label), conf=float(conf), pred_mask=prediction['pred_masks'][:, i])
              for i, (label, conf) in enumerate(zip(prediction['pred_classes'], prediction['pred_scores'], strict=True))]
    original_gt, original_pred = author.assign_instances_for_scan(native, semantic.copy(), instances.copy())
    author_ap, author_rc = author.evaluate_matches({scene: dict(gt=original_gt, pred=original_pred)})
    average = author.compute_averages(author_ap, author_rc)
    expected = np.array([average[key] for key in ('all_ap', 'all_ap_50%', 'all_ap_25%')])
    actual = np.array([metrics[key] for key in ('apall', 'ap50', 'ap25')], dtype=float)
    if (not np.allclose(common_ap, author_ap, rtol=0, atol=1e-12, equal_nan=True)
            or not np.allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)):
        raise ValueError('external Replica common AP differs from the unchanged author evaluator')
    finite = np.isfinite(common_ap) & np.isfinite(author_ap)
    error = float(np.max(np.abs(common_ap[finite] - author_ap[finite]))) if finite.any() else 0.
    synthetic = bool(manifest['provenance'].get('synthetic', False))
    agreement = dict(status='DONE', scene=scene, synthetic=synthetic,
        all_class_IoU_AP_arrays_equal=True, maximum_absolute_error=error,
        actual_common=_json(actual), actual_author=_json(expected), sources=author_sources,
        full_original_point_domain=True, native_confidence_preserved=True)
    write(folder / 'G3_official_entry_agreement.json', agreement)
    record = write(folder / (adapter['method'] + '_matches.json.gz'), dict(status='DONE', matches=_json(matches)))
    return write(path, dict(status='DONE', input_identity=identity, method=adapter['method'],
        variant=adapter['variant'], benchmark='replica48', scene=scene,
        metrics=metrics, per_class=classes, metric_unit='FRACTION',
        matches=dict(path=str(folder / (adapter['method'] + '_matches.json.gz')), identity=record['identity']),
        prediction=manifest['identity'], prediction_identity=manifest['prediction_identity'], GT=gt['identity'],
        original_GT_instances=sum(len(v) for v in gt2pred.values()), point_count=len(xyz),
        full_GT_denominator=True, author_agreement=agreement, synthetic=synthetic,
        formal_G3_verified=not synthetic, protocol=protocol['identity']))

def pool(binding, adapter):
    from .standard_score import ap_metrics
    protocol = verified(Path(binding['output_root']) / 'protocol_lock.json')
    scope = protocol['scopes']['replica48']
    records, matches, missing = {}, {}, []
    for scene in scope['scenes']:
        path = _folder(binding, adapter) / 'evaluation/replica48' / scene / 'receipt.json'
        if not path.is_file():
            missing.append(scene)
            continue
        score = verified(path)
        if score.get('synthetic') or not score.get('formal_G3_verified') or score['status'] != 'DONE':
            missing.append(scene)
            continue
        current = score_prediction(binding, scene, _folder(binding, adapter) / 'predictions/replica48' / scene / 'manifest.json', adapter)
        if current['identity'] != score['identity']:
            raise ValueError('Completed full-method scene score changed before pooling')
        evidence = verified(score['matches']['path'])
        if evidence['identity'] != score['matches']['identity']:
            raise ValueError('Full-method match sufficient statistics changed')
        matches[scene] = evidence['matches'][scene]
        records[scene] = dict(score=score['identity'], prediction=score['prediction'], matches=evidence['identity'])
    metrics, classes = None, None
    if matches:
        metrics, classes, _ = ap_metrics(_evaluator(binding), matches)
    return write(_folder(binding, adapter) / 'pools/replica48.json', dict(
        status='BLOCKED_DEPENDENCY' if missing else 'DONE', method=adapter['method'], benchmark='replica48',
        variant=adapter['variant'], locally_executed=bool(records),
        metrics=metrics, per_class=classes, full_scene_coverage=not missing,
        completed_scenes=len(records), expected_scenes=len(scope['scenes']), ordered_scenes=list(records),
        missing_scenes=missing, source_scenes=records, protocol=protocol['identity'],
        scope='PARTIAL_SUBSET' if missing else 'FULL_STANDARD', formal_G3_verified=bool(records),
        aggregation='OFFICIAL_EVALUATE_MATCHES_ALL_ORDERED_SCENES;NO_SCENE_AP_AVERAGING',
        confidence='UNCHANGED_AUTHOR_VALUES', new_training_updates=0,
        producer=source_identity((adapter['producer'], 'replica_baseline_score'))))
