"""Independent legacy/standard evaluation adapters; no inherited job-count gates."""
import contextlib
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import time

from .common import (verified, read, write, canonical_digest, source_identity,
                     StageFailure, Resolver)


def frozen_legacy_inputs(root):
    """Validate every frozen scene before any GT loader or scorer is imported."""
    root = Path(root)
    path = root / 'legacy/prediction_lock.json'
    if not path.is_file():
        raise StageFailure('All available legacy predictions must be frozen before GT scoring',
                           missing=[str(path)])
    lock = verified(path)
    scenes = [s for names in lock['cohorts'].values() for s in names]
    if len(scenes) != len(set(scenes)) or set(scenes) != set(lock['scenes']):
        raise ValueError('Legacy prediction lock omits or duplicates a frozen scene')
    if not lock['all_available_predictions_frozen_before_new_GT_scoring']:
        raise ValueError('Legacy predictions were not frozen before scoring')
    methods = lock['methods']
    if len(methods) != len(set(methods)):
        raise ValueError('Duplicate legacy methods')
    receipts = {}
    for cohort, names in lock['cohorts'].items():
        for scene in names:
            row = verified(root / 'legacy/predictions' / scene / 'receipt.json')
            if (row['identity'] != lock['scenes'][scene] or row['scene'] != scene
                    or row['cohort'] != cohort or row['status'] != 'DONE'
                    or not set(row['methods']) <= set(methods)):
                raise ValueError('A frozen scene changed before scoring: ' + scene)
            receipts[scene] = row
    if (lock['expected_logical_scene_rows'] != len(scenes) * len(methods)
            or lock['completed_logical_scene_rows'] != sum(len(r['methods']) for r in receipts.values())):
        raise ValueError('Legacy logical coverage differs from the actual frozen matrix')
    return lock, receipts


def _legacy_scene(job):
    # The parent sets these before creating workers; also retain them for spawn.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    from .legacy import numerical_binding
    from static_ovmap.frozen_fc_routing.binding import load_scene
    from static_ovmap.frozen_fc_routing.evaluation import partition_evaluator
    from static_ovmap.module_validation.scannet_study import load_prediction
    from static_ovmap.composition_study.object_evidence import owner_labels
    from static_ovmap.cvpr_compact.evaluation import fraction_metrics, trace_class_metrics
    from static_ovmap.module_validation.native_capture import _array_digest

    binding, scene, global_identity, scene_identity = job
    root = Path(binding['output_root'])
    lock = verified(root / 'legacy/predictions' / scene / 'receipt.json')
    if lock['identity'] != scene_identity:
        raise ValueError('A frozen scene changed after scoring dispatch')
    identity = canonical_digest(dict(global_predictions=global_identity, scene=scene_identity,
                                     producer=source_identity(('scoring',))))
    folder = root / 'legacy/evaluation' / scene
    receipt = folder / 'receipt.json'
    if receipt.exists():
        previous = verified(receipt)
        if previous['input_identity'] != identity:
            raise ValueError('Completed legacy score changed; preserve before an affected repair')
        return previous
    old = numerical_binding(binding)
    p, inputs = load_scene(old, scene)
    rows, scored = [], {}
    evaluator = None
    for method, item in lock['methods'].items():
        payload = load_prediction(item['manifest'])
        if (payload.prediction_key != item['prediction_key']
                or not np.array_equal(payload.owner_ids, inputs.g1.owner_ids)):
            raise ValueError('Frozen legacy mask/label/rank inputs changed')
        if evaluator is None:
            evaluator = partition_evaluator(inputs, payload, old, folder)
        if payload.prediction_key in scored:
            row = dict(scored[payload.prediction_key], reuse_kind='EXACT_MASK_LABEL_RANK_SCORER_ALIAS')
        else:
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / (method + '.log')).open('w') as stream, contextlib.redirect_stdout(stream):
                row = evaluator.evaluate(owner_labels(payload), method, 'OFFICIAL_CURRENT_CLASS', payload.prediction_key)
        score = read(row['evaluation_receipt'])
        confusion = np.asarray(score['confusion'], np.int64)
        valid_gt = int(np.isin(evaluator.targets['gt_semantic'], evaluator.ids).sum())
        if int(confusion.sum()) != valid_gt:
            raise ValueError('Legacy scoring omitted valid GT from its denominator')
        result = write(folder / 'rows' / (method + '.json'), dict(row, scene=scene, method=method,
            status='DONE', cohort=lock['cohort'], metrics=fraction_metrics(score['metrics']),
            per_class=trace_class_metrics(score, index=inputs.index),
            owner_partition_digest=_array_digest(payload.owner_ids),
            record_identity=payload.record_key, prediction_identity=payload.prediction_key,
            prediction_lock=lock['identity'], global_prediction_lock=global_identity,
            metric_unit='FRACTION', full_valid_GT_points=valid_gt,
            scope='EXPOSED_FIXED_ELIGIBLE_UPDATE_NOT_GLOBAL_SOURCE_ABLATION'))
        rows.append(result)
        scored[payload.prediction_key] = result
    inputs.index.write_memo(root / 'legacy/scoring_inputs' / scene / 'verifications.json')
    result = write(receipt, dict(status='DONE', input_identity=identity, scene=scene, rows=rows,
        row_count=len(rows), unique_scorer_inputs=len(scored), GT_access_after_all_prediction_freeze=True))
    print('LEGACY_SCORED', scene, len(rows), 'unique', len(scored), flush=True)
    return result


def legacy_score(binding):
    root = Path(binding['output_root'])
    lock, scenes = frozen_legacy_inputs(root)
    identity = canonical_digest(dict(predictions=lock['identity'], producer=source_identity(('scoring',))))
    path = root / 'legacy/score.json'
    if path.exists():
        previous = verified(path)
        if previous['input_identity'] != identity:
            raise ValueError('Completed legacy pool changed; preserve before an affected repair')
        return previous
    start = time.monotonic()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    jobs = [(binding, scene, lock['identity'], row['identity']) for scene, row in scenes.items()]
    with ProcessPoolExecutor(max_workers=3) as workers:
        receipts = list(workers.map(_legacy_scene, jobs))
    from static_ovmap.structure_confirmation_v1.common import ConsumptionIndex
    from static_ovmap.released_loader import load_released_module
    from static_ovmap.preservation_routing.scorer_compat import adapt
    from static_ovmap.cvpr_compact.evaluation import released_pool_with_classes, fraction_metrics

    rows = [row for receipt in receipts for row in receipt['rows']]
    actual = {(row['scene'], row['method']) for row in rows}
    expected_available = {(scene, method) for scene, row in scenes.items() for method in row['methods']}
    if actual != expected_available or len(actual) != len(rows):
        raise ValueError('Legacy score coverage differs from frozen available predictions')
    index = ConsumptionIndex(root / 'legacy/pool_verifications.json')
    resolver = Resolver(binding['path_map'])
    pools, missing = {}, []
    for cohort, names in lock['cohorts'].items():
        first = next(row for row in rows if row['scene'] == names[0])
        reference = read(first['evaluation_receipt'])
        descriptor = resolver.rewrite(reference['context']['evaluator'])
        index.identity(descriptor['path'], descriptor)
        namespace = load_released_module(descriptor['path'])
        namespace['init']('Replica' if cohort == 'replica8' else 'Scannet200')
        adapt(namespace)
        pools[cohort] = {}
        for method in lock['methods']:
            absent = [scene for scene in names if (scene, method) not in actual]
            if absent:
                missing.append(dict(cohort=cohort, method=method, missing_scenes=absent,
                    reason='UNRESOLVED_REPLICA_TRAINING_BASE_SEMANTIC_MAPPING'))
                continue
            ordered = [next(row for row in rows if row['scene'] == scene and row['method'] == method)
                       for scene in names]
            folder = root / 'legacy/pools' / cohort
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / (method + '.log')).open('w') as stream, contextlib.redirect_stdout(stream):
                pooled, classes = released_pool_with_classes(folder, ordered, namespace, names,
                    method, 'OFFICIAL_CURRENT_CLASS', index)
            pools[cohort][method] = write(folder / (method + '.json'), dict(pooled,
                status='DONE', cohort=cohort, metric_unit='FRACTION', metrics=fraction_metrics(pooled['metrics']),
                per_class_identity=classes['identity'], per_class_receipt=str(folder / (method + '_per_class.json')),
                ordered_scenes=names, scope='EXPOSED_FIXED_ELIGIBLE_UPDATE_NOT_GLOBAL_SOURCE_ABLATION'))
    index.write_memo(root / 'legacy/pool_verifications.json')
    return write(path, dict(status='DONE' if not missing else 'BLOCKED_DEPENDENCY', input_identity=identity,
        rows=rows, pools=pools, expected_logical_scene_rows=lock['expected_logical_scene_rows'],
        completed_logical_scene_rows=len(rows), expected_pools=len(lock['cohorts']) * len(lock['methods']),
        completed_pools=sum(map(len, pools.values())), missing_pools=missing,
        CPU_workers=3, BLAS_threads=1, actual_new_training_updates=0,
        elapsed_seconds=time.monotonic() - start, full_GT_denominator=True,
        scope='EXPOSED_FIXED_ELIGIBLE_UPDATE_NOT_GLOBAL_SOURCE_ABLATION'))
