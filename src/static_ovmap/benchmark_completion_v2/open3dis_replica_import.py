"""Read-only import of a genuinely completed full eight-scene Replica port."""
import math
from pathlib import Path

from .common import canonical_digest, digest_file, source_identity, verified

VARIANT = 'Open3DIS (Replica port, ISBNet + Grounded SAM-HQ + refinement, author SPP branch)'
AGGREGATION = 'OFFICIAL_EVALUATE_MATCHES_ALL_ORDERED_SCENES;NO_SCENE_AP_AVERAGING'
STAGES = {'grounding_2d', 'lift3d', 'refine', 'final'}


def _require(condition, message):
    if not condition:
        raise ValueError('Open3DIS retained Replica pool: ' + message)


def pool(binding):
    root = Path(binding['output_root'])
    folder = root / 'external/Open3DIS'
    path = folder / 'pools/replica48.json'
    if not path.is_file():
        return None
    row = verified(path)
    if row.get('status') != 'DONE' or not row.get('full_scene_coverage'):
        return None
    certificate_path = path.with_name('replica48_completion.json')
    if not certificate_path.is_file():
        return None
    certificate = verified(certificate_path)
    completion, launch, actual, sync = (certificate[key]
        for key in ('completion', 'launch', 'actual', 'synchronized'))
    for record in (completion, launch, actual, sync):
        _require(record['identity'] == canonical_digest({key: value for key, value in record.items()
                 if key != 'identity'}), 'original completion record identity differs')
    _require(certificate['status'] == completion['status'] == sync['status'] == 'DONE'
             and certificate['method'] == completion['method'] == 'Open3DIS'
             and certificate['benchmark'] == completion['benchmark'] == 'replica48'
             and certificate['pool'] == completion['pool'] == row['identity']
             and completion['completed_scenes'] == completion['expected_scenes'] == 8
             and completion['metrics'] == row['metrics'] and completion['aggregation'] == AGGREGATION
             and completion['unchanged_original_aggregator'] is True and completion['new_GPU_calls'] == 0
             and type(actual['actual_child_exit']) is int and actual['actual_child_exit'] == 0
             and actual['launch'] == launch['identity'] and actual['command'] == launch['command']
             and sync['actual_exit'] == actual['identity'] and sync['source_unchanged'] is True
             and sync['source_after_sha256'] == actual['source_sha256'] == launch['source_sha256'],
             'successful original whole-pool execution required')
    protocol = verified(root / 'protocol_lock.json')
    scope = protocol['scopes']['replica48']
    from .semantic_refresh import _ancestor, _same_scope
    ancestor = _ancestor(protocol, row['protocol'])
    _require(_same_scope(ancestor['scopes']['replica48'], scope, calibrated=False)
             and ancestor.get('official_sources', {}) == protocol.get('official_sources', {}),
             'retained protocol numerical scope or official source differs')
    scenes = scope['scenes']
    _require(len(scenes) == len(set(scenes)) == 8 and len(scope['instance_names']) == 48,
             'official eight-scene, 48-class scope required')
    _require(row['method'] == 'Open3DIS' and row['benchmark'] == 'replica48'
             and row['variant'] == VARIANT
             and row['ordered_scenes'] == scenes and set(row['source_scenes']) == set(scenes)
             and row['completed_scenes'] == row['expected_scenes'] == 8 and not row['missing_scenes']
             and row['formal_G3_verified'] is True and row['locally_executed'] is True
             and row['scope'] == 'FULL_STANDARD' and row['new_training_updates'] == 0
             and row['confidence'] == 'UNCHANGED_AUTHOR_VALUES' and row['aggregation'] == AGGREGATION,
             'full method, protocol or coverage differs')
    _require(row['producer'] == source_identity(('open3dis_replica_score', 'replica_baseline_score')),
             'original full-method scoring source changed')
    _require(all(type(row['metrics'].get(key)) in (int, float)
                 and math.isfinite(row['metrics'][key]) and 0 <= row['metrics'][key] <= 1
                 for key in ('apall', 'ap50', 'ap25')), 'invalid pooled AP values')
    for scene in scenes:
        evaluation = folder / 'evaluation/replica48' / scene
        score = verified(evaluation / 'receipt.json')
        raw = verified(evaluation / 'G3_published_raw_GT_agreement.json')
        prediction = verified(folder / 'predictions/replica48' / scene / 'manifest.json')
        refs = row['source_scenes'][scene]
        _require(score['status'] == prediction['status'] == raw['status'] == 'DONE'
                 and score['method'] == prediction['method'] == raw['method'] == 'Open3DIS'
                 and score['benchmark'] == prediction['benchmark'] == raw['benchmark'] == 'replica48'
                 and score['scene'] == prediction['scene'] == raw['scene'] == scene
                 and score['protocol'] == row['protocol'] and score['variant'] == VARIANT
                 and score['identity'] == refs['score'] == raw['common_score']
                 and score['prediction'] == refs['prediction'] == prediction['identity'] == raw['prediction']
                 and score['prediction_identity'] == prediction['prediction_identity']
                 and score['formal_G3_verified'] is raw['formal_G3_verified'] is True
                 and score['synthetic'] is raw['synthetic'] is False
                 and score['full_GT_denominator'] is raw['full_original_GT_denominator'] is True
                 and type(prediction['target_points']) is int and prediction['target_points'] > 0
                 and score['point_count'] == raw['target_points'] == prediction['target_points']
                 and score['metric_unit'] == 'FRACTION', 'scene provenance or full point domain differs: ' + scene)
        _require(raw['class_IoU_entries'] == 480 and raw['AP_array_max_absolute_error'] == 0
                 and raw['all_three_original_AP_arrays_equal'] is True,
                 'full raw-author AP agreement required: ' + scene)
        provenance = prediction['provenance']
        stages = provenance['actual_four_stages']
        _require(provenance['method_variant'] == VARIANT
                 and provenance['full_original_inference_stages'] is True and set(stages) == STAGES
                 and all(all(stage.get(key) for key in ('actual_exit', 'result', 'synchronized'))
                         for stage in stages.values())
                 and provenance['GT_labels_used_for_prediction'] is False
                 and provenance['synthetic'] is False and provenance['new_training_updates'] == 0
                 and provenance['evaluation_confidence'] == 'AUTHOR_inst_run_replica.py_CONSTANT_1'
                 and provenance['archive_confidence_field_used'] is False,
                 'complete original prediction pipeline required: ' + scene)
        arrays = prediction['arrays']
        array_path = Path(arrays['path'])
        _require(not array_path.is_symlink() and array_path.stat().st_size == arrays['bytes']
                 and digest_file(array_path) == arrays['sha256'], 'frozen prediction bytes changed: ' + scene)
        matches = verified(score['matches']['path'])
        _require(matches['identity'] == refs['matches'] == score['matches']['identity']
                 and matches['status'] == 'DONE' and set(matches['matches']) == {scene},
                 'original pooled match statistics changed: ' + scene)
    return row
