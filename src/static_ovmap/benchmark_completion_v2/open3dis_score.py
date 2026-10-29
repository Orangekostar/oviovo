"""Import a completed common-protocol author-prediction score, without scoring.

The configured job must have full frozen coverage and a successful real exit.
Historical source bytes are retained independently of later source upgrades.
"""
import hashlib
import math
from pathlib import Path
import shutil

from . import common
from .common import StageFailure, canonical_digest, digest_file, verified, write

BENCHMARK = 'scannet200_val'
METHOD = 'Open3DIS'
VARIANT = 'ISBNET_GSAM_REFINED_SECOND_3D_PASS'
AUTHOR_PIN = 'bc2a3b3ee7161f5754ad7a7fe77202e9a66adc8e'
CONFIDENCE = 'AUTHOR_EVAL_CONSTANT_1'
AGGREGATION = 'OFFICIAL_EVALUATE_MATCHES_ALL_ORDERED_SCENES;NO_SCENE_AP_AVERAGING'


def _require(condition, message):
    if not condition:
        raise ValueError('Open3DIS score import: ' + message)


def _descriptor(row, *, path=None):
    path = Path(row['path'] if path is None else path)
    _require(path.is_file() and path.stat().st_size == row['bytes']
             and digest_file(path) == row['sha256'], 'file differs from frozen bytes: ' + str(path))


def _metrics(row, names):
    _require(set(row['metrics']) == {'apall', 'ap50', 'ap25'}, 'incomplete AP metrics')
    _require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                 for v in row['metrics'].values()), 'AP metrics are not finite fractions')
    _require(set(row['per_class']) == set(names), 'per-class AP axis differs')
    for values in row['per_class'].values():
        _require(set(values) == {'ap', 'ap50%', 'ap25%'}, 'per-class IoU metrics differ')
        _require(all(v is None or type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                     for v in values.values()), 'invalid per-class AP')


def _sources(scores, config):
    snapshot = verified(scores / 'consumed_source_snapshot.json')
    _require(snapshot['status'] == 'DONE' and snapshot['configuration'] == config['identity']
             and [r['original'] for r in snapshot['files']] == config['sources'],
             'source snapshot does not cover the consumed configuration')
    originals = [r['path'] for r in config['sources']]
    _require(len(originals) == len(set(originals)), 'duplicate consumed source')
    by_original, export = {}, []
    for row in snapshot['files']:
        old, archived = row['original'], row['archived']
        _require(all(old[k] == archived[k] for k in ('bytes', 'sha256')), 'archive identity differs')
        _descriptor(archived)
        raw = Path(archived['path']).read_bytes()
        text = raw.decode('utf-8')
        _require(hashlib.sha256(text.encode('utf-8')).hexdigest() == old['sha256'], 'source encoding changed')
        by_original[old['path']] = archived['path']
        export.append(dict(original=old, utf8=text))
    return snapshot, by_original, export


def _named_source(config, name):
    rows = [r for r in config['sources'] if Path(r['path']).name == name]
    _require(len(rows) == 1, 'ambiguous or absent consumed source: ' + name)
    return rows[0]


def _exit(scores, config):
    folder = scores / 'roles/open3dis_queue'
    launch, actual, sync = [verified(folder / name) for name in
                            ('launch.json', 'actual_exit.json', 'synchronized.json')]
    runner = _named_source(config, 'open3dis_scoring.py')
    _require(actual['status'] == 'EXITED' and type(actual['actual_child_exit']) is int
             and actual['actual_child_exit'] == 0 and actual['local_first_actual_exit'] is True,
             'full scoring process did not actually exit successfully')
    _require(actual['launch'] == launch['identity'] and actual['child'] == launch['child']
             and actual['command'] == launch['command']
             and actual['source_sha256'] == launch['source_sha256'] == runner['sha256']
             and actual['supervisor_sha256'] == launch['supervisor_sha256'],
             'actual exit does not belong to the frozen full-scoring launch')
    _require(launch['command'][-2:] == [runner['path'], 'queue'], 'not the full scoring entry point')
    _require(launch['env']['CUDA_VISIBLE_DEVICES'] == ''
             and all(launch['env'][k] == '1' for k in
                     ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
             'full scoring launch did not use the reviewed CPU environment')
    _require(sync['status'] == 'DONE' and sync['actual_exit'] == actual['identity']
             and sync['source_unchanged'] is True and not sync['errors']
             and sync['source_after_sha256'] == runner['sha256'], 'exit evidence was not fully synchronized')
    _descriptor(actual['log'], path=folder / 'runtime.log')
    return dict(launch=launch, actual_exit=actual, synchronized=sync)


def _protocol(root, scores, config, archived):
    prior = verified(scores / 'protocol_lock.json')
    source = _named_source(config, 'protocol_lock.json')
    _require(verified(archived[source['path']]) == prior and config['protocol'] == prior['identity'],
             'scoring protocol differs from consumed source snapshot')
    current = verified(root / 'protocol_lock.json')
    scope = prior['scopes'][BENCHMARK]
    _require(current['scopes'][BENCHMARK] == scope, 'ScanNet scope changed; scoring must be reviewed again')
    old_eval, new_eval = [p['official_sources']['evaluator_scan'] for p in (prior, current)]
    _require(all(old_eval[k] == new_eval[k] for k in ('bytes', 'sha256')), 'official evaluator changed')
    _require(old_eval['path'] in archived, 'official evaluator absent from consumed source snapshot')
    _descriptor(new_eval)
    _require(len(scope['scenes']) == len(set(scope['scenes'])) == 312
             and scope['expected_scenes'] == 312 and scope['instance_classes'] == 198,
             'not the full ScanNet200 instance protocol')
    carry = dict(scoring_protocol=prior['identity'], current_protocol=current['identity'],
        global_identity_changed=prior['identity'] != current['identity'],
        exact_scannet_scope_equal=True, scannet_scope=canonical_digest(scope),
        official_evaluator_sha256=old_eval['sha256'])
    return current, prior, scope, carry


def _validate(root, scores):
    config = verified(scores / 'configuration.json')
    freeze = verified(scores / 'predictions.json')
    pool = verified(scores / 'pool.json')
    _require(config['status'] == 'REVIEWED_FOR_ONE_LAUNCH'
             and config['variant'] == VARIANT and config['author_pin'] == AUTHOR_PIN
             and config['confidence'] == CONFIDENCE and config['one_CPU_scorer'] is True
             and config['production_sources_modified'] is False
             and config['new_GPU_calls'] == config['new_training_updates'] == 0,
             'unreviewed method variant or scoring configuration')
    snapshot, archived, source_text = _sources(scores, config)
    exits = _exit(scores, config)
    current, prior, scope, carry = _protocol(root, scores, config, archived)
    scenes = scope['scenes']
    _require(config['expected_scenes'] == freeze['scenes'] == scenes
             and freeze['identity'] == config['prediction_freeze']
             and freeze['status'] == 'DONE' and freeze['benchmark'] == BENCHMARK
             and freeze['protocol'] == prior['identity'] and set(freeze['predictions']) == set(scenes)
             and freeze['all_author_predictions_frozen_before_this_job_GT_access'] is True,
             'prediction freeze differs from full scope/configuration')
    _require(pool['status'] == 'DONE' and pool['method'] == METHOD and pool['benchmark'] == BENCHMARK
             and pool['configuration'] == config['identity'] and pool['protocol'] == prior['identity']
             and pool['variant'] == VARIANT and pool['official_pin'] == AUTHOR_PIN
             and pool['confidence'] == CONFIDENCE and pool['synthetic'] is False
             and pool['full_scene_coverage'] is True and pool['full_GT_denominator'] is True
             and pool['completed_scenes'] == pool['expected_scenes'] == 312
             and pool['ordered_scenes'] == scenes and set(pool['source_scenes']) == set(scenes)
             and pool['metric_unit'] == 'FRACTION' and pool['aggregation'] == AGGREGATION
             and pool['locally_scored_author_predictions'] is True and pool['locally_inferred'] is False
             and pool['new_GPU_calls'] == pool['new_training_updates'] == 0,
             'pooled score is incomplete or belongs to another method/protocol')
    _metrics(pool, scope['instance_names'])
    g3_source = _named_source(config, 'receipt.json')
    g3 = verified(archived[g3_source['path']])
    _require(g3['identity'] == config['G3'] == pool['G3_real_first_scene']
             and g3['status'] == 'DONE' and g3['synthetic'] is False
             and g3['all_class_IoU_arrays_equal'] is True and g3['full_GT_denominator'] is True
             and g3['original_protocol'] == prior['identity'] and g3['benchmark'] == BENCHMARK
             and g3['scene'] == scenes[0] and g3['archive_confidence_field_used'] is False
             and 0 <= g3['tolerance'] <= 1e-12, 'no matching real first-scene author/common agreement')
    rows = {}
    for scene in scenes:
        row = verified(scores / 'scenes' / scene / 'receipt.json')
        refs = pool['source_scenes'][scene]
        gt = verified(root / 'benchmark/GT' / BENCHMARK / scene / 'receipt.json')
        model = verified(root / 'benchmark/predictions' / BENCHMARK / scene / 'receipt.json')
        _require(gt['status'] == 'DONE' and gt['scene'] == scene and gt['benchmark'] == BENCHMARK
                 and gt['full_GT_denominator'] is True and gt['GT_filtered_by_prediction'] is False
                 and gt['prediction_freeze'] == model['identity'] and model['status'] == 'DONE'
                 and model['all_available_predictions_frozen_before_new_GT_scoring'] is True,
                 'full original GT freeze differs: ' + scene)
        expected_gt = {k: gt['GT'][k] for k in ('path', 'bytes', 'sha256')}
        _require(row['status'] == 'DONE' and row['scene'] == scene and row['method'] == METHOD
                 and row['benchmark'] == BENCHMARK and row['configuration'] == config['identity']
                 and row['prediction'] == freeze['predictions'][scene]
                 and row['prediction_freeze'] == freeze['identity'] and row['GT_receipt'] == gt['identity']
                 and row['GT'] == expected_gt and row['points'] == gt['target_points']
                 and row['identity'] == refs['score'] and row['matches_identity'] == refs['matches']
                 and row['confidence'] == CONFIDENCE and row['archive_confidence_field_used'] is False
                 and row['full_GT_denominator'] is True and row['synthetic'] is False
                 and row['metric_unit'] == 'FRACTION', 'scene evidence differs: ' + scene)
        identity = canonical_digest(dict(configuration=config['identity'], scene=scene,
            prediction=row['prediction'], GT=gt['identity'], confidence=CONFIDENCE))
        _require(identity == row['input_identity'], 'scene input identity differs: ' + scene)
        _metrics(row, scope['instance_names'])
        match_path = scores / 'scenes' / scene / 'matches.json.gz'
        _require(Path(row['matches_file']['path']) == match_path, 'unexpected match file location: ' + scene)
        _descriptor(row['matches_file'])
        match = verified(match_path)
        _require(match['identity'] == refs['matches'] and match['status'] == 'DONE'
                 and set(match['matches']) == {scene}, 'match contents belong to another scene')
        if scene == scenes[0]:
            _require(row['points'] == g3['points'] and row['masks'] == g3['masks']
                     and all(abs(row['metrics'][k] - g3['common_metrics'][k]) <= 1e-12
                             and abs(g3['common_metrics'][k] - g3['author_metrics'][k]) <= 1e-12
                             for k in ('apall', 'ap50', 'ap25')),
                     'first-scene scores differ from real author agreement')
        rows[scene] = row
    return config, freeze, pool, snapshot, source_text, exits, current, prior, carry, g3, rows


def pool(binding):
    """Return None if unconfigured; refuse pending/invalid declared scores."""
    assets = binding.get('asset_config', {}).get('external', {}).get(METHOD, {})
    source = assets.get('common_scoring_root')
    if source is None:
        return None
    root, scores = Path(binding['output_root']), Path(source)
    required = ['configuration.json', 'predictions.json', 'pool.json', 'protocol_lock.json',
                'consumed_source_snapshot.json', 'roles/open3dis_queue/launch.json',
                'roles/open3dis_queue/actual_exit.json', 'roles/open3dis_queue/synchronized.json',
                'roles/open3dis_queue/runtime.log']
    missing = [str(scores / name) for name in required if not (scores / name).is_file()]
    if missing:
        raise StageFailure('Declared Open3DIS common scoring job has not completed with synchronized real exit',
                           missing=missing)
    (config, freeze, original_pool, snapshot, source_text, exits, current, prior,
     carry, g3, rows) = _validate(root, scores)
    folder = root / 'external' / METHOD
    import_identity = canonical_digest(dict(configuration=config['identity'], pool=original_pool['identity'],
        snapshot=snapshot['identity'], actual_exit=exits['actual_exit']['identity'], protocol=current['identity'],
        producer=canonical_digest(dict(open3dis_score=digest_file(__file__), common=digest_file(common.__file__)))))
    import_path = folder / 'pools/scannet200_val_import.json'
    if import_path.exists():
        _require(verified(import_path)['input_identity'] == import_identity,
                 'existing import has different inputs; explicit refresh/review is required')
    # Validate every existing destination before creating any new formal output.
    destinations = {}
    for scene, row in rows.items():
        destination = folder / 'evaluation' / BENCHMARK / scene / 'Open3DIS_matches.json.gz'
        if destination.exists():
            _descriptor(row['matches_file'], path=destination)
        destinations[scene] = destination
    source_bundle = write(folder / 'pools/scannet200_val_sources.json', dict(status='DONE',
        configuration=config['identity'], consumed_source_snapshot=snapshot,
        reconstruction='UTF8_ENCODE_EACH_SOURCE_THEN_VERIFY_ORIGINAL_SHA256_AND_BYTES', files=source_text))
    formal_rows = {}
    for scene, row in rows.items():
        destination = destinations[scene]
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + '.tmp')
            shutil.copy2(row['matches_file']['path'], temporary)
            _descriptor(row['matches_file'], path=temporary)
            temporary.replace(destination)
        matches_file = dict(row['matches_file'], path=str(destination))
        result = write(destination.with_name('receipt.json'), dict(row,
            source_score=row['identity'], matches_file=matches_file,
            source_matches_file=row['matches_file'], import_input=import_identity,
            protocol=current['identity'], scoring_protocol=prior['identity'],
            locally_scored_author_predictions=True, locally_inferred=False))
        formal_rows[scene] = dict(score=result['identity'], matches=result['matches_identity'])
    receipt = write(import_path, dict(status='DONE', input_identity=import_identity,
        configuration=config, prediction_freeze=freeze, original_pool=original_pool['identity'],
        source_archive=source_bundle['identity'], source_archive_path='external/Open3DIS/pools/scannet200_val_sources.json',
        real_exit_chain=exits, G3=g3, protocol_carry=carry,
        source_scenes=original_pool['source_scenes'], formal_scenes=formal_rows,
        GT_values_read=False, no_inference_or_scoring=True, new_GPU_calls=0, new_training_updates=0))
    return write(folder / 'pools/scannet200_val.json', dict(original_pool,
        protocol=current['identity'], scoring_protocol=prior['identity'], protocol_carry=carry,
        source_pool=original_pool['identity'], source_scenes=formal_rows,
        source_import=receipt['identity'], formal_G3_verified=True,
        source_archive=source_bundle['identity'], original_source_snapshot=snapshot['identity'],
        import_input=import_identity))
