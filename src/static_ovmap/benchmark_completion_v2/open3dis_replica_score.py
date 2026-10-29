"""Full Replica port completion checks and the existing common mask scorer."""
from pathlib import Path

from .common import digest_file, verified

STAGES = ('grounding_2d', 'lift3d', 'refine', 'final')


def _require(condition, message):
    if not condition:
        raise ValueError('Open3DIS full Replica export: ' + message)


def validate_completion(recipe_path, actual_directories, *, scene):
    recipe_path = Path(recipe_path).resolve()
    recipe = verified(recipe_path)
    _require(recipe['scene'] == scene, 'recipe scene changed')
    _require(set(recipe['stages']) == set(actual_directories) == set(STAGES),
             'all four full-method stages are required')
    for path, sha in recipe['sources'].items():
        _require(digest_file(path) == sha, 'frozen source or input changed: ' + path)
    for filename in recipe['source_manifests']:
        for path, sha in verified(filename)['files'].items():
            _require(digest_file(path) == sha, 'author source changed: ' + path)
    completed = {}
    for stage in STAGES:
        details = recipe['stages'][stage]
        proof = verified(details['result'])
        _require(proof['status'] == 'DONE' and proof['stage'] == stage
                 and proof['scene'] == scene and proof['recipe'] == recipe['identity']
                 and proof['point_count'] == recipe['point_count'], 'stage scope changed: ' + stage)
        _require(proof['GT_tensor_loads'] == proof['new_training_updates'] == 0
                 and not proof['failed_tensor_loads']
                 and proof['unchanged_authored_prediction_and_aggregation_functions'],
                 'stage used GT, training or a failed tensor fallback: ' + stage)
        _require(proof['final_prediction_frozen'] == (stage == 'final'),
                 'final prediction freeze missing or premature')
        _require(set(proof['outputs']) == set(details['outputs']), 'stage output set changed')
        for path, descriptor in proof['outputs'].items():
            _require(digest_file(path) == descriptor['sha256']
                     and Path(path).stat().st_size == descriptor['bytes'],
                     'original output changed: ' + path)
        folder = Path(actual_directories[stage])
        launch = verified(folder/'launch.json')
        actual = verified(folder/'actual_exit.json')
        sync = verified(folder/'synchronized.json')
        _require(actual['actual_child_exit'] == 0 and actual['launch'] == launch['identity'],
                 'actual successful stage exit required: ' + stage)
        command = actual['command']
        _require(command == launch['command'] and '--preflight' not in command,
                 'actual command differs or only ran preflight')
        _require('--recipe' in command and '--stage' in command, 'missing actual stage command')
        recipe_index, stage_index = command.index('--recipe'), command.index('--stage')
        _require(recipe_index > 0 and recipe_index+1 < len(command)
                 and stage_index+1 < len(command), 'truncated stage command')
        _require(Path(command[recipe_index+1]).resolve() == recipe_path
                 and command[stage_index+1] == stage, 'actual command used a different recipe or stage')
        worker = str(Path(command[recipe_index-1]).resolve())
        _require(worker in recipe['sources'], 'actual worker is outside frozen recipe')
        sha = recipe['sources'][worker]
        _require(actual['source_sha256'] == launch['source_sha256'] == sha
                 and sync['status'] == 'DONE' and sync['actual_exit'] == actual['identity']
                 and sync['source_unchanged'] and sync['source_after_sha256'] == sha,
                 'actual stage source or synchronization is incomplete')
        completed[stage] = dict(result=proof['identity'], actual_exit=actual['identity'],
                                synchronized=sync['identity'])
    return recipe, completed


def freeze(binding, scene, connection_path):
    import numpy as np
    import yaml
    from .external import decode_open3dis, save_prediction
    from .replica_baseline_score import _xyz

    connection = verified(connection_path)
    _require(connection['scene'] == scene, 'completion connection scene changed')
    _require(all(digest_file(path) == sha for path, sha in connection['sources'].items()),
             'completion connection source changed')
    recipe, completed = validate_completion(connection['recipe_path'],
        connection['actual_directories'], scene=scene)
    _require(recipe['identity'] == connection['recipe'], 'completion connection recipe changed')
    _require(recipe['configuration_frozen_before_target_AP'] and recipe['new_training_updates'] == 0,
             'configuration must precede target AP without new training')
    root = Path(binding['output_root'])
    scope = verified(root/'protocol_lock.json')['scopes']['replica48']
    _require(scene in scope['scenes'] and list(scope['instance_names']) == recipe['class_names'],
             'author class order differs from the frozen Replica protocol')
    cfg = yaml.safe_load(Path(recipe['config']).read_text())
    layout_path = Path(cfg['data']['datapath'])/scene/'reader_layout_v1.json'
    layout = verified(layout_path)
    _require(layout['identity'] == recipe['accepted_RGBD_layout'], 'RGBD geometry identity changed')
    geometry = layout['geometry']
    _require(digest_file(geometry['path']) == geometry['sha256'], 'original geometry changed')
    _require(len(_xyz(geometry['path'])) == recipe['point_count'], 'original full point domain changed')
    final = recipe['stages']['final']['outputs']
    _require(len(final) == 1, 'one original final prediction archive is required')
    prediction = decode_open3dis(final[0], target_points=recipe['point_count'],
                                class_ids=scope['instance_ids'])
    _require(np.equal(prediction['pred_scores'], 1).all(), 'author evaluation confidence changed')
    native = dict(path=final[0],sha256=digest_file(final[0]),bytes=Path(final[0]).stat().st_size)
    return save_prediction(root/'external/Open3DIS/predictions/replica48'/scene, prediction,
        method='Open3DIS',benchmark='replica48',scene=scene,provenance=dict(
            recipe=dict(path=str(connection['recipe_path']),identity=recipe['identity']),
            completion_connection=dict(path=str(connection_path),identity=connection['identity']),
            method_variant=recipe['method_variant'], geometry=geometry, native_prediction=native,
            actual_four_stages=completed, full_original_inference_stages=True,
            GT_labels_used_for_prediction=False,new_training_updates=0,synthetic=False,
            evaluation_confidence='AUTHOR_inst_run_replica.py_CONSTANT_1',
            archive_confidence_field_used=False,original_archive_confidence_bytes_retained=True,
            score_only_GT_source='DERIVED_FROM_PUBLISHED_AUTHOR_GT_TEXT_NOT_ORIGINAL_AUTHOR_TUPLE_BYTES'))


def score_prediction(binding, scene, connection_path):
    from .replica_baseline_score import score_prediction as score
    connection = verified(connection_path)
    recipe = verified(connection['recipe_path'])
    adapter = dict(method='Open3DIS',variant=recipe['method_variant'],
        producer='open3dis_replica_score',
        freeze=lambda active, current: freeze(active,current,connection_path))
    manifest = freeze(binding,scene,connection_path)
    path = Path(binding['output_root'])/'external/Open3DIS/predictions/replica48'/scene/'manifest.json'
    _require(verified(path)['identity'] == manifest['identity'], 'prediction freeze changed before scoring')
    return score(binding,scene,path,adapter)
