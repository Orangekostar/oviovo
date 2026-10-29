"""Common official mask evaluator and complete-denominator score-only GT IO."""
import ast
import contextlib
import importlib.util
import math
from pathlib import Path
import sys
import types
import uuid
import numpy as np

from .common import (verified, read, write, ConsumedInputs, canonical_digest,
                     source_identity, StageFailure)


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluator(binding, benchmark):
    """Load author source unchanged; configure only its public class/threshold axes."""
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    scope = protocol['scopes'][benchmark]
    path = Path(protocol['official_sources']['evaluator_scan']['path'])
    index = ConsumedInputs(root)
    index.identity(path, protocol['official_sources']['evaluator_scan'])
    aliases, prior = {}, {}
    try:
        for name in ('util', 'util_3d', '_ovmap_common_official_evaluator'):
            prior[name] = sys.modules.get(name)
            source = path if name.startswith('_ovmap') else path.with_name(name + '.py')
            index.identity(source)
            if name in ('util', 'util_3d'):
                # These author modules also import imageio/PLY visualization
                # dependencies that the mask evaluator never calls. Compile
                # only the exact unchanged utility ASTs needed by this entry.
                import json
                required = {'print_error'} if name == 'util' else {'load_ids', 'Instance', 'get_instances'}
                tree = ast.parse(source.read_text())
                nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in required]
                if {n.name for n in nodes} != required:
                    raise ValueError('Official evaluator utility boundary changed')
                module = types.ModuleType(name)
                module.__dict__.update(np=np, sys=sys, json=json)
                exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), module.__dict__)
                sys.modules[name] = module
            else:
                spec = importlib.util.spec_from_file_location(name, source)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            aliases[name] = module
        namespace = vars(aliases['_ovmap_common_official_evaluator'])
    finally:
        for name, value in prior.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    from static_ovmap.preservation_routing.scorer_compat import adapt
    adapt(namespace)
    namespace.update(DATASET_NAME='COMMON_OFFICIAL_AP', CLASS_LABELS=tuple(scope['instance_names']),
        VALID_CLASS_IDS=np.asarray(scope['instance_ids']),
        ID_TO_LABEL=dict(zip(scope['instance_ids'], scope['instance_names'], strict=True)),
        LABEL_TO_ID=dict(zip(scope['instance_names'], scope['instance_ids'], strict=True)))
    if not np.array_equal(namespace['opt']['overlaps'], scope['IoU_thresholds']):
        raise ValueError('Official runtime IoU declarations differ from the protocol lock')
    namespace['opt']['min_region_sizes'] = np.asarray([scope['minimum_region_vertices']])
    index.save()
    return namespace


def ground_truth(binding, benchmark, scene, *, prediction_lock):
    """Never called by prediction, input-selection or model workers."""
    root = Path(binding['output_root'])
    freeze = verified(prediction_lock)
    if (freeze['scene'] != scene or freeze['benchmark'] != benchmark
            or not freeze['all_available_predictions_frozen_before_new_GT_scoring']):
        raise ValueError('A complete available per-scene prediction freeze is required before GT access')
    bundle = verified(root / 'benchmark/bundles' / benchmark / scene / 'bundle.json')
    folder = root / 'benchmark/GT' / benchmark / scene
    receipt_path = folder / 'receipt.json'
    index = ConsumedInputs(root)
    if receipt_path.exists():
        old = verified(receipt_path)
        if old['prediction_freeze'] != freeze['identity']:
            raise ValueError('GT receipt was bound to a different available prediction freeze')
        index.identity(old['GT']['path'], old['GT'])
        return old
    folder.mkdir(parents=True, exist_ok=True)
    if benchmark == 'scannet200_val':
        from .acquisition import scannet_scene
        import pandas as pd
        acquired = scannet_scene(binding, scene, ground_truth=True)
        raw = Path(acquired['files']['.aggregation.json']['path']).parent
        protocol = verified(root / 'protocol_lock.json')
        table = Path(protocol['semantic_alias_source']['path'])
        source = root / 'assets/external/MaskClustering/preprocess/scannet/prepare_gt.py'
        constants = root / 'assets/external/MaskClustering/evaluation/constants.py'
        index.identity(table)
        index.identity(source)
        index.identity(constants)
        from .protocol import _literal
        valid = _literal(constants, 'SCANNET_IDS')
        tree = ast.parse(source.read_text())
        names = {'export_gt', 'point_indices_from_group', 'handle_process'}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {n.name for n in nodes} != names:
            raise ValueError('Official full-GT converter API changed')
        import json
        import os
        namespace = dict(np=np, pd=pd, json=json, os=os, SCANNET_IDS=valid)
        # Constants and exact author conversion functions, including its retained
        # object0 and instance renumbering. No predicted support enters this call.
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {
                    'CLOUD_FILE_PFIX', 'SEGMENTS_FILE_PFIX', 'AGGREGATIONS_FILE_PFIX'} for t in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        namespace['handle_process'](str(raw), str(folder), pd.read_csv(table, sep='\t', header=0))
        gt = folder / (scene + '.txt')
        values = np.loadtxt(gt, dtype=np.int64)
        with np.load(bundle['arrays']['geometry']['path'], allow_pickle=False) as arrays:
            count = len(arrays['target_xyz'])
        if values.shape != (count,):
            raise ValueError('Official GT segment indices differ from the original full mesh vertex order')
        input_records = [index.identity(row['path'], row) for suffix, row in acquired['files'].items()
                         if suffix in ('.aggregation.json', '_vh_clean_2.0.010000.segs.json')]
        inputs = input_records + [index.identity(table), index.identity(source), index.identity(constants)]
        definition = 'UNCHANGED_AUTHOR_MASKCLUSTERING_GT_CONVERTER;OBJECT0_RETAINED;FULL_MESH'
    else:
        config = binding['asset_config'].get('datasets', {}).get('replica48', {})
        folder_config = config.get('gt_root')
        source = Path(folder_config) / (scene + '.pth') if folder_config else root / 'assets/datasets/replica48/replica_3d' / (scene + '.pth')
        if not source.is_file():
            raise StageFailure('Author Replica48 full GT and original coordinate ordering are missing', missing=[str(source)])
        raise StageFailure('Replica48 author tuple requires verified format connection before score', missing=[str(source)])
    result = write(receipt_path, dict(status='DONE', benchmark=benchmark, scene=scene,
        GT=index.identity(gt), inputs=inputs, target_points=int(len(values)),
        prediction_freeze=freeze['identity'], geometry=bundle['evaluation_geometry'],
        full_GT_denominator=True, GT_filtered_by_prediction=False, definition=definition))
    index.save()
    return result


def ap_metrics(namespace, matches):
    ap = namespace['evaluate_matches'](matches)[0]
    averages = namespace['compute_averages'](ap)
    metrics = dict(apall=averages['all_ap'], ap50=averages['all_ap_50%'], ap25=averages['all_ap_25%'])
    return _json(metrics), _json(averages['classes']), ap


def score_scene(binding, benchmark, scene, *, formal_connection=False):
    from .standard_export import official_prediction
    root = Path(binding['output_root'])
    prediction_path = root / 'benchmark/predictions' / benchmark / scene / 'receipt.json'
    lock = verified(prediction_path)
    manifests = {}
    for method, descriptor in lock['methods'].items():
        row = verified(descriptor['path'])
        if row['identity'] != descriptor['identity']:
            raise ValueError('GLOBAL prediction changed before score-only GT access')
        official_prediction(row)  # Validate all available arrays before GT.
        manifests[method] = row
    gt = ground_truth(binding, benchmark, scene, prediction_lock=prediction_path)
    namespace = evaluator(binding, benchmark)
    identity = canonical_digest(dict(predictions=lock['identity'], GT=gt['identity'],
        producer=source_identity(('standard_score',))))
    folder = root / 'benchmark/evaluation' / benchmark / scene
    path = folder / 'receipt.json'
    if path.exists():
        previous = verified(path)
        if previous['input_identity'] != identity:
            raise ValueError('Completed standard score changed; preserve before affected repair')
        return previous
    folder.mkdir(parents=True, exist_ok=True)
    rows, aliases = {}, {}
    for method, manifest in manifests.items():
        if manifest['prediction_identity'] in aliases:
            original = aliases[manifest['prediction_identity']]
            rows[method] = dict(rows[original], alias_of=original, method=method)
            continue
        prediction = official_prediction(manifest)
        gt2pred, pred2gt = namespace['assign_instances_for_scan'](prediction, gt['GT']['path'])
        matches = {scene: dict(gt=gt2pred, pred=pred2gt)}
        metrics, classes, _ = ap_metrics(namespace, matches)
        match_path = folder / (method + '_matches.json.gz')
        match_record = write(match_path, dict(status='DONE', matches=_json(matches)))
        rows[method] = dict(status='DONE', method=method, scene=scene, benchmark=benchmark,
            metrics=metrics, per_class=classes, metric_unit='FRACTION',
            prediction=manifest['identity'], prediction_identity=manifest['prediction_identity'],
            matches=dict(path=str(match_path), identity=match_record['identity']), GT=gt['identity'],
            original_GT_instances=sum(len(v) for v in gt2pred.values()),
            point_count=gt['target_points'], full_GT_denominator=True)
        aliases[manifest['prediction_identity']] = method
        print('STANDARD_SCORED', benchmark, scene, method, metrics, flush=True)
    if formal_connection:
        # This is the real author entry on the same first formal predictions/GT,
        # not a synthetic imitation of its AP implementation.
        first = next(iter(manifests))
        with (folder / 'official_entry_agreement.log').open('w') as stream, contextlib.redirect_stdout(stream):
            official = namespace['evaluate']({scene: official_prediction(manifests[first])},
                str(Path(gt['GT']['path']).parent), str(folder / 'official.txt'), dataset='scannet200')[0]
        expected = [official['all_ap'], official['all_ap_50%'], official['all_ap_25%']]
        actual = [rows[first]['metrics'][key] for key in ('apall', 'ap50', 'ap25')]
        if not np.allclose(expected, actual, atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError('Common mask adapter differs from the unchanged official entry')
        write(folder / 'G3_official_entry_agreement.json', dict(status='DONE', formal_scene=scene,
            method=first, actual_common=actual, actual_author=expected, max_absolute_error=0.,
            synthetic=False, full_original_mesh_GT=True))
    return write(path, dict(status='DONE', input_identity=identity, rows=rows,
        benchmark=benchmark, scene=scene, prediction_lock=lock['identity'], GT=gt['identity'],
        official_entry_checked=formal_connection, unique_scorer_inputs=len(aliases),
        logical_rows=len(rows), metric_unit='FRACTION'))
