#!/usr/bin/env python3
"""Run native mask export and released OVI evaluators on paired readout outputs.

Only the named pure export functions are compiled from eval_sem_seg.py to avoid
loading an unrelated CLIP model dependency. Their bodies are not modified.
"""
import argparse
import ast
import contextlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file


def load_exports(source):
    tree = ast.parse(source.read_text())
    names = {'map_gt_mesh', 'map_pred_mesh'}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in body} != names:
        raise ValueError('released export interface changed')
    namespace = {'np': np, 'PlyData': PlyData, 'pjoin': os.path.join}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluator-root', type=Path, required=True)
    parser.add_argument('--semantic-results', type=Path, required=True)
    parser.add_argument('--projected-instance-map', type=Path, required=True)
    parser.add_argument('--projected-owner-array', type=Path)
    parser.add_argument('--original-semantic-map', type=Path, required=True)
    parser.add_argument('--gt-instance-map', type=Path, required=True)
    parser.add_argument('--gt-semantic-map', type=Path, required=True)
    parser.add_argument('--reference-pred-mapping', type=Path, required=True)
    parser.add_argument('--reference-gt-ids', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    semantic_files = sorted(args.semantic_results.glob('*_semantic_labels.npy'))
    if not semantic_files or not (args.semantic_results/'B0_semantic_labels.npy').is_file():
        raise ValueError('completed B0 semantic evaluation and semantic arrays are required')
    args.output.mkdir(parents=True, exist_ok=True)
    instance_map = args.projected_instance_map
    if args.projected_owner_array:
        mesh = PlyData.read(instance_map)
        owners = np.load(args.projected_owner_array, allow_pickle=False)
        target = mesh['vertex']['label']
        if (owners.shape != target.shape or not np.issubdtype(owners.dtype, np.integer)
                or np.any(owners < 0) or np.any(owners > np.iinfo(target.dtype).max)):
            raise ValueError('native owner projection cannot be exported without overflow or misalignment')
        target[:] = owners
        instance_map = args.output/'full_native_instance_projection.ply'
        mesh.write(instance_map)
    source = args.evaluator_root / 'scripts'
    exports = load_exports(source / 'eval_sem_seg.py')
    sys.path.insert(0, str(source))
    evaluator = runpy.run_path(str(source/'eval_utils.py'))
    released = runpy.run_path(str(source/'eval_inst_seg.py'))
    evaluator['init']('Replica')
    gt_file = exports['map_gt_mesh']({'inst_mesh_f': str(args.gt_instance_map),
         'sem_mesh_f': str(args.gt_semantic_map), 'res_folder': str(args.output)})
    if not np.array_equal(np.load(gt_file), np.load(args.reference_gt_ids)):
        raise ValueError('released GT instance conversion parity failed')
    with (args.output/'released_instance_stdout.txt').open('w') as handle:
        with contextlib.redirect_stdout(handle):
            released['assign_pred_inst_to_gt_inst'](str(args.gt_instance_map),
                  str(instance_map), str(args.output))
    values = [float(x) for x in (args.output/'released_instance_stdout.txt').read_text().split()]
    names = ['instance_miou', 'area_weighted_instance_iou', 'mP@0.75', 'mR@0.75',
             'mP@0.50', 'mR@0.50', 'mP@0.25', 'mR@0.25']
    if len(values) != len(names):
        raise ValueError('released instance output layout changed')
    class_agnostic = dict(zip(names, values))
    rows = []
    for array_path in semantic_files:
        condition = array_path.stem.removesuffix('_semantic_labels')
        destination = args.output / condition
        destination.mkdir(exist_ok=True)
        start = time.perf_counter()
        mesh = PlyData.read(args.original_semantic_map)
        prediction = np.load(array_path)
        if prediction.shape != mesh['vertex']['label'].shape or np.any((prediction < 0) | (prediction > 51)):
            raise ValueError('Replica semantic label shape or IDs invalid')
        mesh['vertex']['label'][:] = prediction
        semantic_path = destination/'semantic_map_gt_200.ply'
        mesh.write(semantic_path)
        pred_file = exports['map_pred_mesh']({'inst_mesh_f': str(instance_map),
             'sem_mesh_f': str(semantic_path), 'res_folder': str(destination)})
        if condition == 'B0':
            if Path(pred_file).read_bytes() != args.reference_pred_mapping.read_bytes():
                raise ValueError('native B0 mask/score/class manifest differs')
            for line in args.reference_pred_mapping.read_text().splitlines():
                name = line.split()[0]
                if not np.array_equal(np.load(destination/name), np.load(args.reference_pred_mapping.parent/name)):
                    raise ValueError('native B0 instance mask differs')
        averages = evaluator['evaluate'](str(destination), [pred_file], [gt_file], str(destination))
        record = {'condition': condition, 'released_semantic_instance': finite_json(averages),
                  'released_class_agnostic': class_agnostic,
                  'class_agnostic_geometry': ('identical_full_native_owners_including_unknown_all_conditions'
                        if args.projected_owner_array else 'identical_frozen_native_projected_ids_all_conditions'),
                  'ap_score_definition': 'native_area_divided_by_largest_same_class_area_rounded_6dp',
                  'canonical_class_agnostic_ap': None,
                  'canonical_class_agnostic_ap_reason': 'released_precision_not_integrated_AP',
                  'null_class_ap_reason': 'released_evaluator_no_GT_class_in_this_scene',
                  'evaluation_seconds_including_export': time.perf_counter()-start,
                  'prediction_manifest': pred_file, 'prediction_manifest_sha256': sha256_file(pred_file)}
        (destination/'metrics.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        rows.append(record)
        print(condition, averages['all_ap'], averages['all_ap_50%'], averages['all_ap_25%'], flush=True)
    namespace = evaluator['init'].__globals__
    receipt = {'rows': rows, 'command': sys.argv, 'source_sha256': {
        name:sha256_file(source/name) for name in ('eval_utils.py', 'eval_sem_seg.py', 'eval_inst_seg.py', 'utils/semantic_const.py')},
        'semantic_instance_valid_ids': namespace['VALID_CLASS_IDS'],
        'semantic_instance_class_labels': namespace['CLASS_LABELS'],
        'projection_scope': 'reused_native_1NN_pending_independent_audit',
        'input_sha256': {str(p):sha256_file(p) for p in (args.projected_instance_map,
            args.original_semantic_map, args.gt_instance_map, args.gt_semantic_map,
            args.reference_pred_mapping, args.reference_gt_ids)}}
    if args.projected_owner_array:
        receipt['input_sha256'][str(args.projected_owner_array)] = sha256_file(args.projected_owner_array)
    (args.output/'instance_summary.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
