#!/usr/bin/env python3
"""Evaluate native G1/G2 partitions; GT is confined to this evaluation process."""
import argparse
import contextlib
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import open3d as o3d
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evaluation.evaluate_static_ovmap_instances import load_exports, finite_json
from scripts.evaluation.evaluate_static_ovmap_readout import semantic_metrics
from src.evaluation.static_projected_instances import projected_instance_metrics
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.native_export import compact_export, remap_partition, restore_partition


def best_overlap(mask, gt, min_region=100):
    ids, counts = np.unique(gt, return_counts=True)
    results = []
    for owner, size in zip(ids, counts):
        if owner == 0 or size < min_region:
            continue
        target = gt == owner
        intersection = int((mask & target).sum())
        if intersection:
            results.append({'gt_id': int(owner), 'iou': intersection/int((mask | target).sum()),
                            'prediction_purity': intersection/max(int(mask.sum()), 1)})
    return max(results, key=lambda r: r['iou']) if results else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--split', type=Path)
    p.add_argument('--readout', type=Path, required=True)
    p.add_argument('--native-mesh', type=Path, required=True)
    p.add_argument('--evaluator-root', type=Path, required=True)
    p.add_argument('--original-instance-map', type=Path, required=True)
    p.add_argument('--original-semantic-map', type=Path, required=True)
    p.add_argument('--gt-instance-map', type=Path, required=True)
    p.add_argument('--gt-semantic-map', type=Path, required=True)
    p.add_argument('--reference-pred-mapping', type=Path, required=True)
    p.add_argument('--reference-gt-ids', type=Path, required=True)
    p.add_argument('--text-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    start = time.perf_counter()
    graph = json.loads(args.graph.read_text())
    if graph['input_sha256'].get(str(args.native_mesh)) != sha256_file(args.native_mesh):
        raise ValueError('native geometry does not match graph evidence')
    native_dir = args.graph.parent
    native_b0 = np.load(native_dir/'B0_native_owners.npy', allow_pickle=False)
    native_g1 = np.load(native_dir/'G1_native_owners.npy', allow_pickle=False)
    expected, _ = remap_partition(native_b0, graph['partition']['components'])
    if not np.array_equal(expected, native_g1):
        raise ValueError('G1 array does not match recorded component partition')
    conditions = [('B0', native_b0, native_dir/'B0_native_owners.npy'),
                  ('G1', native_g1, native_dir/'G1_native_owners.npy')]
    if args.split:
        split = json.loads(args.split.read_text())
        if split['input_sha256'].get(str(args.graph)) != sha256_file(args.graph):
            raise ValueError('split and G1 graph identity mismatch')
        native_g2 = np.load(args.split.parent/'G2_native_owners.npy', allow_pickle=False)
        with np.load(args.split.parent/'G2_restoration.npz', allow_pickle=False) as ledger:
            if not np.array_equal(restore_partition(native_g2, ledger), native_b0):
                raise ValueError('G2 does not restore original native partition')
        reconstructed_parent = native_g2.copy()
        for child, parent in split['child_sources'].items():
            reconstructed_parent[native_g2 == int(child)] = parent
        if not np.array_equal(reconstructed_parent, native_g1):
            raise ValueError('G2 child assignment crosses a declared G1 parent')
        conditions.append(('G2', native_g2, args.split.parent/'G2_native_owners.npy'))
    originals = [PlyData.read(f) for f in (args.original_instance_map, args.original_semantic_map,
                                         args.gt_instance_map, args.gt_semantic_map)]
    vertices = originals[0]['vertex'].data
    for mesh in originals[1:]:
        if len(mesh['vertex']) != len(vertices) or any(not np.array_equal(vertices[c], mesh['vertex'][c]) for c in ('x','y','z')):
            raise ValueError('evaluation coordinates/order differ')
    args.output.mkdir(parents=True, exist_ok=False)
    native_mesh = o3d.t.io.read_triangle_mesh(str(args.native_mesh))
    if len(native_mesh.vertex.positions) != len(native_b0):
        raise ValueError('native ID array and geometry length differ')
    tree = o3d.core.nns.NearestNeighborSearch(native_mesh.vertex.positions)
    tree.knn_index()
    target = np.column_stack([vertices[c] for c in ('x','y','z')]).astype(np.float32)
    nearest, squared = tree.knn_search(o3d.core.Tensor(target), 1)
    nearest, squared = nearest.numpy().reshape(-1), squared.numpy().reshape(-1)
    matched = squared < .05**2
    np.savez(args.output/'projection_arrays.npz', nearest=nearest, distance_squared=squared)
    projection_seconds = time.perf_counter()-start
    gt_instance = np.asarray(originals[2]['vertex']['label'], dtype=np.int64)
    gt_semantic = np.asarray(originals[3]['vertex']['label'], dtype=np.int64)
    text = np.load(args.text_cache, allow_pickle=False)
    source = args.evaluator_root/'scripts'
    exports = load_exports(source/'eval_sem_seg.py')
    sys.path.insert(0, str(source))
    evaluator = runpy.run_path(str(source/'eval_utils.py'))
    released = runpy.run_path(str(source/'eval_inst_seg.py'))
    evaluator['init']('Replica')
    gt_file = exports['map_gt_mesh']({'inst_mesh_f': str(args.gt_instance_map),
        'sem_mesh_f': str(args.gt_semantic_map), 'res_folder': str(args.output)})
    if not np.array_equal(np.load(gt_file), np.load(args.reference_gt_ids)):
        raise ValueError('native GT conversion parity failed')
    rows, projected = [], {}
    for condition, native, native_path in conditions:
        condition_start = time.perf_counter()
        output = args.output/condition
        output.mkdir()
        readout_file = args.readout/f'{condition}.json'
        readout = json.loads(readout_file.read_text())
        if readout['graph_sha256'] != sha256_file(args.graph):
            raise ValueError('graph/readout identity mismatch')
        if condition == 'G2' and readout['split_sha256'] != sha256_file(args.split):
            raise ValueError('split/readout identity mismatch')
        owners = np.where(matched, native[nearest], 0)
        projected[condition] = owners
        np.save(output/'projected_owners.npy', owners)
        labels = np.zeros(len(owners), dtype=np.int64)
        for owner, pred in readout['observations'].items():
            if pred is not None:
                labels[owners == int(owner)] = pred['class_id']
        if condition == 'B0':
            if not np.array_equal(labels, originals[1]['vertex']['label']):
                raise ValueError('native B0 semantic projection parity failed')
            if not np.array_equal(np.where(labels > 0, owners, 0), originals[0]['vertex']['label']):
                raise ValueError('native B0 filtered instance projection parity failed')
        np.save(output/'semantic_labels.npy', labels)
        inst_mesh = PlyData.read(args.original_instance_map)
        target_labels = inst_mesh['vertex']['label']
        if int(owners.max()) > np.iinfo(target_labels.dtype).max:
            encoded, id_mapping = compact_export(owners, target_labels.dtype)
        else:
            encoded = owners.astype(target_labels.dtype)
            id_mapping = {int(i): int(i) for i in np.unique(owners)}
        target_labels[:] = encoded
        inst_path = output/'instance_projection.ply'
        inst_mesh.write(inst_path)
        (output/'export_id_mapping.json').write_text(json.dumps(id_mapping, indent=2)+'\n')
        sem_mesh = PlyData.read(args.original_semantic_map)
        if np.any(labels > np.iinfo(sem_mesh['vertex']['label'].dtype).max):
            raise ValueError('semantic label export overflow')
        sem_mesh['vertex']['label'][:] = labels
        sem_path = output/'semantic_projection.ply'
        sem_mesh.write(sem_path)
        with (output/'released_class_agnostic_stdout.txt').open('w') as f, contextlib.redirect_stdout(f):
            released['assign_pred_inst_to_gt_inst'](str(args.gt_instance_map), str(inst_path), str(output))
        numbers = [float(x) for x in (output/'released_class_agnostic_stdout.txt').read_text().split()]
        names = ['instance_miou', 'area_weighted_instance_iou', 'mP@0.75', 'mR@0.75',
                 'mP@0.50', 'mR@0.50', 'mP@0.25', 'mR@0.25']
        if len(numbers) != len(names):
            raise ValueError('released class-agnostic output layout changed')
        counts, sizes = np.unique(native[native > 0], return_counts=True)
        confidence = {int(i): int(n)/int(sizes.max()) for i, n in zip(counts, sizes)}
        canonical = projected_instance_metrics(owners, gt_instance, confidence)
        pred_file = exports['map_pred_mesh']({'inst_mesh_f': str(inst_path),
            'sem_mesh_f': str(sem_path), 'res_folder': str(output)})
        if condition == 'B0':
            if Path(pred_file).read_bytes() != args.reference_pred_mapping.read_bytes():
                raise ValueError('native B0 semantic instance manifest differs')
            for line in args.reference_pred_mapping.read_text().splitlines():
                name = line.split()[0]
                if not np.array_equal(np.load(output/name), np.load(args.reference_pred_mapping.parent/name)):
                    raise ValueError('native B0 semantic instance mask differs')
        with (output/'released_semantic_stdout.txt').open('w') as f, contextlib.redirect_stdout(f):
            ap = evaluator['evaluate'](str(output), [pred_file], [gt_file], str(output))
        row = {'condition': condition, 'released_class_agnostic': dict(zip(names, numbers)),
            'canonical_class_agnostic_ap': canonical, 'released_semantic_instance': finite_json(ap),
            'semantic': semantic_metrics(gt_semantic, labels, text['valid_ids']),
            'canonical_confidence': 'native_vertex_count_divided_by_largest_native_object_no_GT',
            'paper_class_agnostic_ap': None, 'paper_ap_reason': 'paper_AP_implementation_not_bound',
            'evaluation_seconds': time.perf_counter()-condition_start,
            'readout_sha256': sha256_file(readout_file), 'native_owner_sha256': sha256_file(native_path)}
        (output/'metrics.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
        rows.append(row)
        print(condition, 'canonical AP50/AP75', canonical['ap50'], canonical['ap75'],
              'semantic AP', ap['all_ap'], flush=True)
    diagnostics = []
    for component in graph['partition']['components']:
        if len(component) < 2:
            continue
        before = {str(i): best_overlap(projected['B0'] == i, gt_instance) for i in component}
        after = best_overlap(projected['G1'] == min(component), gt_instance)
        dominant = {x['gt_id'] for x in before.values() if x and x['prediction_purity'] >= .5}
        diagnostics.append({'members': component, 'before': before, 'after': after,
                            'multiple_dominant_GT_warning': len(dominant) > 1,
                            'diagnostic_only_not_used_for_layer_selection': True})
    receipt = {'rows': rows, 'merge_diagnostics': diagnostics, 'command': sys.argv,
        'scope': 'Replica_Room0_development_full_native_unknown_geometry_included',
        'projection': {'method': 'fresh_Open3D_CPU_1NN', 'strict_distance_m': .05,
            'unmatched_vertices': int((~matched).sum()), 'seconds': projection_seconds,
            'arrays_sha256': sha256_file(args.output/'projection_arrays.npz')},
        'B0_original_semantic_and_filtered_instance_and_all_mask_manifest_parity': True,
        'input_sha256': {str(f): sha256_file(f) for f in (args.graph, args.native_mesh,
            args.original_instance_map, args.original_semantic_map, args.gt_instance_map,
            args.gt_semantic_map, args.reference_pred_mapping, args.reference_gt_ids, args.text_cache)},
        'source_sha256': {str(f):sha256_file(f) for f in (source/'eval_utils.py', source/'eval_inst_seg.py',
            source/'eval_sem_seg.py', source/'utils/semantic_const.py',
            Path(__file__).resolve().parents[2]/'src/evaluation/static_projected_instances.py',
            Path(__file__).resolve().parents[2]/'src/evaluation/baselines/static_metrics.py')}}
    if args.split:
        receipt['input_sha256'][str(args.split)] = sha256_file(args.split)
        receipt['split_diagnostics'] = []
        for decision in split['decisions']:
            if not decision['accepted']:
                continue
            parent = decision['parent_id']
            descendants = sorted(int(child) for child, source in split['child_sources'].items() if source == parent)
            receipt['split_diagnostics'].append({'parent': parent,
                'before': best_overlap(projected['G1'] == parent, gt_instance),
                'children': {str(child): best_overlap(projected['G2'] == child, gt_instance) for child in descendants},
                'residual': best_overlap(projected['G2'] == parent, gt_instance),
                'diagnostic_only_not_used_for_layer_selection': True})
    (args.output/'graph_evaluation.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
