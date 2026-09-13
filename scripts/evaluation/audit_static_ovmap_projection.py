#!/usr/bin/env python3
"""Rebuild native face ownership and independently check strict 5cm 1NN export."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import open3d as o3d
from plyfile import PlyData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-mesh', type=Path, required=True)
    parser.add_argument('--native-cache', type=Path, required=True)
    parser.add_argument('--baseline-readout', type=Path, required=True)
    parser.add_argument('--original-instance-map', type=Path, required=True)
    parser.add_argument('--original-semantic-map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    with args.native_cache.open('rb') as handle:
        cache = pickle.load(handle)
    baseline = json.loads(args.baseline_readout.read_text())['observations']
    mesh = o3d.t.io.read_triangle_mesh(str(args.native_mesh))
    xyz = mesh.vertex.positions
    lattice = {}
    positions = xyz.numpy()
    for voxel_size in (.01, .02):
        consistent = 0
        for begin in range(0, len(positions), 250000):
            block = positions[begin:begin+250000].astype(np.float64)
            grid = block / voxel_size - .5
            residual = np.abs(grid - np.round(grid)) * voxel_size
            consistent += int(((residual < 2e-6).sum(axis=1) >= 2).sum())
        lattice[str(voxel_size)] = {'vertices_on_two_voxel_center_axes': consistent,
                                  'fraction': consistent / len(positions)}
    colors = mesh.vertex.colors.numpy()
    faces = mesh.triangle.indices.numpy()
    face_colors = (colors[faces[:, 0]] * 255.).astype(np.uint8)
    sem = np.zeros(len(colors), dtype=np.int64)
    instance = np.zeros(len(colors), dtype=np.int64)
    for object_id, record in cache.items():
        pred = baseline[str(object_id)]
        if pred is None:
            continue
        ids = faces[np.all(face_colors == np.asarray(record['color']), axis=1)].reshape(-1)
        sem[ids] = pred['class_id']
        instance[ids] = int(object_id)
    target = PlyData.read(args.original_instance_map)['vertex'].data
    reference_sem = PlyData.read(args.original_semantic_map)['vertex'].data
    if len(target) != len(reference_sem) or any(not np.array_equal(target[c], reference_sem[c]) for c in ('x','y','z')):
        raise ValueError('original projection coordinates differ')
    target_xyz = np.column_stack([target[c] for c in ('x', 'y', 'z')]).astype(np.float32)
    print('loaded native mesh', len(colors), 'vertices;', len(target), 'evaluation vertices', flush=True)
    tree = o3d.core.nns.NearestNeighborSearch(xyz)
    tree.knn_index()
    nearest, squared = tree.knn_search(o3d.core.Tensor(target_xyz), 1)
    nearest, squared = nearest.numpy().reshape(-1), squared.numpy().reshape(-1)
    matched = squared < .05**2
    projected_sem = np.where(matched, sem[nearest], 0)
    projected_ids = np.where(matched, instance[nearest], 0)
    sem_difference = projected_sem != reference_sem['label']
    instance_difference = projected_ids != target['label']
    result = {'status': 'PASS' if not (sem_difference.any() or instance_difference.any()) else 'MISMATCH',
              'native_vertex_count': len(colors), 'target_vertex_count': len(target),
              'voxel_center_lattice_diagnostic': lattice,
              'lattice_tolerance_m': 2e-6,
              'semantic_mismatched_vertices': int(sem_difference.sum()),
              'instance_mismatched_vertices': int(instance_difference.sum()),
              'unmatched_vertices': int((~matched).sum()),
              'at_threshold_vertices': int((squared == .05**2).sum()),
              'threshold': 'distance_squared < 0.05**2', 'unmatched_id': 0,
              'ownership_rule': 'native_first_face_vertex_color_assign_all_face_vertices_in_cache_order',
              'instance_filter': 'native_min_two_semantic_queries; geometry_id_readout_coupling_preserved',
              'open3d_version': o3d.__version__, 'nn_device': str(xyz.device),
              'seconds': time.perf_counter()-start, 'command': sys.argv}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output/'projection_arrays.npz', nearest=nearest, distance_squared=squared,
             semantic_labels=projected_sem, instance_ids=projected_ids)
    (args.output/'projection_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
