#!/usr/bin/env python3
"""Canonical full-owner geometry diagnostic; GT enters only this evaluator."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import open3d as o3d
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.baselines.ovimap import parse_instance_color_log
from src.evaluation.static_projected_instances import projected_instance_metrics
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.native_partition import native_partition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native-mesh', 'native-cache', 'color-log', 'gt-instance-map', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--scene', required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    colors = parse_instance_color_log(args.color_log)
    with args.native_cache.open('rb') as handle:
        cache = pickle.load(handle)
    cache_only = []
    for owner, record in cache.items():
        color = tuple(map(int, record['color']))
        if owner in colors and colors[owner] != color:
            raise ValueError('cache and C++ native colors disagree')
        if owner not in colors:
            cache_only.append(int(owner))
        colors[int(owner)] = color
    mesh = o3d.t.io.read_triangle_mesh(str(args.native_mesh))
    owners = native_partition(mesh.triangle.indices.numpy(),
        (mesh.vertex.colors.numpy()*255.).astype(np.uint8), colors)
    target = PlyData.read(args.gt_instance_map)['vertex'].data
    xyz = np.column_stack([target[k] for k in ('x', 'y', 'z')]).astype(np.float32)
    tree = o3d.core.nns.NearestNeighborSearch(mesh.vertex.positions)
    tree.knn_index()
    nearest, squared = tree.knn_search(o3d.core.Tensor(xyz), 1)
    nearest, squared = nearest.numpy().reshape(-1), squared.numpy().reshape(-1)
    projected = np.where(squared < .05**2, owners[nearest], 0)
    ids, counts = np.unique(owners[owners > 0], return_counts=True)
    confidences = {int(k): int(v)/int(counts.max()) for k, v in zip(ids, counts)}
    metrics = projected_instance_metrics(projected, np.asarray(target['label'], dtype=np.int64), confidences)
    np.save(args.output/'native_owners.npy', owners)
    np.save(args.output/'projected_owners.npy', projected)
    (args.output/'confidence.json').write_text(json.dumps(confidences, indent=2)+'\n')
    result = {'status': 'COMPLETE_CANONICAL_GEOMETRY_DIAGNOSTIC', 'scene': args.scene,
              'canonical_class_agnostic': metrics, 'paper_AP': None,
              'paper_AP_reason': 'canonical diagnostic does not verify paper AP implementation',
              'confidence': 'native_vertex_count_divided_by_largest_native_object_no_GT',
              'geometry_scope': 'full_native_owners_including_unknown_semantics; no min_two_query_filter',
              'strict_projection': 'distance_squared < 0.05**2',
              'native_vertices': len(owners), 'target_vertices': len(target),
              'mesh_backed_owners': len(ids), 'cache_only_color_ids': sorted(cache_only),
              'unmatched_vertices': int(np.count_nonzero(squared >= .05**2)),
              'seconds_including_export': time.perf_counter()-start, 'command': sys.argv,
              'input_sha256': {str(f): sha256_file(f) for f in
                  (args.native_mesh, args.native_cache, args.color_log, args.gt_instance_map)},
              'source_sha256': {str(f): sha256_file(f) for f in
                  (Path(__file__), Path(__file__).resolve().parents[2]/'src/static_ovmap/native_partition.py',
                   Path(__file__).resolve().parents[2]/'src/evaluation/static_projected_instances.py')},
              'output_sha256': {name: sha256_file(args.output/name) for name in
                                ('native_owners.npy', 'projected_owners.npy', 'confidence.json')}}
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['canonical_class_agnostic']))


if __name__ == '__main__':
    main()
