#!/usr/bin/env python3
"""Evaluate overlapping world-coordinate proposals; GT remains evaluation-only."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import open3d as o3d
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evaluation.evaluate_static_ovmap_instances import finite_json, load_exports
from scripts.evaluation.evaluate_static_ovmap_readout import semantic_metrics
from src.evaluation.static_projected_instances import projected_mask_metrics
from src.static_ovmap.cache_io import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('proposals', 'gt-instance-map', 'gt-semantic-map', 'text-cache', 'evaluator-root', 'reference-gt-ids', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with np.load(args.proposals, allow_pickle=False) as data:
        coord, masks, labels, scores = data['coord'], data['masks'], data['class_ids'], data['scores']
        score_definition = str(data['score_definition']) if 'score_definition' in data else 'released_objectness_mask_quality_times_class_probability'
    if (coord.ndim != 2 or coord.shape[1] != 3 or masks.shape != (len(labels), len(coord))
            or masks.dtype != bool or scores.shape != labels.shape or not np.isfinite(coord).all()
            or not np.isfinite(scores).all()):
        raise ValueError('aligned finite world-coordinate proposals required')
    inst_mesh, sem_mesh = [PlyData.read(p)['vertex'].data for p in (args.gt_instance_map, args.gt_semantic_map)]
    if len(inst_mesh) != len(sem_mesh) or any(not np.array_equal(inst_mesh[c], sem_mesh[c]) for c in ('x', 'y', 'z')):
        raise ValueError('GT domains differ')
    target = np.column_stack([inst_mesh[c] for c in ('x', 'y', 'z')]).astype(np.float32)
    tree = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(np.ascontiguousarray(coord, dtype=np.float32)))
    tree.knn_index()
    nearest, squared = tree.knn_search(o3d.core.Tensor(target), 1)
    nearest, squared = nearest.numpy().reshape(-1), squared.numpy().reshape(-1)
    matched = squared < .05**2
    projected = masks[:, nearest] & matched[None, :]
    source_area = masks.sum(axis=1)
    canonical = {'provided_score': projected_mask_metrics(projected, inst_mesh['label'], scores),
                 'common_source_area_score': projected_mask_metrics(projected, inst_mesh['label'], source_area)}
    source = args.evaluator_root/'scripts'
    sys.path.insert(0, str(source))
    exports = load_exports(source/'eval_sem_seg.py')
    from src.static_ovmap.released_loader import load_released_module
    evaluator = load_released_module(source/'eval_utils.py')
    evaluator['init']('Replica')
    gt_file = exports['map_gt_mesh']({'inst_mesh_f': str(args.gt_instance_map), 'sem_mesh_f': str(args.gt_semantic_map),
                                   'res_folder': str(args.output)})
    if not np.array_equal(np.load(gt_file), np.load(args.reference_gt_ids)):
        raise ValueError('released GT conversion differs from native reference')
    rows = []
    valid_ids = np.load(args.text_cache, allow_pickle=False)['valid_ids']
    if not np.isin(labels, valid_ids).all():
        raise ValueError('proposal class IDs outside explicit native vocabulary')
    area_scores = np.zeros(len(labels), dtype=np.float64)
    for label in np.unique(labels):
        keep = labels == label
        area_scores[keep] = source_area[keep]/max(int(source_area[keep].max()), 1)
    for score_name, confidence in [('provided_score', scores), ('native_area_control', area_scores)]:
        destination = args.output/score_name; destination.mkdir()
        manifest = destination/'pred_inst_sem_mapping.txt'
        lines = []
        for i, mask in enumerate(projected):
            name = f'pred_{i:03d}.npy'
            np.save(destination/name, mask)
            lines.append(f'{name} {int(labels[i])} {float(confidence[i]):.6f}')
        manifest.write_text('\n'.join(lines)+'\n')
        averages = evaluator['evaluate'](str(destination), [str(manifest)], [gt_file], str(destination))
        semantic = np.zeros(len(target), dtype=np.int64)
        for i in np.argsort(-confidence, kind='stable'):
            semantic[projected[i] & (semantic == 0)] = labels[i]
        rows.append({'confidence': score_name, 'released_semantic_instance': finite_json(averages),
            'semantic': semantic_metrics(sem_mesh['label'], semantic, valid_ids),
            'semantic_overlap_resolution': 'descending_proposal_confidence_first_assignment'})
    np.savez_compressed(args.output/'projection.npz', nearest=nearest, distance_squared=squared, matched=matched)
    receipt = {'status': 'COMPLETE_OVERLAPPING_PROPOSAL_EVALUATION', 'canonical_class_agnostic': canonical,
        'rows': rows, 'provided_score_definition': score_definition,
        'unmatched_gt_vertices': int((~matched).sum()), 'gt_vertex_count': len(target),
        'point_geometry': 'same_RGBD_frames_new_observed_cloud_NOT_frozen_B0_geometry',
        'projection': 'strict_distance_lt_0.05m_world_coordinates_1NN',
        'metric_scope': 'canonical_diagnostic_AP_and_released_semantic_AP; no_paper_AP_parity_claim',
        'elapsed_seconds': time.perf_counter()-start, 'command': sys.argv,
        'input_sha256': {str(p): sha256_file(p) for p in (args.proposals, args.gt_instance_map, args.gt_semantic_map,
            args.text_cache, args.reference_gt_ids)},
        'source_sha256': {str(p): sha256_file(p) for p in (Path(__file__), source/'eval_utils.py', source/'eval_sem_seg.py')}}
    (args.output/'metrics.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'canonical': canonical, 'semantic_AP': [r['released_semantic_instance']['all_ap'] for r in rows]}, indent=2))


if __name__ == '__main__':
    main()
