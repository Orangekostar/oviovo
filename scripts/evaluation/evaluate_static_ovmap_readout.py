#!/usr/bin/env python3
"""Paired semantic evaluation on a frozen native projected instance map.

Only this evaluation program reads GT. Geometry projection is reused, not
reconstructed from GT. Projection parity must be audited independently.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file


def semantic_metrics(gt, prediction, valid_ids):
    valid = np.asarray(valid_ids)
    keep = np.isin(gt, valid)
    per_class = {}
    for label in valid:
        target = (gt == label) & keep
        count = int(target.sum())
        if not count:
            continue
        predicted = (prediction == label) & keep
        tp = int((target & predicted).sum())
        union = int((target | predicted).sum())
        per_class[str(label)] = {'iou': tp / union, 'accuracy': tp / count,
                                 'gt_vertices': count}
    return {'semantic_miou': float(np.mean([v['iou'] for v in per_class.values()])),
            'semantic_macc': float(np.mean([v['accuracy'] for v in per_class.values()])),
            'per_class': per_class}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout', type=Path, required=True)
    parser.add_argument('--projected-instance-map', type=Path, required=True)
    parser.add_argument('--projected-owner-array', type=Path)
    parser.add_argument('--original-semantic-map', type=Path, required=True)
    parser.add_argument('--gt-semantic-map', type=Path, required=True)
    parser.add_argument('--text-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    meshes = [PlyData.read(p)['vertex'].data for p in
              (args.projected_instance_map, args.original_semantic_map, args.gt_semantic_map)]
    base_vertices = meshes[0]
    for vertices in meshes[1:]:
        if len(vertices) != len(base_vertices) or any(not np.array_equal(vertices[c], base_vertices[c]) for c in ('x', 'y', 'z')):
            raise ValueError('evaluation vertex coordinates/order differ')
    instance_ids, original, gt = [np.asarray(v['label'], dtype=np.int64) for v in meshes]
    if args.projected_owner_array:
        owners = np.load(args.projected_owner_array, allow_pickle=False)
        if owners.shape != instance_ids.shape or not np.issubdtype(owners.dtype, np.integer) or np.any(owners < 0):
            raise ValueError('projected native owners must be aligned nonnegative integers')
        instance_ids = owners.astype(np.int64)
    text = np.load(args.text_cache, allow_pickle=False)
    baseline = json.loads((args.readout/'B0.json').read_text())['observations']
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    conditions = ['B0'] + [p.stem for p in sorted(args.readout.glob('*.json'))
                          if p.stem not in ('B0', 'input_binding')]
    for condition in conditions:
        doc = json.loads((args.readout/(condition+'.json')).read_text())
        result = doc['observations']
        labels = np.zeros(int(instance_ids.max())+1, dtype=np.int64)
        diagnostics = []
        for key, value in result.items():
            if value is None:
                continue
            instance = int(key)
            if instance < len(labels):
                labels[instance] = value['class_id']
            before = baseline[key]
            old_queries = before['selected_query_ids'] if before else []
            old_class = before['class_id'] if before else 0
            old_margin = before['margin'] if before else None
            removed = sorted(set(old_queries) - set(value['selected_query_ids']))
            added = sorted(set(value['selected_query_ids']) - set(old_queries))
            if removed or added or old_class != value['class_id'] or old_margin != value['margin']:
                diagnostics.append({'instance_id': instance, 'removed': removed, 'added': added,
                                    'old_class': old_class, 'new_class': value['class_id'],
                                    'old_margin': old_margin, 'new_margin': value['margin']})
        prediction = labels[instance_ids]
        if condition == 'B0' and not np.array_equal(prediction, original):
            raise ValueError(f'baseline parity failed on {np.count_nonzero(prediction != original)} vertices')
        np.save(args.output/(condition+'_semantic_labels.npy'), prediction)
        metrics = semantic_metrics(gt, prediction, text['valid_ids'])
        row = {'condition': condition, 'scene': doc['scene'], **metrics,
               'changed_observations': diagnostics, 'readout_seconds': doc['readout_seconds'],
               'geometry_and_instance_ids': ('FROZEN_FULL_NATIVE_OWNERS_INCLUDING_UNKNOWN' if args.projected_owner_array
                                             else 'FROZEN_IDENTICAL_PROJECTED_INPUT'),
               'original_semantic_exact_match': bool(np.array_equal(prediction, original)),
               'quality_mode': doc['quality_mode'], 'direction_mode': doc['direction_mode'],
               'history_scope': doc['history_scope'], 'mode': doc['mode'],
               'class_agnostic_ap50': None, 'class_aware_ap50': None,
               'missing_metric_reason': 'instance_evaluation_not_run_in_semantic_smoke',
               'status': 'COMPLETE_SEMANTIC_SMOKE'}
        (args.output/(condition+'_metrics.json')).write_text(json.dumps(row, indent=2)+'\n')
        rows.append({k:v for k,v in row.items() if k not in ('per_class','changed_observations')})
    receipt = {'rows': rows, 'evaluation_seconds': time.perf_counter()-start,
               'command': sys.argv, 'inputs': {str(p):sha256_file(p) for p in
                 (args.projected_instance_map, args.original_semantic_map, args.gt_semantic_map, args.text_cache)},
               'projection_status': 'REUSED_NATIVE_PROJECTION_PENDING_1NN_PARITY_AUDIT',
               'protocol_scope': 'Replica51_Room0_development_not_Replica8'}
    if args.projected_owner_array:
        receipt['inputs'][str(args.projected_owner_array)] = sha256_file(args.projected_owner_array)
    (args.output/'paired_summary.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
