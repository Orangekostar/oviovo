#!/usr/bin/env python3
"""Diagnose added S2 masks with GT only after prediction/export has completed."""
import argparse
import json
from pathlib import Path
import re

import numpy as np


def read_predictions(path):
    result = {}
    for line in path.read_text().splitlines():
        name, label, score = line.split()
        instance = int(re.fullmatch(r'inst-(\d+)_label-\d+\.npy', name).group(1))
        result[instance] = {'mask': np.load(path.parent/name).reshape(-1).astype(bool),
                            'class_id': int(label), 'confidence': float(score)}
    return result


def diagnose_added(before, after, gt_ids, valid_ids, min_region=100):
    ids, sizes = np.unique(gt_ids, return_counts=True)
    gt = {int(i): gt_ids == i for i, n in zip(ids, sizes)
          if int(i)//1000 in valid_ids and n >= min_region}
    void = ~np.isin(gt_ids//1000, valid_ids)
    reports = []
    def matches(prediction):
        mask = prediction['mask']
        pairs = []
        for object_id, target in gt.items():
            intersect = int((mask & target).sum())
            if intersect:
                pairs.append((intersect / int((mask | target).sum()), object_id))
        return sorted(pairs, reverse=True)
    before_matches = {key: matches(value) for key, value in before.items()}
    for key in sorted(set(after)-set(before)):
        pred = after[key]
        overlaps = matches(pred)
        same_class = [(iou, obj) for iou, obj in overlaps if obj//1000 == pred['class_id']]
        best_iou, best_gt = same_class[0] if same_class else (0., None)
        ignored = void.copy()
        for obj, size in zip(ids, sizes):
            if obj//1000 == pred['class_id'] and size < min_region:
                ignored |= gt_ids == obj
        ignore_fraction = float(ignored[pred['mask']].mean())
        thresholds = {}
        for threshold in (.25, .5, .75):
            covered = (best_gt is not None and any(
                before[owner]['class_id'] == pred['class_id']
                and any(obj == best_gt and iou > threshold for iou, obj in pairs)
                for owner, pairs in before_matches.items()))
            if pred['class_id'] not in valid_ids:
                outcome = 'IGNORED'
            elif best_iou > threshold:
                outcome = 'DUPLICATE_FP' if covered else 'NOVEL_TP'
            elif ignore_fraction > threshold:
                outcome = 'IGNORED'
            else:
                outcome = 'FP'
            thresholds[str(threshold)] = {'outcome': outcome, 'best_gt_already_covered_by_B0': covered}
        reports.append({'instance_id': key, 'class_id': pred['class_id'],
                        'predicted_class_in_instance_vocabulary': pred['class_id'] in valid_ids,
                        'export_confidence': pred['confidence'], 'vertices': int(pred['mask'].sum()),
                        'best_class_consistent_iou': best_iou, 'best_class_consistent_gt': best_gt,
                        'best_class_agnostic_iou': overlaps[0][0] if overlaps else 0.,
                        'ignored_fraction': ignore_fraction, 'thresholds': thresholds})
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance-results', type=Path, required=True)
    p.add_argument('--fallback-readout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.instance_results/'instance_summary.json').read_text())
    before = read_predictions(args.instance_results/'B0/pred_inst_sem_mapping.txt')
    after = read_predictions(args.instance_results/'S2/pred_inst_sem_mapping.txt')
    gt = np.load(args.instance_results/'gt_sem_inst_id.npy').reshape(-1)
    diagnostics = diagnose_added(before, after, gt, summary['semantic_instance_valid_ids'])
    ledger = json.loads(args.fallback_readout.read_text())['fallback_ledger']
    accepted = [x for x in ledger if x['accepted']]
    totals = {}
    for threshold in ('0.25', '0.5', '0.75'):
        totals[threshold] = {name: sum(x['thresholds'][threshold]['outcome'] == name for x in diagnostics)
                             for name in ('NOVEL_TP', 'FP', 'DUPLICATE_FP', 'IGNORED')}
    result = {'accepted_native_single_query_instances': len(accepted),
              'exported_added_instances': len(diagnostics), 'diagnostics': diagnostics,
              'accepted_but_not_exported': [x['instance_id'] for x in accepted if x['instance_id'] not in after],
              'nonexport_reason': 'released min_region=100 or no projected geometry',
              'threshold_counts': totals, 'fallback_ledger': ledger,
              'definition': 'best class-consistent GT IoU > t; already-covered GT reported as duplicate; diagnostic not PR-curve attribution'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(totals,indent=2))


if __name__ == '__main__':
    main()
