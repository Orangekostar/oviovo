"""Canonical diagnostic AP on an already bound strict-1NN native vertex projection.

This reuses the repository's confidence-ranked, one-to-one AP integration. It is
separate from released OVI mP/mR and does not establish the paper's AP provenance.
"""
import numpy as np

from .baselines.static_metrics import _instance_threshold


def projected_mask_metrics(predicted_masks, ground_truth, confidences, *, min_region=100):
    """Same canonical AP for overlapping proposals; never collapse to an owner map."""
    predicted_masks, ground_truth, confidences = map(np.asarray, (predicted_masks, ground_truth, confidences))
    if (predicted_masks.ndim != 2 or predicted_masks.dtype != bool or ground_truth.ndim != 1
            or predicted_masks.shape[1] != len(ground_truth) or confidences.shape != (len(predicted_masks),)
            or not np.isfinite(confidences).all() or min_region < 1
            or not np.issubdtype(ground_truth.dtype, np.integer) or np.any(ground_truth < 0)):
        raise ValueError('aligned boolean proposal masks, nonnegative GT IDs and finite confidences required')
    ids, counts = np.unique(ground_truth, return_counts=True)
    gt_sizes = {int(i): int(n) for i, n in zip(ids, counts) if i > 0 and n >= min_region}
    rows = []
    for index, mask in enumerate(predicted_masks):
        count = int(mask.sum())
        if count < min_region:
            continue
        overlaps, sizes = np.unique(ground_truth[mask], return_counts=True)
        rows.append((float(confidences[index]), index, count,
                     {int(i): int(n) for i, n in zip(overlaps, sizes)}))
    rows.sort(key=lambda row: (-row[0], row[1]))
    result = {'ground_truth_instance_count': len(gt_sizes), 'predicted_instance_count': len(rows),
        'definition': 'full_projected_domain_IoU_including_void; overlapping_proposals; greedy_1to1_IoU_ge_t; precision_envelope_AP',
        'min_region': min_region, 'zero_GT_policy': 'undefined_null'}
    for threshold in (.25, .5, .75):
        ap, recall = _instance_threshold(rows, gt_sizes, threshold)
        suffix = str(round(threshold*100))
        result['ap'+suffix] = ap if gt_sizes else None
        result['recall'+suffix] = recall if gt_sizes else None
    return result


def projected_instance_metrics(prediction, ground_truth, confidences, *, min_region=100):
    prediction, ground_truth = map(np.asarray, (prediction, ground_truth))
    if (prediction.ndim != 1 or prediction.shape != ground_truth.shape
            or not np.issubdtype(prediction.dtype, np.integer)
            or not np.issubdtype(ground_truth.dtype, np.integer)
            or np.any(prediction < 0) or np.any(ground_truth < 0) or min_region < 1):
        raise ValueError('aligned nonnegative projected instance IDs required')
    ids, counts = np.unique(ground_truth, return_counts=True)
    gt_sizes = {int(i): int(n) for i, n in zip(ids, counts) if i > 0 and n >= min_region}
    masks = []
    for owner in np.unique(prediction):
        if owner == 0:
            continue
        if int(owner) not in confidences or not np.isfinite(confidences[int(owner)]):
            raise ValueError('every predicted owner requires a finite GT-free confidence')
        mask = prediction == owner
        count = int(mask.sum())
        if count < min_region:
            continue
        overlaps, sizes = np.unique(ground_truth[mask], return_counts=True)
        masks.append((float(confidences[int(owner)]), int(owner), count,
                      {int(i): int(n) for i, n in zip(overlaps, sizes)}))
    masks.sort(key=lambda row: (-row[0], row[1]))
    result = {'ground_truth_instance_count': len(gt_sizes), 'predicted_instance_count': len(masks),
              'definition': 'full_projected_domain_IoU_including_void; greedy_1to1_IoU_ge_t; precision_envelope_AP',
              'min_region': min_region, 'zero_GT_policy': 'undefined_null'}
    for threshold in (.25, .5, .75):
        ap, recall = _instance_threshold(masks, gt_sizes, threshold)
        suffix = str(round(threshold*100))
        result['ap'+suffix] = ap if gt_sizes else None
        result['recall'+suffix] = recall if gt_sizes else None
    return result
