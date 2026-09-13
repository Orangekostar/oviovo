"""GT-free conservative proposal union, with geometric label reuse across spaces."""
import numpy as np


def fuse_proposals(owners, readout, masks, labels, scores, query_ids, *, reuse_iou=.5, nms_iou=.7):
    owners, masks, labels, scores, query_ids = map(np.asarray, (owners, masks, labels, scores, query_ids))
    if (owners.ndim != 1 or not np.issubdtype(owners.dtype, np.integer) or np.any(owners < 0)
            or masks.shape != (len(labels), len(owners)) or masks.dtype != bool
            or scores.shape != labels.shape or query_ids.shape != labels.shape
            or not np.isfinite(scores).all() or not 0 < reuse_iou <= 1 or not 0 < nms_iou <= 1):
        raise ValueError('aligned native owners and finite independent proposals required')
    ids, counts = np.unique(owners, return_counts=True)
    eligible = {int(owner): int(count) for owner, count in zip(ids, counts)
                if owner > 0 and readout.get(str(owner)) is not None}
    candidates, ledger = [], []
    for owner in sorted(eligible, key=lambda i: (-eligible[i], i)):
        candidates.append(owners == owner)
        ledger.append({'source': 'OVI_S1a', 'owner_id': owner, 'class_id': readout[str(owner)]['class_id'],
            'selected_query_ids': readout[str(owner)]['selected_query_ids'], 'area': eligible[owner]})
    for i in np.argsort(-scores, kind='stable'):
        mask = masks[i]
        area = int(mask.sum())
        overlaps, sizes = np.unique(owners[mask], return_counts=True)
        matches = [(int(owner), int(size)/(eligible[int(owner)]+area-int(size)))
                   for owner, size in zip(overlaps, sizes) if int(owner) in eligible]
        best_owner, best_iou = max(matches, key=lambda row: (row[1], -row[0])) if matches else (None, 0.)
        reuse = best_owner if best_iou >= reuse_iou else None
        label = int(labels[i]) if reuse is None else readout[str(reuse)]['class_id']
        candidates.append(mask)
        ledger.append({'source': 'SpaCeFormer', 'query_id': int(query_ids[i]), 'class_id': label,
            'released_class_id': int(labels[i]), 'released_score': float(scores[i]), 'area': area,
            'best_owner_id': best_owner, 'best_owner_iou': best_iou, 'reused_owner': reuse,
            'selected_query_ids': [] if reuse is None else readout[str(reuse)]['selected_query_ids']})
    kept = []
    for i, mask in enumerate(candidates):
        suppressor = None
        if ledger[i]['area']:
            for j in kept:
                intersection = int(np.count_nonzero(mask & candidates[j]))
                overlap = intersection/(ledger[i]['area']+ledger[j]['area']-intersection)
                if overlap >= nms_iou:
                    suppressor = j
                    break
            if suppressor is None:
                kept.append(i)
        ledger[i]['kept'] = suppressor is None and ledger[i]['area'] > 0
        ledger[i]['suppressed_by_candidate'] = suppressor
    return {'masks': np.stack([candidates[i] for i in kept]) if kept else np.zeros((0, len(owners)), dtype=bool),
        'class_ids': np.array([ledger[i]['class_id'] for i in kept], dtype=np.int64),
        'scores': np.array([ledger[i]['area'] for i in kept], dtype=np.float64),
        'candidate_ids': np.array(kept, dtype=np.int64), 'ledger': ledger}
