"""GT-free proposal union; the legacy call retains its original default predictions."""
import numpy as np

from .fusion_attribution import build_frozen_candidates, apply_label_reuse, apply_fusion_nms


def fuse_proposals(owners, readout, masks, labels, scores, query_ids, *, reuse_iou=.5, nms_iou=.7,
                   borrow_labels=True, fusion_nms=True, scene_id='unspecified',
                   prediction_run='unspecified', ovi_readout_id='unbound_legacy_readout'):
    pool = build_frozen_candidates(owners, readout, masks, labels, scores, query_ids,
        scene_id=scene_id, prediction_run=prediction_run, ovi_readout_id=ovi_readout_id)
    final, borrowed = apply_label_reuse(pool, enabled=borrow_labels, reuse_iou=reuse_iou)
    kept, suppression = apply_fusion_nms(pool, enabled=fusion_nms, nms_iou=nms_iou)
    ledger = []
    for i, metadata in enumerate(pool.records):
        row = dict(metadata)
        event = borrowed[i]
        row.update(event, class_id=int(final[i]), area=int(pool.areas[i]),
            kept=suppression[i]['kept'], suppressed_by_candidate=suppression[i]['suppressed_by'],
            suppression_iou=suppression[i]['suppression_iou'], nms_visit_rank=suppression[i]['nms_visit_rank'])
        if row['source'] == 'OVI':
            row['owner_id'] = row['native_owner_id']
            row['selected_query_ids'] = list(row['source_query_ids'])
        else:
            row.update(query_id=row['spaceformer_query_id'], released_class_id=int(pool.original_labels[i]),
                       released_score=float(pool.released_scores[i]))
            owner = event['reused_owner']
            row['selected_query_ids'] = [] if owner is None else readout[str(owner)]['selected_query_ids']
        ledger.append(row)
    return {'masks': pool.masks[kept], 'class_ids': final[kept],
            'scores': pool.areas[kept].astype(np.float64), 'candidate_ids': kept, 'ledger': ledger}
