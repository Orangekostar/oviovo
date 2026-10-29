"""Native full-method formats not covered by the frozen OVI export producer."""
import numpy as np


def decode_maskclustering(path, *, target_points, class_ids):
    """Keep original NPZ column order, overlap, empty slots and author scores."""
    with np.load(path, allow_pickle=False) as data:
        masks, classes, scores = (data[k].copy() for k in ('pred_masks', 'pred_classes', 'pred_score'))
    if masks.ndim != 2 or masks.shape[0] != target_points:
        raise ValueError('Author MaskClustering point domain differs')
    if (classes.shape != (masks.shape[1],) or scores.shape != classes.shape
            or not np.issubdtype(classes.dtype, np.integer) or not np.isfinite(scores).all()):
        raise ValueError('Author MaskClustering class/confidence axes are malformed')
    if not np.isin(masks, [0, 1]).all():
        raise ValueError('Author MaskClustering masks are not binary')
    # The original query leaves an empty zero-label column when no representative
    # masks exist. Keep that column; a nonempty unknown class is incompatible.
    valid = np.isin(classes, class_ids) | ((classes == 0) & ~masks.any(axis=0))
    if not valid.all():
        raise ValueError('Author MaskClustering class IDs differ from the locked vocabulary')
    return dict(pred_masks=masks.astype(bool), pred_classes=classes, pred_scores=scores)
