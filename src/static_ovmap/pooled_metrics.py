"""Pool vertex confusion counts before averaging over present GT classes."""
import numpy as np


def pooled_semantic_metrics(pairs, valid_ids):
    valid = np.asarray(valid_ids, dtype=np.int64)
    if valid.ndim != 1 or not len(valid) or np.any(valid <= 0) or len(np.unique(valid)) != len(valid):
        raise ValueError('unique positive semantic IDs required')
    lookup = np.full(int(valid.max())+1, -1, dtype=np.int64)
    lookup[0] = 0
    lookup[valid] = np.arange(1, len(valid)+1)
    width = len(valid)+1
    confusion = np.zeros((width, width), dtype=np.uint64)
    for gt, prediction in pairs:
        gt, prediction = np.asarray(gt), np.asarray(prediction)
        if (gt.ndim != 1 or gt.shape != prediction.shape
                or not np.issubdtype(gt.dtype, np.integer)
                or not np.issubdtype(prediction.dtype, np.integer)
                or np.any(prediction < 0) or np.any(prediction >= len(lookup))
                or np.any(lookup[prediction] < 0)):
            raise ValueError('aligned integer semantic labels in the bound vocabulary required')
        keep = np.isin(gt, valid)
        codes = lookup[gt[keep]]*width + lookup[prediction[keep]]
        confusion += np.bincount(codes, minlength=width*width).reshape(width, width).astype(np.uint64)
    rows, columns = confusion.sum(axis=1), confusion.sum(axis=0)
    per_class = {}
    for index, label in enumerate(valid, 1):
        if not rows[index]:
            continue
        tp = int(confusion[index, index]); target = int(rows[index]); predicted = int(columns[index])
        per_class[str(label)] = {'iou': tp/(target+predicted-tp), 'accuracy': tp/target,
                                 'gt_vertices': target, 'tp': tp, 'fp': predicted-tp, 'fn': target-tp}
    return {'semantic_miou': float(np.mean([r['iou'] for r in per_class.values()])) if per_class else None,
            'semantic_macc': float(np.mean([r['accuracy'] for r in per_class.values()])) if per_class else None,
            'per_class': per_class, 'confusion': confusion.tolist(),
            'confusion_ids': [0, *valid.tolist()], 'valid_gt_vertices': int(confusion.sum()),
            'definition': 'pooled_confusion; GT void excluded; prediction void remains false negative; mean over present GT classes',
            'empty_gt_policy': 'undefined_null'}
