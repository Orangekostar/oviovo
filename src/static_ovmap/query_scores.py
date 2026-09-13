"""Explicit vocabulary matching in a single bound image/text feature space."""
import numpy as np


def score_queries(feature, text, valid_ids, feature_space_id, text_space_id, temperature=1.):
    feature, text, ids = np.asarray(feature), np.asarray(text), np.asarray(valid_ids)
    if (feature_space_id != text_space_id or feature.ndim != 1 or text.ndim != 2
            or text.shape != (len(ids), len(feature)) or len(ids) < 2 or temperature <= 0
            or not np.issubdtype(ids.dtype, np.integer) or np.any(ids <= 0)
            or len(np.unique(ids)) != len(ids) or not np.isfinite(feature).all()
            or not np.isfinite(text).all()):
        raise ValueError('explicit vocabulary and identical bound feature spaces required')
    scores = text @ feature / (np.maximum(np.linalg.norm(text, axis=1), 1e-8)
                               * max(np.linalg.norm(feature), 1e-8) * temperature)
    order = np.argsort(-scores, kind='stable')
    return {'class_id': int(ids[order[0]]), 'score': float(scores[order[0]]),
            'margin': float(scores[order[0]]-scores[order[1]]), 'feature_space_id': feature_space_id,
            'temperature': float(temperature)}
