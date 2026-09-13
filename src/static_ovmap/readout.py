"""Feature fusion and text matching, independent of geometry and evaluation."""
import numpy as np

from .observation_bank import quality_scores, select_observations


def _check_space(observations):
    identities = {(o.feature_space_id, o.feature_dim) for o in observations}
    objects = {(o.scene_id, o.instance_id) for o in observations}
    if len(identities) != 1:
        raise ValueError('cannot mix feature spaces or dimensions')
    if len(objects) != 1:
        raise ValueError('cannot fuse different instances or scenes')
    return next(iter(identities))


def fuse_features(observations, weighting='vis_area'):
    obs = list(observations)
    _check_space(obs)
    if weighting == 'vis_area':
        weights = np.asarray([o.visible_area_px for o in obs], dtype=np.float64)
    elif weighting == 'quality':
        weights = quality_scores(obs)
    else:
        raise ValueError('unknown weighting: ' + weighting)
    weights = weights / (weights.sum() + 1e-6)
    # Native features are NOT normalized individually before weighted fusion.
    return (np.stack([o.feature for o in obs]) * weights[:, None]).sum(axis=0)


def classify(observations, text_features, valid_ids, feature_space_id,
             *, strategy='last8', weighting='vis_area', seed=0,
             canonical_features=None):
    obs = list(observations)
    if not obs:
        return None
    space, dim = _check_space(obs)
    if space != feature_space_id:
        raise ValueError('text and image feature space mismatch')
    text = np.asarray(text_features)
    ids = np.asarray(valid_ids)
    if (text.ndim != 2 or text.shape[1] != dim or len(text) != len(ids)
            or len(text) < 2 or not np.isfinite(text).all()
            or not np.issubdtype(ids.dtype, np.integer) or np.any(ids <= 0)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('text features and explicit foreground valid IDs must align')
    if len(obs) < 2:
        return None
    chosen = select_observations(obs, strategy, seed=seed)
    fused = fuse_features(chosen, weighting)
    direction = fused / max(float(np.linalg.norm(fused)), 1e-8)
    scores = text @ direction / np.maximum(np.linalg.norm(text, axis=1), 1e-8)
    if canonical_features is not None:
        canon = np.asarray(canonical_features)
        if canon.ndim != 2 or canon.shape[1] != dim or not len(canon) or not np.isfinite(canon).all():
            raise ValueError('invalid canonical feature matrix')
        canonical = canon @ direction / np.maximum(np.linalg.norm(canon, axis=1), 1e-8)
        scores = (1 / (1 + np.exp(canonical[:, None] - scores[None, :]))).min(axis=0)
    order = np.argsort(-scores, kind='stable')
    best = int(order[0])
    return {'class_id': int(ids[best]), 'score': float(scores[best]),
            'margin': float(scores[best] - scores[order[1]]),
            'selected_query_ids': [o.source_query_id for o in chosen],
            'selected_frame_ids': [o.frame_id for o in chosen],
            'missing_fields': sorted(set().union(*(o.missing_fields for o in chosen))),
            'feature_space_id': space, 'candidate_count': len(obs),
            'selected_count': len(chosen), 'strategy': strategy, 'weighting': weighting,
            'mode': 'STATIC_OFFLINE_READOUT'}
