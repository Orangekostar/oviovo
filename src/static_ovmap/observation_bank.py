"""Deterministic selection over a declared offline candidate pool.

The caller must disclose whether the pool is full history or a retained native
subset. This module makes no online or full-history claim.
"""
import numpy as np


STRATEGIES = ('last8', 'random8', 'quality8', 'quality_coverage8', 'all_views')


def quality_scores(observations):
    """Area times geometric mean of available measured quality proxies.

If no proxy is measured, this is explicitly the area-only degradation; absent
fields are not converted to perfect-quality observations or persisted as ones.
"""
    result = []
    for obs in observations:
        measured = [getattr(obs, name) for name in
                    ('depth_valid_ratio', 'mask_quality_proxy', 'sharpness_proxy')
                    if getattr(obs, name) is not None]
        factor = float(np.prod(measured) ** (1 / len(measured))) if measured else 1.
        result.append(obs.visible_area_px * factor)
    return np.asarray(result, dtype=np.float64)


def select_observations(observations, strategy, *, seed=0):
    obs = list(observations)
    if strategy not in STRATEGIES:
        raise ValueError('unknown strategy: ' + strategy)
    n = len(obs)
    if strategy == 'all_views' or n <= 8:
        return obs
    if strategy == 'last8':
        return obs[-8:]
    if strategy == 'random8':
        indices = np.random.default_rng(seed).choice(n, 8, replace=False)
    else:
        quality = quality_scores(obs)
        if strategy == 'quality8':
            indices = np.argsort(quality, kind='stable')[-8:]
        else:
            # Greedy quality-weighted angular facility coverage, O(K*N^2)
            # on object query pools only; never on mesh vertices.
            directions = np.zeros((n, 3))
            known = np.array([o.camera_direction is not None for o in obs])
            if not known.any():
                return select_observations(obs, 'quality8', seed=seed)
            for i in np.flatnonzero(known):
                v = np.asarray(obs[i].camera_direction, dtype=float)
                directions[i] = v / np.linalg.norm(v)
            similarity = np.maximum(0., directions @ directions.T)
            similarity[~known, :] = 0.
            similarity[:, ~known] = 0.
            np.fill_diagonal(similarity, 1.)
            quality = quality / max(float(quality.max()), 1e-12)
            coverage = np.zeros(n)
            indices = []
            for _ in range(8):
                gain = ((np.maximum(coverage[:, None], similarity) - coverage[:, None])
                        * quality[:, None]).sum(axis=0) + .25 * quality
                gain[indices] = -np.inf
                chosen = int(np.argmax(gain))
                indices.append(chosen)
                coverage = np.maximum(coverage, similarity[:, chosen])
    # Fusion order is native order even if the selection was greedy/random.
    return [obs[int(i)] for i in sorted(indices)]
