"""Batched exact active-set QP and anchored complete-class graph."""

import itertools

import numpy as np
from scipy.special import softmax

from .residuals import probability


def simplex_weights(covariances):
    s = np.asarray(covariances, np.float64)
    if s.ndim != 3 or s.shape[1] != s.shape[2] or not 1 <= s.shape[1] <= 3:
        raise ValueError("expected batch of 1..3 source covariances")
    if not np.isfinite(s).all() or not np.allclose(s, s.swapaxes(1, 2), atol=1e-12, rtol=0):
        raise ValueError("invalid covariance")
    if np.linalg.eigvalsh(s).min() <= 0:
        raise ValueError("covariance must be positive definite")
    n, m = s.shape[:2]
    scale = s.diagonal(axis1=1, axis2=2).mean(1)
    normalized = s / scale[:, None, None]
    best = np.full(n, np.inf)
    weights = np.zeros((n, m))
    for size in range(1, m + 1):
        for active in itertools.combinations(range(m), size):
            ix = np.asarray(active)
            sub = normalized[:, ix[:, None], ix]
            raw = np.linalg.solve(sub, np.ones((n, size, 1)))[..., 0]
            raw /= raw.sum(1, keepdims=True)
            valid = raw.min(1) >= -1e-10
            raw = np.maximum(raw, 0)
            raw /= raw.sum(1, keepdims=True)
            objective = np.einsum('pi,pij,pj->p', raw, sub, raw)
            take = valid & (objective < best)
            weights[take] = 0
            weights[np.ix_(take, ix)] = raw[take]
            best[take] = objective[take]
    return weights, np.einsum('pi,pij,pj->p', weights, s, weights)


def graph_solve(differences, weights, base):
    p = probability(base)
    k = len(p)
    a, b = np.triu_indices(k, 1)
    d, w = np.asarray(differences, np.float64), np.asarray(weights, np.float64)
    if d.shape != a.shape or w.shape != a.shape or not np.isfinite(d).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("complete finite positive-weight graph required")
    w = w / w.mean()
    matrix, rhs = np.eye(k) * k, np.zeros(k)
    np.add.at(matrix, (a, a), w)
    np.add.at(matrix, (b, b), w)
    matrix[a, b] -= w
    matrix[b, a] -= w
    np.add.at(rhs, a, w * d)
    np.add.at(rhs, b, -w * d)
    anchor = np.log(p)
    anchor -= anchor.mean()
    u = np.linalg.solve(matrix, rhs + k * anchor)
    u -= u.mean()
    fitted = u[a] - u[b]
    # Unregularized equal-edge projection quantifies nonintegrable cycle energy.
    unweighted_rhs = np.zeros(k)
    np.add.at(unweighted_rhs, a, d)
    np.add.at(unweighted_rhs, b, -d)
    cycle = d - (unweighted_rhs[a] - unweighted_rhs[b]) / k
    return softmax(u), {"graph_fit_squared_residual": float(np.mean(w * (fitted-d)**2)),
                        "cycle_energy": float(np.mean(cycle**2)), "potential_sum": float(u.sum())}
