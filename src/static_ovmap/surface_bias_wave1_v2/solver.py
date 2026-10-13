"""NumPy reference for a proposed, NOT yet validated, FG2 value correction.

No repository integration, encoder calls, training, labels, or map evaluation.
Run: python reference_bias_solver.py --self-test
The production adapter must preserve the original FG2 key/quality branch.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SolverConfig:
    min_shared: int = 8
    huber_delta: float = 1.345
    max_iter: int = 10
    tolerance: float = 1e-8


@dataclass
class BiasEstimate:
    bias: np.ndarray
    status: str
    active_views: list[int]
    edges: list[dict[str, Any]]
    components: list[list[int]]


def _validate(raw: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw, valid = np.asarray(raw), np.asarray(valid)
    if raw.ndim != 3 or min(raw.shape) < 1:
        raise ValueError('raw must be a nonempty [V,S,C] array')
    if raw.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise TypeError('raw must be float32 or float64')
    if valid.shape != raw.shape[:2] or valid.dtype != np.bool_:
        raise ValueError('valid must be a Boolean [V,S] array')
    if not np.isfinite(raw[valid]).all():
        raise ValueError('Nonfinite VALID features are an input error, not a fallback')
    return raw, valid


def huber_location(points: np.ndarray, config: SolverConfig) -> np.ndarray:
    """Vector Huber IRLS with one robust scale, fixed from the initial estimate.

    This is an approximate numerical estimator, not a proof of robustness for
    the user's data. It is neither coordinatewise clipping nor unit-sphere
    feature averaging. The vector feature scale is preserved.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Finite nonempty [N,C] points required')
    m = np.median(p, axis=0)
    r = np.linalg.norm(p - m, axis=1)
    eps = 1e-12 * max(1.0, float(np.linalg.norm(m)))
    scale = max(float(np.median(r)), eps)
    cutoff = config.huber_delta * scale
    for _ in range(config.max_iter):
        r = np.linalg.norm(p - m, axis=1)
        weights = np.minimum(1.0, cutoff / np.maximum(r, eps))
        new_m = (weights[:, None] * p).sum(axis=0) / weights.sum()
        converged = np.linalg.norm(new_m - m) <= config.tolerance * max(1.0, np.linalg.norm(m))
        m = new_m
        if converged:
            break
    return m


def estimate_view_bias(
    raw: np.ndarray,
    valid: np.ndarray,
    *,
    fit_sites: np.ndarray | None = None,
    config: SolverConfig = SolverConfig(),
) -> BiasEstimate:
    """Estimate RELATIVE per-view bias from shared physical site IDs.

    fit_sites controls diagnostic fitting only. Evaluation sites must never
    enter these edges during held-out-site validation. Full-support inference
    may use all observed sites. A disconnected active view graph produces
    zero correction for the entire object, not arbitrary component offsets.
    """
    raw, valid = _validate(raw, valid)
    if config.min_shared < 1 or config.max_iter < 1 or config.huber_delta <= 0 or config.tolerance <= 0:
        raise ValueError('Invalid solver configuration')
    v_count, s_count, channels = raw.shape
    if fit_sites is None:
        fit = np.ones(s_count, dtype=bool)
    else:
        fit = np.asarray(fit_sites)
        if fit.shape != (s_count,) or fit.dtype != np.bool_:
            raise ValueError('fit_sites must be a Boolean [S] array')
    active = np.flatnonzero(valid.any(axis=1)).tolist()
    bias = np.zeros((v_count, channels), dtype=np.float64)
    if len(active) < 2:
        return BiasEstimate(bias, 'FALLBACK_FEWER_THAN_TWO_VIEWS', active, [], [[v] for v in active])
    edges: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    adjacency = {v: set() for v in active}
    for k, v in enumerate(active):
        for u in active[:k]:
            sites = np.flatnonzero(valid[v] & valid[u] & fit)
            if len(sites) < config.min_shared:
                continue
            diffs = raw[v, sites].astype(np.float64) - raw[u, sites].astype(np.float64)
            delta = huber_location(diffs, config)
            edges.append({'v': v, 'u': u, 'n_shared': int(len(sites)),
                          'weight': float(len(sites)), 'delta_norm': float(np.linalg.norm(delta))})
            vectors.append(delta)
            adjacency[v].add(u)
            adjacency[u].add(v)
    remaining = set(active)
    components: list[list[int]] = []
    while remaining:
        todo = [min(remaining)]
        component: list[int] = []
        while todo:
            v = todo.pop()
            if v not in remaining:
                continue
            remaining.remove(v)
            component.append(v)
            todo.extend(sorted(adjacency[v] & remaining, reverse=True))
        components.append(sorted(component))
    if len(components) != 1:
        return BiasEstimate(bias, 'FALLBACK_DISCONNECTED_GRAPH', active, edges, components)
    indices = {v: i for i, v in enumerate(active)}
    n = len(active)
    lap = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros((n, channels), dtype=np.float64)
    for edge, d in zip(edges, vectors):
        i, j, w = indices[edge['v']], indices[edge['u']], edge['weight']
        lap[i, i] += w
        lap[j, j] += w
        lap[i, j] -= w
        lap[j, i] -= w
        rhs[i] += w * d
        rhs[j] -= w * d
    # Equal-view zero-mean gauge. This does NOT identify a shared absolute bias.
    kkt = np.zeros((n + 1, n + 1), dtype=np.float64)
    kkt[:n, :n] = lap
    kkt[n, :n] = 1.0
    kkt[:n, n] = 1.0
    target = np.zeros((n + 1, channels), dtype=np.float64)
    target[:n] = rhs
    try:
        solution = np.linalg.solve(kkt, target)[:n]
    except np.linalg.LinAlgError:
        return BiasEstimate(bias, 'FALLBACK_SINGULAR_SOLVE', active, edges, components)
    if not np.isfinite(solution).all():
        return BiasEstimate(bias, 'FALLBACK_NONFINITE_SOLVE', active, edges, components)
    solution -= solution.mean(axis=0, keepdims=True)
    bias[active] = solution
    return BiasEstimate(bias, 'OK', active, edges, components)


def apply_bias(raw: np.ndarray, valid: np.ndarray, estimate: BiasEstimate, strength: float) -> np.ndarray:
    """Correct only valid RAW VALUE vectors; preserve dtype and invalid padding.

    Return to original FG2, not FC8, when estimate.status is a fallback.
    In production use this array only for FG2's value path. Keep local_unit,
    metadata, FC8, weights and learned queries unchanged in the first wave.
    """
    raw, valid = _validate(raw, valid)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError('strength must be finite and in [0,1]')
    if estimate.bias.shape != (raw.shape[0], raw.shape[2]):
        raise ValueError('Bias shape does not match raw')
    out = raw.copy()
    if strength == 0.0 or estimate.status != 'OK':
        return out
    corrected = raw.astype(np.float64) - strength * estimate.bias[:, None, :]
    out[valid] = corrected[valid].astype(raw.dtype)
    if not np.isfinite(out[valid]).all():
        raise FloatingPointError('Correction became nonfinite after dtype conversion')
    return out

def center_bias(raw,valid,fit_sites=None):
    raw,valid=_validate(raw,valid)
    fit=np.ones(raw.shape[1],bool) if fit_sites is None else np.asarray(fit_sites,bool)
    active=np.flatnonzero(valid.any(1)).tolist()
    bias=np.zeros((raw.shape[0],raw.shape[-1]),np.float64)
    if len(active)<2: return BiasEstimate(bias,'FALLBACK_FEWER_THAN_TWO_VIEWS',active,[],[])
    if any(not (valid[v]&fit).any() for v in active): return BiasEstimate(bias,'FALLBACK_NO_FIT_SUPPORT',active,[],[])
    means=np.stack([raw[v,valid[v]&fit].astype(np.float64).mean(0) for v in active]); means-=means.mean(0)
    bias[active]=means
    return BiasEstimate(bias,'OK',active,[],[active])

def correction(raw,valid,method,strength):
    raw,valid=_validate(raw,valid)
    if strength not in (0,.125,.25,.5): raise ValueError('Frozen lambda grid required')
    if method=='M1_CORRESPONDENCE_GRAPH': est=estimate_view_bias(raw,valid)
    elif method=='C1_FULL_VIEW_CENTER': est=center_bias(raw,valid)
    elif method=='C2_SITE_HUBER_CONSENSUS':
        out=raw.copy(); count=0
        for s in range(raw.shape[1]):
            views=np.flatnonzero(valid[:,s])
            if len(views)<2: continue
            mu=huber_location(raw[views,s],SolverConfig())
            out[views,s]=((1-strength)*raw[views,s].astype(np.float64)+strength*mu).astype(raw.dtype); count+=1
        return out,dict(status='OK' if count else 'FALLBACK_NO_SHARED_SITE',active_views=np.flatnonzero(valid.any(1)).tolist(),consensus_sites=count)
    else: raise ValueError('Unknown fixed method: '+method)
    return apply_bias(raw,valid,est,strength),dict(status=est.status,active_views=est.active_views,edges=est.edges,components=est.components,bias_norm=float(np.linalg.norm(est.bias)),view_bias_norms=np.linalg.norm(est.bias,axis=1).tolist())
