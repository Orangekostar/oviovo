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


def self_test() -> dict[str, Any]:
    rng = np.random.default_rng(17)
    mu = rng.normal(size=(32, 12))
    b = rng.normal(size=(4, 12))
    b -= b.mean(axis=0)
    x = mu[None] + b[:, None]
    valid = np.ones(x.shape[:2], dtype=bool)
    passed: list[str] = []
    estimate = estimate_view_bias(x, valid)
    assert estimate.status == 'OK'
    np.testing.assert_allclose(estimate.bias, b, atol=1e-10, rtol=0)
    passed.append('known_relative_bias_recovery')
    corrected = apply_bias(x, valid, estimate, 1.0)
    np.testing.assert_allclose(corrected, np.broadcast_to(mu, x.shape), atol=1e-10, rtol=0)
    passed.append('additive_model_correction')
    fit = np.arange(32) < 16
    fitted = estimate_view_bias(x, valid, fit_sites=fit)
    e = (x[0, ~fit] - x[1, ~fit]) - (fitted.bias[0] - fitted.bias[1])
    assert np.linalg.norm(e) < 1e-9
    passed.append('heldout_sites_not_used_for_fit')
    common = rng.normal(size=12)
    absolute = estimate_view_bias(x + common, valid)
    np.testing.assert_allclose(absolute.bias, estimate.bias, atol=1e-10, rtol=0)
    np.testing.assert_allclose(apply_bias(x + common, valid, absolute, 1.0),
                               np.broadcast_to(mu + common, x.shape), atol=1e-10, rtol=0)
    passed.append('common_absolute_bias_remains_unidentified')
    x32 = x.astype(np.float32)
    identical = apply_bias(x32, valid, estimate, 0.0)
    assert identical.dtype == x32.dtype and np.array_equal(identical, x32)
    passed.append('lambda_zero_exact_identity')
    disconnected = np.zeros_like(valid)
    disconnected[:2, :16] = True
    disconnected[2:, 16:] = True
    failed = estimate_view_bias(x, disconnected)
    assert failed.status == 'FALLBACK_DISCONNECTED_GRAPH'
    assert np.array_equal(apply_bias(x, disconnected, failed, 0.5), x)
    passed.append('disconnected_graph_exact_fallback')
    missing = valid.copy()
    missing[0, :4] = False
    missing[1, 4:8] = False
    missing[2, 8:12] = False
    recovered = estimate_view_bias(x, missing)
    np.testing.assert_allclose(recovered.bias, b, atol=1e-10, rtol=0)
    passed.append('connected_missing_visibility')
    noisy_padding = x.copy()
    noisy_padding[~missing] = np.nan
    recovered = estimate_view_bias(noisy_padding, missing)
    np.testing.assert_allclose(recovered.bias, b, atol=1e-10, rtol=0)
    passed.append('invalid_padding_excluded')
    noisy_valid = x.copy()
    noisy_valid[0, 0, 0] = np.nan
    try:
        estimate_view_bias(noisy_valid, valid)
    except ValueError:
        passed.append('valid_nonfinite_rejected')
    else:
        raise AssertionError('Expected input validation failure')
    # A mandatory negative sanity case: perfect coverage + equal weights
    # cancels zero-mean correction in pooled raw features, even before phi.
    np.testing.assert_allclose(x.mean(axis=0), corrected.mean(axis=0), atol=1e-10, rtol=0)
    passed.append('balanced_pooling_cancellation_is_expected')
    permutation = rng.permutation(mu.shape[0])
    direct = (x[0] - x[1]).mean(axis=0)
    shuffled = (x[0] - x[1, permutation]).mean(axis=0)
    np.testing.assert_allclose(direct, shuffled, atol=1e-10, rtol=0)
    passed.append('mean_difference_pair_shuffle_equivalence_is_expected')
    p = rng.normal(scale=0.01, size=(30, 12)) + 2.0
    p[-3:] += 100.0
    huber = huber_location(p, SolverConfig())
    assert np.linalg.norm(huber - 2.0) < np.linalg.norm(p.mean(axis=0) - 2.0)
    passed.append('synthetic_outlier_sanity_not_real_data_evidence')
    return {'status': 'SYNTHETIC_SELF_TESTS_PASSED', 'count': len(passed), 'tests': passed,
            'real_FC_used': False, 'repository_integration_tested': False,
            'real_data_evaluated': False, 'optimizer_updates': 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--input', type=Path, help='NPZ containing local_raw, local_valid; optional fit_sites')
    parser.add_argument('--output', type=Path, help='New NPZ path; existing files are never overwritten')
    parser.add_argument('--strength', type=float, default=0.25)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, ensure_ascii=False))
        return
    if args.input is None or args.output is None:
        parser.error('Use --self-test, or supply --input and --output')
    with np.load(args.input, allow_pickle=False) as arrays:
        allowed = {'local_raw', 'local_valid', 'fit_sites'}
        if not {'local_raw', 'local_valid'} <= set(arrays.files) or not set(arrays.files) <= allowed:
            raise ValueError('Unexpected input keys, or missing required tensors')
        raw, valid = arrays['local_raw'].copy(), arrays['local_valid'].copy()
        fit = arrays['fit_sites'].copy() if 'fit_sites' in arrays.files else None
    estimate = estimate_view_bias(raw, valid, fit_sites=fit)
    corrected = apply_bias(raw, valid, estimate, args.strength)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'status': estimate.status, 'active_views': estimate.active_views,
                'edges': estimate.edges, 'components': estimate.components,
                'strength': args.strength, 'solver_config': asdict(SolverConfig())}
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, corrected_raw=corrected, bias=estimate.bias,
                            audit_json=np.array(json.dumps(metadata)))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, TypeError, FloatingPointError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
