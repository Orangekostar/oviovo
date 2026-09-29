"""Observation-support Gram covariance proxy; compatible spaces only."""

import hashlib

import numpy as np

from .pair_graph import graph_solve, simplex_weights


def support_kernel(masks, blocks):
    if len(masks) != len(blocks):
        raise ValueError("misaligned support blocks")
    kernel = np.eye(len(masks))
    for i, mask in enumerate(masks):
        if mask is None:
            continue
        a = np.asarray(mask, bool)
        if not a.any():
            raise ValueError("empty observation support")
        for j in range(i):
            if masks[j] is None or blocks[i] != blocks[j]:
                continue
            b = np.asarray(masks[j], bool)
            if a.shape != b.shape or not b.any():
                raise ValueError("same-image support geometry mismatch")
            kernel[i, j] = kernel[j, i] = np.count_nonzero(a & b) / np.sqrt(float(a.sum()) * b.sum())
    return kernel


def sensitivity(features, areas, delta, temperature):
    f, alpha, delta = np.asarray(features, np.float64), np.asarray(areas, np.float64), np.asarray(delta, np.float64)
    if f.ndim != 2 or alpha.shape != (len(f),) or delta.shape != (f.shape[1],) or np.any(alpha <= 0) or temperature <= 0:
        raise ValueError("invalid sensitivity inputs")
    alpha = alpha / alpha.sum()
    aggregate = alpha @ f
    norm = np.linalg.norm(aggregate)
    if norm <= 1e-12:
        raise ValueError("undefined aggregate direction")
    v = aggregate / norm
    return alpha[:, None] * (delta - v * (v @ delta))[None] / (temperature * norm)


def collapse_sources(sources):
    """Only exact computation AND calibrated evidence aliases collapse."""
    seen, result = {}, []
    for source in sources:
        key = source['computation_identity']
        if key in seen:
            other = seen[key]
            for field in ('scores', 'features', 'areas', 'text'):
                if not np.array_equal(source[field], other[field]):
                    raise ValueError("conflicting exact-source alias")
            if source['temperature'] != other['temperature']:
                raise ValueError("alias calibrated scores differ")
        else:
            seen[key] = source
            result.append(source)
    return result


def covariance_pairs(sources, kernel, chunk=256):
    sources = collapse_sources(sources)
    k = len(sources[0]['scores'])
    first, second = np.triu_indices(k, 1)
    m = len(sources)
    offsets = np.cumsum([0] + [len(s['features']) for s in sources])
    if kernel.shape != (offsets[-1], offsets[-1]):
        raise ValueError("kernel must follow collapsed-source atom order")
    prepared = []
    for source in sources:
        f = np.asarray(source['features'], np.float64)
        area = np.asarray(source['areas'], np.float64)
        alpha = area / area.sum()
        aggregate = alpha @ f
        norm = np.linalg.norm(aggregate)
        if norm <= 1e-12 or np.any(area <= 0):
            raise ValueError("invalid source aggregate")
        v = aggregate / norm
        text = np.asarray(source['text'], np.float64)
        projected = text - (text @ v)[:, None] * v[None]
        loading_scale = alpha * np.linalg.norm(f, axis=1) / (np.sqrt(f.shape[1]) * source['temperature'] * norm)
        prepared.append((projected, loading_scale))
    result = np.zeros((len(first), m, m))
    for start in range(0, len(first), chunk):
        stop = start + chunk
        deltas = [p[first[start:stop]] - p[second[start:stop]] for p, _ in prepared]
        for i in range(m):
            for j in range(i + 1):
                if sources[i]['space'] != sources[j]['space']:
                    continue
                block = kernel[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]]
                coefficient = prepared[i][1] @ block @ prepared[j][1]
                value = coefficient * np.einsum('pd,pd->p', deltas[i], deltas[j])
                result[start:stop, i, j] = value
                result[start:stop, j, i] = value
    floor = .1 * np.maximum(result.diagonal(axis1=1, axis2=2).mean(1), 1e-12)
    result += floor[:, None, None] * np.eye(m)
    z = np.array([np.asarray(s['scores']) / s['temperature'] for s in sources])
    differences = (z[:, first] - z[:, second]).T
    return result, differences, sources


def solve_variants(sources, base, scene, owner):
    sources = collapse_sources(sources)
    masks = [a['mask'] for s in sources for a in s['atoms']]
    blocks = [(a['image'], s['space']) for s in sources for a in s['atoms']]
    kernel = support_kernel(masks, blocks)
    cov, differences, sources = covariance_pairs(sources, kernel)
    diag = np.zeros_like(cov)
    index = np.arange(len(sources))
    diag[:, index, index] = cov[:, index, index]
    shuffled = cov.copy()
    names = [s['name'] for s in sources]
    rho = np.zeros(len(cov))
    if 'N' in names and 'Q' in names:
        n, q = names.index('N'), names.index('Q')
        scale = np.sqrt(cov[:, n, n] * cov[:, q, q])
        rho = cov[:, n, q] / scale
        seed = int.from_bytes(hashlib.sha256(f'1729|{scene}|{owner}'.encode()).digest()[:8], 'big')
        shuffled[:, n, q] = shuffled[:, q, n] = np.random.default_rng(seed).permutation(rho) * scale
    outputs, diagnostics = {}, {}
    for method, matrix in [('PE_D3_DIAGONAL', diag), ('PE_D4_LINEAGE', cov), ('PE_D4_SHUFFLED', shuffled)]:
        weights, variance = simplex_weights(matrix)
        contrasts = np.sum(weights * differences, axis=1)
        edge_weights = 1 / np.maximum(variance, max(1e-6 * np.median(variance), 1e-12))
        outputs[method], graph = graph_solve(contrasts, edge_weights, base)
        active, counts = np.unique(weights > 1e-10, axis=0, return_counts=True)
        diagnostics[method] = {**graph, 'sources': names, 'mean_source_weights': weights.mean(0).tolist(),
            'active_sets': [{'sources': [n for n, yes in zip(names, a) if yes], 'pairs': int(c)} for a, c in zip(active, counts)]}
    return outputs, {'methods': diagnostics, 'atoms': len(masks), 'unknown_support': sum(m is None for m in masks),
                     'shared_atom_pairs': int(np.count_nonzero(np.triu(kernel, 1))),
                     'rho_min': float(rho.min()), 'rho_max': float(rho.max()), 'rho_mean': float(rho.mean()),
                     'nonzero_rho_pairs': int(np.count_nonzero(rho)), 'shuffle_identity': bool(np.ptp(rho) == 0),
                     'D4_equals_D3': bool(np.array_equal(cov, diag))}
