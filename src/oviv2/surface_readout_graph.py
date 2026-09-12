"""CROVE sparse Potts readout; S2 costs are not recovered class posteriors."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.special import softmax


def surface_patches(xyz, normals, triangles, *, visit_ids=None, patch_size=0.02):
    """Weld compatible duplicate rows, then componentize actual in-cell topology.

    Patches are not simple voxel buckets: disconnected surfaces remain separate.
    Graph edges derive only from welded source triangle edges, never global kNN.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int64)
    visits = (
        np.zeros(len(xyz), np.int32)
        if visit_ids is None
        else np.asarray(visit_ids, np.int32)
    )
    norm = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    normal_bins = np.rint(norm * 4).astype(np.int32)
    keys = np.column_stack((xyz.view(np.int32), normal_bins, visits))
    _, first, inverse, counts = np.unique(
        keys, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    del keys
    samples = xyz[first]
    sample_normals = norm[first]
    sample_visits = visits[first]
    cells = np.floor(samples / patch_size).astype(np.int32)
    faces = inverse[triangles]
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]))
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    a, b = edges.T
    compatible = (
        (a != b)
        & (sample_visits[a] == sample_visits[b])
        & (np.einsum("ij,ij->i", sample_normals[a], sample_normals[b]) >= 0.95)
    )
    edges = edges[compatible]
    a, b = edges.T
    inside = np.all(cells[a] == cells[b], axis=1)
    local = sparse.csr_matrix(
        (np.ones(int(inside.sum()), np.uint8), (a[inside], b[inside])),
        shape=(len(samples), len(samples)),
    )
    node_count, sample_patch = connected_components(local, directed=False)
    source_patch = sample_patch[inverse]
    mass = np.bincount(sample_patch, minlength=node_count)
    centers = np.column_stack(
        [
            np.bincount(sample_patch, weights=samples[:, i], minlength=node_count)
            / mass
            for i in range(3)
        ]
    )
    node_normals = np.column_stack(
        [
            np.bincount(
                sample_patch, weights=sample_normals[:, i], minlength=node_count
            )
            / mass
            for i in range(3)
        ]
    )
    graph_edges = np.sort(sample_patch[edges], axis=1)
    graph_edges = np.unique(graph_edges[graph_edges[:, 0] != graph_edges[:, 1]], axis=0)
    a, b = graph_edges.T
    distance = np.linalg.norm(centers[a] - centers[b], axis=1)
    weights = np.exp(-np.square(distance / 0.03)).astype(np.float32)
    matrix = sparse.csr_matrix(
        (np.r_[weights, weights], (np.r_[a, b], np.r_[b, a])),
        shape=(node_count, node_count),
    )
    return {
        "source_patch": source_patch,
        "physical_weight": (1.0 / counts[inverse]).astype(np.float32),
        "centers": centers.astype(np.float32),
        "normals": node_normals.astype(np.float32),
        "adjacency": matrix,
        "physical_samples": len(samples),
    }


def s2_pseudo_unary(patch_ids, semantic_ids, confidence, physical_weights, class_ids):
    """S2_PSEUDO_UNARY_V1; no second multiplication by support reliability."""
    patch_ids = np.asarray(patch_ids)
    ids = np.asarray(semantic_ids)
    g = np.clip(np.asarray(confidence, dtype=np.float64), 0, 1)
    weights = np.asarray(physical_weights, dtype=np.float64)
    classes = np.asarray(class_ids)
    if not (ids.shape == g.shape == weights.shape == patch_ids.shape):
        raise ValueError("one aligned unary input per source row required")
    if (
        not np.isfinite(g).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("unary inputs must be finite and nonnegative")
    if (
        np.any(patch_ids < 0)
        or not len(classes)
        or len(np.unique(classes)) != len(classes)
    ):
        raise ValueError("invalid patch/class IDs")
    n = int(patch_ids.max()) + 1 if len(patch_ids) else 0
    known = np.isin(ids, classes)
    denom = np.bincount(patch_ids[known], weights=weights[known], minlength=n)
    gsum = np.bincount(patch_ids[known], weights=(weights * g)[known], minlength=n)
    unary = np.broadcast_to(gsum[:, None], (n, len(classes))).copy()
    volume = np.zeros_like(unary)
    for j, c in enumerate(classes):
        selected = ids == c
        unary[:, j] -= np.bincount(
            patch_ids[selected], weights=(weights * g)[selected], minlength=n
        )
        volume[:, j] = np.bincount(
            patch_ids[selected], weights=weights[selected], minlength=n
        )
    unary /= np.maximum(denom[:, None], 1e-12)
    # Most physical support, then smallest class ID, is fixed before graph inference.
    order = np.argsort(classes, kind="stable")
    minimizers = np.isclose(unary, unary.min(axis=1, keepdims=True), rtol=0, atol=1e-7)
    ranked_volume = np.where(minimizers, volume, -1.0)
    tie = order[np.argmax(ranked_volume[:, order], axis=1)]
    return unary.astype(np.float32), (denom > 0) & (gsum > 0), tie


def posterior_unary(patch_ids, owner_ids, physical_weights, feature_owners, posterior):
    """Physical-mass mean of real class posteriors; unit reliability when observed.

    No class evidence is invented for missing owners. View reliability has already
    determined M1 feature aggregation; it is not multiplied into costs again.
    """
    patch_ids, owners = np.asarray(patch_ids), np.asarray(owner_ids)
    weights = np.asarray(physical_weights, dtype=np.float64)
    keys = np.asarray(feature_owners)
    p = np.asarray(posterior, dtype=np.float64)
    if patch_ids.shape != owners.shape or weights.shape != owners.shape:
        raise ValueError("aligned source rows required")
    if p.ndim != 2 or p.shape[0] != len(keys) or not p.shape[1]:
        raise ValueError("one full class distribution per feature owner required")
    if len(np.unique(keys)) != len(keys) or np.any(patch_ids < 0):
        raise ValueError("unique owners and nonnegative patch IDs required")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1):
        raise ValueError("actual normalized posteriors required")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("finite nonnegative physical mass required")
    n = int(patch_ids.max()) + 1 if len(patch_ids) else 0
    order = np.argsort(keys)
    columns = np.searchsorted(keys[order], owners)
    supported = np.isin(owners, keys) & (weights > 0)
    counts = sparse.csr_matrix(
        (weights[supported], (patch_ids[supported], columns[supported])),
        shape=(n, len(keys)),
    )
    mass = np.asarray(counts.sum(1)).ravel()
    mean = (counts @ p[order]) / np.maximum(mass[:, None], 1e-12)
    unary = -np.log(np.maximum(mean, 1e-6))
    unary[mass == 0] = 0
    baseline = np.eye(p.shape[1])[np.argmax(p[order], axis=1)]
    baseline_mass = counts @ baseline
    minimizers = np.isclose(unary, unary.min(axis=1, keepdims=True), rtol=0, atol=1e-7)
    tie = np.argmax(np.where(minimizers, baseline_mass, -1.0), axis=1)
    return unary.astype(np.float32), mass > 0, tie


def graph_labels(
    unary, adjacency, valid, baseline_tie, *, strength=0.2, iterations=5, damping=0.5
):
    """Damped sparse mean-field on symmetric normalized nonnegative Potts edges."""
    unary = np.asarray(unary, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    tie = np.asarray(baseline_tie, dtype=np.int64)
    n, c = unary.shape
    if valid.shape != (n,) or tie.shape != (n,) or np.any((tie < 0) | (tie >= c)):
        raise ValueError("node masks/ties do not match unary")
    if (
        strength < 0
        or iterations < 0
        or not 0 < damping <= 1
        or not np.isfinite(unary).all()
    ):
        raise ValueError("invalid graph inference parameters")
    matrix = sparse.csr_matrix(adjacency, dtype=np.float32)
    if (
        matrix.shape != (n, n)
        or np.any(matrix.data < 0)
        or not np.isfinite(matrix.data).all()
    ):
        raise ValueError("nonnegative finite adjacency required")
    delta = matrix - matrix.T
    if delta.nnz and np.max(np.abs(delta.data)) > 1e-6:
        raise ValueError("Potts adjacency must be symmetric")
    unary_best = np.argmin(unary, axis=1)
    tied = np.isclose(unary[np.arange(n), tie], unary.min(axis=1), rtol=0, atol=1e-7)
    unary_best[tied] = tie[tied]
    unary_best[~valid] = -1
    if strength == 0:
        return unary_best
    mask = sparse.diags(valid.astype(np.float32))
    matrix = mask @ matrix @ mask
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    normalizer = sparse.diags(
        np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    )
    matrix = normalizer @ matrix @ normalizer
    q = softmax(-unary, axis=1)
    q[~valid] = 0
    for _ in range(iterations):
        proposal = softmax(-unary + strength * (matrix @ q), axis=1)
        q = (1 - damping) * q + damping * proposal
        q[~valid] = 0
    best = np.argmax(q, axis=1)
    tied = np.isclose(q[np.arange(n), tie], q.max(axis=1), rtol=0, atol=1e-7)
    best[tied] = tie[tied]
    best[~valid] = -1
    return best
