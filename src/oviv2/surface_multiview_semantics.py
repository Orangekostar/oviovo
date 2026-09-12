"""Deterministic independent-view selection and feature aggregation."""

from __future__ import annotations

import numpy as np


def owner_posteriors_to_rows(owner_ids, feature_owners, posterior, class_ids):
    """Exact owner lookup, preserving visit offsets and unsupported source rows."""
    owners, keys = np.asarray(owner_ids), np.asarray(feature_owners)
    p, classes = np.asarray(posterior), np.asarray(class_ids)
    if owners.ndim != 1 or keys.ndim != 1 or p.shape != (len(keys), len(classes)):
        raise ValueError("aligned owners and complete class posteriors required")
    if len(np.unique(keys)) != len(keys) or len(np.unique(classes)) != len(classes):
        raise ValueError("unique owner/class keys required")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1):
        raise ValueError("normalized real class posteriors required")
    ids = np.zeros(len(owners), np.int32)
    confidence = np.zeros(len(owners), np.float32)
    covered = np.isin(owners, keys)
    if len(keys):
        order = np.argsort(keys)
        columns = np.searchsorted(keys[order], owners[covered])
        ids[covered] = classes[p.argmax(1)[order][columns]]
        confidence[covered] = p.max(1)[order][columns]
    return ids, confidence, covered


def diverse_views(directions, visibility, *, k=4):
    """Visibility seed, then angular farthest-first; visibility breaks ties."""
    directions = np.asarray(directions, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=np.float64)
    if directions.shape != (len(visibility), 3) or k < 1:
        raise ValueError("aligned camera directions and positive budget required")
    if not np.isfinite(directions).all() or not np.isfinite(visibility).all():
        raise ValueError("finite geometric evidence required")
    order = np.argsort(-visibility, kind="stable")
    if not len(order):
        return np.empty(0, np.int64)
    norm = np.linalg.norm(directions, axis=1, keepdims=True)
    unit = directions / np.maximum(norm, 1e-12)
    selected = [int(order[0])]
    while len(selected) < min(k, len(order)):
        remaining = order[~np.isin(order, selected)]
        distance = (1 - np.clip(unit[remaining] @ unit[selected].T, -1, 1)).min(1)
        selected.append(int(remaining[np.argmax(distance)]))
    return np.asarray(selected, np.int64)


def aggregate_views(features, weights):
    """Normalize each cached six-crop mean before weighting independent views.

    None denotes no observed feature; callers must preserve their stated fallback.
    The cache has already averaged crops, so this does not recover per-crop norms.
    """
    features = np.asarray(features, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if features.ndim != 2 or weights.shape != (len(features),):
        raise ValueError("aligned view features and weights required")
    if (
        not np.isfinite(features).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("finite features and nonnegative weights required")
    norm = np.linalg.norm(features, axis=1)
    valid = (weights > 0) & (norm > 1e-12)
    if not valid.any():
        return None
    z = ((features[valid] / norm[valid, None]) * weights[valid, None]).sum(0)
    length = np.linalg.norm(z)
    return z / length if length > 1e-12 else None
