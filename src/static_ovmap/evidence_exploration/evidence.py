"""Geometric support on the original fixed G1 frame; no GT dependencies."""

import math

import numpy as np


def occupancy_trigger(occupancy, hard_support):
    w = np.asarray(occupancy, np.float64)
    if not w.size or not np.isfinite(w).all() or np.any((w < 0) | (w > 1)):
        raise ValueError("occupancy requires finite probabilities")
    if isinstance(hard_support, bool) or int(hard_support) != hard_support or hard_support < 0:
        raise ValueError("hard support requires a nonnegative integer")
    mass = float(w.sum())
    if mass <= 0:
        raise ValueError("nonempty bound mask has zero area mass")
    purity = float((w * w).sum() / (mass + 1e-8))
    return bool(hard_support < 4 or purity < .5), purity


def competing_regions(target, visible_owners, bbox):
    target, owners = np.asarray(target), np.asarray(visible_owners)
    if target.dtype != bool or target.shape != owners.shape or target.ndim != 2:
        raise ValueError("controls require aligned original target and categorical owners")
    x0, y0, x1, y1 = map(int, bbox)
    height, width = target.shape
    margin = max(8, math.ceil(.1 * max(x1 - x0, y1 - y0)))
    box = np.zeros(target.shape, bool)
    box[max(0, y0-margin):min(height, y1+margin), max(0, x0-margin):min(width, x1+margin)] = True
    available = box & ~target & (owners > 0)
    ids, counts = np.unique(owners[available], return_counts=True)
    choices = sorted([(int(n), int(i)) for i, n in zip(ids, counts, strict=True) if n >= 100],
                     key=lambda row: (-row[0], row[1]))[:2]
    return [{"owner": i, "pixels": n, "mask": available & (owners == i)} for n, i in choices]
