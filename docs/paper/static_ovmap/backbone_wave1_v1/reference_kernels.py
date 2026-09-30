"""Small specification fixtures only; NOT the OVI native mapper or a model runner.

The native-order reference mirrors the decision structure inspected in OVI-MAP
common_scannet_nyu.py. These functions verify arithmetic and design distinctions,
not scene-level performance, native state integration, or GPU execution.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Fragment:
    mask: np.ndarray
    group_key: str | None
    score: float
    overlap: float


def mask_key(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=bool)
    body = repr(mask.shape).encode() + np.packbits(mask).tobytes()
    return hashlib.sha256(body).hexdigest()


def _inputs(depth_regions, instances):
    d, p = np.asarray(depth_regions), np.asarray(instances)
    if d.ndim != 2 or d.shape != p.shape:
        raise ValueError('aligned 2D label rasters are required')
    if not (np.issubdtype(d.dtype, np.integer) and np.issubdtype(p.dtype, np.integer)):
        raise ValueError('integer label rasters are required')
    if np.any(d < 0) or np.any(p < 0):
        raise ValueError('negative labels are invalid')
    return d, p


def _spatial(mask):
    loc = np.flatnonzero(mask)
    return (int(loc[0]) if loc.size else np.iinfo(np.int64).max, mask_key(mask))


def canonical_fragments(rows: Iterable[Fragment]):
    return tuple(sorted((mask_key(r.mask), r.group_key or '', r.score, round(r.overlap, 14))
                        for r in rows))


def simultaneous_fusion(depth_regions, instances, *, minimum=100) -> list[Fragment]:
    """Reference implementation of C2, without point lifting or native state."""
    d, p = _inputs(depth_regions, instances)
    fg, bg = [], []
    for depth_id in np.unique(d):
        if depth_id == 0:
            continue
        original = d == depth_id
        area = int(original.sum())
        if area < minimum:
            continue
        cover = np.zeros_like(original)
        for obj_id in np.unique(p[original]):
            if obj_id == 0:
                continue
            obj = p == obj_id
            inter = original & obj
            n, total = int(inter.sum()), int(obj.sum())
            if n > .9 * total and n < .5 * area:
                fg.append(Fragment(inter.copy(), mask_key(obj), 1., n / total))
                cover |= inter
        residual = original & ~cover
        n_res = int(residual.sum())
        if not n_res:
            continue
        choices = []
        for obj_id in np.unique(p[residual]):
            if obj_id > 0:
                obj = p == obj_id
                overlap = int((residual & obj).sum())
                choices.append((-overlap, _spatial(obj), obj_id))
        choices.sort()
        if choices and -choices[0][0] >= .2 * n_res:
            overlap, _, obj_id = choices[0]
            fg.append(Fragment(residual.copy(), mask_key(p == obj_id), 1., -overlap / n_res))
        else:
            bg.append(Fragment(residual.copy(), None, .5, float((residual & (p == 0)).sum()) / n_res))
    return sorted(fg, key=lambda r: _spatial(r.mask)) + sorted(bg, key=lambda r: _spatial(r.mask))


def native_order_reference(depth_regions, instances, *, reverse=False, minimum=100) -> list[Fragment]:
    """Mirror native mutation ordering for a counterexample, not a replacement."""
    d, p = _inputs(depth_regions, instances)
    fg, bg = [], []
    for depth_id in np.unique(d):
        if depth_id == 0:
            continue
        mask = d == depth_id
        area = int(mask.sum())
        if area < minimum:
            continue
        counts = Counter(p[mask].tolist())
        visit = list(counts)
        if reverse:
            visit.reverse()
        max_area, max_id = 0, 0
        for obj_id in visit:
            if obj_id == 0:
                continue
            n = counts[obj_id]
            obj = p == obj_id
            total = int(obj.sum())
            if n > .9 * total and n < .5 * area:
                inter = mask & obj
                fg.append(Fragment(inter.copy(), mask_key(obj), 1., n / total))
                area -= n
                mask = mask & ~inter
            elif n > max_area:
                max_area, max_id = n, obj_id
        if area == 0:
            continue
        if max_area >= .2 * area:
            fg.append(Fragment(mask.copy(), mask_key(p == max_id), 1., max_area / area))
        else:
            bg.append(Fragment(mask.copy(), None, .5, counts[0] / area))
    return fg + bg


def positive_assignment(benefit, eligible):
    b, ok = np.asarray(benefit, float), np.asarray(eligible, bool)
    if b.ndim != 2 or b.shape != ok.shape or not np.isfinite(b).all():
        raise ValueError('finite aligned benefit and eligibility matrices required')
    n, m = b.shape
    if not n or not m:
        return set()
    scores = np.zeros((n, m + n), dtype=np.float64)
    scores[:, :m] = np.where(ok, b, -1e6)
    rr, cc = linear_sum_assignment(-scores)
    return {(int(r), int(c)) for r, c in zip(rr, cc)
            if c < m and ok[r, c] and b[r, c] > 0}


def directional_assignments(intersections, local_areas, global_areas, *, min_count=100, threshold=.2):
    inter = np.asarray(intersections, float)
    local, glob = np.asarray(local_areas, float), np.asarray(global_areas, float)
    if inter.shape != (len(local), len(glob)) or inter.ndim != 2:
        raise ValueError('intersection shape mismatch')
    if not all(np.isfinite(x).all() for x in (inter, local, glob)):
        raise ValueError('non-finite input')
    if np.any(inter < 0) or np.any(local < 0) or np.any(glob < 0):
        raise ValueError('negative input')
    if np.any(inter.sum(1) > local + 1e-9) or np.any(inter.sum(0) > glob + 1e-9):
        raise ValueError('intersection counts exceed disjoint support areas')
    f = np.divide(inter, local[:, None], out=np.zeros_like(inter), where=local[:, None] > 0)
    r = np.divide(inter.T, glob[:, None], out=np.zeros_like(inter.T), where=glob[:, None] > 0)
    forward = positive_assignment(f-threshold, (inter >= min_count) & (f > threshold))
    backward = {(j, i) for i, j in positive_assignment(r-threshold, (inter.T >= min_count) & (r > threshold))}
    return {'forward': forward, 'backward': backward, 'mutual': forward & backward, 'F': f, 'R': r}


def native_eligibility(count, size, min_count, ratio, *, enable_ratio):
    if size <= 0:
        return False
    return bool(count > min_count and (not enable_ratio or count > int(ratio * size)))


def pinned_forward_indices(start, maximum, num_frames):
    """Inclusive bounds in the inspected predictor; this does not run SAM2."""
    if start < 0 or maximum < 0 or start >= num_frames:
        raise ValueError('invalid frame range')
    return list(range(start, min(start + maximum, num_frames - 1) + 1))


def choose_development(rows: list[Mapping], *, ap_band_pp=.05, miou_band_pp=.1, ap50_band_pp=.1):
    """Rows contain fraction-scale metrics, already pooled over the whole cohort."""
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique candidates required')
    kept = list(rows)
    for key, band in [('apall', ap_band_pp), ('miou', miou_band_pp), ('ap50', ap50_band_pp)]:
        if any(r[key] is None or not np.isfinite(r[key]) for r in kept):
            raise ValueError('undefined selection metric')
        best = max(r[key] for r in kept)
        kept = [r for r in kept if 100. * (best - r[key]) <= band + 1e-10]
    return min(kept, key=lambda r: (r['added_visual_encodings'], r['median_seconds'], r['changed_blocks'], r['id']))['id']
