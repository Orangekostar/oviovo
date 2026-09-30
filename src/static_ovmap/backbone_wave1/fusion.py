"""Simultaneous fusion on immutable depth support and native point conversion."""

from dataclasses import dataclass
import hashlib

import numpy as np


def mask_key(mask):
    mask = np.asarray(mask, dtype=bool)
    return hashlib.sha256(repr(mask.shape).encode() + np.packbits(mask).tobytes()).hexdigest()


def spatial_key(mask):
    support = np.flatnonzero(mask)
    return (int(support[0]) if support.size else np.iinfo(np.int64).max, mask_key(mask))


@dataclass(frozen=True)
class Fragment:
    mask: np.ndarray
    input_group: int
    group_key: str | None
    is_thing: bool
    score: float
    overlap_ratio: float


def canonical_fragments(rows):
    return tuple(sorted((mask_key(r.mask), r.group_key or "", r.is_thing,
                         r.score, round(r.overlap_ratio, 14)) for r in rows))


def simultaneous_fusion(depth_regions, instances):
    depth, pano = np.asarray(depth_regions), np.asarray(instances)
    if depth.ndim != 2 or depth.shape != pano.shape:
        raise ValueError("aligned 2D label rasters required")
    if any(not np.issubdtype(x.dtype, np.integer) or np.any(x < 0) for x in (depth, pano)):
        raise ValueError("nonnegative integer labels required")
    objects = {int(k): pano == k for k in np.unique(pano) if k > 0}
    keys = {k: mask_key(mask) for k, mask in objects.items()}
    areas = {k: int(mask.sum()) for k, mask in objects.items()}
    foreground, background = [], []
    for depth_id in np.unique(depth):
        if depth_id == 0:
            continue
        original = depth == depth_id
        area = int(original.sum())
        if area < 100:
            continue
        carved = np.zeros_like(original)
        for group in np.unique(pano[original]):
            if group == 0:
                continue
            group = int(group)
            intersection = original & objects[group]
            count = int(intersection.sum())
            if count > .9 * areas[group] and count < .5 * area:
                foreground.append(Fragment(intersection, group, keys[group], True, 1., count / areas[group]))
                carved |= intersection
        residual = original & ~carved
        residual_area = int(residual.sum())
        if not residual_area:
            continue
        candidates = sorted((-int((residual & objects[int(k)]).sum()),
                             spatial_key(objects[int(k)]), int(k))
                            for k in np.unique(pano[residual]) if k > 0)
        if candidates and -candidates[0][0] >= .2 * residual_area:
            negative_count, _, group = candidates[0]
            foreground.append(Fragment(residual, group, keys[group], True, 1., -negative_count / residual_area))
        else:
            fraction = float((residual & (pano == 0)).sum()) / residual_area
            background.append(Fragment(residual, 0, None, False, .5, fraction))
    return sorted(foreground, key=lambda r: spatial_key(r.mask)) + sorted(background, key=lambda r: spatial_key(r.mask))


def lift_fragments(rows, depth_m, intrinsics, pose, native_module):
    """Use the inherited cv2.rgbd converter and native Segment constructor."""
    import cv2

    points = cv2.rgbd.depthTo3d(depth_m, intrinsics)
    segments = []
    for index, row in enumerate(rows):
        semantic = int(row.is_thing)
        segment = native_module.Segment(
            points[row.mask].astype(np.float32).reshape(-1, 3), row.is_thing,
            row.input_group, semantic, row.score, row.overlap_ratio, pose, index,
            sem_feat=native_module.class_id_to_one_hot(semantic, num_classes=2))
        segment.backbone_mask = row.mask
        segments.append(segment)
    return segments
