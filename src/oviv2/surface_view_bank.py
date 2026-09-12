"""Sparse source-row and unique-pixel evidence for fixed-surface readouts."""

from __future__ import annotations

import numpy as np

from src.core.data_structures import Frame
from src.oviv2.ovi_surface_attributes import project_world_points


def project_source_support(
    xyz: np.ndarray,
    source_indices: np.ndarray,
    frame: Frame,
    *,
    depth_tolerance_m: float = 0.05,
    depth_max_m: float = 10.0,
    batch_size: int = 250000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return supported source rows and their pixels; rows preserve input order.

    Duplicate triangle-soup rows remain addressable. Callers count unique pixels,
    never repeated vertices, as observation support. No mask fill or dilation.
    """
    indices = np.asarray(source_indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("source_indices must be an integer vector")
    if batch_size < 1 or depth_tolerance_m <= 0 or depth_max_m <= 0:
        raise ValueError("invalid projection parameters")
    selected_rows, selected_pixels = [], []
    depth = np.asarray(frame.depth)
    h, w = depth.shape
    if frame.rgb.shape[:2] != (h, w):
        raise ValueError("RGB and depth grids must align")
    for start in range(0, len(indices), batch_size):
        rows = indices[start : start + batch_size]
        projection = project_world_points(xyz[rows], frame.pose, frame.intrinsics)
        candidate = np.flatnonzero(projection.projectable)
        r, c = projection.rows[candidate], projection.columns[candidate]
        observed = depth[r, c]
        predicted = projection.camera_xyz[candidate, 2]
        valid = (
            np.isfinite(observed)
            & (observed > 0)
            & (observed <= depth_max_m)
            & (np.abs(observed - predicted) <= depth_tolerance_m)
        )
        selected_rows.append(rows[candidate[valid]])
        selected_pixels.append((r[valid] * w + c[valid]).astype(np.int32))
    return (
        np.concatenate(selected_rows) if selected_rows else np.empty(0, np.int64),
        np.concatenate(selected_pixels) if selected_pixels else np.empty(0, np.int32),
    )
