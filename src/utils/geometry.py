"""Geometry utilities for 3D mapping operations."""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np

from src.core.data_structures import CameraIntrinsics


def depth_to_points(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project depth pixels to 3D world coordinates.

    Args:
        depth: (H, W) depth in meters.
        intrinsics: Camera intrinsic parameters.
        pose: (4, 4) camera-to-world transform.
        mask: Optional (H, W) bool mask to select pixels.

    Returns:
        (N, 3) world-coordinate points.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    if mask is not None:
        valid = mask & (depth > 0)
    else:
        valid = depth > 0

    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = depth[valid].astype(np.float64)

    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy

    points_cam = np.stack([x, y, z], axis=-1)  # (N, 3)

    # Transform to world coordinates
    R = pose[:3, :3]
    t = pose[:3, 3]
    points_world = (R @ points_cam.T).T + t

    return points_world.astype(np.float32)


def compute_bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute axis-aligned bounding box.

    Args:
        points: (N, 3) point cloud.

    Returns:
        (bbox_min, bbox_max) each (3,).
    """
    if len(points) == 0:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    return points.min(axis=0), points.max(axis=0)


def bbox_iou_3d(min_a: np.ndarray, max_a: np.ndarray,
                min_b: np.ndarray, max_b: np.ndarray) -> float:
    """Compute IoU of two 3D axis-aligned bounding boxes.

    Returns:
        IoU value in [0, 1].
    """
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = np.prod(inter_dims)

    vol_a = np.prod(np.maximum(max_a - min_a, 0.0))
    vol_b = np.prod(np.maximum(max_b - min_b, 0.0))
    union_vol = vol_a + vol_b - inter_vol

    if union_vol < 1e-10:
        return 0.0
    return float(inter_vol / union_vol)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Simple voxel grid downsampling.

    Args:
        points: (N, 3) point cloud.
        voxel_size: Voxel edge length.

    Returns:
        Downsampled (M, 3) point cloud.
    """
    if len(points) == 0 or voxel_size <= 0:
        return points
    grid = np.floor(points / voxel_size).astype(np.int32)
    _, idx = np.unique(grid, axis=0, return_index=True)
    return points[idx]


def world_to_voxel(
    points: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Convert world coordinates to voxel grid indices.

    Args:
        points: (N, 3) world-coordinate points.
        origin: (3,) world-coordinate origin of the volume.
        voxel_size: Voxel edge length.

    Returns:
        (N, 3) int64 voxel indices.
    """
    return np.floor((points - origin) / voxel_size).astype(np.int64)


def voxel_to_world(
    indices: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Convert voxel indices to world-coordinate centers.

    Args:
        indices: (N, 3) int voxel indices.
        origin: (3,) world-coordinate origin of the volume.
        voxel_size: Voxel edge length.

    Returns:
        (N, 3) float32 world coordinates (voxel centers).
    """
    return (indices.astype(np.float64) + 0.5) * voxel_size + origin
