"""Frame-depth visibility utilities for 3D map points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.data_structures import CameraIntrinsics


@dataclass(frozen=True)
class ProjectionResult:
    pixel_u: np.ndarray
    pixel_v: np.ndarray
    camera_depth: np.ndarray
    valid_mask: np.ndarray


def project_world_points_to_depth(
    points: np.ndarray,
    pose: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> ProjectionResult:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid_mask = np.zeros(len(points), dtype=bool)
    pixel_u = np.zeros(len(points), dtype=np.int32)
    pixel_v = np.zeros(len(points), dtype=np.int32)
    camera_depth = np.zeros(len(points), dtype=np.float32)
    if len(points) == 0:
        return ProjectionResult(pixel_u, pixel_v, camera_depth, valid_mask)

    world_to_camera = np.linalg.inv(np.asarray(pose, dtype=np.float64))
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
    z = camera_points[:, 2]
    forward = z > 1e-6

    u_float = intrinsics.fx * (camera_points[:, 0] / np.maximum(z, 1e-6)) + intrinsics.cx
    v_float = intrinsics.fy * (camera_points[:, 1] / np.maximum(z, 1e-6)) + intrinsics.cy
    u = np.rint(u_float).astype(np.int32)
    v = np.rint(v_float).astype(np.int32)
    in_bounds = (
        forward
        & (u >= 0)
        & (u < int(intrinsics.width))
        & (v >= 0)
        & (v < int(intrinsics.height))
    )

    pixel_u[:] = u
    pixel_v[:] = v
    camera_depth[:] = z.astype(np.float32)
    valid_mask[:] = in_bounds
    return ProjectionResult(pixel_u, pixel_v, camera_depth, valid_mask)


def filter_points_by_depth_consistency(
    points: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    distance_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    depth = np.asarray(depth, dtype=np.float32)
    projection = project_world_points_to_depth(points, pose, intrinsics)
    keep = np.zeros(len(points), dtype=bool)
    if len(points) == 0:
        return keep, {
            "input_point_count": 0,
            "projected_in_bounds_count": 0,
            "accepted_point_count": 0,
            "depth_rejected_point_count": 0,
            "invalid_depth_point_count": 0,
        }

    valid_indices = np.flatnonzero(projection.valid_mask)
    measured = depth[projection.pixel_v[valid_indices], projection.pixel_u[valid_indices]]
    valid_depth = np.isfinite(measured) & (measured > 0.0)
    depth_delta = np.abs(projection.camera_depth[valid_indices] - measured)
    accepted_local = valid_depth & (depth_delta <= float(distance_threshold))
    keep[valid_indices[accepted_local]] = True

    diagnostics = {
        "input_point_count": int(len(points)),
        "projected_in_bounds_count": int(len(valid_indices)),
        "accepted_point_count": int(keep.sum()),
        "depth_rejected_point_count": int((valid_depth & ~accepted_local).sum()),
        "invalid_depth_point_count": int((~valid_depth).sum()),
    }
    return keep, diagnostics
