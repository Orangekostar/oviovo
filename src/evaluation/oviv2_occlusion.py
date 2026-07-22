"""Prediction-independent depth classification for TESSE-CD occlusion targets."""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

import numpy as np


DepthState = Literal["present", "occluded", "absent", "unobserved"]


def classify_depth(
    *, d_obs: float, d_gt: float, tolerance_m: float = 0.10
) -> DepthState:
    """Classify one projected GT point against the observed depth image."""
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("tolerance_m must be positive and finite")
    if not np.isfinite(d_gt) or d_gt <= 0:
        raise ValueError("d_gt must be positive and finite")
    if not np.isfinite(d_obs) or d_obs <= 0:
        return "unobserved"
    residual = float(d_obs) - float(d_gt)
    if residual < -float(tolerance_m) and not np.isclose(
        residual, -float(tolerance_m), rtol=0.0, atol=1e-12
    ):
        return "occluded"
    if residual > float(tolerance_m) and not np.isclose(
        residual, float(tolerance_m), rtol=0.0, atol=1e-12
    ):
        return "absent"
    return "present"


def classify_voxel_depths(
    voxel_keys: np.ndarray,
    *,
    depth: np.ndarray,
    world_from_camera: np.ndarray,
    camera: Mapping[str, float | int],
    voxel_size_m: float = 0.05,
    tolerance_m: float = 0.10,
) -> dict[DepthState, np.ndarray]:
    """Project sorted voxel centers and partition them into depth states."""
    keys = np.asarray(voxel_keys)
    if keys.ndim != 2 or keys.shape[1] != 3 or keys.dtype != np.int64:
        raise ValueError("voxel_keys must have shape (N, 3) and dtype int64")
    if len(keys) > 1:
        expected = keys[np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))]
        if not np.array_equal(keys, expected) or len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError("voxel_keys must be sorted and unique")
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive and finite")

    image = np.asarray(depth, dtype=np.float64)
    height = int(camera["height"])
    width = int(camera["width"])
    if image.shape != (height, width):
        raise ValueError("depth shape disagrees with camera")
    transform = np.asarray(world_from_camera, dtype=np.float64)
    if (
        transform.shape != (4, 4)
        or not np.all(np.isfinite(transform))
        or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
    ):
        raise ValueError("world_from_camera must be a finite homogeneous transform")
    try:
        camera_from_world = np.linalg.inv(transform)
    except np.linalg.LinAlgError as error:
        raise ValueError("world_from_camera must be invertible") from error

    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    if not all(np.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError("camera intrinsics are invalid")

    points = (keys.astype(np.float64) + 0.5) * float(voxel_size_m)
    camera_points = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    gt_depth = camera_points[:, 2]
    projected = gt_depth > 0
    columns = np.full(len(keys), -1, dtype=np.int64)
    rows = np.full(len(keys), -1, dtype=np.int64)
    columns[projected] = np.floor(
        fx * camera_points[projected, 0] / gt_depth[projected] + cx + 0.5
    ).astype(np.int64)
    rows[projected] = np.floor(
        fy * camera_points[projected, 1] / gt_depth[projected] + cy + 0.5
    ).astype(np.int64)
    projected &= (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)

    selected: dict[DepthState, list[np.ndarray]] = {
        "present": [],
        "occluded": [],
        "absent": [],
        "unobserved": [],
    }
    for index, key in enumerate(keys):
        state: DepthState
        if not projected[index]:
            state = "unobserved"
        else:
            state = classify_depth(
                d_obs=float(image[rows[index], columns[index]]),
                d_gt=float(gt_depth[index]),
                tolerance_m=tolerance_m,
            )
        selected[state].append(key)
    return {
        state: np.asarray(values, dtype=np.int64).reshape((-1, 3))
        for state, values in selected.items()
    }


def sorted_voxel_array(values: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Normalize voxel keys to a sorted, unique ``N x 3 int64`` array."""
    array = np.asarray(values, dtype=np.int64)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("voxel array must have shape (N, 3)")
    return np.unique(array, axis=0)
