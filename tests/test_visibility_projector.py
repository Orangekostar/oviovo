from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import CameraIntrinsics
from src.modules.visibility_projector import (
    filter_points_by_depth_consistency,
    project_world_points_to_depth,
)


def test_project_world_points_to_depth_returns_pixels_and_camera_depths() -> None:
    intr = CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=5, height=5)
    pose = np.eye(4, dtype=np.float64)
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, -0.1, 1.0],
            [10.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    projection = project_world_points_to_depth(points, pose, intr)

    assert projection.valid_mask.tolist() == [True, True, True, False, False]
    assert projection.pixel_u.tolist()[:3] == [2, 3, 2]
    assert projection.pixel_v.tolist()[:3] == [2, 2, 1]
    np.testing.assert_allclose(projection.camera_depth[:3], np.array([1.0, 1.0, 1.0]), atol=1e-6)


def test_filter_points_by_depth_consistency_rejects_occluded_and_invalid_depth_points() -> None:
    intr = CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=5, height=5)
    pose = np.eye(4, dtype=np.float64)
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 1.0
    depth[2, 3] = 0.7
    points = np.array(
        [
            [0.0, 0.0, 1.02],
            [0.1, 0.0, 1.0],
            [0.0, -0.1, 1.0],
        ],
        dtype=np.float32,
    )

    mask, diagnostics = filter_points_by_depth_consistency(
        points,
        depth,
        pose,
        intr,
        distance_threshold=0.05,
    )

    assert mask.tolist() == [True, False, False]
    assert diagnostics["input_point_count"] == 3
    assert diagnostics["projected_in_bounds_count"] == 3
    assert diagnostics["accepted_point_count"] == 1
    assert diagnostics["depth_rejected_point_count"] == 1
    assert diagnostics["invalid_depth_point_count"] == 1
