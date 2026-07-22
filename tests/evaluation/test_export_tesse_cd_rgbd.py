from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.evaluation.export_tesse_cd_rgbd import (
    camera_pose_matrix,
    depth_meters_to_mm,
    quaternion_matrix,
)


def _pose(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=z),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _transform(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(
        translation=SimpleNamespace(x=x, y=y, z=z),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def test_quaternion_matrix_and_camera_pose_compose_world_body_sensor() -> None:
    np.testing.assert_allclose(quaternion_matrix(0.0, 0.0, 0.0, 1.0), np.eye(3))

    pose = camera_pose_matrix(_pose(1.0, 2.0, 3.0), _transform(0.0, 0.05, 0.0))

    np.testing.assert_allclose(pose[:3, :3], np.eye(3))
    np.testing.assert_allclose(pose[:3, 3], [1.0, 2.05, 3.0])
    np.testing.assert_allclose(pose[3], [0.0, 0.0, 0.0, 1.0])


def test_depth_conversion_uses_millimeters_and_zeroes_invalid_values() -> None:
    depth = np.asarray([[0.0, 1.2344, np.nan, np.inf, -1.0]], dtype=np.float32)

    converted = depth_meters_to_mm(depth)

    assert converted.dtype == np.uint16
    np.testing.assert_array_equal(converted, [[0, 1234, 0, 0, 0]])
