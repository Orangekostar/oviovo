from __future__ import annotations

import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.visibility import VisibilityConfig, VisibilityStatus, VoxelVisibilityProjector


def _frame(depth_value: float) -> Frame:
    return Frame(
        frame_id=0,
        rgb=np.zeros((3, 3, 3), dtype=np.uint8),
        depth=np.full((3, 3), depth_value, dtype=np.float32),
        pose=np.eye(4),
        intrinsics=CameraIntrinsics(1.0, 1.0, 1.0, 1.0, 3, 3),
    )


def test_visibility_classifies_present_absent_occluded_and_unobserved() -> None:
    projector = VoxelVisibilityProjector(
        VisibilityConfig(voxel_size_m=1.0, depth_tolerance_m=0.1)
    )
    key = (0, 0, 1)

    assert projector.classify(key, _frame(1.5)) is VisibilityStatus.PRESENT
    assert projector.classify(key, _frame(2.0)) is VisibilityStatus.ABSENT
    assert projector.classify(key, _frame(1.0)) is VisibilityStatus.OCCLUDED
    assert projector.classify(key, _frame(0.0)) is VisibilityStatus.UNOBSERVED
    assert projector.classify((100, 0, 1), _frame(1.5)) is VisibilityStatus.UNOBSERVED


def test_occluded_and_unobserved_never_appear_in_absent_result() -> None:
    projector = VoxelVisibilityProjector(VisibilityConfig(voxel_size_m=1.0))
    keys = ((0, 0, 1), (100, 0, 1))

    result = projector.classify_many(keys, _frame(1.0))

    assert result[VisibilityStatus.ABSENT] == ()
    assert result[VisibilityStatus.OCCLUDED] == ((0, 0, 1),)
    assert result[VisibilityStatus.UNOBSERVED] == ((100, 0, 1),)
