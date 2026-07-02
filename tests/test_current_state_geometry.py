import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame
from src.modules.current_state_geometry import (
    CurrentStateGeometryAccumulator,
    finalize_current_state_geometry,
)


def _frame(depth_value, frame_id):
    rgb = np.full((4, 4, 3), frame_id, dtype=np.uint8)
    depth = np.full((4, 4), depth_value, dtype=np.float32)
    return Frame(
        frame_id=frame_id,
        rgb=rgb,
        depth=depth,
        pose=np.eye(4, dtype=np.float32),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.5, cy=1.5, width=4, height=4),
    )


def test_later_frame_overwrites_same_voxel_color():
    accum = CurrentStateGeometryAccumulator(voxel_size=10.0, free_space_threshold=0.1)
    accum.integrate_frame(_frame(2.0, 10), sample_stride=2)
    accum.integrate_frame(_frame(2.0, 20), sample_stride=2)

    points, colors = finalize_current_state_geometry(accum)

    assert len(points) > 0
    assert np.all(colors == 20)


def test_free_space_clears_old_front_surface():
    accum = CurrentStateGeometryAccumulator(voxel_size=0.1, free_space_threshold=0.2)
    accum.integrate_frame(_frame(1.0, 10), sample_stride=2)
    before = len(accum.voxels)
    accum.integrate_frame(_frame(3.0, 20), sample_stride=2)

    assert len(accum.voxels) <= before
