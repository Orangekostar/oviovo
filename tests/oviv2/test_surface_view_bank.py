import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame


def test_depth_occlusion_and_repeated_vertices_do_not_create_extra_pixel_votes():
    from src.oviv2.surface_view_bank import project_source_support

    xyz = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, -1.0]]
    )
    frame = Frame(
        frame_id=0,
        rgb=np.zeros((3, 3, 3), np.uint8),
        depth=np.ones((3, 3)),
        pose=np.eye(4),
        intrinsics=CameraIntrinsics(1.0, 1.0, 1.0, 1.0, 3, 3),
        timestamp=0.0,
    )
    rows, pixels = project_source_support(xyz, np.arange(4), frame, batch_size=1)
    assert rows.tolist() == [0, 1]
    assert pixels.tolist() == [4, 4]
    assert len(np.unique(pixels)) == 1
