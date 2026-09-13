import numpy as np

from src.static_ovmap.native_partition import native_partition


def test_native_first_vertex_color_preserves_unknown_and_wide_owner_ids():
    faces = np.arange(9).reshape(3, 3)
    rgb = np.array([[10, 20, 30], [255, 0, 0], [0, 255, 0],
                    [50, 60, 70], [0, 0, 0], [0, 0, 0],
                    [99, 99, 99], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
    owners = native_partition(faces, rgb, {70000: (10, 20, 30), 9: (50, 60, 70)})
    assert owners.dtype == np.int64
    assert owners.tolist() == [70000]*3 + [9]*3 + [0]*3
