"""Static final-mask requests keep depth visibility and exact crop identities."""

import numpy as np

from src.static_ovmap.module_validation.static_regions import (
    project_visible_rows,
    static_region,
)


def test_visibility_keeps_one_nearest_row_and_rejects_occluded_depth():
    xyz = np.array([[0., 0., 1.], [0., 0., 1.01], [1., 0., 1.], [0., 0., 2.]])
    depth = np.zeros((3, 3), np.float32)
    depth[1, 1:] = 1
    source, pixels = project_visible_rows(xyz, np.eye(4),
        np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]]), depth)
    np.testing.assert_array_equal(source, [0, 2])
    np.testing.assert_array_equal(pixels, [4, 5])


def test_new_geometry_mask_has_own_bbox_union_lineage_and_request_identity():
    owners = np.array([[0, 0, 0, 0], [0, 7, 7, 0], [0, 7, 7, 0], [0, 0, 0, 0]])
    local = np.zeros_like(owners)
    local[1:3, 1:4] = 3
    frame = {"scene_id": "sceneA", "frame_id": 8, "rgb_sha256": "a" * 64}
    first = static_region(owners, local, 7, frame, "static:original", (7,), 0)
    request, mask, union = first
    assert request.bbox_xyxy == (1, 1, 2, 2)
    assert request.visible_target_pixels == 4
    np.testing.assert_array_equal(mask, owners == 7)
    np.testing.assert_array_equal(union, (owners == 7) | (local == 3))
    owners[2, 2] = 9
    changed = static_region(owners, local, 7, frame, "static:changed", (7,), 0)
    assert changed[0].request_id != request.request_id
    assert changed[0].target_mask_sha256 != request.target_mask_sha256
    assert static_region(owners, local, 9, frame, "static:changed", (7,), 0) is None
