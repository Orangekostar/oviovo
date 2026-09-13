import numpy as np

from src.static_ovmap.geometry_support import supported_owner_pixels, independent_support_frames, native_pixel_rays


def test_geometry_support_rejects_occlusion_and_invalid_depth():
    owners = np.array([[1, 1, 2], [2, 3, 0]], dtype=np.int64)
    predicted = np.array([[1., 1., 2.], [2., 3., 1.]])
    depth = np.array([[1., 0., 1.], [2., 3.2, 1.]])
    assert supported_owner_pixels(owners, predicted, depth) == {1: 1, 2: 1}


def test_independent_support_is_not_duplicate_camera_or_semantic_query_count():
    poses = {0: np.eye(4), 10: np.eye(4), 20: np.eye(4)}
    poses[20][0, 3] = .1
    evidence = {0: {4: 200}, 10: {4: 300}, 20: {4: 200, 5: 20}}
    result = independent_support_frames(evidence, poses, min_pixels=100)
    assert result[4]['support_frame_ids'] == [0, 10, 20]
    assert result[4]['independent_frame_ids'] == [0, 20]
    assert 5 not in result


def test_native_camera_rays_use_integer_pixels_not_half_pixel_shift():
    k = np.array([[600.,0,599.5],[0,600.,339.5],[0,0,1.]])
    pose = np.eye(4); pose[:3,3] = [1,2,3]
    rays = native_pixel_rays(k, pose, 2, 2)
    np.testing.assert_allclose(rays[0,0,:3], [1,2,3])
    np.testing.assert_allclose(rays[0,0,3:], [-599.5/600,-339.5/600,1.], atol=1e-7)
