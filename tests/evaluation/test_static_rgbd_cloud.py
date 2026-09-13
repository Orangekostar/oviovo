import numpy as np
import pytest

from src.static_ovmap.rgbd_cloud import RGBDVoxelCloud, measured_world_points


def test_measured_integer_pixels_pose_and_invalid_depth():
    depth = np.array([[2., 0.], [np.nan, 4.]])
    color = np.arange(12).reshape(2, 2, 3)
    pose = np.eye(4); pose[:3, 3] = [10, 20, 30]
    k = np.array([[2., 0., 0.], [0., 4., 0.], [0., 0., 1.]])
    xyz, rgb = measured_world_points(depth, color, k, pose)
    np.testing.assert_allclose(xyz, [[10, 20, 32], [12, 21, 34]])
    np.testing.assert_array_equal(rgb, [[0, 1, 2], [9, 10, 11]])


def test_cloud_weighted_merge_preserves_negative_voxel_boundary():
    cloud = RGBDVoxelCloud(.01)
    cloud.add(np.array([[-.001, 0, 0], [.001, 0, 0]]), np.array([[10, 0, 0], [20, 0, 0]]))
    cloud.add(np.array([[.003, 0, 0], [.005, 0, 0]]), np.array([[40, 0, 0], [60, 0, 0]]))
    xyz, rgb, count = cloud.arrays()
    np.testing.assert_allclose(xyz, [[-.001, 0, 0], [.003, 0, 0]])
    np.testing.assert_allclose(rgb, [[10, 0, 0], [40, 0, 0]])
    np.testing.assert_array_equal(count, [1, 3])
    with pytest.raises(ValueError, match='range'):
        cloud.add(np.array([[1e6, 0, 0]]), np.zeros((1, 3)))
