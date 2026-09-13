"""Observed RGB-D cloud fusion, independent of semantic masks and GT surfaces."""
import numpy as np


def measured_world_points(depth, rgb, intrinsics, camera_to_world):
    depth, rgb = np.asarray(depth), np.asarray(rgb)
    k, pose = np.asarray(intrinsics), np.asarray(camera_to_world)
    if (depth.ndim != 2 or rgb.shape != (*depth.shape, 3) or k.shape != (3, 3)
            or pose.shape != (4, 4) or not np.isfinite(k).all()
            or not np.isfinite(pose).all() or k[0, 0] <= 0 or k[1, 1] <= 0):
        raise ValueError('aligned depth/RGB and finite pinhole camera required')
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[y, x].astype(np.float64)
    camera = np.column_stack(((x-k[0, 2])*z/k[0, 0], (y-k[1, 2])*z/k[1, 1], z))
    return camera @ pose[:3, :3].T+pose[:3, 3], rgb[y, x]


class RGBDVoxelCloud:
    """Sorted 63-bit voxel keys and sample sums; memory scales with occupied cells."""
    def __init__(self, voxel_size=.01):
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError('positive finite voxel size required')
        self.voxel_size = voxel_size
        self.keys = np.empty(0, dtype=np.uint64)
        self.sums = np.zeros((0, 7), dtype=np.float64)

    def add(self, xyz, rgb):
        xyz, rgb = np.asarray(xyz), np.asarray(rgb)
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape != rgb.shape
                or not np.isfinite(xyz).all() or not np.isfinite(rgb).all()
                or np.any(rgb < 0) or np.any(rgb > 255)):
            raise ValueError('finite aligned XYZ and RGB in [0,255] required')
        grid = np.floor(xyz/self.voxel_size)
        bias = 1 << 20
        if np.any(grid < -bias) or np.any(grid >= bias):
            raise ValueError('voxel coordinate exceeds signed 21-bit range')
        grid = (grid+bias).astype(np.uint64)
        packed = (grid[:, 0] << 42) | (grid[:, 1] << 21) | grid[:, 2]
        keys, inverse = np.unique(packed, return_inverse=True)
        added = np.zeros((len(keys), 7), dtype=np.float64)
        for c in range(3):
            added[:, c] = np.bincount(inverse, weights=xyz[:, c], minlength=len(keys))
            added[:, c+3] = np.bincount(inverse, weights=rgb[:, c], minlength=len(keys))
        added[:, 6] = np.bincount(inverse, minlength=len(keys))
        union = np.union1d(self.keys, keys)
        sums = np.zeros((len(union), 7), dtype=np.float64)
        sums[np.searchsorted(union, self.keys)] = self.sums
        sums[np.searchsorted(union, keys)] += added
        self.keys, self.sums = union, sums

    def arrays(self):
        count = self.sums[:, 6]
        mean = self.sums[:, :6]/count[:, None]
        return mean[:, :3].astype(np.float32), mean[:, 3:].astype(np.float32), count.astype(np.int64)
