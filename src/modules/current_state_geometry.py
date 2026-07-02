"""Final-state RGB-D geometry accumulator with free-space clearing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.data_structures import Frame
from src.modules.visibility_projector import project_world_points_to_depth


@dataclass
class GeometryVoxel:
    point: np.ndarray
    color: np.ndarray
    last_seen_frame: int


@dataclass
class CurrentStateGeometryAccumulator:
    voxel_size: float
    free_space_threshold: float = 0.08
    voxels: dict[tuple[int, int, int], GeometryVoxel] = field(default_factory=dict)

    def integrate_frame(self, frame: Frame, *, sample_stride: int) -> None:
        self.clear_visible_free_space(frame)
        points, colors = self._lift_frame(frame, max(1, int(sample_stride)))
        if len(points) == 0:
            return
        voxel_indices = np.floor(points / max(float(self.voxel_size), 1e-9)).astype(np.int32)
        for voxel, point, color in zip(voxel_indices, points, colors):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            self.voxels[key] = GeometryVoxel(
                point=np.asarray(point, dtype=np.float32).copy(),
                color=np.asarray(color, dtype=np.uint8).copy(),
                last_seen_frame=int(frame.frame_id),
            )

    def clear_visible_free_space(self, frame: Frame) -> None:
        if not self.voxels:
            return
        keys = list(self.voxels.keys())
        points = np.asarray([self.voxels[key].point for key in keys], dtype=np.float32)
        projection = project_world_points_to_depth(points, frame.pose, frame.intrinsics)
        remove_keys: list[tuple[int, int, int]] = []
        valid_indices = np.flatnonzero(projection.valid_mask)
        for idx in valid_indices:
            measured = float(frame.depth[projection.pixel_v[idx], projection.pixel_u[idx]])
            if not np.isfinite(measured) or measured <= 0.0:
                continue
            map_depth = float(projection.camera_depth[idx])
            if map_depth < measured - float(self.free_space_threshold):
                remove_keys.append(keys[int(idx)])
        for key in remove_keys:
            self.voxels.pop(key, None)

    @staticmethod
    def _lift_frame(frame: Frame, sample_stride: int) -> tuple[np.ndarray, np.ndarray]:
        rgb = frame.rgb[::sample_stride, ::sample_stride]
        depth = frame.depth[::sample_stride, ::sample_stride]
        u, v = np.meshgrid(
            np.arange(0, frame.rgb.shape[1], sample_stride),
            np.arange(0, frame.rgb.shape[0], sample_stride),
        )
        valid = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        z = depth[valid].astype(np.float32)
        u = u[valid].astype(np.float32)
        v = v[valid].astype(np.float32)
        x = (u - frame.intrinsics.cx) * z / frame.intrinsics.fx
        y = (v - frame.intrinsics.cy) * z / frame.intrinsics.fy
        points_cam = np.stack([x, y, z], axis=1)
        rotation = frame.pose[:3, :3].astype(np.float32)
        translation = frame.pose[:3, 3].astype(np.float32)
        points_world = (rotation @ points_cam.T).T + translation
        colors = rgb[valid].astype(np.uint8)
        return points_world.astype(np.float32), colors


def finalize_current_state_geometry(
    accum: CurrentStateGeometryAccumulator,
) -> tuple[np.ndarray, np.ndarray]:
    if not accum.voxels:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    ordered_keys = sorted(accum.voxels)
    points = np.asarray([accum.voxels[key].point for key in ordered_keys], dtype=np.float32)
    colors = np.asarray([accum.voxels[key].color for key in ordered_keys], dtype=np.uint8)
    return points, colors
