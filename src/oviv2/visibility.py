from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.core.data_structures import Frame
from src.oviv2.addressing import VoxelKey


class VisibilityStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    OCCLUDED = "occluded"
    UNOBSERVED = "unobserved"


@dataclass(frozen=True)
class VisibilityConfig:
    voxel_size_m: float = 0.05
    depth_tolerance_m: float = 0.1

    def __post_init__(self) -> None:
        for name in ("voxel_size_m", "depth_tolerance_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


class VoxelVisibilityProjector:
    def __init__(self, config: VisibilityConfig = VisibilityConfig()) -> None:
        if not isinstance(config, VisibilityConfig):
            raise TypeError("config must be VisibilityConfig")
        self.config = config

    def classify(self, voxel_key: VoxelKey, frame: Frame) -> VisibilityStatus:
        grouped = self.classify_many((voxel_key,), frame)
        return next(status for status, keys in grouped.items() if keys)

    def classify_many(
        self,
        voxel_keys: tuple[VoxelKey, ...],
        frame: Frame,
    ) -> dict[VisibilityStatus, tuple[VoxelKey, ...]]:
        keys = tuple(sorted(voxel_keys))
        if not keys:
            return {status: () for status in VisibilityStatus}
        try:
            world_to_camera = np.linalg.inv(np.asarray(frame.pose, dtype=np.float64))
        except np.linalg.LinAlgError as exc:
            raise ValueError("frame pose must be invertible") from exc
        centers_world = (
            np.asarray(keys, dtype=np.float64) + 0.5
        ) * self.config.voxel_size_m
        centers_camera = (
            centers_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        )
        projected_depth = centers_camera[:, 2]
        status = np.full(len(keys), VisibilityStatus.UNOBSERVED.value, dtype=object)
        projectable = np.isfinite(projected_depth) & (projected_depth > 0.0)
        intrinsics = frame.intrinsics
        rows = np.zeros(len(keys), dtype=np.int64)
        columns = np.zeros(len(keys), dtype=np.int64)
        rows[projectable] = np.rint(
            intrinsics.fy
            * centers_camera[projectable, 1]
            / projected_depth[projectable]
            + intrinsics.cy
        ).astype(np.int64)
        columns[projectable] = np.rint(
            intrinsics.fx
            * centers_camera[projectable, 0]
            / projected_depth[projectable]
            + intrinsics.cx
        ).astype(np.int64)
        inside = (
            projectable
            & (rows >= 0)
            & (columns >= 0)
            & (rows < frame.depth.shape[0])
            & (columns < frame.depth.shape[1])
        )
        inside_indices = np.flatnonzero(inside)
        observed_depth = np.zeros(len(keys), dtype=np.float64)
        observed_depth[inside_indices] = frame.depth[
            rows[inside_indices], columns[inside_indices]
        ]
        testable = inside & np.isfinite(observed_depth) & (observed_depth > 0.0)
        difference = observed_depth - projected_depth
        status[testable & (np.abs(difference) <= self.config.depth_tolerance_m)] = (
            VisibilityStatus.PRESENT.value
        )
        status[testable & (difference > self.config.depth_tolerance_m)] = (
            VisibilityStatus.ABSENT.value
        )
        status[testable & (difference < -self.config.depth_tolerance_m)] = (
            VisibilityStatus.OCCLUDED.value
        )
        return {
            visibility: tuple(
                key for key, value in zip(keys, status) if value == visibility.value
            )
            for visibility in VisibilityStatus
        }
