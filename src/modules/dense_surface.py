"""Dense semantic surface maintenance for the online dual-map architecture."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from src.core.data_structures import DenseSurfaceEntry, DenseSurfaceMap, ObjectMap, SystemState
from src.modules.semantic_memory import object_export_semantic_label

logger = logging.getLogger("oviovo.modules.dense_surface")


class DenseSurfaceModule:
    """Maintain per-object dense semantic surfaces for active interaction/export."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.voxel_size = float(config.get("dense_surface_voxel", 0.02))
        self.max_points = int(config.get("dense_surface_cap_per_object", 4000))
        self.update_interval = max(1, int(config.get("dense_surface_update_interval", config.get("update_interval", 1)) or 1))
        self.lazy_export = bool(config.get("dense_surface_lazy_export", False))
        self.min_tsdf_voxels_for_no_supplement = 10
        logger.info("DenseSurfaceModule initialized.")

    def process(self, state: SystemState, frame_id: int) -> SystemState:
        """Refresh dense surfaces for objects observed in the current frame."""
        if self.lazy_export:
            for obj in state.objects.values():
                self._refresh_semantic_label(state.dense_surface_map, obj)
            return state

        refresh_points_this_frame = int(frame_id) % int(self.update_interval) == 0
        for object_id, obj in state.objects.items():
            has_current = self._has_current_frame_observation(obj, frame_id)
            if not refresh_points_this_frame:
                self._refresh_semantic_label(state.dense_surface_map, obj)
                continue

            has_pending = self._has_pending_observation_since_refresh(obj)
            if self.update_interval == 1:
                should_refresh_points = has_current
            else:
                should_refresh_points = has_current or has_pending

            if not should_refresh_points:
                self._refresh_semantic_label(state.dense_surface_map, obj)
                continue

            surface_points = self._surface_points_for_object(
                obj,
                frame_id,
                include_pending_since_refresh=self.update_interval > 1 and has_pending,
            )
            if len(surface_points) == 0:
                continue

            entry = state.dense_surface_map.entries.get(int(object_id))
            if entry is None or not entry.resident or len(entry.points) == 0:
                merged = surface_points
            else:
                merged = np.concatenate([entry.points, surface_points], axis=0)

            merged = self._voxel_downsample_mean(merged, self.voxel_size)
            if len(merged) > self.max_points:
                merged = self._deterministic_cap(merged, self.max_points)

            label = self._semantic_label(obj)
            self._store_surface_entry(state, obj, merged, label, frame_id)

        return state

    def refresh_all_for_export(self, state: SystemState, frame_id: int | None = None) -> SystemState:
        """Build resident dense surfaces for export, including lazy-export mode."""
        if frame_id is None:
            frame_id = max((int(obj.last_seen_frame) for obj in state.objects.values()), default=0)
        for obj in state.objects.values():
            surface_points = self._all_surface_points_for_object(obj)
            if len(surface_points) == 0:
                self._refresh_semantic_label(state.dense_surface_map, obj)
                continue
            surface_points = self._voxel_downsample_mean(surface_points, self.voxel_size)
            if len(surface_points) > self.max_points:
                surface_points = self._deterministic_cap(surface_points, self.max_points)
            self._store_surface_entry(state, obj, surface_points, self._semantic_label(obj), int(frame_id))
        return state

    def resident_point_count(self, dense_surface_map: DenseSurfaceMap) -> int:
        return int(
            sum(len(entry.points) for entry in dense_surface_map.entries.values() if entry.resident)
        )

    def _refresh_semantic_label(self, dense_surface_map: DenseSurfaceMap, obj: ObjectMap) -> None:
        entry = dense_surface_map.entries.get(int(obj.object_id))
        if entry is None:
            return
        entry.semantic_label = self._semantic_label(obj)

    def _surface_points_for_object(
        self,
        obj: ObjectMap,
        frame_id: int,
        *,
        include_pending_since_refresh: bool = False,
    ) -> np.ndarray:
        min_frame_id = int(frame_id)
        if include_pending_since_refresh:
            min_frame_id = int(getattr(obj, "last_dense_refresh_frame", -1)) + 1
        point_chunks = [
            np.asarray(observation.patch.points, dtype=np.float32)
            for observation in obj.observations
            if (
                (min_frame_id <= int(observation.frame_id) <= int(frame_id))
                and len(observation.patch.points) > 0
            )
        ]
        if point_chunks:
            points = np.concatenate(point_chunks, axis=0)
        else:
            points = np.empty((0, 3), dtype=np.float32)

        owned_voxel_count = int(
            obj.debug.get("global_instance_substrate", {}).get("owned_voxel_count", 0)
        )
        if owned_voxel_count < self.min_tsdf_voxels_for_no_supplement and len(obj.local_pcd) > 0:
            supplement = self._voxel_downsample_mean(np.asarray(obj.local_pcd, dtype=np.float32), self.voxel_size)
            points = supplement if len(points) == 0 else np.concatenate([points, supplement], axis=0)

        return points.astype(np.float32, copy=False)

    def _semantic_label(self, obj: ObjectMap) -> str:
        return object_export_semantic_label(obj)

    def _store_surface_entry(
        self,
        state: SystemState,
        obj: ObjectMap,
        points: np.ndarray,
        label: str,
        frame_id: int,
    ) -> None:
        state.dense_surface_map.entries[int(obj.object_id)] = DenseSurfaceEntry(
            points=np.asarray(points, dtype=np.float32),
            object_id=int(obj.object_id),
            semantic_label=str(label),
            last_refresh_frame=int(frame_id),
            resident=True,
        )
        obj.last_dense_refresh_frame = int(frame_id)
        obj.dense_surface_resident = True

    def _all_surface_points_for_object(self, obj: ObjectMap) -> np.ndarray:
        point_chunks = [
            np.asarray(observation.patch.points, dtype=np.float32)
            for observation in obj.observations
            if len(observation.patch.points) > 0
        ]
        if point_chunks:
            points = np.concatenate(point_chunks, axis=0)
        else:
            points = np.empty((0, 3), dtype=np.float32)

        owned_voxel_count = int(
            obj.debug.get("global_instance_substrate", {}).get("owned_voxel_count", 0)
        )
        if owned_voxel_count < self.min_tsdf_voxels_for_no_supplement and len(obj.local_pcd) > 0:
            supplement = self._voxel_downsample_mean(np.asarray(obj.local_pcd, dtype=np.float32), self.voxel_size)
            points = supplement if len(points) == 0 else np.concatenate([points, supplement], axis=0)

        return points.astype(np.float32, copy=False)

    @staticmethod
    def _has_current_frame_observation(obj: ObjectMap, frame_id: int) -> bool:
        if int(obj.last_seen_frame) == int(frame_id):
            return True
        return any(int(observation.frame_id) == int(frame_id) for observation in obj.observations)

    @staticmethod
    def _has_pending_observation_since_refresh(obj: ObjectMap) -> bool:
        last_refresh = int(getattr(obj, "last_dense_refresh_frame", -1))
        if last_refresh < 0:
            if any(True for _observation in obj.observations):
                return True
            return int(getattr(obj, "last_seen_frame", -1)) >= 0
        if int(getattr(obj, "last_seen_frame", -1)) > last_refresh:
            return True
        return any(int(observation.frame_id) > last_refresh for observation in obj.observations)

    @staticmethod
    def _deterministic_cap(points: np.ndarray, max_points: int) -> np.ndarray:
        if len(points) <= max_points:
            return points
        order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
        ordered = points[order]
        indices = np.linspace(0, len(ordered) - 1, max_points, dtype=np.int32)
        return ordered[indices]

    @staticmethod
    def _voxel_downsample_mean(points: np.ndarray, voxel_size: float) -> np.ndarray:
        if len(points) == 0 or voxel_size <= 0.0:
            return points.astype(np.float32, copy=False)
        grid = np.floor(points / voxel_size).astype(np.int32)
        unique_grid, inverse = np.unique(grid, axis=0, return_inverse=True)
        downsampled = np.zeros((len(unique_grid), 3), dtype=np.float32)
        counts = np.bincount(inverse)
        for axis in range(3):
            downsampled[:, axis] = np.bincount(inverse, weights=points[:, axis]) / counts
        return downsampled
