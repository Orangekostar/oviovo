"""Online residency tiering for dense/coarse dual-map storage."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from src.core.data_structures import SurfaceTier, SystemState

logger = logging.getLogger("oviovo.modules.map_tiering")


class MapTieringModule:
    """Promote and degrade dense surfaces based on active-set usage."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.active_radius = float(config.get("active_radius", 3.0))
        self.warm_ttl_frames = int(config.get("warm_ttl_frames", 60))
        self.cold_ttl_frames = int(config.get("cold_ttl_frames", 180))
        logger.info("MapTieringModule initialized.")

    def process(self, state: SystemState, current_frame: int, camera_position: np.ndarray) -> SystemState:
        """Update object dense-surface tiers and evict cold surfaces when enabled."""
        active_ids = set(int(value) for value in state.active_set.all_candidate_ids)
        cam_pos = np.asarray(camera_position, dtype=np.float32)

        for object_id, obj in state.objects.items():
            if not self.enabled:
                obj.surface_tier = SurfaceTier.ACTIVE
                obj.dense_surface_resident = True
                entry = state.dense_surface_map.entries.get(int(object_id))
                if entry is not None:
                    entry.resident = True
                continue

            distance = float(np.linalg.norm(np.asarray(obj.centroid, dtype=np.float32) - cam_pos))
            idle_frames = max(int(current_frame) - int(obj.last_seen_frame), 0)
            is_active = int(object_id) in active_ids or distance <= self.active_radius

            if is_active:
                obj.surface_tier = SurfaceTier.ACTIVE
                obj.dense_surface_resident = True
                entry = state.dense_surface_map.entries.get(int(object_id))
                if entry is not None:
                    entry.resident = True
                continue

            if idle_frames <= self.warm_ttl_frames:
                obj.surface_tier = SurfaceTier.WARM
                obj.dense_surface_resident = True
                entry = state.dense_surface_map.entries.get(int(object_id))
                if entry is not None:
                    entry.resident = True
                continue

            obj.surface_tier = SurfaceTier.COLD
            obj.dense_surface_resident = False
            entry = state.dense_surface_map.entries.get(int(object_id))
            if entry is not None:
                entry.points = np.empty((0, 3), dtype=np.float32)
                entry.resident = False
                if idle_frames > self.cold_ttl_frames:
                    del state.dense_surface_map.entries[int(object_id)]

        return state
