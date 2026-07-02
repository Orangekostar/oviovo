"""Current-state geometry helpers for dynamic object lifecycle."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.data_structures import ObjectMap, ObjectState, SystemState


def is_exportable_object(obj: ObjectMap) -> bool:
    if obj.state in (ObjectState.GHOST, ObjectState.REMOVED):
        return False
    return len(obj.local_pcd) > 0


class DynamicGeometryModule:
    """Apply lifecycle decisions to every geometry source used by exports."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def purge_object_geometry(
        self,
        state: SystemState,
        obj: ObjectMap,
        *,
        frame_id: int,
        reason: str,
    ) -> None:
        object_id = int(obj.object_id)
        for support in state.tsdf_volume.owner_support.values():
            if object_id in support.support:
                del support.support[object_id]

        empty = np.empty((0, 3), dtype=np.float32)
        obj.local_pcd = empty.copy()
        obj.association_pcd = empty.copy()
        obj.observations = []
        obj.centroid = np.zeros(3, dtype=np.float32)
        obj.bbox_min = np.zeros(3, dtype=np.float32)
        obj.bbox_max = np.zeros(3, dtype=np.float32)
        obj.state = ObjectState.GHOST
        obj.removed_frame = int(frame_id)
        obj.purged_frame = int(frame_id)
        obj.dynamic_state_reason = str(reason)
        obj.debug["dynamic_geometry_purged"] = {
            "frame_id": int(frame_id),
            "reason": str(reason),
        }

        entry = state.dense_surface_map.entries.get(object_id)
        if entry is not None:
            entry.points = empty.copy()
            entry.resident = False
