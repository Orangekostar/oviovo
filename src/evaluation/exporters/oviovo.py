"""Export authoritative OVIOVO state without depending on dense visualization caches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np

from src.core.data_structures import ObjectMap, ObjectState, SystemState
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.modules.semantic_memory import object_export_semantic_state


def _entity_from_object(obj: ObjectMap) -> EntityPrediction:
    semantic = object_export_semantic_state(obj)
    label = str(semantic.get("export_label", "")).strip() or None
    score = float(semantic.get("posterior_score", 0.0))
    embedding = obj.semantic_memory.aggregated_feature
    metadata = {
        "object_id": int(obj.object_id),
        "update_count": int(obj.update_count),
        "observation_count": int(len(obj.observations)),
        "confidence": float(obj.confidence),
        "semantic_state": str(semantic.get("semantic_state", "unlabeled")),
        "semantic_export_state": str(semantic.get("export_state", "unlabeled")),
        "semantic_export_source": str(semantic.get("export_source", "")),
    }
    return EntityPrediction(
        entity_id=str(obj.object_id),
        points_xyz=np.asarray(obj.local_pcd, dtype=np.float32),
        semantic_embedding=embedding,
        semantic_label=label,
        semantic_score=score,
        lifecycle_state=obj.state.value,
        first_seen=float(obj.creation_frame),
        last_seen=float(obj.last_seen_frame),
        metadata=metadata,
    )


def export_map_snapshot(
    state: SystemState,
    *,
    method: str,
    scene_id: str,
    timestamp: float,
    scope: Literal["current", "history"] = "current",
    runtime: Mapping[str, float] | None = None,
) -> MapSnapshot:
    """Create a causal snapshot from TSDF/ObjectMap authoritative state.

    Current snapshots expose only ACTIVE objects. History snapshots retain every
    object still present in the history pool, including dormant and removed
    records. DenseSurfaceMap is intentionally not read here.
    """
    if scope not in {"current", "history"}:
        raise ValueError(f"invalid snapshot scope: {scope}")
    objects = [state.objects[object_id] for object_id in sorted(state.objects)]
    if scope == "current":
        objects = [obj for obj in objects if obj.state == ObjectState.ACTIVE]
    entities = [_entity_from_object(obj) for obj in objects]
    background = np.asarray(state.background.point_cloud, dtype=np.float32)
    background_xyz = background if len(background) else None
    return MapSnapshot(
        method=method,
        scene_id=scene_id,
        timestamp=float(timestamp),
        entities=entities,
        background_xyz=background_xyz,
        scope=scope,
        runtime=dict(runtime or {}),
    )
