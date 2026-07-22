"""Deterministic TESSE-CD adapters for immutable OVIV2 Stage3 snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.json_contracts import loads_strict
from src.oviv2.entities import EntityRegistry
from src.oviv2.meshing import derive_labeled_mesh
from src.oviv2.semantic_fusion import SemanticFusionConfig
from src.oviv2.snapshot import VoxelMapSnapshot


_SCENES = frozenset({"apartment", "office"})
_ROLES = frozenset({"official", "common_v2"})


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strings(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(item.strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class TesseCausalCheckpoint:
    frame_index: int
    timestamp_ns: int
    relative_timestamp_ns: int
    event_ids: Sequence[str]
    roles: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_index",
            _integer(self.frame_index, "frame_index", minimum=0),
        )
        object.__setattr__(
            self,
            "timestamp_ns",
            _integer(self.timestamp_ns, "timestamp_ns", minimum=1),
        )
        object.__setattr__(
            self,
            "relative_timestamp_ns",
            _integer(
                self.relative_timestamp_ns,
                "relative_timestamp_ns",
                minimum=0,
            ),
        )
        event_ids = _strings(self.event_ids, "event_ids")
        roles = _strings(self.roles, "roles")
        if not roles:
            raise ValueError("roles must be non-empty")
        if any(role not in _ROLES for role in roles):
            raise ValueError("roles contain an unsupported TESSE-CD role")
        object.__setattr__(self, "event_ids", event_ids)
        object.__setattr__(self, "roles", roles)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def load_causal_checkpoints(
    schedule: Path,
    *,
    scene: str,
    frame_count: int,
) -> Sequence[TesseCausalCheckpoint]:
    """Load, sort, and validate every scheduled checkpoint for one scene."""
    if scene not in _SCENES:
        raise ValueError(f"unknown scene: {scene}")
    expected_frame_count = _integer(frame_count, "frame_count", minimum=1)
    source = Path(schedule)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = _mapping(
        loads_strict(source.read_text(encoding="utf-8"), label="schedule"),
        "schedule",
    )
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
        or payload.get("manifest_id") != "tesse_cd_causal_schedule_v2"
        or payload.get("dataset") != "TESSE-CD"
        or payload.get("method_predictions_used") is not False
        or payload.get("parameters", {}).get("frame_indexing") != "zero_based"
    ):
        raise ValueError("causal schedule identity mismatch")
    scenes = _mapping(payload.get("scenes"), "schedule scenes")
    if set(scenes) != _SCENES:
        raise ValueError("causal schedule must contain apartment and office exactly")
    selected = _mapping(scenes[scene], f"schedule scene {scene}")
    declared_frame_count = selected.get("frame_count")
    if type(declared_frame_count) is not int or declared_frame_count <= 0:
        raise ValueError("causal schedule frame count must be a positive integer")
    if declared_frame_count != expected_frame_count:
        raise ValueError("causal schedule frame count mismatch")
    raw_entries = selected.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("causal schedule entries must be a non-empty list")

    checkpoints: list[TesseCausalCheckpoint] = []
    for raw in raw_entries:
        entry = _mapping(raw, "causal schedule entry")
        checkpoints.append(
            TesseCausalCheckpoint(
                frame_index=entry.get("frame_index"),  # type: ignore[arg-type]
                timestamp_ns=entry.get("timestamp_ns"),  # type: ignore[arg-type]
                relative_timestamp_ns=entry.get("relative_timestamp_ns"),  # type: ignore[arg-type]
                event_ids=entry.get("event_ids"),  # type: ignore[arg-type]
                roles=entry.get("roles"),  # type: ignore[arg-type]
            )
        )
    checkpoints.sort(key=lambda item: item.frame_index)
    frames = [item.frame_index for item in checkpoints]
    if len(frames) != len(set(frames)):
        raise ValueError("causal schedule contains duplicate checkpoint frames")
    if any(frame >= expected_frame_count for frame in frames):
        raise ValueError("causal checkpoint is outside the declared frame count")
    if any(
        current.timestamp_ns >= following.timestamp_ns
        for current, following in zip(checkpoints, checkpoints[1:])
    ):
        raise ValueError("causal checkpoint timestamps must be strictly increasing")
    if any(
        current.relative_timestamp_ns >= following.relative_timestamp_ns
        for current, following in zip(checkpoints, checkpoints[1:])
    ):
        raise ValueError("relative checkpoint timestamps must be strictly increasing")
    return tuple(checkpoints)


def _timestamp_sequence(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("timestamp_ns_by_frame must be a sequence")
    result = tuple(
        _integer(value, "timestamp_ns_by_frame", minimum=1) for value in values
    )
    if not result:
        raise ValueError("timestamp_ns_by_frame must be non-empty")
    if any(current >= following for current, following in zip(result, result[1:])):
        raise ValueError("timestamp_ns_by_frame must be strictly increasing")
    return result


def _class_names(values: Sequence[str]) -> tuple[str, ...]:
    names = _strings(values, "class_names")
    if not names:
        raise ValueError("class_names must be non-empty")
    return names


def build_neutral_current_snapshot(
    snapshot: VoxelMapSnapshot,
    *,
    timestamp_ns: int,
    class_names: Sequence[str],
    object_semantic_ids: frozenset[int],
    fusion: SemanticFusionConfig,
    timestamp_ns_by_frame: Sequence[int],
) -> MapSnapshot:
    """Convert immutable Stage3 state into the neutral current-map contract."""
    if not isinstance(snapshot, VoxelMapSnapshot):
        raise TypeError("snapshot must be a VoxelMapSnapshot")
    if not isinstance(snapshot.registry, EntityRegistry):
        raise ValueError("Stage3 neutral export requires a snapshot registry")
    if not isinstance(fusion, SemanticFusionConfig):
        raise TypeError("fusion must be a SemanticFusionConfig")
    current_timestamp_ns = _integer(timestamp_ns, "timestamp_ns", minimum=1)
    frame_timestamps = _timestamp_sequence(timestamp_ns_by_frame)
    frame_index = snapshot.metadata.frame_id
    if frame_index >= len(frame_timestamps):
        raise ValueError("snapshot frame is outside timestamp_ns_by_frame")
    if frame_timestamps[frame_index] != current_timestamp_ns:
        raise ValueError("snapshot frame timestamp does not match timestamp_ns")
    snapshot_timestamp = float(snapshot.metadata.timestamp)
    if snapshot_timestamp not in {
        float(current_timestamp_ns),
        current_timestamp_ns / 1_000_000_000,
    }:
        raise ValueError("snapshot metadata timestamp does not match timestamp_ns")

    names = _class_names(class_names)
    if not isinstance(object_semantic_ids, frozenset):
        raise TypeError("object_semantic_ids must be a frozenset")
    object_ids = frozenset(
        _integer(value, "object semantic ID", minimum=1)
        for value in object_semantic_ids
    )
    if not object_ids:
        raise ValueError("object_semantic_ids must be non-empty")
    if max(object_ids) >= len(names):
        raise ValueError("object semantic ID is outside class_names")

    registry = snapshot.registry
    posteriors: dict[int, tuple[tuple[int, float], ...]] = {}
    for entity_id, entity in sorted(registry.entities.items()):
        first_frame = _integer(
            entity.first_frame_id,
            "entity first_frame_id",
            minimum=0,
        )
        last_frame = _integer(
            entity.last_frame_id,
            "entity last_frame_id",
            minimum=0,
        )
        if first_frame > last_frame:
            raise ValueError("entity first frame must not exceed its last frame")
        if last_frame > frame_index:
            raise ValueError("entity registry state comes from a future current frame")
        probabilities = tuple(entity.semantic_posterior.probabilities)
        if any(semantic_id >= len(names) for semantic_id, _ in probabilities):
            raise ValueError("registry semantic ID is outside class_names")
        posteriors[int(entity_id)] = probabilities
    mesh = derive_labeled_mesh(
        snapshot.geometry,
        snapshot.evidence,
        snapshot.ownership,
        entity_posteriors=posteriors,
        semantic_fusion=fusion,
        valid_semantic_ids=frozenset(range(1, len(names))),
    )

    vertices = np.asarray(mesh.vertices_xyz)
    entity_ids = np.asarray(mesh.entity_ids, dtype=np.int64)
    semantic_ids = np.asarray(mesh.semantic_ids, dtype=np.int64)
    voxel_keys = np.floor(
        vertices.astype(np.float64, copy=False) / snapshot.metadata.voxel_size_m
    ).astype(np.int64)
    order = np.lexsort(
        (
            vertices[:, 2],
            vertices[:, 1],
            vertices[:, 0],
            voxel_keys[:, 2],
            voxel_keys[:, 1],
            voxel_keys[:, 0],
            semantic_ids,
            entity_ids,
        )
    )
    object_mask = (entity_ids > 0) & np.isin(
        semantic_ids,
        np.fromiter(sorted(object_ids), dtype=np.int64),
    )
    object_order = order[object_mask[order]]
    background_indices = order[~object_mask[order]]
    current_owners = {
        int(value) for value in np.unique(entity_ids[object_mask])
    }
    missing_owners = current_owners - set(registry.entities)
    if missing_owners:
        raise ValueError("current mesh owner is missing from the registry")

    ordered_owners = entity_ids[object_order]
    ordered_semantics = semantic_ids[object_order]
    group_starts = np.empty(0, dtype=np.int64)
    group_stops = np.empty(0, dtype=np.int64)
    if len(object_order):
        group_changes = np.empty(len(object_order), dtype=bool)
        group_changes[0] = True
        group_changes[1:] = (ordered_owners[1:] != ordered_owners[:-1]) | (
            ordered_semantics[1:] != ordered_semantics[:-1]
        )
        group_starts = np.flatnonzero(group_changes)
        group_stops = np.concatenate(
            (group_starts[1:], np.asarray([len(object_order)], dtype=np.int64))
        )

    entities: list[EntityPrediction] = []
    for start, stop in zip(group_starts, group_stops, strict=True):
        indices = object_order[start:stop]
        owner_id = int(ordered_owners[start])
        semantic_id = int(ordered_semantics[start])
        owner = registry.entities[owner_id]
        first_frame = _integer(owner.first_frame_id, "entity first_frame_id", minimum=0)
        last_frame = _integer(owner.last_frame_id, "entity last_frame_id", minimum=0)
        if first_frame > last_frame or last_frame >= len(frame_timestamps):
            raise ValueError("entity frame range is outside timestamp_ns_by_frame")
        confidences = np.asarray(mesh.semantic_confidence[indices], dtype=np.float64)
        entities.append(
            EntityPrediction(
                entity_id=f"oviv2:{owner_id}:semantic:{semantic_id}",
                points_xyz=np.asarray(mesh.vertices_xyz[indices], dtype=np.float32),
                semantic_embedding=None,
                semantic_label=names[semantic_id],
                semantic_score=float(np.mean(confidences)),
                lifecycle_state="active",
                first_seen=float(frame_timestamps[first_frame]),
                last_seen=float(frame_timestamps[last_frame]),
                metadata={
                    "entity_type": "object",
                    "owner_entity_id": owner_id,
                    "semantic_id": semantic_id,
                },
            )
        )
    background = np.asarray(
        mesh.vertices_xyz[background_indices],
        dtype=np.float32,
    ).reshape((-1, 3))
    return MapSnapshot(
        method="OVIV2",
        scene_id=snapshot.metadata.scene_id,
        timestamp=float(current_timestamp_ns),
        entities=entities,
        background_xyz=background,
        scope="current",
    )
