"""Export authoritative OVIOVO state without depending on dense visualization caches."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

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


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"metadata value is not JSON serializable: {type(value).__name__}")


def _atomic_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_map_snapshot(snapshot: MapSnapshot, scene_output_dir: str | Path) -> dict[str, Path]:
    """Persist one neutral snapshot with arrays in NPZ and metadata in JSONL."""
    scene_dir = Path(scene_output_dir)
    snapshot_dir = scene_dir / "snapshots"
    entity_dir = scene_dir / "entities"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    entity_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{snapshot.timestamp:.6f}_{snapshot.scope}"
    snapshot_path = snapshot_dir / f"{stem}.npz"
    entities_path = entity_dir / f"{stem}.jsonl"

    offsets = [0]
    point_chunks: list[np.ndarray] = []
    arrays: dict[str, np.ndarray] = {}
    records: list[str] = []
    for index, entity in enumerate(snapshot.entities):
        point_chunks.append(entity.points_xyz)
        offsets.append(offsets[-1] + len(entity.points_xyz))
        embedding_key = ""
        if entity.semantic_embedding is not None:
            embedding_key = f"embedding_{index:06d}"
            arrays[embedding_key] = np.asarray(entity.semantic_embedding, dtype=np.float32)
        record = {
            "index": index,
            "entity_id": entity.entity_id,
            "semantic_label": entity.semantic_label,
            "semantic_score": entity.semantic_score,
            "lifecycle_state": entity.lifecycle_state,
            "first_seen": entity.first_seen,
            "last_seen": entity.last_seen,
            "point_start": offsets[-2],
            "point_count": len(entity.points_xyz),
            "embedding_key": embedding_key,
            "metadata": _jsonable(entity.metadata),
        }
        records.append(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))

    points = (
        np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
        if point_chunks
        else np.empty((0, 3), dtype=np.float32)
    )
    background_present = snapshot.background_xyz is not None
    background = (
        np.asarray(snapshot.background_xyz, dtype=np.float32)
        if background_present
        else np.empty((0, 3), dtype=np.float32)
    )
    arrays.update(
        {
            "entity_points": points,
            "entity_point_offsets": np.asarray(offsets, dtype=np.int64),
            "background_xyz": background,
            "background_present": np.asarray(int(background_present), dtype=np.uint8),
            "method": np.asarray(snapshot.method),
            "scene_id": np.asarray(snapshot.scene_id),
            "timestamp": np.asarray(snapshot.timestamp, dtype=np.float64),
            "scope": np.asarray(snapshot.scope),
            "runtime_json": np.asarray(
                json.dumps(snapshot.runtime, sort_keys=True, separators=(",", ":"), allow_nan=False)
            ),
        }
    )
    _atomic_npz(snapshot_path, arrays)
    _atomic_text(entities_path, "\n".join(records) + ("\n" if records else ""))
    return {"snapshot": snapshot_path, "entities": entities_path}


def read_map_snapshot(snapshot_path: str | Path, entities_path: str | Path) -> MapSnapshot:
    """Load a snapshot written by write_map_snapshot without pickle support."""
    snapshot_path = Path(snapshot_path)
    entities_path = Path(entities_path)
    records = [
        json.loads(line)
        for line in entities_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with np.load(snapshot_path, allow_pickle=False) as arrays:
        points = np.asarray(arrays["entity_points"], dtype=np.float32)
        offsets = np.asarray(arrays["entity_point_offsets"], dtype=np.int64)
        if len(offsets) != len(records) + 1 or offsets[0] != 0 or offsets[-1] != len(points):
            raise ValueError("entity point offsets do not match metadata records")
        entities: list[EntityPrediction] = []
        for index, record in enumerate(records):
            if int(record.get("index", -1)) != index:
                raise ValueError("entity metadata indices must be contiguous")
            start = int(offsets[index])
            stop = int(offsets[index + 1])
            if int(record.get("point_start", -1)) != start or int(record.get("point_count", -1)) != stop - start:
                raise ValueError("entity point metadata does not match NPZ offsets")
            embedding_key = str(record.get("embedding_key", ""))
            embedding = np.asarray(arrays[embedding_key], dtype=np.float32) if embedding_key else None
            entities.append(
                EntityPrediction(
                    entity_id=str(record["entity_id"]),
                    points_xyz=points[start:stop],
                    semantic_embedding=embedding,
                    semantic_label=record.get("semantic_label"),
                    semantic_score=float(record["semantic_score"]),
                    lifecycle_state=str(record["lifecycle_state"]),
                    first_seen=float(record["first_seen"]),
                    last_seen=float(record["last_seen"]),
                    metadata=dict(record.get("metadata", {})),
                )
            )
        background_present = bool(np.asarray(arrays["background_present"]).item())
        background = np.asarray(arrays["background_xyz"], dtype=np.float32) if background_present else None
        return MapSnapshot(
            method=str(np.asarray(arrays["method"]).item()),
            scene_id=str(np.asarray(arrays["scene_id"]).item()),
            timestamp=float(np.asarray(arrays["timestamp"]).item()),
            entities=entities,
            background_xyz=background,
            scope=str(np.asarray(arrays["scope"]).item()),
            runtime=json.loads(str(np.asarray(arrays["runtime_json"]).item())),
        )
