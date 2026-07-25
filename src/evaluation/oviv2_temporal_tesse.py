from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.temporal_lifecycle import TemporalLifecycle
from src.oviv2.temporal_snapshot import (
    TemporalCurrentSnapshot,
    _DirectoryWitness,
    _canonical_json,
    _canonical_npz,
    _publish_new_directory,
    _reject_symlink_components,
    _sha256,
    _strict_json,
    build_temporal_map_snapshot,
)


TEMPORAL_CURRENT_FORMAT = "oviv2_temporal_current_checkpoint"
_INVENTORY = frozenset({"manifest.json", "snapshot.npz", "entities.jsonl", "diagnostics.json", "checksums.json"})
_ARRAY_NAMES = ("entity_points", "entity_point_offsets", "background_xyz", "background_present")


def _hex_digest(value: object, name: str, lengths: set[int]) -> str:
    if not isinstance(value, str) or len(value) not in lengths or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return value


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("metadata values must be finite")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _serialize_map_snapshot(snapshot: MapSnapshot) -> tuple[bytes, bytes]:
    offsets = [0]
    chunks: list[np.ndarray] = []
    records: list[bytes] = []
    for index, entity in enumerate(snapshot.entities):
        points = np.asarray(entity.points_xyz, dtype=np.float32)
        chunks.append(points)
        offsets.append(offsets[-1] + len(points))
        records.append(_canonical_json({
            "index": index,
            "entity_id": entity.entity_id,
            "semantic_embedding": None if entity.semantic_embedding is None else _jsonable(entity.semantic_embedding),
            "semantic_label": entity.semantic_label,
            "semantic_score": entity.semantic_score,
            "lifecycle_state": entity.lifecycle_state,
            "first_seen": entity.first_seen,
            "last_seen": entity.last_seen,
            "point_start": offsets[-2],
            "point_count": len(points),
            "metadata": _jsonable(entity.metadata),
        }).rstrip(b"\n"))
    entity_points = np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.empty((0, 3), dtype=np.float32)
    background_present = snapshot.background_xyz is not None
    background = np.asarray(snapshot.background_xyz, dtype=np.float32) if background_present else np.empty((0, 3), dtype=np.float32)
    archive = _canonical_npz({
        "entity_points": entity_points,
        "entity_point_offsets": np.asarray(offsets, dtype=np.int64),
        "background_xyz": background,
        "background_present": np.asarray([int(background_present)], dtype=np.uint8),
    }, _ARRAY_NAMES)
    return archive, b"\n".join(records) + (b"\n" if records else b"")


def _diagnostics(snapshot: TemporalCurrentSnapshot) -> bytes:
    output: dict[str, list[dict[str, Any]]] = {"dormant": [], "uncertain": []}
    for entity in snapshot.entities:
        lifecycle = entity.lifecycle.lifecycle
        if lifecycle not in {TemporalLifecycle.DORMANT, TemporalLifecycle.UNCERTAIN}:
            continue
        output[lifecycle.value].append({
            "entity_id": entity.lifecycle.entity_id,
            "existence_log_odds": entity.lifecycle.existence_log_odds,
            "absent_streak": entity.lifecycle.absent_streak,
            "distinct_view_bin_count": len(entity.lifecycle.absence_view_bins),
            "first_seen_frame_id": entity.first_seen_frame_id,
            "last_seen_frame_id": entity.last_seen_frame_id,
            "lifecycle_frame_id": entity.lifecycle.last_frame_id,
            "lifecycle_timestamp": entity.lifecycle.last_timestamp,
        })
    return _canonical_json(output)


@dataclass(frozen=True)
class TemporalCurrentCheckpointReceipt:
    path: Path
    content_sha256: str
    source_witness: _DirectoryWitness

    def revalidate_source(self) -> None:
        self.source_witness.revalidate()


def publish_temporal_current_checkpoint(
    target_dir: str | Path,
    snapshot: TemporalCurrentSnapshot,
    class_names: Sequence[str],
    *,
    code_commit: str,
    input_sha256: str,
) -> TemporalCurrentCheckpointReceipt:
    if not isinstance(snapshot, TemporalCurrentSnapshot):
        raise TypeError("snapshot must be TemporalCurrentSnapshot")
    code_commit = _hex_digest(code_commit, "code_commit", {40, 64})
    input_sha256 = _hex_digest(input_sha256, "input_sha256", {64})
    neutral = build_temporal_map_snapshot(snapshot, class_names)
    snapshot_bytes, entities_bytes = _serialize_map_snapshot(neutral)
    diagnostics_bytes = _diagnostics(snapshot)
    manifest_bytes = _canonical_json({
        "format": TEMPORAL_CURRENT_FORMAT,
        "schema_version": 1,
        "scene_id": snapshot.metadata.scene_id,
        "consumed_through_frame": snapshot.metadata.frame_id,
        "consumed_through_timestamp": snapshot.metadata.timestamp,
        "revision": snapshot.metadata.revision,
        "voxel_size_m": snapshot.metadata.voxel_size_m,
        "config_sha256": snapshot.metadata.config_sha256,
        "code_commit": code_commit,
        "input_sha256": input_sha256,
        "method": neutral.method,
        "scope": neutral.scope,
        "class_names": [str(item) for item in class_names],
    })
    data = {
        "manifest.json": manifest_bytes,
        "snapshot.npz": snapshot_bytes,
        "entities.jsonl": entities_bytes,
        "diagnostics.json": diagnostics_bytes,
    }
    checksums = {name: _sha256(content) for name, content in sorted(data.items())}
    files = {**data, "checksums.json": _canonical_json(checksums)}
    witness = _publish_new_directory(target_dir, files, _INVENTORY)
    content_sha256 = hashlib.sha256(_canonical_json(checksums)).hexdigest()
    return TemporalCurrentCheckpointReceipt(witness.path, content_sha256, witness)


def load_temporal_current_checkpoint(checkpoint_dir: str | Path) -> tuple[MapSnapshot, dict[str, Any]]:
    source = Path(os.path.abspath(os.fspath(checkpoint_dir)))
    _reject_symlink_components(source, label="temporal current checkpoint source")
    descriptor = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if set(os.listdir(descriptor)) != _INVENTORY:
            raise ValueError("temporal current checkpoint inventory is invalid")
        contents: dict[str, bytes] = {}
        for name in sorted(_INVENTORY):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                chunks = []
                total = 0
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 512 * 1024 * 1024:
                        raise ValueError("temporal current checkpoint member exceeds size limit")
                    chunks.append(chunk)
                contents[name] = b"".join(chunks)
            finally:
                os.close(fd)
    finally:
        os.close(descriptor)
    checksums = _strict_json(contents["checksums.json"], label="checksums")
    if set(checksums) != _INVENTORY - {"checksums.json"}:
        raise ValueError("temporal current checksum inventory is invalid")
    for name, digest in checksums.items():
        if not isinstance(digest, str) or _sha256(contents[name]) != digest:
            raise ValueError(f"temporal current checksum mismatch for {name}")
    manifest = _strict_json(contents["manifest.json"], label="manifest")
    required = {
        "format", "schema_version", "scene_id", "consumed_through_frame", "consumed_through_timestamp",
        "revision", "voxel_size_m", "config_sha256", "code_commit", "input_sha256", "method", "scope", "class_names",
    }
    if set(manifest) != required or manifest["format"] != TEMPORAL_CURRENT_FORMAT or manifest["schema_version"] != 1:
        raise ValueError("temporal current manifest schema is invalid")
    records = []
    for line in contents["entities.jsonl"].splitlines():
        records.append(_strict_json(line, label="entity record"))
    with np.load(io.BytesIO(contents["snapshot.npz"]), allow_pickle=False) as arrays:
        if set(arrays.files) != set(_ARRAY_NAMES):
            raise ValueError("temporal current array inventory is invalid")
        points = np.array(arrays["entity_points"], copy=True)
        offsets = np.array(arrays["entity_point_offsets"], copy=True)
        background = np.array(arrays["background_xyz"], copy=True)
        present = np.array(arrays["background_present"], copy=True)
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("entity_points dtype or shape is invalid")
    if offsets.dtype != np.int64 or offsets.shape != (len(records) + 1,) or offsets[0] != 0 or offsets[-1] != len(points) or np.any(np.diff(offsets) < 0):
        raise ValueError("entity offsets are invalid")
    if background.dtype != np.float32 or background.ndim != 2 or background.shape[1:] != (3,):
        raise ValueError("background dtype or shape is invalid")
    if present.dtype != np.uint8 or present.shape != (1,) or present[0] not in (0, 1):
        raise ValueError("background_present is invalid")
    entities = []
    previous_temporal_id = -1
    for index, record in enumerate(records):
        expected_fields = {
            "index", "entity_id", "semantic_embedding", "semantic_label", "semantic_score", "lifecycle_state",
            "first_seen", "last_seen", "point_start", "point_count", "metadata",
        }
        if set(record) != expected_fields or record["index"] != index:
            raise ValueError("entity record schema or order is invalid")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if record["point_start"] != start or record["point_count"] != stop - start:
            raise ValueError("entity record point range is invalid")
        entity_id = str(record["entity_id"])
        metadata = record["metadata"]
        if not isinstance(metadata, dict) or set(metadata) != {"temporal_entity_id", "semantic_id"}:
            raise ValueError("entity metadata schema is invalid")
        temporal_id = metadata["temporal_entity_id"]
        if type(temporal_id) is not int or temporal_id < 0 or entity_id != f"temporal:{temporal_id}":
            raise ValueError("entity temporal ID is invalid")
        if temporal_id <= previous_temporal_id:
            raise ValueError("entity records are not strictly ordered")
        previous_temporal_id = temporal_id
        if float(record["last_seen"]) > float(manifest["consumed_through_frame"]):
            raise ValueError("entity record is later than checkpoint")
        embedding = record["semantic_embedding"]
        entities.append(EntityPrediction(
            entity_id=entity_id,
            points_xyz=points[start:stop],
            semantic_embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
            semantic_label=record["semantic_label"],
            semantic_score=record["semantic_score"],
            lifecycle_state=record["lifecycle_state"],
            first_seen=record["first_seen"],
            last_seen=record["last_seen"],
            metadata=metadata,
        ))
    diagnostics = _strict_json(contents["diagnostics.json"], label="diagnostics")
    if set(diagnostics) != {"uncertain", "dormant"}:
        raise ValueError("diagnostics schema is invalid")
    diagnostic_fields = {
        "entity_id", "existence_log_odds", "absent_streak", "distinct_view_bin_count",
        "first_seen_frame_id", "last_seen_frame_id", "lifecycle_frame_id", "lifecycle_timestamp",
    }
    for lifecycle in ("uncertain", "dormant"):
        records_for_lifecycle = diagnostics[lifecycle]
        if not isinstance(records_for_lifecycle, list):
            raise ValueError("diagnostics lifecycle value must be a list")
        previous = -1
        for record in records_for_lifecycle:
            if not isinstance(record, dict) or set(record) != diagnostic_fields:
                raise ValueError("diagnostics record schema is invalid")
            entity_id = record["entity_id"]
            if type(entity_id) is not int or entity_id <= previous:
                raise ValueError("diagnostics records are not strictly ordered")
            previous = entity_id
            if (
                type(record["first_seen_frame_id"]) is not int
                or type(record["last_seen_frame_id"]) is not int
                or type(record["lifecycle_frame_id"]) is not int
                or record["first_seen_frame_id"] < 0
                or record["first_seen_frame_id"] > record["last_seen_frame_id"]
                or record["last_seen_frame_id"] > manifest["consumed_through_frame"]
                or record["lifecycle_frame_id"] > manifest["consumed_through_frame"]
                or not isinstance(record["lifecycle_timestamp"], (int, float))
                or isinstance(record["lifecycle_timestamp"], bool)
                or not np.isfinite(record["lifecycle_timestamp"])
                or record["lifecycle_timestamp"] > manifest["consumed_through_timestamp"]
            ):
                raise ValueError("diagnostics record is later than checkpoint or invalid")
            for counter in ("absent_streak", "distinct_view_bin_count"):
                if type(record[counter]) is not int or record[counter] < 0:
                    raise ValueError("diagnostics evidence counter is invalid")
            if (
                isinstance(record["existence_log_odds"], bool)
                or not isinstance(record["existence_log_odds"], (int, float))
                or not np.isfinite(record["existence_log_odds"])
            ):
                raise ValueError("diagnostics existence_log_odds is invalid")
    prediction = MapSnapshot(
        method=manifest["method"], scene_id=manifest["scene_id"],
        timestamp=manifest["consumed_through_timestamp"], entities=entities,
        background_xyz=background if bool(present[0]) else None, scope=manifest["scope"], runtime={},
    )
    return prediction, diagnostics


__all__ = [
    "TemporalCurrentCheckpointReceipt",
    "publish_temporal_current_checkpoint",
    "load_temporal_current_checkpoint",
]
