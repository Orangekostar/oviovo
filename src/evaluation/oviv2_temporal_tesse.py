from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.temporal_lifecycle import TemporalLifecycle
from src.oviv2.temporal_snapshot import (
    TemporalCheckpointPublicationUncertainError,
    TemporalCurrentSnapshot,
    TemporalSnapshotMetadata,
    _DirectoryWitness,
    _canonical_json,
    _canonical_npz,
    _close_best_effort,
    _fingerprint,
    _publish_new_directory,
    _read_npy_contract,
    _reject_symlink_components,
    _sha256,
    _strict_object,
    _strict_json,
    build_temporal_map_snapshot,
)


TEMPORAL_CURRENT_FORMAT = "oviv2_temporal_current_checkpoint"
_INVENTORY = frozenset({"manifest.json", "snapshot.npz", "entities.jsonl", "diagnostics.json", "checksums.json"})
_ARRAY_NAMES = ("entity_points", "entity_point_offsets", "background_xyz", "background_present")
_MAX_FULL_MEMBER_BYTES = 512 * 1024 * 1024
_MAX_FULL_JSON_BYTES = 64 * 1024 * 1024
_MAX_FULL_ENTITY_RECORD_BYTES = 1024 * 1024
_MAX_FULL_ENTITIES_JSON_BYTES = 64 * 1024 * 1024


def _full_capacities(snapshot: TemporalCurrentSnapshot) -> tuple[int, int, int, int]:
    config = snapshot.background.config
    maximum_entities = config.maximum_entities
    maximum_entity_points = maximum_entities * config.maximum_object_voxels
    maximum_background_points = config.background_block_count * 8**3 * 8
    raw = (
        maximum_entity_points * 3 * np.dtype(np.float32).itemsize
        + (maximum_entities + 1) * np.dtype(np.int64).itemsize
        + maximum_background_points * 3 * np.dtype(np.float32).itemsize
        + 1
    )
    serialized_byte_limit = min(_MAX_FULL_MEMBER_BYTES, raw * 2 + 256 * 1024)
    return maximum_entities, maximum_entity_points, maximum_background_points, serialized_byte_limit


def _positive_exact_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric")
    result = float(value)
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _strict_full_json(content: bytes, *, label: str, max_bytes: int) -> dict[str, Any]:
    if len(content) > max_bytes:
        raise ValueError(f"{label} exceeds size limit")
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


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
        record = _canonical_json({
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
        }).rstrip(b"\n")
        if len(record) > _MAX_FULL_ENTITY_RECORD_BYTES:
            raise ValueError("full entity record exceeds size limit")
        records.append(record)
    entity_points = np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.empty((0, 3), dtype=np.float32)
    background_present = snapshot.background_xyz is not None
    background = np.asarray(snapshot.background_xyz, dtype=np.float32) if background_present else np.empty((0, 3), dtype=np.float32)
    archive = _canonical_npz({
        "entity_points": entity_points,
        "entity_point_offsets": np.asarray(offsets, dtype=np.int64),
        "background_xyz": background,
        "background_present": np.asarray([int(background_present)], dtype=np.uint8),
    }, _ARRAY_NAMES)
    entities_json = b"\n".join(records) + (b"\n" if records else b"")
    if len(entities_json) > _MAX_FULL_ENTITIES_JSON_BYTES:
        raise ValueError("full entities JSON exceeds total size limit")
    return archive, entities_json


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
    capacities = _full_capacities(snapshot)
    maximum_entities, maximum_entity_points, maximum_background_points, serialized_byte_limit = capacities
    if len(neutral.entities) > maximum_entities:
        raise ValueError("full entity capacity exceeded")
    if sum(len(entity.points_xyz) for entity in neutral.entities) > maximum_entity_points:
        raise ValueError("full entity point capacity exceeded")
    if neutral.background_xyz is not None and len(neutral.background_xyz) > maximum_background_points:
        raise ValueError("full background point capacity exceeded")
    snapshot_bytes, entities_bytes = _serialize_map_snapshot(neutral)
    if len(snapshot_bytes) > serialized_byte_limit:
        raise ValueError("full snapshot exceeds serialized byte limit")
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
        "maximum_entities": maximum_entities,
        "maximum_entity_points": maximum_entity_points,
        "maximum_background_points": maximum_background_points,
        "serialized_byte_limit": serialized_byte_limit,
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


def _read_checkpoint_directory(source: Path) -> dict[str, bytes]:
    descriptor = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        directory_before = os.fstat(descriptor)
        if not stat.S_ISDIR(directory_before.st_mode) or set(os.listdir(descriptor)) != _INVENTORY:
            raise ValueError("temporal current checkpoint inventory is invalid")
        contents: dict[str, bytes] = {}
        for name in sorted(_INVENTORY):
            named_before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(named_before.st_mode):
                raise ValueError("temporal current checkpoint member is not regular")
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=descriptor)
            try:
                opened_before = os.fstat(fd)
                if _fingerprint(opened_before) != _fingerprint(named_before):
                    raise ValueError("temporal current checkpoint member identity changed")
                limit = _MAX_FULL_MEMBER_BYTES if name == "snapshot.npz" else _MAX_FULL_JSON_BYTES
                chunks = []
                total = 0
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise ValueError("temporal current checkpoint member exceeds size limit")
                    chunks.append(chunk)
                opened_after = os.fstat(fd)
                named_after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if _fingerprint(opened_before) != _fingerprint(opened_after) or _fingerprint(opened_after) != _fingerprint(named_after):
                    raise ValueError("temporal current checkpoint member changed while reading")
                contents[name] = b"".join(chunks)
            finally:
                _close_best_effort(fd)
        if _fingerprint(directory_before) != _fingerprint(os.fstat(descriptor)):
            raise ValueError("temporal current checkpoint directory changed while reading")
        return contents
    finally:
        _close_best_effort(descriptor)


def _validate_manifest(payload: dict[str, Any]) -> tuple[TemporalSnapshotMetadata, tuple[str, ...], tuple[int, int, int, int]]:
    required = {
        "format", "schema_version", "scene_id", "consumed_through_frame", "consumed_through_timestamp",
        "revision", "voxel_size_m", "config_sha256", "code_commit", "input_sha256", "method", "scope", "class_names",
        "maximum_entities", "maximum_entity_points", "maximum_background_points", "serialized_byte_limit",
    }
    if set(payload) != required or payload["format"] != TEMPORAL_CURRENT_FORMAT:
        raise ValueError("temporal current manifest schema is invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    metadata = TemporalSnapshotMetadata(
        scene_id=payload["scene_id"],
        frame_id=payload["consumed_through_frame"],
        timestamp=payload["consumed_through_timestamp"],
        revision=payload["revision"],
        voxel_size_m=payload["voxel_size_m"],
        config_sha256=payload["config_sha256"],
    )
    _hex_digest(payload["code_commit"], "code_commit", {40, 64})
    _hex_digest(payload["input_sha256"], "input_sha256", {64})
    if payload["method"] != "OVIV2-temporal":
        raise ValueError("method must be OVIV2-temporal")
    if payload["scope"] != "current":
        raise ValueError("scope must be current")
    raw_names = payload["class_names"]
    if not isinstance(raw_names, list) or not raw_names or any(type(item) is not str or not item.strip() for item in raw_names):
        raise ValueError("class_names must be a non-empty list of strings")
    class_names = tuple(raw_names)
    capacities = (
        _positive_exact_int(payload["maximum_entities"], "maximum_entities"),
        _positive_exact_int(payload["maximum_entity_points"], "maximum_entity_points"),
        _positive_exact_int(payload["maximum_background_points"], "maximum_background_points"),
        _positive_exact_int(payload["serialized_byte_limit"], "serialized_byte_limit"),
    )
    maximum_entities, maximum_entity_points, maximum_background_points, serialized_byte_limit = capacities
    if maximum_entity_points < maximum_entities or serialized_byte_limit > _MAX_FULL_MEMBER_BYTES:
        raise ValueError("full artifact capacity metadata is invalid")
    raw = maximum_entity_points * 12 + (maximum_entities + 1) * 8 + maximum_background_points * 12 + 1
    if serialized_byte_limit != min(_MAX_FULL_MEMBER_BYTES, raw * 2 + 256 * 1024):
        raise ValueError("serialized_byte_limit does not match full capacities")
    return metadata, class_names, capacities


def _preflight_full_archive(content: bytes, capacities: tuple[int, int, int, int]) -> None:
    maximum_entities, maximum_entity_points, maximum_background_points, serialized_byte_limit = capacities
    if len(content) > serialized_byte_limit:
        raise ValueError("full archive exceeds serialized capacity limit")
    try:
        archive_context = zipfile.ZipFile(io.BytesIO(content), "r")
    except zipfile.BadZipFile as exc:
        raise ValueError("full archive is invalid") from exc
    with archive_context as archive:
        infos = archive.infolist()
        expected = {f"{name}.npy" for name in _ARRAY_NAMES}
        counts = Counter(info.filename for info in infos)
        if set(counts) != expected or any(count != 1 for count in counts.values()) or len(infos) != len(expected):
            raise ValueError("full archive member inventory or duplicate member is invalid")
        by_name: dict[str, zipfile.ZipInfo] = {}
        total_uncompressed = 0
        for info in infos:
            if (
                info.is_dir() or "/" in info.filename or "\\" in info.filename
                or info.flag_bits & 0x1
                or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            ):
                raise ValueError("full archive member is unsafe")
            total_uncompressed += info.file_size
            if info.file_size > serialized_byte_limit or info.compress_size > serialized_byte_limit:
                raise ValueError("full archive member exceeds capacity limit")
            by_name[info.filename.removesuffix(".npy")] = info
        raw_limit = maximum_entity_points * 12 + (maximum_entities + 1) * 8 + maximum_background_points * 12 + 1 + 64 * 1024
        if total_uncompressed > raw_limit:
            raise ValueError("full archive uncompressed content exceeds capacity limit")
        contracts = {name: _read_npy_contract(archive, by_name[name], name) for name in _ARRAY_NAMES}
        entity_shape, entity_dtype = contracts["entity_points"]
        if entity_dtype != np.dtype(np.float32) or len(entity_shape) != 2 or entity_shape[1:] != (3,) or entity_shape[0] > maximum_entity_points:
            raise ValueError("entity_points dtype, shape, or capacity is invalid")
        offsets_shape, offsets_dtype = contracts["entity_point_offsets"]
        if offsets_dtype != np.dtype(np.int64) or len(offsets_shape) != 1 or not 1 <= offsets_shape[0] <= maximum_entities + 1:
            raise ValueError("entity_point_offsets dtype, shape, or capacity is invalid")
        background_shape, background_dtype = contracts["background_xyz"]
        if background_dtype != np.dtype(np.float32) or len(background_shape) != 2 or background_shape[1:] != (3,) or background_shape[0] > maximum_background_points:
            raise ValueError("background_xyz dtype, shape, or capacity is invalid")
        if contracts["background_present"] != ((1,), np.dtype(np.uint8)):
            raise ValueError("background_present dtype or shape is invalid")


@dataclass(frozen=True)
class LoadedTemporalCurrentCheckpoint:
    snapshot: MapSnapshot
    diagnostics: dict[str, Any]
    source_witness: _DirectoryWitness

    def __iter__(self):
        yield self.snapshot
        yield self.diagnostics

    def revalidate_source(self) -> None:
        self.source_witness.revalidate()


def load_temporal_current_checkpoint(checkpoint_dir: str | Path) -> LoadedTemporalCurrentCheckpoint:
    source = Path(os.path.abspath(os.fspath(checkpoint_dir)))
    _reject_symlink_components(source, label="temporal current checkpoint source")
    witness = _DirectoryWitness.capture(source, _INVENTORY)
    contents = _read_checkpoint_directory(source)
    checksums = _strict_json(contents["checksums.json"], label="checksums")
    if set(checksums) != _INVENTORY - {"checksums.json"}:
        raise ValueError("temporal current checksum inventory is invalid")
    for name, digest in checksums.items():
        if not isinstance(digest, str) or _sha256(contents[name]) != digest:
            raise ValueError(f"temporal current checksum mismatch for {name}")
    manifest = _strict_json(contents["manifest.json"], label="manifest")
    metadata, class_names, capacities = _validate_manifest(manifest)
    entities_content = contents["entities.jsonl"]
    if len(entities_content) > _MAX_FULL_ENTITIES_JSON_BYTES:
        raise ValueError("full entities JSON exceeds total size limit")
    records = []
    for line in entities_content.splitlines():
        records.append(
            _strict_full_json(
                line,
                label="entity record",
                max_bytes=_MAX_FULL_ENTITY_RECORD_BYTES,
            )
        )
    if len(records) > capacities[0]:
        raise ValueError("entity records exceed full capacity")
    _preflight_full_archive(contents["snapshot.npz"], capacities)
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
        if type(record["index"]) is not int or type(record["point_start"]) is not int or type(record["point_count"]) is not int:
            raise ValueError("entity record integer fields are invalid")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if record["point_start"] != start or record["point_count"] != stop - start or record["point_count"] < 0:
            raise ValueError("entity record point range is invalid")
        entity_id = str(record["entity_id"])
        entity_metadata = record["metadata"]
        if not isinstance(entity_metadata, dict) or set(entity_metadata) != {"temporal_entity_id", "semantic_id"}:
            raise ValueError("entity metadata schema is invalid")
        temporal_id = entity_metadata["temporal_entity_id"]
        if type(temporal_id) is not int or temporal_id < 0 or entity_id != f"temporal:{temporal_id}":
            raise ValueError("entity temporal ID is invalid")
        if temporal_id <= previous_temporal_id:
            raise ValueError("entity records are not strictly ordered")
        previous_temporal_id = temporal_id
        semantic_id = entity_metadata["semantic_id"]
        if type(semantic_id) is not int or semantic_id < 0:
            raise ValueError("entity semantic ID is invalid")
        semantic_label = record["semantic_label"]
        if semantic_id < len(class_names):
            if semantic_label != class_names[semantic_id]:
                raise ValueError("entity semantic label does not match class_names")
        elif semantic_label is not None:
            raise ValueError("out-of-vocabulary entity semantic label must be null")
        semantic_score = _finite_number(record["semantic_score"], "entity semantic_score", nonnegative=True)
        if semantic_score > 1.0:
            raise ValueError("entity semantic_score must not exceed one")
        if record["lifecycle_state"] not in {"active", "uncertain"}:
            raise ValueError("entity lifecycle_state is invalid")
        first_seen = _finite_number(record["first_seen"], "entity first_seen", nonnegative=True)
        last_seen = _finite_number(record["last_seen"], "entity last_seen", nonnegative=True)
        if first_seen > last_seen or last_seen > metadata.frame_id:
            raise ValueError("entity record is later than checkpoint")
        embedding = record["semantic_embedding"]
        if embedding is not None:
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("entity semantic_embedding must be a non-empty list or null")
            embedding_array = np.asarray(embedding)
            if embedding_array.ndim != 1 or embedding_array.dtype.kind not in "iuf" or not np.isfinite(embedding_array).all():
                raise ValueError("entity semantic_embedding is invalid")
        else:
            embedding_array = None
        if not np.isfinite(points[start:stop]).all():
            raise ValueError("entity points must be finite")
        entities.append(EntityPrediction(
            entity_id=entity_id,
            points_xyz=points[start:stop],
            semantic_embedding=None if embedding_array is None else embedding_array.astype(np.float32),
            semantic_label=semantic_label,
            semantic_score=semantic_score,
            lifecycle_state=record["lifecycle_state"],
            first_seen=first_seen,
            last_seen=last_seen,
            metadata=entity_metadata,
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
                or record["last_seen_frame_id"] > metadata.frame_id
                or record["lifecycle_frame_id"] > metadata.frame_id
                or not isinstance(record["lifecycle_timestamp"], (int, float))
                or isinstance(record["lifecycle_timestamp"], bool)
                or not np.isfinite(record["lifecycle_timestamp"])
                or record["lifecycle_timestamp"] > metadata.timestamp
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
    uncertain_entity_ids = {
        entity.metadata["temporal_entity_id"]
        for entity in entities
        if entity.lifecycle_state == "uncertain"
    }
    uncertain_diagnostic_ids = {record["entity_id"] for record in diagnostics["uncertain"]}
    dormant_diagnostic_ids = {record["entity_id"] for record in diagnostics["dormant"]}
    if uncertain_entity_ids != uncertain_diagnostic_ids or dormant_diagnostic_ids & {entity.metadata["temporal_entity_id"] for entity in entities}:
        raise ValueError("diagnostics are inconsistent with current entities")
    prediction = MapSnapshot(
        method=manifest["method"], scene_id=metadata.scene_id,
        timestamp=metadata.timestamp, entities=entities,
        background_xyz=background if bool(present[0]) else None, scope=manifest["scope"], runtime={},
    )
    witness.revalidate()
    return LoadedTemporalCurrentCheckpoint(prediction, diagnostics, witness)


__all__ = [
    "TemporalCurrentCheckpointReceipt",
    "LoadedTemporalCurrentCheckpoint",
    "TemporalCheckpointPublicationUncertainError",
    "publish_temporal_current_checkpoint",
    "load_temporal_current_checkpoint",
]
