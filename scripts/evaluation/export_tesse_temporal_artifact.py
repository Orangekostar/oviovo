#!/usr/bin/env python3
"""Assemble hash-bound causal TESSE-CD neutral checkpoint artifacts."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.exporters.oviovo import read_map_snapshot


_SOURCE_RECORD_FIELDS = frozenset({"path", "sha256", "byte_count"})
_SOURCE_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "dataset",
        "mode",
        "method",
        "scene",
        "schedule",
        "capture_status",
        "trajectories",
        "checkpoints",
    }
)
_CAPTURE_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "scene",
        "mode",
        "scheduled_frame_indices",
        "captured_frame_indices",
        "schedule",
        "trajectories",
        "checkpoint_statuses",
    }
)
_CHECKPOINT_INDEX_FIELDS = frozenset(
    {
        "frame_index",
        "timestamp_ns",
        "consumed_through_frame",
        "consumed_through_frame_exclusive",
        "checkpoint_status",
        "snapshot",
        "entities",
    }
)
_SCHEDULE_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "dataset",
        "method_predictions_used",
        "parameters",
        "scenes",
    }
)
_SCHEDULE_FIELDS_WITH_SOURCE = _SCHEDULE_FIELDS | {"source_manifest"}
_SCHEDULE_PARAMETER_FIELDS = frozenset(
    {
        "frame_indexing",
        "official_stride_frames",
        "common_event_step_frames",
        "common_event_horizon_frames",
        "common_checkpoints_per_event",
        "event_frame_rule",
    }
)
_SCHEDULE_SCENE_FIELDS = frozenset({"frame_count", "entries"})
_SCHEDULE_SCENE_FIELDS_WITH_SOURCES = _SCHEDULE_SCENE_FIELDS | {
    "events",
    "first_depth_timestamp_ns",
    "last_depth_timestamp_ns",
    "sources",
}
_SCHEDULE_ENTRY_FIELDS = frozenset({"frame_index", "timestamp_ns"})
_SCHEDULE_ENTRY_FIELDS_FULL = _SCHEDULE_ENTRY_FIELDS | {
    "relative_timestamp_ns",
    "event_ids",
    "roles",
}
_SCHEDULE_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "event_relative_timestamp_ns",
        "intervention_frame_index",
        "intervention_timestamp_ns",
        "intervention_relative_timestamp_ns",
        "common_checkpoint_frame_indices",
    }
)
_GENERIC_CHECKPOINT_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "checkpoint_frame",
        "timestamp_ns",
        "consumed_through_frame",
        "consumed_through_frame_exclusive",
    }
)
_PANOPTIC_CHECKPOINT_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "frame_index",
        "source_timestamp_ns",
        "consumed_through_frame",
        "consumed_through_frame_exclusive",
        "panmap_file",
    }
)


@dataclass(frozen=True)
class _VerifiedSource:
    path: Path
    sha256: str
    byte_count: int
    fingerprint: tuple[int, int, int, int]

def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_unchanged(source: _VerifiedSource, *, label: str) -> None:
    try:
        observed = _fingerprint(source.path)
    except OSError as error:
        raise ValueError(f"{label} source changed while reading") from error
    if observed != source.fingerprint:
        raise ValueError(f"{label} source changed while reading")


def _declared_source(
    record: Mapping[str, Any], *, base: Path, label: str
) -> _VerifiedSource:
    if set(record) != _SOURCE_RECORD_FIELDS:
        raise ValueError(f"{label} source record fields are invalid")
    raw_path = Path(str(record.get("path", "")))
    path = raw_path if raw_path.is_absolute() else base / raw_path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} source is not a file: {path}")
    declared_bytes = record.get("byte_count")
    if type(declared_bytes) is not int or declared_bytes < 0:
        raise ValueError(f"{label} byte count must be a non-negative integer")
    fingerprint = _fingerprint(path)
    if fingerprint[2] != declared_bytes:
        raise ValueError(f"{label} byte count mismatch")
    observed_hash = _sha256(path)
    source = _VerifiedSource(path, observed_hash, declared_bytes, fingerprint)
    _assert_unchanged(source, label=label)
    if str(record.get("sha256", "")) != observed_hash:
        raise ValueError(f"{label} SHA256 mismatch")
    return source


def _direct_source(path: Path, *, label: str) -> _VerifiedSource:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} source is not a file: {path}")
    fingerprint = _fingerprint(path)
    source = _VerifiedSource(path, _sha256(path), fingerprint[2], fingerprint)
    _assert_unchanged(source, label=label)
    return source


def _read_json(source: _VerifiedSource, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    payload = json.loads(
        source.path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    _assert_unchanged(source, label=label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _same_source(left: Mapping[str, Any], right: _VerifiedSource, *, base: Path) -> bool:
    if set(left) != _SOURCE_RECORD_FIELDS:
        return False
    raw_path = Path(str(left.get("path", "")))
    path = (raw_path if raw_path.is_absolute() else base / raw_path).resolve()
    return (
        path == right.path
        and left.get("sha256") == right.sha256
        and left.get("byte_count") == right.byte_count
    )


def _load_schedule(
    source: _VerifiedSource, *, scene: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, int]]]:
    payload = _read_json(source, label="schedule")
    if (
        payload.get("dataset") != "TESSE-CD"
        or payload.get("manifest_id") != "tesse_cd_causal_schedule_v2"
        or payload.get("method_predictions_used") is not False
    ):
        raise ValueError("causal schedule identity mismatch")
    indexing = payload.get("parameters", {}).get("frame_indexing")
    if indexing is not None and indexing != "zero_based":
        raise ValueError("causal schedule must use zero_based frame indexing")
    scene_payload = payload.get("scenes", {}).get(scene)
    if not isinstance(scene_payload, Mapping):
        raise ValueError(f"causal schedule has no scene: {scene}")
    raw_entries = scene_payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("causal schedule entries must be non-empty")
    frame_count = scene_payload.get("frame_count")
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("causal schedule frame count must be a positive integer")
    entries: list[dict[str, int]] = []
    previous_frame = -1
    previous_timestamp = -1
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("causal schedule entry must be an object")
        frame_index = raw_entry.get("frame_index")
        timestamp_ns = raw_entry.get("timestamp_ns")
        if type(frame_index) is not int or type(timestamp_ns) is not int:
            raise ValueError("causal schedule frame and timestamp must be integers")
        if frame_index <= previous_frame or timestamp_ns <= previous_timestamp:
            raise ValueError("checkpoint frames and timestamps must be strictly increasing")
        if frame_index < 0 or timestamp_ns <= 0:
            raise ValueError("checkpoint frame and timestamp must be positive-domain values")
        if frame_index >= frame_count:
            raise ValueError("checkpoint is outside declared frame count")
        entries.append({"frame_index": frame_index, "timestamp_ns": timestamp_ns})
        previous_frame = frame_index
        previous_timestamp = timestamp_ns
    return payload, dict(scene_payload), entries


def _checkpoint_identity(payload: Mapping[str, Any]) -> tuple[int, int, int, int]:
    frame = payload.get("checkpoint_frame", payload.get("frame_index", -1))
    timestamp = payload.get("timestamp_ns", payload.get("source_timestamp_ns", -1))
    consumed = payload.get("consumed_through_frame", -1)
    consumed_exclusive = payload.get("consumed_through_frame_exclusive", -1)
    if (
        type(frame) is not int
        or type(timestamp) is not int
        or type(consumed) is not int
        or type(consumed_exclusive) is not int
    ):
        raise ValueError("checkpoint identity fields must be integers")
    if (
        frame < 0
        or timestamp <= 0
        or consumed != frame
        or consumed_exclusive != frame + 1
    ):
        raise ValueError("checkpoint freeze boundary must be [0, t+1)")
    return frame, timestamp, consumed, consumed_exclusive


def _timestamp_matches(value: float, timestamp_ns: int) -> bool:
    if not math.isfinite(value):
        return False
    if value == float(timestamp_ns):
        return True
    return math.isclose(value, timestamp_ns / 1e9, rel_tol=0.0, abs_tol=1e-9)


def _entity_type(metadata: Mapping[str, Any]) -> str:
    value = str(metadata.get("entity_type", metadata.get("type", "object"))).strip()
    if not value:
        raise ValueError("entity type must be non-empty")
    return value


def _presence_intervals(
    states: Mapping[str, dict[str, Any]], schedule: Sequence[Mapping[str, int]]
) -> list[dict[str, Any]]:
    lifecycles: list[dict[str, Any]] = []
    for entity_id in sorted(states):
        state = states[entity_id]
        positions = list(state["positions"])
        intervals: list[dict[str, int]] = []
        start = previous = positions[0]
        for position in positions[1:]:
            if position == previous + 1:
                previous = position
                continue
            first = schedule[start]
            last = schedule[previous]
            intervals.append(
                {
                    "first_frame_index": int(first["frame_index"]),
                    "first_timestamp_ns": int(first["timestamp_ns"]),
                    "last_frame_index": int(last["frame_index"]),
                    "last_timestamp_ns": int(last["timestamp_ns"]),
                }
            )
            start = previous = position
        first = schedule[start]
        last = schedule[previous]
        intervals.append(
            {
                "first_frame_index": int(first["frame_index"]),
                "first_timestamp_ns": int(first["timestamp_ns"]),
                "last_frame_index": int(last["frame_index"]),
                "last_timestamp_ns": int(last["timestamp_ns"]),
            }
        )
        lifecycles.append(
            {
                "entity_id": entity_id,
                "semantic_label": state["semantic_label"],
                "entity_type": state["entity_type"],
                "presence_intervals": intervals,
            }
        )
    return lifecycles


def _normalize_trajectories(
    source: _VerifiedSource, schedule: Sequence[Mapping[str, int]]
) -> list[dict[str, Any]]:
    checkpoint_frames = [int(item["frame_index"]) for item in schedule]
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    previous_frame = -1
    previous_timestamp = -1
    for line_number, line in enumerate(
        source.path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"trajectory line {line_number} must be an object")
        frame = raw.get("frame_index")
        timestamp = raw.get("timestamp_ns", raw.get("source_timestamp_ns"))
        if type(frame) is not int or type(timestamp) is not int:
            raise ValueError("trajectory frame and timestamp must be integers")
        if frame < previous_frame or timestamp < previous_timestamp:
            raise ValueError("trajectory frames and timestamps must be monotonic")
        if frame == previous_frame and previous_frame >= 0 and timestamp != previous_timestamp:
            raise ValueError("one trajectory frame cannot have multiple timestamps")
        if frame > previous_frame and previous_frame >= 0 and timestamp <= previous_timestamp:
            raise ValueError("trajectory timestamps must increase between frames")
        previous_frame = frame
        previous_timestamp = timestamp

        position = bisect_left(checkpoint_frames, frame)
        if position == len(schedule):
            continue
        checkpoint = schedule[position]
        if timestamp > int(checkpoint["timestamp_ns"]):
            raise ValueError("trajectory timestamp is later than checkpoint")

        raw_entities = raw.get("entities")
        if raw_entities is None:
            raw_entities = [raw]
        if not isinstance(raw_entities, list):
            raise ValueError("trajectory entities must be a list")
        for raw_entity in raw_entities:
            if not isinstance(raw_entity, Mapping):
                raise ValueError("trajectory entity must be an object")
            identifier = raw_entity.get("entity_id")
            if identifier is None and "native_submap_id" in raw_entity:
                identifier = f"panoptic:{raw_entity['native_submap_id']}"
            entity_id = str(identifier or "").strip()
            if not entity_id:
                raise ValueError("trajectory entity ID must be non-empty")
            key = (frame, entity_id)
            if key in seen:
                raise ValueError("duplicate entity trajectory sample within one frame")
            seen.add(key)
            centroid = np.asarray(raw_entity.get("centroid_xyz"), dtype=np.float64)
            if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
                raise ValueError("trajectory centroid must be a finite xyz vector")
            normalized = {
                "frame_index": frame,
                "timestamp_ns": timestamp,
                "entity_id": entity_id,
                "centroid_xyz": centroid.tolist(),
            }
            if "observation_count" in raw_entity:
                count = raw_entity["observation_count"]
                if type(count) is not int or count < 0:
                    raise ValueError("trajectory observation count must be non-negative")
                normalized["observation_count"] = count
            records.append(normalized)
    _assert_unchanged(source, label="trajectories")
    return records


def _output_record(path: Path, *, output: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(output)),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _relative_record(path: Path, *, base: Path) -> dict[str, Any]:
    relative = Path(os.path.relpath(path, start=base)).as_posix()
    if Path(relative).is_absolute():
        raise ValueError("artifact record path must be relative")
    return {
        "path": relative,
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _require_fields(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} fields are invalid")


def _json_integer(value: object, *, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _json_strings(
    value: object,
    *,
    label: str,
    allow_empty: bool,
) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must contain non-empty strings")
    if not allow_empty and not value:
        raise ValueError(f"{label} must be non-empty")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicates")
    return list(value)


def _logical_source_record(
    record: Mapping[str, Any],
    *,
    logical_id: str,
    label: str,
) -> dict[str, Any]:
    _require_fields(record, _SOURCE_RECORD_FIELDS, label=label)
    path = record.get("path")
    sha256 = record.get("sha256")
    byte_count = record.get("byte_count")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{label} path must be non-empty")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError(f"{label} SHA256 must be lowercase hexadecimal")
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError(f"{label} byte count must be a non-negative integer")
    return {
        "logical_id": logical_id,
        "sha256": sha256,
        "byte_count": byte_count,
    }


def _normalize_schedule(
    payload: Mapping[str, Any],
    *,
    selected_scene: str,
) -> dict[str, Any]:
    fields = frozenset(payload)
    if fields not in {_SCHEDULE_FIELDS, _SCHEDULE_FIELDS_WITH_SOURCE}:
        raise ValueError("schedule fields are invalid")
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping) or not set(parameters).issubset(
        _SCHEDULE_PARAMETER_FIELDS
    ):
        raise ValueError("schedule parameter fields are invalid")
    if any(isinstance(value, (Mapping, list)) for value in parameters.values()):
        raise ValueError("schedule parameter values must be scalar")
    if parameters.get("frame_indexing") != "zero_based":
        raise ValueError("schedule frame indexing must be zero_based")
    for name in (
        "official_stride_frames",
        "common_event_step_frames",
        "common_event_horizon_frames",
        "common_checkpoints_per_event",
    ):
        if name in parameters:
            _json_integer(parameters[name], label=f"schedule {name}", minimum=0)
    if "event_frame_rule" in parameters and (
        not isinstance(parameters["event_frame_rule"], str)
        or not parameters["event_frame_rule"]
    ):
        raise ValueError("schedule event_frame_rule must be non-empty")
    scenes = payload.get("scenes")
    scene_names = set(scenes) if isinstance(scenes, Mapping) else set()
    if (
        not scene_names
        or selected_scene not in scene_names
        or not scene_names.issubset({"apartment", "office"})
    ):
        raise ValueError("schedule scenes fields are invalid")

    normalized_scenes: dict[str, Any] = {}
    for scene in sorted(scene_names):
        raw_scene = scenes[scene]
        if not isinstance(raw_scene, Mapping):
            raise ValueError(f"schedule {scene} scene must be an object")
        scene_fields = frozenset(raw_scene)
        if scene_fields not in {
            _SCHEDULE_SCENE_FIELDS,
            _SCHEDULE_SCENE_FIELDS_WITH_SOURCES,
        }:
            raise ValueError(f"schedule {scene} scene fields are invalid")
        raw_entries = raw_scene.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError(f"schedule {scene} entries must be a list")
        frame_count = _json_integer(
            raw_scene.get("frame_count"),
            label=f"schedule {scene} frame_count",
            minimum=1,
        )
        entries: list[dict[str, Any]] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping) or frozenset(raw_entry) not in {
                _SCHEDULE_ENTRY_FIELDS,
                _SCHEDULE_ENTRY_FIELDS_FULL,
            }:
                raise ValueError(f"schedule {scene} entry fields are invalid")
            frame_index = _json_integer(
                raw_entry.get("frame_index"),
                label=f"schedule {scene} entry frame_index",
                minimum=0,
            )
            if frame_index >= frame_count:
                raise ValueError(f"schedule {scene} entry is outside frame_count")
            _json_integer(
                raw_entry.get("timestamp_ns"),
                label=f"schedule {scene} entry timestamp_ns",
                minimum=1,
            )
            if frozenset(raw_entry) == _SCHEDULE_ENTRY_FIELDS_FULL:
                _json_integer(
                    raw_entry.get("relative_timestamp_ns"),
                    label=f"schedule {scene} entry relative_timestamp_ns",
                    minimum=0,
                )
                _json_strings(
                    raw_entry.get("event_ids"),
                    label=f"schedule {scene} entry event_ids",
                    allow_empty=True,
                )
                roles = _json_strings(
                    raw_entry.get("roles"),
                    label=f"schedule {scene} entry roles",
                    allow_empty=False,
                )
                if any(role not in {"official", "common_v2"} for role in roles):
                    raise ValueError(f"schedule {scene} entry role is unsupported")
            entries.append(dict(raw_entry))
        normalized_scene: dict[str, Any] = {
            "frame_count": frame_count,
            "entries": entries,
        }
        if scene_fields == _SCHEDULE_SCENE_FIELDS_WITH_SOURCES:
            raw_events = raw_scene.get("events")
            if not isinstance(raw_events, list):
                raise ValueError(f"schedule {scene} events must be a list")
            events: list[dict[str, Any]] = []
            for raw_event in raw_events:
                if not isinstance(raw_event, Mapping):
                    raise ValueError(f"schedule {scene} event must be an object")
                _require_fields(
                    raw_event,
                    _SCHEDULE_EVENT_FIELDS,
                    label=f"schedule {scene} event",
                )
                event_id = raw_event.get("event_id")
                if not isinstance(event_id, str) or not event_id:
                    raise ValueError(f"schedule {scene} event ID is invalid")
                for name in (
                    "event_relative_timestamp_ns",
                    "intervention_frame_index",
                    "intervention_timestamp_ns",
                    "intervention_relative_timestamp_ns",
                ):
                    _json_integer(
                        raw_event.get(name),
                        label=f"schedule {scene} event {name}",
                        minimum=0,
                    )
                checkpoint_frames = raw_event.get(
                    "common_checkpoint_frame_indices"
                )
                if not isinstance(checkpoint_frames, list) or not checkpoint_frames:
                    raise ValueError(
                        f"schedule {scene} event checkpoint frames are invalid"
                    )
                for checkpoint_frame in checkpoint_frames:
                    normalized_frame = _json_integer(
                        checkpoint_frame,
                        label=f"schedule {scene} event checkpoint frame",
                        minimum=0,
                    )
                    if normalized_frame >= frame_count:
                        raise ValueError(
                            f"schedule {scene} event checkpoint is outside frame_count"
                        )
                events.append(dict(raw_event))
            raw_sources = raw_scene.get("sources")
            if not isinstance(raw_sources, Mapping) or set(raw_sources) != {
                "database",
                "gt_changes",
            }:
                raise ValueError(f"schedule {scene} source fields are invalid")
            normalized_scene.update(
                {
                    "events": events,
                    "first_depth_timestamp_ns": _json_integer(
                        raw_scene.get("first_depth_timestamp_ns"),
                        label=f"schedule {scene} first depth timestamp",
                        minimum=1,
                    ),
                    "last_depth_timestamp_ns": _json_integer(
                        raw_scene.get("last_depth_timestamp_ns"),
                        label=f"schedule {scene} last depth timestamp",
                        minimum=1,
                    ),
                    "sources": {
                        "database": _logical_source_record(
                            raw_sources["database"],
                            logical_id=f"tesse-cd:{scene}:database",
                            label=f"schedule {scene} database",
                        ),
                        "gt_changes": _logical_source_record(
                            raw_sources["gt_changes"],
                            logical_id=f"tesse-cd:{scene}:gt_changes",
                            label=f"schedule {scene} gt_changes",
                        ),
                    },
                }
            )
        normalized_scenes[scene] = normalized_scene

    normalized: dict[str, Any] = {
        "schema_version": payload.get("schema_version"),
        "manifest_id": payload.get("manifest_id"),
        "dataset": payload.get("dataset"),
        "method_predictions_used": payload.get("method_predictions_used"),
        "parameters": dict(parameters),
        "scenes": normalized_scenes,
    }
    if fields == _SCHEDULE_FIELDS_WITH_SOURCE:
        source_manifest = payload.get("source_manifest")
        if not isinstance(source_manifest, Mapping):
            raise ValueError("schedule source manifest must be an object")
        normalized["source_manifest"] = _logical_source_record(
            source_manifest,
            logical_id="tesse-cd:source-manifest",
            label="schedule source manifest",
        )
    return normalized


def _normalize_checkpoint_status(
    payload: Mapping[str, Any],
    *,
    frame_index: int,
) -> dict[str, Any]:
    fields = frozenset(payload)
    generic_with_events = _GENERIC_CHECKPOINT_STATUS_FIELDS | {"event_ids", "roles"}
    if fields in {_GENERIC_CHECKPOINT_STATUS_FIELDS, generic_with_events}:
        if payload.get("schema_version") != 1 or payload.get("status") != "PASS":
            raise ValueError("checkpoint status identity is invalid")
        normalized = dict(payload)
        if fields == generic_with_events:
            for name in ("event_ids", "roles"):
                _json_strings(
                    payload.get(name),
                    label=f"checkpoint status {name}",
                    allow_empty=name == "event_ids",
                )
        return normalized
    if fields == _PANOPTIC_CHECKPOINT_STATUS_FIELDS:
        if payload.get("schema_version") != 1:
            raise ValueError("checkpoint status identity is invalid")
        panmap_file = payload.get("panmap_file")
        if (
            not isinstance(panmap_file, str)
            or not panmap_file
            or Path(panmap_file).is_absolute()
            or Path(panmap_file).name != panmap_file
        ):
            raise ValueError("checkpoint status panmap_file is invalid")
        normalized = dict(payload)
        normalized["panmap_file"] = f"logical:panmap:{frame_index}"
        return normalized
    raise ValueError("checkpoint status fields are invalid")


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir():
            directories.append(path)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        descriptor = os.open(directory, directory_flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def export_temporal_artifact(source_index: Path, output: Path) -> Path:
    index_source = _direct_source(source_index, label="source index")
    index = _read_json(index_source, label="source index")
    if set(index) != _SOURCE_INDEX_FIELDS:
        raise ValueError("source index fields are invalid")
    if (
        index.get("schema_version") != 1
        or index.get("dataset") != "TESSE-CD"
        or index.get("mode") != "causal_checkpoint_exports"
    ):
        raise ValueError("temporal source index identity mismatch")
    scene = str(index.get("scene", "")).strip()
    method = str(index.get("method", "")).strip()
    if not scene or not method:
        raise ValueError("temporal source index requires method and scene")
    base = index_source.path.parent

    schedule_source = _declared_source(
        index.get("schedule", {}), base=base, label="schedule"
    )
    schedule_payload, _, schedule = _load_schedule(schedule_source, scene=scene)
    expected = [
        (int(entry["frame_index"]), int(entry["timestamp_ns"])) for entry in schedule
    ]
    trajectory_source = _declared_source(
        index.get("trajectories", {}), base=base, label="trajectories"
    )
    capture_source = _declared_source(
        index.get("capture_status", {}), base=base, label="capture status"
    )
    capture = _read_json(capture_source, label="capture status")
    if set(capture) != _CAPTURE_STATUS_FIELDS:
        raise ValueError("capture status fields are invalid")
    expected_frames = [frame for frame, _ in expected]
    if (
        capture.get("status") != "PASS"
        or capture.get("scene") != scene
        or capture.get("mode") != "causal_checkpoints"
        or capture.get("scheduled_frame_indices") != expected_frames
        or capture.get("captured_frame_indices") != expected_frames
    ):
        raise ValueError("capture status does not exactly cover schedule")
    if not _same_source(capture.get("schedule", {}), schedule_source, base=base):
        raise ValueError("capture status schedule source mismatch")
    if not _same_source(
        capture.get("trajectories", {}), trajectory_source, base=base
    ):
        raise ValueError("capture status trajectory source mismatch")

    raw_checkpoints = index.get("checkpoints")
    if not isinstance(raw_checkpoints, list):
        raise ValueError("source index checkpoints must be a list")
    observed = [
        (entry.get("frame_index"), entry.get("timestamp_ns"))
        for entry in raw_checkpoints
        if isinstance(entry, Mapping)
    ]
    if observed != expected or len(observed) != len(raw_checkpoints):
        raise ValueError("source checkpoints must exactly cover schedule")

    verified: list[tuple[_VerifiedSource, str]] = [
        (index_source, "source index"),
        (schedule_source, "schedule"),
        (trajectory_source, "trajectories"),
        (capture_source, "capture status"),
    ]
    checkpoint_inputs: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    snapshot_method: str | None = None
    checkpoint_status_sources: list[_VerifiedSource] = []
    for position, (raw_checkpoint, expected_identity) in enumerate(
        zip(raw_checkpoints, expected)
    ):
        assert isinstance(raw_checkpoint, Mapping)
        frame, timestamp = expected_identity
        consumed = raw_checkpoint.get("consumed_through_frame")
        consumed_exclusive = raw_checkpoint.get(
            "consumed_through_frame_exclusive"
        )
        if (
            type(consumed) is not int
            or type(consumed_exclusive) is not int
            or consumed != frame
            or consumed_exclusive != frame + 1
        ):
            raise ValueError("source checkpoint freeze boundary must be [0, t+1)")
        if set(raw_checkpoint) != _CHECKPOINT_INDEX_FIELDS:
            raise ValueError("source checkpoint fields are invalid")
        status_source = _declared_source(
            raw_checkpoint.get("checkpoint_status", {}),
            base=base,
            label=f"checkpoint {frame} status",
        )
        status = _read_json(status_source, label=f"checkpoint {frame} status")
        if _checkpoint_identity(status) != (
            frame,
            timestamp,
            consumed,
            consumed_exclusive,
        ):
            raise ValueError("checkpoint sidecar identity mismatch")
        snapshot_source = _declared_source(
            raw_checkpoint.get("snapshot", {}),
            base=base,
            label=f"checkpoint {frame} snapshot",
        )
        entities_source = _declared_source(
            raw_checkpoint.get("entities", {}),
            base=base,
            label=f"checkpoint {frame} entities",
        )
        snapshot = read_map_snapshot(snapshot_source.path, entities_source.path)
        _assert_unchanged(snapshot_source, label=f"checkpoint {frame} snapshot")
        _assert_unchanged(entities_source, label=f"checkpoint {frame} entities")
        if snapshot.scene_id != scene or snapshot.scope != "current":
            raise ValueError("neutral checkpoint snapshot identity mismatch")
        if not _timestamp_matches(float(snapshot.timestamp), timestamp):
            raise ValueError("neutral checkpoint snapshot timestamp mismatch")
        if snapshot_method is None:
            snapshot_method = snapshot.method
        elif snapshot.method != snapshot_method:
            raise ValueError("neutral checkpoint method conflict")

        entity_ids = [entity.entity_id for entity in snapshot.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("duplicate entity ID within checkpoint")
        for entity in snapshot.entities:
            entity_id = str(entity.entity_id)
            label = entity.semantic_label
            entity_type = _entity_type(entity.metadata)
            state = states.get(entity_id)
            if state is None:
                states[entity_id] = {
                    "semantic_label": label,
                    "entity_type": entity_type,
                    "positions": [position],
                }
                continue
            if state["semantic_label"] != label:
                raise ValueError(f"stable entity ID semantic conflict: {entity_id}")
            if state["entity_type"] != entity_type:
                raise ValueError(f"stable entity ID type conflict: {entity_id}")
            state["positions"].append(position)
        checkpoint_status_sources.append(status_source)
        verified.extend(
            [
                (status_source, f"checkpoint {frame} status"),
                (snapshot_source, f"checkpoint {frame} snapshot"),
                (entities_source, f"checkpoint {frame} entities"),
            ]
        )
        checkpoint_inputs.append(
            {
                "frame_index": frame,
                "timestamp_ns": timestamp,
                "consumed_through_frame": consumed,
                "consumed_through_frame_exclusive": consumed_exclusive,
                "status": status,
                "snapshot": snapshot_source,
                "entities": entities_source,
            }
        )

    declared_statuses = capture.get("checkpoint_statuses")
    if not isinstance(declared_statuses, list) or len(declared_statuses) != len(
        checkpoint_status_sources
    ):
        raise ValueError("capture status checkpoint list mismatch")
    for declared, source in zip(declared_statuses, checkpoint_status_sources):
        if not isinstance(declared, Mapping) or not _same_source(
            declared, source, base=base
        ):
            raise ValueError("capture status checkpoint source mismatch")

    trajectory_records = _normalize_trajectories(trajectory_source, schedule)
    lifecycles = _presence_intervals(states, schedule)
    for source, label in verified:
        _assert_unchanged(source, label=label)

    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    reserved = False
    try:
        sidecar_root = staging / "sidecars"
        sidecar_root.mkdir(parents=True)
        output_checkpoints: list[dict[str, Any]] = []
        checkpoint_paths: list[dict[str, Any]] = []
        for checkpoint in checkpoint_inputs:
            frame = int(checkpoint["frame_index"])
            directory = staging / "checkpoints" / f"{frame:08d}"
            directory.mkdir(parents=True)
            snapshot_path = directory / "snapshot.npz"
            entities_path = directory / "entities.jsonl"
            shutil.copyfile(checkpoint["snapshot"].path, snapshot_path)
            shutil.copyfile(checkpoint["entities"].path, entities_path)
            _assert_unchanged(
                checkpoint["snapshot"], label=f"checkpoint {frame} snapshot"
            )
            _assert_unchanged(
                checkpoint["entities"], label=f"checkpoint {frame} entities"
            )
            output_checkpoints.append(
                {
                    "frame_index": frame,
                    "timestamp_ns": int(checkpoint["timestamp_ns"]),
                    "consumed_through_frame": int(
                        checkpoint["consumed_through_frame"]
                    ),
                    "consumed_through_frame_exclusive": int(
                        checkpoint["consumed_through_frame_exclusive"]
                    ),
                    "snapshot": _output_record(snapshot_path, output=staging),
                    "entities": _output_record(entities_path, output=staging),
                }
            )
            checkpoint_paths.append(
                {
                    "frame_index": frame,
                    "timestamp_ns": int(checkpoint["timestamp_ns"]),
                    "consumed_through_frame": int(
                        checkpoint["consumed_through_frame"]
                    ),
                    "consumed_through_frame_exclusive": int(
                        checkpoint["consumed_through_frame_exclusive"]
                    ),
                    "snapshot_path": snapshot_path,
                    "entities_path": entities_path,
                }
            )

        trajectories_path = staging / "trajectories.jsonl"
        trajectories_path.write_text(
            "".join(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for record in trajectory_records
            ),
            encoding="utf-8",
        )

        schedule_path = sidecar_root / "schedule.json"
        _write_json(
            schedule_path,
            _normalize_schedule(schedule_payload, selected_scene=scene),
        )
        status_root = sidecar_root / "checkpoint_statuses"
        status_root.mkdir()
        status_paths: list[Path] = []
        for checkpoint in checkpoint_inputs:
            frame = int(checkpoint["frame_index"])
            status_path = status_root / f"{frame:08d}.json"
            _write_json(
                status_path,
                _normalize_checkpoint_status(
                    checkpoint["status"],
                    frame_index=frame,
                ),
            )
            status_paths.append(status_path)

        schedule_sidecar_record = _relative_record(
            schedule_path,
            base=sidecar_root,
        )
        trajectory_sidecar_record = _relative_record(
            trajectories_path,
            base=sidecar_root,
        )
        status_sidecar_records = [
            _relative_record(path, base=sidecar_root) for path in status_paths
        ]
        capture_path = sidecar_root / "capture_status.json"
        normalized_capture = {
            "schema_version": capture["schema_version"],
            "status": capture["status"],
            "scene": capture["scene"],
            "mode": capture["mode"],
            "scheduled_frame_indices": capture["scheduled_frame_indices"],
            "captured_frame_indices": capture["captured_frame_indices"],
            "schedule": schedule_sidecar_record,
            "trajectories": trajectory_sidecar_record,
            "checkpoint_statuses": status_sidecar_records,
        }
        _write_json(capture_path, normalized_capture)

        normalized_index_checkpoints = []
        for checkpoint, status_path in zip(checkpoint_paths, status_paths):
            normalized_index_checkpoints.append(
                {
                    "frame_index": checkpoint["frame_index"],
                    "timestamp_ns": checkpoint["timestamp_ns"],
                    "consumed_through_frame": checkpoint[
                        "consumed_through_frame"
                    ],
                    "consumed_through_frame_exclusive": checkpoint[
                        "consumed_through_frame_exclusive"
                    ],
                    "checkpoint_status": _relative_record(
                        status_path,
                        base=sidecar_root,
                    ),
                    "snapshot": _relative_record(
                        checkpoint["snapshot_path"],
                        base=sidecar_root,
                    ),
                    "entities": _relative_record(
                        checkpoint["entities_path"],
                        base=sidecar_root,
                    ),
                }
            )
        index_path = sidecar_root / "source_index.json"
        normalized_index = {
            "schema_version": index["schema_version"],
            "dataset": index["dataset"],
            "mode": index["mode"],
            "method": index["method"],
            "scene": index["scene"],
            "schedule": schedule_sidecar_record,
            "capture_status": _relative_record(
                capture_path,
                base=sidecar_root,
            ),
            "trajectories": trajectory_sidecar_record,
            "checkpoints": normalized_index_checkpoints,
        }
        _write_json(index_path, normalized_index)

        normalized_sources = {
            "source_index": _output_record(index_path, output=staging),
            "schedule": _output_record(schedule_path, output=staging),
            "capture_status": _output_record(capture_path, output=staging),
            "trajectories": _output_record(trajectories_path, output=staging),
            "checkpoint_statuses": [
                _output_record(path, output=staging) for path in status_paths
            ],
        }
        for source, label in verified:
            _assert_unchanged(source, label=label)

        manifest = {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoints",
            "method": method,
            "scene": scene,
            "sources": normalized_sources,
            "checkpoints": output_checkpoints,
            "entity_lifecycles": lifecycles,
            "trajectories": _output_record(
                trajectories_path, output=staging
            ),
        }
        _write_json(staging / "temporal_manifest.json", manifest)
        _fsync_tree(staging)

        output.mkdir()
        reserved = True
        try:
            os.rename(staging, output)
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            descriptor = os.open(output.parent, directory_flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except Exception as error:
            raise RuntimeError(
                "temporal artifact publication-uncertain"
            ) from error
    except Exception:
        if not reserved:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "temporal_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    export_temporal_artifact(args.source_index, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
