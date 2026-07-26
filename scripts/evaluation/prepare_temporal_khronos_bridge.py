#!/usr/bin/env python3
"""Prepare causal neutral checkpoints for the Khronos temporal importer."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.exporters.oviovo import read_map_snapshot
from src.evaluation.json_contracts import loads_strict


OFFICIAL_LABEL_SPACE_SHA256 = {
    "apartment": frozenset(
        {"f7bacfb3c7bafc674bd3a34540e3f9b5d98b0ae11389f91768c70ecc65a79f0f"}
    ),
    "office": frozenset(
        {"91a7b359ee678dd67871653959a50c477ba7f371473b9c2bf1b4f41f63691765"}
    ),
}
OFFICIAL_SCHEDULE_SHA256 = frozenset(
    {"fb97bacee377f9fd67ee9dae8064dc6f33ac32d129ee633629fec4164d5003e0"}
)
_TRAJECTORY_FIELDS = frozenset(
    {
        "frame_index",
        "timestamp_ns",
        "entity_id",
        "centroid_xyz",
        "observation_count",
        "dynamic_state",
        "motion_confidence",
        "geometry_epoch",
        "readout_valid",
    }
)
_COVERAGE_FIELDS = frozenset(
    {"frame_index", "timestamp_ns", "record_count", "event_count"}
)
_LIFECYCLE_FIELDS = frozenset(
    {
        "frame_index",
        "timestamp_ns",
        "entity_id",
        "before",
        "after",
        "evidence",
        "geometry_epoch",
        "readout_valid",
    }
)


@dataclass(frozen=True)
class _VerifiedSource:
    path: Path
    sha256: str
    byte_count: int
    fingerprint: tuple[int, int, int, int, int]


def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    status = path.stat(follow_symlinks=False)
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _direct_source(path: Path, *, label: str) -> _VerifiedSource:
    path = Path(os.path.abspath(path))
    try:
        status = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} is not a file: {path}") from error
    if stat.S_ISLNK(status.st_mode):
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    fingerprint = _fingerprint(path)
    source = _VerifiedSource(path, _sha256(path), status.st_size, fingerprint)
    _assert_unchanged(source, label=label)
    return source


def _assert_unchanged(source: _VerifiedSource, *, label: str) -> None:
    try:
        observed = _fingerprint(source.path)
    except OSError as error:
        raise ValueError(f"{label} source changed while reading") from error
    if observed != source.fingerprint:
        raise ValueError(f"{label} source changed while reading")


def _read_json(source: _VerifiedSource, *, label: str) -> dict[str, Any]:
    payload = loads_strict(source.path.read_text(encoding="utf-8"), label=label)
    _assert_unchanged(source, label=label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _source_path(
    record: Mapping[str, Any], base: Path, label: str
) -> _VerifiedSource:
    if set(record) != {"path", "sha256", "byte_count"}:
        raise ValueError(f"{label} source record fields are invalid")
    raw = Path(str(record.get("path", "")))
    path = raw if raw.is_absolute() else base / raw
    source = _direct_source(path, label=label)
    byte_count = record.get("byte_count")
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError(f"{label} byte count must be a non-negative integer")
    if source.byte_count != byte_count:
        raise ValueError(f"{label} byte count mismatch")
    if source.sha256 != str(record.get("sha256", "")):
        raise ValueError(f"{label} SHA256 mismatch")
    return source


def _source_record(source: _VerifiedSource) -> dict[str, Any]:
    return {
        "path": str(source.path),
        "sha256": source.sha256,
        "byte_count": source.byte_count,
    }


def _artifact_path(manifest_path: Path, raw: object) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        raise ValueError("bridge artifact path must stay inside bridge output")
    root = manifest_path.parent.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("bridge artifact path must stay inside bridge output")
    return resolved


def _normalize_label(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _write_binary_ply(path: Path, points: np.ndarray) -> None:
    values = np.ascontiguousarray(points, dtype="<f4").reshape(-1, 3)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(values)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(values.tobytes(order="C"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _timestamp_matches(value: float, timestamp_ns: int) -> bool:
    return value == float(timestamp_ns) or math.isclose(
        value, timestamp_ns / 1e9, rel_tol=0.0, abs_tol=1e-9
    )


def _presence_runs(
    positions: Sequence[int], checkpoints: Sequence[Mapping[str, Any]]
) -> list[dict[str, int | None]]:
    runs: list[dict[str, int | None]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position == previous + 1:
            previous = position
            continue
        runs.append(
            {
                "start_ns": int(checkpoints[start]["timestamp_ns"]),
                "end_ns_exclusive": int(checkpoints[previous + 1]["timestamp_ns"]),
            }
        )
        start = previous = position
    end = (
        int(checkpoints[previous + 1]["timestamp_ns"])
        if previous + 1 < len(checkpoints)
        else None
    )
    runs.append(
        {
            "start_ns": int(checkpoints[start]["timestamp_ns"]),
            "end_ns_exclusive": end,
        }
    )
    return runs


def _checkpoint_endpoints(
    intervals: Sequence[Mapping[str, Any]], query: int
) -> tuple[list[int], list[int]]:
    if query >= (1 << 64) - 1:
        raise ValueError("checkpoint timestamp cannot encode an ongoing interval")
    ongoing_end = query + 1
    starts: list[int] = []
    ends: list[int] = []
    for interval in intervals:
        start = int(interval["start_ns"])
        if start > query:
            continue
        end = interval["end_ns_exclusive"]
        starts.append(start)
        ends.append(
            int(end) if end is not None and int(end) <= query else ongoing_end
        )
    return starts, ends


def _load_frame_coverage(path: Path) -> list[dict[str, int]]:
    coverage: list[dict[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = loads_strict(line, label="temporal frame coverage")
        if not isinstance(record, Mapping) or frozenset(record) != _COVERAGE_FIELDS:
            raise ValueError("temporal frame coverage fields are invalid")
        frame = record.get("frame_index")
        timestamp = record.get("timestamp_ns")
        if not (
            type(frame) is int
            and frame == len(coverage)
            and type(timestamp) is int
            and timestamp > 0
            and (not coverage or timestamp > coverage[-1]["timestamp_ns"])
            and type(record.get("record_count")) is int
            and record["record_count"] >= 0
            and type(record.get("event_count")) is int
            and record["event_count"] >= 0
        ):
            raise ValueError("temporal frame coverage is incomplete or noncausal")
        coverage.append(dict(record))
    if not coverage:
        raise ValueError("temporal frame coverage is missing")
    return coverage


def _load_lifecycle_transitions(
    path: Path, coverage: Sequence[Mapping[str, int]]
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    counts: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = loads_strict(line, label="temporal lifecycle transition")
        if not isinstance(record, Mapping) or frozenset(record) != _LIFECYCLE_FIELDS:
            raise ValueError("temporal lifecycle transition fields are invalid")
        frame = record.get("frame_index")
        timestamp = record.get("timestamp_ns")
        entity_id = str(record.get("entity_id", "")).strip()
        if not (
            type(frame) is int
            and 0 <= frame < len(coverage)
            and type(timestamp) is int
            and timestamp == coverage[frame]["timestamp_ns"]
            and entity_id
            and record.get("before") in {"active", "uncertain", "dormant"}
            and record.get("after") in {"active", "uncertain", "dormant"}
            and record.get("evidence")
            in {"present", "visible_absent", "occluded", "out_of_view", "depth_unknown"}
            and type(record.get("geometry_epoch")) is int
            and record["geometry_epoch"] >= 0
            and type(record.get("readout_valid")) is bool
        ):
            raise ValueError("temporal lifecycle transition is invalid")
        key = (frame, entity_id)
        if key in seen:
            raise ValueError("duplicate temporal lifecycle transition")
        seen.add(key)
        counts[frame] = counts.get(frame, 0) + 1
        transitions.append({**record, "entity_id": entity_id})
    if any(
        counts.get(int(item["frame_index"]), 0) != int(item["event_count"])
        for item in coverage
    ):
        raise ValueError("temporal lifecycle transition coverage mismatch")
    return transitions


def _load_trajectories(
    path: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, int]],
) -> dict[str, list[dict[str, Any]]]:
    frames = [int(checkpoint["frame_index"]) for checkpoint in checkpoints]
    trajectories: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[int, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = loads_strict(line, label="temporal trajectory")
        if not isinstance(record, Mapping) or frozenset(record) != _TRAJECTORY_FIELDS:
            raise ValueError("temporal trajectory fields are invalid")
        frame = record.get("frame_index")
        timestamp = record.get("timestamp_ns")
        entity_id = str(record.get("entity_id", "")).strip()
        if type(frame) is not int or type(timestamp) is not int or not entity_id:
            raise ValueError("temporal trajectory record is invalid")
        if frame < 0 or timestamp <= 0:
            raise ValueError("temporal trajectory record is invalid")
        if type(record["observation_count"]) is not int or record["observation_count"] < 1:
            raise ValueError("temporal trajectory observation count is invalid")
        if record.get("dynamic_state") not in {"static", "dynamic", "unknown"}:
            raise ValueError("temporal trajectory dynamic state is invalid")
        confidence = record.get("motion_confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
            or type(record.get("geometry_epoch")) is not int
            or record["geometry_epoch"] < 0
            or type(record.get("readout_valid")) is not bool
        ):
            raise ValueError("temporal trajectory explicit state is invalid")
        if frame >= len(coverage) or timestamp != coverage[frame]["timestamp_ns"]:
            raise ValueError("temporal trajectory frame coverage mismatch")
        position = bisect_left(frames, frame)
        if position < len(checkpoints) and timestamp > int(
            checkpoints[position]["timestamp_ns"]
        ):
            raise ValueError("temporal trajectory timestamp is later than query")
        key = (frame, entity_id)
        if key in seen:
            raise ValueError("duplicate temporal trajectory sample")
        seen.add(key)
        centroid = np.asarray(record.get("centroid_xyz"), dtype=np.float64)
        if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
            raise ValueError("temporal trajectory centroid is invalid")
        trajectories.setdefault(entity_id, []).append(
            {
                **record,
                "entity_id": entity_id,
                "centroid_xyz": centroid.tolist(),
            }
        )
    for entity_id, samples in trajectories.items():
        timestamps = [int(sample["timestamp_ns"]) for sample in samples]
        if timestamps != sorted(set(timestamps)):
            raise ValueError(f"trajectory timestamps are not unique for {entity_id}")
    if any(
        sum(
            int(sample["frame_index"]) == int(item["frame_index"])
            for samples in trajectories.values()
            for sample in samples
        )
        != int(item["record_count"])
        for item in coverage
    ):
        raise ValueError("temporal trajectory frame coverage count mismatch")
    return trajectories


def _explicit_presence_runs(
    states: Mapping[str, Mapping[str, Any]],
    coverage: Sequence[Mapping[str, int]],
    trajectories: Mapping[str, Sequence[Mapping[str, Any]]],
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, int | None]]]:
    source_to_entity: dict[str, str] = {}
    for entity_id, state in states.items():
        source_id = str(state.get("source_entity_id", entity_id))
        if source_id in source_to_entity and source_to_entity[source_id] != entity_id:
            raise ValueError("temporal entity source binding is ambiguous")
        source_to_entity[source_id] = entity_id
    updates: dict[int, dict[str, bool]] = {}
    records = [sample for samples in trajectories.values() for sample in samples]
    for record in (*records, *transitions):
        frame = int(record["frame_index"])
        source_id = str(record["entity_id"])
        entity_id = source_to_entity.get(source_id, source_id)
        if entity_id not in states:
            continue
        valid = bool(record["readout_valid"])
        previous = updates.setdefault(frame, {}).get(entity_id)
        if previous is not None and previous is not valid:
            raise ValueError("conflicting temporal readout validity")
        updates[frame][entity_id] = valid
    intervals = {entity_id: [] for entity_id in states}
    starts: dict[str, int | None] = {entity_id: None for entity_id in states}
    for coverage_record in coverage:
        frame = int(coverage_record["frame_index"])
        timestamp = int(coverage_record["timestamp_ns"])
        for entity_id, valid in updates.get(frame, {}).items():
            start = starts[entity_id]
            if valid and start is None:
                starts[entity_id] = timestamp
            elif not valid and start is not None:
                intervals[entity_id].append(
                    {"start_ns": start, "end_ns_exclusive": timestamp}
                )
                starts[entity_id] = None
    for entity_id, start in starts.items():
        if start is not None:
            intervals[entity_id].append(
                {"start_ns": start, "end_ns_exclusive": None}
            )
        if not intervals[entity_id]:
            raise ValueError(f"entity has no explicit presence interval: {entity_id}")
    return intervals


def _label_space(path: Path, *, scene: str) -> dict[str, int]:
    expected = OFFICIAL_LABEL_SPACE_SHA256.get(scene)
    if not expected or _sha256(path) not in expected:
        raise ValueError(f"official bridge requires the official {scene} label space")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    labels = {
        _normalize_label(entry["name"]): int(entry["label"])
        for entry in payload.get("label_names", ())
    }
    if labels.get("unknown") != 0:
        raise ValueError("Khronos label space must define Unknown as label 0")
    return labels


def _scheduled_checkpoints(
    temporal: Mapping[str, Any], *, base: Path, scene: str
) -> tuple[_VerifiedSource, list[tuple[int, int]]]:
    sources = temporal.get("sources")
    if not isinstance(sources, Mapping) or not isinstance(
        sources.get("schedule"), Mapping
    ):
        raise ValueError("temporal artifact requires a hashed schedule source")
    schedule_source = _source_path(sources["schedule"], base, "schedule")
    if schedule_source.sha256 not in OFFICIAL_SCHEDULE_SHA256:
        raise ValueError("official bridge requires the official TESSE-CD causal schedule")
    schedule = _read_json(schedule_source, label="schedule")
    if (
        schedule.get("schema_version") != 2
        or schedule.get("manifest_id") != "tesse_cd_causal_schedule_v2"
        or schedule.get("dataset") != "TESSE-CD"
        or schedule.get("method_predictions_used") is not False
    ):
        raise ValueError("invalid TESSE-CD causal schedule identity")
    scenes = schedule.get("scenes")
    selected = scenes.get(scene) if isinstance(scenes, Mapping) else None
    entries = selected.get("entries") if isinstance(selected, Mapping) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("TESSE-CD causal schedule has no selected scene entries")
    identities: list[tuple[int, int]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("TESSE-CD causal schedule entry is invalid")
        frame = entry.get("frame_index")
        timestamp = entry.get("timestamp_ns")
        if type(frame) is not int or frame < 0 or type(timestamp) is not int or timestamp <= 0:
            raise ValueError("TESSE-CD causal schedule entry is invalid")
        identities.append((frame, timestamp))
    if identities != sorted(set(identities)):
        raise ValueError("TESSE-CD causal schedule entries must be unique and ordered")
    return schedule_source, identities


def validate_temporal_bridge_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_source = _direct_source(manifest_path, label="bridge manifest")
    manifest = _read_json(manifest_source, label="bridge manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("mode") != "temporal_checkpoints"
        or manifest.get("dataset") != "TESSE-CD"
        or manifest.get("method") != "OVIV2"
        or manifest.get("display_mode") != "online"
    ):
        raise ValueError("invalid temporal bridge manifest identity")
    audit = manifest.get("temporal_audit_counts")
    audit_fields = {
        "static_sample_count",
        "dynamic_sample_count",
        "unknown_sample_count",
        "missing_frame_count",
        "lifecycle_transition_count",
        "geometry_epoch_count",
        "invalid_readout_sample_count",
    }
    if (
        not isinstance(audit, Mapping)
        or set(audit) != audit_fields
        or any(type(audit[field]) is not int or audit[field] < 0 for field in audit_fields)
        or audit["missing_frame_count"] != 0
    ):
        raise ValueError("temporal bridge audit counts are invalid")
    timestamps = [int(value) for value in manifest.get("query_timestamps_ns", ())]
    if not timestamps or any(
        current <= previous for previous, current in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("temporal bridge timestamps must be strictly increasing")
    symbols = [entry.get("node_symbol") for entry in manifest.get("symbol_assignments", ())]
    if len(symbols) != len(set(symbols)):
        raise ValueError("temporal bridge node symbols must be unique")

    raw_assignments = manifest.get("symbol_assignments")
    if not isinstance(raw_assignments, list):
        raise ValueError("temporal bridge assignments must be a list")
    assignments: dict[str, Mapping[str, Any]] = {}
    for node_index, assignment in enumerate(raw_assignments):
        if not isinstance(assignment, Mapping):
            raise ValueError("temporal bridge assignment is invalid")
        entity_id = assignment.get("entity_id")
        semantic_label = assignment.get("semantic_label")
        semantic_label_name = assignment.get("semantic_label_name")
        label_matched = assignment.get("label_matched")
        if (
            not isinstance(entity_id, str)
            or not entity_id
            or entity_id in assignments
            or assignment.get("node_index") != node_index
            or assignment.get("node_symbol") != f"O{node_index}"
            or type(semantic_label) is not int
            or semantic_label < 0
            or (
                semantic_label_name is not None
                and not isinstance(semantic_label_name, str)
            )
            or type(label_matched) is not bool
            or not isinstance(assignment.get("source_entity_id"), str)
            or not assignment["source_entity_id"]
        ):
            raise ValueError("temporal bridge assignment identity is invalid")
        intervals = assignment.get("presence_intervals")
        if not isinstance(intervals, list) or not intervals:
            raise ValueError("temporal bridge assignment intervals are invalid")
        starts: list[int] = []
        ends: list[int | None] = []
        previous_end = -1
        for interval in intervals:
            if not isinstance(interval, Mapping) or set(interval) != {
                "start_ns",
                "end_ns_exclusive",
            }:
                raise ValueError("temporal bridge assignment interval is invalid")
            start = interval.get("start_ns")
            end = interval.get("end_ns_exclusive")
            if (
                type(start) is not int
                or start <= previous_end
                or (end is not None and (type(end) is not int or end <= start))
            ):
                raise ValueError("temporal bridge assignment interval is invalid")
            starts.append(start)
            ends.append(end)
            if end is not None:
                previous_end = end
            else:
                if interval is not intervals[-1]:
                    raise ValueError("open temporal bridge interval must be final")
                previous_end = start
        if (
            assignment.get("first_observed_ns") != starts
            or assignment.get("last_observed_ns") != ends
        ):
            raise ValueError("temporal bridge assignment endpoints are invalid")
        latest_dynamic_state = assignment.get("latest_dynamic_state")
        if latest_dynamic_state not in {"static", "dynamic", "unknown", None} or (
            assignment.get("dynamic_track_eligible")
            is not (latest_dynamic_state == "dynamic")
        ):
            raise ValueError("temporal bridge assignment dynamic state is invalid")
        observations = assignment.get("semantic_observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError("temporal bridge semantic history is invalid")
        expected_timestamps = [
            timestamp
            for timestamp in timestamps
            if any(
                interval["start_ns"] <= timestamp
                and (
                    interval["end_ns_exclusive"] is None
                    or timestamp < interval["end_ns_exclusive"]
                )
                for interval in intervals
            )
        ]
        observed_timestamps: list[int] = []
        for observation in observations:
            if not isinstance(observation, Mapping) or set(observation) != {
                "timestamp_ns",
                "semantic_label_name",
                "semantic_label",
                "label_matched",
            }:
                raise ValueError("temporal bridge semantic history is invalid")
            observation_name = observation.get("semantic_label_name")
            observation_label = observation.get("semantic_label")
            observation_matched = observation.get("label_matched")
            observation_timestamp = observation.get("timestamp_ns")
            if (
                type(observation_timestamp) is not int
                or (
                    observation_name is not None
                    and not isinstance(observation_name, str)
                )
                or type(observation_label) is not int
                or observation_label < 0
                or type(observation_matched) is not bool
            ):
                raise ValueError("temporal bridge semantic history is invalid")
            observed_timestamps.append(observation_timestamp)
        if observed_timestamps != expected_timestamps:
            raise ValueError("temporal bridge semantic history coverage is invalid")
        first_observation = observations[0]
        if any(
            assignment.get(field) != first_observation.get(field)
            for field in ("semantic_label_name", "semantic_label", "label_matched")
        ):
            raise ValueError("temporal bridge semantic history legacy fields mismatch")
        assignments[entity_id] = assignment

    for source in manifest.get("hashed_inputs", ()):
        _source_path(source, manifest_path.parent, "hashed bridge input")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(timestamps):
        raise ValueError("temporal bridge checkpoint count mismatch")
    previous_frame = -1
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        query = int(checkpoint["timestamp_ns"])
        frame = checkpoint.get("frame_index")
        if (
            query != timestamps[checkpoint_index]
            or type(frame) is not int
            or frame <= previous_frame
        ):
            raise ValueError("temporal bridge checkpoint identity mismatch")
        previous_frame = frame
        background = _artifact_path(manifest_path, checkpoint["background_ply"])
        if not background.is_file() or background.stat().st_size != int(
            checkpoint["background_byte_count"]
        ):
            raise ValueError("bridge background byte count mismatch")
        if _sha256(background) != str(checkpoint["background_sha256"]):
            raise ValueError("bridge background SHA256 mismatch")
        objects = checkpoint.get("objects")
        if not isinstance(objects, list):
            raise ValueError("temporal bridge checkpoint objects are invalid")
        object_ids: list[str] = []
        node_indices: list[int] = []
        for obj in objects:
            entity_id = obj.get("entity_id")
            assignment = assignments.get(entity_id)
            if assignment is None or any(
                obj.get(field) != assignment.get(field)
                for field in (
                    "node_index",
                    "node_symbol",
                )
            ):
                raise ValueError("temporal bridge object assignment mismatch")
            semantic_observation = next(
                (
                    observation
                    for observation in assignment["semantic_observations"]
                    if observation["timestamp_ns"] == query
                ),
                None,
            )
            if semantic_observation is None or any(
                obj.get(field) != semantic_observation.get(field)
                for field in ("semantic_label_name", "semantic_label", "label_matched")
            ):
                raise ValueError(
                    "temporal bridge object semantic history assignment mismatch"
                )
            expected_starts, expected_ends = _checkpoint_endpoints(
                assignment["presence_intervals"], query
            )
            if (
                obj.get("first_observed_ns") != expected_starts
                or obj.get("last_observed_ns") != expected_ends
            ):
                raise ValueError("temporal bridge object assignment endpoints mismatch")
            object_ids.append(entity_id)
            node_indices.append(int(obj["node_index"]))
            for prefix in ("points", "trajectory"):
                path = _artifact_path(
                    manifest_path,
                    obj[f"{prefix}_ply" if prefix == "points" else f"{prefix}_json"],
                )
                if not path.is_file() or path.stat().st_size != int(
                    obj[f"{prefix}_byte_count"]
                ):
                    raise ValueError(f"bridge {prefix} byte count mismatch")
                if _sha256(path) != str(obj[f"{prefix}_sha256"]):
                    raise ValueError(f"bridge {prefix} SHA256 mismatch")
            trajectory_path = _artifact_path(manifest_path, obj["trajectory_json"])
            trajectory = loads_strict(
                trajectory_path.read_text(encoding="utf-8"),
                label="bridge trajectory",
            )
            if not isinstance(trajectory, list):
                raise ValueError("bridge trajectory must be a list")
            previous_sample_timestamp = -1
            for sample in trajectory:
                if not isinstance(sample, Mapping) or frozenset(sample) != _TRAJECTORY_FIELDS:
                    raise ValueError("bridge trajectory sample fields are invalid")
                centroid = np.asarray(sample.get("centroid_xyz"), dtype=np.float64)
                sample_timestamp = sample.get("timestamp_ns")
                if not (
                    sample.get("entity_id") == assignment["source_entity_id"]
                    and type(sample.get("frame_index")) is int
                    and type(sample_timestamp) is int
                    and sample_timestamp > previous_sample_timestamp
                    and centroid.shape == (3,)
                    and np.all(np.isfinite(centroid))
                    and type(sample.get("observation_count")) is int
                    and sample["observation_count"] >= 1
                    and sample.get("dynamic_state") in {"static", "dynamic", "unknown"}
                    and type(sample.get("geometry_epoch")) is int
                    and sample["geometry_epoch"] >= 0
                    and type(sample.get("readout_valid")) is bool
                ):
                    raise ValueError("bridge trajectory sample is invalid")
                confidence = sample.get("motion_confidence")
                if (
                    isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not math.isfinite(float(confidence))
                    or not 0.0 <= float(confidence) <= 1.0
                ):
                    raise ValueError("bridge trajectory sample is invalid")
                previous_sample_timestamp = sample_timestamp
            if any(
                int(sample["timestamp_ns"]) > query
                or int(sample["frame_index"]) > frame
                for sample in trajectory
            ):
                raise ValueError("bridge trajectory sample is later than checkpoint")
            if not obj["dynamic_track_eligible"] and trajectory:
                raise ValueError("ineligible dynamic track must have no trajectory")
            state_at_query = obj.get("dynamic_state_at_query")
            if state_at_query not in {"static", "dynamic", "unknown", None} or (
                obj["dynamic_track_eligible"] is not (state_at_query == "dynamic")
            ):
                raise ValueError("dynamic track eligibility disagrees with explicit state")
            if trajectory and trajectory[-1].get("dynamic_state") != "dynamic":
                raise ValueError("eligible trajectory does not end in dynamic state")
            if len(trajectory) != int(obj["trajectory_sample_count"]):
                raise ValueError("bridge trajectory sample count mismatch")
        expected_ids = {
            entity_id
            for entity_id, assignment in assignments.items()
            if any(
                interval["start_ns"] <= query
                and (
                    interval["end_ns_exclusive"] is None
                    or query < interval["end_ns_exclusive"]
                )
                for interval in assignment["presence_intervals"]
            )
        }
        if (
            len(object_ids) != len(set(object_ids))
            or set(object_ids) != expected_ids
            or node_indices != sorted(node_indices)
        ):
            raise ValueError("temporal bridge object assignment coverage mismatch")
    return manifest


def prepare_temporal_bridge(
    temporal_manifest_path: Path,
    label_space_path: Path,
    output: Path,
) -> Path:
    temporal_source = _direct_source(
        temporal_manifest_path, label="temporal manifest"
    )
    label_source = _direct_source(label_space_path, label="label space")
    temporal = _read_json(temporal_source, label="temporal manifest")
    if temporal.get("schema_version") != 1:
        raise ValueError("official bridge requires temporal artifact schema 1")
    if temporal.get("dataset") != "TESSE-CD":
        raise ValueError("official bridge requires TESSE-CD")
    if temporal.get("method") != "OVIV2":
        raise ValueError("official bridge requires the frozen OVIV2 artifact")
    if temporal.get("mode") != "causal_checkpoints":
        raise ValueError("official bridge requires causal checkpoints")
    if temporal.get("temporal_export_schema_version") != 1:
        raise ValueError("official bridge requires temporal export schema 1")
    audit = temporal.get("temporal_audit_counts")
    if not isinstance(audit, Mapping) or audit.get("missing_frame_count") != 0:
        raise ValueError("temporal artifact has missing frame coverage")
    scene = str(temporal.get("scene", "")).strip()
    checkpoints = temporal.get("checkpoints")
    if not scene or not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("temporal artifact requires scene and checkpoints")
    timestamps = [int(checkpoint["timestamp_ns"]) for checkpoint in checkpoints]
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError("temporal checkpoint timestamps must be strictly increasing")
    base = temporal_manifest_path.parent
    schedule_source, scheduled = _scheduled_checkpoints(
        temporal, base=base, scene=scene
    )
    sources = temporal.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("temporal artifact sources are invalid")
    coverage_source = _source_path(
        sources.get("frame_coverage", {}), base, "frame coverage"
    )
    lifecycle_source = _source_path(
        sources.get("lifecycle_transitions", {}), base, "lifecycle transitions"
    )
    if (
        temporal.get("frame_coverage") != sources.get("frame_coverage")
        or temporal.get("lifecycle_transitions")
        != sources.get("lifecycle_transitions")
    ):
        raise ValueError("temporal explicit sidecar source binding mismatch")
    coverage = _load_frame_coverage(coverage_source.path)
    transitions = _load_lifecycle_transitions(lifecycle_source.path, coverage)
    observed = [
        (checkpoint.get("frame_index"), checkpoint.get("timestamp_ns"))
        for checkpoint in checkpoints
        if isinstance(checkpoint, Mapping)
    ]
    if len(observed) != len(checkpoints) or observed != scheduled:
        raise ValueError("temporal checkpoints do not exactly match causal schedule")

    snapshots = []
    source_paths = [
        temporal_source,
        label_source,
        schedule_source,
        coverage_source,
        lifecycle_source,
    ]
    states: dict[str, dict[str, Any]] = {}
    for position, checkpoint in enumerate(checkpoints):
        frame_index = checkpoint.get("frame_index")
        if (
            type(frame_index) is not int
            or checkpoint.get("consumed_through_frame") != frame_index
            or checkpoint.get("consumed_through_frame_exclusive") != frame_index + 1
        ):
            raise ValueError("temporal checkpoint causal boundary must be [0,t+1)")
        snapshot_source = _source_path(
            checkpoint["snapshot"], base, f"checkpoint {position} snapshot"
        )
        entities_source = _source_path(
            checkpoint["entities"], base, f"checkpoint {position} entities"
        )
        snapshot = read_map_snapshot(snapshot_source.path, entities_source.path)
        _assert_unchanged(
            snapshot_source, label=f"checkpoint {position} snapshot"
        )
        _assert_unchanged(
            entities_source, label=f"checkpoint {position} entities"
        )
        if (
            snapshot.scene_id != scene
            or snapshot.method != "OVIV2"
            or snapshot.scope != "current"
            or not _timestamp_matches(snapshot.timestamp, timestamps[position])
        ):
            raise ValueError("temporal neutral snapshot identity mismatch")
        snapshots.append(snapshot)
        source_paths.extend((snapshot_source, entities_source))
        for entity in snapshot.entities:
            entity_id = str(entity.entity_id)
            state = states.get(entity_id)
            if state is None:
                states[entity_id] = {
                    "positions": [position],
                    "semantic_label_name": entity.semantic_label,
                    "semantic_label_names": [entity.semantic_label],
                    "entity_type": str(entity.metadata.get("entity_type", "object")),
                    "source_entity_id": str(
                        entity.metadata.get(
                            "temporal_entity_id",
                            entity.metadata.get("owner_entity_id", entity_id),
                        )
                    ),
                }
            else:
                if state["entity_type"] != str(
                    entity.metadata.get("entity_type", "object")
                ):
                    raise ValueError("stable entity type conflict")
                state["positions"].append(position)
                state["semantic_label_names"].append(entity.semantic_label)
                source_entity_id = str(
                    entity.metadata.get(
                        "temporal_entity_id",
                        entity.metadata.get("owner_entity_id", entity_id),
                    )
                )
                if state["source_entity_id"] != source_entity_id:
                    raise ValueError("stable entity temporal binding conflict")

    trajectory_source = _source_path(
        temporal["trajectories"], base, "trajectories"
    )
    source_paths.append(trajectory_source)
    source_trajectories = _load_trajectories(
        trajectory_source.path, checkpoints, coverage
    )
    _assert_unchanged(trajectory_source, label="trajectories")
    all_samples = [
        sample for samples in source_trajectories.values() for sample in samples
    ]
    expected_audit = {
        "static_sample_count": sum(
            sample["dynamic_state"] == "static" for sample in all_samples
        ),
        "dynamic_sample_count": sum(
            sample["dynamic_state"] == "dynamic" for sample in all_samples
        ),
        "unknown_sample_count": sum(
            sample["dynamic_state"] == "unknown" for sample in all_samples
        ),
        "missing_frame_count": 0,
        "lifecycle_transition_count": len(transitions),
        "geometry_epoch_count": len(
            {
                (str(record["entity_id"]), int(record["geometry_epoch"]))
                for record in (*all_samples, *transitions)
            }
        ),
        "invalid_readout_sample_count": sum(
            sample["readout_valid"] is False for sample in all_samples
        ),
    }
    if dict(audit) != expected_audit:
        raise ValueError("temporal artifact audit counts mismatch")
    source_to_entity = {
        str(state["source_entity_id"]): entity_id for entity_id, state in states.items()
    }
    if len(source_to_entity) != len(states):
        raise ValueError("temporal entity source binding is ambiguous")
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for source_id, samples in source_trajectories.items():
        entity_id = source_to_entity.get(source_id, source_id)
        if entity_id in states:
            trajectories[entity_id] = list(samples)
    explicit_intervals = _explicit_presence_runs(
        states, coverage, source_trajectories, transitions
    )
    labels = _label_space(label_source.path, scene=scene)
    _assert_unchanged(label_source, label="label space")

    ordered_ids = sorted(
        states,
        key=lambda entity_id: (
            timestamps[int(states[entity_id]["positions"][0])],
            entity_id,
        ),
    )
    assignments = []
    assignment_by_id: dict[str, dict[str, Any]] = {}
    for node_index, entity_id in enumerate(ordered_ids):
        state = states[entity_id]
        intervals = explicit_intervals[entity_id]
        starts = [int(interval["start_ns"]) for interval in intervals]
        ends = [
            (
                int(interval["end_ns_exclusive"])
                if interval["end_ns_exclusive"] is not None
                else None
            )
            for interval in intervals
        ]
        label_name = state["semantic_label_name"]
        normalized = _normalize_label(label_name or "")
        semantic_observations = []
        for position, observed_label_name in zip(
            state["positions"], state["semantic_label_names"]
        ):
            observed_normalized = _normalize_label(observed_label_name or "")
            semantic_observations.append(
                {
                    "timestamp_ns": timestamps[int(position)],
                    "semantic_label_name": observed_label_name,
                    "semantic_label": labels.get(observed_normalized, 0),
                    "label_matched": (
                        observed_normalized in labels
                        and observed_normalized != "unknown"
                    ),
                }
            )
        assignment = {
            "entity_id": entity_id,
            "source_entity_id": str(state["source_entity_id"]),
            "node_index": node_index,
            "node_symbol": f"O{node_index}",
            "semantic_label_name": label_name,
            "semantic_label": labels.get(normalized, 0),
            "label_matched": normalized in labels and normalized != "unknown",
            "semantic_observations": semantic_observations,
            "entity_type": state["entity_type"],
            "first_observed_ns": starts,
            "last_observed_ns": ends,
            "presence_intervals": intervals,
            "native_trajectory_sample_count": len(trajectories.get(entity_id, ())),
            "latest_dynamic_state": (
                trajectories[entity_id][-1]["dynamic_state"]
                if trajectories.get(entity_id)
                else None
            ),
            "dynamic_track_eligible": bool(
                trajectories.get(entity_id)
                and trajectories[entity_id][-1]["dynamic_state"] == "dynamic"
            ),
        }
        assignments.append(assignment)
        assignment_by_id[entity_id] = assignment

    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FileExistsError(f"bridge output already exists: {output}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    artifact_root = staging
    checkpoint_outputs = []
    for position, (checkpoint, snapshot) in enumerate(zip(checkpoints, snapshots)):
        query = timestamps[position]
        frame = int(checkpoint["frame_index"])
        checkpoint_dir = artifact_root / "checkpoints" / f"{position:08d}"
        objects_dir = checkpoint_dir / "objects"
        trajectories_dir = checkpoint_dir / "trajectories"
        objects_dir.mkdir(parents=True)
        trajectories_dir.mkdir()
        background_path = checkpoint_dir / "background.ply"
        background = (
            np.empty((0, 3), dtype=np.float32)
            if snapshot.background_xyz is None
            else snapshot.background_xyz
        )
        _write_binary_ply(background_path, background)

        objects = []
        for entity in sorted(
            snapshot.entities,
            key=lambda item: int(assignment_by_id[item.entity_id]["node_index"]),
        ):
            assignment = assignment_by_id[entity.entity_id]
            semantic_observation = next(
                observation
                for observation in assignment["semantic_observations"]
                if observation["timestamp_ns"] == query
            )
            prefix = [
                sample
                for sample in trajectories.get(entity.entity_id, ())
                if int(sample["timestamp_ns"]) <= query
                and int(sample["frame_index"]) <= frame
            ]
            state_at_query = prefix[-1]["dynamic_state"] if prefix else None
            eligible = state_at_query == "dynamic"
            if not eligible:
                prefix = []
            points_path = objects_dir / f"O{assignment['node_index']}.ply"
            trajectory_output = trajectories_dir / f"O{assignment['node_index']}.json"
            _write_binary_ply(points_path, entity.points_xyz)
            _write_json(trajectory_output, prefix)
            starts, ends = _checkpoint_endpoints(
                assignment["presence_intervals"], query
            )
            objects.append(
                {
                    "entity_id": entity.entity_id,
                    "node_index": assignment["node_index"],
                    "node_symbol": assignment["node_symbol"],
                    "semantic_label_name": semantic_observation[
                        "semantic_label_name"
                    ],
                    "semantic_label": semantic_observation["semantic_label"],
                    "label_matched": semantic_observation["label_matched"],
                    "first_observed_ns": starts,
                    "last_observed_ns": ends,
                    "points_ply": str(points_path.relative_to(artifact_root)),
                    "points_sha256": _sha256(points_path),
                    "points_byte_count": points_path.stat().st_size,
                    "point_count": len(entity.points_xyz),
                    "trajectory_json": str(
                        trajectory_output.relative_to(artifact_root)
                    ),
                    "trajectory_sha256": _sha256(trajectory_output),
                    "trajectory_byte_count": trajectory_output.stat().st_size,
                    "trajectory_sample_count": len(prefix),
                    "dynamic_track_eligible": eligible,
                    "dynamic_state_at_query": state_at_query,
                    "trajectory_source": "native_per_frame_centroids_only",
                }
            )
        checkpoint_outputs.append(
            {
                "frame_index": int(checkpoint["frame_index"]),
                "timestamp_ns": query,
                "background_ply": str(background_path.relative_to(artifact_root)),
                "background_sha256": _sha256(background_path),
                "background_byte_count": background_path.stat().st_size,
                "background_point_count": len(background),
                "objects": objects,
            }
        )

    manifest = {
        "schema_version": 1,
        "mode": "temporal_checkpoints",
        "dataset": "TESSE-CD",
        "method": temporal.get("method"),
        "display_mode": "online",
        "scene_id": scene,
        "query_timestamps_ns": timestamps,
        "symbol_assignments": assignments,
        "checkpoints": checkpoint_outputs,
        "hashed_inputs": [_source_record(source) for source in source_paths],
        "protocol": {
            "node_symbol_order": "first_appearance_timestamp_then_entity_id",
            "presence_intervals": "closed_open",
            "ongoing_interval_endpoint": "query_timestamp_ns+1",
            "trajectory_bound": (
                "sample_frame_index<=query_frame_index and "
                "sample_timestamp_ns<=query_timestamp_ns"
            ),
            "dynamic_track_state_source": "latest_native_sample_at_or_before_query",
            "sparse_track_policy": "no_synthetic_trajectory",
        },
        "temporal_audit_counts": dict(audit),
    }
    manifest_path = artifact_root / "bridge_manifest.json"
    _write_json(manifest_path, manifest)
    try:
        for index, source in enumerate(source_paths):
            _assert_unchanged(source, label=f"hashed input {index}")
        validate_temporal_bridge_manifest(manifest_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    output.mkdir()
    try:
        os.rename(staging, output)
    except Exception as error:
        raise RuntimeError("temporal bridge publication-uncertain") from error
    return output / "bridge_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-manifest", type=Path, required=True)
    parser.add_argument("--label-space", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepare_temporal_bridge(args.temporal_manifest, args.label_space, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
