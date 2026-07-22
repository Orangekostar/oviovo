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


def _load_trajectories(
    path: Path, checkpoints: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    frames = [int(checkpoint["frame_index"]) for checkpoint in checkpoints]
    trajectories: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[int, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = loads_strict(line, label="temporal trajectory")
        if not isinstance(record, Mapping) or frozenset(record) not in {
            frozenset({"frame_index", "timestamp_ns", "entity_id", "centroid_xyz"}),
            frozenset({
                "frame_index",
                "timestamp_ns",
                "entity_id",
                "centroid_xyz",
                "observation_count",
            }),
        }:
            raise ValueError("temporal trajectory fields are invalid")
        frame = record.get("frame_index")
        timestamp = record.get("timestamp_ns")
        entity_id = str(record.get("entity_id", "")).strip()
        if type(frame) is not int or type(timestamp) is not int or not entity_id:
            raise ValueError("temporal trajectory record is invalid")
        if frame < 0 or timestamp <= 0:
            raise ValueError("temporal trajectory record is invalid")
        if "observation_count" in record and (
            type(record["observation_count"]) is not int
            or record["observation_count"] < 0
        ):
            raise ValueError("temporal trajectory observation count is invalid")
        position = bisect_left(frames, frame)
        if position == len(checkpoints):
            continue
        if timestamp > int(checkpoints[position]["timestamp_ns"]):
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
                "frame_index": frame,
                "timestamp_ns": timestamp,
                "centroid_xyz": centroid.tolist(),
            }
        )
    for entity_id, samples in trajectories.items():
        timestamps = [int(sample["timestamp_ns"]) for sample in samples]
        if timestamps != sorted(set(timestamps)):
            raise ValueError(f"trajectory timestamps are not unique for {entity_id}")
    return trajectories


def _label_space(path: Path) -> dict[str, int]:
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
        if (
            not isinstance(entity_id, str)
            or not entity_id
            or entity_id in assignments
            or assignment.get("node_index") != node_index
            or assignment.get("node_symbol") != f"O{node_index}"
            or type(semantic_label) is not int
            or semantic_label < 0
        ):
            raise ValueError("temporal bridge assignment identity is invalid")
        intervals = assignment.get("presence_intervals")
        if not isinstance(intervals, list) or not intervals:
            raise ValueError("temporal bridge assignment intervals are invalid")
        starts: list[int] = []
        ends: list[int] = []
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
            if end is not None:
                ends.append(end)
                previous_end = end
            else:
                previous_end = start
        if (
            assignment.get("first_observed_ns") != starts
            or assignment.get("last_observed_ns") != ends
        ):
            raise ValueError("temporal bridge assignment endpoints are invalid")
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
                    "semantic_label_name",
                    "semantic_label",
                    "label_matched",
                )
            ):
                raise ValueError("temporal bridge object assignment mismatch")
            expected_starts = [
                value for value in assignment["first_observed_ns"] if value <= query
            ]
            expected_ends = [
                value for value in assignment["last_observed_ns"] if value <= query
            ]
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
            if any(int(sample["timestamp_ns"]) > query for sample in trajectory):
                raise ValueError("bridge trajectory sample is later than query")
            if not obj["dynamic_track_eligible"] and trajectory:
                raise ValueError("ineligible dynamic track must have no trajectory")
            if obj["dynamic_track_eligible"] and len(trajectory) < 2:
                raise ValueError("dynamic track requires at least two native samples")
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
    observed = [
        (checkpoint.get("frame_index"), checkpoint.get("timestamp_ns"))
        for checkpoint in checkpoints
        if isinstance(checkpoint, Mapping)
    ]
    if len(observed) != len(checkpoints) or observed != scheduled:
        raise ValueError("temporal checkpoints do not exactly match causal schedule")

    snapshots = []
    source_paths = [temporal_source, label_source, schedule_source]
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
                    "entity_type": str(entity.metadata.get("entity_type", "object")),
                }
            else:
                if state["semantic_label_name"] != entity.semantic_label:
                    raise ValueError("stable entity semantic conflict")
                if state["entity_type"] != str(entity.metadata.get("entity_type", "object")):
                    raise ValueError("stable entity type conflict")
                state["positions"].append(position)

    trajectory_source = _source_path(
        temporal["trajectories"], base, "trajectories"
    )
    source_paths.append(trajectory_source)
    trajectories = _load_trajectories(trajectory_source.path, checkpoints)
    _assert_unchanged(trajectory_source, label="trajectories")
    labels = _label_space(label_source.path)
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
        intervals = _presence_runs(state["positions"], checkpoints)
        starts = [int(interval["start_ns"]) for interval in intervals]
        ends = [
            int(interval["end_ns_exclusive"])
            for interval in intervals
            if interval["end_ns_exclusive"] is not None
        ]
        label_name = state["semantic_label_name"]
        normalized = _normalize_label(label_name or "")
        assignment = {
            "entity_id": entity_id,
            "node_index": node_index,
            "node_symbol": f"O{node_index}",
            "semantic_label_name": label_name,
            "semantic_label": labels.get(normalized, 0),
            "label_matched": normalized in labels and normalized != "unknown",
            "entity_type": state["entity_type"],
            "first_observed_ns": starts,
            "last_observed_ns": ends,
            "presence_intervals": intervals,
            "native_trajectory_sample_count": len(trajectories.get(entity_id, ())),
            "dynamic_track_eligible": len(trajectories.get(entity_id, ())) >= 2,
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
            prefix = [
                sample
                for sample in trajectories.get(entity.entity_id, ())
                if int(sample["timestamp_ns"]) <= query
            ]
            eligible = len(prefix) >= 2
            if not eligible:
                prefix = []
            points_path = objects_dir / f"O{assignment['node_index']}.ply"
            trajectory_output = trajectories_dir / f"O{assignment['node_index']}.json"
            _write_binary_ply(points_path, entity.points_xyz)
            _write_json(trajectory_output, prefix)
            starts = [
                timestamp
                for timestamp in assignment["first_observed_ns"]
                if timestamp <= query
            ]
            ends = [
                timestamp
                for timestamp in assignment["last_observed_ns"]
                if timestamp <= query
            ]
            objects.append(
                {
                    "entity_id": entity.entity_id,
                    "node_index": assignment["node_index"],
                    "node_symbol": assignment["node_symbol"],
                    "semantic_label_name": assignment["semantic_label_name"],
                    "semantic_label": assignment["semantic_label"],
                    "label_matched": assignment["label_matched"],
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
            "trajectory_bound": "sample_timestamp_ns<=query_timestamp_ns",
            "dynamic_track_min_native_samples": 2,
            "sparse_track_policy": "no_synthetic_trajectory",
        },
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
