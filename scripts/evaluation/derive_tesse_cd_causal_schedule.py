#!/usr/bin/env python3
"""Freeze method-independent causal TESSE-CD checkpoint schedules."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Mapping, Sequence


SCENES = ("apartment", "office")
ROLE_ORDER = ("official", "common_v2")
CANONICAL_OWNER = "MIT-SPARK Khronos official release"
CANONICAL_MANIFEST_ID = "tesse_cd_dynamic_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _serialized_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_byte_count(entry: Mapping[str, Any]) -> int:
    value = entry.get("size_bytes", entry.get("byte_count"))
    if type(value) is not int or value < 0:
        raise ValueError("declared byte count must be a non-negative integer")
    return value


def _stat_fingerprint(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def _assert_source_unchanged(
    path: Path, expected: tuple[int, int, int, int], *, label: str
) -> None:
    try:
        observed = _stat_fingerprint(path)
    except OSError as error:
        raise ValueError(f"{label} source changed while reading") from error
    if observed != expected:
        raise ValueError(f"{label} source changed while reading")


def _verify_source(
    entry: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], tuple[int, int, int, int]]:
    path = Path(str(entry.get("path", "")))
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    fingerprint = _stat_fingerprint(path)
    observed_bytes = fingerprint[2]
    if _declared_byte_count(entry) != observed_bytes:
        raise ValueError(f"{label} byte count mismatch")
    observed_hash = _sha256(path)
    _assert_source_unchanged(path, fingerprint, label=label)
    if str(entry.get("sha256", "")) != observed_hash:
        raise ValueError(f"{label} SHA256 mismatch")
    return (
        {
            "path": _serialized_path(path),
            "sha256": observed_hash,
            "byte_count": observed_bytes,
        },
        fingerprint,
    )


def _validate_timestamps(timestamps: Sequence[int]) -> list[int]:
    values = [int(timestamp) for timestamp in timestamps]
    if not values:
        raise ValueError("depth timestamp stream is empty")
    if any(timestamp <= 0 for timestamp in values):
        raise ValueError("depth timestamps must be positive")
    for previous, current in zip(values, values[1:]):
        if current == previous:
            raise ValueError(f"duplicate depth timestamp: {current}")
        if current < previous:
            raise ValueError(
                f"non-monotonic depth timestamp: {current} follows {previous}"
            )
    return values


def load_depth_timestamps(database: Path, *, topic: str) -> list[int]:
    """Load depth timestamps in recorded message order from one ROS2 bag DB."""
    if not database.is_file():
        raise ValueError(f"database is not a file: {database}")
    with sqlite3.connect(
        f"file:{database}?mode=ro&immutable=1", uri=True
    ) as connection:
        topic_rows = connection.execute(
            "select id from topics where name = ?", (topic,)
        ).fetchall()
        if len(topic_rows) != 1:
            raise ValueError(f"expected exactly one topic row for {topic}")
        rows = connection.execute(
            "select timestamp from messages where topic_id = ? order by id",
            (topic_rows[0][0],),
        ).fetchall()
    return _validate_timestamps([int(row[0]) for row in rows])


def load_event_timestamps(
    changes_path: Path, *, last_relative_timestamp_ns: int
) -> list[int]:
    """Load in-stream intervention times, excluding terminal lifetime sentinels."""
    if not changes_path.is_file():
        raise ValueError(f"gt_changes is not a file: {changes_path}")
    with changes_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {
            "ObjectSymbol",
            "AppearedAt",
            "DisappearedAt",
        } <= set(reader.fieldnames):
            raise ValueError("gt_changes is missing appearance/disappearance fields")
        values: set[int] = set()
        for row in reader:
            if not str(row["ObjectSymbol"]).strip():
                raise ValueError("gt_changes ObjectSymbol must be non-empty")
            parsed: dict[str, int] = {}
            for field in ("AppearedAt", "DisappearedAt"):
                try:
                    timestamp = int(str(row[field]).strip())
                except (TypeError, ValueError) as error:
                    raise ValueError(f"gt_changes {field} must be an integer") from error
                if timestamp < 0:
                    raise ValueError(f"gt_changes {field} must be non-negative")
                parsed[field] = timestamp
            appeared = parsed["AppearedAt"]
            disappeared = parsed["DisappearedAt"]
            if appeared == 0 and disappeared == 0:
                raise ValueError("gt_changes lifecycle cannot be all zero")
            if appeared > 0 and disappeared > 0 and disappeared <= appeared:
                raise ValueError(
                    "gt_changes DisappearedAt must be greater than AppearedAt"
                )
            if appeared > int(last_relative_timestamp_ns):
                raise ValueError("gt_changes appearance exceeds last depth frame")
            if appeared > 0:
                values.add(appeared)
            if 0 < disappeared <= int(last_relative_timestamp_ns):
                values.add(disappeared)
    if not values:
        raise ValueError("gt_changes has no in-stream intervention timestamps")
    return sorted(values)


def _validate_parameters(
    *, official_stride: int, common_step: int, common_horizon: int
) -> None:
    if official_stride <= 0:
        raise ValueError("official_stride must be positive")
    if common_step <= 0:
        raise ValueError("common_step must be positive")
    if common_horizon < 0 or common_horizon % common_step:
        raise ValueError("common_horizon must be a non-negative multiple of common_step")
    if common_horizon // common_step + 1 != 10:
        raise ValueError("common v2 must define exactly ten event checkpoints")


def _validate_source_identity(manifest: Mapping[str, Any]) -> None:
    protocol = manifest.get("protocol", {})
    source = manifest.get("source", {})
    valid = (
        type(manifest.get("schema_version")) is int
        and manifest.get("schema_version") == 1
        and manifest.get("manifest_id") == CANONICAL_MANIFEST_ID
        and isinstance(source, Mapping)
        and source.get("owner") == CANONICAL_OWNER
        and isinstance(protocol, Mapping)
        and protocol.get("future_frames_allowed") is False
        and protocol.get("ground_truth_evaluator_only") is True
        and protocol.get("runtime_ground_truth_access") is False
    )
    if not valid:
        raise ValueError("source manifest does not match canonical source identity")


def derive_scene_schedule(
    scene: str,
    depth_timestamps: Sequence[int],
    event_timestamps: Sequence[int],
    *,
    official_stride: int = 450,
    common_step: int = 50,
    common_horizon: int = 450,
) -> dict[str, Any]:
    """Derive the immutable union of official and common-v2 frame schedules."""
    _validate_parameters(
        official_stride=official_stride,
        common_step=common_step,
        common_horizon=common_horizon,
    )
    timestamps = _validate_timestamps(depth_timestamps)
    relative = [timestamp - timestamps[0] for timestamp in timestamps]
    raw_events = [int(timestamp) for timestamp in event_timestamps]
    if not raw_events or any(timestamp <= 0 for timestamp in raw_events):
        raise ValueError("event timestamps must be non-empty and positive")
    if raw_events != sorted(set(raw_events)):
        raise ValueError("event timestamps must be unique and strictly increasing")

    official_frames = list(range(official_stride, len(timestamps), official_stride))
    if not official_frames:
        raise ValueError("official checkpoint schedule is empty")

    frames: dict[int, dict[str, set[str]]] = {}

    def frame_record(frame_index: int) -> dict[str, set[str]]:
        return frames.setdefault(frame_index, {"event_ids": set(), "roles": set()})

    for frame_index in official_frames:
        frame_record(frame_index)["roles"].add("official")

    events: list[dict[str, Any]] = []
    offsets = list(range(0, common_horizon + 1, common_step))
    for event_index, event_timestamp in enumerate(raw_events, start=1):
        intervention = bisect_left(relative, event_timestamp)
        if intervention >= len(timestamps) or intervention + common_horizon >= len(timestamps):
            raise ValueError(
                f"{scene} event {event_timestamp} does not have ten in-horizon checkpoints"
            )
        event_id = f"{scene}_event_{event_index:02d}"
        checkpoint_frames = [intervention + offset for offset in offsets]
        for frame_index in checkpoint_frames:
            record = frame_record(frame_index)
            record["roles"].add("common_v2")
            record["event_ids"].add(event_id)
        events.append(
            {
                "event_id": event_id,
                "event_relative_timestamp_ns": event_timestamp,
                "intervention_frame_index": intervention,
                "intervention_timestamp_ns": timestamps[intervention],
                "intervention_relative_timestamp_ns": relative[intervention],
                "common_checkpoint_frame_indices": checkpoint_frames,
            }
        )

    entries = [
        {
            "frame_index": frame_index,
            "timestamp_ns": timestamps[frame_index],
            "relative_timestamp_ns": relative[frame_index],
            "event_ids": sorted(record["event_ids"]),
            "roles": [role for role in ROLE_ORDER if role in record["roles"]],
        }
        for frame_index, record in sorted(frames.items())
    ]
    return {
        "frame_count": len(timestamps),
        "first_depth_timestamp_ns": timestamps[0],
        "last_depth_timestamp_ns": timestamps[-1],
        "events": events,
        "entries": entries,
    }


def build_schedule(
    manifest_path: Path,
    *,
    official_stride: int = 450,
    common_step: int = 50,
    common_horizon: int = 450,
) -> dict[str, Any]:
    """Validate declared sources and construct the combined schedule document."""
    _validate_parameters(
        official_stride=official_stride,
        common_step=common_step,
        common_horizon=common_horizon,
    )
    if not manifest_path.is_file():
        raise ValueError(f"manifest is not a file: {manifest_path}")
    manifest_fingerprint = _stat_fingerprint(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    _assert_source_unchanged(
        manifest_path, manifest_fingerprint, label="source manifest"
    )
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest.get("dataset") != "TESSE-CD":
        raise ValueError("source manifest is not TESSE-CD")
    _validate_source_identity(manifest)
    sequences = manifest.get("sequences", {})
    if set(sequences) != set(SCENES):
        raise ValueError("source manifest must contain apartment and office exactly")
    depth_topic = str(manifest.get("topics", {}).get("depth", ""))
    if not depth_topic:
        raise ValueError("source manifest has no depth topic")

    scenes: dict[str, Any] = {}
    for scene in SCENES:
        sequence = sequences[scene]
        database_entry = sequence["bag"]["database"]
        changes_entry = sequence["ground_truth"]["files"]["changes"]
        database_source, database_fingerprint = _verify_source(
            database_entry, label=f"{scene} database"
        )
        changes_source, changes_fingerprint = _verify_source(
            changes_entry, label=f"{scene} gt_changes"
        )
        timestamps = load_depth_timestamps(
            Path(database_source["path"]), topic=depth_topic
        )
        _assert_source_unchanged(
            Path(database_source["path"]),
            database_fingerprint,
            label=f"{scene} database",
        )
        timeline = sequence["timeline"]
        if len(timestamps) != int(timeline["depth_frame_count"]):
            raise ValueError(f"{scene} depth frame count mismatch")
        if timestamps[0] != int(timeline["first_depth_timestamp_ns"]):
            raise ValueError(f"{scene} first depth timestamp mismatch")
        if timestamps[-1] != int(timeline["last_depth_timestamp_ns"]):
            raise ValueError(f"{scene} last depth timestamp mismatch")
        event_timestamps = load_event_timestamps(
            Path(changes_source["path"]),
            last_relative_timestamp_ns=timestamps[-1] - timestamps[0],
        )
        _assert_source_unchanged(
            Path(changes_source["path"]),
            changes_fingerprint,
            label=f"{scene} gt_changes",
        )
        declared_events = [int(value) for value in timeline["change_times_relative_ns"]]
        if event_timestamps != declared_events:
            raise ValueError(f"{scene} gt_changes intervention timestamps mismatch")
        scene_schedule = derive_scene_schedule(
            scene,
            timestamps,
            event_timestamps,
            official_stride=official_stride,
            common_step=common_step,
            common_horizon=common_horizon,
        )
        scene_schedule["sources"] = {
            "database": database_source,
            "gt_changes": changes_source,
        }
        scenes[scene] = scene_schedule

    _assert_source_unchanged(
        manifest_path, manifest_fingerprint, label="source manifest"
    )
    return {
        "schema_version": 2,
        "manifest_id": "tesse_cd_causal_schedule_v2",
        "dataset": "TESSE-CD",
        "source_manifest": {
            "path": _serialized_path(manifest_path),
            "sha256": manifest_hash,
            "byte_count": manifest_fingerprint[2],
        },
        "parameters": {
            "frame_indexing": "zero_based",
            "event_frame_rule": "first relative_timestamp_ns >= event timestamp",
            "official_stride_frames": official_stride,
            "common_event_step_frames": common_step,
            "common_event_horizon_frames": common_horizon,
            "common_checkpoints_per_event": common_horizon // common_step + 1,
        },
        "method_predictions_used": False,
        "scenes": scenes,
    }


def render_schedule(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def run(
    manifest_path: Path,
    output: Path,
    *,
    official_stride: int = 450,
    common_step: int = 50,
    common_horizon: int = 450,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise ValueError(f"output already exists: {output}")
    payload = build_schedule(
        manifest_path,
        official_stride=official_stride,
        common_step=common_step,
        common_horizon=common_horizon,
    )
    content = render_schedule(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    directory_descriptor: int | None = None
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(output.parent, directory_flags)
        os.fsync(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-stride", type=int, default=450)
    parser.add_argument("--common-step", type=int, default=50)
    parser.add_argument("--common-horizon", type=int, default=450)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(
        args.manifest,
        args.output,
        official_stride=args.official_stride,
        common_step=args.common_step,
        common_horizon=args.common_horizon,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
