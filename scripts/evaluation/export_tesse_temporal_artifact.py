#!/usr/bin/env python3
"""Assemble hash-bound causal TESSE-CD neutral checkpoint artifacts."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.exporters.oviovo import read_map_snapshot


@dataclass(frozen=True)
class _VerifiedSource:
    path: Path
    sha256: str
    byte_count: int
    fingerprint: tuple[int, int, int, int]

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path.resolve()),
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }


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
    payload = json.loads(source.path.read_text(encoding="utf-8"))
    _assert_unchanged(source, label=label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _same_source(left: Mapping[str, Any], right: _VerifiedSource, *, base: Path) -> bool:
    raw_path = Path(str(left.get("path", "")))
    path = (raw_path if raw_path.is_absolute() else base / raw_path).resolve()
    return (
        path == right.path
        and left.get("sha256") == right.sha256
        and left.get("byte_count") == right.byte_count
    )


def _load_schedule(
    source: _VerifiedSource, *, scene: str
) -> tuple[dict[str, Any], list[dict[str, int]]]:
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
    return dict(scene_payload), entries


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def export_temporal_artifact(source_index: Path, output: Path) -> Path:
    index_source = _direct_source(source_index, label="source index")
    index = _read_json(index_source, label="source index")
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
    _, schedule = _load_schedule(schedule_source, scene=scene)
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

    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.mkdir(parents=True)
    output_checkpoints: list[dict[str, Any]] = []
    for checkpoint in checkpoint_inputs:
        frame = int(checkpoint["frame_index"])
        directory = output / "checkpoints" / f"{frame:08d}"
        directory.mkdir(parents=True)
        snapshot_path = directory / "snapshot.npz"
        entities_path = directory / "entities.jsonl"
        shutil.copyfile(checkpoint["snapshot"].path, snapshot_path)
        shutil.copyfile(checkpoint["entities"].path, entities_path)
        _assert_unchanged(checkpoint["snapshot"], label=f"checkpoint {frame} snapshot")
        _assert_unchanged(checkpoint["entities"], label=f"checkpoint {frame} entities")
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
                "snapshot": _output_record(snapshot_path, output=output),
                "entities": _output_record(entities_path, output=output),
            }
        )

    trajectories_path = output / "trajectories.jsonl"
    trajectories_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for record in trajectory_records
        ),
        encoding="utf-8",
    )
    for source, label in verified:
        _assert_unchanged(source, label=label)

    manifest = {
        "schema_version": 1,
        "dataset": "TESSE-CD",
        "mode": "causal_checkpoints",
        "method": method,
        "scene": scene,
        "sources": {
            "source_index": index_source.record(),
            "schedule": schedule_source.record(),
            "capture_status": capture_source.record(),
            "trajectories": trajectory_source.record(),
            "checkpoint_statuses": [
                source.record() for source in checkpoint_status_sources
            ],
        },
        "checkpoints": output_checkpoints,
        "entity_lifecycles": lifecycles,
        "trajectories": _output_record(trajectories_path, output=output),
    }
    manifest_path = output / "temporal_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


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
