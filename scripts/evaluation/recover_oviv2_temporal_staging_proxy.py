#!/usr/bin/env python3
"""Recover a nonformal temporal export proxy from a preserved OVIV2 staging tree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.json_contracts import loads_strict


NONFORMAL_STATUS = "NONFORMAL_RECOVERY_PROXY_NOT_SUBMISSION_EVIDENCE"
_SOURCE_RECORD_FIELDS = frozenset({"path", "sha256", "byte_count"})
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
        "frame_coverage",
        "lifecycle_transitions",
        "checkpoint_statuses",
    }
)
_CHECKPOINT_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "checkpoint_frame",
        "timestamp_ns",
        "consumed_through_frame",
        "consumed_through_frame_exclusive",
    }
)
_CHECKPOINT_STATUS_FIELDS_WITH_EVENTS = _CHECKPOINT_STATUS_FIELDS | {
    "event_ids",
    "roles",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class _FileWitness:
    path: Path
    data: bytes
    sha256: str
    byte_count: int
    device: int
    inode: int
    mode: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def capture(cls, path: Path) -> _FileWitness:
        resolved = path.resolve(strict=True)
        before = os.stat(resolved, follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise ValueError(f"source must be a regular non-symlink file: {path}")
        data = resolved.read_bytes()
        after = os.stat(resolved, follow_symlinks=False)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"source changed while reading: {path}")
        return cls(
            path=resolved,
            data=data,
            sha256=_sha256(data),
            byte_count=len(data),
            device=after.st_dev,
            inode=after.st_ino,
            mode=after.st_mode,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        )

    def revalidate(self) -> None:
        current = os.stat(self.path, follow_symlinks=False)
        if (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ) != (
            self.device,
            self.inode,
            self.mode,
            self.byte_count,
            self.mtime_ns,
            self.ctime_ns,
        ) or _sha256(self.path.read_bytes()) != self.sha256:
            raise ValueError(f"source changed before publication: {self.path}")


def _read_json(witness: _FileWitness, *, label: str) -> dict[str, Any]:
    try:
        text = witness.data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    payload = loads_strict(text, label=label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("source record path must be a nonempty string")
    path = Path(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("source record path must be canonical and relative")
    return path


def _record(witness: _FileWitness, *, root: Path) -> dict[str, Any]:
    return {
        "path": witness.path.relative_to(root).as_posix(),
        "sha256": witness.sha256,
        "byte_count": witness.byte_count,
    }


def _resolve_record(
    record: object,
    *,
    root: Path,
    label: str,
) -> tuple[_FileWitness, Path]:
    if not isinstance(record, Mapping) or set(record) != _SOURCE_RECORD_FIELDS:
        raise ValueError(f"{label} source record fields are invalid")
    relative = _relative_path(record.get("path"))
    witness = _FileWitness.capture(root / relative)
    if dict(record) != _record(witness, root=root):
        raise ValueError(f"{label} source record mismatch")
    return witness, relative


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _copy_witness(witness: _FileWitness, *, relative: Path, staging: Path) -> None:
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(witness.data)
    if _sha256(destination.read_bytes()) != witness.sha256:
        raise ValueError(f"staged copy mismatch: {relative.as_posix()}")


def _schedule_entries(
    schedule: Mapping[str, Any], *, scene: str
) -> list[dict[str, int]]:
    if frozenset(schedule) not in {_SCHEDULE_FIELDS, _SCHEDULE_FIELDS_WITH_SOURCE}:
        raise ValueError("schedule fields are invalid")
    parameters = schedule.get("parameters")
    if not isinstance(parameters, Mapping) or frozenset(parameters) not in {
        frozenset({"frame_indexing"}),
        _SCHEDULE_PARAMETER_FIELDS,
    }:
        raise ValueError("schedule parameter fields are invalid")
    scenes = schedule.get("scenes")
    scene_schedule = scenes.get(scene) if isinstance(scenes, Mapping) else None
    if not isinstance(scene_schedule, Mapping) or frozenset(scene_schedule) not in {
        _SCHEDULE_SCENE_FIELDS,
        _SCHEDULE_SCENE_FIELDS_WITH_SOURCES,
    }:
        raise ValueError("schedule scene fields are invalid")
    frame_count = scene_schedule.get("frame_count")
    raw_entries = scene_schedule.get("entries")
    if type(frame_count) is not int or frame_count <= 0 or not isinstance(
        raw_entries, list
    ):
        raise ValueError("schedule scene entries are invalid")
    entries: list[dict[str, int]] = []
    previous_frame = -1
    previous_timestamp = -1
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping) or frozenset(raw_entry) not in {
            _SCHEDULE_ENTRY_FIELDS,
            _SCHEDULE_ENTRY_FIELDS_FULL,
        }:
            raise ValueError("schedule entry fields are invalid")
        frame_index = raw_entry.get("frame_index")
        timestamp_ns = raw_entry.get("timestamp_ns")
        if (
            type(frame_index) is not int
            or type(timestamp_ns) is not int
            or frame_index <= previous_frame
            or timestamp_ns <= previous_timestamp
            or frame_index < 0
            or frame_index >= frame_count
            or timestamp_ns <= 0
        ):
            raise ValueError("schedule frames and timestamps are invalid")
        entries.append({"frame_index": frame_index, "timestamp_ns": timestamp_ns})
        previous_frame = frame_index
        previous_timestamp = timestamp_ns
    return entries


def _checkpoint_identity(status: Mapping[str, Any]) -> tuple[int, int]:
    if frozenset(status) not in {
        _CHECKPOINT_STATUS_FIELDS,
        _CHECKPOINT_STATUS_FIELDS_WITH_EVENTS,
    }:
        raise ValueError("checkpoint status fields are invalid")
    frame_index = status.get("checkpoint_frame")
    timestamp_ns = status.get("timestamp_ns")
    consumed = status.get("consumed_through_frame")
    consumed_exclusive = status.get("consumed_through_frame_exclusive")
    if (
        status.get("schema_version") != 1
        or status.get("status") != "PASS"
        or type(frame_index) is not int
        or type(timestamp_ns) is not int
        or type(consumed) is not int
        or type(consumed_exclusive) is not int
        or frame_index < 0
        or timestamp_ns <= 0
        or consumed != frame_index
        or consumed_exclusive != frame_index + 1
    ):
        raise ValueError("checkpoint status identity mismatch")
    return frame_index, timestamp_ns


def _require_distinct_sources(
    sources: Sequence[tuple[_FileWitness, Path]],
) -> None:
    identities = [(witness.device, witness.inode) for witness, _ in sources]
    paths = [witness.path for witness, _ in sources]
    if len(identities) != len(set(identities)) or len(paths) != len(set(paths)):
        raise ValueError("source file roles must be distinct")


def _official_checkpoint_statuses(
    *,
    source_root: Path,
    entries: Sequence[Mapping[str, int]],
) -> dict[tuple[int, int], tuple[_FileWitness, Path, dict[str, Any]]]:
    checkpoints_root = source_root / "checkpoints"
    if checkpoints_root.is_symlink() or not checkpoints_root.is_dir():
        raise ValueError("checkpoints root must be a regular directory")
    official_identities = {
        (int(entry["frame_index"]), int(entry["timestamp_ns"])) for entry in entries
    }
    matches: dict[
        tuple[int, int], list[tuple[_FileWitness, Path, dict[str, Any]]]
    ] = {identity: [] for identity in official_identities}
    for checkpoint_root in sorted(checkpoints_root.iterdir()):
        if checkpoint_root.is_symlink():
            raise ValueError("checkpoint directory must not be a symlink")
        if not checkpoint_root.is_dir():
            continue
        status_path = checkpoint_root / "checkpoint_status.json"
        if not status_path.exists() and not status_path.is_symlink():
            continue
        witness = _FileWitness.capture(status_path)
        status = _read_json(witness, label="checkpoint status")
        identity = _checkpoint_identity(status)
        if identity in official_identities:
            matches[identity].append(
                (witness, witness.path.relative_to(source_root), status)
            )
    resolved: dict[tuple[int, int], tuple[_FileWitness, Path, dict[str, Any]]] = {}
    for identity, candidates in matches.items():
        if len(candidates) != 1:
            raise ValueError("official checkpoint identity is not unique")
        resolved[identity] = candidates[0]
    return resolved


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir():
            directories.append(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def recover_temporal_staging_proxy(
    *,
    source_root: Path,
    output_root: Path,
) -> Path:
    source_root = source_root.resolve(strict=True)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("source root must be a regular directory")
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    schedule_witness = _FileWitness.capture(source_root / "inputs/schedule.json")
    schedule = _read_json(schedule_witness, label="schedule")
    capture_witness = _FileWitness.capture(source_root / "capture_status.json")
    capture = _read_json(capture_witness, label="capture status")
    if frozenset(capture) != _CAPTURE_STATUS_FIELDS:
        raise ValueError("capture status fields are invalid")
    scene = capture.get("scene")
    scenes = schedule.get("scenes")
    if (
        schedule.get("dataset") != "TESSE-CD"
        or not isinstance(scene, str)
        or not isinstance(scenes, Mapping)
        or scene not in scenes
        or capture.get("schema_version") != 1
        or capture.get("status") != "PASS"
        or capture.get("mode") != "causal_checkpoints"
    ):
        raise ValueError("capture and schedule identity mismatch")
    entries = _schedule_entries(schedule, scene=scene)
    expected_frames = [entry.get("frame_index") for entry in entries]
    if (
        capture.get("scheduled_frame_indices") != expected_frames
        or capture.get("captured_frame_indices") != expected_frames
    ):
        raise ValueError("capture does not exactly cover schedule")

    source_items: list[tuple[_FileWitness, Path]] = [
        (schedule_witness, Path("inputs/schedule.json")),
        (capture_witness, Path("capture_status.json")),
    ]
    stream_roles = {
        "schedule": (schedule_witness, Path("inputs/schedule.json")),
        "trajectories": (None, None),
        "frame_coverage": (None, None),
        "lifecycle_transitions": (None, None),
    }
    if capture.get("schedule") != _record(schedule_witness, root=source_root):
        raise ValueError("capture schedule source record mismatch")
    for role in ("trajectories", "frame_coverage", "lifecycle_transitions"):
        witness, relative = _resolve_record(
            capture.get(role), root=source_root, label=role.replace("_", " ")
        )
        stream_roles[role] = (witness, relative)
        source_items.append((witness, relative))

    raw_statuses = capture.get("checkpoint_statuses")
    if not isinstance(raw_statuses, list) or len(raw_statuses) != len(entries):
        raise ValueError("capture checkpoint status count mismatch")
    official_statuses = _official_checkpoint_statuses(
        source_root=source_root,
        entries=entries,
    )
    checkpoints: list[dict[str, Any]] = []
    for entry, raw_status in zip(entries, raw_statuses, strict=True):
        declared_witness, declared_relative = _resolve_record(
            raw_status, root=source_root, label="checkpoint status"
        )
        frame_index = entry.get("frame_index")
        timestamp_ns = entry.get("timestamp_ns")
        status_witness, status_relative, status = official_statuses[
            (frame_index, timestamp_ns)
        ]
        if declared_witness.path != status_witness.path or declared_relative != status_relative:
            raise ValueError("capture checkpoint status does not bind official checkpoint")
        if _checkpoint_identity(status) != (frame_index, timestamp_ns):
            raise ValueError("checkpoint status identity mismatch")
        if (
            status.get("consumed_through_frame") != frame_index
            or status.get("consumed_through_frame_exclusive") != frame_index + 1
        ):
            raise ValueError("checkpoint status identity mismatch")
        neutral_root = status_witness.path.parent / "neutral_current"
        snapshot_root = neutral_root / "snapshots"
        entities_root = neutral_root / "entities"
        if any(path.is_symlink() for path in (neutral_root, snapshot_root, entities_root)):
            raise ValueError("neutral checkpoint tree must not contain symlinks")
        snapshot_paths = sorted(snapshot_root.glob("*.npz"))
        entities_paths = sorted(entities_root.glob("*.jsonl"))
        if len(snapshot_paths) != 1 or len(entities_paths) != 1:
            raise ValueError("official checkpoint must contain one neutral snapshot")
        snapshot_witness = _FileWitness.capture(snapshot_paths[0])
        entities_witness = _FileWitness.capture(entities_paths[0])
        snapshot_relative = snapshot_witness.path.relative_to(source_root)
        entities_relative = entities_witness.path.relative_to(source_root)
        source_items.extend(
            [
                (status_witness, status_relative),
                (snapshot_witness, snapshot_relative),
                (entities_witness, entities_relative),
            ]
        )
        checkpoints.append(
            {
                "frame_index": frame_index,
                "timestamp_ns": timestamp_ns,
                "consumed_through_frame": frame_index,
                "consumed_through_frame_exclusive": frame_index + 1,
                "checkpoint_status": _record(status_witness, root=source_root),
                "snapshot": _record(snapshot_witness, root=source_root),
                "entities": _record(entities_witness, root=source_root),
            }
        )

    _require_distinct_sources(source_items)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        for witness, relative in source_items:
            _copy_witness(witness, relative=relative, staging=staging)
        index_payload = {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoint_exports",
            "method": "OVIV2",
            "scene": scene,
            "schedule": _record(
                _FileWitness.capture(staging / "inputs/schedule.json"), root=staging
            ),
            "capture_status": _record(
                _FileWitness.capture(staging / "capture_status.json"), root=staging
            ),
            "trajectories": _record(
                _FileWitness.capture(staging / stream_roles["trajectories"][1]),
                root=staging,
            ),
            "frame_coverage": _record(
                _FileWitness.capture(staging / stream_roles["frame_coverage"][1]),
                root=staging,
            ),
            "lifecycle_transitions": _record(
                _FileWitness.capture(
                    staging / stream_roles["lifecycle_transitions"][1]
                ),
                root=staging,
            ),
            "checkpoints": checkpoints,
        }
        index_path = staging / "source_index.json"
        _write_json(index_path, index_payload)
        index_witness = _FileWitness.capture(index_path)
        tool_witness = _FileWitness.capture(Path(__file__))
        _write_json(
            staging / "recovery_receipt.json",
            {
                "schema_version": 1,
                "status": NONFORMAL_STATUS,
                "submission_eligible": False,
                "source_root": str(source_root),
                "source_files": [
                    {
                        "path": str(witness.path),
                        "sha256": witness.sha256,
                        "byte_count": witness.byte_count,
                    }
                    for witness, _ in sorted(
                        source_items, key=lambda item: item[0].path.as_posix()
                    )
                ],
                "source_index": _record(index_witness, root=staging),
                "recovery_tool": {
                    "path": str(tool_witness.path),
                    "sha256": tool_witness.sha256,
                    "byte_count": tool_witness.byte_count,
                },
            },
        )
        for witness, _ in source_items:
            witness.revalidate()
        _fsync_tree(staging)
        os.rename(staging, output_root)
        parent_fd = os.open(
            output_root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_root / "source_index.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_index = recover_temporal_staging_proxy(
        source_root=args.source_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {"source_index": str(source_index.resolve()), "status": NONFORMAL_STATUS},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
