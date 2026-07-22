#!/usr/bin/env python3
"""Finalize OVIV2 common-v2 metrics from canonical independent repeats."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import fmean
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import finalize_tesse_common_v2 as pinned_common
from scripts.evaluation.canonicalize_tesse_common_v2_summary import (
    CapturedArtifact,
    EXTERNAL_SOURCE_ROLES,
    FileIdentity,
    canonical_summary_bytes,
    capture_and_canonicalize_summary,
    _json_exact_equal,
    _stable_regular_file_with_identity,
)
from scripts.evaluation.finalize_tesse_t2 import (
    _absolute_lexical,
    _atomic_json_no_replace,
    _open_directory_no_symlinks,
    _stable_regular_file,
)
from scripts.evaluation.freeze_oviv2_tesse_cd import (
    STAGE3_LINEAGE_COMMIT,
    _algorithm_config,
    _algorithm_hash,
    _validate_repository,
)
from src.evaluation.json_contracts import loads_strict


SCENES = ("apartment", "office")
REPEATS = (1, 2)
_REQUIRED_SHARED_BINDINGS = frozenset(
    {
        "common_target_manifest",
        "common_target_arrays",
        "schedule",
        "alias_map",
        "evaluator",
        "finalizers",
    }
)
_FORMAL_IDENTITY_FIELDS = frozenset({"frozen_run_identity", "run_execution"})
_RUN_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "dataset",
        "method_id",
        "mode",
        "scene",
        "missing_observation_policy",
        "algorithm_hash",
        "normalized_algorithm_config",
        "stage3_lineage_commit",
        "maintenance_parameters",
        "processed_frame_count",
        "official_schedule_frame_indices",
        "evaluation_checkpoint_frames",
        "scheduled_frame_indices",
        "captured_frame_indices",
        "config",
        "schedule",
        "source_bindings",
        "occlusion_checkpoint_index",
        "checkpoints",
    }
) | _FORMAL_IDENTITY_FIELDS
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
) | _FORMAL_IDENTITY_FIELDS
_OCCLUSION_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "dataset",
        "method_id",
        "scene",
        "algorithm_hash",
        "run_config",
        "target_manifest",
        "evaluation_checkpoint_frames_sha256",
        "snapshots",
    }
) | _FORMAL_IDENTITY_FIELDS
_TEMPORAL_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "dataset",
        "mode",
        "method",
        "scene",
        "sources",
        "checkpoints",
        "entity_lifecycles",
        "trajectories",
    }
) | _FORMAL_IDENTITY_FIELDS
_SOURCE_CHECKPOINT_FIELDS = frozenset(
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
_TEMPORAL_CHECKPOINT_FIELDS = frozenset(
    {
        "frame_index",
        "timestamp_ns",
        "consumed_through_frame",
        "consumed_through_frame_exclusive",
        "snapshot",
        "entities",
    }
)
_OCCLUSION_SNAPSHOT_FIELDS = frozenset(
    {
        "scene",
        "frame_index",
        "timestamp_ns",
        "relative_timestamp_ns",
        "consumed_through_frame",
        "consumed_through_frame_exclusive",
        "format",
        "path",
        "checksums_sha256",
    }
)
_RUN_CHECKPOINT_BASE_FIELDS = frozenset(
    {
        "scene",
        "frame_index",
        "timestamp_ns",
        "relative_timestamp_ns",
        "consumed_through_frame",
        "consumed_through_frame_exclusive",
        "event_ids",
        "roles",
        "checkpoint_status",
    }
)
_FULL_RUN_CHECKPOINT_FIELDS = _RUN_CHECKPOINT_BASE_FIELDS | {
    "voxel_snapshot",
    "artifact",
    "neutral_snapshot",
    "neutral_entities",
}
_COMPACT_RUN_CHECKPOINT_FIELDS = _RUN_CHECKPOINT_BASE_FIELDS | {
    "format",
    "ownership_checkpoint",
}
_ENTITY_LIFECYCLE_FIELDS = frozenset(
    {"entity_id", "semantic_label", "entity_type", "presence_intervals"}
)
_PRESENCE_INTERVAL_FIELDS = frozenset(
    {
        "first_frame_index",
        "first_timestamp_ns",
        "last_frame_index",
        "last_timestamp_ns",
    }
)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, object]]:
    digest, byte_count, content = _stable_regular_file(path, label=label, capture=True)
    assert content is not None
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    payload = loads_strict(text, label=label)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return dict(payload), {
        "role": "freeze_manifest",
        "sha256": digest,
        "byte_count": byte_count,
    }


def _verified_binding(
    value: object,
    *,
    label: str,
    expected_path: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    path, record, _ = _verified_binding_data(
        value,
        label=label,
        expected_path=expected_path,
        capture=False,
    )
    return path, record


def _verified_binding_data(
    value: object,
    *,
    label: str,
    expected_path: Path | None = None,
    capture: bool,
) -> tuple[Path, dict[str, object], bytes | None]:
    declaration = _mapping(value, label=f"{label} binding")
    if set(declaration) != {"path", "sha256", "byte_count"}:
        raise ValueError(f"{label} binding fields are invalid")
    raw_path = declaration.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = _absolute_lexical(Path(raw_path))
    if raw_path != os.fspath(path):
        raise ValueError(f"{label} path must be canonical")
    if expected_path is not None and path != _absolute_lexical(expected_path):
        raise ValueError(f"{label} path mismatch")
    digest, byte_count, content = _stable_regular_file(
        path, label=f"{label} file", capture=capture
    )
    if (
        not isinstance(declaration.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(declaration.get("sha256")))
        or type(declaration.get("byte_count")) is not int
        or declaration.get("sha256") != digest
        or declaration.get("byte_count") != byte_count
    ):
        raise ValueError(f"{label} content mismatch")
    return (
        path,
        {
            "role": label,
            "sha256": digest,
            "byte_count": byte_count,
        },
        content,
    )


def _direct_root(path: Path, *, label: str) -> tuple[Path, FileIdentity]:
    absolute = _absolute_lexical(path)
    descriptor, _ = _open_directory_no_symlinks(absolute, label=label)
    try:
        status = os.fstat(descriptor)
        identity = (status.st_dev, status.st_ino)
    finally:
        os.close(descriptor)
    return absolute, identity


def _canonical_record(payload: Mapping[str, Any], *, role: str) -> dict[str, object]:
    data = canonical_summary_bytes(payload)
    return {
        "role": role,
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _compact_json_hash(payload: Mapping[str, Any]) -> str:
    content = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _captured_json_artifact(
    path: Path,
    *,
    label: str,
    artifact_root: tuple[Path, FileIdentity],
) -> tuple[dict[str, Any], dict[str, object], FileIdentity]:
    digest, byte_count, content, identity = _stable_regular_file_with_identity(
        path,
        label=label,
        capture=True,
        expected_ancestor=artifact_root,
    )
    assert content is not None
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    payload = loads_strict(text, label=label)
    return (
        dict(_mapping(payload, label=label)),
        {"sha256": digest, "byte_count": byte_count},
        identity,
    )


def _validate_run_execution(
    value: object,
    *,
    scene: str,
    repeat: int,
    root: Path,
    root_identity: FileIdentity,
) -> dict[str, Any]:
    execution = dict(_mapping(value, label=f"{scene}.run{repeat} execution"))
    base = {
        "schema_version": 1,
        "run_slot": f"{scene}_run{repeat}",
        "output_root": os.fspath(root),
        "root_device": root_identity[0],
        "root_inode": root_identity[1],
    }
    expected = {**base, "execution_id": _compact_json_hash(base)}
    if not _json_exact_equal(execution, expected):
        raise ValueError(f"{scene}.run{repeat} execution identity mismatch")
    return execution


def _content_record(value: object, *, label: str) -> dict[str, object]:
    record = _mapping(value, label=label)
    if set(record) != {"sha256", "byte_count"} or not (
        isinstance(record.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"]))
        and type(record.get("byte_count")) is int
        and int(record["byte_count"]) >= 0
    ):
        raise ValueError(f"{label} is invalid")
    return dict(record)


def _relative_content_record(
    value: object,
    *,
    label: str,
    expected_path: str | None = None,
    allow_parent: bool = False,
) -> dict[str, object]:
    record = _mapping(value, label=label)
    if set(record) != {"path", "sha256", "byte_count"}:
        raise ValueError(f"{label} is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        raise ValueError(f"{label} path is invalid")
    path = Path(raw_path)
    if (
        not raw_path
        or "\\" in raw_path
        or path.is_absolute()
        or "." in path.parts
        or (not allow_parent and ".." in path.parts)
        or path.as_posix() != raw_path
        or (expected_path is not None and raw_path != expected_path)
    ):
        raise ValueError(f"{label} path is invalid")
    content = _content_record(
        {"sha256": record.get("sha256"), "byte_count": record.get("byte_count")},
        label=label,
    )
    return {"path": raw_path, **content}


def _integer_sequence(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"{label} must be an integer list")
    if value != sorted(set(value)):
        raise ValueError(f"{label} must be sorted and unique")
    return list(value)


def _string_sequence(
    value: object, *, label: str, allow_empty: bool
) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must be a string list")
    if not allow_empty and not value:
        raise ValueError(f"{label} must be non-empty")
    return list(value)


def _checkpoint_entries(
    value: object,
    *,
    label: str,
    exact_fields: frozenset[str] | None = None,
) -> dict[int, Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    entries: dict[int, Mapping[str, Any]] = {}
    for raw in value:
        entry = _mapping(raw, label=f"{label} entry")
        if exact_fields is not None and set(entry) != exact_fields:
            raise ValueError(f"{label} entry fields are invalid")
        frame = entry.get("frame_index")
        if type(frame) is not int or frame < 0 or frame in entries:
            raise ValueError(f"{label} frame identities are invalid")
        entries[frame] = entry
    if list(entries) != sorted(entries):
        raise ValueError(f"{label} frames must be sorted")
    return entries


def _validate_checkpoint_identity(
    entry: Mapping[str, Any], *, frame: int, scene: str, label: str
) -> tuple[int, int, int]:
    timestamp = entry.get("timestamp_ns")
    consumed = entry.get("consumed_through_frame")
    consumed_exclusive = entry.get("consumed_through_frame_exclusive")
    if not (
        type(timestamp) is int
        and timestamp >= 0
        and consumed == frame
        and type(consumed) is int
        and consumed_exclusive == frame + 1
        and type(consumed_exclusive) is int
        and ("scene" not in entry or entry.get("scene") == scene)
    ):
        raise ValueError(f"{label} checkpoint identity mismatch")
    return timestamp, consumed, consumed_exclusive


def _same_content_record(first: object, second: object, *, label: str) -> None:
    first_record = _mapping(first, label=f"{label} first record")
    second_record = _mapping(second, label=f"{label} second record")
    first_content = {
        "sha256": first_record.get("sha256"),
        "byte_count": first_record.get("byte_count"),
    }
    second_content = {
        "sha256": second_record.get("sha256"),
        "byte_count": second_record.get("byte_count"),
    }
    if not _json_exact_equal(first_content, second_content):
        raise ValueError(f"{label} content records differ")


def _validate_entity_lifecycles(
    value: object, *, official_frames: list[int]
) -> None:
    if not isinstance(value, list):
        raise ValueError("temporal entity_lifecycles must be a list")
    previous_entity_id: str | None = None
    official_frame_set = set(official_frames)
    for raw_lifecycle in value:
        lifecycle = _mapping(raw_lifecycle, label="temporal entity lifecycle")
        if set(lifecycle) != _ENTITY_LIFECYCLE_FIELDS:
            raise ValueError("temporal entity lifecycle fields are invalid")
        strings = [
            lifecycle.get("entity_id"),
            lifecycle.get("semantic_label"),
            lifecycle.get("entity_type"),
        ]
        if any(not isinstance(item, str) or not item for item in strings):
            raise ValueError("temporal entity lifecycle identity is invalid")
        entity_id = str(lifecycle["entity_id"])
        if previous_entity_id is not None and entity_id <= previous_entity_id:
            raise ValueError("temporal entity lifecycles must be sorted and unique")
        previous_entity_id = entity_id
        intervals = lifecycle.get("presence_intervals")
        if not isinstance(intervals, list) or not intervals:
            raise ValueError("temporal presence intervals must be non-empty")
        previous_last_frame = -1
        for raw_interval in intervals:
            interval = _mapping(raw_interval, label="temporal presence interval")
            if set(interval) != _PRESENCE_INTERVAL_FIELDS:
                raise ValueError("temporal presence interval fields are invalid")
            first_frame = interval.get("first_frame_index")
            first_timestamp = interval.get("first_timestamp_ns")
            last_frame = interval.get("last_frame_index")
            last_timestamp = interval.get("last_timestamp_ns")
            if not (
                type(first_frame) is int
                and type(first_timestamp) is int
                and type(last_frame) is int
                and type(last_timestamp) is int
                and first_frame in official_frame_set
                and last_frame in official_frame_set
                and first_frame <= last_frame
                and first_timestamp >= 0
                and first_timestamp <= last_timestamp
                and first_frame > previous_last_frame
            ):
                raise ValueError("temporal presence interval identity is invalid")
            previous_last_frame = last_frame


def _validate_formal_artifact_contracts(
    payloads: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Mapping[str, object]],
    *,
    scene: str,
    expected_identity: Mapping[str, Any],
    algorithm: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    run_manifest = payloads["run_manifest"]
    if set(run_manifest) != _RUN_MANIFEST_FIELDS or not (
        type(run_manifest.get("schema_version")) is int
        and run_manifest.get("schema_version") == 1
        and run_manifest.get("dataset") == "TESSE-CD"
        and run_manifest.get("method_id") == "OVIV2"
        and run_manifest.get("mode") == "causal_checkpoints"
        and run_manifest.get("scene") == scene
        and run_manifest.get("missing_observation_policy") == "signed_depth"
        and run_manifest.get("algorithm_hash") == expected_identity["algorithm_hash"]
        and run_manifest.get("stage3_lineage_commit") == STAGE3_LINEAGE_COMMIT
        and _json_exact_equal(
            run_manifest.get("normalized_algorithm_config"),
            algorithm.get("normalized_config"),
        )
    ):
        raise ValueError("run_manifest contract mismatch")
    if set(
        _mapping(
            run_manifest.get("maintenance_parameters"),
            label="run maintenance parameters",
        )
    ) != {
        "visibility_depth_tolerance_m",
        "absence_negative_support",
        "ownership_min_net_support",
    }:
        raise ValueError("run_manifest maintenance contract mismatch")
    if not _mapping(
        run_manifest.get("source_bindings"), label="run source bindings"
    ):
        raise ValueError("run_manifest source bindings are empty")
    if not _json_exact_equal(
        _content_record(run_manifest.get("config"), label="run config"),
        expected_identity["config"],
    ):
        raise ValueError("run_manifest config binding mismatch")
    _content_record(run_manifest.get("schedule"), label="run schedule")
    occlusion_declaration = _relative_content_record(
        run_manifest.get("occlusion_checkpoint_index"),
        label="run occlusion index",
        expected_path="occlusion_checkpoint_index.json",
    )
    _same_content_record(
        occlusion_declaration,
        records["occlusion_checkpoint_index"],
        label="run occlusion index",
    )
    official_frames = _integer_sequence(
        run_manifest.get("official_schedule_frame_indices"),
        label="official schedule frames",
    )
    evaluation_frames = _integer_sequence(
        run_manifest.get("evaluation_checkpoint_frames"),
        label="evaluation checkpoint frames",
    )
    scheduled_frames = _integer_sequence(
        run_manifest.get("scheduled_frame_indices"),
        label="scheduled checkpoint frames",
    )
    captured_frames = _integer_sequence(
        run_manifest.get("captured_frame_indices"),
        label="captured checkpoint frames",
    )
    run_checkpoints = _checkpoint_entries(
        run_manifest.get("checkpoints"), label="run checkpoints"
    )
    if not (
        captured_frames == scheduled_frames == list(run_checkpoints)
        and set(scheduled_frames) == set(official_frames) | set(evaluation_frames)
        and type(run_manifest.get("processed_frame_count")) is int
        and int(run_manifest["processed_frame_count"]) > max(scheduled_frames)
    ):
        raise ValueError("run_manifest checkpoint coverage mismatch")
    for frame, entry in run_checkpoints.items():
        timestamp, _, _ = _validate_checkpoint_identity(
            entry, frame=frame, scene=scene, label="run"
        )
        if not (
            type(entry.get("relative_timestamp_ns")) is int
            and int(entry["relative_timestamp_ns"]) >= 0
        ):
            raise ValueError("run checkpoint relative timestamp is invalid")
        _string_sequence(
            entry.get("event_ids"),
            label="run checkpoint event_ids",
            allow_empty=True,
        )
        _string_sequence(
            entry.get("roles"),
            label="run checkpoint roles",
            allow_empty=False,
        )
        checkpoint_root = f"checkpoints/{frame:08d}-{timestamp}"
        _relative_content_record(
            entry.get("checkpoint_status"),
            label="run checkpoint status",
            expected_path=f"{checkpoint_root}/checkpoint_status.json",
        )
        if frame in official_frames:
            if set(entry) != _FULL_RUN_CHECKPOINT_FIELDS:
                raise ValueError("full run checkpoint fields are invalid")
            for role in ("voxel_snapshot", "artifact"):
                _relative_content_record(
                    entry.get(role),
                    label=f"full run checkpoint {role}",
                    expected_path=f"{checkpoint_root}/{role}",
                )
            for role, suffix in (
                ("neutral_snapshot", ".npz"),
                ("neutral_entities", ".jsonl"),
            ):
                record = _relative_content_record(
                    entry.get(role), label=f"full run checkpoint {role}"
                )
                path = Path(str(record["path"]))
                if not (
                    path.is_relative_to(Path(checkpoint_root) / "artifact")
                    and path.suffix == suffix
                ):
                    raise ValueError(f"full run checkpoint {role} path is invalid")
        else:
            if not (
                set(entry) == _COMPACT_RUN_CHECKPOINT_FIELDS
                and entry.get("format")
                == "oviv2_compact_ownership_checkpoint"
            ):
                raise ValueError("compact run checkpoint fields are invalid")
            _relative_content_record(
                entry.get("ownership_checkpoint"),
                label="compact ownership checkpoint",
                expected_path=f"{checkpoint_root}/ownership_checkpoint",
            )

    source_index = payloads["source_index"]
    if set(source_index) != _SOURCE_INDEX_FIELDS or not (
        type(source_index.get("schema_version")) is int
        and source_index.get("schema_version") == 1
        and source_index.get("dataset") == "TESSE-CD"
        and source_index.get("mode") == "causal_checkpoint_exports"
        and source_index.get("method") == "OVIV2"
        and source_index.get("scene") == scene
    ):
        raise ValueError("source_index contract mismatch")
    for role in ("schedule", "capture_status", "trajectories"):
        _relative_content_record(
            source_index.get(role), label=f"source_index {role}"
        )
    source_checkpoints = _checkpoint_entries(
        source_index.get("checkpoints"),
        label="source checkpoints",
        exact_fields=_SOURCE_CHECKPOINT_FIELDS,
    )
    if list(source_checkpoints) != official_frames:
        raise ValueError("source_index checkpoint coverage mismatch")
    for frame, entry in source_checkpoints.items():
        source_identity = _validate_checkpoint_identity(
            entry, frame=frame, scene=scene, label="source_index"
        )
        run_entry = run_checkpoints[frame]
        if _validate_checkpoint_identity(
            run_entry, frame=frame, scene=scene, label="run"
        ) != source_identity:
            raise ValueError("run-to-source checkpoint identities differ")
        for role in ("checkpoint_status", "snapshot", "entities"):
            _relative_content_record(
                entry.get(role), label=f"source checkpoint {role}"
            )
        for source_role, run_role in (
            ("checkpoint_status", "checkpoint_status"),
            ("snapshot", "neutral_snapshot"),
            ("entities", "neutral_entities"),
        ):
            _same_content_record(
                entry.get(source_role),
                run_entry.get(run_role),
                label=f"run-to-source {source_role}",
            )

    occlusion = payloads["occlusion_checkpoint_index"]
    if set(occlusion) != _OCCLUSION_INDEX_FIELDS or not (
        type(occlusion.get("schema_version")) is int
        and occlusion.get("schema_version") == 2
        and occlusion.get("manifest_id")
        == "oviv2_tesse_cd_occlusion_checkpoints_v1"
        and occlusion.get("dataset") == "TESSE-CD"
        and occlusion.get("method_id") == "OVIV2"
        and occlusion.get("scene") == scene
        and occlusion.get("algorithm_hash") == expected_identity["algorithm_hash"]
        and isinstance(occlusion.get("evaluation_checkpoint_frames_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(occlusion["evaluation_checkpoint_frames_sha256"]),
        )
    ):
        raise ValueError("occlusion_checkpoint_index contract mismatch")
    run_config = _relative_content_record(
        occlusion.get("run_config"),
        label="occlusion run config",
        expected_path="normalized_run_config.json",
    )
    if not _json_exact_equal(
        {"sha256": run_config["sha256"], "byte_count": run_config["byte_count"]},
        expected_identity["config"],
    ):
        raise ValueError("occlusion run config binding mismatch")
    _content_record(occlusion.get("target_manifest"), label="occlusion targets")
    occlusion_snapshots = _checkpoint_entries(
        occlusion.get("snapshots"),
        label="occlusion snapshots",
        exact_fields=_OCCLUSION_SNAPSHOT_FIELDS,
    )
    if list(occlusion_snapshots) != evaluation_frames:
        raise ValueError("occlusion checkpoint coverage mismatch")
    for frame, entry in occlusion_snapshots.items():
        _validate_checkpoint_identity(
            entry, frame=frame, scene=scene, label="occlusion"
        )
        if not (
            isinstance(entry.get("format"), str)
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("checksums_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(entry["checksums_sha256"]))
        ):
            raise ValueError("occlusion snapshot contract mismatch")

    temporal_index = payloads["temporal_source_index"]
    if set(temporal_index) != _SOURCE_INDEX_FIELDS or not (
        type(temporal_index.get("schema_version")) is int
        and temporal_index.get("schema_version") == 1
        and temporal_index.get("dataset") == "TESSE-CD"
        and temporal_index.get("mode") == "causal_checkpoint_exports"
        and temporal_index.get("method") == "OVIV2"
        and temporal_index.get("scene") == scene
    ):
        raise ValueError("temporal source_index contract mismatch")
    for role in ("schedule", "capture_status", "trajectories"):
        _relative_content_record(
            temporal_index.get(role),
            label=f"temporal source_index {role}",
            allow_parent=True,
        )
    temporal_index_checkpoints = _checkpoint_entries(
        temporal_index.get("checkpoints"),
        label="temporal source checkpoints",
        exact_fields=_SOURCE_CHECKPOINT_FIELDS,
    )
    if list(temporal_index_checkpoints) != official_frames:
        raise ValueError("temporal source_index checkpoint coverage mismatch")

    temporal = payloads["temporal_manifest"]
    if set(temporal) != _TEMPORAL_MANIFEST_FIELDS or not (
        type(temporal.get("schema_version")) is int
        and temporal.get("schema_version") == 1
        and temporal.get("dataset") == "TESSE-CD"
        and temporal.get("mode") == "causal_checkpoints"
        and temporal.get("method") == "OVIV2"
        and temporal.get("scene") == scene
        and isinstance(temporal.get("entity_lifecycles"), list)
    ):
        raise ValueError("temporal_manifest contract mismatch")
    _validate_entity_lifecycles(
        temporal.get("entity_lifecycles"), official_frames=official_frames
    )
    temporal_sources = _mapping(temporal.get("sources"), label="temporal sources")
    if set(temporal_sources) != {
        "source_index",
        "schedule",
        "capture_status",
        "trajectories",
        "checkpoint_statuses",
    }:
        raise ValueError("temporal source contract mismatch")
    source_index_declaration = _relative_content_record(
        temporal_sources.get("source_index"),
        label="temporal source_index",
        expected_path="sidecars/source_index.json",
    )
    _same_content_record(
        source_index_declaration,
        records["temporal_source_index"],
        label="temporal source_index",
    )
    for role in ("schedule", "capture_status", "trajectories"):
        _relative_content_record(
            temporal_sources.get(role), label=f"temporal {role}"
        )
    statuses = temporal_sources.get("checkpoint_statuses")
    if not isinstance(statuses, list) or len(statuses) != len(official_frames):
        raise ValueError("temporal checkpoint status coverage mismatch")
    for status in statuses:
        _relative_content_record(status, label="temporal checkpoint status")
    _relative_content_record(temporal.get("trajectories"), label="trajectories")
    temporal_checkpoints = _checkpoint_entries(
        temporal.get("checkpoints"),
        label="temporal checkpoints",
        exact_fields=_TEMPORAL_CHECKPOINT_FIELDS,
    )
    if list(temporal_checkpoints) != official_frames:
        raise ValueError("temporal checkpoint coverage mismatch")
    summary_sources = _mapping(summary.get("sources"), label="summary sources")
    summary_frames_by_role: dict[str, set[int]] = {
        "snapshot": set(),
        "entities": set(),
    }
    for source_role in summary_sources:
        match = re.fullmatch(r"(snapshot|entities)\.(\d{6})", source_role)
        if match is not None:
            summary_frames_by_role[match.group(1)].add(int(match.group(2)))
    summary_metrics = _mapping(summary.get("metrics"), label="summary metrics")
    summary_events = _mapping(
        summary_metrics.get("events"), label="summary metric events"
    )
    event_frames: set[int] = set()
    for event_id, raw_event in summary_events.items():
        event = _mapping(raw_event, label=f"summary event {event_id}")
        event_frames.update(
            _integer_sequence(
                event.get("frame_ids"),
                label=f"summary event {event_id} frames",
            )
        )
    if not event_frames or not (
        summary_frames_by_role["snapshot"]
        == summary_frames_by_role["entities"]
        == event_frames
        <= set(temporal_checkpoints)
    ):
        raise ValueError("summary checkpoint coverage mismatch")
    for frame, entry in temporal_checkpoints.items():
        timestamp, consumed, consumed_exclusive = _validate_checkpoint_identity(
            entry, frame=frame, scene=scene, label="temporal"
        )
        sidecar_entry = temporal_index_checkpoints[frame]
        if (
            _validate_checkpoint_identity(
                sidecar_entry,
                frame=frame,
                scene=scene,
                label="temporal source_index",
            )
            != (timestamp, consumed, consumed_exclusive)
        ):
            raise ValueError("temporal checkpoint identities differ")
        root_source_entry = source_checkpoints[frame]
        if (
            _validate_checkpoint_identity(
                root_source_entry,
                frame=frame,
                scene=scene,
                label="root source_index",
            )
            != (timestamp, consumed, consumed_exclusive)
        ):
            raise ValueError("root-to-temporal checkpoint identities differ")
        for role, suffix in (("snapshot", "snapshot.npz"), ("entities", "entities.jsonl")):
            temporal_record = _relative_content_record(
                entry.get(role),
                label=f"temporal checkpoint {role}",
                expected_path=f"checkpoints/{frame:08d}/{suffix}",
            )
            sidecar_record = _relative_content_record(
                sidecar_entry.get(role),
                label=f"temporal source checkpoint {role}",
                expected_path=f"../checkpoints/{frame:08d}/{suffix}",
                allow_parent=True,
            )
            summary_record = summary_sources.get(f"{role}.{frame:06d}")
            _same_content_record(
                temporal_record, sidecar_record, label=f"temporal {role}"
            )
            if role == "snapshot":
                _same_content_record(
                    root_source_entry.get("snapshot"),
                    temporal_record,
                    label="root-to-temporal snapshot",
                )
            if frame in event_frames:
                _same_content_record(
                    temporal_record, summary_record, label=f"summary {role}"
                )


def _validate_captured_summary(
    payload: Mapping[str, Any], *, scene: str
) -> None:
    if not (
        payload.get("schema_version") == 1
        and payload.get("manifest_id") == "tesse_cd_common_v2_scene_summary"
        and payload.get("dataset") == "TESSE-CD"
        and payload.get("protocol") == "tesse_cd_common_v2"
        and payload.get("status") == "PASS"
        and payload.get("method") == "OVIV2"
        and payload.get("mode") == pinned_common.METHODS["OVIV2"]["summary_mode"]
        and payload.get("scene") == scene
    ):
        raise ValueError(f"{scene} scene summary identity mismatch")
    metrics = _mapping(payload.get("metrics"), label=f"{scene} metrics")
    for name in pinned_common.METRICS:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{scene}.{name} must be numeric")
        number = float(value)
        upper = 450.0 if name == "recovery_frames" else 1.0
        if not math.isfinite(number) or not 0.0 <= number <= upper:
            raise ValueError(f"{scene}.{name} is outside its valid range")
    pinned_common._validate_metric_details(metrics, scene=scene)
    sources = _mapping(payload.get("sources"), label=f"{scene} sources")
    if not pinned_common.REQUIRED_SUMMARY_SOURCES <= set(sources):
        raise ValueError(f"{scene} summary source coverage is incomplete")


def _target_declared_record(
    value: object,
    *,
    label: str,
    cache: dict[Path, dict[str, object]],
    base: Path | None = None,
) -> dict[str, object]:
    declaration = _mapping(value, label=f"{label} declaration")
    raw = Path(str(declaration.get("path", "")))
    path = _absolute_lexical(
        raw if raw.is_absolute() else (base / raw if base is not None else raw)
    )
    record = cache.get(path)
    if record is None:
        digest, byte_count, _ = _stable_regular_file(
            path, label=label, capture=False
        )
        record = {
            "path": os.fspath(path),
            "sha256": digest,
            "byte_count": byte_count,
        }
        cache[path] = record
    if (
        not _json_exact_equal(declaration.get("sha256"), record["sha256"])
        or not _json_exact_equal(
            declaration.get("byte_count"), record["byte_count"]
        )
    ):
        raise ValueError(f"{label} hash mismatch")
    return dict(record)


def _validate_captured_target_package(
    content: bytes,
    *,
    manifest_path: Path,
    arrays_path: Path,
    arrays_record: Mapping[str, object],
    schedule_path: Path,
    schedule_record: Mapping[str, object],
) -> dict[str, Any]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("common-v2 target package is not UTF-8") from error
    payload = loads_strict(text, label="common-v2 target package")
    target = _mapping(payload, label="target package")
    metadata = _mapping(target.get("metadata"), label="target metadata")
    if not (
        target.get("schema_version") == 1
        and target.get("manifest_id") == "tesse_cd_common_v2_targets"
        and target.get("dataset") == "TESSE-CD"
        and target.get("status") == "GENERATED"
        and target.get("targets_generated") is True
        and target.get("prediction_inputs_used") is False
        and metadata.get("prediction_inputs_used") is False
        and metadata.get("protocol_complete") is True
        and metadata.get("window_frames") == 450
        and metadata.get("voxel_size_m") == 0.05
        and set(metadata.get("scenes", ())) == set(SCENES)
    ):
        raise ValueError("target package is not complete and prediction-independent")

    cache = {
        arrays_path: {
            "path": os.fspath(arrays_path),
            "sha256": arrays_record["sha256"],
            "byte_count": arrays_record["byte_count"],
        },
        schedule_path: {
            "path": os.fspath(schedule_path),
            "sha256": schedule_record["sha256"],
            "byte_count": schedule_record["byte_count"],
        },
    }
    observed_arrays = _target_declared_record(
        target.get("target_arrays"),
        label="target arrays",
        cache=cache,
        base=manifest_path.parent,
    )
    if observed_arrays != cache[arrays_path]:
        raise ValueError("scene summaries do not bind the target arrays")

    target_sources = target.get("sources")
    if not isinstance(target_sources, list) or not target_sources:
        raise ValueError("target package sources must be non-empty")
    validated_sources = [
        _target_declared_record(
            declaration,
            label="target source",
            cache=cache,
        )
        for declaration in target_sources
    ]
    _target_declared_record(
        metadata.get("source_manifest"),
        label="target source manifest",
        cache=cache,
    )
    declared_schedule = _target_declared_record(
        metadata.get("schedule"),
        label="target schedule",
        cache=cache,
    )
    if declared_schedule != cache[schedule_path]:
        raise ValueError("target package schedule binding mismatch")
    declared_sources = _mapping(
        metadata.get("declared_source_records"),
        label="target declared source records",
    )
    if not declared_sources:
        raise ValueError("target declared source records must be non-empty")
    validated_declared_sources = {
        str(label): _target_declared_record(
            declaration,
            label="target source",
            cache=cache,
        )
        for label, declaration in declared_sources.items()
    }

    observability = _mapping(
        metadata.get("background_observable_by_event"),
        label="target observability",
    )
    if not observability or any(
        type(value) is not bool for value in observability.values()
    ):
        raise ValueError(
            "target observability must contain prediction-independent booleans"
        )
    arrays = _mapping(
        _mapping(target.get("target_arrays"), label="target arrays").get("arrays"),
        label="target array declarations",
    )
    revealed = {
        str(name).removesuffix(".revealed_background"): _mapping(
            declaration, label=f"target array {name}"
        )
        for name, declaration in arrays.items()
        if str(name).endswith(".revealed_background")
    }
    if set(revealed) != set(observability) or any(
        bool(
            pinned_common._nonnegative_int(
                declaration.get("element_count"),
                label=f"{event_id} target elements",
            )
        )
        is not observability[event_id]
        for event_id, declaration in revealed.items()
    ):
        raise ValueError("target observability disagrees with target arrays")
    observable_count = sum(observability.values())
    if (
        metadata.get("background_observable_event_count") != observable_count
        or metadata.get("unobservable_revealed_target_event_count")
        != len(observability) - observable_count
        or any(
            not any(
                value
                for event_id, value in observability.items()
                if str(event_id).startswith(f"{scene}_event_")
            )
            for scene in SCENES
        )
    ):
        raise ValueError("target observability counts are inconsistent")
    return {
        "target_arrays": cache[arrays_path],
        "prediction_inputs_used": False,
        "background_observable_by_event": dict(observability),
        "sources": validated_sources,
        "declared_source_records": validated_declared_sources,
    }


def finalize_oviv2_common_v2_release(
    freeze_manifest: Path,
    *,
    run_id: str,
    output: Path,
) -> Path:
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    freeze, freeze_record = _read_json(freeze_manifest, label="freeze manifest")
    if not (
        freeze.get("schema_version") == 1
        and freeze.get("freeze_id") == "oviv2-tessecd-v1"
        and freeze.get("status") == "FROZEN"
        and freeze.get("method") == "OVIV2"
        and freeze.get("dataset") == "TESSE-CD"
    ):
        raise ValueError("freeze manifest identity mismatch")
    repository = _validate_repository(
        _mapping(freeze.get("repository"), label="freeze repository")
    )
    if repository["stage3_lineage_commit"] != STAGE3_LINEAGE_COMMIT:
        raise ValueError("freeze manifest Stage3 lineage mismatch")
    algorithm = _mapping(freeze.get("algorithm"), label="freeze algorithm")
    if not re.fullmatch(r"[0-9a-f]{64}", str(algorithm.get("sha256", ""))):
        raise ValueError("freeze algorithm hash is invalid")

    shared = _mapping(freeze.get("shared_bindings"), label="shared bindings")
    if not _REQUIRED_SHARED_BINDINGS <= set(shared):
        raise ValueError("freeze shared bindings are incomplete")
    target_manifest_path, target_manifest_record, target_manifest_content = (
        _verified_binding_data(
            shared["common_target_manifest"],
            label="common_target_manifest",
            capture=True,
        )
    )
    assert target_manifest_content is not None
    target_arrays_path, target_arrays_record = _verified_binding(
        shared["common_target_arrays"], label="common_target_arrays"
    )
    schedule_path, schedule_record = _verified_binding(
        shared["schedule"], label="schedule"
    )
    aliases_path, aliases_record = _verified_binding(
        shared["alias_map"], label="alias_map"
    )
    evaluator_path, evaluator_record = _verified_binding(
        shared["evaluator"],
        label="evaluator",
        expected_path=ROOT / "scripts/evaluation/evaluate_tesse_cd_common_v2.py",
    )
    finalizers = _mapping(shared["finalizers"], label="frozen finalizers")
    if set(finalizers) != {"common_v2", "official_t2"}:
        raise ValueError("frozen finalizer bindings are incomplete")
    _, common_finalizer_record = _verified_binding(
        finalizers["common_v2"],
        label="common_v2_finalizer",
        expected_path=ROOT / "scripts/evaluation/finalize_tesse_common_v2.py",
    )
    _, official_finalizer_record = _verified_binding(
        finalizers["official_t2"],
        label="official_t2_finalizer",
        expected_path=ROOT / "scripts/evaluation/finalize_tesse_t2.py",
    )

    release_bindings = _mapping(
        freeze.get("release_bindings"), label="release bindings"
    )
    if set(release_bindings) != {
        "canonical_summary_generator",
        "release_finalizer",
        "label_spaces",
    }:
        raise ValueError("freeze release bindings are incomplete")
    _, canonicalizer_record = _verified_binding(
        release_bindings["canonical_summary_generator"],
        label="canonical_summary_generator",
        expected_path=ROOT
        / "scripts/evaluation/canonicalize_tesse_common_v2_summary.py",
    )
    _, release_finalizer_record = _verified_binding(
        release_bindings["release_finalizer"],
        label="release_finalizer",
        expected_path=Path(__file__),
    )
    label_bindings = _mapping(
        release_bindings["label_spaces"], label="label space bindings"
    )
    if set(label_bindings) != set(SCENES):
        raise ValueError("label space bindings must cover Apartment and Office")
    label_spaces: dict[str, Path] = {}
    label_records: dict[str, dict[str, object]] = {}
    for scene in SCENES:
        label_spaces[scene], label_records[scene] = _verified_binding(
            label_bindings[scene], label=f"{scene}_label_space"
        )

    scene_bindings = _mapping(freeze.get("scenes"), label="freeze scenes")
    if set(scene_bindings) != set(SCENES):
        raise ValueError("freeze scenes must cover Apartment and Office")
    frozen_config_records: dict[str, dict[str, object]] = {}
    expected_run_identities: dict[str, dict[str, Any]] = {}
    for scene in SCENES:
        selected = _mapping(
            scene_bindings[scene], label=f"{scene} freeze scene binding"
        )
        _, config_record, config_content = _verified_binding_data(
            selected.get("frozen_config"),
            label=f"{scene}_frozen_config",
            capture=True,
        )
        assert config_content is not None
        try:
            config_text = config_content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{scene} frozen config is not UTF-8") from error
        config = _mapping(
            loads_strict(config_text, label=f"{scene} frozen config"),
            label=f"{scene} frozen config",
        )
        if not (
            config.get("missing_observation_policy") == "signed_depth"
            and config.get("algorithm_hash") == algorithm["sha256"]
            and _algorithm_hash(config) == algorithm["sha256"]
            and _json_exact_equal(
                _algorithm_config(config), algorithm.get("normalized_config")
            )
        ):
            raise ValueError(f"{scene} frozen config differs from frozen algorithm")
        frozen_config_records[scene] = config_record
        input_bindings = {
            "shared_bindings": dict(shared),
            "scene": dict(selected),
        }
        expected_run_identities[scene] = {
            "schema_version": 1,
            "freeze_id": "oviv2-tessecd-v1",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": scene,
            "freeze_manifest": {
                "sha256": freeze_record["sha256"],
                "byte_count": freeze_record["byte_count"],
            },
            "repository": {
                "commit": repository["commit"],
                "tree": repository["tree"],
            },
            "config": {
                "sha256": config_record["sha256"],
                "byte_count": config_record["byte_count"],
            },
            "algorithm_hash": algorithm["sha256"],
            "missing_observation_policy": "signed_depth",
            "input_bindings_sha256": _compact_json_hash(input_bindings),
        }

    raw_roots = _mapping(freeze.get("output_roots"), label="output roots")
    expected_root_roles = {
        f"{scene}_run{repeat}" for scene in SCENES for repeat in REPEATS
    }
    if set(raw_roots) != expected_root_roles:
        raise ValueError("freeze output roots do not cover exact independent runs")
    roots: dict[tuple[str, int], Path] = {}
    root_identities: dict[tuple[str, int], FileIdentity] = {}
    for scene in SCENES:
        for repeat in REPEATS:
            role = f"{scene}_run{repeat}"
            raw = raw_roots[role]
            if not isinstance(raw, str) or not Path(raw).is_absolute():
                raise ValueError(f"{role} root must be absolute")
            if raw != os.fspath(_absolute_lexical(Path(raw))):
                raise ValueError(f"{role} root must be canonical")
            roots[(scene, repeat)], root_identities[(scene, repeat)] = _direct_root(
                Path(raw), label=f"{role} root"
            )
    if len(set(roots.values())) != len(roots) or len(
        set(root_identities.values())
    ) != len(root_identities):
        raise ValueError("release runs must use physically independent artifact roots")

    external_common = {
        "target_manifest": target_manifest_path,
        "target_arrays": target_arrays_path,
        "aliases": aliases_path,
        "evaluator": evaluator_path,
    }
    raw_summaries: dict[tuple[str, int], dict[str, Any]] = {}
    canonical: dict[tuple[str, int], dict[str, Any]] = {}
    raw_records: dict[tuple[str, int], dict[str, object]] = {}
    captured_identities: dict[
        tuple[str, int], dict[str, FileIdentity]
    ] = {}
    formal_temporal_artifacts: dict[
        tuple[str, int], dict[str, CapturedArtifact]
    ] = {}
    for scene in SCENES:
        for repeat in REPEATS:
            key = (scene, repeat)
            summary_path = roots[key] / "evaluation/summary.json"
            (
                raw_payload,
                canonical_payload,
                raw_content_record,
                identities,
                temporal_artifacts,
            ) = capture_and_canonicalize_summary(
                summary_path,
                artifact_root=roots[key],
                external_sources={
                    **external_common,
                    "label_space": label_spaces[scene],
                },
                expected_artifact_root_identity=root_identities[key],
            )
            _validate_captured_summary(raw_payload, scene=scene)
            raw_summaries[key] = raw_payload
            canonical[key] = canonical_payload
            raw_records[key] = {
                "role": f"{scene}.run{repeat}.raw_summary",
                **raw_content_record,
            }
            captured_identities[key] = identities
            formal_temporal_artifacts[key] = temporal_artifacts

    formal_run_records: dict[
        tuple[str, int], dict[str, dict[str, object]]
    ] = {}
    run_executions: dict[tuple[str, int], dict[str, Any]] = {}
    root_artifact_paths = {
        "run_manifest": Path("run_manifest.json"),
        "source_index": Path("source_index.json"),
        "occlusion_checkpoint_index": Path("occlusion_checkpoint_index.json"),
    }
    for scene in SCENES:
        for repeat in REPEATS:
            key = (scene, repeat)
            temporal_artifacts = formal_temporal_artifacts[key]
            if set(temporal_artifacts) != {
                "temporal_manifest",
                "temporal_source_index",
            }:
                raise ValueError(
                    f"{scene}.run{repeat} formal temporal identity is missing"
                )
            payloads: dict[str, dict[str, Any]] = {}
            records: dict[str, dict[str, object]] = {}
            for role, relative in root_artifact_paths.items():
                payload, record, identity = _captured_json_artifact(
                    roots[key] / relative,
                    label=f"{scene}.run{repeat} {role}",
                    artifact_root=(roots[key], root_identities[key]),
                )
                payloads[role] = payload
                records[role] = {
                    "role": f"{scene}.run{repeat}.{role}",
                    **record,
                }
                captured_identities[key][f"formal_{role}"] = identity
            for role, (payload, record) in temporal_artifacts.items():
                payloads[role] = payload
                records[role] = {
                    "role": f"{scene}.run{repeat}.{role}",
                    **record,
                }
            expected_identity = expected_run_identities[scene]
            if any(
                not _json_exact_equal(
                    payload.get("frozen_run_identity"), expected_identity
                )
                for payload in payloads.values()
            ):
                raise ValueError(f"{scene}.run{repeat} frozen run identity mismatch")
            execution = _validate_run_execution(
                payloads["run_manifest"].get("run_execution"),
                scene=scene,
                repeat=repeat,
                root=roots[key],
                root_identity=root_identities[key],
            )
            if any(
                not _json_exact_equal(payload.get("run_execution"), execution)
                for payload in payloads.values()
            ):
                raise ValueError(f"{scene}.run{repeat} execution identity mismatch")
            _validate_formal_artifact_contracts(
                payloads,
                records,
                scene=scene,
                expected_identity=expected_identity,
                algorithm=algorithm,
                summary=raw_summaries[key],
            )
            formal_run_records[key] = records
            run_executions[key] = execution

    if len(
        {execution["execution_id"] for execution in run_executions.values()}
    ) != len(run_executions):
        raise ValueError("release runs must have distinct execution identities")

    observed_run_local_identities: dict[FileIdentity, tuple[str, int, str]] = {}
    for (scene, repeat), identities in captured_identities.items():
        for role, identity in identities.items():
            if role in EXTERNAL_SOURCE_ROLES:
                continue
            previous = observed_run_local_identities.setdefault(
                identity, (scene, repeat, role)
            )
            if previous != (scene, repeat, role):
                raise ValueError(
                    "run-local sources must be independent files: "
                    f"{previous[0]}.run{previous[1]}.{previous[2]} and "
                    f"{scene}.run{repeat}.{role}"
                )

    for scene in SCENES:
        if canonical_summary_bytes(canonical[(scene, 1)]) != canonical_summary_bytes(
            canonical[(scene, 2)]
        ):
            raise ValueError(
                f"{scene} canonical summaries must be byte-identical"
            )
    shared_source_roles = (
        "target_manifest",
        "target_arrays",
        "schedule",
        "aliases",
        "evaluator",
    )
    for role in shared_source_roles:
        apartment_source = canonical[("apartment", 1)]["sources"][role]
        office_source = canonical[("office", 1)]["sources"][role]
        if apartment_source != office_source:
            raise ValueError(f"Apartment and Office {role} bindings differ")
    if canonical[("apartment", 1)]["sources"]["schedule"] != {
        "role": "schedule",
        "sha256": schedule_record["sha256"],
        "byte_count": schedule_record["byte_count"],
    }:
        raise ValueError("scene summaries disagree with frozen schedule")

    target_evidence = _validate_captured_target_package(
        target_manifest_content,
        manifest_path=target_manifest_path,
        arrays_path=target_arrays_path,
        arrays_record=target_arrays_record,
        schedule_path=schedule_path,
        schedule_record=schedule_record,
    )
    observability = _mapping(
        target_evidence.get("background_observable_by_event"),
        label="target observability",
    )
    for scene in SCENES:
        pinned_common._validate_scene_observability(
            raw_summaries[(scene, 1)],
            scene=scene,
            observability=observability,
        )

    scene_metrics = {
        scene: {
            name: float(raw_summaries[(scene, 1)]["metrics"][name])
            for name in pinned_common.METRICS
        }
        for scene in SCENES
    }
    metrics = {
        name: fmean(scene_metrics[scene][name] for scene in SCENES)
        for name in pinned_common.METRICS
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("release metrics must be finite")
    token_bindings = [
        {
            "token": f"T2_OVIV2_{suffix}",
            "json_pointer": f"/metrics/{name}",
            "precision": 3,
        }
        for name, suffix in pinned_common.METRICS.items()
    ]
    method_config = pinned_common.METHODS["OVIV2"]
    result = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_common_v2_release_v1",
        "status": "VERIFIED",
        "run_id": run_id,
        "method": {
            "key": "OVIV2",
            "display_label": method_config["display_label"],
            "mode": method_config["mode"],
            "eligible_for_ranking": method_config["eligible_for_ranking"],
        },
        "dataset": {"name": "TESSE-CD", "splits": ["macro_test"]},
        "protocol": {
            "name": "tesse_cd_common_v2",
            "aggregation": "macro mean over Apartment and Office scene summaries",
            "deterministic_repeat": "canonical-role-content-byte-identical",
            "recovery": "F@5cm >= 0.9 for three checkpoints; 450-frame restricted censor",
        },
        "metrics": metrics,
        "scene_metrics": scene_metrics,
        "scene_details": {
            scene: dict(raw_summaries[(scene, 1)]["metrics"])
            for scene in SCENES
        },
        "scene_summaries": {
            scene: {
                "primary": _canonical_record(
                    canonical[(scene, 1)], role=f"{scene}.run1.canonical_summary"
                ),
                "repeat": _canonical_record(
                    canonical[(scene, 2)], role=f"{scene}.run2.canonical_summary"
                ),
                "raw_primary": raw_records[(scene, 1)],
                "raw_repeat": raw_records[(scene, 2)],
            }
            for scene in SCENES
        },
        "scene_sources": {
            scene: dict(canonical[(scene, 1)]["sources"]) for scene in SCENES
        },
        "formal_run_artifacts": {
            scene: {
                "primary": formal_run_records[(scene, 1)],
                "repeat": formal_run_records[(scene, 2)],
            }
            for scene in SCENES
        },
        "target_package": {
            "manifest": target_manifest_record,
            "target_arrays": target_arrays_record,
            "schedule": schedule_record,
            "prediction_inputs_used": False,
            "background_observable_by_event": dict(observability),
        },
        "frozen_identity": {
            **freeze_record,
            "freeze_id": "oviv2-tessecd-v1",
            "repository_commit": repository["commit"],
            "algorithm_sha256": algorithm["sha256"],
            "scene_run_identities": expected_run_identities,
        },
        "validation_tools": {
            "evaluator": evaluator_record,
            "common_v2_finalizer": common_finalizer_record,
            "official_t2_finalizer": official_finalizer_record,
            "canonical_summary_generator": canonicalizer_record,
            "release_finalizer": release_finalizer_record,
            "aliases": aliases_record,
            "label_spaces": label_records,
            "frozen_configs": frozen_config_records,
        },
        "token_bindings": token_bindings,
        "unavailable_bindings": [],
        "protocol_deviations": [],
    }
    _atomic_json_no_replace(output, result)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    finalize_oviv2_common_v2_release(
        args.freeze_manifest,
        run_id=args.run_id,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
