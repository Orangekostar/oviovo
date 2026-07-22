#!/usr/bin/env python3
"""Derive prediction-independent TESSE-CD occlusion stress targets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.derive_tesse_cd_common_v2 import (  # noqa: E402
    UNKNOWN_SEMANTIC_LABELS,
    deterministic_npz_bytes,
    dsg_interval_world_points,
    match_event_dsg_records,
    voxel_keys,
)
from src.evaluation.oviv2_occlusion import (  # noqa: E402
    classify_voxel_depths,
    sorted_voxel_array,
)


SCENES = ("apartment", "office")
SINGLE_ROLES = ("source_manifest", "schedule", "rgbd_lock", "camera")
SCENE_ROLES = (
    "changes",
    "dsg_with_mesh",
    "export_manifest",
    "timestamps",
    "trajectory",
)
INPUT_ROLE_PATTERNS = (
    *SINGLE_ROLES,
    *(f"scene.{role}" for role in SCENE_ROLES),
    "scene.depth.NNNNNN",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_SOURCE_COMPONENTS = {
    "method_output",
    "method_outputs",
    "prediction",
    "predictions",
    "snapshot",
    "snapshots",
    "frontend",
    "dense",
    "detections",
}
DEFAULT_CONTRACT = REPO_ROOT / "configs/evaluation/manifests/tesse_cd_occlusion_v1.json"


def _preregistered_parameters() -> dict[str, Any]:
    return {
        "voxel_size_m": 0.05,
        "depth_tolerance_m": 0.10,
        "active_interval": "first <= timestamp < last",
        "anchor_rule": "first prior non-empty present voxel set in lifecycle",
        "target_rule": "anchor present voxels intersect current occluded voxels",
        "episode_rule": "consecutive non-empty source frames",
        "episode_fraction_rule": "maximum checkpoint occluded/anchor voxel fraction",
        "stress_thresholds": [0.50, 0.75, 0.90],
        "headline_stress_threshold": 0.90,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("value is not canonical finite JSON") from error


@dataclass(frozen=True)
class _SourceWitness:
    declared_path: Path
    resolved_path: Path
    fingerprint: tuple[int, int, int, int, int]
    record: dict[str, Any]


def _serialized_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    status = path.stat()
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _capture_source_witness(path: Path) -> _SourceWitness:
    declared = Path(path).absolute()
    candidate = declared.resolve()
    if not candidate.is_file():
        raise ValueError(f"source is not a file: {candidate}")
    before = _fingerprint(candidate)
    digest = hashlib.sha256()
    byte_count = 0
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    after = _fingerprint(candidate)
    if before != after:
        raise ValueError(f"source changed while reading: {candidate}")
    record = {
        "path": _serialized_path(candidate),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }
    return _SourceWitness(declared, candidate, before, record)


def _read_source_bytes(path: Path) -> tuple[bytes, _SourceWitness]:
    """Read once, hashing the exact bytes returned to the parser."""
    declared = Path(path).absolute()
    candidate = declared.resolve()
    if not candidate.is_file():
        raise ValueError(f"source is not a file: {candidate}")
    before = _fingerprint(candidate)
    with candidate.open("rb") as handle:
        content = handle.read()
    after = _fingerprint(candidate)
    if before != after:
        raise ValueError(f"source changed while reading: {candidate}")
    record = {
        "path": _serialized_path(candidate),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }
    return content, _SourceWitness(declared, candidate, before, record)


def _file_record(path: Path) -> dict[str, Any]:
    return _capture_source_witness(path).record


def _revalidate_source_witness(witness: _SourceWitness) -> None:
    try:
        observed = _capture_source_witness(witness.declared_path)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"source changed before publication: {witness.declared_path}"
        ) from error
    if not (
        observed.resolved_path == witness.resolved_path
        and observed.fingerprint == witness.fingerprint
        and observed.record == witness.record
    ):
        raise ValueError(
            f"source changed before publication: {witness.declared_path}"
        )


def _revalidate_source_identity(witness: _SourceWitness) -> None:
    """Create a cheap common-time barrier after the sequential hash pass."""
    try:
        resolved = witness.declared_path.resolve()
        fingerprint = _fingerprint(resolved)
    except OSError as error:
        raise ValueError(
            f"source changed before publication: {witness.declared_path}"
        ) from error
    if resolved != witness.resolved_path or fingerprint != witness.fingerprint:
        raise ValueError(
            f"source changed before publication: {witness.declared_path}"
        )


def _expected_source_roles(
    scene_frame_indices: Mapping[str, Iterable[int]],
) -> tuple[str, ...]:
    if set(scene_frame_indices) != set(SCENES):
        raise ValueError("scene frame indices must contain apartment and office exactly")
    roles = list(SINGLE_ROLES)
    for scene in SCENES:
        indices = tuple(int(value) for value in scene_frame_indices[scene])
        if not indices or indices != tuple(range(len(indices))):
            raise ValueError(f"{scene} source frame indices must be contiguous from zero")
        roles.extend(f"{scene}.{role}" for role in SCENE_ROLES)
        roles.extend(f"{scene}.depth.{index:06d}" for index in indices)
    return tuple(sorted(roles))


def validate_source_allowlist(
    source_paths: Mapping[str, Path],
    *,
    scene_frame_indices: Mapping[str, Iterable[int]],
) -> dict[str, dict[str, Any]]:
    """Require the exact GT/depth/pose/camera input set and bind every file."""
    records, _ = _capture_source_allowlist(
        source_paths, scene_frame_indices=scene_frame_indices
    )
    return records


def _capture_source_allowlist(
    source_paths: Mapping[str, Path],
    *,
    scene_frame_indices: Mapping[str, Iterable[int]],
) -> tuple[dict[str, dict[str, Any]], dict[str, _SourceWitness]]:
    expected = _expected_source_roles(scene_frame_indices)
    observed = tuple(sorted(str(role) for role in source_paths))
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(
            f"exact source allowlist mismatch; missing={missing}, extra={extra}"
        )
    records: dict[str, dict[str, Any]] = {}
    witnesses: dict[str, _SourceWitness] = {}
    for role in expected:
        path = Path(source_paths[role])
        components = {part.lower() for part in path.parts}
        if path.stem.lower() in FORBIDDEN_SOURCE_COMPONENTS or (
            components & FORBIDDEN_SOURCE_COMPONENTS
        ):
            raise ValueError(f"prediction or method output path is forbidden: {path}")
        if ".depth." in role and not re.fullmatch(
            r"(?:apartment|office)\.depth\.\d{6}", role
        ):
            raise ValueError(f"exact source allowlist rejects role: {role}")
        witness = _capture_source_witness(path)
        records[role] = witness.record
        witnesses[role] = witness
    return records, witnesses


def depth_collection_binding(
    source_records: Mapping[str, Mapping[str, Any]],
    *,
    scene: str,
    frame_indices: Iterable[int],
) -> dict[str, Any]:
    """Bind the ordered depth-only collection without reading RGB inputs."""
    if scene not in SCENES:
        raise ValueError(f"unknown TESSE-CD scene: {scene}")
    indices = tuple(int(value) for value in frame_indices)
    if not indices or indices != tuple(range(len(indices))):
        raise ValueError("depth frame indices must be contiguous from zero")
    digest = hashlib.sha256()
    total = 0
    for index in indices:
        role = f"{scene}.depth.{index:06d}"
        record = source_records.get(role)
        if not isinstance(record, Mapping):
            raise ValueError(f"missing depth source record: {role}")
        sha256 = record.get("sha256")
        byte_count = record.get("byte_count")
        if (
            not isinstance(sha256, str)
            or SHA256_PATTERN.fullmatch(sha256) is None
            or type(byte_count) is not int
            or byte_count <= 0
        ):
            raise ValueError(f"invalid depth source record: {role}")
        digest.update(
            role.encode("ascii")
            + b"\0"
            + sha256.encode("ascii")
            + b"\0"
            + str(byte_count).encode("ascii")
            + b"\n"
        )
        total += byte_count
    return {
        "frame_count": len(indices),
        "combined_sha256": digest.hexdigest(),
        "total_byte_count": total,
        "binding_rule": "sha256(role\\0sha256\\0byte_count\\n) in frame order",
    }


def _validate_depth_binding(scene: str, binding: Mapping[str, Any]) -> None:
    required = {
        "frame_count",
        "combined_sha256",
        "total_byte_count",
        "binding_rule",
    }
    if set(binding) not in (required, required | {"root"}):
        raise ValueError(f"invalid depth collection fields: {scene}")
    if type(binding["frame_count"]) is not int or binding["frame_count"] <= 0:
        raise ValueError(f"invalid depth frame count: {scene}")
    if (
        not isinstance(binding["combined_sha256"], str)
        or SHA256_PATTERN.fullmatch(binding["combined_sha256"]) is None
    ):
        raise ValueError(f"invalid depth collection SHA256: {scene}")
    if type(binding["total_byte_count"]) is not int or binding["total_byte_count"] <= 0:
        raise ValueError(f"invalid depth collection byte count: {scene}")
    if binding["binding_rule"] != "sha256(role\\0sha256\\0byte_count\\n) in frame order":
        raise ValueError(f"invalid depth collection binding rule: {scene}")
    if "root" in binding and (
        not isinstance(binding["root"], str) or not Path(binding["root"]).is_absolute()
    ):
        raise ValueError(f"invalid depth collection root: {scene}")


def validate_source_bindings(
    source_paths: Mapping[str, Path],
    *,
    scene_frame_indices: Mapping[str, Iterable[int]],
    expected_source_records: Mapping[str, Mapping[str, Any]],
    expected_depth_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Reject any file drift against the pre-registered source contract."""
    observed, _ = _capture_source_bindings(
        source_paths,
        scene_frame_indices=scene_frame_indices,
        expected_source_records=expected_source_records,
        expected_depth_bindings=expected_depth_bindings,
    )
    return observed


def _capture_source_bindings(
    source_paths: Mapping[str, Path],
    *,
    scene_frame_indices: Mapping[str, Iterable[int]],
    expected_source_records: Mapping[str, Mapping[str, Any]],
    expected_depth_bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, _SourceWitness]]:
    normalized_indices = {
        scene: tuple(values) for scene, values in scene_frame_indices.items()
    }
    observed, witnesses = _capture_source_allowlist(
        source_paths, scene_frame_indices=normalized_indices
    )
    observed_non_depth = {
        role: record for role, record in observed.items() if ".depth." not in role
    }
    normalized_expected = {
        str(role): dict(record)
        for role, record in sorted(expected_source_records.items())
    }
    if observed_non_depth != normalized_expected:
        raise ValueError("source hash or byte count drift in frozen file bindings")
    if set(expected_depth_bindings) != set(SCENES):
        raise ValueError("source hash or byte count drift in depth bindings")
    for scene in SCENES:
        expected = dict(expected_depth_bindings[scene])
        expected.pop("root", None)
        actual = depth_collection_binding(
            observed,
            scene=scene,
            frame_indices=normalized_indices[scene],
        )
        if actual != expected:
            raise ValueError(f"source hash or byte count drift: {scene} depth collection")
    return observed, witnesses


def _validate_record(role: str, record: Mapping[str, Any]) -> None:
    if set(record) != {"path", "sha256", "byte_count"}:
        raise ValueError(f"invalid source record fields: {role}")
    if not isinstance(record["path"], str) or not record["path"]:
        raise ValueError(f"invalid source record path: {role}")
    if (
        not isinstance(record["sha256"], str)
        or SHA256_PATTERN.fullmatch(record["sha256"]) is None
    ):
        raise ValueError(f"invalid source record SHA256: {role}")
    if type(record["byte_count"]) is not int or record["byte_count"] <= 0:
        raise ValueError(f"invalid source record byte count: {role}")


def build_contract_manifest(
    *,
    source_records: Mapping[str, Mapping[str, Any]],
    depth_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Create an input-only pre-registration contract, never a result manifest."""
    normalized_records = {
        str(role): dict(record) for role, record in sorted(source_records.items())
    }
    if not normalized_records:
        raise ValueError("source records must be non-empty")
    for role, record in normalized_records.items():
        _validate_record(role, record)
    if set(depth_bindings) != set(SCENES):
        raise ValueError("depth bindings must contain apartment and office exactly")
    normalized_depth = {
        scene: dict(depth_bindings[scene]) for scene in SCENES
    }
    for scene, binding in normalized_depth.items():
        _validate_depth_binding(scene, binding)
    binding = {
        "source_records": normalized_records,
        "depth_collections": normalized_depth,
    }
    canonical = _canonical_json_bytes(binding)
    return {
        "schema_version": 1,
        "manifest_id": "tesse_cd_occlusion_v1",
        "dataset": "TESSE-CD",
        "status": "CONTRACT_ONLY",
        "targets_generated": False,
        "fixture_tested": True,
        "prediction_inputs_used": False,
        "input_roles": list(INPUT_ROLE_PATTERNS),
        "parameters": _preregistered_parameters(),
        "source_records": normalized_records,
        "depth_collections": normalized_depth,
        "input_binding_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def render_manifest(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _attributes(record: Mapping[str, Any]) -> Mapping[str, Any]:
    attributes = record.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ValueError("DSG object record has no attributes")
    return attributes


def _lifecycles(record: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    attributes = _attributes(record)
    first = attributes.get("first_observed_ns")
    last = attributes.get("last_observed_ns")
    if (
        not isinstance(first, Sequence)
        or isinstance(first, (str, bytes))
        or not isinstance(last, Sequence)
        or isinstance(last, (str, bytes))
        or len(first) != len(last)
        or not first
    ):
        raise ValueError("DSG object lifecycle intervals are invalid")
    intervals: list[tuple[int, int]] = []
    for begin, end in zip(first, last):
        if type(begin) is not int or type(end) is not int or begin < 0 or end <= begin:
            raise ValueError("DSG object lifecycle intervals are invalid")
        intervals.append((begin, end))
    return tuple(intervals)


def _object_lifecycles(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    for record_ordinal, record in enumerate(records):
        attributes = _attributes(record)
        label = attributes.get("semantic_label")
        if type(label) is not int or label in UNKNOWN_SEMANTIC_LABELS:
            continue
        name = str(attributes.get("name", "")).strip()
        if not name:
            raise ValueError("known-label DSG object has no name")
        for interval_index, (first, last) in enumerate(_lifecycles(record)):
            matched = {
                "symbol": name,
                "record_id": record.get("id"),
                "interval_index": interval_index,
                "record": record,
            }
            geometry = dsg_interval_world_points(matched)
            keys = np.asarray(voxel_keys(geometry), dtype=np.int64).reshape((-1, 3))
            if not len(keys):
                raise ValueError(f"{name} lifecycle geometry is empty")
            values.append(
                {
                    "record_ordinal": record_ordinal,
                    "object_id": record.get("id"),
                    "object_name": name,
                    "semantic_label": label,
                    "lifecycle_index": interval_index,
                    "first_timestamp_ns": first,
                    "last_timestamp_ns": last,
                    "voxels": keys,
                }
            )
    return tuple(values)


def _validated_frames(
    scene: str, frames: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    if not frames:
        raise ValueError(f"{scene} depth frame input is empty")
    ordered = tuple(frames)
    indices = [frame.get("frame_index") for frame in ordered]
    timestamps = [frame.get("relative_timestamp_ns") for frame in ordered]
    if any(type(value) is not int for value in indices + timestamps):
        raise ValueError(f"{scene} frame indices and timestamps must be integers")
    if indices != list(range(len(ordered))):
        raise ValueError(f"{scene} frames must be contiguous from zero")
    if timestamps != sorted(set(timestamps)):
        raise ValueError(f"{scene} frame timestamps must be unique and increasing")
    return ordered


def _layer_summary(
    episodes: Sequence[Mapping[str, Any]], threshold: float | None
) -> dict[str, Any]:
    selected = [
        str(episode["episode_id"])
        for episode in episodes
        if threshold is None or float(episode["occlusion_fraction"]) >= threshold
    ]
    payload: dict[str, Any] = {
        "episode_count": len(selected),
        "episode_ids": selected,
    }
    if threshold is not None:
        payload["minimum_occlusion_fraction"] = threshold
    return payload


def derive_occlusion_targets(
    *,
    frames_by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
    dsg_records_by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
    camera: Mapping[str, float | int],
    voxel_size_m: float = 0.05,
    depth_tolerance_m: float = 0.10,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Derive fixed-anchor occlusion episodes from official GT and depth only."""
    parameters = _preregistered_parameters()
    if voxel_size_m != parameters["voxel_size_m"]:
        raise ValueError("voxel_size_m must match the pre-registered value")
    if depth_tolerance_m != parameters["depth_tolerance_m"]:
        raise ValueError("depth_tolerance_m must match the pre-registered value")
    if set(frames_by_scene) != set(SCENES) or set(dsg_records_by_scene) != set(SCENES):
        raise ValueError("occlusion derivation requires apartment and office exactly")
    arrays: dict[str, np.ndarray] = {}
    episodes: list[dict[str, Any]] = []
    scene_frame_indices: dict[str, list[int]] = {}

    for scene in SCENES:
        frames = _validated_frames(scene, frames_by_scene[scene])
        scene_frame_indices[scene] = [int(frame["frame_index"]) for frame in frames]
        lifecycle_records = _object_lifecycles(dsg_records_by_scene[scene])
        scene_episode_count = 0
        for lifecycle in lifecycle_records:
            anchor: dict[str, Any] | None = None
            pending: list[dict[str, Any]] = []

            def flush() -> None:
                nonlocal pending, scene_episode_count
                if not pending or anchor is None:
                    pending = []
                    return
                episode_id = f"{scene}_occlusion_{scene_episode_count:04d}"
                anchor_array = f"{episode_id}.anchor"
                arrays[anchor_array] = anchor["voxels"]
                checkpoints: list[dict[str, Any]] = []
                peak = 0.0
                for sample in pending:
                    array_name = (
                        f"{episode_id}.frame_{int(sample['frame_index']):06d}.occluded"
                    )
                    arrays[array_name] = sample["voxels"]
                    fraction = len(sample["voxels"]) / len(anchor["voxels"])
                    peak = max(peak, fraction)
                    checkpoints.append(
                        {
                            "frame_index": int(sample["frame_index"]),
                            "relative_timestamp_ns": int(sample["relative_timestamp_ns"]),
                            "array": array_name,
                            "occluded_voxel_count": len(sample["voxels"]),
                            "occlusion_fraction": fraction,
                        }
                    )
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "scene": scene,
                        "object_id": lifecycle["object_id"],
                        "object_name": lifecycle["object_name"],
                        "semantic_label": lifecycle["semantic_label"],
                        "lifecycle": {
                            "index": lifecycle["lifecycle_index"],
                            "first_timestamp_ns": lifecycle["first_timestamp_ns"],
                            "last_timestamp_ns": lifecycle["last_timestamp_ns"],
                        },
                        "anchor": {
                            "frame_index": anchor["frame_index"],
                            "relative_timestamp_ns": anchor["relative_timestamp_ns"],
                            "array": anchor_array,
                            "voxel_count": len(anchor["voxels"]),
                        },
                        "start_frame_index": checkpoints[0]["frame_index"],
                        "end_frame_index": checkpoints[-1]["frame_index"],
                        "checkpoints": checkpoints,
                        "occlusion_fraction": peak,
                    }
                )
                scene_episode_count += 1
                pending = []

            for frame in frames:
                timestamp = int(frame["relative_timestamp_ns"])
                if not (
                    int(lifecycle["first_timestamp_ns"])
                    <= timestamp
                    < int(lifecycle["last_timestamp_ns"])
                ):
                    flush()
                    continue
                candidates = (
                    lifecycle["voxels"] if anchor is None else anchor["voxels"]
                )
                states = classify_voxel_depths(
                    candidates,
                    depth=np.asarray(frame["depth"]),
                    world_from_camera=np.asarray(frame["world_from_camera"]),
                    camera=camera,
                    voxel_size_m=voxel_size_m,
                    tolerance_m=depth_tolerance_m,
                )
                if anchor is None:
                    if len(states["present"]):
                        anchor = {
                            "frame_index": int(frame["frame_index"]),
                            "relative_timestamp_ns": timestamp,
                            "voxels": sorted_voxel_array(states["present"]),
                        }
                    continue
                occluded = sorted_voxel_array(states["occluded"])
                if not len(occluded):
                    flush()
                    continue
                if pending and int(frame["frame_index"]) != int(pending[-1]["frame_index"]) + 1:
                    flush()
                pending.append(
                    {
                        "frame_index": int(frame["frame_index"]),
                        "relative_timestamp_ns": timestamp,
                        "voxels": occluded,
                    }
                )
            flush()

    stress_layers = {
        "all": _layer_summary(episodes, None),
        "0.50": _layer_summary(episodes, 0.50),
        "0.75": _layer_summary(episodes, 0.75),
        "0.90": _layer_summary(episodes, 0.90),
    }
    scene_summary: dict[str, dict[str, int]] = {}
    for scene in SCENES:
        scene_episodes = [episode for episode in episodes if episode["scene"] == scene]
        headline_count = sum(
            float(episode["occlusion_fraction"]) >= 0.90
            for episode in scene_episodes
        )
        if headline_count == 0:
            raise ValueError(
                f"{scene} has no 0.90 occlusion episode; no qualifying occlusion episode"
            )
        scene_summary[scene] = {
            "episode_count": len(scene_episodes),
            "headline_episode_count": headline_count,
        }
    return arrays, {
        "prediction_inputs_used": False,
        "parameters": parameters,
        "scene_frame_indices": scene_frame_indices,
        "scenes": scene_summary,
        "stress_layers": stress_layers,
        "episodes": episodes,
    }


def _validate_target_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if not arrays:
        raise ValueError("target arrays must be non-empty")
    for name, raw in arrays.items():
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"invalid target array name: {name!r}")
        values = np.asarray(raw)
        if values.dtype != np.int64 or values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"target array must be N x 3 int64: {name}")
        if not len(values) or not np.array_equal(values, sorted_voxel_array(values)):
            raise ValueError(f"target array must be non-empty, sorted, and unique: {name}")


def _generated_target_error(message: str) -> ValueError:
    return ValueError(f"generated occlusion target is invalid: {message}")


def _plain_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def validate_generated_target(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> None:
    """Validate the complete two-scene metadata-to-NPZ closure."""
    try:
        _validate_target_arrays(arrays)
    except ValueError as error:
        raise _generated_target_error(str(error)) from error
    allowed_metadata_fields = {
        "prediction_inputs_used",
        "parameters",
        "scene_frame_indices",
        "scenes",
        "stress_layers",
        "episodes",
    }
    if set(metadata) not in (
        allowed_metadata_fields,
        allowed_metadata_fields | {"contract"},
    ):
        raise _generated_target_error("metadata fields are not exact")
    if metadata.get("prediction_inputs_used") is not False:
        raise _generated_target_error("prediction_inputs_used must be false")
    try:
        parameters_match = _canonical_json_bytes(
            metadata.get("parameters")
        ) == _canonical_json_bytes(_preregistered_parameters())
    except ValueError as error:
        raise _generated_target_error("parameters are not canonical JSON") from error
    if not parameters_match:
        raise _generated_target_error("parameters differ from pre-registration")
    scene_frames = metadata.get("scene_frame_indices")
    scenes = metadata.get("scenes")
    episodes = metadata.get("episodes")
    layers = metadata.get("stress_layers")
    if not isinstance(scene_frames, Mapping) or set(scene_frames) != set(SCENES):
        raise _generated_target_error("scene frame coverage must be exact")
    if not isinstance(scenes, Mapping) or set(scenes) != set(SCENES):
        raise _generated_target_error("scene summaries must be exact")
    if not isinstance(episodes, list) or not episodes:
        raise _generated_target_error("episodes must be a non-empty list")
    if not isinstance(layers, Mapping) or set(layers) != {
        "all",
        "0.50",
        "0.75",
        "0.90",
    }:
        raise _generated_target_error("stress strata must be exact")

    normalized_frames: dict[str, tuple[int, ...]] = {}
    for scene in SCENES:
        values = scene_frames[scene]
        if not isinstance(values, list) or any(
            not _plain_nonnegative_int(value) for value in values
        ):
            raise _generated_target_error(f"{scene} frame indices are invalid")
        indices = tuple(values)
        if not indices or indices != tuple(range(len(indices))):
            raise _generated_target_error(f"{scene} frame coverage is not contiguous")
        normalized_frames[scene] = indices

    episode_ids: list[str] = []
    array_references: list[str] = []
    expected_episode_fields = {
        "episode_id",
        "scene",
        "object_id",
        "object_name",
        "semantic_label",
        "lifecycle",
        "anchor",
        "start_frame_index",
        "end_frame_index",
        "checkpoints",
        "occlusion_fraction",
    }
    for episode in episodes:
        if not isinstance(episode, Mapping) or set(episode) != expected_episode_fields:
            raise _generated_target_error("episode fields are not exact")
        episode_id = episode["episode_id"]
        scene = episode["scene"]
        if not isinstance(episode_id, str) or not episode_id:
            raise _generated_target_error("episode ID is invalid")
        if episode_id in episode_ids:
            raise _generated_target_error("episode IDs are not unique")
        episode_ids.append(episode_id)
        if scene not in SCENES:
            raise _generated_target_error("episode scene is invalid")
        object_id = episode["object_id"]
        if not (
            _plain_nonnegative_int(object_id)
            or (isinstance(object_id, str) and bool(object_id.strip()))
        ):
            raise _generated_target_error("episode object ID is invalid")
        if not isinstance(episode["object_name"], str) or not episode["object_name"]:
            raise _generated_target_error("episode object name is invalid")
        if type(episode["semantic_label"]) is not int or episode[
            "semantic_label"
        ] in UNKNOWN_SEMANTIC_LABELS:
            raise _generated_target_error("episode semantic label is invalid")

        lifecycle = episode["lifecycle"]
        anchor = episode["anchor"]
        checkpoints = episode["checkpoints"]
        if not isinstance(lifecycle, Mapping) or set(lifecycle) != {
            "index",
            "first_timestamp_ns",
            "last_timestamp_ns",
        }:
            raise _generated_target_error("lifecycle is invalid")
        if not isinstance(anchor, Mapping) or set(anchor) != {
            "frame_index",
            "relative_timestamp_ns",
            "array",
            "voxel_count",
        }:
            raise _generated_target_error("anchor is invalid")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise _generated_target_error("checkpoints must be non-empty")
        lifecycle_index = lifecycle["index"]
        first_timestamp = lifecycle["first_timestamp_ns"]
        last_timestamp = lifecycle["last_timestamp_ns"]
        if not (
            _plain_nonnegative_int(lifecycle_index)
            and _plain_nonnegative_int(first_timestamp)
            and _plain_nonnegative_int(last_timestamp)
            and first_timestamp < last_timestamp
        ):
            raise _generated_target_error("lifecycle bounds are invalid")
        anchor_frame = anchor["frame_index"]
        anchor_timestamp = anchor["relative_timestamp_ns"]
        anchor_count = anchor["voxel_count"]
        anchor_array = anchor["array"]
        if not (
            _plain_nonnegative_int(anchor_frame)
            and anchor_frame in normalized_frames[scene]
            and _plain_nonnegative_int(anchor_timestamp)
            and first_timestamp <= anchor_timestamp < last_timestamp
            and type(anchor_count) is int
            and anchor_count > 0
            and isinstance(anchor_array, str)
            and anchor_array in arrays
            and len(arrays[anchor_array]) == anchor_count
        ):
            raise _generated_target_error("anchor binding is invalid")
        array_references.append(anchor_array)
        anchor_voxels = {
            tuple(int(value) for value in row) for row in arrays[anchor_array]
        }

        checkpoint_frames: list[int] = []
        checkpoint_timestamps: list[int] = []
        checkpoint_fractions: list[float] = []
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
                "frame_index",
                "relative_timestamp_ns",
                "array",
                "occluded_voxel_count",
                "occlusion_fraction",
            }:
                raise _generated_target_error("checkpoint fields are not exact")
            frame_index = checkpoint["frame_index"]
            timestamp = checkpoint["relative_timestamp_ns"]
            array_name = checkpoint["array"]
            count = checkpoint["occluded_voxel_count"]
            fraction = checkpoint["occlusion_fraction"]
            if not (
                _plain_nonnegative_int(frame_index)
                and frame_index in normalized_frames[scene]
                and frame_index > anchor_frame
                and _plain_nonnegative_int(timestamp)
                and anchor_timestamp < timestamp < last_timestamp
                and isinstance(array_name, str)
                and array_name in arrays
                and type(count) is int
                and count > 0
                and len(arrays[array_name]) == count
                and {
                    tuple(int(value) for value in row) for row in arrays[array_name]
                }
                <= anchor_voxels
                and isinstance(fraction, (int, float))
                and not isinstance(fraction, bool)
                and np.isfinite(fraction)
                and 0.0 < float(fraction) <= 1.0
                and np.isclose(
                    float(fraction), count / anchor_count, rtol=0.0, atol=1e-12
                )
            ):
                raise _generated_target_error("checkpoint binding is invalid")
            checkpoint_frames.append(frame_index)
            checkpoint_timestamps.append(timestamp)
            checkpoint_fractions.append(float(fraction))
            array_references.append(array_name)
        if checkpoint_frames != list(
            range(checkpoint_frames[0], checkpoint_frames[0] + len(checkpoint_frames))
        ) or checkpoint_timestamps != sorted(set(checkpoint_timestamps)):
            raise _generated_target_error("checkpoint sequence is not consecutive")
        if not (
            episode["start_frame_index"] == checkpoint_frames[0]
            and episode["end_frame_index"] == checkpoint_frames[-1]
            and isinstance(episode["occlusion_fraction"], (int, float))
            and not isinstance(episode["occlusion_fraction"], bool)
            and np.isclose(
                float(episode["occlusion_fraction"]),
                max(checkpoint_fractions),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise _generated_target_error("episode bounds or fraction are invalid")

    if len(array_references) != len(set(array_references)) or set(
        array_references
    ) != set(arrays):
        raise _generated_target_error("array references are dangling or incomplete")

    for scene in SCENES:
        selected = [episode for episode in episodes if episode["scene"] == scene]
        headline = sum(
            float(episode["occlusion_fraction"]) >= 0.90 for episode in selected
        )
        summary = scenes[scene]
        if not isinstance(summary, Mapping) or summary != {
            "episode_count": len(selected),
            "headline_episode_count": headline,
        }:
            raise _generated_target_error(f"{scene} summary is inconsistent")
        if headline <= 0:
            raise _generated_target_error(f"{scene} has no 0.90 episode")

    for label, threshold in (
        ("all", None),
        ("0.50", 0.50),
        ("0.75", 0.75),
        ("0.90", 0.90),
    ):
        expected_ids = [
            str(episode["episode_id"])
            for episode in episodes
            if threshold is None
            or float(episode["occlusion_fraction"]) >= threshold
        ]
        expected_layer: dict[str, Any] = {
            "episode_count": len(expected_ids),
            "episode_ids": expected_ids,
        }
        if threshold is not None:
            expected_layer["minimum_occlusion_fraction"] = threshold
        if layers[label] != expected_layer:
            raise _generated_target_error(f"{label} stress stratum is inconsistent")


def write_occlusion_package(
    output_dir: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    source_paths: Mapping[str, Path],
    expected_source_records: Mapping[str, Mapping[str, Any]] | None = None,
    expected_depth_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    contract_path: Path | None = None,
    status: str = "GENERATED",
) -> Path:
    """Atomically publish deterministic JSON and NPZ target artifacts."""
    episodes = metadata.get("episodes")
    if not isinstance(episodes, Sequence) or not episodes:
        raise ValueError("no qualifying occlusion episode")
    if metadata.get("prediction_inputs_used") is not False:
        raise ValueError("prediction inputs must not be used")
    if status not in {"FIXTURE", "SMOKE", "GENERATED"}:
        raise ValueError(f"invalid target status: {status}")
    validate_generated_target(arrays, metadata)
    contract_witness: _SourceWitness | None = None
    if status == "GENERATED":
        if (
            contract_path is None
            or expected_source_records is None
            or expected_depth_bindings is None
        ):
            raise ValueError(
                "GENERATED publication requires a frozen CONTRACT_ONLY contract"
            )
        contract, contract_witness = _load_checked_contract_with_witness(
            contract_path
        )
        if not (
            contract["source_records"]
            == {
                str(role): dict(record)
                for role, record in sorted(expected_source_records.items())
            }
            and contract["depth_collections"]
            == {scene: dict(expected_depth_bindings[scene]) for scene in SCENES}
            and _canonical_json_bytes(metadata.get("parameters"))
            == _canonical_json_bytes(contract["parameters"])
            and metadata.get("contract") == contract_witness.record
        ):
            raise ValueError(
                "GENERATED publication frozen CONTRACT_ONLY bindings mismatch"
            )
    scene_frame_indices = metadata.get("scene_frame_indices")
    if not isinstance(scene_frame_indices, Mapping):
        raise ValueError("scene frame indices are missing")
    if (expected_source_records is None) != (expected_depth_bindings is None):
        raise ValueError("frozen source and depth bindings must be provided together")
    if expected_source_records is None:
        sources, source_witnesses = _capture_source_allowlist(
            source_paths, scene_frame_indices=scene_frame_indices
        )
    else:
        assert expected_depth_bindings is not None
        sources, source_witnesses = _capture_source_bindings(
            source_paths,
            scene_frame_indices=scene_frame_indices,
            expected_source_records=expected_source_records,
            expected_depth_bindings=expected_depth_bindings,
        )
    target_bytes = deterministic_npz_bytes(arrays)
    payload = {
        "schema_version": 1,
        "manifest_id": "tesse_cd_occlusion_v1_targets",
        "dataset": "TESSE-CD",
        "status": status,
        "targets_generated": True,
        "prediction_inputs_used": False,
        "sources": sources,
        "metadata": dict(metadata),
        "target_arrays": {
            "path": "targets.npz",
            "sha256": hashlib.sha256(target_bytes).hexdigest(),
            "byte_count": len(target_bytes),
            "count": len(arrays),
            "arrays": {
                name: {
                    "shape": list(np.asarray(array).shape),
                    "dtype": str(np.asarray(array).dtype),
                    "element_count": int(np.asarray(array).size),
                }
                for name, array in sorted(arrays.items())
            },
        },
    }
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        manifest_bytes = render_manifest(payload)
        for name, content in (
            ("targets.npz", target_bytes),
            ("manifest.json", manifest_bytes),
        ):
            with (temporary / name).open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        publication_witnesses = [
            source_witnesses[role] for role in sorted(source_witnesses)
        ]
        if contract_witness is not None:
            publication_witnesses.append(contract_witness)
        for witness in publication_witnesses:
            _revalidate_source_witness(witness)
        for witness in publication_witnesses:
            _revalidate_source_identity(witness)
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / "manifest.json"


def _resolve_record_path(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", "")))
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _load_checked_contract_with_witness(
    path: Path,
) -> tuple[dict[str, Any], _SourceWitness]:
    try:
        content, witness = _read_source_bytes(path)
        payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"occlusion contract is not readable: {path}") from error
    if not isinstance(payload, dict) or not (
        payload.get("schema_version") == 1
        and payload.get("manifest_id") == "tesse_cd_occlusion_v1"
        and payload.get("dataset") == "TESSE-CD"
        and payload.get("status") == "CONTRACT_ONLY"
        and payload.get("targets_generated") is False
        and payload.get("prediction_inputs_used") is False
        and payload.get("input_roles") == list(INPUT_ROLE_PATTERNS)
    ):
        raise ValueError("occlusion contract identity mismatch")
    rebuilt = build_contract_manifest(
        source_records=payload.get("source_records", {}),
        depth_bindings=payload.get("depth_collections", {}),
    )
    if payload != rebuilt:
        raise ValueError("occlusion contract input binding mismatch")
    return payload, witness


def _load_checked_contract(path: Path) -> dict[str, Any]:
    payload, _ = _load_checked_contract_with_witness(path)
    return payload


def _read_bound_source(
    role: str, declaration: Mapping[str, Any]
) -> tuple[bytes, _SourceWitness]:
    _validate_record(role, declaration)
    path = _resolve_record_path(declaration)
    content, witness = _read_source_bytes(path)
    if witness.record != dict(declaration):
        raise ValueError(f"source hash or byte count drift: {role}")
    return content, witness


def _camera_from_payload(payload: Mapping[str, Any]) -> dict[str, float | int]:
    camera = payload.get("camera", payload)
    if not isinstance(camera, Mapping):
        raise ValueError("camera source is invalid")
    return {
        "width": int(camera.get("w", camera.get("width"))),
        "height": int(camera.get("h", camera.get("height"))),
        "fx": float(camera["fx"]),
        "fy": float(camera["fy"]),
        "cx": float(camera["cx"]),
        "cy": float(camera["cy"]),
    }


def _load_formal_inputs(
    contract: Mapping[str, Any],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, float | int],
    dict[str, Path],
]:
    from PIL import Image

    declarations = contract.get("source_records")
    depth_declarations = contract.get("depth_collections")
    if not isinstance(declarations, Mapping) or not isinstance(depth_declarations, Mapping):
        raise ValueError("occlusion contract source bindings are missing")
    expected_non_depth = set(SINGLE_ROLES) | {
        f"{scene}.{role}" for scene in SCENES for role in SCENE_ROLES
    }
    if set(declarations) != expected_non_depth:
        raise ValueError("occlusion contract has a non-exact source allowlist")
    source_paths: dict[str, Path] = {}
    global_bytes: dict[str, bytes] = {}
    for role in SINGLE_ROLES:
        content, witness = _read_bound_source(role, declarations[role])
        global_bytes[role] = content
        source_paths[role] = witness.resolved_path
    source = json.loads(global_bytes["source_manifest"])
    schedule = json.loads(global_bytes["schedule"])
    rgbd_lock = json.loads(global_bytes["rgbd_lock"])
    if not (
        source.get("schema_version") == 1
        and source.get("manifest_id") == "tesse_cd_dynamic_v1"
        and source.get("dataset") == "TESSE-CD"
        and source.get("source", {}).get("owner") == "MIT-SPARK Khronos official release"
        and source.get("protocol", {}).get("future_frames_allowed") is False
        and source.get("protocol", {}).get("ground_truth_evaluator_only") is True
        and source.get("protocol", {}).get("runtime_ground_truth_access") is False
        and schedule.get("schema_version") == 2
        and schedule.get("manifest_id") == "tesse_cd_causal_schedule_v2"
        and schedule.get("dataset") == "TESSE-CD"
        and schedule.get("method_predictions_used") is False
        and schedule.get("source_manifest") == dict(declarations["source_manifest"])
        and set(schedule.get("scenes", {})) == set(SCENES)
        and rgbd_lock.get("schema_version") == 1
        and rgbd_lock.get("manifest_id") == "tesse_cd_rgbd_v1"
        and rgbd_lock.get("dataset") == "TESSE-CD"
        and rgbd_lock.get("source_role") == "official_rgbd_export"
        and rgbd_lock.get("source_manifest") == dict(declarations["source_manifest"])
        and set(rgbd_lock.get("scenes", {})) == set(SCENES)
    ):
        raise ValueError("TESSE-CD source/schedule/RGB-D identity mismatch")
    camera_payload = json.loads(global_bytes["camera"])
    camera = _camera_from_payload(camera_payload)
    expected_camera = source.get("camera", {})
    if not (
        camera["width"] == expected_camera.get("width")
        and camera["height"] == expected_camera.get("height")
        and all(camera[name] == expected_camera.get(name) for name in ("fx", "fy", "cx", "cy"))
    ):
        raise ValueError("camera source disagrees with TESSE-CD manifest")

    frames_by_scene: dict[str, list[dict[str, Any]]] = {}
    records_by_scene: dict[str, list[dict[str, Any]]] = {}
    for scene in SCENES:
        scene_bytes: dict[str, bytes] = {}
        for source_name in SCENE_ROLES:
            role = f"{scene}.{source_name}"
            content, witness = _read_bound_source(role, declarations[role])
            scene_bytes[source_name] = content
            source_paths[role] = witness.resolved_path
        sequence = source["sequences"][scene]
        files = sequence["ground_truth"]["files"]
        for source_name in ("changes", "dsg_with_mesh"):
            expected = declarations[f"{scene}.{source_name}"]
            observed = files[source_name]
            if not (
                Path(str(observed.get("path", ""))).resolve()
                == _resolve_record_path(expected)
                and observed.get("sha256") == expected["sha256"]
                and observed.get("size_bytes") == expected["byte_count"]
            ):
                raise ValueError(f"{scene} {source_name} source declaration mismatch")
        locked_scene = rgbd_lock.get("scenes", {}).get(scene, {})
        if locked_scene.get("export_manifest") != declarations[f"{scene}.export_manifest"]:
            raise ValueError(f"{scene} RGB-D export binding mismatch")

        changes = list(
            csv.DictReader(
                io.StringIO(scene_bytes["changes"].decode("utf-8"), newline="")
            )
        )
        dsg_payload = json.loads(scene_bytes["dsg_with_mesh"])
        del scene_bytes["dsg_with_mesh"]
        dsg_records = dsg_payload.get("nodes")
        if not isinstance(dsg_records, list):
            raise ValueError(f"{scene} DSG nodes must be a list")
        match_event_dsg_records(changes, dsg_records, expected_count=len(changes))
        duration = (
            int(sequence["timeline"]["last_depth_timestamp_ns"])
            - int(sequence["timeline"]["first_depth_timestamp_ns"])
        )
        change_times = sorted(
            {
                int(value)
                for row in changes
                for value in (row["AppearedAt"], row["DisappearedAt"])
                if int(value) > 0 and int(value) <= duration
            }
        )
        schedule_times = [
            int(event["event_relative_timestamp_ns"])
            for event in schedule["scenes"][scene].get("events", [])
        ]
        if schedule_times != change_times:
            raise ValueError(f"{scene} schedule and GT lifecycle changes disagree")
        records_by_scene[scene] = dsg_records

        timestamp_rows = list(
            csv.DictReader(
                scene_bytes["timestamps"].decode("utf-8").splitlines()
            )
        )
        trajectories = [
            row
            for row in scene_bytes["trajectory"].decode("utf-8").splitlines()
            if row.strip()
        ]
        expected_count = int(sequence["timeline"]["depth_frame_count"])
        if len(timestamp_rows) != expected_count or len(trajectories) != expected_count:
            raise ValueError(f"{scene} depth/pose/timestamp inputs are incomplete")
        if [int(row["frame_index"]) for row in timestamp_rows] != list(range(expected_count)):
            raise ValueError(f"{scene} source frame indices are not contiguous")
        relative_timestamps = [
            int(row["relative_timestamp_ns"]) for row in timestamp_rows
        ]
        if relative_timestamps != sorted(set(relative_timestamps)):
            raise ValueError(f"{scene} source timestamps are not unique and increasing")

        depth_root = Path(str(depth_declarations[scene].get("root", ""))).resolve()
        locked_root = Path(str(rgbd_lock.get("derived_root", ""))).resolve()
        if depth_root != locked_root / scene / "results":
            raise ValueError(f"{scene} depth root disagrees with checked RGB-D lock")
        frames: list[dict[str, Any]] = []
        depth_records: dict[str, dict[str, Any]] = {}
        for index, (timestamp_row, trajectory_row) in enumerate(
            zip(timestamp_rows, trajectories)
        ):
            path = depth_root / f"depth{index:06d}.png"
            role = f"{scene}.depth.{index:06d}"
            content, witness = _read_source_bytes(path)
            source_paths[role] = witness.resolved_path
            depth_records[role] = witness.record
            with Image.open(io.BytesIO(content)) as image:
                depth_mm = np.asarray(image).copy()
            if depth_mm.shape != (int(camera["height"]), int(camera["width"])) or not np.issubdtype(
                depth_mm.dtype, np.integer
            ):
                raise ValueError(f"{scene} depth frame is invalid: {index}")
            pose_values = np.fromstring(trajectory_row, sep=" ", dtype=np.float64)
            if (
                pose_values.size != 16
                or not np.all(np.isfinite(pose_values))
                or not np.allclose(pose_values.reshape(4, 4)[3], [0.0, 0.0, 0.0, 1.0])
            ):
                raise ValueError(f"{scene} trajectory row is invalid: {index}")
            frames.append(
                {
                    "frame_index": index,
                    "relative_timestamp_ns": int(timestamp_row["relative_timestamp_ns"]),
                    "depth": depth_mm.astype(np.float32) / 1000.0,
                    "world_from_camera": pose_values.reshape(4, 4),
                }
            )
        observed_depth = depth_collection_binding(
            depth_records,
            scene=scene,
            frame_indices=range(expected_count),
        )
        expected_depth = dict(depth_declarations[scene])
        expected_depth.pop("root", None)
        if observed_depth != expected_depth:
            raise ValueError(f"{scene} depth collection hash or byte count drift")
        frames_by_scene[scene] = frames
    return frames_by_scene, records_by_scene, camera, source_paths


def run_generation(
    contract_path: Path, output_dir: Path
) -> Path:
    contract, contract_witness = _load_checked_contract_with_witness(contract_path)
    frames, records, camera, sources = _load_formal_inputs(contract)
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera=camera,
    )
    metadata["contract"] = contract_witness.record
    return write_occlusion_package(
        output_dir,
        arrays=arrays,
        metadata=metadata,
        source_paths=sources,
        expected_source_records=contract["source_records"],
        expected_depth_bindings=contract["depth_collections"],
        contract_path=contract_path,
        status="GENERATED",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_generation(args.contract, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
