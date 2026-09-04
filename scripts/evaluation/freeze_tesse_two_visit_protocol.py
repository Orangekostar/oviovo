#!/usr/bin/env python3
"""Freeze the source-only TESSE-CD two-visit current-state protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


PROTOCOL_ID = "TESSE_TWO_VISIT_CURRENT_V1"
OFFICE_HELD_OUT_STATUS = "OFFICE_NOT_RUN_HELD_OUT"
APARTMENT_STATUS = "APARTMENT_DEVELOPMENT_READY"
SCENES = ("apartment", "office")
SELECTION_INPUTS = ("source_metadata", "causal_schedule")
SOURCE_BINDING_KEYS = {
    "dataset_manifest",
    "rgbd_lock",
    "causal_schedule",
    "common_v2_target_manifest",
    "common_v2_targets",
}
SCENE_SOURCE_BINDING_KEYS = {
    "rgbd_export_manifest",
    "camera",
    "trajectory",
    "timestamps",
}
POLICY_KEYS = {
    "window_frame_count",
    "minimum_trajectory_overlap_fraction",
    "minimum_common_observable_volume_fraction",
    "minimum_camera_viewpoint_histogram_intersection",
    "minimum_changed_object_count",
    "minimum_changed_object_mass_voxels",
    "minimum_old_location_visibility_fraction",
    "selection_key",
    "trajectory_match_radius_m",
    "observable_voxel_size_m",
    "diagnostic_frame_stride",
    "diagnostic_pixel_stride",
    "diagnostic_ray_samples",
    "maximum_depth_m",
    "viewpoint_azimuth_bins",
    "viewpoint_elevation_bins",
}
CANDIDATE_KEYS = {"candidate_id", "visits", "diagnostics", "evaluator_only"}
VISIT_KEYS = {
    "visit_id",
    "start_frame",
    "end_frame",
    "frame_count",
    "source_frame_ids_sha256",
    "input_sha256",
}
DIAGNOSTIC_KEYS = {
    "no_temporal_overlap",
    "trajectory_overlap_fraction",
    "common_observable_volume_fraction",
    "camera_viewpoint_histogram_intersection",
    "changed_object_count",
    "changed_object_mass_voxels",
    "old_location_visibility_fraction",
}
EVALUATOR_ONLY_KEYS = {
    "event_ids",
    "changed_object_count",
    "changed_object_mass_voxels",
    "old_location_visibility_fraction",
}
OFFICE_FREEZE_BINDINGS = {
    "adapter_config",
    "temporal_reasoner",
    "composer_policy",
    "baseline_matrix",
    "metric_contract",
    "success_gates",
}
EXPECTED_SELECTION_KEY = [
    "old_location_visibility_fraction",
    "common_observable_volume_fraction",
    "trajectory_overlap_fraction",
    "camera_viewpoint_histogram_intersection",
    "earliest_t1_end_frame",
]
DEFAULT_SELECTION_POLICY = {
    "window_frame_count": 256,
    "minimum_trajectory_overlap_fraction": 0.05,
    "minimum_common_observable_volume_fraction": 0.05,
    "minimum_camera_viewpoint_histogram_intersection": 0.05,
    "minimum_changed_object_count": 1,
    "minimum_changed_object_mass_voxels": 1,
    "minimum_old_location_visibility_fraction": 0.05,
    "selection_key": EXPECTED_SELECTION_KEY,
    "trajectory_match_radius_m": 3.0,
    "observable_voxel_size_m": 0.25,
    "diagnostic_frame_stride": 16,
    "diagnostic_pixel_stride": 48,
    "diagnostic_ray_samples": 16,
    "maximum_depth_m": 10.0,
    "viewpoint_azimuth_bins": 16,
    "viewpoint_elevation_bins": 4,
}
_HEX = frozenset("0123456789abcdef")


class ProtocolError(ValueError):
    """Raised when two-visit protocol evidence violates the frozen contract."""


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _json_clone(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def protocol_content_sha256(protocol: Mapping[str, Any]) -> str:
    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be a mapping")
    return _sha256_bytes(_canonical_json(dict(protocol)))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _validate_binding(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ProtocolError(f"{label} binding fields are invalid")
    path = value.get("path")
    digest = value.get("sha256")
    byte_count = value.get("byte_count")
    if not isinstance(path, str) or not path or not _is_sha256(digest):
        raise ProtocolError(f"{label} binding values are invalid")
    if isinstance(byte_count, bool) or not isinstance(byte_count, Integral) or byte_count < 0:
        raise ProtocolError(f"{label} binding values are invalid")
    return {
        "path": path,
        "sha256": str(digest),
        "byte_count": int(byte_count),
    }


def _validate_bindings(
    value: object, *, expected: set[str], label: str
) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ProtocolError(f"{label} binding roles are invalid")
    return {
        role: _validate_binding(value[role], label=f"{label} {role}")
        for role in sorted(expected)
    }


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProtocolError(f"{label} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ProtocolError(f"{label} must be at least {minimum}")
    return normalized


def _fraction(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProtocolError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ProtocolError(f"{label} must be a finite fraction")
    return normalized


def _validate_policy(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != POLICY_KEYS:
        raise ProtocolError("selection policy fields are invalid")
    policy = dict(value)
    policy["window_frame_count"] = _integer(
        policy["window_frame_count"], label="window_frame_count", minimum=2
    )
    for key in (
        "minimum_trajectory_overlap_fraction",
        "minimum_common_observable_volume_fraction",
        "minimum_camera_viewpoint_histogram_intersection",
        "minimum_old_location_visibility_fraction",
    ):
        policy[key] = _fraction(policy[key], label=key)
    policy["minimum_changed_object_count"] = _integer(
        policy["minimum_changed_object_count"],
        label="minimum_changed_object_count",
        minimum=1,
    )
    policy["minimum_changed_object_mass_voxels"] = _integer(
        policy["minimum_changed_object_mass_voxels"],
        label="minimum_changed_object_mass_voxels",
        minimum=1,
    )
    if policy["selection_key"] != EXPECTED_SELECTION_KEY:
        raise ProtocolError("selection key is not the preregistered lexicographic key")
    for key in (
        "trajectory_match_radius_m",
        "observable_voxel_size_m",
        "maximum_depth_m",
    ):
        raw = policy[key]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ProtocolError(f"{key} must be numeric")
        normalized = float(raw)
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ProtocolError(f"{key} must be finite and positive")
        policy[key] = normalized
    for key in (
        "diagnostic_frame_stride",
        "diagnostic_pixel_stride",
        "diagnostic_ray_samples",
        "viewpoint_azimuth_bins",
        "viewpoint_elevation_bins",
    ):
        policy[key] = _integer(policy[key], label=key, minimum=1)
    return policy


def _validate_window(
    value: object, *, visit_id: str, expected_count: int
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != VISIT_KEYS:
        raise ProtocolError(f"{visit_id} window fields are invalid")
    start = _integer(value["start_frame"], label=f"{visit_id} start frame")
    end = _integer(value["end_frame"], label=f"{visit_id} end frame")
    count = _integer(value["frame_count"], label=f"{visit_id} frame count", minimum=1)
    if value.get("visit_id") != visit_id:
        raise ProtocolError(f"{visit_id} window visit identity is invalid")
    if end < start or count != end - start + 1 or count != expected_count:
        raise ProtocolError(f"{visit_id} window frame interval is invalid")
    expected_frame_hash = _sha256_bytes(_canonical_json(list(range(start, end + 1))))
    if value.get("source_frame_ids_sha256") != expected_frame_hash:
        raise ProtocolError(f"{visit_id} source frame hash is invalid")
    if not _is_sha256(value.get("input_sha256")):
        raise ProtocolError(f"{visit_id} input hash is invalid")
    return {
        "visit_id": visit_id,
        "start_frame": start,
        "end_frame": end,
        "frame_count": count,
        "source_frame_ids_sha256": expected_frame_hash,
        "input_sha256": str(value["input_sha256"]),
    }


def _validate_candidate(
    value: object, *, policy: Mapping[str, Any], scene: str
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != CANDIDATE_KEYS:
        raise ProtocolError(f"{scene} candidate fields are invalid")
    candidate_id = value.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip() or candidate_id != candidate_id.strip():
        raise ProtocolError(f"{scene} candidate ID is invalid")
    visits = value.get("visits")
    if not isinstance(visits, Mapping) or set(visits) != {"t0", "t1"}:
        raise ProtocolError(f"{scene} candidate visits are invalid")
    count = int(policy["window_frame_count"])
    t0 = _validate_window(visits["t0"], visit_id="t0", expected_count=count)
    t1 = _validate_window(visits["t1"], visit_id="t1", expected_count=count)
    if int(t0["end_frame"]) >= int(t1["start_frame"]):
        raise ProtocolError(f"{scene} candidate visit windows overlap")

    diagnostics = value.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != DIAGNOSTIC_KEYS:
        raise ProtocolError(f"{scene} candidate diagnostic fields are invalid")
    if diagnostics.get("no_temporal_overlap") is not True:
        raise ProtocolError(f"{scene} candidate temporal overlap diagnostic is invalid")
    normalized_diagnostics: dict[str, object] = {"no_temporal_overlap": True}
    for key in (
        "trajectory_overlap_fraction",
        "common_observable_volume_fraction",
        "camera_viewpoint_histogram_intersection",
        "old_location_visibility_fraction",
    ):
        normalized_diagnostics[key] = _fraction(diagnostics[key], label=key)
    for key in ("changed_object_count", "changed_object_mass_voxels"):
        normalized_diagnostics[key] = _integer(diagnostics[key], label=key)

    evaluator = value.get("evaluator_only")
    if not isinstance(evaluator, Mapping) or set(evaluator) != EVALUATOR_ONLY_KEYS:
        raise ProtocolError(f"{scene} evaluator-only fields are invalid")
    event_ids = evaluator.get("event_ids")
    if (
        not isinstance(event_ids, Sequence)
        or isinstance(event_ids, (str, bytes))
        or not event_ids
        or any(not isinstance(item, str) or not item for item in event_ids)
        or len(set(event_ids)) != len(event_ids)
    ):
        raise ProtocolError(f"{scene} evaluator-only event IDs are invalid")
    for key in (
        "changed_object_count",
        "changed_object_mass_voxels",
        "old_location_visibility_fraction",
    ):
        if evaluator.get(key) != normalized_diagnostics[key]:
            raise ProtocolError(f"{scene} evaluator-only diagnostic mismatch: {key}")

    failures: list[str] = []
    thresholds = (
        (
            "trajectory_overlap_fraction",
            "minimum_trajectory_overlap_fraction",
        ),
        (
            "common_observable_volume_fraction",
            "minimum_common_observable_volume_fraction",
        ),
        (
            "camera_viewpoint_histogram_intersection",
            "minimum_camera_viewpoint_histogram_intersection",
        ),
        (
            "old_location_visibility_fraction",
            "minimum_old_location_visibility_fraction",
        ),
        ("changed_object_count", "minimum_changed_object_count"),
        ("changed_object_mass_voxels", "minimum_changed_object_mass_voxels"),
    )
    for diagnostic, threshold in thresholds:
        if normalized_diagnostics[diagnostic] < policy[threshold]:
            failures.append(diagnostic)
    normalized = {
        "candidate_id": candidate_id,
        "visits": {"t0": t0, "t1": t1},
        "diagnostics": normalized_diagnostics,
        "evaluator_only": {
            "event_ids": list(event_ids),
            "changed_object_count": normalized_diagnostics["changed_object_count"],
            "changed_object_mass_voxels": normalized_diagnostics[
                "changed_object_mass_voxels"
            ],
            "old_location_visibility_fraction": normalized_diagnostics[
                "old_location_visibility_fraction"
            ],
        },
        "eligible": not failures,
        "ineligibility_reasons": failures,
    }
    return normalized, tuple(failures)


def _selection_rank(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    diagnostics = candidate["diagnostics"]
    t1 = candidate["visits"]["t1"]
    return (
        -float(diagnostics["old_location_visibility_fraction"]),
        -float(diagnostics["common_observable_volume_fraction"]),
        -float(diagnostics["trajectory_overlap_fraction"]),
        -float(diagnostics["camera_viewpoint_histogram_intersection"]),
        float(t1["end_frame"]),
    )


def candidate_visit_windows(
    scene_schedule: Mapping[str, Any], *, window_frame_count: int
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Derive causal event-local t0 and post-event t1 checkpoint windows."""

    count = _integer(
        window_frame_count, label="window_frame_count", minimum=2
    )
    if not isinstance(scene_schedule, Mapping):
        raise ProtocolError("scene schedule must be a mapping")
    frame_count = _integer(
        scene_schedule.get("frame_count"), label="scene frame_count", minimum=count
    )
    raw_events = scene_schedule.get("events")
    if (
        not isinstance(raw_events, Sequence)
        or isinstance(raw_events, (str, bytes))
        or not raw_events
    ):
        raise ProtocolError("scene schedule must contain events")
    events: list[tuple[int, tuple[int, ...]]] = []
    for event in raw_events:
        if not isinstance(event, Mapping):
            raise ProtocolError("scene event must be a mapping")
        intervention = _integer(
            event.get("intervention_frame_index"),
            label="intervention_frame_index",
        )
        checkpoints = event.get("common_checkpoint_frame_indices")
        if (
            not isinstance(checkpoints, Sequence)
            or isinstance(checkpoints, (str, bytes))
            or not checkpoints
        ):
            raise ProtocolError("scene event must contain common checkpoints")
        normalized = tuple(
            _integer(item, label="common checkpoint") for item in checkpoints
        )
        if normalized != tuple(sorted(set(normalized))):
            raise ProtocolError("common checkpoints must be sorted and unique")
        events.append((intervention, normalized))
    interventions = tuple(item[0] for item in events)
    if interventions != tuple(sorted(set(interventions))):
        raise ProtocolError("intervention frames must be sorted and unique")
    windows = []
    for intervention, checkpoints in events:
        t0_end = intervention - 1
        t0_start = t0_end - count + 1
        if t0_start < 0:
            continue
        for t1_end in checkpoints:
            t1_start = t1_end - count + 1
            if t1_start <= intervention:
                continue
            if t1_end >= frame_count:
                raise ProtocolError("t1 checkpoint exceeds scene frame count")
            windows.append(((t0_start, t0_end), (t1_start, t1_end)))
    if not windows:
        raise ProtocolError("no event-local post-change t1 window is available")
    return tuple(sorted(set(windows)))


def _read_regular(path: Path, *, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    before = absolute.stat(follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProtocolError(f"{label} must be a regular non-symlink file")
    data = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(data) != after.st_size
    ):
        raise ProtocolError(f"{label} changed while reading")
    return data


def _file_record(
    path: Path, *, label: str, repository_relative: bool = False
) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    data = _read_regular(absolute, label=label)
    record_path = str(absolute)
    if repository_relative:
        try:
            record_path = absolute.relative_to(REPO_ROOT).as_posix()
        except ValueError as error:
            raise ProtocolError(f"{label} must be inside the repository") from error
    return {
        "path": record_path,
        "sha256": _sha256_bytes(data),
        "byte_count": len(data),
    }


def _verify_record(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    observed = _file_record(path, label=label)
    try:
        declared = {
            "path": record["path"],
            "sha256": record["sha256"],
            "byte_count": record["byte_count"],
        }
    except KeyError as error:
        raise ProtocolError(f"{label} binding fields are invalid") from error
    declared["path"] = str(Path(os.path.abspath(os.fspath(path))))
    if observed != declared:
        raise ProtocolError(f"{label} binding mismatch")


def _load_json_file(path: Path, *, label: str) -> dict[str, Any]:
    data = _read_regular(path, label=label)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must contain a JSON object")
    return value


def _trajectory_rows(path: Path, *, frame_count: int) -> tuple[np.ndarray, tuple[bytes, ...]]:
    raw = _read_regular(path, label="trajectory")
    lines = tuple(raw.splitlines(keepends=True))
    if len(lines) != frame_count:
        raise ProtocolError("trajectory row count does not match scene frame count")
    matrices: list[np.ndarray] = []
    for line in lines:
        values = np.fromstring(line.decode("ascii"), sep=" ", dtype=np.float64)
        if values.shape != (16,) or not np.all(np.isfinite(values)):
            raise ProtocolError("trajectory rows must contain finite 4x4 matrices")
        matrix = values.reshape(4, 4)
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ProtocolError("trajectory rows must be homogeneous transforms")
        matrices.append(matrix)
    return np.stack(matrices), lines


def _timestamp_rows(
    path: Path, *, frame_count: int
) -> tuple[dict[str, str], ...]:
    data = _read_regular(path, label="timestamps")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProtocolError("timestamps must be UTF-8") from error
    rows = tuple(csv.DictReader(text.splitlines()))
    expected_fields = {
        "frame_index",
        "sensor_timestamp_ns",
        "relative_timestamp_ns",
    }
    if len(rows) != frame_count or not rows or set(rows[0]) != expected_fields:
        raise ProtocolError("timestamps schema or row count is invalid")
    normalized: list[dict[str, str]] = []
    previous_timestamp = -1
    for index, row in enumerate(rows):
        if int(row["frame_index"]) != index:
            raise ProtocolError("timestamp frame indices must be contiguous")
        timestamp = int(row["sensor_timestamp_ns"])
        relative = int(row["relative_timestamp_ns"])
        if timestamp <= previous_timestamp or relative < 0:
            raise ProtocolError("timestamps must be strictly increasing")
        previous_timestamp = timestamp
        normalized.append(
            {
                "frame_index": str(index),
                "sensor_timestamp_ns": str(timestamp),
                "relative_timestamp_ns": str(relative),
            }
        )
    return tuple(normalized)


def compute_window_input_sha256(
    rgbd_root: Path,
    *,
    scene: str,
    start_frame: int,
    end_frame: int,
    _record_cache: dict[Path, dict[str, object]] | None = None,
    _trajectory_lines: tuple[bytes, ...] | None = None,
    _timestamps: tuple[dict[str, str], ...] | None = None,
) -> str:
    """Hash every RGB-D, pose, timestamp, camera, and export byte in a visit."""

    start = _integer(start_frame, label="start_frame")
    end = _integer(end_frame, label="end_frame")
    if end < start:
        raise ProtocolError("window input frame interval is invalid")
    root = Path(rgbd_root)
    scene_root = root / scene
    export = _load_json_file(scene_root / "export_manifest.json", label="RGB-D export manifest")
    frame_count = _integer(export.get("frame_count"), label="RGB-D frame_count", minimum=1)
    if end >= frame_count:
        raise ProtocolError("window input exceeds RGB-D export")
    if _trajectory_lines is None:
        _, trajectory_lines = _trajectory_rows(
            scene_root / "traj.txt", frame_count=frame_count
        )
    else:
        trajectory_lines = _trajectory_lines
    timestamps = (
        _timestamp_rows(scene_root / "timestamps.csv", frame_count=frame_count)
        if _timestamps is None
        else _timestamps
    )
    cache = {} if _record_cache is None else _record_cache

    def record(path: Path, label: str) -> dict[str, object]:
        absolute = Path(os.path.abspath(os.fspath(path)))
        if absolute not in cache:
            cache[absolute] = _file_record(absolute, label=label)
        return cache[absolute]

    components: list[tuple[str, str, int]] = []
    for role, path in (
        ("camera", root / "cam_params.json"),
        ("export_manifest", scene_root / "export_manifest.json"),
    ):
        item = record(path, role)
        components.append((role, str(item["sha256"]), int(item["byte_count"])))
    for index in range(start, end + 1):
        for role, path in (
            ("rgb", scene_root / "results" / f"frame{index:06d}.jpg"),
            ("depth", scene_root / "results" / f"depth{index:06d}.png"),
        ):
            item = record(path, f"{scene} {role} frame {index}")
            components.append(
                (f"{role}/{index:06d}", str(item["sha256"]), int(item["byte_count"]))
            )
        trajectory_line = trajectory_lines[index]
        timestamp_bytes = _canonical_json(timestamps[index])
        components.append(
            (
                f"trajectory/{index:06d}",
                _sha256_bytes(trajectory_line),
                len(trajectory_line),
            )
        )
        components.append(
            (
                f"timestamp/{index:06d}",
                _sha256_bytes(timestamp_bytes),
                len(timestamp_bytes),
            )
        )
    digest = hashlib.sha256()
    for role, checksum, byte_count in components:
        digest.update(f"{role}\0{checksum}\0{byte_count}\n".encode("ascii"))
    return digest.hexdigest()


def _trajectory_overlap_fraction(
    poses0: np.ndarray, poses1: np.ndarray, *, radius_m: float
) -> float:
    positions0 = poses0[:, :3, 3]
    positions1 = poses1[:, :3, 3]
    distances = np.linalg.norm(
        positions0[:, np.newaxis, :] - positions1[np.newaxis, :, :], axis=2
    )
    coverage0 = float(np.mean(np.min(distances, axis=1) <= radius_m))
    coverage1 = float(np.mean(np.min(distances, axis=0) <= radius_m))
    return min(coverage0, coverage1)


def _viewpoint_histogram_intersection(
    poses0: np.ndarray,
    poses1: np.ndarray,
    *,
    azimuth_bins: int,
    elevation_bins: int,
) -> float:
    def histogram(poses: np.ndarray) -> np.ndarray:
        forward = poses[:, :3, 2]
        norms = np.linalg.norm(forward, axis=1)
        if np.any(norms <= 0.0):
            raise ProtocolError("camera poses contain zero forward axes")
        unit = forward / norms[:, np.newaxis]
        azimuth = np.arctan2(unit[:, 1], unit[:, 0])
        elevation = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))
        values, _, _ = np.histogram2d(
            azimuth,
            elevation,
            bins=(azimuth_bins, elevation_bins),
            range=((-np.pi, np.pi), (-np.pi / 2.0, np.pi / 2.0)),
        )
        return values.ravel() / len(poses)

    return float(np.minimum(histogram(poses0), histogram(poses1)).sum())


def _observable_voxels(
    *,
    scene_root: Path,
    poses: np.ndarray,
    window: tuple[int, int],
    camera: Mapping[str, Any],
    frame_stride: int,
    pixel_stride: int,
    ray_samples: int,
    voxel_size_m: float,
    maximum_depth_m: float,
) -> set[tuple[int, int, int]]:
    import cv2

    start, end = window
    sampled = list(range(start, end + 1, frame_stride))
    if sampled[-1] != end:
        sampled.append(end)
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    height = int(camera["h"])
    width = int(camera["w"])
    rows = np.arange(pixel_stride // 2, height, pixel_stride, dtype=np.int64)
    columns = np.arange(pixel_stride // 2, width, pixel_stride, dtype=np.int64)
    grid_columns, grid_rows = np.meshgrid(columns, rows)
    row_indices = grid_rows.ravel()
    column_indices = grid_columns.ravel()
    fractions = np.linspace(1.0 / ray_samples, 1.0, ray_samples)
    result: set[tuple[int, int, int]] = set()
    for index in sampled:
        path = scene_root / "results" / f"depth{index:06d}.png"
        depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None or depth_mm.shape != (height, width):
            raise ProtocolError(f"cannot read diagnostic depth frame: {path}")
        measured = np.asarray(depth_mm[row_indices, column_indices], dtype=np.float64) / 1000.0
        valid = np.isfinite(measured) & (measured > 0.0) & (measured <= maximum_depth_m)
        if not np.any(valid):
            continue
        depth = measured[valid]
        x = (column_indices[valid] - cx) * depth / fx
        y = (row_indices[valid] - cy) * depth / fy
        endpoints = np.column_stack((x, y, depth))
        camera_points = (endpoints[:, np.newaxis, :] * fractions[np.newaxis, :, np.newaxis]).reshape(-1, 3)
        pose = poses[index]
        world = camera_points @ pose[:3, :3].T + pose[:3, 3]
        keys = np.floor(world / voxel_size_m).astype(np.int64)
        result.update(tuple(int(item) for item in row) for row in keys)
    if not result:
        raise ProtocolError("diagnostic visit has no observable volume")
    return result


def _common_observable_fraction(
    first: set[tuple[int, int, int]], second: set[tuple[int, int, int]]
) -> float:
    denominator = min(len(first), len(second))
    if denominator <= 0:
        raise ProtocolError("observable volume sets must be non-empty")
    return len(first & second) / denominator


def _event_support(
    arrays: Mapping[str, np.ndarray],
    *,
    event_ids: Sequence[str],
    t1_end_frame: int,
) -> tuple[int, float]:
    region_union: set[tuple[int, int, int]] = set()
    free_union: set[tuple[int, int, int]] = set()
    for event_id in event_ids:
        region_name = f"{event_id}.region"
        if region_name not in arrays:
            raise ProtocolError(f"common-v2 targets are missing {region_name}")
        region = np.asarray(arrays[region_name], dtype=np.int64)
        if region.ndim != 2 or region.shape[1] != 3:
            raise ProtocolError(f"common-v2 target shape is invalid: {region_name}")
        region_union.update(tuple(int(item) for item in row) for row in region)
        prefix = f"{event_id}.confirmed_free."
        available = sorted(
            int(name.rsplit(".", 1)[1])
            for name in arrays
            if name.startswith(prefix) and int(name.rsplit(".", 1)[1]) <= t1_end_frame
        )
        if not available:
            raise ProtocolError(f"common-v2 targets have no usable free-space checkpoint for {event_id}")
        free = np.asarray(arrays[f"{prefix}{available[-1]:06d}"], dtype=np.int64)
        if free.ndim != 2 or free.shape[1] != 3:
            raise ProtocolError(f"common-v2 free target shape is invalid: {event_id}")
        free_union.update(tuple(int(item) for item in row) for row in free)
    if not region_union:
        raise ProtocolError("changed-object target mass is empty")
    return len(region_union), len(region_union & free_union) / len(region_union)


def _changed_object_count(
    path: Path, *, t0_relative_ns: int, t1_relative_ns: int
) -> int:
    data = _read_regular(path, label="GT change schedule")
    rows = tuple(csv.DictReader(data.decode("utf-8").splitlines()))
    changed: set[str] = set()
    for row in rows:
        symbol = str(row.get("ObjectSymbol", "")).strip()
        transitions = (
            int(str(row.get("AppearedAt", "0")).strip()),
            int(str(row.get("DisappearedAt", "0")).strip()),
        )
        if symbol and any(t0_relative_ns < value < t1_relative_ns for value in transitions if value):
            changed.add(symbol)
    if not changed:
        raise ProtocolError("candidate pair has no GT-supported changed object")
    return len(changed)


def derive_candidate_metadata(
    *,
    dataset_manifest_path: Path,
    rgbd_lock_path: Path,
    schedule_path: Path,
    common_target_manifest_path: Path,
    rgbd_root: Path,
    selection_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute source-only and evaluator-only candidate diagnostics."""

    policy = _validate_policy(selection_policy)
    dataset_manifest = _load_json_file(dataset_manifest_path, label="TESSE dataset manifest")
    rgbd_lock = _load_json_file(rgbd_lock_path, label="TESSE RGB-D lock")
    schedule = _load_json_file(schedule_path, label="TESSE causal schedule")
    target_manifest = _load_json_file(
        common_target_manifest_path, label="common-v2 target manifest"
    )
    if (
        dataset_manifest.get("dataset") != "TESSE-CD"
        or rgbd_lock.get("dataset") != "TESSE-CD"
        or schedule.get("dataset") != "TESSE-CD"
        or schedule.get("method_predictions_used") is not False
        or target_manifest.get("dataset") != "TESSE-CD"
        or target_manifest.get("prediction_inputs_used") is not False
    ):
        raise ProtocolError("TESSE source manifest identity is invalid")
    target_record = target_manifest.get("target_arrays")
    if not isinstance(target_record, Mapping):
        raise ProtocolError("common-v2 target artifact binding is invalid")
    raw_target_path = target_record.get("path")
    if not isinstance(raw_target_path, str) or Path(raw_target_path).is_absolute():
        raise ProtocolError("common-v2 target artifact path must be relative")
    target_path = common_target_manifest_path.parent / raw_target_path
    _verify_record(target_path, target_record, label="common-v2 target artifact")
    with np.load(target_path, allow_pickle=False) as archive:
        target_arrays = {name: archive[name] for name in archive.files}

    camera_payload = _load_json_file(Path(rgbd_root) / "cam_params.json", label="camera")
    camera = camera_payload.get("camera")
    if not isinstance(camera, Mapping):
        raise ProtocolError("camera parameters are invalid")
    source_bindings = {
        "dataset_manifest": _file_record(
            dataset_manifest_path, label="dataset manifest", repository_relative=True
        ),
        "rgbd_lock": _file_record(
            rgbd_lock_path, label="RGB-D lock", repository_relative=True
        ),
        "causal_schedule": _file_record(
            schedule_path, label="causal schedule", repository_relative=True
        ),
        "common_v2_target_manifest": _file_record(
            common_target_manifest_path, label="common-v2 target manifest"
        ),
        "common_v2_targets": _file_record(target_path, label="common-v2 targets"),
    }
    schedule_scenes = schedule.get("scenes")
    lock_scenes = rgbd_lock.get("scenes")
    if not isinstance(schedule_scenes, Mapping) or not isinstance(lock_scenes, Mapping):
        raise ProtocolError("TESSE scene manifests are invalid")
    scenes: dict[str, Any] = {}
    file_record_cache: dict[Path, dict[str, object]] = {}
    volume_cache: dict[tuple[str, int, int], set[tuple[int, int, int]]] = {}
    for scene in SCENES:
        if scene not in schedule_scenes or scene not in lock_scenes:
            raise ProtocolError(f"TESSE manifests are missing scene: {scene}")
        scene_schedule = schedule_scenes[scene]
        frame_count = _integer(
            scene_schedule.get("frame_count"), label=f"{scene} frame_count", minimum=1
        )
        scene_root = Path(rgbd_root) / scene
        export_path = scene_root / "export_manifest.json"
        lock_export = lock_scenes[scene].get("export_manifest")
        if not isinstance(lock_export, Mapping):
            raise ProtocolError(f"{scene} RGB-D lock export binding is invalid")
        _verify_record(export_path, lock_export, label=f"{scene} RGB-D export manifest")
        export = _load_json_file(export_path, label=f"{scene} RGB-D export manifest")
        if export.get("frame_count") != frame_count or export.get("scene") != scene:
            raise ProtocolError(f"{scene} RGB-D export disagrees with schedule")
        poses, trajectory_lines = _trajectory_rows(
            scene_root / "traj.txt", frame_count=frame_count
        )
        timestamps = _timestamp_rows(
            scene_root / "timestamps.csv", frame_count=frame_count
        )
        change_record = scene_schedule.get("sources", {}).get("gt_changes")
        if not isinstance(change_record, Mapping):
            raise ProtocolError(f"{scene} GT change binding is invalid")
        change_path = Path(str(change_record.get("path", "")))
        _verify_record(change_path, change_record, label=f"{scene} GT changes")
        windows = candidate_visit_windows(
            scene_schedule, window_frame_count=int(policy["window_frame_count"])
        )
        raw_events = scene_schedule["events"]
        candidates: list[dict[str, Any]] = []
        for t0_window, t1_window in windows:
            t0_start, t0_end = t0_window
            t1_start, t1_end = t1_window
            event_ids = [
                str(event["event_id"])
                for event in raw_events
                if t0_end < int(event["intervention_frame_index"]) < t1_start
            ]
            if not event_ids:
                raise ProtocolError(f"{scene} candidate has no causal event support")
            mass, old_visibility = _event_support(
                target_arrays, event_ids=event_ids, t1_end_frame=t1_end
            )
            changed_count = _changed_object_count(
                change_path,
                t0_relative_ns=int(timestamps[t0_end]["relative_timestamp_ns"]),
                t1_relative_ns=int(timestamps[t1_start]["relative_timestamp_ns"]),
            )
            for window in (t0_window, t1_window):
                key = (scene, window[0], window[1])
                if key not in volume_cache:
                    volume_cache[key] = _observable_voxels(
                        scene_root=scene_root,
                        poses=poses,
                        window=window,
                        camera=camera,
                        frame_stride=int(policy["diagnostic_frame_stride"]),
                        pixel_stride=int(policy["diagnostic_pixel_stride"]),
                        ray_samples=int(policy["diagnostic_ray_samples"]),
                        voxel_size_m=float(policy["observable_voxel_size_m"]),
                        maximum_depth_m=float(policy["maximum_depth_m"]),
                    )
            common = _common_observable_fraction(
                volume_cache[(scene, t0_start, t0_end)],
                volume_cache[(scene, t1_start, t1_end)],
            )
            trajectory = _trajectory_overlap_fraction(
                poses[t0_start : t0_end + 1],
                poses[t1_start : t1_end + 1],
                radius_m=float(policy["trajectory_match_radius_m"]),
            )
            camera_overlap = _viewpoint_histogram_intersection(
                poses[t0_start : t0_end + 1],
                poses[t1_start : t1_end + 1],
                azimuth_bins=int(policy["viewpoint_azimuth_bins"]),
                elevation_bins=int(policy["viewpoint_elevation_bins"]),
            )

            def visit_payload(
                visit_id: str,
                interval: tuple[int, int],
                *,
                scene_name: str = scene,
                trajectory_snapshot: Sequence[str] = trajectory_lines,
                timestamp_snapshot: Sequence[Mapping[str, Any]] = timestamps,
            ) -> dict[str, object]:
                start, end = interval
                return {
                    "visit_id": visit_id,
                    "start_frame": start,
                    "end_frame": end,
                    "frame_count": end - start + 1,
                    "source_frame_ids_sha256": _sha256_bytes(
                        _canonical_json(list(range(start, end + 1)))
                    ),
                    "input_sha256": compute_window_input_sha256(
                        rgbd_root,
                        scene=scene_name,
                        start_frame=start,
                        end_frame=end,
                        _record_cache=file_record_cache,
                        _trajectory_lines=trajectory_snapshot,
                        _timestamps=timestamp_snapshot,
                    ),
                }

            identifier = (
                f"{scene}-t0-{t0_start:06d}-{t0_end:06d}"
                f"-t1-{t1_start:06d}-{t1_end:06d}"
            )
            diagnostics = {
                "no_temporal_overlap": True,
                "trajectory_overlap_fraction": trajectory,
                "common_observable_volume_fraction": common,
                "camera_viewpoint_histogram_intersection": camera_overlap,
                "changed_object_count": changed_count,
                "changed_object_mass_voxels": mass,
                "old_location_visibility_fraction": old_visibility,
            }
            candidates.append(
                {
                    "candidate_id": identifier,
                    "visits": {
                        "t0": visit_payload("t0", t0_window),
                        "t1": visit_payload("t1", t1_window),
                    },
                    "diagnostics": diagnostics,
                    "evaluator_only": {
                        "event_ids": event_ids,
                        "changed_object_count": changed_count,
                        "changed_object_mass_voxels": mass,
                        "old_location_visibility_fraction": old_visibility,
                    },
                }
            )
        scenes[scene] = {
            "role": "development" if scene == "apartment" else "held_out",
            "source_bindings": {
                "rgbd_export_manifest": _file_record(
                    export_path, label=f"{scene} export manifest"
                ),
                "camera": _file_record(Path(rgbd_root) / "cam_params.json", label="camera"),
                "trajectory": _file_record(
                    scene_root / "traj.txt", label=f"{scene} trajectory"
                ),
                "timestamps": _file_record(
                    scene_root / "timestamps.csv", label=f"{scene} timestamps"
                ),
            },
            "candidates": candidates,
        }
    return {
        "schema_version": 1,
        "dataset": "TESSE-CD",
        "selection_inputs": list(SELECTION_INPUTS),
        "method_predictions_used": False,
        "selection_policy": policy,
        "source_bindings": source_bindings,
        "scenes": scenes,
    }


def freeze_protocol(candidate_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate source-only candidate evidence and choose one pair per scene."""

    if not isinstance(candidate_metadata, Mapping) or set(candidate_metadata) != {
        "schema_version",
        "dataset",
        "selection_inputs",
        "method_predictions_used",
        "selection_policy",
        "source_bindings",
        "scenes",
    }:
        raise ProtocolError("candidate metadata fields are invalid")
    if (
        candidate_metadata.get("schema_version") != 1
        or candidate_metadata.get("dataset") != "TESSE-CD"
        or candidate_metadata.get("selection_inputs") != list(SELECTION_INPUTS)
        or candidate_metadata.get("method_predictions_used") is not False
    ):
        raise ProtocolError("candidate metadata identity is invalid")
    policy = _validate_policy(candidate_metadata["selection_policy"])
    source_bindings = _validate_bindings(
        candidate_metadata["source_bindings"],
        expected=SOURCE_BINDING_KEYS,
        label="protocol source",
    )
    scenes = candidate_metadata.get("scenes")
    if not isinstance(scenes, Mapping) or set(scenes) != set(SCENES):
        raise ProtocolError("candidate scenes must be exactly apartment and office")

    frozen_scenes: dict[str, Any] = {}
    seen_candidate_ids: set[str] = set()
    for scene in SCENES:
        value = scenes[scene]
        expected_role = "development" if scene == "apartment" else "held_out"
        if (
            not isinstance(value, Mapping)
            or set(value) != {"role", "source_bindings", "candidates"}
            or value.get("role") != expected_role
        ):
            raise ProtocolError(f"{scene} scene metadata fields are invalid")
        scene_bindings = _validate_bindings(
            value["source_bindings"],
            expected=SCENE_SOURCE_BINDING_KEYS,
            label=f"{scene} source",
        )
        candidates_value = value.get("candidates")
        if (
            not isinstance(candidates_value, Sequence)
            or isinstance(candidates_value, (str, bytes))
            or not candidates_value
        ):
            raise ProtocolError(f"{scene} must contain candidate pairs")
        candidates: list[dict[str, Any]] = []
        for raw in candidates_value:
            candidate, _ = _validate_candidate(raw, policy=policy, scene=scene)
            identifier = str(candidate["candidate_id"])
            if identifier in seen_candidate_ids:
                raise ProtocolError("candidate IDs must be globally unique")
            seen_candidate_ids.add(identifier)
            candidates.append(candidate)
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        if not eligible:
            raise ProtocolError(f"{scene} has no eligible candidate pair")
        selected = min(eligible, key=_selection_rank)
        method_input = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "dataset": "TESSE-CD",
            "scene": scene,
            "candidate_id": selected["candidate_id"],
            "visits": _json_clone(selected["visits"]),
            "source_bindings": scene_bindings,
            "runtime_ground_truth_inputs": [],
        }
        frozen_scenes[scene] = {
            "role": expected_role,
            "execution_status": (
                APARTMENT_STATUS if scene == "apartment" else OFFICE_HELD_OUT_STATUS
            ),
            "selected_candidate_id": selected["candidate_id"],
            "candidates": sorted(candidates, key=lambda item: item["candidate_id"]),
            "method_input_manifest": method_input,
        }

    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_BEFORE_METHOD_SCORES",
        "dataset": "TESSE-CD",
        "selection_inputs": list(SELECTION_INPUTS),
        "method_predictions_used": False,
        "ground_truth_policy": (
            "evaluator_only_protocol_definition_never_runtime_method_input"
        ),
        "selection_policy": policy,
        "source_bindings": source_bindings,
        "scenes": frozen_scenes,
    }


def validate_frozen_protocol(protocol: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_version",
        "protocol_id",
        "status",
        "dataset",
        "selection_inputs",
        "method_predictions_used",
        "ground_truth_policy",
        "selection_policy",
        "source_bindings",
        "scenes",
    }
    if (
        not isinstance(protocol, Mapping)
        or set(protocol) != expected_fields
        or protocol.get("schema_version") != 1
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "FROZEN_BEFORE_METHOD_SCORES"
        or protocol.get("dataset") != "TESSE-CD"
        or protocol.get("selection_inputs") != list(SELECTION_INPUTS)
        or protocol.get("method_predictions_used") is not False
    ):
        raise ProtocolError("frozen protocol identity is invalid")
    scenes = protocol.get("scenes")
    if not isinstance(scenes, Mapping) or set(scenes) != set(SCENES):
        raise ProtocolError("frozen protocol scenes are invalid")
    candidate_metadata = {
        "schema_version": protocol["schema_version"],
        "dataset": protocol["dataset"],
        "selection_inputs": protocol["selection_inputs"],
        "method_predictions_used": protocol["method_predictions_used"],
        "selection_policy": protocol["selection_policy"],
        "source_bindings": protocol["source_bindings"],
        "scenes": {},
    }
    for scene in SCENES:
        scene_record = scenes[scene]
        if not isinstance(scene_record, Mapping):
            raise ProtocolError(f"{scene} frozen scene record is invalid")
        method_input = scene_record.get("method_input_manifest")
        candidates = scene_record.get("candidates")
        if not isinstance(method_input, Mapping) or not isinstance(candidates, Sequence):
            raise ProtocolError(f"{scene} frozen scene evidence is invalid")
        candidate_metadata["scenes"][scene] = {
            "role": scene_record.get("role"),
            "source_bindings": method_input.get("source_bindings"),
            "candidates": [
                {
                    key: candidate[key]
                    for key in (
                        "candidate_id",
                        "visits",
                        "diagnostics",
                        "evaluator_only",
                    )
                }
                for candidate in candidates
                if isinstance(candidate, Mapping)
                and all(
                    key in candidate
                    for key in (
                        "candidate_id",
                        "visits",
                        "diagnostics",
                        "evaluator_only",
                    )
                )
            ],
        }
    try:
        replayed = freeze_protocol(candidate_metadata)
    except (KeyError, TypeError, ProtocolError) as error:
        raise ProtocolError("frozen protocol replay failed") from error
    if replayed != dict(protocol):
        raise ProtocolError("frozen protocol replay mismatch")


def authorize_scene_run(
    protocol: Mapping[str, Any],
    scene: str,
    *,
    office_release: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the bound method input only when the scene is authorized."""

    validate_frozen_protocol(protocol)
    if scene not in SCENES:
        raise ProtocolError(f"unknown two-visit scene: {scene}")
    scene_record = protocol["scenes"][scene]
    if scene == "office":
        if office_release is None:
            raise ProtocolError(OFFICE_HELD_OUT_STATUS)
        if (
            not isinstance(office_release, Mapping)
            or office_release.get("status") != "OFFICE_RELEASE_AUTHORIZED"
            or office_release.get("protocol_content_sha256")
            != protocol_content_sha256(protocol)
            or office_release.get("office_attempt_count") != 0
        ):
            raise ProtocolError(f"{OFFICE_HELD_OUT_STATUS}: release receipt is invalid")
        frozen_bindings = office_release.get("frozen_bindings")
        if not isinstance(frozen_bindings, Mapping) or set(frozen_bindings) != OFFICE_FREEZE_BINDINGS:
            raise ProtocolError("Office freeze bindings are incomplete")
        _validate_bindings(
            frozen_bindings,
            expected=OFFICE_FREEZE_BINDINGS,
            label="Office freeze",
        )
    method_input = scene_record.get("method_input_manifest")
    if not isinstance(method_input, Mapping):
        raise ProtocolError(f"{scene} method input manifest is invalid")
    return _json_clone(method_input)


def _atomic_json(path: Path, value: object) -> None:
    data = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    before = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProtocolError(f"{label} must be a regular non-symlink file")
    data = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ProtocolError(f"{label} changed while reading")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metadata", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--rgbd-lock", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--common-target-manifest", type=Path)
    parser.add_argument("--rgbd-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    derive_arguments = (
        args.dataset_manifest,
        args.rgbd_lock,
        args.schedule,
        args.common_target_manifest,
        args.rgbd_root,
    )
    if args.candidate_metadata is not None:
        if any(value is not None for value in derive_arguments):
            raise ProtocolError("candidate metadata and derivation inputs are exclusive")
        metadata = _strict_json(args.candidate_metadata, label="candidate metadata")
    else:
        if any(value is None for value in derive_arguments):
            raise ProtocolError("all derivation inputs are required")
        metadata = derive_candidate_metadata(
            dataset_manifest_path=args.dataset_manifest,
            rgbd_lock_path=args.rgbd_lock,
            schedule_path=args.schedule,
            common_target_manifest_path=args.common_target_manifest,
            rgbd_root=args.rgbd_root,
            selection_policy=DEFAULT_SELECTION_POLICY,
        )
    protocol = freeze_protocol(metadata)
    _atomic_json(args.output, protocol)
    print("TESSE_TWO_VISIT_PROTOCOL_FROZEN")
    return 0


__all__ = [
    "APARTMENT_STATUS",
    "DEFAULT_SELECTION_POLICY",
    "OFFICE_HELD_OUT_STATUS",
    "PROTOCOL_ID",
    "ProtocolError",
    "authorize_scene_run",
    "candidate_visit_windows",
    "compute_window_input_sha256",
    "derive_candidate_metadata",
    "freeze_protocol",
    "main",
    "protocol_content_sha256",
    "validate_frozen_protocol",
]


if __name__ == "__main__":
    raise SystemExit(main())
