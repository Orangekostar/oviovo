#!/usr/bin/env python3
"""Derive prediction-independent TESSE-CD common-v2 target contracts."""

from __future__ import annotations

import argparse
import csv
import io
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np


Voxel = tuple[int, int, int]
UNKNOWN_SEMANTIC_LABELS = {-1, 4_294_967_295}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "configs/evaluation/manifests/tesse_cd.json"
DEFAULT_DERIVED_RGBD_LOCK = (
    REPO_ROOT / "configs/evaluation/manifests/tesse_cd_rgbd_v1.json"
)


def _attributes(record: Mapping[str, Any]) -> Mapping[str, Any]:
    attributes = record.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ValueError("DSG object record has no attributes")
    return attributes


def _known_semantic_label(attributes: Mapping[str, Any]) -> bool:
    value = attributes.get("semantic_label")
    return type(value) is int and value not in UNKNOWN_SEMANTIC_LABELS


def _intervals(attributes: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    first_values = attributes.get("first_observed_ns")
    last_values = attributes.get("last_observed_ns")
    if not isinstance(first_values, Sequence) or not isinstance(
        last_values, Sequence
    ):
        raise ValueError("DSG object record has no observation intervals")
    if len(first_values) != len(last_values) or not first_values:
        raise ValueError("DSG object observation intervals are inconsistent")
    intervals: list[tuple[int, int]] = []
    for first_raw, last_raw in zip(first_values, last_values):
        if type(first_raw) is not int or type(last_raw) is not int:
            raise ValueError("DSG object interval timestamps must be integers")
        first = int(first_raw)
        last = int(last_raw)
        if first < 0 or last <= first:
            raise ValueError("DSG object intervals must satisfy first < last")
        intervals.append((first, last))
    return tuple(intervals)


def active_dsg_records(
    records: Sequence[Mapping[str, Any]], *, timestamp_ns: int
) -> tuple[Mapping[str, Any], ...]:
    """Return known-label objects active under ``first <= timestamp < last``."""
    if type(timestamp_ns) is not int or timestamp_ns < 0:
        raise ValueError("timestamp_ns must be a non-negative integer")
    active: list[Mapping[str, Any]] = []
    for record in records:
        attributes = _attributes(record)
        if not _known_semantic_label(attributes):
            continue
        if any(first <= timestamp_ns < last for first, last in _intervals(attributes)):
            active.append(record)
    return tuple(active)


def _event_timestamp(value: Any, *, field: str) -> int:
    try:
        timestamp = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error
    if timestamp < 0:
        raise ValueError(f"{field} must be non-negative")
    return timestamp


def match_event_dsg_records(
    changes: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = 14,
) -> tuple[dict[str, Any], ...]:
    """Bind every changed object to one known-label DSG lifecycle interval."""
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    if len(changes) != expected_count:
        raise ValueError(f"expected exactly {expected_count} event object records")

    by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        symbol = str(_attributes(record).get("name", "")).strip()
        if symbol:
            by_symbol.setdefault(symbol, []).append(record)

    matched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in changes:
        symbol = str(row.get("ObjectSymbol", "")).strip()
        if not symbol or symbol in seen:
            raise ValueError("event object symbols must be non-empty and unique")
        seen.add(symbol)
        candidates = by_symbol.get(symbol, [])
        if len(candidates) != 1:
            raise ValueError(f"expected exactly {expected_count} event object records")
        record = candidates[0]
        attributes = _attributes(record)
        if not _known_semantic_label(attributes):
            raise ValueError(f"event object {symbol} has unknown semantic label")

        appeared = _event_timestamp(row.get("AppearedAt"), field="AppearedAt")
        disappeared = _event_timestamp(
            row.get("DisappearedAt"), field="DisappearedAt"
        )
        if appeared == 0 and disappeared == 0:
            raise ValueError(f"event object {symbol} has an empty lifecycle")
        interval_matches = [
            (index, first, last)
            for index, (first, last) in enumerate(_intervals(attributes))
            if (appeared == 0 or first == appeared)
            and (disappeared == 0 or last == disappeared)
        ]
        if len(interval_matches) != 1:
            raise ValueError(f"event object {symbol} has no unique DSG interval match")
        interval_index, first, last = interval_matches[0]
        matched.append(
            {
                "symbol": symbol,
                "record_id": record.get("id"),
                "interval_index": interval_index,
                "first_timestamp_ns": first,
                "last_timestamp_ns": last,
                "appeared_at_ns": appeared,
                "disappeared_at_ns": disappeared,
                "record": record,
            }
        )
    return tuple(matched)


def _points3(points: np.ndarray, *, name: str) -> np.ndarray:
    try:
        values = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _homogeneous_transform(matrix: np.ndarray, *, name: str) -> np.ndarray:
    try:
        transform = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite homogeneous 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} must be homogeneous")
    return transform


def world_object_points(
    points: np.ndarray, world_from_object: np.ndarray
) -> np.ndarray:
    """Transform object-frame geometry into the official world frame."""
    local = _points3(points, name="object points")
    transform = _homogeneous_transform(
        world_from_object, name="world_from_object"
    )
    return local @ transform[:3, :3].T + transform[:3, 3]


def dsg_interval_world_points(matched: Mapping[str, Any]) -> np.ndarray:
    attributes = _attributes(matched["record"])
    interval_index = int(matched["interval_index"])
    dynamic = attributes.get("dynamic_object_points")
    if isinstance(dynamic, Sequence) and interval_index < len(dynamic):
        points = _points3(
            np.asarray(dynamic[interval_index]),
            name=f"{matched['symbol']} dynamic world object points",
        )
        if len(points):
            return points

    mesh = attributes.get("mesh")
    bounding_box = attributes.get("bounding_box")
    if not isinstance(mesh, Mapping) or not isinstance(bounding_box, Mapping):
        raise ValueError(f"{matched['symbol']} has no DSG mesh geometry")
    rotation = bounding_box.get("world_R_center")
    if not (
        bounding_box.get("type") == "AABB"
        and isinstance(rotation, Mapping)
        and rotation.get("w") == 1.0
        and rotation.get("x") == 0.0
        and rotation.get("y") == 0.0
        and rotation.get("z") == 0.0
    ):
        raise ValueError(f"{matched['symbol']} DSG mesh transform is unsupported")
    local = _points3(
        np.asarray(mesh.get("points")), name=f"{matched['symbol']} mesh points"
    )
    if len(local) == 0:
        raise ValueError(f"{matched['symbol']} DSG mesh points are empty")
    position = np.asarray(attributes.get("position"), dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"{matched['symbol']} DSG position must be a finite 3-vector")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    return world_object_points(local, transform)


def voxel_keys(
    points: np.ndarray, *, voxel_size_m: float = 0.05
) -> tuple[Voxel, ...]:
    """Floor-quantize points to sorted, unique integer voxel keys."""
    values = _points3(points, name="points")
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive and finite")
    integer = np.floor(values / float(voxel_size_m)).astype(np.int64)
    return tuple(sorted({tuple(int(item) for item in row) for row in integer}))


def _voxel_set(keys: Iterable[Voxel]) -> set[Voxel]:
    result: set[Voxel] = set()
    for key in keys:
        if len(key) != 3 or any(type(value) is not int for value in key):
            raise ValueError("voxel keys must be integer triples")
        result.add((key[0], key[1], key[2]))
    return result


def dilate_voxels(
    keys: Iterable[Voxel], *, radius_voxels: int = 1
) -> tuple[Voxel, ...]:
    """Apply deterministic Chebyshev dilation to integer voxel keys."""
    if type(radius_voxels) is not int or radius_voxels < 0:
        raise ValueError("radius_voxels must be a non-negative integer")
    source = _voxel_set(keys)
    offsets = tuple(product(range(-radius_voxels, radius_voxels + 1), repeat=3))
    return tuple(
        sorted(
            {
                (x + dx, y + dy, z + dz)
                for x, y, z in source
                for dx, dy, dz in offsets
            }
        )
    )


def _segment_voxels(start: np.ndarray, stop: np.ndarray, size: float) -> set[Voxel]:
    current = np.floor(start / size).astype(np.int64)
    target = np.floor(stop / size).astype(np.int64)
    direction = stop - start
    step = np.sign(direction).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        next_boundary = (current + (step > 0).astype(np.int64)) * size
        t_max = np.where(direction != 0, (next_boundary - start) / direction, np.inf)
        t_delta = np.where(direction != 0, size / np.abs(direction), np.inf)

    visited: set[Voxel] = set()
    while True:
        visited.add(tuple(int(value) for value in current))
        if np.array_equal(current, target):
            return visited
        minimum = float(np.min(t_max))
        axes = np.flatnonzero(np.isclose(t_max, minimum, rtol=0.0, atol=1e-12))
        current[axes] += step[axes]
        t_max[axes] += t_delta[axes]


def ray_free_voxels(
    origin: np.ndarray,
    endpoints: np.ndarray,
    *,
    voxel_size_m: float = 0.05,
    endpoint_margin_m: float = 0.05,
) -> tuple[Voxel, ...]:
    """Ray-carve through the point exactly ``endpoint_margin_m`` short."""
    start = np.asarray(origin, dtype=np.float64)
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise ValueError("origin must be a finite 3-vector")
    points = _points3(endpoints, name="endpoints")
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive and finite")
    if not np.isfinite(endpoint_margin_m) or endpoint_margin_m < 0:
        raise ValueError("endpoint_margin_m must be non-negative and finite")

    free: set[Voxel] = set()
    for endpoint in points:
        delta = endpoint - start
        length = float(np.linalg.norm(delta))
        free_length = length - float(endpoint_margin_m)
        if free_length <= 0:
            continue
        stop = start + delta * (free_length / length)
        free.update(_segment_voxels(start, stop, float(voxel_size_m)))
    return tuple(sorted(free))


def _project(
    points_world: np.ndarray,
    *,
    camera_from_world: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = _points3(points_world, name="world points")
    transform = _homogeneous_transform(
        camera_from_world, name="camera_from_world"
    )
    if len(intrinsics) != 4:
        raise ValueError("intrinsics must be (fx, fy, cx, cy)")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
        raise ValueError("camera intrinsics must be finite with positive focal lengths")
    if (
        len(image_size) != 2
        or type(image_size[0]) is not int
        or type(image_size[1]) is not int
        or image_size[0] <= 0
        or image_size[1] <= 0
    ):
        raise ValueError("image_size must contain positive integer dimensions")

    camera = points @ transform[:3, :3].T + transform[:3, 3]
    depth = camera[:, 2]
    positive = depth > 0
    u = np.zeros(len(points), dtype=np.int64)
    v = np.zeros(len(points), dtype=np.int64)
    u[positive] = np.floor(fx * camera[positive, 0] / depth[positive] + cx + 0.5).astype(
        np.int64
    )
    v[positive] = np.floor(fy * camera[positive, 1] / depth[positive] + cy + 0.5).astype(
        np.int64
    )
    height, width = image_size
    valid = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    pixels = np.column_stack((v, u))
    return pixels, depth, valid


def _z_buffer_depths(
    pixels: np.ndarray, depth: np.ndarray, valid: np.ndarray
) -> dict[tuple[int, int], float]:
    nearest: dict[tuple[int, int], float] = {}
    for index in np.flatnonzero(valid):
        pixel = tuple(int(value) for value in pixels[index])
        nearest[pixel] = min(nearest.get(pixel, np.inf), float(depth[index]))
    return nearest


def visible_surface_voxels(
    points_world: np.ndarray,
    *,
    camera_from_world: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    image_size: tuple[int, int],
    voxel_size_m: float = 0.05,
) -> tuple[Voxel, ...]:
    """Return only the nearest positive-depth surface at each image pixel."""
    points = _points3(points_world, name="world points")
    pixels, depth, valid = _project(
        points,
        camera_from_world=camera_from_world,
        intrinsics=intrinsics,
        image_size=image_size,
    )
    nearest = _z_buffer_depths(pixels, depth, valid)
    selected = [
        points[index]
        for index in np.flatnonzero(valid)
        if np.isclose(
            depth[index],
            nearest[tuple(int(value) for value in pixels[index])],
            rtol=0.0,
            atol=1e-12,
        )
    ]
    values = np.asarray(selected, dtype=np.float64).reshape((-1, 3))
    return voxel_keys(values, voxel_size_m=voxel_size_m)


def occlusion_shadow_voxels(
    object_points_world: np.ndarray,
    background_points_world: np.ndarray,
    *,
    camera_from_world: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    image_size: tuple[int, int],
    voxel_size_m: float = 0.05,
    depth_margin_m: float = 0.05,
) -> tuple[Voxel, ...]:
    """Select background behind the object's z-buffer footprint."""
    if not np.isfinite(depth_margin_m) or depth_margin_m < 0:
        raise ValueError("depth_margin_m must be non-negative and finite")
    object_points = _points3(object_points_world, name="object world points")
    background_points = _points3(
        background_points_world, name="background world points"
    )
    object_pixels, object_depth, object_valid = _project(
        object_points,
        camera_from_world=camera_from_world,
        intrinsics=intrinsics,
        image_size=image_size,
    )
    object_z = _z_buffer_depths(object_pixels, object_depth, object_valid)
    background_pixels, background_depth, background_valid = _project(
        background_points,
        camera_from_world=camera_from_world,
        intrinsics=intrinsics,
        image_size=image_size,
    )
    selected = [
        background_points[index]
        for index in np.flatnonzero(background_valid)
        if tuple(int(value) for value in background_pixels[index]) in object_z
        and background_depth[index]
        > object_z[tuple(int(value) for value in background_pixels[index])]
        + depth_margin_m
    ]
    values = np.asarray(selected, dtype=np.float64).reshape((-1, 3))
    return voxel_keys(values, voxel_size_m=voxel_size_m)


def revealed_background_voxels(
    background: Iterable[Voxel],
    shadow: Iterable[Voxel],
    pre_observed: Iterable[Voxel],
    post_observed: Iterable[Voxel],
) -> tuple[Voxel, ...]:
    """Intersect background/shadow/post observations and remove pre observations."""
    revealed = (
        _voxel_set(background)
        & _voxel_set(shadow)
        & _voxel_set(post_observed)
    ) - _voxel_set(pre_observed)
    return tuple(sorted(revealed))


def observed_background_voxels(
    background_points_world: np.ndarray,
    observed_points_world: np.ndarray,
    *,
    max_residual_m: float = 0.05,
    voxel_size_m: float = 0.05,
) -> tuple[Voxel, ...]:
    from scipy.spatial import cKDTree

    background = _points3(background_points_world, name="background world points")
    observed = _points3(observed_points_world, name="observed world points")
    if not np.isfinite(max_residual_m) or max_residual_m < 0:
        raise ValueError("max_residual_m must be non-negative and finite")
    if len(background) == 0 or len(observed) == 0:
        return ()
    distances, _ = cKDTree(observed).query(background, k=1)
    selected = background[distances <= max_residual_m]
    return voxel_keys(selected, voxel_size_m=voxel_size_m)


def semantic_voxel_labels(
    labeled_points: Iterable[tuple[np.ndarray, int]],
    *,
    voxel_size_m: float = 0.05,
) -> tuple[tuple[int, int, int, int], ...]:
    votes: dict[Voxel, dict[int, int]] = {}
    for raw_points, raw_label in labeled_points:
        points = _points3(raw_points, name="semantic world points")
        if type(raw_label) is not int or raw_label in UNKNOWN_SEMANTIC_LABELS:
            raise ValueError("semantic voxel labels must be known integer IDs")
        keys = np.floor(points / voxel_size_m).astype(np.int64)
        for row in keys:
            voxel = tuple(int(value) for value in row)
            label_votes = votes.setdefault(voxel, {})
            label_votes[raw_label] = label_votes.get(raw_label, 0) + 1
    resolved = []
    for voxel, label_votes in sorted(votes.items()):
        label = min(label_votes, key=lambda item: (-label_votes[item], item))
        resolved.append((*voxel, label))
    return tuple(resolved)


def reject_prediction_input_paths(paths: Iterable[str | Path]) -> None:
    """Reject inputs explicitly assigned the prediction source role."""
    for path in paths:
        raise ValueError(f"prediction or method output path is forbidden: {path}")


def _allowed_empty_target(name: str, values: np.ndarray) -> bool:
    return (
        ".confirmed_free." in name and values.shape == (0, 3)
    ) or (
        ".current_semantic." in name and values.shape == (0, 4)
    ) or (
        name.endswith(".revealed_background") and values.shape == (0, 3)
    )


def validate_generation_inputs(
    *,
    source_paths: Iterable[Path],
    target_arrays: Mapping[str, np.ndarray],
    prediction_input_paths: Iterable[Path] = (),
) -> None:
    """Fail closed before publishing any real target package."""
    sources = tuple(Path(path) for path in source_paths)
    reject_prediction_input_paths(prediction_input_paths)
    if not sources:
        raise ValueError("source paths must be non-empty")
    for path in sources:
        if not path.is_file():
            raise ValueError(f"source is not a file: {path}")
    if not target_arrays:
        raise ValueError("target arrays must be non-empty")
    for name, array in target_arrays.items():
        if not str(name).strip():
            raise ValueError("target array names must be non-empty")
        values = np.asarray(array)
        if values.size == 0 and not _allowed_empty_target(str(name), values):
            raise ValueError(f"target array is empty: {name}")
        if not np.issubdtype(values.dtype, np.number) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError(f"target array must be finite and numeric: {name}")


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"source is not a file: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    resolved = path.resolve()
    try:
        serialized_path = resolved.relative_to(REPO_ROOT).as_posix()
        path_base = "repository"
    except ValueError:
        serialized_path = str(resolved)
        path_base = "absolute"
    return {
        "path": serialized_path,
        "path_base": path_base,
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def build_contract_manifest(
    *,
    source_manifest: Path,
    schedule: Path,
    ground_truth_sources: Mapping[str, Path],
) -> dict[str, Any]:
    """Build a deterministic contract manifest without claiming target generation."""
    sources = {
        str(name): Path(path)
        for name, path in sorted(ground_truth_sources.items())
    }
    if not sources or any(not name.strip() for name in sources):
        raise ValueError("ground_truth_sources must be non-empty and named")
    source_record = _file_record(Path(source_manifest))
    schedule_record = _file_record(Path(schedule))
    ground_truth_records = {
        name: _file_record(path) for name, path in sources.items()
    }
    binding_payload = {
        "source_manifest": source_record,
        "schedule": schedule_record,
        "ground_truth": ground_truth_records,
    }
    binding_bytes = json.dumps(
        binding_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "manifest_id": "tesse_cd_common_v2",
        "dataset": "TESSE-CD",
        "status": "CONTRACT_ONLY",
        "targets_generated": False,
        "fixture_tested": True,
        "expected_event_object_count": 14,
        "prediction_inputs_used": False,
        "parameters": {
            "voxel_size_m": 0.05,
            "dilation_voxels": 1,
            "ray_endpoint_margin_m": 0.05,
            "pre_event_frames": 450,
            "post_event_frames": 450,
            "z_buffer": "nearest_positive_depth_per_pixel",
            "active_interval": "first <= timestamp < last",
            "semantic_voxel_label_rule": "point_majority_then_lowest_label_id",
            "unobservable_background_rule": "exclude_from_background_f5_and_recovery",
        },
        "source_manifest": source_record,
        "schedule": schedule_record,
        "ground_truth_sources": ground_truth_records,
        "input_binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
    }


def render_manifest(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_contract_manifest(output: Path, payload: Mapping[str, Any]) -> Path:
    if payload.get("status") != "CONTRACT_ONLY":
        raise ValueError("contract manifest status must be CONTRACT_ONLY")
    if payload.get("targets_generated") is not False:
        raise ValueError("contract manifest cannot claim generated targets")
    if payload.get("prediction_inputs_used") is not False:
        raise ValueError("contract manifest cannot use prediction inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            handle.write(render_manifest(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise ValueError(f"output already exists: {output}") from error
    return output


def load_generation_contract(
    source_manifest: Path, schedule: Path
) -> dict[str, Any]:
    source_record = _file_record(source_manifest)
    schedule_record = _file_record(schedule)
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    protocol = source.get("protocol", {})
    owner = source.get("source", {})
    if not (
        type(source.get("schema_version")) is int
        and source.get("schema_version") == 1
        and source.get("manifest_id") == "tesse_cd_dynamic_v1"
        and source.get("dataset") == "TESSE-CD"
        and isinstance(owner, Mapping)
        and owner.get("owner") == "MIT-SPARK Khronos official release"
        and isinstance(protocol, Mapping)
        and protocol.get("future_frames_allowed") is False
        and protocol.get("ground_truth_evaluator_only") is True
        and protocol.get("runtime_ground_truth_access") is False
    ):
        raise ValueError("source manifest does not match canonical source identity")

    schedule_payload = json.loads(schedule.read_text(encoding="utf-8"))
    legacy_source_record = dict(source_record)
    legacy_source_record.pop("path_base")
    declared_source_record = schedule_payload.get("source_manifest")
    if not (
        schedule_payload.get("schema_version") == 2
        and schedule_payload.get("manifest_id") == "tesse_cd_causal_schedule_v2"
        and schedule_payload.get("dataset") == "TESSE-CD"
        and schedule_payload.get("method_predictions_used") is False
        and declared_source_record in (source_record, legacy_source_record)
    ):
        raise ValueError("schedule is not bound to the canonical source manifest")

    sequences = source.get("sequences", {})
    schedule_scenes = schedule_payload.get("scenes", {})
    if set(sequences) != {"apartment", "office"} or set(schedule_scenes) != {
        "apartment",
        "office",
    }:
        raise ValueError("source and schedule must contain apartment and office exactly")

    source_records: dict[str, dict[str, Any]] = {}
    scenes: dict[str, dict[str, Any]] = {}
    total_event_objects = 0
    scene_event_object_counts: dict[str, int] = {}
    scene_event_counts: dict[str, int] = {}
    for scene in ("apartment", "office"):
        sequence = sequences[scene]
        files = sequence["ground_truth"]["files"]
        declarations = {
            "database": sequence["bag"]["database"],
            "background_mesh": files["background_mesh"],
            "changes": files["changes"],
            "dsg": files["dsg"],
            "dsg_with_mesh": files["dsg_with_mesh"],
        }
        scene_paths = {
            name: Path(str(declaration.get("path", "")))
            for name, declaration in declarations.items()
        }
        for name, path in scene_paths.items():
            observed = _file_record(path)
            declaration = declarations[name]
            if (
                observed["sha256"] != declaration.get("sha256")
                or observed["byte_count"] != declaration.get("size_bytes")
            ):
                raise ValueError(f"{scene} {name} source declaration mismatch")
            source_records[f"{scene}.{name}"] = observed

        with scene_paths["changes"].open(encoding="utf-8", newline="") as handle:
            changes = list(csv.DictReader(handle))
        dsg_payload = json.loads(
            scene_paths["dsg_with_mesh"].read_text(encoding="utf-8")
        )
        dsg_records = dsg_payload.get("nodes", [])
        if not isinstance(dsg_records, list):
            raise ValueError(f"{scene} DSG nodes must be a list")
        matched = match_event_dsg_records(
            changes, dsg_records, expected_count=len(changes)
        )
        total_event_objects += len(matched)
        scene_event_object_counts[scene] = len(matched)

        events = schedule_scenes[scene].get("events", [])
        event_times = [
            int(event["event_relative_timestamp_ns"]) for event in events
        ]
        if event_times != sorted(set(event_times)):
            raise ValueError(f"{scene} schedule event times must be unique and increasing")
        declared_event_times = sorted(
            {
                timestamp
                for row in changes
                for timestamp in (
                    _event_timestamp(row.get("AppearedAt"), field="AppearedAt"),
                    _event_timestamp(
                        row.get("DisappearedAt"), field="DisappearedAt"
                    ),
                )
                if timestamp > 0
                and timestamp
                <= int(sequence["timeline"]["last_depth_timestamp_ns"])
                - int(sequence["timeline"]["first_depth_timestamp_ns"])
            }
        )
        if event_times != declared_event_times:
            raise ValueError(f"{scene} schedule and GT event timestamps disagree")
        scene_event_counts[scene] = len(events)
        scenes[scene] = {
            "sequence": sequence,
            "schedule": schedule_scenes[scene],
            "changes": changes,
            "dsg_records": dsg_records,
            "matched_event_records": matched,
            "paths": scene_paths,
        }

    if total_event_objects != 14:
        raise ValueError("expected exactly 14 event object records")
    return {
        "source_manifest": source_record,
        "schedule": schedule_record,
        "source_payload": source,
        "schedule_payload": schedule_payload,
        "source_records": source_records,
        "scenes": scenes,
        "event_object_count": total_event_objects,
        "scene_event_object_counts": scene_event_object_counts,
        "scene_event_counts": scene_event_counts,
        "prediction_inputs_used": False,
    }


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if not arrays:
        raise ValueError("target arrays must be non-empty")
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, raw in sorted(arrays.items()):
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"invalid target array name: {name!r}")
            array = np.asarray(raw)
            if array.size == 0 and not _allowed_empty_target(name, array):
                raise ValueError(f"target array is empty: {name}")
            if not np.issubdtype(array.dtype, np.number) or not np.all(
                np.isfinite(array)
            ):
                raise ValueError(f"target array must be finite and numeric: {name}")
            if array.ndim == 2 and len(array) > 1:
                order = np.lexsort(tuple(array[:, index] for index in reversed(range(array.shape[1]))))
                array = array[order]
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.ascontiguousarray(array), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def write_target_package(
    output_dir: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    source_paths: Iterable[Path],
    metadata: Mapping[str, Any],
    prediction_input_paths: Iterable[Path] = (),
    status: str = "GENERATED",
) -> Path:
    sources = tuple(Path(path) for path in source_paths)
    validate_generation_inputs(
        source_paths=sources,
        target_arrays=arrays,
        prediction_input_paths=prediction_input_paths,
    )
    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}")
    if status not in {"FIXTURE", "GENERATED", "SMOKE"}:
        raise ValueError(f"invalid target package status: {status}")
    revealed_nonempty = {
        name.removesuffix(".revealed_background"): bool(np.asarray(array).size)
        for name, array in arrays.items()
        if name.endswith(".revealed_background")
    }
    if revealed_nonempty:
        observability = metadata.get("background_observable_by_event")
        if not isinstance(observability, Mapping):
            raise ValueError(
                "revealed targets require explicit background observability metadata"
            )
        if set(observability) != set(revealed_nonempty) or any(
            type(value) is not bool for value in observability.values()
        ):
            raise ValueError("background observability metadata does not cover events")
        if any(
            observability[event_id] is not nonempty
            for event_id, nonempty in revealed_nonempty.items()
        ):
            raise ValueError("background observability disagrees with arrays")
        observable_count = sum(observability.values())
        if (
            metadata.get("background_observable_event_count") != observable_count
            or metadata.get("unobservable_revealed_target_event_count")
            != len(observability) - observable_count
        ):
            raise ValueError("background observability counts disagree")
        scenes = {
            event_id.split("_event_", 1)[0] for event_id in revealed_nonempty
        }
        if any(
            not any(
                value
                for event_id, value in observability.items()
                if event_id.startswith(f"{scene}_event_")
            )
            for scene in scenes
        ):
            raise ValueError(
                "each scene must contain at least one background-observable event"
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        target_bytes = deterministic_npz_bytes(arrays)
        targets_path = temporary / "targets.npz"
        with targets_path.open("xb") as handle:
            handle.write(target_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        array_records = {
            name: {
                "shape": list(np.asarray(array).shape),
                "dtype": str(np.asarray(array).dtype),
                "element_count": int(np.asarray(array).size),
            }
            for name, array in sorted(arrays.items())
        }
        payload = {
            "schema_version": 1,
            "manifest_id": "tesse_cd_common_v2_targets",
            "dataset": "TESSE-CD",
            "status": status,
            "targets_generated": True,
            "prediction_inputs_used": False,
            "metadata": dict(metadata),
            "sources": [_file_record(path) for path in sorted(sources)],
            "target_arrays": {
                "path": "targets.npz",
                "sha256": hashlib.sha256(target_bytes).hexdigest(),
                "byte_count": len(target_bytes),
                "count": len(arrays),
                "arrays": array_records,
            },
        }
        manifest = temporary / "manifest.json"
        with manifest.open("xb") as handle:
            handle.write(render_manifest(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / "manifest.json"


def derive_target_arrays(
    contract: Mapping[str, Any],
    *,
    frames_by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
    background_points_by_scene: Mapping[str, np.ndarray],
    window_frames: int = 450,
    event_limit: int | None = None,
    scenes: Sequence[str] = ("apartment", "office"),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if type(window_frames) is not int or window_frames <= 0:
        raise ValueError("window_frames must be a positive integer")
    if event_limit is not None and (
        type(event_limit) is not int or event_limit <= 0
    ):
        raise ValueError("event_limit must be a positive integer")
    selected_scenes = tuple(scenes)
    if (
        not selected_scenes
        or len(set(selected_scenes)) != len(selected_scenes)
        or not set(selected_scenes) <= {"apartment", "office"}
    ):
        raise ValueError("scenes must be a unique non-empty TESSE-CD scene subset")
    source_payload = contract.get("source_payload", {})
    camera = source_payload.get("camera", {})
    intrinsics = (
        float(camera["fx"]),
        float(camera["fy"]),
        float(camera["cx"]),
        float(camera["cy"]),
    )
    image_size = (int(camera["height"]), int(camera["width"]))
    arrays: dict[str, np.ndarray] = {}
    derived_events = 0
    background_observable_by_event: dict[str, bool] = {}

    def array_from_keys(
        keys: Iterable[Voxel], *, label: str, allow_empty: bool = False
    ) -> np.ndarray:
        values = tuple(sorted(_voxel_set(keys)))
        if not values:
            if allow_empty:
                return np.empty((0, 3), dtype=np.int64)
            raise ValueError(f"target array is empty: {label}")
        return np.asarray(values, dtype=np.int64)

    def centers(keys: Iterable[Voxel]) -> np.ndarray:
        values = np.asarray(tuple(sorted(_voxel_set(keys))), dtype=np.float64)
        return (values + 0.5) * 0.05

    def frame_observations(
        candidate_keys: Iterable[Voxel],
        frame: Mapping[str, Any],
        *,
        mode: str,
    ) -> tuple[Voxel, ...]:
        candidates = tuple(sorted(_voxel_set(candidate_keys)))
        if not candidates:
            return ()
        points = centers(candidates)
        depth_image = np.asarray(frame["depth"], dtype=np.float64)
        if depth_image.shape != image_size:
            raise ValueError("official depth frame shape disagrees with camera manifest")
        world_from_camera = _homogeneous_transform(
            np.asarray(frame["world_from_camera"]), name="world_from_camera"
        )
        try:
            camera_from_world = np.linalg.inv(world_from_camera)
        except np.linalg.LinAlgError as error:
            raise ValueError("world_from_camera must be invertible") from error
        pixels, candidate_depth, valid = _project(
            points,
            camera_from_world=camera_from_world,
            intrinsics=intrinsics,
            image_size=image_size,
        )
        selected: list[Voxel] = []
        for index in np.flatnonzero(valid):
            row, column = (int(value) for value in pixels[index])
            measured = float(depth_image[row, column])
            if not np.isfinite(measured) or measured <= 0:
                continue
            if mode == "observed":
                keep = abs(float(candidate_depth[index]) - measured) <= 0.05
            elif mode == "free":
                keep = float(candidate_depth[index]) <= measured - 0.05
            else:
                raise ValueError(f"unknown frame observation mode: {mode}")
            if keep:
                selected.append(candidates[index])
        return tuple(sorted(set(selected)))

    for scene in selected_scenes:
        scene_contract = contract["scenes"][scene]
        frame_map: dict[int, Mapping[str, Any]] = {}
        for frame in frames_by_scene.get(scene, ()):  # type: ignore[arg-type]
            index = int(frame["frame_index"])
            if index in frame_map:
                raise ValueError(f"{scene} has a duplicate official frame {index}")
            frame_map[index] = frame
        if not frame_map:
            raise ValueError(f"{scene} official frame input is empty")
        background_points = _points3(
            background_points_by_scene[scene], name=f"{scene} background points"
        )
        background_keys = set(voxel_keys(background_points))
        matched = scene_contract["matched_event_records"]
        events = scene_contract["schedule"]["events"]
        if event_limit is not None:
            events = events[:event_limit]

        for event in events:
            event_id = str(event["event_id"])
            timestamp = int(event["event_relative_timestamp_ns"])
            intervention = int(event["intervention_frame_index"])
            changed = [
                item
                for item in matched
                if timestamp
                in {int(item["appeared_at_ns"]), int(item["disappeared_at_ns"])}
            ]
            removed = [
                item
                for item in changed
                if int(item["disappeared_at_ns"]) == timestamp
            ]
            if not changed or not removed:
                raise ValueError(f"{event_id} has no matched disappearing DSG geometry")

            changed_points: list[np.ndarray] = []
            removed_points: list[np.ndarray] = []
            for item in changed:
                points = dsg_interval_world_points(item)
                changed_points.append(points)
                if item in removed:
                    removed_points.append(points)
            changed_surface = set(
                voxel_keys(np.concatenate(changed_points, axis=0))
            )
            region = dilate_voxels(changed_surface)
            arrays[f"{event_id}.region"] = array_from_keys(
                region, label=f"{event_id}.region"
            )

            candidate_background = tuple(sorted(background_keys & set(region)))
            if not candidate_background:
                raise ValueError(f"target array is empty: {event_id}.background_candidates")
            candidate_points = centers(candidate_background)
            removed_world = np.concatenate(removed_points, axis=0)
            pre_indices = range(max(0, intervention - window_frames), intervention)
            post_indices = range(intervention, intervention + window_frames)
            shadow: set[Voxel] = set()
            pre_observed: set[Voxel] = set()
            post_observed: set[Voxel] = set()
            for index in pre_indices:
                if index not in frame_map:
                    raise ValueError(f"{event_id} is missing pre-event frame {index}")
                frame = frame_map[index]
                world_from_camera = _homogeneous_transform(
                    np.asarray(frame["world_from_camera"]), name="world_from_camera"
                )
                shadow.update(
                    occlusion_shadow_voxels(
                        removed_world,
                        candidate_points,
                        camera_from_world=np.linalg.inv(world_from_camera),
                        intrinsics=intrinsics,
                        image_size=image_size,
                    )
                )
                pre_observed.update(
                    frame_observations(candidate_background, frame, mode="observed")
                )
            for index in post_indices:
                if index not in frame_map:
                    raise ValueError(f"{event_id} is missing post-event frame {index}")
                post_observed.update(
                    frame_observations(
                        candidate_background, frame_map[index], mode="observed"
                    )
                )
            revealed = revealed_background_voxels(
                candidate_background,
                shadow,
                pre_observed,
                post_observed,
            )
            background_observable_by_event[event_id] = bool(revealed)
            arrays[f"{event_id}.revealed_background"] = array_from_keys(
                revealed,
                label=f"{event_id}.revealed_background",
                allow_empty=True,
            )

            confirmed_free: set[Voxel] = set()
            consumed_through = intervention - 1
            for checkpoint in event["common_checkpoint_frame_indices"]:
                checkpoint = int(checkpoint)
                for index in range(consumed_through + 1, checkpoint + 1):
                    if index not in frame_map:
                        raise ValueError(f"{event_id} is missing checkpoint frame {index}")
                    confirmed_free.update(
                        frame_observations(region, frame_map[index], mode="free")
                    )
                consumed_through = checkpoint
                name = f"{event_id}.confirmed_free.{checkpoint:06d}"
                arrays[name] = array_from_keys(
                    confirmed_free, label=name, allow_empty=True
                )
            derived_events += 1
            print(
                "COMMON_V2_PROGRESS event "
                f"{event_id} targets_complete "
                f"background_observable={str(bool(revealed)).lower()}",
                flush=True,
            )

        checkpoint_indices = sorted(
            {
                int(checkpoint)
                for event in events
                for checkpoint in event["common_checkpoint_frame_indices"]
            }
        )
        observed_surface: set[Voxel] = set()
        geometry_cache: dict[int, np.ndarray] = {}
        next_frame = 0
        for checkpoint in checkpoint_indices:
            for index in range(next_frame, checkpoint + 1):
                if index not in frame_map:
                    raise ValueError(f"{scene} is missing semantic frame {index}")
                frame = frame_map[index]
                depth = np.asarray(frame["depth"], dtype=np.float64)
                if depth.shape != image_size:
                    raise ValueError(
                        "official depth frame shape disagrees with camera manifest"
                    )
                valid = np.isfinite(depth) & (depth > 0)
                rows, columns = np.nonzero(valid)
                measured = depth[rows, columns]
                fx, fy, cx, cy = intrinsics
                camera_points = np.column_stack(
                    (
                        (columns - cx) * measured / fx,
                        (rows - cy) * measured / fy,
                        measured,
                    )
                )
                world_from_camera = _homogeneous_transform(
                    np.asarray(frame["world_from_camera"]),
                    name="world_from_camera",
                )
                world_points = (
                    camera_points @ world_from_camera[:3, :3].T
                    + world_from_camera[:3, 3]
                )
                observed_surface.update(voxel_keys(world_points))
            next_frame = checkpoint + 1

            frame = frame_map[checkpoint]
            relative_timestamp = int(
                frame.get("relative_timestamp_ns", checkpoint)
            )
            active_records = active_dsg_records(
                scene_contract["dsg_records"],
                timestamp_ns=relative_timestamp,
            )
            visible_labeled_points: list[tuple[np.ndarray, int]] = []
            if observed_surface and active_records:
                from scipy.spatial import cKDTree

                observed_centers = centers(observed_surface)
                observed_tree = cKDTree(observed_centers)
                for record in active_records:
                    attributes = _attributes(record)
                    active_intervals = [
                        interval_index
                        for interval_index, (first, last) in enumerate(
                            _intervals(attributes)
                        )
                        if first <= relative_timestamp < last
                    ]
                    if len(active_intervals) != 1:
                        raise ValueError(
                            f"{scene} active DSG object has no unique interval"
                        )
                    matched_record = {
                        "symbol": str(attributes.get("name", "")),
                        "record": record,
                        "interval_index": active_intervals[0],
                    }
                    cache_key = id(record)
                    if cache_key not in geometry_cache:
                        geometry_cache[cache_key] = dsg_interval_world_points(
                            matched_record
                        )
                    world_geometry = geometry_cache[cache_key]
                    surface = voxel_keys(world_geometry)
                    if not surface:
                        continue
                    surface_centers = centers(surface)
                    distances, _ = observed_tree.query(surface_centers, k=1)
                    visible_voxels = {
                        voxel
                        for voxel, distance in zip(surface, distances)
                        if float(distance) <= 0.05
                    }
                    if not visible_voxels:
                        continue
                    integer = np.floor(world_geometry / 0.05).astype(np.int64)
                    mask = np.asarray(
                        [
                            tuple(int(value) for value in row) in visible_voxels
                            for row in integer
                        ],
                        dtype=bool,
                    )
                    visible_labeled_points.append(
                        (world_geometry[mask], int(attributes["semantic_label"]))
                    )
            semantic_name = f"{scene}.current_semantic.{checkpoint:06d}"
            semantic_labels = semantic_voxel_labels(visible_labeled_points)
            if semantic_labels:
                arrays[semantic_name] = np.asarray(
                    semantic_labels, dtype=np.int64
                )
            else:
                arrays[semantic_name] = np.empty((0, 4), dtype=np.int64)
        print(
            "COMMON_V2_PROGRESS scene "
            f"{scene} semantic_complete checkpoints={len(checkpoint_indices)}",
            flush=True,
        )

    for scene in selected_scenes:
        if not any(
            observable
            for event_id, observable in background_observable_by_event.items()
            if event_id.startswith(f"{scene}_event_")
        ):
            raise ValueError(
                f"{scene} must contain at least one background-observable event"
            )
    observable_count = sum(background_observable_by_event.values())
    metadata = {
        "event_object_count": int(contract["event_object_count"]),
        "derived_event_count": derived_events,
        "window_frames": window_frames,
        "event_limit": event_limit,
        "scenes": list(selected_scenes),
        "voxel_size_m": 0.05,
        "prediction_inputs_used": False,
        "background_observable_by_event": dict(
            sorted(background_observable_by_event.items())
        ),
        "background_observable_event_count": observable_count,
        "unobservable_revealed_target_event_count": (
            len(background_observable_by_event) - observable_count
        ),
    }
    return arrays, metadata


def _load_fixture_bundle(
    path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    if not path.is_file():
        raise ValueError(f"source is not a file: {path}")
    frames: dict[str, list[dict[str, Any]]] = {}
    backgrounds: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as bundle:
        for scene in ("apartment", "office"):
            required = {
                f"{scene}.depth",
                f"{scene}.world_from_camera",
                f"{scene}.background_points",
            }
            if not required <= set(bundle.files):
                raise ValueError(f"fixture bundle is missing {scene} arrays")
            depths = np.asarray(bundle[f"{scene}.depth"])
            poses = np.asarray(bundle[f"{scene}.world_from_camera"])
            if depths.ndim != 3 or poses.shape != (len(depths), 4, 4):
                raise ValueError(f"fixture bundle has invalid {scene} frame arrays")
            frames[scene] = [
                {
                    "frame_index": index,
                    "depth": depths[index],
                    "world_from_camera": poses[index],
                }
                for index in range(len(depths))
            ]
            backgrounds[scene] = _points3(
                bundle[f"{scene}.background_points"],
                name=f"{scene} fixture background points",
            )
    return frames, backgrounds


def _load_binary_ply_xyz(path: Path) -> np.ndarray:
    scalar_types = {
        "char": "i1",
        "uchar": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "float": "<f4",
        "double": "<f8",
    }
    with path.open("rb") as handle:
        if handle.readline() != b"ply\n":
            raise ValueError(f"background is not a PLY file: {path}")
        if handle.readline().decode("ascii").strip() != "format binary_little_endian 1.0":
            raise ValueError("background PLY must be binary_little_endian 1.0")
        vertex_count: int | None = None
        current_element = ""
        properties: list[tuple[str, str]] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("background PLY header is truncated")
            text = line.decode("ascii").strip()
            if text == "end_header":
                break
            fields = text.split()
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                current_element = "vertex"
            elif fields and fields[0] == "element":
                current_element = fields[1]
            elif fields and fields[0] == "property" and current_element == "vertex":
                if fields[1] == "list" or fields[1] not in scalar_types:
                    raise ValueError("background PLY has unsupported vertex properties")
                properties.append((fields[2], scalar_types[fields[1]]))
        if vertex_count is None or not {"x", "y", "z"} <= {
            name for name, _ in properties
        }:
            raise ValueError("background PLY has no XYZ vertex element")
        vertices = np.fromfile(handle, dtype=np.dtype(properties), count=vertex_count)
    if len(vertices) != vertex_count:
        raise ValueError("background PLY vertex payload is truncated")
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"]))


def _load_official_frames(
    contract: Mapping[str, Any], scene: str, *, maximum_frame_index: int
) -> list[dict[str, Any]]:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise RuntimeError(
            "real target generation requires the rosbags Python package"
        ) from error
    from scripts.evaluation.export_tesse_cd_rgbd import (
        _decode_image,
        _load_pose_inputs,
        camera_pose_matrix,
    )

    scene_contract = contract["scenes"][scene]
    bag = Path(scene_contract["sequence"]["bag"]["directory"])
    poses, body_from_camera = _load_pose_inputs(bag)
    depth_topic = str(contract["source_payload"]["topics"]["depth"])
    frames: list[dict[str, Any]] = []
    with AnyReader([bag]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic == depth_topic
        ]
        if len(connections) != 1:
            raise ValueError(f"{scene} bag must contain exactly one depth connection")
        for _, timestamp, raw in reader.messages(connections=connections):
            index = len(frames)
            if index > maximum_frame_index:
                break
            connection = connections[0]
            message = reader.deserialize(raw, connection.msgtype)
            if message.encoding.upper() != "32FC1":
                raise ValueError(f"unexpected TESSE-CD depth encoding: {message.encoding}")
            timestamp = int(timestamp)
            if timestamp not in poses:
                raise ValueError(f"TESSE-CD depth frame has no exact odometry: {timestamp}")
            frames.append(
                {
                    "frame_index": index,
                    "timestamp_ns": timestamp,
                    "depth": _decode_image(
                        message, dtype=np.dtype("<f4"), channels=1
                    ),
                    "world_from_camera": camera_pose_matrix(
                        poses[timestamp], body_from_camera
                    ),
                }
            )
    if len(frames) <= maximum_frame_index:
        raise ValueError(
            f"{scene} bag ended before required frame {maximum_frame_index}"
        )
    return frames


def load_derived_rgbd_frames(
    contract: Mapping[str, Any],
    scene: str,
    derived_root: Path,
    rgbd_lock: Path,
    *,
    maximum_frame_index: int,
) -> list[dict[str, Any]]:
    from PIL import Image
    from scripts.evaluation.export_tesse_cd_rgbd import (
        _read_bound_file,
        compute_export_output_binding,
    )

    if scene not in {"apartment", "office"}:
        raise ValueError(f"unknown TESSE-CD scene: {scene}")
    if type(maximum_frame_index) is not int or maximum_frame_index < 0:
        raise ValueError("maximum_frame_index must be non-negative")
    if not rgbd_lock.is_file():
        raise ValueError(f"checked RGB-D lock is not a file: {rgbd_lock}")
    try:
        lock = json.loads(rgbd_lock.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"checked RGB-D lock is not readable: {rgbd_lock}") from error
    if not isinstance(lock, Mapping):
        raise ValueError("checked RGB-D lock must contain a JSON object")
    if lock.get("source_role") != "official_rgbd_export":
        raise ValueError(
            f"unsupported derived RGB-D source role: {lock.get('source_role')!r}"
        )
    if not (
        lock.get("schema_version") == 1
        and lock.get("manifest_id") == "tesse_cd_rgbd_v1"
        and lock.get("dataset") == "TESSE-CD"
    ):
        raise ValueError("checked RGB-D lock identity mismatch")

    def resolved_record_path(record: Mapping[str, Any]) -> Path:
        candidate = Path(str(record.get("path", "")))
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        return candidate.resolve()

    source_record = lock.get("source_manifest")
    expected_source_record = contract["source_manifest"]
    if not isinstance(source_record, Mapping) or not (
        resolved_record_path(source_record)
        == resolved_record_path(expected_source_record)
        and source_record.get("sha256") == expected_source_record["sha256"]
        and source_record.get("byte_count") == expected_source_record["byte_count"]
    ):
        raise ValueError("checked RGB-D lock source manifest mismatch")
    locked_root = Path(str(lock.get("derived_root", "")))
    if not locked_root.is_absolute():
        locked_root = REPO_ROOT / locked_root
    if locked_root.resolve() != derived_root.resolve():
        raise ValueError("checked RGB-D lock derived root mismatch")

    scene_root = derived_root / scene
    export_path = scene_root / "export_manifest.json"
    lock_scenes = lock.get("scenes")
    locked_scene = lock_scenes.get(scene) if isinstance(lock_scenes, Mapping) else None
    if not isinstance(locked_scene, Mapping):
        raise ValueError(f"checked RGB-D lock has no {scene} scene binding")
    export_record = locked_scene.get("export_manifest")
    if not isinstance(export_record, Mapping) or (
        resolved_record_path(export_record) != export_path.resolve()
    ):
        raise ValueError(f"{scene} checked RGB-D export manifest path mismatch")
    if not export_path.is_file():
        raise ValueError(f"source is not a file: {export_path}")
    export_bytes = _read_bound_file(export_path)
    if (
        export_record.get("sha256") != hashlib.sha256(export_bytes).hexdigest()
        or export_record.get("byte_count") != len(export_bytes)
    ):
        raise ValueError(f"{scene} derived RGB-D export manifest SHA256 mismatch")
    try:
        export = json.loads(export_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{scene} derived RGB-D export manifest is invalid") from error

    sequence = contract["scenes"][scene]["sequence"]
    database = sequence["bag"]["database"]
    expected_frame_count = int(sequence["timeline"]["depth_frame_count"])

    def parse_bound_file(relative: str, content: bytes) -> object:
        if relative == f"{scene}/timestamps.csv":
            text = content.decode("utf-8")
            return list(csv.DictReader(io.StringIO(text, newline="")))
        if relative == f"{scene}/traj.txt":
            return [
                line.strip()
                for line in content.decode("utf-8").splitlines()
                if line.strip()
            ]
        if relative == "cam_params.json":
            return json.loads(content.decode("utf-8"))
        prefix = f"{scene}/results/"
        if not relative.startswith(prefix):
            return None
        name = relative.removeprefix(prefix)
        index = int(name[5:11])
        if index > maximum_frame_index:
            return None
        with Image.open(io.BytesIO(content)) as image:
            if name.startswith("frame"):
                rgb = np.asarray(image.convert("RGB")).copy()
                if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                    raise ValueError(f"{scene} RGB JPEG must decode as uint8 RGB")
                return None
            return np.asarray(image).copy()

    observed_output_binding = compute_export_output_binding(
        derived_root,
        scene,
        expected_frame_count,
        file_parser=parse_bound_file,
    )
    locked_database_sha256 = locked_scene.get("source_database_sha256")
    locked_combined_sha256 = locked_scene.get("combined_output_sha256")
    locked_file_count = locked_scene.get("file_hash_count")
    if (
        locked_combined_sha256
        != observed_output_binding["combined_output_sha256"]
        or locked_file_count != observed_output_binding["file_hash_count"]
    ):
        raise ValueError(f"{scene} derived RGB-D combined output SHA256 mismatch")
    parsed_files = observed_output_binding["parsed_files"]
    parse_errors = [
        error for error in parsed_files.values() if isinstance(error, Exception)
    ]
    if parse_errors:
        raise ValueError(
            f"{scene} derived RGB-D artifact cannot be parsed"
        ) from parse_errors[0]
    if not (
        export.get("schema_version") == 1
        and export.get("dataset") == "TESSE-CD"
        and export.get("scene") == scene
        and export.get("frame_count") == expected_frame_count
        and Path(str(export.get("source_database", ""))).resolve()
        == Path(str(database["path"])).resolve()
        and export.get("source_database_sha256") == database["sha256"]
        and locked_database_sha256 == database["sha256"]
        and export.get("depth_encoding")
        == "uint16 millimeters decoded from official 32FC1 meters"
        and export.get("rgb_encoding")
        == "JPEG quality 95 decoded from official rgb8"
        and export.get("combined_output_sha256")
        == locked_combined_sha256
        and export.get("file_hash_count")
        == locked_file_count
    ):
        raise ValueError(f"{scene} derived RGB-D export is not source-bound")

    timestamp_rows = parsed_files[f"{scene}/timestamps.csv"]
    trajectory_rows = parsed_files[f"{scene}/traj.txt"]
    camera_payload = parsed_files["cam_params.json"]
    if not isinstance(camera_payload, Mapping) or not isinstance(
        camera_payload.get("camera"), Mapping
    ):
        raise ValueError("derived RGB-D camera parameters are invalid")
    observed_camera = camera_payload["camera"]
    expected_camera = contract["source_payload"]["camera"]
    if not (
        int(observed_camera.get("w", observed_camera.get("width", -1)))
        == int(expected_camera["width"])
        and int(observed_camera.get("h", observed_camera.get("height", -1)))
        == int(expected_camera["height"])
        and all(
            float(observed_camera.get(name, float("nan")))
            == float(expected_camera[name])
            for name in ("fx", "fy", "cx", "cy")
        )
    ):
        raise ValueError("derived RGB-D camera parameters disagree")
    if (
        len(timestamp_rows) != expected_frame_count
        or len(trajectory_rows) != expected_frame_count
    ):
        raise ValueError(f"{scene} derived RGB-D frame metadata is incomplete")
    observed_indices = [int(row["frame_index"]) for row in timestamp_rows]
    if observed_indices != list(range(expected_frame_count)):
        raise ValueError(f"{scene} derived RGB-D frame indices are not contiguous")
    first_timestamp = int(sequence["timeline"]["first_depth_timestamp_ns"])
    last_timestamp = int(sequence["timeline"]["last_depth_timestamp_ns"])
    if (
        int(timestamp_rows[0]["sensor_timestamp_ns"]) != first_timestamp
        or int(timestamp_rows[-1]["sensor_timestamp_ns"]) != last_timestamp
    ):
        raise ValueError(f"{scene} derived RGB-D timestamp bounds disagree")
    if maximum_frame_index >= expected_frame_count:
        raise ValueError(f"{scene} derived RGB-D prefix exceeds frame count")

    frames: list[dict[str, Any]] = []
    for index in range(maximum_frame_index + 1):
        depth_mm = parsed_files[f"{scene}/results/depth{index:06d}.png"]
        if depth_mm.ndim != 2 or not np.issubdtype(depth_mm.dtype, np.integer):
            raise ValueError(f"{scene} depth PNG must be a single-channel integer image")
        pose_values = np.fromstring(trajectory_rows[index], sep=" ", dtype=np.float64)
        if pose_values.size != 16:
            raise ValueError(f"{scene} trajectory row must contain 16 values")
        pose = _homogeneous_transform(
            pose_values.reshape(4, 4), name="derived world_from_camera"
        )
        timestamp = int(timestamp_rows[index]["sensor_timestamp_ns"])
        relative = int(timestamp_rows[index]["relative_timestamp_ns"])
        if relative != timestamp - first_timestamp:
            raise ValueError(f"{scene} derived relative timestamp disagrees")
        frames.append(
            {
                "frame_index": index,
                "timestamp_ns": timestamp,
                "relative_timestamp_ns": relative,
                "depth": depth_mm.astype(np.float32) / 1000.0,
                "world_from_camera": pose,
            }
        )
    return frames


def run_generation(
    source_manifest: Path,
    schedule: Path,
    output_dir: Path,
    *,
    window_frames: int = 450,
    event_limit: int | None = None,
    fixture_bundle: Path | None = None,
    derived_rgbd_root: Path | None = None,
    derived_rgbd_lock: Path | None = None,
    scenes: Sequence[str] = ("apartment", "office"),
) -> Path:
    contract = load_generation_contract(source_manifest, schedule)
    selected_scenes = tuple(scenes)
    if fixture_bundle is not None:
        if derived_rgbd_root is not None or derived_rgbd_lock is not None:
            raise ValueError(
                "fixture bundle and derived RGB-D root/lock are mutually exclusive"
            )
        frames, backgrounds = _load_fixture_bundle(fixture_bundle)
        status = "FIXTURE"
        source_paths = (source_manifest, schedule, fixture_bundle)
    else:
        frames = {}
        backgrounds = {}
        if derived_rgbd_root is None:
            ground_truth_root = contract["source_payload"].get("ground_truth_root")
            if ground_truth_root:
                candidate = Path(str(ground_truth_root)).parent / "derived" / "rgbd_v1"
                if candidate.is_dir():
                    derived_rgbd_root = candidate
        if derived_rgbd_root is None and derived_rgbd_lock is not None:
            raise ValueError("derived RGB-D lock requires a derived RGB-D root")
        if derived_rgbd_root is not None and derived_rgbd_lock is None:
            if source_manifest.resolve() != DEFAULT_SOURCE_MANIFEST.resolve():
                raise ValueError(
                    "derived RGB-D lock is required for non-repository sources"
                )
            derived_rgbd_lock = DEFAULT_DERIVED_RGBD_LOCK
        derived_sources: list[Path] = []
        for scene in selected_scenes:
            events = contract["scenes"][scene]["schedule"]["events"]
            if event_limit is not None:
                events = events[:event_limit]
            maximum = max(
                max(int(value) for value in event["common_checkpoint_frame_indices"])
                for event in events
            )
            maximum = max(
                maximum,
                max(int(event["intervention_frame_index"]) for event in events)
                + window_frames
                - 1,
            )
            if derived_rgbd_root is not None:
                frames[scene] = load_derived_rgbd_frames(
                    contract,
                    scene,
                    derived_rgbd_root,
                    derived_rgbd_lock,
                    maximum_frame_index=maximum,
                )
                derived_sources.extend(
                    [
                        derived_rgbd_root / scene / "export_manifest.json",
                        derived_rgbd_root / scene / "timestamps.csv",
                        derived_rgbd_root / scene / "traj.txt",
                    ]
                )
            else:
                frames[scene] = _load_official_frames(
                    contract, scene, maximum_frame_index=maximum
                )
            backgrounds[scene] = _load_binary_ply_xyz(
                contract["scenes"][scene]["paths"]["background_mesh"]
            )
        status = (
            "GENERATED"
            if window_frames == 450
            and event_limit is None
            and set(selected_scenes) == {"apartment", "office"}
            else "SMOKE"
        )
        if derived_rgbd_lock is not None:
            derived_sources.insert(0, derived_rgbd_lock)
        source_paths = (source_manifest, schedule, *derived_sources)
    arrays, metadata = derive_target_arrays(
        contract,
        frames_by_scene=frames,
        background_points_by_scene=backgrounds,
        window_frames=window_frames,
        event_limit=event_limit,
        scenes=selected_scenes,
    )
    metadata.update(
        {
            "protocol_complete": status == "GENERATED",
            "source_manifest": contract["source_manifest"],
            "schedule": contract["schedule"],
            "declared_source_records": contract["source_records"],
        }
    )
    return write_target_package(
        output_dir,
        arrays=arrays,
        source_paths=source_paths,
        metadata=metadata,
        status=status,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=450)
    parser.add_argument("--event-limit", type=int)
    parser.add_argument("--fixture-bundle", type=Path)
    parser.add_argument("--derived-rgbd-root", type=Path)
    parser.add_argument("--derived-rgbd-lock", type=Path)
    parser.add_argument(
        "--scene", action="append", choices=("apartment", "office")
    )
    args = parser.parse_args(argv)
    run_generation(
        args.source_manifest,
        args.schedule,
        args.output_dir,
        window_frames=args.window_frames,
        event_limit=args.event_limit,
        fixture_bundle=args.fixture_bundle,
        derived_rgbd_root=args.derived_rgbd_root,
        derived_rgbd_lock=args.derived_rgbd_lock,
        scenes=tuple(args.scene or ("apartment", "office")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
