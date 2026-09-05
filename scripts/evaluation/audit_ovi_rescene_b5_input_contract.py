#!/usr/bin/env python3
"""Audit native ReScene preprocessing and frozen OVI token compatibility."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.audit_persist4d_rescene_checkpoint import (
    load_checkpoint_provenance,
    verify_bound_git_source,
)


class InputContractError(ValueError):
    """Raised when an input cannot support a deterministic compatibility audit."""


def _array(value: object, *, label: str, columns: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(value)
    except Exception as error:
        raise InputContractError(f"{label} is not array-like") from error
    if result.ndim != 2 or (columns is not None and result.shape[1] != columns):
        raise InputContractError(f"{label} has an invalid shape")
    if not np.issubdtype(result.dtype, np.number) or not np.all(np.isfinite(result)):
        raise InputContractError(f"{label} must be finite numeric data")
    return result


def _range(array: np.ndarray) -> list[float]:
    return [float(np.min(array)), float(np.max(array))]


def summarize_native_preprocessed_sample(
    *,
    raw_coordinates: object,
    dataset_features: object,
    original_colors: object,
    original_normals: object,
    processed: Mapping[str, object],
    sequence_name: str,
) -> dict[str, object]:
    """Summarize the exact arrays emitted by the pinned Pointcept collator."""

    raw_coord = _array(raw_coordinates, label="raw coordinates", columns=4)
    dataset_feat = _array(dataset_features, label="dataset features")
    colors = _array(original_colors, label="original colors", columns=3)
    normals = _array(original_normals, label="original normals", columns=3)
    if not isinstance(processed, Mapping):
        raise InputContractError("processed sample must be a mapping")
    count = len(raw_coord)
    if len(dataset_feat) != count or len(colors) != count or len(normals) != count:
        raise InputContractError("native raw sample row counts differ")
    if not isinstance(sequence_name, str) or not sequence_name:
        raise InputContractError("native sequence name is invalid")

    coord = _array(processed.get("coord"), label="processed coord", columns=5)
    grid = _array(processed.get("grid_coord"), label="grid coord", columns=3)
    model_color = _array(processed.get("color"), label="model color", columns=3)
    feature = _array(processed.get("feat"), label="model feature", columns=9)
    processed_count = len(coord)
    if len(grid) != processed_count or len(model_color) != processed_count or len(
        feature
    ) != processed_count:
        raise InputContractError("processed native sample row counts differ")
    batch_idx = np.asarray(processed.get("batch_idx"))
    temporal = np.asarray(processed.get("t"))
    offsets = np.asarray(processed.get("offset"))
    if (
        batch_idx.shape != (processed_count,)
        or temporal.shape != (processed_count,)
        or offsets.ndim != 1
        or len(offsets) != 2
        or int(offsets[-1]) != processed_count
        or int(offsets[0]) <= 0
        or int(offsets[0]) >= processed_count
    ):
        raise InputContractError("processed batch/temporal offsets are invalid")
    if sorted(np.unique(temporal).tolist()) != [0, 1]:
        raise InputContractError("native temporal values must be exactly 0 and 1")
    if np.unique(batch_idx).tolist() != [0] or np.unique(coord[:, 0]).tolist() != [0.0]:
        raise InputContractError("native sequence batch must remain exactly zero")
    if not np.allclose(feature[:, :3], coord[:, 1:4], rtol=0.0, atol=1e-6):
        raise InputContractError("model coordinate features do not match centered XYZ")
    if not np.allclose(feature[:, 3:6], model_color, rtol=0.0, atol=1e-6):
        raise InputContractError("model color features do not match collated camera RGB")
    if np.min(model_color) < 0.0 or np.max(model_color) > 1.0:
        raise InputContractError("model camera RGB must remain in [0, 1]")
    model_normals = feature[:, 6:9]
    lengths = np.linalg.norm(model_normals, axis=1)
    if np.any(lengths <= 0.0) or not np.allclose(lengths, 1.0, atol=1e-5):
        raise InputContractError("model normals must be finite unit vectors")

    per_stage = {
        str(int(stage)): int(np.count_nonzero(temporal == stage))
        for stage in sorted(np.unique(temporal).tolist())
    }
    return {
        "artifact_id": "PERSIST4D_NATIVE_T2_INPUT_WITNESS_V1",
        "center_shift_rule": "shared_xy_bbox_midpoint_and_zmin",
        "coord_as_feature_presence": True,
        "dataset_feature_shape": list(dataset_feat.shape),
        "dataset_normalized_color_discarded_by_pointcept_collator": True,
        "dataset_normalized_color_range": _range(dataset_feat[:, :3]),
        "grid_coordinate_shape": list(grid.shape),
        "model_color_range": _range(model_color),
        "model_feature_layout": [
            "shared_centered_xyz",
            "camera_rgb_0_1",
            "unit_source_geometry_normals",
        ],
        "model_feature_shape": list(feature.shape),
        "normal_presence": True,
        "original_camera_rgb_range": _range(colors),
        "per_stage_point_count": per_stage,
        "pointcept_stage_batch_values": [0, 1],
        "processed_coordinate_shape": list(coord.shape),
        "raw_coordinate_shape": list(raw_coord.shape),
        "sequence_batch_values": [0],
        "sequence_name": sequence_name,
        "status": "NATIVE_T2_INPUT_WITNESS_PASS",
        "temporal_stage_values": [0, 1],
    }


def _feature_reasons(
    metadata: Mapping[str, object],
    point_count: int,
) -> list[str]:
    reasons: list[str] = []
    rgb_source = metadata.get("point_rgb_source")
    if rgb_source == "instance_palette":
        reasons.append("INSTANCE_PALETTE_RGB_FORBIDDEN")
    if rgb_source != "camera_rgb":
        reasons.append("COLOR_NORMALIZATION_MISMATCH")
    else:
        rgb = np.asarray(metadata.get("point_rgb"))
        if (
            rgb.shape != (point_count, 3)
            or not np.issubdtype(rgb.dtype, np.integer)
            or np.issubdtype(rgb.dtype, np.bool_)
            or np.any(rgb < 0)
            or np.any(rgb > 255)
        ):
            reasons.append("COLOR_NORMALIZATION_MISMATCH")

    if metadata.get("point_normals_source") != "source_geometry":
        reasons.append("MISSING_SOURCE_GEOMETRY_NORMALS")
    else:
        normals = np.asarray(metadata.get("point_normals"), dtype=np.float64)
        if (
            normals.shape != (point_count, 3)
            or not np.all(np.isfinite(normals))
            or np.any(np.linalg.norm(normals, axis=1) <= 0.0)
        ):
            reasons.append("MISSING_SOURCE_GEOMETRY_NORMALS")
    return reasons


def _visit_token_audit(
    visit: Mapping[str, object],
    shared_center: np.ndarray,
    voxel_size_m: float,
) -> tuple[dict[str, object], list[str]]:
    visit_id = visit.get("visit_id")
    if type(visit_id) is not int or visit_id not in {0, 1}:
        raise InputContractError("visit_id must be exactly 0 or 1")
    points = _array(visit.get("points_xyz"), label=f"t{visit_id} points", columns=3)
    offsets = np.asarray(visit.get("entity_offsets"))
    metadata = visit.get("entity_metadata")
    if (
        offsets.ndim != 1
        or not np.issubdtype(offsets.dtype, np.integer)
        or len(offsets) < 2
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(points)
        or np.any(np.diff(offsets) < 0)
        or not isinstance(metadata, Sequence)
        or isinstance(metadata, (str, bytes))
        or len(metadata) != len(offsets) - 1
    ):
        raise InputContractError(f"t{visit_id} entity partition is invalid")

    feature_reasons: list[str] = []
    entity_voxels: list[np.ndarray] = []
    for entity_index, raw_metadata in enumerate(metadata):
        if not isinstance(raw_metadata, Mapping):
            raise InputContractError(f"t{visit_id} entity metadata is invalid")
        start, end = int(offsets[entity_index]), int(offsets[entity_index + 1])
        feature_reasons.extend(_feature_reasons(raw_metadata, end - start))
        if end == start:
            continue
        quantized = np.floor(
            (points[start:end].astype(np.float64) - shared_center) / voxel_size_m
        )
        if not np.all(np.isfinite(quantized)) or np.any(
            np.abs(quantized) > np.iinfo(np.int64).max
        ):
            raise InputContractError(f"t{visit_id} quantized coordinates overflow")
        entity_voxels.append(np.unique(quantized.astype(np.int64), axis=0))

    if not entity_voxels:
        raise InputContractError(f"t{visit_id} has no non-empty entity geometry")
    stacked = np.concatenate(entity_voxels, axis=0)
    _, multiplicity = np.unique(stacked, axis=0, return_counts=True)
    adapter_count = len(stacked)
    model_count = len(multiplicity)
    merge_count = adapter_count - model_count
    collision_count = int(np.count_nonzero(multiplicity > 1))
    result = {
        "adapter_token_count": adapter_count,
        "cross_entity_collision_voxel_count": collision_count,
        "entity_count": len(metadata),
        "maximum_entities_per_spatial_voxel": int(np.max(multiplicity)),
        "model_input_token_count": model_count,
        "permutation_count": adapter_count if merge_count == 0 else None,
        "permutation_possible": merge_count == 0,
        "point_count": len(points),
        "token_drop_count": 0,
        "token_duplication_count": 0,
        "token_merge_count": merge_count,
        "token_merge_fraction": float(merge_count / adapter_count),
        "visit_id": visit_id,
    }
    return result, feature_reasons


def audit_input_contract(
    native_witness: Mapping[str, object],
    visits: Sequence[Mapping[str, object]],
    voxel_size_m: float,
) -> dict[str, object]:
    """Compare native input semantics with entity-preserving OVI tokenization."""

    if not isinstance(native_witness, Mapping):
        raise InputContractError("native witness must be a mapping")
    if (
        native_witness.get("status") != "NATIVE_T2_INPUT_WITNESS_PASS"
        or native_witness.get("model_feature_layout")
        != [
            "shared_centered_xyz",
            "camera_rgb_0_1",
            "unit_source_geometry_normals",
        ]
        or native_witness.get("temporal_stage_values") != [0, 1]
        or native_witness.get("sequence_batch_values") != [0]
        or native_witness.get("pointcept_stage_batch_values") != [0, 1]
    ):
        raise InputContractError("native temporal/feature witness is incompatible")
    if isinstance(voxel_size_m, bool) or not isinstance(
        voxel_size_m, (int, float, np.number)
    ):
        raise InputContractError("neural voxel size is invalid")
    voxel = float(voxel_size_m)
    if not math.isfinite(voxel) or voxel != 0.02:
        raise InputContractError("neural voxel size must remain exactly 0.02 m")
    if (
        not isinstance(visits, Sequence)
        or isinstance(visits, (str, bytes))
        or len(visits) != 2
        or [visit.get("visit_id") for visit in visits] != [0, 1]
    ):
        raise InputContractError("input must contain ordered t0/t1 visits")

    point_arrays = [
        _array(visit.get("points_xyz"), label="visit points", columns=3)
        for visit in visits
    ]
    all_min = np.minimum(point_arrays[0].min(axis=0), point_arrays[1].min(axis=0))
    all_max = np.maximum(point_arrays[0].max(axis=0), point_arrays[1].max(axis=0))
    shared_center = np.asarray(
        [
            (all_min[0] + all_max[0]) / 2.0,
            (all_min[1] + all_max[1]) / 2.0,
            all_min[2],
        ],
        dtype=np.float64,
    )

    feature_reasons: list[str] = []
    visit_results: dict[str, object] = {}
    token_blocked = False
    for visit in visits:
        result, reasons = _visit_token_audit(visit, shared_center, voxel)
        visit_results[f"t{result['visit_id']}"] = result
        feature_reasons.extend(reasons)
        token_blocked = token_blocked or result["token_merge_count"] != 0
    ordered_feature_reasons = list(dict.fromkeys(feature_reasons))
    if ordered_feature_reasons:
        status = "BLOCKED_INPUT_FEATURE_CONTRACT"
        blocking_reasons = ordered_feature_reasons
        if token_blocked:
            blocking_reasons.append("BLOCKED_RESCENE_TOKEN_CONSERVATION")
    elif token_blocked:
        status = "BLOCKED_RESCENE_TOKEN_CONSERVATION"
        blocking_reasons = ["BLOCKED_RESCENE_TOKEN_CONSERVATION"]
    else:
        status = "INPUT_CONTRACT_PASS"
        blocking_reasons = []

    return {
        "artifact_id": "OVI_RESCENE_B5_INPUT_CONTRACT_V1",
        "blocking_reasons": blocking_reasons,
        "feature_schema": "rgb_normals",
        "gpu_authorized": status == "INPUT_CONTRACT_PASS",
        "native_witness_artifact_id": native_witness.get("artifact_id"),
        "neural_voxel_size_m": voxel,
        "schema_version": 1,
        "shared_center_xyz": shared_center.tolist(),
        "status": status,
        "visits": visit_results,
    }


def _stable_file_record(path: str | Path, *, label: str) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise InputContractError(f"{label} is missing") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise InputContractError(f"{label} must be a regular non-symlink file")
    digest = hashlib.sha256()
    byte_count = 0
    with absolute.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    after = absolute.stat(follow_symlinks=False)
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after) or byte_count != before.st_size:
        raise InputContractError(f"{label} changed while being read")
    return {
        "byte_count": byte_count,
        "path": str(absolute),
        "sha256": digest.hexdigest(),
    }


def _bound_path(
    record: object,
    *,
    label: str,
    base: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise InputContractError(f"{label} binding schema is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise InputContractError(f"{label} path is invalid")
    path = Path(raw_path)
    if not path.is_absolute():
        if base is None or ".." in path.parts:
            raise InputContractError(f"{label} relative path is invalid")
        path = base / path
    observed = _stable_file_record(path, label=label)
    expected = {
        "byte_count": record.get("byte_count"),
        "path": str(Path(raw_path) if Path(raw_path).is_absolute() else path.resolve()),
        "sha256": record.get("sha256"),
    }
    if observed != expected:
        raise InputContractError(f"{label} binding mismatch")
    return path.resolve(), observed


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InputContractError(f"{label} must be valid JSON") from error
    if not isinstance(payload, dict):
        raise InputContractError(f"{label} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _git_commit(checkout: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise InputContractError("producer Git commit is unavailable") from error


def _load_bound_module(path: Path, module_name: str) -> Any:
    if module_name in sys.modules:
        raise InputContractError("bound module name is already loaded")
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise InputContractError("bound source module cannot be loaded")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    sys.modules.pop(module_name, None)
    return module


def materialize_native_witness(
    *,
    checkpoint_config_path: Path,
    checkpoint_audit_path: Path,
    sequence_name: str,
    output_path: Path,
) -> dict[str, object]:
    """Run one CPU-native Persist4D dataset/collator sample and save compact stats."""

    provenance = load_checkpoint_provenance(checkpoint_config_path)
    sources = provenance["source_audit"]
    assert isinstance(sources, Mapping)
    persist4d = sources["persist4d"]
    assert isinstance(persist4d, Mapping)
    verify_bound_git_source(
        str(persist4d["checkout_path"]),
        str(persist4d["commit"]),
        persist4d["required_files"],
    )
    checkpoint_audit_record = _stable_file_record(
        checkpoint_audit_path, label="checkpoint audit"
    )
    checkpoint_audit = _load_json(checkpoint_audit_path, label="checkpoint audit")
    if checkpoint_audit.get("status") != "C0_C1_PASS":
        raise InputContractError("checkpoint C0/C1 audit has not passed")

    checkout = Path(str(persist4d["checkout_path"]))
    module_path = checkout / "scripts" / "evaluate_persist4d.py"
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(checkout))
    try:
        os.chdir(checkout)
        module = _load_bound_module(module_path, "_bound_persist4d_evaluate")
        config, _ = module._compose_runtime_config()
        import hydra
        from omegaconf import OmegaConf

        dataset_config = OmegaConf.create(
            OmegaConf.to_container(config.data.validation_dataset, resolve=True)
        )
        dataset_config.temporal_window = 2
        dataset = hydra.utils.instantiate(dataset_config)
        if sequence_name not in dataset.sequence_names:
            raise InputContractError("requested native T=2 sequence is unavailable")
        sequence_index = dataset.sequence_names.index(sequence_name)
        scan_indices = tuple(
            int(value) for value in dataset.sequence_indices[sequence_index]
        )
        sample = dataset.load_scan_indices(
            sequence_index,
            scan_indices,
            change_file=None,
        )
        data, _targets, names = hydra.utils.instantiate(
            config.data.validation_collation
        )([sample])
        if list(names) != [sequence_name]:
            raise InputContractError("native collator returned the wrong sequence")
        result = summarize_native_preprocessed_sample(
            raw_coordinates=sample[0],
            dataset_features=sample[1],
            original_colors=sample[4],
            original_normals=sample[5],
            processed=data,
            sequence_name=sequence_name,
        )
    finally:
        os.chdir(previous_cwd)
        if sys.path and sys.path[0] == str(checkout):
            sys.path.pop(0)

    result["checkpoint_audit"] = checkpoint_audit_record
    result["checkpoint_sha256"] = provenance["canonical_checkpoint"]["sha256"]
    result["persist4d_source_commit"] = persist4d["commit"]
    result["producer_command"] = [sys.executable, *sys.argv]
    result["producer_git_commit"] = _git_commit(Path(__file__).resolve().parents[2])
    result["training_config"] = _stable_file_record(
        provenance["training_config"]["path"], label="training config"
    )
    _write_json(output_path, result)
    return result


def _load_snapshot_visit(
    current_manifest_path: Path,
    current_manifest: Mapping[str, object],
    visit_id: int,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    artifacts = current_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise InputContractError("current-map artifacts are invalid")
    snapshot_path, snapshot_record = _bound_path(
        artifacts.get("snapshot"),
        label=f"B{visit_id * 2} snapshot",
        base=current_manifest_path.parent,
    )
    entities_path, entities_record = _bound_path(
        artifacts.get("entities"),
        label=f"B{visit_id * 2} entities",
        base=current_manifest_path.parent,
    )
    try:
        with np.load(snapshot_path, allow_pickle=False) as arrays:
            points = np.array(arrays["entity_points"], copy=True)
            offsets = np.array(arrays["entity_point_offsets"], copy=True)
    except (OSError, KeyError, ValueError) as error:
        raise InputContractError("current-map snapshot arrays are invalid") from error
    metadata: list[dict[str, object]] = []
    try:
        lines = entities_path.read_text(encoding="utf-8").splitlines()
        entities = [json.loads(line) for line in lines if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InputContractError("current-map entity records are invalid") from error
    if len(entities) != len(offsets) - 1:
        raise InputContractError("entity records and snapshot offsets differ")
    for index, entity in enumerate(entities):
        if (
            not isinstance(entity, Mapping)
            or entity.get("index") != index
            or entity.get("point_start") != int(offsets[index])
            or entity.get("point_count") != int(offsets[index + 1] - offsets[index])
            or not isinstance(entity.get("metadata"), Mapping)
        ):
            raise InputContractError("entity record partition is invalid")
        metadata.append(dict(entity["metadata"]))
    return (
        {
            "entity_metadata": metadata,
            "entity_offsets": offsets,
            "points_xyz": points,
            "visit_id": visit_id,
        },
        {"entities": entities_record, "snapshot": snapshot_record},
    )


def _native_mesh_evidence(native_manifest_path: Path) -> dict[str, object]:
    native = _load_json(native_manifest_path, label="native mapping manifest")
    artifacts = native.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise InputContractError("native mapping artifacts are invalid")
    mesh_path, mesh_record = _bound_path(
        artifacts.get("instance_mesh"), label="native instance mesh"
    )
    log_path, log_record = _bound_path(
        artifacts.get("instance_color_log"), label="native instance color log"
    )
    properties: list[str] = []
    with mesh_path.open("rb") as stream:
        for _ in range(128):
            line = stream.readline(4096)
            if not line:
                break
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise InputContractError("PLY header is not ASCII") from error
            if text.startswith("property "):
                properties.append(text.rsplit(" ", 1)[-1])
            if text == "end_header":
                break
        else:
            raise InputContractError("PLY header exceeds bounded audit limit")
    required = {
        "x",
        "y",
        "z",
        "normal_x",
        "normal_y",
        "normal_z",
        "red",
        "green",
        "blue",
        "alpha",
    }
    if not required.issubset(properties):
        raise InputContractError("PLY vertex property contract is incomplete")
    color_log = log_path.read_text(encoding="utf-8", errors="strict")
    palette_entries = len(
        re.findall(r"Instance:\s+\d+\s+Color:\s*\(\d+,\d+,\d+\)", color_log)
    )
    if palette_entries <= 0:
        raise InputContractError("native instance palette log has no entries")
    return {
        "instance_color_log": log_record,
        "instance_palette_entry_count": palette_entries,
        "instance_ply": mesh_record,
        "ply_rgb_semantics": "instance_palette",
        "ply_vertex_normals_present": True,
        "ply_vertex_properties": properties,
        "snapshot_loader_preserves_ply_normals": False,
    }


def audit_frozen_apartment_input_contract(
    *,
    frozen_config_path: Path,
    native_witness_path: Path,
    checkpoint_audit_path: Path,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, object]:
    """Audit C2 against the frozen Apartment B0/B2 graph without remapping OVI."""

    frozen_config_record = _stable_file_record(frozen_config_path, label="frozen config")
    frozen = _load_json(frozen_config_path, label="frozen config")
    apartment = frozen.get("frozen_inputs", {}).get("apartment")
    if not isinstance(apartment, Mapping):
        raise InputContractError("frozen Apartment inputs are unavailable")
    native_witness_record = _stable_file_record(native_witness_path, label="native witness")
    native_witness = _load_json(native_witness_path, label="native witness")
    checkpoint_audit_record = _stable_file_record(
        checkpoint_audit_path, label="checkpoint audit"
    )
    checkpoint_audit = _load_json(checkpoint_audit_path, label="checkpoint audit")
    if checkpoint_audit.get("status") != "C0_C1_PASS":
        raise InputContractError("checkpoint C0/C1 audit has not passed")

    two_visit_path, two_visit_record = _bound_path(
        apartment.get("two_visit_ovi_manifest"), label="two-visit OVI manifest"
    )
    two_visit = _load_json(two_visit_path, label="two-visit OVI manifest")
    raw_visits = two_visit.get("visits")
    if (
        two_visit.get("status") != "TWO_VISIT_OVI_PASS"
        or two_visit.get("scene") != "apartment"
        or not isinstance(raw_visits, Mapping)
        or set(raw_visits) != {"t0", "t1"}
        or raw_visits["t0"].get("source_frame_interval") != [766, 1021]
        or raw_visits["t1"].get("source_frame_interval") != [1217, 1472]
    ):
        raise InputContractError("two-visit Apartment identity is invalid")

    current_records: dict[str, dict[str, object]] = {}
    visit_inputs: list[dict[str, object]] = []
    native_mesh: dict[str, object] = {}
    for visit_id, variant in ((0, "b0"), (1, "b2")):
        manifest_path, manifest_record = _bound_path(
            apartment.get(f"{variant}_output_manifest"),
            label=f"{variant.upper()} output manifest",
        )
        manifest = _load_json(manifest_path, label=f"{variant.upper()} output manifest")
        if (
            manifest.get("status") != "PASS"
            or manifest.get("scene") != "apartment"
            or manifest.get("variant_id") != variant.upper()
            or manifest.get("source_visit_map_sha256")
            != [
                "cdd46a4a2b6ff82252dba22ae7b15bf3cc35ef46123a98649ba8c48f59cf33af",
                "aca00e36aa6925e8a24f0f365123cc7922742df2c682a5a3da0d96a22cde88b1",
            ]
        ):
            raise InputContractError(f"{variant.upper()} frozen identity is invalid")
        visit, artifacts = _load_snapshot_visit(manifest_path, manifest, visit_id)
        visit_inputs.append(visit)
        current_records[variant] = {
            "manifest": manifest_record,
            **artifacts,
        }
        native_record = raw_visits[f"t{visit_id}"].get("native_manifest")
        native_path, observed_native_record = _bound_path(
            native_record, label=f"t{visit_id} native manifest"
        )
        native_mesh[f"t{visit_id}"] = {
            "native_manifest": observed_native_record,
            **_native_mesh_evidence(native_path),
        }

    result = audit_input_contract(native_witness, visit_inputs, 0.02)
    result["checkpoint_audit"] = checkpoint_audit_record
    result["native_mesh_evidence"] = native_mesh
    result["native_witness"] = native_witness_record
    result["scene"] = "apartment"
    result["source_visit_map_sha256"] = [
        "cdd46a4a2b6ff82252dba22ae7b15bf3cc35ef46123a98649ba8c48f59cf33af",
        "aca00e36aa6925e8a24f0f365123cc7922742df2c682a5a3da0d96a22cde88b1",
    ]
    _write_json(output_path, result)

    receipt = {
        "artifact_id": "OVI_RESCENE_B5_APARTMENT_PAIR_RECEIPT_V1",
        "checkpoint_audit": checkpoint_audit_record,
        "frozen_config": frozen_config_record,
        "input_contract": _stable_file_record(output_path, label="input contract output"),
        "native_witness": native_witness_record,
        "producer_command": [sys.executable, *sys.argv],
        "producer_git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "scene": "apartment",
        "schema_version": 1,
        "source_artifacts": {
            "current_maps": current_records,
            "native_meshes": native_mesh,
            "two_visit_ovi_manifest": two_visit_record,
        },
        "status": result["status"],
    }
    _write_json(receipt_path, receipt)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    native = commands.add_parser("native-witness")
    native.add_argument("--checkpoint-config", type=Path, required=True)
    native.add_argument("--checkpoint-audit", type=Path, required=True)
    native.add_argument("--sequence-name", required=True)
    native.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--frozen-config", type=Path, required=True)
    audit.add_argument("--native-witness", type=Path, required=True)
    audit.add_argument("--checkpoint-audit", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "native-witness":
        materialize_native_witness(
            checkpoint_config_path=arguments.checkpoint_config,
            checkpoint_audit_path=arguments.checkpoint_audit,
            sequence_name=arguments.sequence_name,
            output_path=arguments.output,
        )
    else:
        audit_frozen_apartment_input_contract(
            frozen_config_path=arguments.frozen_config,
            native_witness_path=arguments.native_witness,
            checkpoint_audit_path=arguments.checkpoint_audit,
            output_path=arguments.output,
            receipt_path=arguments.receipt,
        )
    return 0


__all__ = [
    "InputContractError",
    "audit_frozen_apartment_input_contract",
    "audit_input_contract",
    "materialize_native_witness",
    "summarize_native_preprocessed_sample",
]


if __name__ == "__main__":
    raise SystemExit(main())
