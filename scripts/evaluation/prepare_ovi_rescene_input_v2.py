#!/usr/bin/env python3
"""Prepare source-bound OVI surface input for one frozen ReScene pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.data_structures import CameraIntrinsics, Frame
from src.evaluation.baselines.ovimap import parse_instance_color_log
from src.oviv2.ovi_rescene_adapter import (
    load_neural_sample_artifact,
    write_neural_sample_artifact,
)
from src.oviv2.ovi_surface_attributes import (
    SurfaceGroup,
    depth_millimeters_to_meters,
    load_ply_surface_attributes,
    project_world_points,
    sample_depth_consistent_rgb,
    validate_surface_group,
)
from src.oviv2.rescene_input_bridge import (
    AdapterGeometry,
    ModelCandidateMap,
    NativeSamplingMap,
    RecoveredModelSupport,
    SurfaceAttributeBundle,
    bind_surface_attributes,
    build_adapter_geometry,
    build_model_input,
    load_model_input_artifact,
    materialize_neural_sample_map,
    select_model_candidates,
    validate_native_sampling,
    write_model_input_artifact,
)
from src.oviv2.two_visit_contracts import (
    OviEntitySemanticEvidence,
    VisitMap,
    validate_visit_pair,
)

_STATIC_ARRAYS = {
    "coordinates_xyzt": ("geometry", "coordinates_xyzt"),
    "visit_ids": ("geometry", "visit_ids"),
    "source_visit_ids": ("geometry", "source_visit_ids"),
    "source_entity_indices": ("geometry", "source_entity_indices"),
    "source_entity_local_indices": ("geometry", "source_entity_local_indices"),
    "source_point_indices": ("geometry", "source_point_indices"),
    "source_to_adapter_offsets": ("geometry", "source_to_adapter_offsets"),
    "shared_center_xyz": ("geometry", "shared_center_xyz"),
    "source_points_xyz": ("surface", "points_xyz"),
    "normals_xyz": ("surface", "normals_xyz"),
    "normal_valid": ("surface", "normal_valid"),
    "original_vertex_indices": ("surface", "original_vertex_indices"),
    "surface_source_visit_ids": ("surface", "source_visit_ids"),
    "surface_source_entity_indices": ("surface", "source_entity_indices"),
    "adapter_to_model": ("sampling", "adapter_to_model"),
    "selected_adapter_indices": ("sampling", "selected_adapter_indices"),
    "model_grid_coordinates": ("sampling", "model_grid_coordinates"),
    "model_visit_ids": ("sampling", "model_visit_ids"),
    "visit_model_offsets": ("sampling", "visit_model_offsets"),
    "visit_grid_origins": ("sampling", "visit_grid_origins"),
    "candidate_source_point_indices": ("candidates", "source_point_indices"),
    "candidate_counts": ("candidates", "candidate_counts"),
}
_STATIC_MANIFEST_KEYS = {
    "schema_version",
    "artifact_id",
    "status",
    "arrays",
    "geometry",
    "surface",
    "sampling",
    "candidates",
    "input_bindings",
}
_RECOVERY_ARRAYS = (
    "support_valid",
    "representative_source_point_indices",
    "rgb_uint8",
    "local_frame_indices",
    "global_frame_indices",
    "rows",
    "columns",
    "camera_depth_m",
    "observed_depth_m",
    "depth_residual_m",
)


class C2PreparationError(ValueError):
    """Raised when the C2-V2 input cannot be closed from frozen sources."""


@dataclass(frozen=True, slots=True)
class DepthToleranceDecision:
    """Input-only depth calibration selected before result-bearing inference."""

    inlier_counts: tuple[int, int]
    visit_q99_m: tuple[float, float]
    visit_tolerances_m: tuple[float, float]
    selected_tolerance_m: float


@dataclass(frozen=True, slots=True)
class StaticInputArtifactPaths:
    root: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class StaticPreparedInput:
    geometry: AdapterGeometry
    surface: SurfaceAttributeBundle
    sampling: NativeSamplingMap
    candidates: ModelCandidateMap
    input_bindings: dict[str, dict[str, object]]


@dataclass(frozen=True, slots=True)
class MaterializedVisitFrames:
    """One hash-bound reindexed RGB-D window with global frame provenance."""

    manifest_path: Path
    root: Path
    scene: str
    visit_id: int
    global_frame_start: int
    global_frame_end: int
    source_input_sha256: str
    intrinsics: CameraIntrinsics
    camera_to_world: np.ndarray
    sensor_timestamps_ns: tuple[int, ...]
    rgb_paths: tuple[Path, ...]
    depth_paths: tuple[Path, ...]
    rgb_bindings: tuple[dict[str, object], ...]
    depth_bindings: tuple[dict[str, object], ...]

    @property
    def frame_count(self) -> int:
        return len(self.rgb_paths)

    @property
    def global_frame_indices(self) -> tuple[int, ...]:
        return tuple(range(self.global_frame_start, self.global_frame_end + 1))

    def read_frame(self, local_index: int) -> Frame:
        if type(local_index) is not int or not 0 <= local_index < self.frame_count:
            raise C2PreparationError("materialized local frame index is out of range")
        rgb_bytes = _read_bound_bytes(
            self.rgb_paths[local_index],
            self.rgb_bindings[local_index],
            label=f"visit {self.visit_id} RGB frame {local_index}",
        )
        depth_bytes = _read_bound_bytes(
            self.depth_paths[local_index],
            self.depth_bindings[local_index],
            label=f"visit {self.visit_id} depth frame {local_index}",
        )
        try:
            with Image.open(io.BytesIO(rgb_bytes)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            with Image.open(io.BytesIO(depth_bytes)) as image:
                depth_mm = np.asarray(image)
        except (OSError, ValueError) as error:
            raise C2PreparationError("materialized RGB-D frame cannot be decoded") from error
        expected_shape = (self.intrinsics.height, self.intrinsics.width)
        if rgb.shape != (*expected_shape, 3):
            raise C2PreparationError("materialized RGB shape disagrees with camera")
        if depth_mm.shape != expected_shape or depth_mm.dtype != np.uint16:
            raise C2PreparationError("materialized depth must be camera-aligned uint16")
        return Frame(
            frame_id=local_index,
            source_frame_id=self.global_frame_start + local_index,
            rgb=np.ascontiguousarray(rgb, dtype=np.uint8),
            depth=depth_millimeters_to_meters(depth_mm),
            pose=np.array(self.camera_to_world[local_index], copy=True),
            intrinsics=self.intrinsics,
            timestamp=self.sensor_timestamps_ns[local_index] / 1_000_000_000,
        )


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    status: str
    support: RecoveredModelSupport
    unsupported_model_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class FrozenStaticSources:
    source_config_path: Path
    protocol_path: Path
    two_visit_ovi_manifest_path: Path
    b0_output_manifest_record: dict[str, object]
    b2_output_manifest_record: dict[str, object]
    sampler_source_path: Path
    sampler_source_sha256: str
    input_bindings: dict[str, dict[str, object]]


@dataclass(frozen=True, slots=True)
class DepthCalibrationArtifactPaths:
    root: Path
    manifest: Path
    arrays: Path


@dataclass(frozen=True, slots=True)
class LoadedDepthCalibration:
    decision: DepthToleranceDecision
    calibration_indices: dict[int, np.ndarray]
    residuals_by_visit: dict[int, np.ndarray]
    manifest: dict[str, Any]


def _regular_file_sha256(path: Path, *, label: str) -> str:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise C2PreparationError(f"{label} is unavailable: {absolute}") from error
    if not stat.S_ISREG(before.st_mode):
        raise C2PreparationError(f"{label} must be a regular non-symlink file")
    digest = hashlib.sha256()
    byte_count = 0
    with absolute.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or byte_count != after.st_size:
        raise C2PreparationError(f"{label} changed while hashing")
    return digest.hexdigest()


def _absolute_file_record(path: Path, *, label: str) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    digest = _regular_file_sha256(absolute, label=label)
    return {
        "path": str(absolute),
        "sha256": digest,
        "byte_count": absolute.stat(follow_symlinks=False).st_size,
    }


def _read_bound_bytes(
    path: Path, record: Mapping[str, object], *, label: str
) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise C2PreparationError(f"{label} is unavailable: {absolute}") from error
    if not stat.S_ISREG(before.st_mode):
        raise C2PreparationError(f"{label} must be a regular non-symlink file")
    content = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(content) != after.st_size:
        raise C2PreparationError(f"{label} changed while reading")
    if (
        set(record) != {"path", "sha256", "byte_count"}
        or record.get("sha256") != hashlib.sha256(content).hexdigest()
        or record.get("byte_count") != len(content)
    ):
        raise C2PreparationError(f"{label} binding mismatch")
    return content


def _validated_input_bindings(
    bindings: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if not isinstance(bindings, Mapping) or not bindings:
        raise C2PreparationError("input bindings must be a non-empty mapping")
    result: dict[str, dict[str, object]] = {}
    for raw_name in sorted(bindings):
        if not isinstance(raw_name, str) or not raw_name:
            raise C2PreparationError("input binding names must be non-empty strings")
        record = bindings[raw_name]
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "byte_count",
        }:
            raise C2PreparationError(f"input binding schema is invalid: {raw_name}")
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise C2PreparationError(f"input binding path is invalid: {raw_name}")
        observed = _absolute_file_record(Path(raw_path), label=f"input {raw_name}")
        if dict(record) != observed:
            raise C2PreparationError(f"input binding mismatch: {raw_name}")
        result[raw_name] = observed
    return result


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _array_file_record(path: Path, value: np.ndarray) -> dict[str, object]:
    absolute = _absolute_file_record(path, label=f"static array {path.name}")
    return {
        "path": path.name,
        "sha256": absolute["sha256"],
        "byte_count": absolute["byte_count"],
        "dtype": np.asarray(value).dtype.str,
        "shape": list(np.asarray(value).shape),
    }


def _semantic_record(value: OviEntitySemanticEvidence) -> dict[str, object]:
    return {
        "visit_id": value.visit_id,
        "entity_id": value.entity_id,
        "semantic_label": value.semantic_label,
        "semantic_score": value.semantic_score,
        "semantic_embedding": (
            None
            if value.semantic_embedding is None
            else value.semantic_embedding.tolist()
        ),
    }


def _relative_file_record(path: Path, *, relative_path: str, label: str) -> dict[str, object]:
    absolute = _absolute_file_record(path, label=label)
    return {
        "path": relative_path,
        "sha256": absolute["sha256"],
        "byte_count": absolute["byte_count"],
    }


def publish_recovered_input(
    *,
    static_input_root: str | Path,
    calibration_manifest_path: str | Path,
    prepared: StaticPreparedInput,
    recovery: RecoveryResult,
    depth_tolerance_m: float,
    output_root: str | Path,
) -> Path:
    """Atomically publish complete executable input or explicit partial evidence."""

    if not isinstance(prepared, StaticPreparedInput):
        raise TypeError("prepared must be StaticPreparedInput")
    if not isinstance(recovery, RecoveryResult):
        raise TypeError("recovery must be RecoveryResult")
    tolerance = float(depth_tolerance_m)
    if not math.isfinite(tolerance) or not 0.0 < tolerance <= 0.08:
        raise C2PreparationError("depth tolerance must be in (0, 0.08]")
    static_root = Path(os.path.abspath(os.fspath(static_input_root)))
    rebound = load_static_input_artifact(static_root)
    static_hashes = (
        prepared.geometry.content_sha256(),
        prepared.surface.content_sha256(),
        prepared.sampling.content_sha256(),
        prepared.candidates.content_sha256(),
    )
    if static_hashes != (
        rebound.geometry.content_sha256(),
        rebound.surface.content_sha256(),
        rebound.sampling.content_sha256(),
        rebound.candidates.content_sha256(),
    ):
        raise C2PreparationError("prepared input differs from the bound static artifact")
    calibration_path = Path(
        os.path.abspath(os.fspath(calibration_manifest_path))
    )
    calibration_manifest = _load_json_object(
        calibration_path, label="depth calibration manifest"
    )
    if calibration_manifest.get("status") != "PASS":
        raise C2PreparationError("depth calibration status is not PASS")

    support = recovery.support
    model_count = prepared.sampling.model_count
    if len(support.support_valid) != model_count:
        raise C2PreparationError("recovered support model count is inconsistent")
    unsupported = np.flatnonzero(~support.support_valid).astype(np.int64)
    supplied_unsupported = np.asarray(recovery.unsupported_model_indices)
    if (
        supplied_unsupported.ndim != 1
        or not np.issubdtype(supplied_unsupported.dtype, np.integer)
        or not np.array_equal(supplied_unsupported.astype(np.int64), unsupported)
    ):
        raise C2PreparationError("unsupported model indices are inconsistent")
    expected_status = "C2_V2_PASS" if len(unsupported) == 0 else "PARTIAL_INPUT_SCOPE"
    if recovery.status != expected_status:
        raise C2PreparationError("recovery status disagrees with support coverage")

    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.exists() or output.is_symlink():
        raise C2PreparationError(f"recovered input artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        support_path = staging / "recovered_support.npz"
        with support_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                **{name: getattr(support, name) for name in _RECOVERY_ARRAYS},
                unsupported_model_indices=unsupported,
            )
            stream.flush()
            os.fsync(stream.fileno())

        model_manifest_record: dict[str, object] | None = None
        pair_manifest_record: dict[str, object] | None = None
        if expected_status == "C2_V2_PASS":
            model_input = build_model_input(
                prepared.geometry,
                prepared.sampling,
                prepared.surface,
                prepared.candidates,
                support,
            )
            model_paths = write_model_input_artifact(
                model_input, staging / "model_input"
            )
            pair = materialize_neural_sample_map(prepared.geometry, model_input)
            pair_paths = write_neural_sample_artifact(pair, staging / "adapter_pair")
            model_manifest_record = _relative_file_record(
                model_paths.manifest,
                relative_path="model_input/manifest.json",
                label="model input manifest",
            )
            pair_manifest_record = _relative_file_record(
                pair_paths.manifest,
                relative_path="adapter_pair/manifest.json",
                label="adapter pair manifest",
            )

        representatives = support.representative_source_point_indices[
            support.support_valid
        ]
        model_normal_count = int(
            np.count_nonzero(prepared.surface.normal_valid[representatives])
        )
        valid_count = int(np.count_nonzero(support.support_valid))
        receipt = {
            "schema_version": 2,
            "artifact_id": "OVI_RESCENE_C2_INPUT_CONTRACT_V2",
            "status": expected_status,
            "depth_tolerance_m": tolerance,
            "static_input_manifest": _absolute_file_record(
                static_root / "manifest.json", label="static input manifest"
            ),
            "calibration_manifest": _absolute_file_record(
                calibration_path, label="depth calibration manifest"
            ),
            "source_point_count": prepared.geometry.source_point_count,
            "adapter_count": prepared.geometry.adapter_count,
            "model_count": model_count,
            "unsupported_model_count": len(unsupported),
            "attribute_coverage": {
                "source_normal_fraction": float(
                    np.count_nonzero(prepared.surface.normal_valid)
                    / prepared.geometry.source_point_count
                ),
                "model_rgb_fraction": float(valid_count / model_count),
                "model_normal_fraction": float(model_normal_count / model_count),
            },
            "recovered_support": _relative_file_record(
                support_path,
                relative_path=support_path.name,
                label="recovered support arrays",
            ),
            "model_input_manifest": model_manifest_record,
            "adapter_pair_manifest": pair_manifest_record,
        }
        receipt_path = staging / "input_contract_v2.json"
        with receipt_path.open("xb") as stream:
            stream.write(_json_bytes(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "input_contract_v2.json"


def audit_recovered_input(
    *,
    output_root: str | Path,
    static_input_root: str | Path,
    calibration_manifest_path: str | Path,
) -> dict[str, Any]:
    """Revalidate a published C2 input without reopening source RGB-D frames."""

    output = Path(os.path.abspath(os.fspath(output_root)))
    receipt_path = output / "input_contract_v2.json"
    receipt = _load_json_object(receipt_path, label="C2 input contract")
    expected_keys = {
        "schema_version",
        "artifact_id",
        "status",
        "depth_tolerance_m",
        "static_input_manifest",
        "calibration_manifest",
        "source_point_count",
        "adapter_count",
        "model_count",
        "unsupported_model_count",
        "attribute_coverage",
        "recovered_support",
        "model_input_manifest",
        "adapter_pair_manifest",
    }
    status = receipt.get("status")
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version") != 2
        or receipt.get("artifact_id") != "OVI_RESCENE_C2_INPUT_CONTRACT_V2"
        or status not in {"C2_V2_PASS", "PARTIAL_INPUT_SCOPE"}
    ):
        raise C2PreparationError("C2 input contract identity is invalid")
    tolerance = receipt.get("depth_tolerance_m")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or not 0.0 < float(tolerance) <= 0.08
    ):
        raise C2PreparationError("C2 input contract depth tolerance is invalid")
    static_root = Path(os.path.abspath(os.fspath(static_input_root)))
    calibration_path = Path(
        os.path.abspath(os.fspath(calibration_manifest_path))
    )
    if receipt.get("static_input_manifest") != _absolute_file_record(
        static_root / "manifest.json", label="static input manifest"
    ):
        raise C2PreparationError("C2 input static binding mismatch")
    if receipt.get("calibration_manifest") != _absolute_file_record(
        calibration_path, label="depth calibration manifest"
    ):
        raise C2PreparationError("C2 input calibration binding mismatch")
    calibration = _load_json_object(calibration_path, label="depth calibration manifest")
    if calibration.get("status") != "PASS":
        raise C2PreparationError("depth calibration status is not PASS")
    prepared = load_static_input_artifact(static_root)
    expected_counts = {
        "source_point_count": prepared.geometry.source_point_count,
        "adapter_count": prepared.geometry.adapter_count,
        "model_count": prepared.sampling.model_count,
    }
    if any(receipt.get(name) != value for name, value in expected_counts.items()):
        raise C2PreparationError("C2 input domain counts are inconsistent")

    support_path = _bound_relative_path(
        output,
        receipt.get("recovered_support"),
        expected_path="recovered_support.npz",
        label="recovered support",
    )
    expected_array_keys = {*_RECOVERY_ARRAYS, "unsupported_model_indices"}
    try:
        with np.load(support_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_array_keys:
                raise C2PreparationError("recovered support array schema is invalid")
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, C2PreparationError):
            raise
        raise C2PreparationError("recovered support arrays cannot be decoded") from error
    support = RecoveredModelSupport(
        **{name: arrays[name] for name in _RECOVERY_ARRAYS}
    )
    unsupported = np.asarray(arrays["unsupported_model_indices"])
    expected_unsupported = np.flatnonzero(~support.support_valid).astype(np.int64)
    if (
        unsupported.ndim != 1
        or not np.issubdtype(unsupported.dtype, np.integer)
        or not np.array_equal(unsupported.astype(np.int64), expected_unsupported)
        or receipt.get("unsupported_model_count") != len(expected_unsupported)
    ):
        raise C2PreparationError("recovered unsupported indices are inconsistent")
    expected_status = (
        "C2_V2_PASS" if len(expected_unsupported) == 0 else "PARTIAL_INPUT_SCOPE"
    )
    if status != expected_status:
        raise C2PreparationError("C2 input status disagrees with recovered support")
    representatives = support.representative_source_point_indices[
        support.support_valid
    ]
    model_normal_count = int(
        np.count_nonzero(prepared.surface.normal_valid[representatives])
    )
    valid_count = int(np.count_nonzero(support.support_valid))
    expected_coverage = {
        "source_normal_fraction": float(
            np.count_nonzero(prepared.surface.normal_valid)
            / prepared.geometry.source_point_count
        ),
        "model_rgb_fraction": float(valid_count / prepared.sampling.model_count),
        "model_normal_fraction": float(
            model_normal_count / prepared.sampling.model_count
        ),
    }
    if receipt.get("attribute_coverage") != expected_coverage:
        raise C2PreparationError("C2 input attribute coverage is inconsistent")

    model_record = receipt.get("model_input_manifest")
    pair_record = receipt.get("adapter_pair_manifest")
    if status == "PARTIAL_INPUT_SCOPE":
        if (
            model_record is not None
            or pair_record is not None
            or (output / "model_input").exists()
            or (output / "adapter_pair").exists()
        ):
            raise C2PreparationError("partial C2 input contains executable artifacts")
        return receipt
    _bound_relative_path(
        output,
        model_record,
        expected_path="model_input/manifest.json",
        label="model input manifest",
    )
    _bound_relative_path(
        output,
        pair_record,
        expected_path="adapter_pair/manifest.json",
        label="adapter pair manifest",
    )
    model_input = load_model_input_artifact(output / "model_input")
    pair = load_neural_sample_artifact(output / "adapter_pair")
    if (
        model_input.adapter_geometry_sha256 != prepared.geometry.content_sha256()
        or model_input.native_sampling_sha256 != prepared.sampling.content_sha256()
        or model_input.surface_attributes_sha256 != prepared.surface.content_sha256()
        or model_input.model_candidates_sha256 != prepared.candidates.content_sha256()
        or not np.array_equal(
            model_input.adapter_to_model, prepared.sampling.adapter_to_model
        )
        or not np.array_equal(pair.coordinates_xyzt, prepared.geometry.coordinates_xyzt)
        or not np.array_equal(
            pair.features,
            model_input.features[:, 3:9][model_input.adapter_to_model],
        )
    ):
        raise C2PreparationError("executable C2 sidecars disagree with frozen domains")
    return receipt


def write_static_input_artifact(
    *,
    geometry: AdapterGeometry,
    surface: SurfaceAttributeBundle,
    sampling: NativeSamplingMap,
    candidates: ModelCandidateMap,
    input_bindings: Mapping[str, Mapping[str, object]],
    output_root: str | Path,
) -> StaticInputArtifactPaths:
    """Atomically cache D/A/M static domains without compressing large arrays."""

    if not isinstance(geometry, AdapterGeometry):
        raise TypeError("geometry must be AdapterGeometry")
    if not isinstance(surface, SurfaceAttributeBundle):
        raise TypeError("surface must be SurfaceAttributeBundle")
    if not isinstance(sampling, NativeSamplingMap):
        raise TypeError("sampling must be NativeSamplingMap")
    if not isinstance(candidates, ModelCandidateMap):
        raise TypeError("candidates must be ModelCandidateMap")
    if surface.adapter_geometry_sha256 != geometry.content_sha256():
        raise C2PreparationError("surface binds a different adapter geometry")
    normalized_bindings = _validated_input_bindings(input_bindings)
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.exists() or output.is_symlink():
        raise C2PreparationError(f"static input artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    values = {
        "geometry": geometry,
        "surface": surface,
        "sampling": sampling,
        "candidates": candidates,
    }
    try:
        array_records: dict[str, dict[str, object]] = {}
        for output_name, (owner, attribute) in _STATIC_ARRAYS.items():
            value = np.asarray(getattr(values[owner], attribute))
            path = staging / f"{output_name}.npy"
            with path.open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            array_records[output_name] = _array_file_record(path, value)
        manifest = {
            "schema_version": 2,
            "artifact_id": "OVI_RESCENE_STATIC_INPUT_V2",
            "status": "PASS",
            "arrays": array_records,
            "geometry": {
                "content_sha256": geometry.content_sha256(),
                "entity_keys": [list(value) for value in geometry.entity_keys],
                "entity_semantics": [
                    _semantic_record(value) for value in geometry.entity_semantics
                ],
                "neural_voxel_size_m": geometry.neural_voxel_size_m,
                "coordinate_frame_id": geometry.coordinate_frame_id,
                "source_manifest_sha256": geometry.source_manifest_sha256,
                "source_visit_map_sha256": list(geometry.source_visit_map_sha256),
            },
            "surface": {
                "content_sha256": surface.content_sha256(),
                "adapter_geometry_sha256": surface.adapter_geometry_sha256,
                "rgb_source": surface.rgb_source,
            },
            "sampling": {
                "content_sha256": sampling.content_sha256(),
                "voxel_size_m": sampling.voxel_size_m,
                "sampler_seed": sampling.sampler_seed,
                "sampler_mode": sampling.sampler_mode,
                "sampler_source_sha256": sampling.sampler_source_sha256,
            },
            "candidates": {
                "content_sha256": candidates.content_sha256(),
                "maximum_candidates": candidates.maximum_candidates,
            },
            "input_bindings": normalized_bindings,
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return StaticInputArtifactPaths(output, output / "manifest.json")


def _load_static_arrays(
    root: Path, records: object
) -> dict[str, np.ndarray]:
    if not isinstance(records, Mapping) or set(records) != set(_STATIC_ARRAYS):
        raise C2PreparationError("static array manifest schema is invalid")
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(_STATIC_ARRAYS):
        record = records[name]
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "byte_count",
            "dtype",
            "shape",
        }:
            raise C2PreparationError(f"static array record is invalid: {name}")
        if record.get("path") != f"{name}.npy":
            raise C2PreparationError(f"static array path is invalid: {name}")
        path = root / f"{name}.npy"
        observed = _absolute_file_record(path, label=f"static array {name}")
        if (
            observed["sha256"] != record.get("sha256")
            or observed["byte_count"] != record.get("byte_count")
        ):
            raise C2PreparationError(f"static array binding mismatch: {name}")
        try:
            value = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise C2PreparationError(f"static array cannot be decoded: {name}") from error
        if value.dtype.str != record.get("dtype") or list(value.shape) != record.get("shape"):
            raise C2PreparationError(f"static array dtype or shape mismatch: {name}")
        arrays[name] = value
    return arrays


def _required_mapping(value: object, *, label: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise C2PreparationError(f"{label} schema is invalid")
    return value


def load_static_input_artifact(output_root: str | Path) -> StaticPreparedInput:
    """Audit and load a cached static input while revalidating every parent."""

    root = Path(os.path.abspath(os.fspath(output_root)))
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise C2PreparationError("static input manifest is missing or a symlink")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise C2PreparationError("static input manifest is invalid JSON") from error
    if not isinstance(manifest, Mapping) or set(manifest) != _STATIC_MANIFEST_KEYS:
        raise C2PreparationError("static input manifest schema is invalid")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("artifact_id") != "OVI_RESCENE_STATIC_INPUT_V2"
        or manifest.get("status") != "PASS"
    ):
        raise C2PreparationError("static input manifest identity is invalid")
    input_bindings = _validated_input_bindings(manifest.get("input_bindings"))
    arrays = _load_static_arrays(root, manifest.get("arrays"))
    geometry_meta = _required_mapping(
        manifest.get("geometry"),
        label="geometry metadata",
        keys={
            "content_sha256",
            "entity_keys",
            "entity_semantics",
            "neural_voxel_size_m",
            "coordinate_frame_id",
            "source_manifest_sha256",
            "source_visit_map_sha256",
        },
    )
    try:
        semantics = tuple(
            OviEntitySemanticEvidence(
                visit_id=value["visit_id"],
                entity_id=value["entity_id"],
                semantic_label=value["semantic_label"],
                semantic_score=value["semantic_score"],
                semantic_embedding=value["semantic_embedding"],
            )
            for value in geometry_meta["entity_semantics"]
        )
        geometry = AdapterGeometry(
            coordinates_xyzt=arrays["coordinates_xyzt"],
            visit_ids=arrays["visit_ids"],
            source_visit_ids=arrays["source_visit_ids"],
            source_entity_indices=arrays["source_entity_indices"],
            source_entity_local_indices=arrays["source_entity_local_indices"],
            source_point_indices=arrays["source_point_indices"],
            source_to_adapter_offsets=arrays["source_to_adapter_offsets"],
            entity_keys=tuple(tuple(value) for value in geometry_meta["entity_keys"]),
            entity_semantics=semantics,
            neural_voxel_size_m=geometry_meta["neural_voxel_size_m"],
            coordinate_frame_id=geometry_meta["coordinate_frame_id"],
            source_manifest_sha256=geometry_meta["source_manifest_sha256"],
            source_visit_map_sha256=tuple(geometry_meta["source_visit_map_sha256"]),
            shared_center_xyz=arrays["shared_center_xyz"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise C2PreparationError("static adapter geometry cannot be reconstructed") from error
    surface_meta = _required_mapping(
        manifest.get("surface"),
        label="surface metadata",
        keys={"content_sha256", "adapter_geometry_sha256", "rgb_source"},
    )
    surface = SurfaceAttributeBundle(
        points_xyz=arrays["source_points_xyz"],
        normals_xyz=arrays["normals_xyz"],
        normal_valid=arrays["normal_valid"],
        original_vertex_indices=arrays["original_vertex_indices"],
        source_visit_ids=arrays["surface_source_visit_ids"],
        source_entity_indices=arrays["surface_source_entity_indices"],
        adapter_geometry_sha256=surface_meta["adapter_geometry_sha256"],
        rgb_source=surface_meta["rgb_source"],
    )
    sampling_meta = _required_mapping(
        manifest.get("sampling"),
        label="sampling metadata",
        keys={
            "content_sha256",
            "voxel_size_m",
            "sampler_seed",
            "sampler_mode",
            "sampler_source_sha256",
        },
    )
    sampling = NativeSamplingMap(
        adapter_to_model=arrays["adapter_to_model"],
        selected_adapter_indices=arrays["selected_adapter_indices"],
        model_grid_coordinates=arrays["model_grid_coordinates"],
        model_visit_ids=arrays["model_visit_ids"],
        visit_model_offsets=arrays["visit_model_offsets"],
        visit_grid_origins=arrays["visit_grid_origins"],
        voxel_size_m=sampling_meta["voxel_size_m"],
        sampler_seed=sampling_meta["sampler_seed"],
        sampler_mode=sampling_meta["sampler_mode"],
        sampler_source_sha256=sampling_meta["sampler_source_sha256"],
    )
    candidate_meta = _required_mapping(
        manifest.get("candidates"),
        label="candidate metadata",
        keys={"content_sha256", "maximum_candidates"},
    )
    candidates = ModelCandidateMap(
        source_point_indices=arrays["candidate_source_point_indices"],
        candidate_counts=arrays["candidate_counts"],
        maximum_candidates=candidate_meta["maximum_candidates"],
    )
    expected_hashes = {
        "geometry": geometry.content_sha256(),
        "surface": surface.content_sha256(),
        "sampling": sampling.content_sha256(),
        "candidates": candidates.content_sha256(),
    }
    for name, observed in expected_hashes.items():
        if manifest[name].get("content_sha256") != observed:
            raise C2PreparationError(f"static {name} content SHA-256 mismatch")
    return StaticPreparedInput(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings=input_bindings,
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise C2PreparationError(f"{label} must be valid JSON") from error
    if not isinstance(value, dict):
        raise C2PreparationError(f"{label} must contain a JSON object")
    return value


def _bound_relative_path(
    root: Path, record: object, *, expected_path: str, label: str
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise C2PreparationError(f"{label} binding schema is invalid")
    raw_path = record.get("path")
    if raw_path != expected_path or Path(str(raw_path)).is_absolute() or ".." in Path(
        str(raw_path)
    ).parts:
        raise C2PreparationError(f"{label} binding path is invalid")
    path = root / expected_path
    observed = _absolute_file_record(path, label=label)
    if (
        observed["sha256"] != record.get("sha256")
        or observed["byte_count"] != record.get("byte_count")
    ):
        raise C2PreparationError(f"{label} binding mismatch")
    return path


def _source_frame_binding(record: object, *, label: str) -> dict[str, object]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise C2PreparationError(f"{label} binding schema is invalid")
    path = record.get("path")
    digest = record.get("sha256")
    byte_count = record.get("byte_count")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(byte_count) is not int
        or byte_count < 0
    ):
        raise C2PreparationError(f"{label} binding values are invalid")
    return dict(record)


def _bound_absolute_path(record: object, *, label: str) -> Path:
    normalized = _source_frame_binding(record, label=label)
    path = Path(str(normalized["path"]))
    observed = _absolute_file_record(path, label=label)
    if observed != normalized:
        raise C2PreparationError(f"{label} binding mismatch")
    return path


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    status = path.stat(follow_symlinks=False)
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def resolve_frozen_static_sources(
    *,
    source_config_path: str | Path,
    sampler_source_path: str | Path,
    expected_sampler_source_sha256: str,
) -> FrozenStaticSources:
    """Resolve only the frozen Apartment inputs needed before RGB recovery."""

    config_path = Path(os.path.abspath(os.fspath(source_config_path)))
    config = _load_json_object(config_path, label="frozen B7 source config")
    frozen_inputs = config.get("frozen_inputs")
    if (
        config.get("schema_version") != 1
        or config.get("status") != "FROZEN_BEFORE_FIRST_B7_SCORE"
        or not isinstance(frozen_inputs, Mapping)
        or not isinstance(frozen_inputs.get("apartment"), Mapping)
    ):
        raise C2PreparationError("frozen B7 source config identity is invalid")
    records = frozen_inputs["apartment"]
    required = {
        "protocol",
        "two_visit_ovi_manifest",
        "b0_output_manifest",
        "b2_output_manifest",
    }
    if not required <= set(records):
        raise C2PreparationError("frozen Apartment source records are incomplete")
    resolved: dict[str, tuple[Path, dict[str, object]]] = {}
    for name in sorted(required):
        path = _bound_absolute_path(records[name], label=name)
        resolved[name] = (path, dict(records[name]))
    sampler_path = Path(os.path.abspath(os.fspath(sampler_source_path)))
    sampler_record = _absolute_file_record(sampler_path, label="sampler source")
    if sampler_record["sha256"] != expected_sampler_source_sha256:
        raise C2PreparationError("sampler source SHA-256 mismatch")
    return FrozenStaticSources(
        source_config_path=config_path,
        protocol_path=resolved["protocol"][0],
        two_visit_ovi_manifest_path=resolved["two_visit_ovi_manifest"][0],
        b0_output_manifest_record=resolved["b0_output_manifest"][1],
        b2_output_manifest_record=resolved["b2_output_manifest"][1],
        sampler_source_path=sampler_path,
        sampler_source_sha256=str(sampler_record["sha256"]),
        input_bindings={
            "source_config": _absolute_file_record(
                config_path, label="frozen B7 source config"
            ),
            "b0_output_manifest": resolved["b0_output_manifest"][1],
            "b2_output_manifest": resolved["b2_output_manifest"][1],
            "sampler_source": sampler_record,
        },
    )


def load_bound_surface_groups(
    visits: Sequence[VisitMap], native_manifest_paths: Sequence[str | Path]
) -> dict[tuple[int, str], SurfaceGroup]:
    """Recover exact PLY rows and normals through each native instance palette."""

    pair = tuple(visits)
    manifest_paths = tuple(Path(value) for value in native_manifest_paths)
    if len(pair) != 2 or not all(isinstance(value, VisitMap) for value in pair):
        raise C2PreparationError("surface loading requires two VisitMap values")
    try:
        validate_visit_pair(pair[0], pair[1])
    except (TypeError, ValueError) as error:
        raise C2PreparationError("surface VisitMap pair is invalid") from error
    if len(manifest_paths) != 2:
        raise C2PreparationError("surface loading requires two native manifests")
    output: dict[tuple[int, str], SurfaceGroup] = {}
    for visit, manifest_path in zip(pair, manifest_paths, strict=True):
        manifest = _load_json_object(manifest_path, label="native manifest")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "instance_mesh",
            "instance_color_log",
            "semantic_features",
        }:
            raise C2PreparationError("native artifact inventory is invalid")
        mesh_path = _bound_absolute_path(
            artifacts["instance_mesh"], label=f"t{visit.visit_id} instance mesh"
        )
        log_path = _bound_absolute_path(
            artifacts["instance_color_log"],
            label=f"t{visit.visit_id} instance color log",
        )
        mesh_identity = _file_identity(mesh_path)
        colors_by_instance = parse_instance_color_log(log_path)
        requested: dict[tuple[int, int, int], str] = {}
        for entity in sorted(visit.snapshot.entities, key=lambda value: value.entity_id):
            if not len(entity.points_xyz):
                continue
            instance_id = entity.metadata.get("source_instance_id")
            if type(instance_id) is not int or instance_id not in colors_by_instance:
                raise C2PreparationError(
                    f"OVI entity lacks a bound native instance color: {entity.entity_id}"
                )
            color = colors_by_instance[instance_id]
            if color in requested:
                raise C2PreparationError("multiple OVI entities claim one PLY palette color")
            requested[color] = entity.entity_id
        groups_by_color = load_ply_surface_attributes(mesh_path, requested)
        if _file_identity(mesh_path) != mesh_identity:
            raise C2PreparationError("native instance mesh changed during sidecar loading")
        if set(groups_by_color) != set(requested):
            raise C2PreparationError("native PLY does not cover every OVI entity color")
        entities = {
            entity.entity_id: entity
            for entity in visit.snapshot.entities
            if len(entity.points_xyz)
        }
        for color, entity_id in requested.items():
            group = groups_by_color[color]
            try:
                validate_surface_group(entities[entity_id].points_xyz, group)
            except ValueError as error:
                raise C2PreparationError(
                    f"native PLY rows disagree with OVI entity: {entity_id}"
                ) from error
            output[(visit.visit_id, entity_id)] = group
    return output


def prepare_static_contract(
    *,
    visits: Sequence[VisitMap],
    native_manifest_paths: Sequence[str | Path],
    sampler_source_path: str | Path,
    expected_sampler_source_sha256: str,
    input_bindings: Mapping[str, Mapping[str, object]],
    output_root: str | Path,
    neural_voxel_size_m: float = 0.02,
    sampler_seed: int = 45,
    maximum_candidates: int = 8,
    grid_sample_factory: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Build and cache the complete source-bound static D/A/M contract."""

    pair = tuple(visits)
    if len(pair) != 2 or not all(isinstance(value, VisitMap) for value in pair):
        raise C2PreparationError("static preparation requires two VisitMap values")
    geometry = build_adapter_geometry(
        pair[0], pair[1], voxel_size_m=neural_voxel_size_m
    )
    groups = load_bound_surface_groups(pair, native_manifest_paths)
    surface = bind_surface_attributes(geometry, pair[0], pair[1], groups)
    sampling = capture_native_sampling(
        coordinates_xyzt=geometry.coordinates_xyzt,
        visit_ids=geometry.visit_ids,
        shared_center_xyz=geometry.shared_center_xyz,
        voxel_size_m=geometry.neural_voxel_size_m,
        sampler_seed=sampler_seed,
        sampler_source_path=sampler_source_path,
        expected_sampler_source_sha256=expected_sampler_source_sha256,
        grid_sample_factory=grid_sample_factory,
    )
    validate_native_sampling(geometry, sampling)
    candidates = select_model_candidates(
        geometry,
        sampling,
        surface,
        maximum_candidates=maximum_candidates,
    )
    write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings=input_bindings,
        output_root=output_root,
    )
    return {
        "artifact_id": "OVI_RESCENE_STATIC_PREPARATION_SUMMARY_V2",
        "status": "PASS",
        "source_point_count": geometry.source_point_count,
        "adapter_count": geometry.adapter_count,
        "model_count": sampling.model_count,
        "merge_count": sampling.merge_count,
        "cross_entity_model_token_count": sampling.cross_entity_model_token_count(
            geometry
        ),
        "candidate_width": candidates.maximum_candidates,
    }


def prepare_static_from_frozen_config(
    *,
    source_config_path: str | Path,
    sampler_source_path: str | Path,
    expected_sampler_source_sha256: str,
    output_root: str | Path,
    grid_sample_factory: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Reopen the frozen Apartment graph and publish its static C2-V2 cache."""

    from scripts.evaluation.execute_ovi_rescene_two_visit_matrix import (
        load_two_visit_ovi_inputs,
    )
    from scripts.evaluation.run_ovi_rescene_b7 import (
        _load_snapshot_output,
        semantic_labelers_from_frozen_snapshots,
    )
    from src.oviv2.ovimap_visit_loader import load_ovimap_visit

    sources = resolve_frozen_static_sources(
        source_config_path=source_config_path,
        sampler_source_path=sampler_source_path,
        expected_sampler_source_sha256=expected_sampler_source_sha256,
    )
    t0_snapshot, _b0_manifest, _b0_path = _load_snapshot_output(
        sources.b0_output_manifest_record, variant_id="B0"
    )
    t1_snapshot, _b2_manifest, _b2_path = _load_snapshot_output(
        sources.b2_output_manifest_record, variant_id="B2"
    )
    labelers = semantic_labelers_from_frozen_snapshots(t0_snapshot, t1_snapshot)
    loaded = load_two_visit_ovi_inputs(
        protocol_path=sources.protocol_path,
        two_visit_ovi_manifest=sources.two_visit_ovi_manifest_path,
        scene="apartment",
    )
    visits: list[VisitMap] = []
    native_paths: list[Path] = []
    for visit_id, visit_name in enumerate(("t0", "t1")):
        record = loaded["visits"][visit_name]
        visits.append(
            load_ovimap_visit(
                native_manifest=record["native_manifest_path"],
                materialized_manifest=record["materialized_manifest_path"],
                visit_id=visit_id,
                scene="apartment",
                coordinate_frame_id="tesse_cd_world",
                source_manifest_sha256=loaded["protocol_content_sha256"],
                observed_frame_start=int(record["start_frame"]),
                observed_frame_end=int(record["end_frame"]),
                semantic_labeler=labelers[visit_id],
            )
        )
        native_paths.append(Path(record["native_manifest_path"]))
    bindings = dict(loaded["input_bindings"])
    bindings.update(sources.input_bindings)
    return prepare_static_contract(
        visits=tuple(visits),
        native_manifest_paths=tuple(native_paths),
        sampler_source_path=sources.sampler_source_path,
        expected_sampler_source_sha256=sources.sampler_source_sha256,
        input_bindings=bindings,
        output_root=output_root,
        neural_voxel_size_m=0.02,
        sampler_seed=45,
        maximum_candidates=8,
        grid_sample_factory=grid_sample_factory,
    )


def load_materialized_visit(
    *,
    manifest_path: str | Path,
    expected_visit_id: int,
    expected_global_start: int,
    expected_global_end: int,
    expected_source_input_sha256: str,
) -> MaterializedVisitFrames:
    """Load a frozen two-visit materialization without consulting evaluation GT."""

    if type(expected_visit_id) is not int or expected_visit_id not in (0, 1):
        raise C2PreparationError("expected visit ID must be 0 or 1")
    if (
        type(expected_global_start) is not int
        or type(expected_global_end) is not int
        or expected_global_start < 0
        or expected_global_end < expected_global_start
    ):
        raise C2PreparationError("expected global frame interval is invalid")
    path = Path(os.path.abspath(os.fspath(manifest_path)))
    if path.is_symlink() or not path.is_file():
        raise C2PreparationError("materialized manifest is missing or a symlink")
    manifest = _load_json_object(path, label="materialized manifest")
    expected_keys = {
        "schema_version",
        "status",
        "dataset",
        "scene",
        "visit_id",
        "frame_count",
        "source_frame_interval",
        "source_input_sha256",
        "source_bindings",
        "outputs",
    }
    frame_count = expected_global_end - expected_global_start + 1
    scene = manifest.get("scene")
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "MATERIALIZED_INPUT_PASS"
        or manifest.get("dataset") != "TESSE-CD"
        or scene not in {"apartment", "office"}
        or manifest.get("visit_id") != f"t{expected_visit_id}"
        or manifest.get("frame_count") != frame_count
        or manifest.get("source_frame_interval")
        != [expected_global_start, expected_global_end]
        or manifest.get("source_input_sha256") != expected_source_input_sha256
    ):
        raise C2PreparationError("materialized visit identity mismatch")
    root = path.parent.parent
    if path != root / str(scene) / "export_manifest.json":
        raise C2PreparationError("materialized manifest path is not canonical")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "camera",
        "trajectory",
        "timestamps",
    }:
        raise C2PreparationError("materialized output bindings are invalid")
    camera_path = _bound_relative_path(
        root, outputs["camera"], expected_path="cam_params.json", label="camera"
    )
    trajectory_path = _bound_relative_path(
        root,
        outputs["trajectory"],
        expected_path=f"{scene}/traj.txt",
        label="trajectory",
    )
    timestamps_path = _bound_relative_path(
        root,
        outputs["timestamps"],
        expected_path=f"{scene}/timestamps.csv",
        label="timestamps",
    )
    camera_document = _load_json_object(camera_path, label="camera")
    camera = camera_document.get("camera")
    expected_camera = {
        "cx": 360.0,
        "cy": 240.0,
        "fx": 415.69219381653056,
        "fy": 415.69219381653056,
        "h": 480,
        "scale": 1000.0,
        "w": 720,
    }
    if camera != expected_camera:
        raise C2PreparationError("materialized camera parameters are not frozen TESSE-CD")
    intrinsics = CameraIntrinsics(
        fx=expected_camera["fx"],
        fy=expected_camera["fy"],
        cx=expected_camera["cx"],
        cy=expected_camera["cy"],
        width=expected_camera["w"],
        height=expected_camera["h"],
    )
    try:
        poses = np.loadtxt(trajectory_path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise C2PreparationError("materialized trajectory is invalid") from error
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 16)
    if (
        poses.shape != (frame_count, 16)
        or not np.all(np.isfinite(poses))
        or not np.allclose(
            poses[:, 12:16],
            np.tile([0.0, 0.0, 0.0, 1.0], (frame_count, 1)),
            rtol=0.0,
            atol=1e-8,
        )
    ):
        raise C2PreparationError("materialized camera poses are invalid")
    poses = poses.reshape(frame_count, 4, 4)
    try:
        with timestamps_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != [
                "frame_index",
                "sensor_timestamp_ns",
                "relative_timestamp_ns",
            ]:
                raise C2PreparationError("materialized timestamp header is invalid")
            timestamp_rows = tuple(reader)
        sensor_timestamps = tuple(
            int(row["sensor_timestamp_ns"]) for row in timestamp_rows
        )
        relative_timestamps = tuple(
            int(row["relative_timestamp_ns"]) for row in timestamp_rows
        )
        frame_indices = tuple(int(row["frame_index"]) for row in timestamp_rows)
    except (KeyError, OSError, TypeError, ValueError) as error:
        if isinstance(error, C2PreparationError):
            raise
        raise C2PreparationError("materialized timestamps are invalid") from error
    if (
        len(timestamp_rows) != frame_count
        or frame_indices != tuple(range(frame_count))
        or any(a >= b for a, b in pairwise(sensor_timestamps))
        or any(
            relative != timestamp - sensor_timestamps[0]
            for timestamp, relative in zip(
                sensor_timestamps, relative_timestamps, strict=True
            )
        )
    ):
        raise C2PreparationError("materialized timestamps do not form one window")
    source_bindings = manifest.get("source_bindings")
    expected_roles = {
        f"{kind}/{index:06d}"
        for kind in ("rgb", "depth")
        for index in range(frame_count)
    }
    if not isinstance(source_bindings, Mapping) or set(source_bindings) != expected_roles:
        raise C2PreparationError("materialized frame bindings are incomplete")
    rgb_paths = tuple(
        root / str(scene) / "results" / f"frame{index:06d}.jpg"
        for index in range(frame_count)
    )
    depth_paths = tuple(
        root / str(scene) / "results" / f"depth{index:06d}.png"
        for index in range(frame_count)
    )
    rgb_bindings = tuple(
        _source_frame_binding(source_bindings[f"rgb/{index:06d}"], label="RGB frame")
        for index in range(frame_count)
    )
    depth_bindings = tuple(
        _source_frame_binding(
            source_bindings[f"depth/{index:06d}"], label="depth frame"
        )
        for index in range(frame_count)
    )
    for index, (rgb_path, depth_path) in enumerate(
        zip(rgb_paths, depth_paths, strict=True)
    ):
        for local_path, binding, label in (
            (rgb_path, rgb_bindings[index], "RGB"),
            (depth_path, depth_bindings[index], "depth"),
        ):
            try:
                status = local_path.stat(follow_symlinks=False)
            except OSError as error:
                raise C2PreparationError(
                    f"materialized {label} frame is unavailable: {index}"
                ) from error
            if not stat.S_ISREG(status.st_mode) or status.st_size != binding["byte_count"]:
                raise C2PreparationError(
                    f"materialized {label} frame size or type mismatch: {index}"
                )
    poses.setflags(write=False)
    return MaterializedVisitFrames(
        manifest_path=path,
        root=root,
        scene=str(scene),
        visit_id=expected_visit_id,
        global_frame_start=expected_global_start,
        global_frame_end=expected_global_end,
        source_input_sha256=expected_source_input_sha256,
        intrinsics=intrinsics,
        camera_to_world=poses,
        sensor_timestamps_ns=sensor_timestamps,
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        rgb_bindings=rgb_bindings,
        depth_bindings=depth_bindings,
    )


def recover_model_support(
    prepared: StaticPreparedInput,
    visit_windows: Mapping[int, MaterializedVisitFrames],
    *,
    depth_tolerance_m: float,
) -> RecoveryResult:
    """Recover one legal same-visit RGB observation for every possible M row."""

    if not isinstance(prepared, StaticPreparedInput):
        raise TypeError("prepared must be StaticPreparedInput")
    if set(visit_windows) != {0, 1} or any(
        not isinstance(visit_windows[key], MaterializedVisitFrames)
        or visit_windows[key].visit_id != key
        for key in (0, 1)
    ):
        raise C2PreparationError("visit window mapping is inconsistent")
    tolerance = float(depth_tolerance_m)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise C2PreparationError("depth tolerance must be finite and positive")
    geometry = prepared.geometry
    surface = prepared.surface
    sampling = validate_native_sampling(geometry, prepared.sampling)
    candidates = prepared.candidates
    model_count = sampling.model_count
    support_valid = np.zeros(model_count, dtype=np.bool_)
    representatives = np.full(model_count, -1, dtype=np.int64)
    rgb = np.zeros((model_count, 3), dtype=np.uint8)
    local_frames = np.full(model_count, -1, dtype=np.int64)
    global_frames = np.full(model_count, -1, dtype=np.int64)
    rows = np.full(model_count, -1, dtype=np.int64)
    columns = np.full(model_count, -1, dtype=np.int64)
    camera_depth = np.full(model_count, np.nan, dtype=np.float32)
    observed_depth = np.full(model_count, np.nan, dtype=np.float32)
    residual = np.full(model_count, np.nan, dtype=np.float32)

    for visit_id in (0, 1):
        window = visit_windows[visit_id]
        decoded_frames = tuple(
            window.read_frame(index) for index in range(window.frame_count)
        )
        visit_models = np.flatnonzero(sampling.model_visit_ids == visit_id)
        unresolved = visit_models.copy()
        for candidate_rank in range(candidates.maximum_candidates):
            if len(unresolved) == 0:
                break
            source_indices = candidates.source_point_indices[
                unresolved, candidate_rank
            ]
            available = source_indices >= 0
            round_models = unresolved[available]
            round_sources = source_indices[available]
            if len(round_models) == 0:
                continue
            if not np.array_equal(
                surface.source_visit_ids[round_sources],
                np.full(len(round_sources), visit_id, dtype=np.int8),
            ):
                raise C2PreparationError("candidate source crosses visit boundary")
            best_residual = np.full(len(round_models), np.inf, dtype=np.float64)
            best_local = np.full(len(round_models), -1, dtype=np.int64)
            best_rows = np.full(len(round_models), -1, dtype=np.int64)
            best_columns = np.full(len(round_models), -1, dtype=np.int64)
            best_camera = np.full(len(round_models), np.nan, dtype=np.float32)
            best_observed = np.full(len(round_models), np.nan, dtype=np.float32)
            best_rgb = np.zeros((len(round_models), 3), dtype=np.uint8)
            points = surface.points_xyz[round_sources]
            for local_index, frame in enumerate(decoded_frames):
                projection = project_world_points(points, frame.pose, frame.intrinsics)
                supported = sample_depth_consistent_rgb(
                    projection,
                    frame.rgb,
                    frame.depth,
                    depth_tolerance_m=tolerance,
                    candidate_visit_ids=np.full(
                        len(round_models), visit_id, dtype=np.int8
                    ),
                    frame_visit_id=visit_id,
                    rgb_source="camera_rgb",
                )
                positions = supported.candidate_indices
                if len(positions) == 0:
                    continue
                observed_residual = supported.depth_residual_m.astype(np.float64)
                improve = observed_residual < best_residual[positions]
                positions = positions[improve]
                if len(positions) == 0:
                    continue
                best_residual[positions] = supported.depth_residual_m[improve]
                best_local[positions] = local_index
                best_rows[positions] = supported.rows[improve]
                best_columns[positions] = supported.columns[improve]
                best_camera[positions] = supported.camera_depth_m[improve]
                best_observed[positions] = supported.observed_depth_m[improve]
                best_rgb[positions] = supported.rgb_uint8[improve]
            found = best_local >= 0
            found_models = round_models[found]
            found_sources = round_sources[found]
            support_valid[found_models] = True
            representatives[found_models] = found_sources
            rgb[found_models] = best_rgb[found]
            local_frames[found_models] = best_local[found]
            global_frames[found_models] = (
                window.global_frame_start + best_local[found]
            )
            rows[found_models] = best_rows[found]
            columns[found_models] = best_columns[found]
            camera_depth[found_models] = best_camera[found]
            observed_depth[found_models] = best_observed[found]
            residual[found_models] = best_residual[found].astype(np.float32)
            unresolved = unresolved[~np.isin(unresolved, found_models)]
        del decoded_frames

    support = RecoveredModelSupport(
        support_valid=support_valid,
        representative_source_point_indices=representatives,
        rgb_uint8=rgb,
        local_frame_indices=local_frames,
        global_frame_indices=global_frames,
        rows=rows,
        columns=columns,
        camera_depth_m=camera_depth,
        observed_depth_m=observed_depth,
        depth_residual_m=residual,
    )
    unsupported = np.flatnonzero(~support_valid).astype(np.int64)
    unsupported.setflags(write=False)
    return RecoveryResult(
        status="C2_V2_PASS" if len(unsupported) == 0 else "PARTIAL_INPUT_SCOPE",
        support=support,
        unsupported_model_indices=unsupported,
    )


def measure_calibration_residuals(
    prepared: StaticPreparedInput,
    visit_windows: Mapping[int, MaterializedVisitFrames],
    *,
    calibration_indices: Mapping[int, np.ndarray],
    residual_ceiling_m: float = 0.08,
) -> dict[int, np.ndarray]:
    """Measure minimum valid residuals for each frozen first-candidate sample."""

    if not isinstance(prepared, StaticPreparedInput):
        raise TypeError("prepared must be StaticPreparedInput")
    if set(visit_windows) != {0, 1} or any(
        not isinstance(visit_windows[key], MaterializedVisitFrames)
        or visit_windows[key].visit_id != key
        for key in (0, 1)
    ):
        raise C2PreparationError("visit window mapping is inconsistent")
    if set(calibration_indices) != {0, 1}:
        raise C2PreparationError("calibration indices must contain visits 0 and 1")
    ceiling = float(residual_ceiling_m)
    if not math.isfinite(ceiling) or ceiling <= 0.0:
        raise C2PreparationError("calibration residual ceiling must be positive")
    sampling = validate_native_sampling(prepared.geometry, prepared.sampling)
    output: dict[int, np.ndarray] = {}
    for visit_id in (0, 1):
        raw_indices = np.asarray(calibration_indices[visit_id])
        if (
            raw_indices.ndim != 1
            or len(raw_indices) == 0
            or not np.issubdtype(raw_indices.dtype, np.integer)
        ):
            raise C2PreparationError("calibration indices must be non-empty integers")
        indices = raw_indices.astype(np.int64, copy=False)
        if (
            np.any(indices < 0)
            or np.any(indices >= sampling.model_count)
            or len(np.unique(indices)) != len(indices)
            or np.any(np.diff(indices) <= 0)
            or not np.array_equal(
                sampling.model_visit_ids[indices],
                np.full(len(indices), visit_id, dtype=np.int8),
            )
        ):
            raise C2PreparationError("calibration indices are not a valid visit subset")
        if np.any(prepared.candidates.candidate_counts[indices] < 1):
            raise C2PreparationError("calibration sample lacks a valid first candidate")
        source_indices = prepared.candidates.source_point_indices[indices, 0]
        if not np.array_equal(
            prepared.surface.source_visit_ids[source_indices],
            np.full(len(indices), visit_id, dtype=np.int8),
        ):
            raise C2PreparationError("calibration candidate crosses visit boundary")
        points = prepared.surface.points_xyz[source_indices]
        best = np.full(len(indices), np.inf, dtype=np.float64)
        window = visit_windows[visit_id]
        decoded_frames = tuple(
            window.read_frame(index) for index in range(window.frame_count)
        )
        for frame in decoded_frames:
            projection = project_world_points(points, frame.pose, frame.intrinsics)
            supported = sample_depth_consistent_rgb(
                projection,
                frame.rgb,
                frame.depth,
                depth_tolerance_m=ceiling,
                candidate_visit_ids=np.full(len(indices), visit_id, dtype=np.int8),
                frame_visit_id=visit_id,
                rgb_source="camera_rgb",
            )
            positions = supported.candidate_indices
            if len(positions):
                best[positions] = np.minimum(
                    best[positions], supported.depth_residual_m.astype(np.float64)
                )
        del decoded_frames
        best[~np.isfinite(best)] = np.nan
        result = np.ascontiguousarray(best, dtype=np.float64)
        result.setflags(write=False)
        output[visit_id] = result
    return output


def write_depth_calibration_artifact(
    *,
    static_input_root: str | Path,
    prepared: StaticPreparedInput,
    visit_windows: Mapping[int, MaterializedVisitFrames],
    calibration_indices: Mapping[int, np.ndarray],
    residuals_by_visit: Mapping[int, np.ndarray],
    decision: DepthToleranceDecision,
    output_root: str | Path,
    minimum_inlier_count: int = 512,
    residual_ceiling_m: float = 0.08,
    minimum_tolerance_m: float = 0.01,
    maximum_tolerance_m: float = 0.05,
) -> DepthCalibrationArtifactPaths:
    """Atomically publish reproducible input-only depth calibration evidence."""

    if not isinstance(prepared, StaticPreparedInput):
        raise TypeError("prepared must be StaticPreparedInput")
    if not isinstance(decision, DepthToleranceDecision):
        raise TypeError("decision must be DepthToleranceDecision")
    if set(visit_windows) != {0, 1} or any(
        not isinstance(visit_windows[key], MaterializedVisitFrames)
        or visit_windows[key].visit_id != key
        for key in (0, 1)
    ):
        raise C2PreparationError("visit window mapping is inconsistent")
    if set(calibration_indices) != {0, 1} or set(residuals_by_visit) != {0, 1}:
        raise C2PreparationError("calibration arrays must contain both visits")
    arrays: dict[str, np.ndarray] = {}
    for visit_id in (0, 1):
        indices = np.asarray(calibration_indices[visit_id])
        residuals = np.asarray(residuals_by_visit[visit_id], dtype=np.float64)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or residuals.shape != indices.shape
        ):
            raise C2PreparationError("calibration index and residual arrays must align")
        arrays[f"t{visit_id}_model_indices"] = np.ascontiguousarray(
            indices, dtype=np.int64
        )
        arrays[f"t{visit_id}_min_depth_residual_m"] = np.ascontiguousarray(
            residuals, dtype=np.float64
        )
    observed_decision = select_depth_tolerance(
        {0: arrays["t0_min_depth_residual_m"], 1: arrays["t1_min_depth_residual_m"]},
        minimum_inlier_count=minimum_inlier_count,
        residual_ceiling_m=residual_ceiling_m,
        minimum_tolerance_m=minimum_tolerance_m,
        maximum_tolerance_m=maximum_tolerance_m,
    )
    if observed_decision != decision:
        raise C2PreparationError("calibration decision does not match residual evidence")
    static_manifest_path = Path(static_input_root).absolute() / "manifest.json"
    static_manifest_record = _absolute_file_record(
        static_manifest_path, label="static input manifest"
    )
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.exists() or output.is_symlink():
        raise C2PreparationError(f"depth calibration artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arrays_path = staging / "arrays.npz"
        with arrays_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        arrays_record = _absolute_file_record(
            arrays_path, label="depth calibration arrays"
        )
        arrays_record["path"] = arrays_path.name
        manifest = {
            "schema_version": 2,
            "artifact_id": "OVI_RESCENE_DEPTH_CALIBRATION_V2",
            "status": "PASS",
            "method_predictions_used": False,
            "ground_truth_used": False,
            "static_input_manifest": static_manifest_record,
            "static_sampling_sha256": prepared.sampling.content_sha256(),
            "visit_manifests": {
                f"t{visit_id}": _absolute_file_record(
                    visit_windows[visit_id].manifest_path,
                    label=f"t{visit_id} materialized manifest",
                )
                for visit_id in (0, 1)
            },
            "sample_selection": "endpoint_inclusive_evenly_spaced_model_rows_per_visit",
            "candidate_policy": "first_valid_normal_source_candidate_only",
            "sample_count_per_visit": {
                f"t{visit_id}": len(arrays[f"t{visit_id}_model_indices"])
                for visit_id in (0, 1)
            },
            "minimum_inlier_count": minimum_inlier_count,
            "residual_ceiling_m": float(residual_ceiling_m),
            "minimum_tolerance_m": float(minimum_tolerance_m),
            "maximum_tolerance_m": float(maximum_tolerance_m),
            "inlier_counts": {
                "t0": decision.inlier_counts[0],
                "t1": decision.inlier_counts[1],
            },
            "visit_q99_m": {
                "t0": decision.visit_q99_m[0],
                "t1": decision.visit_q99_m[1],
            },
            "visit_tolerances_m": {
                "t0": decision.visit_tolerances_m[0],
                "t1": decision.visit_tolerances_m[1],
            },
            "selected_tolerance_m": decision.selected_tolerance_m,
            "arrays": arrays_record,
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DepthCalibrationArtifactPaths(
        root=output,
        manifest=output / "manifest.json",
        arrays=output / "arrays.npz",
    )


def load_depth_calibration_artifact(
    calibration_root: str | Path,
    *,
    static_input_root: str | Path,
    prepared: StaticPreparedInput,
) -> LoadedDepthCalibration:
    """Verify calibration arrays and recompute the frozen tolerance decision."""

    if not isinstance(prepared, StaticPreparedInput):
        raise TypeError("prepared must be StaticPreparedInput")
    root = Path(os.path.abspath(os.fspath(calibration_root)))
    manifest_path = root / "manifest.json"
    arrays_path = root / "arrays.npz"
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, arrays_path)):
        raise C2PreparationError("depth calibration artifact files are missing")
    manifest = _load_json_object(manifest_path, label="depth calibration manifest")
    expected_keys = {
        "schema_version",
        "artifact_id",
        "status",
        "method_predictions_used",
        "ground_truth_used",
        "static_input_manifest",
        "static_sampling_sha256",
        "visit_manifests",
        "sample_selection",
        "candidate_policy",
        "sample_count_per_visit",
        "minimum_inlier_count",
        "residual_ceiling_m",
        "minimum_tolerance_m",
        "maximum_tolerance_m",
        "inlier_counts",
        "visit_q99_m",
        "visit_tolerances_m",
        "selected_tolerance_m",
        "arrays",
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_version") != 2
        or manifest.get("artifact_id") != "OVI_RESCENE_DEPTH_CALIBRATION_V2"
        or manifest.get("status") != "PASS"
        or manifest.get("method_predictions_used") is not False
        or manifest.get("ground_truth_used") is not False
        or manifest.get("sample_selection")
        != "endpoint_inclusive_evenly_spaced_model_rows_per_visit"
        or manifest.get("candidate_policy")
        != "first_valid_normal_source_candidate_only"
    ):
        raise C2PreparationError("depth calibration manifest identity is invalid")
    static_root = Path(os.path.abspath(os.fspath(static_input_root)))
    static_record = _absolute_file_record(
        static_root / "manifest.json", label="static input manifest"
    )
    if manifest.get("static_input_manifest") != static_record:
        raise C2PreparationError("calibration static input binding mismatch")
    static_manifest = _load_json_object(
        static_root / "manifest.json", label="static input manifest"
    )
    expected_static_hashes = {
        "geometry": prepared.geometry.content_sha256(),
        "surface": prepared.surface.content_sha256(),
        "sampling": prepared.sampling.content_sha256(),
        "candidates": prepared.candidates.content_sha256(),
    }
    if any(
        not isinstance(static_manifest.get(name), Mapping)
        or static_manifest[name].get("content_sha256") != digest
        for name, digest in expected_static_hashes.items()
    ):
        raise C2PreparationError("prepared static content differs from calibration parent")
    if manifest.get("static_sampling_sha256") != prepared.sampling.content_sha256():
        raise C2PreparationError("calibration sampling identity mismatch")
    visit_records = manifest.get("visit_manifests")
    if not isinstance(visit_records, Mapping) or set(visit_records) != {"t0", "t1"}:
        raise C2PreparationError("calibration visit manifest bindings are invalid")
    for visit_name in ("t0", "t1"):
        record = visit_records[visit_name]
        if not isinstance(record, Mapping):
            raise C2PreparationError("calibration visit manifest binding is invalid")
        observed = _absolute_file_record(
            Path(str(record.get("path"))), label=f"{visit_name} materialized manifest"
        )
        if dict(record) != observed:
            raise C2PreparationError(
                f"calibration visit manifest binding mismatch: {visit_name}"
            )
    arrays_record = manifest.get("arrays")
    observed_arrays = _absolute_file_record(
        arrays_path, label="depth calibration arrays"
    )
    if (
        not isinstance(arrays_record, Mapping)
        or set(arrays_record) != {"path", "sha256", "byte_count"}
        or arrays_record.get("path") != "arrays.npz"
        or arrays_record.get("sha256") != observed_arrays["sha256"]
        or arrays_record.get("byte_count") != observed_arrays["byte_count"]
    ):
        raise C2PreparationError("calibration arrays binding mismatch")
    try:
        with np.load(io.BytesIO(arrays_path.read_bytes()), allow_pickle=False) as source:
            expected_arrays = {
                "t0_model_indices",
                "t0_min_depth_residual_m",
                "t1_model_indices",
                "t1_min_depth_residual_m",
            }
            if set(source.files) != expected_arrays:
                raise C2PreparationError("calibration array schema is invalid")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, C2PreparationError):
            raise
        raise C2PreparationError("calibration arrays cannot be decoded") from error
    indices: dict[int, np.ndarray] = {}
    residuals: dict[int, np.ndarray] = {}
    counts = manifest.get("sample_count_per_visit")
    if not isinstance(counts, Mapping) or set(counts) != {"t0", "t1"}:
        raise C2PreparationError("calibration sample counts are invalid")
    for visit_id in (0, 1):
        visit_indices = arrays[f"t{visit_id}_model_indices"]
        visit_residuals = arrays[f"t{visit_id}_min_depth_residual_m"]
        if (
            visit_indices.ndim != 1
            or not np.issubdtype(visit_indices.dtype, np.integer)
            or visit_residuals.shape != visit_indices.shape
            or not np.issubdtype(visit_residuals.dtype, np.floating)
            or counts.get(f"t{visit_id}") != len(visit_indices)
            or len(np.unique(visit_indices)) != len(visit_indices)
            or np.any(np.diff(visit_indices) <= 0)
            or np.any(visit_indices < 0)
            or np.any(visit_indices >= prepared.sampling.model_count)
            or not np.array_equal(
                prepared.sampling.model_visit_ids[visit_indices],
                np.full(len(visit_indices), visit_id, dtype=np.int8),
            )
        ):
            raise C2PreparationError("calibration arrays do not match sampling domain")
        indices[visit_id] = np.ascontiguousarray(visit_indices, dtype=np.int64)
        residuals[visit_id] = np.ascontiguousarray(visit_residuals, dtype=np.float64)
        indices[visit_id].setflags(write=False)
        residuals[visit_id].setflags(write=False)
    try:
        decision = select_depth_tolerance(
            residuals,
            minimum_inlier_count=manifest["minimum_inlier_count"],
            residual_ceiling_m=manifest["residual_ceiling_m"],
            minimum_tolerance_m=manifest["minimum_tolerance_m"],
            maximum_tolerance_m=manifest["maximum_tolerance_m"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise C2PreparationError("calibration policy cannot be reproduced") from error
    declared_decision = {
        "inlier_counts": manifest.get("inlier_counts"),
        "visit_q99_m": manifest.get("visit_q99_m"),
        "visit_tolerances_m": manifest.get("visit_tolerances_m"),
        "selected_tolerance_m": manifest.get("selected_tolerance_m"),
    }
    observed_decision = {
        "inlier_counts": {"t0": decision.inlier_counts[0], "t1": decision.inlier_counts[1]},
        "visit_q99_m": {"t0": decision.visit_q99_m[0], "t1": decision.visit_q99_m[1]},
        "visit_tolerances_m": {
            "t0": decision.visit_tolerances_m[0],
            "t1": decision.visit_tolerances_m[1],
        },
        "selected_tolerance_m": decision.selected_tolerance_m,
    }
    if declared_decision != observed_decision:
        raise C2PreparationError("calibration decision differs from bound residuals")
    return LoadedDepthCalibration(
        decision=decision,
        calibration_indices=indices,
        residuals_by_visit=residuals,
        manifest=manifest,
    )


def load_bound_visit_windows(
    prepared: StaticPreparedInput,
) -> dict[int, MaterializedVisitFrames]:
    """Reopen both materialized windows through the cached frozen input graph."""

    from scripts.evaluation.execute_ovi_rescene_two_visit_matrix import (
        load_two_visit_ovi_inputs,
    )

    required = {
        "protocol",
        "two_visit_ovi_manifest",
        "t0_materialized_manifest",
        "t1_materialized_manifest",
    }
    if not required <= set(prepared.input_bindings):
        raise C2PreparationError("static input lacks frozen materialized visit bindings")
    loaded = load_two_visit_ovi_inputs(
        protocol_path=Path(prepared.input_bindings["protocol"]["path"]),
        two_visit_ovi_manifest=Path(
            prepared.input_bindings["two_visit_ovi_manifest"]["path"]
        ),
        scene="apartment",
    )
    for name, record in loaded["input_bindings"].items():
        if prepared.input_bindings.get(name) != dict(record):
            raise C2PreparationError(f"static input graph binding mismatch: {name}")
    windows: dict[int, MaterializedVisitFrames] = {}
    for visit_id, visit_name in enumerate(("t0", "t1")):
        record = loaded["visits"][visit_name]
        windows[visit_id] = load_materialized_visit(
            manifest_path=record["materialized_manifest_path"],
            expected_visit_id=visit_id,
            expected_global_start=int(record["start_frame"]),
            expected_global_end=int(record["end_frame"]),
            expected_source_input_sha256=str(record["source_input_sha256"]),
        )
    return windows


def calibrate_static_input(
    *, static_input_root: str | Path, output_root: str | Path
) -> dict[str, object]:
    """Run the single frozen input-only 2048-per-visit depth calibration."""

    prepared = load_static_input_artifact(static_input_root)
    windows = load_bound_visit_windows(prepared)
    indices = select_calibration_indices(
        prepared.sampling.model_visit_ids, sample_count_per_visit=2048
    )
    residuals = measure_calibration_residuals(
        prepared,
        windows,
        calibration_indices=indices,
        residual_ceiling_m=0.08,
    )
    decision = select_depth_tolerance(
        residuals,
        minimum_inlier_count=512,
        residual_ceiling_m=0.08,
        minimum_tolerance_m=0.01,
        maximum_tolerance_m=0.05,
    )
    paths = write_depth_calibration_artifact(
        static_input_root=static_input_root,
        prepared=prepared,
        visit_windows=windows,
        calibration_indices=indices,
        residuals_by_visit=residuals,
        decision=decision,
        output_root=output_root,
        minimum_inlier_count=512,
        residual_ceiling_m=0.08,
        minimum_tolerance_m=0.01,
        maximum_tolerance_m=0.05,
    )
    return {
        "artifact_id": "OVI_RESCENE_DEPTH_CALIBRATION_SUMMARY_V2",
        "status": "PASS",
        "sample_count_per_visit": 2048,
        "inlier_counts": {
            "t0": decision.inlier_counts[0],
            "t1": decision.inlier_counts[1],
        },
        "visit_q99_m": {
            "t0": decision.visit_q99_m[0],
            "t1": decision.visit_q99_m[1],
        },
        "selected_tolerance_m": decision.selected_tolerance_m,
        "manifest": str(paths.manifest),
    }


def build_recovered_input(
    *,
    static_input_root: str | Path,
    calibration_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Recover both visits under the frozen tolerance and publish one C2 input."""

    prepared = load_static_input_artifact(static_input_root)
    calibration = load_depth_calibration_artifact(
        calibration_root,
        static_input_root=static_input_root,
        prepared=prepared,
    )
    windows = load_bound_visit_windows(prepared)
    recovery = recover_model_support(
        prepared,
        windows,
        depth_tolerance_m=calibration.decision.selected_tolerance_m,
    )
    calibration_manifest_path = (
        Path(os.path.abspath(os.fspath(calibration_root))) / "manifest.json"
    )
    publish_recovered_input(
        static_input_root=static_input_root,
        calibration_manifest_path=calibration_manifest_path,
        prepared=prepared,
        recovery=recovery,
        depth_tolerance_m=calibration.decision.selected_tolerance_m,
        output_root=output_root,
    )
    return audit_recovered_input(
        output_root=output_root,
        static_input_root=static_input_root,
        calibration_manifest_path=calibration_manifest_path,
    )


def _native_grid_sample_factory(**options: object) -> object:
    from sonata.transform import GridSample

    return GridSample(**options)


def capture_native_sampling(
    *,
    coordinates_xyzt: np.ndarray,
    visit_ids: np.ndarray,
    shared_center_xyz: np.ndarray,
    voxel_size_m: float,
    sampler_seed: int,
    sampler_source_path: str | Path,
    expected_sampler_source_sha256: str,
    grid_sample_factory: Callable[..., object] | None = None,
) -> NativeSamplingMap:
    """Capture the actual train-mode GridSample inverse independently per visit."""

    coordinates = np.asarray(coordinates_xyzt, dtype=np.float64)
    visits = np.asarray(visit_ids)
    center = np.asarray(shared_center_xyz, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4 or not len(coordinates):
        raise C2PreparationError("adapter coordinates must have shape (A, 4)")
    if not np.all(np.isfinite(coordinates)):
        raise C2PreparationError("adapter coordinates must be finite")
    if visits.shape != (len(coordinates),) or not np.issubdtype(
        visits.dtype, np.integer
    ):
        raise C2PreparationError("adapter visit IDs must align with coordinates")
    visits = visits.astype(np.int8, copy=False)
    if set(visits.tolist()) != {0, 1} or not np.array_equal(
        coordinates[:, 3], visits.astype(np.float64)
    ):
        raise C2PreparationError("adapter temporal coordinates must contain visits 0 and 1")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise C2PreparationError("shared center must have three finite coordinates")
    voxel = float(voxel_size_m)
    if not math.isfinite(voxel) or voxel <= 0.0:
        raise C2PreparationError("voxel size must be finite and positive")
    if type(sampler_seed) is not int or sampler_seed < 0:
        raise C2PreparationError("sampler seed must be a nonnegative integer")
    observed_source_sha256 = _regular_file_sha256(
        Path(sampler_source_path), label="sampler source"
    )
    if observed_source_sha256 != expected_sampler_source_sha256:
        raise C2PreparationError("sampler source SHA-256 mismatch")

    factory = grid_sample_factory or _native_grid_sample_factory
    adapter_to_model = np.empty(len(coordinates), dtype=np.int64)
    selected_parts: list[np.ndarray] = []
    grid_parts: list[np.ndarray] = []
    visit_parts: list[np.ndarray] = []
    model_offsets = [0]
    grid_origins = np.empty((2, 3), dtype=np.int64)
    rng_state = np.random.get_state()
    try:
        np.random.seed(sampler_seed)
        for visit_id in (0, 1):
            adapter_indices = np.flatnonzero(visits == visit_id).astype(np.int64)
            centered = coordinates[adapter_indices, :3] - center
            raw_grid = np.floor(centered / voxel).astype(np.int64)
            grid_origins[visit_id] = raw_grid.min(axis=0)
            sampler = factory(
                grid_size=voxel,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
                return_grid_coord=True,
            )
            result = sampler(
                {
                    "coord": np.ascontiguousarray(centered, dtype=np.float64),
                    "adapter_index": adapter_indices.copy(),
                    "index_valid_keys": ["coord", "adapter_index"],
                }
            )
            if not isinstance(result, Mapping):
                raise C2PreparationError("native sampler returned a non-mapping")
            try:
                selected = np.asarray(result["adapter_index"])
                inverse = np.asarray(result["inverse"])
                grid = np.asarray(result["grid_coord"])
            except KeyError as error:
                raise C2PreparationError("native sampler output is incomplete") from error
            if (
                selected.ndim != 1
                or not np.issubdtype(selected.dtype, np.integer)
                or inverse.shape != (len(adapter_indices),)
                or not np.issubdtype(inverse.dtype, np.integer)
                or grid.shape != (len(selected), 3)
                or not np.issubdtype(grid.dtype, np.integer)
            ):
                raise C2PreparationError("native sampler output shapes are invalid")
            selected = selected.astype(np.int64, copy=False)
            inverse = inverse.astype(np.int64, copy=False)
            grid = grid.astype(np.int64, copy=False)
            if (
                len(selected) == 0
                or np.any(inverse < 0)
                or np.any(inverse >= len(selected))
                or not np.array_equal(np.unique(inverse), np.arange(len(selected)))
                or not np.array_equal(visits[selected], np.full(len(selected), visit_id))
            ):
                raise C2PreparationError("native sampler inverse or selection is invalid")
            model_start = model_offsets[-1]
            adapter_to_model[adapter_indices] = inverse + model_start
            selected_parts.append(selected)
            grid_parts.append(grid)
            visit_parts.append(np.full(len(selected), visit_id, dtype=np.int8))
            model_offsets.append(model_start + len(selected))
    finally:
        np.random.set_state(rng_state)

    return NativeSamplingMap(
        adapter_to_model=adapter_to_model,
        selected_adapter_indices=np.concatenate(selected_parts),
        model_grid_coordinates=np.concatenate(grid_parts),
        model_visit_ids=np.concatenate(visit_parts),
        visit_model_offsets=np.asarray(model_offsets, dtype=np.int64),
        visit_grid_origins=grid_origins,
        voxel_size_m=voxel,
        sampler_seed=sampler_seed,
        sampler_mode="train",
        sampler_source_sha256=observed_source_sha256,
    )


def select_calibration_indices(
    model_visit_ids: np.ndarray, *, sample_count_per_visit: int = 2048
) -> dict[int, np.ndarray]:
    """Select fixed endpoint-inclusive, evenly spaced M rows per visit."""

    visits = np.asarray(model_visit_ids)
    if visits.ndim != 1 or not np.issubdtype(visits.dtype, np.integer):
        raise C2PreparationError("model visit IDs must be a one-dimensional integer array")
    if type(sample_count_per_visit) is not int or sample_count_per_visit < 2:
        raise C2PreparationError("calibration sample count must be at least two")
    selected: dict[int, np.ndarray] = {}
    denominator = sample_count_per_visit - 1
    for visit_id in (0, 1):
        available = np.flatnonzero(visits == visit_id).astype(np.int64)
        if len(available) < sample_count_per_visit:
            raise C2PreparationError(
                f"visit {visit_id} has fewer than {sample_count_per_visit} model rows"
            )
        numerators = np.arange(sample_count_per_visit, dtype=np.int64) * (
            len(available) - 1
        )
        positions = (numerators + denominator // 2) // denominator
        selected[visit_id] = available[positions]
    return selected


def select_depth_tolerance(
    residuals_by_visit: Mapping[int, np.ndarray],
    *,
    minimum_inlier_count: int = 512,
    residual_ceiling_m: float = 0.08,
    minimum_tolerance_m: float = 0.01,
    maximum_tolerance_m: float = 0.05,
) -> DepthToleranceDecision:
    """Freeze max per-visit q99 after a one-millimetre upward rounding."""

    if set(residuals_by_visit) != {0, 1}:
        raise C2PreparationError("calibration residuals must contain visits 0 and 1")
    if type(minimum_inlier_count) is not int or minimum_inlier_count < 1:
        raise C2PreparationError("minimum calibration inliers must be positive")
    ceiling = float(residual_ceiling_m)
    lower = float(minimum_tolerance_m)
    upper = float(maximum_tolerance_m)
    if not (0.0 < lower <= upper <= ceiling and all(map(math.isfinite, (lower, upper, ceiling)))):
        raise C2PreparationError("calibration tolerance bounds are invalid")
    counts: list[int] = []
    quantiles: list[float] = []
    tolerances: list[float] = []
    for visit_id in (0, 1):
        raw = np.asarray(residuals_by_visit[visit_id], dtype=np.float64)
        if raw.ndim != 1:
            raise C2PreparationError("calibration residuals must be one-dimensional")
        inliers = np.sort(raw[np.isfinite(raw) & (raw >= 0.0) & (raw <= ceiling)])
        if len(inliers) < minimum_inlier_count:
            raise C2PreparationError(
                f"visit {visit_id} has {len(inliers)} calibration inliers; "
                f"requires {minimum_inlier_count}"
            )
        index = max(0, math.ceil(0.99 * len(inliers)) - 1)
        q99 = float(inliers[index])
        rounded = math.ceil(q99 * 1000.0) / 1000.0
        tolerance = min(upper, max(lower, rounded))
        counts.append(len(inliers))
        quantiles.append(q99)
        tolerances.append(tolerance)
    return DepthToleranceDecision(
        inlier_counts=(counts[0], counts[1]),
        visit_q99_m=(quantiles[0], quantiles[1]),
        visit_tolerances_m=(tolerances[0], tolerances[1]),
        selected_tolerance_m=max(tolerances),
    )


def _native_self_test(
    *,
    sampler_source_path: Path,
    sampler_source_sha256: str,
    grid_sample_factory: Callable[..., object] | None,
) -> dict[str, object]:
    coordinates = np.asarray(
        [
            [0.000, 0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0, 0.0],
            [0.021, 0.0, 0.0, 0.0],
            [1.000, 0.0, 0.0, 1.0],
            [1.001, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    sampling = capture_native_sampling(
        coordinates_xyzt=coordinates,
        visit_ids=np.asarray([0, 0, 0, 1, 1], dtype=np.int8),
        shared_center_xyz=np.zeros(3, dtype=np.float64),
        voxel_size_m=0.02,
        sampler_seed=45,
        sampler_source_path=sampler_source_path,
        expected_sampler_source_sha256=sampler_source_sha256,
        grid_sample_factory=grid_sample_factory,
    )
    expected = {
        "adapter_to_model": [1, 1, 0, 2, 2],
        "selected_adapter_indices": [2, 0, 4],
        "model_grid_coordinates": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        "visit_model_offsets": [0, 2, 3],
    }
    observed = {
        "adapter_to_model": sampling.adapter_to_model.tolist(),
        "selected_adapter_indices": sampling.selected_adapter_indices.tolist(),
        "model_grid_coordinates": sampling.model_grid_coordinates.tolist(),
        "visit_model_offsets": sampling.visit_model_offsets.tolist(),
    }
    if observed != expected:
        raise C2PreparationError("native GridSample witness differs from frozen source")
    return {
        "artifact_id": "OVI_RESCENE_NATIVE_GRID_SAMPLE_WITNESS_V2",
        "status": "PASS",
        **observed,
        "sampler_seed": sampling.sampler_seed,
        "sampler_source_sha256": sampling.sampler_source_sha256,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    native = subparsers.add_parser(
        "native-self-test", help="verify the pinned native GridSample witness"
    )
    native.add_argument("--sampler-source", type=Path, required=True)
    native.add_argument("--sampler-source-sha256", required=True)
    prepare = subparsers.add_parser(
        "prepare-static", help="cache the frozen Apartment D/A/M input domains"
    )
    prepare.add_argument("--source-config", type=Path, required=True)
    prepare.add_argument("--sampler-source", type=Path, required=True)
    prepare.add_argument("--sampler-source-sha256", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    calibrate = subparsers.add_parser(
        "calibrate", help="freeze one input-only RGB-D depth tolerance"
    )
    calibrate.add_argument("--static-input", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    build = subparsers.add_parser(
        "build", help="recover and publish one source-bound C2-V2 model input"
    )
    build.add_argument("--static-input", type=Path, required=True)
    build.add_argument("--calibration", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser(
        "audit", help="verify a published C2-V2 input without source RGB-D decoding"
    )
    audit.add_argument("--built-input", type=Path, required=True)
    audit.add_argument("--static-input", type=Path, required=True)
    audit.add_argument("--calibration", type=Path, required=True)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    grid_sample_factory: Callable[..., object] | None = None,
) -> int:
    args = _argument_parser().parse_args(argv)
    if args.command == "native-self-test":
        payload = _native_self_test(
            sampler_source_path=args.sampler_source,
            sampler_source_sha256=args.sampler_source_sha256,
            grid_sample_factory=grid_sample_factory,
        )
        sys.stdout.write(_json_bytes(payload).decode("utf-8"))
        return 0
    if args.command == "prepare-static":
        payload = prepare_static_from_frozen_config(
            source_config_path=args.source_config,
            sampler_source_path=args.sampler_source,
            expected_sampler_source_sha256=args.sampler_source_sha256,
            output_root=args.output,
            grid_sample_factory=grid_sample_factory,
        )
        sys.stdout.write(_json_bytes(payload).decode("utf-8"))
        return 0
    if args.command == "calibrate":
        payload = calibrate_static_input(
            static_input_root=args.static_input,
            output_root=args.output,
        )
        sys.stdout.write(_json_bytes(payload).decode("utf-8"))
        return 0
    if args.command == "build":
        payload = build_recovered_input(
            static_input_root=args.static_input,
            calibration_root=args.calibration,
            output_root=args.output,
        )
        sys.stdout.write(_json_bytes(payload).decode("utf-8"))
        return 0
    if args.command == "audit":
        payload = audit_recovered_input(
            output_root=args.built_input,
            static_input_root=args.static_input,
            calibration_manifest_path=args.calibration / "manifest.json",
        )
        sys.stdout.write(_json_bytes(payload).decode("utf-8"))
        return 0
    raise AssertionError("unreachable command")


__all__ = [
    "C2PreparationError",
    "DepthCalibrationArtifactPaths",
    "DepthToleranceDecision",
    "FrozenStaticSources",
    "LoadedDepthCalibration",
    "MaterializedVisitFrames",
    "RecoveryResult",
    "StaticInputArtifactPaths",
    "StaticPreparedInput",
    "audit_recovered_input",
    "build_recovered_input",
    "calibrate_static_input",
    "capture_native_sampling",
    "load_bound_surface_groups",
    "load_bound_visit_windows",
    "load_depth_calibration_artifact",
    "load_materialized_visit",
    "load_static_input_artifact",
    "main",
    "measure_calibration_residuals",
    "prepare_static_contract",
    "prepare_static_from_frozen_config",
    "publish_recovered_input",
    "recover_model_support",
    "resolve_frozen_static_sources",
    "select_calibration_indices",
    "select_depth_tolerance",
    "write_depth_calibration_artifact",
    "write_static_input_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
