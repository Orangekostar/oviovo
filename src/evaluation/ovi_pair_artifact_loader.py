"""Strict restoration of a source-bound OVI pair artifact."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
from plyfile import PlyData

from src.evaluation.ovi_pair_views import OviObjectPairView, OviObjectVisitView

_RECORD_KEYS = {"path", "sha256", "byte_count"}
_FULL_OUTPUT_KEYS = {
    "t0_instance_ply",
    "t0_rgb_ply",
    "t1_instance_ply",
    "t1_rgb_ply",
}
_PLY_PROPERTIES = (
    "x",
    "y",
    "z",
    "normal_x",
    "normal_y",
    "normal_z",
    "red",
    "green",
    "blue",
    "entity_index",
    "normal_valid",
    "appearance_valid",
    "appearance_frame_id",
    "appearance_source_frame_id",
    "appearance_row",
    "appearance_column",
    "appearance_camera_depth_m",
    "appearance_observed_depth_m",
    "appearance_depth_residual_m",
    "source_vertex_index",
)


class OviPairArtifactLoadError(ValueError):
    """Raised when a frozen pair artifact is missing or inconsistent."""


def _regular_bytes(path: Path, *, label: str) -> tuple[Path, bytes]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.lstat()
    except OSError as error:
        raise OviPairArtifactLoadError(f"{label} is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise OviPairArtifactLoadError(f"{label} must be a regular file")
    try:
        return absolute, absolute.read_bytes()
    except OSError as error:
        raise OviPairArtifactLoadError(f"{label} cannot be read") from error


def _bound_content(
    record: object, *, label: str, relative_to: Path | None = None
) -> tuple[Path, bytes]:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise OviPairArtifactLoadError(f"{label} binding schema is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise OviPairArtifactLoadError(f"{label} binding path is invalid")
    path = Path(raw_path)
    if not path.is_absolute():
        if relative_to is None or ".." in path.parts:
            raise OviPairArtifactLoadError(f"{label} binding path is invalid")
        path = relative_to / path
    absolute, content = _regular_bytes(path, label=label)
    if (
        record.get("sha256") != hashlib.sha256(content).hexdigest()
        or record.get("byte_count") != len(content)
    ):
        raise OviPairArtifactLoadError(f"{label} binding mismatch")
    return absolute, content


def _json_object(content: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OviPairArtifactLoadError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise OviPairArtifactLoadError(f"{label} must be a JSON object")
    return value


def _exact_field(
    observed: np.ndarray, expected: np.ndarray, *, label: str
) -> None:
    if observed.shape != expected.shape or not np.array_equal(
        observed, expected, equal_nan=True
    ):
        raise OviPairArtifactLoadError(f"frozen pair {label} mismatch")


def _restored_visit(
    rebuilt: OviObjectVisitView, content: bytes
) -> OviObjectVisitView:
    try:
        ply = PlyData.read(io.BytesIO(content))
        if tuple(element.name for element in ply.elements) != ("vertex",):
            raise OviPairArtifactLoadError("frozen pair PLY element schema mismatch")
        vertices = ply["vertex"].data
    except (OSError, ValueError) as error:
        if isinstance(error, OviPairArtifactLoadError):
            raise
        raise OviPairArtifactLoadError("frozen pair PLY is invalid") from error
    if vertices.dtype.names != _PLY_PROPERTIES:
        raise OviPairArtifactLoadError("frozen pair PLY property schema mismatch")
    if len(vertices) != rebuilt.point_count:
        raise OviPairArtifactLoadError("frozen pair point count mismatch")

    fields = {
        "points_xyz": np.column_stack((vertices["x"], vertices["y"], vertices["z"])),
        "normals_xyz": np.column_stack(
            (vertices["normal_x"], vertices["normal_y"], vertices["normal_z"])
        ),
        "entity_owner_indices": vertices["entity_index"].astype(np.int64) - 1,
        "normal_valid": vertices["normal_valid"].astype(bool),
        "appearance_valid": vertices["appearance_valid"].astype(bool),
        "appearance_frame_ids": vertices["appearance_frame_id"].astype(np.int64),
        "appearance_source_frame_ids": vertices[
            "appearance_source_frame_id"
        ].astype(np.int64),
        "appearance_rows": vertices["appearance_row"].astype(np.int64),
        "appearance_columns": vertices["appearance_column"].astype(np.int64),
        "appearance_camera_depth_m": vertices["appearance_camera_depth_m"],
        "appearance_observed_depth_m": vertices["appearance_observed_depth_m"],
        "appearance_depth_residual_m": vertices["appearance_depth_residual_m"],
        "source_vertex_indices": vertices["source_vertex_index"].astype(np.int64),
    }
    for name, observed in fields.items():
        _exact_field(observed, getattr(rebuilt, name), label=name)

    colors = np.column_stack(
        (vertices["red"], vertices["green"], vertices["blue"])
    ).astype(np.uint8, copy=False)
    appearance = rebuilt.appearance_valid
    if np.any(colors[~appearance] != 160):
        raise OviPairArtifactLoadError("frozen pair unsupported RGB marker mismatch")
    camera_rgb = np.zeros_like(colors)
    camera_rgb[appearance] = colors[appearance]
    return replace(rebuilt, camera_rgb_uint8=camera_rgb)


def _validate_visit_manifest(
    record: object, visit: OviObjectVisitView
) -> None:
    if not isinstance(record, Mapping):
        raise OviPairArtifactLoadError("frozen pair visit manifest is invalid")
    expected = {
        "visit_id": visit.visit_id,
        "scan_id": visit.scan_id,
        "frame_count": visit.frame_count,
        "point_count": visit.point_count,
        "entity_count": len(visit.entities),
        "entity_point_count": visit.entity_point_count,
        "background_point_count": visit.background_point_count,
        "normal_valid_count": int(np.count_nonzero(visit.normal_valid)),
        "appearance_support_count": visit.appearance_support_count,
        "native_manifest_sha256": visit.native_manifest_sha256,
        "materialized_manifest_sha256": visit.materialized_manifest_sha256,
        "source_artifact_sha256": dict(visit.source_artifact_sha256),
        "global_alignment_application": visit.global_alignment_application,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise OviPairArtifactLoadError("frozen pair visit identity mismatch")


def restore_bound_ovi_pair_artifact(
    rebuilt_pair: OviObjectPairView, receipt: Mapping[str, object]
) -> OviObjectPairView:
    """Restore exact frozen RGB bytes after validating all source-derived fields."""

    if not isinstance(rebuilt_pair, OviObjectPairView):
        raise TypeError("rebuilt_pair must be OviObjectPairView")
    expected_pair_hash = receipt.get("pair_content_sha256")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_id") != "OVI_RESCENE_D2_PAIR_VIEW_RECEIPT_V1"
        or receipt.get("status") != "REAL_D2_PAIR_VIEW_PASS"
        or receipt.get("pair_id") != rebuilt_pair.pair_id
        or receipt.get("domain_id") != rebuilt_pair.domain_id
        or receipt.get("coordinate_frame_id") != rebuilt_pair.coordinate_frame_id
        or not isinstance(expected_pair_hash, str)
        or len(expected_pair_hash) != 64
    ):
        raise OviPairArtifactLoadError("D2 pair receipt identity mismatch")

    manifest_path, manifest_content = _bound_content(
        receipt.get("local_artifact_manifest"), label="frozen pair manifest"
    )
    manifest = _json_object(manifest_content, label="frozen pair manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_id") != "OVI_RESCENE_D2_PAIR_VIEW_V1"
        or manifest.get("status") != "D2_PAIR_ARTIFACT_PASS"
        or manifest.get("pair_id") != rebuilt_pair.pair_id
        or manifest.get("domain_id") != rebuilt_pair.domain_id
        or manifest.get("coordinate_frame_id") != rebuilt_pair.coordinate_frame_id
        or manifest.get("source_manifest_sha256")
        != rebuilt_pair.source_manifest_sha256
        or manifest.get("pair_content_sha256") != expected_pair_hash
    ):
        raise OviPairArtifactLoadError("frozen pair manifest identity mismatch")
    visits = manifest.get("visits")
    outputs = manifest.get("outputs")
    full_outputs = receipt.get("local_full_outputs")
    if (
        not isinstance(visits, list)
        or len(visits) != 2
        or not isinstance(outputs, Mapping)
        or not isinstance(full_outputs, Mapping)
        or set(full_outputs) != _FULL_OUTPUT_KEYS
    ):
        raise OviPairArtifactLoadError("frozen pair artifact schema is invalid")

    restored_visits: list[OviObjectVisitView] = []
    for rebuilt, visit_record in zip(rebuilt_pair.visits, visits, strict=True):
        _validate_visit_manifest(visit_record, rebuilt)
        role = f"t{rebuilt.visit_id}_rgb_ply"
        manifest_record = outputs.get(role)
        receipt_record = full_outputs.get(role)
        if not isinstance(manifest_record, Mapping) or not isinstance(
            receipt_record, Mapping
        ):
            raise OviPairArtifactLoadError(f"{role} binding is missing")
        path, content = _bound_content(receipt_record, label=role)
        manifest_bound_path, manifest_bound_content = _bound_content(
            manifest_record, label=role, relative_to=manifest_path.parent
        )
        if path != manifest_bound_path or content != manifest_bound_content:
            raise OviPairArtifactLoadError(f"{role} receipt binding mismatch")
        restored_visits.append(_restored_visit(rebuilt, content))

    restored = replace(rebuilt_pair, visits=tuple(restored_visits))
    if restored.content_sha256() != expected_pair_hash:
        raise OviPairArtifactLoadError("restored D2 pair content hash mismatch")
    return restored


__all__ = [
    "OviPairArtifactLoadError",
    "restore_bound_ovi_pair_artifact",
]
