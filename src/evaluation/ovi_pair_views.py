"""Source-bound dense OVI object views for real two-visit 3RScan pairs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from scripts.evaluation.build_tesse_ovimap_static_anchor import PINNED_OVIMAP_COMMIT
from src.core.data_structures import CameraIntrinsics
from src.evaluation.baselines.adapters import adapt_ovimap
from src.evaluation.baselines.contracts import RuntimeBreakdown
from src.evaluation.baselines.ovimap import bind_mesh_instances
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.ovi_surface_attributes import (
    depth_millimeters_to_meters,
    group_surface_attributes_by_color,
    project_world_points,
    sample_depth_consistent_rgb,
)
from src.oviv2.ovimap_visit_loader import (
    BoundOviArtifacts,
    load_bound_ovimap_artifacts,
)
from src.oviv2.two_visit_contracts import NeuralSampleMap, VisitMap, validate_visit_pair
from src.oviv2.two_visit_execution import build_geometric_pair_sample

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_ROLES = (
    "instance_color_log",
    "instance_mesh",
    "semantic_features",
)


class OviPairViewError(ValueError):
    """Raised when a D2 source or view violates the frozen protocol."""


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OviPairViewError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: object, label: str) -> str:
    text = _nonempty(value, label)
    if _SHA256.fullmatch(text) is None:
        raise OviPairViewError(f"{label} must be a lowercase SHA-256")
    return text


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise OviPairViewError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise OviPairViewError(f"{label} must be at least {minimum}")
    return result


def _finite_float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise OviPairViewError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = " finite and positive" if positive else " finite"
        raise OviPairViewError(f"{label} must be{qualifier}")
    return result


def _readonly(
    value: object,
    dtype: np.dtype[Any] | type,
    ndim: int,
    *,
    finite: bool = True,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if result.ndim != ndim:
        raise OviPairViewError(f"array must have {ndim} dimensions")
    if finite and result.size and not np.all(np.isfinite(result)):
        raise OviPairViewError("array must contain finite values")
    result.setflags(write=False)
    return result


def _array_record(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _canonical_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise OviPairViewError("content hash input contains a non-finite value")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _array_record(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    raise OviPairViewError(
        f"content hash input has unsupported type {type(value).__name__}"
    )


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _canonical_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OviPairViewError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _read_regular(path: Path, *, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise OviPairViewError(f"{label} is unavailable") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OviPairViewError(f"{label} must be a regular non-symlink file")
    try:
        data = absolute.read_bytes()
        after = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise OviPairViewError(f"{label} could not be read") from error
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(data) != after.st_size:
        raise OviPairViewError(f"{label} changed while being read")
    return data


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OviPairViewError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise OviPairViewError(f"{label} must contain a JSON object")
    return value


def _record(path: Path, data: bytes) -> dict[str, object]:
    return {
        "path": str(Path(os.path.abspath(os.fspath(path)))),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _safe_relative_path(value: object, *, label: str) -> PurePosixPath:
    text = _nonempty(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise OviPairViewError(f"{label} must be a safe relative POSIX path")
    return path


def _matrix4(value: object, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise OviPairViewError(f"{label} must be a finite 4x4 matrix")
    return result


def _rigid_column_pose(value: object, *, label: str) -> np.ndarray:
    pose = _matrix4(value, label=label)
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=5e-3)
        or float(np.linalg.det(rotation)) <= 0.0
    ):
        raise OviPairViewError(f"{label} must be a rigid camera-to-world matrix")
    return pose


def _rigid_row_alignment_matrix(value: object) -> np.ndarray:
    matrix = _matrix4(value, label="global alignment")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[:3, 3], 0.0, rtol=0.0, atol=1e-6)
        or not math.isclose(float(matrix[3, 3]), 1.0, rel_tol=0.0, abs_tol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=5e-3)
        or float(np.linalg.det(rotation)) <= 0.0
    ):
        raise OviPairViewError("global alignment must be a rigid row-vector transform")
    return _readonly(matrix, np.float64, 2)


def _row_alignment(value: object) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise OviPairViewError("global alignment must be a mapping")
    if (
        value.get("application") != "homogeneous_row_vector_right_multiply"
        or value.get("direction") != "rescan_row_vector_to_reference"
        or value.get("storage") != "row_major_flat_4x4"
    ):
        raise OviPairViewError("global alignment convention is invalid")
    raw = value.get("matrix")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 16:
        raise OviPairViewError("global alignment matrix must contain 16 values")
    return _rigid_row_alignment_matrix(np.asarray(raw, dtype=np.float64).reshape(4, 4))


def _normalize_normals(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(value, dtype=np.float32)
    norms = np.linalg.norm(normals, axis=1)
    valid = np.all(np.isfinite(normals), axis=1) & np.isfinite(norms) & (norms > 0.0)
    normalized = np.zeros(normals.shape, dtype=np.float32)
    normalized[valid] = normals[valid] / norms[valid, None]
    return normalized, valid


@dataclass(frozen=True, slots=True)
class OviObjectEntityView:
    """One mesh-backed OVI entity with dense-row ownership and OVI semantics."""

    visit_id: int
    entity_id: str
    source_instance_id: int
    point_indices: np.ndarray
    palette_rgb: tuple[int, int, int]
    semantic_embedding: np.ndarray | None
    semantic_label: str | None
    semantic_score: float
    observation_frame_ids: tuple[int, ...]
    observation_boxes_xyxy: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        visit_id = _integer(self.visit_id, "entity visit_id")
        if visit_id not in {0, 1}:
            raise OviPairViewError("entity visit_id must be exactly 0 or 1")
        instance_id = _integer(self.source_instance_id, "source instance ID")
        entity_id = _nonempty(self.entity_id, "entity ID")
        if entity_id != f"ovimap:{instance_id}":
            raise OviPairViewError("entity ID must bind its OVI source instance")
        raw_indices = np.asarray(self.point_indices)
        if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
            raise OviPairViewError("entity point indices must be an integer vector")
        indices = _readonly(raw_indices, np.int64, 1)
        if (
            not len(indices)
            or np.any(indices < 0)
            or len(np.unique(indices)) != len(indices)
        ):
            raise OviPairViewError("entity point indices must be non-empty and unique")
        if not np.array_equal(indices, np.sort(indices)):
            raise OviPairViewError(
                "entity point indices must preserve dense source order"
            )
        color = tuple(int(channel) for channel in self.palette_rgb)
        if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
            raise OviPairViewError("entity palette must be an RGB uint8 triple")
        embedding = None
        if self.semantic_embedding is not None:
            embedding = _readonly(self.semantic_embedding, np.float32, 1)
            if not len(embedding):
                raise OviPairViewError("semantic embedding cannot be empty")
        label = None
        if self.semantic_label is not None:
            label = _nonempty(self.semantic_label, "semantic label")
        score = _finite_float(self.semantic_score, "semantic score")
        frames = tuple(
            _integer(item, "observation frame ID")
            for item in self.observation_frame_ids
        )
        if len(frames) != len(set(frames)) or tuple(sorted(frames)) != frames:
            raise OviPairViewError("observation frame IDs must be sorted and unique")
        boxes: list[tuple[int, int, int, int]] = []
        for raw_box in self.observation_boxes_xyxy:
            if len(raw_box) != 4:
                raise OviPairViewError("observation boxes must be xyxy quadruples")
            box = tuple(
                _integer(item, "observation box coordinate") for item in raw_box
            )
            if box[0] > box[2] or box[1] > box[3]:
                raise OviPairViewError("observation box is inverted")
            boxes.append(box)
        if len(frames) != len(boxes):
            raise OviPairViewError("observation frames and boxes must align")
        object.__setattr__(self, "visit_id", visit_id)
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "source_instance_id", instance_id)
        object.__setattr__(self, "point_indices", indices)
        object.__setattr__(self, "palette_rgb", color)
        object.__setattr__(self, "semantic_embedding", embedding)
        object.__setattr__(self, "semantic_label", label)
        object.__setattr__(self, "semantic_score", score)
        object.__setattr__(self, "observation_frame_ids", frames)
        object.__setattr__(self, "observation_boxes_xyxy", tuple(boxes))

    @property
    def point_count(self) -> int:
        return len(self.point_indices)


@dataclass(frozen=True, slots=True)
class OviObjectVisitView:
    """All native OVI PLY rows plus entity ownership and camera support."""

    visit_id: int
    scan_id: str
    frame_count: int
    points_xyz: np.ndarray
    palette_rgb_uint8: np.ndarray
    normals_xyz: np.ndarray
    normal_valid: np.ndarray
    source_vertex_indices: np.ndarray
    source_frame_ids_by_target: np.ndarray
    entity_owner_indices: np.ndarray
    entities: tuple[OviObjectEntityView, ...]
    camera_rgb_uint8: np.ndarray
    appearance_valid: np.ndarray
    appearance_frame_ids: np.ndarray
    appearance_source_frame_ids: np.ndarray
    appearance_rows: np.ndarray
    appearance_columns: np.ndarray
    appearance_camera_depth_m: np.ndarray
    appearance_observed_depth_m: np.ndarray
    appearance_depth_residual_m: np.ndarray
    native_manifest_sha256: str
    materialized_manifest_sha256: str
    source_artifact_sha256: Mapping[str, str]
    global_alignment_application: str

    def __post_init__(self) -> None:
        visit_id = _integer(self.visit_id, "visit_id")
        if visit_id not in {0, 1}:
            raise OviPairViewError("visit_id must be exactly 0 or 1")
        scan_id = _nonempty(self.scan_id, "scan_id")
        frame_count = _integer(self.frame_count, "frame_count", minimum=1)
        points = _readonly(self.points_xyz, np.float32, 2)
        palette = _readonly(self.palette_rgb_uint8, np.uint8, 2)
        normals = _readonly(self.normals_xyz, np.float32, 2)
        valid_normals = _readonly(self.normal_valid, np.bool_, 1)
        source_indices = _readonly(self.source_vertex_indices, np.int64, 1)
        source_frames = _readonly(self.source_frame_ids_by_target, np.int64, 1)
        owners = _readonly(self.entity_owner_indices, np.int64, 1)
        camera_rgb = _readonly(self.camera_rgb_uint8, np.uint8, 2)
        appearance = _readonly(self.appearance_valid, np.bool_, 1)
        appearance_frames = _readonly(self.appearance_frame_ids, np.int64, 1)
        appearance_source_frames = _readonly(
            self.appearance_source_frame_ids, np.int64, 1
        )
        appearance_rows = _readonly(self.appearance_rows, np.int64, 1)
        appearance_columns = _readonly(self.appearance_columns, np.int64, 1)
        camera_depth = _readonly(
            self.appearance_camera_depth_m, np.float32, 1, finite=False
        )
        observed_depth = _readonly(
            self.appearance_observed_depth_m, np.float32, 1, finite=False
        )
        residual = _readonly(
            self.appearance_depth_residual_m, np.float32, 1, finite=False
        )
        count = len(points)
        if not count or points.shape[1:] != (3,):
            raise OviPairViewError("points_xyz must have non-empty shape (N, 3)")
        if palette.shape != (count, 3) or normals.shape != (count, 3):
            raise OviPairViewError("palette and normals must align with dense points")
        if camera_rgb.shape != (count, 3):
            raise OviPairViewError("camera RGB must align with dense points")
        vectors = (
            valid_normals,
            source_indices,
            owners,
            appearance,
            appearance_frames,
            appearance_source_frames,
            appearance_rows,
            appearance_columns,
            camera_depth,
            observed_depth,
            residual,
        )
        if any(item.shape != (count,) for item in vectors):
            raise OviPairViewError("dense visit vectors must align with points")
        if not np.array_equal(source_indices, np.arange(count, dtype=np.int64)):
            raise OviPairViewError(
                "source vertices must conserve every original PLY row"
            )
        if (
            source_frames.shape != (frame_count,)
            or np.any(source_frames < 0)
            or len(np.unique(source_frames)) != frame_count
            or not np.array_equal(source_frames, np.sort(source_frames))
        ):
            raise OviPairViewError(
                "source frame IDs must be a sorted unique target-to-source map"
            )
        entities = tuple(self.entities)
        if any(not isinstance(item, OviObjectEntityView) for item in entities):
            raise OviPairViewError("entities must contain OviObjectEntityView values")
        if any(item.visit_id != visit_id for item in entities):
            raise OviPairViewError("entity visit does not match its dense visit")
        if tuple(item.source_instance_id for item in entities) != tuple(
            sorted(item.source_instance_id for item in entities)
        ):
            raise OviPairViewError("entities must be sorted by source instance ID")
        if len({item.entity_id for item in entities}) != len(entities):
            raise OviPairViewError("entity IDs must be unique within a visit")
        if len(entities):
            if np.any((owners < -1) | (owners >= len(entities))):
                raise OviPairViewError("entity owner index is out of range")
            for owner_index, entity in enumerate(entities):
                expected = np.flatnonzero(owners == owner_index)
                if not np.array_equal(entity.point_indices, expected):
                    raise OviPairViewError(
                        "entity ownership does not match point indices"
                    )
        elif np.any(owners != -1):
            raise OviPairViewError("owner indices exist without entities")
        owned = (
            np.concatenate([item.point_indices for item in entities])
            if entities
            else np.empty(0, dtype=np.int64)
        )
        background = np.flatnonzero(owners < 0)
        if len(np.unique(owned)) != len(owned) or not np.array_equal(
            np.sort(np.concatenate((owned, background))), source_indices
        ):
            raise OviPairViewError(
                "entity ownership does not conserve dense source rows"
            )
        if np.any(appearance & (owners < 0)):
            raise OviPairViewError(
                "background points cannot enter the adapter support domain"
            )
        if np.any(appearance & ~np.isfinite(residual)) or np.any(
            residual[appearance] < 0.0
        ):
            raise OviPairViewError(
                "supported appearance residuals must be finite and nonnegative"
            )
        if np.any(~appearance & ~np.isnan(residual)):
            raise OviPairViewError("unsupported appearance residuals must be NaN")
        if (
            np.any(~np.isfinite(camera_depth[appearance]))
            or np.any(~np.isfinite(observed_depth[appearance]))
            or np.any(camera_depth[appearance] <= 0.0)
            or np.any(observed_depth[appearance] <= 0.0)
            or not np.allclose(
                np.abs(camera_depth[appearance] - observed_depth[appearance]),
                residual[appearance],
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise OviPairViewError("supported appearance depths are inconsistent")
        if np.any(~appearance & ~np.isnan(camera_depth)) or np.any(
            ~appearance & ~np.isnan(observed_depth)
        ):
            raise OviPairViewError("unsupported appearance depths must be NaN")
        for values in (
            appearance_frames,
            appearance_source_frames,
            appearance_rows,
            appearance_columns,
        ):
            if np.any(values[appearance] < 0) or np.any(values[~appearance] != -1):
                raise OviPairViewError("appearance provenance sentinel is invalid")
        if np.any(appearance_frames[appearance] >= frame_count) or not np.array_equal(
            appearance_source_frames[appearance],
            source_frames[appearance_frames[appearance]],
        ):
            raise OviPairViewError(
                "appearance target/source frame provenance is inconsistent"
            )
        native_hash = _sha256(self.native_manifest_sha256, "native manifest SHA-256")
        materialized_hash = _sha256(
            self.materialized_manifest_sha256, "materialized manifest SHA-256"
        )
        raw_artifacts = dict(self.source_artifact_sha256)
        if set(raw_artifacts) != set(_ARTIFACT_ROLES):
            raise OviPairViewError("source artifact hashes must cover all native roles")
        artifacts = MappingProxyType(
            {
                role: _sha256(raw_artifacts[role], f"{role} SHA-256")
                for role in _ARTIFACT_ROLES
            }
        )
        expected_alignment = (
            "identity_reference" if visit_id == 0 else "rescan_to_reference_once"
        )
        if self.global_alignment_application != expected_alignment:
            raise OviPairViewError("global alignment application count is invalid")
        object.__setattr__(self, "visit_id", visit_id)
        object.__setattr__(self, "scan_id", scan_id)
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(self, "points_xyz", points)
        object.__setattr__(self, "palette_rgb_uint8", palette)
        object.__setattr__(self, "normals_xyz", normals)
        object.__setattr__(self, "normal_valid", valid_normals)
        object.__setattr__(self, "source_vertex_indices", source_indices)
        object.__setattr__(self, "source_frame_ids_by_target", source_frames)
        object.__setattr__(self, "entity_owner_indices", owners)
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "camera_rgb_uint8", camera_rgb)
        object.__setattr__(self, "appearance_valid", appearance)
        object.__setattr__(self, "appearance_frame_ids", appearance_frames)
        object.__setattr__(
            self, "appearance_source_frame_ids", appearance_source_frames
        )
        object.__setattr__(self, "appearance_rows", appearance_rows)
        object.__setattr__(self, "appearance_columns", appearance_columns)
        object.__setattr__(self, "appearance_camera_depth_m", camera_depth)
        object.__setattr__(self, "appearance_observed_depth_m", observed_depth)
        object.__setattr__(self, "appearance_depth_residual_m", residual)
        object.__setattr__(self, "native_manifest_sha256", native_hash)
        object.__setattr__(self, "materialized_manifest_sha256", materialized_hash)
        object.__setattr__(self, "source_artifact_sha256", artifacts)

    @property
    def point_count(self) -> int:
        return len(self.points_xyz)

    @property
    def entity_point_count(self) -> int:
        return int(np.count_nonzero(self.entity_owner_indices >= 0))

    @property
    def background_point_count(self) -> int:
        return self.point_count - self.entity_point_count

    @property
    def appearance_support_count(self) -> int:
        return int(np.count_nonzero(self.appearance_valid))

    @property
    def appearance_support_fraction(self) -> float:
        return self.appearance_support_count / self.point_count

    @property
    def candidate_ids(self) -> tuple[tuple[int, str], ...]:
        return tuple((self.visit_id, item.entity_id) for item in self.entities)

    def content_sha256(self) -> str:
        entities = [
            {
                "visit_id": item.visit_id,
                "entity_id": item.entity_id,
                "source_instance_id": item.source_instance_id,
                "point_indices": item.point_indices,
                "palette_rgb": item.palette_rgb,
                "semantic_embedding": item.semantic_embedding,
                "semantic_label": item.semantic_label,
                "semantic_score": item.semantic_score,
                "observation_frame_ids": item.observation_frame_ids,
                "observation_boxes_xyxy": item.observation_boxes_xyxy,
            }
            for item in self.entities
        ]
        return _canonical_sha256(
            {
                "visit_id": self.visit_id,
                "scan_id": self.scan_id,
                "frame_count": self.frame_count,
                "points_xyz": self.points_xyz,
                "palette_rgb_uint8": self.palette_rgb_uint8,
                "normals_xyz": self.normals_xyz,
                "normal_valid": self.normal_valid,
                "source_vertex_indices": self.source_vertex_indices,
                "source_frame_ids_by_target": self.source_frame_ids_by_target,
                "entity_owner_indices": self.entity_owner_indices,
                "entities": entities,
                "camera_rgb_uint8": self.camera_rgb_uint8,
                "appearance_valid": self.appearance_valid,
                "appearance_frame_ids": self.appearance_frame_ids,
                "appearance_source_frame_ids": self.appearance_source_frame_ids,
                "appearance_rows": self.appearance_rows,
                "appearance_columns": self.appearance_columns,
                "appearance_camera_depth_m": np.nan_to_num(
                    self.appearance_camera_depth_m, nan=-1.0
                ),
                "appearance_observed_depth_m": np.nan_to_num(
                    self.appearance_observed_depth_m, nan=-1.0
                ),
                "appearance_depth_residual_m": np.nan_to_num(
                    self.appearance_depth_residual_m, nan=-1.0
                ),
                "native_manifest_sha256": self.native_manifest_sha256,
                "materialized_manifest_sha256": self.materialized_manifest_sha256,
                "source_artifact_sha256": self.source_artifact_sha256,
                "global_alignment_application": self.global_alignment_application,
            }
        )


@dataclass(frozen=True, slots=True)
class OviObjectPairView:
    """Two independently reconstructed OVI visits in the reference coordinates."""

    pair_id: str
    visits: tuple[OviObjectVisitView, OviObjectVisitView]
    source_manifest_sha256: str
    global_alignment: np.ndarray
    coordinate_frame_id: str = "3rscan_reference"

    def __post_init__(self) -> None:
        pair_id = _nonempty(self.pair_id, "pair_id")
        visits = tuple(self.visits)
        if len(visits) != 2 or any(
            not isinstance(item, OviObjectVisitView) for item in visits
        ):
            raise OviPairViewError("visits must contain exactly two OVI object views")
        if tuple(item.visit_id for item in visits) != (0, 1):
            raise OviPairViewError("visits must be ordered t0 then t1")
        if visits[0].scan_id == visits[1].scan_id:
            raise OviPairViewError("two-visit pair must contain distinct scans")
        source_hash = _sha256(self.source_manifest_sha256, "source manifest SHA-256")
        alignment = _rigid_row_alignment_matrix(self.global_alignment)
        if self.coordinate_frame_id != "3rscan_reference":
            raise OviPairViewError("D2 coordinate frame must be 3rscan_reference")
        object.__setattr__(self, "pair_id", pair_id)
        object.__setattr__(self, "visits", visits)
        object.__setattr__(self, "source_manifest_sha256", source_hash)
        object.__setattr__(self, "global_alignment", alignment)

    @property
    def domain_id(self) -> str:
        return "D2_OVI_RECONSTRUCTION"

    @property
    def candidate_ids(
        self,
    ) -> tuple[tuple[tuple[int, str], ...], tuple[tuple[int, str], ...]]:
        return (self.visits[0].candidate_ids, self.visits[1].candidate_ids)

    def content_sha256(self) -> str:
        return _canonical_sha256(
            {
                "pair_id": self.pair_id,
                "domain_id": self.domain_id,
                "coordinate_frame_id": self.coordinate_frame_id,
                "source_manifest_sha256": self.source_manifest_sha256,
                "global_alignment": self.global_alignment,
                "visit_sha256": [item.content_sha256() for item in self.visits],
            }
        )

    def to_visit_maps(
        self, *, supported_only: bool = False
    ) -> tuple[VisitMap, VisitMap]:
        if type(supported_only) is not bool:
            raise OviPairViewError("supported_only must be boolean")
        visit_maps: list[VisitMap] = []
        frame_offset = 0
        for visit in self.visits:
            entities: list[EntityPrediction] = []
            for item in visit.entities:
                selected_indices = item.point_indices
                if supported_only:
                    selected_indices = selected_indices[
                        visit.appearance_valid[selected_indices]
                    ]
                local_first = (
                    item.observation_frame_ids[0] if item.observation_frame_ids else 0
                )
                local_last = (
                    item.observation_frame_ids[-1]
                    if item.observation_frame_ids
                    else visit.frame_count - 1
                )
                support_count = int(
                    np.count_nonzero(visit.appearance_valid[item.point_indices])
                )
                entities.append(
                    EntityPrediction(
                        entity_id=item.entity_id,
                        points_xyz=visit.points_xyz[selected_indices],
                        semantic_embedding=item.semantic_embedding,
                        semantic_label=item.semantic_label,
                        semantic_score=item.semantic_score,
                        lifecycle_state="active",
                        first_seen=float(frame_offset + local_first),
                        last_seen=float(frame_offset + local_last),
                        metadata={
                            "visit_id": visit.visit_id,
                            "source_instance_id": item.source_instance_id,
                            "instance_color_rgb": list(item.palette_rgb),
                            "observation_count": len(item.observation_frame_ids),
                            "observation_frame_ids": list(item.observation_frame_ids),
                            "observation_boxes_xyxy": [
                                list(box) for box in item.observation_boxes_xyxy
                            ],
                            "source_vertex_index_sha256": _array_record(
                                selected_indices
                            )["sha256"],
                            "full_source_vertex_index_sha256": _array_record(
                                item.point_indices
                            )["sha256"],
                            "appearance_support_count": support_count,
                            "appearance_support_fraction": support_count
                            / item.point_count,
                            "geometry_authority": f"ovi_t{visit.visit_id}",
                            "semantic_authority": f"ovi_t{visit.visit_id}",
                            "native_manifest_sha256": visit.native_manifest_sha256,
                            "semantic_label_source": "method_output.feat",
                        },
                    )
                )
            background = visit.points_xyz[visit.entity_owner_indices < 0]
            snapshot = MapSnapshot(
                method=(
                    f"OVI-MAP D2 supported t{visit.visit_id}"
                    if supported_only
                    else f"OVI-MAP D2 full t{visit.visit_id}"
                ),
                scene_id=self.pair_id,
                timestamp=float(frame_offset + visit.frame_count - 1),
                entities=entities,
                background_xyz=(
                    None if supported_only or not len(background) else background
                ),
                scope="current",
            )
            visit_maps.append(
                VisitMap(
                    visit_id=visit.visit_id,
                    snapshot=snapshot,
                    coordinate_frame_id=self.coordinate_frame_id,
                    source_manifest_sha256=self.source_manifest_sha256,
                    map_voxel_size_m=0.01,
                    observed_frame_start=frame_offset,
                    observed_frame_end=frame_offset + visit.frame_count - 1,
                )
            )
            frame_offset += visit.frame_count
        result = (visit_maps[0], visit_maps[1])
        validate_visit_pair(*result)
        return result

    def geometric_sample(self, *, neural_voxel_size_m: float) -> NeuralSampleMap:
        t0, t1 = self.to_visit_maps(supported_only=True)
        return build_geometric_pair_sample(
            t0, t1, neural_voxel_size_m=neural_voxel_size_m
        )


@dataclass(frozen=True, slots=True)
class _MaterializedVisit:
    path: Path
    manifest_bytes: bytes
    payload: Mapping[str, Any]
    root: Path
    files: Mapping[str, bytes]
    depth_intrinsic: np.ndarray
    color_intrinsic: np.ndarray
    depth_width: int
    depth_height: int
    color_width: int
    color_height: int
    frame_count: int
    source_frame_ids_by_target: np.ndarray

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.manifest_bytes).hexdigest()

    def assert_unchanged(self) -> None:
        if (
            _read_regular(self.path, label="materialized manifest")
            != self.manifest_bytes
        ):
            raise OviPairViewError(
                "materialized manifest changed during D2 construction"
            )
        for relative, expected in self.files.items():
            observed = _read_regular(
                self.root / relative, label=f"materialized {relative}"
            )
            if observed != expected:
                raise OviPairViewError(
                    f"materialized output changed during D2 construction: {relative}"
                )


def _validate_materialized_manifest(
    path: Path, *, pair_id: str, scan_id: str, visit_id: int
) -> _MaterializedVisit:
    manifest_path = Path(os.path.abspath(os.fspath(path)))
    manifest_bytes = _read_regular(manifest_path, label="materialized manifest")
    payload = _json_object(manifest_bytes, label="materialized manifest")
    frame_count = _integer(
        payload.get("frame_count"), "materialized frame_count", minimum=1
    )
    source_count = _integer(
        payload.get("source_frame_count"), "materialized source_frame_count", minimum=1
    )
    frame_step = _integer(
        payload.get("frame_step"), "materialized frame_step", minimum=1
    )
    expected_map = [
        {"source_frame_id": source_id, "target_frame_id": target_id}
        for target_id, source_id in enumerate(range(0, source_count, frame_step))
    ]
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_id") != "RSCAN_OVI_VISIT_V1"
        or payload.get("status") != "MATERIALIZED_INPUT_PASS"
        or payload.get("dataset") != "3RScan"
        or payload.get("dataset_adapter") != "scannet_nyu"
        or payload.get("scan_id") != scan_id
        or payload.get("pair_id") != pair_id
        or payload.get("visit_index") != visit_id
        or payload.get("frame_map") != expected_map
        or frame_count != len(expected_map)
        or float(payload.get("depth_shift", -1.0)) != 1000.0
    ):
        raise OviPairViewError("materialized manifest identity mismatch")
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise OviPairViewError("materialized dimensions are missing")
    try:
        color_dimensions = dimensions["color"]
        depth_dimensions = dimensions["depth"]
    except KeyError as error:
        raise OviPairViewError("materialized dimensions are incomplete") from error
    if not isinstance(color_dimensions, Mapping) or not isinstance(
        depth_dimensions, Mapping
    ):
        raise OviPairViewError("materialized dimensions are invalid")
    color_width = _integer(color_dimensions.get("width"), "color width", minimum=1)
    color_height = _integer(color_dimensions.get("height"), "color height", minimum=1)
    depth_width = _integer(depth_dimensions.get("width"), "depth width", minimum=1)
    depth_height = _integer(depth_dimensions.get("height"), "depth height", minimum=1)
    calibration = payload.get("calibration")
    if not isinstance(calibration, Mapping):
        raise OviPairViewError("materialized calibration is missing")
    color_intrinsic = _matrix4(
        calibration.get("color_intrinsic"), label="color intrinsic"
    )
    depth_intrinsic = _matrix4(
        calibration.get("depth_intrinsic"), label="depth intrinsic"
    )
    for label, intrinsic in (("color", color_intrinsic), ("depth", depth_intrinsic)):
        if (
            intrinsic[0, 0] <= 0.0
            or intrinsic[1, 1] <= 0.0
            or not np.allclose(
                intrinsic[2:],
                ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                rtol=0.0,
                atol=1e-8,
            )
        ):
            raise OviPairViewError(f"{label} intrinsic matrix is invalid")
    for label in ("color_extrinsic", "depth_extrinsic"):
        if not np.allclose(
            _matrix4(calibration.get(label), label=label),
            np.eye(4),
            rtol=0.0,
            atol=1e-9,
        ):
            raise OviPairViewError(
                "D2 currently requires identity RGB-depth extrinsics"
            )
    preflight = payload.get("preflight")
    if not isinstance(preflight, Mapping) or (
        preflight.get("depth_encoding") != "uint16"
        or float(preflight.get("depth_units_per_meter", -1.0)) != 1000.0
        or preflight.get("rgb_depth_registration") != "K_depth @ inv(K_color)"
    ):
        raise OviPairViewError("materialized preflight contract mismatch")
    sequence_binding = payload.get("sequence_zip")
    if not isinstance(sequence_binding, Mapping) or set(sequence_binding) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise OviPairViewError("sequence archive binding schema is invalid")
    sequence_path_value = sequence_binding.get("path")
    if (
        not isinstance(sequence_path_value, str)
        or not Path(sequence_path_value).is_absolute()
    ):
        raise OviPairViewError("sequence archive binding path must be absolute")
    sequence_bytes = _read_regular(
        Path(sequence_path_value), label="bound sequence archive"
    )
    if dict(sequence_binding) != _record(Path(sequence_path_value), sequence_bytes):
        raise OviPairViewError("sequence archive binding mismatch")

    output_tree = payload.get("output_tree")
    if not isinstance(output_tree, Mapping) or set(output_tree) != {
        "sha256",
        "byte_count",
        "file_count",
        "files",
    }:
        raise OviPairViewError("materialized output tree schema is invalid")
    raw_files = output_tree.get("files")
    if not isinstance(raw_files, list):
        raise OviPairViewError("materialized output files must be a list")
    expected_paths = {
        *(f"color/{frame_id}.jpg" for frame_id in range(frame_count)),
        *(f"depth/{frame_id}.png" for frame_id in range(frame_count)),
        *(f"pose/{frame_id}.txt" for frame_id in range(frame_count)),
        "intrinsic/intrinsic_color.txt",
        "intrinsic/intrinsic_depth.txt",
    }
    root = manifest_path.parent
    files: dict[str, bytes] = {}
    records: list[dict[str, object]] = []
    for raw_record in raw_files:
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "path",
            "sha256",
            "byte_count",
        }:
            raise OviPairViewError("materialized output file binding schema is invalid")
        relative = _safe_relative_path(
            raw_record.get("path"), label="output path"
        ).as_posix()
        if relative in files:
            raise OviPairViewError("materialized output path is duplicated")
        data = _read_regular(root / relative, label=f"materialized {relative}")
        observed = {
            "path": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_count": len(data),
        }
        if dict(raw_record) != observed:
            raise OviPairViewError(f"materialized output binding mismatch: {relative}")
        files[relative] = data
        records.append(observed)
    if set(files) != expected_paths:
        raise OviPairViewError("materialized output tree has missing or extra files")
    records.sort(key=lambda item: str(item["path"]))
    tree_hash = hashlib.sha256()
    for record in records:
        tree_hash.update(str(record["path"]).encode("utf-8"))
        tree_hash.update(b"\0")
        tree_hash.update(str(record["sha256"]).encode("ascii"))
        tree_hash.update(b"\n")
    observed_tree = {
        "sha256": tree_hash.hexdigest(),
        "byte_count": sum(int(item["byte_count"]) for item in records),
        "file_count": len(records),
        "files": records,
    }
    if dict(output_tree) != observed_tree:
        raise OviPairViewError("materialized output tree binding mismatch")
    for label, expected in (
        ("intrinsic/intrinsic_color.txt", color_intrinsic),
        ("intrinsic/intrinsic_depth.txt", depth_intrinsic),
    ):
        try:
            observed = np.fromstring(files[label].decode("ascii"), sep=" ").reshape(
                4, 4
            )
        except (UnicodeDecodeError, ValueError) as error:
            raise OviPairViewError(
                f"materialized {label} is not a 4x4 matrix"
            ) from error
        if not np.allclose(observed, expected, rtol=0.0, atol=1e-9):
            raise OviPairViewError(f"materialized {label} differs from its manifest")
    return _MaterializedVisit(
        path=manifest_path,
        manifest_bytes=manifest_bytes,
        payload=payload,
        root=root,
        files=MappingProxyType(files),
        depth_intrinsic=_readonly(depth_intrinsic, np.float64, 2),
        color_intrinsic=_readonly(color_intrinsic, np.float64, 2),
        depth_width=depth_width,
        depth_height=depth_height,
        color_width=color_width,
        color_height=color_height,
        frame_count=frame_count,
        source_frame_ids_by_target=_readonly(
            [item["source_frame_id"] for item in expected_map], np.int64, 1
        ),
    )


def _validate_native_identity(
    loaded: BoundOviArtifacts,
    materialized: _MaterializedVisit,
    *,
    scan_id: str,
) -> None:
    payload = loaded.native_payload
    expected_frames = list(range(materialized.frame_count))
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "PASS"
        or payload.get("state") != "MAPPING_PASS"
        or payload.get("stage") != "mapping"
        or payload.get("scene") != scan_id
        or payload.get("dataset") != "scannet_nyu"
        or payload.get("frame_ids") != expected_frames
    ):
        raise OviPairViewError("native OVI manifest identity mismatch")
    preflight = payload.get("preflight")
    if not isinstance(preflight, Mapping) or (
        preflight.get("status") != "PASS"
        or preflight.get("scene") != scan_id
        or preflight.get("dataset") != "scannet_nyu"
        or preflight.get("frame_ids") != expected_frames
        or float(preflight.get("depth_shift", -1.0)) != 1000.0
    ):
        raise OviPairViewError("native OVI preflight identity mismatch")
    expected_materialized = {
        "path": str(materialized.path),
        "sha256": materialized.manifest_sha256,
        "size_bytes": len(materialized.manifest_bytes),
    }
    if preflight.get("materialized_manifest") != expected_materialized:
        raise OviPairViewError("native OVI materialized-manifest binding mismatch")


def _decode_frame(
    materialized: _MaterializedVisit, frame_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        rgb_bgr = cv2.imdecode(
            np.frombuffer(materialized.files[f"color/{frame_id}.jpg"], dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        depth = cv2.imdecode(
            np.frombuffer(materialized.files[f"depth/{frame_id}.png"], dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
    except KeyError as error:
        raise OviPairViewError(f"materialized frame {frame_id} is missing") from error
    if rgb_bgr is None or rgb_bgr.shape != (
        materialized.color_height,
        materialized.color_width,
        3,
    ):
        raise OviPairViewError(f"materialized frame {frame_id} RGB shape mismatch")
    if (
        depth is None
        or depth.dtype != np.uint16
        or depth.shape
        != (
            materialized.depth_height,
            materialized.depth_width,
        )
    ):
        raise OviPairViewError(f"materialized frame {frame_id} depth shape mismatch")
    homography = materialized.depth_intrinsic[:3, :3] @ np.linalg.inv(
        materialized.color_intrinsic[:3, :3]
    )
    aligned_bgr = cv2.warpPerspective(
        rgb_bgr,
        homography,
        (materialized.depth_width, materialized.depth_height),
    )
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    try:
        values = np.fromstring(
            materialized.files[f"pose/{frame_id}.txt"].decode("ascii"), sep=" "
        )
    except UnicodeDecodeError as error:
        raise OviPairViewError(
            f"materialized frame {frame_id} pose is not ASCII"
        ) from error
    if len(values) != 16:
        raise OviPairViewError(f"materialized frame {frame_id} pose is not 4x4")
    pose = _rigid_column_pose(values.reshape(4, 4), label=f"frame {frame_id} pose")
    return rgb, depth_millimeters_to_meters(depth), pose


def _observations(
    record: Mapping[str, Any], *, frame_count: int, depth_width: int, depth_height: int
) -> tuple[tuple[int, ...], tuple[tuple[int, int, int, int], ...]]:
    raw_frames = record.get("frame_id", ())
    raw_boxes = record.get("box_2d", ())
    if isinstance(raw_frames, (str, bytes)) or not isinstance(raw_frames, Sequence):
        raise OviPairViewError("OVI observation frame IDs must be a sequence")
    if isinstance(raw_boxes, (str, bytes)) or not isinstance(raw_boxes, Sequence):
        raise OviPairViewError("OVI observation boxes must be a sequence")
    if len(raw_frames) != len(raw_boxes):
        raise OviPairViewError("OVI observation frames and boxes differ in length")
    by_frame: dict[int, tuple[int, int, int, int]] = {}
    for raw_frame, raw_box in zip(raw_frames, raw_boxes, strict=True):
        frame_id = _integer(raw_frame, "OVI observation frame ID")
        if frame_id >= frame_count or frame_id in by_frame:
            raise OviPairViewError(
                "OVI observation frame ID is duplicate or out of range"
            )
        if (
            isinstance(raw_box, (str, bytes))
            or not isinstance(raw_box, Sequence)
            or len(raw_box) != 4
        ):
            raise OviPairViewError("OVI observation box must be an xyxy sequence")
        box = tuple(
            _integer(item, "OVI observation box coordinate") for item in raw_box
        )
        if (
            box[0] > box[2]
            or box[1] > box[3]
            or box[2] >= depth_width
            or box[3] >= depth_height
        ):
            raise OviPairViewError("OVI observation box is outside the depth image")
        by_frame[frame_id] = box
    frames = tuple(sorted(by_frame))
    return frames, tuple(by_frame[item] for item in frames)


def _load_dense_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData

        vertices = PlyData.read(path, mmap="c")["vertex"]
    except Exception as error:
        raise OviPairViewError("native OVI instance PLY is unreadable") from error
    names = set(vertices.data.dtype.names or ())
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
    }
    if not required.issubset(names):
        raise OviPairViewError(
            f"native OVI instance PLY lacks properties: {sorted(required - names)}"
        )
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float32, copy=False
    )
    normals = np.column_stack(
        (vertices["normal_x"], vertices["normal_y"], vertices["normal_z"])
    ).astype(np.float32, copy=False)
    colors = np.column_stack(
        (vertices["red"], vertices["green"], vertices["blue"])
    ).astype(np.uint8, copy=False)
    if not len(points) or not np.all(np.isfinite(points)):
        raise OviPairViewError("native OVI instance PLY has no finite surface")
    return points, normals, colors


def _transform_surface(
    points: np.ndarray, normals: np.ndarray, alignment: np.ndarray, *, visit_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized, normal_valid = _normalize_normals(normals)
    if visit_id == 0:
        return points.astype(np.float32, copy=True), normalized, normal_valid
    transformed_points = (
        np.asarray(points, dtype=np.float64) @ alignment[:3, :3] + alignment[3, :3]
    )
    transformed_normals = np.asarray(normalized, dtype=np.float64) @ alignment[:3, :3]
    transformed_normals, transformed_valid = _normalize_normals(transformed_normals)
    return (
        transformed_points.astype(np.float32),
        transformed_normals,
        normal_valid & transformed_valid,
    )


def _build_visit(
    *,
    visit_id: int,
    scan_id: str,
    pair_id: str,
    native_manifest: Path,
    materialized_manifest: Path,
    alignment: np.ndarray,
    depth_tolerance_m: float,
    minimum_valid_neighbours: int,
) -> OviObjectVisitView:
    materialized = _validate_materialized_manifest(
        materialized_manifest, pair_id=pair_id, scan_id=scan_id, visit_id=visit_id
    )
    try:
        loaded = load_bound_ovimap_artifacts(native_manifest)
    except (OSError, TypeError, ValueError) as error:
        raise OviPairViewError(
            f"native OVI artifact binding failed: {error}"
        ) from error
    _validate_native_identity(loaded, materialized, scan_id=scan_id)
    mesh_path = loaded.artifacts["instance_mesh"][0]
    points_local, raw_normals, palette = _load_dense_ply(mesh_path)
    groups = group_surface_attributes_by_color(
        points_local,
        palette,
        raw_normals,
        np.arange(len(points_local), dtype=np.int64),
        loaded.colors_by_instance.values(),
    )
    points_by_color = {
        color: _transform_surface(
            group.points_xyz, group.normals_xyz, alignment, visit_id=visit_id
        )[0]
        for color, group in groups.items()
    }
    instances = bind_mesh_instances(
        loaded.semantic_instances, loaded.colors_by_instance, points_by_color
    )
    if not instances:
        raise OviPairViewError("native OVI visit contains no mesh-backed entities")
    artifact = adapt_ovimap(
        instances,
        points_by_color=points_by_color,
        scene_id=pair_id,
        timestamp=float(materialized.frame_count - 1),
        upstream_commit=PINNED_OVIMAP_COMMIT,
        runtime=RuntimeBreakdown(frame_count=materialized.frame_count),
        semantic_label_source="method_output.feat",
        protocol_notes=(
            "visit reconstructed independently from bound 3RScan RGB-D",
            "dense geometry remains native OVI-MAP surface output",
        ),
    )
    snapshot_entities = {item.entity_id: item for item in artifact.snapshot.entities}
    points, normals, normal_valid = _transform_surface(
        points_local, raw_normals, alignment, visit_id=visit_id
    )
    owners = np.full(len(points), -1, dtype=np.int64)
    camera_rgb = np.zeros((len(points), 3), dtype=np.uint8)
    appearance_valid = np.zeros(len(points), dtype=bool)
    appearance_frames = np.full(len(points), -1, dtype=np.int64)
    appearance_source_frames = np.full(len(points), -1, dtype=np.int64)
    appearance_rows = np.full(len(points), -1, dtype=np.int64)
    appearance_columns = np.full(len(points), -1, dtype=np.int64)
    appearance_camera_depth = np.full(len(points), np.nan, dtype=np.float32)
    appearance_observed_depth = np.full(len(points), np.nan, dtype=np.float32)
    appearance_residual = np.full(len(points), np.nan, dtype=np.float32)
    best_residual = np.full(len(points), np.inf, dtype=np.float64)
    frame_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    entities: list[OviObjectEntityView] = []
    intrinsics = CameraIntrinsics(
        fx=float(materialized.depth_intrinsic[0, 0]),
        fy=float(materialized.depth_intrinsic[1, 1]),
        cx=float(materialized.depth_intrinsic[0, 2]),
        cy=float(materialized.depth_intrinsic[1, 2]),
        width=materialized.depth_width,
        height=materialized.depth_height,
    )
    for owner_index, instance_id in enumerate(sorted(instances)):
        color = tuple(int(item) for item in loaded.colors_by_instance[instance_id])
        group = groups[color]
        dense_indices = np.asarray(group.original_vertex_indices, dtype=np.int64)
        if np.any(owners[dense_indices] != -1):
            raise OviPairViewError("native OVI palette groups overlap")
        owners[dense_indices] = owner_index
        record = loaded.semantic_instances.get(instance_id, {})
        frames, boxes = _observations(
            record,
            frame_count=materialized.frame_count,
            depth_width=materialized.depth_width,
            depth_height=materialized.depth_height,
        )
        for frame_id, box in zip(frames, boxes, strict=True):
            if frame_id not in frame_cache:
                frame_cache[frame_id] = _decode_frame(materialized, frame_id)
            rgb, depth, pose = frame_cache[frame_id]
            projection = project_world_points(group.points_xyz, pose, intrinsics)
            support = sample_depth_consistent_rgb(
                projection,
                rgb,
                depth,
                depth_tolerance_m=depth_tolerance_m,
                candidate_visit_ids=np.full(
                    len(group.points_xyz), visit_id, dtype=np.int8
                ),
                frame_visit_id=visit_id,
                rgb_source="camera_rgb",
                minimum_valid_neighbours=minimum_valid_neighbours,
            )
            if not len(support.candidate_indices):
                continue
            x0, y0, x1, y1 = box
            in_box = (
                (support.columns >= x0)
                & (support.columns <= x1)
                & (support.rows >= y0)
                & (support.rows <= y1)
            )
            local_indices = support.candidate_indices[in_box]
            if not len(local_indices):
                continue
            selected_residual = support.depth_residual_m[in_box]
            global_indices = dense_indices[local_indices]
            improve = selected_residual < best_residual[global_indices]
            if not np.any(improve):
                continue
            local_indices = local_indices[improve]
            global_indices = global_indices[improve]
            selected_residual = selected_residual[improve]
            support_positions = np.flatnonzero(in_box)[improve]
            best_residual[global_indices] = selected_residual
            appearance_valid[global_indices] = True
            appearance_frames[global_indices] = frame_id
            appearance_source_frames[global_indices] = (
                materialized.source_frame_ids_by_target[frame_id]
            )
            appearance_rows[global_indices] = support.rows[support_positions]
            appearance_columns[global_indices] = support.columns[support_positions]
            appearance_camera_depth[global_indices] = support.camera_depth_m[
                support_positions
            ]
            appearance_observed_depth[global_indices] = support.observed_depth_m[
                support_positions
            ]
            appearance_residual[global_indices] = selected_residual
            camera_rgb[global_indices] = support.rgb_uint8[support_positions]
        entity_id = f"ovimap:{instance_id}"
        snapshot = snapshot_entities[entity_id]
        entities.append(
            OviObjectEntityView(
                visit_id=visit_id,
                entity_id=entity_id,
                source_instance_id=instance_id,
                point_indices=dense_indices,
                palette_rgb=color,
                semantic_embedding=snapshot.semantic_embedding,
                semantic_label=snapshot.semantic_label,
                semantic_score=snapshot.semantic_score,
                observation_frame_ids=frames,
                observation_boxes_xyxy=boxes,
            )
        )
    loaded.assert_unchanged()
    materialized.assert_unchanged()
    return OviObjectVisitView(
        visit_id=visit_id,
        scan_id=scan_id,
        frame_count=materialized.frame_count,
        points_xyz=points,
        palette_rgb_uint8=palette,
        normals_xyz=normals,
        normal_valid=normal_valid,
        source_vertex_indices=np.arange(len(points), dtype=np.int64),
        source_frame_ids_by_target=materialized.source_frame_ids_by_target,
        entity_owner_indices=owners,
        entities=tuple(entities),
        camera_rgb_uint8=camera_rgb,
        appearance_valid=appearance_valid,
        appearance_frame_ids=appearance_frames,
        appearance_source_frame_ids=appearance_source_frames,
        appearance_rows=appearance_rows,
        appearance_columns=appearance_columns,
        appearance_camera_depth_m=appearance_camera_depth,
        appearance_observed_depth_m=appearance_observed_depth,
        appearance_depth_residual_m=appearance_residual,
        native_manifest_sha256=loaded.native_manifest_sha256,
        materialized_manifest_sha256=materialized.manifest_sha256,
        source_artifact_sha256={
            role: str(loaded.artifacts[role][1]["sha256"]) for role in _ARTIFACT_ROLES
        },
        global_alignment_application=(
            "identity_reference" if visit_id == 0 else "rescan_to_reference_once"
        ),
    )


def build_ovi_object_pair_view(
    *,
    pair_record: object,
    source_manifest_sha256: str,
    native_manifests: Sequence[Path],
    materialized_manifests: Sequence[Path],
    depth_tolerance_m: float = 0.05,
    minimum_valid_neighbours: int = 5,
    processed_visits: object | None = None,
    support_masks: object | None = None,
) -> OviObjectPairView:
    """Construct D2 exclusively from two independently mapped native OVI visits."""

    if processed_visits is not None or support_masks is not None:
        raise OviPairViewError(
            "D2 must use native OVI artifacts, never processed visits or support masks"
        )
    if not isinstance(pair_record, Mapping):
        raise OviPairViewError("pair record must be a mapping")
    pair_id = _nonempty(pair_record.get("pair_id"), "pair_id")
    sessions = pair_record.get("sessions")
    if (
        isinstance(sessions, (str, bytes))
        or not isinstance(sessions, Sequence)
        or len(sessions) != 2
    ):
        raise OviPairViewError("pair sessions must contain t0 and t1")
    scan_ids: list[str] = []
    for visit_id, session in enumerate(sessions):
        if not isinstance(session, Mapping) or session.get("visit_index") != visit_id:
            raise OviPairViewError("pair sessions must be ordered t0 then t1")
        scan_ids.append(_nonempty(session.get("scan_id"), "session scan_id"))
    if pair_record.get("environment_id") != scan_ids[0]:
        raise OviPairViewError("pair environment must name the reference scan")
    common = pair_record.get("common_method_inputs")
    if not isinstance(common, Mapping):
        raise OviPairViewError("pair common method inputs are missing")
    alignment = _row_alignment(common.get("global_alignment"))
    if isinstance(native_manifests, (str, bytes)) or len(native_manifests) != 2:
        raise OviPairViewError("native manifests must contain t0 and t1")
    if (
        isinstance(materialized_manifests, (str, bytes))
        or len(materialized_manifests) != 2
    ):
        raise OviPairViewError("materialized manifests must contain t0 and t1")
    source_hash = _sha256(source_manifest_sha256, "source manifest SHA-256")
    tolerance = _finite_float(depth_tolerance_m, "depth tolerance", positive=True)
    neighbours = _integer(
        minimum_valid_neighbours, "minimum valid neighbours", minimum=1
    )
    if neighbours > 9:
        raise OviPairViewError("minimum valid neighbours cannot exceed 9")
    visits = tuple(
        _build_visit(
            visit_id=visit_id,
            scan_id=scan_ids[visit_id],
            pair_id=pair_id,
            native_manifest=Path(native_manifests[visit_id]),
            materialized_manifest=Path(materialized_manifests[visit_id]),
            alignment=alignment,
            depth_tolerance_m=tolerance,
            minimum_valid_neighbours=neighbours,
        )
        for visit_id in (0, 1)
    )
    return OviObjectPairView(
        pair_id=pair_id,
        visits=visits,  # type: ignore[arg-type]
        source_manifest_sha256=source_hash,
        global_alignment=alignment,
    )


__all__ = [
    "OviObjectEntityView",
    "OviObjectPairView",
    "OviObjectVisitView",
    "OviPairViewError",
    "build_ovi_object_pair_view",
]
