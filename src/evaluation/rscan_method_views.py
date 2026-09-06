"""Ground-truth-free method views for controlled two-visit 3RScan runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.two_visit_contracts import NeuralSampleMap, VisitMap
from src.oviv2.two_visit_execution import build_geometric_pair_sample

DomainId = Literal[
    "D0_NATIVE_PROCESSED",
    "D1_NATIVE_SENSOR_SUPPORT",
    "D2_OVI_RECONSTRUCTION",
]
DomainStatus = Literal["PASS", "MISSING_ASSET"]
ForwardMode = Literal[
    "deterministic_pair", "independent_single_visit", "joint_two_visit"
]

_DOMAINS = (
    "D0_NATIVE_PROCESSED",
    "D1_NATIVE_SENSOR_SUPPORT",
    "D2_OVI_RECONSTRUCTION",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class RScanMethodViewError(ValueError):
    """Raised when a method-visible 3RScan input violates its contract."""


def _readonly(value: object, dtype: np.dtype | type, ndim: int) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if result.ndim != ndim:
        raise RScanMethodViewError(f"array must have {ndim} dimensions")
    result.setflags(write=False)
    return result


def _array_record(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RScanMethodViewError(f"{label} must be a lowercase SHA-256")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RScanMethodViewError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ProcessedVisitView:
    """One method-visible visit; semantic and instance GT are absent by design."""

    visit_id: int
    scan_id: str
    points_xyz: np.ndarray
    rgb: np.ndarray
    normals_xyz: np.ndarray
    segment_ids: np.ndarray
    source_point_indices: np.ndarray
    feature_schema: str = "xyz_rgb_normal_mesh_segment"

    def __post_init__(self) -> None:
        if type(self.visit_id) is not int or self.visit_id not in {0, 1}:
            raise RScanMethodViewError("visit_id must be exactly 0 or 1")
        scan_id = _nonempty(self.scan_id, "scan_id")
        points = _readonly(self.points_xyz, np.float32, 2)
        rgb = _readonly(self.rgb, np.float32, 2)
        normals = _readonly(self.normals_xyz, np.float32, 2)
        raw_segments = np.asarray(self.segment_ids)
        raw_indices = np.asarray(self.source_point_indices)
        if points.shape[1:] != (3,) or not len(points):
            raise RScanMethodViewError("points_xyz must have non-empty shape (N, 3)")
        if rgb.shape != points.shape or normals.shape != points.shape:
            raise RScanMethodViewError("RGB and normal arrays must match XYZ")
        if not all(np.all(np.isfinite(value)) for value in (points, rgb, normals)):
            raise RScanMethodViewError("method feature arrays must be finite")
        if np.any(rgb < 0.0) or np.any(rgb > 1.0):
            raise RScanMethodViewError("processed RGB must remain in [0, 1]")
        if raw_segments.shape != (len(points),) or not np.issubdtype(
            raw_segments.dtype, np.integer
        ):
            raise RScanMethodViewError("mesh segment IDs must be an integer vector")
        if raw_indices.shape != (len(points),) or not np.issubdtype(
            raw_indices.dtype, np.integer
        ):
            raise RScanMethodViewError("source point indices must be an integer vector")
        segments = _readonly(raw_segments, np.int64, 1)
        indices = _readonly(raw_indices, np.int64, 1)
        if np.any(segments < 0):
            raise RScanMethodViewError("mesh segment IDs must be non-negative")
        if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
            raise RScanMethodViewError("source point indices must be unique and non-negative")
        if not np.array_equal(indices, np.sort(indices)):
            raise RScanMethodViewError("source point indices must preserve source order")
        if self.feature_schema != "xyz_rgb_normal_mesh_segment":
            raise RScanMethodViewError("method feature schema cannot include GT columns")
        object.__setattr__(self, "scan_id", scan_id)
        object.__setattr__(self, "points_xyz", points)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "normals_xyz", normals)
        object.__setattr__(self, "segment_ids", segments)
        object.__setattr__(self, "source_point_indices", indices)

    @property
    def point_count(self) -> int:
        return len(self.points_xyz)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            f"segment:{int(value):06d}" for value in np.unique(self.segment_ids)
        )

    def method_tensor_sha256(self) -> str:
        return _canonical_sha256(
            {
                "visit_id": self.visit_id,
                "scan_id": self.scan_id,
                "feature_schema": self.feature_schema,
                "arrays": {
                    "points_xyz": _array_record(self.points_xyz),
                    "rgb": _array_record(self.rgb),
                    "normals_xyz": _array_record(self.normals_xyz),
                    "segment_ids": _array_record(self.segment_ids),
                    "source_point_indices": _array_record(self.source_point_indices),
                },
            }
        )


@dataclass(frozen=True, slots=True)
class RScanMethodPairView:
    pair_id: str
    domain_id: DomainId
    visits: tuple[ProcessedVisitView, ProcessedVisitView]
    source_manifest_sha256: str
    parent_method_tensor_sha256: str | None = None
    coordinate_frame_id: str = "3rscan_reference_pre_aligned"
    global_alignment_application: str = "already_materialized_by_preprocessing"

    def __post_init__(self) -> None:
        pair_id = _nonempty(self.pair_id, "pair_id")
        if self.domain_id not in _DOMAINS:
            raise RScanMethodViewError("domain_id is unsupported")
        if not isinstance(self.visits, tuple) or len(self.visits) != 2:
            raise RScanMethodViewError("visits must contain exactly t0 and t1")
        if tuple(visit.visit_id for visit in self.visits) != (0, 1):
            raise RScanMethodViewError("visits must be ordered t0 then t1")
        if self.visits[0].scan_id == self.visits[1].scan_id:
            raise RScanMethodViewError("two-visit input must contain distinct scans")
        source = _sha256(self.source_manifest_sha256, "source manifest SHA-256")
        parent = self.parent_method_tensor_sha256
        if self.domain_id == "D0_NATIVE_PROCESSED":
            if parent is not None:
                raise RScanMethodViewError("D0 cannot declare a parent support view")
        else:
            parent = _sha256(parent, "parent method tensor SHA-256")
        if self.coordinate_frame_id != "3rscan_reference_pre_aligned":
            raise RScanMethodViewError("processed coordinate frame declaration is invalid")
        if self.global_alignment_application != "already_materialized_by_preprocessing":
            raise RScanMethodViewError("global alignment would be applied ambiguously")
        object.__setattr__(self, "pair_id", pair_id)
        object.__setattr__(self, "source_manifest_sha256", source)
        object.__setattr__(self, "parent_method_tensor_sha256", parent)

    def method_tensor_sha256(self) -> str:
        return _canonical_sha256(
            {
                "pair_id": self.pair_id,
                "coordinate_frame_id": self.coordinate_frame_id,
                "global_alignment_application": self.global_alignment_application,
                "visit_method_tensor_sha256": [
                    visit.method_tensor_sha256() for visit in self.visits
                ],
            }
        )

    def geometric_sample(self, *, neural_voxel_size_m: float) -> NeuralSampleMap:
        if (
            isinstance(neural_voxel_size_m, bool)
            or not isinstance(neural_voxel_size_m, (int, float))
            or not math.isfinite(float(neural_voxel_size_m))
            or float(neural_voxel_size_m) <= 0.0
        ):
            raise RScanMethodViewError("neural voxel size must be finite and positive")
        visit_maps: list[VisitMap] = []
        for visit in self.visits:
            entities = [
                EntityPrediction(
                    entity_id=candidate_id,
                    points_xyz=visit.points_xyz[visit.segment_ids == segment_id],
                    semantic_embedding=None,
                    semantic_label=None,
                    semantic_score=0.0,
                    lifecycle_state="current",
                    first_seen=float(visit.visit_id),
                    last_seen=float(visit.visit_id),
                    metadata={"candidate_source": "mesh_segment"},
                )
                for segment_id, candidate_id in zip(
                    np.unique(visit.segment_ids), visit.candidate_ids, strict=True
                )
            ]
            snapshot = MapSnapshot(
                method="3RSCAN_NATIVE_MESH_SEGMENTS",
                scene_id=self.pair_id,
                timestamp=float(visit.visit_id),
                entities=entities,
                background_xyz=None,
                scope="current",
            )
            visit_maps.append(
                VisitMap(
                    visit_id=visit.visit_id,
                    snapshot=snapshot,
                    coordinate_frame_id=self.coordinate_frame_id,
                    source_manifest_sha256=self.source_manifest_sha256,
                    map_voxel_size_m=float(neural_voxel_size_m),
                    observed_frame_start=visit.visit_id,
                    observed_frame_end=visit.visit_id,
                )
            )
        return build_geometric_pair_sample(
            visit_maps[0], visit_maps[1], neural_voxel_size_m=neural_voxel_size_m
        )


def _processed_visit(
    value: object,
    *,
    visit_id: int,
    scan_id: str,
    support_mask: object | None,
) -> ProcessedVisitView:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] < 10 or not len(raw):
        raise RScanMethodViewError("processed points must have shape (N, >=10)")
    method = np.asarray(raw[:, :10], dtype=np.float32)
    if not np.all(np.isfinite(method)):
        raise RScanMethodViewError("processed method columns must be finite")
    segments_raw = method[:, 9]
    rounded = np.rint(segments_raw)
    if not np.array_equal(segments_raw, rounded):
        raise RScanMethodViewError("mesh segment IDs must be exactly integral")
    indices = np.arange(len(method), dtype=np.int64)
    if support_mask is not None:
        mask = np.asarray(support_mask)
        if mask.dtype != np.bool_ or mask.shape != (len(method),):
            raise RScanMethodViewError("support mask must be a boolean vector aligned to source points")
        if not np.any(mask):
            raise RScanMethodViewError("support mask cannot remove every visit point")
        method = method[mask]
        rounded = rounded[mask]
        indices = indices[mask]
    return ProcessedVisitView(
        visit_id=visit_id,
        scan_id=scan_id,
        points_xyz=method[:, :3],
        rgb=method[:, 3:6],
        normals_xyz=method[:, 6:9],
        segment_ids=rounded.astype(np.int64),
        source_point_indices=indices,
    )


def build_method_pair_view(
    *,
    pair_id: str,
    scan_ids: Sequence[str],
    processed_visits: Sequence[object],
    source_manifest_sha256: str,
    domain_id: DomainId,
    support_masks: Sequence[object] | None = None,
) -> RScanMethodPairView:
    """Build a pair view from GT-free processed columns and optional support masks."""

    if isinstance(scan_ids, (str, bytes)) or len(scan_ids) != 2:
        raise RScanMethodViewError("scan_ids must contain exactly two values")
    if isinstance(processed_visits, (str, bytes)) or len(processed_visits) != 2:
        raise RScanMethodViewError("processed_visits must contain exactly two arrays")
    if domain_id == "D0_NATIVE_PROCESSED":
        if support_masks is not None:
            raise RScanMethodViewError("D0 cannot use a support mask")
        normalized_masks: tuple[None, None] = (None, None)
        parent = None
    else:
        if support_masks is None or len(support_masks) != 2:
            raise RScanMethodViewError("supported domain requires two support masks")
        normalized_masks = (support_masks[0], support_masks[1])
        full = build_method_pair_view(
            pair_id=pair_id,
            scan_ids=scan_ids,
            processed_visits=processed_visits,
            source_manifest_sha256=source_manifest_sha256,
            domain_id="D0_NATIVE_PROCESSED",
        )
        parent = full.method_tensor_sha256()
    visits = tuple(
        _processed_visit(
            processed_visits[visit_id],
            visit_id=visit_id,
            scan_id=scan_ids[visit_id],
            support_mask=normalized_masks[visit_id],
        )
        for visit_id in (0, 1)
    )
    return RScanMethodPairView(
        pair_id=pair_id,
        domain_id=domain_id,
        visits=visits,  # type: ignore[arg-type]
        source_manifest_sha256=source_manifest_sha256,
        parent_method_tensor_sha256=parent,
    )


def _bound_processed_array(record: object, *, label: str) -> np.ndarray:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise RScanMethodViewError(f"{label} binding schema is invalid")
    path_value = record.get("path")
    digest = record.get("sha256")
    byte_count = record.get("byte_count")
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(byte_count) is not int
        or byte_count < 0
    ):
        raise RScanMethodViewError(f"{label} binding value is invalid")
    path = Path(os.path.abspath(path_value))
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as error:
        raise RScanMethodViewError(f"{label} binding file is unavailable") from error
    if not stat.S_ISREG(before.st_mode):
        raise RScanMethodViewError(f"{label} binding must name a regular non-symlink file")
    content = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise RScanMethodViewError(f"{label} binding changed while being read")
    if len(content) != byte_count or hashlib.sha256(content).hexdigest() != digest:
        raise RScanMethodViewError(f"{label} binding mismatch")
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RScanMethodViewError(f"{label} processed NPY is invalid") from error


def build_method_pair_view_from_manifest(
    *,
    pair_record: object,
    source_manifest_sha256: str,
    domain_id: DomainId,
    support_masks: Sequence[object] | None = None,
) -> RScanMethodPairView:
    """Re-audit a frozen pair record and construct its method-visible view."""

    if not isinstance(pair_record, Mapping):
        raise RScanMethodViewError("pair record must be a mapping")
    pair_id = _nonempty(pair_record.get("pair_id"), "pair_id")
    sessions = pair_record.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 2:
        raise RScanMethodViewError("pair sessions must contain exactly two records")
    ordered: list[tuple[str, np.ndarray]] = []
    for expected_visit, session in enumerate(sessions):
        if not isinstance(session, Mapping) or session.get("visit_index") != expected_visit:
            raise RScanMethodViewError("pair sessions must be ordered t0 then t1")
        processed = session.get("processed_assets")
        if not isinstance(processed, Mapping):
            raise RScanMethodViewError("processed asset record is invalid")
        ordered.append(
            (
                _nonempty(session.get("scan_id"), "scan_id"),
                _bound_processed_array(
                    processed.get("points"), label=f"visit {expected_visit} points"
                ),
            )
        )
    return build_method_pair_view(
        pair_id=pair_id,
        scan_ids=(ordered[0][0], ordered[1][0]),
        processed_visits=(ordered[0][1], ordered[1][1]),
        source_manifest_sha256=source_manifest_sha256,
        domain_id=domain_id,
        support_masks=support_masks,
    )


@dataclass(frozen=True, slots=True)
class MethodExecutionPlan:
    method_id: str
    forward_mode: ForwardMode
    method_input_sha256: str
    forward_input_sha256: tuple[str, ...]
    candidate_schema: str
    raw_forward_cache_key: str | None


def build_execution_plans(
    pair: RScanMethodPairView,
) -> tuple[MethodExecutionPlan, ...]:
    """Declare comparable G/F/R inputs and the raw-forward reuse boundary."""

    if not isinstance(pair, RScanMethodPairView):
        raise TypeError("pair must be an RScanMethodPairView")
    pair_hash = pair.method_tensor_sha256()
    visit_hashes = tuple(visit.method_tensor_sha256() for visit in pair.visits)
    raw_key = _canonical_sha256(
        {"backend": "rescene_joint_raw", "method_input_sha256": pair_hash}
    )
    method_ids = (
        ("G_full", "deterministic_pair", (pair_hash,), None),
        ("G_supported", "deterministic_pair", (pair_hash,), None),
        ("F", "independent_single_visit", visit_hashes, None),
        ("R_legacy", "joint_two_visit", (pair_hash,), raw_key),
        ("R_supported", "joint_two_visit", (pair_hash,), raw_key),
    )
    return tuple(
        MethodExecutionPlan(
            method_id=method_id,
            forward_mode=forward_mode,  # type: ignore[arg-type]
            method_input_sha256=pair_hash,
            forward_input_sha256=inputs,
            candidate_schema="per_visit_mesh_segments",
            raw_forward_cache_key=cache_key,
        )
        for method_id, forward_mode, inputs, cache_key in method_ids
    )


@dataclass(frozen=True, slots=True)
class DomainCoverageRow:
    domain_id: DomainId
    status: DomainStatus
    method_tensor_sha256: str | None
    metrics: Mapping[str, float] | None


def domain_coverage(
    *,
    d0: RScanMethodPairView,
    d1: RScanMethodPairView | None,
    d2: RScanMethodPairView | None,
) -> tuple[DomainCoverageRow, ...]:
    """Return explicit domain availability without converting missing rows to zero."""

    supplied = (d0, d1, d2)
    if not isinstance(d0, RScanMethodPairView) or d0.domain_id != _DOMAINS[0]:
        raise RScanMethodViewError("d0 must be the native processed view")
    rows: list[DomainCoverageRow] = []
    for expected, value in zip(_DOMAINS, supplied, strict=True):
        if value is None:
            rows.append(DomainCoverageRow(expected, "MISSING_ASSET", None, None))
            continue
        if not isinstance(value, RScanMethodPairView) or value.domain_id != expected:
            raise RScanMethodViewError("domain view is assigned to the wrong row")
        if value.pair_id != d0.pair_id:
            raise RScanMethodViewError("domain views must describe one pair")
        rows.append(
            DomainCoverageRow(expected, "PASS", value.method_tensor_sha256(), {})
        )
    return tuple(rows)


__all__ = [
    "DomainCoverageRow",
    "MethodExecutionPlan",
    "ProcessedVisitView",
    "RScanMethodPairView",
    "RScanMethodViewError",
    "build_execution_plans",
    "build_method_pair_view",
    "build_method_pair_view_from_manifest",
    "domain_coverage",
]
