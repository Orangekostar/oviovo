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
from types import MappingProxyType
from typing import Literal

import numpy as np

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.query_instance_projection import ProjectionConfig
from src.oviv2.temporal_pair_reasoner import validate_query_evidence
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    OviEntitySemanticEvidence,
    PairRelation,
    TemporalQueryEvidence,
    VisitMap,
)
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
    if domain_id == "D2_OVI_RECONSTRUCTION":
        raise RScanMethodViewError(
            "D2 must be built from native OVI manifests, not processed visits or support masks"
        )
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


def _native_full_domains(
    pair: RScanMethodPairView, arrays: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "inverse_n",
        "lowres_point2segment_m",
        "lowres_visit_ids_m",
        "full_visit_ids_n",
        "full_original_segment_ids_n",
    }
    if not isinstance(arrays, Mapping) or not required.issubset(arrays):
        raise RScanMethodViewError("native arrays lack method-domain mappings")
    inverse = np.asarray(arrays["inverse_n"])
    low_segments = np.asarray(arrays["lowres_point2segment_m"])
    low_visits = np.asarray(arrays["lowres_visit_ids_m"])
    full_visits = np.asarray(arrays["full_visit_ids_n"])
    full_segments = np.asarray(arrays["full_original_segment_ids_n"])
    if any(
        value.ndim != 1 or not np.issubdtype(value.dtype, np.integer)
        for value in (inverse, low_segments, low_visits, full_visits, full_segments)
    ):
        raise RScanMethodViewError("native method-domain mappings must be integer vectors")
    n = sum(visit.point_count for visit in pair.visits)
    m = len(low_segments)
    if (
        len(inverse) != n
        or len(full_visits) != n
        or len(full_segments) != n
        or len(low_visits) != m
        or m == 0
        or np.any(inverse < 0)
        or np.any(inverse >= m)
    ):
        raise RScanMethodViewError("native method-domain mappings are misaligned")
    expected_points, _rgb, _normals, expected_visits, expected_segments = (
        _expected_pair_arrays(pair)
    )
    del expected_points
    if not np.array_equal(full_visits.astype(np.int8), expected_visits) or not np.array_equal(
        full_segments.astype(np.int64), expected_segments
    ):
        raise RScanMethodViewError("native full domain differs from the method view")
    representative_visits = np.full(m, -1, dtype=np.int8)
    for model_index in range(m):
        contributors = full_visits[inverse == model_index]
        if not len(contributors) or len(np.unique(contributors)) != 1:
            raise RScanMethodViewError("native voxel mixes visits or lacks provenance")
        representative_visits[model_index] = np.int8(contributors[0])
    if not np.array_equal(low_visits.astype(np.int8), representative_visits):
        raise RScanMethodViewError("native low-resolution visits differ from inverse provenance")
    if np.any(low_segments < 0):
        raise RScanMethodViewError("native point2segment contains a negative ID")
    return (
        inverse.astype(np.int64, copy=False),
        low_segments.astype(np.int64, copy=False),
        low_visits.astype(np.int8, copy=False),
        full_visits.astype(np.int8, copy=False),
        full_segments.astype(np.int64, copy=False),
    )


def _expected_pair_arrays(
    pair: RScanMethodPairView,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.concatenate([visit.points_xyz for visit in pair.visits], axis=0)
    rgb = np.concatenate([visit.rgb for visit in pair.visits], axis=0)
    normals = np.concatenate([visit.normals_xyz for visit in pair.visits], axis=0)
    visits = np.concatenate(
        [
            np.full(visit.point_count, visit.visit_id, dtype=np.int8)
            for visit in pair.visits
        ]
    )
    first_segments = pair.visits[0].segment_ids
    second_segments = (
        pair.visits[1].segment_ids + int(np.max(first_segments)) + 1
    )
    segments = np.concatenate((first_segments, second_segments)).astype(np.int64)
    return points, rgb, normals, visits, segments


def _candidate_global_segment(
    pair: RScanMethodPairView, visit_id: int, local_segment: int
) -> int:
    if visit_id == 0:
        return local_segment
    return local_segment + int(np.max(pair.visits[0].segment_ids)) + 1


def pool_independent_segment_features(
    pair: RScanMethodPairView, arrays: Mapping[str, object]
) -> Mapping[tuple[int, str], np.ndarray]:
    """Mean-pool visit-local backbone features by GT-free mesh segment."""

    if not isinstance(pair, RScanMethodPairView):
        raise TypeError("pair must be an RScanMethodPairView")
    inverse, _low_segments, _low_visits, full_visits, full_segments = (
        _native_full_domains(pair, arrays)
    )
    features = np.asarray(arrays.get("backbone_features_mf"))
    if (
        features.ndim != 2
        or features.shape[0] != int(np.max(inverse)) + 1
        or not np.issubdtype(features.dtype, np.floating)
        or not np.all(np.isfinite(features))
    ):
        raise RScanMethodViewError("native backbone feature domain is invalid")
    expanded = features[inverse].astype(np.float64, copy=False)
    result: dict[tuple[int, str], np.ndarray] = {}
    for visit in pair.visits:
        for local_segment, candidate_id in zip(
            np.unique(visit.segment_ids), visit.candidate_ids, strict=True
        ):
            global_segment = _candidate_global_segment(
                pair, visit.visit_id, int(local_segment)
            )
            selected = (full_visits == visit.visit_id) & (
                full_segments == global_segment
            )
            if not np.any(selected):
                raise RScanMethodViewError("native features omit a method candidate")
            embedding = np.mean(expanded[selected], axis=0)
            norm = float(np.linalg.norm(embedding))
            if not math.isfinite(norm) or norm <= 0.0:
                raise RScanMethodViewError("pooled candidate embedding has zero norm")
            normalized = np.asarray(embedding / norm, dtype=np.float32)
            normalized.setflags(write=False)
            result[(visit.visit_id, candidate_id)] = normalized
    return MappingProxyType(dict(sorted(result.items())))


def build_feature_geometric_sample(
    pair: RScanMethodPairView,
    arrays: Mapping[str, object],
    *,
    voxel_size_m: float,
) -> NeuralSampleMap:
    """Attach independent Concerto entity means to the frozen geometric sample."""

    base = pair.geometric_sample(neural_voxel_size_m=voxel_size_m)
    embeddings = pool_independent_segment_features(pair, arrays)
    semantics = tuple(
        OviEntitySemanticEvidence(
            visit_id=visit_id,
            entity_id=entity_id,
            semantic_label=None,
            semantic_score=0.0,
            semantic_embedding=embedding,
        )
        for (visit_id, entity_id), embedding in embeddings.items()
    )
    return NeuralSampleMap(
        coordinates_xyzt=base.coordinates_xyzt,
        features=base.features,
        visit_ids=base.visit_ids,
        source_visit_ids=base.source_visit_ids,
        source_entity_ids=base.source_entity_ids,
        source_point_indices=base.source_point_indices,
        source_to_token_offsets=base.source_to_token_offsets,
        neural_voxel_size_m=base.neural_voxel_size_m,
        feature_schema="geometric_only+independent_concerto_entity_mean",
        coordinate_frame_id=base.coordinate_frame_id,
        source_manifest_sha256=base.source_manifest_sha256,
        source_visit_map_sha256=base.source_visit_map_sha256,
        entity_semantics=semantics,
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(value: np.ndarray) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    shifted = raw - raw.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def native_query_evidence(
    pair: RScanMethodPairView,
    sample: NeuralSampleMap,
    arrays: Mapping[str, object],
    *,
    checkpoint_sha256: str,
) -> TemporalQueryEvidence:
    """Project one native raw query tensor onto the shared method candidates."""

    inverse, low_segments, _low_visits, full_visits, full_segments = (
        _native_full_domains(pair, arrays)
    )
    masks = np.asarray(arrays.get("raw_masks_sq"))
    logits = np.asarray(arrays.get("raw_logits_qc"))
    if (
        masks.ndim != 2
        or logits.ndim != 2
        or masks.shape[1] != logits.shape[0]
        or logits.shape[1] < 2
        or not np.all(np.isfinite(masks))
        or not np.all(np.isfinite(logits))
        or np.any(low_segments >= masks.shape[0])
    ):
        raise RScanMethodViewError("native raw query arrays are invalid")
    full_scores = _sigmoid(masks[low_segments[inverse]])
    token_entities = sample.token_entity_ids
    token_scores_all = np.zeros((masks.shape[1], len(token_entities)), dtype=np.float32)
    for visit in pair.visits:
        for local_segment, candidate_id in zip(
            np.unique(visit.segment_ids), visit.candidate_ids, strict=True
        ):
            global_segment = _candidate_global_segment(
                pair, visit.visit_id, int(local_segment)
            )
            selected_points = (full_visits == visit.visit_id) & (
                full_segments == global_segment
            )
            candidate_scores = np.mean(full_scores[selected_points], axis=0)
            selected_tokens = np.asarray(
                [
                    index
                    for index, (token_visit, token_entity) in enumerate(
                        zip(sample.visit_ids, token_entities, strict=True)
                    )
                    if int(token_visit) == visit.visit_id
                    and token_entity == candidate_id
                ],
                dtype=np.int64,
            )
            if not len(selected_tokens):
                raise RScanMethodViewError("geometric sample omits a method candidate")
            token_scores_all[:, selected_tokens] = candidate_scores[:, None]
    positive = token_scores_all > 0.5
    retained = np.flatnonzero(np.any(positive, axis=1))
    if not len(retained):
        raise RScanMethodViewError("native raw queries select no method candidates")
    probabilities = _softmax(logits)
    foreground = probabilities[:, :-1].max(axis=1)
    positive_counts = positive.sum(axis=1)
    mean_positive = np.divide(
        (token_scores_all * positive).sum(axis=1),
        positive_counts,
        out=np.zeros(masks.shape[1], dtype=np.float64),
        where=positive_counts > 0,
    )
    query_scores = np.asarray(foreground * mean_positive, dtype=np.float32)
    backend_hash = _canonical_sha256(
        {
            "backend": "rescene_native_raw_query",
            "candidate_pooling": "full_point_mean_from_native_inverse",
            "mask_threshold": 0.5,
        }
    )
    evidence = TemporalQueryEvidence(
        status="PASS",
        backend_name="rescene:native_raw_query",
        backend_config_sha256=backend_hash,
        pair_sha256=sample.content_sha256(),
        temporal_query_ids=tuple(f"query_{int(index):04d}" for index in retained),
        query_masks=np.ascontiguousarray(positive[retained], dtype=np.bool_),
        token_scores=np.ascontiguousarray(token_scores_all[retained], dtype=np.float32),
        query_scores=np.ascontiguousarray(query_scores[retained], dtype=np.float32),
        checkpoint_sha256=_sha256(checkpoint_sha256, "checkpoint SHA-256"),
        ranking_eligible=True,
        runtime_s=0.0,
        peak_memory_bytes=0,
        diagnostics={"raw_forward_reused": "true"},
    )
    validate_query_evidence(sample, evidence)
    return evidence


def _relation_state(
    t0_ids: tuple[str, ...],
    t1_ids: tuple[str, ...],
    centroids: Mapping[tuple[int, str], np.ndarray],
    config: ProjectionConfig,
) -> tuple[str, float | None]:
    cardinality = (len(t0_ids), len(t1_ids))
    if cardinality == (1, 1):
        distance = float(np.linalg.norm(centroids[(0, t0_ids[0])] - centroids[(1, t1_ids[0])]))
        return (
            "persistent_static"
            if distance <= config.static_centroid_tolerance_m
            else "persistent_moved",
            distance,
        )
    if cardinality == (0, 1):
        return "appeared", None
    if cardinality == (1, 0):
        return "removed_candidate", None
    if cardinality[0] == 1 and cardinality[1] > 1:
        return "split", None
    if cardinality[0] > 1 and cardinality[1] == 1:
        return "merge", None
    return "uncertain", None


def _entity_token_data(
    sample: NeuralSampleMap,
) -> tuple[
    Mapping[tuple[int, str], np.ndarray],
    Mapping[tuple[int, str], np.ndarray],
]:
    groups: dict[tuple[int, str], list[int]] = {}
    for index, entity_id in enumerate(sample.token_entity_ids):
        groups.setdefault((int(sample.visit_ids[index]), entity_id), []).append(index)
    contributor_counts = np.diff(sample.source_to_token_offsets).astype(np.float64)
    indices = {
        key: np.asarray(value, dtype=np.int64) for key, value in sorted(groups.items())
    }
    centroids = {
        key: np.average(
            sample.coordinates_xyzt[value, :3],
            axis=0,
            weights=contributor_counts[value],
        )
        for key, value in indices.items()
    }
    return MappingProxyType(indices), MappingProxyType(centroids)


def project_queries_fast(
    sample: NeuralSampleMap,
    evidence: TemporalQueryEvidence,
    config: ProjectionConfig,
) -> tuple[PairRelation, ...]:
    """Project queries in one pass while preserving the reference relation contract."""

    if not isinstance(config, ProjectionConfig):
        raise TypeError("config must be a ProjectionConfig")
    validate_query_evidence(sample, evidence)
    assert evidence.query_masks is not None
    assert evidence.token_scores is not None
    assert evidence.query_scores is not None
    groups, centroids = _entity_token_data(sample)
    contributor_counts = np.diff(sample.source_to_token_offsets).astype(np.int64)
    identity_source = (
        "rescene"
        if evidence.backend_name.startswith("rescene:")
        else "geometric_baseline"
        if evidence.backend_name == "geometric_semantic"
        else "unmatched"
    )
    relations: list[PairRelation] = []
    for query_index, query_id in enumerate(evidence.temporal_query_ids):
        mask = evidence.query_masks[query_index]
        records: list[tuple[int, str, float, float, float]] = []
        for (visit_id, entity_id), indices in groups.items():
            selected = indices[mask[indices]]
            if not len(selected):
                continue
            token_coverage = len(selected) / len(indices)
            entity_source_count = int(contributor_counts[indices].sum())
            selected_source_count = int(contributor_counts[selected].sum())
            source_coverage = selected_source_count / entity_source_count
            soft_mass = float(evidence.token_scores[query_index, indices].sum())
            records.append(
                (visit_id, entity_id, token_coverage, source_coverage, soft_mass)
            )
        selected_records = [
            record
            for record in records
            if record[2] >= config.minimum_entity_token_coverage
            or record[3] >= config.minimum_source_point_coverage
        ] or records
        selected_records.sort(key=lambda value: (value[0], value[1]))
        t0_ids = tuple(value[1] for value in selected_records if value[0] == 0)
        t1_ids = tuple(value[1] for value in selected_records if value[0] == 1)
        state, centroid_distance = _relation_state(t0_ids, t1_ids, centroids, config)
        source_coverages = [value[3] for value in selected_records]
        relation_evidence = {
            "query_score": float(evidence.query_scores[query_index]),
            "selected_entity_count": float(len(selected_records)),
            "minimum_source_point_coverage": min(source_coverages, default=0.0),
            "soft_mass": sum(value[4] for value in selected_records),
        }
        if centroid_distance is not None:
            relation_evidence["centroid_distance_m"] = centroid_distance
        relations.append(
            PairRelation(
                temporal_query_id=query_id,
                t0_entity_ids=t0_ids,
                t1_entity_ids=t1_ids,
                state=state,
                query_confidence=float(evidence.query_scores[query_index]),
                evidence=relation_evidence,
                identity_source=identity_source,
            )
        )
    return tuple(relations)


def resolve_native_queries_one_to_one(
    pair: RScanMethodPairView,
    evidence: TemporalQueryEvidence,
    sample: NeuralSampleMap,
    *,
    minimum_candidate_score: float = 0.5,
) -> tuple[PairRelation, ...]:
    """Resolve raw ReScene queries through mutual dominant candidate support."""

    if not 0.0 <= float(minimum_candidate_score) <= 1.0:
        raise RScanMethodViewError("minimum candidate score must be in [0, 1]")
    validate_query_evidence(sample, evidence)
    assert evidence.query_masks is not None
    assert evidence.token_scores is not None
    assert evidence.query_scores is not None
    groups, centroids = _entity_token_data(sample)
    keys = tuple(groups)
    scores = np.asarray(
        [
            [float(np.mean(evidence.token_scores[q, groups[key]])) for key in keys]
            for q in range(len(evidence.temporal_query_ids))
        ],
        dtype=np.float64,
    )
    query_best: dict[tuple[int, int], int] = {}
    for query_index in range(len(evidence.temporal_query_ids)):
        for visit_id in (0, 1):
            options = [index for index, key in enumerate(keys) if key[0] == visit_id]
            query_best[(query_index, visit_id)] = max(
                options, key=lambda index: (scores[query_index, index], -index)
            )
    entity_best = {
        entity_index: max(
            range(len(evidence.temporal_query_ids)),
            key=lambda query_index: (scores[query_index, entity_index], -query_index),
        )
        for entity_index in range(len(keys))
    }
    used: set[int] = set()
    relations: list[PairRelation] = []
    for query_index, query_id in enumerate(evidence.temporal_query_ids):
        first = query_best[(query_index, 0)]
        second = query_best[(query_index, 1)]
        if (
            scores[query_index, first] < minimum_candidate_score
            or scores[query_index, second] < minimum_candidate_score
            or entity_best[first] != query_index
            or entity_best[second] != query_index
            or first in used
            or second in used
        ):
            continue
        used.update((first, second))
        t0_id = keys[first][1]
        t1_id = keys[second][1]
        state, distance = _relation_state(
            (t0_id,), (t1_id,), centroids, ProjectionConfig()
        )
        assert distance is not None
        relations.append(
            PairRelation(
                temporal_query_id=query_id,
                t0_entity_ids=(t0_id,),
                t1_entity_ids=(t1_id,),
                state=state,
                query_confidence=float(evidence.query_scores[query_index]),
                evidence={
                    "query_score": float(evidence.query_scores[query_index]),
                    "t0_candidate_score": float(scores[query_index, first]),
                    "t1_candidate_score": float(scores[query_index, second]),
                    "centroid_distance_m": distance,
                },
                identity_source="rescene",
            )
        )
    for entity_index, (visit_id, entity_id) in enumerate(keys):
        if entity_index in used:
            continue
        relations.append(
            PairRelation(
                temporal_query_id=f"unmatched:{visit_id}:{entity_id}",
                t0_entity_ids=(entity_id,) if visit_id == 0 else (),
                t1_entity_ids=(entity_id,) if visit_id == 1 else (),
                state="removed_candidate" if visit_id == 0 else "appeared",
                query_confidence=1.0,
                evidence={"query_score": 1.0, "candidate_unmatched": 1.0},
                identity_source="rescene",
            )
        )
    relations.sort(key=lambda value: str(value.temporal_query_id))
    return tuple(relations)


def resolve_native_queries_fragment_union(
    sample: NeuralSampleMap,
    evidence: TemporalQueryEvidence,
    config: ProjectionConfig,
    *,
    minimum_query_confidence: float,
) -> tuple[PairRelation, ...]:
    """Keep query-consistent fragment unions after a GT-free confidence gate."""

    if (
        isinstance(minimum_query_confidence, bool)
        or not isinstance(minimum_query_confidence, (int, float))
        or not math.isfinite(float(minimum_query_confidence))
        or not 0.0 <= float(minimum_query_confidence) <= 1.0
    ):
        raise RScanMethodViewError("minimum query confidence must be in [0, 1]")
    threshold = float(minimum_query_confidence)
    return tuple(
        relation
        for relation in project_queries_fast(sample, evidence, config)
        if relation.query_confidence >= threshold
    )


__all__ = [
    "DomainCoverageRow",
    "MethodExecutionPlan",
    "ProcessedVisitView",
    "RScanMethodPairView",
    "RScanMethodViewError",
    "build_execution_plans",
    "build_feature_geometric_sample",
    "build_method_pair_view",
    "build_method_pair_view_from_manifest",
    "domain_coverage",
    "native_query_evidence",
    "pool_independent_segment_features",
    "project_queries_fast",
    "resolve_native_queries_fragment_union",
    "resolve_native_queries_one_to_one",
]
