#!/usr/bin/env python3
"""Run fixed-candidate object association on real two-visit OVI maps."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import resource
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.spatial import cKDTree

from scripts.evaluation.prepare_ovi_rescene_input_v2 import capture_native_sampling
from scripts.evaluation.rescene_pair_executor import (
    EXPECTED_CONCERTO_SHA256,
    NativeForwardResult,
    extract_model_state_dict,
    postprocess_native_predictions,
)
from src.evaluation.object_pair_association import (
    AssociationConfig,
    IndependentFeatureBank,
    ObjectPairPrediction,
    ObjectPairScoreMatrix,
    build_feature_score_matrix,
    build_geometric_score_matrix,
    build_rescene_score_matrix,
    solve_object_pair_assignment,
)
from src.evaluation.ovi_pair_views import OviObjectPairView
from src.evaluation.rscan_association_metrics import (
    FixedEndpointBindings,
    build_fixed_endpoint_bindings,
    evaluate_fixed_object_predictions,
)
from src.evaluation.rscan_gt_instances import (
    GroundTruthPair,
    PredictedInstance,
    evaluate_instance_geometry,
    voxelize_points,
)
from src.evaluation.rscan_method_views import (
    ProcessedVisitView,
    RScanMethodPairView,
)
from src.oviv2.ovi_surface_attributes import SurfaceGroup
from src.oviv2.rescene_input_bridge import (
    AdapterGeometry,
    ModelCandidateMap,
    NativeSamplingMap,
    RecoveredModelSupport,
    ReSceneModelInput,
    SurfaceAttributeBundle,
    bind_surface_attributes,
    build_adapter_geometry,
    select_model_candidates,
)
from src.oviv2.rescene_supported_view import (
    SupportedInferenceView,
    build_supported_inference_view,
)
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    PairRelation,
    TemporalQueryEvidence,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_METHOD_IDS = ("G_full", "G_supported", "F_obj", "R_obj")
_RESULT_METRIC_FIELDS = (
    "status",
    "support_domain",
    "paired_prediction_count",
    "candidate_unmatched_count",
    "true_positive_count",
    "false_positive_count",
    "endpoint_failure_count",
    "false_reid_count",
    "same_class_mismatch_count",
    "duplicate_count",
    "precision",
    "end_to_end_recall",
    "persistent_gt_count",
    "representation_conditional_gt_count",
    "representation_conditional_recall",
    "strict_true_positive_count",
    "strict_false_positive_count",
    "strict_duplicate_count",
    "strict_precision",
    "strict_end_to_end_recall",
    "strict_representation_conditional_recall",
    "rigid_true_positive_count",
    "rigid_gt_count",
    "rigid_recall",
    "support_conditional_true_positive_count",
    "support_conditional_gt_count",
    "support_conditional_recall",
    "support_conditional_status",
)


class ObjectLevelTransferError(ValueError):
    """Raised when fixed-candidate inference provenance is inconsistent."""


@dataclass(frozen=True, slots=True)
class ExternalFileBinding:
    path: Path
    sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.is_absolute():
            raise ObjectLevelTransferError("external file path must be absolute")
        if len(self.sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.sha256
        ):
            raise ObjectLevelTransferError("external file SHA-256 is invalid")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ObjectLevelTransferError("external file byte count is invalid")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True)
class ObjectTransferConfig:
    source_path: Path
    source_sha256: str
    output_root: Path
    selection_manifest: Path
    selection_manifest_sha256: str
    minimum_match_scores: Mapping[str, float]
    centroid_scale_m: float
    static_centroid_tolerance_m: float
    sampler_seed: int
    maximum_candidates: int
    device: str
    model_python: Path
    rescene_checkout: Path
    source_commit: str
    concerto_checkpoint: ExternalFileBinding
    rescene_checkpoint: ExternalFileBinding
    native_sampler_source: ExternalFileBinding
    d2_mapping_receipt: Path
    d2_pair_view_receipt: Path
    ground_truth_root: Path


@dataclass(frozen=True, slots=True)
class D2InferenceBundle:
    pair_content_sha256: str
    candidate_ids: tuple[
        tuple[tuple[int, str], ...], tuple[tuple[int, str], ...]
    ]
    geometry: AdapterGeometry
    surface: SurfaceAttributeBundle
    sampling: NativeSamplingMap
    candidates: ModelCandidateMap
    support: RecoveredModelSupport
    supported_view: SupportedInferenceView
    model_input: ReSceneModelInput
    sample: NeuralSampleMap


@dataclass(frozen=True, slots=True)
class _PreparedD2Components:
    geometry: AdapterGeometry
    surface: SurfaceAttributeBundle
    sampling: NativeSamplingMap
    candidates: ModelCandidateMap


@dataclass(frozen=True, slots=True)
class AssociationForwardOutputs:
    independent_model_features: tuple[np.ndarray, np.ndarray]
    visit_forward_sha256: tuple[str, str]
    independent_runtime_s: tuple[float, float]
    joint_forward: NativeForwardResult


@dataclass(frozen=True, slots=True)
class SharedCandidateEvaluation:
    matrices: Mapping[str, ObjectPairScoreMatrix]
    predictions: Mapping[str, tuple[ObjectPairPrediction, ...]]
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    endpoint_bindings: Mapping[str, FixedEndpointBindings]
    feature_bank: IndependentFeatureBank
    query_evidence: TemporalQueryEvidence


def build_d1_sensor_support_view(
    d0_pair: RScanMethodPairView,
    d2_pair: OviObjectPairView,
    *,
    maximum_distance_m: float,
) -> tuple[RScanMethodPairView, dict[str, object]]:
    """Restrict native points using only proximity to the real D2 dense surface."""

    if (
        not isinstance(d0_pair, RScanMethodPairView)
        or d0_pair.domain_id != "D0_NATIVE_PROCESSED"
    ):
        raise ObjectLevelTransferError("D1 parent must be a D0 native pair")
    if not isinstance(d2_pair, OviObjectPairView):
        raise TypeError("d2_pair must be an OviObjectPairView")
    if d0_pair.pair_id != d2_pair.pair_id or tuple(
        visit.scan_id for visit in d0_pair.visits
    ) != tuple(visit.scan_id for visit in d2_pair.visits):
        raise ObjectLevelTransferError("D0 and D2 must describe the same UUID pair")
    if (
        isinstance(maximum_distance_m, bool)
        or not isinstance(maximum_distance_m, (int, float))
        or not math.isfinite(float(maximum_distance_m))
        or float(maximum_distance_m) <= 0.0
    ):
        raise ObjectLevelTransferError("D1 support distance must be finite and positive")
    threshold = float(maximum_distance_m)
    supported_visits: list[ProcessedVisitView] = []
    point_counts: list[int] = []
    support_counts: list[int] = []
    for native, reconstructed in zip(d0_pair.visits, d2_pair.visits, strict=True):
        distances, _nearest = cKDTree(reconstructed.points_xyz).query(
            native.points_xyz,
            k=1,
            workers=1,
        )
        supported = np.asarray(distances <= threshold, dtype=np.bool_)
        if not np.any(supported):
            raise ObjectLevelTransferError(
                f"D1 support removes every point from visit {native.visit_id}"
            )
        supported_visits.append(
            ProcessedVisitView(
                visit_id=native.visit_id,
                scan_id=native.scan_id,
                points_xyz=native.points_xyz[supported],
                rgb=native.rgb[supported],
                normals_xyz=native.normals_xyz[supported],
                segment_ids=native.segment_ids[supported],
                source_point_indices=native.source_point_indices[supported],
            )
        )
        point_counts.append(native.point_count)
        support_counts.append(int(np.count_nonzero(supported)))
    pair = RScanMethodPairView(
        pair_id=d0_pair.pair_id,
        domain_id="D1_NATIVE_SENSOR_SUPPORT",
        visits=(supported_visits[0], supported_visits[1]),
        source_manifest_sha256=d0_pair.source_manifest_sha256,
        parent_method_tensor_sha256=d0_pair.method_tensor_sha256(),
        coordinate_frame_id=d0_pair.coordinate_frame_id,
        global_alignment_application=d0_pair.global_alignment_application,
    )
    total_points = sum(point_counts)
    total_supported = sum(support_counts)
    return pair, {
        "definition": "nearest_d2_dense_surface_within_distance",
        "maximum_distance_m": threshold,
        "visit_point_counts": point_counts,
        "visit_support_counts": support_counts,
        "visit_support_fractions": [
            supported / total
            for supported, total in zip(support_counts, point_counts, strict=True)
        ],
        "overall_support_fraction": total_supported / total_points,
        "ground_truth_used": False,
    }


def _projected_relation_state(
    pair: RScanMethodPairView,
    t0_ids: tuple[str, ...],
    t1_ids: tuple[str, ...],
    *,
    static_centroid_tolerance_m: float,
) -> str:
    cardinality = (len(t0_ids), len(t1_ids))
    if cardinality == (1, 1):
        centroids: list[np.ndarray] = []
        for visit, entity_id in zip(pair.visits, (t0_ids[0], t1_ids[0]), strict=True):
            segment_by_id = dict(zip(visit.candidate_ids, np.unique(visit.segment_ids), strict=True))
            centroids.append(
                np.mean(
                    visit.points_xyz[visit.segment_ids == segment_by_id[entity_id]],
                    axis=0,
                )
            )
        distance = float(np.linalg.norm(centroids[0] - centroids[1]))
        return (
            "persistent_static"
            if distance <= static_centroid_tolerance_m
            else "persistent_moved"
        )
    if cardinality == (0, 1):
        return "appeared"
    if cardinality == (1, 0):
        return "removed_candidate"
    if cardinality[0] == 1 and cardinality[1] > 1:
        return "split"
    if cardinality[0] > 1 and cardinality[1] == 1:
        return "merge"
    return "uncertain"


def project_cached_relations_to_sensor_support(
    d1_pair: RScanMethodPairView,
    records: tuple[Mapping[str, object], ...],
    *,
    static_centroid_tolerance_m: float,
) -> tuple[PairRelation, ...]:
    """Project cached D0 relations onto D1 without claiming a new forward."""

    if (
        not isinstance(d1_pair, RScanMethodPairView)
        or d1_pair.domain_id != "D1_NATIVE_SENSOR_SUPPORT"
    ):
        raise ObjectLevelTransferError("cached projection requires a D1 pair")
    if (
        isinstance(static_centroid_tolerance_m, bool)
        or not isinstance(static_centroid_tolerance_m, (int, float))
        or not math.isfinite(float(static_centroid_tolerance_m))
        or float(static_centroid_tolerance_m) <= 0.0
    ):
        raise ObjectLevelTransferError("static centroid tolerance is invalid")
    if not isinstance(records, tuple):
        raise ObjectLevelTransferError("cached relation records must be a tuple")
    expected = {
        "temporal_query_id",
        "t0_entity_ids",
        "t1_entity_ids",
        "state",
        "query_confidence",
        "identity_source",
        "evidence",
    }
    allowed = tuple(set(visit.candidate_ids) for visit in d1_pair.visits)
    projected: list[PairRelation] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ObjectLevelTransferError("cached relation schema is invalid")
        raw_sides = (record.get("t0_entity_ids"), record.get("t1_entity_ids"))
        if any(
            not isinstance(side, list)
            or any(not isinstance(entity_id, str) for entity_id in side)
            for side in raw_sides
        ):
            raise ObjectLevelTransferError("cached relation entity IDs are invalid")
        sides = tuple(
            tuple(entity_id for entity_id in side if entity_id in visit_allowed)
            for side, visit_allowed in zip(raw_sides, allowed, strict=True)
        )
        if not sides[0] and not sides[1]:
            continue
        evidence = record.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ObjectLevelTransferError("cached relation evidence is invalid")
        projected.append(
            PairRelation(
                temporal_query_id=record.get("temporal_query_id"),
                t0_entity_ids=sides[0],
                t1_entity_ids=sides[1],
                state=_projected_relation_state(
                    d1_pair,
                    sides[0],
                    sides[1],
                    static_centroid_tolerance_m=float(
                        static_centroid_tolerance_m
                    ),
                ),
                query_confidence=record.get("query_confidence"),
                evidence={**evidence, "cached_d0_projection": 1.0},
                identity_source=record.get("identity_source"),
            )
        )
    return tuple(projected)


def measure_native_candidate_representation(
    pair: RScanMethodPairView,
    ground_truth: GroundTruthPair,
    *,
    iou_threshold: float,
) -> dict[str, object]:
    """Measure whether the fixed native candidate pool represents each GT pair."""

    if not isinstance(pair, RScanMethodPairView):
        raise TypeError("pair must be an RScanMethodPairView")
    if not isinstance(ground_truth, GroundTruthPair):
        raise TypeError("ground_truth must be a GroundTruthPair")
    if pair.pair_id != ground_truth.pair_id:
        raise ObjectLevelTransferError("native pair and ground truth IDs differ")
    if iou_threshold not in {0.5, 0.25}:
        raise ObjectLevelTransferError("representation IoU threshold is unsupported")
    matched_ids: list[set[int]] = []
    for visit, targets in zip(pair.visits, ground_truth.visits, strict=True):
        predictions = tuple(
            PredictedInstance(
                prediction_id=candidate_id,
                voxels=voxelize_points(
                    visit.points_xyz[visit.segment_ids == segment_id],
                    voxel_size_m=ground_truth.voxel_size_m,
                ),
            )
            for segment_id, candidate_id in zip(
                np.unique(visit.segment_ids), visit.candidate_ids, strict=True
            )
        )
        geometry = evaluate_instance_geometry(
            predictions,
            targets,
            matching_policy="max_valid_count_then_iou",
        )
        threshold_result = (
            geometry.primary if iou_threshold == 0.5 else geometry.sensitivity
        )
        matched_ids.append(
            {match.gt_instance_id for match in threshold_result.matches}
        )
    left_gt_ids = {target.instance_id for target in ground_truth.visits[0]}
    persistent: dict[int, frozenset[int]] = {}
    for target in ground_truth.visits[1]:
        reference_id = ground_truth.identity_rules.reference_id_for_rescan(
            target.instance_id
        )
        allowed = frozenset(
            instance_id
            for instance_id in left_gt_ids
            if ground_truth.identity_rules.is_ambiguous_equivalent(
                instance_id, reference_id
            )
        )
        if allowed:
            persistent[target.instance_id] = allowed
    represented = sum(
        right_id in matched_ids[1] and bool(allowed & matched_ids[0])
        for right_id, allowed in persistent.items()
    )
    total = len(persistent)
    return {
        "persistent_gt_count": total,
        "represented_persistent_gt_count": represented,
        "proposal_representation_coverage": None if total == 0 else represented / total,
    }


def _configured_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ObjectLevelTransferError(f"{label} path is invalid")
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _external_binding(value: object, *, label: str) -> ExternalFileBinding:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ObjectLevelTransferError(f"{label} binding is invalid")
    return ExternalFileBinding(
        path=_configured_path(value["path"], label=label),
        sha256=value["sha256"],  # type: ignore[arg-type]
        byte_count=value["byte_count"],  # type: ignore[arg-type]
    )


def load_object_transfer_config(path: str | Path) -> ObjectTransferConfig:
    """Load the frozen object-transfer protocol without touching model assets."""

    source_path = Path(path).absolute()
    try:
        content = source_path.read_bytes()
        payload = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObjectLevelTransferError("object-transfer config is unavailable") from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("config_id") != "OVI_RESCENE_OBJECT_LEVEL_TRANSFER_V1"
    ):
        raise ObjectLevelTransferError("object-transfer config identity is invalid")
    selection = payload.get("selection_manifest")
    geometry = payload.get("geometry")
    rescene = payload.get("rescene")
    association = payload.get("association")
    execution = payload.get("execution")
    if any(
        not isinstance(value, Mapping)
        for value in (selection, geometry, rescene, association, execution)
    ):
        raise ObjectLevelTransferError("object-transfer config sections are invalid")
    assert isinstance(selection, Mapping)
    assert isinstance(geometry, Mapping)
    assert isinstance(rescene, Mapping)
    assert isinstance(association, Mapping)
    assert isinstance(execution, Mapping)
    raw_scores = association.get("minimum_match_scores")
    if not isinstance(raw_scores, Mapping) or set(raw_scores) != set(_METHOD_IDS):
        raise ObjectLevelTransferError("association thresholds must cover G/F/R")
    scores: dict[str, float] = {}
    for method in _METHOD_IDS:
        value = raw_scores[method]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ObjectLevelTransferError("association threshold is invalid")
        scores[method] = float(value)
    if (
        association.get("protocol_id")
        != "RSCAN_T2_FIXED_OBJECT_ASSOCIATION_V1"
        or association.get("native_sampler_mode") != "train"
        or association.get("independent_feature_pooling")
        != "source_point_weighted_mean"
        or association.get("independent_feature_extraction")
        != "separate_visit_forward_before_temporal_overlay"
        or float(geometry.get("rescene_neural_voxel_size_m", -1.0)) != 0.02
    ):
        raise ObjectLevelTransferError("association protocol is invalid")
    try:
        centroid_scale = float(association["centroid_scale_m"])
        static_tolerance = float(association["static_centroid_tolerance_m"])
        sampler_seed = association["native_sampler_seed"]
        maximum_candidates = association[
            "maximum_source_candidates_per_model_voxel"
        ]
        source_commit = rescene["source_commit"]
        device = execution["device"]
    except (KeyError, TypeError, ValueError) as error:
        raise ObjectLevelTransferError("association values are incomplete") from error
    if (
        not math.isfinite(centroid_scale)
        or centroid_scale <= 0.0
        or not math.isfinite(static_tolerance)
        or static_tolerance <= 0.0
        or type(sampler_seed) is not int
        or sampler_seed < 0
        or type(maximum_candidates) is not int
        or maximum_candidates <= 0
        or not isinstance(source_commit, str)
        or len(source_commit) != 40
        or not isinstance(device, str)
        or not device.startswith("cuda:")
    ):
        raise ObjectLevelTransferError("association runtime value is invalid")
    checkpoint = _external_binding(
        execution.get("rescene_checkpoint"), label="ReScene checkpoint"
    )
    if (
        checkpoint.path != _configured_path(
            rescene.get("checkpoint_path"), label="ReScene checkpoint"
        )
        or checkpoint.sha256 != rescene.get("checkpoint_sha256")
    ):
        raise ObjectLevelTransferError("duplicate ReScene checkpoint binding differs")
    selection_path = _configured_path(
        selection.get("path"), label="selection manifest"
    )
    selection_sha256 = selection.get("sha256")
    if not isinstance(selection_sha256, str) or len(selection_sha256) != 64:
        raise ObjectLevelTransferError("selection manifest SHA-256 is invalid")
    return ObjectTransferConfig(
        source_path=source_path,
        source_sha256=hashlib.sha256(content).hexdigest(),
        output_root=_configured_path(payload.get("output_root"), label="output root"),
        selection_manifest=selection_path,
        selection_manifest_sha256=selection_sha256,
        minimum_match_scores=MappingProxyType(scores),
        centroid_scale_m=centroid_scale,
        static_centroid_tolerance_m=static_tolerance,
        sampler_seed=sampler_seed,
        maximum_candidates=maximum_candidates,
        device=device,
        model_python=_configured_path(execution.get("model_python"), label="model Python"),
        rescene_checkout=_configured_path(
            execution.get("rescene_checkout"), label="ReScene checkout"
        ),
        source_commit=source_commit,
        concerto_checkpoint=_external_binding(
            execution.get("concerto_checkpoint"), label="Concerto checkpoint"
        ),
        rescene_checkpoint=checkpoint,
        native_sampler_source=_external_binding(
            execution.get("native_sampler_source"), label="native sampler"
        ),
        d2_mapping_receipt=_configured_path(
            execution.get("d2_mapping_receipt"), label="D2 mapping receipt"
        ),
        d2_pair_view_receipt=_configured_path(
            execution.get("d2_pair_view_receipt"), label="D2 pair receipt"
        ),
        ground_truth_root=_configured_path(
            execution.get("ground_truth_root"), label="ground-truth root"
        ),
    )


def _surface_groups(
    pair: OviObjectPairView,
) -> dict[tuple[int, str], SurfaceGroup]:
    groups: dict[tuple[int, str], SurfaceGroup] = {}
    for visit in pair.visits:
        for entity in visit.entities:
            indices = entity.point_indices[visit.appearance_valid[entity.point_indices]]
            if not len(indices):
                continue
            if not np.all(visit.normal_valid[indices]):
                raise ObjectLevelTransferError(
                    f"supported OVI entity lacks valid normals: {entity.entity_id}"
                )
            groups[(visit.visit_id, entity.entity_id)] = SurfaceGroup(
                points_xyz=visit.points_xyz[indices],
                normals_xyz=visit.normals_xyz[indices],
                normal_valid=visit.normal_valid[indices],
                original_vertex_indices=visit.source_vertex_indices[indices],
                palette_rgb=entity.palette_rgb,
            )
    return groups


def build_source_bound_model_support(
    pair: OviObjectPairView,
    surface: SurfaceAttributeBundle,
    candidates: ModelCandidateMap,
) -> RecoveredModelSupport:
    """Recover measured RGB-D provenance while retaining unsupported M rows."""

    valid = candidates.candidate_counts > 0
    model_count = len(valid)
    representatives = np.full(model_count, -1, dtype=np.int64)
    representatives[valid] = candidates.source_point_indices[valid, 0]
    rgb = np.zeros((model_count, 3), dtype=np.uint8)
    local_frames = np.full(model_count, -1, dtype=np.int64)
    source_frames = np.full(model_count, -1, dtype=np.int64)
    rows = np.full(model_count, -1, dtype=np.int64)
    columns = np.full(model_count, -1, dtype=np.int64)
    camera_depth = np.full(model_count, np.nan, dtype=np.float32)
    observed_depth = np.full(model_count, np.nan, dtype=np.float32)
    residual = np.full(model_count, np.nan, dtype=np.float32)
    visits = np.full(model_count, -1, dtype=np.int8)
    original = np.full(model_count, -1, dtype=np.int64)
    visits[valid] = surface.source_visit_ids[representatives[valid]]
    original[valid] = surface.original_vertex_indices[representatives[valid]]
    for visit_id in (0, 1):
        selected = valid & (visits == visit_id)
        dense_indices = original[selected]
        visit = pair.visits[visit_id]
        rgb[selected] = visit.camera_rgb_uint8[dense_indices]
        local_frames[selected] = visit.appearance_frame_ids[dense_indices]
        source_frames[selected] = visit.appearance_source_frame_ids[dense_indices]
        rows[selected] = visit.appearance_rows[dense_indices]
        columns[selected] = visit.appearance_columns[dense_indices]
        camera_depth[selected] = visit.appearance_camera_depth_m[dense_indices]
        observed_depth[selected] = visit.appearance_observed_depth_m[dense_indices]
        residual[selected] = visit.appearance_depth_residual_m[dense_indices]
    return RecoveredModelSupport(
        support_valid=valid,
        representative_source_point_indices=representatives,
        rgb_uint8=rgb,
        local_frame_indices=local_frames,
        global_frame_indices=source_frames,
        rows=rows,
        columns=columns,
        camera_depth_m=camera_depth,
        observed_depth_m=observed_depth,
        depth_residual_m=residual,
    )


def build_d2_inference_bundle(
    pair: OviObjectPairView,
    *,
    sampler_source_path: str | Path,
    sampler_source_sha256: str,
    neural_voxel_size_m: float = 0.02,
    sampler_seed: int = 45,
    maximum_candidates: int = 8,
    grid_sample_factory: Callable[..., object] | None = None,
) -> D2InferenceBundle:
    """Build the source-bound supported D/A/M contract for one D2 pair."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be an OviObjectPairView")
    before = pair.content_sha256()
    visits = pair.to_visit_maps(supported_only=True)
    geometry = build_adapter_geometry(
        visits[0], visits[1], voxel_size_m=neural_voxel_size_m
    )
    surface = bind_surface_attributes(
        geometry, visits[0], visits[1], _surface_groups(pair)
    )
    sampling = capture_native_sampling(
        coordinates_xyzt=geometry.coordinates_xyzt,
        visit_ids=geometry.visit_ids,
        shared_center_xyz=geometry.shared_center_xyz,
        voxel_size_m=geometry.neural_voxel_size_m,
        sampler_seed=sampler_seed,
        sampler_source_path=sampler_source_path,
        expected_sampler_source_sha256=sampler_source_sha256,
        grid_sample_factory=grid_sample_factory,
    )
    candidates = select_model_candidates(
        geometry, sampling, surface, maximum_candidates=maximum_candidates
    )
    support = build_source_bound_model_support(pair, surface, candidates)
    supported_view = build_supported_inference_view(
        _PreparedD2Components(
            geometry=geometry,
            surface=surface,
            sampling=sampling,
            candidates=candidates,
        ),
        support,
    )
    if pair.content_sha256() != before:
        raise ObjectLevelTransferError("D2 input preparation mutated the pair")
    return D2InferenceBundle(
        pair_content_sha256=before,
        candidate_ids=pair.candidate_ids,
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        support=support,
        supported_view=supported_view,
        model_input=supported_view.model_input,
        sample=supported_view.pair,
    )


def _forward_sha256(
    visit_id: int, model_indices: np.ndarray, features: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(b"OVI_RESCENE_INDEPENDENT_CONCERTO_V1\0")
    digest.update(str(visit_id).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(model_indices, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(features, dtype=np.float32).tobytes())
    return digest.hexdigest()


def execute_association_forwards(
    model_input: ReSceneModelInput,
    *,
    independent_forward: Callable[[ReSceneModelInput, int, np.ndarray], np.ndarray],
    joint_forward: Callable[[ReSceneModelInput], NativeForwardResult],
) -> AssociationForwardOutputs:
    """Execute two isolated backbone forwards before exactly one joint forward."""

    if not isinstance(model_input, ReSceneModelInput):
        raise TypeError("model_input must be a ReSceneModelInput")
    feature_parts: list[np.ndarray] = []
    hashes: list[str] = []
    runtimes: list[float] = []
    width: int | None = None
    for visit_id in (0, 1):
        indices = np.flatnonzero(model_input.model_visit_ids == visit_id).astype(
            np.int64
        )
        started = time.perf_counter()
        raw = independent_forward(model_input, visit_id, indices.copy())
        runtimes.append(time.perf_counter() - started)
        features = np.array(raw, dtype=np.float32, copy=True, order="C")
        if (
            features.ndim != 2
            or features.shape[0] != len(indices)
            or features.shape[1] < 1
            or not np.all(np.isfinite(features))
            or (width is not None and features.shape[1] != width)
        ):
            raise ObjectLevelTransferError(
                "independent backbone output has an invalid shape"
            )
        width = features.shape[1]
        hashes.append(_forward_sha256(visit_id, indices, features))
        features.setflags(write=False)
        feature_parts.append(features)
    joint = joint_forward(model_input)
    if not isinstance(joint, NativeForwardResult) or np.asarray(
        joint.pred_masks_mq
    ).shape[0] != len(model_input.model_visit_ids):
        raise ObjectLevelTransferError("joint ReScene output has an invalid M domain")
    return AssociationForwardOutputs(
        independent_model_features=(feature_parts[0], feature_parts[1]),
        visit_forward_sha256=(hashes[0], hashes[1]),
        independent_runtime_s=(runtimes[0], runtimes[1]),
        joint_forward=joint,
    )


def _verify_external_file(binding: ExternalFileBinding, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(binding.path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise ObjectLevelTransferError(f"{label} is unavailable") from error
    if not stat.S_ISREG(before.st_mode) or before.st_size != binding.byte_count:
        raise ObjectLevelTransferError(f"{label} identity is invalid")
    digest = hashlib.sha256()
    with absolute.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = absolute.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or digest.hexdigest() != binding.sha256
    ):
        raise ObjectLevelTransferError(f"{label} SHA-256 mismatch")


def _validated_checkout(config: ObjectTransferConfig) -> None:
    path = config.rescene_checkout
    if path.is_symlink() or not path.is_dir():
        raise ObjectLevelTransferError("ReScene checkout is unavailable")
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ObjectLevelTransferError("ReScene checkout cannot be verified") from error
    if commit != config.source_commit or status:
        raise ObjectLevelTransferError("ReScene checkout identity mismatch")


def run_native_association_forwards(
    bundle: D2InferenceBundle, config: ObjectTransferConfig
) -> AssociationForwardOutputs:
    """Load the pinned model once and execute the three declared forwards."""

    import hydra
    import torch
    from hydra import compose, initialize_config_dir
    from sonata.structure import Point

    if not os.path.samefile(sys.executable, config.model_python):
        raise ObjectLevelTransferError("runner must use the frozen model Python")
    _verify_external_file(config.rescene_checkpoint, label="ReScene checkpoint")
    _verify_external_file(config.concerto_checkpoint, label="Concerto checkpoint")
    _verify_external_file(config.native_sampler_source, label="native sampler source")
    if config.concerto_checkpoint.sha256 != EXPECTED_CONCERTO_SHA256:
        raise ObjectLevelTransferError("Concerto binding differs from executor contract")
    _validated_checkout(config)
    device = torch.device(config.device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise ObjectLevelTransferError("native association requires explicit CUDA")
    torch.cuda.set_device(device)
    checkout_text = str(config.rescene_checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    with initialize_config_dir(
        version_base=None, config_dir=str(config.rescene_checkout / "conf")
    ):
        model_config = compose(
            config_name="config_base_instance_segmentation",
            overrides=[
                "general.train_mode=false",
                "general.train_on_segments=true",
                "general.eval_on_segments=true",
                "general.use_dbscan=false",
                "general.gpus=1",
                "general.seed=45",
                f"backbone.name={config.concerto_checkpoint.path}",
            ],
        )
    if (
        int(model_config.model.D) != 4
        or not bool(model_config.model.train_on_segments)
        or not math.isclose(float(model_config.model.voxel_size), 0.02, abs_tol=1e-12)
    ):
        raise ObjectLevelTransferError("composed ReScene model contract is invalid")
    model = hydra.utils.instantiate(model_config.model)
    payload = torch.load(
        config.rescene_checkpoint.path, map_location="cpu", weights_only=False
    )
    state = extract_model_state_dict(payload)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ObjectLevelTransferError("strict ReScene model load is incompatible")
    del payload, state
    model.to(device)
    model.eval()

    def seed() -> None:
        random.seed(45)
        np.random.seed(45)
        torch.manual_seed(45)
        torch.cuda.manual_seed_all(45)

    def point_for(indices: np.ndarray, *, split_visits: bool) -> object:
        coordinates = torch.from_numpy(
            np.array(
                bundle.model_input.coordinates_bxyzt[indices],
                dtype=np.float32,
                copy=True,
                order="C",
            )
        ).to(device)
        offsets = (
            bundle.model_input.sparse_batch_offsets
            if split_visits
            else np.asarray([len(indices)], dtype=np.int64)
        )
        return Point(
            {
                "coord": coordinates,
                "grid_coord": torch.from_numpy(
                    np.array(
                        bundle.model_input.grid_coordinates_xyz[indices],
                        dtype=np.int32,
                        copy=True,
                        order="C",
                    )
                ).to(device),
                "feat": torch.from_numpy(
                    np.array(
                        bundle.model_input.features[indices],
                        dtype=np.float32,
                        copy=True,
                        order="C",
                    )
                ).to(device),
                "offset": torch.from_numpy(
                    np.array(offsets, dtype=np.int64, copy=True, order="C")
                ).to(device),
            }
        )

    def independent_forward(
        _model_input: ReSceneModelInput, _visit_id: int, indices: np.ndarray
    ) -> np.ndarray:
        seed()
        point = point_for(indices, split_visits=False)
        with torch.inference_mode():
            features, _auxiliary, _coordinates = model.backbone(point)
        if not hasattr(features, "F"):
            raise ObjectLevelTransferError("Concerto output lacks full-resolution features")
        result = np.ascontiguousarray(
            features.F.detach().float().cpu().numpy(), dtype=np.float32
        )
        del point, features, _auxiliary, _coordinates
        torch.cuda.empty_cache()
        return result

    def joint_forward(_model_input: ReSceneModelInput) -> NativeForwardResult:
        indices = np.arange(len(bundle.model_input.model_visit_ids), dtype=np.int64)
        point = point_for(indices, split_visits=True)
        point2segment = [
            torch.from_numpy(
                np.ascontiguousarray(
                    bundle.model_input.point2segment, dtype=np.int64
                )
            ).to(device)
        ]
        coordinates = point.coord
        seed()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(
                point,
                point2segment=point2segment,
                raw_coordinates=coordinates[:, 1:],
                is_eval=True,
            )
        torch.cuda.synchronize(device)
        runtime_s = time.perf_counter() - started
        logits = output.get("pred_logits")
        masks = output.get("pred_masks")
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[0] != 1
            or not isinstance(masks, (list, tuple))
            or len(masks) != 1
            or not isinstance(masks[0], torch.Tensor)
            or masks[0].ndim != 2
        ):
            raise ObjectLevelTransferError("native ReScene output schema is invalid")
        return NativeForwardResult(
            pred_masks_mq=np.ascontiguousarray(
                masks[0].detach().float().cpu().numpy(), dtype=np.float32
            ),
            pred_logits_qc=np.ascontiguousarray(
                logits[0].detach().float().cpu().numpy(), dtype=np.float32
            ),
            runtime_s=runtime_s,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_memory_bytes=int(torch.cuda.max_memory_reserved(device)),
            rss_peak_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            device_name=f"{torch.cuda.get_device_name(device)} ({device})",
            model_tensor_count=796,
        )

    try:
        return execute_association_forwards(
            bundle.model_input,
            independent_forward=independent_forward,
            joint_forward=joint_forward,
        )
    finally:
        torch.cuda.empty_cache()


def pool_independent_candidate_features(
    pair: OviObjectPairView,
    bundle: D2InferenceBundle,
    outputs: AssociationForwardOutputs,
) -> IndependentFeatureBank:
    """Pool independent M features over each candidate's supported D points."""

    if pair.content_sha256() != bundle.pair_content_sha256:
        raise ObjectLevelTransferError("inference bundle binds a different D2 pair")
    width = outputs.independent_model_features[0].shape[1]
    model_features = np.empty(
        (len(bundle.model_input.model_visit_ids), width), dtype=np.float32
    )
    for visit_id in (0, 1):
        model_indices = np.flatnonzero(
            bundle.model_input.model_visit_ids == visit_id
        )
        model_features[model_indices] = outputs.independent_model_features[visit_id]
    contributor_counts = np.diff(bundle.geometry.source_to_adapter_offsets)
    model_by_contributor = np.repeat(
        bundle.sampling.adapter_to_model, contributor_counts
    )
    source_to_old_model = np.empty(
        bundle.geometry.source_point_count, dtype=np.int64
    )
    source_to_old_model[bundle.geometry.source_point_indices] = model_by_contributor
    source_to_model = bundle.supported_view.old_to_new_model_indices[
        source_to_old_model
    ]
    entity_index = {key: index for index, key in enumerate(bundle.geometry.entity_keys)}
    pooled: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    for visit_candidates in pair.candidate_ids:
        features = np.zeros((len(visit_candidates), width), dtype=np.float32)
        available = np.zeros(len(visit_candidates), dtype=bool)
        for candidate_index, key in enumerate(visit_candidates):
            index = entity_index.get(key)
            if index is None:
                continue
            sources = np.flatnonzero(bundle.surface.source_entity_indices == index)
            model_indices = source_to_model[sources]
            model_indices = model_indices[model_indices >= 0]
            if not len(model_indices):
                continue
            features[candidate_index] = np.mean(
                model_features[model_indices], axis=0, dtype=np.float64
            )
            available[candidate_index] = True
        pooled.append(features)
        valid.append(available)
    return IndependentFeatureBank(
        pair_content_sha256=bundle.pair_content_sha256,
        candidate_ids=pair.candidate_ids,
        features=(pooled[0], pooled[1]),
        valid=(valid[0], valid[1]),
        extraction_mode="separate_visit_forward_before_temporal_overlay",
        visit_forward_sha256=outputs.visit_forward_sha256,
    )


def build_temporal_query_evidence(
    bundle: D2InferenceBundle,
    outputs: AssociationForwardOutputs,
    *,
    backend_config_sha256: str,
    checkpoint_sha256: str,
) -> TemporalQueryEvidence:
    """Expand the single joint ReScene forward onto the bound adapter domain."""

    processed = postprocess_native_predictions(
        pred_masks_mq=outputs.joint_forward.pred_masks_mq,
        pred_logits_qc=outputs.joint_forward.pred_logits_qc,
        adapter_to_model=bundle.model_input.adapter_to_model,
    )
    if not processed.temporal_query_ids:
        raise ObjectLevelTransferError("joint ReScene output has no supported query")
    return TemporalQueryEvidence(
        status="PASS",
        backend_name="rescene:ovi-object-transfer-v1",
        backend_config_sha256=backend_config_sha256,
        pair_sha256=bundle.sample.content_sha256(),
        temporal_query_ids=processed.temporal_query_ids,
        query_masks=processed.query_masks_qa,
        token_scores=processed.token_scores_qa,
        query_scores=processed.query_scores,
        checkpoint_sha256=checkpoint_sha256,
        ranking_eligible=True,
        runtime_s=outputs.joint_forward.runtime_s,
        peak_memory_bytes=outputs.joint_forward.peak_memory_bytes,
    )


def evaluate_shared_candidate_methods(
    pair: OviObjectPairView,
    bundle: D2InferenceBundle,
    outputs: AssociationForwardOutputs,
    ground_truth: GroundTruthPair,
    *,
    minimum_match_scores: Mapping[str, float],
    centroid_scale_m: float,
    static_centroid_tolerance_m: float,
    backend_config_sha256: str,
    checkpoint_sha256: str,
) -> SharedCandidateEvaluation:
    """Evaluate G/F/R with fixed candidates and output-independent GT bindings."""

    if set(minimum_match_scores) != set(_METHOD_IDS):
        raise ObjectLevelTransferError("minimum scores must cover all G/F/R methods")
    before = pair.content_sha256()
    bank = pool_independent_candidate_features(pair, bundle, outputs)
    evidence = build_temporal_query_evidence(
        bundle,
        outputs,
        backend_config_sha256=backend_config_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    matrices = {
        "G_full": build_geometric_score_matrix(
            pair, supported_only=False, centroid_scale_m=centroid_scale_m
        ),
        "G_supported": build_geometric_score_matrix(
            pair, supported_only=True, centroid_scale_m=centroid_scale_m
        ),
        "F_obj": build_feature_score_matrix(pair, bank),
        "R_obj": build_rescene_score_matrix(pair, bundle.sample, evidence),
    }
    bindings = {
        "full": build_fixed_endpoint_bindings(
            pair, ground_truth, support_domain="full"
        ),
        "supported": build_fixed_endpoint_bindings(
            pair, ground_truth, support_domain="supported"
        ),
    }
    predictions: dict[str, tuple[ObjectPairPrediction, ...]] = {}
    metrics: dict[str, Mapping[str, Mapping[str, object]]] = {}
    for method in _METHOD_IDS:
        prediction = solve_object_pair_assignment(
            pair,
            matrices[method],
            AssociationConfig(
                minimum_match_score=float(minimum_match_scores[method]),
                static_centroid_tolerance_m=static_centroid_tolerance_m,
            ),
        )
        endpoint_binding = bindings["full" if method == "G_full" else "supported"]
        predictions[method] = prediction
        metrics[method] = MappingProxyType(
            {
                "0.50": MappingProxyType(
                    evaluate_fixed_object_predictions(
                        pair,
                        prediction,
                        ground_truth,
                        endpoint_binding,
                        iou_threshold=0.50,
                    )
                ),
                "0.25": MappingProxyType(
                    evaluate_fixed_object_predictions(
                        pair,
                        prediction,
                        ground_truth,
                        endpoint_binding,
                        iou_threshold=0.25,
                    )
                ),
            }
        )
    if pair.content_sha256() != before:
        raise ObjectLevelTransferError("shared-candidate evaluation mutated the pair")
    return SharedCandidateEvaluation(
        matrices=MappingProxyType(matrices),
        predictions=MappingProxyType(predictions),
        metrics=MappingProxyType(metrics),
        endpoint_bindings=MappingProxyType(bindings),
        feature_bank=bank,
        query_evidence=evidence,
    )


def build_shared_candidate_rows(
    pair: OviObjectPairView,
    bundle: D2InferenceBundle,
    outputs: AssociationForwardOutputs,
    evaluation: SharedCandidateEvaluation,
    *,
    evaluated_commit: str,
    config_sha256: str,
    checkpoint_sha256: str,
) -> tuple[dict[str, object], ...]:
    """Flatten measured G/F/R results without replacing undefined rates by zero."""

    if pair.content_sha256() != bundle.pair_content_sha256:
        raise ObjectLevelTransferError("result bundle binds a different D2 pair")
    if not isinstance(evaluated_commit, str) or len(evaluated_commit) != 40 or any(
        value not in "0123456789abcdef" for value in evaluated_commit
    ):
        raise ObjectLevelTransferError("evaluated commit is invalid")
    for value, label in (
        (config_sha256, "config SHA-256"),
        (checkpoint_sha256, "checkpoint SHA-256"),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ObjectLevelTransferError(f"{label} is invalid")
    rows: list[dict[str, object]] = []
    for method in _METHOD_IDS:
        for threshold_key in ("0.50", "0.25"):
            metrics = evaluation.metrics[method][threshold_key]
            row: dict[str, object] = {
                "pair_id": pair.pair_id,
                "method_id": method,
                "endpoint_iou_threshold": float(metrics["iou_threshold"]),
                "evaluated_commit": evaluated_commit,
                "t0_candidate_count": len(pair.candidate_ids[0]),
                "t1_candidate_count": len(pair.candidate_ids[1]),
                "t0_feature_supported_candidate_count": int(
                    np.count_nonzero(evaluation.feature_bank.valid[0])
                ),
                "t1_feature_supported_candidate_count": int(
                    np.count_nonzero(evaluation.feature_bank.valid[1])
                ),
                "query_count": len(evaluation.query_evidence.temporal_query_ids),
                "full_source_point_count": bundle.geometry.source_point_count,
                "full_adapter_count": bundle.geometry.adapter_count,
                "full_model_count": bundle.sampling.model_count,
                "supported_source_point_count": bundle.sample.source_point_count,
                "supported_adapter_count": len(bundle.sample.visit_ids),
                "supported_model_count": len(bundle.model_input.model_visit_ids),
                "independent_t0_runtime_s": outputs.independent_runtime_s[0],
                "independent_t1_runtime_s": outputs.independent_runtime_s[1],
                "joint_runtime_s": outputs.joint_forward.runtime_s,
                "peak_memory_bytes": outputs.joint_forward.peak_memory_bytes,
                "pair_content_sha256": bundle.pair_content_sha256,
                "sample_content_sha256": bundle.sample.content_sha256(),
                "binding_sha256": metrics["binding_sha256"],
                "config_sha256": config_sha256,
                "checkpoint_sha256": checkpoint_sha256,
            }
            row.update({key: metrics[key] for key in _RESULT_METRIC_FIELDS})
            rows.append(row)
    return tuple(rows)


def write_shared_candidate_csv(
    path: str | Path, rows: tuple[Mapping[str, object], ...]
) -> Path:
    """Write one schema-consistent measured result table using repository LF CSV."""

    if not isinstance(rows, tuple) or not rows:
        raise ObjectLevelTransferError("shared-candidate result rows are empty")
    fields = tuple(rows[0])
    if not fields or any(tuple(row) != fields for row in rows):
        raise ObjectLevelTransferError("shared-candidate rows use inconsistent fields")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    return output


__all__ = [
    "AssociationForwardOutputs",
    "D2InferenceBundle",
    "ObjectLevelTransferError",
    "ObjectTransferConfig",
    "SharedCandidateEvaluation",
    "build_d1_sensor_support_view",
    "build_d2_inference_bundle",
    "build_shared_candidate_rows",
    "build_source_bound_model_support",
    "build_temporal_query_evidence",
    "evaluate_shared_candidate_methods",
    "execute_association_forwards",
    "load_object_transfer_config",
    "measure_native_candidate_representation",
    "pool_independent_candidate_features",
    "project_cached_relations_to_sensor_support",
    "run_native_association_forwards",
    "write_shared_candidate_csv",
]
