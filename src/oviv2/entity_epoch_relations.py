"""Strong conservative entity relations over fixed two-visit OVI samples."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from src.oviv2.entity_epoch_update import RelationInferenceState, RelationSupport
from src.oviv2.query_instance_projection import (
    ProjectionConfig,
    QueryEntityEvidence,
    project_query_evidence,
)
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    PairRelation,
    TemporalQueryEvidence,
)
from src.oviv2.two_visit_registration import (
    RegistrationConfig,
    RegistrationEvidence,
    apply_rigid_transform,
    register_composite_relation,
    validate_rigid_transform,
)


@dataclass(frozen=True, slots=True)
class StrongGeometricRelationConfig:
    registration_config: RegistrationConfig = field(default_factory=RegistrationConfig)
    comparison_voxel_size_m: float = 0.05
    maximum_candidate_centroid_distance_m: float = 2.0
    minimum_relation_score: float = 0.55
    null_score: float = 0.50
    minimum_assignment_margin: float = 0.08
    maximum_registration_candidates_per_entity: int = 4
    minimum_separated_patches: int = 3
    patch_voxel_size_m: float = 0.05
    minimum_motion_residual_improvement_m: float = 0.02
    maximum_static_identity_residual_m: float = 0.03
    minimum_volumetric_singular_ratio: float = 0.02
    minimum_symmetry_separation: float = 0.08
    semantic_mismatch_score: float = 0.25
    centered_shape_weight: float = 0.22
    size_weight: float = 0.10
    extent_weight: float = 0.10
    registration_weight: float = 0.25
    visible_support_weight: float = 0.05
    appearance_weight: float = 0.18
    semantic_weight: float = 0.05
    original_location_iou_weight: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.registration_config, RegistrationConfig):
            raise TypeError("registration_config must be RegistrationConfig")
        for name in (
            "comparison_voxel_size_m",
            "maximum_candidate_centroid_distance_m",
            "patch_voxel_size_m",
            "minimum_motion_residual_improvement_m",
            "maximum_static_identity_residual_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in (
            "minimum_relation_score",
            "null_score",
            "minimum_assignment_margin",
            "minimum_volumetric_singular_ratio",
            "minimum_symmetry_separation",
            "semantic_mismatch_score",
            "centered_shape_weight",
            "size_weight",
            "extent_weight",
            "registration_weight",
            "visible_support_weight",
            "appearance_weight",
            "semantic_weight",
            "original_location_iou_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "maximum_registration_candidates_per_entity",
            "minimum_separated_patches",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        weights = (
            self.centered_shape_weight,
            self.size_weight,
            self.extent_weight,
            self.registration_weight,
            self.visible_support_weight,
            self.appearance_weight,
            self.semantic_weight,
            self.original_location_iou_weight,
        )
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("relation score weights must sum to one")

    def content_sha256(self) -> str:
        payload = asdict(self)
        payload["registration_config"] = self.registration_config.to_json_record()
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class RelationMotionVerification:
    state: RelationInferenceState
    motion_verified: bool
    transform_world_from_t0: np.ndarray | None
    registration_evidence: RegistrationEvidence | None
    registration_score: float | None
    identity_symmetric_median_m: float
    registered_symmetric_median_m: float | None
    residual_improvement_m: float | None
    spatial_support_patch_count: int
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.state, RelationInferenceState):
            raise TypeError("state must be RelationInferenceState")
        if type(self.motion_verified) is not bool:
            raise TypeError("motion_verified must be boolean")
        if self.motion_verified and self.state is not RelationInferenceState.MOVED:
            raise ValueError("motion_verified requires the moved state")
        transform = self.transform_world_from_t0
        if transform is not None:
            transform = validate_rigid_transform(transform)
            object.__setattr__(self, "transform_world_from_t0", transform)
        if (self.state is RelationInferenceState.UNRESOLVED) != (transform is None):
            raise ValueError("only resolved motion states may expose a transform")
        if self.registration_evidence is not None and not isinstance(
            self.registration_evidence, RegistrationEvidence
        ):
            raise TypeError("registration_evidence must be RegistrationEvidence")
        for name in (
            "identity_symmetric_median_m",
            "registered_symmetric_median_m",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.registration_score is not None and (
            not math.isfinite(float(self.registration_score))
            or not 0.0 <= float(self.registration_score) <= 1.0
        ):
            raise ValueError("registration_score must be finite and in [0, 1]")
        if self.residual_improvement_m is not None and not math.isfinite(
            float(self.residual_improvement_m)
        ):
            raise ValueError("residual_improvement_m must be finite")
        if (
            type(self.spatial_support_patch_count) is not int
            or self.spatial_support_patch_count < 0
        ):
            raise ValueError("spatial_support_patch_count must be nonnegative")
        reasons = tuple(sorted(set(self.rejection_reasons)))
        if reasons != self.rejection_reasons:
            raise ValueError("motion rejection reasons must be ordered and unique")
        if (self.state is RelationInferenceState.UNRESOLVED) != bool(reasons):
            raise ValueError("motion rejection reasons must match unresolved state")


@dataclass(frozen=True, slots=True)
class _EntitySummary:
    visit_id: int
    entity_id: str
    owner_entity_id: int
    source_surface_id: str
    source_vertex_indices: np.ndarray
    points_xyz: np.ndarray
    centroid_xyz: np.ndarray
    extent_xyz: np.ndarray
    centered_voxels: frozenset[tuple[int, int, int]]
    world_voxels: frozenset[tuple[int, int, int]]
    source_point_count: int
    semantic_label: str | None
    semantic_embedding: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _Candidate:
    t0_index: int
    t1_index: int
    centroid_distance_m: float
    centered_shape_score: float
    size_score: float
    extent_score: float
    appearance_cosine: float | None
    semantic_compatibility: float | None
    original_location_iou: float
    cheap_score: float
    motion: RelationMotionVerification | None = None
    final_score: float | None = None


def _owner_id(entity_id: str) -> int:
    if not entity_id.startswith("ovimap:"):
        raise ValueError("G1 requires source-bound ovimap:<positive-int> entity IDs")
    try:
        value = int(entity_id.split(":", 1)[1])
    except ValueError as error:
        raise ValueError(
            "G1 requires source-bound ovimap:<positive-int> entity IDs"
        ) from error
    if value <= 0:
        raise ValueError("G1 OVI owner IDs must be positive")
    return value


def _voxel_set(
    points: np.ndarray, voxel_size_m: float
) -> frozenset[tuple[int, int, int]]:
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    return frozenset(tuple(int(value) for value in row) for row in keys)


def _intersection_over_union(
    first: frozenset[tuple[int, int, int]],
    second: frozenset[tuple[int, int, int]],
) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _ratio(first: float, second: float) -> float:
    if first == second == 0.0:
        return 1.0
    maximum = max(first, second)
    return min(first, second) / maximum if maximum > 0.0 else 0.0


def _extent_score(first: np.ndarray, second: np.ndarray) -> float:
    first_sorted = np.sort(first)
    second_sorted = np.sort(second)
    return float(
        np.mean(
            [
                _ratio(float(first_value), float(second_value))
                for first_value, second_value in zip(
                    first_sorted, second_sorted, strict=True
                )
            ]
        )
    )


def _cosine(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None or first.shape != second.shape:
        return None
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0.0:
        return None
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _semantic_score(
    first: str | None, second: str | None, mismatch: float
) -> float | None:
    if first is None or second is None:
        return None
    return 1.0 if first.casefold() == second.casefold() else mismatch


def _weighted_score(
    candidate: _Candidate,
    config: StrongGeometricRelationConfig,
    *,
    visible_support: float | None,
) -> float:
    components = (
        (config.centered_shape_weight, candidate.centered_shape_score),
        (config.size_weight, candidate.size_score),
        (config.extent_weight, candidate.extent_score),
        (
            config.registration_weight,
            None if candidate.motion is None else candidate.motion.registration_score,
        ),
        (config.visible_support_weight, visible_support),
        (
            config.appearance_weight,
            None
            if candidate.appearance_cosine is None
            else (candidate.appearance_cosine + 1.0) / 2.0,
        ),
        (config.semantic_weight, candidate.semantic_compatibility),
        (config.original_location_iou_weight, candidate.original_location_iou),
    )
    available = tuple(
        (weight, value) for weight, value in components if value is not None
    )
    denominator = sum(weight for weight, _value in available)
    if denominator <= 0.0:
        return 0.0
    return float(
        sum(weight * float(value) for weight, value in available) / denominator
    )


def _source_local_rows(pair: NeuralSampleMap) -> np.ndarray:
    result = np.empty(pair.source_point_count, dtype=np.int64)
    for visit_id in (0, 1):
        positions = np.flatnonzero(pair.source_visit_ids == visit_id)
        global_rows = pair.source_point_indices[positions]
        result[positions] = np.searchsorted(np.sort(global_rows), global_rows)
    return result


def _summaries(
    pair: NeuralSampleMap, voxel_size_m: float
) -> tuple[tuple[_EntitySummary, ...], tuple[_EntitySummary, ...]]:
    semantics = {
        (item.visit_id, item.entity_id): item for item in pair.entity_semantics
    }
    contributor_local_rows = _source_local_rows(pair)
    token_groups: dict[tuple[int, str], list[int]] = {}
    for token_index, entity_id in enumerate(pair.token_entity_ids):
        key = (int(pair.visit_ids[token_index]), entity_id)
        token_groups.setdefault(key, []).append(token_index)
    values: list[list[_EntitySummary]] = [[], []]
    for (visit_id, entity_id), token_indices in sorted(token_groups.items()):
        indices = np.asarray(token_indices, dtype=np.int64)
        points = np.asarray(pair.coordinates_xyzt[indices, :3], dtype=np.float64)
        counts = np.diff(pair.source_to_token_offsets)[indices].astype(np.float64)
        centroid = np.average(points, axis=0, weights=counts)
        extent = points.max(axis=0) - points.min(axis=0)
        contributor_positions = np.concatenate(
            [
                np.arange(
                    pair.source_to_token_offsets[index],
                    pair.source_to_token_offsets[index + 1],
                    dtype=np.int64,
                )
                for index in indices
            ]
        )
        rows = np.sort(contributor_local_rows[contributor_positions])
        semantic = semantics.get((visit_id, entity_id))
        values[visit_id].append(
            _EntitySummary(
                visit_id=visit_id,
                entity_id=entity_id,
                owner_entity_id=_owner_id(entity_id),
                source_surface_id=f"ovi-map:{pair.source_visit_map_sha256[visit_id]}",
                source_vertex_indices=rows,
                points_xyz=points,
                centroid_xyz=centroid,
                extent_xyz=extent,
                centered_voxels=_voxel_set(points - centroid, voxel_size_m),
                world_voxels=_voxel_set(points, voxel_size_m),
                source_point_count=int(counts.sum()),
                semantic_label=None if semantic is None else semantic.semantic_label,
                semantic_embedding=(
                    None if semantic is None else semantic.semantic_embedding
                ),
            )
        )
    return tuple(values[0]), tuple(values[1])


def _symmetric_median(first: np.ndarray, second: np.ndarray) -> float:
    first_to_second = cKDTree(second).query(first, k=1, workers=1)[0]
    second_to_first = cKDTree(first).query(second, k=1, workers=1)[0]
    return 0.5 * (float(np.median(first_to_second)) + float(np.median(second_to_first)))


def _registration_score(evidence: RegistrationEvidence) -> float:
    assert evidence.source_to_target_overlap is not None
    assert evidence.target_to_source_overlap is not None
    assert evidence.source_to_target_median_m is not None
    assert evidence.target_to_source_median_m is not None
    overlap = 0.5 * (
        evidence.source_to_target_overlap + evidence.target_to_source_overlap
    )
    median = 0.5 * (
        evidence.source_to_target_median_m + evidence.target_to_source_median_m
    )
    residual = math.exp(-median / 0.05)
    return float(np.clip(0.7 * overlap + 0.3 * residual, 0.0, 1.0))


def _spatial_patch_count(
    source: np.ndarray,
    target: np.ndarray,
    *,
    transform: np.ndarray,
    inlier_distance_m: float,
    voxel_size_m: float,
) -> int:
    transformed = apply_rigid_transform(source, transform)
    source_distances = cKDTree(target).query(transformed, k=1, workers=1)[0]
    target_distances = cKDTree(transformed).query(target, k=1, workers=1)[0]
    source_patches = _voxel_set(
        transformed[source_distances <= inlier_distance_m], voxel_size_m
    )
    target_patches = _voxel_set(
        target[target_distances <= inlier_distance_m], voxel_size_m
    )
    return min(len(source_patches), len(target_patches))


def _ambiguous_motion_geometry(
    points: np.ndarray,
    *,
    minimum_volumetric_ratio: float,
    minimum_symmetry_separation: float,
) -> bool:
    if len(points) < 4:
        return True
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if singular[0] <= np.finfo(np.float64).eps:
        return True
    if singular[2] / singular[0] < minimum_volumetric_ratio:
        return True
    first_gap = 1.0 - singular[1] / singular[0]
    second_gap = 1.0 - singular[2] / singular[1]
    return (
        first_gap < minimum_symmetry_separation
        and second_gap < minimum_symmetry_separation
    )


def verify_relation_motion(
    *,
    relation: RelationSupport,
    source_points_xyz: object,
    target_points_xyz: object,
    registration_config: RegistrationConfig,
    minimum_separated_patches: int,
    patch_voxel_size_m: float,
    minimum_residual_improvement_m: float = 0.02,
    maximum_static_identity_residual_m: float = 0.03,
    minimum_volumetric_singular_ratio: float = 0.02,
    minimum_symmetry_separation: float = 0.08,
) -> RelationMotionVerification:
    """Verify static or moved geometry without using centroid displacement alone."""

    if not isinstance(relation, RelationSupport):
        raise TypeError("relation must be RelationSupport")
    if not isinstance(registration_config, RegistrationConfig):
        raise TypeError("registration_config must be RegistrationConfig")
    source = np.asarray(source_points_xyz, dtype=np.float64)
    target = np.asarray(target_points_xyz, dtype=np.float64)
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1:] != (3,)
        or target.shape[1:] != (3,)
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or not len(source)
        or not len(target)
    ):
        raise ValueError("motion point clouds must be non-empty finite (N, 3) arrays")
    if type(minimum_separated_patches) is not int or minimum_separated_patches < 1:
        raise ValueError("minimum_separated_patches must be a positive integer")
    for value, name in (
        (patch_voxel_size_m, "patch_voxel_size_m"),
        (minimum_residual_improvement_m, "minimum_residual_improvement_m"),
        (maximum_static_identity_residual_m, "maximum_static_identity_residual_m"),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    identity_residual = _symmetric_median(source, target)
    raw_patch_count = min(
        len(_voxel_set(source, float(patch_voxel_size_m))),
        len(_voxel_set(target, float(patch_voxel_size_m))),
    )
    if raw_patch_count < minimum_separated_patches:
        return RelationMotionVerification(
            state=RelationInferenceState.UNRESOLVED,
            motion_verified=False,
            transform_world_from_t0=None,
            registration_evidence=None,
            registration_score=None,
            identity_symmetric_median_m=identity_residual,
            registered_symmetric_median_m=None,
            residual_improvement_m=None,
            spatial_support_patch_count=raw_patch_count,
            rejection_reasons=("insufficient_spatial_support",),
        )
    pair_relation = PairRelation(
        temporal_query_id=relation.relation_id,
        t0_entity_ids=(
            relation.t0_entity_id or f"ovimap:{relation.t0_owner_entity_id}",
        ),
        t1_entity_ids=(
            relation.t1_entity_id or f"ovimap:{relation.t1_owner_entity_id}",
        ),
        state="persistent_moved",
        query_confidence=relation.confidence,
        evidence={},
        identity_source="geometric_baseline",
    )
    registration = register_composite_relation(
        pair_relation,
        source,
        target,
        source_semantic_label=None,
        target_semantic_label=None,
        config=registration_config,
    )
    if not registration.accepted:
        return RelationMotionVerification(
            state=RelationInferenceState.UNRESOLVED,
            motion_verified=False,
            transform_world_from_t0=None,
            registration_evidence=registration,
            registration_score=0.0,
            identity_symmetric_median_m=identity_residual,
            registered_symmetric_median_m=None,
            residual_improvement_m=None,
            spatial_support_patch_count=raw_patch_count,
            rejection_reasons=tuple(
                sorted(
                    f"registration:{reason}"
                    for reason in registration.rejection_reasons
                )
            ),
        )
    assert registration.transform_world_from_t0 is not None
    transformed = apply_rigid_transform(source, registration.transform_world_from_t0)
    registered_residual = _symmetric_median(transformed, target)
    improvement = identity_residual - registered_residual
    patch_count = _spatial_patch_count(
        source,
        target,
        transform=registration.transform_world_from_t0,
        inlier_distance_m=registration_config.inlier_distance_m,
        voxel_size_m=float(patch_voxel_size_m),
    )
    centroid_displacement = float(
        np.linalg.norm(source.mean(axis=0) - target.mean(axis=0))
    )
    reasons: set[str] = set()
    if patch_count < minimum_separated_patches:
        reasons.add("insufficient_spatial_support")
    is_static = (
        identity_residual <= maximum_static_identity_residual_m
        and centroid_displacement
        <= registration_config.maximum_static_centroid_displacement_m
    )
    if is_static and not reasons:
        state = RelationInferenceState.STATIC
    else:
        if improvement < minimum_residual_improvement_m:
            reasons.add("insufficient_residual_improvement")
        if _ambiguous_motion_geometry(
            source,
            minimum_volumetric_ratio=minimum_volumetric_singular_ratio,
            minimum_symmetry_separation=minimum_symmetry_separation,
        ) or _ambiguous_motion_geometry(
            target,
            minimum_volumetric_ratio=minimum_volumetric_singular_ratio,
            minimum_symmetry_separation=minimum_symmetry_separation,
        ):
            reasons.add("ambiguous_motion_geometry")
        state = (
            RelationInferenceState.MOVED
            if not reasons
            else RelationInferenceState.UNRESOLVED
        )
    return RelationMotionVerification(
        state=state,
        motion_verified=state is RelationInferenceState.MOVED,
        transform_world_from_t0=(
            registration.transform_world_from_t0
            if state is not RelationInferenceState.UNRESOLVED
            else None
        ),
        registration_evidence=registration,
        registration_score=_registration_score(registration),
        identity_symmetric_median_m=identity_residual,
        registered_symmetric_median_m=registered_residual,
        residual_improvement_m=improvement,
        spatial_support_patch_count=patch_count,
        rejection_reasons=tuple(sorted(reasons)),
    )


def _candidate(
    first: _EntitySummary,
    second: _EntitySummary,
    t0_index: int,
    t1_index: int,
    config: StrongGeometricRelationConfig,
) -> _Candidate:
    distance = float(np.linalg.norm(first.centroid_xyz - second.centroid_xyz))
    centered = _intersection_over_union(first.centered_voxels, second.centered_voxels)
    size = _ratio(first.source_point_count, second.source_point_count)
    extent = _extent_score(first.extent_xyz, second.extent_xyz)
    appearance = _cosine(first.semantic_embedding, second.semantic_embedding)
    semantic = _semantic_score(
        first.semantic_label, second.semantic_label, config.semantic_mismatch_score
    )
    original_iou = _intersection_over_union(first.world_voxels, second.world_voxels)
    draft = _Candidate(
        t0_index=t0_index,
        t1_index=t1_index,
        centroid_distance_m=distance,
        centered_shape_score=centered,
        size_score=size,
        extent_score=extent,
        appearance_cosine=appearance,
        semantic_compatibility=semantic,
        original_location_iou=original_iou,
        cheap_score=0.0,
    )
    return _Candidate(
        **{
            **asdict(draft),
            "cheap_score": _weighted_score(draft, config, visible_support=None),
        }
    )


def _registration_pool(
    candidates: Mapping[tuple[int, int], _Candidate],
    t0_count: int,
    t1_count: int,
    config: StrongGeometricRelationConfig,
) -> set[tuple[int, int]]:
    pool: set[tuple[int, int]] = set()
    limit = config.maximum_registration_candidates_per_entity
    for first_index in range(t0_count):
        row = [
            candidate
            for candidate in candidates.values()
            if candidate.t0_index == first_index
            and candidate.centroid_distance_m
            <= config.maximum_candidate_centroid_distance_m
        ]
        row.sort(key=lambda item: (-item.cheap_score, item.t1_index))
        pool.update((item.t0_index, item.t1_index) for item in row[:limit])
    for second_index in range(t1_count):
        column = [
            candidate
            for candidate in candidates.values()
            if candidate.t1_index == second_index
            and candidate.centroid_distance_m
            <= config.maximum_candidate_centroid_distance_m
        ]
        column.sort(key=lambda item: (-item.cheap_score, item.t0_index))
        pool.update((item.t0_index, item.t1_index) for item in column[:limit])
    return pool


def _preliminary_relation(
    first: _EntitySummary,
    second: _EntitySummary,
    candidate: _Candidate,
) -> RelationSupport:
    source_identity = first.source_surface_id.rsplit(":", 1)[-1][:12]
    return RelationSupport(
        relation_id=(
            f"geometry-g1:{source_identity}:"
            f"{first.owner_entity_id}:{second.owner_entity_id}"
        ),
        relation_source="geometry-g1",
        stable_entity_id=f"stable:g1:{source_identity}:{first.owner_entity_id}",
        t0_owner_entity_id=first.owner_entity_id,
        t1_owner_entity_id=second.owner_entity_id,
        t0_source_surface_id=first.source_surface_id,
        t1_source_surface_id=second.source_surface_id,
        t0_source_vertex_indices=first.source_vertex_indices,
        t1_source_vertex_indices=second.source_vertex_indices,
        confidence=candidate.cheap_score,
        inference_state=RelationInferenceState.UNRESOLVED,
        accepted=True,
        t0_entity_id=first.entity_id,
        t1_entity_id=second.entity_id,
    )


def build_g1_relation_support(
    pair: NeuralSampleMap,
    config: StrongGeometricRelationConfig,
    *,
    visible_support_by_entity: Mapping[tuple[int, str], float] | None = None,
) -> tuple[RelationSupport, ...]:
    """Build one conservative G1 assignment per t0 OVI entity."""

    if not isinstance(pair, NeuralSampleMap):
        raise TypeError("pair must be NeuralSampleMap")
    if not isinstance(config, StrongGeometricRelationConfig):
        raise TypeError("config must be StrongGeometricRelationConfig")
    visible = (
        {} if visible_support_by_entity is None else dict(visible_support_by_entity)
    )
    for key, value in visible.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] not in {0, 1}
            or not isinstance(key[1], str)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("visible support must map entity keys to values in [0, 1]")
    t0, t1 = _summaries(pair, config.comparison_voxel_size_m)
    candidates = {
        (first_index, second_index): _candidate(
            first, second, first_index, second_index, config
        )
        for first_index, first in enumerate(t0)
        for second_index, second in enumerate(t1)
    }
    pool = _registration_pool(candidates, len(t0), len(t1), config)
    evaluated: dict[tuple[int, int], _Candidate] = dict(candidates)
    for key in sorted(pool):
        candidate = candidates[key]
        first = t0[candidate.t0_index]
        second = t1[candidate.t1_index]
        preliminary = _preliminary_relation(first, second, candidate)
        motion = verify_relation_motion(
            relation=preliminary,
            source_points_xyz=first.points_xyz,
            target_points_xyz=second.points_xyz,
            registration_config=config.registration_config,
            minimum_separated_patches=config.minimum_separated_patches,
            patch_voxel_size_m=config.patch_voxel_size_m,
            minimum_residual_improvement_m=(
                config.minimum_motion_residual_improvement_m
            ),
            maximum_static_identity_residual_m=(
                config.maximum_static_identity_residual_m
            ),
            minimum_volumetric_singular_ratio=(
                config.minimum_volumetric_singular_ratio
            ),
            minimum_symmetry_separation=config.minimum_symmetry_separation,
        )
        visible_values = [
            visible.get((0, first.entity_id)),
            visible.get((1, second.entity_id)),
        ]
        visible_score = (
            None
            if any(value is None for value in visible_values)
            else float(min(value for value in visible_values if value is not None))
        )
        draft = _Candidate(**{**asdict(candidate), "motion": motion})
        evaluated[key] = _Candidate(
            **{
                **asdict(draft),
                "motion": motion,
                "final_score": _weighted_score(
                    draft, config, visible_support=visible_score
                ),
            }
        )
    real_scores = np.full((len(t0), len(t1)), -1.0e6, dtype=np.float64)
    for first_index in range(len(t0)):
        row = [evaluated[key] for key in pool if key[0] == first_index]
        for candidate in row:
            value = candidate.final_score
            assert value is not None
            alternatives = [
                config.null_score,
                *[
                    float(other.final_score)
                    for other in row
                    if other.t1_index != candidate.t1_index
                    and other.final_score is not None
                ],
            ]
            margin = value - max(alternatives)
            if (
                value >= config.minimum_relation_score
                and value >= config.null_score
                and margin >= config.minimum_assignment_margin
                and candidate.motion is not None
                and candidate.motion.spatial_support_patch_count
                >= config.minimum_separated_patches
            ):
                real_scores[candidate.t0_index, candidate.t1_index] = value
    costs = np.full((len(t0), len(t1) + len(t0)), 1.0e6, dtype=np.float64)
    costs[:, : len(t1)] = -real_scores
    for index in range(len(t0)):
        costs[index, len(t1) + index] = -config.null_score
    rows, columns = linear_sum_assignment(costs)
    assigned = dict(zip(rows.tolist(), columns.tolist(), strict=True))
    output: list[RelationSupport] = []
    for first_index, first in enumerate(t0):
        assigned_column = assigned[first_index]
        row_candidates = [
            evaluated[(first_index, second_index)]
            for second_index in range(len(t1))
            if (first_index, second_index) in pool
        ]
        if not row_candidates:
            row_candidates = [
                candidates[(first_index, second_index)]
                for second_index in range(len(t1))
            ]
        row_candidates.sort(
            key=lambda item: (
                -(
                    item.final_score
                    if item.final_score is not None
                    else item.cheap_score
                ),
                item.t1_index,
            )
        )
        selected = (
            evaluated[(first_index, assigned_column)]
            if assigned_column < len(t1)
            else row_candidates[0]
        )
        selected_score = (
            selected.final_score
            if selected.final_score is not None
            else selected.cheap_score
        )
        alternatives = [
            config.null_score,
            *[
                candidate.final_score
                if candidate.final_score is not None
                else candidate.cheap_score
                for candidate in row_candidates
                if candidate.t1_index != selected.t1_index
            ],
        ]
        margin = selected_score - max(alternatives)
        reasons: set[str] = set()
        if assigned_column >= len(t1):
            reasons.add("assigned_null")
        if selected_score < config.minimum_relation_score:
            reasons.add("below_relation_score")
        if selected_score < config.null_score:
            reasons.add("below_null_score")
        if margin < config.minimum_assignment_margin:
            reasons.add("ambiguous_assignment")
        if (
            selected.motion is not None
            and selected.motion.spatial_support_patch_count
            < config.minimum_separated_patches
        ):
            reasons.add("insufficient_spatial_support")
        accepted = not reasons and assigned_column < len(t1)
        second = t1[selected.t1_index]
        motion = selected.motion
        source_identity = first.source_surface_id.rsplit(":", 1)[-1][:12]
        visible_values = [
            visible.get((0, first.entity_id)),
            visible.get((1, second.entity_id)),
        ]
        visible_score = (
            None
            if any(value is None for value in visible_values)
            else float(min(value for value in visible_values if value is not None))
        )
        output.append(
            RelationSupport(
                relation_id=(
                    f"geometry-g1:{source_identity}:"
                    f"{first.owner_entity_id}:{second.owner_entity_id}"
                ),
                relation_source="geometry-g1",
                stable_entity_id=(
                    f"stable:g1:{source_identity}:{first.owner_entity_id}"
                ),
                t0_owner_entity_id=first.owner_entity_id,
                t1_owner_entity_id=second.owner_entity_id,
                t0_source_surface_id=first.source_surface_id,
                t1_source_surface_id=second.source_surface_id,
                t0_source_vertex_indices=first.source_vertex_indices,
                t1_source_vertex_indices=second.source_vertex_indices,
                confidence=float(np.clip(selected_score, 0.0, 1.0)),
                inference_state=(
                    RelationInferenceState.UNRESOLVED
                    if not accepted or motion is None
                    else motion.state
                ),
                accepted=accepted,
                motion_verified=bool(
                    accepted and motion is not None and motion.motion_verified
                ),
                transform_world_from_t0=(
                    motion.transform_world_from_t0
                    if accepted and motion is not None
                    else None
                ),
                t0_entity_id=first.entity_id,
                t1_entity_id=second.entity_id,
                assignment_is_null=not accepted,
                assignment_margin=margin,
                centered_shape_score=selected.centered_shape_score,
                size_score=selected.size_score,
                extent_score=selected.extent_score,
                registration_score=(
                    None if motion is None else motion.registration_score
                ),
                visible_support_score=visible_score,
                appearance_cosine=selected.appearance_cosine,
                appearance_feature_space=(
                    None
                    if selected.appearance_cosine is None
                    else "ovi_semantic_embedding"
                ),
                semantic_compatibility=selected.semantic_compatibility,
                original_location_iou=selected.original_location_iou,
                spatial_support_patch_count=(
                    None if motion is None else motion.spatial_support_patch_count
                ),
                residual_improvement_m=(
                    None if motion is None else motion.residual_improvement_m
                ),
                motion_rejection_reasons=(
                    () if motion is None else motion.rejection_reasons
                ),
                rejection_reasons=tuple(sorted(reasons)),
            )
        )
    return _enforce_one_to_one(tuple(output))


def _enforce_one_to_one(
    relations: tuple[RelationSupport, ...],
) -> tuple[RelationSupport, ...]:
    resolved = list(relations)
    used_t0: set[int] = set()
    used_t1: set[int] = set()
    for index, relation in sorted(
        enumerate(relations),
        key=lambda item: (
            -item[1].confidence,
            -(item[1].competition_margin or 0.0),
            item[1].relation_id,
        ),
    ):
        if not relation.accepted:
            continue
        if (
            relation.t0_owner_entity_id in used_t0
            or relation.t1_owner_entity_id in used_t1
        ):
            resolved[index] = replace(
                relation,
                accepted=False,
                inference_state=RelationInferenceState.UNRESOLVED,
                motion_verified=False,
                transform_world_from_t0=None,
                assignment_is_null=True,
                rejection_reasons=("one_to_one_conflict",),
            )
            continue
        used_t0.add(relation.t0_owner_entity_id)
        used_t1.add(relation.t1_owner_entity_id)
    return tuple(resolved)


def _coverage(record: QueryEntityEvidence) -> float:
    return max(record.entity_token_coverage, record.source_point_coverage)


def _query_source_rows(
    pair: NeuralSampleMap,
    query_mask: np.ndarray,
    *,
    visit_id: int,
    entity_id: str,
    local_rows: np.ndarray,
) -> np.ndarray:
    selected_tokens = [
        index
        for index, token_entity_id in enumerate(pair.token_entity_ids)
        if query_mask[index]
        and int(pair.visit_ids[index]) == visit_id
        and token_entity_id == entity_id
    ]
    contributor_positions = np.concatenate(
        [
            np.arange(
                pair.source_to_token_offsets[index],
                pair.source_to_token_offsets[index + 1],
                dtype=np.int64,
            )
            for index in selected_tokens
        ]
    )
    return np.sort(local_rows[contributor_positions])


def _query_rank(
    records: tuple[QueryEntityEvidence, ...],
) -> tuple[QueryEntityEvidence, float]:
    ordered = sorted(
        records,
        key=lambda item: (-_coverage(item), -item.soft_mass, item.entity_id),
    )
    best = ordered[0]
    runner_up = _coverage(ordered[1]) if len(ordered) > 1 else 0.0
    return best, _coverage(best) - runner_up


def relation_support_from_queries(
    pair: NeuralSampleMap,
    evidence: TemporalQueryEvidence,
    *,
    relation_source: str,
    projection_config: ProjectionConfig,
    minimum_query_score: float,
    minimum_competition_margin: float,
) -> tuple[RelationSupport, ...]:
    """Project query relations without the legacy below-threshold fallback."""

    if not isinstance(pair, NeuralSampleMap):
        raise TypeError("pair must be NeuralSampleMap")
    if not isinstance(evidence, TemporalQueryEvidence):
        raise TypeError("evidence must be TemporalQueryEvidence")
    if not isinstance(projection_config, ProjectionConfig):
        raise TypeError("projection_config must be ProjectionConfig")
    if not isinstance(relation_source, str) or not relation_source.strip():
        raise ValueError("relation_source must be non-empty")
    for value, name in (
        (minimum_query_score, "minimum_query_score"),
        (minimum_competition_margin, "minimum_competition_margin"),
    ):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    matrix = project_query_evidence(pair, evidence)
    assert evidence.query_masks is not None
    assert evidence.query_scores is not None
    local_rows = _source_local_rows(pair)
    pair_identity = pair.content_sha256()[:12]
    output: list[RelationSupport] = []
    for query_index, query_id in enumerate(evidence.temporal_query_ids):
        intersected = tuple(
            record
            for (record_query_id, _entity_id, _visit_id), record in matrix.items()
            if record_query_id == query_id and record.token_intersection_count > 0
        )
        t0_records = tuple(record for record in intersected if record.visit_id == 0)
        t1_records = tuple(record for record in intersected if record.visit_id == 1)
        if not t0_records or not t1_records:
            continue
        t0_record, t0_margin = _query_rank(t0_records)
        t1_record, t1_margin = _query_rank(t1_records)
        t0_qualified = (
            t0_record.entity_token_coverage
            >= projection_config.minimum_entity_token_coverage
            or t0_record.source_point_coverage
            >= projection_config.minimum_source_point_coverage
        )
        t1_qualified = (
            t1_record.entity_token_coverage
            >= projection_config.minimum_entity_token_coverage
            or t1_record.source_point_coverage
            >= projection_config.minimum_source_point_coverage
        )
        reasons: set[str] = set()
        if not t0_qualified:
            reasons.add("insufficient_t0_coverage")
        if not t1_qualified:
            reasons.add("insufficient_t1_coverage")
        competition_margin = min(t0_margin, t1_margin)
        if competition_margin < minimum_competition_margin:
            reasons.add("ambiguous_query_competition")
        query_score = float(evidence.query_scores[query_index])
        if query_score < minimum_query_score:
            reasons.add("below_query_score")
        accepted = not reasons
        t0_rows = _query_source_rows(
            pair,
            evidence.query_masks[query_index],
            visit_id=0,
            entity_id=t0_record.entity_id,
            local_rows=local_rows,
        )
        t1_rows = _query_source_rows(
            pair,
            evidence.query_masks[query_index],
            visit_id=1,
            entity_id=t1_record.entity_id,
            local_rows=local_rows,
        )
        t0_owner = _owner_id(t0_record.entity_id)
        t1_owner = _owner_id(t1_record.entity_id)
        output.append(
            RelationSupport(
                relation_id=(
                    f"{relation_source}:{pair_identity}:{query_id}:"
                    f"{t0_owner}:{t1_owner}"
                ),
                relation_source=relation_source,
                stable_entity_id=(
                    f"stable:{relation_source}:{pair_identity}:{query_id}"
                ),
                t0_owner_entity_id=t0_owner,
                t1_owner_entity_id=t1_owner,
                t0_source_surface_id=f"ovi-map:{pair.source_visit_map_sha256[0]}",
                t1_source_surface_id=f"ovi-map:{pair.source_visit_map_sha256[1]}",
                t0_source_vertex_indices=t0_rows,
                t1_source_vertex_indices=t1_rows,
                confidence=min(
                    query_score,
                    _coverage(t0_record),
                    _coverage(t1_record),
                ),
                inference_state=RelationInferenceState.UNRESOLVED,
                accepted=accepted,
                t0_entity_id=t0_record.entity_id,
                t1_entity_id=t1_record.entity_id,
                assignment_is_null=not accepted,
                assignment_margin=competition_margin,
                t0_mask_coverage=_coverage(t0_record),
                t1_mask_coverage=_coverage(t1_record),
                t0_mask_purity=t0_record.query_mask_fraction,
                t1_mask_purity=t1_record.query_mask_fraction,
                competition_margin=competition_margin,
                rejection_reasons=tuple(sorted(reasons)),
            )
        )
    return _enforce_one_to_one(tuple(output))


__all__ = [
    "RelationMotionVerification",
    "StrongGeometricRelationConfig",
    "build_g1_relation_support",
    "relation_support_from_queries",
    "verify_relation_motion",
]
