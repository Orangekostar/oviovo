"""Execution primitives for the deterministic two-visit baseline ladder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from src.core.data_structures import Frame
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.geometric_pair_reasoner import (
    GeometricPairReasoner,
    GeometricReasonerConfig,
)
from src.oviv2.query_instance_projection import (
    ProjectionConfig,
    project_queries_to_instances,
)
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    OviEntitySemanticEvidence,
    PairRelation,
    VisitMap,
    snapshot_content_sha256,
    validate_visit_pair,
)
from src.oviv2.two_visit_current_map import (
    CompositionConfig,
    SignedVisibilityGrid,
    TwoVisitCurrentMap,
    compose_current_map,
)
from src.oviv2.visibility import (
    VisibilityConfig,
    VisibilityStatus,
    VoxelVisibilityProjector,
)


@dataclass(frozen=True, slots=True)
class SignedVisibilityConfig:
    """Frozen conservative evidence thresholds for revisit free-space authority."""

    voxel_size_m: float = 0.05
    depth_tolerance_m: float = 0.10
    depth_max_m: float = 10.0
    minimum_absent_fraction: float = 0.80
    minimum_absent_observations: int = 6
    minimum_distinct_viewpoints: int = 3
    minimum_viewpoint_baseline_m: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "voxel_size_m",
            "depth_tolerance_m",
            "depth_max_m",
            "minimum_viewpoint_baseline_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, normalized)
        fraction = self.minimum_absent_fraction
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) <= 1.0
        ):
            raise ValueError("minimum_absent_fraction must be in (0, 1]")
        object.__setattr__(self, "minimum_absent_fraction", float(fraction))
        for name in ("minimum_absent_observations", "minimum_distinct_viewpoints"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_absent_observations < self.minimum_distinct_viewpoints:
            raise ValueError(
                "minimum_absent_observations cannot be below distinct viewpoints"
            )


def _visit_voxel_keys(visit: VisitMap, voxel_size_m: float) -> np.ndarray:
    chunks = [np.asarray(entity.points_xyz, dtype=np.float64) for entity in visit.snapshot.entities]
    if visit.snapshot.background_xyz is not None:
        chunks.append(np.asarray(visit.snapshot.background_xyz, dtype=np.float64))
    nonempty = [chunk for chunk in chunks if len(chunk)]
    if not nonempty:
        raise ValueError("t0 visit contains no surface points")
    points = np.concatenate(nonempty, axis=0)
    scaled = np.floor(points / voxel_size_m)
    limit = np.iinfo(np.int64).max
    if not np.all(np.isfinite(scaled)) or np.any(np.abs(scaled) > limit):
        raise ValueError("t0 surface points exceed visibility voxel range")
    return np.unique(scaled.astype(np.int64), axis=0)


def _clean_frame(frame: Frame, depth_max_m: float) -> Frame:
    if not isinstance(frame, Frame):
        raise TypeError("frames must contain Frame values")
    depth = np.asarray(frame.depth, dtype=np.float64)
    clean = np.array(depth, copy=True)
    valid = np.isfinite(clean) & (clean > 0.0) & (clean <= depth_max_m)
    clean[~valid] = 0.0
    return Frame(
        frame_id=frame.frame_id,
        source_frame_id=frame.source_frame_id,
        rgb=frame.rgb,
        depth=clean,
        pose=frame.pose,
        intrinsics=frame.intrinsics,
        timestamp=frame.timestamp,
    )


def derive_signed_visibility(
    t0: VisitMap,
    frames: tuple[Frame, ...],
    config: SignedVisibilityConfig,
    *,
    source_sha256: str,
) -> SignedVisibilityGrid:
    """Aggregate t1 depth evidence without using evaluator visibility targets."""

    if not isinstance(t0, VisitMap) or t0.visit_id != 0:
        raise ValueError("t0 must be visit zero")
    if not isinstance(frames, tuple) or not frames:
        raise ValueError("frames must be a non-empty tuple")
    if not isinstance(config, SignedVisibilityConfig):
        raise TypeError("config must be SignedVisibilityConfig")
    frame_ids = [frame.frame_id for frame in frames if isinstance(frame, Frame)]
    if len(frame_ids) != len(frames) or any(
        current <= previous for previous, current in pairwise(frame_ids)
    ):
        raise ValueError("visibility frame IDs must be strictly increasing")
    t0.assert_unchanged()
    keys = _visit_voxel_keys(t0, config.voxel_size_m)
    key_tuples = tuple(tuple(int(item) for item in key) for key in keys)
    key_indices = {key: index for index, key in enumerate(key_tuples)}
    present = np.zeros(len(keys), dtype=np.uint32)
    absent = np.zeros(len(keys), dtype=np.uint32)
    occluded = np.zeros(len(keys), dtype=np.uint32)
    viewpoint_count = np.zeros(len(keys), dtype=np.uint8)
    viewpoints = np.zeros(
        (len(keys), config.minimum_distinct_viewpoints, 3), dtype=np.float64
    )
    projector = VoxelVisibilityProjector(
        VisibilityConfig(
            voxel_size_m=config.voxel_size_m,
            depth_tolerance_m=config.depth_tolerance_m,
        )
    )
    counters = {
        VisibilityStatus.PRESENT: present,
        VisibilityStatus.ABSENT: absent,
        VisibilityStatus.OCCLUDED: occluded,
    }
    for raw_frame in frames:
        frame = _clean_frame(raw_frame, config.depth_max_m)
        groups = projector.classify_many(key_tuples, frame)
        for status, counter in counters.items():
            status_keys = groups[status]
            if not status_keys:
                continue
            indices = np.fromiter(
                (key_indices[key] for key in status_keys),
                dtype=np.int64,
                count=len(status_keys),
            )
            counter[indices] += 1
            if status is not VisibilityStatus.ABSENT:
                continue
            camera = np.asarray(frame.pose, dtype=np.float64)[:3, 3]
            counts = viewpoint_count[indices]
            distinct = np.ones(len(indices), dtype=np.bool_)
            for slot in range(config.minimum_distinct_viewpoints):
                occupied = counts > slot
                if np.any(occupied):
                    distances = np.linalg.norm(
                        viewpoints[indices[occupied], slot] - camera,
                        axis=1,
                    )
                    distinct[occupied] &= (
                        distances >= config.minimum_viewpoint_baseline_m
                    )
            add = distinct & (counts < config.minimum_distinct_viewpoints)
            if np.any(add):
                selected = indices[add]
                slots = counts[add].astype(np.int64)
                viewpoints[selected, slots] = camera
                viewpoint_count[selected] += 1

    tested = present.astype(np.uint64) + absent + occluded
    absent_fraction = np.divide(
        absent,
        tested,
        out=np.zeros(len(keys), dtype=np.float64),
        where=tested > 0,
    )
    is_free = (
        (present == 0)
        & (absent >= config.minimum_absent_observations)
        & (viewpoint_count >= config.minimum_distinct_viewpoints)
        & (absent_fraction >= config.minimum_absent_fraction)
    )
    statuses = np.full(len(keys), "unobserved", dtype=object)
    statuses[occluded > 0] = "occluded"
    statuses[is_free] = "visible_free"
    statuses[present > 0] = "occupied"
    t0.assert_unchanged()
    return SignedVisibilityGrid(
        voxel_size_m=config.voxel_size_m,
        voxel_keys=keys,
        statuses=tuple(str(value) for value in statuses),
        source_sha256=source_sha256,
    )


def _copy_entity(
    entity: EntityPrediction,
    *,
    entity_id: str,
    geometry_authority: str,
) -> EntityPrediction:
    metadata = dict(entity.metadata)
    metadata["geometry_authority"] = geometry_authority
    metadata["semantic_authority"] = geometry_authority
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(entity.points_xyz, dtype=np.float32),
        semantic_embedding=entity.semantic_embedding,
        semantic_label=entity.semantic_label,
        semantic_score=entity.semantic_score,
        lifecycle_state="current",
        first_seen=entity.first_seen,
        last_seen=entity.last_seen,
        metadata=metadata,
    )


def build_static_baseline_snapshot(
    variant_id: str,
    t0: VisitMap,
    t1: VisitMap,
) -> MapSnapshot:
    """Build the exact B0, B1, or B2 map without temporal inference."""

    validate_visit_pair(t0, t1)
    before = (t0.snapshot_sha256, t1.snapshot_sha256)
    if variant_id not in {"B0", "B1", "B2"}:
        raise ValueError("static baseline variant must be B0, B1, or B2")
    selected = {
        "B0": (("t0", t0),),
        "B1": (("t0", t0), ("t1", t1)),
        "B2": (("t1", t1),),
    }[variant_id]
    entities: list[EntityPrediction] = []
    backgrounds: list[np.ndarray] = []
    for visit_name, visit in selected:
        for entity in sorted(visit.snapshot.entities, key=lambda item: item.entity_id):
            output_id = (
                f"{visit_name}:{entity.entity_id}"
                if variant_id == "B1"
                else entity.entity_id
            )
            entities.append(
                _copy_entity(
                    entity,
                    entity_id=output_id,
                    geometry_authority=f"ovi_{visit_name}",
                )
            )
        if visit.snapshot.background_xyz is not None:
            backgrounds.append(np.asarray(visit.snapshot.background_xyz, dtype=np.float32))
    background = np.concatenate(backgrounds, axis=0) if backgrounds else None
    snapshot = MapSnapshot(
        method={
            "B0": "OVI_T0_ONLY",
            "B1": "OVI_UNION",
            "B2": "OVI_T1_ONLY",
        }[variant_id],
        scene_id=t1.snapshot.scene_id,
        timestamp=t1.snapshot.timestamp,
        entities=entities,
        background_xyz=background,
        scope="current",
    )
    after = (snapshot_content_sha256(t0.snapshot), snapshot_content_sha256(t1.snapshot))
    if after != before:
        raise ValueError("OVI source maps changed while building static baseline")
    return snapshot


def build_geometric_pair_sample(
    t0: VisitMap,
    t1: VisitMap,
    *,
    neural_voxel_size_m: float,
) -> NeuralSampleMap:
    """Build geometry-only tokens for B4; these are never ReScene inputs."""

    validate_visit_pair(t0, t1)
    if not math.isfinite(float(neural_voxel_size_m)) or neural_voxel_size_m <= 0.0:
        raise ValueError("neural_voxel_size_m must be finite and positive")
    before = (t0.snapshot_sha256, t1.snapshot_sha256)
    coordinates: list[np.ndarray] = []
    features: list[np.ndarray] = []
    token_visits: list[np.ndarray] = []
    contributor_visits: list[np.ndarray] = []
    contributor_entities: list[str] = []
    contributor_points: list[np.ndarray] = []
    contributor_counts: list[np.ndarray] = []
    semantics: list[OviEntitySemanticEvidence] = []
    global_point_offset = 0
    for visit in (t0, t1):
        for entity in sorted(visit.snapshot.entities, key=lambda item: item.entity_id):
            points = np.asarray(entity.points_xyz, dtype=np.float64)
            if not len(points):
                continue
            quantized = np.floor(points / neural_voxel_size_m).astype(np.int64)
            _unique, inverse = np.unique(quantized, axis=0, return_inverse=True)
            counts = np.bincount(inverse, minlength=len(_unique)).astype(np.int64)
            sums = np.zeros((len(_unique), 3), dtype=np.float64)
            np.add.at(sums, inverse, points)
            coordinates.append(
                np.column_stack(
                    (
                        sums / counts[:, None],
                        np.full(len(_unique), visit.visit_id, dtype=np.float64),
                    )
                )
            )
            features.append(np.ones((len(_unique), 1), dtype=np.float32))
            token_visits.append(np.full(len(_unique), visit.visit_id, dtype=np.int8))
            order = np.argsort(inverse, kind="stable")
            contributor_visits.append(
                np.full(len(points), visit.visit_id, dtype=np.int8)
            )
            contributor_entities.extend([entity.entity_id] * len(points))
            contributor_points.append(global_point_offset + order.astype(np.int64))
            contributor_counts.append(counts)
            global_point_offset += len(points)
            semantics.append(
                OviEntitySemanticEvidence(
                    visit_id=visit.visit_id,
                    entity_id=entity.entity_id,
                    semantic_label=entity.semantic_label,
                    semantic_score=entity.semantic_score,
                    semantic_embedding=entity.semantic_embedding,
                )
            )
    if not coordinates:
        raise ValueError("OVI visit pair contains no entity surface points")
    counts = np.concatenate(contributor_counts)
    offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    pair = NeuralSampleMap(
        coordinates_xyzt=np.concatenate(coordinates, axis=0),
        features=np.concatenate(features, axis=0),
        visit_ids=np.concatenate(token_visits),
        source_visit_ids=np.concatenate(contributor_visits),
        source_entity_ids=tuple(contributor_entities),
        source_point_indices=np.concatenate(contributor_points),
        source_to_token_offsets=offsets,
        neural_voxel_size_m=float(neural_voxel_size_m),
        feature_schema="geometric_only",
        coordinate_frame_id=t0.coordinate_frame_id,
        source_manifest_sha256=t0.source_manifest_sha256,
        source_visit_map_sha256=before,
        entity_semantics=tuple(semantics),
    )
    after = (snapshot_content_sha256(t0.snapshot), snapshot_content_sha256(t1.snapshot))
    if after != before:
        raise ValueError("OVI source maps changed during geometric sampling")
    return pair


def build_visibility_baseline(
    t0: VisitMap,
    t1: VisitMap,
    visibility: SignedVisibilityGrid,
    *,
    use_geometric_pairing: bool,
    geometric_config: GeometricReasonerConfig,
) -> tuple[TwoVisitCurrentMap, tuple[PairRelation, ...]]:
    """Build B3 or B4 with identical signed visibility authority."""

    if type(use_geometric_pairing) is not bool:
        raise TypeError("use_geometric_pairing must be boolean")
    relations: tuple[PairRelation, ...] = ()
    if use_geometric_pairing:
        pair = build_geometric_pair_sample(
            t0,
            t1,
            neural_voxel_size_m=0.02,
        )
        evidence = GeometricPairReasoner(geometric_config).infer(pair)
        relations = project_queries_to_instances(
            pair,
            evidence,
            ProjectionConfig(),
        ).relations
    method = (
        "OVI_GEOMETRIC_PAIRING_VISIBILITY"
        if use_geometric_pairing
        else "OVI_VISIBILITY_COMPOSE"
    )
    current = compose_current_map(
        t0,
        t1,
        relations,
        visibility,
        CompositionConfig(
            composition_voxel_size_m=visibility.voxel_size_m,
            method_name=method,
        ),
    )
    return current, relations


__all__ = [
    "SignedVisibilityConfig",
    "build_geometric_pair_sample",
    "build_static_baseline_snapshot",
    "build_visibility_baseline",
    "derive_signed_visibility",
]
