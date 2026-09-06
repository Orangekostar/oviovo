"""Deterministic query-guided grouping of immutable dense OVI entities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.ovi_pair_views import OviObjectPairView, OviObjectVisitView
from src.evaluation.rscan_gt_instances import (
    GroundTruthPair,
    PredictedInstance,
    ThresholdGeometryResult,
    evaluate_instance_geometry,
    voxelize_points,
)
from src.oviv2.two_visit_contracts import PairRelation

GroupingVariant = Literal["A_ID", "U0", "U1", "U2", "U3"]
ConflictReason = Literal["abstained_overlap", "lower_rank"]
_VARIANTS = frozenset({"A_ID", "U0", "U1", "U2", "U3"})


class TemporalObjectGroupingError(ValueError):
    """Raised when a composite-object grouping violates dense ownership."""


def _readonly_indices(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise TemporalObjectGroupingError(f"{name} must be an integer vector")
    result = np.array(raw, dtype=np.int64, copy=True, order="C")
    if len(result) == 0 or np.any(result < 0) or len(np.unique(result)) != len(result):
        raise TemporalObjectGroupingError(f"{name} must contain unique nonnegative rows")
    result.setflags(write=False)
    return result


def _sha256_payload(value: object) -> str:
    def normalize(item: object) -> object:
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            return {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        if isinstance(item, Mapping):
            return {
                str(key): normalize(nested)
                for key, nested in sorted(item.items(), key=lambda row: str(row[0]))
            }
        if isinstance(item, (tuple, list)):
            return [normalize(nested) for nested in item]
        return item

    return hashlib.sha256(
        json.dumps(
            normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _xyz_multiset_sha256(values: Sequence[np.ndarray]) -> str:
    points = np.concatenate(
        [np.asarray(value, dtype=np.float32) for value in values], axis=0
    )
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
        raise TemporalObjectGroupingError("XYZ multiset must contain finite 3D points")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return _sha256_payload(np.ascontiguousarray(points[order], dtype=np.float32))


@dataclass(frozen=True, slots=True)
class GroupingConfig:
    minimum_query_confidence: float = 0.3

    def __post_init__(self) -> None:
        value = self.minimum_query_confidence
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise TemporalObjectGroupingError(
                "minimum query confidence must be in [0, 1]"
            )
        object.__setattr__(self, "minimum_query_confidence", float(value))


@dataclass(frozen=True, slots=True)
class CompositeObject:
    visit_id: int
    object_id: str
    member_entity_ids: tuple[str, ...]
    source_point_indices: np.ndarray
    temporal_identity_id: str | None
    query_ids: tuple[str, ...]
    query_confidence: float | None
    query_support: float

    def __post_init__(self) -> None:
        if type(self.visit_id) is not int or self.visit_id not in (0, 1):
            raise TemporalObjectGroupingError("composite visit ID must be 0 or 1")
        if not isinstance(self.object_id, str) or not self.object_id:
            raise TemporalObjectGroupingError("composite object ID is empty")
        members = tuple(self.member_entity_ids)
        queries = tuple(str(value) for value in self.query_ids)
        if (
            not members
            or any(not isinstance(value, str) or not value for value in members)
            or len(members) != len(set(members))
            or len(queries) != len(set(queries))
        ):
            raise TemporalObjectGroupingError("composite membership is invalid")
        if self.temporal_identity_id is not None and (
            not isinstance(self.temporal_identity_id, str)
            or not self.temporal_identity_id
        ):
            raise TemporalObjectGroupingError("temporal identity ID is invalid")
        if queries:
            confidence = self.query_confidence
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
                or self.temporal_identity_id is None
            ):
                raise TemporalObjectGroupingError("query provenance is incomplete")
            normalized_confidence: float | None = float(confidence)
        elif self.query_confidence is not None or self.temporal_identity_id is not None:
            raise TemporalObjectGroupingError("atomic provenance must be consistently null")
        else:
            normalized_confidence = None
        support = float(self.query_support)
        if not math.isfinite(support) or support < 0.0:
            raise TemporalObjectGroupingError("query support must be nonnegative")
        indices = _readonly_indices(
            self.source_point_indices, name="composite source point indices"
        )
        object.__setattr__(self, "member_entity_ids", members)
        object.__setattr__(self, "source_point_indices", indices)
        object.__setattr__(self, "query_ids", queries)
        object.__setattr__(self, "query_confidence", normalized_confidence)
        object.__setattr__(self, "query_support", support)

    @property
    def is_grouped(self) -> bool:
        return len(self.member_entity_ids) > 1


@dataclass(frozen=True, slots=True)
class RejectedGroupingConflict:
    variant_id: GroupingVariant
    query_id: str
    visit_id: int
    entity_id: str
    winner_query_id: str | None
    reason: ConflictReason


@dataclass(frozen=True, slots=True)
class TemporalObjectGrouping:
    variant_id: GroupingVariant
    pair_content_sha256: str
    candidate_ids: tuple[tuple[str, ...], tuple[str, ...]]
    objects: tuple[tuple[CompositeObject, ...], tuple[CompositeObject, ...]]
    owner_object_indices: tuple[np.ndarray, np.ndarray]
    snapshots: tuple[MapSnapshot, MapSnapshot]
    rejected_conflicts: tuple[RejectedGroupingConflict, ...]
    source_xyz_multiset_sha256: str
    output_xyz_multiset_sha256: str
    owned_source_point_count: int
    appearance_supported_owned_point_count: int
    _content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.variant_id not in _VARIANTS:
            raise TemporalObjectGroupingError("grouping variant is invalid")
        if len(self.pair_content_sha256) != 64:
            raise TemporalObjectGroupingError("pair content SHA-256 is invalid")
        if self.source_xyz_multiset_sha256 != self.output_xyz_multiset_sha256:
            raise TemporalObjectGroupingError("grouping changed the XYZ multiset")
        candidates = tuple(tuple(value) for value in self.candidate_ids)
        objects = tuple(tuple(value) for value in self.objects)
        if len(candidates) != 2 or len(objects) != 2:
            raise TemporalObjectGroupingError("grouping must contain two visits")
        owners: list[np.ndarray] = []
        for visit_id in (0, 1):
            raw = np.asarray(self.owner_object_indices[visit_id])
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                raise TemporalObjectGroupingError("owner indices are invalid")
            owner = np.array(raw, dtype=np.int64, copy=True, order="C")
            if np.any(owner < -1) or np.any(owner >= len(objects[visit_id])):
                raise TemporalObjectGroupingError("owner index is outside object range")
            observed_members = [
                entity_id
                for value in objects[visit_id]
                for entity_id in value.member_entity_ids
            ]
            if sorted(observed_members) != sorted(candidates[visit_id]):
                raise TemporalObjectGroupingError(
                    "grouping does not conserve source candidates exactly once"
                )
            owner.setflags(write=False)
            owners.append(owner)
        rejected = tuple(
            sorted(
                self.rejected_conflicts,
                key=lambda value: (
                    value.visit_id,
                    value.entity_id,
                    value.query_id,
                    value.winner_query_id or "",
                ),
            )
        )
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "owner_object_indices", (owners[0], owners[1]))
        object.__setattr__(self, "rejected_conflicts", rejected)
        object.__setattr__(
            self,
            "_content_sha256",
            _sha256_payload(
                {
                    "variant_id": self.variant_id,
                    "pair_content_sha256": self.pair_content_sha256,
                    "candidate_ids": self.candidate_ids,
                    "objects": [
                        [
                            {
                                "visit_id": value.visit_id,
                                "object_id": value.object_id,
                                "member_entity_ids": value.member_entity_ids,
                                "source_point_indices": value.source_point_indices,
                                "temporal_identity_id": value.temporal_identity_id,
                                "query_ids": value.query_ids,
                                "query_confidence": value.query_confidence,
                                "query_support": value.query_support,
                            }
                            for value in visit
                        ]
                        for visit in self.objects
                    ],
                    "rejected_conflicts": [
                        {
                            "variant_id": value.variant_id,
                            "query_id": value.query_id,
                            "visit_id": value.visit_id,
                            "entity_id": value.entity_id,
                            "winner_query_id": value.winner_query_id,
                            "reason": value.reason,
                        }
                        for value in rejected
                    ],
                    "source_xyz_multiset_sha256": self.source_xyz_multiset_sha256,
                    "output_xyz_multiset_sha256": self.output_xyz_multiset_sha256,
                }
            ),
        )

    def content_sha256(self) -> str:
        return self._content_sha256


@dataclass(frozen=True, slots=True)
class _Proposal:
    query_id: str
    visit_id: int
    member_entity_ids: tuple[str, ...]
    confidence: float
    support: float

    @property
    def rank(self) -> tuple[float, float, str]:
        return (-self.confidence, -self.support, self.query_id)


def _proposals(
    pair: OviObjectPairView,
    relations: Sequence[PairRelation],
    *,
    variant_id: GroupingVariant,
    config: GroupingConfig,
) -> tuple[_Proposal, ...]:
    candidate_sets = tuple(
        {entity.entity_id for entity in visit.entities} for visit in pair.visits
    )
    proposals: list[_Proposal] = []
    for relation in relations:
        if not isinstance(relation, PairRelation):
            raise TemporalObjectGroupingError("grouping relation is invalid")
        members_by_visit = (relation.t0_entity_ids, relation.t1_entity_ids)
        for visit_id, members in enumerate(members_by_visit):
            unknown = set(members).difference(candidate_sets[visit_id])
            if unknown:
                raise TemporalObjectGroupingError(
                    f"relation references unknown entity: {sorted(unknown)}"
                )
        if variant_id in {"A_ID", "U2", "U3"} and (
            relation.query_confidence < config.minimum_query_confidence
        ):
            continue
        support = max(float(relation.evidence.get("soft_mass", 0.0)), 0.0)
        for visit_id, members in enumerate(members_by_visit):
            if members:
                proposals.append(
                    _Proposal(
                        query_id=str(relation.temporal_query_id),
                        visit_id=visit_id,
                        member_entity_ids=tuple(sorted(members)),
                        confidence=relation.query_confidence,
                        support=support,
                    )
                )
    return tuple(
        sorted(
            proposals,
            key=lambda value: (value.visit_id, value.rank, value.member_entity_ids),
        )
    )


def _resolve_proposals(
    proposals: Sequence[_Proposal], *, variant_id: GroupingVariant
) -> tuple[dict[tuple[int, str], _Proposal], tuple[RejectedGroupingConflict, ...]]:
    by_entity: dict[tuple[int, str], list[_Proposal]] = {}
    for proposal in proposals:
        for entity_id in proposal.member_entity_ids:
            by_entity.setdefault((proposal.visit_id, entity_id), []).append(proposal)
    assignments: dict[tuple[int, str], _Proposal] = {}
    rejected: list[RejectedGroupingConflict] = []
    if variant_id in {"U1", "U2"}:
        conflicted = {
            key for key, values in by_entity.items() if len(values) > 1
        }
        rejected_queries = {
            proposal.query_id
            for proposal in proposals
            if any(
                (proposal.visit_id, entity_id) in conflicted
                for entity_id in proposal.member_entity_ids
            )
        }
        for proposal in proposals:
            if proposal.query_id in rejected_queries:
                for entity_id in proposal.member_entity_ids:
                    if (proposal.visit_id, entity_id) in conflicted:
                        rejected.append(
                            RejectedGroupingConflict(
                                variant_id=variant_id,
                                query_id=proposal.query_id,
                                visit_id=proposal.visit_id,
                                entity_id=entity_id,
                                winner_query_id=None,
                                reason="abstained_overlap",
                            )
                        )
                continue
            for entity_id in proposal.member_entity_ids:
                assignments[(proposal.visit_id, entity_id)] = proposal
        return assignments, tuple(rejected)
    for key, values in sorted(by_entity.items()):
        winner = min(values, key=lambda value: value.rank)
        assignments[key] = winner
        for loser in values:
            if loser is winner:
                continue
            rejected.append(
                RejectedGroupingConflict(
                    variant_id=variant_id,
                    query_id=loser.query_id,
                    visit_id=key[0],
                    entity_id=key[1],
                    winner_query_id=winner.query_id,
                    reason="lower_rank",
                )
            )
    return assignments, tuple(rejected)


def _composite_object(
    visit: OviObjectVisitView,
    *,
    variant_id: GroupingVariant,
    member_entity_ids: tuple[str, ...],
    proposal: _Proposal | None,
) -> CompositeObject:
    entities = {value.entity_id: value for value in visit.entities}
    indices = np.sort(
        np.concatenate([entities[entity_id].point_indices for entity_id in member_entity_ids])
    )
    grouped = len(member_entity_ids) > 1 and proposal is not None
    object_id = (
        f"{variant_id}:query:{proposal.query_id}:t{visit.visit_id}"
        if grouped
        else f"{variant_id}:atomic:t{visit.visit_id}:{member_entity_ids[0]}"
    )
    return CompositeObject(
        visit_id=visit.visit_id,
        object_id=object_id,
        member_entity_ids=member_entity_ids,
        source_point_indices=indices,
        temporal_identity_id=(
            None if proposal is None else f"{variant_id}:query:{proposal.query_id}"
        ),
        query_ids=() if proposal is None else (proposal.query_id,),
        query_confidence=None if proposal is None else proposal.confidence,
        query_support=0.0 if proposal is None else proposal.support,
    )


def _visit_objects(
    visit: OviObjectVisitView,
    *,
    variant_id: GroupingVariant,
    assignments: Mapping[tuple[int, str], _Proposal],
) -> tuple[CompositeObject, ...]:
    by_query: dict[str, list[str]] = {}
    proposal_by_query: dict[str, _Proposal] = {}
    for entity in visit.entities:
        proposal = assignments.get((visit.visit_id, entity.entity_id))
        if proposal is not None:
            by_query.setdefault(proposal.query_id, []).append(entity.entity_id)
            proposal_by_query[proposal.query_id] = proposal
    grouped_members = {
        entity_id
        for members in by_query.values()
        if len(members) > 1 and variant_id != "A_ID"
        for entity_id in members
    }
    objects = [
        _composite_object(
            visit,
            variant_id=variant_id,
            member_entity_ids=tuple(sorted(members)),
            proposal=proposal_by_query[query_id],
        )
        for query_id, members in sorted(by_query.items())
        if len(members) > 1 and variant_id != "A_ID"
    ]
    for entity in visit.entities:
        if entity.entity_id in grouped_members:
            continue
        proposal = assignments.get((visit.visit_id, entity.entity_id))
        objects.append(
            _composite_object(
                visit,
                variant_id=variant_id,
                member_entity_ids=(entity.entity_id,),
                proposal=proposal,
            )
        )
    return tuple(
        sorted(
            objects,
            key=lambda value: (int(value.source_point_indices[0]), value.object_id),
        )
    )


def _snapshot(
    pair: OviObjectPairView,
    visit: OviObjectVisitView,
    objects: tuple[CompositeObject, ...],
) -> tuple[MapSnapshot, np.ndarray]:
    source_entities = {value.entity_id: value for value in visit.entities}
    owners = np.full(visit.point_count, -1, dtype=np.int64)
    predictions: list[EntityPrediction] = []
    for object_index, value in enumerate(objects):
        if np.any(owners[value.source_point_indices] >= 0):
            raise TemporalObjectGroupingError("source point received multiple owners")
        owners[value.source_point_indices] = object_index
        representative = min(
            (source_entities[entity_id] for entity_id in value.member_entity_ids),
            key=lambda entity: (-entity.point_count, entity.entity_id),
        )
        predictions.append(
            EntityPrediction(
                entity_id=value.object_id,
                points_xyz=visit.points_xyz[value.source_point_indices],
                semantic_embedding=representative.semantic_embedding,
                semantic_label=representative.semantic_label,
                semantic_score=representative.semantic_score,
                lifecycle_state="current",
                first_seen=0.0,
                last_seen=float(visit.frame_count - 1),
                metadata={
                    "grouping_variant": value.object_id.split(":", 1)[0],
                    "source_visit_id": visit.visit_id,
                    "member_entity_ids": list(value.member_entity_ids),
                    "temporal_identity_id": value.temporal_identity_id,
                    "query_ids": list(value.query_ids),
                    "query_confidence": value.query_confidence,
                    "query_support": value.query_support,
                    "geometry_authority": f"ovi_t{visit.visit_id}",
                    "semantic_authority": f"ovi_t{visit.visit_id}",
                },
            )
        )
    expected_owned = visit.entity_owner_indices >= 0
    if not np.array_equal(owners >= 0, expected_owned):
        raise TemporalObjectGroupingError("dense object ownership lost source points")
    snapshot = MapSnapshot(
        method=f"OVI-MAP + temporal grouping {objects[0].object_id.split(':', 1)[0]}",
        scene_id=pair.pair_id,
        timestamp=float(visit.frame_count - 1),
        entities=predictions,
        background_xyz=visit.points_xyz[~expected_owned],
        scope="current",
        runtime={},
    )
    owners.setflags(write=False)
    return snapshot, owners


def build_temporal_object_groups(
    pair: OviObjectPairView,
    relations: Sequence[PairRelation],
    *,
    variant_id: GroupingVariant,
    config: GroupingConfig | None = None,
) -> TemporalObjectGrouping:
    """Build exclusive composites while preserving every dense OVI point."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be an OviObjectPairView")
    if variant_id not in _VARIANTS:
        raise TemporalObjectGroupingError("grouping variant is invalid")
    if config is None:
        config = GroupingConfig()
    if not isinstance(config, GroupingConfig):
        raise TypeError("config must be GroupingConfig")
    rows = tuple(relations)
    if variant_id == "U0" and rows:
        raise TemporalObjectGroupingError("U0 cannot consume temporal relations")
    before = pair.content_sha256()
    proposals = _proposals(pair, rows, variant_id=variant_id, config=config)
    assignments, rejected = _resolve_proposals(proposals, variant_id=variant_id)
    objects = tuple(
        _visit_objects(visit, variant_id=variant_id, assignments=assignments)
        for visit in pair.visits
    )
    snapshots_and_owners = tuple(
        _snapshot(pair, visit, objects[visit.visit_id]) for visit in pair.visits
    )
    snapshots = tuple(value[0] for value in snapshots_and_owners)
    owners = tuple(value[1] for value in snapshots_and_owners)
    source_hash = _xyz_multiset_sha256(
        tuple(visit.points_xyz for visit in pair.visits)
    )
    output_chunks: list[np.ndarray] = []
    for snapshot in snapshots:
        output_chunks.extend(value.points_xyz for value in snapshot.entities)
        if snapshot.background_xyz is not None:
            output_chunks.append(snapshot.background_xyz)
    output_hash = _xyz_multiset_sha256(output_chunks)
    if pair.content_sha256() != before:
        raise TemporalObjectGroupingError("grouping mutated the OVI pair")
    return TemporalObjectGrouping(
        variant_id=variant_id,
        pair_content_sha256=before,
        candidate_ids=tuple(
            tuple(entity.entity_id for entity in visit.entities) for visit in pair.visits
        ),
        objects=objects,
        owner_object_indices=owners,
        snapshots=snapshots,
        rejected_conflicts=rejected,
        source_xyz_multiset_sha256=source_hash,
        output_xyz_multiset_sha256=output_hash,
        owned_source_point_count=sum(
            int(np.count_nonzero(visit.entity_owner_indices >= 0))
            for visit in pair.visits
        ),
        appearance_supported_owned_point_count=sum(
            int(
                np.count_nonzero(
                    visit.appearance_valid & (visit.entity_owner_indices >= 0)
                )
            )
            for visit in pair.visits
        ),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _threshold_metrics(
    value: ThresholdGeometryResult, *, prediction_count: int, gt_count: int
) -> dict[str, object]:
    true_positive = value.matched_count
    precision = _ratio(true_positive, prediction_count)
    recall = _ratio(true_positive, gt_count)
    f_score = (
        None
        if precision is None or recall is None
        else 0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "status": "PASS",
        "iou_threshold": value.threshold,
        "predicted_instance_count": prediction_count,
        "ground_truth_instance_count": gt_count,
        "true_positive_count": true_positive,
        "false_positive_count": prediction_count - true_positive,
        "false_negative_count": gt_count - true_positive,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "fragment_gt_count": value.fragment_gt_count,
        "merge_prediction_count": value.merge_prediction_count,
        "merge_prediction_rate": _ratio(value.merge_prediction_count, prediction_count),
        "duplicate_prediction_count": value.duplicate_prediction_count,
        "mean_matched_iou": value.mean_matched_iou,
    }


def evaluate_current_instance_grouping(
    grouping: TemporalObjectGrouping,
    ground_truth: GroundTruthPair,
) -> Mapping[str, Mapping[str, object]]:
    """Evaluate the dense t1 composite masks as class-agnostic instances."""

    if not isinstance(grouping, TemporalObjectGrouping):
        raise TypeError("grouping must be TemporalObjectGrouping")
    if not isinstance(ground_truth, GroundTruthPair):
        raise TypeError("ground_truth must be GroundTruthPair")
    if ground_truth.pair_id != grouping.snapshots[1].scene_id:
        raise TemporalObjectGroupingError("grouping and GT pair IDs differ")
    predictions = tuple(
        PredictedInstance(
            prediction_id=entity.entity_id,
            voxels=voxelize_points(
                entity.points_xyz, voxel_size_m=ground_truth.voxel_size_m
            ),
        )
        for entity in grouping.snapshots[1].entities
    )
    result = evaluate_instance_geometry(
        predictions,
        ground_truth.visits[1],
        matching_policy="max_valid_count_then_iou",
    )
    shared = {
        "variant_id": grouping.variant_id,
        "pair_content_sha256": grouping.pair_content_sha256,
        "grouping_sha256": grouping.content_sha256(),
        "geometry_unchanged": (
            grouping.source_xyz_multiset_sha256
            == grouping.output_xyz_multiset_sha256
        ),
        "source_candidate_count": len(grouping.candidate_ids[1]),
        "composite_object_count": len(grouping.objects[1]),
        "grouped_source_entity_count": sum(
            len(value.member_entity_ids)
            for value in grouping.objects[1]
            if value.is_grouped
        ),
        "rejected_conflict_count": len(grouping.rejected_conflicts),
        "appearance_supported_owned_point_count": (
            grouping.appearance_supported_owned_point_count
        ),
        "owned_source_point_count": grouping.owned_source_point_count,
    }
    return MappingProxyType(
        {
            "0.50": MappingProxyType(
                shared
                | _threshold_metrics(
                    result.primary,
                    prediction_count=len(predictions),
                    gt_count=len(ground_truth.visits[1]),
                )
            ),
            "0.25": MappingProxyType(
                shared
                | _threshold_metrics(
                    result.sensitivity,
                    prediction_count=len(predictions),
                    gt_count=len(ground_truth.visits[1]),
                )
            ),
        }
    )


__all__ = [
    "CompositeObject",
    "GroupingConfig",
    "RejectedGroupingConflict",
    "TemporalObjectGrouping",
    "TemporalObjectGroupingError",
    "build_temporal_object_groups",
    "evaluate_current_instance_grouping",
]
