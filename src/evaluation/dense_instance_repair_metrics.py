"""Evaluator-facing P0/P1/P2 views and dense instance/identity metrics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.evaluation.object_pair_association import (
    ObjectPairPrediction,
    ObjectPairScoreMatrix,
)
from src.evaluation.ovi_pair_views import OviObjectPairView
from src.evaluation.rscan_association_metrics import CandidateEndpointBinding
from src.evaluation.rscan_gt_instances import (
    GroundTruthInstance,
    GroundTruthPair,
    PredictedInstance,
    ThresholdGeometryResult,
    evaluate_instance_geometry,
    voxelize_points,
)
from src.evaluation.temporal_object_groups import TemporalObjectGrouping
from src.oviv2.rescene_dense_instance_readout import (
    OWNER_BACKGROUND,
    OWNER_OVI_RESIDUAL,
    OWNER_QUERY,
    OWNER_UNKNOWN,
    DensePairReadout,
)

MethodId = Literal["P0", "P1", "P2"]
SupportDomain = Literal["full", "supported"]


class DenseInstanceRepairMetricError(ValueError):
    """Raised when a dense method view or metric input is inconsistent."""


class DenseEndpointBindingError(DenseInstanceRepairMetricError):
    """Raised when identity scoring does not use its frozen candidate binding."""


def _readonly(value: object, dtype: np.dtype | type, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise DenseInstanceRepairMetricError(f"array must have {ndim} dimensions")
    result = np.array(raw, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _normalize(value: object) -> object:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, frozenset):
        return [_normalize(item) for item in sorted(value)]
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _normalize(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _xyz_sha256(pair: OviObjectPairView) -> str:
    return _canonical_sha256(
        tuple((visit.visit_id, visit.points_xyz) for visit in pair.visits)
    )


@dataclass(frozen=True, slots=True)
class DenseMethodCandidate:
    visit_id: int
    candidate_id: str
    point_indices: np.ndarray
    parent_ovi_entity_point_counts: tuple[tuple[str, int], ...]
    owner_source: str
    temporal_identity_id: str | None
    temporal_confidence: float | None
    raw_query_id: str | None

    def __post_init__(self) -> None:
        if self.visit_id not in (0, 1):
            raise DenseInstanceRepairMetricError("candidate visit must be 0 or 1")
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise DenseInstanceRepairMetricError("candidate ID must be non-empty")
        points = _readonly(self.point_indices, np.int64, 1)
        if (
            not len(points)
            or np.any(points < 0)
            or not np.array_equal(points, np.sort(np.unique(points)))
        ):
            raise DenseInstanceRepairMetricError(
                "candidate points must be sorted, non-empty, and unique"
            )
        parents = tuple(self.parent_ovi_entity_point_counts)
        if (
            not parents
            or tuple(name for name, _count in parents)
            != tuple(sorted(name for name, _count in parents))
            or any(
                not isinstance(name, str)
                or not name
                or type(count) is not int
                or count <= 0
                for name, count in parents
            )
            or sum(count for _name, count in parents) != len(points)
        ):
            raise DenseInstanceRepairMetricError("candidate parent counts are invalid")
        if self.owner_source not in {
            "ovi_atomic",
            "ovi_composite",
            "query",
            "ovi_residual",
        }:
            raise DenseInstanceRepairMetricError("candidate owner source is invalid")
        if self.temporal_identity_id is not None and (
            not isinstance(self.temporal_identity_id, str)
            or not self.temporal_identity_id
        ):
            raise DenseInstanceRepairMetricError("temporal identity ID is invalid")
        if self.raw_query_id is not None and (
            not isinstance(self.raw_query_id, str) or not self.raw_query_id
        ):
            raise DenseInstanceRepairMetricError("raw query ID is invalid")
        confidence = self.temporal_confidence
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise DenseInstanceRepairMetricError("temporal confidence is invalid")
        if (self.temporal_identity_id is None) != (confidence is None):
            raise DenseInstanceRepairMetricError(
                "temporal identity and confidence must be jointly present"
            )
        object.__setattr__(self, "point_indices", points)
        object.__setattr__(self, "parent_ovi_entity_point_counts", parents)
        object.__setattr__(
            self,
            "temporal_confidence",
            None if confidence is None else float(confidence),
        )


@dataclass(frozen=True, slots=True)
class DenseMethodView:
    pair_id: str
    method_id: MethodId
    pair_content_sha256: str
    source_xyz_sha256: str
    output_xyz_sha256: str
    candidates: tuple[
        tuple[DenseMethodCandidate, ...], tuple[DenseMethodCandidate, ...]
    ]
    owner_instance_indices: tuple[np.ndarray, np.ndarray]
    owner_source_codes: tuple[np.ndarray, np.ndarray]
    neural_valid: tuple[np.ndarray, np.ndarray]
    raw_candidate_counts: tuple[int, int]
    _content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise DenseInstanceRepairMetricError("pair ID must be non-empty")
        if self.method_id not in {"P0", "P1", "P2"}:
            raise DenseInstanceRepairMetricError("method ID must be P0, P1, or P2")
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in (
                self.pair_content_sha256,
                self.source_xyz_sha256,
                self.output_xyz_sha256,
            )
        ):
            raise DenseInstanceRepairMetricError("method hashes are invalid")
        if self.source_xyz_sha256 != self.output_xyz_sha256:
            raise DenseInstanceRepairMetricError("method view changed dense XYZ")
        candidates = tuple(tuple(rows) for rows in self.candidates)
        if len(candidates) != 2:
            raise DenseInstanceRepairMetricError("method view must contain two visits")
        owners: list[np.ndarray] = []
        sources: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        for visit_id in (0, 1):
            visit_candidates = candidates[visit_id]
            if any(
                not isinstance(row, DenseMethodCandidate)
                or row.visit_id != visit_id
                for row in visit_candidates
            ):
                raise DenseInstanceRepairMetricError("candidate visit binding is invalid")
            ids = tuple(row.candidate_id for row in visit_candidates)
            if len(ids) != len(set(ids)):
                raise DenseInstanceRepairMetricError("candidate IDs are duplicated")
            owner = _readonly(self.owner_instance_indices[visit_id], np.int64, 1)
            source = _readonly(self.owner_source_codes[visit_id], np.uint8, 1)
            neural = _readonly(self.neural_valid[visit_id], np.bool_, 1)
            if (
                not len(owner)
                or len(source) != len(owner)
                or len(neural) != len(owner)
                or np.any(owner < -1)
                or np.any(owner >= len(visit_candidates))
                or np.any(
                    ~np.isin(
                        source,
                        (
                            OWNER_QUERY,
                            OWNER_OVI_RESIDUAL,
                            OWNER_BACKGROUND,
                            OWNER_UNKNOWN,
                        ),
                    )
                )
            ):
                raise DenseInstanceRepairMetricError("dense method arrays are invalid")
            expected = np.full(len(owner), -1, dtype=np.int64)
            for candidate_index, candidate in enumerate(visit_candidates):
                if np.any(candidate.point_indices >= len(owner)):
                    raise DenseInstanceRepairMetricError(
                        "candidate point is outside dense visit"
                    )
                if np.any(expected[candidate.point_indices] >= 0):
                    raise DenseInstanceRepairMetricError("candidate points overlap")
                expected[candidate.point_indices] = candidate_index
                required = (
                    OWNER_QUERY
                    if candidate.owner_source == "query"
                    else OWNER_OVI_RESIDUAL
                )
                if np.any(source[candidate.point_indices] != required):
                    raise DenseInstanceRepairMetricError(
                        "candidate and source ownership disagree"
                    )
            if not np.array_equal(expected, owner):
                raise DenseInstanceRepairMetricError(
                    "candidate records do not conserve owned rows"
                )
            if np.any(
                (owner < 0) != np.isin(source, (OWNER_BACKGROUND, OWNER_UNKNOWN))
            ):
                raise DenseInstanceRepairMetricError(
                    "unowned rows must be background or unknown"
                )
            owners.append(owner)
            sources.append(source)
            valid.append(neural)
        raw_counts = tuple(self.raw_candidate_counts)
        if (
            len(raw_counts) != 2
            or any(type(value) is not int or value < 0 for value in raw_counts)
        ):
            raise DenseInstanceRepairMetricError("raw candidate counts are invalid")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "owner_instance_indices", (owners[0], owners[1]))
        object.__setattr__(self, "owner_source_codes", (sources[0], sources[1]))
        object.__setattr__(self, "neural_valid", (valid[0], valid[1]))
        object.__setattr__(self, "raw_candidate_counts", raw_counts)
        object.__setattr__(
            self,
            "_content_sha256",
            _canonical_sha256(
                {
                    "pair_id": self.pair_id,
                    "method_id": self.method_id,
                    "pair_content_sha256": self.pair_content_sha256,
                    "source_xyz_sha256": self.source_xyz_sha256,
                    "output_xyz_sha256": self.output_xyz_sha256,
                    "candidates": tuple(
                        tuple(
                            {
                                "candidate_id": row.candidate_id,
                                "point_indices": row.point_indices,
                                "parents": row.parent_ovi_entity_point_counts,
                                "owner_source": row.owner_source,
                                "temporal_identity_id": row.temporal_identity_id,
                                "temporal_confidence": row.temporal_confidence,
                                "raw_query_id": row.raw_query_id,
                            }
                            for row in visit
                        )
                        for visit in candidates
                    ),
                    "owners": tuple(owners),
                    "sources": tuple(sources),
                    "neural_valid": tuple(valid),
                    "raw_candidate_counts": raw_counts,
                }
            ),
        )

    def content_sha256(self) -> str:
        return self._content_sha256


def _parent_counts(
    pair: OviObjectPairView, visit_id: int, points: np.ndarray
) -> tuple[tuple[str, int], ...]:
    visit = pair.visits[visit_id]
    owners = visit.entity_owner_indices[points]
    if np.any(owners < 0):
        raise DenseInstanceRepairMetricError("candidate contains source background")
    return tuple(
        sorted(
            (
                visit.entities[entity_index].entity_id,
                int(np.count_nonzero(owners == entity_index)),
            )
            for entity_index in {int(value) for value in owners}
        )
    )


def _base_view(
    pair: OviObjectPairView,
    *,
    method_id: MethodId,
    candidates: tuple[
        tuple[DenseMethodCandidate, ...], tuple[DenseMethodCandidate, ...]
    ],
    owners: tuple[np.ndarray, np.ndarray],
    sources: tuple[np.ndarray, np.ndarray],
    neural_valid: tuple[np.ndarray, np.ndarray],
    raw_counts: tuple[int, int],
) -> DenseMethodView:
    xyz_hash = _xyz_sha256(pair)
    return DenseMethodView(
        pair_id=pair.pair_id,
        method_id=method_id,
        pair_content_sha256=pair.content_sha256(),
        source_xyz_sha256=xyz_hash,
        output_xyz_sha256=xyz_hash,
        candidates=candidates,
        owner_instance_indices=owners,
        owner_source_codes=sources,
        neural_valid=neural_valid,
        raw_candidate_counts=raw_counts,
    )


def build_p0_method_view(pair: OviObjectPairView) -> DenseMethodView:
    """Build the immutable native OVI candidate view."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be OviObjectPairView")
    candidates = tuple(
        tuple(
            DenseMethodCandidate(
                visit_id=visit.visit_id,
                candidate_id=entity.entity_id,
                point_indices=entity.point_indices,
                parent_ovi_entity_point_counts=(
                    (entity.entity_id, entity.point_count),
                ),
                owner_source="ovi_atomic",
                temporal_identity_id=None,
                temporal_confidence=None,
                raw_query_id=None,
            )
            for entity in visit.entities
        )
        for visit in pair.visits
    )
    sources = []
    for visit in pair.visits:
        source = np.full(visit.point_count, OWNER_BACKGROUND, dtype=np.uint8)
        source[visit.entity_owner_indices >= 0] = OWNER_OVI_RESIDUAL
        sources.append(source)
    return _base_view(
        pair,
        method_id="P0",
        candidates=(candidates[0], candidates[1]),
        owners=(
            pair.visits[0].entity_owner_indices,
            pair.visits[1].entity_owner_indices,
        ),
        sources=(sources[0], sources[1]),
        neural_valid=(pair.visits[0].appearance_valid, pair.visits[1].appearance_valid),
        raw_counts=(len(candidates[0]), len(candidates[1])),
    )


def build_p1_method_view(
    pair: OviObjectPairView, grouping: TemporalObjectGrouping
) -> DenseMethodView:
    """Build the existing complete-OVI-entity grouping view."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be OviObjectPairView")
    if not isinstance(grouping, TemporalObjectGrouping):
        raise TypeError("grouping must be TemporalObjectGrouping")
    if (
        grouping.pair_content_sha256 != pair.content_sha256()
        or grouping.source_xyz_multiset_sha256 != grouping.output_xyz_multiset_sha256
    ):
        raise DenseInstanceRepairMetricError("P1 grouping does not bind the OVI pair")
    candidates = tuple(
        tuple(
            DenseMethodCandidate(
                visit_id=visit_id,
                candidate_id=row.object_id,
                point_indices=row.source_point_indices,
                parent_ovi_entity_point_counts=_parent_counts(
                    pair, visit_id, row.source_point_indices
                ),
                owner_source=(
                    "ovi_composite" if row.is_grouped else "ovi_atomic"
                ),
                temporal_identity_id=row.temporal_identity_id,
                temporal_confidence=row.query_confidence,
                raw_query_id=(row.query_ids[0] if len(row.query_ids) == 1 else None),
            )
            for row in grouping.objects[visit_id]
        )
        for visit_id in (0, 1)
    )
    sources = []
    for visit_id in (0, 1):
        source = np.full(pair.visits[visit_id].point_count, OWNER_BACKGROUND, dtype=np.uint8)
        source[grouping.owner_object_indices[visit_id] >= 0] = OWNER_OVI_RESIDUAL
        sources.append(source)
    return _base_view(
        pair,
        method_id="P1",
        candidates=(candidates[0], candidates[1]),
        owners=grouping.owner_object_indices,
        sources=(sources[0], sources[1]),
        neural_valid=(pair.visits[0].appearance_valid, pair.visits[1].appearance_valid),
        raw_counts=(len(pair.visits[0].entities), len(pair.visits[1].entities)),
    )


def build_p2_method_view(
    pair: OviObjectPairView, readout: DensePairReadout
) -> DenseMethodView:
    """Build the point-level ReScene query plus OVI residual view."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be OviObjectPairView")
    if not isinstance(readout, DensePairReadout):
        raise TypeError("readout must be DensePairReadout")
    if readout.pair_content_sha256 != pair.content_sha256():
        raise DenseInstanceRepairMetricError("P2 readout does not bind the OVI pair")
    score_by_query = dict(
        zip(readout.raw_query_indices.tolist(), readout.query_scores.tolist(), strict=True)
    )
    candidates = tuple(
        tuple(
            DenseMethodCandidate(
                visit_id=visit_id,
                candidate_id=row.instance_id,
                point_indices=row.point_indices,
                parent_ovi_entity_point_counts=row.parent_ovi_entity_point_counts,
                owner_source=row.owner_source,
                temporal_identity_id=row.temporal_identity_id,
                temporal_confidence=(
                    None
                    if row.temporal_identity_id is None
                    else score_by_query[row.raw_query_index]
                ),
                raw_query_id=(
                    None
                    if row.raw_query_index is None
                    else f"query_{row.raw_query_index:04d}"
                ),
            )
            for row in readout.visits[visit_id].instances
        )
        for visit_id in (0, 1)
    )
    return _base_view(
        pair,
        method_id="P2",
        candidates=(candidates[0], candidates[1]),
        owners=(
            readout.visits[0].owner_instance_indices,
            readout.visits[1].owner_instance_indices,
        ),
        sources=(
            readout.visits[0].owner_source_codes,
            readout.visits[1].owner_source_codes,
        ),
        neural_valid=(
            readout.visits[0].neural_valid,
            readout.visits[1].neural_valid,
        ),
        raw_counts=(len(readout.raw_proposals[0]), len(readout.raw_proposals[1])),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)


def _geometry_values(
    result: ThresholdGeometryResult, prediction_count: int, gt_count: int
) -> dict[str, object]:
    true_positive = result.matched_count
    precision = _ratio(true_positive, prediction_count)
    recall = _ratio(true_positive, gt_count)
    return {
        "tp": true_positive,
        "fp": prediction_count - true_positive,
        "fn": gt_count - true_positive,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "fragment_gt_count": result.fragment_gt_count,
        "merge_prediction_count": result.merge_prediction_count,
        "duplicate_prediction_count": result.duplicate_prediction_count,
        "mean_matched_iou": result.mean_matched_iou,
    }


def evaluate_dense_instance_method(
    pair: OviObjectPairView,
    view: DenseMethodView,
    ground_truth: GroundTruthPair,
) -> tuple[MappingProxyType, MappingProxyType]:
    """Evaluate full-visit class-agnostic instances at exact 5 cm IoU."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be OviObjectPairView")
    if not isinstance(view, DenseMethodView):
        raise TypeError("view must be DenseMethodView")
    if not isinstance(ground_truth, GroundTruthPair):
        raise TypeError("ground_truth must be GroundTruthPair")
    if (
        pair.pair_id != view.pair_id
        or pair.pair_id != ground_truth.pair_id
        or pair.content_sha256() != view.pair_content_sha256
        or _xyz_sha256(pair) != view.source_xyz_sha256
    ):
        raise DenseInstanceRepairMetricError("pair, method view, and GT differ")
    rows = []
    for visit_id in (0, 1):
        visit = pair.visits[visit_id]
        candidates = view.candidates[visit_id]
        predictions = tuple(
            PredictedInstance(
                prediction_id=row.candidate_id,
                voxels=voxelize_points(
                    visit.points_xyz[row.point_indices],
                    voxel_size_m=ground_truth.voxel_size_m,
                ),
            )
            for row in candidates
        )
        result = evaluate_instance_geometry(
            predictions,
            ground_truth.visits[visit_id],
            matching_policy="max_valid_count_then_iou",
        )
        primary = _geometry_values(
            result.primary, len(predictions), len(ground_truth.visits[visit_id])
        )
        sensitivity = _geometry_values(
            result.sensitivity, len(predictions), len(ground_truth.visits[visit_id])
        )
        parent_to_candidates: dict[str, set[str]] = {}
        for candidate in candidates:
            for parent_id, _count in candidate.parent_ovi_entity_point_counts:
                parent_to_candidates.setdefault(parent_id, set()).add(
                    candidate.candidate_id
                )
        sources = view.owner_source_codes[visit_id]
        count = len(sources)
        row = {
            "pair": pair.pair_id,
            "visit": visit_id,
            "method": view.method_id,
            "evaluation_domain": "full",
            **{
                f"{name}_at_050": value for name, value in primary.items()
            },
            **{
                f"{name}_at_025": value for name, value in sensitivity.items()
            },
            "raw_candidate_count": view.raw_candidate_counts[visit_id],
            "final_instance_count": len(candidates),
            "split_parent_count": sum(
                len(candidate_ids) > 1
                for candidate_ids in parent_to_candidates.values()
            ),
            "merged_candidate_count": sum(
                len(candidate.parent_ovi_entity_point_counts) > 1
                for candidate in candidates
            ),
            "query_owned_fraction": np.count_nonzero(sources == OWNER_QUERY) / count,
            "residual_fraction": np.count_nonzero(sources == OWNER_OVI_RESIDUAL) / count,
            "background_fraction": np.count_nonzero(sources == OWNER_BACKGROUND) / count,
            "unknown_fraction": np.count_nonzero(sources == OWNER_UNKNOWN) / count,
            "source_point_count": count,
            "final_point_count": count,
            "geometric_change_count": 0,
            "method_view_sha256": view.content_sha256(),
        }
        rows.append(MappingProxyType(row))
    return rows[0], rows[1]


def _ground_truth_sha256(ground_truth: GroundTruthPair) -> str:
    rules = ground_truth.identity_rules
    return _canonical_sha256(
        {
            "pair_id": ground_truth.pair_id,
            "voxel_size_m": ground_truth.voxel_size_m,
            "visits": tuple(
                tuple(
                    {
                        "instance_id": row.instance_id,
                        "semantic_label": row.semantic_label,
                        "voxels": tuple(sorted(row.voxels)),
                    }
                    for row in visit
                )
                for visit in ground_truth.visits
            ),
            "identity_rules": {
                "rescan_to_reference": dict(rules.rescan_to_reference),
                "symmetry_by_reference": dict(rules.symmetry_by_reference),
                "rigid_transform_by_reference": dict(
                    rules.rigid_transform_by_reference
                ),
                "removed_reference_ids": rules.removed_reference_ids,
                "ambiguity_groups": rules.ambiguity_groups,
            },
        }
    )


@dataclass(frozen=True, slots=True)
class DenseEndpointBindings:
    pair_id: str
    pair_content_sha256: str
    method_id: MethodId
    method_view_sha256: str
    ground_truth_sha256: str
    voxel_size_m: float
    support_domain: SupportDomain
    candidate_ids: tuple[tuple[str, ...], tuple[str, ...]]
    thresholds: tuple[
        tuple[
            float,
            tuple[
                tuple[CandidateEndpointBinding, ...],
                tuple[CandidateEndpointBinding, ...],
            ],
        ],
        ...,
    ]

    def for_threshold(
        self, threshold: float
    ) -> tuple[
        tuple[CandidateEndpointBinding, ...],
        tuple[CandidateEndpointBinding, ...],
    ]:
        for value, rows in self.thresholds:
            if math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12):
                return rows
        raise DenseEndpointBindingError("endpoint threshold is unavailable")

    def content_sha256(self) -> str:
        return _canonical_sha256(
            {
                "pair_id": self.pair_id,
                "pair_content_sha256": self.pair_content_sha256,
                "method_id": self.method_id,
                "method_view_sha256": self.method_view_sha256,
                "ground_truth_sha256": self.ground_truth_sha256,
                "voxel_size_m": self.voxel_size_m,
                "support_domain": self.support_domain,
                "candidate_ids": self.candidate_ids,
                "thresholds": tuple(
                    (
                        threshold,
                        tuple(
                            tuple(
                                (
                                    row.candidate_id,
                                    row.assigned_gt_instance_id,
                                    row.assigned_iou,
                                    row.best_gt_instance_id,
                                    row.best_iou,
                                    row.qualifying_gt_instance_ids,
                                )
                                for row in visit
                            )
                            for visit in visits
                        ),
                    )
                    for threshold, visits in self.thresholds
                ),
            }
        )


def _candidate_binding(
    candidate: DenseMethodCandidate,
    *,
    points_xyz: np.ndarray,
    targets: tuple[GroundTruthInstance, ...],
    voxel_size_m: float,
    threshold: float,
) -> CandidateEndpointBinding:
    if not len(points_xyz):
        return CandidateEndpointBinding(
            candidate.candidate_id, None, None, None, None, ()
        )
    voxels = voxelize_points(points_xyz, voxel_size_m=voxel_size_m)
    scored = tuple(
        sorted(
            (
                (
                    len(voxels & target.voxels) / len(voxels | target.voxels),
                    target.instance_id,
                )
                for target in targets
            ),
            key=lambda value: (-value[0], value[1]),
        )
    )
    if not scored:
        return CandidateEndpointBinding(
            candidate.candidate_id, None, None, None, None, ()
        )
    best_iou, best_id = scored[0]
    assigned = best_id if best_iou >= threshold else None
    return CandidateEndpointBinding(
        candidate_id=candidate.candidate_id,
        assigned_gt_instance_id=assigned,
        assigned_iou=None if assigned is None else float(best_iou),
        best_gt_instance_id=best_id,
        best_iou=float(best_iou),
        qualifying_gt_instance_ids=tuple(
            sorted(instance_id for iou, instance_id in scored if iou >= threshold)
        ),
    )


def build_dense_endpoint_bindings(
    pair: OviObjectPairView,
    view: DenseMethodView,
    ground_truth: GroundTruthPair,
    *,
    support_domain: SupportDomain,
) -> DenseEndpointBindings:
    """Freeze one method candidate pool against GT before identity scoring."""

    if support_domain not in {"full", "supported"}:
        raise DenseEndpointBindingError("endpoint support domain is invalid")
    if (
        pair.pair_id != view.pair_id
        or pair.pair_id != ground_truth.pair_id
        or pair.content_sha256() != view.pair_content_sha256
    ):
        raise DenseEndpointBindingError("pair, method view, and GT differ")
    thresholds = []
    for threshold in (0.50, 0.25):
        visits = []
        for visit_id in (0, 1):
            visit = pair.visits[visit_id]
            rows = []
            for candidate in view.candidates[visit_id]:
                indices = candidate.point_indices
                if support_domain == "supported":
                    indices = indices[view.neural_valid[visit_id][indices]]
                rows.append(
                    _candidate_binding(
                        candidate,
                        points_xyz=visit.points_xyz[indices],
                        targets=ground_truth.visits[visit_id],
                        voxel_size_m=ground_truth.voxel_size_m,
                        threshold=threshold,
                    )
                )
            visits.append(tuple(rows))
        thresholds.append((threshold, (visits[0], visits[1])))
    return DenseEndpointBindings(
        pair_id=pair.pair_id,
        pair_content_sha256=pair.content_sha256(),
        method_id=view.method_id,
        method_view_sha256=view.content_sha256(),
        ground_truth_sha256=_ground_truth_sha256(ground_truth),
        voxel_size_m=ground_truth.voxel_size_m,
        support_domain=support_domain,
        candidate_ids=tuple(
            tuple(candidate.candidate_id for candidate in visit)
            for visit in view.candidates
        ),
        thresholds=tuple(thresholds),
    )


def _persistent_targets(
    ground_truth: GroundTruthPair,
) -> dict[int, tuple[int, frozenset[int]]]:
    left_ids = {target.instance_id for target in ground_truth.visits[0]}
    result = {}
    for target in ground_truth.visits[1]:
        reference_id = ground_truth.identity_rules.reference_id_for_rescan(
            target.instance_id
        )
        allowed = frozenset(
            instance_id
            for instance_id in left_ids
            if ground_truth.identity_rules.is_ambiguous_equivalent(
                instance_id, reference_id
            )
        )
        if allowed:
            result[target.instance_id] = (reference_id, allowed)
    return result


def evaluate_dense_identity(
    view: DenseMethodView,
    ground_truth: GroundTruthPair,
    bindings: DenseEndpointBindings,
    *,
    iou_threshold: float,
) -> MappingProxyType:
    """Score temporal identities after validating the frozen endpoint pool."""

    if iou_threshold not in {0.50, 0.25}:
        raise DenseEndpointBindingError("identity IoU must be 0.50 or 0.25")
    candidate_ids = tuple(
        tuple(candidate.candidate_id for candidate in visit)
        for visit in view.candidates
    )
    if (
        bindings.pair_id != view.pair_id
        or bindings.pair_id != ground_truth.pair_id
        or bindings.pair_content_sha256 != view.pair_content_sha256
        or bindings.method_id != view.method_id
        or bindings.method_view_sha256 != view.content_sha256()
        or bindings.ground_truth_sha256 != _ground_truth_sha256(ground_truth)
        or bindings.voxel_size_m != ground_truth.voxel_size_m
        or bindings.candidate_ids != candidate_ids
    ):
        raise DenseEndpointBindingError("endpoint binding names a different method view")
    selected = bindings.for_threshold(iou_threshold)
    assigned = tuple(
        {row.candidate_id: row.assigned_gt_instance_id for row in visit}
        for visit in selected
    )
    candidates_by_identity = []
    for visit in view.candidates:
        groups: dict[str, list[DenseMethodCandidate]] = {}
        for candidate in visit:
            if candidate.temporal_identity_id is not None:
                groups.setdefault(candidate.temporal_identity_id, []).append(candidate)
        candidates_by_identity.append(groups)
    shared = sorted(set(candidates_by_identity[0]) & set(candidates_by_identity[1]))
    paired = [
        identity
        for identity in shared
        if len(candidates_by_identity[0][identity]) == 1
        and len(candidates_by_identity[1][identity]) == 1
    ]
    persistent = _persistent_targets(ground_truth)
    represented = tuple(
        {
            value
            for value in visit.values()
            if value is not None
        }
        for visit in assigned
    )
    conditional_targets = {
        right_id
        for right_id, (_reference_id, allowed) in persistent.items()
        if right_id in represented[1] and bool(allowed & represented[0])
    }
    rigid_targets = {
        right_id
        for right_id, (reference_id, _allowed) in persistent.items()
        if reference_id in ground_truth.identity_rules.rigid_transform_by_reference
    }
    labels = tuple(
        {
            target.instance_id: target.semantic_label.casefold()
            for target in visit
        }
        for visit in ground_truth.visits
    )
    hits: set[int] = set()
    duplicates = 0
    endpoint_failures = 0
    false_reids = 0
    same_class_mismatches = 0
    outcomes = []
    for identity in paired:
        left = candidates_by_identity[0][identity][0]
        right = candidates_by_identity[1][identity][0]
        left_gt = assigned[0][left.candidate_id]
        right_gt = assigned[1][right.candidate_id]
        target = None if right_gt is None else persistent.get(right_gt)
        correct = bool(
            target is not None and left_gt is not None and left_gt in target[1]
        )
        if left_gt is None or right_gt is None:
            outcome = "endpoint_failure"
            endpoint_failures += 1
        elif not correct:
            outcome = "false_reid"
            false_reids += 1
            if labels[0][left_gt] == labels[1][right_gt]:
                same_class_mismatches += 1
        elif right_gt in hits:
            outcome = "duplicate"
            duplicates += 1
        else:
            outcome = "true_positive"
            hits.add(right_gt)
        outcomes.append(
            {
                "temporal_identity_id": identity,
                "t0_candidate_id": left.candidate_id,
                "t1_candidate_id": right.candidate_id,
                "t0_gt_instance_id": left_gt,
                "t1_gt_instance_id": right_gt,
                "outcome": outcome,
            }
        )
    conditional_recall = _ratio(
        len(hits & conditional_targets), len(conditional_targets)
    )
    result = {
        "schema_version": 1,
        "protocol_id": "RSCAN_DENSE_FIXED_ENDPOINT_IDENTITY_V1",
        "status": "PASS",
        "pair": view.pair_id,
        "method": view.method_id,
        "support_domain": bindings.support_domain,
        "iou_threshold": iou_threshold,
        "method_view_sha256": view.content_sha256(),
        "binding_sha256": bindings.content_sha256(),
        "paired_prediction_count": len(paired),
        "ambiguous_identity_count": len(shared) - len(paired),
        "true_positive_count": len(hits),
        "false_positive_count": len(paired) - len(hits),
        "endpoint_failure_count": endpoint_failures,
        "false_reid_count": false_reids,
        "same_class_mismatch_count": same_class_mismatches,
        "duplicate_count": duplicates,
        "precision": _ratio(len(hits), len(paired)),
        "persistent_gt_count": len(persistent),
        "end_to_end_recall": _ratio(len(hits), len(persistent)),
        "conditional_gt_count": len(conditional_targets),
        "conditional_recall": conditional_recall,
        "conditional_status": (
            "PASS"
            if conditional_recall is not None
            else "NOT_COMPUTED_NO_FIXED_ENDPOINT_SUPPORT"
        ),
        "rigid_gt_count": len(rigid_targets),
        "rigid_recall": _ratio(len(hits & rigid_targets), len(rigid_targets)),
        "outcomes": tuple(outcomes),
    }
    return MappingProxyType(result)


def _dense_candidate_pool(
    view: DenseMethodView,
) -> tuple[tuple[tuple[int, str], ...], tuple[tuple[int, str], ...]]:
    return tuple(
        tuple((visit_id, row.candidate_id) for row in visit)
        for visit_id, visit in enumerate(view.candidates)
    )  # type: ignore[return-value]


def _shape_descriptor(points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.zeros(6, dtype=np.float64)
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    radii = np.linalg.norm(centered, axis=1)
    covariance = centered.T @ centered / max(len(centered), 1)
    return np.concatenate(
        (np.quantile(radii, (0.25, 0.50, 0.75)), np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0.0)))
    )


def build_dense_geometric_score_matrix(
    pair: OviObjectPairView,
    view: DenseMethodView,
    *,
    supported_only: bool,
    centroid_scale_m: float,
) -> ObjectPairScoreMatrix:
    """Score the fixed dense candidate pool using geometry only."""

    if pair.content_sha256() != view.pair_content_sha256:
        raise DenseInstanceRepairMetricError("dense geometry binds a different pair")
    if type(supported_only) is not bool:
        raise DenseInstanceRepairMetricError("supported_only must be boolean")
    if (
        isinstance(centroid_scale_m, bool)
        or not isinstance(centroid_scale_m, (int, float, np.number))
        or not math.isfinite(float(centroid_scale_m))
        or float(centroid_scale_m) <= 0.0
    ):
        raise DenseInstanceRepairMetricError("centroid scale must be finite and positive")
    before = view.content_sha256()
    points: list[tuple[np.ndarray, ...]] = []
    for visit_id, candidates in enumerate(view.candidates):
        visit_points = []
        for candidate in candidates:
            indices = candidate.point_indices
            if supported_only:
                indices = indices[view.neural_valid[visit_id][indices]]
            visit_points.append(pair.visits[visit_id].points_xyz[indices])
        points.append(tuple(visit_points))
    centroids = tuple(
        tuple(
            np.mean(value, axis=0, dtype=np.float64) if len(value) else np.zeros(3)
            for value in visit
        )
        for visit in points
    )
    shapes = tuple(tuple(_shape_descriptor(value) for value in visit) for visit in points)
    scores = np.zeros((len(points[0]), len(points[1])), dtype=np.float64)
    eligible = np.zeros(scores.shape, dtype=np.bool_)
    scale = float(centroid_scale_m)
    for left in range(scores.shape[0]):
        for right in range(scores.shape[1]):
            available = bool(len(points[0][left]) and len(points[1][right]))
            centroid_distance = float(np.linalg.norm(centroids[0][left] - centroids[1][right]))
            shape_distance = float(np.linalg.norm(shapes[0][left] - shapes[1][right]))
            scores[left, right] = (
                0.5 * math.exp(-min(centroid_distance / scale, 50.0))
                + 0.5 * math.exp(-min(shape_distance, 50.0))
                if available
                else 0.0
            )
            eligible[left, right] = available
    if view.content_sha256() != before:
        raise DenseInstanceRepairMetricError("dense geometry scoring mutated the view")
    return ObjectPairScoreMatrix(
        method_id="G_supported" if supported_only else "G_full",
        pair_content_sha256=view.pair_content_sha256,
        candidate_ids=_dense_candidate_pool(view),
        scores=scores,
        eligible=eligible,
    )


def _candidate_feature_inputs(
    view: DenseMethodView,
    features: tuple[np.ndarray, np.ndarray],
    valid: tuple[np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    if not isinstance(features, tuple) or not isinstance(valid, tuple) or len(features) != 2 or len(valid) != 2:
        raise DenseInstanceRepairMetricError("candidate features must contain two visits")
    output_features = []
    output_valid = []
    width = None
    for visit_id in (0, 1):
        array = np.asarray(features[visit_id], dtype=np.float64)
        mask = np.asarray(valid[visit_id])
        expected = len(view.candidates[visit_id])
        if (
            array.ndim != 2
            or array.shape[0] != expected
            or array.shape[1] < 1
            or not np.all(np.isfinite(array))
            or mask.dtype != np.bool_
            or mask.shape != (expected,)
            or (width is not None and array.shape[1] != width)
        ):
            raise DenseInstanceRepairMetricError("candidate feature inputs are invalid")
        norms = np.linalg.norm(array, axis=1)
        if np.any(norms[mask] <= 0.0):
            raise DenseInstanceRepairMetricError("valid candidate features must be nonzero")
        normalized = np.zeros(array.shape, dtype=np.float64)
        normalized[mask] = array[mask] / norms[mask, None]
        output_features.append(normalized)
        output_valid.append(np.array(mask, copy=True))
        width = array.shape[1]
    return (output_features[0], output_features[1]), (output_valid[0], output_valid[1])


def build_dense_feature_score_matrix(
    view: DenseMethodView,
    features: tuple[np.ndarray, np.ndarray],
    valid: tuple[np.ndarray, np.ndarray],
) -> ObjectPairScoreMatrix:
    """Cosine-score independently extracted features on the fixed dense pool."""

    normalized, masks = _candidate_feature_inputs(view, features, valid)
    eligible = masks[0][:, None] & masks[1][None, :]
    scores = np.where(
        eligible,
        np.clip(normalized[0] @ normalized[1].T, 0.0, 1.0),
        0.0,
    )
    return ObjectPairScoreMatrix(
        method_id="F_obj",
        pair_content_sha256=view.pair_content_sha256,
        candidate_ids=_dense_candidate_pool(view),
        scores=scores,
        eligible=eligible,
    )


def build_dense_rescene_score_matrix(
    view: DenseMethodView,
    *,
    query_scores: np.ndarray,
    candidate_affinities: tuple[np.ndarray, np.ndarray],
    candidate_valid: tuple[np.ndarray, np.ndarray],
) -> ObjectPairScoreMatrix:
    """Apply max-q c_q a_qi^0 a_qj^1 to the fixed dense pool."""

    confidence = np.asarray(query_scores, dtype=np.float64)
    if confidence.ndim != 1 or not len(confidence) or not np.all(np.isfinite(confidence)) or np.any((confidence < 0.0) | (confidence > 1.0)):
        raise DenseInstanceRepairMetricError("query scores are invalid")
    if not isinstance(candidate_affinities, tuple) or not isinstance(candidate_valid, tuple) or len(candidate_affinities) != 2 or len(candidate_valid) != 2:
        raise DenseInstanceRepairMetricError("query candidate inputs must contain two visits")
    affinities = []
    valid = []
    for visit_id in (0, 1):
        array = np.asarray(candidate_affinities[visit_id], dtype=np.float64)
        mask = np.asarray(candidate_valid[visit_id])
        expected = (len(confidence), len(view.candidates[visit_id]))
        if (
            array.shape != expected
            or not np.all(np.isfinite(array))
            or np.any((array < 0.0) | (array > 1.0))
            or mask.dtype != np.bool_
            or mask.shape != (expected[1],)
        ):
            raise DenseInstanceRepairMetricError("query candidate affinities are invalid")
        affinities.append(array)
        valid.append(mask)
    products = confidence[:, None, None] * affinities[0][:, :, None] * affinities[1][:, None, :]
    scores = np.max(products, axis=0)
    eligible = valid[0][:, None] & valid[1][None, :]
    return ObjectPairScoreMatrix(
        method_id="R_obj",
        pair_content_sha256=view.pair_content_sha256,
        candidate_ids=_dense_candidate_pool(view),
        scores=np.where(eligible, scores, 0.0),
        eligible=eligible,
    )


def solve_dense_pair_assignment(
    pair: OviObjectPairView,
    view: DenseMethodView,
    matrix: ObjectPairScoreMatrix,
    *,
    minimum_match_score: float,
    static_centroid_tolerance_m: float,
) -> tuple[ObjectPairPrediction, ...]:
    """Solve one cardinality-first 1:1 assignment on a fixed dense pool."""

    if (
        pair.content_sha256() != view.pair_content_sha256
        or matrix.pair_content_sha256 != view.pair_content_sha256
        or matrix.candidate_ids != _dense_candidate_pool(view)
    ):
        raise DenseInstanceRepairMetricError("dense association substitutes the candidate pool")
    for value, label in (
        (minimum_match_score, "minimum match score"),
        (static_centroid_tolerance_m, "static centroid tolerance"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise DenseInstanceRepairMetricError(f"{label} must be finite and positive")
    if float(minimum_match_score) > 1.0:
        raise DenseInstanceRepairMetricError("minimum match score cannot exceed one")
    first_count, second_count = matrix.scores.shape
    size = first_count + second_count
    utility = np.full((size, size), -1e9, dtype=np.float64)
    allowed = matrix.eligible & (matrix.scores >= float(minimum_match_score))
    utility[:first_count, :second_count][allowed] = min(first_count, second_count) + 1.0 + matrix.scores[allowed]
    for index in range(first_count):
        utility[index, second_count + index] = 0.0
    for index in range(second_count):
        utility[first_count + index, index] = 0.0
    utility[first_count:, second_count:] = 0.0
    rows, columns = linear_sum_assignment(-utility)
    matches = sorted(
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
        if row < first_count and column < second_count and allowed[row, column]
    )
    used_left = {left for left, _right in matches}
    used_right = {right for _left, right in matches}
    centroids = tuple(
        tuple(np.mean(pair.visits[visit_id].points_xyz[row.point_indices], axis=0) for row in visit)
        for visit_id, visit in enumerate(view.candidates)
    )
    output = []
    for left, right in matches:
        distance = float(np.linalg.norm(centroids[0][left] - centroids[1][right]))
        output.append(
            ObjectPairPrediction(
                prediction_id=f"{matrix.method_id}:pair:{left:04d}:{right:04d}",
                pair_id=view.pair_id,
                method_id=matrix.method_id,
                t0_entity_id=view.candidates[0][left].candidate_id,
                t1_entity_id=view.candidates[1][right].candidate_id,
                score=float(matrix.scores[left, right]),
                state="persistent_static" if distance <= float(static_centroid_tolerance_m) else "persistent_moved",
                evidence={"centroid_distance_m": distance, "score_matrix_sha256": matrix.content_sha256()},
            )
        )
    for left in sorted(set(range(first_count)) - used_left):
        output.append(
            ObjectPairPrediction(
                prediction_id=f"{matrix.method_id}:t0:{left:04d}",
                pair_id=view.pair_id,
                method_id=matrix.method_id,
                t0_entity_id=view.candidates[0][left].candidate_id,
                t1_entity_id=None,
                score=None,
                state="unmatched_t0",
            )
        )
    for right in sorted(set(range(second_count)) - used_right):
        output.append(
            ObjectPairPrediction(
                prediction_id=f"{matrix.method_id}:t1:{right:04d}",
                pair_id=view.pair_id,
                method_id=matrix.method_id,
                t0_entity_id=None,
                t1_entity_id=view.candidates[1][right].candidate_id,
                score=None,
                state="unmatched_t1",
            )
        )
    return tuple(output)


def evaluate_dense_association(
    view: DenseMethodView,
    ground_truth: GroundTruthPair,
    bindings: DenseEndpointBindings,
    predictions: tuple[ObjectPairPrediction, ...],
    *,
    iou_threshold: float,
) -> MappingProxyType:
    """Evaluate G/F/R identities against one fixed P2 endpoint binding."""

    if iou_threshold not in {0.50, 0.25}:
        raise DenseEndpointBindingError("identity IoU must be 0.50 or 0.25")
    candidate_ids = tuple(tuple(row.candidate_id for row in visit) for visit in view.candidates)
    if (
        view.method_id != "P2"
        or bindings.pair_id != view.pair_id
        or bindings.pair_content_sha256 != view.pair_content_sha256
        or bindings.method_id != "P2"
        or bindings.method_view_sha256 != view.content_sha256()
        or bindings.ground_truth_sha256 != _ground_truth_sha256(ground_truth)
        or bindings.candidate_ids != candidate_ids
    ):
        raise DenseEndpointBindingError("fixed P2 binding identity mismatch")
    rows = tuple(predictions)
    if not rows or any(not isinstance(row, ObjectPairPrediction) or row.pair_id != view.pair_id for row in rows):
        raise DenseEndpointBindingError("dense association predictions are invalid")
    method_ids = {row.method_id for row in rows}
    if len(method_ids) != 1:
        raise DenseEndpointBindingError("dense association predictions mix methods")
    observed = tuple(
        [getattr(row, f"t{visit_id}_entity_id") for row in rows if getattr(row, f"t{visit_id}_entity_id") is not None]
        for visit_id in (0, 1)
    )
    if any(len(values) != len(set(values)) or set(values) != set(candidate_ids[index]) for index, values in enumerate(observed)):
        raise DenseEndpointBindingError("dense association does not conserve candidates")
    selected = bindings.for_threshold(iou_threshold)
    assigned = tuple({row.candidate_id: row.assigned_gt_instance_id for row in visit} for visit in selected)
    represented = tuple({value for value in visit.values() if value is not None} for visit in assigned)
    persistent = _persistent_targets(ground_truth)
    conditional_targets = {
        right_id
        for right_id, (_reference_id, allowed) in persistent.items()
        if right_id in represented[1] and bool(allowed & represented[0])
    }
    rigid_targets = {
        right_id
        for right_id, (reference_id, _allowed) in persistent.items()
        if reference_id in ground_truth.identity_rules.rigid_transform_by_reference
    }
    labels = tuple(
        {target.instance_id: target.semantic_label.casefold() for target in visit}
        for visit in ground_truth.visits
    )
    hits: set[int] = set()
    endpoint_failures = false_reids = same_class_mismatches = duplicates = 0
    outcomes = []
    paired = sorted((row for row in rows if row.is_matched), key=lambda row: (-float(row.score), row.prediction_id))
    for prediction in paired:
        assert prediction.t0_entity_id is not None and prediction.t1_entity_id is not None
        left_gt = assigned[0][prediction.t0_entity_id]
        right_gt = assigned[1][prediction.t1_entity_id]
        target = None if right_gt is None else persistent.get(right_gt)
        correct = bool(target is not None and left_gt is not None and left_gt in target[1])
        if left_gt is None or right_gt is None:
            outcome = "endpoint_failure"
            endpoint_failures += 1
        elif not correct:
            outcome = "false_reid"
            false_reids += 1
            if labels[0][left_gt] == labels[1][right_gt]:
                same_class_mismatches += 1
        elif right_gt in hits:
            outcome = "duplicate"
            duplicates += 1
        else:
            outcome = "true_positive"
            hits.add(right_gt)
        outcomes.append(
            {
                "prediction_id": prediction.prediction_id,
                "t0_candidate_id": prediction.t0_entity_id,
                "t1_candidate_id": prediction.t1_entity_id,
                "t0_gt_instance_id": left_gt,
                "t1_gt_instance_id": right_gt,
                "outcome": outcome,
            }
        )
    conditional_recall = _ratio(len(hits & conditional_targets), len(conditional_targets))
    return MappingProxyType(
        {
            "schema_version": 1,
            "protocol_id": "RSCAN_DENSE_FIXED_P2_ASSOCIATION_V1",
            "status": "PASS",
            "pair": view.pair_id,
            "method": next(iter(method_ids)),
            "support_domain": bindings.support_domain,
            "iou_threshold": iou_threshold,
            "method_view_sha256": view.content_sha256(),
            "binding_sha256": bindings.content_sha256(),
            "paired_prediction_count": len(paired),
            "ambiguous_identity_count": 0,
            "true_positive_count": len(hits),
            "false_positive_count": len(paired) - len(hits),
            "endpoint_failure_count": endpoint_failures,
            "false_reid_count": false_reids,
            "same_class_mismatch_count": same_class_mismatches,
            "duplicate_count": duplicates,
            "precision": _ratio(len(hits), len(paired)),
            "persistent_gt_count": len(persistent),
            "end_to_end_recall": _ratio(len(hits), len(persistent)),
            "conditional_gt_count": len(conditional_targets),
            "conditional_recall": conditional_recall,
            "conditional_status": "PASS" if conditional_recall is not None else "NOT_COMPUTED_NO_FIXED_ENDPOINT_SUPPORT",
            "rigid_gt_count": len(rigid_targets),
            "rigid_recall": _ratio(len(hits & rigid_targets), len(rigid_targets)),
            "outcomes": tuple(outcomes),
        }
    )


__all__ = [
    "DenseEndpointBindingError",
    "DenseEndpointBindings",
    "DenseInstanceRepairMetricError",
    "DenseMethodCandidate",
    "DenseMethodView",
    "build_dense_endpoint_bindings",
    "build_dense_feature_score_matrix",
    "build_dense_geometric_score_matrix",
    "build_dense_rescene_score_matrix",
    "build_p0_method_view",
    "build_p1_method_view",
    "build_p2_method_view",
    "evaluate_dense_association",
    "evaluate_dense_identity",
    "evaluate_dense_instance_method",
    "solve_dense_pair_assignment",
]
