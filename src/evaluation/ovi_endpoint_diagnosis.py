"""Evaluator-only endpoint diagnosis on immutable OVI dense surfaces."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from itertools import combinations
from types import MappingProxyType
from typing import Mapping

import numpy as np

from src.evaluation.ovi_pair_views import OviObjectVisitView
from src.evaluation.rscan_gt_instances import (
    GroundTruthInstance,
    PredictedInstance,
    Voxel,
    voxelize_points,
)


class EndpointDiagnosisError(ValueError):
    """Raised when endpoint diagnosis inputs violate the evaluation contract."""


def _readonly_indices(value: object, *, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise EndpointDiagnosisError(f"{label} must be a one-dimensional integer array")
    result = np.array(raw, dtype=np.int64, copy=True, order="C")
    if len(result) == 0 or np.any(result < 0) or len(np.unique(result)) != len(result):
        raise EndpointDiagnosisError(f"{label} must contain unique non-negative rows")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class RawDenseProposal:
    query_id: str
    query_score: float
    point_indices: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise EndpointDiagnosisError("raw proposal query ID must be non-empty")
        if (
            isinstance(self.query_score, bool)
            or not isinstance(self.query_score, (int, float))
            or not math.isfinite(float(self.query_score))
            or not 0.0 <= float(self.query_score) <= 1.0
        ):
            raise EndpointDiagnosisError("raw proposal score must lie in [0, 1]")
        object.__setattr__(self, "query_score", float(self.query_score))
        object.__setattr__(
            self,
            "point_indices",
            _readonly_indices(self.point_indices, label="raw proposal point indices"),
        )


@dataclass(frozen=True, slots=True)
class EndpointDiagnosisRow:
    pair_id: str
    visit_id: int
    gt_instance_id: int
    change_type: str
    full_gt_voxels: int
    sensor_visible_gt_voxels: int | None
    sensor_visibility_status: str
    d_intersection_voxels: int
    d_coverage: float
    d_owned_coverage: float
    d_supported_coverage: float
    best_atomic_id: str | None
    best_atomic_iou: float | None
    best_atomic_precision: float | None
    best_atomic_recall: float | None
    second_atomic_iou: float | None
    union_oracle_iou: float | None
    union_search_exact: bool
    union_candidate_count: int
    union_member_ids: tuple[str, ...]
    split_oracle_iou: float
    split_oracle_voxels: frozenset[Voxel] = field(repr=False)
    best_raw_mask_iou: float | None
    best_raw_query_id: str | None
    best_raw_query_score: float | None
    best_confident_mask_iou: float | None
    final_p2_iou: float | None
    provisional_failure_type: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("split_oracle_voxels")
        result["D_coverage"] = result.pop("d_coverage")
        result["D_owned_coverage"] = result.pop("d_owned_coverage")
        result["D_supported_coverage"] = result.pop("d_supported_coverage")
        result["final_P2_iou"] = result.pop("final_p2_iou")
        return result


def _iou(left: frozenset[Voxel], right: frozenset[Voxel]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def _precision_recall(
    prediction: frozenset[Voxel], target: frozenset[Voxel]
) -> tuple[float | None, float]:
    intersection = len(prediction & target)
    return (
        None if not prediction else intersection / len(prediction),
        intersection / len(target),
    )


def _best_union(
    candidates: tuple[tuple[str, frozenset[Voxel]], ...],
    target: frozenset[Voxel],
    *,
    exact_limit: int,
) -> tuple[float | None, bool, tuple[str, ...]]:
    intersecting = tuple(row for row in candidates if row[1] & target)
    if not intersecting:
        return None, True, ()
    if len(intersecting) <= exact_limit:
        best = (-1.0, ())
        for width in range(1, len(intersecting) + 1):
            for selected in combinations(intersecting, width):
                voxels = frozenset().union(*(row[1] for row in selected))
                member_ids = tuple(row[0] for row in selected)
                rank = (_iou(voxels, target), tuple(reversed(member_ids)))
                if rank > best:
                    best = rank
        return best[0], True, tuple(reversed(best[1]))

    start = max(intersecting, key=lambda row: (_iou(row[1], target), row[0]))
    selected = [start]
    union = start[1]
    remaining = [row for row in intersecting if row[0] != start[0]]
    while remaining:
        current = _iou(union, target)
        candidate = max(
            remaining,
            key=lambda row: (_iou(union | row[1], target), row[0]),
        )
        improved = _iou(union | candidate[1], target)
        if improved <= current:
            break
        selected.append(candidate)
        union = union | candidate[1]
        remaining.remove(candidate)
    return _iou(union, target), False, tuple(sorted(row[0] for row in selected))


def _best_proposal(
    proposals: tuple[tuple[str, float, frozenset[Voxel]], ...],
    target: frozenset[Voxel],
) -> tuple[float | None, str | None, float | None]:
    if not proposals:
        return None, None, None
    query_id, score, voxels = min(
        proposals,
        key=lambda row: (-_iou(row[2], target), -row[1], row[0]),
    )
    return _iou(voxels, target), query_id, score


def _failure_type(
    *,
    d_coverage: float,
    sensor_coverage: float | None,
    single_iou: float | None,
    union_iou: float | None,
    split_iou: float,
    raw_iou: float | None,
    final_iou: float | None,
) -> str:
    eps = 1e-12
    flags: set[str] = set()
    if sensor_coverage is not None:
        if sensor_coverage + eps < 1.0 and d_coverage + eps >= sensor_coverage:
            flags.add("observation_limited")
        if sensor_coverage > d_coverage + eps:
            flags.add("reconstruction_or_alignment")
    if union_iou is not None and single_iou is not None and union_iou > single_iou + eps:
        flags.add("oversegmented")
    if union_iou is not None and split_iou > union_iou + eps:
        flags.add("undersegmented")
    if raw_iou is not None and raw_iou + eps < split_iou:
        flags.add("raw_mask_limited")
    if raw_iou is not None and (final_iou is None or raw_iou > final_iou + eps):
        flags.add("readout_limited")
    return next(iter(flags)) if len(flags) == 1 else "mixed"


def diagnose_visit_endpoints(
    *,
    pair_id: str,
    visit: OviObjectVisitView,
    ground_truth: tuple[GroundTruthInstance, ...],
    raw_proposals: tuple[RawDenseProposal, ...],
    confident_query_ids: frozenset[str],
    final_predictions: tuple[PredictedInstance, ...],
    change_type_by_gt: Mapping[int, str],
    voxel_size_m: float,
    neural_supported_point_mask: np.ndarray | None = None,
    sensor_visible_gt_by_instance: Mapping[int, frozenset[Voxel]] | None = None,
    exact_union_candidate_limit: int = 12,
) -> tuple[EndpointDiagnosisRow, ...]:
    """Compare fixed-surface, atomic, union, split, raw, and final endpoints."""

    if not isinstance(pair_id, str) or not pair_id or not isinstance(visit, OviObjectVisitView):
        raise EndpointDiagnosisError("pair and visit identity is invalid")
    if type(exact_union_candidate_limit) is not int or exact_union_candidate_limit < 1:
        raise EndpointDiagnosisError("exact union candidate limit must be positive")
    if any(not isinstance(value, GroundTruthInstance) for value in ground_truth):
        raise EndpointDiagnosisError("ground truth contains an invalid instance")
    if any(not isinstance(value, RawDenseProposal) for value in raw_proposals):
        raise EndpointDiagnosisError("raw proposals contain an invalid value")
    if any(not isinstance(value, PredictedInstance) for value in final_predictions):
        raise EndpointDiagnosisError("final predictions contain an invalid value")
    if len({value.query_id for value in raw_proposals}) != len(raw_proposals):
        raise EndpointDiagnosisError("raw proposal query IDs must be unique")
    if any(np.any(value.point_indices >= visit.point_count) for value in raw_proposals):
        raise EndpointDiagnosisError("raw proposal point index exceeds the visit")
    if not confident_query_ids <= {value.query_id for value in raw_proposals}:
        raise EndpointDiagnosisError("confident queries must name raw proposals")
    supported = (
        visit.appearance_valid & (visit.entity_owner_indices >= 0)
        if neural_supported_point_mask is None
        else np.asarray(neural_supported_point_mask, dtype=np.bool_)
    )
    if supported.shape != (visit.point_count,):
        raise EndpointDiagnosisError("neural support mask must align with dense rows")

    dense_voxels = voxelize_points(visit.points_xyz, voxel_size_m=voxel_size_m)
    owned_voxels = voxelize_points(
        visit.points_xyz[visit.entity_owner_indices >= 0], voxel_size_m=voxel_size_m
    )
    supported_voxels = voxelize_points(
        visit.points_xyz[supported], voxel_size_m=voxel_size_m
    )
    atomic = tuple(
        (
            entity.entity_id,
            voxelize_points(
                visit.points_xyz[entity.point_indices], voxel_size_m=voxel_size_m
            ),
        )
        for entity in visit.entities
    )
    raw = tuple(
        (
            value.query_id,
            value.query_score,
            voxelize_points(
                visit.points_xyz[value.point_indices], voxel_size_m=voxel_size_m
            ),
        )
        for value in raw_proposals
    )
    final = tuple((value.prediction_id, value.voxels) for value in final_predictions)

    rows: list[EndpointDiagnosisRow] = []
    for target in ground_truth:
        target_voxels = target.voxels
        atomic_scores = sorted(
            (
                (
                    _iou(voxels, target_voxels),
                    entity_id,
                    *_precision_recall(voxels, target_voxels),
                )
                for entity_id, voxels in atomic
            ),
            key=lambda row: (-row[0], row[1]),
        )
        best = atomic_scores[0] if atomic_scores else None
        union_iou, union_exact, union_members = _best_union(
            atomic,
            target_voxels,
            exact_limit=exact_union_candidate_limit,
        )
        split_voxels = dense_voxels & target_voxels
        split_iou = _iou(split_voxels, target_voxels)
        raw_iou, raw_id, raw_score = _best_proposal(raw, target_voxels)
        confident_iou, _confident_id, _confident_score = _best_proposal(
            tuple(row for row in raw if row[0] in confident_query_ids),
            target_voxels,
        )
        final_iou, _final_id, _ = _best_proposal(
            tuple((prediction_id, 1.0, voxels) for prediction_id, voxels in final),
            target_voxels,
        )
        sensor_voxels = (
            None
            if sensor_visible_gt_by_instance is None
            else sensor_visible_gt_by_instance.get(target.instance_id, frozenset())
        )
        sensor_coverage = (
            None if sensor_voxels is None else len(sensor_voxels) / len(target_voxels)
        )
        d_coverage = len(dense_voxels & target_voxels) / len(target_voxels)
        rows.append(
            EndpointDiagnosisRow(
                pair_id=pair_id,
                visit_id=visit.visit_id,
                gt_instance_id=target.instance_id,
                change_type=change_type_by_gt.get(target.instance_id, "unchanged_or_unknown"),
                full_gt_voxels=len(target_voxels),
                sensor_visible_gt_voxels=None if sensor_voxels is None else len(sensor_voxels),
                sensor_visibility_status=("NOT_COMPUTED" if sensor_voxels is None else "PASS"),
                d_intersection_voxels=len(dense_voxels & target_voxels),
                d_coverage=d_coverage,
                d_owned_coverage=len(owned_voxels & target_voxels) / len(target_voxels),
                d_supported_coverage=len(supported_voxels & target_voxels) / len(target_voxels),
                best_atomic_id=None if best is None else best[1],
                best_atomic_iou=None if best is None else best[0],
                best_atomic_precision=None if best is None else best[2],
                best_atomic_recall=None if best is None else best[3],
                second_atomic_iou=(None if len(atomic_scores) < 2 else atomic_scores[1][0]),
                union_oracle_iou=union_iou,
                union_search_exact=union_exact,
                union_candidate_count=sum(bool(voxels & target_voxels) for _, voxels in atomic),
                union_member_ids=union_members,
                split_oracle_iou=split_iou,
                split_oracle_voxels=split_voxels,
                best_raw_mask_iou=raw_iou,
                best_raw_query_id=raw_id,
                best_raw_query_score=raw_score,
                best_confident_mask_iou=confident_iou,
                final_p2_iou=final_iou,
                provisional_failure_type=_failure_type(
                    d_coverage=d_coverage,
                    sensor_coverage=sensor_coverage,
                    single_iou=None if best is None else best[0],
                    union_iou=union_iou,
                    split_iou=split_iou,
                    raw_iou=raw_iou,
                    final_iou=final_iou,
                ),
            )
        )
    return tuple(rows)


def endpoint_diagnosis_rows(
    rows: tuple[EndpointDiagnosisRow, ...],
) -> tuple[Mapping[str, object], ...]:
    """Return stable CSV-ready rows sorted by pair, visit, and GT identity."""

    if any(not isinstance(value, EndpointDiagnosisRow) for value in rows):
        raise EndpointDiagnosisError("endpoint diagnosis contains an invalid row")
    return tuple(
        MappingProxyType(value.to_dict())
        for value in sorted(rows, key=lambda item: (item.pair_id, item.visit_id, item.gt_instance_id))
    )


__all__ = [
    "EndpointDiagnosisError",
    "EndpointDiagnosisRow",
    "RawDenseProposal",
    "diagnose_visit_endpoints",
    "endpoint_diagnosis_rows",
]
