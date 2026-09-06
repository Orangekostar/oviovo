"""Evaluator-only geometry binding and T=2 identity metrics for 3RScan."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.evaluation.object_pair_association import ObjectPairPrediction
from src.evaluation.ovi_pair_views import OviObjectPairView
from src.evaluation.rscan_gt_instances import (
    GroundTruthInstance,
    GroundTruthPair,
    MatchingPolicy,
    PredictedInstance,
    ThresholdGeometryResult,
    evaluate_instance_geometry,
    voxelize_points,
)
from src.evaluation.rscan_method_views import RScanMethodPairView
from src.oviv2.two_visit_contracts import PairRelation


class RScanAssociationMetricError(ValueError):
    """Raised when method relations cannot be evaluated without ambiguity."""


EndpointSupportDomain = Literal["full", "supported"]


@dataclass(frozen=True, slots=True)
class CandidateEndpointBinding:
    """Evaluator-only binding for one fixed method candidate at one IoU threshold."""

    candidate_id: str
    assigned_gt_instance_id: int | None
    assigned_iou: float | None
    best_gt_instance_id: int | None
    best_iou: float | None
    qualifying_gt_instance_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FixedEndpointBindings:
    """Method-output-independent candidate-to-GT endpoint bindings."""

    pair_id: str
    pair_content_sha256: str
    ground_truth_sha256: str
    voxel_size_m: float
    support_domain: EndpointSupportDomain
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
        for value, bindings in self.thresholds:
            if math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12):
                return bindings
        raise RScanAssociationMetricError("fixed endpoint threshold is unavailable")

    def content_sha256(self) -> str:
        payload = {
            "pair_id": self.pair_id,
            "pair_content_sha256": self.pair_content_sha256,
            "ground_truth_sha256": self.ground_truth_sha256,
            "voxel_size_m": self.voxel_size_m,
            "support_domain": self.support_domain,
            "candidate_ids": self.candidate_ids,
            "thresholds": [
                {
                    "threshold": threshold,
                    "visits": [
                        [
                            {
                                "candidate_id": value.candidate_id,
                                "assigned_gt_instance_id": value.assigned_gt_instance_id,
                                "assigned_iou": value.assigned_iou,
                                "best_gt_instance_id": value.best_gt_instance_id,
                                "best_iou": value.best_iou,
                                "qualifying_gt_instance_ids": value.qualifying_gt_instance_ids,
                            }
                            for value in visit
                        ]
                        for visit in bindings
                    ],
                }
                for threshold, bindings in self.thresholds
            ],
        }
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _ground_truth_sha256(ground_truth: GroundTruthPair) -> str:
    rules = ground_truth.identity_rules
    payload = {
        "pair_id": ground_truth.pair_id,
        "voxel_size_m": ground_truth.voxel_size_m,
        "visits": [
            [
                {
                    "instance_id": target.instance_id,
                    "semantic_label": target.semantic_label,
                    "voxels": sorted([list(voxel) for voxel in target.voxels]),
                }
                for target in sorted(values, key=lambda item: item.instance_id)
            ]
            for values in ground_truth.visits
        ],
        "identity_rules": {
            "rescan_to_reference": dict(rules.rescan_to_reference),
            "symmetry_by_reference": dict(rules.symmetry_by_reference),
            "rigid_transform_by_reference": {
                key: list(value)
                for key, value in rules.rigid_transform_by_reference.items()
            },
            "removed_reference_ids": sorted(rules.removed_reference_ids),
            "ambiguity_groups": sorted(
                [sorted(group) for group in rules.ambiguity_groups]
            ),
        },
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _candidate_iou(
    candidate_voxels: frozenset[tuple[int, int, int]],
    target: GroundTruthInstance,
) -> float:
    intersection = len(candidate_voxels & target.voxels)
    return intersection / len(candidate_voxels | target.voxels)


def _endpoint_visit_bindings(
    pair: OviObjectPairView,
    ground_truth: GroundTruthPair,
    *,
    visit_id: int,
    support_domain: EndpointSupportDomain,
    threshold: float,
) -> tuple[CandidateEndpointBinding, ...]:
    visit = pair.visits[visit_id]
    targets = ground_truth.visits[visit_id]
    rows: list[CandidateEndpointBinding] = []
    for entity in visit.entities:
        indices = entity.point_indices
        if support_domain == "supported":
            indices = indices[visit.appearance_valid[indices]]
        if not len(indices):
            rows.append(
                CandidateEndpointBinding(
                    candidate_id=entity.entity_id,
                    assigned_gt_instance_id=None,
                    assigned_iou=None,
                    best_gt_instance_id=None,
                    best_iou=None,
                    qualifying_gt_instance_ids=(),
                )
            )
            continue
        voxels = voxelize_points(
            visit.points_xyz[indices], voxel_size_m=ground_truth.voxel_size_m
        )
        scored = tuple(
            sorted(
                (
                    (_candidate_iou(voxels, target), target.instance_id)
                    for target in targets
                ),
                key=lambda item: (-item[0], item[1]),
            )
        )
        best_iou, best_id = scored[0] if scored else (None, None)
        qualifying = tuple(
            sorted(instance_id for iou, instance_id in scored if iou >= threshold)
        )
        assigned = None if best_iou is None or best_iou < threshold else best_id
        rows.append(
            CandidateEndpointBinding(
                candidate_id=entity.entity_id,
                assigned_gt_instance_id=assigned,
                assigned_iou=None if assigned is None else float(best_iou),
                best_gt_instance_id=best_id,
                best_iou=None if best_iou is None else float(best_iou),
                qualifying_gt_instance_ids=qualifying,
            )
        )
    return tuple(rows)


def build_fixed_endpoint_bindings(
    pair: OviObjectPairView,
    ground_truth: GroundTruthPair,
    *,
    support_domain: EndpointSupportDomain,
) -> FixedEndpointBindings:
    """Bind every OVI candidate to GT once, independently of association output."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be an OviObjectPairView")
    if not isinstance(ground_truth, GroundTruthPair):
        raise TypeError("ground_truth must be a GroundTruthPair")
    if pair.pair_id != ground_truth.pair_id:
        raise RScanAssociationMetricError("OVI pair and GT pair IDs differ")
    if support_domain not in {"full", "supported"}:
        raise RScanAssociationMetricError("endpoint support domain is invalid")
    before = pair.content_sha256()
    thresholds = tuple(
        (
            threshold,
            (
                _endpoint_visit_bindings(
                    pair,
                    ground_truth,
                    visit_id=0,
                    support_domain=support_domain,
                    threshold=threshold,
                ),
                _endpoint_visit_bindings(
                    pair,
                    ground_truth,
                    visit_id=1,
                    support_domain=support_domain,
                    threshold=threshold,
                ),
            ),
        )
        for threshold in (0.50, 0.25)
    )
    if pair.content_sha256() != before:
        raise RScanAssociationMetricError("endpoint binding mutated the OVI pair")
    return FixedEndpointBindings(
        pair_id=pair.pair_id,
        pair_content_sha256=before,
        ground_truth_sha256=_ground_truth_sha256(ground_truth),
        voxel_size_m=ground_truth.voxel_size_m,
        support_domain=support_domain,
        candidate_ids=tuple(
            tuple(candidate_id for _visit_id, candidate_id in values)
            for values in pair.candidate_ids
        ),  # type: ignore[arg-type]
        thresholds=thresholds,
    )


def _geometry_dict(value: ThresholdGeometryResult) -> dict[str, object]:
    return {
        "threshold": value.threshold,
        "matched_count": value.matched_count,
        "unmatched_prediction_count": len(value.unmatched_prediction_ids),
        "unmatched_gt_count": len(value.unmatched_gt_instance_ids),
        "duplicate_prediction_count": value.duplicate_prediction_count,
        "fragment_gt_count": value.fragment_gt_count,
        "merge_prediction_count": value.merge_prediction_count,
        "mean_matched_iou": value.mean_matched_iou,
    }


def _prediction_sides(
    pair: RScanMethodPairView,
    relations: tuple[PairRelation, ...],
    *,
    voxel_size_m: float,
) -> tuple[tuple[PredictedInstance, ...], tuple[PredictedInstance, ...]]:
    candidate_points = tuple(
        {
            candidate_id: visit.points_xyz[visit.segment_ids == segment_id]
            for segment_id, candidate_id in zip(
                sorted({int(value) for value in visit.segment_ids}),
                visit.candidate_ids,
                strict=True,
            )
        }
        for visit in pair.visits
    )
    predictions: list[list[PredictedInstance]] = [[], []]
    for relation in relations:
        for visit_id, entity_ids in enumerate(
            (relation.t0_entity_ids, relation.t1_entity_ids)
        ):
            if not entity_ids:
                continue
            unknown = set(entity_ids).difference(candidate_points[visit_id])
            if unknown:
                raise RScanAssociationMetricError(
                    f"relation references unknown visit-{visit_id} candidates"
                )
            points = [candidate_points[visit_id][entity_id] for entity_id in entity_ids]
            predictions[visit_id].append(
                PredictedInstance(
                    prediction_id=str(relation.temporal_query_id),
                    voxels=voxelize_points(
                        np.concatenate(points, axis=0),
                        voxel_size_m=voxel_size_m,
                    ),
                )
            )
    return tuple(predictions[0]), tuple(predictions[1])


def _observed_gt_ids(
    pair: RScanMethodPairView,
    ground_truth: GroundTruthPair,
    *,
    minimum_observed_fraction: float,
) -> tuple[frozenset[int], frozenset[int]]:
    observed: list[frozenset[int]] = []
    for visit, targets in zip(pair.visits, ground_truth.visits, strict=True):
        input_voxels = voxelize_points(
            visit.points_xyz, voxel_size_m=ground_truth.voxel_size_m
        )
        observed.append(
            frozenset(
                target.instance_id
                for target in targets
                if len(target.voxels & input_voxels) / len(target.voxels)
                >= minimum_observed_fraction
            )
        )
    return observed[0], observed[1]


def _persistent_targets(
    ground_truth: GroundTruthPair,
) -> dict[int, tuple[int, frozenset[int]]]:
    left_ids = {target.instance_id for target in ground_truth.visits[0]}
    targets: dict[int, tuple[int, frozenset[int]]] = {}
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
            targets[target.instance_id] = (reference_id, allowed)
    return targets


def _threshold_metrics(
    *,
    relations: tuple[PairRelation, ...],
    ground_truth: GroundTruthPair,
    observed_ids: tuple[frozenset[int], frozenset[int]],
    persistent_targets: Mapping[int, tuple[int, frozenset[int]]],
    first: ThresholdGeometryResult,
    second: ThresholdGeometryResult,
) -> dict[str, object]:
    first_matches = {
        value.prediction_id: value.gt_instance_id for value in first.matches
    }
    second_matches = {
        value.prediction_id: value.gt_instance_id for value in second.matches
    }
    first_labels = {
        value.instance_id: value.semantic_label.casefold()
        for value in ground_truth.visits[0]
    }
    second_labels = {
        value.instance_id: value.semantic_label.casefold()
        for value in ground_truth.visits[1]
    }
    paired = tuple(
        sorted(
            (
                relation
                for relation in relations
                if relation.t0_entity_ids and relation.t1_entity_ids
            ),
            key=lambda value: (-value.query_confidence, str(value.temporal_query_id)),
        )
    )
    strict_hits: set[int] = set()
    aware_hits: set[int] = set()
    strict_duplicate_edges = 0
    aware_duplicate_edges = 0
    false_reid = 0
    same_class_mismatch = 0
    unmatched_endpoint_edges = 0
    for relation in paired:
        query_id = str(relation.temporal_query_id)
        left_id = first_matches.get(query_id)
        right_id = second_matches.get(query_id)
        if left_id is None or right_id is None:
            unmatched_endpoint_edges += 1
            continue
        target = persistent_targets.get(right_id)
        strict_correct = target is not None and left_id == target[0]
        aware_correct = target is not None and left_id in target[1]
        if strict_correct:
            if right_id in strict_hits:
                strict_duplicate_edges += 1
            else:
                strict_hits.add(right_id)
        if aware_correct:
            if right_id in aware_hits:
                aware_duplicate_edges += 1
            else:
                aware_hits.add(right_id)
        else:
            false_reid += 1
            if first_labels[left_id] == second_labels[right_id]:
                same_class_mismatch += 1

    strict_target_ids = {
        right_id
        for right_id, (reference_id, _allowed) in persistent_targets.items()
        if reference_id in first_labels
    }
    aware_target_ids = set(persistent_targets)
    conditional_targets = {
        right_id
        for right_id, (_reference_id, allowed) in persistent_targets.items()
        if right_id in observed_ids[1] and bool(allowed & observed_ids[0])
    }
    rigid_targets = {
        right_id
        for right_id, (reference_id, _allowed) in persistent_targets.items()
        if reference_id in ground_truth.identity_rules.rigid_transform_by_reference
    }
    return {
        "paired_prediction_edges": len(paired),
        "true_positive_edges": len(aware_hits),
        "false_positive_edges": len(paired) - len(aware_hits),
        "duplicate_edge_count": aware_duplicate_edges,
        "false_reid_count": false_reid,
        "same_class_mismatch_count": same_class_mismatch,
        "unmatched_endpoint_edge_count": unmatched_endpoint_edges,
        "paired_precision": _ratio(len(aware_hits), len(paired)),
        "paired_recall": _ratio(len(aware_hits), len(aware_target_ids)),
        "end_to_end_persistence_recall": _ratio(len(aware_hits), len(aware_target_ids)),
        "conditional_association_recall": _ratio(
            len(aware_hits & conditional_targets), len(conditional_targets)
        ),
        "rigid_recall": _ratio(len(aware_hits & rigid_targets), len(rigid_targets)),
        "conditional_true_positive_edges": len(aware_hits & conditional_targets),
        "rigid_true_positive_edges": len(aware_hits & rigid_targets),
        "strict_true_positive_edges": len(strict_hits),
        "strict_false_positive_edges": len(paired) - len(strict_hits),
        "strict_duplicate_edge_count": strict_duplicate_edges,
        "strict_paired_precision": _ratio(len(strict_hits), len(paired)),
        "strict_paired_recall": _ratio(len(strict_hits), len(strict_target_ids)),
        "ground_truth_persistent_edges": len(aware_target_ids),
        "strict_ground_truth_persistent_edges": len(strict_target_ids),
        "conditional_ground_truth_edges": len(conditional_targets),
        "rigid_ground_truth_edges": len(rigid_targets),
        "geometry": {
            "t0": _geometry_dict(first),
            "t1": _geometry_dict(second),
            "duplicate_prediction_count": (
                first.duplicate_prediction_count + second.duplicate_prediction_count
            ),
            "fragment_gt_count": first.fragment_gt_count + second.fragment_gt_count,
            "merge_prediction_count": (
                first.merge_prediction_count + second.merge_prediction_count
            ),
        },
    }


def evaluate_pair_relations(
    pair: RScanMethodPairView,
    relations: Sequence[PairRelation],
    ground_truth: GroundTruthPair,
    *,
    minimum_observed_fraction: float = 0.05,
    matching_policy: MatchingPolicy = "max_total_iou",
) -> dict[str, object]:
    """Bind predicted relation sides to GT before scoring cross-visit identity."""

    if not isinstance(pair, RScanMethodPairView):
        raise TypeError("pair must be an RScanMethodPairView")
    if not isinstance(ground_truth, GroundTruthPair):
        raise TypeError("ground_truth must be a GroundTruthPair")
    if pair.pair_id != ground_truth.pair_id:
        raise RScanAssociationMetricError("method pair and GT pair IDs differ")
    if (
        isinstance(minimum_observed_fraction, bool)
        or not isinstance(minimum_observed_fraction, (int, float))
        or not math.isfinite(float(minimum_observed_fraction))
        or not 0.0 < float(minimum_observed_fraction) <= 1.0
    ):
        raise RScanAssociationMetricError("minimum observed fraction must be in (0, 1]")
    rows = tuple(relations)
    if any(not isinstance(value, PairRelation) for value in rows):
        raise RScanAssociationMetricError("relations contain an invalid value")
    query_ids = [str(value.temporal_query_id) for value in rows]
    if len(query_ids) != len(set(query_ids)):
        raise RScanAssociationMetricError("relation query IDs must be unique")
    predicted = _prediction_sides(pair, rows, voxel_size_m=ground_truth.voxel_size_m)
    geometry = tuple(
        evaluate_instance_geometry(
            values,
            targets,
            matching_policy=matching_policy,
        )
        for values, targets in zip(predicted, ground_truth.visits, strict=True)
    )
    observed = _observed_gt_ids(
        pair,
        ground_truth,
        minimum_observed_fraction=float(minimum_observed_fraction),
    )
    persistent = _persistent_targets(ground_truth)
    thresholds = {
        "iou_0_50": _threshold_metrics(
            relations=rows,
            ground_truth=ground_truth,
            observed_ids=observed,
            persistent_targets=persistent,
            first=geometry[0].primary,
            second=geometry[1].primary,
        ),
        "iou_0_25": _threshold_metrics(
            relations=rows,
            ground_truth=ground_truth,
            observed_ids=observed,
            persistent_targets=persistent,
            first=geometry[0].sensitivity,
            second=geometry[1].sensitivity,
        ),
    }
    return {
        "schema_version": 1,
        "protocol_id": (
            "RSCAN_T2_ASSOCIATION_V1"
            if matching_policy == "max_total_iou"
            else "RSCAN_T2_ASSOCIATION_V2_MAX_VALID_CARDINALITY"
        ),
        "status": "PASS",
        "pair_id": pair.pair_id,
        "method_input_sha256": pair.method_tensor_sha256(),
        "voxel_size_m": ground_truth.voxel_size_m,
        "minimum_observed_fraction": float(minimum_observed_fraction),
        "relation_count": len(rows),
        "paired_relation_count": sum(
            bool(value.t0_entity_ids and value.t1_entity_ids) for value in rows
        ),
        "one_sided_relation_count": sum(
            bool(value.t0_entity_ids) != bool(value.t1_entity_ids) for value in rows
        ),
        "input_gt_observed": {
            "t0_count": len(observed[0]),
            "t0_total": len(ground_truth.visits[0]),
            "t1_count": len(observed[1]),
            "t1_total": len(ground_truth.visits[1]),
        },
        "reactivation_recall": None,
        "reactivation_status": "NOT_APPLICABLE_T2",
        "thresholds": thresholds,
    }


def _best_endpoint_rows(
    predictions: tuple[PredictedInstance, ...],
    ground_truth: tuple[GroundTruthInstance, ...],
) -> tuple[dict[str, tuple[int | None, float | None]], dict[int, int]]:
    best: dict[str, tuple[int | None, float | None]] = {}
    counts: dict[int, int] = {}
    for prediction in predictions:
        candidates = []
        for target in ground_truth:
            intersection = len(prediction.voxels & target.voxels)
            union = len(prediction.voxels | target.voxels)
            candidates.append((intersection / union, target.instance_id))
        if not candidates:
            best[prediction.prediction_id] = (None, None)
            continue
        best_iou, best_id = max(candidates, key=lambda value: (value[0], -value[1]))
        best[prediction.prediction_id] = (best_id, best_iou)
        counts[best_id] = counts.get(best_id, 0) + 1
    return best, counts


def relation_outcome_rows(
    pair: RScanMethodPairView,
    relations: Sequence[PairRelation],
    ground_truth: GroundTruthPair,
    *,
    iou_threshold: float = 0.50,
    matching_policy: MatchingPolicy = "max_total_iou",
) -> tuple[dict[str, object], ...]:
    """Expose per-relation endpoint assignment and identity outcomes."""

    if not isinstance(pair, RScanMethodPairView):
        raise TypeError("pair must be an RScanMethodPairView")
    if not isinstance(ground_truth, GroundTruthPair):
        raise TypeError("ground_truth must be a GroundTruthPair")
    if pair.pair_id != ground_truth.pair_id:
        raise RScanAssociationMetricError("method pair and GT pair IDs differ")
    if iou_threshold not in {0.25, 0.50}:
        raise RScanAssociationMetricError(
            "diagnostic IoU threshold must be 0.25 or 0.50"
        )
    rows = tuple(relations)
    if any(not isinstance(value, PairRelation) for value in rows):
        raise RScanAssociationMetricError("relations contain an invalid value")
    query_ids = [str(value.temporal_query_id) for value in rows]
    if len(query_ids) != len(set(query_ids)):
        raise RScanAssociationMetricError("relation query IDs must be unique")

    predicted = _prediction_sides(
        pair,
        rows,
        voxel_size_m=ground_truth.voxel_size_m,
    )
    geometry = tuple(
        evaluate_instance_geometry(
            values,
            targets,
            matching_policy=matching_policy,
        )
        for values, targets in zip(predicted, ground_truth.visits, strict=True)
    )
    selected = tuple(
        value.primary if iou_threshold == 0.50 else value.sensitivity
        for value in geometry
    )
    matches = tuple(
        {
            match.prediction_id: (match.gt_instance_id, match.iou)
            for match in result.matches
        }
        for result in selected
    )
    best_rows = tuple(
        _best_endpoint_rows(values, targets)
        for values, targets in zip(predicted, ground_truth.visits, strict=True)
    )
    persistent = _persistent_targets(ground_truth)
    labels = tuple(
        {target.instance_id: target.semantic_label.casefold() for target in targets}
        for targets in ground_truth.visits
    )
    aware_hits: set[int] = set()
    result_rows: list[dict[str, object]] = []
    for relation in sorted(
        rows,
        key=lambda value: (-value.query_confidence, str(value.temporal_query_id)),
    ):
        query_id = str(relation.temporal_query_id)
        left = matches[0].get(query_id)
        right = matches[1].get(query_id)
        left_id = None if left is None else left[0]
        right_id = None if right is None else right[0]
        target = None if right_id is None else persistent.get(right_id)
        strict_correct = bool(
            target is not None and left_id is not None and left_id == target[0]
        )
        aware_correct = bool(
            target is not None and left_id is not None and left_id in target[1]
        )
        same_class = bool(
            left_id is not None
            and right_id is not None
            and labels[0][left_id] == labels[1][right_id]
        )
        if not relation.t0_entity_ids or not relation.t1_entity_ids:
            outcome = "one_sided"
        elif left is None or right is None:
            outcome = "endpoint_unmatched"
        elif not aware_correct:
            outcome = "false_reid"
        elif right_id in aware_hits:
            outcome = "duplicate"
        else:
            outcome = "true_positive"
            aware_hits.add(right_id)
        left_best = best_rows[0][0].get(query_id, (None, None))
        right_best = best_rows[1][0].get(query_id, (None, None))
        result_rows.append(
            {
                "query_id": query_id,
                "query_confidence": relation.query_confidence,
                "t0_entity_ids": list(relation.t0_entity_ids),
                "t1_entity_ids": list(relation.t1_entity_ids),
                "t0_assigned_gt_id": left_id,
                "t0_assigned_iou": None if left is None else left[1],
                "t1_assigned_gt_id": right_id,
                "t1_assigned_iou": None if right is None else right[1],
                "t0_best_gt_id": left_best[0],
                "t0_best_iou": left_best[1],
                "t1_best_gt_id": right_best[0],
                "t1_best_iou": right_best[1],
                "t0_best_competitor_count": (
                    0 if left_best[0] is None else best_rows[0][1][left_best[0]]
                ),
                "t1_best_competitor_count": (
                    0 if right_best[0] is None else best_rows[1][1][right_best[0]]
                ),
                "strict_correct": strict_correct,
                "ambiguity_aware_correct": aware_correct,
                "same_class_mismatch": same_class and not aware_correct,
                "outcome": outcome,
            }
        )
    return tuple(sorted(result_rows, key=lambda value: str(value["query_id"])))


def evaluate_fixed_object_predictions(
    pair: OviObjectPairView,
    predictions: Sequence[ObjectPairPrediction],
    ground_truth: GroundTruthPair,
    bindings: FixedEndpointBindings,
    *,
    iou_threshold: float = 0.50,
) -> dict[str, object]:
    """Score 1:1 object relations without recomputing their endpoint bindings."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be an OviObjectPairView")
    if not isinstance(ground_truth, GroundTruthPair):
        raise TypeError("ground_truth must be a GroundTruthPair")
    if not isinstance(bindings, FixedEndpointBindings):
        raise TypeError("bindings must be FixedEndpointBindings")
    if iou_threshold not in {0.25, 0.50}:
        raise RScanAssociationMetricError("fixed endpoint IoU must be 0.25 or 0.50")
    pair_hash = pair.content_sha256()
    candidate_ids = tuple(
        tuple(candidate_id for _visit_id, candidate_id in values)
        for values in pair.candidate_ids
    )
    if (
        bindings.pair_id != pair.pair_id
        or bindings.pair_content_sha256 != pair_hash
        or bindings.ground_truth_sha256 != _ground_truth_sha256(ground_truth)
        or bindings.voxel_size_m != ground_truth.voxel_size_m
        or bindings.candidate_ids != candidate_ids
    ):
        raise RScanAssociationMetricError("fixed endpoint identity mismatch")
    rows = tuple(predictions)
    if not rows or any(not isinstance(value, ObjectPairPrediction) for value in rows):
        raise RScanAssociationMetricError("object predictions are empty or invalid")
    if any(value.pair_id != pair.pair_id for value in rows):
        raise RScanAssociationMetricError("object prediction pair ID mismatch")
    method_ids = {value.method_id for value in rows}
    if len(method_ids) != 1:
        raise RScanAssociationMetricError("object predictions mix methods")
    observed_candidates = (
        [value.t0_entity_id for value in rows if value.t0_entity_id is not None],
        [value.t1_entity_id for value in rows if value.t1_entity_id is not None],
    )
    for visit_id in (0, 1):
        if len(observed_candidates[visit_id]) != len(
            set(observed_candidates[visit_id])
        ) or set(observed_candidates[visit_id]) != set(candidate_ids[visit_id]):
            raise RScanAssociationMetricError(
                "object predictions do not conserve the candidate pool"
            )

    selected = bindings.for_threshold(iou_threshold)
    assigned = tuple(
        {value.candidate_id: value.assigned_gt_instance_id for value in visit}
        for visit in selected
    )
    assigned_gt = tuple(
        {value for value in visit.values() if value is not None} for visit in assigned
    )
    persistent = _persistent_targets(ground_truth)
    representation_targets = {
        right_id
        for right_id, (_reference_id, allowed) in persistent.items()
        if right_id in assigned_gt[1] and bool(allowed & assigned_gt[0])
    }
    strict_representation_targets = {
        right_id
        for right_id, (reference_id, _allowed) in persistent.items()
        if right_id in assigned_gt[1] and reference_id in assigned_gt[0]
    }
    rigid_targets = {
        right_id
        for right_id, (reference_id, _allowed) in persistent.items()
        if reference_id in ground_truth.identity_rules.rigid_transform_by_reference
    }
    labels = tuple(
        {
            value.instance_id: value.semantic_label.casefold()
            for value in ground_truth.visits[visit_id]
        }
        for visit_id in (0, 1)
    )
    paired = sorted(
        (value for value in rows if value.is_matched),
        key=lambda value: (-float(value.score), value.prediction_id),
    )
    aware_hits: set[int] = set()
    strict_hits: set[int] = set()
    endpoint_failures = 0
    false_reids = 0
    same_class_mismatches = 0
    duplicates = 0
    strict_duplicates = 0
    outcomes: list[dict[str, object]] = []
    for prediction in paired:
        assert prediction.t0_entity_id is not None
        assert prediction.t1_entity_id is not None
        left_id = assigned[0][prediction.t0_entity_id]
        right_id = assigned[1][prediction.t1_entity_id]
        target = None if right_id is None else persistent.get(right_id)
        strict_correct = bool(
            target is not None and left_id is not None and left_id == target[0]
        )
        aware_correct = bool(
            target is not None and left_id is not None and left_id in target[1]
        )
        if left_id is None or right_id is None:
            outcome = "endpoint_failure"
            endpoint_failures += 1
        elif not aware_correct:
            outcome = "false_reid"
            false_reids += 1
            if labels[0][left_id] == labels[1][right_id]:
                same_class_mismatches += 1
        elif right_id in aware_hits:
            outcome = "duplicate"
            duplicates += 1
        else:
            outcome = "true_positive"
            aware_hits.add(right_id)
        if strict_correct:
            if right_id in strict_hits:
                strict_duplicates += 1
            else:
                assert right_id is not None
                strict_hits.add(right_id)
        outcomes.append(
            {
                "prediction_id": prediction.prediction_id,
                "score": prediction.score,
                "t0_entity_id": prediction.t0_entity_id,
                "t1_entity_id": prediction.t1_entity_id,
                "t0_gt_instance_id": left_id,
                "t1_gt_instance_id": right_id,
                "strict_correct": strict_correct,
                "ambiguity_aware_correct": aware_correct,
                "outcome": outcome,
            }
        )

    unmatched = [value for value in rows if not value.is_matched]
    for prediction in sorted(unmatched, key=lambda value: value.prediction_id):
        outcomes.append(
            {
                "prediction_id": prediction.prediction_id,
                "score": None,
                "t0_entity_id": prediction.t0_entity_id,
                "t1_entity_id": prediction.t1_entity_id,
                "t0_gt_instance_id": (
                    None
                    if prediction.t0_entity_id is None
                    else assigned[0][prediction.t0_entity_id]
                ),
                "t1_gt_instance_id": (
                    None
                    if prediction.t1_entity_id is None
                    else assigned[1][prediction.t1_entity_id]
                ),
                "strict_correct": False,
                "ambiguity_aware_correct": False,
                "outcome": "candidate_unmatched",
            }
        )

    persistent_count = len(persistent)
    conditional_count = len(representation_targets)
    strict_persistent_targets = {
        right_id
        for right_id, (reference_id, _allowed) in persistent.items()
        if any(target.instance_id == reference_id for target in ground_truth.visits[0])
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": "RSCAN_T2_FIXED_OBJECT_ASSOCIATION_V1",
        "status": "PASS",
        "pair_id": pair.pair_id,
        "method_id": next(iter(method_ids)),
        "pair_content_sha256": pair_hash,
        "binding_sha256": bindings.content_sha256(),
        "support_domain": bindings.support_domain,
        "iou_threshold": iou_threshold,
        "paired_prediction_count": len(paired),
        "candidate_unmatched_count": len(unmatched),
        "true_positive_count": len(aware_hits),
        "false_positive_count": len(paired) - len(aware_hits),
        "endpoint_failure_count": endpoint_failures,
        "false_reid_count": false_reids,
        "same_class_mismatch_count": same_class_mismatches,
        "duplicate_count": duplicates,
        "precision": _ratio(len(aware_hits), len(paired)),
        "end_to_end_recall": _ratio(len(aware_hits), persistent_count),
        "representation_conditional_recall": _ratio(
            len(aware_hits & representation_targets), conditional_count
        ),
        "persistent_gt_count": persistent_count,
        "representation_conditional_gt_count": conditional_count,
        "strict_true_positive_count": len(strict_hits),
        "strict_false_positive_count": len(paired) - len(strict_hits),
        "strict_duplicate_count": strict_duplicates,
        "strict_precision": _ratio(len(strict_hits), len(paired)),
        "strict_end_to_end_recall": _ratio(
            len(strict_hits), len(strict_persistent_targets)
        ),
        "strict_representation_conditional_recall": _ratio(
            len(strict_hits & strict_representation_targets),
            len(strict_representation_targets),
        ),
        "rigid_true_positive_count": len(aware_hits & rigid_targets),
        "rigid_gt_count": len(rigid_targets),
        "rigid_recall": _ratio(len(aware_hits & rigid_targets), len(rigid_targets)),
        "outcomes": outcomes,
    }
    if bindings.support_domain == "supported":
        result.update(
            {
                "support_conditional_true_positive_count": len(
                    aware_hits & representation_targets
                ),
                "support_conditional_gt_count": conditional_count,
                "support_conditional_recall": _ratio(
                    len(aware_hits & representation_targets), conditional_count
                ),
                "support_conditional_status": "PASS",
            }
        )
    else:
        result.update(
            {
                "support_conditional_true_positive_count": None,
                "support_conditional_gt_count": None,
                "support_conditional_recall": None,
                "support_conditional_status": "NOT_APPLICABLE_FULL_DOMAIN",
            }
        )
    if pair.content_sha256() != pair_hash:
        raise RScanAssociationMetricError("fixed object evaluation mutated the pair")
    return result


__all__ = [
    "CandidateEndpointBinding",
    "FixedEndpointBindings",
    "RScanAssociationMetricError",
    "build_fixed_endpoint_bindings",
    "evaluate_fixed_object_predictions",
    "evaluate_pair_relations",
    "relation_outcome_rows",
]
