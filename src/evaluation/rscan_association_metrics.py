"""Evaluator-only geometry binding and T=2 identity metrics for 3RScan."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from src.evaluation.rscan_gt_instances import (
    GroundTruthPair,
    PredictedInstance,
    ThresholdGeometryResult,
    evaluate_instance_geometry,
    voxelize_points,
)
from src.evaluation.rscan_method_views import RScanMethodPairView
from src.oviv2.two_visit_contracts import PairRelation


class RScanAssociationMetricError(ValueError):
    """Raised when method relations cannot be evaluated without ambiguity."""


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


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
    first_matches = {value.prediction_id: value.gt_instance_id for value in first.matches}
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
        "end_to_end_persistence_recall": _ratio(
            len(aware_hits), len(aware_target_ids)
        ),
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
        raise RScanAssociationMetricError(
            "minimum observed fraction must be in (0, 1]"
        )
    rows = tuple(relations)
    if any(not isinstance(value, PairRelation) for value in rows):
        raise RScanAssociationMetricError("relations contain an invalid value")
    query_ids = [str(value.temporal_query_id) for value in rows]
    if len(query_ids) != len(set(query_ids)):
        raise RScanAssociationMetricError("relation query IDs must be unique")
    predicted = _prediction_sides(
        pair, rows, voxel_size_m=ground_truth.voxel_size_m
    )
    geometry = tuple(
        evaluate_instance_geometry(values, targets)
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
        "protocol_id": "RSCAN_T2_ASSOCIATION_V1",
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


__all__ = [
    "RScanAssociationMetricError",
    "evaluate_pair_relations",
]
