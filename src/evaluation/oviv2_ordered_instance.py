from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from src.evaluation.oviv2_instance_head import (
    ProjectedInstanceHypothesis,
    _POPCOUNT_UINT8,
    deduplicate_projected_hypotheses,
)
from src.evaluation.oviv2_replica import (
    ReplicaGroundTruth,
    _average_precision,
    _ground_truth_instance_masks,
)


def _positive_integer(value: Integral, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _iou_threshold(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if not 0.0 < normalized <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")
    return normalized


def _validated_predictions(
    values: Iterable[ProjectedInstanceHypothesis],
    name: str,
) -> tuple[ProjectedInstanceHypothesis, ...]:
    try:
        predictions = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be iterable") from error
    expected_length: int | None = None
    identifiers: set[str] = set()
    for item in predictions:
        if not isinstance(item, ProjectedInstanceHypothesis):
            raise TypeError(f"{name} must contain ProjectedInstanceHypothesis values")
        if expected_length is None:
            expected_length = len(item.mask)
        elif len(item.mask) != expected_length:
            raise ValueError("projected masks must have equal length")
        if item.hypothesis_id in identifiers:
            raise ValueError(f"{name} contains duplicate hypothesis IDs")
        identifiers.add(item.hypothesis_id)
    return predictions


def _validate_equal_mask_lengths(
    primary: Sequence[ProjectedInstanceHypothesis],
    suffix: Sequence[ProjectedInstanceHypothesis],
) -> None:
    masks = (*primary, *suffix)
    if masks and any(len(item.mask) != len(masks[0].mask) for item in masks[1:]):
        raise ValueError("projected masks must have equal length")


def _validate_unique_ids(
    primary: Sequence[ProjectedInstanceHypothesis],
    suffix: Sequence[ProjectedInstanceHypothesis],
) -> None:
    identifiers = [item.hypothesis_id for item in (*primary, *suffix)]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("primary and suffix must not share hypothesis IDs")


def _pack_mask(item: ProjectedInstanceHypothesis) -> tuple[np.ndarray, int]:
    return np.packbits(item.mask, bitorder="big"), int(np.sum(item.mask))


def _packed_iou(
    left_packed: np.ndarray,
    left_size: int,
    right_packed: np.ndarray,
    right_size: int,
) -> float:
    intersection = int(
        _POPCOUNT_UINT8[np.bitwise_and(left_packed, right_packed)].sum(
            dtype=np.int64
        )
    )
    union = left_size + right_size - intersection
    return float(intersection / union) if union else 1.0


def _semantic_id_set(values: set[int]) -> set[int]:
    if not isinstance(values, set):
        raise TypeError("instance_semantic_ids must be a set")
    normalized: set[int] = set()
    for value in values:
        normalized.add(_positive_integer(value, "instance_semantic_ids value"))
    return normalized


@dataclass(frozen=True)
class OrderedPredictionComposition:
    primary: tuple[ProjectedInstanceHypothesis, ...]
    suffix: tuple[ProjectedInstanceHypothesis, ...]
    rejected_suffix_ids: tuple[str, ...]
    primary_sha256: str

    @property
    def ordered(self) -> tuple[ProjectedInstanceHypothesis, ...]:
        return self.primary + self.suffix

    @property
    def primary_ids(self) -> tuple[str, ...]:
        return tuple(item.hypothesis_id for item in self.primary)

    @property
    def ordered_ids(self) -> tuple[str, ...]:
        return tuple(item.hypothesis_id for item in self.ordered)


def projected_iou(
    left: ProjectedInstanceHypothesis,
    right: ProjectedInstanceHypothesis,
) -> float:
    if not isinstance(left, ProjectedInstanceHypothesis):
        raise TypeError("left must be a ProjectedInstanceHypothesis")
    if not isinstance(right, ProjectedInstanceHypothesis):
        raise TypeError("right must be a ProjectedInstanceHypothesis")
    if len(left.mask) != len(right.mask):
        raise ValueError("projected masks must have equal length")
    intersection = int(np.sum(left.mask & right.mask))
    union = int(np.sum(left.mask | right.mask))
    return float(intersection / union) if union else 1.0


def fingerprint_predictions(
    predictions: Iterable[ProjectedInstanceHypothesis],
) -> str:
    values = _validated_predictions(predictions, "predictions")
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(values)))
    for item in values:
        identifier = item.hypothesis_id.encode("utf-8")
        packed_mask = np.packbits(item.mask, bitorder="big").tobytes()
        digest.update(struct.pack("<Q", len(identifier)))
        digest.update(identifier)
        digest.update(np.asarray([item.score], dtype="<f8").tobytes())
        digest.update(struct.pack("<Q", len(item.mask)))
        digest.update(struct.pack("<Q", len(packed_mask)))
        digest.update(packed_mask)
    return digest.hexdigest()


def compose_ordered_predictions(
    primary: Iterable[ProjectedInstanceHypothesis],
    suffix: Iterable[ProjectedInstanceHypothesis],
    *,
    min_instance_vertices: int,
    primary_deduplication_iou: float,
    suffix_deduplication_iou: float,
) -> OrderedPredictionComposition:
    minimum = _positive_integer(min_instance_vertices, "min_instance_vertices")
    primary_threshold = _iou_threshold(
        primary_deduplication_iou, "primary_deduplication_iou"
    )
    suffix_threshold = _iou_threshold(
        suffix_deduplication_iou, "suffix_deduplication_iou"
    )
    primary_values = _validated_predictions(primary, "primary")
    suffix_values = _validated_predictions(suffix, "suffix")
    _validate_equal_mask_lengths(primary_values, suffix_values)
    _validate_unique_ids(primary_values, suffix_values)
    packed_masks = {
        item.hypothesis_id: _pack_mask(item)
        for item in (*primary_values, *suffix_values)
    }

    supported_primary = tuple(
        item for item in primary_values if packed_masks[item.hypothesis_id][1] >= minimum
    )
    accepted_primary = deduplicate_projected_hypotheses(
        supported_primary, primary_threshold
    )
    if not accepted_primary:
        raise ValueError("primary must contain a supported, deduplicated hypothesis")
    minimum_primary_score = min(item.score for item in accepted_primary)
    if minimum_primary_score <= 0.0:
        raise ValueError("minimum primary score must be positive")

    supported_suffix = [
        item for item in suffix_values if packed_masks[item.hypothesis_id][1] >= minimum
    ]
    kept_suffix: list[ProjectedInstanceHypothesis] = []
    screened_masks = [
        packed_masks[item.hypothesis_id] for item in accepted_primary
    ]
    rejected_suffix_ids: list[str] = []
    for candidate in sorted(
        supported_suffix, key=lambda item: (-item.score, item.hypothesis_id)
    ):
        candidate_packed, candidate_size = packed_masks[candidate.hypothesis_id]
        if any(
            _packed_iou(candidate_packed, candidate_size, accepted_packed, accepted_size)
            >= suffix_threshold
            for accepted_packed, accepted_size in screened_masks
        ):
            rejected_suffix_ids.append(candidate.hypothesis_id)
        else:
            kept_suffix.append(candidate)
            screened_masks.append((candidate_packed, candidate_size))

    if kept_suffix:
        maximum_suffix_score = max(item.score for item in kept_suffix)
        if maximum_suffix_score <= 0.0:
            raise ValueError("kept suffix scores must include a positive value")
        ceiling = float(np.nextafter(minimum_primary_score, 0.0))
        scaled_suffix = tuple(
            ProjectedInstanceHypothesis(
                item.hypothesis_id,
                item.entity_id,
                item.semantic_id,
                item.kind,
                min(ceiling * (item.score / maximum_suffix_score), ceiling),
                item.mask,
            )
            for item in kept_suffix
        )
        if any(
            not np.isfinite(item.score)
            or item.score < 0.0
            or item.score >= minimum_primary_score
            for item in scaled_suffix
        ):
            raise ValueError("suffix score scaling failed strict separation")
    else:
        scaled_suffix = ()

    return OrderedPredictionComposition(
        primary=accepted_primary,
        suffix=scaled_suffix,
        rejected_suffix_ids=tuple(rejected_suffix_ids),
        primary_sha256=fingerprint_predictions(accepted_primary),
    )


def evaluate_ordered_projected_hypotheses(
    ordered: Sequence[ProjectedInstanceHypothesis],
    ground_truth: ReplicaGroundTruth,
    *,
    instance_semantic_ids: set[int],
    min_instance_vertices: int,
) -> dict[str, object]:
    if not isinstance(ground_truth, ReplicaGroundTruth):
        raise TypeError("ground_truth must be ReplicaGroundTruth")
    minimum = _positive_integer(min_instance_vertices, "min_instance_vertices")
    semantic_ids = _semantic_id_set(instance_semantic_ids)
    values = _validated_predictions(ordered, "ordered")
    for item in values:
        if len(item.mask) != len(ground_truth.vertices_xyz):
            raise ValueError("ordered mask size must match ground-truth vertices")
        if int(np.sum(item.mask)) < minimum:
            raise ValueError("ordered predictions must satisfy minimum support")

    gt_masks = _ground_truth_instance_masks(ground_truth, semantic_ids, minimum)
    gt_count = len(gt_masks)
    gt_indices = np.full(len(ground_truth.vertices_xyz), -1, dtype=np.int64)
    gt_sizes = np.empty(gt_count, dtype=np.int64)
    for index, (_, mask) in enumerate(gt_masks):
        gt_indices[mask] = index
        gt_sizes[index] = int(np.sum(mask))
    ious = np.zeros((len(values), gt_count), dtype=np.float64)
    for index, hypothesis in enumerate(values):
        predicted_size = int(np.sum(hypothesis.mask))
        matched_indices = gt_indices[hypothesis.mask]
        matched_indices = matched_indices[matched_indices >= 0]
        intersections = np.bincount(matched_indices, minlength=gt_count)
        unions = predicted_size + gt_sizes - intersections
        np.divide(intersections, unions, out=ious[index], where=unions > 0)

    def threshold_metrics(threshold: float) -> tuple[float, float]:
        matched_gt: set[int] = set()
        tp: list[float] = []
        fp: list[float] = []
        for row in ious:
            available = [index for index in range(gt_count) if index not in matched_gt]
            best_index = max(available, key=lambda index: row[index], default=-1)
            is_match = best_index >= 0 and row[best_index] >= threshold
            tp.append(1.0 if is_match else 0.0)
            fp.append(0.0 if is_match else 1.0)
            if is_match:
                matched_gt.add(best_index)
        return (
            _average_precision(np.asarray(tp), np.asarray(fp), gt_count),
            float(len(matched_gt) / gt_count) if gt_count else 0.0,
        )

    ap25, recall25 = threshold_metrics(0.25)
    ap50, recall50 = threshold_metrics(0.50)
    return {
        "ap25": ap25,
        "ap50": ap50,
        "recall25": recall25,
        "recall50": recall50,
        "raw_hypothesis_count": len(values),
        "supported_hypothesis_count": len(values),
        "predicted_instance_count": len(values),
        "ground_truth_instance_count": gt_count,
        "prediction_hypothesis_ids": [item.hypothesis_id for item in values],
        "prediction_confidences": [item.score for item in values],
    }
