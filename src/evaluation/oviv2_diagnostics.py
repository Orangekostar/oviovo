"""Pure diagnostics for projected OVIV2 semantic and entity labels."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _id_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (N,)")
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer IDs")
    normalized = np.asarray(array, dtype=np.int64)
    if np.any(normalized < 0):
        raise ValueError(f"{name} must contain non-negative IDs")
    return normalized


def _positive_id_set(values: Iterable[int]) -> set[int]:
    try:
        items = list(values)
    except TypeError as error:
        raise ValueError("valid_semantic_ids must contain only positive integer IDs") from error
    if not items:
        raise ValueError("valid_semantic_ids must be non-empty")
    result: set[int] = set()
    for value in items:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
            raise ValueError("valid_semantic_ids must contain only positive integer IDs")
        result.add(int(value))
    return result


def _validated_semantic_inputs(
    predicted: Any,
    ground_truth: Any,
    valid_semantic_ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, set[int]]:
    predicted_ids = _id_array(predicted, name="predicted")
    gt_ids = _id_array(ground_truth, name="ground_truth")
    if predicted_ids.shape != gt_ids.shape:
        raise ValueError("predicted and ground_truth must have the same shape")
    return predicted_ids, gt_ids, _positive_id_set(valid_semantic_ids)


def semantic_iou_from_ids(
    predicted: Any,
    ground_truth: Any,
    valid_semantic_ids: Iterable[int],
) -> dict[str, Any]:
    """Compute semantic IoU on vertices with valid ground-truth labels."""
    predicted_ids, gt_ids, valid_ids = _validated_semantic_inputs(
        predicted,
        ground_truth,
        valid_semantic_ids,
    )
    domain = np.isin(gt_ids, sorted(valid_ids))
    evaluated_gt = gt_ids[domain]
    evaluated_predicted = predicted_ids[domain]
    classes = sorted(int(value) for value in np.unique(evaluated_gt))

    per_class_iou: dict[str, float] = {}
    for semantic_id in classes:
        gt_positive = evaluated_gt == semantic_id
        predicted_positive = evaluated_predicted == semantic_id
        intersection = int(np.sum(gt_positive & predicted_positive))
        union = int(np.sum(gt_positive | predicted_positive))
        per_class_iou[str(semantic_id)] = float(intersection / union) if union else 0.0

    return {
        "miou": float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0,
        "per_class_iou": per_class_iou,
        "evaluated_vertex_count": int(np.sum(domain)),
    }


def _minimum_overlap_vertices(value: Any) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
        raise ValueError("minimum_overlap_vertices must be a positive integer")
    return int(value)


def _minimum_overlap_fraction(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("minimum_overlap_fraction must be finite and in [0, 1]")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("minimum_overlap_fraction must be finite and in [0, 1]") from error
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("minimum_overlap_fraction must be finite and in [0, 1]")
    return normalized


def diagnose_projected_labels(
    predicted_semantic_ids: Any,
    predicted_entity_ids: Any,
    gt_semantic_ids: Any,
    gt_instance_ids: Any,
    *,
    valid_semantic_ids: Iterable[int],
    minimum_overlap_vertices: int = 100,
    minimum_overlap_fraction: float = 0.10,
) -> dict[str, Any]:
    """Diagnose semantic labeling and entity association without changing inputs."""
    predicted_semantic = _id_array(
        predicted_semantic_ids,
        name="predicted_semantic_ids",
    )
    predicted_entities = _id_array(predicted_entity_ids, name="predicted_entity_ids")
    gt_semantic = _id_array(gt_semantic_ids, name="gt_semantic_ids")
    gt_instances = _id_array(gt_instance_ids, name="gt_instance_ids")
    shapes = {
        predicted_semantic.shape,
        predicted_entities.shape,
        gt_semantic.shape,
        gt_instances.shape,
    }
    if len(shapes) != 1:
        raise ValueError("projected and ground-truth label arrays must have the same shape")
    valid_ids = _positive_id_set(valid_semantic_ids)
    minimum_vertices = _minimum_overlap_vertices(minimum_overlap_vertices)
    minimum_fraction = _minimum_overlap_fraction(minimum_overlap_fraction)

    oracle_semantic = np.zeros(gt_semantic.shape, dtype=np.int64)
    oracle_entity_labels: dict[str, int] = {}
    significant_by_entity: dict[str, list[int]] = {}
    entities_by_gt_instance: dict[int, list[int]] = {}
    overmerged_entities: list[int] = []

    entity_ids = sorted(int(value) for value in np.unique(predicted_entities) if value > 0)
    valid_ids_sorted = sorted(valid_ids)
    for entity_id in entity_ids:
        entity_mask = predicted_entities == entity_id
        entity_size = int(np.sum(entity_mask))

        valid_labels = gt_semantic[entity_mask & np.isin(gt_semantic, valid_ids_sorted)]
        if valid_labels.size:
            labels, counts = np.unique(valid_labels, return_counts=True)
            oracle_label = int(labels[np.argmax(counts)])
            oracle_entity_labels[str(entity_id)] = oracle_label
            oracle_semantic[entity_mask] = oracle_label

        positive_instances = gt_instances[entity_mask & (gt_instances > 0)]
        instance_ids, overlap_counts = np.unique(positive_instances, return_counts=True)
        significant_instances = [
            int(instance_id)
            for instance_id, count in zip(instance_ids, overlap_counts, strict=True)
            if count >= minimum_vertices and count / entity_size >= minimum_fraction
        ]
        significant_by_entity[str(entity_id)] = significant_instances
        if len(significant_instances) >= 2:
            overmerged_entities.append(entity_id)
        for instance_id in significant_instances:
            entities_by_gt_instance.setdefault(instance_id, []).append(entity_id)

    fragmented_instances = sorted(
        instance_id
        for instance_id, represented_entities in entities_by_gt_instance.items()
        if len(represented_entities) >= 2
    )
    current_metrics = semantic_iou_from_ids(
        predicted_semantic,
        gt_semantic,
        valid_ids,
    )
    oracle_metrics = semantic_iou_from_ids(oracle_semantic, gt_semantic, valid_ids)

    return {
        "headline_eligible": False,
        "semantic": {
            "current_miou": current_metrics["miou"],
            "oracle_miou": oracle_metrics["miou"],
            "oracle_entity_labels": oracle_entity_labels,
        },
        "entity_geometry": {
            "overmerged_entity_ids": overmerged_entities,
            "fragmented_gt_instance_ids": fragmented_instances,
            "significant_gt_instances_by_entity": significant_by_entity,
        },
    }
