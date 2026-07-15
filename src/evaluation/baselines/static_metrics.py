"""Fast common-protocol static metrics for external baseline snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from src.evaluation.contracts import GroundTruthSnapshot, MapSnapshot

UNMATCHED_LABEL = "__unmatched__"


def _distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32).reshape(-1, 3)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1, 3)
    if len(query) == 0:
        return np.empty(0, dtype=np.float32)
    if len(reference) == 0:
        return np.full(len(query), np.inf, dtype=np.float32)
    distances, _ = cKDTree(reference).query(query, k=1, workers=-1)
    return np.asarray(distances, dtype=np.float32)


def _prediction_points_labels(
    prediction: MapSnapshot,
    vocabulary: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    labels: list[str] = []
    for entity in prediction.entities:
        label = None if entity.semantic_label is None else str(entity.semantic_label)
        if label not in vocabulary or len(entity.points_xyz) == 0:
            continue
        points.append(entity.points_xyz)
        labels.extend([label] * len(entity.points_xyz))
    if not points:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=object)
    return np.concatenate(points, axis=0), np.asarray(labels, dtype=object)


def _semantic_metrics(
    prediction: MapSnapshot,
    ground_truth: GroundTruthSnapshot,
    vocabulary: set[str],
    threshold: float,
) -> dict[str, Any]:
    keep = np.asarray(
        [str(label) in vocabulary for label in ground_truth.semantic_labels],
        dtype=bool,
    )
    gt_points = ground_truth.points_xyz[keep]
    gt_labels = np.asarray(ground_truth.semantic_labels[keep], dtype=object)
    pred_points, pred_labels = _prediction_points_labels(prediction, vocabulary)

    predicted = np.full(len(gt_points), UNMATCHED_LABEL, dtype=object)
    matched = np.zeros(len(gt_points), dtype=bool)
    if len(pred_points):
        distances, nearest = cKDTree(pred_points).query(gt_points, k=1, workers=-1)
        matched = np.asarray(distances <= threshold, dtype=bool)
        predicted[matched] = pred_labels[np.asarray(nearest[matched], dtype=np.int64)]

    classes = sorted(set(str(label) for label in gt_labels))
    ious: list[float] = []
    accuracies: list[float] = []
    frequencies: list[float] = []
    per_class: dict[str, dict[str, float]] = {}
    total = max(len(gt_labels), 1)
    for label in classes:
        gt_positive = gt_labels == label
        pred_positive = predicted == label
        true_positive = int(np.sum(gt_positive & pred_positive))
        false_positive = int(np.sum(~gt_positive & pred_positive))
        false_negative = int(np.sum(gt_positive & ~pred_positive))
        support = int(np.sum(gt_positive))
        union = true_positive + false_positive + false_negative
        iou = float(true_positive / union) if union else 1.0
        accuracy = float(true_positive / support) if support else 1.0
        per_class[label] = {"iou": iou, "accuracy": accuracy, "support": float(support)}
        ious.append(iou)
        accuracies.append(accuracy)
        frequencies.append(support / total)

    return {
        "miou": float(np.mean(ious)) if ious else 1.0,
        "macc": float(np.mean(accuracies)) if accuracies else 1.0,
        "f_miou": float(np.dot(frequencies, ious)) if ious else 1.0,
        "matched_point_ratio": float(np.mean(matched)) if len(matched) else 1.0,
        "evaluated_point_count": int(len(gt_labels)),
        "per_class": per_class,
    }


def _average_precision(tp: np.ndarray, fp: np.ndarray, gt_count: int) -> float:
    if gt_count == 0:
        return 1.0 if len(tp) == 0 else 0.0
    cumulative_tp = np.cumsum(tp, dtype=np.float64)
    cumulative_fp = np.cumsum(fp, dtype=np.float64)
    recall = np.concatenate(([0.0], cumulative_tp / gt_count, [1.0]))
    precision = np.concatenate(
        ([0.0], cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12), [0.0])
    )
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.flatnonzero(recall[1:] != recall[:-1]) + 1
    return float(np.sum((recall[changes] - recall[changes - 1]) * precision[changes]))


def _instance_threshold(
    masks: list[tuple[float, str, int, dict[int, int]]],
    gt_sizes: dict[int, int],
    threshold: float,
) -> tuple[float, float]:
    matched_gt: set[int] = set()
    tp: list[float] = []
    fp: list[float] = []
    for _, _, predicted_size, intersections in masks:
        best_id = -1
        best_iou = 0.0
        for gt_id, gt_size in gt_sizes.items():
            if gt_id in matched_gt:
                continue
            intersection = intersections.get(gt_id, 0)
            union = predicted_size + gt_size - intersection
            iou = float(intersection / union) if union else 0.0
            if iou > best_iou:
                best_id = gt_id
                best_iou = iou
        is_match = best_id >= 0 and best_iou >= threshold
        tp.append(1.0 if is_match else 0.0)
        fp.append(0.0 if is_match else 1.0)
        if is_match:
            matched_gt.add(best_id)
    ap = _average_precision(np.asarray(tp), np.asarray(fp), len(gt_sizes))
    recall = float(len(matched_gt) / len(gt_sizes)) if gt_sizes else 1.0
    return ap, recall


def _instance_metrics(
    prediction: MapSnapshot,
    ground_truth: GroundTruthSnapshot,
    vocabulary: set[str],
    threshold: float,
    min_instance_points: int,
) -> dict[str, Any]:
    valid_domain = np.asarray(
        [
            int(instance_id) >= 0 and str(label) in vocabulary
            for instance_id, label in zip(
                ground_truth.instance_ids,
                ground_truth.semantic_labels,
                strict=True,
            )
        ],
        dtype=bool,
    )
    domain_points = ground_truth.points_xyz[valid_domain]
    domain_ids = ground_truth.instance_ids[valid_domain]
    gt_ids = [
        int(value)
        for value in np.unique(domain_ids)
        if int(np.sum(domain_ids == value)) >= min_instance_points
    ]
    keep_gt = np.isin(domain_ids, gt_ids)
    domain_points = domain_points[keep_gt]
    domain_ids = domain_ids[keep_gt]
    gt_sizes = {gt_id: int(np.sum(domain_ids == gt_id)) for gt_id in gt_ids}

    masks: list[tuple[float, str, int, dict[int, int]]] = []
    for entity in prediction.entities:
        if len(entity.points_xyz) == 0:
            continue
        mask = _distances(domain_points, entity.points_xyz) <= threshold
        matched_ids, matched_counts = np.unique(domain_ids[mask], return_counts=True)
        intersections = {
            int(gt_id): int(count)
            for gt_id, count in zip(matched_ids, matched_counts, strict=True)
        }
        masks.append(
            (
                float(entity.semantic_score),
                entity.entity_id,
                int(np.sum(mask)),
                intersections,
            )
        )
    masks.sort(key=lambda item: (-item[0], item[1]))

    ap25, recall25 = _instance_threshold(masks, gt_sizes, 0.25)
    ap50, recall50 = _instance_threshold(masks, gt_sizes, 0.50)
    return {
        "ap25": ap25,
        "ap50": ap50,
        "recall25": recall25,
        "recall50": recall50,
        "predicted_instance_count": len(masks),
        "ground_truth_instance_count": len(gt_sizes),
    }


def _geometry_metrics(
    prediction: MapSnapshot,
    ground_truth: GroundTruthSnapshot,
    threshold: float,
) -> dict[str, float]:
    point_sets = [entity.points_xyz for entity in prediction.entities if len(entity.points_xyz)]
    if prediction.background_xyz is not None and len(prediction.background_xyz):
        point_sets.append(prediction.background_xyz)
    pred_points = (
        np.concatenate(point_sets, axis=0)
        if point_sets
        else np.empty((0, 3), dtype=np.float32)
    )
    gt_points = ground_truth.points_xyz
    precision = (
        float(np.mean(_distances(pred_points, gt_points) <= threshold))
        if len(pred_points)
        else (1.0 if len(gt_points) == 0 else 0.0)
    )
    recall = (
        float(np.mean(_distances(gt_points, pred_points) <= threshold))
        if len(gt_points)
        else 1.0
    )
    f_score = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f5": f_score}


def evaluate_static_snapshot(
    prediction: MapSnapshot,
    ground_truth: GroundTruthSnapshot,
    *,
    semantic_vocabulary: Sequence[str],
    instance_vocabulary: Sequence[str],
    distance_threshold_m: float = 0.05,
    min_instance_points: int = 100,
) -> dict[str, Any]:
    """Evaluate one baseline snapshot using frozen Replica static settings."""
    if prediction.scene_id != ground_truth.scene_id:
        raise ValueError("prediction and ground truth scene mismatch")
    if distance_threshold_m <= 0.0:
        raise ValueError("distance_threshold_m must be positive")
    if min_instance_points <= 0:
        raise ValueError("min_instance_points must be positive")
    semantic_labels = {str(label) for label in semantic_vocabulary}
    instance_labels = {str(label) for label in instance_vocabulary}
    return {
        "semantic": _semantic_metrics(
            prediction,
            ground_truth,
            semantic_labels,
            float(distance_threshold_m),
        ),
        "instance": _instance_metrics(
            prediction,
            ground_truth,
            instance_labels,
            float(distance_threshold_m),
            int(min_instance_points),
        ),
        "geometry": _geometry_metrics(
            prediction,
            ground_truth,
            float(distance_threshold_m),
        ),
        "protocol": {
            "distance_threshold_m": float(distance_threshold_m),
            "min_instance_points": int(min_instance_points),
            "semantic_vocabulary_size": len(semantic_labels),
            "instance_vocabulary_size": len(instance_labels),
        },
    }
