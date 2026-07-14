"""Class-agnostic point-mask instance AP for neutral MapSnapshot outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.contracts import GroundTruthSnapshot, MapSnapshot
from src.evaluation.semantic_metrics import nearest_distances_and_indices


def _average_precision(true_positive: np.ndarray, false_positive: np.ndarray, gt_count: int) -> float:
    if gt_count == 0:
        return 1.0 if len(true_positive) == 0 else 0.0
    cumulative_tp = np.cumsum(true_positive, dtype=np.float64)
    cumulative_fp = np.cumsum(false_positive, dtype=np.float64)
    recall = cumulative_tp / float(gt_count)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.flatnonzero(recall[1:] != recall[:-1]) + 1
    return float(np.sum((recall[changes] - recall[changes - 1]) * precision[changes]))


def _predicted_masks(
    prediction: MapSnapshot,
    gt_points: np.ndarray,
    max_distance: float,
) -> list[tuple[float, str, np.ndarray]]:
    masks: list[tuple[float, str, np.ndarray]] = []
    for entity in prediction.entities:
        distances, nearest = nearest_distances_and_indices(gt_points, entity.points_xyz)
        mask = (nearest >= 0) & (distances <= max_distance)
        masks.append((float(entity.semantic_score), entity.entity_id, mask))
    return sorted(masks, key=lambda item: (-item[0], item[1]))


def _evaluate_threshold(
    masks: list[tuple[float, str, np.ndarray]],
    gt_masks: dict[int, np.ndarray],
    threshold: float,
) -> tuple[float, float]:
    matched_gt: set[int] = set()
    true_positive: list[float] = []
    false_positive: list[float] = []
    for _, _, mask in masks:
        best_id = -1
        best_iou = 0.0
        for gt_id, gt_mask in gt_masks.items():
            if gt_id in matched_gt:
                continue
            intersection = int(np.sum(mask & gt_mask))
            union = int(np.sum(mask | gt_mask))
            iou = float(intersection / union) if union else 0.0
            if iou > best_iou:
                best_id = gt_id
                best_iou = iou
        is_match = best_id >= 0 and best_iou >= threshold
        true_positive.append(1.0 if is_match else 0.0)
        false_positive.append(0.0 if is_match else 1.0)
        if is_match:
            matched_gt.add(best_id)
    tp = np.asarray(true_positive, dtype=np.float64)
    fp = np.asarray(false_positive, dtype=np.float64)
    ap = _average_precision(tp, fp, len(gt_masks))
    recall = float(len(matched_gt) / len(gt_masks)) if gt_masks else 1.0
    return ap, recall


def evaluate_instance_snapshot(
    prediction: MapSnapshot,
    ground_truth: GroundTruthSnapshot,
    *,
    max_distance: float = 0.05,
) -> dict[str, Any]:
    if prediction.scene_id != ground_truth.scene_id:
        raise ValueError("prediction and ground truth scene mismatch")
    if max_distance <= 0.0:
        raise ValueError("max_distance must be positive")
    gt_ids = sorted(int(value) for value in np.unique(ground_truth.instance_ids) if int(value) >= 0)
    gt_masks = {gt_id: ground_truth.instance_ids == gt_id for gt_id in gt_ids}
    masks = _predicted_masks(prediction, ground_truth.points_xyz, float(max_distance))
    ap25, recall25 = _evaluate_threshold(masks, gt_masks, 0.25)
    ap50, recall50 = _evaluate_threshold(masks, gt_masks, 0.50)
    return {
        "ap25": ap25,
        "ap50": ap50,
        "recall25": recall25,
        "recall50": recall50,
        "predicted_instance_count": int(len(masks)),
        "ground_truth_instance_count": int(len(gt_masks)),
    }
