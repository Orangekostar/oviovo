"""Runtime semantic metrics evaluated on a fixed ground-truth point set."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from src.evaluation.contracts import GroundTruthSnapshot, MapSnapshot

UNMATCHED_LABEL = "__unmatched__"


def nearest_distances_and_indices(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    chunk_size: int = 1024,
    reference_chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact nearest-reference distances and indices using bounded memory."""
    query = np.asarray(query_points, dtype=np.float32).reshape(-1, 3)
    reference = np.asarray(reference_points, dtype=np.float32).reshape(-1, 3)
    distances = np.full(len(query), np.inf, dtype=np.float32)
    indices = np.full(len(query), -1, dtype=np.int64)
    if len(reference) == 0:
        return distances, indices
    query_chunk = max(int(chunk_size), 1)
    reference_chunk = max(int(reference_chunk_size), 1)
    for start in range(0, len(query), query_chunk):
        stop = min(start + query_chunk, len(query))
        best_squared = np.full(stop - start, np.inf, dtype=np.float32)
        best_indices = np.full(stop - start, -1, dtype=np.int64)
        for reference_start in range(0, len(reference), reference_chunk):
            reference_stop = min(reference_start + reference_chunk, len(reference))
            delta = query[start:stop, None, :] - reference[None, reference_start:reference_stop, :]
            squared = np.einsum("qri,qri->qr", delta, delta)
            local_indices = np.argmin(squared, axis=1)
            local_squared = squared[np.arange(stop - start), local_indices]
            improved = local_squared < best_squared
            best_squared[improved] = local_squared[improved]
            best_indices[improved] = reference_start + local_indices[improved]
        indices[start:stop] = best_indices
        distances[start:stop] = np.sqrt(best_squared)
    return distances, indices


def _prediction_points_and_labels(snapshot: MapSnapshot) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    labels: list[str] = []
    for entity in snapshot.entities:
        if not entity.semantic_label or len(entity.points_xyz) == 0:
            continue
        points.append(entity.points_xyz)
        labels.extend([str(entity.semantic_label)] * len(entity.points_xyz))
    if not points:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=object)
    return np.concatenate(points, axis=0), np.asarray(labels, dtype=object)


def evaluate_semantic_snapshot(
    prediction: MapSnapshot,
    ground_truth: GroundTruthSnapshot,
    *,
    max_distance: float = 0.05,
    ignored_labels: Iterable[str] = (),
) -> dict[str, Any]:
    """Evaluate runtime labels; ground truth is used only by this evaluator."""
    if prediction.scene_id != ground_truth.scene_id:
        raise ValueError("prediction and ground truth scene mismatch")
    if max_distance <= 0.0:
        raise ValueError("max_distance must be positive")
    ignored = {str(label) for label in ignored_labels}
    keep = np.asarray([str(label) not in ignored for label in ground_truth.semantic_labels], dtype=bool)
    gt_points = ground_truth.points_xyz[keep]
    gt_labels = np.asarray([str(label) for label in ground_truth.semantic_labels[keep]], dtype=object)
    pred_points, point_labels = _prediction_points_and_labels(prediction)
    distances, nearest = nearest_distances_and_indices(gt_points, pred_points)
    predicted_labels = np.full(len(gt_points), UNMATCHED_LABEL, dtype=object)
    matched = (nearest >= 0) & (distances <= float(max_distance))
    predicted_labels[matched] = point_labels[nearest[matched]]

    classes = sorted(set(gt_labels.tolist()))
    per_class: dict[str, dict[str, float]] = {}
    ious: list[float] = []
    accuracies: list[float] = []
    frequencies: list[float] = []
    total = max(len(gt_labels), 1)
    for label in classes:
        gt_positive = gt_labels == label
        pred_positive = predicted_labels == label
        true_positive = int(np.sum(gt_positive & pred_positive))
        false_positive = int(np.sum(~gt_positive & pred_positive))
        false_negative = int(np.sum(gt_positive & ~pred_positive))
        union = true_positive + false_positive + false_negative
        support = int(np.sum(gt_positive))
        iou = float(true_positive / union) if union else 1.0
        accuracy = float(true_positive / support) if support else 1.0
        per_class[label] = {"iou": iou, "accuracy": accuracy, "support": float(support)}
        ious.append(iou)
        accuracies.append(accuracy)
        frequencies.append(support / total)

    miou = float(np.mean(ious)) if ious else 1.0
    macc = float(np.mean(accuracies)) if accuracies else 1.0
    f_miou = float(np.dot(frequencies, ious)) if ious else 1.0
    f_macc = float(np.mean(predicted_labels == gt_labels)) if len(gt_labels) else 1.0
    return {
        "miou": miou,
        "macc": macc,
        "f_miou": f_miou,
        "f_macc": f_macc,
        "matched_point_ratio": float(np.mean(matched)) if len(matched) else 1.0,
        "evaluated_point_count": int(len(gt_labels)),
        "per_class": per_class,
    }
