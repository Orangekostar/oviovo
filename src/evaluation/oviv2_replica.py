from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from src.oviv2.meshing import LabeledMesh


def _read_only(value: np.ndarray, *, dtype) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ReplicaGroundTruth:
    vertices_xyz: np.ndarray
    semantic_ids: np.ndarray
    instance_ids: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices_xyz", _read_only(self.vertices_xyz, dtype=np.float32))
        object.__setattr__(self, "semantic_ids", _read_only(self.semantic_ids, dtype=np.int64))
        object.__setattr__(self, "instance_ids", _read_only(self.instance_ids, dtype=np.int64))
        count = self.vertices_xyz.shape[0]
        if self.vertices_xyz.shape != (count, 3):
            raise ValueError("ground-truth vertices_xyz must have shape (N, 3)")
        if self.semantic_ids.shape != (count,) or self.instance_ids.shape != (count,):
            raise ValueError("ground-truth ID arrays must have shape (N,)")
        if not np.isfinite(self.vertices_xyz).all():
            raise ValueError("ground-truth vertices must be finite")
        if np.any(self.semantic_ids < 0) or np.any(self.instance_ids < 0):
            raise ValueError("ground-truth IDs must be non-negative")


@dataclass(frozen=True)
class EntityEvaluationInfo:
    entity_id: int
    semantic_id: int
    accepted_view_count: int
    semantic_confidence: float = 1.0

    def __post_init__(self) -> None:
        for name in ("entity_id", "semantic_id"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.accepted_view_count, (int, np.integer))
            or isinstance(self.accepted_view_count, bool)
            or self.accepted_view_count < 0
        ):
            raise ValueError("accepted_view_count must be a non-negative integer")
        if isinstance(self.semantic_confidence, (bool, np.bool_)) or not isinstance(
            self.semantic_confidence,
            Real,
        ):
            raise TypeError("semantic_confidence must be a real number")
        semantic_confidence = float(self.semantic_confidence)
        if not np.isfinite(semantic_confidence) or not 0.0 <= semantic_confidence <= 1.0:
            raise ValueError("semantic_confidence must be finite and lie in [0, 1]")
        object.__setattr__(self, "semantic_confidence", semantic_confidence)


@dataclass(frozen=True)
class ProjectedLabels:
    semantic_ids: np.ndarray
    entity_ids: np.ndarray
    distances_m: np.ndarray
    matched: np.ndarray
    nearest_predicted_indices: np.ndarray

    def __post_init__(self) -> None:
        dtypes = {
            "semantic_ids": np.int64,
            "entity_ids": np.int64,
            "distances_m": np.float64,
            "matched": np.bool_,
            "nearest_predicted_indices": np.int64,
        }
        for field in fields(self):
            object.__setattr__(
                self,
                field.name,
                _read_only(getattr(self, field.name), dtype=dtypes[field.name]),
            )
        count = self.semantic_ids.shape[0]
        if any(getattr(self, field.name).shape != (count,) for field in fields(self)):
            raise ValueError("projected label arrays must all have shape (N,)")


def _positive_float(value: float, name: str) -> float:
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return normalized


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_id_set(values: Iterable[int], name: str) -> set[int]:
    result: set[int] = set()
    for value in values:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must contain only positive integer IDs")
        result.add(int(value))
    return result


def project_mesh_to_gt(
    mesh: LabeledMesh,
    gt_vertices_xyz: np.ndarray,
    max_distance_m: float = 0.05,
) -> ProjectedLabels:
    if not isinstance(mesh, LabeledMesh):
        raise TypeError("mesh must be a LabeledMesh")
    threshold = _positive_float(max_distance_m, "max_distance_m")
    gt_vertices = np.asarray(gt_vertices_xyz, dtype=np.float32)
    if gt_vertices.ndim != 2 or gt_vertices.shape[1] != 3:
        raise ValueError("gt_vertices_xyz must have shape (N, 3)")
    if not np.isfinite(gt_vertices).all():
        raise ValueError("gt_vertices_xyz must contain only finite values")

    count = gt_vertices.shape[0]
    semantic_ids = np.zeros(count, dtype=np.int64)
    entity_ids = np.zeros(count, dtype=np.int64)
    distances = np.full(count, np.inf, dtype=np.float64)
    nearest = np.full(count, -1, dtype=np.int64)
    matched = np.zeros(count, dtype=bool)
    if count and len(mesh.vertices_xyz):
        queried_distances, queried_indices = cKDTree(mesh.vertices_xyz).query(
            gt_vertices,
            k=1,
            workers=-1,
        )
        distances = np.asarray(queried_distances, dtype=np.float64)
        nearest_candidates = np.asarray(queried_indices, dtype=np.int64)
        matched = distances < threshold
        nearest[matched] = nearest_candidates[matched]
        semantic_ids[matched] = mesh.semantic_ids[nearest_candidates[matched]]
        entity_ids[matched] = mesh.entity_ids[nearest_candidates[matched]]
    return ProjectedLabels(semantic_ids, entity_ids, distances, matched, nearest)


def _semantic_metrics(
    projected: ProjectedLabels,
    ground_truth: ReplicaGroundTruth,
    valid_semantic_ids: set[int],
) -> dict[str, Any]:
    domain = np.isin(ground_truth.semantic_ids, sorted(valid_semantic_ids))
    gt = ground_truth.semantic_ids[domain]
    predicted = projected.semantic_ids[domain].copy()
    predicted[~np.isin(predicted, sorted(valid_semantic_ids))] = 0
    classes = sorted(int(value) for value in np.unique(gt))
    ious: list[float] = []
    accuracies: list[float] = []
    frequencies: list[float] = []
    per_class: dict[str, dict[str, float | int]] = {}
    total = len(gt)
    for semantic_id in classes:
        gt_positive = gt == semantic_id
        predicted_positive = predicted == semantic_id
        true_positive = int(np.sum(gt_positive & predicted_positive))
        false_positive = int(np.sum(~gt_positive & predicted_positive))
        false_negative = int(np.sum(gt_positive & ~predicted_positive))
        support = int(np.sum(gt_positive))
        union = true_positive + false_positive + false_negative
        iou = float(true_positive / union) if union else 0.0
        accuracy = float(true_positive / support) if support else 0.0
        ious.append(iou)
        accuracies.append(accuracy)
        frequencies.append(support / total if total else 0.0)
        per_class[str(semantic_id)] = {
            "iou": iou,
            "accuracy": accuracy,
            "support": support,
        }
    return {
        "miou": float(np.mean(ious)) if ious else 0.0,
        "macc": float(np.mean(accuracies)) if accuracies else 0.0,
        "f_miou": float(np.dot(frequencies, ious)) if ious else 0.0,
        "matched_vertex_ratio": (
            float(np.mean(projected.matched[domain])) if total else 0.0
        ),
        "evaluated_vertex_count": total,
        "per_class": per_class,
    }


def _average_precision(tp: np.ndarray, fp: np.ndarray, gt_count: int) -> float:
    if gt_count == 0:
        return 0.0
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
    predictions: Sequence[tuple[float, int, np.ndarray]],
    ground_truth_masks: Sequence[tuple[object, np.ndarray]],
    threshold: float,
) -> tuple[float, float]:
    matched_gt: set[int] = set()
    tp: list[float] = []
    fp: list[float] = []
    for _, _, predicted_mask in predictions:
        best_index = -1
        best_iou = 0.0
        for index, (_, gt_mask) in enumerate(ground_truth_masks):
            if index in matched_gt:
                continue
            intersection = int(np.sum(predicted_mask & gt_mask))
            union = int(np.sum(predicted_mask | gt_mask))
            iou = float(intersection / union) if union else 0.0
            if iou > best_iou:
                best_index = index
                best_iou = iou
        is_match = best_index >= 0 and best_iou >= threshold
        tp.append(1.0 if is_match else 0.0)
        fp.append(0.0 if is_match else 1.0)
        if is_match:
            matched_gt.add(best_index)
    gt_count = len(ground_truth_masks)
    return (
        _average_precision(np.asarray(tp), np.asarray(fp), gt_count),
        float(len(matched_gt) / gt_count) if gt_count else 0.0,
    )


def _ground_truth_instance_masks(
    ground_truth: ReplicaGroundTruth,
    instance_semantic_ids: set[int],
    min_instance_vertices: int,
) -> list[tuple[tuple[int, int], np.ndarray]]:
    masks: list[tuple[tuple[int, int], np.ndarray]] = []
    for semantic_id in sorted(instance_semantic_ids):
        class_domain = ground_truth.semantic_ids == semantic_id
        for instance_id in sorted(
            int(value) for value in np.unique(ground_truth.instance_ids[class_domain])
        ):
            if instance_id <= 0:
                continue
            mask = class_domain & (ground_truth.instance_ids == instance_id)
            if int(np.sum(mask)) >= min_instance_vertices:
                masks.append(((semantic_id, instance_id), mask))
    return masks


def _predicted_entity_masks(
    projected: ProjectedLabels,
    min_instance_vertices: int,
) -> list[tuple[float, int, np.ndarray]]:
    unscaled: list[tuple[int, int, np.ndarray]] = []
    for entity_id in sorted(int(value) for value in np.unique(projected.entity_ids)):
        if entity_id <= 0:
            continue
        mask = projected.entity_ids == entity_id
        size = int(np.sum(mask))
        if size >= min_instance_vertices:
            unscaled.append((entity_id, size, mask))
    if not unscaled:
        return []
    maximum_size = max(size for _, size, _ in unscaled)
    return sorted(
        [
            (float(size / maximum_size), entity_id, mask)
            for entity_id, size, mask in unscaled
        ],
        key=lambda item: (-item[0], item[1]),
    )


def _class_agnostic_instance_metrics(
    projected: ProjectedLabels,
    ground_truth: ReplicaGroundTruth,
    instance_semantic_ids: set[int],
    min_instance_vertices: int,
) -> dict[str, Any]:
    ground_truth_masks = _ground_truth_instance_masks(
        ground_truth,
        instance_semantic_ids,
        min_instance_vertices,
    )
    predictions = _predicted_entity_masks(projected, min_instance_vertices)
    ap25, recall25 = _instance_threshold(predictions, ground_truth_masks, 0.25)
    ap50, recall50 = _instance_threshold(predictions, ground_truth_masks, 0.50)
    return {
        "ap25": ap25,
        "ap50": ap50,
        "recall25": recall25,
        "recall50": recall50,
        "predicted_instance_count": len(predictions),
        "ground_truth_instance_count": len(ground_truth_masks),
        "prediction_entity_ids": [entity_id for _, entity_id, _ in predictions],
        "prediction_confidences": [confidence for confidence, _, _ in predictions],
    }


def _semantic_class_constrained_instance_metrics(
    projected: ProjectedLabels,
    ground_truth: ReplicaGroundTruth,
    entity_info: Sequence[EntityEvaluationInfo],
    instance_semantic_ids: set[int],
    min_instance_vertices: int,
) -> dict[str, Any]:
    info_by_entity: dict[int, EntityEvaluationInfo] = {}
    for info in entity_info:
        if not isinstance(info, EntityEvaluationInfo):
            raise TypeError("entity_info must contain EntityEvaluationInfo values")
        if info.entity_id in info_by_entity:
            raise ValueError(f"duplicate entity evaluation info for {info.entity_id}")
        info_by_entity[info.entity_id] = info

    gt_by_class: dict[int, list[tuple[int, np.ndarray]]] = {}
    for semantic_id in sorted(instance_semantic_ids):
        class_domain = ground_truth.semantic_ids == semantic_id
        for instance_id in sorted(int(value) for value in np.unique(ground_truth.instance_ids[class_domain])):
            if instance_id <= 0:
                continue
            mask = class_domain & (ground_truth.instance_ids == instance_id)
            if int(np.sum(mask)) >= min_instance_vertices:
                gt_by_class.setdefault(semantic_id, []).append((instance_id, mask))

    unscaled_predictions: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    for entity_id in sorted(int(value) for value in np.unique(projected.entity_ids)):
        if entity_id <= 0:
            continue
        info = info_by_entity.get(entity_id)
        if (
            info is None
            or info.accepted_view_count < 2
            or info.semantic_id not in instance_semantic_ids
        ):
            continue
        mask = projected.entity_ids == entity_id
        size = int(np.sum(mask))
        if size < min_instance_vertices:
            continue
        unscaled_predictions.setdefault(info.semantic_id, []).append((entity_id, size, mask))

    predictions_by_class: dict[int, list[tuple[float, int, np.ndarray]]] = {}
    for semantic_id, class_predictions in unscaled_predictions.items():
        maximum_size = max(size for _, size, _ in class_predictions)
        predictions_by_class[semantic_id] = sorted(
            [
                (float(size / maximum_size), entity_id, mask)
                for entity_id, size, mask in class_predictions
            ],
            key=lambda item: (-item[0], item[1]),
        )

    ap25_values: list[float] = []
    ap50_values: list[float] = []
    per_class: dict[str, Any] = {}
    for semantic_id in sorted(gt_by_class):
        gt_masks = gt_by_class[semantic_id]
        predictions = predictions_by_class.get(semantic_id, [])
        ap25, recall25 = _instance_threshold(predictions, gt_masks, 0.25)
        ap50, recall50 = _instance_threshold(predictions, gt_masks, 0.50)
        ap25_values.append(ap25)
        ap50_values.append(ap50)
        per_class[str(semantic_id)] = {
            "ap25": ap25,
            "ap50": ap50,
            "recall25": recall25,
            "recall50": recall50,
            "ground_truth_count": len(gt_masks),
            "prediction_count": len(predictions),
            "prediction_entity_ids": [entity_id for _, entity_id, _ in predictions],
            "prediction_confidences": [confidence for confidence, _, _ in predictions],
        }
    return {
        "ap25": float(np.mean(ap25_values)) if ap25_values else 0.0,
        "ap50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "predicted_instance_count": sum(len(values) for values in predictions_by_class.values()),
        "ground_truth_instance_count": sum(len(values) for values in gt_by_class.values()),
        "per_class": per_class,
    }


def _nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(query) == 0:
        return np.empty(0, dtype=np.float64)
    if len(reference) == 0:
        return np.full(len(query), np.inf, dtype=np.float64)
    distances, _ = cKDTree(reference).query(query, k=1, workers=-1)
    return np.asarray(distances, dtype=np.float64)


def _geometry_metrics(
    predicted_vertices: np.ndarray,
    ground_truth_vertices: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    precision = (
        float(np.mean(_nearest_distances(predicted_vertices, ground_truth_vertices) < threshold))
        if len(predicted_vertices)
        else (1.0 if len(ground_truth_vertices) == 0 else 0.0)
    )
    recall = (
        float(np.mean(_nearest_distances(ground_truth_vertices, predicted_vertices) < threshold))
        if len(ground_truth_vertices)
        else 1.0
    )
    f5 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f5": f5}


def evaluate_replica_voxel_map(
    mesh: LabeledMesh,
    ground_truth: ReplicaGroundTruth,
    entity_info: Sequence[EntityEvaluationInfo],
    *,
    valid_semantic_ids: Iterable[int],
    instance_semantic_ids: Iterable[int],
    min_instance_vertices: int = 100,
    distance_threshold_m: float = 0.05,
) -> dict[str, Any]:
    if not isinstance(mesh, LabeledMesh):
        raise TypeError("mesh must be a LabeledMesh")
    if not isinstance(ground_truth, ReplicaGroundTruth):
        raise TypeError("ground_truth must be ReplicaGroundTruth")
    semantic_ids = _positive_id_set(valid_semantic_ids, "valid_semantic_ids")
    instance_ids = _positive_id_set(instance_semantic_ids, "instance_semantic_ids")
    if not instance_ids.issubset(semantic_ids):
        raise ValueError("instance_semantic_ids must be a subset of valid_semantic_ids")
    minimum_vertices = _positive_integer(min_instance_vertices, "min_instance_vertices")
    threshold = _positive_float(distance_threshold_m, "distance_threshold_m")
    projected = project_mesh_to_gt(mesh, ground_truth.vertices_xyz, threshold)
    semantic = _semantic_metrics(projected, ground_truth, semantic_ids)
    class_agnostic = _class_agnostic_instance_metrics(
        projected,
        ground_truth,
        instance_ids,
        minimum_vertices,
    )
    semantic_constrained = _semantic_class_constrained_instance_metrics(
        projected,
        ground_truth,
        entity_info,
        instance_ids,
        minimum_vertices,
    )
    geometry = _geometry_metrics(mesh.vertices_xyz, ground_truth.vertices_xyz, threshold)
    return {
        "miou": semantic["miou"],
        "macc": semantic["macc"],
        "f_miou": semantic["f_miou"],
        "ap25": class_agnostic["ap25"],
        "ap50": class_agnostic["ap50"],
        "f5": geometry["f5"],
        "semantic": semantic,
        "instance": {
            "class_agnostic": class_agnostic,
            "semantic_class_constrained": semantic_constrained,
        },
        "geometry": geometry,
        "projection": {
            "matched_vertex_count": int(np.sum(projected.matched)),
            "ground_truth_vertex_count": len(ground_truth.vertices_xyz),
        },
        "protocol": {
            "distance_threshold_m": threshold,
            "projection_comparator": "strict_less_than",
            "min_instance_vertices": minimum_vertices,
            "valid_semantic_ids": sorted(semantic_ids),
            "instance_semantic_ids": sorted(instance_ids),
            "headline_instance_protocol": "class_agnostic",
            "semantic_instance_protocol": "diagnostic_only",
        },
    }
