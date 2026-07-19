from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from statistics import fmean
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from src.evaluation.oviv2_replica import ReplicaGroundTruth, evaluate_replica_voxel_map
from src.oviv2.meshing import LabeledMesh


def _read_only_points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.ascontiguousarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    points.setflags(write=False)
    return points


def _frame_id(value: int, name: str = "frame_id") -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _unit_metric(value: float, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return normalized


def _count(value: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True)
class DynamicGroundTruthFrame:
    frame_id: int
    intervention_id: str
    current: ReplicaGroundTruth
    removed_region_vertices_xyz: np.ndarray
    revealed_background_vertices_xyz: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _frame_id(self.frame_id))
        intervention = str(self.intervention_id).strip()
        if not intervention:
            raise ValueError("intervention_id must be non-empty")
        object.__setattr__(self, "intervention_id", intervention)
        if not isinstance(self.current, ReplicaGroundTruth):
            raise TypeError("current must be ReplicaGroundTruth")
        object.__setattr__(
            self,
            "removed_region_vertices_xyz",
            _read_only_points(
                self.removed_region_vertices_xyz,
                "removed_region_vertices_xyz",
            ),
        )
        object.__setattr__(
            self,
            "revealed_background_vertices_xyz",
            _read_only_points(
                self.revealed_background_vertices_xyz,
                "revealed_background_vertices_xyz",
            ),
        )


@dataclass(frozen=True)
class DynamicSnapshotEvaluation:
    frame_id: int
    intervention_id: str
    current_miou: float
    ghost_rate: float
    background_f5: float
    predicted_vertex_count: int = 0
    ghost_vertex_count: int = 0
    background_precision_match_count: int = 0
    background_precision_denominator: int = 0
    background_recall_match_count: int = 0
    background_recall_denominator: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _frame_id(self.frame_id))
        intervention = str(self.intervention_id).strip()
        if not intervention:
            raise ValueError("intervention_id must be non-empty")
        object.__setattr__(self, "intervention_id", intervention)
        for name in ("current_miou", "ghost_rate", "background_f5"):
            object.__setattr__(self, name, _unit_metric(getattr(self, name), name))
        count_names = (
            "predicted_vertex_count",
            "ghost_vertex_count",
            "background_precision_match_count",
            "background_precision_denominator",
            "background_recall_match_count",
            "background_recall_denominator",
        )
        for name in count_names:
            object.__setattr__(self, name, _count(getattr(self, name), name))
        if self.ghost_vertex_count > self.predicted_vertex_count:
            raise ValueError("ghost_vertex_count cannot exceed predicted_vertex_count")
        if self.background_precision_match_count > self.background_precision_denominator:
            raise ValueError("background precision matches cannot exceed denominator")
        if self.background_recall_match_count > self.background_recall_denominator:
            raise ValueError("background recall matches cannot exceed denominator")


def _match_count(query: np.ndarray, reference: np.ndarray, threshold: float) -> int:
    if len(query) == 0 or len(reference) == 0:
        return 0
    distances, _ = cKDTree(reference).query(query, k=1, workers=-1)
    return int(np.sum(np.asarray(distances, dtype=np.float64) < threshold))


def _f1(precision: float, recall: float) -> float:
    return (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )


def evaluate_dynamic_snapshot(
    mesh: LabeledMesh,
    ground_truth: DynamicGroundTruthFrame,
    *,
    valid_semantic_ids: Iterable[int],
    distance_threshold_m: float = 0.05,
) -> DynamicSnapshotEvaluation:
    if not isinstance(mesh, LabeledMesh):
        raise TypeError("mesh must be LabeledMesh")
    if not isinstance(ground_truth, DynamicGroundTruthFrame):
        raise TypeError("ground_truth must be DynamicGroundTruthFrame")
    threshold = float(distance_threshold_m)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("distance_threshold_m must be finite and positive")
    static = evaluate_replica_voxel_map(
        mesh,
        ground_truth.current,
        [],
        valid_semantic_ids=valid_semantic_ids,
        instance_semantic_ids=set(),
        min_instance_vertices=1,
        distance_threshold_m=threshold,
    )

    predicted = mesh.vertices_xyz
    removed = ground_truth.removed_region_vertices_xyz
    revealed = ground_truth.revealed_background_vertices_xyz
    ghost_count = _match_count(predicted, removed, threshold)
    predicted_count = len(predicted)
    ghost_rate = float(ghost_count / predicted_count) if predicted_count else 0.0

    precision_matches = _match_count(predicted, revealed, threshold)
    recall_matches = _match_count(revealed, predicted, threshold)
    precision = (
        float(precision_matches / predicted_count)
        if predicted_count
        else (1.0 if len(revealed) == 0 else 0.0)
    )
    recall = (
        float(recall_matches / len(revealed)) if len(revealed) else 1.0
    )
    return DynamicSnapshotEvaluation(
        frame_id=ground_truth.frame_id,
        intervention_id=ground_truth.intervention_id,
        current_miou=static["miou"],
        ghost_rate=ghost_rate,
        background_f5=_f1(precision, recall),
        predicted_vertex_count=predicted_count,
        ghost_vertex_count=ghost_count,
        background_precision_match_count=precision_matches,
        background_precision_denominator=predicted_count,
        background_recall_match_count=recall_matches,
        background_recall_denominator=len(revealed),
    )


def aggregate_dynamic_sequence(
    evaluations: Sequence[DynamicSnapshotEvaluation],
    *,
    intervention_frame_id: int,
    recovery_background_f5: float = 0.9,
) -> dict[str, Any]:
    intervention_frame = _frame_id(intervention_frame_id, "intervention_frame_id")
    threshold = _unit_metric(recovery_background_f5, "recovery_background_f5")
    values = tuple(evaluations)
    if not values:
        raise ValueError("evaluations must be non-empty")
    if any(not isinstance(value, DynamicSnapshotEvaluation) for value in values):
        raise TypeError("evaluations must contain DynamicSnapshotEvaluation values")
    frame_ids = [value.frame_id for value in values]
    if any(current <= previous for previous, current in zip(frame_ids, frame_ids[1:])):
        raise ValueError("evaluation frame IDs must be strictly increasing")
    intervention_ids = {value.intervention_id for value in values}
    if len(intervention_ids) != 1:
        raise ValueError("evaluations must share one intervention_id")
    recovery = next(
        (
            value.frame_id - intervention_frame
            for value in values
            if value.frame_id >= intervention_frame
            and value.background_f5 >= threshold
        ),
        None,
    )
    return {
        "schema_version": 1,
        "intervention_id": values[0].intervention_id,
        "intervention_frame_id": intervention_frame,
        "frame_ids": frame_ids,
        "macro_current_miou": fmean(value.current_miou for value in values),
        "macro_ghost_rate": fmean(value.ghost_rate for value in values),
        "macro_background_f5": fmean(value.background_f5 for value in values),
        "recovery_background_f5": threshold,
        "recovered": recovery is not None,
        "recovery_frames": recovery,
    }
