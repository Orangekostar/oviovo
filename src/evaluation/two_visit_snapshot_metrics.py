"""Single-checkpoint metrics for frozen two-visit current-map snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.evaluation.evaluate_tesse_cd_common_v2 import (
    _crop_points_to_voxels,
    _prediction_semantics,
    _voxel_centers,
)
from src.evaluation.baselines.dynamic_metrics import evaluate_dynamic_frame
from src.evaluation.baselines.tesse_semantics import TesseSemanticCrosswalk
from src.evaluation.contracts import MapSnapshot
from src.evaluation.two_visit_current_metrics import (
    RegionEvaluationInput,
    evaluate_regions,
)


def _integer_voxels(value: object, *, columns: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.shape[1:] != (columns,)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError(f"{name} must be an integer (N, {columns}) array")
    result = np.array(raw, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _boolean_mask(value: object, *, count: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.bool_ or raw.shape != (count,):
        raise ValueError(f"{name} must have one boolean per target voxel")
    result = np.array(raw, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class TwoVisitEvaluationContext:
    """Evaluator-only final-frame arrays; never exposed to the method."""

    event_id: str
    frame_id: int
    intervention_frame_id: int
    current_semantic_voxels: np.ndarray
    changed_region_voxels: np.ndarray
    confirmed_free_voxels: np.ndarray
    revealed_background_voxels: np.ndarray
    current_target_t1_unobserved_mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        for name in ("frame_id", "intervention_frame_id"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        current = _integer_voxels(
            self.current_semantic_voxels,
            columns=4,
            name="current_semantic_voxels",
        )
        if not len(current):
            raise ValueError("current semantic target must not be empty")
        object.__setattr__(self, "current_semantic_voxels", current)
        for name in (
            "changed_region_voxels",
            "confirmed_free_voxels",
            "revealed_background_voxels",
        ):
            object.__setattr__(
                self,
                name,
                _integer_voxels(getattr(self, name), columns=3, name=name),
            )
        if not len(self.changed_region_voxels):
            raise ValueError("changed region target must not be empty")
        object.__setattr__(
            self,
            "current_target_t1_unobserved_mask",
            _boolean_mask(
                self.current_target_t1_unobserved_mask,
                count=len(current),
                name="current_target_t1_unobserved_mask",
            ),
        )


def _snapshot_points_and_semantics(
    snapshot: MapSnapshot,
    crosswalk: TesseSemanticCrosswalk,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    chunks: list[np.ndarray] = []
    semantic_chunks: list[np.ndarray] = []
    background_chunks: list[np.ndarray] = []
    for entity in snapshot.entities:
        points = np.asarray(entity.points_xyz, dtype=np.float32)
        lookup = (
            crosswalk.lookup(entity.semantic_label)
            if entity.semantic_label is not None
            else crosswalk.unknown
        )
        if lookup.matched and lookup.semantic_id not in crosswalk.valid_semantic_ids:
            background_chunks.append(points)
            continue
        chunks.append(points)
        semantic_chunks.append(
            np.full(len(points), lookup.semantic_id, dtype=np.int64)
        )
    points = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.empty((0, 3), dtype=np.float32)
    )
    semantic_ids = (
        np.concatenate(semantic_chunks)
        if semantic_chunks
        else np.empty((0,), dtype=np.int64)
    )
    return points, semantic_ids, chunks, background_chunks


def evaluate_two_visit_snapshot(
    snapshot: MapSnapshot,
    context: TwoVisitEvaluationContext,
    crosswalk: TesseSemanticCrosswalk,
    *,
    retained_t0_xyz: np.ndarray,
    retained_t0_t1_observed_mask: np.ndarray,
) -> dict[str, float]:
    """Evaluate one final snapshot with the common-v2 5 cm definitions."""

    if not isinstance(snapshot, MapSnapshot) or snapshot.scope != "current":
        raise ValueError("snapshot must be a current MapSnapshot")
    if not isinstance(context, TwoVisitEvaluationContext):
        raise TypeError("context must be TwoVisitEvaluationContext")
    if not isinstance(crosswalk, TesseSemanticCrosswalk):
        raise TypeError("crosswalk must be TesseSemanticCrosswalk")
    if snapshot.scene_id != crosswalk.scene:
        raise ValueError("snapshot and semantic crosswalk scene differ")
    if snapshot.timestamp != float(context.frame_id):
        raise ValueError("snapshot does not represent the frozen final frame")

    (
        object_points,
        semantic_ids,
        object_chunks,
        semantic_background_chunks,
    ) = _snapshot_points_and_semantics(snapshot, crosswalk)
    gt_ids, predicted_ids = _prediction_semantics(
        object_points,
        semantic_ids,
        context.current_semantic_voxels,
    )
    predicted_changed = _crop_points_to_voxels(
        object_chunks,
        context.changed_region_voxels,
    )
    background_chunks = list(semantic_background_chunks)
    if snapshot.background_xyz is not None:
        background_chunks.append(
            np.asarray(snapshot.background_xyz, dtype=np.float32)
        )
    predicted_background = _crop_points_to_voxels(
        background_chunks,
        context.changed_region_voxels,
    )
    current = evaluate_dynamic_frame(
        event_id=context.event_id,
        frame_id=context.frame_id,
        intervention_frame_id=context.intervention_frame_id,
        ground_truth_semantic_ids=gt_ids,
        predicted_semantic_ids=predicted_ids,
        valid_semantic_ids=crosswalk.valid_semantic_ids,
        predicted_object_points_in_changed_region=predicted_changed,
        confirmed_free_space_points=_voxel_centers(context.confirmed_free_voxels),
        predicted_background_points_in_revealed_region=predicted_background,
        ground_truth_revealed_background_points=_voxel_centers(
            context.revealed_background_voxels
        ),
        distance_threshold_m=0.05,
    )
    retained = np.asarray(retained_t0_xyz, dtype=np.float32)
    retained_mask = np.asarray(retained_t0_t1_observed_mask)
    regions = evaluate_regions(
        RegionEvaluationInput(
            predicted_current_xyz=object_points,
            ground_truth_current_xyz=_voxel_centers(
                context.current_semantic_voxels[:, :3]
            ),
            ground_truth_t1_unobserved_mask=(
                context.current_target_t1_unobserved_mask
            ),
            retained_t0_xyz=retained,
            retained_t0_t1_observed_mask=retained_mask,
            confirmed_free_xyz=_voxel_centers(context.confirmed_free_voxels),
            predicted_background_xyz=predicted_background,
            ground_truth_background_xyz=_voxel_centers(
                context.revealed_background_voxels
            ),
            distance_threshold_m=0.05,
        )
    )
    return {
        "current_miou": current.current_miou,
        "ghost": current.ghost_rate,
        "background_f1_at_5cm": current.background_f5,
        "surface_precision_at_5cm": regions.surface_precision,
        "surface_recall_at_5cm": regions.surface_recall,
        "surface_f1_at_5cm": regions.surface_f1,
        "total_current_surface_coverage": regions.total_current_surface_coverage,
        "t1_unobserved_region_recall": regions.unobserved_region_recall,
        "t1_observed_region_stale_precision": regions.observed_region_stale_precision,
    }


__all__ = [
    "TwoVisitEvaluationContext",
    "evaluate_two_visit_snapshot",
]
