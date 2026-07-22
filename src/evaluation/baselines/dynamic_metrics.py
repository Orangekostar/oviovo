"""Pure common-protocol metrics for dynamic baseline snapshots."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from statistics import fmean
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def _unit(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite and within [0, 1]")
    return number


def _count(value: int, label: str) -> int:
    number = int(value)
    if isinstance(value, bool) or number < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _frame(value: int, label: str) -> int:
    return _count(value, label)


def _points(value: np.ndarray, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} must contain only finite points")
    return points


def _match_count(query: np.ndarray, reference: np.ndarray, threshold: float) -> int:
    if len(query) == 0 or len(reference) == 0:
        return 0
    distances, _ = cKDTree(reference).query(query, k=1, workers=-1)
    return int(np.sum(np.asarray(distances, dtype=np.float64) < threshold))


def _semantic_miou(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    valid_ids: set[int],
) -> float:
    if ground_truth.ndim != 1 or prediction.ndim != 1:
        raise ValueError("semantic ID arrays must be one-dimensional")
    if ground_truth.shape != prediction.shape:
        raise ValueError("ground-truth and predicted semantic IDs must have equal shape")
    domain = np.isin(ground_truth, list(valid_ids))
    gt = ground_truth[domain]
    pred = prediction[domain]
    classes = sorted(set(int(value) for value in gt))
    if not classes:
        return 1.0
    ious = []
    for semantic_id in classes:
        gt_positive = gt == semantic_id
        pred_positive = pred == semantic_id
        intersection = int(np.sum(gt_positive & pred_positive))
        union = int(np.sum(gt_positive | pred_positive))
        ious.append(intersection / union if union else 1.0)
    return float(fmean(ious))


def _surface_fscore(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float,
) -> tuple[float, int, int]:
    precision_matches = _match_count(prediction, ground_truth, threshold)
    recall_matches = _match_count(ground_truth, prediction, threshold)
    precision = (
        precision_matches / len(prediction)
        if len(prediction)
        else (1.0 if len(ground_truth) == 0 else 0.0)
    )
    recall = (
        recall_matches / len(ground_truth)
        if len(ground_truth)
        else 1.0
    )
    fscore = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return float(fscore), precision_matches, recall_matches


@dataclass(frozen=True)
class DynamicFrameMetrics:
    event_id: str
    frame_id: int
    intervention_frame_id: int
    current_miou: float
    ghost_rate: float
    background_f5: float
    predicted_changed_object_count: int
    ghost_count: int
    predicted_revealed_background_count: int
    ground_truth_revealed_background_count: int
    background_precision_match_count: int = 0
    background_recall_match_count: int = 0

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        if not event_id:
            raise ValueError("event_id must be non-empty")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "frame_id", _frame(self.frame_id, "frame_id"))
        object.__setattr__(
            self,
            "intervention_frame_id",
            _frame(self.intervention_frame_id, "intervention_frame_id"),
        )
        for label in ("current_miou", "ghost_rate", "background_f5"):
            object.__setattr__(self, label, _unit(getattr(self, label), label))
        for label in (
            "predicted_changed_object_count",
            "ghost_count",
            "predicted_revealed_background_count",
            "ground_truth_revealed_background_count",
            "background_precision_match_count",
            "background_recall_match_count",
        ):
            object.__setattr__(self, label, _count(getattr(self, label), label))
        if self.ghost_count > self.predicted_changed_object_count:
            raise ValueError("ghost count exceeds changed-region prediction count")
        if self.background_precision_match_count > self.predicted_revealed_background_count:
            raise ValueError("background precision matches exceed prediction count")
        if self.background_recall_match_count > self.ground_truth_revealed_background_count:
            raise ValueError("background recall matches exceed ground-truth count")


def evaluate_dynamic_frame(
    *,
    event_id: str,
    frame_id: int,
    intervention_frame_id: int,
    ground_truth_semantic_ids: np.ndarray,
    predicted_semantic_ids: np.ndarray,
    valid_semantic_ids: Iterable[int],
    predicted_object_points_in_changed_region: np.ndarray,
    confirmed_free_space_points: np.ndarray,
    predicted_background_points_in_revealed_region: np.ndarray,
    ground_truth_revealed_background_points: np.ndarray,
    distance_threshold_m: float = 0.05,
) -> DynamicFrameMetrics:
    threshold = float(distance_threshold_m)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("distance_threshold_m must be finite and positive")
    valid_ids = {int(value) for value in valid_semantic_ids}
    if not valid_ids:
        raise ValueError("valid_semantic_ids must be non-empty")
    gt_ids = np.asarray(ground_truth_semantic_ids, dtype=np.int64)
    pred_ids = np.asarray(predicted_semantic_ids, dtype=np.int64)
    predicted_changed = _points(
        predicted_object_points_in_changed_region,
        "predicted_object_points_in_changed_region",
    )
    free_space = _points(confirmed_free_space_points, "confirmed_free_space_points")
    predicted_background = _points(
        predicted_background_points_in_revealed_region,
        "predicted_background_points_in_revealed_region",
    )
    gt_background = _points(
        ground_truth_revealed_background_points,
        "ground_truth_revealed_background_points",
    )

    ghost_count = _match_count(predicted_changed, free_space, threshold)
    ghost_rate = ghost_count / len(predicted_changed) if len(predicted_changed) else 0.0
    background_f5, precision_matches, recall_matches = _surface_fscore(
        predicted_background,
        gt_background,
        threshold,
    )
    return DynamicFrameMetrics(
        event_id=event_id,
        frame_id=frame_id,
        intervention_frame_id=intervention_frame_id,
        current_miou=_semantic_miou(gt_ids, pred_ids, valid_ids),
        ghost_rate=float(ghost_rate),
        background_f5=background_f5,
        predicted_changed_object_count=len(predicted_changed),
        ghost_count=ghost_count,
        predicted_revealed_background_count=len(predicted_background),
        ground_truth_revealed_background_count=len(gt_background),
        background_precision_match_count=precision_matches,
        background_recall_match_count=recall_matches,
    )


def sustained_recovery_frame(
    values: Sequence[DynamicFrameMetrics],
    *,
    threshold: float = 0.9,
    consecutive: int = 3,
) -> int | None:
    """Return the first frame of a sustained background-recovery run."""
    threshold_value = _unit(threshold, "threshold")
    consecutive_count = _count(consecutive, "consecutive")
    if consecutive_count == 0:
        raise ValueError("consecutive must be positive")
    frames = tuple(values)
    if any(not isinstance(value, DynamicFrameMetrics) for value in frames):
        raise TypeError("values must contain DynamicFrameMetrics values")
    for start in range(len(frames) - consecutive_count + 1):
        run = frames[start : start + consecutive_count]
        if all(value.background_f5 >= threshold_value for value in run):
            return run[0].frame_id
    return None


def aggregate_dynamic_metrics(
    evaluations: Sequence[DynamicFrameMetrics],
    *,
    recovery_background_f5: float = 0.9,
    recovery_consecutive: int = 3,
    checkpoint_step_frames: int = 50,
    recovery_horizon_frames: int = 450,
    overlap_censor_frames: Mapping[str, int] | None = None,
    background_unobservable_events: Iterable[str] | None = None,
) -> dict[str, Any]:
    values = tuple(evaluations)
    if not values:
        raise ValueError("evaluations must be non-empty")
    if any(not isinstance(value, DynamicFrameMetrics) for value in values):
        raise TypeError("evaluations must contain DynamicFrameMetrics values")
    threshold = _unit(recovery_background_f5, "recovery_background_f5")
    consecutive = _count(recovery_consecutive, "recovery_consecutive")
    checkpoint_step = _count(checkpoint_step_frames, "checkpoint_step_frames")
    recovery_horizon = _count(recovery_horizon_frames, "recovery_horizon_frames")
    if consecutive == 0:
        raise ValueError("recovery_consecutive must be positive")
    if checkpoint_step == 0:
        raise ValueError("checkpoint_step_frames must be positive")
    if recovery_horizon == 0 or recovery_horizon % checkpoint_step:
        raise ValueError(
            "recovery_horizon_frames must be a positive multiple of checkpoint_step_frames"
        )
    overlap_censors = {
        str(event_id): _frame(frame_id, f"overlap censor frame for {event_id}")
        for event_id, frame_id in (overlap_censor_frames or {}).items()
    }
    unobservable_events = {
        str(event_id).strip() for event_id in (background_unobservable_events or ())
    }
    if "" in unobservable_events:
        raise ValueError("background_unobservable_events must contain non-empty IDs")
    grouped: dict[str, list[DynamicFrameMetrics]] = defaultdict(list)
    for value in values:
        grouped[value.event_id].append(value)
    unknown_overlap_events = set(overlap_censors) - set(grouped)
    if unknown_overlap_events:
        raise ValueError(
            "overlap_censor_frames contains unknown events: "
            + ", ".join(sorted(unknown_overlap_events))
        )
    unknown_unobservable_events = unobservable_events - set(grouped)
    if unknown_unobservable_events:
        raise ValueError(
            "background_unobservable_events contains unknown events: "
            + ", ".join(sorted(unknown_unobservable_events))
        )
    if set(grouped) == unobservable_events:
        raise ValueError("scene must contain at least one background-observable event")

    event_results: dict[str, dict[str, Any]] = {}
    for event_id in sorted(grouped):
        event_values = grouped[event_id]
        interventions = {value.intervention_frame_id for value in event_values}
        if len(interventions) != 1:
            raise ValueError(f"event {event_id} has inconsistent intervention frames")
        frame_ids = [value.frame_id for value in event_values]
        if any(current <= previous for previous, current in zip(frame_ids, frame_ids[1:])):
            raise ValueError(f"event {event_id} frame IDs must be strictly increasing")
        intervention = event_values[0].intervention_frame_id
        expected_frame_ids = list(
            range(
                intervention,
                intervention + recovery_horizon + 1,
                checkpoint_step,
            )
        )
        if frame_ids != expected_frame_ids:
            raise ValueError(
                f"event {event_id} must use the exact {checkpoint_step}-frame checkpoint grid"
            )
        administrative_censor = intervention + recovery_horizon
        overlap_censor = overlap_censors.get(event_id)
        if overlap_censor is not None and not (
            intervention < overlap_censor <= administrative_censor
        ):
            raise ValueError(
                f"event {event_id} overlap censor frame must be inside its event horizon"
            )
        censor_frame = (
            overlap_censor if overlap_censor is not None else administrative_censor
        )
        if event_id in unobservable_events:
            event_results[event_id] = {
                "intervention_frame_id": intervention,
                "frame_ids": frame_ids,
                "background_observable": False,
                "recovered": None,
                "recovery_frames": None,
                "right_censored": False,
                "censor_frame": None,
                "overlapping_intervention": overlap_censor is not None,
                "censor_reason": "unobservable_revealed_target",
            }
            continue
        eligible_values = [
            value
            for value in event_values
            if overlap_censor is None or value.frame_id < overlap_censor
        ]
        recovered_frame = sustained_recovery_frame(
            eligible_values,
            threshold=threshold,
            consecutive=consecutive,
        )
        recovered = recovered_frame is not None
        recovery_frames = (
            int(recovered_frame - intervention)
            if recovered_frame is not None
            else recovery_horizon
        )
        event_results[event_id] = {
            "intervention_frame_id": intervention,
            "frame_ids": frame_ids,
            "background_observable": True,
            "recovered": recovered,
            "recovery_frames": recovery_frames,
            "right_censored": not recovered,
            "censor_frame": censor_frame,
            "overlapping_intervention": overlap_censor is not None,
            "censor_reason": (
                "overlapping_intervention"
                if overlap_censor is not None and not recovered
                else "administrative_horizon"
                if not recovered
                else None
            ),
        }

    event_values = list(event_results.values())
    observable_ids = set(grouped) - unobservable_events
    observable_frame_values = [
        value for value in values if value.event_id in observable_ids
    ]
    observable_event_values = [
        value for value in event_values if value["background_observable"]
    ]
    return {
        "current_miou": fmean(value.current_miou for value in values),
        "ghost_rate": fmean(value.ghost_rate for value in values),
        "background_f5": fmean(
            value.background_f5 for value in observable_frame_values
        ),
        "recovery_frames": fmean(
            value["recovery_frames"] for value in observable_event_values
        ),
        "recovery_background_f5": threshold,
        "recovery_consecutive": consecutive,
        "checkpoint_step_frames": checkpoint_step,
        "recovery_horizon_frames": recovery_horizon,
        "event_count": len(event_values),
        "background_observable_event_count": len(observable_event_values),
        "unobservable_revealed_target_event_count": len(unobservable_events),
        "recovered_event_count": sum(
            bool(value["recovered"]) for value in observable_event_values
        ),
        "censored_event_count": sum(
            value["right_censored"] for value in observable_event_values
        ),
        "events": event_results,
    }
