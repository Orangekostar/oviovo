"""Causal current-state metrics for the Panoptic Mapping Flat pilot."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.evaluation.datasets.panoptic_flat import CHANGE_TYPES, input_condition


class FlatProtocolError(ValueError):
    """Raised when a Flat prediction or aggregation violates the protocol."""


@dataclass(frozen=True, slots=True)
class FlatGroundTruth:
    frame_index: int
    object_id: str
    present: bool
    change_type: str | None
    change_frame: int | None


@dataclass(frozen=True, slots=True)
class FlatPrediction:
    frame_index: int
    track_id: str
    matched_object_id: str | None
    stale_geometry_fp: int
    geometry_tp: int
    geometry_fp: int
    geometry_fn: int
    free_space_tp: int
    free_space_fp: int
    free_space_fn: int


@dataclass(frozen=True, slots=True)
class FlatCurrentResult:
    condition: str
    oracle: bool
    evaluation_frame: int | None
    run_count: int
    object_tp: int
    object_prediction_count: int
    object_gt_count: int
    change_hits: Mapping[str, int]
    change_totals: Mapping[str, int]
    stale_geometry_fp: int
    geometry_tp: int
    geometry_fp: int
    geometry_fn: int
    free_space_tp: int
    free_space_fp: int
    free_space_fn: int
    recovery_latency_sum: int
    recovered_change_count: int
    recoverable_change_count: int

    @property
    def current_object_precision(self) -> float | None:
        return _ratio(self.object_tp, self.object_prediction_count)

    @property
    def current_object_recall(self) -> float | None:
        return _ratio(self.object_tp, self.object_gt_count)

    @property
    def change_recall(self) -> dict[str, float | None]:
        return {
            change_type: _ratio(
                self.change_hits[change_type], self.change_totals[change_type]
            )
            for change_type in CHANGE_TYPES
        }

    @property
    def current_geometry_f5cm(self) -> float | None:
        denominator = 2 * self.geometry_tp + self.geometry_fp + self.geometry_fn
        return _ratio(2 * self.geometry_tp, denominator)

    @property
    def background_free_space_recall(self) -> float | None:
        return _ratio(self.free_space_tp, self.free_space_tp + self.free_space_fn)

    @property
    def recovery_latency_frames(self) -> float | None:
        return _ratio(self.recovery_latency_sum, self.recovered_change_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": "PASS",
            "protocol_id": "PANOPTIC_FLAT_CURRENT_CAUSAL_V1",
            "condition": self.condition,
            "evidence_class": "ORACLE_DIAGNOSTIC" if self.oracle else "NON_ORACLE",
            "oracle": self.oracle,
            "evaluation_frame": self.evaluation_frame,
            "run_count": self.run_count,
            "current_object_precision": self.current_object_precision,
            "current_object_recall": self.current_object_recall,
            "change_recall": self.change_recall,
            "stale_geometry_fp": self.stale_geometry_fp,
            "current_geometry_f5cm": self.current_geometry_f5cm,
            "background_free_space_recall": self.background_free_space_recall,
            "recovery_latency_frames": self.recovery_latency_frames,
            "recovered_change_count": self.recovered_change_count,
            "recoverable_change_count": self.recoverable_change_count,
            "counts": {
                "object_tp": self.object_tp,
                "object_prediction_count": self.object_prediction_count,
                "object_gt_count": self.object_gt_count,
                "change_hits": dict(self.change_hits),
                "change_totals": dict(self.change_totals),
                "geometry_tp": self.geometry_tp,
                "geometry_fp": self.geometry_fp,
                "geometry_fn": self.geometry_fn,
                "free_space_tp": self.free_space_tp,
                "free_space_fp": self.free_space_fp,
                "free_space_fn": self.free_space_fn,
            },
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise FlatProtocolError(f"{label} must be a non-negative integer")
    return value


def _validate_ground_truth(rows: Sequence[FlatGroundTruth]) -> None:
    seen: set[tuple[int, str]] = set()
    for row in rows:
        _nonnegative_integer(row.frame_index, label="ground-truth frame")
        if not isinstance(row.object_id, str) or not row.object_id:
            raise FlatProtocolError("ground-truth object ID is invalid")
        if type(row.present) is not bool:
            raise FlatProtocolError("ground-truth presence must be boolean")
        if row.change_type not in (*CHANGE_TYPES, None):
            raise FlatProtocolError("ground-truth change type is invalid")
        if row.change_type is None:
            if row.change_frame is not None:
                raise FlatProtocolError("unchanged ground truth has a change frame")
        else:
            change_frame = _nonnegative_integer(row.change_frame, label="change frame")
            if change_frame > row.frame_index:
                raise FlatProtocolError("change frame is after its ground-truth row")
            if row.change_type == "removed" and row.present:
                raise FlatProtocolError("removed ground truth cannot be present")
            if row.change_type != "removed" and not row.present:
                raise FlatProtocolError("moved or added ground truth must be present")
        key = (row.frame_index, row.object_id)
        if key in seen:
            raise FlatProtocolError("duplicate ground-truth object in frame")
        seen.add(key)


def _validate_predictions(
    rows: Sequence[FlatPrediction], *, evaluation_frame: int
) -> None:
    seen: set[tuple[int, str]] = set()
    for row in rows:
        frame = _nonnegative_integer(row.frame_index, label="prediction frame")
        if frame > evaluation_frame:
            raise FlatProtocolError("prediction from a future frame is not causal")
        if not isinstance(row.track_id, str) or not row.track_id:
            raise FlatProtocolError("prediction track ID is invalid")
        if row.matched_object_id is not None and (
            not isinstance(row.matched_object_id, str) or not row.matched_object_id
        ):
            raise FlatProtocolError("matched object ID is invalid")
        for name in (
            "stale_geometry_fp",
            "geometry_tp",
            "geometry_fp",
            "geometry_fn",
            "free_space_tp",
            "free_space_fp",
            "free_space_fn",
        ):
            _nonnegative_integer(getattr(row, name), label=name)
        key = (frame, row.track_id)
        if key in seen:
            raise FlatProtocolError("duplicate prediction track in frame")
        seen.add(key)


def evaluate_flat_current(
    ground_truth: Sequence[FlatGroundTruth],
    *,
    predictions: Sequence[FlatPrediction],
    evaluation_frame: int,
    condition: str,
) -> FlatCurrentResult:
    """Evaluate one causal checkpoint without exposing future inputs."""

    evaluation_frame = _nonnegative_integer(evaluation_frame, label="evaluation frame")
    try:
        condition_record = input_condition(condition)
    except ValueError as exc:
        raise FlatProtocolError(str(exc)) from exc
    _validate_ground_truth(ground_truth)
    _validate_predictions(predictions, evaluation_frame=evaluation_frame)
    by_frame: dict[int, dict[str, FlatGroundTruth]] = defaultdict(dict)
    for row in ground_truth:
        by_frame[row.frame_index][row.object_id] = row
    if 0 not in by_frame or evaluation_frame not in by_frame:
        raise FlatProtocolError("ground truth lacks reference or evaluation frame")
    reference_ids = {key for key, row in by_frame[0].items() if row.present}
    current = by_frame[evaluation_frame]
    current_present = {key for key, row in current.items() if row.present}
    for key, row in current.items():
        if row.change_type == "added" and key in reference_ids:
            raise FlatProtocolError("added object exists in the reference frame")
        if row.change_type in {"moved", "removed"} and key not in reference_ids:
            raise FlatProtocolError("moved or removed object is absent from the reference frame")

    current_predictions = [
        row for row in predictions if row.frame_index == evaluation_frame
    ]
    matched_once: set[str] = set()
    object_tp = 0
    for row in current_predictions:
        matched = row.matched_object_id
        if matched in current_present and matched not in matched_once:
            matched_once.add(matched)
            object_tp += 1

    change_hits: dict[str, int] = {}
    change_totals: dict[str, int] = {}
    for change_type in CHANGE_TYPES:
        targets = [row for row in current.values() if row.change_type == change_type]
        change_totals[change_type] = len(targets)
        if change_type == "removed":
            change_hits[change_type] = sum(
                not any(
                    prediction.matched_object_id == target.object_id
                    for prediction in current_predictions
                )
                for target in targets
            )
        else:
            change_hits[change_type] = sum(
                target.object_id in matched_once for target in targets
            )

    recoverable = [
        row
        for row in current.values()
        if row.present and row.change_type in {"moved", "added"}
    ]
    latency_sum = 0
    recovered = 0
    for target in recoverable:
        assert target.change_frame is not None
        recovery_frames = [
            row.frame_index
            for row in predictions
            if row.frame_index >= target.change_frame
            and row.matched_object_id == target.object_id
        ]
        if recovery_frames:
            recovered += 1
            latency_sum += min(recovery_frames) - target.change_frame

    return FlatCurrentResult(
        condition,
        condition_record.oracle,
        evaluation_frame,
        1,
        object_tp,
        len(current_predictions),
        len(current_present),
        change_hits,
        change_totals,
        sum(row.stale_geometry_fp for row in current_predictions),
        sum(row.geometry_tp for row in current_predictions),
        sum(row.geometry_fp for row in current_predictions),
        sum(row.geometry_fn for row in current_predictions),
        sum(row.free_space_tp for row in current_predictions),
        sum(row.free_space_fp for row in current_predictions),
        sum(row.free_space_fn for row in current_predictions),
        latency_sum,
        recovered,
        len(recoverable),
    )


def aggregate_flat_results(results: Sequence[FlatCurrentResult]) -> FlatCurrentResult:
    """Micro-aggregate source-compatible runs without mixing input conditions."""

    if not results:
        raise FlatProtocolError("at least one Flat result is required")
    conditions = {result.condition for result in results}
    if len(conditions) != 1:
        raise FlatProtocolError("Flat input conditions cannot be aggregated together")
    condition = results[0].condition
    if any(result.oracle != results[0].oracle for result in results):
        raise FlatProtocolError("Flat oracle classification is inconsistent")
    return FlatCurrentResult(
        condition,
        results[0].oracle,
        None,
        sum(result.run_count for result in results),
        sum(result.object_tp for result in results),
        sum(result.object_prediction_count for result in results),
        sum(result.object_gt_count for result in results),
        {
            change_type: sum(result.change_hits[change_type] for result in results)
            for change_type in CHANGE_TYPES
        },
        {
            change_type: sum(result.change_totals[change_type] for result in results)
            for change_type in CHANGE_TYPES
        },
        sum(result.stale_geometry_fp for result in results),
        sum(result.geometry_tp for result in results),
        sum(result.geometry_fp for result in results),
        sum(result.geometry_fn for result in results),
        sum(result.free_space_tp for result in results),
        sum(result.free_space_fp for result in results),
        sum(result.free_space_fn for result in results),
        sum(result.recovery_latency_sum for result in results),
        sum(result.recovered_change_count for result in results),
        sum(result.recoverable_change_count for result in results),
    )
