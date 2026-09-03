from __future__ import annotations

import pytest

from src.evaluation.panoptic_flat_current import (
    FlatGroundTruth,
    FlatPrediction,
    FlatProtocolError,
    aggregate_flat_results,
    evaluate_flat_current,
)

GT = (
    FlatGroundTruth(0, "A", True, None, None),
    FlatGroundTruth(0, "B", True, None, None),
    FlatGroundTruth(2, "A", True, "moved", 2),
    FlatGroundTruth(2, "B", False, "removed", 2),
    FlatGroundTruth(2, "C", True, "added", 2),
)

PREDICTIONS = (
    FlatPrediction(0, "track-a", "A", 0, 5, 1, 1, 4, 0, 1),
    FlatPrediction(2, "track-a", "A", 1, 6, 1, 1, 5, 1, 0),
    FlatPrediction(2, "track-c", "C", 0, 4, 1, 0, 3, 0, 1),
    FlatPrediction(2, "stale-b", "B", 1, 0, 2, 0, 0, 1, 0),
)


def test_flat_conditions_cannot_be_aggregated_together() -> None:
    oracle = evaluate_flat_current(
        GT,
        predictions=PREDICTIONS,
        evaluation_frame=2,
        condition="Flat-GT-Panoptic",
    )
    predicted = evaluate_flat_current(
        GT,
        predictions=PREDICTIONS,
        evaluation_frame=2,
        condition="Flat-Predicted-Panoptic",
    )

    with pytest.raises(FlatProtocolError, match="input condition"):
        aggregate_flat_results((oracle, predicted))


def test_same_condition_results_are_micro_aggregated() -> None:
    first = evaluate_flat_current(
        GT,
        predictions=PREDICTIONS,
        evaluation_frame=2,
        condition="Flat-Predicted-Panoptic",
    )

    combined = aggregate_flat_results((first, first))

    assert combined.run_count == 2
    assert combined.current_object_precision == pytest.approx(2 / 3)
    assert combined.current_geometry_f5cm == pytest.approx(20 / 25)


def test_current_metrics_count_stale_geometry_and_recovery() -> None:
    result = evaluate_flat_current(
        GT,
        predictions=PREDICTIONS,
        evaluation_frame=2,
        condition="Flat-Predicted-Panoptic",
    )

    assert result.oracle is False
    assert result.current_object_precision == pytest.approx(2 / 3)
    assert result.current_object_recall == pytest.approx(1.0)
    assert result.change_recall == {
        "moved": pytest.approx(1.0),
        "added": pytest.approx(1.0),
        "removed": pytest.approx(0.0),
    }
    assert result.stale_geometry_fp == 2
    assert result.current_geometry_f5cm == pytest.approx(20 / 25)
    assert result.background_free_space_recall == pytest.approx(8 / 9)
    assert result.recovery_latency_frames == pytest.approx(0.0)


def test_predictions_after_checkpoint_are_rejected() -> None:
    with pytest.raises(FlatProtocolError, match="future frame"):
        evaluate_flat_current(
            GT,
            predictions=(*PREDICTIONS, FlatPrediction(3, "future", "A", 0, 0, 0, 0, 0, 0, 0)),
            evaluation_frame=2,
            condition="Flat-Predicted-Panoptic",
        )


def test_recovery_latency_is_measured_from_change_frame() -> None:
    delayed = (
        FlatGroundTruth(0, "A", True, None, None),
        FlatGroundTruth(1, "A", True, "moved", 1),
        FlatGroundTruth(2, "A", True, "moved", 1),
    )
    prediction = FlatPrediction(2, "a", "A", 0, 1, 0, 0, 1, 0, 0)

    result = evaluate_flat_current(
        delayed,
        predictions=(prediction,),
        evaluation_frame=2,
        condition="Flat-GT-Panoptic",
    )

    assert result.recovery_latency_frames == pytest.approx(1.0)
