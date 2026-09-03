from __future__ import annotations

import pytest

from src.evaluation.rscan_temporal import (
    RScanProtocolError,
    TemporalGroundTruth,
    TemporalPrediction,
    derive_added_ids,
    evaluate_temporal_identity,
)

GT = (
    TemporalGroundTruth(0, "A", True, None),
    TemporalGroundTruth(0, "B", True, None),
    TemporalGroundTruth(0, "C", True, None),
    TemporalGroundTruth(1, "A", True, "rigid"),
    TemporalGroundTruth(1, "B", True, "rigid"),
    TemporalGroundTruth(1, "C", False, "removed"),
    TemporalGroundTruth(2, "A", True, "rigid"),
    TemporalGroundTruth(2, "B", True, "rigid"),
    TemporalGroundTruth(2, "C", True, None),
    TemporalGroundTruth(2, "D", True, None),
)

PREDICTIONS = (
    TemporalPrediction(0, "p1", "A"),
    TemporalPrediction(0, "p2", "B"),
    TemporalPrediction(0, "p5", "C"),
    TemporalPrediction(1, "p3", "A"),
    TemporalPrediction(1, "p2", "B"),
    TemporalPrediction(1, "stale", "C"),
    TemporalPrediction(2, "p3", "A"),
    TemporalPrediction(2, "p2", "A"),
    TemporalPrediction(2, "p4", "D"),
    TemporalPrediction(2, "p5", "C"),
)


def test_session_prefix_rejects_future_predictions() -> None:
    with pytest.raises(RScanProtocolError, match="future session"):
        evaluate_temporal_identity(
            GT,
            predictions=(
                *PREDICTIONS,
                TemporalPrediction(3, "future", "A"),
            ),
            evaluation_session=2,
        )


def test_exact_identity_metrics_count_switch_reid_and_change_types() -> None:
    metrics = evaluate_temporal_identity(
        GT,
        predictions=PREDICTIONS,
        evaluation_session=2,
    )

    assert metrics.id_switches == 1
    assert metrics.false_reid == 1
    assert metrics.recall_by_change_type["rigid"] == pytest.approx(0.5)
    assert metrics.recall_by_change_type["added"] == pytest.approx(1.0)
    assert metrics.stale_object_fp == 0
    assert metrics.current_object_recall == pytest.approx(3 / 4)
    assert metrics.reactivation_recall == pytest.approx(1.0)
    assert metrics.community_metrics_status == "NOT_COMPUTED_MISSING_COMMUNITY_INPUT"


def test_added_ids_are_derived_only_from_session_annotations() -> None:
    assert derive_added_ids({"A", "B"}, {"A", "B", "C"}) == frozenset({"C"})
