from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.baselines.dynamic_metrics import (
    DynamicFrameMetrics,
    aggregate_dynamic_metrics,
    evaluate_dynamic_frame,
)


def _points(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 3)


@pytest.mark.parametrize(
    (
        "predicted_labels",
        "predicted_changed",
        "predicted_background",
        "expected",
    ),
    [
        (
            [1, 2],
            [],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            (1.0, 0.0, 1.0),
        ),
        ([-1, -1], [], [], (0.0, 0.0, 0.0)),
        ([-1, -1], [[10.0, 0.0, 0.0]], [], (0.0, 1.0, 0.0)),
        (
            [1, -1],
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            (0.5, 0.5, 2.0 / 3.0),
        ),
    ],
)
def test_dynamic_frame_perfect_empty_stale_and_partial(
    predicted_labels,
    predicted_changed,
    predicted_background,
    expected,
) -> None:
    result = evaluate_dynamic_frame(
        event_id="remove-chair",
        frame_id=12,
        intervention_frame_id=10,
        ground_truth_semantic_ids=np.asarray([1, 2], dtype=np.int64),
        predicted_semantic_ids=np.asarray(predicted_labels, dtype=np.int64),
        valid_semantic_ids={1, 2},
        predicted_object_points_in_changed_region=_points(predicted_changed),
        confirmed_free_space_points=_points([[10.0, 0.0, 0.0]]),
        predicted_background_points_in_revealed_region=_points(predicted_background),
        ground_truth_revealed_background_points=_points(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        distance_threshold_m=0.05,
    )

    assert result.current_miou == pytest.approx(expected[0])
    assert result.ghost_rate == pytest.approx(expected[1])
    assert result.background_f5 == pytest.approx(expected[2])


def _frame(event: str, frame_id: int, intervention: int, background_f5: float):
    return DynamicFrameMetrics(
        event_id=event,
        frame_id=frame_id,
        intervention_frame_id=intervention,
        current_miou=0.8,
        ghost_rate=0.1,
        background_f5=background_f5,
        predicted_changed_object_count=10,
        ghost_count=1,
        predicted_revealed_background_count=10,
        ground_truth_revealed_background_count=10,
    )


def test_aggregate_dynamic_metrics_reports_recovered_and_censored_events() -> None:
    recovered_scores = [0.2, 0.95, 0.96, 0.97, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    censored_scores = [0.8] * 10
    result = aggregate_dynamic_metrics(
        [
            *(
                _frame("a", 10 + 50 * index, 10, score)
                for index, score in enumerate(recovered_scores)
            ),
            *(
                _frame("b", 20 + 50 * index, 20, score)
                for index, score in enumerate(censored_scores)
            ),
        ],
        recovery_background_f5=0.9,
    )

    assert result["current_miou"] == pytest.approx(0.8)
    assert result["ghost_rate"] == pytest.approx(0.1)
    assert result["background_f5"] == pytest.approx(
        sum(recovered_scores + censored_scores) / 20
    )
    assert result["recovery_frames"] == pytest.approx((50 + 450) / 2)
    assert result["recovered_event_count"] == 1
    assert result["censored_event_count"] == 1
    assert result["event_count"] == 2
    assert result["events"]["b"]["recovery_frames"] == 450
    assert result["events"]["b"]["recovered"] is False


def test_aggregate_dynamic_metrics_rejects_duplicate_or_inconsistent_events() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        aggregate_dynamic_metrics(
            [_frame("a", 10, 10, 0.1), _frame("a", 10, 10, 0.2)]
        )
    with pytest.raises(ValueError, match="intervention"):
        aggregate_dynamic_metrics(
            [_frame("a", 10, 10, 0.1), _frame("a", 11, 9, 0.2)]
        )
