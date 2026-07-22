from __future__ import annotations

import pytest

from src.evaluation.baselines.dynamic_metrics import (
    DynamicFrameMetrics,
    aggregate_dynamic_metrics,
    sustained_recovery_frame,
)


def _frame(
    event_id: str,
    frame_id: int,
    intervention_frame_id: int,
    background_f5: float,
) -> DynamicFrameMetrics:
    return DynamicFrameMetrics(
        event_id=event_id,
        frame_id=frame_id,
        intervention_frame_id=intervention_frame_id,
        current_miou=0.8,
        ghost_rate=0.1,
        background_f5=background_f5,
        predicted_changed_object_count=10,
        ghost_count=1,
        predicted_revealed_background_count=10,
        ground_truth_revealed_background_count=10,
    )


def _event(
    values: list[float], *, event_id: str = "event", intervention: int = 100
) -> list[DynamicFrameMetrics]:
    assert len(values) == 10
    return [
        _frame(event_id, intervention + 50 * index, intervention, value)
        for index, value in enumerate(values)
    ]


def test_sustained_recovery_requires_three_consecutive_checkpoints() -> None:
    interrupted = _event([0.1, 0.91, 0.95, 0.8, 0.92, 0.93, 0.94, 0.2, 0.2, 0.2])

    assert sustained_recovery_frame(interrupted) == 300
    assert sustained_recovery_frame(interrupted[:6]) is None


def test_common_v2_uses_fixed_horizon_for_right_censoring() -> None:
    result = aggregate_dynamic_metrics(
        _event([0.1] * 10),
        checkpoint_step_frames=50,
        recovery_horizon_frames=450,
        recovery_consecutive=3,
    )

    event = result["events"]["event"]
    assert event["recovered"] is False
    assert event["right_censored"] is True
    assert event["censor_frame"] == 550
    assert event["recovery_frames"] == 450
    assert event["overlapping_intervention"] is False
    assert result["recovery_frames"] == 450
    assert result["recovered_event_count"] == 0


def test_overlap_censors_before_later_intervention_without_counting_success() -> None:
    result = aggregate_dynamic_metrics(
        _event([0.1, 0.92, 0.93, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94]),
        checkpoint_step_frames=50,
        recovery_horizon_frames=450,
        recovery_consecutive=3,
        overlap_censor_frames={"event": 225},
    )

    event = result["events"]["event"]
    assert event["recovered"] is False
    assert event["right_censored"] is True
    assert event["censor_frame"] == 225
    assert event["recovery_frames"] == 450
    assert event["overlapping_intervention"] is True


def test_recovery_before_overlap_remains_observed() -> None:
    result = aggregate_dynamic_metrics(
        _event([0.91, 0.92, 0.93, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        overlap_censor_frames={"event": 275},
    )

    event = result["events"]["event"]
    assert event["recovered"] is True
    assert event["right_censored"] is False
    assert event["recovery_frames"] == 0
    assert event["censor_frame"] == 275
    assert event["overlapping_intervention"] is True


def test_common_v2_requires_exact_50_frame_checkpoint_grid() -> None:
    values = _event([0.1] * 10)
    values[4] = _frame("event", 301, 100, 0.1)

    with pytest.raises(ValueError, match="50-frame checkpoint grid"):
        aggregate_dynamic_metrics(values)


@pytest.mark.parametrize(
    "overlap_frame",
    [100, 99, 551],
)
def test_overlap_censor_must_be_inside_event_horizon(overlap_frame: int) -> None:
    with pytest.raises(ValueError, match="overlap censor frame"):
        aggregate_dynamic_metrics(
            _event([0.1] * 10), overlap_censor_frames={"event": overlap_frame}
        )


def test_overlap_censor_mapping_rejects_unknown_events() -> None:
    with pytest.raises(ValueError, match="unknown events"):
        aggregate_dynamic_metrics(
            _event([0.1] * 10), overlap_censor_frames={"other": 200}
        )


def test_unobservable_background_is_excluded_without_penalizing_current_metrics() -> None:
    observable = _event([1.0] * 10, event_id="observable")
    unobservable = _event([0.0] * 10, event_id="unobservable", intervention=600)

    result = aggregate_dynamic_metrics(
        [*observable, *unobservable],
        background_unobservable_events={"unobservable"},
    )

    assert result["current_miou"] == pytest.approx(0.8)
    assert result["ghost_rate"] == pytest.approx(0.1)
    assert result["background_f5"] == pytest.approx(1.0)
    assert result["recovery_frames"] == pytest.approx(0.0)
    assert result["event_count"] == 2
    assert result["background_observable_event_count"] == 1
    assert result["unobservable_revealed_target_event_count"] == 1
    hidden = result["events"]["unobservable"]
    assert hidden["background_observable"] is False
    assert hidden["recovered"] is None
    assert hidden["recovery_frames"] is None
    assert hidden["right_censored"] is False
    assert hidden["censor_reason"] == "unobservable_revealed_target"


def test_scene_requires_at_least_one_background_observable_event() -> None:
    with pytest.raises(ValueError, match="at least one background-observable event"):
        aggregate_dynamic_metrics(
            _event([0.0] * 10),
            background_unobservable_events={"event"},
        )
