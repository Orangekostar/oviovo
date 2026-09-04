from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.two_visit_current_metrics import (
    IdentityRelation,
    RegionEvaluationInput,
    evaluate_identity,
    evaluate_regions,
)


def _points(values: list[float]) -> np.ndarray:
    return np.asarray(
        [[value, 0.0, 0.0] for value in values], dtype=np.float64
    ).reshape(-1, 3)


def test_unobserved_completeness_and_observed_stale_precision_are_separate() -> None:
    fixture = RegionEvaluationInput(
        predicted_current_xyz=_points([0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 13.0, 14.0]),
        ground_truth_current_xyz=_points([0.0, 1.0, 2.0, 3.0]),
        ground_truth_t1_unobserved_mask=np.asarray([True, True, True, True]),
        retained_t0_xyz=_points([10.0, 11.0, 12.0, 13.0, 14.0]),
        retained_t0_t1_observed_mask=np.asarray([True, True, True, True, True]),
        confirmed_free_xyz=_points([10.0]),
        predicted_background_xyz=_points([20.0, 21.0]),
        ground_truth_background_xyz=_points([20.0, 21.0, 22.0]),
        distance_threshold_m=0.05,
    )

    result = evaluate_regions(fixture)

    assert result.unobserved_region_recall == 0.75
    assert result.observed_region_stale_precision == 0.80
    assert result.surface_recall == 0.75
    assert result.total_current_surface_coverage == result.surface_recall
    assert result.background_precision == 1.0
    assert result.background_recall == pytest.approx(2.0 / 3.0)
    assert result.confirmed_free_ghost_count == 1
    assert result.observed_historical_count == 5


def test_region_metrics_fail_closed_on_mask_or_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="unobserved mask"):
        RegionEvaluationInput(
            predicted_current_xyz=_points([0.0]),
            ground_truth_current_xyz=_points([0.0, 1.0]),
            ground_truth_t1_unobserved_mask=np.asarray([True]),
            retained_t0_xyz=_points([]),
            retained_t0_t1_observed_mask=np.asarray([], dtype=np.bool_),
            confirmed_free_xyz=_points([]),
            predicted_background_xyz=_points([]),
            ground_truth_background_xyz=_points([]),
        )

    values = _points([0.0])
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        RegionEvaluationInput(
            predicted_current_xyz=values,
            ground_truth_current_xyz=_points([0.0]),
            ground_truth_t1_unobserved_mask=np.asarray([False]),
            retained_t0_xyz=_points([]),
            retained_t0_t1_observed_mask=np.asarray([], dtype=np.bool_),
            confirmed_free_xyz=_points([]),
            predicted_background_xyz=_points([]),
            ground_truth_background_xyz=_points([]),
        )


def test_five_centimeter_boundary_matches_common_v2_strict_threshold() -> None:
    result = evaluate_regions(
        RegionEvaluationInput(
            predicted_current_xyz=_points([0.05]),
            ground_truth_current_xyz=_points([0.0]),
            ground_truth_t1_unobserved_mask=np.asarray([True]),
            retained_t0_xyz=_points([]),
            retained_t0_t1_observed_mask=np.asarray([], dtype=np.bool_),
            confirmed_free_xyz=_points([]),
            predicted_background_xyz=_points([]),
            ground_truth_background_xyz=_points([]),
            distance_threshold_m=0.05,
        )
    )

    assert result.surface_precision == 0.0
    assert result.surface_recall == 0.0


def test_empty_or_inapplicable_identity_metric_is_na_not_zero() -> None:
    result = evaluate_identity((), None)

    assert result.available is False
    assert result.persistent_precision is None
    assert result.persistent_recall is None
    assert result.moved_accuracy is None
    assert result.appeared_precision is None
    assert result.removed_recall is None
    assert result.unavailable_reason == "identity_ground_truth_unavailable"


def test_identity_metrics_preserve_na_for_absent_classes() -> None:
    persistent = IdentityRelation(
        t0_entity_ids=("old-chair",),
        t1_entity_ids=("new-chair",),
        state="persistent_moved",
    )
    appeared = IdentityRelation(
        t0_entity_ids=(),
        t1_entity_ids=("lamp",),
        state="appeared",
    )

    result = evaluate_identity((persistent, appeared), (persistent, appeared))

    assert result.available is True
    assert result.persistent_precision == 1.0
    assert result.persistent_recall == 1.0
    assert result.moved_accuracy == 1.0
    assert result.appeared_precision == 1.0
    assert result.appeared_recall == 1.0
    assert result.removed_precision is None
    assert result.removed_recall is None
    assert result.split_accuracy is None
    assert result.merge_accuracy is None
