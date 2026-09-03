from __future__ import annotations

import pytest

from src.evaluation.benchmark_alignment import (
    classify_frontend_gap,
    classify_rank_mismatch,
)


def test_frontend_classification_uses_majority_at_fifty_percent() -> None:
    result = classify_frontend_gap(
        {"object_f1": 0.8, "dynamic_f1": 0.7, "change_f1": 0.6},
        {"object_f1": 0.2, "dynamic_f1": 0.3, "change_f1": 0.4},
        {"object_f1": 0.6, "dynamic_f1": 0.55, "change_f1": 0.45},
    )

    assert result.status == "FRONTEND_DOMINATED"
    assert result.metrics["object_f1"].closure == pytest.approx(2 / 3)
    assert result.metrics["dynamic_f1"].closure == pytest.approx(0.625)
    assert result.metrics["change_f1"].closure == pytest.approx(0.25)


def test_frontend_classification_is_inconclusive_without_two_conditions() -> None:
    result = classify_frontend_gap(
        None,
        {"object_f1": 0.2, "dynamic_f1": 0.3},
        None,
    )

    assert result.status == "INCONCLUSIVE_MISSING_CONDITION"
    assert result.metrics == {}


def test_single_favorable_cell_is_not_material_mismatch() -> None:
    result = classify_rank_mismatch(
        {
            "A6": {"object_f1": 0.40, "change_f1": 0.20},
            "P5": {"object_f1": 0.39, "change_f1": 0.21},
        },
        {
            "A6": {"object_f1": 0.40, "change_f1": 0.20},
            "P5": {"object_f1": 0.41, "change_f1": 0.21},
        },
    )

    assert not result.material
    assert result.direction_reversal_count == 1
    assert result.official_ranks["object_f1"] == ("A6", "P5")
    assert result.current_ranks["object_f1"] == ("P5", "A6")
    assert result.pairwise_deltas["P5"]["object_f1"]["reversal"] is True


def test_multiple_systematic_reversals_are_material() -> None:
    result = classify_rank_mismatch(
        {
            "A6": {"object_f1": 0.40, "change_f1": 0.30},
            "P5": {"object_f1": 0.35, "change_f1": 0.25},
        },
        {
            "A6": {"object_f1": 0.40, "change_f1": 0.30},
            "P5": {"object_f1": 0.45, "change_f1": 0.35},
        },
    )

    assert result.material
    assert result.systematic_direction_reversal
