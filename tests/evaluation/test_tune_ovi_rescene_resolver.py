from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.evaluation.tune_ovi_rescene_resolver import (
    aggregate_candidate_metrics,
    build_resolver_effect_rows,
    select_candidate,
    summarize_fixed_threshold_transfer,
    write_resolver_effect_csv,
    write_fixed_threshold_transfer_csv,
)


def _threshold(*, predictions: int, true_positives: int, ground_truth: int) -> dict[str, int]:
    return {
        "paired_prediction_edges": predictions,
        "true_positive_edges": true_positives,
        "false_positive_edges": predictions - true_positives,
        "ground_truth_persistent_edges": ground_truth,
        "rigid_true_positive_edges": 1,
        "rigid_ground_truth_edges": 2,
        "false_reid_count": 0,
        "same_class_mismatch_count": 0,
        "unmatched_endpoint_edge_count": predictions - true_positives,
    }


def _pair(first: tuple[int, int, int], second: tuple[int, int, int]):
    return {
        0.2: {
            "iou_0_50": _threshold(
                predictions=first[0], true_positives=first[1], ground_truth=first[2]
            ),
            "iou_0_25": _threshold(
                predictions=first[0], true_positives=first[1], ground_truth=first[2]
            ),
        },
        0.3: {
            "iou_0_50": _threshold(
                predictions=second[0], true_positives=second[1], ground_truth=second[2]
            ),
            "iou_0_25": _threshold(
                predictions=second[0], true_positives=second[1], ground_truth=second[2]
            ),
        },
    }


def test_aggregate_and_select_use_pooled_primary_f1() -> None:
    aggregate = aggregate_candidate_metrics(
        (
            _pair((10, 4, 20), (8, 4, 20)),
            _pair((10, 2, 10), (7, 3, 10)),
        )
    )

    assert aggregate[0.2]["iou_0_50"]["paired_precision"] == 0.3
    assert aggregate[0.2]["iou_0_50"]["paired_recall"] == 0.2
    assert aggregate[0.2]["iou_0_50"]["paired_f1"] == 0.24
    assert select_candidate(aggregate) == 0.3


def test_selection_tie_breaks_on_precision_then_lower_threshold() -> None:
    tied = {
        0.2: {"iou_0_50": {"paired_f1": 0.4, "paired_precision": 0.5, "paired_recall": 1 / 3}},
        0.3: {"iou_0_50": {"paired_f1": 0.4, "paired_precision": 0.6, "paired_recall": 0.3}},
        0.4: {"iou_0_50": {"paired_f1": 0.4, "paired_precision": 0.6, "paired_recall": 0.3}},
    }

    assert select_candidate(tied) == 0.3


def _diagnostic(
    query_id: str,
    confidence: float,
    outcome: str,
    *,
    assigned: tuple[int | None, int | None],
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query_confidence": confidence,
        "t0_entity_ids": [f"{query_id}-left"],
        "t1_entity_ids": [f"{query_id}-right"],
        "t0_assigned_gt_id": assigned[0],
        "t0_assigned_iou": 0.7 if assigned[0] is not None else None,
        "t1_assigned_gt_id": assigned[1],
        "t1_assigned_iou": 0.7 if assigned[1] is not None else None,
        "t0_best_gt_id": 10,
        "t0_best_iou": 0.8,
        "t1_best_gt_id": 10,
        "t1_best_iou": 0.8,
        "t0_best_competitor_count": 2,
        "t1_best_competitor_count": 1,
        "strict_correct": outcome == "true_positive",
        "ambiguity_aware_correct": outcome == "true_positive",
        "same_class_mismatch": False,
        "outcome": outcome,
    }


def test_resolver_effect_rows_separate_filtering_from_reassignment() -> None:
    before = (
        _diagnostic(
            "q-reassigned", 0.4, "endpoint_unmatched", assigned=(None, 10)
        ),
        _diagnostic("q-low", 0.2, "true_positive", assigned=(10, 10)),
    )
    after = (
        _diagnostic("q-reassigned", 0.4, "true_positive", assigned=(10, 10)),
    )

    rows = build_resolver_effect_rows(
        pair_id="pair-a",
        before=before,
        after=after,
        selected_threshold=0.3,
    )

    assert [row["query_id"] for row in rows] == ["q-low", "q-reassigned"]
    assert rows[0]["retained"] is False
    assert rows[0]["change_source"] == "filtered_low_confidence"
    assert rows[1]["retained"] is True
    assert rows[1]["mask_membership_unchanged"] is True
    assert rows[1]["change_source"] == "newly_true_positive_after_reassignment"
    assert rows[1]["before_t0_assigned_gt_id"] is None
    assert rows[1]["after_t0_assigned_gt_id"] == 10


def test_resolver_effect_csv_is_stable_and_refuses_overwrite(tmp_path: Path) -> None:
    rows = build_resolver_effect_rows(
        pair_id="pair-a",
        before=(
            _diagnostic("q0", 0.2, "endpoint_unmatched", assigned=(None, 10)),
        ),
        after=(),
        selected_threshold=0.3,
    )
    output = tmp_path / "effect.csv"

    write_resolver_effect_csv(output, rows)

    assert b"\r\n" not in output.read_bytes()
    with output.open(newline="", encoding="utf-8") as stream:
        loaded = list(csv.DictReader(stream))
    assert loaded[0]["pair_id"] == "pair-a"
    assert loaded[0]["retained"] == "false"
    assert loaded[0]["before_t0_assigned_gt_id"] == ""
    with pytest.raises(ValueError, match="already exists"):
        write_resolver_effect_csv(output, rows)


def _fixed_pair(
    pair_id: str,
    environment_id: str,
    *,
    predictions: int,
    true_positives: int,
    ground_truth: int,
    rigid_true_positives: int,
    rigid_ground_truth: int,
) -> dict[str, object]:
    metrics = _threshold(
        predictions=predictions,
        true_positives=true_positives,
        ground_truth=ground_truth,
    )
    metrics["rigid_true_positive_edges"] = rigid_true_positives
    metrics["rigid_ground_truth_edges"] = rigid_ground_truth
    return {
        "pair_id": pair_id,
        "environment_id": environment_id,
        "thresholds": {
            "iou_0_50": metrics,
            "iou_0_25": dict(metrics),
        },
    }


def test_fixed_threshold_transfer_reports_pair_micro_and_environment_macro() -> None:
    rows = summarize_fixed_threshold_transfer(
        (
            _fixed_pair(
                "pair-a",
                "environment-a",
                predictions=4,
                true_positives=2,
                ground_truth=5,
                rigid_true_positives=1,
                rigid_ground_truth=2,
            ),
            _fixed_pair(
                "pair-b",
                "environment-b",
                predictions=2,
                true_positives=1,
                ground_truth=1,
                rigid_true_positives=1,
                rigid_ground_truth=1,
            ),
        ),
        selected_threshold=0.3,
    )

    primary = [row for row in rows if row["instance_iou_threshold"] == 0.5]
    assert [row["aggregation"] for row in primary] == [
        "environment",
        "environment",
        "micro",
        "macro",
    ]
    assert primary[0]["paired_precision"] == 0.5
    assert primary[0]["paired_recall"] == 0.4
    assert primary[0]["paired_f1"] == pytest.approx(4 / 9)
    assert primary[2]["paired_prediction_edges"] == 6
    assert primary[2]["true_positive_edges"] == 3
    assert primary[2]["false_positive_edges"] == 3
    assert primary[2]["ground_truth_persistent_edges"] == 6
    assert primary[2]["paired_precision"] == 0.5
    assert primary[2]["paired_recall"] == 0.5
    assert primary[2]["paired_f1"] == 0.5
    assert primary[2]["rigid_recall"] == pytest.approx(2 / 3)
    assert primary[3]["paired_prediction_edges"] is None
    assert primary[3]["true_positive_edges"] is None
    assert primary[3]["paired_precision"] == 0.5
    assert primary[3]["paired_recall"] == pytest.approx(0.7)
    assert primary[3]["paired_f1"] == pytest.approx(5 / 9)
    assert primary[3]["rigid_recall"] == 0.75
    assert primary[3]["count_status"] == "NOT_APPLICABLE"
    assert primary[3]["count_null_reason"] == "ENVIRONMENT_MACRO_HAS_NO_COUNTS"


def test_fixed_threshold_transfer_rejects_non_frozen_threshold() -> None:
    pair = _fixed_pair(
        "pair-a",
        "environment-a",
        predictions=1,
        true_positives=1,
        ground_truth=1,
        rigid_true_positives=0,
        rigid_ground_truth=0,
    )

    with pytest.raises(ValueError, match="frozen threshold 0.3"):
        summarize_fixed_threshold_transfer((pair,), selected_threshold=0.2)


def test_fixed_threshold_transfer_csv_preserves_null_reason(tmp_path: Path) -> None:
    rows = summarize_fixed_threshold_transfer(
        (
            _fixed_pair(
                "pair-a",
                "environment-a",
                predictions=1,
                true_positives=1,
                ground_truth=1,
                rigid_true_positives=0,
                rigid_ground_truth=0,
            ),
        ),
        selected_threshold=0.3,
    )
    output = tmp_path / "transfer.csv"

    write_fixed_threshold_transfer_csv(output, rows)

    assert b"\r\n" not in output.read_bytes()
    with output.open(newline="", encoding="utf-8") as stream:
        loaded = list(csv.DictReader(stream))
    macro = next(
        row
        for row in loaded
        if row["aggregation"] == "macro"
        and row["instance_iou_threshold"] == "0.5"
    )
    assert macro["true_positive_edges"] == ""
    assert macro["count_status"] == "NOT_APPLICABLE"
    assert macro["count_null_reason"] == "ENVIRONMENT_MACRO_HAS_NO_COUNTS"
    with pytest.raises(ValueError, match="already exists"):
        write_fixed_threshold_transfer_csv(output, rows)
