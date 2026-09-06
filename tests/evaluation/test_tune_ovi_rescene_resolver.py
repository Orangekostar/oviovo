from __future__ import annotations

from scripts.evaluation.tune_ovi_rescene_resolver import (
    aggregate_candidate_metrics,
    select_candidate,
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
