from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.oviv2_diagnostics import (
    diagnose_projected_labels,
    semantic_iou_from_ids,
)


def test_semantic_iou_uses_valid_ground_truth_domain_and_sorted_classes() -> None:
    result = semantic_iou_from_ids(
        predicted=[3, 2, 3, 99, 2],
        ground_truth=[2, 2, 3, 0, 7],
        valid_semantic_ids={7, 3, 2},
    )

    assert result == {
        "miou": pytest.approx((1.0 / 3.0 + 0.5 + 0.0) / 3.0),
        "per_class_iou": {
            "2": pytest.approx(1.0 / 3.0),
            "3": pytest.approx(0.5),
            "7": pytest.approx(0.0),
        },
        "evaluated_vertex_count": 4,
    }


def test_diagnostics_expose_association_failures_and_semantic_oracle() -> None:
    result = diagnose_projected_labels(
        predicted_semantic_ids=[3, 3, 3, 3, 2, 2, 0],
        predicted_entity_ids=[10, 10, 10, 10, 11, 11, 0],
        gt_semantic_ids=[2, 2, 3, 3, 3, 3, 2],
        gt_instance_ids=[1, 1, 2, 2, 2, 2, 3],
        valid_semantic_ids={2, 3},
        minimum_overlap_vertices=1,
        minimum_overlap_fraction=0.20,
    )

    assert result["headline_eligible"] is False
    assert result["entity_geometry"] == {
        "overmerged_entity_ids": [10],
        "fragmented_gt_instance_ids": [2],
        "significant_gt_instances_by_entity": {"10": [1, 2], "11": [2]},
    }
    assert result["semantic"]["oracle_entity_labels"] == {"10": 2, "11": 3}
    assert result["semantic"]["oracle_miou"] > result["semantic"]["current_miou"]


def test_diagnostics_ignore_zero_ids_and_break_majority_ties_by_smallest_label() -> None:
    result = diagnose_projected_labels(
        predicted_semantic_ids=[0, 0, 0, 0],
        predicted_entity_ids=[0, 8, 8, 9],
        gt_semantic_ids=[2, 3, 2, 0],
        gt_instance_ids=[4, 0, 5, 6],
        valid_semantic_ids={2, 3},
        minimum_overlap_vertices=1,
        minimum_overlap_fraction=0.5,
    )

    assert result["semantic"]["oracle_entity_labels"] == {"8": 2}
    assert result["entity_geometry"] == {
        "overmerged_entity_ids": [],
        "fragmented_gt_instance_ids": [],
        "significant_gt_instances_by_entity": {"8": [5], "9": [6]},
    }


def test_zero_overlap_fraction_keeps_unassociated_positive_entities() -> None:
    result = diagnose_projected_labels(
        predicted_semantic_ids=[1, 1],
        predicted_entity_ids=[7, 8],
        gt_semantic_ids=[1, 1],
        gt_instance_ids=[0, 5],
        valid_semantic_ids={1},
        minimum_overlap_vertices=1,
        minimum_overlap_fraction=0.0,
    )

    assert result["entity_geometry"]["significant_gt_instances_by_entity"] == {
        "7": [],
        "8": [5],
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"predicted_semantic_ids": [1, 2]},
        {"predicted_entity_ids": np.asarray([[1]])},
        {"gt_semantic_ids": [-1]},
        {"gt_instance_ids": [-1]},
    ],
)
def test_diagnostics_reject_invalid_label_arrays(kwargs: dict) -> None:
    arguments = {
        "predicted_semantic_ids": [1],
        "predicted_entity_ids": [1],
        "gt_semantic_ids": [1],
        "gt_instance_ids": [1],
        "valid_semantic_ids": {1},
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError):
        diagnose_projected_labels(**arguments)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"valid_semantic_ids": set()}, "non-empty"),
        ({"valid_semantic_ids": {0, 1}}, "positive"),
        ({"minimum_overlap_vertices": 0}, "minimum_overlap_vertices"),
        ({"minimum_overlap_vertices": 1.5}, "minimum_overlap_vertices"),
        ({"minimum_overlap_fraction": -0.01}, "minimum_overlap_fraction"),
        ({"minimum_overlap_fraction": 1.01}, "minimum_overlap_fraction"),
        ({"minimum_overlap_fraction": np.nan}, "minimum_overlap_fraction"),
    ],
)
def test_diagnostics_reject_invalid_domains_and_thresholds(
    kwargs: dict,
    message: str,
) -> None:
    arguments = {
        "predicted_semantic_ids": [1],
        "predicted_entity_ids": [1],
        "gt_semantic_ids": [1],
        "gt_instance_ids": [1],
        "valid_semantic_ids": {1},
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        diagnose_projected_labels(**arguments)


def test_empty_arrays_produce_zero_metrics_and_empty_diagnostics() -> None:
    semantic = semantic_iou_from_ids([], [], valid_semantic_ids={1})
    diagnostic = diagnose_projected_labels(
        [],
        [],
        [],
        [],
        valid_semantic_ids={1},
    )

    assert semantic == {
        "miou": 0.0,
        "per_class_iou": {},
        "evaluated_vertex_count": 0,
    }
    assert diagnostic == {
        "headline_eligible": False,
        "semantic": {
            "current_miou": 0.0,
            "oracle_miou": 0.0,
            "oracle_entity_labels": {},
        },
        "entity_geometry": {
            "overmerged_entity_ids": [],
            "fragmented_gt_instance_ids": [],
            "significant_gt_instances_by_entity": {},
        },
    }
