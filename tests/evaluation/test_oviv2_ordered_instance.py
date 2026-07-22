from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.oviv2_instance_head import (
    ProjectedInstanceHypothesis,
    evaluate_projected_instance_hypotheses,
)
from src.evaluation.oviv2_replica import ReplicaGroundTruth, _average_precision


MODULE_PATH = Path(__file__).parents[2] / "src" / "evaluation" / "oviv2_ordered_instance.py"


def _api():
    from src.evaluation.oviv2_ordered_instance import (
        OrderedPredictionComposition,
        compose_ordered_predictions,
        evaluate_ordered_projected_hypotheses,
        fingerprint_predictions,
        projected_iou,
    )

    return {
        "composition": OrderedPredictionComposition,
        "compose": compose_ordered_predictions,
        "evaluate": evaluate_ordered_projected_hypotheses,
        "fingerprint": fingerprint_predictions,
        "iou": projected_iou,
    }


def _hypothesis(identifier: str, mask, *, score: float = 0.5):
    return ProjectedInstanceHypothesis(
        hypothesis_id=identifier,
        entity_id=10,
        semantic_id=4,
        kind="parent",
        score=score,
        mask=np.asarray(mask, dtype=bool),
    )


def _ground_truth(instance_ids=(1, 1, 1, 2, 2, 2), semantic_ids=None):
    count = len(instance_ids)
    return ReplicaGroundTruth(
        vertices_xyz=np.column_stack((np.arange(count, dtype=np.float32), np.zeros((count, 2)))),
        semantic_ids=np.full(count, 4, dtype=np.int64)
        if semantic_ids is None
        else np.asarray(semantic_ids, dtype=np.int64),
        instance_ids=np.asarray(instance_ids, dtype=np.int64),
    )


def test_ordered_instance_module_exists() -> None:
    assert MODULE_PATH.is_file()


def test_composition_freezes_deduplicated_primary_prefix_and_separates_suffix() -> None:
    api = _api()
    primary_a = _hypothesis("p:a", [1, 1, 1, 0, 0, 0], score=0.9)
    duplicate_a = _hypothesis("p:duplicate-a", [1, 1, 1, 0, 0, 0], score=0.8)
    primary_b = _hypothesis("p:b", [0, 0, 0, 1, 1, 1], score=0.7)
    unsupported = _hypothesis("p:small", [1, 0, 0, 0, 0, 0], score=1.0)
    novel_c = _hypothesis("vc:c", [1, 1, 0, 1, 0, 0], score=0.8)
    duplicate_b = _hypothesis("vc:duplicate-b", [0, 0, 0, 1, 1, 1], score=0.9)

    composition = api["compose"](
        primary=(primary_a, duplicate_a, primary_b, unsupported),
        suffix=(novel_c, duplicate_b),
        min_instance_vertices=2,
        primary_deduplication_iou=0.7,
        suffix_deduplication_iou=0.9,
    )

    assert composition.primary_ids == ("p:a", "p:b")
    assert composition.primary[0] is primary_a
    assert composition.primary[1] is primary_b
    assert composition.primary_sha256 == api["fingerprint"](composition.primary)
    assert composition.ordered_ids[:2] == composition.primary_ids
    assert composition.ordered_ids[2:] == ("vc:c",)
    assert composition.rejected_suffix_ids == ("vc:duplicate-b",)
    assert max(item.score for item in composition.suffix) < min(
        item.score for item in composition.primary
    )
    assert isinstance(composition.primary, tuple)
    with pytest.raises(FrozenInstanceError):
        composition.primary = ()


def test_suffix_screening_is_deterministic_and_checks_earlier_kept_suffixes() -> None:
    api = _api()
    composition = api["compose"](
        primary=(_hypothesis("p", [1, 1, 0, 0, 0, 0], score=0.8),),
        suffix=(
            _hypothesis("vc:z", [0, 0, 1, 1, 0, 0], score=0.5),
            _hypothesis("vc:a", [0, 0, 1, 1, 0, 0], score=0.5),
            _hypothesis("vc:b", [0, 0, 0, 0, 1, 1], score=0.6),
        ),
        min_instance_vertices=2,
        primary_deduplication_iou=0.7,
        suffix_deduplication_iou=0.9,
    )

    assert composition.ordered_ids == ("p", "vc:b", "vc:a")
    assert composition.rejected_suffix_ids == ("vc:z",)
    assert all(
        left.score >= right.score
        for left, right in zip(composition.suffix, composition.suffix[1:])
    )


def test_iou_and_fingerprint_are_ordered_lossless_audit_values() -> None:
    api = _api()
    left = _hypothesis("a", [1, 1, 0, 0, 0, 0], score=0.5)
    right = _hypothesis("b", [1, 0, 1, 0, 0, 0], score=0.5)

    assert api["iou"](left, right) == pytest.approx(1 / 3)
    assert api["fingerprint"]((left, right)) != api["fingerprint"]((right, left))
    assert api["fingerprint"]((left,)) != api["fingerprint"](
        (_hypothesis("a", [1, 1, 0, 0, 0, 0], score=np.nextafter(0.5, 1.0)),)
    )


def test_composition_rejects_invalid_inputs_and_unseparable_primary_or_suffix() -> None:
    api = _api()
    good = _hypothesis("p", [1, 1, 0, 0, 0, 0], score=0.8)
    zero = _hypothesis("zero", [0, 0, 1, 1, 0, 0], score=0.0)
    short = _hypothesis("short", [1, 1, 0], score=0.5)
    unsupported = _hypothesis("small", [1, 0, 0, 0, 0, 0], score=0.8)

    for kwargs in (
        {"min_instance_vertices": 0},
        {"primary_deduplication_iou": 0.0},
        {"suffix_deduplication_iou": 1.1},
    ):
        with pytest.raises((TypeError, ValueError)):
            api["compose"]((good,), (), **kwargs)
    with pytest.raises(ValueError, match="equal length"):
        api["compose"]((good,), (short,), min_instance_vertices=2,
                       primary_deduplication_iou=0.7, suffix_deduplication_iou=0.9)
    with pytest.raises(TypeError):
        api["compose"]((object(),), (), min_instance_vertices=2,
                       primary_deduplication_iou=0.7, suffix_deduplication_iou=0.9)
    with pytest.raises(ValueError, match="primary"):
        api["compose"]((), (), min_instance_vertices=2,
                       primary_deduplication_iou=0.7, suffix_deduplication_iou=0.9)
    with pytest.raises(ValueError, match="primary"):
        api["compose"]((unsupported,), (), min_instance_vertices=2,
                       primary_deduplication_iou=0.7, suffix_deduplication_iou=0.9)
    with pytest.raises(ValueError, match="positive"):
        api["compose"]((zero,), (), min_instance_vertices=2,
                       primary_deduplication_iou=0.7, suffix_deduplication_iou=0.9)
    with pytest.raises(ValueError, match="suffix"):
        api["compose"]((good,), (zero,), min_instance_vertices=2,
                       primary_deduplication_iou=0.7, suffix_deduplication_iou=0.9)


def test_ordered_evaluation_matches_existing_evaluator_when_given_its_order() -> None:
    api = _api()
    ground_truth = _ground_truth()
    projected = (
        _hypothesis("b", [0, 0, 0, 1, 1, 1], score=0.7),
        _hypothesis("a", [1, 1, 1, 0, 0, 0], score=0.9),
        _hypothesis("duplicate-a", [1, 1, 1, 0, 0, 0], score=0.8),
    )
    formal = evaluate_projected_instance_hypotheses(
        projected, ground_truth, instance_semantic_ids={4}, min_instance_vertices=2,
        deduplication_iou_threshold=0.9,
    )
    ordered = tuple(
        next(item for item in projected if item.hypothesis_id == identifier)
        for identifier in formal["prediction_hypothesis_ids"]
    )

    result = api["evaluate"](
        ordered, ground_truth, instance_semantic_ids={4}, min_instance_vertices=2
    )

    for key in (
        "ap25",
        "ap50",
        "recall25",
        "recall50",
        "predicted_instance_count",
        "ground_truth_instance_count",
        "prediction_hypothesis_ids",
        "prediction_confidences",
    ):
        assert result[key] == formal[key]
    assert result["raw_hypothesis_count"] == 2
    assert result["supported_hypothesis_count"] == 2


def test_ordered_evaluation_does_not_reorder_and_rejects_unsupported_or_duplicate_ids() -> None:
    api = _api()
    ground_truth = _ground_truth()
    ordered = (
        _hypothesis("late-score", [1, 1, 1, 0, 0, 0], score=0.1),
        _hypothesis("early-score", [0, 0, 0, 1, 1, 1], score=0.9),
        _hypothesis("same-score", [1, 1, 1, 1, 1, 1], score=0.9),
    )
    result = api["evaluate"](
        ordered, ground_truth, instance_semantic_ids={4}, min_instance_vertices=2
    )
    assert result["prediction_hypothesis_ids"] == [
        "late-score", "early-score", "same-score"
    ]
    assert result["prediction_confidences"] == [0.1, 0.9, 0.9]

    with pytest.raises(ValueError, match="support"):
        api["evaluate"]((_hypothesis("small", [1, 0, 0, 0, 0, 0]),), ground_truth,
                        instance_semantic_ids={4}, min_instance_vertices=2)
    with pytest.raises(ValueError, match="duplicate"):
        api["evaluate"]((ordered[0], _hypothesis("late-score", [0, 0, 0, 1, 1, 1])),
                        ground_truth, instance_semantic_ids={4}, min_instance_vertices=2)


def test_ordered_evaluation_validates_ground_truth_masks_and_semantic_ids() -> None:
    api = _api()
    valid = _hypothesis("valid", [1, 1, 1, 0, 0, 0])
    with pytest.raises(TypeError, match="ground_truth"):
        api["evaluate"]((valid,), object(), instance_semantic_ids={4}, min_instance_vertices=2)
    with pytest.raises(ValueError, match="size"):
        api["evaluate"]((valid,), _ground_truth((1, 1, 1)), instance_semantic_ids={4},
                        min_instance_vertices=2)
    with pytest.raises((TypeError, ValueError)):
        api["evaluate"]((valid,), _ground_truth(), instance_semantic_ids={0}, min_instance_vertices=2)


def test_ordered_evaluation_handles_zero_ground_truth() -> None:
    api = _api()
    result = api["evaluate"](
        (_hypothesis("prediction", [1, 1, 1, 0, 0, 0]),),
        _ground_truth(semantic_ids=(99, 99, 99, 99, 99, 99)),
        instance_semantic_ids={4}, min_instance_vertices=2,
    )
    assert result["ap25"] == result["ap50"] == 0.0
    assert result["recall25"] == result["recall50"] == 0.0
    assert result["ground_truth_instance_count"] == 0


def test_voc_ap_never_decreases_when_binary_suffixes_are_appended() -> None:
    cache: dict[tuple[tuple[int, ...], int], float] = {}

    def ap(bits: tuple[int, ...], gt_count: int) -> float:
        key = (bits, gt_count)
        if key not in cache:
            values = np.asarray(bits, dtype=np.float64)
            cache[key] = _average_precision(values, 1.0 - values, gt_count)
        return cache[key]

    for gt_count in range(1, 5):
        for prefix_length in range(9):
            for prefix in product((0, 1), repeat=prefix_length):
                prefix_ap = ap(prefix, gt_count)
                for suffix_length in range(9):
                    for suffix in product((0, 1), repeat=suffix_length):
                        assert ap(prefix + suffix, gt_count) + 1e-15 >= prefix_ap
