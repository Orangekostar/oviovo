from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.baselines.static_metrics import evaluate_static_snapshot
from src.evaluation.contracts import EntityPrediction, GroundTruthSnapshot, MapSnapshot


def _entity(entity_id: str, points: list[list[float]], label: str, score: float) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(points, dtype=np.float32),
        semantic_embedding=None,
        semantic_label=label,
        semantic_score=score,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=1.0,
    )


def _ground_truth() -> GroundTruthSnapshot:
    return GroundTruthSnapshot(
        scene_id="room0",
        timestamp=1.0,
        points_xyz=np.asarray(
            [
                [0.00, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [1.00, 0.0, 0.0],
                [1.02, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        semantic_labels=np.asarray(["chair", "chair", "table", "table"], dtype=object),
        instance_ids=np.asarray([1001, 1001, 2001, 2001], dtype=np.int64),
    )


def _snapshot(*, include_false_positive: bool = False) -> MapSnapshot:
    entities = [
        _entity("chair", [[0.00, 0.0, 0.0], [0.02, 0.0, 0.0]], "chair", 0.9),
        _entity("table", [[1.00, 0.0, 0.0], [1.02, 0.0, 0.0]], "table", 0.8),
    ]
    if include_false_positive:
        entities.insert(0, _entity("stale", [[4.0, 0.0, 0.0]], "chair", 1.0))
    return MapSnapshot(
        method="baseline",
        scene_id="room0",
        timestamp=1.0,
        entities=entities,
        background_xyz=None,
        scope="current",
    )


def test_perfect_static_snapshot_scores_one() -> None:
    metrics = evaluate_static_snapshot(
        _snapshot(),
        _ground_truth(),
        semantic_vocabulary=("chair", "table"),
        instance_vocabulary=("chair", "table"),
        distance_threshold_m=0.05,
        min_instance_points=1,
    )

    assert metrics["semantic"]["miou"] == pytest.approx(1.0)
    assert metrics["semantic"]["macc"] == pytest.approx(1.0)
    assert metrics["semantic"]["f_miou"] == pytest.approx(1.0)
    assert metrics["instance"]["ap25"] == pytest.approx(1.0)
    assert metrics["instance"]["ap50"] == pytest.approx(1.0)
    assert metrics["geometry"]["f5"] == pytest.approx(1.0)


def test_empty_and_stale_predictions_are_penalized() -> None:
    empty = MapSnapshot("baseline", "room0", 1.0, [], None, "current")
    empty_metrics = evaluate_static_snapshot(
        empty,
        _ground_truth(),
        semantic_vocabulary=("chair", "table"),
        instance_vocabulary=("chair", "table"),
        min_instance_points=1,
    )
    stale_metrics = evaluate_static_snapshot(
        _snapshot(include_false_positive=True),
        _ground_truth(),
        semantic_vocabulary=("chair", "table"),
        instance_vocabulary=("chair", "table"),
        min_instance_points=1,
    )

    assert empty_metrics["semantic"]["miou"] == 0.0
    assert empty_metrics["instance"]["ap25"] == 0.0
    assert empty_metrics["geometry"]["f5"] == 0.0
    assert stale_metrics["instance"]["ap50"] < 1.0
    assert stale_metrics["geometry"]["precision"] < 1.0


def test_class_agnostic_instance_metric_filters_gt_domain_not_predictions() -> None:
    ground_truth = _ground_truth()
    ground_truth.instance_ids = np.asarray([1001, 1001, 2001, -1], dtype=np.int64)
    metrics = evaluate_static_snapshot(
        _snapshot(),
        ground_truth,
        semantic_vocabulary=("chair", "table"),
        instance_vocabulary=("chair",),
        min_instance_points=2,
    )

    assert metrics["instance"]["ground_truth_instance_count"] == 1
    assert metrics["instance"]["predicted_instance_count"] == 2


def test_static_metrics_marks_non_native_instance_output_unavailable() -> None:
    metrics = evaluate_static_snapshot(
        _snapshot(),
        _ground_truth(),
        semantic_vocabulary=("chair", "table"),
        instance_vocabulary=None,
        min_instance_points=1,
    )

    assert metrics["semantic"]["miou"] == pytest.approx(1.0)
    assert metrics["instance"] == {
        "status": "N/A",
        "reason": "Method provides no native entity instances.",
    }
    assert metrics["protocol"]["instance_metrics_available"] is False
