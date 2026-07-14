from __future__ import annotations

import numpy as np

from src.evaluation.contracts import (
    EntityPrediction,
    GroundTruthSnapshot,
    MapSnapshot,
    QueryResult,
    QueryTarget,
)
from src.evaluation.instance_metrics import evaluate_instance_snapshot
from src.evaluation.query_metrics import evaluate_query_results
from src.evaluation.semantic_metrics import evaluate_semantic_snapshot


def _entity(entity_id: str, points: list[list[float]], label: str, score: float = 0.9) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(points, dtype=np.float32),
        semantic_embedding=None,
        semantic_label=label,
        semantic_score=score,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=1.0,
        metadata={},
    )


def _ground_truth() -> GroundTruthSnapshot:
    return GroundTruthSnapshot(
        scene_id="scene-a",
        timestamp=1.0,
        points_xyz=np.array(
            [[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [1.0, 0.0, 1.0], [1.05, 0.0, 1.0]],
            dtype=np.float32,
        ),
        semantic_labels=np.array(["chair", "chair", "table", "table"], dtype=object),
        instance_ids=np.array([1, 1, 2, 2], dtype=np.int64),
    )


def _snapshot(second_label: str = "table") -> MapSnapshot:
    return MapSnapshot(
        method="OVIOVO",
        scene_id="scene-a",
        timestamp=1.0,
        entities=[
            _entity("1", [[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], "chair", 0.95),
            _entity("2", [[1.0, 0.0, 1.0], [1.05, 0.0, 1.0]], second_label, 0.90),
        ],
        background_xyz=None,
        scope="current",
        runtime={"total_s": 0.1},
    )


def test_perfect_runtime_semantics_and_instances_score_one() -> None:
    prediction = _snapshot()
    ground_truth = _ground_truth()

    semantic = evaluate_semantic_snapshot(prediction, ground_truth, max_distance=0.06)
    instance = evaluate_instance_snapshot(prediction, ground_truth, max_distance=0.06)

    assert semantic["miou"] == 1.0
    assert semantic["macc"] == 1.0
    assert semantic["f_miou"] == 1.0
    assert instance["ap25"] == 1.0
    assert instance["ap50"] == 1.0
    assert instance["recall50"] == 1.0


def test_wrong_runtime_label_reduces_semantics_without_changing_instance_ap() -> None:
    prediction = _snapshot(second_label="chair")
    ground_truth = _ground_truth()

    semantic = evaluate_semantic_snapshot(prediction, ground_truth, max_distance=0.06)
    instance = evaluate_instance_snapshot(prediction, ground_truth, max_distance=0.06)

    assert semantic["miou"] < 1.0
    assert semantic["macc"] < 1.0
    assert instance["ap50"] == 1.0


def test_empty_prediction_scores_zero_against_nonempty_ground_truth() -> None:
    prediction = MapSnapshot(
        method="OVIOVO",
        scene_id="scene-a",
        timestamp=1.0,
        entities=[],
        background_xyz=None,
        scope="current",
        runtime={},
    )
    semantic = evaluate_semantic_snapshot(prediction, _ground_truth(), max_distance=0.06)
    instance = evaluate_instance_snapshot(prediction, _ground_truth(), max_distance=0.06)

    assert semantic["miou"] == 0.0
    assert semantic["matched_point_ratio"] == 0.0
    assert instance["ap25"] == 0.0
    assert instance["recall50"] == 0.0


def test_query_metrics_separate_ranking_rejection_and_stale_false_positive() -> None:
    targets = [
        QueryTarget(query_id="present", valid_entity_ids=("1",)),
        QueryTarget(query_id="absent", valid_entity_ids=()),
    ]
    perfect = [
        QueryResult("present", ["1"], [0.9], "1", False, 2.0),
        QueryResult("absent", [], [], None, True, 1.0),
    ]
    metrics = evaluate_query_results(perfect, targets)
    assert metrics["current_r1"] == 1.0
    assert metrics["not_found_f1"] == 1.0
    assert metrics["stale_fp"] == 0.0

    stale = [
        perfect[0],
        QueryResult("absent", ["dormant-2"], [0.7], "dormant-2", False, 1.5),
    ]
    stale_metrics = evaluate_query_results(stale, targets)
    assert stale_metrics["current_r1"] == 1.0
    assert stale_metrics["not_found_f1"] < 1.0
    assert stale_metrics["stale_fp"] == 1.0

    multi_target = [QueryTarget(query_id="multi", valid_entity_ids=("1", "2"))]
    multi_result = [QueryResult("multi", ["2", "3"], [0.8, 0.4], "2", False, 1.0)]
    assert evaluate_query_results(multi_result, multi_target)["current_r1"] == 1.0
