from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.oviv2_replica import (
    EntityEvaluationInfo,
    ReplicaGroundTruth,
    evaluate_replica_voxel_map,
    project_mesh_to_gt,
)
from src.oviv2.meshing import LabeledMesh


def _mesh(
    vertices: np.ndarray,
    semantic_ids: np.ndarray | None = None,
    entity_ids: np.ndarray | None = None,
) -> LabeledMesh:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    count = len(vertices)
    return LabeledMesh(
        vertices_xyz=vertices,
        triangles=np.empty((0, 3), dtype=np.int64),
        colors_rgb=np.zeros((count, 3), dtype=np.float32),
        semantic_ids=(
            np.zeros(count, dtype=np.int64)
            if semantic_ids is None
            else np.asarray(semantic_ids, dtype=np.int64)
        ),
        entity_ids=(
            np.zeros(count, dtype=np.int64)
            if entity_ids is None
            else np.asarray(entity_ids, dtype=np.int64)
        ),
        semantic_confidence=np.ones(count, dtype=np.float32),
        ownership_confidence=np.ones(count, dtype=np.float32),
    )


def _ground_truth(vertices, semantic_ids, instance_ids) -> ReplicaGroundTruth:
    return ReplicaGroundTruth(
        vertices_xyz=np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
        semantic_ids=np.asarray(semantic_ids, dtype=np.int64),
        instance_ids=np.asarray(instance_ids, dtype=np.int64),
    )


def test_projection_uses_strict_five_centimeter_gate() -> None:
    mesh = _mesh([[0.0, 0.0, 0.0]], [7], [11])
    gt_vertices = np.asarray([[0.049999, 0.0, 0.0], [0.05, 0.0, 0.0]])

    projected = project_mesh_to_gt(mesh, gt_vertices, max_distance_m=0.05)

    np.testing.assert_array_equal(projected.semantic_ids, [7, 0])
    np.testing.assert_array_equal(projected.entity_ids, [11, 0])
    np.testing.assert_array_equal(projected.matched, [True, False])


def test_semantic_confusion_matches_hand_computed_metrics() -> None:
    vertices = np.asarray([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    mesh = _mesh(vertices, [1, 2, 2, 0], [0, 0, 0, 0])
    gt = _ground_truth(vertices, [1, 1, 2, 2], [1, 1, 2, 2])

    metrics = evaluate_replica_voxel_map(
        mesh,
        gt,
        [],
        valid_semantic_ids={1, 2},
        instance_semantic_ids=set(),
        min_instance_vertices=1,
    )

    assert metrics["semantic"]["per_class"]["1"] == {
        "iou": pytest.approx(0.5),
        "accuracy": pytest.approx(0.5),
        "support": 2,
    }
    assert metrics["semantic"]["per_class"]["2"] == {
        "iou": pytest.approx(1.0 / 3.0),
        "accuracy": pytest.approx(0.5),
        "support": 2,
    }
    assert metrics["miou"] == pytest.approx(5.0 / 12.0)
    assert metrics["macc"] == pytest.approx(0.5)
    assert metrics["f_miou"] == pytest.approx(5.0 / 12.0)


def test_instance_matching_is_class_constrained_when_gt_ids_repeat() -> None:
    vertices = np.column_stack((np.arange(200, dtype=np.float32), np.zeros((200, 2))))
    mesh = _mesh(
        vertices,
        np.repeat([1, 2], 100),
        np.repeat([10, 20], 100),
    )
    gt = _ground_truth(vertices, np.repeat([1, 2], 100), np.ones(200, dtype=np.int64))
    info = [
        EntityEvaluationInfo(10, semantic_id=1, accepted_view_count=2),
        EntityEvaluationInfo(20, semantic_id=2, accepted_view_count=2),
    ]

    metrics = evaluate_replica_voxel_map(
        mesh,
        gt,
        info,
        valid_semantic_ids={1, 2},
        instance_semantic_ids={1, 2},
    )

    assert metrics["ap25"] == pytest.approx(1.0)
    assert metrics["ap50"] == pytest.approx(1.0)
    assert metrics["instance"]["ground_truth_instance_count"] == 2


def test_area_confidence_and_filters_produce_hand_computed_ap() -> None:
    vertices = np.column_stack((np.arange(230, dtype=np.float32), np.zeros((230, 2))))
    semantic = np.concatenate((np.ones(100, dtype=np.int64), np.full(130, 2, dtype=np.int64)))
    gt_instances = np.concatenate((np.ones(100, dtype=np.int64), np.full(130, 2, dtype=np.int64)))
    predicted_entities = np.concatenate(
        (
            np.full(100, 10, dtype=np.int64),
            np.full(120, 20, dtype=np.int64),
            np.full(5, 30, dtype=np.int64),
            np.full(5, 40, dtype=np.int64),
        )
    )
    mesh = _mesh(vertices, np.ones(230, dtype=np.int64), predicted_entities)
    gt = _ground_truth(vertices, semantic, gt_instances)
    info = [
        EntityEvaluationInfo(10, 1, 2),
        EntityEvaluationInfo(20, 1, 2),
        EntityEvaluationInfo(30, 1, 1),
        EntityEvaluationInfo(40, 1, 2),
    ]

    metrics = evaluate_replica_voxel_map(
        mesh,
        gt,
        info,
        valid_semantic_ids={1, 2},
        instance_semantic_ids={1},
        min_instance_vertices=100,
    )

    assert metrics["instance"]["predicted_instance_count"] == 2
    assert metrics["instance"]["per_class"]["1"]["prediction_confidences"] == [
        1.0,
        pytest.approx(100.0 / 120.0),
    ]
    assert metrics["ap25"] == pytest.approx(0.5)
    assert metrics["ap50"] == pytest.approx(0.5)


def test_geometry_f5_uses_raw_mesh_before_projection() -> None:
    mesh = _mesh([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    gt = _ground_truth([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [1, 1], [1, 1])

    metrics = evaluate_replica_voxel_map(
        mesh,
        gt,
        [],
        valid_semantic_ids={1},
        instance_semantic_ids=set(),
        min_instance_vertices=1,
    )

    assert metrics["geometry"] == {
        "precision": pytest.approx(0.5),
        "recall": pytest.approx(0.5),
        "f5": pytest.approx(0.5),
    }


def test_empty_prediction_returns_finite_zero_headlines() -> None:
    mesh = _mesh(np.empty((0, 3)))
    gt = _ground_truth([[0.0, 0.0, 0.0]], [1], [1])

    metrics = evaluate_replica_voxel_map(
        mesh,
        gt,
        [],
        valid_semantic_ids={1},
        instance_semantic_ids={1},
        min_instance_vertices=1,
    )

    for name in ("miou", "macc", "f_miou", "ap25", "ap50", "f5"):
        assert metrics[name] == 0.0
        assert np.isfinite(metrics[name])
