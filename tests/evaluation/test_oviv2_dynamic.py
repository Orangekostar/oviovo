from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.oviv2_dynamic import (
    DynamicGroundTruthFrame,
    DynamicSnapshotEvaluation,
    aggregate_dynamic_sequence,
    evaluate_dynamic_snapshot,
)
from src.evaluation.oviv2_replica import ReplicaGroundTruth
from src.oviv2.meshing import LabeledMesh


def _mesh(vertices, semantic_ids=None) -> LabeledMesh:
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    count = len(points)
    return LabeledMesh(
        vertices_xyz=points,
        triangles=np.empty((0, 3), dtype=np.int64),
        colors_rgb=np.zeros((count, 3), dtype=np.float32),
        semantic_ids=(
            np.ones(count, dtype=np.int64)
            if semantic_ids is None
            else np.asarray(semantic_ids, dtype=np.int64)
        ),
        entity_ids=np.ones(count, dtype=np.int64),
        semantic_confidence=np.ones(count, dtype=np.float32),
        ownership_confidence=np.ones(count, dtype=np.float32),
    )


def _frame() -> DynamicGroundTruthFrame:
    current = ReplicaGroundTruth(
        vertices_xyz=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        semantic_ids=np.asarray([1], dtype=np.int64),
        instance_ids=np.asarray([1], dtype=np.int64),
    )
    return DynamicGroundTruthFrame(
        frame_id=10,
        intervention_id="remove-chair",
        current=current,
        removed_region_vertices_xyz=np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32),
        revealed_background_vertices_xyz=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
    )


def test_dynamic_ground_truth_arrays_are_immutable() -> None:
    frame = _frame()

    assert frame.removed_region_vertices_xyz.flags.writeable is False
    assert frame.revealed_background_vertices_xyz.flags.writeable is False
    with pytest.raises(ValueError):
        frame.removed_region_vertices_xyz[0, 0] = 1.0


@pytest.mark.parametrize(
    ("mesh", "current_miou", "ghost_rate", "background_f5"),
    [
        (_mesh([[0.0, 0.0, 0.0]]), 1.0, 0.0, 1.0),
        (_mesh([[10.0, 0.0, 0.0]]), 0.0, 1.0, 0.0),
        (_mesh(np.empty((0, 3))), 0.0, 0.0, 0.0),
    ],
)
def test_dynamic_snapshot_hand_computed_metrics(
    mesh: LabeledMesh,
    current_miou: float,
    ghost_rate: float,
    background_f5: float,
) -> None:
    evaluation = evaluate_dynamic_snapshot(
        mesh,
        _frame(),
        valid_semantic_ids={1},
        distance_threshold_m=0.05,
    )

    assert evaluation.current_miou == pytest.approx(current_miou)
    assert evaluation.ghost_rate == pytest.approx(ghost_rate)
    assert evaluation.background_f5 == pytest.approx(background_f5)
    assert evaluation.predicted_vertex_count == len(mesh.vertices_xyz)
    assert np.isfinite(
        [evaluation.current_miou, evaluation.ghost_rate, evaluation.background_f5]
    ).all()


def _evaluation(frame_id: int, background_f5: float) -> DynamicSnapshotEvaluation:
    return DynamicSnapshotEvaluation(
        frame_id=frame_id,
        intervention_id="remove-chair",
        current_miou=0.8,
        ghost_rate=0.1,
        background_f5=background_f5,
        predicted_vertex_count=10,
        ghost_vertex_count=1,
        background_precision_match_count=5,
        background_precision_denominator=10,
        background_recall_match_count=5,
        background_recall_denominator=5,
    )


def test_sequence_aggregation_reports_recovery_frames() -> None:
    evaluations = [
        _evaluation(10, 0.1),
        _evaluation(11, 0.6),
        _evaluation(12, 0.91),
        _evaluation(13, 0.94),
    ]

    result = aggregate_dynamic_sequence(
        evaluations,
        intervention_frame_id=10,
        recovery_background_f5=0.9,
    )

    assert result["recovered"] is True
    assert result["recovery_frames"] == 2
    assert result["frame_ids"] == [10, 11, 12, 13]
    assert result["macro_current_miou"] == pytest.approx(0.8)


def test_sequence_aggregation_reports_unrecovered_and_rejects_bad_order() -> None:
    evaluations = [_evaluation(10, 0.1), _evaluation(11, 0.6)]

    result = aggregate_dynamic_sequence(
        evaluations,
        intervention_frame_id=10,
        recovery_background_f5=0.9,
    )

    assert result["recovered"] is False
    assert result["recovery_frames"] is None
    with pytest.raises(ValueError, match="strictly increasing"):
        aggregate_dynamic_sequence(
            [evaluations[1], evaluations[0]],
            intervention_frame_id=10,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        aggregate_dynamic_sequence(
            [evaluations[0], evaluations[0]],
            intervention_frame_id=10,
        )


def test_dynamic_contract_rejects_negative_frame_id() -> None:
    with pytest.raises(ValueError, match="frame_id"):
        DynamicSnapshotEvaluation(
            frame_id=-1,
            intervention_id="remove-chair",
            current_miou=0.0,
            ghost_rate=0.0,
            background_f5=0.0,
        )
