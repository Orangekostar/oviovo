"""Agreement, invariant features, local targets, and geometry selection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.static_ovmap.module_validation.entity_hypotheses import (
    ConflictGroup,
    PartitionHypothesis,
    build_frame_leaf_evidence,
    build_native_leaves,
    build_surface_graph,
)
from src.static_ovmap.module_validation.partition_quality import (
    PartitionTrainingExample,
    apply_geometry_margin,
    build_partition_features,
    geometry_target_support_status,
    local_pq_target,
    partition_agreement,
    predict_partition_quality,
    select_agreement_hypothesis,
    select_geometry_margin,
    train_geometry_head,
)


def _fixture():
    xyz = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.02, 0.0, 1.0], [0.03, 0.0, 1.0]])
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    normals = np.tile([0.0, 0.0, 1.0], (4, 1))
    graph = build_surface_graph(xyz, faces, normals, np.ones(4, dtype=bool))
    leaves = build_native_leaves(
        xyz,
        np.array([1, 1, 2, 2]),
        np.array([10, 10, 20, 20]),
        alias_table=(),
        surface_graph=graph,
    )
    group = ConflictGroup("group:1-2", (1, 2), (0, 1), 0.5)
    original = PartitionHypothesis.create("scene-a", group, leaves, "ORIGINAL", (0, 1))
    merged = PartitionHypothesis.create("scene-a", group, leaves, "MERGE", (0, 0))
    pixel_leaves = np.array([0] * 10 + [1] * 10)
    entities = np.array([1] * 8 + [0] * 2 + [2] * 6 + [0] * 4)
    frame = build_frame_leaf_evidence(1, pixel_leaves, entities, leaf_count=2)
    return xyz, graph, leaves, group, original, merged, frame


def test_hungarian_agreement_includes_unknown_pixels_and_ties_original() -> None:
    _xyz, _graph, _leaves, _group, original, merged, frame = _fixture()

    original_score = partition_agreement(original, (frame,))
    merged_score = partition_agreement(merged, (frame,))

    assert original_score.mean_frame_agreement == pytest.approx(0.7)
    assert original_score.visible_pixel_weighted_agreement == pytest.approx(0.7)
    assert merged_score.mean_frame_agreement == pytest.approx(0.4)
    assert select_agreement_hypothesis((original, merged), (frame,)).kind == "ORIGINAL"


def test_partition_feature_vector_has_twenty_values_and_availability_bits() -> None:
    xyz, graph, leaves, group, original, _merged, frame = _fixture()
    features = build_partition_features(
        original,
        group,
        leaves,
        graph,
        (frame,),
        xyz,
    )

    assert features.values.shape == (20,)
    assert features.available.shape == (20,)
    assert features.values[0] == pytest.approx(math.log1p(4))
    assert features.values[1] == pytest.approx(math.log1p(2))
    assert features.values[2] == 2
    assert features.values[3] == pytest.approx(0.5)
    assert features.values[4] == pytest.approx(0.5)
    assert features.values[5] == pytest.approx(1.0)
    assert features.values[14] == pytest.approx(0.3)
    assert features.values[17] == pytest.approx(0.7)
    assert np.all(features.values[~features.available] == 0.0)
    assert features.standardized(np.zeros(20), np.ones(20)).shape == (40,)


def test_local_pq_target_uses_full_gt_denominator_and_strict_half_iou() -> None:
    target = local_pq_target(
        predicted_components=({0, 1}, {2, 3}),
        group_support={0, 1, 2, 3},
        whole_gt_objects=({0, 1}, {2, 3, 4}),
    )
    strict = local_pq_target(
        predicted_components=({0},),
        group_support={0},
        whole_gt_objects=({0, 1},),
    )

    assert target.included_gt_count == 2
    assert target.true_positives == 2
    assert target.quality == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert strict.true_positives == 0
    assert strict.quality == 0.0


def test_geometry_support_margin_selection_and_strict_adoption() -> None:
    support = [
        {"scene_id": f"s{index % 4}", "targets": (0.1, 0.7)}
        for index in range(20)
    ]
    assert geometry_target_support_status(support) == "SUPPORTED"
    assert geometry_target_support_status(support[:-1]) == "BLOCKED_PARTITION_TARGET_SUPPORT"
    margin = select_geometry_margin(
        (
            {"margin": 0.0, "mean_ap50": 0.5, "mean_ap75": 0.4, "changed_points": 20},
            {"margin": 0.02, "mean_ap50": 0.5, "mean_ap75": 0.4, "changed_points": 10},
            {"margin": 0.05, "mean_ap50": 0.5, "mean_ap75": 0.4, "changed_points": 10},
        )
    )
    assert margin == 0.05
    assert apply_geometry_margin(np.array([0.05, 0.05001]), margin).tolist() == [False, True]
    assert apply_geometry_margin(np.array([10.0]), "KEEP_ALL").tolist() == [False]


def test_fixed_geometry_head_trains_once_and_restores_cal_checkpoint() -> None:
    pytest.importorskip("torch", exc_type=OSError)
    rng = np.random.default_rng(17)
    fit = tuple(
        PartitionTrainingExample(
            scene_id=f"s{group % 4}",
            group_id=f"g{group}",
            features=rng.normal(size=40).astype(np.float32),
            target=float(hypothesis),
        )
        for group in range(20)
        for hypothesis in (0, 1)
    )
    cal = tuple(
        PartitionTrainingExample(
            scene_id="cal",
            group_id=f"g{group}",
            features=rng.normal(size=40).astype(np.float32),
            target=float(hypothesis),
        )
        for group in range(3)
        for hypothesis in (0, 1)
    )
    first = train_geometry_head(fit, cal_examples=cal, max_epochs=10)
    second = train_geometry_head(fit, cal_examples=cal, max_epochs=10)

    assert first.status == "COMPLETE"
    assert first.checkpoint_epoch in (5, 10)
    assert first.parameter_count == 1345
    assert first.optimizer_state_dict
    for name in first.state_dict:
        np.testing.assert_array_equal(
            first.state_dict[name].numpy(), second.state_dict[name].numpy()
        )
    scores = predict_partition_quality(
        first.state_dict, np.stack([row.features for row in cal])
    )
    assert scores.shape == (6,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
