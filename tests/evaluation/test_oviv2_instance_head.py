from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[2] / "src" / "evaluation" / "oviv2_instance_head.py"
)


def test_instance_head_module_exists() -> None:
    assert MODULE_PATH.is_file()


def _api():
    from src.evaluation.oviv2_instance_head import (
        InstanceHeadConfig,
        InstanceHypothesis,
        ProjectedInstanceHypothesis,
        build_instance_hypotheses,
        deduplicate_projected_hypotheses,
        evaluate_instance_hypotheses,
        evaluate_projected_instance_hypotheses,
        project_instance_hypotheses,
    )

    return {
        "config": InstanceHeadConfig,
        "hypothesis": InstanceHypothesis,
        "projected": ProjectedInstanceHypothesis,
        "build": build_instance_hypotheses,
        "deduplicate": deduplicate_projected_hypotheses,
        "evaluate": evaluate_instance_hypotheses,
        "evaluate_projected": evaluate_projected_instance_hypotheses,
        "project": project_instance_hypotheses,
    }


def _mesh(
    vertices: list[list[float]] | np.ndarray,
    triangles: list[list[int]] | np.ndarray,
    *,
    entity_ids: list[int] | np.ndarray,
    semantic_ids: list[int] | np.ndarray | None = None,
):
    from src.oviv2.meshing import LabeledMesh

    vertices_array = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    count = len(vertices_array)
    return LabeledMesh(
        vertices_xyz=vertices_array,
        triangles=np.asarray(triangles, dtype=np.int64).reshape(-1, 3),
        colors_rgb=np.zeros((count, 3), dtype=np.float32),
        semantic_ids=(
            np.ones(count, dtype=np.int64)
            if semantic_ids is None
            else np.asarray(semantic_ids, dtype=np.int64)
        ),
        entity_ids=np.asarray(entity_ids, dtype=np.int64),
        semantic_confidence=np.ones(count, dtype=np.float32),
        ownership_confidence=np.ones(count, dtype=np.float32),
    )


def _info(
    entity_id: int,
    *,
    semantic_id: int = 4,
    views: int = 2,
    margin: float = 0.75,
):
    from src.evaluation.oviv2_replica import EntityEvaluationInfo

    return EntityEvaluationInfo(
        entity_id=entity_id,
        semantic_id=semantic_id,
        accepted_view_count=views,
        semantic_confidence=margin,
    )


def _hypothesis(
    hypothesis_id: str,
    vertex_indices: list[int],
    *,
    score: float = 0.5,
    entity_id: int = 10,
    kind: str = "parent",
):
    api = _api()
    return api["hypothesis"](
        hypothesis_id=hypothesis_id,
        entity_id=entity_id,
        semantic_id=4,
        kind=kind,
        vertex_indices=np.asarray(vertex_indices, dtype=np.int64),
        score=score,
    )


def _projected(hypothesis_id: str, mask, *, score: float = 0.5):
    api = _api()
    return api["projected"](
        hypothesis_id=hypothesis_id,
        entity_id=10,
        semantic_id=4,
        kind="parent",
        score=score,
        mask=np.asarray(mask, dtype=bool),
    )


def test_config_and_hypothesis_contracts_are_frozen_and_validate_inputs() -> None:
    api = _api()
    config = api["config"]()
    assert config.minimum_component_vertices == 200
    assert config.child_score_multiplier == 2.0
    assert config.view_count_exponent == 1.0
    assert config.semantic_evidence_exponent == 1.0
    with pytest.raises(FrozenInstanceError):
        config.minimum_component_vertices = 1

    source = np.asarray([1, 2, 3], dtype=np.int64)
    hypothesis = api["hypothesis"]("e10:parent", 10, 4, "parent", source, 0.5)
    source[:] = 0
    np.testing.assert_array_equal(hypothesis.vertex_indices, [1, 2, 3])
    assert not hypothesis.vertex_indices.flags.writeable
    with pytest.raises(FrozenInstanceError):
        hypothesis.score = 0.1

    invalid_configs = (
        {"minimum_component_vertices": 0},
        {"child_score_multiplier": np.inf},
        {"maximum_projection_distance_m": 0.0},
        {"deduplication_iou_threshold": 0.0},
        {"deduplication_iou_threshold": 1.1},
        {"view_count_exponent": -0.1},
        {"semantic_evidence_exponent": np.inf},
    )
    for values in invalid_configs:
        with pytest.raises((TypeError, ValueError)):
            api["config"](**values)
    with pytest.raises(ValueError, match="strictly increasing"):
        api["hypothesis"]("bad", 1, 4, "parent", np.asarray([2, 1]), 0.5)
    with pytest.raises(ValueError, match="finite"):
        api["hypothesis"]("bad", 1, 4, "parent", np.asarray([1]), np.nan)
    with pytest.raises(TypeError, match="integer"):
        api["hypothesis"]("bad", 1, 4, "parent", np.asarray([1.5]), 0.5)
    with pytest.raises(TypeError, match="boolean"):
        api["projected"]("bad", 1, 4, "parent", 0.5, np.asarray([0, 1]))


def test_components_respect_mixed_entity_triangle_boundaries_and_isolated_vertices() -> None:
    api = _api()
    mesh = _mesh(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]],
        [[0, 1, 2], [2, 3, 4]],
        entity_ids=[10, 10, 10, 20, 10, 10],
    )
    config = api["config"](minimum_component_vertices=1)

    hypotheses = api["build"](mesh, [_info(10), _info(20)], config)

    entity10 = [item for item in hypotheses if item.entity_id == 10]
    assert [(item.kind, item.vertex_indices.tolist()) for item in entity10] == [
        ("parent", [0, 1, 2, 4, 5]),
        ("child", [0, 1, 2]),
        ("child", [4]),
        ("child", [5]),
    ]
    entity20 = [item for item in hypotheses if item.entity_id == 20]
    assert [(item.kind, item.vertex_indices.tolist()) for item in entity20] == [
        ("parent", [3]),
    ]


def test_component_order_is_geometry_deterministic_and_single_component_has_no_child() -> None:
    api = _api()
    mesh = _mesh(
        [[5, 0, 0], [6, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        [[0, 1, 1], [2, 3, 4]],
        entity_ids=[10, 10, 10, 10, 10, 20],
    )
    config = api["config"](minimum_component_vertices=1)

    first = api["build"](mesh, [_info(20), _info(10)], config)
    second = api["build"](mesh, [_info(10), _info(20)], config)

    signature = lambda values: [
        (item.hypothesis_id, item.vertex_indices.tolist()) for item in values
    ]
    assert signature(first) == signature(second)
    assert signature(first) == [
        ("e10:parent", [0, 1, 2, 3, 4]),
        ("e10:component:0000", [2, 3, 4]),
        ("e10:component:0001", [0, 1]),
        ("e20:parent", [5]),
    ]


def test_component_support_threshold_and_scores_are_finite_with_deterministic_ties() -> None:
    api = _api()
    mesh = _mesh(
        np.column_stack((np.arange(8, dtype=np.float32), np.zeros((8, 2)))),
        [[0, 1, 2], [3, 4, 5]],
        entity_ids=[20, 20, 20, 10, 10, 10, 10, 10],
    )
    config = api["config"](minimum_component_vertices=2)

    values = api["build"](
        mesh,
        [_info(20, views=3, margin=0.5), _info(10, views=3, margin=0.5)],
        config,
    )

    assert [item.hypothesis_id for item in values] == [
        "e10:parent",
        "e10:component:0000",
        "e20:parent",
    ]
    assert all(np.isfinite(item.score) and 0.0 <= item.score <= 1.0 for item in values)
    parent = values[0]
    child = values[1]
    assert child.score == pytest.approx(
        np.clip(parent.score * 2.0 * (3.0 / 5.0) ** 2, 0.0, 1.0)
    )
    assert values[2].score == parent.score


def test_parent_scores_use_normalized_view_count_and_semantic_evidence_powers() -> None:
    api = _api()
    mesh = _mesh([[0, 0, 0], [1, 0, 0]], [], entity_ids=[10, 20])
    config = api["config"](
        view_count_exponent=2.0,
        semantic_evidence_exponent=1.0,
    )

    values = api["build"](
        mesh,
        [_info(10, views=4, margin=0.5), _info(20, views=2, margin=1.0)],
        config,
    )

    assert values[0].score == pytest.approx(0.5)
    assert values[1].score == pytest.approx(0.25)


def test_build_rejects_duplicate_or_missing_info_without_semantic_filtering() -> None:
    api = _api()
    mesh = _mesh([[0, 0, 0], [1, 0, 0]], [], entity_ids=[10, 20])
    with pytest.raises(ValueError, match="duplicate"):
        api["build"](mesh, [_info(10), _info(10)], api["config"]())
    with pytest.raises(ValueError, match="missing"):
        api["build"](mesh, [_info(10)], api["config"]())

    values = api["build"](
        mesh,
        [_info(10, semantic_id=1), _info(20, semantic_id=4)],
        api["config"](),
    )
    assert [item.entity_id for item in values] == [10, 20]


def test_projection_is_independent_so_competing_hypotheses_overlap() -> None:
    api = _api()
    mesh = _mesh(
        [[0.0, 0, 0], [0.04, 0, 0]],
        [],
        entity_ids=[10, 20],
    )
    gt_vertices = np.asarray([[0.02, 0, 0], [0.09, 0, 0]], dtype=np.float32)
    hypotheses = (
        _hypothesis("a", [0], score=0.8, entity_id=10),
        _hypothesis("b", [1], score=0.7, entity_id=20),
    )

    projected = api["project"](mesh, hypotheses, gt_vertices, 0.05)

    assert [item.mask.tolist() for item in projected] == [[True, False], [True, False]]
    assert all(not item.mask.flags.writeable for item in projected)


def test_projected_mask_deduplication_is_greedy_by_score_then_id() -> None:
    api = _api()
    values = (
        _projected("z", [1, 1, 0], score=0.9),
        _projected("a", [1, 1, 0], score=0.9),
        _projected("q", [0, 0, 1], score=0.8),
    )

    kept = api["deduplicate"](values, 0.95)

    assert [item.hypothesis_id for item in kept] == ["a", "q"]


def test_preprojected_evaluation_filters_support_and_deduplicates_sources() -> None:
    from src.evaluation.oviv2_replica import ReplicaGroundTruth

    api = _api()
    vertices = np.column_stack((np.arange(6, dtype=np.float32), np.zeros((6, 2))))
    ground_truth = ReplicaGroundTruth(
        vertices_xyz=vertices,
        semantic_ids=np.full(6, 4, dtype=np.int64),
        instance_ids=np.repeat([1, 2], 3),
    )
    projected = (
        _projected("unsupported", [1, 0, 0, 0, 0, 0], score=1.0),
        _projected("base:e1", [1, 1, 1, 0, 0, 0], score=0.9),
        _projected("aux:e1", [1, 1, 1, 0, 0, 0], score=0.8),
        _projected("aux:e2", [0, 0, 0, 1, 1, 1], score=0.7),
    )

    result = api["evaluate_projected"](
        projected,
        ground_truth,
        instance_semantic_ids={4},
        min_instance_vertices=2,
        deduplication_iou_threshold=0.9,
    )

    assert result["ap25"] == pytest.approx(1.0)
    assert result["ap50"] == pytest.approx(1.0)
    assert result["recall25"] == pytest.approx(1.0)
    assert result["recall50"] == pytest.approx(1.0)
    assert result["raw_hypothesis_count"] == 4
    assert result["supported_hypothesis_count"] == 3
    assert result["predicted_instance_count"] == 2
    assert result["prediction_hypothesis_ids"] == ["base:e1", "aux:e2"]


def test_instance_ap_reuses_semantic_instance_gt_identity_but_ignores_prediction_semantics() -> None:
    from src.evaluation.oviv2_replica import ReplicaGroundTruth

    api = _api()
    vertices = np.column_stack((np.arange(8, dtype=np.float32), np.zeros((8, 2))))
    mesh = _mesh(
        vertices,
        [[0, 1, 2], [4, 5, 6]],
        entity_ids=np.repeat([10, 20], 4),
        semantic_ids=np.repeat([99, 98], 4),
    )
    ground_truth = ReplicaGroundTruth(
        vertices_xyz=vertices,
        semantic_ids=np.repeat([4, 5], 4),
        instance_ids=np.ones(8, dtype=np.int64),
    )
    config = api["config"](
        minimum_component_vertices=10,
        maximum_projection_distance_m=0.05,
    )
    hypotheses = api["build"](
        mesh,
        [_info(10, semantic_id=99), _info(20, semantic_id=98)],
        config,
    )

    result = api["evaluate"](
        mesh,
        hypotheses,
        ground_truth,
        instance_semantic_ids={4, 5},
        min_instance_vertices=4,
        config=config,
    )

    assert result["ap25"] == pytest.approx(1.0)
    assert result["ap50"] == pytest.approx(1.0)
    assert result["ground_truth_instance_count"] == 2
    assert result["prediction_hypothesis_ids"] == ["e10:parent", "e20:parent"]


def test_hypothesis_generation_signature_contains_no_ground_truth_parameter() -> None:
    import inspect

    api = _api()
    assert "ground_truth" not in inspect.signature(api["build"]).parameters
