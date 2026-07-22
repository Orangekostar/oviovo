from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import math

import numpy as np
import pytest

from src.evaluation.oviv2_view_graph import (
    ObservationEdge,
    ViewGraphConfig,
    build_sparse_observation_edges,
    select_observation_edges,
)
from src.oviv2.observations import FrameObservation, ObservationKind


def _observation(
    observation_id: int,
    frame_id: int,
    voxels: set[tuple[int, int, int]],
    **changes: object,
) -> FrameObservation:
    values = {
        "observation_id": observation_id,
        "frame_id": frame_id,
        "timestamp": float(frame_id),
        "kind": ObservationKind.OBJECT,
        "label": "chair",
        "semantic_id": 2,
        "confidence": 0.9,
        "mask": np.ones((2, 2), dtype=bool),
        "bbox_xyxy": (0.0, 0.0, 2.0, 2.0),
        "voxel_keys": frozenset(voxels),
        "centroid_xyz": (0.0, 0.0, 1.0),
        "bounds_min_xyz": (0.0, 0.0, 1.0),
        "bounds_max_xyz": (0.0, 0.0, 1.0),
    }
    values.update(changes)
    return FrameObservation(**values)


def _edge(
    left_id: int,
    right_id: int,
    *,
    iou: float = 0.2,
    left_coverage: float = 0.5,
    right_coverage: float = 0.5,
    feature_cosine: float | None = None,
) -> ObservationEdge:
    return ObservationEdge(
        left_id,
        right_id,
        shared_voxels=2,
        voxel_iou=iou,
        left_coverage=left_coverage,
        right_coverage=right_coverage,
        feature_cosine=feature_cosine,
        view_direction_cosine=None,
    )


def test_builds_exact_sparse_overlap_evidence_for_different_frames() -> None:
    observations = (
        _observation(10, 3, {(0, 0, 0), (1, 0, 0), (2, 0, 0)}),
        _observation(21, 4, {(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)}),
    )

    evidence = build_sparse_observation_edges(
        observations,
        ViewGraphConfig(minimum_overlap_voxels=2),
    )

    assert len(evidence) == 1
    edge = evidence[0]
    assert (edge.left_id, edge.right_id, edge.shared_voxels) == (10, 21, 2)
    assert edge.voxel_iou == pytest.approx(0.4)
    assert edge.left_coverage == pytest.approx(2.0 / 3.0)
    assert edge.right_coverage == pytest.approx(0.5)
    assert edge.feature_cosine is None
    assert edge.view_direction_cosine is None


def test_excludes_same_frame_pairs_and_disjoint_masks() -> None:
    observations = (
        _observation(10, 3, {(0, 0, 0), (1, 0, 0)}),
        _observation(21, 3, {(0, 0, 0), (1, 0, 0)}),
        _observation(33, 4, {(8, 0, 0), (9, 0, 0)}),
    )

    assert build_sparse_observation_edges(observations, ViewGraphConfig(2)) == ()


def test_ignores_non_object_and_unknown_semantic_observations() -> None:
    observations = (
        _observation(10, 3, {(0, 0, 0), (1, 0, 0)}),
        _observation(21, 4, {(0, 0, 0), (1, 0, 0)}, kind=ObservationKind.STRUCTURE),
        _observation(33, 5, {(0, 0, 0), (1, 0, 0)}, semantic_id=0),
    )

    assert build_sparse_observation_edges(observations, ViewGraphConfig(2)) == ()


def test_semantic_mismatch_is_configurable_before_feature_evidence() -> None:
    observations = (
        _observation(10, 3, {(0, 0, 0), (1, 0, 0)}, image_feature=np.array([1.0, 0.0]), feature_model_id="m"),
        _observation(21, 4, {(0, 0, 0), (1, 0, 0)}, semantic_id=3, image_feature=np.array([1.0, 0.0]), feature_model_id="m"),
    )

    assert build_sparse_observation_edges(observations, ViewGraphConfig(2)) == ()
    allowed = build_sparse_observation_edges(observations, ViewGraphConfig(2, False))
    assert allowed[0].feature_cosine == pytest.approx(1.0)


def test_feature_and_view_direction_evidence_are_optional_and_compatible() -> None:
    base_voxels = {(0, 0, 0), (1, 0, 0)}
    matching = build_sparse_observation_edges(
        (
            _observation(10, 3, base_voxels, image_feature=np.array([1.0, 0.0]), feature_model_id="m", view_direction_xyz=(1.0, 0.0, 0.0)),
            _observation(21, 4, base_voxels, image_feature=np.array([0.0, 1.0]), feature_model_id="m", view_direction_xyz=(0.0, 1.0, 0.0)),
        ),
        ViewGraphConfig(2),
    )
    model_mismatch = build_sparse_observation_edges(
        (
            _observation(30, 5, base_voxels, image_feature=np.array([1.0, 0.0]), feature_model_id="m1"),
            _observation(42, 6, base_voxels, image_feature=np.array([1.0, 0.0]), feature_model_id="m2"),
        ),
        ViewGraphConfig(2),
    )

    assert matching[0].feature_cosine == pytest.approx(0.0)
    assert matching[0].view_direction_cosine == pytest.approx(0.0)
    assert model_mismatch[0].feature_cosine is None
    assert model_mismatch[0].view_direction_cosine is None


def test_matching_feature_model_with_different_dimensions_fails_clearly() -> None:
    observations = (
        _observation(10, 3, {(0, 0, 0), (1, 0, 0)}, image_feature=np.array([1.0, 0.0]), feature_model_id="m"),
        _observation(21, 4, {(0, 0, 0), (1, 0, 0)}, image_feature=np.array([1.0, 0.0, 0.0]), feature_model_id="m"),
    )

    with pytest.raises(ValueError, match="dimension"):
        build_sparse_observation_edges(observations, ViewGraphConfig(2))


def test_building_is_input_order_independent_and_rejects_duplicate_ids() -> None:
    observations = (
        _observation(10, 3, {(0, 0, 0), (1, 0, 0)}),
        _observation(21, 4, {(0, 0, 0), (1, 0, 0)}),
        _observation(33, 5, {(0, 0, 0), (1, 0, 0)}),
    )
    expected = build_sparse_observation_edges(observations, ViewGraphConfig(2))

    assert build_sparse_observation_edges(tuple(reversed(observations)), ViewGraphConfig(2)) == expected
    with pytest.raises(ValueError, match="duplicate"):
        build_sparse_observation_edges((observations[0], observations[0]), ViewGraphConfig(2))


def test_building_validates_observation_sequence_and_config_types() -> None:
    with pytest.raises(TypeError, match="observations"):
        build_sparse_observation_edges(object(), ViewGraphConfig(2))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config"):
        build_sparse_observation_edges((), object())  # type: ignore[arg-type]


def test_selects_iou_or_reciprocal_coverage_and_gates_feature_evidence() -> None:
    evidence = (
        _edge(1, 2, iou=0.10, left_coverage=0.2, right_coverage=0.2, feature_cosine=0.2),
        _edge(3, 4, iou=0.02, left_coverage=0.5, right_coverage=0.5, feature_cosine=None),
        _edge(5, 6, iou=0.8, left_coverage=0.8, right_coverage=0.8, feature_cosine=0.19),
    )

    selected = select_observation_edges(
        evidence,
        minimum_voxel_iou=0.1,
        minimum_directed_coverage=0.25,
        minimum_feature_cosine=0.2,
    )

    assert [(edge.left_id, edge.right_id) for edge in selected] == [(1, 2), (3, 4)]


def test_selection_admits_exact_thresholds_and_returns_sorted_edges() -> None:
    selected = select_observation_edges(
        (
            _edge(30, 42, iou=0.05, left_coverage=0.25, right_coverage=0.25, feature_cosine=0.20),
            _edge(10, 21, iou=0.05, left_coverage=0.25, right_coverage=0.25, feature_cosine=0.20),
        ),
        minimum_voxel_iou=0.05,
        minimum_directed_coverage=0.25,
        minimum_feature_cosine=0.20,
    )

    assert [(edge.left_id, edge.right_id) for edge in selected] == [(10, 21), (30, 42)]
    assert selected[0].shared_voxels == 2


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_config_rejects_boolean_and_invalid_overlap_thresholds(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="minimum_overlap_voxels"):
        ViewGraphConfig(minimum_overlap_voxels=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"left_id": True}, TypeError),
        ({"right_id": 10}, ValueError),
        ({"left_id": -1}, ValueError),
        ({"shared_voxels": True}, TypeError),
        ({"shared_voxels": 0}, ValueError),
        ({"voxel_iou": math.nan}, ValueError),
        ({"left_coverage": 1.01}, ValueError),
        ({"feature_cosine": math.inf}, ValueError),
        ({"view_direction_cosine": -1.1}, ValueError),
    ],
)
def test_edge_record_validates_ids_counts_and_ranges(changes: dict[str, object], error: type[Exception]) -> None:
    values: dict[str, object] = {
        "left_id": 10,
        "right_id": 21,
        "shared_voxels": 2,
        "voxel_iou": 0.25,
        "left_coverage": 0.5,
        "right_coverage": 0.5,
        "feature_cosine": None,
        "view_direction_cosine": None,
    }
    values.update(changes)
    with pytest.raises(error):
        ObservationEdge(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -0.1, 1.1])
def test_selection_rejects_invalid_geometry_thresholds(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        select_observation_edges((), minimum_voxel_iou=value, minimum_directed_coverage=0.0, minimum_feature_cosine=0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -1.1, 1.1])
def test_selection_rejects_invalid_feature_threshold(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        select_observation_edges((), minimum_voxel_iou=0.0, minimum_directed_coverage=0.0, minimum_feature_cosine=value)  # type: ignore[arg-type]


def test_selection_rejects_duplicate_pairs() -> None:
    edge = _edge(10, 21)
    with pytest.raises(ValueError, match="duplicate"):
        select_observation_edges((edge, edge), minimum_voxel_iou=0.0, minimum_directed_coverage=0.0, minimum_feature_cosine=0.0)


def test_selection_validates_evidence_types() -> None:
    with pytest.raises(TypeError, match="evidence"):
        select_observation_edges((object(),), minimum_voxel_iou=0.0, minimum_directed_coverage=0.0, minimum_feature_cosine=0.0)  # type: ignore[arg-type]


def test_public_generation_functions_do_not_accept_ground_truth() -> None:
    for function in (build_sparse_observation_edges, select_observation_edges):
        assert "ground_truth" not in inspect.signature(function).parameters


def test_uses_sparse_inverted_index_without_square_node_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    observations = [
        _observation(index, index, {(0, 0, 0), (1, 0, 0)} if index in {0, 1} else {(index, 1, 0)})
        for index in range(1_000)
    ]
    import src.evaluation.oviv2_view_graph as graph_module

    original_zeros = np.zeros

    def reject_square(shape: object, *args: object, **kwargs: object) -> np.ndarray:
        if isinstance(shape, tuple) and len(shape) == 2 and shape[0] == shape[1] == len(observations):
            raise AssertionError("dense observation-by-observation allocation")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(graph_module.np, "zeros", reject_square)
    evidence = build_sparse_observation_edges(observations, ViewGraphConfig(2))

    assert [(edge.left_id, edge.right_id) for edge in evidence] == [(0, 1)]


def test_records_are_frozen_and_functions_return_immutable_tuples() -> None:
    config = ViewGraphConfig(2)
    edge = _edge(10, 21)
    with pytest.raises(FrozenInstanceError):
        config.minimum_overlap_voxels = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        edge.shared_voxels = 3  # type: ignore[misc]

    result = build_sparse_observation_edges(
        (_observation(10, 3, {(0, 0, 0), (1, 0, 0)}), _observation(21, 4, {(0, 0, 0), (1, 0, 0)})),
        config,
    )
    assert isinstance(result, tuple)
