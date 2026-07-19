from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.oviv2.observation_graph import (
    CausalObservationGraph,
    ObservationEdge,
    ObservationNode,
)


def _node(observation_id: int, frame_id: int) -> ObservationNode:
    return ObservationNode(observation_id=observation_id, frame_id=frame_id)


def _edge(
    left_observation_id: int,
    right_observation_id: int,
    source_frame_id: int,
    target_frame_id: int,
    score: float,
) -> ObservationEdge:
    return ObservationEdge(
        left_observation_id=left_observation_id,
        right_observation_id=right_observation_id,
        source_frame_id=source_frame_id,
        target_frame_id=target_frame_id,
        score=score,
    )


def test_future_edge_is_rejected_and_window_keeps_latest_frames() -> None:
    graph = CausalObservationGraph(window_size=3)
    graph.add_frame(10, [_node(1, 10)])

    with pytest.raises(ValueError, match="future"):
        graph.add_edge(_edge(1, 2, 10, 11, 0.8))

    graph.add_frame(11, [_node(2, 11)])
    graph.add_frame(12, [_node(3, 12)])
    graph.add_frame(13, [_node(4, 13)])

    assert graph.observation_ids == (2, 3, 4)
    assert graph.frame_ids == (11, 12, 13)


def test_ambiguous_edge_is_supported_by_an_independent_third_view() -> None:
    graph = CausalObservationGraph(window_size=3)
    graph.add_frame(0, [_node(1, 0)])
    graph.add_frame(1, [_node(2, 1)])
    graph.add_edge(_edge(1, 2, 0, 1, 0.62))
    graph.add_frame(2, [_node(3, 2)])
    graph.add_edge(_edge(1, 3, 0, 2, 0.80))
    graph.add_edge(_edge(2, 3, 1, 2, 0.78))

    assert graph.edge_is_supported(
        1,
        2,
        ambiguous_below=0.70,
        third_view_min_score=0.75,
    )


def test_strong_edge_needs_no_third_view_and_removal_revokes_support() -> None:
    graph = CausalObservationGraph(window_size=2)
    graph.add_frame(0, [_node(1, 0)])
    graph.add_frame(1, [_node(2, 1)])
    graph.add_edge(_edge(1, 2, 0, 1, 0.70))

    assert graph.edge_is_supported(
        1,
        2,
        ambiguous_below=0.70,
        third_view_min_score=0.75,
    )
    assert graph.remove_edge(2, 1) is True
    assert graph.edge_is_supported(
        1,
        2,
        ambiguous_below=0.70,
        third_view_min_score=0.75,
    ) is False
    assert graph.remove_edge(1, 2) is False


def test_expired_supporting_node_removes_incident_edges_and_weakens_edge() -> None:
    graph = CausalObservationGraph(window_size=3)
    graph.add_frame(0, [_node(3, 0)])
    graph.add_frame(1, [_node(1, 1)])
    graph.add_frame(2, [_node(2, 2)])
    graph.add_edge(_edge(1, 2, 1, 2, 0.62))
    graph.add_edge(_edge(3, 1, 0, 1, 0.80))
    graph.add_edge(_edge(3, 2, 0, 2, 0.78))
    assert graph.edge_is_supported(
        1,
        2,
        ambiguous_below=0.70,
        third_view_min_score=0.75,
    )
    assert graph.edge_count == 3

    graph.add_frame(3, [])

    assert graph.observation_ids == (1, 2)
    assert graph.edge_count == 1
    assert graph.edge_is_supported(
        1,
        2,
        ambiguous_below=0.70,
        third_view_min_score=0.75,
    ) is False


def test_third_node_from_an_endpoint_frame_is_not_an_independent_view() -> None:
    graph = CausalObservationGraph(window_size=2)
    graph.add_frame(0, [_node(1, 0)])
    graph.add_frame(1, [_node(2, 1), _node(3, 1)])
    graph.add_edge(_edge(1, 2, 0, 1, 0.62))
    graph.add_edge(_edge(1, 3, 0, 1, 0.80))
    graph.add_edge(_edge(2, 3, 1, 1, 0.80))

    assert graph.edge_is_supported(
        1,
        2,
        ambiguous_below=0.70,
        third_view_min_score=0.75,
    ) is False


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("observation_id", True, TypeError),
        ("observation_id", 1.5, TypeError),
        ("observation_id", -1, ValueError),
        ("frame_id", False, TypeError),
        ("frame_id", -1, ValueError),
    ],
)
def test_observation_node_validates_identifiers(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    values = {"observation_id": 1, "frame_id": 2}
    values[field] = value

    with pytest.raises(error, match=field):
        ObservationNode(**values)


def test_observation_node_is_frozen() -> None:
    node = _node(1, 2)

    with pytest.raises(FrozenInstanceError):
        node.frame_id = 3


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"left_observation_id": True}, TypeError, "left_observation_id"),
        ({"right_observation_id": -1}, ValueError, "right_observation_id"),
        ({"source_frame_id": 1.5}, TypeError, "source_frame_id"),
        ({"target_frame_id": -1}, ValueError, "target_frame_id"),
        ({"right_observation_id": 1}, ValueError, "different"),
        ({"source_frame_id": 3}, ValueError, "source_frame_id"),
        ({"score": True}, TypeError, "score"),
        ({"score": float("nan")}, ValueError, "score"),
        ({"score": float("inf")}, ValueError, "score"),
        ({"score": -0.01}, ValueError, "score"),
        ({"score": 1.01}, ValueError, "score"),
    ],
)
def test_observation_edge_validates_values(
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    values = {
        "left_observation_id": 1,
        "right_observation_id": 2,
        "source_frame_id": 1,
        "target_frame_id": 2,
        "score": 0.5,
    }
    values.update(changes)

    with pytest.raises(error, match=message):
        ObservationEdge(**values)


def test_observation_edge_accepts_score_boundaries_and_is_frozen() -> None:
    assert _edge(1, 2, 0, 0, 0.0).score == 0.0
    edge = _edge(1, 2, 0, 1, 1.0)
    assert edge.score == 1.0

    with pytest.raises(FrozenInstanceError):
        edge.score = 0.5


@pytest.mark.parametrize("window_size", [True, False, 0, -1, 1.5])
def test_graph_requires_a_positive_integer_window(window_size: object) -> None:
    with pytest.raises((TypeError, ValueError), match="window_size"):
        CausalObservationGraph(window_size=window_size)


@pytest.mark.parametrize("frame_id", [True, -1, 1.5])
def test_add_frame_validates_frame_identifier(frame_id: object) -> None:
    graph = CausalObservationGraph(window_size=2)

    with pytest.raises((TypeError, ValueError), match="frame_id"):
        graph.add_frame(frame_id, [])


def test_add_frame_requires_strictly_increasing_frame_ids() -> None:
    graph = CausalObservationGraph(window_size=2)
    graph.add_frame(4, [])

    for frame_id in (4, 3):
        with pytest.raises(ValueError, match="increasing"):
            graph.add_frame(frame_id, [])


def test_add_frame_rejects_node_with_mismatched_frame_metadata() -> None:
    graph = CausalObservationGraph(window_size=2)

    with pytest.raises(ValueError, match="frame_id"):
        graph.add_frame(4, [_node(1, 3)])


def test_add_frame_rejects_duplicate_incoming_observation_ids_atomically() -> None:
    graph = CausalObservationGraph(window_size=2)

    with pytest.raises(ValueError, match="duplicate"):
        graph.add_frame(0, [_node(1, 0), _node(1, 0)])

    assert graph.frame_ids == ()
    assert graph.observation_ids == ()


def test_add_frame_rejects_active_id_conflict_without_mutating_state() -> None:
    graph = CausalObservationGraph(window_size=2)
    graph.add_frame(0, [_node(1, 0)])

    with pytest.raises(ValueError, match="active"):
        graph.add_frame(1, [_node(2, 1), _node(1, 1)])

    assert graph.frame_ids == (0,)
    assert graph.observation_ids == (1,)
    graph.add_frame(1, [_node(2, 1)])
    assert graph.frame_ids == (0, 1)


def test_add_frame_rejects_non_node_without_mutating_state() -> None:
    graph = CausalObservationGraph(window_size=2)

    with pytest.raises(TypeError, match="ObservationNode"):
        graph.add_frame(0, [object()])

    assert graph.frame_ids == ()


def test_empty_frames_count_toward_the_window() -> None:
    graph = CausalObservationGraph(window_size=2)
    graph.add_frame(0, [_node(1, 0)])
    graph.add_frame(1, [])
    graph.add_frame(2, [])

    assert graph.frame_ids == (1, 2)
    assert graph.observation_ids == ()


def test_add_edge_rejects_future_target_before_missing_endpoints() -> None:
    graph = CausalObservationGraph(window_size=2)

    with pytest.raises(ValueError, match="future"):
        graph.add_edge(_edge(1, 2, 0, 0, 0.8))


def test_add_edge_requires_committed_frames() -> None:
    graph = CausalObservationGraph(window_size=3)
    graph.add_frame(0, [_node(1, 0)])
    graph.add_frame(2, [_node(2, 2)])

    with pytest.raises(ValueError, match="committed"):
        graph.add_edge(_edge(1, 2, 0, 1, 0.8))


def test_add_edge_requires_active_endpoints() -> None:
    graph = CausalObservationGraph(window_size=2)
    graph.add_frame(0, [_node(1, 0)])
    graph.add_frame(1, [])

    with pytest.raises(ValueError, match="endpoint"):
        graph.add_edge(_edge(1, 2, 0, 1, 0.8))


@pytest.mark.parametrize(
    "edge",
    [
        _edge(1, 2, 1, 1, 0.8),
        _edge(1, 2, 0, 0, 0.8),
    ],
)
def test_add_edge_requires_endpoint_frame_metadata_to_match(edge: ObservationEdge) -> None:
    graph = CausalObservationGraph(window_size=2)
    graph.add_frame(0, [_node(1, 0)])
    graph.add_frame(1, [_node(2, 1)])

    with pytest.raises(ValueError, match="frame metadata"):
        graph.add_edge(edge)


def test_only_a_strictly_higher_score_replaces_an_undirected_edge() -> None:
    graph = CausalObservationGraph(window_size=2)
    graph.add_frame(0, [_node(1, 0)])
    graph.add_frame(1, [_node(2, 1)])
    graph.add_edge(_edge(1, 2, 0, 1, 0.80))

    graph.add_edge(_edge(1, 2, 0, 1, 0.70))
    assert graph.edge_is_supported(
        2,
        1,
        ambiguous_below=0.75,
        third_view_min_score=1.0,
    )

    graph.add_edge(_edge(1, 2, 0, 1, 0.90))
    assert graph.edge_is_supported(
        1,
        2,
        ambiguous_below=0.85,
        third_view_min_score=1.0,
    )
    assert graph.edge_count == 1


def test_third_view_threshold_is_inclusive() -> None:
    graph = CausalObservationGraph(window_size=3)
    graph.add_frame(0, [_node(1, 0)])
    graph.add_frame(1, [_node(2, 1)])
    graph.add_frame(2, [_node(3, 2)])
    graph.add_edge(_edge(1, 2, 0, 1, 0.69))
    graph.add_edge(_edge(1, 3, 0, 2, 0.75))
    graph.add_edge(_edge(2, 3, 1, 2, 0.75))

    assert graph.edge_is_supported(
        1,
        2,
        ambiguous_below=0.70,
        third_view_min_score=0.75,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ambiguous_below", True),
        ("ambiguous_below", float("nan")),
        ("ambiguous_below", -0.01),
        ("ambiguous_below", 1.01),
        ("third_view_min_score", False),
        ("third_view_min_score", float("inf")),
        ("third_view_min_score", -0.01),
        ("third_view_min_score", 1.01),
    ],
)
def test_edge_support_validates_thresholds(field: str, value: object) -> None:
    graph = CausalObservationGraph(window_size=1)
    values = {"ambiguous_below": 0.70, "third_view_min_score": 0.75}
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        graph.edge_is_supported(1, 2, **values)
