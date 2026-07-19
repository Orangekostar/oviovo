from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Iterable


def _non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return normalized


def _unit_interval(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be finite and lie in [0, 1]")
    return normalized


def _edge_key(left_observation_id: int, right_observation_id: int) -> tuple[int, int]:
    return tuple(sorted((left_observation_id, right_observation_id)))


@dataclass(frozen=True)
class ObservationNode:
    observation_id: int
    frame_id: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _non_negative_integer(self.observation_id, "observation_id"),
        )
        object.__setattr__(self, "frame_id", _non_negative_integer(self.frame_id, "frame_id"))


@dataclass(frozen=True)
class ObservationEdge:
    left_observation_id: int
    right_observation_id: int
    source_frame_id: int
    target_frame_id: int
    score: float

    def __post_init__(self) -> None:
        left_observation_id = _non_negative_integer(
            self.left_observation_id,
            "left_observation_id",
        )
        right_observation_id = _non_negative_integer(
            self.right_observation_id,
            "right_observation_id",
        )
        source_frame_id = _non_negative_integer(self.source_frame_id, "source_frame_id")
        target_frame_id = _non_negative_integer(self.target_frame_id, "target_frame_id")
        if left_observation_id == right_observation_id:
            raise ValueError("edge endpoints must be different observations")
        if source_frame_id > target_frame_id:
            raise ValueError("source_frame_id must not exceed target_frame_id")
        object.__setattr__(self, "left_observation_id", left_observation_id)
        object.__setattr__(self, "right_observation_id", right_observation_id)
        object.__setattr__(self, "source_frame_id", source_frame_id)
        object.__setattr__(self, "target_frame_id", target_frame_id)
        object.__setattr__(self, "score", _unit_interval(self.score, "score"))


class CausalObservationGraph:
    def __init__(self, window_size: int) -> None:
        window_size = _non_negative_integer(window_size, "window_size")
        if window_size == 0:
            raise ValueError("window_size must be positive")
        self._window_size = window_size
        self._nodes: dict[int, ObservationNode] = {}
        self._frame_ids: list[int] = []
        self._edges: dict[tuple[int, int], ObservationEdge] = {}

    @property
    def observation_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._nodes))

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(self._frame_ids)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def add_frame(self, frame_id: int, nodes: Iterable[ObservationNode]) -> None:
        frame_id = _non_negative_integer(frame_id, "frame_id")
        if self._frame_ids and frame_id <= self._frame_ids[-1]:
            raise ValueError("frame_id must be strictly increasing")
        try:
            incoming = tuple(nodes)
        except TypeError as error:
            raise TypeError("nodes must be an iterable of ObservationNode values") from error

        incoming_ids: set[int] = set()
        for node in incoming:
            if not isinstance(node, ObservationNode):
                raise TypeError("nodes must contain only ObservationNode values")
            if node.frame_id != frame_id:
                raise ValueError("node frame_id must match the committed frame_id")
            if node.observation_id in incoming_ids:
                raise ValueError("duplicate incoming observation_id")
            if node.observation_id in self._nodes:
                raise ValueError("incoming observation_id conflicts with an active node")
            incoming_ids.add(node.observation_id)

        self._frame_ids.append(frame_id)
        self._nodes.update((node.observation_id, node) for node in incoming)
        while len(self._frame_ids) > self._window_size:
            self._expire_frame(self._frame_ids.pop(0))

    def add_edge(self, edge: ObservationEdge) -> None:
        if not isinstance(edge, ObservationEdge):
            raise TypeError("edge must be an ObservationEdge")
        if not self._frame_ids or edge.target_frame_id > self._frame_ids[-1]:
            raise ValueError("cannot add an edge targeting a future frame")
        active_frames = set(self._frame_ids)
        if edge.source_frame_id not in active_frames or edge.target_frame_id not in active_frames:
            raise ValueError("edge source and target frames must be committed")
        left = self._nodes.get(edge.left_observation_id)
        right = self._nodes.get(edge.right_observation_id)
        if left is None or right is None:
            raise ValueError("edge endpoints must both be active")
        if left.frame_id != edge.source_frame_id or right.frame_id != edge.target_frame_id:
            raise ValueError("edge frame metadata must match its endpoint nodes")

        key = _edge_key(edge.left_observation_id, edge.right_observation_id)
        existing = self._edges.get(key)
        if existing is None or edge.score > existing.score:
            self._edges[key] = edge

    def remove_edge(self, left_observation_id: int, right_observation_id: int) -> bool:
        left_observation_id = _non_negative_integer(
            left_observation_id,
            "left_observation_id",
        )
        right_observation_id = _non_negative_integer(
            right_observation_id,
            "right_observation_id",
        )
        if left_observation_id == right_observation_id:
            return False
        key = _edge_key(left_observation_id, right_observation_id)
        return self._edges.pop(key, None) is not None

    def edge_is_supported(
        self,
        left_observation_id: int,
        right_observation_id: int,
        *,
        ambiguous_below: float,
        third_min: float,
    ) -> bool:
        ambiguous_below = _unit_interval(ambiguous_below, "ambiguous_below")
        third_min = _unit_interval(third_min, "third_min")
        left_observation_id = _non_negative_integer(
            left_observation_id,
            "left_observation_id",
        )
        right_observation_id = _non_negative_integer(
            right_observation_id,
            "right_observation_id",
        )
        if left_observation_id == right_observation_id:
            return False

        edge = self._edges.get(_edge_key(left_observation_id, right_observation_id))
        if edge is None:
            return False
        if edge.score >= ambiguous_below:
            return True

        left = self._nodes[left_observation_id]
        right = self._nodes[right_observation_id]
        endpoint_frames = {left.frame_id, right.frame_id}
        for third_id, third in self._nodes.items():
            if third_id in {left_observation_id, right_observation_id}:
                continue
            if third.frame_id in endpoint_frames:
                continue
            left_support = self._edges.get(_edge_key(left_observation_id, third_id))
            right_support = self._edges.get(_edge_key(right_observation_id, third_id))
            if (
                left_support is not None
                and right_support is not None
                and left_support.score >= third_min
                and right_support.score >= third_min
            ):
                return True
        return False

    def _expire_frame(self, frame_id: int) -> None:
        expired_ids = {
            observation_id
            for observation_id, node in self._nodes.items()
            if node.frame_id == frame_id
        }
        for observation_id in expired_ids:
            del self._nodes[observation_id]
        for key in tuple(self._edges):
            if expired_ids.intersection(key):
                del self._edges[key]
