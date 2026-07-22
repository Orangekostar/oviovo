from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from numbers import Integral, Real
from typing import Literal

import numpy as np

from src.evaluation.oviv2_view_graph import ObservationEdge, select_observation_edges
from src.oviv2.addressing import VoxelKey
from src.oviv2.observations import FrameObservation, ObservationKind


ProposalKind = Literal["consensus", "core", "union", "hierarchical", "hybrid"]

_KIND_PRIORITY = {"core": 0, "consensus": 1, "union": 2, "hierarchical": 3, "hybrid": 4}


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _bounded_real(value: object, name: str, lower: float, upper: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if not lower <= normalized <= upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    return normalized


@dataclass(frozen=True)
class ProposalPyramidConfig:
    strong_voxel_iou: float = 0.05
    strong_directed_coverage: float = 0.25
    minimum_feature_cosine: float = 0.20
    minimum_distinct_views: int = 2
    core_vote_fraction: float = 0.50
    inclusive_coverage: float = 0.25
    weak_merge_iou: float = 0.10
    minimum_voxels: int = 10
    maximum_proposals: int = 1024

    def __post_init__(self) -> None:
        for name in (
            "strong_voxel_iou",
            "strong_directed_coverage",
            "core_vote_fraction",
            "inclusive_coverage",
            "weak_merge_iou",
        ):
            object.__setattr__(self, name, _bounded_real(getattr(self, name), name, 0.0, 1.0))
        object.__setattr__(
            self,
            "minimum_feature_cosine",
            _bounded_real(self.minimum_feature_cosine, "minimum_feature_cosine", -1.0, 1.0),
        )
        for name in ("minimum_distinct_views", "minimum_voxels", "maximum_proposals"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))


@dataclass(frozen=True)
class VoxelProposal:
    proposal_id: str
    semantic_id: int
    kind: ProposalKind
    voxel_keys: frozenset[VoxelKey]
    observation_ids: tuple[int, ...]
    supporter_frame_ids: tuple[int, ...]
    mean_confidence: float
    consensus_density: float
    border_fraction: float

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, str):
            raise TypeError("proposal_id must be a string")
        if not self.proposal_id.strip():
            raise ValueError("proposal_id must be nonempty")
        object.__setattr__(self, "semantic_id", _positive_integer(self.semantic_id, "semantic_id"))
        if self.kind not in _KIND_PRIORITY:
            raise ValueError("kind is not a supported proposal kind")
        try:
            voxel_keys = frozenset(self.voxel_keys)
        except TypeError as exc:
            raise TypeError("voxel_keys must be an iterable of VoxelKey values") from exc
        if not voxel_keys:
            raise ValueError("voxel_keys must be nonempty")
        object.__setattr__(self, "voxel_keys", voxel_keys)
        object.__setattr__(
            self,
            "observation_ids",
            _sorted_ids(self.observation_ids, "observation_ids"),
        )
        object.__setattr__(
            self,
            "supporter_frame_ids",
            _sorted_ids(self.supporter_frame_ids, "supporter_frame_ids"),
        )
        if not self.observation_ids or not self.supporter_frame_ids:
            raise ValueError("observation_ids and supporter_frame_ids must be nonempty")
        for name in ("mean_confidence", "consensus_density", "border_fraction"):
            object.__setattr__(self, name, _bounded_real(getattr(self, name), name, 0.0, 1.0))
        if not np.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must lie in [0, 1]")

    @property
    def score(self) -> float:
        distinct_views = len(self.supporter_frame_ids)
        return (
            0.30 * min(distinct_views / 8.0, 1.0)
            + 0.25 * self.mean_confidence
            + 0.25 * self.consensus_density
            + 0.20 * (1.0 - self.border_fraction)
        )


def _sorted_ids(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be an iterable of IDs")
    normalized = tuple(_nonnegative_integer(item, name) for item in value)
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{name} must contain sorted unique IDs")
    return normalized


class _UnionFind:
    def __init__(self, elements: Iterable[int]) -> None:
        self.parent = {element: element for element in elements}

    def find(self, element: int) -> int:
        parent = self.parent[element]
        if parent != element:
            self.parent[element] = self.find(parent)
        return self.parent[element]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


class _StrongEdgeIndex:
    """Sparse canonical-left adjacency for extracting component-internal edges."""

    def __init__(self, edges: Iterable[ObservationEdge]) -> None:
        by_left: dict[int, list[ObservationEdge]] = defaultdict(list)
        for edge in edges:
            by_left[edge.left_id].append(edge)
        self.by_left = {
            observation_id: tuple(sorted(values, key=lambda edge: edge.right_id))
            for observation_id, values in by_left.items()
        }

    def edges_for(self, observation_ids: tuple[int, ...]) -> tuple[ObservationEdge, ...]:
        members = frozenset(observation_ids)
        return tuple(
            edge
            for observation_id in observation_ids
            for edge in self.by_left.get(observation_id, ())
            if edge.right_id in members
        )


@dataclass(frozen=True)
class _Candidate:
    semantic_id: int
    kind: ProposalKind
    voxel_keys: frozenset[VoxelKey]
    observation_ids: tuple[int, ...]
    strong_edges: tuple[ObservationEdge, ...]


def _core_overlap_fraction(core: frozenset[VoxelKey], candidate: FrameObservation) -> float:
    return len(core & candidate.voxel_keys) / len(core)


def _validate_inputs(
    observations: Sequence[FrameObservation],
    evidence: Sequence[ObservationEdge],
    config: ProposalPyramidConfig,
) -> tuple[dict[int, FrameObservation], tuple[ObservationEdge, ...]]:
    if not isinstance(observations, Sequence):
        raise TypeError("observations must be a sequence of FrameObservation values")
    if not isinstance(evidence, Sequence):
        raise TypeError("evidence must be a sequence of ObservationEdge values")
    if not isinstance(config, ProposalPyramidConfig):
        raise TypeError("config must be a ProposalPyramidConfig")
    by_id: dict[int, FrameObservation] = {}
    for observation in observations:
        if not isinstance(observation, FrameObservation):
            raise TypeError("observations must contain only FrameObservation values")
        if observation.observation_id in by_id:
            raise ValueError(f"duplicate observation_id {observation.observation_id}")
        by_id[observation.observation_id] = observation
    pairs: set[tuple[int, int]] = set()
    normalized_edges: list[ObservationEdge] = []
    for edge in evidence:
        if not isinstance(edge, ObservationEdge):
            raise TypeError("evidence must contain only ObservationEdge values")
        pair = edge.left_id, edge.right_id
        if pair in pairs:
            raise ValueError(f"duplicate evidence edge pair {pair}")
        pairs.add(pair)
        if edge.left_id not in by_id or edge.right_id not in by_id:
            raise ValueError("evidence endpoint does not identify an observation")
        if by_id[edge.left_id].frame_id == by_id[edge.right_id].frame_id:
            raise ValueError("evidence endpoints cannot belong to the same frame")
        if by_id[edge.left_id].semantic_id != by_id[edge.right_id].semantic_id:
            raise ValueError("evidence endpoints must have compatible semantics")
        normalized_edges.append(edge)
    return by_id, tuple(sorted(normalized_edges, key=lambda edge: (edge.left_id, edge.right_id)))


def _components(
    observation_ids: tuple[int, ...],
    strong_edges: tuple[ObservationEdge, ...],
) -> tuple[tuple[int, ...], ...]:
    union_find = _UnionFind(observation_ids)
    for edge in strong_edges:
        union_find.union(edge.left_id, edge.right_id)
    grouped: dict[int, list[int]] = defaultdict(list)
    for observation_id in observation_ids:
        grouped[union_find.find(observation_id)].append(observation_id)
    return tuple(tuple(sorted(ids)) for _, ids in sorted(grouped.items()))


def _component_is_eligible(
    observation_ids: tuple[int, ...],
    observations: dict[int, FrameObservation],
    config: ProposalPyramidConfig,
) -> bool:
    return (
        len({observations[observation_id].frame_id for observation_id in observation_ids})
        >= config.minimum_distinct_views
        and len(set().union(*(observations[observation_id].voxel_keys for observation_id in observation_ids)))
        >= config.minimum_voxels
    )


def _strong_edges_for(
    observation_ids: tuple[int, ...],
    edge_index: _StrongEdgeIndex,
) -> tuple[ObservationEdge, ...]:
    return edge_index.edges_for(observation_ids)


def _required_core_votes(vote_fraction: float, distinct_view_count: int) -> int:
    """Ceiling vote threshold from the user-visible decimal configuration."""
    threshold = Decimal(str(vote_fraction)) * distinct_view_count
    return int(threshold.to_integral_value(rounding=ROUND_CEILING))


def _candidate(
    semantic_id: int,
    kind: ProposalKind,
    voxel_keys: frozenset[VoxelKey],
    observation_ids: Iterable[int],
    strong_edges: Iterable[ObservationEdge],
) -> _Candidate:
    return _Candidate(
        semantic_id=semantic_id,
        kind=kind,
        voxel_keys=frozenset(voxel_keys),
        observation_ids=tuple(sorted(set(observation_ids))),
        strong_edges=tuple(sorted(strong_edges, key=lambda edge: (edge.left_id, edge.right_id))),
    )


def _select_inclusive_observation_ids(
    semantic_id: int,
    core: frozenset[VoxelKey],
    inclusive_coverage: float,
    postings: dict[tuple[int, VoxelKey], list[int]],
    eligible_by_semantic: dict[int, tuple[int, ...]],
) -> tuple[int, ...]:
    if inclusive_coverage == 0.0:
        return eligible_by_semantic[semantic_id]
    overlap_counts: dict[int, int] = defaultdict(int)
    for voxel_key in core:
        for observation_id in postings[semantic_id, voxel_key]:
            overlap_counts[observation_id] += 1
    return tuple(
        observation_id
        for observation_id in sorted(overlap_counts)
        if overlap_counts[observation_id] / len(core) >= inclusive_coverage
    )


def _build_union_candidate(
    semantic_id: int,
    selected_ids: tuple[int, ...],
    observations: dict[int, FrameObservation],
    strong_edge_index: _StrongEdgeIndex,
) -> _Candidate:
    union_voxels = frozenset().union(*(observations[observation_id].voxel_keys for observation_id in selected_ids))
    return _candidate(
        semantic_id,
        "union",
        union_voxels,
        selected_ids,
        _strong_edges_for(selected_ids, strong_edge_index),
    )


def _proposal_from_candidate(candidate: _Candidate, observations: dict[int, FrameObservation]) -> VoxelProposal:
    source = tuple(observations[observation_id] for observation_id in candidate.observation_ids)
    frame_counts: dict[int, int] = defaultdict(int)
    for observation in source:
        frame_counts[observation.frame_id] += 1
    frame_ids = tuple(sorted(frame_counts))
    possible_pairs = len(source) * (len(source) - 1) // 2 - sum(
        count * (count - 1) // 2 for count in frame_counts.values()
    )
    density = 0.0 if possible_pairs == 0 else len(candidate.strong_edges) / possible_pairs
    density = float(np.clip(density, 0.0, 1.0))
    return VoxelProposal(
        proposal_id="pending",
        semantic_id=candidate.semantic_id,
        kind=candidate.kind,
        voxel_keys=candidate.voxel_keys,
        observation_ids=candidate.observation_ids,
        supporter_frame_ids=frame_ids,
        mean_confidence=sum(observation.confidence for observation in source) / len(source),
        consensus_density=density,
        border_fraction=sum(observation.border_contact_fraction for observation in source) / len(source),
    )


def _candidate_choice_key(
    candidate: _Candidate,
    observations: dict[int, FrameObservation],
) -> tuple[float | int | tuple[int, ...] | tuple[tuple[int, int], ...], ...]:
    return (
        _KIND_PRIORITY[candidate.kind],
        -_proposal_from_candidate(candidate, observations).score,
        candidate.observation_ids,
        tuple((edge.left_id, edge.right_id) for edge in candidate.strong_edges),
    )


class _RetainedCandidates:
    def __init__(self, observations: dict[int, FrameObservation], maximum_proposals: int) -> None:
        self._observations = observations
        self._maximum_proposals = maximum_proposals
        self._retained: dict[tuple[int, frozenset[VoxelKey]], _Candidate] = {}
        self._choice_keys: dict[_Candidate, tuple[float | int | tuple[int, ...] | tuple[tuple[int, int], ...], ...]] = {}

    def add(self, candidate: _Candidate) -> None:
        key = candidate.semantic_id, candidate.voxel_keys
        previous = self._retained.get(key)
        if previous is None:
            if len(self._retained) >= self._maximum_proposals:
                raise ValueError("post-dedup proposal count exceeds maximum_proposals")
            self._retained[key] = candidate
            return
        candidate_key = self._choice_keys.setdefault(
            candidate,
            _candidate_choice_key(candidate, self._observations),
        )
        previous_key = self._choice_keys.setdefault(
            previous,
            _candidate_choice_key(previous, self._observations),
        )
        if candidate_key < previous_key:
            self._retained[key] = candidate

    def values(self) -> tuple[_Candidate, ...]:
        return tuple(self._retained.values())


def _assign_ids(candidates: Iterable[_Candidate], observations: dict[int, FrameObservation]) -> tuple[VoxelProposal, ...]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.semantic_id,
            _KIND_PRIORITY[candidate.kind],
            min(candidate.voxel_keys),
            candidate.observation_ids,
            tuple(sorted(candidate.voxel_keys)),
        ),
    )
    proposals: list[VoxelProposal] = []
    for index, candidate in enumerate(ordered):
        proposal = _proposal_from_candidate(candidate, observations)
        proposals.append(
            VoxelProposal(
                proposal_id=f"vc:{proposal.semantic_id:03d}:{proposal.kind}:{index:04d}",
                semantic_id=proposal.semantic_id,
                kind=proposal.kind,
                voxel_keys=proposal.voxel_keys,
                observation_ids=proposal.observation_ids,
                supporter_frame_ids=proposal.supporter_frame_ids,
                mean_confidence=proposal.mean_confidence,
                consensus_density=proposal.consensus_density,
                border_fraction=proposal.border_fraction,
            )
        )
    return tuple(proposals)


def build_proposal_pyramid(
    observations: Sequence[FrameObservation],
    evidence: Sequence[ObservationEdge],
    config: ProposalPyramidConfig,
) -> tuple[VoxelProposal, ...]:
    by_id, all_edges = _validate_inputs(observations, evidence, config)
    eligible = tuple(
        sorted(
            (
                observation
                for observation in by_id.values()
                if observation.kind is ObservationKind.OBJECT and observation.semantic_id > 0
            ),
            key=lambda observation: observation.observation_id,
        )
    )
    if not eligible:
        return ()
    eligible_ids = frozenset(observation.observation_id for observation in eligible)
    relevant_edges = tuple(
        edge for edge in all_edges if edge.left_id in eligible_ids and edge.right_id in eligible_ids
    )
    strong_edges = tuple(
        edge
        for edge in select_observation_edges(
            relevant_edges,
            minimum_voxel_iou=config.strong_voxel_iou,
            minimum_directed_coverage=config.strong_directed_coverage,
            minimum_feature_cosine=config.minimum_feature_cosine,
        )
        if by_id[edge.left_id].semantic_id == by_id[edge.right_id].semantic_id
    )
    retained = _RetainedCandidates(by_id, config.maximum_proposals)
    component_records: list[tuple[int, tuple[int, ...]]] = []
    postings: dict[tuple[int, VoxelKey], list[int]] = defaultdict(list)
    eligible_by_semantic: dict[int, tuple[int, ...]] = {}
    semantic_observation_ids: dict[int, list[int]] = defaultdict(list)
    for observation in eligible:
        semantic_observation_ids[observation.semantic_id].append(observation.observation_id)
        for voxel_key in observation.voxel_keys:
            postings[observation.semantic_id, voxel_key].append(observation.observation_id)
    for semantic_id, observation_ids in semantic_observation_ids.items():
        eligible_by_semantic[semantic_id] = tuple(observation_ids)
    strong_edge_index = _StrongEdgeIndex(strong_edges)
    inclusive_selection_cache: dict[tuple[int, frozenset[VoxelKey]], tuple[int, ...]] = {}
    union_candidate_cache: dict[tuple[int, tuple[int, ...]], _Candidate] = {}

    for component in _components(tuple(observation.observation_id for observation in eligible), strong_edges):
        semantic_id = by_id[component[0]].semantic_id
        if any(by_id[observation_id].semantic_id != semantic_id for observation_id in component):
            continue
        component_records.append((semantic_id, component))
        if not _component_is_eligible(component, by_id, config):
            continue
        component_edges = _strong_edges_for(component, strong_edge_index)
        consensus = frozenset().union(*(by_id[observation_id].voxel_keys for observation_id in component))
        retained.add(_candidate(semantic_id, "consensus", consensus, component, component_edges))
        by_frame: dict[int, set[VoxelKey]] = defaultdict(set)
        for observation_id in component:
            observation = by_id[observation_id]
            by_frame[observation.frame_id].update(observation.voxel_keys)
        required_votes = _required_core_votes(config.core_vote_fraction, len(by_frame))
        votes: dict[VoxelKey, int] = defaultdict(int)
        for voxel_keys in by_frame.values():
            for voxel_key in voxel_keys:
                votes[voxel_key] += 1
        core = frozenset(voxel_key for voxel_key, count in votes.items() if count >= required_votes)
        if len(core) < config.minimum_voxels:
            continue
        retained.add(_candidate(semantic_id, "core", core, component, component_edges))
        selection_key = semantic_id, core
        selected_ids = inclusive_selection_cache.get(selection_key)
        if selected_ids is None:
            selected_ids = _select_inclusive_observation_ids(
                semantic_id,
                core,
                config.inclusive_coverage,
                postings,
                eligible_by_semantic,
            )
            inclusive_selection_cache[selection_key] = selected_ids
        if not selected_ids:
            continue
        union_key = semantic_id, selected_ids
        union_candidate = union_candidate_cache.get(union_key)
        if union_candidate is None:
            union_candidate = _build_union_candidate(
                semantic_id,
                selected_ids,
                by_id,
                strong_edge_index,
            )
            union_candidate_cache[union_key] = union_candidate
        retained.add(union_candidate)

    component_index = {
        observation_id: component_index
        for component_index, (_, component) in enumerate(component_records)
        for observation_id in component
    }
    component_union_find = _UnionFind(range(len(component_records)))
    strong_pairs = {(edge.left_id, edge.right_id) for edge in strong_edges}
    for edge in all_edges:
        left_component = component_index.get(edge.left_id)
        right_component = component_index.get(edge.right_id)
        if left_component is None or right_component is None or left_component == right_component:
            continue
        if (edge.left_id, edge.right_id) in strong_pairs:
            continue
        if edge.voxel_iou < config.weak_merge_iou:
            continue
        if edge.feature_cosine is not None and edge.feature_cosine < config.minimum_feature_cosine:
            continue
        if component_records[left_component][0] != component_records[right_component][0]:
            continue
        component_union_find.union(left_component, right_component)
    hierarchical_groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(component_records)):
        hierarchical_groups[component_union_find.find(index)].append(index)
    for component_indexes in hierarchical_groups.values():
        if len(component_indexes) < 2:
            continue
        semantic_id = component_records[component_indexes[0]][0]
        observation_ids = tuple(sorted(
            observation_id
            for index in component_indexes
            for observation_id in component_records[index][1]
        ))
        if not _component_is_eligible(observation_ids, by_id, config):
            continue
        voxel_keys = frozenset().union(*(by_id[observation_id].voxel_keys for observation_id in observation_ids))
        retained.add(
            _candidate(
                semantic_id,
                "hierarchical",
                voxel_keys,
                observation_ids,
                _strong_edges_for(observation_ids, strong_edge_index),
            )
        )
    return _assign_ids(retained.values(), by_id)
