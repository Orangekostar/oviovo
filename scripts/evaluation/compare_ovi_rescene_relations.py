#!/usr/bin/env python3
"""Compare frozen geometric B4 and source-bound ReScene B5 relations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

import numpy as np

from src.oviv2.geometric_pair_reasoner import (
    GeometricPairReasoner,
    GeometricReasonerConfig,
)
from src.oviv2.query_instance_projection import (
    ProjectionConfig,
    project_queries_to_instances,
    project_query_evidence,
)
from src.oviv2.rescene_input_bridge import NativeSamplingMap
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    PairRelation,
    TemporalQueryEvidence,
    VisitMap,
)
from src.oviv2.two_visit_execution import build_geometric_pair_sample

TopologyKey = tuple[str, tuple[str, ...], tuple[str, ...]]
_PERSISTENT_STATES = frozenset({"persistent_static", "persistent_moved"})


class RelationComparisonError(ValueError):
    """Raised when relation evidence cannot be compared without ambiguity."""


@dataclass(frozen=True, slots=True)
class RelationDelta:
    b4_raw_count: int
    b4_unique_count: int
    b5_raw_count: int
    b5_unique_count: int
    b4_state_counts: dict[str, int]
    b5_state_counts: dict[str, int]
    intersection: tuple[TopologyKey, ...]
    b4_only: tuple[TopologyKey, ...]
    b5_only: tuple[TopologyKey, ...]
    b4_unique_persistent_one_to_one_count: int
    b5_unique_persistent_one_to_one_count: int


@dataclass(frozen=True, slots=True)
class B5RelationDiagnostic:
    temporal_query_id: str
    topology: TopologyKey
    threshold_qualified: bool
    fallback_only: bool
    contradictory: bool
    collision_fraction: float
    collision_dominated: bool


def topology_key(relation: PairRelation) -> TopologyKey:
    if not isinstance(relation, PairRelation):
        raise TypeError("relation must be PairRelation")
    return (
        relation.state,
        tuple(sorted(relation.t0_entity_ids)),
        tuple(sorted(relation.t1_entity_ids)),
    )


def _relations(
    values: Sequence[PairRelation], *, label: str
) -> tuple[PairRelation, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a relation sequence")
    result = tuple(values)
    if any(not isinstance(value, PairRelation) for value in result):
        raise TypeError(f"{label} contains a non-relation value")
    return result


def _unique_topologies(relations: tuple[PairRelation, ...]) -> set[TopologyKey]:
    return {topology_key(value) for value in relations}


def _state_counts(values: set[TopologyKey]) -> dict[str, int]:
    return dict(sorted(Counter(key[0] for key in values).items()))


def _persistent_one_to_one_count(values: set[TopologyKey]) -> int:
    return sum(
        key[0] in _PERSISTENT_STATES and len(key[1]) == 1 and len(key[2]) == 1
        for key in values
    )


def compare_relation_topologies(
    b4: Sequence[PairRelation], b5: Sequence[PairRelation]
) -> RelationDelta:
    """Compare unique relation topology without conflating duplicate queries."""

    b4_relations = _relations(b4, label="B4")
    b5_relations = _relations(b5, label="B5")
    b4_keys = _unique_topologies(b4_relations)
    b5_keys = _unique_topologies(b5_relations)
    return RelationDelta(
        b4_raw_count=len(b4_relations),
        b4_unique_count=len(b4_keys),
        b5_raw_count=len(b5_relations),
        b5_unique_count=len(b5_keys),
        b4_state_counts=_state_counts(b4_keys),
        b5_state_counts=_state_counts(b5_keys),
        intersection=tuple(sorted(b4_keys & b5_keys)),
        b4_only=tuple(sorted(b4_keys - b5_keys)),
        b5_only=tuple(sorted(b5_keys - b4_keys)),
        b4_unique_persistent_one_to_one_count=_persistent_one_to_one_count(b4_keys),
        b5_unique_persistent_one_to_one_count=_persistent_one_to_one_count(b5_keys),
    )


def rebuild_b4_relations(
    earlier: VisitMap,
    later: VisitMap,
    *,
    neural_voxel_size_m: float = 0.02,
    reasoner_config: GeometricReasonerConfig | None = None,
    projection_config: ProjectionConfig | None = None,
) -> tuple[PairRelation, ...]:
    """Rebuild B4 once through the unchanged geometry reasoner and projector."""

    pair = build_geometric_pair_sample(
        earlier, later, neural_voxel_size_m=neural_voxel_size_m
    )
    evidence = GeometricPairReasoner(
        reasoner_config or GeometricReasonerConfig()
    ).infer(pair)
    return project_queries_to_instances(
        pair, evidence, projection_config or ProjectionConfig()
    ).relations


def _validate_sampling(pair: NeuralSampleMap, sampling: NativeSamplingMap) -> None:
    if not isinstance(sampling, NativeSamplingMap):
        raise TypeError("sampling must be NativeSamplingMap")
    mapping = sampling.adapter_to_model
    if len(mapping) != len(pair.visit_ids):
        raise RelationComparisonError("sampling and pair adapter counts differ")
    if not np.array_equal(sampling.model_visit_ids[mapping], pair.visit_ids):
        raise RelationComparisonError("sampling and pair visits differ")


def _contradictory_query_ids(
    relations: tuple[PairRelation, ...],
) -> frozenset[str]:
    endpoints: Counter[tuple[int, str]] = Counter()
    eligible: list[PairRelation] = []
    for relation in relations:
        if (
            relation.state in _PERSISTENT_STATES
            and len(relation.t0_entity_ids) == 1
            and len(relation.t1_entity_ids) == 1
        ):
            eligible.append(relation)
            endpoints[(0, relation.t0_entity_ids[0])] += 1
            endpoints[(1, relation.t1_entity_ids[0])] += 1
    return frozenset(
        str(relation.temporal_query_id)
        for relation in eligible
        if endpoints[(0, relation.t0_entity_ids[0])] > 1
        or endpoints[(1, relation.t1_entity_ids[0])] > 1
    )


def classify_b5_relations(
    *,
    pair: NeuralSampleMap,
    evidence: TemporalQueryEvidence,
    relations: Sequence[PairRelation],
    sampling: NativeSamplingMap,
    projection_config: ProjectionConfig,
    collision_dominance_threshold: float = 0.5,
) -> dict[str, B5RelationDiagnostic]:
    """Classify B5 relations without changing the frozen projection output."""

    if not isinstance(pair, NeuralSampleMap):
        raise TypeError("pair must be NeuralSampleMap")
    if not isinstance(evidence, TemporalQueryEvidence):
        raise TypeError("evidence must be TemporalQueryEvidence")
    if not isinstance(projection_config, ProjectionConfig):
        raise TypeError("projection_config must be ProjectionConfig")
    threshold = float(collision_dominance_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise RelationComparisonError("collision dominance threshold must be in [0, 1]")
    _validate_sampling(pair, sampling)
    try:
        matrix = project_query_evidence(pair, evidence)
    except ValueError as error:
        raise RelationComparisonError(
            "B5 evidence does not bind the supplied pair"
        ) from error
    relation_values = _relations(relations, label="B5")
    query_ids = tuple(str(value.temporal_query_id) for value in relation_values)
    if len(query_ids) != len(set(query_ids)):
        raise RelationComparisonError("B5 relation query IDs must be unique")
    evidence_indices = {
        query_id: index for index, query_id in enumerate(evidence.temporal_query_ids)
    }
    if any(query_id not in evidence_indices for query_id in query_ids):
        raise RelationComparisonError("B5 relation query is absent from evidence")
    contradictory = _contradictory_query_ids(relation_values)
    entity_codes = np.empty(len(pair.visit_ids), dtype=np.int32)
    code_by_key: dict[tuple[int, str], int] = {}
    for index, entity_id in enumerate(pair.token_entity_ids):
        key = (int(pair.visit_ids[index]), entity_id)
        entity_codes[index] = code_by_key.setdefault(key, len(code_by_key))
    minimum_codes = np.full(
        sampling.model_count, np.iinfo(np.int32).max, dtype=np.int32
    )
    maximum_codes = np.full(sampling.model_count, -1, dtype=np.int32)
    np.minimum.at(minimum_codes, sampling.adapter_to_model, entity_codes)
    np.maximum.at(maximum_codes, sampling.adapter_to_model, entity_codes)
    model_collision = minimum_codes != maximum_codes

    diagnostics: dict[str, B5RelationDiagnostic] = {}
    for relation, query_id in zip(relation_values, query_ids, strict=True):
        records = []
        for visit_id, entity_ids in (
            (0, relation.t0_entity_ids),
            (1, relation.t1_entity_ids),
        ):
            for entity_id in entity_ids:
                try:
                    records.append(matrix[(query_id, entity_id, visit_id)])
                except KeyError as error:
                    raise RelationComparisonError(
                        "B5 relation entity is absent from projection evidence"
                    ) from error
        qualified = [
            record.entity_token_coverage
            >= projection_config.minimum_entity_token_coverage
            or record.source_point_coverage
            >= projection_config.minimum_source_point_coverage
            for record in records
        ]
        query_mask = evidence.query_masks[evidence_indices[query_id]]
        selected_models = np.unique(sampling.adapter_to_model[query_mask])
        collision_fraction = (
            float(np.mean(model_collision[selected_models]))
            if len(selected_models)
            else 0.0
        )
        threshold_qualified = any(qualified)
        diagnostics[query_id] = B5RelationDiagnostic(
            temporal_query_id=query_id,
            topology=topology_key(relation),
            threshold_qualified=threshold_qualified,
            fallback_only=bool(records) and not threshold_qualified,
            contradictory=query_id in contradictory,
            collision_fraction=collision_fraction,
            collision_dominated=(
                bool(len(selected_models)) and collision_fraction >= threshold
            ),
        )
    return diagnostics


def select_registration_candidates(
    relations: Sequence[PairRelation],
    diagnostics: Mapping[str, B5RelationDiagnostic],
    *,
    b5_only: AbstractSet[TopologyKey],
    maximum_count: int = 20,
) -> tuple[PairRelation, ...]:
    """Select at most 20 qualified, noncontradictory B5-only 1:1 relations."""

    relation_values = _relations(relations, label="B5")
    if type(maximum_count) is not int or not 1 <= maximum_count <= 20:
        raise RelationComparisonError("registration candidate limit must be in [1, 20]")
    if not isinstance(diagnostics, Mapping):
        raise TypeError("diagnostics must be a mapping")
    candidates: list[PairRelation] = []
    for relation in relation_values:
        query_id = str(relation.temporal_query_id)
        diagnostic = diagnostics.get(query_id)
        if not isinstance(diagnostic, B5RelationDiagnostic):
            raise RelationComparisonError("relation diagnostic is missing")
        if (
            topology_key(relation) in b5_only
            and relation.state in _PERSISTENT_STATES
            and len(relation.t0_entity_ids) == 1
            and len(relation.t1_entity_ids) == 1
            and diagnostic.threshold_qualified
            and not diagnostic.fallback_only
            and not diagnostic.contradictory
        ):
            candidates.append(relation)
    candidates.sort(
        key=lambda value: (
            -value.query_confidence,
            value.t0_entity_ids,
            value.t1_entity_ids,
            str(value.temporal_query_id),
        )
    )
    return tuple(candidates[:maximum_count])


__all__ = [
    "B5RelationDiagnostic",
    "RelationComparisonError",
    "RelationDelta",
    "classify_b5_relations",
    "compare_relation_topologies",
    "rebuild_b4_relations",
    "select_registration_candidates",
    "topology_key",
]
