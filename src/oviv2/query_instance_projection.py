"""Lossless projection of temporal token queries back to OVI entities."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from src.oviv2.temporal_pair_reasoner import validate_query_evidence
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    PairRelation,
    TemporalQueryEvidence,
)


EntityKey = tuple[int, str]
MatrixKey = tuple[str, str, int]


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    minimum_entity_token_coverage: float = 0.25
    minimum_source_point_coverage: float = 0.25
    static_centroid_tolerance_m: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "minimum_entity_token_coverage",
            "minimum_source_point_coverage",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, normalized)
        tolerance = self.static_centroid_tolerance_m
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise ValueError("static_centroid_tolerance_m must be numeric")
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("static_centroid_tolerance_m must be finite and non-negative")
        object.__setattr__(self, "static_centroid_tolerance_m", tolerance)


@dataclass(frozen=True, slots=True)
class QueryEntityEvidence:
    query_id: str
    entity_id: str
    visit_id: int
    token_intersection_count: int
    entity_token_count: int
    entity_token_coverage: float
    query_mask_fraction: float
    soft_mass: float
    source_point_count: int
    entity_source_point_count: int
    source_point_coverage: float
    centroid_xyz: np.ndarray | None
    semantic_compatibility: float | None

    def __post_init__(self) -> None:
        if self.centroid_xyz is not None:
            centroid = np.array(self.centroid_xyz, dtype=np.float64, copy=True)
            if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
                raise ValueError("centroid_xyz must contain three finite values")
            centroid.setflags(write=False)
            object.__setattr__(self, "centroid_xyz", centroid)


class QueryEvidenceMatrix(Mapping[MatrixKey, QueryEntityEvidence]):
    def __init__(self, records: Mapping[MatrixKey, QueryEntityEvidence]) -> None:
        self._records = MappingProxyType(dict(records))

    def __getitem__(self, key: MatrixKey) -> QueryEntityEvidence:
        return self._records[key]

    def __iter__(self) -> Iterator[MatrixKey]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    relations: tuple[PairRelation, ...]
    evidence_matrix: QueryEvidenceMatrix


def _entity_token_indices(pair: NeuralSampleMap) -> dict[EntityKey, np.ndarray]:
    groups: dict[EntityKey, list[int]] = {}
    for token_index, entity_id in enumerate(pair.token_entity_ids):
        key = (int(pair.visit_ids[token_index]), entity_id)
        groups.setdefault(key, []).append(token_index)
    return {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in sorted(groups.items())
    }


def _project(
    pair: NeuralSampleMap,
    evidence: TemporalQueryEvidence,
) -> QueryEvidenceMatrix:
    validate_query_evidence(pair, evidence)
    if evidence.status != "PASS" or evidence.query_masks is None:
        raise ValueError("query projection requires PASS temporal evidence")
    assert evidence.token_scores is not None
    groups = _entity_token_indices(pair)
    contributor_counts = np.diff(pair.source_to_token_offsets).astype(np.int64)
    records: dict[MatrixKey, QueryEntityEvidence] = {}
    for query_index, query_id in enumerate(evidence.temporal_query_ids):
        mask = evidence.query_masks[query_index]
        query_token_count = int(np.count_nonzero(mask))
        if query_token_count == 0:
            raise ValueError(f"temporal query {query_id} selects no tokens")
        for (visit_id, entity_id), indices in groups.items():
            selected = indices[mask[indices]]
            token_count = len(indices)
            intersection = len(selected)
            entity_source_count = int(contributor_counts[indices].sum())
            selected_source_count = int(contributor_counts[selected].sum())
            centroid = None
            if intersection:
                centroid = np.average(
                    pair.coordinates_xyzt[selected, :3],
                    axis=0,
                    weights=contributor_counts[selected],
                )
            records[(query_id, entity_id, visit_id)] = QueryEntityEvidence(
                query_id=query_id,
                entity_id=entity_id,
                visit_id=visit_id,
                token_intersection_count=intersection,
                entity_token_count=token_count,
                entity_token_coverage=intersection / token_count,
                query_mask_fraction=intersection / query_token_count,
                soft_mass=float(evidence.token_scores[query_index, indices].sum()),
                source_point_count=selected_source_count,
                entity_source_point_count=entity_source_count,
                source_point_coverage=(
                    selected_source_count / entity_source_count
                    if entity_source_count
                    else 0.0
                ),
                centroid_xyz=centroid,
                semantic_compatibility=None,
            )
    return QueryEvidenceMatrix(records)


def project_query_evidence(
    pair: NeuralSampleMap,
    evidence: TemporalQueryEvidence,
) -> QueryEvidenceMatrix:
    """Return the complete query-by-entity-by-visit evidence matrix."""

    return _project(pair, evidence)


def _selected_records(
    matrix: QueryEvidenceMatrix,
    query_id: str,
    config: ProjectionConfig,
) -> tuple[QueryEntityEvidence, ...]:
    records = tuple(
        record
        for key, record in matrix.items()
        if key[0] == query_id and record.token_intersection_count > 0
    )
    selected = tuple(
        record
        for record in records
        if record.entity_token_coverage >= config.minimum_entity_token_coverage
        or record.source_point_coverage >= config.minimum_source_point_coverage
    )
    return tuple(sorted(selected or records, key=lambda item: (item.visit_id, item.entity_id)))


def _relation_state(
    t0: tuple[QueryEntityEvidence, ...],
    t1: tuple[QueryEntityEvidence, ...],
    config: ProjectionConfig,
) -> tuple[str, float | None]:
    cardinality = (len(t0), len(t1))
    if cardinality == (1, 1):
        assert t0[0].centroid_xyz is not None and t1[0].centroid_xyz is not None
        distance = float(np.linalg.norm(t0[0].centroid_xyz - t1[0].centroid_xyz))
        state = (
            "persistent_static"
            if distance <= config.static_centroid_tolerance_m
            else "persistent_moved"
        )
        return state, distance
    if cardinality == (0, 1):
        return "appeared", None
    if cardinality == (1, 0):
        return "removed_candidate", None
    if cardinality[0] == 1 and cardinality[1] > 1:
        return "split", None
    if cardinality[0] > 1 and cardinality[1] == 1:
        return "merge", None
    return "uncertain", None


def project_queries_to_instances(
    pair: NeuralSampleMap,
    evidence: TemporalQueryEvidence,
    config: ProjectionConfig,
) -> ProjectionResult:
    """Project query support and derive explicit cross-visit relation topology."""

    if not isinstance(config, ProjectionConfig):
        raise TypeError("config must be ProjectionConfig")
    matrix = _project(pair, evidence)
    assert evidence.query_scores is not None
    if evidence.backend_name.startswith("rescene:"):
        identity_source = "rescene"
    elif evidence.backend_name == "geometric_semantic":
        identity_source = "geometric_baseline"
    else:
        identity_source = "unmatched"
    relations: list[PairRelation] = []
    for query_index, query_id in enumerate(evidence.temporal_query_ids):
        selected = _selected_records(matrix, query_id, config)
        t0 = tuple(item for item in selected if item.visit_id == 0)
        t1 = tuple(item for item in selected if item.visit_id == 1)
        state, centroid_distance = _relation_state(t0, t1, config)
        coverage_values = [item.source_point_coverage for item in selected]
        relation_evidence = {
            "query_score": float(evidence.query_scores[query_index]),
            "selected_entity_count": float(len(selected)),
            "minimum_source_point_coverage": (
                min(coverage_values) if coverage_values else 0.0
            ),
            "soft_mass": sum(item.soft_mass for item in selected),
        }
        if centroid_distance is not None:
            relation_evidence["centroid_distance_m"] = centroid_distance
        relations.append(
            PairRelation(
                temporal_query_id=query_id,
                t0_entity_ids=tuple(item.entity_id for item in t0),
                t1_entity_ids=tuple(item.entity_id for item in t1),
                state=state,
                query_confidence=float(evidence.query_scores[query_index]),
                evidence=relation_evidence,
                identity_source=identity_source,
            )
        )
    return ProjectionResult(tuple(relations), matrix)
