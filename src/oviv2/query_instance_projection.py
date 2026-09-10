"""Lossless projection of temporal token queries back to OVI entities."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from src.oviv2.entity_epoch_update import RelationInferenceState, RelationSupport
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


def _strict_owner_id(entity_id: str) -> int:
    prefix, separator, suffix = entity_id.partition(":")
    if prefix != "ovimap" or separator != ":":
        raise ValueError("strict ReScene projection requires ovimap:<positive-int> IDs")
    try:
        owner_id = int(suffix)
    except ValueError as error:
        raise ValueError(
            "strict ReScene projection requires ovimap:<positive-int> IDs"
        ) from error
    if owner_id <= 0:
        raise ValueError("strict ReScene projection requires positive OVI owner IDs")
    return owner_id


def _source_local_rows(pair: NeuralSampleMap) -> np.ndarray:
    rows = np.empty(pair.source_point_count, dtype=np.int64)
    for visit_id in (0, 1):
        positions = np.flatnonzero(pair.source_visit_ids == visit_id)
        source_indices = pair.source_point_indices[positions]
        rows[positions] = np.searchsorted(np.sort(source_indices), source_indices)
    return rows


def _strict_query_source_rows(
    pair: NeuralSampleMap,
    query_mask: np.ndarray,
    *,
    visit_id: int,
    entity_id: str,
    local_rows: np.ndarray,
    entity_token_indices: Mapping[EntityKey, np.ndarray],
) -> np.ndarray:
    candidate_tokens = entity_token_indices[(visit_id, entity_id)]
    token_indices = candidate_tokens[query_mask[candidate_tokens]]
    positions = np.concatenate(
        tuple(
            np.arange(
                pair.source_to_token_offsets[index],
                pair.source_to_token_offsets[index + 1],
                dtype=np.int64,
            )
            for index in token_indices
        )
    )
    return np.sort(local_rows[positions])


def _strict_query_rank(
    records: tuple[QueryEntityEvidence, ...],
) -> tuple[QueryEntityEvidence, float, float]:
    ordered = sorted(
        records,
        key=lambda item: (
            -item.source_point_coverage,
            -item.entity_token_coverage,
            -item.soft_mass,
            item.entity_id,
        ),
    )
    best = ordered[0]
    competing_score = ordered[1].source_point_coverage if len(ordered) > 1 else 0.0
    return best, competing_score, best.source_point_coverage - competing_score


def _strict_one_to_one(
    relations: tuple[RelationSupport, ...],
) -> tuple[RelationSupport, ...]:
    resolved = list(relations)
    used_t0: set[int] = set()
    used_t1: set[int] = set()
    for index, relation in sorted(
        enumerate(relations),
        key=lambda item: (
            -item[1].confidence,
            -(item[1].competition_margin or 0.0),
            item[1].relation_id,
        ),
    ):
        if not relation.accepted:
            continue
        if (
            relation.t0_owner_entity_id in used_t0
            or relation.t1_owner_entity_id in used_t1
        ):
            resolved[index] = replace(
                relation,
                accepted=False,
                assignment_is_null=True,
                rejection_reasons=("one_to_one_conflict",),
            )
            continue
        used_t0.add(relation.t0_owner_entity_id)
        used_t1.add(relation.t1_owner_entity_id)
    return tuple(resolved)


def project_queries_to_relation_support(
    pair: NeuralSampleMap,
    evidence: TemporalQueryEvidence,
    *,
    minimum_source_coverage: float = 0.25,
    minimum_competition_margin: float = 0.08,
    minimum_query_confidence: float = 0.0,
) -> tuple[RelationSupport, ...]:
    """Project frozen ReScene queries onto exact fixed-OVI source rows."""

    if not isinstance(pair, NeuralSampleMap):
        raise TypeError("pair must be NeuralSampleMap")
    if not isinstance(evidence, TemporalQueryEvidence):
        raise TypeError("evidence must be TemporalQueryEvidence")
    if not evidence.backend_name.startswith("rescene:"):
        raise ValueError("strict query projection requires ReScene evidence")
    for value, name in (
        (minimum_source_coverage, "minimum_source_coverage"),
        (minimum_competition_margin, "minimum_competition_margin"),
        (minimum_query_confidence, "minimum_query_confidence"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    matrix = _project(pair, evidence)
    assert evidence.query_masks is not None
    assert evidence.query_scores is not None
    local_rows = _source_local_rows(pair)
    entity_token_indices = _entity_token_indices(pair)
    pair_identity = pair.content_sha256()[:12]
    output: list[RelationSupport] = []
    for query_index, query_id in enumerate(evidence.temporal_query_ids):
        intersected = tuple(
            record
            for (record_query_id, _entity_id, _visit_id), record in matrix.items()
            if record_query_id == query_id and record.token_intersection_count > 0
        )
        t0_records = tuple(record for record in intersected if record.visit_id == 0)
        t1_records = tuple(record for record in intersected if record.visit_id == 1)
        if not t0_records or not t1_records:
            continue
        t0_record, t0_competing, t0_margin = _strict_query_rank(t0_records)
        t1_record, t1_competing, t1_margin = _strict_query_rank(t1_records)
        competition_margin = min(t0_margin, t1_margin)
        query_confidence = float(evidence.query_scores[query_index])
        reasons: set[str] = set()
        if t0_record.source_point_coverage < minimum_source_coverage:
            reasons.add("insufficient_t0_coverage")
        if t1_record.source_point_coverage < minimum_source_coverage:
            reasons.add("insufficient_t1_coverage")
        if competition_margin < minimum_competition_margin:
            reasons.add("ambiguous_query_competition")
        if query_confidence < minimum_query_confidence:
            reasons.add("below_query_score")
        accepted = not reasons
        t0_owner = _strict_owner_id(t0_record.entity_id)
        t1_owner = _strict_owner_id(t1_record.entity_id)
        output.append(
            RelationSupport(
                relation_id=(
                    f"frozen-rescene:{pair_identity}:{query_id}:{t0_owner}:{t1_owner}"
                ),
                relation_source="frozen-rescene",
                stable_entity_id=f"stable:frozen-rescene:{pair_identity}:{query_id}",
                t0_owner_entity_id=t0_owner,
                t1_owner_entity_id=t1_owner,
                t0_source_surface_id=f"ovi-map:{pair.source_visit_map_sha256[0]}",
                t1_source_surface_id=f"ovi-map:{pair.source_visit_map_sha256[1]}",
                t0_source_vertex_indices=_strict_query_source_rows(
                    pair,
                    evidence.query_masks[query_index],
                    visit_id=0,
                    entity_id=t0_record.entity_id,
                    local_rows=local_rows,
                    entity_token_indices=entity_token_indices,
                ),
                t1_source_vertex_indices=_strict_query_source_rows(
                    pair,
                    evidence.query_masks[query_index],
                    visit_id=1,
                    entity_id=t1_record.entity_id,
                    local_rows=local_rows,
                    entity_token_indices=entity_token_indices,
                ),
                confidence=min(
                    query_confidence,
                    t0_record.source_point_coverage,
                    t1_record.source_point_coverage,
                ),
                inference_state=RelationInferenceState.UNRESOLVED,
                accepted=accepted,
                t0_entity_id=t0_record.entity_id,
                t1_entity_id=t1_record.entity_id,
                assignment_is_null=not accepted,
                assignment_margin=competition_margin,
                t0_mask_coverage=t0_record.source_point_coverage,
                t1_mask_coverage=t1_record.source_point_coverage,
                t0_mask_purity=t0_record.query_mask_fraction,
                t1_mask_purity=t1_record.query_mask_fraction,
                t0_competing_score=t0_competing,
                t1_competing_score=t1_competing,
                query_confidence=query_confidence,
                competition_margin=competition_margin,
                rejection_reasons=tuple(sorted(reasons)),
            )
        )
    return _strict_one_to_one(tuple(output))
