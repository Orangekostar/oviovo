"""Supported-aware mutual-dominance resolution of ReScene queries to OVI entities."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from src.oviv2.rescene_supported_view import EntityCoverage
from src.oviv2.temporal_pair_reasoner import validate_query_evidence
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    PairRelation,
    TemporalQueryEvidence,
)

EvidenceKey = tuple[str, int, str]
EntityKey = tuple[int, str]


class ResolverError(ValueError):
    """Raised when query evidence and supported-domain provenance disagree."""


def _fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResolverError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ResolverError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class ResolverConfig:
    minimum_supported_entity_coverage: float = 0.25
    minimum_query_precision: float = 0.25
    minimum_query_confidence: float = 0.20
    static_centroid_tolerance_m: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "minimum_supported_entity_coverage",
            "minimum_query_precision",
            "minimum_query_confidence",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name=name))
        tolerance = self.static_centroid_tolerance_m
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise ResolverError("static_centroid_tolerance_m must be numeric")
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ResolverError(
                "static_centroid_tolerance_m must be finite and non-negative"
            )
        object.__setattr__(self, "static_centroid_tolerance_m", tolerance)


@dataclass(frozen=True, slots=True)
class ResolverDomainCoverage:
    full_source_count: int
    supported_source_count: int
    full_adapter_count: int
    supported_adapter_count: int
    full_model_count: int
    supported_model_count: int

    def __post_init__(self) -> None:
        for full_name, supported_name in (
            ("full_source_count", "supported_source_count"),
            ("full_adapter_count", "supported_adapter_count"),
            ("full_model_count", "supported_model_count"),
        ):
            full = getattr(self, full_name)
            supported = getattr(self, supported_name)
            if (
                type(full) is not int
                or type(supported) is not int
                or full <= 0
                or not 0 < supported <= full
            ):
                raise ResolverError("domain coverage counts are invalid")

    def fractions(self) -> dict[str, float]:
        return {
            "source_fraction": self.supported_source_count / self.full_source_count,
            "adapter_fraction": self.supported_adapter_count / self.full_adapter_count,
            "model_fraction": self.supported_model_count / self.full_model_count,
        }


@dataclass(frozen=True, slots=True)
class ResolverEntityEvidence:
    query_id: str
    visit_id: int
    entity_id: str
    token_intersection_count: int
    supported_entity_token_count: int
    full_entity_token_count: int
    supported_entity_token_coverage: float
    full_entity_token_evidence: float
    query_to_entity_precision: float
    selected_source_point_count: int
    supported_entity_source_point_count: int
    full_entity_source_point_count: int
    supported_source_point_coverage: float
    full_source_point_evidence: float
    soft_mass: float
    dominance_score: float


@dataclass(frozen=True, slots=True)
class SupportedResolverResult:
    relations: tuple[PairRelation, ...]
    evidence: Mapping[EvidenceKey, ResolverEntityEvidence]
    fallback_query_ids: tuple[str, ...]
    conflict_query_ids: tuple[str, ...]
    one_sided_query_ids: tuple[str, ...]
    collision_model_count: int
    duplicate_topology_count: int
    input_coverage: Mapping[str, float]
    conditional_query_coverage: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "relations", tuple(self.relations))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        object.__setattr__(self, "fallback_query_ids", tuple(self.fallback_query_ids))
        object.__setattr__(self, "conflict_query_ids", tuple(self.conflict_query_ids))
        object.__setattr__(self, "one_sided_query_ids", tuple(self.one_sided_query_ids))
        object.__setattr__(
            self, "input_coverage", MappingProxyType(dict(self.input_coverage))
        )


def _coverage_by_key(
    pair: NeuralSampleMap,
    values: Sequence[EntityCoverage],
    contributor_counts: np.ndarray,
) -> tuple[dict[EntityKey, EntityCoverage], tuple[EntityKey, ...], np.ndarray]:
    coverage = tuple(values)
    if not coverage or any(not isinstance(row, EntityCoverage) for row in coverage):
        raise ResolverError("entity coverage records are invalid")
    by_key = {(row.visit_id, row.entity_id): row for row in coverage}
    if len(by_key) != len(coverage):
        raise ResolverError("entity coverage keys are duplicated")
    token_keys = tuple(
        (int(visit), entity)
        for visit, entity in zip(pair.visit_ids, pair.token_entity_ids, strict=True)
    )
    supported_keys = tuple(sorted(set(token_keys)))
    if set(supported_keys) != {
        key for key, row in by_key.items() if row.supported_adapter_token_count > 0
    }:
        raise ResolverError("entity coverage does not match supported pair entities")
    code_by_key = {key: index for index, key in enumerate(supported_keys)}
    entity_codes = np.asarray([code_by_key[key] for key in token_keys], dtype=np.int32)
    observed_tokens = np.bincount(entity_codes, minlength=len(supported_keys))
    observed_sources = np.bincount(
        entity_codes,
        weights=contributor_counts,
        minlength=len(supported_keys),
    ).astype(np.int64)
    for index, key in enumerate(supported_keys):
        row = by_key[key]
        if (
            row.supported_adapter_token_count != observed_tokens[index]
            or row.supported_source_point_count != observed_sources[index]
        ):
            raise ResolverError("entity coverage numerators disagree with the pair")
    return by_key, supported_keys, entity_codes


def _validate_domains(
    pair: NeuralSampleMap,
    adapter_to_model: object,
    coverage: ResolverDomainCoverage,
    entity_coverage: Mapping[EntityKey, EntityCoverage],
) -> np.ndarray:
    if not isinstance(coverage, ResolverDomainCoverage):
        raise TypeError("domain_coverage must be ResolverDomainCoverage")
    raw = np.asarray(adapter_to_model)
    if (
        raw.ndim != 1
        or len(raw) != len(pair.visit_ids)
        or not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ResolverError("adapter-to-model mapping is invalid")
    mapping = raw.astype(np.int64, copy=False)
    if (
        np.any(mapping < 0)
        or np.any(mapping >= coverage.supported_model_count)
        or not np.array_equal(
            np.unique(mapping), np.arange(coverage.supported_model_count)
        )
    ):
        raise ResolverError("adapter-to-model mapping does not cover supported M")
    if (
        coverage.supported_adapter_count != len(pair.visit_ids)
        or coverage.supported_source_count != pair.source_point_count
        or sum(row.full_adapter_token_count for row in entity_coverage.values())
        != coverage.full_adapter_count
        or sum(row.full_source_point_count for row in entity_coverage.values())
        != coverage.full_source_count
    ):
        raise ResolverError("domain coverage disagrees with entity coverage or pair")
    return mapping


def _edge_order(record: ResolverEntityEvidence) -> tuple[float, float, float, str]:
    return (
        -record.dominance_score,
        -record.supported_entity_token_coverage,
        -record.query_to_entity_precision,
        record.entity_id,
    )


def _query_order(
    item: tuple[str, ResolverEntityEvidence], query_scores: Mapping[str, float]
) -> tuple[float, float, str]:
    query_id, record = item
    return (-record.dominance_score, -query_scores[query_id], query_id)


def _collision_count(
    pair: NeuralSampleMap, mapping: np.ndarray, model_count: int
) -> int:
    keys = tuple(
        (int(visit), entity)
        for visit, entity in zip(pair.visit_ids, pair.token_entity_ids, strict=True)
    )
    code_by_key = {key: index for index, key in enumerate(sorted(set(keys)))}
    codes = np.asarray([code_by_key[key] for key in keys], dtype=np.int32)
    minimum = np.full(model_count, np.iinfo(np.int32).max, dtype=np.int32)
    maximum = np.full(model_count, -1, dtype=np.int32)
    np.minimum.at(minimum, mapping, codes)
    np.maximum.at(maximum, mapping, codes)
    return int(np.count_nonzero(minimum != maximum))


def resolve_supported_queries(
    *,
    pair: NeuralSampleMap,
    evidence: TemporalQueryEvidence,
    entity_coverage: Sequence[EntityCoverage],
    adapter_to_model: object,
    domain_coverage: ResolverDomainCoverage,
    config: ResolverConfig,
) -> SupportedResolverResult:
    """Resolve trusted one-to-one relations without consulting evaluator GT."""

    if not isinstance(config, ResolverConfig):
        raise TypeError("config must be ResolverConfig")
    try:
        validate_query_evidence(pair, evidence)
    except (TypeError, ValueError) as error:
        raise ResolverError("query evidence does not bind the supplied pair") from error
    if evidence.status != "PASS" or evidence.query_masks is None:
        raise ResolverError("resolver requires PASS query evidence")
    assert evidence.token_scores is not None and evidence.query_scores is not None
    contributor_counts = np.diff(pair.source_to_token_offsets).astype(np.int64)
    by_key, supported_keys, entity_codes = _coverage_by_key(
        pair, entity_coverage, contributor_counts
    )
    mapping = _validate_domains(pair, adapter_to_model, domain_coverage, by_key)
    visits = pair.visit_ids
    query_scores = {
        query_id: float(evidence.query_scores[index])
        for index, query_id in enumerate(evidence.temporal_query_ids)
    }

    records: dict[EvidenceKey, ResolverEntityEvidence] = {}
    entity_index_by_key = {key: index for index, key in enumerate(supported_keys)}
    eligible: dict[tuple[str, int], list[ResolverEntityEvidence]] = {}
    any_intersection: dict[str, bool] = {}
    for query_index, query_id in enumerate(evidence.temporal_query_ids):
        mask = evidence.query_masks[query_index]
        intersections = np.bincount(
            entity_codes[mask], minlength=len(supported_keys)
        )
        selected_sources = np.bincount(
            entity_codes[mask],
            weights=contributor_counts[mask],
            minlength=len(supported_keys),
        ).astype(np.int64)
        soft_mass = np.bincount(
            entity_codes,
            weights=evidence.token_scores[query_index],
            minlength=len(supported_keys),
        )
        visit_query_counts = {
            visit_id: int(np.count_nonzero(mask & (visits == visit_id)))
            for visit_id in (0, 1)
        }
        any_intersection[query_id] = bool(np.any(intersections))
        for entity_index, (visit_id, entity_id) in enumerate(supported_keys):
            coverage = by_key[(visit_id, entity_id)]
            intersection = int(intersections[entity_index])
            selected_source = int(selected_sources[entity_index])
            entity_token_coverage = (
                intersection / coverage.supported_adapter_token_count
            )
            full_token_evidence = intersection / coverage.full_adapter_token_count
            query_precision = (
                intersection / visit_query_counts[visit_id]
                if visit_query_counts[visit_id]
                else 0.0
            )
            supported_source_coverage = (
                selected_source / coverage.supported_source_point_count
            )
            full_source_evidence = selected_source / coverage.full_source_point_count
            dominance = query_scores[query_id] * math.sqrt(
                entity_token_coverage * query_precision
            )
            record = ResolverEntityEvidence(
                query_id=query_id,
                visit_id=visit_id,
                entity_id=entity_id,
                token_intersection_count=intersection,
                supported_entity_token_count=coverage.supported_adapter_token_count,
                full_entity_token_count=coverage.full_adapter_token_count,
                supported_entity_token_coverage=entity_token_coverage,
                full_entity_token_evidence=full_token_evidence,
                query_to_entity_precision=query_precision,
                selected_source_point_count=selected_source,
                supported_entity_source_point_count=coverage.supported_source_point_count,
                full_entity_source_point_count=coverage.full_source_point_count,
                supported_source_point_coverage=supported_source_coverage,
                full_source_point_evidence=full_source_evidence,
                soft_mass=float(soft_mass[entity_index]),
                dominance_score=dominance,
            )
            records[(query_id, visit_id, entity_id)] = record
            if (
                intersection > 0
                and query_scores[query_id] >= config.minimum_query_confidence
                and entity_token_coverage
                >= config.minimum_supported_entity_coverage
                and query_precision >= config.minimum_query_precision
            ):
                eligible.setdefault((query_id, visit_id), []).append(record)

    top_by_query_visit = {
        key: sorted(values, key=_edge_order)[0] for key, values in eligible.items()
    }
    candidates_by_entity: dict[EntityKey, list[tuple[str, ResolverEntityEvidence]]] = {}
    for (query_id, visit_id), values in eligible.items():
        for record in values:
            candidates_by_entity.setdefault(
                (visit_id, record.entity_id), []
            ).append((query_id, record))
    top_by_entity = {
        key: sorted(values, key=lambda item: _query_order(item, query_scores))[0][0]
        for key, values in candidates_by_entity.items()
    }
    mutual: dict[tuple[str, int], ResolverEntityEvidence] = {}
    for key, record in top_by_query_visit.items():
        query_id, visit_id = key
        if top_by_entity[(visit_id, record.entity_id)] == query_id:
            mutual[key] = record

    raw_topologies = [
        (
            top_by_query_visit[(query_id, 0)].entity_id,
            top_by_query_visit[(query_id, 1)].entity_id,
        )
        for query_id in evidence.temporal_query_ids
        if (query_id, 0) in top_by_query_visit
        and (query_id, 1) in top_by_query_visit
    ]
    topology_counts = Counter(raw_topologies)
    duplicate_topology_count = sum(count - 1 for count in topology_counts.values())

    relations: list[PairRelation] = []
    fallback: list[str] = []
    conflicts: list[str] = []
    one_sided: list[str] = []
    for query_index, query_id in enumerate(evidence.temporal_query_ids):
        eligible_visits = {
            visit_id for visit_id in (0, 1) if (query_id, visit_id) in eligible
        }
        if (query_id, 0) in mutual and (query_id, 1) in mutual:
            before = mutual[(query_id, 0)]
            after = mutual[(query_id, 1)]
            centroids = []
            for record in (before, after):
                selected = (
                    evidence.query_masks[query_index]
                    & (visits == record.visit_id)
                    & (
                        entity_codes
                        == entity_index_by_key[(record.visit_id, record.entity_id)]
                    )
                )
                centroids.append(
                    np.average(
                        pair.coordinates_xyzt[selected, :3],
                        axis=0,
                        weights=contributor_counts[selected],
                    )
                )
            distance = float(np.linalg.norm(centroids[0] - centroids[1]))
            state = (
                "persistent_static"
                if distance <= config.static_centroid_tolerance_m
                else "persistent_moved"
            )
            relations.append(
                PairRelation(
                    temporal_query_id=query_id,
                    t0_entity_ids=(before.entity_id,),
                    t1_entity_ids=(after.entity_id,),
                    state=state,
                    query_confidence=float(evidence.query_scores[query_index]),
                    evidence={
                        "centroid_distance_m": distance,
                        "t0_dominance_score": before.dominance_score,
                        "t1_dominance_score": after.dominance_score,
                        "t0_full_entity_evidence": before.full_entity_token_evidence,
                        "t1_full_entity_evidence": after.full_entity_token_evidence,
                    },
                    identity_source="rescene",
                )
            )
        elif eligible_visits == {0, 1}:
            conflicts.append(query_id)
        elif len(eligible_visits) == 1:
            one_sided.append(query_id)
        elif any_intersection[query_id]:
            fallback.append(query_id)

    input_coverage = domain_coverage.fractions()
    input_coverage["entity_fraction"] = sum(
        row.supported_adapter_token_count > 0 for row in by_key.values()
    ) / len(by_key)
    return SupportedResolverResult(
        relations=tuple(relations),
        evidence=records,
        fallback_query_ids=tuple(fallback),
        conflict_query_ids=tuple(conflicts),
        one_sided_query_ids=tuple(one_sided),
        collision_model_count=_collision_count(
            pair, mapping, domain_coverage.supported_model_count
        ),
        duplicate_topology_count=duplicate_topology_count,
        input_coverage=input_coverage,
        conditional_query_coverage=len(relations) / len(evidence.temporal_query_ids),
    )


__all__ = [
    "ResolverConfig",
    "ResolverDomainCoverage",
    "ResolverEntityEvidence",
    "ResolverError",
    "SupportedResolverResult",
    "resolve_supported_queries",
]
