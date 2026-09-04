"""Deterministic geometry and OVI-semantics temporal query baseline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from src.oviv2.temporal_pair_reasoner import validate_query_evidence
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    OviEntitySemanticEvidence,
    TemporalQueryEvidence,
)


@dataclass(frozen=True, slots=True)
class GeometricReasonerConfig:
    comparison_voxel_size_m: float = 0.05
    maximum_centroid_distance_m: float = 2.0
    minimum_pair_score: float = 0.50
    require_known_label_compatibility: bool = True
    voxel_iou_weight: float = 0.20
    symmetric_coverage_weight: float = 0.10
    centroid_weight: float = 0.25
    size_weight: float = 0.15
    extent_weight: float = 0.10
    semantic_weight: float = 0.20

    def __post_init__(self) -> None:
        positive = ("comparison_voxel_size_m", "maximum_centroid_distance_m")
        probabilities = (
            "minimum_pair_score",
            "voxel_iou_weight",
            "symmetric_coverage_weight",
            "centroid_weight",
            "size_weight",
            "extent_weight",
            "semantic_weight",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        for name in probabilities:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, normalized)
        total = sum(float(getattr(self, name)) for name in probabilities[1:])
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise ValueError("geometric reasoner weights must sum to 1")
        if type(self.require_known_label_compatibility) is not bool:
            raise ValueError("require_known_label_compatibility must be boolean")

    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class _EntitySummary:
    key: tuple[int, str]
    token_indices: tuple[int, ...]
    centroid: np.ndarray
    extent: np.ndarray
    voxel_keys: frozenset[tuple[int, int, int]]
    source_point_count: int
    semantic: OviEntitySemanticEvidence | None


def _entity_summaries(
    pair: NeuralSampleMap, voxel_size_m: float
) -> dict[tuple[int, str], _EntitySummary]:
    semantics = {
        (item.visit_id, item.entity_id): item for item in pair.entity_semantics
    }
    token_entities = pair.token_entity_ids
    contributor_counts = np.diff(pair.source_to_token_offsets).astype(np.float64)
    token_groups: dict[tuple[int, str], list[int]] = {}
    for token_index, entity_id in enumerate(token_entities):
        key = (int(pair.visit_ids[token_index]), entity_id)
        token_groups.setdefault(key, []).append(token_index)
    summaries: dict[tuple[int, str], _EntitySummary] = {}
    for key in sorted(token_groups):
        indices = np.asarray(token_groups[key], dtype=np.int64)
        xyz = pair.coordinates_xyzt[indices, :3]
        weights = contributor_counts[indices]
        centroid = np.average(xyz, axis=0, weights=weights)
        extent = xyz.max(axis=0) - xyz.min(axis=0)
        quantized = np.floor(xyz / voxel_size_m).astype(np.int64)
        voxels = frozenset(tuple(int(item) for item in row) for row in quantized)
        summaries[key] = _EntitySummary(
            key=key,
            token_indices=tuple(int(item) for item in indices),
            centroid=centroid,
            extent=extent,
            voxel_keys=voxels,
            source_point_count=int(weights.sum()),
            semantic=semantics.get(key),
        )
    return summaries


def _ratio(first: float, second: float) -> float:
    if first == second == 0.0:
        return 1.0
    maximum = max(first, second)
    return min(first, second) / maximum if maximum > 0.0 else 0.0


def _cosine(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None or first.shape != second.shape:
        return None
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0.0:
        return None
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _pair_score(
    first: _EntitySummary,
    second: _EntitySummary,
    config: GeometricReasonerConfig,
) -> float | None:
    distance = float(np.linalg.norm(first.centroid - second.centroid))
    if distance > config.maximum_centroid_distance_m:
        return None
    first_label = first.semantic.semantic_label if first.semantic is not None else None
    second_label = second.semantic.semantic_label if second.semantic is not None else None
    if (
        config.require_known_label_compatibility
        and first_label is not None
        and second_label is not None
        and first_label.casefold() != second_label.casefold()
    ):
        return None
    intersection = len(first.voxel_keys & second.voxel_keys)
    union = len(first.voxel_keys | second.voxel_keys)
    iou = intersection / union if union else 0.0
    coverage = 0.5 * (
        intersection / len(first.voxel_keys) + intersection / len(second.voxel_keys)
    )
    centroid_score = 1.0 - distance / config.maximum_centroid_distance_m
    size_ratio = _ratio(first.source_point_count, second.source_point_count)
    extent_ratio = _ratio(float(np.linalg.norm(first.extent)), float(np.linalg.norm(second.extent)))
    embedding_cosine = _cosine(
        first.semantic.semantic_embedding if first.semantic is not None else None,
        second.semantic.semantic_embedding if second.semantic is not None else None,
    )
    if first_label is not None and second_label is not None:
        label_score = float(first_label.casefold() == second_label.casefold())
    else:
        label_score = 0.5
    embedding_score = 0.5 if embedding_cosine is None else (embedding_cosine + 1.0) / 2.0
    semantic_score = 0.5 * (label_score + embedding_score)
    return float(
        config.voxel_iou_weight * iou
        + config.symmetric_coverage_weight * coverage
        + config.centroid_weight * centroid_score
        + config.size_weight * size_ratio
        + config.extent_weight * extent_ratio
        + config.semantic_weight * semantic_score
    )


def _components(
    summaries: dict[tuple[int, str], _EntitySummary],
    config: GeometricReasonerConfig,
) -> tuple[tuple[tuple[tuple[int, str], ...], float], ...]:
    t0 = tuple(key for key in sorted(summaries) if key[0] == 0)
    t1 = tuple(key for key in sorted(summaries) if key[0] == 1)
    adjacency: dict[tuple[int, str], set[tuple[int, str]]] = {
        key: set() for key in (*t0, *t1)
    }
    edge_scores: dict[frozenset[tuple[int, str]], float] = {}
    for first_key in t0:
        for second_key in t1:
            score = _pair_score(summaries[first_key], summaries[second_key], config)
            if score is None or score < config.minimum_pair_score:
                continue
            adjacency[first_key].add(second_key)
            adjacency[second_key].add(first_key)
            edge_scores[frozenset((first_key, second_key))] = score
    components: list[tuple[tuple[tuple[int, str], ...], float]] = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        pending = [start]
        members: set[tuple[int, str]] = set()
        while pending:
            current = pending.pop()
            if current in members:
                continue
            members.add(current)
            pending.extend(sorted(adjacency[current] - members, reverse=True))
        remaining -= members
        ordered = tuple(sorted(members))
        scores = [
            score
            for edge, score in edge_scores.items()
            if edge.issubset(members)
        ]
        confidence = max(scores) if scores else 1.0
        components.append((ordered, confidence))
    components.sort(key=lambda item: min(item[0]))
    return tuple(components)


class GeometricPairReasoner:
    """Build temporal queries from conservative entity-level evidence."""

    def __init__(self, config: GeometricReasonerConfig) -> None:
        if not isinstance(config, GeometricReasonerConfig):
            raise TypeError("config must be GeometricReasonerConfig")
        self._config = config

    def infer(self, pair: NeuralSampleMap) -> TemporalQueryEvidence:
        if not isinstance(pair, NeuralSampleMap):
            raise TypeError("pair must be a NeuralSampleMap")
        summaries = _entity_summaries(pair, self._config.comparison_voxel_size_m)
        components = _components(summaries, self._config)
        query_masks = np.zeros(
            (len(components), len(pair.visit_ids)), dtype=np.bool_
        )
        token_scores = np.zeros(query_masks.shape, dtype=np.float32)
        query_scores = np.empty(len(components), dtype=np.float32)
        query_ids: list[str] = []
        for query_index, (members, confidence) in enumerate(components):
            query_id = f"geom:{query_index:06d}"
            query_ids.append(query_id)
            query_scores[query_index] = confidence
            for member in members:
                indices = summaries[member].token_indices
                query_masks[query_index, list(indices)] = True
                token_scores[query_index, list(indices)] = confidence
        evidence = TemporalQueryEvidence(
            status="PASS",
            backend_name="geometric_semantic",
            backend_config_sha256=self._config.sha256(),
            pair_sha256=pair.content_sha256(),
            temporal_query_ids=tuple(query_ids),
            query_masks=query_masks,
            token_scores=token_scores,
            query_scores=query_scores,
            checkpoint_sha256=None,
            ranking_eligible=True,
            runtime_s=0.0,
            peak_memory_bytes=0,
            diagnostics={"relation_builder": "bipartite_connected_components"},
        )
        validate_query_evidence(pair, evidence)
        return evidence
