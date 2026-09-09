"""Semantic readout strategies for the canonical fine surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial import cKDTree

from src.oviv2.current_surface import SemanticSource


class SurfaceSemanticStrategy(str, Enum):
    S0 = "s0_nearest"
    S1 = "s1_surface_vote"
    S2 = "s2_reliable_surface_owner_fallback"


@dataclass(frozen=True, slots=True)
class SurfaceSemanticConfig:
    strategy: SurfaceSemanticStrategy
    maximum_correspondence_distance_m: float = 0.05
    distance_sigma_m: float = 0.025
    minimum_local_support: float = 2.0
    entity_weight_scale: float = 0.75
    minimum_entity_confidence: float = 0.2

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", SurfaceSemanticStrategy(self.strategy))
        if self.maximum_correspondence_distance_m <= 0.0:
            raise ValueError("maximum_correspondence_distance_m must be positive")
        if self.distance_sigma_m <= 0.0 or self.minimum_local_support <= 0.0:
            raise ValueError("distance sigma and minimum support must be positive")
        if not 0.0 <= self.entity_weight_scale <= 1.0:
            raise ValueError("entity_weight_scale must lie in [0, 1]")
        if not 0.0 <= self.minimum_entity_confidence <= 1.0:
            raise ValueError("minimum_entity_confidence must lie in [0, 1]")


def _immutable(value: object, dtype: object) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    if result.flags.writeable or result.base is not None:
        result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SurfaceSemanticResult:
    semantic_ids: np.ndarray
    semantic_confidences: np.ndarray
    support_reliabilities: np.ndarray
    semantic_source_codes: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_ids", _immutable(self.semantic_ids, np.int32))
        object.__setattr__(
            self,
            "semantic_confidences",
            _immutable(self.semantic_confidences, np.float32),
        )
        object.__setattr__(
            self,
            "support_reliabilities",
            _immutable(self.support_reliabilities, np.float32),
        )
        object.__setattr__(
            self,
            "semantic_source_codes",
            _immutable(self.semantic_source_codes, np.uint8),
        )


def _validated_inputs(
    local_semantic_ids: np.ndarray,
    local_supports: np.ndarray,
    local_distances_m: np.ndarray,
    entity_semantic_ids: np.ndarray,
    entity_confidences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(local_semantic_ids, dtype=np.int32)
    supports = np.asarray(local_supports, dtype=np.float32)
    distances = np.asarray(local_distances_m, dtype=np.float32)
    entity_ids = np.asarray(entity_semantic_ids, dtype=np.int32)
    entity_conf = np.asarray(entity_confidences, dtype=np.float32)
    if ids.ndim != 2 or supports.shape != ids.shape or distances.shape != ids.shape:
        raise ValueError("local semantic arrays must have equal shape (N, K)")
    count = len(ids)
    if entity_ids.shape != (count,) or entity_conf.shape != (count,):
        raise ValueError("entity semantic arrays must have shape (N,)")
    if np.any(ids < 0) or np.any(entity_ids < 0) or np.any(supports < 0.0):
        raise ValueError("semantic IDs and supports must be nonnegative")
    if not np.isfinite(supports).all():
        raise ValueError("local supports must be finite")
    if np.isnan(distances).any() or np.any(distances < 0.0):
        raise ValueError("local distances must be nonnegative and not NaN")
    if not np.isfinite(entity_conf).all() or np.any((entity_conf < 0.0) | (entity_conf > 1.0)):
        raise ValueError("entity confidences must be finite and lie in [0, 1]")
    return ids, supports, distances, entity_ids, entity_conf


def read_surface_semantics(
    *,
    local_semantic_ids: np.ndarray,
    local_supports: np.ndarray,
    local_distances_m: np.ndarray,
    entity_semantic_ids: np.ndarray,
    entity_confidences: np.ndarray,
    config: SurfaceSemanticConfig,
) -> SurfaceSemanticResult:
    """Read bounded local semantics, optionally backed by reliable owner semantics."""

    if not isinstance(config, SurfaceSemanticConfig):
        raise TypeError("config must be SurfaceSemanticConfig")
    ids, supports, distances, entity_ids, entity_conf = _validated_inputs(
        local_semantic_ids,
        local_supports,
        local_distances_m,
        entity_semantic_ids,
        entity_confidences,
    )
    count, neighbors = ids.shape
    valid = (
        (ids > 0)
        & (supports > 0.0)
        & np.isfinite(distances)
        & (distances <= config.maximum_correspondence_distance_m)
    )
    source = np.full(count, int(SemanticSource.UNKNOWN), dtype=np.uint8)
    semantic_ids = np.zeros(count, dtype=np.int32)
    confidence = np.zeros(count, dtype=np.float32)
    reliability = np.zeros(count, dtype=np.float32)

    if config.strategy == SurfaceSemanticStrategy.S0:
        bounded = np.where(valid, distances, np.inf)
        nearest = np.argmin(bounded, axis=1) if neighbors else np.zeros(count, dtype=np.int64)
        has_local = np.any(valid, axis=1)
        rows = np.flatnonzero(has_local)
        semantic_ids[rows] = ids[rows, nearest[rows]]
        confidence[rows] = np.minimum(1.0, supports[rows, nearest[rows]])
        reliability[rows] = confidence[rows]
        source[rows] = int(SemanticSource.LOCAL_SURFACE)
        return SurfaceSemanticResult(semantic_ids, confidence, reliability, source)

    kernel = np.exp(-0.5 * np.square(distances / config.distance_sigma_m))
    weights = np.where(valid, supports * kernel, 0.0).astype(np.float32, copy=False)
    total_weight = weights.sum(axis=1, dtype=np.float32)
    best_labels = np.zeros(count, dtype=np.int32)
    best_mass = np.zeros(count, dtype=np.float32)
    for column in range(neighbors):
        label = ids[:, column]
        candidate_mass = np.sum(
            np.where(ids == label[:, None], weights, 0.0), axis=1, dtype=np.float32
        )
        candidate_mass = np.where(label > 0, candidate_mass, 0.0)
        better = (candidate_mass > best_mass) | (
            (candidate_mass == best_mass)
            & (candidate_mass > 0.0)
            & ((best_labels == 0) | (label < best_labels))
        )
        best_mass[better] = candidate_mass[better]
        best_labels[better] = label[better]
    has_local = total_weight > 0.0
    normalized_confidence = np.divide(
        best_mass,
        total_weight,
        out=np.zeros(count, dtype=np.float32),
        where=has_local,
    )
    bounded_distance = np.where(valid, distances, np.inf)
    nearest_distance = (
        bounded_distance.min(axis=1)
        if neighbors
        else np.full(count, np.inf, dtype=np.float32)
    )
    support_fraction = np.minimum(
        1.0,
        np.sum(np.where(valid, supports, 0.0), axis=1, dtype=np.float32)
        / np.float32(config.minimum_local_support),
    )
    nearest_kernel = np.where(
        np.isfinite(nearest_distance),
        np.exp(-0.5 * np.square(nearest_distance / config.distance_sigma_m)),
        0.0,
    )
    reliability[:] = (support_fraction * nearest_kernel).astype(np.float32)

    if config.strategy == SurfaceSemanticStrategy.S1:
        semantic_ids[has_local] = best_labels[has_local]
        confidence[has_local] = normalized_confidence[has_local]
        source[has_local] = int(SemanticSource.LOCAL_SURFACE)
        return SurfaceSemanticResult(semantic_ids, confidence, reliability, source)

    local_score = normalized_confidence * reliability
    entity_valid = (
        (entity_ids > 0) & (entity_conf >= config.minimum_entity_confidence)
    )
    entity_score = np.where(
        entity_valid, entity_conf * np.float32(config.entity_weight_scale), 0.0
    )
    use_local = has_local & ((local_score >= entity_score) | ~entity_valid)
    use_entity = entity_valid & ~use_local
    semantic_ids[use_local] = best_labels[use_local]
    confidence[use_local] = local_score[use_local]
    source[use_local] = int(SemanticSource.LOCAL_SURFACE)
    semantic_ids[use_entity] = entity_ids[use_entity]
    confidence[use_entity] = entity_conf[use_entity]
    source[use_entity] = int(SemanticSource.OWNER_ENTITY)
    return SurfaceSemanticResult(semantic_ids, confidence, reliability, source)


def _vertices(value: object, name: str, *, allow_empty: bool) -> np.ndarray:
    try:
        vertices = np.ascontiguousarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be convertible to float32") from error
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3)")
    if not allow_empty and len(vertices) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{name} must contain finite values")
    return vertices


def transfer_surface_semantics(
    *,
    fine_vertices_xyz: object,
    reference_vertices_xyz: object,
    reference_semantic_ids: np.ndarray,
    reference_supports: np.ndarray,
    entity_semantic_ids: np.ndarray,
    entity_confidences: np.ndarray,
    configs: tuple[SurfaceSemanticConfig, ...],
    neighbor_count: int = 8,
    point_batch_size: int = 250_000,
) -> dict[SurfaceSemanticStrategy, SurfaceSemanticResult]:
    """Transfer bounded reference evidence to a fine surface in fixed-size chunks."""

    fine = _vertices(fine_vertices_xyz, "fine_vertices_xyz", allow_empty=True)
    reference = _vertices(
        reference_vertices_xyz, "reference_vertices_xyz", allow_empty=False
    )
    reference_ids = np.asarray(reference_semantic_ids, dtype=np.int32)
    reference_weights = np.asarray(reference_supports, dtype=np.float32)
    if reference_ids.shape != (len(reference),) or reference_weights.shape != (
        len(reference),
    ):
        raise ValueError("reference semantic IDs and supports must align with vertices")
    if np.any(reference_ids < 0) or not np.isfinite(reference_weights).all() or np.any(
        reference_weights < 0.0
    ):
        raise ValueError("reference IDs and supports must be nonnegative and finite")
    entity_ids = np.asarray(entity_semantic_ids, dtype=np.int32)
    entity_scores = np.asarray(entity_confidences, dtype=np.float32)
    if entity_ids.shape != (len(fine),) or entity_scores.shape != (len(fine),):
        raise ValueError("entity semantic arrays must align with the fine surface")
    if not isinstance(configs, tuple) or not configs or any(
        not isinstance(config, SurfaceSemanticConfig) for config in configs
    ):
        raise TypeError("configs must be a non-empty tuple of SurfaceSemanticConfig")
    strategies = tuple(config.strategy for config in configs)
    if len(set(strategies)) != len(strategies):
        raise ValueError("configs must contain unique semantic strategies")
    if type(neighbor_count) is not int or neighbor_count <= 0:
        raise ValueError("neighbor_count must be a positive integer")
    if type(point_batch_size) is not int or point_batch_size <= 0:
        raise ValueError("point_batch_size must be a positive integer")

    arrays: dict[SurfaceSemanticStrategy, dict[str, np.ndarray]] = {
        strategy: {
            "semantic_ids": np.zeros(len(fine), dtype=np.int32),
            "semantic_confidences": np.zeros(len(fine), dtype=np.float32),
            "support_reliabilities": np.zeros(len(fine), dtype=np.float32),
            "semantic_source_codes": np.zeros(len(fine), dtype=np.uint8),
        }
        for strategy in strategies
    }
    tree = cKDTree(reference)
    effective_neighbors = min(neighbor_count, len(reference))
    for start in range(0, len(fine), point_batch_size):
        stop = min(start + point_batch_size, len(fine))
        distances, indices = tree.query(
            fine[start:stop], k=effective_neighbors, workers=-1
        )
        distances = np.asarray(distances, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int64)
        if effective_neighbors == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        local_ids = reference_ids[indices]
        local_supports = reference_weights[indices]
        for config in configs:
            result = read_surface_semantics(
                local_semantic_ids=local_ids,
                local_supports=local_supports,
                local_distances_m=distances,
                entity_semantic_ids=entity_ids[start:stop],
                entity_confidences=entity_scores[start:stop],
                config=config,
            )
            destination = arrays[config.strategy]
            destination["semantic_ids"][start:stop] = result.semantic_ids
            destination["semantic_confidences"][start:stop] = (
                result.semantic_confidences
            )
            destination["support_reliabilities"][start:stop] = (
                result.support_reliabilities
            )
            destination["semantic_source_codes"][start:stop] = (
                result.semantic_source_codes
            )
    return {
        strategy: SurfaceSemanticResult(**payload)
        for strategy, payload in arrays.items()
    }


__all__ = [
    "SurfaceSemanticConfig",
    "SurfaceSemanticResult",
    "SurfaceSemanticStrategy",
    "read_surface_semantics",
    "transfer_surface_semantics",
]
