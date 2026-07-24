from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from src.oviv2.association import (
    AssociationConfig,
    AssociationTarget,
    CandidateScore,
    solve_assignment,
)
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.temporal_config import TemporalAssociationConfig
from src.oviv2.temporal_lifecycle import TemporalLifecycle


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _point3(value: object, name: str) -> tuple[float, float, float]:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain three finite values")
    return tuple(float(item) for item in point)  # type: ignore[return-value]


def _positive_extent(value: object, name: str) -> tuple[float, float, float]:
    extent = _point3(value, name)
    if any(item <= 0.0 for item in extent):
        raise ValueError(f"{name} must contain positive values")
    return extent


def _normalized_immutable_vector(value: object, name: str) -> np.ndarray:
    try:
        vector = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-zero vector") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite non-zero vector")
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        raise ValueError(f"{name} must be a finite non-zero vector")
    normalized = np.ascontiguousarray(vector / scale, dtype=np.float64)
    normalized /= np.linalg.norm(normalized)
    return np.frombuffer(normalized.tobytes(), dtype=normalized.dtype)


@dataclass(frozen=True)
class TemporalAssociationTarget:
    entity_id: int
    lifecycle: TemporalLifecycle
    centroid_xyz: tuple[float, float, float]
    extent_xyz: tuple[float, float, float]
    image_prototype: np.ndarray | None
    semantic_probabilities: tuple[tuple[int, float], ...]
    predicted_centroid_xyz: tuple[float, float, float]
    feature_model_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _integer(self.entity_id, "entity_id"))
        if not isinstance(self.lifecycle, TemporalLifecycle):
            raise TypeError("lifecycle must be a TemporalLifecycle")
        object.__setattr__(
            self, "centroid_xyz", _point3(self.centroid_xyz, "centroid_xyz")
        )
        object.__setattr__(
            self,
            "predicted_centroid_xyz",
            _point3(self.predicted_centroid_xyz, "predicted_centroid_xyz"),
        )
        object.__setattr__(
            self, "extent_xyz", _positive_extent(self.extent_xyz, "extent_xyz")
        )

        if self.image_prototype is not None:
            object.__setattr__(
                self,
                "image_prototype",
                _normalized_immutable_vector(self.image_prototype, "image_prototype"),
            )
        if self.feature_model_id is not None:
            if not isinstance(self.feature_model_id, str) or not self.feature_model_id.strip():
                raise ValueError("feature_model_id must be a non-empty string or None")
            object.__setattr__(self, "feature_model_id", self.feature_model_id.strip())
        if self.image_prototype is None and self.feature_model_id is not None:
            raise ValueError("feature_model_id requires image_prototype")

        if not isinstance(self.semantic_probabilities, tuple):
            raise TypeError("semantic_probabilities must be a tuple")
        normalized_probabilities: list[tuple[int, float]] = []
        previous_class_id = 0
        for item in self.semantic_probabilities:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("semantic probabilities must be (class_id, probability) tuples")
            class_id = _integer(item[0], "semantic class_id", positive=True)
            if class_id <= previous_class_id:
                raise ValueError("semantic probabilities must have unique sorted class IDs")
            probability = _finite(item[1], "semantic probability")
            if not 0.0 <= probability <= 1.0:
                raise ValueError("semantic probability must lie in [0, 1]")
            normalized_probabilities.append((class_id, probability))
            previous_class_id = class_id
        if normalized_probabilities and not math.isclose(
            math.fsum(probability for _, probability in normalized_probabilities),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("semantic probabilities must sum to 1")
        object.__setattr__(
            self, "semantic_probabilities", tuple(normalized_probabilities)
        )


@dataclass(frozen=True)
class TemporalAssociationResult:
    assignments: tuple[tuple[int, int], ...]
    unmatched_observation_ids: tuple[int, ...]
    unmatched_entity_ids: tuple[int, ...]


def _validate_config(config: object) -> TemporalAssociationConfig:
    if not isinstance(config, TemporalAssociationConfig):
        raise TypeError("config must be a TemporalAssociationConfig")
    weights = []
    for name in (
        "visual_weight",
        "semantic_weight",
        "size_weight",
        "motion_weight",
        "geometry_weight",
    ):
        value = _finite(getattr(config, name), f"config.{name}")
        if value < 0.0:
            raise ValueError(f"config.{name} must be non-negative")
        weights.append(value)
    if not any(weight > 0.0 for weight in weights):
        raise ValueError("at least one association weight must be positive")
    for name in (
        "minimum_score",
        "semantic_conflict_probability",
        "conflict_override_visual",
        "conflict_override_geometry",
    ):
        value = _finite(getattr(config, name), f"config.{name}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"config.{name} must lie in [0, 1]")
    maximum_distance = _finite(
        config.maximum_centroid_distance_m,
        "config.maximum_centroid_distance_m",
    )
    if maximum_distance <= 0.0:
        raise ValueError("config.maximum_centroid_distance_m must be positive")
    return config


@dataclass(frozen=True)
class _PreparedObservation:
    value: FrameObservation
    centroid: tuple[float, float, float]
    extent: tuple[float, float, float]
    image_feature: np.ndarray | None


def _prepare_observation(value: FrameObservation) -> _PreparedObservation:
    _integer(value.observation_id, "observation_id")
    if value.kind is not ObservationKind.OBJECT:
        raise ValueError("temporal association accepts only ObservationKind.OBJECT")
    _integer(value.semantic_id, "semantic_id")
    confidence = _finite(value.confidence, "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must lie in [0, 1]")
    centroid = _point3(value.centroid_xyz, "observation centroid_xyz")
    bounds_min = np.asarray(_point3(value.bounds_min_xyz, "observation bounds_min_xyz"))
    bounds_max = np.asarray(_point3(value.bounds_max_xyz, "observation bounds_max_xyz"))
    extent = _positive_extent(bounds_max - bounds_min, "observation extent")
    image_feature = (
        None
        if value.image_feature is None
        else _normalized_immutable_vector(value.image_feature, "observation image_feature")
    )
    return _PreparedObservation(value, centroid, extent, image_feature)


def _distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return float(math.dist(left, right))


def _score_edge(
    observation: _PreparedObservation,
    target: TemporalAssociationTarget,
    config: TemporalAssociationConfig,
) -> CandidateScore | None:
    maximum_distance = config.maximum_centroid_distance_m
    predicted_distance = _distance(observation.centroid, target.predicted_centroid_xyz)
    if predicted_distance > maximum_distance:
        return None
    motion = max(0.0, 1.0 - predicted_distance / maximum_distance)
    current_distance = _distance(observation.centroid, target.centroid_xyz)
    geometry = max(0.0, 1.0 - current_distance / maximum_distance)
    size = float(
        np.mean(
            [
                min(observed, stored) / max(observed, stored)
                for observed, stored in zip(observation.extent, target.extent_xyz)
            ]
        )
    )

    visual: float | None = None
    if (
        observation.image_feature is not None
        and target.image_prototype is not None
        and observation.value.feature_model_id == target.feature_model_id
        and observation.image_feature.shape == target.image_prototype.shape
    ):
        cosine = float(
            np.clip(
                np.dot(observation.image_feature, target.image_prototype), -1.0, 1.0
            )
        )
        visual = float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))

    semantic: float | None = None
    semantic_probabilities = dict(target.semantic_probabilities)
    if semantic_probabilities and observation.value.semantic_id > 0:
        semantic = semantic_probabilities.get(observation.value.semantic_id, 0.0)
        top_class_id, top_probability = max(
            target.semantic_probabilities, key=lambda item: item[1]
        )
        conflict = (
            observation.value.semantic_id != top_class_id
            and observation.value.confidence >= config.semantic_conflict_probability
            and top_probability >= config.semantic_conflict_probability
        )
        if conflict and not (
            visual is not None
            and visual >= config.conflict_override_visual
            and geometry >= config.conflict_override_geometry
        ):
            return None

    components = (
        (config.visual_weight, visual),
        (config.semantic_weight, semantic),
        (config.size_weight, size),
        (config.motion_weight, motion),
        (config.geometry_weight, geometry),
    )
    available = [
        (weight, value)
        for weight, value in components
        if weight > 0.0 and value is not None
    ]
    if not available:
        return None
    scale = max(weight for weight, _ in available)
    scaled = [(weight / scale, value) for weight, value in available]
    denominator = math.fsum(weight for weight, _ in scaled)
    score = math.fsum(weight * value for weight, value in scaled) / denominator
    score = float(np.clip(score, 0.0, 1.0))
    if score < config.minimum_score:
        return None
    return CandidateScore(
        observation.value.observation_id,
        target.entity_id,
        score,
        size,
        motion,
        visual,
        False,
        True,
    )


def _placeholder(target_id: int) -> AssociationTarget:
    return AssociationTarget(
        target_id=target_id,
        voxel_keys=frozenset({(0, 0, 0)}),
        centroid_xyz=(0.0, 0.0, 0.0),
        bounds_min_xyz=(0.0, 0.0, 0.0),
        bounds_max_xyz=(1.0, 1.0, 1.0),
    )


_ASSIGNMENT_CONFIG = AssociationConfig()


def _assign_stage(
    observations: tuple[_PreparedObservation, ...],
    targets: tuple[TemporalAssociationTarget, ...],
    config: TemporalAssociationConfig,
) -> tuple[tuple[int, int], ...]:
    if not observations or not targets:
        return ()
    edges = {
        (observation.value.observation_id, target.entity_id): candidate
        for observation in observations
        for target in targets
        if (candidate := _score_edge(observation, target, config)) is not None
    }

    def precomputed_scorer(
        left: AssociationTarget,
        right: AssociationTarget,
        _config: AssociationConfig,
    ) -> CandidateScore | None:
        return edges.get((left.target_id, right.target_id))

    assignments = solve_assignment(
        tuple(_placeholder(item.value.observation_id) for item in observations),
        tuple(_placeholder(item.entity_id) for item in targets),
        _ASSIGNMENT_CONFIG,
        candidate_scorer=precomputed_scorer,
    )
    return tuple((item.left_id, item.right_id) for item in assignments)


def associate_temporal_observations(
    observations: tuple[FrameObservation, ...],
    targets: tuple[TemporalAssociationTarget, ...],
    config: TemporalAssociationConfig,
) -> TemporalAssociationResult:
    if type(observations) is not tuple or type(targets) is not tuple:
        raise TypeError("observations and targets must be exact tuples")
    if any(not isinstance(value, FrameObservation) for value in observations):
        raise TypeError("observations must contain FrameObservation values")
    if any(not isinstance(value, TemporalAssociationTarget) for value in targets):
        raise TypeError("targets must contain TemporalAssociationTarget values")
    validated_config = _validate_config(config)

    prepared = tuple(_prepare_observation(value) for value in observations)
    observation_ids = [item.value.observation_id for item in prepared]
    entity_ids = [item.entity_id for item in targets]
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("observation IDs must be unique")
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("entity IDs must be unique")
    prepared = tuple(sorted(prepared, key=lambda item: item.value.observation_id))
    sorted_targets = tuple(sorted(targets, key=lambda item: item.entity_id))

    primary_targets = tuple(
        item
        for item in sorted_targets
        if item.lifecycle in (TemporalLifecycle.ACTIVE, TemporalLifecycle.UNCERTAIN)
    )
    assignments = list(_assign_stage(prepared, primary_targets, validated_config))
    matched_observations = {observation_id for observation_id, _ in assignments}
    remaining_observations = tuple(
        item
        for item in prepared
        if item.value.observation_id not in matched_observations
    )
    dormant_targets = tuple(
        item for item in sorted_targets if item.lifecycle is TemporalLifecycle.DORMANT
    )
    assignments.extend(
        _assign_stage(remaining_observations, dormant_targets, validated_config)
    )

    matched_observations = {observation_id for observation_id, _ in assignments}
    matched_entities = {entity_id for _, entity_id in assignments}
    return TemporalAssociationResult(
        assignments=tuple(sorted(assignments)),
        unmatched_observation_ids=tuple(
            sorted(set(observation_ids) - matched_observations)
        ),
        unmatched_entity_ids=tuple(sorted(set(entity_ids) - matched_entities)),
    )
