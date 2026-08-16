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
from src.oviv2.temporal_config import TemporalAssociationConfig, TemporalIdentityConfig
from src.oviv2.temporal_lifecycle import TemporalLifecycle


_INT64_MAX = 2**63 - 1


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    if result > _INT64_MAX:
        raise ValueError(f"{name} must fit signed int64")
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
class TemporalAssignmentDiagnostic:
    observation_id: int
    entity_id: int
    score: float
    target_lifecycle: TemporalLifecycle
    appearance_similarity: float | None
    feature_model_id: str | None
    feature_model_match: bool
    semantic_qualified: bool
    high_confidence_identity_match: bool

    def __post_init__(self) -> None:
        if type(self.observation_id) is not int or not 0 <= self.observation_id <= _INT64_MAX:
            raise TypeError("observation_id must be an exact non-negative integer")
        if type(self.entity_id) is not int or not 0 <= self.entity_id <= _INT64_MAX:
            raise TypeError("entity_id must be an exact non-negative integer")
        score = _finite(self.score, "score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must lie in [0, 1]")
        object.__setattr__(self, "score", score)
        if not isinstance(self.target_lifecycle, TemporalLifecycle):
            raise TypeError("target_lifecycle must be a TemporalLifecycle")
        if self.appearance_similarity is not None:
            similarity = _finite(self.appearance_similarity, "appearance_similarity")
            if not -1.0 <= similarity <= 1.0:
                raise ValueError("appearance_similarity must lie in [-1, 1]")
            object.__setattr__(self, "appearance_similarity", similarity)
        if self.feature_model_id is not None:
            if not isinstance(self.feature_model_id, str) or not self.feature_model_id.strip():
                raise ValueError("feature_model_id must be a non-empty string or None")
            object.__setattr__(self, "feature_model_id", self.feature_model_id.strip())
        for name in (
            "feature_model_match",
            "semantic_qualified",
            "high_confidence_identity_match",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.feature_model_match != (
            self.feature_model_id is not None and self.appearance_similarity is not None
        ):
            raise ValueError("feature model match must agree with appearance provenance")
        if self.high_confidence_identity_match and not (
            self.feature_model_match
            and self.appearance_similarity is not None
            and self.semantic_qualified
        ):
            raise ValueError("high confidence identity match requires qualified evidence")


@dataclass(frozen=True)
class TemporalAssociationResult:
    assignments: tuple[tuple[int, int], ...]
    unmatched_observation_ids: tuple[int, ...]
    unmatched_entity_ids: tuple[int, ...]
    reid_opportunity_count: int = 0
    reid_trigger_count: int = 0
    assignment_diagnostics: tuple[TemporalAssignmentDiagnostic, ...] = ()
    reid_opportunity_pairs: tuple[tuple[int, int], ...] = ()
    reid_trigger_pairs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if type(self.assignments) is not tuple:
            raise TypeError("assignments must be an exact tuple")
        for item in self.assignments:
            if (
                type(item) is not tuple
                or len(item) != 2
                or any(type(value) is not int or not 0 <= value <= _INT64_MAX for value in item)
            ):
                raise TypeError("assignments must contain exact non-negative integer pairs")
        if self.assignments != tuple(sorted(set(self.assignments))):
            raise ValueError("assignments must be sorted and unique")
        if len({item[0] for item in self.assignments}) != len(self.assignments) or len(
            {item[1] for item in self.assignments}
        ) != len(self.assignments):
            raise ValueError("assignments must be one-to-one")
        for name in ("unmatched_observation_ids", "unmatched_entity_ids"):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not int or not 0 <= value <= _INT64_MAX for value in values
            ):
                raise TypeError(f"{name} must be an exact tuple of non-negative integers")
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        object.__setattr__(
            self,
            "reid_opportunity_count",
            _integer(self.reid_opportunity_count, "reid_opportunity_count"),
        )
        object.__setattr__(
            self,
            "reid_trigger_count",
            _integer(self.reid_trigger_count, "reid_trigger_count"),
        )
        if self.reid_trigger_count > self.reid_opportunity_count:
            raise ValueError("reid_trigger_count cannot exceed reid_opportunity_count")
        for name, pairs, count in (
            ("reid_opportunity_pairs", self.reid_opportunity_pairs, self.reid_opportunity_count),
            ("reid_trigger_pairs", self.reid_trigger_pairs, self.reid_trigger_count),
        ):
            if (
                type(pairs) is not tuple
                or pairs != tuple(sorted(set(pairs)))
                or len(pairs) != count
                or any(
                    type(item) is not tuple
                    or len(item) != 2
                    or any(type(value) is not int or not 0 <= value <= _INT64_MAX for value in item)
                    for item in pairs
                )
            ):
                raise ValueError(f"{name} must uniquely identify every counted pair")
        if not set(self.reid_trigger_pairs) <= set(self.reid_opportunity_pairs):
            raise ValueError("re-ID trigger pairs must be opportunity pairs")
        if type(self.assignment_diagnostics) is not tuple or any(
            not isinstance(item, TemporalAssignmentDiagnostic)
            for item in self.assignment_diagnostics
        ):
            raise TypeError("assignment_diagnostics must be an exact tuple of diagnostics")
        diagnostic_pairs = tuple(
            (item.observation_id, item.entity_id) for item in self.assignment_diagnostics
        )
        if diagnostic_pairs != tuple(sorted(set(diagnostic_pairs))):
            raise ValueError("assignment_diagnostics must be sorted and unique")
        if diagnostic_pairs != self.assignments:
            raise ValueError("assignment_diagnostics must completely cover sorted assignments")


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


@dataclass(frozen=True)
class _IdentityEvidence:
    appearance_similarity: float | None
    feature_model_match: bool
    semantic_score: float | None
    semantic_qualified: bool


def _extract_identity_evidence(
    observation: _PreparedObservation,
    target: TemporalAssociationTarget,
    config: TemporalAssociationConfig,
    *,
    dormant: bool,
) -> _IdentityEvidence:
    model_match = bool(
        observation.image_feature is not None
        and target.image_prototype is not None
        and observation.value.feature_model_id == target.feature_model_id
        and observation.image_feature.shape == target.image_prototype.shape
    )
    cosine: float | None = None
    if model_match:
        assert observation.image_feature is not None
        assert target.image_prototype is not None
        cosine = float(
            np.clip(
                np.dot(observation.image_feature, target.image_prototype), -1.0, 1.0
            )
        )

    semantic: float | None = None
    semantic_qualified = True
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
        if conflict:
            visual = None if cosine is None else (cosine + 1.0) / 2.0
            visual_override = bool(
                visual is not None and visual >= config.conflict_override_visual
            )
            if dormant:
                semantic_qualified = visual_override
            else:
                current_distance = _distance(
                    observation.centroid, target.centroid_xyz
                )
                geometry = max(
                    0.0,
                    1.0
                    - current_distance / config.maximum_centroid_distance_m,
                )
                semantic_qualified = bool(
                    visual_override
                    and geometry >= config.conflict_override_geometry
                )
    return _IdentityEvidence(cosine, model_match, semantic, semantic_qualified)


def _score_identity_edge(
    observation: _PreparedObservation,
    target: TemporalAssociationTarget,
    config: TemporalAssociationConfig,
    evidence: _IdentityEvidence,
    *,
    gate_distance_m: float,
) -> CandidateScore | None:
    current_distance = _distance(observation.centroid, target.centroid_xyz)
    if current_distance > gate_distance_m or not evidence.semantic_qualified:
        return None
    geometry = max(
        0.0,
        1.0 - current_distance / config.maximum_centroid_distance_m,
    )
    size = float(
        np.mean(
            [
                min(observed, stored) / max(observed, stored)
                for observed, stored in zip(observation.extent, target.extent_xyz)
            ]
        )
    )
    visual = (
        None
        if evidence.appearance_similarity is None
        else float(np.clip((evidence.appearance_similarity + 1.0) / 2.0, 0.0, 1.0))
    )

    components = (
        (config.visual_weight, visual),
        (config.semantic_weight, evidence.semantic_score),
        (config.size_weight, size),
        (config.geometry_weight, geometry),
    )
    available = [
        (weight, value)
        for weight, value in components
        if weight > 0.0 and value is not None
    ]
    if not available:
        if any(weight > 0.0 for weight, _ in components):
            return None
        fallback_values = [
            value
            for value in (visual, evidence.semantic_score, size, geometry)
            if value is not None
        ]
        available = [(1.0, value) for value in fallback_values]
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
        geometry,
        visual,
        False,
        True,
    )


def _score_edge(
    observation: _PreparedObservation,
    target: TemporalAssociationTarget,
    config: TemporalAssociationConfig,
) -> CandidateScore | None:
    evidence = _extract_identity_evidence(
        observation, target, config, dormant=False
    )
    return _score_identity_edge(
        observation,
        target,
        config,
        evidence,
        gate_distance_m=config.maximum_centroid_distance_m,
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


def _validate_reid_config(config: object) -> TemporalIdentityConfig:
    if not isinstance(config, TemporalIdentityConfig):
        raise TypeError("dormant_reid must be a TemporalIdentityConfig or None")
    _integer(config.maximum_identities, "dormant_reid.maximum_identities", positive=True)
    _integer(config.maximum_dormant_frames, "dormant_reid.maximum_dormant_frames", positive=True)
    similarity = _finite(config.minimum_reid_similarity, "dormant_reid.minimum_reid_similarity")
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("dormant_reid.minimum_reid_similarity must lie in [0, 1]")
    distance = _finite(config.maximum_reid_distance_m, "dormant_reid.maximum_reid_distance_m")
    if distance <= 0.0:
        raise ValueError("dormant_reid.maximum_reid_distance_m must be positive")
    return config


def _identity_qualification(
    evidence: _IdentityEvidence,
    reid_config: TemporalIdentityConfig,
) -> bool:
    return bool(
        evidence.feature_model_match
        and evidence.appearance_similarity is not None
        and evidence.appearance_similarity >= reid_config.minimum_reid_similarity
        and evidence.semantic_qualified
    )


def _score_dormant_edge(
    observation: _PreparedObservation,
    target: TemporalAssociationTarget,
    config: TemporalAssociationConfig,
    reid_config: TemporalIdentityConfig,
    evidence: _IdentityEvidence,
) -> CandidateScore | None:
    return _score_identity_edge(
        observation,
        target,
        config,
        evidence,
        gate_distance_m=reid_config.maximum_reid_distance_m,
    )


def associate_temporal_observations(
    observations: tuple[FrameObservation, ...],
    targets: tuple[TemporalAssociationTarget, ...],
    config: TemporalAssociationConfig,
    *,
    dormant_reid: TemporalIdentityConfig | None = None,
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

    reid_config = None if dormant_reid is None else _validate_reid_config(dormant_reid)
    eligible_targets = tuple(
        item for item in sorted_targets
        if item.lifecycle in (TemporalLifecycle.ACTIVE, TemporalLifecycle.UNCERTAIN)
        or (item.lifecycle is TemporalLifecycle.DORMANT and reid_config is not None)
    )
    reid_opportunities: set[tuple[int, int]] = set()
    edges: dict[tuple[int, int], CandidateScore] = {}
    identity_evidence: dict[tuple[int, int], _IdentityEvidence] = {}
    high_confidence_pairs: set[tuple[int, int]] = set()
    for observation in prepared:
        for target in eligible_targets:
            pair = (observation.value.observation_id, target.entity_id)
            evidence = _extract_identity_evidence(
                observation,
                target,
                validated_config,
                dormant=target.lifecycle is TemporalLifecycle.DORMANT,
            )
            identity_evidence[pair] = evidence
            if reid_config is not None and _identity_qualification(
                evidence, reid_config
            ):
                high_confidence_pairs.add(pair)
            if target.lifecycle is TemporalLifecycle.DORMANT:
                assert reid_config is not None
                if pair not in high_confidence_pairs:
                    continue
                reid_opportunities.add(pair)
                candidate = _score_dormant_edge(
                    observation,
                    target,
                    validated_config,
                    reid_config,
                    evidence,
                )
            else:
                candidate = _score_identity_edge(
                    observation,
                    target,
                    validated_config,
                    evidence,
                    gate_distance_m=(
                        reid_config.maximum_reid_distance_m
                        if reid_config is not None
                        and pair in high_confidence_pairs
                        else validated_config.maximum_centroid_distance_m
                    ),
                )
            if candidate is not None:
                edges[pair] = candidate

    def precomputed_scorer(
        left: AssociationTarget,
        right: AssociationTarget,
        _config: AssociationConfig,
    ) -> CandidateScore | None:
        return edges.get((left.target_id, right.target_id))

    solved = solve_assignment(
        tuple(_placeholder(item.value.observation_id) for item in prepared),
        tuple(_placeholder(item.entity_id) for item in eligible_targets),
        _ASSIGNMENT_CONFIG,
        candidate_scorer=precomputed_scorer,
    ) if prepared and eligible_targets else ()
    assignments = [(item.left_id, item.right_id) for item in solved]
    target_by_id = {item.entity_id: item for item in eligible_targets}
    observation_by_id = {item.value.observation_id: item for item in prepared}
    diagnostics: list[TemporalAssignmentDiagnostic] = []
    for observation_id, entity_id in assignments:
        pair = (observation_id, entity_id)
        evidence = identity_evidence[pair]
        observation = observation_by_id[observation_id]
        target = target_by_id[entity_id]
        diagnostics.append(
            TemporalAssignmentDiagnostic(
                observation_id=observation_id,
                entity_id=entity_id,
                score=edges[pair].score,
                target_lifecycle=target.lifecycle,
                appearance_similarity=evidence.appearance_similarity,
                feature_model_id=(
                    observation.value.feature_model_id
                    if evidence.feature_model_match
                    else None
                ),
                feature_model_match=evidence.feature_model_match,
                semantic_qualified=evidence.semantic_qualified,
                high_confidence_identity_match=pair in high_confidence_pairs,
            )
        )

    matched_observations = {observation_id for observation_id, _ in assignments}
    matched_entities = {entity_id for _, entity_id in assignments}
    return TemporalAssociationResult(
        assignments=tuple(sorted(assignments)),
        unmatched_observation_ids=tuple(
            sorted(set(observation_ids) - matched_observations)
        ),
        unmatched_entity_ids=tuple(sorted(set(entity_ids) - matched_entities)),
        reid_opportunity_count=len(reid_opportunities),
        reid_trigger_count=sum(
            (observation_id, entity_id) in reid_opportunities
            for observation_id, entity_id in assignments
        ),
        assignment_diagnostics=tuple(diagnostics),
        reid_opportunity_pairs=tuple(sorted(reid_opportunities)),
        reid_trigger_pairs=tuple(
            sorted(set(assignments) & reid_opportunities)
        ),
    )
