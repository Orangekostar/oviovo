from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from src.oviv2.temporal_config import ExecutionProfile, TemporalDynamicConfig


class TemporalStateComponent(str, Enum):
    IDENTITY_PROTOTYPE = "identity_prototype"
    EXISTENCE = "existence"
    CURRENT_POSE = "current_pose"
    DYNAMIC_STATE = "dynamic_state"
    GEOMETRY_EPOCH = "geometry_epoch"
    BACKGROUND_OWNERSHIP = "background_ownership"
    CURRENT_READOUT = "current_readout"
    CUMULATIVE_MAP = "cumulative_map"


class CausalEvidenceType(str, Enum):
    PRESENT_SUPPORT = "present_support"
    VISIBLE_FREE_SPACE = "visible_free_space"
    OCCLUDED = "occluded"
    OUT_OF_VIEW = "out_of_view"
    DEPTH_UNKNOWN = "depth_unknown"
    LOCAL_GEOMETRY = "local_geometry"
    QUALIFIED_IDENTITY = "qualified_identity"
    CURRENT_OBSERVATION_CENTER = "current_observation_center"
    FROZEN_STATIC_FUSION = "frozen_static_fusion"


class MotionEvidenceSource(str, Enum):
    NONE = "none"
    GEOMETRY = "geometry"
    IDENTITY = "identity"
    COMBINED = "combined"


_ADMISSIBLE_COMPONENTS = MappingProxyType(
    {
        CausalEvidenceType.PRESENT_SUPPORT: frozenset(
            {TemporalStateComponent.EXISTENCE}
        ),
        CausalEvidenceType.VISIBLE_FREE_SPACE: frozenset(
            {
                TemporalStateComponent.EXISTENCE,
                TemporalStateComponent.BACKGROUND_OWNERSHIP,
            }
        ),
        CausalEvidenceType.OCCLUDED: frozenset(),
        CausalEvidenceType.OUT_OF_VIEW: frozenset(),
        CausalEvidenceType.DEPTH_UNKNOWN: frozenset(),
        CausalEvidenceType.LOCAL_GEOMETRY: frozenset(
            {
                TemporalStateComponent.CURRENT_POSE,
                TemporalStateComponent.DYNAMIC_STATE,
                TemporalStateComponent.GEOMETRY_EPOCH,
                TemporalStateComponent.CURRENT_READOUT,
            }
        ),
        CausalEvidenceType.QUALIFIED_IDENTITY: frozenset(
            {
                TemporalStateComponent.IDENTITY_PROTOTYPE,
                TemporalStateComponent.DYNAMIC_STATE,
            }
        ),
        CausalEvidenceType.CURRENT_OBSERVATION_CENTER: frozenset(
            {
                TemporalStateComponent.CURRENT_POSE,
                TemporalStateComponent.CURRENT_READOUT,
            }
        ),
        CausalEvidenceType.FROZEN_STATIC_FUSION: frozenset(
            {TemporalStateComponent.CUMULATIVE_MAP}
        ),
    }
)


@dataclass(frozen=True)
class RoutedMotionEvidence:
    source: MotionEvidenceSource
    accepted: bool
    displacement_m: float
    confidence: float
    qualifies_as_motion: bool

    def __post_init__(self) -> None:
        if type(self.source) is not MotionEvidenceSource:
            raise TypeError("source must be a MotionEvidenceSource")
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be an exact bool")
        displacement = _finite(self.displacement_m, "displacement_m")
        if displacement < 0.0:
            raise ValueError("displacement_m must be nonnegative")
        confidence = _probability(self.confidence, "confidence")
        if type(self.qualifies_as_motion) is not bool:
            raise TypeError("qualifies_as_motion must be an exact bool")
        if self.accepted is (self.source is MotionEvidenceSource.NONE):
            raise ValueError("accepted must agree with the evidence source")
        if self.qualifies_as_motion and not self.accepted:
            raise ValueError("qualifying motion must be accepted")
        object.__setattr__(self, "displacement_m", displacement)
        object.__setattr__(self, "confidence", confidence)


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _probability(value: object, name: str) -> float:
    normalized = _finite(value, name)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return normalized


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _dynamic_config(config: object) -> TemporalDynamicConfig:
    if type(config) is not TemporalDynamicConfig:
        raise TypeError("config must be a TemporalDynamicConfig")
    _integer(
        config.minimum_consecutive_motion_frames,
        "config.minimum_consecutive_motion_frames",
        minimum=1,
    )
    displacement_floor = _finite(
        config.displacement_floor_m,
        "config.displacement_floor_m",
    )
    if displacement_floor < 0.0:
        raise ValueError("config.displacement_floor_m must be nonnegative")
    _probability(
        config.minimum_motion_confidence,
        "config.minimum_motion_confidence",
    )
    _integer(
        config.static_off_streak_frames,
        "config.static_off_streak_frames",
        minimum=1,
    )
    return config


def _appearance_similarity(value: object) -> float | None:
    if value is None:
        return None
    normalized = _finite(value, "appearance_similarity")
    if not -1.0 <= normalized <= 1.0:
        raise ValueError("appearance_similarity must be in [-1, 1]")
    return normalized


def _center(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise TypeError(f"{name} must contain three coordinates")
    if len(value) != 3:  # type: ignore[arg-type]
        raise ValueError(f"{name} must contain three coordinates")
    return tuple(_finite(item, name) for item in value)  # type: ignore[arg-type,return-value]


def admissible_components(
    evidence_type: CausalEvidenceType,
) -> frozenset[TemporalStateComponent]:
    if type(evidence_type) is not CausalEvidenceType:
        raise TypeError("evidence_type must be a CausalEvidenceType")
    return _ADMISSIBLE_COMPONENTS[evidence_type]


def route_active_motion_evidence(
    *,
    geometry_accepted: bool,
    geometry_confidence: float,
    geometry_displacement_m: float,
    identity_qualified: bool,
    appearance_similarity: float | None,
    identity_displacement_m: float,
    config: TemporalDynamicConfig,
) -> RoutedMotionEvidence:
    geometry_is_accepted = _exact_bool(
        geometry_accepted,
        "geometry_accepted",
    )
    normalized_geometry_confidence = _probability(
        geometry_confidence,
        "geometry_confidence",
    )
    geometry_displacement = _finite(
        geometry_displacement_m,
        "geometry_displacement_m",
    )
    if geometry_displacement < 0.0:
        raise ValueError("geometry_displacement_m must be nonnegative")
    identity_is_qualified = _exact_bool(
        identity_qualified,
        "identity_qualified",
    )
    similarity = _appearance_similarity(appearance_similarity)
    identity_displacement = _finite(
        identity_displacement_m,
        "identity_displacement_m",
    )
    if identity_displacement < 0.0:
        raise ValueError("identity_displacement_m must be nonnegative")
    normalized_config = _dynamic_config(config)

    geometry_admitted = (
        geometry_is_accepted and normalized_geometry_confidence > 0.0
    )
    identity_admitted = identity_is_qualified and similarity is not None
    identity_confidence = (
        float(np.clip(similarity, 0.0, 1.0))
        if identity_admitted
        else 0.0
    )

    if geometry_admitted and identity_admitted:
        source = MotionEvidenceSource.COMBINED
    elif geometry_admitted:
        source = MotionEvidenceSource.GEOMETRY
    elif identity_admitted:
        source = MotionEvidenceSource.IDENTITY
    else:
        source = MotionEvidenceSource.NONE
    accepted = source is not MotionEvidenceSource.NONE
    candidates: list[tuple[MotionEvidenceSource, float, float]] = []
    if geometry_admitted:
        candidates.append(
            (
                MotionEvidenceSource.GEOMETRY,
                geometry_displacement,
                normalized_geometry_confidence,
            )
        )
    if identity_admitted:
        candidates.append(
            (
                MotionEvidenceSource.IDENTITY,
                identity_displacement,
                identity_confidence,
            )
        )
    qualifying = [
        candidate
        for candidate in candidates
        if candidate[1] >= normalized_config.displacement_floor_m
        and candidate[2] >= normalized_config.minimum_motion_confidence
    ]
    moving = [
        candidate
        for candidate in candidates
        if candidate[1] >= normalized_config.displacement_floor_m
    ]
    selectable = qualifying or moving
    if selectable:
        _, displacement, confidence = max(
            selectable,
            key=lambda candidate: (
                candidate[2],
                candidate[0] is MotionEvidenceSource.GEOMETRY,
            ),
        )
    else:
        displacement = 0.0
        confidence = 0.0
    qualifies = bool(qualifying)
    return RoutedMotionEvidence(
        source=source,
        accepted=accepted,
        displacement_m=displacement,
        confidence=confidence,
        qualifies_as_motion=qualifies,
    )


def admits_identity_prototype_update(
    *,
    identity_qualified: bool,
    appearance_similarity: float | None,
) -> bool:
    qualified = _exact_bool(identity_qualified, "identity_qualified")
    similarity = _appearance_similarity(appearance_similarity)
    return qualified and similarity is not None


def select_temporal_readout_center(
    *,
    execution_profile: ExecutionProfile,
    cumulative_center: tuple[float, float, float],
    current_observation_center: tuple[float, float, float],
) -> tuple[float, float, float]:
    if type(execution_profile) is not ExecutionProfile:
        raise TypeError("execution_profile must be an ExecutionProfile")
    cumulative = _center(cumulative_center, "cumulative_center")
    current = _center(
        current_observation_center,
        "current_observation_center",
    )
    return current if execution_profile is ExecutionProfile.A4 else cumulative
