from __future__ import annotations

import pytest

from src.oviv2.temporal_config import ExecutionProfile, TemporalDynamicConfig
from src.oviv2.temporal_evidence_router import (
    CausalEvidenceType,
    MotionEvidenceSource,
    TemporalStateComponent,
    admissible_components,
    admits_identity_prototype_update,
    route_active_motion_evidence,
    select_temporal_readout_center,
)


EXPECTED_COMPONENTS = {
    CausalEvidenceType.PRESENT_SUPPORT: {
        TemporalStateComponent.EXISTENCE,
    },
    CausalEvidenceType.VISIBLE_FREE_SPACE: {
        TemporalStateComponent.EXISTENCE,
        TemporalStateComponent.BACKGROUND_OWNERSHIP,
    },
    CausalEvidenceType.OCCLUDED: set(),
    CausalEvidenceType.OUT_OF_VIEW: set(),
    CausalEvidenceType.DEPTH_UNKNOWN: set(),
    CausalEvidenceType.LOCAL_GEOMETRY: {
        TemporalStateComponent.CURRENT_POSE,
        TemporalStateComponent.DYNAMIC_STATE,
        TemporalStateComponent.GEOMETRY_EPOCH,
        TemporalStateComponent.CURRENT_READOUT,
    },
    CausalEvidenceType.QUALIFIED_IDENTITY: {
        TemporalStateComponent.IDENTITY_PROTOTYPE,
        TemporalStateComponent.DYNAMIC_STATE,
    },
    CausalEvidenceType.CURRENT_OBSERVATION_CENTER: {
        TemporalStateComponent.CURRENT_POSE,
        TemporalStateComponent.CURRENT_READOUT,
    },
    CausalEvidenceType.FROZEN_STATIC_FUSION: {
        TemporalStateComponent.CUMULATIVE_MAP,
    },
}


def _dynamic_config() -> TemporalDynamicConfig:
    return TemporalDynamicConfig(
        minimum_consecutive_motion_frames=2,
        displacement_floor_m=0.15,
        minimum_motion_confidence=0.7,
        static_off_streak_frames=2,
    )


@pytest.mark.parametrize("kind", tuple(CausalEvidenceType))
def test_admissibility_matrix_is_closed(kind: CausalEvidenceType) -> None:
    assert admissible_components(kind) == frozenset(EXPECTED_COMPONENTS[kind])


def test_admissibility_rejects_non_evidence_values() -> None:
    with pytest.raises(TypeError, match="CausalEvidenceType"):
        admissible_components("occluded")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("geometry", "identity", "similarity", "source", "confidence"),
    [
        (False, False, None, MotionEvidenceSource.NONE, 0.0),
        (True, False, None, MotionEvidenceSource.GEOMETRY, 0.8),
        (False, True, 0.9, MotionEvidenceSource.IDENTITY, 0.9),
        (True, True, 0.9, MotionEvidenceSource.COMBINED, 0.9),
    ],
)
def test_active_motion_routes_only_admissible_sources(
    geometry: bool,
    identity: bool,
    similarity: float | None,
    source: MotionEvidenceSource,
    confidence: float,
) -> None:
    result = route_active_motion_evidence(
        geometry_accepted=geometry,
        geometry_confidence=0.8 if geometry else 0.0,
        identity_qualified=identity,
        appearance_similarity=similarity,
        displacement_m=0.2,
        config=_dynamic_config(),
    )

    assert result.source is source
    assert result.confidence == pytest.approx(confidence)
    assert result.accepted is (source is not MotionEvidenceSource.NONE)
    assert result.qualifies_as_motion is (
        source is not MotionEvidenceSource.NONE
    )


def test_active_motion_retains_displacement_and_confidence_thresholds() -> None:
    low_displacement = route_active_motion_evidence(
        geometry_accepted=False,
        geometry_confidence=0.0,
        identity_qualified=True,
        appearance_similarity=0.9,
        displacement_m=0.149,
        config=_dynamic_config(),
    )
    low_confidence = route_active_motion_evidence(
        geometry_accepted=False,
        geometry_confidence=0.0,
        identity_qualified=True,
        appearance_similarity=0.69,
        displacement_m=0.2,
        config=_dynamic_config(),
    )

    assert low_displacement.accepted is True
    assert low_displacement.qualifies_as_motion is False
    assert low_confidence.accepted is True
    assert low_confidence.qualifies_as_motion is False


def test_missing_identity_similarity_fails_closed() -> None:
    result = route_active_motion_evidence(
        geometry_accepted=False,
        geometry_confidence=0.0,
        identity_qualified=True,
        appearance_similarity=None,
        displacement_m=0.2,
        config=_dynamic_config(),
    )

    assert result.source is MotionEvidenceSource.NONE
    assert result.accepted is False
    assert result.confidence == 0.0
    assert result.qualifies_as_motion is False


@pytest.mark.parametrize(
    "changes",
    [
        {"geometry_accepted": 1},
        {"geometry_confidence": float("nan")},
        {"geometry_confidence": 1.01},
        {"identity_qualified": 1},
        {"appearance_similarity": True},
        {"appearance_similarity": float("inf")},
        {"appearance_similarity": 1.01},
        {"displacement_m": -0.01},
        {"displacement_m": float("inf")},
        {"config": object()},
    ],
)
def test_active_motion_rejects_malformed_inputs(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "geometry_accepted": True,
        "geometry_confidence": 0.8,
        "identity_qualified": True,
        "appearance_similarity": 0.9,
        "displacement_m": 0.2,
        "config": _dynamic_config(),
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        route_active_motion_evidence(**values)  # type: ignore[arg-type]


def test_identity_prototype_requires_qualified_finite_appearance() -> None:
    assert admits_identity_prototype_update(
        identity_qualified=True,
        appearance_similarity=0.8,
    )
    assert not admits_identity_prototype_update(
        identity_qualified=False,
        appearance_similarity=0.8,
    )
    assert not admits_identity_prototype_update(
        identity_qualified=False,
        appearance_similarity=None,
    )
    assert not admits_identity_prototype_update(
        identity_qualified=True,
        appearance_similarity=None,
    )


@pytest.mark.parametrize("profile", (ExecutionProfile.A2, ExecutionProfile.A3))
def test_legacy_profiles_select_cumulative_center(
    profile: ExecutionProfile,
) -> None:
    assert select_temporal_readout_center(
        execution_profile=profile,
        cumulative_center=(1.0, 2.0, 3.0),
        current_observation_center=(4.0, 5.0, 6.0),
    ) == (1.0, 2.0, 3.0)


def test_a4_selects_current_observation_center() -> None:
    assert select_temporal_readout_center(
        execution_profile=ExecutionProfile.A4,
        cumulative_center=(1.0, 2.0, 3.0),
        current_observation_center=(4.0, 5.0, 6.0),
    ) == (4.0, 5.0, 6.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"execution_profile": "a4"},
        {"cumulative_center": (1.0, 2.0)},
        {"cumulative_center": (1.0, float("nan"), 3.0)},
        {"current_observation_center": None},
        {"current_observation_center": (4.0, 5.0, float("inf"))},
    ],
)
def test_readout_center_rejects_malformed_inputs(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "execution_profile": ExecutionProfile.A4,
        "cumulative_center": (1.0, 2.0, 3.0),
        "current_observation_center": (4.0, 5.0, 6.0),
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        select_temporal_readout_center(**values)  # type: ignore[arg-type]
