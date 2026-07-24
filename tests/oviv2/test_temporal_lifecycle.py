from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math

import pytest

from src.oviv2.temporal_config import TemporalLifecycleConfig
from src.oviv2.temporal_lifecycle import (
    TemporalEvidence,
    TemporalEvidenceKind,
    TemporalLifecycle,
    TemporalLifecycleState,
    advance_lifecycle,
)


@pytest.fixture
def config() -> TemporalLifecycleConfig:
    return TemporalLifecycleConfig(
        initial_log_odds=0.0,
        present_log_likelihood=4.0,
        absent_log_likelihood=-2.0,
        log_odds_limit=8.0,
        decay_half_life_seconds=10.0,
        active_on_probability=0.75,
        dormant_off_probability=0.25,
        minimum_absent_streak=3,
        minimum_distinct_view_bins=2,
        visibility_depth_tolerance_m=0.1,
        minimum_visible_pixel_count=10,
        minimum_visible_fraction=0.05,
        view_bin_azimuth_count=4,
        view_bin_elevation_count=2,
    )


def state(
    *,
    lifecycle: TemporalLifecycle = TemporalLifecycle.ACTIVE,
    log_odds: float = 2.0,
    frame_id: int = 10,
    timestamp: float = 10.0,
    absent_streak: int = 0,
    bins: tuple[int, ...] = (),
) -> TemporalLifecycleState:
    return TemporalLifecycleState(
        entity_id=7,
        lifecycle=lifecycle,
        existence_log_odds=log_odds,
        last_frame_id=frame_id,
        last_timestamp=timestamp,
        absent_streak=absent_streak,
        absence_view_bins=bins,
    )


def evidence(
    kind: TemporalEvidenceKind,
    *,
    strength: float = 1.0,
    frame_id: int = 11,
    timestamp: float = 11.0,
    view_bin: int | None = None,
) -> TemporalEvidence:
    return TemporalEvidence(kind, strength, frame_id, timestamp, view_bin)


def probability(log_odds: float) -> float:
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    exp_value = math.exp(log_odds)
    return exp_value / (1.0 + exp_value)


@pytest.mark.parametrize(
    "kind",
    [
        TemporalEvidenceKind.OCCLUDED,
        TemporalEvidenceKind.OUT_OF_VIEW,
        TemporalEvidenceKind.DEPTH_UNKNOWN,
    ],
)
def test_neutral_evidence_preserves_existence_and_active_lifecycle(
    config: TemporalLifecycleConfig, kind: TemporalEvidenceKind
) -> None:
    initial = state(absent_streak=2, bins=(1, 3))

    result = advance_lifecycle(initial, evidence(kind), config)

    assert result.existence_log_odds == initial.existence_log_odds
    assert probability(result.existence_log_odds) == probability(
        initial.existence_log_odds
    )
    assert result.lifecycle is TemporalLifecycle.ACTIVE
    assert result.absent_streak == 0
    assert result.absence_view_bins == ()
    assert (result.last_frame_id, result.last_timestamp) == (11, 11.0)


def test_one_visible_absence_cannot_retire_entity(
    config: TemporalLifecycleConfig,
) -> None:
    initial = state(log_odds=-0.5, lifecycle=TemporalLifecycle.ACTIVE)

    result = advance_lifecycle(
        initial,
        evidence(TemporalEvidenceKind.VISIBLE_ABSENT, view_bin=2),
        config,
    )

    assert result.lifecycle is TemporalLifecycle.UNCERTAIN
    assert result.absent_streak == 1
    assert result.absence_view_bins == (2,)


def test_repeated_view_bin_has_smaller_negative_update_than_new_bin(
    config: TemporalLifecycleConfig,
) -> None:
    initial = state(log_odds=2.0, absent_streak=1, bins=(1,))
    repeated = advance_lifecycle(
        initial,
        evidence(TemporalEvidenceKind.VISIBLE_ABSENT, view_bin=1),
        config,
    )
    novel = advance_lifecycle(
        initial,
        evidence(TemporalEvidenceKind.VISIBLE_ABSENT, view_bin=2),
        config,
    )
    decayed_prior = initial.existence_log_odds * 0.5 ** (1.0 / 10.0)

    assert abs(repeated.existence_log_odds - decayed_prior) < abs(
        novel.existence_log_odds - decayed_prior
    )
    assert repeated.absence_view_bins == (1,)
    assert novel.absence_view_bins == (1, 2)


def test_required_consecutive_absences_and_distinct_bins_enter_dormant(
    config: TemporalLifecycleConfig,
) -> None:
    current = state(log_odds=1.0)
    for frame_id, view_bin in ((11, 0), (12, 0), (13, 1)):
        current = advance_lifecycle(
            current,
            evidence(
                TemporalEvidenceKind.VISIBLE_ABSENT,
                frame_id=frame_id,
                timestamp=float(frame_id),
                view_bin=view_bin,
            ),
            config,
        )

    assert current.lifecycle is TemporalLifecycle.DORMANT
    assert current.absent_streak == 3
    assert current.absence_view_bins == (0, 1)
    assert probability(current.existence_log_odds) <= config.dormant_off_probability


def test_neutral_evidence_prevents_old_view_bins_from_retiring_a_later_streak(
    config: TemporalLifecycleConfig,
) -> None:
    current = advance_lifecycle(
        state(),
        evidence(TemporalEvidenceKind.VISIBLE_ABSENT, view_bin=0),
        config,
    )
    current = advance_lifecycle(
        current,
        evidence(
            TemporalEvidenceKind.OCCLUDED,
            frame_id=12,
            timestamp=12.0,
        ),
        config,
    )
    for frame_id in (13, 14, 15):
        current = advance_lifecycle(
            current,
            evidence(
                TemporalEvidenceKind.VISIBLE_ABSENT,
                frame_id=frame_id,
                timestamp=float(frame_id),
                view_bin=1,
            ),
            config,
        )

    assert current.lifecycle is TemporalLifecycle.UNCERTAIN
    assert current.absent_streak == 3
    assert current.absence_view_bins == (1,)


def test_strong_present_reactivates_same_dormant_entity(
    config: TemporalLifecycleConfig,
) -> None:
    initial = state(
        lifecycle=TemporalLifecycle.DORMANT,
        log_odds=-2.0,
        absent_streak=5,
        bins=(0, 1, 3),
    )

    result = advance_lifecycle(
        initial, evidence(TemporalEvidenceKind.PRESENT), config
    )

    assert result.entity_id == initial.entity_id
    assert result.lifecycle is TemporalLifecycle.ACTIVE
    assert result.absent_streak == 0
    assert result.absence_view_bins == ()


@pytest.mark.parametrize(
    ("kind", "view_bin"),
    [
        (TemporalEvidenceKind.VISIBLE_ABSENT, 0),
        (TemporalEvidenceKind.OCCLUDED, None),
        (TemporalEvidenceKind.OUT_OF_VIEW, None),
        (TemporalEvidenceKind.DEPTH_UNKNOWN, None),
    ],
)
def test_non_present_evidence_cannot_reactivate_dormant_entity(
    config: TemporalLifecycleConfig,
    kind: TemporalEvidenceKind,
    view_bin: int | None,
) -> None:
    permissive_config = replace(
        config,
        active_on_probability=0.4,
        dormant_off_probability=0.2,
    )
    dormant = state(lifecycle=TemporalLifecycle.DORMANT, log_odds=-4.0)

    result = advance_lifecycle(
        dormant,
        evidence(
            kind,
            strength=0.0,
            timestamp=1010.0,
            view_bin=view_bin,
        ),
        permissive_config,
    )

    assert result.lifecycle is TemporalLifecycle.DORMANT


def test_hysteresis_does_not_oscillate_between_active_and_dormant(
    config: TemporalLifecycleConfig,
) -> None:
    between_thresholds = 0.0
    active = state(lifecycle=TemporalLifecycle.ACTIVE, log_odds=between_thresholds)
    active_result = advance_lifecycle(
        active,
        evidence(TemporalEvidenceKind.VISIBLE_ABSENT, strength=0.0, view_bin=1),
        config,
    )
    dormant = state(lifecycle=TemporalLifecycle.DORMANT, log_odds=between_thresholds)
    dormant_result = advance_lifecycle(
        dormant,
        evidence(TemporalEvidenceKind.PRESENT, strength=0.0),
        config,
    )

    assert active_result.lifecycle is TemporalLifecycle.UNCERTAIN
    assert dormant_result.lifecycle is TemporalLifecycle.DORMANT


@pytest.mark.parametrize(
    "bad_evidence",
    [
        evidence(TemporalEvidenceKind.PRESENT, frame_id=10),
        evidence(TemporalEvidenceKind.PRESENT, frame_id=9),
        evidence(TemporalEvidenceKind.PRESENT, timestamp=10.0),
        evidence(TemporalEvidenceKind.PRESENT, timestamp=9.0),
        evidence(TemporalEvidenceKind.PRESENT, strength=float("nan")),
        evidence(TemporalEvidenceKind.PRESENT, strength=float("inf")),
        evidence(TemporalEvidenceKind.PRESENT, strength=-0.01),
        evidence(TemporalEvidenceKind.PRESENT, strength=1.01),
        evidence(TemporalEvidenceKind.PRESENT, timestamp=float("nan")),
        evidence(TemporalEvidenceKind.VISIBLE_ABSENT),
        evidence(TemporalEvidenceKind.VISIBLE_ABSENT, view_bin=-1),
        evidence(TemporalEvidenceKind.VISIBLE_ABSENT, view_bin=8),
        evidence(TemporalEvidenceKind.VISIBLE_ABSENT, view_bin=True),
        evidence(TemporalEvidenceKind.PRESENT, view_bin=1),
    ],
)
def test_invalid_evidence_fails_without_mutating_state(
    config: TemporalLifecycleConfig, bad_evidence: TemporalEvidence
) -> None:
    initial = state(absent_streak=1, bins=(2,))

    with pytest.raises((TypeError, ValueError)):
        advance_lifecycle(initial, bad_evidence, config)

    assert initial == state(absent_streak=1, bins=(2,))


@pytest.mark.parametrize(
    "bad_state",
    [
        state(bins=(2, 1)),
        state(bins=(1, 1)),
        state(bins=(-1,)),
        state(bins=(8,)),
        state(absent_streak=0, bins=(1,)),
        state(absent_streak=1, bins=(1, 2)),
        state(absent_streak=1, bins=()),
        state(absent_streak=2, bins=()),
        state(absent_streak=-1),
        replace(state(), entity_id=-1),
        state(frame_id=-1),
        state(timestamp=float("inf")),
        state(log_odds=float("nan")),
        state(log_odds=8.1),
    ],
)
def test_invalid_state_is_rejected(
    config: TemporalLifecycleConfig, bad_state: TemporalLifecycleState
) -> None:
    with pytest.raises((TypeError, ValueError)):
        advance_lifecycle(
            bad_state, evidence(TemporalEvidenceKind.PRESENT), config
        )


@pytest.mark.parametrize("argument", [None, object()])
def test_argument_types_are_checked(
    config: TemporalLifecycleConfig, argument: object
) -> None:
    initial = state()
    observation = evidence(TemporalEvidenceKind.PRESENT)

    with pytest.raises(TypeError):
        advance_lifecycle(argument, observation, config)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        advance_lifecycle(initial, argument, config)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        advance_lifecycle(initial, observation, argument)  # type: ignore[arg-type]


def test_non_finite_config_is_rejected_before_transition(
    config: TemporalLifecycleConfig,
) -> None:
    initial = state()
    bad_config = replace(config, decay_half_life_seconds=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        advance_lifecycle(
            initial, evidence(TemporalEvidenceKind.PRESENT), bad_config
        )

    assert initial == state()


def test_log_odds_are_clamped_at_both_limits_and_sigmoid_is_stable() -> None:
    extreme = TemporalLifecycleConfig(
        initial_log_odds=0.0,
        present_log_likelihood=1e308,
        absent_log_likelihood=-1e308,
        log_odds_limit=1e308,
        decay_half_life_seconds=1e308,
        active_on_probability=1.0,
        dormant_off_probability=0.0,
        minimum_absent_streak=1,
        minimum_distinct_view_bins=1,
        visibility_depth_tolerance_m=0.1,
        minimum_visible_pixel_count=0,
        minimum_visible_fraction=0.0,
        view_bin_azimuth_count=1,
        view_bin_elevation_count=1,
    )
    high = advance_lifecycle(
        state(log_odds=1e308),
        evidence(TemporalEvidenceKind.PRESENT),
        extreme,
    )
    low = advance_lifecycle(
        state(log_odds=-1e308, absent_streak=1, bins=(0,)),
        evidence(TemporalEvidenceKind.VISIBLE_ABSENT, view_bin=0),
        extreme,
    )

    assert high.existence_log_odds == 1e308
    assert low.existence_log_odds == -1e308
    assert high.lifecycle is TemporalLifecycle.ACTIVE
    assert low.lifecycle is TemporalLifecycle.DORMANT


def test_same_evidence_sequence_is_fully_deterministic(
    config: TemporalLifecycleConfig,
) -> None:
    sequence = (
        evidence(TemporalEvidenceKind.VISIBLE_ABSENT, frame_id=11, view_bin=3),
        evidence(
            TemporalEvidenceKind.OCCLUDED,
            frame_id=12,
            timestamp=12.0,
        ),
        evidence(
            TemporalEvidenceKind.PRESENT,
            strength=0.7,
            frame_id=13,
            timestamp=13.0,
        ),
    )

    def run() -> tuple[TemporalLifecycleState, ...]:
        current = state()
        results = []
        for item in sequence:
            current = advance_lifecycle(current, item, config)
            results.append(current)
        return tuple(results)

    assert run() == run()


def test_public_dataclasses_are_frozen(config: TemporalLifecycleConfig) -> None:
    lifecycle_state = state()
    lifecycle_evidence = evidence(TemporalEvidenceKind.PRESENT)

    with pytest.raises(FrozenInstanceError):
        lifecycle_state.absent_streak = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        lifecycle_evidence.strength = 0.5  # type: ignore[misc]

    assert advance_lifecycle(lifecycle_state, lifecycle_evidence, config) != lifecycle_state


def test_decay_is_applied_before_informative_update(
    config: TemporalLifecycleConfig,
) -> None:
    initial = state(log_odds=4.0)

    result = advance_lifecycle(
        initial,
        evidence(
            TemporalEvidenceKind.PRESENT,
            strength=0.5,
            timestamp=20.0,
        ),
        config,
    )

    assert result.existence_log_odds == pytest.approx(4.0)
