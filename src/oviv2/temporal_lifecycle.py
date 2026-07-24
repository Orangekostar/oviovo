from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

from src.oviv2.temporal_config import TemporalLifecycleConfig


class TemporalLifecycle(str, Enum):
    ACTIVE = "active"
    UNCERTAIN = "uncertain"
    DORMANT = "dormant"


class TemporalEvidenceKind(str, Enum):
    PRESENT = "present"
    VISIBLE_ABSENT = "visible_absent"
    OCCLUDED = "occluded"
    OUT_OF_VIEW = "out_of_view"
    DEPTH_UNKNOWN = "depth_unknown"


@dataclass(frozen=True)
class TemporalLifecycleState:
    entity_id: int
    lifecycle: TemporalLifecycle
    existence_log_odds: float
    last_frame_id: int
    last_timestamp: float
    absent_streak: int
    absence_view_bins: tuple[int, ...]


@dataclass(frozen=True)
class TemporalEvidence:
    kind: TemporalEvidenceKind
    strength: float
    frame_id: int
    timestamp: float
    view_bin: int | None


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _validate_config(config: object) -> TemporalLifecycleConfig:
    if not isinstance(config, TemporalLifecycleConfig):
        raise TypeError("config must be a TemporalLifecycleConfig")

    numeric_fields = (
        "initial_log_odds",
        "present_log_likelihood",
        "absent_log_likelihood",
        "log_odds_limit",
        "decay_half_life_seconds",
        "active_on_probability",
        "dormant_off_probability",
        "visibility_depth_tolerance_m",
        "minimum_visible_fraction",
    )
    values = {
        name: _finite_number(getattr(config, name), f"config.{name}")
        for name in numeric_fields
    }
    for name in (
        "minimum_absent_streak",
        "minimum_distinct_view_bins",
        "view_bin_azimuth_count",
        "view_bin_elevation_count",
    ):
        _integer(getattr(config, name), f"config.{name}", minimum=1)
    _integer(
        config.minimum_visible_pixel_count,
        "config.minimum_visible_pixel_count",
    )

    limit = values["log_odds_limit"]
    if limit <= 0.0:
        raise ValueError("config.log_odds_limit must be positive")
    if abs(values["initial_log_odds"]) > limit:
        raise ValueError("config.initial_log_odds must be within log_odds_limit")
    if not 0.0 < values["present_log_likelihood"] <= limit:
        raise ValueError("config.present_log_likelihood must be positive and bounded")
    if not -limit <= values["absent_log_likelihood"] < 0.0:
        raise ValueError("config.absent_log_likelihood must be negative and bounded")
    if values["decay_half_life_seconds"] <= 0.0:
        raise ValueError("config.decay_half_life_seconds must be positive")
    if values["visibility_depth_tolerance_m"] <= 0.0:
        raise ValueError("config.visibility_depth_tolerance_m must be positive")
    for name in (
        "active_on_probability",
        "dormant_off_probability",
        "minimum_visible_fraction",
    ):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"config.{name} must be in [0, 1]")
    if values["dormant_off_probability"] >= values["active_on_probability"]:
        raise ValueError("config dormant threshold must be below active threshold")
    available_bins = config.view_bin_azimuth_count * config.view_bin_elevation_count
    if config.minimum_distinct_view_bins > available_bins:
        raise ValueError("config.minimum_distinct_view_bins exceeds available bins")
    return config


def _validate_state(
    state: object, config: TemporalLifecycleConfig
) -> TemporalLifecycleState:
    if not isinstance(state, TemporalLifecycleState):
        raise TypeError("state must be a TemporalLifecycleState")
    _integer(state.entity_id, "state.entity_id")
    if not isinstance(state.lifecycle, TemporalLifecycle):
        raise TypeError("state.lifecycle must be a TemporalLifecycle")
    log_odds = _finite_number(state.existence_log_odds, "state.existence_log_odds")
    if abs(log_odds) > config.log_odds_limit:
        raise ValueError("state.existence_log_odds exceeds config.log_odds_limit")
    _integer(state.last_frame_id, "state.last_frame_id")
    _finite_number(state.last_timestamp, "state.last_timestamp")
    _integer(state.absent_streak, "state.absent_streak")
    if not isinstance(state.absence_view_bins, tuple):
        raise TypeError("state.absence_view_bins must be a tuple")
    available_bins = config.view_bin_azimuth_count * config.view_bin_elevation_count
    normalized_bins = tuple(
        _integer(value, "state.absence_view_bins item")
        for value in state.absence_view_bins
    )
    if normalized_bins != tuple(sorted(set(normalized_bins))):
        raise ValueError("state.absence_view_bins must be sorted and unique")
    if any(value >= available_bins for value in normalized_bins):
        raise ValueError("state.absence_view_bins item is out of range")
    if len(normalized_bins) > state.absent_streak:
        raise ValueError("state.absence_view_bins cannot exceed absent_streak")
    return state


def _validate_evidence(
    evidence: object,
    state: TemporalLifecycleState,
    config: TemporalLifecycleConfig,
) -> TemporalEvidence:
    if not isinstance(evidence, TemporalEvidence):
        raise TypeError("evidence must be a TemporalEvidence")
    if not isinstance(evidence.kind, TemporalEvidenceKind):
        raise TypeError("evidence.kind must be a TemporalEvidenceKind")
    strength = _finite_number(evidence.strength, "evidence.strength")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("evidence.strength must be in [0, 1]")
    frame_id = _integer(evidence.frame_id, "evidence.frame_id")
    timestamp = _finite_number(evidence.timestamp, "evidence.timestamp")
    if frame_id <= state.last_frame_id:
        raise ValueError("evidence.frame_id must increase strictly")
    if timestamp <= state.last_timestamp:
        raise ValueError("evidence.timestamp must increase strictly")

    if evidence.kind is TemporalEvidenceKind.VISIBLE_ABSENT:
        view_bin = _integer(evidence.view_bin, "evidence.view_bin")
        available_bins = config.view_bin_azimuth_count * config.view_bin_elevation_count
        if view_bin >= available_bins:
            raise ValueError("evidence.view_bin is out of range")
    elif evidence.view_bin is not None:
        raise ValueError("evidence.view_bin must be None for this evidence kind")
    return evidence


def _sigmoid(log_odds: float) -> float:
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    exp_value = math.exp(log_odds)
    return exp_value / (1.0 + exp_value)


def _clamp_log_odds(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def advance_lifecycle(
    state: TemporalLifecycleState,
    evidence: TemporalEvidence,
    config: TemporalLifecycleConfig,
) -> TemporalLifecycleState:
    config = _validate_config(config)
    state = _validate_state(state, config)
    evidence = _validate_evidence(evidence, state, config)

    if evidence.kind in (
        TemporalEvidenceKind.OCCLUDED,
        TemporalEvidenceKind.OUT_OF_VIEW,
        TemporalEvidenceKind.DEPTH_UNKNOWN,
    ):
        return TemporalLifecycleState(
            entity_id=state.entity_id,
            lifecycle=state.lifecycle,
            existence_log_odds=state.existence_log_odds,
            last_frame_id=evidence.frame_id,
            last_timestamp=evidence.timestamp,
            absent_streak=0,
            absence_view_bins=(),
        )

    delta_t = evidence.timestamp - state.last_timestamp
    decay = 0.5 ** (delta_t / config.decay_half_life_seconds)
    decayed_log_odds = state.existence_log_odds * decay

    if evidence.kind is TemporalEvidenceKind.PRESENT:
        log_odds = decayed_log_odds + config.present_log_likelihood * evidence.strength
        absent_streak = 0
        absence_view_bins: tuple[int, ...] = ()
    else:
        assert evidence.view_bin is not None
        repeated_view = evidence.view_bin in state.absence_view_bins
        effective_strength = evidence.strength * (0.5 if repeated_view else 1.0)
        log_odds = decayed_log_odds + config.absent_log_likelihood * effective_strength
        absent_streak = state.absent_streak + 1
        absence_view_bins = tuple(
            sorted((*state.absence_view_bins, evidence.view_bin))
        )
        if repeated_view:
            absence_view_bins = state.absence_view_bins

    log_odds = _clamp_log_odds(log_odds, config.log_odds_limit)
    probability = _sigmoid(log_odds)
    if state.lifecycle is TemporalLifecycle.DORMANT:
        lifecycle = (
            TemporalLifecycle.ACTIVE
            if evidence.kind is TemporalEvidenceKind.PRESENT
            and probability >= config.active_on_probability
            else TemporalLifecycle.DORMANT
        )
    elif probability >= config.active_on_probability:
        lifecycle = TemporalLifecycle.ACTIVE
    elif (
        probability <= config.dormant_off_probability
        and absent_streak >= max(2, config.minimum_absent_streak)
        and len(absence_view_bins) >= config.minimum_distinct_view_bins
    ):
        lifecycle = TemporalLifecycle.DORMANT
    else:
        lifecycle = TemporalLifecycle.UNCERTAIN

    return TemporalLifecycleState(
        entity_id=state.entity_id,
        lifecycle=lifecycle,
        existence_log_odds=log_odds,
        last_frame_id=evidence.frame_id,
        last_timestamp=evidence.timestamp,
        absent_streak=absent_streak,
        absence_view_bins=absence_view_bins,
    )
