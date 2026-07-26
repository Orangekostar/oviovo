from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any

import numpy as np

from src.oviv2.temporal_config import TemporalDynamicConfig
from src.oviv2.temporal_lifecycle import TemporalEvidenceKind, TemporalLifecycle


_MAX_INT64 = 2**63 - 1


def _integer(value: object, name: str, *, minimum: int, maximum: int = _MAX_INT64) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: object, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _point(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise TypeError(f"{name} must contain three coordinates")
    if len(value) != 3:  # type: ignore[arg-type]
        raise ValueError(f"{name} must contain three coordinates")
    return tuple(_finite(item, name) for item in value)  # type: ignore[arg-type,return-value]


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")
    return value


def timestamp_seconds_to_ns(value: object) -> int:
    seconds = _finite(value, "timestamp")
    if seconds < 0.0:
        raise ValueError("timestamp must be nonnegative")
    scaled = seconds * 1_000_000_000
    if not math.isfinite(scaled):
        raise ValueError("timestamp cannot be represented as nanoseconds")
    timestamp_ns = int(round(scaled))
    if timestamp_ns > _MAX_INT64 or timestamp_ns / 1_000_000_000 != seconds:
        raise ValueError("timestamp must round-trip exactly through integer nanoseconds")
    return timestamp_ns


class DynamicState(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DynamicEvidenceState:
    dynamic_state: DynamicState
    motion_streak: int
    static_streak: int

    def __post_init__(self) -> None:
        if type(self.dynamic_state) is not DynamicState:
            raise TypeError("dynamic_state must be a DynamicState")
        object.__setattr__(self, "motion_streak", _integer(self.motion_streak, "motion_streak", minimum=0))
        object.__setattr__(self, "static_streak", _integer(self.static_streak, "static_streak", minimum=0))
        if self.motion_streak and self.static_streak:
            raise ValueError("motion_streak and static_streak cannot both be positive")


def _validate_dynamic_config(config: object) -> TemporalDynamicConfig:
    if type(config) is not TemporalDynamicConfig:
        raise TypeError("config must be a TemporalDynamicConfig")
    _integer(
        config.minimum_consecutive_motion_frames,
        "config.minimum_consecutive_motion_frames",
        minimum=1,
    )
    displacement_floor = _finite(config.displacement_floor_m, "config.displacement_floor_m")
    if displacement_floor < 0.0:
        raise ValueError("config.displacement_floor_m must be nonnegative")
    _probability(config.minimum_motion_confidence, "config.minimum_motion_confidence")
    _integer(config.static_off_streak_frames, "config.static_off_streak_frames", minimum=1)
    return config


def advance_dynamic_state(
    state: DynamicEvidenceState,
    *,
    accepted_motion: bool,
    displacement_m: float,
    confidence: float,
    config: TemporalDynamicConfig,
) -> DynamicEvidenceState:
    if type(state) is not DynamicEvidenceState:
        raise TypeError("state must be a DynamicEvidenceState")
    accepted = _exact_bool(accepted_motion, "accepted_motion")
    displacement = _finite(displacement_m, "displacement_m")
    if displacement < 0.0:
        raise ValueError("displacement_m must be nonnegative")
    normalized_confidence = _probability(confidence, "confidence")
    dynamic_config = _validate_dynamic_config(config)

    qualifies = (
        accepted
        and displacement >= dynamic_config.displacement_floor_m
        and normalized_confidence >= dynamic_config.minimum_motion_confidence
    )
    if qualifies:
        motion_streak = state.motion_streak + 1
        dynamic_state = (
            DynamicState.DYNAMIC
            if motion_streak >= dynamic_config.minimum_consecutive_motion_frames
            else state.dynamic_state
        )
        return DynamicEvidenceState(dynamic_state, motion_streak, 0)

    static_streak = state.static_streak + 1
    dynamic_state = (
        DynamicState.STATIC
        if static_streak >= dynamic_config.static_off_streak_frames
        else state.dynamic_state
    )
    return DynamicEvidenceState(dynamic_state, 0, static_streak)


@dataclass(frozen=True)
class TemporalExportSample:
    frame_index: int
    timestamp_ns: int
    entity_id: int
    centroid_xyz: tuple[float, float, float]
    observation_count: int
    dynamic_state: DynamicState
    motion_confidence: float
    geometry_epoch: int
    readout_valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_index", _integer(self.frame_index, "frame_index", minimum=0))
        object.__setattr__(self, "timestamp_ns", _integer(self.timestamp_ns, "timestamp_ns", minimum=0))
        object.__setattr__(self, "entity_id", _integer(self.entity_id, "entity_id", minimum=1))
        object.__setattr__(self, "centroid_xyz", _point(self.centroid_xyz, "centroid_xyz"))
        object.__setattr__(self, "observation_count", _integer(self.observation_count, "observation_count", minimum=1))
        if type(self.dynamic_state) is not DynamicState:
            raise TypeError("dynamic_state must be a DynamicState")
        object.__setattr__(self, "motion_confidence", _probability(self.motion_confidence, "motion_confidence"))
        object.__setattr__(self, "geometry_epoch", _integer(self.geometry_epoch, "geometry_epoch", minimum=0))
        object.__setattr__(self, "readout_valid", _exact_bool(self.readout_valid, "readout_valid"))

    def to_json_record(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ns": self.timestamp_ns,
            "entity_id": self.entity_id,
            "centroid_xyz": list(self.centroid_xyz),
            "observation_count": self.observation_count,
            "dynamic_state": self.dynamic_state.value,
            "motion_confidence": self.motion_confidence,
            "geometry_epoch": self.geometry_epoch,
            "readout_valid": self.readout_valid,
        }


@dataclass(frozen=True)
class TemporalLifecycleEvent:
    frame_index: int
    timestamp_ns: int
    entity_id: int
    before_lifecycle: TemporalLifecycle
    after_lifecycle: TemporalLifecycle
    evidence_kind: TemporalEvidenceKind
    geometry_epoch: int
    readout_valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_index", _integer(self.frame_index, "frame_index", minimum=0))
        object.__setattr__(self, "timestamp_ns", _integer(self.timestamp_ns, "timestamp_ns", minimum=0))
        object.__setattr__(self, "entity_id", _integer(self.entity_id, "entity_id", minimum=1))
        if type(self.before_lifecycle) is not TemporalLifecycle:
            raise TypeError("before_lifecycle must be a TemporalLifecycle")
        if type(self.after_lifecycle) is not TemporalLifecycle:
            raise TypeError("after_lifecycle must be a TemporalLifecycle")
        if type(self.evidence_kind) is not TemporalEvidenceKind:
            raise TypeError("evidence_kind must be a TemporalEvidenceKind")
        object.__setattr__(self, "geometry_epoch", _integer(self.geometry_epoch, "geometry_epoch", minimum=0))
        object.__setattr__(self, "readout_valid", _exact_bool(self.readout_valid, "readout_valid"))

    def to_json_record(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ns": self.timestamp_ns,
            "entity_id": self.entity_id,
            "before_lifecycle": self.before_lifecycle.value,
            "after_lifecycle": self.after_lifecycle.value,
            "evidence_kind": self.evidence_kind.value,
            "geometry_epoch": self.geometry_epoch,
            "readout_valid": self.readout_valid,
        }


@dataclass(frozen=True)
class TemporalExportBatch:
    frame_index: int
    timestamp_ns: int
    samples: tuple[TemporalExportSample, ...]
    events: tuple[TemporalLifecycleEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_index", _integer(self.frame_index, "frame_index", minimum=0))
        object.__setattr__(self, "timestamp_ns", _integer(self.timestamp_ns, "timestamp_ns", minimum=0))
        if type(self.samples) is not tuple or any(type(item) is not TemporalExportSample for item in self.samples):
            raise TypeError("samples must be an exact tuple of TemporalExportSample values")
        if type(self.events) is not tuple or any(type(item) is not TemporalLifecycleEvent for item in self.events):
            raise TypeError("events must be an exact tuple of TemporalLifecycleEvent values")
        samples = tuple(sorted(self.samples, key=lambda item: item.entity_id))
        events = tuple(sorted(self.events, key=lambda item: item.entity_id))
        if len({item.entity_id for item in samples}) != len(samples):
            raise ValueError("sample entity IDs must be unique")
        if len({item.entity_id for item in events}) != len(events):
            raise ValueError("event entity IDs must be unique")
        for item in (*samples, *events):
            if item.frame_index != self.frame_index or item.timestamp_ns != self.timestamp_ns:
                raise ValueError("child frame and timestamp must match batch")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "events", events)

    def to_json_record(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ns": self.timestamp_ns,
            "samples": [item.to_json_record() for item in self.samples],
            "events": [item.to_json_record() for item in self.events],
        }

    def to_canonical_json(self) -> bytes:
        return (
            json.dumps(
                self.to_json_record(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
