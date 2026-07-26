from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from src.oviv2 import temporal_export
from src.oviv2.temporal_config import TemporalDynamicConfig
from src.oviv2.temporal_export import (
    DynamicEvidenceState,
    DynamicState,
    TemporalExportBatch,
    TemporalExportSample,
    TemporalLifecycleEvent,
    advance_dynamic_state,
)
from src.oviv2.temporal_lifecycle import TemporalEvidenceKind, TemporalLifecycle


CONFIG = TemporalDynamicConfig(2, 0.1, 0.7, 3)


def state(
    dynamic_state: DynamicState = DynamicState.UNKNOWN,
    motion_streak: int = 0,
    static_streak: int = 0,
) -> DynamicEvidenceState:
    return DynamicEvidenceState(dynamic_state, motion_streak, static_streak)


def sample(entity_id: int = 1, **changes: object) -> TemporalExportSample:
    values: dict[str, object] = {
        "frame_index": 3,
        "timestamp_ns": 1_250_000_000,
        "entity_id": entity_id,
        "centroid_xyz": (1.0, 2.0, 3.0),
        "observation_count": 2,
        "dynamic_state": DynamicState.STATIC,
        "motion_confidence": 0.75,
        "geometry_epoch": 0,
        "readout_valid": True,
    }
    values.update(changes)
    return TemporalExportSample(**values)  # type: ignore[arg-type]


def event(entity_id: int = 1, **changes: object) -> TemporalLifecycleEvent:
    values: dict[str, object] = {
        "frame_index": 3,
        "timestamp_ns": 1_250_000_000,
        "entity_id": entity_id,
        "before": TemporalLifecycle.ACTIVE,
        "after": TemporalLifecycle.UNCERTAIN,
        "evidence": TemporalEvidenceKind.VISIBLE_ABSENT,
        "geometry_epoch": 0,
        "readout_valid": False,
    }
    values.update(changes)
    return TemporalLifecycleEvent(**values)  # type: ignore[arg-type]


def test_dynamic_state_is_explicit_frozen_and_rejects_invalid_streaks() -> None:
    with pytest.raises(TypeError, match="dynamic_state"):
        DynamicEvidenceState("unknown", 0, 0)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        DynamicEvidenceState(DynamicState.UNKNOWN, True, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DynamicEvidenceState(DynamicState.UNKNOWN, -1, 0)
    with pytest.raises(FrozenInstanceError):
        state().motion_streak = 1  # type: ignore[misc]


def test_planned_dynamic_state_api_is_static_and_string_compatible() -> None:
    initial = DynamicEvidenceState.static()

    assert initial == DynamicEvidenceState(DynamicState.STATIC, 0, 0)
    assert initial.state is DynamicState.STATIC
    assert isinstance(DynamicState.STATIC, str)
    with pytest.raises(FrozenInstanceError):
        initial.state = DynamicState.DYNAMIC  # type: ignore[misc]


def test_rejected_motion_never_accumulates_dynamic_evidence() -> None:
    current = state()
    for _ in range(4):
        current = advance_dynamic_state(
            current,
            accepted_motion=False,
            displacement_m=1.0,
            confidence=1.0,
            config=CONFIG,
        )
    assert current.dynamic_state is not DynamicState.DYNAMIC
    assert current.motion_streak == 0


def test_dynamic_state_uses_consecutive_motion_and_static_off_hysteresis() -> None:
    first = advance_dynamic_state(
        state(), accepted_motion=True, displacement_m=0.1, confidence=0.7, config=CONFIG
    )
    dynamic = advance_dynamic_state(
        first, accepted_motion=True, displacement_m=0.2, confidence=0.9, config=CONFIG
    )
    assert first == state(DynamicState.UNKNOWN, motion_streak=1)
    assert dynamic == state(DynamicState.DYNAMIC, motion_streak=2)

    for expected_streak in (1, 2):
        dynamic = advance_dynamic_state(
            dynamic,
            accepted_motion=True,
            displacement_m=0.01,
            confidence=1.0,
            config=CONFIG,
        )
        assert dynamic == state(DynamicState.DYNAMIC, static_streak=expected_streak)
    assert advance_dynamic_state(
        dynamic,
        accepted_motion=True,
        displacement_m=0.01,
        confidence=1.0,
        config=CONFIG,
    ) == state(DynamicState.STATIC, static_streak=3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accepted_motion", 1),
        ("displacement_m", True),
        ("displacement_m", -0.1),
        ("displacement_m", float("nan")),
        ("confidence", False),
        ("confidence", float("inf")),
    ],
)
def test_dynamic_state_rejects_non_exact_or_nonfinite_inputs(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "accepted_motion": True,
        "displacement_m": 0.2,
        "confidence": 1.0,
        "config": CONFIG,
    }
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        advance_dynamic_state(state(), **arguments)  # type: ignore[arg-type]


def test_empty_batch_and_visible_absence_without_sample_are_canonical_json() -> None:
    empty = TemporalExportBatch(3, 1_250_000_000, (), ())
    absent = TemporalExportBatch(3, 1_250_000_000, (), (event(),))

    assert empty.to_json_record() == {
        "frame_index": 3,
        "timestamp_ns": 1_250_000_000,
        "samples": [],
        "events": [],
    }
    assert absent.samples == ()
    assert empty.to_json_records() == ()
    assert absent.to_json_records() == (event().to_json_record(),)
    assert json.loads(absent.to_canonical_json()) == absent.to_json_record()
    json.dumps(absent.to_json_record(), allow_nan=False)


def test_lifecycle_event_uses_planned_fields_with_read_only_compatibility_aliases() -> None:
    lifecycle_event = event()

    assert lifecycle_event.before is TemporalLifecycle.ACTIVE
    assert lifecycle_event.after is TemporalLifecycle.UNCERTAIN
    assert lifecycle_event.evidence is TemporalEvidenceKind.VISIBLE_ABSENT
    assert lifecycle_event.before_lifecycle is lifecycle_event.before
    assert lifecycle_event.after_lifecycle is lifecycle_event.after
    assert lifecycle_event.evidence_kind is lifecycle_event.evidence
    assert set(lifecycle_event.to_json_record()) >= {"before", "after", "evidence"}
    assert "before_lifecycle" not in lifecycle_event.to_json_record()


def test_batch_sorts_children_and_requires_unique_matching_exact_tuples() -> None:
    batch = TemporalExportBatch(
        3,
        1_250_000_000,
        (sample(9), sample(2)),
        (event(7), event(4)),
    )
    assert tuple(item.entity_id for item in batch.samples) == (2, 9)
    assert tuple(item.entity_id for item in batch.events) == (4, 7)

    with pytest.raises(TypeError):
        TemporalExportBatch(3, 1_250_000_000, [], ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        TemporalExportBatch(3, 1_250_000_000, (sample(), sample()), ())
    with pytest.raises(ValueError, match="match"):
        TemporalExportBatch(4, 1_250_000_000, (sample(),), ())
    with pytest.raises(ValueError, match="match"):
        TemporalExportBatch(3, 1_250_000_000, (), (event(timestamp_ns=2),))


def test_batch_validate_after_requires_strict_progress_and_observation_growth() -> None:
    previous = TemporalExportBatch(3, 1_250_000_000, (sample(observation_count=2),), ())
    current = TemporalExportBatch(
        4,
        1_500_000_000,
        (sample(frame_index=4, timestamp_ns=1_500_000_000, observation_count=3),),
        (),
    )

    assert current.validate_after(previous) is None
    assert temporal_export.validate_temporal_export_sequence((previous, current)) is None

    invalid = (
        TemporalExportBatch(3, 1_500_000_000, (), ()),
        TemporalExportBatch(4, 1_250_000_000, (), ()),
        TemporalExportBatch(
            4,
            1_500_000_000,
            (sample(frame_index=4, timestamp_ns=1_500_000_000, observation_count=2),),
            (),
        ),
        TemporalExportBatch(
            4,
            1_500_000_000,
            (sample(frame_index=4, timestamp_ns=1_500_000_000, observation_count=1),),
            (),
        ),
    )
    for batch in invalid:
        with pytest.raises(ValueError):
            batch.validate_after(previous)


def test_sequence_validation_tracks_counts_across_empty_batches() -> None:
    first = TemporalExportBatch(1, 100, (sample(frame_index=1, timestamp_ns=100),), ())
    empty = TemporalExportBatch(2, 200, (), ())
    regressed = TemporalExportBatch(
        3,
        300,
        (sample(frame_index=3, timestamp_ns=300, observation_count=2),),
        (),
    )

    with pytest.raises(ValueError, match="observation_count"):
        temporal_export.validate_temporal_export_sequence((first, empty, regressed))


@pytest.mark.parametrize(
    "bad_sample",
    [
        lambda: sample(entity_id=True),
        lambda: sample(centroid_xyz=(0.0, float("nan"), 0.0)),
        lambda: sample(observation_count=0),
        lambda: sample(dynamic_state="static"),
        lambda: sample(motion_confidence=float("inf")),
        lambda: sample(geometry_epoch=-1),
        lambda: sample(readout_valid=1),
    ],
)
def test_sample_rejects_invalid_contract_values(bad_sample) -> None:
    with pytest.raises((TypeError, ValueError)):
        bad_sample()


@pytest.mark.parametrize(
    "bad_event",
    [
        lambda: event(before="active"),
        lambda: event(after="uncertain"),
        lambda: event(evidence="visible_absent"),
        lambda: event(readout_valid=0),
    ],
)
def test_event_rejects_invalid_contract_values(bad_event) -> None:
    with pytest.raises((TypeError, ValueError)):
        bad_event()
