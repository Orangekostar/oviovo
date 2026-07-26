from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.temporal_background import TemporalBackgroundVolume
from src.oviv2.temporal_background_ledger import (
    BackgroundContribution,
    BackgroundLedgerEvidence,
    LedgerDecision,
    ReversibleBackgroundLedger,
)
from src.oviv2.temporal_config import TemporalBackgroundLedgerConfig, TemporalGeometryConfig
from src.oviv2.temporal_lifecycle import TemporalEvidenceKind


def _geometry() -> TemporalGeometryConfig:
    return TemporalGeometryConfig(1.0, 4.0, 4, 4, 4, 16, 0, 3, 0.7, 0.2, 2.0)


def _ledger_config(**changes: int) -> TemporalBackgroundLedgerConfig:
    values = dict(
        maximum_journal_blocks=8,
        commit_support_frames=2,
        commit_distinct_view_bins=2,
        minimum_commit_frame_gap=2,
        maximum_records_per_block=4,
    )
    values.update(changes)
    return TemporalBackgroundLedgerConfig(**values)


def _frame(frame_id: int) -> Frame:
    size = 4
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (4.0, 4.0, 3.0)
    return Frame(
        frame_id=frame_id,
        timestamp=float(frame_id),
        rgb=np.zeros((size, size, 3), dtype=np.uint8),
        depth=np.ones((size, size), dtype=np.float32),
        pose=pose,
        intrinsics=CameraIntrinsics(2.0, 2.0, 1.5, 1.5, size, size),
    )


def _evidence(
    *,
    entity_id: int = 1,
    epoch: int = 2,
    frame_id: int = 10,
    view_bin: int | None = 0,
    kind: TemporalEvidenceKind = TemporalEvidenceKind.VISIBLE_ABSENT,
    blocks: tuple[tuple[int, int, int], ...] = ((0, 0, 0),),
) -> BackgroundLedgerEvidence:
    frame = _frame(frame_id)
    contributions = (
        tuple(BackgroundContribution(block) for block in blocks)
        if kind is TemporalEvidenceKind.VISIBLE_ABSENT
        else ()
    )
    return BackgroundLedgerEvidence(
        entity_id=entity_id,
        geometry_epoch=epoch,
        frame_id=frame_id,
        timestamp=float(frame_id),
        kind=kind,
        view_bin=view_bin if kind is TemporalEvidenceKind.VISIBLE_ABSENT else None,
        contributions=contributions,
        frame=frame if kind is TemporalEvidenceKind.VISIBLE_ABSENT else None,
        depth_m=frame.depth if kind is TemporalEvidenceKind.VISIBLE_ABSENT else None,
    )


def _multi_block_evidence(
    *, frame_id: int, view_bin: int, reverse: bool = False
) -> BackgroundLedgerEvidence:
    frame = _frame(frame_id)
    frame.pose[:3, 3] = (8.0, 8.0, 3.0)
    keys = ((0, 0, 0), (1, 1, 0))
    if reverse:
        keys = tuple(reversed(keys))
    return BackgroundLedgerEvidence(
        entity_id=1,
        geometry_epoch=2,
        frame_id=frame_id,
        timestamp=float(frame_id),
        kind=TemporalEvidenceKind.VISIBLE_ABSENT,
        view_bin=view_bin,
        contributions=tuple(BackgroundContribution(key) for key in keys),
        frame=frame,
        depth_m=frame.depth,
    )


def _ledger(**changes: int) -> ReversibleBackgroundLedger:
    return ReversibleBackgroundLedger(_geometry(), _ledger_config(**changes))


def _masked_evidence(
    *,
    entity_id: int,
    frame: Frame,
    view_bin: int,
    masked_depth: np.ndarray,
) -> BackgroundLedgerEvidence:
    return BackgroundLedgerEvidence(
        entity_id=entity_id,
        geometry_epoch=2,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        kind=TemporalEvidenceKind.VISIBLE_ABSENT,
        view_bin=view_bin,
        contributions=(BackgroundContribution((0, 0, 0)),),
        frame=frame,
        depth_m=masked_depth,
    )


def test_visible_absent_is_provisional_until_all_three_thresholds_hold() -> None:
    ledger = _ledger(commit_support_frames=3, minimum_commit_frame_gap=3)
    assert ledger.stage(_evidence(frame_id=10, view_bin=0)) is LedgerDecision.STAGED
    assert ledger.stage(_evidence(frame_id=11, view_bin=1)) is LedgerDecision.STAGED
    assert ledger.stage(_evidence(frame_id=13, view_bin=1)) is LedgerDecision.COMMITTED
    assert ledger.provisional_count == 0
    assert ledger.committed_record_count == 3
    assert ledger.committed_generation == 1


@pytest.mark.parametrize("kind", [TemporalEvidenceKind.PRESENT, TemporalEvidenceKind.OCCLUDED])
def test_present_or_occluded_cancels_only_matching_entity_epoch_provisional(
    kind: TemporalEvidenceKind,
) -> None:
    ledger = _ledger()
    ledger.stage(_evidence(entity_id=1, epoch=2, frame_id=10))
    ledger.stage(_evidence(entity_id=1, epoch=3, frame_id=11))
    ledger.stage(_evidence(entity_id=2, epoch=2, frame_id=12))
    assert ledger.stage(
        _evidence(entity_id=1, epoch=2, frame_id=13, kind=kind)
    ) is LedgerDecision.CANCELLED
    assert ledger.provisional_keys == (
        (1, 3, 11, (0, 0, 0)),
        (2, 2, 12, (0, 0, 0)),
    )


def test_cancel_does_not_roll_back_already_committed_background() -> None:
    ledger = _ledger()
    ledger.stage(_evidence(frame_id=10, view_bin=0))
    ledger.stage(_evidence(frame_id=12, view_bin=1))
    before = ledger.committed_digest()
    assert ledger.stage(
        _evidence(frame_id=13, kind=TemporalEvidenceKind.PRESENT)
    ) is LedgerDecision.NO_OP
    assert ledger.committed_digest() == before
    assert ledger.committed_generation == 1


def test_duplicate_is_idempotent_but_conflicting_duplicate_is_rejected() -> None:
    ledger = _ledger()
    evidence = _evidence(frame_id=10)
    assert ledger.stage(evidence) is LedgerDecision.STAGED
    before = ledger.journal_digest()
    assert ledger.stage(evidence) is LedgerDecision.NO_OP
    assert ledger.journal_digest() == before
    changed = replace(evidence, view_bin=1)
    with pytest.raises(ValueError, match="duplicate|conflict"):
        ledger.stage(changed)
    assert ledger.journal_digest() == before


def test_per_block_record_cap_rejects_without_mutation() -> None:
    ledger = _ledger(
        commit_support_frames=3,
        commit_distinct_view_bins=2,
        minimum_commit_frame_gap=99,
        maximum_records_per_block=3,
    )
    for frame_id in (10, 11, 12):
        assert (
            ledger.stage(_evidence(frame_id=frame_id, view_bin=frame_id % 2))
            is LedgerDecision.STAGED
        )
    before = ledger.journal_digest()
    assert ledger.stage(_evidence(frame_id=13, view_bin=1)) is LedgerDecision.REJECTED_CAPACITY
    assert ledger.journal_digest() == before


def test_maximum_journal_blocks_counts_unique_provisional_and_committed_blocks() -> None:
    ledger = _ledger(maximum_journal_blocks=2)
    assert (
        ledger.stage(_multi_block_evidence(frame_id=10, view_bin=0))
        is LedgerDecision.STAGED
    )
    before = ledger.journal_digest()
    third_block = _frame(11)
    third_block.pose[:3, 3] = (12.0, 4.0, 3.0)
    evidence = BackgroundLedgerEvidence(
        entity_id=2,
        geometry_epoch=2,
        frame_id=11,
        timestamp=11.0,
        kind=TemporalEvidenceKind.VISIBLE_ABSENT,
        view_bin=0,
        contributions=(BackgroundContribution((1, 0, 0)),),
        frame=third_block,
        depth_m=third_block.depth,
    )
    assert (
        ledger.stage(evidence) is LedgerDecision.REJECTED_CAPACITY
    )
    assert ledger.journal_digest() == before


def test_rebuild_is_deterministic_across_evidence_and_block_order() -> None:
    left = _ledger()
    right = _ledger()
    left.stage(_multi_block_evidence(frame_id=10, view_bin=0, reverse=True))
    left.stage(_multi_block_evidence(frame_id=12, view_bin=1, reverse=True))
    right.stage(_multi_block_evidence(frame_id=10, view_bin=0))
    right.stage(_multi_block_evidence(frame_id=12, view_bin=1))
    assert left.committed_digest() == right.committed_digest()
    assert (
        left.committed_volume.canonical_block_state()
        == right.committed_volume.canonical_block_state()
    )


@pytest.mark.parametrize("failure_site", ["integration", "rebuild"])
def test_integration_or_rebuild_failure_rolls_back_complete_transaction(
    monkeypatch: pytest.MonkeyPatch, failure_site: str
) -> None:
    ledger = _ledger()
    ledger.stage(_evidence(frame_id=10, view_bin=0))
    before = (ledger.journal_digest(), ledger.committed_digest(), ledger.committed_generation)
    target = (
        "trial_integrate_blocks"
        if failure_site == "integration"
        else "rebuild_blocks"
    )

    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected")

    monkeypatch.setattr(TemporalBackgroundVolume, target, fail)
    assert ledger.stage(_evidence(frame_id=12, view_bin=1)) is LedgerDecision.REJECTED_INTEGRATION
    assert (
        ledger.journal_digest(),
        ledger.committed_digest(),
        ledger.committed_generation,
    ) == before
    assert ledger.provisional_count == 1


def test_inputs_are_deep_copied_readonly_and_frozen() -> None:
    frame = _frame(10)
    depth = frame.depth.copy()
    contribution = BackgroundContribution((0, 0, 0))
    evidence = BackgroundLedgerEvidence(
        entity_id=1,
        geometry_epoch=2,
        frame_id=10,
        timestamp=10.0,
        kind=TemporalEvidenceKind.VISIBLE_ABSENT,
        view_bin=0,
        contributions=(contribution,),
        frame=frame,
        depth_m=depth,
    )
    frame.depth[:] = 2.0
    depth[:] = 3.0
    assert evidence.frame is not None
    assert evidence.depth_m is not None
    assert np.all(evidence.frame.depth == 1.0)
    assert np.all(evidence.depth_m == 1.0)
    assert not evidence.frame.depth.flags.writeable
    assert not evidence.depth_m.flags.writeable
    with pytest.raises(FrozenInstanceError):
        contribution.block_key = (1, 0, 0)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        evidence.frame.timestamp = 11.0
    with pytest.raises(FrozenInstanceError):
        evidence.frame.intrinsics.fx = 3.0


@pytest.mark.parametrize(
    "bad",
    [
        lambda e: replace(e, entity_id=True),
        lambda e: replace(e, geometry_epoch=True),
        lambda e: replace(e, frame_id=True),
        lambda e: replace(e, timestamp=float("nan")),
        lambda e: replace(e, view_bin=True),
        lambda e: replace(e, contributions=e.contributions + e.contributions),
        lambda e: replace(e, contributions=(replace(e.contributions[0], block_key=[0, 0, 0]),)),
    ],
)
def test_strict_evidence_validation_is_fail_closed(bad: object) -> None:
    ledger = _ledger()
    before = ledger.journal_digest()
    with pytest.raises((TypeError, ValueError)):
        ledger.stage(bad(_evidence()))  # type: ignore[operator]
    assert ledger.journal_digest() == before


def test_frame_and_time_must_not_go_backwards() -> None:
    ledger = _ledger()
    ledger.stage(_evidence(frame_id=10))
    before = ledger.journal_digest()
    with pytest.raises(ValueError, match="frame_id|timestamp"):
        ledger.stage(replace(_evidence(frame_id=9), timestamp=11.0))
    assert ledger.journal_digest() == before


def test_same_frame_accepts_multiple_entities_but_rejects_frame_or_time_rollback() -> None:
    ledger = _ledger()
    ledger.stage(_evidence(frame_id=10))
    assert ledger.stage(
        _evidence(entity_id=99, frame_id=11, kind=TemporalEvidenceKind.PRESENT)
    ) is LedgerDecision.NO_OP
    assert ledger.stage(_evidence(entity_id=2, frame_id=11)) is LedgerDecision.STAGED
    with pytest.raises(ValueError, match="frame_id"):
        ledger.stage(_evidence(entity_id=3, frame_id=10))
    with pytest.raises(ValueError, match="timestamp"):
        ledger.stage(
            replace(
                _evidence(
                    entity_id=3,
                    frame_id=11,
                    kind=TemporalEvidenceKind.PRESENT,
                ),
                timestamp=12.0,
            )
        )


@pytest.mark.parametrize("damage", ["rgb", "pose", "depth"])
def test_bad_contribution_geometry_is_rejected_before_provisional_publication(
    damage: str,
) -> None:
    frame = _frame(10)
    depth = frame.depth.copy()
    if damage == "rgb":
        frame.rgb = np.ones((4, 4, 3), dtype=np.int32)
    elif damage == "pose":
        frame.pose[3, 3] = 2.0
    else:
        depth[:] = 5.0
    ledger = _ledger()
    before = ledger.journal_digest()
    with pytest.raises((TypeError, ValueError)):
        ledger.stage(
            BackgroundLedgerEvidence(
                entity_id=1,
                geometry_epoch=2,
                frame_id=10,
                timestamp=10.0,
                kind=TemporalEvidenceKind.VISIBLE_ABSENT,
                view_bin=0,
                contributions=(BackgroundContribution((0, 0, 0)),),
                frame=frame,
                depth_m=depth,
            )
        )
    assert ledger.journal_digest() == before
    assert ledger.provisional_count == 0


def test_declared_blocks_must_exactly_match_observation_touched_blocks() -> None:
    ledger = _ledger()
    before = ledger.journal_digest()
    with pytest.raises(ValueError, match="block_key|touched blocks"):
        ledger.stage(_evidence(frame_id=10, blocks=((7, 0, 0),)))
    assert ledger.journal_digest() == before


def test_multiblock_observation_is_integrated_once_per_entity_frame() -> None:
    ledger = _ledger()
    first = _multi_block_evidence(frame_id=10, view_bin=0)
    second = _multi_block_evidence(frame_id=12, view_bin=1)
    ledger.stage(first)
    assert ledger.stage(second) is LedgerDecision.COMMITTED
    expected = TemporalBackgroundVolume.rebuild(
        _geometry(),
        (
            ((1, 2, 10), first.frame, first.depth_m),
            ((1, 2, 12), second.frame, second.depth_m),
        ),
    )
    assert ledger.committed_volume.canonical_block_state() == expected.canonical_block_state()


def test_provisional_block_never_leaks_through_partially_committed_observation() -> None:
    ledger = _ledger()
    ledger.stage(_evidence(frame_id=10, view_bin=0))
    assert ledger.stage(
        _multi_block_evidence(frame_id=12, view_bin=1)
    ) is LedgerDecision.COMMITTED
    assert ledger.provisional_keys == ((1, 2, 12, (1, 1, 0)),)
    assert ledger.committed_volume.active_block_count == 1
    before = ledger.committed_digest()
    assert ledger.stage(
        _evidence(frame_id=13, kind=TemporalEvidenceKind.PRESENT)
    ) is LedgerDecision.CANCELLED
    assert ledger.committed_digest() == before
    assert ledger.committed_volume.active_block_count == 1


def test_multiblock_records_share_one_frozen_observation() -> None:
    evidence = _multi_block_evidence(frame_id=10, view_bin=0)
    assert evidence.frame is not None
    assert evidence.depth_m is not None
    assert not hasattr(evidence.contributions[0], "frame")
    assert not hasattr(evidence.contributions[0], "depth_m")


@pytest.mark.parametrize(
    "config",
    [
        _ledger_config(commit_support_frames=1),
        _ledger_config(commit_distinct_view_bins=1),
        _ledger_config(maximum_journal_blocks=17),
    ],
)
def test_direct_config_construction_cannot_bypass_frozen_constraints(
    config: TemporalBackgroundLedgerConfig,
) -> None:
    with pytest.raises(ValueError):
        ReversibleBackgroundLedger(_geometry(), config)


def test_journal_digest_covers_accepted_event_watermark() -> None:
    left = _ledger()
    right = _ledger()
    left.stage(_evidence(frame_id=10))
    right.stage(_evidence(frame_id=10))
    right.stage(_evidence(entity_id=9, frame_id=11, kind=TemporalEvidenceKind.PRESENT))
    assert left.journal_digest() != right.journal_digest()


def test_cross_entity_duplicate_native_observations_integrate_once_per_frame_block() -> None:
    ledger = _ledger(maximum_records_per_block=2)
    evidence: dict[tuple[int, int], BackgroundLedgerEvidence] = {}
    for frame_id, view_bin in ((10, 0), (12, 1)):
        frame = _frame(frame_id)
        for entity_id in (1, 2):
            item = _masked_evidence(
                entity_id=entity_id,
                frame=frame,
                view_bin=view_bin,
                masked_depth=frame.depth,
            )
            evidence[(entity_id, frame_id)] = item
            assert ledger.stage(item) is not LedgerDecision.REJECTED_CAPACITY

    expected = TemporalBackgroundVolume.rebuild_blocks(
        _geometry(),
        tuple(
            (
                (0, frame_id, (0, 0, 0)),
                (0, 0, 0),
                evidence[(1, frame_id)].frame,
                evidence[(1, frame_id)].depth_m,
            )
            for frame_id in (10, 12)
        ),
    )
    assert ledger.committed_record_count == 4
    assert ledger.committed_volume.canonical_block_state() == expected.canonical_block_state()


def test_cross_entity_complementary_masks_union_then_integrate_once() -> None:
    ledger = _ledger(maximum_records_per_block=2)
    full_observations: list[tuple[object, tuple[int, int, int], Frame, np.ndarray]] = []
    for frame_id, view_bin in ((10, 0), (12, 1)):
        frame = _frame(frame_id)
        frame.rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        frame.depth = np.ones((16, 16), dtype=np.float32)
        frame.intrinsics = CameraIntrinsics(8.0, 8.0, 7.5, 7.5, 16, 16)
        left = np.zeros_like(frame.depth)
        right = np.zeros_like(frame.depth)
        left[:, :8] = frame.depth[:, :8]
        right[:, 8:] = frame.depth[:, 8:]
        ledger.stage(
            _masked_evidence(
                entity_id=1,
                frame=frame,
                view_bin=view_bin,
                masked_depth=left,
            )
        )
        ledger.stage(
            _masked_evidence(
                entity_id=2,
                frame=frame,
                view_bin=view_bin,
                masked_depth=right,
            )
        )
        full_observations.append(
            (
                (0, frame_id, (0, 0, 0)),
                (0, 0, 0),
                frame,
                frame.depth,
            )
        )
    expected = TemporalBackgroundVolume.rebuild_blocks(
        _geometry(), tuple(full_observations)
    )
    assert ledger.committed_volume.canonical_block_state() == expected.canonical_block_state()


@pytest.mark.parametrize(
    "damage", ["rgb", "depth", "pose", "intrinsics", "timestamp", "source"]
)
def test_native_frame_conflict_rejects_without_any_state_change(
    damage: str,
) -> None:
    ledger = _ledger()
    ledger.stage(_evidence(entity_id=1, frame_id=10))
    conflicting = _frame(10)
    if damage == "rgb":
        conflicting.rgb[0, 0] = (1, 2, 3)
    elif damage == "depth":
        conflicting.depth[0, 0] = 2.0
    elif damage == "pose":
        conflicting.pose[0, 3] += 0.25
    elif damage == "intrinsics":
        conflicting.intrinsics.fx += 0.25
    elif damage == "timestamp":
        conflicting.timestamp += 0.25
    else:
        conflicting.source_frame_id = 99
    evidence = _masked_evidence(
        entity_id=2,
        frame=conflicting,
        view_bin=0,
        masked_depth=conflicting.depth,
    )
    before = (ledger.journal_digest(), ledger.committed_digest())
    with pytest.raises(ValueError, match="native frame|frame content|timestamp"):
        ledger.stage(evidence)
    assert (ledger.journal_digest(), ledger.committed_digest()) == before


def test_masked_nonzero_depth_must_equal_valid_native_depth() -> None:
    frame = _frame(10)
    forged = frame.depth.copy()
    forged[0, 0] = 2.0
    with pytest.raises(ValueError, match="native frame.depth|masked depth"):
        _masked_evidence(
            entity_id=1,
            frame=frame,
            view_bin=0,
            masked_depth=forged,
        )


def test_duplicate_ownership_does_not_consume_observation_capacity_early() -> None:
    ledger = _ledger(maximum_records_per_block=2)
    for frame_id, view_bin in ((10, 0), (12, 1)):
        frame = _frame(frame_id)
        for entity_id in range(1, _geometry().maximum_entities + 1):
            decision = ledger.stage(
                _masked_evidence(
                    entity_id=entity_id,
                    frame=frame,
                    view_bin=view_bin,
                    masked_depth=frame.depth,
                )
            )
            assert decision is not LedgerDecision.REJECTED_CAPACITY
    assert ledger.committed_record_count == 8


def test_ownership_records_per_native_observation_are_geometry_bounded() -> None:
    ledger = _ledger()
    frame = _frame(10)
    for entity_id in range(1, _geometry().maximum_entities + 1):
        assert ledger.stage(
            _masked_evidence(
                entity_id=entity_id,
                frame=frame,
                view_bin=0,
                masked_depth=frame.depth,
            )
        ) is LedgerDecision.STAGED
    before = ledger.journal_digest()
    assert ledger.stage(
        _masked_evidence(
            entity_id=_geometry().maximum_entities + 1,
            frame=frame,
            view_bin=0,
            masked_depth=frame.depth,
        )
    ) is LedgerDecision.REJECTED_CAPACITY
    assert ledger.journal_digest() == before


def test_same_source_frame_under_new_internal_id_does_not_add_support() -> None:
    ledger = _ledger()
    first = _frame(10)
    first.source_frame_id = 77
    second = _frame(12)
    second.source_frame_id = 77
    second.timestamp = first.timestamp
    assert ledger.stage(
        _masked_evidence(
            entity_id=1,
            frame=first,
            view_bin=0,
            masked_depth=first.depth,
        )
    ) is LedgerDecision.STAGED
    assert ledger.stage(
        _masked_evidence(
            entity_id=1,
            frame=second,
            view_bin=1,
            masked_depth=second.depth,
        )
    ) is LedgerDecision.STAGED
    assert ledger.committed_record_count == 0


def test_same_source_frame_content_conflict_across_internal_ids_rolls_back() -> None:
    ledger = _ledger()
    first = _frame(10)
    first.source_frame_id = 77
    ledger.stage(
        _masked_evidence(
            entity_id=1,
            frame=first,
            view_bin=0,
            masked_depth=first.depth,
        )
    )
    conflicting = _frame(12)
    conflicting.source_frame_id = 77
    conflicting.timestamp = first.timestamp
    conflicting.rgb[0, 0] = (1, 2, 3)
    before = ledger.journal_digest()
    with pytest.raises(ValueError, match="native frame|frame content"):
        ledger.stage(
            _masked_evidence(
                entity_id=2,
                frame=conflicting,
                view_bin=0,
                masked_depth=conflicting.depth,
            )
        )
    assert ledger.journal_digest() == before
