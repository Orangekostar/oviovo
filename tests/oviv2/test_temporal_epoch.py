from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import itertools

import numpy as np
import pytest

from src.oviv2.temporal_config import TemporalGeometryConfig
from src.oviv2.temporal_epoch import GeometryEpoch, start_new_epoch
from src.oviv2.temporal_geometry import MotionDecision, ObjectMotionEstimate, ObjectSubmap
from src.oviv2.temporal_lifecycle import TemporalEvidenceKind


def _config(**changes: object) -> TemporalGeometryConfig:
    values = {
        "voxel_size_m": 1.0,
        "depth_max_m": 4.0,
        "maximum_entities": 10,
        "maximum_object_voxels": 10,
        "maximum_visibility_points_per_entity": 10,
        "background_block_count": 10,
        "background_mask_dilation_px": 0,
        "minimum_icp_points": 3,
        "minimum_icp_fitness": 0.7,
        "maximum_icp_rmse_m": 0.2,
        "maximum_motion_m": 2.0,
    }
    values.update(changes)
    return TemporalGeometryConfig(**values)


def _submap() -> ObjectSubmap:
    return ObjectSubmap(
        (0.0, 0.0, 0.0),
        ((0, 0, 0),),
        np.asarray([[0.1, 0.0, 0.0]]),
        np.asarray([1.0]),
        np.asarray([1], dtype=np.int64),
    )


def _estimate(
    decision: MotionDecision = MotionDecision.TRANSLATION_ACCEPTED,
    x: float = 0.0,
) -> ObjectMotionEstimate:
    pose = np.eye(4)
    pose[0, 3] = x
    return ObjectMotionEstimate(pose, decision, 0.0, 0.2)


def _epoch(*, epoch_id: int = 3, readout_valid: bool = True) -> GeometryEpoch:
    return GeometryEpoch(7, epoch_id, np.eye(4), _submap(), readout_valid)


def test_epoch_is_strictly_frozen_owned_and_array_safe() -> None:
    pose = np.eye(4)
    epoch = GeometryEpoch(7, 3, pose, _submap(), True)
    pose[:] = 9.0

    np.testing.assert_array_equal(epoch.object_to_world, np.eye(4))
    assert not epoch.object_to_world.flags.writeable
    copied = copy.deepcopy(epoch)
    assert copied == epoch and copied is not epoch
    assert copied.last_processed_frame_id == epoch.last_processed_frame_id == 1
    for original, duplicate in (
        (epoch.object_to_world, copied.object_to_world),
        (epoch.submap.local_points_xyz, copied.submap.local_points_xyz),
        (epoch.submap.weights, copied.submap.weights),
        (epoch.submap.last_seen_frame_ids, copied.submap.last_seen_frame_ids),
    ):
        assert not duplicate.flags.writeable
        assert not np.shares_memory(original, duplicate)
        with pytest.raises(ValueError, match="read-only"):
            duplicate.flat[0] = 99
    with pytest.raises(FrozenInstanceError):
        epoch.readout_valid = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "args",
    [
        (0, 0, np.eye(4), _submap(), True),
        (1, -1, np.eye(4), _submap(), True),
        (1, 0, np.eye(4), object(), True),
        (1, 0, np.eye(4), _submap(), np.bool_(True)),
    ],
)
def test_epoch_rejects_invalid_contract(args: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        GeometryEpoch(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "last_frame,error_type",
    [
        (True, TypeError),
        (-1, ValueError),
        (np.iinfo(np.int64).max + 1, ValueError),
        (0, ValueError),
    ],
)
def test_epoch_rejects_invalid_last_processed_frame(
    last_frame: object, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type, match="last_processed_frame_id"):
        GeometryEpoch(
            7,
            3,
            np.eye(4),
            _submap(),
            True,
            last_processed_frame_id=last_frame,  # type: ignore[arg-type]
        )


def test_accepted_estimate_integrates_in_same_epoch_without_mutating_inputs() -> None:
    epoch = _epoch()
    points = np.asarray([[0.3, 0.0, 0.0]])
    original = points.copy()

    updated = epoch.integrate(
        _estimate(MotionDecision.TRANSLATION_ACCEPTED), points, frame_id=2, config=_config()
    )

    assert updated.entity_id == epoch.entity_id and updated.epoch_id == epoch.epoch_id
    assert updated.motion_decision is MotionDecision.TRANSLATION_ACCEPTED
    assert updated.last_processed_frame_id == 2
    assert updated.readout_valid is True
    assert updated.submap.weights[0] == 2.0
    np.testing.assert_array_equal(points, original)


def test_invalid_epoch_empty_accepted_integration_does_not_revive_readout() -> None:
    epoch = _epoch(readout_valid=False)

    updated = epoch.integrate(
        _estimate(MotionDecision.TRANSLATION_ACCEPTED),
        np.empty((0, 3)),
        frame_id=2,
        config=_config(),
    )

    assert updated.readout_valid is False
    assert updated.submap == epoch.submap
    assert updated.last_processed_frame_id == 2


def test_invalid_epoch_nonempty_accepted_integration_revives_readout() -> None:
    updated = _epoch(readout_valid=False).integrate(
        _estimate(MotionDecision.TRANSLATION_ACCEPTED),
        np.asarray([[0.3, 0.0, 0.0]]),
        frame_id=2,
        config=_config(),
    )

    assert updated.readout_valid is True


def test_invalid_epoch_does_not_revive_when_capacity_discards_observation() -> None:
    config = _config(maximum_object_voxels=1)
    established = _epoch().integrate(
        _estimate(), np.asarray([[0.3, 0.0, 0.0]]), frame_id=2, config=config
    )
    invalid = established.apply_evidence(TemporalEvidenceKind.VISIBLE_ABSENT)

    unchanged = invalid.integrate(
        _estimate(), np.asarray([[1.3, 0.0, 0.0]]), frame_id=3, config=config
    )

    assert unchanged.submap == invalid.submap
    assert unchanged.readout_valid is False
    assert unchanged.last_processed_frame_id == 3
    pose_before = unchanged.object_to_world.tobytes()
    with pytest.raises(ValueError, match="frame_id"):
        unchanged.integrate(
            _estimate(x=2.0),
            np.asarray([[0.3, 0.0, 0.0]]),
            frame_id=3,
            config=config,
        )
    with pytest.raises(ValueError, match="frame_id"):
        unchanged.integrate(
            _estimate(x=2.0),
            np.asarray([[0.3, 0.0, 0.0]]),
            frame_id=2,
            config=config,
        )
    assert unchanged.object_to_world.tobytes() == pose_before


def test_rejected_motion_cannot_integrate_into_existing_epoch() -> None:
    epoch = _epoch()
    before = epoch.submap.local_points_xyz.tobytes(), epoch.submap.weights.tobytes()
    with pytest.raises(ValueError, match="rejected motion"):
        epoch.integrate(
            _estimate(MotionDecision.REJECTED, 2.0),
            np.asarray([[2.0, 0.0, 0.0]]),
            frame_id=8,
        )
    assert before == (epoch.submap.local_points_xyz.tobytes(), epoch.submap.weights.tobytes())


def test_epoch_rejects_integration_for_different_entity() -> None:
    with pytest.raises(ValueError, match="entity_id"):
        _epoch().integrate(
            _estimate(), np.asarray([[0.2, 0.0, 0.0]]), frame_id=2, config=_config(), entity_id=8
        )


@pytest.mark.parametrize(
    "evidence,expected",
    [
        (TemporalEvidenceKind.VISIBLE_ABSENT, False),
        (TemporalEvidenceKind.OCCLUDED, True),
        (TemporalEvidenceKind.OUT_OF_VIEW, True),
        (TemporalEvidenceKind.DEPTH_UNKNOWN, True),
        (TemporalEvidenceKind.PRESENT, False),
    ],
)
def test_evidence_updates_readout_fail_closed(
    evidence: TemporalEvidenceKind, expected: bool
) -> None:
    epoch = _epoch(readout_valid=evidence is not TemporalEvidenceKind.PRESENT)
    assert epoch.apply_evidence(evidence).readout_valid is expected


def test_neutral_evidence_does_not_revive_invalid_epoch_and_unknown_is_rejected() -> None:
    invalid = _epoch(readout_valid=False)
    for evidence in (
        TemporalEvidenceKind.OCCLUDED,
        TemporalEvidenceKind.OUT_OF_VIEW,
        TemporalEvidenceKind.DEPTH_UNKNOWN,
    ):
        assert invalid.apply_evidence(evidence).readout_valid is False
    with pytest.raises(TypeError, match="evidence"):
        invalid.apply_evidence("occluded")  # type: ignore[arg-type]


def test_visible_absent_then_present_does_not_revive_without_geometry_update() -> None:
    invalid = _epoch().apply_evidence(TemporalEvidenceKind.VISIBLE_ABSENT)

    present = invalid.apply_evidence(TemporalEvidenceKind.PRESENT)

    assert present.readout_valid is False
    assert present.last_processed_frame_id == invalid.last_processed_frame_id


def test_new_epoch_retains_identity_and_uses_only_current_observation() -> None:
    old = _epoch(epoch_id=3)
    points = np.asarray([[10.1, 0.0, 0.0], [11.1, 0.0, 0.0]])
    new = start_new_epoch(
        old,
        _estimate(MotionDecision.REJECTED, 999.0),
        points,
        observed_centroid_xyz=(10.0, 0.0, 0.0),
        frame_id=8,
        config=_config(),
    )

    assert new.entity_id == old.entity_id and new.epoch_id == 4
    assert new.motion_decision is MotionDecision.REJECTED
    assert new.last_processed_frame_id == 8
    np.testing.assert_array_equal(new.object_to_world[:3, 3], [10.0, 0.0, 0.0])
    assert new.submap.local_voxel_keys == ((0, 0, 0), (1, 0, 0))
    np.testing.assert_array_equal(new.submap.weights, [1.0, 1.0])
    np.testing.assert_array_equal(new.submap.last_seen_frame_ids, [8, 8])
    assert new.readout_valid is True


def test_new_epoch_accepts_explicit_pose_and_rejects_entity_or_epoch_overflow() -> None:
    explicit = np.eye(4)
    explicit[1, 3] = 5.0
    new = start_new_epoch(
        _epoch(),
        _estimate(MotionDecision.REJECTED, 999.0),
        np.asarray([[0.0, 5.1, 0.0]]),
        observed_centroid_xyz=(999.0, 0.0, 0.0),
        initial_object_to_world=explicit,
        frame_id=8,
        config=_config(),
    )
    np.testing.assert_array_equal(new.object_to_world, explicit)
    with pytest.raises(ValueError, match="entity_id"):
        start_new_epoch(
            _epoch(), _estimate(), np.ones((1, 3)), (0.0, 0.0, 0.0), 8, _config(), entity_id=8
        )
    with pytest.raises(OverflowError, match="epoch_id"):
        start_new_epoch(
            _epoch(epoch_id=np.iinfo(np.int64).max),
            _estimate(),
            np.ones((1, 3)),
            (0.0, 0.0, 0.0),
            8,
            _config(),
        )


def test_new_epoch_frame_fallback_capacity_and_determinism() -> None:
    old = _epoch()
    points = np.asarray([[2.1, 0.0, 0.0], [0.1, 0.0, 0.0], [1.1, 0.0, 0.0]])
    results = [
        start_new_epoch(
            old,
            _estimate(MotionDecision.REJECTED),
            points[list(order)],
            (0.0, 0.0, 0.0),
            0,
            _config(maximum_object_voxels=2),
        )
        for order in itertools.permutations(range(3))
    ]
    assert all(result.submap.last_seen_frame_ids.tolist() == [0, 0] for result in results)
    for result in results[1:]:
        assert result.submap.local_voxel_keys == results[0].submap.local_voxel_keys
        assert (
            result.submap.local_points_xyz.tobytes()
            == results[0].submap.local_points_xyz.tobytes()
        )
        assert result.submap.weights.tobytes() == results[0].submap.weights.tobytes()
