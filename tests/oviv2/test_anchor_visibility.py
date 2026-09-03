from __future__ import annotations

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.anchor_visibility import (
    LOCALIZED_POLICY_ID,
    AnchorCurrentOwnership,
    AnchorVisibilityConfig,
    AnchorVisibilityEvidence,
    AnchorVisibilityEvidenceKind,
    AnchorVisibilityState,
    AnchorVoxelVisibilityEvidence,
    AnchorVoxelVisibilityEvidenceKind,
    advance_anchor_current_ownership,
    advance_anchor_visibility,
    classify_anchor_voxel_visibility,
    classify_anchor_visibility,
    initialize_anchor_current_ownership,
    pack_anchor_current_mask,
    sample_anchor_voxels,
)


def _config(**overrides: object) -> AnchorVisibilityConfig:
    values = {
        "voxel_size_m": 1.0,
        "depth_tolerance_m": 0.1,
        "depth_max_m": 10.0,
        "maximum_voxels_per_anchor": 1_000,
        "minimum_tested_voxels": 1,
        "minimum_absent_fraction": 0.8,
        "minimum_present_fraction": 0.8,
        "minimum_absent_observations": 6,
        "minimum_distinct_viewpoints": 3,
        "minimum_viewpoint_baseline_m": 0.25,
        "minimum_present_streak": 2,
    }
    values.update(overrides)
    return AnchorVisibilityConfig(**values)


def _frame(frame_id: int, depth: float, *, camera_x: float = 0.0) -> Frame:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = camera_x
    return Frame(
        frame_id=frame_id,
        source_frame_id=frame_id,
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth=np.full((5, 5), depth, dtype=np.float32),
        pose=pose,
        intrinsics=CameraIntrinsics(3.0, 3.0, 2.0, 2.0, 5, 5),
        timestamp=float(frame_id),
    )


def _state(*, active: bool = True) -> AnchorVisibilityState:
    return AnchorVisibilityState(
        entity_id="ovimap:5",
        active=active,
        last_frame_id=0,
        last_timestamp=0.0,
        absence_observation_count=0,
        absence_viewpoints_xyz=(),
        present_streak=0,
    )


def _evidence(
    frame_id: int,
    kind: AnchorVisibilityEvidenceKind,
    *,
    camera_x: float = 0.0,
) -> AnchorVisibilityEvidence:
    return AnchorVisibilityEvidence(
        kind=kind,
        frame_id=frame_id,
        timestamp=float(frame_id),
        tested_voxel_count=10 if kind is not AnchorVisibilityEvidenceKind.DEPTH_UNKNOWN else 0,
        present_voxel_count=10 if kind is AnchorVisibilityEvidenceKind.PRESENT else 0,
        absent_voxel_count=10 if kind is AnchorVisibilityEvidenceKind.VISIBLE_ABSENT else 0,
        occluded_voxel_count=10 if kind is AnchorVisibilityEvidenceKind.OCCLUDED else 0,
        camera_position_xyz=(camera_x, 0.0, 0.0),
    )


def test_anchor_voxel_sampling_is_sorted_bounded_and_deterministic() -> None:
    points = np.column_stack(
        (
            np.arange(2_001, dtype=np.float64),
            np.zeros(2_001),
            np.ones(2_001),
        )
    )

    first = sample_anchor_voxels(points, _config())
    second = sample_anchor_voxels(points[::-1], _config())

    assert first == second
    assert len(first) == 1_000
    assert first == tuple(sorted(set(first)))
    assert first[0] == (0, 0, 1)
    assert first[-1] == (2_000, 0, 1)


@pytest.mark.parametrize(
    ("depth", "expected"),
    (
        (2.0, AnchorVisibilityEvidenceKind.VISIBLE_ABSENT),
        (1.5, AnchorVisibilityEvidenceKind.PRESENT),
        (1.0, AnchorVisibilityEvidenceKind.OCCLUDED),
        (0.0, AnchorVisibilityEvidenceKind.DEPTH_UNKNOWN),
    ),
)
def test_visibility_evidence_distinguishes_free_space_from_occlusion(
    depth: float,
    expected: AnchorVisibilityEvidenceKind,
) -> None:
    evidence = classify_anchor_visibility(
        ((0, 0, 1),),
        _frame(1, depth),
        _config(),
    )

    assert evidence.kind is expected


def test_mixed_surface_and_free_space_is_neutral() -> None:
    altered = _frame(1, 2.0)
    altered.depth[3, 3] = 1.5
    evidence = classify_anchor_visibility(
        ((-1, 0, 1), (0, 0, 1)),
        altered,
        _config(minimum_tested_voxels=2),
    )

    assert evidence.kind is AnchorVisibilityEvidenceKind.OCCLUDED


def test_six_absences_from_three_separated_viewpoints_make_anchor_dormant() -> None:
    state = _state()
    cameras = (0.0, 0.0, 0.3, 0.3, 0.6, 0.6)

    for frame_id, camera_x in enumerate(cameras, start=1):
        state = advance_anchor_visibility(
            state,
            _evidence(
                frame_id,
                AnchorVisibilityEvidenceKind.VISIBLE_ABSENT,
                camera_x=camera_x,
            ),
            _config(),
        )

    assert state.active is False
    assert state.absence_observation_count == 6
    assert state.absence_viewpoints_xyz == (
        (0.0, 0.0, 0.0),
        (0.3, 0.0, 0.0),
        (0.6, 0.0, 0.0),
    )


def test_occluded_and_unknown_frames_never_suppress_or_erase_support() -> None:
    state = _state()
    state = advance_anchor_visibility(
        state,
        _evidence(1, AnchorVisibilityEvidenceKind.VISIBLE_ABSENT),
        _config(),
    )
    for frame_id, kind in (
        (2, AnchorVisibilityEvidenceKind.OCCLUDED),
        (3, AnchorVisibilityEvidenceKind.DEPTH_UNKNOWN),
    ):
        state = advance_anchor_visibility(
            state,
            _evidence(frame_id, kind),
            _config(),
        )

    assert state.active is True
    assert state.absence_observation_count == 1
    assert state.absence_viewpoints_xyz == ((0.0, 0.0, 0.0),)


def test_strong_presence_clears_pending_absence_support() -> None:
    state = _state()
    for frame_id, camera_x in ((1, 0.0), (2, 0.3), (3, 0.6)):
        state = advance_anchor_visibility(
            state,
            _evidence(
                frame_id,
                AnchorVisibilityEvidenceKind.VISIBLE_ABSENT,
                camera_x=camera_x,
            ),
            _config(),
        )

    state = advance_anchor_visibility(
        state,
        _evidence(4, AnchorVisibilityEvidenceKind.PRESENT),
        _config(),
    )

    assert state.active is True
    assert state.absence_observation_count == 0
    assert state.absence_viewpoints_xyz == ()


def test_dormant_anchor_reactivates_after_two_consecutive_present_frames() -> None:
    state = _state(active=False)

    first = advance_anchor_visibility(
        state,
        _evidence(1, AnchorVisibilityEvidenceKind.PRESENT),
        _config(),
    )
    second = advance_anchor_visibility(
        first,
        _evidence(2, AnchorVisibilityEvidenceKind.PRESENT),
        _config(),
    )

    assert first.active is False
    assert first.present_streak == 1
    assert second.active is True
    assert second.present_streak == 0


def test_visibility_evidence_must_advance_strictly() -> None:
    with pytest.raises(ValueError, match="increase strictly"):
        advance_anchor_visibility(
            _state(),
            _evidence(0, AnchorVisibilityEvidenceKind.DEPTH_UNKNOWN),
            _config(),
        )


def _ownership_points() -> np.ndarray:
    return np.asarray(
        (
            (0.1, 0.1, 1.1),
            (1.1, 0.1, 1.1),
            (2.1, 0.1, 1.1),
            (0.2, 0.2, 1.2),
        ),
        dtype=np.float64,
    )


def _voxel_evidence(
    state: AnchorCurrentOwnership,
    frame_id: int,
    statuses: tuple[AnchorVoxelVisibilityEvidenceKind, ...],
    *,
    camera_x: float = 0.0,
) -> AnchorVoxelVisibilityEvidence:
    return AnchorVoxelVisibilityEvidence(
        entity_id=state.entity_id,
        frame_id=frame_id,
        timestamp=float(frame_id),
        voxel_indices=state.evidence_sample_indices,
        status_codes=np.asarray([item.value for item in statuses], dtype=np.uint8),
        camera_position_xyz=(camera_x, 0.0, 0.0),
    )


def test_localized_ownership_initializes_full_state_and_bounded_sample() -> None:
    points = _ownership_points()
    original = points.copy()
    config = _config(maximum_voxels_per_anchor=2)

    first = initialize_anchor_current_ownership(
        "ovimap:5", points, config, frame_id=0, timestamp=0.0
    )
    second = initialize_anchor_current_ownership(
        "ovimap:5", points[::-1], config, frame_id=0, timestamp=0.0
    )

    assert first.policy_id == LOCALIZED_POLICY_ID
    assert first.voxel_size_m == 1.0
    assert first.voxel_keys.tolist() == [[0, 0, 1], [1, 0, 1], [2, 0, 1]]
    assert first.evidence_sample_indices.tolist() == [0, 2]
    assert np.array_equal(first.voxel_keys, second.voxel_keys)
    assert np.array_equal(
        first.evidence_sample_indices, second.evidence_sample_indices
    )
    assert first.whole_anchor_status == "unchanged"
    assert first.current_voxel_count == 3
    assert first.suppressed_voxel_count == 0
    assert np.array_equal(points, original)
    for array in (
        first.voxel_keys,
        first.evidence_sample_indices,
        first.current_mask,
        first.absence_observation_counts,
        first.absence_viewpoints_xyz,
        first.absence_viewpoint_counts,
        first.present_streaks,
        first.first_absence_frames,
    ):
        assert array.flags.writeable is False
    with pytest.raises(ValueError):
        first.current_mask[0] = False
    assert first.state_byte_count == sum(
        array.nbytes
        for array in (
            first.voxel_keys,
            first.evidence_sample_indices,
            first.current_mask,
            first.absence_observation_counts,
            first.absence_viewpoints_xyz,
            first.absence_viewpoint_counts,
            first.present_streaks,
            first.first_absence_frames,
        )
    )


def test_localized_absence_suppresses_only_supported_voxel_and_is_reversible() -> None:
    config = _config(maximum_voxels_per_anchor=3)
    state = initialize_anchor_current_ownership(
        "ovimap:5", _ownership_points(), config, frame_id=0, timestamp=0.0
    )
    cameras = (0.0, 0.0, 0.3, 0.3, 0.6, 0.6)
    for frame_id, camera_x in enumerate(cameras, start=1):
        state = advance_anchor_current_ownership(
            state,
            _voxel_evidence(
                state,
                frame_id,
                (
                    AnchorVoxelVisibilityEvidenceKind.VISIBLE_ABSENT,
                    AnchorVoxelVisibilityEvidenceKind.OCCLUDED,
                    AnchorVoxelVisibilityEvidenceKind.UNOBSERVED,
                ),
                camera_x=camera_x,
            ),
            config,
        )

    assert state.current_mask.tolist() == [False, True, True]
    assert state.whole_anchor_status == "partially_suppressed"
    assert state.absence_observation_counts.tolist() == [6, 0, 0]
    assert state.absence_viewpoint_counts.tolist() == [3, 0, 0]
    assert state.first_absence_frames.tolist() == [1, -1, -1]
    packed = pack_anchor_current_mask(state)
    assert packed == bytes((0b00000110,))

    first = advance_anchor_current_ownership(
        state,
        _voxel_evidence(
            state,
            7,
            (
                AnchorVoxelVisibilityEvidenceKind.PRESENT,
                AnchorVoxelVisibilityEvidenceKind.UNOBSERVED,
                AnchorVoxelVisibilityEvidenceKind.UNOBSERVED,
            ),
        ),
        config,
    )
    restored = advance_anchor_current_ownership(
        first,
        _voxel_evidence(
            first,
            8,
            (
                AnchorVoxelVisibilityEvidenceKind.PRESENT,
                AnchorVoxelVisibilityEvidenceKind.UNOBSERVED,
                AnchorVoxelVisibilityEvidenceKind.UNOBSERVED,
            ),
        ),
        config,
    )
    assert first.current_mask.tolist() == [False, True, True]
    assert first.present_streaks.tolist() == [1, 0, 0]
    assert restored.current_mask.tolist() == [True, True, True]
    assert restored.whole_anchor_status == "unchanged"
    assert restored.absence_observation_counts.tolist() == [0, 0, 0]
    assert restored.absence_viewpoint_counts.tolist() == [0, 0, 0]
    assert restored.present_streaks.tolist() == [0, 0, 0]
    assert restored.first_absence_frames.tolist() == [-1, -1, -1]


def test_localized_occluded_unobserved_and_invalid_depth_are_neutral() -> None:
    config = _config(maximum_voxels_per_anchor=3)
    state = initialize_anchor_current_ownership(
        "ovimap:5", _ownership_points(), config, frame_id=0, timestamp=0.0
    )
    state = advance_anchor_current_ownership(
        state,
        _voxel_evidence(
            state,
            1,
            (
                AnchorVoxelVisibilityEvidenceKind.VISIBLE_ABSENT,
                AnchorVoxelVisibilityEvidenceKind.OCCLUDED,
                AnchorVoxelVisibilityEvidenceKind.UNOBSERVED,
            ),
        ),
        config,
    )
    preserved = advance_anchor_current_ownership(
        state,
        _voxel_evidence(
            state,
            2,
            (
                AnchorVoxelVisibilityEvidenceKind.OCCLUDED,
                AnchorVoxelVisibilityEvidenceKind.UNOBSERVED,
                AnchorVoxelVisibilityEvidenceKind.UNOBSERVED,
            ),
        ),
        config,
    )
    assert preserved.absence_observation_counts.tolist() == [1, 0, 0]
    assert preserved.first_absence_frames.tolist() == [1, -1, -1]

    invalid = _frame(3, float("nan"))
    classified = classify_anchor_voxel_visibility(preserved, invalid, config)
    assert classified.status_codes.tolist() == [
        AnchorVoxelVisibilityEvidenceKind.UNOBSERVED.value
    ] * 3
    after_invalid = advance_anchor_current_ownership(
        preserved, classified, config
    )
    assert after_invalid.absence_observation_counts.tolist() == [1, 0, 0]


def test_localized_present_clears_pending_absence_and_time_must_advance() -> None:
    config = _config(maximum_voxels_per_anchor=3)
    state = initialize_anchor_current_ownership(
        "ovimap:5", _ownership_points(), config, frame_id=0, timestamp=0.0
    )
    state = advance_anchor_current_ownership(
        state,
        _voxel_evidence(
            state,
            1,
            (
                AnchorVoxelVisibilityEvidenceKind.VISIBLE_ABSENT,
                AnchorVoxelVisibilityEvidenceKind.UNOBSERVED,
                AnchorVoxelVisibilityEvidenceKind.UNOBSERVED,
            ),
        ),
        config,
    )
    cleared = advance_anchor_current_ownership(
        state,
        _voxel_evidence(
            state,
            2,
            (
                AnchorVoxelVisibilityEvidenceKind.PRESENT,
                AnchorVoxelVisibilityEvidenceKind.UNOBSERVED,
                AnchorVoxelVisibilityEvidenceKind.UNOBSERVED,
            ),
        ),
        config,
    )
    assert cleared.absence_observation_counts.tolist() == [0, 0, 0]
    assert cleared.first_absence_frames.tolist() == [-1, -1, -1]
    with pytest.raises(ValueError, match="increase strictly"):
        advance_anchor_current_ownership(
            cleared,
            AnchorVoxelVisibilityEvidence(
                entity_id=cleared.entity_id,
                frame_id=2,
                timestamp=3.0,
                voxel_indices=cleared.evidence_sample_indices,
                status_codes=np.full(
                    len(cleared.evidence_sample_indices),
                    AnchorVoxelVisibilityEvidenceKind.UNOBSERVED.value,
                    dtype=np.uint8,
                ),
                camera_position_xyz=(0.0, 0.0, 0.0),
            ),
            config,
        )


def test_localized_status_is_dormant_only_when_every_voxel_is_suppressed() -> None:
    config = _config(maximum_voxels_per_anchor=3)
    state = initialize_anchor_current_ownership(
        "ovimap:5", _ownership_points(), config, frame_id=0, timestamp=0.0
    )
    absent = (AnchorVoxelVisibilityEvidenceKind.VISIBLE_ABSENT,) * 3
    for frame_id, camera_x in enumerate((0.0, 0.0, 0.3, 0.3, 0.6, 0.6), start=1):
        previous = state
        state = advance_anchor_current_ownership(
            state,
            _voxel_evidence(state, frame_id, absent, camera_x=camera_x),
            config,
        )
        assert previous.last_frame_id == frame_id - 1
        assert previous.current_mask.flags.writeable is False
    assert state.current_mask.tolist() == [False, False, False]
    assert state.whole_anchor_status == "dormant"
    assert state.current_voxel_count == 0
    assert state.suppressed_voxel_count == 3
