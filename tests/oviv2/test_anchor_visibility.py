from __future__ import annotations

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.anchor_visibility import (
    AnchorVisibilityConfig,
    AnchorVisibilityEvidence,
    AnchorVisibilityEvidenceKind,
    AnchorVisibilityState,
    advance_anchor_visibility,
    classify_anchor_visibility,
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
