from __future__ import annotations

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.fine_surface_validity import (
    CurrentEvidenceState,
    FineEvidenceProjectionConfig,
    FineSurfaceEvidence,
    FineValidityConfig,
    observe_fine_surface,
    resolve_fine_current_validity,
)


def _projection_frame(
    frame_id: int,
    *,
    camera_x: float,
    present_rgb: tuple[int, int, int],
) -> Frame:
    depth = np.zeros((5, 5), dtype=np.float32)
    present_column = int(np.rint(2.0 * (0.0 - camera_x) + 2.0))
    absent_column = int(np.rint(2.0 * (1.0 - camera_x) + 2.0))
    occluded_column = int(np.rint(2.0 * (-1.0 - camera_x) / 2.0 + 2.0))
    depth[2, present_column] = 1.0
    depth[2, absent_column] = 2.0
    depth[2, occluded_column] = 1.0
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    rgb[depth == 1.0] = present_rgb
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = camera_x
    return Frame(
        frame_id=frame_id,
        source_frame_id=frame_id,
        rgb=rgb,
        depth=depth,
        pose=pose,
        intrinsics=CameraIntrinsics(
            fx=2.0,
            fy=2.0,
            cx=2.0,
            cy=2.0,
            width=5,
            height=5,
        ),
        timestamp=float(frame_id),
    )


def test_coarse_candidate_needs_local_multiview_free_space_evidence() -> None:
    evidence = FineSurfaceEvidence(
        present_observations=np.asarray([0, 0, 0, 0, 1, 0], dtype=np.uint16),
        visible_absent_observations=np.asarray(
            [2, 2, 0, 2, 0, 0], dtype=np.uint16
        ),
        occluded_observations=np.asarray([0, 0, 1, 0, 0, 0], dtype=np.uint16),
        distinct_absent_viewpoints=np.asarray([2, 1, 0, 2, 0, 0], dtype=np.uint8),
        last_supported_frames=np.asarray([7, 7, 4, 7, 9, 10], dtype=np.int32),
    )

    result = resolve_fine_current_validity(
        source_visit_ids=np.asarray([0, 0, 0, 0, 0, 1], dtype=np.int16),
        latest_visit_id=1,
        coarse_visible_free_candidates=np.asarray(
            [True, True, True, False, True, False]
        ),
        evidence=evidence,
        config=FineValidityConfig(
            minimum_absent_observations=2,
            minimum_distinct_viewpoints=2,
        ),
    )

    assert result.current_valid.tolist() == [False, True, True, True, False, True]
    assert result.evidence_state_codes.tolist() == [
        CurrentEvidenceState.REVOKED_VISIBLE_FREE,
        CurrentEvidenceState.HISTORICAL_UNCERTAIN,
        CurrentEvidenceState.HISTORICAL_OCCLUDED,
        CurrentEvidenceState.HISTORICAL_UNCERTAIN,
        CurrentEvidenceState.REPLACED_BY_CURRENT,
        CurrentEvidenceState.CURRENT_OBSERVED,
    ]


def test_occlusion_and_unobserved_evidence_never_revoke() -> None:
    evidence = FineSurfaceEvidence(
        present_observations=np.zeros(2, dtype=np.uint16),
        visible_absent_observations=np.zeros(2, dtype=np.uint16),
        occluded_observations=np.asarray([9, 0], dtype=np.uint16),
        distinct_absent_viewpoints=np.zeros(2, dtype=np.uint8),
        last_supported_frames=np.asarray([20, -1], dtype=np.int32),
    )
    result = resolve_fine_current_validity(
        source_visit_ids=np.zeros(2, dtype=np.int16),
        latest_visit_id=1,
        coarse_visible_free_candidates=np.ones(2, dtype=bool),
        evidence=evidence,
        config=FineValidityConfig(),
    )

    assert result.current_valid.tolist() == [True, True]
    assert result.evidence_state_codes.tolist() == [
        CurrentEvidenceState.HISTORICAL_OCCLUDED,
        CurrentEvidenceState.HISTORICAL_UNOBSERVED,
    ]


def test_validity_rejects_future_source_visit() -> None:
    evidence = FineSurfaceEvidence.empty(1)
    with pytest.raises(ValueError, match="latest visit"):
        resolve_fine_current_validity(
            source_visit_ids=np.asarray([2], dtype=np.int16),
            latest_visit_id=1,
            coarse_visible_free_candidates=np.asarray([False]),
            evidence=evidence,
            config=FineValidityConfig(),
        )


def test_fine_projection_accumulates_signed_evidence_and_camera_rgb() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 2.0],
            [20.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    frames = (
        _projection_frame(7, camera_x=0.0, present_rgb=(11, 22, 33)),
        _projection_frame(9, camera_x=0.5, present_rgb=(44, 55, 66)),
    )

    observation = observe_fine_surface(
        points,
        frames,
        FineEvidenceProjectionConfig(
            depth_tolerance_m=0.05,
            depth_max_m=5.0,
            minimum_viewpoint_baseline_m=0.25,
            maximum_distinct_viewpoints=2,
            minimum_rgb_neighbours=1,
        ),
        point_batch_size=2,
    )

    evidence = observation.evidence
    assert evidence.present_observations.tolist() == [2, 0, 0, 0]
    assert evidence.visible_absent_observations.tolist() == [0, 2, 0, 0]
    assert evidence.occluded_observations.tolist() == [0, 0, 2, 0]
    assert evidence.distinct_absent_viewpoints.tolist() == [0, 2, 0, 0]
    assert evidence.last_supported_frames.tolist() == [9, -1, -1, -1]
    assert evidence.last_absent_frames.tolist() == [-1, 9, -1, -1]
    assert evidence.last_occluded_frames.tolist() == [-1, -1, 9, -1]
    assert observation.rgb_valid.tolist() == [True, False, False, False]
    assert observation.observed_rgb_uint8.tolist() == [
        [11, 22, 33],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert np.allclose(observation.best_rgb_depth_residual_m[0], 0.0)
    assert np.isinf(observation.best_rgb_depth_residual_m[1:]).all()


def test_omitted_evidence_timestamps_default_to_immutable_unknown() -> None:
    evidence = FineSurfaceEvidence(
        present_observations=np.asarray([1, 0], dtype=np.uint16),
        visible_absent_observations=np.asarray([0, 1], dtype=np.uint16),
        occluded_observations=np.asarray([0, 0], dtype=np.uint16),
        distinct_absent_viewpoints=np.asarray([0, 1], dtype=np.uint8),
        last_supported_frames=np.asarray([7, -1], dtype=np.int32),
    )

    assert evidence.last_absent_frames.tolist() == [-1, -1]
    assert evidence.last_occluded_frames.tolist() == [-1, -1]
    assert not evidence.last_absent_frames.flags.writeable
    assert not evidence.last_occluded_frames.flags.writeable


def test_fine_projection_requires_geometrically_distinct_absent_views() -> None:
    point = np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32)
    frames = (
        _projection_frame(0, camera_x=0.0, present_rgb=(0, 0, 0)),
        _projection_frame(1, camera_x=0.1, present_rgb=(0, 0, 0)),
    )

    observation = observe_fine_surface(
        point,
        frames,
        FineEvidenceProjectionConfig(
            minimum_viewpoint_baseline_m=0.25,
            maximum_distinct_viewpoints=2,
            minimum_rgb_neighbours=1,
        ),
    )

    assert observation.evidence.visible_absent_observations.tolist() == [2]
    assert observation.evidence.distinct_absent_viewpoints.tolist() == [1]


def test_fine_projection_is_batch_size_invariant() -> None:
    points = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    frames = (
        _projection_frame(0, camera_x=0.0, present_rgb=(11, 22, 33)),
        _projection_frame(1, camera_x=0.5, present_rgb=(44, 55, 66)),
    )
    config = FineEvidenceProjectionConfig(
        minimum_viewpoint_baseline_m=0.25,
        maximum_distinct_viewpoints=2,
        minimum_rgb_neighbours=1,
    )

    scalar = observe_fine_surface(points, frames, config, point_batch_size=1)
    vector = observe_fine_surface(points, frames, config, point_batch_size=99)

    for name in (
        "present_observations",
        "visible_absent_observations",
        "occluded_observations",
        "distinct_absent_viewpoints",
        "last_supported_frames",
    ):
        np.testing.assert_array_equal(
            getattr(scalar.evidence, name), getattr(vector.evidence, name)
        )
    np.testing.assert_array_equal(scalar.observed_rgb_uint8, vector.observed_rgb_uint8)
    np.testing.assert_array_equal(scalar.rgb_valid, vector.rgb_valid)
    np.testing.assert_array_equal(
        scalar.best_rgb_depth_residual_m, vector.best_rgb_depth_residual_m
    )
