from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import warnings

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.geometry import SparseTsdfVolume
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.temporal_background import (
    BackgroundMaskResult,
    TemporalBackgroundVolume,
    build_background_depth,
)
from src.oviv2.temporal_config import TemporalGeometryConfig


def _config(**changes: object) -> TemporalGeometryConfig:
    values = {
        "voxel_size_m": 0.05,
        "depth_max_m": 4.0,
        "maximum_entities": 10,
        "maximum_object_voxels": 10,
        "maximum_visibility_points_per_entity": 10,
        "background_block_count": 100,
        "background_mask_dilation_px": 0,
        "minimum_icp_points": 3,
        "minimum_icp_fitness": 0.7,
        "maximum_icp_rmse_m": 0.2,
        "maximum_motion_m": 2.0,
    }
    values.update(changes)
    return TemporalGeometryConfig(**values)


def _frame(frame_id: int = 7, *, size: int = 5, depth: float = 1.0) -> Frame:
    return Frame(
        frame_id=frame_id,
        timestamp=1.5 + frame_id,
        rgb=np.zeros((size, size, 3), dtype=np.uint8),
        depth=np.full((size, size), depth, dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(2.0, 2.0, 2.0, 2.0, size, size),
    )


def _observation(
    frame: Frame,
    observation_id: int,
    kind: ObservationKind,
    mask: np.ndarray,
) -> FrameObservation:
    return FrameObservation(
        observation_id=observation_id,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        kind=kind,
        label=kind.value,
        semantic_id=observation_id,
        confidence=1.0,
        mask=mask,
        bbox_xyxy=(0.0, 0.0, float(mask.shape[1]), float(mask.shape[0])),
        voxel_keys=frozenset({(observation_id, 0, 0)}),
        centroid_xyz=(0.0, 0.0, 1.0),
        bounds_min_xyz=(0.0, 0.0, 1.0),
        bounds_max_xyz=(0.0, 0.0, 1.0),
    )


def test_masks_object_and_unknown_union_but_preserves_structure() -> None:
    frame = _frame()
    object_mask = np.zeros((5, 5), dtype=bool)
    unknown_mask = object_mask.copy()
    structure_mask = object_mask.copy()
    object_mask[0, 0] = True
    unknown_mask[1, 1] = True
    structure_mask[2, 2] = True

    result = build_background_depth(
        frame,
        (
            _observation(frame, 1, ObservationKind.OBJECT, object_mask),
            _observation(frame, 2, ObservationKind.UNKNOWN, unknown_mask),
            _observation(frame, 3, ObservationKind.STRUCTURE, structure_mask),
        ),
        (),
        _config(),
    )

    assert result.excluded_pixel_count == 2
    assert result.valid_background_pixel_count == 23
    assert result.depth_m[0, 0] == result.depth_m[1, 1] == 0.0
    assert result.depth_m[2, 2] == 1.0


def test_projected_points_use_inverse_pose_and_nearest_pixel() -> None:
    frame = _frame()
    frame.pose[0, 3] = 10.0
    points = np.asarray(
        [
            [10.24, 0.24, 1.0],  # u=v=2.48 -> (2, 2)
            [10.26, 0.26, 1.0],  # u=v=2.52 -> (3, 3)
            [10.0, 0.0, -1.0],
            [1e4, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    result = build_background_depth(frame, (), (points,), _config())

    assert result.excluded_pixel_count == 2
    assert result.depth_m[2, 2] == result.depth_m[3, 3] == 0.0
    assert result.depth_m[2, 3] == 1.0


def test_projection_ignores_finite_points_that_overflow_image_coordinates() -> None:
    frame = _frame()
    points = np.asarray([[1e308, 0.0, 1e-308]], dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = build_background_depth(frame, (), (points,), _config())
    np.testing.assert_array_equal(result.depth_m, frame.depth)


def test_empty_protected_tuple_does_not_protect_old_location() -> None:
    frame = _frame()
    result = build_background_depth(frame, (), (), _config())
    np.testing.assert_array_equal(result.depth_m, frame.depth)
    assert result.excluded_pixel_count == 0


def test_dilation_is_exact_chebyshev_radius_and_clipped_at_border() -> None:
    frame = _frame()
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    mask[3, 3] = True
    result = build_background_depth(
        frame,
        (_observation(frame, 1, ObservationKind.OBJECT, mask),),
        (),
        _config(background_mask_dilation_px=1),
    )
    expected = np.ones((5, 5), dtype=np.float32)
    expected[0:2, 0:2] = 0.0
    expected[2:5, 2:5] = 0.0
    np.testing.assert_array_equal(result.depth_m, expected)
    assert result.excluded_pixel_count == 13


def test_invalid_depth_is_zeroed_while_valid_bits_and_inputs_are_preserved() -> None:
    frame = _frame()
    frame.depth = np.asarray(
        [[np.nan, np.inf, -1.0, 0.0, 5.0], [1.1, 1.2, 1.3, 1.4, 1.5], *np.ones((3, 5))],
        dtype=np.float32,
    )
    original = frame.depth.copy()
    result = build_background_depth(frame, (), (), _config())
    np.testing.assert_array_equal(result.depth_m[0], np.zeros(5))
    assert result.depth_m[1].tobytes() == original[1].tobytes()
    np.testing.assert_equal(frame.depth, original)
    assert result.depth_m.dtype == frame.depth.dtype
    assert result.depth_m.flags.c_contiguous and not result.depth_m.flags.writeable
    with pytest.raises(ValueError):
        result.depth_m[0, 0] = 1.0


def test_background_result_validates_and_compares_arrays_by_value() -> None:
    left = BackgroundMaskResult(np.asarray([[0.0, 1.0]], dtype=np.float32), 1, 1)
    right = BackgroundMaskResult(np.asarray([[0.0, 1.0]], dtype=np.float32), 1, 1)
    assert (left == right) is True
    assert (left == object()) is False
    with pytest.raises(TypeError):
        hash(left)
    with pytest.raises((TypeError, ValueError)):
        BackgroundMaskResult(np.ones((1, 1), dtype=bool), 0, 1)
    with pytest.raises(ValueError):
        BackgroundMaskResult(np.ones((1, 1)), 0, 0)
    with pytest.raises(FrozenInstanceError):
        left.excluded_pixel_count = 0  # type: ignore[misc]


def test_all_zero_trial_is_independent_equal_clone_and_touches_nothing() -> None:
    volume = TemporalBackgroundVolume(_config())
    frame = _frame()
    trial = volume.trial_integrate(frame, np.zeros_like(frame.depth))
    assert trial is not volume
    assert trial == volume
    assert trial.last_blocks_touched == 0
    assert trial.canonical_block_state() == volume.canonical_block_state() == ()


def test_all_zero_trial_equals_integrated_source_despite_diagnostic_touch_count() -> None:
    frame = _frame(size=16)
    frame.intrinsics = CameraIntrinsics(8.0, 8.0, 7.5, 7.5, 16, 16)
    original = TemporalBackgroundVolume(_config()).trial_integrate(frame, frame.depth)
    assert original.last_blocks_touched > 0

    trial = original.trial_integrate(frame, np.zeros_like(frame.depth))

    assert trial.last_blocks_touched == 0
    assert trial.canonical_block_state() == original.canonical_block_state()
    assert (trial == original) is True


def test_object_mask_then_dormant_reveal_controls_integration() -> None:
    frame = _frame(size=16)
    frame.intrinsics = CameraIntrinsics(8.0, 8.0, 7.5, 7.5, 16, 16)
    mask = np.ones((16, 16), dtype=bool)
    hidden = build_background_depth(
        frame,
        (_observation(frame, 1, ObservationKind.OBJECT, mask),),
        (),
        _config(),
    )
    volume = TemporalBackgroundVolume(_config())
    still_empty = volume.trial_integrate(frame, hidden.depth_m)
    assert still_empty.active_block_count == 0

    revealed = build_background_depth(frame, (), (), _config())
    integrated = still_empty.trial_integrate(frame, revealed.depth_m)
    assert integrated.active_block_count > 0
    assert integrated.last_blocks_touched > 0


def test_trial_preserves_original_blocks_updates_clone_and_rolls_back_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame(size=16)
    frame.intrinsics = CameraIntrinsics(8.0, 8.0, 7.5, 7.5, 16, 16)
    original = TemporalBackgroundVolume(_config()).trial_integrate(frame, frame.depth)
    before = original.canonical_block_state()
    moved = _frame(8, size=16)
    moved.intrinsics = frame.intrinsics
    moved.pose[0, 3] = 0.25
    trial = original.trial_integrate(moved, moved.depth)
    assert original.canonical_block_state() == before
    assert trial.active_block_count >= original.active_block_count
    assert trial.canonical_block_state() != before

    real_integrate = SparseTsdfVolume.integrate

    def fail_after_integrating(self: SparseTsdfVolume, *args: object, **kwargs: object) -> int:
        real_integrate(self, *args, **kwargs)
        raise RuntimeError("injected")

    monkeypatch.setattr(SparseTsdfVolume, "integrate", fail_after_integrating)
    with pytest.raises(RuntimeError, match="injected"):
        original.trial_integrate(moved, moved.depth)
    assert original.canonical_block_state() == before


@pytest.mark.parametrize(
    "mutate,exception",
    [
        (lambda f, o, p, c: (f, list(o), p, c), TypeError),
        (lambda f, o, p, c: (f, o, list(p), c), TypeError),
        (lambda f, o, p, c: (f, o + (o[0],), p, c), ValueError),
        (lambda f, o, p, c: (f, o, (np.ones((2, 2)),), c), ValueError),
        (lambda f, o, p, c: (f, o, (np.asarray([[np.nan, 0, 1]]),), c), ValueError),
        (lambda f, o, p, c: (f, o, tuple(np.empty((0, 3)) for _ in range(11)), c), ValueError),
        (lambda f, o, p, c: (f, o, (np.ones((11, 3)),), c), ValueError),
        (lambda f, o, p, c: (f, o, p, replace(c, maximum_entities=True)), TypeError),
    ],
)
def test_build_fails_closed_for_bad_contracts(mutate: object, exception: type[Exception]) -> None:
    frame = _frame()
    observation = _observation(frame, 1, ObservationKind.OBJECT, np.zeros((5, 5), bool))
    args = mutate(frame, (observation,), (), _config())  # type: ignore[operator]
    with pytest.raises(exception):
        build_background_depth(*args)


@pytest.mark.parametrize(
    "points",
    [
        np.asarray([["0", "0", "1"]]),
        np.asarray([[0.0 + 1.0j, 0.0, 1.0]]),
    ],
)
def test_protected_points_reject_non_real_dtypes_without_warnings(
    points: np.ndarray,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match="protected_world_points"):
            build_background_depth(_frame(), (), (points,), _config())


def test_build_rejects_frame_mismatch_bad_mask_and_bad_frame() -> None:
    frame = _frame()
    observation = _observation(frame, 1, ObservationKind.OBJECT, np.zeros((5, 5), bool))
    mismatch = replace(observation, frame_id=8)
    with pytest.raises(ValueError, match="frame_id"):
        build_background_depth(frame, (mismatch,), (), _config())
    object.__setattr__(observation, "mask", np.zeros((4, 5), dtype=bool))
    with pytest.raises(ValueError, match="mask"):
        build_background_depth(frame, (observation,), (), _config())
    frame.depth = np.ones((5, 5), dtype=np.int32)
    with pytest.raises(TypeError, match="depth"):
        build_background_depth(frame, (), (), _config())


def test_build_rejects_zero_sized_frames_and_boolean_observation_ids() -> None:
    empty = _frame()
    empty.depth = np.empty((0, 0), dtype=np.float32)
    empty.rgb = np.empty((0, 0, 3), dtype=np.uint8)
    empty.intrinsics = CameraIntrinsics(1.0, 1.0, 0.0, 0.0, 0, 0)
    with pytest.raises(ValueError, match="dimensions|width|height"):
        build_background_depth(empty, (), (), _config())

    frame = _frame()
    observation = _observation(frame, True, ObservationKind.OBJECT, np.zeros((5, 5), bool))
    with pytest.raises(TypeError, match="observation_id"):
        build_background_depth(frame, (observation,), (), _config())


@pytest.mark.parametrize(
    "rgb",
    [
        np.full((5, 5, 3), -1, dtype=np.int32),
        np.full((5, 5, 3), 1000, dtype=np.uint16),
    ],
)
def test_build_and_trial_reject_non_uint8_integer_rgb(rgb: np.ndarray) -> None:
    frame = _frame()
    frame.rgb = rgb
    with pytest.raises(TypeError, match="rgb"):
        build_background_depth(frame, (), (), _config())
    with pytest.raises(TypeError, match="rgb"):
        TemporalBackgroundVolume(_config()).trial_integrate(
            frame, np.zeros_like(frame.depth)
        )


@pytest.mark.parametrize(
    "rgb",
    [
        np.full((5, 5, 3), 127, dtype=np.uint8),
        np.full((5, 5, 3), 0.5, dtype=np.float32),
    ],
)
def test_build_and_trial_accept_uint8_and_unit_float_rgb(rgb: np.ndarray) -> None:
    frame = _frame()
    frame.rgb = rgb
    result = build_background_depth(frame, (), (), _config())
    trial = TemporalBackgroundVolume(_config()).trial_integrate(
        frame, np.zeros_like(frame.depth)
    )
    assert result.valid_background_pixel_count == frame.depth.size
    assert trial.last_blocks_touched == 0


@pytest.mark.parametrize(
    "masked",
    [
        np.ones((5, 5), dtype=np.int32),
        np.ones((4, 5), dtype=np.float32),
        np.full((5, 5), np.nan, dtype=np.float32),
        np.full((5, 5), -1.0, dtype=np.float32),
        np.full((5, 5), 5.0, dtype=np.float32),
    ],
)
def test_trial_rejects_bad_masked_depth_without_mutation(masked: np.ndarray) -> None:
    volume = TemporalBackgroundVolume(_config())
    before = volume.canonical_block_state()
    with pytest.raises((TypeError, ValueError)):
        volume.trial_integrate(_frame(), masked)
    assert volume.canonical_block_state() == before


def test_volume_value_equality_and_repeated_runs_are_deterministic() -> None:
    frame = _frame(size=16)
    frame.intrinsics = CameraIntrinsics(8.0, 8.0, 7.5, 7.5, 16, 16)
    depth1 = build_background_depth(frame, (), (), _config())
    depth2 = build_background_depth(frame, (), (), _config())
    assert depth1 == depth2
    left = TemporalBackgroundVolume(_config()).trial_integrate(frame, depth1.depth_m)
    right = TemporalBackgroundVolume(_config()).trial_integrate(frame, depth2.depth_m)
    assert (left == right) is True
    assert (left == object()) is False
    assert left.canonical_block_state() == right.canonical_block_state()
    with pytest.raises(TypeError):
        hash(left)
