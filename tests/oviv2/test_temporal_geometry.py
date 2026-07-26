from __future__ import annotations

from dataclasses import FrozenInstanceError
import itertools
import warnings

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.temporal_config import TemporalGeometryConfig
from src.oviv2.temporal_geometry import (
    MotionDecision,
    ObjectMotionEstimate,
    ObjectSubmap,
    backproject_observation,
    estimate_object_motion,
    estimate_object_translation,
    integrate_object_submap,
)


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


def _frame() -> Frame:
    return Frame(
        frame_id=7,
        timestamp=1.5,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth=np.asarray([[1.0, np.nan, 2.0], [0.0, -1.0, 5.0]], dtype=np.float32),
        pose=np.asarray(
            [[0.0, -1.0, 0.0, 10.0], [1.0, 0.0, 0.0, 20.0], [0.0, 0.0, 1.0, 30.0], [0.0, 0.0, 0.0, 1.0]]
        ),
        intrinsics=CameraIntrinsics(2.0, 4.0, 1.0, 0.5, 3, 2),
    )


def _observation(**changes: object) -> FrameObservation:
    values = {
        "observation_id": 1,
        "frame_id": 7,
        "timestamp": 1.5,
        "kind": ObservationKind.OBJECT,
        "label": "chair",
        "semantic_id": 1,
        "confidence": 1.0,
        "mask": np.ones((2, 3), dtype=bool),
        "bbox_xyxy": (0.0, 0.0, 3.0, 2.0),
        "voxel_keys": frozenset({(0, 0, 0)}),
        "centroid_xyz": (0.0, 0.0, 0.0),
        "bounds_min_xyz": (0.0, 0.0, 0.0),
        "bounds_max_xyz": (0.0, 0.0, 0.0),
    }
    values.update(changes)
    return FrameObservation(**values)


def _submap(
    points: np.ndarray | None = None,
    *,
    reference: tuple[float, float, float] = (10.0, 0.0, 0.0),
) -> ObjectSubmap:
    local = np.empty((0, 3)) if points is None else np.asarray(points, dtype=np.float64)
    records = sorted((tuple(int(v) for v in np.floor(point)), point) for point in local)
    keys = tuple(key for key, _ in records)
    local = np.asarray([point for _, point in records], dtype=np.float64).reshape((-1, 3))
    return ObjectSubmap(
        reference_centroid_xyz=reference,
        local_voxel_keys=keys,
        local_points_xyz=local,
        weights=np.ones(len(local)),
        last_seen_frame_ids=np.zeros(len(local), dtype=np.int64),
    )


def test_backproject_filters_depth_transforms_in_pixel_order_and_preserves_inputs() -> None:
    frame = _frame()
    observation = _observation()
    originals = tuple(array.copy() for array in (frame.depth, frame.rgb, frame.pose, observation.mask))

    points = backproject_observation(frame, observation, _config())

    np.testing.assert_allclose(points, [[10.125, 19.5, 31.0], [10.25, 21.0, 32.0]])
    assert points.dtype == np.float64 and not points.flags.writeable
    for actual, expected in zip((frame.depth, frame.rgb, frame.pose, observation.mask), originals):
        np.testing.assert_equal(actual, expected)


def test_backproject_empty_is_typed_readonly_and_unknown_is_allowed() -> None:
    points = backproject_observation(
        _frame(), _observation(kind=ObservationKind.UNKNOWN, mask=np.zeros((2, 3), bool)), _config()
    )
    assert points.shape == (0, 3) and points.dtype == np.float64 and not points.flags.writeable


@pytest.mark.parametrize(
    "change,message",
    [
        ({"kind": ObservationKind.STRUCTURE}, "kind"),
        ({"frame_id": 8}, "frame_id"),
        ({"timestamp": 2.0}, "timestamp"),
        ({"mask": np.ones((1, 1), bool)}, "mask"),
    ],
)
def test_backproject_rejects_mismatched_observations(change: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        backproject_observation(_frame(), _observation(**change), _config())


def test_backproject_rejects_invalid_frame_geometry() -> None:
    frame = _frame()
    frame.pose[3, 0] = 1.0
    with pytest.raises(ValueError, match="pose"):
        backproject_observation(frame, _observation(), _config())
    frame = _frame()
    frame.intrinsics.fx = 0.0
    with pytest.raises(ValueError, match="intrinsics"):
        backproject_observation(frame, _observation(), _config())


def test_submap_copies_freezes_and_maps_local_points_to_world() -> None:
    points = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    weights = np.asarray([1.0, 2.0])
    seen = np.asarray([2, 3], dtype=np.int64)
    submap = ObjectSubmap((10.0, 20.0, 30.0), ((0, 1, 2), (1, 2, 3)), points, weights, seen)
    points[:] = 99.0
    weights[:] = 99.0
    seen[:] = 99

    np.testing.assert_equal(submap.local_points_xyz, [[0, 1, 2], [1, 2, 3]])
    np.testing.assert_equal(submap.world_points(), [[10, 21, 32], [11, 22, 33]])
    moved = np.eye(4)
    moved[:3, 3] = (-5.0, 0.0, 1.0)
    np.testing.assert_equal(submap.world_points(moved), [[-5, 1, 3], [-4, 2, 4]])
    for array in (submap.local_points_xyz, submap.weights, submap.last_seen_frame_ids, submap.world_points()):
        assert array.flags.c_contiguous and not array.flags.writeable
    with pytest.raises(FrozenInstanceError):
        submap.weights = np.ones(2)  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_centroid_xyz": (0.0, np.nan, 0.0)},
        {"local_voxel_keys": [(0, 0, 0)]},
        {"local_voxel_keys": ((1, 0, 0), (0, 0, 0))},
        {"local_voxel_keys": ((0, 0, 0), (0, 0, 0))},
        {"local_voxel_keys": ((True, 0, 0),)},
        {"local_points_xyz": np.zeros((1, 2))},
        {"local_points_xyz": np.asarray([[np.inf, 0.0, 0.0]])},
        {"weights": np.asarray([0.0])},
        {"last_seen_frame_ids": np.asarray([-1])},
    ],
)
def test_submap_rejects_invalid_contract(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "reference_centroid_xyz": (0.0, 0.0, 0.0),
        "local_voxel_keys": ((0, 0, 0),),
        "local_points_xyz": np.zeros((1, 3)),
        "weights": np.ones(1),
        "last_seen_frame_ids": np.zeros(1, dtype=np.int64),
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        ObjectSubmap(**values)


def test_integrate_uses_reference_local_coordinates_and_weighted_voxel_means() -> None:
    submap = ObjectSubmap((10.0, 0.0, 0.0), ((0, 0, 0),), np.asarray([[0.2, 0.0, 0.0]]), np.asarray([2.0]), np.asarray([1]))
    result = integrate_object_submap(
        submap, np.asarray([[10.8, 0.0, 0.0], [11.2, 0.0, 0.0]]), 2, _config()
    )
    assert result.local_voxel_keys == ((0, 0, 0), (1, 0, 0))
    np.testing.assert_allclose(result.local_points_xyz, [[0.4, 0, 0], [1.2, 0, 0]])
    np.testing.assert_equal(result.weights, [3.0, 1.0])
    np.testing.assert_equal(result.last_seen_frame_ids, [2, 2])


def test_integrate_is_bit_identical_for_all_input_permutations() -> None:
    points = np.asarray([[0.1, 0, 0], [0.3, 0, 0], [1.1, 0, 0], [0.2, 0, 0]])
    results = [integrate_object_submap(_submap(reference=(0.0, 0.0, 0.0)), points[list(order)], 1, _config()) for order in itertools.permutations(range(4))]
    for result in results[1:]:
        assert result.local_voxel_keys == results[0].local_voxel_keys
        assert result.local_points_xyz.tobytes() == results[0].local_points_xyz.tobytes()
        assert result.weights.tobytes() == results[0].weights.tobytes()


def test_integrate_capacity_evicts_by_weight_then_age_then_key() -> None:
    submap = ObjectSubmap(
        (0.0, 0.0, 0.0), ((0, 0, 0), (1, 0, 0), (2, 0, 0)),
        np.asarray([[0.1, 0, 0], [1.1, 0, 0], [2.1, 0, 0]]),
        np.asarray([1.0, 1.0, 2.0]), np.asarray([1, 2, 1]),
    )
    result = integrate_object_submap(submap, np.asarray([[3.1, 0, 0]]), 3, _config(maximum_object_voxels=2))
    assert result.local_voxel_keys == ((2, 0, 0), (3, 0, 0))


def test_integrate_empty_is_identity_and_frame_ids_are_monotonic() -> None:
    submap = _submap(np.asarray([[0.1, 0, 0]]), reference=(0.0, 0.0, 0.0))
    assert integrate_object_submap(submap, np.empty((0, 3)), 1, _config()) is submap
    with pytest.raises(ValueError, match="frame_id"):
        integrate_object_submap(submap, np.ones((1, 3)), 0, _config())
    with pytest.raises(TypeError, match="frame_id"):
        integrate_object_submap(submap, np.ones((1, 3)), True, _config())


def test_motion_empty_returns_reference_pose_and_centroid_fallback_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = _submap()
    result = estimate_object_motion(empty, np.ones((3, 3)), (11.0, 0.0, 0.0), _config())
    assert result.decision is MotionDecision.REJECTED
    assert result.object_to_world[0, 3] == 10.0

    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    monkeypatch.setattr("src.oviv2.temporal_geometry._run_icp", lambda *args: (_ for _ in ()).throw(RuntimeError("bad")))
    result = estimate_object_motion(submap, submap.world_points(), (11.0, 0.0, 0.0), _config())
    assert result.decision is MotionDecision.TRANSLATION_ACCEPTED
    assert result.object_to_world[0, 3] == 11.0 and np.isfinite(result.rmse_m)
    result = estimate_object_motion(submap, submap.world_points(), (20.0, 0.0, 0.0), _config())
    assert result.decision is MotionDecision.REJECTED
    assert result.object_to_world[0, 3] == 10.0


def test_translation_motion_is_bounded_and_never_calls_icp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    monkeypatch.setattr(
        "src.oviv2.temporal_geometry._run_icp",
        lambda *args: (_ for _ in ()).throw(AssertionError("ICP must not run")),
    )

    moved = estimate_object_translation(
        submap,
        submap.world_points(),
        (11.0, 0.0, 0.0),
        _config(),
    )
    rejected = estimate_object_translation(
        submap,
        submap.world_points(),
        (13.0, 0.0, 0.0),
        _config(),
    )

    assert moved.decision is MotionDecision.TRANSLATION_ACCEPTED
    assert moved.object_to_world[0, 3] == 11.0
    assert rejected.decision is MotionDecision.REJECTED
    assert rejected.object_to_world[0, 3] == 10.0


def test_translation_extreme_finite_centroids_fail_closed_without_warning() -> None:
    submap = _submap(np.asarray([[0.0, 0.0, 0.0]]))
    maximum = float(np.finfo(np.float64).max)
    previous = np.eye(4, dtype=np.float64)
    previous[0, 3] = -maximum

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = estimate_object_translation(
            submap,
            np.zeros((1, 3), dtype=np.float64),
            (maximum, 0.0, 0.0),
            _config(),
            previous_object_to_world=previous,
        )

    np.testing.assert_equal(result.object_to_world, previous)
    assert result.decision is MotionDecision.REJECTED


@pytest.mark.parametrize(
    "points,centroid",
    [
        (np.empty((0, 3)), (11.0, 0.0, 0.0)),
        (np.asarray([[np.nan, 0.0, 0.0]]), (11.0, 0.0, 0.0)),
        (np.ones((1, 3)), (np.inf, 0.0, 0.0)),
    ],
)
def test_translation_rejects_empty_or_nonfinite_observation_without_moving(
    points: np.ndarray, centroid: tuple[float, float, float]
) -> None:
    submap = _submap(np.asarray([[0.0, 0.0, 0.0]]))
    previous = np.eye(4)
    previous[0, 3] = 10.0

    result = estimate_object_translation(
        submap,
        points,
        centroid,
        _config(),
        previous_object_to_world=previous,
    )

    assert result.decision is MotionDecision.REJECTED
    np.testing.assert_array_equal(result.object_to_world, previous)
    assert np.isfinite(result.fitness) and np.isfinite(result.rmse_m)


def test_motion_rejects_when_source_is_empty_even_with_valid_target() -> None:
    previous = np.eye(4)
    previous[0, 3] = 10.0
    result = estimate_object_motion(
        _submap(),
        np.ones((3, 3)),
        (11.0, 0.0, 0.0),
        _config(),
        previous_object_to_world=previous,
    )
    assert result.decision is MotionDecision.REJECTED
    np.testing.assert_array_equal(result.object_to_world, previous)


@pytest.mark.parametrize(
    "fitness,rmse,translation",
    [(0.69, 0.1, 11.0), (0.9, 0.21, 11.0), (0.9, 0.1, 13.0)],
)
def test_motion_rejects_each_icp_quality_gate(
    monkeypatch: pytest.MonkeyPatch, fitness: float, rmse: float, translation: float
) -> None:
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    transform = np.eye(4)
    transform[0, 3] = translation
    monkeypatch.setattr("src.oviv2.temporal_geometry._run_icp", lambda *args: (transform, fitness, rmse))
    result = estimate_object_motion(submap, submap.world_points(), (11.0, 0.0, 0.0), _config())
    assert result.decision is MotionDecision.TRANSLATION_ACCEPTED
    assert result.object_to_world[0, 3] == 11.0


def test_motion_rejects_when_icp_and_translation_both_fail_quality_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    previous = np.eye(4)
    previous[0, 3] = 10.0
    invalid_icp = np.eye(4)
    invalid_icp[0, 3] = 20.0
    monkeypatch.setattr(
        "src.oviv2.temporal_geometry._run_icp",
        lambda *args: (invalid_icp, 0.1, np.inf),
    )

    result = estimate_object_motion(
        submap,
        submap.world_points(previous),
        (20.0, 0.0, 0.0),
        _config(),
        previous_object_to_world=previous,
    )

    assert result.decision is MotionDecision.REJECTED
    np.testing.assert_array_equal(result.object_to_world, previous)
    assert np.isfinite(result.fitness) and np.isfinite(result.rmse_m)


def test_motion_accepts_valid_runner_result_and_freezes_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    transform = np.eye(4)
    transform[0, 3] = 11.0
    monkeypatch.setattr("src.oviv2.temporal_geometry._run_icp", lambda *args: (transform, 0.9, 0.1))
    result = estimate_object_motion(submap, submap.world_points(), (11.0, 0.0, 0.0), _config())
    assert result.decision is MotionDecision.ICP_ACCEPTED
    assert result.fitness == 0.9 and result.rmse_m == 0.1
    assert not result.object_to_world.flags.writeable


def test_legacy_four_position_bool_constructor_normalizes_at_boundary() -> None:
    icp = ObjectMotionEstimate(np.eye(4), True, 0.9, 0.1)
    translation = ObjectMotionEstimate(np.eye(4), False, 0.0, 0.2)

    assert icp.decision is MotionDecision.ICP_ACCEPTED and icp.used_icp is True
    assert translation.decision is MotionDecision.TRANSLATION_ACCEPTED
    assert translation.used_icp is False
    assert "used_icp" not in vars(translation)


def test_motion_point_count_degeneracy_and_invalid_icp_transform_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    too_small = _submap(np.asarray([[0, 0, 0], [1, 0, 0]]))
    result = estimate_object_motion(
        too_small, too_small.world_points(), (11.0, 0.0, 0.0), _config()
    )
    assert result.decision is MotionDecision.TRANSLATION_ACCEPTED
    assert result.object_to_world[0, 3] == 11.0

    degenerate = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    result = estimate_object_motion(
        degenerate, degenerate.world_points(), (11.0, 0.0, 0.0), _config()
    )
    assert result.decision is MotionDecision.TRANSLATION_ACCEPTED

    valid = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    monkeypatch.setattr(
        "src.oviv2.temporal_geometry._run_icp", lambda *args: (invalid, 1.0, 0.0)
    )
    result = estimate_object_motion(
        valid, valid.world_points(), (11.0, 0.0, 0.0), _config()
    )
    assert result.decision is MotionDecision.TRANSLATION_ACCEPTED
    assert result.object_to_world[0, 3] == 11.0


def test_extreme_finite_geometry_is_rejected_without_numeric_warning() -> None:
    maximum = float(np.finfo(np.float64).max)
    points = np.asarray(
        [[maximum, 0.0, 0.0], [maximum, 1.0, 0.0], [-maximum, 0.0, 1.0]]
    )
    submap = ObjectSubmap(
        (0.0, 0.0, 0.0),
        ((-1, 0, 1), (1, 0, 0), (1, 1, 0)),
        points[[2, 0, 1]],
        np.ones(3),
        np.zeros(3, dtype=np.int64),
    )
    previous = np.eye(4)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = estimate_object_motion(
            submap,
            points,
            (0.1, 0.0, 0.0),
            _config(),
            previous_object_to_world=previous,
        )

    assert result.decision is MotionDecision.REJECTED
    np.testing.assert_array_equal(result.object_to_world, previous)
    assert np.isfinite(result.fitness) and np.isfinite(result.rmse_m)


@pytest.mark.parametrize("extreme_side", ["source", "target", "both"])
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (2, 1, 0)])
@pytest.mark.parametrize("minimum_icp_points", [3, 4])
def test_unrepresentable_finite_euclidean_geometry_is_rejected_for_all_orders(
    extreme_side: str, order: tuple[int, int, int], minimum_icp_points: int
) -> None:
    maximum = float(np.finfo(np.float64).max)
    extreme = np.asarray(
        [
            [maximum, maximum, 0.0],
            [maximum, maximum, 1.0],
            [maximum, maximum - 1.0, 2.0],
        ]
    )[list(order)]
    ordinary = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    source = extreme if extreme_side in ("source", "both") else ordinary
    target = extreme.copy() if extreme_side in ("target", "both") else ordinary.copy()
    submap = ObjectSubmap(
        (0.0, 0.0, 0.0),
        ((0, 0, 0), (1, 0, 0), (2, 0, 0)),
        source,
        np.ones(3),
        np.zeros(3, dtype=np.int64),
    )
    previous = np.eye(4)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = estimate_object_motion(
            submap,
            target,
            (0.0, 0.0, 0.0),
            _config(
                depth_max_m=maximum,
                minimum_icp_points=minimum_icp_points,
            ),
            previous_object_to_world=previous,
        )

    assert result.decision is MotionDecision.REJECTED
    np.testing.assert_array_equal(result.object_to_world, previous)
    assert np.isfinite(result.fitness) and np.isfinite(result.rmse_m)


def test_motion_uses_configured_correspondence_distance(monkeypatch: pytest.MonkeyPatch) -> None:
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    captured: list[float] = []

    def runner(source: np.ndarray, target: np.ndarray, initial: np.ndarray, distance: float):
        captured.append(distance)
        return initial, 1.0, 0.0

    monkeypatch.setattr("src.oviv2.temporal_geometry._run_icp", runner)
    estimate_object_motion(
        submap,
        submap.world_points(),
        (10.0, 0.0, 0.0),
        _config(voxel_size_m=0.05, maximum_icp_rmse_m=0.3),
    )
    assert captured == [0.3]


def test_motion_uses_reference_icp_initial_pose_when_centroid_exceeds_motion_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    initials: list[np.ndarray] = []

    def runner(source: np.ndarray, target: np.ndarray, initial: np.ndarray, distance: float):
        initials.append(initial.copy())
        return initial, 1.0, 0.0

    monkeypatch.setattr("src.oviv2.temporal_geometry._run_icp", runner)
    result = estimate_object_motion(
        submap, submap.world_points(), (20.0, 0.0, 0.0), _config()
    )

    assert initials[0][0, 3] == 10.0
    assert result.object_to_world[0, 3] == 10.0


def test_real_open3d_icp_recovers_translation() -> None:
    rng = np.random.default_rng(4)
    local = rng.normal(scale=0.04, size=(40, 3))
    submap = ObjectSubmap(
        (0.0, 0.0, 0.0), tuple(sorted(tuple(int(v) for v in row) for row in np.arange(40)[:, None] * np.asarray([[1, 0, 0]]))),
        local, np.ones(40), np.zeros(40, dtype=np.int64),
    )
    target = local + np.asarray([0.1, -0.05, 0.02])
    result = estimate_object_motion(submap, target, (0.1, -0.05, 0.02), _config(voxel_size_m=0.1, minimum_icp_points=20))
    assert result.decision is MotionDecision.ICP_ACCEPTED
    np.testing.assert_allclose(result.object_to_world[:3, 3], [0.1, -0.05, 0.02], atol=1e-3)


def test_motion_estimate_and_config_fail_closed() -> None:
    bad = np.eye(4)
    bad[0, 0] = 2.0
    with pytest.raises(ValueError):
        ObjectMotionEstimate(bad, MotionDecision.REJECTED, 0.0, 0.0)
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    with pytest.raises(ValueError, match="voxel_size_m"):
        estimate_object_motion(submap, submap.world_points(), (10.0, 0.0, 0.0), _config(voxel_size_m=np.nan))
    with pytest.raises(TypeError, match="minimum_icp_points"):
        estimate_object_motion(submap, submap.world_points(), (10.0, 0.0, 0.0), _config(minimum_icp_points=True))
    with pytest.raises(TypeError, match="points_world"):
        estimate_object_motion(
            submap, np.ones((3, 3), dtype=bool), (10.0, 0.0, 0.0), _config()
        )


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"voxel_size_m": 0.0}, "voxel_size_m"),
        ({"depth_max_m": np.nan}, "depth_max_m"),
        ({"maximum_entities": 0}, "maximum_entities"),
        ({"maximum_object_voxels": 0}, "maximum_object_voxels"),
        ({"maximum_visibility_points_per_entity": 0}, "maximum_visibility_points_per_entity"),
        ({"background_block_count": 0}, "background_block_count"),
        ({"background_mask_dilation_px": -1}, "background_mask_dilation_px"),
        ({"background_mask_dilation_px": True}, "background_mask_dilation_px"),
        ({"minimum_icp_points": 0}, "minimum_icp_points"),
        ({"minimum_icp_fitness": 1.1}, "minimum_icp_fitness"),
        ({"maximum_icp_rmse_m": 0.0}, "maximum_icp_rmse_m"),
        ({"maximum_motion_m": 0.0}, "maximum_motion_m"),
        ({"maximum_object_voxels": 11}, "maximum_object_voxels"),
        ({"maximum_entities": 11}, "maximum_entities"),
        ({"maximum_visibility_points_per_entity": 11}, "maximum_visibility_points_per_entity"),
    ],
)
def test_direct_config_validation_matches_task1_bounds(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        integrate_object_submap(_submap(), np.empty((0, 3)), 0, _config(**changes))


def test_bool_geometry_inputs_are_rejected() -> None:
    values = {
        "reference_centroid_xyz": (True, 0.0, 0.0),
        "local_voxel_keys": ((0, 0, 0),),
        "local_points_xyz": np.zeros((1, 3)),
        "weights": np.ones(1),
        "last_seen_frame_ids": np.zeros(1, dtype=np.int64),
    }
    with pytest.raises(TypeError, match="reference_centroid_xyz"):
        ObjectSubmap(**values)
    values["reference_centroid_xyz"] = (0.0, 0.0, 0.0)
    values["weights"] = np.ones(1, dtype=bool)
    with pytest.raises(TypeError, match="weights"):
        ObjectSubmap(**values)

    with pytest.raises(TypeError, match="object_to_world"):
        ObjectMotionEstimate(np.eye(4, dtype=bool), MotionDecision.REJECTED, 0.0, 0.0)
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    with pytest.raises(TypeError, match="observed_centroid_xyz"):
        estimate_object_motion(
            submap, submap.world_points(), (True, 0.0, 0.0), _config()
        )

    frame = _frame()
    frame.depth = np.ones(frame.depth.shape, dtype=bool)
    with pytest.raises(TypeError, match="depth"):
        backproject_observation(frame, _observation(), _config())
    frame = _frame()
    frame.intrinsics.fx = True
    with pytest.raises(TypeError, match="intrinsics"):
        backproject_observation(frame, _observation(), _config())


@pytest.mark.parametrize("error_type", [RuntimeError, FloatingPointError])
def test_expected_icp_runtime_failures_fall_back(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))

    def runner(*args: object) -> tuple[np.ndarray, float, float]:
        raise error_type("expected numerical failure")

    monkeypatch.setattr("src.oviv2.temporal_geometry._run_icp", runner)
    result = estimate_object_motion(
        submap, submap.world_points(), (11.0, 0.0, 0.0), _config()
    )
    assert result.decision is MotionDecision.TRANSLATION_ACCEPTED
    assert result.object_to_world[0, 3] == 11.0


@pytest.mark.parametrize("error_type", [TypeError, MemoryError])
def test_programming_and_resource_icp_errors_propagate(
    monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
) -> None:
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))

    def runner(*args: object) -> tuple[np.ndarray, float, float]:
        raise error_type("must propagate")

    monkeypatch.setattr("src.oviv2.temporal_geometry._run_icp", runner)
    with pytest.raises(error_type, match="must propagate"):
        estimate_object_motion(
            submap, submap.world_points(), (11.0, 0.0, 0.0), _config()
        )


def test_invalid_icp_return_values_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    monkeypatch.setattr(
        "src.oviv2.temporal_geometry._run_icp",
        lambda *args: (np.eye(3), "invalid", object()),
    )
    result = estimate_object_motion(
        submap, submap.world_points(), (11.0, 0.0, 0.0), _config()
    )
    assert result.decision is MotionDecision.TRANSLATION_ACCEPTED
    assert result.object_to_world[0, 3] == 11.0


def test_integrate_with_current_pose_returns_points_to_canonical_local_frame() -> None:
    submap = ObjectSubmap(
        (0.0, 0.0, 0.0),
        ((0, 0, 0),),
        np.asarray([[0.2, 0.0, 0.0]]),
        np.asarray([1.0]),
        np.asarray([0], dtype=np.int64),
    )
    pose = np.asarray(
        [[0.0, -1.0, 0.0, 5.0], [1.0, 0.0, 0.0, 6.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    original_pose = pose.copy()
    point_world = np.asarray([[5.0, 6.4, 0.0]])

    result = integrate_object_submap(
        submap, point_world, 1, _config(), object_to_world=pose
    )

    assert result.local_voxel_keys == ((0, 0, 0),)
    np.testing.assert_allclose(result.local_points_xyz, [[0.3, 0.0, 0.0]])
    np.testing.assert_equal(pose, original_pose)


def test_motion_composes_bounded_steps_from_previous_pose() -> None:
    submap = _submap(
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        reference=(0.0, 0.0, 0.0),
    )
    config = _config(minimum_icp_points=4, maximum_motion_m=1.0)
    first = estimate_object_motion(
        submap, submap.world_points(), (0.75, 0.0, 0.0), config
    )
    second = estimate_object_motion(
        submap,
        submap.world_points(first.object_to_world),
        (1.5, 0.0, 0.0),
        config,
        previous_object_to_world=first.object_to_world,
    )
    too_far = estimate_object_motion(
        submap,
        submap.world_points(second.object_to_world),
        (2.6, 0.0, 0.0),
        config,
        previous_object_to_world=second.object_to_world,
    )

    assert first.object_to_world[0, 3] == 0.75
    assert second.object_to_world[0, 3] == 1.5
    np.testing.assert_equal(too_far.object_to_world, second.object_to_world)


def test_previous_pose_rotation_is_preserved_and_input_is_not_mutated() -> None:
    submap = _submap(
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        reference=(0.0, 0.0, 0.0),
    )
    previous = np.asarray(
        [[0.0, -1.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    original = previous.copy()
    result = estimate_object_motion(
        submap,
        submap.world_points(previous),
        (0.75, 0.0, 0.0),
        _config(minimum_icp_points=4),
        previous_object_to_world=previous,
    )

    np.testing.assert_equal(result.object_to_world[:3, :3], previous[:3, :3])
    np.testing.assert_equal(previous, original)
    assert not result.object_to_world.flags.writeable


def test_empty_motion_returns_explicit_previous_pose() -> None:
    previous = np.eye(4)
    previous[0, 3] = 3.0
    result = estimate_object_motion(
        _submap(),
        np.ones((1, 3)),
        (4.0, 0.0, 0.0),
        _config(),
        previous_object_to_world=previous,
    )
    np.testing.assert_equal(result.object_to_world, previous)


def test_temporal_geometry_value_equality_is_array_aware() -> None:
    submap = _submap(np.asarray([[0.1, 0.0, 0.0]]), reference=(0.0, 0.0, 0.0))
    same_submap = ObjectSubmap(
        submap.reference_centroid_xyz,
        submap.local_voxel_keys,
        submap.local_points_xyz.copy(),
        submap.weights.copy(),
        submap.last_seen_frame_ids.copy(),
    )
    different_submap = integrate_object_submap(
        submap, np.asarray([[0.2, 0.0, 0.0]]), 1, _config()
    )
    pose = np.eye(4)
    motion = ObjectMotionEstimate(pose, MotionDecision.REJECTED, 0.0, 0.2)
    same_motion = ObjectMotionEstimate(pose.copy(), MotionDecision.REJECTED, 0.0, 0.2)
    other_motion = ObjectMotionEstimate(pose.copy(), MotionDecision.REJECTED, 0.1, 0.2)

    assert (submap == same_submap) is True
    assert (submap == different_submap) is False
    assert (submap == object()) is False
    assert (motion == same_motion) is True
    assert (motion == other_motion) is False
    assert (motion == object()) is False
    with pytest.raises(TypeError):
        hash(submap)
    with pytest.raises(TypeError):
        hash(motion)


def test_integrate_rejects_frame_id_beyond_int64_even_when_empty() -> None:
    submap = _submap()
    with pytest.raises(ValueError, match="frame_id"):
        integrate_object_submap(
            submap, np.empty((0, 3)), np.iinfo(np.int64).max + 1, _config()
        )


def test_integrate_voxel_division_overflow_is_transactional() -> None:
    submap = _submap(np.asarray([[0.1, 0.0, 0.0]]), reference=(0.0, 0.0, 0.0))
    original_points = submap.local_points_xyz.copy()
    original_weights = submap.weights.copy()
    original_seen = submap.last_seen_frame_ids.copy()

    with pytest.raises(ValueError, match="voxel"):
        integrate_object_submap(
            submap,
            np.asarray([[1e308, 0.0, 0.0]]),
            1,
            _config(voxel_size_m=1e-308, depth_max_m=1e308),
        )

    np.testing.assert_equal(submap.local_points_xyz, original_points)
    np.testing.assert_equal(submap.weights, original_weights)
    np.testing.assert_equal(submap.last_seen_frame_ids, original_seen)
