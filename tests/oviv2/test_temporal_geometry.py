from __future__ import annotations

from dataclasses import FrozenInstanceError
import itertools

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.temporal_config import TemporalGeometryConfig
from src.oviv2.temporal_geometry import (
    ObjectMotionEstimate,
    ObjectSubmap,
    backproject_observation,
    estimate_object_motion,
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
    assert not result.used_icp and result.object_to_world[0, 3] == 10.0

    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    monkeypatch.setattr("src.oviv2.temporal_geometry._run_icp", lambda *args: (_ for _ in ()).throw(RuntimeError("bad")))
    result = estimate_object_motion(submap, submap.world_points(), (11.0, 0.0, 0.0), _config())
    assert not result.used_icp and result.object_to_world[0, 3] == 11.0 and np.isfinite(result.rmse_m)
    result = estimate_object_motion(submap, submap.world_points(), (20.0, 0.0, 0.0), _config())
    assert not result.used_icp and result.object_to_world[0, 3] == 10.0


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
    assert not result.used_icp and result.object_to_world[0, 3] == 11.0


def test_motion_accepts_valid_runner_result_and_freezes_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    transform = np.eye(4)
    transform[0, 3] = 11.0
    monkeypatch.setattr("src.oviv2.temporal_geometry._run_icp", lambda *args: (transform, 0.9, 0.1))
    result = estimate_object_motion(submap, submap.world_points(), (11.0, 0.0, 0.0), _config())
    assert result.used_icp and result.fitness == 0.9 and result.rmse_m == 0.1
    assert not result.object_to_world.flags.writeable


def test_motion_point_count_degeneracy_and_invalid_icp_transform_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    too_small = _submap(np.asarray([[0, 0, 0], [1, 0, 0]]))
    result = estimate_object_motion(
        too_small, too_small.world_points(), (11.0, 0.0, 0.0), _config()
    )
    assert not result.used_icp and result.object_to_world[0, 3] == 11.0

    degenerate = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    result = estimate_object_motion(
        degenerate, degenerate.world_points(), (11.0, 0.0, 0.0), _config()
    )
    assert not result.used_icp

    valid = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    monkeypatch.setattr(
        "src.oviv2.temporal_geometry._run_icp", lambda *args: (invalid, 1.0, 0.0)
    )
    result = estimate_object_motion(
        valid, valid.world_points(), (11.0, 0.0, 0.0), _config()
    )
    assert not result.used_icp and result.object_to_world[0, 3] == 11.0


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


def test_real_open3d_icp_recovers_translation() -> None:
    rng = np.random.default_rng(4)
    local = rng.normal(scale=0.04, size=(40, 3))
    submap = ObjectSubmap(
        (0.0, 0.0, 0.0), tuple(sorted(tuple(int(v) for v in row) for row in np.arange(40)[:, None] * np.asarray([[1, 0, 0]]))),
        local, np.ones(40), np.zeros(40, dtype=np.int64),
    )
    target = local + np.asarray([0.1, -0.05, 0.02])
    result = estimate_object_motion(submap, target, (0.1, -0.05, 0.02), _config(voxel_size_m=0.1, minimum_icp_points=20))
    assert result.used_icp
    np.testing.assert_allclose(result.object_to_world[:3, 3], [0.1, -0.05, 0.02], atol=1e-3)


def test_motion_estimate_and_config_fail_closed() -> None:
    bad = np.eye(4)
    bad[0, 0] = 2.0
    with pytest.raises(ValueError):
        ObjectMotionEstimate(bad, False, 0.0, 0.0)
    submap = _submap(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    with pytest.raises(ValueError, match="voxel_size_m"):
        estimate_object_motion(submap, submap.world_points(), (10.0, 0.0, 0.0), _config(voxel_size_m=np.nan))
    with pytest.raises(TypeError, match="minimum_icp_points"):
        estimate_object_motion(submap, submap.world_points(), (10.0, 0.0, 0.0), _config(minimum_icp_points=True))
    with pytest.raises(TypeError, match="points_world"):
        estimate_object_motion(
            submap, np.ones((3, 3), dtype=bool), (10.0, 0.0, 0.0), _config()
        )
