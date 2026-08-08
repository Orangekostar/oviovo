from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.observations import ObservationKind
from src.oviv2.temporal_dense_observations import (
    TemporalDenseObservationConfig,
    generate_temporal_dense_observations,
)

IMAGE_SHAPE = (480, 720)
NATIVE_SHAPE = (120, 180)


def _frame(*, frame_id: int = 7, depth: np.ndarray | None = None) -> Frame:
    if depth is None:
        depth = np.full(IMAGE_SHAPE, 2.0, dtype=np.float32)
    return Frame(
        frame_id=frame_id,
        source_frame_id=frame_id + 100,
        timestamp=1.25,
        rgb=np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8),
        depth=depth,
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(500.0, 500.0, 359.5, 239.5, 720, 480),
    )


def _dense(
    ids: np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    frame_id: int = 7,
    class_count: int = 3,
    sample_stride: int = 4,
) -> DenseSemanticFrame:
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim == 2:
        ids = ids[..., None]
    if probabilities is None:
        probabilities = np.where(ids > 0, 0.8, 0.0).astype(np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim == 2:
        probabilities = probabilities[..., None]
    second = probabilities[..., 1] if probabilities.shape[-1] > 1 else 0.0
    return DenseSemanticFrame(
        cache_frame_id=frame_id,
        source_frame_id=frame_id + 100,
        image_shape=IMAGE_SHAPE,
        sample_stride=sample_stride,
        class_count=class_count,
        class_ids=ids,
        probabilities=probabilities,
        entropy=np.zeros(ids.shape[:2], dtype=np.float32),
        margin=np.asarray(probabilities[..., 0] - second, dtype=np.float32),
    )


def _config(**changes: object) -> TemporalDenseObservationConfig:
    values: dict[str, object] = {
        "sample_stride": 4,
        "depth_max_m": 10.0,
        "minimum_area_px": 10,
        "maximum_area_px": 62_208,
        "maximum_observations": 32,
        "voxel_size_m": 0.05,
        "pixel_stride": 2,
        "min_valid_points": 1,
    }
    values.update(changes)
    return TemporalDenseObservationConfig(**values)


def test_config_is_frozen_and_strictly_validated() -> None:
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.depth_max_m = 8.0  # type: ignore[misc]

    for changes, match in (
        ({"sample_stride": 2}, "sample_stride"),
        ({"sample_stride": True}, "sample_stride"),
        ({"depth_max_m": np.inf}, "depth_max_m"),
        ({"depth_max_m": True}, "depth_max_m"),
        ({"minimum_area_px": 0}, "minimum_area_px"),
        ({"maximum_area_px": 9}, "maximum_area_px"),
        ({"maximum_observations": 0}, "maximum_observations"),
        ({"maximum_observations": 2**20 + 1}, "maximum_observations"),
        ({"voxel_size_m": 0.0}, "voxel_size_m"),
        ({"pixel_stride": 0}, "pixel_stride"),
        ({"min_valid_points": 0}, "min_valid_points"),
    ):
        with pytest.raises((TypeError, ValueError), match=match):
            _config(**changes)


def test_generates_four_connected_components_with_nearest_expansion_and_geometry() -> (
    None
):
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    ids[1, 1] = 1
    ids[2, 2] = 1  # diagonal contact is a separate 4-connected component
    observations = generate_temporal_dense_observations(
        _frame(), _dense(ids), (" Chair ", "table", "lamp"), _config()
    )

    assert len(observations) == 2
    assert [item.visible_pixel_count for item in observations] == [16, 16]
    assert [item.bbox_xyxy for item in observations] == [
        (4.0, 4.0, 8.0, 8.0),
        (8.0, 8.0, 12.0, 12.0),
    ]
    assert all(item.kind is ObservationKind.OBJECT for item in observations)
    assert all(item.label == "chair" and item.semantic_id == 1 for item in observations)
    assert all(item.mask.flags.writeable is False for item in observations)


def test_filters_invalid_depth_before_area_confidence_bbox_and_lift() -> None:
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    ids[3, 4:6] = 2
    probabilities = np.zeros((*NATIVE_SHAPE, 1), dtype=np.float32)
    probabilities[3, 4] = 0.2
    probabilities[3, 5] = 0.8
    depth = np.full(IMAGE_SHAPE, 2.0, dtype=np.float32)
    depth[12:16, 16:20] = np.nan
    depth[12:13, 20:24] = 11.0

    (item,) = generate_temporal_dense_observations(
        _frame(depth=depth),
        _dense(ids, probabilities),
        ("chair", "Side_Table", "lamp"),
        _config(),
    )

    assert item.visible_pixel_count == 12
    assert item.confidence == pytest.approx(0.8)
    assert item.bbox_xyxy == (20.0, 13.0, 24.0, 16.0)
    assert item.label == "side-table"


def test_area_filter_is_inclusive_and_uses_valid_full_resolution_pixels() -> None:
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    ids[0, 0] = 1
    ids[10:12, 10:12] = 2
    depth = np.full(IMAGE_SHAPE, np.nan, dtype=np.float32)
    depth[0:4, 0:4] = 2.0
    depth[40:48, 40:48] = 2.0

    observations = generate_temporal_dense_observations(
        _frame(depth=depth),
        _dense(ids),
        ("chair", "table", "lamp"),
        _config(minimum_area_px=16, maximum_area_px=16),
    )

    assert len(observations) == 1
    assert observations[0].semantic_id == 1
    assert observations[0].visible_pixel_count == 16


def test_sorting_and_cap_are_deterministic_and_target_blind() -> None:
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    ids[5, 10:12] = 2  # area 32, confidence .4
    ids[8, 20:22] = 1  # area 32, confidence .9, first among confidence ties
    ids[8, 30:32] = 3  # area 32, confidence .9
    probabilities = np.zeros((*NATIVE_SHAPE, 1), dtype=np.float32)
    probabilities[ids == 2] = 0.4
    probabilities[(ids == 1) | (ids == 3)] = 0.9

    expected = generate_temporal_dense_observations(
        _frame(),
        _dense(ids, probabilities),
        ("chair", "table", "lamp"),
        _config(maximum_observations=2),
    )
    repeated = generate_temporal_dense_observations(
        _frame(),
        _dense(ids, probabilities),
        ("chair", "table", "lamp"),
        _config(maximum_observations=2),
    )

    assert [item.semantic_id for item in expected] == [1, 3]
    assert [item.observation_id for item in expected] == [
        2**61 + 7 * 2**20,
        2**61 + 7 * 2**20 + 1,
    ]
    assert [(item.semantic_id, item.bbox_xyxy) for item in repeated] == [
        (item.semantic_id, item.bbox_xyxy) for item in expected
    ]


def test_lift_failure_does_not_consume_capacity() -> None:
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    ids[1, 1] = 1  # area 16, cannot meet min_valid_points 20
    ids[5, 5:7] = 2  # area 32, succeeds

    (item,) = generate_temporal_dense_observations(
        _frame(),
        _dense(ids),
        ("chair", "table", "lamp"),
        _config(maximum_observations=1, pixel_stride=1, min_valid_points=20),
    )

    assert item.semantic_id == 2
    assert item.observation_id == 2**61 + 7 * 2**20


def test_border_contact_counts_distinct_pixels_on_opposite_edges() -> None:
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    ids[:, 5] = 1

    (item,) = generate_temporal_dense_observations(
        _frame(), _dense(ids), ("chair", "table", "lamp"), _config()
    )

    assert item.visible_pixel_count == 480 * 4
    assert item.border_contact_fraction == pytest.approx(8 / (480 * 4))


def test_ignores_background_and_uses_only_top_one_candidate() -> None:
    ids = np.zeros((*NATIVE_SHAPE, 2), dtype=np.int64)
    probs = np.zeros((*NATIVE_SHAPE, 2), dtype=np.float32)
    ids[2, 2] = (1, 2)
    probs[2, 2] = (0.6, 0.3)

    observations = generate_temporal_dense_observations(
        _frame(), _dense(ids, probs), ("chair", "table", "lamp"), _config()
    )

    assert len(observations) == 1
    assert observations[0].semantic_id == 1


@pytest.mark.parametrize(
    ("frame_change", "dense_change", "class_names", "match"),
    [
        ({"frame_id": -1}, {}, ("a", "b", "c"), "frame_id"),
        ({}, {"frame_id": 8}, ("a", "b", "c"), "cache_frame_id"),
        ({}, {}, ["a", "b", "c"], "class_names"),
        ({}, {}, ("a", "b"), "class_names"),
        ({}, {}, ("a", "a", "c"), "unique"),
        ({}, {}, ("a", " ", "c"), "non-empty"),
    ],
)
def test_rejects_invalid_frame_dense_and_vocabulary_contracts(
    frame_change: dict[str, object],
    dense_change: dict[str, object],
    class_names: object,
    match: str,
) -> None:
    frame = _frame()
    for key, value in frame_change.items():
        setattr(frame, key, value)
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    dense = _dense(ids, **dense_change)
    with pytest.raises((TypeError, ValueError), match=match):
        generate_temporal_dense_observations(frame, dense, class_names, _config())  # type: ignore[arg-type]


def test_rejects_wrong_shapes_stride_class_ids_and_observation_id_overflow() -> None:
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    for frame, dense, match in (
        (
            replace(_frame(), depth=np.ones((479, 720), dtype=np.float32)),
            _dense(ids),
            "480",
        ),
        (
            _frame(),
            _dense(np.zeros((240, 360), dtype=np.int64), sample_stride=2),
            "sample_stride",
        ),
        (
            _frame(frame_id=7_000_000_000_000),
            _dense(ids, frame_id=7_000_000_000_000),
            "signed int64",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            generate_temporal_dense_observations(
                frame, dense, ("a", "b", "c"), _config()
            )


def test_does_not_mutate_frame_dense_or_class_names() -> None:
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    ids[1, 1] = 1
    frame = _frame()
    dense = _dense(ids)
    names = ("chair", "table", "lamp")
    depth_before = frame.depth.copy()
    ids_before = dense.class_ids.copy()
    probs_before = dense.probabilities.copy()

    generate_temporal_dense_observations(frame, dense, names, _config())

    assert np.array_equal(frame.depth, depth_before)
    assert np.array_equal(dense.class_ids, ids_before)
    assert np.array_equal(dense.probabilities, probs_before)
    assert names == ("chair", "table", "lamp")
