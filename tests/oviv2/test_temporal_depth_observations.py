from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import src.oviv2.temporal_depth_observations as temporal_depth_observations
from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.observations import LiftedMask, ObservationKind
from src.oviv2.temporal_config import TemporalIdentityConfig
from src.oviv2.temporal_dense_observations import (
    TemporalDenseObservationConfig,
    generate_temporal_dense_observations,
)
from src.oviv2.temporal_depth_observations import (
    TemporalDepthObservationConfig,
    generate_temporal_depth_observations,
)
from src.oviv2.temporal_identity import IdentityMemoryBank
from src.oviv2.temporal_proposals import RecoveredTemporalProposal
from src.oviv2.temporal_runtime import _proposal_observation

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
    ids: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    *,
    frame_id: int = 7,
    source_frame_id: int | None = None,
    class_count: int = 3,
    sample_stride: int = 4,
) -> DenseSemanticFrame:
    if ids is None:
        ids = np.ones(NATIVE_SHAPE, dtype=np.int64)
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
        source_frame_id=(frame_id + 100 if source_frame_id is None else source_frame_id),
        image_shape=IMAGE_SHAPE,
        sample_stride=sample_stride,
        class_count=class_count,
        class_ids=ids,
        probabilities=probabilities,
        entropy=np.zeros(ids.shape[:2], dtype=np.float32),
        margin=np.asarray(probabilities[..., 0] - second, dtype=np.float32),
    )


def _config(**changes: object) -> TemporalDepthObservationConfig:
    values: dict[str, object] = {
        "edge_threshold_m": 0.05,
        "minimum_area_px": 1,
        "maximum_area_px": 62_208,
        "plane_minimum_area_px": 50_000,
        "planar_rmse_threshold_m": 0.02,
        "semantic_minimum_votes": 1,
        "semantic_minimum_fraction": 0.5,
        "semantic_minimum_probability": 0.1,
        "maximum_observations": 32,
        "maximum_unknown_observations": 32,
        "depth_max_m": 10.0,
        "voxel_size_m": 0.05,
        "pixel_stride": 1,
        "min_valid_points": 1,
    }
    values.update(changes)
    return TemporalDepthObservationConfig(**values)


def _split_depth() -> np.ndarray:
    depth = np.full(IMAGE_SHAPE, np.nan, dtype=np.float32)
    depth[10:14, 10:14] = 1.0
    depth[20:24, 20:24] = 2.0
    return depth


def test_config_is_frozen_and_strictly_validated() -> None:
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.depth_max_m = 8.0  # type: ignore[misc]

    for changes, match in (
        ({"edge_threshold_m": True}, "edge_threshold_m"),
        ({"edge_threshold_m": 0.0}, "edge_threshold_m"),
        ({"minimum_area_px": 0}, "minimum_area_px"),
        ({"maximum_area_px": True}, "maximum_area_px"),
        ({"maximum_area_px": 0}, "maximum_area_px"),
        ({"plane_minimum_area_px": 0}, "plane_minimum_area_px"),
        ({"planar_rmse_threshold_m": -0.01}, "planar_rmse_threshold_m"),
        ({"semantic_minimum_votes": 0}, "semantic_minimum_votes"),
        ({"semantic_minimum_fraction": -0.01}, "semantic_minimum_fraction"),
        ({"semantic_minimum_fraction": 1.01}, "semantic_minimum_fraction"),
        ({"semantic_minimum_probability": -0.01}, "semantic_minimum_probability"),
        ({"semantic_minimum_probability": 1.01}, "semantic_minimum_probability"),
        ({"maximum_observations": 2**20 + 1}, "maximum_observations"),
        ({"maximum_unknown_observations": 0}, "maximum_unknown_observations"),
        ({"depth_max_m": np.nan}, "depth_max_m"),
        ({"voxel_size_m": 0.0}, "voxel_size_m"),
        ({"pixel_stride": 0}, "pixel_stride"),
        ({"min_valid_points": 0}, "min_valid_points"),
    ):
        with pytest.raises((TypeError, ValueError), match=match):
            _config(**changes)


def test_depth_edges_form_four_connected_components_and_preserve_inputs() -> None:
    depth = _split_depth()
    frame = _frame(depth=depth)
    dense = _dense()
    depth_before = depth.copy()
    ids_before = dense.class_ids.copy()

    observations = generate_temporal_depth_observations(
        frame, dense, (" Chair ", "table", "lamp"), _config()
    )

    assert [item.bbox_xyxy for item in observations] == [
        (10.0, 10.0, 14.0, 14.0),
        (20.0, 20.0, 24.0, 24.0),
    ]
    assert all(item.kind is ObservationKind.OBJECT for item in observations)
    assert all(item.label == "chair" for item in observations)
    np.testing.assert_array_equal(depth, depth_before)
    np.testing.assert_array_equal(dense.class_ids, ids_before)
    assert depth.dtype == np.float32


def test_depth_diff_edge_pixel_is_removed_and_diagonal_is_not_connected() -> None:
    depth = np.full(IMAGE_SHAPE, np.nan, dtype=np.float32)
    depth[1, 1] = 1.0
    depth[2, 2] = 1.0

    observations = generate_temporal_depth_observations(
        _frame(depth=depth), _dense(), ("a", "b", "c"), _config()
    )

    assert [item.visible_pixel_count for item in observations] == [1, 1]
    assert [item.bbox_xyxy for item in observations] == [
        (1.0, 1.0, 2.0, 2.0),
        (2.0, 2.0, 3.0, 3.0),
    ]


def test_depth_edge_comparison_is_strictly_greater_than_threshold() -> None:
    depth = np.full(IMAGE_SHAPE, np.nan, dtype=np.float64)
    depth[5, 5:7] = (1.0, 1.125)
    retained = generate_temporal_depth_observations(
        _frame(depth=depth),
        _dense(),
        ("a", "b", "c"),
        _config(edge_threshold_m=0.125),
    )
    assert [item.visible_pixel_count for item in retained] == [2]

    depth[5, 6] = 1.25
    removed = generate_temporal_depth_observations(
        _frame(depth=depth),
        _dense(),
        ("a", "b", "c"),
        _config(edge_threshold_m=0.125),
    )
    assert [item.visible_pixel_count for item in removed] == [1]


def test_area_boundaries_are_inclusive_and_invalid_depth_is_excluded() -> None:
    depth = np.full(IMAGE_SHAPE, np.nan, dtype=np.float32)
    depth[3:5, 3:5] = 1.0
    depth[9:11, 9:12] = 2.0
    observations = generate_temporal_depth_observations(
        _frame(depth=depth),
        _dense(),
        ("a", "b", "c"),
        _config(minimum_area_px=4, maximum_area_px=6),
    )
    assert [item.visible_pixel_count for item in observations] == [6, 4]


def test_plane_fit_rejects_low_rmse_and_keeps_nonplanar_component() -> None:
    plane = np.full(IMAGE_SHAPE, 2.0, dtype=np.float32)
    rejected = generate_temporal_depth_observations(
        _frame(depth=plane),
        _dense(),
        ("a", "b", "c"),
        _config(maximum_area_px=IMAGE_SHAPE[0] * IMAGE_SHAPE[1], plane_minimum_area_px=100),
    )
    assert rejected == ()

    nonplanar = np.full(IMAGE_SHAPE, np.nan, dtype=np.float32)
    rows, columns = np.indices((20, 20))
    nonplanar[30:50, 30:50] = 2.0 + ((rows + columns) % 2) * 0.04
    accepted = generate_temporal_depth_observations(
        _frame(depth=nonplanar),
        _dense(),
        ("a", "b", "c"),
        _config(
            edge_threshold_m=0.05,
            plane_minimum_area_px=100,
            planar_rmse_threshold_m=0.01,
        ),
    )
    assert len(accepted) == 1


def test_semantic_majority_thresholds_and_unknown_fallback() -> None:
    depth = np.full(IMAGE_SHAPE, np.nan, dtype=np.float32)
    depth[0:12, 0:12] = 2.0
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    probabilities = np.zeros(NATIVE_SHAPE, dtype=np.float32)
    ids[0, 0] = 2
    ids[0, 1] = 2
    ids[1, 0] = 1
    probabilities[0, 0] = 0.8
    probabilities[0, 1] = 0.6
    probabilities[1, 0] = 0.9

    (known,) = generate_temporal_depth_observations(
        _frame(depth=depth),
        _dense(ids, probabilities),
        ("chair", "Side_Table", "lamp"),
        _config(
            semantic_minimum_votes=2,
            semantic_minimum_fraction=2 / 9,
            semantic_minimum_probability=0.7,
        ),
    )
    assert (known.semantic_id, known.label, known.confidence) == pytest.approx(
        (2, "side-table", 0.7)
    )

    (unknown,) = generate_temporal_depth_observations(
        _frame(depth=depth),
        _dense(ids, probabilities),
        ("chair", "table", "lamp"),
        _config(semantic_minimum_votes=3),
    )
    assert unknown.semantic_id == 0
    assert unknown.label == "unknown"
    assert unknown.confidence == 0.0


def test_sparse_positive_semantic_votes_do_not_label_a_large_component() -> None:
    depth = np.full(IMAGE_SHAPE, np.nan, dtype=np.float32)
    depth[0:40, 0:40] = 2.0
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    probabilities = np.zeros(NATIVE_SHAPE, dtype=np.float32)
    ids[0, 0] = 2
    probabilities[0, 0] = 0.99

    (item,) = generate_temporal_depth_observations(
        _frame(depth=depth),
        _dense(ids, probabilities),
        ("chair", "table", "lamp"),
        _config(semantic_minimum_fraction=0.5),
    )

    assert item.semantic_id == 0
    assert item.label == "unknown"
    assert item.confidence == 0.0


def test_unknown_cap_retains_known_candidates_and_skips_excess_unknowns() -> None:
    depth = np.full(IMAGE_SHAPE, np.nan, dtype=np.float32)
    depth[20:24, 20:28] = 1.0
    depth[40:44, 40:44] = 2.0
    depth[60:64, 60:64] = 3.0
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    probabilities = np.zeros(NATIVE_SHAPE, dtype=np.float32)
    ids[15, 15] = 1
    probabilities[15, 15] = 0.9

    observations = generate_temporal_depth_observations(
        _frame(depth=depth),
        _dense(ids, probabilities),
        ("chair", "table", "lamp"),
        _config(
            maximum_observations=3,
            maximum_unknown_observations=1,
        ),
    )

    assert [item.semantic_id for item in observations] == [0, 1]


def test_failed_unknown_lift_does_not_consume_unknown_cap_or_total_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = np.full(IMAGE_SHAPE, np.nan, dtype=np.float32)
    depth[8:14, 8:16] = 1.0  # area 48, unknown, forced lift failure
    depth[24:29, 24:32] = 2.0  # area 40, unknown, retained
    depth[40:44, 40:48] = 3.0  # area 32, known, retained
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    probabilities = np.zeros(NATIVE_SHAPE, dtype=np.float32)
    ids[10, 10:12] = 2
    probabilities[10, 10:12] = 0.8
    lift_calls: list[tuple[int, int, int, int]] = []
    original_lift = temporal_depth_observations.lift_mask_to_voxels

    def fail_first_lift(
        frame: Frame,
        mask: np.ndarray,
        *,
        voxel_size_m: float,
        pixel_stride: int,
        min_valid_points: int,
    ) -> LiftedMask | None:
        rows, columns = np.nonzero(mask)
        lift_calls.append(
            (
                int(columns.min()),
                int(rows.min()),
                int(columns.max() + 1),
                int(rows.max() + 1),
            )
        )
        if len(lift_calls) == 1:
            return None
        return original_lift(
            frame,
            mask,
            voxel_size_m=voxel_size_m,
            pixel_stride=pixel_stride,
            min_valid_points=min_valid_points,
        )

    monkeypatch.setattr(
        temporal_depth_observations, "lift_mask_to_voxels", fail_first_lift
    )

    observations = generate_temporal_depth_observations(
        _frame(depth=depth),
        _dense(ids, probabilities),
        ("a", "b", "c"),
        _config(maximum_observations=2, maximum_unknown_observations=1),
    )

    assert lift_calls == [
        (8, 8, 16, 14),
        (24, 24, 32, 29),
        (40, 40, 48, 44),
    ]
    assert [
        (item.observation_id, item.semantic_id, item.label) for item in observations
    ] == [
        (3 * 2**61 + 7 * 2**20, 0, "unknown"),
        (3 * 2**61 + 7 * 2**20 + 1, 2, "b"),
    ]


def test_semantic_votes_use_component_and_sampled_valid_intersection() -> None:
    depth = np.full(IMAGE_SHAPE, np.nan, dtype=np.float32)
    depth[1:9, 1:9] = 2.0
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    ids[0, 0] = 2  # sampled pixel (0, 0) is outside component
    ids[1, 1] = 3  # sampled pixel (4, 4) is inside component

    (item,) = generate_temporal_depth_observations(
        _frame(depth=depth),
        _dense(ids),
        ("a", "b", "c"),
        _config(semantic_minimum_fraction=0.25),
    )
    assert item.semantic_id == 3


def test_sort_cap_ids_bbox_and_border_are_deterministic() -> None:
    depth = np.full(IMAGE_SHAPE, np.nan, dtype=np.float32)
    depth[10:14, 10:14] = 1.0  # area 16, excluded after capacity fills
    depth[20:24, 20:28] = 2.0  # area 32, first accepted
    depth[0:4, 40:48] = 3.0  # area 32, border, second accepted
    ids = np.zeros(NATIVE_SHAPE, dtype=np.int64)
    probabilities = np.zeros(NATIVE_SHAPE, dtype=np.float32)
    ids[3, 3] = 1
    probabilities[3, 3] = 0.99
    ids[5, 5:7] = 2
    probabilities[5, 5:7] = 0.6
    ids[0, 10:12] = 3
    probabilities[0, 10:12] = 0.8
    config = _config(maximum_observations=2, min_valid_points=20)

    first = generate_temporal_depth_observations(
        _frame(depth=depth), _dense(ids, probabilities), ("a", "b", "c"), config
    )
    second = generate_temporal_depth_observations(
        _frame(depth=depth), _dense(ids, probabilities), ("a", "b", "c"), config
    )

    assert [item.semantic_id for item in first] == [3, 2]
    assert [item.observation_id for item in first] == [
        3 * 2**61 + 7 * 2**20,
        3 * 2**61 + 7 * 2**20 + 1,
    ]
    assert first[0].bbox_xyxy == (40.0, 0.0, 48.0, 4.0)
    assert first[0].border_contact_fraction == pytest.approx(8 / 32)
    assert [(item.semantic_id, item.bbox_xyxy) for item in second] == [
        (item.semantic_id, item.bbox_xyxy) for item in first
    ]


@pytest.mark.parametrize(
    ("frame_change", "dense_change", "class_names", "match"),
    [
        ({"frame_id": -1}, {}, ("a", "b", "c"), "frame_id"),
        ({}, {"frame_id": 8}, ("a", "b", "c"), "cache_frame_id"),
        ({}, {"source_frame_id": 999}, ("a", "b", "c"), "source_frame_id"),
        ({}, {"sample_stride": 2}, ("a", "b", "c"), "sample_stride"),
        ({}, {}, ["a", "b", "c"], "class_names"),
        ({}, {}, ("a", "b"), "class_names"),
        ({}, {}, ("a", "A", "c"), "unique"),
    ],
)
def test_rejects_invalid_identity_and_vocabulary(
    frame_change: dict[str, object],
    dense_change: dict[str, object],
    class_names: object,
    match: str,
) -> None:
    frame = _frame(frame_id=int(frame_change.get("frame_id", 7)))
    if dense_change.get("sample_stride") == 2:
        dense = _dense(
            np.ones((240, 360), dtype=np.int64),
            frame_id=7,
            sample_stride=2,
        )
    else:
        dense = _dense(**dense_change)
    with pytest.raises((TypeError, ValueError), match=match):
        generate_temporal_depth_observations(frame, dense, class_names, _config())  # type: ignore[arg-type]


def test_rejects_signed_int64_id_overflow() -> None:
    maximum_frame = ((2**63 - 1) - 3 * 2**61) // 2**20
    frame = _frame(frame_id=maximum_frame + 1)
    dense = _dense(frame_id=maximum_frame + 1)
    with pytest.raises(ValueError, match="int64"):
        generate_temporal_depth_observations(
            frame, dense, ("a", "b", "c"), _config()
        )


def test_depth_ids_do_not_overlap_frontend_dense_or_recovery_namespaces() -> None:
    identity_bank = IdentityMemoryBank(
        TemporalIdentityConfig(4, 10, 0.5, 2.0)
    )
    for frame_id in (0, 17):
        depth = _split_depth()
        frame = _frame(frame_id=frame_id, depth=depth)
        dense = _dense(frame_id=frame_id)
        depth_ids = {
            item.observation_id
            for item in generate_temporal_depth_observations(
                frame, dense, ("a", "b", "c"), _config()
            )
        }
        dense_ids = {
            item.observation_id
            for item in generate_temporal_dense_observations(
                frame,
                dense,
                ("a", "b", "c"),
                TemporalDenseObservationConfig(
                    sample_stride=4,
                    depth_max_m=10.0,
                    minimum_area_px=1,
                    maximum_area_px=62_208,
                    maximum_observations=32,
                    voxel_size_m=0.05,
                    pixel_stride=1,
                    min_valid_points=1,
                ),
            )
        }
        frontend_ids = {frame_id * 1_000_000 + rank for rank in range(32)}
        if frame_id == 0:
            recovery_ids = {2**62 + rank for rank in range(32)}
        else:
            mask = np.ones((1, 1), dtype=bool)
            proposal = RecoveredTemporalProposal(
                0,
                frame_id,
                float(frame_id),
                1,
                mask,
                1,
                (0, 0, 1, 1),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
                1.0,
                frame_id - 1,
                "a" * 64,
                frame_id,
                "b" * 64,
                False,
                None,
                None,
                None,
            )
            recovery_ids = {
                _proposal_observation(
                    proposal, identity_bank, voxel_size_m=0.05, capacity=32
                ).observation_id
            }
        namespaces = (depth_ids, frontend_ids, dense_ids, recovery_ids)
        assert depth_ids
        assert dense_ids
        for left_index, left in enumerate(namespaces):
            for right in namespaces[left_index + 1 :]:
                assert left.isdisjoint(right)
