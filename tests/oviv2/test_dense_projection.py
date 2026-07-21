from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.dense_projection import (
    DenseProjectionResult,
    DenseSemanticConfig,
    DenseSemanticIntegrator,
)
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.evidence import EvidenceConfig, SparseEvidenceStore


def make_frame(
    *,
    depth: np.ndarray | None = None,
    frame_id: int = 7,
    pose: np.ndarray | None = None,
    fx: float = 1.0,
    fy: float = 1.0,
    cx: float = 0.0,
    cy: float = 0.0,
) -> Frame:
    if depth is None:
        depth = np.full((3, 3), 2.0, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    return Frame(
        frame_id=frame_id,
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        depth=depth,
        pose=np.eye(4, dtype=np.float64) if pose is None else pose,
        intrinsics=CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
        ),
        timestamp=1.0,
    )


def make_dense_frame(
    *,
    image_shape: tuple[int, int] = (3, 3),
    stride: int = 1,
    source_frame_id: int = 7,
    class_count: int = 3,
    class_ids: np.ndarray | list[object] | None = None,
    probabilities: np.ndarray | list[object] | None = None,
    entropy: np.ndarray | list[object] | float | None = None,
) -> DenseSemanticFrame:
    sampled_shape = (
        math.ceil(image_shape[0] / stride),
        math.ceil(image_shape[1] / stride),
    )
    if class_ids is None:
        ids = np.ones((*sampled_shape, 1), dtype=np.int64)
    else:
        ids = np.asarray(class_ids, dtype=np.int64)
    if probabilities is None:
        probs = np.ones(ids.shape, dtype=np.float32)
    else:
        probs = np.asarray(probabilities, dtype=np.float32)
    if entropy is None:
        entropy_values = np.zeros(sampled_shape, dtype=np.float32)
    else:
        entropy_values = np.broadcast_to(
            np.asarray(entropy, dtype=np.float32),
            sampled_shape,
        ).copy()
    second = probs[..., 1] if probs.shape[-1] > 1 else 0.0
    margin = np.asarray(probs[..., 0] - second, dtype=np.float32)
    return DenseSemanticFrame(
        cache_frame_id=2,
        source_frame_id=source_frame_id,
        image_shape=image_shape,
        sample_stride=stride,
        class_count=class_count,
        class_ids=ids,
        probabilities=probs,
        entropy=entropy_values,
        margin=margin,
    )


def test_config_is_frozen_and_normalizes_finite_numeric_values() -> None:
    config = DenseSemanticConfig(
        voxel_size_m=np.float32(0.5),
        integration_radius_m=4,
        minimum_probability=np.float64(0.2),
        minimum_quality=0,
        entropy_power=2,
        view_angle_power=3,
    )

    assert config == DenseSemanticConfig(0.5, 4.0, 0.2, 0.0, 2.0, 3.0)
    assert all(type(value) is float for value in config.__dict__.values())
    with pytest.raises(FrozenInstanceError):
        config.voxel_size_m = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("voxel_size_m", 0.0),
        ("voxel_size_m", -1.0),
        ("voxel_size_m", np.nan),
        ("voxel_size_m", np.inf),
        ("voxel_size_m", True),
        ("voxel_size_m", "0.5"),
        ("integration_radius_m", 0.0),
        ("integration_radius_m", -1.0),
        ("integration_radius_m", np.nan),
        ("minimum_probability", -0.01),
        ("minimum_probability", 1.01),
        ("minimum_probability", np.nan),
        ("minimum_probability", True),
        ("minimum_quality", -0.01),
        ("minimum_quality", 1.01),
        ("minimum_quality", np.inf),
        ("entropy_power", -0.01),
        ("entropy_power", np.nan),
        ("entropy_power", True),
        ("view_angle_power", -0.01),
        ("view_angle_power", np.inf),
    ],
)
def test_config_rejects_invalid_values(field_name: str, invalid: object) -> None:
    values: dict[str, object] = {"voxel_size_m": 0.5}
    values[field_name] = invalid

    with pytest.raises(ValueError, match=field_name):
        DenseSemanticConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    [
        "voxel_size_m",
        "integration_radius_m",
        "minimum_probability",
        "minimum_quality",
        "entropy_power",
        "view_angle_power",
    ],
)
def test_config_wraps_huge_integer_conversion_overflow(field_name: str) -> None:
    values: dict[str, object] = {"voxel_size_m": 0.5}
    values[field_name] = 10**400

    with pytest.raises(ValueError, match=field_name):
        DenseSemanticConfig(**values)  # type: ignore[arg-type]


def test_integrator_projects_topk_probabilities_to_expected_voxel() -> None:
    frame = make_frame(depth=np.full((4, 4), 2.0, dtype=np.float32))
    dense = make_dense_frame(
        image_shape=(4, 4),
        stride=4,
        class_ids=[[[2, 3]]],
        probabilities=[[[0.8, 0.2]]],
    )
    store = SparseEvidenceStore(EvidenceConfig(block_resolution=8, semantic_top_k=4))

    result = DenseSemanticIntegrator(DenseSemanticConfig(voxel_size_m=0.5)).integrate(
        frame,
        dense,
        store,
        revision=1,
    )

    assert result == DenseProjectionResult(1, 1, 1)
    candidates = store.semantic_candidates((0, 0, 4))
    assert [item.label_id for item in candidates] == [2, 3]
    assert candidates[0].support == pytest.approx(0.8)
    assert candidates[1].support == pytest.approx(0.2)


def test_integrator_supports_validated_class_specific_entropy_powers() -> None:
    frame = make_frame(depth=np.full((4, 4), 2.0, dtype=np.float32))
    dense = make_dense_frame(
        image_shape=(4, 4),
        stride=4,
        class_count=4,
        class_ids=[[[1, 4]]],
        probabilities=[[[0.6, 0.3]]],
        entropy=np.log(4.0) * 0.5,
    )
    store = SparseEvidenceStore(EvidenceConfig(block_resolution=8, semantic_top_k=4))

    DenseSemanticIntegrator(
        DenseSemanticConfig(voxel_size_m=0.5, entropy_power=1.0)
    ).integrate(
        frame,
        dense,
        store,
        revision=1,
        entropy_power_by_class={1: 2.0},
    )

    candidates = {item.label_id: item.support for item in store.semantic_candidates((0, 0, 4))}
    assert candidates[1] == pytest.approx(0.6 * 0.5**2)
    assert candidates[4] == pytest.approx(0.3 * 0.5)


@pytest.mark.parametrize(
    "mapping",
    ({0: 2.0}, {4: -1.0}, {4: np.inf}, {True: 2.0}, {4: True}),
)
def test_integrator_rejects_invalid_class_entropy_mapping(mapping: dict) -> None:
    with pytest.raises((TypeError, ValueError), match="entropy|class"):
        DenseSemanticIntegrator(DenseSemanticConfig(voxel_size_m=0.5)).integrate(
            make_frame(),
            make_dense_frame(),
            SparseEvidenceStore(),
            revision=1,
            entropy_power_by_class=mapping,
        )


def test_projection_result_is_frozen() -> None:
    result = DenseProjectionResult(3, 2, 1)

    with pytest.raises(FrozenInstanceError):
        result.valid_pixel_count = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("sampled_pixel_count", -1),
        ("sampled_pixel_count", True),
        ("valid_pixel_count", 1.5),
        ("updated_voxel_count", -1),
    ],
)
def test_projection_result_rejects_invalid_counts(field_name: str, invalid: object) -> None:
    values: dict[str, object] = {
        "sampled_pixel_count": 1,
        "valid_pixel_count": 1,
        "updated_voxel_count": 1,
    }
    values[field_name] = invalid

    with pytest.raises(ValueError, match=field_name):
        DenseProjectionResult(**values)  # type: ignore[arg-type]


def test_integrator_rejects_future_or_stale_dense_frame_without_updates() -> None:
    frame = make_frame(frame_id=10)
    dense = make_dense_frame(source_frame_id=20)
    store = SparseEvidenceStore()

    with pytest.raises(ValueError, match="source_frame_id|frame"):
        DenseSemanticIntegrator(DenseSemanticConfig(0.5)).integrate(
            frame,
            dense,
            store,
            revision=1,
        )

    assert store.allocated_block_count == 0


@pytest.mark.parametrize("revision", [-1, True, 1.0, np.iinfo(np.int64).max + 1])
def test_integrator_rejects_invalid_revision_without_updates(revision: object) -> None:
    store = SparseEvidenceStore()

    with pytest.raises(ValueError, match="revision"):
        DenseSemanticIntegrator(DenseSemanticConfig(0.5)).integrate(
            make_frame(),
            make_dense_frame(),
            store,
            revision=revision,  # type: ignore[arg-type]
        )

    assert store.allocated_block_count == 0


@pytest.mark.parametrize("broken", ["rgb", "depth", "intrinsics", "sample_grid"])
def test_integrator_rejects_inconsistent_shapes_before_updating(broken: str) -> None:
    frame = make_frame()
    dense = make_dense_frame()
    if broken == "rgb":
        frame.rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    elif broken == "depth":
        frame.depth = np.ones((2, 3), dtype=np.float32)
    elif broken == "intrinsics":
        frame.intrinsics.width = 4
    else:
        object.__setattr__(dense, "entropy", np.zeros((1, 1), dtype=np.float32))
    store = SparseEvidenceStore()

    with pytest.raises(ValueError, match="shape|image|sample|intrinsics"):
        DenseSemanticIntegrator(DenseSemanticConfig(0.5)).integrate(
            frame,
            dense,
            store,
            revision=1,
        )

    assert store.allocated_block_count == 0


@pytest.mark.parametrize("broken", ["pose_shape", "pose_finite", "focal", "principal"])
def test_integrator_rejects_invalid_camera_geometry_before_updating(broken: str) -> None:
    frame = make_frame()
    if broken == "pose_shape":
        frame.pose = np.eye(3)
    elif broken == "pose_finite":
        frame.pose[0, 0] = np.nan
    elif broken == "focal":
        frame.intrinsics.fx = 0.0
    else:
        frame.intrinsics.cx = np.inf
    store = SparseEvidenceStore()

    with pytest.raises(ValueError, match="pose|intrinsics"):
        DenseSemanticIntegrator(DenseSemanticConfig(0.5)).integrate(
            frame,
            make_dense_frame(),
            store,
            revision=1,
        )

    assert store.allocated_block_count == 0


@pytest.mark.parametrize("broken", ["intrinsics", "pose"])
def test_integrator_wraps_camera_numeric_conversion_overflow(broken: str) -> None:
    frame = make_frame()
    if broken == "intrinsics":
        frame.intrinsics.fx = 10**400
    else:
        pose = np.asarray(frame.pose, dtype=object)
        pose[0, 0] = 10**400
        frame.pose = pose
    store = SparseEvidenceStore()

    with pytest.raises(ValueError, match="intrinsics|pose"):
        DenseSemanticIntegrator(DenseSemanticConfig(0.5)).integrate(
            frame,
            make_dense_frame(),
            store,
            revision=1,
        )

    assert store.allocated_block_count == 0


def test_integrator_drops_invalid_depth_high_entropy_and_out_of_radius() -> None:
    frame = make_frame(depth=np.asarray([[np.nan, 1.0, 9.0]], dtype=np.float32))
    dense = make_dense_frame(
        image_shape=(1, 3),
        class_ids=[[[1], [1], [1]]],
        probabilities=[[[1.0], [1.0], [1.0]]],
        entropy=[[0.0, math.log(3), 0.0]],
    )
    store = SparseEvidenceStore()

    result = DenseSemanticIntegrator(
        DenseSemanticConfig(
            voxel_size_m=0.5,
            integration_radius_m=3.0,
            minimum_quality=0.01,
        )
    ).integrate(frame, dense, store, revision=1)

    assert result == DenseProjectionResult(3, 1, 0)
    assert store.allocated_block_count == 0


@pytest.mark.parametrize("invalid_depth", [0.0, -1.0, np.inf, -np.inf])
def test_integrator_rejects_nonpositive_or_nonfinite_depth_samples(
    invalid_depth: float,
) -> None:
    frame = make_frame(depth=np.asarray([[invalid_depth]], dtype=np.float32))
    dense = make_dense_frame(image_shape=(1, 1), class_ids=[[[1]]])

    result = DenseSemanticIntegrator(DenseSemanticConfig(0.5)).integrate(
        frame,
        dense,
        SparseEvidenceStore(),
        revision=1,
    )

    assert result == DenseProjectionResult(1, 0, 0)


def test_integrator_samples_nondivisible_last_row_and_column() -> None:
    depth = np.full((5, 6), np.nan, dtype=np.float32)
    depth[np.ix_([0, 4], [0, 4])] = 1.0
    frame = make_frame(depth=depth)
    dense = make_dense_frame(
        image_shape=(5, 6),
        stride=4,
        class_count=1,
        class_ids=np.ones((2, 2, 1), dtype=np.int64),
    )
    store = SparseEvidenceStore()

    result = DenseSemanticIntegrator(
        DenseSemanticConfig(voxel_size_m=1.0, integration_radius_m=10.0)
    ).integrate(frame, dense, store, revision=1)

    assert result == DenseProjectionResult(4, 4, 4)
    assert {
        key
        for key in ((0, 0, 1), (4, 0, 1), (0, 4, 1), (4, 4, 1))
        if store.semantic_candidates(key)
    } == {(0, 0, 1), (4, 0, 1), (0, 4, 1), (4, 4, 1)}


def test_integrator_applies_camera_to_world_pose_before_quantization() -> None:
    pose = np.asarray(
        [
            [0.0, -1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    frame = make_frame(depth=np.ones((1, 2), dtype=np.float32), pose=pose)
    dense = make_dense_frame(
        image_shape=(1, 2),
        class_count=1,
        class_ids=[[[0], [1]]],
        probabilities=[[[0.0], [1.0]]],
    )
    store = SparseEvidenceStore()

    DenseSemanticIntegrator(DenseSemanticConfig(1.0)).integrate(
        frame,
        dense,
        store,
        revision=1,
    )

    assert store.semantic_candidates((2, 0, 1))[0].support == pytest.approx(1.0)


def test_integration_radius_uses_camera_point_distance_not_only_z_depth() -> None:
    frame = make_frame(
        depth=np.full((1, 2), 2.0, dtype=np.float32),
        fx=0.25,
    )
    dense = make_dense_frame(
        image_shape=(1, 2),
        class_count=1,
        class_ids=[[[0], [1]]],
        probabilities=[[[0.0], [1.0]]],
    )
    store = SparseEvidenceStore()

    result = DenseSemanticIntegrator(
        DenseSemanticConfig(1.0, integration_radius_m=3.0)
    ).integrate(frame, dense, store, revision=1)

    assert result == DenseProjectionResult(2, 1, 0)
    assert store.allocated_block_count == 0


def test_single_class_entropy_quality_is_defined_as_one() -> None:
    frame = make_frame(depth=np.asarray([[2.0]], dtype=np.float32))
    dense = make_dense_frame(
        image_shape=(1, 1),
        class_count=1,
        class_ids=[[[1]]],
        probabilities=[[[1.0]]],
        entropy=0.0,
    )
    store = SparseEvidenceStore()

    DenseSemanticIntegrator(
        DenseSemanticConfig(1.0, entropy_power=7.0)
    ).integrate(frame, dense, store, revision=1)

    assert store.semantic_candidates((0, 0, 2))[0].support == pytest.approx(1.0)


def test_entropy_quality_uses_normalized_entropy_and_power() -> None:
    frame = make_frame(depth=np.asarray([[2.0]], dtype=np.float32))
    dense = make_dense_frame(
        image_shape=(1, 1),
        class_count=2,
        class_ids=[[[1]]],
        probabilities=[[[1.0]]],
        entropy=0.5 * math.log(2),
    )
    store = SparseEvidenceStore()

    DenseSemanticIntegrator(
        DenseSemanticConfig(1.0, entropy_power=2.0)
    ).integrate(frame, dense, store, revision=1)

    assert store.semantic_candidates((0, 0, 2))[0].support == pytest.approx(0.25)


def test_front_facing_normal_is_oriented_toward_camera() -> None:
    frame = make_frame(
        depth=np.full((3, 3), 2.0, dtype=np.float32),
        cx=1.0,
        cy=1.0,
    )
    ids = np.zeros((3, 3, 1), dtype=np.int64)
    probs = np.zeros((3, 3, 1), dtype=np.float32)
    ids[1, 1, 0] = 1
    probs[1, 1, 0] = 1.0
    dense = make_dense_frame(
        class_count=1,
        class_ids=ids,
        probabilities=probs,
    )
    store = SparseEvidenceStore()

    DenseSemanticIntegrator(DenseSemanticConfig(1.0)).integrate(
        frame,
        dense,
        store,
        revision=1,
    )

    assert store.semantic_candidates((0, 0, 2))[0].support == pytest.approx(1.0)


def test_slanted_normal_uses_surface_to_camera_viewing_ray() -> None:
    depth = np.tile(np.asarray([1.0, 2.0, 3.0], dtype=np.float32), (3, 1))
    frame = make_frame(depth=depth, cx=1.0, cy=1.0)
    ids = np.zeros((3, 3, 1), dtype=np.int64)
    probs = np.zeros((3, 3, 1), dtype=np.float32)
    ids[1, 1, 0] = 1
    probs[1, 1, 0] = 1.0
    dense = make_dense_frame(
        class_count=1,
        class_ids=ids,
        probabilities=probs,
    )
    store = SparseEvidenceStore()

    DenseSemanticIntegrator(DenseSemanticConfig(1.0)).integrate(
        frame,
        dense,
        store,
        revision=1,
    )

    assert store.semantic_candidates((0, 0, 2))[0].support == pytest.approx(
        2.0 / math.sqrt(5.0)
    )


def test_radius_filter_does_not_remove_finite_normal_neighbors() -> None:
    depth = np.tile(np.asarray([1.0, 2.0, 3.0], dtype=np.float32), (3, 1))
    frame = make_frame(depth=depth, cx=1.0, cy=1.0)
    ids = np.zeros((3, 3, 1), dtype=np.int64)
    probs = np.zeros((3, 3, 1), dtype=np.float32)
    ids[1, 1, 0] = 1
    probs[1, 1, 0] = 1.0
    dense = make_dense_frame(
        class_count=1,
        class_ids=ids,
        probabilities=probs,
    )
    store = SparseEvidenceStore()

    result = DenseSemanticIntegrator(
        DenseSemanticConfig(1.0, integration_radius_m=2.1)
    ).integrate(frame, dense, store, revision=1)

    assert result.valid_pixel_count == 4
    assert store.semantic_candidates((0, 0, 2))[0].support == pytest.approx(
        2.0 / math.sqrt(5.0)
    )


def test_missing_valid_normal_neighbors_falls_back_to_unit_view_quality() -> None:
    depth = np.full((3, 3), np.nan, dtype=np.float32)
    depth[1, 1] = 2.0
    frame = make_frame(depth=depth, cx=1.0, cy=1.0)
    ids = np.zeros((3, 3, 1), dtype=np.int64)
    probs = np.zeros((3, 3, 1), dtype=np.float32)
    ids[1, 1, 0] = 1
    probs[1, 1, 0] = 1.0
    dense = make_dense_frame(
        class_count=1,
        class_ids=ids,
        probabilities=probs,
    )
    store = SparseEvidenceStore()

    DenseSemanticIntegrator(DenseSemanticConfig(1.0)).integrate(
        frame,
        dense,
        store,
        revision=1,
    )

    assert store.semantic_candidates((0, 0, 2))[0].support == pytest.approx(1.0)


def test_probability_and_quality_thresholds_filter_support() -> None:
    frame = make_frame(depth=np.asarray([[1.0, 1.0]], dtype=np.float32))
    dense = make_dense_frame(
        image_shape=(1, 2),
        class_count=2,
        class_ids=[[[1], [2]]],
        probabilities=[[[0.49], [0.8]]],
        entropy=[[0.0, 0.75 * math.log(2)]],
    )
    store = SparseEvidenceStore()

    result = DenseSemanticIntegrator(
        DenseSemanticConfig(
            voxel_size_m=1.0,
            minimum_probability=0.5,
            minimum_quality=0.3,
        )
    ).integrate(frame, dense, store, revision=1)

    assert result == DenseProjectionResult(2, 2, 0)
    assert store.allocated_block_count == 0


def test_duplicate_pixels_in_a_voxel_aggregate_support_once_per_pixel() -> None:
    frame = make_frame(depth=np.ones((1, 2), dtype=np.float32))
    dense = make_dense_frame(
        image_shape=(1, 2),
        class_count=2,
        class_ids=[[[1], [1]]],
        probabilities=[[[0.6], [0.6]]],
    )
    store = SparseEvidenceStore()

    result = DenseSemanticIntegrator(DenseSemanticConfig(10.0)).integrate(
        frame,
        dense,
        store,
        revision=1,
    )

    assert result.updated_voxel_count == 1
    assert store.semantic_candidates((0, 0, 0))[0].support == pytest.approx(1.2)


def test_duplicate_topk_label_at_one_pixel_is_not_amplified() -> None:
    frame = make_frame(depth=np.asarray([[1.0]], dtype=np.float32))
    dense = make_dense_frame(
        image_shape=(1, 1),
        class_count=2,
        class_ids=[[[1, 2]]],
        probabilities=[[[0.6, 0.4]]],
    )
    object.__setattr__(dense, "class_ids", np.asarray([[[1, 1]]], dtype=np.int64))
    store = SparseEvidenceStore()

    DenseSemanticIntegrator(DenseSemanticConfig(1.0)).integrate(
        frame,
        dense,
        store,
        revision=1,
    )

    candidates = store.semantic_candidates((0, 0, 1))
    assert len(candidates) == 1
    assert candidates[0].support == pytest.approx(0.6)


def test_revision_rollback_is_preflighted_without_partial_updates() -> None:
    frame = make_frame(depth=np.ones((1, 2), dtype=np.float32))
    dense = make_dense_frame(
        image_shape=(1, 2),
        class_count=1,
        class_ids=[[[1], [1]]],
    )
    store = SparseEvidenceStore()
    store.update_semantic((1, 0, 1), 1, 2.0, revision=5)

    with pytest.raises(ValueError, match="revision"):
        DenseSemanticIntegrator(DenseSemanticConfig(1.0)).integrate(
            frame,
            dense,
            store,
            revision=4,
        )

    assert store.semantic_candidates((0, 0, 1)) == ()
    existing = store.semantic_candidates((1, 0, 1))[0]
    assert existing.support == pytest.approx(2.0)
    assert existing.revision == 5


def test_integrator_documents_external_store_serialization_contract() -> None:
    docstring = DenseSemanticIntegrator.integrate.__doc__ or ""

    assert "not thread-safe" in docstring
    assert "externally serialized" in docstring


def test_existing_label_accumulates_before_capacity_one_admission() -> None:
    frame = make_frame(depth=np.asarray([[1.0]], dtype=np.float32))
    dense = make_dense_frame(
        image_shape=(1, 1),
        class_count=2,
        class_ids=[[[1, 2]]],
        probabilities=[[[0.6, 0.4]]],
    )
    store = SparseEvidenceStore(EvidenceConfig(semantic_top_k=1))
    store.update_semantic((0, 0, 1), 2, 0.3, revision=0)

    DenseSemanticIntegrator(DenseSemanticConfig(1.0)).integrate(
        frame,
        dense,
        store,
        revision=1,
    )

    candidates = store.semantic_candidates((0, 0, 1))
    assert [(item.label_id, item.support) for item in candidates] == [
        (2, pytest.approx(0.7))
    ]


def test_existing_labels_accumulate_before_capacity_two_admission() -> None:
    frame = make_frame(depth=np.asarray([[1.0]], dtype=np.float32))
    dense = make_dense_frame(
        image_shape=(1, 1),
        class_count=4,
        class_ids=[[[1, 2, 3]]],
        probabilities=[[[0.3, 0.25, 0.2]]],
    )
    store = SparseEvidenceStore(EvidenceConfig(semantic_top_k=2))
    store.update_semantic((0, 0, 1), 2, 0.25, revision=0)
    store.update_semantic((0, 0, 1), 4, 0.6, revision=0)

    DenseSemanticIntegrator(DenseSemanticConfig(1.0)).integrate(
        frame,
        dense,
        store,
        revision=1,
    )

    candidates = store.semantic_candidates((0, 0, 1))
    assert [(item.label_id, item.support) for item in candidates] == [
        (4, pytest.approx(0.6)),
        (2, pytest.approx(0.5)),
    ]


def test_new_label_ties_use_stable_id_after_existing_candidates() -> None:
    frame = make_frame(depth=np.asarray([[1.0]], dtype=np.float32))
    dense = make_dense_frame(
        image_shape=(1, 1),
        class_count=4,
        class_ids=[[[1, 2, 3]]],
        probabilities=[[[0.3, 0.3, 0.3]]],
    )
    store = SparseEvidenceStore(EvidenceConfig(semantic_top_k=2))
    store.update_semantic((0, 0, 1), 4, 0.6, revision=0)

    DenseSemanticIntegrator(DenseSemanticConfig(1.0)).integrate(
        frame,
        dense,
        store,
        revision=1,
    )

    candidates = store.semantic_candidates((0, 0, 1))
    assert [(item.label_id, item.support) for item in candidates] == [
        (4, pytest.approx(0.6)),
        (1, pytest.approx(0.3)),
    ]


def test_malformed_class_id_is_rejected_without_partial_updates() -> None:
    dense = make_dense_frame()
    invalid_ids = np.array(dense.class_ids, copy=True)
    invalid_ids[-1, -1, 0] = dense.class_count + 1
    object.__setattr__(dense, "class_ids", invalid_ids)
    store = SparseEvidenceStore()

    with pytest.raises(ValueError, match="class_ids"):
        DenseSemanticIntegrator(DenseSemanticConfig(1.0)).integrate(
            make_frame(),
            dense,
            store,
            revision=1,
        )

    assert store.allocated_block_count == 0


def test_out_of_int64_voxel_key_is_rejected_without_wrapping_or_updates() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(2**63)
    frame = make_frame(depth=np.asarray([[1.0]], dtype=np.float32), pose=pose)
    dense = make_dense_frame(image_shape=(1, 1))
    store = SparseEvidenceStore()

    with pytest.raises(ValueError, match="voxel keys|int64"):
        DenseSemanticIntegrator(DenseSemanticConfig(1.0)).integrate(
            frame,
            dense,
            store,
            revision=1,
        )

    assert store.allocated_block_count == 0


def test_no_valid_samples_returns_stable_empty_result() -> None:
    frame = make_frame(depth=np.full((2, 2), np.nan, dtype=np.float32))
    dense = make_dense_frame(image_shape=(2, 2))
    store = SparseEvidenceStore()

    result = DenseSemanticIntegrator(DenseSemanticConfig(1.0)).integrate(
        frame,
        dense,
        store,
        revision=0,
    )

    assert result == DenseProjectionResult(4, 0, 0)
    assert store.allocated_block_count == 0
