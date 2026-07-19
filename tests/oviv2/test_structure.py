from __future__ import annotations

import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.observations import FrameObservation, ObservationKind, ReplicaVocabulary
from src.oviv2.structure import (
    DepthStructureConfig,
    DepthStructureFrontend,
    classify_structure_masks,
)


def _frame(size: int = 16) -> Frame:
    return Frame(
        frame_id=7,
        source_frame_id=70,
        rgb=np.zeros((size, size, 3), dtype=np.uint8),
        depth=np.ones((size, size), dtype=np.float32),
        pose=np.eye(4),
        intrinsics=CameraIntrinsics(20.0, 20.0, 7.5, 7.5, size, size),
        timestamp=7.0,
    )


def _vocabulary() -> ReplicaVocabulary:
    return ReplicaVocabulary(
        classes=("wall", "floor", "ceiling", "chair"),
        aliases={},
    )


def _object_observation(mask: np.ndarray) -> FrameObservation:
    return FrameObservation(
        observation_id=7_000_001,
        frame_id=7,
        timestamp=7.0,
        kind=ObservationKind.OBJECT,
        label="chair",
        semantic_id=4,
        confidence=0.9,
        mask=mask,
        bbox_xyxy=(5.0, 5.0, 11.0, 11.0),
        voxel_keys=frozenset({(0, 0, 1)}),
        centroid_xyz=(0.0, 0.0, 1.0),
        bounds_min_xyz=(0.0, 0.0, 1.0),
        bounds_max_xyz=(0.0, 0.0, 1.0),
    )


def test_classification_separates_floor_ceiling_and_wall_by_normal_and_image_half() -> None:
    normal_y = np.zeros((6, 4), dtype=np.float64)
    normal_y[:2] = -0.9
    normal_y[4:] = 0.9

    masks = classify_structure_masks(
        normal_y,
        np.ones_like(normal_y, dtype=bool),
        cy=2.5,
        horizontal_threshold=0.6,
        wall_vertical_threshold=0.5,
    )

    assert masks["ceiling"][:2].all()
    assert not masks["ceiling"][2:].any()
    assert masks["floor"][4:].all()
    assert not masks["floor"][:4].any()
    assert masks["wall"][2:4].all()
    assert sum(int(mask.sum()) for mask in masks.values()) == normal_y.size


def test_constant_depth_plane_emits_only_wall_structure() -> None:
    frontend = DepthStructureFrontend(
        _vocabulary(),
        DepthStructureConfig(
            voxel_size_m=0.1,
            pixel_stride=1,
            min_valid_points=1,
            min_component_pixels=4,
            min_component_fraction=0.0,
            object_exclusion_dilation=0,
        ),
    )

    observations = frontend.observe(_frame(), object_observations=())

    assert {item.kind for item in observations} == {ObservationKind.STRUCTURE}
    assert {item.label for item in observations} == {"wall"}
    assert {item.semantic_id for item in observations} == {1}
    assert all(item.mask.flags.writeable is False for item in observations)


def test_structure_frontend_dilates_and_excludes_object_masks() -> None:
    frame = _frame()
    object_mask = np.zeros(frame.depth.shape, dtype=bool)
    object_mask[6:10, 6:10] = True
    frontend = DepthStructureFrontend(
        _vocabulary(),
        DepthStructureConfig(
            voxel_size_m=0.1,
            pixel_stride=1,
            min_valid_points=1,
            min_component_pixels=4,
            min_component_fraction=0.0,
            object_exclusion_dilation=2,
        ),
    )

    observations = frontend.observe(
        frame,
        object_observations=(_object_observation(object_mask),),
    )

    excluded = np.zeros_like(object_mask)
    excluded[4:12, 4:12] = True
    assert observations
    assert all(not np.logical_and(item.mask, excluded).any() for item in observations)


def test_disabled_structure_frontend_returns_no_observations() -> None:
    frontend = DepthStructureFrontend(
        _vocabulary(),
        DepthStructureConfig(enabled=False),
    )

    assert frontend.observe(_frame(), object_observations=()) == ()
