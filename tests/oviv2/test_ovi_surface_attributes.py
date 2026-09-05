from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from src.core.data_structures import CameraIntrinsics
from src.oviv2.ovi_surface_attributes import (
    SurfaceAttributeError,
    depth_millimeters_to_meters,
    group_surface_attributes_by_color,
    load_ply_surface_attributes,
    project_world_points,
    sample_depth_consistent_rgb,
    validate_surface_group,
)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=2.0,
        fy=2.0,
        cx=2.0,
        cy=2.0,
        width=5,
        height=5,
    )


def test_surface_grouping_keeps_xyz_normal_and_original_index_in_stable_order() -> None:
    points = np.asarray(
        [
            [9.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    colors = np.asarray(
        [[2, 2, 2], [1, 1, 1], [2, 2, 2], [1, 1, 1]], dtype=np.uint8
    )
    normals = np.asarray(
        [[9.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    original = np.asarray([10, 11, 12, 13], dtype=np.int64)

    grouped = group_surface_attributes_by_color(
        points,
        colors,
        normals,
        original,
        [(2, 2, 2), (1, 1, 1)],
    )

    first = grouped[(1, 1, 1)]
    assert np.array_equal(first.points_xyz, points[[1, 3]])
    assert np.array_equal(first.original_vertex_indices, [11, 13])
    assert np.allclose(first.normals_xyz, [[0.0, 1.0, 0.0], [2**-0.5, 2**-0.5, 0.0]])
    assert np.array_equal(first.normal_valid, [True, True])
    assert first.rgb_source == "instance_palette"

    second = grouped[(2, 2, 2)]
    assert np.array_equal(second.points_xyz, points[[0, 2]])
    assert np.array_equal(second.original_vertex_indices, [10, 12])


def test_duplicate_xyz_rows_remain_distinguishable_by_original_index() -> None:
    points = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    group = group_surface_attributes_by_color(
        points,
        np.asarray([[7, 8, 9], [7, 8, 9]], dtype=np.uint8),
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        np.asarray([41, 99], dtype=np.int64),
        [(7, 8, 9)],
    )[(7, 8, 9)]

    validate_surface_group(points, group)

    assert np.array_equal(group.original_vertex_indices, [41, 99])
    assert np.array_equal(group.normals_xyz, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_ply_reader_uses_source_row_as_original_vertex_index(tmp_path: Path) -> None:
    vertices = np.empty(
        3,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("normal_x", "f4"),
            ("normal_y", "f4"),
            ("normal_z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices[:] = [
        (3, 0, 0, 0, 0, 2, 8, 8, 8),
        (1, 0, 0, 2, 0, 0, 7, 7, 7),
        (2, 0, 0, 0, 3, 0, 7, 7, 7),
    ]
    path = tmp_path / "surface.ply"
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)

    group = load_ply_surface_attributes(path, [(7, 7, 7)])[(7, 7, 7)]

    assert np.array_equal(group.points_xyz[:, 0], [1.0, 2.0])
    assert np.array_equal(group.original_vertex_indices, [1, 2])
    assert np.array_equal(group.normals_xyz, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_surface_group_validation_is_exact_and_invalid_normals_stay_unsupported() -> None:
    group = group_surface_attributes_by_color(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64),
        np.asarray([[1, 2, 3], [1, 2, 3]], dtype=np.uint8),
        np.asarray([[0.0, 0.0, 0.0], [np.nan, 1.0, 0.0]], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64),
        [(1, 2, 3)],
    )[(1, 2, 3)]

    assert np.array_equal(group.normal_valid, [False, False])
    assert np.array_equal(group.normals_xyz, np.zeros((2, 3), dtype=np.float32))
    validate_surface_group(np.asarray(group.points_xyz, dtype=np.float64), group)
    with pytest.raises(SurfaceAttributeError, match="exactly match"):
        validate_surface_group(group.points_xyz[::-1], group)


def test_projection_uses_inverse_pose_camera_z_and_fixed_pixel_rounding() -> None:
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[0, 3] = 1.0
    points = np.asarray(
        [
            [1.0, 0.0, 2.0],
            [1.5, 0.5, 2.0],
            [1.5, 0.5, -1.0],
        ],
        dtype=np.float64,
    )

    projection = project_world_points(points, camera_to_world, _intrinsics())

    assert np.allclose(projection.camera_xyz[:2], [[0.0, 0.0, 2.0], [0.5, 0.5, 2.0]])
    assert np.array_equal(projection.rows, [2, 2, -1])
    assert np.array_equal(projection.columns, [2, 2, -1])
    assert np.array_equal(projection.projectable, [True, True, False])


def test_depth_units_and_known_pixel_camera_rgb_are_preserved() -> None:
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    rgb[2, 2] = [12, 34, 56]
    depth = depth_millimeters_to_meters(
        np.full((5, 5), 2000, dtype=np.uint16)
    )
    projection = project_world_points(
        np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32),
        np.eye(4),
        _intrinsics(),
    )

    support = sample_depth_consistent_rgb(
        projection,
        rgb,
        depth,
        depth_tolerance_m=0.01,
        candidate_visit_ids=np.asarray([0], dtype=np.int8),
        frame_visit_id=0,
        rgb_source="camera_rgb",
    )

    assert support.candidate_indices.tolist() == [0]
    assert support.rows.tolist() == [2]
    assert support.columns.tolist() == [2]
    assert support.rgb_uint8.tolist() == [[12, 34, 56]]
    assert np.allclose(support.camera_depth_m, [2.0])
    assert np.allclose(support.observed_depth_m, [2.0])
    assert np.allclose(support.depth_residual_m, [0.0])


@pytest.mark.parametrize("bad_depth", [0.0, np.nan, 1.5, 2.5])
def test_invalid_or_inconsistent_depth_is_rejected(bad_depth: float) -> None:
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    depth[2, 2] = bad_depth
    projection = project_world_points(
        np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32),
        np.eye(4),
        _intrinsics(),
    )

    support = sample_depth_consistent_rgb(
        projection,
        np.zeros((5, 5, 3), dtype=np.uint8),
        depth,
        depth_tolerance_m=0.01,
        candidate_visit_ids=np.asarray([0], dtype=np.int8),
        frame_visit_id=0,
        rgb_source="camera_rgb",
    )

    assert len(support.candidate_indices) == 0


def test_depth_discontinuity_cross_visit_and_palette_rgb_are_rejected() -> None:
    projection = project_world_points(
        np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32),
        np.eye(4),
        _intrinsics(),
    )
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    depth_edge = np.full((5, 5), 2.0, dtype=np.float32)
    depth_edge[1, 1:3] = 1.0

    edge = sample_depth_consistent_rgb(
        projection,
        rgb,
        depth_edge,
        depth_tolerance_m=0.01,
        candidate_visit_ids=np.asarray([0], dtype=np.int8),
        frame_visit_id=0,
        rgb_source="camera_rgb",
    )
    assert len(edge.candidate_indices) == 0

    with pytest.raises(SurfaceAttributeError, match="same visit"):
        sample_depth_consistent_rgb(
            projection,
            rgb,
            np.full((5, 5), 2.0, dtype=np.float32),
            depth_tolerance_m=0.01,
            candidate_visit_ids=np.asarray([1], dtype=np.int8),
            frame_visit_id=0,
            rgb_source="camera_rgb",
        )
    with pytest.raises(SurfaceAttributeError, match="palette"):
        sample_depth_consistent_rgb(
            projection,
            rgb,
            np.full((5, 5), 2.0, dtype=np.float32),
            depth_tolerance_m=0.01,
            candidate_visit_ids=np.asarray([0], dtype=np.int8),
            frame_visit_id=0,
            rgb_source="instance_palette",
        )
