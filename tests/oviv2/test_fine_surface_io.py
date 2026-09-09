from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from src.oviv2 import fine_surface_io
from src.oviv2.fine_surface_io import (
    bind_ovi_owner_ids,
    load_cached_native_ovi_surface,
    load_native_ovi_surface,
)


def _write_surface(path: Path, *, quad_face: bool = False) -> None:
    vertices = np.empty(
        4,
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
            ("alpha", "u1"),
        ],
    )
    vertices[:] = [
        (0, 0, 1, 0, 0, 1, 10, 20, 30, 255),
        (1, 0, 1, 0, 0, 1, 40, 50, 60, 255),
        (0, 1, 1, 0, 0, 1, 10, 20, 30, 255),
        (1, 1, 1, 0, 0, 1, 1, 2, 3, 255),
    ]
    if quad_face:
        faces = np.empty(1, dtype=[("vertex_indices", "i4", (4,))])
        faces["vertex_indices"] = [[0, 1, 2, 3]]
    else:
        faces = np.empty(2, dtype=[("vertex_indices", "i4", (3,))])
        faces["vertex_indices"] = [[0, 1, 2], [1, 3, 2]]
    PlyData(
        [PlyElement.describe(vertices, "vertex"), PlyElement.describe(faces, "face")],
        text=False,
    ).write(path)


def test_native_ovi_surface_keeps_source_rows_faces_and_palette(tmp_path: Path) -> None:
    path = tmp_path / "native.ply"
    _write_surface(path)

    surface = load_native_ovi_surface(path)

    assert surface.vertices_xyz.tolist() == [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    assert surface.triangles.tolist() == [[0, 1, 2], [1, 3, 2]]
    assert surface.source_vertex_indices.tolist() == [0, 1, 2, 3]
    assert surface.palette_rgb.tolist()[-1] == [1, 2, 3]


def test_palette_binding_uses_authoritative_color_map_and_unknown_zero(tmp_path: Path) -> None:
    path = tmp_path / "native.ply"
    _write_surface(path)
    surface = load_native_ovi_surface(path)

    owners = bind_ovi_owner_ids(
        surface.palette_rgb,
        {7: (10, 20, 30), 99: (40, 50, 60)},
    )

    assert owners.tolist() == [7, 99, 7, 0]


def test_native_ovi_surface_rejects_non_triangular_faces(tmp_path: Path) -> None:
    path = tmp_path / "quad.ply"
    _write_surface(path, quad_face=True)

    with pytest.raises(ValueError, match="triangular"):
        load_native_ovi_surface(path)


def test_native_ovi_surface_cache_is_bound_to_source_hash(tmp_path: Path) -> None:
    path = tmp_path / "native.ply"
    cache = tmp_path / "cache" / "surface.npz"
    _write_surface(path)

    first = load_cached_native_ovi_surface(path, cache, source_sha256="a" * 64)
    second = load_cached_native_ovi_surface(path, cache, source_sha256="a" * 64)

    assert cache.is_file()
    assert np.array_equal(first.vertices_xyz, second.vertices_xyz)
    with pytest.raises(ValueError, match="source hash"):
        load_cached_native_ovi_surface(path, cache, source_sha256="b" * 64)


def test_select_source_surface_voxels_uses_floor_and_first_source_row() -> None:
    vertices = np.asarray(
        [
            [0.019, 0.0, 0.0],
            [0.011, 0.0, 0.0],
            [-0.001, 0.0, 0.0],
            [-0.019, 0.0, 0.0],
            [-0.021, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    selected = fine_surface_io.select_source_surface_voxels(
        vertices,
        voxel_size_m=0.02,
        origin_xyz=np.zeros(3, dtype=np.float64),
    )

    assert selected.tolist() == [0, 2, 4]


@pytest.mark.parametrize("voxel_size_m", [0.0, -0.02, float("nan")])
def test_select_source_surface_voxels_rejects_invalid_size(
    voxel_size_m: float,
) -> None:
    with pytest.raises(ValueError, match="voxel_size_m"):
        fine_surface_io.select_source_surface_voxels(
            np.zeros((1, 3), dtype=np.float32),
            voxel_size_m=voxel_size_m,
        )
