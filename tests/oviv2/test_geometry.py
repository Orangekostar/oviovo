from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.oviv2.geometry import SparseTsdfVolume, TsdfConfig


def _plane_frame(
    *,
    height: int = 48,
    width: int = 64,
    depth_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.full((height, width), depth_m, dtype=np.float32)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[...] = (64, 128, 192)
    focal = 60.0
    intrinsics = np.asarray(
        [
            [focal, 0.0, (width - 1) / 2.0],
            [0.0, focal, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return depth, rgb, intrinsics


def _integrate_twice(
    volume: SparseTsdfVolume,
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> int:
    touched = volume.integrate(depth, rgb, intrinsics, camera_to_world)
    assert volume.integrate(depth, rgb, intrinsics, camera_to_world) == touched
    return touched


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"voxel_size_m": 0.0}, "voxel_size_m"),
        ({"block_resolution": 0}, "block_resolution"),
        ({"block_count": 0}, "block_count"),
        ({"depth_max_m": 0.0}, "depth_max_m"),
        ({"trunc_voxel_multiplier": 0.0}, "trunc_voxel_multiplier"),
    ],
)
def test_config_rejects_non_positive_values(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TsdfConfig(**overrides)


def test_synthetic_plane_allocates_blocks_and_extracts_mesh() -> None:
    depth, rgb, intrinsics = _plane_frame()
    volume = SparseTsdfVolume()

    touched = _integrate_twice(volume, depth, rgb, intrinsics, np.eye(4))
    mesh = volume.extract_mesh()

    assert touched > 0
    assert volume.active_block_count == touched
    assert mesh.vertex.positions.shape[0] > 0
    assert mesh.triangle.indices.shape[0] > 0


def test_invalid_depth_pixels_are_ignored_without_mutating_input() -> None:
    depth, rgb, intrinsics = _plane_frame()
    depth[0, :4] = np.asarray([0.0, -1.0, np.nan, np.inf], dtype=np.float32)
    original = depth.copy()
    volume = SparseTsdfVolume()

    touched = _integrate_twice(volume, depth, rgb, intrinsics, np.eye(4))

    assert touched > 0
    np.testing.assert_equal(depth, original)
    assert volume.extract_mesh().vertex.positions.shape[0] > 0


def test_camera_to_world_is_inverted_for_open3d_integration() -> None:
    depth, rgb, intrinsics = _plane_frame()
    camera_to_world = np.eye(4)
    camera_to_world[0, 3] = 0.5
    volume = SparseTsdfVolume()

    _integrate_twice(volume, depth, rgb, intrinsics, camera_to_world)
    vertices = volume.extract_mesh().vertex.positions.numpy()

    assert float(vertices[:, 0].mean()) > 0.25


def test_save_load_preserves_blocks_and_mesh(tmp_path: Path) -> None:
    depth, rgb, intrinsics = _plane_frame()
    config = TsdfConfig()
    volume = SparseTsdfVolume(config)
    _integrate_twice(volume, depth, rgb, intrinsics, np.eye(4))
    expected_vertices = volume.extract_mesh().vertex.positions.shape[0]
    path = tmp_path / "geometry.npz"

    volume.save(path)
    restored = SparseTsdfVolume.load(path, config)

    assert expected_vertices > 0
    assert restored.active_block_count == volume.active_block_count
    assert restored.extract_mesh().vertex.positions.shape[0] == expected_vertices
    with pytest.raises(ValueError, match="voxel_size_m"):
        SparseTsdfVolume.load(path, TsdfConfig(voxel_size_m=0.1))


def test_volume_exposes_no_dense_point_state() -> None:
    volume = SparseTsdfVolume()

    forbidden = {"points", "point_cloud", "dense_map", "local_pcd"}
    assert forbidden.isdisjoint(vars(volume))
    for name in forbidden:
        assert not hasattr(volume, name)
