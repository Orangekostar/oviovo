from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class TsdfConfig:
    voxel_size_m: float = 0.05
    block_resolution: int = 8
    block_count: int = 100_000
    depth_max_m: float = 10.0
    trunc_voxel_multiplier: float = 4.0

    def __post_init__(self) -> None:
        for name in ("voxel_size_m", "depth_max_m", "trunc_voxel_multiplier"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("block_resolution", "block_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class SparseTsdfVolume:
    """Sparse Open3D TSDF volume used as OVIV2's geometry authority."""

    _ATTRIBUTE_NAMES = ("tsdf", "weight", "color")
    _ATTRIBUTE_DTYPES = (o3d.core.float32, o3d.core.float32, o3d.core.float32)
    _ATTRIBUTE_CHANNELS = ((1,), (1,), (3,))

    def __init__(self, config: TsdfConfig = TsdfConfig()) -> None:
        if not isinstance(config, TsdfConfig):
            raise TypeError("config must be a TsdfConfig")
        self._config = config
        self._grid = o3d.t.geometry.VoxelBlockGrid(
            self._ATTRIBUTE_NAMES,
            self._ATTRIBUTE_DTYPES,
            self._ATTRIBUTE_CHANNELS,
            voxel_size=config.voxel_size_m,
            block_resolution=config.block_resolution,
            block_count=config.block_count,
            device=o3d.core.Device("CPU:0"),
        )

    @property
    def config(self) -> TsdfConfig:
        return self._config

    @property
    def active_block_count(self) -> int:
        return int(self._grid.hashmap().size())

    def integrate(
        self,
        depth_m: np.ndarray,
        rgb: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> int:
        depth = np.asarray(depth_m)
        color = np.asarray(rgb)
        intrinsic = np.asarray(intrinsics, dtype=np.float64)
        pose = np.asarray(camera_to_world, dtype=np.float64)

        if depth.ndim != 2:
            raise ValueError("depth_m must have shape (H, W)")
        if color.shape != (*depth.shape, 3):
            raise ValueError("rgb must have shape (H, W, 3) matching depth_m")
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("intrinsics must be a finite 3x3 matrix")
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise ValueError("intrinsics focal lengths must be positive")
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("camera_to_world must be a finite 4x4 matrix")

        try:
            world_to_camera = np.linalg.inv(pose)
        except np.linalg.LinAlgError as exc:
            raise ValueError("camera_to_world must be invertible") from exc

        clean_depth = np.array(depth, dtype=np.float32, copy=True, order="C")
        valid_depth = (
            np.isfinite(clean_depth)
            & (clean_depth > 0.0)
            & (clean_depth <= self._config.depth_max_m)
        )
        clean_depth[~valid_depth] = 0.0

        if color.dtype == np.uint8:
            clean_color = np.ascontiguousarray(color, dtype=np.float32) / 255.0
        else:
            clean_color = np.ascontiguousarray(color, dtype=np.float32)
            if not np.isfinite(clean_color).all():
                raise ValueError("rgb must contain only finite values")
            if clean_color.size and (clean_color.min() < 0.0 or clean_color.max() > 1.0):
                raise ValueError("floating-point rgb values must lie in [0, 1]")

        depth_image = o3d.t.geometry.Image(o3d.core.Tensor(clean_depth))
        color_image = o3d.t.geometry.Image(o3d.core.Tensor(clean_color))
        intrinsic_tensor = o3d.core.Tensor(intrinsic, dtype=o3d.core.float64)
        extrinsic_tensor = o3d.core.Tensor(world_to_camera, dtype=o3d.core.float64)
        block_coords = self._grid.compute_unique_block_coordinates(
            depth_image,
            intrinsic_tensor,
            extrinsic_tensor,
            depth_scale=1.0,
            depth_max=self._config.depth_max_m,
            trunc_voxel_multiplier=self._config.trunc_voxel_multiplier,
        )
        touched_count = int(block_coords.shape[0])
        if touched_count == 0:
            return 0
        self._grid.integrate(
            block_coords,
            depth_image,
            color_image,
            intrinsic_tensor,
            extrinsic_tensor,
            depth_scale=1.0,
            depth_max=self._config.depth_max_m,
            trunc_voxel_multiplier=self._config.trunc_voxel_multiplier,
        )
        return touched_count

    def extract_mesh(
        self,
        weight_threshold: float = 1.0,
    ) -> o3d.t.geometry.TriangleMesh:
        threshold = float(weight_threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("weight_threshold must be finite and non-negative")
        if self.active_block_count == 0:
            mesh = o3d.t.geometry.TriangleMesh(o3d.core.Device("CPU:0"))
            mesh.vertex["positions"] = o3d.core.Tensor(
                np.empty((0, 3), dtype=np.float32)
            )
            mesh.vertex["colors"] = o3d.core.Tensor(
                np.empty((0, 3), dtype=np.float32)
            )
            mesh.triangle["indices"] = o3d.core.Tensor(
                np.empty((0, 3), dtype=np.int64)
            )
            return mesh
        return self._grid.extract_triangle_mesh(weight_threshold=threshold)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("geometry path must end in .npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.stem}.",
                suffix=".npz",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            self._grid.save(str(temporary_path))
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(cls, path: str | Path, config: TsdfConfig) -> "SparseTsdfVolume":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        if not isinstance(config, TsdfConfig):
            raise TypeError("config must be a TsdfConfig")
        with np.load(source, allow_pickle=False) as payload:
            stored_voxel_size = float(np.asarray(payload["voxel_size"]).reshape(-1)[0])
            stored_resolution = int(np.asarray(payload["block_resolution"]).reshape(-1)[0])
            stored_block_count = int(np.asarray(payload["key"]).shape[0])
        if not np.isclose(stored_voxel_size, config.voxel_size_m, rtol=0.0, atol=1e-7):
            raise ValueError("stored voxel_size_m does not match config")
        if stored_resolution != config.block_resolution:
            raise ValueError("stored block_resolution does not match config")

        if stored_block_count == 0:
            return cls(config)
        instance = cls.__new__(cls)
        instance._config = config
        instance._grid = o3d.t.geometry.VoxelBlockGrid.load(str(source))
        return instance
