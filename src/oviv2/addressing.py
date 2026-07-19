from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np


VoxelKey: TypeAlias = tuple[int, int, int]
BlockKey: TypeAlias = tuple[int, int, int]


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _integer_key(value: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three integer coordinates")
    normalized: list[int] = []
    for component in value:
        if not isinstance(component, (int, np.integer)) or isinstance(component, bool):
            raise ValueError(f"{name} must contain three integer coordinates")
        normalized.append(int(component))
    return normalized[0], normalized[1], normalized[2]


def point_to_voxel(point_xyz: Sequence[float], voxel_size_m: float) -> VoxelKey:
    point = np.asarray(point_xyz, dtype=np.float64)
    size = float(voxel_size_m)
    if point.shape != (3,):
        raise ValueError("point_xyz must contain three coordinates")
    if not np.isfinite(point).all():
        raise ValueError("point_xyz must contain only finite coordinates")
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")
    key = np.floor(point / size).astype(np.int64)
    return int(key[0]), int(key[1]), int(key[2])


def split_voxel_key(
    key: VoxelKey,
    block_resolution: int,
) -> tuple[BlockKey, tuple[int, int, int]]:
    voxel_key = _integer_key(key, "voxel key")
    resolution = _positive_integer(block_resolution, "block_resolution")
    block = tuple(component // resolution for component in voxel_key)
    local = tuple(component % resolution for component in voxel_key)
    return block, local


def join_voxel_key(
    block_key: BlockKey,
    local_key: tuple[int, int, int],
    block_resolution: int,
) -> VoxelKey:
    block = _integer_key(block_key, "block key")
    local = _integer_key(local_key, "local key")
    resolution = _positive_integer(block_resolution, "block_resolution")
    if any(component < 0 or component >= resolution for component in local):
        raise ValueError("local key coordinates must lie inside the block")
    return tuple(
        block_component * resolution + local_component
        for block_component, local_component in zip(block, local)
    )
