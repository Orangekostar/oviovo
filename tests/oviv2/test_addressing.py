from __future__ import annotations

import numpy as np
import pytest

from src.oviv2.addressing import join_voxel_key, point_to_voxel, split_voxel_key


@pytest.mark.parametrize(
    "voxel_key",
    [
        (0, 0, 0),
        (7, 8, 9),
        (-1, -8, -9),
        (-17, 31, -64),
    ],
)
def test_split_join_round_trip_handles_negative_keys(voxel_key: tuple[int, int, int]) -> None:
    block_key, local_key = split_voxel_key(voxel_key, block_resolution=8)

    assert all(0 <= component < 8 for component in local_key)
    assert join_voxel_key(block_key, local_key, block_resolution=8) == voxel_key


def test_point_to_voxel_uses_floor_at_boundaries() -> None:
    assert point_to_voxel([0.0, 0.049, -0.001], 0.05) == (0, 0, -1)
    assert point_to_voxel(np.asarray([0.1, -0.1, 1.0]), 0.05) == (2, -2, 20)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: point_to_voxel([0.0, np.nan, 1.0], 0.05), "finite"),
        (lambda: point_to_voxel([0.0, 1.0], 0.05), "three"),
        (lambda: split_voxel_key((0, 0, 0), 0), "block_resolution"),
        (lambda: join_voxel_key((0, 0, 0), (8, 0, 0), 8), "local"),
    ],
)
def test_addressing_rejects_invalid_input(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
