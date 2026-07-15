from __future__ import annotations

from typing import Any

import numpy as np


def readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type,
    ndim: int,
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, got shape={array.shape}")
    if trailing_shape and array.shape[-len(trailing_shape) :] != trailing_shape:
        raise ValueError(f"expected trailing shape {trailing_shape}, got shape={array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("array must contain only finite values")
    array.setflags(write=False)
    return array
