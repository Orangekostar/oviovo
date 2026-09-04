"""B7 arbitrary-point adapter for the frozen two-visit visibility engine."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from src.core.data_structures import Frame
from src.evaluation.contracts import MapSnapshot
from src.oviv2.two_visit_contracts import VisitMap
from src.oviv2.two_visit_current_map import SignedVisibilityGrid
from src.oviv2.two_visit_execution import (
    SignedVisibilityConfig,
    derive_signed_visibility,
)

_ADAPTER_SOURCE_SHA256 = "0" * 64


def _points(value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise TypeError("points_xyz must be numeric")
    try:
        points = np.array(raw, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("points_xyz must be convertible to float64") from error
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
        raise ValueError("points_xyz must have shape (N, 3) with finite values")
    points.setflags(write=False)
    return points


def _validate_frames(frames: tuple[Frame, ...], config: SignedVisibilityConfig) -> None:
    if not isinstance(frames, tuple) or not frames:
        raise ValueError("frames must be a non-empty tuple")
    if not isinstance(config, SignedVisibilityConfig):
        raise TypeError("config must be SignedVisibilityConfig")
    frame_ids = [frame.frame_id for frame in frames if isinstance(frame, Frame)]
    if len(frame_ids) != len(frames) or any(
        current <= previous for previous, current in pairwise(frame_ids)
    ):
        raise ValueError("visibility frame IDs must be strictly increasing")


def derive_signed_visibility_for_points(
    points_xyz: object,
    frames: tuple[Frame, ...],
    config: SignedVisibilityConfig,
    *,
    source_sha256: str,
) -> SignedVisibilityGrid:
    """Classify B7 candidates through the unchanged B3 visibility engine."""

    points = _points(points_xyz)
    _validate_frames(frames, config)
    if not len(points):
        return SignedVisibilityGrid.empty(config.voxel_size_m, source_sha256)
    scaled = np.floor(points / config.voxel_size_m)
    limit = np.iinfo(np.int64).max
    if np.any(np.abs(scaled) > limit):
        raise ValueError("points_xyz exceed visibility voxel range")
    expected_keys = np.unique(scaled.astype(np.int64), axis=0)
    voxel_centers = (expected_keys.astype(np.float64) + 0.5) * config.voxel_size_m
    synthetic = VisitMap(
        visit_id=0,
        snapshot=MapSnapshot(
            method="OVI_B7_VISIBILITY_ADAPTER",
            scene_id="b7-candidates",
            timestamp=0.0,
            entities=[],
            background_xyz=voxel_centers,
            scope="current",
        ),
        coordinate_frame_id="tesse_cd_world",
        source_manifest_sha256=_ADAPTER_SOURCE_SHA256,
        map_voxel_size_m=config.voxel_size_m,
        observed_frame_start=0,
        observed_frame_end=0,
    )
    result = derive_signed_visibility(
        synthetic,
        frames,
        config,
        source_sha256=source_sha256,
    )
    if not np.array_equal(result.voxel_keys, expected_keys):
        raise ValueError("visibility adapter could not preserve candidate voxel keys")
    return result


__all__ = ["derive_signed_visibility_for_points"]
