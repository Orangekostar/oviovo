"""Fine-surface current-validity decisions with conservative revocation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import pairwise

import numpy as np

from src.core.data_structures import Frame
from src.oviv2.current_surface import CurrentEvidenceState
from src.oviv2.ovi_surface_attributes import (
    SurfaceAttributeError,
    project_world_points,
    sample_depth_consistent_rgb,
)


def _immutable(value: object, dtype: object) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.flags.writeable or array.base is not None:
        array = array.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FineSurfaceEvidence:
    present_observations: np.ndarray
    visible_absent_observations: np.ndarray
    occluded_observations: np.ndarray
    distinct_absent_viewpoints: np.ndarray
    last_supported_frames: np.ndarray
    last_absent_frames: np.ndarray | None = None
    last_occluded_frames: np.ndarray | None = None

    def __post_init__(self) -> None:
        dtypes = {
            "present_observations": np.uint16,
            "visible_absent_observations": np.uint16,
            "occluded_observations": np.uint16,
            "distinct_absent_viewpoints": np.uint8,
            "last_supported_frames": np.int32,
        }
        for field in fields(self)[:5]:
            object.__setattr__(
                self, field.name, _immutable(getattr(self, field.name), dtypes[field.name])
            )
        count = len(self.present_observations)
        for name in ("last_absent_frames", "last_occluded_frames"):
            raw = getattr(self, name)
            value = (
                np.full(count, -1, dtype=np.int32)
                if raw is None
                else _immutable(raw, np.int32)
            )
            object.__setattr__(self, name, _immutable(value, np.int32))
        if any(getattr(self, field.name).shape != (count,) for field in fields(self)):
            raise ValueError("fine-surface evidence arrays must have equal one-dimensional shape")
        if np.any(self.last_supported_frames < -1):
            raise ValueError("last_supported_frames cannot be below -1")
        if np.any(self.last_absent_frames < -1) or np.any(
            self.last_occluded_frames < -1
        ):
            raise ValueError("evidence frame IDs cannot be below -1")
        if np.any(self.distinct_absent_viewpoints > self.visible_absent_observations):
            raise ValueError("distinct absent viewpoints cannot exceed absent observations")

    @classmethod
    def empty(cls, count: int) -> FineSurfaceEvidence:
        if count < 0:
            raise ValueError("count must be nonnegative")
        return cls(
            present_observations=np.zeros(count, dtype=np.uint16),
            visible_absent_observations=np.zeros(count, dtype=np.uint16),
            occluded_observations=np.zeros(count, dtype=np.uint16),
            distinct_absent_viewpoints=np.zeros(count, dtype=np.uint8),
            last_supported_frames=np.full(count, -1, dtype=np.int32),
        )


@dataclass(frozen=True, slots=True)
class FineSurfaceObservation:
    evidence: FineSurfaceEvidence
    observed_rgb_uint8: np.ndarray
    rgb_valid: np.ndarray
    best_rgb_depth_residual_m: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, FineSurfaceEvidence):
            raise TypeError("evidence must be FineSurfaceEvidence")
        count = len(self.evidence.present_observations)
        rgb = np.asarray(self.observed_rgb_uint8)
        valid = np.asarray(self.rgb_valid)
        residual = np.asarray(self.best_rgb_depth_residual_m)
        if rgb.shape != (count, 3):
            raise ValueError("observed RGB must have shape (N, 3)")
        if valid.shape != (count,) or valid.dtype != np.bool_:
            raise ValueError("rgb_valid must be an aligned boolean array")
        if residual.shape != (count,) or np.isnan(residual).any():
            raise ValueError("RGB residuals must be aligned and not NaN")
        if np.any(valid != np.isfinite(residual)):
            raise ValueError("RGB validity must match finite residuals")
        object.__setattr__(self, "observed_rgb_uint8", _immutable(rgb, np.uint8))
        object.__setattr__(self, "rgb_valid", _immutable(valid, np.bool_))
        object.__setattr__(
            self,
            "best_rgb_depth_residual_m",
            _immutable(residual, np.float32),
        )


@dataclass(frozen=True, slots=True)
class FineEvidenceProjectionConfig:
    depth_tolerance_m: float = 0.05
    depth_max_m: float = 10.0
    minimum_viewpoint_baseline_m: float = 0.25
    maximum_distinct_viewpoints: int = 3
    minimum_rgb_neighbours: int = 5
    rgb_edge_range_factor: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "depth_tolerance_m",
            "depth_max_m",
            "minimum_viewpoint_baseline_m",
            "rgb_edge_range_factor",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            type(self.maximum_distinct_viewpoints) is not int
            or not 1 <= self.maximum_distinct_viewpoints <= 255
        ):
            raise ValueError("maximum_distinct_viewpoints must be in [1, 255]")
        if (
            type(self.minimum_rgb_neighbours) is not int
            or not 1 <= self.minimum_rgb_neighbours <= 9
        ):
            raise ValueError("minimum_rgb_neighbours must be in [1, 9]")


def _fine_points(points_xyz: object) -> np.ndarray:
    raw = np.asarray(points_xyz)
    if raw.dtype.kind == "b":
        raise TypeError("points_xyz must be numeric")
    try:
        points = np.ascontiguousarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("points_xyz must be convertible to float64") from error
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("points_xyz must have shape (N, 3) with finite values")
    return points


def _validated_projection_frames(frames: tuple[Frame, ...]) -> None:
    if not isinstance(frames, tuple) or not frames:
        raise ValueError("frames must be a non-empty tuple")
    frame_ids: list[int] = []
    for frame in frames:
        if not isinstance(frame, Frame):
            raise TypeError("frames must contain Frame values")
        if type(frame.frame_id) is not int:
            raise ValueError("frame IDs must be integers")
        frame_ids.append(frame.frame_id)
        rgb = np.asarray(frame.rgb)
        depth = np.asarray(frame.depth)
        if rgb.ndim != 3 or rgb.shape[2:] != (3,) or rgb.dtype != np.uint8:
            raise ValueError("frame RGB must be HxWx3 uint8")
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError("frame depth must align with RGB")
        if (frame.intrinsics.height, frame.intrinsics.width) != depth.shape:
            raise ValueError("frame intrinsics must match image dimensions")
    if any(current <= previous for previous, current in pairwise(frame_ids)):
        raise ValueError("frame IDs must be strictly increasing")


def _increment(counter: np.ndarray, indices: np.ndarray) -> None:
    if len(indices):
        counter[indices] = np.minimum(
            counter[indices] + np.uint32(1), np.iinfo(np.uint16).max
        )


def observe_fine_surface(
    points_xyz: object,
    frames: tuple[Frame, ...],
    config: FineEvidenceProjectionConfig,
    *,
    point_batch_size: int = 250_000,
) -> FineSurfaceObservation:
    """Accumulate signed depth evidence and best same-visit RGB in bounded memory."""

    points = _fine_points(points_xyz)
    _validated_projection_frames(frames)
    if not isinstance(config, FineEvidenceProjectionConfig):
        raise TypeError("config must be FineEvidenceProjectionConfig")
    if type(point_batch_size) is not int or point_batch_size <= 0:
        raise ValueError("point_batch_size must be a positive integer")

    count = len(points)
    present = np.zeros(count, dtype=np.uint16)
    absent = np.zeros(count, dtype=np.uint16)
    occluded = np.zeros(count, dtype=np.uint16)
    viewpoint_count = np.zeros(count, dtype=np.uint8)
    last_supported = np.full(count, -1, dtype=np.int32)
    last_absent = np.full(count, -1, dtype=np.int32)
    last_occluded = np.full(count, -1, dtype=np.int32)
    observed_rgb = np.zeros((count, 3), dtype=np.uint8)
    best_residual = np.full(count, np.inf, dtype=np.float32)

    for start in range(0, count, point_batch_size):
        stop = min(start + point_batch_size, count)
        batch = points[start:stop]
        size = stop - start
        batch_present = np.zeros(size, dtype=np.uint32)
        batch_absent = np.zeros(size, dtype=np.uint32)
        batch_occluded = np.zeros(size, dtype=np.uint32)
        batch_viewpoint_count = np.zeros(size, dtype=np.uint8)
        viewpoints = np.empty(
            (size, config.maximum_distinct_viewpoints, 3), dtype=np.float64
        )
        batch_last_supported = np.full(size, -1, dtype=np.int32)
        batch_last_absent = np.full(size, -1, dtype=np.int32)
        batch_last_occluded = np.full(size, -1, dtype=np.int32)
        batch_rgb = np.zeros((size, 3), dtype=np.uint8)
        batch_best_residual = np.full(size, np.inf, dtype=np.float32)

        for frame in frames:
            projection = project_world_points(batch, frame.pose, frame.intrinsics)
            projected = np.flatnonzero(projection.projectable)
            clean_depth = np.asarray(frame.depth, dtype=np.float32)
            if len(projected):
                rows = projection.rows[projected]
                columns = projection.columns[projected]
                measured = np.asarray(clean_depth[rows, columns], dtype=np.float64)
                camera_depth = projection.camera_xyz[projected, 2]
                testable = (
                    np.isfinite(measured)
                    & (measured > 0.0)
                    & (measured <= config.depth_max_m)
                )
                tested = projected[testable]
                differences = measured[testable] - camera_depth[testable]
                present_indices = tested[
                    np.abs(differences) <= config.depth_tolerance_m
                ]
                absent_indices = tested[
                    differences > config.depth_tolerance_m
                ]
                occluded_indices = tested[
                    differences < -config.depth_tolerance_m
                ]
                _increment(batch_present, present_indices)
                _increment(batch_absent, absent_indices)
                _increment(batch_occluded, occluded_indices)
                batch_last_supported[present_indices] = frame.frame_id
                batch_last_absent[absent_indices] = frame.frame_id
                batch_last_occluded[occluded_indices] = frame.frame_id

                if len(absent_indices):
                    camera = np.asarray(frame.pose, dtype=np.float64)[:3, 3]
                    counts = batch_viewpoint_count[absent_indices]
                    distinct = np.ones(len(absent_indices), dtype=np.bool_)
                    for slot in range(config.maximum_distinct_viewpoints):
                        occupied = counts > slot
                        if np.any(occupied):
                            distances = np.linalg.norm(
                                viewpoints[absent_indices[occupied], slot] - camera,
                                axis=1,
                            )
                            distinct[occupied] &= (
                                distances >= config.minimum_viewpoint_baseline_m
                            )
                    add = distinct & (
                        counts < config.maximum_distinct_viewpoints
                    )
                    if np.any(add):
                        selected = absent_indices[add]
                        slots = counts[add].astype(np.int64)
                        viewpoints[selected, slots] = camera
                        batch_viewpoint_count[selected] += 1

            try:
                support = sample_depth_consistent_rgb(
                    projection,
                    frame.rgb,
                    clean_depth,
                    depth_tolerance_m=config.depth_tolerance_m,
                    candidate_visit_ids=np.zeros(size, dtype=np.int8),
                    frame_visit_id=0,
                    rgb_source="camera_rgb",
                    minimum_valid_neighbours=config.minimum_rgb_neighbours,
                    edge_range_factor=config.rgb_edge_range_factor,
                )
            except SurfaceAttributeError as error:
                raise ValueError(f"invalid frame surface support: {error}") from error
            rgb_rows = support.candidate_indices
            better = support.depth_residual_m < batch_best_residual[rgb_rows]
            selected_rgb_rows = rgb_rows[better]
            batch_rgb[selected_rgb_rows] = support.rgb_uint8[better]
            batch_best_residual[selected_rgb_rows] = support.depth_residual_m[better]

        present[start:stop] = batch_present.astype(np.uint16)
        absent[start:stop] = batch_absent.astype(np.uint16)
        occluded[start:stop] = batch_occluded.astype(np.uint16)
        viewpoint_count[start:stop] = batch_viewpoint_count
        last_supported[start:stop] = batch_last_supported
        last_absent[start:stop] = batch_last_absent
        last_occluded[start:stop] = batch_last_occluded
        observed_rgb[start:stop] = batch_rgb
        best_residual[start:stop] = batch_best_residual

    evidence = FineSurfaceEvidence(
        present_observations=present,
        visible_absent_observations=absent,
        occluded_observations=occluded,
        distinct_absent_viewpoints=viewpoint_count,
        last_supported_frames=last_supported,
        last_absent_frames=last_absent,
        last_occluded_frames=last_occluded,
    )
    return FineSurfaceObservation(
        evidence=evidence,
        observed_rgb_uint8=observed_rgb,
        rgb_valid=np.isfinite(best_residual),
        best_rgb_depth_residual_m=best_residual,
    )


@dataclass(frozen=True, slots=True)
class FineValidityConfig:
    minimum_absent_observations: int = 2
    minimum_distinct_viewpoints: int = 2

    def __post_init__(self) -> None:
        if self.minimum_absent_observations < 1:
            raise ValueError("minimum_absent_observations must be positive")
        if self.minimum_distinct_viewpoints < 1:
            raise ValueError("minimum_distinct_viewpoints must be positive")


@dataclass(frozen=True, slots=True)
class FineValidityResult:
    current_valid: np.ndarray
    evidence_state_codes: np.ndarray
    last_supported_frames: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_valid", _immutable(self.current_valid, np.bool_))
        object.__setattr__(
            self,
            "evidence_state_codes",
            _immutable(self.evidence_state_codes, np.uint8),
        )
        object.__setattr__(
            self,
            "last_supported_frames",
            _immutable(self.last_supported_frames, np.int32),
        )


def resolve_fine_current_validity(
    *,
    source_visit_ids: np.ndarray,
    latest_visit_id: int,
    coarse_visible_free_candidates: np.ndarray,
    evidence: FineSurfaceEvidence,
    config: FineValidityConfig,
) -> FineValidityResult:
    """Resolve currentness; coarse free space is only a candidate revocation signal."""

    visits = np.asarray(source_visit_ids)
    candidates = np.asarray(coarse_visible_free_candidates)
    count = len(evidence.present_observations)
    if visits.shape != (count,) or not np.issubdtype(visits.dtype, np.integer):
        raise ValueError("source_visit_ids must be an integer array aligned to evidence")
    if candidates.shape != (count,) or candidates.dtype != np.bool_:
        raise ValueError("coarse_visible_free_candidates must be an aligned boolean array")
    if latest_visit_id < 0 or np.any(visits < 0) or np.any(visits > latest_visit_id):
        raise ValueError("source visits must not exceed the latest visit")
    if not isinstance(config, FineValidityConfig):
        raise TypeError("config must be FineValidityConfig")

    current = np.ones(count, dtype=np.bool_)
    states = np.full(
        count, int(CurrentEvidenceState.HISTORICAL_UNOBSERVED), dtype=np.uint8
    )
    latest = visits == latest_visit_id
    states[latest] = int(CurrentEvidenceState.CURRENT_OBSERVED)

    historical = ~latest
    replaced = historical & (evidence.present_observations > 0)
    current[replaced] = False
    states[replaced] = int(CurrentEvidenceState.REPLACED_BY_CURRENT)

    remaining = historical & ~replaced
    revoke = (
        remaining
        & candidates
        & (evidence.visible_absent_observations >= config.minimum_absent_observations)
        & (evidence.distinct_absent_viewpoints >= config.minimum_distinct_viewpoints)
    )
    current[revoke] = False
    states[revoke] = int(CurrentEvidenceState.REVOKED_VISIBLE_FREE)

    remaining &= ~revoke
    occluded = remaining & (evidence.occluded_observations > 0)
    states[occluded] = int(CurrentEvidenceState.HISTORICAL_OCCLUDED)
    uncertain = remaining & ~occluded & (evidence.visible_absent_observations > 0)
    states[uncertain] = int(CurrentEvidenceState.HISTORICAL_UNCERTAIN)
    return FineValidityResult(current, states, evidence.last_supported_frames)


__all__ = [
    "CurrentEvidenceState",
    "FineEvidenceProjectionConfig",
    "FineSurfaceEvidence",
    "FineSurfaceObservation",
    "FineValidityConfig",
    "FineValidityResult",
    "observe_fine_surface",
    "resolve_fine_current_validity",
]
