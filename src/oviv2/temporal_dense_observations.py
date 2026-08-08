from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy import ndimage

from src.core.data_structures import Frame
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.observations import (
    FrameObservation,
    ObservationKind,
    lift_mask_to_voxels,
)

_IMAGE_SHAPE = (480, 720)
_NATIVE_SHAPE = (120, 180)
_SAMPLE_STRIDE = 4
_OBSERVATION_ID_BASE = 2**61
_OBSERVATION_ID_FRAME_STRIDE = 2**20
_INT64_MAX = 2**63 - 1
_FOUR_CONNECTED = np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8)


def _exact_positive_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class TemporalDenseObservationConfig:
    sample_stride: int
    depth_max_m: float
    minimum_area_px: int
    maximum_area_px: int
    maximum_observations: int
    voxel_size_m: float
    pixel_stride: int
    min_valid_points: int

    def __post_init__(self) -> None:
        sample_stride = _exact_positive_int(self.sample_stride, "sample_stride")
        if sample_stride != _SAMPLE_STRIDE:
            raise ValueError("sample_stride must equal 4")
        minimum_area = _exact_positive_int(self.minimum_area_px, "minimum_area_px")
        maximum_area = _exact_positive_int(self.maximum_area_px, "maximum_area_px")
        if maximum_area < minimum_area:
            raise ValueError("maximum_area_px must be at least minimum_area_px")
        maximum_observations = _exact_positive_int(
            self.maximum_observations, "maximum_observations"
        )
        if maximum_observations > _OBSERVATION_ID_FRAME_STRIDE:
            raise ValueError("maximum_observations cannot exceed 2**20")
        object.__setattr__(self, "sample_stride", sample_stride)
        object.__setattr__(
            self, "depth_max_m", _finite_positive(self.depth_max_m, "depth_max_m")
        )
        object.__setattr__(self, "minimum_area_px", minimum_area)
        object.__setattr__(self, "maximum_area_px", maximum_area)
        object.__setattr__(self, "maximum_observations", maximum_observations)
        object.__setattr__(
            self, "voxel_size_m", _finite_positive(self.voxel_size_m, "voxel_size_m")
        )
        object.__setattr__(
            self, "pixel_stride", _exact_positive_int(self.pixel_stride, "pixel_stride")
        )
        object.__setattr__(
            self,
            "min_valid_points",
            _exact_positive_int(self.min_valid_points, "min_valid_points"),
        )


@dataclass(frozen=True)
class _Candidate:
    area: int
    confidence: float
    semantic_id: int
    first_native_flat: int
    native_slice: tuple[slice, slice]
    native_mask: np.ndarray


def _normalize_class_names(class_names: object, class_count: int) -> tuple[str, ...]:
    if type(class_names) is not tuple:
        raise TypeError("class_names must be an exact tuple")
    if len(class_names) != class_count:
        raise ValueError("class_names length must equal dense class_count")
    normalized: list[str] = []
    for value in class_names:
        if not isinstance(value, str):
            raise TypeError("class_names must contain strings")
        label = value.strip().lower().replace("_", "-")
        if not label:
            raise ValueError("class_names must contain non-empty labels")
        normalized.append(label)
    if len(normalized) != len(set(normalized)):
        raise ValueError("class_names must be unique after normalization")
    return tuple(normalized)


def _validate_inputs(
    frame: Frame,
    dense: DenseSemanticFrame,
    class_names: object,
    config: TemporalDenseObservationConfig,
) -> tuple[str, ...]:
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    if not isinstance(dense, DenseSemanticFrame):
        raise TypeError("dense must be a DenseSemanticFrame")
    if not isinstance(config, TemporalDenseObservationConfig):
        raise TypeError("config must be a TemporalDenseObservationConfig")
    if type(frame.frame_id) is not int:
        raise TypeError("frame_id must be an exact integer")
    if frame.frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    if frame.source_frame_id is not None and type(frame.source_frame_id) is not int:
        raise TypeError("source_frame_id must be an exact integer or None")
    if not isinstance(frame.timestamp, Real) or isinstance(
        frame.timestamp, (bool, np.bool_)
    ):
        raise TypeError("frame timestamp must be numeric")
    if not math.isfinite(float(frame.timestamp)):
        raise ValueError("frame timestamp must be finite")
    if not isinstance(frame.depth, np.ndarray) or frame.depth.shape != _IMAGE_SHAPE:
        raise ValueError("frame depth must have shape (480, 720)")
    if not isinstance(frame.rgb, np.ndarray) or frame.rgb.shape != (*_IMAGE_SHAPE, 3):
        raise ValueError("frame rgb must have shape (480, 720, 3)")
    pose = np.asarray(frame.pose)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("frame pose must be a finite (4, 4) matrix")
    intrinsics = frame.intrinsics
    values = np.asarray(
        (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy), dtype=np.float64
    )
    if (
        not np.isfinite(values).all()
        or intrinsics.fx == 0.0
        or intrinsics.fy == 0.0
        or (intrinsics.height, intrinsics.width) != _IMAGE_SHAPE
    ):
        raise ValueError("frame intrinsics must match the 480x720 image and be finite")
    if dense.cache_frame_id != frame.frame_id:
        raise ValueError("dense cache_frame_id must match frame_id")
    expected_source = (
        frame.frame_id if frame.source_frame_id is None else frame.source_frame_id
    )
    if dense.source_frame_id != expected_source:
        raise ValueError("dense source_frame_id must match frame source_frame_id")
    if dense.image_shape != _IMAGE_SHAPE:
        raise ValueError("dense image_shape must equal (480, 720)")
    if dense.sample_stride != config.sample_stride:
        raise ValueError("dense sample_stride must equal config sample_stride")
    if dense.class_ids.shape[:2] != _NATIVE_SHAPE:
        raise ValueError("dense native shape must equal (120, 180)")
    maximum_id = (
        _OBSERVATION_ID_BASE
        + frame.frame_id * _OBSERVATION_ID_FRAME_STRIDE
        + config.maximum_observations
        - 1
    )
    if maximum_id > _INT64_MAX:
        raise ValueError("generated observation IDs must fit signed int64")
    return _normalize_class_names(class_names, dense.class_count)


def _expanded(array: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(array, _SAMPLE_STRIDE, axis=0), _SAMPLE_STRIDE, axis=1)


def _full_resolution_slice(native_slice: tuple[slice, slice]) -> tuple[slice, slice]:
    rows, columns = native_slice
    return (
        slice(rows.start * _SAMPLE_STRIDE, rows.stop * _SAMPLE_STRIDE),
        slice(columns.start * _SAMPLE_STRIDE, columns.stop * _SAMPLE_STRIDE),
    )


def _view_direction(
    frame: Frame, centroid: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    direction = np.asarray(centroid, dtype=np.float64) - np.asarray(
        frame.pose[:3, 3], dtype=np.float64
    )
    if not np.isfinite(direction).all() or not np.any(direction):
        return None
    direction /= np.linalg.norm(direction)
    return tuple(float(value) for value in direction)


def generate_temporal_dense_observations(
    frame: Frame,
    dense: DenseSemanticFrame,
    class_names: tuple[str, ...],
    config: TemporalDenseObservationConfig,
) -> tuple[FrameObservation, ...]:
    """Generate current-frame object observations from dense top-1 semantics."""
    labels = _validate_inputs(frame, dense, class_names, config)
    top_ids = dense.class_ids[..., 0]
    top_probabilities = dense.probabilities[..., 0]
    valid_depth = (
        np.isfinite(frame.depth)
        & (frame.depth > 0.0)
        & (frame.depth <= config.depth_max_m)
    )
    candidates: list[_Candidate] = []

    for semantic_id in range(1, dense.class_count + 1):
        components, _ = ndimage.label(top_ids == semantic_id, structure=_FOUR_CONNECTED)
        for component_id, native_slice in enumerate(
            ndimage.find_objects(components), start=1
        ):
            if native_slice is None:
                continue
            native_mask = components[native_slice] == component_id
            native_rows, native_columns = np.nonzero(native_mask)
            first_native_flat = int(
                (native_rows[0] + native_slice[0].start) * _NATIVE_SHAPE[1]
                + native_columns[0]
                + native_slice[1].start
            )
            full_slice = _full_resolution_slice(native_slice)
            local_mask = _expanded(native_mask) & valid_depth[full_slice]
            area = int(np.count_nonzero(local_mask))
            if not config.minimum_area_px <= area <= config.maximum_area_px:
                continue
            local_probabilities = _expanded(top_probabilities[native_slice])
            confidence = float(
                np.mean(local_probabilities[local_mask], dtype=np.float64)
            )
            candidates.append(
                _Candidate(
                    area,
                    confidence,
                    semantic_id,
                    first_native_flat,
                    native_slice,
                    np.array(native_mask, copy=True),
                )
            )

    candidates.sort(
        key=lambda item: (
            -item.area,
            -item.confidence,
            item.semantic_id,
            item.first_native_flat,
        )
    )
    observations: list[FrameObservation] = []
    for candidate in candidates:
        if len(observations) == config.maximum_observations:
            break
        mask = np.zeros(_IMAGE_SHAPE, dtype=bool)
        full_slice = _full_resolution_slice(candidate.native_slice)
        mask[full_slice] = _expanded(candidate.native_mask) & valid_depth[full_slice]
        lifted = lift_mask_to_voxels(
            frame,
            mask,
            voxel_size_m=config.voxel_size_m,
            pixel_stride=config.pixel_stride,
            min_valid_points=config.min_valid_points,
        )
        if lifted is None:
            continue
        voxel_keys, centroid, bounds_min, bounds_max = lifted
        rows, columns = np.nonzero(mask)
        border = np.zeros(_IMAGE_SHAPE, dtype=bool)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        border_pixels = int(np.count_nonzero(mask & border))
        rank = len(observations)
        observations.append(
            FrameObservation(
                observation_id=(
                    _OBSERVATION_ID_BASE
                    + frame.frame_id * _OBSERVATION_ID_FRAME_STRIDE
                    + rank
                ),
                frame_id=frame.frame_id,
                timestamp=float(frame.timestamp),
                kind=ObservationKind.OBJECT,
                label=labels[candidate.semantic_id - 1],
                semantic_id=candidate.semantic_id,
                confidence=candidate.confidence,
                mask=mask,
                bbox_xyxy=(
                    float(columns.min()),
                    float(rows.min()),
                    float(columns.max() + 1),
                    float(rows.max() + 1),
                ),
                voxel_keys=voxel_keys,
                centroid_xyz=centroid,
                bounds_min_xyz=bounds_min,
                bounds_max_xyz=bounds_max,
                view_direction_xyz=_view_direction(frame, centroid),
                visible_pixel_count=candidate.area,
                border_contact_fraction=border_pixels / candidate.area,
            )
        )
    return tuple(observations)
