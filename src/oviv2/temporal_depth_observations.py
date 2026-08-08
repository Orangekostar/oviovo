from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy import ndimage

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.observations import (
    FrameObservation,
    ObservationKind,
    lift_mask_to_voxels,
)

_IMAGE_SHAPE = (480, 720)
_NATIVE_SHAPE = (120, 180)
_SAMPLE_STRIDE = 4
_OBSERVATION_ID_BASE = 3 * 2**61
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


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _unit_interval(value: object, name: str, *, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    lower_valid = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not lower_valid or result > 1.0:
        qualifier = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must lie in {qualifier}")
    return result


@dataclass(frozen=True)
class TemporalDepthObservationConfig:
    edge_threshold_m: float
    minimum_area_px: int
    maximum_area_px: int
    plane_minimum_area_px: int
    planar_rmse_threshold_m: float
    semantic_minimum_votes: int
    semantic_minimum_fraction: float
    semantic_minimum_probability: float
    maximum_observations: int
    maximum_unknown_observations: int
    depth_max_m: float
    voxel_size_m: float
    pixel_stride: int
    min_valid_points: int

    def __post_init__(self) -> None:
        minimum_area = _exact_positive_int(self.minimum_area_px, "minimum_area_px")
        maximum_area = _exact_positive_int(self.maximum_area_px, "maximum_area_px")
        if maximum_area < minimum_area:
            raise ValueError("maximum_area_px must be at least minimum_area_px")
        maximum_observations = _exact_positive_int(
            self.maximum_observations, "maximum_observations"
        )
        if maximum_observations > _OBSERVATION_ID_FRAME_STRIDE:
            raise ValueError("maximum_observations cannot exceed 2**20")
        maximum_unknown_observations = _exact_positive_int(
            self.maximum_unknown_observations,
            "maximum_unknown_observations",
        )

        object.__setattr__(
            self,
            "edge_threshold_m",
            _finite_positive(self.edge_threshold_m, "edge_threshold_m"),
        )
        object.__setattr__(self, "minimum_area_px", minimum_area)
        object.__setattr__(self, "maximum_area_px", maximum_area)
        object.__setattr__(
            self,
            "plane_minimum_area_px",
            _exact_positive_int(self.plane_minimum_area_px, "plane_minimum_area_px"),
        )
        object.__setattr__(
            self,
            "planar_rmse_threshold_m",
            _finite_nonnegative(
                self.planar_rmse_threshold_m, "planar_rmse_threshold_m"
            ),
        )
        object.__setattr__(
            self,
            "semantic_minimum_votes",
            _exact_positive_int(self.semantic_minimum_votes, "semantic_minimum_votes"),
        )
        object.__setattr__(
            self,
            "semantic_minimum_fraction",
            _unit_interval(
                self.semantic_minimum_fraction,
                "semantic_minimum_fraction",
                positive=False,
            ),
        )
        object.__setattr__(
            self,
            "semantic_minimum_probability",
            _unit_interval(
                self.semantic_minimum_probability,
                "semantic_minimum_probability",
                positive=False,
            ),
        )
        object.__setattr__(self, "maximum_observations", maximum_observations)
        object.__setattr__(
            self,
            "maximum_unknown_observations",
            maximum_unknown_observations,
        )
        object.__setattr__(
            self, "depth_max_m", _finite_positive(self.depth_max_m, "depth_max_m")
        )
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
    label: str
    first_full_flat: int
    component_slice: tuple[slice, slice]
    local_mask: np.ndarray


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
    config: TemporalDepthObservationConfig,
) -> tuple[str, ...]:
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    if not isinstance(dense, DenseSemanticFrame):
        raise TypeError("dense must be a DenseSemanticFrame")
    if not isinstance(config, TemporalDepthObservationConfig):
        raise TypeError("config must be a TemporalDepthObservationConfig")
    if type(frame.frame_id) is not int:
        raise TypeError("frame_id must be an exact integer")
    if frame.frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    if frame.source_frame_id is not None and type(frame.source_frame_id) is not int:
        raise TypeError("source_frame_id must be an exact integer or None")
    if isinstance(frame.timestamp, (bool, np.bool_)) or not isinstance(
        frame.timestamp, Real
    ):
        raise TypeError("frame timestamp must be numeric")
    if not math.isfinite(float(frame.timestamp)):
        raise ValueError("frame timestamp must be finite")
    if not isinstance(frame.depth, np.ndarray) or frame.depth.shape != _IMAGE_SHAPE:
        raise ValueError("frame depth must have shape (480, 720)")
    if not np.issubdtype(frame.depth.dtype, np.floating):
        raise TypeError("frame depth must have a floating dtype")
    if not isinstance(frame.rgb, np.ndarray) or frame.rgb.shape != (*_IMAGE_SHAPE, 3):
        raise ValueError("frame rgb must have shape (480, 720, 3)")
    if not isinstance(frame.pose, np.ndarray):
        raise TypeError("frame pose must be a numpy ndarray")
    if frame.pose.shape != (4, 4) or not np.isfinite(frame.pose).all():
        raise ValueError("frame pose must be a finite (4, 4) matrix")
    if not isinstance(frame.intrinsics, CameraIntrinsics):
        raise TypeError("frame intrinsics must be CameraIntrinsics")
    intrinsics = frame.intrinsics
    if type(intrinsics.width) is not int or type(intrinsics.height) is not int:
        raise TypeError("frame intrinsics dimensions must be exact integers")
    intrinsic_values = np.asarray(
        (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy), dtype=np.float64
    )
    if (
        not np.isfinite(intrinsic_values).all()
        or intrinsics.fx == 0.0
        or intrinsics.fy == 0.0
        or (intrinsics.height, intrinsics.width) != _IMAGE_SHAPE
    ):
        raise ValueError("frame intrinsics must match the 480x720 image and be finite")
    if dense.cache_frame_id != frame.frame_id:
        raise ValueError("dense cache_frame_id must match frame_id")
    expected_source = frame.frame_id if frame.source_frame_id is None else frame.source_frame_id
    if dense.source_frame_id != expected_source:
        raise ValueError("dense source_frame_id must match frame source_frame_id")
    if dense.image_shape != _IMAGE_SHAPE:
        raise ValueError("dense image_shape must equal (480, 720)")
    if dense.sample_stride != _SAMPLE_STRIDE:
        raise ValueError("dense sample_stride must equal 4")
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


def _semantic_assignment(
    component_slice: tuple[slice, slice],
    local_mask: np.ndarray,
    dense: DenseSemanticFrame,
    labels: tuple[str, ...],
    config: TemporalDepthObservationConfig,
) -> tuple[int, str, float]:
    local_rows, local_columns = np.nonzero(local_mask)
    rows = local_rows + component_slice[0].start
    columns = local_columns + component_slice[1].start
    sampled = (rows % _SAMPLE_STRIDE == 0) & (columns % _SAMPLE_STRIDE == 0)
    native_rows = rows[sampled] // _SAMPLE_STRIDE
    native_columns = columns[sampled] // _SAMPLE_STRIDE
    sampled_count = int(native_rows.size)
    if sampled_count == 0:
        return 0, "unknown", 0.0
    top_ids = dense.class_ids[native_rows, native_columns, 0]
    top_probabilities = dense.probabilities[native_rows, native_columns, 0]
    positive = top_ids > 0
    positive_count = int(np.count_nonzero(positive))
    if positive_count == 0:
        return 0, "unknown", 0.0
    positive_ids = top_ids[positive]
    counts = np.bincount(positive_ids, minlength=dense.class_count + 1)
    winner = int(np.argmax(counts[1:]) + 1)
    winner_votes = int(counts[winner])
    winner_probabilities = top_probabilities[positive][positive_ids == winner]
    confidence = float(np.mean(winner_probabilities, dtype=np.float64))
    if (
        winner_votes < config.semantic_minimum_votes
        or winner_votes / sampled_count < config.semantic_minimum_fraction
        or confidence < config.semantic_minimum_probability
    ):
        return 0, "unknown", 0.0
    return winner, labels[winner - 1], confidence


def _is_planar(
    component_slice: tuple[slice, slice],
    local_mask: np.ndarray,
    depth: np.ndarray,
    threshold: float,
) -> bool:
    local_rows, local_columns = np.nonzero(local_mask)
    rows = local_rows + component_slice[0].start
    columns = local_columns + component_slice[1].start
    design = np.column_stack(
        (
            columns.astype(np.float64) / 719.0,
            rows.astype(np.float64) / 479.0,
            np.ones(rows.size, dtype=np.float64),
        )
    )
    values = depth[rows, columns].astype(np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    rmse = float(np.sqrt(np.mean(residual * residual, dtype=np.float64)))
    return rmse <= threshold


def _view_direction(
    frame: Frame, centroid: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    direction = np.asarray(centroid, dtype=np.float64) - frame.pose[:3, 3]
    if not np.isfinite(direction).all() or not np.any(direction):
        return None
    direction /= np.linalg.norm(direction)
    return tuple(float(value) for value in direction)


def generate_temporal_depth_observations(
    frame: Frame,
    dense: DenseSemanticFrame,
    class_names: tuple[str, ...],
    config: TemporalDepthObservationConfig,
) -> tuple[FrameObservation, ...]:
    """Generate target-blind current-frame observations from depth connectivity."""
    labels = _validate_inputs(frame, dense, class_names, config)
    depth = frame.depth
    gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    gy = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
    valid_depth = np.isfinite(depth) & (depth > 0.0) & (depth <= config.depth_max_m)
    edges = (gx > config.edge_threshold_m) | (gy > config.edge_threshold_m)
    components, component_count = ndimage.label(
        valid_depth & ~edges, structure=_FOUR_CONNECTED
    )
    candidates: list[_Candidate] = []

    for component_id, component_slice in enumerate(
        ndimage.find_objects(components, max_label=component_count), start=1
    ):
        if component_slice is None:
            continue
        local_mask = components[component_slice] == component_id
        area = int(np.count_nonzero(local_mask))
        if not config.minimum_area_px <= area <= config.maximum_area_px:
            continue
        if area >= config.plane_minimum_area_px and _is_planar(
            component_slice,
            local_mask,
            depth,
            config.planar_rmse_threshold_m,
        ):
            continue
        semantic_id, label, confidence = _semantic_assignment(
            component_slice, local_mask, dense, labels, config
        )
        local_rows, local_columns = np.nonzero(local_mask)
        first_full_flat = int(
            (local_rows[0] + component_slice[0].start) * _IMAGE_SHAPE[1]
            + local_columns[0]
            + component_slice[1].start
        )
        candidates.append(
            _Candidate(
                area=area,
                confidence=confidence,
                semantic_id=semantic_id,
                label=label,
                first_full_flat=first_full_flat,
                component_slice=component_slice,
                local_mask=np.array(local_mask, copy=True),
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.area,
            -item.confidence,
            item.semantic_id,
            item.first_full_flat,
        )
    )
    border = np.zeros(_IMAGE_SHAPE, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    observations: list[FrameObservation] = []
    unknown_observation_count = 0
    for candidate in candidates:
        if len(observations) == config.maximum_observations:
            break
        if (
            candidate.semantic_id == 0
            and unknown_observation_count == config.maximum_unknown_observations
        ):
            continue
        mask = np.zeros(_IMAGE_SHAPE, dtype=bool)
        mask[candidate.component_slice] = candidate.local_mask
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
                label=candidate.label,
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
        if candidate.semantic_id == 0:
            unknown_observation_count += 1
    return tuple(observations)
