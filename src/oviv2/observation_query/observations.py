"""Geometry and sparse-incidence primitives for raw OVI observations."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral, Real

import numpy as np

from src.oviv2.observation_query.contracts import (
    OBSERVATION_METADATA_COLUMNS,
    ObservationBank,
    ObservationBankError,
)


class DepthRelation(IntEnum):
    UNKNOWN = 0
    SUPPORTED = 1
    OCCLUDED = 2
    VISIBLE_FREE = 3


@dataclass(frozen=True, slots=True)
class RawRegionObservation:
    """One raw frontend parent or its same-frame depth fragment."""

    key: str
    parent_region_key: str
    scan_uuid: str
    visit_id: int
    source_frame_id: int
    frontend_instance_id: int
    fragment_id: int
    depth_shape: tuple[int, int]
    bbox_xyxy: tuple[int, int, int, int]
    parent_color_bbox_xyxy: tuple[int, int, int, int]
    mask_pixel_indices: np.ndarray
    support_pixel_indices: np.ndarray
    valid_depth_fraction: float
    border_contact_fraction: float

    def __post_init__(self) -> None:
        if self.key != stable_region_key(
            self.scan_uuid,
            self.visit_id,
            self.source_frame_id,
            self.frontend_instance_id,
            self.fragment_id,
        ):
            raise ObservationBankError("raw region key disagrees with its identity")
        expected_parent = stable_region_key(
            self.scan_uuid,
            self.visit_id,
            self.source_frame_id,
            self.frontend_instance_id,
            -1,
        )
        if self.parent_region_key != expected_parent:
            raise ObservationBankError("raw region parent key is invalid")
        height, width = self.depth_shape
        if height <= 0 or width <= 0:
            raise ObservationBankError("raw region depth shape must be positive")
        for bbox, shape, name in (
            (self.bbox_xyxy, self.depth_shape, "bbox_xyxy"),
            (self.parent_color_bbox_xyxy, None, "parent_color_bbox_xyxy"),
        ):
            if len(bbox) != 4 or any(
                isinstance(value, bool) or not isinstance(value, Integral) for value in bbox
            ):
                raise ObservationBankError(f"{name} must contain four integers")
            x1, y1, x2, y2 = (int(value) for value in bbox)
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                raise ObservationBankError(f"{name} is not a positive half-open box")
            if shape is not None and (x2 > shape[1] or y2 > shape[0]):
                raise ObservationBankError(f"{name} exceeds the image")
        mask = np.asarray(self.mask_pixel_indices)
        support = np.asarray(self.support_pixel_indices)
        for values, name in ((mask, "mask_pixel_indices"), (support, "support_pixel_indices")):
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise ObservationBankError(f"{name} must be a one-dimensional integer array")
            if len(values) == 0 or np.any(values < 0) or np.any(values >= height * width):
                raise ObservationBankError(f"{name} is outside the depth image")
            if not np.array_equal(values, np.unique(values)):
                raise ObservationBankError(f"{name} must be sorted and unique")
        if not np.all(np.isin(support, mask, assume_unique=True)):
            raise ObservationBankError("support pixels must be a subset of the region mask")
        valid_fraction = float(self.valid_depth_fraction)
        border_fraction = float(self.border_contact_fraction)
        if not math.isclose(valid_fraction, len(support) / len(mask), rel_tol=1e-6, abs_tol=1e-6):
            raise ObservationBankError("valid_depth_fraction disagrees with region pixels")
        if not 0.0 <= border_fraction <= 1.0:
            raise ObservationBankError("border_contact_fraction must be in [0, 1]")
        mask = np.array(mask, dtype=np.int64, copy=True, order="C")
        support = np.array(support, dtype=np.int64, copy=True, order="C")
        mask.setflags(write=False)
        support.setflags(write=False)
        object.__setattr__(self, "mask_pixel_indices", mask)
        object.__setattr__(self, "support_pixel_indices", support)
        object.__setattr__(self, "valid_depth_fraction", valid_fraction)
        object.__setattr__(self, "border_contact_fraction", border_fraction)


@dataclass(frozen=True, slots=True)
class RegionSupportCandidate:
    """One positive projection before physical-pixel deduplication."""

    region_index: int
    model_index: int
    visit_id: int
    source_frame_id: int
    pixel_index: int
    weight: float
    is_fragment: bool

    def __post_init__(self) -> None:
        for name in (
            "region_index",
            "model_index",
            "visit_id",
            "source_frame_id",
            "pixel_index",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))
        if self.visit_id not in (0, 1):
            raise ValueError("visit_id must be 0 or 1")
        if isinstance(self.weight, (bool, np.bool_)) or not isinstance(self.weight, Real):
            raise TypeError("weight must be real")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be finite and nonnegative")
        if not isinstance(self.is_fragment, (bool, np.bool_)):
            raise TypeError("is_fragment must be boolean")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "is_fragment", bool(self.is_fragment))


def stable_region_key(
    scan_uuid: str,
    visit_id: int,
    source_frame_id: int,
    frontend_instance_id: int,
    fragment_id: int,
) -> str:
    """Encode the five-part raw-region identity without lossy delimiters."""

    if not isinstance(scan_uuid, str) or not scan_uuid.strip():
        raise ValueError("scan_uuid must be a non-empty string")
    values = (visit_id, source_frame_id, frontend_instance_id, fragment_id)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in values):
        raise TypeError("region identity indices must be integers")
    if visit_id not in (0, 1):
        raise ValueError("visit_id must be 0 or 1")
    if source_frame_id < 0 or frontend_instance_id <= 0 or fragment_id < -1:
        raise ValueError("region identity indices are outside their legal ranges")
    return json.dumps(
        [
            scan_uuid.strip(),
            int(visit_id),
            int(source_frame_id),
            int(frontend_instance_id),
            int(fragment_id),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )


def classify_depth_relations(
    *,
    observed_depth_m: object,
    projected_depth_m: object,
    neighborhood_reliable: object,
    tolerance_m: float,
) -> np.ndarray:
    """Classify signed observed-minus-projected depth residuals."""

    observed = np.asarray(observed_depth_m, dtype=np.float64)
    projected = np.asarray(projected_depth_m, dtype=np.float64)
    reliable = np.asarray(neighborhood_reliable)
    if observed.shape != projected.shape or observed.shape != reliable.shape:
        raise ObservationBankError("depth relation inputs must have identical shapes")
    if reliable.dtype != np.bool_:
        raise ObservationBankError("neighborhood_reliable must be boolean")
    if isinstance(tolerance_m, (bool, np.bool_)) or not isinstance(tolerance_m, Real):
        raise TypeError("tolerance_m must be real")
    tolerance = float(tolerance_m)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance_m must be finite and positive")

    result = np.full(observed.shape, DepthRelation.UNKNOWN, dtype=np.int8)
    valid = (
        reliable
        & np.isfinite(observed)
        & np.isfinite(projected)
        & (observed > 0.0)
        & (projected > 0.0)
    )
    residual = observed - projected
    absolute_residual = np.abs(residual)
    within_tolerance = (absolute_residual <= tolerance) | np.isclose(
        absolute_residual, tolerance, rtol=1e-12, atol=1e-12
    )
    result[valid & within_tolerance] = DepthRelation.SUPPORTED
    result[valid & ~within_tolerance & (residual < 0.0)] = DepthRelation.OCCLUDED
    result[valid & ~within_tolerance & (residual > 0.0)] = DepthRelation.VISIBLE_FREE
    result.setflags(write=False)
    return result


def register_labels_nearest(
    frontend_labels: object,
    color_to_depth_homography: object,
    depth_shape: tuple[int, int],
) -> np.ndarray:
    """Register an integer label image without interpolating instance IDs."""

    labels = np.asarray(frontend_labels)
    transform = np.asarray(color_to_depth_homography, dtype=np.float64)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ObservationBankError("frontend_labels must be a two-dimensional integer image")
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        raise ObservationBankError("color_to_depth_homography must be finite 3x3")
    if (
        not isinstance(depth_shape, tuple)
        or len(depth_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in depth_shape)
    ):
        raise ObservationBankError("depth_shape must contain positive integer height and width")
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError as error:
        raise ObservationBankError("color_to_depth_homography must be invertible") from error

    height, width = (int(depth_shape[0]), int(depth_shape[1]))
    rows, columns = np.indices((height, width), dtype=np.float64)
    target = np.stack((columns.ravel(), rows.ravel(), np.ones(height * width)), axis=0)
    source = inverse @ target
    scale = source[2]
    finite = np.isfinite(source).all(axis=0) & (np.abs(scale) > np.finfo(np.float64).eps)
    source_columns = np.zeros(height * width, dtype=np.int64)
    source_rows = np.zeros(height * width, dtype=np.int64)
    source_columns[finite] = np.floor(source[0, finite] / scale[finite] + 0.5).astype(np.int64)
    source_rows[finite] = np.floor(source[1, finite] / scale[finite] + 0.5).astype(np.int64)
    inside = (
        finite
        & (source_columns >= 0)
        & (source_columns < labels.shape[1])
        & (source_rows >= 0)
        & (source_rows < labels.shape[0])
    )
    result = np.zeros(height * width, dtype=labels.dtype)
    result[inside] = labels[source_rows[inside], source_columns[inside]]
    return np.ascontiguousarray(result.reshape(height, width))


def select_observation_frames(
    *, frame_count: int, maximum_frames: int
) -> tuple[int, ...]:
    """Select a deterministic, endpoint-preserving uniform frame subset."""

    for value, name in ((frame_count, "frame_count"), (maximum_frames, "maximum_frames")):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ObservationBankError(f"{name} must be a positive integer")
    count = min(int(frame_count), int(maximum_frames))
    if count == int(frame_count):
        return tuple(range(int(frame_count)))
    selected = np.rint(np.linspace(0, int(frame_count) - 1, count)).astype(np.int64)
    if len(np.unique(selected)) != count:
        raise ObservationBankError("uniform frame selection produced duplicate frames")
    return tuple(int(value) for value in selected)


def _half_open_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _border_fraction(mask: np.ndarray) -> float:
    border = np.zeros(mask.shape, dtype=np.bool_)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    return float(np.count_nonzero(mask & border) / np.count_nonzero(mask))


def extract_raw_regions(
    *,
    scan_uuid: str,
    visit_id: int,
    source_frame_id: int,
    frontend_labels_color: object,
    geometric_labels_depth: object,
    depth_m: object,
    color_to_depth_homography: object,
    minimum_valid_depth_pixels: int,
) -> tuple[RawRegionObservation, ...]:
    """Rebuild traceable parent regions and intersections with raw depth fragments."""

    frontend = np.asarray(frontend_labels_color)
    geometric = np.asarray(geometric_labels_depth)
    depth = np.asarray(depth_m, dtype=np.float64)
    if geometric.ndim != 2 or not np.issubdtype(geometric.dtype, np.integer):
        raise ObservationBankError("geometric_labels_depth must be an integer image")
    if depth.shape != geometric.shape:
        raise ObservationBankError("depth and geometric labels must have identical shape")
    if (
        isinstance(minimum_valid_depth_pixels, bool)
        or not isinstance(minimum_valid_depth_pixels, Integral)
        or minimum_valid_depth_pixels <= 0
    ):
        raise ObservationBankError("minimum_valid_depth_pixels must be positive")
    registered = register_labels_nearest(
        frontend, color_to_depth_homography, geometric.shape
    )
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    result: list[RawRegionObservation] = []
    for raw_instance_id in np.unique(frontend):
        instance_id = int(raw_instance_id)
        if instance_id <= 0:
            continue
        color_mask = frontend == raw_instance_id
        depth_mask = registered == raw_instance_id
        support_mask = depth_mask & valid_depth
        if np.count_nonzero(support_mask) < minimum_valid_depth_pixels:
            continue
        parent_key = stable_region_key(
            scan_uuid, visit_id, source_frame_id, instance_id, -1
        )
        color_bbox = _half_open_bbox(color_mask)

        def append(
            mask: np.ndarray,
            support: np.ndarray,
            fragment_id: int,
            *,
            instance_id: int = instance_id,
            parent_key: str = parent_key,
            color_bbox: tuple[int, int, int, int] = color_bbox,
        ) -> None:
            mask_indices = np.flatnonzero(mask).astype(np.int64)
            support_indices = np.flatnonzero(support).astype(np.int64)
            result.append(
                RawRegionObservation(
                    key=stable_region_key(
                        scan_uuid,
                        visit_id,
                        source_frame_id,
                        instance_id,
                        fragment_id,
                    ),
                    parent_region_key=parent_key,
                    scan_uuid=scan_uuid,
                    visit_id=visit_id,
                    source_frame_id=source_frame_id,
                    frontend_instance_id=instance_id,
                    fragment_id=fragment_id,
                    depth_shape=geometric.shape,
                    bbox_xyxy=_half_open_bbox(mask),
                    parent_color_bbox_xyxy=color_bbox,
                    mask_pixel_indices=mask_indices,
                    support_pixel_indices=support_indices,
                    valid_depth_fraction=float(len(support_indices) / len(mask_indices)),
                    border_contact_fraction=_border_fraction(mask),
                )
            )

        append(depth_mask, support_mask, -1)
        fragment_ids = np.unique(geometric[depth_mask])
        for raw_fragment_id in fragment_ids:
            fragment_id = int(raw_fragment_id)
            if fragment_id <= 0:
                continue
            fragment_mask = depth_mask & (geometric == raw_fragment_id)
            fragment_support = fragment_mask & valid_depth
            if np.count_nonzero(fragment_support) < minimum_valid_depth_pixels:
                continue
            append(fragment_mask, fragment_support, fragment_id)
    return tuple(result)


def camera_to_reference(
    camera_to_local: object,
    visit_id: int,
    rescan_to_reference_row: object,
) -> np.ndarray:
    """Convert a column-vector camera pose using the official row alignment once."""

    pose = np.asarray(camera_to_local, dtype=np.float64)
    alignment = np.asarray(rescan_to_reference_row, dtype=np.float64)
    if pose.shape != (4, 4) or alignment.shape != (4, 4):
        raise ObservationBankError("camera pose and row alignment must be 4x4")
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(alignment)):
        raise ObservationBankError("camera pose and row alignment must be finite")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ObservationBankError("camera_to_local is not a column-vector pose")
    if not np.allclose(alignment[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ObservationBankError("rescan alignment is not a row-vector transform")
    if isinstance(visit_id, bool) or not isinstance(visit_id, Integral) or visit_id not in (0, 1):
        raise ObservationBankError("visit_id must be 0 or 1")
    result = pose if int(visit_id) == 0 else alignment.T @ pose
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def neighborhood_depth_reliability(
    depth_m: object,
    rows: object,
    columns: object,
    *,
    tolerance_m: float,
    minimum_valid_neighbours: int = 5,
    edge_range_factor: float = 2.0,
) -> np.ndarray:
    """Mark projections whose complete 3x3 neighborhood is locally reliable."""

    depth = np.asarray(depth_m, dtype=np.float64)
    row_values = np.asarray(rows)
    column_values = np.asarray(columns)
    if depth.ndim != 2:
        raise ObservationBankError("depth_m must be a two-dimensional image")
    if (
        row_values.ndim != 1
        or column_values.ndim != 1
        or len(row_values) != len(column_values)
        or not np.issubdtype(row_values.dtype, np.integer)
        or not np.issubdtype(column_values.dtype, np.integer)
    ):
        raise ObservationBankError("projection rows and columns must be aligned integer vectors")
    if isinstance(tolerance_m, bool) or not isinstance(tolerance_m, Real):
        raise TypeError("tolerance_m must be real")
    tolerance = float(tolerance_m)
    factor = float(edge_range_factor)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ObservationBankError("tolerance_m must be finite and positive")
    if not math.isfinite(factor) or factor <= 0.0:
        raise ObservationBankError("edge_range_factor must be finite and positive")
    if (
        isinstance(minimum_valid_neighbours, bool)
        or not isinstance(minimum_valid_neighbours, Integral)
        or not 1 <= minimum_valid_neighbours <= 9
    ):
        raise ObservationBankError("minimum_valid_neighbours must be in [1, 9]")

    height, width = depth.shape
    interior = (
        (row_values > 0)
        & (row_values < height - 1)
        & (column_values > 0)
        & (column_values < width - 1)
    )
    result = np.zeros(len(row_values), dtype=np.bool_)
    selected = np.flatnonzero(interior)
    if len(selected):
        selected_rows = row_values[selected].astype(np.int64)
        selected_columns = column_values[selected].astype(np.int64)
        neighbours = np.stack(
            [
                depth[selected_rows + row_offset, selected_columns + column_offset]
                for row_offset in (-1, 0, 1)
                for column_offset in (-1, 0, 1)
            ],
            axis=1,
        )
        valid = np.isfinite(neighbours) & (neighbours > 0.0)
        counts = valid.sum(axis=1)
        minimum = np.min(np.where(valid, neighbours, np.inf), axis=1)
        maximum = np.max(np.where(valid, neighbours, -np.inf), axis=1)
        result[selected] = (
            (counts >= int(minimum_valid_neighbours))
            & ((maximum - minimum) <= factor * tolerance)
        )
    result.setflags(write=False)
    return result


def build_region_metadata(
    region: RawRegionObservation,
    *,
    depth_m: object,
    depth_intrinsic: object,
    camera_to_reference: object,
    signed_depth_residuals_m: object | None = None,
) -> tuple[np.ndarray, float]:
    """Compute the documented non-GT geometry row and fixed reliability weight."""

    if not isinstance(region, RawRegionObservation):
        raise TypeError("region must be RawRegionObservation")
    depth = np.asarray(depth_m, dtype=np.float64)
    intrinsic = np.asarray(depth_intrinsic, dtype=np.float64)
    pose = np.asarray(camera_to_reference, dtype=np.float64)
    if depth.shape != region.depth_shape:
        raise ObservationBankError("metadata depth shape differs from the raw region")
    if intrinsic.shape != (3, 3) or pose.shape != (4, 4):
        raise ObservationBankError("metadata calibration matrices have invalid shapes")
    if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(pose)):
        raise ObservationBankError("metadata calibration matrices must be finite")
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    if fx <= 0.0 or fy <= 0.0:
        raise ObservationBankError("metadata focal lengths must be positive")

    rows, columns = np.unravel_index(region.support_pixel_indices, region.depth_shape)
    z = depth[rows, columns]
    if np.any(~np.isfinite(z)) or np.any(z <= 0.0):
        raise ObservationBankError("raw region support contains invalid depth")
    camera_points = np.column_stack(
        ((columns - cx) * z / fx, (rows - cy) * z / fy, z)
    )
    reference_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
    center = reference_points.mean(axis=0, dtype=np.float64)
    extent = reference_points.max(axis=0) - reference_points.min(axis=0)
    direction = center - pose[:3, 3]
    direction_norm = np.linalg.norm(direction)
    direction = direction / direction_norm if direction_norm > 0.0 else np.zeros(3)
    if signed_depth_residuals_m is None:
        residual_mean = residual_std = 0.0
    else:
        residuals = np.asarray(signed_depth_residuals_m, dtype=np.float64)
        if residuals.ndim != 1 or np.any(~np.isfinite(residuals)):
            raise ObservationBankError("signed depth residuals must be a finite vector")
        residual_mean = float(residuals.mean()) if len(residuals) else 0.0
        residual_std = float(residuals.std()) if len(residuals) else 0.0
    metadata = np.asarray(
        [
            *center,
            *extent,
            region.valid_depth_fraction,
            *direction,
            residual_mean,
            residual_std,
            float(region.visit_id),
            region.border_contact_fraction,
            float(len(region.mask_pixel_indices)),
        ],
        dtype=np.float32,
    )
    if metadata.shape != (len(OBSERVATION_METADATA_COLUMNS),) or np.any(~np.isfinite(metadata)):
        raise ObservationBankError("computed region metadata is invalid")
    reliability = float(
        np.clip(
            region.valid_depth_fraction * (1.0 - 0.5 * region.border_contact_fraction),
            0.0,
            1.0,
        )
    )
    metadata.setflags(write=False)
    return metadata, reliability


def build_region_model_incidence(
    *,
    region_visit_ids: object,
    model_visit_ids: object,
    support_candidates: Iterable[RegionSupportCandidate],
    maximum_distinct_frames_per_model_point: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build R-major CSR while conserving each physical pixel's vote."""

    region_visits = np.asarray(region_visit_ids)
    model_visits = np.asarray(model_visit_ids)
    for values, name in (
        (region_visits, "region_visit_ids"),
        (model_visits, "model_visit_ids"),
    ):
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ObservationBankError(f"{name} must be a one-dimensional integer array")
        if np.any((values != 0) & (values != 1)):
            raise ObservationBankError(f"{name} must contain only 0 and 1")
    if (
        isinstance(maximum_distinct_frames_per_model_point, bool)
        or not isinstance(maximum_distinct_frames_per_model_point, Integral)
        or maximum_distinct_frames_per_model_point <= 0
    ):
        raise ObservationBankError(
            "maximum_distinct_frames_per_model_point must be a positive integer"
        )

    physical: dict[tuple[int, int, int, int], list[RegionSupportCandidate]] = defaultdict(list)
    for candidate in support_candidates:
        if not isinstance(candidate, RegionSupportCandidate):
            raise TypeError("support_candidates must contain RegionSupportCandidate")
        if candidate.region_index >= len(region_visits) or candidate.model_index >= len(model_visits):
            raise ObservationBankError("support candidate index is outside the R or M domain")
        if (
            int(region_visits[candidate.region_index]) != candidate.visit_id
            or int(model_visits[candidate.model_index]) != candidate.visit_id
        ):
            raise ObservationBankError("a support candidate crosses visits")
        if candidate.weight == 0.0:
            continue
        physical[
            (
                candidate.visit_id,
                candidate.source_frame_id,
                candidate.pixel_index,
                candidate.model_index,
            )
        ].append(candidate)

    atoms: list[tuple[int, int, int, float]] = []
    for candidates in physical.values():
        active = [candidate for candidate in candidates if candidate.is_fragment]
        if not active:
            active = candidates
        by_region: dict[int, float] = {}
        for candidate in active:
            by_region[candidate.region_index] = max(
                by_region.get(candidate.region_index, 0.0), candidate.weight
            )
        total = sum(by_region.values())
        conserved = max(by_region.values())
        scale = conserved / total
        frame_id = active[0].source_frame_id
        model_index = active[0].model_index
        atoms.extend(
            (region_index, model_index, frame_id, weight * scale)
            for region_index, weight in by_region.items()
        )

    frame_scores: dict[tuple[int, int], float] = defaultdict(float)
    for _region, model, frame, weight in atoms:
        frame_scores[(model, frame)] += weight
    selected_frames: dict[int, set[int]] = defaultdict(set)
    frames_by_model: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (model, frame), score in frame_scores.items():
        frames_by_model[model].append((frame, score))
    for model, frames in frames_by_model.items():
        ranked = sorted(frames, key=lambda item: (-item[1], item[0]))
        selected_frames[model].update(
            frame for frame, _score in ranked[:maximum_distinct_frames_per_model_point]
        )

    edge_weights: dict[tuple[int, int], float] = defaultdict(float)
    for region, model, frame, weight in atoms:
        if frame in selected_frames[model]:
            edge_weights[(region, model)] += weight

    ordered = sorted(edge_weights.items())
    counts = np.zeros(len(region_visits), dtype=np.int64)
    for (region, _model), _weight in ordered:
        counts[region] += 1
    indptr = np.empty(len(region_visits) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = np.asarray([key[1] for key, _weight in ordered], dtype=np.int64)
    weights = np.asarray([weight for _key, weight in ordered], dtype=np.float32)
    for value in (indptr, indices, weights):
        value.setflags(write=False)
    return indptr, indices, weights


def prepare_observation_bank(
    *,
    pair_id: str,
    model_input_sha256: str,
    region_keys: tuple[str, ...],
    region_visit_ids: object,
    region_frame_ids: object,
    region_features: object,
    region_metadata: object,
    region_reliability: object,
    model_visit_ids: object,
    support_candidates: Iterable[RegionSupportCandidate],
    source_manifest: dict[str, object],
    maximum_distinct_frames_per_model_point: int = 8,
) -> ObservationBank:
    """Assemble validated region arrays and positive supports into one bank."""

    indptr, indices, weights = build_region_model_incidence(
        region_visit_ids=region_visit_ids,
        model_visit_ids=model_visit_ids,
        support_candidates=support_candidates,
        maximum_distinct_frames_per_model_point=maximum_distinct_frames_per_model_point,
    )
    return ObservationBank(
        pair_id=pair_id,
        model_input_sha256=model_input_sha256,
        region_keys=region_keys,
        region_visit_ids=np.asarray(region_visit_ids),
        region_frame_ids=np.asarray(region_frame_ids),
        region_features=np.asarray(region_features),
        region_metadata=np.asarray(region_metadata),
        csr_indptr=indptr,
        csr_model_indices=indices,
        csr_weights=weights,
        region_reliability=np.asarray(region_reliability),
        model_visit_ids=np.asarray(model_visit_ids),
        source_manifest=source_manifest,
    )
