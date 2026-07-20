from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.evidence import SparseEvidenceStore


_INT64_MAX = int(np.iinfo(np.int64).max)
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_UPPER_EXCLUSIVE = float(2**63)


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0 or normalized > _INT64_MAX:
        raise ValueError(f"{field_name} must be a non-negative signed int64")
    return normalized


@dataclass(frozen=True)
class DenseSemanticConfig:
    voxel_size_m: float
    integration_radius_m: float = 6.0
    minimum_probability: float = 0.01
    minimum_quality: float = 0.01
    entropy_power: float = 1.0
    view_angle_power: float = 1.0

    def __post_init__(self) -> None:
        normalized = {
            name: _finite_number(getattr(self, name), name)
            for name in (
                "voxel_size_m",
                "integration_radius_m",
                "minimum_probability",
                "minimum_quality",
                "entropy_power",
                "view_angle_power",
            )
        }
        for name in ("voxel_size_m", "integration_radius_m"):
            if normalized[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("minimum_probability", "minimum_quality"):
            if not 0.0 <= normalized[name] <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("entropy_power", "view_angle_power"):
            if normalized[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class DenseProjectionResult:
    sampled_pixel_count: int
    valid_pixel_count: int
    updated_voxel_count: int

    def __post_init__(self) -> None:
        for name in (
            "sampled_pixel_count",
            "valid_pixel_count",
            "updated_voxel_count",
        ):
            object.__setattr__(
                self,
                name,
                _non_negative_integer(getattr(self, name), name),
            )
        if self.valid_pixel_count > self.sampled_pixel_count:
            raise ValueError("valid_pixel_count cannot exceed sampled_pixel_count")
        if self.updated_voxel_count > self.valid_pixel_count:
            raise ValueError("updated_voxel_count cannot exceed valid_pixel_count")


def _validate_dense_arrays(dense: DenseSemanticFrame) -> tuple[int, int, int]:
    stride = _non_negative_integer(dense.sample_stride, "sample_stride")
    if stride == 0:
        raise ValueError("sample_stride must be positive")
    class_count = _non_negative_integer(dense.class_count, "class_count")
    if class_count == 0:
        raise ValueError("class_count must be positive")
    if (
        not isinstance(dense.image_shape, tuple)
        or len(dense.image_shape) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in dense.image_shape
        )
    ):
        raise ValueError("dense image_shape must contain two positive integers")
    height, width = (int(value) for value in dense.image_shape)
    sampled_shape = (
        (height + stride - 1) // stride,
        (width + stride - 1) // stride,
    )
    if not isinstance(dense.class_ids, np.ndarray) or dense.class_ids.ndim != 3:
        raise ValueError("dense class_ids has an invalid sample grid shape or dtype")
    top_k = dense.class_ids.shape[2]
    expected_topk_shape = (*sampled_shape, top_k)
    array_contracts = (
        ("class_ids", dense.class_ids, np.dtype(np.int64), expected_topk_shape),
        (
            "probabilities",
            dense.probabilities,
            np.dtype(np.float32),
            expected_topk_shape,
        ),
        ("entropy", dense.entropy, np.dtype(np.float32), sampled_shape),
        ("margin", dense.margin, np.dtype(np.float32), sampled_shape),
    )
    for name, array, dtype, shape in array_contracts:
        if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != shape:
            raise ValueError(f"dense {name} has an invalid sample grid shape or dtype")
    if not 1 <= top_k <= class_count:
        raise ValueError("dense top-k must be between one and class_count")

    class_ids = dense.class_ids
    probabilities = dense.probabilities
    if np.any(class_ids < 0) or np.any(class_ids > class_count):
        raise ValueError("dense class_ids must be one-based and not exceed class_count")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0) or np.any(
        probabilities > 1.0
    ):
        raise ValueError("dense probabilities must be finite and lie in [0, 1]")
    if np.any((class_ids == 0) & (probabilities != 0.0)) or np.any(
        (class_ids > 0) & (probabilities <= 0.0)
    ):
        raise ValueError("dense class_ids and probabilities are inconsistent")
    if np.any(np.sum(probabilities, axis=-1, dtype=np.float64) > 1.0 + 1e-6):
        raise ValueError("dense probability mass must not exceed one")

    maximum_entropy = math.log(class_count)
    entropy_tolerance = 0.0 if class_count == 1 else 1e-6
    if (
        not np.all(np.isfinite(dense.entropy))
        or np.any(dense.entropy < 0.0)
        or np.any(dense.entropy > maximum_entropy + entropy_tolerance)
    ):
        raise ValueError("dense entropy is outside its valid range")
    if (
        not np.all(np.isfinite(dense.margin))
        or np.any(dense.margin < 0.0)
        or np.any(dense.margin > 1.0)
    ):
        raise ValueError("dense margin must be finite and lie in [0, 1]")
    return height, width, stride


def _validate_frame(
    frame: Frame,
    dense: DenseSemanticFrame,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    frame_id = _non_negative_integer(frame.frame_id, "frame.frame_id")
    if dense.source_frame_id != frame_id:
        raise ValueError("dense source_frame_id must equal frame.frame_id")
    if not isinstance(frame.rgb, np.ndarray) or frame.rgb.ndim != 3 or frame.rgb.shape[2] != 3:
        raise ValueError("frame RGB image must have shape (height, width, 3)")
    if not isinstance(frame.depth, np.ndarray) or frame.depth.ndim != 2:
        raise ValueError("frame depth must have shape (height, width)")
    if frame.rgb.shape[:2] != image_shape or frame.depth.shape != image_shape:
        raise ValueError("frame image and depth shapes must match dense image_shape")

    intrinsics = frame.intrinsics
    if not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError("frame intrinsics must be CameraIntrinsics")
    for name, expected in (("height", image_shape[0]), ("width", image_shape[1])):
        value = getattr(intrinsics, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) != expected
        ):
            raise ValueError("frame intrinsics image shape does not match dense image_shape")
    try:
        intrinsic_values = np.asarray(
            [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("frame intrinsics must contain numeric values") from exc
    if (
        not np.all(np.isfinite(intrinsic_values))
        or intrinsic_values[0] <= 0.0
        or intrinsic_values[1] <= 0.0
    ):
        raise ValueError("frame intrinsics must be finite with positive focal lengths")

    try:
        pose = np.asarray(frame.pose, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame pose must contain numeric values") from exc
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("frame pose must be a finite (4, 4) matrix")
    return np.asarray(frame.depth, dtype=np.float64), pose, intrinsics


def _neighbor_tangent(
    points: np.ndarray,
    valid: np.ndarray,
    *,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    before_points = np.zeros_like(points)
    after_points = np.zeros_like(points)
    before_valid = np.zeros_like(valid)
    after_valid = np.zeros_like(valid)
    if axis == 1:
        before_points[:, 1:] = points[:, :-1]
        after_points[:, :-1] = points[:, 1:]
        before_valid[:, 1:] = valid[:, :-1]
        after_valid[:, :-1] = valid[:, 1:]
    else:
        before_points[1:] = points[:-1]
        after_points[:-1] = points[1:]
        before_valid[1:] = valid[:-1]
        after_valid[:-1] = valid[1:]

    tangent = np.zeros_like(points)
    both = before_valid & after_valid
    after_only = ~both & after_valid
    before_only = ~both & ~after_valid & before_valid
    tangent[both] = after_points[both] - before_points[both]
    tangent[after_only] = after_points[after_only] - points[after_only]
    tangent[before_only] = points[before_only] - before_points[before_only]
    return tangent, both | after_only | before_only


def _view_cosine(camera_points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    horizontal, horizontal_valid = _neighbor_tangent(
        camera_points,
        valid,
        axis=1,
    )
    vertical, vertical_valid = _neighbor_tangent(
        camera_points,
        valid,
        axis=0,
    )
    normals = np.cross(horizontal, vertical)
    lengths = np.linalg.norm(normals, axis=-1)
    normal_valid = (
        valid
        & horizontal_valid
        & vertical_valid
        & np.isfinite(lengths)
        & (lengths > 0.0)
    )
    view_cosine = np.ones(valid.shape, dtype=np.float64)
    if not np.any(normal_valid):
        return view_cosine

    unit_normals = np.zeros_like(normals)
    unit_normals[normal_valid] = normals[normal_valid] / lengths[normal_valid, None]
    ray_lengths = np.linalg.norm(camera_points, axis=-1)
    ray_valid = normal_valid & np.isfinite(ray_lengths) & (ray_lengths > 0.0)
    surface_to_camera = np.zeros_like(camera_points)
    surface_to_camera[ray_valid] = -camera_points[ray_valid] / ray_lengths[ray_valid, None]
    facing = np.sum(unit_normals * surface_to_camera, axis=-1)
    unit_normals[facing < 0.0] *= -1.0
    cosine = np.sum(unit_normals * surface_to_camera, axis=-1)
    view_cosine[ray_valid] = np.clip(cosine[ray_valid], 0.0, 1.0)
    return view_cosine


def _aggregate_updates(
    voxel_keys: np.ndarray,
    class_ids: np.ndarray,
    support: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_pixel_count, top_k = class_ids.shape
    pixel_indices = np.broadcast_to(
        np.arange(sampled_pixel_count, dtype=np.int64)[:, None],
        (sampled_pixel_count, top_k),
    )
    selected_pixels = pixel_indices[candidate_mask]
    selected_labels = class_ids[candidate_mask]
    selected_support = support[candidate_mask]
    if selected_pixels.size == 0:
        return np.empty((0, 4), dtype=np.int64), np.empty(0, dtype=np.float64)

    pixel_label = np.column_stack((selected_pixels, selected_labels))
    unique_pixel_label, pixel_inverse = np.unique(
        pixel_label,
        axis=0,
        return_inverse=True,
    )
    per_pixel_support = np.full(unique_pixel_label.shape[0], -np.inf, dtype=np.float64)
    np.maximum.at(per_pixel_support, pixel_inverse, selected_support)

    deduplicated_pixels = unique_pixel_label[:, 0]
    voxel_label = np.column_stack(
        (
            voxel_keys[deduplicated_pixels],
            unique_pixel_label[:, 1],
        )
    )
    unique_voxel_label, voxel_inverse = np.unique(
        voxel_label,
        axis=0,
        return_inverse=True,
    )
    aggregated_support = np.zeros(unique_voxel_label.shape[0], dtype=np.float64)
    np.add.at(aggregated_support, voxel_inverse, per_pixel_support)
    if not np.all(np.isfinite(aggregated_support)) or np.any(aggregated_support <= 0.0):
        raise ValueError("aggregated semantic support must be finite and positive")
    order = np.lexsort(
        (
            unique_voxel_label[:, 3],
            unique_voxel_label[:, 2],
            unique_voxel_label[:, 1],
            unique_voxel_label[:, 0],
        )
    )
    return unique_voxel_label[order], aggregated_support[order]


class DenseSemanticIntegrator:
    def __init__(self, config: DenseSemanticConfig) -> None:
        if not isinstance(config, DenseSemanticConfig):
            raise TypeError("config must be a DenseSemanticConfig")
        self.config = config

    def integrate(
        self,
        frame: Frame,
        dense: DenseSemanticFrame,
        store: SparseEvidenceStore,
        revision: int,
    ) -> DenseProjectionResult:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        if not isinstance(dense, DenseSemanticFrame):
            raise TypeError("dense must be a DenseSemanticFrame")
        if not isinstance(store, SparseEvidenceStore):
            raise TypeError("store must be a SparseEvidenceStore")
        normalized_revision = _non_negative_integer(revision, "revision")
        height, width, stride = _validate_dense_arrays(dense)
        depth, pose, intrinsics = _validate_frame(frame, dense, (height, width))

        sampled_rows = np.arange(dense.entropy.shape[0], dtype=np.int64) * stride
        sampled_columns = np.arange(dense.entropy.shape[1], dtype=np.int64) * stride
        if sampled_rows[-1] >= height or sampled_columns[-1] >= width:
            raise ValueError("dense sample grid extends beyond the image")
        sampled_depth = depth[np.ix_(sampled_rows, sampled_columns)]
        rows, columns = np.meshgrid(
            sampled_rows.astype(np.float64),
            sampled_columns.astype(np.float64),
            indexing="ij",
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            camera_points = np.stack(
                (
                    (columns - intrinsics.cx) * sampled_depth / intrinsics.fx,
                    (rows - intrinsics.cy) * sampled_depth / intrinsics.fy,
                    sampled_depth,
                ),
                axis=-1,
            )
            camera_distance = np.linalg.norm(camera_points, axis=-1)
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth > 0.0)
            & np.all(np.isfinite(camera_points), axis=-1)
            & np.isfinite(camera_distance)
            & (camera_distance <= self.config.integration_radius_m)
        )
        sampled_pixel_count = int(sampled_depth.size)
        valid_pixel_count = int(np.count_nonzero(valid))

        view_cosine = _view_cosine(camera_points, valid)
        if dense.class_count == 1:
            entropy_base = np.ones(dense.entropy.shape, dtype=np.float64)
        else:
            entropy_base = np.clip(
                1.0 - dense.entropy.astype(np.float64) / math.log(dense.class_count),
                0.0,
                1.0,
            )
        quality = np.power(entropy_base, self.config.entropy_power) * np.power(
            np.clip(view_cosine, 0.0, 1.0),
            self.config.view_angle_power,
        )
        if not np.all(np.isfinite(quality)):
            raise ValueError("dense projection quality must be finite")

        camera_flat = camera_points.reshape(-1, 3)
        valid_flat = valid.reshape(-1)
        world = np.zeros_like(camera_flat)
        if np.any(valid_flat):
            world[valid_flat] = (
                (pose[:3, :3] @ camera_flat[valid_flat].T).T + pose[:3, 3]
            )
            if not np.all(np.isfinite(world[valid_flat])):
                raise ValueError("projected world points must be finite")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            quantized = np.floor(world[valid_flat] / self.config.voxel_size_m)
        if np.any(quantized < _INT64_MIN) or np.any(
            quantized >= _INT64_UPPER_EXCLUSIVE
        ):
            raise ValueError("projected voxel keys must fit in signed int64")
        voxel_keys = np.zeros((sampled_pixel_count, 3), dtype=np.int64)
        voxel_keys[valid_flat] = quantized.astype(np.int64)

        class_ids = dense.class_ids.reshape(sampled_pixel_count, -1)
        probabilities = dense.probabilities.astype(np.float64).reshape(
            sampled_pixel_count,
            -1,
        )
        quality_flat = quality.reshape(sampled_pixel_count)
        support = quality_flat[:, None] * probabilities
        candidate_mask = (
            valid_flat[:, None]
            & (quality_flat[:, None] >= self.config.minimum_quality)
            & (class_ids > 0)
            & (probabilities >= self.config.minimum_probability)
            & np.isfinite(support)
            & (support > 0.0)
        )
        updates, support_deltas = _aggregate_updates(
            voxel_keys,
            class_ids,
            support,
            candidate_mask,
        )
        if updates.size == 0:
            return DenseProjectionResult(
                sampled_pixel_count,
                valid_pixel_count,
                0,
            )

        unique_voxels = np.unique(updates[:, :3], axis=0)
        result = DenseProjectionResult(
            sampled_pixel_count,
            valid_pixel_count,
            int(unique_voxels.shape[0]),
        )
        update_lookup = {
            (int(row[0]), int(row[1]), int(row[2]), int(row[3])): float(delta)
            for row, delta in zip(updates, support_deltas)
        }
        for voxel_array in unique_voxels:
            voxel_key = tuple(int(value) for value in voxel_array)
            for candidate in store.semantic_candidates(voxel_key):
                if normalized_revision < candidate.revision:
                    raise ValueError("revision cannot move backwards")
                delta = update_lookup.get((*voxel_key, candidate.label_id))
                if delta is not None and not math.isfinite(candidate.support + delta):
                    raise ValueError("semantic support accumulation must remain finite")

        for row, delta in zip(updates, support_deltas):
            store.update_semantic(
                (int(row[0]), int(row[1]), int(row[2])),
                label_id=int(row[3]),
                support_delta=float(delta),
                revision=normalized_revision,
            )
        return result
