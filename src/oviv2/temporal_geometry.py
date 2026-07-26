from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real

import numpy as np
import open3d as o3d

from src.core.data_structures import Frame
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.temporal_config import TemporalGeometryConfig


_RIGID_ATOL = 1e-6


def _contains_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            return True
        if value.dtype.kind == "O":
            return any(_contains_bool(item) for item in value.flat)
        return False
    if isinstance(value, (tuple, list)):
        return any(_contains_bool(item) for item in value)
    return False


def _readonly_array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.array(value, dtype=dtype, copy=True, order="C")
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _points(value: object, name: str) -> np.ndarray:
    if _contains_bool(value):
        raise TypeError(f"{name} must contain numeric non-boolean values")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a float64 array") from exc
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must contain numeric non-boolean values")
    try:
        points = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a float64 array") from exc
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points


def _finite_xyz(value: object, name: str) -> tuple[float, float, float]:
    if _contains_bool(value):
        raise TypeError(f"{name} must contain numeric non-boolean values")
    try:
        xyz = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if xyz.shape != (3,):
        raise ValueError(f"{name} must have three elements")
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"{name} must contain only finite values")
    return tuple(float(component) for component in xyz)


def _rigid_transform(value: object, name: str) -> np.ndarray:
    if _contains_bool(value):
        raise TypeError(f"{name} must contain numeric non-boolean values")
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=_RIGID_ATOL):
        raise ValueError(f"{name} must have a rigid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=_RIGID_ATOL):
        raise ValueError(f"{name} rotation must be orthogonal")
    determinant = float(np.linalg.det(rotation))
    if not math.isfinite(determinant) or determinant <= 0.0 or not math.isclose(
        determinant, 1.0, rel_tol=0.0, abs_tol=_RIGID_ATOL
    ):
        raise ValueError(f"{name} rotation must have determinant one")
    return transform


def _validate_config(config: TemporalGeometryConfig) -> None:
    if not isinstance(config, TemporalGeometryConfig):
        raise TypeError("config must be a TemporalGeometryConfig")
    for name in ("voxel_size_m", "depth_max_m", "maximum_icp_rmse_m", "maximum_motion_m"):
        value = getattr(config, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    positive_integer_fields = (
        "maximum_entities",
        "maximum_object_voxels",
        "maximum_visibility_points_per_entity",
        "background_block_count",
        "minimum_icp_points",
    )
    for name in positive_integer_fields:
        value = getattr(config, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    dilation = config.background_mask_dilation_px
    if isinstance(dilation, (bool, np.bool_)) or not isinstance(dilation, int):
        raise TypeError("background_mask_dilation_px must be an integer")
    if dilation < 0:
        raise ValueError("background_mask_dilation_px must be nonnegative")
    fitness = config.minimum_icp_fitness
    if isinstance(fitness, (bool, np.bool_)) or not isinstance(fitness, Real):
        raise TypeError("minimum_icp_fitness must be numeric")
    if not math.isfinite(float(fitness)) or not 0.0 <= float(fitness) <= 1.0:
        raise ValueError("minimum_icp_fitness must be finite and in [0, 1]")
    if float(config.voxel_size_m) > float(config.depth_max_m):
        raise ValueError("voxel_size_m cannot exceed depth_max_m")
    for name in (
        "maximum_entities",
        "maximum_object_voxels",
        "maximum_visibility_points_per_entity",
    ):
        if getattr(config, name) > config.background_block_count:
            raise ValueError(f"{name} cannot exceed background_block_count")


def _translation_pose(xyz: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = xyz
    return pose


@dataclass(frozen=True, eq=False)
class ObjectSubmap:
    reference_centroid_xyz: tuple[float, float, float]
    local_voxel_keys: tuple[tuple[int, int, int], ...]
    local_points_xyz: np.ndarray
    weights: np.ndarray
    last_seen_frame_ids: np.ndarray

    __hash__ = None

    def __post_init__(self) -> None:
        reference = _finite_xyz(self.reference_centroid_xyz, "reference_centroid_xyz")
        if type(self.local_voxel_keys) is not tuple:
            raise TypeError("local_voxel_keys must be an exact tuple")
        normalized_keys: list[tuple[int, int, int]] = []
        for key in self.local_voxel_keys:
            if type(key) is not tuple or len(key) != 3:
                raise TypeError("each local voxel key must be an exact three-element tuple")
            if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in key):
                raise TypeError("voxel key components must be non-boolean integers")
            normalized_keys.append(tuple(int(value) for value in key))
        keys = tuple(normalized_keys)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("local_voxel_keys must be sorted and unique")

        points = _points(self.local_points_xyz, "local_points_xyz")
        if _contains_bool(self.weights):
            raise TypeError("weights must contain numeric non-boolean values")
        try:
            weights = np.asarray(self.weights, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("weights must be numeric") from exc
        if weights.ndim != 1:
            raise ValueError("weights must have shape (N,)")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("weights must be finite and positive")

        raw_seen = np.asarray(self.last_seen_frame_ids)
        if raw_seen.ndim != 1:
            raise ValueError("last_seen_frame_ids must have shape (N,)")
        if raw_seen.dtype.kind not in "iu" or raw_seen.dtype.kind == "b":
            raise TypeError("last_seen_frame_ids must contain integers")
        if raw_seen.size and (np.any(raw_seen < 0) or np.any(raw_seen > np.iinfo(np.int64).max)):
            raise ValueError("last_seen_frame_ids must be nonnegative int64 values")
        seen = np.asarray(raw_seen, dtype=np.int64)
        count = len(keys)
        if points.shape[0] != count or weights.shape[0] != count or seen.shape[0] != count:
            raise ValueError("submap arrays and voxel keys must have matching lengths")

        object.__setattr__(self, "reference_centroid_xyz", reference)
        object.__setattr__(self, "local_voxel_keys", keys)
        object.__setattr__(self, "local_points_xyz", _readonly_array(points, np.dtype(np.float64)))
        object.__setattr__(self, "weights", _readonly_array(weights, np.dtype(np.float64)))
        object.__setattr__(self, "last_seen_frame_ids", _readonly_array(seen, np.dtype(np.int64)))

    def __eq__(self, other: object) -> bool:
        if type(other) is not ObjectSubmap:
            return False
        assert isinstance(other, ObjectSubmap)
        return bool(
            self.reference_centroid_xyz == other.reference_centroid_xyz
            and self.local_voxel_keys == other.local_voxel_keys
            and np.array_equal(self.local_points_xyz, other.local_points_xyz)
            and np.array_equal(self.weights, other.weights)
            and np.array_equal(self.last_seen_frame_ids, other.last_seen_frame_ids)
        )

    def world_points(self, object_to_world: np.ndarray | None = None) -> np.ndarray:
        transform = (
            _translation_pose(self.reference_centroid_xyz)
            if object_to_world is None
            else _rigid_transform(object_to_world, "object_to_world")
        )
        world = (transform[:3, :3] @ self.local_points_xyz.T).T + transform[:3, 3]
        if not np.all(np.isfinite(world)):
            raise ValueError("world point transformation produced non-finite values")
        return _readonly_array(world.reshape((-1, 3)), np.dtype(np.float64))


class MotionDecision(str, Enum):
    ICP_ACCEPTED = "icp_accepted"
    TRANSLATION_ACCEPTED = "translation_accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, eq=False, init=False)
class ObjectMotionEstimate:
    object_to_world: np.ndarray
    decision: MotionDecision
    fitness: float
    rmse_m: float

    __hash__ = None

    def __init__(
        self,
        object_to_world: np.ndarray,
        decision: MotionDecision | bool | None = None,
        fitness: float = 0.0,
        rmse_m: float = 0.0,
        *,
        used_icp: bool | None = None,
    ) -> None:
        if decision is not None and used_icp is not None:
            raise TypeError("provide decision or used_icp, not both")
        boundary_value = used_icp if decision is None else decision
        if type(boundary_value) is bool:
            normalized_decision = (
                MotionDecision.ICP_ACCEPTED
                if boundary_value
                else MotionDecision.TRANSLATION_ACCEPTED
            )
        elif type(boundary_value) is MotionDecision:
            normalized_decision = boundary_value
        else:
            raise TypeError("decision must be a MotionDecision")

        transform = _rigid_transform(object_to_world, "object_to_world")
        for name in ("fitness", "rmse_m"):
            value = fitness if name == "fitness" else rmse_m
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(fitness) <= 1.0:
            raise ValueError("fitness must lie in [0, 1]")
        if float(rmse_m) < 0.0:
            raise ValueError("rmse_m must be nonnegative")
        object.__setattr__(self, "object_to_world", _readonly_array(transform, np.dtype(np.float64)))
        object.__setattr__(self, "decision", normalized_decision)
        object.__setattr__(self, "fitness", float(fitness))
        object.__setattr__(self, "rmse_m", float(rmse_m))

    @property
    def used_icp(self) -> bool:
        return self.decision is MotionDecision.ICP_ACCEPTED

    def __eq__(self, other: object) -> bool:
        if type(other) is not ObjectMotionEstimate:
            return False
        assert isinstance(other, ObjectMotionEstimate)
        return bool(
            self.decision is other.decision
            and self.fitness == other.fitness
            and self.rmse_m == other.rmse_m
            and np.array_equal(self.object_to_world, other.object_to_world)
        )


def backproject_observation(
    frame: Frame,
    observation: FrameObservation,
    config: TemporalGeometryConfig,
) -> np.ndarray:
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    if not isinstance(observation, FrameObservation):
        raise TypeError("observation must be a FrameObservation")
    _validate_config(config)
    if observation.kind not in (ObservationKind.OBJECT, ObservationKind.UNKNOWN):
        raise ValueError("observation kind must be OBJECT or UNKNOWN")
    if isinstance(frame.frame_id, (bool, np.bool_)) or not isinstance(frame.frame_id, Integral) or frame.frame_id < 0:
        raise ValueError("frame.frame_id must be a nonnegative integer")
    if isinstance(observation.frame_id, (bool, np.bool_)) or observation.frame_id != frame.frame_id:
        raise ValueError("observation.frame_id must match frame.frame_id")
    if _contains_bool(frame.timestamp) or _contains_bool(observation.timestamp):
        raise TypeError("frame and observation timestamps must be numeric non-boolean values")
    try:
        frame_timestamp = float(frame.timestamp)
        observation_timestamp = float(observation.timestamp)
    except (TypeError, ValueError) as exc:
        raise TypeError("frame and observation timestamps must be numeric") from exc
    if not math.isfinite(frame_timestamp) or not math.isfinite(observation_timestamp):
        raise ValueError("frame and observation timestamps must be finite")
    if observation_timestamp != frame_timestamp:
        raise ValueError("observation.timestamp must match frame.timestamp")

    depth = np.asarray(frame.depth)
    rgb = np.asarray(frame.rgb)
    mask = np.asarray(observation.mask)
    if _contains_bool(frame.depth):
        raise TypeError("frame depth must contain numeric non-boolean values")
    if depth.ndim != 2:
        raise ValueError("frame depth must have shape (H, W)")
    if rgb.shape != (*depth.shape, 3):
        raise ValueError("frame RGB must have shape (H, W, 3) matching depth")
    if mask.shape != depth.shape or mask.dtype.kind != "b":
        raise ValueError("observation mask shape must match frame depth")
    intrinsics = frame.intrinsics
    for name in ("width", "height"):
        value = getattr(intrinsics, name, None)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
            raise ValueError("frame intrinsics dimensions must be positive integers")
    if (int(intrinsics.height), int(intrinsics.width)) != depth.shape:
        raise ValueError("frame intrinsics dimensions must match depth")
    if any(
        _contains_bool(value)
        for value in (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    ):
        raise TypeError("frame intrinsics must contain numeric non-boolean values")
    try:
        intrinsic_values = np.asarray(
            [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy], dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("frame intrinsics must be numeric") from exc
    if not np.all(np.isfinite(intrinsic_values)) or intrinsic_values[0] <= 0.0 or intrinsic_values[1] <= 0.0:
        raise ValueError("frame intrinsics must be finite with positive focal lengths")
    pose = _rigid_transform(frame.pose, "frame pose")
    try:
        valid = mask & np.isfinite(depth) & (depth > 0.0) & (depth <= float(config.depth_max_m))
    except TypeError as exc:
        raise TypeError("frame depth must be numeric") from exc
    rows, columns = np.nonzero(valid)
    if rows.size == 0:
        return _readonly_array(np.empty((0, 3), dtype=np.float64), np.dtype(np.float64))
    z = np.asarray(depth[rows, columns], dtype=np.float64)
    camera = np.column_stack(
        ((columns - intrinsic_values[2]) * z / intrinsic_values[0],
         (rows - intrinsic_values[3]) * z / intrinsic_values[1], z)
    )
    world = (pose[:3, :3] @ camera.T).T + pose[:3, 3]
    if not np.all(np.isfinite(world)):
        raise ValueError("backprojection produced non-finite points")
    return _readonly_array(world, np.dtype(np.float64))


def integrate_object_submap(
    submap: ObjectSubmap,
    points_world: np.ndarray,
    frame_id: int,
    config: TemporalGeometryConfig,
    *,
    object_to_world: np.ndarray | None = None,
) -> ObjectSubmap:
    if not isinstance(submap, ObjectSubmap):
        raise TypeError("submap must be an ObjectSubmap")
    _validate_config(config)
    points = _points(points_world, "points_world")
    if isinstance(frame_id, (bool, np.bool_)) or not isinstance(frame_id, Integral):
        raise TypeError("frame_id must be an integer")
    normalized_frame_id = int(frame_id)
    if normalized_frame_id < 0 or normalized_frame_id > np.iinfo(np.int64).max:
        raise ValueError("frame_id must be a nonnegative int64 value")
    if submap.last_seen_frame_ids.size and normalized_frame_id <= int(submap.last_seen_frame_ids.max()):
        raise ValueError("frame_id must increase strictly")
    current_pose = (
        _translation_pose(submap.reference_centroid_xyz)
        if object_to_world is None
        else _rigid_transform(object_to_world, "object_to_world")
    )
    if points.shape[0] == 0:
        return submap

    with np.errstate(over="ignore", invalid="ignore"):
        local = (points - current_pose[:3, 3]) @ current_pose[:3, :3]
    if not np.all(np.isfinite(local)):
        raise ValueError("local coordinate conversion produced non-finite values")
    grouped: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
    voxel_size = float(config.voxel_size_m)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = local / voxel_size
    if not np.all(np.isfinite(scaled)):
        raise ValueError("voxel coordinate conversion produced non-finite values")
    for point, voxel_point in zip(local, scaled):
        key = tuple(math.floor(float(component)) for component in voxel_point)
        grouped.setdefault(key, []).append(tuple(float(component) for component in point))

    records = {
        key: (tuple(float(value) for value in point), float(weight), int(seen))
        for key, point, weight, seen in zip(
            submap.local_voxel_keys,
            submap.local_points_xyz,
            submap.weights,
            submap.last_seen_frame_ids,
        )
    }
    for key in sorted(grouped):
        samples = sorted(grouped[key])
        old_point, old_weight, old_seen = records.get(key, ((0.0, 0.0, 0.0), 0.0, normalized_frame_id))
        new_weight = old_weight + float(len(samples))
        coordinates = tuple(
            math.fsum([old_point[axis] * old_weight] + [sample[axis] for sample in samples]) / new_weight
            for axis in range(3)
        )
        if not math.isfinite(new_weight) or not all(math.isfinite(value) for value in coordinates):
            raise ValueError("voxel aggregation produced non-finite values")
        records[key] = (coordinates, new_weight, max(old_seen, normalized_frame_id))

    overflow = len(records) - int(config.maximum_object_voxels)
    if overflow > 0:
        eviction_order = sorted(
            records,
            key=lambda key: (records[key][1], records[key][2], key),
        )
        for key in eviction_order[:overflow]:
            del records[key]
    keys = tuple(sorted(records))
    return ObjectSubmap(
        reference_centroid_xyz=submap.reference_centroid_xyz,
        local_voxel_keys=keys,
        local_points_xyz=np.asarray([records[key][0] for key in keys], dtype=np.float64).reshape((-1, 3)),
        weights=np.asarray([records[key][1] for key in keys], dtype=np.float64),
        last_seen_frame_ids=np.asarray([records[key][2] for key in keys], dtype=np.int64),
    )


def _run_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_transform: np.ndarray,
    maximum_correspondence_distance_m: float,
) -> tuple[np.ndarray, float, float]:
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(
        np.array(source_points, dtype=np.float64, copy=True, order="C")
    )
    target.points = o3d.utility.Vector3dVector(
        np.array(target_points, dtype=np.float64, copy=True, order="C")
    )
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        maximum_correspondence_distance_m,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return np.asarray(result.transformation, dtype=np.float64), float(result.fitness), float(result.inlier_rmse)


def _motion_result(
    transform: np.ndarray,
    decision: MotionDecision,
    fitness: float,
    rmse_m: float,
) -> ObjectMotionEstimate:
    return ObjectMotionEstimate(transform, decision, fitness, rmse_m)


def _bounded_translation_motion(
    submap: ObjectSubmap,
    target: np.ndarray,
    observed_centroid: tuple[float, float, float],
    config: TemporalGeometryConfig,
    previous_pose: np.ndarray,
) -> ObjectMotionEstimate:
    diagnostic_rmse = float(config.maximum_icp_rmse_m)
    if submap.local_points_xyz.shape[0] == 0 or target.shape[0] == 0:
        return _motion_result(
            previous_pose, MotionDecision.REJECTED, 0.0, diagnostic_rmse
        )
    with np.errstate(over="ignore", invalid="ignore"):
        displacement = float(
            np.linalg.norm(
                np.asarray(observed_centroid, dtype=np.float64)
                - previous_pose[:3, 3]
            )
        )
    pose = np.array(previous_pose, dtype=np.float64, copy=True, order="C")
    if math.isfinite(displacement) and displacement <= float(config.maximum_motion_m):
        pose[:3, 3] = observed_centroid
        return _motion_result(
            pose, MotionDecision.TRANSLATION_ACCEPTED, 0.0, diagnostic_rmse
        )
    return _motion_result(
        previous_pose, MotionDecision.REJECTED, 0.0, diagnostic_rmse
    )


def estimate_object_translation(
    submap: ObjectSubmap,
    points_world: np.ndarray,
    observed_centroid_xyz: tuple[float, float, float],
    config: TemporalGeometryConfig,
    *,
    previous_object_to_world: np.ndarray | None = None,
) -> ObjectMotionEstimate:
    if not isinstance(submap, ObjectSubmap):
        raise TypeError("submap must be an ObjectSubmap")
    _validate_config(config)
    previous_pose = (
        _translation_pose(submap.reference_centroid_xyz)
        if previous_object_to_world is None
        else _rigid_transform(previous_object_to_world, "previous_object_to_world")
    )
    try:
        target = _points(points_world, "points_world")
        observed_centroid = _finite_xyz(observed_centroid_xyz, "observed_centroid_xyz")
    except ValueError:
        return _motion_result(
            previous_pose,
            MotionDecision.REJECTED,
            0.0,
            float(config.maximum_icp_rmse_m),
        )
    return _bounded_translation_motion(
        submap, target, observed_centroid, config, previous_pose
    )


def estimate_object_motion(
    submap: ObjectSubmap,
    points_world: np.ndarray,
    observed_centroid_xyz: tuple[float, float, float],
    config: TemporalGeometryConfig,
    *,
    previous_object_to_world: np.ndarray | None = None,
) -> ObjectMotionEstimate:
    if not isinstance(submap, ObjectSubmap):
        raise TypeError("submap must be an ObjectSubmap")
    _validate_config(config)
    previous_pose = (
        _translation_pose(submap.reference_centroid_xyz)
        if previous_object_to_world is None
        else _rigid_transform(previous_object_to_world, "previous_object_to_world")
    )
    try:
        target = _points(points_world, "points_world")
        observed_centroid = _finite_xyz(observed_centroid_xyz, "observed_centroid_xyz")
    except ValueError:
        return _motion_result(
            previous_pose,
            MotionDecision.REJECTED,
            0.0,
            float(config.maximum_icp_rmse_m),
        )
    fallback = _bounded_translation_motion(
        submap, target, observed_centroid, config, previous_pose
    )
    fallback_pose = fallback.object_to_world
    diagnostic_rmse = float(config.maximum_icp_rmse_m)
    if submap.local_points_xyz.shape[0] == 0 or target.shape[0] == 0:
        return fallback
    minimum_points = int(config.minimum_icp_points)
    source = submap.local_points_xyz
    if source.shape[0] < minimum_points or target.shape[0] < minimum_points:
        return fallback
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2 or np.linalg.matrix_rank(target - target.mean(axis=0)) < 2:
        return fallback

    try:
        icp_result = _run_icp(
            source,
            target,
            fallback_pose,
            max(2.0 * float(config.voxel_size_m), float(config.maximum_icp_rmse_m)),
        )
    except (RuntimeError, FloatingPointError):
        return fallback
    try:
        transform, fitness, rmse = icp_result
        transform = _rigid_transform(transform, "ICP transform")
        if isinstance(fitness, (bool, np.bool_)) or isinstance(rmse, (bool, np.bool_)):
            raise TypeError("ICP metrics must be numeric non-boolean values")
        fitness = float(fitness)
        rmse = float(rmse)
        icp_displacement = float(
            np.linalg.norm(transform[:3, 3] - previous_pose[:3, 3])
        )
        accepted = (
            math.isfinite(fitness)
            and 0.0 <= fitness <= 1.0
            and math.isfinite(rmse)
            and rmse >= 0.0
            and fitness >= float(config.minimum_icp_fitness)
            and rmse <= float(config.maximum_icp_rmse_m)
            and math.isfinite(icp_displacement)
            and icp_displacement <= float(config.maximum_motion_m)
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return fallback
    if accepted:
        return _motion_result(transform, MotionDecision.ICP_ACCEPTED, fitness, rmse)
    safe_fitness = fitness if math.isfinite(fitness) and 0.0 <= fitness <= 1.0 else 0.0
    safe_rmse = rmse if math.isfinite(rmse) and rmse >= 0.0 else diagnostic_rmse
    return _motion_result(fallback_pose, fallback.decision, safe_fitness, safe_rmse)
