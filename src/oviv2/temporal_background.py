from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real

import numpy as np
import open3d as o3d

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.geometry import SparseTsdfVolume, TsdfConfig
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.temporal_config import TemporalGeometryConfig


_RIGID_ATOL = 1e-6
_TSDF_ATTRIBUTES = ("tsdf", "weight", "color")


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    contiguous = np.array(value, copy=True, order="C")
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _is_bool(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _canonical_rebuild_key(value: object) -> object:
    if _is_bool(value):
        raise TypeError("observation key must contain only integers and tuples")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, tuple) and value:
        return tuple(_canonical_rebuild_key(item) for item in value)
    raise TypeError("observation key must be an integer or non-empty integer tuple")


def _rebuild_sort_key(value: object) -> tuple[object, ...]:
    if isinstance(value, int):
        return (0, value)
    assert isinstance(value, tuple)
    return (1, tuple(_rebuild_sort_key(item) for item in value))


def _validate_config(config: object) -> TemporalGeometryConfig:
    if not isinstance(config, TemporalGeometryConfig):
        raise TypeError("config must be a TemporalGeometryConfig")
    for name in (
        "voxel_size_m",
        "depth_max_m",
        "maximum_icp_rmse_m",
        "maximum_motion_m",
    ):
        value = getattr(config, name)
        if _is_bool(value) or not isinstance(value, Real):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    integer_names = (
        "maximum_entities",
        "maximum_object_voxels",
        "maximum_visibility_points_per_entity",
        "background_block_count",
        "minimum_icp_points",
    )
    normalized: dict[str, int] = {}
    for name in integer_names:
        value = getattr(config, name)
        if _is_bool(value) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        normalized[name] = int(value)
        if normalized[name] <= 0:
            raise ValueError(f"{name} must be positive")
    dilation = config.background_mask_dilation_px
    if _is_bool(dilation) or not isinstance(dilation, Integral):
        raise TypeError("background_mask_dilation_px must be an integer")
    normalized["background_mask_dilation_px"] = int(dilation)
    if normalized["background_mask_dilation_px"] < 0:
        raise ValueError("background_mask_dilation_px must be nonnegative")
    config = replace(config, **normalized)
    fitness = config.minimum_icp_fitness
    if _is_bool(fitness) or not isinstance(fitness, Real):
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
    return config


def _rigid_pose(value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind in "bOUSV":
        raise TypeError("frame.pose must be numeric and non-boolean")
    try:
        pose = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("frame.pose must be numeric") from exc
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("frame.pose must be a finite 4x4 matrix")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=_RIGID_ATOL):
        raise ValueError("frame.pose must have a rigid homogeneous bottom row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=_RIGID_ATOL):
        raise ValueError("frame.pose rotation must be orthogonal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=_RIGID_ATOL):
        raise ValueError("frame.pose rotation must have determinant one")
    return pose


def _finite_real(value: object, name: str, *, positive: bool = False) -> float:
    if _is_bool(value) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_masked_depth(
    masked_depth: object,
    frame_depth: np.ndarray,
    config: TemporalGeometryConfig,
) -> np.ndarray:
    depth = np.asarray(masked_depth)
    if depth.dtype.kind != "f":
        raise TypeError("masked_depth must have a floating dtype")
    if depth.shape != frame_depth.shape:
        raise ValueError("masked_depth shape must match frame.depth")
    if not np.isfinite(depth).all():
        raise ValueError("masked_depth must contain only finite values")
    if np.any(depth < 0.0) or np.any(depth > config.depth_max_m):
        raise ValueError("masked_depth must lie in [0, depth_max_m]")
    return depth


def _validate_frame(
    frame: object,
) -> tuple[Frame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    if _is_bool(frame.frame_id) or not isinstance(frame.frame_id, Integral):
        raise TypeError("frame.frame_id must be an integer")
    if int(frame.frame_id) < 0:
        raise ValueError("frame.frame_id must be nonnegative")
    _finite_real(frame.timestamp, "frame.timestamp")

    depth = np.asarray(frame.depth)
    if depth.dtype.kind != "f":
        raise TypeError("frame.depth must have a floating dtype")
    if depth.ndim != 2:
        raise ValueError("frame.depth must have shape (H, W)")
    height, width = depth.shape

    rgb = np.asarray(frame.rgb)
    if rgb.shape != (height, width, 3):
        raise ValueError("frame.rgb must have shape (H, W, 3) matching depth")
    if rgb.dtype != np.dtype(np.uint8):
        if rgb.dtype.kind != "f":
            raise TypeError("frame.rgb must be uint8 or floating point")
        if not np.isfinite(rgb).all():
            raise ValueError("frame.rgb must contain only finite values")
        if rgb.size and (float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0):
            raise ValueError("floating frame.rgb values must lie in [0, 1]")

    if not isinstance(frame.intrinsics, CameraIntrinsics):
        raise TypeError("frame.intrinsics must be CameraIntrinsics")
    intrinsics = frame.intrinsics
    if _is_bool(intrinsics.width) or not isinstance(intrinsics.width, Integral):
        raise TypeError("intrinsics.width must be an integer")
    if _is_bool(intrinsics.height) or not isinstance(intrinsics.height, Integral):
        raise TypeError("intrinsics.height must be an integer")
    if int(intrinsics.width) <= 0 or int(intrinsics.height) <= 0:
        raise ValueError("intrinsics dimensions must be positive")
    if (int(intrinsics.width), int(intrinsics.height)) != (width, height):
        raise ValueError("intrinsics width and height must match frame depth")
    fx = _finite_real(intrinsics.fx, "intrinsics.fx", positive=True)
    fy = _finite_real(intrinsics.fy, "intrinsics.fy", positive=True)
    cx = _finite_real(intrinsics.cx, "intrinsics.cx")
    cy = _finite_real(intrinsics.cy, "intrinsics.cy")
    matrix = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return frame, depth, rgb, _rigid_pose(frame.pose), matrix


def _validate_observations(
    observations: object, frame: Frame, shape: tuple[int, int]
) -> tuple[FrameObservation, ...]:
    if type(observations) is not tuple:
        raise TypeError("observations must be an exact tuple")
    if any(not isinstance(value, FrameObservation) for value in observations):
        raise TypeError("observations must contain FrameObservation values")
    ids: set[int] = set()
    for observation in observations:
        if _is_bool(observation.observation_id) or not isinstance(
            observation.observation_id, Integral
        ):
            raise TypeError("observation_id must be an integer")
        if int(observation.observation_id) < 0:
            raise ValueError("observation_id must be nonnegative")
        if observation.observation_id in ids:
            raise ValueError("observation IDs must be unique")
        ids.add(observation.observation_id)
        if _is_bool(observation.frame_id) or not isinstance(
            observation.frame_id, Integral
        ):
            raise TypeError("observation frame_id must be an integer")
        if observation.frame_id != frame.frame_id:
            raise ValueError("observation frame_id must match frame")
        observation_timestamp = _finite_real(
            observation.timestamp, "observation timestamp"
        )
        if observation_timestamp != float(frame.timestamp):
            raise ValueError("observation timestamp must match frame")
        mask = np.asarray(observation.mask)
        if mask.dtype.kind != "b":
            raise TypeError("observation mask must have bool dtype")
        if mask.shape != shape:
            raise ValueError("observation mask shape must match frame depth")
    return observations


def _validate_protected_points(
    protected: object, config: TemporalGeometryConfig
) -> tuple[np.ndarray, ...]:
    if type(protected) is not tuple:
        raise TypeError("protected_world_points must be an exact tuple")
    if len(protected) > config.maximum_entities:
        raise ValueError("protected_world_points exceeds maximum_entities")
    validated: list[np.ndarray] = []
    for index, value in enumerate(protected):
        name = f"protected_world_points[{index}]"
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real numeric array") from exc
        if raw.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real numeric values")
        wide = np.asarray(raw, dtype=np.longdouble)
        if not np.isfinite(wide).all():
            raise ValueError(f"{name} must contain only finite values")
        float64_limit = np.longdouble(np.finfo(np.float64).max)
        if np.any(wide > float64_limit) or np.any(wide < -float64_limit):
            raise ValueError(f"{name} values must lie within the float64 range")
        points = np.asarray(raw, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError(f"{name} must have shape (N, 3)")
        if points.shape[0] > config.maximum_visibility_points_per_entity:
            raise ValueError(f"{name} exceeds maximum_visibility_points_per_entity")
        if not np.isfinite(points).all():
            raise ValueError(f"{name} must contain only finite values")
        validated.append(points)
    return tuple(validated)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0 or not mask.any():
        return mask.copy()
    height, width = mask.shape
    radius = min(radius, max(height, width))
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(mask, axis=0), axis=1)
    rows = np.arange(height)
    columns = np.arange(width)
    y0 = np.maximum(rows - radius, 0)
    y1 = np.minimum(rows + radius + 1, height)
    x0 = np.maximum(columns - radius, 0)
    x1 = np.minimum(columns + radius + 1, width)
    window_counts = (
        integral[y1[:, None], x1]
        - integral[y0[:, None], x1]
        - integral[y1[:, None], x0]
        + integral[y0[:, None], x0]
    )
    return window_counts > 0


@dataclass(frozen=True, eq=False)
class BackgroundMaskResult:
    depth_m: np.ndarray
    excluded_pixel_count: int
    valid_background_pixel_count: int

    __hash__ = None

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m)
        if depth.dtype.kind != "f":
            raise TypeError("depth_m must have a floating dtype")
        if depth.ndim != 2:
            raise ValueError("depth_m must be two dimensional")
        if not np.isfinite(depth).all() or np.any(depth < 0.0):
            raise ValueError("depth_m must be finite and nonnegative")
        pixel_count = depth.size
        for name in ("excluded_pixel_count", "valid_background_pixel_count"):
            value = getattr(self, name)
            if _is_bool(value) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= int(value) <= pixel_count:
                raise ValueError(f"{name} must lie within the pixel count")
            object.__setattr__(self, name, int(value))
        if self.valid_background_pixel_count != int(np.count_nonzero(depth)):
            raise ValueError("valid_background_pixel_count must equal nonzero depth count")
        object.__setattr__(self, "depth_m", _readonly_copy(depth))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BackgroundMaskResult):
            return False
        return bool(
            self.excluded_pixel_count == other.excluded_pixel_count
            and self.valid_background_pixel_count == other.valid_background_pixel_count
            and self.depth_m.dtype == other.depth_m.dtype
            and self.depth_m.shape == other.depth_m.shape
            and np.array_equal(self.depth_m, other.depth_m)
        )


def build_background_depth(
    frame: Frame,
    observations: tuple[FrameObservation, ...],
    protected_world_points: tuple[np.ndarray, ...],
    config: TemporalGeometryConfig,
) -> BackgroundMaskResult:
    config = _validate_config(config)
    frame, depth, _, pose, intrinsic = _validate_frame(frame)
    observations = _validate_observations(observations, frame, depth.shape)
    protected = _validate_protected_points(protected_world_points, config)

    excluded = np.zeros(depth.shape, dtype=bool)
    for observation in observations:
        if observation.kind in (ObservationKind.OBJECT, ObservationKind.UNKNOWN):
            excluded |= observation.mask

    if protected:
        world_to_camera = np.linalg.inv(pose)
        for points in protected:
            if points.shape[0] == 0:
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                camera = (
                    points @ world_to_camera[:3, :3].T
                    + world_to_camera[:3, 3]
                )
            z = camera[:, 2]
            front = np.isfinite(camera).all(axis=1) & (z > 0.0)
            if not front.any():
                continue
            camera = camera[front]
            z = camera[:, 2]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                u = intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]
                v = intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]
            in_bounds = (
                np.isfinite(u)
                & np.isfinite(v)
                & (u >= -0.5)
                & (u < depth.shape[1] - 0.5)
                & (v >= -0.5)
                & (v < depth.shape[0] - 0.5)
            )
            if not in_bounds.any():
                continue
            u = u[in_bounds]
            v = v[in_bounds]
            x = np.floor(u + 0.5).astype(np.int64)
            y = np.floor(v + 0.5).astype(np.int64)
            rounded_in_bounds = (
                (x >= 0)
                & (x < depth.shape[1])
                & (y >= 0)
                & (y < depth.shape[0])
            )
            excluded[y[rounded_in_bounds], x[rounded_in_bounds]] = True

    excluded = _dilate(excluded, config.background_mask_dilation_px)
    clean = np.array(depth, copy=True, order="C")
    valid = np.isfinite(clean) & (clean > 0.0) & (clean <= config.depth_max_m)
    clean[~valid] = 0.0
    clean[excluded] = 0.0
    return BackgroundMaskResult(
        clean,
        int(np.count_nonzero(excluded)),
        int(np.count_nonzero(clean)),
    )


def _new_sparse_volume(
    config: TemporalGeometryConfig, physical_capacity: int
) -> SparseTsdfVolume:
    return SparseTsdfVolume(
        TsdfConfig(
            voxel_size_m=float(config.voxel_size_m),
            block_resolution=8,
            block_count=max(1, int(physical_capacity)),
            depth_max_m=float(config.depth_max_m),
            trunc_voxel_multiplier=4.0,
        )
    )


def _active_block_keys(volume: SparseTsdfVolume) -> np.ndarray:
    grid = volume._grid
    active = grid.hashmap().active_buf_indices()
    if int(active.shape[0]) == 0:
        return np.empty((0, 3), dtype=np.int32)
    return np.array(grid.hashmap().key_tensor()[active].numpy(), copy=True)


def _candidate_block_keys(
    volume: SparseTsdfVolume,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    config: TemporalGeometryConfig,
) -> np.ndarray:
    if np.count_nonzero(depth) == 0:
        return np.empty((0, 3), dtype=np.int32)
    clean_depth = np.array(depth, dtype=np.float32, copy=True, order="C")
    valid = (
        np.isfinite(clean_depth)
        & (clean_depth > 0.0)
        & (clean_depth <= config.depth_max_m)
    )
    clean_depth[~valid] = 0.0
    world_to_camera = np.linalg.inv(camera_to_world)
    coordinates = volume._grid.compute_unique_block_coordinates(
        o3d.t.geometry.Image(o3d.core.Tensor(clean_depth)),
        o3d.core.Tensor(intrinsic, dtype=o3d.core.float64),
        o3d.core.Tensor(world_to_camera, dtype=o3d.core.float64),
        depth_scale=1.0,
        depth_max=float(config.depth_max_m),
        trunc_voxel_multiplier=4.0,
    )
    return np.array(coordinates.numpy(), copy=True).reshape((-1, 3))


def _block_key_union_count(existing: np.ndarray, candidate: np.ndarray) -> int:
    if existing.shape[0] == 0:
        return int(candidate.shape[0])
    if candidate.shape[0] == 0:
        return int(existing.shape[0])
    return int(np.unique(np.concatenate((existing, candidate), axis=0), axis=0).shape[0])


class TemporalBackgroundVolume:
    __hash__ = None

    def __init__(self, config: TemporalGeometryConfig) -> None:
        self._config = _validate_config(config)
        self._volume = _new_sparse_volume(config, 1)
        self._last_blocks_touched = 0

    @property
    def config(self) -> TemporalGeometryConfig:
        return self._config

    @property
    def active_block_count(self) -> int:
        return self._volume.active_block_count

    @property
    def last_blocks_touched(self) -> int:
        return self._last_blocks_touched

    def canonical_block_state(self) -> tuple[object, ...]:
        grid = self._volume._grid
        active = grid.hashmap().active_buf_indices()
        if int(active.shape[0]) == 0:
            return ()
        keys = grid.hashmap().key_tensor()[active].numpy()
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        arrays = [keys[order]]
        arrays.extend(grid.attribute(name)[active].numpy()[order] for name in _TSDF_ATTRIBUTES)
        return tuple((value.dtype.str, value.shape, value.tobytes()) for value in arrays)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TemporalBackgroundVolume):
            return False
        return bool(
            self.config == other.config
            and self.canonical_block_state() == other.canonical_block_state()
        )

    def __deepcopy__(self, memo: dict[int, object]) -> TemporalBackgroundVolume:
        del memo
        return self.clone()

    def clone(self) -> TemporalBackgroundVolume:
        """Return an independent exact snapshot without exposing mutable TSDF state."""
        snapshot = self._clone(max(1, self.active_block_count))
        snapshot._last_blocks_touched = self.last_blocks_touched
        return snapshot

    @classmethod
    def rebuild(
        cls,
        config: TemporalGeometryConfig,
        observations: tuple[tuple[object, Frame, np.ndarray], ...],
    ) -> TemporalBackgroundVolume:
        """Build a fresh volume by integrating observations in canonical key order."""
        _validate_config(config)
        if not isinstance(observations, tuple):
            raise TypeError("observations must be a tuple")
        canonical: list[tuple[object, Frame, np.ndarray]] = []
        keys: set[object] = set()
        for item in observations:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError("each observation must be a (key, frame, depth) tuple")
            raw_key, frame, depth = item
            key = _canonical_rebuild_key(raw_key)
            if key in keys:
                raise ValueError("observation keys must be unique")
            keys.add(key)
            canonical.append((key, frame, depth))
        rebuilt = cls(config)
        for _, frame, depth in sorted(canonical, key=lambda item: _rebuild_sort_key(item[0])):
            rebuilt = rebuilt.trial_integrate(frame, depth)
        return rebuilt

    @classmethod
    def rebuild_blocks(
        cls,
        config: TemporalGeometryConfig,
        observations: tuple[
            tuple[object, tuple[int, int, int], Frame, np.ndarray], ...
        ],
    ) -> TemporalBackgroundVolume:
        """Rebuild from stable records while integrating only their owned blocks."""
        _validate_config(config)
        if not isinstance(observations, tuple):
            raise TypeError("observations must be a tuple")
        canonical: list[
            tuple[object, tuple[int, int, int], Frame, np.ndarray]
        ] = []
        keys: set[object] = set()
        for item in observations:
            if not isinstance(item, tuple) or len(item) != 4:
                raise TypeError(
                    "each block observation must be a (key, block_key, frame, depth) tuple"
                )
            raw_key, raw_block_key, frame, depth = item
            key = _canonical_rebuild_key(raw_key)
            if key in keys:
                raise ValueError("observation keys must be unique")
            if (
                not isinstance(raw_block_key, tuple)
                or len(raw_block_key) != 3
                or any(
                    _is_bool(value) or not isinstance(value, Integral)
                    for value in raw_block_key
                )
            ):
                raise TypeError("block_key must be a canonical three-integer tuple")
            keys.add(key)
            block_key = tuple(int(value) for value in raw_block_key)
            canonical.append((key, block_key, frame, depth))
        rebuilt = cls(config)
        for _, block_key, frame, depth in sorted(
            canonical, key=lambda item: _rebuild_sort_key(item[0])
        ):
            rebuilt = rebuilt.trial_integrate_blocks(frame, depth, (block_key,))
        return rebuilt

    def candidate_block_keys(
        self, frame: Frame, masked_depth: np.ndarray
    ) -> tuple[tuple[int, int, int], ...]:
        """Return the canonical TSDF blocks touched by one validated observation."""
        _, frame_depth, _, pose, intrinsic = _validate_frame(frame)
        depth = _validate_masked_depth(masked_depth, frame_depth, self.config)
        keys = _candidate_block_keys(
            self._volume, depth, intrinsic, pose, self.config
        )
        return tuple(
            sorted(tuple(int(value) for value in row) for row in keys.tolist())
        )

    def trial_integrate_blocks(
        self,
        frame: Frame,
        masked_depth: np.ndarray,
        block_keys: tuple[tuple[int, int, int], ...],
    ) -> TemporalBackgroundVolume:
        """Integrate one observation into an explicit subset of its touched blocks."""
        _, frame_depth, rgb, pose, intrinsic = _validate_frame(frame)
        depth = _validate_masked_depth(masked_depth, frame_depth, self.config)
        if not isinstance(block_keys, tuple) or not block_keys:
            raise TypeError("block_keys must be a non-empty tuple")
        if any(
            not isinstance(key, tuple)
            or len(key) != 3
            or any(
                _is_bool(value) or not isinstance(value, Integral) for value in key
            )
            for key in block_keys
        ):
            raise TypeError("block_keys must contain canonical three-integer tuples")
        normalized = tuple(tuple(int(value) for value in key) for key in block_keys)
        selected = tuple(sorted(set(normalized)))
        if selected != normalized:
            raise ValueError("block_keys must be sorted and unique")
        candidate = self.candidate_block_keys(frame, depth)
        if not set(selected).issubset(candidate):
            raise ValueError("block_keys must be touched by the observation")
        existing = {tuple(int(value) for value in row) for row in _active_block_keys(self._volume)}
        if len(existing | set(selected)) > self.config.background_block_count:
            raise ValueError("TSDF block capacity would exceed background_block_count")
        trial = self._clone(max(1, len(existing | set(selected))))
        trial._integrate_owned_blocks(depth, rgb, intrinsic, pose, selected)
        return trial

    def _clone(self, physical_capacity: int) -> TemporalBackgroundVolume:
        trial = self.__class__.__new__(self.__class__)
        trial._config = self.config
        trial._volume = _new_sparse_volume(self.config, physical_capacity)
        trial._last_blocks_touched = 0
        source_grid = self._volume._grid
        source_indices = source_grid.hashmap().active_buf_indices()
        source_count = int(source_indices.shape[0])
        if source_count > self.config.background_block_count:
            raise RuntimeError("active TSDF blocks exceed background_block_count")
        if source_count:
            source_keys = source_grid.hashmap().key_tensor()[source_indices]
            destination_grid = trial._volume._grid
            destination_indices, activated = destination_grid.hashmap().activate(source_keys)
            if not np.all(activated.numpy()):
                raise RuntimeError("failed to clone active TSDF blocks")
            for name in _TSDF_ATTRIBUTES:
                destination_grid.attribute(name)[destination_indices] = source_grid.attribute(
                    name
                )[source_indices]
        return trial

    def trial_integrate(
        self, frame: Frame, masked_depth: np.ndarray
    ) -> TemporalBackgroundVolume:
        return self._owned_trial_integrate(frame, masked_depth)

    def _owned_trial_integrate(
        self, frame: Frame, masked_depth: np.ndarray
    ) -> TemporalBackgroundVolume:
        frame, frame_depth, rgb, pose, intrinsic = _validate_frame(frame)
        depth = _validate_masked_depth(masked_depth, frame_depth, self.config)

        candidate_keys = _candidate_block_keys(
            self._volume, depth, intrinsic, pose, self.config
        )
        existing_keys = _active_block_keys(self._volume)
        union_count = _block_key_union_count(existing_keys, candidate_keys)
        if union_count > self.config.background_block_count:
            raise ValueError(
                "TSDF block capacity would exceed background_block_count"
            )

        trial = self._clone(max(1, union_count))
        if candidate_keys.shape[0] == 0:
            return trial
        trial._integrate_owned(depth, rgb, intrinsic, pose)
        return trial

    def _integrate_owned(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        intrinsic: np.ndarray,
        pose: np.ndarray,
    ) -> None:
        self._last_blocks_touched = self._volume.integrate(
            depth,
            rgb,
            intrinsic,
            pose,
        )
        if self.active_block_count > self.config.background_block_count:
            raise RuntimeError("TSDF integration exceeded background_block_count")

    def _integrate_owned_blocks(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        intrinsic: np.ndarray,
        pose: np.ndarray,
        block_keys: tuple[tuple[int, int, int], ...],
    ) -> None:
        clean_depth = np.ascontiguousarray(depth, dtype=np.float32)
        clean_color = np.ascontiguousarray(rgb, dtype=np.float32)
        if rgb.dtype == np.uint8:
            clean_color /= 255.0
        depth_image = o3d.t.geometry.Image(o3d.core.Tensor(clean_depth))
        color_image = o3d.t.geometry.Image(o3d.core.Tensor(clean_color))
        intrinsic_tensor = o3d.core.Tensor(intrinsic, dtype=o3d.core.float64)
        extrinsic_tensor = o3d.core.Tensor(
            np.linalg.inv(pose), dtype=o3d.core.float64
        )
        block_coords = o3d.core.Tensor(
            np.asarray(block_keys, dtype=np.int32), dtype=o3d.core.int32
        )
        self._volume._grid.integrate(
            block_coords,
            depth_image,
            color_image,
            intrinsic_tensor,
            extrinsic_tensor,
            depth_scale=1.0,
            depth_max=self._volume.config.depth_max_m,
            trunc_voxel_multiplier=self._volume.config.trunc_voxel_multiplier,
        )
        self._last_blocks_touched = len(block_keys)
        if self.active_block_count > self.config.background_block_count:
            raise RuntimeError("TSDF integration exceeded background_block_count")
