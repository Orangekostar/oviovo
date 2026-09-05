"""Source-bound surface attributes for frozen OVI-MAP instance meshes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.data_structures import CameraIntrinsics
from src.evaluation.baselines.ovimap import _color_codes


class SurfaceAttributeError(ValueError):
    """Raised when surface attributes lack an exact, legal source binding."""


def _immutable(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype)
    if result.base is not None or result.flags.writeable:
        result = result.copy()
    result.setflags(write=False)
    return result


def _matrix(values: object, *, name: str, columns: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 2 or result.shape[1] != columns:
        raise SurfaceAttributeError(f"{name} must have shape (N, {columns})")
    return result


@dataclass(frozen=True, slots=True)
class SurfaceGroup:
    """PLY rows aligned to one OVI instance-palette color."""

    points_xyz: np.ndarray
    normals_xyz: np.ndarray
    normal_valid: np.ndarray
    original_vertex_indices: np.ndarray
    palette_rgb: tuple[int, int, int]
    rgb_source: str = "instance_palette"

    def __post_init__(self) -> None:
        points = _matrix(self.points_xyz, name="surface points", columns=3)
        normals = _matrix(self.normals_xyz, name="surface normals", columns=3)
        valid = np.asarray(self.normal_valid)
        indices = np.asarray(self.original_vertex_indices)
        if normals.shape != points.shape:
            raise SurfaceAttributeError("surface normals must align with points")
        if valid.shape != (len(points),) or valid.dtype != np.bool_:
            raise SurfaceAttributeError("normal_valid must be a boolean row mask")
        if indices.shape != (len(points),) or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise SurfaceAttributeError("original vertex indices must align with points")
        if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
            raise SurfaceAttributeError("original vertex indices must be unique and nonnegative")
        color = tuple(int(value) for value in self.palette_rgb)
        if len(color) != 3 or any(value < 0 or value > 255 for value in color):
            raise SurfaceAttributeError("palette RGB must be a uint8 triple")
        if self.rgb_source != "instance_palette":
            raise SurfaceAttributeError("PLY color source must remain instance_palette")
        object.__setattr__(self, "points_xyz", _immutable(points, np.float32))
        object.__setattr__(self, "normals_xyz", _immutable(normals, np.float32))
        object.__setattr__(self, "normal_valid", _immutable(valid, np.bool_))
        object.__setattr__(
            self,
            "original_vertex_indices",
            _immutable(indices, np.int64),
        )
        object.__setattr__(self, "palette_rgb", color)


@dataclass(frozen=True, slots=True)
class Projection:
    """Vectorized world-to-camera projection before depth validation."""

    camera_xyz: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    projectable: np.ndarray

    def __post_init__(self) -> None:
        camera = _matrix(self.camera_xyz, name="camera coordinates", columns=3)
        rows = np.asarray(self.rows)
        columns = np.asarray(self.columns)
        projectable = np.asarray(self.projectable)
        count = len(camera)
        if rows.shape != (count,) or columns.shape != (count,):
            raise SurfaceAttributeError("projection pixel arrays must align with points")
        if projectable.shape != (count,) or projectable.dtype != np.bool_:
            raise SurfaceAttributeError("projectable must be a boolean row mask")
        object.__setattr__(self, "camera_xyz", _immutable(camera, np.float64))
        object.__setattr__(self, "rows", _immutable(rows, np.int64))
        object.__setattr__(self, "columns", _immutable(columns, np.int64))
        object.__setattr__(self, "projectable", _immutable(projectable, np.bool_))


@dataclass(frozen=True, slots=True)
class SurfaceSupport:
    """Depth-consistent camera observations for a subset of candidates."""

    candidate_indices: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    rgb_uint8: np.ndarray
    camera_depth_m: np.ndarray
    observed_depth_m: np.ndarray
    depth_residual_m: np.ndarray
    rgb_source: str = "same_visit_rgbd_reprojected"

    def __post_init__(self) -> None:
        indices = np.asarray(self.candidate_indices)
        rows = np.asarray(self.rows)
        columns = np.asarray(self.columns)
        rgb = _matrix(self.rgb_uint8, name="support RGB", columns=3)
        camera_depth = np.asarray(self.camera_depth_m)
        observed_depth = np.asarray(self.observed_depth_m)
        residual = np.asarray(self.depth_residual_m)
        count = len(indices)
        if any(
            value.shape != (count,)
            for value in (rows, columns, camera_depth, observed_depth, residual)
        ) or len(rgb) != count:
            raise SurfaceAttributeError("surface support arrays must have equal rows")
        if self.rgb_source != "same_visit_rgbd_reprojected":
            raise SurfaceAttributeError("surface support RGB source is invalid")
        object.__setattr__(self, "candidate_indices", _immutable(indices, np.int64))
        object.__setattr__(self, "rows", _immutable(rows, np.int64))
        object.__setattr__(self, "columns", _immutable(columns, np.int64))
        object.__setattr__(self, "rgb_uint8", _immutable(rgb, np.uint8))
        object.__setattr__(
            self, "camera_depth_m", _immutable(camera_depth, np.float32)
        )
        object.__setattr__(
            self, "observed_depth_m", _immutable(observed_depth, np.float32)
        )
        object.__setattr__(
            self, "depth_residual_m", _immutable(residual, np.float32)
        )


def _normalized_requested_colors(
    requested_colors: Iterable[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], ...]:
    colors = tuple(
        sorted(set(tuple(int(value) for value in color) for color in requested_colors))
    )
    if any(len(color) != 3 or any(value < 0 or value > 255 for value in color) for color in colors):
        raise SurfaceAttributeError("requested colors must be RGB uint8 triples")
    return colors


def group_surface_attributes_by_color(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    normals_xyz: np.ndarray,
    original_vertex_indices: np.ndarray,
    requested_colors: Iterable[tuple[int, int, int]],
) -> dict[tuple[int, int, int], SurfaceGroup]:
    """Apply OVI's stable palette grouping to coordinates and aligned attributes."""

    points = _matrix(points_xyz, name="surface points", columns=3)
    normals = _matrix(normals_xyz, name="surface normals", columns=3)
    colors = _matrix(colors_rgb, name="surface colors", columns=3)
    indices = np.asarray(original_vertex_indices)
    if len(normals) != len(points) or len(colors) != len(points):
        raise SurfaceAttributeError("PLY vertex attributes must have equal rows")
    if indices.shape != (len(points),) or not np.issubdtype(indices.dtype, np.integer):
        raise SurfaceAttributeError("original vertex indices must align with PLY rows")
    if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise SurfaceAttributeError("original vertex indices must be unique and nonnegative")

    codes = _color_codes(colors)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    normal_values = np.asarray(normals, dtype=np.float32)
    normal_norms = np.linalg.norm(normal_values, axis=1)
    normal_valid = np.all(np.isfinite(normal_values), axis=1) & np.isfinite(normal_norms) & (
        normal_norms > 0.0
    )
    normalized_normals = np.zeros(normal_values.shape, dtype=np.float32)
    normalized_normals[normal_valid] = (
        normal_values[normal_valid] / normal_norms[normal_valid, None]
    )

    grouped: dict[tuple[int, int, int], SurfaceGroup] = {}
    for color in _normalized_requested_colors(requested_colors):
        code = (color[0] << 16) | (color[1] << 8) | color[2]
        left = int(np.searchsorted(sorted_codes, code, side="left"))
        right = int(np.searchsorted(sorted_codes, code, side="right"))
        if left == right:
            continue
        selected = order[left:right]
        grouped[color] = SurfaceGroup(
            points_xyz=np.asarray(points[selected], dtype=np.float32),
            normals_xyz=normalized_normals[selected],
            normal_valid=normal_valid[selected],
            original_vertex_indices=indices[selected],
            palette_rgb=color,
        )
    return grouped


def load_ply_surface_attributes(
    path: str | Path,
    requested_colors: Iterable[tuple[int, int, int]],
) -> dict[tuple[int, int, int], SurfaceGroup]:
    """Read one fixed-record PLY and retain source row identity."""

    from plyfile import PlyData

    ply_path = Path(path)
    if ply_path.is_symlink() or not ply_path.is_file():
        raise SurfaceAttributeError("OVI instance PLY is missing or a symlink")
    vertices = PlyData.read(ply_path, mmap="c")["vertex"]
    names = set(vertices.data.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "normal_x",
        "normal_y",
        "normal_z",
        "red",
        "green",
        "blue",
    }
    if not required.issubset(names):
        raise SurfaceAttributeError(
            f"OVI instance PLY lacks properties: {sorted(required - names)}"
        )
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    normals = np.column_stack(
        (vertices["normal_x"], vertices["normal_y"], vertices["normal_z"])
    )
    colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"]))
    return group_surface_attributes_by_color(
        points,
        colors,
        normals,
        np.arange(len(points), dtype=np.int64),
        requested_colors,
    )


def validate_surface_group(entity_points_xyz: np.ndarray, group: SurfaceGroup) -> None:
    """Require exact float32 row alignment with a frozen OVI entity."""

    if not isinstance(group, SurfaceGroup):
        raise TypeError("group must be SurfaceGroup")
    entity_points = _matrix(entity_points_xyz, name="entity points", columns=3)
    if not np.array_equal(np.asarray(entity_points, dtype=np.float32), group.points_xyz):
        raise SurfaceAttributeError(
            "grouped PLY points must exactly match frozen entity points"
        )


def depth_millimeters_to_meters(depth_mm: np.ndarray) -> np.ndarray:
    """Convert the bound TESSE uint16 depth encoding to contiguous metres."""

    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise SurfaceAttributeError("source depth must be a 2D uint16 millimetre image")
    return np.ascontiguousarray(depth, dtype=np.float32) / np.float32(1000.0)


def project_world_points(
    points_xyz: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> Projection:
    """Project world points with the frozen camera-to-world convention."""

    points = np.asarray(
        _matrix(points_xyz, name="world points", columns=3), dtype=np.float64
    )
    pose = np.asarray(camera_to_world, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise SurfaceAttributeError("camera_to_world must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise SurfaceAttributeError("camera_to_world homogeneous row is invalid")
    if not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError("intrinsics must be CameraIntrinsics")
    try:
        world_to_camera = np.linalg.inv(pose)
    except np.linalg.LinAlgError as error:
        raise SurfaceAttributeError("camera_to_world must be invertible") from error
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = camera[:, 2]
    finite = np.all(np.isfinite(camera), axis=1)
    in_front = finite & (depth > 0.0)
    rows = np.full(len(points), -1, dtype=np.int64)
    columns = np.full(len(points), -1, dtype=np.int64)
    rows[in_front] = np.rint(
        intrinsics.fy * camera[in_front, 1] / depth[in_front] + intrinsics.cy
    ).astype(np.int64)
    columns[in_front] = np.rint(
        intrinsics.fx * camera[in_front, 0] / depth[in_front] + intrinsics.cx
    ).astype(np.int64)
    projectable = (
        in_front
        & (rows >= 0)
        & (rows < intrinsics.height)
        & (columns >= 0)
        & (columns < intrinsics.width)
    )
    rows[~projectable] = -1
    columns[~projectable] = -1
    return Projection(camera, rows, columns, projectable)


def sample_depth_consistent_rgb(
    projection: Projection,
    rgb_uint8: np.ndarray,
    depth_m: np.ndarray,
    *,
    depth_tolerance_m: float,
    candidate_visit_ids: np.ndarray,
    frame_visit_id: int,
    rgb_source: str,
    minimum_valid_neighbours: int = 5,
    edge_range_factor: float = 2.0,
) -> SurfaceSupport:
    """Select same-visit camera RGB where surface and measured depth agree."""

    if not isinstance(projection, Projection):
        raise TypeError("projection must be Projection")
    rgb = np.asarray(rgb_uint8)
    depth = np.asarray(depth_m)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise SurfaceAttributeError("camera RGB must be an HxWx3 uint8 image")
    if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise SurfaceAttributeError("depth must align with camera RGB")
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise SurfaceAttributeError("camera images must be non-empty")
    if rgb_source != "camera_rgb":
        if "palette" in str(rgb_source):
            raise SurfaceAttributeError("instance palette cannot provide camera RGB")
        raise SurfaceAttributeError("RGB must come from the source camera")
    tolerance = float(depth_tolerance_m)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise SurfaceAttributeError("depth tolerance must be finite and positive")
    if type(frame_visit_id) is not int or frame_visit_id not in (0, 1):
        raise SurfaceAttributeError("frame visit ID must be 0 or 1")
    visits = np.asarray(candidate_visit_ids)
    if visits.shape != (len(projection.camera_xyz),) or not np.issubdtype(
        visits.dtype, np.integer
    ):
        raise SurfaceAttributeError("candidate visit IDs must align with projection")
    if np.any(visits != frame_visit_id):
        raise SurfaceAttributeError("surface RGB support must come from the same visit")
    if type(minimum_valid_neighbours) is not int or not 1 <= minimum_valid_neighbours <= 9:
        raise SurfaceAttributeError("minimum valid neighbours must be in [1, 9]")
    factor = float(edge_range_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise SurfaceAttributeError("edge range factor must be finite and positive")

    height, width = depth.shape
    inside = (
        projection.projectable
        & (projection.rows >= 0)
        & (projection.rows < height)
        & (projection.columns >= 0)
        & (projection.columns < width)
    )
    candidate_indices = np.flatnonzero(inside)
    if len(candidate_indices) == 0:
        return _empty_support()
    rows = projection.rows[candidate_indices]
    columns = projection.columns[candidate_indices]
    observed = np.asarray(depth[rows, columns], dtype=np.float64)
    camera_depth = projection.camera_xyz[candidate_indices, 2]

    padded = np.pad(
        np.asarray(depth, dtype=np.float64),
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=np.nan,
    )
    neighbours = np.stack(
        [
            padded[rows + row_offset + 1, columns + column_offset + 1]
            for row_offset in (-1, 0, 1)
            for column_offset in (-1, 0, 1)
        ],
        axis=1,
    )
    valid_neighbours = np.isfinite(neighbours) & (neighbours > 0.0)
    neighbour_count = valid_neighbours.sum(axis=1)
    minimum = np.min(np.where(valid_neighbours, neighbours, np.inf), axis=1)
    maximum = np.max(np.where(valid_neighbours, neighbours, -np.inf), axis=1)
    residual = np.abs(observed - camera_depth)
    valid = (
        np.isfinite(observed)
        & (observed > 0.0)
        & np.isfinite(camera_depth)
        & (camera_depth > 0.0)
        & (residual <= tolerance)
        & (neighbour_count >= minimum_valid_neighbours)
        & ((maximum - minimum) <= factor * tolerance)
    )
    selected = candidate_indices[valid]
    if len(selected) == 0:
        return _empty_support()
    return SurfaceSupport(
        candidate_indices=selected,
        rows=projection.rows[selected],
        columns=projection.columns[selected],
        rgb_uint8=rgb[projection.rows[selected], projection.columns[selected]],
        camera_depth_m=projection.camera_xyz[selected, 2],
        observed_depth_m=depth[projection.rows[selected], projection.columns[selected]],
        depth_residual_m=np.abs(
            depth[projection.rows[selected], projection.columns[selected]]
            - projection.camera_xyz[selected, 2]
        ),
    )


def _empty_support() -> SurfaceSupport:
    return SurfaceSupport(
        candidate_indices=np.empty(0, dtype=np.int64),
        rows=np.empty(0, dtype=np.int64),
        columns=np.empty(0, dtype=np.int64),
        rgb_uint8=np.empty((0, 3), dtype=np.uint8),
        camera_depth_m=np.empty(0, dtype=np.float32),
        observed_depth_m=np.empty(0, dtype=np.float32),
        depth_residual_m=np.empty(0, dtype=np.float32),
    )
