"""Complete, source-row-preserving I/O for native OVI fine meshes."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from plyfile import PlyData

from src.evaluation.baselines.ovimap import _color_codes


def _immutable(value: object, dtype: object) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    if result.flags.writeable or result.base is not None:
        result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class NativeOviSurface:
    vertices_xyz: np.ndarray
    normals_xyz: np.ndarray
    triangles: np.ndarray
    palette_rgb: np.ndarray
    source_vertex_indices: np.ndarray

    def __post_init__(self) -> None:
        dtypes = {
            "vertices_xyz": np.float32,
            "normals_xyz": np.float32,
            "triangles": np.int64,
            "palette_rgb": np.uint8,
            "source_vertex_indices": np.int64,
        }
        for field in fields(self):
            object.__setattr__(
                self, field.name, _immutable(getattr(self, field.name), dtypes[field.name])
            )
        count = len(self.vertices_xyz)
        if self.vertices_xyz.shape != (count, 3):
            raise ValueError("vertices_xyz must have shape (N, 3)")
        if self.normals_xyz.shape != (count, 3):
            raise ValueError("normals_xyz must have shape (N, 3)")
        if self.palette_rgb.shape != (count, 3):
            raise ValueError("palette_rgb must have shape (N, 3)")
        if self.triangles.ndim != 2 or self.triangles.shape[1:] != (3,):
            raise ValueError("OVI source mesh faces must be triangular")
        if self.source_vertex_indices.shape != (count,):
            raise ValueError("source vertex indices must align with vertices")
        if not np.isfinite(self.vertices_xyz).all() or not np.isfinite(
            self.normals_xyz
        ).all():
            raise ValueError("OVI source geometry must be finite")
        if self.triangles.size and (
            int(self.triangles.min()) < 0 or int(self.triangles.max()) >= count
        ):
            raise ValueError("OVI source triangle indices lie outside the vertex array")


def _triangles(face_values: np.ndarray) -> np.ndarray:
    if len(face_values) == 0:
        return np.empty((0, 3), dtype=np.int64)
    if face_values.ndim == 2:
        values = np.asarray(face_values)
        if values.shape[1] != 3:
            raise ValueError("OVI source mesh faces must be triangular")
        return values.astype(np.int64, copy=False)
    lengths = np.fromiter((len(value) for value in face_values), dtype=np.int64)
    if np.any(lengths != 3):
        raise ValueError("OVI source mesh faces must be triangular")
    return np.vstack(face_values).astype(np.int64, copy=False)


def load_native_ovi_surface(path: str | Path) -> NativeOviSurface:
    """Load native OVI vertices and faces without color-to-label reinterpretation."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("OVI source mesh is missing or a symlink")
    ply = PlyData.read(source, mmap="c")
    if "vertex" not in ply or "face" not in ply:
        raise ValueError("OVI source mesh must contain vertex and face elements")
    vertices = ply["vertex"]
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
        raise ValueError(f"OVI source mesh lacks properties: {sorted(required - names)}")
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    normals = np.column_stack(
        (vertices["normal_x"], vertices["normal_y"], vertices["normal_z"])
    )
    colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"]))
    triangles = _triangles(ply["face"]["vertex_indices"])
    return NativeOviSurface(
        vertices_xyz=points,
        normals_xyz=normals,
        triangles=triangles,
        palette_rgb=colors,
        source_vertex_indices=np.arange(len(points), dtype=np.int64),
    )


def load_cached_native_ovi_surface(
    path: str | Path,
    cache_path: str | Path,
    *,
    source_sha256: str,
) -> NativeOviSurface:
    """Load a native surface through a local cache bound to its verified hash."""

    if not isinstance(source_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", source_sha256
    ) is None:
        raise ValueError("source_sha256 must be a lowercase SHA256 digest")
    cache = Path(cache_path)
    if cache.is_symlink():
        raise ValueError("native surface cache must not be a symlink")
    field_names = {field.name for field in fields(NativeOviSurface)}
    if cache.exists():
        if not cache.is_file():
            raise ValueError("native surface cache must be a regular file")
        try:
            with np.load(cache, allow_pickle=False) as payload:
                if set(payload.files) != {"source_sha256", *field_names}:
                    raise ValueError("native surface cache schema is invalid")
                if str(payload["source_sha256"].item()) != source_sha256:
                    raise ValueError("native surface cache source hash differs")
                return NativeOviSurface(
                    **{name: payload[name] for name in field_names}
                )
        except (OSError, ValueError) as error:
            if isinstance(error, ValueError) and "source hash" in str(error):
                raise
            raise ValueError("native surface cache is unreadable") from error

    surface = load_native_ovi_surface(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache.name}.", suffix=".tmp", dir=cache.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                source_sha256=np.asarray(source_sha256),
                **{name: getattr(surface, name) for name in field_names},
            )
        temporary.replace(cache)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return surface


def bind_ovi_owner_ids(
    palette_rgb: np.ndarray,
    colors_by_instance: Mapping[int, tuple[int, int, int]],
) -> np.ndarray:
    """Convert OVI palette colors to explicit owner IDs; unmatched rows stay zero."""

    if not isinstance(colors_by_instance, Mapping) or not colors_by_instance:
        raise ValueError("colors_by_instance must be a non-empty mapping")
    pairs: list[tuple[int, int]] = []
    seen_codes: set[int] = set()
    for raw_instance_id, raw_color in colors_by_instance.items():
        instance_id = int(raw_instance_id)
        color = tuple(int(value) for value in raw_color)
        if instance_id <= 0 or len(color) != 3 or any(value < 0 or value > 255 for value in color):
            raise ValueError("owner bindings require positive IDs and RGB uint8 triples")
        code = (color[0] << 16) | (color[1] << 8) | color[2]
        if code in seen_codes:
            raise ValueError("OVI owner colors must be unique")
        seen_codes.add(code)
        pairs.append((code, instance_id))
    pairs.sort()
    known_codes = np.asarray([item[0] for item in pairs], dtype=np.uint32)
    known_ids = np.asarray([item[1] for item in pairs], dtype=np.int64)
    observed_codes = _color_codes(palette_rgb)
    locations = np.searchsorted(known_codes, observed_codes)
    bounded = np.minimum(locations, len(known_codes) - 1)
    matched = (locations < len(known_codes)) & (
        known_codes[bounded] == observed_codes
    )
    result = np.zeros(len(observed_codes), dtype=np.int64)
    result[matched] = known_ids[bounded[matched]]
    result.setflags(write=False)
    return result


def select_source_surface_voxels(
    vertices_xyz: np.ndarray,
    *,
    voxel_size_m: float,
    origin_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Select the first source row in each world-coordinate voxel."""

    size = float(voxel_size_m)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")
    vertices = np.asarray(vertices_xyz, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("vertices_xyz must have shape (N, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices_xyz must be finite")
    origin = (
        np.zeros(3, dtype=np.float64)
        if origin_xyz is None
        else np.asarray(origin_xyz, dtype=np.float64)
    )
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin_xyz must be a finite length-three vector")
    if len(vertices) == 0:
        return np.empty(0, dtype=np.int64)
    scaled = np.floor((vertices - origin) / size)
    if not np.isfinite(scaled).all():
        raise ValueError("voxel coordinates exceed finite range")
    limits = np.iinfo(np.int64)
    if np.any(scaled < limits.min) or np.any(scaled > limits.max):
        raise ValueError("voxel coordinates exceed int64 range")
    keys = scaled.astype(np.int64)
    _, first_rows = np.unique(keys, axis=0, return_index=True)
    selected = np.sort(first_rows.astype(np.int64, copy=False))
    selected.setflags(write=False)
    return selected


__all__ = [
    "NativeOviSurface",
    "bind_ovi_owner_ids",
    "load_cached_native_ovi_surface",
    "load_native_ovi_surface",
    "select_source_surface_voxels",
]
