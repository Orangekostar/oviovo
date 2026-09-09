"""Canonical fine-surface state and source-grounded visualization exports."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement


class CurrentEvidenceState(IntEnum):
    CURRENT_OBSERVED = 1
    HISTORICAL_OCCLUDED = 2
    HISTORICAL_UNOBSERVED = 3
    HISTORICAL_UNCERTAIN = 4
    REPLACED_BY_CURRENT = 5
    REVOKED_VISIBLE_FREE = 6


class SemanticSource(IntEnum):
    UNKNOWN = 0
    LOCAL_SURFACE = 1
    OWNER_ENTITY = 2
    BLENDED = 3


_ARRAY_DTYPES: dict[str, Any] = {
    "vertices_xyz": np.float32,
    "normals_xyz": np.float32,
    "triangles": np.int64,
    "source_surface_indices": np.uint16,
    "source_vertex_indices": np.int64,
    "source_visit_ids": np.int16,
    "geometry_epochs": np.int32,
    "observed_rgb_uint8": np.uint8,
    "rgb_valid": np.bool_,
    "current_valid": np.bool_,
    "evidence_state_codes": np.uint8,
    "last_supported_frames": np.int32,
    "owner_entity_ids": np.int64,
    "owner_confidences": np.float32,
    "semantic_ids": np.int32,
    "semantic_confidences": np.float32,
    "semantic_support_reliabilities": np.float32,
    "semantic_source_codes": np.uint8,
}


def _immutable(value: object, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.flags.writeable or array.base is not None:
        array = array.copy()
    array.setflags(write=False)
    return array


def _validate_probability(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} must contain finite values in [0, 1]")


@dataclass(frozen=True, slots=True)
class CurrentSurfaceView:
    """One immutable fine surface from which every published view is derived."""

    surface_id: str
    vertices_xyz: np.ndarray
    normals_xyz: np.ndarray
    triangles: np.ndarray
    source_surface_indices: np.ndarray
    source_vertex_indices: np.ndarray
    source_visit_ids: np.ndarray
    geometry_epochs: np.ndarray
    observed_rgb_uint8: np.ndarray
    rgb_valid: np.ndarray
    current_valid: np.ndarray
    evidence_state_codes: np.ndarray
    last_supported_frames: np.ndarray
    owner_entity_ids: np.ndarray
    owner_confidences: np.ndarray
    semantic_ids: np.ndarray
    semantic_confidences: np.ndarray
    semantic_support_reliabilities: np.ndarray
    semantic_source_codes: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.surface_id, str) or not self.surface_id.strip():
            raise ValueError("surface_id must be a non-empty string")
        for field in fields(self):
            if field.name == "surface_id":
                continue
            object.__setattr__(
                self,
                field.name,
                _immutable(getattr(self, field.name), _ARRAY_DTYPES[field.name]),
            )

        if self.vertices_xyz.ndim != 2 or self.vertices_xyz.shape[1] != 3:
            raise ValueError("vertices_xyz must have shape (N, 3)")
        count = len(self.vertices_xyz)
        if self.normals_xyz.shape != (count, 3):
            raise ValueError("normals_xyz must have shape (N, 3)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (M, 3)")
        if self.observed_rgb_uint8.shape != (count, 3):
            raise ValueError("observed_rgb_uint8 must have shape (N, 3)")
        for name in _ARRAY_DTYPES:
            if name in {"vertices_xyz", "normals_xyz", "triangles", "observed_rgb_uint8"}:
                continue
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
        if not np.isfinite(self.vertices_xyz).all() or not np.isfinite(
            self.normals_xyz
        ).all():
            raise ValueError("surface geometry must be finite")
        if self.triangles.size and (
            int(self.triangles.min()) < 0 or int(self.triangles.max()) >= count
        ):
            raise ValueError("triangle indices lie outside the vertex array")
        if np.any(self.source_vertex_indices < 0) or np.any(self.source_visit_ids < 0):
            raise ValueError("source vertex and visit identifiers must be nonnegative")
        if np.any(self.geometry_epochs < 0) or np.any(self.owner_entity_ids < 0):
            raise ValueError("geometry epochs and owner identifiers must be nonnegative")
        if np.any(self.semantic_ids < 0):
            raise ValueError("semantic identifiers must be nonnegative")
        if count > 1:
            previous_surface = self.source_surface_indices[:-1]
            current_surface = self.source_surface_indices[1:]
            previous_vertex = self.source_vertex_indices[:-1]
            current_vertex = self.source_vertex_indices[1:]
            invalid_order = (current_surface < previous_surface) | (
                (current_surface == previous_surface)
                & (current_vertex <= previous_vertex)
            )
            if np.any(invalid_order):
                raise ValueError(
                    "source surface/vertex pairs must be unique and lexicographically ordered"
                )
        if self.triangles.size:
            triangle_sources = self.source_surface_indices[self.triangles]
            if np.any(triangle_sources != triangle_sources[:, :1]):
                raise ValueError("a triangle cannot cross source surfaces")
        _validate_probability("owner_confidences", self.owner_confidences)
        _validate_probability("semantic_confidences", self.semantic_confidences)
        _validate_probability(
            "semantic_support_reliabilities", self.semantic_support_reliabilities
        )
        evidence_values = {int(value) for value in CurrentEvidenceState}
        if not set(int(value) for value in np.unique(self.evidence_state_codes)) <= evidence_values:
            raise ValueError("evidence_state_codes contain an unknown state")
        source_values = {int(value) for value in SemanticSource}
        if not set(int(value) for value in np.unique(self.semantic_source_codes)) <= source_values:
            raise ValueError("semantic_source_codes contain an unknown source")
        terminal_invalid = np.isin(
            self.evidence_state_codes,
            [
                int(CurrentEvidenceState.REPLACED_BY_CURRENT),
                int(CurrentEvidenceState.REVOKED_VISIBLE_FREE),
            ],
        )
        if np.any(self.current_valid == terminal_invalid):
            raise ValueError("current_valid disagrees with terminal evidence state")


@dataclass(frozen=True, slots=True)
class CurrentSurfaceSelection:
    source_row_indices: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_row_indices", _immutable(self.source_row_indices, np.int64)
        )
        object.__setattr__(self, "triangles", _immutable(self.triangles, np.int64))


@dataclass(frozen=True, slots=True)
class ExportedCurrentSurface:
    output_dir: Path
    manifest: Path


def select_current_surface(surface: CurrentSurfaceView) -> CurrentSurfaceSelection:
    if not isinstance(surface, CurrentSurfaceView):
        raise TypeError("surface must be a CurrentSurfaceView")
    selected = np.flatnonzero(surface.current_valid).astype(np.int64, copy=False)
    inverse = np.full(len(surface.vertices_xyz), -1, dtype=np.int64)
    inverse[selected] = np.arange(len(selected), dtype=np.int64)
    if len(surface.triangles):
        keep = np.all(surface.current_valid[surface.triangles], axis=1)
        triangles = inverse[surface.triangles[keep]]
    else:
        triangles = np.empty((0, 3), dtype=np.int64)
    return CurrentSurfaceSelection(selected, triangles)


def _stable_entity_colors(entity_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(entity_ids, dtype=np.uint64)
    mixed = values * np.uint64(0x9E3779B185EBCA87)
    mixed ^= mixed >> np.uint64(29)
    colors = np.column_stack(
        (
            48 + (mixed & np.uint64(191)),
            48 + ((mixed >> np.uint64(8)) & np.uint64(191)),
            48 + ((mixed >> np.uint64(16)) & np.uint64(191)),
        )
    ).astype(np.uint8)
    colors[entity_ids == 0] = (96, 96, 96)
    return colors


_STATE_COLORS: dict[int, tuple[int, int, int]] = {
    int(CurrentEvidenceState.CURRENT_OBSERVED): (30, 160, 75),
    int(CurrentEvidenceState.HISTORICAL_OCCLUDED): (45, 105, 190),
    int(CurrentEvidenceState.HISTORICAL_UNOBSERVED): (125, 125, 125),
    int(CurrentEvidenceState.HISTORICAL_UNCERTAIN): (235, 175, 30),
    int(CurrentEvidenceState.REPLACED_BY_CURRENT): (145, 70, 180),
    int(CurrentEvidenceState.REVOKED_VISIBLE_FREE): (210, 45, 45),
}


def _view_colors(
    surface: CurrentSurfaceView,
    rows: np.ndarray,
    semantic_palette: Mapping[int, tuple[int, int, int]],
    mode: str,
) -> np.ndarray:
    if mode == "rgb":
        rgb = surface.observed_rgb_uint8[rows].copy()
        rgb[~surface.rgb_valid[rows]] = (128, 128, 128)
        return rgb
    if mode == "instance":
        return _stable_entity_colors(surface.owner_entity_ids[rows])
    if mode == "semantic":
        keys = np.asarray(sorted(semantic_palette), dtype=np.int64)
        colors = np.asarray([semantic_palette[int(key)] for key in keys], dtype=np.uint8)
        locations = np.searchsorted(keys, surface.semantic_ids[rows])
        return colors[locations]
    if mode == "state":
        keys = np.asarray(sorted(_STATE_COLORS), dtype=np.uint8)
        colors = np.asarray([_STATE_COLORS[int(key)] for key in keys], dtype=np.uint8)
        locations = np.searchsorted(keys, surface.evidence_state_codes[rows])
        return colors[locations]
    raise ValueError(f"unknown current-surface color mode: {mode}")


_PLY_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("normal_x", "<f4"),
        ("normal_y", "<f4"),
        ("normal_z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("source_surface_index", "<u2"),
        ("source_vertex_index", "<i4"),
        ("source_visit", "<i2"),
        ("geometry_epoch", "<i4"),
        ("rgb_valid", "u1"),
        ("evidence_state", "u1"),
        ("last_supported_frame", "<i4"),
        ("owner_entity_id", "<i4"),
        ("owner_confidence", "<f4"),
        ("semantic_id", "<i4"),
        ("semantic_confidence", "<f4"),
        ("semantic_support_reliability", "<f4"),
        ("semantic_source", "u1"),
    ]
)


def _checked_int32(name: str, values: np.ndarray) -> np.ndarray:
    if values.size and (
        int(values.min()) < int(np.iinfo(np.int32).min)
        or int(values.max()) > int(np.iinfo(np.int32).max)
    ):
        raise ValueError(f"{name} cannot be represented in a PLY int32 property")
    return values.astype(np.int32, copy=False)


def _ply_vertices(
    surface: CurrentSurfaceView, rows: np.ndarray, colors: np.ndarray
) -> np.ndarray:
    vertices = np.empty(len(rows), dtype=_PLY_VERTEX_DTYPE)
    vertices["x"], vertices["y"], vertices["z"] = surface.vertices_xyz[rows].T
    vertices["normal_x"], vertices["normal_y"], vertices["normal_z"] = (
        surface.normals_xyz[rows].T
    )
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    vertices["source_surface_index"] = surface.source_surface_indices[rows]
    vertices["source_vertex_index"] = _checked_int32(
        "source_vertex_indices", surface.source_vertex_indices[rows]
    )
    vertices["source_visit"] = surface.source_visit_ids[rows]
    vertices["geometry_epoch"] = surface.geometry_epochs[rows]
    vertices["rgb_valid"] = surface.rgb_valid[rows]
    vertices["evidence_state"] = surface.evidence_state_codes[rows]
    vertices["last_supported_frame"] = surface.last_supported_frames[rows]
    vertices["owner_entity_id"] = _checked_int32(
        "owner_entity_ids", surface.owner_entity_ids[rows]
    )
    vertices["owner_confidence"] = surface.owner_confidences[rows]
    vertices["semantic_id"] = surface.semantic_ids[rows]
    vertices["semantic_confidence"] = surface.semantic_confidences[rows]
    vertices["semantic_support_reliability"] = (
        surface.semantic_support_reliabilities[rows]
    )
    vertices["semantic_source"] = surface.semantic_source_codes[rows]
    return vertices


def _write_ply(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    elements = [PlyElement.describe(vertices, "vertex")]
    if len(triangles):
        faces = np.empty(len(triangles), dtype=[("vertex_indices", "<i4", (3,))])
        faces["vertex_indices"] = triangles.astype(np.int32, copy=False)
        elements.append(PlyElement.describe(faces, "face"))
    PlyData(elements, text=False, byte_order="<").write(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _publish_directory_no_replace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError("atomic no-clobber directory publication is unavailable") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        number = ctypes.get_errno()
        if number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(target)
        raise OSError(number, os.strerror(number), target)


def _normalized_palette(
    palette: Mapping[int, tuple[int, int, int]], required_ids: set[int]
) -> dict[int, tuple[int, int, int]]:
    result: dict[int, tuple[int, int, int]] = {}
    for raw_id, raw_color in palette.items():
        semantic_id = int(raw_id)
        color = tuple(int(channel) for channel in raw_color)
        if semantic_id < 0 or len(color) != 3 or any(c < 0 or c > 255 for c in color):
            raise ValueError("semantic palette entries must be nonnegative IDs and RGB triples")
        result[semantic_id] = color
    missing = sorted(required_ids - set(result))
    if missing:
        raise ValueError(f"semantic palette is missing current IDs: {missing}")
    return result


def _normalized_owner_id_table(
    table: Mapping[int, str], required_ids: set[int]
) -> dict[int, str]:
    if not isinstance(table, Mapping):
        raise TypeError("owner_id_table must be a mapping")
    result: dict[int, str] = {}
    names: set[str] = set()
    for raw_id, raw_name in table.items():
        if isinstance(raw_id, bool):
            raise ValueError("owner ID table keys must be nonnegative integers")
        owner_id = int(raw_id)
        name = str(raw_name).strip()
        if owner_id < 0 or not name:
            raise ValueError("owner ID table entries require nonnegative IDs and names")
        if name in names:
            raise ValueError("owner ID table names must be unique")
        result[owner_id] = name
        names.add(name)
    missing = sorted(required_ids - set(result))
    if missing:
        raise ValueError(f"owner ID table is missing current IDs: {missing}")
    return result


def export_current_surface(
    surface: CurrentSurfaceView,
    output_dir: Path,
    *,
    semantic_palette: Mapping[int, tuple[int, int, int]],
    owner_id_table: Mapping[int, str],
    source_surfaces: Sequence[Mapping[str, object]],
) -> ExportedCurrentSurface:
    """Publish four recolorings of one filtered geometry plus its full sidecar."""

    if not isinstance(surface, CurrentSurfaceView):
        raise TypeError("surface must be a CurrentSurfaceView")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    selection = select_current_surface(surface)
    required_ids = set(int(value) for value in surface.semantic_ids[selection.source_row_indices])
    palette = _normalized_palette(semantic_palette, required_ids)
    required_owner_ids = set(
        int(value) for value in surface.owner_entity_ids[selection.source_row_indices]
    )
    owners = _normalized_owner_id_table(owner_id_table, required_owner_ids)
    sources = [dict(record) for record in source_surfaces]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for mode in ("rgb", "instance", "semantic", "state"):
            view_colors = _view_colors(
                surface, selection.source_row_indices, palette, mode
            )
            _write_ply(
                staging / f"current_{mode}.ply",
                _ply_vertices(surface, selection.source_row_indices, view_colors),
                selection.triangles,
            )
        np.savez_compressed(
            staging / "current_surface.npz",
            surface_id=np.asarray(surface.surface_id),
            **{
                name: getattr(surface, name)
                for name in _ARRAY_DTYPES
            },
        )
        (staging / "label_palette.json").write_bytes(
            _canonical_json(
                {str(key): list(value) for key, value in sorted(palette.items())}
            )
        )
        (staging / "owner_id_table.json").write_bytes(
            _canonical_json(
                {str(key): value for key, value in sorted(owners.items())}
            )
        )
        artifact_names = [
            "current_rgb.ply",
            "current_instance.ply",
            "current_semantic.ply",
            "current_state.ply",
            "current_surface.npz",
            "label_palette.json",
            "owner_id_table.json",
        ]
        manifest_payload = {
            "schema_version": 1,
            "status": "PASS",
            "surface_id": surface.surface_id,
            "canonical_vertex_count": len(surface.vertices_xyz),
            "canonical_face_count": len(surface.triangles),
            "current_vertex_count": len(selection.source_row_indices),
            "current_face_count": len(selection.triangles),
            "geometry_shared_across_views": True,
            "current_selection_sha256": hashlib.sha256(
                selection.source_row_indices.astype("<i8", copy=False).tobytes()
                + selection.triangles.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "source_surfaces": sources,
            "artifacts": {
                name: {"sha256": _sha256(staging / name), "byte_count": (staging / name).stat().st_size}
                for name in artifact_names
            },
        }
        manifest_name = "current_surface_manifest.json"
        (staging / manifest_name).write_bytes(_canonical_json(manifest_payload))
        _publish_directory_no_replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return ExportedCurrentSurface(
        output_dir=output,
        manifest=output / "current_surface_manifest.json",
    )
