"""Recolor labeled CROVE meshes without changing benchmark state or geometry."""

from __future__ import annotations

import colorsys
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from plyfile import PlyData

VisualizationMode = Literal["rgb", "instance", "semantic", "dynamic"]

DYNAMIC_STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "static": (46, 160, 67),
    "dynamic": (214, 39, 40),
    "unknown": (255, 193, 7),
    "uncertain": (255, 193, 7),
    "dormant": (117, 117, 117),
    "removed": (117, 117, 117),
}

_RGB_PROPERTIES = ("red", "green", "blue")
_POSITION_PROPERTIES = ("x", "y", "z")
_VALID_MODES = frozenset(("rgb", "instance", "semantic", "dynamic"))
_MAX_SIDECAR_BYTES = 16 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: str | Path, label: str) -> Path:
    value = Path(os.path.abspath(os.fspath(path)))
    try:
        current = Path(value.anchor)
        for component in value.parts[1:]:
            current /= component
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"{label} must be an existing regular file")
    except FileNotFoundError as exc:
        raise ValueError(f"{label} must be an existing regular file") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be an existing regular file")
    return value


def _file_record(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    return {
        "path": relative_path if relative_path is not None else str(path),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label)
    content = source.read_bytes()
    if len(content) > _MAX_SIDECAR_BYTES:
        raise ValueError(f"{label} exceeds the size limit")
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return source, payload


def _rgb(value: object, label: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must be an RGB triplet")
    if any(type(item) is not int or not 0 <= item <= 255 for item in value):
        raise ValueError(f"{label} must contain uint8 values")
    return tuple(value)  # type: ignore[return-value]


def _semantic_colors(path: str | Path) -> tuple[Path, dict[int, tuple[int, int, int]]]:
    source, payload = _load_json(path, "semantic_palette")
    if payload.get("schema_version") != 1:
        raise ValueError("semantic_palette schema_version must be 1")
    background = payload.get("background")
    classes = payload.get("classes")
    if not isinstance(background, Mapping) or not isinstance(classes, list):
        raise TypeError("semantic_palette must define background and classes")
    background_id = background.get("semantic_id", 0)
    if type(background_id) is not int or background_id < 0:
        raise ValueError("semantic_palette background semantic_id is invalid")
    colors = {background_id: _rgb(background.get("rgb"), "background.rgb")}
    for index, item in enumerate(classes, start=1):
        if not isinstance(item, Mapping):
            raise TypeError("semantic_palette classes must contain objects")
        semantic_id = item.get("semantic_id", index)
        if type(semantic_id) is not int or semantic_id < 0:
            raise ValueError(
                "semantic_palette semantic IDs must be nonnegative integers"
            )
        if semantic_id in colors:
            raise ValueError("semantic_palette semantic IDs must be unique")
        colors[semantic_id] = _rgb(item.get("rgb"), f"semantic_id {semantic_id} rgb")
    return source, colors


def _dynamic_states(path: str | Path) -> tuple[Path, dict[int, str]]:
    source, payload = _load_json(path, "dynamic_states")
    rows = payload.get("entity_states")
    if payload.get("schema_version") != 1 or not isinstance(rows, list):
        raise ValueError("dynamic_states must use schema_version 1 and entity_states")
    states: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("dynamic_states entity_states must contain objects")
        entity_id = row.get("entity_id")
        state = row.get("state")
        if type(entity_id) is not int or entity_id <= 0:
            raise ValueError("dynamic_states entity IDs must be positive integers")
        if not isinstance(state, str) or state not in DYNAMIC_STATE_COLORS:
            raise ValueError("dynamic_states contains an unsupported state")
        if entity_id in states:
            raise ValueError("dynamic_states entity IDs must be unique")
        states[entity_id] = state
    return source, states


def _instance_color(entity_id: int) -> tuple[int, int, int]:
    if entity_id == 0:
        return (160, 160, 160)
    digest = hashlib.sha256(f"crove-instance:{entity_id}".encode("ascii")).digest()
    hue = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.94)
    return tuple(round(value * 255.0) for value in (red, green, blue))


def _require_properties(
    names: tuple[str, ...], required: tuple[str, ...], label: str
) -> None:
    missing = sorted(set(required) - set(names))
    if missing:
        raise ValueError(f"PLY must contain {label}: missing {missing}")


def _set_colors(vertices: np.ndarray, colors: np.ndarray) -> None:
    if colors.shape != (len(vertices), 3) or colors.dtype != np.dtype(np.uint8):
        raise ValueError("computed visualization colors are invalid")
    for index, name in enumerate(_RGB_PROPERTIES):
        vertices[name] = colors[:, index]


def _copy_non_rgb_values(vertices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: np.array(vertices[name], copy=True)
        for name in vertices.dtype.names or ()
        if name not in _RGB_PROPERTIES
    }


def _validate_recoloring(
    source: PlyData,
    output: PlyData,
    non_rgb: Mapping[str, np.ndarray],
) -> None:
    if [item.name for item in source.elements] != [
        item.name for item in output.elements
    ]:
        raise RuntimeError("visualization changed the PLY element inventory")
    source_vertices = source["vertex"].data
    output_vertices = output["vertex"].data
    if source_vertices.dtype.names != output_vertices.dtype.names:
        raise RuntimeError("visualization changed the vertex property inventory")
    for name, values in non_rgb.items():
        if not np.array_equal(values, output_vertices[name]):
            raise RuntimeError(f"visualization changed non-RGB vertex property: {name}")
    for element in source.elements:
        if element.name == "vertex":
            continue
        if not _structured_array_equal(element.data, output[element.name].data):
            raise RuntimeError(f"visualization changed PLY element: {element.name}")


def _structured_array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype.names != right.dtype.names:
        return False
    for name in left.dtype.names or ():
        left_values = left[name]
        right_values = right[name]
        if left_values.dtype.kind != "O":
            if not np.array_equal(left_values, right_values):
                return False
            continue
        if any(
            not np.array_equal(np.asarray(left_item), np.asarray(right_item))
            for left_item, right_item in zip(left_values, right_values, strict=True)
        ):
            return False
    return True


def _instance_invariant(vertices: np.ndarray) -> dict[str, Any]:
    entity_ids = np.asarray(vertices["entity_id"], dtype=np.int64)
    colors = np.column_stack(tuple(vertices[name] for name in _RGB_PROPERTIES))
    maximum_unique = 0
    minimum_dominant = 1.0
    for entity_id in np.unique(entity_ids):
        rows = colors[entity_ids == entity_id]
        counts = Counter(tuple(int(value) for value in row) for row in rows)
        maximum_unique = max(maximum_unique, len(counts))
        minimum_dominant = min(minimum_dominant, max(counts.values()) / len(rows))
    return {
        "passed": maximum_unique == 1 and minimum_dominant == 1.0,
        "entity_count": len(np.unique(entity_ids)),
        "maximum_unique_rgb_per_entity": maximum_unique,
        "dominant_rgb_fraction_min": minimum_dominant,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def export_labeled_ply_visualization(
    source: str | Path,
    output: str | Path,
    *,
    mode: VisualizationMode,
    rgb_provenance: str | None = None,
    semantic_palette: str | Path | None = None,
    dynamic_states: str | Path | None = None,
) -> dict[str, Any]:
    """Publish a new recolored PLY directory without changing source semantics."""

    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}")
    source_path = _regular_file(source, "source PLY")
    if source_path.suffix.lower() != ".ply":
        raise ValueError("source PLY path must end in .ply")
    output_path = Path(output).resolve()
    if source_path == output_path or source_path.is_relative_to(output_path):
        raise ValueError("source PLY must remain outside output directory")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_record = _file_record(source_path)
    ply = PlyData.read(source_path, mmap=False)
    try:
        vertices = ply["vertex"].data
    except KeyError as exc:
        raise ValueError("PLY must contain a vertex element") from exc
    names = vertices.dtype.names or ()
    _require_properties(names, _POSITION_PROPERTIES + _RGB_PROPERTIES, "x/y/z and RGB")
    positions = np.column_stack(tuple(vertices[name] for name in _POSITION_PROPERTIES))
    if positions.dtype.kind not in "iuf" or not np.isfinite(positions).all():
        raise ValueError("PLY positions must be finite numeric values")
    non_rgb = _copy_non_rgb_values(vertices)

    palette_record = None
    states_record = None
    if mode == "rgb":
        if rgb_provenance != "raw_tsdf":
            raise ValueError("rgb mode requires rgb_provenance='raw_tsdf'")
    elif mode == "instance":
        _require_properties(names, ("entity_id",), "entity_id")
        entity_ids = np.asarray(vertices["entity_id"])
        if entity_ids.dtype.kind not in "iu" or np.any(entity_ids < 0):
            raise ValueError("entity_id values must be nonnegative integers")
        colors = np.asarray(
            [_instance_color(int(entity_id)) for entity_id in entity_ids],
            dtype=np.uint8,
        )
        _set_colors(vertices, colors)
    elif mode == "semantic":
        if semantic_palette is None:
            raise ValueError("semantic mode requires semantic_palette")
        _require_properties(names, ("semantic_id",), "semantic_id")
        palette_path, palette = _semantic_colors(semantic_palette)
        semantic_ids = np.asarray(vertices["semantic_id"])
        if semantic_ids.dtype.kind not in "iu" or np.any(semantic_ids < 0):
            raise ValueError("semantic_id values must be nonnegative integers")
        missing = sorted({int(value) for value in semantic_ids} - set(palette))
        if missing:
            raise ValueError(
                f"semantic_palette does not cover observed semantic IDs: {missing}"
            )
        colors = np.asarray(
            [palette[int(value)] for value in semantic_ids], dtype=np.uint8
        )
        _set_colors(vertices, colors)
        palette_record = _file_record(palette_path)
    else:
        if dynamic_states is None:
            raise ValueError("dynamic mode requires dynamic_states")
        _require_properties(names, ("entity_id",), "entity_id")
        states_path, states = _dynamic_states(dynamic_states)
        entity_ids = np.asarray(vertices["entity_id"])
        if entity_ids.dtype.kind not in "iu" or np.any(entity_ids < 0):
            raise ValueError("entity_id values must be nonnegative integers")
        observed = {int(value) for value in entity_ids if int(value) > 0}
        missing = sorted(observed - set(states))
        if missing:
            raise ValueError(
                f"dynamic_states does not cover observed entity IDs: {missing}"
            )
        colors = np.asarray(
            [
                DYNAMIC_STATE_COLORS[
                    "unknown" if int(entity_id) == 0 else states[int(entity_id)]
                ]
                for entity_id in entity_ids
            ],
            dtype=np.uint8,
        )
        _set_colors(vertices, colors)
        states_record = _file_record(states_path)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        mesh_path = temporary / "visualization.ply"
        ply.write(mesh_path)
        with mesh_path.open("rb") as handle:
            os.fsync(handle.fileno())
        rendered = PlyData.read(mesh_path, mmap=False)
        _validate_recoloring(ply, rendered, non_rgb)
        invariant = (
            _instance_invariant(rendered["vertex"].data) if mode == "instance" else None
        )
        if invariant is not None and not invariant["passed"]:
            raise RuntimeError("instance palette invariant failed")
        if _file_record(source_path) != source_record:
            raise RuntimeError("source PLY changed during visualization export")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "crove_labeled_ply_visualization",
            "mode": mode,
            "visualization_only": True,
            "recoloring_only": True,
            "benchmark_state_changed": False,
            "source": source_record,
            "output": _file_record(mesh_path, relative_path="visualization.ply"),
            "vertex_count": len(vertices),
            "face_count": len(ply["face"].data) if "face" in ply else 0,
            "vertex_properties": list(names),
            "rgb_provenance": rgb_provenance if mode == "rgb" else None,
            "semantic_palette": palette_record,
            "dynamic_states": states_record,
            "instance_palette_invariant": invariant,
        }
        _write_json(temporary / "manifest.json", manifest)
        os.rename(temporary, output_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return {
        "mode": mode,
        "output": str(output_path),
        "mesh": str(output_path / "visualization.ply"),
        "manifest": str(output_path / "manifest.json"),
    }


__all__ = [
    "DYNAMIC_STATE_COLORS",
    "VisualizationMode",
    "export_labeled_ply_visualization",
]
