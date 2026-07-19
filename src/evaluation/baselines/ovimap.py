"""OVI-MAP mesh loading and official zero-shot matching helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import numpy as np


_INSTANCE_COLOR_RE = re.compile(
    r"Instance:\s*(?P<instance_id>\d+)\s+Color:\s*\((?P<red>\d+),(?P<green>\d+),(?P<blue>\d+)\)"
)


def parse_instance_color_log(path: str | Path) -> dict[int, tuple[int, int, int]]:
    """Read OVI-MAP's authoritative global instance-to-PLY-color log."""
    colors: dict[int, tuple[int, int, int]] = {}
    instances_by_color: dict[tuple[int, int, int], int] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = _INSTANCE_COLOR_RE.search(line)
        if match is not None:
            instance_id = int(match.group("instance_id"))
            color = tuple(
                int(match.group(channel)) for channel in ("red", "green", "blue")
            )
            existing_color = colors.get(instance_id)
            if existing_color is not None and existing_color != color:
                raise ValueError(
                    f"OVI-MAP instance {instance_id} has conflicting colors: "
                    f"{existing_color} and {color}"
                )
            existing_instance = instances_by_color.get(color)
            if existing_instance is not None and existing_instance != instance_id:
                raise ValueError(
                    f"OVI-MAP mesh color {color} is reused by instances "
                    f"{existing_instance} and {instance_id}"
                )
            colors[instance_id] = color
            instances_by_color[color] = instance_id
    if not colors:
        raise ValueError(f"OVI-MAP instance color log contains no Instance: ... Color entries: {path}")
    return colors


def remap_instance_colors(
    instances: dict[int, dict],
    colors_by_instance: dict[int, tuple[int, int, int]],
) -> dict[int, dict]:
    """Copy method records while replacing input-mask colors with PLY colors."""
    remapped: dict[int, dict] = {}
    for instance_id, instance in instances.items():
        record = dict(instance)
        if int(instance_id) in colors_by_instance:
            record["color"] = colors_by_instance[int(instance_id)]
        remapped[int(instance_id)] = record
    return remapped


def bind_mesh_instances(
    semantic_instances: Mapping[int, Mapping[str, Any]],
    colors_by_instance: Mapping[int, tuple[int, int, int]],
    points_by_color: Mapping[tuple[int, int, int], np.ndarray],
) -> dict[int, dict[str, Any]]:
    """Bind every mesh-backed global instance to optional semantic features."""
    bound: dict[int, dict[str, Any]] = {}
    for instance_id, color in sorted(colors_by_instance.items()):
        normalized_color = tuple(int(value) for value in color)
        if normalized_color not in points_by_color:
            continue
        record = dict(semantic_instances.get(int(instance_id), {}))
        record["color"] = normalized_color
        bound[int(instance_id)] = record
    return bound


def _color_codes(colors: np.ndarray) -> np.ndarray:
    values = np.asarray(colors)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("OVI-MAP mesh colors must have shape (N, 3)")
    if np.issubdtype(values.dtype, np.floating) and values.size and float(values.max()) <= 1.0:
        values = np.rint(values * 255.0)
    values = np.clip(np.rint(values), 0, 255).astype(np.uint32)
    return (values[:, 0] << 16) | (values[:, 1] << 8) | values[:, 2]


def group_points_by_color(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    requested_colors: Iterable[tuple[int, int, int]],
) -> dict[tuple[int, int, int], np.ndarray]:
    """Group mesh vertices by requested RGB colors with one shared sort."""
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("OVI-MAP mesh points must have shape (N, 3)")
    codes = _color_codes(colors_rgb)
    if len(points) != len(codes):
        raise ValueError("OVI-MAP mesh points and colors must have equal length")
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    grouped: dict[tuple[int, int, int], np.ndarray] = {}
    for color in sorted(set(tuple(int(value) for value in item) for item in requested_colors)):
        if len(color) != 3 or any(value < 0 or value > 255 for value in color):
            raise ValueError("OVI-MAP requested colors must be RGB uint8 triples")
        code = (color[0] << 16) | (color[1] << 8) | color[2]
        left = int(np.searchsorted(sorted_codes, code, side="left"))
        right = int(np.searchsorted(sorted_codes, code, side="right"))
        if left != right:
            grouped[color] = points[order[left:right]]
    return grouped


def load_instance_mesh(
    path: str | Path,
    requested_colors: Iterable[tuple[int, int, int]],
) -> tuple[dict[tuple[int, int, int], np.ndarray], np.ndarray]:
    from plyfile import PlyData

    vertices = PlyData.read(path)["vertex"]
    names = set(vertices.data.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue"}
    if not required.issubset(names):
        raise ValueError(f"OVI-MAP instance mesh lacks properties: {sorted(required - names)}")
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"]))
    requested = set(tuple(int(value) for value in color) for color in requested_colors)
    grouped = group_points_by_color(points, colors, requested)
    requested_codes = np.asarray(
        [(red << 16) | (green << 8) | blue for red, green, blue in requested],
        dtype=np.uint32,
    )
    background = points[~np.isin(_color_codes(colors), requested_codes)]
    return grouped, np.asarray(background, dtype=np.float32)


def _normalized_rows(values: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(f"{name} contains a zero vector")
    return rows / norms


def relative_similarity_labels(
    entity_features: np.ndarray,
    text_features: np.ndarray,
    canonical_features: np.ndarray,
    class_names: Sequence[str],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Apply OVI-MAP's minimum canonical-relative SigLIP score."""
    entities = _normalized_rows(entity_features, "entity_features")
    texts = _normalized_rows(text_features, "text_features")
    canonical = _normalized_rows(canonical_features, "canonical_features")
    if entities.shape[1] != texts.shape[1] or entities.shape[1] != canonical.shape[1]:
        raise ValueError("OVI-MAP semantic feature dimensions must match")
    names = tuple(str(name) for name in class_names)
    if len(names) != len(texts):
        raise ValueError("OVI-MAP class names must match text feature count")

    query_similarity = entities @ texts.T
    canonical_similarity = entities @ canonical.T
    relative = 1.0 / (
        1.0 + np.exp(canonical_similarity[:, None, :] - query_similarity[:, :, None])
    )
    scores = relative.min(axis=2)
    matches = np.argmax(scores, axis=1)
    return (
        tuple(names[int(index)] for index in matches),
        tuple(float(scores[row, index]) for row, index in enumerate(matches)),
    )
