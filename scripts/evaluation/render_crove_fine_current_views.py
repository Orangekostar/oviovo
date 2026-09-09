#!/usr/bin/env python3
"""Render a fixed-camera audit plate from the four exported current PLY views."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

_MODES = ("rgb", "instance", "semantic", "state")
_TITLES = {
    "rgb": "(a) Observed RGB",
    "instance": "(b) Instance owner",
    "semantic": "(c) Semantic label",
    "state": "(d) Current evidence state",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_view(path: Path, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    ply = PlyData.read(path, mmap="r")
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    vertices = ply["vertex"]
    names = set(vertices.data.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue"}
    if not required <= names:
        raise ValueError(f"PLY lacks render fields: {sorted(required - names)}")
    count = len(vertices)
    if rows.size and int(rows[-1]) >= count:
        raise ValueError("four current views have different vertex counts")
    points = np.column_stack(
        (vertices["x"][rows], vertices["y"][rows], vertices["z"][rows])
    ).astype(np.float32, copy=False)
    colors = np.column_stack(
        (vertices["red"][rows], vertices["green"][rows], vertices["blue"][rows])
    ).astype(np.uint8, copy=False)
    return points, colors, count


def _camera_basis(azimuth_degrees: float, elevation_degrees: float) -> np.ndarray:
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    forward = np.asarray(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return np.stack((right, up, forward), axis=1)


def _rasterize(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    width: int,
    height: int,
    basis: np.ndarray,
) -> Image.Image:
    if not len(points):
        raise ValueError("cannot render an empty current surface")
    camera = (points - np.median(points, axis=0)) @ basis
    lower = np.quantile(camera[:, :2], 0.005, axis=0)
    upper = np.quantile(camera[:, :2], 0.995, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    scale = min((width - 10) / span[0], (height - 10) / span[1])
    projected = (camera[:, :2] - (lower + upper) / 2.0) * scale
    x = np.rint(projected[:, 0] + width / 2.0).astype(np.int64)
    y = np.rint(height / 2.0 - projected[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]
    depth = camera[valid, 2]
    visible_colors = colors[valid]
    flat = y * width + x
    order = np.lexsort((depth, flat))
    sorted_flat = flat[order]
    last = np.concatenate((sorted_flat[1:] != sorted_flat[:-1], [True]))
    chosen = order[last]

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    chosen_x = x[chosen]
    chosen_y = y[chosen]
    chosen_colors = visible_colors[chosen]
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        px = chosen_x + dx
        py = chosen_y + dy
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        canvas[py[inside], px[inside]] = chosen_colors[inside]
    return Image.fromarray(canvas, mode="RGB")


def render_four_views(
    current_map: Path,
    output: Path,
    *,
    scene: str,
    width: int = 900,
    height: int = 650,
    maximum_points: int = 800_000,
    azimuth_degrees: float = -55.0,
    elevation_degrees: float = 22.0,
) -> dict[str, object]:
    if width < 100 or height < 100 or maximum_points < 1:
        raise ValueError("render dimensions and maximum_points are invalid")
    paths = {mode: current_map / f"current_{mode}.ply" for mode in _MODES}
    first = PlyData.read(paths["rgb"], mmap="r")
    source_vertex_count = len(first["vertex"])
    if source_vertex_count < 1:
        raise ValueError("cannot render an empty current surface")
    sampled_count = min(source_vertex_count, maximum_points)
    rows = np.unique(
        np.linspace(0, source_vertex_count - 1, sampled_count, dtype=np.int64)
    )
    basis = _camera_basis(azimuth_degrees, elevation_degrees)
    panels: dict[str, Image.Image] = {}
    reference_points: np.ndarray | None = None
    bindings: dict[str, dict[str, object]] = {}
    for mode in _MODES:
        points, colors, count = _sample_view(paths[mode], rows)
        if count != source_vertex_count:
            raise ValueError("four current views have different vertex counts")
        if reference_points is None:
            reference_points = points
        elif not np.array_equal(points, reference_points):
            raise ValueError("four current views do not share sampled geometry")
        panels[mode] = _rasterize(
            points, colors, width=width, height=height, basis=basis
        )
        bindings[mode] = {
            "path": str(paths[mode]),
            "byte_count": paths[mode].stat().st_size,
            "sha256": _sha256(paths[mode]),
        }

    margin = 28
    title_height = 36
    gap = 18
    plate = Image.new(
        "RGB",
        (2 * width + 2 * margin + gap, 2 * (height + title_height) + 2 * margin + gap),
        "white",
    )
    draw = ImageDraw.Draw(plate)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 24)
    footer_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    for index, mode in enumerate(_MODES):
        column = index % 2
        row = index // 2
        left = margin + column * (width + gap)
        top = margin + row * (height + title_height + gap)
        draw.text(
            (left, top), _TITLES[mode], fill=(24, 24, 24), font=title_font
        )
        plate.paste(panels[mode], (left, top + title_height))
    draw.text(
        (margin, plate.height - 18),
        f"{scene} | shared canonical current geometry | orthographic audit render",
        fill=(70, 70, 70),
        font=footer_font,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".png", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        plate.save(temporary, format="PNG", optimize=True)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "scene": scene,
        "geometry_shared_across_views": True,
        "source_vertex_count": source_vertex_count,
        "sampled_vertex_count": len(rows),
        "sampling_rule": "equal_index_spacing_over_canonical_vertex_order",
        "camera": {
            "projection": "orthographic",
            "azimuth_degrees": azimuth_degrees,
            "elevation_degrees": elevation_degrees,
        },
        "sources": bindings,
        "output": {
            "path": str(output),
            "byte_count": output.stat().st_size,
            "sha256": _sha256(output),
        },
    }
    receipt_path = output.with_suffix(".json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def render_legacy_views(
    paths: Mapping[str, Path],
    output: Path,
    *,
    scene: str,
    width: int = 600,
    height: int = 450,
    maximum_points: int = 200_000,
    azimuth_degrees: float = -55.0,
    elevation_degrees: float = 22.0,
) -> dict[str, object]:
    """Render the three original legacy color modes without changing predictions."""

    modes = ("rgb", "instance", "semantic")
    if set(paths) != set(modes):
        raise ValueError("legacy paths must contain rgb, instance, and semantic")
    if width < 100 or height < 100 or maximum_points < 1:
        raise ValueError("render dimensions and maximum_points are invalid")
    first = PlyData.read(paths["rgb"], mmap="r")
    source_vertex_count = len(first["vertex"])
    if source_vertex_count < 1:
        raise ValueError("cannot render an empty legacy surface")
    sampled_count = min(source_vertex_count, maximum_points)
    rows = np.unique(
        np.linspace(0, source_vertex_count - 1, sampled_count, dtype=np.int64)
    )
    basis = _camera_basis(azimuth_degrees, elevation_degrees)
    panels: dict[str, Image.Image] = {}
    reference_points: np.ndarray | None = None
    bindings: dict[str, dict[str, object]] = {}
    for mode in modes:
        path = paths[mode]
        points, colors, count = _sample_view(path, rows)
        if count != source_vertex_count:
            raise ValueError("legacy views have different vertex counts")
        if reference_points is None:
            reference_points = points
        elif not np.array_equal(points, reference_points):
            raise ValueError("legacy views do not share sampled geometry")
        panels[mode] = _rasterize(
            points, colors, width=width, height=height, basis=basis
        )
        vertex_data = PlyData.read(path, mmap="r")["vertex"].data
        names = set(vertex_data.dtype.names or ())
        full_colors = np.column_stack(
            (vertex_data["red"], vertex_data["green"], vertex_data["blue"])
        )
        binding: dict[str, object] = {
            "path": str(path),
            "byte_count": path.stat().st_size,
            "sha256": _sha256(path),
            "unique_standard_rgb_count": len(np.unique(full_colors, axis=0)),
        }
        for field in ("semantic_id", "entity_id"):
            if field in names:
                binding[f"unique_{field}_count"] = len(np.unique(vertex_data[field]))
        bindings[mode] = binding

    margin = 28
    title_height = 36
    gap = 18
    plate = Image.new(
        "RGB",
        (3 * width + 2 * margin + 2 * gap, height + title_height + 2 * margin),
        "white",
    )
    draw = ImageDraw.Draw(plate)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 24)
    footer_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    for index, mode in enumerate(modes):
        left = margin + index * (width + gap)
        draw.text(
            (left, margin), _TITLES[mode], fill=(24, 24, 24), font=title_font
        )
        plate.paste(panels[mode], (left, margin + title_height))
    draw.text(
        (margin, plate.height - 18),
        f"{scene} | legacy display-only control | unchanged source geometry",
        fill=(70, 70, 70),
        font=footer_font,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".png", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        plate.save(temporary, format="PNG", optimize=True)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "scene": scene,
        "display_only": True,
        "geometry_shared_across_views": True,
        "source_vertex_count": source_vertex_count,
        "sampled_vertex_count": len(rows),
        "sampling_rule": "equal_index_spacing_over_legacy_vertex_order",
        "camera": {
            "projection": "orthographic",
            "azimuth_degrees": azimuth_degrees,
            "elevation_degrees": elevation_degrees,
        },
        "sources": bindings,
        "output": {
            "path": str(output),
            "byte_count": output.stat().st_size,
            "sha256": _sha256(output),
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--maximum-points", type=int, default=800_000)
    parser.add_argument("--azimuth-degrees", type=float, default=-55.0)
    parser.add_argument("--elevation-degrees", type=float, default=22.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    render_four_views(
        args.current_map,
        args.output,
        scene=args.scene,
        maximum_points=args.maximum_points,
        azimuth_degrees=args.azimuth_degrees,
        elevation_degrees=args.elevation_degrees,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
