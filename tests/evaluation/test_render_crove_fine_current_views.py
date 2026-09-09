from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scripts.evaluation import render_crove_fine_current_views as renderer


def _write_view(path: Path, color: tuple[int, int, int]) -> None:
    vertices = np.empty(
        5,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    points = [(-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0), (0, 0, 1)]
    for index, point in enumerate(points):
        vertices[index] = (*point, *color)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def test_four_view_renderer_checks_shared_geometry_and_writes_real_pixels(
    tmp_path: Path,
) -> None:
    current_map = tmp_path / "current_map"
    current_map.mkdir()
    colors = {
        "rgb": (10, 20, 30),
        "instance": (40, 50, 60),
        "semantic": (70, 80, 90),
        "state": (100, 110, 120),
    }
    for mode, color in colors.items():
        _write_view(current_map / f"current_{mode}.ply", color)

    output = tmp_path / "four_views.png"
    receipt = renderer.render_four_views(
        current_map,
        output,
        scene="fixture",
        width=160,
        height=120,
        maximum_points=5,
        azimuth_degrees=-55.0,
        elevation_degrees=22.0,
    )

    assert output.is_file()
    assert receipt["geometry_shared_across_views"] is True
    assert receipt["source_vertex_count"] == 5
    assert receipt["sampled_vertex_count"] == 5
    pixels = np.asarray(Image.open(output).convert("RGB"))
    assert np.any(pixels != 255)


def test_legacy_renderer_keeps_three_original_color_views_display_only(
    tmp_path: Path,
) -> None:
    paths = {}
    for mode, color in {
        "rgb": (10, 20, 30),
        "instance": (40, 50, 60),
        "semantic": (70, 80, 90),
    }.items():
        paths[mode] = tmp_path / f"{mode}.ply"
        _write_view(paths[mode], color)

    output = tmp_path / "legacy_views.png"
    receipt = renderer.render_legacy_views(
        paths,
        output,
        scene="fixture",
        width=120,
        height=100,
        maximum_points=5,
    )

    assert output.is_file()
    assert receipt["display_only"] is True
    assert receipt["geometry_shared_across_views"] is True
    assert set(receipt["sources"]) == {"rgb", "instance", "semantic"}
    assert np.any(np.asarray(Image.open(output).convert("RGB")) != 255)
