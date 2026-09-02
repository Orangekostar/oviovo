from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from src.evaluation.crove_map_visualization import (
    DYNAMIC_STATE_COLORS,
    export_labeled_ply_visualization,
)


def _write_source(path: Path, *, include_ids: bool = True) -> Path:
    dtype: list[tuple[str, str]] = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    if include_ids:
        dtype.extend((("semantic_id", "i4"), ("entity_id", "i4")))
    dtype.extend((("semantic_confidence", "f4"), ("ownership_confidence", "f4")))
    vertices = np.zeros(5, dtype=dtype)
    vertices["x"] = [0.0, 1.0, 0.0, 1.0, 2.0]
    vertices["y"] = [0.0, 0.0, 1.0, 1.0, 2.0]
    vertices["z"] = [0.0, 0.0, 0.0, 0.0, 1.0]
    vertices["red"] = [1, 2, 3, 4, 5]
    vertices["green"] = [6, 7, 8, 9, 10]
    vertices["blue"] = [11, 12, 13, 14, 15]
    if include_ids:
        vertices["semantic_id"] = [1, 1, 2, 2, 0]
        vertices["entity_id"] = [7, 7, 9, 9, 0]
    vertices["semantic_confidence"] = [0.9, 0.8, 0.7, 0.6, 0.0]
    vertices["ownership_confidence"] = [0.8, 0.7, 0.6, 0.5, 0.0]
    faces = np.empty(2, dtype=[("vertex_indices", "i4", (3,))])
    faces["vertex_indices"] = [[0, 1, 2], [1, 2, 3]]
    edges = np.zeros(1, dtype=[("vertex1", "i4"), ("vertex2", "i4")])
    edges["vertex1"] = 2
    edges["vertex2"] = 4
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [
            PlyElement.describe(vertices, "vertex"),
            PlyElement.describe(faces, "face"),
            PlyElement.describe(edges, "edge"),
        ],
        text=False,
        byte_order="<",
        comments=["CROVE test source"],
    ).write(path)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _palette(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": 1,
            "background": {"semantic_id": 0, "rgb": [20, 21, 22]},
            "classes": [
                {"semantic_id": 1, "name": "chair", "rgb": [30, 31, 32]},
                {"semantic_id": 2, "name": "table", "rgb": [40, 41, 42]},
            ],
        },
    )


def _states(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": 1,
            "entity_states": [
                {"entity_id": 7, "state": "dynamic"},
                {"entity_id": 9, "state": "uncertain"},
            ],
        },
    )


def _rgb(data: np.ndarray) -> np.ndarray:
    return np.column_stack((data["red"], data["green"], data["blue"]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_recolor_only(source: Path, output: Path) -> None:
    before = PlyData.read(source)
    after = PlyData.read(output)
    assert [item.name for item in before.elements] == [
        item.name for item in after.elements
    ]
    before_vertices = before["vertex"].data
    after_vertices = after["vertex"].data
    assert before_vertices.dtype.names == after_vertices.dtype.names
    for name in before_vertices.dtype.names or ():
        if name not in {"red", "green", "blue"}:
            np.testing.assert_array_equal(before_vertices[name], after_vertices[name])
    for before_face, after_face in zip(
        before["face"]["vertex_indices"],
        after["face"]["vertex_indices"],
        strict=True,
    ):
        np.testing.assert_array_equal(before_face, after_face)
    np.testing.assert_array_equal(before["edge"].data, after["edge"].data)


def test_rgb_mode_requires_and_preserves_raw_tsdf_colors(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.ply")
    source_hash = _sha256(source)

    result = export_labeled_ply_visualization(
        source,
        tmp_path / "rgb",
        mode="rgb",
        rgb_provenance="raw_tsdf",
    )

    output = Path(result["mesh"])
    np.testing.assert_array_equal(
        _rgb(PlyData.read(output)["vertex"].data),
        _rgb(PlyData.read(source)["vertex"].data),
    )
    _assert_recolor_only(source, output)
    assert _sha256(source) == source_hash
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["mode"] == "rgb"
    assert manifest["rgb_provenance"] == "raw_tsdf"
    assert manifest["recoloring_only"] is True


def test_rgb_mode_rejects_unproven_instance_palette(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.ply")

    with pytest.raises(ValueError, match="raw_tsdf"):
        export_labeled_ply_visualization(source, tmp_path / "out", mode="rgb")

    assert not (tmp_path / "out").exists()


def test_instance_mode_is_uniform_stable_and_preserves_ids(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.ply")

    first = export_labeled_ply_visualization(
        source, tmp_path / "first", mode="instance"
    )
    second = export_labeled_ply_visualization(
        source, tmp_path / "second", mode="instance"
    )

    first_mesh = Path(first["mesh"])
    second_mesh = Path(second["mesh"])
    vertices = PlyData.read(first_mesh)["vertex"].data
    assert _sha256(first_mesh) == _sha256(second_mesh)
    for entity_id in (0, 7, 9):
        assert (
            len(np.unique(_rgb(vertices)[vertices["entity_id"] == entity_id], axis=0))
            == 1
        )
    assert not np.array_equal(
        _rgb(vertices)[vertices["entity_id"] == 7][0],
        _rgb(vertices)[vertices["entity_id"] == 9][0],
    )
    _assert_recolor_only(source, first_mesh)
    manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
    assert manifest["instance_palette_invariant"] == {
        "dominant_rgb_fraction_min": 1.0,
        "entity_count": 3,
        "maximum_unique_rgb_per_entity": 1,
        "passed": True,
    }


def test_semantic_mode_uses_bound_palette(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.ply")
    palette = _palette(tmp_path / "palette.json")

    result = export_labeled_ply_visualization(
        source,
        tmp_path / "semantic",
        mode="semantic",
        semantic_palette=palette,
    )

    vertices = PlyData.read(Path(result["mesh"]))["vertex"].data
    colors = _rgb(vertices)
    np.testing.assert_array_equal(colors[vertices["semantic_id"] == 0], [[20, 21, 22]])
    np.testing.assert_array_equal(
        colors[vertices["semantic_id"] == 1], [[30, 31, 32], [30, 31, 32]]
    )
    np.testing.assert_array_equal(
        colors[vertices["semantic_id"] == 2], [[40, 41, 42], [40, 41, 42]]
    )
    _assert_recolor_only(source, Path(result["mesh"]))
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["semantic_palette"]["sha256"] == _sha256(palette)


def test_dynamic_mode_uses_fixed_alias_palette(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.ply")
    states = _states(tmp_path / "states.json")

    result = export_labeled_ply_visualization(
        source,
        tmp_path / "dynamic",
        mode="dynamic",
        dynamic_states=states,
    )

    vertices = PlyData.read(Path(result["mesh"]))["vertex"].data
    colors = _rgb(vertices)
    np.testing.assert_array_equal(
        colors[vertices["entity_id"] == 7],
        np.tile(DYNAMIC_STATE_COLORS["dynamic"], (2, 1)),
    )
    np.testing.assert_array_equal(
        colors[vertices["entity_id"] == 9],
        np.tile(DYNAMIC_STATE_COLORS["unknown"], (2, 1)),
    )
    np.testing.assert_array_equal(
        colors[vertices["entity_id"] == 0],
        np.asarray([DYNAMIC_STATE_COLORS["unknown"]]),
    )
    _assert_recolor_only(source, Path(result["mesh"]))


@pytest.mark.parametrize(
    ("mode", "keyword", "message"),
    [
        ("semantic", {}, "semantic_palette"),
        ("dynamic", {}, "dynamic_states"),
        ("instance", {}, "entity_id"),
    ],
)
def test_modes_fail_closed_when_required_evidence_is_missing(
    tmp_path: Path, mode: str, keyword: dict, message: str
) -> None:
    source = _write_source(
        tmp_path / "source.ply", include_ids=mode not in {"instance"}
    )

    with pytest.raises(ValueError, match=message):
        export_labeled_ply_visualization(source, tmp_path / "out", mode=mode, **keyword)


def test_palette_and_state_sidecars_must_cover_observed_ids(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.ply")
    palette = _palette(tmp_path / "palette.json")
    payload = json.loads(palette.read_text(encoding="utf-8"))
    payload["classes"].pop()
    _write_json(palette, payload)

    with pytest.raises(ValueError, match="semantic IDs"):
        export_labeled_ply_visualization(
            source,
            tmp_path / "semantic",
            mode="semantic",
            semantic_palette=palette,
        )

    states = _write_json(
        tmp_path / "states.json",
        {"schema_version": 1, "entity_states": [{"entity_id": 7, "state": "static"}]},
    )
    with pytest.raises(ValueError, match="entity IDs"):
        export_labeled_ply_visualization(
            source,
            tmp_path / "dynamic",
            mode="dynamic",
            dynamic_states=states,
        )


def test_rejects_existing_output_and_source_inside_output(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.ply")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        export_labeled_ply_visualization(source, existing, mode="instance")

    nested_source = _write_source(tmp_path / "future" / "source.ply")
    with pytest.raises(ValueError, match="outside output"):
        export_labeled_ply_visualization(
            nested_source, tmp_path / "future", mode="instance"
        )


def test_rejects_symlinked_source_and_sidecars(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.ply")
    source_link = tmp_path / "source-link.ply"
    source_link.symlink_to(source)
    with pytest.raises(ValueError, match="regular file"):
        export_labeled_ply_visualization(
            source_link, tmp_path / "source-output", mode="instance"
        )

    palette = _palette(tmp_path / "palette.json")
    palette_link = tmp_path / "palette-link.json"
    palette_link.symlink_to(palette)
    with pytest.raises(ValueError, match="regular file"):
        export_labeled_ply_visualization(
            source,
            tmp_path / "palette-output",
            mode="semantic",
            semantic_palette=palette_link,
        )


def test_output_is_deterministic_and_manifest_binds_mesh(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.ply")
    first = export_labeled_ply_visualization(source, tmp_path / "a", mode="instance")
    second = export_labeled_ply_visualization(source, tmp_path / "b", mode="instance")

    assert Path(first["mesh"]).read_bytes() == Path(second["mesh"]).read_bytes()
    assert Path(first["manifest"]).read_bytes() == Path(second["manifest"]).read_bytes()
    manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
    assert manifest["source"]["sha256"] == _sha256(source)
    assert manifest["output"]["sha256"] == _sha256(Path(first["mesh"]))


def test_cli_help_and_instance_export(tmp_path: Path) -> None:
    script = Path("scripts/evaluation/export_crove_map_visualization.py")
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--mode" in help_result.stdout

    source = _write_source(tmp_path / "source.ply")
    output = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source",
            str(source),
            "--output",
            str(output),
            "--mode",
            "instance",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["mode"] == "instance"
    assert (output / "visualization.ply").is_file()
    assert (output / "manifest.json").is_file()
