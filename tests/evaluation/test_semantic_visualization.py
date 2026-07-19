from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pytest

from src.evaluation.semantic_visualization import (
    export_semantic_ply,
    export_semantic_visualizations,
    load_semantic_palette,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "replica8.json"
    _write_json(
        path,
        {
            "vocabulary": {
                "name": "replica_runtime_semantic_41",
                "classes": ["wall", "floor", "base-cabinet", "cup"],
            },
            "scenes": [{"scene": "room0"}],
        },
    )
    return path


def _batch_manifest(tmp_path: Path, scenes: tuple[str, ...] = ("room0", "room1")) -> Path:
    frozen = json.loads(
        Path("configs/evaluation/manifests/replica8.json").read_text(encoding="utf-8")
    )
    path = tmp_path / "batch-replica8.json"
    _write_json(
        path,
        {
            "vocabulary": frozen["vocabulary"],
            "scenes": [{"scene": scene} for scene in scenes],
        },
    )
    return path


def _batch_tree(tmp_path: Path, scenes: tuple[str, ...] = ("room0", "room1")) -> Path:
    root = tmp_path / "batch"
    for scene in scenes:
        _write_labeled_ply(root / scene / "final" / "oviv2_instance_mesh.ply")
        _write_labeled_ply(root / scene / "evaluation" / "semantic_map_gt.ply")
    return root


def _palette():
    return load_semantic_palette(
        Path("configs/evaluation/replica41_semantic_palette.json"),
        Path("configs/evaluation/manifests/replica8.json"),
    )


def _write_labeled_ply(
    path: Path,
    *,
    semantic_ids: tuple[int, ...] = (0, 1, 2),
    include_semantic_id: bool = True,
    positions: np.ndarray | None = None,
    face: tuple[int, ...] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    if positions is None:
        positions = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
    dtype: list[tuple] = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    if include_semantic_id:
        dtype.append(("semantic_id", "i4"))
    dtype.extend(
        [
            ("entity_id", "i4"),
            ("semantic_confidence", "f4"),
            ("ownership_confidence", "f4"),
        ]
    )
    vertices = np.zeros(len(positions), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = positions.T
    vertices["red"] = 10
    vertices["green"] = 20
    vertices["blue"] = 30
    if include_semantic_id:
        vertices["semantic_id"] = np.asarray(semantic_ids, dtype=np.int32)
    vertices["entity_id"] = np.arange(len(positions), dtype=np.int32)
    vertices["semantic_confidence"] = 0.75
    vertices["ownership_confidence"] = 0.5
    faces = np.empty(1, dtype=[("vertex_indices", "i4", (len(face),))])
    faces["vertex_indices"][0] = np.asarray(face, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex"), PlyElement.describe(faces, "face")],
        text=False,
    ).write(path)
    return positions, np.asarray(face, dtype=np.int32)


def test_palette_covers_manifest_and_reuses_ovimap_colors() -> None:
    palette = load_semantic_palette(
        Path("configs/evaluation/replica41_semantic_palette.json"),
        Path("configs/evaluation/manifests/replica8.json"),
    )

    assert palette.vocabulary_name == "replica_runtime_semantic_41"
    assert palette.colors_by_id[0] == (200, 200, 200)
    assert palette.colors_by_id[1] == (196, 51, 182)
    assert palette.colors_by_id[2] == (188, 189, 34)
    assert len(palette.names_by_id) == 42
    assert palette.names_by_id[0] == "background"
    assert palette.names_by_id[1] == "wall"
    assert palette.names_by_id[34] == "base-cabinet"
    assert palette.names_by_id[36] == "cup"


def test_palette_rejects_missing_or_duplicate_entries(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    palette_path = tmp_path / "palette.json"
    _write_json(
        palette_path,
        {
            "schema_version": 1,
            "vocabulary_name": "replica_runtime_semantic_41",
            "background": {"rgb": [200, 200, 200]},
            "classes": [
                {"name": "wall", "rgb": [1, 2, 3]},
                {"name": "floor", "rgb": [1, 2, 3]},
            ],
        },
    )

    with pytest.raises(ValueError, match="exactly cover manifest classes"):
        load_semantic_palette(palette_path, manifest)


def test_export_semantic_ply_bakes_colors_and_preserves_geometry(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    destination = tmp_path / "semantic.ply"
    source_positions, source_face = _write_labeled_ply(source)

    exported = export_semantic_ply(source, destination, _palette())

    output = PlyData.read(destination)
    assert output["vertex"].data.dtype.names == (
        "x",
        "y",
        "z",
        "red",
        "green",
        "blue",
    )
    np.testing.assert_array_equal(
        np.column_stack(
            (output["vertex"]["red"], output["vertex"]["green"], output["vertex"]["blue"])
        ),
        [[200, 200, 200], [196, 51, 182], [188, 189, 34]],
    )
    np.testing.assert_array_equal(
        np.column_stack((output["vertex"]["x"], output["vertex"]["y"], output["vertex"]["z"])),
        source_positions,
    )
    np.testing.assert_array_equal(output["face"]["vertex_indices"][0], source_face)
    assert exported.vertex_count == 3
    assert exported.face_count == 1
    assert exported.semantic_counts == {0: 1, 1: 1, 2: 1}


def test_export_semantic_ply_rejects_missing_semantic_id(tmp_path: Path) -> None:
    source = tmp_path / "missing.ply"
    _write_labeled_ply(source, include_semantic_id=False)

    with pytest.raises(ValueError, match="must contain x/y/z and semantic_id"):
        export_semantic_ply(source, tmp_path / "out.ply", _palette())


def test_export_semantic_ply_rejects_out_of_vocabulary_id(tmp_path: Path) -> None:
    source = tmp_path / "bad-semantic.ply"
    _write_labeled_ply(source, semantic_ids=(0, 1, 42))

    with pytest.raises(ValueError, match="outside frozen palette"):
        export_semantic_ply(source, tmp_path / "out.ply", _palette())


def test_export_semantic_ply_rejects_nonfinite_positions(tmp_path: Path) -> None:
    source = tmp_path / "nonfinite.ply"
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    _write_labeled_ply(source, positions=positions)

    with pytest.raises(ValueError, match="positions must be finite"):
        export_semantic_ply(source, tmp_path / "out.ply", _palette())


def test_export_semantic_ply_preserves_quad_faces(tmp_path: Path) -> None:
    source = tmp_path / "quad.ply"
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    _write_labeled_ply(
        source,
        semantic_ids=(0, 1, 2, 3),
        positions=positions,
        face=(0, 1, 2, 3),
    )

    export_semantic_ply(source, tmp_path / "out.ply", _palette())

    output = PlyData.read(tmp_path / "out.ply")
    np.testing.assert_array_equal(output["face"]["vertex_indices"][0], [0, 1, 2, 3])


@pytest.mark.parametrize("face", [(0, 1, 3), (0, 1)])
def test_export_semantic_ply_rejects_invalid_faces(tmp_path: Path, face: tuple[int, ...]) -> None:
    source = tmp_path / "bad-face.ply"
    _write_labeled_ply(source, face=face)

    with pytest.raises(ValueError, match="at least three|outside vertex array"):
        export_semantic_ply(source, tmp_path / "out.ply", _palette())


def test_export_semantic_ply_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    _write_labeled_ply(source)

    first = export_semantic_ply(source, tmp_path / "first.ply", _palette())
    second = export_semantic_ply(source, tmp_path / "second.ply", _palette())

    assert first.output_sha256 == second.output_sha256
    assert (tmp_path / "first.ply").read_bytes() == (tmp_path / "second.ply").read_bytes()


def test_batch_export_publishes_both_meshes_legend_and_provenance(tmp_path: Path) -> None:
    batch_root = _batch_tree(tmp_path)
    manifest_path = _batch_manifest(tmp_path)
    output = tmp_path / "paper_visualizations"

    result = export_semantic_visualizations(
        batch_root,
        manifest_path,
        Path("configs/evaluation/replica41_semantic_palette.json"),
        output,
    )

    assert result == {
        "output": str(output.resolve()),
        "scene_count": 2,
        "scenes": ["room0", "room1"],
    }
    assert (output / "semantic_palette.json").is_file()
    legend = Image.open(output / "semantic_legend.png")
    assert legend.width > 0 and legend.height > 0
    for scene in ("room0", "room1"):
        native = output / scene / "oviv2_semantic_native.ply"
        aligned = output / scene / "oviv2_semantic_gt_aligned.ply"
        assert native.is_file()
        assert aligned.is_file()
        manifest = json.loads(
            (output / scene / "export_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["scene"] == scene
        assert manifest["native"]["vertex_count"] == 3
        assert manifest["gt_aligned"]["semantic_counts"] == {"0": 1, "1": 1, "2": 1}
        assert ".tmp" not in json.dumps(manifest)


def test_batch_export_is_atomic_when_later_scene_is_invalid(tmp_path: Path) -> None:
    batch_root = _batch_tree(tmp_path)
    manifest_path = _batch_manifest(tmp_path)
    output = tmp_path / "paper_visualizations"
    _write_labeled_ply(
        batch_root / "room1" / "evaluation" / "semantic_map_gt.ply",
        include_semantic_id=False,
    )

    with pytest.raises(ValueError, match="semantic_id"):
        export_semantic_visualizations(
            batch_root,
            manifest_path,
            Path("configs/evaluation/replica41_semantic_palette.json"),
            output,
        )

    assert not output.exists()


def test_batch_export_selects_requested_manifest_scene(tmp_path: Path) -> None:
    batch_root = _batch_tree(tmp_path)
    output = tmp_path / "selected"

    result = export_semantic_visualizations(
        batch_root,
        _batch_manifest(tmp_path),
        Path("configs/evaluation/replica41_semantic_palette.json"),
        output,
        scenes=["room1"],
    )

    assert result["scenes"] == ["room1"]
    assert (output / "room1").is_dir()
    assert not (output / "room0").exists()


def test_batch_export_rejects_scene_outside_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside frozen manifest"):
        export_semantic_visualizations(
            _batch_tree(tmp_path),
            _batch_manifest(tmp_path),
            Path("configs/evaluation/replica41_semantic_palette.json"),
            tmp_path / "output",
            scenes=["office4"],
        )
