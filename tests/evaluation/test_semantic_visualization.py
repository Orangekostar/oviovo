from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.semantic_visualization import load_semantic_palette


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
