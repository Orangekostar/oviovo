# OVIV2 Semantic Visualization Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export deterministic, viewer-compatible native and GT-aligned semantic PLY files for all verified OVIV2 Replica scenes using an OVI-MAP-aligned class palette.

**Architecture:** A repository-owned JSON file freezes the class-name palette independently of the external OVI-MAP checkout. A focused evaluation utility validates source PLY fields, replaces vertex RGB from `semantic_id`, writes presentation-only PLY files and a legend, and atomically publishes batch outputs with hashes. A thin CLI resolves the frozen manifest and selects all or requested scenes; it never mutates verified run artifacts.

**Tech Stack:** Python 3.10, NumPy, `plyfile`, Pillow, pytest, JSON/SHA256 provenance.

---

## File Structure

- Create `configs/evaluation/replica41_semantic_palette.json`: frozen class-name RGB contract and OVI-MAP source metadata.
- Create `src/evaluation/semantic_visualization.py`: palette validation, semantic PLY recoloring, legend generation, atomic batch export, and provenance manifests.
- Create `scripts/evaluation/export_oviv2_semantic_visualizations.py`: direct-execution-safe CLI only.
- Create `tests/evaluation/test_semantic_visualization.py`: palette, PLY fidelity, validation, determinism, and batch tests.
- Create `tests/evaluation/test_export_oviv2_semantic_visualizations_cli.py`: CLI help and scene-selection tests.

### Task 1: Freeze And Validate The Replica-41 Semantic Palette

**Files:**
- Create: `configs/evaluation/replica41_semantic_palette.json`
- Create: `src/evaluation/semantic_visualization.py`
- Create: `tests/evaluation/test_semantic_visualization.py`

- [ ] **Step 1: Write the failing palette contract tests**

Create `tests/evaluation/test_semantic_visualization.py` with a manifest fixture and these tests:

```python
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


def test_palette_covers_manifest_and_reuses_ovimap_colors(tmp_path: Path) -> None:
    palette = load_semantic_palette(
        Path("configs/evaluation/replica41_semantic_palette.json"),
        _manifest(tmp_path),
    )

    assert palette.vocabulary_name == "replica_runtime_semantic_41"
    assert palette.colors_by_id[0] == (200, 200, 200)
    assert palette.colors_by_id[1] == (196, 51, 182)  # OVI-MAP wall
    assert palette.colors_by_id[2] == (188, 189, 34)   # OVI-MAP floor
    assert palette.names_by_id == {
        0: "background",
        1: "wall",
        2: "floor",
        3: "base-cabinet",
        4: "cup",
    }


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
```

- [ ] **Step 2: Run the palette tests and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_semantic_visualization.py -v
```

Expected: collection fails with `ModuleNotFoundError: src.evaluation.semantic_visualization`.

- [ ] **Step 3: Add the complete palette JSON**

Create `configs/evaluation/replica41_semantic_palette.json` with:

```json
{
  "schema_version": 1,
  "vocabulary_name": "replica_runtime_semantic_41",
  "reference": {
    "method": "OVI-MAP",
    "commit": "58a804e2d7c82ba05a489eb071aba3367301fed8",
    "symbol": "REPLICA_52_PALETTE",
    "mapping": "class_name"
  },
  "background": {"rgb": [200, 200, 200], "source": "ovimap_replica52"},
  "classes": [
    {"name": "wall", "rgb": [196, 51, 182], "source": "ovimap_replica52"},
    {"name": "floor", "rgb": [188, 189, 34], "source": "ovimap_replica52"},
    {"name": "ceiling", "rgb": [174, 199, 232], "source": "ovimap_replica52"},
    {"name": "door", "rgb": [91, 229, 110], "source": "ovimap_replica52"},
    {"name": "window", "rgb": [229, 91, 104], "source": "ovimap_replica52"},
    {"name": "blinds", "rgb": [255, 152, 150], "source": "ovimap_replica52"},
    {"name": "cabinet", "rgb": [196, 156, 148], "source": "ovimap_replica52"},
    {"name": "chair", "rgb": [152, 223, 138], "source": "ovimap_replica52"},
    {"name": "table", "rgb": [91, 135, 229], "source": "ovimap_replica52"},
    {"name": "sofa", "rgb": [214, 39, 40], "source": "ovimap_replica52"},
    {"name": "rug", "rgb": [31, 119, 180], "source": "ovimap_replica52"},
    {"name": "bed", "rgb": [88, 218, 137], "source": "ovimap_replica52"},
    {"name": "basket", "rgb": [23, 190, 207], "source": "ovimap_replica52"},
    {"name": "blanket", "rgb": [44, 160, 44], "source": "ovimap_replica52"},
    {"name": "book", "rgb": [38, 100, 128], "source": "ovimap_replica52"},
    {"name": "bottle", "rgb": [237, 80, 38], "source": "ovimap_replica52"},
    {"name": "bowl", "rgb": [158, 218, 229], "source": "ovimap_replica52"},
    {"name": "cushion", "rgb": [229, 91, 223], "source": "ovimap_replica52"},
    {"name": "lamp", "rgb": [247, 182, 210], "source": "ovimap_replica52"},
    {"name": "picture", "rgb": [177, 82, 239], "source": "ovimap_replica52"},
    {"name": "indoor-plant", "rgb": [255, 127, 14], "source": "ovimap_replica52"},
    {"name": "plant-stand", "rgb": [137, 63, 14], "source": "ovimap_replica52"},
    {"name": "pillow", "rgb": [255, 187, 120], "source": "ovimap_replica52"},
    {"name": "shelf", "rgb": [34, 14, 130], "source": "ovimap_replica52"},
    {"name": "stool", "rgb": [58, 98, 137], "source": "ovimap_replica52"},
    {"name": "vase", "rgb": [143, 45, 115], "source": "ovimap_replica52"},
    {"name": "candle", "rgb": [138, 175, 62], "source": "ovimap_replica52"},
    {"name": "pillar", "rgb": [197, 176, 213], "source": "ovimap_replica52"},
    {"name": "plate", "rgb": [16, 212, 139], "source": "ovimap_replica52"},
    {"name": "pot", "rgb": [208, 49, 84], "source": "ovimap_replica52"},
    {"name": "switch", "rgb": [190, 77, 246], "source": "ovimap_replica52"},
    {"name": "vent", "rgb": [192, 229, 91], "source": "ovimap_replica52"},
    {"name": "wall-plug", "rgb": [227, 119, 194], "source": "ovimap_replica52"},
    {"name": "base-cabinet", "rgb": [62, 143, 148], "source": "ovimap_unused_replica52"},
    {"name": "countertop", "rgb": [143, 245, 111], "source": "ovimap_unused_replica52"},
    {"name": "cup", "rgb": [219, 219, 141], "source": "ovimap_unused_replica52"},
    {"name": "curtain", "rgb": [41, 206, 32], "source": "ovimap_unused_replica52"},
    {"name": "desk", "rgb": [237, 204, 37], "source": "ovimap_replica52"},
    {"name": "monitor", "rgb": [90, 119, 201], "source": "ovimap_replica52"},
    {"name": "tv-screen", "rgb": [112, 128, 144], "source": "ovimap_replica52"},
    {"name": "tv-stand", "rgb": [148, 103, 189], "source": "ovimap_replica52"}
  ]
}
```

- [ ] **Step 4: Implement strict palette loading**

Create `src/evaluation/semantic_visualization.py` with the public contract:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SemanticPalette:
    vocabulary_name: str
    names_by_id: dict[int, str]
    colors_by_id: dict[int, tuple[int, int, int]]
    source_sha256: str


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _rgb(value: object, name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"palette RGB must contain three channels: {name}")
    rgb = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"palette RGB channel is outside [0, 255]: {name}")
    return rgb


def load_semantic_palette(palette_path: str | Path, manifest_path: str | Path) -> SemanticPalette:
    palette_path = Path(palette_path)
    manifest = _load_json(Path(manifest_path))
    payload = _load_json(palette_path)
    vocabulary = manifest.get("vocabulary", {})
    name = str(vocabulary.get("name", ""))
    classes = [str(value) for value in vocabulary.get("classes", ())]
    entries = payload.get("classes", ())
    entry_names = [str(item.get("name", "")) for item in entries]
    if payload.get("schema_version") != 1 or payload.get("vocabulary_name") != name:
        raise ValueError("palette vocabulary does not match manifest")
    if entry_names != classes or len(set(entry_names)) != len(entry_names):
        raise ValueError("palette classes must exactly cover manifest classes in order")
    names_by_id = {0: "background"}
    colors_by_id = {0: _rgb(payload.get("background", {}).get("rgb"), "background")}
    for semantic_id, item in enumerate(entries, start=1):
        names_by_id[semantic_id] = entry_names[semantic_id - 1]
        colors_by_id[semantic_id] = _rgb(item.get("rgb"), entry_names[semantic_id - 1])
    if len(set(colors_by_id.values())) != len(colors_by_id):
        raise ValueError("palette RGB values must be collision-free")
    return SemanticPalette(
        vocabulary_name=name,
        names_by_id=names_by_id,
        colors_by_id=colors_by_id,
        source_sha256=hashlib.sha256(palette_path.read_bytes()).hexdigest(),
    )
```

- [ ] **Step 5: Run palette tests and commit**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_semantic_visualization.py -v
```

Expected: `2 passed`.

Commit:

```bash
git add configs/evaluation/replica41_semantic_palette.json \
  src/evaluation/semantic_visualization.py \
  tests/evaluation/test_semantic_visualization.py
git commit -m "feat: freeze OVIV2 semantic visualization palette"
```

### Task 2: Recolor Native And GT-Aligned PLY Files

**Files:**
- Modify: `src/evaluation/semantic_visualization.py`
- Modify: `tests/evaluation/test_semantic_visualization.py`

- [ ] **Step 1: Add failing synthetic PLY tests**

Extend the test file with a helper that writes vertices containing `semantic_id`, extra
custom fields, and one face. Add tests that call
`export_semantic_ply(source, destination, palette)` and assert:

```python
output = PlyData.read(destination)
assert output["vertex"].data.dtype.names == ("x", "y", "z", "red", "green", "blue")
np.testing.assert_array_equal(
    np.column_stack((output["vertex"]["red"], output["vertex"]["green"], output["vertex"]["blue"])),
    [[200, 200, 200], [196, 51, 182], [188, 189, 34]],
)
np.testing.assert_array_equal(output["vertex"]["x"], source_vertices["x"])
np.testing.assert_array_equal(output["face"]["vertex_indices"][0], [0, 1, 2])
assert "semantic_id" not in output["vertex"].data.dtype.names
```

Add validation cases for a missing `semantic_id`, semantic ID `42`, NaN positions, and an
out-of-range face index. Add a repeated-export assertion that compares output SHA256.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_semantic_visualization.py -k 'ply or invalid or deterministic' -v
```

Expected: import fails because `export_semantic_ply` is undefined.

- [ ] **Step 3: Implement PLY validation and deterministic recoloring**

Add `ExportedSemanticMesh`, `_sha256`, `_validate_faces`, and
`export_semantic_ply` to the evaluation module. The implementation must:

```python
@dataclass(frozen=True)
class ExportedSemanticMesh:
    source_sha256: str
    output_sha256: str
    vertex_count: int
    face_count: int
    semantic_counts: dict[int, int]


def export_semantic_ply(
    source_path: str | Path,
    destination_path: str | Path,
    palette: SemanticPalette,
) -> ExportedSemanticMesh:
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    source = PlyData.read(source_path)
    vertices = source["vertex"].data
    required = {"x", "y", "z", "semantic_id"}
    if not required.issubset(vertices.dtype.names or ()):
        raise ValueError("source PLY must contain x/y/z and semantic_id")
    positions = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    if not np.isfinite(positions).all():
        raise ValueError("source PLY positions must be finite")
    semantic_ids = np.asarray(vertices["semantic_id"], dtype=np.int64)
    invalid = sorted(set(int(value) for value in semantic_ids) - set(palette.colors_by_id))
    if invalid:
        raise ValueError(f"semantic IDs are outside frozen palette: {invalid}")
    faces = source["face"].data.copy() if "face" in source else None
    _validate_faces(faces, len(vertices))
    output_vertices = np.empty(
        len(vertices),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    output_vertices["x"], output_vertices["y"], output_vertices["z"] = positions.T
    colors = np.asarray([palette.colors_by_id[int(value)] for value in semantic_ids], dtype=np.uint8)
    output_vertices["red"], output_vertices["green"], output_vertices["blue"] = colors.T
    elements = [PlyElement.describe(output_vertices, "vertex")]
    if faces is not None:
        elements.append(PlyElement.describe(faces, "face"))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=False, byte_order="<").write(destination_path)
    values, counts = np.unique(semantic_ids, return_counts=True)
    return ExportedSemanticMesh(
        source_sha256=_sha256(source_path),
        output_sha256=_sha256(destination_path),
        vertex_count=len(vertices),
        face_count=0 if faces is None else len(faces),
        semantic_counts={int(key): int(value) for key, value in zip(values, counts, strict=True)},
    )
```

`_validate_faces` must flatten every list-valued `vertex_indices`, reject non-triangles,
negative indices, and indices greater than or equal to `vertex_count`.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_semantic_visualization.py -v
```

Expected: all tests pass.

Commit:

```bash
git add src/evaluation/semantic_visualization.py \
  tests/evaluation/test_semantic_visualization.py
git commit -m "feat: export semantic-colored OVIV2 meshes"
```

### Task 3: Add Atomic Batch CLI, Legend, And Real Replica-8 Export

**Files:**
- Modify: `src/evaluation/semantic_visualization.py`
- Create: `scripts/evaluation/export_oviv2_semantic_visualizations.py`
- Modify: `tests/evaluation/test_semantic_visualization.py`
- Create: `tests/evaluation/test_export_oviv2_semantic_visualizations_cli.py`

- [ ] **Step 1: Write failing batch and CLI tests**

Build two synthetic scene trees containing `final/oviv2_instance_mesh.ply` and
`evaluation/semantic_map_gt.ply`. Assert `export_semantic_visualizations(...)` creates:

```python
assert (output / "semantic_palette.json").is_file()
assert (output / "semantic_legend.png").is_file()
assert (output / "room0" / "oviv2_semantic_native.ply").is_file()
assert (output / "room0" / "oviv2_semantic_gt_aligned.ply").is_file()
manifest = json.loads((output / "room0" / "export_manifest.json").read_text())
assert manifest["scene"] == "room0"
assert manifest["native"]["vertex_count"] == 3
assert manifest["gt_aligned"]["semantic_counts"] == {"0": 1, "1": 1, "2": 1}
```

Test atomic failure by making the second scene invalid and asserting the output directory
does not exist. Test CLI help with:

```python
result = subprocess.run(
    [sys.executable, "scripts/evaluation/export_oviv2_semantic_visualizations.py", "--help"],
    check=False,
    capture_output=True,
    text=True,
)
assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run batch/CLI tests and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_semantic_visualization.py \
  tests/evaluation/test_export_oviv2_semantic_visualizations_cli.py -v
```

Expected: failures because the batch API and CLI do not exist.

- [ ] **Step 3: Implement legend and atomic batch publication**

Add `write_semantic_legend` using Pillow with a white canvas, 24 px rows, three columns,
18x18 RGB swatches, semantic ID, and class name. Add
`export_semantic_visualizations(batch_root, manifest_path, palette_path, output, scenes=None)`.
It must validate manifest scene order, validate requested scenes, export both source meshes
into a sibling temporary directory, write sorted/indented JSON with `allow_nan=False`, and
publish with `os.replace` only after every scene succeeds. When output already exists,
rename it to a sibling backup, replace it with the completed temporary directory, restore
the backup on publication failure, and remove the backup after success.

The batch palette JSON must contain the resolved semantic ID, class name, RGB value,
source palette SHA256, vocabulary name, and manifest SHA256. Each scene manifest must use
paths relative to `batch_root` or the output root, never transient temporary paths.

- [ ] **Step 4: Add the thin direct-execution CLI**

Create `scripts/evaluation/export_oviv2_semantic_visualizations.py`:

```python
#!/usr/bin/env python3
"""Export viewer-compatible semantic meshes from verified OVIV2 Replica runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.semantic_visualization import export_semantic_visualizations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--palette",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/replica41_semantic_palette.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", action="append", dest="scenes")
    args = parser.parse_args(argv)
    result = export_semantic_visualizations(
        args.batch_root,
        args.manifest,
        args.palette,
        args.output,
        scenes=args.scenes,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run all focused tests and commit**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/evaluation/test_semantic_visualization.py \
  tests/evaluation/test_export_oviv2_semantic_visualizations_cli.py
```

Expected: all tests pass.

Commit:

```bash
git add src/evaluation/semantic_visualization.py \
  scripts/evaluation/export_oviv2_semantic_visualizations.py \
  tests/evaluation/test_semantic_visualization.py \
  tests/evaluation/test_export_oviv2_semantic_visualizations_cli.py
git commit -m "feat: batch export OVIV2 semantic visualizations"
```

- [ ] **Step 6: Export all verified Replica scenes**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  scripts/evaluation/export_oviv2_semantic_visualizations.py \
  --batch-root outputs/oviv2_replica8_frozen \
  --manifest configs/evaluation/manifests/replica8.json \
  --output outputs/oviv2_replica8_frozen/paper_visualizations
```

Expected JSON includes `"scene_count": 8`.

- [ ] **Step 7: Verify real files and deterministic re-export**

Run the exporter a second time, then run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python - <<'PY'
import json
from pathlib import Path
from plyfile import PlyData

root = Path("outputs/oviv2_replica8_frozen/paper_visualizations")
scenes = ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")
for scene in scenes:
    manifest = json.loads((root / scene / "export_manifest.json").read_text())
    for name in ("oviv2_semantic_native.ply", "oviv2_semantic_gt_aligned.ply"):
        ply = PlyData.read(root / scene / name)
        assert ply["vertex"].data.dtype.names == ("x", "y", "z", "red", "green", "blue")
    print(scene, manifest["native"]["vertex_count"], manifest["gt_aligned"]["vertex_count"])
PY
```

Expected: eight scene lines and no assertion failure. Open the room0 native and GT-aligned
PLY files in the configured VS Code viewer and inspect `semantic_legend.png`.

- [ ] **Step 8: Run the complete relevant regression suite**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2 \
  tests/architecture/test_oviv2_voxel_first.py \
  tests/evaluation/test_oviv2_replica.py \
  tests/evaluation/test_evaluate_oviv2_replica_cli.py \
  tests/evaluation/test_semantic_visualization.py \
  tests/evaluation/test_export_oviv2_semantic_visualizations_cli.py
```

Expected: zero failures.

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; unrelated pre-existing worktree changes remain unstaged.
