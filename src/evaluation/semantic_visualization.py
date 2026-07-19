from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData, PlyElement


@dataclass(frozen=True)
class SemanticPalette:
    vocabulary_name: str
    names_by_id: dict[int, str]
    colors_by_id: dict[int, tuple[int, int, int]]
    source_sha256: str


@dataclass(frozen=True)
class ExportedSemanticMesh:
    source_sha256: str
    output_sha256: str
    vertex_count: int
    face_count: int
    semantic_counts: dict[int, int]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb(value: object, name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"palette RGB must contain three channels: {name}")
    if any(not isinstance(channel, int) or isinstance(channel, bool) for channel in value):
        raise ValueError(f"palette RGB channels must be integers: {name}")
    rgb = tuple(value)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"palette RGB channel is outside [0, 255]: {name}")
    return rgb


def load_semantic_palette(
    palette_path: str | Path,
    manifest_path: str | Path,
) -> SemanticPalette:
    palette_path = Path(palette_path)
    manifest = _load_json(Path(manifest_path))
    payload = _load_json(palette_path)
    vocabulary = manifest.get("vocabulary", {})
    name = str(vocabulary.get("name", ""))
    classes = [str(value) for value in vocabulary.get("classes", ())]
    entries = payload.get("classes", ())
    if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
        raise ValueError("palette classes must be a list of objects")
    entry_names = [str(item.get("name", "")) for item in entries]
    if payload.get("schema_version") != 1 or payload.get("vocabulary_name") != name:
        raise ValueError("palette vocabulary does not match manifest")
    if entry_names != classes or len(set(entry_names)) != len(entry_names):
        raise ValueError("palette classes must exactly cover manifest classes in order")
    background = payload.get("background", {})
    if not isinstance(background, dict):
        raise ValueError("palette background must be an object")
    names_by_id = {0: "background"}
    colors_by_id = {0: _rgb(background.get("rgb"), "background")}
    for semantic_id, item in enumerate(entries, start=1):
        class_name = entry_names[semantic_id - 1]
        names_by_id[semantic_id] = class_name
        colors_by_id[semantic_id] = _rgb(item.get("rgb"), class_name)
    if len(set(colors_by_id.values())) != len(colors_by_id):
        raise ValueError("palette RGB values must be collision-free")
    return SemanticPalette(
        vocabulary_name=name,
        names_by_id=names_by_id,
        colors_by_id=colors_by_id,
        source_sha256=hashlib.sha256(palette_path.read_bytes()).hexdigest(),
    )


def _polygons(source: PlyData, vertex_count: int) -> tuple[np.ndarray, ...]:
    if "face" not in source:
        return ()
    faces = source["face"].data
    if "vertex_indices" not in (faces.dtype.names or ()):
        raise ValueError("source PLY face element must contain vertex_indices")
    polygons: list[np.ndarray] = []
    for face in faces["vertex_indices"]:
        indices = np.asarray(face, dtype=np.int64).reshape(-1)
        if len(indices) < 3:
            raise ValueError("source PLY faces must contain at least three vertices")
        if np.any(indices < 0) or np.any(indices >= vertex_count):
            raise ValueError("source PLY face index lies outside vertex array")
        polygons.append(indices.astype(np.int32))
    return tuple(polygons)


def export_semantic_ply(
    source_path: str | Path,
    destination_path: str | Path,
    palette: SemanticPalette,
) -> ExportedSemanticMesh:
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    if source_path.suffix.lower() != ".ply" or destination_path.suffix.lower() != ".ply":
        raise ValueError("semantic mesh source and destination must end in .ply")
    source = PlyData.read(source_path)
    vertices = source["vertex"].data
    required = {"x", "y", "z", "semantic_id"}
    if not required.issubset(vertices.dtype.names or ()):
        raise ValueError("source PLY must contain x/y/z and semantic_id")
    positions = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float32
    )
    if not np.isfinite(positions).all():
        raise ValueError("source PLY positions must be finite")
    semantic_ids = np.asarray(vertices["semantic_id"], dtype=np.int64)
    invalid = sorted(set(int(value) for value in semantic_ids) - set(palette.colors_by_id))
    if invalid:
        raise ValueError(f"semantic IDs are outside frozen palette: {invalid}")
    polygons = _polygons(source, len(vertices))

    output_vertices = np.empty(
        len(vertices),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    output_vertices["x"], output_vertices["y"], output_vertices["z"] = positions.T
    colors = np.asarray(
        [palette.colors_by_id[int(value)] for value in semantic_ids],
        dtype=np.uint8,
    )
    output_vertices["red"], output_vertices["green"], output_vertices["blue"] = colors.T
    elements = [PlyElement.describe(output_vertices, "vertex")]
    if polygons:
        output_faces = np.empty(len(polygons), dtype=[("vertex_indices", "O")])
        for index, polygon in enumerate(polygons):
            output_faces["vertex_indices"][index] = polygon
        elements.append(PlyElement.describe(output_faces, "face"))

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination_path.parent,
            prefix=f".{destination_path.stem}.",
            suffix=".ply",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        PlyData(elements, text=False, byte_order="<").write(temporary_path)
        os.replace(temporary_path, destination_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    values, counts = np.unique(semantic_ids, return_counts=True)
    return ExportedSemanticMesh(
        source_sha256=_sha256(source_path),
        output_sha256=_sha256(destination_path),
        vertex_count=len(vertices),
        face_count=len(polygons),
        semantic_counts={
            int(semantic_id): int(count)
            for semantic_id, count in zip(values, counts, strict=True)
        },
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_semantic_legend(path: str | Path, palette: SemanticPalette) -> None:
    path = Path(path)
    columns = 3
    row_height = 28
    cell_width = 240
    margin = 12
    items = sorted(palette.names_by_id.items())
    rows = math.ceil(len(items) / columns)
    image = Image.new(
        "RGB",
        (margin * 2 + columns * cell_width, margin * 2 + rows * row_height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for index, (semantic_id, class_name) in enumerate(items):
        column = index // rows
        row = index % rows
        x = margin + column * cell_width
        y = margin + row * row_height
        draw.rectangle(
            (x, y + 4, x + 18, y + 22),
            fill=palette.colors_by_id[semantic_id],
            outline=(32, 32, 32),
        )
        draw.text(
            (x + 26, y + 6),
            f"{semantic_id:02d}  {class_name}",
            fill=(0, 0, 0),
            font=font,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _mesh_record(
    stats: ExportedSemanticMesh,
    *,
    source_path: str,
    output_path: str,
) -> dict[str, Any]:
    return {
        "source_path": source_path,
        "source_sha256": stats.source_sha256,
        "output_path": output_path,
        "output_sha256": stats.output_sha256,
        "vertex_count": stats.vertex_count,
        "face_count": stats.face_count,
        "semantic_counts": stats.semantic_counts,
        "background_vertex_count": stats.semantic_counts.get(0, 0),
    }


def _publish_directory(temporary: Path, output: Path) -> None:
    backup: Path | None = None
    if output.exists() and not output.is_dir():
        raise ValueError("semantic visualization output exists and is not a directory")
    if output.exists():
        backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
        os.replace(output, backup)
    try:
        os.replace(temporary, output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def export_semantic_visualizations(
    batch_root: str | Path,
    manifest_path: str | Path,
    palette_path: str | Path,
    output: str | Path,
    *,
    scenes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    batch_root = Path(batch_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    palette_path = Path(palette_path).resolve()
    output = Path(output).resolve()
    manifest = _load_json(manifest_path)
    scene_items = manifest.get("scenes", ())
    if not isinstance(scene_items, list) or any(not isinstance(item, dict) for item in scene_items):
        raise ValueError("frozen manifest scenes must be a list of objects")
    manifest_scenes = tuple(str(item.get("scene", "")) for item in scene_items)
    if not manifest_scenes or "" in manifest_scenes or len(set(manifest_scenes)) != len(manifest_scenes):
        raise ValueError("frozen manifest scene IDs must be non-empty and unique")
    selected_scenes = manifest_scenes if scenes is None else tuple(str(scene) for scene in scenes)
    if not selected_scenes or len(set(selected_scenes)) != len(selected_scenes):
        raise ValueError("requested scene IDs must be non-empty and unique")
    outside = sorted(set(selected_scenes) - set(manifest_scenes))
    if outside:
        raise ValueError(f"requested scenes are outside frozen manifest: {outside}")

    palette = load_semantic_palette(palette_path, manifest_path)
    manifest_sha256 = _sha256(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        resolved_palette = {
            "schema_version": 1,
            "vocabulary_name": palette.vocabulary_name,
            "source_palette_sha256": palette.source_sha256,
            "manifest_sha256": manifest_sha256,
            "classes": [
                {
                    "semantic_id": semantic_id,
                    "name": palette.names_by_id[semantic_id],
                    "rgb": list(palette.colors_by_id[semantic_id]),
                }
                for semantic_id in sorted(palette.names_by_id)
            ],
        }
        _write_json(temporary / "semantic_palette.json", resolved_palette)
        write_semantic_legend(temporary / "semantic_legend.png", palette)

        for scene in selected_scenes:
            source_native = batch_root / scene / "final" / "oviv2_instance_mesh.ply"
            source_aligned = batch_root / scene / "evaluation" / "semantic_map_gt.ply"
            destination_dir = temporary / scene
            destination_native = destination_dir / "oviv2_semantic_native.ply"
            destination_aligned = destination_dir / "oviv2_semantic_gt_aligned.ply"
            native = export_semantic_ply(source_native, destination_native, palette)
            aligned = export_semantic_ply(source_aligned, destination_aligned, palette)
            scene_manifest = {
                "schema_version": 1,
                "scene": scene,
                "vocabulary_name": palette.vocabulary_name,
                "palette_sha256": palette.source_sha256,
                "benchmark_manifest_sha256": manifest_sha256,
                "native": _mesh_record(
                    native,
                    source_path=str(source_native.relative_to(batch_root)),
                    output_path=f"{scene}/oviv2_semantic_native.ply",
                ),
                "gt_aligned": _mesh_record(
                    aligned,
                    source_path=str(source_aligned.relative_to(batch_root)),
                    output_path=f"{scene}/oviv2_semantic_gt_aligned.ply",
                ),
            }
            _write_json(destination_dir / "export_manifest.json", scene_manifest)

        _publish_directory(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return {
        "output": str(output),
        "scene_count": len(selected_scenes),
        "scenes": list(selected_scenes),
    }
