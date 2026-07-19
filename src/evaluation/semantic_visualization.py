from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
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
