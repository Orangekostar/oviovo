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
