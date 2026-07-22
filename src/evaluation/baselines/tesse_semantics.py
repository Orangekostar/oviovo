"""Frozen name-based semantic crosswalks for TESSE-CD common v2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


def normalize_semantic_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class SemanticLookup:
    semantic_id: int
    native_name: str
    matched: bool


@dataclass(frozen=True)
class TesseSemanticCrosswalk:
    scene: str
    unknown: SemanticLookup
    aliases: Mapping[str, SemanticLookup]
    valid_semantic_ids: frozenset[int]
    label_space_sha256: str
    alias_config_sha256: str

    def lookup(self, label: str) -> SemanticLookup:
        if not isinstance(label, str):
            raise TypeError("numeric semantic IDs cannot be compared across label spaces")
        normalized = normalize_semantic_name(label)
        if not normalized:
            return self.unknown
        return self.aliases.get(normalized, self.unknown)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_tesse_semantic_crosswalk(
    alias_config_path: Path,
    scene: str,
    label_space_path: Path,
) -> TesseSemanticCrosswalk:
    config = _mapping(
        yaml.safe_load(alias_config_path.read_text(encoding="utf-8")),
        "alias config",
    )
    if config.get("schema_version") != 1 or config.get("protocol") != "tesse_cd_common_v2":
        raise ValueError("invalid TESSE-CD common-v2 alias config identity")
    scene_config = _mapping(
        _mapping(config.get("scenes"), "alias scenes").get(scene),
        f"alias scene {scene}",
    )
    source = _mapping(scene_config.get("label_space"), "label_space declaration")
    if source.get("filename") != label_space_path.name:
        raise ValueError("native label-space filename mismatch")
    label_space_hash = _sha256(label_space_path)
    if source.get("sha256") != label_space_hash:
        raise ValueError("native label-space SHA256 mismatch")

    label_space = _mapping(
        yaml.safe_load(label_space_path.read_text(encoding="utf-8")),
        "native label space",
    )
    names_by_id: dict[int, str] = {}
    ids_by_name: dict[str, int] = {}
    for raw_entry in label_space.get("label_names", ()):
        entry = _mapping(raw_entry, "native label entry")
        semantic_id = int(entry["label"])
        native_name = str(entry["name"]).strip()
        normalized = normalize_semantic_name(native_name)
        if semantic_id in names_by_id or not normalized or normalized in ids_by_name:
            raise ValueError("native label space contains duplicate labels")
        names_by_id[semantic_id] = native_name
        ids_by_name[normalized] = semantic_id

    unknown_name = str(config.get("unknown_name", "Unknown"))
    unknown_id = ids_by_name.get(normalize_semantic_name(unknown_name))
    if unknown_id != 0:
        raise ValueError("native label space must define Unknown as semantic ID 0")
    object_ids = frozenset(int(value) for value in label_space.get("object_labels", ()))
    if not object_ids or not object_ids <= set(names_by_id):
        raise ValueError("native label space has an invalid object vocabulary")

    resolved: dict[str, SemanticLookup] = {}
    aliases = _mapping(scene_config.get("aliases"), "scene aliases")
    for target_name, raw_aliases in aliases.items():
        target_id = ids_by_name.get(normalize_semantic_name(target_name))
        if target_id is None or target_id not in object_ids:
            raise ValueError(f"alias target is not an object label: {target_name}")
        if not isinstance(raw_aliases, list) or not raw_aliases:
            raise ValueError(f"aliases for {target_name} must be a non-empty list")
        lookup = SemanticLookup(
            semantic_id=target_id,
            native_name=names_by_id[target_id],
            matched=True,
        )
        for raw_alias in raw_aliases:
            if not isinstance(raw_alias, str):
                raise TypeError("semantic aliases must be names, not numeric IDs")
            normalized = normalize_semantic_name(raw_alias)
            if not normalized:
                raise ValueError("semantic aliases must be non-empty")
            if normalized in resolved:
                raise ValueError(f"duplicate normalized alias: {raw_alias}")
            resolved[normalized] = lookup

    return TesseSemanticCrosswalk(
        scene=scene,
        unknown=SemanticLookup(semantic_id=0, native_name=names_by_id[0], matched=False),
        aliases=resolved,
        valid_semantic_ids=object_ids,
        label_space_sha256=label_space_hash,
        alias_config_sha256=_sha256(alias_config_path),
    )
