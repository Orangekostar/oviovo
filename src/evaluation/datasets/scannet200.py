from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


SCANNET200_5_SCENES = (
    "scene0011_00",
    "scene0050_00",
    "scene0231_00",
    "scene0378_00",
    "scene0518_00",
)

_SCENE_ID = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_REQUIRED_SUFFIXES = {
    "sensor": ".sens",
    "metadata": ".txt",
    "mesh": "_vh_clean_2.ply",
    "labels_mesh": "_vh_clean_2.labels.ply",
    "segments": "_vh_clean_2.0.010000.segs.json",
    "aggregation": ".aggregation.json",
}


class ScanNetValidationError(ValueError):
    """Raised when official ScanNet inputs do not satisfy the frozen contract."""


@dataclass(frozen=True)
class ScanNetFileRecord:
    role: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ScanNetSceneInventory:
    scene_id: str
    dataset_root: str
    files: Mapping[str, ScanNetFileRecord]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_scannet200_scene(
    dataset_root: str | Path,
    scene_id: str,
) -> ScanNetSceneInventory:
    normalized_scene = str(scene_id)
    if _SCENE_ID.fullmatch(normalized_scene) is None:
        raise ScanNetValidationError(f"invalid ScanNet scene ID: {normalized_scene!r}")
    root = Path(dataset_root).expanduser().resolve()
    errors = []
    if not root.is_dir():
        errors.append(f"dataset_root is not a directory: {root}")
    scene_root = root / normalized_scene
    records: dict[str, ScanNetFileRecord] = {}
    roles_by_resolved_path: dict[Path, str] = {}
    for role, suffix in _REQUIRED_SUFFIXES.items():
        candidate = scene_root / f"{normalized_scene}{suffix}"
        if not candidate.exists():
            errors.append(f"{role}: missing file {candidate}")
            continue
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            errors.append(f"{role}: resolved path is outside dataset_root: {resolved}")
            continue
        if not resolved.is_file():
            errors.append(f"{role}: expected a regular file: {resolved}")
            continue
        previous_role = roles_by_resolved_path.get(resolved)
        if previous_role is not None:
            errors.append(
                f"{role}: duplicate resolved path with {previous_role}: {resolved}"
            )
            continue
        size = resolved.stat().st_size
        if size <= 0:
            errors.append(f"{role}: empty file {resolved}")
            continue
        roles_by_resolved_path[resolved] = role
        records[role] = ScanNetFileRecord(
            role=role,
            path=str(resolved),
            size_bytes=size,
            sha256=_sha256(resolved),
        )
    if errors:
        raise ScanNetValidationError(
            f"ScanNet200 scene {normalized_scene} validation failed: "
            + "; ".join(errors)
        )
    return ScanNetSceneInventory(
        scene_id=normalized_scene,
        dataset_root=str(root),
        files=MappingProxyType(dict(sorted(records.items()))),
    )


def validate_scannet200_split(
    dataset_root: str | Path,
    scene_ids: tuple[str, ...],
) -> tuple[ScanNetSceneInventory, ...]:
    normalized = tuple(scene_ids)
    if normalized != SCANNET200_5_SCENES:
        raise ScanNetValidationError(
            "scene IDs must exactly equal the frozen ScanNet200 five-scene split"
        )
    inventories = []
    failures = []
    for scene_id in normalized:
        try:
            inventories.append(validate_scannet200_scene(dataset_root, scene_id))
        except ScanNetValidationError as error:
            failures.append(f"{scene_id}: {error}")
    if failures:
        raise ScanNetValidationError(
            "ScanNet200 split validation failed: " + " | ".join(failures)
        )
    return tuple(inventories)
