from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.datasets.scannet200 import (
    SCANNET200_5_SCENES,
    ScanNetValidationError,
    validate_scannet200_scene,
    validate_scannet200_split,
)


REQUIRED_SUFFIXES = {
    "sensor": ".sens",
    "metadata": ".txt",
    "mesh": "_vh_clean_2.ply",
    "labels_mesh": "_vh_clean_2.labels.ply",
    "segments": "_vh_clean_2.0.010000.segs.json",
    "aggregation": ".aggregation.json",
}


def _scene(root: Path, scene_id: str) -> Path:
    directory = root / scene_id
    directory.mkdir(parents=True)
    for role, suffix in REQUIRED_SUFFIXES.items():
        (directory / f"{scene_id}{suffix}").write_bytes(f"{role}\n".encode("ascii"))
    return directory


def test_validate_scannet200_scene_returns_immutable_hashed_inventory(
    tmp_path: Path,
) -> None:
    _scene(tmp_path, "scene0011_00")

    inventory = validate_scannet200_scene(tmp_path, "scene0011_00")

    assert inventory.scene_id == "scene0011_00"
    assert Path(inventory.dataset_root) == tmp_path.resolve()
    assert set(inventory.files) == set(REQUIRED_SUFFIXES)
    assert all(record.sha256 for record in inventory.files.values())
    assert all(record.size_bytes > 0 for record in inventory.files.values())
    with pytest.raises(TypeError):
        inventory.files["new"] = inventory.files["mesh"]


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "missing"),
        ("empty", "empty"),
        ("directory", "regular file"),
        ("escape", "outside dataset_root"),
        ("duplicate", "duplicate resolved"),
    ],
)
def test_validate_scannet200_scene_rejects_invalid_files(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    scene_id = "scene0011_00"
    directory = _scene(tmp_path, scene_id)
    sensor = directory / f"{scene_id}.sens"
    metadata = directory / f"{scene_id}.txt"
    if failure == "missing":
        sensor.unlink()
    elif failure == "empty":
        sensor.write_bytes(b"")
    elif failure == "directory":
        sensor.unlink()
        sensor.mkdir()
    elif failure == "escape":
        outside = tmp_path.parent / "outside-scannet.sens"
        outside.write_bytes(b"outside")
        sensor.unlink()
        sensor.symlink_to(outside)
    else:
        metadata.unlink()
        metadata.symlink_to(sensor)

    with pytest.raises(ScanNetValidationError, match=message):
        validate_scannet200_scene(tmp_path, scene_id)


@pytest.mark.parametrize("scene_id", ["scene11_00", "scene0011", "../scene0011_00"])
def test_validate_scannet200_scene_rejects_invalid_scene_id(
    tmp_path: Path,
    scene_id: str,
) -> None:
    with pytest.raises(ScanNetValidationError, match="scene ID"):
        validate_scannet200_scene(tmp_path, scene_id)


def test_validate_scannet200_split_preserves_frozen_order(tmp_path: Path) -> None:
    for scene_id in reversed(SCANNET200_5_SCENES):
        _scene(tmp_path, scene_id)

    inventories = validate_scannet200_split(tmp_path, SCANNET200_5_SCENES)

    assert tuple(item.scene_id for item in inventories) == SCANNET200_5_SCENES


@pytest.mark.parametrize(
    "scene_ids",
    [
        SCANNET200_5_SCENES[:-1],
        SCANNET200_5_SCENES + (SCANNET200_5_SCENES[0],),
        tuple(reversed(SCANNET200_5_SCENES)),
        SCANNET200_5_SCENES[:-1] + ("scene9999_00",),
    ],
)
def test_validate_scannet200_split_requires_exact_candidate_set(
    tmp_path: Path,
    scene_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ScanNetValidationError, match="frozen ScanNet200 five-scene split"):
        validate_scannet200_split(tmp_path, scene_ids)


def test_validate_scannet200_split_aggregates_all_scene_failures(tmp_path: Path) -> None:
    _scene(tmp_path, SCANNET200_5_SCENES[0])

    with pytest.raises(ScanNetValidationError) as error:
        validate_scannet200_split(tmp_path, SCANNET200_5_SCENES)

    for scene_id in SCANNET200_5_SCENES[1:]:
        assert scene_id in str(error.value)
