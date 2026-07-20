from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


SCRIPT = Path("scripts/evaluation/freeze_oviv2_scannet200_manifest.py")
SCENES = (
    "scene0011_00",
    "scene0050_00",
    "scene0231_00",
    "scene0378_00",
    "scene0518_00",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    raw_root = tmp_path / "raw"
    exported_root = tmp_path / "exported"
    gt_root = tmp_path / "gt"
    gt_root.mkdir()
    raw_scenes = []
    for scene in SCENES:
        raw_scene = raw_root / scene
        raw_scene.mkdir(parents=True)
        metadata = raw_scene / f"{scene}.txt"
        metadata.write_text(
            "axisAlignment = " + " ".join(str(value) for value in np.eye(4).reshape(-1)) + "\n",
            encoding="utf-8",
        )
        raw_scenes.append(
            {
                "scene_id": scene,
                "files": {
                    "metadata": {
                        "path": str(metadata.resolve()),
                        "sha256": _sha256(metadata),
                    }
                },
            }
        )
        exported = exported_root / scene
        for name in ("color", "depth", "pose", "intrinsic"):
            (exported / name).mkdir(parents=True)
        np.savetxt(exported / "intrinsic" / "intrinsic_depth.txt", np.eye(4))
        np.savetxt(exported / "intrinsic" / "intrinsic_color.txt", np.eye(4))
        for source_id in range(11):
            Image.fromarray(np.full((4, 4, 3), source_id, dtype=np.uint8)).save(
                exported / "color" / f"{source_id}.jpg"
            )
            Image.fromarray(np.full((2, 2), 1000, dtype=np.uint16)).save(
                exported / "depth" / f"{source_id}.png"
            )
            pose = np.eye(4)
            if scene == "scene0050_00" and source_id == 10:
                pose[:] = np.nan
            np.savetxt(exported / "pose" / f"{source_id}.txt", pose)
        (gt_root / f"{scene}.ply").write_bytes(b"ply\nend_header\n")

    raw_manifest = tmp_path / "raw_manifest.json"
    raw_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "scannet200_5_static_v1",
                "dataset": "ScanNet200",
                "scenes": raw_scenes,
            }
        ),
        encoding="utf-8",
    )
    classes = [f"class-{index:03d}" for index in range(200)]
    classes_json = tmp_path / "scannet200_classes.json"
    classes_json.write_text(
        json.dumps({"classes": classes, "aliases": {}}) + "\n",
        encoding="utf-8",
    )
    classes_json.with_suffix(".txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    constants = tmp_path / "scannet200_constants.py"
    constants.write_text(
        f"VALID_CLASS_IDS_200 = {tuple(range(1, 201))!r}\n"
        f"CLASS_LABELS_200 = {tuple(classes)!r}\n",
        encoding="utf-8",
    )
    return {
        "raw_manifest": raw_manifest,
        "exported_root": exported_root,
        "gt_root": gt_root,
        "classes_json": classes_json,
        "constants": constants,
    }


def _run(inputs: dict[str, Path], output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--raw-manifest",
            str(inputs["raw_manifest"]),
            "--exported-root",
            str(inputs["exported_root"]),
            "--official-gt-root",
            str(inputs["gt_root"]),
            "--official-constants",
            str(inputs["constants"]),
            "--classes-json",
            str(inputs["classes_json"]),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_freezer_writes_deterministic_explicit_stage4_manifest(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_result = _run(inputs, first)
    second_result = _run(inputs, second)

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    assert first.read_bytes() == second.read_bytes()
    manifest = json.loads(first.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["manifest_id"] == "oviv2_scannet200_5_stage4_v1"
    assert manifest["dataset"] == "ScanNet200"
    assert manifest["split"] == "scannet200_5_heldout"
    assert manifest["vocabulary"]["class_count"] == 200
    assert manifest["vocabulary"]["class_ids"] == list(range(1, 201))
    assert [scene["scene"] for scene in manifest["scenes"]] == list(SCENES)
    assert [scene["source_frame_ids"] for scene in manifest["scenes"]] == [
        [0, 10],
        [0],
        [0, 10],
        [0, 10],
        [0, 10],
    ]
    assert set(manifest["scenes"][0]["frame_inputs"]["0"]) == {
        "color_sha256",
        "depth_sha256",
        "pose_sha256",
    }


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing_depth", "frame inventory"),
        ("symlink_color", "non-symlink"),
        ("symlink_gt", "official gt"),
        ("class_order", "official scannet200"),
        ("metadata_hash", "metadata hash"),
    ],
)
def test_freezer_rejects_invalid_inputs_without_publishing(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    inputs = _fixture(tmp_path)
    if failure == "missing_depth":
        (inputs["exported_root"] / "scene0011_00" / "depth" / "10.png").unlink()
    elif failure == "symlink_color":
        color = inputs["exported_root"] / "scene0011_00" / "color" / "10.jpg"
        target = color.with_name("target.jpg")
        color.rename(target)
        color.symlink_to(target)
    elif failure == "symlink_gt":
        gt = inputs["gt_root"] / "scene0011_00.ply"
        target = inputs["gt_root"] / "target.ply"
        gt.rename(target)
        gt.symlink_to(target)
    elif failure == "class_order":
        classes = json.loads(inputs["classes_json"].read_text(encoding="utf-8"))
        classes["classes"][0], classes["classes"][1] = (
            classes["classes"][1],
            classes["classes"][0],
        )
        inputs["classes_json"].write_text(json.dumps(classes) + "\n", encoding="utf-8")
    else:
        raw = json.loads(inputs["raw_manifest"].read_text(encoding="utf-8"))
        raw["scenes"][0]["files"]["metadata"]["sha256"] = "0" * 64
        inputs["raw_manifest"].write_text(json.dumps(raw), encoding="utf-8")
    output = tmp_path / "failed.json"

    result = _run(inputs, output)

    assert result.returncode != 0
    assert message in result.stderr.lower()
    assert not output.exists()


def test_freezer_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    output = tmp_path / "manifest.json"
    output.write_text("existing\n", encoding="utf-8")

    result = _run(inputs, output)

    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert output.read_text(encoding="utf-8") == "existing\n"
