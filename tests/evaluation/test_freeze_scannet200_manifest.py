from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.evaluation.datasets.scannet200 import SCANNET200_5_SCENES


SCRIPT = Path("scripts/evaluation/freeze_scannet200_manifest.py")
SUFFIXES = (
    ".sens",
    ".txt",
    "_vh_clean_2.ply",
    "_vh_clean_2.labels.ply",
    "_vh_clean_2.0.010000.segs.json",
    ".aggregation.json",
)


def _dataset(root: Path) -> None:
    for scene_id in SCANNET200_5_SCENES:
        scene = root / scene_id
        scene.mkdir(parents=True)
        for suffix in SUFFIXES:
            (scene / f"{scene_id}{suffix}").write_bytes(suffix.encode("ascii"))


def _run(
    root: Path,
    vocabulary: Path,
    digest: str,
    output: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(root),
            "--vocabulary",
            str(vocabulary),
            "--vocabulary-sha256",
            digest,
            "--frozen-at",
            "2026-07-19T00:00:00Z",
            "--output",
            str(output),
            *extra,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_freezer_writes_byte_identical_validated_manifests(tmp_path: Path) -> None:
    root = tmp_path / "scannet"
    _dataset(root)
    vocabulary = tmp_path / "scannet200_classes.txt"
    vocabulary.write_text("wall\nfloor\n", encoding="utf-8")
    digest = hashlib.sha256(vocabulary.read_bytes()).hexdigest()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_result = _run(root, vocabulary, digest, first)
    second_result = _run(root, vocabulary, digest, second)

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    assert first.read_bytes() == second.read_bytes()
    manifest = json.loads(first.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["manifest_id"] == "scannet200_5_static_v1"
    assert manifest["dataset"] == "ScanNet200"
    assert manifest["split"] == "scannet200_5"
    assert manifest["vocabulary"] == {
        "name": "scannet200",
        "source_path": str(vocabulary.resolve()),
        "source_sha256": digest,
    }
    assert tuple(scene["scene_id"] for scene in manifest["scenes"]) == SCANNET200_5_SCENES
    assert all(scene["pose_source"] == "sensor" for scene in manifest["scenes"])
    assert all(scene["depth_source"] == "sensor" for scene in manifest["scenes"])


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("incomplete", "scene0518_00"),
        ("missing_vocabulary", "vocabulary"),
        ("wrong_hash", "hash"),
        ("existing", "exists"),
    ],
)
def test_freezer_refuses_invalid_or_overwriting_inputs(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    root = tmp_path / "scannet"
    _dataset(root)
    vocabulary = tmp_path / "scannet200_classes.txt"
    vocabulary.write_text("wall\n", encoding="utf-8")
    digest = hashlib.sha256(vocabulary.read_bytes()).hexdigest()
    output = tmp_path / "manifest.json"
    if failure == "incomplete":
        missing = root / "scene0518_00" / "scene0518_00.sens"
        missing.unlink()
    elif failure == "missing_vocabulary":
        vocabulary.unlink()
    elif failure == "wrong_hash":
        digest = "0" * 64
    else:
        output.write_text("existing\n", encoding="utf-8")

    result = _run(root, vocabulary, digest, output)

    assert result.returncode != 0
    assert message in result.stderr.lower()
    if failure == "existing":
        assert output.read_text(encoding="utf-8") == "existing\n"
    else:
        assert not output.exists()
