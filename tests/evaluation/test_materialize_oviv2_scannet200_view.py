from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.materialize_oviv2_scannet200_view import materialize_scene


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> Path:
    source = tmp_path / "exported" / "scene0011_00"
    for name in ("color", "depth", "pose", "intrinsic"):
        (source / name).mkdir(parents=True)
    np.savetxt(source / "intrinsic" / "intrinsic_depth.txt", np.eye(4))
    frame_inputs: dict[str, dict[str, str]] = {}
    for source_id in (0, 10):
        color = source / "color" / f"{source_id}.jpg"
        depth = source / "depth" / f"{source_id}.png"
        pose = source / "pose" / f"{source_id}.txt"
        Image.fromarray(np.full((4, 4, 3), 20 + source_id, dtype=np.uint8)).save(color)
        Image.fromarray(np.full((2, 2), 1000 + source_id, dtype=np.uint16)).save(depth)
        matrix = np.eye(4)
        matrix[0, 3] = source_id / 10
        np.savetxt(pose, matrix)
        frame_inputs[str(source_id)] = {
            "color_sha256": _sha256(color),
            "depth_sha256": _sha256(depth),
            "pose_sha256": _sha256(pose),
        }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "oviv2_scannet200_5_stage4_v1",
                "dataset": "ScanNet200",
                "scenes": [
                    {
                        "scene": "scene0011_00",
                        "dataset_root": str(source),
                        "source_frame_ids": [0, 10],
                        "frame_count": 2,
                        "frame_inputs": frame_inputs,
                        "image_shape": [2, 2],
                        "depth_scale": 1000.0,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_materialize_scene_writes_contiguous_hash_bound_view(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "view" / "scene0011_00"

    result = materialize_scene(manifest, "scene0011_00", output)

    assert result["scene"] == "scene0011_00"
    assert result["source_frame_ids"] == [0, 10]
    assert result["frame_count"] == 2
    assert (output / "results" / "frame000000.jpg").is_file()
    assert (output / "results" / "frame000001.jpg").is_file()
    assert (output / "results" / "depth000000.png").is_file()
    assert (output / "results" / "depth000001.png").is_file()
    with Image.open(output / "results" / "frame000001.jpg") as image:
        assert image.size == (2, 2)
    with Image.open(output / "results" / "depth000001.png") as image:
        np.testing.assert_array_equal(np.asarray(image), np.full((2, 2), 1010))
    trajectory = np.loadtxt(output / "traj.txt").reshape(-1, 4, 4)
    assert trajectory.shape == (2, 4, 4)
    np.testing.assert_allclose(trajectory[1, 0, 3], 1.0)
    published = json.loads((output / "view_manifest.json").read_text(encoding="utf-8"))
    assert published == result
    assert set(published["output_files_sha256"]) == {
        "results/depth000000.png",
        "results/depth000001.png",
        "results/frame000000.jpg",
        "results/frame000001.jpg",
        "traj.txt",
    }


def test_materialize_scene_rejects_manifest_frame_count_mismatch(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scenes"][0]["frame_count"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="frame_count"):
        materialize_scene(manifest_path, "scene0011_00", tmp_path / "view")
