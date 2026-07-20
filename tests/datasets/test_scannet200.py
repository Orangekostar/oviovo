from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.datasets.scannet200 import ScanNet200Dataset


def _write_scene(root: Path, source_ids: tuple[int, ...] = (0, 10)) -> None:
    for name in ("color", "depth", "pose", "intrinsic"):
        (root / name).mkdir(parents=True, exist_ok=True)
    intrinsic = np.array(
        [
            [2.0, 0.0, 0.5, 0.0],
            [0.0, 2.0, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.savetxt(root / "intrinsic" / "intrinsic_depth.txt", intrinsic)
    for source_id in source_ids:
        rgb = np.full((4, 4, 3), source_id + 20, dtype=np.uint8)
        depth = np.full((2, 2), source_id + 1000, dtype=np.uint16)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = source_id / 10.0
        Image.fromarray(rgb).save(root / "color" / f"{source_id}.jpg")
        Image.fromarray(depth).save(root / "depth" / f"{source_id}.png")
        np.savetxt(root / "pose" / f"{source_id}.txt", pose)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_hashes(root: Path, source_ids: tuple[int, ...]) -> dict[int, dict[str, str]]:
    return {
        source_id: {
            "color": _sha256(root / "color" / f"{source_id}.jpg"),
            "depth": _sha256(root / "depth" / f"{source_id}.png"),
            "pose": _sha256(root / "pose" / f"{source_id}.txt"),
        }
        for source_id in source_ids
    }


def test_scannet200_dataset_loads_explicit_source_frames(tmp_path: Path) -> None:
    _write_scene(tmp_path)

    dataset = ScanNet200Dataset(
        tmp_path,
        source_frame_ids=(0, 10),
        expected_image_shape=(2, 2),
    )

    assert len(dataset) == 2
    assert dataset.frame_indices == (0, 10)
    frame = dataset[1]
    assert frame.frame_id == 10
    assert frame.source_frame_id == 10
    assert frame.rgb.shape == (2, 2, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.depth.shape == (2, 2)
    assert frame.depth.dtype == np.float32
    np.testing.assert_allclose(frame.depth, 1.01)
    np.testing.assert_allclose(frame.pose[:3, 3], (1.0, 0.0, 0.0))
    assert frame.intrinsics.width == 2
    assert frame.intrinsics.height == 2
    assert frame.intrinsics.fx == 2.0
    assert frame.intrinsics.fy == 2.0


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_scannet200_dataset_rejects_nonfinite_pose(
    tmp_path: Path,
    invalid_value: float,
) -> None:
    _write_scene(tmp_path)
    pose_path = tmp_path / "pose" / "10.txt"
    pose = np.loadtxt(pose_path)
    pose[0, 0] = invalid_value
    np.savetxt(pose_path, pose)

    with pytest.raises(ValueError, match="finite pose"):
        ScanNet200Dataset(
            tmp_path,
            source_frame_ids=(0, 10),
            expected_image_shape=(2, 2),
        )


def test_scannet200_dataset_rejects_noninvertible_pose(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    pose_path = tmp_path / "pose" / "10.txt"
    pose = np.loadtxt(pose_path)
    pose[2] = 0.0
    np.savetxt(pose_path, pose)

    with pytest.raises(ValueError, match="invertible pose"):
        ScanNet200Dataset(
            tmp_path,
            source_frame_ids=(0, 10),
            expected_image_shape=(2, 2),
        )


def test_scannet200_dataset_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    _write_scene(tmp_path)

    with pytest.raises(ValueError, match="unique source frame IDs"):
        ScanNet200Dataset(
            tmp_path,
            source_frame_ids=(0, 0),
            expected_image_shape=(2, 2),
        )


def test_scannet200_dataset_rejects_symlinked_frame_input(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    color = tmp_path / "color" / "10.jpg"
    target = tmp_path / "color" / "target.jpg"
    color.rename(target)
    color.symlink_to(target)

    with pytest.raises(ValueError, match="regular non-symlink"):
        ScanNet200Dataset(
            tmp_path,
            source_frame_ids=(0, 10),
            expected_image_shape=(2, 2),
        )


def test_scannet200_dataset_rejects_frame_hash_mismatch(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    hashes = _input_hashes(tmp_path, (0, 10))
    hashes[10]["depth"] = "0" * 64

    with pytest.raises(ValueError, match="depth hash mismatch"):
        ScanNet200Dataset(
            tmp_path,
            source_frame_ids=(0, 10),
            input_hashes=hashes,
            expected_image_shape=(2, 2),
        )


def test_scannet200_dataset_rejects_wrong_depth_shape(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    Image.fromarray(np.ones((3, 2), dtype=np.uint16)).save(
        tmp_path / "depth" / "10.png"
    )

    with pytest.raises(ValueError, match="depth shape"):
        ScanNet200Dataset(
            tmp_path,
            source_frame_ids=(0, 10),
            expected_image_shape=(2, 2),
        )
