"""Tests for the Replica dataset adapter."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import CameraIntrinsics, Frame
from src.datasets import ReplicaRoom0Dataset


def _write_rgb(path: Path, shape: tuple[int, int, int], value: int) -> None:
    rgb = np.full(shape, value, dtype=np.uint8)
    Image.fromarray(rgb).save(path)


def _write_depth(path: Path, shape: tuple[int, int], value: int) -> None:
    depth = np.full(shape, value, dtype=np.uint16)
    Image.fromarray(depth).save(path)


def _write_traj(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            pose = np.eye(4, dtype=np.float64)
            pose[0, 3] = index * 0.25
            handle.write(" ".join(str(x) for x in pose.reshape(-1)) + "\n")


def _write_manifest(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["idx", "rgb", "depth", "pose16"])
        writer.writeheader()
        for index in range(count):
            pose = np.eye(4, dtype=np.float64)
            pose[1, 3] = 1.0 + index * 0.5
            writer.writerow(
                {
                    "idx": index,
                    "rgb": f"/legacy/frame{index:06d}.jpg",
                    "depth": f"/legacy/depth{index:06d}.png",
                    "pose16": " ".join(str(x) for x in pose.reshape(-1)),
                }
            )


def test_replica_dataset_loads_flat_layout(tmp_path: Path) -> None:
    root = tmp_path / "room0"
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)

    _write_rgb(root / "rgb" / "frame000000.jpg", (4, 6, 3), 10)
    _write_rgb(root / "rgb" / "frame000001.jpg", (4, 6, 3), 20)
    _write_depth(root / "depth" / "depth000000.png", (4, 6), 2000)
    _write_depth(root / "depth" / "depth000001.png", (4, 6), 3000)
    _write_traj(root / "traj.txt", count=2)

    intrinsics = CameraIntrinsics(fx=5.0, fy=5.0, cx=2.5, cy=1.5, width=6, height=4)
    dataset = ReplicaRoom0Dataset(root, intrinsics=intrinsics, depth_scale=1000.0)

    assert len(dataset) == 2
    frame = dataset[1]
    assert isinstance(frame, Frame)
    assert frame.frame_id == 1
    assert frame.rgb.shape == (4, 6, 3)
    assert frame.depth.shape == (4, 6)
    assert np.isclose(frame.depth[0, 0], 3.0)
    assert np.isclose(frame.pose[0, 3], 0.25)
    assert list(dataset.frame_indices) == [0, 1]

    limited_frames = list(dataset.iter_frames(limit=1))
    assert len(limited_frames) == 1
    assert limited_frames[0].frame_id == 0


def test_replica_dataset_prefers_manifest_pose_when_present(tmp_path: Path) -> None:
    root = tmp_path / "room0"
    results = root / "results"
    results.mkdir(parents=True)

    _write_rgb(results / "frame000000.jpg", (4, 6, 3), 30)
    _write_depth(results / "depth000000.png", (4, 6), 4000)
    _write_traj(root / "traj.txt", count=1)
    _write_manifest(root / "frame_manifest.txt", count=1)

    intrinsics = CameraIntrinsics(fx=5.0, fy=5.0, cx=2.5, cy=1.5, width=6, height=4)
    dataset = ReplicaRoom0Dataset(root, intrinsics=intrinsics, depth_scale=1000.0)

    frame = dataset[0]
    assert frame.frame_id == 0
    assert np.isclose(frame.pose[1, 3], 1.0)
    assert dataset.summary()["layout"] == "results_dir"
    assert dataset.summary()["pose_source"] == "frame_manifest.txt"
