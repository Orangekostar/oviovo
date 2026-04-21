#!/usr/bin/env python3
"""Minimal real-data demo for the Replica room0 adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure project root is importable
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.core.data_structures import Frame
from src.datasets import ReplicaRoom0Dataset
from src.utils.geometry import depth_to_points

DATASET_ROOT = project_root / "data" / "input" / "Datasets" / "Replica" / "room0"
OUTPUT_DIR = project_root / "outputs" / "replica_room0_demo"


def save_rgb_visualization(frame: Frame, path: Path) -> None:
    Image.fromarray(frame.rgb).save(path)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0.0
    color = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return color

    lo, hi = np.percentile(depth[valid], [2.0, 98.0])
    hi = max(float(hi), float(lo) + 1e-6)

    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)

    red = (norm * 255.0).astype(np.uint8)
    green = (np.sqrt(norm) * 255.0).astype(np.uint8)
    blue = ((1.0 - norm) * 255.0).astype(np.uint8)
    color[..., 0] = red
    color[..., 1] = green
    color[..., 2] = blue
    return color


def save_depth_visualization(frame: Frame, path: Path) -> None:
    Image.fromarray(colorize_depth(frame.depth)).save(path)


def save_point_cloud_summary(frame: Frame, path: Path) -> None:
    stride = 8
    mask = np.zeros(frame.depth.shape, dtype=bool)
    mask[::stride, ::stride] = frame.depth[::stride, ::stride] > 0.0

    points = depth_to_points(frame.depth, frame.intrinsics, frame.pose, mask=mask)
    if len(points) == 0:
        return

    canvas_size = 768
    xy = points[:, :2]
    z = points[:, 2]

    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)

    coords = (xy - mins) / span
    u = np.clip((coords[:, 0] * (canvas_size - 1)).astype(np.int32), 0, canvas_size - 1)
    v = np.clip(((1.0 - coords[:, 1]) * (canvas_size - 1)).astype(np.int32), 0, canvas_size - 1)

    z_lo, z_hi = np.percentile(z, [2.0, 98.0])
    z_hi = max(float(z_hi), float(z_lo) + 1e-6)
    z_norm = np.clip((z - z_lo) / (z_hi - z_lo), 0.0, 1.0)

    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    canvas[v, u, 0] = (z_norm * 255.0).astype(np.uint8)
    canvas[v, u, 1] = ((1.0 - np.abs(z_norm - 0.5) * 2.0) * 255.0).astype(np.uint8)
    canvas[v, u, 2] = ((1.0 - z_norm) * 255.0).astype(np.uint8)
    Image.fromarray(canvas).save(path)


def main() -> None:
    dataset = ReplicaRoom0Dataset(DATASET_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Replica room0 dataset summary")
    print("=" * 60)
    for key, value in dataset.summary().items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("First 10 frames")
    print("=" * 60)

    frames = list(dataset.iter_frames(limit=10))
    for frame in frames:
        print(f"\nframe index: {frame.frame_id}")
        print(f"rgb shape:    {frame.rgb.shape}")
        print(f"depth shape:  {frame.depth.shape}")
        print("pose:")
        print(np.array2string(frame.pose, precision=5, suppress_small=True))

    if frames:
        first_frame = frames[0]
        save_rgb_visualization(first_frame, OUTPUT_DIR / "frame000000_rgb.png")
        save_depth_visualization(first_frame, OUTPUT_DIR / "frame000000_depth.png")
        save_point_cloud_summary(first_frame, OUTPUT_DIR / "frame000000_pointcloud_xy.png")

        print("\nSaved visualizations:")
        print(f"- {OUTPUT_DIR / 'frame000000_rgb.png'}")
        print(f"- {OUTPUT_DIR / 'frame000000_depth.png'}")
        print(f"- {OUTPUT_DIR / 'frame000000_pointcloud_xy.png'}")


if __name__ == "__main__":
    main()
