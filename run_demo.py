#!/usr/bin/env python3
"""Minimal runnable demo for the OVIOVO backbone.

Generates fake RGB-D frames and runs them through the full pipeline,
printing intermediate outputs at each stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.core.data_structures import CameraIntrinsics
from src.pipelines.main_pipeline import Pipeline


def generate_fake_frame(
    frame_id: int,
    height: int = 480,
    width: int = 640,
) -> tuple:
    """Generate a synthetic RGB-D frame with camera parameters.

    Returns:
        (rgb, depth, pose, intrinsics, timestamp)
    """
    # Random RGB image
    rgb = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Synthetic depth: a plane at ~2m with some noise
    depth = np.full((height, width), 2.0, dtype=np.float32)
    depth += np.random.normal(0, 0.1, depth.shape).astype(np.float32)
    depth = np.clip(depth, 0.1, 10.0)

    # Camera pose: slight translation per frame
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = frame_id * 0.1  # move 10cm per frame along x

    # Standard pinhole intrinsics
    intrinsics = CameraIntrinsics(
        fx=525.0, fy=525.0,
        cx=width / 2.0, cy=height / 2.0,
        width=width, height=height,
    )

    timestamp = frame_id * 0.033  # ~30fps

    return rgb, depth, pose, intrinsics, timestamp


def main():
    print("=" * 60)
    print("OVIOVO Backbone Demo")
    print("=" * 60)

    # Initialize pipeline
    config_path = str(project_root / "configs" / "default.yaml")
    pipeline = Pipeline(config_path=config_path)

    # Process a few fake frames
    n_frames = 5
    for i in range(n_frames):
        rgb, depth, pose, intrinsics, ts = generate_fake_frame(i)
        state = pipeline.process_frame(rgb, depth, pose, intrinsics, timestamp=ts)

        # Print summary
        print(f"\n--- After frame {i} ---")
        print(f"  Total objects:     {len(state.objects)}")
        print(f"  Background points: {len(state.background.point_cloud)}")
        for obj_id, obj in state.objects.items():
            print(
                f"  Object {obj_id}: state={obj.state.value}, "
                f"points={len(obj.points)}, "
                f"observations={len(obj.observations)}, "
                f"semantic_obs={obj.semantic_memory.observation_count}"
            )

    print("\n" + "=" * 60)
    print("Demo complete. All modules executed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
