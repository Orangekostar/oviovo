#!/usr/bin/env python3
"""Re-run Replica room0 and export instance map PLY files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from run_replica_room0_analysis import build_proposal_module
from src.core.data_structures import ObjectMap, ObjectState, SystemState
from src.datasets import ReplicaRoom0Dataset
from src.pipelines.main_pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "input" / "Datasets" / "Replica" / "room0",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=project_root / "configs" / "default.yaml",
    )
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Existing experiment directory where PLY files should be written.",
    )
    parser.add_argument("--proposal-backend", type=str, default="sam2")
    parser.add_argument("--proposal-device", type=str, default="cuda")
    parser.add_argument("--sam-version", type=str, default="2")
    parser.add_argument("--sam-encoder", type=str, default="hiera_l")
    parser.add_argument("--sam-repo-root", type=Path, default=None)
    parser.add_argument(
        "--sam-ckpt-path",
        type=Path,
        default=Path("/home/phl/vv/paper2/DovSG/checkpoints/segment-anything-2"),
    )
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument("--stability-score-th", type=float, default=0.95)
    parser.add_argument("--nms-iou-th", type=float, default=0.8)
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    parser.add_argument("--pipeline-verbose", action="store_true")
    parser.add_argument(
        "--local-memory-downsample-voxel",
        type=float,
        default=0.0,
        help="Optional voxel downsample size for exported local-memory points. 0 disables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ReplicaRoom0Dataset(args.dataset_root)
    pipeline = Pipeline(config_path=str(args.config_path))
    pipeline.verbose = bool(args.pipeline_verbose)
    pipeline.proposal = build_proposal_module(pipeline.proposal.config, args)

    print("=" * 60)
    print("Replica room0 instance-map export")
    print("=" * 60)
    print(f"dataset_root: {args.dataset_root}")
    print(f"output_dir:   {output_dir}")
    print(f"num_frames:   {args.num_frames}")
    print(f"backend:      {args.proposal_backend}")
    print(f"device:       {args.proposal_device}")

    processed = 0
    for frame in dataset.iter_frames(limit=args.num_frames):
        pipeline.process_frame(
            frame.rgb,
            frame.depth,
            frame.pose,
            frame.intrinsics,
            timestamp=frame.timestamp,
        )
        processed += 1
        if processed == 1 or processed % 10 == 0 or processed == args.num_frames:
            print(
                f"frame={frame.frame_id:04d} processed={processed:03d} "
                f"objects={len(pipeline.state.objects):03d} "
                f"backend={pipeline.proposal.active_backend_name}"
            )

    state = pipeline.state

    tsdf_records = build_tsdf_backbone_records(state)
    local_memory_records = build_local_memory_records(
        state,
        downsample_voxel=float(args.local_memory_downsample_voxel),
    )

    instance_map_path = output_dir / "instance_map.ply"
    tsdf_backbone_path = output_dir / "instance_map_tsdf_backbone.ply"
    local_memory_path = output_dir / "instance_map_local_memory.ply"

    write_binary_ply(instance_map_path, tsdf_records)
    write_binary_ply(tsdf_backbone_path, tsdf_records)
    write_binary_ply(local_memory_path, local_memory_records)

    summary = {
        "dataset_summary": dataset.summary(),
        "run": {
            "num_frames_requested": int(args.num_frames),
            "num_frames_processed": int(processed),
            "python_executable": sys.executable,
        },
        "proposal_backend": {
            "requested_backend": pipeline.proposal.requested_backend_name,
            "active_backend": pipeline.proposal.active_backend_name,
            "backend_init_error": pipeline.proposal.backend_init_error,
            "last_generation_info": sanitize(getattr(pipeline.proposal.backend, "last_generation_info", {})),
        },
        "files": {
            "instance_map_ply": str(instance_map_path),
            "instance_map_tsdf_backbone_ply": str(tsdf_backbone_path),
            "instance_map_local_memory_ply": str(local_memory_path),
        },
        "counts": {
            "final_object_count": int(len(state.objects)),
            "tsdf_backbone_vertex_count": int(len(tsdf_records)),
            "local_memory_vertex_count": int(len(local_memory_records)),
            "active_state_count": int(sum(obj.state == ObjectState.ACTIVE for obj in state.objects.values())),
            "dormant_state_count": int(sum(obj.state == ObjectState.DORMANT for obj in state.objects.values())),
            "inactive_state_count": int(sum(obj.state == ObjectState.INACTIVE for obj in state.objects.values())),
        },
    }
    summary_path = output_dir / "instance_map_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved files:")
    print(f"- {instance_map_path}")
    print(f"- {tsdf_backbone_path}")
    print(f"- {local_memory_path}")
    print(f"- {summary_path}")


def build_tsdf_backbone_records(state: SystemState) -> np.ndarray:
    rows = []
    voxel_size = state.tsdf_volume.voxel_size
    for voxel_key, owner_support in state.tsdf_volume.owner_support.items():
        object_id = int(owner_support.owner_id)
        if object_id < 0 or object_id not in state.objects:
            continue
        obj = state.objects[object_id]
        point = (np.asarray(voxel_key, dtype=np.float32) + 0.5) * float(voxel_size)
        color = instance_color(object_id)
        rows.append(
            (
                float(point[0]),
                float(point[1]),
                float(point[2]),
                color[0],
                color[1],
                color[2],
                object_id,
                int(state_id(obj.state)),
                float(owner_support.owner_support_value),
            )
        )
    return to_vertex_array(rows)


def build_local_memory_records(state: SystemState, downsample_voxel: float = 0.0) -> np.ndarray:
    rows = []
    for object_id, obj in sorted(state.objects.items()):
        points = np.asarray(obj.local_pcd, dtype=np.float32)
        if points.size == 0:
            continue
        if downsample_voxel > 0.0:
            points = voxel_downsample_points(points, downsample_voxel)
        color = instance_color(object_id)
        stability = float(obj.debug.get("global_instance_substrate", {}).get("stability_score", 0.0))
        for point in points:
            rows.append(
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    color[0],
                    color[1],
                    color[2],
                    int(object_id),
                    int(state_id(obj.state)),
                    stability,
                )
            )
    return to_vertex_array(rows)


def to_vertex_array(rows: list[tuple[Any, ...]]) -> np.ndarray:
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("object_id", "<i4"),
            ("state_id", "u1"),
            ("support", "<f4"),
        ]
    )
    return np.asarray(rows, dtype=dtype)


def write_binary_ply(path: Path, vertex_array: np.ndarray) -> None:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(vertex_array)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int object_id",
            "property uchar state_id",
            "property float support",
            "end_header",
            "",
        ]
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertex_array.tobytes())


def state_id(state: ObjectState) -> int:
    mapping = {
        ObjectState.ACTIVE: 1,
        ObjectState.INACTIVE: 2,
        ObjectState.GHOST: 3,
        ObjectState.REMOVED: 4,
        ObjectState.DORMANT: 5,
    }
    return mapping.get(state, 0)


def instance_color(object_id: int) -> tuple[int, int, int]:
    palette = [
        (255, 99, 71),
        (0, 191, 255),
        (60, 179, 113),
        (255, 215, 0),
        (186, 85, 211),
        (255, 140, 0),
        (64, 224, 208),
        (220, 20, 60),
        (123, 104, 238),
        (46, 139, 87),
        (70, 130, 180),
        (244, 162, 97),
    ]
    return palette[object_id % len(palette)]


def voxel_downsample_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    unique_indices.sort()
    return points[unique_indices]


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
