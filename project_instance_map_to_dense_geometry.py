#!/usr/bin/env python3
"""Fuse dense geometry from RGB-D and project stable instance labels onto it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.datasets import ReplicaRoom0Dataset


VERTEX_DTYPE = np.dtype(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Directory containing instance_map.ply and the analysis summary.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "input" / "Datasets" / "Replica" / "room0",
    )
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--geometry-voxel-size", type=float, default=0.03)
    parser.add_argument("--instance-voxel-size", type=float, default=0.05)
    parser.add_argument("--projection-neighbor-radius", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = analysis_dir / "dense_geometry_projection"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ReplicaRoom0Dataset(args.dataset_root)
    stable_instances = read_binary_vertex_ply(analysis_dir / "instance_map.ply")

    dense_points, dense_colors = fuse_dense_geometry(
        dataset=dataset,
        num_frames=args.num_frames,
        sample_stride=max(1, int(args.sample_stride)),
        voxel_size=float(args.geometry_voxel_size),
    )
    labels, state_ids, supports = project_instances_to_dense_points(
        dense_points=dense_points,
        stable_instances=stable_instances,
        instance_voxel_size=float(args.instance_voxel_size),
        neighbor_radius=max(0, int(args.projection_neighbor_radius)),
    )

    projected_colors = np.asarray(
        [instance_color(int(object_id)) if object_id >= 0 else (180, 180, 180) for object_id in labels],
        dtype=np.uint8,
    )

    fused_rgb_path = output_dir / "dense_geometry_fused_rgb.ply"
    projected_path = output_dir / "dense_geometry_instance_projected.ply"
    write_vertex_ply(
        fused_rgb_path,
        dense_points,
        dense_colors,
        np.full(len(dense_points), -1, dtype=np.int32),
        np.zeros(len(dense_points), dtype=np.uint8),
        np.zeros(len(dense_points), dtype=np.float32),
    )
    write_vertex_ply(
        projected_path,
        dense_points,
        projected_colors,
        labels.astype(np.int32),
        state_ids.astype(np.uint8),
        supports.astype(np.float32),
    )

    rgb_xz = output_dir / "preview_xz_rgb.png"
    inst_xz = output_dir / "preview_xz_instance.png"
    rgb_xy = output_dir / "preview_xy_rgb.png"
    inst_xy = output_dir / "preview_xy_instance.png"
    render_projection(dense_points, dense_colors, plane="xz", output_path=rgb_xz)
    render_projection(dense_points, projected_colors, plane="xz", output_path=inst_xz)
    render_projection(dense_points, dense_colors, plane="xy", output_path=rgb_xy)
    render_projection(dense_points, projected_colors, plane="xy", output_path=inst_xy)
    render_preview_sheet([rgb_xz, inst_xz, rgb_xy, inst_xy], output_dir / "preview_sheet.png")

    labeled_mask = labels >= 0
    summary = {
        "analysis_dir": str(analysis_dir),
        "dataset_root": str(args.dataset_root),
        "num_frames": int(args.num_frames),
        "sample_stride": int(args.sample_stride),
        "geometry_voxel_size": float(args.geometry_voxel_size),
        "instance_voxel_size": float(args.instance_voxel_size),
        "projection_neighbor_radius": int(args.projection_neighbor_radius),
        "files": {
            "dense_geometry_fused_rgb_ply": str(fused_rgb_path),
            "dense_geometry_instance_projected_ply": str(projected_path),
            "preview_xz_rgb_png": str(rgb_xz),
            "preview_xz_instance_png": str(inst_xz),
            "preview_xy_rgb_png": str(rgb_xy),
            "preview_xy_instance_png": str(inst_xy),
            "preview_sheet_png": str(output_dir / "preview_sheet.png"),
        },
        "counts": {
            "dense_geometry_point_count": int(len(dense_points)),
            "stable_instance_point_count": int(len(stable_instances)),
            "labeled_dense_point_count": int(np.count_nonzero(labeled_mask)),
            "unlabeled_dense_point_count": int(np.count_nonzero(~labeled_mask)),
            "label_coverage_ratio": float(np.count_nonzero(labeled_mask) / max(len(dense_points), 1)),
            "projected_object_count": int(len(set(labels[labeled_mask].tolist())) if np.any(labeled_mask) else 0),
        },
    }
    summary_path = output_dir / "projection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved dense geometry projection outputs:")
    print(f"- {fused_rgb_path}")
    print(f"- {projected_path}")
    print(f"- {output_dir / 'preview_sheet.png'}")
    print(f"- {summary_path}")


def fuse_dense_geometry(
    dataset: ReplicaRoom0Dataset,
    num_frames: int,
    sample_stride: int,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    accum: dict[tuple[int, int, int], list[np.ndarray | int]] = {}

    for count, frame in enumerate(dataset.iter_frames(limit=num_frames), start=1):
        rgb = frame.rgb[::sample_stride, ::sample_stride]
        depth = frame.depth[::sample_stride, ::sample_stride]
        h, w = depth.shape
        u, v = np.meshgrid(
            np.arange(0, frame.rgb.shape[1], sample_stride),
            np.arange(0, frame.rgb.shape[0], sample_stride),
        )
        valid = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid):
            continue

        z = depth[valid].astype(np.float32)
        u = u[valid].astype(np.float32)
        v = v[valid].astype(np.float32)
        x = (u - frame.intrinsics.cx) * z / frame.intrinsics.fx
        y = (v - frame.intrinsics.cy) * z / frame.intrinsics.fy
        points_cam = np.stack([x, y, z], axis=1)
        R = frame.pose[:3, :3].astype(np.float32)
        t = frame.pose[:3, 3].astype(np.float32)
        points_world = (R @ points_cam.T).T + t
        colors = rgb[valid].astype(np.float32)

        voxel_indices = np.floor(points_world / voxel_size).astype(np.int32)
        unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
        counts_per_voxel = np.bincount(inverse)

        sum_points = np.zeros((len(unique_voxels), 3), dtype=np.float64)
        sum_colors = np.zeros((len(unique_voxels), 3), dtype=np.float64)
        for axis in range(3):
            sum_points[:, axis] = np.bincount(inverse, weights=points_world[:, axis], minlength=len(unique_voxels))
            sum_colors[:, axis] = np.bincount(inverse, weights=colors[:, axis], minlength=len(unique_voxels))

        mean_points = (sum_points / counts_per_voxel[:, None]).astype(np.float32)
        mean_colors = (sum_colors / counts_per_voxel[:, None]).astype(np.float32)

        for voxel, point, color, local_count in zip(unique_voxels, mean_points, mean_colors, counts_per_voxel):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            if key not in accum:
                accum[key] = [point.astype(np.float64), color.astype(np.float64), int(local_count)]
            else:
                prev_point, prev_color, prev_count = accum[key]
                total = prev_count + int(local_count)
                accum[key][0] = (prev_point * prev_count + point * local_count) / total
                accum[key][1] = (prev_color * prev_count + color * local_count) / total
                accum[key][2] = total

        if count == 1 or count % 20 == 0 or count == num_frames:
            print(f"geometry frame {count:03d}/{num_frames}: voxels={len(accum)}")

    points = np.asarray([item[0] for item in accum.values()], dtype=np.float32)
    colors = np.clip(np.asarray([item[1] for item in accum.values()], dtype=np.float32), 0, 255).astype(np.uint8)
    return points, colors


def project_instances_to_dense_points(
    dense_points: np.ndarray,
    stable_instances: np.ndarray,
    instance_voxel_size: float,
    neighbor_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    owner_map: dict[tuple[int, int, int], tuple[int, int, float]] = {}
    stable_points = np.stack([stable_instances["x"], stable_instances["y"], stable_instances["z"]], axis=1)
    stable_indices = np.floor((stable_points / instance_voxel_size)).astype(np.int32)

    for voxel, row in zip(stable_indices, stable_instances):
        owner_map[(int(voxel[0]), int(voxel[1]), int(voxel[2]))] = (
            int(row["object_id"]),
            int(row["state_id"]),
            float(row["support"]),
        )

    labels = np.full(len(dense_points), -1, dtype=np.int32)
    state_ids = np.zeros(len(dense_points), dtype=np.uint8)
    supports = np.zeros(len(dense_points), dtype=np.float32)
    dense_indices = np.floor((dense_points / instance_voxel_size)).astype(np.int32)

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in range(-neighbor_radius, neighbor_radius + 1)
        for dy in range(-neighbor_radius, neighbor_radius + 1)
        for dz in range(-neighbor_radius, neighbor_radius + 1)
    ]

    for idx, voxel in enumerate(dense_indices):
        key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        record = owner_map.get(key)
        if record is None and neighbor_radius > 0:
            best = None
            best_dist = float("inf")
            for dx, dy, dz in neighbor_offsets:
                nkey = (key[0] + dx, key[1] + dy, key[2] + dz)
                candidate = owner_map.get(nkey)
                if candidate is None:
                    continue
                center = (np.asarray(nkey, dtype=np.float32) + 0.5) * instance_voxel_size
                dist = float(np.linalg.norm(dense_points[idx] - center))
                if dist < best_dist:
                    best_dist = dist
                    best = candidate
            record = best

        if record is not None:
            labels[idx] = record[0]
            state_ids[idx] = record[1]
            supports[idx] = record[2]

    return labels, state_ids, supports


def write_vertex_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    state_ids: np.ndarray,
    supports: np.ndarray,
) -> None:
    vertex_array = np.zeros(len(points), dtype=VERTEX_DTYPE)
    vertex_array["x"] = points[:, 0]
    vertex_array["y"] = points[:, 1]
    vertex_array["z"] = points[:, 2]
    vertex_array["red"] = colors[:, 0]
    vertex_array["green"] = colors[:, 1]
    vertex_array["blue"] = colors[:, 2]
    vertex_array["object_id"] = object_ids
    vertex_array["state_id"] = state_ids
    vertex_array["support"] = supports

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


def read_binary_vertex_ply(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading PLY header: {path}")
            header_lines.append(line.decode("ascii").strip())
            if header_lines[-1] == "end_header":
                break
        vertex_count = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                break
        return np.fromfile(handle, dtype=VERTEX_DTYPE, count=vertex_count)


def render_projection(
    points: np.ndarray,
    colors: np.ndarray,
    plane: str,
    output_path: Path,
    canvas_size: tuple[int, int] = (1400, 900),
) -> None:
    if plane == "xz":
        coords = points[:, [0, 2]]
        depth_axis = points[:, 1]
    elif plane == "xy":
        coords = points[:, [0, 1]]
        depth_axis = points[:, 2]
    else:
        raise ValueError(f"Unsupported plane: {plane}")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    normalized = (coords - mins) / span

    width, height = canvas_size
    u = np.clip((normalized[:, 0] * (width - 1)).astype(np.int32), 0, width - 1)
    v = np.clip(((1.0 - normalized[:, 1]) * (height - 1)).astype(np.int32), 0, height - 1)
    order = np.argsort(depth_axis)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for index in order:
        uu = u[index]
        vv = v[index]
        color = colors[index]
        u0 = max(uu - 1, 0)
        u1 = min(uu + 2, width)
        v0 = max(vv - 1, 0)
        v1 = min(vv + 2, height)
        canvas[v0:v1, u0:u1] = color
    Image.fromarray(canvas).save(output_path)


def render_preview_sheet(image_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        width, height = images[0].size
        sheet = Image.new("RGB", (2 * width, 2 * height), color=(255, 255, 255))
        positions = [(0, 0), (width, 0), (0, height), (width, height)]
        for image, pos in zip(images, positions):
            sheet.paste(image, pos)
        sheet.save(output_path)
    finally:
        for image in images:
            image.close()


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


if __name__ == "__main__":
    main()
