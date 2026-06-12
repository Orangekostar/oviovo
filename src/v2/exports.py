"""Export helpers for the simplified V2 semantic mapping pipeline."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.v2.types import CoarseVoxelOwner, ObjectPool, SemanticMapV2State, V2ObjectState

from run_room0_full_eval import (
    PLY_DTYPE,
    evaluate_semantics,
    instance_color,
    read_class_names,
    read_gt_labels,
    read_gt_vertices,
    render_preview_sheet,
    render_projection,
    semantic_surface_color,
    write_binary_ply,
    write_eval_artifacts,
    write_vertex_ply,
)


@dataclass
class RoomSemanticProjection:
    dense_points: np.ndarray
    dense_colors: np.ndarray
    labels: np.ndarray
    state_ids: np.ndarray
    supports: np.ndarray


def v2_state_id(state: V2ObjectState) -> int:
    if state == V2ObjectState.ACTIVE:
        return 1
    if state == V2ObjectState.PROVISIONAL:
        return 2
    if state == V2ObjectState.ARCHIVED:
        return 5
    return 0


def finalize_geometry_accum(accum: dict[tuple[int, int, int], list[np.ndarray | int]]) -> tuple[np.ndarray, np.ndarray]:
    if not accum:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    points = np.asarray([item[0] for item in accum.values()], dtype=np.float32)
    colors = np.clip(np.asarray([item[1] for item in accum.values()], dtype=np.float32), 0, 255).astype(np.uint8)
    return points, colors


def build_coarse_voxel_records(state: SemanticMapV2State, voxel_size: float) -> np.ndarray:
    rows = []
    for voxel_key, owner in sorted(state.coarse_owners.items()):
        point = (np.asarray(voxel_key, dtype=np.float32) + 0.5) * float(voxel_size)
        color = semantic_surface_color(owner.canonical_label, int(owner.owner_object_id))
        rows.append(
            (
                float(point[0]),
                float(point[1]),
                float(point[2]),
                color[0],
                color[1],
                color[2],
                int(owner.owner_object_id),
                int(v2_state_id(state.object_pools.get(owner.owner_object_id, ObjectPool(-1)).state))
                if owner.owner_object_id in state.object_pools
                else 0,
                float(owner.owner_confidence),
            )
        )
    return np.asarray(rows, dtype=PLY_DTYPE)


def build_object_pool_records(state: SemanticMapV2State) -> np.ndarray:
    rows = []
    for object_id, pool in sorted(state.object_pools.items()):
        if pool.points.size == 0:
            continue
        color = semantic_surface_color(pool.canonical_label, int(object_id))
        state_code = int(v2_state_id(pool.state))
        for point, support in zip(pool.points, pool.point_confidence):
            rows.append(
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    color[0],
                    color[1],
                    color[2],
                    int(object_id),
                    state_code,
                    float(support),
                )
            )
    return np.asarray(rows, dtype=PLY_DTYPE)


def build_object_pool_debug_records(state: SemanticMapV2State) -> np.ndarray:
    rows = []
    for object_id, pool in sorted(state.object_pools.items()):
        if pool.points.size == 0:
            continue
        color = instance_color(int(object_id))
        state_code = int(v2_state_id(pool.state))
        for point, support in zip(pool.points, pool.point_confidence):
            rows.append(
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    color[0],
                    color[1],
                    color[2],
                    int(object_id),
                    state_code,
                    float(support),
                )
            )
    return np.asarray(rows, dtype=PLY_DTYPE)


def project_room_semantic_map(
    state: SemanticMapV2State,
    dense_points: np.ndarray,
    *,
    coarse_voxel_size: float,
    neighbor_radius: int,
) -> RoomSemanticProjection:
    labels = np.full(len(dense_points), -1, dtype=np.int32)
    state_ids = np.zeros(len(dense_points), dtype=np.uint8)
    supports = np.zeros(len(dense_points), dtype=np.float32)
    if len(dense_points) == 0 or not state.coarse_owners:
        return RoomSemanticProjection(
            dense_points=dense_points,
            dense_colors=np.zeros((len(dense_points), 3), dtype=np.uint8),
            labels=labels,
            state_ids=state_ids,
            supports=supports,
        )

    owner_map = {
        key: (int(owner.owner_object_id), float(owner.owner_confidence))
        for key, owner in state.coarse_owners.items()
        if owner.owner_object_id >= 0
    }
    dense_indices = np.floor(dense_points / coarse_voxel_size).astype(np.int32)
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in range(-neighbor_radius, neighbor_radius + 1)
        for dy in range(-neighbor_radius, neighbor_radius + 1)
        for dz in range(-neighbor_radius, neighbor_radius + 1)
    ]
    dense_colors = np.full((len(dense_points), 3), 180, dtype=np.uint8)

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
                center = (np.asarray(nkey, dtype=np.float32) + 0.5) * coarse_voxel_size
                dist = float(np.linalg.norm(dense_points[idx] - center))
                if dist < best_dist:
                    best_dist = dist
                    best = candidate
            record = best
        if record is None:
            continue
        object_id, confidence = record
        pool = state.object_pools.get(int(object_id))
        labels[idx] = int(object_id)
        supports[idx] = float(confidence)
        if pool is not None:
            state_ids[idx] = int(v2_state_id(pool.state))
            dense_colors[idx] = semantic_surface_color(pool.canonical_label, int(object_id))
        else:
            dense_colors[idx] = instance_color(int(object_id))

    return RoomSemanticProjection(
        dense_points=dense_points,
        dense_colors=dense_colors,
        labels=labels,
        state_ids=state_ids,
        supports=supports,
    )


def export_standard_outputs(
    *,
    state: SemanticMapV2State,
    exports_dir: Path,
    coarse_voxel_size: float,
    neighbor_radius: int,
    dense_points: np.ndarray,
    dense_rgb: np.ndarray,
) -> tuple[np.ndarray, RoomSemanticProjection, np.ndarray, np.ndarray]:
    exports_dir.mkdir(parents=True, exist_ok=True)
    coarse_records = build_coarse_voxel_records(state, coarse_voxel_size)
    pool_records = build_object_pool_records(state)
    pool_debug_records = build_object_pool_debug_records(state)
    room_projection = project_room_semantic_map(
        state,
        dense_points,
        coarse_voxel_size=coarse_voxel_size,
        neighbor_radius=neighbor_radius,
    )

    coarse_map_path = exports_dir / "room0_coarse_voxel_semantic_map.ply"
    room_map_path = exports_dir / "room0_room_semantic_object_map.ply"
    fused_rgb_path = exports_dir / "room0_dense_geometry_fused_rgb.ply"
    object_pool_debug_dir = exports_dir / "object_pool_debug"
    object_pool_debug_dir.mkdir(parents=True, exist_ok=True)

    write_binary_ply(coarse_map_path, coarse_records)
    write_vertex_ply(
        room_map_path,
        room_projection.dense_points,
        room_projection.dense_colors,
        room_projection.labels.astype(np.int32),
        room_projection.state_ids.astype(np.uint8),
        room_projection.supports.astype(np.float32),
    )
    write_vertex_ply(
        fused_rgb_path,
        dense_points,
        dense_rgb,
        np.full(len(dense_points), -1, dtype=np.int32),
        np.zeros(len(dense_points), dtype=np.uint8),
        np.zeros(len(dense_points), dtype=np.float32),
    )

    combined_debug_path = object_pool_debug_dir / "room0_object_pool_debug_combined.ply"
    write_binary_ply(combined_debug_path, pool_debug_records)
    np.savez_compressed(
        exports_dir / "room0_coarse_voxel_semantic_map.npz",
        vertex_array=coarse_records,
    )
    np.savez_compressed(
        exports_dir / "room0_room_semantic_object_map.npz",
        points=room_projection.dense_points,
        colors=room_projection.dense_colors,
        object_ids=room_projection.labels,
        state_ids=room_projection.state_ids,
        supports=room_projection.supports,
    )
    for object_id, pool in sorted(state.object_pools.items()):
        if pool.points.size == 0:
            continue
        rows = []
        color = instance_color(int(object_id))
        for point, support in zip(pool.points, pool.point_confidence):
            rows.append(
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    color[0],
                    color[1],
                    color[2],
                    int(object_id),
                    int(v2_state_id(pool.state)),
                    float(support),
                )
            )
        write_binary_ply(
            object_pool_debug_dir / f"object_{int(object_id):03d}_{pool.canonical_label or 'unlabeled'}.ply",
            np.asarray(rows, dtype=PLY_DTYPE),
        )

    return coarse_records, room_projection, pool_records, pool_debug_records


def export_legacy_outputs(
    *,
    exports_dir: Path,
    coarse_records: np.ndarray,
    room_pool_records: np.ndarray,
    room_projection: RoomSemanticProjection,
    pool_debug_records: np.ndarray,
) -> dict[str, Path]:
    instance_map_path = exports_dir / "room0_instance_map.ply"
    dense_surface_path = exports_dir / "room0_instance_map_dense_surface.ply"
    tsdf_backbone_path = exports_dir / "room0_instance_map_tsdf_backbone.ply"
    local_memory_path = exports_dir / "room0_instance_map_local_memory.ply"
    dense_inst_path = exports_dir / "room0_dense_geometry_instance_projected.ply"

    write_binary_ply(instance_map_path, room_pool_records)
    write_binary_ply(dense_surface_path, room_pool_records)
    write_binary_ply(tsdf_backbone_path, coarse_records)
    write_binary_ply(local_memory_path, pool_debug_records)
    write_vertex_ply(
        dense_inst_path,
        room_projection.dense_points,
        room_projection.dense_colors,
        room_projection.labels.astype(np.int32),
        room_projection.state_ids.astype(np.uint8),
        room_projection.supports.astype(np.float32),
    )
    return {
        "instance_map": instance_map_path,
        "dense_surface": dense_surface_path,
        "tsdf_backbone": tsdf_backbone_path,
        "local_memory": local_memory_path,
        "dense_geometry_instance_projected": dense_inst_path,
    }


def render_standard_previews(
    *,
    exports_dir: Path,
    dense_points: np.ndarray,
    dense_rgb: np.ndarray,
    room_projection: RoomSemanticProjection,
) -> None:
    rgb_xz = exports_dir / "room0_dense_xz_rgb.png"
    inst_xz = exports_dir / "room0_dense_xz_instance.png"
    rgb_xy = exports_dir / "room0_dense_xy_rgb.png"
    inst_xy = exports_dir / "room0_dense_xy_instance.png"
    render_projection(dense_points, dense_rgb, plane="xz", output_path=rgb_xz)
    render_projection(dense_points, room_projection.dense_colors, plane="xz", output_path=inst_xz)
    render_projection(dense_points, dense_rgb, plane="xy", output_path=rgb_xy)
    render_projection(dense_points, room_projection.dense_colors, plane="xy", output_path=inst_xy)
    render_preview_sheet(
        [rgb_xz, inst_xz, rgb_xy, inst_xy],
        exports_dir / "room0_dense_projection_preview.png",
    )


def evaluate_and_write_reports(
    *,
    eval_dir: Path,
    scene_dir: Path,
    experiment_name: str,
    frame_limit: int,
    dense_points: np.ndarray,
    room_projection: RoomSemanticProjection,
    coarse_records: np.ndarray,
    room_pool_records: np.ndarray,
    pool_debug_records: np.ndarray,
    state: SemanticMapV2State,
    frame_metrics: list[dict[str, Any]],
    gt_labels_path: Path,
    gt_mesh_path: Path,
    gt_info_path: Path,
) -> dict[str, Any]:
    gt_vertices = read_gt_vertices(gt_mesh_path)
    gt_labels = read_gt_labels(gt_labels_path)
    class_names = read_class_names(gt_info_path)
    evaluation = evaluate_semantics(
        gt_vertices=gt_vertices,
        gt_labels=gt_labels,
        dense_points=dense_points,
        object_ids=room_projection.labels,
        class_names=class_names,
    )
    write_eval_artifacts(eval_dir, gt_vertices, gt_labels, evaluation, class_names)

    object_state_counts = Counter(pool.state.value for pool in state.object_pools.values())
    frozen_label_count = int(sum(1 for pool in state.object_pools.values() if pool.label_frozen))
    report = {
        "miou": evaluation["miou"],
        "macc": evaluation["macc"],
        "fmiou": evaluation["fmiou"],
        "fmacc": evaluation["fmacc"],
        "per_class": evaluation["per_class"],
        "frame_count": int(frame_limit),
        "final_object_count": int(len(state.object_pools)),
        "frozen_label_count": frozen_label_count,
        "active_object_count": int(object_state_counts.get(V2ObjectState.ACTIVE.value, 0)),
        "archived_object_count": int(object_state_counts.get(V2ObjectState.ARCHIVED.value, 0)),
        "coarse_point_count": int(len(coarse_records)),
        "room_semantic_point_count": int(np.count_nonzero(room_projection.labels >= 0)),
        "object_pool_debug_point_count": int(len(pool_debug_records)),
        "projected_dense_point_count": int(np.count_nonzero(room_projection.labels >= 0)),
        "mean_raw_proposal_count": float(np.mean([item["raw_proposal_count"] for item in frame_metrics])) if frame_metrics else 0.0,
        "mean_candidate_patch_count": float(np.mean([item["candidate_patch_count"] for item in frame_metrics])) if frame_metrics else 0.0,
    }
    report_path = scene_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    top_lines = [
        f"- `{item['class_name']}`: IoU `{item['iou']:.4f}`, Acc `{item['acc']:.4f}`"
        for item in sorted(evaluation["per_class"], key=lambda item: item["iou"], reverse=True)[:25]
    ]
    md = "\n".join(
        [
            "# V2 Semantic Map Run Report",
            "",
            f"- Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            f"- Run dir: `{scene_dir}`",
            f"- Eval dir: `{eval_dir}`",
            f"- `experiment_name`: `{experiment_name}`",
            f"- `frame_count`: `{frame_limit}`",
            "",
            "## Metrics",
            f"- `mIoU`: {evaluation['miou']:.4f}",
            f"- `mAcc`: {evaluation['macc']:.4f}",
            f"- `f-mIoU`: {evaluation['fmiou']:.4f}",
            f"- `f-mAcc`: {evaluation['fmacc']:.4f}",
            "",
            "## V2 State",
            f"- `final_object_count`: `{len(state.object_pools)}`",
            f"- `frozen_label_count`: `{frozen_label_count}`",
            f"- `active_object_count`: `{int(object_state_counts.get(V2ObjectState.ACTIVE.value, 0))}`",
            f"- `archived_object_count`: `{int(object_state_counts.get(V2ObjectState.ARCHIVED.value, 0))}`",
            f"- `coarse_point_count`: `{len(coarse_records)}`",
            f"- `room_semantic_point_count`: `{len(room_pool_records)}`",
            f"- `object_pool_debug_point_count`: `{len(pool_debug_records)}`",
            "",
            "## Per-class",
            *(top_lines or ["- No per-class metrics available"]),
            "",
        ]
    )
    (scene_dir / "run_report.md").write_text(md, encoding="utf-8")
    return report
