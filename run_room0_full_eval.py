#!/usr/bin/env python3
"""Full room0 run with instance-map export, dense projection, evaluation, and report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from run_replica_room0_analysis import build_proposal_module
from src.core.data_structures import Frame, ObjectState, SystemState
from src.datasets import ReplicaRoom0Dataset
from src.modules.semantic_memory import (
    object_export_semantic_label,
    object_export_semantic_state,
    object_semantic_commit_state,
)
from src.pipelines.main_pipeline import Pipeline
from src.pipelines.proposal_prefetch import FrameProposalPrefetcher
from src.utils.geometry import voxel_downsample
from src.utils.visualization import (
    add_panel_title,
    anchor_overlay_image,
    anchor_vote_overlay_image,
    proposal_overlay_image,
    runtime_group_overlay_image,
    save_multi_panel,
)


PLY_DTYPE = np.dtype(
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

STRUCTURAL_OVERLAY_ID_OFFSET = 10000


def structural_overlay_object_id(class_id: int) -> int:
    return -int(STRUCTURAL_OVERLAY_ID_OFFSET + int(class_id))


def structural_overlay_direct_label_maps(
    class_names: dict[int, str],
    structure_labels: set[str],
) -> tuple[dict[int, int], dict[int, str]]:
    normalized = {str(label).strip().lower() for label in structure_labels}
    direct_ids_to_class: dict[int, int] = {}
    direct_ids_to_name: dict[int, str] = {}
    for class_id, class_name in sorted(class_names.items()):
        label = str(class_name).strip().lower()
        if label not in normalized:
            continue
        object_id = structural_overlay_object_id(int(class_id))
        direct_ids_to_class[object_id] = int(class_id)
        direct_ids_to_name[object_id] = label
    return direct_ids_to_class, direct_ids_to_name


@dataclass(frozen=True)
class OutputProfile:
    name: str
    benchmark_audit_enabled: bool
    write_primary_exports: bool = True
    write_debug_ply: bool = True
    write_eval_mesh_ply: bool = True
    write_preview_render: bool = True
    write_final_audit: bool = True
    write_reports: bool = True


@dataclass(frozen=True)
class Room0RunLayout:
    run_root: Path
    scene_dir: Path
    eval_dir: Path
    exports_dir: Path
    vis_dir: Path
    audit_dir: Path
    audit_vis_dir: Path


@dataclass(frozen=True)
class Room0ExportResult:
    report_path: Path
    results_path: Path
    instance_map_path: Path
    dense_surface_path: Path
    dense_instance_path: Path
    miou: float
    macc: float
    fmiou: float
    fmacc: float
    final_object_count: int
    dense_geometry_point_count: int


ROOM0_CHECKPOINT_VERSION = 1


def normalize_scene_name(scene_name: Any, default: str = "room0") -> str:
    normalized = str(scene_name if scene_name is not None else "").strip()
    return normalized or default


def build_room0_run_layout(output_root: Path, experiment_name: str, scene_name: str = "room0") -> Room0RunLayout:
    run_root = Path(output_root) / experiment_name
    scene_dir = run_root / normalize_scene_name(scene_name)
    return Room0RunLayout(
        run_root=run_root,
        scene_dir=scene_dir,
        eval_dir=run_root / "replica",
        exports_dir=scene_dir / "exports",
        vis_dir=scene_dir / "vis",
        audit_dir=scene_dir / "local_memory_audit",
        audit_vis_dir=scene_dir / "local_memory_audit" / "vis",
    )


def ensure_room0_run_directories(layout: Room0RunLayout, output_profile: OutputProfile) -> None:
    directories = [layout.scene_dir, layout.eval_dir, layout.exports_dir, layout.vis_dir]
    if output_profile.benchmark_audit_enabled:
        directories.extend([layout.audit_dir, layout.audit_vis_dir])
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def output_profile_to_dict(output_profile: OutputProfile) -> dict[str, Any]:
    return {
        "name": output_profile.name,
        "benchmark_audit_enabled": bool(output_profile.benchmark_audit_enabled),
        "write_primary_exports": bool(output_profile.write_primary_exports),
        "write_debug_ply": bool(output_profile.write_debug_ply),
        "write_eval_mesh_ply": bool(output_profile.write_eval_mesh_ply),
        "write_preview_render": bool(output_profile.write_preview_render),
        "write_final_audit": bool(output_profile.write_final_audit),
        "write_reports": bool(output_profile.write_reports),
    }


def output_profile_from_checkpoint(raw_profile: Any) -> OutputProfile:
    if isinstance(raw_profile, OutputProfile):
        return raw_profile
    profile = dict(raw_profile or {})
    return OutputProfile(
        name=str(profile.get("name", "full_debug")),
        benchmark_audit_enabled=bool(profile.get("benchmark_audit_enabled", True)),
        write_primary_exports=bool(profile.get("write_primary_exports", True)),
        write_debug_ply=bool(profile.get("write_debug_ply", True)),
        write_eval_mesh_ply=bool(profile.get("write_eval_mesh_ply", True)),
        write_preview_render=bool(profile.get("write_preview_render", True)),
        write_final_audit=bool(profile.get("write_final_audit", True)),
        write_reports=bool(profile.get("write_reports", True)),
    )


def args_from_room0_checkpoint(payload: dict[str, Any]) -> argparse.Namespace:
    args = dict(payload.get("args", {}) or {})
    for key in (
        "dataset_root",
        "gt_labels",
        "gt_mesh_ply",
        "gt_info_json",
        "config_path",
        "output_root",
        "proposal_cache_dir",
        "proposal_cache_manifest",
        "sam_repo_root",
        "sam_ckpt_path",
        "sam3_worker_python",
        "sam3_worker_script",
        "sam3_repo_root",
        "sam3_checkpoint_path",
        "checkpoint_path",
    ):
        if args.get(key) is not None:
            args[key] = Path(args[key])
    return argparse.Namespace(**args)


def pipeline_from_room0_checkpoint(payload: dict[str, Any]) -> Any:
    return SimpleNamespace(
        config=dict(payload.get("pipeline_config", {}) or {}),
        proposal=SimpleNamespace(
            active_backend_name=str(payload.get("proposal_active_backend_name", "unknown") or "unknown")
        ),
    )


def build_room0_checkpoint_payload(
    *,
    experiment_name: str,
    layout: Room0RunLayout,
    args: argparse.Namespace,
    pipeline: Any,
    state: SystemState,
    geometry_accum: dict[tuple[int, int, int], list[np.ndarray | int]],
    frame_metrics: list[dict[str, Any]],
    frame_audits: list[dict[str, Any]] | None,
    frame_limit: int,
    selected_dataset_indices: list[int],
    output_profile: OutputProfile,
    build_timing: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": ROOM0_CHECKPOINT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": experiment_name,
        "scene_name": normalize_scene_name(getattr(args, "scene_name", "room0")),
        "run_root": layout.run_root,
        "scene_dir": layout.scene_dir,
        "eval_dir": layout.eval_dir,
        "exports_dir": layout.exports_dir,
        "vis_dir": layout.vis_dir,
        "audit_dir": layout.audit_dir,
        "args": dict(vars(args)),
        "pipeline_config": dict(getattr(pipeline, "config", {}) or {}),
        "proposal_active_backend_name": str(
            getattr(getattr(pipeline, "proposal", None), "active_backend_name", "unknown")
        ),
        "state": state,
        "geometry_accum": geometry_accum,
        "frame_metrics": frame_metrics,
        "frame_audits": list(frame_audits or []),
        "frame_limit": int(frame_limit),
        "selected_dataset_indices": [int(index) for index in selected_dataset_indices],
        "output_profile": output_profile_to_dict(output_profile),
        "build_timing": dict(build_timing),
    }


def save_room0_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def load_room0_checkpoint(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"room0 checkpoint must contain a dict payload: {path}")
    version = int(payload.get("version", 0) or 0)
    if version != ROOM0_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported room0 checkpoint version {version}; expected {ROOM0_CHECKPOINT_VERSION}"
        )
    return payload


def build_output_profile(args: argparse.Namespace) -> OutputProfile:
    """Build the output artifact profile without changing pipeline behavior."""
    if bool(getattr(args, "full_debug", False)):
        return OutputProfile(
            name="full_debug",
            benchmark_audit_enabled=True,
            write_primary_exports=True,
            write_debug_ply=True,
            write_eval_mesh_ply=True,
            write_preview_render=True,
            write_final_audit=True,
            write_reports=True,
        )

    if bool(getattr(args, "fast_eval", False)):
        return OutputProfile(
            name="fast_eval",
            benchmark_audit_enabled=False,
            write_primary_exports=True,
            write_debug_ply=False,
            write_eval_mesh_ply=False,
            write_preview_render=False,
            write_final_audit=True,
            write_reports=True,
        )

    benchmark_audit_enabled = not bool(getattr(args, "lightweight_benchmark", False))
    debug_ply_enabled = not bool(getattr(args, "skip_debug_ply", False))
    eval_mesh_ply_enabled = not bool(getattr(args, "skip_eval_ply", False))
    preview_render_enabled = not bool(getattr(args, "skip_preview_render", False))
    name = (
        "custom"
        if (
            bool(getattr(args, "skip_debug_ply", False))
            or bool(getattr(args, "skip_eval_ply", False))
            or bool(getattr(args, "skip_preview_render", False))
            or bool(getattr(args, "lightweight_benchmark", False))
        )
        else "full_debug"
    )
    return OutputProfile(
        name=name,
        benchmark_audit_enabled=benchmark_audit_enabled,
        write_primary_exports=True,
        write_debug_ply=debug_ply_enabled,
        write_eval_mesh_ply=eval_mesh_ply_enabled,
        write_preview_render=preview_render_enabled,
        write_final_audit=True,
        write_reports=True,
    )


def build_prefetch_pipeline(config_path: Path, args: argparse.Namespace) -> Pipeline:
    prefetch_pipeline = Pipeline(config_path=str(config_path))
    prefetch_pipeline.verbose = False
    prefetch_pipeline.proposal = build_proposal_module(prefetch_pipeline.proposal.config, args)
    return prefetch_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-name", type=str, default="room0")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "input" / "Datasets" / "Replica" / "room0",
    )
    parser.add_argument(
        "--gt-labels",
        type=Path,
        default=project_root / "data" / "input" / "replica_semantic_gt" / "room0.txt",
    )
    parser.add_argument(
        "--gt-mesh-ply",
        type=Path,
        default=Path("/home/phl/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply"),
    )
    parser.add_argument(
        "--gt-info-json",
        type=Path,
        default=Path("/home/phl/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json"),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=project_root / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "outputs",
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="0 means the full room0 sequence.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame from the dataset sequence.",
    )
    parser.add_argument("--proposal-backend", type=str, default="sam2")
    parser.add_argument("--proposal-device", type=str, default="cuda")
    parser.add_argument("--proposal-cache-dir", type=Path, default=None)
    parser.add_argument("--proposal-cache-manifest", type=Path, default=None)
    parser.add_argument("--sam-version", type=str, default="2")
    parser.add_argument("--sam-encoder", type=str, default="hiera_l")
    parser.add_argument("--sam-repo-root", type=Path, default=Path("/home/phl/vv/paper2/OVO/thirdParty/segment-anything-2"))
    parser.add_argument("--sam-ckpt-path", type=Path, default=Path("/home/phl/vv/paper2/DovSG/checkpoints/segment-anything-2"))
    parser.add_argument("--sam3-worker-python", type=Path, default=None)
    parser.add_argument("--sam3-worker-script", type=Path, default=None)
    parser.add_argument("--sam3-mode", type=str, default=None)
    parser.add_argument("--sam3-repo-root", type=Path, default=None)
    parser.add_argument("--sam3-checkpoint-path", type=Path, default=None)
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument("--stability-score-th", type=float, default=0.95)
    parser.add_argument("--nms-iou-th", type=float, default=0.8)
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    parser.add_argument("--geometry-sample-stride", type=int, default=8)
    parser.add_argument("--geometry-voxel-size", type=float, default=0.03)
    parser.add_argument("--instance-voxel-size", type=float, default=0.05)
    parser.add_argument("--projection-neighbor-radius", type=int, default=1)
    parser.add_argument(
        "--lightweight-benchmark",
        action="store_true",
        help="Skip heavy local-memory audit generation to reduce benchmark runtime.",
    )
    parser.add_argument(
        "--fast-eval",
        action="store_true",
        help=(
            "Use fast evaluation outputs: keep metrics, primary exports, final semantic audit, "
            "and reports; skip local-memory audit, debug duplicate PLYs, GT mesh PLYs, and previews."
        ),
    )
    parser.add_argument(
        "--full-debug",
        action="store_true",
        help="Force all debug artifacts on, overriding --fast-eval and skip flags.",
    )
    parser.add_argument(
        "--skip-debug-ply",
        action="store_true",
        help="Skip debug-only duplicate PLY exports such as local-memory compatibility PLY.",
    )
    parser.add_argument(
        "--skip-eval-ply",
        action="store_true",
        help="Skip GT mesh semantic visualization PLYs while keeping JSON evaluation metrics.",
    )
    parser.add_argument(
        "--skip-preview-render",
        action="store_true",
        help="Skip dense projection preview PNG rendering.",
    )
    parser.add_argument("--pipeline-verbose", action="store_true")
    return parser.parse_args()


def export_room0_outputs(
    *,
    layout: Room0RunLayout,
    experiment_name: str,
    dataset: ReplicaRoom0Dataset,
    args: argparse.Namespace,
    frame_limit: int,
    pipeline: Any,
    state: SystemState,
    geometry_accum: dict[tuple[int, int, int], list[np.ndarray | int]],
    frame_metrics: list[dict[str, Any]],
    output_profile: OutputProfile,
    frame_audits: list[dict[str, Any]] | None = None,
) -> Room0ExportResult:
    ensure_room0_run_directories(layout, output_profile)
    benchmark_audit_enabled = bool(output_profile.benchmark_audit_enabled)
    tsdf_records = build_tsdf_backbone_records(state)
    pool_semantic_records = build_pool_semantic_records(state)
    dense_surface_records = build_dense_surface_records(state)
    pool_debug_records = build_pool_debug_records(state)

    instance_map_path = layout.exports_dir / "room0_instance_map.ply"
    dense_surface_path = layout.exports_dir / "room0_instance_map_dense_surface.ply"
    instance_backbone_path = layout.exports_dir / "room0_instance_map_tsdf_backbone.ply"
    local_memory_path = layout.exports_dir / "room0_instance_map_local_memory.ply"
    write_binary_ply(instance_map_path, pool_semantic_records)
    write_binary_ply(dense_surface_path, dense_surface_records)
    write_binary_ply(instance_backbone_path, tsdf_records)
    if output_profile.write_debug_ply:
        write_binary_ply(local_memory_path, pool_debug_records)

    dense_points, dense_rgb = finalize_geometry_accum(geometry_accum)
    gt_vertices = read_gt_vertices(args.gt_mesh_ply)
    gt_labels = read_gt_labels(args.gt_labels)
    class_names = read_class_names(args.gt_info_json)
    labels, state_ids, supports = project_instances_to_dense_points(
        dense_points=dense_points,
        instance_records=pool_semantic_records,
        instance_voxel_size=float(args.instance_voxel_size),
        neighbor_radius=max(0, int(args.projection_neighbor_radius)),
    )
    structural_cfg = dict(getattr(pipeline, "config", {}).get("structural_overlay", {}) or {})
    structure_labels = {
        str(label).strip().lower()
        for label in structural_cfg.get("classes", [])
        if str(label).strip()
    }
    protected_labels = {
        str(label).strip().lower()
        for label in structural_cfg.get("protected_labels", [])
        if str(label).strip()
    }
    direct_object_label_ids, direct_object_label_names = structural_overlay_direct_label_maps(
        class_names=class_names,
        structure_labels=structure_labels,
    )
    overlay_records = build_structural_overlay_records(
        state.structural_overlay_map,
        direct_ids_to_class=direct_object_label_ids,
        direct_ids_to_name=direct_object_label_names,
    )
    overlay_path = layout.exports_dir / "room0_structural_overlay.ply"
    write_binary_ply(overlay_path, overlay_records)
    overlay_ids, overlay_supports = project_structural_overlay_to_dense_points(
        dense_points=dense_points,
        structural_overlay_map=state.structural_overlay_map,
        direct_ids_to_class=direct_object_label_ids,
        class_names=class_names,
        neighbor_radius=max(0, int(structural_cfg.get("projection_neighbor_radius", 0))),
    )
    if bool(structural_cfg.get("conservative_fusion", False)):
        labels, state_ids, supports, structural_overlay_fusion = apply_conservative_structural_overlay(
            labels=labels,
            state_ids=state_ids,
            supports=supports,
            overlay_ids=overlay_ids,
            overlay_supports=overlay_supports,
            state=state,
            structure_labels=structure_labels,
            protected_labels=protected_labels,
        )
    else:
        structural_overlay_fusion = {
            "overlay_candidate_point_count": int(np.count_nonzero(overlay_ids < 0)),
            "overlay_replaced_point_count": 0,
            "overlay_protected_point_count": 0,
        }
    projected_colors = colors_for_object_labels(
        labels,
        state,
        direct_object_label_names=direct_object_label_names,
    )
    dense_rgb_path = layout.exports_dir / "room0_dense_geometry_fused_rgb.ply"
    dense_inst_path = layout.exports_dir / "room0_dense_geometry_instance_projected.ply"
    write_vertex_ply(
        dense_rgb_path,
        dense_points,
        dense_rgb,
        np.full(len(dense_points), -1, dtype=np.int32),
        np.zeros(len(dense_points), dtype=np.uint8),
        np.zeros(len(dense_points), dtype=np.float32),
    )
    write_vertex_ply(
        dense_inst_path,
        dense_points,
        projected_colors,
        labels.astype(np.int32),
        state_ids.astype(np.uint8),
        supports.astype(np.float32),
    )

    if output_profile.write_preview_render:
        render_projection(dense_points, dense_rgb, plane="xz", output_path=layout.vis_dir / "room0_dense_xz_rgb.png")
        render_projection(
            dense_points,
            projected_colors,
            plane="xz",
            output_path=layout.vis_dir / "room0_dense_xz_instance.png",
        )
        render_projection(dense_points, dense_rgb, plane="xy", output_path=layout.vis_dir / "room0_dense_xy_rgb.png")
        render_projection(
            dense_points,
            projected_colors,
            plane="xy",
            output_path=layout.vis_dir / "room0_dense_xy_instance.png",
        )
        render_preview_sheet(
            [
                layout.vis_dir / "room0_dense_xz_rgb.png",
                layout.vis_dir / "room0_dense_xz_instance.png",
                layout.vis_dir / "room0_dense_xy_rgb.png",
                layout.vis_dir / "room0_dense_xy_instance.png",
            ],
            layout.vis_dir / "room0_dense_projection_preview.png",
        )

    evaluation = evaluate_semantics(
        gt_vertices=gt_vertices,
        gt_labels=gt_labels,
        dense_points=dense_points,
        object_ids=labels,
        class_names=class_names,
        direct_object_label_ids=direct_object_label_ids,
    )
    final_object_semantic_audit = write_final_object_semantic_audit(
        scene_dir=layout.scene_dir,
        state=state,
        evaluation=evaluation,
        class_names=class_names,
    )

    write_eval_artifacts(
        layout.eval_dir,
        gt_vertices,
        gt_labels,
        evaluation,
        class_names,
        write_mesh_ply=output_profile.write_eval_mesh_ply,
    )
    if benchmark_audit_enabled:
        audits = list(frame_audits or [])
        audit_summary = summarize_local_memory_audits(audits, state)
        (layout.audit_dir / "local_memory_audit_summary.json").write_text(
            json.dumps(audit_summary, indent=2),
            encoding="utf-8",
        )
        (layout.audit_dir / "local_memory_audit.log").write_text(
            render_local_memory_audit_log(audits),
            encoding="utf-8",
        )
    if output_profile.write_reports:
        write_run_report(
            scene_dir=layout.scene_dir,
            eval_dir=layout.eval_dir,
            experiment_name=experiment_name,
            dataset=dataset,
            args=args,
            frame_limit=frame_limit,
            pipeline=pipeline,
            state=state,
            tsdf_records=tsdf_records,
            pool_semantic_records=pool_semantic_records,
            dense_surface_records=dense_surface_records,
            dense_points=dense_points,
            labels=labels,
            evaluation=evaluation,
            final_object_semantic_audit=final_object_semantic_audit,
            frame_metrics=frame_metrics,
            audit_dir=layout.audit_dir if benchmark_audit_enabled else None,
            output_profile=output_profile,
            structural_overlay_fusion=structural_overlay_fusion,
        )

    return Room0ExportResult(
        report_path=layout.scene_dir / "run_report.md",
        results_path=layout.eval_dir / "results.json",
        instance_map_path=instance_map_path,
        dense_surface_path=dense_surface_path,
        dense_instance_path=dense_inst_path,
        miou=float(evaluation["miou"]),
        macc=float(evaluation["macc"]),
        fmiou=float(evaluation["fmiou"]),
        fmacc=float(evaluation["fmacc"]),
        final_object_count=int(len(state.objects)),
        dense_geometry_point_count=int(len(dense_points)),
    )


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = args.experiment_name or f"{timestamp}_room0_ovopro_sam2v2_full"
    output_profile = build_output_profile(args)
    layout = build_room0_run_layout(args.output_root, experiment_name, scene_name=args.scene_name)
    scene_dir = layout.scene_dir
    audit_dir = layout.audit_dir
    audit_vis_dir = layout.audit_vis_dir
    benchmark_audit_enabled = bool(output_profile.benchmark_audit_enabled)
    ensure_room0_run_directories(layout, output_profile)

    dataset = ReplicaRoom0Dataset(args.dataset_root)
    frame_stride = max(1, int(args.frame_stride))
    all_frame_indices = list(range(0, len(dataset), frame_stride))
    selected_dataset_indices = (
        all_frame_indices
        if args.num_frames <= 0
        else all_frame_indices[: min(int(args.num_frames), len(all_frame_indices))]
    )
    frame_limit = len(selected_dataset_indices)

    pipeline = Pipeline(config_path=str(args.config_path))
    pipeline.verbose = bool(args.pipeline_verbose)
    pipeline.proposal = build_proposal_module(pipeline.proposal.config, args)
    pipeline_cfg = dict(pipeline.config.get("pipeline", {}) or {})
    prefetch_enabled = bool(pipeline_cfg.get("proposal_prefetch_enabled", False))
    prefetch_wait_timeout_sec = float(pipeline_cfg.get("proposal_prefetch_wait_timeout_sec", -1.0))
    prefetch_pipeline = build_prefetch_pipeline(args.config_path, args) if prefetch_enabled else None
    prefetcher = (
        FrameProposalPrefetcher(
            prefetch_pipeline._build_proposal_bundle,
            enabled=True,
            wait_timeout_sec=prefetch_wait_timeout_sec,
        )
        if prefetch_pipeline is not None
        else None
    )

    def build_prefetch_frame(dataset_frame: Frame, processed_frame_id: int) -> Frame:
        return Frame(
            frame_id=int(processed_frame_id),
            rgb=dataset_frame.rgb,
            depth=dataset_frame.depth,
            pose=dataset_frame.pose,
            intrinsics=dataset_frame.intrinsics,
            timestamp=float(dataset_frame.timestamp),
            source_frame_id=int(dataset_frame.frame_id),
        )

    frame_metrics_path = scene_dir / "frame_metrics.jsonl"
    audit_jsonl_path = audit_dir / "local_memory_audit.jsonl"
    geometry_accum: dict[tuple[int, int, int], list[np.ndarray | int]] = {}
    frame_metrics: list[dict[str, Any]] = []
    frame_audits: list[dict[str, Any]] = []

    print("=" * 60)
    print("OVIOVO Full room0 run")
    print("=" * 60)
    print(f"run_root:     {layout.run_root}")
    print(f"scene_dir:    {scene_dir}")
    print(f"frame_limit:  {frame_limit}")
    print(f"backend:      {args.proposal_backend}")
    print(f"device:       {args.proposal_device}")
    print(f"conda_env:    {os.environ.get('CONDA_DEFAULT_ENV')}")
    print(f"benchmark_audit_enabled: {benchmark_audit_enabled}")
    print(f"output_profile: {output_profile.name}")

    with frame_metrics_path.open("w", encoding="utf-8") as metrics_handle:
        audit_handle = (
            audit_jsonl_path.open("w", encoding="utf-8")
            if benchmark_audit_enabled
            else None
        )
        try:
            next_prefetch_index = 0
            submitted_prefetch_frame_ids: set[int] = set()
            if prefetch_enabled and selected_dataset_indices:
                first_frame = dataset[int(selected_dataset_indices[0])]
                if prefetcher is not None and prefetcher.submit(build_prefetch_frame(first_frame, processed_frame_id=0)):
                    submitted_prefetch_frame_ids.add(0)
                    next_prefetch_index = 1
            for idx, dataset_index in enumerate(selected_dataset_indices, start=1):
                frame = dataset[int(dataset_index)]
                processed_frame_id = idx - 1
                scheduling_debug = {
                    "prefetch_enabled": bool(prefetch_enabled),
                    "prefetch_submitted": processed_frame_id in submitted_prefetch_frame_ids,
                    "prefetch_hit": False,
                    "prefetch_miss": False,
                    "prefetch_wait_sec": 0.0,
                    "prefetch_exception": "",
                    "frontend_source": "inline",
                }
                proposal_bundle = None
                if prefetcher is not None:
                    stats_before = prefetcher.snapshot_stats()
                    proposal_bundle = prefetcher.result_for(processed_frame_id)
                    stats = prefetcher.snapshot_stats()
                    scheduling_debug["prefetch_wait_sec"] = max(
                        0.0,
                        float(stats.get("wait_sec_total", 0.0) or 0.0)
                        - float(stats_before.get("wait_sec_total", 0.0) or 0.0),
                    )
                    if proposal_bundle is not None:
                        scheduling_debug["prefetch_hit"] = True
                        scheduling_debug["frontend_source"] = "prefetch"
                    else:
                        scheduling_debug["prefetch_miss"] = True
                    previous_exception_count = int(stats_before.get("exception_count", 0) or 0)
                    current_exception_count = int(stats.get("exception_count", 0) or 0)
                    previous_exception = str(stats_before.get("last_exception", "") or "")
                    current_exception = str(stats.get("last_exception", "") or "")
                    if current_exception and (
                        current_exception_count > previous_exception_count
                        or current_exception != previous_exception
                    ):
                        scheduling_debug["prefetch_exception"] = current_exception

                submitted_next_before_processing = False
                if (
                    proposal_bundle is not None
                    and prefetcher is not None
                    and next_prefetch_index < len(selected_dataset_indices)
                ):
                    next_dataset_index = selected_dataset_indices[next_prefetch_index]
                    next_frame = dataset[int(next_dataset_index)]
                    submitted = prefetcher.submit(
                        build_prefetch_frame(next_frame, processed_frame_id=next_prefetch_index)
                    )
                    if submitted:
                        submitted_prefetch_frame_ids.add(next_prefetch_index)
                        next_prefetch_index += 1
                        submitted_next_before_processing = True

                prev_object_ids = set(pipeline.state.objects)
                prev_next_object_id = int(pipeline.state.next_object_id)
                state = pipeline.process_frame(
                    frame.rgb,
                    frame.depth,
                    frame.pose,
                    frame.intrinsics,
                    timestamp=frame.timestamp,
                    source_frame_id=frame.frame_id,
                    proposal_bundle=proposal_bundle,
                )
                scheduling_debug["frontend_source"] = str(
                    pipeline.last_frame_debug.get("frontend_source", scheduling_debug["frontend_source"])
                )
                pipeline.last_frame_debug["scheduling"] = dict(scheduling_debug)
                if (
                    prefetcher is not None
                    and not submitted_next_before_processing
                    and next_prefetch_index < len(selected_dataset_indices)
                ):
                    next_dataset_index = selected_dataset_indices[next_prefetch_index]
                    next_frame = dataset[int(next_dataset_index)]
                    submitted = prefetcher.submit(
                        build_prefetch_frame(next_frame, processed_frame_id=next_prefetch_index)
                    )
                    if submitted:
                        submitted_prefetch_frame_ids.add(next_prefetch_index)
                        next_prefetch_index += 1
                accumulate_geometry(
                    geometry_accum,
                    frame=frame,
                    sample_stride=max(1, int(args.geometry_sample_stride)),
                    voxel_size=float(args.geometry_voxel_size),
                )

                association = pipeline.last_association
                runtime_output = pipeline.last_runtime_vis_output
                if association is None or runtime_output is None:
                    raise RuntimeError("Pipeline debug outputs missing during full run.")
                state_counts = summarize_state_counts(state)
                runtime_merged_proposals = list(getattr(runtime_output, "merged_proposals", []))
                anchor_voted_unanchored_count = sum(
                    1
                    for proposal in pipeline.last_raw_proposals
                    if int((getattr(proposal, "metadata", {}) or {}).get("anchor_id", -1)) < 0
                )
                record = {
                    "frame_id": int(frame.frame_id),
                    "raw_proposal_count": len(pipeline.last_raw_proposals),
                    "merged_group_count": len(runtime_output.groups),
                    "refined_proposal_count": len(pipeline.last_refined_proposals),
                    "patch_count": len(pipeline.last_patches),
                    "matched_patch_count": len(association.matched),
                    "new_patch_count": len(association.new_object_patches),
                    "ambiguous_patch_count": int(pipeline.last_frame_debug.get("ambiguous_patch_count", 0)),
                    "ambiguous_matched_count": int(pipeline.last_frame_debug.get("ambiguous_matched_count", 0)),
                    "ambiguous_new_patch_count": int(pipeline.last_frame_debug.get("ambiguous_new_patch_count", 0)),
                    "total_object_count": len(state.objects),
                    "provisional_object_count": len(state.provisional_objects),
                    "active_object_count": state_counts.get("active", 0),
                    "dormant_object_count": state_counts.get("dormant", 0),
                    "inactive_object_count": state_counts.get("inactive", 0),
                    "semantic_updated_count": len(pipeline.last_frame_debug.get("semantic_updated_object_ids", [])),
                    "local_memory_point_count_total": int(
                        pipeline.last_frame_debug.get(
                            "local_memory_point_count_total",
                            sum(len(obj.local_pcd) for obj in state.objects.values()),
                        )
                    ),
                    "surface_owner_gate": dict(pipeline.last_frame_debug.get("surface_owner_gate", {}) or {}),
                    "structural_overlay": dict(pipeline.last_frame_debug.get("structural_overlay", {}) or {}),
                    "stage_timings": dict(pipeline.last_frame_debug.get("stage_timings", {}) or {}),
                    "association_summary": dict(
                        (pipeline.last_frame_debug.get("association_debug", {}) or {}).get("summary", {}) or {}
                    ),
                    "depth_refinement": dict(pipeline.last_frame_debug.get("depth_refinement", {}) or {}),
                    "async_refinement": dict(pipeline.last_frame_debug.get("async_refinement", {}) or {}),
                    "current_frame_visibility_gate": dict(
                        pipeline.last_frame_debug.get("current_frame_visibility_gate", {}) or {}
                    ),
                    "scheduling": dict(pipeline.last_frame_debug.get("scheduling", {}) or {}),
                    "frontend_stage": {
                        "source_sam_proposal_count": len(pipeline.last_source_proposals),
                        "anchor_voted_proposal_count": len(pipeline.last_raw_proposals),
                        "runtime_merged_proposal_count": len(runtime_merged_proposals),
                        "runtime_merge_count": max(
                            0,
                            len(pipeline.last_raw_proposals) - len(runtime_merged_proposals),
                        ),
                        "anchor_voted_unanchored_count": int(anchor_voted_unanchored_count),
                        "runtime_anchor_label_missing_edge_count": int(
                            runtime_output.group_stats.get("anchor_label_missing_edge_count", 0)
                        ),
                        "runtime_anchor_label_mismatch_edge_count": int(
                            runtime_output.group_stats.get("anchor_label_mismatch_edge_count", 0)
                        ),
                        "runtime_anchor_identity_mismatch_edge_count": int(
                            runtime_output.group_stats.get("anchor_identity_mismatch_edge_count", 0)
                        ),
                        "anchor_guided_sam": dict(pipeline.last_frame_debug.get("anchor_guided_sam", {}) or {}),
                    },
                    "actual_backend": pipeline.proposal.active_backend_name,
                }
                frame_metrics.append(record)
                metrics_handle.write(json.dumps(record) + "\n")

                if benchmark_audit_enabled and audit_handle is not None:
                    audit_record = build_local_memory_frame_audit(
                        frame=frame,
                        pipeline=pipeline,
                        prev_object_ids=prev_object_ids,
                        prev_next_object_id=prev_next_object_id,
                    )
                    should_save_overlay = (
                        audit_record.get("anchor_count", 0) > 0
                        or audit_record["new_instance_count"] > 0
                        or audit_record["hard_failure_count"] > 0
                    )
                    if should_save_overlay:
                        overlay_path = audit_vis_dir / f"{frame.frame_id:04d}.png"
                        save_local_memory_audit_overlay(
                            frame=frame,
                            source_proposals=getattr(pipeline, "last_source_proposals", []),
                            raw_proposals=pipeline.last_raw_proposals,
                            raw_records=audit_record["raw_proposals"],
                            anchors=audit_record.get("anchors", []),
                            output_path=overlay_path,
                            runtime_groups=list(getattr(pipeline.last_runtime_vis_output, "groups", [])),
                        )
                        audit_record["overlay_path"] = str(overlay_path)
                    frame_audits.append(audit_record)
                    audit_handle.write(json.dumps(audit_record) + "\n")

                if idx == 1 or idx % 20 == 0 or idx == frame_limit:
                    print(
                        f"frame={frame.frame_id:04d} processed={idx:04d}/{frame_limit:04d} "
                        f"objects={len(state.objects):03d} raw={record['raw_proposal_count']:03d} "
                        f"matched={record['matched_patch_count']:03d}"
                    )
        finally:
            if prefetcher is not None:
                prefetcher.close()
            if audit_handle is not None:
                audit_handle.close()

    export_result = export_room0_outputs(
        layout=layout,
        experiment_name=experiment_name,
        dataset=dataset,
        args=args,
        frame_limit=frame_limit,
        pipeline=pipeline,
        state=pipeline.state,
        geometry_accum=geometry_accum,
        frame_metrics=frame_metrics,
        output_profile=output_profile,
        frame_audits=frame_audits,
    )

    print("\nSaved outputs:")
    print(f"- run_root:              {layout.run_root}")
    print(f"- report:                {export_result.report_path}")
    print(f"- eval results:          {export_result.results_path}")
    print(f"- pool instance map:     {export_result.instance_map_path}")
    print(f"- dense surface map:     {export_result.dense_surface_path}")
    print(f"- dense instance map:    {export_result.dense_instance_path}")


def build_local_memory_frame_audit(
    frame,
    pipeline: Pipeline,
    prev_object_ids: set[int],
    prev_next_object_id: int,
) -> dict[str, Any]:
    runtime_output = pipeline.last_runtime_vis_output
    association = pipeline.last_association
    if runtime_output is None or association is None:
        raise RuntimeError("Pipeline debug outputs missing while building local-memory audit.")

    state = pipeline.state
    raw_proposals = list(pipeline.last_raw_proposals)
    source_sam_proposals = list(getattr(pipeline, "last_source_proposals", []))
    anchor_voted_proposals = list(pipeline.last_raw_proposals)
    runtime_merged_groups = list(runtime_output.groups)
    refined_proposals = list(pipeline.last_refined_proposals)
    patches = list(pipeline.last_patches)
    anchors = list(
        getattr(
            pipeline,
            "last_anchors",
            getattr(getattr(pipeline, "object_anchor", None), "last_anchors", []),
        )
    )
    anchor_assignments = list(
        getattr(
            pipeline,
            "last_anchor_assignments",
            getattr(getattr(pipeline, "object_anchor", None), "last_assignments", []),
        )
    )
    serialized_anchors = [serialize_anchor(anchor) for anchor in anchors]
    anchor_assignment_lookup = {
        int(serialized["proposal_id"]): serialized
        for serialized in (
            serialize_anchor_assignment(assignment)
            for assignment in anchor_assignments
        )
    }

    group_lookup = {int(group.group_id): group for group in runtime_output.groups}
    profile_lookup = {int(profile.proposal_id): profile for profile in runtime_output.proposal_profiles}

    group_members_by_id = {
        int(group.group_id): [int(value) for value in getattr(group, "member_mask_ids", [])]
        for group in runtime_output.groups
    }
    refined_by_member_raw: dict[int, list[Any]] = defaultdict(list)
    for refined in refined_proposals:
        raw_id = int(refined.metadata.get("source_raw_proposal_id", refined.proposal_id))
        member_ids = group_members_by_id.get(raw_id)
        if member_ids:
            for member_id in member_ids:
                refined_by_member_raw[member_id].append(refined)
        else:
            refined_by_member_raw[raw_id].append(refined)

    patches_by_refined: dict[int, list[Any]] = defaultdict(list)
    for patch in patches:
        source_refined_id = int(patch.metadata.get("source_proposal_id", patch.patch_id))
        patches_by_refined[source_refined_id].append(patch)

    bg_patch_ids = {int(value) for value in pipeline.last_frame_debug.get("bg_patch_ids", [])}
    obj_patch_ids = {int(value) for value in pipeline.last_frame_debug.get("obj_patch_ids", [])}
    amb_patch_ids = {int(value) for value in pipeline.last_frame_debug.get("amb_patch_ids", [])}

    matched_by_patch: dict[int, dict[str, Any]] = {}
    for patch_id, object_id, score in association.matched:
        matched_by_patch[int(patch_id)] = {
            "matched_object_id": int(object_id),
            "total_score": float(score.total_score),
            "voxel_vote_score": float(score.voxel_vote_score),
            "geometry_overlap": float(score.geometry_overlap),
            "bbox_overlap": float(score.bbox_overlap),
            "centroid_distance": float(score.centroid_distance),
        }

    contested_matches = list(getattr(association, "contested_matches", []))
    contested_residual_events = []
    for contested in contested_matches:
        score = getattr(contested, "score", None)
        contested_residual_events.append({
            "patch_id": int(contested.patch_id),
            "blocked_object_id": int(contested.blocked_object_id),
            "patch_label": str(contested.patch_label),
            "object_label": str(contested.object_label),
            "reason": str(contested.reason),
            "score": float(getattr(score, "total_score", 0.0) if score is not None else 0.0),
        })

    contested_patch_ids = {
        int(patch_id) for patch_id in getattr(association, "contested_object_patches", [])
    }
    matched_patch_ids = {int(patch_id) for patch_id in matched_by_patch}
    new_patch_ids = [int(patch_id) for patch_id in association.new_object_patches]
    new_patch_id_set = {int(patch_id) for patch_id in new_patch_ids}
    new_patch_to_object_id = {
        int(patch_id): int(prev_next_object_id + offset)
        for offset, patch_id in enumerate(new_patch_ids)
        if int(prev_next_object_id + offset) < int(state.next_object_id)
    }

    candidate_scores_by_patch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (patch_id, object_id), score in association.scores.items():
        candidate_scores_by_patch[int(patch_id)].append(
            {
                "object_id": int(object_id),
                "total_score": float(score.total_score),
                "voxel_vote_score": float(score.voxel_vote_score),
                "geometry_overlap": float(score.geometry_overlap),
                "bbox_overlap": float(score.bbox_overlap),
                "centroid_distance": float(score.centroid_distance),
            }
        )
    for patch_id, scores in candidate_scores_by_patch.items():
        scores.sort(key=lambda item: item["total_score"], reverse=True)

    raw_records: list[dict[str, Any]] = []
    new_instance_events: list[dict[str, Any]] = []
    seen_new_instance_events: set[tuple[int, int | None]] = set()
    outcome_counts: Counter[str] = Counter()

    for proposal in sorted(raw_proposals, key=lambda item: int(item.proposal_id)):
        raw_id = int(proposal.proposal_id)
        profile = profile_lookup.get(raw_id)
        group_id = runtime_output.raw_to_group.get(raw_id)
        group = group_lookup.get(int(group_id)) if group_id is not None else None
        refined_components = sorted(
            refined_by_member_raw.get(raw_id, []),
            key=lambda item: int(item.proposal_id),
        )

        refined_records: list[dict[str, Any]] = []
        patch_records: list[dict[str, Any]] = []
        created_object_ids: list[int] = []
        matched_existing_ids: list[int] = []
        bg_count = 0
        amb_count = 0
        obj_count = 0

        for refined in refined_components:
            refined_id = int(refined.proposal_id)
            related_patches = sorted(patches_by_refined.get(refined_id, []), key=lambda item: int(item.patch_id))
            refined_records.append(
                {
                    "refined_proposal_id": refined_id,
                    "area": int(refined.area),
                    "valid_depth_point_count": int(np.count_nonzero(refined.mask & np.isfinite(frame.depth) & (frame.depth > 0.0))),
                    "depth_valid_ratio": float(refined.geometric_features.depth_valid_ratio),
                    "depth_variance": float(refined.geometric_features.depth_variance),
                    "border_touch_ratio": float(refined.geometric_features.border_touch_ratio),
                    "planar_fit_residual": float(refined.geometric_features.planar_fit_residual),
                    "objectness_score": float(refined.soft_scores.objectness_score),
                    "backgroundness_score": float(refined.soft_scores.backgroundness_score),
                    "attachedness_score": float(refined.soft_scores.attachedness_score),
                    "produced_patch_ids": [int(patch.patch_id) for patch in related_patches],
                }
            )

            for patch in related_patches:
                patch_id = int(patch.patch_id)
                if patch_id in bg_patch_ids:
                    split_category = "background"
                    bg_count += 1
                elif patch_id in amb_patch_ids:
                    split_category = "ambiguous"
                    amb_count += 1
                elif patch_id in obj_patch_ids:
                    split_category = "object"
                    obj_count += 1
                else:
                    split_category = "unknown"

                matched_payload = matched_by_patch.get(patch_id)
                created_object_id = new_patch_to_object_id.get(patch_id)
                if matched_payload is not None:
                    matched_existing_ids.append(int(matched_payload["matched_object_id"]))
                if created_object_id is not None:
                    created_object_ids.append(int(created_object_id))

                if patch_id in contested_patch_ids:
                    association_outcome = "contested_residual"
                elif patch_id in matched_patch_ids:
                    association_outcome = "matched"
                elif patch_id in new_patch_id_set:
                    association_outcome = "new_object"
                else:
                    association_outcome = "unassociated"

                patch_records.append(
                    {
                        "patch_id": patch_id,
                        "source_refined_proposal_id": refined_id,
                        "point_count": int(len(patch.points)),
                        "split_category": split_category,
                        "association_outcome": association_outcome,
                        "contested_parent_object_id": int(patch.metadata.get("contested_parent_object_id", -1)),
                        "contested_parent_label": str(patch.metadata.get("contested_parent_label", "")),
                        "contested_patch_label": str(patch.metadata.get("contested_patch_label", "")),
                        "contested_reason": str(patch.metadata.get("contested_reason", "")),
                        "matched_existing": matched_payload,
                        "new_object_id": int(created_object_id) if created_object_id is not None else None,
                        "top_association_candidates": candidate_scores_by_patch.get(patch_id, [])[:3],
                        "foreground_depth_filter": dict(
                            patch.metadata.get("foreground_depth_filter", {}) or {}
                        ),
                        "surface_owner_gate": dict(patch.metadata.get("surface_owner_gate", {}) or {}),
                        "current_frame_visibility_gate": dict(
                            patch.metadata.get("current_frame_visibility_gate", {}) or {}
                        ),
                    }
                )

        created_object_ids = sorted(set(created_object_ids))
        matched_existing_ids = sorted(set(matched_existing_ids))
        final_outcome = determine_local_memory_outcome(
            created_object_ids=created_object_ids,
            matched_existing_ids=matched_existing_ids,
            refined_records=refined_records,
            patch_records=patch_records,
            bg_count=bg_count,
            amb_count=amb_count,
            obj_count=obj_count,
        )
        reason = explain_local_memory_outcome(
            final_outcome=final_outcome,
            created_object_ids=created_object_ids,
            matched_existing_ids=matched_existing_ids,
            refined_records=refined_records,
            patch_records=patch_records,
            profile=profile,
            match_threshold=float(pipeline.association.match_threshold),
            min_patch_points=int(pipeline.patch_lifting.min_points),
            min_refine_area=int(pipeline.depth_refinement.min_area),
        )

        raw_record = {
            "proposal_id": raw_id,
            "area": int(proposal.area),
            "confidence": float(proposal.confidence),
            "anchor_assignment": anchor_assignment_lookup.get(raw_id),
            "runtime_vis": {
                "proposal_class": profile.proposal_class if profile is not None else "unknown",
                "objectness_score": float(profile.objectness_score) if profile is not None else 0.0,
                "backgroundness_score": float(profile.backgroundness_score) if profile is not None else 0.0,
                "group_id": int(group_id) if group_id is not None else None,
                "group_member_mask_ids": [int(value) for value in (group.member_mask_ids if group is not None else [])],
                "group_area": int(group.area) if group is not None else 0,
                "group_merge_reason": group.merge_reason if group is not None else "missing",
                "linked_object_id": int(group.linked_object_id) if group is not None and group.linked_object_id is not None else None,
            },
            "refined_components": refined_records,
            "patches": patch_records,
            "created_object_ids": created_object_ids,
            "matched_existing_ids": matched_existing_ids,
            "final_outcome": final_outcome,
            "reason": reason,
        }
        raw_records.append(raw_record)
        outcome_counts[final_outcome] += 1

        for object_id in created_object_ids:
            obj = state.objects.get(object_id)
            related_patch = next((record for record in patch_records if record.get("new_object_id") == object_id), None)
            source_patch_id = int(related_patch["patch_id"]) if related_patch is not None else None
            event_key = (int(object_id), source_patch_id)
            if event_key in seen_new_instance_events:
                continue
            seen_new_instance_events.add(event_key)
            anchor_assignment = anchor_assignment_lookup.get(raw_id) or {}
            new_instance_events.append(
                {
                    "object_id": int(object_id),
                    "source_proposal_id": raw_id,
                    "source_patch_id": source_patch_id,
                    "anchor_id": int(anchor_assignment.get("anchor_id", -1)),
                    "anchor_class_name": str(anchor_assignment.get("class_name", "")),
                    "anchor_confidence": float(anchor_assignment.get("confidence", 0.0)),
                    "reason": reason,
                    "local_point_count": int(len(obj.local_pcd)) if obj is not None else 0,
                    "update_count": int(obj.update_count) if obj is not None else 0,
                    "state": obj.state.value if obj is not None else "missing",
                    "stability_score": float(
                        obj.debug.get("global_instance_substrate", {}).get("stability_score", 0.0)
                    ) if obj is not None else 0.0,
                }
            )

    current_object_ids = set(state.objects)
    object_update = getattr(pipeline, "object_update", None)
    return {
        "frame_id": int(frame.frame_id),
        "raw_proposal_count": len(raw_proposals),
        "source_sam_proposal_count": len(source_sam_proposals),
        "anchor_voted_proposal_count": len(anchor_voted_proposals),
        "runtime_merged_proposal_count": len(runtime_merged_groups),
        "anchor_count": int(len(serialized_anchors)),
        "anchored_proposal_count": int(
            sum(1 for assignment in anchor_assignment_lookup.values() if int(assignment.get("anchor_id", -1)) >= 0)
        ),
        "refined_proposal_count": len(refined_proposals),
        "patch_count": len(patches),
        "bg_patch_count": len(bg_patch_ids),
        "obj_patch_count": len(obj_patch_ids),
        "amb_patch_count": len(amb_patch_ids),
        "new_instance_count": len(new_instance_events),
        "new_instance_events": new_instance_events,
        "contested_residual_count": int(len(contested_residual_events)),
        "contested_residual_events": contested_residual_events,
        "contested_residual_patch_ids": [
            int(patch_id) for patch_id in getattr(association, "contested_object_patches", [])
        ],
        "contested_residual_promoted_object_ids": [
            int(object_id) for object_id in getattr(
                getattr(pipeline, "object_update", None),
                "last_contested_residual_promoted_object_ids",
                [],
            )
        ],
        "hard_failure_count": int(
            sum(
                1
                for record in raw_records
                if record["final_outcome"] in {
                    "depth_refinement_removed",
                    "patch_lifting_failed",
                    "bg_obj_split_background",
                    "bg_obj_split_ambiguous",
                    "bg_obj_split_mixed_non_object",
                    "unresolved_after_split",
                }
            )
        ),
        "new_object_ids_this_frame": sorted(int(object_id) for object_id in (current_object_ids - prev_object_ids)),
        "next_object_id_before": int(prev_next_object_id),
        "next_object_id_after": int(state.next_object_id),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "current_frame_visibility_gate": dict(
            getattr(object_update, "last_current_frame_visibility_gate_stats", {})
            or pipeline.last_frame_debug.get("current_frame_visibility_gate", {})
            or {}
        ),
        "anchors": serialized_anchors,
        "source_sam_proposals": [
            serialize_proposal_stage_record(proposal) for proposal in source_sam_proposals
        ],
        "anchor_voted_proposals": [
            serialize_proposal_stage_record(proposal) for proposal in anchor_voted_proposals
        ],
        "runtime_merged_proposals": [
            serialize_runtime_group_record(group) for group in runtime_merged_groups
        ],
        "raw_proposals": raw_records,
        "overlay_path": None,
    }


def serialize_anchor(anchor: Any) -> dict[str, Any]:
    bbox = np.asarray(_record_field(anchor, "bbox_xyxy", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(4)
    metadata = _record_field(anchor, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "anchor_id": int(_record_field(anchor, "anchor_id", -1)),
        "bbox_xyxy": [float(value) for value in bbox.tolist()],
        "class_name": str(_record_field(anchor, "class_name", "")),
        "confidence": float(_record_field(anchor, "confidence", 0.0)),
        "metadata": metadata,
    }


def serialize_anchor_assignment(assignment: Any) -> dict[str, Any]:
    return {
        "proposal_id": int(_record_field(assignment, "proposal_id", -1)),
        "anchor_id": int(_record_field(assignment, "anchor_id", -1)),
        "class_name": str(_record_field(assignment, "class_name", "")),
        "confidence": float(_record_field(assignment, "confidence", 0.0)),
        "bbox_iou": float(_record_field(assignment, "bbox_iou", 0.0)),
        "center_inside": bool(_record_field(assignment, "center_inside", False)),
        "keepalive": bool(_record_field(assignment, "keepalive", False)),
    }


def serialize_proposal_stage_record(proposal: Any) -> dict[str, Any]:
    bbox = np.asarray(_record_field(proposal, "bbox_xyxy", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(4)
    metadata = _record_field(proposal, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    blocked_candidates = metadata.get("anchor_blocked_candidates", [])
    if not isinstance(blocked_candidates, list):
        blocked_candidates = []
    return {
        "proposal_id": int(_record_field(proposal, "proposal_id", -1)),
        "area": int(_record_field(proposal, "area", 0)),
        "confidence": float(_record_field(proposal, "confidence", 0.0)),
        "backend_name": str(_record_field(proposal, "backend_name", "unknown")),
        "bbox_xyxy": [float(value) for value in bbox.tolist()],
        "anchor_id": int(metadata.get("anchor_id", -1)),
        "anchor_class_name": str(metadata.get("anchor_class_name", "")),
        "anchor_confidence": float(metadata.get("anchor_confidence", 0.0)),
        "anchor_vote_score": float(metadata.get("anchor_vote_score", 0.0)),
        "source_raw_proposal_ids": [int(value) for value in metadata.get("source_raw_proposal_ids", [])],
        "geometry_source": str(metadata.get("geometry_source", "")),
        "anchor_label_strength": str(metadata.get("anchor_label_strength", "")),
        "mask_source": str(metadata.get("mask_source", "")),
        "anchor_proposal_coverage": float(metadata.get("anchor_proposal_coverage", 0.0)),
        "anchor_anchor_coverage": float(metadata.get("anchor_anchor_coverage", 0.0)),
        "anchor_blocked_candidates": list(blocked_candidates),
    }


def serialize_runtime_group_record(group: Any) -> dict[str, Any]:
    bbox = np.asarray(_record_field(group, "merged_bbox_xyxy", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(4)
    linked_object_id = _record_field(group, "linked_object_id", None)
    return {
        "group_id": int(_record_field(group, "group_id", -1)),
        "member_mask_ids": [int(value) for value in _record_field(group, "member_mask_ids", [])],
        "area": int(_record_field(group, "area", 0)),
        "confidence": float(_record_field(group, "confidence", 0.0)),
        "bbox_xyxy": [float(value) for value in bbox.tolist()],
        "linked_object_id": int(linked_object_id) if linked_object_id is not None else None,
        "whole_prior_used": bool(_record_field(group, "whole_prior_used", False)),
        "merge_reason": str(_record_field(group, "merge_reason", "")),
    }


def build_frontend_stage_report_payload(frame_metrics: list[dict[str, Any]]) -> dict[str, int | float]:
    keys = [
        "source_sam_proposal_count",
        "anchor_voted_proposal_count",
        "runtime_merged_proposal_count",
        "runtime_merge_count",
        "anchor_voted_unanchored_count",
        "runtime_anchor_label_missing_edge_count",
        "runtime_anchor_label_mismatch_edge_count",
        "runtime_anchor_identity_mismatch_edge_count",
    ]
    anchor_guided_sam_keys = [
        "source_sam_proposal_count",
        "anchored_proposal_count",
        "unknown_residual_count",
        "dropped_proposal_count",
        "mean_sam_candidates_per_anchor",
        "semantic_blocked_residual_count",
    ]
    totals: dict[str, int | float] = {f"{key}_total": 0 for key in keys}
    totals.update({f"anchor_guided_sam_{key}_total": 0 for key in anchor_guided_sam_keys})
    for metrics in frame_metrics:
        frontend_stage = metrics.get("frontend_stage", {}) or {}
        for key in keys:
            totals[f"{key}_total"] += int(frontend_stage.get(key, 0) or 0)
        anchor_guided_sam = frontend_stage.get("anchor_guided_sam", {}) or {}
        for key in anchor_guided_sam_keys:
            totals[f"anchor_guided_sam_{key}_total"] += float(anchor_guided_sam.get(key, 0) or 0)
    return totals


def build_scheduling_report_payload(frame_metrics: list[dict[str, Any]]) -> dict[str, int | float | bool | str]:
    submitted_count = 0
    hit_count = 0
    miss_count = 0
    wait_sec_total = 0.0
    prefetch_frames = 0
    inline_frames = 0
    exception_count = 0
    last_exception = ""
    enabled = False
    association_score_parallel_used_count = 0
    association_score_parallel_candidate_count_total = 0
    depth_refinement_parallel_used_count = 0
    depth_refinement_parallel_worker_count_max = 1
    for metrics in frame_metrics:
        scheduling = dict(metrics.get("scheduling", {}) or {})
        association_summary = dict(metrics.get("association_summary", {}) or {})
        depth_refinement = dict(metrics.get("depth_refinement", {}) or {})
        enabled = enabled or bool(scheduling.get("prefetch_enabled", False))
        if bool(scheduling.get("prefetch_submitted", False)):
            submitted_count += 1
        if bool(scheduling.get("prefetch_hit", False)):
            hit_count += 1
        if bool(scheduling.get("prefetch_miss", False)):
            miss_count += 1
        try:
            wait_sec_total += float(scheduling.get("prefetch_wait_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
        if str(scheduling.get("frontend_source", "")) == "prefetch":
            prefetch_frames += 1
        if str(scheduling.get("frontend_source", "")) == "inline":
            inline_frames += 1
        exception = str(scheduling.get("prefetch_exception", "") or "")
        if exception:
            exception_count += 1
            last_exception = exception
        association_score_parallel_used_count += int(
            association_summary.get("score_parallel_used_count", 0) or 0
        )
        association_score_parallel_candidate_count_total += int(
            association_summary.get("score_parallel_candidate_count_total", 0) or 0
        )
        if bool(depth_refinement.get("parallel_used", False)):
            depth_refinement_parallel_used_count += 1
        depth_refinement_parallel_worker_count_max = max(
            depth_refinement_parallel_worker_count_max,
            int(depth_refinement.get("parallel_worker_count", 1) or 1),
        )
    return {
        "prefetch_enabled": bool(enabled),
        "prefetch_submitted_count": int(submitted_count),
        "prefetch_hit_count": int(hit_count),
        "prefetch_miss_count": int(miss_count),
        "prefetch_wait_sec_total": float(wait_sec_total),
        "frontend_prefetch_frame_count": int(prefetch_frames),
        "frontend_inline_frame_count": int(inline_frames),
        "frontend_exception_count": int(exception_count),
        "frontend_last_exception": last_exception,
        "association_score_parallel_used_count": int(association_score_parallel_used_count),
        "association_score_parallel_candidate_count_total": int(
            association_score_parallel_candidate_count_total
        ),
        "depth_refinement_parallel_used_count": int(depth_refinement_parallel_used_count),
        "depth_refinement_parallel_worker_count_max": int(depth_refinement_parallel_worker_count_max),
    }


def build_async_refinement_summary(frame_metrics: list[dict[str, Any]]) -> dict[str, int]:
    keys = {
        "sam_proposal_count": "async_refinement_sam_proposal_count_total",
        "fine_proposal_count": "async_refinement_fine_proposal_count_total",
        "fine_patch_count": "async_refinement_fine_patch_count_total",
        "replaced_observation_count": "async_refinement_replaced_observation_count_total",
        "inserted_observation_count": "async_refinement_inserted_observation_count_total",
        "updated_object_count": "async_refinement_updated_object_count_total",
    }
    totals = {output_key: 0 for output_key in keys.values()}
    for metrics in frame_metrics:
        raw_refinement = metrics.get("async_refinement", {}) or {}
        if not isinstance(raw_refinement, dict):
            continue
        refinement = dict(raw_refinement)
        for input_key, output_key in keys.items():
            try:
                value = float(refinement.get(input_key, 0) or 0)
            except (TypeError, ValueError):
                continue
            if value < 0.0 or not np.isfinite(value):
                continue
            totals[output_key] += int(value)
    return totals


def build_stage_timing_summary(frame_metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stage_values: dict[str, list[tuple[int, float]]] = {}
    for metrics in frame_metrics:
        try:
            frame_id = int(metrics.get("frame_id", -1))
        except (TypeError, ValueError, OverflowError):
            frame_id = -1
        timings = dict(metrics.get("stage_timings", {}) or {})
        for stage, value in timings.items():
            try:
                timing_value = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(timing_value):
                continue
            stage_values.setdefault(str(stage), []).append((frame_id, timing_value))
    summary: dict[str, dict[str, Any]] = {}
    for stage, pairs in sorted(stage_values.items()):
        if not pairs:
            continue
        values = np.asarray([value for _frame_id, value in pairs], dtype=np.float64)
        median = float(np.median(values))
        sorted_values = np.sort(values)
        p95_index = int(round(0.95 * (len(sorted_values) - 1)))
        p95 = float(sorted_values[p95_index])
        max_index = int(np.argmax(values))
        outlier_frame_ids = [
            int(frame_id)
            for frame_id, value in pairs
            if float(value) >= p95 and float(value) > median
        ]
        summary[stage] = {
            "mean_sec": float(np.mean(values)),
            "median_sec": median,
            "p95_sec": p95,
            "max_sec": float(np.max(values)),
            "total_sec": float(np.sum(values)),
            "max_frame_id": int(pairs[max_index][0]),
            "outlier_frame_ids": outlier_frame_ids[:10],
        }
    return summary


def build_association_pruning_summary(frame_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [
        dict(metrics.get("association_summary", {}) or {})
        for metrics in frame_metrics
        if metrics.get("association_summary")
    ]
    if not summaries:
        return {}
    candidate_score_count = int(sum(int(item.get("candidate_score_count", 0)) for item in summaries))
    geometry_score_count = int(sum(int(item.get("geometry_score_count", 0)) for item in summaries))
    return {
        "frame_count": int(len(summaries)),
        "candidate_score_count_total": candidate_score_count,
        "geometry_score_count_total": geometry_score_count,
        "geometry_pruned_candidate_count_total": int(max(0, candidate_score_count - geometry_score_count)),
        "mean_candidate_score_count": float(candidate_score_count / max(len(summaries), 1)),
        "mean_geometry_score_count": float(geometry_score_count / max(len(summaries), 1)),
    }


def _record_field(record: Any, name: str, default: Any) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def determine_local_memory_outcome(
    *,
    created_object_ids: list[int],
    matched_existing_ids: list[int],
    refined_records: list[dict[str, Any]],
    patch_records: list[dict[str, Any]],
    bg_count: int,
    amb_count: int,
    obj_count: int,
) -> str:
    if created_object_ids:
        return "new_instance_created"
    if matched_existing_ids:
        return "matched_existing_instance"
    if any(record.get("association_outcome") == "contested_residual" for record in patch_records):
        return "contested_residual"
    if not refined_records:
        return "depth_refinement_removed"
    if refined_records and not patch_records:
        return "patch_lifting_failed"
    if patch_records and obj_count == 0 and bg_count > 0 and amb_count == 0:
        return "bg_obj_split_background"
    if patch_records and obj_count == 0 and amb_count > 0 and bg_count == 0:
        return "bg_obj_split_ambiguous"
    if patch_records and obj_count == 0 and (bg_count > 0 or amb_count > 0):
        return "bg_obj_split_mixed_non_object"
    return "unresolved_after_split"


def explain_local_memory_outcome(
    *,
    final_outcome: str,
    created_object_ids: list[int],
    matched_existing_ids: list[int],
    refined_records: list[dict[str, Any]],
    patch_records: list[dict[str, Any]],
    profile,
    match_threshold: float,
    min_patch_points: int,
    min_refine_area: int,
) -> str:
    proposal_class = getattr(profile, "proposal_class", "unknown")

    if final_outcome == "new_instance_created":
        best_candidate = None
        for patch_record in patch_records:
            candidates = patch_record.get("top_association_candidates") or []
            if candidates and (best_candidate is None or candidates[0]["total_score"] > best_candidate["total_score"]):
                best_candidate = candidates[0]
        if best_candidate is not None:
            return (
                f"created new object(s) {created_object_ids}; object patch survived to association but no existing "
                f"instance exceeded match_threshold={match_threshold:.2f} "
                f"(best existing {best_candidate['object_id']} score={best_candidate['total_score']:.3f})."
            )
        return f"created new object(s) {created_object_ids}; object patch reached association with no acceptable existing instance."

    if final_outcome == "matched_existing_instance":
        best_match = None
        for patch_record in patch_records:
            matched_existing = patch_record.get("matched_existing")
            if matched_existing is not None and (
                best_match is None or matched_existing["total_score"] > best_match["total_score"]
            ):
                best_match = matched_existing
        if best_match is not None:
            return (
                f"did not create a new instance because the patch matched existing object(s) {matched_existing_ids}; "
                f"best match object {best_match['matched_object_id']} score={best_match['total_score']:.3f}."
            )
        return f"did not create a new instance because it matched existing object(s) {matched_existing_ids}."

    if final_outcome == "contested_residual":
        contested = next(
            (record for record in patch_records if record.get("association_outcome") == "contested_residual"),
            {},
        )
        parent_id = contested.get("contested_parent_object_id", -1)
        patch_label = contested.get("contested_patch_label", "")
        parent_label = contested.get("contested_parent_label", "")
        return (
            f"frontend observation '{patch_label}' was blocked from cross-label parent object "
            f"{parent_id} ('{parent_label}') and routed to contested residual tracking."
        )

    if final_outcome == "depth_refinement_removed":
        return (
            f"no refined proposal survived after depth refinement; all connected components were empty or below "
            f"min_mask_area_after_refine={min_refine_area}. runtime_vis class={proposal_class}."
        )

    if final_outcome == "patch_lifting_failed":
        valid_depth = max((record["valid_depth_point_count"] for record in refined_records), default=0)
        return (
            f"refined proposal existed but no 3D patch was lifted; best refined component had "
            f"{valid_depth} valid depth points and patch lifting needs at least {min_patch_points}."
        )

    if final_outcome == "bg_obj_split_background":
        return f"patches were classified as background before association. runtime_vis class={proposal_class}."

    if final_outcome == "bg_obj_split_ambiguous":
        return (
            f"patches remained ambiguous after bg/object split; they reached the ambiguous-association lane "
            f"but did not match an existing instance or qualify for object creation. runtime_vis class={proposal_class}."
        )

    if final_outcome == "bg_obj_split_mixed_non_object":
        return (
            "all lifted patches resolved into background or non-promoted ambiguous outcomes, "
            "so none became tracked object updates."
        )

    return "proposal survived early stages but did not end in a tracked local-memory instance; inspect patch-level association candidates."


def summarize_local_memory_audits(frame_audits: list[dict[str, Any]], state: SystemState) -> dict[str, Any]:
    outcome_counts: Counter[str] = Counter()
    anchor_class_counts: Counter[str] = Counter()
    created_frames: list[int] = []
    anchor_frames: list[int] = []
    overlay_frames: list[int] = []
    total_anchor_detections = 0
    total_anchored_proposals = 0
    for frame_audit in frame_audits:
        if frame_audit.get("new_instance_count", 0) > 0:
            created_frames.append(int(frame_audit["frame_id"]))
        anchor_count = int(frame_audit.get("anchor_count", 0))
        if anchor_count > 0:
            anchor_frames.append(int(frame_audit["frame_id"]))
        total_anchor_detections += anchor_count
        total_anchored_proposals += int(frame_audit.get("anchored_proposal_count", 0))
        if frame_audit.get("overlay_path"):
            overlay_frames.append(int(frame_audit["frame_id"]))
        for outcome, count in frame_audit.get("outcome_counts", {}).items():
            outcome_counts[str(outcome)] += int(count)
        for anchor in frame_audit.get("anchors", []):
            class_name = str(anchor.get("class_name", "")).strip()
            if class_name:
                anchor_class_counts[class_name] += 1

    return {
        "frame_count": int(len(frame_audits)),
        "frames_with_new_instances": created_frames,
        "frames_with_anchors": anchor_frames,
        "frames_with_saved_overlays": overlay_frames,
        "total_new_instance_events": int(sum(item.get("new_instance_count", 0) for item in frame_audits)),
        "total_anchor_detections": int(total_anchor_detections),
        "total_anchored_proposal_events": int(total_anchored_proposals),
        "anchor_class_counts": dict(sorted(anchor_class_counts.items())),
        "raw_outcome_counts": dict(sorted(outcome_counts.items())),
        "final_object_count": int(len(state.objects)),
        "objects_with_local_memory_points": int(sum(1 for obj in state.objects.values() if len(obj.local_pcd) > 0)),
    }


def render_local_memory_audit_log(frame_audits: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for frame_audit in frame_audits:
        frame_id = int(frame_audit["frame_id"])
        if (
            frame_audit.get("new_instance_count", 0) <= 0
            and frame_audit.get("hard_failure_count", 0) <= 0
            and frame_audit.get("anchor_count", 0) <= 0
        ):
            continue
        lines.append(f"frame={frame_id:04d}")
        if frame_audit.get("overlay_path"):
            lines.append(f"overlay={frame_audit['overlay_path']}")
        if frame_audit.get("anchor_count", 0) > 0:
            lines.append(
                "anchors="
                f"{frame_audit['anchor_count']} anchored_proposals={frame_audit.get('anchored_proposal_count', 0)}"
            )
        for event in frame_audit.get("new_instance_events", []):
            anchor_suffix = ""
            if int(event.get("anchor_id", -1)) >= 0:
                anchor_suffix = (
                    f" anchor={event['anchor_id']}:{event.get('anchor_class_name', '')}"
                    f" conf={float(event.get('anchor_confidence', 0.0)):.2f}"
                )
            lines.append(
                "  new_instance "
                f"object_id={event['object_id']} source_proposal={event['source_proposal_id']} "
                f"source_patch={event['source_patch_id']} local_points={event['local_point_count']} "
                f"stability={event['stability_score']:.3f}{anchor_suffix} reason={event['reason']}"
            )
        for raw_record in frame_audit.get("raw_proposals", []):
            if raw_record.get("final_outcome") == "new_instance_created":
                continue
            anchor_assignment = raw_record.get("anchor_assignment") or {}
            anchor_suffix = ""
            if int(anchor_assignment.get("anchor_id", -1)) >= 0:
                anchor_suffix = (
                    f" anchor={anchor_assignment['anchor_id']}:{anchor_assignment.get('class_name', '')}"
                    f" iou={float(anchor_assignment.get('bbox_iou', 0.0)):.2f}"
                )
            lines.append(
                "  raw_proposal "
                f"id={raw_record['proposal_id']} outcome={raw_record['final_outcome']} "
                f"reason={raw_record['reason']}{anchor_suffix}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_local_memory_audit_overlay(
    frame,
    source_proposals: list[Any],
    raw_proposals: list[Any],
    raw_records: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    output_path: Path,
    runtime_groups: list[Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_groups = list(runtime_groups or [])
    yolo_panel = add_panel_title(
        anchor_overlay_image(np.asarray(frame.rgb, dtype=np.uint8), anchors),
        "YOLO Anchors",
    )
    sam_panel = add_panel_title(
        proposal_overlay_image(np.asarray(frame.rgb, dtype=np.uint8), source_proposals),
        "SAM2 Source Masks",
    )
    anchor_vote_panel = add_panel_title(
        anchor_vote_overlay_image(np.asarray(frame.rgb, dtype=np.uint8), raw_proposals),
        "Anchor Votes on SAM2 Masks",
    )
    runtime_panel = add_panel_title(
        runtime_group_overlay_image(np.asarray(frame.rgb, dtype=np.uint8), runtime_groups),
        "RuntimeVis Merged Masks",
    )
    intersection_panel = add_panel_title(
        _intersection_overlay_image(
            np.asarray(frame.rgb, dtype=np.uint8),
            raw_proposals=raw_proposals,
            raw_records=raw_records,
            anchors=anchors,
        ),
        "Patch Outcomes",
    )
    save_multi_panel([yolo_panel, sam_panel, anchor_vote_panel, runtime_panel, intersection_panel], output_path)


def _intersection_overlay_image(
    rgb: np.ndarray,
    *,
    raw_proposals: list[Any],
    raw_records: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> Image.Image:
    status_lookup = {int(record["proposal_id"]): record for record in raw_records}
    canvas = np.asarray(rgb, dtype=np.uint8).copy()
    overlay = canvas.astype(np.float32)

    for proposal in sorted(raw_proposals, key=lambda item: int(item.area), reverse=True):
        record = status_lookup.get(int(proposal.proposal_id), {})
        color = np.asarray(audit_color(record.get("final_outcome", "unknown")), dtype=np.float32)
        mask = np.asarray(proposal.mask, dtype=bool)
        if mask.any():
            overlay[mask] = overlay[mask] * 0.72 + color * 0.28

    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for proposal in raw_proposals:
        record = status_lookup.get(int(proposal.proposal_id), {})
        color = audit_color(record.get("final_outcome", "unknown"))
        bbox = np.asarray(proposal.bbox_xyxy, dtype=np.float32)
        x1, y1, x2, y2 = [int(round(value)) for value in bbox.tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text(
            (x1, max(0, y1 - 12)),
            build_overlay_label(record),
            fill=color,
        )

    for anchor in anchors:
        bbox = np.asarray(anchor.get("bbox_xyxy", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(4)
        x1, y1, x2, y2 = [int(round(value)) for value in bbox.tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        color = anchor_color(int(anchor.get("anchor_id", -1)))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    return image


def audit_color(outcome: str) -> tuple[int, int, int]:
    mapping = {
        "new_instance_created": (72, 201, 126),
        "matched_existing_instance": (77, 144, 254),
        "bg_obj_split_background": (140, 140, 140),
        "bg_obj_split_ambiguous": (255, 177, 66),
        "bg_obj_split_mixed_non_object": (214, 133, 27),
        "patch_lifting_failed": (231, 111, 81),
        "depth_refinement_removed": (186, 85, 211),
        "unresolved_after_split": (255, 215, 0),
    }
    return mapping.get(outcome, (255, 255, 255))


def build_overlay_label(raw_record: dict[str, Any]) -> str:
    proposal_id = int(raw_record.get("proposal_id", -1))
    outcome = str(raw_record.get("final_outcome", "unknown"))
    anchor_assignment = raw_record.get("anchor_assignment") or {}
    anchor_suffix = ""
    if int(anchor_assignment.get("anchor_id", -1)) >= 0:
        anchor_suffix = f"->a{int(anchor_assignment['anchor_id'])}"
    if outcome == "new_instance_created":
        object_ids = ",".join(str(value) for value in raw_record.get("created_object_ids", []))
        return f"r{proposal_id}{anchor_suffix}: new {object_ids}"
    if outcome == "matched_existing_instance":
        object_ids = ",".join(str(value) for value in raw_record.get("matched_existing_ids", []))
        return f"r{proposal_id}{anchor_suffix}: match {object_ids}"
    if outcome == "bg_obj_split_background":
        return f"r{proposal_id}{anchor_suffix}: bg"
    if outcome == "bg_obj_split_ambiguous":
        return f"r{proposal_id}{anchor_suffix}: amb"
    if outcome == "patch_lifting_failed":
        return f"r{proposal_id}{anchor_suffix}: no3d"
    if outcome == "depth_refinement_removed":
        return f"r{proposal_id}{anchor_suffix}: refine_drop"
    return f"r{proposal_id}{anchor_suffix}: {outcome}"


def build_anchor_overlay_label(anchor: dict[str, Any]) -> str:
    anchor_id = int(anchor.get("anchor_id", -1))
    class_name = str(anchor.get("class_name", "")).strip()
    confidence = float(anchor.get("confidence", 0.0))
    label = f"A{anchor_id}"
    if class_name:
        label += f" {class_name}"
    if confidence > 0.0:
        label += f" {confidence:.2f}"
    return label


def anchor_color(anchor_id: int) -> tuple[int, int, int]:
    palette = [
        (255, 106, 0),
        (255, 196, 0),
        (255, 80, 80),
        (0, 214, 201),
        (255, 149, 5),
        (122, 162, 247),
    ]
    if anchor_id < 0:
        return (255, 106, 0)
    return palette[anchor_id % len(palette)]


def accumulate_geometry(
    accum: dict[tuple[int, int, int], list[np.ndarray | int]],
    frame,
    sample_stride: int,
    voxel_size: float,
) -> None:
    rgb = frame.rgb[::sample_stride, ::sample_stride]
    depth = frame.depth[::sample_stride, ::sample_stride]
    u, v = np.meshgrid(
        np.arange(0, frame.rgb.shape[1], sample_stride),
        np.arange(0, frame.rgb.shape[0], sample_stride),
    )
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return

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


def finalize_geometry_accum(accum: dict[tuple[int, int, int], list[np.ndarray | int]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray([item[0] for item in accum.values()], dtype=np.float32)
    colors = np.clip(np.asarray([item[1] for item in accum.values()], dtype=np.float32), 0, 255).astype(np.uint8)
    return points, colors


def build_tsdf_backbone_records(state: SystemState) -> np.ndarray:
    rows = []
    voxel_size = state.tsdf_volume.voxel_size
    for voxel_key, owner_support in state.tsdf_volume.owner_support.items():
        object_id = int(owner_support.owner_id)
        if object_id < 0 or object_id not in state.objects:
            continue
        obj = state.objects[object_id]
        point = (np.asarray(voxel_key, dtype=np.float32) + 0.5) * float(voxel_size)
        color = semantic_surface_color(object_semantic_label(obj), int(object_id))
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
    return np.asarray(rows, dtype=PLY_DTYPE)


def object_stability_score(obj) -> float:
    return float(obj.debug.get("global_instance_substrate", {}).get("stability_score", 0.0))


def object_semantic_label(obj) -> str:
    return object_export_semantic_label(obj)


def build_pool_semantic_records(state: SystemState) -> np.ndarray:
    rows = []
    for object_id, obj in sorted(state.objects.items()):
        points = np.asarray(obj.local_pcd, dtype=np.float32)
        if points.size == 0:
            continue
        color = semantic_surface_color(object_semantic_label(obj), int(object_id))
        stability = object_stability_score(obj)
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
    return np.asarray(rows, dtype=PLY_DTYPE)


def build_dense_surface_records(state: SystemState) -> np.ndarray:
    rows = []
    for object_id, entry in sorted(state.dense_surface_map.entries.items()):
        if not entry.resident or len(entry.points) == 0:
            continue
        obj = state.objects.get(int(object_id))
        if obj is None:
            continue
        color = semantic_surface_color(entry.semantic_label, int(object_id))
        stability = object_stability_score(obj)
        for point in np.asarray(entry.points, dtype=np.float32):
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
    return np.asarray(rows, dtype=PLY_DTYPE)


def build_structural_overlay_records(
    structural_overlay_map,
    direct_ids_to_class: dict[int, int],
    direct_ids_to_name: dict[int, str],
) -> np.ndarray:
    rows = []
    label_to_direct_id = {str(label).strip().lower(): int(object_id) for object_id, label in direct_ids_to_name.items()}
    voxel_size = float(structural_overlay_map.voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("structural overlay voxel_size must be positive and finite")
    for voxel_key, voxel in sorted(structural_overlay_map.voxels.items()):
        label = str(voxel.top_label).strip().lower()
        object_id = label_to_direct_id.get(label)
        if object_id is None or object_id not in direct_ids_to_class:
            continue
        center = (np.asarray(voxel_key, dtype=np.float32) + 0.5) * voxel_size
        color = semantic_surface_color(label, int(object_id))
        rows.append(
            (
                float(center[0]),
                float(center[1]),
                float(center[2]),
                color[0],
                color[1],
                color[2],
                int(object_id),
                0,
                float(voxel.top_support),
            )
        )
    return np.asarray(rows, dtype=PLY_DTYPE)


def build_pool_debug_records(state: SystemState, downsample_voxel: float = 0.0) -> np.ndarray:
    rows = []
    for object_id, obj in sorted(state.objects.items()):
        points = np.asarray(obj.local_pcd, dtype=np.float32)
        if points.size == 0:
            continue
        if downsample_voxel > 0.0:
            points = voxel_downsample(points, downsample_voxel)
        color = semantic_surface_color(object_semantic_label(obj), int(object_id))
        stability = object_stability_score(obj)
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
    return np.asarray(rows, dtype=PLY_DTYPE)


def build_local_memory_records(state: SystemState) -> np.ndarray:
    """Compatibility wrapper for the pool-geometry debug export."""
    return build_pool_debug_records(state)


def project_instances_to_dense_points(
    dense_points: np.ndarray,
    instance_records: np.ndarray,
    instance_voxel_size: float,
    neighbor_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.full(len(dense_points), -1, dtype=np.int32)
    state_ids = np.zeros(len(dense_points), dtype=np.uint8)
    supports = np.zeros(len(dense_points), dtype=np.float32)
    if len(instance_records) == 0:
        return labels, state_ids, supports

    owner_map: dict[tuple[int, int, int], tuple[int, int, float]] = {}
    instance_points = np.stack([instance_records["x"], instance_records["y"], instance_records["z"]], axis=1)
    instance_indices = np.floor(instance_points / instance_voxel_size).astype(np.int32)

    for voxel, row in zip(instance_indices, instance_records):
        owner_map[(int(voxel[0]), int(voxel[1]), int(voxel[2]))] = (
            int(row["object_id"]),
            int(row["state_id"]),
            float(row["support"]),
        )

    dense_indices = np.floor(dense_points / instance_voxel_size).astype(np.int32)

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


def project_structural_overlay_to_dense_points(
    dense_points: np.ndarray,
    structural_overlay_map,
    direct_ids_to_class: dict[int, int],
    class_names: dict[int, str],
    neighbor_radius: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    overlay_ids = np.full(len(dense_points), -1, dtype=np.int32)
    overlay_supports = np.zeros(len(dense_points), dtype=np.float32)
    if len(dense_points) == 0 or not getattr(structural_overlay_map, "voxels", None):
        return overlay_ids, overlay_supports

    label_to_direct_id = {
        str(class_names[class_id]).strip().lower(): int(object_id)
        for object_id, class_id in direct_ids_to_class.items()
        if int(class_id) in class_names
    }
    if not label_to_direct_id:
        return overlay_ids, overlay_supports

    voxel_size = float(structural_overlay_map.voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("structural overlay voxel_size must be positive and finite")

    dense_indices = np.floor(dense_points / voxel_size).astype(np.int32)
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in range(-int(neighbor_radius), int(neighbor_radius) + 1)
        for dy in range(-int(neighbor_radius), int(neighbor_radius) + 1)
        for dz in range(-int(neighbor_radius), int(neighbor_radius) + 1)
    ]
    for index, voxel in enumerate(dense_indices):
        key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        entry = structural_overlay_map.voxels.get(key)
        if entry is None and neighbor_radius > 0:
            best_entry = None
            best_support = -1.0
            for dx, dy, dz in neighbor_offsets:
                candidate = structural_overlay_map.voxels.get((key[0] + dx, key[1] + dy, key[2] + dz))
                if candidate is None:
                    continue
                if candidate.top_support > best_support:
                    best_support = candidate.top_support
                    best_entry = candidate
            entry = best_entry
        if entry is None:
            continue
        direct_id = label_to_direct_id.get(str(entry.top_label).strip().lower())
        if direct_id is None:
            continue
        overlay_ids[index] = int(direct_id)
        overlay_supports[index] = float(entry.top_support)
    return overlay_ids, overlay_supports


def apply_conservative_structural_overlay(
    labels: np.ndarray,
    state_ids: np.ndarray,
    supports: np.ndarray,
    overlay_ids: np.ndarray,
    overlay_supports: np.ndarray,
    state: SystemState,
    structure_labels: set[str],
    protected_labels: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    fused_labels = np.asarray(labels, dtype=np.int32).copy()
    fused_state_ids = np.asarray(state_ids, dtype=np.uint8).copy()
    fused_supports = np.asarray(supports, dtype=np.float32).copy()
    overlay_ids = np.asarray(overlay_ids, dtype=np.int32)
    overlay_supports = np.asarray(overlay_supports, dtype=np.float32)
    arrays = {
        "labels": fused_labels,
        "state_ids": fused_state_ids,
        "supports": fused_supports,
        "overlay_ids": overlay_ids,
        "overlay_supports": overlay_supports,
    }
    if any(array.ndim != 1 for array in arrays.values()):
        shapes = {name: array.shape for name, array in arrays.items()}
        raise ValueError(f"structural overlay fusion arrays must be 1-D with the same length; shapes={shapes}")
    expected_length = len(fused_labels)
    if any(len(array) != expected_length for array in arrays.values()):
        lengths = {name: len(array) for name, array in arrays.items()}
        raise ValueError(f"structural overlay fusion arrays must have the same length; lengths={lengths}")
    normalized_structure = {str(label).strip().lower() for label in structure_labels}
    normalized_protected = {str(label).strip().lower() for label in protected_labels}
    summary = {
        "overlay_candidate_point_count": int(np.count_nonzero(overlay_ids < 0)),
        "overlay_replaced_point_count": 0,
        "overlay_protected_point_count": 0,
    }
    for index, overlay_id in enumerate(overlay_ids):
        if int(overlay_id) >= 0:
            continue
        current_id = int(fused_labels[index])
        can_replace = current_id < 0
        if current_id >= 0:
            obj = state.objects.get(current_id)
            current_label = object_semantic_label(obj).strip().lower() if obj is not None else ""
            if current_label in normalized_protected:
                summary["overlay_protected_point_count"] += 1
                continue
            can_replace = (not current_label) or current_label in normalized_structure
        if not can_replace:
            summary["overlay_protected_point_count"] += 1
            continue
        fused_labels[index] = int(overlay_id)
        fused_state_ids[index] = 0
        fused_supports[index] = float(overlay_supports[index])
        summary["overlay_replaced_point_count"] += 1
    return fused_labels, fused_state_ids, fused_supports, summary


def evaluate_semantics(
    gt_vertices: np.ndarray,
    gt_labels: np.ndarray,
    dense_points: np.ndarray,
    object_ids: np.ndarray,
    class_names: dict[int, str],
    direct_object_label_ids: dict[int, int] | None = None,
) -> dict[str, Any]:
    tree = cKDTree(dense_points)
    _, nn_indices = tree.query(gt_vertices, k=1, workers=-1)
    nearest_object_ids = object_ids[nn_indices]
    direct_object_label_ids = {int(key): int(value) for key, value in (direct_object_label_ids or {}).items()}

    per_object_votes: dict[int, Counter] = defaultdict(Counter)
    valid_gt_mask = gt_labels >= 0
    for object_id, gt_label in zip(nearest_object_ids[valid_gt_mask], gt_labels[valid_gt_mask]):
        object_id = int(object_id)
        if object_id < 0 and object_id not in direct_object_label_ids:
            continue
        if object_id in direct_object_label_ids:
            continue
        per_object_votes[object_id][int(gt_label)] += 1

    object_to_class: dict[int, int] = {}
    for object_id, counter in per_object_votes.items():
        if counter:
            object_to_class[object_id] = counter.most_common(1)[0][0]
    object_to_class.update(direct_object_label_ids)

    pred_labels = np.full_like(gt_labels, fill_value=-1)
    for index, object_id in enumerate(nearest_object_ids):
        object_id = int(object_id)
        if object_id < 0 and object_id not in direct_object_label_ids:
            continue
        pred_labels[index] = object_to_class.get(object_id, -1)

    valid_gt_classes = sorted(int(x) for x in np.unique(gt_labels[gt_labels >= 0]))
    per_class = []
    class_ious = {}
    class_accs = {}
    gt_class_counts = {}

    for class_id in valid_gt_classes:
        gt_mask = gt_labels == class_id
        pred_mask = pred_labels == class_id
        tp = int(np.count_nonzero(gt_mask & pred_mask))
        fp = int(np.count_nonzero(~gt_mask & pred_mask))
        fn = int(np.count_nonzero(gt_mask & ~pred_mask))
        gt_count = int(np.count_nonzero(gt_mask))
        iou = float(tp / max(tp + fp + fn, 1))
        acc = float(tp / max(gt_count, 1))
        class_name = class_names.get(class_id, f"class_{class_id}")
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "iou": iou,
                "acc": acc,
                "gt_count": gt_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
        class_ious[class_name] = iou
        class_accs[class_name] = acc
        gt_class_counts[class_name] = gt_count

    miou = float(np.mean([item["iou"] for item in per_class])) if per_class else 0.0
    macc = float(np.mean([item["acc"] for item in per_class])) if per_class else 0.0
    total_gt = float(sum(item["gt_count"] for item in per_class))
    fmiou = float(sum(item["iou"] * item["gt_count"] for item in per_class) / max(total_gt, 1.0))
    fmacc = float(sum(item["acc"] * item["gt_count"] for item in per_class) / max(total_gt, 1.0))

    correctness = np.zeros((len(gt_labels), 3), dtype=np.uint8)
    correctness[:] = (90, 90, 90)
    valid_eval = gt_labels >= 0
    correctness[valid_eval & (pred_labels == gt_labels)] = (72, 201, 126)
    correctness[valid_eval & (pred_labels != gt_labels)] = (231, 111, 81)

    return {
        "pred_labels": pred_labels,
        "gt_labels": gt_labels,
        "nearest_object_ids": nearest_object_ids,
        "object_to_class": object_to_class,
        "per_class": per_class,
        "miou": miou,
        "macc": macc,
        "fmiou": fmiou,
        "fmacc": fmacc,
        "class_ious": class_ious,
        "class_accs": class_accs,
        "gt_class_counts": gt_class_counts,
        "correctness_colors": correctness,
    }


def write_eval_artifacts(
    eval_dir: Path,
    gt_vertices: np.ndarray,
    gt_labels: np.ndarray,
    evaluation: dict[str, Any],
    class_names: dict[int, str],
    *,
    write_mesh_ply: bool = True,
) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)

    if write_mesh_ply:
        pred_labels = evaluation["pred_labels"]
        correctness = evaluation["correctness_colors"]
        gt_colors = labels_to_class_colors(gt_labels, class_names=class_names)
        pred_colors = labels_to_class_colors(pred_labels, class_names=class_names)

        write_vertex_ply(
            eval_dir / "room0_gtmesh_gt_semantic.ply",
            gt_vertices,
            gt_colors,
            gt_labels.astype(np.int32),
            np.zeros(len(gt_vertices), dtype=np.uint8),
            np.zeros(len(gt_vertices), dtype=np.float32),
        )
        write_vertex_ply(
            eval_dir / "room0_gtmesh_pred_semantic.ply",
            gt_vertices,
            pred_colors,
            pred_labels.astype(np.int32),
            np.zeros(len(gt_vertices), dtype=np.uint8),
            np.zeros(len(gt_vertices), dtype=np.float32),
        )
        write_vertex_ply(
            eval_dir / "room0_gtmesh_semantic_correctness.ply",
            gt_vertices,
            correctness,
            pred_labels.astype(np.int32),
            np.zeros(len(gt_vertices), dtype=np.uint8),
            np.zeros(len(gt_vertices), dtype=np.float32),
        )

    results = {
        "miou": evaluation["miou"],
        "macc": evaluation["macc"],
        "fmiou": evaluation["fmiou"],
        "fmacc": evaluation["fmacc"],
    }
    (eval_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (eval_dir / "classes_iou.json").write_text(json.dumps(evaluation["class_ious"], indent=2), encoding="utf-8")
    (eval_dir / "classes_acc.json").write_text(json.dumps(evaluation["class_accs"], indent=2), encoding="utf-8")

    stats_lines = ["class,acc,iou"]
    for item in evaluation["per_class"]:
        stats_lines.append(f"{item['class_name']},{item['acc']:.6f},{item['iou']:.6f}")
    (eval_dir / "statistics.txt").write_text("\n".join(stats_lines), encoding="utf-8")


def write_final_object_semantic_audit(
    scene_dir: Path,
    state: SystemState,
    evaluation: dict[str, Any],
    class_names: dict[int, str],
) -> dict[str, Any]:
    audit = build_final_object_semantic_audit(state, evaluation, class_names)
    (scene_dir / "final_object_semantic_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    (scene_dir / "final_object_semantic_audit.md").write_text(
        render_final_object_semantic_audit(audit),
        encoding="utf-8",
    )
    return audit


def build_final_object_semantic_audit(
    state: SystemState,
    evaluation: dict[str, Any],
    class_names: dict[int, str],
) -> dict[str, Any]:
    """Summarize final object semantics and GT-class absorption.

    The report intentionally separates detector-anchor votes from GT-majority
    evaluation. It is a debugging aid for finding classes that were observed by
    anchors but absorbed into another final object instance.
    """
    per_object_gt_votes = _per_object_gt_vote_counters(evaluation)
    object_class_lookup = {
        int(object_id): int(class_id)
        for object_id, class_id in (evaluation.get("object_to_class", {}) or {}).items()
    }
    per_class_metrics = {
        int(item["class_id"]): item
        for item in evaluation.get("per_class", [])
    }
    gt_class_counts: Counter[int] = Counter(
        int(class_id)
        for class_id in np.asarray(evaluation.get("gt_labels", []), dtype=np.int64)
        if int(class_id) >= 0
    )

    object_records: list[dict[str, Any]] = []
    object_record_by_id: dict[int, dict[str, Any]] = {}
    for object_id, obj in sorted(state.objects.items()):
        object_id = int(object_id)
        gt_votes = per_object_gt_votes.get(object_id, Counter())
        gt_top_classes = _top_gt_class_records(gt_votes, class_names)
        gt_majority = gt_top_classes[0] if gt_top_classes else None
        source_anchor_classes = _source_anchor_class_records(obj)
        anchor_semantics = _json_anchor_semantics(obj.debug.get("anchor_semantics", {}))
        semantic_commit = object_semantic_commit_state(obj)
        semantic_export = object_export_semantic_state(obj)
        record = {
            "object_id": object_id,
            "state": obj.state.value,
            "surface_tier": obj.surface_tier.value,
            "exported_label": str(semantic_export["export_label"]),
            "export_state": str(semantic_export["export_state"]),
            "export_source": str(semantic_export["export_source"]),
            "export_reason": str(semantic_export["export_reason"]),
            "semantic_state": str(semantic_commit["semantic_state"]),
            "committed_label": str(semantic_commit["committed_label"]),
            "posterior_label": str(semantic_commit["posterior_label"]),
            "commit_reason": str(semantic_commit["commit_reason"]),
            "contextual_label_score_sum": {
                str(label): float(score)
                for label, score in (anchor_semantics.get("contextual_label_score_sum", {}) or {}).items()
            },
            "point_count": int(len(obj.local_pcd)),
            "observation_count": int(len(obj.observations)),
            "creation_frame": int(obj.creation_frame),
            "last_seen_frame": int(obj.last_seen_frame),
            "update_count": int(obj.update_count),
            "bbox_min": [float(value) for value in np.asarray(obj.bbox_min, dtype=np.float32).tolist()],
            "bbox_max": [float(value) for value in np.asarray(obj.bbox_max, dtype=np.float32).tolist()],
            "anchor_semantics": anchor_semantics,
            "source_anchor_classes": source_anchor_classes,
            "source_anchor_top_class": source_anchor_classes[0]["class_name"] if source_anchor_classes else "",
            "gt_total_votes": int(sum(gt_votes.values())),
            "gt_majority": gt_majority,
            "gt_top_classes": gt_top_classes,
            "gt_mixed_classes": gt_top_classes[1:] if len(gt_top_classes) > 1 else [],
            "eval_majority_class_id": int(object_class_lookup.get(object_id, -1)),
        }
        object_records.append(record)
        object_record_by_id[object_id] = record

    class_records: list[dict[str, Any]] = []
    class_ids = sorted(set(gt_class_counts) | set(per_class_metrics))
    for class_id in class_ids:
        gt_count = int(gt_class_counts.get(class_id, 0))
        class_name = class_names.get(class_id, f"class_{class_id}")
        covering_objects = []
        covered_count = 0
        absorbed_by_other_majority_count = 0
        matching_majority_count = 0
        for object_id, gt_votes in sorted(per_object_gt_votes.items()):
            count = int(gt_votes.get(class_id, 0))
            if count <= 0:
                continue
            covered_count += count
            object_record = object_record_by_id.get(object_id, {})
            majority = object_record.get("gt_majority") or {}
            majority_class_id = int(majority.get("class_id", -1)) if isinstance(majority, dict) else -1
            absorbed = bool(majority_class_id >= 0 and majority_class_id != class_id)
            if absorbed:
                absorbed_by_other_majority_count += count
            else:
                matching_majority_count += count
            covering_objects.append(
                {
                    "object_id": int(object_id),
                    "gt_vertex_count": count,
                    "share_of_class": float(count / max(gt_count, 1)),
                    "absorbed_by_other_majority": absorbed,
                    "object_gt_majority_class": str(majority.get("class_name", "")) if isinstance(majority, dict) else "",
                    "object_gt_majority_count": int(majority.get("count", 0)) if isinstance(majority, dict) else 0,
                    "object_exported_label": str(object_record.get("exported_label", "")),
                    "object_source_anchor_top_class": str(object_record.get("source_anchor_top_class", "")),
                }
            )

        covering_objects.sort(key=lambda item: item["gt_vertex_count"], reverse=True)
        metrics = per_class_metrics.get(class_id, {})
        class_records.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "gt_count": gt_count,
                "covered_gt_count": int(covered_count),
                "uncovered_or_unassigned_gt_count": int(max(0, gt_count - covered_count)),
                "matching_majority_gt_count": int(matching_majority_count),
                "absorbed_by_other_majority_gt_count": int(absorbed_by_other_majority_count),
                "absorbed_by_other_majority_fraction": float(
                    absorbed_by_other_majority_count / max(gt_count, 1)
                ),
                "iou": float(metrics.get("iou", 0.0)),
                "acc": float(metrics.get("acc", 0.0)),
                "top_covering_objects": covering_objects[:10],
            }
        )

    semantic_state_counts = {}
    export_state_counts = {}
    for record in object_records:
        semantic_state = str(record.get("semantic_state", "unlabeled"))
        semantic_state_counts[semantic_state] = int(semantic_state_counts.get(semantic_state, 0) + 1)
        export_state = str(record.get("export_state", "unlabeled"))
        export_state_counts[export_state] = int(export_state_counts.get(export_state, 0) + 1)

    return {
        "object_count": int(len(state.objects)),
        "provisional_object_count": int(len(state.provisional_objects)),
        "semantic_state_counts": semantic_state_counts,
        "export_state_counts": export_state_counts,
        "objects": object_records,
        "classes": class_records,
        "notes": [
            "anchor_semantics/source_anchor_classes are detector-derived votes from object observations.",
            "gt_majority/gt_top_classes are evaluation-only nearest-GT summaries, not runtime labels.",
            "absorbed_by_other_majority means GT vertices of this class were assigned to an object whose GT-majority class is different.",
        ],
    }


def _md_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def render_final_object_semantic_audit(audit: dict[str, Any]) -> str:
    class_records = sorted(
        audit.get("classes", []),
        key=lambda item: (float(item.get("absorbed_by_other_majority_fraction", 0.0)), int(item.get("gt_count", 0))),
        reverse=True,
    )
    object_records = sorted(
        audit.get("objects", []),
        key=lambda item: int(item.get("gt_total_votes", 0)),
        reverse=True,
    )

    lines = [
        "# Final Object Semantic Audit",
        "",
        f"- object_count: `{int(audit.get('object_count', 0))}`",
        f"- provisional_object_count: `{int(audit.get('provisional_object_count', 0))}`",
        f"- semantic_state_counts: `{json.dumps(audit.get('semantic_state_counts', {}) or {}, sort_keys=True)}`",
        f"- export_state_counts: `{json.dumps(audit.get('export_state_counts', {}) or {}, sort_keys=True)}`",
        "",
        "## Class Absorption",
        "| class | gt | acc | iou | absorbed_by_other_majority | top covering objects |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in class_records:
        top_objects = ", ".join(
            "{object_id}:{gt_vertex_count}->{majority} export={exported} source={source}".format(
                object_id=int(obj["object_id"]),
                gt_vertex_count=int(obj["gt_vertex_count"]),
                majority=obj.get("object_gt_majority_class") or "?",
                exported=obj.get("object_exported_label", ""),
                source=obj.get("object_source_anchor_top_class", ""),
            )
            for obj in item.get("top_covering_objects", [])[:5]
        )
        lines.append(
            "| {class_name} | {gt_count} | {acc:.4f} | {iou:.4f} | {absorbed} ({fraction:.2%}) | {top_objects} |".format(
                class_name=_md_cell(item.get("class_name", "")),
                gt_count=int(item.get("gt_count", 0)),
                acc=float(item.get("acc", 0.0)),
                iou=float(item.get("iou", 0.0)),
                absorbed=int(item.get("absorbed_by_other_majority_gt_count", 0)),
                fraction=float(item.get("absorbed_by_other_majority_fraction", 0.0)),
                top_objects=_md_cell(top_objects),
            )
        )

    lines.extend(
        [
            "",
            "## Final Objects",
            "| object | exported_label | source_anchor_top | gt_majority | gt_top_classes | points | observations |",
            "| ---: | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for item in object_records:
        gt_majority = item.get("gt_majority") or {}
        gt_majority_text = ""
        if isinstance(gt_majority, dict):
            gt_majority_text = f"{gt_majority.get('class_name', '')}:{int(gt_majority.get('count', 0))}"
        top_classes = ", ".join(
            f"{entry['class_name']}:{entry['count']}"
            for entry in item.get("gt_top_classes", [])[:5]
        )
        lines.append(
            "| {object_id} | {exported_label} | {source_anchor_top_class} | {gt_majority} | {top_classes} | {point_count} | {observation_count} |".format(
                object_id=int(item.get("object_id", -1)),
                exported_label=_md_cell(item.get("exported_label", "")),
                source_anchor_top_class=_md_cell(item.get("source_anchor_top_class", "")),
                gt_majority=_md_cell(gt_majority_text),
                top_classes=_md_cell(top_classes),
                point_count=int(item.get("point_count", 0)),
                observation_count=int(item.get("observation_count", 0)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _per_object_gt_vote_counters(evaluation: dict[str, Any]) -> dict[int, Counter[int]]:
    nearest_object_ids = np.asarray(evaluation.get("nearest_object_ids", []), dtype=np.int64)
    gt_labels = np.asarray(evaluation.get("gt_labels", []), dtype=np.int64)
    counters: dict[int, Counter[int]] = defaultdict(Counter)
    if len(nearest_object_ids) != len(gt_labels):
        return counters
    for object_id, class_id in zip(nearest_object_ids, gt_labels):
        object_id = int(object_id)
        class_id = int(class_id)
        if object_id < 0 or class_id < 0:
            continue
        counters[object_id][class_id] += 1
    return counters


def _top_gt_class_records(
    counter: Counter[int],
    class_names: dict[int, str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    total = int(sum(counter.values()))
    return [
        {
            "class_id": int(class_id),
            "class_name": class_names.get(int(class_id), f"class_{int(class_id)}"),
            "count": int(count),
            "fraction": float(count / max(total, 1)),
        }
        for class_id, count in counter.most_common(limit)
    ]


def _source_anchor_class_records(obj: Any) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for observation in getattr(obj, "observations", []):
        patch = getattr(observation, "patch", None)
        if patch is None:
            continue
        label = str(patch.metadata.get("anchor_class_name", "")).strip()
        if not label:
            continue
        confidence = float(patch.metadata.get("anchor_confidence", 0.0))
        frame_id = int(getattr(observation, "frame_id", getattr(patch, "source_frame_id", 0)))
        item = stats.setdefault(
            label,
            {
                "class_name": label,
                "observation_count": 0,
                "score_sum": 0.0,
                "max_confidence": 0.0,
                "frames": set(),
            },
        )
        item["observation_count"] += 1
        item["score_sum"] += confidence if confidence > 0.0 else 1.0
        item["max_confidence"] = max(float(item["max_confidence"]), confidence)
        item["frames"].add(frame_id)

    records = []
    for item in stats.values():
        frames = sorted(int(frame_id) for frame_id in item.pop("frames"))
        records.append(
            {
                "class_name": str(item["class_name"]),
                "observation_count": int(item["observation_count"]),
                "score_sum": float(item["score_sum"]),
                "max_confidence": float(item["max_confidence"]),
                "frame_count": int(len(frames)),
                "first_frame": int(frames[0]) if frames else -1,
                "last_frame": int(frames[-1]) if frames else -1,
            }
        )
    records.sort(
        key=lambda item: (
            int(item["frame_count"]),
            float(item["score_sum"]),
            float(item["max_confidence"]),
        ),
        reverse=True,
    )
    return records


def _json_anchor_semantics(anchor_semantics: Any) -> dict[str, Any]:
    if not isinstance(anchor_semantics, dict):
        return {}
    return {
        "canonical_label": str(anchor_semantics.get("canonical_label", "")),
        "canonical_score": float(anchor_semantics.get("canonical_score", 0.0)),
        "canonical_frame_hits": int(anchor_semantics.get("canonical_frame_hits", 0)),
        "canonical_confidence": float(anchor_semantics.get("canonical_confidence", 0.0)),
        "canonical_best_view_quality": float(anchor_semantics.get("canonical_best_view_quality", 0.0)),
        "source": str(anchor_semantics.get("source", "")),
        "label_score_sum": {
            str(label): float(score)
            for label, score in (anchor_semantics.get("label_score_sum", {}) or {}).items()
        },
        "label_weighted_score": {
            str(label): float(score)
            for label, score in (anchor_semantics.get("label_weighted_score", {}) or {}).items()
        },
        "contextual_label_score_sum": {
            str(label): float(score)
            for label, score in (anchor_semantics.get("contextual_label_score_sum", {}) or {}).items()
        },
        "label_recent_score": {
            str(label): float(score)
            for label, score in (anchor_semantics.get("label_recent_score", {}) or {}).items()
        },
        "label_max_confidence": {
            str(label): float(score)
            for label, score in (anchor_semantics.get("label_max_confidence", {}) or {}).items()
        },
        "label_best_view_quality": {
            str(label): float(score)
            for label, score in (anchor_semantics.get("label_best_view_quality", {}) or {}).items()
        },
        "label_frame_hits": {
            str(label): int(count)
            for label, count in (anchor_semantics.get("label_frame_hits", {}) or {}).items()
        },
        "label_high_quality_hits": {
            str(label): int(count)
            for label, count in (anchor_semantics.get("label_high_quality_hits", {}) or {}).items()
        },
        "label_seen_frames": {
            str(label): [int(frame_id) for frame_id in frame_ids]
            for label, frame_ids in (anchor_semantics.get("label_seen_frames", {}) or {}).items()
        },
        "relabel_events": list(anchor_semantics.get("relabel_events", []) or []),
    }


def write_run_report(
    scene_dir: Path,
    eval_dir: Path,
    experiment_name: str,
    dataset: ReplicaRoom0Dataset,
    args: argparse.Namespace,
    frame_limit: int,
    pipeline: Pipeline,
    state: SystemState,
    tsdf_records: np.ndarray,
    pool_semantic_records: np.ndarray,
    dense_surface_records: np.ndarray,
    dense_points: np.ndarray,
    labels: np.ndarray,
    evaluation: dict[str, Any],
    frame_metrics: list[dict[str, Any]],
    audit_dir: Path | None = None,
    final_object_semantic_audit: dict[str, Any] | None = None,
    output_profile: OutputProfile | None = None,
    structural_overlay_fusion: dict[str, int] | None = None,
) -> None:
    final_object_semantic_audit = final_object_semantic_audit or {}
    output_profile = output_profile or OutputProfile(
        name="full_debug",
        benchmark_audit_enabled=audit_dir is not None,
    )
    active_count = int(sum(obj.state == ObjectState.ACTIVE for obj in state.objects.values()))
    dormant_count = int(sum(obj.state == ObjectState.DORMANT for obj in state.objects.values()))
    inactive_count = int(sum(obj.state == ObjectState.INACTIVE for obj in state.objects.values()))
    surface_tier_counts = Counter(obj.surface_tier.value for obj in state.objects.values())
    resident_dense_points = int(
        sum(len(entry.points) for entry in state.dense_surface_map.entries.values() if entry.resident)
    )
    final_local_memory_point_count_total = int(sum(len(obj.local_pcd) for obj in state.objects.values()))
    projected_dense_labeled_point_count = int(np.count_nonzero(labels >= 0))
    per_class_sorted = sorted(evaluation["per_class"], key=lambda item: item["iou"], reverse=True)
    top_lines = [f"- `{item['class_name']}`: IoU `{item['iou']:.4f}`, Acc `{item['acc']:.4f}`" for item in per_class_sorted[:25]]
    current_frame_visibility_rejected_patch_total = int(
        sum(
            int(metrics.get("current_frame_visibility_gate", {}).get("rejected_patch_count", 0))
            for metrics in frame_metrics
        )
    )
    current_frame_visibility_depth_rejected_point_total = int(
        sum(
            int(metrics.get("current_frame_visibility_gate", {}).get("depth_rejected_point_count", 0))
            for metrics in frame_metrics
        )
    )
    frontend_stage_totals = build_frontend_stage_report_payload(frame_metrics)
    scheduling_totals = build_scheduling_report_payload(frame_metrics)
    async_refinement_totals = build_async_refinement_summary(frame_metrics)
    stage_timing_summary = build_stage_timing_summary(frame_metrics)
    association_pruning_summary = build_association_pruning_summary(frame_metrics)
    structural_overlay_fusion = structural_overlay_fusion or {}
    scene_name = normalize_scene_name(getattr(args, "scene_name", None), default=scene_dir.name or "room0")
    structural_overlay_frame_totals = Counter()
    for metrics in frame_metrics:
        overlay_metrics = dict(metrics.get("structural_overlay", {}) or {})
        for key in (
            "structure_anchor_count",
            "sam_proposal_count",
            "accepted_pair_count",
            "mismatched_mask_count",
            "voted_pixel_count",
            "new_voxel_count",
        ):
            structural_overlay_frame_totals[key] += int(overlay_metrics.get(key, 0))

    export_lines = [
        f"- `room0_instance_map.ply` (pool-based semantic object map): `{scene_dir / 'exports' / 'room0_instance_map.ply'}`",
        f"- `room0_instance_map_dense_surface.ply` (resident dense surface export): `{scene_dir / 'exports' / 'room0_instance_map_dense_surface.ply'}`",
        f"- `room0_instance_map_tsdf_backbone.ply` (coarse TSDF owner/support/stability backbone): `{scene_dir / 'exports' / 'room0_instance_map_tsdf_backbone.ply'}`",
        f"- `room0_dense_geometry_fused_rgb.ply` (dense RGB-D fused geometry): `{scene_dir / 'exports' / 'room0_dense_geometry_fused_rgb.ply'}`",
        f"- `room0_dense_geometry_instance_projected.ply` (projected from the pool-based instance map): `{scene_dir / 'exports' / 'room0_dense_geometry_instance_projected.ply'}`",
        f"- `room0_structural_overlay.ply` (synthetic negative-id structure overlay): `{scene_dir / 'exports' / 'room0_structural_overlay.ply'}`",
    ]
    if output_profile.write_debug_ply:
        export_lines.append(
            f"- `room0_instance_map_local_memory.ply` (compat/debug pool view): `{scene_dir / 'exports' / 'room0_instance_map_local_memory.ply'}`"
        )
    if output_profile.write_eval_mesh_ply:
        export_lines.extend(
            [
                f"- `room0_gtmesh_pred_semantic.ply`: `{eval_dir / 'room0_gtmesh_pred_semantic.ply'}`",
                f"- `room0_gtmesh_gt_semantic.ply`: `{eval_dir / 'room0_gtmesh_gt_semantic.ply'}`",
                f"- `room0_gtmesh_semantic_correctness.ply`: `{eval_dir / 'room0_gtmesh_semantic_correctness.ply'}`",
            ]
        )
    export_lines.extend(
        [
            f"- `final_object_semantic_audit.json`: `{scene_dir / 'final_object_semantic_audit.json'}`",
            f"- `final_object_semantic_audit.md`: `{scene_dir / 'final_object_semantic_audit.md'}`",
        ]
    )
    if audit_dir is not None:
        export_lines.extend(
            [
                f"- `local_memory_audit.jsonl`: `{audit_dir / 'local_memory_audit.jsonl'}`",
                f"- `local_memory_audit.log`: `{audit_dir / 'local_memory_audit.log'}`",
                f"- `local_memory_audit_summary.json`: `{audit_dir / 'local_memory_audit_summary.json'}`",
                f"- `local_memory_audit_vis/`: `{audit_dir / 'vis'}`",
            ]
        )

    lines = [
        "# OVO Run Report",
        "",
        f"- Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Run dir: `{scene_dir}`",
        "- Dataset: `replica`",
        f"- Scene: `{scene_name}`",
        f"- Eval dir: `{eval_dir}`",
        "",
        "## Config",
        f"- `experiment_name`: `{experiment_name}`",
        "- `slam_module`: `oviovo_pipeline`",
        f"- `device`: `{args.proposal_device}`",
        f"- `proposal_backend`: `{pipeline.proposal.active_backend_name}`",
        f"- `sam_version`: `{args.sam_version}`",
        f"- `sam_encoder`: `{args.sam_encoder}`",
        f"- `points_per_side`: `{args.points_per_side}`",
        f"- `max_proposals`: `{args.max_proposals}`",
        f"- `frame_count`: `{frame_limit}`",
        f"- `output_profile`: `{output_profile.name}`",
        "",
        "## Metrics",
        f"- `mIoU`: {evaluation['miou']:.4f}",
        f"- `mAcc`: {evaluation['macc']:.4f}",
        f"- `f-mIoU`: {evaluation['fmiou']:.4f}",
        f"- `f-mAcc`: {evaluation['fmacc']:.4f}",
        "",
        "## Evaluation Contract",
        "- `room0_instance_map.ply`: pool-based semantic object map built from `ObjectMap.local_pcd`.",
        "- `room0_instance_map_dense_surface.ply`: resident dense surface export.",
        "- `room0_instance_map_tsdf_backbone.ply`: coarse TSDF owner/support/stability backbone.",
        "- `room0_dense_geometry_instance_projected.ply`: dense geometry projected from the pool-based instance map; it may include conservative structural overlay fusion with synthetic negative structural overlay ids.",
        "- `evaluation source`: `dense_geometry_instance_projected` after optional conservative structural overlay fusion.",
        "- `derived_surface`: dense RGB-D fused geometry sampled into a voxelized point surface.",
        "- `semantic assignment`: object ids use majority GT class over nearest GT mesh vertices they cover; synthetic negative structural overlay ids, when present, use direct semantic labels.",
        "- `mIoU` / `mAcc`: computed on GT mesh vertices after nearest-neighbor transfer from the predicted dense geometry projection.",
        "",
        "## Evaluation Metadata",
        f"- `final_object_count`: `{len(state.objects)}`",
        f"- `final_provisional_object_count`: `{len(state.provisional_objects)}`",
        f"- `active_object_count`: `{active_count}`",
        f"- `dormant_object_count`: `{dormant_count}`",
        f"- `inactive_object_count`: `{inactive_count}`",
        f"- `surface_tier_active_count`: `{int(surface_tier_counts.get('active', 0))}`",
        f"- `surface_tier_warm_count`: `{int(surface_tier_counts.get('warm', 0))}`",
        f"- `surface_tier_cold_count`: `{int(surface_tier_counts.get('cold', 0))}`",
        f"- `cold_object_count`: `{int(surface_tier_counts.get('cold', 0))}`",
        f"- `coarse_point_count`: `{len(tsdf_records)}`",
        f"- `pool_point_count`: `{len(pool_semantic_records)}`",
        f"- `dense_surface_point_count`: `{len(dense_surface_records)}`",
        f"- `dense_surface_resident_point_count`: `{resident_dense_points}`",
        f"- `dense_geometry_point_count`: `{len(dense_points)}`",
        f"- `projected_dense_labeled_point_count`: `{projected_dense_labeled_point_count}`",
        f"- `projected_dense_object_count`: `{len(set(labels[labels >= 0].tolist())) if np.any(labels >= 0) else 0}`",
        f"- `structural_overlay_voxel_count`: `{len(state.structural_overlay_map.voxels)}`",
        f"- `structural_overlay_accepted_pair_count_total`: `{int(structural_overlay_frame_totals['accepted_pair_count'])}`",
        f"- `structural_overlay_voted_pixel_count_total`: `{int(structural_overlay_frame_totals['voted_pixel_count'])}`",
        f"- `structural_overlay_replaced_point_count`: `{int(structural_overlay_fusion.get('overlay_replaced_point_count', 0))}`",
        f"- `structural_overlay_protected_point_count`: `{int(structural_overlay_fusion.get('overlay_protected_point_count', 0))}`",
        f"- `final_local_memory_point_count_total`: `{final_local_memory_point_count_total}` (v1 pool geometry total from `local_pcd`)",
        f"- `mean_raw_proposal_count`: `{float(np.mean([item['raw_proposal_count'] for item in frame_metrics])):.4f}`",
        f"- `mean_matched_patch_count`: `{float(np.mean([item['matched_patch_count'] for item in frame_metrics])):.4f}`",
        f"- `mean_ambiguous_patch_count`: `{float(np.mean([item['ambiguous_patch_count'] for item in frame_metrics])):.4f}`",
        f"- `mean_ambiguous_matched_count`: `{float(np.mean([item['ambiguous_matched_count'] for item in frame_metrics])):.4f}`",
        f"- `mean_ambiguous_new_patch_count`: `{float(np.mean([item['ambiguous_new_patch_count'] for item in frame_metrics])):.4f}`",
        f"- `max_local_memory_point_count_total`: `{int(max(item['local_memory_point_count_total'] for item in frame_metrics))}`",
        f"- `current_frame_visibility_rejected_patch_total`: `{current_frame_visibility_rejected_patch_total}`",
        f"- `current_frame_visibility_depth_rejected_point_total`: `{current_frame_visibility_depth_rejected_point_total}`",
        f"- `prefetch_enabled`: `{scheduling_totals['prefetch_enabled']}`",
        f"- `prefetch_submitted_count`: `{scheduling_totals['prefetch_submitted_count']}`",
        f"- `prefetch_hit_count`: `{scheduling_totals['prefetch_hit_count']}`",
        f"- `prefetch_miss_count`: `{scheduling_totals['prefetch_miss_count']}`",
        f"- `prefetch_wait_sec_total`: `{scheduling_totals['prefetch_wait_sec_total']:.4f}`",
        f"- `frontend_prefetch_frame_count`: `{scheduling_totals['frontend_prefetch_frame_count']}`",
        f"- `frontend_inline_frame_count`: `{scheduling_totals['frontend_inline_frame_count']}`",
        f"- `frontend_exception_count`: `{scheduling_totals['frontend_exception_count']}`",
        f"- `frontend_last_exception`: `{scheduling_totals['frontend_last_exception']}`",
        f"- `association_score_parallel_used_count`: `{scheduling_totals['association_score_parallel_used_count']}`",
        f"- `association_score_parallel_candidate_count_total`: `{scheduling_totals['association_score_parallel_candidate_count_total']}`",
        f"- `depth_refinement_parallel_used_count`: `{scheduling_totals['depth_refinement_parallel_used_count']}`",
        f"- `depth_refinement_parallel_worker_count_max`: `{scheduling_totals['depth_refinement_parallel_worker_count_max']}`",
        f"- `source_sam_proposal_count_total`: `{frontend_stage_totals['source_sam_proposal_count_total']}`",
        f"- `anchor_voted_proposal_count_total`: `{frontend_stage_totals['anchor_voted_proposal_count_total']}`",
        f"- `runtime_merged_proposal_count_total`: `{frontend_stage_totals['runtime_merged_proposal_count_total']}`",
        f"- `runtime_merge_count_total`: `{frontend_stage_totals['runtime_merge_count_total']}`",
        f"- `anchor_voted_unanchored_count_total`: `{frontend_stage_totals['anchor_voted_unanchored_count_total']}`",
        f"- `runtime_anchor_label_missing_edge_count_total`: `{frontend_stage_totals['runtime_anchor_label_missing_edge_count_total']}`",
        f"- `runtime_anchor_label_mismatch_edge_count_total`: `{frontend_stage_totals['runtime_anchor_label_mismatch_edge_count_total']}`",
        f"- `runtime_anchor_identity_mismatch_edge_count_total`: `{frontend_stage_totals['runtime_anchor_identity_mismatch_edge_count_total']}`",
        f"- `async_refinement_sam_proposal_count_total`: `{async_refinement_totals['async_refinement_sam_proposal_count_total']}`",
        f"- `async_refinement_fine_proposal_count_total`: `{async_refinement_totals['async_refinement_fine_proposal_count_total']}`",
        f"- `async_refinement_fine_patch_count_total`: `{async_refinement_totals['async_refinement_fine_patch_count_total']}`",
        f"- `async_refinement_replaced_observation_count_total`: `{async_refinement_totals['async_refinement_replaced_observation_count_total']}`",
        f"- `async_refinement_inserted_observation_count_total`: `{async_refinement_totals['async_refinement_inserted_observation_count_total']}`",
        f"- `async_refinement_updated_object_count_total`: `{async_refinement_totals['async_refinement_updated_object_count_total']}`",
        "",
        "## Per-class",
        *(top_lines or ["- No per-class metrics available"]),
    ]
    if stage_timing_summary:
        lines.extend(["", "## Stage Timing Summary"])
        for stage, values in stage_timing_summary.items():
            lines.append(
                f"- `{stage}`: mean `{values['mean_sec']:.4f}s`, "
                f"median `{values['median_sec']:.4f}s`, "
                f"p95 `{values['p95_sec']:.4f}s`, "
                f"max `{values['max_sec']:.4f}s` at frame `{values['max_frame_id']}`, "
                f"total `{values['total_sec']:.4f}s`"
            )
    if association_pruning_summary:
        lines.append("")
        lines.append("## Association Pruning Summary")
        for key, value in association_pruning_summary.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Exports",
            *export_lines,
            "",
        ]
    )
    report = "\n".join(lines)
    report_path = scene_dir / "run_report.md"
    report_path.write_text(report, encoding="utf-8")

    sidecar = {
        "output_profile": output_profile.name,
        "output_profile_artifacts": {
            "benchmark_audit_enabled": bool(output_profile.benchmark_audit_enabled),
            "write_primary_exports": bool(output_profile.write_primary_exports),
            "write_debug_ply": bool(output_profile.write_debug_ply),
            "write_eval_mesh_ply": bool(output_profile.write_eval_mesh_ply),
            "write_preview_render": bool(output_profile.write_preview_render),
            "write_final_audit": bool(output_profile.write_final_audit),
            "write_reports": bool(output_profile.write_reports),
        },
        "miou": evaluation["miou"],
        "macc": evaluation["macc"],
        "fmiou": evaluation["fmiou"],
        "fmacc": evaluation["fmacc"],
        "per_class": evaluation["per_class"],
        "final_object_semantic_audit_path": str(scene_dir / "final_object_semantic_audit.json"),
        "final_object_semantic_audit_summary": {
            "object_count": int(final_object_semantic_audit.get("object_count", 0)),
            "provisional_object_count": int(final_object_semantic_audit.get("provisional_object_count", 0)),
            "semantic_state_counts": {
                str(label): int(count)
                for label, count in (final_object_semantic_audit.get("semantic_state_counts", {}) or {}).items()
            },
            "high_absorption_classes": [
                {
                    "class_name": str(item.get("class_name", "")),
                    "gt_count": int(item.get("gt_count", 0)),
                    "absorbed_by_other_majority_gt_count": int(
                        item.get("absorbed_by_other_majority_gt_count", 0)
                    ),
                    "absorbed_by_other_majority_fraction": float(
                        item.get("absorbed_by_other_majority_fraction", 0.0)
                    ),
                }
                for item in sorted(
                    final_object_semantic_audit.get("classes", []),
                    key=lambda value: (
                        float(value.get("absorbed_by_other_majority_fraction", 0.0)),
                        int(value.get("gt_count", 0)),
                    ),
                    reverse=True,
                )[:10]
            ],
        },
        "frame_count": frame_limit,
        "final_object_count": len(state.objects),
        "final_local_memory_point_count_total": final_local_memory_point_count_total,
        "surface_tier_counts": {
            "active": int(surface_tier_counts.get("active", 0)),
            "warm": int(surface_tier_counts.get("warm", 0)),
            "cold": int(surface_tier_counts.get("cold", 0)),
        },
        "coarse_point_count": int(len(tsdf_records)),
        "pool_point_count": int(len(pool_semantic_records)),
        "dense_surface_point_count": int(len(dense_surface_records)),
        "resident_dense_surface_point_count": resident_dense_points,
        "projected_dense_labeled_point_count": projected_dense_labeled_point_count,
        "structural_overlay_voxel_count": int(len(state.structural_overlay_map.voxels)),
        "structural_overlay_frame_totals": {
            str(key): int(value) for key, value in structural_overlay_frame_totals.items()
        },
        "structural_overlay_fusion": {
            str(key): int(value) for key, value in structural_overlay_fusion.items()
        },
        "mean_ambiguous_patch_count": float(np.mean([item["ambiguous_patch_count"] for item in frame_metrics])),
        "mean_ambiguous_matched_count": float(np.mean([item["ambiguous_matched_count"] for item in frame_metrics])),
        "mean_ambiguous_new_patch_count": float(np.mean([item["ambiguous_new_patch_count"] for item in frame_metrics])),
        "max_local_memory_point_count_total": int(max(item["local_memory_point_count_total"] for item in frame_metrics)),
        "current_frame_visibility_rejected_patch_total": current_frame_visibility_rejected_patch_total,
        "current_frame_visibility_depth_rejected_point_total": current_frame_visibility_depth_rejected_point_total,
        "stage_timing_summary": stage_timing_summary,
        "association_pruning_summary": association_pruning_summary,
        **frontend_stage_totals,
        **scheduling_totals,
        **async_refinement_totals,
    }
    (scene_dir / "run_report.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def read_gt_vertices(path: Path) -> np.ndarray:
    header, handle = read_ply_header(path)
    vertex_count, vertex_dtype = parse_vertex_layout(header)
    with handle:
        vertex_array = np.fromfile(handle, dtype=vertex_dtype, count=vertex_count)
    return np.stack([vertex_array["x"], vertex_array["y"], vertex_array["z"]], axis=1).astype(np.float32)


def read_gt_labels(path: Path) -> np.ndarray:
    labels = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return np.asarray(labels, dtype=np.int32)


def read_class_names(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["id"]): item["name"] for item in payload.get("classes", [])}


def read_ply_header(path: Path) -> tuple[list[str], Any]:
    handle = path.open("rb")
    header = []
    while True:
        line = handle.readline()
        if not line:
            raise RuntimeError(f"Unexpected EOF in PLY header: {path}")
        text = line.decode("ascii", errors="ignore").strip()
        header.append(text)
        if text == "end_header":
            break
    return header, handle


def parse_vertex_layout(header: list[str]) -> tuple[int, np.dtype]:
    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "ushort": "<u2",
        "uint16": "<u2",
    }
    vertex_count = 0
    properties = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and parts[0] == "property" and len(parts) == 3:
            properties.append((parts[2], type_map[parts[1]]))
        elif in_vertex and parts[0] == "element":
            break
    return vertex_count, np.dtype(properties)


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


def write_vertex_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    state_ids: np.ndarray,
    supports: np.ndarray,
) -> None:
    vertex_array = np.zeros(len(points), dtype=PLY_DTYPE)
    vertex_array["x"] = points[:, 0]
    vertex_array["y"] = points[:, 1]
    vertex_array["z"] = points[:, 2]
    vertex_array["red"] = colors[:, 0]
    vertex_array["green"] = colors[:, 1]
    vertex_array["blue"] = colors[:, 2]
    vertex_array["object_id"] = object_ids
    vertex_array["state_id"] = state_ids
    vertex_array["support"] = supports
    write_binary_ply(path, vertex_array)


def render_projection(points: np.ndarray, colors: np.ndarray, plane: str, output_path: Path, canvas_size: tuple[int, int] = (1400, 900)) -> None:
    if plane == "xz":
        coords = points[:, [0, 2]]
        depth_axis = points[:, 1]
    elif plane == "xy":
        coords = points[:, [0, 1]]
        depth_axis = points[:, 2]
    else:
        raise ValueError(plane)

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
        canvas[max(vv - 1, 0):min(vv + 2, height), max(uu - 1, 0):min(uu + 2, width)] = color
    Image.fromarray(canvas).save(output_path)


def render_preview_sheet(image_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        width, height = images[0].size
        sheet = Image.new("RGB", (2 * width, 2 * height), color=(255, 255, 255))
        for image, pos in zip(images, [(0, 0), (width, 0), (0, height), (width, height)]):
            sheet.paste(image, pos)
        sheet.save(output_path)
    finally:
        for image in images:
            image.close()


def summarize_state_counts(state: SystemState) -> dict[str, int]:
    counts = Counter(obj.state.value for obj in state.objects.values())
    return {key: int(value) for key, value in sorted(counts.items())}


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
        (106, 153, 78),
        (168, 218, 220),
        (233, 196, 106),
        (38, 70, 83),
    ]
    return palette[object_id % len(palette)]


def semantic_surface_color(semantic_label: str, object_id: int) -> tuple[int, int, int]:
    if not semantic_label:
        return instance_color(object_id)
    label = semantic_label.strip().lower()
    digest = hashlib.md5(label.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], byteorder="little") / 65535.0
    return hsv_to_rgb_uint8(hue, 0.88, 1.0)


def hsv_to_rgb_uint8(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    hue = float(hue) % 1.0
    saturation = min(max(float(saturation), 0.0), 1.0)
    value = min(max(float(value), 0.0), 1.0)
    sector = hue * 6.0
    index = int(math.floor(sector))
    frac = sector - index
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * frac)
    t = value * (1.0 - saturation * (1.0 - frac))
    if index == 0:
        rgb = (value, t, p)
    elif index == 1:
        rgb = (q, value, p)
    elif index == 2:
        rgb = (p, value, t)
    elif index == 3:
        rgb = (p, q, value)
    elif index == 4:
        rgb = (t, p, value)
    else:
        rgb = (value, p, q)
    return tuple(int(round(channel * 255.0)) for channel in rgb)


def colors_for_object_labels(
    object_ids: np.ndarray,
    state: SystemState,
    default_color: tuple[int, int, int] = (180, 180, 180),
    direct_object_label_names: dict[int, str] | None = None,
) -> np.ndarray:
    object_ids = np.asarray(object_ids, dtype=np.int32)
    colors = np.empty((len(object_ids), 3), dtype=np.uint8)
    colors[:] = np.asarray(default_color, dtype=np.uint8)
    label_color_cache: dict[str, tuple[int, int, int]] = {}
    direct_object_label_names = direct_object_label_names or {}
    for object_id, semantic_label in direct_object_label_names.items():
        color = semantic_surface_color(str(semantic_label), int(object_id))
        colors[object_ids == int(object_id)] = np.asarray(color, dtype=np.uint8)
    for object_id in np.unique(object_ids[object_ids >= 0]).tolist():
        obj = state.objects.get(int(object_id))
        if obj is None:
            continue
        semantic_label = object_semantic_label(obj)
        cache_key = semantic_label.strip().lower() if semantic_label else f"object:{int(object_id)}"
        color = label_color_cache.setdefault(
            cache_key,
            semantic_surface_color(semantic_label, int(object_id)),
        )
        colors[object_ids == int(object_id)] = np.asarray(color, dtype=np.uint8)
    return colors


@lru_cache(maxsize=2048)
def class_color(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(class_id) * 2654435761 % (2**32))
    color = (rng.random(3) * 205 + 40).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def labels_to_class_colors(
    labels: np.ndarray,
    default_color: tuple[int, int, int] = (80, 80, 80),
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    colors = np.empty((len(labels), 3), dtype=np.uint8)
    if len(labels) == 0:
        return colors

    colors[:] = np.asarray(default_color, dtype=np.uint8)
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return colors

    valid_labels = labels[valid_mask]
    unique_labels = np.unique(valid_labels)
    for class_id in unique_labels.tolist():
        semantic_label = class_names.get(int(class_id), "") if class_names else ""
        if semantic_label:
            color = np.asarray(semantic_surface_color(semantic_label, int(class_id)), dtype=np.uint8)
        else:
            color = np.asarray(class_color(int(class_id)), dtype=np.uint8)
        colors[labels == int(class_id)] = color
    return colors


if __name__ == "__main__":
    main()
