#!/usr/bin/env python3
"""Checkpointed room0 mapping and export runner."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from run_room0_full_eval import (  # noqa: E402
    Frame,
    FrameProposalPrefetcher,
    Pipeline,
    ReplicaRoom0Dataset,
    accumulate_geometry,
    build_benchmark_contract_payload,
    audit_checkpoint_payload_sizes,
    build_local_memory_frame_audit,
    build_output_profile,
    build_prefetch_pipeline,
    build_proposal_module,
    build_room0_checkpoint_payload,
    build_room0_run_layout,
    ensure_room0_run_directories,
    export_room0_outputs,
    load_room0_checkpoint,
    normalize_scene_name,
    output_profile_from_checkpoint,
    apply_runtime_profile_to_pipeline,
    pipeline_from_room0_checkpoint,
    save_local_memory_audit_overlay,
    save_room0_checkpoint,
    summarize_state_counts,
    args_from_room0_checkpoint,
)
from src.utils.visualization import (  # noqa: E402
    add_panel_title,
    anchor_vote_overlay_image,
    runtime_group_overlay_image,
    runtime_proposal_class_overlay_image,
    save_multi_panel,
)


DEFAULT_DATASET_ROOT = Path("/home/ww/vv/dataset/Replica/room0")
DEFAULT_GT_LABELS = REPO_ROOT / "data" / "input" / "replica_semantic_gt" / "room0.txt"
DEFAULT_GT_MESH_PLY = Path("/home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply")
DEFAULT_GT_INFO_JSON = Path("/home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json")
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "room0_surface_gate_fast_high_iou_4090.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "export", "build-export"), default="build-export")
    parser.add_argument("--scene-name", type=str, default="room0")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--gt-labels", type=Path, default=DEFAULT_GT_LABELS)
    parser.add_argument("--gt-mesh-ply", type=Path, default=DEFAULT_GT_MESH_PLY)
    parser.add_argument("--gt-info-json", type=Path, default=DEFAULT_GT_INFO_JSON)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "tmp_validation")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--proposal-backend", type=str, default="precomputed")
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
    parser.add_argument("--lightweight-benchmark", action="store_true")
    parser.add_argument("--fast-eval", action="store_true")
    parser.add_argument("--full-debug", action="store_true")
    parser.add_argument(
        "--benchmark-profile",
        choices=("fast", "debug_overlay"),
        default=None,
        help="Benchmark metadata profile. Existing --fast-eval/--full-debug behavior is preserved.",
    )
    parser.add_argument("--skip-debug-ply", action="store_true")
    parser.add_argument("--skip-eval-ply", action="store_true")
    parser.add_argument("--skip-preview-render", action="store_true")
    parser.add_argument("--pipeline-verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-frame progress prints.")
    parser.add_argument(
        "--runtime-vis-debug",
        action="store_true",
        help="Write per-frame RuntimeVis JSONL and visual overlay panels.",
    )
    parser.add_argument(
        "--runtime-vis-debug-every",
        type=int,
        default=10,
        help="Save RuntimeVis visual overlay every N processed frames when --runtime-vis-debug is set.",
    )
    parser.add_argument(
        "--runtime-vis-debug-max-decisions",
        type=int,
        default=80,
        help="Maximum merge-decision rows to keep per frame in RuntimeVis debug JSONL.",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _update_report_sidecar(path: Path, payload: dict[str, Any]) -> None:
    if not path.exists():
        return
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict):
        return
    sidecar.update(payload)
    _write_json(path, sidecar)


def _runtime_vis_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def _runtime_vis_payload(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _runtime_vis_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_runtime_vis_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return _runtime_vis_scalar(value)


def _proposal_debug_record(proposal: Any, *, profile: Any | None = None, group_id: int | None = None) -> dict[str, Any]:
    metadata = dict(getattr(proposal, "metadata", {}) or {})
    bbox = np.asarray(getattr(proposal, "bbox_xyxy", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(4)
    mask = np.asarray(getattr(proposal, "mask", np.zeros((0, 0), dtype=bool))).astype(bool)
    image_area = max(1, int(mask.size))
    record: dict[str, Any] = {
        "proposal_id": int(getattr(proposal, "proposal_id", -1)),
        "group_id": None if group_id is None else int(group_id),
        "backend_name": str(getattr(proposal, "backend_name", "")),
        "area": int(getattr(proposal, "area", int(mask.sum()))),
        "area_ratio": float(int(getattr(proposal, "area", int(mask.sum()))) / image_area),
        "bbox_xyxy": [float(v) for v in bbox.tolist()],
        "confidence": float(getattr(proposal, "confidence", 0.0)),
        "class_name": str(metadata.get("anchor_class_name", metadata.get("sedpp_class_name", "")) or ""),
        "semantic_commit_allowed": bool(metadata.get("semantic_commit_allowed", False)),
        "anchor_label_strength": str(metadata.get("anchor_label_strength", "") or ""),
        "source": str(metadata.get("source", "") or ""),
    }
    if profile is not None:
        profile_payload = _runtime_vis_payload(profile)
        if isinstance(profile_payload, dict):
            for key in (
                "proposal_class",
                "objectness_score",
                "backgroundness_score",
                "depth_valid_ratio",
                "depth_variance",
                "planar_fit_residual",
                "border_touch_ratio",
                "depth_edge_density",
                "local_closedness",
                "mask_extent_ratio",
                "best_prior_object_id",
                "best_prior_score",
            ):
                if key in profile_payload:
                    record[key] = profile_payload[key]
    return _runtime_vis_payload(record)


def _group_debug_record(group: Any) -> dict[str, Any]:
    bbox = np.asarray(getattr(group, "merged_bbox_xyxy", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(4)
    mask = np.asarray(getattr(group, "merged_mask", np.zeros((0, 0), dtype=bool))).astype(bool)
    image_area = max(1, int(mask.size))
    return _runtime_vis_payload(
        {
            "group_id": int(getattr(group, "group_id", -1)),
            "member_mask_ids": [int(item) for item in getattr(group, "member_mask_ids", [])],
            "member_count": len(getattr(group, "member_mask_ids", []) or []),
            "area": int(getattr(group, "area", int(mask.sum()))),
            "area_ratio": float(int(getattr(group, "area", int(mask.sum()))) / image_area),
            "bbox_xyxy": [float(v) for v in bbox.tolist()],
            "confidence": float(getattr(group, "confidence", 0.0)),
            "linked_object_id": getattr(group, "linked_object_id", None),
            "whole_prior_used": bool(getattr(group, "whole_prior_used", False)),
            "merge_reason": str(getattr(group, "merge_reason", "") or ""),
            "merge_reason_summary": dict(getattr(group, "merge_reason_summary", {}) or {}),
        }
    )


def _decision_debug_record(decision: Any) -> dict[str, Any]:
    payload = _runtime_vis_payload(decision)
    if not isinstance(payload, dict):
        return {}
    keep_keys = (
        "mask_id_a",
        "mask_id_b",
        "adjacency_score",
        "depth_continuity_score",
        "plane_similarity_score",
        "bbox_plausibility_score",
        "containment_score",
        "median_depth_gap",
        "boundary_depth_continuity",
        "plane_compatibility",
        "merged_bbox_compactness",
        "base_score",
        "whole_prior_score",
        "background_conflict_penalty",
        "final_score",
        "accepted",
        "accepted_reason",
        "boosted_by_whole_prior",
        "linked_object_id",
        "rejected_due_to_background_conflict",
    )
    return {key: payload.get(key) for key in keep_keys if key in payload}


def build_runtime_vis_debug_record(
    *,
    frame_id: int,
    processed_frame_index: int,
    raw_proposals: list[Any],
    runtime_output: Any,
    stage_timings: dict[str, Any],
    max_decisions: int,
) -> dict[str, Any]:
    profile_map = {
        int(getattr(profile, "proposal_id", -1)): profile
        for profile in list(getattr(runtime_output, "proposal_profiles", []) or [])
    }
    raw_to_group = {
        int(key): int(value)
        for key, value in dict(getattr(runtime_output, "raw_to_group", {}) or {}).items()
    }
    decisions = list(getattr(runtime_output, "merge_decisions", []) or [])
    decisions_sorted = sorted(
        decisions,
        key=lambda decision: (
            bool(getattr(decision, "accepted", False)),
            float(getattr(decision, "final_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return {
        "frame_id": int(frame_id),
        "processed_frame_index": int(processed_frame_index),
        "runtime_vis_sec": float(stage_timings.get("runtime_vis", 0.0) or 0.0),
        "raw_proposal_count": int(len(raw_proposals)),
        "merged_group_count": int(len(getattr(runtime_output, "groups", []) or [])),
        "raw_to_group": raw_to_group,
        "group_stats": _runtime_vis_payload(dict(getattr(runtime_output, "group_stats", {}) or {})),
        "raw_proposals": [
            _proposal_debug_record(
                proposal,
                profile=profile_map.get(int(getattr(proposal, "proposal_id", -1))),
                group_id=raw_to_group.get(int(getattr(proposal, "proposal_id", -1))),
            )
            for proposal in raw_proposals
        ],
        "groups": [_group_debug_record(group) for group in list(getattr(runtime_output, "groups", []) or [])],
        "merge_decisions": [_decision_debug_record(decision) for decision in decisions_sorted[: max(0, int(max_decisions))]],
        "merge_decision_count": int(len(decisions)),
    }


def save_runtime_vis_debug_overlay(
    *,
    frame: Frame,
    raw_proposals: list[Any],
    runtime_output: Any,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(frame.rgb, dtype=np.uint8)
    source_panel = add_panel_title(
        anchor_vote_overlay_image(rgb, raw_proposals),
        "Raw SED++ proposals: class/confidence",
    )
    class_panel = add_panel_title(
        runtime_proposal_class_overlay_image(
            rgb,
            raw_proposals,
            list(getattr(runtime_output, "proposal_profiles", []) or []),
        ),
        "RuntimeVis classes: green object, red background, yellow uncertain",
    )
    group_panel = add_panel_title(
        runtime_group_overlay_image(rgb, list(getattr(runtime_output, "groups", []) or [])),
        "RuntimeVis merged groups",
    )
    save_multi_panel([source_panel, class_panel, group_panel], output_path)


def _experiment_name(args: argparse.Namespace) -> str:
    if args.experiment_name:
        return str(args.experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    scene_name = normalize_scene_name(getattr(args, "scene_name", "room0"))
    return f"{timestamp}_{scene_name}_checkpointed_s{int(args.frame_stride)}_{int(args.num_frames)}f"


def _default_checkpoint_path(layout) -> Path:
    return layout.scene_dir / "mapping_state.pkl"


def _build_prefetch_frame(dataset_frame: Frame, processed_frame_id: int) -> Frame:
    return Frame(
        frame_id=int(processed_frame_id),
        rgb=dataset_frame.rgb,
        depth=dataset_frame.depth,
        pose=dataset_frame.pose,
        intrinsics=dataset_frame.intrinsics,
        timestamp=float(dataset_frame.timestamp),
        source_frame_id=int(dataset_frame.frame_id),
    )


def _prefetch_stats(prefetcher: FrameProposalPrefetcher | None) -> dict[str, Any]:
    return dict(prefetcher.snapshot_stats()) if prefetcher is not None else {}


def build_checkpoint(args: argparse.Namespace, *, experiment_name: str, checkpoint_path: Path) -> dict[str, Any]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    scene_name = normalize_scene_name(getattr(args, "scene_name", "room0"))
    args.scene_name = scene_name
    layout = build_room0_run_layout(args.output_root, experiment_name, scene_name=scene_name)
    output_profile = build_output_profile(args)
    ensure_room0_run_directories(layout, output_profile)

    dataset = ReplicaRoom0Dataset(args.dataset_root)
    frame_stride = max(1, int(args.frame_stride))
    all_frame_indices = list(range(0, len(dataset), frame_stride))
    selected_dataset_indices = (
        all_frame_indices
        if int(args.num_frames) <= 0
        else all_frame_indices[: min(int(args.num_frames), len(all_frame_indices))]
    )
    frame_limit = len(selected_dataset_indices)

    init_start = time.perf_counter()
    pipeline = Pipeline(config_path=str(args.config_path))
    pipeline.verbose = bool(args.pipeline_verbose)
    apply_runtime_profile_to_pipeline(pipeline, output_profile)
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
    init_sec = time.perf_counter() - init_start

    frame_metrics_path = layout.scene_dir / "frame_metrics.jsonl"
    audit_jsonl_path = layout.audit_dir / "local_memory_audit.jsonl"
    runtime_vis_debug_dir = layout.scene_dir / "runtime_vis_debug"
    runtime_vis_debug_vis_dir = runtime_vis_debug_dir / "vis"
    runtime_vis_debug_jsonl_path = runtime_vis_debug_dir / "runtime_vis_debug.jsonl"
    if bool(args.runtime_vis_debug):
        runtime_vis_debug_vis_dir.mkdir(parents=True, exist_ok=True)
    frame_metrics: list[dict[str, Any]] = []
    frame_audits: list[dict[str, Any]] = []
    runtime_vis_debug_overlay_sec = 0.0
    geometry_accum: dict[tuple[int, int, int], list[Any]] = {}
    submitted_prefetch_frame_ids: set[int] = set()
    process_frame_sec_total = 0.0
    mapping_start = time.perf_counter()
    processed = 0
    audit_handle = (
        audit_jsonl_path.open("w", encoding="utf-8")
        if output_profile.benchmark_audit_enabled
        else None
    )
    runtime_vis_debug_handle = (
        runtime_vis_debug_jsonl_path.open("w", encoding="utf-8")
        if bool(args.runtime_vis_debug)
        else None
    )
    try:
        with frame_metrics_path.open("w", encoding="utf-8") as metrics_handle:
            next_prefetch_index = 0
            if prefetch_enabled and selected_dataset_indices:
                first_frame = dataset[int(selected_dataset_indices[0])]
                if prefetcher is not None and prefetcher.submit(_build_prefetch_frame(first_frame, 0)):
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
                    submitted = prefetcher.submit(_build_prefetch_frame(next_frame, next_prefetch_index))
                    if submitted:
                        submitted_prefetch_frame_ids.add(next_prefetch_index)
                        next_prefetch_index += 1
                        submitted_next_before_processing = True

                prev_object_ids = set(pipeline.state.objects)
                prev_next_object_id = int(pipeline.state.next_object_id)
                process_start = time.perf_counter()
                state = pipeline.process_frame(
                    frame.rgb,
                    frame.depth,
                    frame.pose,
                    frame.intrinsics,
                    timestamp=frame.timestamp,
                    source_frame_id=frame.frame_id,
                    proposal_bundle=proposal_bundle,
                )
                process_frame_sec_total += time.perf_counter() - process_start
                processed += 1
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
                    submitted = prefetcher.submit(_build_prefetch_frame(next_frame, next_prefetch_index))
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
                    raise RuntimeError("Pipeline debug outputs missing during checkpointed run.")
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
                    "object_update_stage_timings": dict(
                        pipeline.last_frame_debug.get("object_update_stage_timings", {}) or {}
                    ),
                    "association_summary": dict(
                        (pipeline.last_frame_debug.get("association_debug", {}) or {}).get("summary", {}) or {}
                    ),
                    "active_set_debug": dict(pipeline.last_frame_debug.get("active_set_debug", {}) or {}),
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
                if runtime_vis_debug_handle is not None:
                    runtime_debug_record = build_runtime_vis_debug_record(
                        frame_id=int(frame.frame_id),
                        processed_frame_index=int(processed_frame_id),
                        raw_proposals=list(pipeline.last_raw_proposals),
                        runtime_output=runtime_output,
                        stage_timings=dict(record["stage_timings"]),
                        max_decisions=int(args.runtime_vis_debug_max_decisions),
                    )
                    overlay_every = max(1, int(args.runtime_vis_debug_every))
                    should_save_runtime_overlay = (
                        bool(output_profile.runtime_vis_debug_images_enabled)
                        and (
                            processed_frame_id == 0
                            or processed_frame_id % overlay_every == 0
                            or idx == frame_limit
                        )
                    )
                    if should_save_runtime_overlay:
                        overlay_path = runtime_vis_debug_vis_dir / f"{frame.frame_id:04d}.png"
                        overlay_start = time.perf_counter()
                        save_runtime_vis_debug_overlay(
                            frame=frame,
                            raw_proposals=list(pipeline.last_raw_proposals),
                            runtime_output=runtime_output,
                            output_path=overlay_path,
                        )
                        runtime_vis_debug_overlay_sec += float(time.perf_counter() - overlay_start)
                        runtime_debug_record["overlay_path"] = str(overlay_path)
                    runtime_vis_debug_handle.write(json.dumps(runtime_debug_record) + "\n")
                if audit_handle is not None:
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
                        overlay_path = layout.audit_vis_dir / f"{frame.frame_id:04d}.png"
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
                if not args.quiet and (idx == 1 or idx % 20 == 0 or idx == frame_limit):
                    print(
                        f"frame={frame.frame_id:04d} processed={idx:04d}/{frame_limit:04d} "
                        f"objects={len(state.objects):03d} raw={record['raw_proposal_count']:03d} "
                        f"matched={record['matched_patch_count']:03d}"
                    )
    finally:
        prefetch_stats = _prefetch_stats(prefetcher)
        if prefetcher is not None:
            prefetcher.close()
        if audit_handle is not None:
            audit_handle.close()
        if runtime_vis_debug_handle is not None:
            runtime_vis_debug_handle.close()

    mapping_sec = time.perf_counter() - mapping_start
    build_timing = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mapping_loop_sec_excluding_init_and_final_outputs": float(mapping_sec),
        "process_frame_sec_total": float(process_frame_sec_total),
        "init_sec_excluded": float(init_sec),
        "sec_per_frame": float(mapping_sec / processed) if processed else None,
        "frames_per_sec": float(processed / mapping_sec) if mapping_sec > 0 else None,
        "processed_frames": int(processed),
        "requested_frames": int(args.num_frames),
        "frame_stride": int(frame_stride),
        "final_outputs_generated": False,
        "checkpoint_path": checkpoint_path,
        "prefetch_enabled": bool(prefetch_enabled),
        "prefetch_submitted_count": int(prefetch_stats.get("submitted_count", len(submitted_prefetch_frame_ids)) or 0),
        "prefetch_hit_count": int(prefetch_stats.get("hit_count", 0) or 0),
        "prefetch_miss_count": int(prefetch_stats.get("miss_count", 0) or 0),
        "prefetch_wait_sec_total": float(prefetch_stats.get("wait_sec_total", 0.0) or 0.0),
        "prefetch_exception_count": int(prefetch_stats.get("exception_count", 0) or 0),
        "prefetch_last_exception": str(prefetch_stats.get("last_exception", "") or ""),
        "final_object_count": int(len(pipeline.state.objects)),
        "final_provisional_object_count": int(len(pipeline.state.provisional_objects)),
        "runtime_vis_debug_enabled": bool(args.runtime_vis_debug),
        "runtime_vis_debug_jsonl_path": runtime_vis_debug_jsonl_path if bool(args.runtime_vis_debug) else "",
        "runtime_vis_debug_vis_dir": runtime_vis_debug_vis_dir if bool(args.runtime_vis_debug) else "",
        "benchmark_profile": output_profile.benchmark_profile,
        "headline_excludes_runtime_vis_debug_artifacts": bool(
            output_profile.headline_excludes_runtime_vis_debug_artifacts
        ),
        "runtime_vis_debug_images_enabled": bool(output_profile.runtime_vis_debug_images_enabled),
        "runtime_vis_debug_overlay_sec_excluded_from_headline": float(runtime_vis_debug_overlay_sec),
    }
    object_update_module = getattr(pipeline, "object_update", None)
    if object_update_module is not None and hasattr(object_update_module, "flush_deferred_geometry"):
        flush_start = time.perf_counter()
        build_timing["local_pcd_deferred_flush_object_count"] = int(
            object_update_module.flush_deferred_geometry(pipeline.state)
        )
        build_timing["local_pcd_deferred_flush_sec_excluded_from_mapping"] = float(time.perf_counter() - flush_start)
    checkpoint_payload = build_room0_checkpoint_payload(
        experiment_name=experiment_name,
        layout=layout,
        args=args,
        pipeline=pipeline,
        state=pipeline.state,
        geometry_accum=geometry_accum,
        frame_metrics=frame_metrics,
        frame_audits=frame_audits,
        frame_limit=frame_limit,
        selected_dataset_indices=selected_dataset_indices,
        output_profile=output_profile,
        build_timing=build_timing,
    )
    save_start = time.perf_counter()
    save_room0_checkpoint(checkpoint_path, checkpoint_payload)
    build_timing["checkpoint_save_sec_excluded_from_mapping"] = float(time.perf_counter() - save_start)
    build_timing["mapping_state_size_bytes"] = int(checkpoint_path.stat().st_size) if checkpoint_path.exists() else 0
    checkpoint_size_audit = audit_checkpoint_payload_sizes(checkpoint_payload)
    build_timing["checkpoint_size_audit"] = checkpoint_size_audit
    build_timing["checkpoint_largest_payload_keys"] = checkpoint_size_audit["largest_keys"]
    build_timing.update(
        build_benchmark_contract_payload(
            state=pipeline.state,
            frame_metrics=frame_metrics,
            online_mapping_sec=build_timing["mapping_loop_sec_excluding_init_and_final_outputs"],
            finalization_sec=(
                float(build_timing.get("local_pcd_deferred_flush_sec_excluded_from_mapping", 0.0) or 0.0)
                + float(build_timing.get("checkpoint_save_sec_excluded_from_mapping", 0.0) or 0.0)
            ),
            eval_io_sec=0.0,
            mapping_state_size_bytes=build_timing["mapping_state_size_bytes"],
        )
    )
    if output_profile.headline_excludes_runtime_vis_debug_artifacts:
        build_timing["headline_online_mapping_sec"] = max(
            0.0,
            float(build_timing.get("online_mapping_sec", 0.0) or 0.0)
            - float(build_timing.get("runtime_vis_debug_overlay_sec_excluded_from_headline", 0.0) or 0.0),
        )
    checkpoint_payload["build_timing"] = dict(build_timing)
    save_room0_checkpoint(checkpoint_path, checkpoint_payload)
    build_timing["mapping_state_size_bytes"] = int(checkpoint_path.stat().st_size) if checkpoint_path.exists() else 0
    checkpoint_size_audit = audit_checkpoint_payload_sizes(checkpoint_payload)
    build_timing["checkpoint_size_audit"] = checkpoint_size_audit
    build_timing["checkpoint_largest_payload_keys"] = checkpoint_size_audit["largest_keys"]
    checkpoint_payload["build_timing"] = dict(build_timing)
    save_room0_checkpoint(checkpoint_path, checkpoint_payload)
    _write_json(layout.run_root / "mapping_timer_result.json", build_timing)
    return checkpoint_payload


def export_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    payload = load_room0_checkpoint(checkpoint_path)
    args = args_from_room0_checkpoint(payload)
    experiment_name = str(payload["experiment_name"])
    scene_name = normalize_scene_name(getattr(args, "scene_name", payload.get("scene_name", "room0")))
    args.scene_name = scene_name
    layout = build_room0_run_layout(Path(args.output_root), experiment_name, scene_name=scene_name)
    dataset = ReplicaRoom0Dataset(args.dataset_root)
    pipeline = pipeline_from_room0_checkpoint(payload)
    output_profile = output_profile_from_checkpoint(payload.get("output_profile"))
    build_timing = dict(payload.get("build_timing", {}) or {})
    mapping_timer_path = layout.run_root / "mapping_timer_result.json"
    if mapping_timer_path.exists():
        build_timing.update(json.loads(mapping_timer_path.read_text(encoding="utf-8")))
    export_start = time.perf_counter()
    export_result = export_room0_outputs(
        layout=layout,
        experiment_name=experiment_name,
        dataset=dataset,
        args=args,
        frame_limit=int(payload["frame_limit"]),
        pipeline=pipeline,
        state=payload["state"],
        geometry_accum=payload["geometry_accum"],
        frame_metrics=list(payload.get("frame_metrics", []) or []),
        output_profile=output_profile,
        frame_audits=list(payload.get("frame_audits", []) or []),
        benchmark_timing={
            "online_mapping_sec": build_timing.get("online_mapping_sec"),
            "headline_online_mapping_sec": build_timing.get("headline_online_mapping_sec"),
            "finalization_sec": build_timing.get("finalization_sec"),
            "mapping_state_size_bytes": build_timing.get("mapping_state_size_bytes"),
            "benchmark_profile": build_timing.get("benchmark_profile"),
            "headline_excludes_runtime_vis_debug_artifacts": build_timing.get(
                "headline_excludes_runtime_vis_debug_artifacts"
            ),
            "runtime_vis_debug_overlay_sec_excluded_from_headline": build_timing.get(
                "runtime_vis_debug_overlay_sec_excluded_from_headline"
            ),
            "checkpoint_size_audit": build_timing.get("checkpoint_size_audit", {}),
            "checkpoint_largest_payload_keys": build_timing.get("checkpoint_largest_payload_keys", []),
        },
    )
    export_total_sec = float(time.perf_counter() - export_start)
    build_finalization_sec = float(build_timing.get("finalization_sec", 0.0) or 0.0)
    finalization_sec = float(export_result.finalization_sec)
    export_finalization_sec = max(0.0, finalization_sec - build_finalization_sec)
    eval_io_sec = max(0.0, export_total_sec - export_finalization_sec)
    benchmark_contract = build_benchmark_contract_payload(
        state=payload["state"],
        frame_metrics=list(payload.get("frame_metrics", []) or []),
        online_mapping_sec=build_timing.get("online_mapping_sec"),
        finalization_sec=finalization_sec,
        eval_io_sec=eval_io_sec,
        mapping_state_size_bytes=build_timing.get("mapping_state_size_bytes"),
        largest_export_size_bytes=export_result.largest_export_size_bytes,
    )
    export_timing = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": checkpoint_path,
        "export_eval_sec": export_total_sec,
        "report_path": export_result.report_path,
        "results_path": export_result.results_path,
        "instance_map_path": export_result.instance_map_path,
        "dense_surface_path": export_result.dense_surface_path,
        "dense_instance_path": export_result.dense_instance_path,
        "miou": export_result.miou,
        "macc": export_result.macc,
        "fmiou": export_result.fmiou,
        "fmacc": export_result.fmacc,
        "final_object_count": export_result.final_object_count,
        "dense_geometry_point_count": export_result.dense_geometry_point_count,
        **benchmark_contract,
        "benchmark_profile": build_timing.get("benchmark_profile", output_profile.benchmark_profile),
        "headline_online_mapping_sec": build_timing.get(
            "headline_online_mapping_sec",
            benchmark_contract.get("headline_online_mapping_sec"),
        ),
        "headline_excludes_runtime_vis_debug_artifacts": bool(
            build_timing.get(
                "headline_excludes_runtime_vis_debug_artifacts",
                output_profile.headline_excludes_runtime_vis_debug_artifacts,
            )
        ),
        "runtime_vis_debug_overlay_sec_excluded_from_headline": float(
            build_timing.get("runtime_vis_debug_overlay_sec_excluded_from_headline", 0.0) or 0.0
        ),
        "checkpoint_size_audit": build_timing.get("checkpoint_size_audit", {}),
        "checkpoint_largest_payload_keys": build_timing.get("checkpoint_largest_payload_keys", []),
    }
    _write_json(layout.run_root / "export_eval_timer_result.json", export_timing)
    _update_report_sidecar(Path(export_result.report_path).with_suffix(".json"), benchmark_contract)
    if mapping_timer_path.exists():
        mapping_timer = json.loads(mapping_timer_path.read_text(encoding="utf-8"))
        mapping_timer["final_outputs_generated"] = True
        mapping_timer["export_eval_timer_result_path"] = str(layout.run_root / "export_eval_timer_result.json")
        mapping_timer.update(benchmark_contract)
        _write_json(mapping_timer_path, mapping_timer)
    return export_timing


def run_selected_modes(args: argparse.Namespace, *, experiment_name: str, checkpoint_path: Path) -> None:
    if args.mode in {"build", "build-export"}:
        build_checkpoint(args, experiment_name=experiment_name, checkpoint_path=checkpoint_path)
    if args.mode in {"export", "build-export"}:
        export_checkpoint(checkpoint_path)


def main() -> None:
    args = parse_args()
    if args.mode == "export" and args.checkpoint_path is not None:
        bootstrap_payload = load_room0_checkpoint(args.checkpoint_path)
        bootstrap_args = args_from_room0_checkpoint(bootstrap_payload)
        experiment_name = str(bootstrap_payload["experiment_name"])
        scene_name = normalize_scene_name(
            getattr(bootstrap_args, "scene_name", bootstrap_payload.get("scene_name", "room0"))
        )
        layout = build_room0_run_layout(Path(bootstrap_args.output_root), experiment_name, scene_name=scene_name)
        checkpoint_path = args.checkpoint_path
    else:
        if args.mode == "export" and args.experiment_name is None:
            raise ValueError("--mode export requires --checkpoint-path or --experiment-name")
        experiment_name = _experiment_name(args)
        scene_name = normalize_scene_name(getattr(args, "scene_name", "room0"))
        args.scene_name = scene_name
        layout = build_room0_run_layout(args.output_root, experiment_name, scene_name=scene_name)
        checkpoint_path = args.checkpoint_path or _default_checkpoint_path(layout)

    _write_json(
        layout.run_root / "status.json",
        {
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "mode": args.mode,
            "scene_name": scene_name,
            "checkpoint_path": checkpoint_path,
        },
    )
    try:
        if args.quiet:
            log_path = layout.run_root / "silent_run.log"
            with log_path.open("a", encoding="utf-8") as log_handle:
                with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                    run_selected_modes(args, experiment_name=experiment_name, checkpoint_path=checkpoint_path)
        else:
            run_selected_modes(args, experiment_name=experiment_name, checkpoint_path=checkpoint_path)
        _write_json(
            layout.run_root / "status.json",
            {
                "status": "complete",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "mode": args.mode,
                "scene_name": scene_name,
                "checkpoint_path": checkpoint_path,
                "run_root": layout.run_root,
                "log_path": layout.run_root / "silent_run.log" if args.quiet else "",
            },
        )
        if not args.quiet:
            print(f"run_root: {layout.run_root}")
            print(f"checkpoint: {checkpoint_path}")
    except Exception as exc:
        _write_json(
            layout.run_root / "status.json",
            {
                "status": "failed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "mode": args.mode,
                "scene_name": scene_name,
                "checkpoint_path": checkpoint_path,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


if __name__ == "__main__":
    main()
