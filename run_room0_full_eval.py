#!/usr/bin/env python3
"""Full room0 run with instance-map export, dense projection, evaluation, and report."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from run_replica_room0_analysis import build_proposal_module
from src.core.data_structures import ObjectState, SystemState
from src.datasets import ReplicaRoom0Dataset
from src.pipelines.main_pipeline import Pipeline


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--proposal-backend", type=str, default="sam2")
    parser.add_argument("--proposal-device", type=str, default="cuda")
    parser.add_argument("--sam-version", type=str, default="2")
    parser.add_argument("--sam-encoder", type=str, default="hiera_l")
    parser.add_argument("--sam-repo-root", type=Path, default=Path("/home/phl/vv/paper2/OVO/thirdParty/segment-anything-2"))
    parser.add_argument("--sam-ckpt-path", type=Path, default=Path("/home/phl/vv/paper2/DovSG/checkpoints/segment-anything-2"))
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
    parser.add_argument("--pipeline-verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = args.experiment_name or f"{timestamp}_room0_ovopro_sam2v2_full"
    run_root = args.output_root / experiment_name
    scene_dir = run_root / "room0"
    eval_dir = run_root / "replica"
    exports_dir = scene_dir / "exports"
    vis_dir = scene_dir / "vis"
    audit_dir = scene_dir / "local_memory_audit"
    audit_vis_dir = audit_dir / "vis"
    for directory in [scene_dir, eval_dir, exports_dir, vis_dir, audit_dir, audit_vis_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    dataset = ReplicaRoom0Dataset(args.dataset_root)
    frame_limit = len(dataset) if args.num_frames <= 0 else min(args.num_frames, len(dataset))

    pipeline = Pipeline(config_path=str(args.config_path))
    pipeline.verbose = bool(args.pipeline_verbose)
    pipeline.proposal = build_proposal_module(pipeline.proposal.config, args)

    frame_metrics_path = scene_dir / "frame_metrics.jsonl"
    audit_jsonl_path = audit_dir / "local_memory_audit.jsonl"
    audit_log_path = audit_dir / "local_memory_audit.log"
    audit_summary_path = audit_dir / "local_memory_audit_summary.json"
    geometry_accum: dict[tuple[int, int, int], list[np.ndarray | int]] = {}
    frame_metrics: list[dict[str, Any]] = []
    frame_audits: list[dict[str, Any]] = []

    print("=" * 60)
    print("OVIOVO Full room0 run")
    print("=" * 60)
    print(f"run_root:     {run_root}")
    print(f"scene_dir:    {scene_dir}")
    print(f"frame_limit:  {frame_limit}")
    print(f"backend:      {args.proposal_backend}")
    print(f"device:       {args.proposal_device}")
    print(f"conda_env:    {os.environ.get('CONDA_DEFAULT_ENV')}")

    with (
        frame_metrics_path.open("w", encoding="utf-8") as metrics_handle,
        audit_jsonl_path.open("w", encoding="utf-8") as audit_handle,
    ):
        for idx, frame in enumerate(dataset.iter_frames(limit=frame_limit), start=1):
            prev_object_ids = set(pipeline.state.objects)
            prev_next_object_id = int(pipeline.state.next_object_id)
            state = pipeline.process_frame(
                frame.rgb,
                frame.depth,
                frame.pose,
                frame.intrinsics,
                timestamp=frame.timestamp,
            )
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
            record = {
                "frame_id": int(frame.frame_id),
                "raw_proposal_count": len(pipeline.last_raw_proposals),
                "merged_group_count": len(runtime_output.groups),
                "refined_proposal_count": len(pipeline.last_refined_proposals),
                "patch_count": len(pipeline.last_patches),
                "matched_patch_count": len(association.matched),
                "new_patch_count": len(association.new_object_patches),
                "total_object_count": len(state.objects),
                "active_object_count": state_counts.get("active", 0),
                "dormant_object_count": state_counts.get("dormant", 0),
                "inactive_object_count": state_counts.get("inactive", 0),
                "semantic_updated_count": len(pipeline.last_frame_debug.get("semantic_updated_object_ids", [])),
                "actual_backend": pipeline.proposal.active_backend_name,
            }
            frame_metrics.append(record)
            metrics_handle.write(json.dumps(record) + "\n")

            audit_record = build_local_memory_frame_audit(
                frame=frame,
                pipeline=pipeline,
                prev_object_ids=prev_object_ids,
                prev_next_object_id=prev_next_object_id,
            )
            if audit_record["new_instance_count"] > 0 or audit_record["hard_failure_count"] > 0:
                overlay_path = audit_vis_dir / f"{frame.frame_id:04d}.png"
                save_local_memory_audit_overlay(
                    frame=frame,
                    raw_proposals=pipeline.last_raw_proposals,
                    raw_records=audit_record["raw_proposals"],
                    output_path=overlay_path,
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

    state = pipeline.state
    tsdf_records = build_tsdf_backbone_records(state)
    local_records = build_local_memory_records(state)

    instance_map_path = exports_dir / "room0_instance_map.ply"
    instance_backbone_path = exports_dir / "room0_instance_map_tsdf_backbone.ply"
    local_memory_path = exports_dir / "room0_instance_map_local_memory.ply"
    write_binary_ply(instance_map_path, tsdf_records)
    write_binary_ply(instance_backbone_path, tsdf_records)
    write_binary_ply(local_memory_path, local_records)

    dense_points, dense_rgb = finalize_geometry_accum(geometry_accum)
    labels, state_ids, supports = project_instances_to_dense_points(
        dense_points=dense_points,
        stable_instances=tsdf_records,
        instance_voxel_size=float(args.instance_voxel_size),
        neighbor_radius=max(0, int(args.projection_neighbor_radius)),
    )
    projected_colors = np.asarray(
        [instance_color(int(object_id)) if object_id >= 0 else (180, 180, 180) for object_id in labels],
        dtype=np.uint8,
    )
    dense_rgb_path = exports_dir / "room0_dense_geometry_fused_rgb.ply"
    dense_inst_path = exports_dir / "room0_dense_geometry_instance_projected.ply"
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

    render_projection(dense_points, dense_rgb, plane="xz", output_path=vis_dir / "room0_dense_xz_rgb.png")
    render_projection(dense_points, projected_colors, plane="xz", output_path=vis_dir / "room0_dense_xz_instance.png")
    render_projection(dense_points, dense_rgb, plane="xy", output_path=vis_dir / "room0_dense_xy_rgb.png")
    render_projection(dense_points, projected_colors, plane="xy", output_path=vis_dir / "room0_dense_xy_instance.png")
    render_preview_sheet(
        [
            vis_dir / "room0_dense_xz_rgb.png",
            vis_dir / "room0_dense_xz_instance.png",
            vis_dir / "room0_dense_xy_rgb.png",
            vis_dir / "room0_dense_xy_instance.png",
        ],
        vis_dir / "room0_dense_projection_preview.png",
    )

    gt_vertices = read_gt_vertices(args.gt_mesh_ply)
    gt_labels = read_gt_labels(args.gt_labels)
    class_names = read_class_names(args.gt_info_json)
    evaluation = evaluate_semantics(
        gt_vertices=gt_vertices,
        gt_labels=gt_labels,
        dense_points=dense_points,
        object_ids=labels,
        class_names=class_names,
    )

    write_eval_artifacts(eval_dir, gt_vertices, gt_labels, evaluation, class_names)
    audit_summary = summarize_local_memory_audits(frame_audits, state)
    audit_summary_path.write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    audit_log_path.write_text(render_local_memory_audit_log(frame_audits), encoding="utf-8")
    write_run_report(
        scene_dir=scene_dir,
        eval_dir=eval_dir,
        experiment_name=experiment_name,
        dataset=dataset,
        args=args,
        frame_limit=frame_limit,
        pipeline=pipeline,
        state=state,
        tsdf_records=tsdf_records,
        dense_points=dense_points,
        labels=labels,
        evaluation=evaluation,
        frame_metrics=frame_metrics,
        audit_dir=audit_dir,
    )

    print("\nSaved outputs:")
    print(f"- run_root:              {run_root}")
    print(f"- report:                {scene_dir / 'run_report.md'}")
    print(f"- eval results:          {eval_dir / 'results.json'}")
    print(f"- stable instance map:   {instance_map_path}")
    print(f"- dense instance map:    {dense_inst_path}")


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
    refined_proposals = list(pipeline.last_refined_proposals)
    patches = list(pipeline.last_patches)

    group_lookup = {int(group.group_id): group for group in runtime_output.groups}
    profile_lookup = {int(profile.proposal_id): profile for profile in runtime_output.proposal_profiles}

    refined_by_raw: dict[int, list[Any]] = defaultdict(list)
    for refined in refined_proposals:
        raw_id = int(refined.metadata.get("source_raw_proposal_id", refined.proposal_id))
        refined_by_raw[raw_id].append(refined)

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

    new_patch_ids = [int(patch_id) for patch_id in association.new_object_patches]
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
    outcome_counts: Counter[str] = Counter()

    for proposal in sorted(raw_proposals, key=lambda item: int(item.proposal_id)):
        raw_id = int(proposal.proposal_id)
        profile = profile_lookup.get(raw_id)
        group_id = runtime_output.raw_to_group.get(raw_id)
        group = group_lookup.get(int(group_id)) if group_id is not None else None
        refined_components = sorted(refined_by_raw.get(raw_id, []), key=lambda item: int(item.proposal_id))

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

                patch_records.append(
                    {
                        "patch_id": patch_id,
                        "source_refined_proposal_id": refined_id,
                        "point_count": int(len(patch.points)),
                        "split_category": split_category,
                        "matched_existing": matched_payload,
                        "new_object_id": int(created_object_id) if created_object_id is not None else None,
                        "top_association_candidates": candidate_scores_by_patch.get(patch_id, [])[:3],
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
            new_instance_events.append(
                {
                    "object_id": int(object_id),
                    "source_proposal_id": raw_id,
                    "source_patch_id": int(related_patch["patch_id"]) if related_patch is not None else None,
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
    return {
        "frame_id": int(frame.frame_id),
        "raw_proposal_count": len(raw_proposals),
        "refined_proposal_count": len(refined_proposals),
        "patch_count": len(patches),
        "bg_patch_count": len(bg_patch_ids),
        "obj_patch_count": len(obj_patch_ids),
        "amb_patch_count": len(amb_patch_ids),
        "new_instance_count": len(new_instance_events),
        "new_instance_events": new_instance_events,
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
        "raw_proposals": raw_records,
        "overlay_path": None,
    }


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
        return f"patches remained ambiguous after bg/object split and were not eligible for instance association. runtime_vis class={proposal_class}."

    if final_outcome == "bg_obj_split_mixed_non_object":
        return "all lifted patches were filtered into background/ambiguous buckets, so none reached object association."

    return "proposal survived early stages but did not end in a tracked local-memory instance; inspect patch-level association candidates."


def summarize_local_memory_audits(frame_audits: list[dict[str, Any]], state: SystemState) -> dict[str, Any]:
    outcome_counts: Counter[str] = Counter()
    created_frames: list[int] = []
    overlay_frames: list[int] = []
    for frame_audit in frame_audits:
        if frame_audit.get("new_instance_count", 0) > 0:
            created_frames.append(int(frame_audit["frame_id"]))
        if frame_audit.get("overlay_path"):
            overlay_frames.append(int(frame_audit["frame_id"]))
        for outcome, count in frame_audit.get("outcome_counts", {}).items():
            outcome_counts[str(outcome)] += int(count)

    return {
        "frame_count": int(len(frame_audits)),
        "frames_with_new_instances": created_frames,
        "frames_with_saved_overlays": overlay_frames,
        "total_new_instance_events": int(sum(item.get("new_instance_count", 0) for item in frame_audits)),
        "raw_outcome_counts": dict(sorted(outcome_counts.items())),
        "final_object_count": int(len(state.objects)),
        "objects_with_local_memory_points": int(sum(1 for obj in state.objects.values() if len(obj.local_pcd) > 0)),
    }


def render_local_memory_audit_log(frame_audits: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for frame_audit in frame_audits:
        frame_id = int(frame_audit["frame_id"])
        if frame_audit.get("new_instance_count", 0) <= 0 and frame_audit.get("hard_failure_count", 0) <= 0:
            continue
        lines.append(f"frame={frame_id:04d}")
        if frame_audit.get("overlay_path"):
            lines.append(f"overlay={frame_audit['overlay_path']}")
        for event in frame_audit.get("new_instance_events", []):
            lines.append(
                "  new_instance "
                f"object_id={event['object_id']} source_proposal={event['source_proposal_id']} "
                f"source_patch={event['source_patch_id']} local_points={event['local_point_count']} "
                f"stability={event['stability_score']:.3f} reason={event['reason']}"
            )
        for raw_record in frame_audit.get("raw_proposals", []):
            if raw_record.get("final_outcome") == "new_instance_created":
                continue
            lines.append(
                "  raw_proposal "
                f"id={raw_record['proposal_id']} outcome={raw_record['final_outcome']} "
                f"reason={raw_record['reason']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_local_memory_audit_overlay(
    frame,
    raw_proposals: list[Any],
    raw_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_lookup = {int(record["proposal_id"]): record for record in raw_records}
    canvas = np.asarray(frame.rgb, dtype=np.uint8).copy()
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
    image.save(output_path)


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
    if outcome == "new_instance_created":
        object_ids = ",".join(str(value) for value in raw_record.get("created_object_ids", []))
        return f"r{proposal_id}: new {object_ids}"
    if outcome == "matched_existing_instance":
        object_ids = ",".join(str(value) for value in raw_record.get("matched_existing_ids", []))
        return f"r{proposal_id}: match {object_ids}"
    if outcome == "bg_obj_split_background":
        return f"r{proposal_id}: bg"
    if outcome == "bg_obj_split_ambiguous":
        return f"r{proposal_id}: amb"
    if outcome == "patch_lifting_failed":
        return f"r{proposal_id}: no3d"
    if outcome == "depth_refinement_removed":
        return f"r{proposal_id}: refine_drop"
    return f"r{proposal_id}: {outcome}"


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
    return np.asarray(rows, dtype=PLY_DTYPE)


def build_local_memory_records(state: SystemState) -> np.ndarray:
    rows = []
    for object_id, obj in sorted(state.objects.items()):
        points = np.asarray(obj.local_pcd, dtype=np.float32)
        if points.size == 0:
            continue
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
    return np.asarray(rows, dtype=PLY_DTYPE)


def project_instances_to_dense_points(
    dense_points: np.ndarray,
    stable_instances: np.ndarray,
    instance_voxel_size: float,
    neighbor_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    owner_map: dict[tuple[int, int, int], tuple[int, int, float]] = {}
    stable_points = np.stack([stable_instances["x"], stable_instances["y"], stable_instances["z"]], axis=1)
    stable_indices = np.floor(stable_points / instance_voxel_size).astype(np.int32)

    for voxel, row in zip(stable_indices, stable_instances):
        owner_map[(int(voxel[0]), int(voxel[1]), int(voxel[2]))] = (
            int(row["object_id"]),
            int(row["state_id"]),
            float(row["support"]),
        )

    labels = np.full(len(dense_points), -1, dtype=np.int32)
    state_ids = np.zeros(len(dense_points), dtype=np.uint8)
    supports = np.zeros(len(dense_points), dtype=np.float32)
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


def evaluate_semantics(
    gt_vertices: np.ndarray,
    gt_labels: np.ndarray,
    dense_points: np.ndarray,
    object_ids: np.ndarray,
    class_names: dict[int, str],
) -> dict[str, Any]:
    tree = cKDTree(dense_points)
    _, nn_indices = tree.query(gt_vertices, k=1, workers=-1)
    nearest_object_ids = object_ids[nn_indices]

    per_object_votes: dict[int, Counter] = defaultdict(Counter)
    valid_gt_mask = gt_labels >= 0
    for object_id, gt_label in zip(nearest_object_ids[valid_gt_mask], gt_labels[valid_gt_mask]):
        if object_id < 0:
            continue
        per_object_votes[int(object_id)][int(gt_label)] += 1

    object_to_class: dict[int, int] = {}
    for object_id, counter in per_object_votes.items():
        if counter:
            object_to_class[object_id] = counter.most_common(1)[0][0]

    pred_labels = np.full_like(gt_labels, fill_value=-1)
    for index, object_id in enumerate(nearest_object_ids):
        if object_id < 0:
            continue
        pred_labels[index] = object_to_class.get(int(object_id), -1)

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
) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    pred_labels = evaluation["pred_labels"]
    correctness = evaluation["correctness_colors"]

    gt_colors = np.asarray([class_color(int(label)) if label >= 0 else (80, 80, 80) for label in gt_labels], dtype=np.uint8)
    pred_colors = np.asarray([class_color(int(label)) if label >= 0 else (80, 80, 80) for label in pred_labels], dtype=np.uint8)

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
    dense_points: np.ndarray,
    labels: np.ndarray,
    evaluation: dict[str, Any],
    frame_metrics: list[dict[str, Any]],
    audit_dir: Path | None = None,
) -> None:
    active_count = int(sum(obj.state == ObjectState.ACTIVE for obj in state.objects.values()))
    dormant_count = int(sum(obj.state == ObjectState.DORMANT for obj in state.objects.values()))
    inactive_count = int(sum(obj.state == ObjectState.INACTIVE for obj in state.objects.values()))
    per_class_sorted = sorted(evaluation["per_class"], key=lambda item: item["iou"], reverse=True)
    top_lines = [f"- `{item['class_name']}`: IoU `{item['iou']:.4f}`, Acc `{item['acc']:.4f}`" for item in per_class_sorted[:25]]

    report = "\n".join(
        [
            "# OVO Run Report",
            "",
            f"- Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            f"- Run dir: `{scene_dir}`",
            "- Dataset: `replica`",
            "- Scene: `room0`",
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
            "",
            "## Metrics",
            f"- `mIoU`: {evaluation['miou']:.4f}",
            f"- `mAcc`: {evaluation['macc']:.4f}",
            f"- `f-mIoU`: {evaluation['fmiou']:.4f}",
            f"- `f-mAcc`: {evaluation['fmacc']:.4f}",
            "",
            "## Evaluation Contract",
            "- `voxel-native map`: exported from the OVIOVO TSDF instance backbone as voxel-center points.",
            "- `evaluation source`: `dense_geometry_instance_projected`.",
            "- `derived_surface`: dense RGB-D fused geometry sampled into a voxelized point surface.",
            "- `semantic assignment`: each predicted instance id is mapped to a semantic class by majority GT class over nearest GT mesh vertices it covers.",
            "- `mIoU` / `mAcc`: computed on GT mesh vertices after nearest-neighbor transfer from the predicted dense geometry projection.",
            "",
            "## Evaluation Metadata",
            f"- `final_object_count`: `{len(state.objects)}`",
            f"- `active_object_count`: `{active_count}`",
            f"- `dormant_object_count`: `{dormant_count}`",
            f"- `inactive_object_count`: `{inactive_count}`",
            f"- `voxel_native_point_count`: `{len(tsdf_records)}`",
            f"- `dense_geometry_point_count`: `{len(dense_points)}`",
            f"- `projected_dense_labeled_point_count`: `{int(np.count_nonzero(labels >= 0))}`",
            f"- `projected_dense_object_count`: `{len(set(labels[labels >= 0].tolist())) if np.any(labels >= 0) else 0}`",
            f"- `mean_raw_proposal_count`: `{float(np.mean([item['raw_proposal_count'] for item in frame_metrics])):.4f}`",
            f"- `mean_matched_patch_count`: `{float(np.mean([item['matched_patch_count'] for item in frame_metrics])):.4f}`",
            "",
            "## Per-class",
            *(top_lines or ["- No per-class metrics available"]),
            "",
            "## Exports",
            f"- `room0_instance_map.ply`: `{scene_dir / 'exports' / 'room0_instance_map.ply'}`",
            f"- `room0_instance_map_local_memory.ply`: `{scene_dir / 'exports' / 'room0_instance_map_local_memory.ply'}`",
            f"- `room0_dense_geometry_instance_projected.ply`: `{scene_dir / 'exports' / 'room0_dense_geometry_instance_projected.ply'}`",
            f"- `room0_gtmesh_pred_semantic.ply`: `{eval_dir / 'room0_gtmesh_pred_semantic.ply'}`",
            f"- `room0_gtmesh_gt_semantic.ply`: `{eval_dir / 'room0_gtmesh_gt_semantic.ply'}`",
            f"- `room0_gtmesh_semantic_correctness.ply`: `{eval_dir / 'room0_gtmesh_semantic_correctness.ply'}`",
            *(
                [
                    f"- `local_memory_audit.jsonl`: `{audit_dir / 'local_memory_audit.jsonl'}`",
                    f"- `local_memory_audit.log`: `{audit_dir / 'local_memory_audit.log'}`",
                    f"- `local_memory_audit_summary.json`: `{audit_dir / 'local_memory_audit_summary.json'}`",
                    f"- `local_memory_audit_vis/`: `{audit_dir / 'vis'}`",
                ]
                if audit_dir is not None
                else []
            ),
            "",
        ]
    )
    report_path = scene_dir / "run_report.md"
    report_path.write_text(report, encoding="utf-8")

    sidecar = {
        "miou": evaluation["miou"],
        "macc": evaluation["macc"],
        "fmiou": evaluation["fmiou"],
        "fmacc": evaluation["fmacc"],
        "per_class": evaluation["per_class"],
        "frame_count": frame_limit,
        "final_object_count": len(state.objects),
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


def class_color(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(class_id) * 2654435761 % (2**32))
    color = (rng.random(3) * 205 + 40).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


if __name__ == "__main__":
    main()
