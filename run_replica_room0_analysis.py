#!/usr/bin/env python3
"""Run the OVIOVO pipeline on Replica room0 and export per-frame analysis JSON."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.datasets import ReplicaRoom0Dataset
from src.modules.proposal import ProposalModule
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
        "--output-root",
        type=Path,
        default=project_root / "outputs",
    )
    parser.add_argument("--run-tag", type=str, default="replica_room0_200f_analysis")
    parser.add_argument("--proposal-backend", type=str, default="sam2")
    parser.add_argument("--proposal-device", type=str, default="cuda")
    parser.add_argument("--sam-version", type=str, default="2.1")
    parser.add_argument("--sam-encoder", type=str, default="hiera_l")
    parser.add_argument("--sam-repo-root", type=Path, default=None)
    parser.add_argument(
        "--sam-ckpt-path",
        type=Path,
        default=project_root / "data" / "input" / "sam_ckpts",
    )
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument("--stability-score-th", type=float, default=0.95)
    parser.add_argument("--nms-iou-th", type=float, default=0.8)
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    parser.add_argument("--pipeline-verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"{timestamp}_{args.run_tag}"
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    dataset = ReplicaRoom0Dataset(args.dataset_root)
    pipeline = Pipeline(config_path=str(args.config_path))
    pipeline.verbose = bool(args.pipeline_verbose)
    pipeline.proposal = build_proposal_module(pipeline.proposal.config, args)

    frame_metrics: list[dict[str, Any]] = []
    object_birth_events: list[dict[str, Any]] = []
    object_histories: dict[int, dict[str, Any]] = {}

    frame_metrics_jsonl = output_dir / "frame_metrics.jsonl"
    with frame_metrics_jsonl.open("w", encoding="utf-8") as jsonl_handle:
        for frame in dataset.iter_frames(limit=args.num_frames):
            prev_objects = {
                object_id: {
                    "update_count": obj.update_count,
                    "semantic_observation_count": obj.semantic_memory.observation_count,
                    "state": obj.state.value,
                }
                for object_id, obj in pipeline.state.objects.items()
            }
            prev_object_ids = set(prev_objects)

            state = pipeline.process_frame(
                frame.rgb,
                frame.depth,
                frame.pose,
                frame.intrinsics,
                timestamp=frame.timestamp,
            )

            raw_proposals = pipeline.last_raw_proposals
            refined = pipeline.last_refined_proposals
            patches = pipeline.last_patches
            runtime_output = pipeline.last_runtime_vis_output
            association = pipeline.last_association
            if runtime_output is None or association is None:
                raise RuntimeError("Pipeline did not expose runtime_vis/association debug data.")

            current_object_ids = set(state.objects)
            new_object_ids = sorted(current_object_ids - prev_object_ids)
            updated_existing_ids = sorted(
                object_id
                for object_id in (current_object_ids & prev_object_ids)
                if state.objects[object_id].update_count > prev_objects[object_id]["update_count"]
            )
            semantic_updated_ids = list(pipeline.last_frame_debug.get("semantic_updated_object_ids", []))
            semantic_newly_started_ids = sorted(
                object_id
                for object_id in current_object_ids
                if state.objects[object_id].semantic_memory.observation_count
                > prev_objects.get(object_id, {}).get("semantic_observation_count", 0)
            )

            state_counts = summarize_state_counts(state)
            frame_payload = {
                "frame_id": int(frame.frame_id),
                "timestamp": float(frame.timestamp),
                "proposal": {
                    "requested_backend": pipeline.proposal.requested_backend_name,
                    "active_backend": pipeline.proposal.active_backend_name,
                    "backend_init_error": pipeline.proposal.backend_init_error,
                    "raw_count": len(raw_proposals),
                    "runtime_vis_merged_count": len(runtime_output.merged_proposals),
                    "refined_count": len(refined),
                    "patch_count": len(patches),
                },
                "runtime_vis": {
                    "group_stats": serialize(runtime_output.group_stats),
                    "group_to_linked_object": serialize(runtime_output.group_to_linked_object),
                },
                "active_set": {
                    "visible_ids": sorted(state.active_set.visible_ids),
                    "nearby_ids": sorted(state.active_set.nearby_ids),
                    "whole_prior_ids": sorted(state.active_set.whole_prior_ids),
                    "new_object_candidate_ids": sorted(state.active_set.new_object_candidate_ids),
                    "candidate_count": len(state.active_set.all_candidate_ids),
                },
                "association": {
                    "matched_count": len(association.matched),
                    "new_patch_count": len(association.new_object_patches),
                    "matched_pairs": [
                        {
                            "patch_id": int(patch_id),
                            "object_id": int(object_id),
                            "total_score": float(score.total_score),
                            "voxel_vote_score": float(score.voxel_vote_score),
                            "geometry_overlap": float(score.geometry_overlap),
                            "bbox_overlap": float(score.bbox_overlap),
                            "centroid_distance": float(score.centroid_distance),
                            "vote_result": serialize(score.vote_result),
                        }
                        for patch_id, object_id, score in association.matched
                    ],
                    "new_object_patches": [int(patch_id) for patch_id in association.new_object_patches],
                    "debug": serialize(association.debug),
                },
                "objects": {
                    "total_count": len(state.objects),
                    "state_counts": state_counts,
                    "new_object_ids": new_object_ids,
                    "updated_existing_ids": updated_existing_ids,
                    "semantic_updated_ids": semantic_updated_ids,
                    "semantic_newly_started_ids": semantic_newly_started_ids,
                    "background_point_count": int(len(state.background.point_cloud)),
                    "all_objects": [snapshot_object(state.objects[object_id]) for object_id in sorted(state.objects)],
                },
            }

            frame_metrics_record = {
                "frame_id": int(frame.frame_id),
                "raw_proposal_count": len(raw_proposals),
                "merged_proposal_count": len(runtime_output.merged_proposals),
                "refined_proposal_count": len(refined),
                "patch_count": len(patches),
                "active_set_candidate_count": len(state.active_set.all_candidate_ids),
                "matched_patch_count": len(association.matched),
                "new_patch_count": len(association.new_object_patches),
                "new_object_count": len(new_object_ids),
                "total_object_count": len(state.objects),
                "active_object_count": state_counts.get("active", 0),
                "semantic_updated_count": len(semantic_updated_ids),
                "actual_backend": pipeline.proposal.active_backend_name,
            }
            frame_metrics.append(frame_metrics_record)
            jsonl_handle.write(json.dumps(frame_metrics_record) + "\n")

            frame_json_path = frames_dir / f"frame{frame.frame_id:06d}.json"
            frame_json_path.write_text(json.dumps(serialize(frame_payload), indent=2), encoding="utf-8")

            new_patch_ids = [int(patch_id) for patch_id in association.new_object_patches]
            for patch_id, object_id in zip(new_patch_ids, new_object_ids):
                obj = state.objects[object_id]
                birth_event = {
                    "frame_id": int(frame.frame_id),
                    "patch_id": int(patch_id),
                    "object_id": int(object_id),
                    "initial_state": obj.state.value,
                    "initial_update_count": int(obj.update_count),
                    "initial_stability_score": float(
                        obj.debug.get("global_instance_substrate", {}).get("stability_score", 0.0)
                    ),
                    "initial_whole_evidence_score": float(obj.whole_evidence.whole_evidence_score),
                    "initial_assignment_score": float(obj.whole_evidence.assignment_score),
                    "initial_point_count": int(len(obj.local_pcd)),
                }
                object_birth_events.append(birth_event)

            for object_id, obj in state.objects.items():
                history = object_histories.setdefault(
                    object_id,
                    {
                        "object_id": int(object_id),
                        "creation_frame": int(obj.creation_frame),
                        "frames_present": [],
                        "frames_updated": [],
                        "frames_semantic_updated": [],
                    },
                )
                history["last_seen_frame"] = int(obj.last_seen_frame)
                history["final_state"] = obj.state.value
                history["final_update_count"] = int(obj.update_count)
                history["final_semantic_observation_count"] = int(obj.semantic_memory.observation_count)
                history["final_whole_evidence_score"] = float(obj.whole_evidence.whole_evidence_score)
                history["final_part_evidence_score"] = float(obj.whole_evidence.part_evidence_score)
                history["final_assignment_score"] = float(obj.whole_evidence.assignment_score)
                history["final_stability_score"] = float(
                    obj.debug.get("global_instance_substrate", {}).get("stability_score", 0.0)
                )
                history["final_label_hypotheses"] = serialize(obj.semantic_memory.label_hypotheses)
                history["frames_present"].append(int(frame.frame_id))
                if object_id in new_object_ids or object_id in updated_existing_ids:
                    history["frames_updated"].append(int(frame.frame_id))
                if object_id in semantic_newly_started_ids:
                    history["frames_semantic_updated"].append(int(frame.frame_id))

            print(
                f"frame={frame.frame_id:04d} raw={len(raw_proposals):03d} "
                f"merged={len(runtime_output.merged_proposals):03d} refined={len(refined):03d} "
                f"patches={len(patches):03d} matched={len(association.matched):03d} "
                f"new_obj={len(new_object_ids):03d} total_obj={len(state.objects):03d} "
                f"backend={pipeline.proposal.active_backend_name}"
            )

    object_lifecycle_summary = {
        "objects": [
            serialize(object_histories[object_id])
            for object_id in sorted(object_histories)
        ]
    }
    object_lifecycle_path = output_dir / "object_lifecycle_summary.json"
    object_lifecycle_path.write_text(json.dumps(object_lifecycle_summary, indent=2), encoding="utf-8")

    birth_events_path = output_dir / "instance_birth_events.json"
    birth_events_path.write_text(json.dumps(serialize(object_birth_events), indent=2), encoding="utf-8")

    frame_metrics_path = output_dir / "frame_metrics.json"
    frame_metrics_path.write_text(json.dumps(serialize(frame_metrics), indent=2), encoding="utf-8")

    summary = build_summary(
        dataset=dataset,
        pipeline=pipeline,
        args=args,
        output_dir=output_dir,
        frame_metrics=frame_metrics,
        birth_events=object_birth_events,
        object_histories=object_histories,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(serialize(summary), indent=2), encoding="utf-8")

    manifest = {
        "summary_json": str(summary_path),
        "frame_metrics_json": str(frame_metrics_path),
        "frame_metrics_jsonl": str(frame_metrics_jsonl),
        "instance_birth_events_json": str(birth_events_path),
        "object_lifecycle_summary_json": str(object_lifecycle_path),
        "frames_dir": str(frames_dir),
        "frame_json_count": len(frame_metrics),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"- output_dir:                 {output_dir}")
    print(f"- manifest:                   {manifest_path}")
    print(f"- summary:                    {summary_path}")
    print(f"- frame_metrics.json:         {frame_metrics_path}")
    print(f"- frame_metrics.jsonl:        {frame_metrics_jsonl}")
    print(f"- instance_birth_events.json: {birth_events_path}")
    print(f"- object_lifecycle_summary:   {object_lifecycle_path}")
    print(f"- frame detail jsons:         {len(frame_metrics)}")


def build_proposal_module(base_config: dict[str, Any], args: argparse.Namespace) -> ProposalModule:
    config = json.loads(json.dumps(base_config))
    config["backend"] = args.proposal_backend
    config["min_mask_area"] = args.min_mask_area
    nested = config.setdefault(args.proposal_backend, {})
    if args.proposal_backend == "sam2":
        nested.update(
            {
                "device": args.proposal_device,
                "sam_version": args.sam_version,
                "sam_encoder": args.sam_encoder,
                "sam_repo_root": str(args.sam_repo_root) if args.sam_repo_root is not None else "",
                "sam_ckpt_path": str(args.sam_ckpt_path),
                "points_per_side": args.points_per_side,
                "max_proposals": args.max_proposals,
                "confidence_threshold": args.confidence_threshold,
                "stability_score_th": args.stability_score_th,
                "nms_iou_th": args.nms_iou_th,
                "min_mask_region_area": args.min_mask_region_area,
            }
        )
        config.update(nested)
    return ProposalModule(config)


def summarize_state_counts(state) -> dict[str, int]:
    counts = Counter(obj.state.value for obj in state.objects.values())
    return {key: int(value) for key, value in sorted(counts.items())}


def snapshot_object(obj) -> dict[str, Any]:
    return {
        "object_id": int(obj.object_id),
        "state": obj.state.value,
        "creation_frame": int(obj.creation_frame),
        "last_seen_frame": int(obj.last_seen_frame),
        "update_count": int(obj.update_count),
        "local_point_count": int(len(obj.local_pcd)),
        "whole_evidence_score": float(obj.whole_evidence.whole_evidence_score),
        "part_evidence_score": float(obj.whole_evidence.part_evidence_score),
        "assignment_score": float(obj.whole_evidence.assignment_score),
        "stability_score": float(obj.debug.get("global_instance_substrate", {}).get("stability_score", 0.0)),
        "owned_voxel_count": int(obj.debug.get("global_instance_substrate", {}).get("owned_voxel_count", 0)),
        "semantic_observation_count": int(obj.semantic_memory.observation_count),
        "label_hypotheses": serialize(obj.semantic_memory.label_hypotheses),
    }


def build_summary(
    dataset: ReplicaRoom0Dataset,
    pipeline: Pipeline,
    args: argparse.Namespace,
    output_dir: Path,
    frame_metrics: list[dict[str, Any]],
    birth_events: list[dict[str, Any]],
    object_histories: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    final_state = pipeline.state
    frame_count = len(frame_metrics)
    if frame_count == 0:
        raise RuntimeError("No frames were processed.")

    backend = pipeline.proposal.backend
    backend_info = getattr(backend, "last_generation_info", {})

    return {
        "dataset_summary": dataset.summary(),
        "run": {
            "num_frames_requested": int(args.num_frames),
            "num_frames_processed": int(frame_count),
            "output_dir": str(output_dir),
            "python_executable": sys.executable,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "proposal_backend": {
            "requested_backend": pipeline.proposal.requested_backend_name,
            "active_backend": pipeline.proposal.active_backend_name,
            "backend_init_error": pipeline.proposal.backend_init_error,
            "last_generation_info": serialize(backend_info),
        },
        "aggregate_metrics": {
            "total_objects_final": int(len(final_state.objects)),
            "instance_birth_event_count": int(len(birth_events)),
            "frames_with_new_instances": int(sum(1 for item in frame_metrics if item["new_object_count"] > 0)),
            "frames_with_semantic_updates": int(sum(1 for item in frame_metrics if item["semantic_updated_count"] > 0)),
            "max_total_object_count": int(max(item["total_object_count"] for item in frame_metrics)),
            "mean_raw_proposal_count": float(np.mean([item["raw_proposal_count"] for item in frame_metrics])),
            "mean_refined_proposal_count": float(np.mean([item["refined_proposal_count"] for item in frame_metrics])),
            "mean_active_set_candidate_count": float(np.mean([item["active_set_candidate_count"] for item in frame_metrics])),
        },
        "final_object_state_counts": summarize_state_counts(final_state),
        "object_ids_final": [int(object_id) for object_id in sorted(final_state.objects)],
        "object_history_count": int(len(object_histories)),
    }


def serialize(value: Any) -> Any:
    if is_dataclass(value):
        return serialize(asdict(value))
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize(item) for item in value]
    if isinstance(value, tuple):
        return [serialize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
