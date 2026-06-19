#!/usr/bin/env python3
"""Run the simplified V2 semantic mapping pipeline on Replica room0."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parent

from run_replica_room0_analysis import build_proposal_module
from src.datasets import ReplicaRoom0Dataset
from src.v2.exports import (
    evaluate_and_write_reports,
    export_legacy_outputs,
    export_standard_outputs,
    finalize_geometry_accum,
    render_standard_previews,
)
from src.v2.pipeline import SemanticMapV2Pipeline
from src.utils.visualization import (
    proposal_overlay_image,
    runtime_group_overlay_image,
    runtime_proposal_class_overlay_image,
    save_contact_sheet,
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
        default=project_root / "configs" / "semantic_map_v2.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "outputs",
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--proposal-backend", type=str, default="sam2")
    parser.add_argument("--proposal-device", type=str, default="cuda")
    parser.add_argument("--proposal-cache-dir", type=Path, default=None)
    parser.add_argument("--proposal-cache-manifest", type=Path, default=None)
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
    parser.add_argument("--export-runtime-vis", action="store_true")
    parser.add_argument("--runtime-vis-dir", type=Path, default=None)
    parser.add_argument("--pipeline-verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = args.experiment_name or f"{timestamp}_room0_semantic_map_v2"
    run_root = args.output_root / experiment_name
    scene_dir = run_root / "room0"
    eval_dir = run_root / "replica"
    exports_dir = scene_dir / "exports"
    standard_dir = exports_dir / "v2"
    for directory in [scene_dir, eval_dir, exports_dir, standard_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    runtime_vis_dir: Path | None = None
    raw_overlay_paths: list[Path] = []
    class_overlay_paths: list[Path] = []
    merged_overlay_paths: list[Path] = []
    if args.export_runtime_vis:
        runtime_vis_dir = args.runtime_vis_dir or (scene_dir / "runtime_vis")
        runtime_vis_dir.mkdir(parents=True, exist_ok=True)

    dataset = ReplicaRoom0Dataset(args.dataset_root)
    frame_limit = len(dataset) if args.num_frames <= 0 else min(int(args.num_frames), len(dataset))

    pipeline = SemanticMapV2Pipeline(config_path=args.config_path)
    pipeline.verbose = bool(args.pipeline_verbose)
    pipeline.proposal = build_proposal_module(pipeline.proposal.config, args)

    frame_metrics_path = scene_dir / "frame_metrics.jsonl"
    frame_metrics: list[dict[str, Any]] = []

    print("=" * 60)
    print("Replica room0 semantic map V2 run")
    print("=" * 60)
    print(f"run_root:     {run_root}")
    print(f"scene_dir:    {scene_dir}")
    print(f"frame_limit:  {frame_limit}")
    print(f"backend:      {args.proposal_backend}")
    print(f"device:       {args.proposal_device}")

    with frame_metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for idx, frame in enumerate(dataset.iter_frames(limit=frame_limit), start=1):
            state = pipeline.process_frame(
                frame.rgb,
                frame.depth,
                frame.pose,
                frame.intrinsics,
                timestamp=frame.timestamp,
                source_frame_id=frame.frame_id,
            )
            record = dict(state.last_frame_debug)
            frame_metrics.append(record)
            metrics_handle.write(f"{json_dump(record)}\n")

            if runtime_vis_dir is not None:
                runtime_output = pipeline.last_runtime_vis_output
                if runtime_output is None:
                    raise RuntimeError("V2 pipeline did not expose runtime_vis output.")
                raw_overlay = proposal_overlay_image(frame.rgb, pipeline.last_raw_proposals)
                class_overlay = runtime_proposal_class_overlay_image(
                    frame.rgb,
                    pipeline.last_raw_proposals,
                    runtime_output.proposal_profiles,
                )
                merged_overlay = runtime_group_overlay_image(frame.rgb, runtime_output.groups)

                raw_path = runtime_vis_dir / f"frame{frame.frame_id:06d}_raw_overlay.png"
                class_path = runtime_vis_dir / f"frame{frame.frame_id:06d}_class_overlay.png"
                merged_path = runtime_vis_dir / f"frame{frame.frame_id:06d}_merged_overlay.png"
                raw_overlay.save(raw_path)
                class_overlay.save(class_path)
                merged_overlay.save(merged_path)

                raw_overlay_paths.append(raw_path)
                class_overlay_paths.append(class_path)
                merged_overlay_paths.append(merged_path)

            if idx == 1 or idx % 20 == 0 or idx == frame_limit:
                print(
                    f"frame={frame.frame_id:04d} processed={idx:04d}/{frame_limit:04d} "
                    f"objects={len(state.object_pools):03d} candidates={record['candidate_patch_count']:03d} "
                    f"matched={record['matched_patch_count']:03d}"
                )

    dense_points, dense_rgb = finalize_geometry_accum(pipeline.state.geometry_accum)
    coarse_records, room_projection, room_pool_records, pool_debug_records = export_standard_outputs(
        state=pipeline.state,
        exports_dir=standard_dir,
        coarse_voxel_size=pipeline.coarse_voxel_size,
        neighbor_radius=pipeline.projection_neighbor_radius,
        dense_points=dense_points,
        dense_rgb=dense_rgb,
    )
    export_legacy_outputs(
        exports_dir=exports_dir,
        coarse_records=coarse_records,
        room_pool_records=room_pool_records,
        room_projection=room_projection,
        pool_debug_records=pool_debug_records,
    )
    render_standard_previews(
        exports_dir=exports_dir,
        dense_points=dense_points,
        dense_rgb=dense_rgb,
        room_projection=room_projection,
    )
    evaluate_and_write_reports(
        eval_dir=eval_dir,
        scene_dir=scene_dir,
        experiment_name=experiment_name,
        frame_limit=frame_limit,
        dense_points=dense_points,
        room_projection=room_projection,
        coarse_records=coarse_records,
        room_pool_records=room_pool_records,
        pool_debug_records=pool_debug_records,
        state=pipeline.state,
        frame_metrics=frame_metrics,
        gt_labels_path=args.gt_labels,
        gt_mesh_path=args.gt_mesh_ply,
        gt_info_path=args.gt_info_json,
    )

    if runtime_vis_dir is not None and raw_overlay_paths:
        raw_contact = runtime_vis_dir / "contact_sheet_raw.png"
        class_contact = runtime_vis_dir / "contact_sheet_class.png"
        merged_contact = runtime_vis_dir / "contact_sheet_merged.png"
        save_contact_sheet(raw_overlay_paths, raw_contact, columns=4, tile_width=320)
        save_contact_sheet(class_overlay_paths, class_contact, columns=4, tile_width=320)
        save_contact_sheet(merged_overlay_paths, merged_contact, columns=4, tile_width=320)

        manifest = {
            "frame_count": len(raw_overlay_paths),
            "runtime_vis_dir": str(runtime_vis_dir),
            "raw_contact_sheet": str(raw_contact),
            "class_contact_sheet": str(class_contact),
            "merged_contact_sheet": str(merged_contact),
            "frames": [
                {
                    "frame_id": int(path.stem.replace("frame", "").replace("_raw_overlay", "")),
                    "raw_overlay_path": str(path),
                    "class_overlay_path": str(class_overlay_paths[idx]),
                    "merged_overlay_path": str(merged_overlay_paths[idx]),
                }
                for idx, path in enumerate(raw_overlay_paths)
            ],
        }
        (runtime_vis_dir / "runtime_vis_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    print("\nSaved outputs:")
    print(f"- report:                {scene_dir / 'run_report.md'}")
    print(f"- eval results:          {eval_dir / 'results.json'}")
    print(f"- coarse semantic map:   {standard_dir / 'room0_coarse_voxel_semantic_map.ply'}")
    print(f"- room semantic map:     {standard_dir / 'room0_room_semantic_object_map.ply'}")
    print(f"- object pool debug:     {standard_dir / 'object_pool_debug'}")
    if runtime_vis_dir is not None and raw_overlay_paths:
        print(f"- runtime vis overlays:  {runtime_vis_dir}")


def json_dump(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    main()
