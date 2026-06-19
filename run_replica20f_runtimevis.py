#!/usr/bin/env python3
"""Run runtime_vis on Replica room0 and save debug visualizations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.datasets import ReplicaRoom0Dataset
from src.pipelines.main_pipeline import Pipeline
from src.utils.visualization import (
    proposal_overlay_image,
    runtime_proposal_class_overlay_image,
    runtime_group_overlay_image,
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
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "runtimevis_test",
    )
    parser.add_argument("--config-path", type=Path, default=project_root / "configs" / "default.yaml")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--proposal-backend", type=str, default="sam2")
    parser.add_argument("--proposal-device", type=str, default="cuda")
    parser.add_argument("--sam-version", type=str, default="2.1")
    parser.add_argument("--sam-encoder", type=str, default="hiera_l")
    parser.add_argument("--sam-repo-root", type=Path, default=None)
    parser.add_argument("--sam-ckpt-path", type=Path, default=project_root / "data" / "input" / "sam_ckpts")
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument("--stability-score-th", type=float, default=0.95)
    parser.add_argument("--nms-iou-th", type=float, default=0.8)
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    parser.add_argument("--use-m2m", action="store_true")
    parser.add_argument("--side-by-side-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = ReplicaRoom0Dataset(args.dataset_root)
    pipeline = Pipeline(config_path=str(args.config_path))
    pipeline.verbose = False

    pipeline.proposal.config.update(
        {
            "backend": args.proposal_backend,
            "min_mask_area": args.min_mask_area,
        }
    )
    backend_cfg = pipeline.proposal.config.setdefault(args.proposal_backend, {})
    backend_cfg.update(
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
            "use_m2m": args.use_m2m,
        }
    )
    if args.proposal_backend == "entitysam":
        entitysam_cfg = pipeline.proposal.config.setdefault("entitysam", {})
        if not entitysam_cfg.get("sequence_dir"):
            entitysam_cfg["sequence_dir"] = str(
                prepare_entitysam_sequence_dir(
                    source_dir=dataset.rgb_dir,
                    output_root=args.output_dir,
                    num_frames=args.num_frames,
                )
            )
    pipeline.proposal.backend = pipeline.proposal._create_backend(pipeline.proposal.config)
    pipeline.proposal.backend_config = pipeline.proposal._resolve_backend_config(
        pipeline.proposal.config,
        backend_name=args.proposal_backend,
    )
    pipeline.proposal.backend.initialize(pipeline.proposal.backend_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pre_dir = args.output_dir / "precompute_vis"
    runtime_dir = args.output_dir / "runtime_vis"
    pre_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    raw_overlay_paths: list[Path] = []
    class_overlay_paths: list[Path] = []
    merged_overlay_paths: list[Path] = []
    side_by_side_paths: list[Path] = []
    frame_manifest: list[dict[str, Any]] = []

    print("=" * 60)
    print("Replica room0 runtime_vis run")
    print("=" * 60)
    print(f"dataset_root: {args.dataset_root}")
    print(f"output_dir:   {args.output_dir}")
    print(f"num_frames:   {args.num_frames}")
    print(f"device:       {args.proposal_device}")
    print(f"cuda_visible: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    for frame in dataset.iter_frames(limit=args.num_frames):
        state = pipeline.process_frame(
            frame.rgb,
            frame.depth,
            frame.pose,
            frame.intrinsics,
            timestamp=frame.timestamp,
            source_frame_id=frame.frame_id,
        )
        raw_proposals = pipeline.last_raw_proposals
        raw_precompute_proposals = (
            getattr(pipeline.proposal.backend, "last_generation_info", {}).get("raw_proposals")
            or raw_proposals
        )
        runtime_output = pipeline.last_runtime_vis_output
        if runtime_output is None:
            raise RuntimeError("Pipeline did not produce runtime_vis output.")

        raw_overlay = proposal_overlay_image(frame.rgb, raw_precompute_proposals)
        class_overlay = runtime_proposal_class_overlay_image(
            frame.rgb,
            raw_proposals,
            runtime_output.proposal_profiles,
        )
        merged_overlay = runtime_group_overlay_image(frame.rgb, runtime_output.groups)

        raw_path = pre_dir / f"frame{frame.frame_id:06d}_raw_overlay.png"
        class_path = runtime_dir / f"frame{frame.frame_id:06d}_class_overlay.png"
        merged_path = runtime_dir / f"frame{frame.frame_id:06d}_merged_overlay.png"
        side_path = runtime_dir / f"frame{frame.frame_id:06d}_side_by_side.png"
        json_path = runtime_dir / f"frame{frame.frame_id:06d}_groups.json"

        save_three_panel(raw_overlay, class_overlay, merged_overlay, side_path)
        side_by_side_paths.append(side_path)
        if not args.side_by_side_only:
            raw_overlay.save(raw_path)
            class_overlay.save(class_path)
            merged_overlay.save(merged_path)
            raw_overlay_paths.append(raw_path)
            class_overlay_paths.append(class_path)
            merged_overlay_paths.append(merged_path)

        frame_payload = {
            "frame_id": frame.frame_id,
            "raw_mask_count": len(raw_precompute_proposals),
            "filtered_proposal_count": len(raw_proposals),
            "merged_group_count": len(runtime_output.groups),
            "accepted_edge_count": runtime_output.group_stats.get("accepted_edge_count", 0),
            "whole_prior_boosted_edge_count": runtime_output.group_stats.get(
                "whole_prior_boosted_edge_count",
                0,
            ),
            "proposal_profiles": [serialize(profile) for profile in runtime_output.proposal_profiles],
            "group_to_linked_object": runtime_output.group_to_linked_object,
            "raw_to_group": runtime_output.raw_to_group,
            "groups": [serialize(group, include_masks=False) for group in runtime_output.groups],
            "merge_decisions": [serialize(decision) for decision in runtime_output.merge_decisions],
            "runtime_vis_stats": runtime_output.group_stats,
            "raw_overlay_path": str(raw_path),
            "class_overlay_path": str(class_path),
            "merged_overlay_path": str(merged_path),
            "side_by_side_path": str(side_path),
            "state_object_count": len(state.objects),
        }
        if not args.side_by_side_only:
            json_path.write_text(json.dumps(frame_payload, indent=2), encoding="utf-8")
        frame_manifest.append(frame_payload)

        print(
            f"frame={frame.frame_id:04d} raw={len(raw_precompute_proposals):03d} "
            f"filtered={len(raw_proposals):03d} "
            f"merged={len(runtime_output.groups):03d} "
            f"accepted_edges={runtime_output.group_stats.get('accepted_edge_count', 0):03d} "
            f"whole_prior_edges={runtime_output.group_stats.get('whole_prior_boosted_edge_count', 0):03d}"
        )

    raw_contact = pre_dir / "contact_sheet_raw.png"
    class_contact = runtime_dir / "contact_sheet_class.png"
    merged_contact = runtime_dir / "contact_sheet_merged.png"
    side_contact = runtime_dir / "contact_sheet_side_by_side.png"
    if not args.side_by_side_only:
        save_contact_sheet(raw_overlay_paths, raw_contact, columns=4, tile_width=320)
        save_contact_sheet(class_overlay_paths, class_contact, columns=4, tile_width=320)
        save_contact_sheet(merged_overlay_paths, merged_contact, columns=4, tile_width=320)
    save_contact_sheet(side_by_side_paths, side_contact, columns=4, tile_width=320)

    manifest = {
        "dataset_summary": dataset.summary(),
        "proposal_backend": {
            "backend": args.proposal_backend,
            "device": args.proposal_device,
            "sam_version": args.sam_version,
            "sam_encoder": args.sam_encoder,
            "sam_ckpt_path": str(args.sam_ckpt_path),
            "points_per_side": args.points_per_side,
            "max_proposals": args.max_proposals,
            "confidence_threshold": args.confidence_threshold,
        },
        "runtime_vis_config": pipeline.runtime_vis.config,
        "run": {
            "num_frames": args.num_frames,
                "output_dir": str(args.output_dir),
                "precompute_vis_dir": str(pre_dir),
                "runtime_vis_dir": str(runtime_dir),
                "raw_contact_sheet": str(raw_contact) if not args.side_by_side_only else None,
                "class_contact_sheet": str(class_contact) if not args.side_by_side_only else None,
                "merged_contact_sheet": str(merged_contact) if not args.side_by_side_only else None,
                "side_by_side_contact_sheet": str(side_contact),
                "side_by_side_only": bool(args.side_by_side_only),
            },
            "frames": [
                {
                    "frame_id": item["frame_id"],
                "raw_mask_count": item["raw_mask_count"],
                "merged_group_count": item["merged_group_count"],
                "accepted_edge_count": item["accepted_edge_count"],
                "whole_prior_boosted_edge_count": item["whole_prior_boosted_edge_count"],
                "raw_overlay_path": item["raw_overlay_path"],
                    "class_overlay_path": item["class_overlay_path"],
                    "merged_overlay_path": item["merged_overlay_path"],
                    "side_by_side_path": item["side_by_side_path"],
                    "groups_json": (
                        str(runtime_dir / f"frame{item['frame_id']:06d}_groups.json")
                        if not args.side_by_side_only
                        else None
                    ),
                }
                for item in frame_manifest
            ],
        }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSaved outputs:")
    if not args.side_by_side_only:
        print(f"- raw contact sheet:    {raw_contact}")
        print(f"- class contact sheet:  {class_contact}")
        print(f"- merged contact sheet: {merged_contact}")
    print(f"- side-by-side sheet:   {side_contact}")
    print(f"- manifest:             {manifest_path}")
    print(f"- frame jsons:          {len(frame_manifest)}")


def save_three_panel(
    left: Image.Image,
    center: Image.Image,
    right: Image.Image,
    output_path: Path,
) -> None:
    """Save three images as a horizontal comparison panel."""
    left_rgb = left.convert("RGB")
    center_rgb = center.convert("RGB")
    right_rgb = right.convert("RGB")
    canvas = Image.new(
        "RGB",
        (left_rgb.width + center_rgb.width + right_rgb.width, max(left_rgb.height, center_rgb.height, right_rgb.height)),
    )
    canvas.paste(left_rgb, (0, 0))
    canvas.paste(center_rgb, (left_rgb.width, 0))
    canvas.paste(right_rgb, (left_rgb.width + center_rgb.width, 0))
    canvas.save(output_path)


def prepare_entitysam_sequence_dir(source_dir: Path, output_root: Path, num_frames: int) -> Path:
    """Create a lightweight frame subset directory for EntitySAM video inference."""
    subset_dir = output_root / "_entitysam_sequence"
    subset_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = [
        path
        for path in sorted(source_dir.iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ][:num_frames]
    if not frame_paths:
        raise RuntimeError(f"No RGB frames found in {source_dir} for EntitySAM sequence setup.")

    for existing in subset_dir.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()
        elif existing.is_dir():
            shutil.rmtree(existing)

    for frame_path in frame_paths:
        target = subset_dir / frame_path.name
        os.symlink(frame_path, target)
    return subset_dir


def serialize(value: Any, include_masks: bool = True) -> Any:
    """Convert dataclasses and numpy-heavy structures to JSON-safe objects."""
    if is_dataclass(value):
        payload = asdict(value)
        if not include_masks:
            payload.pop("merged_mask", None)
        return serialize(payload, include_masks=include_masks)
    if isinstance(value, dict):
        return {str(key): serialize(item, include_masks=include_masks) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize(item, include_masks=include_masks) for item in value]
    if isinstance(value, tuple):
        return [serialize(item, include_masks=include_masks) for item in value]
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
