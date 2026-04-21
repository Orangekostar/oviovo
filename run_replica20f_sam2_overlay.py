#!/usr/bin/env python3
"""Run SAM2 frontend proposals on Replica room0 and save overlay visualizations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.datasets import ReplicaRoom0Dataset
from src.modules.proposal import ProposalModule
from src.utils.visualization import proposal_overlay_image, save_contact_sheet


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
        default=project_root / "outputs" / "replica20fTest",
    )
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sam-version", type=str, default="2.1")
    parser.add_argument("--sam-encoder", type=str, default="hiera_l")
    parser.add_argument("--sam-ckpt-path", type=Path, default=project_root / "data" / "input" / "sam_ckpts")
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument("--stability-score-th", type=float, default=0.95)
    parser.add_argument("--nms-iou-th", type=float, default=0.8)
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    parser.add_argument("--use-m2m", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = ReplicaRoom0Dataset(args.dataset_root)
    proposal_module = ProposalModule(
        {
            "backend": "sam2",
            "device": args.device,
            "sam_version": args.sam_version,
            "sam_encoder": args.sam_encoder,
            "sam_ckpt_path": str(args.sam_ckpt_path),
            "points_per_side": args.points_per_side,
            "max_proposals": args.max_proposals,
            "confidence_threshold": args.confidence_threshold,
            "min_mask_area": args.min_mask_area,
            "stability_score_th": args.stability_score_th,
            "nms_iou_th": args.nms_iou_th,
            "min_mask_region_area": args.min_mask_region_area,
            "use_m2m": args.use_m2m,
        }
    )
    proposal_module.backend_config = proposal_module._resolve_backend_config(proposal_module.config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_paths: list[Path] = []
    frame_stats: list[dict[str, object]] = []

    print("=" * 60)
    print("Replica room0 SAM2 frontend overlay run")
    print("=" * 60)
    print(f"dataset_root: {args.dataset_root}")
    print(f"output_dir:   {args.output_dir}")
    print(f"num_frames:   {args.num_frames}")
    print(f"device:       {args.device}")
    print(f"cuda_visible: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    for frame in dataset.iter_frames(limit=args.num_frames):
        proposals = proposal_module.process(frame.rgb, frame.depth, frame=frame)
        overlay = proposal_overlay_image(frame.rgb, proposals)
        overlay_path = args.output_dir / f"frame{frame.frame_id:06d}_overlay.png"
        overlay.save(overlay_path)
        overlay_paths.append(overlay_path)

        info = getattr(proposal_module.backend, "last_generation_info", {})
        frame_stats.append(
            {
                "frame_id": frame.frame_id,
                "proposal_count": len(proposals),
                "raw_mask_count": info.get("raw_mask_count"),
                "overlay_path": str(overlay_path),
                "top_confidences": [round(float(p.confidence), 4) for p in proposals[:10]],
            }
        )
        print(
            f"frame={frame.frame_id:04d} proposals={len(proposals):03d} "
            f"raw_masks={info.get('raw_mask_count', 'n/a')}"
        )

    contact_sheet_path = args.output_dir / "contact_sheet.png"
    save_contact_sheet(overlay_paths, contact_sheet_path, columns=4, tile_width=320)

    manifest = {
        "dataset_summary": dataset.summary(),
        "sam2_backend": getattr(proposal_module.backend, "last_generation_info", {}),
        "run": {
            "num_frames": args.num_frames,
            "device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "output_dir": str(args.output_dir),
            "contact_sheet": str(contact_sheet_path),
        },
        "frames": frame_stats,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"- contact sheet: {contact_sheet_path}")
    print(f"- manifest:      {manifest_path}")
    print(f"- overlays:      {len(overlay_paths)} files")


if __name__ == "__main__":
    main()
