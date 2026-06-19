#!/usr/bin/env python3
"""Compare SAM2, E-SAM, and EntitySAM proposal backends on the same Replica frames."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

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
        default=project_root / "outputs" / "replica20f_compare",
    )
    parser.add_argument("--num-frames", type=int, default=20)

    parser.add_argument("--sam2-device", type=str, default="cuda")
    parser.add_argument("--sam2-version", type=str, default="2.1")
    parser.add_argument("--sam2-encoder", type=str, default="hiera_l")
    parser.add_argument("--sam2-ckpt-path", type=Path, default=project_root / "data" / "input" / "sam_ckpts")
    parser.add_argument("--sam2-points-per-side", type=int, default=16)
    parser.add_argument("--sam2-max-proposals", type=int, default=64)
    parser.add_argument("--sam2-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--sam2-min-mask-area", type=int, default=100)
    parser.add_argument("--sam2-stability-score-th", type=float, default=0.95)
    parser.add_argument("--sam2-nms-iou-th", type=float, default=0.8)
    parser.add_argument("--sam2-min-mask-region-area", type=int, default=100)
    parser.add_argument("--sam2-use-m2m", action="store_true")

    parser.add_argument("--esam-device", type=str, default="cuda")
    parser.add_argument("--esam-repo-root", type=Path, default=Path("/home/phl/vv/E-SAM"))
    parser.add_argument("--esam-python-path", action="append", default=[])
    parser.add_argument("--esam-factory", type=str, default="")
    parser.add_argument("--esam-predictor", type=str, default="")
    parser.add_argument("--esam-predictor-is-factory", action="store_true")
    parser.add_argument("--esam-call-mode", type=str, default="auto")
    parser.add_argument("--esam-bbox-format", type=str, default="auto")
    parser.add_argument("--esam-config-path", type=Path, default=None)
    parser.add_argument("--esam-checkpoint-path", type=Path, default=None)
    parser.add_argument("--esam-max-proposals", type=int, default=64)
    parser.add_argument("--esam-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--esam-min-mask-area", type=int, default=100)

    parser.add_argument("--entitysam-device", type=str, default="cuda")
    parser.add_argument("--entitysam-repo-root", type=Path, default=Path("/home/phl/vv/paper2/entitysam"))
    parser.add_argument("--entitysam-config-path", type=str, default="configs/sam2.1_hiera_l.yaml")
    parser.add_argument("--entitysam-checkpoint-path", type=str, default="checkpoints/vit-l/model_0009999.pth")
    parser.add_argument("--entitysam-mask-decoder-depth", type=int, default=8)
    parser.add_argument("--entitysam-points-per-side", type=int, default=16)
    parser.add_argument("--entitysam-max-proposals", type=int, default=64)
    parser.add_argument("--entitysam-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--entitysam-min-mask-area", type=int, default=100)
    parser.add_argument("--entitysam-stability-score-th", type=float, default=0.95)
    parser.add_argument("--entitysam-nms-iou-th", type=float, default=0.8)
    parser.add_argument("--entitysam-min-mask-region-area", type=int, default=100)
    parser.add_argument("--entitysam-mask-binary-threshold", type=float, default=0.5)
    parser.add_argument("--entitysam-object-mask-threshold", type=float, default=0.05)
    parser.add_argument("--entitysam-topk-per-frame", type=int, default=100)
    parser.add_argument("--entitysam-use-m2m", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ReplicaRoom0Dataset(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Replica room0 proposal backend comparison")
    print("=" * 60)
    print(f"dataset_root: {args.dataset_root}")
    print(f"output_dir:   {args.output_dir}")
    print(f"num_frames:   {args.num_frames}")
    print(f"cuda_visible: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    backend_runs = {
        "sam2": build_sam2_config(args),
        "esam": build_esam_config(args),
        "entitysam": build_entitysam_config(args),
    }

    summary: Dict[str, Any] = {
        "dataset_summary": dataset.summary(),
        "run": {
            "num_frames": args.num_frames,
            "output_dir": str(args.output_dir),
        },
        "backends": {},
    }

    for backend_name, backend_config in backend_runs.items():
        backend_output_dir = args.output_dir / backend_name
        backend_output_dir.mkdir(parents=True, exist_ok=True)
        summary["backends"][backend_name] = run_backend(
            backend_name=backend_name,
            backend_config=backend_config,
            dataset=dataset,
            num_frames=args.num_frames,
            output_dir=backend_output_dir,
        )

    top_manifest_path = args.output_dir / "manifest.json"
    top_manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved comparison manifest:")
    print(f"- {top_manifest_path}")


def run_backend(
    backend_name: str,
    backend_config: Dict[str, Any],
    dataset: ReplicaRoom0Dataset,
    num_frames: int,
    output_dir: Path,
) -> Dict[str, Any]:
    print(f"\n[{backend_name}] starting")
    output_dir.mkdir(parents=True, exist_ok=True)
    if backend_name == "entitysam":
        backend_config = json.loads(json.dumps(stringify_paths(backend_config)))
        backend_config.setdefault("entitysam", {})
        sequence_dir = getattr(dataset, "rgb_dir", None)
        if sequence_dir is not None and not backend_config["entitysam"].get("sequence_dir"):
            backend_config["entitysam"]["sequence_dir"] = str(
                prepare_entitysam_sequence_dir(
                    source_dir=Path(sequence_dir),
                    output_root=output_dir,
                    num_frames=num_frames,
                )
            )
    try:
        proposal_module = ProposalModule(backend_config)
    except Exception as exc:
        error_manifest = {
            "backend": backend_name,
            "status": "error",
            "error": str(exc),
            "config": stringify_paths(backend_config),
        }
        (output_dir / "manifest.json").write_text(json.dumps(error_manifest, indent=2), encoding="utf-8")
        print(f"[{backend_name}] initialization failed: {exc}")
        return error_manifest

    overlay_paths: list[Path] = []
    frame_stats: list[dict[str, Any]] = []

    try:
        for frame in dataset.iter_frames(limit=num_frames):
            proposals = proposal_module.process(frame.rgb, frame.depth, frame=frame)
            raw_precompute_proposals = (
                getattr(proposal_module.backend, "last_generation_info", {}).get("raw_proposals")
                or proposals
            )
            overlay = proposal_overlay_image(frame.rgb, raw_precompute_proposals)
            overlay_path = output_dir / f"frame{frame.frame_id:06d}_overlay.png"
            overlay.save(overlay_path)
            overlay_paths.append(overlay_path)

            info = getattr(proposal_module.backend, "last_generation_info", {})
            frame_stats.append(
                {
                    "frame_id": frame.frame_id,
                    "frontend": backend_name,
                    "proposal_count": len(proposals),
                    "raw_mask_count": len(raw_precompute_proposals),
                    "avg_mask_area": round(
                        sum(float(proposal.area) for proposal in raw_precompute_proposals)
                        / max(len(raw_precompute_proposals), 1),
                        2,
                    ),
                    "overlay_path": str(overlay_path),
                    "backend_names": sorted({proposal.backend_name for proposal in raw_precompute_proposals}),
                }
            )
            print(
                f"[{backend_name}] frame={frame.frame_id:04d} "
                f"raw={len(raw_precompute_proposals):03d} filtered={len(proposals):03d}"
            )
    except Exception as exc:
        error_manifest = {
            "backend": backend_name,
            "status": "error",
            "error": str(exc),
            "config": stringify_paths(backend_config),
            "frames_completed": len(frame_stats),
        }
        (output_dir / "manifest.json").write_text(json.dumps(error_manifest, indent=2), encoding="utf-8")
        print(f"[{backend_name}] runtime failed: {exc}")
        return error_manifest

    contact_sheet_path = output_dir / "contact_sheet.png"
    save_contact_sheet(overlay_paths, contact_sheet_path, columns=4, tile_width=320)
    avg_count = round(sum(item["proposal_count"] for item in frame_stats) / max(len(frame_stats), 1), 2)
    avg_area = round(sum(item["avg_mask_area"] for item in frame_stats) / max(len(frame_stats), 1), 2)
    manifest = {
        "backend": backend_name,
        "status": "ok",
        "config": stringify_paths(backend_config),
        "backend_info": getattr(proposal_module.backend, "last_generation_info", {}),
        "summary": {
            "num_frames": len(frame_stats),
            "average_proposal_count": avg_count,
            "average_mask_area": avg_area,
        },
        "contact_sheet": str(contact_sheet_path),
        "frames": frame_stats,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_sam2_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "backend": "sam2",
        "min_mask_area": args.sam2_min_mask_area,
        "max_proposals": args.sam2_max_proposals,
        "confidence_threshold": args.sam2_confidence_threshold,
        "sam2": {
            "device": args.sam2_device,
            "sam_version": args.sam2_version,
            "sam_encoder": args.sam2_encoder,
            "sam_ckpt_path": str(args.sam2_ckpt_path),
            "points_per_side": args.sam2_points_per_side,
            "max_proposals": args.sam2_max_proposals,
            "confidence_threshold": args.sam2_confidence_threshold,
            "stability_score_th": args.sam2_stability_score_th,
            "nms_iou_th": args.sam2_nms_iou_th,
            "min_mask_region_area": args.sam2_min_mask_region_area,
            "use_m2m": args.sam2_use_m2m,
        },
    }


def build_esam_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "backend": "esam",
        "min_mask_area": args.esam_min_mask_area,
        "max_proposals": args.esam_max_proposals,
        "confidence_threshold": args.esam_confidence_threshold,
        "esam": {
            "device": args.esam_device,
            "repo_root": str(args.esam_repo_root),
            "python_paths": list(args.esam_python_path),
            "factory": args.esam_factory,
            "predictor": args.esam_predictor,
            "predictor_is_factory": args.esam_predictor_is_factory,
            "call_mode": args.esam_call_mode,
            "bbox_format": args.esam_bbox_format,
            "config_path": str(args.esam_config_path) if args.esam_config_path else "",
            "checkpoint_path": str(args.esam_checkpoint_path) if args.esam_checkpoint_path else "",
            "max_proposals": args.esam_max_proposals,
            "confidence_threshold": args.esam_confidence_threshold,
        },
    }


def build_entitysam_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "backend": "entitysam",
        "min_mask_area": args.entitysam_min_mask_area,
        "max_proposals": args.entitysam_max_proposals,
        "confidence_threshold": args.entitysam_confidence_threshold,
        "entitysam": {
            "device": args.entitysam_device,
            "repo_root": str(args.entitysam_repo_root),
            "sequence_dir": "",
            "config_path": args.entitysam_config_path,
            "checkpoint_path": args.entitysam_checkpoint_path,
            "mask_decoder_depth": args.entitysam_mask_decoder_depth,
            "points_per_side": args.entitysam_points_per_side,
            "max_proposals": args.entitysam_max_proposals,
            "confidence_threshold": args.entitysam_confidence_threshold,
            "stability_score_th": args.entitysam_stability_score_th,
            "nms_iou_th": args.entitysam_nms_iou_th,
            "min_mask_region_area": args.entitysam_min_mask_region_area,
            "mask_binary_threshold": args.entitysam_mask_binary_threshold,
            "object_mask_threshold": args.entitysam_object_mask_threshold,
            "topk_per_frame": args.entitysam_topk_per_frame,
            "use_m2m": args.entitysam_use_m2m,
        },
    }


def stringify_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stringify_paths(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def prepare_entitysam_sequence_dir(source_dir: Path, output_root: Path, num_frames: int) -> Path:
    """Create a frame subset directory for EntitySAM sequence-based inference."""
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


if __name__ == "__main__":
    main()
