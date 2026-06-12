#!/usr/bin/env python3
"""Parallel SAM2 proposal precompute for OVIOVO frontend caches."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from frontend.proposal_cache import frame_path, save_proposals, write_manifest
from src.datasets import ReplicaRoom0Dataset
from src.modules.proposal import ProposalModule


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
        default=project_root / "outputs" / "frontend_proposals" / "replica_room0_sam2_precomputed",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="0 means the full dataset sequence.",
    )
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--sam2-version", type=str, default="2.1")
    parser.add_argument("--sam2-encoder", type=str, default="hiera_l")
    parser.add_argument("--sam-repo-root", type=Path, default=Path("/home/phl/vv/paper2/OVO/thirdParty/segment-anything-2"))
    parser.add_argument(
        "--sam-ckpt-path",
        type=Path,
        default=project_root / "data" / "input" / "sam_ckpts",
    )
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=0,
        help="Area filter applied before cache write. Default 0 keeps all normalized SAM2 proposals.",
    )
    parser.add_argument("--stability-score-th", type=float, default=0.95)
    parser.add_argument("--nms-iou-th", type=float, default=0.8)
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--worker-device", type=str, default="")
    return parser.parse_args()


def build_backend_config(args: argparse.Namespace, *, device: str) -> dict[str, Any]:
    return {
        "backend": "sam2",
        "min_mask_area": int(args.min_mask_area),
        "max_proposals": int(args.max_proposals),
        "confidence_threshold": float(args.confidence_threshold),
        "sam2": {
            "device": device,
            "sam_version": str(args.sam2_version),
            "sam_encoder": str(args.sam2_encoder),
            "sam_repo_root": str(args.sam_repo_root) if args.sam_repo_root is not None else "",
            "sam_ckpt_path": str(args.sam_ckpt_path),
            "points_per_side": int(args.points_per_side),
            "max_proposals": int(args.max_proposals),
            "confidence_threshold": float(args.confidence_threshold),
            "stability_score_th": float(args.stability_score_th),
            "nms_iou_th": float(args.nms_iou_th),
            "min_mask_region_area": int(args.min_mask_region_area),
        },
    }


def launch_workers(args: argparse.Namespace) -> int:
    devices = [entry.strip() for entry in args.devices.split(",") if entry.strip()]
    if not devices:
        raise ValueError("No worker devices specified.")

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    script_path = Path(__file__).resolve()
    for worker_id, device_spec in enumerate(devices):
        env = os.environ.copy()
        worker_device = device_spec
        if device_spec.startswith("cuda"):
            cuda_slot = device_spec.split(":")[-1] if ":" in device_spec else device_spec
            env["CUDA_VISIBLE_DEVICES"] = cuda_slot
            worker_device = "cuda"
        cmd = [
            sys.executable,
            str(script_path),
            "--dataset-root",
            str(args.dataset_root),
            "--output-dir",
            str(args.output_dir),
            "--num-frames",
            str(args.num_frames),
            "--sam2-version",
            str(args.sam2_version),
            "--sam2-encoder",
            str(args.sam2_encoder),
            "--sam-repo-root",
            str(args.sam_repo_root),
            "--sam-ckpt-path",
            str(args.sam_ckpt_path),
            "--points-per-side",
            str(args.points_per_side),
            "--max-proposals",
            str(args.max_proposals),
            "--confidence-threshold",
            str(args.confidence_threshold),
            "--min-mask-area",
            str(args.min_mask_area),
            "--stability-score-th",
            str(args.stability_score_th),
            "--nms-iou-th",
            str(args.nms_iou_th),
            "--min-mask-region-area",
            str(args.min_mask_region_area),
            "--worker-id",
            str(worker_id),
            "--num-workers",
            str(len(devices)),
            "--worker-device",
            worker_device,
        ]
        procs.append(subprocess.Popen(cmd, cwd=project_root, env=env))

    exit_code = 0
    for proc in procs:
        ret = proc.wait()
        if ret != 0 and exit_code == 0:
            exit_code = ret
    return exit_code


def run_worker(args: argparse.Namespace) -> None:
    if args.worker_id is None or args.num_workers is None:
        raise ValueError("Worker mode requires --worker-id and --num-workers.")

    dataset = ReplicaRoom0Dataset(args.dataset_root)
    frame_limit = len(dataset) if args.num_frames <= 0 else min(int(args.num_frames), len(dataset))
    proposal_module = ProposalModule(build_backend_config(args, device=args.worker_device or "cuda"))

    assigned_count = 0
    for slot, frame in enumerate(dataset.iter_frames(limit=frame_limit)):
        if slot % int(args.num_workers) != int(args.worker_id):
            continue
        cache_file = frame_path(args.output_dir, int(frame.frame_id))
        if cache_file.exists():
            continue

        proposals = proposal_module.process(frame.rgb, frame.depth, frame=frame)
        generation_info = dict(getattr(proposal_module.backend, "last_generation_info", {}))
        generation_info.pop("raw_proposals", None)
        save_proposals(
            args.output_dir,
            frame_id=int(frame.frame_id),
            image_shape=tuple(frame.rgb.shape[:2]),
            proposals=proposals,
            source_backend=str(proposal_module.active_backend_name),
            generation_info=generation_info,
        )
        assigned_count += 1
        print(
            f"[worker {args.worker_id}/{args.num_workers}] frame={frame.frame_id:04d} "
            f"saved={len(proposals):03d} cache={cache_file}"
        )

    print(f"[worker {args.worker_id}/{args.num_workers}] completed {assigned_count} assigned frames")


def finalize_manifest(args: argparse.Namespace) -> Path:
    dataset = ReplicaRoom0Dataset(args.dataset_root)
    frame_limit = len(dataset) if args.num_frames <= 0 else min(int(args.num_frames), len(dataset))
    frame_entries = []
    for frame in dataset.iter_frames(limit=frame_limit):
        cache_file = frame_path(args.output_dir, int(frame.frame_id))
        if not cache_file.exists():
            continue
        with np.load(cache_file, allow_pickle=False) as payload:
            generation_info_json = (
                str(payload["generation_info_json"][0]) if payload["generation_info_json"].size else "{}"
            )
            frame_entries.append(
                {
                    "frame_id": int(frame.frame_id),
                    "file": str(cache_file.relative_to(args.output_dir)),
                    "proposal_count": int(payload["proposal_ids"].shape[0]),
                    "image_shape": [int(value) for value in payload["image_shape"].tolist()],
                    "source_backend": str(payload["source_backend"][0]) if payload["source_backend"].size else "unknown",
                    "generation_info": json.loads(generation_info_json),
                }
            )
    frame_entries.sort(key=lambda item: int(item["frame_id"]))

    notes = {
        "num_frames_requested": int(args.num_frames),
        "devices": [entry.strip() for entry in args.devices.split(",") if entry.strip()],
        "cache_layout": {
            "manifest": "manifest.json",
            "frames": "frames/frameXXXXXX_proposals.npz",
        },
        "usage_hint": (
            "Use with --proposal-backend precomputed "
            f"--proposal-cache-dir {args.output_dir}"
        ),
    }
    return write_manifest(
        args.output_dir,
        dataset_summary=dataset.summary(),
        backend_config=build_backend_config(args, device="cuda"),
        frame_entries=frame_entries,
        notes=notes,
    )


def main() -> None:
    args = parse_args()
    if args.worker_id is None:
        exit_code = launch_workers(args)
        if exit_code != 0:
            raise SystemExit(exit_code)
        manifest_path = finalize_manifest(args)
        print(f"Saved proposal cache manifest: {manifest_path}")
        return

    run_worker(args)


if __name__ == "__main__":
    main()
