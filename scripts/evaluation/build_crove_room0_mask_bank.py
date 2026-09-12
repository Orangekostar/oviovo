"""Bind actual CropFormer observations to frozen patch representatives, without GT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.datasets.replica import ReplicaRoom0Dataset
from src.oviv2.surface_mask_evidence import interior_labels
from src.oviv2.surface_view_bank import project_source_support


def file_hash(filename):
    with filename.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    case = json.loads(path(config["source_config"]).read_text())["cases"][
        "replica_room0_static"
    ]
    room = path(config["run_root"]) / "dev/room0"
    output = room / "independent_mask_bank"
    output.mkdir(exist_ok=True)
    frontend = room / "cropformer_cpu/frontend"
    patch_file = room / "graph_s2/patch_mapping.npz"
    settings = {
        "case": "room0",
        "frames": list(range(0, 2000, 10)),
        "patch_sha256": file_hash(patch_file),
        "representative": "actual source point nearest fixed patch center; ties by source row order",
        "source": "official CropFormer full-resolution pretrained inference, CPU device; independent of owner projection",
        "projection": "same first-frame-normalized pose and positive measured depth within 0.05 m",
        "boundary": "one-pixel 8-neighbor erosion, image boundary excluded",
        "unknown_or_occluded": "no usable observer; never a negative vote",
        "graph_boundary": {
            "min_joint_views": 2,
            "factor": "1 - 0.8 * disagreeing_views / jointly_observed_views",
            "no_joint_evidence_factor": 1,
        },
        "scope": "fixed local-patch representative adaptation; not full original MaskClustering reproduction",
    }
    registry = (
        path(config["compact_output_root"])
        / "independent_mask_bank_room0_registry.json"
    )
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("frozen independent mask configuration differs")
    else:
        _atomic_json(registry, settings)
    representatives = output / "representatives.npz"
    if representatives.exists():
        with np.load(representatives) as data:
            points, rows = data["xyz"], data["source_rows"]
    else:
        with np.load(patch_file) as data:
            patch, centers = data["source_patch"], data["centers"]
        with np.load(
            path(
                "$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/replica_room0_static_full_v1/current_map/current_surface.npz"
            )
        ) as data:
            xyz = data["vertices_xyz"]
            if not data["current_valid"].all() or np.any(data["source_visit_ids"] != 0):
                raise ValueError("static source state differs")
        distance = np.sum(np.square(xyz - centers[patch]), axis=1)
        nearest = np.full(len(centers), np.inf)
        np.minimum.at(nearest, patch, distance)
        candidates = np.flatnonzero(distance == nearest[patch])
        rows = np.full(len(centers), len(xyz), np.int64)
        np.minimum.at(rows, patch[candidates], candidates)
        if np.any(rows >= len(xyz)):
            raise ValueError("patch lacks a physical source representative")
        points = xyz[rows]
        atomic_npz(representatives, xyz=points, source_rows=rows)
        del xyz, patch, distance, nearest, candidates, centers
    dataset = ReplicaRoom0Dataset(path(case["rgb_projection"]["dataset_root"]))
    all_nodes = np.arange(len(points))
    completed, missing = [], []
    for frame_id in settings["frames"]:
        mask_file = frontend / f"frame{frame_id:06d}.png"
        diagnostic = frontend / "frame_diagnostics" / f"frame{frame_id:06d}.json"
        if not mask_file.exists() or not diagnostic.exists():
            missing.append(frame_id)
            continue
        # Producer emits the diagnostic after the image; parse before consuming.
        json.loads(diagnostic.read_text())
        digest = file_hash(mask_file)
        target = output / f"{frame_id:06d}.npz"
        if target.exists():
            with np.load(target) as data:
                if (
                    str(data["mask_sha256"]) != digest
                    or int(data["frame_id"]) != frame_id
                ):
                    raise ValueError("cached mask observation binding changed")
            completed.append(frame_id)
            continue
        frame = dataset[frame_id]
        labels = np.asarray(Image.open(mask_file))
        if labels.shape != frame.depth.shape:
            raise ValueError("independent mask/RGBD grid mismatch")
        clean = interior_labels(labels)
        visible, pixels = project_source_support(points, all_nodes, frame)
        mask_ids = clean.ravel()[pixels]
        valid = mask_ids > 0
        nodes = visible[valid]
        atomic_npz(
            target,
            frame_id=frame_id,
            visit_id=0,
            mask_sha256=digest,
            node_ids=nodes,
            mask_ids=mask_ids[valid],
            pixels=pixels[valid],
            rgb=frame.rgb.reshape(-1, 3)[pixels[valid]],
            depth_visible_nodes=visible,
            boundary_or_unsegmented_count=int((~valid).sum()),
        )
        completed.append(frame_id)
        print(f"frame {frame_id}: {len(nodes)} interior patch observations", flush=True)
    status = {
        "case": "room0",
        "status": "COMPLETE" if not missing else "PARTIAL",
        "frame_count": len(completed),
        "authorized_frame_count": 200,
        "completed_frames": completed,
        "missing_frames": missing,
        "patch_count": len(points),
        "GT_read": False,
        "registry": registry.name,
    }
    _atomic_json(
        path(config["compact_output_root"]) / "independent_mask_bank_room0_status.json",
        status,
    )
    print(status["status"], len(completed), "/ 200", flush=True)
    if missing and not args.available_only:
        raise RuntimeError(
            "full evidence bank is not ready; frontend is still producing frames"
        )


if __name__ == "__main__":
    main()
