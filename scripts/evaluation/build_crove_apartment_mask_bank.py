"""Bind complete native independent masks to each frozen Apartment state/visit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.build_crove_room0_mask_bank import file_hash
from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.datasets.tesse_cd import TesseCdRgbdDataset
from src.oviv2.surface_mask_evidence import interior_labels
from src.oviv2.surface_view_bank import project_source_support


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    room = path(config["run_root"]) / "dev/apartment"
    output = room / "independent_mask_bank"
    output.mkdir(exist_ok=True)
    binding = json.loads(
        (room / "native_cached_batch/native_feature_binding.json").read_text()
    )
    native_roots, mask_hashes = [], {}
    for visit in (0, 1):
        root = path(
            f"$HOME/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/native-runs/apartment/t{visit}/2f638911509f-apartment-t{visit}"
        )
        if (
            file_hash(root / "native_mapping_manifest.json")
            != binding[visit]["native_manifest_sha256"]
        ):
            raise ValueError("native source binding changed")
        start, stop = pair["visit_frame_ranges"][f"t{visit}"]
        for frame in range(start, stop + 1):
            mask = root / "frontend" / f"frame{frame - start:06d}.png"
            mask_hashes[str(frame)] = file_hash(mask)
        native_roots.append(root)
    settings = {
        "case": pair["pair_id"],
        "states": ["B3", "H2"],
        "frame_windows": pair["visit_frame_ranges"],
        "native_manifest_sha256": [b["native_manifest_sha256"] for b in binding],
        "mask_sha256": mask_hashes,
        "patch_sha256": {
            s: file_hash(room / "graph_topology" / s / "patch_mapping.npz")
            for s in ("B3", "H2")
        },
        "representative": "actual current source row nearest frozen patch center; ties by source row order",
        "projection": "original global TESSE pose, own visit only, positive measured depth within 0.05 m",
        "boundary": "one-pixel 8-neighbor erosion; image boundary excluded",
        "missing": "occluded or unsegmented patches supply no observer",
        "independence": "native CropFormer PNG labels, not projected owner masks",
        "cross_visit_union": False,
        "GT_read": False,
    }
    registry = (
        path(config["compact_output_root"])
        / "independent_mask_bank_apartment_registry.json"
    )
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("frozen independent mask policy changed")
    else:
        _atomic_json(registry, settings)
    with np.load(path(pair["current_map_root"]) / "current_surface.npz") as data:
        xyz = data["vertices_xyz"]
        source_visits = data["source_visit_ids"]
    dataset = TesseCdRgbdDataset(
        path(pair["rgbd_root"]),
        pair["scene"],
        path(pair["rgbd_export_manifest"]),
        path(pair["causal_schedule"]),
    )
    status = {}
    for state in ("B3", "H2"):
        root = output / state
        root.mkdir(exist_ok=True)
        representatives = root / "representatives.npz"
        with np.load(room / "graph_topology" / state / "patch_mapping.npz") as data:
            source, patch, centers = (
                data["source_indices"],
                data["source_patch"],
                data["centers"],
            )
            visits = data["source_visit_ids"]
        if not np.array_equal(visits, source_visits[source]):
            raise ValueError("patch/source visit identity mismatch")
        if representatives.exists():
            with np.load(representatives) as data:
                points, rows, node_visits = (
                    data["xyz"],
                    data["source_rows"],
                    data["visit_ids"],
                )
            if len(points) != len(centers) or not np.array_equal(points, xyz[rows]):
                raise ValueError("cached representative identity mismatch")
        else:
            distance = np.sum(np.square(xyz[source] - centers[patch]), axis=1)
            nearest = np.full(len(centers), np.inf)
            np.minimum.at(nearest, patch, distance)
            candidates = np.flatnonzero(distance == nearest[patch])
            rows = np.full(len(centers), len(xyz), np.int64)
            np.minimum.at(rows, patch[candidates], source[candidates])
            if np.any(rows >= len(xyz)):
                raise ValueError("patch lacks a current source representative")
            points, node_visits = xyz[rows], source_visits[rows]
            atomic_npz(
                representatives, xyz=points, source_rows=rows, visit_ids=node_visits
            )
            del distance, nearest, candidates
        if not np.array_equal(node_visits[patch], visits):
            raise ValueError("patch crosses visit boundary")
        del patch, centers, source, visits
        completed = []
        for visit in (0, 1):
            nodes = np.flatnonzero(node_visits == visit)
            start, stop = pair["visit_frame_ranges"][f"t{visit}"]
            for frame_id in range(start, stop + 1):
                target = root / f"{frame_id:06d}.npz"
                digest = mask_hashes[str(frame_id)]
                if target.exists():
                    with np.load(target) as data:
                        if (
                            str(data["mask_sha256"]) != digest
                            or int(data["frame_id"]) != frame_id
                            or int(data["visit_id"]) != visit
                        ):
                            raise ValueError("cached frame binding changed")
                    completed.append(frame_id)
                    continue
                frame = dataset[frame_id]
                labels = np.asarray(
                    Image.open(
                        native_roots[visit]
                        / "frontend"
                        / f"frame{frame_id - start:06d}.png"
                    )
                )
                if labels.shape != frame.depth.shape:
                    raise ValueError("mask and depth grids differ")
                clean = interior_labels(labels)
                visible, pixels = project_source_support(points, nodes, frame)
                mask_ids = clean.ravel()[pixels]
                usable = mask_ids > 0
                atomic_npz(
                    target,
                    frame_id=frame_id,
                    local_frame_id=frame_id - start,
                    visit_id=visit,
                    mask_sha256=digest,
                    node_ids=visible[usable],
                    mask_ids=mask_ids[usable],
                    pixels=pixels[usable],
                    rgb=frame.rgb.reshape(-1, 3)[pixels[usable]],
                    depth_visible_nodes=visible,
                    boundary_or_unsegmented_count=int((~usable).sum()),
                )
                completed.append(frame_id)
                if frame_id % 32 == 0:
                    print(
                        state,
                        frame_id,
                        len(visible[usable]),
                        "interior patch observations",
                        flush=True,
                    )
        status[state] = {
            "status": "COMPLETE",
            "completed_frames": completed,
            "frame_count": len(completed),
            "patch_count": len(points),
            "GT_read": False,
        }
        _atomic_json(
            path(config["compact_output_root"])
            / "independent_mask_bank_apartment_status.json",
            status,
        )
        print(state, "complete", len(completed), "frames", flush=True)


if __name__ == "__main__":
    main()
