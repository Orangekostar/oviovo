"""Actual six-crop observations for frozen local structural-background regions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.build_crove_room0_mask_bank import file_hash
from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_crove_fine_current_map import _owner_row_groups
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.datasets.replica import ReplicaRoom0Dataset
from src.datasets.tesse_cd import TesseCdRgbdDataset
from src.oviv2.ovi_surface_attributes import project_world_points
from src.oviv2.surface_multiview_semantics import (
    aggregate_views,
    diverse_views,
    native_crop_inputs,
    native_six_crops,
)
from src.oviv2.surface_view_bank import project_source_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=["room0", "room1", "apartment_B3", "apartment_H2"],
    )
    args = parser.parse_args()
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    run, compact = path(config["run_root"]), path(config["compact_output_root"])
    pair = json.loads(path(config["state_config"]).read_text())["splits"]["dev"][
        "pairs"
    ][0]
    root = (
        run
        / (
            "confirm/local_background_regions"
            if args.case == "room1"
            else "dev/local_background_regions"
        )
        / args.case
    )
    output = root / "observations"
    output.mkdir(exist_ok=True)
    static = args.case in ("room0", "room1")
    if static:
        case = json.loads(path(config["source_config"]).read_text())["cases"][
            "replica_room0_static"
        ]
        room = run / ("dev/room0" if args.case == "room0" else "confirm/room1")
        bank = room / "independent_mask_bank"
        dataset = ReplicaRoom0Dataset(
            path(case["rgb_projection"]["dataset_root"]).with_name(args.case)
        )
        frames = list(range(0, 2000, 10))
        geometry = path(
            "$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/replica_room0_static_full_v1/current_map/current_surface.npz"
        )
        if args.case == "room1":
            geometry = room / "inputs/current_surface.npz"

        def mask_path(frame):
            frontend = (
                "cropformer_cpu/frontend"
                if args.case == "room0"
                else "native_cpu/frontend"
            )
            return room / frontend / f"frame{frame:06d}.png"

        def camera(frame):
            return dataset._records[frame].pose[:3, 3]
    else:
        room = run / "dev/apartment"
        bank = room / "independent_mask_bank" / args.case.split("_")[1]
        dataset = TesseCdRgbdDataset(
            path(pair["rgbd_root"]),
            pair["scene"],
            path(pair["rgbd_export_manifest"]),
            path(pair["causal_schedule"]),
        )
        frames = [
            f
            for visit in (0, 1)
            for f in range(
                pair["visit_frame_ranges"][f"t{visit}"][0],
                pair["visit_frame_ranges"][f"t{visit}"][1] + 1,
            )
        ]
        geometry = path(pair["current_map_root"]) / "current_surface.npz"

        def mask_path(frame):
            visit = int(frame >= pair["visit_frame_ranges"]["t1"][0])
            local = frame - pair["visit_frame_ranges"][f"t{visit}"][0]
            return path(
                f"$HOME/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/native-runs/apartment/t{visit}/2f638911509f-apartment-t{visit}/frontend/frame{local:06d}.png"
            )

        def camera(frame):
            return dataset.records[frame].camera_to_world[:3, 3]

    settings = {
        "case": args.case,
        "region_sha256": file_hash(root / "regions.npz"),
        "region_binding_sha256": file_hash(root / "input_binding.json"),
        "frames": frames,
        "minimum_unique_interior_pixels": 16,
        "minimum_views": 2,
        "k": 4,
        "selection": "same angular diverse selection with maximum visibility seed",
        "encoding": "shared SigLIP native six RGB/black-mask crops, slow PIL explicit channels_last; crop normalization then within-view mean",
        "quality": "depth-supported/projectable unique pixels * sqrt(min(1,support/touched independent mask union area)) * image truncation factor .5 or 1",
        "aggregation": "normalized views, quality-weighted; no labels invented for missing features",
        "GT_read": False,
    }
    registry = compact / f"{args.case}_local_background_encoding_registry.json"
    if registry.exists() and json.loads(registry.read_text()) != settings:
        raise ValueError("frozen local encoding policy changed")
    _atomic_json(registry, settings)
    with np.load(root / "regions.npz") as data:
        source, regions, centers = (
            data["source_indices"],
            data["source_region"],
            data["centers"],
        )
        region_visits = data["region_visit_ids"]
    with np.load(geometry) as data:
        xyz = data["vertices_xyz"]
    with np.load(bank / "representatives.npz") as data:
        representatives = data["source_rows"]
    local = np.searchsorted(source, representatives)
    valid = local < len(source)
    valid[valid] &= source[local[valid]] == representatives[valid]
    node_regions = np.full(len(representatives), -1, np.int64)
    node_regions[valid] = regions[local[valid]]
    candidates = defaultdict(list)
    for frame_id in frames:
        mask = mask_path(frame_id)
        with np.load(bank / f"{frame_id:06d}.npz") as data:
            if file_hash(mask) != str(data["mask_sha256"]):
                raise ValueError("independent mask differs from frozen bank")
            observed = node_regions[data["node_ids"]]
            pixels, mask_ids = data["pixels"], data["mask_ids"]
        areas = np.bincount(np.asarray(Image.open(mask)).ravel())
        for region, rows in _owner_row_groups(observed).items():
            if region < 0:
                continue
            count = len(np.unique(pixels[rows]))
            if count >= 16:
                candidates[region].append(
                    (frame_id, count, int(areas[np.unique(mask_ids[rows])].sum()))
                )
    jobs, selected = defaultdict(list), {}
    for region, observations in sorted(candidates.items()):
        if len(observations) < 2:
            continue
        frame_ids = [v[0] for v in observations]
        if not static and any(
            int(f >= pair["visit_frame_ranges"]["t1"][0]) != region_visits[region]
            for f in frame_ids
        ):
            raise ValueError("local region selection crossed visits")
        chosen = diverse_views(
            np.array([camera(f) for f in frame_ids]) - centers[region],
            np.array([v[1] for v in observations]),
            k=4,
        )
        selected[region] = [observations[i][0] for i in chosen]
        for i in chosen:
            frame_id, visibility, area = observations[i]
            jobs[frame_id].append((region, visibility, area))
    _atomic_json(root / "selected_views.json", {str(k): v for k, v in selected.items()})
    print(
        args.case,
        len(selected),
        "regions;",
        sum(map(len, jobs.values())),
        "six-crop batches selected",
        flush=True,
    )
    text_file = room / "native_cached_batch/text_features.npz"
    with np.load(text_file) as data:
        text, scale = data["text"], float(data["logit_scale"])
        classes = (
            data["class_ids"] if "class_ids" in data else np.arange(1, len(text) + 1)
        )
    model_path = path(
        "$HOME/oviovo_baseline_builds/ovimap-ubuntu24-native/siglip-large-patch16-384"
    )
    binding = {
        "text": file_hash(text_file),
        "weights": {p.name: file_hash(p) for p in model_path.glob("*.safetensors")},
        "policy": file_hash(registry),
    }
    binding_file = output / "input_binding.json"
    if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
        raise ValueError("local encoding cache input changed")
    _atomic_json(binding_file, binding)
    groups = _owner_row_groups(regions)
    model = processor = None
    features, forwards, encoding_seconds = {}, 0, 0.0
    for frame_id, jobs_in_frame in sorted(jobs.items()):
        target = output / f"{frame_id:06d}.npz"
        if not target.exists():
            frame = dataset[frame_id]
            keys, zs, qs, supports, offsets = [], [], [], [], [0]
            for region, visibility, area in jobs_in_frame:
                rows = source[groups[region]]
                _, pixels = project_source_support(xyz, rows, frame)
                pixels = np.unique(pixels)
                if len(pixels) < 16:
                    continue
                projected = project_world_points(
                    xyz[rows], frame.pose, frame.intrinsics
                )
                valid = projected.projectable
                count = len(
                    np.unique(
                        projected.rows[valid] * frame.depth.shape[1]
                        + projected.columns[valid]
                    )
                )
                mask = np.zeros(frame.depth.size, bool)
                mask[pixels] = True
                mask = mask.reshape(frame.depth.shape)
                crops = native_six_crops(frame.rgb, mask)
                if not crops:
                    continue
                truncated = (
                    mask[0].any()
                    or mask[-1].any()
                    or mask[:, 0].any()
                    or mask[:, -1].any()
                )
                q = (
                    min(1.0, len(pixels) / max(1, count))
                    * np.sqrt(min(1.0, len(pixels) / max(1, area)))
                    * (0.5 if truncated else 1.0)
                )
                if model is None:
                    processor = AutoProcessor.from_pretrained(
                        model_path, local_files_only=True, use_fast=False
                    )
                    model = (
                        AutoModel.from_pretrained(model_path, local_files_only=True)
                        .eval()
                        .to("cuda:1")
                    )
                inputs = native_crop_inputs(processor, crops).to("cuda:1")
                started = time.monotonic()
                with torch.inference_mode():
                    z = model.get_image_features(**inputs)
                    z = (z / z.norm(dim=-1, keepdim=True)).mean(0).cpu().numpy()
                encoding_seconds += time.monotonic() - started
                forwards += 1
                keys.append(region)
                zs.append(z)
                qs.append(q)
                supports.append(pixels)
                offsets.append(offsets[-1] + len(pixels))
            atomic_npz(
                target,
                region_ids=np.array(keys, np.int64),
                embeddings=np.array(zs).reshape(-1, text.shape[1]),
                quality=np.array(qs),
                mask_pixels=np.concatenate(supports)
                if supports
                else np.empty(0, np.int32),
                mask_offsets=np.array(offsets),
            )
        with np.load(target) as data:
            for i, region in enumerate(data["region_ids"]):
                features[(int(region), frame_id)] = (
                    data["embeddings"][i],
                    data["quality"][i],
                )
        if frame_id % 16 == 0:
            print(args.case, frame_id, "encoded/cached", flush=True)
    keys, posterior = [], []
    for region, frames in sorted(selected.items()):
        values = [
            features[(region, f)]
            for f in frames
            if (region, f) in features and features[(region, f)][1] > 0
        ]
        if len(values) < 2:
            continue
        z, q = map(np.array, zip(*values))
        embedding = aggregate_views(z, q)
        if embedding is None:
            continue
        logits = scale * (embedding @ text.T)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        keys.append(region)
        posterior.append(p)
    atomic_npz(
        root / "local_posteriors.npz",
        region_ids=np.array(keys, np.int64),
        posterior=np.array(posterior).reshape(-1, len(classes)),
        class_ids=classes,
    )
    _atomic_json(
        compact / f"{args.case}_local_background_encoding.json",
        {
            "case": args.case,
            "status": "REAL_LOCAL_FEATURES_COMPLETE_READOUT_PENDING",
            "feature_region_count": len(keys),
            "selected_regions": len(selected),
            "encoded_observations": len(features),
            "new_six_crop_batches": forwards,
            "encoding_seconds": encoding_seconds,
            "runtime_scope": "this execution encoder forwards only",
            "GT_read": False,
        },
    )
    print(args.case, "local posteriors saved", len(keys), flush=True)


if __name__ == "__main__":
    main()
