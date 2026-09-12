"""Paired full-surface M4 readouts from positive-depth native owner projections."""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_crove_fine_current_map import (
    NON_INSTANCE_CLASSES,
    EntityEvaluationInfo,
    LabeledMesh,
    _owner_row_groups,
    evaluate_replica_voxel_map,
    load_ovimap_semantic_mapping,
    load_replica_ground_truth,
)
from src.datasets.replica import ReplicaRoom0Dataset
from src.oviv2.mask_adapter_readout import MaskAdapterReadout
from src.oviv2.surface_view_bank import project_source_support


def path(value):
    p = Path(os.path.expandvars(str(value)))
    return p if p.is_absolute() else ROOT / p


def atomic_npz(target, **values):
    temporary = target.with_suffix(".partial")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **values)
    temporary.replace(target)


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    case = json.loads(path(config["source_config"]).read_text())["cases"][
        "replica_room0_static"
    ]
    manifest = json.loads(path(case["benchmark_manifest"]).read_text())
    classes = manifest["vocabulary"]["classes"]
    run_root = path(config["run_root"])
    output = run_root / "dev/room0/adapter_projected_top4"
    output.mkdir(parents=True, exist_ok=True)
    bank_root = output / "view_bank"
    bank_root.mkdir(exist_ok=True)
    compact = path(config["compact_output_root"])
    registry = compact / "adapter_room0_registry.json"
    settings = {
        "case": "room0",
        "split": "dev",
        "top_k": 4,
        "candidate_pool": "native retained same-visit observations",
        "mask": "positive-depth source-row projection; no dilation; no GT",
        "depth_tolerance_m": 0.05,
        "minimum_unique_pixels": 16,
        "aggregation": "normalize each view embedding then uniform mean and normalize",
        "fallback": "B_SEM_OVI_NATIVE",
        "crop_count": 0,
        "methods": ["ADAPTER_CLIP_MEAN", "ADAPTER_CLIP_LEARNED"],
        "checkpoint_sha256": "5686dc21e3461918d385d5cee02b103320e8ca30714485cb8f1694f0d590b55e",
    }
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("existing projection/adapter configuration differs")
    else:
        with registry.open("x") as f:
            json.dump(settings, f, indent=2)
    surface = path(
        "$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/replica_room0_static_full_v1/current_map/current_surface.npz"
    )
    with np.load(surface) as data:
        xyz, owners = data["vertices_xyz"], data["owner_entity_ids"]
        source = data["source_vertex_indices"]
        if not data["current_valid"].all() or np.any(data["source_visit_ids"] != 0):
            raise ValueError("static source state changed")
    with np.load(
        run_root / "dev/room0/native_cached_batch/B_SEM_OVI_NATIVE.npz"
    ) as base:
        if not np.array_equal(base["source_indices"], source) or not np.array_equal(
            base["owner_ids"], owners
        ):
            raise ValueError("native source/owner mapping differs from static export")
        base_ids, base_conf = base["semantic_ids"], base["semantic_confidence"]
    groups = _owner_row_groups(owners)
    native_path = (
        path(case["fine_surface"]).parent
        / "inst_sem_siglip-l-16-384_200_incre_combine.pkl"
    )
    with native_path.open("rb") as f:
        native = pickle.load(f)
    jobs = defaultdict(list)
    for owner, entry in sorted(native.items()):
        if owner not in groups or len(entry["frame_id"]) < 2:
            continue
        area = np.asarray(entry["vis_area"])
        for i in np.argsort(-area, kind="stable")[:4]:
            frame_id = int(entry["frame_id"][i])
            if frame_id < 0 or frame_id >= 2000 or frame_id % 10:
                raise ValueError("frame outside authorized input")
            jobs[frame_id].append((owner, np.asarray(entry["pose"][i])))
    dataset = ReplicaRoom0Dataset(path(case["rgb_projection"]["dataset_root"]))
    model = MaskAdapterReadout(
        path(
            "$HOME/oviovo_baseline_builds/maskadapter/fcclip_convnext_large_maskadapter.pth"
        )
    )
    text = model.text_features(classes)
    scale = float(model.clip.logit_scale.exp().clamp(max=100).cpu())
    torch.cuda.reset_peak_memory_stats()
    print(
        f"M4 {len(jobs)} frames; {sum(map(len, jobs.values()))} owner observations",
        flush=True,
    )
    for index, (frame_id, entries) in enumerate(sorted(jobs.items())):
        target = bank_root / f"{frame_id:06d}.npz"
        if target.exists():
            continue
        started = time.monotonic()
        frame = dataset[frame_id]
        masks, selected_owners, source_chunks, pixel_chunks = [], [], [], []
        offsets = [0]
        unique_pixels = []
        for owner, pose in entries:
            if not np.allclose(frame.pose, pose, atol=1e-5, rtol=1e-5):
                raise ValueError("cached native pose differs from RGBD pose")
            rows, pixels = project_source_support(xyz, groups[owner], frame)
            unique = np.unique(pixels)
            if len(unique) < 16:
                continue
            mask = np.zeros(frame.depth.size, bool)
            mask[unique] = True
            masks.append(mask.reshape(frame.depth.shape))
            selected_owners.append(owner)
            source_chunks.append(rows)
            pixel_chunks.append(pixels)
            offsets.append(offsets[-1] + len(rows))
            unique_pixels.append(len(unique))
        projected_seconds = time.monotonic() - started
        if masks:
            result = model.classify_external_masks(frame.rgb, np.stack(masks), text)
            kept = result["mask_indices"].cpu().numpy()
            mean = result["mean_embeddings"].cpu().numpy()
            learned = result["learned_embeddings"].cpu().numpy()
        else:
            kept = np.empty(0, np.int64)
            mean = learned = np.empty((0, 768), np.float32)
        atomic_npz(
            target,
            owner_ids=np.asarray(selected_owners, np.int64),
            source_indices=np.concatenate(source_chunks)
            if source_chunks
            else np.empty(0, np.int64),
            source_pixels=np.concatenate(pixel_chunks)
            if pixel_chunks
            else np.empty(0, np.int32),
            source_offsets=np.asarray(offsets),
            unique_pixel_counts=np.asarray(unique_pixels),
            feature_owner_ids=np.asarray(selected_owners, np.int64)[kept],
            mean_embeddings=mean,
            learned_embeddings=learned,
            frame_id=frame_id,
            visit_id=0,
            projection_seconds=projected_seconds,
            total_seconds=time.monotonic() - started,
        )
        if index < 3 and masks:
            overlay = frame.rgb.copy()
            overlay[masks[0]] = (
                0.5 * overlay[masks[0]] + 0.5 * np.array([0, 255, 0])
            ).astype(np.uint8)
            Image.fromarray(overlay).save(bank_root / f"{frame_id:06d}_projection.png")
        print(
            f"frame {frame_id}: {len(kept)} masks ({index + 1}/{len(jobs)})", flush=True
        )
    peak = torch.cuda.max_memory_allocated()
    text_np = text.cpu().numpy()
    del model
    torch.cuda.empty_cache()
    # Pool independent views at owner level; zero-support owners retain fallback.
    feature = {"mean": defaultdict(list), "learned": defaultdict(list)}
    references = defaultdict(list)
    times = {"projection_seconds": 0.0, "total_seconds": 0.0}
    encoder_forwards = 0
    for frame_id in sorted(jobs):
        with np.load(bank_root / f"{frame_id:06d}.npz") as data:
            encoder_forwards += int(len(data["owner_ids"]) > 0)
            for name in times:
                times[name] += float(data[name])
            for i, owner in enumerate(data["feature_owner_ids"]):
                for kind in feature:
                    feature[kind][int(owner)].append(data[f"{kind}_embeddings"][i])
                references[int(owner)].append(frame_id)
    for kind, method in [
        ("mean", "ADAPTER_CLIP_MEAN"),
        ("learned", "ADAPTER_CLIP_LEARNED"),
    ]:
        target = output / f"{method}.npz"
        if target.exists():
            continue
        ids, conf = base_ids.copy(), base_conf.copy()
        coverage = np.zeros(len(ids), bool)
        feature_owners, posteriors = [], []
        for owner, embeddings in sorted(feature[kind].items()):
            z = np.mean(embeddings, axis=0)
            z /= max(np.linalg.norm(z), 1e-12)
            logits = scale * (z @ text_np.T)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            ids[groups[owner]] = int(p.argmax()) + 1
            conf[groups[owner]] = float(p.max())
            coverage[groups[owner]] = True
            feature_owners.append(owner)
            posteriors.append(p)
        atomic_npz(
            target,
            semantic_ids=ids,
            semantic_confidence=conf,
            owner_ids=owners,
            source_indices=source,
            feature_coverage=coverage,
            feature_owners=np.asarray(feature_owners),
            owner_posterior=np.asarray(posteriors),
            class_ids=np.arange(1, len(classes) + 1),
        )
    # GT is loaded only after both complete predictions are saved.
    scene = next(s for s in manifest["scenes"] if s["scene"] == "room0")
    gt = path(manifest["ground_truth_root"]) / scene["ground_truth_scene"] / "habitat"
    class_ids = {name: i + 1 for i, name in enumerate(classes)}
    truth = load_replica_ground_truth(
        gt / "mesh_semantic.ply",
        gt / "info_semantic.json",
        class_to_id=class_ids,
        aliases=manifest["aliases"],
    )
    mapping = load_ovimap_semantic_mapping(
        path(case["ovimap_semantic_mapping"]),
        source_vocabulary=tuple(case["ovimap_source_vocabulary"]),
        target_vocabulary=tuple(classes),
    )
    info = [
        EntityEvaluationInfo(int(o), mapping[o][0], 2, mapping[o][1])
        for o in np.unique(owners)
        if mapping.get(o, (0, 0))[0] > 0
    ]
    for method in settings["methods"]:
        target = compact / f"room0_{method}.json"
        if target.exists():
            continue
        with np.load(output / f"{method}.npz") as data:
            ids, conf = data["semantic_ids"], data["semantic_confidence"]
            coverage = float(data["feature_coverage"].mean())
        mesh = LabeledMesh(
            xyz,
            np.empty((0, 3), np.int64),
            np.zeros_like(xyz),
            ids,
            owners,
            conf,
            (owners > 0).astype(np.float32),
        )
        measured = evaluate_replica_voxel_map(
            mesh,
            truth,
            info,
            valid_semantic_ids=class_ids.values(),
            instance_semantic_ids={
                i for c, i in class_ids.items() if c not in NON_INSTANCE_CLASSES
            },
            min_instance_vertices=int(case["min_instance_vertices"]),
            distance_threshold_m=0.05,
        )
        record = {
            "method": method,
            "case": "room0",
            "split": "dev",
            "status": "FULL_MAP_EVALUATED",
            "metrics": measured,
            "geometry_fixed": True,
            "owner_changed": False,
            "feature_coverage": coverage,
            "source_observations": dict(references),
            "frame_count": len(jobs),
            "shared_runtime": times,
            "peak_gpu_bytes": peak,
            "geometry_mask_policy": settings["mask"],
            "independent_views": sum(map(len, references.values())),
            "extra_dense_encoder_forwards": encoder_forwards,
            "training_updates": 0,
            "semantic_constrained_AP": "diagnostic with frozen native entity-info",
        }
        with target.open("x") as f:
            json.dump(record, f, indent=2, allow_nan=False)
        print(
            method,
            {k: measured[k] for k in ["miou", "f_miou", "ap50", "f5"]},
            flush=True,
        )


if __name__ == "__main__":
    main()
