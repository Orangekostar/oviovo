"""Matched original/consensus region re-encoding on the complete room0 bank."""

from __future__ import annotations

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
from scripts.evaluation.run_crove_fine_current_map import (
    NON_INSTANCE_CLASSES,
    EntityEvaluationInfo,
    LabeledMesh,
    _owner_row_groups,
    evaluate_replica_voxel_map,
    load_replica_ground_truth,
)
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.datasets.replica import ReplicaRoom0Dataset
from src.oviv2.ovi_surface_attributes import project_world_points
from src.oviv2.surface_multiview_semantics import (
    aggregate_views,
    diverse_views,
    native_crop_inputs,
    native_six_crops,
    owner_posteriors_to_rows,
)
from src.oviv2.surface_view_bank import project_source_support


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    case = json.loads(path(config["source_config"]).read_text())["cases"][
        "replica_room0_static"
    ]
    manifest = json.loads(path(case["benchmark_manifest"]).read_text())
    room = path(config["run_root"]) / "dev/room0"
    compact = path(config["compact_output_root"])
    output = room / "reencoded_regions"
    output.mkdir(exist_ok=True)
    methods = [
        "B_SEM_OVI_REENCODE",
        "MV_REENCODE_DIVERSE",
        "MV_REENCODE_QUALITY",
        "INST_CONSENSUS_RESEM",
    ]
    frames = list(range(0, 2000, 10))
    status = json.loads(
        (compact / "independent_mask_bank_room0_status.json").read_text()
    )
    if status["completed_frames"] != frames or status["status"] != "COMPLETE":
        raise ValueError("complete 200-frame independent bank required")
    settings = {
        "case": "room0",
        "methods": methods,
        "frames": frames,
        "k": 4,
        "candidate": "at least 16 distinct independent-mask-interior representative pixels",
        "selection": "native top visibility four; diverse/quality/resem angular farthest first with maximum visibility seed",
        "crops": "shared native six crops, each crop normalized before within-view mean; slow PIL processor with explicit channels_last",
        "projection": "same saved Replica poses as independent bank; measured depth within .05m, depth <=10m, no dilation",
        "quality": "depth-supported/projectable unique pixels * sqrt(min(1,support/touched independent mask union area)) * truncation factor .5 or 1",
        "aggregation": "native visibility-weighted raw view means; diverse normalized uniform; quality/resem normalized quality weighted",
        "minimum_feature_views": 2,
        "missing": "KEEP_SOURCE semantic IDs and confidence",
        "text": "same saved complete public Replica41 text features and native logit scale",
        "geometry": "all source rows retained; original owners for controls, fixed saved consensus owners for resem",
        "GT_read_before_all_predictions": False,
    }
    registry = compact / "reencoded_regions_room0_registry.json"
    if registry.exists() and json.loads(registry.read_text()) != settings:
        raise ValueError("frozen re-encoding policy changed")
    _atomic_json(registry, settings)
    dataset = ReplicaRoom0Dataset(path(case["rgb_projection"]["dataset_root"]))
    classes = np.arange(1, len(manifest["vocabulary"]["classes"]) + 1)
    text_file = room / "native_cached_batch/text_features.npz"
    with np.load(text_file) as data:
        text, scale = data["text"], float(data["logit_scale"])
    if len(text) != len(classes):
        raise ValueError("text vocabulary changed")
    native_file = room / "native_cached_batch/B_SEM_OVI_NATIVE.npz"
    with np.load(native_file) as data:
        native = {
            k: data[k]
            for k in (
                "source_indices",
                "owner_ids",
                "semantic_ids",
                "semantic_confidence",
            )
        }
    source = native["source_indices"]
    with np.load(
        path(
            "$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/replica_room0_static_full_v1/current_map/current_surface.npz"
        )
    ) as data:
        xyz = data["vertices_xyz"]
        if (
            not np.array_equal(data["source_vertex_indices"], source)
            or not np.array_equal(source, np.arange(len(xyz)))
            or not np.array_equal(data["owner_entity_ids"], native["owner_ids"])
            or not data["current_valid"].all()
        ):
            raise ValueError("native source geometry/state differs")
    bank = room / "independent_mask_bank"
    with np.load(bank / "representatives.npz") as data:
        representative_rows = data["source_rows"]
    model_path = path(
        "$HOME/oviovo_baseline_builds/ovimap-ubuntu24-native/siglip-large-patch16-384"
    )
    weights = sorted(model_path.glob("*.safetensors"))
    if not weights:
        raise ValueError("local encoder weights missing")
    binding_common = {
        "model_weights": {p.name: file_hash(p) for p in weights},
        "text_sha256": file_hash(text_file),
        "mask_registry_sha256": file_hash(
            compact / "independent_mask_bank_room0_registry.json"
        ),
        "policy_sha256": file_hash(registry),
    }
    model = processor = None
    total_forwards, projection_seconds, encoding_seconds = 0, 0.0, 0.0
    for partition in ("native", "consensus"):
        root = output / partition
        root.mkdir(exist_ok=True)
        baseline_file = (
            native_file
            if partition == "native"
            else room / "consensus/INST_CONSENSUS_OWNER.npz"
        )
        with np.load(baseline_file) as data:
            owners = data["owner_ids"]
            for key in ("source_indices", "semantic_ids", "semantic_confidence"):
                if not np.array_equal(data[key], native[key]):
                    raise ValueError("owner-only baseline changed source semantics")
        binding = dict(binding_common, prediction_sha256=file_hash(baseline_file))
        binding_file = root / "input_binding.json"
        if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
            raise ValueError("cached re-encoding inputs changed")
        _atomic_json(binding_file, binding)
        groups = _owner_row_groups(owners)
        candidates = defaultdict(list)
        for frame_id in frames:
            mask_file = room / "cropformer_cpu/frontend" / f"frame{frame_id:06d}.png"
            with np.load(bank / f"{frame_id:06d}.npz") as data:
                if str(data["mask_sha256"]) != file_hash(mask_file):
                    raise ValueError("independent mask changed")
                observed = owners[representative_rows[data["node_ids"]]]
                pixels, mask_ids = data["pixels"], data["mask_ids"]
            areas = np.bincount(np.asarray(Image.open(mask_file)).ravel())
            for owner, rows in _owner_row_groups(observed).items():
                count = len(np.unique(pixels[rows]))
                if owner > 0 and count >= 16:
                    candidates[int(owner)].append(
                        (frame_id, count, int(areas[np.unique(mask_ids[rows])].sum()))
                    )
        jobs, selected = defaultdict(list), {}
        for owner, observations in sorted(candidates.items()):
            if len(observations) < 2:
                continue
            center = np.unique(xyz[groups[owner]], axis=0).mean(0, dtype=np.float64)
            visibility = np.array([v[1] for v in observations])
            directions = (
                np.array([dataset._records[v[0]].pose[:3, 3] for v in observations])
                - center
            )
            diverse = diverse_views(directions, visibility, k=4)
            top = np.argsort(-visibility, kind="stable")[:4]
            selected[owner] = {
                "top": [observations[i][0] for i in top],
                "diverse": [observations[i][0] for i in diverse],
            }
            chosen = np.union1d(top, diverse) if partition == "native" else diverse
            for i in chosen:
                frame_id, visibility, area = observations[i]
                jobs[frame_id].append((owner, visibility, area))
        _atomic_json(
            root / "selected_views.json", {str(k): v for k, v in selected.items()}
        )
        print(
            partition,
            len(selected),
            "regions;",
            len(jobs),
            "frames;",
            sum(map(len, jobs.values())),
            "region-view batches",
            flush=True,
        )
        features = {}
        for frame_id, regions in sorted(jobs.items()):
            target = root / f"{frame_id:06d}.npz"
            if not target.exists():
                frame = dataset[frame_id]
                out_owners, embeddings, qualities, visible_counts = [], [], [], []
                mask_pixels, offsets = [], [0]
                for owner, visibility, area in regions:
                    started = time.monotonic()
                    rows = groups[owner]
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
                    quality = (
                        min(1.0, len(pixels) / max(1, count))
                        * np.sqrt(min(1.0, len(pixels) / max(1, area)))
                        * (0.5 if truncated else 1.0)
                    )
                    projection_seconds += time.monotonic() - started
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
                    total_forwards += 1
                    out_owners.append(owner)
                    embeddings.append(z)
                    qualities.append(quality)
                    visible_counts.append(visibility)
                    mask_pixels.append(pixels)
                    offsets.append(offsets[-1] + len(pixels))
                atomic_npz(
                    target,
                    feature_owners=np.array(out_owners, np.int64),
                    embeddings=np.array(embeddings).reshape(-1, text.shape[1]),
                    quality=np.array(qualities),
                    visibility=np.array(visible_counts),
                    mask_pixels=np.concatenate(mask_pixels)
                    if mask_pixels
                    else np.empty(0, np.int32),
                    mask_offsets=np.array(offsets),
                )
            with np.load(target) as data:
                for i, owner in enumerate(data["feature_owners"]):
                    features[(int(owner), frame_id)] = (
                        data["embeddings"][i],
                        float(data["quality"][i]),
                        float(data["visibility"][i]),
                    )
            print(partition, frame_id, "encoded/cached", flush=True)
        for method in methods[:3] if partition == "native" else methods[3:]:
            feature_owners, posterior = [], []
            for owner, indices in sorted(selected.items()):
                mode = "top" if method == methods[0] else "diverse"
                values = [
                    features[(owner, f)]
                    for f in indices[mode]
                    if (owner, f) in features
                ]
                if len(values) < 2:
                    continue
                z, quality, visibility = map(np.array, zip(*values))
                if method == methods[0]:
                    embedding = (z * visibility[:, None]).sum(0)
                    embedding /= np.linalg.norm(embedding)
                else:
                    embedding = aggregate_views(
                        z, np.ones(len(z)) if method == methods[1] else quality
                    )
                if embedding is None:
                    continue
                logits = scale * (embedding @ text.T)
                p = np.exp(logits - logits.max())
                p /= p.sum()
                feature_owners.append(owner)
                posterior.append(p)
            posterior = np.array(posterior).reshape(-1, len(classes))
            proposed, confidence, covered = owner_posteriors_to_rows(
                owners, np.array(feature_owners), posterior, classes
            )
            ids = native["semantic_ids"].copy()
            ids[covered] = proposed[covered]
            confidence[~covered] = native["semantic_confidence"][~covered]
            atomic_npz(
                output / f"{method}.npz",
                source_indices=source,
                owner_ids=owners,
                semantic_ids=ids,
                semantic_confidence=confidence,
                feature_covered=covered,
                class_ids=classes,
                feature_owners=np.array(feature_owners),
                owner_posterior=posterior,
            )
            _atomic_json(
                output / f"{method}_prediction.json",
                {
                    "method": method,
                    "source_rows": len(source),
                    "geometry_fixed": True,
                    "owner_fixed": partition == "native",
                    "owner_partition": partition,
                    "feature_coverage": float(covered.mean()),
                    "feature_owner_count": len(feature_owners),
                    "changed_semantic_rows": int(
                        np.count_nonzero(ids != native["semantic_ids"])
                    ),
                },
            )
            print(method, "full prediction saved", flush=True)
    _atomic_json(
        output / "execution_cost.json",
        {
            "new_six_crop_forward_batches": total_forwards,
            "new_crop_images": 6 * total_forwards,
            "projection_seconds": projection_seconds,
            "encoding_seconds": encoding_seconds,
            "scope": "this execution only; excludes model loading, cache I/O, selection and evaluation",
        },
    )
    # All four full predictions exist before opening GT.
    scene = next(s for s in manifest["scenes"] if s["scene"] == "room0")
    gt = path(manifest["ground_truth_root"]) / scene["ground_truth_scene"] / "habitat"
    vocabulary = {c: i + 1 for i, c in enumerate(manifest["vocabulary"]["classes"])}
    truth = load_replica_ground_truth(
        gt / "mesh_semantic.ply",
        gt / "info_semantic.json",
        class_to_id=vocabulary,
        aliases=manifest["aliases"],
    )
    for method in methods:
        target = compact / f"room0_{method}.json"
        if target.exists():
            continue
        with np.load(output / f"{method}.npz") as data:
            owners, ids, confidence = (
                data["owner_ids"],
                data["semantic_ids"],
                data["semantic_confidence"],
            )
            if not np.array_equal(data["source_indices"], source):
                raise ValueError("source rows changed")
            if method != methods[3] and not np.array_equal(owners, native["owner_ids"]):
                raise ValueError("original owner control changed owners")
            missing = ~data["feature_covered"]
            if not np.array_equal(
                ids[missing], native["semantic_ids"][missing]
            ) or not np.array_equal(
                confidence[missing], native["semantic_confidence"][missing]
            ):
                raise ValueError("missing feature changed native semantic fallback")
        if method == methods[3]:
            with np.load(room / "consensus/INST_CONSENSUS_OWNER.npz") as data:
                if not np.array_equal(owners, data["owner_ids"]):
                    raise ValueError("re-estimation changed consensus owners")
        info = []
        for owner, rows in _owner_row_groups(owners).items():
            values = ids[rows]
            values = values[values > 0]
            if owner > 0 and len(values):
                info.append(
                    EntityEvaluationInfo(
                        int(owner),
                        int(np.bincount(values).argmax()),
                        2,
                        float(confidence[rows].mean()),
                    )
                )
        mesh = LabeledMesh(
            xyz,
            np.empty((0, 3), np.int64),
            np.zeros_like(xyz),
            ids,
            owners,
            confidence,
            (native["owner_ids"] > 0).astype(np.float32),
        )
        measured = evaluate_replica_voxel_map(
            mesh,
            truth,
            info,
            valid_semantic_ids=classes,
            instance_semantic_ids={
                i for c, i in vocabulary.items() if c not in NON_INSTANCE_CLASSES
            },
            min_instance_vertices=100,
            distance_threshold_m=0.05,
        )
        record = json.loads((output / f"{method}_prediction.json").read_text())
        record.update(
            status="FULL_MAP_EVALUATED",
            metrics=measured,
            instance_ranking="existing class-agnostic size ranking; semantic-constrained AP is diagnostic only",
        )
        _atomic_json(target, record)
        print(
            method,
            {k: measured[k] for k in ("miou", "f_miou", "ap50", "f5")},
            flush=True,
        )


if __name__ == "__main__":
    main()
