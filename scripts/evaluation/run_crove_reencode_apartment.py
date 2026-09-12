"""Matched six-crop original-owner controls and consensus-owner semantic re-estimation."""

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
from scripts.evaluation.run_crove_fine_current_map import _owner_row_groups
from scripts.evaluation.run_crove_multiview_apartment import evaluate_saved_readouts
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.datasets.tesse_cd import TesseCdRgbdDataset
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.oviv2.ovi_surface_attributes import project_world_points
from src.oviv2.surface_multiview_semantics import (
    aggregate_views,
    diverse_views,
    native_crop_inputs,
    native_six_crops,
    owner_posteriors_to_rows,
)
from src.oviv2.surface_readout import resolve_semantic_update
from src.oviv2.surface_view_bank import project_source_support


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    room = path(config["run_root"]) / "dev/apartment"
    output = room / "reencoded_regions"
    output.mkdir(exist_ok=True)
    methods = [
        "B_SEM_OVI_REENCODE",
        "MV_REENCODE_DIVERSE",
        "MV_REENCODE_QUALITY",
        "INST_CONSENSUS_RESEM",
    ]
    settings = {
        "case": pair["pair_id"],
        "states": ["B3", "H2"],
        "methods": methods,
        "k": 4,
        "encoder": "same native SigLIP large patch16 384 and shared complete public 20-class text",
        "preprocessing": "SiglipImageProcessor slow PIL resize, explicit channels_last RGB axis; shared by every control",
        "crops": "native three box expansions 0/.1/.2, each original RGB and black-background mask; normalize every crop before within-view mean",
        "mask": "own-state current source projection, measured depth within 0.05 m, no dilation",
        "candidate_views": "all authorized own-visit frames with at least 16 distinct independent-mask-interior representative pixels",
        "selection": "native control top four visible views; other methods same angular farthest-first rule with maximum visibility seed",
        "quality": "depth-consistent / projectable unique pixels * sqrt(min(1,support/union area of touched independent mask IDs)) * image-truncation factor .5 or 1",
        "area_adaptation": "new owners have no native visible-area field; use touched independent-mask union area for BOTH original and consensus owner controls",
        "aggregation": "native visibility-weighted raw six-crop means; diverse normalized-view uniform mean; quality/resem normalized-view quality mean",
        "cross_visit_union": False,
        "minimum_feature_views": 2,
        "missing": "KEEP_SOURCE label, confidence and role; no invented unknown",
        "GT_read_before_all_predictions": False,
    }
    registry = (
        path(config["compact_output_root"])
        / "reencoded_regions_apartment_registry.json"
    )
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("frozen re-encoding policy changed")
    else:
        _atomic_json(registry, settings)
    mask_registry = json.loads(
        (
            path(config["compact_output_root"])
            / "independent_mask_bank_apartment_registry.json"
        ).read_text()
    )
    bank_status = json.loads(
        (
            path(config["compact_output_root"])
            / "independent_mask_bank_apartment_status.json"
        ).read_text()
    )
    frames = [
        f
        for visit in (0, 1)
        for f in range(
            pair["visit_frame_ranges"][f"t{visit}"][0],
            pair["visit_frame_ranges"][f"t{visit}"][1] + 1,
        )
    ]
    if any(bank_status[s]["completed_frames"] != frames for s in ("B3", "H2")):
        raise ValueError("complete independent frame banks required")
    dataset = TesseCdRgbdDataset(
        path(pair["rgbd_root"]),
        pair["scene"],
        path(pair["rgbd_export_manifest"]),
        path(pair["causal_schedule"]),
    )
    crosswalk = load_tesse_semantic_crosswalk(
        path(pair["semantic_aliases"]),
        pair["scene"],
        path(pair["semantic_label_space"]),
    )
    classes = np.array(
        sorted({v.semantic_id for v in crosswalk.aliases.values() if v.matched})
    )
    with np.load(room / "native_cached_batch/text_features.npz") as data:
        if not np.array_equal(classes, data["class_ids"]):
            raise ValueError("text vocabulary changed")
        text, scale = data["text"], float(data["logit_scale"])
    with np.load(path(pair["current_map_root"]) / "current_surface.npz") as data:
        xyz, visits = data["vertices_xyz"], data["source_visit_ids"]
    model_path = path(
        "$HOME/oviovo_baseline_builds/ovimap-ubuntu24-native/siglip-large-patch16-384"
    )
    model_files = sorted(model_path.glob("*.safetensors")) or sorted(
        model_path.glob("pytorch_model*.bin")
    )
    if not model_files:
        raise ValueError("no local encoder weights")
    model_binding = {p.name: file_hash(p) for p in model_files}
    text_binding = file_hash(room / "native_cached_batch/text_features.npz")
    model = processor = None
    total_forwards, total_projection, total_encoding = 0, 0.0, 0.0
    for state in ("B3", "H2"):
        with np.load(
            room / "native_cached_batch" / f"{state}_MV_NATIVE_CACHED.npz"
        ) as data:
            native_source, native_owners = data["source_indices"], data["owner_ids"]
            native_ids, native_roles = data["semantic_ids"], data["eval_role"]
        with np.load(
            room / "independent_mask_bank" / state / "representatives.npz"
        ) as data:
            representative_rows = data["source_rows"]
        representative_local = np.searchsorted(native_source, representative_rows)
        if not np.array_equal(native_source[representative_local], representative_rows):
            raise ValueError("representatives not on frozen current source")
        for partition in ("native", "consensus"):
            variants = methods[:3] if partition == "native" else methods[3:]
            root = output / state / partition
            root.mkdir(parents=True, exist_ok=True)
            baseline_file = (
                room
                / ("native_cached_batch" if partition == "native" else "consensus")
                / f"{state}_{'MV_NATIVE_CACHED' if partition == 'native' else 'INST_CONSENSUS_OWNER'}.npz"
            )
            with np.load(baseline_file) as data:
                baseline = {
                    key: data[key] for key in data.files if key != "patch_candidate"
                }
            source, owners = baseline["source_indices"], baseline["owner_ids"]
            if (
                not np.array_equal(source, native_source)
                or not np.array_equal(baseline["semantic_ids"], native_ids)
                or not np.array_equal(baseline["eval_role"], native_roles)
            ):
                raise ValueError("semantic or state baseline changed")
            if partition == "native" and not np.array_equal(owners, native_owners):
                raise ValueError("original owners changed")
            binding_file = root / "input_binding.json"
            binding = {
                "prediction_sha256": file_hash(baseline_file),
                "model_weight_sha256": model_binding,
                "text_sha256": text_binding,
                "mask_registry_sha256": file_hash(
                    path(config["compact_output_root"])
                    / "independent_mask_bank_apartment_registry.json"
                ),
            }
            if (
                binding_file.exists()
                and json.loads(binding_file.read_text()) != binding
            ):
                raise ValueError("re-encoding cache input changed")
            _atomic_json(binding_file, binding)
            groups = _owner_row_groups(owners)
            node_owners = owners[representative_local]
            candidates = defaultdict(list)
            for frame_id in frames:
                visit = int(frame_id >= pair["visit_frame_ranges"]["t1"][0])
                local_frame = frame_id - pair["visit_frame_ranges"][f"t{visit}"][0]
                mask_path = path(
                    f"$HOME/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/native-runs/apartment/t{visit}/2f638911509f-apartment-t{visit}/frontend/frame{local_frame:06d}.png"
                )
                digest = file_hash(mask_path)
                with np.load(
                    room / "independent_mask_bank" / state / f"{frame_id:06d}.npz"
                ) as data:
                    if (
                        digest != str(data["mask_sha256"])
                        or digest != mask_registry["mask_sha256"][str(frame_id)]
                    ):
                        raise ValueError("independent mask changed")
                    observed_owners, pixels, mask_ids = (
                        node_owners[data["node_ids"]],
                        data["pixels"],
                        data["mask_ids"],
                    )
                areas = np.bincount(np.asarray(Image.open(mask_path)).ravel())
                for owner, rows in _owner_row_groups(observed_owners).items():
                    if owner <= 0:
                        continue
                    count = len(np.unique(pixels[rows]))
                    if count >= 16:
                        candidates[int(owner)].append(
                            (
                                frame_id,
                                count,
                                int(areas[np.unique(mask_ids[rows])].sum()),
                            )
                        )
            jobs, selected = defaultdict(list), {}
            for owner, observations in sorted(candidates.items()):
                if len(observations) < 2:
                    continue
                rows = source[groups[owner]]
                if len(np.unique(visits[rows])) != 1:
                    raise ValueError("region crosses visits")
                center = np.unique(xyz[rows], axis=0).mean(0, dtype=np.float64)
                observed_frames = np.array([v[0] for v in observations])
                visibility = np.array([v[1] for v in observations])
                directions = (
                    np.array(
                        [
                            dataset.records[f].camera_to_world[:3, 3]
                            for f in observed_frames
                        ]
                    )
                    - center
                )
                diverse = diverse_views(directions, visibility, k=4)
                top = np.argsort(-visibility, kind="stable")[:4]
                chosen = np.union1d(diverse, top) if partition == "native" else diverse
                selected[owner] = {
                    "top": [observations[i][0] for i in top],
                    "diverse": [observations[i][0] for i in diverse],
                }
                for index in chosen:
                    frame_id, visibility, area = observations[index]
                    jobs[frame_id].append((owner, visibility, area))
            _atomic_json(
                root / "selected_views.json", {str(k): v for k, v in selected.items()}
            )
            print(
                state,
                partition,
                len(selected),
                "regions;",
                len(jobs),
                "frames;",
                sum(len(v) for v in jobs.values()),
                "region-view crop batches",
                flush=True,
            )
            features = {}
            for frame_id, regions in sorted(jobs.items()):
                target = root / f"{frame_id:06d}.npz"
                if not target.exists():
                    frame = dataset[frame_id]
                    (
                        out_owners,
                        embeddings,
                        qualities,
                        visible_counts,
                        mask_pixels,
                        offsets,
                    ) = [], [], [], [], [], [0]
                    for owner, visibility, area in regions:
                        started = time.monotonic()
                        rows = source[groups[owner]]
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
                        total_projection += time.monotonic() - started
                        if model is None:
                            processor = AutoProcessor.from_pretrained(
                                model_path, local_files_only=True, use_fast=False
                            )
                            model = (
                                AutoModel.from_pretrained(
                                    model_path, local_files_only=True
                                )
                                .eval()
                                .to("cuda:1")
                            )
                        inputs = native_crop_inputs(processor, crops).to("cuda:1")
                        started = time.monotonic()
                        with torch.inference_mode():
                            z = model.get_image_features(**inputs)
                            z = z / z.norm(dim=-1, keepdim=True)
                            z = z.mean(0).cpu().numpy()
                        total_encoding += time.monotonic() - started
                        total_forwards += 1
                        out_owners.append(owner)
                        embeddings.append(z)
                        qualities.append(q)
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
                if frame_id % 32 == 0:
                    print(state, partition, frame_id, "encoded/cached", flush=True)
            for method in variants:
                feature_owners, posterior = [], []
                for owner, indices in sorted(selected.items()):
                    mode = "top" if method == "B_SEM_OVI_REENCODE" else "diverse"
                    values = [
                        features[(owner, f)]
                        for f in indices[mode]
                        if (owner, f) in features
                    ]
                    if len(values) < 2:
                        continue
                    z, quality, visibility = map(np.array, zip(*values))
                    if method == "B_SEM_OVI_REENCODE":
                        embedding = (z * visibility[:, None]).sum(0)
                        embedding /= np.linalg.norm(embedding)
                    else:
                        embedding = aggregate_views(
                            z,
                            np.ones(len(z))
                            if method == "MV_REENCODE_DIVERSE"
                            else quality,
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
                ids, roles = resolve_semantic_update(
                    native_ids,
                    native_roles,
                    proposed,
                    covered.astype(np.uint8),
                    crosswalk,
                )
                confidence[~covered] = baseline["semantic_confidence"][~covered]
                prediction = dict(baseline)
                prediction.update(
                    semantic_ids=ids,
                    semantic_confidence=confidence,
                    eval_role=roles,
                    feature_covered=covered,
                    semantic_update_kind=covered.astype(np.uint8),
                )
                atomic_npz(output / f"{state}_{method}.npz", **prediction)
                atomic_npz(
                    output / f"{state}_{method}_owner_features.npz",
                    feature_owners=np.array(feature_owners),
                    owner_posterior=posterior,
                    class_ids=classes,
                )
                _atomic_json(
                    output / f"{state}_{method}_prediction.json",
                    {
                        "state": state,
                        "method": method,
                        "source_rows": len(source),
                        "feature_coverage": float(covered.mean()),
                        "feature_owner_count": len(feature_owners),
                        "changed_semantic_rows": int(
                            np.count_nonzero(ids != native_ids)
                        ),
                        "changed_role_rows": int(
                            np.count_nonzero(roles != native_roles)
                        ),
                        "geometry_fixed": True,
                        "owner_fixed": partition == "native",
                        "owner_partition": partition,
                        "confidence": "new normalized full-class posterior; missing features keep old confidence",
                    },
                )
                print(state, method, "full prediction saved", flush=True)
            del baseline, groups, features
    _atomic_json(
        output / "execution_cost.json",
        {
            "new_six_crop_forward_batches": total_forwards,
            "new_crop_images": 6 * total_forwards,
            "projection_seconds": total_projection,
            "encoding_seconds": total_encoding,
            "scope": "this execution only, excludes model loading, cache I/O, selection and evaluation; cached observations counted in feature bank",
        },
    )
    evaluate_saved_readouts(config, state_config, pair, crosswalk, output, methods[:3])
    evaluate_saved_readouts(
        config,
        state_config,
        pair,
        crosswalk,
        output,
        methods[3:],
        allow_owner_changes=True,
    )


if __name__ == "__main__":
    main()
