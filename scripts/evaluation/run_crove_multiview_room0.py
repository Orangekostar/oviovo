"""First full-map room0 readouts using real cached native six-crop features.

This is an initial DEV batch, not completion of M1: quality/current-aware and
other scenes still require the shared projection bank. All rows keep XYZ/owner.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_crove_fine_current_map import (
    NON_INSTANCE_CLASSES,
    EntityEvaluationInfo,
    LabeledMesh,
    _owner_semantic_arrays,
    bind_ovi_owner_ids,
    evaluate_replica_voxel_map,
    load_native_ovi_surface,
    load_ovimap_semantic_mapping,
    load_replica_ground_truth,
    parse_instance_color_log,
)


def path(value):
    p = Path(os.path.expandvars(str(value)))
    return p if p.is_absolute() else ROOT / p


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    fine_config = json.loads(path(config["source_config"]).read_text())
    case = fine_config["cases"]["replica_room0_static"]
    manifest = json.loads(path(case["benchmark_manifest"]).read_text())
    classes = manifest["vocabulary"]["classes"]
    output = path(config["run_root"]) / "dev/room0/native_cached_batch"
    output.mkdir(parents=True, exist_ok=True)
    compact = path(config["compact_output_root"])
    methods = {
        "B_SEM_OVI_NATIVE": "saved native class crosswalk, no new encoding",
        "MV_NATIVE_CACHED": "native retained last 8 views, visibility-weighted raw six-crop means, fixed 41-class text",
        "MV_SINGLE": "highest visibility one view; normalized six-crop mean",
        "MV_TOPK_MEAN": "highest visibility four views; normalized six-crop means, uniform aggregation",
    }
    registry = compact / "initial_room0_batch_registry.json"
    if not registry.exists():
        registry.parent.mkdir(parents=True, exist_ok=True)
        with registry.open("x") as f:
            json.dump(
                {
                    "methods": methods,
                    "candidate_pool": "native retained observations only",
                    "text": "bare complete Replica41 class names as official TextEmbedder",
                    "scope": "partial_M1_DEV_batch",
                    "new_image_forwards": 0,
                },
                f,
                indent=2,
            )
    print("Loading frozen room0 surface and native observations", flush=True)
    fine = load_native_ovi_surface(path(case["fine_surface"]))
    owners = bind_ovi_owner_ids(
        fine.palette_rgb, parse_instance_color_log(path(case["instance_color_log"]))
    )
    mapping = load_ovimap_semantic_mapping(
        path(case["ovimap_semantic_mapping"]),
        source_vocabulary=tuple(case["ovimap_source_vocabulary"]),
        target_vocabulary=tuple(classes),
    )
    base_ids, base_conf = _owner_semantic_arrays(owners, mapping)
    bank_path = (
        path(case["fine_surface"]).parent
        / "inst_sem_siglip-l-16-384_200_incre_combine.pkl"
    )
    with bank_path.open("rb") as f:
        bank = pickle.load(f)
    model_path = path(
        "$HOME/oviovo_baseline_builds/ovimap-ubuntu24-native/siglip-large-patch16-384"
    )
    text_cache = output / "text_features.npz"
    if not text_cache.exists():
        model = (
            AutoModel.from_pretrained(model_path, local_files_only=True)
            .eval()
            .to("cuda:0")
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokens = tokenizer(
            classes, padding="max_length", max_length=64, return_tensors="pt"
        ).to("cuda:0")
        with torch.inference_mode():
            text = model.get_text_features(**tokens)
            if not isinstance(text, torch.Tensor):
                text = text.pooler_output
            text = torch.nn.functional.normalize(text, dim=-1).cpu().numpy()
            scale = float(model.logit_scale.exp().cpu())
        with text_cache.open("xb") as f:
            np.savez(f, text=text, logit_scale=scale)
        del model
        torch.cuda.empty_cache()
    with np.load(text_cache) as cache:
        text, scale = cache["text"], float(cache["logit_scale"])
    for method in methods:
        target = output / f"{method}.npz"
        if target.exists():
            continue
        started = time.monotonic()
        ids, confidence = base_ids.copy(), base_conf.copy()
        feature_owners, distributions, chosen_refs = [], [], []
        covered = np.zeros(len(ids), bool)
        if method != "B_SEM_OVI_NATIVE":
            for owner, entry in sorted(bank.items()):
                frames = np.asarray(entry["frame_id"], dtype=np.int64)
                if np.any((frames < 0) | (frames >= 2000) | (frames % 10 != 0)):
                    raise ValueError("cached view outside authorized window")
                if len(set(frames)) != len(frames):
                    raise ValueError(
                        "duplicate native frame needs explicit deduplication"
                    )
                if len(frames) < 2:
                    continue
                area = np.asarray(entry["vis_area"], dtype=np.float32)
                features = np.asarray(entry["feat"], dtype=np.float32)
                if method == "MV_NATIVE_CACHED":
                    selected = np.arange(max(0, len(frames) - 8), len(frames))
                    weights = area[selected] / area[selected].sum()
                    z = (features[selected] * weights[:, None]).sum(0)
                else:
                    k = 1 if method == "MV_SINGLE" else 4
                    selected = np.argsort(-area, kind="stable")[:k]
                    normalized = features[selected] / np.maximum(
                        np.linalg.norm(features[selected], axis=1, keepdims=True), 1e-12
                    )
                    z = normalized.mean(0)
                z = z / max(float(np.linalg.norm(z)), 1e-12)
                logits = scale * (z @ text.T)
                posterior = np.exp(logits - logits.max())
                posterior /= posterior.sum()
                label = int(posterior.argmax()) + 1
                rows = owners == owner
                ids[rows] = label
                confidence[rows] = posterior.max()
                covered[rows] = True
                feature_owners.append(owner)
                distributions.append(posterior)
                chosen_refs.append(
                    {"owner": int(owner), "frames": frames[selected].tolist()}
                )
        with target.open("xb") as f:
            np.savez_compressed(
                f,
                semantic_ids=ids,
                semantic_confidence=confidence,
                owner_ids=owners,
                source_indices=fine.source_vertex_indices,
                class_ids=np.arange(1, len(classes) + 1),
                feature_owners=np.array(feature_owners),
                owner_posterior=np.asarray(distributions),
                feature_covered=covered,
            )
        with (output / f"{method}_views.json").open("x") as f:
            json.dump(
                {
                    "observations": chosen_refs,
                    "prediction_seconds": time.monotonic() - started,
                    "feature_coverage": float(covered.mean()),
                },
                f,
            )
        print(method, "prediction saved", flush=True)
    # Open target geometry only after all batch predictions are frozen on disk.
    scene = next(s for s in manifest["scenes"] if s["scene"] == "room0")
    gt = path(manifest["ground_truth_root"]) / scene["ground_truth_scene"] / "habitat"
    class_ids = {name: i + 1 for i, name in enumerate(classes)}
    truth = load_replica_ground_truth(
        gt / "mesh_semantic.ply",
        gt / "info_semantic.json",
        class_to_id=class_ids,
        aliases=manifest["aliases"],
    )
    entity_info = [
        EntityEvaluationInfo(int(owner), mapping[owner][0], 2, mapping[owner][1])
        for owner in np.unique(owners)
        if mapping.get(owner, (0, 0))[0] > 0
    ]
    for method, implementation in methods.items():
        target = compact / f"room0_{method}.json"
        if target.exists():
            continue
        with np.load(output / f"{method}.npz") as prediction:
            mesh = LabeledMesh(
                fine.vertices_xyz,
                np.empty((0, 3), np.int64),
                np.zeros_like(fine.vertices_xyz),
                prediction["semantic_ids"],
                prediction["owner_ids"],
                prediction["semantic_confidence"],
                (owners > 0).astype(np.float32),
            )
        started = time.monotonic()
        measured = evaluate_replica_voxel_map(
            mesh,
            truth,
            entity_info,
            valid_semantic_ids=class_ids.values(),
            instance_semantic_ids={
                i for c, i in class_ids.items() if c not in NON_INSTANCE_CLASSES
            },
            min_instance_vertices=int(case["min_instance_vertices"]),
            distance_threshold_m=0.05,
        )
        record = {
            "case": "room0",
            "split": "dev",
            "method": method,
            "implementation": implementation,
            "status": "FULL_MAP_EVALUATED",
            "geometry_fixed": True,
            "owner_changed": False,
            "metrics": measured,
            "evaluation_seconds": time.monotonic() - started,
            "semantic_constrained_AP": "diagnostic with frozen native entity-info; not reselected by new label",
        }
        with target.open("x") as f:
            json.dump(record, f, indent=2, allow_nan=False)
        print(
            method,
            {k: measured[k] for k in ["miou", "f_miou", "ap25", "ap50", "f5"]},
            flush=True,
        )


if __name__ == "__main__":
    main()
