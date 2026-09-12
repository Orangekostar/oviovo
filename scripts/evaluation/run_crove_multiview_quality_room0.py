"""Same-cache k=4 diverse/quality M1 controls; static current-aware is an alias."""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
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
from src.oviv2.ovi_surface_attributes import project_world_points
from src.oviv2.surface_multiview_semantics import aggregate_views, diverse_views
from src.oviv2.surface_view_bank import project_source_support


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    case = json.loads(path(config["source_config"]).read_text())["cases"][
        "replica_room0_static"
    ]
    manifest = json.loads(path(case["benchmark_manifest"]).read_text())
    classes = manifest["vocabulary"]["classes"]
    run = path(config["run_root"]) / "dev/room0"
    output = run / "native_diverse_quality"
    output.mkdir(parents=True, exist_ok=True)
    compact = path(config["compact_output_root"])
    methods = ["MV_DIVERSE_MEAN", "MV_QUALITY"]
    settings = {
        "case": "room0",
        "top_k": 4,
        "candidate_pool": "identical retained native six-crop view feature bank as MV_TOPK_MEAN",
        "selection": "highest native visible area seed; angular farthest-first around physical owner centroid; ties by area then cached order",
        "quality": "depth_consistent_unique_pixels / projectable_unique_pixels * sqrt(min(1, consistent_unique_pixels / native_visible_area)) * truncation_factor",
        "truncation_factor": "0.5 if cached crop bounding box touches image boundary; otherwise 1",
        "depth_tolerance_m": 0.05,
        "minimum_unique_pixels": 16,
        "no_positive_quality": "native semantic fallback, never uniform pseudo-evidence",
        "features": "normalize cached six-crop mean per view, aggregate, normalize",
        "text": "shared original SigLIP bare complete Replica41 class text cache",
        "new_encoder_forwards": 0,
        "methods": methods,
        "static_current_aware": "identical to MV_QUALITY; one visit, no temporal policy branch; reuse exact prediction",
        "scope": "partial M1; per-crop features, dynamic and structural patch evidence remain separate obligations",
    }
    registry = compact / "multiview_quality_room0_registry.json"
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("frozen view policy changed")
    else:
        with registry.open("x") as f:
            json.dump(settings, f, indent=2)
    with np.load(
        path(
            "$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/replica_room0_static_full_v1/current_map/current_surface.npz"
        )
    ) as d:
        xyz, owners, source = (
            d["vertices_xyz"],
            d["owner_entity_ids"],
            d["source_vertex_indices"],
        )
        if not d["current_valid"].all() or np.any(d["source_visit_ids"] != 0):
            raise ValueError("static source state changed")
    with np.load(run / "native_cached_batch/B_SEM_OVI_NATIVE.npz") as d:
        if not np.array_equal(d["source_indices"], source) or not np.array_equal(
            d["owner_ids"], owners
        ):
            raise ValueError("native source mapping differs")
        base_ids, base_conf = d["semantic_ids"], d["semantic_confidence"]
    with np.load(run / "native_cached_batch/text_features.npz") as d:
        text, scale = d["text"], float(d["logit_scale"])
    with (
        path(case["fine_surface"]).parent
        / "inst_sem_siglip-l-16-384_200_incre_combine.pkl"
    ).open("rb") as f:
        bank = pickle.load(f)
    groups = _owner_row_groups(owners)
    selected, jobs = {}, defaultdict(list)
    start = time.monotonic()
    for owner, entry in sorted(bank.items()):
        if owner not in groups or len(entry["frame_id"]) < 2:
            continue
        frames = np.asarray(entry["frame_id"], np.int64)
        if len(np.unique(frames)) != len(frames) or np.any(
            (frames < 0) | (frames >= 2000) | (frames % 10 != 0)
        ):
            raise ValueError("duplicate or unauthorized source frame")
        physical = np.unique(xyz[groups[owner]], axis=0)
        center = physical.mean(axis=0, dtype=np.float64)
        directions = np.asarray(entry["pose"])[:, :3, 3] - center
        indices = diverse_views(directions, entry["vis_area"], k=4)
        selected[owner] = indices
        for i in indices:
            jobs[int(frames[i])].append((owner, int(i)))
    selection_seconds = time.monotonic() - start
    quality_cache = output / "geometric_quality.json"
    if quality_cache.exists():
        quality_record = json.loads(quality_cache.read_text())
    else:
        dataset = ReplicaRoom0Dataset(path(case["rgb_projection"]["dataset_root"]))
        evidence = {}
        start = time.monotonic()
        for frame_id, entries in sorted(jobs.items()):
            frame = dataset[frame_id]
            h, w = frame.depth.shape
            for owner, i in entries:
                entry = bank[owner]
                if not np.allclose(frame.pose, entry["pose"][i], rtol=1e-5, atol=1e-5):
                    raise ValueError("pose and cached feature disagree")
                rows = groups[owner]
                _, pixels = project_source_support(xyz, rows, frame)
                support = len(np.unique(pixels))
                projection = project_world_points(
                    xyz[rows], frame.pose, frame.intrinsics
                )
                visible = projection.projectable
                projected_count = len(
                    np.unique(
                        projection.rows[visible] * w + projection.columns[visible]
                    )
                )
                box = np.asarray(entry["box_2d"][i])
                truncated = bool(
                    box[0] <= 0 or box[1] <= 0 or box[2] >= w - 1 or box[3] >= h - 1
                )
                depth_fraction = min(1.0, support / max(1, projected_count))
                mask_support = min(1.0, support / max(1.0, float(entry["vis_area"][i])))
                weight = (
                    depth_fraction * np.sqrt(mask_support) * (0.5 if truncated else 1.0)
                    if support >= 16
                    else 0.0
                )
                evidence[f"{owner}:{i}"] = {
                    "frame": frame_id,
                    "unique_support": support,
                    "unique_projectable": projected_count,
                    "depth_fraction": depth_fraction,
                    "mask_support_fraction": mask_support,
                    "truncated": truncated,
                    "weight": float(weight),
                }
            print(f"quality frame {frame_id}", flush=True)
        quality_record = {
            "evidence": evidence,
            "projection_seconds": time.monotonic() - start,
        }
        with quality_cache.open("x") as f:
            json.dump(quality_record, f, indent=2)
    evidence = quality_record["evidence"]
    for method in methods:
        target = output / f"{method}.npz"
        if target.exists():
            continue
        start = time.monotonic()
        ids, conf = base_ids.copy(), base_conf.copy()
        covered = np.zeros(len(ids), bool)
        feature_owners, posteriors, references = [], [], []
        for owner, indices in selected.items():
            entry = bank[owner]
            weights = (
                np.ones(len(indices))
                if method == "MV_DIVERSE_MEAN"
                else np.array([evidence[f"{owner}:{i}"]["weight"] for i in indices])
            )
            z = aggregate_views(np.asarray(entry["feat"])[indices], weights)
            references.append(
                {
                    "owner": int(owner),
                    "frames": np.asarray(entry["frame_id"])[indices].tolist(),
                    "weights": weights.tolist(),
                    "fallback": z is None,
                }
            )
            if z is None:
                continue
            logits = scale * (z @ text.T)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            rows = groups[owner]
            ids[rows], conf[rows], covered[rows] = int(p.argmax()) + 1, p.max(), True
            feature_owners.append(owner)
            posteriors.append(p)
        atomic_npz(
            target,
            semantic_ids=ids,
            semantic_confidence=conf,
            owner_ids=owners,
            source_indices=source,
            feature_covered=covered,
            feature_owners=np.asarray(feature_owners),
            owner_posterior=np.asarray(posteriors),
            class_ids=np.arange(1, len(classes) + 1),
        )
        record = {
            "observations": references,
            "feature_coverage": float(covered.mean()),
            "prediction_seconds": time.monotonic() - start,
            "selection_seconds": selection_seconds,
            "projection_seconds_shared": quality_record["projection_seconds"],
        }
        with (output / f"{method}_views.json").open("x") as f:
            json.dump(record, f, indent=2)
        print(method, "full prediction saved", flush=True)
    # All predictions are frozen before target GT is opened.
    scene = next(s for s in manifest["scenes"] if s["scene"] == "room0")
    gt = path(manifest["ground_truth_root"]) / scene["ground_truth_scene"] / "habitat"
    vocabulary = {c: i + 1 for i, c in enumerate(classes)}
    truth = load_replica_ground_truth(
        gt / "mesh_semantic.ply",
        gt / "info_semantic.json",
        class_to_id=vocabulary,
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
    for method in methods:
        target = compact / f"room0_{method}.json"
        if target.exists():
            continue
        with np.load(output / f"{method}.npz") as d:
            mesh = LabeledMesh(
                xyz,
                np.empty((0, 3), np.int64),
                np.zeros_like(xyz),
                d["semantic_ids"],
                owners,
                d["semantic_confidence"],
                (owners > 0).astype(np.float32),
            )
            coverage = float(d["feature_covered"].mean())
        measured = evaluate_replica_voxel_map(
            mesh,
            truth,
            info,
            valid_semantic_ids=vocabulary.values(),
            instance_semantic_ids={
                i for c, i in vocabulary.items() if c not in NON_INSTANCE_CLASSES
            },
            min_instance_vertices=int(case["min_instance_vertices"]),
            distance_threshold_m=0.05,
        )
        record = {
            "case": "room0",
            "split": "dev",
            "method": method,
            "status": "FULL_MAP_EVALUATED",
            "geometry_fixed": True,
            "owner_changed": False,
            "feature_coverage": coverage,
            "metrics": measured,
            "new_encoder_forwards": 0,
            "prediction": str(
                (output / f"{method}.npz").relative_to(path(config["run_root"]))
            ),
            "semantic_constrained_AP": "diagnostic with frozen native entity-info",
        }
        with target.open("x") as f:
            json.dump(record, f, indent=2, allow_nan=False)
        print(
            method,
            {k: measured[k] for k in ("miou", "f_miou", "ap50", "f5")},
            flush=True,
        )
    alias = compact / "room0_MV_CURRENT_AWARE.json"
    if not alias.exists():
        record = json.loads((compact / "room0_MV_QUALITY.json").read_text())
        record.update(
            method="MV_CURRENT_AWARE",
            prediction_alias="MV_QUALITY",
            reason="static single-visit input: current-aware policy reduces exactly to quality aggregation; no new inference",
        )
        with alias.open("x") as f:
            json.dump(record, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
