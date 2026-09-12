"""Same-cache diverse/quality M1 controls for frozen Apartment B3/H2."""

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

from scripts.evaluation.build_crove_room0_mask_bank import file_hash
from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_crove_fine_current_map import (
    _entity_records,
    _owner_row_groups,
)
from scripts.evaluation.run_crove_multiview_apartment import (
    apply_and_evaluate_owner_features,
)
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.datasets.tesse_cd import TesseCdRgbdDataset
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.oviv2.ovi_surface_attributes import project_world_points
from src.oviv2.surface_multiview_semantics import aggregate_views, diverse_views
from src.oviv2.surface_view_bank import project_source_support


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    crosswalk = load_tesse_semantic_crosswalk(
        path(pair["semantic_aliases"]),
        pair["scene"],
        path(pair["semantic_label_space"]),
    )
    classes = np.array(
        sorted({v.semantic_id for v in crosswalk.aliases.values() if v.matched}),
        np.int32,
    )
    run = path(config["run_root"])
    room = run / "dev/apartment"
    output = room / "native_diverse_quality"
    cache = output / "quality_bank"
    cache.mkdir(parents=True, exist_ok=True)
    methods = ["MV_DIVERSE_MEAN", "MV_QUALITY"]
    settings = {
        "scene": pair["scene"],
        "states": ["B3", "H2"],
        "methods": methods,
        "top_k": 4,
        "selection": "same room0 angular farthest-first, visible-area seed; original source-owner physical centroid; identical selected views in B3/H2",
        "quality": "same room0 depth-consistent / projectable unique pixels * sqrt(min(1, support / native visible area)) * truncation factor; only own-state valid source rows",
        "minimum_unique_pixels": 16,
        "depth_tolerance_m": 0.05,
        "truncation_factor": "0.5 when original crop bbox touches image boundary, otherwise 1",
        "features": "same retained native six-crop means and shared SigLIP text as initial Apartment batch",
        "missing_quality": "KEEP_SOURCE, no uniform pseudo-evidence",
        "frame_windows": pair["visit_frame_ranges"],
        "new_image_encoder_forwards": 0,
        "current_aware_status": "separate cross-visit current-evidence policy still pending",
    }
    registry = (
        path(config["compact_output_root"])
        / "multiview_quality_apartment_registry.json"
    )
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("frozen quality policy changed")
    else:
        _atomic_json(registry, settings)
    with np.load(path(pair["current_map_root"]) / "current_surface.npz") as data:
        xyz, owners = data["vertices_xyz"], data["owner_entity_ids"]
        visits = data["source_visit_ids"]
    groups = _owner_row_groups(owners)
    states = []
    for state in ("B3", "H2"):
        with np.load(run / "bridge" / f"{state}_legacy_prediction.npz") as data:
            rows = data["source_indices"]
            if not np.array_equal(data["owner_ids"], owners[rows]):
                raise ValueError("bridge owner binding changed")
        current = np.zeros(len(xyz), bool)
        current[rows] = True
        states.append(current)
    binding = json.loads(
        (room / "native_cached_batch/native_feature_binding.json").read_text()
    )
    bank, selected, jobs, owner_confidence = {}, {}, defaultdict(list), {}
    start_time = time.monotonic()
    for visit, record_key in enumerate(("b0_entities", "b2_entities")):
        records = _entity_records(path(pair[record_key]))
        offset = int(pair["t1_owner_offset"]) if visit else 0
        owner_confidence.update(
            {
                key + offset: float(record["semantic_score"])
                for key, record in records.items()
            }
        )
        native_root = path(
            f"$HOME/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/native-runs/apartment/t{visit}/2f638911509f-apartment-t{visit}"
        )
        manifest_path = native_root / "native_mapping_manifest.json"
        if file_hash(manifest_path) != binding[visit]["native_manifest_sha256"]:
            raise ValueError("native manifest differs from validated binding")
        manifest = json.loads(manifest_path.read_text())
        feature = Path(manifest["artifacts"]["semantic_features"]["path"])
        if file_hash(feature) != binding[visit]["feature_sha256"]:
            raise ValueError("native features changed")
        with feature.open("rb") as handle:
            native = pickle.load(handle)
        start, stop = pair["visit_frame_ranges"][f"t{visit}"]
        for local_owner, entry in sorted(native.items()):
            owner = int(local_owner) + offset
            if (
                owner not in groups
                or int(local_owner) not in records
                or len(entry["frame_id"]) < 2
            ):
                continue
            if not np.all(visits[groups[owner]] == visit):
                raise ValueError("owner crosses visit boundary")
            frames = np.asarray(entry["frame_id"], np.int64)
            if len(np.unique(frames)) != len(frames) or np.any(
                (frames < 0) | (frames > stop - start)
            ):
                raise ValueError("duplicate or unauthorized frames")
            center = np.unique(xyz[groups[owner]], axis=0).mean(0, dtype=np.float64)
            indices = diverse_views(
                np.asarray(entry["pose"])[:, :3, 3] - center, entry["vis_area"], k=4
            )
            bank[owner] = entry
            selected[owner] = (indices, frames + start, visit)
            for index in indices:
                jobs[int(frames[index]) + start].append((owner, int(index)))
    selection_seconds = time.monotonic() - start_time
    dataset = TesseCdRgbdDataset(
        path(pair["rgbd_root"]),
        pair["scene"],
        path(pair["rgbd_export_manifest"]),
        path(pair["causal_schedule"]),
    )
    for frame_id, entries in sorted(jobs.items()):
        target = cache / f"{frame_id:06d}.npz"
        if target.exists():
            continue
        start_time = time.monotonic()
        frame = dataset[frame_id]
        h, w = frame.depth.shape
        references, quality, counts = [], [], []
        for owner, index in entries:
            entry = bank[owner]
            if not np.allclose(frame.pose, entry["pose"][index], rtol=1e-5, atol=1e-5):
                raise ValueError("global RGBD/native pose mismatch")
            box = np.asarray(entry["box_2d"][index])
            truncated = box[0] <= 0 or box[1] <= 0 or box[2] >= w - 1 or box[3] >= h - 1
            for state_index, current in enumerate(states):
                rows = groups[owner]
                rows = rows[current[rows]]
                _, pixels = project_source_support(xyz, rows, frame)
                support = len(np.unique(pixels))
                projection = project_world_points(
                    xyz[rows], frame.pose, frame.intrinsics
                )
                visible = projection.projectable
                projectable = len(
                    np.unique(
                        projection.rows[visible] * w + projection.columns[visible]
                    )
                )
                q = (
                    min(1.0, support / max(1, projectable))
                    * np.sqrt(
                        min(1.0, support / max(1.0, float(entry["vis_area"][index])))
                    )
                    * (0.5 if truncated else 1.0)
                    if support >= 16
                    else 0.0
                )
                references.append((state_index, owner, index))
                quality.append(q)
                counts.append((support, projectable))
        atomic_npz(
            target,
            state_owner_index=np.asarray(references),
            quality=np.asarray(quality),
            support_projectable_counts=np.asarray(counts),
            projection_seconds=time.monotonic() - start_time,
        )
        if frame_id % 20 == 0:
            print("quality frame", frame_id, flush=True)
    weights = {}
    projection_seconds = 0.0
    for frame_id in sorted(jobs):
        with np.load(cache / f"{frame_id:06d}.npz") as data:
            projection_seconds += float(data["projection_seconds"])
            for key, q in zip(data["state_owner_index"], data["quality"]):
                weights[tuple(int(v) for v in key)] = float(q)
    with np.load(room / "native_cached_batch/text_features.npz") as data:
        if not np.array_equal(data["class_ids"], classes):
            raise ValueError("shared text vocabulary changed")
        text, scale = data["text"], float(data["logit_scale"])
    for state_index, state in enumerate(("B3", "H2")):
        for method in methods:
            feature_owners, posteriors, references = [], [], []
            for owner, (indices, frames, visit) in sorted(selected.items()):
                q = (
                    np.ones(len(indices))
                    if method == "MV_DIVERSE_MEAN"
                    else np.array(
                        [weights[(state_index, owner, int(index))] for index in indices]
                    )
                )
                z = aggregate_views(np.asarray(bank[owner]["feat"])[indices], q)
                references.append(
                    {
                        "owner": owner,
                        "visit": visit,
                        "original_frames": frames[indices].tolist(),
                        "weights": q.tolist(),
                        "fallback": z is None,
                    }
                )
                if z is None:
                    continue
                logits = scale * (z @ text.T)
                p = np.exp(logits - logits.max())
                p /= p.sum()
                feature_owners.append(owner)
                posteriors.append(p)
            atomic_npz(
                output / f"{state}_{method}_owner_features.npz",
                feature_owners=np.asarray(feature_owners),
                owner_posterior=np.asarray(posteriors).reshape(-1, len(classes)),
                class_ids=classes,
            )
            _atomic_json(
                output / f"{state}_{method}_views.json",
                {"observations": references, "new_image_encoder_forwards": 0},
            )
    _atomic_json(
        output / "shared_cost.json",
        {
            "selection_seconds": selection_seconds,
            "projection_seconds": projection_seconds,
            "frame_count": len(jobs),
            "candidate_owner_views": sum(len(e) for e in jobs.values()),
            "scope": "shared B3/H2 and methods; excludes loading, writes and evaluation",
        },
    )
    apply_and_evaluate_owner_features(
        config, state_config, pair, crosswalk, output, methods, owner_confidence
    )


if __name__ == "__main__":
    main()
