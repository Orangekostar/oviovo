"""Official trained MaskAdapter and mean pooling on frozen Apartment B3/H2."""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

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
from src.oviv2.mask_adapter_readout import MaskAdapterReadout
from src.oviv2.surface_multiview_semantics import aggregate_views
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
    vocabulary = {
        v.semantic_id: v.native_name for v in crosswalk.aliases.values() if v.matched
    }
    class_ids = np.array(sorted(vocabulary), np.int32)
    names = [vocabulary[key] for key in class_ids]
    run = path(config["run_root"])
    output = run / "dev/apartment/adapter_projected_top4"
    bank_root = output / "view_bank"
    bank_root.mkdir(parents=True, exist_ok=True)
    methods = ["ADAPTER_CLIP_MEAN", "ADAPTER_CLIP_LEARNED"]
    settings = {
        "scene": pair["scene"],
        "pair_id": pair["pair_id"],
        "states": ["B3", "H2"],
        "methods": methods,
        "class_ids": class_ids.tolist(),
        "class_names": names,
        "checkpoint_sha256": "5686dc21e3461918d385d5cee02b103320e8ca30714485cb8f1694f0d590b55e",
        "text": "one shared photo-of-class template for full common-v2 vocabulary",
        "top_k": 4,
        "candidate_pool": "native retained source-visit observations, same indices in B3/H2",
        "mask": "project only exact current-valid source owner rows for each saved state; depth tolerance 0.05 m, no dilation",
        "minimum_unique_pixels": 16,
        "frame_windows": pair["visit_frame_ranges"],
        "coordinates": "original global camera poses; no per-visit first-pose normalization",
        "paired_features": "one dense image forward per frame shared by both states and both pooling heads",
        "view_aggregation": "normalize each embedding, uniform independent-view mean, normalize",
        "fallback": "KEEP_SOURCE native semantics/roles without valid mask features",
        "scope": "source-visit top-four DEV pair; no new training or cross-visit identity unions",
    }
    registry = path(config["compact_output_root"]) / "adapter_apartment_registry.json"
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("frozen adapter protocol differs")
    else:
        _atomic_json(registry, settings)
    with np.load(path(pair["current_map_root"]) / "current_surface.npz") as data:
        xyz, owners, visits, source = (
            data["vertices_xyz"],
            data["owner_entity_ids"],
            data["source_visit_ids"],
            data["source_vertex_indices"],
        )
    groups = _owner_row_groups(owners)
    hypotheses = (
        path(state_config["run_root"]) / "dev/pairs" / pair["pair_id"] / "hypotheses"
    )
    states = []
    for state, hypothesis, variant in [
        ("B3", "H1_B3", "D1_B3"),
        ("H2", "H2_INHERIT", "D2_INHERIT"),
    ]:
        with np.load(
            hypotheses
            / hypothesis
            / variant
            / "entity_epoch_state/entity_epoch_state.npz"
        ) as data:
            if not np.array_equal(
                data["source_visit_ids"], visits
            ) or not np.array_equal(data["source_vertex_indices"], source):
                raise ValueError("frozen state source keys differ")
            current = data["current_valid_after"]
        with np.load(run / "bridge" / f"{state}_legacy_prediction.npz") as data:
            if not np.array_equal(
                data["source_indices"], np.flatnonzero(current)
            ) or not np.array_equal(data["owner_ids"], owners[current]):
                raise ValueError("state and verified bridge source binding disagree")
        states.append(current)
    jobs = defaultdict(list)
    owner_confidence = {}
    native_root = path(
        "$HOME/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/native-runs/apartment"
    )
    for visit, record_key in enumerate(("b0_entities", "b2_entities")):
        records = _entity_records(path(pair[record_key]))
        offset = int(pair["t1_owner_offset"]) if visit else 0
        owner_confidence.update(
            {
                key + offset: float(record["semantic_score"])
                for key, record in records.items()
            }
        )
        manifest_path = (
            native_root
            / f"t{visit}"
            / f"2f638911509f-apartment-t{visit}"
            / "native_mapping_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        feature = manifest["artifacts"]["semantic_features"]
        if file_hash(Path(feature["path"])) != feature["sha256"]:
            raise ValueError("native feature bytes changed")
        with Path(feature["path"]).open("rb") as handle:
            native = pickle.load(handle)
        start, stop = pair["visit_frame_ranges"][f"t{visit}"]
        for local_owner, entry in sorted(native.items()):
            owner = int(local_owner) + offset
            if (
                owner not in groups
                or local_owner not in records
                or len(entry["frame_id"]) < 2
            ):
                continue
            if not np.all(visits[groups[owner]] == visit):
                raise ValueError("source owner crosses visit boundary")
            frames = np.asarray(entry["frame_id"], np.int64)
            if len(frames) != len(np.unique(frames)) or np.any(
                (frames < 0) | (frames > stop - start)
            ):
                raise ValueError("duplicate or unauthorized source-visit observation")
            for i in np.argsort(-np.asarray(entry["vis_area"]), kind="stable")[:4]:
                jobs[int(frames[i]) + start].append(
                    (owner, np.asarray(entry["pose"][i]))
                )
    dataset = TesseCdRgbdDataset(
        path(pair["rgbd_root"]),
        pair["scene"],
        path(pair["rgbd_export_manifest"]),
        path(pair["causal_schedule"]),
    )
    model = MaskAdapterReadout(
        path(
            "$HOME/oviovo_baseline_builds/maskadapter/fcclip_convnext_large_maskadapter.pth"
        ),
        device="cuda:1",
    )
    text = model.text_features(names)
    text_np = text.cpu().numpy()
    scale = float(model.clip.logit_scale.exp().clamp(max=100).cpu())
    torch.cuda.reset_peak_memory_stats("cuda:1")
    print(
        "Apartment adapter",
        len(jobs),
        "frames;",
        sum(len(v) for v in jobs.values()),
        "source owner views",
        flush=True,
    )
    for frame_id, entries in sorted(jobs.items()):
        target = bank_root / f"{frame_id:06d}.npz"
        if target.exists():
            continue
        started = time.monotonic()
        frame = dataset[frame_id]
        masks, metadata, counts, row_parts, pixel_parts, offsets = (
            [],
            [],
            [],
            [],
            [],
            [0],
        )
        for owner, pose in entries:
            if not np.allclose(frame.pose, pose, atol=1e-5, rtol=1e-5):
                raise ValueError("RGBD global pose differs from native feature pose")
            for state_index, current in enumerate(states):
                rows = groups[owner]
                rows = rows[current[rows]]
                supported, pixels = project_source_support(xyz, rows, frame)
                unique = np.unique(pixels)
                if len(unique) < 16:
                    continue
                mask = np.zeros(frame.depth.size, bool)
                mask[unique] = True
                masks.append(mask.reshape(frame.depth.shape))
                metadata.append((state_index, owner))
                counts.append(len(unique))
                row_parts.append(supported)
                pixel_parts.append(pixels)
                offsets.append(offsets[-1] + len(supported))
        projection_seconds = time.monotonic() - started
        if masks:
            result = model.classify_external_masks(frame.rgb, np.stack(masks), text)
            kept = result["mask_indices"].cpu().numpy()
            mean, learned = (
                result["mean_embeddings"].cpu().numpy(),
                result["learned_embeddings"].cpu().numpy(),
            )
        else:
            kept = np.empty(0, np.int64)
            mean = learned = np.empty((0, len(text_np[0])), np.float32)
        metadata_array = np.asarray(metadata, np.int64).reshape(-1, 2)
        atomic_npz(
            target,
            frame_id=frame_id,
            feature_state_owner=metadata_array[kept],
            mean_embeddings=mean,
            learned_embeddings=learned,
            mask_state_owner=metadata_array,
            unique_pixel_counts=np.asarray(counts),
            source_indices=np.concatenate(row_parts)
            if row_parts
            else np.empty(0, np.int64),
            source_pixels=np.concatenate(pixel_parts)
            if pixel_parts
            else np.empty(0, np.int32),
            source_offsets=np.asarray(offsets),
            image_forward=int(bool(masks)),
            projection_seconds=projection_seconds,
            frame_seconds_before_write=time.monotonic() - started,
        )
        print("frame", frame_id, len(kept), "paired mask features", flush=True)
    peak = torch.cuda.max_memory_allocated("cuda:1")
    del model
    torch.cuda.empty_cache()
    features = {
        (state, head): defaultdict(list)
        for state in range(2)
        for head in ("mean", "learned")
    }
    refs = defaultdict(list)
    timings = {"projection_seconds": 0.0, "frame_seconds_before_write": 0.0}
    forwards = 0
    for frame_id in sorted(jobs):
        with np.load(bank_root / f"{frame_id:06d}.npz") as data:
            forwards += int(data["image_forward"])
            for key in timings:
                timings[key] += float(data[key])
            for i, (state, owner) in enumerate(data["feature_state_owner"]):
                refs[(int(state), int(owner))].append(frame_id)
                for head in ("mean", "learned"):
                    features[(int(state), head)][int(owner)].append(
                        data[f"{head}_embeddings"][i]
                    )
    for state_index, state in enumerate(("B3", "H2")):
        for method, head in zip(methods, ("mean", "learned")):
            feature_owners, posteriors = [], []
            for owner, embeddings in sorted(features[(state_index, head)].items()):
                z = aggregate_views(np.asarray(embeddings), np.ones(len(embeddings)))
                if z is None:
                    continue
                logits = scale * (z @ text_np.T)
                p = np.exp(logits - logits.max())
                p /= p.sum()
                feature_owners.append(owner)
                posteriors.append(p)
            atomic_npz(
                output / f"{state}_{method}_owner_features.npz",
                feature_owners=np.asarray(feature_owners),
                owner_posterior=np.asarray(posteriors),
                class_ids=class_ids,
            )
    _atomic_json(
        output / "shared_cost_and_views.json",
        {
            "selected_frames": len(jobs),
            "extra_dense_encoder_forwards": forwards,
            "peak_allocated_gpu_bytes": peak,
            **timings,
            "scope": "shared both states and heads; excludes loading, cache writes and GT evaluation",
            "observations": [
                {
                    "state": ("B3", "H2")[state],
                    "owner": owner,
                    "original_frames": frame_ids,
                }
                for (state, owner), frame_ids in sorted(refs.items())
            ],
        },
    )
    apply_and_evaluate_owner_features(
        config, state_config, pair, crosswalk, output, methods, owner_confidence
    )


if __name__ == "__main__":
    main()
