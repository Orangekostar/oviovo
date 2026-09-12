"""Current RGBD-supported historical-patch semantics on frozen B3/H2 states."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.build_crove_room0_mask_bank import file_hash
from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_crove_fine_current_map import _owner_row_groups
from scripts.evaluation.run_crove_multiview_apartment import evaluate_saved_readouts
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.datasets.tesse_cd import TesseCdRgbdDataset
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.oviv2.surface_current_semantics import (
    current_view_embedding,
    unique_region_pixels,
)
from src.oviv2.surface_mask_evidence import interior_labels
from src.oviv2.surface_readout import resolve_semantic_update
from src.oviv2.surface_view_bank import project_source_support


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    room = path(config["run_root"]) / "dev/apartment"
    compact = path(config["compact_output_root"])
    output = room / "current_aware"
    output.mkdir(exist_ok=True)
    settings = {
        "method": "MV_CURRENT_AWARE",
        "states": ["B3", "H2"],
        "baseline": "MV_REENCODE_QUALITY; unchanged current-visit rows and unsupported historical patches",
        "current_features": "same state native-owner diverse selected six-crop observations from the completed re-encoding bank",
        "support": "historical fixed 2cm patch representative projects into measured current depth within .05m, current source mask and eroded independent CropFormer interior",
        "ambiguity": "multiple encoded current regions on a pixel supply no vote; duplicate frame observations excluded",
        "selection": "at least two positive-quality distinct current frames; at most four angular diverse current observations, quality seed and quality aggregation",
        "priority": "reliable current-only feature replaces historical readout at supported local patches; no current evidence keeps historical semantics/confidence/role",
        "cross_visit_geometry_union": False,
        "owner_changes": False,
        "new_image_forwards": 0,
        "adaptation": "representative-supported local-patch backprojection; not individual depth certification for every triangle-soup row",
        "GT_read_before_both_predictions": False,
    }
    registry = compact / "current_aware_apartment_registry.json"
    if registry.exists() and json.loads(registry.read_text()) != settings:
        raise ValueError("frozen current-aware policy changed")
    _atomic_json(registry, settings)
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
    with np.load(room / "native_cached_batch/text_features.npz") as data:
        text, scale, classes = (
            data["text"],
            float(data["logit_scale"]),
            data["class_ids"],
        )
    for state in ("B3", "H2"):
        started = time.monotonic()
        root = output / state
        root.mkdir(exist_ok=True)
        encoded = room / "reencoded_regions" / state / "native"
        selected = json.loads((encoded / "selected_views.json").read_text())
        start, stop = pair["visit_frame_ranges"]["t1"]
        selected = {
            int(owner): values["diverse"]
            for owner, values in selected.items()
            if int(owner) > int(pair["t1_owner_offset"])
        }
        frames = sorted({f for values in selected.values() for f in values})
        if any(f < start or f > stop for f in frames):
            raise ValueError("current features outside authorized current visit")
        baseline_file = room / "reencoded_regions" / f"{state}_MV_REENCODE_QUALITY.npz"
        representatives = room / "independent_mask_bank" / state / "representatives.npz"
        mapping_file = room / "graph_topology" / state / "patch_mapping.npz"
        binding = {
            "baseline": file_hash(baseline_file),
            "representatives": file_hash(representatives),
            "patch_mapping": file_hash(mapping_file),
            "selected_views": file_hash(encoded / "selected_views.json"),
            "encoder_binding": file_hash(encoded / "input_binding.json"),
            "policy": file_hash(registry),
            "frame_features": {
                str(f): file_hash(encoded / f"{f:06d}.npz") for f in frames
            },
        }
        binding_file = root / "input_binding.json"
        if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
            raise ValueError("current observation inputs changed")
        _atomic_json(binding_file, binding)
        with np.load(representatives) as data:
            points, node_visits = data["xyz"], data["visit_ids"]
        historical = np.flatnonzero(node_visits == 0)
        all_nodes, all_features, feature_frames, feature_quality, embeddings = (
            [],
            [],
            [],
            [],
            [],
        )
        for frame_id in frames:
            target = root / f"{frame_id:06d}.npz"
            with np.load(encoded / f"{frame_id:06d}.npz") as data:
                slots = [
                    i
                    for i, o in enumerate(data["feature_owners"])
                    if frame_id in selected.get(int(o), []) and data["quality"][i] > 0
                ]
                z, q = data["embeddings"][slots], data["quality"][slots]
                region_pixels = [
                    data["mask_pixels"][
                        data["mask_offsets"][i] : data["mask_offsets"][i + 1]
                    ]
                    for i in slots
                ]
            mask_file = path(
                f"$HOME/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/native-runs/apartment/t1/2f638911509f-apartment-t1/frontend/frame{frame_id - start:06d}.png"
            )
            mask_hash = file_hash(mask_file)
            if not target.exists():
                frame = dataset[frame_id]
                pixels_to_slot = unique_region_pixels(region_pixels, frame.depth.size)
                mask = interior_labels(np.asarray(Image.open(mask_file)))
                if mask.shape != frame.depth.shape:
                    raise ValueError("current mask/RGBD grids differ")
                nodes, pixels = project_source_support(points, historical, frame)
                usable = (pixels_to_slot[pixels] >= 0) & (mask.ravel()[pixels] > 0)
                atomic_npz(
                    target,
                    node_ids=nodes[usable],
                    feature_slots=pixels_to_slot[pixels[usable]],
                    frame_id=frame_id,
                    mask_sha256=mask_hash,
                )
            with np.load(target) as data:
                if (
                    str(data["mask_sha256"]) != mask_hash
                    or int(data["frame_id"]) != frame_id
                ):
                    raise ValueError("cached current observation binding changed")
                nodes, slot = data["node_ids"], data["feature_slots"]
                if np.any(node_visits[nodes] != 0) or len(np.unique(nodes)) != len(
                    nodes
                ):
                    raise ValueError("current observation repeats node or alters visit")
                all_nodes.append(nodes)
                all_features.append(slot + len(embeddings))
            embeddings.extend(z)
            feature_quality.extend(q)
            feature_frames.extend([frame_id] * len(z))
            if frame_id % 16 == 0:
                print(
                    state,
                    frame_id,
                    len(nodes),
                    "historical patches currently supported",
                    flush=True,
                )
        nodes, feature_ids = np.concatenate(all_nodes), np.concatenate(all_features)
        embeddings, quality, feature_frames = map(
            np.asarray, (embeddings, feature_quality, feature_frames)
        )
        atomic_npz(
            root / "current_feature_bank.npz",
            embeddings=embeddings,
            quality=quality,
            frame_ids=feature_frames,
            node_ids=nodes,
            feature_ids=feature_ids,
        )
        node_labels = np.zeros(len(points), np.int32)
        node_confidence = np.zeros(len(points), np.float32)
        node_count = np.zeros(len(points), np.uint16)
        posterior_nodes, posteriors = [], []
        cameras = np.array(
            [dataset.records[int(f)].camera_to_world[:3, 3] for f in feature_frames]
        )
        for node, rows in _owner_row_groups(nodes).items():
            ids = feature_ids[rows]
            embedding = current_view_embedding(
                embeddings[ids],
                quality[ids],
                feature_frames[ids],
                cameras[ids] - points[node],
            )
            if embedding is None:
                continue
            logits = scale * (embedding @ text.T)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            node_labels[node] = classes[p.argmax()]
            node_confidence[node] = p.max()
            node_count[node] = len(np.unique(feature_frames[ids]))
            posterior_nodes.append(node)
            posteriors.append(p)
        with np.load(baseline_file) as data:
            prediction = {key: data[key] for key in data.files}
        with np.load(mapping_file) as data:
            if not np.array_equal(data["source_indices"], prediction["source_indices"]):
                raise ValueError("patch/source mapping differs from current readout")
            patch, visits = data["source_patch"], data["source_visit_ids"]
        if not np.array_equal(node_visits[patch], visits):
            raise ValueError("patch crosses visit")
        covered = (node_labels[patch] > 0) & (visits == 0)
        old_ids, old_roles = prediction["semantic_ids"], prediction["eval_role"]
        labels, roles = resolve_semantic_update(
            old_ids, old_roles, node_labels[patch], covered.astype(np.uint8), crosswalk
        )
        confidence = prediction["semantic_confidence"].copy()
        confidence[covered] = node_confidence[patch[covered]]
        if not np.array_equal(
            labels[~covered], old_ids[~covered]
        ) or not np.array_equal(roles[~covered], old_roles[~covered]):
            raise ValueError("unsupported historical/current rows changed")
        prediction.update(
            semantic_ids=labels,
            semantic_confidence=confidence,
            eval_role=roles,
            current_feature_covered=covered,
            feature_covered=prediction["feature_covered"] | covered,
            semantic_update_kind=np.where(
                covered, 1, prediction["semantic_update_kind"]
            ).astype(np.uint8),
        )
        prediction_file = output / f"{state}_MV_CURRENT_AWARE.npz"
        if prediction_file.exists():
            with np.load(prediction_file) as saved:
                if set(saved.files) != set(prediction) or any(
                    not np.array_equal(saved[key], value)
                    for key, value in prediction.items()
                ):
                    raise ValueError("cached reconstruction changed frozen prediction")
        else:
            atomic_npz(prediction_file, **prediction)
        atomic_npz(
            root / "current_patch_readout.npz",
            semantic_ids=node_labels,
            semantic_confidence=node_confidence,
            independent_current_frames=node_count,
            posterior_node_ids=np.array(posterior_nodes, np.int64),
            patch_posterior=np.array(posteriors).reshape(-1, len(classes)),
            class_ids=classes,
        )
        record_file = output / f"{state}_MV_CURRENT_AWARE_prediction.json"
        if not record_file.exists():
            _atomic_json(
                record_file,
                {
                    "state": state,
                    "method": "MV_CURRENT_AWARE",
                    "source_rows": len(labels),
                    "current_supported_patch_count": int((node_count >= 2).sum()),
                    "current_supported_source_rows": int(covered.sum()),
                    "changed_semantic_rows": int(np.count_nonzero(labels != old_ids)),
                    "changed_role_rows": int(np.count_nonzero(roles != old_roles)),
                    "geometry_and_owner_fixed": True,
                    "new_image_forwards": 0,
                    "prediction_and_write_seconds": time.monotonic() - started,
                },
            )
        print(state, "MV_CURRENT_AWARE full prediction saved", flush=True)
    evaluate_saved_readouts(
        config, state_config, pair, crosswalk, output, ["MV_CURRENT_AWARE"]
    )


if __name__ == "__main__":
    main()
