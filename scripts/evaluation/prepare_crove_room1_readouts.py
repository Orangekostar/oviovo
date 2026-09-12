"""Bind room1 native and S0/S2 inputs without opening confirmation GT."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.build_crove_room0_mask_bank import file_hash
from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_crove_fine_current_map import (
    _load_crove_reference,
    _owner_row_groups,
    _semantic_configs,
    bind_ovi_owner_ids,
    load_native_ovi_surface,
    parse_instance_color_log,
)
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.evaluation.baselines.ovimap import relative_similarity_labels
from src.oviv2.surface_semantics import (
    SurfaceSemanticStrategy,
    transfer_surface_semantics,
)


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    fine_config = json.loads(path(config["source_config"]).read_text())
    benchmark = json.loads(
        path(
            fine_config["cases"]["replica_room0_static"]["benchmark_manifest"]
        ).read_text()
    )
    classes = benchmark["vocabulary"]["classes"]
    room = path(config["run_root"]) / "confirm/room1"
    manifest_file = room / "native_cpu/native_mapping_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    if manifest["state"] != "MAPPING_PASS" or manifest["frame_ids"] != list(
        range(0, 2000, 10)
    ):
        raise ValueError("complete authorized native input required")
    artifacts = {k: Path(v["path"]) for k, v in manifest["artifacts"].items()}
    for k, p in artifacts.items():
        if file_hash(p) != manifest["artifacts"][k]["sha256"]:
            raise ValueError("native artifact hash differs")
    reference = room / "crove_local_reference/final/oviv2_fused_mesh.ply"
    shared_text = (
        path(config["run_root"]) / "dev/room0/native_cached_batch/text_features.npz"
    )
    settings = {
        "scene": "room1",
        "native_manifest_sha256": file_hash(manifest_file),
        "crove_reference_sha256": file_hash(reference),
        "shared_text_sha256": file_hash(shared_text),
        "class_names": classes,
        "native": "last eight visibility-weighted raw cached six-crop means; minimum two views; canonical-relative sigmoid score",
        "canonical_prompts": ["object", "things", "stuff", "texture"],
        "semantic_transfer": fine_config["semantic_transfer"],
        "S0_S2": "same original room0 transfer rules applied to newly generated CROVE fused surface",
        "GT_read": False,
        "confirmation_scored": False,
    }
    compact = path(config["compact_output_root"])
    registry = compact / "room1_readout_inputs_registry.json"
    if registry.exists() and json.loads(registry.read_text()) != settings:
        raise ValueError("frozen room1 input binding changed")
    _atomic_json(registry, settings)
    output = room / "native_cached_batch"
    output.mkdir(exist_ok=True)
    with np.load(shared_text) as data:
        text, scale = data["text"], float(data["logit_scale"])
    canonical_file = output / "canonical_text.npz"
    if not canonical_file.exists():
        model_path = path(
            "$HOME/oviovo_baseline_builds/ovimap-ubuntu24-native/siglip-large-patch16-384"
        )
        model = (
            AutoModel.from_pretrained(model_path, local_files_only=True)
            .eval()
            .to("cuda:1")
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        inputs = tokenizer(
            settings["canonical_prompts"],
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to("cuda:1")
        with torch.inference_mode():
            canonical = model.get_text_features(**inputs).float().cpu().numpy()
        atomic_npz(canonical_file, canonical=canonical)
        del model
        torch.cuda.empty_cache()
    with np.load(canonical_file) as data:
        canonical = data["canonical"]
    atomic_npz(
        output / "text_features.npz",
        text=text,
        logit_scale=scale,
        class_ids=np.arange(1, len(classes) + 1),
    )
    print("loading complete room1 native mesh", flush=True)
    surface = load_native_ovi_surface(artifacts["instance_mesh"])
    colors = parse_instance_color_log(artifacts["instance_color_log"])
    owners = bind_ovi_owner_ids(surface.palette_rgb, colors)
    source = surface.source_vertex_indices
    groups = _owner_row_groups(owners)
    with artifacts["semantic_features"].open("rb") as handle:
        bank = pickle.load(handle)
    feature_owners, features = [], []
    for owner, entry in sorted(bank.items()):
        frames = np.asarray(entry["frame_id"])
        if np.any((frames < 0) | (frames >= 2000) | (frames % 10 != 0)) or len(
            np.unique(frames)
        ) != len(frames):
            raise ValueError("native retained frames outside authorized set")
        if int(owner) not in groups or len(frames) < 2:
            continue
        if not np.array_equal(entry["color"], colors[int(owner)]):
            raise ValueError("native feature/color identity differs")
        weights = np.asarray(entry["vis_area"])[-8:]
        z = np.asarray(entry["feat"])[-8:]
        if weights.sum() <= 0:
            continue
        features.append(np.average(z, axis=0, weights=weights))
        feature_owners.append(int(owner))
    names, scores = relative_similarity_labels(
        np.array(features), text, canonical, classes
    )
    ids, confidence = np.zeros(len(source), np.int32), np.zeros(len(source), np.float32)
    for owner, name, score in zip(feature_owners, names, scores):
        ids[groups[owner]] = classes.index(name) + 1
        confidence[groups[owner]] = score
    atomic_npz(
        output / "B_SEM_OVI_NATIVE.npz",
        source_indices=source,
        owner_ids=owners,
        semantic_ids=ids,
        semantic_confidence=confidence,
    )
    geometry = room / "inputs"
    geometry.mkdir(exist_ok=True)
    zeros = np.zeros(len(source), np.int32)
    atomic_npz(
        geometry / "current_surface.npz",
        vertices_xyz=surface.vertices_xyz,
        normals_xyz=surface.normals_xyz,
        triangles=surface.triangles,
        palette_rgb=surface.palette_rgb,
        source_vertex_indices=source,
        owner_entity_ids=owners,
        source_visit_ids=zeros,
        source_surface_indices=zeros,
        geometry_epochs=zeros,
        current_valid=np.ones(len(source), bool),
        semantic_ids=ids,
        semantic_confidences=confidence,
    )
    ref_xyz, ref_ids, ref_support = _load_crove_reference(reference)
    result = transfer_surface_semantics(
        fine_vertices_xyz=surface.vertices_xyz,
        reference_vertices_xyz=ref_xyz,
        reference_semantic_ids=ref_ids,
        reference_supports=ref_support,
        entity_semantic_ids=ids,
        entity_confidences=confidence,
        configs=tuple(
            c
            for c in _semantic_configs(fine_config)
            if c.strategy in (SurfaceSemanticStrategy.S0, SurfaceSemanticStrategy.S2)
        ),
        neighbor_count=fine_config["semantic_transfer"]["neighbor_count"],
        point_batch_size=fine_config["semantic_transfer"]["point_batch_size"],
    )
    controls = room / "semantic_controls"
    controls.mkdir(exist_ok=True)
    for strategy, value in result.items():
        name = (
            "B_SEM_CROVE_S0"
            if strategy == SurfaceSemanticStrategy.S0
            else "B_SEM_CROVE_S2"
        )
        atomic_npz(
            controls / f"{name}.npz",
            source_indices=source,
            owner_ids=owners,
            semantic_ids=value.semantic_ids,
            semantic_confidence=value.semantic_confidences,
            support_reliability=value.support_reliabilities,
            semantic_source_codes=value.semantic_source_codes,
        )
    _atomic_json(
        compact / "room1_readout_inputs.json",
        {
            "scene": "room1",
            "status": "NATIVE_S0_S2_INPUTS_READY_UNSCORED",
            "source_rows": len(source),
            "source_faces": len(surface.triangles),
            "native_feature_owners": len(feature_owners),
            "crove_reference_vertices": len(ref_xyz),
            "GT_read": False,
            "confirmation_scored": False,
        },
    )
    print("room1 native/S0/S2 inputs saved without GT", len(source), flush=True)


if __name__ == "__main__":
    main()
