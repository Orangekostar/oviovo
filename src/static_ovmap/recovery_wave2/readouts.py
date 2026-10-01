"""Geometry-gated fresh N/Q/F sources and fixed released own-map readouts."""

import argparse
import copy
from dataclasses import asdict
import pickle
from pathlib import Path

import numpy as np

from static_ovmap.backbone_wave1.readouts import _worker, evaluate_map, fuse_readout
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_study import (
    build_native_prediction, freeze_projection, load_prediction, native_readout,
    relabel_prediction, save_prediction,
)
from static_ovmap.module_validation.semantic_study import prepare_semantic_manifest

from .binding import ConsumptionIndex, PathResolver, read
from .light import owner_labels, require_transfer_freeze


def require_semantic_gate(binding, scene, map_id):
    if scene in binding["datasets"]["development"]:
        screen = read(Path(binding["output_root"]) / "geometry/screens" / (map_id + ".json"))
        if (screen["status"] != "GEOMETRY_PASS" or screen["scene_order"] != binding["datasets"]["development"]
                or screen["map_id"] != map_id):
            raise ValueError("fresh semantic inference requires the complete four-scene geometry pass")
        return screen["identity"]
    require_transfer_freeze(binding, scene)
    freeze = read(Path(binding["output_root"]) / "freeze/receipt.json")
    if freeze["nominated_map"] != map_id:
        raise ValueError("semantic transfer requires the single committed map nominee")
    return freeze["selection_identity"]


def build_sources(binding, scene, map_id, build, *, gpu="2"):
    gate_identity = require_semantic_gate(binding, scene, map_id)
    data, root = binding["scenes"][scene], Path(binding["output_root"]) / "readouts" / scene / map_id
    root.mkdir(parents=True, exist_ok=True)
    resolver = PathResolver(binding["path_map"])
    inherited = data["inherited"]
    map_path = Path(binding["output_root"]) / "maps" / scene / map_id / "map_receipt.json"
    mapping = read(map_path)
    if mapping["status"] != "COMPLETE":
        raise ValueError("fresh readouts require a complete raw map")
    capture_path, native_root, fc_root = Path(mapping["capture_manifest"]), root / "native_query", root / "fc"
    index = ConsumptionIndex(root / "input_verifications.json")
    for path in (map_path, __file__):
        index.identity(path)
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "map_id": map_id,
        "map_input_identity": mapping["input_identity"], "semantic_gate": gate_identity, "inputs": index.entries()})
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        previous = read(receipt_path)
        if previous["status"] != "PREDICTIONS_LOCKED" or previous["input_identity"] != identity:
            raise ValueError("own-map readouts cannot resume with changed locked inputs")
        for item in previous["outputs"]:
            index.identity(item["path"], item)
        return previous
    native_job = {"output_root": str(native_root), "map_id": map_id, "map_receipt": str(map_path),
        "capture_manifest": str(capture_path), "deferred_metadata": mapping["deferred_metadata"],
        "model": inherited["models"]["native"], "checkpoint": inherited["checkpoint"],
        "upstream": build["upstream_worktree"], "gpu_lock": f"/mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-{gpu}.lock",
        "encoder_cache_root": str(Path(binding["output_root"]) / "content_cache/native"),
        "parent_encoder_cache_roots": [binding["parent_native_cache_root"]], "path_map": binding["path_map"]}
    _worker(binding, inherited["runtime"]["native_perception_python"], "static_ovmap.recovery_wave2.native_worker",
            native_job, native_root / "run.log", gpu)
    nq_path, nq = native_root / "native_query_receipt.json", read(native_root / "native_query_receipt.json")
    capture, base_config = read(capture_path), resolver.rewrite(read(data["config"]))
    annotation = resolver.rewrite(read(base_config["scenes"][scene]["annotations"]))
    with np.load(annotation["arrays_path"], allow_pickle=False) as arrays:
        target_xyz = arrays["xyz"]
    with np.load(capture_path.parent / capture["surface"]["path"], allow_pickle=False) as arrays:
        projection = freeze_projection(arrays["surface_xyz"], target_xyz, root / "projection")
    native_path = build_native_prediction(capture_path, native_root / "native_features.pkl",
        Path(inherited["models"]["native"]["text"]["path"]), projection, root / "anchor")
    native, ids = load_prediction(native_path), inherited["models"]["native"]["valid_ids"]
    labels = owner_labels(native)
    manifest_path = root / "fc_requests.json"
    manifest = prepare_semantic_manifest(capture_path, native, manifest_path)
    job = {"binding": str(Path(binding["output_root"]) / "resolved_inputs.json"), "scene": scene,
        "map_id": map_id, "output_root": str(fc_root), "request_manifest": str(manifest_path),
        "owners": sorted(labels), "gpu": str(gpu), "gpu_lock": native_job["gpu_lock"]}
    _worker(binding, binding["fc"]["python"], "static_ovmap.recovery_wave2.region_worker", job, fc_root / "run.log", gpu)
    fc = read(fc_root / "receipt.json")
    if nq["status"] != "COMPLETE" or fc["status"] != "COMPLETE":
        raise ValueError("own-map N/Q/F evidence is incomplete")
    sources = {"F": read(fc_root / "F.json"), "N": {"source": "N", "objects": {}}, "Q": {"source": "Q", "objects": {}}}
    with (native_root / "native_features.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    with np.load(inherited["models"]["native"]["text"]["path"], allow_pickle=False) as arrays:
        text, canonical = arrays["text_embeddings"], arrays["canonical_embeddings"]
    readout = native_readout(saved, text, canonical, tuple(ids))
    import torch

    for owner in labels:
        row, available = readout[owner], readout[owner]["status"] == "AVAILABLE"
        scores = torch.nn.functional.cosine_similarity(torch.as_tensor(row["feature"], dtype=torch.float32),
            torch.as_tensor(text, dtype=torch.float32), dim=-1).numpy() if available else None
        sources["N"]["objects"][str(owner)] = {"available": available, "scores": scores.tolist() if available else None,
            "label": int(ids[int(scores.argmax())]) if available else None, "used_frames": row["used_frames"]}
    with np.load(native_root / "query_scores.npz", allow_pickle=False) as arrays:
        positions = {int(owner): i for i, owner in enumerate(arrays["owner_ids"])}
        if arrays["valid_ids"].tolist() != ids:
            raise ValueError("fresh Q score vocabulary differs from the fixed native classifier")
        for owner in labels:
            position = positions.get(owner)
            available = position is not None and bool(arrays["available"][position])
            scores = arrays["scores"][position] if available else None
            sources["Q"]["objects"][str(owner)] = {"available": available, "scores": scores.tolist() if available else None,
                                                  "label": int(ids[int(scores.argmax())]) if available else None}
    for name in ("N", "Q"):
        sources[name].update(map_id=map_id, valid_ids=ids, native_record_key=native.record_key,
                             source_receipt=str(nq_path), model_identity=inherited["models"]["native"]["identity"])
        sources[name]["identity"] = canonical_digest(sources[name])
    for name, source in sources.items():
        atomic_write_json(root / "sources" / (name + ".json"), source)
    predictions = {"NATIVE_READOUT": str(native_path)}
    for method in ("FC_EQ", "D2"):
        replacement, audit = fuse_readout(sources, data["temperatures"], method, ids, labels)
        payload = relabel_prediction(native, method, "COMBO", replacement,
            dict(native.logical_cost, query_attempts=nq["query_budget_accounting"]["attempts"], fc_attempts=len(manifest["requests"])),
            {"map_id": map_id, "sources": {name: source["identity"] for name, source in sources.items()},
             "temperatures": data["temperatures"], "fresh_anchor": native.record_key})
        predictions[method] = str(save_prediction(payload, root / "predictions" / method))
        atomic_write_json(root / "decisions" / (method + ".json"), audit)
    config = copy.deepcopy(base_config)
    config["attempt_root"], config["runtime"]["upstream"] = str(root), build["upstream_worktree"]
    config["scenes"] = {scene: dict(base_config["scenes"][scene], native_prediction=str(native_path),
        native_features=str(native_root / "native_features.pkl"), native_geometry=asdict(native.geometry),
        capture=str(capture_path), projection=str(root / "projection/manifest.json"))}
    for path in (config["scenes"][scene]["projection"], config["scenes"][scene]["annotations"]):
        index.identity(path)
    atomic_write_json(root / "source_manifest.json", {"entries": index.entries()})
    atomic_write_json(root / "config.json", config)
    required = dict(nq["standalone_required_content"], **fc["required_image_contents"])
    sam = read(data["sam_receipt"]) if mapping["recipe"]["frontend"] != "cropformer" else None
    source_paths = {name: str(root / "sources" / (name + ".json")) for name in sources}
    outputs = [index.identity(path) for path in [*predictions.values(), *source_paths.values(), root / "config.json",
               *(root / "decisions" / (method + ".json") for method in ("FC_EQ", "D2"))]]
    for manifest_path in predictions.values():
        row = read(manifest_path)
        outputs.append(index.identity(Path(manifest_path).parent / row["arrays"]["path"], row["arrays"]))
    receipt = {"status": "PREDICTIONS_LOCKED", "scene": scene, "map_id": map_id, "input_identity": identity,
        "predictions": predictions, "sources": source_paths, "config": str(root / "config.json"), "outputs": outputs,
        "required_image_encodings": sum(required.values()) + (0 if sam is None else sam["counters"]["physical_image_encodings"]),
        "standalone_required_contents": required, "native_eligible_owners": len(labels),
        "source_availability": {name: sum(row["available"] for row in source["objects"].values()) for name, source in sources.items()},
        "fc_target_cap_exclusions": manifest["excluded_targets"], "all_predictions_locked_before_semantic_diagnostics": True,
        "query_budget_accounting": nq["query_budget_accounting"], "GT_input": False,
        "new_text_forwards": 0, "new_SAM_forwards": 0, "new_CropFormer_forwards": 0,
        "physical_native_Q_image_inputs": nq["physical_image_encodings"], "physical_FC_image_inputs": fc["physical_image_encodings"],
        "physical_FC_region_poolings": fc["physical_region_poolings"],
        "attributable_map_plus_readout_seconds": mapping["elapsed_seconds"] + nq["attributable_standalone_seconds"]
            + fc["attributable_standalone_seconds"] + (0 if sam is None else sam["elapsed_seconds"]),
        "timing_kind": "MEASURED_CONTENT_STAGES_PLUS_MAP_AND_MODEL_LOADS_NOT_COLD_RERUN",
        "standalone_timing_missing_components": nq["standalone_timing_missing_components"] + fc["standalone_timing_missing_components"]}
    receipt["identity"] = canonical_digest(receipt)
    atomic_write_json(receipt_path, receipt)
    index.write_memo(root / "input_verifications.json")
    print(scene, map_id, "N/Q/FC readouts locked", flush=True)
    return receipt


def recovery_context(binding, scene, map_id):
    data = copy.deepcopy(binding["scenes"][scene])
    root = Path(binding["output_root"]) / "readouts" / scene / map_id
    receipt = read(root / "receipt.json")
    mapping = read(Path(binding["output_root"]) / "maps" / scene / map_id / "map_receipt.json")
    if receipt["status"] != "PREDICTIONS_LOCKED":
        raise ValueError("own-map recovery requires all standard readouts locked")
    data.update(map_id=map_id, parent_readout_receipt=str(root / "receipt.json"), config=receipt["config"],
        predictions=receipt["predictions"], sources={name: {"path": path, "identity": read(path)["identity"]}
            for name, path in receipt["sources"].items()}, native_query_root=str(root / "native_query"),
        capture_manifest=mapping["capture_manifest"], deferred_metadata=mapping["deferred_metadata"],
        parent_readout_identity=receipt["identity"], fc_root=str(root / "fc"),
        FC_model_reference_root=binding["scenes"][scene]["fc_root"])
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--map-id", required=True)
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()
    binding = read(args.binding)
    receipt = build_sources(binding, args.scene, args.map_id, read(args.build), gpu=args.gpu)
    evaluate_map(binding, args.scene, receipt)
