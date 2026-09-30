"""Fresh anchors, genuine source distributions, and fixed three-readout exports."""

import copy
from dataclasses import asdict
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from static_ovmap.m2_reviewer_study.binding import InputIndex
from static_ovmap.m2_reviewer_study.evaluation import SceneEvaluator
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_study import (
    build_native_prediction, freeze_projection, load_prediction, native_readout,
    relabel_prediction, save_prediction,
)
from static_ovmap.module_validation.semantic_study import prepare_semantic_manifest
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.paired_evidence_study.residuals import pool, simple_dependence
from .binding import read
from .runtime import execute, exclusive_lock


def fuse_readout(sources, temperatures, method, ids, incumbent):
    if method not in {"FC_EQ", "D2"}:
        raise ValueError("unregistered backbone readout")
    labels, audit = {}, {}
    for owner, native_label in incumbent.items():
        scores = {name: (source["objects"][str(owner)]["scores"]
                        if source["objects"][str(owner)]["available"] else None)
                  for name, source in sources.items()}
        base = pool(scores, temperatures, ("N", "Q", "F"))
        probabilities = base if method == "FC_EQ" else simple_dependence(
            scores, temperatures, base, "PE_D2_GROUPED")
        labels[owner] = native_label if probabilities is None else int(ids[int(probabilities.argmax())])
        audit[str(owner)] = {"available_sources": [n for n, s in scores.items() if s is not None],
            "probabilities": None if probabilities is None else probabilities.tolist(),
            "label": labels[owner], "all_unavailable_fallback": probabilities is None}
    return labels, audit


def _worker(binding, python, module, job, log, gpu, *, sam=False):
    repo = Path(binding["repository_root"])
    job_path = Path(log).with_suffix(".job.json")
    atomic_write_json(job_path, job)
    paths = [str(repo / "src"), str(repo)]
    if sam:
        paths.insert(0, binding["sam2"]["code"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=":".join(paths),
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    return execute([python, "-m", module, "--job", job_path], repo, log, env=env)


def ensure_frontend(binding, scene, *, gpu):
    root = Path(binding["output_root"]) / "frontend" / scene / "SAM2_PAIRED"
    path = root / "receipt.json"
    if path.is_file() and read(path)["status"] == "COMPLETE":
        receipt = read(path)
        index = InputIndex()
        for item in receipt["inputs_and_outputs"]:
            index.identity(item["path"], item)
        return receipt
    sam = binding["sam2"]
    job = {"output_root": str(root), "capture_manifest": binding["scenes"][scene]["parent_capture"],
           "checkpoint": sam["checkpoint"], "sam2_repo": sam["code"], "sam2_commit": sam["commit"],
           "sam2_config": sam["config"], "preflight": False,
           "gpu_lock": f"/mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-{gpu}.lock"}
    _worker(binding, sam["python"], "static_ovmap.backbone_wave1.frontend_sam2", job,
            root / "run.log", gpu, sam=True)
    return read(path)


def build_sources(binding, spec, scene, mapping, *, gpu):
    root = Path(binding["output_root"]) / "readouts" / scene / mapping["map_id"]
    root.mkdir(parents=True, exist_ok=True)
    data, capture_path = binding["scenes"][scene], Path(mapping["capture_manifest"])
    cache = Path(binding["output_root"]) / "content_cache"
    gpu_lock = f"/mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-{gpu}.lock"
    native_root = root / "native_query"
    nq_path = native_root / "native_query_receipt.json"
    if not nq_path.is_file() or read(nq_path)["status"] != "COMPLETE":
        job = {"output_root": str(native_root), "map_id": mapping["map_id"],
            "capture_manifest": str(capture_path), "deferred_metadata": mapping["deferred_metadata"],
            "model": data["models"]["native"], "checkpoint": data["checkpoint"],
            "upstream": spec["upstream_worktree"], "gpu_lock": gpu_lock,
            "encoder_cache_root": str(cache / "native")}
        _worker(binding, data["runtime"]["native_perception_python"],
            "static_ovmap.backbone_wave1.semantic_readout", job, native_root / "run.log", gpu)
    nq = read(nq_path)
    capture = read(capture_path)
    annotation = read(data["parent_scene"]["annotations"])
    # Only target coordinates are opened here; labels wait until all readouts lock.
    with np.load(annotation["arrays_path"], allow_pickle=False) as arrays:
        target_xyz = arrays["xyz"]
    with np.load(capture_path.parent / capture["surface"]["path"], allow_pickle=False) as arrays:
        projection = freeze_projection(arrays["surface_xyz"], target_xyz, root / "projection")
    native_path = build_native_prediction(capture_path, native_root / "native_features.pkl",
        Path(data["models"]["native"]["text"]["path"]), projection, root / "anchor")
    native = load_prediction(native_path)
    labels, ids = owner_labels(native), data["models"]["native"]["valid_ids"]
    manifest_path = root / "fc_requests.json"
    manifest = prepare_semantic_manifest(capture_path, native, manifest_path)
    fc_root = root / "fc"
    if not (fc_root / "receipt.json").is_file() or read(fc_root / "receipt.json")["status"] != "COMPLETE":
        job = {"output_root": str(fc_root), "map_id": mapping["map_id"],
            "request_manifest": str(manifest_path), "owners": sorted(labels),
            "assets_root": binding["assets_root"], "cache_root": str(cache / "fc"),
            "class_names": data["models"]["native"]["class_names"], "valid_ids": ids,
            "gpu_lock": gpu_lock}
        _worker(binding, binding["fc"]["python"], "static_ovmap.backbone_wave1.region_readout",
                job, fc_root / "run.log", gpu)
    sources = {"F": read(fc_root / "F.json"), "N": {"source": "N", "objects": {}},
               "Q": {"source": "Q", "objects": {}}}
    with (native_root / "native_features.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    with np.load(data["models"]["native"]["text"]["path"], allow_pickle=False) as arrays:
        text, canonical = arrays["text_embeddings"], arrays["canonical_embeddings"]
    readout = native_readout(saved, text, canonical, tuple(ids))
    import torch
    for owner in labels:
        row = readout[owner]
        available = row["status"] == "AVAILABLE"
        scores = torch.nn.functional.cosine_similarity(torch.as_tensor(row["feature"], dtype=torch.float32),
            torch.as_tensor(text, dtype=torch.float32), dim=-1).numpy() if available else None
        sources["N"]["objects"][str(owner)] = {"available": available,
            "scores": scores.tolist() if available else None,
            "label": int(ids[int(scores.argmax())]) if available else None,
            "used_frames": row["used_frames"]}
    with np.load(native_root / "query_scores.npz", allow_pickle=False) as arrays:
        positions = {int(owner): i for i, owner in enumerate(arrays["owner_ids"])}
        for owner in labels:
            i = positions.get(owner)
            available = i is not None and bool(arrays["available"][i])
            scores = arrays["scores"][i] if available else None
            sources["Q"]["objects"][str(owner)] = {"available": available,
                "scores": scores.tolist() if available else None,
                "label": int(ids[int(scores.argmax())]) if available else None}
    for name in ("N", "Q"):
        sources[name].update(map_id=mapping["map_id"], valid_ids=ids,
            native_record_key=native.record_key, source_receipt=str(nq_path),
            model_identity=data["models"]["native"]["identity"])
        sources[name]["identity"] = canonical_digest(sources[name])
    for name, source in sources.items():
        atomic_write_json(root / "sources" / (name + ".json"), source)
    predictions = {"NATIVE_READOUT": str(native_path)}
    for method in ("FC_EQ", "D2"):
        replacement, audit = fuse_readout(sources, data["temperatures"], method, ids, labels)
        payload = relabel_prediction(native, method, "COMBO", replacement,
            dict(native.logical_cost, query_attempts=200, fc_attempts=len(manifest["requests"])),
            {"map_id": mapping["map_id"], "sources": {n: s["identity"] for n, s in sources.items()},
             "temperatures": data["temperatures"], "fresh_anchor": native.record_key})
        predictions[method] = str(save_prediction(payload, root / "predictions" / method))
        atomic_write_json(root / "decisions" / (method + ".json"), audit)
    config = copy.deepcopy(read(data["parent_config"]))
    config["attempt_root"] = str(root)
    config["runtime"]["upstream"] = spec["upstream_worktree"]
    config["scenes"] = {scene: dict(data["parent_scene"], native_prediction=str(native_path),
        native_features=str(native_root / "native_features.pkl"), native_geometry=asdict(native.geometry),
        capture=str(capture_path), projection=str(root / "projection/manifest.json"))}
    index = InputIndex()
    for path in (config["scenes"][scene]["projection"], config["scenes"][scene]["annotations"]):
        index.identity(path)
    atomic_write_json(root / "source_manifest.json", {"entries": index.entries()})
    atomic_write_json(root / "config.json", config)
    fc = read(fc_root / "receipt.json")
    required = dict(nq["standalone_required_content"], **fc["required_image_contents"])
    sam = None if mapping["recipe"]["frontend"] == "cropformer" else read(
        Path(binding["output_root"]) / "frontend" / scene / "SAM2_PAIRED/receipt.json")
    receipt = {"status": "PREDICTIONS_LOCKED", "scene": scene, "map_id": mapping["map_id"],
        "predictions": predictions, "sources": {n: str(root / "sources" / (n + ".json")) for n in sources},
        "config": str(root / "config.json"), "required_image_encodings": sum(required.values()) +
            (0 if sam is None else sam["counters"]["physical_image_encodings"]),
        "standalone_required_contents": required,
        "attributable_map_plus_readout_seconds": mapping["elapsed_seconds"] + nq["attributable_standalone_seconds"] +
            fc["attributable_standalone_seconds"] + (0 if sam is None else sam["elapsed_seconds"]),
        "timing_kind": "MEASURED_COMMON_CONTENT_STAGES_PLUS_MAP_AND_MODEL_LOADS_NOT_COLD_RERUN",
        "standalone_timing_missing_components": nq["standalone_timing_missing_components"] + fc["standalone_timing_missing_components"],
        "source_availability": {n: sum(r["available"] for r in s["objects"].values()) for n, s in sources.items()},
        "fc_target_cap_exclusions": manifest["excluded_targets"], "native_eligible_owners": len(labels)}
    receipt["identity"] = canonical_digest(receipt)
    atomic_write_json(root / "receipt.json", receipt)
    return receipt


def evaluate_map(binding, scene, receipt):
    root = Path(receipt["config"]).parent
    with exclusive_lock(root / ".evaluation.lock"):
        config = read(receipt["config"])
        native = load_prediction(receipt["predictions"]["NATIVE_READOUT"])
        sources = {"N0": read(receipt["sources"]["N"])}
        evidence = SimpleNamespace(scene=scene, dataset=binding["scenes"][scene]["dataset"],
                                   config=config, native=native, sources=sources)
        evaluator = SceneEvaluator(evidence, root / "evaluation")
        rows = []
        for method, path in receipt["predictions"].items():
            prediction = load_prediction(path)
            for rank in ("OFFICIAL_CURRENT_CLASS", "FROZEN_N0"):
                row = evaluator.evaluate(owner_labels(prediction), method, rank, prediction.prediction_key)
                rows.append(dict(row, map_id=receipt["map_id"]))
        atomic_write_json(root / "evaluation_rows.json", {"status": "COMPLETE", "rows": rows})
        return rows
