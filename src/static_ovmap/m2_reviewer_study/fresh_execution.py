"""Real fresh capture/source/prediction path under this study's own contract."""

import argparse
import fcntl
import os
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.binding import _bind_scene
from src.static_ovmap.composition_study.capture_recipe import capture_one
from src.static_ovmap.composition_study.execution import _paid_requests
from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.composition_study.object_evidence import (
    owner_labels,
    prepare_static_sources,
    query_objects,
    source_bundle,
)
from src.static_ovmap.composition_study.visual_requests import VisualRequestLoader
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.query_gain_policy import (
    QueryFeatureStandardizer,
    predict_query_gain,
)
from src.static_ovmap.module_validation.query_pipeline import reconcile_export
from src.static_ovmap.module_validation.query_state import FeatureStore
from src.static_ovmap.module_validation.query_study import CapturedFrames
from src.static_ovmap.module_validation.scannet_download import verified_download
from src.static_ovmap.module_validation.scannet_frames import export_sensor
from src.static_ovmap.module_validation.scannet_ground_truth import (
    load_ground_truth,
    prepare_ground_truth,
)
from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu
from src.static_ovmap.module_validation.scannet_study import (
    build_native_prediction,
    freeze_projection,
    load_prediction,
    relabel_prediction,
    save_prediction,
)
from src.static_ovmap.module_validation.semantic_models import (
    encode_semantic_requests,
    load_semantic_records,
)
from src.static_ovmap.module_validation.semantic_study import (
    direct_readouts,
    prepare_semantic_manifest,
)

from .binding import ROOT, InputIndex
from .diagnostics import SOURCES
from .evaluation import SceneEvaluator, pool
from .fresh import audit_fresh, freeze_fresh_plan
from .fusion import fuse
from .query_replay import replay_study
from .scores import read_scene


def contract(binding, scene):
    root = Path(binding["output_root"]) / "fresh"
    plan = read_json(root / "plan.json")
    if plan["status"] != "FROZEN" or plan["binding"] != binding["identity"] or scene not in plan["scenes"]:
        raise ValueError("fresh scene is not authorized by the frozen new-study plan")
    if canonical_digest({k: v for k, v in plan.items() if k != "identity"}) != plan["identity"]:
        raise ValueError("fresh plan identity changed")
    audit = read_json(root / "audits" / (plan["audit_identity"] + ".json"))
    if audit["identity"] != plan["audit_identity"] or audit["binding"] != binding["identity"] or canonical_digest(
            {k: v for k, v in audit.items() if k != "identity"}) != audit["identity"]:
        raise ValueError("fresh exposure audit identity changed")
    if audit["chosen"] != plan["scenes"] or len({s.split("_")[0] for s in plan["scenes"]}) != 4:
        raise ValueError("fresh physical-family selection changed")
    row = next(r for r in plan["rows"] if r["scene"] == scene)
    if row != next(r for r in audit["inventory"] if r["scene"] == scene):
        raise ValueError("fresh row differs from audited inventory")
    if row["exposed"] or not all(row[k] for k in ("authorized", "complete", "exposure_known")):
        raise ValueError("fresh row is incomplete, exposed or unresolved")
    config = read_json(Path(binding["composition_root"]) / "resolved_config.json")
    return root, plan, row, config


def prepare_fresh_scene(binding, scene, gpu):
    root, plan, row, original = contract(binding, scene)
    output = root / "scenes" / scene
    runtime = {**original["runtime"], "cuda_device": str(gpu), "output_root": str(output / "native")}
    lock = Path(original["gpu_lock"]).with_name(f".visual-gpu-{gpu}.lock")
    raw = Path(runtime["data_root"])
    # Verification is local-only; verified_download never initiates a download.
    for suffix, remote in row["authorization"]["files"].items():
        if not verified_download(raw / "scans" / scene / (scene + suffix), remote):
            raise ValueError("fresh raw bytes differ from authorized acquisition")
    native_input = output / "exported" / scene
    export_sensor(raw / "scans" / scene / (scene + ".sens"), native_input, row["schedule"])
    study = output / "study"
    annotations = prepare_ground_truth(Path(runtime["upstream"]), raw / "scans" / scene, scene,
                                      raw / "scannetv2-labels.combined.tsv", study / "annotations" / scene)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        stop, samples, sample_errors = threading.Event(), [], []

        def sample_gpu():
            while not stop.is_set():
                try:
                    value = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                                     "--format=csv,noheader,nounits"], text=True, timeout=5)
                    samples.append(int(value.strip()))
                except (subprocess.SubprocessError, ValueError) as error:
                    sample_errors.append(str(error))
                stop.wait(.25)

        sampler = threading.Thread(target=sample_gpu, daemon=True)
        sampler.start()
        try:
            mapping = capture_one(runtime, {"scene_id": scene, "role": "reviewer_fresh", "schedule": row["schedule"]},
                                  native_input, output / "native")
        finally:
            stop.set()
            sampler.join(timeout=6)
            if not (output / "capture_peak_memory.json").exists():
                write_once(output / "capture_peak_memory.json", {"gpu": str(gpu), "sampled_peak_device_mib": max(samples, default=None),
                    "samples": len(samples), "sampling_seconds": .25, "errors": sample_errors,
                    "scope": "GPU-device memory during locked frontend/native child processes; sampled, not allocator-exact"})
    capture_path = Path(mapping["capture_manifest"])
    capture = read_json(capture_path)
    base = study / "scenes" / scene
    targets = load_ground_truth(annotations)
    with np.load(capture_path.parent / capture["surface"]["path"], allow_pickle=False) as surface:
        projection = freeze_projection(surface["surface_xyz"], targets["xyz"], base / "projection")
    native_path = build_native_prediction(capture_path, Path(mapping["native_features"]),
                                         Path(original["models"]["native"]["text"]["path"]), projection, base / "baseline")
    native = load_prediction(native_path)
    manifest_path = base / "semantic_requests.json"
    manifest = prepare_semantic_manifest(capture_path, native, manifest_path)
    config = {**original, "attempt_root": str(output), "runtime": runtime, "gpu_lock": str(lock),
              "source": {**original["source"], "study_root": str(study)}, "fresh_plan_identity": plan["identity"]}
    write_once(output / "prepared_config.json", config)
    subprocess.run([runtime["semantic_python"], "-m", "src.static_ovmap.m2_reviewer_study.fresh_execution",
                    "--binding", str(Path(binding["output_root"]) / "source_binding.json"), "--scene", scene,
                    "--leaf", "semantic", "--gpu", str(gpu)], cwd=ROOT, check=True)
    records = load_semantic_records(base / "semantic_models/siglip2")
    with np.load(config["models"]["siglip2"]["text"]["path"], allow_pickle=False) as text:
        readouts = direct_readouts(manifest, records, text["text_embeddings"], tuple(map(int, text["valid_ids"])),
                                  owner_labels(native), model="siglip2")["S_SIGLIP2_AREA"]
    payload = relabel_prediction(native, "S_SIGLIP2_AREA", "S", {o: r["label_id"] for o, r in readouts.items()},
                                {"native_requests": 0, "siglip2_requests": len(records), "crop_inputs": 6 * len(records)},
                                {"request_manifest": manifest["identity"]})
    save_prediction(payload, base / "semantic/direct/S_SIGLIP2_AREA")
    index = InputIndex()
    for model in config["models"].values():
        index.identity(model["text"]["path"], model["text"])
    scene_data = _bind_scene(index, scene, "reviewer_fresh", runtime, config["source"])
    config["scenes"] = {scene: scene_data}
    write_once(output / "source_manifest.json", index.manifest())
    write_once(output / "resolved_config.json", config)
    prepare_static_sources(config, scene, output / "sources" / scene)
    return config


def semantic_leaf(binding, scene, gpu):
    import torch

    root, plan, _, _ = contract(binding, scene)
    output = root / "scenes" / scene
    config = read_json(output / "prepared_config.json")
    if config["fresh_plan_identity"] != plan["identity"]:
        raise ValueError("prepared scene differs from fresh plan")
    base = Path(config["source"]["study_root"]) / "scenes" / scene
    with Path(config["gpu_lock"]).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        require_idle_gpu(str(gpu), output)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        receipt = encode_semantic_requests(base / "semantic_requests.json", "siglip2", config["source"], base / "semantic_models/siglip2")
        torch.cuda.synchronize()
        memory = {"allocated_bytes": torch.cuda.max_memory_allocated(), "reserved_bytes": torch.cuda.max_memory_reserved(),
                  "receipt_identity": receipt["input_identity"], "gpu": str(gpu)}
        if not (output / "semantic_peak_memory.json").exists():
            write_once(output / "semantic_peak_memory.json", memory)


def gain_leaf(binding, scene, gpu):
    import torch

    root, plan, _, _ = contract(binding, scene)
    output = root / "scenes" / scene
    config = read_json(output / "resolved_config.json")
    if config["fresh_plan_identity"] != plan["identity"]:
        raise ValueError("fresh query plan changed")
    destination = output / "query" / scene / "Q_GAIN"
    destination.mkdir(parents=True, exist_ok=True)
    index = InputIndex()
    if (destination / "receipt.json").exists():
        receipt = read_json(destination / "receipt.json")
        for entry in receipt["inputs"] + receipt["outputs"]:
            index.identity(entry["path"], entry)
        return receipt
    data, model = config["scenes"][scene], config["models"]["native"]
    for key in ("capture", "native_prediction", "surface"):
        index.identity(data[key])
    index.identity(config["checkpoint"]["path"], config["checkpoint"])
    index.identity(model["text"]["path"], model["text"])
    with np.load(model["text"]["path"], allow_pickle=False) as arrays:
        text, ids = arrays["text_embeddings"], tuple(map(int, arrays["valid_ids"]))
    checkpoint = torch.load(config["checkpoint"]["path"], map_location="cpu", weights_only=False)
    if checkpoint["status"] != "COMPLETE":
        raise ValueError("frozen query checkpoint incomplete")
    scaler = QueryFeatureStandardizer(**checkpoint["scaler_state_dict"])
    frames = CapturedFrames(data["capture"])
    loader = VisualRequestLoader(frames, config, "native", output / "cache/native")
    store = FeatureStore(loader)
    started = time.monotonic()
    with Path(config["gpu_lock"]).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        require_idle_gpu(str(gpu), destination)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        result = replay_study(frames, "Q_GAIN", store, text, budget=200,
                              predictor=lambda x: predict_query_gain(checkpoint["state_dict"], x), standardizer=scaler)
        with np.load(data["surface"], allow_pickle=False) as surface:
            reconcile_export(result, surface, text)
        physical = {**asdict(store.physical_ledger), "model_loads": loader.model_loads,
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}
    native = load_prediction(data["native_prediction"])
    objects = query_objects(result["state"], sorted(owner_labels(native)), text, ids)
    logical = asdict(result["state"].logical_ledger)
    costs = {"native_requests": logical["attempts"], "siglip2_requests": 0, "crop_inputs": logical["crop_inputs"]}
    prediction = relabel_prediction(native, "Q_GAIN", "Q", {o: r["label"] for o, r in objects.items()}, costs,
                                    {"fresh_plan": plan["identity"], "frame_barrier": True, "budget": 200})
    path = save_prediction(prediction, destination / "prediction")
    evidence = source_bundle(scene, "Q_GAIN", native, prediction, objects, model, index.entries(), costs)
    write_once(destination / "source_evidence.json", evidence)
    write_once(destination / "decisions.json", {"frames": result["decisions"],
        "paid_requests": _paid_requests(data["capture"], result["decisions"]),
        "retained_features": {str(o): [f.request_id for f in s.features] for o, s in result["state"].objects.items()},
        "discarded_ambiguous_features": sorted(result["lineage"].dropped_feature_ids)})
    for entry in loader.index.manifest()["entries"]:
        index.identity(entry["path"], entry)
    receipt = {"status": "COMPLETE", "logical": logical, "physical": physical, "elapsed_seconds": time.monotonic() - started,
        "prediction_manifest": str(path), "inputs": index.entries(), "outputs": [index.identity(p) for p in
            (path, path.parent / "prediction.npz", destination / "source_evidence.json", destination / "decisions.json")]}
    write_once(destination / "receipt.json", receipt)
    return receipt


def run_fresh(binding, gpu="2"):
    root = Path(binding["output_root"]) / "fresh"
    if (root / "status.json").exists() and not (root / "plan.json").exists():
        previous = read_json(root / "status.json")
        audit = read_json(root / "audits" / (previous["audit_identity"] + ".json"))
        raw = Path(audit["authorized_inventory_root"])
        observed = sorted(p.name for p in (raw / "scans").glob("scene*") if p.is_dir())
        if observed != sorted(r["scene"] for r in audit["inventory"]):
            raise ValueError("local fresh inventory changed; bind a new attempt")
        acquisition = next(r for r in audit["inputs"] if r["path"] == str(raw / "acquisition_lock.json"))
        InputIndex().identity(acquisition["path"], acquisition)
        if all(scene.split("_")[0] in audit["explicit_minimum_exclusions"] for scene in observed):
            return previous
    if (root / "plan.json").exists():
        plan = read_json(root / "plan.json")
        contract(binding, plan["scenes"][0])
    else:
        audit = audit_fresh(binding)
        if not audit["chosen"]:
            result = {"status": audit["status"], "audit_identity": audit["identity"],
                      "eligible_families": audit["eligible_unexposed_family_count"],
                      "required_families": 4, "actual_fresh_scenes_evaluated": 0,
                      "success_path": "fresh_execution.run_fresh: export/capture/S2/GAIN/locked-fusion/dual-rank/pool",
                      "no_new_download": True}
            write_once(root / "status.json", result)
            return result
        plan = freeze_fresh_plan(binding, audit)
    original = read_json(Path(binding["composition_root"]) / "resolved_config.json")
    for scene in plan["scenes"]:
        for leaf in ("prepare", "gain"):
            subprocess.run([original["runtime"]["mapping_python"], "-m", "src.static_ovmap.m2_reviewer_study.fresh_execution",
                            "--binding", str(Path(binding["output_root"]) / "source_binding.json"), "--scene", scene,
                            "--leaf", leaf, "--gpu", str(gpu)], cwd=ROOT, check=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "8",
                                "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
    local = {**binding, "scenes": {}}
    for scene in plan["scenes"]:
        output = root / "scenes" / scene
        config_path = output / "resolved_config.json"
        config = read_json(config_path)
        native = read_json(config["scenes"][scene]["native_prediction"])
        local["scenes"][scene] = {"dataset": "ScanNet", "role": "FRESH_CONFIRMATION", "config": str(config_path),
            "native_prediction_key": native["prediction_key"], "native_record_key": native["record_key"],
            "sources": {"N0": str(output / "sources" / scene / "N0.json"),
                        "S_SIGLIP2_AREA": str(output / "sources" / scene / "S_SIGLIP2_AREA.json"),
                        "Q_GAIN": str(output / "query" / scene / "Q_GAIN/source_evidence.json")}}
    spec = read_json(binding["spec"])
    final = read_json(Path(binding["output_root"]) / "calibration/new_final.json")
    definitions = {r["method_id"]: r for r in spec["ablations"]}
    locked = {}
    # Finish the entire four-scene prediction matrix before metric evaluation.
    for scene in plan["scenes"]:
        evidence = read_scene(local, scene)
        labels = owner_labels(evidence.native)
        locked[scene] = {}
        for method in plan["methods"]:
            ts, sources = dict(plan["temperatures"]), SOURCES
            hard = empty = False
            if method in SOURCES:
                result = {int(o): r["label"] for o, r in evidence.sources[method]["objects"].items()}
            else:
                if method in definitions:
                    definition = definitions[method]
                    sources, hard = definition["sources"], definition["fusion"] == "hard_vote"
                    empty = definition["empty_source_action"] == "class_0"
                    if definition["temperature_mode"] == "constant_0.01":
                        ts = {name: .01 for name in SOURCES}
                    elif definition["temperature_mode"] == "shared_refit":
                        ts = {name: final["shared"]["temperature"] for name in SOURCES}
                result, _ = fuse(labels, evidence.sources, evidence.config["models"]["native"]["valid_ids"], sources, ts,
                                 hard=hard, empty_class0=empty)
            prediction = relabel_prediction(evidence.native, method, "S", result, {},
                                            {"fresh_plan": plan["identity"], "temperatures": ts})
            manifest = save_prediction(prediction, root / "predictions" / scene / method)
            locked[scene][method] = {"labels": result, "prediction_identity": prediction.prediction_key, "manifest": str(manifest)}
    write_once(root / "locked_predictions.json", locked)
    rows, namespace = [], None
    for scene in plan["scenes"]:
        evaluator = SceneEvaluator(read_scene(local, scene), root)
        namespace = evaluator.namespace
        for method, prediction in locked[scene].items():
            for rank in ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS"):
                rows.append(evaluator.evaluate(prediction["labels"], method, rank, prediction["prediction_identity"]))
    pooled = []
    for method in plan["methods"]:
        for rank in ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS"):
            pooled.append({**pool(root, rows, namespace, plan["scenes"], method, rank), "method": method})
    result = {"status": "FRESH_CONFIRMATION_COMPLETE", "plan_identity": plan["identity"], "scenes": plan["scenes"],
              "methods": plan["methods"], "rows": rows, "pooled": pooled, "new_fitting": False}
    write_once(root / "status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--leaf", choices=("prepare", "semantic", "gain"), required=True)
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "8"
    binding = read_json(args.binding)
    {"prepare": prepare_fresh_scene, "semantic": semantic_leaf, "gain": gain_leaf}[args.leaf](binding, args.scene, args.gpu)
    print("FRESH_LEAF_COMPLETE", args.scene, args.leaf, flush=True)


if __name__ == "__main__":
    main()
