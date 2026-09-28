"""Study-owned causal query jobs; historical experiments remain immutable."""

import fcntl
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.execution import _paid_requests
from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.composition_study.object_evidence import (
    owner_labels,
    query_objects,
    source_bundle,
)
from src.static_ovmap.composition_study.visual_requests import (
    VisualRequestLoader,
    operation_identity,
)
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.query_gain_policy import (
    QueryFeatureStandardizer,
    predict_query_gain,
)
from src.static_ovmap.module_validation.query_pipeline import reconcile_export
from src.static_ovmap.module_validation.query_state import FeatureStore
from src.static_ovmap.module_validation.query_study import CapturedFrames
from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu
from src.static_ovmap.module_validation.scannet_study import (
    load_prediction,
    relabel_prediction,
    save_prediction,
)

from .binding import InputIndex
from .query_replay import replay_study


def authorize_query(binding, scene, policy, budget, seed):
    if scene not in binding["scenes"]:
        raise ValueError("query scene outside study binding")
    if policy == "RV_Q_RANDOM":
        if seed not in (17, 23, 41) or budget != 200:
            raise ValueError("random control requires B200 and a frozen seed")
    elif policy not in ("Q_GAIN", "Q_COMBINE") or seed is not None or budget not in (100, 200, 400):
        raise ValueError("unsupported query policy/budget/seed")
    if budget != 200:
        gate_path = Path(binding["output_root"]) / "query_controls/curve_gate.json"
        if binding["scenes"][scene]["role"] == "CAL" or not gate_path.is_file():
            raise ValueError("budget curves require frozen CAL gate and transfer scene")
        gate = read_json(gate_path)
        if gate.get("status") != "FROZEN" or gate.get("triggered") is not True or gate.get("binding") != binding["identity"]:
            raise ValueError("budget curves are not authorized by this binding")
    return f"{policy}" + (f"_s{seed}" if seed is not None else "") + f"_B{budget}"


class StudyLoader(VisualRequestLoader):
    """Import only old cache entries attested by the frozen GAIN receipt."""

    def __init__(self, frames, config, cache_root, historical, index):
        super().__init__(frames, config, "native", cache_root)
        for entry in self.index.manifest()["entries"]:
            index.identity(entry["path"], entry)
        self.index = index
        self.historical = {str(Path(row["path"]).resolve()): row for row in historical["inputs"]}
        self.old_cache = Path(config["attempt_root"]) / "cache/native" / self.bound["identity"]

    def _import_native(self, request):
        identity = operation_identity(self.bound["identity"], request)
        path = self.old_cache / (identity + ".json")
        expected = self.historical.get(str(path.resolve()))
        if expected is None:
            return super()._import_native(request)
        self.index.identity(path, expected)
        row = read_json(path)
        if row["input_identity"] != identity or row["model_identity"] != self.bound["identity"] or canonical_digest(row["request"]) != canonical_digest(request):
            raise ValueError("historical query cache operation changed")
        return path, row


def run_query_job(binding, scene, policy, *, budget=200, seed=None, gpu="1"):
    method = authorize_query(binding, scene, policy, budget, seed)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("bind GPU before launching the query process")
    root = Path(binding["output_root"])
    output = root / "query_controls" / scene / method
    output.mkdir(parents=True, exist_ok=True)
    index = InputIndex()
    bound = {row["path"]: row for row in binding["inputs"]}
    config_path = binding["scenes"][scene]["config"]
    index.identity(config_path, bound[config_path])
    config = read_json(config_path)
    data, model = config["scenes"][scene], config["models"]["native"]
    old_path = Path(config["attempt_root"]) / "query" / scene / "Q_GAIN/receipt.json"
    index.identity(old_path)
    old = read_json(old_path)
    if old["status"] != "COMPLETE":
        raise ValueError("historical GAIN source is incomplete")
    # Its evidence must be exactly the source used in the bound core experiment.
    index.identity(old["evidence_path"], bound[binding["scenes"][scene]["sources"]["Q_GAIN"]])
    source_entries = {row["path"]: row for row in read_json(Path(config["attempt_root"]) / "source_manifest.json")["entries"]}
    for name in ("capture", "surface", "native_prediction"):
        index.identity(data[name], source_entries[data[name]])
    index.identity(model["text"]["path"], model["text"])
    index.identity(config["checkpoint"]["path"], config["checkpoint"])
    for path in sorted(Path(__file__).parent.glob("query*.py")):
        index.identity(path)
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "method": method,
                                 "inputs": index.entries(), "model": model["identity"], "budget": budget})
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["input_identity"] != identity or receipt["status"] != "COMPLETE":
            raise ValueError("query resume dependencies changed")
        for row in receipt["inputs"] + receipt["outputs"]:
            index.identity(row["path"], row)
        return receipt
    started = time.monotonic()
    if policy == "Q_GAIN" and budget == 200:
        for row in old["outputs"]:
            index.identity(row["path"], row)
        receipt = {**old, "input_identity": identity, "method_id": method,
                   "execution_mode": "REUSE_FROZEN_GAIN_B200", "historical_receipt": str(old_path),
                   "historical_physical": old["physical"],
                   "physical": {"model_loads": 0, "model_forwards": 0, "crop_inputs": 0,
                                "peak_gpu_allocated_bytes": None, "peak_memory_status": "NO_NEW_INFERENCE"},
                   "inputs": index.entries(), "elapsed_seconds": time.monotonic() - started}
        write_once(receipt_path, receipt)
        return receipt
    import torch

    with np.load(model["text"]["path"], allow_pickle=False) as arrays:
        text, valid_ids = arrays["text_embeddings"], tuple(map(int, arrays["valid_ids"]))
    if valid_ids != tuple(model["valid_ids"]):
        raise ValueError("query vocabulary changed")
    frames = CapturedFrames(data["capture"])
    loader = StudyLoader(frames, config, root / "query_cache/native", old, index)
    store = FeatureStore(loader)
    predictor = scaler = None
    if policy == "Q_GAIN":
        checkpoint = torch.load(config["checkpoint"]["path"], map_location="cpu", weights_only=False)
        if checkpoint["status"] != "COMPLETE":
            raise ValueError("query checkpoint is incomplete")
        scaler = QueryFeatureStandardizer(**checkpoint["scaler_state_dict"])
        predictor = lambda features: predict_query_gain(checkpoint["state_dict"], features)
    lock = Path(config["gpu_lock"]).with_name(f".visual-gpu-{gpu}.lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        require_idle_gpu(str(gpu), output)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        result = replay_study(frames, policy, store, text, budget=budget, seed=seed,
                              predictor=predictor, standardizer=scaler)
        # No final ownership data is opened before the last causal frame barrier.
        with np.load(data["surface"], allow_pickle=False) as surface:
            reconcile_export(result, surface, text)
        torch.cuda.synchronize()
        physical = {**asdict(store.physical_ledger), "model_loads": loader.model_loads,
                    "model_load_seconds": loader.model_load_seconds,
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "gpu": str(gpu), "peak_memory_status": "MEASURED"}
    native = load_prediction(data["native_prediction"])
    objects = query_objects(result["state"], sorted(owner_labels(native)), text, valid_ids)
    logical = asdict(result["state"].logical_ledger)
    if logical["attempts"] > budget or logical["crop_inputs"] != 6 * logical["attempts"]:
        raise ValueError("logical query budget/six-crop charging differs")
    costs = {"native_requests": logical["attempts"], "siglip2_requests": 0, "crop_inputs": logical["crop_inputs"]}
    prediction = relabel_prediction(native, method, "Q", {owner: row["label"] for owner, row in objects.items()},
                                    costs, {"budget": budget, "seed": seed, "frame_barrier": True,
                                            "final_reconciliation": True, "controller": policy})
    prediction_path = save_prediction(prediction, output / "prediction")
    evidence = source_bundle(scene, method, native, prediction, objects, model, index.entries(), costs)
    decisions = {"scene_id": scene, "method_id": method, "frames": result["decisions"],
                 "paid_requests": _paid_requests(data["capture"], result["decisions"]),
                 "retained_features": {str(owner): [row.request_id for row in obj.features]
                                       for owner, obj in result["state"].objects.items()},
                 "discarded_ambiguous_features": sorted(result["lineage"].dropped_feature_ids),
                 "lineage": result["lineage"].frame_diagnostics}
    write_once(output / "source_evidence.json", evidence)
    write_once(output / "decisions.json", decisions)
    write_once(output / "cache_imports.json", {"imports": loader.import_map})
    outputs = [index.identity(path) for path in (output / "source_evidence.json", output / "decisions.json",
               output / "cache_imports.json", prediction_path, prediction_path.parent / "prediction.npz")]
    receipt = {"status": "COMPLETE", "input_identity": identity, "scene_id": scene, "method_id": method,
               "policy": policy, "seed": seed, "budget": budget, "execution_mode": "NEW_CAUSAL_CONTROLLER",
               "logical": logical, "physical": physical, "required_logical": costs,
               "elapsed_seconds": time.monotonic() - started, "prediction_manifest": str(prediction_path),
               "evidence_path": str(output / "source_evidence.json"), "decisions_path": str(output / "decisions.json"),
               "inputs": index.entries(), "outputs": outputs}
    write_once(receipt_path, receipt)
    return receipt
