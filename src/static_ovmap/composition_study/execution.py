"""Authorized real query leaves, immutable results, and content-based resume."""

import fcntl
import gc
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.query_gain_policy import (
    QueryFeatureStandardizer,
    predict_query_gain,
)
from src.static_ovmap.module_validation.query_pipeline import reconcile_export
from src.static_ovmap.module_validation.query_state import FeatureStore
from src.static_ovmap.module_validation.query_study import (
    CapturedFrames,
    replay_captured,
)
from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu
from src.static_ovmap.module_validation.scannet_study import (
    load_prediction,
    relabel_prediction,
    save_prediction,
)

from .io import ROOT, SourceIndex, read_json, write_once
from .mixed_query import replay_mixed
from .object_evidence import owner_labels, query_objects, source_bundle
from .selection import COMPOSITIONS, CONTROLS
from .trajectory_reread import replay_fixed
from .visual_requests import VisualRequestLoader

READERS = {
    "CP_M3_COMBINE_S2": "Q_COMBINE",
    "CP_M4_GAIN_S2": "Q_GAIN",
    "CP_M6_MIX50_S2": "CP_M5_MIX50_NATIVE",
}


def require_access(config, scene, phase, method=None):
    roles = config["spec"]["data"]
    role = next(
        (
            key
            for key in ("compose_cal", "regression_only", "confirmation")
            if scene in roles[key]
        ),
        None,
    )
    expected = {
        "prepare-cal": "compose_cal",
        "calibrate": "compose_cal",
        "compose-cal": "compose_cal",
        "regression": "regression_only",
        "confirm": "confirmation",
    }.get(phase)
    if role is None or expected != role:
        raise ValueError("phase does not authorize this scene role")
    if method is not None and method not in (*CONTROLS, *COMPOSITIONS):
        raise ValueError("unknown composition method")
    if phase == "prepare-cal" and method is not None and method not in CONTROLS:
        raise ValueError("prepare-cal only authorizes source controls")
    if phase == "calibrate" and method is not None:
        raise ValueError("calibration does not authorize prediction leaf jobs")
    if config.get("parent_attempt_root") and phase != "confirm":
        raise ValueError("confirmation sub-attempt cannot launch development jobs")
    root = Path(config["attempt_root"])
    if role != "compose_cal":
        path = root / "selection.json"
        if not path.is_file():
            raise ValueError("new composition selection must be frozen first")
        lock = read_json(path)
        if lock["status"] != "FROZEN" or lock["binding_key"] != config["binding_key"]:
            raise ValueError("composition selection binding changed")
        if (
            "identity" in lock
            and canonical_digest(
                {key: value for key, value in lock.items() if key != "identity"}
            )
            != lock["identity"]
        ):
            raise ValueError("composition selection content changed")
        for key in ("models", "checkpoint", "runtime", "gpu_lock"):
            if lock.get(key) != config.get(key):
                raise ValueError(f"frozen composition {key} changed")
        index = SourceIndex()
        for entry in lock["code"]["files"]:
            index.identity(entry["path"], entry)
        if method == COMPOSITIONS[-1] and not lock["m6_gate"]["enabled"]:
            raise ValueError("M6 CAL gate is closed")
        if role == "confirmation":
            if config["confirmation_exposure"]["blocked_scenes"]:
                raise ValueError("BLOCKED_CONFIRMATION_EXPOSURE")
            if (
                method is not None
                and method not in lock["nomination"]["confirmation_methods"]
            ):
                raise ValueError("method is outside frozen confirmation contract")
    elif method == COMPOSITIONS[-1]:
        gate = root / "calibration/m6_gate.json"
        if not gate.is_file() or not read_json(gate)["enabled"]:
            raise ValueError("M6 CAL gate is closed")
    return role


def kernel_identity():
    """Only real query/readout dependencies, not later report or HEAD changes."""
    local = (
        "execution.py",
        "object_evidence.py",
        "trajectory_reread.py",
        "mixed_query.py",
        "visual_requests.py",
    )
    old = (
        "query_study.py",
        "query_state.py",
        "query_lineage.py",
        "query_gain_policy.py",
        "query_pipeline.py",
        "native_capture.py",
        "region_evidence.py",
        "rgb_siglip.py",
        "scannet_study.py",
    )
    index = SourceIndex()
    paths = [Path(__file__).with_name(name) for name in local]
    paths += [ROOT / "src/static_ovmap/module_validation" / name for name in old]
    entries = [index.identity(path) for path in paths if path.is_file()]
    return {"files": entries, "digest": canonical_digest(entries)}


def verified_receipt(path, *, identity=None):
    row = read_json(path)
    if row["status"] != "COMPLETE" or (
        identity is not None and row["input_identity"] != identity
    ):
        raise ValueError(f"completed job identity differs: {path}; use a new attempt")
    index = SourceIndex()
    for entry in (*row.get("inputs", []), *row["outputs"]):
        index.identity(entry["path"], entry)
    return row


def _parity_reuse(config, scene, method, index):
    path = (
        Path(config["attempt_root"])
        / "technical/native_parity"
        / scene
        / "receipt.json"
    )
    if method != "Q_COMBINE" or not path.is_file():
        return None
    row = read_json(path)
    if (
        row["status"] != "PASS"
        or not row["exact_labels"]
        or not row["exact_retained_ids"]
    ):
        raise ValueError("native parity check did not pass")
    relevant = {
        "trajectory_reread.py",
        "visual_requests.py",
        "query_study.py",
        "query_state.py",
        "query_lineage.py",
        "query_pipeline.py",
        "native_capture.py",
        "region_evidence.py",
        "rgb_siglip.py",
    }
    for entry in row["code"]["files"]:
        if Path(entry["path"]).name in relevant:
            index.identity(entry["path"], entry)
    for entry in [row["source_trace"], *row["outputs"]]:
        index.identity(entry["path"], entry)
    index.identity(path)
    objects = {
        int(owner): {"owner_id": int(owner), **value}
        for owner, value in read_json(path.with_name("source_evidence.json"))[
            "objects"
        ].items()
    }
    return row, objects, read_json(path.with_name("decisions.json")), path


def _paid_requests(capture_path, decisions):
    capture = read_json(capture_path)
    frames = {row["frame_id"]: row for row in capture["frames"]}
    paid = []
    for frame in decisions:
        by_id = {
            row["request_id"]: row
            for row in frames.get(frame["frame_id"], {}).get("requests", [])
        }
        for result in frame["results"]:
            paid.append(
                {
                    "frame_index": frame["frame_index"],
                    "frame_id": frame["frame_id"],
                    "request": by_id[result["request_id"]],
                }
            )
    return paid


def run_query(config, scene, method, phase):
    role = require_access(config, scene, phase, method)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(config["runtime"]["cuda_device"]):
        raise ValueError(
            "query leaf must bind the configured GPU before importing models"
        )
    if method not in ("Q_COMBINE", "Q_GAIN", COMPOSITIONS[5], *READERS):
        raise ValueError("not a query execution method")
    data = config["scenes"][scene]
    root = Path(config["attempt_root"])
    output = root / "query" / scene / method
    index = SourceIndex()
    source_entries = {
        row["path"]: row for row in read_json(root / "source_manifest.json")["entries"]
    }
    for name in ("capture", "surface", "native_prediction"):
        path = data[name]
        index.identity(path, source_entries[path])
    model_name = "siglip2" if method in READERS else "native"
    model = config["models"][model_name]
    index.identity(model["text"]["path"], model["text"])
    index.identity(config["checkpoint"]["path"], config["checkpoint"])
    controller = None
    trace = None
    old_prediction_path = None
    if method in READERS:
        controller = verified_receipt(
            root / "query" / scene / READERS[method] / "receipt.json"
        )
        trace_path = Path(controller["decisions_path"])
        index.identity(trace_path)
        trace = read_json(trace_path)
    elif (
        method in ("Q_COMBINE", "Q_GAIN")
        and data["controls"][method]["status"] == "BOUND"
    ):
        old_prediction_path = Path(data["controls"][method]["prediction"])
        trace_path = old_prediction_path.parent.parent / "decisions.json"
        old_receipt = read_json(trace_path.with_name("receipt.json"))
        index.expected_output(trace_path, old_receipt)
        index.identity(old_prediction_path, source_entries[str(old_prediction_path)])
        trace = read_json(trace_path)
    code = kernel_identity()
    identity = canonical_digest(
        {
            "binding": config["binding_key"],
            "scene": scene,
            "method": method,
            "code": code["digest"],
            "inputs": index.manifest(),
            "model": model["identity"],
            "budget": 200,
            "controller": None if controller is None else controller["input_identity"],
        }
    )
    receipt_path = output / "receipt.json"
    if receipt_path.is_file():
        return verified_receipt(receipt_path, identity=identity)
    started = time.monotonic()
    with np.load(model["text"]["path"], allow_pickle=False) as arrays:
        text, valid_ids = (
            arrays["text_embeddings"],
            tuple(map(int, arrays["valid_ids"])),
        )
    if valid_ids != tuple(model["valid_ids"]):
        raise ValueError("query text vocabulary changed")
    parity = _parity_reuse(config, scene, method, index)
    loader = None
    if parity is not None:
        check, objects, decisions, parity_path = parity
        logical, physical = check["logical"], check["physical"]
        physical = {
            **physical,
            "reused_integration_check": str(parity_path),
            "charged_once": True,
        }
        import_map = check["import_map"]
        execution_mode = "REUSE_REQUIRED_NATIVE_PARITY"
    else:
        frames = CapturedFrames(data["capture"])
        loader = VisualRequestLoader(
            frames, config, model_name, root / "cache" / model_name
        )
        store = FeatureStore(loader)
        predictor = scaler = None
        if method in ("Q_GAIN", COMPOSITIONS[5]) and trace is None:
            import torch

            checkpoint = torch.load(
                config["checkpoint"]["path"], map_location="cpu", weights_only=False
            )
            if checkpoint["status"] != "COMPLETE":
                raise ValueError("frozen Q checkpoint is incomplete")
            scaler = QueryFeatureStandardizer(**checkpoint["scaler_state_dict"])
            predictor = lambda features: predict_query_gain(
                checkpoint["state_dict"], features
            )
        # The same historical lock serializes this task with existing GPU workers.
        with Path(config["gpu_lock"]).open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            require_idle_gpu(str(config["runtime"]["cuda_device"]), output)
            if trace is not None:
                result = replay_fixed(frames, trace, store, text)
            elif method == COMPOSITIONS[5]:
                result = replay_mixed(
                    frames, store, text, predictor=predictor, standardizer=scaler
                )
            else:
                result = replay_captured(
                    frames,
                    method,
                    store,
                    text,
                    budget=200,
                    predictor=predictor,
                    standardizer=scaler,
                )
            # Final surface arrays are first opened after the last frame barrier.
            with np.load(data["surface"], allow_pickle=False) as surface:
                reconcile_export(result, surface, text)
            store.physical_ledger.model_loads = loader.model_loads
            logical, physical = (
                asdict(result["state"].logical_ledger),
                asdict(store.physical_ledger),
            )
            physical["model_load_seconds"] = loader.model_load_seconds
            if loader.backend is not None:
                import torch

                loader.backend = None
                gc.collect()
                torch.cuda.empty_cache()
        native = load_prediction(data["native_prediction"])
        owners = sorted(owner_labels(native))
        objects = query_objects(result["state"], owners, text, valid_ids)
        decisions = {
            "scene_id": scene,
            "method_id": method,
            "frames": result["decisions"],
            "paid_requests": result.get("paid_requests")
            or _paid_requests(data["capture"], result["decisions"]),
            "retained_features": {
                str(owner): [row.request_id for row in obj.features]
                for owner, obj in result["state"].objects.items()
            },
            "discarded_ambiguous_features": sorted(
                result["lineage"].dropped_feature_ids
            ),
            "lineage": result["lineage"].frame_diagnostics,
            "lanes": result.get("lane_ledger", []),
        }
        import_map = loader.import_map
        if (
            old_prediction_path is not None
            and decisions["retained_features"] != trace["retained_features"]
        ):
            raise ValueError("native replay retained IDs differ from frozen controller")
        execution_mode = (
            "FORCED_TRAJECTORY" if trace is not None else "NEW_CAUSAL_CONTROLLER"
        )
    native = load_prediction(data["native_prediction"])
    labels = {owner: row["label"] for owner, row in objects.items()}
    if logical["attempts"] > 200 or logical["crop_inputs"] != 6 * logical["attempts"]:
        raise ValueError("query exceeds B200 or violates six-crop charging")
    costs = {
        "native_requests": logical["attempts"]
        if model_name == "native"
        else controller["logical"]["attempts"],
        "siglip2_requests": logical["attempts"] if model_name == "siglip2" else 0,
    }
    costs["crop_inputs"] = 6 * (costs["native_requests"] + costs["siglip2_requests"])
    if old_prediction_path is not None:
        prediction_path = old_prediction_path
        prediction = load_prediction(prediction_path)
        if owner_labels(prediction) != labels:
            raise ValueError("native replay labels differ from frozen controller")
    else:
        prediction = relabel_prediction(
            native,
            method,
            "Q",
            labels,
            costs,
            {
                "budget": 200,
                "frame_barrier": True,
                "final_reconciliation": True,
                "controller": READERS.get(method, method),
                "reader": model_name,
            },
        )
        prediction_path = save_prediction(prediction, output / "prediction")
    evidence = source_bundle(
        scene,
        method,
        native,
        prediction,
        objects,
        model,
        index.manifest()["entries"],
        costs,
    )
    write_once(output / "source_evidence.json", evidence)
    write_once(output / "decisions.json", decisions)
    write_once(output / "cache_imports.json", {"imports": import_map})
    if loader is not None:
        for entry in loader.index.manifest()["entries"]:
            index.identity(entry["path"], entry)
    outputs = [
        index.identity(path)
        for path in (
            output / "source_evidence.json",
            output / "decisions.json",
            output / "cache_imports.json",
            prediction_path,
            prediction_path.parent / "prediction.npz",
        )
    ]
    receipt = {
        "status": "COMPLETE",
        "input_identity": identity,
        "scene_id": scene,
        "role": role,
        "method_id": method,
        "model": model_name,
        "model_identity": model["identity"],
        "execution_mode": execution_mode,
        "logical": logical,
        "required_logical": costs,
        "physical": physical,
        "elapsed_seconds": time.monotonic() - started,
        "prediction_manifest": str(prediction_path),
        "evidence_path": str(output / "source_evidence.json"),
        "decisions_path": str(output / "decisions.json"),
        "code": code,
        "inputs": index.manifest()["entries"],
        "outputs": outputs,
        "controller_receipt": None
        if controller is None
        else controller["input_identity"],
    }
    write_once(receipt_path, receipt)
    return receipt
