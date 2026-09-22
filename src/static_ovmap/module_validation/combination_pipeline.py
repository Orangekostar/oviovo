"""At most two frozen static/hybrid variants with separately paid matched controls."""

from __future__ import annotations

import fcntl
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from .boundary_jobs import file_identity
from .contracts import atomic_write_json, canonical_digest
from .evaluation import validate_prediction_invariants
from .frozen_semantic import prepare_frozen_semantic
from .geometry_models import final_mask_labels, load_static_records
from .geometry_pipeline import _quality_scores, prepare_geometry_prediction
from .geometry_study import load_geometry_pool, prepare_geometry_pool
from .query_pipeline import run_query_scene
from .scannet_runtime import require_idle_gpu
from .scannet_study import load_prediction, relabel_prediction
from .semantic_selector import area_readout
from .static_regions import prepare_static_manifest
from .study_execution import (
    ROOT,
    config_inputs,
    evaluation_outputs,
    read_json,
    receipt,
    reuse,
    roles,
    store_prediction,
    verify_receipt,
)
from .study_scene import bind_scene, evaluate_predictions

METHODS = ("COMBO_GS", "COMBO_Q_REFINEMENT")


def _sources():
    return [file_identity(Path(__file__).with_name(name)) for name in ("combination_pipeline.py", "frozen_semantic.py",
        "geometry_pipeline.py", "geometry_models.py", "geometry_study.py", "query_pipeline.py", "static_regions.py")]


def _aggregates(manifest, records, text, valid_ids):
    result = {}
    for target, requests in manifest["views"].items():
        good = [key for key in requests if records[key]["status"] == "COMPLETE"]
        if good:
            result[int(target.split(":")[1])] = area_readout(
                np.stack([records[key]["feature"] for key in good]),
                [manifest["requests"][key]["request"]["visible_target_pixels"] for key in good],
                text, valid_ids=valid_ids).aggregate_feature
    return result


def _query_readout(scene, role, method, modules, runtime, config, config_path, confirmation_lock):
    calibration = verify_receipt(Path(config["study_root"]) / "query/calibration_receipt.json")
    policy = "Q_GAIN" if method == "COMBO_Q_REFINEMENT" else modules["query"]["locked_comparator"]
    parent = method if role == "confirm" else None
    lock = Path(runtime["output_root"]).parent / f".visual-gpu-{runtime['cuda_device']}.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        require_idle_gpu(str(runtime["cuda_device"]), Path(config["study_root"]) / "selection")
        result = run_query_scene(scene, role, policy, runtime, config, config_path,
            checkpoint_path=calibration["checkpoint"]["path"] if policy == "Q_GAIN" else None,
            confirmation_lock=confirmation_lock, confirmation_parent=parent)
    leaf = "constituent_B200" if parent else "B200"
    output = Path(config["study_root"]) / "scenes" / scene / "query" / leaf / policy
    return load_prediction(output / "prediction/manifest.json"), output / "receipt.json", result


def _static_refinement(data, native_data, modules, runtime, config, config_path, output):
    """Incumbents are the supplied map (Q for hybrids), never another N0 map."""
    import torch

    output = Path(output)
    g, s = modules["geometry"]["selected_method"], modules["semantic"]["selected_method"]
    if g == "G_ORIGINAL" and s == "N0":
        raise ValueError("hybrid requires an independently retained static refinement")
    paths = []
    if g != "G_ORIGINAL":
        pool_path = prepare_geometry_pool(native_data, native_data["output"] / "geometry/pool")
        pool = load_geometry_pool(pool_path)
        calibration_path = Path(config["study_root"]) / "geometry/calibration_receipt.json"
        calibration = verify_receipt(calibration_path)
        checkpoint_path, scores = None, None
        if g == "G_QUALITY":
            checkpoint_path = calibration["checkpoint"]["path"]
            scores = _quality_scores(pool, torch.load(checkpoint_path, map_location="cpu", weights_only=False))
        native, _ = prepare_geometry_prediction(data, pool, g, runtime, config, config_path, output / "geometry",
            scores=scores, margin=calibration["margin"], checkpoint_path=checkpoint_path)
        native_path = output / "geometry/prediction/manifest.json"
        manifest_path = output / "geometry/requests/manifest.json"
        records = load_static_records(output / "geometry/inference")
        paths += [pool_path, calibration_path, output / "geometry/receipt.json"]
    else:
        # S-only hybrids explicitly pay for a native incumbent reread on each
        # actual final mask, preserving the causal map's owner vector and ranks.
        owners = data["native"].owner_ids
        ancestry = {int(owner): (int(owner),) for owner in np.unique(owners) if owner > 0}
        manifest_path = prepare_static_manifest(data, owners, ancestry, output / "native_requests")
        subprocess.run([runtime["native_perception_python"], str(ROOT / "scripts/evaluation/run_ovimap_geometry_model.py"),
            "--config", str(config_path), "--request-manifest", str(manifest_path), "--output", str(output / "native_inference")], check=True)
        records = load_static_records(output / "native_inference")
        with np.load(config["native_text_cache"], allow_pickle=False) as text:
            labels, semantic_ledger = final_mask_labels(data["native"], owners, read_json(manifest_path), records,
                text["text_embeddings"], tuple(map(int, text["valid_ids"])))
        costs = {**data["native"].logical_cost, "added_attempts": len(records), "added_crop_inputs": 6 * len(records),
                 "added_seconds": sum(row["elapsed_seconds"] for row in records.values())}
        native = relabel_prediction(data["native"], "PAID_STATIC_NATIVE", "S", labels, costs,
            {"request_identity": read_json(manifest_path)["identity"], "incumbent_readout": "paid_actual_final_mask"})
        native_path = store_prediction(native, output / "paid_native")
        ledger_path = output / "paid_native/decisions.json"
        atomic_write_json(ledger_path, semantic_ledger)
        paths += [output / "native_requests/receipt.json", output / "native_inference/receipt.json",
                  native_path, native_path.parent / "prediction.npz", ledger_path]
    if s == "N0":
        return native, paths
    manifest = read_json(manifest_path)
    with np.load(config["native_text_cache"], allow_pickle=False) as text:
        aggregates = _aggregates(manifest, records, text["text_embeddings"], tuple(map(int, text["valid_ids"])))
    refined_data = {**data, "native": native, "native_manifest_path": native_path}
    semantic_root = output / "semantic"
    result, _ = prepare_frozen_semantic(refined_data, s, runtime, config, config_path,
        Path(config["study_root"]) / "semantic/calibration_receipt.json", semantic_root / "evidence", semantic_root,
        manifest_path=manifest_path, native_aggregates=aggregates)
    return result, [*paths, semantic_root / "receipt.json"]


def run_combination_scene(scene, role, method, runtime, config, config_path, *, confirmation_lock=None):
    root = Path(config["study_root"])
    module_path = root / "selection/module_receipt.json"
    modules = verify_receipt(module_path)["decision"]
    split, lock_path = roles(runtime)
    normalized = "COMBO_Q_REFINEMENT" if method == "COMBO_Q_REFINEMENT_CONTROL" else method
    if normalized not in modules["combination_plan"]["required"]:
        raise ValueError("combination did not pass independent module retention")
    if role == "confirm":
        from .confirmation_access import require_confirmation_method

        require_confirmation_method(scene, method, runtime, config, confirmation_lock)
    elif role != "select" or scene not in split["select"]:
        raise ValueError("combinations are restricted to locked SELECT or authorized CONFIRM scenes")
    output = (root / "confirmation/scenes" / scene / method if role == "confirm"
              else root / "selection/combinations" / method / "scenes" / scene)
    query, query_path = None, None
    if normalized == "COMBO_Q_REFINEMENT":
        query, query_path, _ = _query_readout(scene, role, method, modules, runtime, config, config_path, confirmation_lock)
    # For a hybrid this is deliberately after its completed causal readout.
    native_data, targets = bind_scene(scene, runtime, config)
    data = native_data if query is None else {**native_data, "native": query,
        "native_manifest_path": query_path.parent / "prediction/manifest.json"}
    inputs = config_inputs(config, config_path) + [file_identity(path) for path in
        (module_path, lock_path, data["native_manifest_path"], *(root / branch / "calibration_receipt.json"
                                                             for branch in ("semantic", "geometry", "query")))]
    if query_path is not None:
        inputs.append(file_identity(query_path))
    if confirmation_lock is not None:
        inputs.append(file_identity(confirmation_lock))
    identity = canonical_digest({"inputs": inputs, "sources": _sources(), "method": method, "role": role})
    cached = reuse(output / "receipt.json", identity)
    if cached:
        return verify_receipt(output / "receipt.json")
    refined, outputs = _static_refinement(data, native_data, modules, runtime, config, config_path, output / "refinement")
    changed_geometry = modules["geometry"]["selected_method"] != "G_ORIGINAL"
    payload = replace(refined, method_id=method, branch="G" if changed_geometry else "Q",
        metadata={**refined.metadata, "components": {key: modules[key]["selected_method"] for key in ("semantic", "geometry", "query")},
            "query_policy": None if query is None else query.method_id,
            "pipeline_kind": "causal query plus paid static refinement" if query is not None else "static geometry and semantics",
            "cost_scope": "B200 plus separately charged refinement" if query is not None else "native mapping plus charged refinement"})
    validate_prediction_invariants(payload, native_data["native"])
    payload.lock()
    prediction_path = store_prediction(payload, output / "prediction")
    row = evaluate_predictions([payload], {scene: targets}, Path(runtime["upstream"]), output / "evaluation")[0]
    row.update(logical_cost=dict(payload.logical_cost), role=role,
        added_seconds=payload.logical_cost.get("added_seconds", 0.),
        effective_changes=int(np.count_nonzero(payload.semantic_labels != native_data["native"].semantic_labels)))
    outputs += [prediction_path, prediction_path.parent / "prediction.npz", *evaluation_outputs(output / "evaluation")]
    return receipt(output / "receipt.json", identity, inputs, outputs, _sources(), method_id=method,
        scene_id=scene, role=role, row=row, logical_cost=dict(payload.logical_cost), prediction_manifest=str(prediction_path))


def static_noninferiority(rows, baseline):
    left = {row["scene_id"]: row["metrics"] for row in rows}
    right = {row["scene_id"]: row["metrics"] for row in baseline}
    if set(left) != set(right) or len(left) != 2 or len(rows) != 2 or len(baseline) != 2:
        raise ValueError("static comparison requires the same two SELECT scenes")
    if any(values.get(metric) is None for values in (*left.values(), *right.values()) for metric in ("uap", "miou")):
        return "INCONCLUSIVE_UNDEFINED_STATIC_COMPARISON"
    passes = all(np.mean([values[metric] for values in left.values()]) >=
                 np.mean([values[metric] for values in right.values()]) - 1e-10 for metric in ("uap", "miou"))
    return "NONINFERIOR" if passes else "NOT_REQUIRED_STATIC_NONINFERIORITY_FAILED"


def run_combinations(runtime, config, config_path):
    root = Path(config["study_root"])
    module_path = root / "selection/module_receipt.json"
    modules = verify_receipt(module_path)["decision"]
    split, lock_path = roles(runtime)
    results, static_status = [], "NONINFERIOR"
    for method in modules["combination_plan"]["required"]:
        output = root / "selection/combinations" / method
        inputs = config_inputs(config, config_path) + [file_identity(module_path), file_identity(lock_path)]
        if method == "COMBO_GS":
            baseline_path = root / "geometry/select_receipt.json"
            baseline = [row for row in verify_receipt(baseline_path)["rows"] if row["method_id"] == "G_ORIGINAL"]
            inputs.append(file_identity(baseline_path))
        elif "COMBO_GS" in modules["combination_plan"]["required"]:
            inputs.append(file_identity(root / "selection/combinations/COMBO_GS/receipt.json"))
        identity = canonical_digest({"inputs": inputs, "sources": _sources(), "method": method})
        cached = reuse(output / "receipt.json", identity)
        if cached:
            result = verify_receipt(output / "receipt.json")
        elif method == "COMBO_Q_REFINEMENT" and static_status != "NONINFERIOR":
            gate_path = output / "gate.json"
            atomic_write_json(gate_path, {"status": static_status, "static_evidence": inputs[-1]})
            result = receipt(output / "receipt.json", identity, inputs, [gate_path], _sources(), method_id=method,
                disposition=static_status, rows=[], matched_rows=[], reason="frozen GS comparison did not pass static noninferiority")
        else:
            measured, outputs = [], []
            for scene in split["select"]:
                value = run_combination_scene(scene, "select", method, runtime, config, config_path)
                measured.append(value["row"])
                outputs.append(output / "scenes" / scene / "receipt.json")
            if method == "COMBO_Q_REFINEMENT":
                baseline = []
                for scene in split["select"]:
                    value = run_combination_scene(scene, "select", "COMBO_Q_REFINEMENT_CONTROL", runtime, config, config_path)
                    baseline.append(value["row"])
                    outputs.append(root / "selection/combinations/COMBO_Q_REFINEMENT_CONTROL/scenes" / scene / "receipt.json")
            result = receipt(output / "receipt.json", identity, inputs, outputs, _sources(),
                method_id=method, disposition="MEASURED", rows=measured, matched_rows=baseline)
        results.append(result)
        if method == "COMBO_GS":
            static_status = static_noninferiority(result["rows"], result["matched_rows"])
    return results
