"""Apply one CAL-frozen S choice to native or paid fresh-mask evidence, without GT."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from .boundary_jobs import file_identity
from .contracts import atomic_write_json, canonical_digest
from .native_capture import _write_npz
from .scannet_study import load_prediction, relabel_prediction
from .semantic_features import semantic_feature_rows
from .semantic_models import load_semantic_records
from .semantic_pipeline import _costs, _model, _read_text, selector_prediction
from .semantic_selector import (
    HEAD_IDS,
    TEACHER_IDS,
    FeatureStandardizer,
    adoption_values,
)
from .semantic_study import direct_readouts, prepare_semantic_manifest
from .study_execution import (
    ROOT,
    config_inputs,
    read_json,
    receipt,
    reuse,
    store_prediction,
    verify_receipt,
)


def _sources():
    return [file_identity(Path(__file__).with_name(name)) for name in ("frozen_semantic.py", "semantic_features.py",
        "semantic_models.py", "semantic_pipeline.py", "semantic_selector.py", "semantic_study.py")]


def prepare_frozen_semantic(data, method, runtime, config, config_path, calibration_path, evidence_root, output,
                            *, manifest_path=None, native_aggregates=None, run_models=True):
    """Use only the chosen teacher/head; caller enforces SELECT or confirmation access.

    Static refinements pass a real final-mask manifest and paid native aggregates.
    Missing incumbent aggregate evidence is a recorded technical inability of the
    frozen head for that object; it never triggers a free N0 feature read.
    """
    import torch

    calibration = verify_receipt(calibration_path)
    learned = method in HEAD_IDS
    if method not in {*HEAD_IDS, *TEACHER_IDS, "S_NATIVE_AREA", "S_NATIVE_VOTE"}:
        raise ValueError("unlisted frozen semantic method")
    if learned and (method not in calibration["heads"] or calibration["heads"][method]["status"] != "COMPLETE"):
        raise ValueError("selected semantic head lacks a completed CAL checkpoint and threshold")
    teacher = calibration["teacher_id"] if learned else method
    if method in TEACHER_IDS and method != calibration["teacher_id"]:
        raise ValueError("frozen semantic execution cannot reselect another teacher")
    model = _model(teacher)
    evidence_root, output = Path(evidence_root), Path(output)
    if manifest_path is None:
        manifest_path = evidence_root / "semantic_requests.json"
        prepare_semantic_manifest(data["capture_path"], data["native"], manifest_path)
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    static = manifest.get("artifact_type") == "OVIMAP_STATIC_FINAL_MASK_REQUESTS"
    if static and learned and native_aggregates is None:
        raise ValueError("fresh-mask frozen head requires explicitly paid native aggregates")
    models = tuple(dict.fromkeys(("native", model))) if learned else (model,)
    for name in models:
        if run_models:
            python = runtime["native_perception_python"] if name == "native" else runtime["semantic_python"]
            subprocess.run([python, str(ROOT / "scripts/evaluation/run_ovimap_semantic_model.py"),
                "--config", str(config_path), "--request-manifest", str(manifest_path), "--model", name,
                "--output", str(evidence_root / name)], check=True)
    model_paths = [evidence_root / name / "receipt.json" for name in models]
    for path in model_paths:
        verify_receipt(path)
    inputs = config_inputs(config, config_path) + [file_identity(path) for path in
        (calibration_path, manifest_path, data["native_manifest_path"], *model_paths)]
    aggregate_path = None
    if learned:
        if native_aggregates is None:
            aggregate_path = data["output"] / "baseline/native_readout.npz"
            with np.load(aggregate_path, allow_pickle=False) as arrays:
                native_aggregates = dict(zip(map(int, arrays["owner_ids"]), arrays["features"], strict=True))
        else:
            aggregate_path = evidence_root / "paid_native_aggregates.npz"
            owners = sorted(native_aggregates)
            arrays = {"owner_ids": np.asarray(owners, np.int64), "features": np.asarray([native_aggregates[owner] for owner in owners])}
            if aggregate_path.exists():
                with np.load(aggregate_path, allow_pickle=False) as previous:
                    if any(not np.array_equal(previous[key], value) for key, value in arrays.items()):
                        raise ValueError("paid native aggregate evidence changed")
            else:
                _write_npz(aggregate_path, arrays)
        inputs.append(file_identity(aggregate_path))
    identity = canonical_digest({"inputs": inputs, "sources": _sources(), "method": method,
                                  "native_prediction": data["native"].prediction_key})
    cached = reuse(output / "receipt.json", identity)
    if cached:
        return load_prediction(output / "prediction/manifest.json"), verify_receipt(output / "receipt.json")
    records = {name: load_semantic_records(evidence_root / name) for name in models}
    native_text, valid_ids = _read_text(config["native_text_cache"])
    teacher_text, teacher_ids = _read_text(config["siglip2_text_cache"]) if model == "siglip2" else (native_text, valid_ids)
    if teacher_ids != valid_ids:
        raise ValueError("frozen semantic text vocabularies differ")
    normalized = {**manifest, "requests": {key: value["request"] if static else value for key, value in manifest["requests"].items()}}
    native = data["native"]
    incumbent = {int(owner): int(native.semantic_labels[np.flatnonzero(native.owner_ids == owner)[0]])
                 for owner in np.unique(native.owner_ids) if owner > 0}
    suggestions = direct_readouts(normalized, records[model], teacher_text, valid_ids, incumbent, model=model)[teacher]
    costs = _costs(native, model, records, context=learned)
    for key in ("added_attempts", "added_crop_inputs", "added_background_inputs", "added_seconds", "generations"):
        if key in native.logical_cost:
            costs[key] = costs.get(key, 0) + native.logical_cost[key]
    metadata = {"teacher_id": teacher, "request_manifest": manifest["identity"], "frozen_calibration": file_identity(calibration_path),
                "fresh_mask_evidence": static}
    unavailable = []
    if learned:
        unavailable = sorted(owner for owner, row in suggestions.items() if not row["technical_fallback"]
                             and row["label_id"] != incumbent[owner] and owner not in native_aggregates)
        supported = {owner: {**row, "technical_fallback": True} if owner in unavailable else row
                     for owner, row in suggestions.items()}
        features = semantic_feature_rows(native, normalized, supported, records["native"], records[model],
            native_aggregates, native_text, teacher_text, valid_ids, teacher_model=model)
        settings = calibration["heads"][method]
        checkpoint = torch.load(settings["checkpoint"]["path"], map_location="cpu", weights_only=False)
        scaler = FeatureStandardizer(**checkpoint["scaler_state_dict"])
        owners = sorted(features)
        matrix = np.array([scaler.transform(features[owner]) for owner in owners]).reshape(-1, 64)
        start = time.monotonic()
        values = dict(zip(owners, map(float, adoption_values(method, checkpoint["state_dict"], matrix)), strict=True)) if owners else {}
        costs["selector_seconds"] = time.monotonic() - start
        metadata["checkpoint"] = settings["checkpoint"]
        payload, decisions = selector_prediction(native, method, supported, values, settings["threshold"], costs, metadata)
    else:
        payload = relabel_prediction(native, method, "S", {owner: row["label_id"] for owner, row in suggestions.items()}, costs, metadata)
        decisions = [{"owner_id": owner, "incumbent": incumbent[owner], "suggestion": row["label_id"],
            "technical_fallback": row["technical_fallback"], "accepted": row["label_id"] != incumbent[owner] and not row["technical_fallback"],
            "final_label": row["label_id"]} for owner, row in sorted(suggestions.items())]
    prediction_path = store_prediction(payload, output / "prediction")
    decision_path = output / "decisions.json"
    atomic_write_json(decision_path, {"decisions": decisions, "unavailable_native_aggregate_owners": unavailable,
        "prediction_key": payload.prediction_key, "prediction_locked_before_GT": True})
    outputs = [prediction_path, prediction_path.parent / "prediction.npz", decision_path]
    result = receipt(output / "receipt.json", identity, inputs, outputs, _sources(), method_id=method,
        scene_id=native.scene_id, prediction_manifest=str(prediction_path), logical_cost=dict(payload.logical_cost),
        effective_changes=sum(row["accepted"] for row in decisions), teacher_id=teacher,
        unavailable_native_aggregate_owners=unavailable)
    return load_prediction(prediction_path), result
