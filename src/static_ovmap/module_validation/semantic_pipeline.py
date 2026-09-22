"""Frozen ScanNet S execution: acquired evidence, CAL settings, then SELECT."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .assets import sha256_file
from .boundary_jobs import file_identity
from .contracts import atomic_write_json, canonical_digest
from .native_capture import _write_npz
from .scannet_runtime import reusable_job
from .scannet_study import load_prediction, relabel_prediction, save_prediction
from .semantic_features import semantic_feature_rows
from .semantic_models import load_semantic_records
from .semantic_selector import (
    ADOPTION_THRESHOLDS,
    HEAD_IDS,
    TEACHER_IDS,
    FeatureStandardizer,
    adoption_values,
    apply_adoption_threshold,
    event_support_status,
    inverse_scene_weights,
    select_adoption_threshold,
    select_teacher,
    train_semantic_head,
)
from .semantic_study import direct_readouts, prepare_semantic_manifest
from .study_scene import bind_scene, evaluate_predictions
from .study_targets import semantic_events

DIRECT_IDS = ("S_NATIVE_AREA", "S_NATIVE_VOTE", *TEACHER_IDS)
SOURCE_NAMES = ("semantic_pipeline.py", "semantic_features.py", "semantic_models.py", "semantic_study.py",
                "semantic_selector.py", "study_targets.py", "study_scene.py", "evaluation.py")


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json(path):
    return json.loads(Path(path).read_text())


def _sources():
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__).with_name(name) for name in SOURCE_NAMES]
    paths += [root / "scripts/evaluation" / name for name in (
        "run_ovimap_scannet_semantic.py", "run_ovimap_semantic_model.py")]
    paths.append(root / "docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json")
    return [file_identity(path) for path in paths]


def _config_inputs(config, config_path):
    runtime_path = Path(config["runtime_config"])
    if not runtime_path.is_absolute():
        runtime_path = Path(__file__).resolve().parents[3] / runtime_path
    return [file_identity(path) for path in (config_path, runtime_path)]


def _identity(inputs, **settings):
    return canonical_digest({"inputs": inputs, "settings": settings, "sources": _sources()})


def _complete(path: Path, identity: str):
    if reusable_job(path, identity):
        return _json(path)
    if path.exists():
        raise ValueError(f"completed S artifact changed; choose a new output root: {path}")
    return None


def _receipt(path, identity, inputs, outputs, **values):
    result = {"status": "COMPLETE", "input_identity": identity, "inputs": inputs,
              "sources": _sources(), "outputs": [file_identity(p) for p in outputs],
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                  cwd=Path(__file__).resolve().parents[3], text=True).strip(), "command": sys.argv, **values}
    atomic_write_json(path, _plain(result))
    return _plain(result)


def _evaluation_outputs(path):
    return sorted(file for file in Path(path).rglob("*") if file.is_file())


def _store_prediction(payload, directory):
    """Resume exact predictions while preserving their original measured timings."""
    path = Path(directory) / "manifest.json"
    if path.exists():
        previous = load_prediction(path)
        def stable_costs(row):
            return {key: value for key, value in row.logical_cost.items() if not key.endswith("seconds")}
        if (previous.prediction_key != payload.prediction_key or previous.method_id != payload.method_id
                or previous.metadata != payload.metadata or stable_costs(previous) != stable_costs(payload)):
            raise ValueError("frozen semantic prediction changed on resume")
        return path
    return save_prediction(payload, directory)


def _verified_receipt(path, seen=None):
    path = Path(path)
    value = _json(path)
    seen = set() if seen is None else seen
    if path.resolve() in seen:
        return value
    if not reusable_job(path, value["input_identity"]):
        raise ValueError(f"S artifact is incomplete or changed: {path}")
    seen.add(path.resolve())
    for row in [*value.get("inputs", []), *value.get("sources", [])]:
        if sha256_file(row["path"]) != row["sha256"]:
            raise ValueError(f"S input changed after its receipt: {row['path']}")
        source = Path(row["path"])
        if source.suffix == ".json" and source.stem.endswith("receipt"):
            _verified_receipt(source, seen)
    return value


def study_roles(runtime):
    lock_path = Path(runtime["data_root"]) / "acquisition_lock.json"
    lock = _json(lock_path)
    roles = {role: tuple(row["scene_id"] for row in lock["selected"] if row["role"] == role)
             for role in ("fit", "cal", "select", "confirm")}
    if tuple(map(len, roles.values())) != (8, 2, 2, 2):
        raise ValueError("S requires the frozen 8/2/2/2 physical-family split")
    return roles, lock_path


def calibrated_teacher(cal_results: list[dict], cal_scenes: tuple[str, ...]) -> dict:
    """Reject wrong roles, partial scene sets and all-fallback alternatives."""
    if (len(cal_results) != len(cal_scenes) or {row["scene_id"] for row in cal_results} != set(cal_scenes)
            or any(row["role"] != "cal" for row in cal_results)):
        raise ValueError("teacher selection requires exactly the locked CAL scenes")
    candidates, undefined = [], []
    for method in TEACHER_IDS:
        rows = [next(row for row in scene["rows"] if row["method_id"] == method) for scene in cal_results]
        if not any(row["successful_requests"] for row in rows):
            continue
        if any(row["metrics"].get(name) is None for row in rows for name in ("uap", "miou")):
            undefined.append(method)
            continue
        candidates.append({"method": method, "mean_uap": float(np.mean([row["metrics"]["uap"] for row in rows])),
            "mean_miou": float(np.mean([row["metrics"]["miou"] for row in rows])),
            "median_cost": float(np.median([row["added_seconds"] for row in rows]))})
    return {"teacher_id": select_teacher(candidates) if candidates and not undefined else None,
            "candidates": candidates, "undefined_candidates": undefined,
            "selection_scenes": list(cal_scenes), "status": "INCONCLUSIVE_UNDEFINED_CAL_METRIC" if undefined
            else "COMPLETE" if candidates else "BLOCKED_NO_AVAILABLE_TEACHER"}


def selector_prediction(native, head_id, suggestions, values, threshold, costs, metadata=None):
    labels, decisions = {}, []
    for owner, suggestion in sorted(suggestions.items()):
        incumbent = int(np.unique(native.semantic_labels[native.owner_ids == owner])[0])
        proposed = int(suggestion["label_id"])
        value = values.get(owner)
        accepted = bool(value is not None and incumbent != proposed and not suggestion["technical_fallback"]
                        and apply_adoption_threshold([value], threshold)[0])
        labels[owner] = proposed if accepted else incumbent
        decisions.append({"owner_id": int(owner), "incumbent": incumbent, "suggestion": proposed,
            "technical_fallback": suggestion["technical_fallback"], "adoption_value": value,
            "accepted": accepted, "final_label": labels[owner]})
    payload = relabel_prediction(native, head_id, "S", labels, costs,
                                  {"threshold": threshold, **(metadata or {})})
    return payload, decisions


def _model(method):
    return "wow" if method.startswith("S_WOW") else "siglip2" if method.startswith("S_SIGLIP2") else "native"


def _read_text(path):
    with np.load(path, allow_pickle=False) as rows:
        return rows["text_embeddings"], tuple(map(int, rows["valid_ids"]))


def _records_and_readouts(data, config):
    manifest = _json(data["output"] / "semantic_requests.json")
    native = data["native"]
    incumbent = {int(owner): int(native.semantic_labels[np.flatnonzero(native.owner_ids == owner)[0]])
                 for owner in np.unique(native.owner_ids) if owner > 0}
    native_text, valid_ids = _read_text(config["native_text_cache"])
    alternative_text, alternate_ids = _read_text(config["siglip2_text_cache"])
    if valid_ids != alternate_ids:
        raise ValueError("semantic model text vocabularies differ")
    records, readouts = {}, {}
    for model in ("native", "siglip2", "wow"):
        records[model] = load_semantic_records(data["output"] / "semantic_models" / model)
        text = alternative_text if model == "siglip2" else native_text
        readouts.update(direct_readouts(manifest, records[model], text, valid_ids, incumbent, model=model))
    return manifest, records, readouts, native_text, alternative_text, valid_ids


def _costs(native, model, records, *, context=False):
    selected = records[model]
    seconds = sum(row.get("six_crop_seconds", row["elapsed_seconds"]) if model == "native" and not context
                  else row["elapsed_seconds"] for row in selected.values())
    costs = dict(native.logical_cost)
    costs.update(added_attempts=len(selected), added_crop_inputs=sum(row["crop_inputs"] for row in selected.values()),
        added_background_inputs=sum(row["background_crop_inputs"] for row in selected.values()) if context else 0,
        generations=sum(row["generations"] for row in selected.values()), added_seconds=seconds)
    if context and model != "native":
        native_cost = _costs(native, "native", records, context=True)
        for key in ("added_attempts", "added_crop_inputs", "added_background_inputs", "added_seconds"):
            costs[key] += native_cost[key]
    return costs


def prepare_semantic_scene(scene, role, runtime, config, config_path, *, run_models=True):
    """Acquire direct evidence and lock all predictions before annotation diagnostics."""
    roles, lock_path = study_roles(runtime)
    if role not in {"fit", "cal", "select"} or scene not in roles[role]:
        raise ValueError("S development scene is outside its frozen role")
    branch = Path(config["study_root"]) / "semantic"
    if role == "select":
        _verified_receipt(branch / "calibration_receipt.json")
    data, targets = bind_scene(scene, runtime, config)
    output = data["output"] / "semantic"
    manifest_path = data["output"] / "semantic_requests.json"
    prepare_semantic_manifest(data["capture_path"], data["native"], manifest_path)
    if run_models:
        root = Path(__file__).resolve().parents[3]
        for model in ("native", "siglip2", "wow"):
            python = runtime["native_perception_python"] if model == "native" else runtime["semantic_python"]
            subprocess.run([python, str(root / "scripts/evaluation/run_ovimap_semantic_model.py"),
                "--config", str(config_path), "--request-manifest", str(manifest_path), "--model", model,
                "--output", str(data["output"] / "semantic_models" / model)], check=True)
    model_paths = [data["output"] / "semantic_models" / model / "receipt.json" for model in ("native", "siglip2", "wow")]
    for path in model_paths:
        _verified_receipt(path)
    inputs = _config_inputs(config, config_path) + [file_identity(path) for path in [lock_path,
        data["native_manifest_path"], data["output"] / "baseline/receipt.json", manifest_path, *model_paths]]
    identity = _identity(inputs, scene=scene, role=role)
    cached = _complete(output / "direct_receipt.json", identity)
    if cached:
        return cached
    manifest, records, readouts, _, _, valid_ids = _records_and_readouts(data, config)
    payloads, paths = [data["native"]], []
    ledger = {}
    for method in DIRECT_IDS:
        suggestions = readouts[method]
        costs = _costs(data["native"], _model(method), records)
        payload = relabel_prediction(data["native"], method, "S",
            {owner: row["label_id"] for owner, row in suggestions.items()}, costs,
            {"request_manifest": manifest["identity"], "readout_id": method})
        path = _store_prediction(payload, output / "direct" / method)
        paths += [path, path.parent / "prediction.npz"]
        payloads.append(payload)
        ledger[method] = suggestions
    ledger_path = output / "direct_suggestions.json"
    atomic_write_json(ledger_path, _plain({"suggestions": ledger, "request_manifest": manifest["identity"]}))
    paths.append(ledger_path)
    rows = evaluate_predictions(payloads, {scene: targets}, Path(runtime["upstream"]), output / "direct_evaluation")
    paths += _evaluation_outputs(output / "direct_evaluation")
    events = {method: semantic_events(data["native"], readouts[method], targets["nearest"], targets["matched"],
        np.load(targets["gt_instance_path"], allow_pickle=False), valid_ids) for method in DIRECT_IDS}
    events_path = output / "direct_evaluation/object_events.json"
    atomic_write_json(events_path, {"events": events, "predictions_locked_before_evaluation": True})
    paths.append(events_path)
    for row, payload in zip(rows, payloads, strict=True):
        method = row["method_id"]
        row["added_seconds"] = payload.logical_cost.get("added_seconds", 0.)
        row["logical_cost"] = dict(payload.logical_cost)
        row["effective_changes"] = 0 if method == "N0" else sum(
            event["incumbent_label"] != event["suggestion_label"] for event in events[method])
        row["successful_requests"] = 0 if method == "N0" else sum(
            record["status"] == "COMPLETE" for record in records[_model(method)].values())
    return _receipt(output / "direct_receipt.json", identity, inputs, paths, scene_id=scene, role=role, rows=rows)


def _features_for_scene(data, config, teacher):
    manifest, records, readouts, native_text, alternative_text, valid_ids = _records_and_readouts(data, config)
    with np.load(data["output"] / "baseline/native_readout.npz", allow_pickle=False) as arrays:
        aggregates = dict(zip(map(int, arrays["owner_ids"]), arrays["features"], strict=True))
    model = _model(teacher)
    rows = semantic_feature_rows(data["native"], manifest, readouts[teacher], records["native"], records[model],
        aggregates, native_text, alternative_text, valid_ids, teacher_model=model)
    return rows, readouts[teacher], _costs(data["native"], model, records, context=True), manifest


def calibrate_semantic(runtime, config, config_path):
    """Train on FIT only; teacher, checkpoint and threshold selection use CAL only."""
    import torch

    roles, lock_path = study_roles(runtime)
    output = Path(config["study_root"]) / "semantic"
    scene_root = Path(config["study_root"]) / "scenes"
    direct_paths = [scene_root / scene / "semantic/direct_receipt.json" for scene in (*roles["fit"], *roles["cal"])]
    direct = [_verified_receipt(path) for path in direct_paths]
    inputs = _config_inputs(config, config_path) + [file_identity(path) for path in [lock_path, *direct_paths]]
    identity = _identity(inputs, phase="FIT_CAL")
    cached = _complete(output / "calibration_receipt.json", identity)
    if cached:
        return cached
    choice = calibrated_teacher([row for row in direct if row["role"] == "cal"], roles["cal"])
    choice.update(inputs=inputs, sources=_sources(), settings={key: config[key] for key in (
        "native_model", "siglip2_model", "wow_model", "wow_code", "name_model")})
    teacher_path = output / "teacher_selection.json"
    if teacher_path.exists() and _json(teacher_path) != choice:
        raise ValueError("frozen CAL teacher selection changed")
    atomic_write_json(teacher_path, choice)
    paths = [teacher_path]
    if choice["teacher_id"] is None:
        return _receipt(output / "calibration_receipt.json", identity, inputs, paths,
            teacher_id=None, learned_status=choice["status"], heads={})
    teacher = choice["teacher_id"]
    examples = {"fit": [], "cal": []}
    features_by_scene, events_by_scene = {}, {}
    for role in ("fit", "cal"):
        for scene in roles[role]:
            data, targets = bind_scene(scene, runtime, config)
            features, suggestions, _, _ = _features_for_scene(data, config, teacher)
            events = _json(data["output"] / "semantic/direct_evaluation/object_events.json")["events"][teacher]
            features_by_scene[scene], events_by_scene[scene] = features, events
            for event in events:
                if event["event"] is not None:
                    examples[role].append({**event, "features": features[event["owner_id"]]})
            feature_path = output / "features" / f"{scene}.npz"
            owner_ids = list(features)
            _write_npz(feature_path, {"owner_ids": np.asarray(owner_ids, np.int64),
                "values": np.asarray([features[owner].values for owner in owner_ids]).reshape(-1, 32),
                "available": np.asarray([features[owner].available for owner in owner_ids], bool).reshape(-1, 32)})
            paths.append(feature_path)
            del data, targets
    support = event_support_status(examples["fit"])
    support_path = output / "event_support.json"
    atomic_write_json(support_path, {"status": support, "counts": dict(Counter(row["event"] for row in examples["fit"])),
        "scenes_by_event": {event: sorted({row["scene_id"] for row in examples["fit"] if row["event"] == event})
            for event in ("GAIN", "HARM", "OTHER")},
        "examples": {role: [{key: value for key, value in row.items() if key != "features"} for row in rows]
                     for role, rows in examples.items()}})
    paths.append(support_path)
    if support != "SUPPORTED":
        return _receipt(output / "calibration_receipt.json", identity, inputs, paths,
            teacher_id=teacher, learned_status=support, heads={})
    standardizer = FeatureStandardizer.fit([row["features"] for row in examples["fit"]])
    matrices = {role: np.asarray([standardizer.transform(row["features"]) for row in rows]).reshape(-1, 64)
                for role, rows in examples.items()}
    cal_contexts = {}
    for scene in roles["cal"]:
        data, targets = bind_scene(scene, runtime, config)
        _, suggestions, costs, manifest = _features_for_scene(data, config, teacher)
        cal_contexts[scene] = (data, targets, suggestions, costs, manifest)
    heads = {}
    for head in HEAD_IDS:
        head_dir = output / "heads" / head
        checkpoint_path = head_dir / "checkpoint.pt"
        training_path = head_dir / "training.json"
        training_identity = _identity(inputs, teacher=teacher, head=head)
        if training_path.exists():
            trained = _verified_receipt(training_path)
            if trained["input_identity"] != training_identity:
                raise ValueError("S training inputs changed after checkpoint freeze")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        else:
            fit_started = time.monotonic()
            result = train_semantic_head(head, matrices["fit"], [row["event"] for row in examples["fit"]],
                inverse_scene_weights([row["scene_id"] for row in examples["fit"]]),
                cal_features=matrices["cal"], cal_labels=[row["event"] for row in examples["cal"]])
            checkpoint = asdict(result)
            checkpoint["scaler_state_dict"] = {name: getattr(standardizer, name) for name in ("mean", "std", "fitted")}
            checkpoint.update(teacher_id=teacher, input_identity=training_identity, sources=_sources(), seed=17,
                feature_schema="semantic_selector_32_scalar_values_plus_32_availability_v1", torch_version=torch.__version__)
            head_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            trained = _receipt(training_path, training_identity, inputs, [checkpoint_path],
                training_status=result.status, checkpoint_epoch=result.checkpoint_epoch, fit_epochs=result.fit_epochs,
                calibration_nll=result.calibration_nll, parameter_counts=result.parameter_counts,
                training_seconds=time.monotonic() - fit_started)
        paths += [training_path, checkpoint_path]
        scores = {}
        for scene in roles["cal"]:
            features = features_by_scene[scene]
            owners = list(features)
            matrix = np.asarray([standardizer.transform(features[owner]) for owner in owners]).reshape(-1, 64)
            started = time.monotonic()
            values = dict(zip(owners, map(float, adoption_values(head, checkpoint["state_dict"], matrix)), strict=True)) if owners else {}
            scores[scene] = (values, time.monotonic() - started)
        threshold_rows = []
        for threshold in ADOPTION_THRESHOLDS:
            token = str(threshold)
            scene_rows, harmful, replacements = [], 0, 0
            for scene in roles["cal"]:
                data, targets, suggestions, base_costs, manifest = cal_contexts[scene]
                values, selector_seconds = scores[scene]
                costs = {**base_costs, "selector_seconds": selector_seconds}
                payload, decisions = selector_prediction(data["native"], head, suggestions, values, threshold, costs,
                    {"teacher_id": teacher, "checkpoint_sha256": sha256_file(checkpoint_path), "request_manifest": manifest["identity"]})
                target_dir = head_dir / "cal" / token / scene
                path = _store_prediction(payload, target_dir)
                # A resumed completed payload keeps its original measured cost.
                payload = load_prediction(path)
                decision_path = target_dir / "decisions.json"
                atomic_write_json(decision_path, {"decisions": decisions, "prediction_key": payload.prediction_key})
                evaluated = evaluate_predictions([payload], {scene: targets}, Path(runtime["upstream"]), target_dir / "evaluation")[0]
                scene_rows.append(evaluated)
                accepted = {row["owner_id"] for row in decisions if row["accepted"]}
                replacements += len(accepted)
                harmful += sum(row["event"] == "HARM" and row["owner_id"] in accepted for row in events_by_scene[scene])
                paths += [path, path.parent / "prediction.npz", decision_path, *_evaluation_outputs(target_dir / "evaluation")]
            threshold_rows.append({"threshold": threshold, "scene_rows": scene_rows, "harmful": harmful,
                "replacements": replacements, **{f"mean_{name}": None if any(row["metrics"][name] is None for row in scene_rows)
                    else float(np.mean([row["metrics"][name] for row in scene_rows])) for name in ("uap", "miou")}})
        undefined = any(row[name] is None for row in threshold_rows for name in ("mean_uap", "mean_miou"))
        threshold = "KEEP_ALL" if undefined else select_adoption_threshold(threshold_rows)
        status = "INCONCLUSIVE_UNDEFINED_CAL_METRIC" if undefined else trained["training_status"]
        calibration_path = head_dir / "calibration.json"
        atomic_write_json(calibration_path, {"head_id": head, "status": status, "threshold": threshold,
            "rows": threshold_rows, "checkpoint": file_identity(checkpoint_path)})
        paths.append(calibration_path)
        heads[head] = {"status": status, "threshold": threshold, "checkpoint": file_identity(checkpoint_path),
                       "calibration": file_identity(calibration_path)}
    return _receipt(output / "calibration_receipt.json", identity, inputs, paths,
        teacher_id=teacher, learned_status="COMPLETE" if all(row["status"] == "COMPLETE" for row in heads.values())
        else "UNCALIBRATED", heads=heads)


def run_semantic_select(runtime, config, config_path, *, run_models=True):
    """Execute the frozen learned rows without any fitting on SELECT."""
    import torch

    roles, lock_path = study_roles(runtime)
    output = Path(config["study_root"]) / "semantic"
    calibration_path = output / "calibration_receipt.json"
    calibration = _verified_receipt(calibration_path)
    inputs = _config_inputs(config, config_path) + [file_identity(path) for path in (lock_path, calibration_path)]
    identity = _identity(inputs, phase="SELECT")
    cached = _complete(output / "select_receipt.json", identity)
    if cached:
        return cached
    paths, all_rows, event_ledgers = [], [], {}
    for scene in roles["select"]:
        direct = prepare_semantic_scene(scene, "select", runtime, config, config_path, run_models=run_models)
        all_rows.extend(direct["rows"])
        data, targets = bind_scene(scene, runtime, config)
        paths.append(data["output"] / "semantic/direct_receipt.json")
        if not calibration["heads"]:
            continue
        features, suggestions, costs, manifest = _features_for_scene(data, config, calibration["teacher_id"])
        owners = list(features)
        locked, decision_ledgers = [], {}
        for head, settings in calibration["heads"].items():
            checkpoint = torch.load(settings["checkpoint"]["path"], map_location="cpu", weights_only=False)
            scaler = FeatureStandardizer(**checkpoint["scaler_state_dict"])
            matrix = np.asarray([scaler.transform(features[owner]) for owner in owners]).reshape(-1, 64)
            started = time.monotonic()
            values = dict(zip(owners, map(float, adoption_values(head, checkpoint["state_dict"], matrix)), strict=True)) if owners else {}
            model_costs = {**costs, "selector_seconds": time.monotonic() - started}
            payload, decisions = selector_prediction(data["native"], head, suggestions, values, settings["threshold"], model_costs,
                {"teacher_id": calibration["teacher_id"], "checkpoint_sha256": settings["checkpoint"]["sha256"],
                 "request_manifest": manifest["identity"], "calibration_status": settings["status"]})
            path = _store_prediction(payload, data["output"] / "semantic/learned" / head)
            locked.append(load_prediction(path))
            decision_path = path.parent / "decisions.json"
            atomic_write_json(decision_path, {"decisions": decisions, "features": {
                str(owner): {"values": features[owner].values.tolist(), "available": features[owner].available.tolist()}
                for owner in owners}, "prediction_key": payload.prediction_key})
            paths += [path, path.parent / "prediction.npz", decision_path]
            decision_ledgers[head] = decisions
        # No SELECT outcome has been read by a selector before all its payloads lock.
        evaluated = evaluate_predictions(locked, {scene: targets}, Path(runtime["upstream"]), data["output"] / "semantic/learned_evaluation")
        paths += _evaluation_outputs(data["output"] / "semantic/learned_evaluation")
        events = _json(data["output"] / "semantic/direct_evaluation/object_events.json")["events"][calibration["teacher_id"]]
        event_ledgers[scene] = {}
        for row, payload in zip(evaluated, locked, strict=True):
            method = row["method_id"]
            accepted = {decision["owner_id"] for decision in decision_ledgers[method] if decision["accepted"]}
            row.update(effective_changes=len(accepted), added_seconds=payload.logical_cost["added_seconds"],
                       logical_cost=dict(payload.logical_cost), calibration_status=calibration["heads"][method]["status"])
            event_ledgers[scene][method] = [{**event, "accepted": event["owner_id"] in accepted} for event in events]
        all_rows.extend(evaluated)
    event_path = output / "select_object_events.json"
    atomic_write_json(event_path, {"events": event_ledgers})
    paths.append(event_path)
    return _receipt(output / "select_receipt.json", identity, inputs, paths, rows=all_rows,
        teacher_id=calibration["teacher_id"], learned_status=calibration["learned_status"])
