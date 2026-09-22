"""Finite ScanNet G experiment: common pools, supervised quality, fresh rereads."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from .assets import sha256_file
from .boundary_jobs import file_identity
from .contracts import atomic_write_json, canonical_digest
from .entity_hypotheses import apply_selected_partitions
from .evaluation import PredictionPayload, validate_prediction_invariants
from .geometry_models import final_mask_labels, load_static_records
from .geometry_study import (
    choose_geometry,
    load_geometry_pool,
    prepare_geometry_pool,
)
from .geometry_targets import prepare_geometry_targets
from .partition_quality import (
    GEOMETRY_MARGINS,
    PartitionFeatureStandardizer,
    PartitionTrainingExample,
    geometry_target_support_status,
    predict_partition_quality,
    select_geometry_margin,
    train_geometry_head,
)
from .scannet_study import load_prediction
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


def _sources():
    paths = [Path(__file__).with_name(name) for name in ("geometry_pipeline.py", "geometry_models.py", "geometry_study.py",
        "geometry_targets.py", "entity_hypotheses.py", "partition_quality.py", "static_regions.py",
        "study_execution.py", "study_scene.py", "evaluation.py")]
    paths += [ROOT / "scripts/evaluation" / name for name in ("run_ovimap_scannet_geometry.py", "run_ovimap_geometry_model.py")]
    paths.append(ROOT / "docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json")
    return [file_identity(path) for path in paths]


def _identity(inputs, **settings):
    return canonical_digest({"inputs": inputs, "sources": _sources(), "settings": settings})


def geometry_prediction(native, ownership, labels, costs, metadata):
    owners = np.asarray(ownership, np.int64)
    semantic = np.zeros(owners.shape, np.int64)
    ids, counts = np.unique(owners, return_counts=True)
    for owner in ids:
        if owner > 0:
            semantic[owners == owner] = labels[int(owner)]
    ranks = tuple((int(owner), float(count)) for owner, count in zip(ids, counts, strict=True) if owner > 0)
    payload = PredictionPayload(metadata["method_id"], "G", native.scene_id, native.geometry,
        owners, semantic, ranks, costs, {"native_record_key": native.record_key, **metadata})
    validate_prediction_invariants(payload, native)
    payload.lock()
    return payload


def prepare_geometry_prediction(data, pool, method, runtime, config, config_path, output,
                                *, scores=None, margin="KEEP_ALL", checkpoint_path=None, run_model=True):
    output = Path(output)
    pool_path = data["output"] / "geometry/pool/receipt.json"
    inputs = config_inputs(config, config_path) + [file_identity(pool_path), file_identity(data["native_manifest_path"])]
    if checkpoint_path is not None:
        inputs.append(file_identity(checkpoint_path))
    identity = _identity(inputs, method=method, margin=margin, scores=scores)
    cached = reuse(output / "receipt.json", identity)
    if cached:
        verify_receipt(output / "receipt.json")
        return load_prediction(output / "prediction/manifest.json"), cached
    started = time.monotonic()
    selected = choose_geometry(pool, method, scores=scores, margin=margin)
    ownership = apply_selected_partitions(data["scene_id"], data["native"].owner_ids, pool["leaves"], selected)
    selection_seconds = time.monotonic() - started
    manifest_path = prepare_static_manifest(data, ownership.owner_ids, ownership.ancestry, output / "requests")
    if run_model:
        subprocess.run([runtime["native_perception_python"], str(ROOT / "scripts/evaluation/run_ovimap_geometry_model.py"),
            "--config", str(config_path), "--request-manifest", str(manifest_path), "--output", str(output / "inference")], check=True)
    model_receipt_path = output / "inference/receipt.json"
    model_receipt = verify_receipt(model_receipt_path)
    records = load_static_records(output / "inference")
    manifest = read_json(manifest_path)
    with np.load(config["native_text_cache"], allow_pickle=False) as text:
        labels, semantics = final_mask_labels(data["native"], ownership.owner_ids, manifest, records,
            text["text_embeddings"], tuple(map(int, text["valid_ids"])))
    costs = {**dict(data["native"].logical_cost), "added_attempts": len(records), "added_crop_inputs": 6 * len(records),
        "added_seconds": sum(row["elapsed_seconds"] for row in records.values()), "selection_seconds": selection_seconds}
    payload = geometry_prediction(data["native"], ownership.owner_ids, labels, costs, {"method_id": method,
        "pool_identity": pool["receipt"]["input_identity"], "requests_identity": manifest["identity"], "margin": margin,
        "selected_hypotheses": dict(ownership.selected_hypotheses), "rank_rule": "source_component_point_count",
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None})
    prediction_path = store_prediction(payload, output / "prediction")
    payload = load_prediction(prediction_path)
    prediction_ledger = output / "prediction/decisions.json"
    atomic_write_json(prediction_ledger, {"selected_hypotheses": dict(ownership.selected_hypotheses),
        "ancestry": {str(owner): list(parents) for owner, parents in ownership.ancestry.items()},
        "semantics": semantics, "prediction_key": payload.prediction_key, "GT_input": False})
    paths = [prediction_path, prediction_path.parent / "prediction.npz", prediction_ledger,
             manifest_path.parent / "receipt.json", model_receipt_path]
    changed_points = int(np.count_nonzero(ownership.owner_ids != data["native"].owner_ids))
    result = receipt(output / "receipt.json", identity, inputs, paths, _sources(), method_id=method,
        scene_id=data["scene_id"], changed_points=changed_points, unknown_components=sum(label == 0 for label in labels.values()),
        prediction_manifest=str(prediction_path), logical_cost=dict(payload.logical_cost),
        physical_attempts_this_invocation=model_receipt["physical_attempts_this_invocation"])
    return payload, result


def prepare_geometry_scene(scene, role, runtime, config, config_path, *, run_models=True):
    split, lock_path = roles(runtime)
    if role not in {"fit", "cal", "select"} or scene not in split[role]:
        raise ValueError("G scene does not belong to its locked development role")
    if role == "select":
        verify_receipt(Path(config["study_root"]) / "geometry/calibration_receipt.json")
    data, targets = bind_scene(scene, runtime, config)
    output = data["output"] / "geometry"
    pool_path = prepare_geometry_pool(data, output / "pool")
    inputs = config_inputs(config, config_path) + [file_identity(lock_path), file_identity(pool_path)]
    identity = _identity(inputs, scene=scene, role=role)
    cached = reuse(output / "direct_receipt.json", identity)
    if cached:
        verify_receipt(output / "direct_receipt.json")
        return cached
    pool = load_geometry_pool(pool_path)
    paths, payloads, details = [], [data["native"]], {}
    for method in ("G_ORIGINAL", "G_AGREEMENT"):
        payload, result = prepare_geometry_prediction(data, pool, method, runtime, config, config_path,
            output / "direct" / method, run_model=run_models)
        payloads.append(payload)
        details[method] = result
        paths.append(output / "direct" / method / "receipt.json")
    rows = evaluate_predictions(payloads, {scene: targets}, Path(runtime["upstream"]), output / "direct_evaluation")
    paths += evaluation_outputs(output / "direct_evaluation")
    # Supervised targets/oracle diagnostics are built after the common pool and
    # both predicted direct rows are frozen; never passed to their selectors.
    prepare_geometry_targets(pool, pool_path, targets, output / "targets")
    paths.append(output / "targets/receipt.json")
    for row, payload in zip(rows, payloads, strict=True):
        detail = details.get(row["method_id"], {})
        row.update(logical_cost=dict(payload.logical_cost), changed_points=detail.get("changed_points", 0),
            effective_changes=detail.get("changed_points", 0), unknown_components=detail.get("unknown_components", 0),
            added_seconds=payload.logical_cost.get("added_seconds", 0.))
    return receipt(output / "direct_receipt.json", identity, inputs, paths, _sources(), scene_id=scene, role=role, rows=rows)


def _quality_scores(pool, checkpoint):
    scaler = PartitionFeatureStandardizer(**checkpoint["scaler_state_dict"])
    ids = list(pool["features"])
    if not ids:
        return {}
    features = np.stack([scaler.transform(pool["features"][key]) for key in ids])
    scores = predict_partition_quality(checkpoint["state_dict"], features)
    return dict(zip(ids, map(float, scores), strict=True))


def calibrate_geometry(runtime, config, config_path, *, run_models=True):
    import torch

    split, lock_path = roles(runtime)
    output, scene_root = Path(config["study_root"]) / "geometry", Path(config["study_root"]) / "scenes"
    direct_paths = [scene_root / scene / "geometry/direct_receipt.json" for scene in (*split["fit"], *split["cal"])]
    for path in direct_paths:
        verify_receipt(path)
    inputs = config_inputs(config, config_path) + [file_identity(path) for path in (lock_path, *direct_paths)]
    identity = _identity(inputs, phase="FIT_CAL")
    cached = reuse(output / "calibration_receipt.json", identity)
    if cached:
        return cached
    rows_by_role, support_rows = {"fit": [], "cal": []}, []
    for role in ("fit", "cal"):
        for scene in split[role]:
            pool = load_geometry_pool(scene_root / scene / "geometry/pool/receipt.json")
            target = read_json(scene_root / scene / "geometry/targets/targets.json")
            for group, rows in target["targets"].items():
                if role == "fit":
                    support_rows.append({"scene_id": scene, "group_id": group, "targets": [row["quality"] for row in rows]})
                for row in rows:
                    rows_by_role[role].append((scene, group, pool["features"][row["hypothesis_id"]], row["quality"]))
            del pool
    support = geometry_target_support_status(support_rows)
    support_path = output / "target_support.json"
    atomic_write_json(support_path, {"status": support, "groups": support_rows,
        "differing_group_count": sum(len(set(row["targets"])) > 1 for row in support_rows)})
    paths = [support_path]
    if support != "SUPPORTED":
        return receipt(output / "calibration_receipt.json", identity, inputs, paths, _sources(),
                        learned_status=support, checkpoint=None, margin=None)
    scaler = PartitionFeatureStandardizer.fit([row[2] for row in rows_by_role["fit"]])
    examples = {role: [PartitionTrainingExample(scene, group, scaler.transform(features), target)
        for scene, group, features, target in rows] for role, rows in rows_by_role.items()}
    checkpoint_path, training_path = output / "checkpoint.pt", output / "training_receipt.json"
    training_identity = _identity(inputs, phase="QUALITY_FIT")
    training = reuse(training_path, training_identity)
    if training:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    else:
        started = time.monotonic()
        result = train_geometry_head(examples["fit"], cal_examples=examples["cal"])
        checkpoint = {"state_dict": dict(result.state_dict), "optimizer_state_dict": dict(result.optimizer_state_dict),
            "scaler_state_dict": {name: getattr(scaler, name) for name in ("mean", "std", "fitted")},
            "status": result.status, "checkpoint_epoch": result.checkpoint_epoch, "fit_epochs": result.fit_epochs,
            "calibration_loss": result.calibration_loss, "parameter_count": result.parameter_count,
            "sources": _sources(), "input_identity": training_identity, "torch_version": torch.__version__, "seed": 17}
        output.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        training = receipt(training_path, training_identity, inputs, [checkpoint_path], _sources(),
            training_status=result.status, training_seconds=time.monotonic() - started,
            checkpoint_epoch=result.checkpoint_epoch, calibration_loss=result.calibration_loss,
            full_parameter_count=result.parameter_count, active_parameter_count=result.parameter_count)
    paths += [checkpoint_path, training_path]
    margin_rows = {margin: [] for margin in GEOMETRY_MARGINS}
    changed = {margin: 0 for margin in GEOMETRY_MARGINS}
    for scene in split["cal"]:
        data, targets = bind_scene(scene, runtime, config)
        pool = load_geometry_pool(data["output"] / "geometry/pool/receipt.json")
        scores = _quality_scores(pool, checkpoint)
        for margin in GEOMETRY_MARGINS:
            destination = output / "cal" / str(margin) / scene
            payload, detail = prepare_geometry_prediction(data, pool, "G_QUALITY", runtime, config, config_path,
                destination, scores=scores, margin=margin, checkpoint_path=checkpoint_path, run_model=run_models)
            row = evaluate_predictions([payload], {scene: targets}, Path(runtime["upstream"]), destination / "evaluation")[0]
            margin_rows[margin].append(row)
            changed[margin] += detail["changed_points"]
            paths += [destination / "receipt.json", *evaluation_outputs(destination / "evaluation")]
        del data, targets, pool
    thresholds = [{"margin": margin, "scene_rows": rows, "changed_points": changed[margin],
        **{f"mean_{name}": None if any(row["metrics"]["canonical_" + name] is None for row in rows)
            else float(np.mean([row["metrics"]["canonical_" + name] for row in rows])) for name in ("ap50", "ap75")}}
        for margin, rows in margin_rows.items()]
    undefined = any(row[name] is None for row in thresholds for name in ("mean_ap50", "mean_ap75"))
    margin = "KEEP_ALL" if undefined else select_geometry_margin(thresholds)
    calibration_path = output / "margin_selection.json"
    status = "INCONCLUSIVE_UNDEFINED_CAL_METRIC" if undefined else training["training_status"]
    atomic_write_json(calibration_path, {"margin": margin, "rows": thresholds, "status": status, "CAL_scenes": split["cal"]})
    paths.append(calibration_path)
    return receipt(output / "calibration_receipt.json", identity, inputs, paths, _sources(),
        learned_status=status, checkpoint=file_identity(checkpoint_path), margin=margin)


def run_geometry_select(runtime, config, config_path, *, run_models=True):
    import torch

    split, lock_path = roles(runtime)
    output = Path(config["study_root"]) / "geometry"
    calibration_path = output / "calibration_receipt.json"
    calibration = verify_receipt(calibration_path)
    inputs = config_inputs(config, config_path) + [file_identity(path) for path in (lock_path, calibration_path)]
    identity = _identity(inputs, phase="SELECT")
    cached = reuse(output / "select_receipt.json", identity)
    if cached:
        return cached
    paths, rows = [], []
    for scene in split["select"]:
        direct = prepare_geometry_scene(scene, "select", runtime, config, config_path, run_models=run_models)
        rows.extend(direct["rows"])
        scene_root = Path(config["study_root"]) / "scenes" / scene
        paths.append(scene_root / "geometry/direct_receipt.json")
        if calibration["checkpoint"] is None:
            continue
        data, targets = bind_scene(scene, runtime, config)
        pool = load_geometry_pool(scene_root / "geometry/pool/receipt.json")
        checkpoint_path = Path(calibration["checkpoint"]["path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        scores = _quality_scores(pool, checkpoint)
        destination = scene_root / "geometry/learned/G_QUALITY"
        payload, detail = prepare_geometry_prediction(data, pool, "G_QUALITY", runtime, config, config_path, destination,
            scores=scores, margin=calibration["margin"], checkpoint_path=checkpoint_path, run_model=run_models)
        row = evaluate_predictions([payload], {scene: targets}, Path(runtime["upstream"]), destination / "evaluation")[0]
        row.update(logical_cost=dict(payload.logical_cost), effective_changes=detail["changed_points"],
            changed_points=detail["changed_points"], unknown_components=detail["unknown_components"],
            added_seconds=payload.logical_cost.get("added_seconds", 0.), calibration_status=calibration["learned_status"])
        rows.append(row)
        paths += [destination / "receipt.json", *evaluation_outputs(destination / "evaluation")]
    return receipt(output / "select_receipt.json", identity, inputs, paths, _sources(),
                   rows=rows, learned_status=calibration["learned_status"], margin=calibration["margin"])
