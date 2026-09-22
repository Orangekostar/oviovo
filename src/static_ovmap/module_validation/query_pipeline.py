"""Frozen Q acquisition traces, scene-weighted fitting and held-out execution."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .boundary_jobs import file_identity
from .contracts import atomic_write_json, canonical_digest
from .metric_order import metric_max
from .native_capture import _write_npz
from .query_gain_policy import (
    QueryFeatureRow,
    QueryFeatureStandardizer,
    QueryTrainingExample,
    predict_query_gain,
    query_target_support_status,
    train_query_gain_head,
)
from .query_models import NativeQueryLoader
from .query_state import FeatureStore, export_query_labels
from .query_study import CapturedFrames, replay_captured
from .query_targets import CurrentTargetBuilder
from .scannet_ground_truth import load_ground_truth
from .scannet_study import relabel_prediction
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

CONTROLS = ("Q_COMBINE", "Q_AREA", "Q_UNCERTAINTY")
SOURCE_NAMES = ("query_pipeline.py", "query_models.py", "query_study.py", "query_lineage.py", "query_state.py",
                "query_gain_policy.py", "query_targets.py", "static_regions.py", "study_execution.py", "study_scene.py", "evaluation.py",
                "metric_order.py", "region_evidence.py", "scannet_ground_truth.py", "scannet_study.py", "confirmation_access.py")


def _sources():
    return [file_identity(path) for path in [*(Path(__file__).with_name(name) for name in SOURCE_NAMES),
        ROOT / "scripts/evaluation/run_ovimap_scannet_query.py",
        ROOT / "docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json"]]


def _identity(inputs, **settings):
    return canonical_digest({"inputs": inputs, "sources": _sources(), "settings": settings})


def reconcile_export(result, surface, text):
    """The final native registry is read ONLY after the last frame barrier."""
    pairs = np.unique(np.column_stack((surface["segment_labels"], surface["original_owner"])), axis=0)
    mapping = dict(result["lineage"].membership)
    seen = {}
    for segment, owner in pairs:
        segment, owner = int(segment), int(owner)
        if segment <= 0:
            continue
        if segment in seen and seen[segment] != owner:
            raise ValueError("BLOCKED_CAUSAL_LINEAGE: final segment has inconsistent numerical owners")
        seen[segment] = owner
    mapping.update(seen)
    snapshot = {"label_instances_scope": "all_known_labels", "aliases": [], "label_instances": [
        {"segment_label": segment, "instance_label": owner} for segment, owner in sorted(mapping.items())]}
    result["lineage"].advance(snapshot, result["state"], result["combine"], text)


def run_query_scene(scene, role, policy, runtime, config, config_path, *, budget=200, checkpoint_path=None,
                    confirmation_lock=None, confirmation_parent=None):
    import torch

    split, lock_path = roles(runtime)
    if role not in split or scene not in split[role]:
        raise ValueError("query execution requires an authorized development role")
    if role == "confirm":
        if budget != 200:
            raise ValueError("confirmation queries are restricted to B200")
        if confirmation_lock is None:
            raise ValueError("confirmation query requires a frozen selection lock")
        from .confirmation_access import require_confirmation_method

        contract = require_confirmation_method(scene, confirmation_parent or policy, runtime, config, confirmation_lock)
        if confirmation_parent is not None:
            module = contract["selection"]["module_selection"]["query"]
            expected = {"COMBO_Q_REFINEMENT": "Q_GAIN", "COMBO_Q_REFINEMENT_CONTROL": module["locked_comparator"]}
            if expected.get(confirmation_parent) != policy:
                raise ValueError("confirmation hybrid query constituent differs from its frozen parent")
        if policy == "Q_GAIN":
            calibration = read_json(contract["frozen"]["calibration"]["query"]["path"])
            if checkpoint_path is None or file_identity(checkpoint_path) != calibration["checkpoint"]:
                raise ValueError("confirmation query checkpoint differs from the frozen CAL head")
    random = policy == "Q_RANDOM_TRACE"
    if confirmation_parent is not None and role != "confirm":
        raise ValueError("query confirmation parent is restricted to the frozen holdout pipeline")
    if random and role not in {"fit", "cal"}:
        raise ValueError("query exploration is restricted to FIT/CAL")
    if random and budget != (512 if role == "fit" else 256):
        raise ValueError("query exploration budget differs from its frozen split")
    if policy not in {*CONTROLS, "Q_GAIN", "Q_RANDOM_TRACE"}:
        raise ValueError("unknown study query policy")
    leaf = "trace" if random else f"B{budget}/{policy}"
    if confirmation_parent is not None:
        leaf = f"constituent_B{budget}/{policy}"
    output = Path(config["study_root"]) / "scenes" / scene / "query" / leaf
    mapping_path = Path(runtime["output_root"]) / scene / "mapping_job/receipt.json"
    mapping = read_json(mapping_path)
    if mapping["status"] != "COMPLETE":
        raise ValueError("query requires a completed native capture")
    capture_path = Path(mapping["capture_manifest"])
    inputs = config_inputs(config, config_path) + [file_identity(path) for path in (
        lock_path, mapping_path, capture_path, config["native_text_cache"])]
    if role == "confirm":
        inputs.append(file_identity(confirmation_lock))
    seed = (17 if role == "fit" else 23) if random else None
    if checkpoint_path is not None:
        inputs.append(file_identity(checkpoint_path))
    if random:
        annotation_path = Path(config["study_root"]) / "annotations" / scene / "receipt.json"
        inputs.append(file_identity(annotation_path))
    frames = CapturedFrames(capture_path)
    loader = NativeQueryLoader(frames, config, Path(config["study_root"]) / "query/native_request_cache")
    inputs.extend(loader.inputs)
    identity = _identity(inputs, role=role, policy=policy, budget=budget, seed=seed, model_identity=loader.model_identity,
                         confirmation_parent=confirmation_parent)
    cached = reuse(output / "receipt.json", identity)
    if cached:
        return verify_receipt(output / "receipt.json")
    with np.load(config["native_text_cache"], allow_pickle=False) as arrays:
        text, class_ids = arrays["text_embeddings"], arrays["valid_ids"]
    store = FeatureStore(loader)
    predictor = scaler = None
    if policy == "Q_GAIN":
        if checkpoint_path is None:
            raise ValueError("Q_GAIN requires the frozen FIT/CAL checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["status"] != "COMPLETE":
            raise ValueError("Q_GAIN checkpoint lacks completed CAL assessment")
        scaler = QueryFeatureStandardizer(**checkpoint["scaler_state_dict"])
        predictor = lambda features: predict_query_gain(checkpoint["state_dict"], features)
    targets = CurrentTargetBuilder(load_ground_truth(annotation_path), class_ids) if random else None
    result = replay_captured(frames, policy, store, text, budget=budget, seed=seed,
                             predictor=predictor, standardizer=scaler, acquired_callback=targets)
    store.physical_ledger.model_loads = loader.model_loads
    logical, physical = asdict(result["state"].logical_ledger), asdict(store.physical_ledger)
    physical["model_load_seconds"] = loader.model_load_seconds
    row = None
    outputs = []
    if not random:
        # No final map/registry or evaluator input is opened before this point.
        data, evaluation_targets = bind_scene(scene, runtime, config)
        reconcile_export(result, data["surface"], text)
        owners = sorted(set(map(int, np.unique(data["native"].owner_ids))) - {0})
        labels = export_query_labels(result["state"], owners, text, valid_class_ids=class_ids)
        payload = relabel_prediction(data["native"], policy, "Q", labels,
            {**logical, "allowance": budget, "query_seconds": result["elapsed_seconds"]},
            {"schedule": "PER_FRAME_RESULT_BARRIER", "budget": budget, "unobserved_class": 0,
             "masks_without_observations": sum(value == 0 for value in labels.values()),
             "causal_capture": file_identity(capture_path), "final_registry_read_after_last_barrier": True})
        prediction_path = store_prediction(payload, output / "prediction")
        outputs += [prediction_path, prediction_path.parent / "prediction.npz"]
        if confirmation_parent is None:
            row = evaluate_predictions([payload], {scene: evaluation_targets}, Path(runtime["upstream"]), output / "evaluation")[0]
            row.update(logical_cost=logical, added_seconds=result["elapsed_seconds"],
                       unobserved_owners=sum(value == 0 for value in labels.values()))
            outputs += evaluation_outputs(output / "evaluation")
    decision_path, events_path, arrays_path = output / "decisions.json", output / "events.json", output / "prefix.npz"
    events = result["events"]
    atomic_write_json(decision_path, {"scene_id": scene, "policy": policy, "budget": budget,
        "planned_frame_count": len(frames.schedule), "frame_barrier": True, "frames": result["decisions"],
        "lineage": result["lineage"].frame_diagnostics,
        "discarded_ambiguous_features": sorted(result["lineage"].dropped_feature_ids),
        "retained_features": {str(owner): [feature.request_id for feature in obj.features]
                              for owner, obj in result["state"].objects.items()},
        "logical_cost": logical, "physical_cost": physical})
    atomic_write_json(events_path, {"events": [{key: value for key, value in event.items()
        if key not in {"features", "before_scores", "after_scores"}} for event in events],
        "supervised_targets": [] if targets is None else targets.events,
        "annotation_role": role if random else None})
    _write_npz(arrays_path, {"features": np.asarray([event["features"] for event in events], np.float32).reshape(-1, 40),
        **{name: np.asarray([np.zeros(len(text)) if event[name] is None else event[name] for event in events], np.float32).reshape(-1, len(text))
           for name in ("before_scores", "after_scores")},
        **{name + "_available": np.array([event[name] is not None for event in events], bool)
           for name in ("before_scores", "after_scores")}})
    outputs += [decision_path, events_path, arrays_path]
    for entry in loader.used_receipts.values():
        path = Path(entry["path"])
        outputs += [path, Path(read_json(path)["arrays_path"])]
    return receipt(output / "receipt.json", identity, inputs, outputs, _sources(), scene_id=scene, role=role,
        method_id=policy, budget=budget, seed=seed, row=row, logical_cost=logical, physical_cost=physical,
        identifiable_events=0 if targets is None else sum(event["gain_target"] is not None for event in targets.events),
        end_to_end_seconds=result["elapsed_seconds"], confirmation_parent=confirmation_parent,
        constituent_readout_only=confirmation_parent is not None)


def choose_comparator(rows, cal_scenes):
    expected = {(scene, method) for scene in cal_scenes for method in CONTROLS}
    actual = {(row["scene_id"], row["method_id"]) for row in rows}
    if (actual != expected or len(rows) != len(expected)
            or any(row["role"] != "cal" or row["budget"] != 200 for row in rows)):
        raise ValueError("query comparator requires exact CAL control rows")
    if any(row["row"]["metrics"].get(metric) is None for row in rows for metric in ("uap", "miou")):
        return None
    return metric_max(CONTROLS, key=lambda method: (
        float(np.mean([row["row"]["metrics"]["uap"] for row in rows if row["method_id"] == method])),
        float(np.mean([row["row"]["metrics"]["miou"] for row in rows if row["method_id"] == method])),
        -sum(row["logical_cost"]["attempts"] for row in rows if row["method_id"] == method), -CONTROLS.index(method)))


def calibrate_query(runtime, config, config_path):
    import torch

    split, lock_path = roles(runtime)
    root, output = Path(config["study_root"]) / "scenes", Path(config["study_root"]) / "query"
    trace_paths = [root / scene / "query/trace/receipt.json" for scene in (*split["fit"], *split["cal"])]
    control_paths = [root / scene / f"query/B200/{method}/receipt.json" for scene in split["cal"] for method in CONTROLS]
    controls = [verify_receipt(path) for path in control_paths]
    for path in trace_paths:
        verify_receipt(path)
    inputs = config_inputs(config, config_path) + [file_identity(path) for path in (lock_path, *trace_paths, *control_paths)]
    identity = _identity(inputs, phase="FIT_CAL")
    cached = reuse(output / "calibration_receipt.json", identity)
    if cached:
        return verify_receipt(output / "calibration_receipt.json")
    comparator = choose_comparator(controls, split["cal"])
    traces, fit_support = {"fit": [], "cal": []}, []
    for role, trace_rows in traces.items():
        for scene in split[role]:
            events = read_json(root / scene / "query/trace/events.json")
            with np.load(root / scene / "query/trace/prefix.npz", allow_pickle=False) as arrays:
                features = arrays["features"]
            if len(features) != len(events["supervised_targets"]):
                raise ValueError("query prefix features do not align with supervised target events")
            for vector, target, acquisition in zip(features, events["supervised_targets"], events["events"], strict=True):
                if target["request_id"] != acquisition["request_id"]:
                    raise ValueError("query label request identity differs from acquired event")
                if target["gain_target"] is not None:
                    trace_rows.append((scene, QueryFeatureRow(vector[:20], vector[20:].astype(bool)), target["gain_target"]))
                    if role == "fit":
                        fit_support.append({"scene_id": scene, "target": target["gain_target"]})
    support = query_target_support_status(fit_support)
    status = support
    checkpoint_path = None
    outputs = []
    if support == "SUPPORTED" and traces["cal"]:
        scaler = QueryFeatureStandardizer.fit([features for _, features, _ in traces["fit"]])
        examples = {role: [QueryTrainingExample(scene, scaler.transform(features), value)
            for scene, features, value in rows] for role, rows in traces.items()}
        started = time.monotonic()
        trained = train_query_gain_head(examples["fit"], cal_examples=examples["cal"])
        status = trained.status
        checkpoint_path = output / "checkpoint.pt"
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = {"state_dict": dict(trained.state_dict), "optimizer_state_dict": dict(trained.optimizer_state_dict),
            "scaler_state_dict": {name: getattr(scaler, name) for name in ("mean", "std", "fitted")},
            "status": status, "checkpoint_epoch": trained.checkpoint_epoch, "fit_epochs": trained.fit_epochs,
            "calibration_mse": trained.calibration_mse, "parameter_count": trained.parameter_count,
            "seed": 17, "torch_version": torch.__version__, "sources": _sources(), "input_identity": identity}
        torch.save(checkpoint, checkpoint_path)
        training_path = output / "training.json"
        atomic_write_json(training_path, {"status": status, "elapsed_seconds": time.monotonic() - started,
            "checkpoint_epoch": trained.checkpoint_epoch, "fit_epochs": trained.fit_epochs,
            "calibration_mse": trained.calibration_mse, "full_parameter_count": trained.parameter_count,
            "active_parameter_count": trained.parameter_count})
        outputs += [checkpoint_path, training_path]
    elif support == "SUPPORTED":
        status = "BLOCKED_QUERY_CAL_TARGET_SUPPORT"
    if comparator is None:
        status = "INCONCLUSIVE_UNDEFINED_CAL_METRIC"
    support_path = output / "target_support.json"
    atomic_write_json(support_path, {"status": support, "fit_scene_count": len({row["scene_id"] for row in fit_support}),
        "identifiable_fit_events": len(fit_support), "identifiable_cal_events": len(traces["cal"]),
        "positive_fit_events": sum(row["target"] > 1e-6 for row in fit_support),
        "nonpositive_fit_events": sum(row["target"] <= 1e-6 for row in fit_support)})
    outputs.append(support_path)
    return receipt(output / "calibration_receipt.json", identity, inputs, outputs, _sources(), learned_status=status,
        comparator_id=comparator, control_rows=controls, checkpoint=None if checkpoint_path is None else file_identity(checkpoint_path))


def run_query_select(runtime, config, config_path):
    split, lock_path = roles(runtime)
    output = Path(config["study_root"]) / "query"
    calibration_path = output / "calibration_receipt.json"
    calibration = verify_receipt(calibration_path)
    inputs = config_inputs(config, config_path) + [file_identity(path) for path in (lock_path, calibration_path)]
    identity = _identity(inputs, phase="SELECT", budget=200)
    cached = reuse(output / "select_receipt.json", identity)
    if cached:
        return verify_receipt(output / "select_receipt.json")
    methods = (*CONTROLS, "Q_GAIN") if calibration["learned_status"] == "COMPLETE" else CONTROLS
    checkpoint = calibration["checkpoint"]
    results, paths = [], []
    for scene in split["select"]:
        for method in methods:
            results.append(run_query_scene(scene, "select", method, runtime, config, config_path, budget=200,
                checkpoint_path=checkpoint["path"] if method == "Q_GAIN" else None))
            paths.append(Path(config["study_root"]) / "scenes" / scene / f"query/B200/{method}/receipt.json")
    return receipt(output / "select_receipt.json", identity, inputs, paths, _sources(),
        learned_status=calibration["learned_status"], comparator_id=calibration["comparator_id"], results=results)
