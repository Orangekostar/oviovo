"""Run only frozen confirmation rows; outcomes never select another candidate."""

from __future__ import annotations

import fcntl
from dataclasses import asdict
from pathlib import Path

from .boundary_jobs import file_identity
from .combination_pipeline import run_combination_scene
from .confirmation_access import confirmation_contract, confirmation_rows
from .contracts import atomic_write_json, canonical_digest
from .frozen_semantic import prepare_frozen_semantic
from .geometry_pipeline import _quality_scores, prepare_geometry_prediction
from .geometry_study import load_geometry_pool, prepare_geometry_pool
from .query_pipeline import run_query_scene
from .scannet_frames import export_sensor
from .scannet_ground_truth import prepare_ground_truth
from .scannet_runtime import capture_development, require_idle_gpu
from .selection import bootstrap_mean_delta, select_final_candidate
from .selection_pipeline import _group
from .study_execution import (
    config_inputs,
    evaluation_outputs,
    read_json,
    receipt,
    reuse,
    roles,
    verify_receipt,
)
from .study_scene import audit_native_export, bind_scene, evaluate_predictions


def assess_confirmation(selection, rows, scenes):
    methods = selection["confirmation"]["rows"]
    expected = {(scene, method) for scene in scenes for method in methods}
    if {(row["scene_id"], row["method_id"]) for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("confirmation requires the exact preselected scenes and at-most-four methods")
    grouped = _group(rows, scenes, role="CONFIRM")
    method = selection["final_candidate"]
    baseline = "G_ORIGINAL" if method.startswith("G_") or method == "COMBO_GS" else "N0"
    if method == "Q_GAIN":
        baseline = selection["module_selection"]["query"]["locked_comparator"]
    elif method == "COMBO_Q_REFINEMENT":
        baseline = "COMBO_Q_REFINEMENT_CONTROL"
    if baseline not in grouped:
        raise ValueError("confirmation is missing the preselected candidate's matched baseline")
    decision = select_final_candidate({method: grouped[method]}, matched_baselines={method: grouped[baseline]}, simplicity_order=(method,))
    confirmed = decision.selected_method == method
    status = "CONFIRMED" if confirmed else "INCONCLUSIVE_UNDEFINED_CONFIRMATION_METRIC" \
        if decision.science_status.startswith("INCONCLUSIVE") else "NOT_CONFIRMED"
    bootstrap = None
    if confirmed:
        bootstrap = asdict(bootstrap_mean_delta([row.uap for row in grouped[method]], [row.uap for row in grouped[baseline]]))
        bootstrap["per_scene_delta_pp"] = {left.scene_id: 100 * (left.uap - right.uap)
            for left, right in zip(grouped[method], grouped[baseline], strict=True)}
        bootstrap["interpretation"] = "descriptive two-scene bootstrap, not population assurance"
    return {"status": status, "final_candidate": method, "baseline": baseline, "rows": rows, "bootstrap": bootstrap,
            "settings_changed_after_confirmation": False, "candidate_reselected": False}


def _sources():
    return [file_identity(Path(__file__).with_name(name)) for name in ("confirmation_pipeline.py", "confirmation_access.py",
        "frozen_semantic.py", "combination_pipeline.py", "geometry_pipeline.py", "query_pipeline.py", "selection.py")]


def run_confirmation(runtime, config, config_path, selection_receipt):
    import torch

    selection_evidence = verify_receipt(selection_receipt)
    selection = read_json(selection_evidence["selection_path"])
    split, lock_path = roles(runtime)
    root, output = Path(config["study_root"]), Path(config["study_root"]) / "confirmation"
    inputs = config_inputs(config, config_path) + [file_identity(path) for path in (selection_receipt, lock_path)]
    identity = canonical_digest({"inputs": inputs, "sources": _sources(), "phase": "CONFIRM"})
    cached = reuse(output / "receipt.json", identity)
    if cached:
        return verify_receipt(output / "receipt.json")
    if selection["final_candidate"] == "N0":
        result = {"status": "NOT_REQUIRED_NO_RETAINED_CANDIDATE", "final_candidate": "N0", "rows": []}
        path = output / "confirmation.json"
        atomic_write_json(path, result)
        return receipt(output / "receipt.json", identity, inputs, [path], _sources(),
            confirmation_path=str(path), confirmation_status=result["status"], rows=[])
    contract = confirmation_contract(selection_receipt, runtime, config)
    locked = read_json(lock_path)
    rows = confirmation_rows(locked, contract)
    data_root = Path(runtime["data_root"])
    paths = []
    # The immutable access contract is checked before either RGB-D export or GT conversion.
    for row in rows:
        scene = row["scene_id"]
        export_sensor(data_root / "scans" / scene / f"{scene}.sens", data_root / "exported" / scene, row["schedule"])
        annotation_path = prepare_ground_truth(Path(runtime["upstream"]), data_root / "scans" / scene, scene,
            data_root / "scannetv2-labels.combined.tsv", root / "annotations" / scene)
        paths.append(annotation_path)
    capture_development(runtime, scenes=list(split["confirm"]), confirmation_lock=selection_receipt)
    scene_rows = []
    for scene in split["confirm"]:
        scene_output = output / "scenes" / scene
        precomputed, payloads, details = {}, [], {}
        # Complete all Q causal readouts before this caller opens the final map.
        for method in selection["confirmation"]["rows"]:
            if method.startswith("Q_"):
                query_cal = verify_receipt(root / "query/calibration_receipt.json")
                lock = Path(runtime["output_root"]).parent / f".visual-gpu-{runtime['cuda_device']}.lock"
                with lock.open("a") as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                    require_idle_gpu(str(runtime["cuda_device"]), scene_output)
                    value = run_query_scene(scene, "confirm", method, runtime, config, config_path,
                        checkpoint_path=query_cal["checkpoint"]["path"] if method == "Q_GAIN" else None,
                        confirmation_lock=selection_receipt)
                precomputed[method] = value["row"]
                paths.append(root / "scenes" / scene / f"query/B200/{method}/receipt.json")
            elif method.startswith("COMBO_"):
                value = run_combination_scene(scene, "confirm", method, runtime, config, config_path,
                                              confirmation_lock=selection_receipt)
                precomputed[method] = value["row"]
                paths.append(scene_output / method / "receipt.json")
        data, targets = bind_scene(scene, runtime, config)
        parity = audit_native_export(data["native"], targets, Path(runtime["upstream"]), data["output"] / "native_export_parity")
        parity_path = scene_output / "native_export_parity.json"
        atomic_write_json(parity_path, parity)
        payloads.append(data["native"])
        paths += [parity_path, data["output"] / "baseline/receipt.json", Path(runtime["output_root"]) / scene / "mapping_job/receipt.json"]
        for method in selection["confirmation"]["rows"]:
            if method.startswith("S_"):
                payload, detail = prepare_frozen_semantic(data, method, runtime, config, config_path,
                    root / "semantic/calibration_receipt.json", scene_output / "semantic_evidence", scene_output / method)
                payloads.append(payload)
                details[method] = detail
                paths.append(scene_output / method / "receipt.json")
            elif method.startswith("G_"):
                pool_path = prepare_geometry_pool(data, data["output"] / "geometry/pool")
                pool = load_geometry_pool(pool_path)
                calibration = verify_receipt(root / "geometry/calibration_receipt.json")
                checkpoint_path, scores = None, None
                if method == "G_QUALITY":
                    checkpoint_path = calibration["checkpoint"]["path"]
                    scores = _quality_scores(pool, torch.load(checkpoint_path, map_location="cpu", weights_only=False))
                payload, detail = prepare_geometry_prediction(data, pool, method, runtime, config, config_path,
                    scene_output / method, scores=scores, margin=calibration["margin"], checkpoint_path=checkpoint_path)
                payloads.append(payload)
                details[method] = detail
                paths.append(scene_output / method / "receipt.json")
        evaluated = evaluate_predictions(payloads, {scene: targets}, Path(runtime["upstream"]), scene_output / "evaluation")
        for row, payload in zip(evaluated, payloads, strict=True):
            row.update(logical_cost=dict(payload.logical_cost), role="confirm",
                effective_changes=details.get(row["method_id"], {}).get("effective_changes", 0))
            precomputed[row["method_id"]] = row
        scene_rows.extend(precomputed[method] for method in selection["confirmation"]["rows"])
        paths += evaluation_outputs(scene_output / "evaluation")
    assessment = assess_confirmation(selection, scene_rows, split["confirm"])
    path = output / "confirmation.json"
    atomic_write_json(path, {**assessment, "selection_receipt": file_identity(selection_receipt),
                             "prediction_code_commit": selection["prediction_code_commit"]})
    return receipt(output / "receipt.json", identity, inputs, [path, *paths], _sources(),
        confirmation_path=str(path), confirmation_status=assessment["status"], rows=scene_rows)
