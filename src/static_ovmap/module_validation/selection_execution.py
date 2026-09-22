"""Conditional SELECT execution and immutable pre-confirmation selection lock."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path

from .boundary_jobs import file_identity
from .contracts import atomic_write_json, canonical_digest
from .query_pipeline import run_query_scene
from .selection import bootstrap_mean_delta, plan_confirmation, select_final_candidate
from .selection_pipeline import _group, standalone_selection
from .study_execution import (
    ROOT,
    config_inputs,
    read_json,
    receipt,
    reuse,
    roles,
    verify_receipt,
)


def _sources():
    return [file_identity(Path(__file__).with_name(name)) for name in
        ("selection_execution.py", "selection_pipeline.py", "selection.py", "metric_order.py")]


def _identity(inputs, phase):
    return canonical_digest({"inputs": inputs, "sources": _sources(), "phase": phase})


def prepare_selection(runtime, config, config_path):
    split, lock_path = roles(runtime)
    root = Path(config["study_root"])
    paths = [root / name for name in ("semantic/select_receipt.json", "geometry/select_receipt.json",
        "query/calibration_receipt.json", "query/select_receipt.json")]
    seen = set()
    results = [verify_receipt(path, seen) for path in paths]
    inputs = config_inputs(config, config_path) + [file_identity(path) for path in (lock_path, *paths)]
    identity = _identity(inputs, "STANDALONE_SELECT")
    output = root / "selection"
    cached = reuse(output / "module_receipt.json", identity)
    if cached:
        return verify_receipt(output / "module_receipt.json")
    decision = standalone_selection(*results, select_scenes=split["select"], cal_scenes=split["cal"])
    path = output / "module_decision.json"
    atomic_write_json(path, decision)
    return receipt(output / "module_receipt.json", identity, inputs, [path], _sources(), decision=decision)


def run_budget_curves(runtime, config, config_path):
    root = Path(config["study_root"])
    module_path = root / "selection/module_receipt.json"
    decision = verify_receipt(module_path)["decision"]
    calibration_path = root / "query/calibration_receipt.json"
    calibration = verify_receipt(calibration_path)
    split, lock_path = roles(runtime)
    inputs = config_inputs(config, config_path) + [file_identity(path) for path in
        (module_path, calibration_path, lock_path)]
    identity = _identity(inputs, "CONDITIONAL_QUERY_CURVES")
    path = root / "selection/budget_curve_receipt.json"
    cached = reuse(path, identity)
    if cached:
        return verify_receipt(path)
    required = decision["budget_curves"]["status"] == "REQUIRED"
    if required and (decision["query"]["gain_status"] != "ELIGIBLE_ACCURACY"
                     or calibration["learned_status"] != "COMPLETE"):
        raise ValueError("budget curves require independently eligible frozen Q_GAIN")
    rows, outputs = [], []
    if required:
        for scene in split["select"]:
            for budget in (100, 400):
                for method in (calibration["comparator_id"], "Q_GAIN"):
                    result = run_query_scene(scene, "select", method, runtime, config, config_path, budget=budget,
                        checkpoint_path=calibration["checkpoint"]["path"] if method == "Q_GAIN" else None)
                    rows.append(result)
                    outputs.append(root / "scenes" / scene / f"query/B{budget}/{method}/receipt.json")
    return receipt(path, identity, inputs, outputs, _sources(), rows=rows,
        gate_status="MEASURED" if required else decision["budget_curves"]["status"])


def _validate_curves(decision, curves):
    if decision["budget_curves"]["status"] != "REQUIRED":
        if curves and curves.get("rows"):
            raise ValueError("query budget curves ran without the frozen eligibility gate")
        return
    if curves is None or curves.get("gate_status") != "MEASURED":
        raise ValueError("required query budget curves are pending")
    expected = {(scene, method, budget) for scene in decision["SELECT_scenes"]
        for method in (decision["budget_curves"]["comparator_id"], "Q_GAIN") for budget in (100, 400)}
    actual = {(row["scene_id"], row["method_id"], row["budget"]) for row in curves["rows"]}
    if actual != expected or len(curves["rows"]) != len(expected):
        raise ValueError("query budget curves do not match frozen SELECT obligations")


def _confirmation(candidate, decision):
    teacher = decision["semantic"]["frozen_teacher_id"]
    comparison = {"S_PAIRED": "S_SIMPLE", "S_NO_CONTEXT": "S_SIMPLE", "S_SIMPLE": teacher,
        "S_SIGLIP2_AREA": "S_NATIVE_AREA", "S_SIGLIP2_VOTE": "S_NATIVE_VOTE", "S_WOW_VOTE": "S_NATIVE_VOTE",
        "G_QUALITY": "G_AGREEMENT", "G_AGREEMENT": "G_ORIGINAL",
        "Q_GAIN": decision["query"]["locked_comparator"],
        "COMBO_GS": "G_ORIGINAL", "COMBO_Q_REFINEMENT": "COMBO_Q_REFINEMENT_CONTROL"}.get(candidate)
    plan = asdict(plan_confirmation(candidate, nearest_comparison=comparison))
    if candidate == "COMBO_Q_REFINEMENT":
        plan["rows"] = tuple(dict.fromkeys((*plan["rows"], decision["query"]["locked_comparator"])))
    return plan


def finalize_decision(decision, candidate_rows, matched_rows, *, combinations, curves):
    """Pure final hierarchy. Measured negatives remain visible; no pending work is waived."""
    _validate_curves(decision, curves)
    expected = set(decision["combination_plan"]["required"])
    provided = [row["method_id"] for row in combinations]
    if len(set(provided)) != len(provided) or set(provided) != expected:
        raise ValueError(f"pending or extraneous required combinations: {sorted(expected ^ set(provided))}")
    candidates, baselines = dict(candidate_rows), dict(matched_rows)
    for combination in combinations:
        method = combination["method_id"]
        status = combination["status"]
        if status == "MEASURED":
            if not combination.get("evidence_path"):
                raise ValueError("measured combination requires prediction provenance")
            candidates[method] = combination["rows"]
            baselines[method] = combination["matched_rows"]
        elif status == "BLOCKED_FRESH_MASK_PROVENANCE":
            if not combination.get("evidence_path") or not combination.get("reason"):
                raise ValueError("blocked combination requires an evidenced actual provenance limitation")
        elif status == "NOT_REQUIRED_STATIC_NONINFERIORITY_FAILED" and method == "COMBO_Q_REFINEMENT":
            gs = next((row for row in combinations if row["method_id"] == "COMBO_GS"), None)
            if gs is None or gs["status"] != "MEASURED":
                raise ValueError("hybrid gate requires the measured GS comparison")
            g = _group(gs["rows"], decision["SELECT_scenes"], role="SELECT")["COMBO_GS"]
            b = tuple(next(iter(_group(gs["matched_rows"], decision["SELECT_scenes"], role="SELECT").values())))
            if any(row.uap is None or row.miou is None for row in (*g, *b)):
                raise ValueError("undefined GS comparison cannot be asserted as noninferiority failure")
            if (sum(row.uap for row in g) >= sum(row.uap for row in b) - 2e-10
                    and sum(row.miou for row in g) >= sum(row.miou for row in b) - 2e-10):
                raise ValueError("GS passes noninferiority; required hybrid cannot be skipped")
        else:
            raise ValueError(f"unsupported combination disposition: {status}")
    c = {method: _group(rows, decision["SELECT_scenes"], role="SELECT")[method]
         for method, rows in candidates.items()}
    b = {}
    for method, rows in baselines.items():
        groups = _group(rows, decision["SELECT_scenes"], role="SELECT")
        if len(groups) != 1:
            raise ValueError("one matched baseline is required for each candidate")
        b[method] = next(iter(groups.values()))
    selected = select_final_candidate(c, matched_baselines=b, simplicity_order=tuple(c))
    final = asdict(selected)
    if not c and decision["independent_final"]["science_status"].startswith("INCONCLUSIVE"):
        final["science_status"] = decision["independent_final"]["science_status"]
    bootstrap = {}
    for method in selected.eligible_methods:
        bootstrap[method] = {**asdict(bootstrap_mean_delta([row.uap for row in c[method]], [row.uap for row in b[method]])),
            "per_scene_delta": {row.scene_id: row.uap - base.uap for row, base in zip(c[method], b[method], strict=True)},
            "interpretation": "descriptive two-scene interval, not population-level assurance"}
    confirmation = _confirmation(selected.selected_method, decision)
    return {"status": "FROZEN", "final_candidate": selected.selected_method, **final,
        "module_selection": decision, "combinations": combinations, "budget_curves": curves,
        "confirmation": confirmation, "confirmation_authorized": selected.selected_method != "N0", "bootstrap": bootstrap}


def _independent_rows(decision, semantic, geometry, query):
    candidates, matched = {}, {}
    for branch, source, baseline in (("semantic", semantic["rows"], "N0"),
                                     ("geometry", geometry["rows"], "G_ORIGINAL")):
        method = decision[branch]["selected_method"]
        if method != baseline:
            candidates[method] = [row for row in source if row["method_id"] == method]
            matched[method] = [row for row in source if row["method_id"] == baseline]
    if decision["query"]["selected_method"] == "Q_GAIN":
        source = [{**result["row"], "logical_cost": result["logical_cost"], "added_seconds": result["end_to_end_seconds"]}
                  for result in query["results"]]
        candidates["Q_GAIN"] = [row for row in source if row["method_id"] == "Q_GAIN"]
        matched["Q_GAIN"] = [row for row in source if row["method_id"] == decision["query"]["locked_comparator"]]
    return candidates, matched


def freeze_selection(runtime, config, config_path):
    """Bind all selection evidence and the exact committed code before any holdout access."""
    root = Path(config["study_root"])
    module_path = root / "selection/module_receipt.json"
    decision = verify_receipt(module_path)["decision"]
    paths = [root / name for name in ("semantic/select_receipt.json", "geometry/select_receipt.json", "query/select_receipt.json")]
    seen = set()
    semantic, geometry, query = [verify_receipt(path, seen) for path in paths]
    curves_path = root / "selection/budget_curve_receipt.json"
    curves = verify_receipt(curves_path)
    combinations = []
    for method in decision["combination_plan"]["required"]:
        path = root / "selection/combinations" / method / "receipt.json"
        combinations.append(verify_receipt(path))
        paths.append(path)
    candidates, matched = _independent_rows(decision, semantic, geometry, query)
    selection = finalize_decision(decision, candidates, matched, combinations=combinations, curves=curves)
    _, lock_path = roles(runtime)
    inputs = config_inputs(config, config_path) + [file_identity(path) for path in
        (module_path, curves_path, lock_path, *paths)]
    identity = _identity(inputs, "FINAL_SELECT")
    output = root / "selection"
    cached = reuse(output / "receipt.json", identity)
    if cached:
        return verify_receipt(output / "receipt.json")
    code_paths = ("src/static_ovmap/module_validation", "scripts/evaluation", "configs/evaluation",
                  "third_party_patches/ovimap/module_validation_v1")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *code_paths], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError("prediction/configuration code must be committed before freezing confirmation")
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_paths = [path for name in code_paths for path in (ROOT / name).rglob("*")
                    if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".patch"}]
    frozen_path = output / "frozen_config.json"
    atomic_write_json(frozen_path, {"study": config, "runtime": runtime, "split": read_json(lock_path),
        "prediction_code_commit": code, "sources": [file_identity(path) for path in sorted(source_paths)],
        "calibration": {branch: file_identity(root / branch / "calibration_receipt.json")
                        for branch in ("semantic", "geometry", "query")}, "evidence_inputs": inputs})
    selection.update(prediction_code_commit=code, frozen_config=file_identity(frozen_path),
                     confirmation_status=selection["confirmation"]["status"])
    selection_path = output / "selection.json"
    atomic_write_json(selection_path, selection)
    return receipt(output / "receipt.json", identity, inputs, [selection_path, frozen_path], _sources(),
                   selection_path=str(selection_path), final_candidate=selection["final_candidate"],
                   science_status=selection["science_status"], confirmation_status=selection["confirmation_status"])
