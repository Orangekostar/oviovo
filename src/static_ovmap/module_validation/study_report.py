"""Evidence-only ScanNet reporting; development roles and budgets never pool."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .boundary_jobs import file_identity
from .reporting import _table
from .study_execution import read_json, roles, verify_receipt

METRICS = ("uap", "ap50", "ap25", "miou", "macc", "canonical_ap50", "canonical_ap75")


def summarize_rows(rows):
    grouped, seen = defaultdict(list), set()
    for row in rows:
        key = row["role"], row["method_id"], row.get("budget")
        identity = (*key, row["scene_id"])
        if identity in seen:
            raise ValueError(f"duplicate scientific scene row: {identity}")
        seen.add(identity)
        grouped[key].append(row)
    result = []
    for (role, method, budget), values in sorted(grouped.items(), key=lambda item: str(item[0])):
        means, counts = {}, {}
        for metric in METRICS:
            defined = [row["metrics"][metric] for row in values if row["metrics"].get(metric) is not None]
            if any(not np.isfinite(value) or not 0 <= value <= 1 for value in defined):
                raise ValueError(f"invalid measured {metric}")
            means[metric], counts[metric] = (float(np.mean(defined)) if defined else None), len(defined)
        required = ("uap", "miou", "canonical_ap50", "canonical_ap75") if method.startswith("G_") else ("uap", "miou")
        costs = {key: sum(row.get("logical_cost", {}).get(key, 0) for row in values)
                 for key in sorted({key for row in values for key in row.get("logical_cost", {})})}
        result.append({"role": role, "method_id": method, "budget": budget, "scene_count": len(values),
            "metrics": means, "defined_scene_counts": counts, "logical_cost_sum": costs,
            "selection_status": "COMPLETE" if all(counts[key] == len(values) for key in required)
            else "INCONCLUSIVE_UNDEFINED_METRIC"})
    return result


def _semantic_summary(scene, role, method, events, evidence):
    usable = [row for row in events if row["technical_available"]]
    changed = [row for row in usable if not row["same_label"]]
    adopted = [row for row in changed if row.get("accepted", True)]
    return {"scene_id": scene, "role": role, "method_id": method, "suggestions": len(changed),
        "adoptions": len(adopted), "corrections": sum(row["event"] == "GAIN" for row in adopted),
        "damage": sum(row["event"] == "HARM" for row in adopted),
        "unmatched_or_ambiguous": sum(row["event"] is None for row in adopted), "evidence_path": str(evidence)}


def collect_study_evidence(runtime, config):
    root = Path(config["study_root"])
    split, _ = roles(runtime)
    role_by_scene = {scene: role for role, scenes in split.items() for scene in scenes}
    rows, inputs, verified = {}, {}, set()
    semantic_events, geometry_events, query_events, costs, method_status = [], [], [], {}, {}

    def load(path):
        path = Path(path)
        result = verify_receipt(path, verified)
        inputs[str(path)] = file_identity(path)
        return result

    def add(row, path, *, role=None, budget=None, cost=None):
        scene = row["scene_id"]
        bound = role_by_scene[scene]
        if role is not None and role != bound:
            raise ValueError("report receipt scene role differs from the frozen split")
        value = {**row, "role": bound, "budget": budget, "evidence_path": str(path)}
        if cost is not None:
            value["logical_cost"] = cost
        key = scene, row["method_id"], budget
        if key in rows:
            old = rows[key]
            if old["prediction_key"] != row["prediction_key"] or old["metrics"] != row["metrics"]:
                raise ValueError(f"inconsistent repeated scientific row: {key}")
            return
        rows[key] = value

    for role in ("fit", "cal", "select"):
        for scene in split[role]:
            base = root / "scenes" / scene
            baseline = base / "baseline_evaluation/metrics.json"
            if baseline.is_file():
                load(base / "baseline/receipt.json")
                inputs[str(baseline)] = file_identity(baseline)
                native_cost = read_json(base / "baseline/N0/manifest.json")["logical_cost"]
                for row in read_json(baseline)["rows"]:
                    add(row, baseline, role=role, cost=native_cost)
            for branch in ("semantic", "geometry"):
                path = base / branch / "direct_receipt.json"
                if path.is_file():
                    result = load(path)
                    for row in result["rows"]:
                        add(row, path, role=result["role"])
            events_path = base / "semantic/direct_evaluation/object_events.json"
            if (base / "semantic/direct_receipt.json").is_file():
                for method, events in read_json(events_path)["events"].items():
                    semantic_events.append(_semantic_summary(scene, role, method, events, events_path))
            for model in ("native", "siglip2", "wow"):
                path = base / "semantic_models" / model / "receipt.json"
                if path.is_file():
                    result = load(path)
                    costs[f"{scene}/S/{model}"] = {key: result[key] for key in (
                        "request_count", "successful_requests", "crop_inputs", "background_crop_inputs", "generations",
                        "model_load_seconds", "request_seconds", "elapsed_seconds_this_invocation")}
                    costs[f"{scene}/S/{model}"].update({key: value for key, value in result.items() if key.startswith("physical_")})
            for path in sorted((base / "query").glob("**/receipt.json")):
                result = load(path)
                if result.get("method_id") is None:
                    continue
                costs[f"{scene}/Q/{result['method_id']}/B{result['budget']}"] = {
                    "logical": result["logical_cost"], "physical": result["physical_cost"],
                    "end_to_end_seconds": result["end_to_end_seconds"]}
                decisions = read_json(path.parent / "decisions.json")
                events = read_json(path.parent / "events.json")
                capture_path = Path(runtime["output_root"]) / scene / "capture" / scene / "manifest.json"
                capture = read_json(capture_path)
                prediction_path = path.parent / "prediction/manifest.json"
                prediction = read_json(prediction_path) if prediction_path.is_file() else {}
                query_events.append({"scene_id": scene, "role": role, "method_id": result["method_id"],
                    "budget": result["budget"], **result["logical_cost"],
                    "identifiable_events": result["identifiable_events"],
                    "available_requests": sum(len(frame["requests"]) for frame in capture["frames"]),
                    "availability_definition": "technical native candidates before policy-specific ranking filters",
                    "ranked_requests": sum(len(frame["ranked_request_ids"]) for frame in decisions["frames"]),
                    "unobserved_owners": prediction.get("metadata", {}).get("masks_without_observations"),
                    "positive_targets": sum(row["gain_target"] is not None and row["gain_target"] > 1e-6
                                            for row in events["supervised_targets"]),
                    "physical": result["physical_cost"], "evidence_path": str(path)})
                if result["row"] is not None:
                    add(result["row"], path, role=result["role"], budget=result["budget"], cost=result["logical_cost"])
            capture_path = Path(runtime["output_root"]) / scene / "mapping_job/receipt.json"
            if capture_path.is_file():
                result = load(capture_path)
                costs[f"{scene}/native_capture"] = {key: result[key] for key in (
                    "elapsed_seconds", "scheduled_count", "completed_count", "invalid_pose_frame_ids")}
                front_path = capture_path.parents[1] / "frontend_job/receipt.json"
                front = load(front_path)
                costs[f"{scene}/CropFormer"] = {"elapsed_seconds": front["elapsed_seconds"]}

    for branch in ("semantic", "geometry", "query"):
        path = root / branch / "select_receipt.json"
        if not path.is_file():
            continue
        result = load(path)
        learned = ("S_SIMPLE", "S_NO_CONTEXT", "S_PAIRED") if branch == "semantic" else ("G_QUALITY",) if branch == "geometry" else ("Q_GAIN",)
        branch_rows = result.get("rows", [entry["row"] for entry in result.get("results", [])])
        for method in learned:
            if method not in {row["method_id"] for row in branch_rows}:
                method_status[method] = result["learned_status"]
        if branch != "query":
            for row in branch_rows:
                add(row, path, role="select")
        if branch == "semantic":
            event_path = root / "semantic/select_object_events.json"
            for scene, methods in read_json(event_path)["events"].items():
                for method, events in methods.items():
                    semantic_events.append(_semantic_summary(scene, "select", method, events, event_path))
    # Geometry diagnostic counts refer to whole partitions, never overlapping masks.
    for value in list(rows.values()):
        if not value["method_id"].startswith("G_"):
            continue
        base = root / "scenes" / value["scene_id"] / "geometry"
        method = value["method_id"]
        destination = base / ("learned" if method == "G_QUALITY" else "direct") / method
        details = load(destination / "receipt.json")
        decisions = read_json(destination / "prediction/decisions.json")
        pool = load(base / "pool/receipt.json")
        inference = load(destination / "inference/receipt.json")
        costs[f"{value['scene_id']}/G/{method}"] = {"logical": details["logical_cost"],
            "physical_attempts_this_invocation": inference["physical_attempts_this_invocation"],
            "model_load_seconds_this_invocation": inference["model_load_seconds_this_invocation"]}
        hypotheses = read_json(base / "pool/hypotheses.json")["hypotheses"]
        changed = sum(selected != hypotheses[group][0]["hypothesis_id"]
                      for group, selected in decisions["selected_hypotheses"].items())
        geometry_events.append({"scene_id": value["scene_id"], "role": value["role"], "method_id": method,
            "groups": pool["group_count"], "hypotheses": pool["hypothesis_count"], "changed_partitions": changed,
            "changed_points": details["changed_points"], "unknown_components": details["unknown_components"],
            "preserved_objects": None, "evidence_path": str(destination / "receipt.json")})
    for path in sorted((root / "selection/combinations").glob("*/receipt.json")):
        result = load(path)
        if result["disposition"] == "MEASURED":
            for row in [*result["rows"], *result["matched_rows"]]:
                add(row, path, role="select")
        else:
            method_status[result["method_id"]] = result["disposition"]
    confirm_path = root / "confirmation/receipt.json"
    if confirm_path.is_file():
        confirmation = load(confirm_path)
        for row in confirmation["rows"]:
            add(row, confirm_path, role="confirm", budget=200 if row["method_id"].startswith("Q_") else None)
        if confirmation["rows"]:
            for scene in split["confirm"]:
                mapping = load(Path(runtime["output_root"]) / scene / "mapping_job/receipt.json")
                frontend = load(Path(runtime["output_root"]) / scene / "frontend_job/receipt.json")
                costs[f"{scene}/native_capture"] = {key: mapping[key] for key in (
                    "elapsed_seconds", "scheduled_count", "completed_count", "invalid_pose_frame_ids")}
                costs[f"{scene}/CropFormer"] = {"elapsed_seconds": frontend["elapsed_seconds"]}
    for branch in ("semantic", "geometry", "query"):
        calibration_path = root / branch / "calibration_receipt.json"
        if calibration_path.is_file():
            calibration = load(calibration_path)
            costs[f"{branch}/calibration"] = {"learned_status": calibration["learned_status"], "evidence_path": str(calibration_path)}
            for path in sorted((root / branch).glob("**/training*.json")):
                values = read_json(path)
                costs[str(path.relative_to(root))] = {key: value for key, value in values.items()
                    if key not in {"inputs", "outputs", "sources", "state_dict", "optimizer_state_dict", "scaler_state_dict"}}
    unique_query = {}
    for scene in (*split["fit"], *split["cal"], *split["select"]):
        for path in sorted((root / "scenes" / scene / "query").glob("**/receipt.json")):
            for item in read_json(path).get("outputs", []):
                source = Path(item["path"])
                if "native_request_cache" in source.parts and source.suffix == ".json":
                    unique_query[str(source)] = read_json(source)
    costs["Q_unique_referenced_acquisitions"] = {"request_count": len(unique_query),
        "crop_inputs": sum(row["crop_inputs"] for row in unique_query.values()),
        "inference_seconds": sum(row["elapsed_seconds"] for row in unique_query.values()),
        "definition": "unique actual request-cache acquisitions used by active traces/policies, including cache origins"}
    values = sorted(rows.values(), key=lambda row: (row["role"], row["scene_id"], row["method_id"], row["budget"] or 0))
    return {"rows": values, "summaries": summarize_rows(values), "semantic_events": semantic_events,
        "geometry_events": geometry_events, "query_events": query_events, "costs": costs,
        "method_status": method_status, "inputs": list(inputs.values()),
        "scope": "active native-v9 artifacts only; historical runs excluded"}


def released_comparisons(evidence, config, comparator=None):
    """Read the unchanged evaluator's actual match/FN trace after predictions lock."""
    from src.static_ovmap.attribution_objects import released_object_outcomes

    study = Path(config["study_root"])
    root = study / "scenes"
    available = {(row["scene_id"], row["method_id"], row.get("budget")): row for row in evidence["rows"]}
    trace_index, traces, result = {}, {}, []
    for scene in {row["scene_id"] for row in evidence["rows"]}:
        scopes = [root / scene, study / "confirmation/scenes" / scene,
                  *(study / "selection/combinations").glob(f"*/scenes/{scene}")]
        for scope in scopes:
            for path in sorted(scope.rglob("released_trace.json.gz")):
                trace_index.setdefault((scene, path.parents[2].name), path)

    def load(row):
        key = row["scene_id"], row["prediction_key"]
        if key not in traces:
            path = trace_index.get(key)
            if path is None:
                raise ValueError(f"released object trace is missing for measured row: {key}")
            with gzip.open(path, "rt") as handle:
                trace = json.load(handle)
            with gzip.open(path.with_name("released_matches.json.gz"), "rt") as handle:
                matches = json.load(handle)
            eligible = {Path(pred["filename"]).stem: int(pred["label_id"])
                        for scan in matches.values() for values in scan["pred"].values() for pred in values}
            traces[key] = trace, eligible, file_identity(path)
        return traces[key]

    for row in evidence["rows"]:
        method, budget = row["method_id"], row.get("budget")
        if method == "N0":
            continue
        reference = comparator if method.startswith("Q_") and comparator is not None else "N0"
        if method.startswith("G_") and method != "G_ORIGINAL":
            reference = "G_ORIGINAL"
        if method == "COMBO_GS":
            reference = "G_ORIGINAL"
        elif method == "COMBO_Q_REFINEMENT":
            reference = "COMBO_Q_REFINEMENT_CONTROL"
        if method == reference:
            continue
        baseline = available.get((row["scene_id"], reference, budget if reference.startswith("Q_") else None))
        if baseline is None:
            continue
        candidate, cand_eligible, cand_path = load(row)
        native, base_eligible, base_path = load(baseline)
        thresholds = sorted({state["overlap_threshold"] for state in candidate["states"]})
        comparisons = []
        for threshold in thresholds:
            a = released_object_outcomes(candidate, threshold)
            b = released_object_outcomes(native, threshold)
            left, right = set(a["matched"]), set(b["matched"])
            comparisons.append({"threshold": threshold, "gained_gt_ids": sorted(left - right),
                "lost_gt_ids": sorted(right - left), "preserved_gt_ids": sorted(left & right),
                "candidate_hard_fn_gt_ids": a["hard_fn_gt_ids"], "baseline_hard_fn_gt_ids": b["hard_fn_gt_ids"]})
        result.append({"scene_id": row["scene_id"], "role": row["role"], "method_id": method, "budget": budget,
            "baseline": reference, "thresholds": comparisons,
            "eligibility_entries": sorted(set(cand_eligible) - set(base_eligible)),
            "eligibility_exits": sorted(set(base_eligible) - set(cand_eligible)),
            "eligible_label_changes": {owner: [base_eligible[owner], cand_eligible[owner]]
                for owner in sorted(set(base_eligible) & set(cand_eligible)) if base_eligible[owner] != cand_eligible[owner]},
            "owner_comparability": "geometry-specific component IDs" if method.startswith(("G_", "COMBO_")) else "fixed native masks",
            "candidate_trace": cand_path, "baseline_trace": base_path})
    return result


def _metric(value, count, total):
    prefix = "null" if value is None else f"{value!r} ({100 * value:.6g}%)"
    return f"{prefix}; {count}/{total}"


def render_scientific_results(evidence, selection, confirmation):
    a = []
    for row in evidence["rows"]:
        a.append([row["role"], row["scene_id"], row["method_id"], row.get("budget"),
            *[_metric(row["metrics"].get(metric), int(row["metrics"].get(metric) is not None), 1) for metric in METRICS],
            row.get("logical_cost", {}), row.get("evidence_path")])
    for row in evidence["summaries"]:
        a.append([row["role"], "MEAN", row["method_id"], row["budget"],
            *[_metric(row["metrics"][metric], row["defined_scene_counts"][metric], row["scene_count"]) for metric in METRICS],
            row["logical_cost_sum"], row["selection_status"]])
    for method, status in evidence["method_status"].items():
        a.append(["select", "NOT_MEASURED", method, None, *([None] * len(METRICS)), None, status])
    tables = [_table("## Table A. Per-scene metrics and logical cost",
        ("Role", "Scene", "Method", "Budget", "uAP ↑", "AP50 ↑", "AP25 ↑", "mIoU ↑", "mAcc ↑", "Canonical AP50 ↑",
         "Canonical AP75 ↑", "Logical cost (sum for MEAN)", "Evidence/status"), a)]
    specs = (("B. Semantic suggestion → adoption → correction/damage", "semantic_events",
              ("role", "scene_id", "method_id", "suggestions", "adoptions", "corrections", "damage", "unmatched_or_ambiguous", "evidence_path")),
             ("C. Complete geometry partitions and object preservation", "geometry_events",
              ("role", "scene_id", "method_id", "groups", "hypotheses", "changed_partitions", "changed_points", "preserved_objects", "unknown_components", "evidence_path")),
             ("D. Query availability, paid acquisition and readout", "query_events",
              ("role", "scene_id", "method_id", "budget", "available_requests", "attempts", "successes", "failures", "crop_inputs", "identifiable_events", "positive_targets", "evidence_path")))
    for title, key, names in specs:
        tables.append(_table("## Table " + title, names, [[row.get(name) for name in names] for row in evidence[key]]))
    choices = []
    for branch, value in selection.get("module_selection", {}).items():
        if branch in {"semantic", "geometry", "query"}:
            choices.append([branch, value["selected_method"], value.get("status"), value.get("mechanism_status", value.get("gain_status"))])
    choices.extend([row["method_id"], row["method_id"], row["status"], row.get("evidence_path")] for row in selection.get("combinations", []))
    choices += [["Final", selection.get("final_candidate", "UNSELECTED"), selection.get("science_status", "PENDING"), "selection.json"],
                ["Confirmation", selection.get("final_candidate", "UNSELECTED"), confirmation.get("status", "PENDING"), "confirmation.json"]]
    tables.append(_table("## Table E. Modules, combinations, final selection and confirmation",
                         ("Decision", "Candidate", "Status", "Evidence/mechanism"), choices))
    return ("# OVI-MAP Module Validation Results\n\n"
        "Active ScanNet captures only. Means are unweighted within each role and budget. Cells show the raw metric, "
        "percentage and defined/total scene count. Missing values remain null; required nulls make selection inconclusive. "
        "Class-agnostic canonical AP is separate from released semantic-instance AP. Logical costs are charged per method; "
        "physical reuse and training costs are recorded in scientific_evidence.json.\n\n" + "\n\n".join(tables) + "\n")
