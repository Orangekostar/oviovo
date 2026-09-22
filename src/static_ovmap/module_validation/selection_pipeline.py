"""Receipt-backed module decisions, without inventing absent learned results."""

from __future__ import annotations

from dataclasses import asdict

from .query_pipeline import CONTROLS, choose_comparator
from .selection import (
    SceneMethodMetrics,
    SemanticSelection,
    bootstrap_mean_delta,
    plan_combinations,
    select_final_candidate,
    select_geometry_module,
    select_query_module,
    select_semantic_module,
)


def _group(rows, expected_scenes, *, role):
    groups = {}
    for row in rows:
        method, scene = row["method_id"], row["scene_id"]
        metrics, cost = row["metrics"], row.get("logical_cost", {})
        groups.setdefault(method, []).append(SceneMethodMetrics(method, scene, metrics.get("uap"), metrics.get("miou"),
            metrics.get("canonical_ap50"), metrics.get("canonical_ap75"), int(cost.get("attempts", 0)),
            float(row.get("added_seconds", cost.get("query_seconds", 0.))), int(row.get("effective_changes", 0))))
    for method, values in groups.items():
        if len(values) != len(expected_scenes) or {row.scene_id for row in values} != set(expected_scenes):
            raise ValueError(f"{method} requires the exact frozen {role} scene set")
        groups[method] = tuple(sorted(values, key=lambda row: row.scene_id))
    return groups


def _absent_heads(groups, heads, status):
    missing = {method: status for method in heads if method not in groups}
    if missing and status == "COMPLETE":
        raise ValueError("complete learned calibration has missing SELECT outputs")
    return missing


def standalone_selection(semantic, geometry, query_cal, query, *, select_scenes, cal_scenes):
    """Select modules on SELECT and expose conditional work before final freeze.

    A better non-frozen teacher in the full direct table is descriptive only.
    Required combinations and curves remain explicit executable obligations;
    this result is not the final confirmation authorization.
    """
    semantic_rows = _group(semantic["rows"], select_scenes, role="SELECT")
    geometry_rows = _group(geometry["rows"], select_scenes, role="SELECT")
    required_s = {"N0", "S_NATIVE_AREA", "S_NATIVE_VOTE", "S_SIGLIP2_AREA", "S_SIGLIP2_VOTE", "S_WOW_VOTE"}
    if not required_s <= set(semantic_rows) or not {"N0", "G_ORIGINAL", "G_AGREEMENT"} <= set(geometry_rows):
        raise ValueError("required SELECT direct controls are missing")
    s_native = {row["scene_id"]: row for row in semantic["rows"] if row["method_id"] == "N0"}
    g_native = {row["scene_id"]: row for row in geometry["rows"] if row["method_id"] == "N0"}
    for scene in select_scenes:
        if (s_native[scene]["prediction_key"] != g_native[scene]["prediction_key"]
                or s_native[scene]["metrics"] != g_native[scene]["metrics"]):
            raise ValueError("S and G N0 bridges differ on a shared SELECT scene")
    unavailable = _absent_heads(semantic_rows, ("S_SIMPLE", "S_NO_CONTEXT", "S_PAIRED"), semantic["learned_status"])
    if semantic["teacher_id"] is None:
        s_selection = SemanticSelection("N0", None, False, semantic["learned_status"],
            semantic["learned_status"] if semantic["learned_status"].startswith("INCONCLUSIVE")
            else "N0_FALLBACK_NO_AVAILABLE_TEACHER")
    else:
        s_selection = select_semantic_module(semantic_rows, frozen_teacher_id=semantic["teacher_id"], unavailable_methods=unavailable)
    g_unavailable = _absent_heads(geometry_rows, ("G_QUALITY",), geometry["learned_status"])
    unavailable.update(g_unavailable)
    g_selection = select_geometry_module(geometry_rows, unavailable_methods=g_unavailable)
    q_rows = [{**result["row"], "logical_cost": result["logical_cost"],
               "added_seconds": result.get("end_to_end_seconds", 0.)} for result in query["results"]]
    if any(result["budget"] != 200 for result in query["results"]):
        raise ValueError("standalone SELECT selection may only use B200 rows")
    query_rows = _group(q_rows, select_scenes, role="SELECT")
    if not set(CONTROLS) <= set(query_rows):
        raise ValueError("required SELECT query controls are missing")
    comparator = choose_comparator(query_cal["control_rows"], cal_scenes)
    if comparator != query_cal["comparator_id"] or comparator != query["comparator_id"]:
        raise ValueError("query comparator differs from its frozen CAL decision")
    cal_rows = _group([{**result["row"], "logical_cost": result["logical_cost"]}
                       for result in query_cal["control_rows"]], cal_scenes, role="CAL")
    q_unavailable = _absent_heads(query_rows, ("Q_GAIN",), query["learned_status"])
    unavailable.update(q_unavailable)
    q_selection = select_query_module(cal_rows, query_rows, unavailable_methods=q_unavailable)
    if comparator is not None and q_selection.locked_comparator != comparator:
        raise ValueError("query metric ordering differs between CAL and SELECT adapters")

    candidates, baselines, bootstrap = {}, {}, {}
    if s_selection.selected_method != "N0":
        candidates[s_selection.selected_method] = semantic_rows[s_selection.selected_method]
        baselines[s_selection.selected_method] = semantic_rows["N0"]
    if g_selection.selected_method != "G_ORIGINAL":
        candidates[g_selection.selected_method] = geometry_rows[g_selection.selected_method]
        baselines[g_selection.selected_method] = geometry_rows["G_ORIGINAL"]
    q_retained = q_selection.status in {"RETAINED", "RETAINED_ENGINEERING"}
    if q_retained:
        candidates["Q_GAIN"] = query_rows["Q_GAIN"]
        baselines["Q_GAIN"] = query_rows[comparator]
    independent_final = select_final_candidate(candidates, matched_baselines=baselines,
                                                simplicity_order=tuple(candidates))
    for method in independent_final.eligible_methods:
        bootstrap[method] = {**asdict(bootstrap_mean_delta([row.uap for row in candidates[method]],
            [row.uap for row in baselines[method]])), "per_scene_delta": {
                row.scene_id: row.uap - baseline.uap for row, baseline in zip(candidates[method], baselines[method], strict=True)},
            "domain": "B200 allowance" if method.startswith("Q_") else "matched native geometry domain"
                if method.startswith("G_") else "fixed N0 masks and ranks",
            "interpretation": "descriptive two-scene bootstrap; not population-level assurance"}
    combinations = plan_combinations(semantic_method=s_selection.selected_method,
        geometry_method=g_selection.selected_method, query_method="Q_GAIN" if q_retained else "NONE",
        fresh_mask_supported=True, query_provenance_supported=True)
    inconclusive = [name for name, status in (("S", s_selection.status), ("G", g_selection.status), ("Q", q_selection.gain_status))
                    if status.startswith("INCONCLUSIVE")]
    final = asdict(independent_final)
    if inconclusive and final["selected_method"] == "N0":
        final["science_status"] = "INCONCLUSIVE_UNDEFINED_METRIC"
    return {"status": "STANDALONE_MODULES_SELECTED", "semantic": asdict(s_selection), "geometry": asdict(g_selection),
        "query": asdict(q_selection), "independent_final": final, "combination_plan": asdict(combinations),
        "unavailable_methods": unavailable, "bootstrap": bootstrap, "inconclusive_branches": inconclusive,
        "budget_curves": {"status": "REQUIRED" if q_selection.gain_status == "ELIGIBLE_ACCURACY"
            else "NOT_REQUIRED_QUERY_GAIN_NOT_ELIGIBLE", "budgets": [100, 400], "comparator_id": comparator},
        "SELECT_scenes": list(select_scenes), "CAL_scenes": list(cal_scenes),
        "confirmation_authorized": False}
