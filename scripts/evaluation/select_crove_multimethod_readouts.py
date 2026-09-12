"""Freeze complete DEV task selections before static confirmation scoring."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_crove_adapter_room0 import path
from scripts.evaluation.run_ovimap_native import _atomic_json, _sha256


def rank_candidates(values, primary, secondary, *, eligible_only=False, tolerance=1e-6):
    candidates = [
        v
        for v in values
        if not eligible_only or v.get("passes_original_per_case_gates") is True
    ]
    if not candidates:
        return None
    for key in [primary, *secondary]:
        if any(v.get(key) is None for v in candidates):
            if key == primary:
                raise ValueError("primary selection metric missing")
            continue
        if any(not math.isfinite(v[key]) for v in candidates):
            raise ValueError("nonfinite selection metric")
        maximum = max(v[key] for v in candidates)
        candidates = [v for v in candidates if maximum - v[key] <= tolerance]
    return min(candidates, key=lambda v: v.get("config_id", v["exact_implementation"]))


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    compact = path(config["compact_output_root"])
    report_path = compact / "all_method_results.json"
    report = json.loads(report_path.read_text())
    if report["missing_local_background_results"]:
        raise RuntimeError(
            "complete local-background DEV matrix required before selection freeze"
        )
    rows = report["rows"]
    lookup = {
        (r["case"], r["state_variant"], r["exact_implementation"]): r for r in rows
    }
    if len(lookup) != len(rows):
        raise ValueError("duplicate method/case/state result")
    for case, state in (("room0", None), ("apartment", "B3"), ("apartment", "H2")):
        for method in ("MV_BG_OWNER_QUALITY", "MV_LOCAL_BG_QUALITY"):
            if (case, state, method) not in lookup:
                raise RuntimeError(
                    "all six local-background scores must be present in the collected matrix"
                )
    for row in rows:
        if _sha256(compact / row["result_file"]) != row["result_sha256"]:
            raise ValueError("scored results changed since collection")
        if not row["geometry_fixed"] or (
            not row["exact_implementation"].startswith("INST_") and row["owner_changed"]
        ):
            raise ValueError("candidate violates frozen geometry/owner contract")
    tolerance = config["selection_policy"]["metric_absolute_tolerance"]
    native_static = "B_SEM_OVI_NATIVE"
    native_dynamic = "MV_NATIVE_CACHED"
    m1 = [
        "MV_SINGLE",
        "MV_TOPK_MEAN",
        "MV_DIVERSE_MEAN",
        "MV_QUALITY",
        "MV_REENCODE_DIVERSE",
        "MV_REENCODE_QUALITY",
        "MV_BG_OWNER_QUALITY",
        "MV_LOCAL_BG_QUALITY",
    ]
    m1_graph = [
        "MV_QUALITY_PATCH_ONLY",
        "MV_QUALITY_GRAPH_GEOM",
        "MV_QUALITY_GRAPH_BOUNDARY",
    ]
    m4 = ["ADAPTER_CLIP_MEAN", "ADAPTER_CLIP_LEARNED"]
    selections = []

    def choose(family, case, task, new_methods, controls):
        state = None if case == "room0" else "B3"
        primary, secondary = (
            ("CA_AP50", ["CA_AP25", "CA_AR50"])
            if task == "STATIC_INSTANCE"
            else ("static_miou", ["f_miou"])
            if state is None
            else ("dynamic_current_miou", ["Surface_F5"])
        )
        candidates = [
            lookup[(case, state, m)] for m in sorted(set(new_methods + controls))
        ]
        fresh = [v for v in candidates if v["exact_implementation"] in new_methods]
        result = {
            "family": family,
            "dataset_protocol": "Replica41_static"
            if state is None
            else "Tesse_common_v2",
            "task": task,
            "dev_cases": [case],
            "selection_state": state,
            "primary_metric": primary,
            "candidate_methods": [v["exact_implementation"] for v in candidates],
            "metric_winner": rank_candidates(
                candidates, primary, secondary, tolerance=tolerance
            ),
            "best_new_candidate": rank_candidates(
                fresh, primary, secondary, tolerance=tolerance
            ),
            "eligible_winner": rank_candidates(
                candidates, primary, secondary, eligible_only=True, tolerance=tolerance
            )
            if state
            else None,
        }
        selections.append(result)
        return result

    for case in ("room0", "apartment"):
        static = case == "room0"
        controls = (
            [native_static, "B_SEM_CROVE_S0", "B_SEM_CROVE_S2"]
            if static
            else [
                native_dynamic,
                "B_SEM_CROVE_S0_DENSE_REPLAY",
                "B_SEM_CROVE_S2_DENSE_REPLAY",
            ]
        )
        task = "STATIC_SEMANTIC" if static else "DYNAMIC_SEMANTIC"
        m1_selection = choose(
            "M1",
            case,
            task,
            m1 + ([] if static else ["MV_CURRENT_AWARE"]),
            controls + ["B_SEM_OVI_REENCODE"],
        )
        representative = m1_selection["best_new_candidate"]["exact_implementation"]
        graph_prefix = {"MV_QUALITY": "MV_QUALITY", "MV_TOPK_MEAN": "MV_TOPK"}.get(
            representative
        )
        if graph_prefix is None or any(
            (case, None if static else "B3", f"{graph_prefix}_{variant}") not in lookup
            for variant in ("PATCH_ONLY", "GRAPH_GEOM")
        ):
            raise RuntimeError(
                f"strongest new M1 representative {case}/{representative} requires its same-graph direct combination before freeze"
            )
        prefix = "S2" if static else "S2_DENSE_REPLAY"
        choose(
            "M2",
            case,
            task,
            [f"{prefix}_{v}" for v in ("PATCH_ONLY", "GRAPH_GEOM", "GRAPH_BOUNDARY")]
            + m1_graph
            + ([] if static else ["MV_TOPK_PATCH_ONLY", "MV_TOPK_GRAPH_GEOM"]),
            controls + ["MV_QUALITY"],
        )
        choose("M4", case, task, m4, controls)
        choose(
            "M3_RESEM", case, task, ["INST_CONSENSUS_RESEM"], ["INST_CONSENSUS_OWNER"]
        )
    instance = choose(
        "M3",
        "room0",
        "STATIC_INSTANCE",
        ["INST_PAIRWISE", "INST_CONSENSUS_OWNER"],
        [native_static],
    )
    # All dynamic owner-only choices transfer from static instance evidence.
    instance["dynamic_transfer"] = {
        "metric_winner": native_dynamic
        if instance["metric_winner"]["exact_implementation"] == native_static
        else instance["metric_winner"]["exact_implementation"],
        "new_representative": instance["best_new_candidate"]["exact_implementation"],
        "dynamic_instance_scores_used": False,
    }
    finalists = {native_dynamic}
    for entry in selections:
        if entry["task"] == "DYNAMIC_SEMANTIC":
            for key in ("metric_winner", "eligible_winner"):
                if entry[key] is not None:
                    finalists.add(entry[key]["exact_implementation"])
    finalists.update(["ADAPTER_LEARNED_PATCH_ONLY", "ADAPTER_LEARNED_GRAPH_GEOM"])
    final_rows = []
    for state in ("B3", "H2"):
        for method in sorted(finalists):
            value = dict(lookup[("apartment", state, method)])
            value["config_id"] = f"{method}@{state}"
            final_rows.append(value)
    final = rank_candidates(
        final_rows,
        "dynamic_current_miou",
        ["Surface_F5"],
        eligible_only=True,
        tolerance=tolerance,
    )
    confirmation_methods = {native_static, "B_SEM_CROVE_S0", "B_SEM_CROVE_S2"}
    for entry in selections:
        if entry["task"].startswith("STATIC_") and entry["family"] != "M3_RESEM":
            for key in ("metric_winner", "best_new_candidate"):
                row = entry[key]
                confirmation_methods.add(row["exact_implementation"])
                if row["direct_control"]:
                    confirmation_methods.add(row["direct_control"])
    result = {
        "status": "DEV_SELECTION_FROZEN",
        "selection_policy": config["selection_policy"],
        "source_summary_file": "selection_source_results.json",
        "source_summary_sha256": _sha256(report_path),
        "runtime_tie_policy": "skip: no common complete added-cost measurement across all tied candidates; absent stages are not zero",
        "static_alias_policy": "MV_CURRENT_AWARE is the exact MV_QUALITY configuration in a one-visit scene; report row retained, duplicate configuration not ranked twice",
        "candidate_domains": "shared native/S0/S2 baselines participate in M1/M2/M4; resem compares the same consensus partition; M3 owner-only uses native instance control",
        "task_selections": selections,
        "FINAL_CURRENT_READOUT": final,
        "final_current_status": "ELIGIBLE_CONFIGURATION"
        if final
        else "NO_ELIGIBLE_CONFIGURATION",
        "final_current_candidates": [
            {
                "config_id": r["config_id"],
                "eligible": r["passes_original_per_case_gates"],
            }
            for r in final_rows
        ],
        "static_confirmation": {
            "scene": "room1",
            "methods": sorted(confirmation_methods),
            "retuning_allowed": False,
            "include_best_new_representative_even_if_baseline_wins": True,
        },
        "method_registries": {
            p.name: {"sha256": _sha256(p), "settings": json.loads(p.read_text())}
            for p in sorted(compact.glob("*_registry.json"))
        },
    }
    target = compact / "selected_configs.json"
    if target.exists():
        if json.loads(target.read_text()) != result:
            raise ValueError("existing frozen DEV selection differs")
        print("frozen selection unchanged", flush=True)
        return
    for filename in compact.glob("room1_*.json"):
        if json.loads(filename.read_text()).get("status") == "FULL_MAP_EVALUATED":
            raise ValueError("confirmation was scored before DEV selection freeze")
    _atomic_json(compact / "selection_source_results.json", report)
    _atomic_json(target, result)
    print(
        "DEV selection frozen; room1 methods", sorted(confirmation_methods), flush=True
    )


if __name__ == "__main__":
    main()
