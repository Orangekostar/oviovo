"""Leave-one-scene-out pools of fixed measured methods, without fitting."""

from pathlib import Path

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest

from .binding import read
from .costs import light_costs
from .evaluation import pool_cohort
from .light import METHODS
from .selection import banded_rank, feasible, light_selection_rows, net_gain


def development_sensitivity(binding):
    from .workflow import pool_standard

    root, scenes = Path(binding["output_root"]), binding["datasets"]["development"]
    costs = {row["id"]: row for row in light_selection_rows(binding, include_combination=True)}
    fixed_selection = read(root / "selection/final.json")
    rows, rankings = [], []
    for excluded in scenes:
        retained = [scene for scene in scenes if scene != excluded]
        pools = pool_cohort(binding, "development", retained, [*METHODS, "RW_LIGHT_COMBO"], leave_out=excluded)
        baseline = pools["methods"]["RW_B_D2"]["metrics"]
        retained_costs = [light_costs(binding, scene)["methods"] for scene in retained]
        candidates = []
        for method, pool in pools["methods"].items():
            item = {"leave_out": excluded, "scene_order": retained, "method": method,
                    "map_id": "BB00_NATIVE", "status": "COMPLETE", "metrics": pool["metrics"],
                    "delta_pp": {key: 100 * (pool["metrics"][key] - baseline[key]) for key in
                                 ("apall", "ap50", "ap25", "miou", "macc")},
                    "feasible_in_diagnostic_pool": feasible(pool["metrics"], baseline),
                    "net_gain_in_diagnostic_pool": net_gain(pool["metrics"], baseline),
                    "pool_identity": pool["identity"]}
            rows.append(item)
            if item["feasible_in_diagnostic_pool"]:
                cost_method = method if method in METHODS else fixed_selection["components"]["U_selected_id"]
                candidates.append({**costs[method], "metrics": pool["metrics"],
                    "standalone_new_encoder_inputs": sum(value[cost_method]["standalone_new_encoder_inputs"] for value in retained_costs)})
        rankings.append({"leave_out": excluded, "fixed_light_ranking": banded_rank(candidates),
                         "frozen_four_scene_choice": fixed_selection["light_package"]["id"]})
        for item in fixed_selection["map_inputs"]:
            if item["geometry_status"] != "GEOMETRY_PASS":
                rows.append({"leave_out": excluded, "map_id": item["id"], "method": "D2",
                             "scene_order": retained, "status": item["geometry_status"], "metrics": None})
                continue
            standard = pool_standard(binding, "development", item["id"], leave_out=excluded)
            for name, pool in standard["pools"].items():
                if pool["rank_mode"] != "OFFICIAL_CURRENT_CLASS":
                    continue
                rows.append({"leave_out": excluded, "scene_order": retained, "map_id": item["id"],
                    "method": pool["method"], "status": "COMPLETE", "metrics": pool["metrics"],
                    "delta_pp": {key: 100 * (pool["metrics"][key] - baseline[key]) for key in
                                 ("apall", "ap50", "ap25", "miou", "macc")},
                    "net_gain_in_diagnostic_pool": net_gain(pool["metrics"], baseline), "pool_identity": pool["identity"]})
    receipt = {"status": "COMPLETE", "rows": rows, "rankings": rankings,
        "fixed_methods_and_original_predictions_only": True, "new_algorithms_fit": 0,
        "scene_average_used": False, "correlated_pools_are_not_independent_trials": True,
        "selection_changed": False, "four_scene_selection_identity": fixed_selection["identity"]}
    receipt["identity"] = canonical_digest(receipt)
    atomic_write_json(root / "selection/leave_one_out.json", receipt)
    return receipt
