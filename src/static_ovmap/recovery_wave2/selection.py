"""Fraction-scale fixed bands, single light composition, and committed transfer."""

import math
from pathlib import Path
import subprocess

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest

from .binding import read
from .costs import light_costs, map_cost
from .light import METHODS, RECOVERIES, WEIGHTS


def feasible(metrics, baseline):
    return all(metrics[key] + 1e-12 >= baseline[key] + delta for key, delta in
               (("apall", -.0005), ("ap50", -.001), ("miou", -.001)))


def net_gain(metrics, baseline):
    return all(metrics[key] + 1e-12 >= baseline[key] + delta for key, delta in
               (("apall", .002), ("ap50", -.001), ("miou", -.001)))


def _tie(row):
    return (row["standalone_new_encoder_inputs"], row["changed_blocks"], row["parameter_distance"], row["id"])


def banded_rank(rows):
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("selection requires a nonempty unique complete candidate set")
    if any(not all(math.isfinite(row["metrics"][key]) for key in ("apall", "miou", "ap50")) for row in rows):
        raise ValueError("missing selection metrics cannot be substituted with zero")
    remaining, ranking, steps = list(rows), [], []
    while remaining:
        band, trace = list(remaining), []
        for metric, width in (("apall", .0005), ("miou", .001), ("ap50", .001)):
            best = max(row["metrics"][metric] for row in band)
            band = [row for row in band if row["metrics"][metric] + 1e-12 >= best - width]
            trace.append({"metric": metric, "maximum": best, "width_fraction": width,
                          "retained": sorted(row["id"] for row in band)})
        band.sort(key=_tie)
        chosen = band[0]
        ranking.append(chosen["id"])
        steps.append({"chosen": chosen["id"], "metric_bands": trace,
                      "cost_and_complexity_order": [row["id"] for row in band]})
        remaining = [row for row in remaining if row["id"] != chosen["id"]]
    strict = sorted(rows, key=lambda row: (-row["metrics"]["apall"], -row["metrics"]["miou"],
                                          -row["metrics"]["ap50"], *_tie(row)))
    return {"banded_preference": ranking, "strict_metric_ranking": [row["id"] for row in strict], "steps": steps,
            "bands_are_preferences_not_statistical_equivalence": True}


def select_components(rows):
    if {row["id"] for row in rows} != set(METHODS) or len(rows) != len(METHODS):
        raise ValueError("component selection requires all eight fixed complete light conditions")
    base = next(row for row in rows if row["id"] == "RW_B_D2")["metrics"]
    admissible = [row for row in rows if feasible(row["metrics"], base)]
    w = banded_rank([row for row in admissible if row["id"] in WEIGHTS])
    u = banded_rank([row for row in admissible if row["id"] in RECOVERIES or row["id"] == "RW_B_D2"])
    gamma, recovery = WEIGHTS[w["banded_preference"][0]], RECOVERIES.get(u["banded_preference"][0])
    return {"gamma": gamma, "recovery": recovery, "W_selected_id": w["banded_preference"][0],
        "U_selected_id": u["banded_preference"][0], "W_ranking": w, "U_ranking": u,
        "feasible_ids": [row["id"] for row in admissible],
        "combination": {"RW_LIGHT_COMBO": {"gamma": gamma, "recovery": recovery}},
        "combination_count": 1, "Replica_results_read": False}


def light_selection_rows(binding, *, include_combination=False):
    root, scenes = Path(binding["output_root"]), binding["datasets"]["development"]
    costs = [light_costs(binding, scene)["methods"] for scene in scenes]
    methods = METHODS + (["RW_LIGHT_COMBO"] if include_combination else [])
    components = read(root / "selection/components.json") if include_combination else None
    rows = []
    for method in methods:
        pool = read(root / "light/pools/development/BB00_NATIVE" / (method + ".json"))
        if pool["status"] != "COMPLETE" or pool["scene_order"] != scenes:
            raise ValueError("light selection requires exact ordered complete development pools")
        if method in WEIGHTS:
            recipe = {"gamma": WEIGHTS[method], "recovery": None}
        elif method in RECOVERIES:
            recipe = {"gamma": .5, "recovery": RECOVERIES[method]}
        else:
            recipe = components["combination"][method]
        equivalent_cost_method = next((name for name, arm in RECOVERIES.items() if arm == recipe["recovery"]), "RW_B_D2")
        rows.append({"id": method, "recipe": recipe, "metrics": pool["metrics"], "pool_identity": pool["identity"],
            "standalone_new_encoder_inputs": sum(cost[equivalent_cost_method]["standalone_new_encoder_inputs"] for cost in costs),
            "changed_blocks": int(recipe["gamma"] != .5) + int(recipe["recovery"] is not None),
            "parameter_distance": abs(recipe["gamma"] - .5)})
    return rows


def write_components(binding):
    rows = light_selection_rows(binding)
    result = {"status": "COMPONENTS_SELECTED_DEVELOPMENT_ONLY", **select_components(rows),
              "inputs": rows, "binding_identity": binding["identity"]}
    result["identity"] = canonical_digest(result)
    path = Path(binding["output_root"]) / "selection/components.json"
    if path.is_file() and read(path) != result:
        raise ValueError("fixed component selection changed; retain the original decision")
    atomic_write_json(path, result)
    return result


def select_final(binding):
    root, spec = Path(binding["output_root"]), read(binding["spec"])
    lights = light_selection_rows(binding, include_combination=True)
    baseline = next(row for row in lights if row["id"] == "RW_B_D2")["metrics"]
    feasible_lights = [row for row in lights if feasible(row["metrics"], baseline)]
    light_rank = banded_rank(feasible_lights)
    chosen_light = next(row for row in feasible_lights if row["id"] == light_rank["banded_preference"][0])
    maps, qualified = [], []
    for recipe in spec["map_variants"]:
        map_id = recipe["id"]
        screen = read(root / "geometry/screens" / (map_id + ".json"))
        row = {"id": map_id, "recipe": recipe, "geometry_status": screen["status"],
               "geometry_screen_identity": screen["identity"], "metrics": None, "qualifies": False}
        if screen["status"] == "GEOMETRY_PASS":
            pool = read(root / "pools/development" / map_id / "D2/OFFICIAL_CURRENT_CLASS.json")
            if pool["status"] != "COMPLETE" or pool["scene_order"] != binding["datasets"]["development"]:
                raise ValueError("map nomination requires its complete four-scene fixed-D2 pool")
            costs = [map_cost(binding, scene, map_id) for scene in binding["datasets"]["development"]]
            row.update(metrics=pool["metrics"], pool_identity=pool["identity"],
                standalone_new_encoder_inputs=sum(cost["standalone_new_encoder_inputs"] for cost in costs),
                changed_blocks=1, parameter_distance=0., qualifies=net_gain(pool["metrics"], baseline))
            if row["qualifies"]:
                qualified.append(row)
        elif screen["status"] not in {"SCREENED_OUT_GEOMETRY", "EQUIVALENT_INPUT", "BLOCKED_MISSING_ARTIFACT"}:
            raise ValueError("map selection cannot treat an unfinished arm as complete")
        maps.append(row)
    map_rank = banded_rank(qualified) if qualified else None
    nominee = map_rank["banded_preference"][0] if map_rank else None
    result = {"status": "DEVELOPMENT_SELECTION_LOCKED", "binding_identity": binding["identity"],
        "scene_order": binding["datasets"]["development"], "light_inputs": lights, "light_ranking": light_rank,
        "light_package": chosen_light, "components": read(root / "selection/components.json"),
        "map_inputs": maps, "map_ranking": map_rank, "nominated_map": nominee,
        "technical_caps": spec["limits"], "recovery_caps": spec["recovery"],
        "selection_rules": spec["selection"], "replica_scene_order": binding["datasets"]["replica"],
        "Replica_results_read": False, "deployment": "N0_UNCHANGED", "new_A_plus_S_combinations": 0}
    result["identity"] = canonical_digest(result)
    path = root / "selection/final.json"
    if path.is_file() and read(path) != result:
        raise ValueError("development selection changed after locking")
    atomic_write_json(path, result)
    return result


def commit_freeze(binding, selection, composition_check):
    if selection["status"] != "DEVELOPMENT_SELECTION_LOCKED" or composition_check["status"] != "COMPLETE":
        raise ValueError("transfer freeze requires complete scalar/map selection and two-by-two evidence")
    repo, root = Path(binding["repository_root"]), Path(binding["output_root"])
    relative = "artifacts/static_ovmap/recovery_wave2_v1/selection.json"
    committed = {"selection": selection, "development_composition_check": composition_check,
        "frozen_methods": read(binding["spec"])["light_methods"],
        "no_Replica_retuning": True, "deployment": "N0_UNCHANGED"}
    path = repo / relative
    if path.is_file() and read(path) != committed:
        raise ValueError("committed transfer choices cannot be replaced")
    atomic_write_json(path, committed)
    subprocess.run(["git", "add", relative], cwd=repo, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative], cwd=repo).returncode
    if changed:
        subprocess.run(["git", "commit", "--only", "-m", "freeze: lock recovery development choices before Replica", relative],
                       cwd=repo, check=True)
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", relative], cwd=repo, text=True).strip()
    observed = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=repo)
    import json

    if canonical_digest(json.loads(observed)) != canonical_digest(committed):
        raise ValueError("pre-Replica freeze commit content differs from the reviewed frozen package")
    receipt = {"status": "FROZEN_COMMITTED", "commit": commit, "repository_file": relative,
        "selection_identity": canonical_digest(committed), "nominated_map": selection["nominated_map"],
        "light_package": selection["light_package"], "Replica_predictions_started_before_freeze": False,
        "binding_identity": binding["identity"]}
    destination = root / "freeze/receipt.json"
    if destination.is_file() and read(destination) != receipt:
        raise ValueError("pre-Replica commit receipt is immutable")
    atomic_write_json(destination, receipt)
    return receipt
