"""Predeclared bands, standalone costs, and immutable candidate/transfer freezes."""

from pathlib import Path
import statistics

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from .binding import read


STRUCTURAL_CHAMPION_IDS = {"BB01_SYNC", "BB05_FORWARD", "BB05_BIDIR"}


def banded_rank(rows):
    if not rows:
        raise ValueError("selection requires complete candidate pools")
    remaining = list(rows)
    ranking, steps = [], []
    while remaining:
        band = list(remaining)
        trace = []
        for key, width in (("apall", .0005), ("miou", .001), ("ap50", .001)):
            best = max(row["metrics"][key] for row in band)
            band = [row for row in band if row["metrics"][key] >= best - width]
            trace.append({"metric": key, "maximum": best, "band_width": width,
                          "retained": [row["id"] for row in band]})
        band.sort(key=lambda row: (row["required_image_encodings"], row["median_standalone_seconds"],
                                  row["changed_blocks"], row["id"]))
        selected = band[0]
        ranking.append(selected["id"])
        steps.append({"selected": selected["id"], "metric_bands": trace,
                      "cost_tie_order": [row["id"] for row in band]})
        remaining = [row for row in remaining if row["id"] != selected["id"]]
    strict = sorted(rows, key=lambda row: (-row["metrics"]["apall"], -row["metrics"]["miou"],
        -row["metrics"]["ap50"], row["required_image_encodings"], row["median_standalone_seconds"],
        row["changed_blocks"], row["id"]))
    return {"banded_preference": ranking, "strict_metric_ranking": [row["id"] for row in strict], "steps": steps}


def selection_inputs(binding, spec, recipes):
    root = Path(binding["output_root"])
    result = []
    for recipe in recipes:
        path = root / "pools/development" / recipe["id"] / "D2/OFFICIAL_CURRENT_CLASS.json"
        if not path.is_file():
            continue
        pool = read(path)
        scenes = spec["datasets"]["development"]
        if pool["status"] != "COMPLETE" or pool["scene_order"] != scenes:
            continue
        receipts = [read(root / "readouts" / scene / recipe["id"] / "receipt.json") for scene in scenes]
        interventions = [read(root / "readouts" / scene / recipe["id"] / "intervention.json")
                         for scene in scenes] if recipe["id"] != "BB00_NATIVE" else []
        changed = sum(recipe[key] != value for key, value in
                      (("depth_fusion", "native"), ("association", "native"), ("frontend", "cropformer")))
        result.append({"id": recipe["id"], "recipe": recipe, "config_hash": canonical_digest(recipe),
            "pool_identity": pool["identity"], "metrics": pool["metrics"],
            "required_image_encodings": sum(row["required_image_encodings"] for row in receipts),
            "median_standalone_seconds": statistics.median(row["attributable_map_plus_readout_seconds"] for row in receipts),
            "changed_blocks": changed, "real_intervention": any(row["real_raw_partition_intervention"] for row in interventions),
            "interventions": interventions, "readout_identities": [row["identity"] for row in receipts]})
    return result


def write_frozen(path, value):
    value = dict(value, identity=canonical_digest(value))
    path = Path(path)
    if path.is_file() and read(path) != value:
        raise ValueError("frozen selection changed; retain the original freeze")
    atomic_write_json(path, value)
    return value


def freeze_candidates(binding, spec, implementation_commit):
    rows = selection_inputs(binding, spec, spec["map_variants"])
    if len(rows) != 7:
        raise ValueError("candidate freeze requires all seven complete four-scene configurations")
    existing = Path(binding["output_root"]) / "candidate_freeze.json"
    if existing.is_file():
        saved = read(existing)
        if saved["inputs"] != rows or saved["input_identity"] != binding["identity"]:
            raise ValueError("development inputs changed after candidate freeze")
        return saved
    families = {}
    for family in ("structural", "frontend"):
        family_rows = [row for row in rows if row["recipe"]["family"] == family and
                       (family != "structural" or row["id"] in STRUCTURAL_CHAMPION_IDS)]
        rank = banded_rank(family_rows)
        chosen = next(row for row in family_rows if row["id"] == rank["banded_preference"][0])
        families[family] = {"chosen": chosen, "ranking": rank, "eligible_ids": [row["id"] for row in family_rows]}
    composition = None
    if all(families[name]["chosen"]["real_intervention"] for name in families):
        structural, frontend = families["structural"]["chosen"], families["frontend"]["chosen"]
        composition = dict(structural["recipe"], id="BBX_COMPOSE", family="composition",
            frontend=frontend["recipe"]["frontend"],
            constituents={"structural": structural["id"], "frontend": frontend["id"]})
    result = {"status": "CANDIDATES_FROZEN", "implementation_commit": implementation_commit,
        "input_identity": binding["identity"], "selection_unit": "D2_DEVELOPMENT_RELEASED_POOL",
        "families": families, "inputs": rows, "composition_recipe": composition,
        "structural_control_exclusions": {"BB05_RATIO_GATE": "MANDATORY_SIMPLE_CONTROL_NOT_NONCONTROL_STRUCTURAL_CHAMPION"},
        "composition_skip_reason": None if composition else "NO_REAL_RAW_PARTITION_INTERVENTION_IN_AT_LEAST_ONE_CHAMPION",
        "individual_net_gain_required": False, "Replica_results_read": False}
    return write_frozen(Path(binding["output_root"]) / "candidate_freeze.json", result)


def freeze_transfer(binding, spec, implementation_commit):
    root = Path(binding["output_root"])
    candidates = read(root / "candidate_freeze.json")
    recipes = [spec["map_variants"][0], *[candidates["families"][name]["chosen"]["recipe"]
               for name in ("structural", "frontend")]]
    if candidates["composition_recipe"]:
        recipes.append(candidates["composition_recipe"])
    rows = selection_inputs(binding, spec, recipes)
    if len(rows) != len(recipes) or len(recipes) > 4:
        raise ValueError("transfer requires complete frozen candidate/composition measurements")
    ranking = banded_rank(rows)
    if (root / "selection.json").is_file():
        saved = read(root / "selection.json")
        if saved["inputs"] != rows or saved["candidate_freeze"] != candidates["identity"]:
            raise ValueError("development inputs changed after transfer freeze")
        return saved
    return write_frozen(root / "selection.json", {"status": "TRANSFER_FROZEN", "candidate_freeze": candidates["identity"],
        "implementation_commit": implementation_commit, "ranking": ranking, "inputs": rows,
        "nominee": ranking["banded_preference"][0], "replica_recipes": recipes,
        "replica_scene_order": spec["datasets"]["replica"], "Replica_results_read": False,
        "deployment": "N0_UNCHANGED", "refit_after_Replica": False})
