"""Fixed CAL-only metric ordering with measured operation-count tie breaks."""

import math
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest

from .calibration import source_path
from .evaluation import RANKS
from .operations import method_operations, scene_operations


def rank_candidates(rows, registry):
    """Resolve tolerance ties against the best remaining metric at each step."""
    order = {name: i for i, name in enumerate(registry)}
    if len({r["method"] for r in rows}) != len(rows):
        raise ValueError("duplicate selection candidate")
    for row in rows:
        if row.get("split") != "cal" or row.get("rank_mode") != "OFFICIAL_CURRENT_CLASS":
            raise ValueError("research nomination requires CAL official pools")
        if row.get("status") != "COMPLETE" or row["method"] not in order:
            raise ValueError("unmeasured or unregistered selection candidate")
        if any(not math.isfinite(row["metrics"][k]) for k in ("apall", "miou", "ap50")):
            raise ValueError("selection metric is undefined")
        count = row["unique_required_operations"]
        if not isinstance(count, int) or count < 0:
            raise ValueError("selection requires exact operation count")
    remaining, ranked = list(rows), []
    while remaining:
        tied = remaining
        for metric in ("apall", "miou", "ap50"):
            best = max(r["metrics"][metric] for r in tied)
            tied = [r for r in tied if best - r["metrics"][metric] <= 1e-10]
        winner = min(tied, key=lambda r: (r["unique_required_operations"], order[r["method"]]))
        ranked.append(winner)
        remaining = [r for r in remaining if r["method"] != winner["method"]]
    return ranked


def collect_cal(binding, *, pair=None):
    root, index = Path(binding["output_root"]), InputIndex()
    spec = read_json(binding["spec"])
    variants = [r["id"] for r in spec["source_variants"]]
    methods = spec["references"] + [v + suffix for v in variants for suffix in ("_DIRECT", "_A7")]
    methods.append("AW_E04_SHORTLIST")
    access_path = root / "e04/access_probe.json"
    index.identity(access_path)
    access = read_json(access_path)
    blocked = []
    if access["status"] == "BLOCKED_ASSET_ACCESS":
        if access["http_status"] not in (401, 403) or access["model_executed"]:
            raise ValueError("inconsistent external asset block")
        blocked = ["AW_E04_SPATIAL", "AW_E04_MIX50"]
    else:
        methods += ["AW_E04_SPATIAL", "AW_E04_MIX50"]
    if pair:
        methods.append("AW_COMBO_QR")
    operations = {scene: scene_operations(binding, scene, variants) for scene in spec["datasets"]["calibration"]}
    rows, changes = {}, {}
    for method in methods:
        for scene in spec["datasets"]["calibration"]:
            for rank in RANKS:
                path = root / "rows" / scene / method / (rank + ".json")
                index.identity(path)
                if read_json(path)["status"] != "COMPLETE":
                    raise ValueError("CAL matrix incomplete before selection")
        for rank in RANKS:
            path = root / "pooled/cal" / method / (rank + ".json")
            index.identity(path)
            row = read_json(path)
            if row["status"] != "COMPLETE" or row["scene_order"] != spec["datasets"]["calibration"]:
                raise ValueError("CAL official pool scope incomplete")
            if rank == "OFFICIAL_CURRENT_CLASS":
                union = {}
                for scene in operations:
                    union.update(method_operations(operations[scene]["sources"], method, pair))
                rows[method] = {**row, "unique_required_operations": len(union),
                                "required_operation_union_identity": canonical_digest(sorted(union))}
    for definition in spec["source_variants"]:
        variant, slot = definition["id"], definition["slot"]
        changed, maximum = 0, 0.
        for scene in spec["datasets"]["calibration"]:
            path = source_path(binding, scene, variant)
            index.identity(path)
            source = read_json(path)
            if canonical_digest({k: v for k, v in source.items() if k != "identity"}) != source["identity"]:
                raise ValueError("source changed before selection")
            original_path = binding["scenes"][scene]["sources"][slot]
            index.identity(original_path)
            original = read_json(original_path)
            for owner, obj in source["objects"].items():
                old = original["objects"][owner]
                if not obj["available"] or not old["available"]:
                    continue
                delta = float(np.max(np.abs(np.asarray(obj["scores"]) - np.asarray(old["scores"])) ))
                if not np.isfinite(delta):
                    raise ValueError("nonfinite candidate source difference")
                changed += delta > 0
                maximum = max(maximum, delta)
        changes[variant] = {"changed_genuine_common_owners": int(changed), "maximum_score_difference": maximum}
    return {"rows": rows, "source_changes": changes, "blocked_methods": blocked, "inputs": index.entries()}


def freeze_pair(binding):
    root = Path(binding["output_root"])
    if (root / "composition.json").exists():
        return read_json(root / "composition.json")
    cal = collect_cal(binding)
    spec = read_json(binding["spec"])
    registry = binding["methods"] + [binding["optional_method"]]
    groups = {}
    for name, key in (("q", "q_candidates"), ("region", "region_candidates")):
        candidates = [cal["rows"][v + "_A7"] for v in spec["composition"][key]
                      if cal["source_changes"][v]["changed_genuine_common_owners"] > 0]
        groups[name] = rank_candidates(candidates, registry)
    pair = {"binding": binding["identity"], "status": "PAIR_FROZEN" if all(groups.values()) else "NO_ELIGIBLE_PAIR",
            "q_variant": groups["q"][0]["method"].removesuffix("_A7") if groups["q"] else None,
            "region_variant": groups["region"][0]["method"].removesuffix("_A7") if groups["region"] else None,
            "cal_rankings": groups, "source_changes": cal["source_changes"], "inputs": cal["inputs"],
            "requires_individual_net_gain": False, "new_scalar_fit": False, "Replica_used_for_selection": False}
    pair["identity"] = canonical_digest(pair)
    write_once(root / "composition.json", pair)
    return pair


def freeze_nomination(binding):
    root = Path(binding["output_root"])
    if (root / "nomination.json").exists():
        return read_json(root / "nomination.json")
    pair = read_json(root / "composition.json")
    cal = collect_cal(binding, pair=pair if pair["status"] == "PAIR_FROZEN" else None)
    eligible = [r for name, r in cal["rows"].items()
                if name.endswith("_A7") or name in {"N0", "RV_A7_COS_REFIT", "AW_E04_MIX50", "AW_COMBO_QR"}]
    rankings = rank_candidates(eligible, binding["methods"] + [binding["optional_method"]])
    result = {"status": "CAL_NOMINATION_FROZEN", "binding": binding["identity"],
              "nomination": rankings[0]["method"], "rankings": rankings,
              "composition_identity": pair["identity"], "blocked_methods": cal["blocked_methods"],
              "inputs": cal["inputs"], "Replica_used_for_selection": False,
              "data_role": "PREVIOUSLY_EXPOSED_DEVELOPMENT", "deployment": "N0_UNCHANGED"}
    result["identity"] = canonical_digest(result)
    write_once(root / "nomination.json", result)
    return result


def lock_transfer(binding):
    """Bind actual final scalars and choices before the first new Replica job."""
    root, index = Path(binding["output_root"]), InputIndex()
    path = root / "transfer_lock.json"
    if path.exists():
        previous = read_json(path)
        if previous["binding"] != binding["identity"]:
            raise ValueError("transfer input binding changed")
        for item in previous["inputs"]:
            index.identity(item["path"], item)
        return previous
    for name in ("nomination.json", "composition.json"):
        index.identity(root / name)
        value = read_json(root / name)
        if canonical_digest({k: v for k, v in value.items() if k != "identity"}) != value["identity"]:
            raise ValueError("frozen choice identity changed")
    nomination = read_json(root / "nomination.json")
    if nomination["status"] != "CAL_NOMINATION_FROZEN" or nomination["Replica_used_for_selection"]:
        raise ValueError("Replica transfer has no valid CAL nomination")
    spec = read_json(binding["spec"])
    scalars = {}
    for variant in spec["source_variants"]:
        fit_path = root / "calibration" / (variant["id"] + ".json")
        index.identity(fit_path)
        fit = read_json(fit_path)
        if fit["status"] != "CAL_FITS_COMPLETE" or fit["Replica_fitting"]:
            raise ValueError("transfer scalar was not CAL-only")
        scalars[variant["id"]] = fit["final"]["temperature"]
    for item in binding["inputs"]:
        index.identity(item["path"], item)
    result = {"status": "TRANSFER_INPUTS_FROZEN", "binding": binding["identity"],
              "nomination_identity": nomination["identity"], "final_scalars": scalars,
              "inputs": index.entries(), "Replica_fitting": False}
    result["identity"] = canonical_digest(result)
    write_once(path, result)
    return result
