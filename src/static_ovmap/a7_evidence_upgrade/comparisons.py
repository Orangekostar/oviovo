"""Explicit pooled, scene-mean and matched-factor comparisons of real results."""

import math
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest

from .audit import METRICS, measured_methods
from .evaluation import RANKS


def delta(new, old):
    return {key: 100 * (new[key] - old[key]) for key in METRICS}


def interaction_delta(values, base, q, region, combo):
    return {k: 100 * (values[combo][k] - values[q][k] - values[region][k] + values[base][k]) for k in METRICS}


def compare_split(binding, split):
    if split not in {"cal", "replica"}:
        raise ValueError("comparison requires one declared split")
    root, index = Path(binding["output_root"]), InputIndex()
    spec = read_json(binding["spec"])
    scenes = spec["datasets"]["calibration" if split == "cal" else "replica"]
    methods, blocked = measured_methods(binding)
    index.identity(__file__)
    pair_path = root / "composition.json"
    index.identity(pair_path)
    pair = read_json(pair_path)
    results = {}
    for rank in RANKS:
        pooled, per_scene = {}, {}
        for method in methods:
            path = root / "pooled" / split / method / (rank + ".json")
            index.identity(path)
            row = read_json(path)
            if row["status"] != "COMPLETE" or row["aggregation"] != "RELEASED_DATASET_POOL" or row["scene_order"] != scenes:
                raise ValueError("comparison requires complete ordered released pools")
            pooled[method] = row["metrics"]
            per_scene[method] = {}
            for scene in scenes:
                path = root / "rows" / scene / method / (rank + ".json")
                index.identity(path)
                row = read_json(path)
                if row["status"] != "COMPLETE":
                    raise ValueError("comparison requires complete scene rows")
                per_scene[method][scene] = row["metrics"]
            for metrics in [pooled[method], *per_scene[method].values()]:
                if any(not math.isfinite(metrics[k]) for k in METRICS):
                    raise ValueError("missing/nonfinite metrics cannot be reported as zero")
        rows = {}
        for method in methods:
            rows[method] = {
                "released_pool_fraction": pooled[method],
                "scene_mean_fraction": {k: sum(v[k] for v in per_scene[method].values()) / len(scenes) for k in METRICS},
                "per_scene_fraction": per_scene[method],
                "pooled_delta_N0_pp": delta(pooled[method], pooled["N0"]),
                "pooled_delta_A7_pp": delta(pooled[method], pooled["RV_A7_COS_REFIT"]),
                "per_scene_delta_A7_pp": {s: delta(per_scene[method][s], per_scene["RV_A7_COS_REFIT"][s]) for s in scenes},
                "worst_scene_delta_A7_pp": {k: min(100 * (per_scene[method][s][k] - per_scene["RV_A7_COS_REFIT"][s][k]) for s in scenes) for k in METRICS},
            }
        contrasts = {}
        for mode in ("DIRECT", "A7"):
            for label, new, old in (
                ("E01_normalization", "AW_E01_UNIT_EQ", "AW_E01_RAW_EQ"),
                ("E01_robustness", "AW_E01_GMED", "AW_E01_UNIT_EQ"),
                ("E02_SAM2_vs_GLOBAL", "AW_E02_SAM2", "AW_E02_GLOBAL"),
                ("E03_tuned_vs_frozen", "AW_E03_OVR", "AW_E03_FC_FROZEN"),
            ):
                new, old = new + "_" + mode, old + "_" + mode
                contrasts[label + "_" + mode] = {"new": new, "old": old,
                    "pooled_delta_pp": delta(pooled[new], pooled[old]),
                    "per_scene_delta_pp": {s: delta(per_scene[new][s], per_scene[old][s]) for s in scenes}}
        interaction = None
        if "AW_COMBO_QR" in methods:
            base, q, region, combo = "RV_A7_COS_REFIT", pair["q_variant"] + "_A7", pair["region_variant"] + "_A7", "AW_COMBO_QR"

            interaction = {"formula": "Q+region - Q - region + base A7", "base": base, "Q": q, "region": region,
                           "Q+region": combo, "pooled_interaction_pp": interaction_delta(pooled, base, q, region, combo),
                           "per_scene_interaction_pp": {s: interaction_delta({m: per_scene[m][s] for m in methods}, base, q, region, combo) for s in scenes}}
        results[rank] = {"methods": rows, "matched_contrasts": contrasts, "interaction": interaction}
    result = {"status": "MEASURED_COMPARISONS", "split": split, "scene_order": scenes,
              "blocked_methods_without_numeric_results": blocked, "ranks": results,
              "units": "raw metrics are fractions; all deltas and interactions are percentage points",
              "scene_mean_is_not_released_pool": True, "inputs": index.entries()}
    result["identity"] = canonical_digest(result)
    write_once(root / "comparisons" / (split + ".json"), result)
    return result
