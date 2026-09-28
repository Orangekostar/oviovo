"""Released query-control pools and separate random-seed spread, not ensembles."""

from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.released_loader import load_released_module

from .budget_controls import POLICIES
from .evaluation import pool


def summarize_queries(binding, dataset):
    root = Path(binding["output_root"])
    spec = read_json(binding["spec"])
    scenes = spec["datasets"]["official_replica_pool_order" if dataset == "Replica" else "calibration"]
    config = read_json(binding["scenes"][scenes[0]]["config"])
    namespace = load_released_module(Path(config["runtime"]["upstream"]) / "scripts/eval_utils.py")
    namespace["init"]("Replica" if dataset == "Replica" else "Scannet200")
    methods = [(f"{p}_B200" if mode == "STANDALONE" else f"RV_B_{p}_B200_{mode}", p, mode)
               for p in POLICIES for mode in ("STANDALONE", "RAW", "CAL")]
    if dataset == "Replica" and read_json(root / "query_controls/curve_gate.json")["triggered"]:
        methods += [(f"{p}_B{budget}" if mode == "STANDALONE" else f"RV_B_{p}_B{budget}_{mode}", p, mode)
                    for budget in (100, 400) for p in POLICIES[:2] for mode in ("STANDALONE", "RAW", "CAL")]
    rows = [read_json(root / "rows" / scene / method / (rank + ".json"))
            for scene in scenes for method, _, _ in methods for rank in ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS")]
    if any(row["status"] != "COMPLETE" for row in rows):
        raise ValueError("query summaries require every prescribed scene row")
    pooled, macro = [], []
    for method, policy, mode in methods:
        for rank in ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS"):
            result = {**pool(root, rows, namespace, scenes, method, rank), "method": method,
                      "policy": policy, "mode": mode, "rank_mode": rank, "dataset": dataset,
                      "aggregation": "RELEASED_DATASET_POOL"}
            write_once(root / "query_controls/pooled" / dataset / method / (rank + ".json"), result)
            pooled.append(result)
            selected = [r for r in rows if r["method"] == method and r["rank_mode"] == rank]
            macro.append({"dataset": dataset, "method": method, "policy": policy, "mode": mode, "rank_mode": rank,
                          "aggregation": "SCENE_MACRO", "scene_order": scenes, "metrics": {
                              key: float(np.mean([r["metrics"][key] for r in selected]))
                              for key in ("uap", "apall", "ap25", "ap50", "miou", "macc")}})
    spread = []
    for aggregation, results in (("RELEASED_DATASET_POOL", pooled), ("SCENE_MACRO", macro)):
        for rank in ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS"):
            for mode in ("STANDALONE", "RAW", "CAL"):
                selected = [r for r in results if r["policy"] in POLICIES[2:] and r["rank_mode"] == rank and r["mode"] == mode]
                if len(selected) != 3:
                    raise ValueError("random spread requires exactly three independent replicate rows")
                spread.append({"dataset": dataset, "aggregation": aggregation, "rank_mode": rank, "mode": mode,
                               "seeds": [17, 23, 41], "ensemble": False, "metrics": {
                    key: {"mean": float(np.mean(v)), "std_population": float(np.std(v)), "min": min(v), "max": max(v), "replicates": v}
                    for key in ("uap", "apall", "ap25", "ap50", "miou", "macc")
                    for v in [[r["metrics"][key] for r in selected]]}})
    result = {"status": "COMPLETE", "dataset": dataset, "scenes": scenes, "pooled": pooled, "macro": macro,
              "random_seed_spread": spread, "scene_rows": len(rows), "CAL_independent_benchmark": False}
    write_once(root / "query_controls" / (dataset + "_summary.json"), result)
    return result
