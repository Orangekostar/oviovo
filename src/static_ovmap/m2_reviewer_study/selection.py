"""Freeze comparator nomination from completed cross-fitted CAL records only."""

import time
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once

from .calibration import fit_shared
from .costs import required_union, scene_operations


def select_simple(summaries, order):
    candidates = [r for r in summaries if r["changed_owners"] > 0]
    if not candidates:
        return {"status": "NO_EFFECTIVE_INTERVENTION", "selected": "RV_A5_T001", "reason": "declared no-intervention diagnostic"}
    chosen = min(candidates, key=lambda r: (-r["uap"], -r["miou"], r["unique_operations"], order.index(r["method"])))
    return {"status": "NOMINATED", "selected": chosen["method"]}


def nominate(binding):
    root = Path(binding["output_root"])
    spec = read_json(binding["spec"])
    scenes = spec["datasets"]["calibration"]
    if any(list((root / "rows" / s).glob("*/*.json")) for s in spec["datasets"]["replica"]):
        if (root / "nomination.json").exists():
            return read_json(root / "nomination.json")
        raise ValueError("cannot nominate after new Replica evaluation")
    operations = {scene: scene_operations(binding, scene) for scene in scenes}
    locked = {s: read_json(root / "predictions" / s / "locked.json") for s in scenes}
    definitions = {r["method_id"]: r for r in spec["ablations"]}
    summaries = []
    for method in [*spec["selection"]["simple_candidates"], "Q_GAIN", "S_SIGLIP2_AREA"]:
        rows = [read_json(root / "rows" / scene / method / "FROZEN_N0.json") for scene in scenes]
        if any(r["status"] != "COMPLETE" or r["dataset"] != "ScanNet" for r in rows):
            raise ValueError("nomination requires completed ScanNet CAL rows")
        sources = definitions[method]["sources"] if method in definitions else [method]
        summaries.append({"method": method, "uap": float(np.mean([r["metrics"]["uap"] for r in rows])),
                          "miou": float(np.mean([r["metrics"]["miou"] for r in rows])),
                          "unique_operations": sum(len(required_union(operations[s]["operations"], sources)) for s in scenes),
                          "changed_owners": sum(sum(v != locked[s]["N0"]["labels"][o] for o, v in locked[s][method]["labels"].items()) for s in scenes),
                          "rows": rows})
    choice = select_simple(summaries[:6], spec["selection"]["simple_candidates"])
    single = min(summaries[6:], key=lambda r: (-r["uap"], -r["miou"], r["unique_operations"], ["Q_GAIN", "S_SIGLIP2_AREA"].index(r["method"])))
    result = {**choice, "primary": "CP_M2_EQUAL_CAL", "single_alternative": single["method"], "cal_scenes": scenes,
              "rule": spec["selection"]["order"], "summaries": summaries, "operations": operations,
              "frozen_before_replica": True, "deployment": "N0_UNCHANGED", "historical_Q_CAL_exposure": True}
    write_once(root / "nomination.json", result)
    return result


def fit_final(binding):
    root = Path(binding["output_root"])
    nomination = read_json(root / "nomination.json")
    path = root / "calibration/new_final.json"
    if path.exists():
        return read_json(path)
    composition = Path(binding["composition_root"])
    examples = read_json(composition / "calibration/examples.json")["examples"]
    folds = read_json(root / "calibration/new_folds.json")
    config = read_json(composition / "resolved_config.json")
    start = time.perf_counter()
    result = {"shared": fit_shared(examples, nomination["cal_scenes"], config["models"]["native"]["valid_ids"]),
              "cosine_N0": fit_shared({"N0": folds["cosine_examples"]}, nomination["cal_scenes"], config["models"]["native"]["valid_ids"])}
    result.update(elapsed_seconds=time.perf_counter() - start, nomination=nomination["selected"], original_M2_temperatures_unchanged=True)
    write_once(path, result)
    return result
