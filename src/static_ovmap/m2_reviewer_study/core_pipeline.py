"""Reproducible CAL-first core stage, including scalar setup and real pools."""

import time
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once

from .binding import InputIndex
from .calibration import fit_shared
from .core import evaluate_core_scene
from .pooling import evaluate_pools, validate_official_export, validate_pool_path
from .scores import read_scene
from .selection import fit_final, nominate


def fit_core_folds(binding):
    root = Path(binding["output_root"])
    index = InputIndex()
    path = Path(binding["composition_root"]) / "calibration/examples.json"
    expected = next(row for row in binding["inputs"] if row["path"] == str(path))
    index.identity(path, expected)
    original = read_json(path)["examples"]
    scenes = read_json(binding["spec"])["datasets"]["calibration"]
    examples = []
    for scene in scenes:
        evidence = read_scene(binding, scene, index)
        for row in original["N0"]:
            if row["scene_id"] == scene:
                source = evidence.cosine_native["objects"][str(row["owner_id"])]
                examples.append({**row, "scores": source["scores"], "available": source["available"]})
    ids = evidence.config["models"]["native"]["valid_ids"]
    destination = root / "calibration/new_folds.json"
    if destination.exists():
        result = read_json(destination)
        if result["cosine_examples"] != examples:
            raise ValueError("core fold inputs differ from saved calibration examples")
        return result
    started = time.monotonic()
    folds = {scene: {"shared": fit_shared(original, [s for s in scenes if s != scene], ids),
                     "cosine_N0": fit_shared({"N0": examples}, [s for s in scenes if s != scene], ids)} for scene in scenes}
    result = {"folds": folds, "cosine_examples": examples}
    write_once(destination, result)
    write_once(root / "calibration/fold_timing.json", {"elapsed_seconds": time.monotonic() - started, "fits": 4})
    return result


def run_core(binding):
    index = InputIndex()
    for entry in binding["inputs"]:
        index.identity(entry["path"], entry)
    spec = read_json(binding["spec"])
    fit_core_folds(binding)
    for scene in spec["datasets"]["calibration"]:
        evaluate_core_scene(binding, scene)
    nominate(binding)
    fit_final(binding)
    for scene in spec["datasets"]["official_replica_pool_order"]:
        evaluate_core_scene(binding, scene)
    validate_official_export(binding)
    for dataset in ("ScanNet", "Replica"):
        evaluate_pools(binding, dataset)
        validate_pool_path(binding, dataset)
    return {"status": "CORE_COMPLETE", "scene_methods": 130, "scene_rank_rows": 260,
            "Replica_pooled_rows": 26, "deployment": "N0_UNCHANGED"}
