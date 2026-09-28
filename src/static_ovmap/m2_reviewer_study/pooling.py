"""Ordered multi-scene pools and a real official export/path parity check."""

import copy
import gzip
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from scripts.evaluation.evaluate_static_ovmap_instances import load_exports
from src.static_ovmap.composition_study.evaluation import load_targets
from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.module_validation.scannet_study import project_values
from src.static_ovmap.released_loader import load_released_module
from src.static_ovmap.released_trace import trace_released_matches

from .evaluation import plain, pool
from .scores import read_scene


def validate_official_export(binding, scene="scene0534_00", method="CP_M2_EQUAL_CAL"):
    root = Path(binding["output_root"])
    destination = root / "validation/official_export"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "receipt.json").exists():
        return read_json(destination / "receipt.json")
    evidence = read_scene(binding, scene)
    targets = load_targets(evidence.config, scene, evidence.native)
    frozen = read_json(root / "predictions" / scene / "locked.json")[method]
    labels = {int(k): int(v) for k, v in frozen["labels"].items()}
    owners = project_values(evidence.native.owner_ids, targets["nearest"], targets["matched"])
    semantic = np.zeros(owners.shape, np.int64)
    for owner, label in labels.items():
        semantic[owners == owner] = label
    template = PlyData.read(Path(targets["gt_instance_path"]).parent / "gt_instance_mesh.ply")
    if max(owners.max(), semantic.max()) > np.iinfo(template["vertex"]["label"].dtype).max:
        raise ValueError("official mesh label dtype cannot encode projected labels")
    for name, values in (("instance", owners), ("semantic", semantic)):
        mesh = copy.deepcopy(template)
        mesh["vertex"]["label"] = values.astype(mesh["vertex"]["label"].dtype)
        mesh.write(destination / (name + ".ply"))
    source = Path(evidence.config["runtime"]["upstream"]) / "scripts/eval_sem_seg.py"
    official = load_exports(source)["map_pred_mesh"]({"res_folder": str(destination), "inst_mesh_f": str(destination / "instance.ply"), "sem_mesh_f": str(destination / "semantic.ply")})
    row = read_json(root / "rows" / scene / method / "OFFICIAL_CURRENT_CLASS.json")
    receipt = read_json(row["evaluation_receipt"])
    expected = {int(k): v for k, v in receipt["view"].items()}
    actual = {}
    for line in Path(official).read_text().splitlines():
        file, label, rank = line.split()
        owner = int(file.split("_label-")[0].removeprefix("inst-"))
        if not np.array_equal(np.load(destination / file).reshape(-1), owners == owner):
            raise ValueError("official exported masks differ")
        actual[owner] = {"label": int(label), "rank": rank, "area": int((owners == owner).sum())}
    if actual != expected:
        raise ValueError("official current-class export differs")
    result = {"status": "COMPLETE", "scene": scene, "method": method, "exact_masks_labels_serialized_ranks": True,
              "official_manifest": str(official), "evaluation_receipt": row["evaluation_receipt"], "instances": len(actual)}
    write_once(destination / "receipt.json", result)
    return result


def evaluate_pools(binding, dataset="Replica"):
    root = Path(binding["output_root"])
    spec = read_json(binding["spec"])
    scenes = spec["datasets"]["official_replica_pool_order" if dataset == "Replica" else "calibration"]
    config = read_json(binding["scenes"][scenes[0]]["config"])
    namespace = load_released_module(Path(config["runtime"]["upstream"]) / "scripts/eval_utils.py")
    namespace["init"]("Replica" if dataset == "Replica" else "Scannet200")
    rows = [read_json(p) for scene in scenes for p in sorted((root / "rows" / scene).glob("*/*.json"))]
    results = []
    for method in binding["methods"]:
        for rank in ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS"):
            value = pool(root, rows, namespace, scenes, method, rank)
            # An identical prediction set may share evaluator work, not method provenance.
            result = {**value, "method": method, "rank_mode": rank}
            write_once(root / "pooled" / dataset / method / (rank + ".json"), result)
            results.append(result)
    return results


def validate_pool_path(binding, dataset="ScanNet"):
    root = Path(binding["output_root"])
    spec = read_json(binding["spec"])
    scenes = spec["datasets"]["calibration" if dataset == "ScanNet" else "official_replica_pool_order"]
    config = read_json(binding["scenes"][scenes[0]]["config"])
    namespace = load_released_module(Path(config["runtime"]["upstream"]) / "scripts/eval_utils.py")
    namespace["init"]("Scannet200" if dataset == "ScanNet" else "Replica")
    pooled = read_json(root / "pooled" / dataset / "N0/OFFICIAL_CURRENT_CLASS.json")
    matches = {}
    for scene in scenes:
        row = read_json(root / "rows" / scene / "N0/OFFICIAL_CURRENT_CLASS.json")
        with gzip.open(Path(row["evaluation_receipt"]).parent / "matches.json.gz", "rt") as handle:
            matches.update(json.load(handle))
    ap, trace = trace_released_matches(namespace, matches)
    values = plain(namespace["compute_averages"](ap))
    for name, key in (("uap", "all_ap"), ("ap25", "all_ap_25%"), ("ap50", "all_ap_50%")):
        if values[key] != pooled["metrics"][name]:
            raise ValueError("pooled original evaluate path and merged released trace differ")
    result = {"status": "COMPLETE", "dataset": dataset, "scenes": scenes, "method": "N0", "rank_mode": "OFFICIAL_CURRENT_CLASS",
              "original_evaluate_vs_combined_released_matches": "EXACT", "trace_parity": trace["parity"], "pool_identity": pooled["identity"]}
    write_once(root / "validation" / (dataset + "_pool.json"), result)
    return result
