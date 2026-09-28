"""Separate rank views and released dataset pooling, without source mutation."""

import ast
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.evaluation import load_targets
from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.scannet_study import project_values
from src.static_ovmap.released_loader import load_released_module
from src.static_ovmap.released_trace import trace_released_matches

from .binding import InputIndex


def official_view(projected_owners, labels, minimum):
    unique, counts = np.unique(projected_owners, return_counts=True)
    areas = {int(o): int(n) for o, n in zip(unique, counts, strict=True) if o > 0 and n >= minimum}
    maxima = {}
    for owner, area in areas.items():
        label = labels[owner]
        maxima[label] = max(maxima.get(label, 0), area)
    return {owner: {"label": labels[owner], "rank": f"{area / maxima[labels[owner]]:.6f}", "area": area}
            for owner, area in areas.items() if labels[owner] != 0}


def confusion(gt, prediction, valid_ids):
    ids = (0, *[int(i) for i in valid_ids if i != 0])
    gt, prediction = np.asarray(gt), np.asarray(prediction)
    keep = np.isin(gt, ids[1:])
    g, p = np.zeros(gt.shape, np.int64), np.zeros(gt.shape, np.int64)
    for index, label in enumerate(ids):
        g[gt == label] = index
        p[prediction == label] = index
    result = np.zeros((len(ids), len(ids)), np.int64)
    np.add.at(result, (g[keep], p[keep]), 1)
    return result


def semantic_summary(matrix):
    matrix = np.asarray(matrix)
    counts = matrix.sum(1)
    selected = np.flatnonzero(counts[1:]) + 1
    if not len(selected):
        return {"miou": None, "macc": None, "defined_classes": 0}
    tp = matrix.diagonal()[selected]
    return {"miou": float(np.mean(tp / (counts[selected] + matrix.sum(0)[selected] - tp))),
            "macc": float(np.mean(tp / counts[selected])), "defined_classes": len(selected)}


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_gzip(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        json.dump(plain(value), handle, sort_keys=True, allow_nan=False)


def write_manifest(path, masks, view):
    path = Path(path)
    lines = [f'{os.path.relpath(masks[o], path.parent)} {r["label"]} {r["rank"]}' for o, r in view.items()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


class SceneEvaluator:
    def __init__(self, evidence, root, index=None):
        self.evidence, self.root = evidence, Path(root)
        self.index = index or InputIndex()
        self.targets = load_targets(evidence.config, evidence.scene, evidence.native)
        self.owners = project_values(evidence.native.owner_ids, self.targets["nearest"], self.targets["matched"])
        self.ids = evidence.config["models"]["native"]["valid_ids"]
        upstream = Path(evidence.config["runtime"]["upstream"])
        self.namespace = load_released_module(upstream / "scripts/eval_utils.py")
        self.namespace["init"]("Replica" if evidence.dataset == "Replica" else "Scannet200")
        export_source = upstream / "scripts/eval_sem_seg.py"
        fn = next(n for n in ast.parse(export_source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "map_pred_mesh")
        self.minimum = next(ast.literal_eval(n.value) for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "min_region_sizes" for t in n.targets))
        data = evidence.config["scenes"][evidence.scene]
        self.context = {"dataset": evidence.dataset, "native_geometry": data["native_geometry"],
                        "native_prediction_key": evidence.native.prediction_key,
                        "study_evaluator": self.index.identity(__file__),
                        "projection": self.index.identity(data["projection"]),
                        "projection_arrays": self.index.identity(Path(data["projection"]).with_name("projection.npz")),
                        "gt": self.index.identity(self.targets["gt_instance_path"]),
                        "annotation": self.index.identity(data["annotations"]),
                        "evaluator": self.index.identity(upstream / "scripts/eval_utils.py"),
                        "exporter": self.index.identity(export_source), "valid_ids": self.ids,
                        "class_names": evidence.config["models"]["native"].get("class_names"),
                        "runtime_overlaps": self.namespace["overlaps"].tolist(), "minimum": self.minimum}
        self.mask_root = self.root / "shared_masks" / evidence.scene
        self.mask_root.mkdir(parents=True, exist_ok=True)
        self.masks = {}
        for owner in sorted(map(int, evidence.sources["N0"]["objects"])):
            path = self.mask_root / f"owner_{owner}.npy"
            mask = self.owners == owner
            if path.exists():
                if not np.array_equal(np.load(path, allow_pickle=False), mask):
                    raise ValueError(f"projected mask differs: {path}")
            else:
                np.save(path, mask)
            self.masks[owner] = str(path)

    def evaluate(self, labels, method, rank_mode, prediction_identity):
        if rank_mode not in ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS"):
            raise ValueError("unknown evaluation ranking")
        if rank_mode == "OFFICIAL_CURRENT_CLASS":
            view = official_view(self.owners, labels, self.minimum)
        else:
            ranks = dict(self.evidence.native.instance_ranks)
            view = {o: {"label": labels[o], "rank": f"{ranks[o]:.6f}"} for o in sorted(labels)}
        identity = canonical_digest({"context": self.context, "labels": labels, "view": view, "rank_mode": rank_mode})
        output = self.root / "evaluation_cache" / identity
        receipt_path = output / "receipt.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
        else:
            output.mkdir(parents=True, exist_ok=True)
            manifest = output / "pred_inst_sem_mapping.txt"
            write_manifest(manifest, self.masks, view)
            start = time.perf_counter()
            gt_path = self.targets["gt_instance_path"]
            gt2pred, pred2gt = self.namespace["assign_instances_for_scan"](str(self.root), str(manifest), str(gt_path))
            matches = {str(Path(gt_path).resolve()): {"gt": gt2pred, "pred": pred2gt}}
            ap, trace = trace_released_matches(self.namespace, matches)
            averages = plain(self.namespace["compute_averages"](ap))
            predicted = np.zeros(self.owners.shape, np.int64)
            for owner, label in labels.items():
                predicted[self.owners == owner] = label
            matrix = confusion(self.targets["gt_semantic"], predicted, self.ids)
            write_gzip(output / "matches.json.gz", matches)
            write_gzip(output / "trace.json.gz", trace)
            metrics = {"uap": averages["all_ap"], "apall": averages["all_ap"], "ap50": averages["all_ap_50%"], "ap25": averages["all_ap_25%"], **semantic_summary(matrix)}
            receipt = {"status": "COMPLETE", "identity": identity, "metrics": metrics, "context": self.context,
                       "rank_mode": rank_mode, "confusion": matrix.tolist(), "view": view,
                       "manifest": str(manifest), "gt_path": str(gt_path), "trace_parity": trace["parity"],
                       "elapsed_seconds": time.perf_counter() - start}
            write_once(receipt_path, receipt)
        result = {"status": "COMPLETE", "scene": self.evidence.scene, "dataset": self.evidence.dataset,
                  "method": method, "rank_mode": rank_mode, "aggregation": "SCENE", "metrics": receipt["metrics"],
                  "prediction_identity": prediction_identity, "evaluation_receipt": str(receipt_path), "evaluation_identity": identity}
        write_once(self.root / "rows" / self.evidence.scene / method / (rank_mode + ".json"), result)
        return result


def pool(root, rows, namespace, scene_order, method, rank_mode):
    root = Path(root)
    by_scene = {r["scene"]: r for r in rows if r["method"] == method and r["rank_mode"] == rank_mode}
    if set(by_scene) != set(scene_order):
        raise ValueError("pool requires exact complete scene set")
    receipts = [read_json(by_scene[s]["evaluation_receipt"]) for s in scene_order]
    identity = canonical_digest({"ordered_inputs": [r["identity"] for r in receipts], "rank_mode": rank_mode})
    output = root / "pooled_cache" / identity
    output.mkdir(parents=True, exist_ok=True)
    path = output / "receipt.json"
    if path.exists():
        return read_json(path)
    start = time.perf_counter()
    # One unchanged released evaluate call over all ordered scene files.
    values = plain(namespace["evaluate"](str(root), [r["manifest"] for r in receipts], [r["gt_path"] for r in receipts], str(output)))
    matrix = np.sum([np.asarray(r["confusion"], np.int64) for r in receipts], axis=0)
    result = {"status": "COMPLETE", "identity": identity, "dataset": by_scene[scene_order[0]]["dataset"],
              "method": method, "rank_mode": rank_mode, "aggregation": "RELEASED_DATASET_POOL", "scene_order": list(scene_order),
              "metrics": {"uap": values["all_ap"], "apall": values["all_ap"], "ap50": values["all_ap_50%"], "ap25": values["all_ap_25%"], **semantic_summary(matrix)},
              "ordered_inputs": [r["identity"] for r in receipts], "elapsed_seconds": time.perf_counter() - start}
    write_once(path, result)
    return result
