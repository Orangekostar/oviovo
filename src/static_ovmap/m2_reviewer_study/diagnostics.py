"""Evaluation-only object populations, calibration scores and scene bootstrap."""

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import softmax

from src.static_ovmap.composition_study.evaluation import object_correspondence
from src.static_ovmap.composition_study.io import read_json, write_once

from .evaluation import write_gzip
from .scores import read_scene

SOURCES = ("N0", "Q_GAIN", "S_SIGLIP2_AREA")


def probability_metrics(probabilities, targets, predictions=None):
    p, y = np.asarray(probabilities, np.float64), np.asarray(targets, np.int64)
    if len(y) == 0:
        return {"objects": 0, "nll": None, "brier": None, "ece15": None, "accuracy": None, "bins": []}
    if p.ndim != 2 or len(p) != len(y) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1):
        raise ValueError("probabilities must be finite normalized rows")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("target column outside vocabulary")
    labels = p.argmax(1) if predictions is None else np.asarray(predictions, np.int64)
    if labels.shape != y.shape or np.any(labels < 0) or np.any(labels >= p.shape[1]):
        raise ValueError("declared prediction column outside vocabulary")
    confidence = p[np.arange(len(y)), labels]
    if not np.array_equal(confidence, p.max(1)):
        raise ValueError("declared prediction is not a maximum-probability label")
    correct = labels == y
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1
    membership = np.minimum((confidence * 15).astype(int), 14)
    bins, ece = [], 0.
    for i in range(15):
        mask = membership == i
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        conf = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy - conf)
        bins.append({"lower": i / 15, "upper": (i + 1) / 15, "count": count,
                     "accuracy": accuracy, "confidence": conf})
    return {"objects": len(y), "nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
            "brier": float(np.square(p - onehot).sum(1).mean()), "ece15": float(ece),
            "accuracy": float(correct.mean()), "bins": bins}


def bootstrap_delta(deltas):
    values = np.asarray(deltas, np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap requires defined paired scene deltas")
    rng = np.random.default_rng(17)
    draws = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "scenes": len(values), "draws": 2000, "seed": 17, "unit": "scene", "aggregation": "SCENE_MACRO",
            "pooled_ci": False, "deltas": values.tolist()}


def _rank(probabilities, target):
    return np.argsort(-np.asarray(probabilities), kind="stable").tolist().index(target) + 1


def diagnose_scene(binding, scene):
    root = Path(binding["output_root"])
    evidence = read_scene(binding, scene)
    ids = evidence.config["models"]["native"]["valid_ids"]
    locked = read_json(root / "predictions" / scene / "locked.json")
    with gzip.open(root / "probabilities" / (scene + ".json.gz"), "rt") as handle:
        distributions = json.load(handle)
    if evidence.dataset == "ScanNet":
        fitted = read_json(Path(binding["composition_root"]) / "calibration/folds.json")["folds"][scene]
        temperatures = {name: fitted[name]["temperature"] for name in SOURCES}
    else:
        temperatures = read_json(binding["transfer"])["temperatures"]
    # Single-source probabilities are derived from actual scores, never from labels.
    for name in SOURCES:
        for suffix, temperature in (("", temperatures[name]), ("_T007", .07)):
            distributions[name + suffix] = {str(owner): {
                "probabilities": softmax(np.asarray(row["scores"], np.float64) / temperature).tolist() if row["available"] else None,
                "label": row["label"], "available_sources": [name] if row["available"] else []}
                for owner, row in evidence.sources[name]["objects"].items()}
    correspondences = object_correspondence(evidence.config, scene, evidence.native)
    records, categories = [], {method: Counter() for method in distributions}
    populations = {method: {"paired": [], "union": []} for method in distributions}
    qualitative = {method: {"corrected": [], "harmed": []} for method in distributions}
    for match in sorted(correspondences, key=lambda row: row["owner_id"]):
        owner = str(match["owner_id"])
        source_rows = {name: evidence.sources[name]["objects"][owner] for name in SOURCES}
        available = [name for name in SOURCES if source_rows[name]["available"]]
        identifiable = match["correspondence"] == "unique" and match["geometry_iou"] > .5 and match["gt_label"] in ids
        target = ids.index(match["gt_label"]) if identifiable else None
        all_three = identifiable and len(available) == 3
        native_label = locked["N0"]["labels"][owner]
        correct = {name: row["label"] == match["gt_label"] if identifiable and row["available"] else None
                   for name, row in source_rows.items()}
        native_used = set(source_rows["N0"].get("used_request_ids", []))
        query_used = set(source_rows["Q_GAIN"].get("used_request_ids", []))
        intersection = len(native_used & query_used)
        methods = {}
        for method, details in distributions.items():
            row = details[owner]
            p = row["probabilities"]
            label = locked[method]["labels"][owner] if method in locked else row["label"]
            if p is not None and (label not in ids or p[ids.index(label)] != max(p)):
                raise ValueError(f"diagnostic probabilities disagree with locked labels: {scene}/{method}/{owner}")
            category = "UNIDENTIFIABLE"
            if identifiable:
                before, after = native_label == match["gt_label"], label == match["gt_label"]
                category = {(False, True): "CORRECTED", (True, False): "HARMED",
                            (True, True): "PRESERVED_CORRECT", (False, False): "STILL_WRONG"}[before, after]
                categories[method][category] += 1
                categories[method][f"AVAILABLE_SOURCES_{len(available)}"] += 1
                if label == 0:
                    categories[method]["TRUE_ABSTENTION"] += 1
                if p is not None:
                    populations[method]["union"].append((match["owner_id"], p, target, ids.index(label)))
                    if all_three:
                        populations[method]["paired"].append((match["owner_id"], p, target, ids.index(label)))
                if category in ("CORRECTED", "HARMED"):
                    key = category.lower()
                    if len(qualitative[method][key]) < 2:
                        qualitative[method][key].append(match["owner_id"])
                if available:
                    categories[method]["SOURCE_TOP1_HEADROOM_DENOMINATOR"] += 1
                    categories[method]["SOFT_RESCUE"] += int(after and all(correct[n] is False for n in available))
                    categories[method]["UNUSED_CORRECT_SOURCE"] += int(not after and any(correct[n] is True for n in available))
            sorted_p = sorted(p, reverse=True) if p is not None else []
            methods[method] = {"label": label, "changed": label != native_label, "probabilities": p,
                               "gt_rank": _rank(p, target) if identifiable and p is not None else None,
                               "top2_margin": sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else None,
                               "outcome": category, "available_sources": row["available_sources"]}
        records.append({**match, "identifiable": identifiable, "paired_population": all_three,
                        "source_evidence": source_rows, "source_correct": correct, "methods": methods,
                        "N0_Q_used_request_overlap": {"intersection": intersection,
                            "jaccard": intersection / len(native_used | query_used) if native_used | query_used else None,
                            "overlap_coefficient": intersection / min(len(native_used), len(query_used)) if native_used and query_used else None}})
    paired_ids = [row["owner_id"] for row in records if row["paired_population"]]
    identifiable_count = sum(row["identifiable"] for row in records)
    probability = {}
    for method, groups in populations.items():
        probability[method] = {}
        for group, rows in groups.items():
            if group == "paired" and [r[0] for r in rows] != paired_ids:
                raise ValueError("primary probability population differs across methods")
            probability[method][group] = {**probability_metrics([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows]),
                                          "owner_ids": [r[0] for r in rows], "identifiable_denominator": identifiable_count,
                                          "coverage": len(rows) / identifiable_count if identifiable_count else None}
    summary = {"status": "OBJECT_AND_PROBABILITY_COMPLETE", "scene": scene, "dataset": evidence.dataset,
               "owners": len(records), "identifiable": identifiable_count, "paired_owner_ids": paired_ids,
               "correspondence_counts": dict(Counter(row["correspondence"] for row in records)),
               "outcomes": {k: dict(v) for k, v in categories.items()}, "probability": probability,
               "qualitative_first_sorted_owners": qualitative,
               "AP_attribution": "SEPARATE_RELEASED_TRACE_REQUIRED", "temperatures": temperatures,
               "probability_scope": "conditional identifiable objects; not all false positives"}
    write_gzip(root / "diagnostics" / scene / "objects.json.gz", records)
    write_once(root / "diagnostics" / scene / "summary.json", summary)
    return summary


def core_macro_bootstrap(binding):
    root = Path(binding["output_root"])
    scenes = read_json(binding["spec"])["datasets"]["official_replica_pool_order"]
    results = {}
    for rank in ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS"):
        for method in binding["methods"]:
            rows = [read_json(root / "rows" / scene / method / (rank + ".json"))["metrics"] for scene in scenes]
            native = [read_json(root / "rows" / scene / "N0" / (rank + ".json"))["metrics"] for scene in scenes]
            results[f"{method}/{rank}"] = {"scene_order": scenes, "metrics": {
                key: bootstrap_delta([a[key] - b[key] for a, b in zip(rows, native, strict=True)])
                for key in ("uap", "ap25", "ap50", "miou", "macc")}}
    write_once(root / "diagnostics/Replica_macro_bootstrap.json", results)
    return results
