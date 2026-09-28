"""Fixed-trajectory text stress evaluation in each encoder's own feature space."""

import gzip
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.composition_study.object_evidence import native_scores

from .binding import InputIndex
from .diagnostics import SOURCES, probability_metrics
from .evaluation import write_gzip
from .scores import read_scene


def stress_metrics(probabilities, targets, original_predictions, original_count, predictions=None):
    result = probability_metrics(probabilities, targets, predictions)
    if not targets:
        return {**result, "distractor_selection_rate": None, "original_to_original_denominator": 0,
                "original_to_original_changes": 0, "restricted_original_argmax_changes": 0}
    p = np.asarray(probabilities)
    prediction = p.argmax(1) if predictions is None else np.asarray(predictions)
    original_predictions = np.asarray(original_predictions)
    original_wins = prediction < original_count
    return {**result, "distractor_selection_rate": float((~original_wins).mean()),
            "original_to_original_denominator": int(original_wins.sum()),
            "original_to_original_changes": int(((prediction != original_predictions) & original_wins).sum()),
            "restricted_original_argmax_changes": int((p[:, :original_count].argmax(1) != original_predictions).sum())}


def evaluate_robustness_scene(binding, scene):
    started, index = time.monotonic(), InputIndex()
    if binding["scenes"][scene]["dataset"] != "Replica":
        raise ValueError("frozen word-set robustness protocol is Replica-only")
    evidence = read_scene(binding, scene, index)
    root = Path(binding["output_root"])
    aggregate_root = root / "robustness/aggregates" / scene
    aggregate_receipt = read_json(aggregate_root / "receipt.json")
    index.identity(aggregate_root / "receipt.json")
    for row in aggregate_receipt["outputs"]:
        index.identity(row["path"], row)
    with np.load(aggregate_root / "aggregates.npz", allow_pickle=False) as payload:
        aggregates = {name: payload[name] for name in payload.files}
    texts, text_receipts = {}, {}
    for name in ("native", "siglip2"):
        path = root / "robustness/text/Replica" / name / "receipt.json"
        receipt = read_json(path)
        index.identity(path)
        if receipt["model_identity"] != evidence.config["models"][name]["identity"]:
            raise ValueError("text/image model space mismatch")
        for row in receipt["outputs"]:
            index.identity(row["path"], row)
        with np.load(path.parent / "text.npz", allow_pickle=False) as payload:
            texts[name] = {key: payload[key] for key in payload.files}
        text_receipts[name] = receipt
    ids = evidence.config["models"]["native"]["valid_ids"]
    if any(row["valid_ids"] != ids for row in text_receipts.values()):
        raise ValueError("text columns differ across model spaces")
    selected = read_json(root / "nomination.json")["selected"]
    spec = read_json(binding["spec"])
    definition = next(row for row in spec["ablations"] if row["method_id"] == selected)
    methods = [*SOURCES, "CP_M2_EQUAL_RAW", "CP_M2_EQUAL_CAL", selected]
    original_t = read_json(binding["transfer"])["temperatures"]
    final_fit = read_json(root / "calibration/new_final.json")
    locked = read_json(root / "predictions" / scene / "locked.json")
    with gzip.open(root / "diagnostics" / scene / "objects.json.gz", "rt") as handle:
        matches = {str(row["owner_id"]): row for row in json.load(handle)}
    records, distributions, baseline_predictions = {}, {}, {}
    for variant in ("original", "photo", "closeup", "expanded"):
        records[variant], distributions[variant] = {}, {}
        count = len(text_receipts["native"]["texts"][variant])
        for owner in sorted(evidence.sources["N0"]["objects"], key=int):
            scores = {}
            for name in SOURCES:
                source = evidence.sources[name]["objects"][owner]
                if not source["available"]:
                    continue
                model = "siglip2" if name == "S_SIGLIP2_AREA" else "native"
                feature, text = aggregates[f"{name}_{owner}"], texts[model][variant]
                if name == "N0":
                    values = native_scores(feature, text, texts["native"]["canonical"])
                else:
                    text = np.asarray(text, np.float64)
                    text = text / np.linalg.norm(text, axis=1, keepdims=True)
                    values = text @ (feature / np.linalg.norm(feature))
                if variant == "original" and not np.allclose(values, source["scores"], atol=1e-12, rtol=0):
                    raise ValueError("original-vocabulary score reconstruction changed")
                scores[name] = values
            outputs = {}
            for method in methods:
                chosen = (method,) if method in SOURCES else definition["sources"] if method == selected else SOURCES
                temps = dict(original_t)
                if method == "CP_M2_EQUAL_RAW":
                    temps = {name: .07 for name in SOURCES}
                if method == selected and definition["temperature_mode"] == "constant_0.01":
                    temps = {name: .01 for name in SOURCES}
                if method == selected and definition["temperature_mode"] == "shared_refit":
                    temps = {name: final_fit["shared"]["temperature"] for name in SOURCES}
                active = [name for name in chosen if name in scores]
                p = None
                if active:
                    if method == selected and definition["fusion"] == "hard_vote":
                        votes = [int(scores[name].argmax()) for name in active]
                        p = np.bincount(votes, minlength=count) / len(votes)
                    else:
                        p = np.mean([softmax(scores[name] / temps[name]) for name in active], axis=0)
                        p /= p.sum()
                prediction = None if p is None else int(p.argmax())
                if p is not None and method == selected and definition["fusion"] == "hard_vote" and "N0" in active:
                    native_vote = int(scores["N0"].argmax())
                    if p[native_vote] == p.max():
                        prediction = native_vote
                if variant == "original" and prediction is not None:
                    if ids[prediction] != locked[method]["labels"][owner]:
                        raise ValueError("original robust readout does not reproduce frozen method")
                    baseline_predictions[method, owner] = prediction
                outputs[method] = {"probabilities": None if p is None else p.tolist(), "prediction_column": prediction,
                                   "prediction_text": None if prediction is None else text_receipts["native"]["texts"][variant][prediction],
                                   "distractor": prediction is not None and prediction >= len(ids), "available_sources": active,
                                   "temperatures": {name: temps[name] for name in active}}
            records[variant][owner] = {"type": "VOCABULARY_STRESS_EVIDENCE", "owner_id": int(owner),
                                       "source_scores": {name: value.tolist() for name, value in scores.items()}, "methods": outputs}
        for method in methods:
            distributions[variant][method] = {}
            for population in ("paired", "union"):
                rows = [(owner, row["methods"][method]) for owner, row in records[variant].items()
                        if matches[owner]["identifiable"] and row["methods"][method]["probabilities"] is not None
                        and (population == "union" or matches[owner]["paired_population"])]
                result = stress_metrics([row["probabilities"] for _, row in rows],
                    [ids.index(matches[owner]["gt_label"]) for owner, _ in rows],
                    [baseline_predictions[method, owner] for owner, _ in rows], len(ids),
                    [row["prediction_column"] for _, row in rows])
                denominator = sum(row["identifiable"] for row in matches.values())
                distributions[variant][method][population] = {**result, "owner_ids": [int(owner) for owner, _ in rows],
                    "coverage_denominator": denominator, "coverage": len(rows) / denominator if denominator else None}
    output = root / "robustness/results" / scene
    write_gzip(output / "objects.json.gz", records)
    summary = {"status": "COMPLETE", "scene": scene, "dataset": "Replica", "methods": methods,
               "metrics": distributions, "valid_original_ids": ids,
               "text_receipts": {name: value["input_identity"] for name, value in text_receipts.items()},
               "expanded_vocabulary_official_AP": False, "query_replayed": False, "temperatures_refit": False,
               "physical_image_forwards": 0, "singleton_softmax": "identically1; not validated arbitrary-text retrieval",
               "inputs": index.entries()}
    write_once(output / "summary.json", summary)
    if not (output / "timing.json").exists():
        write_once(output / "timing.json", {"elapsed_seconds": time.monotonic() - started})
    return summary
