"""Recover per-class evidence using the exact frozen common-v2 projection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.evaluate_tesse_cd_common_v2 import _prediction_semantics
from scripts.evaluation.run_crove_adapter_room0 import path
from scripts.evaluation.run_ovimap_native import _atomic_json, _sha256
from scripts.evaluation.summarize_crove_multimethod_readouts import METHODS
from src.evaluation.baselines.dynamic_metrics import _semantic_miou
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk


def per_class_counts(ground_truth, prediction, valid_ids):
    domain = np.isin(ground_truth, list(valid_ids))
    gt, pred = ground_truth[domain], prediction[domain]
    result = {}
    for label in sorted(set(int(value) for value in gt)):
        positive, predicted = gt == label, pred == label
        intersection = int(np.count_nonzero(positive & predicted))
        union = int(np.count_nonzero(positive | predicted))
        result[str(label)] = {
            "iou": intersection / union if union else 1.0,
            "support": int(np.count_nonzero(positive)),
            "intersection": intersection,
            "union": union,
        }
    return result


def main():
    config = json.loads((ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text())
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    run, compact = path(config["run_root"]), path(config["compact_output_root"])
    crosswalk = load_tesse_semantic_crosswalk(
        path(pair["semantic_aliases"]), pair["scene"], path(pair["semantic_label_space"])
    )
    geometry = path(pair["current_map_root"]) / "current_surface.npz"
    context = run / "bridge/evaluation_context.npz"
    binding = {"geometry_sha256": _sha256(geometry), "context_sha256": _sha256(context)}
    target = compact / "apartment_per_class_audit.json"
    report = json.loads(target.read_text()) if target.exists() else {
        "binding": binding,
        "protocol": "exact legacy_eval_order; eval_role != 0; original common-v2 strict 0.05m nearest projection; GT-present valid classes only",
        "readouts": {},
    }
    if report["binding"] != binding:
        raise ValueError("per-class audit geometry or GT context changed")
    with np.load(geometry) as data:
        xyz = data["vertices_xyz"]
    with np.load(context) as data:
        targets = data["current_semantic_voxels"]
    for state in ("B3", "H2"):
        for source in sorted(compact.glob(f"apartment_{state}_*.json")):
            score = json.loads(source.read_text())
            if score.get("status") != "FULL_MAP_EVALUATED" or "metrics" not in score:
                continue
            method = score["method"]
            directory = METHODS[method][1]
            prediction = run / "dev/apartment" / directory / f"{state}_{method}.npz"
            key = f"{state}_{method}"
            hashes = {"result_sha256": _sha256(source), "prediction_sha256": _sha256(prediction)}
            if key in report["readouts"]:
                if any(report["readouts"][key][k] != v for k, v in hashes.items()):
                    raise ValueError(f"audited prediction or score changed: {key}")
                continue
            with np.load(prediction) as data:
                canonical, order = data["source_indices"], data["legacy_eval_order"]
                inverse = np.searchsorted(canonical, order)
                if not np.array_equal(canonical[inverse], order):
                    raise ValueError(f"invalid legacy source order: {key}")
                mask = data["eval_role"][inverse] != 0
                ids = data["semantic_ids"][inverse][mask]
            gt, pred = _prediction_semantics(xyz[order][mask], ids, targets)
            counts = per_class_counts(gt, pred, crosswalk.valid_semantic_ids)
            mean = float(np.mean([v["iou"] for v in counts.values()])) if counts else 1.0
            original = _semantic_miou(gt, pred, crosswalk.valid_semantic_ids)
            if not np.isclose(mean, original, atol=1e-12, rtol=0) or not np.isclose(
                mean, score["metrics"]["current_miou"], atol=1e-12, rtol=0
            ):
                raise ValueError(f"per-class reconstruction changed headline score: {key}")
            report["readouts"][key] = {
                **hashes, "miou": mean, "per_class": counts,
                "headline_reconstruction": "PASS", "target_count": len(gt),
            }
            _atomic_json(target, report)
            print(key, "per-class reconstruction PASS", flush=True)


if __name__ == "__main__":
    main()
