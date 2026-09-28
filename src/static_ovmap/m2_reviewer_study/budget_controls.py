"""CAL-only replacement-query fits, dual-rank evaluation and frozen curve gate."""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.composition_study.object_evidence import owner_labels
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.scannet_study import (
    relabel_prediction,
    save_prediction,
)

from .binding import InputIndex
from .calibration import fit_shared
from .evaluation import SceneEvaluator, write_gzip
from .fusion import fuse
from .scores import read_scene

POLICIES = ("Q_GAIN", "Q_COMBINE", "RV_Q_RANDOM_s17", "RV_Q_RANDOM_s23", "RV_Q_RANDOM_s41")
CAL = ("scene0056_00", "scene0534_00")


def curve_decision(gain, combine, random):
    if len(random) != 3:
        raise ValueError("curve gate requires all three random replicates")
    mean = {key: float(np.mean([row[key] for row in random])) for key in ("uap", "miou")}
    triggered = all(gain["uap"] > row["uap"] and gain["miou"] >= row["miou"] - 1e-10
                    for row in (combine, mean))
    return {"triggered": triggered, "reason": "CAL_GAIN_DOMINATES" if triggered else "NOT_TRIGGERED_CAL_GAIN_DOES_NOT_DOMINATE",
            "gain": gain, "combine": combine, "random_seed_mean": mean, "random_replicates": random,
            "miou_tolerance": 1e-10, "ap_comparison": "STRICT"}


def source_for(root, scene, policy, budget=200, index=None):
    index = index or InputIndex()
    receipt_path = root / "query_controls" / scene / f"{policy}_B{budget}/receipt.json"
    index.identity(receipt_path)
    receipt = read_json(receipt_path)
    if receipt["status"] != "COMPLETE":
        raise ValueError("query acquisition incomplete")
    index.expected_output(receipt["evidence_path"], receipt)
    source = read_json(receipt["evidence_path"])
    if canonical_digest({k: v for k, v in source.items() if k != "identity"}) != source["identity"]:
        raise ValueError("query source identity changed")
    return receipt, source


def fit_query_temperatures(binding):
    root, index = Path(binding["output_root"]), InputIndex()
    path = Path(binding["composition_root"]) / "calibration/examples.json"
    expected = next(row for row in binding["inputs"] if row["path"] == str(path))
    index.identity(path, expected)
    correspondences = read_json(path)["correspondences"]
    examples, ids = {}, None
    for policy in POLICIES[1:]:
        examples[policy] = []
        for scene in CAL:
            _, source = source_for(root, scene, policy, index=index)
            if ids is not None and ids != source["valid_ids"]:
                raise ValueError("replacement query vocabulary mismatch")
            ids = source["valid_ids"]
            for match in correspondences[scene]:
                row = source["objects"][str(match["owner_id"])]
                if match["correspondence"] == "unique" and match["geometry_iou"] > .5 and row["available"]:
                    examples[policy].append({"scene_id": scene, "owner_id": match["owner_id"],
                                             "gt_label": match["gt_label"], "scores": row["scores"]})
    identity = canonical_digest({"binding": binding["identity"], "inputs": index.entries(),
                                 "calibration_code": index.identity(Path(__file__).with_name("calibration.py")),
                                 "code": index.identity(__file__)})
    output = root / "query_controls/temperatures.json"
    if output.exists():
        fitted = read_json(output)
        if fitted["identity"] != identity:
            raise ValueError("replacement calibration inputs changed")
        return fitted
    started = time.monotonic()
    folds = {scene: {p: fit_shared({p: examples[p]}, [s for s in CAL if s != scene], ids)
                     for p in POLICIES[1:]} for scene in CAL}
    final = {p: fit_shared({p: examples[p]}, CAL, ids) for p in POLICIES[1:]}
    fitted = {"identity": identity, "status": "FITTED", "folds": folds, "final": final,
              "examples": examples, "inputs": index.entries(), "elapsed_seconds": time.monotonic() - started,
              "native_and_siglip2_temperatures": "UNCHANGED", "query_checkpoint_CAL_preexposure": True}
    write_once(output, fitted)
    return fitted


def evaluate_query_scene(binding, scene, *, budget=200):
    root = Path(binding["output_root"])
    if scene not in CAL:
        gate = read_json(root / "query_controls/curve_gate.json")
        if gate["status"] != "FROZEN" or gate["binding"] != binding["identity"]:
            raise ValueError("transfer controls require frozen CAL curve decision")
        if budget != 200 and not gate["triggered"]:
            raise ValueError("budget curves were not triggered")
    elif budget != 200:
        raise ValueError("CAL budget curves are not in the protocol")
    fitted = read_json(root / "query_controls/temperatures.json")
    evidence = read_scene(binding, scene)
    native_labels = owner_labels(evidence.native)
    ids = evidence.config["models"]["native"]["valid_ids"]
    if scene in CAL:
        original = {k: row["temperature"] for k, row in read_json(Path(binding["composition_root"]) / "calibration/folds.json")["folds"][scene].items()}
        temperatures = fitted["folds"][scene]
    else:
        original = read_json(binding["transfer"])["temperatures"]
        temperatures = fitted["final"]
    locked, probabilities = {}, {}
    for policy in POLICIES if budget == 200 else POLICIES[:2]:
        receipt, source = source_for(root, scene, policy, budget)
        if source["native_record_key"] != evidence.native.record_key or source["valid_ids"] != ids:
            raise ValueError("replacement source differs in geometry or vocabulary")
        method = f"{policy}_B{budget}"
        locked[method] = {"labels": {int(k): v["label"] for k, v in source["objects"].items()},
                          "prediction_identity": source["prediction_key"], "prediction_manifest": receipt["prediction_manifest"]}
        sources = {**evidence.sources, "Q_GAIN": source}
        for mode in ("RAW", "CAL"):
            ts = {k: .07 for k in original} if mode == "RAW" else dict(original)
            if mode == "CAL" and policy != "Q_GAIN":
                ts["Q_GAIN"] = temperatures[policy]["temperature"]
            labels, details = fuse(native_labels, sources, ids, ("N0", "Q_GAIN", "S_SIGLIP2_AREA"), ts)
            name = f"RV_B_{policy}_B{budget}_{mode}"
            prediction = relabel_prediction(evidence.native, name, "S", labels, receipt["required_logical"],
                                            {"study_binding": binding["identity"], "temperatures": ts,
                                             "query_source": source["identity"], "budget": budget})
            manifest = save_prediction(prediction, root / "predictions" / scene / name)
            locked[name] = {"labels": labels, "prediction_identity": prediction.prediction_key,
                            "prediction_manifest": str(manifest)}
            probabilities[name] = details
    write_once(root / "query_controls" / scene / f"locked_B{budget}.json", locked)
    write_gzip(root / "query_controls" / scene / f"probabilities_B{budget}.json.gz", probabilities)
    evaluator = SceneEvaluator(evidence, root)
    return [evaluator.evaluate(row["labels"], method, rank, row["prediction_identity"])
            for method, row in locked.items() for rank in ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS")]


def freeze_curve_gate(binding):
    root = Path(binding["output_root"])
    summaries, inputs = {}, []
    index = InputIndex()
    for policy in POLICIES:
        rows = []
        for scene in CAL:
            path = root / "rows" / scene / f"RV_B_{policy}_B200_CAL/FROZEN_N0.json"
            inputs.append(index.identity(path))
            rows.append(read_json(path)["metrics"])
        summaries[policy] = {k: float(np.mean([row[k] for row in rows])) for k in ("uap", "miou")}
    decision = {**curve_decision(summaries["Q_GAIN"], summaries["Q_COMBINE"], [summaries[p] for p in POLICIES[2:]]),
                "status": "FROZEN", "binding": binding["identity"], "inputs": inputs,
                "rank_mode": "FROZEN_N0", "scenes": list(CAL), "selection_uses_Replica": False}
    write_once(root / "query_controls/curve_gate.json", decision)
    return decision


def run_query_controls(binding, gpu="1"):
    """Each visual leaf owns a fresh process and releases its CUDA context."""
    root = Path(binding["output_root"])
    script = Path(__file__).resolve().parents[3] / "scripts/evaluation/run_ovimap_m2_reviewer_study.py"

    def acquire(scene, budget=200):
        for policy in POLICIES if budget == 200 else POLICIES[:2]:
            name, separator, seed = policy.partition("_s")
            command = [sys.executable, str(script), "--phase", "query-controls", "--output-root", str(root),
                       "--scene", scene, "--query-policy", name, "--budget", str(budget), "--gpu", str(gpu), "--resume"]
            if separator:
                command += ["--seed", seed]
            subprocess.run(command, check=True)

    for scene in CAL:
        acquire(scene)
    fit_query_temperatures(binding)
    for scene in CAL:
        evaluate_query_scene(binding, scene)
    gate = freeze_curve_gate(binding)
    scenes = read_json(binding["spec"])["datasets"]["official_replica_pool_order"]
    for scene in scenes:
        acquire(scene)
        evaluate_query_scene(binding, scene)
    if gate["triggered"]:
        for budget in (100, 400):
            for scene in scenes:
                acquire(scene, budget)
                evaluate_query_scene(binding, scene, budget=budget)
    receipts = [read_json(path) for scene in binding["scenes"]
                for path in (root / "query_controls" / scene).glob("*_B*/receipt.json")]
    b200 = sum(row["logical"]["attempts"] for row in receipts if row["method_id"].endswith("_B200"))
    curves = sum(row["logical"]["attempts"] for row in receipts if not row["method_id"].endswith("_B200"))
    if b200 > 10000 or curves > 8000:
        raise ValueError("study logical request ceiling exceeded")
    result = {"status": "ACQUISITION_AND_SCENE_EVALUATION_COMPLETE", "binding": binding["identity"],
              "B200_logical_requests": b200, "curve_logical_requests": curves, "curve_gate": gate,
              "query_pool_and_cost_reporting": "PENDING"}
    write_once(root / "query_controls/scene_stage.json", result)
    return result
