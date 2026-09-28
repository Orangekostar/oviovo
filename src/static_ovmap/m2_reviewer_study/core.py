"""Core predictions are locked before annotation-side evaluation."""

from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.composition_study.object_evidence import owner_labels
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.scannet_study import (
    relabel_prediction,
    save_prediction,
)

from .evaluation import SceneEvaluator, write_gzip
from .fusion import fuse
from .scores import legacy_fusion_parity, read_scene, write_reconstruction_audit


def evaluate_core_scene(binding, scene):
    spec = read_json(binding["spec"])
    output = Path(binding["output_root"])
    evidence = read_scene(binding, scene)
    write_reconstruction_audit(evidence, output)
    write_once(output / "legacy_parity" / (scene + ".json"), legacy_fusion_parity(binding, evidence))
    if evidence.dataset != "ScanNet":
        # Nomination must exist before opening new transfer evaluations.
        nomination = read_json(output / "nomination.json")
        if nomination["status"] not in ("NOMINATED", "NO_EFFECTIVE_INTERVENTION"):
            raise ValueError("Replica evaluation requires frozen CAL nomination")
        original = read_json(binding["transfer"])["temperatures"]
        fitted = read_json(output / "calibration/new_final.json")
    else:
        fold = read_json(Path(binding["composition_root"]) / "calibration/folds.json")["folds"][scene]
        original = {name: fit["temperature"] for name, fit in fold.items()}
        fitted = read_json(output / "calibration/new_folds.json")["folds"][scene]
    native_labels = owner_labels(evidence.native)
    ids = evidence.config["models"]["native"]["valid_ids"]
    definitions = {row["method_id"]: row for row in spec["ablations"]}
    locked = {}
    for method in binding["methods"]:
        if method in spec["controls"]:
            row = read_json(Path(binding["scenes"][scene]["legacy_rows"]) / (method + ".json"))
            if method in evidence.sources:
                labels = {int(k): v["label"] for k, v in evidence.sources[method]["objects"].items()}
                details = None
            else:
                ts = {k: .07 for k in original} if method.endswith("RAW") else original
                labels, details = fuse(native_labels, evidence.sources, ids, ("N0", "Q_GAIN", "S_SIGLIP2_AREA"), ts)
            locked[method] = {"labels": labels, "prediction_identity": row["prediction_key"], "prediction_manifest": row["prediction_manifest"], "details": details}
            continue
        definition = definitions[method]
        sources = dict(evidence.sources)
        temperatures = dict(original)
        mode = definition["temperature_mode"]
        if mode == "constant_0.01":
            temperatures = {k: .01 for k in original}
        elif mode == "shared_refit":
            temperatures = {k: fitted["shared"]["temperature"] for k in original}
        elif mode == "cosine_per_source_refit":
            temperatures["N0"] = fitted["cosine_N0"]["temperature"]
        if definition["native_score_mode"] == "native_cosine":
            sources["N0"] = evidence.cosine_native
        labels, details = fuse(native_labels, sources, ids, definition["sources"], temperatures,
                               hard=definition["fusion"] == "hard_vote", empty_class0=definition["empty_source_action"] == "class_0")
        metadata = {"study_binding": binding["identity"], "definition": definition, "temperatures": temperatures,
                    "source_identities": {k: v.get("identity", canonical_digest(v)) for k, v in sources.items()}}
        payload = relabel_prediction(evidence.native, method, "S", labels, {}, metadata)
        manifest = save_prediction(payload, output / "predictions" / scene / method)
        locked[method] = {"labels": labels, "prediction_identity": payload.prediction_key, "prediction_manifest": str(manifest), "details": details}
    # All 13 predictions for this scene exist before loading GT via evaluator.
    write_once(output / "predictions" / scene / "locked.json", {m: {k: v for k, v in r.items() if k != "details"} for m, r in locked.items()})
    write_gzip(output / "probabilities" / (scene + ".json.gz"), {m: r["details"] for m, r in locked.items()})
    evaluator = SceneEvaluator(evidence, output)
    results = []
    for method, row in locked.items():
        for rank in ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS"):
            result = evaluator.evaluate(row["labels"], method, rank, row["prediction_identity"])
            if method in spec["controls"] and rank == "FROZEN_N0":
                old = read_json(Path(binding["scenes"][scene]["legacy_rows"]) / (method + ".json"))
                differences = {k: [old["metrics"][k], result["metrics"][k]] for k in ("uap", "ap50", "ap25", "miou", "macc")
                               if abs(old["metrics"][k] - result["metrics"][k]) > 1e-10}
                if differences:
                    write_once(output / "legacy_discrepancies" / f"{scene}_{method}.json", differences)
                    raise ValueError(f"legacy metrics differ: {scene}/{method}: {differences}")
            results.append(result)
            print(f"core {scene} {method} {rank}: {result['metrics']}", flush=True)
    return results
