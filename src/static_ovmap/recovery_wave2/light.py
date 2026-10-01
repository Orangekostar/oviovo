"""Locked light-condition exports on the baseline's unchanged source geometry."""

import argparse
from pathlib import Path
import time

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction, save_prediction

from .binding import ConsumptionIndex, PathResolver, read
from .export import expanded_prediction
from .weight_readout import weight_decisions


WEIGHTS = {"RW_B_D2": .5, "RW_B_FCEQ": 1 / 3, "RW_W040": .4, "RW_W045": .45}
RECOVERIES = {"RW_U1_NATIVE_SINGLE": "U1", "RW_UQ_PAID_QUERY": "UQ",
              "RW_U2_FC_MATCHED_SINGLE": "U2", "RW_U3_FC_CAPTURED": "U3"}
METHODS = [*WEIGHTS, *RECOVERIES]


def owner_labels(prediction):
    owners, positions = np.unique(prediction.owner_ids, return_index=True)
    return {int(owner): int(prediction.semantic_labels[position])
            for owner, position in zip(owners, positions, strict=True) if owner > 0}


def require_transfer_freeze(binding, scene):
    if scene not in binding["datasets"]["replica"]:
        return
    import subprocess

    path = Path(binding["output_root"]) / "freeze/receipt.json"
    freeze = read(path)
    if freeze["status"] != "FROZEN_COMMITTED":
        raise ValueError("Replica predictions require the committed development freeze")
    subprocess.run(["git", "merge-base", "--is-ancestor", freeze["commit"], "HEAD"],
                   cwd=binding["repository_root"], check=True)
    committed = subprocess.check_output(["git", "show", freeze["commit"] + ":" + freeze["repository_file"]],
                                       cwd=binding["repository_root"])
    if canonical_digest(__import__("json").loads(committed)) != freeze["selection_identity"]:
        raise ValueError("Replica transfer selection differs from its committed freeze")


def build_scene(binding, scene, *, map_id="BB00_NATIVE", context=None, extra_conditions=None,
                conditions_override=None, output_subdir=None):
    require_transfer_freeze(binding, scene)
    data = context or binding["scenes"][scene]
    root = Path(binding["output_root"]) / "light" / scene / map_id
    if output_subdir is not None:
        if Path(output_subdir).is_absolute() or ".." in Path(output_subdir).parts:
            raise ValueError("additional condition outputs must stay within their own map")
        root = root / output_subdir
    source_root = Path(binding["output_root"]) / "recovery" / scene / map_id
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    index = ConsumptionIndex(Path(binding["output_root"]) / "validation/input_verifications.json")
    resolver = PathResolver(binding["path_map"])
    documents = {}
    for name in ("registry", "captured_requests", "cached_sources", "fc_request_plan",
                 "cached_source_receipt", "fc_recovery_receipt"):
        path = source_root / (name + ".json")
        index.identity(path)
        documents[name] = read(path)
    registry = documents["registry"]
    restored, fc = documents["cached_sources"], documents["fc_recovery_receipt"]
    if fc["status"] != "COMPLETE":
        raise ValueError("light exports require completed FC recovery evidence")
    for row in documents["cached_source_receipt"]["outputs"] + fc["outputs"]:
        index.identity(row["path"], row)
    sources = dict(restored["sources"])
    sources.update({arm: read(row["path"])["objects"] for arm, row in fc["sources"].items()})
    config = resolver.rewrite(read(data["config"]))
    ids = config["models"]["native"]["valid_ids"]
    index.identity(data["predictions"]["D2"])
    baseline = load_prediction(data["predictions"]["D2"])
    index.identity(data["predictions"]["NATIVE_READOUT"])
    incumbent = owner_labels(load_prediction(data["predictions"]["NATIVE_READOUT"]))
    if set(incumbent) != set(owner_labels(baseline)):
        raise ValueError("native and D2 incumbent partitions differ")
    capture = read(data["capture_manifest"])
    surface_path = Path(data["capture_manifest"]).parent / capture["surface"]["path"]
    index.identity(surface_path, capture["surface"])
    with np.load(surface_path, allow_pickle=False) as arrays:
        raw = arrays["original_owner"]
    projection_path = Path(config["scenes"][scene]["projection"])
    projection = read(projection_path)
    index.identity(projection_path)
    index.identity(projection_path.with_name("projection.npz"), projection)
    with np.load(projection_path.with_name("projection.npz"), allow_pickle=False) as arrays:
        nearest, matched = arrays["nearest"], arrays["matched"]
    decisions = {}
    for method in ("FC_EQ", "D2"):
        path = Path(data["parent_readout_receipt"]).parent / "decisions" / (method + ".json")
        index.identity(path)
        decisions[method] = read(path)
    conditions = {method: {"gamma": gamma, "recovery": None} for method, gamma in WEIGHTS.items()}
    conditions.update({method: {"gamma": .5, "recovery": arm} for method, arm in RECOVERIES.items()})
    conditions.update(extra_conditions or {})
    if conditions_override is not None:
        conditions = dict(conditions_override)
    locked, decision_outputs, partition_aliases = {}, {}, {}
    for method, recipe in conditions.items():
        labels, audit = weight_decisions(decisions["FC_EQ"], decisions["D2"], recipe["gamma"], ids, incumbent)
        arm = recipe["recovery"]
        recovered = {} if arm is None else {int(owner): int(row["label"])
            for owner, row in sources[arm].items() if row["available"]}
        costs = {"recovery_logical_FC_views": 0 if arm not in ("U2", "U3") else
                 sum(len(row.get("attempted_request_ids", [])) for row in sources[arm].values())}
        payload = expanded_prediction(baseline, raw, registry, recovered, ids, nearest, matched,
            method, costs, incumbent_labels=None if recipe["gamma"] == .5 else labels,
            metadata={"gamma": recipe["gamma"], "recovery": arm, "request_plan_identity": documents["fc_request_plan"]["identity"]})
        payload_path = save_prediction(payload, root / "predictions" / method)
        output = {"method": method, "recipe": recipe, "prediction_manifest": str(payload_path),
            "prediction_key": payload.prediction_key, "record_key": payload.record_key,
            "owner_partition_sha256": _array_digest(payload.owner_ids),
            "semantic_sha256": _array_digest(payload.semantic_labels),
            "incumbent_labels": labels, "recovered_labels": recovered,
            "new_positive_owners": len(recovered), "source_available": len(recovered),
            "baseline_old_support_preserved": bool(np.array_equal(payload.owner_ids[baseline.owner_ids > 0],
                                                                    baseline.owner_ids[baseline.owner_ids > 0])),
            "baseline_old_classes_preserved": bool(np.array_equal(payload.semantic_labels[baseline.owner_ids > 0],
                                                                     baseline.semantic_labels[baseline.owner_ids > 0]))}
        for earlier, previous in locked.items():
            if output["owner_partition_sha256"] == previous["owner_partition_sha256"]:
                output["owner_partition_alias"] = earlier
                if payload.prediction_key == previous["prediction_key"]:
                    output["prediction_alias"] = earlier
                break
        partition_aliases.setdefault(output["owner_partition_sha256"], method)
        locked[method] = output
        audit_path = root / "decisions" / (method + ".json")
        atomic_write_json(audit_path, {"method": method, "recipe": recipe, "old_owners": audit,
            "recovery": {} if arm is None else sources[arm], "GT_input": False})
        decision_outputs[method] = index.identity(audit_path)
        del payload
    receipt = {"status": "PREDICTIONS_LOCKED", "scene": scene, "map_id": map_id,
        "conditions": locked, "sources": {"cached": str(source_root / "cached_sources.json"),
                                            "FC": fc["sources"]},
        "registry_identity": registry["identity"], "GT_input": False,
        "all_predictions_locked_before_diagnostics": True, "decisions": decision_outputs,
        "inputs": index.entries(), "elapsed_seconds": time.monotonic() - started}
    receipt["identity"] = canonical_digest({k: v for k, v in receipt.items() if k not in ("elapsed_seconds", "inputs")})
    existing = root / "receipt.json"
    if existing.is_file() and read(existing)["identity"] != receipt["identity"]:
        raise ValueError("completed light exports changed")
    atomic_write_json(existing, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()
    result = build_scene(read(args.binding), args.scene)
    print(args.scene, result["status"], len(result["conditions"]), flush=True)
