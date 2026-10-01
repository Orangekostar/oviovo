"""Expanded registry evaluation through the unchanged released-scoring adapter."""

import argparse
import contextlib
import copy
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from static_ovmap.m2_reviewer_study.evaluation import SceneEvaluator, official_view
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction

from .binding import ConsumptionIndex, PathResolver, read
from .light import owner_labels


def expanded_native_registry(native_source, labels):
    source = copy.deepcopy(native_source)
    for owner in labels:
        key = str(owner)
        if key not in source["objects"]:
            source["objects"][key] = {"owner_id": owner, "available": False, "scores": None,
                "label": None, "used_request_ids": [], "reason": "NOT_AVAILABLE_ORIGINAL_NATIVE_READOUT"}
    if set(map(int, source["objects"])) != set(labels):
        raise ValueError("native mask registry must cover exactly the complete exported owner universe")
    source.pop("identity", None)
    source["identity"] = canonical_digest(source)
    return source


def scorer_context(context):
    result = {key: value for key, value in context.items() if key != "native_prediction_key"}
    for key, value in result.items():
        if isinstance(value, dict) and "sha256" in value and "bytes" in value:
            result[key] = {"sha256": value["sha256"], "bytes": value["bytes"]}
    return result


def _parent_alias(evaluator, payload, data, parent_predictions, parent_rows, index):
    labels = owner_labels(payload)
    view = {str(k): v for k, v in official_view(evaluator.owners, labels, evaluator.minimum).items()}
    for method, parent in parent_predictions.items():
        if (payload.geometry != parent.geometry or not np.array_equal(payload.owner_ids, parent.owner_ids)
                or not np.array_equal(payload.semantic_labels, parent.semantic_labels)):
            continue
        row = next(r for r in parent_rows if r["method"] == method and r["rank_mode"] == "OFFICIAL_CURRENT_CLASS")
        index.identity(row["evaluation_receipt"])
        receipt = read(row["evaluation_receipt"])
        if (receipt["status"] != "COMPLETE" or receipt["view"] != view
                or scorer_context(receipt["context"]) != scorer_context(evaluator.context)):
            raise ValueError("parent prediction equality does not establish scorer-context equality")
        manifest = Path(receipt["manifest"])
        index.identity(manifest)
        index.identity(manifest.with_name("matches.json.gz"))
        index.identity(manifest.with_name("trace.json.gz"))
        return {**row, "reuse_kind": "EXACT_PARENT_PREDICTION_AND_SCORER_CONTEXT",
            "equality_proof": {"projected_owner_sha256": _array_digest(evaluator.owners),
                "source_owner_sha256": _array_digest(payload.owner_ids),
                "semantic_sha256": _array_digest(payload.semantic_labels),
                "official_view_identity": canonical_digest(view),
                "scorer_context_identity": canonical_digest(scorer_context(evaluator.context)),
                "parent_method": method, "parent_prediction_key": parent.prediction_key}}
    return None


def evaluate_scene(binding, scene, *, map_id="BB00_NATIVE", context=None):
    data = context or binding["scenes"][scene]
    root = Path(binding["output_root"]) / "light" / scene / map_id
    index = ConsumptionIndex(Path(binding["output_root"]) / "validation/input_verifications.json")
    index.identity(root / "receipt.json")
    lock = read(root / "receipt.json")
    if lock["status"] != "PREDICTIONS_LOCKED" or not lock["all_predictions_locked_before_diagnostics"]:
        raise ValueError("official evaluation requires every planned prediction to be locked")
    started = time.monotonic()
    resolver = PathResolver(binding["path_map"])
    config = resolver.rewrite(read(data["config"]))
    source_path = data["sources"]["N"]["path"]
    index.identity(source_path)
    native_source = read(source_path)
    parent_predictions = {method: load_prediction(data["predictions"][method]) for method in ("D2", "FC_EQ")}
    parent_rows = resolver.rewrite(read(Path(data["parent_readout_receipt"]).parent / "evaluation_rows.json"))["rows"]
    evaluators, scored, rows, registry_checks = {}, {}, [], {}
    for method, condition in lock["conditions"].items():
        index.identity(condition["prediction_manifest"])
        payload = load_prediction(condition["prediction_manifest"])
        if payload.prediction_key != condition["prediction_key"]:
            raise ValueError("locked light prediction changed before evaluation")
        labels = owner_labels(payload)
        partition = condition["owner_partition_sha256"]
        if partition not in evaluators:
            source = expanded_native_registry(native_source, labels)
            evidence = SimpleNamespace(scene=scene, dataset=data["dataset"], config=config,
                                       native=payload, sources={"N0": source})
            evaluator = SceneEvaluator(evidence, root / "evaluation" / partition, index=index)
            if set(evaluator.masks) != set(labels):
                raise ValueError("recovered owners were lost before the released evaluator")
            evaluators[partition] = evaluator
        evaluator = evaluators[partition]
        source_positive = set(labels)
        expected_view = official_view(evaluator.owners, labels, evaluator.minimum)
        if not set(expected_view) <= set(evaluator.masks):
            raise ValueError("official manifest has an owner absent from mask registry")
        if payload.prediction_key in scored:
            row = {**scored[payload.prediction_key], "reuse_kind": "EXACT_LOCKED_PREDICTION_ALIAS"}
        else:
            row = _parent_alias(evaluator, payload, data, parent_predictions, parent_rows, index)
            if row is None:
                row = evaluator.evaluate(labels, method, "OFFICIAL_CURRENT_CLASS", payload.prediction_key)
                row["reuse_kind"] = "RELEASED_SCORER_EXECUTED"
            scored[payload.prediction_key] = row
        row = {**row, "method": method, "map_id": map_id, "prediction_identity": payload.prediction_key,
               "record_identity": payload.record_key}
        rows.append(row)
        receipt = read(row["evaluation_receipt"])
        if {int(k) for k in receipt["view"]} != set(expected_view):
            raise ValueError("actual released export dropped or introduced owners")
        registry_checks[method] = {"positive_source_owners": sorted(source_positive),
            "mask_registry_owners": sorted(evaluator.masks), "official_manifest_owners": sorted(expected_view),
            "new_positive_owners": condition["new_positive_owners"],
            "recovered_official_eligible_owners": sorted(set(map(int, condition["recovered_labels"])) & set(expected_view)),
            "recovered_source_eligible_but_target_small": sorted(set(map(int, condition["recovered_labels"])) - set(expected_view)),
            "official_current_class_ranks": receipt["view"], "genuine_native_unavailable_rows":
                sorted(source_positive - set(map(int, native_source["objects"])))}
        atomic_write_json(root / "rows" / (method + ".json"), row)
        del payload
    result = {"status": "COMPLETE", "scene": scene, "map_id": map_id, "rows": rows,
        "prediction_lock_identity": lock["identity"], "registry_checks": registry_checks,
        "inputs": index.entries(), "elapsed_seconds": time.monotonic() - started,
        "ordered_conditions": list(lock["conditions"])}
    result["identity"] = canonical_digest({"rows": rows, "prediction_lock_identity": lock["identity"],
                                           "registry_checks": registry_checks})
    atomic_write_json(root / "evaluation_rows.json", result)
    return result


def pool_cohort(binding, cohort, scene_order, methods, *, map_id="BB00_NATIVE"):
    from static_ovmap.m2_reviewer_study.evaluation import pool
    from static_ovmap.released_loader import load_released_module

    root = Path(binding["output_root"])
    rows = []
    for scene in scene_order:
        receipt = read(root / "light" / scene / map_id / "evaluation_rows.json")
        if receipt["status"] != "COMPLETE":
            raise ValueError("official pooling requires every scene in the exact cohort")
        rows.extend(receipt["rows"])
    config = read(binding["scenes"][scene_order[0]]["config"])
    namespace = load_released_module(Path(config["runtime"]["upstream"]) / "scripts/eval_utils.py")
    dataset = binding["scenes"][scene_order[0]]["dataset"]
    namespace["init"]("Replica" if dataset == "Replica" else "Scannet200")
    output = root / "light/pools" / cohort / map_id
    results = {}
    for method in methods:
        selected = {row["scene"]: row for row in rows if row["method"] == method}
        if set(selected) != set(scene_order):
            raise ValueError("a light pool cannot omit scenes or methods")
        identities = [read(selected[scene]["evaluation_receipt"])["identity"] for scene in scene_order]
        identity = canonical_digest({"ordered_inputs": identities, "rank_mode": "OFFICIAL_CURRENT_CLASS"})
        result = None
        if map_id == "BB00_NATIVE":
            for parent_method in ("D2", "FC_EQ"):
                path = Path(binding["parent_root"]) / "pools" / cohort / map_id / parent_method / "OFFICIAL_CURRENT_CLASS.json"
                if path.is_file() and read(path)["identity"] == identity:
                    result = {**read(path), "reuse_kind": "EXACT_ORDERED_PARENT_SCORER_RECEIPTS",
                              "parent_pool": str(path)}
                    break
        if result is None:
            output.mkdir(parents=True, exist_ok=True)
            with (output / (method + ".log")).open("a") as stream, contextlib.redirect_stdout(stream):
                result = {**pool(output, rows, namespace, scene_order, method, "OFFICIAL_CURRENT_CLASS"),
                          "reuse_kind": "RELEASED_DATASET_POOL_EXECUTED_OR_EXACT_LOCAL_RECEIPTS"}
        result.update(method=method, map_id=map_id)
        atomic_write_json(output / (method + ".json"), result)
        results[method] = result
    receipt = {"status": "COMPLETE", "cohort": cohort, "scene_order": scene_order,
               "map_id": map_id, "methods": results, "aggregation": "RELEASED_DATASET_POOL"}
    receipt["identity"] = canonical_digest(receipt)
    atomic_write_json(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--scene")
    parser.add_argument("--cohort", choices=("development", "replica"))
    args = parser.parse_args()
    binding = read(args.binding)
    if args.cohort:
        from .light import METHODS
        result = pool_cohort(binding, args.cohort, binding["datasets"][args.cohort], METHODS)
        print(args.cohort, result["status"], {method: row["metrics"] for method, row in result["methods"].items()}, flush=True)
    else:
        if args.scene is None:
            parser.error("--scene or --cohort is required")
        result = evaluate_scene(binding, args.scene)
        print(args.scene, result["status"], len(result["rows"]), flush=True)
