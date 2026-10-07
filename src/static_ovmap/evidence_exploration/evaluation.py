"""Prediction-locked adapter to unchanged released scene and full-cohort scoring."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import contextlib
import subprocess
import time

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.evaluation import fraction_metrics, trace_class_metrics, released_pool_with_classes
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.m2_reviewer_study.evaluation import SceneEvaluator, official_view
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read
from static_ovmap.recovery_wave2.evaluation import expanded_native_registry, scorer_context
from static_ovmap.released_loader import load_released_module

from .outputs import prediction_content


def evaluate_scene(job):
    binding, root_text, scene = job
    root, index = Path(root_text), ConsumptionIndex()
    dest = root/"evaluation"/scene
    lock = read(root/"predictions"/scene/"receipt.json")
    _verified_identity(lock)
    if lock["status"] != "PREDICTIONS_LOCKED" or len(lock["outcomes"]) != 9:
        raise ValueError("all nine predictions must lock before scene GT scoring")
    key = canonical_digest({"prediction_lock": lock["identity"], "producer": index.identity(__file__)["sha256"]})
    target = dest/"receipt.json"
    if target.exists():
        previous = read(target)
        _verified_identity(previous)
        if previous["input_identity"] != key:
            raise ValueError("completed released scoring inputs changed")
        return previous
    started = time.monotonic()
    resolver = PathResolver(binding["path_map"])
    data = resolver.rewrite(read(binding["scenes"][scene]["context"]["path"]))
    config = resolver.rewrite(read(data["config"]))
    native_source = read(data["sources"]["N"]["path"])
    rows, seen, evaluators = [], {}, {}
    new_calls = 0
    for method in binding["specification"]["methods"]:
        name, outcome = method["id"], lock["outcomes"][method["id"]]
        payload = load_prediction(outcome["manifest"])
        content = prediction_content(payload)
        if content != outcome["content_identity"]:
            raise ValueError("locked prediction changed before its GT scoring")
        if name in binding["scenes"][scene]["baseline_rows"]:
            row = dict(binding["scenes"][scene]["baseline_rows"][name]["row"])
            score = read(row["evaluation_receipt"])
            classes = trace_class_metrics(score, index=index)
            row.update(reuse_kind="IDENTITY_PROVEN_BASELINE_IMPORT", alias_of=None)
        else:
            labels = owner_labels(payload)
            partition = _array_digest(payload.owner_ids)
            if partition not in evaluators:
                source = expanded_native_registry(native_source, labels)
                evidence = SimpleNamespace(scene=scene, dataset=data["dataset"], config=config,
                    native=payload, sources={"N0": source})
                evaluators[partition] = SceneEvaluator(evidence, dest/"partitions"/partition, index=index)
                if set(evaluators[partition].masks) != set(labels):
                    raise ValueError("accepted recovered registry does not reach the scorer")
            evaluator = evaluators[partition]
            view = {str(k): v for k, v in official_view(evaluator.owners, labels, evaluator.minimum).items()}
            if content in seen:
                previous = seen[content]
                score = read(previous["evaluation_receipt"])
                if score["view"] != view or scorer_context(score["context"]) != scorer_context(evaluator.context):
                    raise ValueError("prediction alias does not establish identical official view/scorer context")
                row = {**previous, "reuse_kind": "IDENTICAL_PREDICTION_VIEW_SCORER_ALIAS", "alias_of": previous["method"]}
                classes = previous["classes"]
            else:
                with (dest/(name+".log")).open("w") as log, contextlib.redirect_stdout(log):
                    row = evaluator.evaluate(labels, name, "OFFICIAL_CURRENT_CLASS", payload.prediction_key)
                score = read(row["evaluation_receipt"])
                classes = trace_class_metrics(score, index=index)
                row.update(reuse_kind="UNCHANGED_RELEASED_SCORER_EXECUTED", alias_of=None)
                new_calls += 1
        row = seal({**{k: v for k, v in row.items() if k != "identity"}, "method": name, "method_id": name,
            "scene": scene, "cohort": binding["scenes"][scene]["cohort"], "metric_unit": "FRACTION",
            "metrics": fraction_metrics(row["metrics"]), "classes": classes,
            "prediction_content_identity": content, "prediction_lock_identity": lock["identity"],
            "scorer_context_identity": canonical_digest(scorer_context(score["context"])),
            "candidate_decisions": outcome["decisions"], "counts": outcome["counts"]})
        atomic_write_json(dest/"rows"/(name+".json"), row)
        rows.append(row)
        seen[content] = row
    result = seal({"status": "COMPLETE", "scene": scene, "input_identity": key, "rows": rows,
        "unique_new_scoring_calls": new_calls, "baseline_imports": 2, "new_outcomes": 7,
        "elapsed_seconds": time.monotonic()-started})
    atomic_write_json(target, result)
    print("EVALUATED", scene, "9 rows", new_calls, "new scorer calls", flush=True)
    return result


def require_freeze(root):
    index = ConsumptionIndex()
    freeze = read(Path(root)/"implementation_freeze.json")
    _verified_identity(freeze)
    if freeze["status"] != "IMPLEMENTATION_CONSTANTS_ASSETS_FROZEN_COMMITTED":
        raise ValueError("full-cohort scoring requires the implementation commit freeze")
    for item in freeze["operator_sources"]+freeze["assets"]:
        index.identity(item["path"], item)
    subprocess.run(["git", "cat-file", "-e", freeze["commit"]+"^{commit}"], check=True)
    return freeze


def evaluate(binding, root):
    root = Path(root)
    require_freeze(root)
    with ProcessPoolExecutor(max_workers=3) as executor:
        scenes = list(executor.map(evaluate_scene, [(binding, str(root), scene) for scene in binding["scenes"]]))
    rows = [row for result in scenes for row in result["rows"]]
    if len(rows) != 234:
        raise ValueError("fixed scene-method coverage is incomplete")
    pools = []
    index = ConsumptionIndex()
    for cohort, order in binding["cohorts"].items():
        namespace = None
        seen = {}
        for method in binding["specification"]["methods"]:
            name = method["id"]
            selected = [next(row for row in rows if row["scene"] == scene and row["method"] == name) for scene in order]
            content = canonical_digest([x["prediction_content_identity"] for x in selected])
            if name in binding["baseline_pools"][cohort]:
                inherited = binding["baseline_pools"][cohort][name]
                source = read(inherited["receipt_path"])
                details_path = Path(inherited["receipt_path"]).with_name(source["method"]+"_per_class.json")
                details = read(details_path)
                _verified_identity(details)
                if source["ordered_inputs"] != [x["evaluation_identity"] for x in selected]:
                    raise ValueError("imported full pool does not match ordered actual scene scores")
                measured, reuse, alias = source, "IDENTITY_PROVEN_BASELINE_POOL_IMPORT", None
            elif content in seen:
                previous = seen[content]
                measured, details, reuse, alias = previous, {"classes": previous["classes"]}, "IDENTICAL_ORDERED_POOL_ALIAS", previous["method"]
            else:
                score = read(selected[0]["evaluation_receipt"])
                if namespace is None:
                    namespace = load_released_module(score["context"]["evaluator"]["path"])
                    namespace["init"]("Replica" if cohort == "replica8" else "Scannet200")
                pool_root = root/"pools"/cohort
                pool_root.mkdir(parents=True, exist_ok=True)
                with (pool_root/(name+".log")).open("w") as log, contextlib.redirect_stdout(log):
                    measured, details = released_pool_with_classes(pool_root, selected, namespace, order, name, "OFFICIAL_CURRENT_CLASS", index)
                reuse, alias = "UNCHANGED_RELEASED_FULL_COHORT_POOL", None
            result = seal({"status": "COMPLETE", "method": name, "method_id": name, "cohort": cohort,
                "scene_order": list(order), "metrics": fraction_metrics(measured["metrics"]), "metric_unit": "FRACTION",
                "classes": details["classes"], "reuse_kind": reuse, "alias_of": alias,
                "ordered_scene_row_identities": [x["identity"] for x in selected],
                "ordered_prediction_content_identity": content, "source_evaluation_identity": measured["identity"]})
            atomic_write_json(root/"pools"/cohort/(name+".json"), result)
            pools.append(result)
            seen[content] = result
    if len(pools) != 18:
        raise ValueError("required 18 full-cohort records are incomplete")
    store = seal({"status": "SCIENCE_COMPLETE", "scene_metrics": rows, "pooled_metrics": pools,
        "scene_method_coverage": len(rows), "full_cohort_pool_coverage": len(pools),
        "new_unique_scene_scoring_calls": sum(x["unique_new_scoring_calls"] for x in scenes),
        "imported_baseline_rows": 52, "new_method_outcomes": 182,
        "new_unique_full_pool_calls": sum(x["reuse_kind"] == "UNCHANGED_RELEASED_FULL_COHORT_POOL" for x in pools)})
    atomic_write_json(root/"result_store.json", store)
    print("SCIENCE_COMPLETE", len(rows), "scene rows", len(pools), "full pools", flush=True)
    return store
