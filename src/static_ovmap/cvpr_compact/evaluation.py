"""Locked full-registry predictions scored and pooled by the released evaluator."""

import argparse
import contextlib
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.m2_reviewer_study.evaluation import SceneEvaluator, official_view, plain, pool
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read
from static_ovmap.recovery_wave2.evaluation import expanded_native_registry, scorer_context
from static_ovmap.recovery_wave2.mechanisms import read_gzip
from static_ovmap.released_loader import load_released_module

from .projected_views import _verified_identity
from .protocol import load_spec, validate_overlaps
from .runtime import require_frozen_execution


METRICS = ("apall", "ap50", "ap25", "miou", "macc")
RANK_MODE = "OFFICIAL_CURRENT_CLASS"


def fraction_metrics(values):
    if not set(METRICS) <= set(values):
        raise ValueError("complete five-metric fractional record is required")
    result = {name: values[name] for name in METRICS}
    if any(value is not None and (not np.isfinite(value) or not 0 <= value <= 1)
           for value in result.values()):
        raise ValueError("official metrics must be finite fractions or genuinely undefined nulls")
    return result


def per_class_metrics(averages, confusion, semantic_ids, semantic_names, instance_ids, instance_names):
    semantic_ids, instance_ids = list(map(int, semantic_ids)), list(map(int, instance_ids))
    matrix = np.asarray(confusion, np.int64)
    if (0 in semantic_ids or len(set(semantic_ids)) != len(semantic_ids)
            or len(semantic_ids) != len(semantic_names) or len(instance_ids) != len(instance_names)
            or len(set(instance_ids)) != len(instance_ids) or matrix.shape != (len(semantic_ids) + 1,) * 2
            or np.any(matrix < 0) or matrix[0].sum() != 0):
        raise ValueError("per-class metrics changed full valid semantic IDs or whole-scene confusion coverage")
    names = dict(zip(semantic_ids, semantic_names, strict=True))
    instance = dict(zip(instance_ids, instance_names, strict=True))
    if (not set(instance) <= set(names) or any(names[key] != name for key, name in instance.items())
            or set(averages["classes"]) != set(instance_names)):
        raise ValueError("per-class instance vocabulary differs from the actual released ID/name order")
    rows = []
    for position, (label, name) in enumerate(zip(semantic_ids, semantic_names, strict=True), 1):
        gt, predicted, tp = int(matrix[position].sum()), int(matrix[:, position].sum()), int(matrix[position, position])
        ap = averages["classes"].get(name)
        rows.append({"class_id": label, "class_name": name, "gt_points": gt, "predicted_points": predicted,
            "gt_present": gt > 0, "iou": tp / (gt + predicted - tp) if gt else None,
            "accuracy": tp / gt if gt else None, "instance_class_in_released_evaluator": label in instance,
            "instance_unavailable_reason": "NOT_IN_RELEASED_INSTANCE_VOCABULARY" if ap is None else None,
            **{metric: plain(ap[source]) if ap is not None else None for metric, source in
               (("apall", "ap"), ("ap50", "ap50%"), ("ap25", "ap25%"))}})
    return rows


def _class_namespace(context, index):
    index.identity(context["evaluator"]["path"], context["evaluator"])
    namespace = load_released_module(context["evaluator"]["path"])
    namespace["init"]("Replica" if context["dataset"] == "Replica" else "Scannet200")
    validate_overlaps(namespace["overlaps"])
    if namespace["overlaps"].tolist() != context["runtime_overlaps"]:
        raise ValueError("per-class summary changed the exact loaded released overlap vector")
    return namespace


def trace_class_metrics(scoring, *, index=None):
    index = index or ConsumptionIndex()
    context = scoring["context"]
    namespace = _class_namespace(context, index)
    path = Path(scoring["manifest"]).with_name("trace.json.gz")
    index.identity(path)
    trace = read_gzip(path)
    if scoring["status"] != "COMPLETE" or not all(trace["parity"][key] for key in ("ap_exact", "pr_and_fn_exact")):
        raise ValueError("per-class summary requires completed unchanged released scoring")
    shape = (len(namespace["dist_threshes"]), len(namespace["CLASS_LABELS"]), len(namespace["overlaps"]))
    ap = np.full(shape, np.nan)
    seen = set()
    for state in trace["states"]:
        key = tuple(state[name] for name in ("distance_index", "class_index", "overlap_index"))
        if (key in seen or any(not 0 <= value < maximum for value, maximum in zip(key, shape, strict=True))
                or state["class_label"] != namespace["CLASS_LABELS"][key[1]]
                or abs(state["overlap_threshold"] - namespace["overlaps"][key[2]]) > 1e-12):
            raise ValueError("per-class trace changed or duplicated its original class/threshold state")
        seen.add(key)
        ap[key] = np.nan if state["ap"] is None else state["ap"]
    if len(seen) != int(np.prod(shape)):
        raise ValueError("per-class trace omitted actual released terminal AP states")
    averages = plain(namespace["compute_averages"](ap))
    for metric, key in (("apall", "all_ap"), ("ap50", "all_ap_50%"), ("ap25", "all_ap_25%")):
        before, after = scoring["metrics"][metric], averages[key]
        if (before is None) != (after is None) or (before is not None and abs(before - after) > 1e-12):
            raise ValueError("per-class trace averages differ from the already released metric receipt")
    return per_class_metrics(averages, scoring["confusion"], context["valid_ids"], context["class_names"],
                             namespace["VALID_CLASS_IDS"], namespace["CLASS_LABELS"])


def released_pool_with_classes(root, rows, namespace, scenes, method, rank_mode, index):
    by_scene = {row["scene"]: row for row in rows if row["method"] == method and row["rank_mode"] == rank_mode}
    if len(by_scene) != len(rows) or set(by_scene) != set(scenes):
        raise ValueError("per-class pool requires complete unique ordered same-method scoring rows")
    scores = []
    for scene in scenes:
        path = by_scene[scene]["evaluation_receipt"]
        index.identity(path)
        score = read(path)
        if score["status"] != "COMPLETE" or score["identity"] != by_scene[scene]["evaluation_identity"]:
            raise ValueError("per-class pool changed its actual completed released scene receipt")
        scores.append(score)
    context = scores[0]["context"]
    for score in scores:
        other = score["context"]
        if any(other[key] != context[key] for key in ("valid_ids", "class_names", "runtime_overlaps")):
            raise ValueError("per-class pool changed its actual class/threshold order across scenes")
        if other["evaluator"]["sha256"] != context["evaluator"]["sha256"]:
            raise ValueError("per-class pool changed the original scoring implementation")
    matrix = np.sum([np.asarray(score["confusion"], np.int64) for score in scores], axis=0)
    identity = canonical_digest({"ordered_inputs": [score["identity"] for score in scores], "rank_mode": rank_mode,
        "evaluator": index.identity(context["evaluator"]["path"], context["evaluator"]),
        "producer": index.identity(__file__), "semantic_ids": context["valid_ids"], "semantic_names": context["class_names"]})
    path = Path(root) / (method + "_per_class.json")
    captured, original = [], namespace["evaluate"]
    def capture(*args, **kwargs):
        values = original(*args, **kwargs)
        captured.append(plain(values))
        return values
    namespace["evaluate"] = capture
    try:
        measured = pool(root, rows, namespace, scenes, method, rank_mode)
    finally:
        namespace["evaluate"] = original
    if path.is_file():
        index.identity(path)
        details = read(path)
        _verified_identity(details)
        if details["input_identity"] != identity:
            raise ValueError("completed per-class pool changed its exact ordered scoring inputs")
    else:
        if len(captured) != 1:
            raise ValueError("cached pool lacks captured original per-class values; explicit invalidation is required")
        details = {"status": "COMPLETE", "method": method, "scene_order": list(scenes), "input_identity": identity,
            "ordered_inputs": [score["identity"] for score in scores], "metric_unit": "FRACTION",
            "raw_released_averages": captured[0], "confusion": matrix.tolist(),
            "classes": per_class_metrics(captured[0], matrix, context["valid_ids"], context["class_names"],
                                         namespace["VALID_CLASS_IDS"], namespace["CLASS_LABELS"])}
        details["identity"] = canonical_digest(details)
        atomic_write_json(path, details)
    for metric, key in (("apall", "all_ap"), ("ap50", "all_ap_50%"), ("ap25", "all_ap_25%")):
        if measured["metrics"][metric] != details["raw_released_averages"][key]:
            raise ValueError("captured per-class pool differs from actual original dataset AP")
    return measured, details


def equivalent_scoring_input(prediction, reference, context, reference_context, view, reference_view):
    return (prediction.locked and reference.locked and prediction.scene_id == reference.scene_id
        and prediction.geometry == reference.geometry
        and np.array_equal(prediction.owner_ids, reference.owner_ids)
        and np.array_equal(prediction.semantic_labels, reference.semantic_labels)
        and prediction.instance_ranks == reference.instance_ranks
        and view == reference_view and scorer_context(context) == scorer_context(reference_context))


def select_pool_rows(spec, cohort, method, receipts):
    permitted = [row["id"] for row in spec["methods"] if cohort in row["cohorts"]]
    if cohort not in spec["cohorts"] or method not in permitted:
        raise ValueError("pool method leaves its fixed cohort")
    scenes = spec["cohorts"][cohort]
    if set(receipts) != set(scenes):
        raise ValueError("official pool requires the complete fixed scene set")
    result = []
    for scene in scenes:
        receipt = receipts[scene]
        if receipt["status"] != "COMPLETE" or receipt["scene"] != scene:
            raise ValueError("official pool requires complete exact-scene receipts")
        rows = [row for row in receipt["rows"] if row["method"] == method and row["rank_mode"] == RANK_MODE]
        if len(rows) != 1:
            raise ValueError("each fixed scene-method must appear exactly once in its pool")
        if rows[0]["status"] != "COMPLETE" or rows[0]["scene"] != scene:
            raise ValueError("official pool requires complete exact-scene rows")
        result.append(rows[0])
    return result


def _context(binding, scene, context):
    if context is not None:
        return context
    path = Path(binding["output_root"]) / "contexts" / (scene + ".json")
    if path.is_file():
        data = read(path)
        _verified_identity(data)
        return data
    return binding["scenes"][scene]


def _prepare_annotations(binding, scene, data, config, index):
    if scene not in binding["cohorts"]["scannet_cf18"]:
        return
    from plyfile import PlyData
    from static_ovmap.module_validation.scannet_ground_truth import prepare_ground_truth

    preparation = read(data["preparation_receipt"])
    _verified_identity(preparation)
    inputs = preparation["raw_files"]
    for suffix in ("_vh_clean_2.labels.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json"):
        index.identity(inputs[suffix]["path"], inputs[suffix])
    index.identity(preparation["label_map"]["path"], preparation["label_map"])
    planned = Path(config["scenes"][scene]["annotations"])
    if not planned.resolve().is_relative_to(Path(binding["output_root"]).resolve()):
        raise ValueError("new annotation conversion must stay within its own task")
    actual = prepare_ground_truth(Path(config["runtime"]["upstream"]),
        Path(inputs["_vh_clean_2.labels.ply"]["path"]).parent, scene,
        Path(preparation["label_map"]["path"]), planned.parent)
    if actual != planned:
        raise ValueError("converted annotations leave the fixed planned scene path")
    receipt = read(actual)
    index.identity(actual)
    for item in receipt["outputs"]:
        index.identity(item["path"], item)
    geometry = inputs["_vh_clean_2.ply"]
    index.identity(geometry["path"], geometry)
    vertices = PlyData.read(geometry["path"])["vertex"].data
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
    with np.load(receipt["arrays_path"], allow_pickle=False) as arrays:
        if not np.array_equal(arrays["xyz"], xyz):
            raise ValueError("evaluation changed the whole geometry-only target coordinate order")
    index.identity(config["scenes"][scene]["projection"])
    atomic_write_json(Path(config["attempt_root"]) / "source_manifest.json", {"entries": index.entries(),
        "prediction_lock_precedes_annotation_conversion": True})


def _score_outputs(receipt, index):
    path = Path(receipt["manifest"])
    for file in (path, path.with_name("receipt.json"), path.with_name("matches.json.gz"), path.with_name("trace.json.gz")):
        index.identity(file)


def _parent_alias(binding, scene, data, evaluator, payload, view, index):
    if scene in binding["cohorts"]["scannet_cf18"]:
        return None
    path = Path(data["parent_readout_receipt"]).parent / "evaluation_rows.json"
    index.identity(path)
    resolver = PathResolver(binding["path_map"])
    rows = resolver.rewrite(read(path))["rows"]
    for method in ("NATIVE_READOUT", "D2", "FC_EQ"):
        parent_path = data["predictions"].get(method)
        if not parent_path:
            continue
        parent = load_prediction(parent_path)
        if (payload.geometry != parent.geometry or payload.instance_ranks != parent.instance_ranks
                or not np.array_equal(payload.owner_ids, parent.owner_ids)
                or not np.array_equal(payload.semantic_labels, parent.semantic_labels)):
            continue
        matches = [row for row in rows if row["method"] == method and row["rank_mode"] == RANK_MODE]
        if len(matches) != 1:
            raise ValueError("parent alias requires exactly one actual released scene row")
        row = matches[0]
        index.identity(row["evaluation_receipt"])
        receipt = resolver.rewrite(read(row["evaluation_receipt"]))
        if receipt["status"] != "COMPLETE":
            continue
        if equivalent_scoring_input(payload, parent, evaluator.context, receipt["context"], view, receipt["view"]):
            _score_outputs(receipt, index)
            index.identity(parent_path)
            return {**row, "reuse_kind": "EXACT_PARENT_ARRAYS_RANKS_AND_SCORER_CONTEXT",
                "alias_proof": {"parent_prediction_manifest": parent_path, "parent_prediction_key": parent.prediction_key,
                    "source_owner_sha256": _array_digest(payload.owner_ids),
                    "semantic_sha256": _array_digest(payload.semantic_labels),
                    "instance_rank_identity": canonical_digest(payload.instance_ranks),
                    "official_view_identity": canonical_digest(view),
                    "scorer_context_identity": canonical_digest(scorer_context(evaluator.context))}}
    return None


def evaluate_scene(binding, scene, prediction_lock, output_root, *, context=None):
    freeze = require_frozen_execution(binding, scene)
    spec, root = load_spec(binding["spec"]), Path(output_root)
    _verified_identity(prediction_lock)
    if prediction_lock["status"] != "PREDICTIONS_LOCKED" or prediction_lock["scene"] != scene:
        raise ValueError("official scoring requires all same-scene predictions locked before labels")
    cohort = "scannet_cf18" if scene in spec["cohorts"]["scannet_cf18"] else "replica8"
    methods = [row["id"] for row in spec["methods"] if cohort in row["cohorts"]]
    if list(prediction_lock["predictions"]) != sorted(methods) and list(prediction_lock["predictions"]) != methods:
        raise ValueError("prediction lock must contain every fixed same-cohort method exactly once")
    data = _context(binding, scene, context)
    index = ConsumptionIndex(root / "input_verifications.json")
    for item in prediction_lock["inputs"] + prediction_lock["outputs"]:
        index.identity(item["path"], item)
    index.identity(data["config"])
    config = PathResolver(binding["path_map"]).rewrite(read(data["config"]))
    _prepare_annotations(binding, scene, data, config, index)
    index.identity(config["scenes"][scene]["annotations"])
    index.identity(data["sources"]["N"]["path"])
    native_source = read(data["sources"]["N"]["path"])
    _verified_identity(native_source)
    identity = canonical_digest({"prediction_lock": prediction_lock["identity"], "freeze": freeze,
        "context_config": index.identity(data["config"]), "annotation": index.identity(config["scenes"][scene]["annotations"]),
        "native_source": native_source["identity"], "producer": index.identity(__file__),
        "released_adapter": index.identity(Path(__file__).parents[1] / "m2_reviewer_study/evaluation.py")})
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        old = read(receipt_path)
        _verified_identity(old)
        if old["status"] == "COMPLETE":
            if old["input_identity"] != identity:
                raise ValueError("completed official scoring inputs changed; invalidate explicitly")
            for item in old["outputs"]:
                index.identity(item["path"], item)
            return old
        receipt_path.rename(root / ("receipt.failed_" + str(time.time_ns()) + ".json"))
    started = time.monotonic()
    result = {"status": "RUNNING", "scene": scene, "cohort": cohort, "input_identity": identity,
        "freeze": freeze, "prediction_lock_identity": prediction_lock["identity"],
        "predictions_locked_before_annotation_conversion": True}
    atomic_write_json(receipt_path, result)
    rows, evaluators, registry_checks, outputs = [], {}, {}, []
    try:
        for method in methods:
            path = prediction_lock["predictions"][method]
            payload = load_prediction(path)
            if payload.prediction_key != prediction_lock["prediction_identities"][method]["prediction_key"]:
                raise ValueError("locked prediction changed before released scoring")
            labels = owner_labels(payload)
            partition = _array_digest(payload.owner_ids)
            if partition not in evaluators:
                expanded = expanded_native_registry(native_source, labels)
                evidence = SimpleNamespace(scene=scene, dataset=data["dataset"], config=config,
                                           native=payload, sources={"N0": expanded})
                evaluator = SceneEvaluator(evidence, root / "partitions" / partition, index=index)
                validate_overlaps(evaluator.namespace["overlaps"])
                if evaluator.minimum != 100 or set(evaluator.masks) != set(labels):
                    raise ValueError("released minimum or full positive-owner mask registry changed")
                evaluators[partition] = evaluator
            evaluator = evaluators[partition]
            view = {str(owner): row for owner, row in official_view(evaluator.owners, labels, evaluator.minimum).items()}
            ranks = dict(payload.instance_ranks)
            if set(ranks) != set(labels) or any(f"{ranks[owner]:.6f}" != view.get(str(owner), {"rank": "0.000000"})["rank"]
                                               for owner in labels):
                raise ValueError("locked ranks differ from the actual released current-class target-area export")
            row = _parent_alias(binding, scene, data, evaluator, payload, view, index)
            if row is None:
                log_path = root / (method + ".log")
                with log_path.open("a") as stream, contextlib.redirect_stdout(stream):
                    row = evaluator.evaluate(labels, method, RANK_MODE, payload.prediction_key)
                row = {**row, "reuse_kind": "RELEASED_SCORER_EXECUTED_OR_EXACT_LOCAL_RECEIPT"}
            scoring = read(row["evaluation_receipt"])
            if scoring["view"] != view:
                raise ValueError("actual released manifest differs from the locked official view")
            _score_outputs(scoring, index)
            matrix = np.asarray(scoring["confusion"], np.int64)
            valid_targets = int(np.isin(evaluator.targets["gt_semantic"], evaluator.ids).sum())
            if matrix.shape != (len(evaluator.ids) + 1,) * 2 or int(matrix.sum()) != valid_targets:
                raise ValueError("semantic evaluation dropped whole-scene target errors or unknown predictions")
            row.update(method=method, scene=scene, dataset=data["dataset"], cohort=cohort,
                record_identity=payload.record_key, prediction_identity=payload.prediction_key,
                metrics=fraction_metrics(scoring["metrics"]), metric_unit="FRACTION", runtime_overlaps=scoring["context"]["runtime_overlaps"],
                scorer_context_identity=canonical_digest(scorer_context(scoring["context"])))
            row["identity"] = canonical_digest({k: v for k, v in row.items() if k != "identity"})
            row_path = root / "rows" / (method + ".json")
            atomic_write_json(row_path, row)
            outputs.append(index.identity(row_path))
            rows.append(row)
            recovered = set(map(int, payload.metadata.get("recovered_labels", {})))
            unknown = sorted(owner for owner, label in labels.items() if label == 0)
            if any(ranks[owner] != 0 or str(owner) in view for owner in unknown):
                raise ValueError("unknown semantic owners must retain support with zero rank and no instance export")
            added = sorted(set(labels) - set(map(int, native_source["objects"])))
            if any(evaluator.evidence.sources["N0"]["objects"][str(owner)]["available"] for owner in added):
                raise ValueError("expanded mask registry invented available Native evidence for a recovered owner")
            registry_checks[method] = {"source_positive_owners": sorted(labels), "mask_registry_owners": sorted(evaluator.masks),
                "official_manifest_owners": sorted(map(int, view)), "recovered_source_available": sorted(recovered),
                "recovered_scoring_eligible": sorted(recovered & set(map(int, view))),
                "recovered_target_small": sorted(recovered - set(map(int, view))), "unknown_semantic_positive_owners": unknown,
                "native_unavailable_added_rows": added,
                "native_unavailable_added_rows_are_false": True, "official_current_class_ranks": view,
                "semantic_valid_target_count": valid_targets, "semantic_confusion_count": int(matrix.sum()),
                "source_owner_partition_sha256": partition}
        for row in rows:
            scoring = read(row["evaluation_receipt"])
            for file in (Path(row["evaluation_receipt"]), Path(scoring["manifest"]),
                         Path(scoring["manifest"]).with_name("matches.json.gz"), Path(scoring["manifest"]).with_name("trace.json.gz")):
                outputs.append(index.identity(file))
        result.update(status="COMPLETE", rows=rows, registry_checks=registry_checks, outputs=outputs,
            metric_unit="FRACTION", row_count=len(rows), aggregation="SCENE")
    except BaseException as exc:
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", rows=rows, outputs=outputs)
        raise
    finally:
        result.update(inputs=index.entries(), elapsed_seconds=time.monotonic() - started)
        result["identity"] = canonical_digest({k: v for k, v in result.items() if k != "identity"})
        atomic_write_json(receipt_path, result)
        index.write_memo(root / "input_verifications.json")
    return result


def pool_cohort(binding, cohort):
    spec = load_spec(binding["spec"])
    if cohort not in spec["cohorts"]:
        raise ValueError("pool leaves the fixed main cohorts")
    scenes = spec["cohorts"][cohort]
    require_frozen_execution(binding, scenes[0])
    root = Path(binding["output_root"])
    output = root / "pools" / cohort
    output.mkdir(parents=True, exist_ok=True)
    index = ConsumptionIndex(output / "input_verifications.json")
    receipts = {}
    for scene in scenes:
        path = root / "evaluation" / scene / "receipt.json"
        index.identity(path)
        receipt = read(path)
        _verified_identity(receipt)
        for item in receipt["outputs"]:
            index.identity(item["path"], item)
        receipts[scene] = receipt
    methods = [row["id"] for row in spec["methods"] if cohort in row["cohorts"]]
    data = _context(binding, scenes[0], None)
    config = PathResolver(binding["path_map"]).rewrite(read(data["config"]))
    evaluator_path = Path(config["runtime"]["upstream"]) / "scripts/eval_utils.py"
    evaluator_identity = index.identity(evaluator_path)
    namespace = load_released_module(evaluator_path)
    namespace["init"]("Replica" if cohort == "replica8" else "Scannet200")
    validate_overlaps(namespace["overlaps"])
    results = {}
    for method in methods:
        rows = select_pool_rows(spec, cohort, method, receipts)
        for row in rows:
            scoring = read(row["evaluation_receipt"])
            if scoring["context"]["evaluator"]["sha256"] != evaluator_identity["sha256"]:
                raise ValueError("whole-cohort pooling changed its actual released scorer bytes")
        with (output / (method + ".log")).open("a") as stream, contextlib.redirect_stdout(stream):
            measured, classes = released_pool_with_classes(output, rows, namespace, scenes, method, RANK_MODE, index)
        result = {**measured, "method": method, "cohort": cohort, "metrics": fraction_metrics(measured["metrics"]),
            "metric_unit": "FRACTION", "row_identities": [row["identity"] for row in rows],
            "runtime_overlaps": namespace["overlaps"].tolist(), "released_evaluator": evaluator_identity,
            "reuse_kind": "EXACT_ORDERED_RELEASED_POOL_INPUTS",
            "per_class_receipt": index.identity(output / (method + "_per_class.json")),
            "per_class_identity": classes["identity"], "scoring_definition": {
                "semantic_valid_ids": read(rows[0]["evaluation_receipt"])["context"]["valid_ids"],
                "instance_valid_ids": list(map(int, namespace["VALID_CLASS_IDS"])),
                "instance_class_names": list(namespace["CLASS_LABELS"]), "target_min_region_size": 100,
                "semantic_average": "GT_PRESENT_CLASSES_OF_SUMMED_WHOLE_SCENE_CONFUSIONS",
                "projection": "FULL_OPEN3D_FP32_EXACT_1NN_STRICT_DISTANCE_SQUARED_LT_0.05_SQUARED"}}
        result["identity"] = canonical_digest({k: v for k, v in result.items() if k != "identity"})
        atomic_write_json(output / (method + ".json"), result)
        results[method] = result
    receipt = {"status": "COMPLETE", "cohort": cohort, "scene_order": scenes, "methods": results,
        "aggregation": "RELEASED_DATASET_POOL", "metric_unit": "FRACTION", "inputs": index.entries(),
        "outputs": [index.identity(output / (method + suffix)) for method in methods
                    for suffix in (".json", "_per_class.json")]}
    receipt["identity"] = canonical_digest(receipt)
    atomic_write_json(output / "receipt.json", receipt)
    index.write_memo(output / "input_verifications.json")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--scene")
    parser.add_argument("--lock")
    parser.add_argument("--output-root")
    parser.add_argument("--cohort", choices=("replica8", "scannet_cf18"))
    args = parser.parse_args()
    binding = read(args.binding)
    if args.cohort:
        result = pool_cohort(binding, args.cohort)
    else:
        if not args.scene or not args.lock or not args.output_root:
            parser.error("scene evaluation requires --scene, --lock and --output-root")
        result = evaluate_scene(binding, args.scene, read(args.lock), args.output_root)
    print(result.get("scene", result.get("cohort")), result["status"], flush=True)
