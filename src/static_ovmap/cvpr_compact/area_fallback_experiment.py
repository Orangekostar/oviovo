"""Read-only v1 evidence, independent v2 sources, locks and released scoring."""

import contextlib
import copy
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor, signed_mask
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.m2_reviewer_study.evaluation import SceneEvaluator, official_view
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.module_validation.scannet_study import load_prediction, save_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read
from static_ovmap.recovery_wave2.evaluation import expanded_native_registry, scorer_context
from static_ovmap.released_loader import load_released_module

from .area_fallback import PROTOCOL, region_vector
from .costs import recovery_usage
from .evaluation import METRICS, RANK_MODE, fraction_metrics, released_pool_with_classes
from .outputs import build_method_outputs
from .projected_views import _verified_identity, load_projected_request
from .protocol import load_spec, validate_overlaps
from .region_worker import FCSession, classify_regions, projected_plan, validate_request_outcomes
from .runtime import FREEZE_FILE


RECOVERY_METHODS = {"CT_A2_R": "G1", "CT_A3_ER": "G1", "CT_A5_FC_ONLY": "G1", "CT_G3": "G3"}


def seal(value):
    value["identity"] = canonical_digest({key: item for key, item in value.items() if key != "identity"})
    return value


def document(path, index, expected=None, *, verify=True):
    index.identity(path, expected)
    value = read(path)
    if verify:
        _verified_identity(value)
    return value


def environment(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("v2 output must be independent of its read-only v1 parent")
    index = ConsumptionIndex(output / "input_verifications.json")
    binding = document(parent / "resolved_inputs.json", index)
    spec = load_spec(binding["spec"])
    return parent, output, binding, spec, index


def scene_context(parent, binding, scene, index):
    path = parent / "contexts" / (scene + ".json")
    return document(path, index) if path.is_file() else binding["scenes"][scene]


def scene_evidence(parent, spec, scene, index):
    manifest_path = parent / "projected_views" / scene / "manifest.json"
    manifest = document(manifest_path, index)
    views = document(manifest_path.with_name("receipt.json"), index)
    registry = document(views["registry"], index)
    arms = ("G1", "G3") if scene in spec["cohorts"]["replica8"] else ("G1",)
    plan = projected_plan(registry, manifest, arms)
    selected = list(dict.fromkeys(rid for arm in plan.values() for rows in arm.values() for rid in rows))
    original = document(parent / "recovery" / scene / "receipt.json", index)
    if any(rid not in original["requests"] for rid in selected):
        raise ValueError("v2 cannot invent a projected request absent from the original worker")
    features = {}
    if original.get("features"):
        item = original["features"]
        index.identity(item["path"], item)
        with np.load(item["path"], allow_pickle=False) as arrays:
            features = {str(rid): vector.copy() for rid, vector in zip(arrays["request_ids"], arrays["features"])}
    features = {rid: vector for rid, vector in features.items() if rid in selected}
    completed = {rid for rid in selected if original["requests"][rid]["status"] == "COMPLETE"}
    if set(features) != completed:
        raise ValueError("original successful projected vectors differ from their exact request outcomes")
    for vector in features.values():
        if (vector.dtype != np.float32 or vector.ndim != 1 or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1., rtol=0, atol=1e-5)):
            raise ValueError("original successful region feature is not finite unit FP32")
    return manifest_path, manifest, registry, plan, selected, original, features


def preflight(parent, output):
    parent, output, binding, spec, index = environment(parent, output)
    frozen = document(Path(binding["repository_root"]) / FREEZE_FILE, index)
    if frozen["binding_identity"] != binding["identity"]:
        raise ValueError("parent freeze and binding differ")
    for item in frozen["implementation_sources"]:
        index.identity(Path(binding["repository_root"]) / item["path"], item)
    scenes = {}
    for scene in [*spec["cohorts"]["replica8"], *spec["cohorts"]["scannet_cf18"]]:
        _, _, _, plan, selected, original, features = scene_evidence(parent, spec, scene, index)
        failures = {rid: original["requests"][rid]["reason"] for rid in selected if rid not in features}
        if any(reason != "EMPTY_DENSE_MASK_SUPPORT" for reason in failures.values()):
            raise ValueError("area fallback may repair only original EMPTY_DENSE_MASK_SUPPORT failures")
        scenes[scene] = {"selected": len(selected), "original_successes": len(features),
                         "empty_requests": list(failures), "plan": plan}
    result = seal({"status": "PREFLIGHT_COMPLETE", "protocol": PROTOCOL,
        "parent_binding_identity": binding["identity"], "parent": str(parent), "output": str(output),
        "scope": "FIXED_PROJECTED_G1_G3_RECOVERY_ONLY", "incumbent_and_archived_U2_evidence": "UNCHANGED",
        "fallback": "ONLY_EMPTY_HARD_SUPPORT; AREA_OCCUPANCY_WEIGHTED_MEAN; ORIGINAL_FROZEN_HEAD",
        "GT_input": False, "metrics_exposed_before_new_protocol": True, "scenes": scenes,
        "cohorts": spec["cohorts"], "producer": index.identity(Path(__file__).with_name("area_fallback.py")),
        "driver": index.identity(__file__), "inputs": index.entries()})
    path = output / "experiment.json"
    if path.exists():
        previous = document(path, index)
        for key in ("protocol", "parent_binding_identity", "scenes", "producer", "driver"):
            if previous[key] != result[key]:
                raise ValueError("v2 experiment changed; use a fresh output directory")
        return previous
    atomic_write_json(path, result)
    index.write_memo(output / "input_verifications.json")
    print("PREFLIGHT", sum(row["selected"] for row in scenes.values()), "requests;",
          sum(len(row["empty_requests"]) for row in scenes.values()), "empty masks", flush=True)
    return result


def run_regions(parent, output):
    import torch

    parent, output, binding, spec, index = environment(parent, output)
    experiment = document(output / "experiment.json", index)
    if index.identity(__file__) != experiment["driver"]:
        raise ValueError("v2 driver changed after its preflight")
    torch.set_num_threads(3)
    shared_model, weight_audit = None, None
    for scene in experiment["scenes"]:
        target = output / "regions" / scene / "receipt.json"
        if target.exists():
            previous = document(target, index)
            if previous["experiment_identity"] != experiment["identity"]:
                raise ValueError("region receipt belongs to another v2 experiment")
            for item in previous["outputs"]:
                index.identity(item["path"], item)
            continue
        started = time.monotonic()
        data = scene_context(parent, binding, scene, index)
        manifest_path, manifest, registry, plan, selected, original, features = scene_evidence(parent, spec, scene, index)
        session = FCSession(binding, data, index, device="cpu")
        session.model, session.weight_audit = shared_model, weight_audit
        rows = {rid: copy.deepcopy(original["requests"][rid]) for rid in selected}
        repaired, parity = [], {}
        with torch.inference_mode():
            for rid in selected:
                row = rows[rid]
                if rid in features:
                    row.update(v2_outcome="EXACT_ORIGINAL_FP32_VECTOR_REUSE", fallback=False)
                    parity[rid] = _array_digest(features[rid])
                    continue
                if row.get("reason") != "EMPTY_DENSE_MASK_SUPPORT":
                    raise ValueError("unexpected original failure is outside the area fallback scope")
                value = load_projected_request(manifest_path, rid, index=index)
                image, size = image_tensor(value["image"], "cpu")
                tensor_key = canonical_digest({"model": session.model_key, "tensor": _array_digest(image.numpy())})
                if tensor_key != row["input_tensor_key"] or _array_digest(value["target"]) != row["target_mask_sha256"]:
                    raise ValueError("fallback changed original RGB preprocessing or full-resolution mask")
                dense_path = Path(original["required_dense_receipts"][row["image_content_key"]])
                dense_receipt = document(dense_path, index, verify=False)
                if (dense_receipt["input_tensor_key"] != tensor_key
                        or dense_receipt["image_content_key"] != row["image_content_key"]):
                    raise ValueError("fallback dense cache belongs to another RGB image or tensor")
                item = dense_receipt["arrays"]
                index.identity(item["path"], item)
                with np.load(item["path"], allow_pickle=False) as arrays:
                    dense = torch.from_numpy(arrays["dense"].copy())
                signed, _, _ = signed_mask(value["target"], size, image.shape[-2:], dense.shape[-2:], "cpu")
                model = session.load_model()
                shared_model, weight_audit = model, session.weight_audit
                vector, audit = region_vector(model, session.operators, dense, signed)
                if not audit["fallback"]:
                    raise ValueError("original empty-mask failure did not reproduce")
                features[rid] = vector.numpy().copy()
                row.pop("reason")
                row.update(status="COMPLETE", v2_outcome="AREA_FALLBACK_CPU_FROZEN_HEAD", **audit,
                           original_failure="EMPTY_DENSE_MASK_SUPPORT", dense_arrays=index.identity(item["path"], item),
                           effective_FP32_feature_sha256=_array_digest(features[rid]))
                repaired.append(rid)
        validate_request_outcomes(dict.fromkeys(selected), features, {"requests": rows})
        classified = classify_regions(registry, plan, manifest["requests"], features, session.text, session.ids)
        sources, outputs = {}, []
        for arm, objects in classified.items():
            source = seal({"source": arm, "scene": scene, "objects": objects, "protocol": PROTOCOL,
                "valid_ids": list(map(int, session.ids)), "model_identity": session.model_key,
                "text_identity": session.text_identity, "registry_identity": registry["identity"],
                "request_manifest_identity": manifest["identity"], "score_kind": "FROZEN_FC_COSINE",
                "GT_input": False, "failed_view_replacement": False})
            path = target.parent / (arm + "_FC.json")
            atomic_write_json(path, source)
            sources[arm] = index.identity(path)
            outputs.append(sources[arm])
        ordered = sorted(features)
        vectors = target.parent / "feature_vectors.npz"
        _write_npz(vectors, {"request_ids": np.asarray(ordered, dtype="U64"),
            "features": np.stack([features[rid] for rid in ordered]) if ordered else np.empty((0, session.text.shape[1]), np.float32)})
        outputs.append(index.identity(vectors))
        result = seal({"status": "COMPLETE", "scene": scene, "protocol": PROTOCOL,
            "experiment_identity": experiment["identity"], "parent_receipt_identity": original["identity"],
            "requests": rows, "plan": plan, "sources": sources, "features": index.identity(vectors),
            "repaired_requests": repaired, "original_vector_sha256": parity,
            "physical_image_encodings": 0, "GPU_inference_calls": 0, "CPU_projection_head_calls": len(repaired),
            "physical_region_poolings": len(repaired), "GT_input": False,
            "failed_view_replacement": False, "elapsed_seconds": time.monotonic() - started,
            "model_load_seconds": session.model_load_seconds, "outputs": outputs, "inputs": index.entries()})
        atomic_write_json(target, result)
        index.write_memo(output / "input_verifications.json")
        print("REGIONS", scene, "repaired", len(repaired), "reused", len(parity), flush=True)
    if weight_audit is not None:
        atomic_write_json(output / "weight_audit.json", weight_audit)


def lock_predictions(parent, output, scene):
    parent, output, binding, spec, index = environment(parent, output)
    target = output / "predictions" / scene / "receipt.json"
    if target.exists():
        result = document(target, index)
        for item in result["outputs"]:
            index.identity(item["path"], item)
        return result
    experiment = document(output / "experiment.json", index)
    data = scene_context(parent, binding, scene, index)
    original_lock = document(parent / "predictions" / scene / "receipt.json", index)
    regions = document(output / "regions" / scene / "receipt.json", index)
    views = document(parent / "projected_views" / scene / "receipt.json", index)
    registry = document(views["registry"], index)
    cohort = "replica8" if scene in spec["cohorts"]["replica8"] else "scannet_cf18"
    methods = [row for row in spec["methods"] if cohort in row["cohorts"]]
    recovery = {arm + "_FC": document(item["path"], index, item) for arm, item in regions["sources"].items()}
    paths, keys, aliases, changed, outputs = {}, {}, {}, [], []
    for method in methods:
        name = method["id"]
        path = original_lock["predictions"].get(name)
        original = document(path, index, verify=False) if path else None
        if name in RECOVERY_METHODS:
            new_labels = {owner: row["label"] for owner, row in recovery[RECOVERY_METHODS[name] + "_FC"]["objects"].items()
                          if row["available"]}
            if original is None or original["metadata"]["recovered_labels"] != new_labels:
                changed.append(method)
                continue
            if original["metadata"]["existing_source_identities"] != {name: item["identity"] for name, item in data["sources"].items()}:
                raise ValueError("prediction reuse changed incumbent N/Q/F evidence")
        if original is None or not original["locked"] or original["scene_id"] != scene:
            raise ValueError("an unchanged control lacks its original locked prediction")
        array_item = index.identity(Path(path).parent / original["arrays"]["path"], original["arrays"])
        outputs.extend([index.identity(path), array_item])
        paths[name] = str(path)
        keys[name] = {"record_key": original["record_key"], "prediction_key": original["prediction_key"]}
        aliases[name] = {"kind": "EXACT_ORIGINAL_PREDICTION", "parent_lock": original_lock["identity"],
            "recovered_decisions_equal": True, "unchanged_geometry_and_incumbent_evidence": True}
    if changed:
        baseline_path = data["predictions"]["NATIVE_READOUT"]
        baseline_manifest = document(baseline_path, index, verify=False)
        index.identity(Path(baseline_path).parent / baseline_manifest["arrays"]["path"], baseline_manifest["arrays"])
        baseline = load_prediction(baseline_path)
        capture = document(data["capture_manifest"], index)
        surface = Path(data["capture_manifest"]).parent / capture["surface"]["path"]
        index.identity(surface, capture["surface"])
        with np.load(surface, allow_pickle=False) as arrays:
            raw = arrays["original_owner"].copy()
        config = PathResolver(binding["path_map"]).rewrite(document(data["config"], index, verify=False))
        projection = Path(config["scenes"][scene]["projection"])
        projection_receipt = document(projection, index, verify=False)
        index.identity(projection.with_name("projection.npz"), {"sha256": projection_receipt["sha256"]})
        with np.load(projection.with_name("projection.npz"), allow_pickle=False) as arrays:
            nearest, matched = arrays["nearest"].copy(), arrays["matched"].copy()
        sources = {name: document(item["path"], index) for name, item in data["sources"].items()}
        costs = {name: recovery_usage(source, regions, name) for name, source in recovery.items()}
        predicted = build_method_outputs(baseline, raw, registry, sources, binding["final_temperatures"],
            data["models"]["native"]["valid_ids"], nearest, matched, recovery, changed,
            cost_inventory=original_lock["base_cost_inventory"], recovery_costs=costs)
        for name, payload in predicted.items():
            path = save_prediction(payload, target.parent / name)
            manifest = read(path)
            paths[name] = str(path)
            keys[name] = {"record_key": payload.record_key, "prediction_key": payload.prediction_key}
            outputs.extend([index.identity(path), index.identity(path.parent / manifest["arrays"]["path"], manifest["arrays"])])
    if set(paths) != {row["id"] for row in methods}:
        raise ValueError("v2 prediction lock omitted a fixed method")
    result = seal({"status": "PREDICTIONS_LOCKED", "scene": scene, "cohort": cohort, "protocol": PROTOCOL,
        "experiment_identity": experiment["identity"], "regions_identity": regions["identity"],
        "predictions": paths, "prediction_identities": keys, "aliases": aliases,
        "changed_methods": [row["id"] for row in changed], "GT_input": False,
        "output_count": len(paths), "outputs": outputs, "inputs": index.entries()})
    atomic_write_json(target, result)
    index.write_memo(output / "input_verifications.json")
    print("PREDICTIONS", scene, "changed", result["changed_methods"], flush=True)
    return result


def score_scene(parent, output, scene):
    parent, output, binding, spec, index = environment(parent, output)
    target = output / "evaluation" / scene / "receipt.json"
    if target.exists():
        previous = document(target, index)
        for item in previous["outputs"]:
            index.identity(item["path"], item)
        return previous
    lock = document(output / "predictions" / scene / "receipt.json", index)
    if lock["status"] != "PREDICTIONS_LOCKED":
        raise ValueError("official labels may be opened only after the v2 predictions lock")
    parent_scores = document(parent / "evaluation" / scene / "receipt.json", index)
    parent_rows = {row["method"]: row for row in parent_scores["rows"]}
    data = scene_context(parent, binding, scene, index)
    config = PathResolver(binding["path_map"]).rewrite(document(data["config"], index, verify=False))
    native_source = document(data["sources"]["N"]["path"], index)
    rows, checks, evaluators, outputs = [], {}, {}, []
    for name, path in lock["predictions"].items():
        manifest = document(path, index, verify=False)
        if manifest["prediction_key"] != lock["prediction_identities"][name]["prediction_key"]:
            raise ValueError("prediction changed after its v2 lock")
        if name in lock["aliases"]:
            row = copy.deepcopy(parent_rows[name])
            if row["prediction_identity"] != manifest["prediction_key"]:
                raise ValueError("original scorer alias differs from the exact unchanged prediction")
            check = copy.deepcopy(parent_scores["registry_checks"][name])
        else:
            index.identity(Path(path).parent / manifest["arrays"]["path"], manifest["arrays"])
            payload = load_prediction(path)
            labels = owner_labels(payload)
            partition = _array_digest(payload.owner_ids)
            if partition not in evaluators:
                expanded = expanded_native_registry(native_source, labels)
                evidence = SimpleNamespace(scene=scene, dataset=data["dataset"], config=config,
                                           native=payload, sources={"N0": expanded})
                evaluators[partition] = SceneEvaluator(evidence, target.parent / "partitions" / partition, index=index)
            evaluator = evaluators[partition]
            validate_overlaps(evaluator.namespace["overlaps"])
            view = {str(owner): value for owner, value in official_view(evaluator.owners, labels, evaluator.minimum).items()}
            if evaluator.minimum != 100 or any(f"{rank:.6f}" != view.get(str(owner), {"rank": "0.000000"})["rank"]
                                              for owner, rank in payload.instance_ranks):
                raise ValueError("v2 changed the official target minimum or current-class ranks")
            with (target.parent / (name + ".log")).open("a") as stream, contextlib.redirect_stdout(stream):
                row = evaluator.evaluate(labels, name, RANK_MODE, payload.prediction_key)
            scoring = document(row["evaluation_receipt"], index, verify=False)
            matrix = np.asarray(scoring["confusion"], np.int64)
            valid = int(np.isin(evaluator.targets["gt_semantic"], evaluator.ids).sum())
            if (scoring["view"] != view or matrix.shape != (len(evaluator.ids) + 1,) * 2
                    or int(matrix.sum()) != valid or not all(scoring["trace_parity"].values())):
                raise ValueError("v2 scoring changed whole-scene coverage or released matcher parity")
            added = set(map(int, manifest["metadata"]["recovered_labels"]))
            eligible = added & set(map(int, view))
            check = {"recovered_source_available": sorted(added), "recovered_scoring_eligible": sorted(eligible),
                "recovered_target_small": sorted(added - eligible), "semantic_valid_target_count": valid,
                "semantic_confusion_count": int(matrix.sum()), "official_current_class_ranks": view,
                "source_owner_partition_sha256": partition}
            row.update(cohort=lock["cohort"], record_identity=payload.record_key, metric_unit="FRACTION",
                runtime_overlaps=scoring["context"]["runtime_overlaps"], metrics=fraction_metrics(scoring["metrics"]),
                scorer_context_identity=canonical_digest(scorer_context(scoring["context"])))
        row.update(protocol=PROTOCOL, v2_prediction_lock_identity=lock["identity"],
                   reuse_kind="EXACT_V1_PREDICTION_AND_SCORING" if name in lock["aliases"] else "V2_RELEASED_SCORER")
        seal(row)
        row_path = target.parent / "rows" / (name + ".json")
        atomic_write_json(row_path, row)
        outputs.append(index.identity(row_path))
        scoring = document(row["evaluation_receipt"], index, verify=False)
        for file in (Path(row["evaluation_receipt"]), Path(scoring["manifest"]),
                     Path(scoring["manifest"]).with_name("matches.json.gz"), Path(scoring["manifest"]).with_name("trace.json.gz")):
            outputs.append(index.identity(file))
        rows.append(row)
        checks[name] = check
    result = seal({"status": "COMPLETE", "scene": scene, "cohort": lock["cohort"], "protocol": PROTOCOL,
        "prediction_lock_identity": lock["identity"], "predictions_locked_before_GT": True,
        "rows": rows, "registry_checks": checks, "row_count": len(rows), "metric_unit": "FRACTION",
        "outputs": outputs, "inputs": index.entries()})
    atomic_write_json(target, result)
    index.write_memo(target.parent / "input_verifications.json")
    print("EVALUATION", scene, "rows", len(rows), flush=True)
    return result


def pool_results(parent, output):
    parent, output, binding, spec, index = environment(parent, output)
    for cohort, scenes in spec["cohorts"].items():
        receipts = {scene: document(output / "evaluation" / scene / "receipt.json", index) for scene in scenes}
        config = read(scene_context(parent, binding, scenes[0], index)["config"])
        evaluator_path = Path(config["runtime"]["upstream"]) / "scripts/eval_utils.py"
        namespace = load_released_module(evaluator_path)
        namespace["init"]("Replica" if cohort == "replica8" else "Scannet200")
        validate_overlaps(namespace["overlaps"])
        root = output / "pools" / cohort
        root.mkdir(parents=True, exist_ok=True)
        pooled = {}
        for method in [row["id"] for row in spec["methods"] if cohort in row["cohorts"]]:
            path = root / (method + ".json")
            if path.exists():
                pooled[method] = document(path, index)
                continue
            rows = [next(row for row in receipts[scene]["rows"] if row["method"] == method) for scene in scenes]
            with (root / (method + ".log")).open("a") as stream, contextlib.redirect_stdout(stream):
                measured, details = released_pool_with_classes(root, rows, namespace, scenes, method, RANK_MODE, index)
            context = read(rows[0]["evaluation_receipt"])["context"]
            result = seal({**measured, "cohort": cohort, "method": method, "protocol": PROTOCOL,
                "metrics": fraction_metrics(measured["metrics"]), "metric_unit": "FRACTION",
                "row_identities": [row["identity"] for row in rows], "runtime_overlaps": namespace["overlaps"].tolist(),
                "released_evaluator": index.identity(evaluator_path), "per_class_identity": details["identity"],
                "scoring_definition": {"semantic_valid_ids": context["valid_ids"],
                    "instance_valid_ids": list(map(int, namespace["VALID_CLASS_IDS"])),
                    "instance_class_names": list(namespace["CLASS_LABELS"]), "target_min_region_size": 100,
                    "semantic_average": "GT_PRESENT_CLASSES_OF_SUMMED_WHOLE_SCENE_CONFUSIONS",
                    "projection": "FULL_OPEN3D_FP32_EXACT_1NN_STRICT_DISTANCE_SQUARED_LT_0.05_SQUARED"}})
            atomic_write_json(path, result)
            pooled[method] = result
            print("POOL", cohort, method, {key: round(result["metrics"][key] * 100, 4) for key in METRICS}, flush=True)
        atomic_write_json(root / "receipt.json", seal({"status": "COMPLETE", "cohort": cohort,
            "scene_order": scenes, "methods": pooled, "protocol": PROTOCOL, "inputs": index.entries()}))
    index.write_memo(output / "input_verifications.json")
