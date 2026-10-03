"""Continue fixed independent conditions while retaining proven technical blocks."""

import argparse
import ast
import copy
import inspect
from pathlib import Path
import time

import numpy as np

from static_ovmap.backbone_wave1.readouts import fuse_readout
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_study import load_prediction, save_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from . import evaluation
from .costs import recovery_usage
from .outputs import build_method_outputs
from .projected_views import _verified_identity
from .protocol import load_spec
from .recovery_run import load_recovery_inputs
from .runtime import require_frozen_execution


PARTIAL_LOCK = "PARTIAL_PREDICTIONS_LOCKED"
PARTIAL_RESULT = "PARTIAL_WITH_TECHNICAL_BLOCKS"
ARM_SOURCE = {"G1": "G1_FC", "G3": "G3_FC", "U2": "ARCHIVED_U2_FC"}


def _seal(value):
    value["identity"] = canonical_digest({key: item for key, item in value.items() if key != "identity"})
    return value


def semantic_block(failed_receipt, projected_manifest, arm):
    for value in (failed_receipt, projected_manifest):
        _verified_identity(value)
        if value.get("GT_input") is not False or value.get("failed_view_replacement") is not False:
            raise ValueError("technical block requires unchanged label-free selected views")
    scene = projected_manifest["scene_id"]
    if (arm not in ("G1", "G3") or arm not in failed_receipt.get("arms", [])
            or failed_receipt["scene"] != scene or failed_receipt["status"] != "FAILED"
            or "BLOCKED_ALL_SEMANTIC_REQUESTS_FAILED" not in failed_receipt.get("error", "")):
        raise ValueError("technical block requires an actual same-scene all-failed semantic receipt")
    views, g1 = projected_manifest["views"], projected_manifest["g1"]
    if set(views) != set(g1) or any(g1[owner] != rows[:1] for owner, rows in views.items()):
        raise ValueError("selected G1 must remain the exact G3 prefix")
    selected = [rid for owner in sorted(views) for rid in (g1 if arm == "G1" else views)[owner]]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("technical block needs nonempty unique selected requests; no-view is a valid result")
    failures = []
    for rid in selected:
        expected = projected_manifest["requests"].get(rid)
        actual = failed_receipt.get("requests", {}).get(rid)
        if (expected is None or actual is None or actual.get("status") != "UNAVAILABLE_TECHNICAL_FAILURE"
                or not actual.get("reason") or expected.get("scene") != scene
                or any(actual.get(key) != expected.get(key)
                       for key in ("request_id", "frame_id", "target_mask_sha256"))):
            raise ValueError("every exact selected mask must have a recorded technical failure")
        failures.append({key: actual[key] for key in
                         ("request_id", "frame_id", "target_mask_sha256", "status", "reason")})
    return _seal({"schema": "compact-selected-semantic-block-v1", "status": "BLOCKED_TECHNICAL",
        "scene": scene, "recovery_source": ARM_SOURCE[arm], "selected_request_ids": selected,
        "failed_requests": failures, "failed_receipt_identity": failed_receipt["identity"],
        "projected_manifest_identity": projected_manifest["identity"],
        "GT_input": False, "failed_view_replacement": False})


def _methods(spec, scene):
    if scene in spec["cohorts"]["replica8"] or scene == spec["smoke_scene"]:
        cohort = "replica8"
    elif scene in spec["cohorts"]["scannet_cf18"]:
        cohort = "scannet_cf18"
    else:
        raise ValueError("method scope leaves the fixed scene set")
    return [row for row in spec["methods"] if cohort in row["cohorts"]]


def partition_methods(spec, scene, available_sources, blocked_sources):
    methods = _methods(spec, scene)
    required = {row["recovery"] for row in methods} - {"NONE"}
    available, blocked = set(available_sources), set(blocked_sources)
    if available & blocked or available | blocked != required:
        raise ValueError("recovery source partition must cover every fixed dependency exactly once")
    for name, block in blocked_sources.items():
        _verified_identity(block)
        if (block.get("schema") != "compact-selected-semantic-block-v1" or block["status"] != "BLOCKED_TECHNICAL"
                or block["scene"] != scene or block["recovery_source"] != name
                or not block["selected_request_ids"] or block["GT_input"] is not False
                or block["failed_view_replacement"] is not False
                or [row["request_id"] for row in block["failed_requests"]] != block["selected_request_ids"]
                or any(row["status"] != "UNAVAILABLE_TECHNICAL_FAILURE" or not row["reason"]
                       for row in block["failed_requests"])):
            raise ValueError("method partition requires proven exact-source technical blocks")
    selected, missing = [], {}
    for row in methods:
        name = row["recovery"]
        if name in blocked:
            missing[row["id"]] = {"status": "BLOCKED_TECHNICAL", "method": row["id"], "scene": scene,
                "recovery_source": name, "block_identity": blocked_sources[name]["identity"],
                "reason": "ALL_SELECTED_SEMANTIC_REQUESTS_FAILED"}
        else:
            selected.append(row)
    return selected, missing


def validate_partial_scope(spec, receipt):
    if receipt["status"] not in (PARTIAL_LOCK, PARTIAL_RESULT):
        raise ValueError("partial scope requires an explicitly partial receipt")
    methods, blocked = partition_methods(spec, receipt["scene"], receipt["available_recovery_sources"],
                                         receipt["blocked_sources"])
    ids = [row["id"] for row in methods]
    if (not blocked or receipt["expected_method_ids"] != [row["id"] for row in _methods(spec, receipt["scene"])]
            or receipt["blocked_methods"] != blocked):
        raise ValueError("partial scope changed the fixed expected methods or blocked dependencies")
    if receipt["status"] == PARTIAL_LOCK:
        if set(receipt["predictions"]) != set(ids) or set(receipt["prediction_identities"]) != set(ids):
            raise ValueError("partial prediction scope must contain every independent method exactly once")
    else:
        rows = receipt["rows"]
        if (len(rows) != len(ids) or {row["method"] for row in rows} != set(ids)
                or any(row["status"] != "COMPLETE" or row["scene"] != receipt["scene"]
                       or row["rank_mode"] != evaluation.RANK_MODE for row in rows)):
            raise ValueError("partial evaluation scope must contain every independent same-scene method exactly once")
    return ids


def _scope_fields(spec, scene, available, blocks):
    selected, blocked = partition_methods(spec, scene, available, blocks)
    return selected, {"expected_method_ids": [row["id"] for row in _methods(spec, scene)],
        "available_recovery_sources": sorted(available), "blocked_sources": blocks, "blocked_methods": blocked}


def _receipt(path, scene, status, index):
    index.identity(path)
    value = read(path)
    _verified_identity(value)
    if value["scene"] != scene or value["status"] != status:
        raise ValueError("independent prerequisite lacks its actual same-scene required status")
    for item in value.get("inputs", []) + value.get("outputs", []):
        index.identity(item["path"], item)
    return value


def lock_partial_outputs(binding, scene, output_root, *, context=None):
    freeze = require_frozen_execution(binding, scene)
    spec, task, root = load_spec(binding["spec"]), Path(binding["output_root"]), Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    index = ConsumptionIndex(root / "input_verifications.json")
    inputs = load_recovery_inputs(binding, scene, context=context, index=index)
    projected = _receipt(task / "projected_views" / scene / "receipt.json", scene, "COMPLETE", index)
    manifest = read(projected["manifest"])
    _verified_identity(manifest)
    registry = read(projected["registry"])
    _verified_identity(registry)
    failed = _receipt(task / "recovery" / scene / "receipt.json", scene, "FAILED", index)
    blocks = {ARM_SOURCE[arm]: semantic_block(failed, manifest, arm) for arm in ("G1", "G3")
              if ARM_SOURCE[arm] in {row["recovery"] for row in _methods(spec, scene)}}
    native = _receipt(task / "native_recovery" / scene / "receipt.json", scene, "COMPLETE", index)
    sources = {"G1_NATIVE": read(native["source"]["path"])}
    source_receipts = {"G1_NATIVE": native}
    standalone = []
    if "ARCHIVED_U2_FC" in {row["recovery"] for row in _methods(spec, scene)}:
        u2 = _receipt(task / "recovery" / (scene + "_U2") / "receipt.json", scene, "COMPLETE", index)
        if u2["arms"] != ["U2"]:
            raise ValueError("independent archived source must be its actual U2-only production call")
        sources["ARCHIVED_U2_FC"] = read(u2["sources"]["U2"]["path"])
        source_receipts["ARCHIVED_U2_FC"] = u2["regions"]
        standalone.append(u2)
    for source in sources.values():
        _verified_identity(source)
    methods, scope = _scope_fields(spec, scene, sources, blocks)
    costs = {name: recovery_usage(source, source_receipts[name], name) for name, source in sources.items()}
    identity = canonical_digest({"binding": binding["identity"], "resident_inputs": inputs.identity,
        "freeze": freeze, "registry": registry["identity"], "projected": projected["identity"], "scope": scope,
        "sources": {name: source["identity"] for name, source in sources.items()},
        "costs": {name: value["identity"] for name, value in costs.items()},
        "producer": index.identity(__file__)})
    path = root / "receipt.json"
    if path.is_file():
        old = read(path)
        _verified_identity(old)
        if old["status"] == PARTIAL_LOCK:
            if old["input_identity"] != identity:
                raise ValueError("completed independent prediction inputs changed; invalidate explicitly")
            validate_partial_scope(spec, old)
            for item in old["inputs"] + old["outputs"]:
                index.identity(item["path"], item)
            for value in old["predictions"].values():
                load_prediction(value)
            return old
        path.rename(root / ("receipt.failed_" + str(time.time_ns()) + ".json"))
    started = time.monotonic()
    result = {"status": "RUNNING", "scene": scene, "input_identity": identity, "freeze": freeze,
        "cohort": "replica8" if scene in spec["cohorts"]["replica8"] or scene == spec["smoke_scene"] else "scannet_cf18",
        "GT_input": False, **scope}
    atomic_write_json(path, result)
    try:
        outputs = build_method_outputs(inputs.baseline, inputs.raw, registry, inputs.sources,
            binding["final_temperatures"], inputs.valid_ids, inputs.nearest, inputs.matched,
            sources, methods, cost_inventory=inputs.cost_inventory, recovery_costs=costs)
        a0 = outputs["CT_A0_NATIVE"]
        if (not np.array_equal(a0.owner_ids, inputs.baseline.owner_ids)
                or not np.array_equal(a0.semantic_labels, inputs.baseline.semantic_labels)
                or a0.instance_ranks != inputs.baseline.instance_ranks):
            raise RuntimeError("independent A0 changed exact Native arrays or official ranks")
        if scene in spec["cohorts"]["replica8"]:
            index.identity(inputs.data["predictions"]["D2"])
            d2 = load_prediction(inputs.data["predictions"]["D2"])
            if (not np.array_equal(outputs["CT_A1_E"].owner_ids, d2.owner_ids)
                    or not np.array_equal(outputs["CT_A1_E"].semantic_labels, d2.semantic_labels)):
                raise RuntimeError("independent A1 changed frozen Replica D2 decisions")
        for receipt in standalone:
            for method, exported in receipt["exports"].items():
                if (outputs[method].record_key != exported["record_key"]
                        or outputs[method].prediction_key != exported["prediction_key"]):
                    raise RuntimeError("independent final output differs from its unchanged production export")
        decisions, audit = fuse_readout(inputs.sources, binding["final_temperatures"], "D2",
                                        inputs.valid_ids, owner_labels(inputs.baseline))
        if {str(owner): label for owner, label in decisions.items()} != outputs["CT_A1_E"].metadata["owner_semantic_decisions"]:
            raise RuntimeError("independent D2 probability audit changed the actual original operator")
        audit_path = root / "existing_D2_probability_audit.json"
        atomic_write_json(audit_path, {"scene": scene, "original_operator": "backbone_wave1.readouts.fuse_readout",
            "decision_identity": outputs["CT_A1_E"].metadata["existing_D2_decision_identity"],
            "probabilities_are_not_official_instance_confidence": True, "owners": audit})
        files, predictions, keys = [index.identity(audit_path)], {}, {}
        for method, payload in outputs.items():
            prediction_path = save_prediction(payload, root / method)
            record = read(prediction_path)
            files.extend([index.identity(prediction_path),
                          index.identity(prediction_path.parent / record["arrays"]["path"], record["arrays"])])
            predictions[method] = str(prediction_path)
            keys[method] = {"record_key": payload.record_key, "prediction_key": payload.prediction_key}
        result.update(status=PARTIAL_LOCK, predictions=predictions, prediction_identities=keys, outputs=files,
            output_count=len(predictions), required_output_count=len(scope["expected_method_ids"]),
            baseline_array_and_rank_parity=True, final_temperature_vector=binding["final_temperatures"],
            geometry_identity=inputs.baseline.geometry.to_dict(), base_cost_inventory=inputs.cost_inventory,
            recovery_cost_identities={name: value["identity"] for name, value in costs.items()})
        validate_partial_scope(spec, result)
    except BaseException as exc:
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", outputs=[])
        raise
    finally:
        result.update(inputs=index.entries(), elapsed_seconds=time.monotonic() - started)
        atomic_write_json(path, _seal(result))
        index.write_memo(root / "input_verifications.json")
    return result


def _adapt(function, replacements, namespace):
    original = ast.parse(inspect.getsource(function))
    targets = [(ast.dump(ast.parse(before, mode=mode).body if mode == "eval" else ast.parse(before).body[0],
                         include_attributes=False), ast.parse(after, mode=mode).body if mode == "eval"
                else ast.parse(after).body, count) for before, after, mode, count in replacements]
    counts = [0] * len(targets)

    class Bridge(ast.NodeTransformer):
        def visit(self, node):
            for position, (target, replacement, _) in enumerate(targets):
                if ast.dump(node, include_attributes=False) == target:
                    counts[position] += 1
                    value = copy.deepcopy(replacement)
                    if isinstance(value, list):
                        return [ast.copy_location(item, node) for item in value]
                    return ast.copy_location(value, node)
            return super().visit(node)

    adapted = Bridge().visit(copy.deepcopy(original))
    if counts != [count for _, _, count in targets]:
        raise ValueError("unchanged production function differs from the exact partial-scope bridge")
    ast.fix_missing_locations(adapted)
    proof = {"original_AST": canonical_digest(ast.dump(original)), "effective_AST": canonical_digest(ast.dump(adapted)),
             "changed_AST_nodes": sum(counts), "inherited_function": function.__module__ + "." + function.__name__}
    scope = {**vars(evaluation), **namespace, "__file__": __file__, "_bridge_proof": proof}
    exec(compile(adapted, function.__code__.co_filename, "exec"), scope)
    return scope[function.__name__], proof


def _evaluation_scope(lock):
    return {key: lock[key] for key in
            ("expected_method_ids", "available_recovery_sources", "blocked_sources", "blocked_methods")}


def evaluate_partial_scene(binding, scene, prediction_lock, output_root, *, context=None):
    replacements = [
        ('result.update(status="COMPLETE", rows=rows, registry_checks=registry_checks, outputs=outputs, '
         'metric_unit="FRACTION", row_count=len(rows), aggregation="SCENE")',
         'result.update(status=PARTIAL_RESULT, rows=rows, registry_checks=registry_checks, outputs=outputs, '
         'metric_unit="FRACTION", row_count=len(rows), aggregation="SCENE", '
         'partial_bridge=_bridge_proof, **_evaluation_scope(prediction_lock))', "exec", 1),
        ('"PREDICTIONS_LOCKED"', repr(PARTIAL_LOCK), "eval", 1),
        ('"COMPLETE"', repr(PARTIAL_RESULT), "eval", 1),
        ('methods = [row["id"] for row in spec["methods"] if cohort in row["cohorts"]]',
         'methods = validate_partial_scope(spec, prediction_lock)', "exec", 1)]
    function, _ = _adapt(evaluation.evaluate_scene, replacements,
        {"validate_partial_scope": validate_partial_scope, "_evaluation_scope": _evaluation_scope,
         "PARTIAL_RESULT": PARTIAL_RESULT})
    return function(binding, scene, prediction_lock, output_root, context=context)


def select_partial_pool_rows(spec, cohort, method, receipts):
    if cohort not in spec["cohorts"] or method not in [row["id"] for row in spec["methods"] if cohort in row["cohorts"]]:
        raise ValueError("pool method leaves its fixed cohort")
    scenes = spec["cohorts"][cohort]
    if set(receipts) != set(scenes):
        raise ValueError("official pool requires the complete fixed scene set")
    selected = []
    for scene in scenes:
        receipt = receipts[scene]
        if receipt["scene"] != scene:
            raise ValueError("pool receipt changed its fixed scene")
        if receipt["status"] == PARTIAL_RESULT:
            available = validate_partial_scope(spec, receipt)
            if method not in available:
                raise ValueError("BLOCKED_TECHNICAL: full-cohort method has a proven blocked scene: " + scene + "/" + method)
        elif receipt["status"] == "COMPLETE":
            ids = [row["id"] for row in _methods(spec, scene)]
            if len(receipt["rows"]) != len(ids) or {row["method"] for row in receipt["rows"]} != set(ids):
                raise ValueError("complete evaluation scope must retain every fixed method")
        else:
            raise ValueError("pool requires completed scoring or an explicit proven partial scope")
        rows = [row for row in receipt["rows"] if row["method"] == method and row["rank_mode"] == evaluation.RANK_MODE]
        if len(rows) != 1 or rows[0]["scene"] != scene or rows[0]["status"] != "COMPLETE":
            raise ValueError("each fixed scene-method must have exactly one complete official row")
        selected.append(rows[0])
    return selected


def _pool_partition(spec, cohort, receipts):
    methods, blocked = [], {}
    if set(receipts) != set(spec["cohorts"][cohort]):
        raise ValueError("partial pool still requires the complete fixed scene set")
    for row in spec["methods"]:
        if cohort not in row["cohorts"]:
            continue
        missing = {}
        for scene, receipt in receipts.items():
            if receipt["status"] == PARTIAL_RESULT:
                validate_partial_scope(spec, receipt)
                if row["id"] in receipt["blocked_methods"]:
                    missing[scene] = receipt["blocked_methods"][row["id"]]
            elif receipt["status"] != "COMPLETE":
                raise ValueError("pool may not infer a technical block from incomplete scoring")
        if missing:
            blocked[row["id"]] = {"status": "BLOCKED_TECHNICAL", "method": row["id"], "cohort": cohort,
                "blocked_scenes": missing, "required_scene_order": spec["cohorts"][cohort]}
        else:
            select_partial_pool_rows(spec, cohort, row["id"], receipts)
            methods.append(row["id"])
    return methods, blocked


def _finish_pool(receipt, spec, cohort, receipts, proof):
    methods, blocked = _pool_partition(spec, cohort, receipts)
    if not blocked or set(receipt["methods"]) != set(methods):
        raise ValueError("partial pool cannot hide unproven missing methods")
    receipt.update(expected_method_ids=[row["id"] for row in spec["methods"] if cohort in row["cohorts"]],
        blocked_methods=blocked, partial_bridge=proof, complete_pool_count=len(methods),
        required_pool_count=len(methods) + len(blocked))
    _seal(receipt)


def _released_pool_with_classes(root, rows, namespace, scenes, method, rank_mode, index):
    root = Path(root)
    path = root / (method + "_per_class.json")
    if not path.is_file():
        by_scene = {row["scene"]: row for row in rows if row["method"] == method and row["rank_mode"] == rank_mode}
        if len(by_scene) != len(rows) or set(by_scene) != set(scenes):
            raise ValueError("per-class alias still requires the full exact ordered scene set")
        scores = []
        for scene in scenes:
            row = by_scene[scene]
            index.identity(row["evaluation_receipt"])
            score = read(row["evaluation_receipt"])
            if score["status"] != "COMPLETE" or score["identity"] != row["evaluation_identity"]:
                raise ValueError("per-class alias changed its actual completed scoring input")
            scores.append(score)
        context = scores[0]["context"]
        identity = canonical_digest({"ordered_inputs": [score["identity"] for score in scores], "rank_mode": rank_mode,
            "evaluator": index.identity(context["evaluator"]["path"], context["evaluator"]),
            "producer": index.identity(evaluation.__file__), "semantic_ids": context["valid_ids"],
            "semantic_names": context["class_names"]})
        for candidate in sorted(root.glob("*_per_class.json")):
            existing = read(candidate)
            _verified_identity(existing)
            if existing["input_identity"] != identity:
                continue
            if existing["status"] != "COMPLETE" or existing["scene_order"] != list(scenes):
                raise ValueError("exact per-class alias lacks its completed full-scene proof")
            alias = {**existing, "method": method, "exact_per_class_alias": index.identity(candidate),
                     "exact_per_class_alias_identity": existing["identity"]}
            atomic_write_json(path, _seal(alias))
            break
    return evaluation.released_pool_with_classes(root, rows, namespace, scenes, method, rank_mode, index)


def pool_partial_cohort(binding, cohort):
    spec = load_spec(binding["spec"])
    if cohort not in spec["cohorts"]:
        raise ValueError("pool leaves the fixed main cohorts")
    freeze = require_frozen_execution(binding, spec["cohorts"][cohort][0])
    root = Path(binding["output_root"])
    output = root / "pools" / cohort
    index = ConsumptionIndex(output / "input_verifications.json")
    parents = [index.identity(root / "evaluation" / scene / "receipt.json") for scene in spec["cohorts"][cohort]]
    identity = canonical_digest({"binding": binding["identity"], "cohort": cohort, "freeze": freeze,
        "ordered_parents": parents, "producer": index.identity(__file__),
        "inherited_evaluation": index.identity(evaluation.__file__)})
    path = output / "receipt.json"
    if path.is_file():
        old = read(path)
        _verified_identity(old)
        if old["status"] == PARTIAL_RESULT:
            if old["input_identity"] != identity:
                raise ValueError("completed partial pool changed its exact inputs; invalidate explicitly")
            receipts = {scene: read(root / "evaluation" / scene / "receipt.json") for scene in spec["cohorts"][cohort]}
            for receipt in receipts.values():
                _verified_identity(receipt)
            methods, blocked = _pool_partition(spec, cohort, receipts)
            if set(old["methods"]) != set(methods) or old["blocked_methods"] != blocked:
                raise ValueError("completed partial pool changed its fixed method scope")
            for item in old["inputs"] + old["outputs"]:
                index.identity(item["path"], item)
            return old
        raise ValueError("partial pool must not overwrite an existing incompatible aggregate")
    replacements = [
        ('methods = [row["id"] for row in spec["methods"] if cohort in row["cohorts"]]',
         'methods = _pool_partition(spec, cohort, receipts)[0]', "exec", 1),
        ('"COMPLETE"', repr(PARTIAL_RESULT), "eval", 1),
        ('receipt["identity"] = canonical_digest(receipt)',
         '_finish_pool(receipt, spec, cohort, receipts, _bridge_proof)', "exec", 1)]
    function, _ = _adapt(evaluation.pool_cohort, replacements,
        {"select_pool_rows": select_partial_pool_rows, "_pool_partition": _pool_partition,
         "_finish_pool": _finish_pool, "released_pool_with_classes": _released_pool_with_classes})
    result = function(binding, cohort)
    result.update(input_identity=identity, freeze=freeze)
    atomic_write_json(path, _seal(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--operation", choices=("predict", "evaluate", "pool"), required=True)
    parser.add_argument("--scene")
    parser.add_argument("--lock")
    parser.add_argument("--output-root")
    parser.add_argument("--context")
    parser.add_argument("--cohort", choices=("replica8", "scannet_cf18"))
    args = parser.parse_args()
    binding = read(args.binding)
    if args.operation == "pool":
        if not args.cohort:
            parser.error("pool requires --cohort")
        result = pool_partial_cohort(binding, args.cohort)
    else:
        if not args.scene or not args.output_root:
            parser.error("scene operation requires --scene and --output-root")
        context = read(args.context) if args.context else None
        if args.operation == "predict":
            result = lock_partial_outputs(binding, args.scene, args.output_root, context=context)
        else:
            if not args.lock:
                parser.error("evaluate requires --lock")
            result = evaluate_partial_scene(binding, args.scene, read(args.lock), args.output_root, context=context)
    print(result.get("scene", result.get("cohort")), result["status"], flush=True)
