"""Single resident-model cold recovery measurements and scientific parity."""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .projected_views import _verified_identity
from .protocol import experiment_matrix, load_spec
from .recovery_run import ARM_METHOD, ARM_SOURCE, gpu_lease, load_recovery_inputs, recover_fc
from .region_worker import FCSession
from .runtime import require_frozen_execution


PRODUCTION_CALLABLE = "static_ovmap.cvpr_compact.recovery_run.recover_fc"
FEATURE_TOLERANCE = {"atol": 1e-5, "rtol": 1e-5, "dtype": "float32"}


def recovery_arm(name):
    if name in ARM_METHOD:
        return name
    reverse = {source: arm for arm, source in ARM_SOURCE.items()}
    if name not in reverse:
        raise ValueError("unknown fixed FC timing arm")
    return reverse[name]


def _document(value, index):
    if isinstance(value, (str, Path)):
        index.identity(value)
        value = read(value)
    _verified_identity(value)
    return value


def _features(record, index):
    index.identity(record["path"], record)
    with np.load(record["path"], allow_pickle=False) as arrays:
        ids, features = arrays["request_ids"], arrays["features"]
    if (ids.ndim != 1 or features.ndim != 2 or len(ids) != len(features)
            or len(set(ids.tolist())) != len(ids) or features.dtype != np.float32
            or not np.isfinite(features).all()):
        raise ValueError("cold parity requires complete finite FP32 per-request features")
    return dict(zip(ids.tolist(), features, strict=True))


def _seal(value):
    value["identity"] = canonical_digest({key: item for key, item in value.items() if key != "identity"})
    return value


def _verified_parent(parent, scene, arm, index):
    from .partial_execution import semantic_block

    _verified_identity(parent)
    if parent["scene"] != scene or parent["arm"] != arm:
        raise ValueError("cold preflight parent leaves its exact fixed scene/arm")
    core = recovery_arm(arm)
    if parent["status"] == "BLOCKED_TECHNICAL_PARENT":
        if core not in ("G1", "G3"):
            raise ValueError("projected semantic block cannot replace archived U2 evidence")
        documents = []
        for key in ("failed_receipt", "projected_manifest"):
            item = parent[key]
            index.identity(item["path"], item)
            documents.append(_document(read(item["path"]), index))
        actual = semantic_block(*documents, core)
        if parent["technical_block"] != actual:
            raise ValueError("cold block differs from the actual unchanged selected-mask failure")
        return None
    if parent["status"] != "COMPLETE_PARENT":
        raise ValueError("cold preflight cannot infer a valid scientific parent from a missing prerequisite")
    item = parent["receipt"]
    index.identity(item["path"], item)
    scientific = _document(read(item["path"]), index)
    if (scientific["status"] != "COMPLETE" or scientific["scene"] != scene
            or scientific["identity"] != parent["scientific_identity"]
            or scientific.get("cold") is not False or scientific.get("GT_input") is not False
            or core not in scientific["arms"] or core not in scientific["sources"]
            or core not in scientific["plan"] or ARM_METHOD[core] not in scientific["exports"]):
        raise ValueError("cold scientific parent lacks its actual complete same-arm production evidence")
    for item in scientific["inputs"] + scientific["outputs"]:
        index.identity(item["path"], item)
    item = scientific["sources"][core]
    index.identity(item["path"], item)
    source = _document(read(item["path"]), index)
    plan = scientific["plan"][core]
    if set(source["objects"]) != set(plan):
        raise ValueError("cold preflight changed the complete candidate registry")
    attempted, used = [], []
    for owner, row in source["objects"].items():
        if (row["attempted_request_ids"] != plan[owner]
                or not set(row["used_request_ids"]) <= set(plan[owner])
                or (row["available"] and not row["used_request_ids"])):
            raise ValueError("cold preflight changed actual selected requests or source availability")
        attempted.extend(row["attempted_request_ids"])
        used.extend(row["used_request_ids"])
    if attempted and not used:
        raise ValueError("all selected requests failed in this arm; a complete warm aggregate cannot certify a scientific zero")
    return scientific


def build_scientific_parent_plan(binding):
    from .partial_execution import validate_partial_scope

    spec, root = load_spec(binding["spec"]), Path(binding["output_root"])
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    scenes, arms = spec["cohorts"][spec["timing"]["cohort"]], spec["timing"]["arms"]
    parents, complete, blocked = {}, 0, 0
    for scene in scenes:
        path = root / "recovery" / scene / "receipt.json"
        scientific = _document(path, index)
        lock = None
        if scientific["status"] != "COMPLETE":
            lock_path = root / "predictions" / scene / "receipt.json"
            lock = _document(lock_path, index)
            validate_partial_scope(spec, lock)
            if lock["scene"] != scene:
                raise ValueError("cold parent partition changed its actual prediction-lock scene")
        parents[scene] = {}
        for arm in arms:
            if lock is not None and arm in lock["blocked_sources"]:
                parent = {"status": "BLOCKED_TECHNICAL_PARENT", "scene": scene, "arm": arm,
                    "technical_block": lock["blocked_sources"][arm], "failed_receipt": index.identity(path),
                    "projected_manifest": index.identity(root / "projected_views" / scene / "manifest.json")}
                blocked += 1
            else:
                actual_path = path if lock is None else root / "recovery" / (scene + "_U2") / "receipt.json"
                actual = _document(actual_path, index)
                parent = {"status": "COMPLETE_PARENT", "scene": scene, "arm": arm,
                    "receipt": index.identity(actual_path), "scientific_identity": actual["identity"]}
                complete += 1
            parents[scene][arm] = _seal(parent)
            _verified_parent(parent, scene, arm, index)
    result = {"schema": "compact-cold-parent-plan-v1", "binding_identity": binding["identity"],
        "matrix_identity": experiment_matrix(spec)["identity"], "scene_order": scenes, "arm_order": arms,
        "status": "PARTIAL_WITH_TECHNICAL_BLOCKS" if blocked else "COMPLETE", "parents": parents,
        "complete_parent_count": complete, "blocked_parent_count": blocked,
        "required_leaf_count": len(scenes) * len(arms), "inputs": index.entries()}
    index.write_memo(root / "validation/input_verifications.json")
    return _seal(result)


def _preflight_parents(binding, spec, scientific_receipts, index):
    scenes, arms = spec["cohorts"][spec["timing"]["cohort"]], spec["timing"]["arms"]
    if scientific_receipts.get("schema") == "compact-cold-parent-plan-v1":
        _verified_identity(scientific_receipts)
        if (scientific_receipts["binding_identity"] != binding["identity"]
                or scientific_receipts["matrix_identity"] != experiment_matrix(spec)["identity"]
                or scientific_receipts["scene_order"] != scenes or scientific_receipts["arm_order"] != arms):
            raise ValueError("cold parent plan changed its fixed binding, matrix or order")
        parents = scientific_receipts["parents"]
    else:
        if set(scientific_receipts) != set(scenes):
            raise ValueError("cold timing requires all eight fixed scientific recovery parents")
        parents = {}
        for scene in scenes:
            scientific = _document(scientific_receipts[scene], index)
            parents[scene] = {arm: _seal({"status": "COMPLETE_PARENT", "scene": scene, "arm": arm,
                "receipt": index.identity(scientific_receipts[scene]), "scientific_identity": scientific["identity"]})
                for arm in arms}
    if set(parents) != set(scenes) or any(set(parents[scene]) != set(arms) for scene in scenes):
        raise ValueError("cold preflight must cover all24 fixed scene/arm leaves")
    documents = {scene: {} for scene in scenes}
    for scene in scenes:
        for arm in arms:
            documents[scene][arm] = _verified_parent(parents[scene][arm], scene, arm, index)
    return parents, documents


def _validate_unmeasured_block(row):
    block = row["technical_block"]
    _verified_identity(block)
    if (row["status"] != "BLOCKED_UNMEASURED" or row["measured_seconds"] is not None
            or row["parity"] is not None or row["physical_calls_reserved"] != 0
            or row["measurement_reserved"] is not False or row["cold_call_executed"] is not False
            or block.get("schema") != "compact-selected-semantic-block-v1"
            or block.get("GT_input") is not False or block.get("failed_view_replacement") is not False
            or block["status"] != "BLOCKED_TECHNICAL" or block["scene"] != row["scene"]
            or block["recovery_source"] != row["arm"] or not block["selected_request_ids"]
            or len(set(block["selected_request_ids"])) != len(block["selected_request_ids"])
            or [item["request_id"] for item in block["failed_requests"]] != block["selected_request_ids"]
            or any(item["status"] != "UNAVAILABLE_TECHNICAL_FAILURE" or not item["reason"]
                   for item in block["failed_requests"])):
        raise ValueError("blocked timing must retain proven unmeasured status without a reserved or executed call")


def record_blocked_measurement(binding, scene, arm, parent, path, index):
    if parent["status"] != "BLOCKED_TECHNICAL_PARENT" or _verified_parent(parent, scene, arm, index) is not None:
        raise ValueError("unmeasured cold leaf requires its actual technical block")
    path = Path(path)
    producers = [index.identity(__file__), index.identity(Path(__file__).with_name("recovery_run.py"))]
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "arm": arm,
        "parent": parent["identity"], "producers": producers, "measurement_reserved": False})
    if path.is_file():
        previous = read(path)
        if previous.get("status") != "BLOCKED_UNMEASURED" or previous.get("input_identity") != identity:
            raise ValueError("unmeasured block must not overwrite a reserved, observed or incompatible cold call")
        _verified_identity(previous)
        _validate_unmeasured_block(previous)
        for item in previous["inputs"]:
            index.identity(item["path"], item)
        return previous
    value = {"status": "BLOCKED_UNMEASURED", "binding_identity": binding["identity"], "scene": scene,
        "arm": arm, "input_identity": identity, "technical_block": parent["technical_block"],
        "scientific_parent_identity": parent["identity"], "planned_gpu": str(binding["gpu"]), "hardware": None,
        "measured_seconds": None, "parity": None, "measurement_reserved": False, "physical_calls_reserved": 0,
        "cold_call_executed": False, "planned_production_callable": PRODUCTION_CALLABLE,
        "unavailable_reason": "ALL_SELECTED_SEMANTIC_REQUESTS_FAILED_IN_FIXED_SCIENTIFIC_PARENT",
        "inputs": [parent["failed_receipt"], parent["projected_manifest"], *producers], "outputs": []}
    atomic_write_json(path, _seal(value))
    return value


def compare_recovery_parity(scientific, cold, arm, *, index=None):
    arm = recovery_arm(arm)
    index = index or ConsumptionIndex()
    scientific, cold = (_document(value, index) for value in (scientific, cold))
    if (arm not in ARM_METHOD or scientific["status"] != "COMPLETE" or cold["status"] != "COMPLETE"
            or scientific["scene"] != cold["scene"]
            or scientific["resident_input_identity"] != cold["resident_input_identity"]
            or scientific["registry_identity"] != cold["registry_identity"]
            or scientific["plan"][arm] != cold["plan"][arm]):
        raise ValueError("cold recovery changed its baseline, candidates or fixed view selections")
    sources = []
    for record in (scientific, cold):
        item = record["sources"][arm]
        index.identity(item["path"], item)
        sources.append(_document(read(item["path"]), index))
    first, second = sources
    if first["valid_ids"] != second["valid_ids"] or set(first["objects"]) != set(second["objects"]):
        raise ValueError("cold recovery changed its complete candidate or class registry")
    attempted, used = set(), set()
    for owner, row in first["objects"].items():
        other = second["objects"][owner]
        for key in ("available", "label", "attempted_request_ids", "used_request_ids", "failed_request_ids"):
            if row[key] != other[key]:
                raise ValueError("cold recovery changed candidate availability, labels or retained requests")
        attempted.update(row["attempted_request_ids"])
        used.update(row["used_request_ids"])
        if row["available"] and not np.allclose(row["scores"], other["scores"],
                atol=FEATURE_TOLERANCE["atol"], rtol=FEATURE_TOLERANCE["rtol"]):
            raise ValueError("cold recovery classification scores exceeded FP32 feature tolerance")
    warm_vectors, cold_vectors = (_features(row["features"], index) for row in (scientific, cold))
    if set(cold_vectors) != used or set(warm_vectors) & attempted != used:
        raise ValueError("cold recovery changed successful selected per-view features")
    maximum = 0.
    for rid in sorted(used):
        first_vector, second_vector = warm_vectors[rid], cold_vectors[rid]
        if (first_vector.shape != second_vector.shape or not np.allclose(first_vector, second_vector,
                atol=FEATURE_TOLERANCE["atol"], rtol=FEATURE_TOLERANCE["rtol"])):
            raise ValueError("cold recovery per-view feature exceeded declared FP32 tolerance")
        maximum = max(maximum, float(np.max(np.abs(first_vector - second_vector))))
    predictions = []
    method = ARM_METHOD[arm]
    for row in (scientific, cold):
        export = row["exports"][method]
        for item in export["files"]:
            index.identity(item["path"], item)
        payload = load_prediction(export["manifest"])
        if payload.method_id != method or payload.scene_id != row["scene"] or not payload.locked:
            raise ValueError("cold recovery export left its fixed locked scene/method")
        predictions.append(payload)
    first, second = predictions
    if first.geometry != second.geometry or not np.array_equal(first.owner_ids, second.owner_ids):
        raise ValueError("cold recovery changed exported geometry or full instance support")
    if not np.array_equal(first.semantic_labels, second.semantic_labels):
        raise ValueError("cold recovery changed exported semantic labels")
    if first.instance_ranks != second.instance_ranks:
        raise ValueError("cold recovery changed original current-class instance ranks")
    result = {"status": "PASS", "arm": arm, "scene": scientific["scene"],
        "scientific_identity": scientific["identity"], "cold_identity": cold["identity"],
        "compared_request_count": len(used), "feature_tolerance": dict(FEATURE_TOLERANCE),
        "maximum_feature_absolute_difference": maximum, "full_support_exact": True,
        "semantic_labels_exact": True, "current_ranks_exact": True}
    result["identity"] = canonical_digest(result)
    return result


def aggregate_timings(spec, receipts):
    expected = [(row["scene"], row["arm"]) for row in experiment_matrix(spec)["timings"]]
    rows = {}
    hardware = set()
    for row in receipts:
        _verified_identity(row)
        key = (row["scene"], row["arm"])
        if key in rows:
            raise ValueError("duplicate cold timing leaf; exactly24 distinct measurements are required")
        if row["status"] == "BLOCKED_UNMEASURED":
            _validate_unmeasured_block(row)
            rows[key] = row
            continue
        if (row["status"] != "COMPLETE" or row["parity"]["status"] != "PASS"
                or row["production_callable"] != PRODUCTION_CALLABLE or not row["resident_model"]
                or any(row[name] for name in ("persistent_feature_cache_enabled",
                    "persistent_view_cache_enabled", "persistent_result_cache_enabled"))):
            raise ValueError("cold timing leaf is incomplete or violates its measured production contract")
        seconds = row["measured_seconds"]
        if isinstance(seconds, bool) or not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("every real scene measurement, including empty candidates, must be positive")
        device = row["hardware"]
        hardware.add((device["uuid"], device["name"], device["compute_capability"]))
        rows[key] = row
    if set(rows) != set(expected):
        raise ValueError("complete cold timing requires exactly24 fixed scene/arm leaves")
    complete = [rows[key] for key in expected if rows[key]["status"] == "COMPLETE"]
    blocked = [rows[key] for key in expected if rows[key]["status"] == "BLOCKED_UNMEASURED"]
    if complete and len(hardware) != 1:
        raise ValueError("the timing mean requires one physical GPU and architecture")
    arms = {}
    scenes = spec["cohorts"][spec["timing"]["cohort"]]
    for arm in spec["timing"]["arms"]:
        selected = [rows[(scene, arm)] for scene in scenes]
        measured = [row for row in selected if row["status"] == "COMPLETE"]
        available = len(measured) == len(scenes)
        total = sum(row["measured_seconds"] for row in measured)
        arms[arm] = {"mean_seconds": total / len(scenes) if available else None, "sum_seconds": total if available else None,
            "scene_count": len(scenes), "scene_order": list(scenes),
            "complete_scene_count": len(measured), "blocked_scene_count": len(scenes) - len(measured),
            "receipt_identities": [row["identity"] for row in selected],
            "measured_sum_seconds": total,
            "unavailable_reason": None if available else "FULL_EIGHT_SCENE_MEAN_HAS_BLOCKED_UNMEASURED_LEAVES"}
    result = {"status": "PARTIAL_WITH_TECHNICAL_BLOCKS" if blocked else "COMPLETE", "leaf_count": len(complete),
        "planned_leaf_count": len(rows), "blocked_leaf_count": len(blocked), "arms": arms,
        "hardware": dict(complete[0]["hardware"]) if complete else None, "none_seconds": 0.,
        "none_basis": "NO_RECOVERY_BY_DEFINITION", "empty_candidate_scenes_measured": True,
        "definition": "FEATURE_CACHE_COLD_INCREMENTAL_RECOVERY_MODEL_RESIDENT"}
    result["identity"] = canonical_digest(result)
    return result


def _process(pid):
    path = Path("/proc") / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (path / "cmdline").read_bytes().decode().split("\0")[:-1]
    except FileNotFoundError:
        return None
    return {"pid": int(pid), "state": fields[0], "start_ticks": fields[19], "argv": argv,
            "live": fields[0] not in ("Z", "X")}


def assert_serial_execution(binding):
    root = Path(binding["output_root"])
    background = root / "execution/anchors/background_controller.json"
    if background.is_file():
        row = read(background)
        actual = _process(row["pid"])
        if (actual and actual["live"] and actual["start_ticks"] == row["process_start_ticks"]
                and actual["argv"] == row["argv"]):
            raise RuntimeError("cold timing must wait for the live anchor controller to finish")
    for path in (root / "execution").glob("**/command_*.json"):
        for row in read(path).get("attempts", []):
            if row.get("status") != "RUNNING" or "child_pid" not in row:
                continue
            actual = _process(row["child_pid"])
            if actual and actual["live"] and actual["pid"] != os.getpid() and actual["argv"] == row["argv"]:
                raise RuntimeError("cold timing must serialize against this task's live mapping/evaluation workers")


def reserve_measurement(path, binding_identity, scene, arm, input_identity):
    path = Path(path)
    with exclusive_lock(path.parent / ".measurement.lock"):
        if path.exists():
            raise RuntimeError("cold measurement already reserved; never replay an observed or uncertain call")
        process = _process(os.getpid())
        row = {"status": "RESERVED", "binding_identity": binding_identity, "scene": scene, "arm": arm,
            "input_identity": input_identity, "physical_calls_reserved": 1, "controller_pid": os.getpid(),
            "controller_start_ticks": process["start_ticks"], "reserved_at_unix": time.time()}
        atomic_write_json(path, row)
    return row


def _hardware(gpu):
    result = subprocess.run(["nvidia-smi", "-i", str(gpu),
        "--query-gpu=uuid,name,compute_cap,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("timing GPU identity is not one physical device")
    return dict(zip(("uuid", "name", "compute_capability", "driver_version"), rows[0], strict=True))


def run_timings(binding, scientific_receipts, *, output_root=None):
    import torch

    spec = load_spec(binding["spec"])
    scenes = spec["cohorts"][spec["timing"]["cohort"]]
    for scene in scenes:
        require_frozen_execution(binding, scene)
    assert_serial_execution(binding)
    root = Path(output_root or Path(binding["output_root"]) / "timing").resolve()
    index = ConsumptionIndex(root / "input_verifications.json")
    parents, scientific_documents = _preflight_parents(binding, spec, scientific_receipts, index)
    hardware = _hardware(binding["gpu"])
    measurements, resident = [], None
    with exclusive_lock(root / ".controller.lock"), gpu_lease(binding, root) as lease:
        for scene in scenes:
            inputs = load_recovery_inputs(binding, scene, index=index)
            session, model_load_reference, new_model_load_seconds = None, None, 0.
            for arm in spec["timing"]["arms"]:
                core_arm = recovery_arm(arm)
                leaf = root / scene / arm
                path = leaf / "receipt.json"
                scientific = scientific_documents[scene][arm]
                if scientific is None:
                    row = record_blocked_measurement(binding, scene, arm, parents[scene][arm], path, index)
                    measurements.append(row)
                    print(f"Cold {scene}/{arm}: BLOCKED_UNMEASURED; no physical call reserved", flush=True)
                    continue
                input_identity = canonical_digest({"binding": binding["identity"], "resident_inputs": inputs.identity,
                    "scientific": scientific["identity"], "arm": arm, "hardware": hardware,
                    "producer": index.identity(__file__), "production_operator": index.identity(Path(__file__).with_name("recovery_run.py"))})
                if path.is_file():
                    row = read(path)
                    if row["input_identity"] != input_identity:
                        raise ValueError("an already reserved cold measurement changed inputs; replay is forbidden")
                    if row["status"] == "COMPLETE":
                        _verified_identity(row)
                        for item in row["inputs"] + row["outputs"]:
                            index.identity(item["path"], item)
                        measurements.append(row)
                        continue
                    if row["status"] == "MEASURED":
                        _finish_parity(row, path, scientific, index)
                        measurements.append(read(path))
                        continue
                    actual = _process(row["controller_pid"])
                    live = bool(actual and actual["live"] and actual["start_ticks"] == row["controller_start_ticks"])
                    raise RuntimeError(f"cold measurement is {row['status']}; controller_live={live}; a second physical call is forbidden")
                assert_serial_execution(binding)
                if session is None:
                    session = FCSession(binding, inputs.data, index, cache=None)
                    if resident is not None:
                        if (session.model_key, session.text_identity["sha256"], list(session.ids)) != resident[:3]:
                            raise ValueError("resident timing model/text identity changed between Replica scenes")
                        session.model, session.weight_audit = resident[3], resident[4]
                        model_load_reference = resident[5]
                    else:
                        session.load_model()
                        model_load = {"model_identity": session.model_key, "text_identity": session.text_identity,
                            "valid_ids": list(map(int, session.ids)), "hardware": hardware,
                            "seconds": session.model_load_seconds, "weight_audit": session.weight_audit,
                            "excluded_from_incremental_timer": True, "loaded_at_unix": time.time()}
                        model_load["identity"] = canonical_digest(model_load)
                        model_load_path = root / "model_loading" / (str(time.time_ns()) + ".json")
                        atomic_write_json(model_load_path, model_load)
                        model_load_reference = index.identity(model_load_path)
                        new_model_load_seconds = session.model_load_seconds
                        resident = (session.model_key, session.text_identity["sha256"], list(session.ids),
                                    session.model, session.weight_audit, model_load_reference)
                row = reserve_measurement(path, binding["identity"], scene, arm, input_identity)
                row.update(hardware=hardware, resident_model=True, model_load_seconds=new_model_load_seconds,
                    model_load_reference=model_load_reference,
                    model_load_included=False, production_callable=PRODUCTION_CALLABLE,
                    persistent_feature_cache_enabled=False, persistent_view_cache_enabled=False,
                    persistent_result_cache_enabled=False, frozen_common_input_identity=inputs.identity,
                    common_input_policy="RESIDENT_ANCHOR_AND_EXISTING_NQF;NO_NEW_RECOVERY_FEATURES_OR_VIEWS",
                    include=list(spec["timing"]["include"]), exclude=list(spec["timing"]["exclude"]),
                    status="CALL_STARTED", started_at_unix=time.time())
                new_model_load_seconds = 0.
                atomic_write_json(path, row)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                try:
                    result = recover_fc(binding, inputs, leaf / "call", arms=(core_arm,), session=session, cold=True, lease=lease)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    row.update(status="MEASURED", measured_seconds=elapsed, recovery_receipt=str(leaf / "call/receipt.json"),
                        recovery_identity=result["identity"], peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        candidate_count=result["candidate_count"], source_available_additions=result["source_available_additions"][core_arm],
                        physical_image_encodings=result["physical_image_encodings"],
                        physical_region_poolings=result["physical_region_poolings"],
                        inputs=index.entries(), outputs=[index.identity(leaf / "call/receipt.json"), *result["outputs"]])
                    atomic_write_json(path, row)
                except BaseException as exc:
                    row.update(status="FAILED_MEASUREMENT", observed_elapsed_seconds=time.perf_counter() - started,
                        error=f"{type(exc).__name__}: {exc}", outputs=[], inputs=index.entries())
                    failed_path = leaf / "call/receipt.json"
                    if failed_path.is_file():
                        failed = read(failed_path)
                        _verified_identity(failed)
                        row.update(failed_recovery_receipt=index.identity(failed_path),
                            failed_recovery_identity=failed["identity"], outputs=[index.identity(failed_path), *failed["outputs"]],
                            failed_recovery_costs={key: failed[key] for key in
                                ("physical_image_encodings", "physical_region_poolings", "requests", "elapsed_seconds") if key in failed})
                    row["identity"] = canonical_digest({k: v for k, v in row.items() if k != "identity"})
                    atomic_write_json(path, row)
                    raise
                _finish_parity(row, path, scientific, index)
                measurements.append(read(path))
                print(f"Cold {scene}/{arm}: {elapsed:.3f}s, parity PASS", flush=True)
        result = aggregate_timings(spec, measurements)
        model_loads = [read(path) for path in sorted((root / "model_loading").glob("*.json"))]
        for event in model_loads:
            _verified_identity(event)
        result.update(model_loading_events=model_loads,
                      model_loading_seconds_total=sum(event["seconds"] for event in model_loads),
                      inputs=index.entries(), outputs=[index.identity(root / row["scene"] / row["arm"] / "receipt.json")
                          for row in measurements] + [item for row in measurements for item in row["outputs"]])
        result["identity"] = canonical_digest({k: v for k, v in result.items() if k != "identity"})
        atomic_write_json(root / "pool.json", result)
        index.write_memo(root / "input_verifications.json")
    return result


def _finish_parity(row, path, scientific, index):
    try:
        for item in row["inputs"] + row["outputs"]:
            index.identity(item["path"], item)
        row["parity"] = compare_recovery_parity(scientific, row["recovery_receipt"], row["arm"], index=index)
        row["status"] = "COMPLETE"
    except BaseException as exc:
        row.update(status="PARITY_FAILED", parity={"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        row["identity"] = canonical_digest({k: v for k, v in row.items() if k != "identity"})
        atomic_write_json(path, row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--scientific-receipts", required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    result = run_timings(read(args.binding), read(args.scientific_receipts), output_root=args.output_root)
    print("Cold timing:", result["status"], result["leaf_count"], flush=True)
