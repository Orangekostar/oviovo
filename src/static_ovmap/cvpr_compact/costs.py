"""Method dependency unions and separate, once-per-worker physical payments."""

from pathlib import Path

import numpy as np

from static_ovmap.module_validation.contracts import canonical_digest
from static_ovmap.recovery_wave2.binding import PathResolver, read

from .projected_views import _verified_identity


CONTENT_FIELDS = ("native_contents", "FC_dense_contents", "FC_mask_contents")


def _seal(value):
    value["identity"] = canonical_digest({key: row for key, row in value.items() if key != "identity"})
    return value


def _contents(value):
    result = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key or isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("required content must have nonempty identities and positive integer input counts")
        result[key] = count
    return result


def _request(row):
    if row.get("status") not in ("COMPLETE", "UNAVAILABLE_TECHNICAL_FAILURE"):
        raise ValueError("cost accounting requires an actual terminal request outcome")
    result = {"status": row["status"]}
    if row["status"] != "COMPLETE":
        result["reason"] = row.get("reason", row.get("error", row.get("failure_reason")))
        if not result["reason"]:
            raise ValueError("failed paid request lacks its technical failure reason")
    return result


def _usage(native=None, dense=None, masks=None, requests=None, unresolved=()):
    value = {"native_contents": _contents(native or {}), "FC_dense_contents": _contents(dense or {}),
        "FC_mask_contents": _contents(masks or {}), "requests": requests or {},
        "unresolved_content_requests": sorted(set(unresolved))}
    partial = bool(value["unresolved_content_requests"])
    native_inputs, images = sum(value["native_contents"].values()), sum(value["FC_dense_contents"].values())
    value.update(status="PARTIAL_CONTENT_IDENTITIES" if partial else "COMPLETE",
        native_required_crop_inputs=native_inputs, FC_required_image_inputs=images,
        FC_selected_mask_inputs=sum(value["FC_mask_contents"].values()),
        required_image_encodings=None if partial else native_inputs + images,
        known_required_image_encodings_lower_bound=native_inputs + images,
        logical_attempted_requests=len(value["requests"]),
        failed_request_count=sum(row["status"] != "COMPLETE" for row in value["requests"].values()))
    return _seal(value)


def native_usage(contents, rows, stage):
    requests = {}
    for row in rows:
        key = stage + ":" + row["request_id"]
        if key in requests:
            raise ValueError("a native/query request was charged twice within one stage")
        requests[key] = _request(row)
    return _usage(native=contents, requests=requests)


def fc_usage(receipt, request_ids, stage, *, manifest=None, dense_records=None):
    selected = list(request_ids)
    if len(set(selected)) != len(selected) or not set(selected) <= set(receipt["requests"]):
        raise ValueError("FC cost inputs leave the actual preselected request ledger")
    dense, masks, outcomes, unresolved = {}, {}, {}, []
    for rid in selected:
        row = receipt["requests"][rid]
        key = stage + ":" + rid
        outcomes[key] = _request(row)
        tensor = row.get("input_tensor_key")
        image = row.get("image_content_key", row.get("image_content_identity"))
        mask = row.get("target_mask_sha256")
        if manifest is not None:
            request = manifest["requests"][rid]
            if mask is not None and mask != request["target_mask_sha256"]:
                raise ValueError("FC cost mask differs from its actual selected request")
            mask = request["target_mask_sha256"]
        if tensor is None and image in (dense_records or {}):
            record = dense_records[image]
            if record["image_content_key"] != image:
                raise ValueError("legacy FC dense receipt belongs to a different physical image")
            tensor = record["input_tensor_key"]
        if manifest is not None and image is not None and "content_identity" in row:
            legacy_key = canonical_digest({"image": image, "mask": mask, "region": "original_signed_mask_pooling"})
            if legacy_key != row["content_identity"]:
                raise ValueError("legacy FC mask content differs from its selected request")
        if tensor is not None:
            dense[tensor] = 1
        if tensor is not None and mask is not None:
            masks[canonical_digest({"image": tensor, "mask": mask, "region": "original_signed_mask_pooling"})] = 1
        else:
            unresolved.append(key)
    return _usage(dense=dense, masks=masks, requests=outcomes, unresolved=unresolved)


def union_usage(*parts):
    contents, requests, unresolved = {name: {} for name in CONTENT_FIELDS}, {}, set()
    for part in parts:
        _verified_identity(part)
        for name in CONTENT_FIELDS:
            for key, count in _contents(part[name]).items():
                if key in contents[name] and contents[name][key] != count:
                    raise ValueError("one physical content identity has inconsistent required crop/input counts")
                contents[name][key] = count
        for key, row in part["requests"].items():
            if key in requests and requests[key] != row:
                raise ValueError("one selected request has inconsistent cost outcomes")
            requests[key] = row
        unresolved.update(part["unresolved_content_requests"])
    return _usage(contents["native_contents"], contents["FC_dense_contents"],
                  contents["FC_mask_contents"], requests, unresolved)


def method_costs(method, inventory, recovery):
    _verified_identity(inventory)
    if inventory["status"] != "COMPLETE":
        raise ValueError("method dependencies need the complete bound base cost inventory")
    components, parts = ["N_SUPPORT"], [inventory["support"]]
    existing = method["existing"]
    if existing == "D2":
        components.append("Q")
        parts.append(inventory["query"])
    elif existing not in ("NATIVE", "FC_ONLY_SAME_EVIDENCE"):
        raise ValueError("unknown existing-evidence method cost recipe")
    if existing in ("D2", "FC_ONLY_SAME_EVIDENCE"):
        components.append("F")
        parts.append(inventory["F"])
    if method["recovery"] != "NONE":
        if recovery is None:
            raise ValueError("nonempty recovery recipe lacks its selected-request cost evidence")
        components.append(method["recovery"])
        parts.append(recovery)
    elif recovery is not None:
        raise ValueError("no-recovery recipe cannot acquire recovery cost inputs")
    value = {key: row for key, row in union_usage(*parts).items() if key != "identity"}
    value.update(method_id=method["id"], scene=inventory["scene"],
        base_inventory_identity=inventory["identity"], required_components=components,
        native_support_prerequisite_included=True, common_map_receipt=inventory["common_map_receipt"],
        query_attempts=len(inventory["query"]["requests"]) if existing == "D2" else 0,
        recovery_attempted_requests=len(recovery["requests"]) if recovery is not None else 0,
        standalone_wall_seconds=None, timing_basis="LOGICAL_CONTENT_UNION_NOT_A_COLD_WALL_TIME",
        mask_input_basis="PRESELECTED_MASKS_INCLUDING_FAILURES;NOT_PHYSICAL_POOLING_CALLS",
        method_costs_may_be_summed=False)
    return _seal(value)


def physical_stage(stage, receipt, file_identity, *, current_task):
    if receipt["status"] not in ("COMPLETE", "FAILED", "FAILED_MEASUREMENT"):
        raise ValueError("physical cost accounting requires a terminal worker receipt")
    requests = receipt.get("requests", receipt.get("native_requests", {}))
    rows = requests.values() if isinstance(requests, dict) else requests
    failures = [row for row in rows if row.get("status") == "UNAVAILABLE_TECHNICAL_FAILURE"]
    count = receipt.get("physical_image_encodings", receipt.get("physical_image_inputs"))
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
        raise ValueError("physical image/crop payments must be nonnegative integer counters")
    for name in ("elapsed_seconds", "model_load_seconds"):
        value = receipt.get(name)
        if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value < 0):
            raise ValueError("physical cost times must be observed finite nonnegative durations")
    return _seal({"stage": stage, "status": receipt["status"], "receipt": file_identity,
        "source_origin": "CURRENT_TASK" if current_task else "READ_ONLY_HISTORICAL_PARENT",
        "current_task_physical_image_inputs": count if current_task else 0,
        "historical_physical_image_inputs": 0 if current_task else count,
        "current_task_physical_region_poolings": receipt.get("physical_region_poolings") if current_task else 0,
        "historical_physical_region_poolings": 0 if current_task else receipt.get("physical_region_poolings"),
        "physical_encoder_batch_calls": receipt.get("encoder_batch_calls", receipt.get("physical_encoder_calls")),
        "source_worker_wall_seconds": receipt.get("elapsed_seconds"),
        "model_load_seconds": receipt.get("model_load_seconds"), "failed_request_count": len(failures),
        "Q_logical_ledger": receipt.get("query_logical_ledger"),
        "failed_requests": [{key: row[key] for key in ("request_id", "status", "reason", "error") if key in row}
                            for row in failures],
        "method_costs_may_be_summed": False, "table3_cold_wall_time": False})


def load_base_cost_inventory(binding, data, *, index):
    resolver = PathResolver(binding["path_map"])
    consumed = []
    expected = {resolver.resolve(row["path"]): row for row in binding["inputs"]}

    def consume(path, known=None):
        path = resolver.resolve(path)
        item = index.identity(path, known or expected.get(path))
        consumed.append(item)
        value = read(path)
        if "identity" in value:
            _verified_identity(value)
        return resolver.rewrite(value)

    nq = consume(data["native_query_receipt"])
    fc = consume(Path(data["fc_root"]) / "receipt.json")
    manifest = consume(Path(data["fc_root"]).parent / "fc_requests.json")
    decisions = consume(nq["query_decisions"]["path"], nq["query_decisions"])
    if (nq["status"] != "COMPLETE" or fc["status"] != "COMPLETE"
            or nq["map_id"] != "BB00_NATIVE"
            or fc["physical_model_identity"] != data["FC_physical_model_identity"]):
        raise ValueError("base cost inventory requires the exact complete Native/Q and static FC receipts")
    if fc.get("request_manifest", manifest["identity"]) != manifest["identity"]:
        raise ValueError("base FC cost receipt differs from its original selected request manifest")
    if manifest["native_record_key"] != read(data["predictions"]["NATIVE_READOUT"])["record_key"]:
        raise ValueError("static FC costs come from a different common Native support")
    if not set(nq["native_required_content"]) <= set(nq["standalone_required_content"]):
        raise ValueError("Native support content is absent from the actual N/Q physical union")
    query_rows = [{"request_id": row["request_id"], "status": "COMPLETE" if row["success"] else "UNAVAILABLE_TECHNICAL_FAILURE",
                   "reason": row["failure_reason"]} for frame in decisions["frames"] for row in frame["results"]]
    if len(query_rows) != nq["query_logical_ledger"]["attempts"] or decisions["logical"] != nq["query_logical_ledger"]:
        raise ValueError("query paid-request costs differ from their original logical ledger")
    if set(manifest["requests"]) != set(fc["requests"]):
        raise ValueError("static FC cost ledger omitted a preselected request, including failures")
    dense = {image: consume(path) for image, path in fc["required_dense_receipts"].items()}
    images = {row.get("image_content_key", row.get("image_content_identity")) for row in fc["requests"].values()
              if row.get("input_tensor_key") is None}
    missing_images = images - set(dense) - {None}
    # The original worker registers direct dense dependencies only after a successful pooling.
    for item in fc.get("inputs", []) if missing_images else ():
        array_path = Path(item["path"])
        if array_path.parent.name != "dense" or array_path.suffix != ".npz":
            continue
        record = consume(array_path.with_suffix(".json"))
        image = record["image_content_key"]
        if image not in missing_images:
            continue
        if (record["arrays"] != item or record["input_tensor_key"] != array_path.stem
                or array_path.parent.parent.name != fc["physical_model_identity"]):
            raise ValueError("failed FC dense dependency differs from its original paid array/model input")
        index.identity(array_path, item)
        dense[image] = record
        missing_images.remove(image)
        if not missing_images:
            break
    for record in dense.values():
        index.identity(resolver.resolve(record["arrays"]["path"]), record["arrays"])
    map_path = resolver.resolve(data["parent_map_receipt"])
    mapping = consume(map_path)
    if mapping["status"] != "COMPLETE" or mapping["scene"] != manifest["scene_id"] or mapping["map_id"] != "BB00_NATIVE":
        raise ValueError("cost inventory must retain the actual same-scene common Native map prerequisite")
    value = {"status": "COMPLETE", "scene": manifest["scene_id"],
        "support": native_usage(nq["native_required_content"], nq["native_requests"], "N_SUPPORT"),
        "query": native_usage(nq["standalone_required_content"], query_rows, "Q"),
        "F": fc_usage(fc, fc["requests"], "F", manifest=manifest, dense_records=dense),
        "common_map_receipt": index.identity(map_path), "source_receipts": consumed,
        "common_mapping_is_required": True, "new_neural_inference": 0}
    return _seal(value)


def recovery_usage(source, receipt, stage):
    selected = [rid for row in source["objects"].values() for rid in row["attempted_request_ids"]]
    if len(set(selected)) != len(selected):
        raise ValueError("one recovery request was assigned to multiple candidate owners")
    if stage == "G1_NATIVE":
        if set(selected) != set(receipt["requests"]):
            raise ValueError("same-view Native cost ledger differs from the actual G1 plan")
        return native_usage(receipt.get("required_content", {}),
            [{**receipt["requests"][rid], "request_id": rid} for rid in selected], stage)
    return fc_usage(receipt, selected, stage)


PHYSICAL_UNITS = {"N_Q_SHARED": "NATIVE_CROP", "NATIVE_RECOVERY": "NATIVE_CROP",
    "FC_STATIC_SHARED": "FC_IMAGE", "FC_RECOVERY_SHARED": "FC_IMAGE", "FC_RECOVERY_COLD": "FC_IMAGE",
    "CROPFORMER_SHARED": "CROPFORMER_IMAGE", "COMMON_MAPPING": None}


def physical_ledger(records):
    workers, unique = [], {}
    buckets = {name: {unit: [] for unit in ("NATIVE_CROP", "FC_IMAGE", "CROPFORMER_IMAGE")}
               for name in ("current_task_warm", "current_task_cold", "historical_parent")}
    pooling = {name: [] for name in buckets}
    for record in records:
        path = record["file"]["path"]
        if path in unique:
            if unique[path] != record:
                raise ValueError("one physical worker receipt was assigned inconsistent cost provenance")
            continue
        unique[path] = record
        stage = record["stage"]
        if stage not in PHYSICAL_UNITS:
            raise ValueError("unknown physical worker cost scope")
        row = physical_stage(stage, record["receipt"], record["file"], current_task=record["current_task"])
        row.update(scene=record["scene"], image_input_unit=PHYSICAL_UNITS[stage], cold=record["cold"],
            model_loading_scope="RESIDENT_MODEL_LOAD_REPORTED_SEPARATELY" if record["cold"] else "SOURCE_WORKER_LOAD")
        if record["cold"]:
            row["resident_session_model_load_seconds_metadata"] = row["model_load_seconds"]
            row["model_load_seconds"] = None
        row = _seal(row)
        workers.append(row)
        bucket = "historical_parent" if not record["current_task"] else "current_task_cold" if record["cold"] else "current_task_warm"
        unit = PHYSICAL_UNITS[stage]
        if unit is not None:
            value = row["historical_physical_image_inputs"] if bucket == "historical_parent" else row["current_task_physical_image_inputs"]
            buckets[bucket][unit].append(value)
        if unit == "FC_IMAGE":
            pooling[bucket].append(row["historical_physical_region_poolings"] if bucket == "historical_parent"
                                   else row["current_task_physical_region_poolings"])
    result = {"status": "COMPLETE", "workers": workers, "logical_method_budgets_may_be_summed": False,
        "worker_wall_times_are_not_pipeline_elapsed_time": True,
        "cold_resident_model_loading_excluded_from_each_call": True}
    for bucket, values in buckets.items():
        result[bucket] = {unit: None if any(row is None for row in rows) else sum(rows) for unit, rows in values.items()}
        result[bucket + "_known_lower_bound"] = {unit: sum(row for row in rows if row is not None) for unit, rows in values.items()}
        rows = pooling[bucket]
        result[bucket + "_FC_region_poolings"] = None if any(row is None for row in rows) else sum(rows)
        if any(value is None for value in result[bucket].values()) or result[bucket + "_FC_region_poolings"] is None:
            result["status"] = "PARTIAL_UNMEASURED_FAILED_COST"
    return _seal(result)


def collect_physical_costs(binding, scenes, *, index):
    root, records = Path(binding["output_root"]).resolve(), []
    resolver = PathResolver(binding["path_map"])

    def collect(scene, stage, path, *, cold=False, archived=True):
        path = Path(resolver.resolve(path)).resolve()
        candidates = sorted(path.parent.glob(path.stem + ".failed_*.json")) if archived else []
        candidates.append(path)
        for candidate in candidates:
            item = index.identity(candidate)
            receipt = read(candidate)
            if "identity" in receipt:
                _verified_identity(receipt)
            records.append({"scene": scene, "stage": stage, "receipt": receipt, "file": item,
                            "current_task": candidate.is_relative_to(root), "cold": cold})

    for scene in scenes:
        path = root / "contexts" / (scene + ".json")
        if path.is_file():
            index.identity(path)
            data = read(path)
            _verified_identity(data)
        else:
            data = binding["scenes"][scene]
        collect(scene, "N_Q_SHARED", data["native_query_receipt"])
        collect(scene, "FC_STATIC_SHARED", Path(data["fc_root"]) / "receipt.json")
        collect(scene, "COMMON_MAPPING", data["parent_map_receipt"])
        collect(scene, "FC_RECOVERY_SHARED", root / "recovery" / scene / "receipt.json")
        collect(scene, "NATIVE_RECOVERY", root / "native_recovery" / scene / "receipt.json")
        if scene in binding["cohorts"]["scannet_cf18"]:
            collect(scene, "CROPFORMER_SHARED", root / "frontend" / scene / "cropformer/receipt.json")
        if scene in binding["cohorts"]["replica8"]:
            for arm in ("ARCHIVED_U2_FC", "G1_FC", "G3_FC"):
                collect(scene, "FC_RECOVERY_COLD", root / "timing" / scene / arm / "call/receipt.json", cold=True, archived=False)
    result = physical_ledger(records)
    loading = []
    for path in sorted((root / "timing/model_loading").glob("*.json")):
        item = index.identity(path)
        event = read(path)
        _verified_identity(event)
        if not event["excluded_from_incremental_timer"] or not np.isfinite(event["seconds"]) or event["seconds"] < 0:
            raise ValueError("cold model loading must retain its separate actual payment record")
        loading.append({"receipt": item, "event": event})
    result.update(cold_model_loading_events=loading,
                  cold_model_loading_seconds_total=sum(row["event"]["seconds"] for row in loading))
    result.update(scene_order=list(scenes), shared_workers_counted_once=True)
    return _seal(result)
