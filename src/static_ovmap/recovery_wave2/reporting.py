"""Dynamic publication evidence from complete fixed cohorts and locked predictions."""

from collections import Counter
import gzip
import json
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest

from .binding import ConsumptionIndex, PathResolver, read
from .costs import light_costs, map_cost
from .light import METHODS, require_transfer_freeze
from .mechanisms import light_scene, native_actions, paired_classification, recovery_funnel, sam_actions
from .selection import feasible, net_gain
from .workflow import A_MAPS, S_MAPS


METRICS = ("apall", "ap50", "ap25", "miou", "macc")
COMPACT = "artifacts/static_ovmap/recovery_wave2_v1"


def _pool(path, scenes):
    result = read(path)
    if result["status"] != "COMPLETE" or result["scene_order"] != scenes:
        raise ValueError("report cannot replace an incomplete fixed cohort with a partial average")
    return result


def _gzip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
        zipped.write(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())


def _table(columns, rows):
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |" for row in rows]])


def _number(value, *, percent=False):
    return "NA" if value is None else f"{value * (100 if percent else 1):.4f}"


def collect_matrix(binding):
    root = Path(binding["output_root"])
    freeze = read(root / "freeze/receipt.json")
    selection = read(root / "selection/final.json")
    rows, pools = [], []
    for cohort, scenes in binding["datasets"].items():
        baseline = _pool(root / "light/pools" / cohort / "BB00_NATIVE/RW_B_D2.json", scenes)["metrics"]
        methods = [*METHODS, "RW_LIGHT_COMBO"] if cohort == "development" else list(METHODS)
        if cohort == "replica" and freeze["light_package"]["id"] not in METHODS:
            methods.append("RW_FROZEN_LIGHT")
        for method in methods:
            path = root / "light/pools" / cohort / "BB00_NATIVE" / (method + ".json")
            result = _pool(path, scenes)
            pools.append({"cohort": cohort, "map_id": "BB00_NATIVE", "path": str(path), **result})
            rows.append({"cohort": cohort, "map_id": "BB00_NATIVE", "method": method,
                "status": "COMPLETE", "coverage": f"{len(scenes)}/{len(scenes)}", "scene_order": scenes,
                "metrics": result["metrics"], "delta_pp": {key: 100 * (result["metrics"][key] - baseline[key]) for key in METRICS},
                "feasible": feasible(result["metrics"], baseline), "NET_GAIN": net_gain(result["metrics"], baseline),
                "pool_identity": result["identity"], "pool_path": str(path), "aggregation": "RELEASED_DATASET_POOL"})
        parent_native = Path(binding["parent_root"]) / "pools" / cohort / "BB00_NATIVE/NATIVE_READOUT/OFFICIAL_CURRENT_CLASS.json"
        result = _pool(parent_native, scenes)
        pools.append({"cohort": cohort, "map_id": "BB00_NATIVE", "path": str(parent_native), **result})
        rows.append({"cohort": cohort, "map_id": "BB00_NATIVE", "method": "NATIVE_READOUT", "status": "COMPLETE",
            "coverage": f"{len(scenes)}/{len(scenes)}", "scene_order": scenes, "metrics": result["metrics"],
            "delta_pp": {key: 100 * (result["metrics"][key] - baseline[key]) for key in METRICS}, "pool_identity": result["identity"]})
        for item in selection["map_inputs"]:
            map_id = item["id"]
            status = item["geometry_status"] if cohort == "development" else "NOT_RUN_NOT_FROZEN_NOMINEE"
            if cohort == "replica" and freeze["nominated_map"] == map_id:
                status = "GEOMETRY_PASS"
            for method in ("NATIVE_READOUT", "FC_EQ", "D2"):
                if status == "GEOMETRY_PASS":
                    path = root / "pools" / cohort / map_id / method / "OFFICIAL_CURRENT_CLASS.json"
                    result = _pool(path, scenes)
                    pools.append({"cohort": cohort, "map_id": map_id, "path": str(path), **result})
                    metrics, status_row, coverage = result["metrics"], "COMPLETE", f"{len(scenes)}/{len(scenes)}"
                elif status == "EQUIVALENT_INPUT":
                    path = Path(binding["parent_root"]) / "pools" / cohort / "BB00_NATIVE" / method / "OFFICIAL_CURRENT_CLASS.json"
                    result = _pool(path, scenes)
                    metrics, status_row, coverage = result["metrics"], "EQUIVALENT_INPUT_EXACT_PARENT_ALIAS", f"{len(scenes)}/{len(scenes)}"
                else:
                    metrics, status_row, coverage = None, "NOT_RUN_RESOURCE_SCREEN" if status == "SCREENED_OUT_GEOMETRY" else status, f"0/{len(scenes)}"
                    result = None
                rows.append({"cohort": cohort, "map_id": map_id, "method": method, "status": status_row,
                    "coverage": coverage, "scene_order": scenes, "metrics": metrics,
                    "delta_pp": None if metrics is None else {key: 100 * (metrics[key] - baseline[key]) for key in METRICS},
                    "NET_GAIN": None if metrics is None else net_gain(metrics, baseline),
                    "geometry_status": status, "pool_identity": result["identity"] if result else None})
    return rows, pools


def scene_evidence(binding):
    root, rows = Path(binding["output_root"]), []
    resolver = PathResolver(binding["path_map"])
    paths = sorted((root / "light").glob("*/*/evaluation_rows.json"))
    paths += sorted((root / "light").glob("*/*/extra_conditions/*/evaluation_rows.json"))
    paths += sorted((root / "readouts").glob("*/*/evaluation_rows.json"))
    for path in paths:
        source = read(path)
        if source["status"] != "COMPLETE":
            raise ValueError("compact scene evidence cannot include an unfinished scorer")
        for row in source["rows"]:
            receipt = resolver.rewrite(read(row["evaluation_receipt"]))
            matrix = np.asarray(receipt["confusion"], np.int64)
            sparse = [[int(i), int(j), int(matrix[i, j])] for i, j in zip(*np.nonzero(matrix), strict=True)]
            rows.append({**row, "evaluation_identity": receipt["identity"], "source_rows_receipt": str(path),
                "actual_official_view": receipt["view"], "trace_parity": receipt["trace_parity"],
                "runtime_overlaps": receipt["context"]["runtime_overlaps"],
                "class_order": receipt["context"]["valid_ids"], "confusion_shape": list(matrix.shape),
                "confusion_sparse_index_count": sparse, "confusion_identity": canonical_digest(receipt["confusion"])})
    return rows


def _compact_light(row):
    result = dict(row)
    match = dict(result["matcher"])
    actual = dict(match["actual"])
    added = set(row["method_source_available_owners"])
    actual["tp_matches"] = [item for item in actual["tp_matches"] if set(item["possible_owners"]) & added]
    actual["fp_entries"] = [item for item in actual["fp_entries"] if set(item["possible_owners"]) & added]
    actual["compact_entries_scope"] = "ALL_ADDED_OWNER_CONTRIBUTIONS; COMPLETE_INCUMBENT_TRACE_EXTERNAL"
    match["actual"], result["matcher"] = actual, match
    return result


def collect_mechanisms(binding):
    root, light, funnels, paired, actions, sam = Path(binding["output_root"]), [], [], [], [], []
    sources, decisions, geometry = {}, {}, {"screens": {}, "scenes": []}
    for cohort, scenes in binding["datasets"].items():
        for scene in scenes:
            light.extend(_compact_light(row) for row in light_scene(binding, scene))
            funnels.extend(recovery_funnel(binding, scene))
            paired.append(paired_classification(binding, scene))
            source_root = root / "recovery" / scene / "BB00_NATIVE"
            cached, fc = read(source_root / "cached_sources.json"), read(source_root / "fc_recovery_receipt.json")
            sources[scene] = {"valid_ids": cached["valid_ids"], "cached": cached["sources"],
                "FC": {arm: read(value["path"])["objects"] for arm, value in fc["sources"].items()},
                "registry": read(source_root / "registry.json"), "request_plan": read(source_root / "fc_request_plan.json")}
            light_root = root / "light" / scene / "BB00_NATIVE"
            decisions[scene] = {path.stem: read(path) for path in sorted((light_root / "decisions").glob("*.json"))}
            for directory in sorted((light_root / "extra_conditions").glob("*")):
                if (directory / "evaluation_rows.json").is_file():
                    light.extend(_compact_light(row) for row in light_scene(binding, scene, subdir="extra_conditions/" + directory.name))
                    decisions[scene].update({path.stem: read(path) for path in sorted((directory / "decisions").glob("*.json"))})
    for map_id in (*A_MAPS, *S_MAPS):
        screen = read(root / "geometry/screens" / (map_id + ".json"))
        geometry["screens"][map_id] = screen
        if screen["status"] == "EQUIVALENT_INPUT":
            continue
        for scene in binding["datasets"]["development"]:
            diagnostic = read(root / "geometry" / scene / map_id / "diagnostic.json")
            geometry["scenes"].append(diagnostic)
            if map_id in A_MAPS:
                action = native_actions(binding, scene, map_id)
                atomic_write_json(root / "mechanisms" / f"{scene}_{map_id}.json", action)
                examples = [row for row in action["frames"] if row["duplicate_owner_followers"] or row["mixed_fallback_constrained_candidate_visits"]]
                actions.append({**{key: value for key, value in action.items() if key != "frames"},
                    "actual_frame_count": len(action["frames"]), "examples": examples[:3] or action["frames"][:2],
                    "complete_frame_evidence": str(root / "mechanisms" / f"{scene}_{map_id}.json")})
            else:
                action = sam_actions(binding, scene, map_id)
                atomic_write_json(root / "mechanisms" / f"{scene}_{map_id}.json", action)
                sam.append({**{key: value for key, value in action.items() if key != "S2_frames"},
                    "S2_frame_count": len(action["S2_frames"]),
                    "complete_frame_evidence": str(root / "mechanisms" / f"{scene}_{map_id}.json")})
    coverage = {cohort: read(root / "mechanisms" / (cohort + "_common_coverage.json")) for cohort in binding["datasets"]}
    return {"light": light, "paired_U1_U2": paired, "association": actions, "SAM": sam,
            "same_covered_subset": coverage}, funnels, sources, decisions, geometry


def collect_costs(binding):
    root, lights, maps, operations, failures = Path(binding["output_root"]), [], [], [], []
    for scene in binding["scenes"]:
        lights.extend(light_costs(binding, scene)["methods"].values())
    for path in sorted((root / "readouts").glob("*/*/receipt.json")):
        receipt = read(path)
        if receipt["status"] != "PREDICTIONS_LOCKED":
            raise ValueError("cost reporting requires actual locked own-map readouts")
        maps.append(map_cost(binding, receipt["scene"], receipt["map_id"]))
    paths = sorted((root / "recovery").glob("*/*/fc_recovery_receipt*.json"))
    paths += sorted((root / "readouts").glob("*/*/fc/receipt*.json"))
    paths += sorted((root / "readouts").glob("*/*/native_query/native_query_receipt*.json"))
    paths += sorted((root / "confirmations").glob("**/fc_recovery_receipt*.json"))
    keys = ("physical_image_encodings", "physical_encoder_calls", "encoder_batch_calls", "physical_region_poolings",
            "physical_text_inputs", "new_text_forwards", "elapsed_seconds", "model_load_seconds", "dense_cache_hits",
            "region_cache_hits", "native_physical_image_encodings", "peak_gpu_allocated_bytes")
    for path in paths:
        receipt = read(path)
        kind = "NATIVE_Q" if path.name.startswith("native_query") else "RECOVERY_FC" if "fc_recovery" in path.name else "STANDARD_FC"
        operations.append({"path": str(path), "operation": kind, "scene": receipt.get("scene", path.parents[2].name),
            "map_id": receipt.get("map_id"), "status": receipt["status"],
            **{key: receipt.get(key) for key in keys},
            "parent_native_cache_hits": len(receipt.get("parent_physical_cache_hits", {})),
            "missing_timings": receipt.get("standalone_timing_missing_components", []),
            "timing_kind": "WORKER_WALL_SECONDS_NOT_CUDA_EVENT_TIME"})
        if kind == "NATIVE_Q" and receipt.get("native_physical_image_encodings") is not None:
            operations[-1]["physical_native_image_inputs"] = receipt["native_physical_image_encodings"]
            operations[-1]["physical_Q_image_inputs"] = receipt["physical_image_encodings"] - receipt["native_physical_image_encodings"]
        if receipt["status"] != "COMPLETE":
            failures.append({"path": str(path), "status": receipt["status"], "error": receipt.get("error"),
                             "recorded_costs": operations[-1]})
    for path in sorted((root / "maps").glob("*/*/map_receipt.json")):
        receipt = read(path)
        operations.append({"path": str(path), "operation": "FULL_MAP_CPU", "scene": receipt["scene"], "map_id": receipt["map_id"],
            "status": receipt["status"], "elapsed_seconds": receipt.get("elapsed_seconds"), "physical_image_encodings": 0})
        if receipt["status"] != "COMPLETE":
            failures.append({"path": str(path), "status": receipt["status"], "error": receipt.get("error")})
    for path in sorted((root / "frontend").glob("*/S1*/receipt.json")):
        receipt = read(path)
        operations.append({"path": str(path), "operation": "CACHED_SAM_CPU_COMPOSITION", "scene": receipt["scene"],
            "map_id": "S1_FRONTEND", "status": receipt["status"], "elapsed_seconds": receipt.get("elapsed_seconds"),
            "physical_image_encodings": 0, "physical_region_poolings": 0})
    preserved = [str(path) for pattern in ("**/*failed*", "**/follower_diagnostic", "**/S1_producer_*") for path in root.glob(pattern)]
    totals = Counter()
    for row in operations:
        if row["status"] == "COMPLETE":
            totals[row["operation"] + "_physical_image_inputs"] += row.get("physical_image_encodings") or 0
            totals[row["operation"] + "_physical_region_poolings"] += row.get("physical_region_poolings") or 0
    return {"method_light_costs": lights, "map_standalone_costs": maps, "physical_operations": operations,
        "complete_operation_totals": dict(totals), "failed_operations": failures, "preserved_attempt_paths": sorted(set(preserved)),
        "new_SAM_forwards": 0, "new_CropFormer_forwards": 0, "new_text_forwards": 0,
        "cold_runs_performed": False, "operation_wall_sums_are_not_project_elapsed_time": True,
        "CUDA_event_and_separate_CPU_timings": "NOT_RECORDED; WORKER_WALL_TIME_REPORTED",
        "warm_cache_counts_do_not_equal_standalone_logical_cost": True}


def _external_manifest(binding, rows):
    root, identities = Path(binding["output_root"]), {}
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    paths = [root / "resolved_inputs.json", root.parent / "tooling/recovery_native_v2/recovery_native_build_receipt.json"]
    paths += sorted((root / "maps").glob("*/*/map_receipt.json"))
    paths += sorted((root / "readouts").glob("*/*/receipt.json"))
    paths += sorted((root / "readouts").glob("*/*/native_query/native_query_receipt.json"))
    paths += sorted((root / "readouts").glob("*/*/fc/receipt.json"))
    paths += sorted((root / "recovery").glob("*/*/*receipt.json"))
    paths += sorted((root / "frontend").glob("*/S1/receipt.json"))
    paths += sorted((root / "light").glob("*/*/receipt.json"))
    paths += sorted((root / "light").glob("*/*/extra_conditions/*/receipt.json"))
    paths += [Path(row["evaluation_receipt"]) for row in rows]
    for path in sorted(set(paths)):
        actual = index.identity(path)
        identities[(actual["path"], actual["sha256"])] = actual
        receipt = read(path)
        for name in ("inputs", "outputs", "recovery_inputs", "inputs_and_outputs"):
            for row in receipt.get(name, []):
                if isinstance(row, dict) and row.get("path") and row.get("sha256"):
                    identities[(row["path"], row["sha256"])] = row
        for name in ("arrays", "native_features", "query_scores", "query_decisions", "source", "extension"):
            row = receipt.get(name)
            if isinstance(row, dict) and row.get("path") and row.get("sha256"):
                identities[(row["path"], row["sha256"])] = row
        if receipt.get("manifest"):
            for name in ("trace.json.gz", "matches.json.gz"):
                path_trace = Path(PathResolver(binding["path_map"]).resolve(receipt["manifest"])).with_name(name)
                actual = index.identity(path_trace)
                identities[(actual["path"], actual["sha256"])] = actual
    repo = Path(binding["repository_root"])
    manifests = []
    for path in sorted((root / "light").glob("*/*/**/predictions/*/manifest.json")):
        manifest = read(path)
        row = manifest["arrays"]
        payload = path.parent / row["path"]
        manifests.append(index.identity(path))
        identities[(str(payload), row["sha256"])] = {**row, "path": str(payload), "bytes": payload.stat().st_size}
    index.write_memo(root / "validation/input_verifications.json")
    from .provenance import archive_producers

    producers = archive_producers(binding, identities.values())
    return {"status": "CONTENT_MANIFEST_AND_RECONSTRUCTION_COMMANDS", "task_root": str(root),
        "parent_root": binding["parent_root"], "files": sorted(identities.values(), key=lambda row: row["path"]),
        "byte_exact_Python_producer_archives": producers,
        "multiple_source_versions_at_one_original_path_are_preserved": True,
        "prediction_manifests": manifests, "bytes_recorded": sum(row.get("bytes", 0) for row in identities.values()),
        "large_data_committed": False, "availability": "EXISTING_SHARED_STORAGE_NO_PUBLIC_DOWNLOAD_PROMISE",
        "path_rebinding": "USE --path-map WITH ABSOLUTE OLD_TO_NEW ROOTS; VERIFY CONTENT HASHES",
        "reconstruction_command": shlex.join([read(binding["spec"])["default_python"],
            str(repo / "scripts/evaluation/run_ovimap_recovery_wave2.py"), "--phase", "all", "--resume"]),
        "native_build_command": shlex.join([read(binding["spec"])["default_python"], "-m", "static_ovmap.recovery_wave2.runtime",
            "--binding", str(root / "resolved_inputs.json"), "--upstream", str(root.parent / "upstream"),
            "--build-root", str(root.parent / "tooling/recovery_native_v2"), "--resume"]),
        "historical_producers": "MAP running.json source_commit AND FILE SHA; FC receipts effective worker/operator SHA",
        "licensed_model_and_dataset_access_required": True}


def generate(binding):
    root, repo = Path(binding["output_root"]), Path(binding["repository_root"])
    for scene in binding["datasets"]["replica"]:
        require_transfer_freeze(binding, scene)
    selection, composition, sensitivity = (read(root / "selection" / (name + ".json")) for name in
                                           ("final", "composition_check", "leave_one_out"))
    if composition["status"] != "COMPLETE" or sensitivity["status"] != "COMPLETE":
        raise ValueError("report requires all authorized fixed composition and sensitivity evidence")
    matrix, pools = collect_matrix(binding)
    mechanisms, funnels, sources, decisions, geometry = collect_mechanisms(binding)
    rows, costs = scene_evidence(binding), collect_costs(binding)
    external = _external_manifest(binding, rows)
    compact = repo / COMPACT
    compact.mkdir(parents=True, exist_ok=True)
    artifacts = {"resolved_inputs.json": binding, "experiment_matrix.json": {"rows": matrix, "spec": read(binding["spec"])},
        "recovery_funnel.json": funnels, "official_scene_rows.json": rows, "official_pools.json": pools,
        "geometry_summary.json": geometry, "mechanism_comparisons.json": mechanisms,
        "failure_and_costs.json": costs, "external_artifacts.json": external,
        "leave_one_out.json": sensitivity, "composition_check.json": composition}
    for filename, value in artifacts.items():
        atomic_write_json(compact / filename, value)
    _gzip(compact / "source_scores.json.gz", sources)
    _gzip(compact / "locked_decisions.json.gz", decisions)
    validation = []
    for path in sorted((root / "validation").glob("*.xml")):
        parsed = ET.parse(path).getroot()
        suites = [parsed] if parsed.tag == "testsuite" else list(parsed)
        validation.append({"path": str(path), "tests": sum(int(row.get("tests", 0)) for row in suites),
            "failures": sum(int(row.get("failures", 0)) for row in suites), "errors": sum(int(row.get("errors", 0)) for row in suites),
            "test_ids": [row.get("classname", "") + "::" + row.get("name", "") for row in parsed.iter("testcase")]})
    atomic_write_json(compact / "validation/targeted_runs.json", validation)
    for name in ("validation/native/validation_receipt.json", "freeze/receipt.json"):
        atomic_write_json(compact / "validation" / Path(name).name, read(root / name))
    atomic_write_json(compact / "validation/recovery_native_build_receipt.json",
                     read(root.parent / "tooling/recovery_native_v2/recovery_native_build_receipt.json"))
    primary_path = compact / "selection.json"
    if not primary_path.is_file():
        raise ValueError("report requires the existing committed pre-Replica selection artifact")
    complete_maps = [row for row in costs["physical_operations"] if row["operation"] == "FULL_MAP_CPU" and row["status"] == "COMPLETE"]
    dev_count = sum(row["scene"] in binding["datasets"]["development"] for row in complete_maps)
    replica_count = sum(row["scene"] in binding["datasets"]["replica"] for row in complete_maps)
    if dev_count > 20 or replica_count > 8:
        raise ValueError("actual complete map count exceeded the protocol")
    chosen_id, nominee = selection["light_package"]["id"], selection["nominated_map"]
    chosen = next(row for row in matrix if row["cohort"] == "development" and row["map_id"] == "BB00_NATIVE" and row["method"] == chosen_id)
    outcome = "RETAIN_MEASURED_GAIN_CANDIDATE" if nominee or chosen["NET_GAIN"] else "RETAIN_BASELINE"
    completion = {"implementation_status": "IMPLEMENTATION_COMPLETE", "scientific_outcome": outcome,
        "publication_status": "POST_PUSH_EXTERNAL_RECEIPT_REQUIRED", "deployment": "N0_UNCHANGED",
        "development_full_map_successes": dev_count, "Replica_full_map_successes": replica_count,
        "new_full_baseline_maps": 0, "nominated_map": nominee, "light_selected_id": chosen_id,
        "development_NET_GAIN": bool(nominee or chosen["NET_GAIN"]),
        "new_SAM_CropFormer_text_forwards": 0, "primary_final_review": "EXTERNAL_PRIMARY_REVIEW_REQUIRED_BEFORE_PUBLISH",
        "all_Replica_scenes_already_exposed": True, "all_required_scientific_tables": 6,
        "actual_model_inputs": costs["complete_operation_totals"], "pending_scientific_leaves": [],
        "binding_identity": binding["identity"]}
    atomic_write_json(compact / "completion.json", completion)
    reports = write_reports(binding, matrix, mechanisms, funnels, geometry, costs, sensitivity, selection, completion)
    sizes = {str(path.relative_to(repo)): path.stat().st_size for path in compact.rglob("*") if path.is_file()}
    receipt = {"status": "REPORTS_GENERATED", "reports": reports, "compact_bytes": sum(sizes.values()),
        "soft_50_MiB_target_met": sum(sizes.values()) <= 50 * 1024 ** 2, "compact_files": sizes,
        "implementation_status": completion["implementation_status"], "primary_audit_pending": True}
    receipt["identity"] = canonical_digest(receipt)
    atomic_write_json(root / "reporting/receipt.json", receipt)
    return receipt


def write_reports(binding, matrix, mechanisms, funnels, geometry, costs, sensitivity, selection, completion):
    repo, root = Path(binding["repository_root"]), Path(binding["output_root"])
    score_table = _table(["Cohort", "Map / Method", "Status / Coverage", "APall", "AP50", "AP25", "mIoU", "mAcc", "Delta APall pp", "Delta mIoU pp"],
        [[row["cohort"], row["map_id"] + " / " + row["method"], row["status"] + " " + row["coverage"],
          *[_number(None if row["metrics"] is None else row["metrics"][key], percent=True) for key in METRICS],
          _number(None if row.get("delta_pp") is None else row["delta_pp"]["apall"]),
          _number(None if row.get("delta_pp") is None else row["delta_pp"]["miou"])] for row in matrix])
    funnel_table = _table(["Scene", "Arm", "RAW", "Painted", "Candidates", "U1 eligible", "Legal views", "Source available", "Used requests", "Added owners", "Target min100", "Cap exclusions"],
        [[row["scene"], row["method"], row["raw_positive_owners"], row["native_painted_owners"], row["capped_source_min100_candidates"],
          row["single_retained_native_eligible"], row["legal_captured_request_count"], row["source_available_owners"], row["used_evidence_requests"],
          row["exported_source_positive_owners"], len(row["official_target_min100_owners"]), row["FC_cap_excluded_requests"]] for row in funnels])
    light_table = _table(["Scene", "Method", "Added TP50", "Added FP50", "Ambiguous TP/FP50", "Lost old TP entries all overlaps", "Old rank changes", "Old class changes", "Correct-point delta"],
        [[row["scene"], row["method"], row["matcher"]["actual"]["by_overlap"]["0.5"]["added_tp_score_entries"],
          row["matcher"]["actual"]["by_overlap"]["0.5"]["added_fp_score_entries"],
          str(row["matcher"]["actual"]["by_overlap"]["0.5"]["ambiguous_added_tp_score_entries"]) + "/" + str(row["matcher"]["actual"]["by_overlap"]["0.5"]["ambiguous_added_fp_score_entries"]),
          len(row["matcher"]["lost_baseline_GT_TP_entries"]), len(row["old_owner_rank_changes"]), len(row["class_changed_incumbents"]),
          row["point_confusion_diagonal_delta"]] for row in mechanisms["light"] if row["method"] in METHODS])
    association_table = _table(["Scene", "Arm", "Assign / Native", "Followers", "Candidate / Alias veto visits", "Mixed fallback visits", "Planned-realized mismatch", "Fresh owner rate", "Mean best IoU", "R50"],
        [[row["scene"], row["map_id"], str(row["totals"].get("ASSIGN_EXISTING", 0)) + " / " + str(row["totals"].get("USE_NATIVE", 0)),
          row["totals"].get("duplicate_owner_followers", 0), str(row["totals"].get("candidate_veto_visits", 0)) + " / " + str(row["totals"].get("alias_veto_visits", 0)),
          row["totals"].get("mixed_fallback_constrained_candidate_visits", 0), row["totals"].get("planned_realized_discrepancies", 0), _number(row["fresh_group_owner_rate"], percent=True),
          _number(next(item for item in geometry["scenes"] if item["scene"] == row["scene"] and item["map_id"] == row["map_id"])["mean_best_gt_iou"], percent=True),
          next(item for item in geometry["scenes"] if item["scene"] == row["scene"] and item["map_id"] == row["map_id"])["matches"]["0.5"]["matched_gt"]] for row in mechanisms["association"]])
    sam_table = _table(["Scene", "Arm", "Protected crop pixels", "S1 additions", "Stable known pixels", "Self / Other track pixels", "Suppressed tracks / pixels", "History abstentions", "Crop preserved"],
        [[row["scene"], row["map_id"], row["protected_crop_pixels"], row["S1_added_pixels"],
          (row["S2_totals"] or {}).get("stable_known_pixels", "NA"), str((row["S2_totals"] or {}).get("known_self_track_pixels", "NA")) + " / " + str((row["S2_totals"] or {}).get("known_other_track_pixels", "NA")),
          str((row["S2_totals"] or {}).get("suppressed_tracks", "NA")) + " / " + str((row["S2_totals"] or {}).get("removed_additions", "NA")),
          (row["S2_totals"] or {}).get("history_abstention_tracks", "NA"), row["all_positive_crop_pixels_unchanged"]] for row in mechanisms["SAM"]])
    cost_table = _table(["Scene", "Map", "Operation", "Status", "Physical image inputs", "Encoder calls", "Region poolings", "Model load seconds", "Worker wall seconds"],
        [[row.get("scene"), row.get("map_id"), row["operation"], row["status"], row.get("physical_image_encodings", "NA"),
          row.get("physical_encoder_calls", row.get("encoder_batch_calls", "NA")), row.get("physical_region_poolings", "NA"),
          _number(row.get("model_load_seconds")), _number(row.get("elapsed_seconds"))] for row in costs["physical_operations"]])
    logical_rows = []
    for cohort, scenes in binding["datasets"].items():
        for method in METHODS:
            method_rows = [row for row in costs["method_light_costs"] if row["scene"] in scenes and row["method"] == method]
            logical_rows.append([cohort, "BB00_NATIVE / " + method,
                sum(row["standalone_new_encoder_inputs"] for row in method_rows),
                sum(row["standalone_required_image_inputs_including_baseline"] for row in method_rows),
                sum(row["standalone_additional_region_poolings"] for row in method_rows), 0])
        for map_id in (*A_MAPS, *S_MAPS):
            method_rows = [row for row in costs["map_standalone_costs"] if row["scene"] in scenes and row["map_id"] == map_id]
            if method_rows:
                logical_rows.append([cohort, map_id,
                    sum(row["standalone_new_encoder_inputs"] for row in method_rows),
                    sum(row["standalone_required_image_encodings"] for row in method_rows), "SEE_PHYSICAL_OPERATION_ROWS",
                    sum(row["standalone_SAM_image_inputs"] for row in method_rows)])
    logical_table = _table(["Cohort", "Map / Method", "Standalone additional image inputs", "Required image inputs", "Additional recovery poolings", "Required cached-SAM cold inputs"], logical_rows)
    geometry_table = _table(["Arm", "Status", "Best IoU drop pp", "R50 count loss", "Fragment ratio", "Semantic status"],
        [[name, row["status"], _number(row.get("mean_best_iou_drop_pp")), row.get("R50_count_loss", "NA"),
          _number(row.get("fragment_ratio")), row["semantic_status"]] for name, row in geometry["screens"].items()])
    paired_rows = [row for scene in mechanisms["paired_U1_U2"] for row in scene["objects"]]
    paired_table = _table(["Scene", "Owner", "U1 / U2 class", "Agreement", "Target points", "Valid GT points", "GT majority / purity", "U1 / U2 correct points"],
        [[row["scene"], row["owner"], str(row["U1_label"]) + " / " + str(row["U2_label"]), row["label_agreement"], row["projected_support_points"], row["valid_GT_points"],
          str(row["unique_GT_majority_class"]) + " / " + _number(row["GT_majority_purity"]), str(row["U1_correct_points"]) + " / " + str(row["U2_correct_points"])] for row in paired_rows])
    same_table = _table(["Cohort", "Diagnostic", "Common source owners", "APall", "AP50", "AP25", "mIoU", "mAcc"],
        [[cohort, method, sum(len(owners) for owners in result["covered_owners"].values()),
          *[_number(pool["metrics"][key], percent=True) for key in METRICS]]
         for cohort, result in mechanisms["same_covered_subset"].items() for method, pool in result["pools"].items()])
    prefix = ("# Recovery Wave 2 Results\n\n"
        f"Decision: **{completion['scientific_outcome']}**. Deployment remains **N0_UNCHANGED**. "
        f"Frozen light: `{selection['light_package']['id']}`; map nominee: `{selection['nominated_map']}`.\n\n"
        "Values are percentages; paired deltas are percentage points against each cohort's exact BB00_NATIVE + D2. "
        "Development uses the original four scenes; all eight Replica scenes were already exposed. "
        "APall uses the actual released .50-.90 overlap vector. Ordered released pooling and summed confusion matrices are used.\n\n")
    results = prefix + "## Table 1: Official Metrics and Coverage\n\n" + score_table
    results += "\n\n## Raw Geometry Screen\n\n" + geometry_table
    results += "\n\n## Table 2: Recovery Funnel\n\n" + funnel_table
    results += "\n\nSource min100 rows and target min100 points are separate. Candidate clipping retains the exact observed mask; cached pooling was not performed on the smaller residual support.\n\n"
    results += "## Table 3: Actual Matcher and Semantic Changes\n\n" + light_table
    results += "\n\nTP/FP counts are actual score entries at overlap .50, not unique objects. Duplicate minimum-score FP and later unvisited FP can both contribute. Numeric ties retain ambiguous identity. Full per-overlap losses and confusion deltas are in the compact mechanism artifact.\n\n"
    results += "## Paired U1/U2 Classification\n\n" + paired_table
    results += "\n\nThe two arms use the same successful retained request. Technical failures are excluded from this paired diagnostic and remain in full output coverage. Different native relative and FC cosine score scales are not treated as calibrated equivalents. GT majority is post-lock descriptive evidence.\n\n"
    results += "## Same-Covered-Subset Scores\n\n" + same_table
    results += "\n\nThe scope is the intersection of U2/U3 source successes, selected without GT. These diagnostics do not enter selection.\n\n"
    results += "## Table 4: Planned and Realized Association\n\n" + association_table
    results += "\n\nFresh owner rate counts actual positive mode4 group/owner pairs absent from the factor0 preinsert prior set. Candidate and alias vetoes count visits. No explicit CREATE_NEW action is used. Full traces separate unknown count-owner evidence from mismatches.\n\n"
    results += "## Table 5: Cached SAM Completion\n\n" + sam_table
    results += "\n\nS2 uses two preceding completed factor0 snapshots. Unknown self/history is abstention; only additions are suppressed. CropFormer positives and their separate groups are protected. Per-track pixels may overlap and are not unique image-pixel counts.\n\n"
    results += "## Table 6: Actual Physical Operations\n\n" + cost_table
    results += "\n\n### Standalone Logical Requirements\n\n" + logical_table
    results += "\n\nStandalone additional image inputs and required baseline contents are recorded separately in `failure_and_costs.json`. Physical cache reuse is not zero standalone compute. Timings are worker wall seconds; separate CUDA-event and CPU timings were not recorded. Concurrent duration sums are not project elapsed time.\n"
    loo = _table(["Leave Out", "Map / Method", "Status", "APall", "AP50", "mIoU", "Delta APall pp", "NET_GAIN diagnostic"],
        [[row["leave_out"], row["map_id"] + " / " + row["method"], row["status"],
          *[_number(None if row["metrics"] is None else row["metrics"][key], percent=True) for key in ("apall", "ap50", "miou")],
          _number((row.get("delta_pp") or {}).get("apall")), row.get("net_gain_in_diagnostic_pool", "NA")] for row in sensitivity["rows"]])
    selected = ("# Recovery Wave 2 Selection\n\n"
        f"Frozen light `{selection['light_package']['id']}`: `{selection['light_package']['recipe']}`. "
        f"Nominated map: `{selection['nominated_map']}`. Freeze commit: `{read(root / 'freeze/receipt.json')['commit']}`.\n\n"
        "All choices use the complete four-scene development pool. Feasibility: APall >= B-0.05pp, AP50 and mIoU >= B-0.10pp. "
        "NET_GAIN: APall >= B+0.20pp with those AP50/mIoU guards. Preference bands are APall 0.05pp, mIoU 0.10pp, AP50 0.10pp, "
        "then standalone new encoder inputs, changed blocks, gamma distance and method ID. Bands are not statistical equivalence. "
        "The baseline is a candidate. One W x U composition is measured; at most one qualifying map enters the two-by-two check.\n\n"
        f"Strict light metric order: `{selection['light_ranking']['strict_metric_ranking']}`.\n\n"
        f"Banded light preference: `{selection['light_ranking']['banded_preference']}`.\n\n"
        f"Composition status: `{read(root / 'selection/composition_check.json')['map_composition']}`. No A x S sweep.\n\n"
        "## Fixed-Method Leave-One-Scene-Out Sensitivity\n\n" + loo + "\n\nThese correlated three-scene pools reuse fixed predictions and the released evaluator. No algorithms are refit; neither selection nor Replica parameters change.\n")
    handoff = ("# Recovery Wave 2 Handoff\n\n"
        f"Implementation: `{completion['implementation_status']}`. Scientific outcome: `{completion['scientific_outcome']}`. "
        "Publication is verified only by the external `publication/final.json` full-SHA comparison. Deployment: `N0_UNCHANGED`.\n\n"
        f"Task root: `{root}`. Immutable parent: `{binding['parent_root']}`. Branch: `research/ovimap-recovery-wave2-v1`.\n\n"
        f"Completed new full maps: development {completion['development_full_map_successes']}, Replica {completion['Replica_full_map_successes']}. "
        "No baseline maps, new SAM/CropFormer/text forwards, training, checkpoint or temperature fits. Fresh own-map N/Q/FC regeneration is separately measured.\n\n"
        f"Default Python: `{read(binding['spec'])['default_python']}`. FC Python: `{binding['fc']['python']}`. "
        "Original FP32 model/operator/checkpoint identities and frozen category/template order are in `resolved_inputs.json`; native patch/build receipts are retained. "
        "The isolated upstream patch stack is capture, backbone, then recovery. Original binaries remain unchanged.\n\n"
        "Use the existing shared artifacts listed with content hashes in `external_artifacts.json`; large RGB-D, models, dense caches, maps, TSDF and verbose native logs are external. "
        "These paths are not public download links. Restore the licensed original data/model access and hash-verify files; use `--path-map` when relocating roots.\n\n"
        "```bash\n" + shlex.join([read(binding["spec"])["default_python"], str(repo / "scripts/evaluation/run_ovimap_recovery_wave2.py"), "--phase", "all", "--resume"]) + "\n```\n\n"
        "Mapping defaults to 2 x 8 CPU threads, evaluation to 3 x 4 BLAS threads, one model worker under the existing GPU lock. "
        "Complete receipts resume only with verified content and recorded producer identities. Failed map attempts restart at frame zero in a new retained directory.\n\n"
        "`experiment_matrix.json` records measured, exact-equivalent, geometry-screened and non-nominated transfer leaves. "
        "Failure receipts and missing timings remain visible in `failure_and_costs.json`. "
        "Primary requirement audit and targeted production/native validation are under the compact `validation/` directory.\n")
    claims = ("# Recovery Wave 2 Claims\n\n"
        f"Supported decision: `{completion['scientific_outcome']}`; measured development NET_GAIN: `{completion['development_NET_GAIN']}`.\n\n"
        "Supported claims are the exact ordered cohort metrics, post-lock recovery coverage/classification diagnostics, actual native action effects and protected cached-SAM changes documented in RESULTS. "
        "A gain, tradeoff or failure must be named by method, cohort and metric. The frozen transfer recipe is the only nominated package; a descriptive best Replica result cannot replace it.\n\n"
        "U changes exported coverage on a fixed raw geometry, so it does not establish raw geometry improvement. Equal AP does not establish equal probabilities, semantics or partitions. "
        "Same-mask U1/U2 comparisons support only their common successful source subset. Resource-screened semantics are unmeasured, not zero and not copied baseline scores.\n\n"
        "Replica scenes were already exposed and are not new held-out confirmation. Leave-one-scene-out pools are correlated sensitivity analyses, not independent trials. "
        "No significance, calibrated cross-encoder score, broad generalization or automatic deployment claim is made. "
        "Warm-cache zero added image encodings for recovery does not imply zero GPU for fresh own-map semantics or zero cold standalone cost.\n")
    reports = []
    for name, body in (("RESULTS", results), ("SELECTION", selected), ("HANDOFF", handoff), ("CLAIMS", claims)):
        path = repo / "docs/paper/static_ovmap" / ("RECOVERY_WAVE2_" + name + ".md")
        path.write_text(body)
        reports.append(str(path.relative_to(repo)))
    return reports
