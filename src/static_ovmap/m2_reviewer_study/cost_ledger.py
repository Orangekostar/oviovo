"""Separate shared mapping, source requests, operation unions and actual work."""

import csv
import io
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.module_validation.contracts import canonical_digest

from .costs import operation_key, required_union, scene_operations


def build_cost_ledger(binding):
    root = Path(binding["output_root"])
    spec = read_json(binding["spec"])
    definitions = {row["method_id"]: row for row in spec["ablations"]}
    shared, methods, jobs, historical_sources = [], [], [], []
    for scene in binding["scenes"]:
        config = read_json(binding["scenes"][scene]["config"])
        data = config["scenes"][scene]
        ops = scene_operations(binding, scene)
        mapping_path = Path(data["mapping_receipt"])
        mapping = read_json(mapping_path)
        frontend_path = mapping_path.parent.parent / "frontend_job/receipt.json"
        frontend = read_json(frontend_path) if frontend_path.exists() else {}
        shared.append({"scene": scene, "mapping_seconds": mapping.get("elapsed_seconds"),
                       "frontend_seconds": frontend.get("elapsed_seconds"), "native_requests": ops["logical_attempts"]["N0"],
                       "native_crop_inputs": ops["crop_inputs"]["N0"], "mapping_receipt": str(mapping_path),
                       "frontend_receipt": str(frontend_path), "peak_gpu_memory": None,
                       "peak_memory_status": "HISTORICAL_NOT_RECORDED", "included_for_every_method": True})
        for name, path in (("Q_GAIN", Path(config["attempt_root"]) / "query" / scene / "Q_GAIN/receipt.json"),
                           ("S_SIGLIP2_AREA", Path(data["source_directory"]) / "semantic_models/siglip2/receipt.json")):
            value = read_json(path)
            historical_sources.append({"scene": scene, "source": name, "receipt": str(path),
                "logical_requests": ops["logical_attempts"][name], "logical_crop_inputs": ops["crop_inputs"][name],
                "wall_seconds": value.get("elapsed_seconds", value.get("elapsed_seconds_this_invocation")),
                "inference_seconds": value.get("physical", {}).get("inference_seconds", value.get("request_seconds")),
                "historical_physical": value.get("physical"), "peak_memory_status": "HISTORICAL_NOT_RECORDED",
                "study_new_image_forwards_for_reusing_this_source": 0})

        def add_method(method, selected, operations, attempts, scene=scene):
            selected = set(selected)
            required = {"N0", *selected}
            union = required_union(operations, selected)
            source_counts = {name: len(operations[name]) for name in sorted(required)}
            methods.append({"scene": scene, "dataset": binding["scenes"][scene]["dataset"], "method": method,
                "selected_score_sources": sorted(selected), "required_sources_including_map": sorted(required),
                "logical_attempts_by_source": {name: attempts[name] for name in sorted(required)},
                "logical_requests_total": sum(attempts[name] for name in required),
                "logical_crop_inputs_total": 6 * sum(attempts[name] for name in required),
                "unique_operations_by_source": source_counts, "unique_required_operations": len(union),
                "unique_required_crop_inputs": 6 * len(union), "source_operations": {n: operations[n] for n in sorted(required)},
                "proved_shared_operations_saved": sum(source_counts.values()) - len(union),
                "new_inference_seconds_is_not_end_to_end_latency": True})

        for method in binding["methods"]:
            selected = definitions[method]["sources"] if method in definitions else [method] if method in ops["operations"] else list(ops["operations"])
            add_method(method, selected, ops["operations"], ops["logical_attempts"])
        for receipt_path in sorted((root / "query_controls" / scene).glob("*_B*/receipt.json")):
            receipt = read_json(receipt_path)
            if receipt["status"] != "COMPLETE":
                continue
            decisions = read_json(receipt["decisions_path"])
            name = receipt["method_id"]
            query_ops = sorted({operation_key(config["models"]["native"]["identity"], r["request"]) for r in decisions["paid_requests"]})
            operations = {**ops["operations"], "Q_GAIN": query_ops}
            attempts = {**ops["logical_attempts"], "Q_GAIN": receipt["logical"]["attempts"]}
            add_method(name, ["Q_GAIN"], operations, attempts)
            policy = name.rsplit("_B", 1)[0]
            budget = name.rsplit("_B", 1)[1]
            for mode in ("RAW", "CAL"):
                add_method(f"RV_B_{policy}_B{budget}_{mode}", list(operations), operations, attempts)
            jobs.append({"scene": scene, "method": name, "receipt": str(receipt_path), "logical": receipt["logical"],
                         "physical": receipt["physical"], "wall_seconds": receipt["elapsed_seconds"],
                         "execution_mode": receipt["execution_mode"]})
    evaluations = []
    for folder in ("evaluation_cache", "pooled_cache"):
        for path in sorted((root / folder).glob("*/receipt.json")):
            value = read_json(path)
            evaluations.append({"receipt": str(path), "identity": value["identity"],
                                "seconds": value.get("elapsed_seconds"), "kind": folder})
    text = [read_json(p) for p in sorted((root / "robustness/text").glob("*/*/receipt.json"))]
    fit_jobs = []
    for path, count in ((root / "calibration/new_final.json", 2), (root / "query_controls/temperatures.json", 12)):
        if path.exists():
            value = read_json(path)
            fit_jobs.append({"path": str(path), "fits": count, "seconds": value.get("elapsed_seconds")})
    fold_timing = root / "calibration/fold_timing.json"
    fit_jobs.append({"path": str(root / "calibration/new_folds.json"), "fits": 4,
                     "seconds": read_json(fold_timing)["elapsed_seconds"] if fold_timing.exists() else None,
                     "timing_status": "MEASURED" if fold_timing.exists() else "INITIAL_ADHOC_FITS_NOT_TIMED"})
    totals = {key: sum(job["physical"].get(key, 0) for job in jobs)
              for key in ("model_loads", "model_forwards", "cache_hits", "crop_inputs", "inference_seconds")}
    completed_b200 = sum(job["method"].endswith("_B200") for job in jobs)
    result = {"status": "COMPLETE_QUERY_JOB_COVERAGE" if completed_b200 == 50 else "PARTIAL_QUERY_JOB_COVERAGE",
              "binding": binding["identity"], "completed_B200_jobs": completed_b200, "expected_B200_jobs": 50,
              "shared_historical_baseline": shared, "historical_sources": historical_sources,
              "method_requirements": methods, "study_query_jobs": jobs,
              "study_physical_query_totals": totals, "study_evaluations": evaluations,
              "study_evaluation_seconds": sum(r["seconds"] or 0 for r in evaluations),
              "text_encoding": [{k: r[k] for k in ("model", "dataset", "model_loads", "text_inputs_encoded", "elapsed_seconds", "device")} for r in text],
              "scalar_fits": fit_jobs, "known_scalar_fit_seconds": sum(row["seconds"] or 0 for row in fit_jobs),
              "scalar_timing_complete": all(row["seconds"] is not None for row in fit_jobs),
              "failed_initial_attempt_work": "NOT_FULLY_MEASURED; includes initial cache-import/GPU-busy attempts and same-scene concurrent-cache write failure; see repairs/query_json_roundtrip and execution checkpoint",
              "total_wall_latency_claim": "NOT_MEASURED; cached offline evidence is not online real-time latency",
              "physical_cost_scope": "Recorded completed jobs; incomplete/failed attempt time may be additional"}
    result["identity"] = canonical_digest(result)
    output = root / "costs" / result["identity"]
    write_once(output / "ledger.json", result)
    fields = ("scene", "dataset", "method", "logical_requests_total", "logical_crop_inputs_total",
              "unique_required_operations", "unique_required_crop_inputs", "proved_shared_operations_saved")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(methods)
    path = output / "costs.csv"
    if path.exists() and path.read_text() != stream.getvalue().replace("\r\n", "\n"):
        raise ValueError("cost CSV changed under same ledger identity")
    path.write_text(stream.getvalue())
    return result, output
