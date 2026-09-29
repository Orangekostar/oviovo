"""Collect measured report tables only after all prescribed results exist."""

from collections import Counter
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest

from .audit import validate_matrix
from .comparisons import compare_split
from .costs import scene_costs
from .diagnostics import diagnose_scene
from .source_contract import export_contract


def request_intervention(root, scene, variant):
    if variant.startswith("AW_E01_"):
        return {"request_accounting": "original retained Q features; no new visual requests"}
    family = "c0/requests" if variant.startswith("AW_C0_") else "e02" if variant.startswith("AW_E02_") else "e03"
    folder = root / family / scene
    if family != "c0/requests":
        folder /= variant + "/requests"
    rows = [read_json(p) for p in sorted(folder.glob("*.json")) if p.name != "receipt.json"]
    if not rows:
        raise ValueError("measured visual source has no request receipts")
    status = Counter(r["status"] for r in rows)
    failure = Counter(r.get("error", r["status"]) for r in rows if r["status"] != "COMPLETE")
    return {"requests": len(rows), "status_counts": dict(status), "failure_counts": dict(failure),
            "nonempty_final_support_views": sum(r.get("dense_support", 0) > 0 if "dense_support" in r
                else any(n > 0 for n in r.get("processor_support", [])) for r in rows),
            "recorded_request_elapsed_seconds": sum(r.get("elapsed_seconds", 0) for r in rows),
            "request_elapsed_missing": sum("elapsed_seconds" not in r for r in rows)}


def collect(binding):
    root, index = Path(binding["output_root"]), InputIndex()
    index.identity(__file__)
    matrix = validate_matrix(binding)
    index.identity(root / "audit/matrix.json")
    methods = matrix["methods"]
    spec = read_json(binding["spec"])
    comparisons = {split: compare_split(binding, split) for split in ("cal", "replica")}
    diagnostics, costs, intervention = {}, {}, {}
    for scene, bound in binding["scenes"].items():
        manifest = read_json(bound["static_manifest"]["path"])
        index.identity(bound["static_manifest"]["path"], bound["static_manifest"])
        intervention[scene] = {"static_capped_targets": len(manifest["selected_targets"]),
                               "static_excluded_targets": len(manifest["excluded_targets"]), "sources": {}}
        for variant in spec["source_variants"]:
            name = variant["id"]
            contract_path = root / "source_contracts" / scene / name / "contract.json"
            if not contract_path.exists():
                export_contract(binding, scene, name)
            contract = read_json(contract_path)
            for item in contract["inputs"]:
                index.identity(item["path"], item)
            index.identity(contract_path)
            reasons = Counter(o["explicit_fallback_reason"] for o in contract["owners"].values() if not o["available"])
            intervention[scene]["sources"][name] = {"fallback_owner_reasons": dict(reasons),
                **request_intervention(root, scene, name)}
        diagnostic_path = root / "diagnostics" / scene / "summary.json"
        if not diagnostic_path.exists():
            diagnose_scene(binding, scene, methods)
        diagnostics[scene] = read_json(diagnostic_path)
        index.identity(diagnostic_path)
        costs[scene] = scene_costs(binding, scene)
        index.identity(root / "costs" / (scene + ".json"))
    split_tables = {}
    for split, scenes in (("cal", spec["datasets"]["calibration"]), ("replica", spec["datasets"]["replica"])):
        sources, changes = {}, {}
        for variant in spec["source_variants"]:
            name = variant["id"]
            rows = [diagnostics[s]["source_intervention"][name] for s in scenes]
            totals = {k: sum(r[k] for r in rows) for k in rows[0]}
            requests = [intervention[s]["sources"][name] for s in scenes]
            failures, fallback = Counter(), Counter()
            for r in requests:
                failures.update(r.get("failure_counts", {}))
                fallback.update(r["fallback_owner_reasons"])
            totals.update(failed_view_reasons=dict(failures), fallback_owner_reasons=dict(fallback),
                          nonempty_final_support_views=None if name.startswith("AW_E01_") else sum(r["nonempty_final_support_views"] for r in requests),
                          static_capped_targets=None if name.startswith("AW_E01_") else sum(intervention[s]["static_capped_targets"] for s in scenes))
            sources[name] = totals
        for method in methods:
            rows = [diagnostics[s]["methods"][method] for s in scenes]
            changes[method] = {k: sum(r[k] for r in rows) for k in rows[0] if k not in {"scene", "method"}}
        workers = {}
        for scene in scenes:
            for worker, value in costs[scene]["physical_completed_worker_invocations"].items():
                totals = workers.setdefault(worker, {})
                for key, number in value["measured"].items():
                    if not isinstance(number, (int, float)):
                        continue
                    if key.startswith("peak_gpu_") or key == "parameter_count":
                        totals[key] = max(totals.get(key, 0), number)
                    else:
                        totals[key] = totals.get(key, 0) + number
        split_tables[split] = {"sources": sources, "corrections_and_damage": changes, "worker_costs": workers}
    downloads = {}
    for model in ("sam2", "so400m", "fc_frozen", "ovrcoat"):
        path = root.parent / "assets" / model / "download_receipt.json"
        index.identity(path)
        value = read_json(path)
        files = value.get("files", [value.get("checkpoint")])
        for item in files:
            index.identity(item["path"], item)
        downloads[model] = {"receipt": value, "verified_bytes": sum(v["bytes"] for v in files)}
    downloaded = sum(v["verified_bytes"] for v in downloads.values())
    if downloaded > 25 * 1024**3:
        raise ValueError("new model download allowance exceeded")
    smoke_paths = [root / "c0/smoke/scene0056_00/receipt.json", root / "e02/sam2_smoke/scene0056_00/receipt.json",
                   *[root / "e03/smoke/scene0056_00" / branch / "receipt.json" for branch in ("FC_FROZEN", "OVR")]]
    smokes = []
    for path in smoke_paths:
        index.identity(path)
        value = read_json(path)
        if value["status"] != "SMOKE_COMPLETE":
            raise ValueError("adapter lacks a successful real smoke")
        smokes.append({"receipt": str(path), **{k: v for k, v in value.items() if k not in {"inputs", "outputs"}}})
    # Evaluator receipt identity deduplicates exact prediction/rank cache reuse.
    evaluations = {}
    for path in sorted((root / "rows").glob("*/*/*.json")):
        row = read_json(path)
        receipt_path = Path(row["evaluation_receipt"])
        if str(receipt_path) not in evaluations:
            index.identity(receipt_path)
            evaluations[str(receipt_path)] = read_json(receipt_path)["elapsed_seconds"]
    fits = {}
    for variant in spec["source_variants"]:
        path = root / "calibration" / (variant["id"] + ".json")
        index.identity(path)
        fits[variant["id"]] = read_json(path)["elapsed_seconds"]
    result = {"status": "MEASURED_REPORT_DATA", "matrix_identity": matrix["identity"],
              "comparisons": comparisons, "split_tables": split_tables, "scene_intervention": intervention,
              "diagnostics": diagnostics, "scene_costs": costs, "downloads": downloads,
              "new_model_verified_bytes": downloaded, "smokes": smokes, "source_fit_seconds": fits,
              "distinct_evaluator_receipt_seconds": evaluations,
              "timing_scope": "Recorded evaluator receipt work includes reused historical evaluations; not this invocation's new wall time.",
              "inputs": index.entries()}
    result["identity"] = canonical_digest(result)
    write_once(root / "report/data.json", result)
    return result
