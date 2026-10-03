"""Complete receipt-backed result store, typed cells and supplied booktabs layouts."""

import csv
import io
from pathlib import Path
import subprocess

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read

from .costs import physical_ledger
from .diagnostics import TABLE3_METHODS
from .evaluation import METRICS, fraction_metrics
from .partial_execution import PARTIAL_RESULT, select_partial_pool_rows, validate_partial_scope
from .projected_views import _verified_identity
from .protocol import PACKAGE, experiment_matrix, load_spec, validate_overlaps
from .recovery_run import ARM_SOURCE
from .runtime import require_frozen_execution


TABLE_FILES = {"table1": "table1_main.tex", "table2": "table2_ablation.tex", "table3": "table3_recovery.tex"}
ROW_LABELS = {
    "table1": ("Mask3D + OpenMask3D", "Segment3D + OpenMask3D", "OVO-SLAM",
               "OVI-MAP (reproduced, A0)", "Ours: refinement only (A1)", "Ours: refinement + recovery (A3)"),
    "table2": ("A0: native & Native & None", "A1: E only & D2 & None", "A2: R only & Native & G1-FC",
               "A3: E + R & D2 & G1-FC", "A4: native recovery & D2 & G1-Native", "A5: FC-only & FC & G1-FC"),
    "table3": ("None (A1)", "Archived U2", "Independent G1 (A3)", "Independent G3"),
}


def _seal(value):
    value["identity"] = canonical_digest({k: v for k, v in value.items() if k != "identity"})
    return value


def result_store(spec, scene_receipts, pool_receipts, root):
    matrix, root = experiment_matrix(spec), Path(root)
    expected_scenes = [r["scene"] for r in matrix["anchors"]]
    if set(scene_receipts) != set(expected_scenes):
        raise ValueError("result store requires complete fixed 26-scene coverage")
    scene_records, pool_records = [], []
    blocked_scenes = {}
    for anchor in matrix["anchors"]:
        scene, cohort = anchor["scene"], anchor["cohort"]
        receipt = scene_receipts[scene]
        _verified_identity(receipt)
        methods = [r["id"] for r in spec["methods"] if cohort in r["cohorts"]]
        if receipt["status"] == PARTIAL_RESULT:
            available = validate_partial_scope(spec, receipt)
            blocked_scenes[scene] = receipt["blocked_methods"]
        elif receipt["status"] == "COMPLETE":
            available = methods
        else:
            raise ValueError("result store cannot infer technical blocks from incomplete scoring")
        if (receipt["scene"] != scene or receipt["cohort"] != cohort
                or len(receipt["rows"]) != len(available)
                or {r["method"] for r in receipt["rows"]} != set(available)):
            raise ValueError("result store scene has incomplete or extra fixed conditions")
        for method in methods:
            if method not in available:
                block = receipt["blocked_methods"][method]
                scene_records.append({"scene": scene, "cohort": cohort, "method_id": method,
                    "status": "BLOCKED_TECHNICAL", "metrics": dict.fromkeys(METRICS), "metric_unit": "FRACTION",
                    "source_kind": "UNAVAILABLE", "unavailable_reason": block["reason"],
                    "blocked_condition": block, "technical_block": receipt["blocked_sources"][block["recovery_source"]],
                    "receipt_path": str(root / "evaluation" / scene / "receipt.json"),
                    "receipt_identity": receipt["identity"], "evaluation_identity": None, "scorer_context_identity": None})
                continue
            row = next(r for r in receipt["rows"] if r["method"] == method)
            _verified_identity(row)
            if row["status"] != "COMPLETE" or row["scene"] != scene or row["cohort"] != cohort:
                raise ValueError("result store row leaves its exact completed scene/cohort")
            if row["rank_mode"] != "OFFICIAL_CURRENT_CLASS" or row["metric_unit"] != "FRACTION":
                raise ValueError("result store changed its official rank or fractional units")
            validate_overlaps(row["runtime_overlaps"])
            scene_records.append({"scene": scene, "cohort": cohort, "method_id": method, "status": "COMPLETE",
                "metrics": fraction_metrics(row["metrics"]), "metric_unit": "FRACTION",
                "receipt_path": str(root / "evaluation" / scene / "rows" / (method + ".json")),
                "receipt_identity": row["identity"], "evaluation_identity": row["evaluation_identity"],
                "scorer_context_identity": row["scorer_context_identity"],
                "source_kind": "EXACT_REUSE" if row.get("alias_proof") else "MEASURED"})
    blocked_pools = {(row["cohort"], row["method"]): {
        scene: blocked_scenes[scene][row["method"]] for scene in row["scene_order"]
        if row["method"] in blocked_scenes.get(scene, {})} for row in matrix["pools"]}
    expected_pools = {key for key, missing in blocked_pools.items() if not missing}
    if set(pool_receipts) != expected_pools:
        raise ValueError("result store requires exactly every complete fixed pool and no blocked/subset substitute")
    for planned in matrix["pools"]:
        cohort, method = planned["cohort"], planned["method"]
        missing = blocked_pools[(cohort, method)]
        if missing:
            first = next(iter(missing))
            pool_records.append({"cohort": cohort, "method_id": method, "status": "BLOCKED_TECHNICAL",
                "metrics": dict.fromkeys(METRICS), "metric_unit": "FRACTION", "source_kind": "UNAVAILABLE",
                "ordered_scene_ids": planned["scene_order"], "completed_coverage": len(planned["scene_order"]) - len(missing),
                "required_coverage": len(planned["scene_order"]), "pool_executed": False,
                "blocked_scenes": missing, "unavailable_reason": "FULL_COHORT_POOL_HAS_BLOCKED_CONDITIONS",
                "scoring_protocol_id": "UNAVAILABLE_NO_COMPLETE_RELEASED_POOL", "scoring_definition": None,
                "receipt_path": str(root / "evaluation" / first / "receipt.json"),
                "receipt_identity": scene_receipts[first]["identity"],
                "blocked_scene_receipt_identities": {scene: scene_receipts[scene]["identity"] for scene in missing}})
            continue
        receipt = pool_receipts[(cohort, method)]
        _verified_identity(receipt)
        selected = select_partial_pool_rows(spec, cohort, method,
            {scene: scene_receipts[scene] for scene in spec["cohorts"][cohort]})
        if (receipt["status"] != "COMPLETE" or receipt["cohort"] != cohort or receipt["method"] != method
                or receipt["aggregation"] != "RELEASED_DATASET_POOL"
                or receipt["rank_mode"] != "OFFICIAL_CURRENT_CLASS" or receipt["metric_unit"] != "FRACTION"
                or receipt["scene_order"] != planned["scene_order"]
                or receipt["row_identities"] != [r["identity"] for r in selected]
                or receipt["ordered_inputs"] != [r["evaluation_identity"] for r in selected]
                or any(r["runtime_overlaps"] != receipt["runtime_overlaps"] for r in selected)):
            raise ValueError("official pool changed its complete ordered scoring inputs or fractional protocol")
        validate_overlaps(receipt["runtime_overlaps"])
        protocol = {"evaluator_sha256": receipt["released_evaluator"]["sha256"], "cohort": cohort,
            "rank_mode": receipt["rank_mode"], "runtime_overlaps": receipt["runtime_overlaps"],
            "aggregation": receipt["aggregation"], "scoring_definition": receipt.get("scoring_definition")}
        pool_records.append({"cohort": cohort, "method_id": method, "status": "COMPLETE", "metrics": fraction_metrics(receipt["metrics"]),
            "metric_unit": "FRACTION", "ordered_scene_ids": planned["scene_order"],
            "completed_coverage": len(selected), "required_coverage": len(planned["scene_order"]),
            "scoring_protocol_id": canonical_digest(protocol), "scoring_definition": protocol,
            "receipt_path": str(root / "pools" / cohort / (method + ".json")),
            "receipt_identity": receipt["identity"], "row_identities": receipt["row_identities"],
            "source_kind": "MEASURED"})
    missing_outputs = sum(len(value) for value in blocked_scenes.values())
    return _seal({"status": PARTIAL_RESULT if missing_outputs else "COMPLETE",
        "matrix_identity": matrix["identity"], "metric_unit": "FRACTION",
        "main_scene_outputs": {"complete": len(scene_records) - missing_outputs, "blocked": missing_outputs,
                               "required": len(matrix["outputs"])},
        "internal_pools": {"complete": len(pool_receipts), "blocked": len(pool_records) - len(pool_receipts),
                           "required": len(matrix["pools"])},
        "scene_metrics": scene_records, "pooled_metrics": pool_records})


def _measured_cell(store, table, method, cohort, metric):
    rows = [r for r in store["pooled_metrics"] if r["cohort"] == cohort and r["method_id"] == method]
    if len(rows) != 1:
        raise ValueError("each repeated metric cell must resolve to exactly one complete official pool")
    row, value = rows[0], rows[0]["metrics"][metric]
    return {"table_id": table, "row_id": method, "column_id": cohort + "." + metric, "metric": metric,
        "source_kind": row["source_kind"] if value is not None else "UNAVAILABLE", "value_fraction": value,
        "display_unit": "percent", "display_decimal_places": 2, "cohort": cohort,
        "ordered_scene_ids": row["ordered_scene_ids"], "completed_coverage": row["completed_coverage"],
        "required_coverage": row["required_coverage"], "method_id": method,
        "scoring_protocol_id": row["scoring_protocol_id"], "receipt_path": row["receipt_path"],
        "receipt_identity": row["receipt_identity"],
        "source_id": canonical_digest({"pool": row["receipt_identity"], "method": method, "cohort": cohort, "metric": metric}),
        "unavailable_reason": row.get("unavailable_reason", "OFFICIAL_SCORER_UNDEFINED_METRIC") if value is None else None}


def _reported_cell(reference, method, cohort, metric, ordered):
    row = next(r for r in reference["rows"] if r["method"] == method)
    value = row[cohort][metric]
    return {"table_id": "table1", "row_id": method, "column_id": cohort + "." + metric, "metric": metric,
        "source_kind": "AUTHOR_REPORTED", "value_fraction": value / 100, "published_percent": value,
        "display_unit": "percent", "display_decimal_places": reference["published_decimal_places"],
        "cohort": cohort, "ordered_scene_ids": [], "completed_coverage": None, "required_coverage": None,
        "paired_cohort_context": ordered, "author_cohort_identity_independently_verified": False,
        "method_id": method, "scoring_protocol_id": "AUTHOR_REPORTED_PROTOCOL_NOT_INDEPENDENTLY_VERIFIED",
        "external_protocol_status": "AUTHOR_PROTOCOL_AS_REPORTED",
        "source_url": reference["source_url"], "source_version": reference["source_version"],
        "source_table": reference["source_table"], "source_row": method,
        "source_column": {"apall": "APall", "ap50": "AP50", "miou": "mIoU"}[metric],
        "reference_identity": reference["identity"],
        "source_id": canonical_digest({"reference": reference["identity"], "method": method, "cohort": cohort, "metric": metric})}


def main_tables(spec, store, reference, diagnosis, timing, root):
    for value in (store, reference, diagnosis, timing):
        _verified_identity(value)
    if (store["status"] not in ("COMPLETE", PARTIAL_RESULT) or reference["status"] != "AUTHOR_REPORTED_NOT_REPRODUCED"
            or diagnosis["status"] not in ("COMPLETE", PARTIAL_RESULT) or diagnosis["scene_count"] != 8
            or timing["status"] not in ("COMPLETE", PARTIAL_RESULT)
            or timing.get("planned_leaf_count", timing["leaf_count"]) != 24
            or timing["leaf_count"] + timing.get("blocked_leaf_count", 0) != 24):
        raise ValueError("main tables require complete scientific, external, diagnosis and24-call timing receipts")
    layout = read(PACKAGE / "TABLE_BINDINGS.json")
    result = {"table1": [], "table2": [], "table3": []}
    for method in layout["table1"]["reported_rows"]:
        cells = [_reported_cell(reference, method, cohort, metric, spec["cohorts"][cohort])
            for cohort in layout["table1"]["cohorts"] for metric in layout["table1"]["columns"]]
        result["table1"].append({"row_id": method, "block": "AUTHOR_REPORTED_NOT_RERUN", "cells": cells})
    for table, methods in (("table1", layout["table1"]["measured_rows"]), ("table2", layout["table2"]["rows"])):
        for method in methods:
            cells = [_measured_cell(store, table, method, cohort, metric)
                for cohort in layout[table]["cohorts"] for metric in layout[table]["columns"]]
            result[table].append({"row_id": method, "block": "PAIRED_RELEASED_POOL", "cells": cells})
    scenes, root = spec["cohorts"]["replica8"], Path(root)
    for position, (arm, method) in enumerate(TABLE3_METHODS.items()):
        row = diagnosis["arms"][arm]
        if (row["method"] != method or row["scene_order"] != scenes
                or len(row["scene_receipt_identities"]) != 8 or row["N"] != diagnosis["common_candidate_denominator"]
                or (row["n"] is not None and not 0 <= row["n"] <= row["N"])
                or (row["n"] is None and (row.get("status") != "BLOCKED_TECHNICAL" or not row.get("blocked_scenes")))):
            raise ValueError("Table3 diagnosis leaves its shared complete eight-scene candidate registry")
        cells = []
        for column, value in (("max_views", layout["table3"]["max_views"][position]),
                ("recovered_n_over_N", None), ("added_tp50_entries", row["added_TP50"]),
                ("added_fp50_entries", row["added_FP50"])):
            cell = {"table_id": "table3", "row_id": method, "column_id": column, "metric": column,
                "source_kind": "EXACT_REUSE" if column == "max_views" else "MEASURED",
                "cohort": "replica8", "ordered_scene_ids": scenes,
                "completed_coverage": 8 if column == "max_views" else row.get("complete_scene_count", 8), "required_coverage": 8,
                "method_id": method, "scoring_protocol_id": "RELEASED_SCORE_ENTRIES_AT_ACTUAL_AP50",
                "receipt_path": str(root / "diagnostics/replica8_recovery.json"), "receipt_identity": diagnosis["identity"]}
            if column == "max_views":
                cell.update(receipt_path=str(PACKAGE / "PROTOCOL_SPEC.json"), receipt_identity=canonical_digest(spec),
                            scoring_protocol_id="FROZEN_MAX_VIEW_LIMIT")
            if column == "recovered_n_over_N":
                cell.update(display_unit="count_ratio", numerator=row["n"], denominator=row["N"])
            else:
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                    raise ValueError("Table3 count cells require nonnegative integer counts")
                cell.update(display_unit="count", value_count=value)
                if column.startswith("added_"):
                    cell["ambiguous_additional_entries"] = row["ambiguous_added_TP50" if column == "added_tp50_entries"
                                                             else "ambiguous_added_FP50"]
            if (column == "recovered_n_over_N" and row["n"] is None) or (column != "recovered_n_over_N" and value is None):
                cell.update(source_kind="UNAVAILABLE", unavailable_reason=row["unavailable_reason"])
            cell["source_id"] = canonical_digest({"receipt": cell["receipt_identity"], "arm": arm, "column": column})
            cells.append(cell)
        cells.extend(_measured_cell(store, "table3", method, "replica8", metric) for metric in ("apall", "miou"))
        if arm == "NONE":
            seconds, timing_ids = timing["none_seconds"], []
            if seconds != 0 or any(row[key] for key in ("n", "added_TP50", "added_FP50")):
                raise ValueError("none recovery must retain its literal zero-by-definition counts/time")
        else:
            timed = timing["arms"][ARM_SOURCE[arm]]
            seconds, timing_ids = timed["mean_seconds"], timed["receipt_identities"]
            if (timed["scene_order"] != scenes or timed["scene_count"] != 8 or len(timing_ids) != 8
                    or (seconds is not None and (not np.isfinite(seconds) or seconds <= 0))
                    or (seconds is None and (not timed.get("blocked_scene_count") or not timed.get("unavailable_reason")))):
                raise ValueError("Table3 time must average all eight actual measured scenes")
        cells.append({"table_id": "table3", "row_id": method, "column_id": "cold_feature_incremental_seconds_mean",
            "metric": "cold_feature_incremental_seconds_mean",
            "source_kind": "UNAVAILABLE" if seconds is None else "EXACT_REUSE" if arm == "NONE" else "MEASURED",
            "value_seconds": seconds, "display_unit": "seconds_per_scene", "cohort": "replica8",
            "ordered_scene_ids": scenes, "completed_coverage": 8 if arm == "NONE" else timed.get("complete_scene_count", 8),
            "required_coverage": 8, "method_id": method,
            "scoring_protocol_id": "FEATURE_CACHE_COLD_INCREMENTAL_RECOVERY_MODEL_RESIDENT",
            "receipt_path": str(root / "timing/pool.json"), "receipt_identity": timing["identity"],
            "measurement_receipt_identities": timing_ids, "zero_basis": "NO_RECOVERY_BY_DEFINITION" if arm == "NONE" else None,
            "unavailable_reason": timed["unavailable_reason"] if seconds is None else None,
            "source_id": canonical_digest({"timing": timing["identity"], "arm": arm})})
        result["table3"].append({"row_id": method, "arm": arm, "block": "PAIRED_RELEASED_POOL", "cells": cells})
    return _seal({"status": PARTIAL_RESULT if any(value["status"] == PARTIAL_RESULT for value in (store, diagnosis, timing)) else "COMPLETE",
        "tables": result, "result_store_identity": store["identity"],
        "external_reference_identity": reference["identity"], "external_source_version": reference["source_version"],
        "diagnosis_identity": diagnosis["identity"], "timing_identity": timing["identity"], "minimum_body_font_pt": 9})


def format_cell(cell):
    unit = cell["display_unit"]
    value_key = {"percent": "value_fraction", "count_ratio": "numerator", "count": "value_count",
                 "seconds_per_scene": "value_seconds"}.get(unit)
    if value_key is not None and cell[value_key] is None:
        if not cell.get("unavailable_reason"):
            raise ValueError("null cells require an explicit unavailable reason")
        return "--"
    if unit == "percent":
        value = cell["value_fraction"]
        if value is None:
            if not cell.get("unavailable_reason"):
                raise ValueError("null cells require an explicit unavailable reason")
            return "--"
        percent = cell.get("published_percent", value * 100)
        return f'{percent:.{cell["display_decimal_places"]}f}'
    if unit == "count_ratio":
        return f'${cell["numerator"]}/{cell["denominator"]}$'
    if unit == "count":
        return str(cell["value_count"]) + (r"$\dagger$" if cell.get("ambiguous_additional_entries") else "")
    if unit == "seconds_per_scene":
        return "0" if cell["zero_basis"] else f'{cell["value_seconds"]:.2f}'
    raise ValueError("unknown typed display unit")


def render_tables(tables):
    _verified_identity(tables)
    result = {}
    for table, filename in TABLE_FILES.items():
        source = (PACKAGE / "tables" / filename).read_text()
        lines = source.splitlines()
        rows, prefixes = tables["tables"][table], ROW_LABELS[table]
        if len(rows) != len(prefixes):
            raise ValueError("rendered table changed its fixed6/6/4 data-row layout")
        for prefix, row in zip(prefixes, rows, strict=True):
            indices = [i for i, line in enumerate(lines) if line.startswith(prefix + " & ")]
            if len(indices) != 1:
                raise ValueError("supplied template data row is missing or duplicated: " + prefix)
            lines[indices[0]] = prefix + " & " + " & ".join(format_cell(cell) for cell in row["cells"]) + r" \\"
        lines[0] = "% Generated from typed receipt-backed tables_main.json."
        suffix = " Undefined metrics are --; reasons are recorded in cell provenance."
        if tables["status"] == PARTIAL_RESULT:
            suffix += " Technical blocks are --: office1 has no positive FC support at the fixed dense resolution; full cohorts are never replaced by subsets."
        if table == "table1":
            suffix += (" External rows use the CVF final PDF, Table 3, checked against the arXiv v1 PDF; "
                       "the supplied HTML transcription differences are retained in the source audit. "
                       "The authors' ScanNet sequence membership is not verified as CF18.")
        elif table == "table2":
            suffix += " A5 retains Native-derived incumbent support; FC-only denotes its conditional readout."
        else:
            suffix += (r" $\dagger$ marks additional mixed old/new tied score entries with unresolved attribution; "
                       "displayed TP/FP counts contain only definite added entries.")
        caption = next(i for i, line in enumerate(lines) if line.startswith(r"\caption{"))
        lines[caption] = lines[caption][:-1] + suffix + "}"
        result[filename] = "\n".join(lines) + "\n"
    preview = (PACKAGE / "tables/table_layout_preview.tex").read_text()
    start, stop = preview.index(r"\noindent\textbf"), preview.index(r"\input{table1_main.tex}")
    result["table_layout_preview.tex"] = preview[:start] + preview[stop:]
    return result


def _complete(path, index):
    index.identity(path)
    result = read(path)
    _verified_identity(result)
    if result["status"] != "COMPLETE":
        raise ValueError("publication input is incomplete: " + str(path))
    for item in result.get("inputs", []) + result.get("outputs", []):
        index.identity(item["path"], item)
    return result


def _evidence(path, index):
    index.identity(path)
    result = read(path)
    _verified_identity(result)
    if result["status"] not in ("COMPLETE", PARTIAL_RESULT):
        raise ValueError("evidence requires complete or explicitly blocked fixed coverage: " + str(path))
    for item in result.get("inputs", []) + result.get("outputs", []):
        index.identity(item["path"], item)
    return result


def collect_actual_physical_costs(binding, scenes, *, index):
    from .partial_execution import semantic_block
    from .timing import _validate_unmeasured_block, recovery_arm

    root, records, blocked, cold_calls = Path(binding["output_root"]).resolve(), [], [], 0
    resolver = PathResolver(binding["path_map"])

    def collect(scene, stage, path, *, cold=False, archived=True):
        path = Path(resolver.resolve(path)).resolve()
        candidates = sorted(path.parent.glob(path.stem + ".failed_*.json")) if archived else []
        for candidate in [*candidates, path]:
            item = index.identity(candidate)
            receipt = read(candidate)
            if "identity" in receipt:
                _verified_identity(receipt)
            for source in receipt.get("inputs", []) + receipt.get("outputs", []):
                index.identity(source["path"], source)
            records.append({"scene": scene, "stage": stage, "receipt": receipt, "file": item,
                            "current_task": candidate.is_relative_to(root), "cold": cold})

    for scene in scenes:
        context = root / "contexts" / (scene + ".json")
        if context.is_file():
            index.identity(context)
            data = read(context)
            _verified_identity(data)
        else:
            data = binding["scenes"][scene]
        collect(scene, "N_Q_SHARED", data["native_query_receipt"])
        collect(scene, "FC_STATIC_SHARED", Path(data["fc_root"]) / "receipt.json")
        collect(scene, "COMMON_MAPPING", data["parent_map_receipt"])
        collect(scene, "FC_RECOVERY_SHARED", root / "recovery" / scene / "receipt.json")
        independent = root / "recovery" / (scene + "_U2") / "receipt.json"
        if independent.is_file():
            extra = _complete(independent, index)
            if (extra["scene"] != scene or extra.get("cold") is not False
                    or extra.get("GT_input") is not False or extra["arms"] != ["U2"]):
                raise ValueError("independent U2 costs require an actual same-scene warm worker")
            collect(scene, "FC_RECOVERY_SHARED", independent)
        collect(scene, "NATIVE_RECOVERY", root / "native_recovery" / scene / "receipt.json")
        if scene in binding["cohorts"]["scannet_cf18"]:
            collect(scene, "CROPFORMER_SHARED", root / "frontend" / scene / "cropformer/receipt.json")
        if scene not in binding["cohorts"]["replica8"]:
            continue
        for arm in ("ARCHIVED_U2_FC", "G1_FC", "G3_FC"):
            path = root / "timing" / scene / arm / "receipt.json"
            item = index.identity(path)
            leaf = read(path)
            _verified_identity(leaf)
            if leaf["scene"] != scene or leaf["arm"] != arm:
                raise ValueError("cold cost leaves must retain their fixed scene/arm")
            for source in leaf["inputs"] + leaf["outputs"]:
                index.identity(source["path"], source)
            call = path.parent / "call/receipt.json"
            if leaf["status"] == "BLOCKED_UNMEASURED":
                _validate_unmeasured_block(leaf)
                parents = [read(source["path"]) for source in leaf["inputs"][:2]]
                if (call.exists() or leaf["technical_block"] != semantic_block(*parents, recovery_arm(arm))):
                    raise ValueError("blocked unreserved cold leaf cannot have a physical call or fabricated failure")
                blocked.append({"scene": scene, "arm": arm, "receipt": item,
                                "technical_block": leaf["technical_block"], "physical_call_executed": False})
            elif leaf["status"] == "COMPLETE":
                if Path(leaf["recovery_receipt"]).resolve() != call.resolve():
                    raise ValueError("cold cost must resolve to its actual isolated measured call")
                collect(scene, "FC_RECOVERY_COLD", call, cold=True, archived=False)
                cold_calls += 1
            else:
                raise ValueError("cold cost cannot infer a missing or failed measurement as an unreserved block")
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
        cold_model_loading_seconds_total=sum(row["event"]["seconds"] for row in loading),
        scene_order=list(scenes), shared_workers_counted_once=True,
        complete_cold_call_count=cold_calls, unreserved_blocked_cold_leaf_count=len(blocked),
        unreserved_blocked_cold_leaves=blocked)
    return _seal(result)


def build_tables(binding, diagnostics):
    from .evaluation import trace_class_metrics
    from .partial_execution import _pool_partition

    spec, root = load_spec(binding["spec"]), Path(binding["output_root"])
    for scene in [*spec["cohorts"]["replica8"], *spec["cohorts"]["scannet_cf18"]]:
        require_frozen_execution(binding, scene)
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    _verified_identity(diagnostics)
    expected = [r["scene"] for r in experiment_matrix(spec)["anchors"]]
    if diagnostics["status"] not in ("COMPLETE", PARTIAL_RESULT) or diagnostics["scene_order"] != expected:
        raise ValueError("publication needs all26 fixed post-lock diagnoses with explicit blocked scope")
    for item in diagnostics["outputs"]:
        index.identity(item["path"], item)
    diagnosis = _evidence(root / "diagnostics/replica8_recovery.json", index)
    timing = _evidence(root / "timing/pool.json", index)
    reference_path = root / "external/reference.json"
    index.identity(reference_path)
    reference = read(reference_path)
    _verified_identity(reference)
    for item in reference["inputs"]:
        index.identity(item["path"], item)
    scenes = {s: _evidence(root / "evaluation" / s / "receipt.json", index) for s in expected}
    pools = {}
    for cohort in spec["cohorts"]:
        receipt = _evidence(root / "pools" / cohort / "receipt.json", index)
        if receipt["scene_order"] != spec["cohorts"][cohort]:
            raise ValueError("publication pool changed its full ordered cohort")
        methods, blocked = _pool_partition(spec, cohort, {scene: scenes[scene] for scene in receipt["scene_order"]})
        if set(receipt["methods"]) != set(methods) or receipt.get("blocked_methods", {}) != blocked:
            raise ValueError("pool availability differs from actual scene scoring dependencies")
        for method, value in receipt["methods"].items():
            pools[(cohort, method)] = _complete(root / "pools" / cohort / (method + ".json"), index)
            if value != pools[(cohort, method)]:
                raise ValueError("pool manifest differs from its actual method receipt")
    store = result_store(spec, scenes, pools, root)
    tables = main_tables(spec, store, reference, diagnosis, timing, root)
    costs, per_scene, per_class_pools = [], [], []
    for scene, receipt in scenes.items():
        lock = read(root / "predictions" / scene / "receipt.json")
        _verified_identity(lock)
        for row in receipt["rows"]:
            scoring = read(row["evaluation_receipt"])
            per_scene.append({"scene": scene, "method_id": row["method"],
                "evaluation_identity": row["evaluation_identity"], "classes": trace_class_metrics(scoring, index=index)})
            payload = load_prediction(lock["predictions"][row["method"]])
            dependency = payload.metadata.get("dependency_costs", {})
            _verified_identity(dependency)
            if dependency["status"] not in ("COMPLETE", "PARTIAL_CONTENT_IDENTITIES") or not dependency["native_support_prerequisite_included"]:
                raise ValueError("main results lack honest method-specific common Native support costs")
            costs.append({"scene": scene, "method_id": row["method"], "prediction_key": payload.prediction_key,
                          "logical_cost": payload.logical_cost, "dependency_costs": dependency,
                          "metadata": payload.metadata})
    for (cohort, method), receipt in pools.items():
        item = receipt["per_class_receipt"]
        index.identity(item["path"], item)
        details = read(item["path"])
        _verified_identity(details)
        if details["ordered_inputs"] != receipt["ordered_inputs"]:
            raise ValueError("per-class pool changed its complete ordered official scoring inputs")
        per_class_pools.append({"cohort": cohort, "method_id": method, "receipt": item, "classes": details["classes"]})
    blocked_costs = [{key: row[key] for key in ("scene", "cohort", "method_id", "status", "unavailable_reason",
        "blocked_condition", "technical_block", "receipt_path", "receipt_identity")}
        for row in store["scene_metrics"] if row["status"] == "BLOCKED_TECHNICAL"]
    physical_costs = collect_actual_physical_costs(binding, expected, index=index)
    supplement = _seal({"status": store["status"], "artifact_generation_status": "COMPLETE",
        "main_scene_outputs": store["main_scene_outputs"], "internal_pools": store["internal_pools"],
        "metric_unit": "FRACTION", "scene_metrics": store["scene_metrics"],
        "pooled_metrics": store["pooled_metrics"], "per_class_scene_metrics": per_scene,
        "per_class_pooled_metrics": per_class_pools, "costs": costs, "blocked_method_dependencies": blocked_costs,
        "physical_costs": physical_costs,
        "post_lock_diagnostics": diagnostics, "cold_timing": timing, "external_provenance": reference})
    destination = root / "tables"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "receipt.json").is_file():
        previous = _evidence(destination / "receipt.json", index)
        if (previous["result_store_identity"] != store["identity"] or previous["tables_identity"] != tables["identity"]
                or _evidence(destination / "supplement.json", index)["identity"] != supplement["identity"]):
            raise ValueError("completed measured tables changed; invalidate their affected descendants explicitly")
        return previous
    for filename, value in (("scene_metrics.json", store["scene_metrics"]), ("pooled_metrics.json", store["pooled_metrics"]),
            ("result_store.json", store), ("tables_main.json", tables), ("supplement.json", supplement),
            ("costs.json", _seal({"status": store["status"], "method_dependencies": costs,
                "blocked_method_dependencies": blocked_costs, "main_scene_outputs": store["main_scene_outputs"],
                "physical_payments": physical_costs}))):
        atomic_write_json(destination / filename, value)
    cells = [cell for rows in tables["tables"].values() for row in rows for cell in row["cells"]]
    atomic_write_json(destination / "cell_provenance.json", _seal({"cells": cells, "tables_identity": tables["identity"]}))
    buffer = io.StringIO()
    fields = ("table_id", "row_id", "column_id", "metric", "source_kind", "display_unit", "display", "value_fraction",
              "value_count", "value_seconds", "numerator", "denominator", "source_id", "receipt_identity")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for cell in cells:
        writer.writerow({name: format_cell(cell) if name == "display" else cell.get(name) for name in fields})
    (destination / "tables_main.csv").write_text(buffer.getvalue())
    for filename, source in render_tables(tables).items():
        (destination / filename).write_text(source)
    log = destination / "latex_build.log"
    with log.open("wb") as stream:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "table_layout_preview.tex"],
                           cwd=destination, stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = {"status": tables["status"], "artifact_generation_status": "COMPLETE", "scene_count": 26,
        "prediction_count": store["main_scene_outputs"]["complete"], "required_prediction_count": 172,
        "pool_count": store["internal_pools"]["complete"], "required_pool_count": 14,
        "main_scene_outputs": store["main_scene_outputs"], "internal_pools": store["internal_pools"],
        "table_row_counts": {name: len(rows) for name, rows in tables["tables"].items()},
        "result_store_identity": store["identity"], "tables_identity": tables["identity"],
        "minimum_body_font_pt": 9, "latex_compilation": "PASS", "visual_pdf_QA": "PENDING",
        "outputs": [index.identity(path) for path in sorted(destination.iterdir()) if path.is_file() and path.name != "receipt.json"],
        "inputs": index.entries()}
    _seal(result)
    atomic_write_json(destination / "receipt.json", result)
    index.write_memo(root / "validation/input_verifications.json")
    return result
