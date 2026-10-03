"""Complete receipt-backed result store, typed cells and supplied booktabs layouts."""

import csv
import io
from pathlib import Path
import subprocess

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .costs import collect_physical_costs
from .diagnostics import TABLE3_METHODS
from .evaluation import METRICS, fraction_metrics, select_pool_rows
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
    expected_pools = {(r["cohort"], r["method"]) for r in matrix["pools"]}
    if set(pool_receipts) != expected_pools:
        raise ValueError("result store requires all14 complete fixed official pools")
    scene_records, pool_records = [], []
    for anchor in matrix["anchors"]:
        scene, cohort = anchor["scene"], anchor["cohort"]
        receipt = scene_receipts[scene]
        _verified_identity(receipt)
        methods = [r["id"] for r in spec["methods"] if cohort in r["cohorts"]]
        if (receipt["status"] != "COMPLETE" or receipt["scene"] != scene or receipt["cohort"] != cohort
                or len(receipt["rows"]) != len(methods)
                or {r["method"] for r in receipt["rows"]} != set(methods)):
            raise ValueError("result store scene has incomplete or extra fixed conditions")
        for method in methods:
            row = next(r for r in receipt["rows"] if r["method"] == method)
            _verified_identity(row)
            if row["status"] != "COMPLETE" or row["scene"] != scene or row["cohort"] != cohort:
                raise ValueError("result store row leaves its exact completed scene/cohort")
            if row["rank_mode"] != "OFFICIAL_CURRENT_CLASS" or row["metric_unit"] != "FRACTION":
                raise ValueError("result store changed its official rank or fractional units")
            validate_overlaps(row["runtime_overlaps"])
            scene_records.append({"scene": scene, "cohort": cohort, "method_id": method,
                "metrics": fraction_metrics(row["metrics"]), "metric_unit": "FRACTION",
                "receipt_path": str(root / "evaluation" / scene / "rows" / (method + ".json")),
                "receipt_identity": row["identity"], "evaluation_identity": row["evaluation_identity"],
                "scorer_context_identity": row["scorer_context_identity"],
                "source_kind": "EXACT_REUSE" if row.get("alias_proof") else "MEASURED"})
    for planned in matrix["pools"]:
        cohort, method = planned["cohort"], planned["method"]
        receipt = pool_receipts[(cohort, method)]
        _verified_identity(receipt)
        selected = select_pool_rows(spec, cohort, method,
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
        pool_records.append({"cohort": cohort, "method_id": method, "metrics": fraction_metrics(receipt["metrics"]),
            "metric_unit": "FRACTION", "ordered_scene_ids": planned["scene_order"],
            "completed_coverage": len(selected), "required_coverage": len(planned["scene_order"]),
            "scoring_protocol_id": canonical_digest(protocol), "scoring_definition": protocol,
            "receipt_path": str(root / "pools" / cohort / (method + ".json")),
            "receipt_identity": receipt["identity"], "row_identities": receipt["row_identities"],
            "source_kind": "MEASURED"})
    return _seal({"status": "COMPLETE", "matrix_identity": matrix["identity"], "metric_unit": "FRACTION",
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
        "unavailable_reason": "OFFICIAL_SCORER_UNDEFINED_METRIC" if value is None else None}


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
    if (store["status"] != "COMPLETE" or reference["status"] != "AUTHOR_REPORTED_NOT_REPRODUCED"
            or diagnosis["status"] != "COMPLETE" or diagnosis["scene_count"] != 8
            or timing["status"] != "COMPLETE" or timing["leaf_count"] != 24):
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
                or not 0 <= row["n"] <= row["N"]):
            raise ValueError("Table3 diagnosis leaves its shared complete eight-scene candidate registry")
        cells = []
        for column, value in (("max_views", layout["table3"]["max_views"][position]),
                ("recovered_n_over_N", None), ("added_tp50_entries", row["added_TP50"]),
                ("added_fp50_entries", row["added_FP50"])):
            cell = {"table_id": "table3", "row_id": method, "column_id": column, "metric": column,
                "source_kind": "EXACT_REUSE" if column == "max_views" else "MEASURED",
                "cohort": "replica8", "ordered_scene_ids": scenes, "completed_coverage": 8, "required_coverage": 8,
                "method_id": method, "scoring_protocol_id": "RELEASED_SCORE_ENTRIES_AT_ACTUAL_AP50",
                "receipt_path": str(root / "diagnostics/replica8_recovery.json"), "receipt_identity": diagnosis["identity"]}
            if column == "max_views":
                cell.update(receipt_path=str(PACKAGE / "PROTOCOL_SPEC.json"), receipt_identity=canonical_digest(spec),
                            scoring_protocol_id="FROZEN_MAX_VIEW_LIMIT")
            if column == "recovered_n_over_N":
                cell.update(display_unit="count_ratio", numerator=row["n"], denominator=row["N"])
            else:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("Table3 count cells require nonnegative integer counts")
                cell.update(display_unit="count", value_count=value)
                if column.startswith("added_"):
                    cell["ambiguous_additional_entries"] = row["ambiguous_added_TP50" if column == "added_tp50_entries"
                                                             else "ambiguous_added_FP50"]
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
                    or not np.isfinite(seconds) or seconds <= 0):
                raise ValueError("Table3 time must average all eight actual measured scenes")
        cells.append({"table_id": "table3", "row_id": method, "column_id": "cold_feature_incremental_seconds_mean",
            "metric": "cold_feature_incremental_seconds_mean", "source_kind": "EXACT_REUSE" if arm == "NONE" else "MEASURED",
            "value_seconds": seconds, "display_unit": "seconds_per_scene", "cohort": "replica8",
            "ordered_scene_ids": scenes, "completed_coverage": 8, "required_coverage": 8, "method_id": method,
            "scoring_protocol_id": "FEATURE_CACHE_COLD_INCREMENTAL_RECOVERY_MODEL_RESIDENT",
            "receipt_path": str(root / "timing/pool.json"), "receipt_identity": timing["identity"],
            "measurement_receipt_identities": timing_ids, "zero_basis": "NO_RECOVERY_BY_DEFINITION" if arm == "NONE" else None,
            "source_id": canonical_digest({"timing": timing["identity"], "arm": arm})})
        result["table3"].append({"row_id": method, "arm": arm, "block": "PAIRED_RELEASED_POOL", "cells": cells})
    return _seal({"status": "COMPLETE", "tables": result, "result_store_identity": store["identity"],
        "external_reference_identity": reference["identity"], "external_source_version": reference["source_version"],
        "diagnosis_identity": diagnosis["identity"], "timing_identity": timing["identity"], "minimum_body_font_pt": 9})


def format_cell(cell):
    unit = cell["display_unit"]
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


def build_tables(binding, diagnostics):
    from .evaluation import trace_class_metrics

    spec, root = load_spec(binding["spec"]), Path(binding["output_root"])
    for scene in [*spec["cohorts"]["replica8"], *spec["cohorts"]["scannet_cf18"]]:
        require_frozen_execution(binding, scene)
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    _verified_identity(diagnostics)
    expected = [r["scene"] for r in experiment_matrix(spec)["anchors"]]
    if diagnostics["status"] != "COMPLETE" or diagnostics["scene_order"] != expected:
        raise ValueError("publication needs all26 complete fixed post-lock diagnoses")
    for item in diagnostics["outputs"]:
        index.identity(item["path"], item)
    diagnosis = _complete(root / "diagnostics/replica8_recovery.json", index)
    timing = _complete(root / "timing/pool.json", index)
    reference_path = root / "external/reference.json"
    index.identity(reference_path)
    reference = read(reference_path)
    _verified_identity(reference)
    for item in reference["inputs"]:
        index.identity(item["path"], item)
    scenes = {s: _complete(root / "evaluation" / s / "receipt.json", index) for s in expected}
    pools = {}
    for cohort in spec["cohorts"]:
        receipt = _complete(root / "pools" / cohort / "receipt.json", index)
        if receipt["scene_order"] != spec["cohorts"][cohort]:
            raise ValueError("publication pool changed its full ordered cohort")
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
    physical_costs = collect_physical_costs(binding, expected, index=index)
    supplement = _seal({"status": "COMPLETE", "metric_unit": "FRACTION", "scene_metrics": store["scene_metrics"],
        "pooled_metrics": store["pooled_metrics"], "per_class_scene_metrics": per_scene,
        "per_class_pooled_metrics": per_class_pools, "costs": costs, "physical_costs": physical_costs,
        "post_lock_diagnostics": diagnostics, "cold_timing": timing, "external_provenance": reference})
    destination = root / "tables"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "receipt.json").is_file():
        previous = _complete(destination / "receipt.json", index)
        if (previous["result_store_identity"] != store["identity"] or previous["tables_identity"] != tables["identity"]
                or read(destination / "supplement.json") != supplement):
            raise ValueError("completed measured tables changed; invalidate their affected descendants explicitly")
        return previous
    for filename, value in (("scene_metrics.json", store["scene_metrics"]), ("pooled_metrics.json", store["pooled_metrics"]),
            ("result_store.json", store), ("tables_main.json", tables), ("supplement.json", supplement),
            ("costs.json", _seal({"status": "COMPLETE", "method_dependencies": costs,
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
    result = {"status": "COMPLETE", "scene_count": 26, "prediction_count": 172, "pool_count": 14,
        "table_row_counts": {name: len(rows) for name, rows in tables["tables"].items()},
        "result_store_identity": store["identity"], "tables_identity": tables["identity"],
        "minimum_body_font_pt": 9, "latex_compilation": "PASS", "visual_pdf_QA": "PENDING",
        "outputs": [index.identity(path) for path in sorted(destination.iterdir()) if path.is_file() and path.name != "receipt.json"],
        "inputs": index.entries()}
    _seal(result)
    atomic_write_json(destination / "receipt.json", result)
    index.write_memo(root / "validation/input_verifications.json")
    return result
