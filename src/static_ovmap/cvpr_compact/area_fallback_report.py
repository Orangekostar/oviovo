"""Complete v2 accuracy tables and recovery counts with explicit timing scope."""

import copy
import csv
import io
from pathlib import Path
import subprocess

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.recovery_wave2.binding import read

from .area_fallback import PROTOCOL
from .area_fallback_experiment import document, environment, seal
from .diagnostics import TABLE3_METHODS, aggregate_recovery_diagnostics, analyze_scoring_comparison
from .evaluation import METRICS
from .tables import _measured_cell, render_tables, result_store


def cold_timing(output, spec, index):
    path = output / "timing" / "pool.json"
    if not path.exists():
        return None
    from .area_fallback_timing import ARMS, aggregate_v2_timings

    timing = document(path, index)
    for item in [timing["producer"], *timing["inputs"], *timing["outputs"]]:
        index.identity(item["path"], item)
    leaves = [document(output / "timing" / scene / arm / "receipt.json", index)
              for scene in spec["cohorts"]["replica8"] for arm in ARMS]
    aggregate = aggregate_v2_timings(spec, leaves)
    for key in ("status", "protocol", "arms", "hardware", "leaf_count", "definition"):
        if timing[key] != aggregate[key]:
            raise ValueError("v2 timing pool differs from its 16 verified measurement receipts")
    return timing


def recovery_diagnostics(parent, output, spec, scenes, index):
    reports = []
    for scene in spec["cohorts"]["replica8"]:
        lock = document(output / "predictions" / scene / "receipt.json", index)
        regions = document(output / "regions" / scene / "receipt.json", index)
        views = document(parent / "projected_views" / scene / "receipt.json", index)
        registry = document(views["registry"], index)
        rows = {row["method"]: row for row in scenes[scene]["rows"]}
        baseline_path = rows["CT_A1_E"]["evaluation_receipt"]
        baseline_prediction = document(lock["predictions"]["CT_A1_E"], index, verify=False)
        incumbents = baseline_prediction["metadata"]["owner_semantic_decisions"]
        conditions = {}
        for arm, method in TABLE3_METHODS.items():
            manifest = document(lock["predictions"][method], index, verify=False)
            added = set(map(int, manifest["metadata"]["recovered_labels"]))
            labels = manifest["metadata"]["owner_semantic_decisions"]
            if any(labels[owner] != label for owner, label in incumbents.items()):
                raise ValueError("v2 recovery changed an incumbent D2 semantic decision")
            if arm in regions["sources"]:
                item = regions["sources"][arm]
                source = document(item["path"], index, item)
                available = {int(owner) for owner, row in source["objects"].items() if row["available"]}
                if added != available:
                    raise ValueError("v2 recovered output disagrees with its semantic source availability")
            score_path = rows[method]["evaluation_receipt"]
            score = document(score_path, index, verify=False)
            eligible = added & set(map(int, score["view"]))
            comparison = analyze_scoring_comparison(baseline_path, score_path, added, index=index)
            conditions[method] = {"source_available_additions": len(added),
                "source_available_owners": sorted(added), "target_min100_additions": len(eligible),
                "target_small_additions": len(added - eligible), "old_incumbent_labels_exact": True,
                **comparison}
        report = seal({"status": "COMPLETE", "scene": scene, "protocol": PROTOCOL,
            "candidate_count": len(registry["candidates"]), "registry_identity": registry["identity"],
            "conditions": conditions, "prediction_lock_identity": lock["identity"],
            "GT_used_only_after_prediction_lock": True})
        atomic_write_json(output / "diagnostics" / (scene + ".json"), report)
        reports.append(report)
    diagnosis = aggregate_recovery_diagnostics(spec, reports)
    atomic_write_json(output / "diagnostics" / "replica8_recovery.json", diagnosis)
    return diagnosis


def typed_tables(parent, output, store, diagnosis, index, timing=None):
    tables = copy.deepcopy(document(parent / "tables" / "tables_main.json", index))
    for table in ("table1", "table2"):
        for row in tables["tables"][table]:
            if row["block"] == "AUTHOR_REPORTED_NOT_RERUN":
                continue
            row["cells"] = [_measured_cell(store, table, row["row_id"], cell["cohort"], cell["metric"])
                            for cell in row["cells"]]
            row["block"] = "V2_PAIRED_RELEASED_POOL"
    for row in tables["tables"]["table3"]:
        arm, method = row["arm"], row["row_id"]
        counts = diagnosis["arms"][arm]
        for cell in row["cells"]:
            metric = cell["metric"]
            if metric in ("apall", "miou"):
                cell.clear()
                cell.update(_measured_cell(store, "table3", method, "replica8", metric))
            elif metric in ("recovered_n_over_N", "added_tp50_entries", "added_fp50_entries"):
                cell.pop("unavailable_reason", None)
                cell.update(source_kind="MEASURED", completed_coverage=8,
                    receipt_path=str(output / "diagnostics" / "replica8_recovery.json"),
                    receipt_identity=diagnosis["identity"])
                if metric == "recovered_n_over_N":
                    cell.update(numerator=counts["n"], denominator=counts["N"])
                else:
                    tp = metric == "added_tp50_entries"
                    cell.update(value_count=counts["added_TP50" if tp else "added_FP50"],
                        ambiguous_additional_entries=counts["ambiguous_added_TP50" if tp else "ambiguous_added_FP50"])
                cell["source_id"] = seal({"diagnosis": diagnosis["identity"], "arm": arm, "metric": metric})["identity"]
            elif metric == "cold_feature_incremental_seconds_mean" and arm in ("G1", "G3"):
                if timing is not None:
                    measured = timing["arms"][arm + "_FC"]
                    cell.update(value_seconds=measured["mean_seconds"], source_kind="MEASURED", completed_coverage=8,
                        unavailable_reason=None, receipt_path=str(output / "timing" / "pool.json"),
                        receipt_identity=timing["identity"], measurement_receipt_identities=measured["receipt_identities"],
                        source_id=seal({"timing": timing["identity"], "arm": arm})["identity"])
                else:
                    cell.update(value_seconds=None, source_kind="UNAVAILABLE", completed_coverage=0,
                        unavailable_reason="V2_GPU_COLD_RECOVERY_NOT_MEASURED; CACHE_ASSISTED_CPU_TIME_IS_NOT_COLD_LATENCY",
                        receipt_path=str(output / "timing_scope.json"), measurement_receipt_identities=[])
        row["block"] = "V2_PAIRED_RELEASED_POOL"
    scope = seal({"status": "COMPLETE" if timing is not None else "NOT_MEASURED_FOR_V2_PROJECTED_RECOVERY", "protocol": PROTOCOL,
        "G1_seconds_per_scene": timing["arms"]["G1_FC"]["mean_seconds"] if timing is not None else None,
        "G3_seconds_per_scene": timing["arms"]["G3_FC"]["mean_seconds"] if timing is not None else None,
        "U2_control": "EXACT_UNCHANGED_ARCHIVED_V1_CONTROL",
        "parent_timing_receipt": str(parent / "timing" / "pool.json"),
        "measurement_receipt": str(output / "timing" / "pool.json") if timing is not None else None,
        "measurement_identity": timing["identity"] if timing is not None else None,
        "reason": "16 serial model-resident feature-cache-cold calls; all eight scenes per arm; exact discrete prediction parity."
            if timing is not None else "The v2 accuracy probe uses cached dense features and CPU projection heads; no v2 GPU cold recovery was replayed."})
    atomic_write_json(output / "timing_scope.json", scope)
    for row in tables["tables"]["table3"]:
        if timing is None and row["arm"] in ("G1", "G3"):
            cell = row["cells"][-1]
            cell["receipt_identity"] = scope["identity"]
            cell["source_id"] = seal({"timing": scope["identity"], "arm": row["arm"]})["identity"]
    tables.update(status="COMPLETE" if timing is not None else "ACCURACY_COMPLETE_TIMING_NOT_MEASURED", protocol=PROTOCOL,
                  result_store_identity=store["identity"], diagnosis_identity=diagnosis["identity"], timing_identity=scope["identity"])
    return seal(tables)


def markdown_report(parent, output, spec, store, diagnosis, summary, index, timing=None):
    methods = {row["id"]: row["short"] for row in spec["methods"]}
    lines = ["# Area Fallback V2 Results", "", "Protocol: `" + PROTOCOL + "`.", "",
        "Post-hoc diagnostic on already exposed scenes. Only empty projected G1/G3 FC masks use area-weighted pooling. "
        "The incumbent N/Q/F evidence, archived U2, fixed views, full-resolution masks, geometry, vocabulary, frozen head and released scoring are unchanged.", "",
        f"Coverage: {summary['scene_method_records']}/172 scene-method scores; {summary['complete_pools']}/14 complete official pools. "
        f"{summary['repaired_requests']}/{summary['original_empty_requests']} empty masks repaired; "
        f"{summary['reused_original_vectors']} original successful vectors reused bit-for-bit. "
        f"Accuracy probe: new GPU inference 0; new full-image encoding 0; CPU projection-head calls {summary['CPU_projection_head_calls']}.", "",
        "Methods: A0 = Native; A1 = D2 refinement; A2 = Native + repaired G1-FC recovery; "
        "A3 = D2 + repaired G1-FC recovery; A4 = D2 + original G1-Native recovery; "
        "A5 = existing FC-only readout + repaired G1-FC recovery; U2 = unchanged archived recovery; "
        "G3 = D2 + repaired three-view FC recovery.", "",
        "| Cohort | Method | APall (%) | AP50 (%) | AP25 (%) | mIoU (%) | mAcc (%) |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    pools = {(row["cohort"], row["method_id"]): row for row in store["pooled_metrics"]}
    for row in store["pooled_metrics"]:
        values = [f"{row['metrics'][metric] * 100:.2f}" for metric in METRICS]
        lines.append("| " + " | ".join([row["cohort"], methods[row["method_id"]], *values]) + " |")
    lines += ["", "## A3 Compared With A1", "",
              "| Cohort | APall (pp) | AP50 (pp) | AP25 (pp) | mIoU (pp) | mAcc (pp) |",
              "| --- | --- | --- | --- | --- | --- |"]
    deltas = {}
    for cohort in spec["cohorts"]:
        before = pools[(cohort, "CT_A1_E")]["metrics"]
        after = pools[(cohort, "CT_A3_ER")]["metrics"]
        deltas[cohort] = {metric: (after[metric] - before[metric]) * 100 for metric in METRICS}
        lines.append("| " + " | ".join([cohort, *[f"{deltas[cohort][metric]:+.2f}" for metric in METRICS]]) + " |")
    lines += ["", "## V1 To V2 On The Identical Complete Cohort", "",
        "Replica v1 recovery pools were unavailable because office1 was blocked; no v1-to-v2 full-Replica gain can be claimed. "
        "The following CF18 changes use the same full 18-scene cohort.", "",
        "| Method | APall (pp) | AP50 (pp) | AP25 (pp) | mIoU (pp) | mAcc (pp) |",
        "| --- | --- | --- | --- | --- | --- |"]
    for method in ("CT_A2_R", "CT_A3_ER", "CT_A5_FC_ONLY"):
        original = document(parent / "pools" / "scannet_cf18" / (method + ".json"), index)
        current = pools[("scannet_cf18", method)]["metrics"]
        values = [f"{(current[metric] - original['metrics'][metric]) * 100:+.4f}" for metric in METRICS]
        lines.append("| " + " | ".join([methods[method], *values]) + " |")
    lines += ["", "## Replica Recovery", "",
              "| Arm | Source-available n/N | Added TP50 | Added FP50 | Ambiguous TP/FP | Cold Seconds/Scene |",
              "| --- | --- | --- | --- | --- | --- |"]
    parent_timing = document(parent / "timing" / "pool.json", index)
    for arm, counts in diagnosis["arms"].items():
        seconds = ("0 (by definition)" if arm == "NONE" else
            f"{parent_timing['arms']['ARCHIVED_U2_FC']['mean_seconds']:.2f} (unchanged control)" if arm == "U2" else
            f"{timing['arms'][arm + '_FC']['mean_seconds']:.2f}" if timing is not None else "-- (v2 unmeasured)")
        lines.append("| " + " | ".join([arm, f"{counts['n']}/{counts['N']}", str(counts["added_TP50"]),
            str(counts["added_FP50"]), f"{counts['ambiguous_added_TP50']}/{counts['ambiguous_added_FP50']}", seconds]) + " |")
    if timing is not None:
        lines += ["", "## V2 Cold Remeasurement", "",
            "| Scene | G1 (s) | G3 (s) | Discrete Output Parity |", "| --- | --- | --- | --- |"]
        leaves = []
        for scene in spec["cohorts"]["replica8"]:
            pair = [document(output / "timing" / scene / (arm + "_FC") / "receipt.json", index) for arm in ("G1", "G3")]
            leaves.extend(pair)
            lines.append(f"| {scene} | {pair[0]['measured_seconds']:.3f} | {pair[1]['measured_seconds']:.3f} | exact / exact |")
        maximum = max(row["parity"]["maximum_feature_absolute_difference"] for row in leaves)
        lines += ["", f"16/16 serial calls completed on {timing['hardware']['name']} ({timing['hardware']['uuid']}). "
            f"New GPU full-image encodings: {timing['physical_image_encodings']}; GPU region-pool/head calls: {timing['physical_region_poolings']}; "
            f"area-fallback poolings: {timing['area_fallback_poolings']}. Model loading: {timing['model_loading_seconds_total']:.3f}s, excluded from the means.", "",
            "Each mean is the sum of all eight scene costs divided by eight (one sample per scene/arm). "
            "Timing includes fresh projection, RGB decode/preprocessing, image encoding, pooling/head, classification and output/rank export. "
            "Model loading, geometry mapping and benchmark evaluation are excluded. Feature, view and result caches are disabled; "
            "same-frame sharing is allowed only within one call. Feature-cache-cold does not mean the OS page cache was cleared.", "",
            f"All fixed masks, view selections, source labels/availability, full instance support, semantic labels and current-class ranks match the v2 scientific outputs. "
            f"Maximum FP32 feature absolute difference: {maximum:.9g}; inherited tolerance: atol=rtol=1e-5. Accuracy metrics are unchanged."]
    lines += ["", "TP/FP are definite entries from the unchanged released matcher, not unique recovered objects. "
        "Source availability is not correct detection. Semantic unknown errors remain in whole-scene confusion matrices.", "",
        "## Artifacts", "", "The three filled LaTeX tables, typed cell provenance, per-scene CSV, complete result store, "
        "official per-class pool receipts and preview PDF are under `tables/`. Full predictions, vectors and scoring traces remain in this independent attempt.", "",
        "G1/G3 latency comes from the independent v2 GPU cold remeasurement; archived U2 retains its unchanged v1 control measurement. Deployment is unchanged."
        if timing is not None else "Cache-assisted CPU pooling/head times are not GPU cold latency. G1/G3 latency is unmeasured in v2; v1 timing is not carried across the changed protocol. Deployment is unchanged.", "",
        f"Parent: `{parent}`. V2 attempt: `{output}`."]
    return "\n".join(lines) + "\n", deltas


def build_report(parent, output):
    parent, output, _, spec, index = environment(parent, output)
    experiment = document(output / "experiment.json", index)
    for item in experiment["inputs"]:
        index.identity(item["path"], item)
    audit = document(output / "weight_audit.json", index, verify=False)
    if audit["branch"] != "FC_FROZEN" or not audit["strict"] or audit["random_active_parameters"]:
        raise ValueError("v2 projection head lacks a strict original frozen weight audit")
    scenes = {scene: document(output / "evaluation" / scene / "receipt.json", index) for scene in experiment["scenes"]}
    pools = {(cohort, row["id"]): document(output / "pools" / cohort / (row["id"] + ".json"), index)
             for cohort in spec["cohorts"] for row in spec["methods"] if cohort in row["cohorts"]}
    store = result_store(spec, scenes, pools, output)
    diagnosis = recovery_diagnostics(parent, output, spec, scenes, index)
    summary = {"protocol": PROTOCOL, "scene_count": len(scenes), "scene_method_records": len(store["scene_metrics"]),
        "complete_pools": len(pools), "original_empty_requests": sum(len(row["empty_requests"]) for row in experiment["scenes"].values()),
        "repaired_requests": 0, "reused_original_vectors": 0, "CPU_projection_head_calls": 0,
        "accuracy_probe_GPU_inference_calls": 0, "accuracy_probe_full_image_encodings": 0,
        "changed_prediction_count": 0, "reused_prediction_count": 0}
    all_outputs, per_scene = [], []
    for scene in scenes:
        regions = document(output / "regions" / scene / "receipt.json", index)
        original = document(parent / "recovery" / scene / "receipt.json", index)
        lock = document(output / "predictions" / scene / "receipt.json", index)
        if set(regions["repaired_requests"]) != set(experiment["scenes"][scene]["empty_requests"]):
            raise ValueError("fallback scope differs from the preflight empty-mask request set")
        summary["repaired_requests"] += len(regions["repaired_requests"])
        summary["reused_original_vectors"] += len(regions["original_vector_sha256"])
        summary["CPU_projection_head_calls"] += regions["CPU_projection_head_calls"]
        summary["changed_prediction_count"] += len(lock["changed_methods"])
        summary["reused_prediction_count"] += len(lock["aliases"])
        if regions["physical_image_encodings"] or regions["GPU_inference_calls"]:
            raise ValueError("the cache-only v2 probe unexpectedly encoded an image or used a GPU")
        features = regions["features"]
        index.identity(features["path"], features)
        with np.load(features["path"], allow_pickle=False) as arrays:
            vectors = {str(rid): vector for rid, vector in zip(arrays["request_ids"], arrays["features"])}
        selected = {rid for arm in regions["plan"].values() for rows in arm.values() for rid in rows}
        if set(vectors) != selected or len(regions["requests"]) != len(selected):
            raise ValueError("v2 omitted a selected projected feature or changed its view set")
        if original.get("features"):
            item = original["features"]
            index.identity(item["path"], item)
            with np.load(item["path"], allow_pickle=False) as arrays:
                parent_vectors = {str(rid): vector for rid, vector in zip(arrays["request_ids"], arrays["features"])}
            for rid, digest in regions["original_vector_sha256"].items():
                if not np.array_equal(vectors[rid], parent_vectors[rid]) or _array_digest(vectors[rid]) != digest:
                    raise ValueError("v2 changed an original successful FP32 feature")
        for rid in regions["repaired_requests"]:
            row = regions["requests"][rid]
            if (row["original_support"] != 0 or row["area_support"] <= 0 or row["area_mass"] <= 0
                    or not np.isfinite(vectors[rid]).all() or not np.isclose(np.linalg.norm(vectors[rid]), 1., rtol=0, atol=1e-5)):
                raise ValueError("an area-repaired region lacks nonempty support or a valid final unit vector")
        for row in scenes[scene]["rows"]:
            score = document(row["evaluation_receipt"], index, verify=False)
            check = scenes[scene]["registry_checks"][row["method"]]
            if int(np.asarray(score["confusion"]).sum()) != check["semantic_valid_target_count"]:
                raise ValueError("v2 scoring omitted valid whole-scene semantic target points")
            per_scene.append({"scene": scene, "cohort": row["cohort"], "method": row["method"],
                "reuse_kind": row["reuse_kind"], **{metric + "_percent": row["metrics"][metric] * 100 for metric in METRICS}})
        for stage in (regions, lock, scenes[scene]):
            all_outputs.extend(stage["outputs"])
    if (summary["scene_method_records"] != 172 or summary["complete_pools"] != 14
            or summary["repaired_requests"] != summary["original_empty_requests"]):
        raise ValueError("v2 scientific coverage is incomplete")
    timing = cold_timing(output, spec, index)
    if timing is not None:
        summary.update(cold_call_count=timing["leaf_count"], cold_GPU_full_image_encodings=timing["physical_image_encodings"],
            cold_GPU_pool_head_calls=timing["physical_region_poolings"], cold_area_fallback_poolings=timing["area_fallback_poolings"])
        all_outputs.extend([index.identity(output / "timing" / "pool.json"), *timing["outputs"]])
    tables = typed_tables(parent, output, store, diagnosis, index, timing)
    destination = output / "tables"
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in (("result_store.json", store), ("tables_main.json", tables), ("validation_summary.json", summary)):
        atomic_write_json(destination / name, value)
    atomic_write_json(destination / "cell_provenance.json", seal({"protocol": PROTOCOL,
        "cells": [cell for rows in tables["tables"].values() for row in rows for cell in row["cells"]]}))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(per_scene[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(per_scene)
    (destination / "scene_metrics.csv").write_text(buffer.getvalue())
    for filename, source in render_tables(tables).items():
        if filename.endswith("_main.tex") or filename.endswith("_ablation.tex") or filename.endswith("_recovery.tex"):
            caption = source.index("\\caption{")
            source = source[:caption] + source[caption:].replace("\\caption{", "\\caption{V2 diagnostic: empty projected FC masks use area-weighted fallback. ", 1)
            if filename == "table3_recovery.tex":
                timing_caption = ("G1/G3 latency is remeasured on all eight scenes with the resident model and no feature/view/result cache; U2 is the unchanged control. "
                    if timing is not None else "G1/G3 v2 GPU cold latency is unmeasured; CPU cache time is excluded. ")
                source = source.replace("\\caption{", "\\caption{" + timing_caption, 1)
        (destination / filename).write_text(source)
    with (destination / "latex_build.log").open("wb") as stream:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "table_layout_preview.tex"],
                           cwd=destination, stdout=stream, stderr=subprocess.STDOUT, check=True)
    subprocess.run(["pdftoppm", "-scale-to", "2200", "-singlefile", "-png", "table_layout_preview.pdf", "table_layout_preview"],
                   cwd=destination, check=True)
    text, deltas = markdown_report(parent, output, spec, store, diagnosis, summary, index, timing)
    (output / "AREA_FALLBACK_V2_RESULTS.md").write_text(text)
    all_outputs.extend(index.identity(path) for path in destination.iterdir() if path.is_file())
    all_outputs.extend([index.identity(output / "AREA_FALLBACK_V2_RESULTS.md"),
                        index.identity(output / "weight_audit.json"), index.identity(output / "timing_scope.json")])
    unique = {row["path"]: row for row in all_outputs}
    for item in unique.values():
        index.identity(item["path"], item)
    validation = seal({"status": "PASS", "protocol": PROTOCOL, **summary,
        "A3_vs_A1_pp": deltas, "all_fixed_views_preserved": True, "original_successful_vectors_bit_exact": True,
        "complete_ordered_official_pools": True, "whole_scene_semantic_errors_preserved": True,
        "cold_timing": "16_COMPLETE_V2_CALLS_EXACT_DISCRETE_PARITY" if timing is not None else "NOT_MEASURED_FOR_V2_G1_G3", "parent_v1_read_only": True,
        "outputs": list(unique.values()), "inputs": index.entries()})
    validation["report_producer"] = index.identity(__file__)
    seal(validation)
    atomic_write_json(output / "validation_receipt.json", validation)
    index.write_memo(output / "input_verifications.json")
    print("VALIDATED", summary, "A3_vs_A1_pp", deltas, flush=True)
    return validation
