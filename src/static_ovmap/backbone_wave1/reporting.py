"""Measured compact release, four reports, and external verified-push receipt."""

from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from static_ovmap.m2_reviewer_study.binding import InputIndex
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from .binding import read


def net_gain(metrics, baseline):
    if any(row.get(key) is None for row in (metrics, baseline) for key in ("apall", "miou")):
        return "UNDEFINED_METRICS"
    da, dm = 100 * (metrics["apall"] - baseline["apall"]), 100 * (metrics["miou"] - baseline["miou"])
    if da > 0 and dm > 0 and max(da, dm) > .05:
        return "MEASURED_NET_GAIN"
    if da * dm < 0:
        return "TRADEOFF"
    return "NO_MEASURED_NET_GAIN"


def _table(pools):
    lines = ["| Map | Readout | APall | AP50 | AP25 | mIoU | mAcc |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(pools, key=lambda r: (r["map_id"], r["method"])):
        metrics = row["metrics"]
        lines.append("| " + " | ".join([row["map_id"], row["method"], *[
            f'{100 * metrics[key]:.2f}' if metrics[key] is not None else "undefined"
            for key in ("apall", "ap50", "ap25", "miou", "macc")]]) + " |")
    return "\n".join(lines)


def geometric_pools(spec, diagnostics):
    result = []
    for cohort in ("development", "replica"):
        scenes = spec["datasets"][cohort]
        for map_id in sorted({r["map_id"] for r in diagnostics}):
            selected = [r for r in diagnostics if r["map_id"] == map_id and r["scene"] in scenes]
            if {r["scene"] for r in selected} != set(scenes):
                continue
            eligible = sum(r["eligible_gt"] for r in selected)
            gt = [g for r in selected for g in r["per_gt"]]
            predicted_rows = sum(r["surface"]["predicted_rows"] for r in selected)
            target_rows = sum(r["surface"]["valid_target_rows"] for r in selected)
            precision = sum(r["surface"]["precision_5cm"] * r["surface"]["predicted_rows"] for r in selected) / predicted_rows
            completeness = sum(r["surface"]["completeness_5cm"] * r["surface"]["valid_target_rows"] for r in selected) / target_rows
            result.append({"cohort": cohort, "map_id": map_id, "scene_order": scenes,
                "eligible_gt": eligible, "raw_recall": {str(t): sum(r["matches"][str(t)]["matched_gt"] for r in selected) / eligible
                    if eligible else None for t in (.25, .5, .75)},
                "mean_best_gt_iou": float(np.mean([g["best_iou"] for g in gt])) if gt else None,
                "mean_substantial_fragments_per_gt": float(np.mean([g["substantial_fragments"] for g in gt])) if gt else None,
                "surface_precision_5cm": precision, "surface_completeness_5cm": completeness,
                "surface_fscore_5cm": 2 * precision * completeness / (precision + completeness) if precision + completeness else 0.,
                "aggregation": "SUM_MATCHED_AND_ALL_ELIGIBLE_GT;FULL_SURFACE_ROW_WEIGHTED_PRECISION_COMPLETENESS"})
    return result


def metric_comparisons(pools, geometry):
    metrics = {(r["cohort"], r["map_id"], r["method"]): r["metrics"] for r in pools}
    geometric = {(r["cohort"], r["map_id"]): r for r in geometry}
    pairs = [("BB01_SYNC", "BB00_NATIVE"), ("BB05_RATIO_GATE", "BB00_NATIVE"),
             ("BB05_FORWARD", "BB05_RATIO_GATE"), ("BB05_BIDIR", "BB05_FORWARD"),
             ("BB03_SAM2_RAW", "BB00_NATIVE"), ("BB03_SAM2_GEOM", "BB03_SAM2_RAW")]
    pairs += [(r["map_id"], "BB00_NATIVE") for r in pools if r["map_id"] != "BB00_NATIVE"]
    result = []
    for cohort in ("development", "replica"):
        for variant, control in sorted(set(pairs)):
            for method in ("NATIVE_READOUT", "FC_EQ", "D2"):
                a, b = metrics.get((cohort, variant, method)), metrics.get((cohort, control, method))
                if a is None or b is None:
                    continue
                row = {"cohort": cohort, "variant": variant, "control": control, "method": method,
                    "delta_pp": {k: 100 * (a[k] - b[k]) if a[k] is not None and b[k] is not None else None
                        for k in ("apall", "ap50", "ap25", "miou", "macc")}, "measurement_label": net_gain(a, b)}
                ga, gb = geometric.get((cohort, variant)), geometric.get((cohort, control))
                if ga and gb:
                    row["raw_recall_delta_pp"] = {t: 100 * (ga["raw_recall"][t] - gb["raw_recall"][t])
                        if ga["raw_recall"][t] is not None and gb["raw_recall"][t] is not None else None for t in ga["raw_recall"]}
                    row["surface_fscore_delta_pp"] = 100 * (ga["surface_fscore_5cm"] - gb["surface_fscore_5cm"])
                    row["mean_best_gt_iou_delta"] = ga["mean_best_gt_iou"] - gb["mean_best_gt_iou"] if ga["mean_best_gt_iou"] is not None and gb["mean_best_gt_iou"] is not None else None
                result.append(row)
    return result


def _comparison_table(rows):
    lines = ["| Cohort | Variant / Control | Readout | APall delta (pp) | mIoU delta (pp) | Raw R50 delta (pp) | Surface F5 delta (pp) | Label |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        values = [r["delta_pp"]["apall"], r["delta_pp"]["miou"], r.get("raw_recall_delta_pp", {}).get("0.5"), r.get("surface_fscore_delta_pp")]
        lines.append("| " + " | ".join([r["cohort"], r["variant"] + " / " + r["control"], r["method"],
            *[f"{v:+.3f}" if v is not None else "undefined" for v in values], r["measurement_label"]]) + " |")
    return "\n".join(lines)


def _geometry_table(rows):
    lines = ["| Cohort | Map | Eligible GT | Raw R25 | Raw R50 | Raw R75 | Best GT IoU | Fragments/GT | Surface P5 | Surface C5 | Surface F5 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        values = [*[r["raw_recall"][str(t)] for t in (.25, .5, .75)], r["mean_best_gt_iou"]]
        lines.append("| " + " | ".join([r["cohort"], r["map_id"], str(r["eligible_gt"]),
            *[f"{100 * v:.3f}" if v is not None else "undefined" for v in values],
            f'{r["mean_substantial_fragments_per_gt"]:.3f}' if r["mean_substantial_fragments_per_gt"] is not None else "undefined",
            *[f'{100 * r[k]:.3f}' for k in ("surface_precision_5cm", "surface_completeness_5cm", "surface_fscore_5cm")]]) + " |")
    return "\n".join(lines)


def _scene_delta_table(rows):
    lines = ["| Scene | Map | Readout | APall delta (pp) | AP50 delta (pp) | AP25 delta (pp) | mIoU delta (pp) | mAcc delta (pp) |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if r["rank_mode"] == "OFFICIAL_CURRENT_CLASS":
            lines.append("| " + " | ".join([r["scene"], r["map_id"], r["method"],
                *[f'{r["delta_pp"][k]:+.3f}' if r["delta_pp"][k] is not None else "undefined"
                  for k in ("apall", "ap50", "ap25", "miou", "macc")]]) + " |")
    return "\n".join(lines)


def physical_work_ledger(root, spec):
    workers, commands, resources = [], [], []
    patterns = [("NATIVE_QUERY", "readouts/*/*/native_query/*receipt*.json"),
                ("FC", "readouts/*/*/fc/*receipt*.json"), ("SAM2", "frontend/*/SAM2_PAIRED/*receipt*.json"),
                ("SAM2_PREFLIGHT", "prepare/sam2_preflight*/*receipt*.json")]
    for stage, pattern in patterns:
        for path in sorted(root.glob(pattern)):
            row = read(path)
            counters = row.get("counters", row)
            workers.append({"stage": stage, "receipt": str(path), "status": row["status"],
                "physical_image_encodings": counters.get("physical_image_encodings", 0),
                "physical_encoder_calls": counters.get("physical_encoder_calls"),
                "physical_region_poolings": row.get("physical_region_poolings", 0),
                "physical_text_inputs": row.get("physical_text_inputs", 0),
                "model_load_seconds": row.get("model_load_seconds"), "wall_seconds": row.get("elapsed_seconds"),
                "source_commit": row.get("source_commit"), "error": row.get("error"),
                "validation_only": stage == "SAM2_PREFLIGHT"})
    for directory in (root, Path(spec["native_build_root"])):
        for path in sorted(directory.rglob("*.commands.json")):
            for attempt, row in enumerate(read(path), 1):
                commands.append(dict(row, command_receipt=str(path), attempt=attempt))
    for path in sorted((root / "execution").glob("*.resources.json")):
        if path.stat().st_size:
            resources.append(dict(json.loads(path.read_text().splitlines()[-1]), receipt=str(path)))
    return {"worker_attempts": workers, "command_attempts": commands, "phase_resources": resources,
        "totals": {"physical_image_encodings": sum(r["physical_image_encodings"] for r in workers),
            "physical_image_encoder_calls_known": sum(r["physical_encoder_calls"] or 0 for r in workers),
            "physical_FC_region_poolings": sum(r["physical_region_poolings"] for r in workers),
            "physical_FC_text_inputs": sum(r["physical_text_inputs"] for r in workers),
            "model_loads_with_recorded_duration": sum(r["model_load_seconds"] is not None for r in workers),
            "worker_stage_wall_seconds_including_validation": sum(r["wall_seconds"] or 0 for r in workers),
            "recorded_phase_wall_seconds": sum(r["wall_seconds"] for r in resources),
            "recorded_phase_user_cpu_seconds": sum(r["user_cpu_seconds"] for r in resources),
            "recorded_phase_system_cpu_seconds": sum(r["system_cpu_seconds"] for r in resources)},
        "timing_interpretation": "Concurrent stage durations and phase durations overlap; neither sum is end-to-end attempt wall time. GPU worker wall time includes model loading, CPU preprocessing and IO.",
        "missing_timing_components": ["device_only_GPU_event_seconds_not_instrumented", "pre_screen_phase_CPU_seconds_not_instrumented"]}


def per_class_and_matches(rows, pools, spec):
    from static_ovmap.m2_reviewer_study.evaluation import plain
    from static_ovmap.released_loader import load_released_module
    summaries, matches = [], []
    cached = {}
    for row in rows:
        path = Path(row["evaluation_receipt"])
        if str(path) not in cached:
            receipt = read(path)
            with gzip.open(path.parent / "trace.json.gz", "rt") as stream:
                trace = json.load(stream)
            matrix = np.asarray(receipt["confusion"], np.int64)
            classes = []
            for index, label in enumerate(receipt["context"]["class_names"]):
                states = [s for s in trace["states"] if s["distance_index"] == 0 and s["class_index"] == index]
                selected = [s["ap"] for s in states if not np.isclose(s["overlap_threshold"], .25)]
                ap = float(np.mean(selected)) if selected and all(v is not None for v in selected) else None
                tp, gt, pred = int(matrix[index + 1, index + 1]), int(matrix[index + 1].sum()), int(matrix[:, index + 1].sum())
                classes.append({"class": label, "class_id": receipt["context"]["valid_ids"][index], "apall": ap,
                    "ap50": next((s["ap"] for s in states if np.isclose(s["overlap_threshold"], .5)), None),
                    "ap25": next((s["ap"] for s in states if np.isclose(s["overlap_threshold"], .25)), None),
                    "iou": tp / (gt + pred - tp) if gt else None, "accuracy": tp / gt if gt else None,
                    "GT_surface_rows": gt, "predicted_surface_rows": pred,
                    "undefined_reason": "no_eligible_class_GT_for_AP_or_no_valid_semantic_GT_rows" if ap is None or not gt else None})
            matching = {}
            for threshold in (.25, .5, .75):
                states = [s for s in trace["states"] if s["distance_index"] == 0 and np.isclose(s["overlap_threshold"], threshold)]
                matching[str(threshold)] = {"matched_gt": sorted({e["gt_id"] for e in trace["events"]
                    if e["distance_index"] == 0 and np.isclose(e["overlap_threshold"], threshold) and e["event"] in {"first_match", "duplicate"}}),
                    "false_positive_score_entries": sum(sum(v == 0 for v in s["y_true"]) for s in states),
                    "hard_false_negatives": sum(s["hard_false_negatives"] for s in states)}
            cached[str(path)] = classes, matching
        classes, matching = cached[str(path)]
        context = {k: row[k] for k in ("scene", "map_id", "method", "rank_mode")}
        summaries.append(dict(context, classes=classes))
        matches.append(dict(context, thresholds=matching))
    differences = []
    lookup = {(r["scene"], r["map_id"], r["method"], r["rank_mode"]): r for r in matches}
    for r in matches:
        baseline = lookup.get((r["scene"], "BB00_NATIVE", r["method"], r["rank_mode"]))
        if r["map_id"] == "BB00_NATIVE" or baseline is None:
            continue
        differences.append({**{k: r[k] for k in ("scene", "map_id", "method", "rank_mode")}, "thresholds": {
            t: {"added_matched_GT": sorted(set(v["matched_gt"]) - set(baseline["thresholds"][t]["matched_gt"])),
                "removed_matched_GT": sorted(set(baseline["thresholds"][t]["matched_gt"]) - set(v["matched_gt"])),
                "false_positive_score_entries_delta": v["false_positive_score_entries"] - baseline["thresholds"][t]["false_positive_score_entries"],
                "hard_false_negatives_delta": v["hard_false_negatives"] - baseline["thresholds"][t]["hard_false_negatives"]}
            for t, v in r["thresholds"].items()}, "FP_unit": "released_score_entries_including_duplicate_rule;not_unique_owner_count"})
    pooled = []
    namespace = load_released_module(Path(spec["upstream_worktree"]) / "scripts/eval_utils.py")
    for row in pools:
        namespace["init"]("Replica" if row["dataset"] == "Replica" else "Scannet200")
        selected = [r for r in rows if r["scene"] in row["scene_order"] and r["map_id"] == row["map_id"] and r["method"] == row["method"] and r["rank_mode"] == row["rank_mode"]]
        combined, matrices = {}, []
        for scene in row["scene_order"]:
            receipt = read(next(r["evaluation_receipt"] for r in selected if r["scene"] == scene))
            matrices.append(receipt["confusion"])
            with gzip.open(Path(receipt["manifest"]).parent / "matches.json.gz", "rt") as stream:
                combined.update(json.load(stream))
        averages = plain(namespace["compute_averages"](namespace["evaluate_matches"](combined)))
        if any(averages[k] != row["metrics"][m] for k, m in (("all_ap", "apall"), ("all_ap_50%", "ap50"), ("all_ap_25%", "ap25"))):
            raise ValueError("per-class pool reconstruction differs from the official locked pool")
        pooled.append({"cohort": row["cohort"], "map_id": row["map_id"], "method": row["method"],
            "rank_mode": row["rank_mode"], "class_AP": averages["classes"], "confusion": np.sum(matrices, axis=0).tolist(),
            "scene_order": row["scene_order"], "official_AP_exact": True})
    return {"scene_classes": summaries, "pooled_classes": pooled}, differences


def frontend_interventions(binding, receipts):
    from PIL import Image
    from .diagnostics import canonical_partition

    result = []
    for receipt in receipts:
        path = Path(binding["scenes"][receipt["scene"]]["parent_capture"])
        capture = read(path)
        originals = {f["frame_id"]: f for f in capture["frames"]}
        frames, decisions = [], {}
        for frame in receipt["frames"]:
            original = np.asarray(Image.open(path.parent / originals[frame["frame_id"]]["panoptic_path"]))
            rasters = {name: np.asarray(Image.open(frame["outputs"][name]["path"])) for name in ("raw", "geom")}
            canonical = {name: canonical_partition(raster.reshape(-1)) for name, raster in dict(rasters, original=original).items()}
            diagnostic = frame["diagnostic"]
            if diagnostic.get("discovery_groups", []) != diagnostic.get("geom_discovery_groups", []):
                raise ValueError("paired frontend changed its RAW-defined discovery candidates")
            for track in diagnostic.get("tracks", []):
                decisions[track["decision"]] = decisions.get(track["decision"], 0) + 1
            frames.append({"frame_id": frame["frame_id"], "chunk": frame["chunk"],
                "raw_vs_cropformer_canonical_changed_pixels": int(np.count_nonzero(canonical["raw"] != canonical["original"])),
                "geom_vs_cropformer_canonical_changed_pixels": int(np.count_nonzero(canonical["geom"] != canonical["original"])),
                "geom_vs_raw_canonical_changed_pixels": int(np.count_nonzero(canonical["geom"] != canonical["raw"])),
                "RAW_foreground_changed_pixels": int(np.count_nonzero((rasters["raw"] > 0) != (original > 0))),
                "GEOM_foreground_changed_pixels": int(np.count_nonzero((rasters["geom"] > 0) != (original > 0))),
                "RAW_defined_discovery_candidates": len(diagnostic.get("discovery_groups", [])),
                "raw_union_pixels": diagnostic.get("raw_union_pixels"), "filter": diagnostic})
        result.append({"scene": receipt["scene"], "identity": receipt["identity"], "frames": frames,
            "filter_decision_counts": decisions, "shared_RAW_discovery_verified": True,
            "changed_pixel_convention": "positive_groups_canonicalized_by_first_row_major_support;foreground_symmetric_difference_separate",
            "counts_do_not_infer_cross_frame_or_cross_map_owner_correspondence": True})
    return result


def render(binding, spec):
    root, repo = Path(binding["output_root"]), Path(binding["repository_root"])
    release = repo / spec["publication"]["repo_artifacts"]
    docs = repo / "docs/paper/static_ovmap"
    release.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    rows, pools, secondary_pools, diagnostics, interventions, maps, costs = [], [], [], [], [], [], []
    for path in (root / "readouts").glob("*/*/evaluation_rows.json"):
        rows.extend(read(path)["rows"])
    for path in (root / "pools").glob("*/*/*/OFFICIAL_CURRENT_CLASS.json"):
        value = read(path)
        value.update(cohort=path.relative_to(root / "pools").parts[0], map_id=path.relative_to(root / "pools").parts[1])
        pools.append(value)
    for path in (root / "pools").glob("*/*/*/FROZEN_N0.json"):
        value = read(path)
        value.update(cohort=path.relative_to(root / "pools").parts[0], map_id=path.relative_to(root / "pools").parts[1])
        secondary_pools.append(value)
    for path in (root / "readouts").glob("*/*/raw_geometry_diagnostics.json"):
        diagnostics.append(read(path))
    for path in (root / "readouts").glob("*/*/intervention.json"):
        interventions.append(read(path))
    for path in (root / "maps").glob("*/*/map_receipt.json"):
        maps.append(read(path))
    for path in (root / "readouts").glob("*/*/receipt.json"):
        value = read(path)
        nq, fc = read(path.parent / "native_query/native_query_receipt.json"), read(path.parent / "fc/receipt.json")
        costs.append({"scene": value["scene"], "map_id": value["map_id"],
            "required_image_encodings": value["required_image_encodings"],
            "attributable_standalone_seconds": value["attributable_map_plus_readout_seconds"],
            "timing_missing_components": value["standalone_timing_missing_components"],
            "physical_native_crop_encodings": nq["physical_image_encodings"],
            "physical_FC_image_encodings": fc["physical_image_encodings"],
            "physical_FC_region_poolings": fc["physical_region_poolings"],
            "native_query_wall_seconds": nq["elapsed_seconds"], "FC_wall_seconds": fc["elapsed_seconds"],
            "native_query_source_commit": nq.get("source_commit"), "FC_source_commit": fc.get("source_commit"),
            "Q_logical_attempts": nq["query_logical_ledger"]["attempts"],
            "source_availability": value["source_availability"], "cap_exclusions": value["fc_target_cap_exclusions"]})
    bridge = read(root / "bridge_parity.json") if (root / "bridge_parity.json").is_file() else {"status": "NOT_MEASURED"}
    selection = read(root / "selection.json") if (root / "selection.json").is_file() else None
    expected = len(spec["map_variants"]) * 4
    if selection:
        expected += 8 * len(selection["replica_recipes"]) + (4 if any(r["id"] == "BBX_COMPOSE" for r in selection["replica_recipes"]) else 0)
    success = [m for m in maps if m["status"] == "COMPLETE"]
    primary = [r for r in rows if r["rank_mode"] == "OFFICIAL_CURRENT_CLASS"]
    expected_pairs = {(scene, recipe["id"]) for scene in spec["datasets"]["development"] for recipe in spec["map_variants"]}
    if selection:
        expected_pairs |= {(scene, recipe["id"]) for scene in spec["datasets"]["replica"] for recipe in selection["replica_recipes"]}
        if any(r["id"] == "BBX_COMPOSE" for r in selection["replica_recipes"]):
            expected_pairs |= {(scene, "BBX_COMPOSE") for scene in spec["datasets"]["development"]}
    measured_pairs = {(m["scene"], m["map_id"]) for m in success}
    expected_rows = {(scene, map_id, method) for scene, map_id in expected_pairs for method in spec["semantics"]["readouts"]}
    measured_rows = {(r["scene"], r["map_id"], r["method"]) for r in primary}
    complete = bool(selection) and measured_pairs == expected_pairs and measured_rows == expected_rows and bridge["status"] == "VERIFIED"
    baseline = next((r["metrics"] for r in pools if r["cohort"] == "replica" and r["map_id"] == "BB00_NATIVE" and r["method"] == "D2"), None)
    nominee = next((r["metrics"] for r in pools if selection and r["cohort"] == "replica" and r["map_id"] == selection["nominee"] and r["method"] == "D2"), None)
    conclusion = net_gain(nominee, baseline) if nominee and baseline else "NOT_MEASURED"
    status = ("COMPLETE_MEASURED_NET_GAIN" if conclusion == "MEASURED_NET_GAIN" else "COMPLETE_NO_NET_GAIN") if complete else "INCOMPLETE"
    per_class = []
    for row in rows:
        receipt = read(row["evaluation_receipt"])
        per_class.append({"scene": row["scene"], "map_id": row["map_id"], "method": row["method"],
            "rank_mode": row["rank_mode"], "confusion": receipt["confusion"],
            "context": receipt["context"], "view": receipt["view"]})
    geometry_pools = geometric_pools(spec, diagnostics)
    comparisons = metric_comparisons(pools, geometry_pools)
    class_summaries, official_match_differences = per_class_and_matches(rows, pools + secondary_pools, spec)
    physical = physical_work_ledger(root, spec)
    scene_lookup = {(r["scene"], r["map_id"], r["method"], r["rank_mode"]): r for r in rows}
    scene_deltas = []
    for r in rows:
        base = scene_lookup.get((r["scene"], "BB00_NATIVE", r["method"], r["rank_mode"]))
        if r["map_id"] != "BB00_NATIVE" and base:
            scene_deltas.append({**{k: r[k] for k in ("scene", "map_id", "method", "rank_mode")},
                "delta_pp": {k: 100 * (r["metrics"][k] - base["metrics"][k])
                    if r["metrics"][k] is not None and base["metrics"][k] is not None else None
                    for k in ("apall", "ap50", "ap25", "miou", "macc")}})
    external = {"output_root": str(root), "native_build_root": spec["native_build_root"],
        "sam2_source": binding["sam2"]["code"], "assets_root": binding["assets_root"],
        "map_outputs": [{"scene": m["scene"], "map_id": m["map_id"], "capture": m.get("capture_manifest"),
                         "status": m["status"], "input_identity": m.get("input_identity")} for m in maps],
        "restore": "Restore the external map/capture/source/cache arrays at the recorded absolute paths and verify their receipts; weights/data are not in Git."}
    source_outputs = InputIndex()
    for path in (root / "readouts").glob("*/*/receipt.json"):
        receipt = read(path)
        for leaf in ("receipt.json", "native_query/native_query_receipt.json", "native_query/native_features.pkl",
                     "native_query/query_scores.npz", "native_query/query_decisions.json", "fc/receipt.json",
                     "fc/F.json", "fc/weight_audit.json", "projection/manifest.json", "projection/projection.npz",
                     "sources/N.json", "sources/Q.json", "sources/F.json", "decisions/FC_EQ.json", "decisions/D2.json"):
            source_outputs.identity(path.parent / leaf)
        for manifest in receipt["predictions"].values():
            source_outputs.identity(manifest)
            source_outputs.identity(Path(manifest).parent / read(manifest)["arrays"]["path"])
    external["source_output_identities"] = source_outputs.entries()
    external["recreation_commands"] = {"full_study": [spec["runtime_default"], spec["runner"], "--phase", "all",
        "--spec", str(repo / spec["repository_spec_root"] / "PROTOCOL_SPEC.json"), "--output-root", str(root),
        "--gpu", "2", "--mapping-workers", "2", "--mapping-threads", "8", "--evaluation-workers", "3", "--resume"],
        "map_jobs": [m["command"]["argv"] for m in success],
        "worker_jobs": [c["argv"] for c in physical["command_attempts"] if "--job" in c["argv"]],
        "environment_and_job_inputs": "Use each mapping running.json and each preserved worker .job.json; verified data/model assets must be restored before recreation."}
    events = []
    for mapping in success:
        capture_path = Path(mapping["capture_manifest"])
        capture = read(capture_path)
        for frame in capture["frames"]:
            state = read(capture_path.parent / frame["native_state"]["path"])["native_state"]
            association = state.get("association", {})
            frame_diagnostics = capture_path.parents[2] / "diagnostics" / f'{frame["frame_id"]:06d}'
            events.append({"scene": mapping["scene"], "map_id": mapping["map_id"], "frame_id": frame["frame_id"],
                "namespaces": {"current_2D_groups": sorted(set(row["input_instance_label"] for row in state["segments"])),
                    "superpoint_labels": [row["registered_label"] for row in state["segments"]],
                    "count_object_owners": sorted(set(row["instance_label"] for row in state["label_instances"]))},
                "fragments": len(state["segments"]), "aliases": state["aliases"],
                "assigned_superpoints": state["segments"], "realized_count_owners": state["label_instances"],
                "association": {k: v for k, v in association.items() if k not in {"prior_owners", "alias_tokens"}},
                "realized_owner_discrepancies": [row for row in state["segments"] if row.get("owner_discrepancy")],
                "native_selected_requests": len(frame["native_selected_request_ids"]),
                "admissible_requests": len(frame["requests"]), "source_map_state_id": frame["map_state_id"]})
            for name in ("association_plan", "order_diagnostic"):
                if (frame_diagnostics / (name + ".json")).is_file():
                    events[-1][name] = read(frame_diagnostics / (name + ".json"))
    semantic_differences = []
    from static_ovmap.module_validation.scannet_study import load_prediction, project_values
    for path in (root / "readouts").glob("*/*/receipt.json"):
        receipt = read(path)
        anchor = load_prediction(receipt["predictions"]["NATIVE_READOUT"])
        with np.load(path.parent / "projection/projection.npz", allow_pickle=False) as arrays:
            nearest, matched = arrays["nearest"], arrays["matched"]
        baseline_root = root / "readouts" / receipt["scene"] / "BB00_NATIVE"
        for method, manifest in receipt["predictions"].items():
            prediction = load_prediction(manifest)
            changed = prediction.semantic_labels != anchor.semantic_labels
            row = {"scene": receipt["scene"], "map_id": receipt["map_id"], "readout": method,
                "same_map_semantic_changed_source_rows": int(changed.sum()),
                "same_map_geometry_and_owners_exact": prediction.geometry == anchor.geometry and np.array_equal(prediction.owner_ids, anchor.owner_ids)}
            if (baseline_root / "receipt.json").is_file():
                base_receipt = read(baseline_root / "receipt.json")
                base = load_prediction(base_receipt["predictions"][method])
                with np.load(baseline_root / "projection/projection.npz", allow_pickle=False) as arrays:
                    base_labels = project_values(base.semantic_labels, arrays["nearest"], arrays["matched"])
                current_labels = project_values(prediction.semantic_labels, nearest, matched)
                row["cross_map_same_readout_changed_target_labels"] = int(np.count_nonzero(current_labels != base_labels))
            if method == "D2":
                fc_decisions, d2_decisions = read(path.parent / "decisions/FC_EQ.json"), read(path.parent / "decisions/D2.json")
                differences = [np.max(np.abs(np.asarray(fc_decisions[k]["probabilities"]) - np.asarray(value["probabilities"])))
                               for k, value in d2_decisions.items() if value["probabilities"] is not None]
                row["D2_vs_FC_EQ_max_probability_difference"] = float(max(differences, default=0.))
                row["D2_vs_FC_EQ_changed_owner_labels"] = sum(value["label"] != fc_decisions[k]["label"] for k, value in d2_decisions.items())
            semantic_differences.append(row)
    physical_frontends = [read(path) for path in (root / "frontend").glob("*/SAM2_PAIRED/receipt.json")]
    frontend_evidence = frontend_interventions(binding, physical_frontends)
    executions = [read(path) for path in (root / "execution").glob("*.json")
                  if not path.name.endswith(".resources.json")]
    failures = [row for row in executions if row.get("failures")]
    implementation_commits = {row["implementation_commit"] for row in executions if "implementation_commit" in row}
    implementation_commits.update(r["source_commit"] for r in physical["worker_attempts"] if r["source_commit"])
    provenance = {"base_commit": spec["base_commit"], "upstream_commit": spec["upstream_commit"],
        "implementation_commits": sorted(implementation_commits),
        "consumed_code_and_model_identities": binding["inputs"], "fresh_source_output_identities": source_outputs.entries(),
        "map_source_and_output_identities": [
            {"scene": m["scene"], "map_id": m["map_id"], "inputs": m["inputs"], "outputs": m["outputs"],
             "controller_reuse_alias": read(Path(m["capture_manifest"]).parents[2] / "controller_reuse_alias.json")
                if (Path(m["capture_manifest"]).parents[2] / "controller_reuse_alias.json").is_file() else None} for m in success]}
    outputs = {"resolved_inputs.json": binding, "experiment_matrix.json": read(root / "matrix.json"),
        "bridge_parity.json": bridge, "scene_rows.json": primary, "secondary_rank_rows.json": [r for r in rows if r not in primary],
        "dataset_pools.json": pools, "secondary_rank_pools.json": secondary_pools,
        "scene_deltas.json": scene_deltas, "control_comparisons.json": comparisons,
        "intervention_summary.json": interventions, "raw_geometry_pools.json": geometry_pools,
        "raw_geometry_diagnostics.json": diagnostics, "costs.json": costs, "physical_work.json": physical,
        "semantic_differences.json": semantic_differences, "code_model_output_provenance.json": provenance,
        "failures.json": failures, "frontend_physical_costs.json": [{"scene": r["scene"], "status": r["status"],
            "counters": r["counters"], "elapsed_seconds": r["elapsed_seconds"],
            "peak_gpu_allocated_bytes": r["peak_gpu_allocated_bytes"], "identity": r["identity"]} for r in physical_frontends],
        "per_class_confusions.json": per_class, "per_class_summaries.json": class_summaries,
        "official_match_differences.json": official_match_differences, "external_artifacts.json": external,
        "completion.json": {"status": status, "implementation": "IMPLEMENTED_SCOPED_RUNTIME_VERIFIED",
            "experimental_coverage": {"successful_map_scenes": len(success), "expected": expected,
                "primary_rows": len(primary), "expected_primary_rows": 3 * expected},
            "scientific": conclusion, "publication": "EXTERNAL_RECEIPT_REQUIRED",
            "all_scenes_exposed": True, "deployment": "N0_UNCHANGED"}}
    for leaf in ("candidate_freeze.json", "selection.json", "bridge_diagnosis.json", "transfer_freeze_commit.json"):
        if (root / leaf).is_file():
            outputs[leaf] = read(root / leaf)
    for name, data in outputs.items():
        atomic_write_json(release / name, data)
    with gzip.open(release / "native_event_ledger.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(events, stream, sort_keys=True, allow_nan=False)
    with gzip.open(release / "frontend_intervention_evidence.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(frontend_evidence, stream, sort_keys=True, allow_nan=False)
    for leaf in ("native_trace/summary.json", "sam2_preflight/receipt.json"):
        path = root / "prepare" / leaf
        if path.is_file():
            atomic_write_json(release / "validation" / (Path(leaf).parent.name + ".json"), read(path))
    build = Path(spec["native_build_root"]) / "native_build_receipt.json"
    if build.is_file():
        atomic_write_json(release / "validation/native_build_receipt.json", read(build))
    test_path = root / "review/scoped_tests.json"
    if test_path.is_file():
        atomic_write_json(release / "validation/scoped_tests.json", read(test_path))
    reports = {
        "BACKBONE_WAVE1_RESULTS.md": f"# Backbone wave-1 measured results\n\nStatus: `{status}`. Successful map-scene configurations: {len(success)}/{expected}; primary official rows: {len(primary)}/{3 * expected}. All data are exposed. Deployment: N0_UNCHANGED.\n\n## Development Official Pools (%)\n\n" + _table([r for r in pools if r["cohort"] == "development"]) +
            "\n\n## Replica Official Pools (%)\n\n" + _table([r for r in pools if r["cohort"] == "replica"]) +
            "\n\n## Raw Geometry (%)\n\n" + _geometry_table(geometry_pools) +
            "\n\n## Mechanism and Simple-Control Comparisons\n\n" + _comparison_table(comparisons) +
            "\n\nThese labels describe APall/mIoU measurements only. The raw columns separately expose instance and surface changes; a positive semantic label alone does not establish stronger geometry. BIDIR is compared with FORWARD, FORWARD with the ratio switch, and GEOM with RAW as well as BB00. Missing complete cohorts produce no pooled comparison.\n\n" +
            "## Per-Scene Changes Against the Fresh BB00\n\n" + _scene_delta_table(scene_deltas) +
            f"\n\nNominee measurement label: `{conclusion}`. Bridge: `{bridge['status']}`. APall uses the actual released overlap vector in the evaluation contexts, .50 through .90; AP25 is separate. Pools use the released evaluator over complete ordered cohorts. Semantic metrics sum scene confusion matrices.\n\n" +
            "Raw numeric geometry and official native triangle-first-color paint are separate. Class-agnostic recall uses strict IoU > .25/.5/.75, maximum-cardinality one-to-one matches and every eligible GT in the denominator. Missed GT best-IoU is zero. Surface precision/completeness use every exported vertex and every valid target vertex at strict 5cm; duplicated native vertices are retained consistently. Detailed raw geometry, GT-indexed gains/losses and predicted overlap associations are in the compact release.\n\n" +
            "## Actual Physical Work and Failures\n\n```json\n" + json.dumps(physical["totals"], indent=2) + "\n```\n\n" +
            f"Recorded failed command attempts: {sum(c['exit_code'] != 0 for c in physical['command_attempts'])}. Recorded failed worker receipts: {sum(w['status'] == 'FAILED' for w in physical['worker_attempts'])}. Phase failures: {len(failures)}. Details and retained logs are in `physical_work.json` and `failures.json`.\n\n" +
            "Costs distinguish actual forwards from content-deduplicated standalone obligations. Common SAM predictions are required by both RAW/GEOM. N/Q/FC costs are required by both fused readouts. Timing is measured stage attribution, not a cold end-to-end rerun; missing timing components are disclosed per job. GPU-worker wall time includes CPU preprocessing, loading and IO; device-only GPU time was not instrumented. Recorded phase CPU totals cover the phases listed in `physical_work.json`, including children, and exclude earlier uninstrumented preparation/bridge work. Concurrent stages overlap and their duration sum is not end-to-end elapsed time. No significance or independent-generalization claim is made.\n",
        "BACKBONE_WAVE1_HANDOFF.md": "# Backbone wave-1 handoff\n\n" + f"Repository: `{repo}`\nExternal attempt: `{root}`\nIsolated upstream: `{spec['upstream_worktree']}`\nIsolated native build: `{spec['native_build_root']}`\n\n" +
            "Apply the original module_validation_v1 patch, then third_party_patches/ovimap/backbone_wave1_v1/backbone_wave1_v1.patch to the pinned upstream. The old native extension remains intact. SAM uses its pinned source checkout and historical environment; optional connected-components CUDA extension absence uses the official loader fallback and is recorded.\n\n" +
            f"```bash\n{spec['runtime_default']} scripts/evaluation/run_ovimap_backbone_wave1.py --phase all --spec docs/paper/static_ovmap/backbone_wave1_v1/PROTOCOL_SPEC.json --output-root {root} --gpu 2 --mapping-workers 2 --mapping-threads 8 --evaluation-workers 3 --resume\n```\n\n" +
            "Report and publish never launch mapping/models. The coordinator, map outputs and GPU use real locks. Interrupted maps restart from frame zero with previous attempts retained. Restore external arrays/data/weights from external_artifacts.json and validate consumed receipts. Implementation and transfer-freeze commits are in execution/freeze receipts; the final publication SHA is external in publication/final.json.\n\n" +
            "Actual consumed implementation commits: " + ", ".join(f"`{c}`" for c in provenance["implementation_commits"]) + ".\n\n" +
            "Every measured map's raw inputs/outputs are identified by path, bytes and SHA256 in `code_model_output_provenance.json`. The unchanged measured bridge retains its original receipt and explicitly records the validation-only C6 controller alias. Per-class summaries reconstruct the original released AP from locked matches and verify exact pooled AP agreement. Secondary ranks use each new map's own native rank table.\n",
        "BACKBONE_WAVE1_SELECTION.md": "# Backbone wave-1 selection\n\n" +
            (f"Frozen nominee: `{selection['nominee']}`. Transfer map IDs: " + ", ".join(r["id"] for r in selection["replica_recipes"]) + ".\n\n" if selection else "Selection is incomplete; no Replica nominee is asserted.\n\n") +
            "Selection uses D2 four-scene official development pooling only. APall band .05pp, mIoU band .1pp, AP50 band .1pp, then required standalone image encodings, median attributable time, changed block count and method ID. The structural champion is selected from BB01_SYNC/BB05_FORWARD/BB05_BIDIR: BB05_RATIO_GATE remains a mandatory measured simple control and is excluded by the normative noncontrol-champion rule. Strict metric ranking, banded ranking and every tie step are frozen in candidate_freeze.json/selection.json before composition/Replica. Individual positive gain is not required for composition; both families require complete inputs and actual raw partition intervention. Replica results do not refit weights, temperatures or the Q model.\n",
        "BACKBONE_WAVE1_CLAIMS.md": "# Backbone wave-1 claims\n\n" + f"Completion status: `{status}`. Measured nominee label: `{conclusion}`.\n\n" +
            "Supported implementation claims: the new native extension exposes a read-only prior probe and enforces object plans through candidate filtering, mode4 counts and compatible aliases; real short-trace receipts report realized ownership. Simultaneous fusion uses unchanged original depth support. Bounded SAM uses current-only forward outputs, five-frame resets and shared RAW-defined discoveries, with geometry checked against previous RAW support.\n\n" +
            "Scientific mechanism claims require the measured raw geometry ledgers and relevant simple-control comparisons. A class-conditioned rank or semantic coverage change alone does not establish stronger geometry. A stronger-backbone claim is not automatically authorized by AP gain. Mixed APall/mIoU signs are a tradeoff. Negative complete studies remain complete without a deployment change.\n\n" +
            "## Measured Mechanism Comparisons\n\n" + _comparison_table([r for r in comparisons if r["method"] == "D2"]) + "\n\n" +
            "`semantic_differences.json` measures actual label/probability differences; `official_match_differences.json` records added/removed GT matches and released FP score-entry deltas. `per_class_summaries.json` and `secondary_rank_pools.json` preserve class and rank context. A map with zero changed target partition rows is recorded separately from any semantic intervention. No fixed gain sentence substitutes for these measured outputs.\n\n" +
            "Untested/out of scope: independent confirmation; B02/B04/B06-B12; full OVRCOAT wrapper; SAM3; new temperature/Q/quality-head fitting; deployment improvement. Four development and eight Replica scenes are historically exposed; extra visual inference is explicitly accounted. Numeric owner IDs are not correspondence across maps.\n"}
    for name, value in reports.items():
        (docs / name).write_text(value, encoding="utf-8")
    index = InputIndex()
    included = [path for path in release.rglob("*") if path.is_file() and path.name != "release_manifest.json"]
    included += [docs / name for name in reports]
    for path in included:
        if path.stat().st_size >= 95 * 1024 ** 2:
            raise ValueError("compact release exceeds the per-file size contract")
        index.identity(path)
    atomic_write_json(release / "release_manifest.json", {"entries": index.entries(),
        "total_bytes": sum(path.stat().st_size for path in included), "raw_arrays_in_Git": False})
    atomic_write_json(root / "report/render_receipt.json", {"status": status, "release": str(release),
        "reports": list(reports), "output_identities": index.entries()})
    return outputs["completion.json"]


def publish(binding, spec):
    repo, root = Path(binding["repository_root"]), Path(binding["output_root"])
    completion = read(repo / spec["publication"]["repo_artifacts"] / "completion.json")
    if not completion["status"].startswith("COMPLETE"):
        raise ValueError("unfinished study cannot be published as completed")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    if branch != spec["publication"]["branch"]:
        raise ValueError("publication branch differs from the fixed study branch")
    scope = [spec["implementation_root"], spec["runner"], spec["repository_spec_root"],
             "third_party_patches/ovimap/backbone_wave1_v1", spec["publication"]["repo_artifacts"],
             *["docs/paper/static_ovmap/" + name for name in spec["publication"]["reports"]],
             "tests/evaluation/test_backbone_wave1_kernels.py", "tests/evaluation/test_backbone_wave1_frontend.py",
             "tests/evaluation/test_backbone_wave1_readouts.py"]
    subprocess.run(["git", "add", "--", *scope], cwd=repo, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode:
        subprocess.run(["git", "commit", "-m", "research: publish measured backbone-wave1 evidence"], cwd=repo, check=True)
    local = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    receipt = {"branch": branch, "local_sha": local, "remote_sha": None, "status": "PUSH_PENDING",
               "release_path": spec["publication"]["repo_artifacts"]}
    path = root / spec["publication"]["external_receipt"]
    try:
        subprocess.run(["git", "push", "origin", "HEAD:refs/heads/" + branch], cwd=repo, check=True)
        remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=repo, text=True).split()[0]
        if remote != local:
            raise ValueError("full local/remote publication SHA mismatch")
        receipt.update(status="PUSH_VERIFIED", remote_sha=remote)
    except BaseException as exc:
        receipt.update(status="BLOCKED_PUSH", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(path, receipt)
    return receipt
