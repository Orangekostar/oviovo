"""Four evidence tables from actual completed predictions and released traces."""

import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from src.static_ovmap.attribution_objects import released_object_outcomes
from src.static_ovmap.module_validation.contracts import (
    atomic_write_json,
    canonical_digest,
)
from src.static_ovmap.module_validation.scannet_study import load_prediction

from .calibration_jobs import sources_for_scene
from .evaluation import object_correspondence
from .execution import verified_receipt
from .io import ROOT, read_json, write_once
from .selection import COMPOSITIONS, CONTROLS, TOLERANCE

METRICS = ("uap", "ap50", "ap25", "miou", "macc")


def metric_summary(rows, role, method, scenes):
    selected = [
        row
        for row in rows
        if row["role"] == role
        and row["method_id"] == method
        and row["scene_id"] in scenes
    ]
    means, counts = {}, {}
    for key in METRICS:
        values = [
            row["metrics"].get(key) for row in selected if row["status"] == "COMPLETE"
        ]
        values = [
            value for value in values if value is not None and math.isfinite(value)
        ]
        means[key] = float(np.mean(values)) if values else None
        counts[key] = {"defined": len(values), "total": len(scenes)}
    status = "COMPLETE"
    if len(selected) != len(scenes):
        status = "MISSING_ROWS"
    elif any(counts[key]["defined"] != len(scenes) for key in ("uap", "miou")):
        status = "INCONCLUSIVE_UNDEFINED_METRIC"
    return {
        "status": status,
        "role": role,
        "method_id": method,
        "means": means,
        "denominators": counts,
    }


def compare_methods(rows, role, candidate, reference, scenes):
    left, right = (
        metric_summary(rows, role, method, scenes) for method in (candidate, reference)
    )
    if left["status"] != "COMPLETE" or right["status"] != "COMPLETE":
        return {
            "status": "INCONCLUSIVE_INCOMPLETE_ROWS",
            "candidate": candidate,
            "reference": reference,
            "role": role,
        }
    mean_delta = {
        key: left["means"][key] - right["means"][key] for key in ("uap", "miou")
    }
    by_key = {
        (row["scene_id"], row["method_id"]): row for row in rows if row["role"] == role
    }
    deltas = {
        scene: {
            key: by_key[scene, candidate]["metrics"][key]
            - by_key[scene, reference]["metrics"][key]
            for key in ("uap", "miou")
        }
        for scene in scenes
    }
    nonnegative = all(
        value >= -TOLERANCE for row in deltas.values() for value in row.values()
    )
    strict = all(value > TOLERANCE for row in deltas.values() for value in row.values())
    mean_gain = all(value >= -TOLERANCE for value in mean_delta.values()) and any(
        value > TOLERANCE for value in mean_delta.values()
    )
    status = (
        (
            "MEAN_GAIN_NO_OBSERVED_SCENE_LOSS"
            if nonnegative
            else "MEAN_GAIN_WITH_SCENE_TRADEOFF"
        )
        if mean_gain
        else "NO_MEAN_GAIN"
    )
    return {
        "status": status,
        "role": role,
        "candidate": candidate,
        "reference": reference,
        "mean_deltas": mean_delta,
        "scene_deltas": deltas,
        "every_scene_nonnegative": nonnegative,
        "every_scene_strictly_positive": strict,
        "worst_scene_deltas": {
            key: min(row[key] for row in deltas.values()) for key in mean_delta
        },
    }


def _gz(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def _scene_config(config, scene):
    path = (
        Path(config["attempt_root"]) / "confirmation" / scene / "resolved_config.json"
    )
    return read_json(path) if path.is_file() else config


def complementarity(config, rows):
    summary, all_objects, routing = {}, {}, {}
    for scene in sorted({row["scene_id"] for row in rows}):
        local = _scene_config(config, scene)
        try:
            sources = sources_for_scene(local, scene)
        except FileNotFoundError:
            summary[scene] = {"status": "SOURCE_EVIDENCE_INCOMPLETE"}
            continue
        native = load_prediction(local["scenes"][scene]["native_prediction"])
        correspondences = object_correspondence(local, scene, native)
        patterns, categories, agreement, objects = Counter(), Counter(), Counter(), []
        for event in correspondences:
            owner = str(event["owner_id"])
            source_rows = {
                key: source["objects"][owner] for key, source in sources.items()
            }
            labels = {key: row["label"] for key, row in source_rows.items()}
            available = {key: row["available"] for key, row in source_rows.items()}
            agreement["Q_S2_equal"] += labels["Q_GAIN"] == labels["S_SIGLIP2_AREA"]
            agreement["Q_S2_real_positive_equal"] += (
                available["Q_GAIN"]
                and available["S_SIGLIP2_AREA"]
                and labels["Q_GAIN"] > 0
                and labels["Q_GAIN"] == labels["S_SIGLIP2_AREA"]
            )
            correct, ranks = {}, {}
            for name, row in source_rows.items():
                correct[name] = (
                    None
                    if not row["available"] or event["correspondence"] != "unique"
                    else row["label"] == event["gt_label"]
                )
                ranks[name] = None
                if correct[name] is not None:
                    order = np.argsort(
                        -np.asarray(row["scores"]), kind="stable"
                    ).tolist()
                    target = sources[name]["valid_ids"].index(event["gt_label"])
                    ranks[name] = order.index(target) + 1
            if event["correspondence"] != "unique":
                category = "SHAPE_" + event["correspondence"].upper()
            elif any(not value for value in available.values()):
                category = "SOURCE_UNAVAILABLE"
            else:
                n0, q, s2 = (
                    correct[name] for name in ("N0", "Q_GAIN", "S_SIGLIP2_AREA")
                )
                category = {
                    (True, False, False): "N0_ONLY_CORRECT_Q_S2_JOINTLY_WRONG",
                    (False, True, False): "Q_ONLY_CORRECT",
                    (False, False, True): "S2_ONLY_CORRECT",
                    (False, True, True): "Q_AND_S2_CORRECT_AGAINST_N0",
                    (False, False, False): "ALL_WRONG",
                    (True, True, False): "N0_Q_CORRECT",
                    (True, False, True): "N0_S2_CORRECT",
                    (True, True, True): "ALL_CORRECT",
                }[n0, q, s2]
            pattern = "/".join(
                "UNAVAILABLE"
                if not available[name]
                else "NO_CORRESPONDENCE"
                if correct[name] is None
                else "CORRECT"
                if correct[name]
                else "WRONG"
                for name in ("N0", "Q_GAIN", "S_SIGLIP2_AREA")
            )
            patterns[pattern] += 1
            categories[category] += 1
            objects.append(
                {
                    **event,
                    "source_labels": labels,
                    "available": available,
                    "correct": correct,
                    "gt_class_source_ranks": ranks,
                    "category": category,
                }
            )
        all_objects[scene] = objects
        summary[scene] = {
            "status": "COMPLETE",
            "owners": len(objects),
            "patterns_N0_Q_S2": dict(patterns),
            "categories": dict(categories),
            "agreement": dict(agreement),
            "fallback_counts": {
                name: sum(not row["available"] for row in source["objects"].values())
                for name, source in sources.items()
            },
            "oracle_correctable_object_count": sum(
                row["correct"]["N0"] is False
                and any(
                    row["correct"][name] is True
                    for name in ("Q_GAIN", "S_SIGLIP2_AREA")
                )
                for row in objects
            ),
            "oracle_count_is_not_AP_upper_bound": True,
        }
        by_owner = {row["owner_id"]: row for row in objects}
        for method in COMPOSITIONS[:3]:
            path = (
                Path(local["attempt_root"])
                / "compositions"
                / scene
                / method
                / "decisions.json"
            )
            if not path.is_file():
                continue
            decisions = read_json(path)["objects"]
            outcomes = Counter()
            for owner, decision in decisions.items():
                event = by_owner[int(owner)]
                if decision["changed"]:
                    key = (
                        event["correspondence"]
                        if event["gt_label"] is None
                        else (
                            (
                                "right"
                                if decision["native_label"] == event["gt_label"]
                                else "wrong"
                            )
                            + "_to_"
                            + (
                                "right"
                                if decision["label"] == event["gt_label"]
                                else "wrong"
                            )
                        )
                    )
                    outcomes[key] += 1
            routing[f"{scene}/{method}"] = {
                "reasons": dict(Counter(row["reason"] for row in decisions.values())),
                "available_source_counts": dict(
                    Counter(len(row["available_sources"]) for row in decisions.values())
                ),
                "changed_object_correctness": dict(outcomes),
            }
    return {"scenes": summary, "objects": all_objects, "routing": routing}


def trajectory_diagnostics(config, rows):
    details = {}
    for scene in sorted({row["scene_id"] for row in rows}):
        local = _scene_config(config, scene)
        traces = {}
        for method in ("Q_COMBINE", "Q_GAIN", *COMPOSITIONS[3:]):
            path = (
                Path(local["attempt_root"])
                / "query"
                / scene
                / method
                / "decisions.json"
            )
            if path.is_file():
                traces[method] = read_json(path)
        selected = {
            method: {
                attempt["request_id"]
                for frame in trace["frames"]
                for attempt in frame["results"]
            }
            for method, trace in traces.items()
        }
        scene_details = {}
        for method, trace in traces.items():
            retained = {
                key for values in trace["retained_features"].values() for key in values
            }
            attempts = [
                attempt for frame in trace["frames"] for attempt in frame["results"]
            ]
            successful = {row["request_id"] for row in attempts if row["success"]}
            scene_details[method] = {
                "attempts": len(attempts),
                "successes": len(successful),
                "failures": sum(not row["success"] for row in attempts),
                "retained_features": len(retained),
                "successful_not_finally_retained": len(successful - retained),
                "irreversible_ambiguous_drops": len(
                    trace["discarded_ambiguous_features"]
                ),
                "paid_owner_counts": dict(Counter(row["owner_id"] for row in attempts)),
                "retained_owner_counts": {
                    owner: len(values)
                    for owner, values in trace["retained_features"].items()
                },
            }
            lanes = trace.get("lanes", [])
            if lanes:
                scene_details[method]["lane_fields"] = sorted(lanes[0])
                scene_details[method]["preferred_lanes"] = dict(
                    Counter(row["preferred_lane"] for row in lanes)
                )
                scene_details[method]["winning_lanes"] = dict(
                    Counter(row["winning_lane"] for row in lanes)
                )
                scene_details[method]["fallback_share"] = sum(
                    row["preferred_lane"] != row["winning_lane"] for row in lanes
                ) / len(lanes)
        overlaps = {}
        for left, left_ids in selected.items():
            for right, right_ids in selected.items():
                if left >= right:
                    continue
                union = left_ids | right_ids
                overlaps[f"{left}/{right}"] = {
                    "intersection": len(left_ids & right_ids),
                    "union": len(union),
                    "jaccard": len(left_ids & right_ids) / len(union)
                    if union
                    else None,
                }
        details[scene] = {"methods": scene_details, "request_overlaps": overlaps}
    return details


def released_transitions(rows):
    output = []
    natives = {row["scene_id"]: row for row in rows if row["method_id"] == "N0"}
    for row in rows:
        reference = natives.get(row["scene_id"])
        if reference is None or row["method_id"] == "N0":
            continue
        current_trace, native_trace = (
            _gz(row["trace_path"]),
            _gz(reference["trace_path"]),
        )
        for threshold in (0.5, 0.75):
            old, new = (
                released_object_outcomes(trace, threshold)
                for trace in (native_trace, current_trace)
            )
            gained = sorted(set(new["matched"]) - set(old["matched"]), key=int)
            lost = sorted(set(old["matched"]) - set(new["matched"]), key=int)

            def fp_events(trace, threshold=threshold):
                return sum(
                    (
                        event["event"] == "duplicate"
                        or (event["event"] == "ignore_test" and event["counted_fp"])
                    )
                    for event in trace["events"]
                    if abs(event["overlap_threshold"] - threshold) < 1e-10
                )

            output.append(
                {
                    "scene_id": row["scene_id"],
                    "method_id": row["method_id"],
                    "threshold": threshold,
                    "gained_released_gt_ids": gained,
                    "lost_released_gt_ids": lost,
                    "gained_details": {key: new["matched"][key] for key in gained},
                    "lost_details": {key: old["matched"][key] for key in lost},
                    "native_fp_events": fp_events(native_trace),
                    "method_fp_events": fp_events(current_trace),
                    "native_evaluator_prediction_count": reference[
                        "evaluator_prediction_count"
                    ],
                    "method_evaluator_prediction_count": row[
                        "evaluator_prediction_count"
                    ],
                    "uap_unchanged_with_label_changes": row["changed_owners"] > 0
                    and row["metrics"]["uap"] == reference["metrics"]["uap"],
                    "trace_reference": row["trace_path"],
                    "native_trace_reference": reference["trace_path"],
                }
            )
    return output


def physical_work(config):
    root = Path(config["attempt_root"])
    receipts = list((root / "query").glob("*/*/receipt.json"))
    receipts += list((root / "confirmation").glob("*/query/*/*/receipt.json"))
    query, totals = [], Counter()
    for path in sorted(receipts):
        row = verified_receipt(path)
        physical = row["physical"]
        for key in (
            "model_loads",
            "model_forwards",
            "crop_inputs",
            "cache_hits",
            "inference_seconds",
        ):
            totals[key] += physical.get(key, 0)
        query.append(
            {
                "scene_id": row["scene_id"],
                "method_id": row["method_id"],
                "receipt": str(path),
                "logical_attempts": row["logical"]["attempts"],
                "physical": physical,
            }
        )
    static, captures = [], []
    for path in sorted(
        (root / "confirmation").glob(
            "*/study/scenes/*/semantic_models/siglip2/receipt.json"
        )
    ):
        row = read_json(path)
        totals["static_siglip2_forwards"] += row["physical_attempts_this_invocation"]
        totals["static_siglip2_crops"] += row["physical_crop_inputs_this_invocation"]
        static.append(
            {
                "receipt": str(path),
                "physical_attempts": row["physical_attempts_this_invocation"],
                "physical_crops": row["physical_crop_inputs_this_invocation"],
            }
        )
    for path in sorted(
        (root / "confirmation").glob("*/native/*/mapping_job/receipt.json")
    ):
        row = read_json(path)
        captures.append(
            {
                "receipt": str(path),
                "scene_id": row["scene_id"],
                "scheduled_count": row["scheduled_count"],
                "completed_count": row["completed_count"],
                "elapsed_seconds": row["elapsed_seconds"],
            }
        )
    return {
        "query_totals": dict(totals),
        "query_jobs": query,
        "new_static_jobs": static,
        "new_capture_jobs": captures,
        "shared_dependencies_counted_once": True,
    }


def _display(value):
    return "—" if value is None else f"{100 * value:.6f}"


def report(config):
    root = Path(config["attempt_root"])
    rows = [read_json(path) for path in sorted((root / "rows").glob("*/*/*.json"))]
    for row in rows:
        receipt = verified_receipt(row["evaluation_receipt"])
        if (
            receipt["metrics"] != row["metrics"]
            or receipt["prediction_key"] != row["prediction_key"]
        ):
            raise ValueError("report row differs from actual released evaluation")
        prediction = load_prediction(row["prediction_manifest"])
        if prediction.record_key != row["record_key"]:
            raise ValueError("report prediction changed")
    selection = (
        read_json(root / "selection.json")
        if (root / "selection.json").is_file()
        else None
    )
    gate = (
        read_json(root / "calibration/m6_gate.json")
        if (root / "calibration/m6_gate.json").is_file()
        else None
    )
    methods = (
        COMPOSITIONS if gate is not None and gate["enabled"] else COMPOSITIONS[:-1]
    )
    expected = {
        (role, scene, method)
        for role in ("compose_cal", "regression_only")
        for scene in config["spec"]["data"][role]
        for method in (*CONTROLS, *methods)
    }
    present = {
        (row["role"], row["scene_id"], row["method_id"])
        for row in rows
        if row["status"] == "COMPLETE"
    }
    missing = sorted(expected - present)
    confirmation = (
        read_json(root / "confirmation/receipt.json")
        if (root / "confirmation/receipt.json").is_file()
        else None
    )
    confirmation_status = "NOT_RUN" if confirmation is None else confirmation["status"]
    status = (
        "COMPLETE"
        if not missing
        and selection is not None
        and confirmation_status
        in {"COMPLETE", "NOT_REQUIRED_NO_EFFECTIVE_INTERVENTION"}
        else "PARTIAL"
    )
    means = [
        metric_summary(rows, role, method, config["spec"]["data"][role])
        for role in ("compose_cal", "regression_only", "confirmation")
        for method in (*CONTROLS, *COMPOSITIONS)
        if any(row["role"] == role and row["method_id"] == method for row in rows)
    ]
    contrasts = []
    pairs = [
        (COMPOSITIONS[3], "Q_COMBINE"),
        (COMPOSITIONS[4], "Q_GAIN"),
        (COMPOSITIONS[4], COMPOSITIONS[3]),
        (COMPOSITIONS[2], COMPOSITIONS[1]),
        (COMPOSITIONS[5], "Q_COMBINE"),
        (COMPOSITIONS[5], "Q_GAIN"),
        (COMPOSITIONS[6], COMPOSITIONS[5]),
    ]
    for role in ("compose_cal", "regression_only", "confirmation"):
        for left, right in pairs:
            comparison = compare_methods(
                rows, role, left, right, config["spec"]["data"][role]
            )
            if "mean_deltas" in comparison:
                contrasts.append(comparison)
        values = {
            (row["candidate"], row["reference"]): row
            for row in contrasts
            if row["role"] == role
        }
        if (COMPOSITIONS[3], "Q_COMBINE") in values and (
            COMPOSITIONS[4],
            "Q_GAIN",
        ) in values:
            interaction = {
                key: values[COMPOSITIONS[4], "Q_GAIN"]["mean_deltas"][key]
                - values[COMPOSITIONS[3], "Q_COMBINE"]["mean_deltas"][key]
                for key in ("uap", "miou")
            }
            contrasts.append(
                {
                    "role": role,
                    "candidate": "2x2_INTERACTION",
                    "reference": "(M4-Q_GAIN)-(M3-Q_COMBINE)",
                    "mean_deltas": interaction,
                }
            )
    complement = complementarity(config, rows)
    trajectory = trajectory_diagnostics(config, rows)
    transitions = released_transitions(rows)
    work = physical_work(config)
    comparisons = []
    if selection is not None and selection["nomination"]["nominee"]:
        nominee = selection["nomination"]["nominee"]
        parent = {
            COMPOSITIONS[3]: "Q_COMBINE",
            COMPOSITIONS[4]: "Q_GAIN",
            COMPOSITIONS[5]: "Q_COMBINE",
            COMPOSITIONS[6]: COMPOSITIONS[5],
        }.get(nominee, "N0")
        for role in ("compose_cal", "regression_only", "confirmation"):
            for reference in dict.fromkeys(
                ("N0", selection["nomination"]["best_single_CAL"], parent)
            ):
                comparisons.append(
                    compare_methods(
                        rows, role, nominee, reference, config["spec"]["data"][role]
                    )
                )
    result = {
        "status": status,
        "experiment_status": "COMPLETE"
        if not missing
        else "PARTIAL_MISSING_MANDATORY_ROWS",
        "confirmation_status": confirmation_status,
        "missing_rows": missing,
        "rows": rows,
        "means": means,
        "m6_gate": gate,
        "selection": selection,
        "nominee_comparisons": comparisons,
        "complementarity": complement,
        "contrasts": contrasts,
        "trajectory_diagnostics": trajectory,
        "released_transitions": transitions,
        "physical_work": work,
    }
    destination = root / "reports" / canonical_digest(result)
    write_once(destination / "tables.json", result)
    lines = [
        "# Complementary composition results",
        "",
        f"Status: **{status}**. Confirmation: **{confirmation_status}**.",
        "",
        "Historical CAL and regression scenes are previously exposed; Q_GAIN checkpoint selection already used historical CAL. Cross-fitted temperatures do not create a fresh holdout. Two confirmation scenes cannot establish generalization. No deployment was changed.",
        "",
        "## Table A — available measured performance",
        "",
        "Metrics are percentages. Logical N/S2 counts are conservative required source operations; the common native map is listed separately in each numerical row. Physical shared work is counted once in Table D.",
        "",
        "| Role | Scene | Method | uAP | AP50 | AP25 | mIoU | mAcc | Changed/all owners | Positive/evaluated | Logical N/S2/crops | Physical work / shared dependencies | Source reuse | Evaluation first method |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['role']} | {row['scene_id']} | {row['method_id']} | "
            + " | ".join(_display(row["metrics"].get(key)) for key in METRICS)
            + f" | {row['changed_owners']}/{row['owners']} | {row['positive_owners']}/{row['evaluator_prediction_count']} | {row['logical']['native_requests']}/{row['logical']['siglip2_requests']}/{row['logical']['crop_inputs']} |"
            + f" `{json.dumps(row['physical'], sort_keys=True)}` | {row['source_reuse']} | {row['shared_evaluation_first_method']} |"
        )
    lines += [
        "",
        "| Role | Method | Mean uAP (defined/total) | Mean mIoU (defined/total) | Status |",
        "|---|---|---:|---:|---|",
    ]
    for row in means:
        displays = [
            f"{_display(row['means'][key])} ({row['denominators'][key]['defined']}/{row['denominators'][key]['total']})"
            for key in ("uap", "miou")
        ]
        lines.append(
            f"| {row['role']} | {row['method_id']} | {' | '.join(displays)} | {row['status']} |"
        )
    lines += [
        "",
        f"Missing required control/composition records: {len(missing)}. Missing rows are not zero-valued measurements.",
        "",
        "## Table B — complementarity and routing",
        "",
        "| Scene | Status | Owners | Correctness categories | Source unavailable counts |",
        "|---|---|---:|---|---|",
    ]
    for scene, value in complement["scenes"].items():
        lines.append(
            f"| {scene} | {value['status']} | {value.get('owners', '—')} | {value.get('categories', '—')} | {value.get('fallback_counts', '—')} |"
        )
    lines += [
        "",
        "For scenes with complete source evidence, the numerical tables preserve all-owner correctness, source GT-class ranks, unavailable/unmatched/ambiguous categories and available M1/M2 routing counts. SOURCE_EVIDENCE_INCOMPLETE means those analyses remain pending. Oracle-correctable object counts are diagnostic, not an AP upper bound.",
        "",
        "## Table C — controlled contrasts and trajectory mechanics",
        "",
        "| Role | Contrast | ΔuAP (pp) | ΔmIoU (pp) |",
        "|---|---|---:|---:|",
    ]
    for row in contrasts:
        lines.append(
            f"| {row['role']} | {row['candidate']} − {row['reference']} | {_display(row['mean_deltas']['uap'])} | {_display(row['mean_deltas']['miou'])} |"
        )
    lines += [
        "",
        "Lane preference/winner/fallback counts, request Jaccards, per-owner paid budgets and retained/dropped evidence are in `trajectory_diagnostics`. A changed trajectory alone is not a gain.",
        "",
        "## Table D — frozen nomination, confirmation and operations",
        "",
    ]
    lines += [
        f"Nominee: **{selection['nomination']['nominee'] if selection else 'NOT_FROZEN'}**.",
        f"Experiment commit A: `{selection['experiment_commit_A'] if selection else 'NOT_FROZEN'}`.",
        f"Query physical totals: `{work['query_totals']}`; new native captures: {len(work['new_capture_jobs'])}.",
        "",
        "| Role | Nominee versus | Status | Worst ΔuAP / ΔmIoU (pp) | Every scene nonnegative |",
        "|---|---|---|---:|---|",
    ]
    for row in comparisons:
        worst = row.get("worst_scene_deltas", {})
        lines.append(
            f"| {row['role']} | {row['reference']} | {row['status']} | {_display(worst.get('uap'))} / {_display(worst.get('miou'))} | {row.get('every_scene_nonnegative', '—')} |"
        )
    lines += [
        "",
        "Actual released TP/FN gains/losses and FP events at 0.5/0.75 are linked in `released_transitions`, separately from geometric class correctness. AP increments are not added across objects.",
        "",
        f"Complete numerical tables: `{destination / 'tables.json'}`.",
        f"Publication receipt (written only after verified ordinary push): `{root / 'publication_receipt.json'}`.",
        "",
    ]
    text = "\n".join(lines)
    (destination / "COMPOSITION_RESULTS.md").write_text(text)
    tracked = ROOT / "docs/paper/static_ovmap/COMPOSITION_RESULTS.md"
    tracked.write_text(text)
    handoff = (
        "# Complementary composition handoff\n\n"
        f"Current result: {status}; confirmation: {confirmation_status}. Missing required rows: {len(missing)}.\n\n"
        f"Resolved config: `{root / 'resolved_config.json'}`.\n\n"
        "Resume the actual ordered jobs (unchanged completed content is reused):\n\n```bash\n"
        f"{config['runtime']['native_perception_python']} scripts/evaluation/run_ovimap_composition_study.py --resolved-config {root / 'resolved_config.json'} --phase all\n```\n\n"
        f"External publication receipt: `{root / 'publication_receipt.json'}`.\n"
        "The previous worktree, selection guards, checkpoint/scaler, data roles, geometry and deployment remain unchanged.\n"
    )
    (ROOT / "docs/paper/static_ovmap/COMPOSITION_HANDOFF.md").write_text(handoff)
    atomic_write_json(
        root / "report_receipt.json",
        {
            "status": status,
            "tables": str(destination / "tables.json"),
            "rows": len(rows),
            "missing_rows": missing,
            "results": str(tracked),
            "handoff": str(ROOT / "docs/paper/static_ovmap/COMPOSITION_HANDOFF.md"),
        },
    )
    return {
        "status": status,
        "rows": len(rows),
        "missing_rows": len(missing),
        "tables": str(destination / "tables.json"),
        "results": str(tracked),
        "confirmation_status": confirmation_status,
    }
