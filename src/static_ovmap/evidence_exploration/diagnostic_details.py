"""Post-lock supplements: actual ignore events and distinct paired score/GT changes.

This preserves the original cache-only diagnosis and its exactly two diagnostic
pools. The legacy ineligible-ID pixel overlap is not a released ignore count.
"""

from collections import Counter
from pathlib import Path

from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import read

from .analysis import zipped, owner_from_file, trace_entries


COMPARISONS = (("EV03_CONTRAST", "EV02_MARGIN"), ("EV05_ANYUP", "EV04_BILINEAR"),
    ("EV06_OWNER_ANYUP", "EV05_ANYUP"), ("EV06_OWNER_ANYUP", "EV08_NO_DEPTH"),
    ("EV07_COMBINATION", "EV03_CONTRAST"), ("EV07_COMBINATION", "EV06_OWNER_ANYUP"))


def owner_events(trace):
    result = {}
    for event in trace["events"]:
        fields = ["candidate_file", "previous_score_owner", "tp_score_owner", "fp_score_owner"]
        owners = {owner_from_file(event.get(field)) for field in fields} - {None}
        for owner in owners:
            result.setdefault(owner, []).append(event)
    return result


def scored50(trace):
    return [row for row in trace_entries(trace) if row["overlap_threshold"] == .5]


def entry_signature(row):
    # Scores may change without changing the identity of an emitted entry.
    return (row["kind"], row["owner"], row["gt_id"], row["class_label"], row["distance_index"])


def compare_entries(candidate, reference):
    a, b = {entry_signature(x): x for x in candidate}, {entry_signature(x): x for x in reference}
    added, removed = [a[k] for k in a.keys()-b.keys()], [b[k] for k in b.keys()-a.keys()]
    def number(rows, prefix, ambiguous):
        return sum(x["kind"].startswith(prefix) and x["ambiguous_tie"] == ambiguous for x in rows)
    gta = {(x["class_label"], x["gt_id"]) for x in candidate if x["kind"] == "TP"}
    gtb = {(x["class_label"], x["gt_id"]) for x in reference if x["kind"] == "TP"}
    return {"added_definite_TP_score_entries50": number(added, "TP", False),
        "lost_definite_TP_score_entries50": number(removed, "TP", False),
        "added_definite_FP_score_entries50": number(added, "FP", False),
        "removed_definite_FP_score_entries50": number(removed, "FP", False),
        "added_ambiguous_TP_score_entries50": number(added, "TP", True),
        "lost_ambiguous_TP_score_entries50": number(removed, "TP", True),
        "added_ambiguous_FP_score_entries50": number(added, "FP", True),
        "removed_ambiguous_FP_score_entries50": number(removed, "FP", True),
        "new_unique_GT_matches50": len(gta-gtb), "lost_unique_GT_matches50": len(gtb-gta),
        "added_score_entries50": added, "removed_score_entries50": removed,
        "new_unique_GT_ids50": sorted(gta-gtb), "lost_unique_GT_ids50": sorted(gtb-gta)}


def supplement_baselines(binding, root):
    root = Path(root)
    cohort_presence = {}
    records = {scene: read(root/"diagnostics"/scene/"baseline.json") for scene in binding["scenes"]}
    for cohort, names in binding["cohorts"].items():
        cohort_presence[cohort] = {scene: {int(k) for k, n in records[scene]["eligible_GT_class_counts"].items() if n} for scene in names}
    results = {}
    for scene, diagnostic in records.items():
        cohort = binding["scenes"][scene]["cohort"]
        score = read(binding["scenes"][scene]["baseline_rows"]["EV01_G1_V2"]["row"]["evaluation_receipt"])
        trace = zipped(Path(score["manifest"]).with_name("trace.json.gz"))
        events = owner_events(trace)
        rows = []
        for original in diagnostic["ledger"]:
            owner, assigned = original["owner"], original["assigned_class"]
            actual = events.get(owner, [])
            ignore = [x for x in actual if x["event"] == "ignore_test"]
            rows.append({**original, "actual_released_raw_events": actual, "actual_released_ignore_tests": ignore,
                "overlap_pixels_in_ineligible_GT_ids": original["ignored_group_void_small_pixels"],
                "legacy_ignore_field_is_spatial_overlap_not_released_ignore_count": True,
                "class_has_eligible_GT_in_other_capture": any(assigned in presence for name, presence in cohort_presence[cohort].items() if name != scene),
                "unobservable_or_unavailable": not original["baseline_available"],
                "geometry_insufficient_available_candidate_at50": original["baseline_available"] and "GEOMETRIC_INSUFFICIENCY_AT_50" in original["raw_explanations_nonexclusive"],
                "ignored_at50": any(x["overlap_threshold"] == .5 and not x["counted_fp"] for x in ignore)})
        result = seal({"scene": scene, "cohort": cohort, "parent_diagnostic_identity": diagnostic["identity"],
            "ledger": rows, "available_denominator": sum(x["baseline_available"] for x in rows),
            "candidate_denominator": len(rows), "diagnostic_pool_calls": 0})
        atomic_write_json(root/"diagnostics"/scene/"detail_supplement.json", result)
        results[scene] = result
    summary = seal({"status": "BASELINE_DETAILS_COMPLETE", "new_scoring_calls": 0, "new_counterfactual_pools": 0,
        "by_cohort": {cohort: {"candidates": sum(results[s]["candidate_denominator"] for s in names),
            "available": sum(results[s]["available_denominator"] for s in names),
            "geometry_insufficient_available_at50": sum(x["geometry_insufficient_available_candidate_at50"] for s in names for x in results[s]["ledger"]),
            "target_small_available": sum(x["baseline_available"] and x["evaluation_size"] < 100 for s in names for x in results[s]["ledger"]),
            "actually_ignored_at50": sum(x["ignored_at50"] for s in names for x in results[s]["ledger"])} for cohort, names in binding["cohorts"].items()},
        "supplements": {scene: value["identity"] for scene, value in results.items()}})
    atomic_write_json(root/"diagnostics/detail_summary.json", summary)
    return results, cohort_presence


def paired_analysis(binding, root):
    root = Path(root)
    store = read(root/"result_store.json")
    _verified_identity(store)
    diagnostics, presence = supplement_baselines(binding, root)
    rows = {(x["scene"], x["method"]): x for x in store["scene_metrics"]}
    pools = {(x["cohort"], x["method"]): x for x in store["pooled_metrics"]}
    details, owner_ledgers = [], []
    traces = {}
    for scene in binding["scenes"]:
        for method in binding["specification"]["methods"]:
            name = method["id"]
            score = read(rows[scene, name]["evaluation_receipt"])
            path = str(Path(score["manifest"]).with_name("trace.json.gz"))
            if path not in traces:
                traces[path] = zipped(path)
            trace = traces[path]
            decisions = read(rows[scene, name]["candidate_decisions"]["path"])["objects"]
            raw_events = owner_events(trace)
            baseline = {x["owner"]: x for x in diagnostics[scene]["ledger"]}
            for owner, decision in decisions.items():
                base = baseline[int(owner)]
                events = raw_events.get(int(owner), [])
                class_id = decision["proposed_class"]
                absent_here = class_id not in presence[binding["scenes"][scene]["cohort"]][scene]
                elsewhere = any(class_id in labels for other, labels in presence[binding["scenes"][scene]["cohort"]].items() if other != scene)
                entries = [x for x in scored50(trace) if x["owner"] == int(owner)]
                best = base["class_agnostic_best_eligible_GT"]
                owner_ledgers.append({"scene": scene, "cohort": binding["scenes"][scene]["cohort"], "method": name,
                    "owner": int(owner), "decision": decision, "baseline_geometry": base,
                    "actual_raw_events": events, "actual_score_entries50": entries,
                    "corrected_baseline_category_error50": bool(decision["accepted"] and best and best["iou"] > .5
                        and best["class"] != base["assigned_class"] and class_id == best["class"]),
                    "geometry_insufficient_at50": base["geometry_insufficient_available_candidate_at50"],
                    "absent_class_cross_capture_FP50_entries": sum(x["kind"].startswith("FP") for x in entries) if absent_here and elsewhere else 0})
        # Drop detailed attention-independent trace state after this scene.
        for candidate, reference in COMPARISONS:
            sa, sb = (read(rows[scene, name]["evaluation_receipt"]) for name in (candidate, reference))
            a, b = (scored50(traces[str(Path(score["manifest"]).with_name("trace.json.gz"))]) for score in (sa, sb))
            details.append({"scene": scene, "cohort": binding["scenes"][scene]["cohort"], "candidate": candidate, "reference": reference,
                **compare_entries(a, b)})
        traces.clear()
    comparisons = []
    for cohort, names in binding["cohorts"].items():
        for candidate, reference in COMPARISONS:
            scenes = [x for x in details if x["cohort"] == cohort and x["candidate"] == candidate and x["reference"] == reference]
            counts = Counter()
            for scene in scenes:
                counts.update({key: value for key, value in scene.items() if isinstance(value, int)})
            owner_rows = [x for x in owner_ledgers if x["cohort"] == cohort and x["method"] == candidate]
            classes_a = {x["class_id"]: x for x in pools[cohort, candidate]["classes"]}
            classes_b = {x["class_id"]: x for x in pools[cohort, reference]["classes"]}
            metrics = ("apall", "ap50", "ap25", "miou", "macc")
            class_deltas = [{"class_id": cid, "class_name": a["class_name"], **{metric: None if a.get(metric) is None or classes_b[cid].get(metric) is None
                else a[metric]-classes_b[cid][metric] for metric in ("apall", "ap50", "ap25", "iou", "accuracy")}} for cid, a in classes_a.items()]
            decisions_a = Counter()
            decisions_b = Counter()
            for scene in names:
                decisions_a.update(rows[scene, candidate]["counts"])
                decisions_b.update(rows[scene, reference]["counts"])
            comparisons.append({"cohort": cohort, "candidate": candidate, "reference": reference,
                "metric_deltas_fraction": {m: pools[cohort, candidate]["metrics"][m]-pools[cohort, reference]["metrics"][m] for m in metrics},
                "candidate_decisions": dict(decisions_a), "reference_decisions": dict(decisions_b),
                "paired_score_and_GT_counts": dict(counts), "class_deltas_fraction": class_deltas,
                "corrected_baseline_category_errors50": sum(x["corrected_baseline_category_error50"] for x in owner_rows),
                "accepted_geometry_insufficient50": sum(x["decision"]["accepted"] and x["geometry_insufficient_at50"] for x in owner_rows),
                "absent_class_cross_capture_FP50_entries": sum(x["absent_class_cross_capture_FP50_entries"] for x in owner_rows)})
    result = seal({"status": "PAIRED_DIAGNOSTICS_COMPLETE", "result_store_identity": store["identity"],
        "comparisons": comparisons, "paired_scene_details": details, "candidate_method_ledger": owner_ledgers,
        "counts_are_nonexclusive": True, "TP_FP_score_entries_are_not_unique_GT_recall": True})
    atomic_write_json(root/"diagnostics/paired.json", result)
    return result
