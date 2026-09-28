"""Summarize actual released matcher events; never substitute another matcher."""

import gzip
import json
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once

from .evaluation import write_gzip


def read_trace(root, scene, method, rank):
    row = read_json(root / "rows" / scene / method / (rank + ".json"))
    receipt = read_json(row["evaluation_receipt"])
    with gzip.open(Path(row["evaluation_receipt"]).parent / "trace.json.gz", "rt") as handle:
        trace = json.load(handle)
    if trace["parity"]["ap_exact"] is not True or trace["parity"]["pr_and_fn_exact"] is not True:
        raise ValueError("AP attribution requires exact original matcher parity")
    return row, receipt, trace


def _owner(event):
    path = event.get("candidate_file")
    return int(Path(path).stem.removeprefix("owner_")) if path else None


def attribute_scene(binding, scene):
    root = Path(binding["output_root"])
    with gzip.open(root / "diagnostics" / scene / "objects.json.gz", "rt") as handle:
        objects = {row["owner_id"]: row for row in json.load(handle)}
    report = {}
    for rank in ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS"):
        native_row, native_receipt, baseline = read_trace(root, scene, "N0", rank)
        for method in binding["methods"]:
            row, receipt, trace = read_trace(root, scene, method, rank)
            thresholds = []
            for overlap in sorted({e["overlap_index"] for e in baseline["states"]}):
                old = [e for e in baseline["events"] if e["overlap_index"] == overlap]
                new = [e for e in trace["events"] if e["overlap_index"] == overlap]
                old_hits = {(e["class_index"], e["gt_id"]): e for e in old if e["event"] == "first_match"}
                new_hits = {(e["class_index"], e["gt_id"]): e for e in new if e["event"] == "first_match"}

                def with_object(event, method=method):
                    owner = _owner(event)
                    diagnostic = objects.get(owner)
                    return {**event, "owner_id": owner, "semantic_diagnostic": None if diagnostic is None else {
                        "correspondence": diagnostic["correspondence"], "gt_label": diagnostic["gt_label"],
                        "source_labels": {k: v["label"] for k, v in diagnostic["source_evidence"].items()},
                        "source_available": {k: v["available"] for k, v in diagnostic["source_evidence"].items()},
                        "method": diagnostic["methods"][method]}}

                states = [s for s in trace["states"] if s["overlap_index"] == overlap]
                old_states = {s["class_index"]: s for s in baseline["states"] if s["overlap_index"] == overlap}
                class_deltas = [{"class_label": s["class_label"], "class_index": s["class_index"],
                                 "native_ap": old_states[s["class_index"]]["ap"], "method_ap": s["ap"],
                                 "native_hard_fn": old_states[s["class_index"]]["hard_false_negatives"],
                                 "method_hard_fn": s["hard_false_negatives"]} for s in states
                                if s["ap"] != old_states[s["class_index"]]["ap"]
                                or s["hard_false_negatives"] != old_states[s["class_index"]]["hard_false_negatives"]]
                thresholds.append({"overlap_index": overlap, "overlap_threshold": states[0]["overlap_threshold"],
                    "added_gt_matches": [with_object(new_hits[k]) for k in sorted(new_hits.keys() - old_hits.keys())],
                    "lost_gt_matches": [with_object(old_hits[k]) for k in sorted(old_hits.keys() - new_hits.keys())],
                    "native_matched_gt": len(old_hits), "method_matched_gt": len(new_hits),
                    "native_duplicates": [with_object(e) for e in old if e["event"] == "duplicate"],
                    "method_duplicates": [with_object(e) for e in new if e["event"] == "duplicate"],
                    "native_ignored": [with_object(e) for e in old if e["event"] == "ignore_test" and not e["counted_fp"]],
                    "method_ignored": [with_object(e) for e in new if e["event"] == "ignore_test" and not e["counted_fp"]],
                    "native_unmatched_fp": [with_object(e) for e in old if e["event"] == "ignore_test" and e["counted_fp"]],
                    "method_unmatched_fp": [with_object(e) for e in new if e["event"] == "ignore_test" and e["counted_fp"]],
                    "class_ap_changes": class_deltas})
            native_eligible = set(native_receipt["view"])
            eligible = set(receipt["view"])
            report[f"{method}/{rank}"] = {"scene": scene, "method": method, "rank_mode": rank,
                "metrics": row["metrics"], "native_metrics": native_row["metrics"], "thresholds": thresholds,
                "eligibility_added_owners": sorted(map(int, eligible - native_eligible)),
                "eligibility_removed_owners": sorted(map(int, native_eligible - eligible)),
                "trace_parity": trace["parity"], "evaluation_receipt": row["evaluation_receipt"],
                "interpretation": "GT match-set differences; duplicate/ignore events retained separately, not one-event-per-prediction"}
    write_gzip(root / "diagnostics" / scene / "released_attribution.json.gz", report)
    compact = {key: {k: value[k] for k in ("metrics", "native_metrics", "eligibility_added_owners", "eligibility_removed_owners")} | {
        "AP25_AP50": [{k: t[k] for k in ("overlap_threshold", "added_gt_matches", "lost_gt_matches", "class_ap_changes")}
                      for t in value["thresholds"] if t["overlap_threshold"] in (.25, .5)]}
               for key, value in report.items()}
    write_once(root / "diagnostics" / scene / "attribution_AP25_AP50.json", compact)
    return compact
