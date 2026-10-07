"""Evaluation-only baseline and paired diagnostics. Never imported by predictors."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import contextlib
import gzip
import json
from pathlib import Path
import time

import numpy as np

from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read
from static_ovmap.released_loader import load_released_module


def zipped(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def owner_from_file(filename):
    return None if filename is None else int(Path(filename).stem.removeprefix("owner_"))


def trace_entries(trace):
    """Track final TP score ownership and each emitted FP; preserve ambiguous ties."""
    winners, false = {}, []
    for event in trace["events"]:
        key = tuple(event[k] for k in ("scene_key", "distance_index", "class_index", "overlap_index"))
        row = {k: event[k] for k in ("class_label", "overlap_threshold", "distance_index", "overlap_index")}
        if event["event"] == "first_match":
            winners[(*key, event["gt_id"])] = {**row, "kind": "TP", "owner": owner_from_file(event["candidate_file"]),
                "gt_id": event["gt_id"], "ambiguous_tie": False, "score": event["confidence"]}
        elif event["event"] == "duplicate":
            tied = bool(event["score_identity_ambiguous_tie"])
            winners[(*key, event["gt_id"])] = {**row, "kind": "TP", "owner": owner_from_file(event["tp_score_owner"]),
                "gt_id": event["gt_id"], "ambiguous_tie": tied, "score": event["tp_score"]}
            false.append({**row, "kind": "FP_DUPLICATE", "owner": owner_from_file(event["fp_score_owner"]),
                          "gt_id": event["gt_id"], "ambiguous_tie": tied, "score": event["fp_score"]})
        elif event["event"] == "ignore_test" and event["counted_fp"]:
            false.append({**row, "kind": "FP_UNMATCHED", "owner": owner_from_file(event["candidate_file"]),
                          "gt_id": None, "ambiguous_tie": False, "score": event["confidence"]})
    return [*winners.values(), *false]


def baseline_scene(job):
    binding, scene, root_text = job
    root, index = Path(root_text), ConsumptionIndex()
    row = binding["scenes"][scene]
    target = root / "diagnostics" / scene / "baseline.json"
    source = read(row["G1_source"]["path"])
    scores = {m: read(v["row"]["evaluation_receipt"]) for m, v in row["baseline_rows"].items()}
    input_key = canonical_digest({"binding": binding["identity"], "scene": scene,
        "scoring": {m: s["identity"] for m, s in scores.items()}, "producer": index.identity(__file__)["sha256"]})
    if target.exists():
        result = read(target)
        if result["input_identity"] != input_key:
            raise ValueError("completed diagnostic inputs changed")
        return result
    start = time.perf_counter()
    b0, b1 = scores["EV00_D2"], scores["EV01_G1_V2"]
    for s in scores.values():
        if s["status"] != "COMPLETE":
            raise ValueError("baseline diagnosis requires completed locked official predictions")
        index.identity(s["manifest"])
        index.identity(s["gt_path"], s["context"]["gt"])
        for name in ("matches.json.gz", "trace.json.gz"):
            index.identity(Path(s["manifest"]).with_name(name))
    namespace = load_released_module(b1["context"]["evaluator"]["path"])
    namespace["init"]("Replica" if row["cohort"] == "replica8" else "Scannet200")
    minimum, distance, confidence = (namespace[name][0] for name in ("min_region_sizes", "dist_threshes", "dist_confs"))
    eligible = lambda g: g["instance_id"] >= 1000 and g["vert_count"] >= minimum and g["med_dist"] <= distance and g["dist_conf"] >= confidence
    matches = next(iter(zipped(Path(b1["manifest"]).with_name("matches.json.gz")).values()))
    gt_records = [g for values in matches["gt"].values() for g in values]
    eligible_by_id = {g["instance_id"]: g for g in gt_records if eligible(g)}
    gt_class_counts = Counter(g["label_id"] for g in eligible_by_id.values())
    gt = np.load(b1["gt_path"], allow_pickle=False).reshape(-1)
    gt_ids, gt_sizes = np.unique(gt, return_counts=True)
    all_gt_sizes = dict(zip(map(int, gt_ids), map(int, gt_sizes), strict=True))
    entries = trace_entries(zipped(Path(b1["manifest"]).with_name("trace.json.gz")))
    masks = {}
    for line in Path(b1["manifest"]).read_text().splitlines():
        name, label, rank = line.split()
        masks[owner_from_file(name)] = (Path(b1["manifest"]).parent / name).resolve()
    ledger = []
    for owner_text, base in source["objects"].items():
        owner = int(owner_text)
        path = masks.get(owner)
        if path is None and base["available"]:
            # Small exported regions still have an expanded-registry mask even when
            # the official manifest legitimately omits them.
            path = Path(b1["manifest"]).parents[2] / "shared_masks" / scene / f"owner_{owner}.npy"
            if not path.exists():
                raise ValueError("available recovered owner has no actual expanded-registry mask")
        mask = np.load(path, allow_pickle=False).reshape(-1) if path is not None else np.zeros(gt.shape, bool)
        if mask.shape != gt.shape:
            raise ValueError("diagnostic mask and actual evaluator GT shape differ")
        area = int(mask.sum())
        overlaps = []
        for gid, intersection in zip(*np.unique(gt[mask], return_counts=True), strict=True):
            gid, intersection = int(gid), int(intersection)
            if gid in eligible_by_id:
                overlaps.append({"gt_id": gid, "class": eligible_by_id[gid]["label_id"],
                                 "iou": intersection / (area + all_gt_sizes[gid] - intersection), "intersection": intersection})
        best = max(overlaps, key=lambda x: x["iou"], default=None)
        assigned = max((x["iou"] for x in overlaps if x["class"] == base["label"]), default=0.)
        actual_entries = [e for e in entries if e["owner"] == owner]
        e50 = [e for e in actual_entries if e["overlap_threshold"] == .5]
        reasons = []
        if area < minimum:
            reasons.append("TARGET_SMALL_OR_NOT_EXPORTED")
        if best is None or best["iou"] <= .5:
            reasons.append("GEOMETRIC_INSUFFICIENCY_AT_50")
        elif assigned <= .5:
            reasons.append("CATEGORY_ERROR_AT_50")
        if any(e["kind"] == "FP_DUPLICATE" for e in e50):
            reasons.append("DUPLICATE_SCORE_ENTRY")
        if any(e["kind"] == "TP" and not e["ambiguous_tie"] for e in e50):
            reasons.append("DEFINITE_TP_SCORE_ENTRY")
        if any(e["ambiguous_tie"] for e in e50):
            reasons.append("TIE_OR_AMBIGUOUS_SCORE_IDENTITY")
        ignored = sum(int(n) for gid, n in zip(*np.unique(gt[mask], return_counts=True), strict=True)
                      if int(gid) not in eligible_by_id)
        ledger.append({"owner": owner, "baseline_available": base["available"], "assigned_class": base["label"],
            "evaluation_size": area, "class_agnostic_best_eligible_GT": best,
            "assigned_class_best_iou": assigned, "ignored_group_void_small_pixels": ignored,
            "assigned_class_has_eligible_GT_in_scan": bool(gt_class_counts[base["label"]]),
            "official_manifest_included": owner_text in b1["view"], "raw_explanations_nonexclusive": reasons,
            "actual_score_entries": actual_entries,
            "definite_TP50_entries": sum(e["kind"] == "TP" and not e["ambiguous_tie"] for e in e50),
            "definite_FP50_entries": sum(e["kind"].startswith("FP") and not e["ambiguous_tie"] for e in e50),
            "ambiguous50_entries": sum(e["ambiguous_tie"] for e in e50)})
    old_ranks = [{"owner": int(o), "class": v["label"], "B0_rank": v["rank"], "B1_rank": b1["view"][o]["rank"]}
                 for o, v in b0["view"].items() if o in b1["view"] and v["rank"] != b1["view"][o]["rank"]]
    counter = root / "diagnostics" / scene / "counterfactual_old_B0_ranks.txt"
    counter.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in Path(b1["manifest"]).read_text().splitlines():
        name, label, rank = line.split()
        owner = str(owner_from_file(name))
        rank = b0["view"].get(owner, {}).get("rank", rank)
        lines.append(f'{(Path(b1["manifest"]).parent/name).resolve()} {label} {rank}')
    counter.write_text("\n".join(lines) + ("\n" if lines else ""))
    result = seal({"status": "COMPLETE_CACHE_ONLY_DIAGNOSIS", "scene": scene, "cohort": row["cohort"],
        "input_identity": input_key, "locked_predictions": {m: s["identity"] for m, s in scores.items()},
        "ledger": ledger, "eligible_GT_class_counts": dict(gt_class_counts), "old_rank_changes": old_ranks,
        "counterfactual_manifest": str(counter), "gt_path": b1["gt_path"],
        "evaluator_path": b1["context"]["evaluator"]["path"], "actual_score_entries": entries,
        "new_GPU_inference": 0, "elapsed_seconds": time.perf_counter()-start, "inputs": index.entries()})
    atomic_write_json(target, result)
    print("DIAGNOSED", scene, len(ledger), "candidates", flush=True)
    return result


def diagnose(binding, output_root):
    root = Path(output_root)
    scenes = [s for names in binding["cohorts"].values() for s in names]
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(baseline_scene, [(binding, s, str(root)) for s in scenes]))
    pools, summaries = {}, {}
    for cohort, names in binding["cohorts"].items():
        ordered = [next(r for r in results if r["scene"] == s) for s in names]
        present = set(int(k) for r in ordered for k, n in r["eligible_GT_class_counts"].items() if n)
        explanations = Counter()
        for r in ordered:
            for obj in r["ledger"]:
                obj["assigned_class_has_eligible_GT_in_cohort"] = obj["assigned_class"] in present
                obj["absent_scan_class_present_elsewhere"] = not obj["assigned_class_has_eligible_GT_in_scan"] and obj["assigned_class"] in present
                explanations.update(obj["raw_explanations_nonexclusive"])
        summaries[cohort] = {"candidate_count": sum(len(r["ledger"]) for r in ordered),
                            "explanations_nonexclusive": dict(explanations), "scenes": names}
        target = root / "diagnostics" / "pools" / (cohort + ".json")
        if target.exists():
            pools[cohort] = read(target)
            continue
        namespace = load_released_module(ordered[0]["evaluator_path"])
        namespace["init"]("Replica" if cohort == "replica8" else "Scannet200")
        output = target.parent / cohort
        output.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with (output / "counterfactual.log").open("w") as stream, contextlib.redirect_stdout(stream):
            values = namespace["evaluate"](str(output), [r["counterfactual_manifest"] for r in ordered],
                                            [r["gt_path"] for r in ordered], str(output))
        pools[cohort] = seal({"status": "DIAGNOSTIC_NOT_OFFICIAL_PERFORMANCE", "cohort": cohort,
            "ordered_scene_ids": names, "definition": "B1 predictions; incumbent B0 ranks fixed; new B1 ranks retained",
            "APall": float(values["all_ap"]), "AP50": float(values["all_ap_50%"]), "AP25": float(values["all_ap_25%"]),
            "elapsed_seconds": time.perf_counter()-started, "input_diagnostics": [r["identity"] for r in ordered]})
        atomic_write_json(target, pools[cohort])
    result = seal({"status": "COMPLETE_CACHE_ONLY_DIAGNOSIS", "binding_identity": binding["identity"],
        "summaries": summaries, "scenes": results, "counterfactual_pools": pools,
        "new_GPU_inference": 0, "predictor_GT_input": False})
    atomic_write_json(root / "diagnostics" / "summary.json", result)
    return result
