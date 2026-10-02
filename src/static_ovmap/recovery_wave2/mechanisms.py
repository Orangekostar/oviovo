"""Post-lock diagnostics from genuine source, matcher and native-state receipts."""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re

import numpy as np

from static_ovmap.module_validation.contracts import canonical_digest

from .binding import PathResolver, read
from .evaluation import scorer_context
from .light import RECOVERIES


def read_gzip(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def _owner(path):
    match = re.fullmatch(r"owner_([1-9][0-9]*)", Path(path).stem)
    if match is None:
        raise ValueError("matcher candidate does not identify an actual exported owner")
    return int(match.group(1))


def _state_key(row):
    return (row["distance_index"], row["overlap_index"], row["class_index"])


def _gt_key(row):
    return (*_state_key(row), Path(row["scene_key"]).parent.name, row["gt_id"])


def released_attribution(trace, added):
    if not trace["parity"]["ap_exact"] or not trace["parity"]["pr_and_fn_exact"]:
        raise ValueError("released matcher parity is not verified")
    added, winners, false, thresholds = set(added), {}, [], {}
    for row in trace["events"]:
        thresholds[_state_key(row)] = row["overlap_threshold"]
        if row["event"] == "first_match":
            key = _gt_key(row)
            if key in winners:
                raise ValueError("multiple initial matcher entries for one GT")
            winners[key] = ({_owner(row["candidate_file"])}, row["confidence"])
        elif row["event"] == "duplicate":
            key, candidate = _gt_key(row), {_owner(row["candidate_file"])}
            previous, score = winners[key]
            if score != row["previous_score"]:
                raise ValueError("duplicate score does not follow the actual preceding matcher entry")
            if row["confidence"] > score:
                winners[key], losers = (candidate, row["confidence"]), previous
            elif row["confidence"] < score:
                losers = candidate
            else:
                losers = previous | candidate
                winners[key] = (losers, score)
            false.append((_state_key(row), losers, "duplicate_min_score"))
        elif row["event"] == "ignore_test" and row["counted_fp"]:
            false.append((_state_key(row), {_owner(row["candidate_file"])}, "unvisited_counted_fp"))
    actual = defaultdict(Counter)
    for key in winners:
        actual[key[:3]]["TP"] += 1
    for key, _, _ in false:
        actual[key]["FP"] += 1
    for state in trace["states"]:
        key = _state_key(state)
        thresholds[key] = state["overlap_threshold"]
        truth = np.asarray(state["y_true"])
        if int((truth == 1).sum()) != actual[key]["TP"] or int((truth == 0).sum()) != actual[key]["FP"]:
            raise ValueError("event contribution counts differ from verified terminal score entries")
    counts = {str(float(value)): Counter() for value in thresholds.values()}
    names = ("total_tp_score_entries", "total_fp_score_entries", "added_tp_score_entries", "added_fp_score_entries",
             "ambiguous_added_tp_score_entries", "ambiguous_added_fp_score_entries", "hard_false_negatives")
    for row in counts.values():
        row.update({name: 0 for name in names})

    def count(key, owners, kind):
        row = counts[str(float(thresholds[key]))]
        row["total_" + kind + "_score_entries"] += 1
        if owners <= added:
            row["added_" + kind + "_score_entries"] += 1
        elif owners & added:
            row["ambiguous_added_" + kind + "_score_entries"] += 1

    for key, (owners, _) in winners.items():
        count(key[:3], owners, "tp")
    for key, owners, _ in false:
        count(key, owners, "fp")
    for state in trace["states"]:
        counts[str(float(state["overlap_threshold"]))]["hard_false_negatives"] += state["hard_false_negatives"]
    return {"by_overlap": {key: dict(value) for key, value in counts.items()},
        "tp_matches": [{"key": list(key), "possible_owners": sorted(owners), "score": score}
                       for key, (owners, score) in winners.items()],
        "fp_entries": [{"state_key": list(key), "possible_owners": sorted(owners), "kind": kind}
                       for key, owners, kind in false],
        "unit": "RELEASED_SCORE_ENTRIES_PER_OVERLAP_NOT_UNIQUE_OBJECTS",
        "duplicate_and_unvisited_FP_are_both_counted": True,
        "numeric_ties_do_not_establish_unique_candidate_identity": True}


def matcher_comparison(baseline, variant, added):
    before = released_attribution(baseline, set())
    after = released_attribution(variant, added)
    old = {tuple(row["key"]): row for row in before["tp_matches"]}
    new = {tuple(row["key"]): row for row in after["tp_matches"]}
    loss = [old[key] for key in sorted(old.keys() - new.keys())]
    displaced, ambiguous = [], []
    for key in sorted(old.keys() & new.keys()):
        previous, current = set(old[key]["possible_owners"]), set(new[key]["possible_owners"])
        if previous.isdisjoint(added) and current <= set(added):
            displaced.append({"key": list(key), "previous_owners": sorted(previous), "new_owners": sorted(current)})
        elif previous.isdisjoint(added) and current & set(added):
            ambiguous.append({"key": list(key), "possible_new_owners": sorted(current)})
    return {"actual": after, "baseline_by_overlap": before["by_overlap"],
        "lost_baseline_GT_TP_entries": loss, "new_GT_TP_entries": [new[key] for key in sorted(new.keys() - old.keys())],
        "old_TP_score_displaced_by_added": displaced, "old_TP_score_displacement_ambiguous": ambiguous}


def light_scene(binding, scene, *, map_id="BB00_NATIVE", subdir=None):
    root = Path(binding["output_root"]) / "light" / scene / map_id
    if subdir:
        root /= subdir
    lock, evaluation = read(root / "receipt.json"), read(root / "evaluation_rows.json")
    if lock["status"] != "PREDICTIONS_LOCKED" or evaluation["prediction_lock_identity"] != lock["identity"]:
        raise ValueError("mechanism analysis requires the complete actual prediction lock")
    if map_id == "BB00_NATIVE":
        baseline_root = Path(binding["output_root"]) / "light" / scene / map_id
        baseline_lock = read(baseline_root / "receipt.json")
        baseline_row = next(row for row in read(baseline_root / "evaluation_rows.json")["rows"] if row["method"] == "RW_B_D2")
        baseline_condition = baseline_lock["conditions"]["RW_B_D2"]
        baseline_decisions = read(baseline_root / "decisions/RW_B_D2.json")["old_owners"]
    else:
        raise ValueError("new-map light diagnostics require explicit map-local baseline support")
    resolver = PathResolver(binding["path_map"])
    baseline_receipt = resolver.rewrite(read(baseline_row["evaluation_receipt"]))
    baseline_trace = read_gzip(Path(baseline_receipt["manifest"]).with_name("trace.json.gz"))
    result = []
    for row in evaluation["rows"]:
        method, condition = row["method"], lock["conditions"][row["method"]]
        current = resolver.rewrite(read(row["evaluation_receipt"]))
        current_trace = read_gzip(Path(current["manifest"]).with_name("trace.json.gz"))
        audit = read(root / "decisions" / (method + ".json"))
        old = audit["old_owners"]
        probability_changes = [owner for owner in old if old[owner]["probabilities"] != baseline_decisions[owner]["probabilities"]]
        label_changes = [owner for owner in old if old[owner]["label"] != baseline_decisions[owner]["label"]]
        rank_changes = [{"owner": int(owner), "baseline": value, "actual": current["view"][owner]}
            for owner, value in baseline_receipt["view"].items()
            if owner in current["view"] and value["rank"] != current["view"][owner]["rank"]]
        delta = np.asarray(current["confusion"], np.int64) - np.asarray(baseline_receipt["confusion"], np.int64)
        ids = [0, *[value for value in current["context"]["valid_ids"] if value != 0]]
        sparse = [{"GT_class": ids[i], "predicted_class": ids[j], "point_delta": int(delta[i, j])}
                  for i, j in zip(*np.nonzero(delta), strict=True)]
        added = set(map(int, condition["recovered_labels"]))
        result.append({"scene": scene, "map_id": map_id, "method": method,
            "source_probability_equal_on_incumbents": not probability_changes,
            "probability_changed_incumbents": list(map(int, probability_changes)),
            "class_changed_incumbents": list(map(int, label_changes)),
            "owner_partition_equal": condition["owner_partition_sha256"] == baseline_condition["owner_partition_sha256"],
            "semantic_equal": condition["semantic_sha256"] == baseline_condition["semantic_sha256"],
            "official_view_equal": current["view"] == baseline_receipt["view"],
            "scorer_context_equal": scorer_context(current["context"]) == scorer_context(baseline_receipt["context"]),
            "old_owner_rank_changes": rank_changes, "point_confusion_delta": sparse,
            "point_confusion_diagonal_delta": int(np.trace(delta)),
            "method_source_available_owners": sorted(added),
            "method_used_request_ids": {owner: source["used_request_ids"] for owner, source in audit["recovery"].items() if source["available"]},
            "matcher": matcher_comparison(baseline_trace, current_trace, added),
            "evaluation_identity": current["identity"], "trace_parity": current["trace_parity"]})
    return result


def recovery_funnel(binding, scene, *, map_id="BB00_NATIVE"):
    root = Path(binding["output_root"])
    source_root = root / "recovery" / scene / map_id
    registry, manifest, cached, plan, fc = (read(source_root / (name + ".json")) for name in
        ("registry", "captured_requests", "cached_sources", "fc_request_plan", "fc_recovery_receipt"))
    sources = dict(cached["sources"])
    sources.update({arm: read(value["path"])["objects"] for arm, value in fc["sources"].items()})
    evaluation = read(root / "light" / scene / map_id / "evaluation_rows.json")
    rows = []
    for method, arm in RECOVERIES.items():
        objects = sources[arm]
        check = evaluation["registry_checks"][method]
        eligible = {owner for owner, source in cached["sources"]["U1"].items() if source["available"]}
        attempted = set(cached[arm + "_request_ids"]) if arm in {"U2", "U3"} else set()
        rows.append({"scene": scene, "map_id": map_id, "method": method,
            "raw_positive_owners": registry["raw_positive_owner_count"],
            "native_painted_owners": registry["native_painted_owner_count"],
            "capped_source_min100_candidates": len(registry["candidates"]),
            "registry_excluded_owners": len(registry["excluded"]),
            "registry_exclusions_by_reason": dict(Counter(row["reason"] for row in registry["excluded"])),
            "unowned_source_min100_or_cap_excluded": sum(row["reason"] != "ALREADY_ACTIVE_NATIVE_PAINTED" for row in registry["excluded"]),
            "single_retained_native_eligible": len(eligible),
            "candidates_with_legal_captured_views": sum(bool(manifest["views"][f"owner:{row['raw_owner']}"]) for row in registry["candidates"]),
            "legal_captured_request_count": sum(len(manifest["views"][f"owner:{row['raw_owner']}"]) for row in registry["candidates"]),
            "source_available_owners": sum(source["available"] for source in objects.values()),
            "used_evidence_requests": sum(len(source["used_request_ids"]) for source in objects.values()),
            "FC_preselected_requests": len(attempted),
            "FC_cap_excluded_requests": sum(not plan["requests"][rid]["authorized"] for rid in attempted),
            "FC_authorized_initially_missing_images": len({plan["requests"][rid]["image_content_identity"] for rid in attempted
                                                          if plan["requests"][rid]["authorized"] and not plan["requests"][rid]["cached_in_initial_inventory"]}),
            "exported_source_positive_owners": check["new_positive_owners"],
            "official_target_min100_owners": check["recovered_official_eligible_owners"],
            "target_small_owners": check["recovered_source_eligible_but_target_small"],
            "unavailable_by_reason": dict(Counter(source["reason"] for source in objects.values() if not source["available"])),
            "clipping": registry["candidates"], "source_min100_is_not_target_min100": True})
    return rows


def paired_classification(binding, scene):
    root = Path(binding["output_root"])
    light = root / "light" / scene / "BB00_NATIVE"
    lock = read(light / "receipt.json")
    row = next(row for row in read(light / "evaluation_rows.json")["rows"] if row["method"] == "RW_U1_NATIVE_SINGLE")
    receipt = PathResolver(binding["path_map"]).rewrite(read(row["evaluation_receipt"]))
    one, two = (read(light / "decisions" / (method + ".json"))["recovery"] for method in
                ("RW_U1_NATIVE_SINGLE", "RW_U2_FC_MATCHED_SINGLE"))
    common = sorted(owner for owner in one if one[owner]["available"] and two[owner]["available"])
    gt = np.load(receipt["gt_path"], allow_pickle=False) // 1000
    mask_root = light / "evaluation" / lock["conditions"]["RW_U1_NATIVE_SINGLE"]["owner_partition_sha256"] / "shared_masks" / scene
    ids = set(receipt["context"]["valid_ids"])
    result = []
    for owner in common:
        if one[owner]["used_request_ids"] != two[owner]["used_request_ids"]:
            raise ValueError("paired U1/U2 success did not use the exact same selected request")
        mask = np.load(mask_root / f"owner_{owner}.npy", allow_pickle=False)
        valid = mask & np.isin(gt, list(ids))
        labels, counts = np.unique(gt[valid], return_counts=True)
        maximum = counts.max() if len(counts) else 0
        dominant = labels[counts == maximum]
        majority = int(dominant[0]) if len(dominant) == 1 else None
        result.append({"scene": scene, "owner": int(owner), "request_ids": one[owner]["used_request_ids"],
            "U1_label": one[owner]["label"], "U2_label": two[owner]["label"],
            "label_agreement": one[owner]["label"] == two[owner]["label"],
            "projected_support_points": int(mask.sum()), "valid_GT_points": int(valid.sum()),
            "unique_GT_majority_class": majority, "GT_majority_purity": float(maximum / valid.sum()) if valid.any() else None,
            "U1_correct_points": int((valid & (gt == one[owner]["label"])).sum()),
            "U2_correct_points": int((valid & (gt == two[owner]["label"])).sum()),
            "U1_majority_correct": None if majority is None else one[owner]["label"] == majority,
            "U2_majority_correct": None if majority is None else two[owner]["label"] == majority})
    return {"scene": scene, "common_success_count": len(common), "objects": result,
        "U1_full_available": sum(value["available"] for value in one.values()),
        "U2_full_available": sum(value["available"] for value in two.values()),
        "technical_failures_are_excluded_from_paired_classification": True,
        "score_scales_are_different_and_not_compared_as_calibrated_probabilities": True,
        "GT_used_only_after_prediction_lock_for_diagnostics": True}


def native_actions(binding, scene, map_id):
    root = Path(binding["output_root"]) / "maps" / scene / map_id
    receipt = read(root / "map_receipt.json")
    if receipt["status"] != "COMPLETE":
        raise ValueError("native mechanism evidence requires a completed own map")
    totals, frames = Counter(), []
    for path in sorted((root / "diagnostics").glob("*/association_plan.json")):
        raw, state = read(path), read(path.with_name("realized_state.json"))
        plan, native = raw["plan"], state["association"]
        planned = {int(group): owner for group, owner in native["planned_owners"].items()}
        prior = set(native["prior_owners"].values()) - {0}
        group_owners = defaultdict(set)
        for segment in state["segments"]:
            if segment["input_instance_label"] == 0:
                totals["background_segments_without_object_assignment"] += 1
                continue
            group_owners[segment["input_instance_label"]].add(segment["realized_frame_assignment"])
        positive = [(group, owner) for group, owners in group_owners.items() for owner in owners if owner > 0]
        fresh = [(group, owner) for group, owner in positive if owner not in prior]
        row = {"frame_id": raw["frame_id"], "action_counts": plan["action_counts"],
            "planned_pairs": plan["accepted_pairs"], "duplicate_owner_followers": plan["duplicate_owner_followers"],
            "candidate_veto_visits": native["candidate_vetoes"], "alias_veto_visits": native["alias_vetoes"],
            "mixed_fallback_constrained_candidate_visits": native["mixed_fallback_constrained_candidate_visits"],
            "explicit_allocated_owners": native["allocated_owners"],
            "planning_state_unchanged": raw["native_allocation"]["state_before"] == raw["native_allocation"]["state_after"],
            "planned_realized_discrepancies": sum(planned[group] != owner for group, owner in positive if group in planned),
            "positive_group_owner_pairs": len(positive), "fresh_group_owner_pairs": len(fresh),
            "realized_group_owners": {str(group): sorted(owners) for group, owners in group_owners.items()},
            "fresh_pairs": fresh,
            "realized_frame_vs_count_owner_discrepancies": sum(segment.get("owner_discrepancy") is True for segment in state["segments"] if segment["input_instance_label"] > 0),
            "realized_frame_vs_count_owner_unknown": sum(segment.get("owner_discrepancy") is None for segment in state["segments"] if segment["input_instance_label"] > 0)}
        if not row["planning_state_unchanged"] or row["explicit_allocated_owners"]:
            raise ValueError("main A planner mutated native allocation before fallback integration")
        totals.update(plan["action_counts"])
        totals.update({key: row[key] for key in ("duplicate_owner_followers", "candidate_veto_visits", "alias_veto_visits",
            "mixed_fallback_constrained_candidate_visits", "positive_group_owner_pairs", "fresh_group_owner_pairs",
            "planned_realized_discrepancies", "realized_frame_vs_count_owner_discrepancies", "realized_frame_vs_count_owner_unknown")})
        frames.append(row)
    return {"scene": scene, "map_id": map_id, "frames": frames, "totals": dict(totals),
        "fresh_group_owner_rate": totals["fresh_group_owner_pairs"] / totals["positive_group_owner_pairs"] if totals["positive_group_owner_pairs"] else None,
        "fresh_definition": "ACTUAL_MODE4_FRAME_ASSIGNMENT_NOT_IN_FACTOR0_PREINSERT_PRIOR_OWNER_SET",
        "candidate_and_alias_counts_are_visits_not_unique_objects": True,
        "numeric_new_owner_ids_are_not_alone_evidence_of_partition_change": True,
        "map_lock_identity": receipt["input_identity"]}


def sam_actions(binding, scene, map_id):
    root = Path(binding["output_root"])
    front = read(root / "frontend" / scene / "S1/receipt.json")
    mapping = read(root / "maps" / scene / map_id / "map_receipt.json")
    if mapping["status"] != "COMPLETE":
        raise ValueError("SAM mechanism evidence requires its complete own map")
    totals, decisions, reasons, frames = Counter(), Counter(), Counter(), []
    for frame in front["frames"]:
        totals.update({key: frame[key] for key in ("added_pixels", "protected_crop_pixels")})
        for track in frame["tracks"]:
            decisions[track["action"]] += 1
    if map_id == "RW_S2_CONFLICT":
        for path in sorted((root / "maps" / scene / map_id / "diagnostics").glob("*/s2_conflict.json")):
            row = read(path)
            if row["state_before"] != row["state_after"]:
                raise ValueError("S2 diagnostic probe modified its own native map")
            totals["stable_known_pixels"] += row["stable_known_pixels"]
            totals["removed_additions"] += row["removed_additions"]
            for track in row["tracks"]:
                reasons[track["self_resolution"]] += 1
                totals["history_abstention_tracks"] += int(track["history_abstention"])
                totals["suppressed_tracks"] += int(track["decision"] == "DROP_SAM_ADDITIONS")
                if track["known_self"] is not None:
                    totals["known_self_track_pixels"] += track["known_self"]
                    totals["known_other_track_pixels"] += track["known_other"]
            frames.append({key: row[key] for key in ("frame_id", "chunk", "past_snapshot_frames", "stable_known_pixels", "removed_additions", "tracks")})
    return {"scene": scene, "map_id": map_id, "S1_frontend_identity": front["identity"],
        "protected_crop_pixels": front["protected_crop_pixels"], "S1_added_pixels": front["added_pixels"],
        "all_positive_crop_pixels_unchanged": front["all_positive_crop_pixels_unchanged"],
        "original_separate_crop_groups_preserved": front["original_separate_crop_groups_preserved"],
        "changed_canonical_S1_frames": front["changed_canonical_frames"],
        "S1_track_actions": dict(decisions), "S2_totals": dict(totals) if map_id == "RW_S2_CONFLICT" else None,
        "S2_self_resolution": dict(reasons), "S2_frames": frames,
        "new_SAM_forwards": 0, "new_CropFormer_forwards": 0, "factor0_history_snapshots": 2,
        "unknown_space_is_not_foreign_evidence": True}
