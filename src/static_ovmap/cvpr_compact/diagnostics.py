"""Post-lock recovery coverage and unchanged released matcher attribution."""

from collections import Counter
from pathlib import Path

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read
from static_ovmap.recovery_wave2.mechanisms import matcher_comparison, read_gzip

from .projected_views import _verified_identity
from .partial_execution import PARTIAL_RESULT, partition_methods, semantic_block, validate_partial_scope
from .protocol import load_spec
from .runtime import require_frozen_execution


TABLE3_METHODS = {"NONE": "CT_A1_E", "U2": "CT_H_U2", "G1": "CT_A3_ER", "G3": "CT_G3"}


def _seal(value):
    value["identity"] = canonical_digest({key: item for key, item in value.items() if key != "identity"})
    return value


def _diagnostic_blocks(spec, receipt):
    if receipt["status"] == "COMPLETE":
        return {}
    if receipt["status"] != PARTIAL_RESULT:
        raise ValueError("recovery diagnosis scene is incomplete without proved technical blocks")
    methods, blocks = partition_methods(spec, receipt["scene"], receipt["available_recovery_sources"], receipt["blocked_sources"])
    expected = [row["id"] for row in spec["methods"] if "replica8" in row["cohorts"]]
    if (receipt["expected_method_ids"] != expected or receipt["blocked_methods"] != blocks
            or set(receipt["conditions"]) != {row["id"] for row in methods}):
        raise ValueError("partial diagnosis must retain every independent condition and its exact blocked scope")
    return blocks


def common_success_comparison(fc, native, *, technical_block=None):
    if fc is None:
        if technical_block is None:
            raise ValueError("missing common-success FC source requires its actual technical block")
        _verified_identity(technical_block)
        return {"status": "BLOCKED_TECHNICAL", "technical_block": technical_block,
            "common_success_owners": None, "label_agreement_count": None, "objects": [],
            "new_model_inference": 0, "new_benchmark_row": False,
            "cross_encoder_scores_commensurate": False}
    if fc["valid_ids"] != native["valid_ids"] or set(fc["objects"]) != set(native["objects"]):
        raise ValueError("same-view common-success comparison changed its candidate registry or vocabulary")
    usable = [{int(owner) for owner, row in source["objects"].items() if row["available"]} for source in (fc, native)]
    objects = []
    for owner in sorted(usable[0] & usable[1]):
        first, second = (source["objects"][str(owner)] for source in (fc, native))
        if (len(first["used_request_ids"]) != 1 or first["used_request_ids"] != second["used_request_ids"]):
            raise ValueError("common-success FC/Native comparison requires the same single G1 view")
        for row in (first, second):
            if (row["label"] not in fc["valid_ids"] or len(row["scores"]) != len(fc["valid_ids"])
                    or not np.isfinite(row["scores"]).all()):
                raise ValueError("common-success comparison requires complete saved source scores")
        objects.append({"owner": owner, "request_id": first["used_request_ids"][0],
            "FC_label": first["label"], "Native_label": second["label"],
            "FC_scores": first["scores"], "Native_scores": second["scores"],
            "FC_score_kind": first.get("score_kind"), "Native_score_kind": second.get("score_kind"),
            "labels_agree": first["label"] == second["label"]})
    return {"status": "COMPLETE", "valid_ids": fc["valid_ids"], "objects": objects,
        "common_success_owners": sorted(usable[0] & usable[1]),
        "FC_only_success_owners": sorted(usable[0] - usable[1]),
        "native_only_success_owners": sorted(usable[1] - usable[0]),
        "label_agreement_count": sum(row["labels_agree"] for row in objects),
        "new_model_inference": 0, "new_benchmark_row": False,
        "cross_encoder_scores_commensurate": False, "label_agreement_is_not_accuracy": True}


def _half_state(trace):
    selected = {(row["distance_index"], row["overlap_index"]) for row in trace["states"]
                if abs(row["overlap_threshold"] - .5) <= 1e-12}
    if len(selected) != 1:
        raise ValueError("released TP50 attribution requires its single actual official distance/overlap state")
    return next(iter(selected))


def analyze_scoring_comparison(baseline_path, current_path, added, *, index=None):
    index = index or ConsumptionIndex()
    documents, traces = [], []
    for path in (baseline_path, current_path):
        index.identity(path)
        record = read(path)
        if record["status"] != "COMPLETE":
            raise ValueError("matching diagnosis requires completed official scoring")
        trace_path = Path(record["manifest"]).with_name("trace.json.gz")
        index.identity(trace_path)
        documents.append(record)
        traces.append(read_gzip(trace_path))
    before, after = documents
    if (before["context"]["valid_ids"] != after["context"]["valid_ids"]
            or before["context"]["runtime_overlaps"] != after["context"]["runtime_overlaps"]
            or before["context"]["gt"]["sha256"] != after["context"]["gt"]["sha256"]
            or before["context"]["evaluator"]["sha256"] != after["context"]["evaluator"]["sha256"]):
        raise ValueError("matching diagnosis changed its actual GT, vocabulary or released scoring definition")
    state = _half_state(traces[1])
    if _half_state(traces[0]) != state:
        raise ValueError("matching diagnosis changed its actual loaded AP50 state")
    comparison = matcher_comparison(traces[0], traces[1], set(added))
    halves = [row for threshold, row in comparison["actual"]["by_overlap"].items()
              if abs(float(threshold) - .5) <= 1e-12]
    if len(halves) != 1:
        raise ValueError("released matcher omitted or duplicated the AP50 score-entry counts")
    half = halves[0]
    rank_changes = []
    for owner, row in before["view"].items():
        current = after["view"].get(owner)
        new_rank = current["rank"] if current is not None else "0.000000"
        if current is None or new_rank != row["rank"]:
            rank_changes.append({"owner": int(owner), "rank_before": row["rank"], "rank_after": new_rank,
                "label_before": row["label"], "label_after": current["label"] if current is not None else None,
                "exported_before": True, "exported_after": current is not None})
    def at_half(rows):
        return [row for row in rows if tuple(row["key"][:2]) == state]
    old_confusion, new_confusion = (np.asarray(row["confusion"], np.int64) for row in documents)
    if old_confusion.shape != new_confusion.shape or old_confusion.sum() != new_confusion.sum():
        raise ValueError("post-lock confusion comparison changed whole-scene semantic coverage")
    delta = new_confusion - old_confusion
    ids = [0, *[value for value in after["context"]["valid_ids"] if value != 0]]
    sparse = [{"GT_class": ids[i], "predicted_class": ids[j], "point_delta": int(delta[i, j])}
              for i, j in zip(*np.nonzero(delta), strict=True)]
    ambiguous_tp = half["ambiguous_added_tp_score_entries"]
    ambiguous_fp = half["ambiguous_added_fp_score_entries"]
    return {"added_TP50": half["added_tp_score_entries"], "added_FP50": half["added_fp_score_entries"],
        "ambiguous_added_TP50": ambiguous_tp, "ambiguous_added_FP50": ambiguous_fp,
        "added_matching_ambiguous": bool(ambiguous_tp or ambiguous_fp),
        "total_TP50": half["total_tp_score_entries"], "total_FP50": half["total_fp_score_entries"],
        "lost_old_GT_matches50": at_half(comparison["lost_baseline_GT_TP_entries"]),
        "old_TP_displaced_by_added50": at_half(comparison["old_TP_score_displaced_by_added"]),
        "old_TP_displacement_ambiguous50": at_half(comparison["old_TP_score_displacement_ambiguous"]),
        "old_owner_rank_changes": rank_changes, "point_confusion_delta": sparse,
        "point_confusion_diagonal_delta": int(np.trace(delta)), "matcher": comparison,
        "unit": "RELEASED_SCORE_ENTRIES_NOT_UNIQUE_OBJECTS", "GT_diagnostic_only_after_lock": True}


def aggregate_recovery_diagnostics(spec, receipts):
    expected = spec["cohorts"]["replica8"]
    by_scene = {}
    for row in receipts:
        _verified_identity(row)
        if row["scene"] in by_scene:
            raise ValueError("duplicate recovery diagnosis scene")
        if row["candidate_count"] < 0:
            raise ValueError("recovery diagnosis scene is incomplete")
        _diagnostic_blocks(spec, row)
        by_scene[row["scene"]] = row
    if set(by_scene) != set(expected):
        raise ValueError("complete Table3 diagnosis requires all eight fixed Replica scenes")
    denominator = sum(by_scene[scene]["candidate_count"] for scene in expected)
    arms = {}
    for arm, method in TABLE3_METHODS.items():
        missing = {scene: _diagnostic_blocks(spec, by_scene[scene])[method]
                   for scene in expected if method in _diagnostic_blocks(spec, by_scene[scene])}
        measured_scenes = [scene for scene in expected if scene not in missing]
        rows = [by_scene[scene]["conditions"][method] for scene in measured_scenes]
        for scene, row in zip(measured_scenes, rows, strict=True):
            n = row["source_available_additions"]
            if (not 0 <= n <= by_scene[scene]["candidate_count"]
                    or row["target_min100_additions"] + row["target_small_additions"] != n
                    or any(row[key] < 0 for key in ("added_TP50", "added_FP50",
                        "ambiguous_added_TP50", "ambiguous_added_FP50"))
                    or (arm == "NONE" and any(row[key] for key in ("source_available_additions", "added_TP50", "added_FP50")))):
                raise ValueError("recovery source coverage or actual matching counts violate the shared registry")
        counts = {key: sum(row[key] for row in rows) for key in ("source_available_additions",
            "target_min100_additions", "target_small_additions", "added_TP50", "added_FP50",
            "ambiguous_added_TP50", "ambiguous_added_FP50")}
        n = counts["source_available_additions"] if not missing else None
        arms[arm] = {"method": method, "n": n, "N": denominator,
            "status": "BLOCKED_TECHNICAL" if missing else "COMPLETE",
            "recovered_fraction": n / denominator if n is not None and denominator else None,
            **{key: value if not missing else None for key, value in counts.items() if key != "source_available_additions"},
            "matching_ambiguity_flag": None if missing else any(row["ambiguous_added_TP50"] or row["ambiguous_added_FP50"] for row in rows),
            "complete_scene_count": len(measured_scenes), "blocked_scenes": list(missing), "blocked_conditions": missing,
            "known_measured_counts": counts, "known_measured_scene_order": measured_scenes,
            "unavailable_reason": "FULL_COHORT_DIAGNOSTIC_HAS_BLOCKED_CONDITIONS" if missing else None,
            "scene_order": list(expected), "scene_receipt_identities": [by_scene[scene]["identity"] for scene in expected]}
    result = {"status": PARTIAL_RESULT if any(row["blocked_scenes"] for row in arms.values()) else "COMPLETE",
        "cohort": "replica8", "scene_count": len(expected), "arms": arms,
        "common_candidate_denominator": denominator, "aggregation": "SUM_COUNTS_NOT_MEAN_SCENE_RATIOS",
        "matching_unit": "RELEASED_SCORE_ENTRIES_NOT_UNIQUE_OBJECTS", "ambiguous_ties_arbitrarily_attributed": False}
    result["identity"] = canonical_digest(result)
    return result


def scene_diagnostics(binding, scene):
    require_frozen_execution(binding, scene)
    root = Path(binding["output_root"])
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    lock_path, evaluation_path = (root / directory / scene / "receipt.json" for directory in ("predictions", "evaluation"))
    documents = []
    for path in (lock_path, evaluation_path, root / "projected_views" / scene / "receipt.json",
                 root / "recovery" / scene / "receipt.json", root / "native_recovery" / scene / "receipt.json"):
        index.identity(path)
        value = read(path)
        _verified_identity(value)
        for item in value["outputs"]:
            index.identity(item["path"], item)
        documents.append(value)
    lock, evaluation, projected, fc, native = documents
    spec = load_spec(binding["spec"])
    partial = lock["status"] == "PARTIAL_PREDICTIONS_LOCKED"
    if partial:
        if (validate_partial_scope(spec, lock) != validate_partial_scope(spec, evaluation)
                or lock["blocked_methods"] != evaluation["blocked_methods"]):
            raise ValueError("post-lock diagnostic partial scope changed its actual scoring conditions")
        index.identity(projected["manifest"])
        manifest = read(projected["manifest"])
        for name, block in lock["blocked_sources"].items():
            if name not in ("G1_FC", "G3_FC") or semantic_block(fc, manifest, name[:2]) != block:
                raise ValueError("partial diagnosis changed its actual selected-mask technical block")
        independent_path = root / "recovery" / (scene + "_U2") / "receipt.json"
        index.identity(independent_path)
        independent = read(independent_path)
        _verified_identity(independent)
        if (independent["status"] != "COMPLETE" or independent["scene"] != scene
                or independent["arms"] != ["U2"] or independent["cold"] is not False):
            raise ValueError("partial diagnosis lacks its actual independent U2 source")
        for item in independent["inputs"] + independent["outputs"]:
            index.identity(item["path"], item)
        documents.append(independent)
    elif lock["status"] != "PREDICTIONS_LOCKED" or evaluation["status"] != "COMPLETE" or fc["status"] != "COMPLETE":
        raise ValueError("diagnostics require completed or proven blocked same-scene evidence")
    if (evaluation["prediction_lock_identity"] != lock["identity"]
            or projected["status"] != "COMPLETE" or native["status"] != "COMPLETE"):
        raise ValueError("diagnostics require all same-scene outputs locked and officially scored")
    registry = read(projected["registry"])
    _verified_identity(registry)
    candidates = {row["raw_owner"] for row in registry["candidates"]}
    input_identity = canonical_digest({"binding": binding["identity"], "scene": scene,
        "parents": [row["identity"] for row in documents], "registry": registry["identity"],
        "producer": index.identity(__file__), "released_attribution": index.identity(Path(__file__).parents[1] / "recovery_wave2/mechanisms.py")})
    output_path = root / "diagnostics" / (scene + ".json")
    if output_path.is_file():
        old = read(output_path)
        _verified_identity(old)
        if old["input_identity"] != input_identity:
            raise ValueError("completed post-lock diagnosis changed; invalidate explicitly")
        for item in old["inputs"]:
            index.identity(item["path"], item)
        return old
    rows = {row["method"]: row for row in evaluation["rows"]}
    baseline_row = rows["CT_A1_E"]
    baseline = load_prediction(lock["predictions"]["CT_A1_E"])
    incumbent_labels = owner_labels(baseline)
    condition_sources = {}
    for arm, item in (independent if partial else fc)["sources"].items():
        index.identity(item["path"], item)
        source = read(item["path"])
        _verified_identity(source)
        condition_sources[{"G1": "G1_FC", "G3": "G3_FC", "U2": "ARCHIVED_U2_FC"}[arm]] = source
    condition_sources["G1_NATIVE"] = read(native["source"]["path"])
    index.identity(native["source"]["path"], native["source"])
    _verified_identity(condition_sources["G1_NATIVE"])
    conditions = {}
    for method, row in rows.items():
        payload = load_prediction(lock["predictions"][method])
        check = evaluation["registry_checks"][method]
        added = {int(owner) for owner in payload.metadata["recovered_labels"]}
        recipe = payload.metadata["method_recipe"]
        source = condition_sources.get(recipe["recovery"])
        source_available = {int(owner) for owner, value in source["objects"].items() if value["available"]} if source else set()
        if added != source_available or not added <= candidates:
            raise ValueError("source-available owners differ from the actual appended locked output")
        eligible = set(check["recovered_scoring_eligible"])
        target_small = set(check["recovered_target_small"])
        if eligible | target_small != added or eligible & target_small:
            raise ValueError("source-success and target-min100 recovery coverage disagree")
        labels = owner_labels(payload)
        changed = [owner for owner, label in incumbent_labels.items() if labels[owner] != label]
        if not np.array_equal(payload.owner_ids[baseline.owner_ids > 0], baseline.owner_ids[baseline.owner_ids > 0]):
            raise ValueError("post-lock evidence found replaced incumbent instance support")
        comparison = analyze_scoring_comparison(baseline_row["evaluation_receipt"], row["evaluation_receipt"], added, index=index)
        conditions[method] = {"method": method, "recovery": recipe["recovery"], "source_available_additions": len(added),
            "source_available_owners": sorted(added), "target_min100_additions": len(eligible),
            "target_small_additions": len(target_small), "target_min100_owners": sorted(eligible),
            "target_small_owners": sorted(target_small), "old_incumbent_label_changes": changed,
            "old_instance_support_exact": True, "unavailable_by_reason": dict(Counter(
                value["reason"] for value in source["objects"].values() if not value["available"])) if source else {},
            "used_request_ids": {owner: value["used_request_ids"] for owner, value in source["objects"].items()
                                 if value["available"]} if source else {},
            "evaluation_identity": row["evaluation_identity"], **comparison}
    result = {"status": PARTIAL_RESULT if partial else "COMPLETE", "scene": scene, "input_identity": input_identity,
        "candidate_count": len(candidates), "registry_identity": registry["identity"], "conditions": conditions,
        "candidates_chosen_before_GT": True, "GT_used_only_after_prediction_lock": True,
        "prediction_lock_identity": lock["identity"], "evaluation_receipt_identity": evaluation["identity"],
        "source_min100_is_not_target_min100": True,
        "same_view_common_success": common_success_comparison(condition_sources.get("G1_FC"),
            condition_sources["G1_NATIVE"], technical_block=lock.get("blocked_sources", {}).get("G1_FC")),
        "inputs": index.entries()}
    if partial:
        result.update({key: lock[key] for key in ("expected_method_ids", "available_recovery_sources",
                                                  "blocked_sources", "blocked_methods")})
    result["identity"] = canonical_digest(result)
    atomic_write_json(output_path, result)
    index.write_memo(root / "validation/input_verifications.json")
    return result


def run_diagnostics(binding):
    spec = load_spec(binding["spec"])
    scenes = [*spec["cohorts"]["replica8"], *spec["cohorts"]["scannet_cf18"]]
    receipts = {scene: scene_diagnostics(binding, scene) for scene in scenes}
    pooled = aggregate_recovery_diagnostics(spec, [receipts[scene] for scene in spec["cohorts"]["replica8"]])
    root = Path(binding["output_root"])
    atomic_write_json(root / "diagnostics/replica8_recovery.json", pooled)
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    result = {"status": PARTIAL_RESULT if any(row["status"] == PARTIAL_RESULT for row in receipts.values()) else "COMPLETE",
        "main_scene_count": len(receipts),
        "scene_order": scenes, "scene_identities": {scene: row["identity"] for scene, row in receipts.items()},
        "recovery_pool_identity": pooled["identity"], "outputs": [index.identity(root / "diagnostics" / (scene + ".json"))
            for scene in scenes] + [index.identity(root / "diagnostics/replica8_recovery.json")]}
    result["identity"] = canonical_digest(result)
    atomic_write_json(root / "diagnostics/receipt.json", result)
    return result
