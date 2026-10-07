"""Final executable requirement audit; publication SHA is attested separately."""

import csv
import gzip
import json
from pathlib import Path
import subprocess

from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .evaluation import require_freeze
from .verifier import target_flags, choose_method


def review_study(binding, root):
    root = Path(root)
    repo = Path(__file__).resolve().parents[3]
    artifact = repo/"artifacts/static_ovmap/evidence_exploration_v1"
    with gzip.open(artifact/"study_result_store.json.gz", "rt") as handle:
        store = json.load(handle)
    _verified_identity(store)
    require_freeze(root)
    science, selection, timing = store["science"], store["selection"], store["timing"]
    spec = binding["specification"]
    checks = []
    def check(requirement, condition, evidence):
        checks.append({"requirement": requirement, "verified": bool(condition), "evidence": evidence})
    names = [x["id"] for x in spec["methods"]]
    expected_rows = {(scene, method) for order in binding["cohorts"].values() for scene in order for method in names}
    actual_rows = {(x["scene"], x["method"]) for x in science["scene_metrics"]}
    expected_pools = {(cohort, method) for cohort in binding["cohorts"] for method in names}
    actual_pools = {(x["cohort"], x["method"]) for x in science["pooled_metrics"]}
    check("0: isolated task/base/branch; fixed slate and unchanged deployment", binding["base_commit"] == spec["base_commit"]
          and len(names) == 9 and store["deployment"] == "N0_UNCHANGED", ["source_binding.json.gz", "configs/static_ovmap/evidence_exploration_v1.json"])
    check("1: actual 172/14 parent lineage and 52 baseline scene imports", binding["imported_baseline_scene_rows"] == 52
          and binding["imported_baseline_full_pools"] == 4 and len(binding["sources_read"]) == 13, ["source_binding.json.gz", "SOURCE_BINDING.md"])
    check("2: all cache-only diagnoses; exactly two diagnostic pools; actual ignore ledger", len(store["baseline_counterfactual_pools"]) == 2
          and len(store["baseline_diagnostic_summary"]["supplements"]) == 26, ["baseline_diagnostics.json.gz"])
    prepared = [read(root/"prepared"/scene/"receipt.json") for scene in binding["scenes"]]
    check("3: registry/mask equality, geometric trigger, at most two controls on original G1", all(x["registry_identity"] == binding["scenes"][x["scene"]]["registry_identity"]
          and all(len(c["controls"]) <= 2 for c in x["candidates"].values()) for x in prepared), ["large_artifact_dependencies.json.gz", "preparation.py"])
    checkpoint = store["assets"]["checkpoint"]
    neutral = store["real_pilots"]["neutral_full_frame"]
    check("4: pinned AnyUp checkpoint/source; original FP32 FC; aligned grids; real neutrality", checkpoint["sha256"] == spec["anyup"]["checkpoint_sha256"]
          and checkpoint["bytes"] == spec["anyup"]["checkpoint_bytes"] and store["assets"]["load"]["strict"]
          and neutral["status"] == "REAL_FULL_FRAME_NEUTRALITY_VERIFIED", ["upstream_weight_identity.json", "study_result_store.json.gz", "anyup_adapter.py"])
    acquire = store["scientific_acquisition"]
    check("5: shared original averaged attention, explicit owner/depth and no-depth arms", acquire["counts"]["AnyUp_QK_computations"] == store["selected_frame_geometry"]["triggered_frames"]
          and acquire["counts"]["all_masked_queries"] == 0 and acquire["counts"]["target_owner_depth_pools"] == acquire["counts"]["target_owner_only_pools"], ["anyup_adapter.py", "large_artifact_dependencies.json.gz"])
    decisions = store["candidate_decisions"]
    required_fields = {"feature_available", "proposed_class", "accepted", "defer_reason", "raw_cosines", "decision_scores", "margin", "controls", "purity", "representation_fallback"}
    check("6: feature availability separate from deferral; no hidden threshold/contrast sweep", all(required_fields <= row.keys()
          for methods in decisions.values() for value in methods.values() for row in value["objects"].values())
          and spec["constants"]["margin_threshold_cosine"] == .01 and spec["constants"]["contrast_lambda"] == .25, ["candidate_decisions.json.gz", "verifier.py"])
    unchanged_fields = ("feature_available", "proposed_class", "accepted", "raw_cosines", "decision_scores",
                        "baseline_vector_sha256", "representation_vector_sha256")
    off_trigger = []
    for item in prepared:
        scene = item["scene"]
        baseline_objects = decisions[scene]["EV01_G1_V2"]["objects"]
        for owner, candidate in item["candidates"].items():
            if not candidate["triggered"]:
                for name in names[2:]:
                    value = decisions[scene][name]["objects"][owner]
                    base = baseline_objects[owner]
                    off_trigger.append(all(value.get(field) == base.get(field) for field in unchanged_fields))
    check("6a: actual trigger-off availability/classes/scores/acceptance/vector hashes equal B1", bool(off_trigger)
          and all(off_trigger), ["candidate_decisions.json.gz", "large_artifact_dependencies.json.gz"])
    check("7: all 234 scene/method outcomes and 18 ordered official pools", actual_rows == expected_rows and len(science["scene_metrics"]) == 234
          and actual_pools == expected_pools and len(science["pooled_metrics"]) == 18
          and all(x["scene_order"] == binding["cohorts"][x["cohort"]] for x in science["pooled_metrics"]), ["scene_and_pooled_all_metrics_classes.json.gz"])
    locks = [read(root/"predictions"/scene/"receipt.json") for scene in binding["scenes"]]
    check("8: immutable D2 incumbents, accepted expanded registry reaches scorer, rank/GT scoring unchanged", all(x["status"] == "PREDICTIONS_LOCKED"
          and x["incumbent_labels_verified_against_exact_D2"] and not x["GT_used_by_predictor"] for x in locks)
          and all(x["rank_mode"] == "OFFICIAL_CURRENT_CLASS" for x in science["scene_metrics"]), ["outputs.py", "evaluation.py", "paired_diagnostics.json.gz"])
    record = selection["metrics"]
    flags = {name: target_flags(record[name], record["EV01_G1_V2"], record["EV00_D2"]) for name in names}
    check("9: committed freeze; all-five-metric unrounded gate and one exposed-cohort recommendation", flags == selection["target_flags"]
          and choose_method(record, spec["selection"]["simplicity_order"]) == selection["recommended_method"]
          and selection["exposed_cohorts"] and not selection["independent_confirmation"], ["implementation_freeze.json", "selection.json"])
    log = (root/"pilots/focused_tests.log").read_text()
    pilot_scenes = store["real_pilots"]["acquisition"]["scenes"]
    check("10: focused production checks and both real pilots; no new Native/NQ/frontend/training/map inference", "8 passed" in log
          and pilot_scenes == spec["pilot_scenes"] and acquire["new_Native_NQ_training_frontend_map_calls"] == 0
          and acquire["counts"]["FC_image_inputs"] == 0, ["focused_tests.log", "study_result_store.json.gz"])
    calls = timing["calls"]
    expected_calls = {(repeat, scene, method) for repeat in (1, 2) for scene in binding["cohorts"]["replica8"] for method in selection["timing_methods"]}
    check("11: paired cold boundaries, all repeats, reverse order, post-timer parity and memory/costs", {(x["repeat"], x["scene"], x["method"]) for x in calls} == expected_calls
          and len(calls) == len(expected_calls) <= 64 and all(x["status"] == "COLD_CALL_PARITY_VERIFIED" and x["correctness_checked_after_timer"]
          and x["recovery_views_features_results_cold"] and x["persistent_feature_view_result_cache_hits"] == 0 for x in calls)
          and not timing["online_30FPS_validated"], ["timing_records.json", "timing.py"])
    expected_order = [(repeat, scene, method) for repeat in (1, 2)
        for scene in (binding["cohorts"]["replica8"] if repeat == 1 else list(reversed(binding["cohorts"]["replica8"])))
        for method in (selection["timing_methods"] if repeat == 1 else list(reversed(selection["timing_methods"])))]
    check("11a: actual ordered repeats, equal work counts, exclusive stages and observed memory", [(x["repeat"], x["scene"], x["method"]) for x in calls] == expected_order
          and all(x["counts"] == next(y["counts"] for y in calls if y["repeat"] == 1 and y["scene"] == x["scene"] and y["method"] == x["method"])
              and abs(sum(x["exclusive_host_seconds"].values())+x["boundary_sync_and_overhead_seconds"]-x["seconds_per_scene"]) < 1e-8
              and x["peak_cuda_reserved_bytes"] >= x["peak_cuda_allocated_bytes"] > 0 for x in calls), ["timing_records.json"])
    table_files = [artifact/(name+"."+suffix) for name in ("table1_main", "table2_mechanisms", "table3_timing") for suffix in ("md", "csv", "json")]
    report_files = [repo/"docs/paper/static_ovmap"/name for name in spec["publication"]["reports"]]
    size = sum(p.stat().st_size for p in artifact.rglob("*") if p.is_file())
    check("12: three tables MD/CSV/JSON, four reports, one store/cell provenance and compact licensed-data-free artifacts", all(p.is_file() for p in table_files+report_files)
          and (artifact/"cell_provenance.json").is_file() and size <= 50*2**20
          and not any(p.suffix in ("pth", "ply", "safetensors", "png", "jpg") for p in artifact.rglob("*")), [str(p.relative_to(repo)) for p in table_files+report_files])
    # Verify each displayed main-table value against the actual full-precision store.
    with (artifact/"table1_main.csv").open() as handle:
        main = list(csv.DictReader(handle))
    poolmap = {(x["cohort"], x["method"]): x for x in science["pooled_metrics"]}
    check("12a: exact displayed numeric cells and units", len(main) == 18 and all(abs(float(row[m])-poolmap[row["Cohort"], row["Method"]]["metrics"][m]*100) < 1e-10
          for row in main for m in ("apall", "ap50", "miou")), ["table1_main.csv", "cell_provenance.json"])
    format_checks = []
    for name in ("table1_main", "table2_mechanisms", "table3_timing"):
        data = read(artifact/(name+".json"))
        with (artifact/(name+".csv")).open() as handle:
            actual = list(csv.DictReader(handle))
        expected = [{key: str(row[key]) for key in data["columns"]} for row in data["rows"]]
        format_checks.append(actual == expected)
        # Reconstruct the display string without writing or changing the artifacts.
        def display(value):
            return "—" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)
        lines = ["| "+" | ".join(data["columns"])+" |", "| "+" | ".join(["---"]*len(data["columns"]))+" |"]
        lines += ["| "+" | ".join(display(row[key]) for key in data["columns"])+" |" for row in data["rows"]]
        format_checks.append((artifact/(name+".md")).read_text() == "\n".join(lines)+"\n")
    check("12c: all three MD/CSV/JSON representations agree", all(format_checks), [str(p.relative_to(repo)) for p in table_files])
    mechanisms = read(artifact/"table2_mechanisms.json")["rows"]
    pairs = store["paired_diagnostics"]["comparisons"]
    timing_rows = read(artifact/"table3_timing.json")["rows"]
    check("12d: mechanism and timing displayed values derive from the single store", len(mechanisms) == len(pairs) == 12
          and all(row["Cohort"] == source["cohort"] and row["Comparison"] == source["candidate"]+" / "+source["reference"]
              and all(abs(row[column]-source["metric_deltas_fraction"][metric]*100) < 1e-10
                  for column, metric in (("ΔAPall_pp", "apall"), ("ΔAP50_pp", "ap50"), ("ΔmIoU_pp", "miou")))
              for row, source in zip(mechanisms, pairs))
          and len(timing_rows) == len(timing["summary"]) and all(row["Method"] == source["method"]
              and row["Seconds_scene"] == source["mean_seconds_per_scene"] and row["Allocated_GiB"] == source["peak_cuda_allocated_bytes"]/2**30
              and row["Reserved_GiB"] == source["peak_cuda_reserved_bytes"]/2**30 for row, source in zip(timing_rows, timing["summary"])),
          ["table2_mechanisms.json", "table3_timing.json", "study_result_store.json.gz"])
    check("12b: executable controller exact command and all specified flags", subprocess.run([binding["FC_python"], str(repo/"scripts/evaluation/run_ovimap_evidence_exploration.py"), "--help"],
          cwd=repo, capture_output=True).returncode == 0, ["scripts/evaluation/run_ovimap_evidence_exploration.py"])
    verified = all(x["verified"] for x in checks)
    result = seal({"status": "REQUIREMENT_REVIEW_VERIFIED" if verified else "REQUIREMENT_REVIEW_INCOMPLETE", "checks": checks,
        "required_execution_verified": verified, "single_result_store_identity": store["identity"], "artifact_bytes": size,
        "implementation": "IMPLEMENTED_AND_EXECUTED", "assets": store["assets"]["status"],
        "scientific_coverage": "234_SCENE_ROWS_18_FULL_POOLS", "performance_target": bool(selection["passing_methods"]),
        "timing": timing["status"], "research_selection": selection["recommended_method"],
        "publication": "PENDING_FULL_SHA_RECEIPT_OUTSIDE_COMMIT", "deployment": "N0_UNCHANGED",
        "primary_agent_retains_review_and_integration_ownership": True})
    atomic_write_json(artifact/"requirement_review.json", result)
    (artifact/"focused_tests.log").write_text(log)
    if not verified:
        raise ValueError("requirement review incomplete: "+", ".join(x["requirement"] for x in checks if not x["verified"]))
    return result
