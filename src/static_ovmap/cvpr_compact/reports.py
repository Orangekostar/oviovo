"""Evidence-bound paired analysis and the four required compact-table reports."""

from pathlib import Path

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .diagnostics import TABLE3_METHODS
from .evaluation import METRICS
from .partial_execution import PARTIAL_RESULT
from .projected_views import _verified_identity
from .protocol import experiment_matrix, load_spec
from .recovery_run import ARM_SOURCE
from .runtime import require_frozen_execution
from .tables import _evidence


CONDITIONS = ("CT_A0_NATIVE", "CT_A1_E", "CT_A2_R", "CT_A3_ER", "CT_A4_NATIVE_RECOVERY", "CT_A5_FC_ONLY")


def _delta(first, second):
    return {metric: None if first[metric] is None or second[metric] is None
            else 100 * (first[metric] - second[metric]) for metric in METRICS}


def _dominates(first, second, metrics):
    return (all(first[m] is not None and second[m] is not None for m in metrics)
            and all(first[m] >= second[m] for m in metrics) and any(first[m] > second[m] for m in metrics))


def _difference(first, second):
    return None if first is None or second is None else first - second


def analyze_results(spec, store, diagnosis, timing):
    for value in (store, diagnosis, timing):
        _verified_identity(value)
    matrix = experiment_matrix(spec)
    if (store["status"] not in ("COMPLETE", PARTIAL_RESULT) or store["matrix_identity"] != matrix["identity"]
            or len(store["scene_metrics"]) != 172 or len(store["pooled_metrics"]) != 14
            or diagnosis["status"] not in ("COMPLETE", PARTIAL_RESULT)
            or timing["status"] not in ("COMPLETE", PARTIAL_RESULT)
            or timing.get("planned_leaf_count", timing["leaf_count"]) != 24
            or timing["leaf_count"] + timing.get("blocked_leaf_count", 0) != 24):
        raise ValueError("paired reporting requires all fixed positions with measured or proved blocked evidence")
    pools = {(r["cohort"], r["method_id"]): r for r in store["pooled_metrics"]}
    expected = {(r["cohort"], r["method"]) for r in matrix["pools"]}
    if set(pools) != expected or len(pools) != len(store["pooled_metrics"]):
        raise ValueError("paired reporting changed its exact complete14-pool matrix")
    cohorts = {}
    for cohort in spec["cohorts"]:
        values = {method: pools[(cohort, method)]["metrics"] for method in CONDITIONS}
        a0, a1, a2, a3 = (values[method] for method in CONDITIONS[:4])
        deltas = {"A3_vs_" + short + "_pp": _delta(a3, values[method])
                  for short, method in (("A0", CONDITIONS[0]), ("A1", CONDITIONS[1]),
                                        ("A4", CONDITIONS[4]), ("A5", CONDITIONS[5]))}
        recovery = deltas["A3_vs_A1_pp"]
        flags = {"APall_positive": None if recovery["apall"] is None else recovery["apall"] > 0,
            "AP50_drop_at_most_0.10pp": None if recovery["ap50"] is None else recovery["ap50"] >= -.10,
            "mIoU_drop_at_most_0.10pp": None if recovery["miou"] is None else recovery["miou"] >= -.10}
        flags["all_preferences_met"] = None if any(value is None for value in flags.values()) else all(flags.values())
        dominators = {method: [other for other in CONDITIONS if other != method
            and _dominates(values[other], values[method], ("apall", "ap50", "miou"))] for method in CONDITIONS}
        unavailable = [method for method in CONDITIONS
                       if any(values[method][m] is None for m in ("apall", "ap50", "miou"))]
        cohorts[cohort] = {**deltas, "raw_metrics": values,
            "E_only_pp": _delta(a1, a0), "R_only_pp": _delta(a2, a0),
            "E_given_R_pp": _delta(a3, a2), "R_given_E_pp": recovery,
            "interaction_pp": {metric: None if any(row[metric] is None for row in (a0, a1, a2, a3))
                               else 100 * (a3[metric] - a2[metric] - a1[metric] + a0[metric]) for metric in METRICS},
            "guardrails_vs_A1": flags, "metric_pareto_frontier": [method for method in CONDITIONS
                if not dominators[method] and all(values[method][m] is not None for m in ("apall", "ap50", "miou"))],
            "metric_pareto_unavailable": unavailable, "metric_pareto_scope_complete": not unavailable,
            "A3_dominated_by": None if "CT_A3_ER" in unavailable else dominators["CT_A3_ER"],
            "pareto_metrics": ["apall", "ap50", "miou"],
            "pool_identities": {method: pools[(cohort, method)]["receipt_identity"] for method in CONDITIONS}}
    recovery = {}
    for later, earlier in (("G1", "U2"), ("G3", "G1")):
        first, second = diagnosis["arms"][later], diagnosis["arms"][earlier]
        if first["N"] != second["N"]:
            raise ValueError("recovery comparison changed the shared candidate denominator")
        recovery[later + "_vs_" + earlier] = {
            "source_available_delta": _difference(first["n"], second["n"]), "shared_candidate_N": first["N"],
            "added_TP50_entry_delta": _difference(first["added_TP50"], second["added_TP50"]),
            "added_FP50_entry_delta": _difference(first["added_FP50"], second["added_FP50"]),
            "matching_attribution_ambiguous": None if any(row["matching_ambiguity_flag"] is None for row in (first, second))
                else first["matching_ambiguity_flag"] or second["matching_ambiguity_flag"],
            "metric_deltas_pp": _delta(pools[("replica8", TABLE3_METHODS[later])]["metrics"],
                                      pools[("replica8", TABLE3_METHODS[earlier])]["metrics"]),
            "mean_seconds_delta": _difference(timing["arms"][ARM_SOURCE[later]]["mean_seconds"],
                                              timing["arms"][ARM_SOURCE[earlier]]["mean_seconds"]),
            "time_definition": "CONDITIONAL_INCREMENTAL_RECOVERY_MODEL_RESIDENT",
            "coverage_is_not_accuracy": True}
    result = {"status": PARTIAL_RESULT if any(row["status"] == PARTIAL_RESULT for row in (store, diagnosis, timing)) else "COMPLETE",
        "primary_method": spec["selection"]["primary_method"],
        "main_scene_outputs": store["main_scene_outputs"], "internal_pools": store["internal_pools"],
        "cold_timing_leaves": {"complete": timing["leaf_count"], "blocked": timing.get("blocked_leaf_count", 0), "required": 24},
        "deployment": "N0_UNCHANGED", "cohorts": cohorts, "recovery": recovery,
        "result_store_identity": store["identity"], "diagnosis_identity": diagnosis["identity"],
        "timing_identity": timing["identity"], "external_values_used_in_paired_analysis": False,
        "guardrails_are_descriptive_not_significance": True, "automatic_method_reselection": False}
    result["identity"] = canonical_digest(result)
    return result


def _number(value, *, percent=False, signed=False):
    if value is None:
        return "--"
    return format(value * (100 if percent else 1), "+.2f" if signed else ".2f")


def _markdown(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]]) + "\n"


def _count(value):
    return "--" if value is None else str(value)


def render_reports(binding, store, diagnosis, timing, reference, analysis, freeze_revision, *, cost_data=None,
                   scene_diagnostics=None):
    metrics = _markdown(["Cohort", "Method", "APall (%)", "AP50 (%)", "AP25 (%)", "mIoU (%)", "mAcc (%)"],
        [[r["cohort"], r["method_id"], *[_number(r["metrics"][metric], percent=True) for metric in METRICS]]
         for r in store["pooled_metrics"]])
    delta_rows = [[cohort, contrast, *[_number(values[metric], signed=True) for metric in METRICS]]
        for cohort, result in analysis["cohorts"].items() for contrast, values in result.items()
        if contrast.endswith("_pp")]
    deltas = _markdown(["Cohort", "Contrast (pp)", *METRICS], delta_rows)
    coverage = _markdown(["Arm", "Source-available n/N", "Added TP50", "Added FP50", "Ambiguous TP/FP", "Seconds/scene"],
        [[arm, "--" if row["n"] is None else f'{row["n"]}/{row["N"]}',
          _count(row["added_TP50"]), _count(row["added_FP50"]),
          "--" if row["ambiguous_added_TP50"] is None else f'{row["ambiguous_added_TP50"]}/{row["ambiguous_added_FP50"]}',
          "0 (by definition)" if arm == "NONE" else _number(timing["arms"][ARM_SOURCE[arm]]["mean_seconds"])]
         for arm, row in diagnosis["arms"].items()])
    flags = _markdown(["Cohort", "APall > A1", "AP50 drop <= 0.10pp", "mIoU drop <= 0.10pp", "A3 dominated by"],
        [[cohort, *[_count(result["guardrails_vs_A1"][name]) for name in
            ("APall_positive", "AP50_drop_at_most_0.10pp", "mIoU_drop_at_most_0.10pp")],
          "--" if result["A3_dominated_by"] is None else ", ".join(result["A3_dominated_by"]) or "none"]
         for cohort, result in analysis["cohorts"].items()])
    external = (f'External rows: [OVI-MAP final PDF, Table 3]({reference["source_url"]}), '
        f'also checked against [arXiv v1 PDF](https://arxiv.org/pdf/2603.26541v1). '
        f'{len(reference["differences_from_provided_transcription"])} differences from the supplied HTML transcription '
        'are preserved in the provenance audit. These are author-reported context, not rerun systems; '
        'their raw scorer, rank protocol and ScanNet sequence membership are not independently verified.\n')
    root, repository = binding["output_root"], binding["repository_root"]
    outputs, pools, cold = (analysis[key] for key in ("main_scene_outputs", "internal_pools", "cold_timing_leaves"))
    status = (f'Implementation: FROZEN. Internal scientific coverage: {store["status"]} '
        f'({outputs["complete"]}/{outputs["required"]} outputs, {pools["complete"]}/{pools["required"]} ordered pools). '
        f'Timing: {timing["status"]} ({cold["complete"]}/{cold["required"]} actual calls with scientific parity; '
        f'{cold["blocked"]} proved unmeasured leaves). External protocol: AUTHOR_PROTOCOL_AS_REPORTED, '
        "NOT_INDEPENDENTLY_VERIFIED. Performance: see raw deltas and descriptive flags below. "
        "Publication: PENDING_PUSH_VERIFICATION; the final external publication receipt is authoritative.\n")
    if store["status"] == PARTIAL_RESULT:
        status += ("Scene office1 A2/A3/A5/G3 are blocked because every preselected FC mask has empty support at the "
            "original dense resolution. Their full Replica pools and dependent counts/times are unavailable (--), "
            "not zero. All eight scene positions and the shared candidate denominator remain fixed. "
            "No seven-scene pool or mean replaces the required complete cohort.\n")
    setup = ("The common geometry is BB00_NATIVE with the pinned original extension, 0.01m voxels, association4, "
        "CropFormer single-label masks and exactly200 original schedule slots without refill. "
        "The paper text states0.1m voxels; this reproduction follows the pinned released-code0.01m configuration. "
        "Whole predicted meshes use exact Open3D FP32 1NN and strict squared distance<0.05 squared. "
        "Positive projected owners need100 target points for instance export; confidence is current-class area "
        "divided by the maximum same-class area, written to six decimals. Semantic unknown0 errors remain counted. "
        "APall retains the actual original .50--.90 overlap vector, plus separate AP25. "
        "Full valid-ID orders, individual scenes/classes, lost old matches, rank changes and failures are in the SI.\n")
    limits = ("Replica is historically exposed. CF18 is18 captures from7 physical ScanNet families, not18 "
        "independent rooms or the full ScanNet200 validation set. Known Q fit/calibration families do not overlap "
        "CF18, but unrecorded benchmark exposure is unverified. Q and the calibrated N/Q/F scalars are fitted "
        "components, frozen for this experiment; no new fit was performed. No untouched-generalization, "
        "all-components-training-free, statistical-significance, global SOTA, real-time or end-to-end FPS claim follows.\n")
    results = ("# Compact Tables Results\n\n" + status + "\n" + metrics + "\n## Paired Effects\n\n" + deltas
        + "\nThese are raw differences in percentage points. A3 remains the fixed full method even when a control wins. "
        "The E x R interaction is A3-A2-A1+A0; undefined values remain --.\n\n" + flags
        + "\nFlags are engineering preferences, not significance tests or selection gates. A listed dominator is a "
        "simple control with no lower APall/AP50/mIoU and at least one strictly higher metric.\n\n"
        + "## Recovery Evidence\n\n" + coverage
        + "\nTP/FP count released matcher score entries, not unique objects. Ambiguous old/new score ties are separate. "
        "n/N includes source-available target-small owners and never implies true-positive coverage. "
        "Available cold means require all eight actual scene calls, including real empty-view overhead. Model loading, mapping "
        "and evaluation are excluded and reported separately.\n\n"
        + f'Model loading total: {_number(timing["model_loading_seconds_total"])} seconds, excluded from all means. '
        f'GPU: `{timing["hardware"]["name"]}`, compute capability `{timing["hardware"]["compute_capability"]}`, '
        f'physical UUID `{timing["hardware"]["uuid"]}`.\n\n'
        + _markdown(["Comparison", "Delta n", "Delta TP50", "Delta FP50", "Delta APall (pp)", "Delta mIoU (pp)", "Delta seconds"],
            [[name, _count(row["source_available_delta"]), _count(row["added_TP50_entry_delta"]), _count(row["added_FP50_entry_delta"]),
              _number(row["metric_deltas_pp"]["apall"], signed=True), _number(row["metric_deltas_pp"]["miou"], signed=True),
              _number(row["mean_seconds_delta"], signed=True)] for name, row in analysis["recovery"].items()])
        + "\nG3's additional cost is assessed against its displayed full-map gains and ambiguity, not its recovered count alone.\n\n"
        + "## Protocol And Boundaries\n\n" + setup + "\n" + external + "\n" + limits)
    command = ("/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_cvpr_compact.py "
               "--spec configs/static_ovmap/cvpr_compact_tables_v1.json --phase all --resume")
    handoff = ("# Compact Tables Handoff\n\n" + status + f"\nFreeze commit: `{freeze_revision}`. "
        f'Binding: `{binding["identity"]}`. Worktree: `{repository}`. Large evidence: `{root}`.\n\n'
        "```bash\n" + command + "\n```\n\n"
        "The fixed scope is26 BB00 anchors,172 records,14 official pools and24 timing leaves. "
        "Completed identities and hashes must agree before reuse; live exact workers are waited on, not restarted. "
        "The same leaf permits at most two automatic retries; cold calls are never repeated once reserved. "
        "Keep immutable parents read-only and preserve failed logs and paid costs.\n\n"
        "The compact repository artifact directory is `artifacts/static_ovmap/cvpr_compact_tables_v1/`. "
        "Per-scene/per-class/full five-metric data, scores, decisions, costs, provenance and validation accompany "
        "the three actual9pt tables. Large scans, tensors and masks stay in shared storage with reconstruction commands. "
        "Table preview requires an explicit inspected-PDF QA receipt before publication.\n\n"
        "Normal-push the named branch and compare full local and remote40-character SHAs. "
        "`publication/final.json` stays outside the commit it verifies. Deployment remains N0_UNCHANGED.\n")
    selection = ("# Compact Tables Selection\n\nFixed primary: `CT_A3_ER`. Deployment: `N0_UNCHANGED`. "
        "No retrospective method reselection, parameter sweep or automatic deployment.\n\n" + flags
        + "\n" + _markdown(["Cohort", "Metric Pareto frontier", "Undefined Pareto input"],
            [[cohort, ", ".join(row["metric_pareto_frontier"]), ", ".join(row["metric_pareto_unavailable"]) or "none"]
             for cohort, row in analysis["cohorts"].items()])
        + "\nPareto frontiers use only controls with available APall/AP50/mIoU; missing controls are listed, "
        "and the complete fixed-method frontier cannot be established when any input is unavailable. "
        "An unavailable A3 comparison is --, not evidence that A3 has no dominator. "
        "Any A3 dominator is reported without suppression. A3 versus A4 tests representation under the same "
        "G1 frame/mask; A3 versus A5 tests incumbent refinement using identical F and G1 evidence. "
        "A5's FC-only readout retains Native-derived support and requires that prerequisite in standalone costs.\n")
    claims = ("# Compact Tables Claims\n\n" + status + "\n" + limits + "\n" + external
        + "\nPermitted claims are the measured paired full-map deltas, coverage and actual conditional recovery times. "
        "Negative A3-A0/A1/A4/A5 effects and unfavorable G3 tradeoffs remain visible. "
        "A larger source-available n/N alone proves neither accuracy nor superiority. "
        "No external/internal paired delta, global-best bolding or external speedup is computed.\n\n"
        "A2/A3/A5 share the exact successful G1-FC addition set; A4 uses the same G1 frame/mask with original "
        "Native representation. G3 never replaces a failed preselected view. A5 uses unknown0 without Native "
        "semantic fallback. All-selected-inference failure is a technical block, not valid zero recovery.\n\n"
        "The actual source and producer identities, overlap vector, class orders, budgets and exposure ledger "
        "bound the evidence. Publication status is established only by the external final full-SHA verification receipt.\n")
    if cost_data is not None:
        _verified_identity(cost_data)
        groups = {}
        scene_cohorts = {row["scene"]: row["cohort"] for row in store["scene_metrics"]}
        for row in cost_data["method_dependencies"]:
            cohort = scene_cohorts[row["scene"]]
            groups.setdefault((cohort, row["method_id"]), []).append(row["dependency_costs"])
        rows = []
        for planned in store["pooled_metrics"]:
            cohort, method = planned["cohort"], planned["method_id"]
            values = groups.get((cohort, method), [])
            complete = len(values) == planned["required_coverage"]
            partial = not complete or any(value["status"] != "COMPLETE" for value in values)
            rows.append([cohort, method,
                "--" if not complete else sum(value["native_required_crop_inputs"] for value in values),
                "--" if partial else sum(value["FC_required_image_inputs"] for value in values),
                "--" if partial else sum(value["FC_selected_mask_inputs"] for value in values),
                "--" if not complete else sum(value["failed_request_count"] for value in values),
                f'{len(values)}/{planned["required_coverage"]}'])
        budget_table = _markdown(["Cohort", "Method", "Native crop inputs", "FC image inputs", "FC selected masks",
                                 "Failed requests", "Recorded/required scenes"], rows)
        physical = cost_data["physical_payments"]
        paid_table = _markdown(["Cost origin", "Native crop inputs", "FC image inputs", "CropFormer image inputs", "FC pooling calls"],
            [[bucket, *[_number(physical[bucket][unit]) for unit in ("NATIVE_CROP", "FC_IMAGE", "CROPFORMER_IMAGE")],
              _number(physical[bucket + "_FC_region_poolings"])]
             for bucket in ("current_task_warm", "current_task_cold", "historical_parent")])
        results += ("\n## Dependency And Paid Costs\n\n" + budget_table + "\n" + paid_table + "\n"
            "Complete method budgets sum independent per-scene content unions. Incomplete full-cohort budgets are --; "
            "their known per-scene costs remain in the SI with explicit blocked dependencies. Shared worker payments are counted once, "
            "with historical parent, current acquisition and actual cold replays separate. Selected mask counts "
            "include failed requests and are not physical pooling calls. Native-painted support is a required "
            "A5 prerequisite. These are input budgets, not measured whole-pipeline standalone wall times. "
            f'Physical-cost status: {physical["status"]}. Unknown paid counters are --; known lower bounds, '
            "all source hashes and retained failed receipts are in tables/costs.json.\n")
        handoff += f'\nCost ledger: `{root}/tables/costs.json`; shared physical workers are counted once.\n'
    if scene_diagnostics is not None:
        comparisons = []
        for scene, diagnostic in scene_diagnostics.items():
            _verified_identity(diagnostic)
            common = diagnostic["same_view_common_success"]
            if common["status"] == "BLOCKED_TECHNICAL":
                comparisons.append([scene, "BLOCKED_TECHNICAL", "--", "--", "--", "--"])
            else:
                comparisons.append([scene, "COMPLETE", len(common["common_success_owners"]),
                    common["label_agreement_count"], len(common["FC_only_success_owners"]),
                    len(common["native_only_success_owners"])])
        results += ("\n## Same-View Common-Success Diagnostic\n\n"
            + _markdown(["Scene", "Status", "Both successful", "Labels agree", "FC-only successful", "Native-only successful"], comparisons)
            + "\nThis post-lock A3/A4 diagnostic uses the same selected G1 frame/mask and saved scores only. "
            "Individual scores and class orders are retained in diagnostics/<scene>.json. FC and Native scores "
            "belong to different encoders and are not numerically comparable. Label agreement is not accuracy; "
            "success-set differences also affect the full-map A3/A4 comparison. No new inference or benchmark row was added.\n")
    return dict(zip(("COMPACT_TABLES_RESULTS.md", "COMPACT_TABLES_HANDOFF.md", "COMPACT_TABLES_SELECTION.md",
                     "COMPACT_TABLES_CLAIMS.md"), (results, handoff, selection, claims), strict=True))


def write_reports(binding):
    spec, root = load_spec(binding["spec"]), Path(binding["output_root"])
    freeze_revision = require_frozen_execution(binding, spec["cohorts"]["replica8"][0])
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    table_receipt = _evidence(root / "tables/receipt.json", index)
    diagnosis = _evidence(root / "diagnostics/replica8_recovery.json", index)
    timing = _evidence(root / "timing/pool.json", index)
    store = _evidence(root / "tables/result_store.json", index)
    costs = _evidence(root / "tables/costs.json", index)
    diagnostics = {row["scene"]: _evidence(root / "diagnostics" / (row["scene"] + ".json"), index)
                   for row in experiment_matrix(spec)["anchors"]}
    reference_path = root / "external/reference.json"
    index.identity(reference_path)
    reference = read(reference_path)
    _verified_identity(reference)
    if table_receipt["result_store_identity"] != store["identity"]:
        raise ValueError("reports differ from the actual generated table result store")
    analysis = analyze_results(spec, store, diagnosis, timing)
    atomic_write_json(root / "reports/analysis.json", analysis)
    directory = Path(binding["repository_root"]) / "docs/paper/static_ovmap"
    outputs = [index.identity(root / "reports/analysis.json")]
    for filename, text in render_reports(binding, store, diagnosis, timing, reference, analysis, freeze_revision,
                                         cost_data=costs, scene_diagnostics=diagnostics).items():
        path = directory / filename
        path.write_text(text)
        outputs.append(index.identity(path))
    result = {"status": analysis["status"], "artifact_generation_status": "COMPLETE",
        "analysis_identity": analysis["identity"], "primary_method": "CT_A3_ER",
        "deployment": "N0_UNCHANGED", "implementation_status": "FROZEN", "scientific_status": store["status"],
        "main_scene_outputs": analysis["main_scene_outputs"], "internal_pools": analysis["internal_pools"],
        "cold_timing_leaves": analysis["cold_timing_leaves"],
        "benchmark_coverage": {"replica8": {"capture_count": 8, "physical_families": 8, "exposure": "HISTORICALLY_EXPOSED"},
            "scannet_cf18": {"capture_count": 18, "physical_families": 7, "unrecorded_exposure": "UNVERIFIED"}},
        "timing_status": timing["status"], "external_protocol_status": "AUTHOR_PROTOCOL_AS_REPORTED_NOT_INDEPENDENTLY_VERIFIED",
        "performance_outcome": {cohort: row["guardrails_vs_A1"] for cohort, row in analysis["cohorts"].items()},
        "publication_status": "PENDING_PUSH_VERIFICATION", "inputs": index.entries(), "outputs": outputs}
    result["identity"] = canonical_digest(result)
    atomic_write_json(root / "reports/receipt.json", result)
    index.write_memo(root / "validation/input_verifications.json")
    return result
