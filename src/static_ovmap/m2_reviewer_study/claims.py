"""Explicit, conservative reviewer decisions with measured matched controls."""


def claim_ledger(report):
    pools = {r["method"]: r["metrics"] for r in report["pooled_rows"]
             if r["dataset"] == "Replica" and r["rank_mode"] == "OFFICIAL_CURRENT_CLASS"}
    primary = pools["CP_M2_EQUAL_CAL"]

    def delta(method, comparator):
        return {k: pools[method][k] - pools[comparator][k] for k in ("uap", "ap25", "ap50", "miou", "macc")}

    rows = []

    def add(claim, status, comparison, evidence, limitation, decision):
        rows.append({"claim": claim, "status": status, "matched_control": comparison,
                     "evidence": evidence, "limitation": limitation, "decision": decision})

    add("N0 score source is necessary", "MIXED", "M2_CAL versus A3_QS2", delta("CP_M2_EQUAL_CAL", "RV_A3_QS2"),
        "Small pooled difference; native mapping/owner/rank cost remains mandatory even when N0 scores are deleted. Eight exposed scenes do not establish necessity.", "SIMPLIFY_NECESSITY_CLAIM")
    add("Query observations add useful evidence", "SUPPORTED_IN_THIS_STUDY", "M2_CAL versus CAL-selected A2_NS2", delta("CP_M2_EQUAL_CAL", "RV_A2_NS2"),
        "Supports this source inclusion comparison, not superiority of learned selection; query observations add compute.", "RETAIN_BOUNDED_SOURCE_ABLATION")
    add("Static S2 contributes beyond N0+Q", "MIXED", "M2_CAL versus A1_NQ", delta("CP_M2_EQUAL_CAL", "RV_A1_NQ"),
        "Measured gains are conditional on the same fixed maps and extra frozen S2 model; no universal three-source necessity.", "RETAIN_WITH_COST_AND_VARIABILITY")
    add("Soft distributions outperform hard voting", "SUPPORTED_IN_THIS_STUDY", "M2_CAL versus A4_HARD", delta("CP_M2_EQUAL_CAL", "RV_A4_HARD"),
        "Hard votes use declared normalized counts; this does not make either distribution universally calibrated.", "RETAIN_FIXED_VOCABULARY_COMPARISON")
    add("Source-specific fitting is required beyond sharpening", "NOT_SUPPORTED", "M2_CAL versus A5_T001 and A6_SHARED_T",
        {m: delta("CP_M2_EQUAL_CAL", m) for m in ("RV_A5_T001", "RV_A6_SHARED_T")},
        "Small label/metric differences; common sharpening closely matches performance and has slightly better paired transfer NLL/Brier. Additional CAL supervision is disclosed.", "SIMPLIFY_TO_SHARPENING_EXPLANATION")
    add("Native score representation affects hidden weighting", "SUPPORTED_IN_THIS_STUDY", "A7_COS_FIXED/REFIT versus unchanged M2_CAL",
        {m: delta(m, "CP_M2_EQUAL_CAL") for m in ("RV_A7_COS_FIXED", "RV_A7_COS_REFIT")},
        "Same native aggregate and source temperatures for fixed probe. This is a representation mechanism probe, not a newly selected deployment method.", "RETAIN_REPRESENTATION_CAVEAT")
    query = report["query"]["Replica"]
    if query is None:
        add("Learned query selection outperforms matched alternatives", "NOT_TESTED_PREREQUISITE", "B200 COMBINE and random17/23/41",
            {"CAL_curve_gate": report["curve_gate"], "Replica_status": "PENDING"}, "Replica matched controls are still running.", "WITHHOLD_CLAIM")
    else:
        selected = {r["policy"]: r["metrics"] for r in query["pooled"] if r["mode"] == "CAL" and r["rank_mode"] == "OFFICIAL_CURRENT_CLASS"}
        random = {key: sum(selected[p][key] for p in ("RV_Q_RANDOM_s17", "RV_Q_RANDOM_s23", "RV_Q_RANDOM_s41")) / 3 for key in ("uap", "miou")}
        comparisons = {name: {key: selected["Q_GAIN"][key] - value[key] for key in ("uap", "miou")}
                       for name, value in (("Q_COMBINE", selected["Q_COMBINE"]), ("RANDOM_SEED_MEAN", random))}
        status = "MIXED" if any(v > 0 for row in comparisons.values() for v in row.values()) else "NOT_SUPPORTED"
        add("Learned query selection outperforms matched alternatives", status, "B200 COMBINE and random17/23/41", comparisons,
            "CAL gate did not show dominance; Replica was historically exposed. Seeds are separate replicates, never an ensemble or post-hoc seed choice.", "NO_BROAD_LEARNED_SELECTION_SUPERIORITY")
    add("Gains survive official reranking and pooling", "MIXED", "M2_CAL versus N0; official-current-class released dataset pool",
        {"primary": primary, "delta_vs_N0": delta("CP_M2_EQUAL_CAL", "N0")},
        "APall improves modestly while AP50 and mIoU decrease. Macro-bootstrap intervals are not pooled intervals.", "REPORT_METRIC_TRADEOFF")
    probability = {r["method"]: {k: r[k] for k in ("objects", "nll", "brier", "ece15", "accuracy")}
                   for r in report["probability_rows"] if r["dataset"] == "Replica" and r["scene"] == "OBJECT_POOL" and r["population"] == "paired"}
    add("Temperature calibration improves transfer probabilities", "SUPPORTED_IN_THIS_STUDY", "M2_CAL versus M2_RAW on the same paired population",
        {k: probability[k] for k in ("CP_M2_EQUAL_RAW", "CP_M2_EQUAL_CAL", "RV_A5_T001")},
        "Conditional on identifiable objects with all three actual sources; not calibration of every false positive. Common sharpening explains much of the improvement.", "RETAIN_CONDITIONAL_PROBABILITY_RESULT")
    add("Current formula supports arbitrary single-text retrieval", "NOT_SUPPORTED", "Singleton softmax degeneracy fixture",
        {"softmax_one_text": 1, "image_inference_in_text_test": 0},
        "Prompt and fixed-distractor tests are vocabulary stress tests. They do not validate arbitrary-text retrieval or certify distractors absent in every region.", "RESTRICT_TO_FIXED_VOCABULARY_RECOGNITION")
    fresh_complete = report["fresh"] and report["fresh"]["status"] == "FRESH_CONFIRMATION_COMPLETE"
    add("Independent fresh scenes justify broad generalization", "MIXED" if fresh_complete else "NOT_TESTED_PREREQUISITE", "Four genuinely unexposed local families",
        report["fresh"], "Four local families remain a limited sample." if fresh_complete else "No eligible local family; old confirmation and Replica cannot be relabelled fresh.",
        "REPORT_NEW_SCENE_EVIDENCE_WITH_LIMITS" if fresh_complete else "WITHHOLD_GENERALIZATION_CLAIM")
    return {"status": "PROVISIONAL" if report["status"] != "FINAL_EVIDENCE_READY" else "FINAL",
            "scientific_recommendation": "SIMPLIFY: fixed-vocabulary score sharpening and complementary evidence with representation sensitivity; no blanket necessity or online/retrieval claim",
            "deployment": "N0_UNCHANGED", "rank_mode": "OFFICIAL_CURRENT_CLASS", "aggregation": "RELEASED_DATASET_POOL",
            "claims": rows}
