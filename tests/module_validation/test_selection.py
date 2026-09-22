"""Frozen S/G/Q selection, combinations, and descriptive intervals."""

from __future__ import annotations

from src.static_ovmap.module_validation.selection import (
    SceneMethodMetrics,
    bootstrap_mean_delta,
    plan_combinations,
    plan_confirmation,
    select_final_candidate,
    select_geometry_module,
    select_query_module,
    select_semantic_module,
)


def _rows(method, uap, miou, *, ap50=(0.0, 0.0), ap75=(0.0, 0.0), cost=10, changes=1):
    return tuple(
        SceneMethodMetrics(
            method_id=method,
            scene_id=f"s{index}",
            uap=uap[index],
            miou=miou[index],
            canonical_ap50=ap50[index],
            canonical_ap75=ap75[index],
            logical_attempts=cost,
            added_cost=float(cost),
            effective_changes=changes,
        )
        for index in range(2)
    )


def test_semantic_mechanism_gate_and_simpler_no_context_tie() -> None:
    rows = {
        "N0": _rows("N0", (0.50, 0.50), (0.60, 0.60), cost=0, changes=0),
        "S_SIGLIP2_AREA": _rows("S_SIGLIP2_AREA", (0.51, 0.51), (0.60, 0.60), cost=20),
        "S_SIMPLE": _rows("S_SIMPLE", (0.51, 0.51), (0.60, 0.60), cost=2),
        "S_NO_CONTEXT": _rows("S_NO_CONTEXT", (0.53, 0.53), (0.61, 0.61), cost=3),
        "S_PAIRED": _rows("S_PAIRED", (0.53, 0.53), (0.61, 0.61), cost=4),
    }
    decision = select_semantic_module(rows, frozen_teacher_id="S_SIGLIP2_AREA")
    assert decision.selected_method == "S_NO_CONTEXT"
    assert decision.mechanism_status == "UNSUPPORTED_CONTEXT_MATCHED_BY_NO_CONTEXT"
    assert decision.paired_eligible is True

    degraded = dict(rows)
    degraded["S_PAIRED"] = _rows("S_PAIRED", (0.54, 0.49), (0.61, 0.61))
    degraded["S_NO_CONTEXT"] = _rows("S_NO_CONTEXT", (0.49, 0.49), (0.61, 0.61))
    assert select_semantic_module(
        degraded, frozen_teacher_id="S_SIGLIP2_AREA"
    ).paired_eligible is False


def test_geometry_quality_gate_and_original_fallback() -> None:
    rows = {
        "G_ORIGINAL": _rows("G_ORIGINAL", (0.50, 0.50), (0.60, 0.60), ap50=(0.4, 0.4), ap75=(0.3, 0.3), changes=0),
        "G_AGREEMENT": _rows("G_AGREEMENT", (0.50, 0.50), (0.60, 0.60), ap50=(0.41, 0.41), ap75=(0.3, 0.3)),
        "G_QUALITY": _rows("G_QUALITY", (0.51, 0.51), (0.60, 0.60), ap50=(0.43, 0.43), ap75=(0.31, 0.31)),
    }
    decision = select_geometry_module(rows)
    assert decision.selected_method == "G_QUALITY"
    assert decision.quality_eligible is True
    broken = dict(rows)
    broken["G_QUALITY"] = _rows("G_QUALITY", (0.51, 0.51), (0.60, 0.60), ap50=(0.43, 0.39), ap75=(0.31, 0.31))
    broken["G_AGREEMENT"] = _rows("G_AGREEMENT", (0.49, 0.49), (0.60, 0.60), ap50=(0.39, 0.39), ap75=(0.3, 0.3))
    assert select_geometry_module(broken).selected_method == "G_ORIGINAL"


def test_query_cal_comparator_lock_gain_gate_and_efficiency_only() -> None:
    cal = {
        "Q_COMBINE": _rows("Q_COMBINE", (0.40, 0.40), (0.50, 0.50), cost=180),
        "Q_AREA": _rows("Q_AREA", (0.41, 0.41), (0.50, 0.50), cost=200),
        "Q_UNCERTAINTY": _rows("Q_UNCERTAINTY", (0.41, 0.41), (0.50, 0.50), cost=190),
    }
    select = {
        "Q_UNCERTAINTY": _rows("Q_UNCERTAINTY", (0.45, 0.45), (0.55, 0.55), cost=190),
        "Q_GAIN": _rows("Q_GAIN", (0.46, 0.46), (0.55, 0.56), cost=200),
    }
    decision = select_query_module(cal, select)
    assert decision.locked_comparator == "Q_UNCERTAINTY"
    assert decision.selected_method == "Q_GAIN"
    assert decision.gain_status == "ELIGIBLE_ACCURACY"

    efficiency = dict(select)
    efficiency["Q_GAIN"] = _rows("Q_GAIN", (0.45, 0.45), (0.55, 0.55), cost=100)
    efficient = select_query_module(cal, efficiency)
    assert efficient.gain_status == "EFFICIENCY_ONLY"


def test_bootstrap_is_seeded_and_confirmation_is_frozen_and_bounded() -> None:
    first = bootstrap_mean_delta((0.1, 0.2), (0.0, 0.0))
    second = bootstrap_mean_delta((0.1, 0.2), (0.0, 0.0))
    assert first == second
    assert first.resamples == 2000
    assert first.seed == 17
    plan = plan_confirmation("Q_GAIN", nearest_comparison="Q_UNCERTAINTY")
    assert plan.rows == ("N0", "Q_GAIN", "Q_UNCERTAINTY")
    assert len(plan.rows) <= 4
    assert plan_confirmation("N0").status == "NOT_REQUIRED_NO_RETAINED_CANDIDATE"


def test_combination_cap_and_final_matched_baseline_gate() -> None:
    combinations = plan_combinations(
        semantic_method="S_NO_CONTEXT",
        geometry_method="G_QUALITY",
        query_method="Q_GAIN",
        fresh_mask_supported=True,
        query_provenance_supported=True,
    )
    assert combinations.required == ("COMBO_GS", "COMBO_Q_REFINEMENT")
    assert len(combinations.required) == 2
    blocked = plan_combinations(
        semantic_method="S_NO_CONTEXT",
        geometry_method="G_ORIGINAL",
        query_method="Q_GAIN",
        fresh_mask_supported=False,
        query_provenance_supported=False,
    )
    assert blocked.required == ()
    assert blocked.blocked["COMBO_Q_REFINEMENT"] == "BLOCKED_FRESH_MASK_PROVENANCE"

    baseline = _rows("N0", (0.50, 0.50), (0.60, 0.60), cost=0, changes=0)
    candidates = {
        "S_NO_CONTEXT": _rows("S_NO_CONTEXT", (0.52, 0.52), (0.60, 0.61), cost=10),
        "COMBO_GS": _rows("COMBO_GS", (0.54, 0.49), (0.62, 0.62), cost=30),
    }
    final = select_final_candidate(
        candidates,
        matched_baselines={"S_NO_CONTEXT": baseline, "COMBO_GS": baseline},
        simplicity_order=("S_NO_CONTEXT", "COMBO_GS"),
    )
    assert final.selected_method == "S_NO_CONTEXT"
    assert final.science_status == "RETAINED_NET_GAIN"
    no_gain = select_final_candidate(
        {"S_NO_CONTEXT": _rows("S_NO_CONTEXT", (0.50, 0.50), (0.60, 0.60))},
        matched_baselines={"S_NO_CONTEXT": baseline},
        simplicity_order=("S_NO_CONTEXT",),
    )
    assert no_gain.selected_method == "N0"
    assert no_gain.science_status == "NO_NET_GAIN"


def test_final_tie_within_tolerance_prefers_lower_cost() -> None:
    baseline = _rows("N0", (.4, .4), (.5, .5))
    candidates = {
        "simple": _rows("simple", (.5, .5), (.6, .6), cost=1),
        "complex": _rows("complex", (.5 + 5e-11, .5 + 5e-11), (.6, .6), cost=10),
    }
    decision = select_final_candidate(
        candidates, matched_baselines={key: baseline for key in candidates},
        simplicity_order=("simple", "complex"),
    )
    assert decision.selected_method == "simple"


def test_final_missing_metric_is_inconclusive_not_negative() -> None:
    decision = select_final_candidate(
        {"S_SIMPLE": _rows("S_SIMPLE", (None, .5), (.6, .6))},
        matched_baselines={"S_SIMPLE": _rows("N0", (.4, .4), (.5, .5))},
        simplicity_order=("S_SIMPLE",),
    )
    assert decision.science_status == "INCONCLUSIVE_UNDEFINED_METRIC"


def test_cal_teacher_and_margins_use_metric_tolerance() -> None:
    from src.static_ovmap.module_validation.partition_quality import (
        select_geometry_margin,
    )
    from src.static_ovmap.module_validation.semantic_selector import (
        select_adoption_threshold,
        select_teacher,
    )

    assert select_teacher([
        {"method": "S_SIGLIP2_AREA", "mean_uap": .5, "mean_miou": .6, "median_cost": 1},
        {"method": "S_WOW_VOTE", "mean_uap": .5 + 5e-11, "mean_miou": .6, "median_cost": 10},
    ]) == "S_SIGLIP2_AREA"
    assert select_adoption_threshold([
        {"threshold": 0, "mean_uap": .5 + 5e-11, "mean_miou": .6, "harmful": 1, "replacements": 2},
        {"threshold": "KEEP_ALL", "mean_uap": .5, "mean_miou": .6, "harmful": 0, "replacements": 0},
    ]) == "KEEP_ALL"
    assert select_geometry_margin([
        {"margin": 0, "mean_ap50": .5 + 5e-11, "mean_ap75": .6, "changed_points": 20},
        {"margin": "KEEP_ALL", "mean_ap50": .5, "mean_ap75": .6, "changed_points": 0},
    ]) == "KEEP_ALL"
