"""Reports keep scene denominators, split roles and query allowances separate."""

import gzip
import json

import pytest

from src.static_ovmap.module_validation.study_report import (
    released_comparisons,
    render_scientific_results,
    summarize_rows,
)


def test_defined_scene_means_keep_denominators_and_never_pool_roles_or_budgets():
    rows = [{"method_id": "Q_GAIN", "scene_id": scene, "role": role, "budget": budget,
             "metrics": {"uap": ap, "miou": .4}, "logical_cost": {"attempts": budget}}
            for scene, role, budget, ap in (("s1", "select", 200, .2), ("s2", "select", 200, None),
                ("s1", "select", 400, .8), ("c1", "cal", 200, .6))]
    summary = summarize_rows(rows)
    group = next(row for row in summary if row["role"] == "select" and row["budget"] == 200)
    assert group["metrics"]["uap"] == .2
    assert group["defined_scene_counts"]["uap"] == 1
    assert group["scene_count"] == 2
    assert group["selection_status"] == "INCONCLUSIVE_UNDEFINED_METRIC"
    assert len(summary) == 3
    with pytest.raises(ValueError, match="duplicate"):
        summarize_rows([*rows, rows[0]])


def test_report_has_five_tables_with_real_scene_rows_and_nulls():
    rows = [{"method_id": "N0", "scene_id": "s1", "role": "select", "budget": None,
             "metrics": {"uap": .125, "miou": None}, "logical_cost": {"attempts": 7}, "evidence_path": "native.json"}]
    evidence = {"rows": rows, "summaries": summarize_rows(rows), "semantic_events": [],
                "geometry_events": [], "query_events": [], "method_status": {"Q_GAIN": "BLOCKED_QUERY_TARGET_SUPPORT"}}
    text = render_scientific_results(evidence, {"final_candidate": "N0", "science_status": "NO_NET_GAIN"},
        {"status": "NOT_REQUIRED_NO_RETAINED_CANDIDATE", "rows": []})
    assert sum(text.count(f"## Table {letter}.") for letter in "ABCDE") == 5
    assert "s1" in text and "12.5%" in text and "null" in text
    assert "BLOCKED_QUERY_TARGET_SUPPORT" in text
    assert "0/1" in text


def test_object_comparison_uses_actual_released_match_and_eligibility_events(tmp_path):
    rows = []
    for method, key, matched, owner in (("N0", "a", 1000, "owner_000001"), ("S_PAIRED", "b", 2000, "owner_000002")):
        base = tmp_path / "scenes/s1/semantic/released/s1" / key / "context/released"
        base.mkdir(parents=True)
        trace = {"states": [{"overlap_threshold": .5}], "events": [{"overlap_threshold": .5,
            "event": "first_match", "gt_id": matched, "candidate_file": owner + ".npy", "confidence": 1., "overlap": .75}]}
        matches = {"s1": {"pred": {"chair": [{"filename": owner + ".npy", "label_id": 2}]}}}
        for name, value in (("released_trace.json.gz", trace), ("released_matches.json.gz", matches)):
            with gzip.open(base / name, "wt") as handle:
                json.dump(value, handle)
        rows.append({"method_id": method, "scene_id": "s1", "role": "select", "prediction_key": key})
    result = released_comparisons({"rows": rows}, {"study_root": str(tmp_path)})[0]
    assert result["thresholds"][0]["gained_gt_ids"] == ["2000"]
    assert result["thresholds"][0]["lost_gt_ids"] == ["1000"]
    assert result["eligibility_entries"] == ["owner_000002"]
    assert result["eligibility_exits"] == ["owner_000001"]
