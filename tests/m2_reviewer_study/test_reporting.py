import pytest


def test_measured_matrix_requires_exact_keys_metrics_and_pool_inputs():
    from src.static_ovmap.m2_reviewer_study.audit import validate_matrix

    metrics = {"uap": .1, "apall": .1, "ap25": .3, "ap50": .2, "miou": .4, "macc": .5}
    rows = [{"status": "COMPLETE", "scene": s, "dataset": "Replica", "method": "M", "rank_mode": "R",
             "evaluation_identity": s, "metrics": metrics} for s in ("a", "b")]
    pools = [{"status": "COMPLETE", "dataset": "Replica", "method": "M", "rank_mode": "R",
              "scene_order": ["a", "b"], "ordered_inputs": ["a", "b"], "metrics": metrics}]
    arguments = {"datasets": {"Replica": ["a", "b"]}, "methods": ["M"], "ranks": ["R"]}
    assert validate_matrix(rows, pools, **arguments) == {"scene_rows": 2, "pooled_rows": 1}
    with pytest.raises(ValueError, match="duplicate"):
        validate_matrix([rows[0], rows[0]], pools, **arguments)
    with pytest.raises(ValueError, match="ordered"):
        validate_matrix(rows, [{**pools[0], "ordered_inputs": ["b", "a"]}], **arguments)
    with pytest.raises(ValueError, match="APall"):
        validate_matrix([{**rows[0], "metrics": {**metrics, "apall": .2}}, rows[1]], pools, **arguments)
    with pytest.raises(ValueError, match="matrix"):
        validate_matrix(rows[:1], pools, **arguments)


def test_report_gate_rejects_smoke_or_missing_evidence_and_accepts_declared_fresh_block():
    from src.static_ovmap.m2_reviewer_study.reporting import assert_report_ready

    states = {"core": "COMPLETE", "query": "COMPLETE", "diagnostics": "COMPLETE", "robustness": "COMPLETE",
              "fresh": "FRESH_BLOCKED_INSUFFICIENT_LOCAL_UNEXPOSED_FAMILIES"}
    assert_report_ready(states)
    for replacement in ("PENDING", "SMOKE_COMPLETE", None):
        with pytest.raises(ValueError, match="query"):
            assert_report_ready({**states, "query": replacement})
    with pytest.raises(ValueError, match="fresh"):
        assert_report_ready({k: v for k, v in states.items() if k != "fresh"})
