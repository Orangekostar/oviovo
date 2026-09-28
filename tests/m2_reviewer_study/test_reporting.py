import pytest


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
