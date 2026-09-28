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


def test_publication_copy_preserves_bytes_and_rejects_tampering(tmp_path):
    from src.static_ovmap.m2_reviewer_study.publication import (
        copy_artifact,
        verify_bundle,
    )

    source = tmp_path / "metric.json"
    source.write_bytes(b'{ "apall": 0.125 }\n')
    destination = tmp_path / "bundle"
    row = copy_artifact(source, destination, "rows/metric.json")
    assert (destination / row["relative"]).read_bytes() == source.read_bytes()
    assert verify_bundle(destination, [row], limit=1000) == len(source.read_bytes())
    with pytest.raises(ValueError, match="limit"):
        verify_bundle(destination, [row], limit=1)
    with pytest.raises(ValueError, match="relative"):
        copy_artifact(source, destination, "../escape.json")
    (destination / row["relative"]).write_bytes(b'{}')
    with pytest.raises(ValueError, match="changed"):
        verify_bundle(destination, [row])


def test_workflow_failed_query_does_not_block_independent_diagnostics(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.static_ovmap.m2_reviewer_study import workflow

    called = []

    def run(command, **kwargs):
        phase = command[command.index("--phase") + 1]
        called.append(phase)
        return SimpleNamespace(returncode=1 if phase == "query-controls" else 0)

    monkeypatch.setattr(workflow.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="incomplete"):
        workflow.run_all(spec="spec.json", source_transfer="transfer.json", output_root=tmp_path, gpu="1")
    assert called == ["bind", "core", "query-controls", "diagnostics", "robustness", "fresh"]
