"""Required DAG order, read-only phase dispatch, and pre-Replica source guards."""

from argparse import Namespace

import pytest


def test_all_dispatches_each_required_phase_once_and_report_never_maps(monkeypatch, tmp_path):
    from static_ovmap.recovery_wave2.binding import read
    from static_ovmap.module_validation.contracts import atomic_write_json
    from static_ovmap.recovery_wave2.workflow import PHASES, Workflow

    spec = tmp_path / "spec.json"
    atomic_write_json(spec, {})
    args = Namespace(spec=str(spec), output_root=str(tmp_path / "output"), mapping_workers=2,
                     mapping_threads=8, evaluation_workers=3, resume=True)
    calls = []

    def handler(name):
        def run(self):
            calls.append(name)
            self.binding = {"identity": "bound"}
            return {"status": "COMPLETE"}
        return run

    for phase in PHASES:
        monkeypatch.setattr(Workflow, phase.replace("-", "_"), handler(phase))
    workflow = Workflow(args)
    workflow.run("all")
    assert calls == list(PHASES)
    assert read(tmp_path / "output/execution/phases/publish.json")["status"] == "COMPLETE"
    calls.clear()
    workflow.run("report")
    assert calls == ["bind", "report"]
    calls.clear()
    workflow.run("publish")
    assert calls == ["bind", "publish"]


def test_source_entry_points_reject_replica_before_any_source_scoring(tmp_path):
    from static_ovmap.recovery_wave2.recovery_sources import prepare_cached_sources
    from static_ovmap.recovery_wave2.recovery_fc_worker import run_scene

    binding = {"datasets": {"replica": ["room0"]}, "output_root": str(tmp_path)}
    with pytest.raises(FileNotFoundError, match="freeze/receipt"):
        prepare_cached_sources(binding, "room0", {})
    with pytest.raises(FileNotFoundError, match="freeze/receipt"):
        run_scene(binding, "room0")
