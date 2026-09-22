"""Public phases execute the actual scientific drivers and bind deep outputs."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation import run_ovimap_module_study as entry
from src.static_ovmap.module_validation.boundary_jobs import file_identity
from src.static_ovmap.module_validation.contracts import ReceiptStatus, StudySpec


@pytest.mark.parametrize("phase", ["semantic", "geometry", "query"])
def test_scannet_phase_uses_real_leaf_cli_and_records_gate_status(tmp_path, monkeypatch, phase):
    runtime = {"mapping_python": sys.executable}
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime))
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps({"runtime_config": str(runtime_path), "study_root": str(tmp_path / "study")}))
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        output = tmp_path / "study" / phase
        output.mkdir(parents=True)
        blob = output / "prediction.txt"
        blob.write_text("actual locked output")
        child = output / "child_receipt.json"
        child.write_text(json.dumps({"status": "COMPLETE", "input_identity": "child", "outputs": [file_identity(blob)]}))
        rows = [{"method_id": "example", "scene_id": scene, "metrics": {"uap": .1, "miou": .2}} for scene in ("s1", "s2")]
        (output / "select_receipt.json").write_text(json.dumps({"status": "COMPLETE", "input_identity": "select",
            "outputs": [file_identity(child)], "inputs": [file_identity(config_path)],
            "rows": rows, "results": [{"row": row} for row in rows], "learned_status": "BLOCKED_TARGET_SUPPORT"}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", run)
    spec = StudySpec.load(Path(entry.REPOSITORY_ROOT) / "docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json")
    context = entry.PhaseContext(phase, attempt, "key", spec,
        {"scannet_runtime": runtime, "scannet_study_config": str(config_path)},
        {"capture": SimpleNamespace(status=ReceiptStatus.COMPLETE, blockers=())})
    result = getattr(entry, phase + "_phase")(context)
    assert result.status == ReceiptStatus.COMPLETE
    assert result.metrics["scientific_evaluation_rows"] == 2
    assert result.metrics["learned_status"] == "BLOCKED_TARGET_SUPPORT"
    assert result.blockers == ()
    assert Path(commands[0][1]).name == f"run_ovimap_scannet_{phase}.py"
    assert commands[0][-2:] == ["--phase", "all"]
    manifest = json.loads(Path(result.outputs["runtime"]).read_text())
    assert any(Path(row["path"]).name == "prediction.txt" for row in manifest["input_identities"])


def test_selection_phase_executes_leaf_and_copies_verified_selection(tmp_path, monkeypatch):
    runtime = {"mapping_python": sys.executable}
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime))
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps({"runtime_config": str(runtime_path), "study_root": str(tmp_path / "study")}))
    attempt = tmp_path / "attempt"
    attempt.mkdir()

    def run(command, **kwargs):
        assert Path(command[1]).name == "run_ovimap_scannet_selection.py"
        output = tmp_path / "study/selection"
        output.mkdir(parents=True)
        selection = output / "selection.json"
        selection.write_text(json.dumps({"status": "FROZEN", "final_candidate": "N0", "science_status": "NO_NET_GAIN",
            "confirmation_status": "NOT_REQUIRED_NO_RETAINED_CANDIDATE", "combinations": [],
            "module_selection": {"semantic": {"selected_method": "N0"}, "geometry": {"selected_method": "G_ORIGINAL"},
                                 "query": {"selected_method": "Q_COMBINE"}}}))
        (output / "receipt.json").write_text(json.dumps({"status": "COMPLETE", "input_identity": "selection",
            "outputs": [file_identity(selection)], "selection_path": str(selection)}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", run)
    spec = StudySpec.load(Path(entry.REPOSITORY_ROOT) / "docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json")
    context = entry.PhaseContext("select", attempt, "key", spec,
        {"scannet_runtime": runtime, "scannet_study_config": str(config_path)},
        {phase: SimpleNamespace(status=ReceiptStatus.COMPLETE, blockers=()) for phase in ("semantic", "geometry", "query")})
    result = entry.select_phase(context)
    assert result.status == ReceiptStatus.COMPLETE
    assert json.loads((attempt / "selection.json").read_text())["final_candidate"] == "N0"
    assert result.metrics["retained_modules"] == 0
