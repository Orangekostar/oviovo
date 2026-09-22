"""Resumable phase orchestration without overwriting prior studies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.run_ovimap_module_study import (
    PhaseResult,
    StudyRunner,
    _base_resolved_config,
    bind_phase,
    default_phase_handlers,
)
from src.static_ovmap.module_validation.contracts import ReceiptStatus, StudySpec

SPEC_PATH = (
    Path(__file__).parents[2]
    / "docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json"
)


def test_compatible_complete_phase_is_reused(tmp_path: Path) -> None:
    calls: list[str] = []

    def bind(context):
        calls.append(context.phase)
        return PhaseResult(
            status=ReceiptStatus.COMPLETE,
            outputs={"bound": "yes"},
            metrics={"count": 1},
        )

    spec = StudySpec.load(SPEC_PATH)
    runner = StudyRunner(spec, tmp_path, {"device": "cpu"}, handlers={"bind": bind})
    first = runner.run("bind")["bind"]
    resumed = StudyRunner(spec, tmp_path, {"device": "cpu"}, handlers={"bind": bind})
    second = resumed.run("bind")["bind"]

    assert first.reused is False
    assert second.reused is True
    assert calls == ["bind"]
    assert runner.attempt_dir == tmp_path / "attempt_001"
    assert resumed.attempt_dir == runner.attempt_dir


def test_changed_full_cache_key_creates_new_attempt(tmp_path: Path) -> None:
    spec = StudySpec.load(SPEC_PATH)
    handler = lambda context: PhaseResult(status=ReceiptStatus.COMPLETE)

    first = StudyRunner(spec, tmp_path, {"device": "cpu"}, handlers={"bind": handler})
    first.run("bind")
    changed = StudyRunner(spec, tmp_path, {"device": "cuda:0"}, handlers={"bind": handler})
    changed.run("bind")

    assert first.cache_key != changed.cache_key
    assert first.attempt_dir == tmp_path / "attempt_001"
    assert changed.attempt_dir == tmp_path / "attempt_002"
    assert json.loads((first.attempt_dir / "resolved_config.json").read_text())["cache_key"] == first.cache_key
    assert json.loads((changed.attempt_dir / "resolved_config.json").read_text())["cache_key"] == changed.cache_key


def test_leaf_resolved_config_unwraps_existing_attempt(tmp_path: Path) -> None:
    spec = StudySpec.load(SPEC_PATH)
    configured = {"device": "cpu", "assets": {"digest": "abc"}}
    runner = StudyRunner(spec, tmp_path, configured)

    loaded = _base_resolved_config(spec, runner.attempt_dir / "resolved_config.json")

    assert loaded == configured


def test_all_continues_to_report_after_blocked_branch(tmp_path: Path) -> None:
    called: list[str] = []

    def handler(context):
        called.append(context.phase)
        if context.phase == "capture":
            return PhaseResult(
                status=ReceiptStatus.BLOCKED_PREREQUISITE,
                blockers=("BLOCKED_NATIVE_CAPTURE",),
            )
        return PhaseResult(status=ReceiptStatus.COMPLETE)

    spec = StudySpec.load(SPEC_PATH)
    handlers = {phase: handler for phase in spec.phases if phase != "all"}
    runner = StudyRunner(spec, tmp_path, {"device": "cpu"}, handlers=handlers)

    results = runner.run("all")

    assert called == list(spec.phases[:-1])
    assert results["capture"].receipt.status is ReceiptStatus.BLOCKED_PREREQUISITE
    assert results["report"].receipt.status is ReceiptStatus.COMPLETE
    assert (runner.attempt_dir / "receipts/report.json").is_file()
    progress = (runner.attempt_dir / "progress.md").read_text(encoding="utf-8")
    assert "| capture | BLOCKED_PREREQUISITE |" in progress
    assert "| report | COMPLETE |" in progress


def test_bind_phase_records_missing_scannet_as_specific_blocker(tmp_path: Path) -> None:
    spec = StudySpec.load(SPEC_PATH)
    config = {
        "repository_root": str(Path(__file__).parents[2]),
        "asset_resolution": {
            "status": "BLOCKED_PREREQUISITE",
            "bindings": {},
            "ambiguities": {},
            "missing": ["scannet_root"],
            "searched_roots": [str(tmp_path / "authorized_data")],
        },
    }
    runner = StudyRunner(spec, tmp_path / "out", config, handlers={"bind": bind_phase})

    result = runner.run("bind")["bind"].receipt

    assert result.status is ReceiptStatus.BLOCKED_INDEPENDENT_SCENES
    assert result.blockers == ("BLOCKED_INDEPENDENT_SCENES",)
    inventory = json.loads((runner.attempt_dir / "scene_inventory.json").read_text())
    splits = json.loads((runner.attempt_dir / "splits.json").read_text())
    assert inventory["status"] == "MISSING_SCANNET_ROOT"
    assert splits["status"] == "BLOCKED_INDEPENDENT_SCENES"


def test_default_handlers_cover_every_leaf_phase() -> None:
    spec = StudySpec.load(SPEC_PATH)
    assert tuple(default_phase_handlers()) == spec.phases[:-1]


def test_deleted_or_modified_output_is_rebuilt_and_invalidates_downstream(tmp_path) -> None:
    calls = []

    def handler(context):
        calls.append(context.phase)
        output = context.attempt_dir / f"{context.phase}.json"
        output.write_text(json.dumps({"call": len(calls)}))
        return PhaseResult(ReceiptStatus.COMPLETE, outputs={"result": output.name})

    spec = StudySpec.load(SPEC_PATH)
    runner = StudyRunner(spec, tmp_path, {}, handlers={"bind": handler, "capture": handler})
    runner.run("bind")
    runner.run("capture")
    (runner.attempt_dir / "bind.json").unlink()
    assert not runner.run("bind")["bind"].reused
    assert not runner.run("capture")["capture"].reused
    (runner.attempt_dir / "capture.json").write_text("tampered")
    assert not runner.run("capture")["capture"].reused
    assert calls == ["bind", "capture", "bind", "capture", "capture"]


def test_all_records_exception_and_still_runs_report(tmp_path) -> None:
    def handler(context):
        if context.phase == "geometry":
            raise ValueError("corrupted surface")
        return PhaseResult(ReceiptStatus.COMPLETE)

    spec = StudySpec.load(SPEC_PATH)
    runner = StudyRunner(spec, tmp_path, {}, handlers={
        phase: handler for phase in spec.phases[:-1]
    })
    results = runner.run("all")
    assert results["geometry"].receipt.status == ReceiptStatus.FAILED
    assert results["report"].receipt.status == ReceiptStatus.COMPLETE
    assert (runner.attempt_dir / "errors/geometry.json").is_file()


def test_frozen_config_cannot_reuse_old_code_identity(tmp_path, monkeypatch) -> None:
    import scripts.evaluation.run_ovimap_module_study as study

    spec = StudySpec.load(SPEC_PATH)
    runner = StudyRunner(spec, tmp_path, {"code_identity": {"head": "a" * 40}})
    monkeypatch.setattr(study, "_code_identity", lambda root: {"head": "b" * 40})
    loaded = _base_resolved_config(spec, runner.attempt_dir / "resolved_config.json")
    assert loaded["code_identity"]["head"] == "b" * 40
    updated = StudyRunner(spec, tmp_path, loaded)
    assert updated.attempt_dir != runner.attempt_dir


def test_attempt_lock_rejects_concurrent_execution(tmp_path) -> None:
    import fcntl

    spec = StudySpec.load(SPEC_PATH)
    runner = StudyRunner(spec, tmp_path, {}, handlers={})
    with (runner.attempt_dir / ".run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            runner.run("bind")


def test_report_only_is_isolated_and_does_not_claim_execution(tmp_path) -> None:
    spec = StudySpec.load(SPEC_PATH)
    runner = StudyRunner(spec, tmp_path, {}, handlers=default_phase_handlers())
    receipt = runner.run("report")["report"].receipt
    for raw_path in receipt.outputs.values():
        assert Path(raw_path).is_relative_to(runner.attempt_dir)
    manifest = json.loads((runner.attempt_dir / "execution_manifest.json").read_text())
    assert manifest["statuses"]["implementation"] == "PARTIAL"
    assert manifest["cost"] == {}
    assert manifest["statuses"]["confirmation"] == "NOT_RUN_PREREQUISITES"
    assert manifest["errors"]


def test_blocked_development_is_not_a_frozen_negative_scientific_result(tmp_path) -> None:
    spec = StudySpec.load(SPEC_PATH)
    handlers = dict(default_phase_handlers())
    for name in ("bind", "capture", "semantic", "geometry", "query"):
        handlers[name] = lambda context: PhaseResult(
            ReceiptStatus.BLOCKED_INDEPENDENT_SCENES, blockers=("BLOCKED_INDEPENDENT_SCENES",)
        )
    runner = StudyRunner(spec, tmp_path, {}, handlers=handlers)
    result = runner.run("all")
    assert result["select"].receipt.status == ReceiptStatus.BLOCKED_PREREQUISITE
    assert result["confirm"].receipt.status == ReceiptStatus.BLOCKED_PREREQUISITE
    selection = json.loads((runner.attempt_dir / "selection.json").read_text())
    assert selection["status"] == "NOT_FROZEN_PREREQUISITES"


def test_export_contains_final_report_receipt_and_progress(tmp_path):
    from scripts.evaluation.run_ovimap_module_study import export_attempt

    spec = StudySpec.load(SPEC_PATH)
    handlers = {name: lambda context: PhaseResult(ReceiptStatus.COMPLETE)
                for name in spec.phases[:-1]}
    runner = StudyRunner(spec, tmp_path / "run", {}, handlers=handlers)
    runner.run("all")
    export_attempt(runner, tmp_path / "export")
    assert json.loads((tmp_path / "export/receipts/report.json").read_text())["status"] == "COMPLETE"
    assert "| report | COMPLETE |" in (tmp_path / "export/progress.md").read_text()
    with pytest.raises(FileExistsError):
        export_attempt(runner, tmp_path / "export")


def test_changed_smoke_input_invalidates_phase_receipt(tmp_path):
    from src.static_ovmap.module_validation.assets import sha256_file

    external = tmp_path / "frame.png"
    external.write_bytes(b"original")
    def handler(context):
        output = context.attempt_dir / "capture_smoke.json"
        output.write_text(json.dumps({"input_identities": [{"path": str(external),
            "sha256": sha256_file(external), "bytes": external.stat().st_size}]}))
        return PhaseResult(ReceiptStatus.COMPLETE, outputs={"smoke": str(output)})
    spec = StudySpec.load(SPEC_PATH)
    runner = StudyRunner(spec, tmp_path / "run", {}, handlers={"bind": handler})
    runner.run("bind")
    assert runner.run("bind")["bind"].reused
    external.write_bytes(b"modified")
    assert not runner.run("bind")["bind"].reused


def test_capture_runtime_receipt_binds_transitive_scene_payloads(tmp_path):
    from src.static_ovmap.module_validation.assets import sha256_file

    external = tmp_path / "native_surface.npz"
    external.write_bytes(b"surface")

    def handler(context):
        output = context.attempt_dir / "capture_runtime.json"
        output.write_text(json.dumps({"input_identities": [{"path": str(external),
            "sha256": sha256_file(external), "bytes": external.stat().st_size}]}))
        return PhaseResult(ReceiptStatus.COMPLETE, outputs={"runtime": str(output)})

    runner = StudyRunner(StudySpec.load(SPEC_PATH), tmp_path / "run", {}, handlers={"bind": handler})
    runner.run("bind")
    assert runner.run("bind")["bind"].reused
    external.write_bytes(b"changed native surface")
    assert not runner.run("bind")["bind"].reused
