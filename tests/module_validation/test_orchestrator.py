"""Resumable phase orchestration without overwriting prior studies."""

from __future__ import annotations

import json
from pathlib import Path

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
