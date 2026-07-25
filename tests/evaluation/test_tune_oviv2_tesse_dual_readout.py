from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.evaluation.tune_oviv2_tesse_dual_readout as tuner
from scripts.evaluation.evaluate_oviv2_tesse_occlusion import canonical_algorithm_hash
from scripts.evaluation.package_oviv2_tesse_dual_readout_result import package_result
from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (
    input_binding_values_sha256,
    non_temporal_config_sha256,
)
from scripts.evaluation.tune_oviv2_tesse_dual_readout import tune, tune_results_root


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO_ROOT
    / "configs/evaluation/manifests/oviv2_tesse_dual_readout_search_v1.json"
)
APARTMENT_CONFIG = REPO_ROOT / "configs/oviv2_tesse_cd_apartment_v2.json"


def _bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(value))
    return path


def _record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _metrics(**overrides: float) -> dict[str, float]:
    values = {
        "current_miou": 0.25,
        "object_f1": 0.55,
        "ghost_rate": 0.15,
        "background_f5_cm": 0.20,
        "recovery_frames": 300.0,
        "dynamic_f1": 0.40,
        "change_f1": 0.35,
        "runtime_seconds": 100.0,
    }
    values.update(overrides)
    return values


def _write_result(
    root: Path,
    candidate_id: str,
    *,
    metrics: dict[str, float],
    result_name: str | None = None,
    input_marker: str = "1",
    non_temporal_variant: bool = False,
    optional_available: bool = True,
) -> Path:
    manifest = json.loads(MANIFEST.read_text())
    declarations = {item["candidate_id"]: item for item in manifest["candidates"]}
    declaration = declarations[candidate_id]
    evidence_root = root / f".{candidate_id}-evidence"
    evidence_root.mkdir(parents=True)
    config = json.loads(APARTMENT_CONFIG.read_text())
    config["temporal_readout"] = declaration["temporal_readout"]
    if non_temporal_variant:
        config["source_stride"] += 1
    config["algorithm_hash"] = canonical_algorithm_hash(config)
    algorithm_hash = config["algorithm_hash"]
    config_path = _write(evidence_root / "config.json", config)
    config_sha256 = hashlib.sha256(_canonical(config)).hexdigest()
    source_bindings = {
        "dataset": input_marker * 64,
        "schedule": "2" * 64,
        "occlusion_targets": "3" * 64,
        "aliases": "4" * 64,
    }
    code_commit = "c" * 40
    stdout_path = evidence_root / "stdout.log"
    stderr_path = evidence_root / "stderr.log"
    stdout_path.write_text("candidate passed\n")
    stderr_path.write_text("")
    run_root = evidence_root / "run"
    normalized = _write(run_root / "normalized_run_config.json", config)
    index = _write(
        run_root / "occlusion_checkpoint_index.json",
        {
            "schema_version": 1,
            "format": "oviv2_temporal_compact_v1",
            "protocol_id": "oviv2-tessecd-v2",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": "apartment",
            "algorithm_hash": algorithm_hash,
            "schedule": {"sha256": "2" * 64, "byte_count": 10},
            "target_manifest": {"sha256": "3" * 64, "byte_count": 11},
            "input_sha256": "5" * 64,
            "code_commit": code_commit,
            "source_bindings": source_bindings,
            "checkpoints": [
                {
                    "frame_index": 2,
                    "consumed_through_frame": 2,
                    "consumed_through_frame_exclusive": 3,
                }
            ],
        },
    )
    schedule_file = evidence_root / "schedule.json"
    schedule_file.write_text("{}\n")
    checkpoint_status = _write(run_root / "checkpoint_status.json", {"status": "PASS"})
    snapshot = evidence_root / "snapshot.npz"
    snapshot.write_bytes(b"snapshot")
    entities = evidence_root / "entities.json"
    entities.write_text("[]\n")
    checkpoint = {
        "frame_index": 2,
        "timestamp_ns": 200,
        "consumed_through_frame": 2,
        "consumed_through_frame_exclusive": 3,
        "checkpoint_status": _record(checkpoint_status),
        "snapshot": _record(snapshot),
        "entities": _record(entities),
    }
    run_source_index = _write(
        run_root / "source_index.json",
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoint_exports",
            "method": "OVIV2",
            "scene": "apartment",
            "schedule": _record(schedule_file),
            "checkpoints": [checkpoint],
        },
    )
    run_manifest = _write(
        run_root / "run_manifest.json",
        {
            "schema_version": 2,
            "protocol_id": "oviv2-tessecd-v2",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": "apartment",
            "mode": "dual_readout_causal_checkpoints",
            "algorithm_hash": algorithm_hash,
            "config": {
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "byte_count": config_path.stat().st_size,
            },
            "normalized_run_config": {
                "path": "normalized_run_config.json",
                "sha256": hashlib.sha256(normalized.read_bytes()).hexdigest(),
                "byte_count": normalized.stat().st_size,
            },
            "schedule": {"sha256": "2" * 64, "byte_count": 10},
            "target_manifest": {"sha256": "3" * 64, "byte_count": 11},
            "source_bindings": source_bindings,
            "input_sha256": "5" * 64,
            "code_commit": code_commit,
            "occlusion_checkpoint_index": {
                "path": "occlusion_checkpoint_index.json",
                "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                "byte_count": index.stat().st_size,
            },
            "source_index": {
                "path": "source_index.json",
                "sha256": hashlib.sha256(run_source_index.read_bytes()).hexdigest(),
                "byte_count": run_source_index.stat().st_size,
            },
            "checkpoints": [checkpoint],
        },
    )
    status = _write(
        evidence_root / "search_status.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "manifest": _record(MANIFEST),
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "scene": "apartment",
                    "status": "PASS",
                    "exit_code": 0,
                    "config_path": str(config_path.resolve()),
                    "config_file": _record(config_path),
                    "config_sha256": config_sha256,
                    "output_root": str(run_root.resolve()),
                    "algorithm_hash": algorithm_hash,
                    "non_temporal_config_sha256": non_temporal_config_sha256(config),
                    "input_binding_values_sha256": input_binding_values_sha256(config),
                    "input_hashes": source_bindings,
                    "stdout_path": str(stdout_path.resolve()),
                    "stdout_file": _record(stdout_path),
                    "stderr_path": str(stderr_path.resolve()),
                    "stderr_file": _record(stderr_path),
                    "runtime_seconds": metrics["runtime_seconds"],
                }
            ],
        },
    )
    export_root = evidence_root / "export"
    export_source_index = _write(
        export_root / "source_index.json",
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoint_exports",
            "method": "OVIV2",
            "scene": "apartment",
            "schedule": _record(schedule_file),
            "checkpoints": [checkpoint],
        },
    )
    temporal_manifest = _write(
        export_root / "temporal_manifest.json",
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoints",
            "method": "OVIV2",
            "scene": "apartment",
            "sources": {"source_index": _record(export_source_index)},
            "checkpoints": [checkpoint],
            "entity_lifecycles": [],
        },
    )
    common = _write(
        evidence_root / "common.json",
        {
            "schema_version": 1,
            "manifest_id": "tesse_cd_common_v2_scene_summary",
            "dataset": "TESSE-CD",
            "protocol": "tesse_cd_common_v2",
            "status": "PASS",
            "method": "OVIV2",
            "mode": "causal_checkpoints",
            "scene": "apartment",
            "metrics": {
                "current_miou": metrics["current_miou"],
                "ghost_rate": metrics["ghost_rate"],
                "background_f5": metrics["background_f5_cm"],
                "recovery_frames": metrics["recovery_frames"],
            },
            "sources": {"temporal_index": _record(temporal_manifest)},
        },
    )
    occlusion = _write(
        evidence_root / "occlusion.json",
        {
            "format": "oviv2_temporal_compact_v1",
            "scene": "apartment",
            "algorithm_hash": algorithm_hash,
            "input_sha256": "5" * 64,
            "code_commit": code_commit,
            "source_bindings": source_bindings,
            "input_bindings": {"indexes": [_record(index)]},
        },
    )
    official_source = evidence_root / "official-source.csv"
    official_source.write_text("header\n")
    official_metrics = {
        "object_f1": metrics["object_f1"],
        "dynamic_f1": metrics["dynamic_f1"] if optional_available else None,
        "change_f1": metrics["change_f1"] if optional_available else None,
    }
    official = _write(
        evidence_root / "official.json",
        {
            "status": "PASS",
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "method": "OVIV2",
            "mode": "causal_checkpoints",
            "run_identity": {
                "run_id": f"{candidate_id}-apartment",
                "config_sha256": config_sha256,
            },
            "metrics": official_metrics,
            "unavailable": (
                {}
                if optional_available
                else {
                    "dynamic_f1": "not_reported_by_official_evaluator",
                    "change_f1": "not_reported_by_official_evaluator",
                }
            ),
            "sources": [_record(official_source)],
        },
    )
    protected = [
        {"path": "src/oviv2/dual_readout.py", "sha256": "7" * 64, "bytes": 123}
    ]
    test_sources = [
        {"path": "tests/oviv2/test_dual_readout.py", "sha256": "8" * 64, "bytes": 456}
    ]

    def gate(name: str, digest: str) -> dict[str, object]:
        return {
            "scope": "shared_code_and_A0-A4_fixture",
            "status": "PASS",
            "code_commit": code_commit,
            "code_tree": "d" * 64,
            "protected_records": protected,
            "test_records": [
                {
                    "argv": ["pytest", "-q", name],
                    "returncode": 0,
                    "stdout_sha256": digest,
                    "stdout_bytes": 10,
                    "stderr_sha256": "0" * 64,
                    "stderr_bytes": 0,
                }
            ],
        }

    evidence = _write(
        evidence_root / "gates.json",
        {
            "schema_version": 1,
            "manifest_id": "oviv2_dual_readout_development_gates_v1",
            "deterministic_evidence": {
                "base_commit": "e" * 40,
                "code_commit": code_commit,
                "code_tree": "d" * 64,
                "protected_files": protected,
                "test_sources": test_sources,
                "gates": {
                    "t1_exact": gate("t1_exact", "9" * 64),
                    "determinism": gate("determinism", "b" * 64),
                },
            },
            "receipt": {"created_at_utc": "2026-07-25T00:00:00Z"},
        },
    )
    output = root / (result_name or f"{candidate_id}.json")
    package_result(
        manifest=MANIFEST,
        search_status=status,
        candidate_id=candidate_id,
        candidate_config=config_path,
        run_manifest=run_manifest,
        common_v2_summary=common,
        temporal_occlusion_result=occlusion,
        official_metrics=official,
        t1_exact_evidence=evidence,
        determinism_evidence=evidence,
        output=output,
    )
    return output


def _write_complete_results(
    root: Path,
    *,
    metrics_by_candidate: dict[str, dict[str, float]] | None = None,
    kwargs_by_candidate: dict[str, dict[str, object]] | None = None,
) -> list[Path]:
    metrics_by_candidate = metrics_by_candidate or {}
    kwargs_by_candidate = kwargs_by_candidate or {}
    return [
        _write_result(
            root,
            candidate_id,
            metrics=metrics_by_candidate.get(candidate_id, _metrics()),
            **kwargs_by_candidate.get(candidate_id, {}),
        )
        for candidate_id in ("a0", "a1", "a2", "a3", "a4")
    ]


def test_tuner_uses_source_backed_gates_and_lexicographic_promotion(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    paths = [
        _write_result(
            results,
            "a0",
            metrics=_metrics(current_miou=0.20, object_f1=0.50, ghost_rate=0.459),
        ),
        _write_result(results, "a1", metrics=_metrics(ghost_rate=0.22)),
        _write_result(results, "a2", metrics=_metrics(ghost_rate=0.14, background_f5_cm=0.17)),
        _write_result(results, "a3", metrics=_metrics(ghost_rate=0.14, background_f5_cm=0.21)),
        _write_result(results, "a4", metrics=_metrics(current_miou=0.19, ghost_rate=0.01)),
    ]
    before = {path: path.read_bytes() for path in paths}
    selection = tune(MANIFEST, paths, tmp_path / "selection.json")

    assert selection["manifest_id"] == "oviv2_tesse_cd_v2_selection"
    assert selection["selected_candidate_id"] == "a3"
    assert selection["algorithm_hash"] == selection["rejection_ledger"][3]["algorithm_hash"]
    assert selection["selected_config"]["temporal_readout"]["execution_profile"] == "a3"
    assert selection["selected_config_record"] == selection["rejection_ledger"][3]["sources"]["candidate_config"]
    ledger = {item["candidate_id"]: item for item in selection["rejection_ledger"]}
    assert "current_miou_below_a0_floor" in ledger["a4"]["reasons"]
    assert "lost_lexicographic_promotion" in ledger["a1"]["reasons"]
    assert all(set(item["sources"]) == {
        "search_manifest", "search_status", "candidate_config", "run_manifest",
        "common_v2_summary", "temporal_occlusion_result", "official_metrics",
        "t1_exact_evidence", "determinism_evidence",
    } for item in selection["rejection_ledger"])
    assert {path: path.read_bytes() for path in paths} == before


def test_a0_is_an_eligible_fallback_and_no_t1_scalarization(tmp_path: Path) -> None:
    paths = _write_complete_results(
        tmp_path,
        metrics_by_candidate={
            candidate_id: _metrics(ghost_rate=0.15 + position * 0.01)
            for position, candidate_id in enumerate(("a0", "a1", "a2", "a3", "a4"))
        },
    )
    selection = tune(MANIFEST, paths, tmp_path / "selection.json")
    assert selection["selected_candidate_id"] == "a0"
    assert selection["rejection_ledger"][0]["reasons"] == []
    assert "score" not in selection

    tampered = json.loads(paths[0].read_text())
    tampered["gates"]["t1_exact"]["passed"] = False
    paths[0].write_bytes(_bytes(tampered))
    with pytest.raises(ValueError, match="does not match revalidated sources"):
        tune(MANIFEST, paths, tmp_path / "tampered.json")


def test_public_tune_requires_every_manifest_candidate(tmp_path: Path) -> None:
    result = _write_result(tmp_path, "a0", metrics=_metrics())
    with pytest.raises(ValueError, match=r"missing=.*a1.*a2.*a3.*a4"):
        tune(MANIFEST, (result,), tmp_path / "selection.json")


def test_tuner_revalidates_sources_immediately_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_complete_results(tmp_path)
    packaged = json.loads(paths[0].read_text())
    common_path = Path(packaged["sources"]["common_v2_summary"]["path"])
    real_loader = tuner.load_and_revalidate_result
    calls = 0

    def mutate_after_first_validation(
        path: str | Path, *, manifest: dict[str, object]
    ) -> dict[str, object]:
        nonlocal calls
        loaded = real_loader(path, manifest=manifest)
        calls += 1
        if calls == 1:
            common_path.write_bytes(common_path.read_bytes() + b" ")
        return loaded

    monkeypatch.setattr(tuner, "load_and_revalidate_result", mutate_after_first_validation)
    output = tmp_path / "selection.json"
    with pytest.raises(ValueError, match=r"source record .*mismatch"):
        tune(MANIFEST, paths, output)
    assert calls == 5
    assert not output.exists()


def test_optional_metric_availability_matches_a0_or_fails_closed(tmp_path: Path) -> None:
    unavailable = _write_complete_results(
        tmp_path / "unavailable",
        kwargs_by_candidate={
            candidate_id: {"optional_available": False}
            for candidate_id in ("a0", "a1", "a2", "a3", "a4")
        },
    )
    selection = tune(MANIFEST, unavailable, tmp_path / "selection.json")
    assert selection["skipped_optional_tie_axes"] == {
        "change_f1": "all_candidates_unavailable_matching_a0",
        "dynamic_f1": "all_candidates_unavailable_matching_a0",
    }
    assert all(item["notes"] for item in selection["rejection_ledger"])

    partial = _write_complete_results(
        tmp_path / "partial",
        kwargs_by_candidate={
            candidate_id: {"optional_available": candidate_id == "a2"}
            for candidate_id in ("a0", "a1", "a2", "a3", "a4")
        },
    )
    with pytest.raises(ValueError, match="availability differs"):
        tune(MANIFEST, partial, tmp_path / "partial.json")


@pytest.mark.parametrize(
    ("candidate_kwargs", "match"),
    [
        ({"non_temporal_variant": True}, "non-temporal"),
        ({"input_marker": "9"}, "input hash"),
    ],
)
def test_tuner_rejects_non_temporal_or_input_binding_drift(
    tmp_path: Path, candidate_kwargs: dict[str, object], match: str
) -> None:
    paths = _write_complete_results(
        tmp_path,
        kwargs_by_candidate={"a1": candidate_kwargs},
    )
    with pytest.raises(ValueError, match=match):
        tune(MANIFEST, paths, tmp_path / "selection.json")


def test_stable_read_rejects_oversized_file_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"123456789")
    monkeypatch.setattr(
        tuner.os,
        "read",
        lambda descriptor, count: pytest.fail("oversized file content was read"),
    )
    with pytest.raises(ValueError, match="exceeds 8 byte limit"):
        tuner._stable_bytes(oversized, "packaged candidate result", max_bytes=8)


@pytest.mark.parametrize("failure", ["open", "fsync", "close"])
def test_atomic_publish_removes_destination_after_directory_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    output = tmp_path / "selection.json"
    real_open = tuner.os.open
    real_fsync = tuner.os.fsync
    real_close = tuner.os.close
    fsync_calls = 0
    close_calls = 0

    def failing_open(path: str | Path, flags: int, *args: object) -> int:
        if failure == "open" and Path(path) == output.parent and flags == tuner.os.O_RDONLY:
            raise OSError("injected directory open failure")
        return real_open(path, flags, *args)

    def failing_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if failure == "fsync" and fsync_calls == 2:
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    def failing_close(descriptor: int) -> None:
        nonlocal close_calls
        close_calls += 1
        real_close(descriptor)
        if failure == "close" and close_calls == 2:
            raise OSError("injected directory close failure")

    monkeypatch.setattr(tuner.os, "open", failing_open)
    monkeypatch.setattr(tuner.os, "fsync", failing_fsync)
    monkeypatch.setattr(tuner.os, "close", failing_close)
    with pytest.raises(OSError, match=f"directory {failure} failure"):
        tuner._atomic_write_new(output, {"status": "PASS"})
    assert not output.exists()
    assert not list(tmp_path.glob(".selection.json.tmp-*"))


def test_results_root_uses_exact_bound_paths_and_requires_all_candidates(tmp_path: Path) -> None:
    root = tmp_path / "search"
    for candidate_id in ("a0", "a1", "a2", "a3", "a4"):
        result_dir = root / "candidates" / candidate_id / "apartment"
        result_dir.mkdir(parents=True)
        _write_result(
            result_dir,
            candidate_id,
            metrics=_metrics(),
            result_name="result.json",
        )
    selection = tune_results_root(MANIFEST, root, tmp_path / "selection.json")
    assert selection["result_contract"] == "candidates/<candidate_id>/apartment/result.json"
    assert [Path(item["path"]).relative_to(root).as_posix() for item in selection["result_files"]] == [
        f"candidates/{candidate_id}/apartment/result.json"
        for candidate_id in ("a0", "a1", "a2", "a3", "a4")
    ]

    (root / "candidates/a4/apartment/result.json").unlink()
    with pytest.raises(FileNotFoundError, match="candidates/a4/apartment/result.json"):
        tune_results_root(MANIFEST, root, tmp_path / "missing.json")


def test_tuner_rejects_naked_duplicate_nonfinite_office_and_overwrite(tmp_path: Path) -> None:
    valid = _write_result(tmp_path, "a0", metrics=_metrics())
    occupied = tmp_path / "occupied.json"
    occupied.write_text("occupied")
    with pytest.raises(FileExistsError):
        tune(MANIFEST, (valid,), occupied)

    naked = tmp_path / "naked.json"
    naked.write_text('{"candidate_id":"a0","metrics":{},"gates":{}}\n')
    with pytest.raises(ValueError, match="schema is not exact"):
        tune(MANIFEST, (naked,), tmp_path / "naked-selection.json")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"candidate_id":"a0","candidate_id":"a1"}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        tune(MANIFEST, (duplicate,), tmp_path / "duplicate-selection.json")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"candidate_id":"a0","metric":NaN}\n')
    with pytest.raises(ValueError, match="non-finite"):
        tune(MANIFEST, (nonfinite,), tmp_path / "nonfinite-selection.json")

    office = json.loads(valid.read_text())
    office["scene"] = "office"
    valid.write_bytes(_bytes(office))
    with pytest.raises(ValueError, match="does not match revalidated sources"):
        tune(MANIFEST, (valid,), tmp_path / "office-selection.json")
