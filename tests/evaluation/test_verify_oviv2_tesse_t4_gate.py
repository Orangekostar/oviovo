from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.measure_oviv2_tesse_t4 import measure_t4
from scripts.evaluation.verify_oviv2_tesse_t4_gate import (
    T4GateError,
    T4PublicationUncertain,
    verify_t4_gate,
)
from tests.evaluation.test_measure_oviv2_tesse_t4 import _fixture, _sha256


BOUNDS = {
    "total_runtime_s_per_frame": 6.42,
    "query_mean_ms": 11.92,
    "query_p95_ms": 12.12,
    "peak_gpu_gb": 12.76,
    "peak_ram_gb": 9.36,
    "final_map_mb": 46.77,
}


def _assert_uncertain_cause(call, match: str) -> None:
    with pytest.raises(T4PublicationUncertain, match="preserved") as raised:
        call()
    assert isinstance(raised.value.__cause__, T4GateError)
    assert re.search(match, str(raised.value.__cause__))
    assert raised.value.preserved


def _record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path.absolute()), "sha256": hashlib.sha256(data).hexdigest(), "byte_count": len(data)}


def _run_record(path: Path, run_manifest: Path) -> dict[str, object]:
    return {
        **_record(path),
        "path": path.relative_to(run_manifest.parent).as_posix(),
    }


def _valid_t4_evidence(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    import scripts.evaluation.measure_oviv2_tesse_t4 as collector

    fixture = _fixture(tmp_path)
    evidence = measure_t4(fixture)
    protocol = json.loads((Path(__file__).resolve().parents[2] / "configs/evaluation/manifests/oviv2_tesse_t4_v1.json").read_text())
    query = json.loads(Path(fixture["query_measurements"]).read_text())
    protocol["query"]["vocabulary_path"] = query["sources"]["queries"]["path"]
    protocol["query"]["vocabulary_sha256"] = query["sources"]["queries"]["sha256"]
    protocol["query"]["checkpoint_sha256"] = query["sources"]["checkpoint"]["sha256"]
    Path(fixture["protocol"]).write_text(json.dumps(protocol, sort_keys=True) + "\n", encoding="utf-8")
    query["query_protocol"] = _record(Path(fixture["protocol"]))
    Path(fixture["query_measurements"]).write_text(json.dumps(query, sort_keys=True) + "\n", encoding="utf-8")
    evidence["sources"]["query_measurements"] = _record(Path(fixture["query_measurements"]))
    evidence["sources"]["query_measurements_sha256"] = _sha256(Path(fixture["query_measurements"]))
    collector.FROZEN_PROTOCOL_PATH = Path(fixture["protocol"])
    evidence["sources"]["protocol"] = _record(Path(fixture["protocol"]))
    evidence["sources"]["protocol_sha256"] = _sha256(Path(fixture["protocol"]))
    Path(fixture["output"]).unlink()
    Path(fixture["output"]).write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
    return evidence, fixture


def test_verifier_recomputes_sources_and_emits_exact_downstream_schema(tmp_path: Path) -> None:
    evidence, fixture = _valid_t4_evidence(tmp_path)
    matrix_path = tmp_path / "matrix.json"
    matrix = verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), matrix_path)

    assert set(matrix) == {"schema_version", "manifest_id", "status", "shortlist", "protocol", "candidates", "root_sha256"}
    assert matrix["candidates"]["a4"]["metrics"]["total_runtime_s_per_frame"] == 2.0
    protocol = json.loads(Path(matrix["protocol"]["path"]).read_text())
    assert set(protocol) == {"schema_version", "manifest_id", "dataset", "method_id", "protocol_id", "scene", "bounds", "candidates"}
    assert set(protocol["candidates"]["a4"]["metric_sources"]) == set(BOUNDS)
    raw_sources = protocol["candidates"]["a4"]["raw_sources"]
    source_names = {
        "config", "run_manifest", "time_log", "gpu_samples", "query_measurements",
        "final_map_inventory", "protocol", "shortlist",
    }
    assert set(raw_sources) == source_names | {f"{name}_sha256" for name in source_names}
    assert all(raw_sources[f"{name}_sha256"] == raw_sources[name]["sha256"] for name in source_names)


def test_verifier_rejects_metric_without_raw_source(tmp_path: Path) -> None:
    evidence, fixture = _valid_t4_evidence(tmp_path)
    evidence["sources"].pop("gpu_samples")
    Path(fixture["output"]).write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
    _assert_uncertain_cause(lambda: verify_t4_gate(evidence), "gpu samples")


def test_verifier_requires_every_flat_source_hash(tmp_path: Path) -> None:
    evidence, fixture = _valid_t4_evidence(tmp_path)
    evidence["sources"].pop("time_log_sha256")
    Path(fixture["output"]).write_text(json.dumps(evidence), encoding="utf-8")
    _assert_uncertain_cause(
        lambda: verify_t4_gate(
            [Path(fixture["output"])], Path(fixture["shortlist"]), tmp_path / "matrix.json"
        ),
        "time log hash binding",
    )


def test_verifier_rejects_raw_source_drift_and_cross_run_splice(tmp_path: Path) -> None:
    evidence, fixture = _valid_t4_evidence(tmp_path / "drift")
    Path(fixture["time_log"]).write_text("changed", encoding="utf-8")
    _assert_uncertain_cause(
        lambda: verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), tmp_path / "drift-matrix.json"),
        "time log.*binding",
    )

    evidence, fixture = _valid_t4_evidence(tmp_path / "splice")
    run = json.loads(Path(fixture["run_manifest"]).read_text())
    normalized_path = Path(fixture["run_manifest"]).parent / run["normalized_run_config"]["path"]
    normalized = json.loads(normalized_path.read_text())
    normalized["temporal_readout"]["execution_profile"] = "a3"
    normalized_path.write_text(json.dumps(normalized), encoding="utf-8")
    run["normalized_run_config"] = {
        "path": normalized_path.name, **{key: _record(normalized_path)[key] for key in ("sha256", "byte_count")}
    }
    Path(fixture["run_manifest"]).write_text(json.dumps(run), encoding="utf-8")
    evidence["sources"]["run_manifest"] = _record(Path(fixture["run_manifest"]))
    evidence["sources"]["run_manifest_sha256"] = _sha256(Path(fixture["run_manifest"]))
    Path(fixture["output"]).write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
    _assert_uncertain_cause(
        lambda: verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), tmp_path / "splice-matrix.json"),
        "whole-profile",
    )


def test_verifier_rejects_shortlist_config_and_final_map_splice(tmp_path: Path) -> None:
    evidence, fixture = _valid_t4_evidence(tmp_path / "config")
    shortlist = json.loads(Path(fixture["shortlist"]).read_text())
    shortlist["shortlisted_candidates"][0]["config_sha256"] = "0" * 64
    Path(fixture["shortlist"]).write_text(json.dumps(shortlist), encoding="utf-8")
    evidence["sources"]["shortlist"] = _record(Path(fixture["shortlist"]))
    evidence["sources"]["shortlist_sha256"] = _sha256(Path(fixture["shortlist"]))
    Path(fixture["output"]).write_text(json.dumps(evidence), encoding="utf-8")
    _assert_uncertain_cause(
        lambda: verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), tmp_path / "config-matrix.json"),
        "shortlist config",
    )

    evidence, fixture = _valid_t4_evidence(tmp_path / "profile")
    shortlist = json.loads(Path(fixture["shortlist"]).read_text())
    shortlist["shortlisted_candidates"][0]["profile"] = "a3"
    shortlist["shortlisted_candidates"][0]["algorithm_hash"] = "f" * 64
    Path(fixture["shortlist"]).write_text(json.dumps(shortlist), encoding="utf-8")
    evidence["sources"]["shortlist"] = _record(Path(fixture["shortlist"]))
    evidence["sources"]["shortlist_sha256"] = _sha256(Path(fixture["shortlist"]))
    Path(fixture["output"]).write_text(json.dumps(evidence), encoding="utf-8")
    _assert_uncertain_cause(
        lambda: verify_t4_gate(
            [Path(fixture["output"])], Path(fixture["shortlist"]),
            tmp_path / "profile-matrix.json",
        ),
        "shortlist config candidate identity",
    )

    evidence, fixture = _valid_t4_evidence(tmp_path / "map")
    run_manifest = Path(fixture["run_manifest"])
    run = json.loads(run_manifest.read_text())
    snapshot = run_manifest.parent / run["final_current_map"]["snapshot"]["path"]
    snapshot.write_bytes(b"not-an-npz")
    run["final_current_map"]["snapshot"] = _run_record(snapshot, run_manifest)
    query = json.loads(Path(fixture["query_measurements"]).read_text())
    query["sources"]["snapshot"] = _record(snapshot)
    Path(fixture["query_measurements"]).write_text(json.dumps(query), encoding="utf-8")
    inventory_path = Path(evidence["sources"]["final_map_inventory"]["path"])
    inventory = json.loads(inventory_path.read_text())
    inventory["files"][0] = {"role": "snapshot", **_record(snapshot)}
    inventory["total_bytes"] = sum(item["byte_count"] for item in inventory["files"])
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    Path(fixture["run_manifest"]).write_text(json.dumps(run), encoding="utf-8")
    evidence["sources"]["run_manifest"] = _record(Path(fixture["run_manifest"]))
    evidence["sources"]["run_manifest_sha256"] = _sha256(Path(fixture["run_manifest"]))
    evidence["sources"]["query_measurements"] = _record(Path(fixture["query_measurements"]))
    evidence["sources"]["query_measurements_sha256"] = _sha256(Path(fixture["query_measurements"]))
    evidence["sources"]["final_map_inventory"] = _record(inventory_path)
    evidence["sources"]["final_map_inventory_sha256"] = _sha256(inventory_path)
    Path(fixture["output"]).write_text(json.dumps(evidence), encoding="utf-8")
    _assert_uncertain_cause(
        lambda: verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), tmp_path / "map-matrix.json"),
        "final current map",
    )


def test_verifier_requires_full_gpu_and_query_raw_identity(tmp_path: Path) -> None:
    evidence, fixture = _valid_t4_evidence(tmp_path / "gpu")
    gpu = json.loads(Path(fixture["gpu_samples"]).read_text())
    gpu["samples"][0].pop("processes")
    Path(fixture["gpu_samples"]).write_text(json.dumps(gpu), encoding="utf-8")
    evidence["sources"]["gpu_samples"] = _record(Path(fixture["gpu_samples"]))
    evidence["sources"]["gpu_samples_sha256"] = _sha256(Path(fixture["gpu_samples"]))
    Path(fixture["output"]).write_text(json.dumps(evidence), encoding="utf-8")
    _assert_uncertain_cause(
        lambda: verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), tmp_path / "gpu-matrix.json"),
        "process inventory",
    )

    evidence, fixture = _valid_t4_evidence(tmp_path / "query")
    query = json.loads(Path(fixture["query_measurements"]).read_text())
    query.pop("baseline")
    Path(fixture["query_measurements"]).write_text(json.dumps(query), encoding="utf-8")
    evidence["sources"]["query_measurements"] = _record(Path(fixture["query_measurements"]))
    evidence["sources"]["query_measurements_sha256"] = _sha256(Path(fixture["query_measurements"]))
    Path(fixture["output"]).write_text(json.dumps(evidence), encoding="utf-8")
    _assert_uncertain_cause(
        lambda: verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), tmp_path / "query-matrix.json"),
        "query evidence",
    )


def test_verifier_rejects_nested_runner_identity_and_hardlink_alias(tmp_path: Path) -> None:
    evidence, fixture = _valid_t4_evidence(tmp_path / "nested")
    actual_path = Path(fixture["run_manifest"])
    actual = json.loads(actual_path.read_text())
    normalized_path = actual_path.parent / actual["normalized_run_config"]["path"]
    normalized = json.loads(normalized_path.read_text())
    normalized["temporal_readout"]["execution_profile"] = "a3"
    normalized_path.write_text(json.dumps(normalized), encoding="utf-8")
    actual["normalized_run_config"] = {
        "path": normalized_path.name,
        **{key: _record(normalized_path)[key] for key in ("sha256", "byte_count")},
    }
    actual_path.write_text(json.dumps(actual), encoding="utf-8")
    wrapper_path = actual_path.parent.parent / "measurement-run.json"
    wrapper = {
        "schema_version": 1, "manifest_id": "oviv2_tesse_t4_measurement_run_v1",
        "dataset": "TESSE-CD", "method_id": "OVIV2", "protocol_id": "oviv2-tessecd-v2",
        "scene": "apartment", "candidate_id": "a4", "config_sha256": evidence["config_sha256"],
        "processed_frame_count": 4, "collection_protocol": evidence["sources"]["protocol"],
        "shortlist": evidence["sources"]["shortlist"], "candidate_config": evidence["sources"]["config"],
        "runner_manifest": _record(actual_path), "time_log": evidence["sources"]["time_log"],
        "gpu_samples": evidence["sources"]["gpu_samples"],
        "query_measurements": evidence["sources"]["query_measurements"],
        "final_map_inventory": evidence["sources"]["final_map_inventory"],
    }
    wrapper_path.write_text(json.dumps(wrapper), encoding="utf-8")
    evidence["sources"]["run_manifest"] = _record(wrapper_path)
    evidence["sources"]["run_manifest_sha256"] = _sha256(wrapper_path)
    Path(fixture["output"]).write_text(json.dumps(evidence), encoding="utf-8")
    _assert_uncertain_cause(
        lambda: verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), tmp_path / "nested-matrix.json"),
        "whole-profile",
    )

    evidence, fixture = _valid_t4_evidence(tmp_path / "alias")
    inventory_path = Path(evidence["sources"]["final_map_inventory"]["path"])
    inventory = json.loads(inventory_path.read_text())
    snapshot = Path(inventory["files"][0]["path"])
    entities = Path(inventory["files"][1]["path"])
    entities.unlink()
    entities.hardlink_to(snapshot)
    inventory["files"][1] = {"role": "entities", **_record(entities)}
    inventory["total_bytes"] = sum(item["byte_count"] for item in inventory["files"])
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    run = json.loads(Path(fixture["run_manifest"]).read_text())
    run["final_current_map"]["entities"] = _run_record(
        entities, Path(fixture["run_manifest"])
    )
    query = json.loads(Path(fixture["query_measurements"]).read_text())
    query["sources"]["entities"] = _record(entities)
    Path(fixture["query_measurements"]).write_text(json.dumps(query), encoding="utf-8")
    Path(fixture["run_manifest"]).write_text(json.dumps(run), encoding="utf-8")
    evidence["sources"]["final_map_inventory"] = _record(inventory_path)
    evidence["sources"]["final_map_inventory_sha256"] = _sha256(inventory_path)
    evidence["sources"]["run_manifest"] = _record(Path(fixture["run_manifest"]))
    evidence["sources"]["run_manifest_sha256"] = _sha256(Path(fixture["run_manifest"]))
    evidence["sources"]["query_measurements"] = _record(Path(fixture["query_measurements"]))
    evidence["sources"]["query_measurements_sha256"] = _sha256(Path(fixture["query_measurements"]))
    Path(fixture["output"]).write_text(json.dumps(evidence), encoding="utf-8")
    _assert_uncertain_cause(
        lambda: verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), tmp_path / "alias-matrix.json"),
        "same inode",
    )


@pytest.mark.parametrize("metric", list(BOUNDS))
def test_every_upper_bound_is_enforced_from_raw_source(tmp_path: Path, metric: str) -> None:
    evidence, fixture = _valid_t4_evidence(tmp_path / metric)
    if metric == "total_runtime_s_per_frame":
        Path(fixture["time_log"]).write_text("Elapsed (wall clock) time (h:mm:ss or m:ss): 0:26.00\nMaximum resident set size (kbytes): 2500000\n", encoding="utf-8")
    elif metric == "peak_ram_gb":
        Path(fixture["time_log"]).write_text("Elapsed (wall clock) time (h:mm:ss or m:ss): 0:08.00\nMaximum resident set size (kbytes): 10000000\n", encoding="utf-8")
    elif metric in {"query_mean_ms", "query_p95_ms"}:
        query = json.loads(Path(fixture["query_measurements"]).read_text())
        query["latencies_ms"] = [13.0] * 5
        for sample in query["samples"]:
            sample["latency_ms"] = 13.0
        query["query_mean_ms"] = query["query_p50_ms"] = query["query_p95_ms"] = 13.0
        query["summary"] = {"query_mean_ms": 13.0, "query_p50_ms": 13.0, "query_p95_ms": 13.0}
        Path(fixture["query_measurements"]).write_text(json.dumps(query), encoding="utf-8")
    elif metric == "peak_gpu_gb":
        gpu = json.loads(Path(fixture["gpu_samples"]).read_text())
        gpu["samples"][0]["used_memory_mib"] = 13_000
        gpu["samples"][0]["processes"][0]["used_memory_mib"] = 13_000
        Path(fixture["gpu_samples"]).write_text(json.dumps(gpu), encoding="utf-8")
    else:
        inventory = json.loads(Path(evidence["sources"]["final_map_inventory"]["path"]).read_text())
        snapshot = Path(inventory["files"][0]["path"])
        np.savez(
            snapshot,
            background_xyz=np.arange(47_000_000, dtype=np.uint8),
            timestamp=np.asarray(4_000_000_000.0),
            scope=np.asarray("current"),
            scene_id=np.asarray("apartment"),
        )
        inventory["files"][0] = {"role": "snapshot", **_record(snapshot)}
        inventory["total_bytes"] = sum(item["byte_count"] for item in inventory["files"])
        Path(evidence["sources"]["final_map_inventory"]["path"]).write_text(json.dumps(inventory), encoding="utf-8")
        query = json.loads(Path(fixture["query_measurements"]).read_text())
        query["sources"]["snapshot"] = _record(snapshot)
        Path(fixture["query_measurements"]).write_text(json.dumps(query), encoding="utf-8")
        run = json.loads(Path(fixture["run_manifest"]).read_text())
        run["final_current_map"]["snapshot"] = _run_record(
            snapshot, Path(fixture["run_manifest"])
        )
        Path(fixture["run_manifest"]).write_text(json.dumps(run), encoding="utf-8")
        evidence["sources"]["query_measurements"] = _record(Path(fixture["query_measurements"]))
        evidence["sources"]["query_measurements_sha256"] = _sha256(Path(fixture["query_measurements"]))
        evidence["sources"]["run_manifest"] = _record(Path(fixture["run_manifest"]))
        evidence["sources"]["run_manifest_sha256"] = _sha256(Path(fixture["run_manifest"]))

    source_key = {"total_runtime_s_per_frame": "time_log", "peak_ram_gb": "time_log", "query_mean_ms": "query_measurements", "query_p95_ms": "query_measurements", "peak_gpu_gb": "gpu_samples", "final_map_mb": "final_map_inventory"}[metric]
    source = Path(evidence["sources"][source_key]["path"])
    evidence["sources"][source_key] = _record(source)
    evidence["sources"][f"{source_key}_sha256"] = _sha256(source)
    Path(fixture["output"]).write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
    matrix = verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), tmp_path / f"{metric}-matrix.json")
    assert matrix["candidates"]["a4"]["gates"][metric] is False


def test_verifier_no_clobber(tmp_path: Path) -> None:
    _, fixture = _valid_t4_evidence(tmp_path)
    output = tmp_path / "matrix.json"
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), output)


def test_verifier_staging_failure_preserves_artifact_and_retry_uses_new_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.verify_oviv2_tesse_t4_gate as verifier

    _, fixture = _valid_t4_evidence(tmp_path)
    output = tmp_path / "matrix.json"
    original = verifier._write_staged_bytes
    calls = 0

    def fail_once(path: Path, data: bytes, *, parent_fd: int | None = None) -> None:
        nonlocal calls
        calls += 1
        original(path, data, parent_fd=parent_fd)
        if calls == 1:
            raise OSError("injected staging failure")

    monkeypatch.setattr(verifier, "_write_staged_bytes", fail_once)
    before_fds = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(verifier.T4PublicationUncertain, match="preserved") as raised:
        verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), output)
    assert isinstance(raised.value.__cause__, OSError)
    assert not output.exists()
    assert not (tmp_path / "matrix.protocol.json").exists()
    assert not (tmp_path / "matrix.sources").exists()
    staging = list(tmp_path.glob(".matrix.sources.staging-*"))
    assert len(staging) == 1
    original_inode = staging[0].stat().st_ino
    record = next(item for item in raised.value.preserved if item.inode == original_inode)
    assert record.name == staging[0].name and record.ownership == "owned"
    assert len(list(Path("/proc/self/fd").iterdir())) == before_fds

    monkeypatch.setattr(verifier, "_write_staged_bytes", original)
    verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), output)
    assert output.is_file()
    assert staging[0].stat().st_ino == original_inode


def test_verifier_mkdir_wrapper_failure_reports_unbound_preserved_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.verify_oviv2_tesse_t4_gate as verifier

    _, fixture = _valid_t4_evidence(tmp_path / "evidence")
    output = tmp_path / "matrix.json"
    original_mkdir = verifier.os.mkdir

    def create_then_fail(path, *args, **kwargs):
        original_mkdir(path, *args, **kwargs)
        if isinstance(path, str) and path.startswith(".matrix.sources.staging-"):
            verifier.os.chmod(path, 0o750, dir_fd=kwargs["dir_fd"])
            raise OSError("injected post-mkdir failure")

    monkeypatch.setattr(verifier.os, "mkdir", create_then_fail)
    before_fds = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(verifier.T4PublicationUncertain, match="preserved") as raised:
        verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), output)
    assert isinstance(raised.value.__cause__, OSError)
    assert "post-mkdir" in str(raised.value.__cause__)
    staging = list(tmp_path.glob(".matrix.sources.staging-*"))
    assert len(staging) == 1
    record = next(item for item in raised.value.preserved if item.name == staging[0].name)
    assert record.ownership == "unbound"
    assert record.inode == staging[0].stat().st_ino
    assert record.mode == 0o40750
    assert staging[0].is_dir()
    assert len(list(Path("/proc/self/fd").iterdir())) == before_fds


def test_verifier_mkdir_eexist_preserves_competitor_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.verify_oviv2_tesse_t4_gate as verifier

    _, fixture = _valid_t4_evidence(tmp_path / "evidence")
    output = tmp_path / "matrix.json"
    original_mkdir = verifier.os.mkdir
    competitor_inode = None

    def competitor_wins(path, *args, **kwargs):
        nonlocal competitor_inode
        if isinstance(path, str) and path.startswith(".matrix.sources.staging-"):
            original_mkdir(path, *args, **kwargs)
            competitor = Path(f"/proc/self/fd/{kwargs['dir_fd']}") / path
            (competitor / "marker").write_text("competitor", encoding="utf-8")
            competitor_inode = competitor.stat().st_ino
            raise FileExistsError(verifier.errno.EEXIST, "injected competitor", path)
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(verifier.os, "mkdir", competitor_wins)
    before_fds = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(FileExistsError, match="competitor") as raised:
        verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), output)
    assert raised.value.__cause__ is None
    staging = list(tmp_path.glob(".matrix.sources.staging-*"))
    assert len(staging) == 1
    assert (staging[0] / "marker").read_text(encoding="utf-8") == "competitor"
    assert staging[0].stat().st_ino == competitor_inode
    assert len(list(Path("/proc/self/fd").iterdir())) == before_fds


def test_verifier_does_not_unlink_racing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.verify_oviv2_tesse_t4_gate as verifier

    _, fixture = _valid_t4_evidence(tmp_path)
    output = tmp_path / "matrix.json"
    original = verifier._write_new

    def race(path: Path, value: object, *, parent_fd: int | None = None) -> None:
        if path.absolute() == output.absolute():
            output.write_text("racer-owned\n", encoding="utf-8")
        original(path, value, parent_fd=parent_fd)

    monkeypatch.setattr(verifier, "_write_new", race)
    with pytest.raises(verifier.T4PublicationUncertain) as raised:
        verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), output)
    assert isinstance(raised.value.__cause__, FileExistsError)
    assert output.read_text(encoding="utf-8") == "racer-owned\n"


def test_verifier_failure_preserves_replacement_and_owned_renamed_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.verify_oviv2_tesse_t4_gate as verifier

    _, fixture = _valid_t4_evidence(tmp_path)
    output = tmp_path / "matrix.json"
    source_dir = tmp_path / "matrix.sources"
    moved = tmp_path / "owned-sources-moved"
    original = verifier._write_new

    def replace_then_fail(path: Path, value: object, *, parent_fd: int | None = None):
        if path.name == "matrix.protocol.json":
            source_dir.rename(moved)
            source_dir.mkdir()
            (source_dir / "replacement").write_text("replacement", encoding="utf-8")
            raise OSError("injected replacement")
        return original(path, value, parent_fd=parent_fd)

    monkeypatch.setattr(verifier, "_write_new", replace_then_fail)
    with pytest.raises(verifier.T4PublicationUncertain, match="preserved") as raised:
        verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), output)
    assert (source_dir / "replacement").read_text(encoding="utf-8") == "replacement"
    assert moved.is_dir()
    assert not output.exists()
    identities = {(item.name, item.ownership, item.inode) for item in raised.value.preserved}
    assert (moved.name, "owned", moved.stat().st_ino) in identities
    assert (source_dir.name, "unknown", source_dir.stat().st_ino) in identities


def test_verifier_parent_swap_fails_without_publishing_to_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.verify_oviv2_tesse_t4_gate as verifier

    _, fixture = _valid_t4_evidence(tmp_path / "evidence")
    parent = tmp_path / "publish"
    parent.mkdir()
    moved = tmp_path / "publish-moved"
    output = parent / "matrix.json"
    original = verifier._write_staged_bytes
    swapped = False

    def swap_parent(path: Path, data: bytes, *, parent_fd=None) -> None:
        nonlocal swapped
        original(path, data, parent_fd=parent_fd)
        if not swapped:
            swapped = True
            parent.rename(moved)
            parent.mkdir()
            (parent / "replacement").write_text("replacement", encoding="utf-8")

    monkeypatch.setattr(verifier, "_write_staged_bytes", swap_parent)
    before_fds = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(verifier.T4PublicationUncertain, match="preserved") as raised:
        verify_t4_gate([Path(fixture["output"])], Path(fixture["shortlist"]), output)
    assert (parent / "replacement").read_text(encoding="utf-8") == "replacement"
    assert not output.exists()
    staging = list(moved.glob(".matrix.sources.staging-*"))
    assert len(staging) == 1
    assert next(item for item in raised.value.preserved if item.inode == staging[0].stat().st_ino)
    assert len(list(Path("/proc/self/fd").iterdir())) == before_fds
