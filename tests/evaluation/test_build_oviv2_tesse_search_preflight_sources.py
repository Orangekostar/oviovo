from __future__ import annotations

import inspect
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.evaluation import build_oviv2_tesse_search_preflight_sources as module
from scripts.evaluation.build_oviv2_tesse_search_preflight_sources import (
    _audit_future_leakage,
    _select_candidate_executions,
    build_preflight_sources,
)


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def _record(path: Path, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _absolute_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "byte_count": len(data)}


def _tree_record(path: Path, root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    total = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        data = item.read_bytes()
        total += len(data)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
        digest.update(b"\n")
    return {"path": path.relative_to(root).as_posix(), "sha256": digest.hexdigest(), "byte_count": total}


def _causal_fixture(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    coverage = root / "temporal_frame_coverage.jsonl"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    coverage.write_text("".join(
        json.dumps({"frame_index": frame, "timestamp_ns": 100 + frame, "record_count": 1, "event_count": 0}, sort_keys=True, separators=(",", ":")) + "\n"
        for frame in range(5)
    ))
    trajectories = root / "trajectories.jsonl"
    trajectories.write_text("".join(
        json.dumps({"frame_index": frame, "timestamp_ns": 100 + frame}, sort_keys=True, separators=(",", ":")) + "\n"
        for frame in range(5)
    ))
    frontend = root / "frontend"
    dense = root / "dense"
    frontend.mkdir(); dense.mkdir()
    frontend_hashes: dict[str, str] = {}
    dense_hashes: dict[str, str] = {}
    for frame in range(5):
        front = frontend / f"frame{frame:06d}.pkl.gz"
        front.write_bytes(f"frontend-{frame}".encode())
        frontend_hashes[front.name] = hashlib.sha256(front.read_bytes()).hexdigest()
        dense_path = dense / f"frame{frame:06d}.npz"
        np.savez(dense_path, cache_frame_id=np.asarray(frame), source_frame_id=np.asarray(frame))
        dense_hashes[dense_path.name] = hashlib.sha256(dense_path.read_bytes()).hexdigest()
    frontend_manifest = _write_json(frontend / "frontend_manifest.json", {
        "frame_count": 5, "source_frame_ids": list(range(5)), "cache_files_sha256": frontend_hashes,
    })
    dense_manifest = _write_json(dense / "dense_manifest.json", {
        "frame_count": 5, "source_frame_ids": list(range(5)), "cache_files_sha256": dense_hashes,
    })
    source_bindings = {
        "dataset": "f" * 64,
        "frontend_manifest": {"sha256": hashlib.sha256(frontend_manifest.read_bytes()).hexdigest(), "byte_count": frontend_manifest.stat().st_size},
        "dense_manifest": {"sha256": hashlib.sha256(dense_manifest.read_bytes()).hexdigest(), "byte_count": dense_manifest.stat().st_size},
    }
    config = _write_json(root / "normalized_run_config.json", {
        "frontend_manifest": str(frontend_manifest), "dense_manifest": str(dense_manifest),
    })
    status = _write_json(root / "checkpoints/00000004-104/checkpoint_status.json", {
        "schema_version": 1, "status": "PASS", "checkpoint_frame": 4,
        "timestamp_ns": 104, "event_ids": [], "roles": ["occlusion_v1"],
        "consumed_through_frame": 4, "consumed_through_frame_exclusive": 5,
    })
    compact = root / "checkpoints/00000004-104/temporal_compact"
    metadata = _write_json(compact / "metadata.json", {
        "frame_id": 4, "revision": 5, "timestamp": 104 / 1e9,
    })
    (compact / "ownership.npz").write_bytes(b"ownership")
    _write_json(compact / "checksums.json", {
        "metadata.json": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "ownership.npz": hashlib.sha256((compact / "ownership.npz").read_bytes()).hexdigest(),
    })
    source_index_path = root / "source_index.json"
    source_index: dict[str, Any] = {
        "schema_version": 1, "dataset": "TESSE-CD", "mode": "causal_checkpoint_exports",
        "method": "OVIV2", "scene": "apartment", "frame_coverage": _record(coverage, root),
        "trajectories": _record(trajectories, root),
        "checkpoints": [{"frame_index": 4, "timestamp_ns": 104,
            "consumed_through_frame": 4, "consumed_through_frame_exclusive": 5,
            "checkpoint_status": _record(status, root)}],
    }
    _write_json(source_index_path, source_index)
    checkpoint_index: dict[str, Any] = {
        "schema_version": 1, "format": "oviv2_temporal_compact_v1", "protocol_id": "oviv2-tessecd-v2",
        "dataset": "TESSE-CD", "method_id": "OVIV2", "scene": "apartment", "algorithm_hash": "a" * 64,
        "schedule": {"sha256": "b" * 64, "byte_count": 1}, "target_manifest": {"sha256": "c" * 64, "byte_count": 1},
        "input_sha256": "d" * 64, "code_commit": "e" * 40, "source_bindings": source_bindings,
        "checkpoints": [{"scene": "apartment", "frame_index": 4, "timestamp_ns": 104,
            "relative_timestamp_ns": 4, "consumed_through_frame": 4,
            "consumed_through_frame_exclusive": 5, "event_ids": [], "roles": ["occlusion_v1"],
            "format": "oviv2_temporal_compact_v1", "maximum_entities": 1,
            "maximum_object_voxels": 1, "artifact": _tree_record(compact, root),
            "checksums_sha256": hashlib.sha256((compact / "checksums.json").read_bytes()).hexdigest()}],
    }
    checkpoint_path = _write_json(root / "occlusion_checkpoint_index.json", checkpoint_index)
    run: dict[str, Any] = {
        "processed_frame_count": 5, "covered_frame_count": 5, "first_frame_index": 0,
        "last_frame_index": 4, "scheduled_frame_indices": [4], "captured_frame_indices": [4],
        "algorithm_hash": "a" * 64, "schedule": checkpoint_index["schedule"],
        "target_manifest": checkpoint_index["target_manifest"], "input_sha256": "d" * 64,
        "code_commit": "e" * 40, "source_bindings": source_bindings,
        "normalized_run_config": _record(config, root), "source_index": _record(source_index_path, root),
        "occlusion_checkpoint_index": _record(checkpoint_path, root),
        "checkpoints": [{"frame_index": 4, "timestamp_ns": 104,
            "consumed_through_frame": 4, "consumed_through_frame_exclusive": 5,
            "checkpoint_status": _record(status, root), "artifacts": {
                "temporal_compact": {"artifact": _tree_record(compact, root),
                    "checksums_sha256": hashlib.sha256((compact / "checksums.json").read_bytes()).hexdigest()}}}],
    }
    return run, source_index, checkpoint_index


def test_public_builder_has_exact_keyword_only_api() -> None:
    parameters = inspect.signature(build_preflight_sources).parameters
    assert list(parameters) == [
        "development_evidence",
        "search_manifest",
        "apartment_base_config",
        "targets",
        "dataset_root",
        "output",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )
    assert all(
        parameter.annotation == "str | Path" for parameter in parameters.values()
    )
    assert inspect.signature(build_preflight_sources).return_annotation == "dict[str, Any]"


def test_reverifies_exact_transaction_and_selects_fixed_candidate_positions() -> None:
    sequence = ["reference", "a0", "a1", "a0", "a2", "a0", "a3", "a0", "a4"]
    executions = [
        {"profile": profile, "output_root": f"/run/{position}"}
        for position, profile in enumerate(sequence)
    ]
    exact = {
        "format": "oviv2_t1_exact_transaction_v1",
        "sequence": sequence,
        "executions": executions,
        "profiles": {},
    }
    calls: list[list[dict[str, object]]] = []

    def verify(records: list[dict[str, object]]) -> dict[str, object]:
        calls.append(records)
        return exact

    selected = _select_candidate_executions(exact, verifier=verify)

    assert calls == [executions]
    assert {name: record["output_root"] for name, record in selected.items()} == {
        "a0": "/run/1",
        "a1": "/run/2",
        "a2": "/run/4",
        "a3": "/run/6",
        "a4": "/run/8",
    }


@pytest.mark.parametrize("drift", ("sequence", "position", "reverified"))
def test_rejects_drifted_exact_transaction(drift: str) -> None:
    sequence = ["reference", "a0", "a1", "a0", "a2", "a0", "a3", "a0", "a4"]
    executions = [{"profile": profile} for profile in sequence]
    exact: dict[str, Any] = {
        "format": "oviv2_t1_exact_transaction_v1",
        "sequence": list(sequence),
        "executions": executions,
        "profiles": {},
    }
    if drift == "sequence":
        exact["sequence"][2] = "a2"
    if drift == "position":
        executions[4]["profile"] = "a3"

    def verify(_: list[dict[str, object]]) -> dict[str, Any]:
        return {**exact, "profiles": {"changed": {}}} if drift == "reverified" else exact

    with pytest.raises(ValueError, match="exact transaction"):
        _select_candidate_executions(exact, verifier=verify)


def test_future_leakage_audit_returns_exact_empty_evidence_after_full_audit(tmp_path: Path) -> None:
    run, source_index, checkpoint_index = _causal_fixture(tmp_path)

    result = _audit_future_leakage(
        candidate_id="a4", run_root=tmp_path, run=run,
        source_index=source_index, checkpoint_index=checkpoint_index,
    )

    assert result == {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_future_leakage_evidence_v1",
        "scene": "apartment",
        "candidate_id": "a4",
        "source_index": run["source_index"],
        "records": [],
    }


def test_future_leakage_accepts_distinct_evaluation_schedule_and_official_inventories(
    tmp_path: Path,
) -> None:
    run, source_index, checkpoint_index = _causal_fixture(tmp_path)
    run["scheduled_frame_indices"] = [1, 2, 3, 4]
    run["captured_frame_indices"] = [1, 2, 3, 4]
    source_index["checkpoints"] = []

    result = _audit_future_leakage(
        candidate_id="a4", run_root=tmp_path, run=run,
        source_index=source_index, checkpoint_index=checkpoint_index,
    )

    assert result["records"] == []


def test_audit_witnesses_retain_only_file_identity(tmp_path: Path) -> None:
    source = tmp_path / "large-cache.bin"
    source.write_bytes(b"x" * (2 * 1024 * 1024))
    captured: list[object] = []
    token = module._AUDIT_WITNESSES.set(captured)
    try:
        snapshot = module._snapshot(source, "cache")
    finally:
        module._AUDIT_WITNESSES.reset(token)

    assert snapshot.data == source.read_bytes()
    assert len(captured) == 1
    assert not hasattr(captured[0], "data")


def test_development_source_witness_rejects_protected_or_commit_drift(
    tmp_path: Path,
) -> None:
    manifest = _write_json(tmp_path / "source_manifest.json", {"files": {}})
    snapshot = module._snapshot(manifest, "source manifest")
    protected = [{"path": "protected.py", "sha256": "a" * 64, "bytes": 1}]
    deterministic = {
        "code_commit": "b" * 40,
        "code_tree": "c" * 40,
        "protected_files": protected,
    }

    witness = module._verify_development_sources(
        deterministic, snapshot, repo=tmp_path,
        verifier=lambda payload, **kwargs: protected,
        code_identity=lambda repo: ("b" * 40, "c" * 40),
    )
    witness.revalidate()

    with pytest.raises(ValueError, match="protected source"):
        module._verify_development_sources(
            deterministic, snapshot, repo=tmp_path,
            verifier=lambda payload, **kwargs: [],
            code_identity=lambda repo: ("b" * 40, "c" * 40),
        )
    with pytest.raises(ValueError, match="code commit/tree"):
        module._verify_development_sources(
            deterministic, snapshot, repo=tmp_path,
            verifier=lambda payload, **kwargs: protected,
            code_identity=lambda repo: ("d" * 40, "c" * 40),
        )


def test_staging_tree_witness_rejects_member_rewrite(tmp_path: Path) -> None:
    stage = tmp_path / ".bundle.staging"
    stage.mkdir()
    member = stage / "preflight.json"
    member.write_bytes(b"original")
    descriptor = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
    try:
        witness = module._StagingTreeWitness.capture(stage, descriptor)
        member.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="staged bundle changed"):
            witness.revalidate()
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "violation",
    (
        "future_cache_frame", "coverage_gap", "coverage_timestamp_drift",
        "checkpoint_consumed_boundary", "compact_frame", "compact_revision",
        "compact_timestamp", "checkpoint_source_binding",
    ),
)
def test_future_leakage_audit_reports_deterministic_causal_detail(
    tmp_path: Path, violation: str
) -> None:
    run, source_index, checkpoint_index = _causal_fixture(tmp_path)
    if violation == "future_cache_frame":
        dense_path = tmp_path / "dense/frame000004.npz"
        np.savez(dense_path, cache_frame_id=np.asarray(4), source_frame_id=np.asarray(5))
        manifest_path = tmp_path / "dense/dense_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["cache_files_sha256"][dense_path.name] = hashlib.sha256(dense_path.read_bytes()).hexdigest()
        _write_json(manifest_path, manifest)
    elif violation.startswith("coverage_"):
        coverage_path = tmp_path / "temporal_frame_coverage.jsonl"
        rows = [json.loads(line) for line in coverage_path.read_text().splitlines()]
        if violation == "coverage_gap":
            del rows[2]
        else:
            rows[3]["timestamp_ns"] = 999
        coverage_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
        source_index["frame_coverage"] = _record(coverage_path, tmp_path)
    elif violation == "checkpoint_consumed_boundary":
        checkpoint_index["checkpoints"][0]["consumed_through_frame"] = 3
        run["checkpoints"][0]["consumed_through_frame"] = 3
        source_index["checkpoints"][0]["consumed_through_frame"] = 3
        status_path = tmp_path / source_index["checkpoints"][0]["checkpoint_status"]["path"]
        status = json.loads(status_path.read_text())
        status["consumed_through_frame"] = 3
        _write_json(status_path, status)
        record = _record(status_path, tmp_path)
        run["checkpoints"][0]["checkpoint_status"] = record
        source_index["checkpoints"][0]["checkpoint_status"] = record
    elif violation.startswith("compact_"):
        metadata_path = tmp_path / "checkpoints/00000004-104/temporal_compact/metadata.json"
        metadata = json.loads(metadata_path.read_text())
        key = {"compact_frame": "frame_id", "compact_revision": "revision", "compact_timestamp": "timestamp"}[violation]
        metadata[key] = {"frame_id": 3, "revision": 4, "timestamp": 999 / 1e9}[key]
        _write_json(metadata_path, metadata)
        compact = metadata_path.parent
        checksums_path = compact / "checksums.json"
        checksums = json.loads(checksums_path.read_text())
        checksums["metadata.json"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        _write_json(checksums_path, checksums)
        artifact = _tree_record(compact, tmp_path)
        digest = hashlib.sha256(checksums_path.read_bytes()).hexdigest()
        checkpoint_index["checkpoints"][0]["artifact"] = artifact
        checkpoint_index["checkpoints"][0]["checksums_sha256"] = digest
        run["checkpoints"][0]["artifacts"]["temporal_compact"] = {
            "artifact": artifact, "checksums_sha256": digest,
        }
    else:
        checkpoint_index["source_bindings"] = {"dataset": "0" * 64}

    with pytest.raises(ValueError) as raised:
        _audit_future_leakage(
            candidate_id="a4", run_root=tmp_path, run=run,
            source_index=source_index, checkpoint_index=checkpoint_index,
        )
    detail = str(raised.value)
    assert "candidate=a4" in detail
    assert "role=" in detail
    assert "observed=" in detail
    assert "bound=" in detail


def test_builds_byte_identical_independent_a0_a4_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.evaluation import build_oviv2_tesse_search_preflight_sources as module

    source_manifest = _write_json(tmp_path / "source_manifest.json", {"files": {}})
    roots: dict[str, Path] = {}
    selected = {"a0": 1, "a1": 2, "a2": 4, "a3": 6, "a4": 8}
    sequence = ["reference", "a0", "a1", "a0", "a2", "a0", "a3", "a0", "a4"]
    executions: list[dict[str, Any]] = []
    originals: dict[tuple[str, str], bytes] = {}
    for position, profile in enumerate(sequence):
        root = tmp_path / f"run-{position}"
        root.mkdir()
        if position in selected.values():
            roots[profile] = root
        role_records: dict[str, Any] = {}
        for role, relative in (
            ("trajectories", "records/trajectories.jsonl"),
            ("lifecycle_transitions", "records/lifecycle.jsonl"),
            ("frame_coverage", "records/coverage.jsonl"),
            ("runtime_diagnostics", "records/runtime.json"),
        ):
            path = root / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"{position}:{role}\n".encode())
            role_records[role] = _record(path, root)
            if selected.get(profile) == position:
                originals[(profile, relative)] = path.read_bytes()
        source_index_path = _write_json(root / "source_index.json", {
            "schema_version": 1, "dataset": "TESSE-CD", "method": "OVIV2", "scene": "apartment",
            **role_records,
        })
        checkpoint_index_path = _write_json(root / "occlusion_checkpoint_index.json", {
            "schema_version": 1, "candidate": profile,
        })
        run_manifest_path = _write_json(root / "run_manifest.json", {
            "schema_version": 2, "dataset": "TESSE-CD", "protocol_id": "oviv2-tessecd-v2",
            "scene": "apartment", "method_id": "OVIV2", "algorithm_hash": "a" * 64,
            "code_commit": "e" * 40, "source_index": _record(source_index_path, root),
            "occlusion_checkpoint_index": _record(checkpoint_index_path, root),
        })
        config = _write_json(tmp_path / f"config-{position}.json", {"scene": "apartment", "algorithm_hash": "a" * 64})
        observation = _write_json(tmp_path / f"observation-{position}.json", {
            "profile": profile, "output_root": str(root.resolve()),
            "run_manifest": _absolute_record(run_manifest_path), "config": _absolute_record(config),
            "source_manifest": _absolute_record(source_manifest),
        })
        executions.append({
            "profile": profile, "output_root": str(root.resolve()),
            "code_commit": "e" * 40,
            "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            "observation_receipt": _absolute_record(observation),
        })
        if selected.get(profile) == position:
            originals[(profile, "run_manifest.json")] = run_manifest_path.read_bytes()
            originals[(profile, "source_index.json")] = source_index_path.read_bytes()
    exact = {"format": "oviv2_t1_exact_transaction_v1", "sequence": sequence, "executions": executions, "profiles": {}}
    evidence = _write_json(tmp_path / "development.json", {
        "schema_version": 1, "manifest_id": "oviv2_dual_readout_development_gates_v1",
        "deterministic_evidence": {"base_commit": "b" * 40, "code_commit": "e" * 40,
            "code_tree": "t" * 40, "protected_files": [], "test_sources": [],
            "source_manifest": _absolute_record(source_manifest), "cumulative_exact": exact, "gates": {}},
        "receipt": {"created_at_utc": "2026-07-27T00:00:00Z"},
    })
    search = _write_json(tmp_path / "search.json", {"candidates": [{"candidate_id": name} for name in selected]})
    base = _write_json(tmp_path / "base.json", {"scene": "apartment"})
    targets = _write_json(tmp_path / "targets.json", {"scene": "apartment"})
    dataset = tmp_path / "dataset"; dataset.mkdir()

    monkeypatch.setattr(module, "verify_exact_profile_runs", lambda records: exact)
    monkeypatch.setattr(module, "_TRUSTED_SOURCE_MANIFEST", source_manifest)
    monkeypatch.setattr(
        module, "_verify_development_sources",
        lambda *args, **kwargs: module._FileWitness(
            source_manifest, module._identity(os.stat(source_manifest))
        ),
    )
    monkeypatch.setattr(module, "_materialize_config", lambda base, declaration: {"scene": "apartment", "algorithm_hash": "a" * 64})
    monkeypatch.setattr(module, "_audit_future_leakage", lambda **kwargs: {
        "schema_version": 1, "manifest_id": "oviv2_tesse_future_leakage_evidence_v1",
        "scene": "apartment", "candidate_id": kwargs["candidate_id"],
        "source_index": kwargs["run"]["source_index"], "records": [],
    })
    def evaluate(*, output: Path, **_: object) -> dict[str, Any]:
        result = {"anchor_mappings": [], "macro": {"anchor_coverage_gate": {
            "eligible_count": 66, "uniquely_mapped_count": 53,
        }}}
        _write_json(output, result)
        return result
    monkeypatch.setattr(module, "evaluate_temporal_occlusion_package", evaluate)
    monkeypatch.setattr(module, "_anchor_coverage", lambda result: {"eligible_count": 66, "mapped_count": 53})
    def preflight(*, candidate_sources: Path, output: Path, **_: object) -> dict[str, Any]:
        sources = json.loads(candidate_sources.read_text())
        assert [item["candidate_id"] for item in sources["candidates"]] == list(selected)
        result = {"candidates": []}
        _write_json(output, result)
        return result
    monkeypatch.setattr(module, "build_preflight", preflight)
    monkeypatch.setattr(module, "_validate_preflight_gate_evidence", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_revalidate_preflight_witnesses", lambda record: None)
    output = tmp_path / "bundle"

    result = build_preflight_sources(
        development_evidence=evidence, search_manifest=search,
        apartment_base_config=base, targets=targets, dataset_root=dataset, output=output,
    )

    assert result["manifest_id"] == "oviv2_tesse_search_preflight_sources_v1"
    assert [item["candidate_id"] for item in result["candidates"]] == list(selected)
    for candidate_id in selected:
        for (profile, relative), content in originals.items():
            if profile == candidate_id:
                assert (output / candidate_id / relative).read_bytes() == content
    assert (output / "candidate_sources.json").is_file()
    assert (output / "preflight.json").is_file()
    assert (output / "publication_receipt.json").is_file()

    kwargs = {
        "development_evidence": evidence, "search_manifest": search,
        "apartment_base_config": base, "targets": targets,
        "dataset_root": dataset,
    }
    marker = output / "keep"
    marker.write_text("keep\n")
    with pytest.raises(FileExistsError):
        build_preflight_sources(**kwargs, output=output)
    assert marker.read_text() == "keep\n"

    baseline_fds = len(list(Path("/proc/self/fd").iterdir()))
    real_fsync = module.os.fsync
    with monkeypatch.context() as patcher:
        patcher.setattr(module.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync failed")))
        with pytest.raises(module.PreflightPublicationUncertain) as raised:
            build_preflight_sources(**kwargs, output=tmp_path / "fsync-failure")
        assert raised.value.preserved[0].ownership == "owned"
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds

    anchor_failure = tmp_path / "anchor-52"
    with monkeypatch.context() as patcher:
        patcher.setattr(module, "_anchor_coverage", lambda result: {"eligible_count": 66, "mapped_count": 52})
        with pytest.raises(module.PreflightPublicationUncertain) as raised:
            build_preflight_sources(**kwargs, output=anchor_failure)
        assert "53/66" in str(raised.value.__cause__)
    assert not anchor_failure.exists()
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds

    with monkeypatch.context() as patcher:
        real_mkdir = module.os.mkdir
        def create_stage_then_fail(path: object, *args: object, **kwargs: object) -> None:
            real_mkdir(path, *args, **kwargs)
            if str(path).startswith(".post-mkdir.staging-"):
                raise OSError("post mkdir failure")
        patcher.setattr(module.os, "mkdir", create_stage_then_fail)
        with pytest.raises(module.PreflightPublicationUncertain) as raised:
            build_preflight_sources(**kwargs, output=tmp_path / "post-mkdir")
        assert raised.value.preserved[0].ownership == "unknown"
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds

    collision = tmp_path / "collision"
    with monkeypatch.context() as patcher:
        def collide(parent_fd: int, source: str, destination: str) -> None:
            os.mkdir(destination, dir_fd=parent_fd)
            raise FileExistsError(destination)
        patcher.setattr(module, "_rename_noreplace", collide)
        with pytest.raises(FileExistsError):
            build_preflight_sources(**kwargs, output=collision)
    assert collision.is_dir()
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds

    post_rename = tmp_path / "post-rename-fsync"
    renamed = False
    real_rename = module._rename_noreplace
    with monkeypatch.context() as patcher:
        def rename_then_mark(parent_fd: int, source: str, destination: str) -> None:
            nonlocal renamed
            real_rename(parent_fd, source, destination)
            renamed = True
        def fail_after_rename(fd: int) -> None:
            if renamed:
                raise OSError("parent fsync failed")
            real_fsync(fd)
        patcher.setattr(module, "_rename_noreplace", rename_then_mark)
        patcher.setattr(module.os, "fsync", fail_after_rename)
        with pytest.raises(module.PreflightPublicationUncertain) as raised:
            build_preflight_sources(**kwargs, output=post_rename)
        assert raised.value.preserved[0].logical_path == str(post_rename)
        assert raised.value.preserved[0].ownership == "owned"
    assert post_rename.is_dir()
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds
