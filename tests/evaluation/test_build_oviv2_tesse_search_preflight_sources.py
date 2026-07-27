from __future__ import annotations

import inspect
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
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
    lifecycle = root / "lifecycle_transitions.jsonl"
    lifecycle.write_text('{"entity_id":1,"frame_index":1,"timestamp_ns":101}\n')
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
    official_status = _write_json(root / "checkpoints/00000001-101/checkpoint_status.json", {
        "schema_version": 1, "status": "PASS", "checkpoint_frame": 1,
        "timestamp_ns": 101, "event_ids": ["official"], "roles": ["official_v1"],
        "consumed_through_frame": 1, "consumed_through_frame_exclusive": 2,
    })
    official_snapshot = root / "checkpoints/00000001-101/neutral_snapshot.npz"
    official_snapshot.write_bytes(b"official-snapshot")
    official_entities = root / "checkpoints/00000001-101/neutral_entities.jsonl"
    official_entities.write_bytes(b'{"entity_id":1}\n')
    schedule_copy = root / "inputs/schedule.json"
    schedule_copy.parent.mkdir()
    schedule_copy.write_bytes(b"x")
    schedule_record = _record(schedule_copy, root)
    official_source = {"frame_index": 1, "timestamp_ns": 101,
        "consumed_through_frame": 1, "consumed_through_frame_exclusive": 2,
        "checkpoint_status": _record(official_status, root),
        "snapshot": _record(official_snapshot, root),
        "entities": _record(official_entities, root)}
    capture_status = _write_json(root / "capture_status.json", {
        "schema_version": 1, "status": "PASS", "scene": "apartment",
        "mode": "causal_checkpoints", "scheduled_frame_indices": [1],
        "captured_frame_indices": [1], "schedule": schedule_record,
        "trajectories": _record(trajectories, root),
        "frame_coverage": _record(coverage, root),
        "lifecycle_transitions": _record(lifecycle, root),
        "checkpoint_statuses": [official_source["checkpoint_status"]],
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
        "lifecycle_transitions": _record(lifecycle, root),
        "schedule": schedule_record, "capture_status": _record(capture_status, root),
        "checkpoints": [official_source],
    }
    _write_json(source_index_path, source_index)
    checkpoint_index: dict[str, Any] = {
        "schema_version": 1, "format": "oviv2_temporal_compact_v1", "protocol_id": "oviv2-tessecd-v2",
        "dataset": "TESSE-CD", "method_id": "OVIV2", "scene": "apartment", "algorithm_hash": "a" * 64,
        "schedule": {"sha256": schedule_record["sha256"], "byte_count": 1}, "target_manifest": {"sha256": "c" * 64, "byte_count": 1},
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
        "last_frame_index": 4, "scheduled_frame_indices": [1, 4], "captured_frame_indices": [1, 4],
        "algorithm_hash": "a" * 64, "schedule": checkpoint_index["schedule"],
        "target_manifest": checkpoint_index["target_manifest"], "input_sha256": "d" * 64,
        "code_commit": "e" * 40, "source_bindings": source_bindings,
        "normalized_run_config": _record(config, root), "source_index": _record(source_index_path, root),
        "occlusion_checkpoint_index": _record(checkpoint_path, root),
        "checkpoints": [{"frame_index": 1, "timestamp_ns": 101,
            "consumed_through_frame": 1, "consumed_through_frame_exclusive": 2,
            "checkpoint_status": _record(official_status, root),
            "neutral_snapshot": _record(official_snapshot, root),
            "neutral_entities": _record(official_entities, root)},
            {"frame_index": 4, "timestamp_ns": 104,
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

    assert [item["frame_index"] for item in checkpoint_index["checkpoints"]] == [4]
    assert [item["frame_index"] for item in source_index["checkpoints"]] == [1]
    assert [item["frame_index"] for item in run["checkpoints"]] == [1, 4]

    result = _audit_future_leakage(
        candidate_id="a4", run_root=tmp_path, run=run,
        source_index=source_index, checkpoint_index=checkpoint_index,
    )

    assert result["records"] == []


@pytest.mark.parametrize(
    "violation",
    (
        "empty_official", "missing_official_snapshot", "official_sidecar_tamper",
        "run_union_gap", "schedule_union_gap",
    ),
)
def test_future_leakage_rejects_incomplete_three_inventory_transaction(
    tmp_path: Path, violation: str,
) -> None:
    run, source_index, checkpoint_index = _causal_fixture(tmp_path)
    if violation == "empty_official":
        source_index["checkpoints"] = []
        run["checkpoints"] = run["checkpoints"][1:]
        run["scheduled_frame_indices"] = [4]
        run["captured_frame_indices"] = [4]
    elif violation == "missing_official_snapshot":
        source_index["checkpoints"][0].pop("snapshot")
    elif violation == "official_sidecar_tamper":
        path = tmp_path / source_index["checkpoints"][0]["entities"]["path"]
        path.write_bytes(b'{"entity_id":2}\n')
    elif violation == "run_union_gap":
        run["checkpoints"] = run["checkpoints"][1:]
    else:
        run["scheduled_frame_indices"] = [4]

    with pytest.raises(ValueError, match="future leakage violation"):
        _audit_future_leakage(
            candidate_id="a4", run_root=tmp_path, run=run,
            source_index=source_index, checkpoint_index=checkpoint_index,
        )


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


@pytest.mark.parametrize(
    ("role", "renamed"),
    (("frontend_manifest", "renamed_frontend.json"),
     ("dense_manifest", "renamed_dense.json")),
)
def test_future_leakage_rejects_renamed_cache_manifest_with_updated_bindings(
    tmp_path: Path, role: str, renamed: str,
) -> None:
    run, source_index, checkpoint_index = _causal_fixture(tmp_path)
    config_path = tmp_path / run["normalized_run_config"]["path"]
    config = json.loads(config_path.read_text())
    original = Path(config[role])
    replacement = original.with_name(renamed)
    original.rename(replacement)
    config[role] = str(replacement)
    _write_json(config_path, config)
    run["normalized_run_config"] = _record(config_path, tmp_path)

    with pytest.raises(ValueError, match=role):
        _audit_future_leakage(
            candidate_id="a4", run_root=tmp_path, run=run,
            source_index=source_index, checkpoint_index=checkpoint_index,
        )


def test_future_leakage_accepts_reordered_exact_cache_hash_inventory(
    tmp_path: Path,
) -> None:
    run, source_index, checkpoint_index = _causal_fixture(tmp_path)
    for role in ("frontend_manifest", "dense_manifest"):
        config_path = tmp_path / run["normalized_run_config"]["path"]
        config = json.loads(config_path.read_text())
        manifest_path = Path(config[role])
        manifest = json.loads(manifest_path.read_text())
        manifest["cache_files_sha256"] = dict(
            reversed(list(manifest["cache_files_sha256"].items()))
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=False, separators=(",", ":")) + "\n"
        )
        run["source_bindings"][role] = {
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "byte_count": manifest_path.stat().st_size,
        }

    _audit_future_leakage(
        candidate_id="a4", run_root=tmp_path, run=run,
        source_index=source_index, checkpoint_index=checkpoint_index,
    )


@pytest.mark.parametrize("mutation", ("missing", "extra", "wrong"))
def test_future_leakage_rejects_nonexact_cache_hash_inventory(
    tmp_path: Path, mutation: str,
) -> None:
    run, source_index, checkpoint_index = _causal_fixture(tmp_path)
    config_path = tmp_path / run["normalized_run_config"]["path"]
    config = json.loads(config_path.read_text())
    manifest_path = Path(config["frontend_manifest"])
    manifest = json.loads(manifest_path.read_text())
    hashes = manifest["cache_files_sha256"]
    removed = hashes.pop("frame000004.pkl.gz")
    if mutation == "extra":
        hashes["frame000004.pkl.gz"] = removed
        hashes["extra.pkl.gz"] = "f" * 64
    elif mutation == "wrong":
        hashes["wrong.pkl.gz"] = removed
    _write_json(manifest_path, manifest)
    run["source_bindings"]["frontend_manifest"] = {
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "byte_count": manifest_path.stat().st_size,
    }

    with pytest.raises(ValueError, match="cache_inventory"):
        _audit_future_leakage(
            candidate_id="a4", run_root=tmp_path, run=run,
            source_index=source_index, checkpoint_index=checkpoint_index,
        )


def test_producer_cli_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/build_oviv2_tesse_search_preflight_sources.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


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


def test_fd_relative_tree_stays_on_owned_inode_after_stage_name_swap(
    tmp_path: Path,
) -> None:
    stage = tmp_path / ".bundle.staging"
    stage.mkdir()
    descriptor = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        module._write_bytes_at(descriptor, Path("owned/data.bin"), b"owned")
        witness = module._StagingTreeWitness.capture_at(
            descriptor, parent_fd, stage.name
        )
        stage.rename(tmp_path / ".owned-preserved")
        stage.mkdir()
        (stage / "attacker").write_bytes(b"attacker")

        inventory, _ = module._root_inventory_at(descriptor)
        assert [item["path"] for item in inventory] == ["owned/data.bin"]
        assert module._read_bytes_at(
            descriptor, Path("owned/data.bin"), "owned member"
        ) == b"owned"
        with pytest.raises(ValueError, match="staged bundle changed"):
            witness.revalidate()
    finally:
        os.close(parent_fd)
        os.close(descriptor)


def test_fd_relative_write_handles_partial_os_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_write = module.os.write
    try:
        monkeypatch.setattr(
            module.os, "write",
            lambda fd, data: real_write(fd, data[: max(1, len(data) // 3)]),
        )
        module._write_bytes_at(descriptor, Path("partial.bin"), b"0123456789")
    finally:
        os.close(descriptor)
    assert (tmp_path / "partial.bin").read_bytes() == b"0123456789"


def test_fd_relative_write_holds_parent_directory_across_file_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    displaced = tmp_path / "displaced"
    original_open_directory = module._open_directory_at
    swapped = False

    def swap_parent(root_fd: int, relative: Path, *, create: bool) -> int:
        nonlocal swapped
        parent_fd = original_open_directory(root_fd, relative, create=create)
        if create and relative == Path("candidate") and not swapped:
            (root / "candidate").rename(displaced)
            (root / "candidate").symlink_to(outside, target_is_directory=True)
            swapped = True
        return parent_fd

    monkeypatch.setattr(module, "_open_directory_at", swap_parent)
    baseline_fds = len(list(Path("/proc/self/fd").iterdir()))
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        module._write_bytes_at(
            descriptor, Path("candidate/result.json"), b'{"safe":true}\n'
        )
    finally:
        os.close(descriptor)

    assert swapped
    assert not (outside / "result.json").exists()
    assert not (root / "candidate/result.json").exists()
    assert (displaced / "result.json").read_bytes() == b'{"safe":true}\n'
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds


def test_open_directory_at_closes_each_owned_fd_once_when_close_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "child").mkdir()
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    baseline = {int(path.name) for path in Path("/proc/self/fd").iterdir()}
    real_close = module._RAW_CLOSE
    closed: list[int] = []

    def close_then_raise(descriptor: int) -> None:
        if descriptor in closed:
            raise AssertionError(f"descriptor closed twice: {descriptor}")
        closed.append(descriptor)
        real_close(descriptor)
        if len(closed) == 1:
            raise OSError("close current failed")

    monkeypatch.setattr(module, "_RAW_CLOSE", close_then_raise)
    try:
        with pytest.raises(OSError, match="close current failed"):
            module._open_directory_at(root_fd, Path("child"), create=False)
        after_failure_fds = len(list(Path("/proc/self/fd").iterdir()))
    finally:
        monkeypatch.setattr(module, "_RAW_CLOSE", real_close)
        for path in Path("/proc/self/fd").iterdir():
            descriptor = int(path.name)
            if descriptor not in baseline and descriptor != root_fd:
                try:
                    real_close(descriptor)
                except OSError:
                    pass
        real_close(root_fd)
    assert len(closed) == len(set(closed)) == 2
    assert after_failure_fds == len(baseline)
    assert len(list(Path("/proc/self/fd").iterdir())) == len(baseline) - 1


def test_open_directory_at_raw_closes_same_fd_when_close_raises_before_syscall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "child").mkdir()
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    baseline_fds = len(list(Path("/proc/self/fd").iterdir()))
    real_close = module._RAW_CLOSE
    attempts: list[int] = []

    def fail_before_close(descriptor: int) -> None:
        if descriptor in attempts:
            raise AssertionError(f"descriptor close retried through os.close: {descriptor}")
        attempts.append(descriptor)
        if len(attempts) == 1:
            raise OSError("close failed before syscall")
        real_close(descriptor)

    monkeypatch.setattr(module, "_RAW_CLOSE", fail_before_close)
    try:
        with pytest.raises(OSError, match="before syscall"):
            module._open_directory_at(root_fd, Path("child"), create=False)
        assert attempts
        os.fstat(attempts[0])
        after_failure_fds = len(list(Path("/proc/self/fd").iterdir()))
    finally:
        monkeypatch.setattr(module, "_RAW_CLOSE", real_close)
        if attempts:
            real_close(attempts[0])
        real_close(root_fd)

    assert len(attempts) == len(set(attempts)) == 2
    assert after_failure_fds == baseline_fds + 1
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds - 1


def test_open_directory_at_does_not_close_same_inode_reused_fd_after_close_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "child").mkdir()
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    baseline_fds = len(list(Path("/proc/self/fd").iterdir()))
    real_close = module._RAW_CLOSE
    real_open = module.os.open
    reused_fd: int | None = None
    first = True

    def close_reuse_then_raise(descriptor: int) -> None:
        nonlocal first, reused_fd
        if first:
            first = False
            real_close(descriptor)
            reused_fd = os.dup(root_fd)
            assert reused_fd == descriptor
            raise OSError("close failed after fd reuse")
        real_close(descriptor)

    monkeypatch.setattr(module, "_RAW_CLOSE", close_reuse_then_raise)
    try:
        with pytest.raises(OSError, match="after fd reuse"):
            module._open_directory_at(root_fd, Path("child"), create=False)
        assert reused_fd is not None
        os.fstat(reused_fd)
        assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds + 1
    finally:
        monkeypatch.setattr(module, "_RAW_CLOSE", real_close)
        if reused_fd is not None:
            real_close(reused_fd)
        real_close(root_fd)

    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds - 1


def test_open_directory_at_closes_duplicate_when_child_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "child").mkdir()
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    baseline_fds = len(list(Path("/proc/self/fd").iterdir()))
    real_open = module.os.open

    def fail_open(path: object, *args: object, **kwargs: object) -> int:
        if os.fspath(path) == "child":
            raise OSError("child open failed")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", fail_open)
    try:
        with pytest.raises(OSError, match="child open failed"):
            module._open_directory_at(root_fd, Path("child"), create=False)
    finally:
        monkeypatch.setattr(module.os, "open", real_open)
        os.close(root_fd)

    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds - 1


def test_open_directory_at_does_not_depend_on_raw_fstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "child").mkdir()
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    baseline_fds = len(list(Path("/proc/self/fd").iterdir()))
    descriptor = -1
    monkeypatch.setattr(
        module, "_RAW_FSTAT",
        lambda fd: (_ for _ in ()).throw(OSError("raw fstat unavailable")),
    )
    try:
        descriptor = module._open_directory_at(
            root_fd, Path("child"), create=False
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(root_fd)
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds - 1


def test_open_directory_at_preserves_fstat_error_and_closes_all_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "child").mkdir()
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    baseline = {int(path.name) for path in Path("/proc/self/fd").iterdir()}
    real_close = module._RAW_CLOSE
    real_fstat = module.os.fstat
    opened: list[int] = []
    closed: list[int] = []
    real_open = module.os.open

    def record_open(*args: object, **kwargs: object) -> int:
        descriptor = real_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def fail_fstat(descriptor: int) -> os.stat_result:
        if descriptor in opened:
            raise OSError("fstat failed")
        return real_fstat(descriptor)

    def close_after_error(descriptor: int) -> None:
        if descriptor in closed:
            raise AssertionError(f"descriptor closed twice: {descriptor}")
        closed.append(descriptor)
        real_close(descriptor)
        if descriptor == opened[-1]:
            raise OSError("cleanup close failed")

    monkeypatch.setattr(module.os, "open", record_open)
    monkeypatch.setattr(module.os, "fstat", fail_fstat)
    monkeypatch.setattr(module, "_RAW_CLOSE", close_after_error)
    try:
        with pytest.raises(OSError, match="fstat failed"):
            module._open_directory_at(root_fd, Path("child"), create=False)
        after_failure_fds = len(list(Path("/proc/self/fd").iterdir()))
    finally:
        monkeypatch.setattr(module.os, "open", real_open)
        monkeypatch.setattr(module.os, "fstat", real_fstat)
        monkeypatch.setattr(module, "_RAW_CLOSE", real_close)
        for path in Path("/proc/self/fd").iterdir():
            descriptor = int(path.name)
            if descriptor not in baseline and descriptor != root_fd:
                try:
                    real_close(descriptor)
                except OSError:
                    pass
        real_close(root_fd)
    assert len(closed) == len(set(closed)) == 2
    assert after_failure_fds == len(baseline)
    assert len(list(Path("/proc/self/fd").iterdir())) == len(baseline) - 1


@pytest.mark.parametrize("fault", ("stat", "open", "fstat"))
def test_staging_creation_failures_preserve_identity_and_close_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    baseline_fds = len(list(Path("/proc/self/fd").iterdir()))
    real_stat = module.os.stat
    real_open = module.os.open
    real_fstat = module.os.fstat
    stage_fd: int | None = None

    def fail_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if fault == "stat" and os.fspath(path) == "stage":
            raise OSError("stat failed")
        return real_stat(path, *args, **kwargs)

    def fail_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal stage_fd
        if fault == "open" and os.fspath(path) == "stage":
            raise OSError("open failed")
        descriptor = real_open(path, flags, *args, **kwargs)
        if os.fspath(path) == "stage":
            stage_fd = descriptor
        return descriptor

    def fail_fstat(descriptor: int) -> os.stat_result:
        if fault == "fstat" and descriptor == stage_fd:
            raise OSError("fstat failed")
        return real_fstat(descriptor)

    monkeypatch.setattr(module.os, "stat", fail_stat)
    monkeypatch.setattr(module.os, "open", fail_open)
    monkeypatch.setattr(module.os, "fstat", fail_fstat)
    try:
        with pytest.raises(module._StagingCreationError) as raised:
            module._create_staging_directory_at(parent_fd, "stage")
    finally:
        monkeypatch.setattr(module.os, "stat", real_stat)
        monkeypatch.setattr(module.os, "open", real_open)
        monkeypatch.setattr(module.os, "fstat", real_fstat)
        os.close(parent_fd)

    if fault == "stat":
        assert raised.value.created is None
    else:
        assert raised.value.created is not None
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds - 1


def test_staging_creation_existing_name_propagates_file_exists(
    tmp_path: Path,
) -> None:
    (tmp_path / "stage").mkdir()
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(FileExistsError) as raised:
            module._create_staging_directory_at(parent_fd, "stage")
    finally:
        os.close(parent_fd)
    assert not isinstance(raised.value, module._StagingCreationError)


def test_staging_creation_propagates_proven_no_create_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_mkdir = module.os.mkdir

    def deny_create(path: object, *args: object, **kwargs: object) -> None:
        if os.fspath(path) == "stage":
            raise PermissionError("mkdir denied")
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "mkdir", deny_create)
    try:
        with pytest.raises(PermissionError, match="mkdir denied"):
            module._create_staging_directory_at(parent_fd, "stage")
    finally:
        monkeypatch.setattr(module.os, "mkdir", real_mkdir)
        os.close(parent_fd)
    assert not (tmp_path / "stage").exists()


def test_staging_fstat_failure_closes_fd_without_raw_fstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    baseline_fds = len(list(Path("/proc/self/fd").iterdir()))
    real_open = module.os.open
    real_fstat = module.os.fstat
    real_raw_fstat = module._RAW_FSTAT
    stage_fd: int | None = None

    def record_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal stage_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        if os.fspath(path) == "stage":
            stage_fd = descriptor
        return descriptor

    def fail_stage_fstat(descriptor: int) -> os.stat_result:
        if descriptor == stage_fd:
            raise OSError("stage fstat failed")
        return real_fstat(descriptor)

    def fail_stage_raw_fstat(descriptor: int) -> os.stat_result:
        if descriptor == stage_fd:
            raise OSError("stage raw fstat failed")
        return real_raw_fstat(descriptor)

    monkeypatch.setattr(module.os, "open", record_open)
    monkeypatch.setattr(module.os, "fstat", fail_stage_fstat)
    monkeypatch.setattr(module, "_RAW_FSTAT", fail_stage_raw_fstat)
    try:
        with pytest.raises(module._StagingCreationError):
            module._create_staging_directory_at(parent_fd, "stage")
    finally:
        monkeypatch.setattr(module.os, "open", real_open)
        monkeypatch.setattr(module.os, "fstat", real_fstat)
        monkeypatch.setattr(module, "_RAW_FSTAT", real_raw_fstat)
        os.close(parent_fd)
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds - 1


def test_staging_creation_detects_name_swap_before_first_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_stat = module.os.stat
    swapped = False

    def swap_before_stat(
        path: object, *args: object, **kwargs: object,
    ) -> os.stat_result:
        nonlocal swapped
        if (
            not swapped
            and os.fspath(path) == "stage"
            and kwargs.get("dir_fd") == parent_fd
        ):
            os.rename(
                "stage", "stage.displaced",
                src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
            )
            os.mkdir("stage", 0o700, dir_fd=parent_fd)
            swapped = True
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "stat", swap_before_stat)
    try:
        with pytest.raises(module._StagingCreationError) as raised:
            module._create_staging_directory_at(parent_fd, "stage")
        preserved = module._staging_creation_preserved(
            parent_fd, tmp_path, "stage", raised.value
        )
    finally:
        monkeypatch.setattr(module.os, "stat", real_stat)
        os.close(parent_fd)

    assert swapped
    assert raised.value.created is None
    assert len(preserved) == 2
    assert preserved[0].ownership == "unbound"
    assert preserved[0].artifact_inode is None
    assert preserved[1].ownership == "unknown"
    assert preserved[1].artifact_inode == (tmp_path / "stage").stat().st_ino


def test_evaluator_input_witness_rejects_member_replacement(tmp_path: Path) -> None:
    member = tmp_path / "depth.npy"
    member.write_bytes(b"original")
    witness = module._EvaluatorInputsWitness(
        (module._FileWitness(member, module._identity(os.stat(member))),)
    )
    member.replace(tmp_path / "old-depth.npy")
    member.write_bytes(b"replacement")

    with pytest.raises(ValueError, match="evaluator input changed"):
        witness.revalidate()


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
        run["checkpoints"][1]["consumed_through_frame"] = 3
        status_path = tmp_path / run["checkpoints"][1]["checkpoint_status"]["path"]
        status = json.loads(status_path.read_text())
        status["consumed_through_frame"] = 3
        _write_json(status_path, status)
        record = _record(status_path, tmp_path)
        run["checkpoints"][1]["checkpoint_status"] = record
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
        run["checkpoints"][1]["artifacts"]["temporal_compact"] = {
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
    dataset_member = dataset / "depth.npy"
    dataset_member.write_bytes(b"dataset-original")

    monkeypatch.setattr(module, "verify_exact_profile_runs", lambda records: exact)
    monkeypatch.setattr(module, "_TRUSTED_SOURCE_MANIFEST", source_manifest)
    monkeypatch.setattr(
        module, "_verify_development_sources",
        lambda *args, **kwargs: module._FileWitness(
            source_manifest, module._identity(os.stat(source_manifest))
        ),
    )
    monkeypatch.setattr(
        module, "_capture_evaluator_inputs",
        lambda *args, **kwargs: module._EvaluatorInputsWitness((
            module._FileWitness(
                dataset_member, module._identity(os.stat(dataset_member))
            ),
        )),
    )
    monkeypatch.setattr(module, "_materialize_config", lambda base, declaration: {"scene": "apartment", "algorithm_hash": "a" * 64})
    monkeypatch.setattr(module, "_audit_future_leakage", lambda **kwargs: {
        "schema_version": 1, "manifest_id": "oviv2_tesse_future_leakage_evidence_v1",
        "scene": "apartment", "candidate_id": kwargs["candidate_id"],
        "source_index": kwargs["run"]["source_index"], "records": [],
    })
    def evaluate(*, output: Path | None, **_: object) -> dict[str, Any]:
        assert output is None
        result = {"anchor_mappings": [], "macro": {"anchor_coverage_gate": {
            "eligible_count": 66, "uniquely_mapped_count": 53,
        }}}
        return result
    monkeypatch.setattr(module, "evaluate_temporal_occlusion_package", evaluate)
    monkeypatch.setattr(module, "_anchor_coverage", lambda result: {"eligible_count": 66, "mapped_count": 53})
    def preflight_at(
        *, root_fd: int, candidate_sources: Path, output: Path, **_: object,
    ) -> dict[str, Any]:
        source_bytes = os.open(
            candidate_sources.as_posix(), os.O_RDONLY, dir_fd=root_fd
        )
        try:
            sources = json.loads(os.read(source_bytes, 1024 * 1024))
        finally:
            os.close(source_bytes)
        assert [item["candidate_id"] for item in sources["candidates"]] == list(selected)
        result = {"candidates": []}
        module._write_json_at(root_fd, output, result)
        return result
    monkeypatch.setattr(module, "_build_preflight_at", preflight_at)
    monkeypatch.setattr(
        module, "_validate_preflight_gate_evidence_at",
        lambda *args, **kwargs: {},
    )
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

    for hook_name in ("builder", "validator"):
        race_output = tmp_path / f"{hook_name}-stage-swap"
        with monkeypatch.context() as patcher:
            def swap_stage(root_fd: int) -> Path:
                stage = Path(os.readlink(f"/proc/self/fd/{root_fd}"))
                owned = stage.with_name(stage.name + ".owned")
                stage.rename(owned)
                stage.mkdir()
                (stage / "competitor").write_bytes(b"untouched")
                return owned
            if hook_name == "builder":
                def swap_in_builder(**call: object) -> dict[str, Any]:
                    owned = swap_stage(int(call["root_fd"]))
                    result = preflight_at(**call)
                    assert (owned / "preflight.json").is_file()
                    return result
                patcher.setattr(module, "_build_preflight_at", swap_in_builder)
            else:
                def swap_in_validator(root_fd: int, *args: object, **kwargs: object) -> dict[str, Any]:
                    owned = swap_stage(root_fd)
                    assert (owned / "preflight.json").is_file()
                    return {}
                patcher.setattr(
                    module, "_validate_preflight_gate_evidence_at",
                    swap_in_validator,
                )
            with pytest.raises(module.PreflightPublicationUncertain) as raised:
                build_preflight_sources(
                    development_evidence=evidence, search_manifest=search,
                    apartment_base_config=base, targets=targets,
                    dataset_root=dataset, output=race_output,
                )
        assert raised.value.preserved[0].ownership == "owned"
        assert raised.value.preserved[0].name.endswith(".owned")
        assert not race_output.exists()

    evaluator_parent = tmp_path / "evaluator-parent"
    evaluator_parent.mkdir()
    evaluator_parent_old = tmp_path / "evaluator-parent-old"
    swapped_parent = False
    with monkeypatch.context() as patcher:
        def evaluate_then_swap_parent(**call: object) -> dict[str, Any]:
            nonlocal swapped_parent
            result = evaluate(**call)
            if not swapped_parent:
                evaluator_parent.rename(evaluator_parent_old)
                evaluator_parent.mkdir()
                swapped_parent = True
            return result
        patcher.setattr(
            module, "evaluate_temporal_occlusion_package", evaluate_then_swap_parent
        )
        with pytest.raises(module.PreflightPublicationUncertain) as raised:
            build_preflight_sources(
                development_evidence=evidence, search_manifest=search,
                apartment_base_config=base, targets=targets,
                dataset_root=dataset, output=evaluator_parent / "bundle",
            )
    assert raised.value.preserved[0].ownership == "owned"
    assert not (evaluator_parent / "bundle").exists()
    assert any(evaluator_parent_old.glob(".bundle.staging-*"))

    with monkeypatch.context() as patcher:
        def evaluate_then_replace(**call: object) -> dict[str, Any]:
            result = evaluate(**call)
            dataset_member.replace(dataset / "old-depth.npy")
            dataset_member.write_bytes(b"dataset-replacement")
            return result
        patcher.setattr(module, "evaluate_temporal_occlusion_package", evaluate_then_replace)
        with pytest.raises(module.PreflightPublicationUncertain) as raised:
            build_preflight_sources(
                development_evidence=evidence, search_manifest=search,
                apartment_base_config=base, targets=targets,
                dataset_root=dataset, output=tmp_path / "dataset-replaced",
            )
        assert "evaluator input changed" in str(raised.value.__cause__)
    dataset_member.write_bytes(b"dataset-original")

    checkpoint_member = roots["a0"] / "records/coverage.jsonl"
    checkpoint_member_old = roots["a0"] / "records/coverage-old.jsonl"
    with monkeypatch.context() as patcher:
        changed = False
        def evaluate_then_replace_checkpoint(**call: object) -> dict[str, Any]:
            nonlocal changed
            result = evaluate(**call)
            if not changed:
                checkpoint_member.replace(checkpoint_member_old)
                checkpoint_member.write_bytes(b"replacement-checkpoint-source")
                changed = True
            return result
        patcher.setattr(
            module, "evaluate_temporal_occlusion_package",
            evaluate_then_replace_checkpoint,
        )
        with pytest.raises(module.PreflightPublicationUncertain):
            build_preflight_sources(
                development_evidence=evidence, search_manifest=search,
                apartment_base_config=base, targets=targets,
                dataset_root=dataset, output=tmp_path / "checkpoint-source-replaced",
            )
    checkpoint_member.unlink()
    checkpoint_member_old.rename(checkpoint_member)

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
        assert [item.ownership for item in raised.value.preserved] == [
            "unbound", "unknown",
        ]
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds

    stage_race = tmp_path / "stage-create-race"
    with monkeypatch.context() as patcher:
        real_open = module.os.open
        created_identity: tuple[int, int] | None = None
        replacement_identity: tuple[int, int] | None = None
        swapped = False

        def swap_created_stage_before_open(
            path: object, flags: int, *args: object, **kwargs: object,
        ) -> int:
            nonlocal created_identity, replacement_identity, swapped
            name = os.fspath(path)
            parent_descriptor = kwargs.get("dir_fd")
            if (
                not swapped
                and isinstance(name, str)
                and name.startswith(".stage-create-race.staging-")
                and isinstance(parent_descriptor, int)
                and flags & os.O_DIRECTORY
            ):
                created = os.stat(
                    name, dir_fd=parent_descriptor, follow_symlinks=False
                )
                created_identity = (created.st_dev, created.st_ino)
                os.rename(
                    name, name + ".displaced",
                    src_dir_fd=parent_descriptor, dst_dir_fd=parent_descriptor,
                )
                os.mkdir(name, 0o700, dir_fd=parent_descriptor)
                replacement = os.stat(
                    name, dir_fd=parent_descriptor, follow_symlinks=False
                )
                replacement_identity = (replacement.st_dev, replacement.st_ino)
                swapped = True
            return real_open(path, flags, *args, **kwargs)

        patcher.setattr(module.os, "open", swap_created_stage_before_open)
        with pytest.raises(module.PreflightPublicationUncertain) as raised:
            build_preflight_sources(**kwargs, output=stage_race)

    assert swapped
    assert created_identity is not None
    assert replacement_identity is not None
    reported = {
        (item.artifact_device, item.artifact_inode): item.ownership
        for item in raised.value.preserved
    }
    assert created_identity in reported
    assert replacement_identity in reported
    assert reported[replacement_identity] == "unknown"
    assert not stage_race.exists()
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds

    collision = tmp_path / "collision"
    with monkeypatch.context() as patcher:
        def collide(parent_fd: int, source: str, destination: str) -> None:
            os.mkdir(destination, dir_fd=parent_fd)
            destination_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent_fd)
            try:
                marker_fd = os.open(
                    "competitor", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600, dir_fd=destination_fd,
                )
                os.write(marker_fd, b"untouched")
                os.close(marker_fd)
            finally:
                os.close(destination_fd)
            raise FileExistsError(destination)
        patcher.setattr(module, "_rename_noreplace", collide)
        with pytest.raises(module.PreflightPublicationUncertain) as raised:
            build_preflight_sources(**kwargs, output=collision)
    assert collision.is_dir()
    assert (collision / "competitor").read_bytes() == b"untouched"
    assert raised.value.preserved[0].ownership == "owned"
    assert Path(raised.value.preserved[0].logical_path).is_dir()
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds

    rename_member = tmp_path / "rename-member-tamper"
    with monkeypatch.context() as patcher:
        real_rename = module._rename_noreplace
        def tamper_during_rename(parent_fd: int, source: str, destination: str) -> None:
            source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent_fd)
            try:
                member_fd = os.open(
                    "preflight.json", os.O_WRONLY | os.O_TRUNC, dir_fd=source_fd
                )
                try:
                    os.write(member_fd, b'{"tampered":true}\n')
                    os.fsync(member_fd)
                finally:
                    os.close(member_fd)
            finally:
                os.close(source_fd)
            real_rename(parent_fd, source, destination)
        patcher.setattr(module, "_rename_noreplace", tamper_during_rename)
        with pytest.raises(module.PreflightPublicationUncertain) as raised:
            build_preflight_sources(**kwargs, output=rename_member)
    assert rename_member.is_dir()
    assert raised.value.preserved[0].ownership == "owned"
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

    parent_swap = tmp_path / "rename-parent"
    parent_swap.mkdir()
    parent_swap_old = tmp_path / "rename-parent-old"
    with monkeypatch.context() as patcher:
        real_rename = module._rename_noreplace
        def rename_then_swap_parent(parent_fd: int, source: str, destination: str) -> None:
            real_rename(parent_fd, source, destination)
            parent_swap.rename(parent_swap_old)
            parent_swap.mkdir()
        patcher.setattr(module, "_rename_noreplace", rename_then_swap_parent)
        with pytest.raises(module.PreflightPublicationUncertain) as raised:
            build_preflight_sources(**kwargs, output=parent_swap / "bundle")
    assert raised.value.preserved[0].ownership == "owned"
    assert (parent_swap_old / "bundle").is_dir()
    assert not (parent_swap / "bundle").exists()
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline_fds
