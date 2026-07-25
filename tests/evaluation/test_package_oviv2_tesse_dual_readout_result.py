from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.evaluation import package_oviv2_tesse_dual_readout_result as package_module
from scripts.evaluation.evaluate_oviv2_tesse_occlusion import canonical_algorithm_hash
from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (
    input_binding_values_sha256,
    non_temporal_config_sha256,
)
from scripts.evaluation.package_oviv2_tesse_dual_readout_result import (
    load_and_revalidate_result,
    package_result,
)


def _bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(value))
    return path


def _record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "byte_count": len(data)}


def _fixture(root: Path) -> dict[str, Path]:
    h = lambda character: character * 64
    commit = "c" * 40
    candidate = {
        "candidate_id": "a2",
        "components": {
            "association_mode": "active_uncertain", "background_mode": "v1_cumulative",
            "geometry_mode": "object_submap", "lifecycle_mode": "probabilistic_hysteresis",
            "motion_mode": "translation",
        },
        "temporal_readout": {"execution_profile": "a2", "lifecycle": {"active_on_probability": 0.75}},
    }
    manifest = _write(root / "manifest.json", {
        "schema_version": 1, "manifest_id": "oviv2-tesse-dual-readout-search-v1",
        "dataset": "TESSE-CD", "method_id": "OVIV2", "protocol_id": "oviv2-tessecd-v2",
        "development_scene": "apartment", "transfer_scene": "office",
        "transfer_policy": "bind_only_never_execute", "candidates": [candidate],
    })
    config = {
        "scene": "apartment",
        "temporal_readout": candidate["temporal_readout"], "dataset_root": "/frozen/tesse",
        "dense_manifest": "/frozen/dense.json", "evaluation_checkpoint_frames_sha256": h("e"),
        "export_manifest": "/frozen/export.json", "frontend_manifest": "/frozen/frontend.json",
        "input_manifest": "/frozen/input.json", "occlusion_target_manifest_sha256": h("3"),
        "schedule_manifest": "/frozen/schedule.json", "occlusion_target_manifest": "/frozen/targets.json",
    }
    config["algorithm_hash"] = canonical_algorithm_hash(config)
    algorithm = config["algorithm_hash"]
    config_path = _write(root / "config.json", config)
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    input_hashes = {"dataset": h("1"), "schedule": h("2"), "occlusion_targets": h("3"), "aliases": h("4")}
    run_root = root / "run"
    normalized = _write(run_root / "normalized_run_config.json", config)
    index = _write(run_root / "occlusion_checkpoint_index.json", {
        "schema_version": 1, "format": "oviv2_temporal_compact_v1", "protocol_id": "oviv2-tessecd-v2",
        "dataset": "TESSE-CD", "method_id": "OVIV2", "scene": "apartment", "algorithm_hash": algorithm,
        "schedule": {"sha256": h("2"), "byte_count": 10}, "target_manifest": {"sha256": h("3"), "byte_count": 11},
        "input_sha256": h("5"), "code_commit": commit, "source_bindings": input_hashes,
        "checkpoints": [{"frame_index": 2, "consumed_through_frame": 2, "consumed_through_frame_exclusive": 3}],
    })
    schedule_file = root / "schedule.json"; schedule_file.write_text("{}\n")
    checkpoint_status = _write(run_root / "checkpoint_status.json", {"status": "PASS"})
    snapshot = root / "snapshot.npz"; snapshot.write_bytes(b"snapshot")
    entities = root / "entities.json"; entities.write_text("[]\n")
    checkpoint = {"frame_index": 2, "timestamp_ns": 200, "consumed_through_frame": 2,
        "consumed_through_frame_exclusive": 3, "checkpoint_status": _record(checkpoint_status),
        "snapshot": _record(snapshot), "entities": _record(entities)}
    run_source_index = _write(run_root / "source_index.json", {"schema_version": 1, "dataset": "TESSE-CD",
        "mode": "causal_checkpoint_exports", "method": "OVIV2", "scene": "apartment",
        "schedule": _record(schedule_file), "checkpoints": [checkpoint]})
    run_manifest = _write(run_root / "run_manifest.json", {
        "schema_version": 2, "protocol_id": "oviv2-tessecd-v2", "dataset": "TESSE-CD", "method_id": "OVIV2",
        "scene": "apartment", "mode": "dual_readout_causal_checkpoints", "algorithm_hash": algorithm,
        "config": {"sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(), "byte_count": config_path.stat().st_size},
        "normalized_run_config": {"path": "normalized_run_config.json", "sha256": hashlib.sha256(normalized.read_bytes()).hexdigest(), "byte_count": normalized.stat().st_size},
        "schedule": {"sha256": h("2"), "byte_count": 10}, "target_manifest": {"sha256": h("3"), "byte_count": 11},
        "source_bindings": input_hashes, "input_sha256": h("5"), "code_commit": commit,
        "occlusion_checkpoint_index": {"path": "occlusion_checkpoint_index.json", "sha256": hashlib.sha256(index.read_bytes()).hexdigest(), "byte_count": index.stat().st_size},
        "source_index": {"path": "source_index.json", "sha256": hashlib.sha256(run_source_index.read_bytes()).hexdigest(), "byte_count": run_source_index.stat().st_size},
        "checkpoints": [{"frame_index": 2, "consumed_through_frame": 2, "consumed_through_frame_exclusive": 3}],
    })
    stdout = root / "stdout.log"; stdout.write_text("ok\n")
    stderr = root / "stderr.log"; stderr.write_text("")
    status = _write(root / "search_status.json", {
        "schema_version": 1, "status": "PASS", "manifest": _record(manifest),
        "candidates": [{"candidate_id": "a2", "scene": "apartment", "status": "PASS", "exit_code": 0,
            "config_path": str(config_path.resolve()), "config_file": _record(config_path), "config_sha256": config_sha,
            "output_root": str(run_root.resolve()), "algorithm_hash": algorithm,
            "stdout_path": str(stdout.resolve()), "stderr_path": str(stderr.resolve()),
            "stdout_file": _record(stdout), "stderr_file": _record(stderr),
            "non_temporal_config_sha256": non_temporal_config_sha256(config),
            "input_binding_values_sha256": input_binding_values_sha256(config), "runtime_seconds": 12.5}],
    })
    export_root = root / "export"
    export_source_index = _write(export_root / "source_index.json", {"schema_version": 1, "dataset": "TESSE-CD",
        "mode": "causal_checkpoint_exports", "method": "OVIV2", "scene": "apartment",
        "schedule": _record(schedule_file), "checkpoints": [checkpoint]})
    temporal_manifest = _write(export_root / "temporal_manifest.json", {"schema_version": 1, "dataset": "TESSE-CD",
        "mode": "causal_checkpoints", "method": "OVIV2", "scene": "apartment",
        "sources": {"source_index": _record(export_source_index)}, "checkpoints": [checkpoint], "entity_lifecycles": []})
    common = _write(root / "common.json", {
        "schema_version": 1, "manifest_id": "tesse_cd_common_v2_scene_summary", "dataset": "TESSE-CD",
        "protocol": "tesse_cd_common_v2", "status": "PASS", "method": "OVIV2", "mode": "frozen", "scene": "apartment",
        "metrics": {"current_miou": 0.4, "ghost_rate": 0.1, "background_f5": 0.3, "recovery_frames": 100.0},
        "sources": {"temporal_index": _record(temporal_manifest)},
    })
    occlusion = _write(root / "occlusion.json", {
        "format": "oviv2_temporal_compact_v1", "scene": "apartment", "algorithm_hash": algorithm,
        "input_sha256": h("5"), "code_commit": commit, "source_bindings": input_hashes,
        "input_bindings": {"indexes": [_record(index)]},
    })
    official_sources = root / "official-source.csv"
    official_sources.write_text("header\n")
    official = _write(root / "official_metrics.json", {
        "status": "PASS", "dataset": "TESSE-CD", "scene": "apartment", "split": "apartment_test",
        "method": "OVIV2", "mode": "causal_checkpoints", "display_mode": "online",
        "aggregation": "upstream online 4D plotting aggregation",
        "run_identity": {"run_id": "a2-apartment", "config_sha256": config_sha},
        "metrics": {"object_f1": 0.7, "dynamic_f1": None, "change_f1": 0.6},
        "unavailable": {"dynamic_f1": "not_reported_by_official_evaluator"},
        "sources": [_record(official_sources)],
    })
    protected = [{"path": "src/oviv2/dual_readout.py", "sha256": h("7"), "bytes": 123}]
    tests = [{"path": "tests/oviv2/test_dual_readout.py", "sha256": h("8"), "bytes": 456}]
    def gate(name: str, digest: str) -> dict[str, object]:
        return {"scope": "shared_code_and_A0-A4_fixture", "status": "PASS", "code_commit": commit,
            "code_tree": h("d"), "protected_records": protected, "test_records": [{"argv": ["python", "-m", "pytest", "-q", name],
                "returncode": 0, "stdout_sha256": digest, "stdout_bytes": 10, "stderr_sha256": h("0"), "stderr_bytes": 0}]}
    evidence = _write(root / "development_gates.json", {
        "schema_version": 1, "manifest_id": "oviv2_dual_readout_development_gates_v1",
        "deterministic_evidence": {"base_commit": "e" * 40, "code_commit": commit, "code_tree": h("d"),
            "protected_files": protected, "test_sources": tests,
            "gates": {"t1_exact": gate("test_t1_exact", h("9")), "determinism": gate("test_repeat_byte_identical", h("b"))}},
        "receipt": {"created_at_utc": "2026-07-25T00:00:00Z"},
    })
    return {"manifest": manifest, "search_status": status, "candidate_config": config_path, "run_manifest": run_manifest,
            "common_v2_summary": common, "temporal_occlusion_result": occlusion, "official_metrics": official,
            "t1_exact_evidence": evidence, "determinism_evidence": evidence}


def _package(paths: dict[str, Path], output: Path) -> dict[str, object]:
    return package_result(candidate_id="a2", output=output, **paths)


def test_packages_exact_structured_result_and_revalidates(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "result.json"
    result = _package(paths, output)
    assert set(result) == {"schema_version", "manifest_id", "candidate_id", "scene", "status", "sources", "bindings", "run_identity", "gates", "metrics"}
    assert set(result["sources"]) == ({*paths} - {"manifest"}) | {"search_manifest"}
    assert all(set(record) == {"path", "sha256", "byte_count"} for record in result["sources"].values())
    assert all(set(gate) == {"passed", "reason", "source"} and gate["passed"] for gate in result["gates"].values())
    assert result["metrics"]["background_f5_cm"]["value"] == 0.3
    assert result["metrics"]["runtime_seconds"]["value"] == 12.5
    assert result["metrics"]["dynamic_f1"] == {"available": False, "value": None, "reason": "not_reported_by_official_evaluator", "source": "official_metrics"}
    assert load_and_revalidate_result(output, manifest=paths["manifest"]) == result


@pytest.mark.parametrize("mutation", ["metric", "gate", "source_hash", "coherent_metrics"])
def test_reload_rejects_result_tampering(tmp_path: Path, mutation: str) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "result.json"
    result = _package(paths, output)
    if mutation == "metric": result["metrics"]["object_f1"]["value"] = 0.99
    elif mutation == "gate": result["gates"]["causality"]["passed"] = False
    elif mutation == "source_hash": result["sources"]["common_v2_summary"]["sha256"] = "f" * 64
    else:
        for name in result["metrics"]:
            if result["metrics"][name]["available"]:
                result["metrics"][name]["value"] = 0.5
    output.write_bytes(_bytes(result))
    with pytest.raises(ValueError, match="does not match revalidated sources|source record"):
        load_and_revalidate_result(output, manifest=paths["manifest"])


@pytest.mark.parametrize("source,field,value,match", [
    ("search_status", "status", "FAIL", "search status"),
    ("common_v2_summary", "scene", "office", "Apartment"),
    ("temporal_occlusion_result", "format", "wrong", "occlusion"),
    ("official_metrics", "method", "DUALMAP", "official"),
    ("t1_exact_evidence", "status", "FAIL", "t1_exact"),
])
def test_rejects_broken_cross_source_chain(tmp_path: Path, source: str, field: str, value: object, match: str) -> None:
    paths = _fixture(tmp_path)
    payload = json.loads(paths[source].read_text())
    payload[field] = value
    _write(paths[source], payload)
    with pytest.raises(ValueError, match=match):
        _package(paths, tmp_path / "result.json")


def test_rejects_another_apartment_oviv2_temporal_artifact(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    common = json.loads(paths["common_v2_summary"].read_text())
    temporal_path = Path(common["sources"]["temporal_index"]["path"])
    temporal = json.loads(temporal_path.read_text())
    export_index_path = Path(temporal["sources"]["source_index"]["path"])
    export_index = json.loads(export_index_path.read_text())
    replacement = tmp_path / "other-snapshot.npz"
    replacement.write_bytes(b"another candidate")
    export_index["checkpoints"][0]["snapshot"] = _record(replacement)
    other_index = _write(tmp_path / "other-export-index.json", export_index)
    temporal["sources"]["source_index"] = _record(other_index)
    temporal["checkpoints"][0]["snapshot"] = _record(replacement)
    other_temporal = _write(tmp_path / "other-temporal-manifest.json", temporal)
    common["sources"]["temporal_index"] = _record(other_temporal)
    _write(paths["common_v2_summary"], common)
    with pytest.raises(ValueError, match="snapshot.*mismatch"):
        _package(paths, tmp_path / "result.json")


def test_rejects_duplicate_nonfinite_fifo_symlink_oversize_and_overwrite(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"PASS","status":"PASS"}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _package({**paths, "common_v2_summary": duplicate}, tmp_path / "a.json")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x":NaN}\n')
    with pytest.raises(ValueError, match="non-finite"):
        _package({**paths, "common_v2_summary": nonfinite}, tmp_path / "b.json")
    link = tmp_path / "link.json"; link.symlink_to(paths["common_v2_summary"])
    with pytest.raises(ValueError, match="symlink"):
        _package({**paths, "common_v2_summary": link}, tmp_path / "c.json")
    fifo = tmp_path / "fifo"; os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        _package({**paths, "common_v2_summary": fifo}, tmp_path / "d.json")
    huge = tmp_path / "huge.json"; huge.write_bytes(b" " * (8 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="too large"):
        _package({**paths, "common_v2_summary": huge}, tmp_path / "e.json")
    output = tmp_path / "occupied.json"; output.write_text("occupied")
    with pytest.raises(FileExistsError):
        _package(paths, output)
    assert output.read_text() == "occupied"
    assert not list(tmp_path.glob(".occupied.json.*"))


def test_publication_revalidates_indirect_sources_and_cleans_temporary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "result.json"
    original = package_module.Snapshot.revalidate
    mutated = False

    def change_once(snapshot: package_module.Snapshot) -> None:
        nonlocal mutated
        if not mutated and snapshot.path.name == "source_index.json":
            mutated = True
            snapshot.path.write_bytes(snapshot.data + b" ")
        original(snapshot)

    monkeypatch.setattr(package_module.Snapshot, "revalidate", change_once)
    with pytest.raises(ValueError, match="changed before publication"):
        _package(paths, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".result.json.*"))
