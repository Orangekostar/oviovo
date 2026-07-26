from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

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
from src.evaluation.baselines.tesse_cd import (
    summarize_khronos_official_metrics_partial,
)
from src.oviv2.temporal_config import ExecutionProfile


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_MANIFEST = (
    REPO_ROOT
    / "configs/evaluation/manifests/oviv2_tesse_dual_readout_search_v1.json"
)
COMMON_METRICS = {
    "current_miou": 0.4,
    "ghost_rate": 0.1,
    "background_f5": 0.3,
    "recovery_frames": 100.0,
}
ORIGINAL_COMMON_REPLAY = package_module._recompute_common_v2_metrics


@pytest.fixture(autouse=True)
def _stub_expensive_common_v2_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        package_module,
        "_recompute_common_v2_metrics",
        lambda *args, **kwargs: dict(COMMON_METRICS),
        raising=False,
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
    manifest_payload = json.loads(PRODUCTION_MANIFEST.read_text())
    base_temporal = json.loads(
        (REPO_ROOT / "configs/oviv2_tesse_cd_apartment_v2.json").read_text()
    )["temporal_readout"]
    for declaration in manifest_payload["candidates"]:
        profile = ExecutionProfile.from_id(declaration["candidate_id"])
        declaration["temporal_readout"] = {
            **base_temporal,
            "execution_profile": profile.profile_id,
            "components": profile.components,
        }
    manifest = _write(root / "search_manifest.json", manifest_payload)
    candidate = next(
        item for item in manifest_payload["candidates"] if item["candidate_id"] == "a2"
    )
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
        "protocol": "tesse_cd_common_v2", "status": "PASS", "method": "OVIV2", "mode": "causal_checkpoints", "scene": "apartment",
        "metrics": COMMON_METRICS,
        "sources": {"temporal_index": _record(temporal_manifest)},
    })
    occlusion = _write(root / "occlusion.json", {
        "format": "oviv2_temporal_compact_v1", "scene": "apartment", "algorithm_hash": algorithm,
        "input_sha256": h("5"), "code_commit": commit, "source_bindings": input_hashes,
        "input_bindings": {"indexes": [_record(index)]},
    })
    official_results = root / "khronos/map/results"
    static_source = official_results / "static_objects.csv"
    static_source.parent.mkdir(parents=True, exist_ok=True)
    static_source.write_text(
        "Name,Query,NumObjDetected,NumObjHallucinated,NumObjMissed,"
        "AppearedTP,AppearedFP,AppearedFN,DisappearedTP,DisappearedFP,"
        "DisappearedFN\n0,0,4,1,1,3,1,1,2,1,1\n"
    )
    background_source = official_results / "background_mesh.csv"
    background_source.write_text("Name,Accuracy@0.2,Completeness@0.2\n0,0.5,0.5\n")
    official_partial = summarize_khronos_official_metrics_partial(official_results)
    missing_dynamic = official_results / "dynamic_objects.csv"
    official = _write(root / "khronos/evaluation/official_metrics.json", {
        "status": "PARTIAL", "dataset": "TESSE-CD", "scene": "apartment", "split": "apartment_test",
        "method": "OVIV2", "mode": "causal_checkpoints", "display_mode": "online",
        "aggregation": "upstream online 4D plotting aggregation",
        "run_identity": {"run_id": "a2-apartment", "config_sha256": config_sha},
        "metrics": {
            "state_count": official_partial["state_count"],
            **official_partial["metrics"],
        },
        "unavailable": official_partial["unavailable"],
        "sources": [
            _record(static_source),
            {"path": str(missing_dynamic), "status": "MISSING"},
            _record(background_source),
        ],
    })
    protected = [{"path": "src/oviv2/dual_readout.py", "sha256": h("7"), "bytes": 123}]
    tests = [{"path": "tests/oviv2/test_dual_readout.py", "sha256": h("8"), "bytes": 456}]
    def gate(files: tuple[str, ...], digest: str) -> dict[str, object]:
        return {"scope": "shared_code_and_A0-A4_fixture", "status": "PASS", "code_commit": commit,
            "code_tree": h("d"), "protected_records": protected, "test_records": [{"argv": ["python", "-m", "pytest", "-q", "-rA", "-o", "addopts=", *files],
                "returncode": 0, "stdout_sha256": digest, "stdout_bytes": 10, "stderr_sha256": h("0"), "stderr_bytes": 0}]}
    t1_files = ("tests/oviv2/test_t1_noninterference.py",)
    determinism_files = (
        "tests/oviv2/test_temporal_config.py", "tests/oviv2/test_temporal_lifecycle.py",
        "tests/oviv2/test_temporal_association.py", "tests/oviv2/test_temporal_geometry.py",
        "tests/oviv2/test_temporal_background.py", "tests/oviv2/test_temporal_runtime.py",
        "tests/oviv2/test_temporal_snapshot.py", "tests/oviv2/test_dual_readout.py",
        "tests/oviv2/test_reference_readout.py", "tests/evaluation/test_oviv2_temporal_tesse.py",
        "tests/evaluation/test_run_oviv2_tesse_cd_v2.py",
    )
    source_manifest = REPO_ROOT / "configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json"
    source_digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    sequence = ["reference", "a0", "a1", "a0", "a2", "a0", "a3", "a0", "a4"]
    executions = [{
        "profile": profile, "argv": ["python", "runner.py", "--profile", profile],
        "pid": 100 + position, "code_commit": commit,
        "source_manifest_sha256": source_digest, "input_fingerprints": input_hashes,
        "output_root": str(run_root.resolve()),
    } for position, profile in enumerate(sequence)]
    profiles = {profile: {"cumulative_root_sha256": h("6"), "checkpoint_frames": [2],
                          "inventory": [{"path": "checkpoint/00000000/00000002/artifact/entities/neutral.jsonl",
                                         "sha256": h("a"), "byte_count": 3}]}
                for profile in ("a0", "a1", "a2", "a3", "a4")}
    evidence = _write(root / "development_gates.json", {
        "schema_version": 1, "manifest_id": "oviv2_dual_readout_development_gates_v1",
        "deterministic_evidence": {"base_commit": package_module.CUMULATIVE_BASE_COMMIT, "code_commit": commit, "code_tree": h("d"),
            "protected_files": protected, "test_sources": tests,
            "source_manifest": _record(source_manifest),
            "cumulative_exact": {"format": "oviv2_t1_exact_transaction_v1", "sequence": sequence,
                "executions": executions, "profiles": profiles},
            "gates": {"t1_exact": gate(t1_files, h("9")), "determinism": gate(determinism_files, h("b"))}},
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
    assert result["metrics"]["dynamic_f1"]["available"] is False
    assert "dynamic_objects.csv" in result["metrics"]["dynamic_f1"]["reason"]
    assert load_and_revalidate_result(output, manifest=paths["manifest"]) == result


def test_rejects_metrics_that_do_not_match_recomputed_sources(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    official = json.loads(paths["official_metrics"].read_text())
    official["metrics"]["object_f1"] = 0.99
    _write(paths["official_metrics"], official)
    with pytest.raises(ValueError, match="official metrics differ from recomputed CSV"):
        _package(paths, tmp_path / "official-tamper.json")

    paths = _fixture(tmp_path / "common")
    summary = json.loads(paths["common_v2_summary"].read_text())
    summary["metrics"]["current_miou"] = 0.99
    _write(paths["common_v2_summary"], summary)
    with pytest.raises(ValueError, match="common-v2 metrics differ from replay"):
        _package(paths, tmp_path / "common-tamper.json")


def test_rejects_unavailable_reason_that_does_not_match_recomputed_sources(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    official = json.loads(paths["official_metrics"].read_text())
    official["unavailable"]["dynamic_f1"] = "forged unavailable reason"
    _write(paths["official_metrics"], official)

    with pytest.raises(
        ValueError, match="official unavailable metrics differ from recomputed CSV"
    ):
        _package(paths, tmp_path / "official-unavailable-tamper.json")


def test_rejects_official_sources_outside_khronos_results_directory(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    official = json.loads(paths["official_metrics"].read_text())
    official["sources"][0]["path"] = str(
        (tmp_path / "unrelated" / "static_objects.csv").resolve()
    )
    _write(paths["official_metrics"], official)
    with pytest.raises(ValueError, match="official source path is not canonical"):
        _package(paths, tmp_path / "result.json")


def test_rejects_noncanonical_official_metrics_path(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    moved = paths["official_metrics"].with_name("not_official_metrics.json")
    paths["official_metrics"].rename(moved)
    paths["official_metrics"] = moved

    with pytest.raises(ValueError, match="official metrics path is not canonical"):
        _package(paths, tmp_path / "result.json")


def test_rejects_existing_official_source_without_byte_count(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    official = json.loads(paths["official_metrics"].read_text())
    del official["sources"][0]["byte_count"]
    _write(paths["official_metrics"], official)
    with pytest.raises(ValueError, match="official source record is not exact"):
        _package(paths, tmp_path / "result.json")


def test_accepts_production_relative_official_source_paths(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    official = json.loads(paths["official_metrics"].read_text())
    for record, name in zip(
        official["sources"],
        ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv"),
    ):
        record["path"] = f"../map/results/{name}"
    _write(paths["official_metrics"], official)

    result = _package(paths, tmp_path / "result.json")

    assert result["status"] == "PASS"


def test_official_metrics_recompute_uses_snapshotted_csv_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    results_dir = tmp_path / "khronos/map/results"
    attack_dir = tmp_path / "khronos/map/attack-results"
    attack_dir.mkdir()
    (attack_dir / "static_objects.csv").write_text(
        "Name,Query,NumObjDetected,NumObjHallucinated,NumObjMissed,"
        "AppearedTP,AppearedFP,AppearedFN,DisappearedTP,DisappearedFP,"
        "DisappearedFN\n0,0,1,0,0,1,0,0,1,0,0\n"
    )
    (attack_dir / "background_mesh.csv").write_bytes(
        (results_dir / "background_mesh.csv").read_bytes()
    )
    attack_metrics = summarize_khronos_official_metrics_partial(attack_dir)
    official = json.loads(paths["official_metrics"].read_text())
    official["metrics"] = {
        "state_count": attack_metrics["state_count"],
        **attack_metrics["metrics"],
    }
    _write(paths["official_metrics"], official)
    original_summarizer = package_module.summarize_khronos_official_metrics_partial
    parked_dir = tmp_path / "khronos/map/parked-results"

    def summarize_during_directory_swap(source_dir: Path) -> dict[str, object]:
        results_dir.rename(parked_dir)
        attack_dir.rename(results_dir)
        try:
            return original_summarizer(source_dir)
        finally:
            results_dir.rename(attack_dir)
            parked_dir.rename(results_dir)

    monkeypatch.setattr(
        package_module,
        "summarize_khronos_official_metrics_partial",
        summarize_during_directory_swap,
    )
    output = tmp_path / "result.json"

    with pytest.raises(ValueError, match="official metrics differ from recomputed CSV"):
        _package(paths, output)

    assert not output.exists()


def test_direct_cli_help_works_from_repo_root() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts/evaluation/package_oviv2_tesse_dual_readout_result.py"
            ),
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_common_v2_replay_uses_declared_production_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporal = tmp_path / "temporal.json"
    target = tmp_path / "targets.json"
    aliases = tmp_path / "aliases.yaml"
    label_space = tmp_path / "labels.yaml"
    for path in (temporal, target, aliases, label_space):
        path.write_text("{}\n")
    evaluator = REPO_ROOT / "scripts/evaluation/evaluate_tesse_cd_common_v2.py"
    sources = {
        "temporal_index": _record(temporal),
        "target_manifest": _record(target),
        "aliases": _record(aliases),
        "label_space": _record(label_space),
        "evaluator": _record(evaluator),
    }
    payload = {
        "metrics": COMMON_METRICS,
        "frames": [{"event_id": "event", "frame_id": 0}],
        "event_region_prediction_counts": {"event": {"0": 0}},
        "event_background_prediction_counts": {"event": {"0": 0}},
        "sources": sources,
    }
    common_path = _write(tmp_path / "common.json", payload)
    common_snapshot = package_module._snapshot(common_path, "common-v2 summary")

    def replay(
        temporal_index: Path,
        target_manifest: Path,
        aliases_path: Path,
        label_space_path: Path,
        output: Path,
    ) -> Path:
        assert (temporal_index, target_manifest, aliases_path, label_space_path) == (
            temporal,
            target,
            aliases,
            label_space,
        )
        return _write(output / "summary.json", payload)

    monkeypatch.setattr(package_module, "evaluate_common_v2", replay)
    witnesses: list[package_module.Snapshot | package_module.FileIdentityWitness] = []
    assert ORIGINAL_COMMON_REPLAY(common_snapshot, witnesses) == COMMON_METRICS
    assert {item.path for item in witnesses} == {
        temporal,
        target,
        aliases,
        label_space,
        evaluator,
    }


def test_rejects_shallow_nonproduction_search_manifest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    shallow = _write(
        tmp_path / "shallow-manifest.json",
        {
            "schema_version": 1,
            "manifest_id": "oviv2-tesse-dual-readout-search-v1",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "protocol_id": "oviv2-tessecd-v2",
            "development_scene": "apartment",
            "candidates": [{"candidate_id": "a2"}],
        },
    )
    status = json.loads(paths["search_status"].read_text())
    status["manifest"] = _record(shallow)
    _write(paths["search_status"], status)
    paths["manifest"] = shallow
    with pytest.raises(ValueError, match="manifest|keys|candidate"):
        _package(paths, tmp_path / "result.json")


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


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("a4_drift", "cumulative"),
        ("missing_profile", "profile"),
        ("wrong_argv", "argv"),
        ("wrong_base", "base commit"),
    ],
)
def test_rejects_inexact_cumulative_development_evidence(
    tmp_path: Path, mutation: str, match: str
) -> None:
    paths = _fixture(tmp_path)
    payload = json.loads(paths["t1_exact_evidence"].read_text())
    evidence = payload["deterministic_evidence"]
    if mutation == "a4_drift":
        evidence["cumulative_exact"]["profiles"]["a4"]["cumulative_root_sha256"] = "f" * 64
    elif mutation == "missing_profile":
        del evidence["cumulative_exact"]["profiles"]["a3"]
    elif mutation == "wrong_argv":
        evidence["gates"]["t1_exact"]["test_records"][0]["argv"].append("-k")
    else:
        evidence["base_commit"] = "f" * 40
    _write(paths["t1_exact_evidence"], payload)
    with pytest.raises(ValueError, match=match):
        _package(paths, tmp_path / "result.json")


def test_every_temporal_scalar_changes_only_algorithm_identity(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = json.loads(paths["candidate_config"].read_text())
    baseline_algorithm = canonical_algorithm_hash(config)
    baseline_non_temporal = non_temporal_config_sha256(config)
    evidence = json.loads(paths["t1_exact_evidence"].read_text())
    roots = {
        record["cumulative_root_sha256"]
        for record in evidence["deterministic_evidence"]["cumulative_exact"]["profiles"].values()
    }
    assert len(roots) == 1

    leaves: list[tuple[tuple[str, ...], object]] = []

    def visit(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, key))
        else:
            leaves.append((path, value))

    visit(config["temporal_readout"], ())
    for path, original in leaves:
        mutated = json.loads(json.dumps(config))
        target = mutated["temporal_readout"]
        for key in path[:-1]:
            target = target[key]
        if isinstance(original, bool):
            target[path[-1]] = not original
        elif isinstance(original, int):
            target[path[-1]] = original + 1
        elif isinstance(original, float):
            target[path[-1]] = original + 0.000001
        else:
            target[path[-1]] = f"{original}-mutated"
        assert canonical_algorithm_hash(mutated) != baseline_algorithm, path
        assert non_temporal_config_sha256(mutated) == baseline_non_temporal, path
        assert {
            record["cumulative_root_sha256"]
            for record in evidence["deterministic_evidence"]["cumulative_exact"]["profiles"].values()
        } == roots


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


def test_publication_rejects_missing_official_source_that_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "result.json"
    missing = tmp_path / "khronos/map/results/dynamic_objects.csv"
    original = package_module.Snapshot.revalidate
    created = False

    def create_missing_once(snapshot: package_module.Snapshot) -> None:
        nonlocal created
        if not created:
            created = True
            missing.write_text("Name,Query,NumObjDetected\n0,0,1\n")
        original(snapshot)

    monkeypatch.setattr(package_module.Snapshot, "revalidate", create_missing_once)

    with pytest.raises(ValueError, match="changed before publication"):
        _package(paths, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".result.json.*"))
