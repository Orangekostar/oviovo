from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np
import pytest

from scripts.evaluation import package_oviv2_tesse_dual_readout_result as package_module
from scripts.evaluation import run_oviv2_tesse_cd_v2 as production_runner
from scripts.evaluation import verify_oviv2_dual_readout_development_gates as gates_module
from scripts.evaluation.evaluate_oviv2_tesse_occlusion import canonical_algorithm_hash
from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (
    input_binding_values_sha256,
    non_temporal_config_sha256,
)
from scripts.evaluation.package_oviv2_tesse_dual_readout_result import (
    load_and_revalidate_result,
    package_result,
)
from scripts.evaluation.compare_oviv2_cumulative_artifacts import (
    ArtifactMismatch,
    compare_cumulative_artifacts,
)
from src.evaluation.baselines.tesse_cd import (
    summarize_khronos_official_metrics_partial,
)
from src.oviv2.temporal_config import ExecutionProfile, temporal_config_from_json
from tests.evaluation import test_run_oviv2_tesse_cd_v2 as runner_fixture


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
T2_DIRECTIONS = {
    "dynamic_f1": "maximize_strict",
    "change_f1": "maximize_strict",
    "ghost_rate": "minimize_strict",
    "background_f5_cm": "maximize_strict",
    "recovery_frames": "minimize_strict",
    "current_miou": "maximize_noninferior",
    "object_f1": "maximize_noninferior",
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


def _relative_record(path: Path, root: Path) -> dict[str, object]:
    record = _record(path)
    record["path"] = path.relative_to(root).as_posix()
    return record


def _baseline_evidence(root: Path) -> Path:
    release = _write(
        root / "baselines/oviv2_release.json",
        {
            "schema_version": 1,
            "manifest_id": "tesse-cd-frozen-baseline-scene-result-v1",
            "dataset": "TESSE-CD",
            "protocol_id": "oviv2-tessecd-v2",
            "scene": "apartment",
            "status": "PASS",
            "method_id": "OVIV2_RELEASE",
            "oracle": False,
            "metrics": {"current_miou": 0.3, "object_f1": 0.7},
        },
    )
    strongest = _write(
        root / "baselines/strongest_non_oracle.json",
        {
            "schema_version": 1,
            "manifest_id": "tesse-cd-frozen-baseline-scene-result-v1",
            "dataset": "TESSE-CD",
            "protocol_id": "oviv2-tessecd-v2",
            "scene": "apartment",
            "status": "PASS",
            "method_id": "STRONGEST_NON_ORACLE",
            "oracle": False,
            "metrics": {
                "dynamic_f1": 0.7,
                "change_f1": 0.5,
                "ghost_rate": 0.2,
                "background_f5_cm": 0.2,
                "recovery_frames": 200.0,
            },
        },
    )
    return _write(
        root / "baseline_evidence.json",
        {
            "schema_version": 1,
            "manifest_id": "oviv2-tesse-dual-readout-baseline-evidence-v1",
            "dataset": "TESSE-CD",
            "protocol_id": "oviv2-tessecd-v2",
            "scene": "apartment",
            "baselines": {
                name: {
                    "metric": name,
                    "source": _record(
                        release
                        if name in {"current_miou", "object_f1"}
                        else strongest
                    ),
                }
                for name in T2_DIRECTIONS
            },
        },
    )


def _mechanism_sources(
    root: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    trajectories = root / "run/trajectories.jsonl"
    trajectories.write_bytes(b"")
    lifecycle = root / "run/lifecycle_transitions.jsonl"
    lifecycle.write_bytes(
        _bytes(
            {
                "frame_index": 1,
                "timestamp_ns": 200,
                "entity_id": 7,
                "before": "active",
                "after": "uncertain",
                "evidence": "visible_absent",
                "geometry_epoch": 1,
                "readout_valid": False,
            }
        )
    )
    frame_coverage = root / "run/temporal_frame_coverage.jsonl"
    frame_coverage.write_bytes(
        b"".join(
            _bytes(
                {
                    "frame_index": frame,
                    "timestamp_ns": (frame + 1) * 100,
                    "record_count": 0,
                    "event_count": int(frame == 1),
                }
            )
            for frame in range(3)
        )
    )
    diagnostics = _write(
        root / "run/runtime_diagnostics.json",
        {
            "schema_version": 1,
            "execution_profile": "a2",
            "processed_frame_count": 3,
            "counters": {
                "proposal_opportunity_count": 2,
                "proposal_trigger_count": 1,
                "reid_opportunity_count": 0,
                "reid_trigger_count": 0,
                "identity_expiry_count": 0,
                "geometry_reclaim_count": 0,
                "motion_rejection_count": 1,
                "ledger_rejection_count": 0,
                "epoch_reset_opportunity_count": 1,
                "epoch_reset_trigger_count": 1,
                "icp_opportunity_count": 0,
                "icp_accept_count": 0,
                "icp_reject_count": 0,
                "ledger_stage_count": 0,
                "ledger_commit_count": 0,
                "ledger_reclaim_count": 0,
            },
            "mechanism_records": {
                "proposal_opportunity_count": ["proposal:0", "proposal:1"],
                "proposal_trigger_count": ["proposal:0"],
                "reid_opportunity_count": [],
                "reid_trigger_count": [],
                "identity_expiry_count": [],
                "geometry_reclaim_count": [],
                "motion_rejection_count": ["motion:0"],
                "ledger_rejection_count": [],
                "epoch_reset_opportunity_count": ["motion:0"],
                "epoch_reset_trigger_count": ["motion:0"],
                "icp_opportunity_count": [],
                "icp_accept_count": [],
                "icp_reject_count": [],
                "ledger_stage_count": [],
                "ledger_commit_count": [],
                "ledger_reclaim_count": [],
            },
            "diagnostic": None,
        },
    )
    run_root = root / "run"
    source_records = {
        "trajectories": _relative_record(trajectories, run_root),
        "lifecycle_transitions": _relative_record(lifecycle, run_root),
        "frame_coverage": _relative_record(frame_coverage, run_root),
        "runtime_diagnostics": _relative_record(diagnostics, run_root),
    }
    lifecycle_record = source_records["lifecycle_transitions"]
    diagnostics_record = source_records["runtime_diagnostics"]
    telemetry = {
        "absence": {
            "opportunities": 1,
            "triggers": 1,
            "available": True,
            "passed": True,
            "reason": None,
            "source": lifecycle_record,
        },
        "readout_invalidation": {
            "opportunities": 1,
            "triggers": 1,
            "available": True,
            "passed": True,
            "reason": None,
            "source": lifecycle_record,
        },
        "proposal_recovery": {
            "opportunities": 2,
            "triggers": 1,
            "available": True,
            "passed": True,
            "reason": None,
            "source": diagnostics_record,
        },
        "epoch_reset": {
            "opportunities": 1,
            "triggers": 1,
            "available": True,
            "passed": True,
            "reason": None,
            "source": diagnostics_record,
        },
        "motion_rejection": {
            "opportunities": 1,
            "triggers": 1,
            "available": True,
            "passed": True,
            "reason": None,
            "source": diagnostics_record,
        },
    }
    return source_records, telemetry


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
    mechanism_sources, mechanism_telemetry = _mechanism_sources(root)
    run_source_index = _write(run_root / "source_index.json", {"schema_version": 1, "dataset": "TESSE-CD",
        "mode": "causal_checkpoint_exports", "method": "OVIV2", "scene": "apartment",
        "schedule": _record(schedule_file),
        **mechanism_sources,
        "checkpoints": [checkpoint]})
    final_snapshot = run_root / "final_current_map" / "snapshot.npz"
    final_snapshot.parent.mkdir(parents=True)
    np.savez_compressed(
        final_snapshot,
        background_xyz=np.asarray([[1.0, 2.0, 3.0]]),
        timestamp=np.asarray(300.0),
        scope=np.asarray("current"),
        scene_id=np.asarray("apartment"),
    )
    final_entities = run_root / "final_current_map" / "entities.jsonl"
    final_entities.write_text("", encoding="utf-8")
    run_artifact_inventory = sorted(
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file()
    )
    run_manifest = _write(run_root / "run_manifest.json", {
        "schema_version": 2, "protocol_id": "oviv2-tessecd-v2", "dataset": "TESSE-CD", "method_id": "OVIV2",
        "scene": "apartment", "mode": "dual_readout_causal_checkpoints", "algorithm_hash": algorithm,
        "processed_frame_count": 3, "covered_frame_count": 3,
        "trajectory_frame_count": 3, "first_frame_index": 0,
        "last_frame_index": 2, "temporal_export_schema_version": 1,
        "scheduled_frame_indices": [2], "captured_frame_indices": [2],
        "config": {"sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(), "byte_count": config_path.stat().st_size},
        "normalized_run_config": {"path": "normalized_run_config.json", "sha256": hashlib.sha256(normalized.read_bytes()).hexdigest(), "byte_count": normalized.stat().st_size},
        "schedule": {"sha256": h("2"), "byte_count": 10}, "target_manifest": {"sha256": h("3"), "byte_count": 11},
        "source_bindings": input_hashes, "input_sha256": h("5"), "code_commit": commit,
        "occlusion_checkpoint_index": {"path": "occlusion_checkpoint_index.json", "sha256": hashlib.sha256(index.read_bytes()).hexdigest(), "byte_count": index.stat().st_size},
        "source_index": {"path": "source_index.json", "sha256": hashlib.sha256(run_source_index.read_bytes()).hexdigest(), "byte_count": run_source_index.stat().st_size},
        "checkpoints": [{"frame_index": 2, "consumed_through_frame": 2, "consumed_through_frame_exclusive": 3}],
        "artifact_inventory": run_artifact_inventory,
        "final_current_map": {
            "frame_index": 2, "timestamp_ns": 300, "scope": "current",
            "snapshot": _relative_record(final_snapshot, run_root),
            "entities": _relative_record(final_entities, run_root),
            "background_storage": "snapshot.npz:background_xyz",
        },
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
    anchor_mappings = [
        {
            "scene": "apartment",
            "object_id": f"object-{position}",
            "lifecycle_index": 0,
            "anchor_frame_index": 2,
            "anchor_relative_timestamp_ns": 200,
            "eligible": True,
            "target_voxel_count": 1,
            "mapped_temporal_id": position if position < 53 else None,
            "overlap_voxel_count": 1 if position < 53 else 0,
            "ambiguous": False,
        }
        for position in range(66)
    ]
    anchor_gate = {
        "scene": "apartment",
        "eligible_count": 66,
        "uniquely_mapped_count": 53,
        "zero_overlap_count": 13,
        "ambiguous_count": 0,
        "required_eligible_count": 66,
        "required_mapped_count": 53,
        "available": True,
        "passed": True,
        "reason": None,
    }
    occlusion = _write(root / "occlusion.json", {
        "format": "oviv2_temporal_compact_v1",
        "mapping_rule": "maximum_world_voxel_overlap_unique_winner_minimum_one_voxel",
        "anchor_mappings": anchor_mappings,
        "events": [],
        "mechanism_telemetry": mechanism_telemetry,
        "input_bindings": {
            "target_manifest": {"sha256": h("3"), "byte_count": 11},
            "indexes": [
                {
                    "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                    "byte_count": index.stat().st_size,
                }
            ],
            "source_indexes": [
                {
                    "scene": "apartment",
                    **_relative_record(run_source_index, run_root),
                }
            ],
            "maximum_cached_checkpoints": 1,
        },
        "macro": {
            "anchor_coverage_gate": anchor_gate,
            "anchor_mapping_coverage": {
                "available": True,
                "mapped": 53,
                "total": 66,
                "value": 53 / 66,
            },
            "occluded_retention_rate": {
                "available": False, "denominator": 0, "value": None,
                "unavailable_reason": "no_eligible_occluded_events",
            },
            "stale_removal_accuracy": {
                "available": False, "denominator": 0, "value": None,
                "unavailable_reason": "no_eligible_absent_events",
            },
            "reactivation_identity_accuracy": {
                "available": False, "denominator": 0, "value": None,
                "unavailable_reason": "no_eligible_reactivation_events",
            },
            "mechanism_telemetry": mechanism_telemetry,
        },
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
    dynamic_source = official_results / "dynamic_objects.csv"
    dynamic_source.write_text(
        "Name,Query,NumObjDetected,NumObjHallucinated,NumObjMissed\n"
        "0,0,4,1,1\n"
    )
    official_partial = summarize_khronos_official_metrics_partial(official_results)
    official = _write(root / "khronos/evaluation/official_metrics.json", {
        "status": "PASS", "dataset": "TESSE-CD", "scene": "apartment", "split": "apartment_test",
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
            _record(dynamic_source),
            _record(background_source),
        ],
    })
    trusted_sources = json.loads(
        (REPO_ROOT / "configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json").read_text()
    )["files"]
    protected = [
        {"path": path, "sha256": digest, "bytes": (REPO_ROOT / path).stat().st_size}
        for path, digest in trusted_sources.items()
    ]
    test_paths = (*package_module.T1_TEST_FILES, *package_module.DETERMINISM_TEST_FILES)
    tests = [
        {"path": path, "sha256": hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest(),
         "bytes": (REPO_ROOT / path).stat().st_size}
        for path in test_paths
    ]
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
    cumulative_exact = _materialize_gate_transaction(
        root / "exact-transaction", config, commit, source_manifest
    )
    evidence = _write(root / "development_gates.json", {
        "schema_version": 1, "manifest_id": "oviv2_dual_readout_development_gates_v1",
        "deterministic_evidence": {"base_commit": package_module.CUMULATIVE_BASE_COMMIT, "code_commit": commit, "code_tree": h("d"),
            "protected_files": protected, "test_sources": tests,
            "source_manifest": _record(source_manifest),
            "cumulative_exact": cumulative_exact,
            "gates": {"t1_exact": gate(t1_files, h("9")), "determinism": gate(determinism_files, h("b"))}},
        "receipt": {"created_at_utc": "2026-07-25T00:00:00Z"},
    })
    baseline_evidence = _baseline_evidence(root)
    return {"manifest": manifest, "search_status": status, "candidate_config": config_path, "run_manifest": run_manifest,
            "common_v2_summary": common, "temporal_occlusion_result": occlusion, "official_metrics": official,
            "t1_exact_evidence": evidence, "determinism_evidence": evidence,
            "short_gate_evidence": occlusion, "anchor_evidence": occlusion,
            "baseline_evidence": baseline_evidence}


def _package(paths: dict[str, Path], output: Path) -> dict[str, object]:
    return package_result(candidate_id="a2", output=output, **paths)


def test_packages_exact_structured_result_and_revalidates(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "result.json"
    result = _package(paths, output)
    assert set(result) == {
        "schema_version", "manifest_id", "candidate_id", "scene", "status",
        "evidence_scope", "sources", "bindings", "run_identity", "gates",
        "metrics", "profile", "component_map", "parameter_values",
        "mechanism_telemetry", "anchor_coverage_gate", "promotion_evidence",
    }
    assert result["evidence_scope"] == {
        "publication": "pre_and_post_link_revalidated",
        "snapshot": "point_in_time_not_permanent",
    }
    assert set(result["sources"]) == ({*paths} - {"manifest"}) | {"search_manifest"}
    assert all(set(record) == {"path", "sha256", "byte_count"} for record in result["sources"].values())
    assert set(result["gates"]) == {
        "correctness", "causality", "determinism", "t1_exact",
        "mechanisms", "anchor_coverage", "t2_metrics",
    }
    assert all(
        set(result["gates"][name]) == {"passed", "reason", "source"}
        and result["gates"][name]["passed"]
        for name in ("correctness", "causality", "determinism", "t1_exact")
    )
    assert result["profile"] == {
        "candidate_id": "a2",
        "kind": "main",
        "execution_profile": "a2",
        "selectable": True,
    }
    assert result["component_map"] == ExecutionProfile.A2.components
    assert result["parameter_values"] == {
        "lifecycle.minimum_absent_streak": 2,
        "proposal.minimum_depth_residual_m": 0.1,
        "dynamic_state.minimum_motion_confidence": 0.7,
        "motion.minimum_translation_confidence": 0.6,
        "motion.maximum_translation_residual_m": 0.25,
    }
    assert result["mechanism_telemetry"] == result["gates"]["mechanisms"]
    assert result["anchor_coverage_gate"] == result["gates"]["anchor_coverage"]
    assert result["anchor_coverage_gate"]["uniquely_mapped_count"] == 53
    assert set(result["gates"]["t2_metrics"]) == set(T2_DIRECTIONS)
    for name, gate in result["gates"]["t2_metrics"].items():
        assert set(gate) == {
            "direction", "baseline", "value", "available", "passed", "source"
        }
        assert gate["direction"] == T2_DIRECTIONS[name]
        assert gate["available"] is True
        assert gate["passed"] is True
    assert result["promotion_evidence"] == {
        "selectable": True,
        "passed": True,
        "failed_gates": [],
        "source": "derived_from_source_backed_gates",
    }
    assert "t4" not in json.dumps(result).lower()
    assert result["metrics"]["background_f5_cm"]["value"] == 0.3
    assert result["metrics"]["runtime_seconds"]["value"] == 12.5
    assert result["metrics"]["dynamic_f1"]["available"] is True
    assert load_and_revalidate_result(output, manifest=paths["manifest"]) == result


def test_recomputes_mechanism_telemetry_and_fails_closed_on_empty_opportunity(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    occlusion = json.loads(paths["temporal_occlusion_result"].read_text())
    telemetry = occlusion["mechanism_telemetry"]["proposal_recovery"]
    diagnostics_path = paths["run_manifest"].parent / telemetry["source"]["path"]
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["counters"]["proposal_opportunity_count"] = 0
    diagnostics["counters"]["proposal_trigger_count"] = 0
    _write(diagnostics_path, diagnostics)
    diagnostics_record = _relative_record(diagnostics_path, paths["run_manifest"].parent)
    source_index_path = paths["run_manifest"].parent / "source_index.json"
    source_index = json.loads(source_index_path.read_text())
    source_index["runtime_diagnostics"] = diagnostics_record
    _write(source_index_path, source_index)
    run_manifest = json.loads(paths["run_manifest"].read_text())
    run_manifest["source_index"] = _relative_record(
        source_index_path, paths["run_manifest"].parent
    )
    _write(paths["run_manifest"], run_manifest)
    for name in ("proposal_recovery", "epoch_reset"):
        occlusion["mechanism_telemetry"][name]["source"] = diagnostics_record
    telemetry.update(opportunities=1, triggers=1, available=True, passed=True)
    occlusion["macro"]["mechanism_telemetry"] = occlusion["mechanism_telemetry"]
    occlusion["input_bindings"]["source_indexes"][0] = {
        "scene": "apartment",
        **run_manifest["source_index"],
    }
    _write(paths["temporal_occlusion_result"], occlusion)

    with pytest.raises(ValueError, match="proposal_recovery.*source|opportunit"):
        _package(paths, tmp_path / "result.json")


def test_rejects_manual_pass_boolean_in_formal_evaluator_artifact(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    occlusion = json.loads(paths["temporal_occlusion_result"].read_text())
    occlusion["cached_short_gate_passed"] = True
    _write(paths["temporal_occlusion_result"], occlusion)

    with pytest.raises(ValueError, match="format|schema|exact"):
        _package(paths, tmp_path / "manual-pass.json")


@pytest.mark.parametrize(("mapped", "eligible"), [(52, 66), (0, 0)])
def test_recomputes_anchor_mapping_gate_and_requires_53_of_66(
    tmp_path: Path, mapped: int, eligible: int
) -> None:
    paths = _fixture(tmp_path)
    occlusion = json.loads(paths["temporal_occlusion_result"].read_text())
    mappings = occlusion["anchor_mappings"]
    if eligible == 0:
        mappings.clear()
    else:
        mappings[52]["mapped_temporal_id"] = None
        mappings[52]["overlap_voxel_count"] = 0
    occlusion["macro"]["anchor_coverage_gate"].update(
        eligible_count=66,
        uniquely_mapped_count=53,
        available=True,
        passed=True,
        reason=None,
    )
    _write(paths["temporal_occlusion_result"], occlusion)

    with pytest.raises(ValueError, match="anchor.*(source|coverage|eligible|53)"):
        _package(paths, tmp_path / f"anchor-{mapped}-{eligible}.json")


@pytest.mark.parametrize("metric", list(T2_DIRECTIONS))
def test_each_t2_metric_is_an_independent_source_backed_gate(
    tmp_path: Path, metric: str
) -> None:
    paths = _fixture(tmp_path)
    baseline = json.loads(paths["baseline_evidence"].read_text())
    source_path = Path(baseline["baselines"][metric]["source"]["path"])
    source = json.loads(source_path.read_text())
    if metric in {"ghost_rate", "recovery_frames"}:
        source["metrics"][metric] = 0.0
    else:
        source["metrics"][metric] = 1.0
    _write(source_path, source)
    for entry in baseline["baselines"].values():
        if Path(entry["source"]["path"]) == source_path:
            entry["source"] = _record(source_path)
    _write(paths["baseline_evidence"], baseline)

    result = _package(paths, tmp_path / f"failed-{metric}.json")

    assert result["gates"]["t2_metrics"][metric]["passed"] is False
    assert result["promotion_evidence"] == {
        "selectable": True,
        "passed": False,
        "failed_gates": [f"t2_metrics.{metric}"],
        "source": "derived_from_source_backed_gates",
    }


def test_current_and_object_noninferiority_accept_equality(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    baseline = json.loads(paths["baseline_evidence"].read_text())
    release_path = Path(baseline["baselines"]["object_f1"]["source"]["path"])
    release = json.loads(release_path.read_text())
    official = json.loads(paths["official_metrics"].read_text())
    common = json.loads(paths["common_v2_summary"].read_text())
    release["metrics"]["object_f1"] = official["metrics"]["object_f1"]
    release["metrics"]["current_miou"] = common["metrics"]["current_miou"]
    _write(release_path, release)
    for entry in baseline["baselines"].values():
        if Path(entry["source"]["path"]) == release_path:
            entry["source"] = _record(release_path)
    _write(paths["baseline_evidence"], baseline)

    result = _package(paths, tmp_path / "equal-noninferior.json")

    assert result["gates"]["t2_metrics"]["object_f1"]["passed"] is True
    assert result["gates"]["t2_metrics"]["current_miou"]["passed"] is True


def test_rejects_unknown_or_profile_incompatible_parameter(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = json.loads(paths["manifest"].read_text())
    declaration = next(
        item for item in manifest["candidates"] if item["candidate_id"] == "a2"
    )
    declaration["tunable_parameters"] = ["unknown.axis"]
    _write(paths["manifest"], manifest)
    status = json.loads(paths["search_status"].read_text())
    status["manifest"] = _record(paths["manifest"])
    _write(paths["search_status"], status)

    with pytest.raises(ValueError, match="manifest|parameter|keys"):
        _package(paths, tmp_path / "unknown-parameter.json")


def test_revalidation_rejects_micro_diagnostic_masquerading_as_main(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "result.json"
    result = _package(paths, output)
    result["profile"] = {
        "candidate_id": "a2_no_proposal",
        "kind": "main",
        "execution_profile": "a2",
        "selectable": True,
    }
    result["candidate_id"] = "a2_no_proposal"
    output.write_bytes(_bytes(result))

    with pytest.raises(ValueError, match="candidate|A0-A4|manifest|revalidated"):
        load_and_revalidate_result(output, manifest=paths["manifest"])


def test_exact_transaction_uses_reference_and_dual_production_receipt_schemas(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    evidence = json.loads(paths["t1_exact_evidence"].read_text())
    executions = evidence["deterministic_evidence"]["cumulative_exact"]["executions"]

    assert len(executions) == 9
    for execution in executions:
        root = Path(execution["output_root"])
        manifest = json.loads((root / "run_manifest.json").read_text())
        if execution["profile"] == "reference":
            assert manifest["schema_version"] == 1
            assert (root / "t1_exact_receipt.json").is_file()
            assert not (root / "execution_receipt.json").exists()
        else:
            assert manifest["schema_version"] == 2
            assert (root / "execution_receipt.json").is_file()
            assert not (root / "t1_exact_receipt.json").exists()


def _observe_exact_verifications(
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    calls: list[int] = []
    original = package_module.verify_exact_profile_runs

    def observe(executions: list[dict[str, object]]) -> dict[str, object]:
        calls.append(len(executions))
        return original(executions)

    monkeypatch.setattr(package_module, "verify_exact_profile_runs", observe)
    return calls


def test_shared_exact_transaction_is_verified_once_per_publication_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path / "evidence")
    calls = _observe_exact_verifications(monkeypatch)

    result = _package(paths, tmp_path / "publication/result.json")

    assert result["status"] == "PASS"
    assert calls == [9, 9, 9]
    assert sum(calls) == 27


def test_distinct_exact_transactions_are_never_deduplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path / "evidence")
    determinism = json.loads(paths["determinism_evidence"].read_text())
    config = json.loads(paths["candidate_config"].read_text())
    source_manifest = (
        REPO_ROOT
        / "configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json"
    )
    determinism["deterministic_evidence"]["cumulative_exact"] = (
        _materialize_gate_transaction(
            tmp_path / "distinct-exact-transaction",
            config,
            "c" * 40,
            source_manifest,
        )
    )
    paths["determinism_evidence"] = _write(
        tmp_path / "distinct-determinism-evidence.json", determinism
    )
    calls = _observe_exact_verifications(monkeypatch)

    with pytest.raises(ValueError, match="evidence transactions differ"):
        _package(paths, tmp_path / "publication/result.json")

    assert calls == [9, 9]
    assert sum(calls) == 18


def test_rejects_gate_evidence_with_nonexistent_execution_roots(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    evidence = json.loads(paths["t1_exact_evidence"].read_text())
    exact = evidence["deterministic_evidence"]["cumulative_exact"]
    exact["executions"][0]["output_root"] = str(
        (tmp_path / "missing-execution-root").resolve()
    )
    _write(paths["t1_exact_evidence"], evidence)

    with pytest.raises(ValueError, match="execution root|receipt|manifest"):
        _package(paths, tmp_path / "result.json")


def test_rejects_run_manifest_with_extra_field(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = json.loads(paths["run_manifest"].read_text())
    manifest["untrusted_extension"] = True
    _write(paths["run_manifest"], manifest)

    with pytest.raises(ValueError, match="field inventory"):
        _package(paths, tmp_path / "result.json")


def test_rejects_stale_final_current_map_record(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = json.loads(paths["run_manifest"].read_text())
    snapshot = paths["run_manifest"].parent / manifest["final_current_map"]["snapshot"]["path"]
    snapshot.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="final current map snapshot"):
        _package(paths, tmp_path / "result.json")


def test_new_runner_artifacts_cannot_omit_final_current_map_binding(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    manifest = json.loads(paths["run_manifest"].read_text())
    manifest.pop("final_current_map")
    _write(paths["run_manifest"], manifest)

    with pytest.raises(ValueError, match="lacks final current map"):
        _package(paths, tmp_path / "result.json")


def test_rejects_stale_final_map_metadata_and_undeclared_directory(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path / "metadata")
    manifest = json.loads(paths["run_manifest"].read_text())
    snapshot = paths["run_manifest"].parent / manifest["final_current_map"]["snapshot"]["path"]
    np.savez_compressed(
        snapshot,
        background_xyz=np.asarray([[1.0, 2.0, 3.0]]),
        timestamp=np.asarray(100.0),
        scope=np.asarray("current"),
        scene_id=np.asarray("apartment"),
    )
    manifest["final_current_map"]["snapshot"] = _relative_record(
        snapshot, paths["run_manifest"].parent
    )
    _write(paths["run_manifest"], manifest)
    with pytest.raises(ValueError, match="snapshot metadata is stale"):
        _package(paths, tmp_path / "metadata-result.json")

    paths = _fixture(tmp_path / "coverage")
    manifest = json.loads(paths["run_manifest"].read_text())
    snapshot = paths["run_manifest"].parent / manifest["final_current_map"]["snapshot"]["path"]
    np.savez_compressed(
        snapshot,
        background_xyz=np.asarray([[1.0, 2.0, 3.0]]),
        timestamp=np.asarray(200.0),
        scope=np.asarray("current"),
        scene_id=np.asarray("apartment"),
    )
    manifest["final_current_map"]["timestamp_ns"] = 200
    manifest["final_current_map"]["snapshot"] = _relative_record(
        snapshot, paths["run_manifest"].parent
    )
    _write(paths["run_manifest"], manifest)
    with pytest.raises(ValueError, match="run-end frame coverage"):
        _package(paths, tmp_path / "coverage-result.json")

    paths = _fixture(tmp_path / "external")
    manifest = json.loads(paths["run_manifest"].read_text())
    outside_coverage = tmp_path / "external-coverage.jsonl"
    outside_coverage.write_text(
        json.dumps({"frame_index": 2, "timestamp_ns": 300}) + "\n",
        encoding="utf-8",
    )
    outside_index = _write(tmp_path / "external-source-index.json", {
        "frame_coverage": _relative_record(outside_coverage, tmp_path),
    })
    manifest["source_index"] = _relative_record(outside_index, tmp_path)
    manifest["source_index"]["path"] = str(outside_index.absolute())
    _write(paths["run_manifest"], manifest)
    with pytest.raises(ValueError, match="canonical run artifact"):
        _package(paths, tmp_path / "external-result.json")

    paths = _fixture(tmp_path / "directory")
    (paths["run_manifest"].parent / "undeclared-empty").mkdir()
    with pytest.raises(ValueError, match="directory inventory"):
        _package(paths, tmp_path / "directory-result.json")


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
        ("wrong_exact_argv", "argv"),
        ("duplicate_root", "root"),
        ("ancestor_root", "root"),
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
    elif mutation == "wrong_exact_argv":
        evidence["cumulative_exact"]["executions"][2]["argv"] = ["python", "runner.py"]
    elif mutation == "duplicate_root":
        evidence["cumulative_exact"]["executions"][2]["output_root"] = evidence["cumulative_exact"]["executions"][1]["output_root"]
    elif mutation == "ancestor_root":
        parent = Path(evidence["cumulative_exact"]["executions"][1]["output_root"])
        child = str((parent / "child").resolve())
        record = evidence["cumulative_exact"]["executions"][2]
        record["output_root"] = child
        record["argv"][5] = child
    else:
        evidence["base_commit"] = "f" * 40
    _write(paths["t1_exact_evidence"], payload)
    with pytest.raises(ValueError, match=match):
        _package(paths, tmp_path / "result.json")


@pytest.mark.parametrize(
    ("collection", "mutation"),
    [
        ("protected_files", "missing"),
        ("protected_files", "extra"),
        ("protected_files", "duplicate"),
        ("protected_files", "path"),
        ("test_sources", "missing"),
        ("test_sources", "extra"),
        ("test_sources", "duplicate"),
        ("test_sources", "path"),
    ],
)
def test_rejects_noncanonical_protected_and_test_source_sets(
    tmp_path: Path, collection: str, mutation: str
) -> None:
    paths = _fixture(tmp_path)
    payload = json.loads(paths["t1_exact_evidence"].read_text())
    records = payload["deterministic_evidence"][collection]
    if mutation == "missing":
        records.pop()
    elif mutation == "extra":
        records.append({"path": "arbitrary.py", "sha256": "f" * 64, "bytes": 1})
    elif mutation == "duplicate":
        records.append(dict(records[0]))
    else:
        records[0]["path"] = "arbitrary.py"
    _write(paths["t1_exact_evidence"], payload)
    with pytest.raises(ValueError, match="protected|test source"):
        _package(paths, tmp_path / "result.json")


def _tree_binding(path: Path, root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    byte_count = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        data = item.read_bytes()
        relative = item.relative_to(path).as_posix()
        byte_count += len(data)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(hashlib.sha256(data).hexdigest()))
        digest.update(b"\n")
    return {"path": path.relative_to(root).as_posix(), "sha256": digest.hexdigest(), "byte_count": byte_count}


def _materialize_mutation_transaction(
    root: Path,
    config: dict[str, object],
    *,
    leak_temporal: bool = False,
    commit: str = "a" * 40,
) -> dict[str, object]:
    algorithm = canonical_algorithm_hash(config)
    non_temporal = non_temporal_config_sha256(config)
    root.mkdir(parents=True)
    config_path = root.parent / f"{root.name}.config.json"
    config_path.write_bytes(_bytes(config))
    input_path = Path(str(config["input_manifest"]))
    schedule_path = Path(str(config["schedule_manifest"]))
    target_path = Path(str(config["occlusion_target_manifest"]))
    content_record = lambda path: {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }
    normalized = root / "normalized_run_config.json"
    normalized.write_bytes(
        _bytes(
            {
                **config,
                "algorithm_hash": algorithm,
            }
        )
    )
    checkpoint = root / "checkpoints/00000002-100"
    audit_root = checkpoint / "cumulative_audit"
    artifact = audit_root / "artifact"
    voxel = audit_root / "voxel_snapshot"
    artifact.mkdir(parents=True)
    voxel.mkdir()
    cumulative_bytes = (algorithm if leak_temporal else non_temporal).encode()
    (artifact / "neutral.bin").write_bytes(cumulative_bytes)
    (artifact / "entities.jsonl").write_bytes(cumulative_bytes + b"\n")
    (artifact / "manifest.json").write_bytes(b'{"format":"cumulative-v1"}\n')
    (artifact / "checkpoint_status.json").write_bytes(b'{"status":"PASS"}\n')
    (artifact / "final.bin").write_bytes(b"final-cumulative")
    (voxel / "ownership.bin").write_bytes(cumulative_bytes)
    status = checkpoint / "checkpoint_status.json"
    status.write_bytes(_bytes({"algorithm_hash": algorithm, "status": "PASS"}))
    source = REPO_ROOT / "configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json"
    source_data = source.read_bytes()
    source_sha = hashlib.sha256(source_data).hexdigest()
    cumulative_audit = {
        "format": "oviv2_cumulative_audit_v1",
        "artifact": _tree_binding(artifact, root),
        "snapshot": _file_binding(artifact / "neutral.bin", root),
        "entities": _file_binding(artifact / "entities.jsonl", root),
        "voxel_snapshot": _tree_binding(voxel, root),
    }
    schedule_copy = root / "inputs/schedule.json"
    schedule_copy.parent.mkdir()
    schedule_copy.write_bytes(schedule_path.read_bytes())
    capture_status = _write(root / "capture_status.json", {"status": "PASS"})
    trajectories = root / "trajectories.jsonl"
    trajectories.write_bytes(b"")
    frame_coverage = root / "temporal_frame_coverage.jsonl"
    frame_coverage.write_bytes(b"")
    lifecycle = root / "lifecycle_transitions.jsonl"
    lifecycle.write_bytes(b"")
    source_index = _write(
        root / "source_index.json",
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoint_exports",
            "method": "OVIV2",
            "scene": "apartment",
            "schedule": _file_binding(schedule_copy, root),
            "capture_status": _file_binding(capture_status, root),
            "trajectories": _file_binding(trajectories, root),
            "frame_coverage": _file_binding(frame_coverage, root),
            "lifecycle_transitions": _file_binding(lifecycle, root),
            "checkpoints": [
                {
                    "frame_index": 2,
                    "timestamp_ns": 100,
                    "consumed_through_frame": 2,
                    "consumed_through_frame_exclusive": 3,
                    "checkpoint_status": _file_binding(status, root),
                    "snapshot": _file_binding(artifact / "neutral.bin", root),
                    "entities": _file_binding(artifact / "entities.jsonl", root),
                }
            ],
        },
    )
    occlusion_index = _write(
        root / "occlusion_checkpoint_index.json", {"status": "PASS"}
    )
    artifact_record = _tree_binding(artifact, root)
    checksum = hashlib.sha256(b"fixture-checkpoint").hexdigest()
    checkpoint_record = {
        "scene": "apartment",
        "frame_index": 2,
        "timestamp_ns": 100,
        "relative_timestamp_ns": 0,
        "consumed_through_frame": 2,
        "consumed_through_frame_exclusive": 3,
        "event_ids": [],
        "roles": ["cumulative"],
        "format": "fixture",
        "artifact": artifact_record,
        "checksums_sha256": checksum,
        "artifacts": {
            "neutral_current": {
                "format": "fixture",
                "artifact": artifact_record,
                "checksums_sha256": checksum,
            }
        },
        "cumulative_audit": cumulative_audit,
        "checkpoint_status": _file_binding(status, root),
    }
    manifest = {
        "schema_version": 2,
        "protocol_id": "oviv2-tessecd-v2",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "scene": "apartment",
        "mode": "dual_readout_causal_checkpoints",
        "algorithm_hash": algorithm,
        "processed_frame_count": 3,
        "covered_frame_count": 3,
        "trajectory_frame_count": 0,
        "first_frame_index": 0,
        "last_frame_index": 2,
        "temporal_export_schema_version": 1,
        "scheduled_frame_indices": [2],
        "captured_frame_indices": [2],
        "code_commit": commit,
        "config": content_record(config_path),
        "schedule": content_record(schedule_path),
        "target_manifest": content_record(target_path),
        "source_bindings": {
            "source_manifest_sha256": source_sha,
            "input_manifest": content_record(input_path),
        },
        "input_sha256": hashlib.sha256(b"fixture-inputs").hexdigest(),
        "normalized_run_config": _file_binding(normalized, root),
        "checkpoints": [checkpoint_record],
        "occlusion_checkpoint_index": _file_binding(occlusion_index, root),
        "source_index": _file_binding(source_index, root),
    }
    manifest["artifact_inventory"] = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    (root / "run_manifest.json").write_bytes(_bytes(manifest))
    (root / "execution_receipt.json").write_bytes(
        _bytes(
            {
                "schema_version": 1,
                "provenance": _production_receipt_provenance(algorithm, commit),
                "environment": {"pid": os.getpid()},
            }
        )
    )
    return compare_cumulative_artifacts(root, root)


def _materialize_reference_transaction(
    root: Path,
    config: dict[str, object],
    *,
    commit: str,
    source_manifest: Path,
) -> dict[str, object]:
    algorithm = canonical_algorithm_hash(config)
    cumulative_bytes = non_temporal_config_sha256(config).encode()
    root.mkdir(parents=True)
    config_path = root.parent / f"{root.name}.config.json"
    config_path.write_bytes(_bytes(config))
    input_path = Path(str(config["input_manifest"]))
    schedule_path = Path(str(config["schedule_manifest"]))
    target_path = Path(str(config["occlusion_target_manifest"]))
    content_record = lambda path: {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }
    checkpoint = root / "checkpoints/00000002-100"
    artifact = checkpoint / "artifact"
    voxel = checkpoint / "voxel_snapshot"
    artifact.mkdir(parents=True)
    voxel.mkdir()
    neutral = artifact / "neutral.bin"
    entities = artifact / "entities.jsonl"
    neutral.write_bytes(cumulative_bytes)
    entities.write_bytes(cumulative_bytes + b"\n")
    (artifact / "manifest.json").write_bytes(b'{"format":"cumulative-v1"}\n')
    (artifact / "checkpoint_status.json").write_bytes(b'{"status":"PASS"}\n')
    (artifact / "final.bin").write_bytes(b"final-cumulative")
    (voxel / "ownership.bin").write_bytes(cumulative_bytes)
    status = checkpoint / "checkpoint_status.json"
    status.write_bytes(_bytes({"algorithm_hash": algorithm, "status": "PASS"}))
    final = root / "final.bin"
    final.write_bytes(algorithm.encode())
    source_manifest = source_manifest.resolve(strict=True)
    source_data = source_manifest.read_bytes()
    source_sha = hashlib.sha256(source_data).hexdigest()
    manifest = {
        "schema_version": 1,
        "scene": "apartment",
        "mode": "causal_checkpoints",
        "algorithm_hash": algorithm,
        "code_commit": commit,
        "config": content_record(config_path),
        "schedule": content_record(schedule_path),
        "target_manifest": content_record(target_path),
        "source_bindings": {
            "source_manifest_sha256": source_sha,
            "input_manifest": content_record(input_path),
        },
        "checkpoints": [
            {
                "frame_index": 2,
                "artifact": _tree_binding(artifact, root),
                "voxel_snapshot": _tree_binding(voxel, root),
                "checkpoint_status": _file_binding(status, root),
                "neutral_snapshot": _file_binding(neutral, root),
                "neutral_entities": _file_binding(entities, root),
            }
        ],
        "final_artifact": _file_binding(final, root),
    }
    (root / "run_manifest.json").write_bytes(_bytes(manifest))
    audit = compare_cumulative_artifacts(root, root)
    receipt = {
        "schema_version": 1,
        "format": "oviv2_t1_exact_execution_receipt_v1",
        "execution": {
            "profile": "reference",
            "mode": gates_module.DEVELOPMENT_MODE,
            "argv": ["pending-reference-command"],
            "pid": os.getpid(),
            "code_commit": commit,
            "source_manifest_sha256": source_sha,
            "input_fingerprints": {
                "config": hashlib.sha256(config_path.read_bytes()).hexdigest()
            },
            "output_root": str(root.resolve()),
        },
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": source_sha,
            "byte_count": len(source_data),
        },
        "artifact_inventory": audit["inventory"],
        "checkpoint_frames": audit["checkpoint_frames"],
        "cumulative_root_sha256": audit["root_sha256"],
    }
    receipt_path = root / "t1_exact_receipt.json"
    receipt_path.write_bytes(_bytes(receipt))
    assert json.loads(receipt_path.read_text()) == receipt
    return audit


def _file_binding(path: Path, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _production_receipt_provenance(
    algorithm: str, commit: str = "a" * 40
) -> dict[str, object]:
    return {
        "repository_commit": commit,
        "repository_tree": "b" * 40,
        "dirty_state_digest": hashlib.sha256(b"").hexdigest(),
        "command": ["python", "run_oviv2_tesse_cd_v2.py", algorithm],
        "hostname": "fixture-host",
        "platform": "fixture-platform",
        "machine": "x86_64",
        "cuda_visible_devices": None,
        "torch_cuda_version": "unavailable",
        "cudnn_version": None,
        "nvcc_version": [],
        "gpu_inventory": [],
        "library_versions": {
            name: "fixture"
            for name in ("numpy", "open3d", "torch", "scipy", "pillow")
        },
    }


def _materialize_gate_transaction(
    transaction: Path,
    config: dict[str, object],
    commit: str,
    source_manifest: Path,
) -> dict[str, object]:
    transaction.mkdir(parents=True)
    input_manifest = _write(transaction / "input_manifest.json", {"fixture": "input"})
    schedule_manifest = _write(transaction / "schedule_manifest.json", {"fixture": "schedule"})
    target_manifest = _write(transaction / "target_manifest.json", {"fixture": "target"})
    transaction_config = {
        **config,
        "scene": "apartment",
        "input_manifest": str(input_manifest.resolve()),
        "schedule_manifest": str(schedule_manifest.resolve()),
        "occlusion_target_manifest": str(target_manifest.resolve()),
        "occlusion_target_manifest_sha256": hashlib.sha256(
            target_manifest.read_bytes()
        ).hexdigest(),
    }
    transaction_config["algorithm_hash"] = canonical_algorithm_hash(
        transaction_config
    )
    receipts = transaction / "receipts"
    receipts.mkdir()
    executions: list[dict[str, object]] = []
    python = str(Path(sys.executable).resolve())
    source_manifest = source_manifest.resolve(strict=True)
    for position, profile in enumerate(gates_module.EXACT_PROFILE_SEQUENCE):
        output = transaction / f"run-{position:02d}-{profile}"
        if profile == "reference":
            _materialize_reference_transaction(
                output,
                transaction_config,
                commit=commit,
                source_manifest=source_manifest,
            )
        else:
            _materialize_mutation_transaction(
                output, transaction_config, commit=commit
            )
        config_path = (transaction / f"run-{position:02d}-{profile}.config.json").resolve()
        pid = 10_000 + position
        runner = (
            REPO_ROOT / "scripts/evaluation/run_oviv2_t1_reference.py"
            if profile == "reference"
            else REPO_ROOT / "scripts/evaluation/run_oviv2_tesse_cd_v2.py"
        ).resolve()
        argv = [
            python,
            str(runner),
            "--config",
            str(config_path),
            "--output",
            str(output.resolve()),
        ]
        if profile == "reference":
            argv.extend(
                [
                    "--receipt",
                    str((output / "t1_exact_receipt.json").resolve()),
                    "--source-manifest",
                    str(source_manifest),
                ]
            )
            receipt_path = output / "t1_exact_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["execution"].update(
                {
                    "profile": profile,
                    "argv": argv,
                    "pid": pid,
                    "code_commit": commit,
                    "output_root": str(output.resolve()),
                }
            )
            receipt_path.write_bytes(_bytes(receipt))
        else:
            execution_receipt_path = output / "execution_receipt.json"
            execution_receipt = json.loads(execution_receipt_path.read_text())
            execution_receipt["provenance"]["command"] = argv[1:]
            execution_receipt_path.write_bytes(_bytes(execution_receipt))
        derived = gates_module._reopen_completed_execution(
            profile, output.resolve(), argv, pid, 0, source_manifest
        )
        root_status = os.stat(output, follow_symlinks=False)
        production_receipt = output / (
            "t1_exact_receipt.json" if profile == "reference" else "execution_receipt.json"
        )
        observation = {
            "schema_version": 1,
            "format": "oviv2_exact_process_observation_v1",
            "position": position,
            "profile": profile,
            "argv": argv,
            "pid": pid,
            "returncode": 0,
            "config": gates_module._absolute_file_record(config_path),
            "source_manifest": gates_module._absolute_file_record(source_manifest),
            "output_root": str(output.resolve()),
            "root_device": root_status.st_dev,
            "root_inode": root_status.st_ino,
            "run_manifest": gates_module._absolute_file_record(output / "run_manifest.json"),
            "production_receipt": gates_module._absolute_file_record(production_receipt),
            "completed_execution": derived,
            "trust_model": gates_module.LOCAL_PROCESS_TRUST_MODEL,
            "execution_context": gates_module.DEVELOPMENT_EXECUTION_CONTEXT,
        }
        observation_path = _write(
            receipts / f"{position:03d}-{profile}.json", observation
        )
        executions.append(
            {
                **derived,
                "observation_receipt": gates_module._absolute_file_record(
                    observation_path.resolve()
                ),
            }
        )
    return gates_module.verify_exact_profile_runs(executions)


_TEMPORAL_IDENTITY_BOUND_LEAVES = {
    ("geometry", "voxel_size_m"),
    ("geometry", "depth_max_m"),
}
_TEMPORAL_FIXED_LEAVES = {
    ("motion", "require_explicit_rejection"),
}
_TEMPORAL_DISCRETE_REPLACEMENTS = {
    ("dynamic_state", "displacement_floor_m"): 0.15,
    ("dynamic_state", "minimum_motion_confidence"): 0.8,
    ("dynamic_state", "static_off_streak_frames"): 20,
}


def _temporal_mutation_cases() -> tuple[tuple[tuple[str, ...], object], ...]:
    temporal = runner_fixture._temporal_readout()
    cases: list[tuple[tuple[str, ...], object]] = [
        (
            ("execution_profile", "components"),
            (ExecutionProfile.A3.profile_id, ExecutionProfile.A3.components),
        )
    ]
    for section, values in temporal.items():
        if section in {"execution_profile", "components"}:
            continue
        assert isinstance(values, dict)
        for leaf, value in values.items():
            if (section, leaf) in (
                _TEMPORAL_IDENTITY_BOUND_LEAVES | _TEMPORAL_FIXED_LEAVES
            ):
                continue
            if (section, leaf) in _TEMPORAL_DISCRETE_REPLACEMENTS:
                replacement = _TEMPORAL_DISCRETE_REPLACEMENTS[(section, leaf)]
            elif isinstance(value, bool):
                replacement: object = not value
            elif isinstance(value, int):
                replacement = value + 1
                if (section, leaf) in {
                    ("proposal", "maximum_recovered_proposals"),
                    ("background_ledger", "maximum_journal_blocks"),
                }:
                    replacement = value - 1
            else:
                assert isinstance(value, float)
                replacement = value - 0.000001 if value < 0.0 else value + 0.000001
            cases.append(((section, leaf), replacement))
    return tuple(cases)


TEMPORAL_MUTATION_CASES = _temporal_mutation_cases()


class _TwoFrameDataset(runner_fixture._Dataset):
    def __len__(self) -> int:
        return 2


def _two_frame_config(tmp_path: Path) -> Path:
    baseline_config = runner_fixture._materialize_config(
        production_runner, tmp_path
    )
    config = json.loads(baseline_config.read_text())
    config["frame_count"] = 2
    config["evaluation_checkpoint_frames"] = [0, 1]
    schedule_path = Path(config["schedule_manifest"])
    schedule = json.loads(schedule_path.read_text())
    for scene in ("apartment", "office"):
        schedule["scenes"][scene]["frame_count"] = 2
        schedule["scenes"][scene]["entries"] = [
            {
                "event_ids": ["event-official", "event-common"],
                "frame_index": 1,
                "relative_timestamp_ns": 10,
                "roles": ["official", "common_v2"],
                "timestamp_ns": 110,
            }
        ]
    _write(schedule_path, schedule)
    target_path = Path(config["occlusion_target_manifest"])
    target = json.loads(target_path.read_text())
    target["metadata"]["scene_frame_indices"] = {
        "apartment": [0, 1],
        "office": [0, 1],
    }
    target["metadata"]["episodes"] = [
        {
            "scene": scene,
            "anchor": {"frame_index": 0, "relative_timestamp_ns": 0},
            "checkpoints": [{"frame_index": 1, "relative_timestamp_ns": 10}],
        }
        for scene in ("apartment", "office")
    ]
    _write(target_path, target)
    config["occlusion_target_manifest_sha256"] = hashlib.sha256(
        target_path.read_bytes()
    ).hexdigest()
    evaluation_plan = {
        "schema_version": 1,
        "manifest_id": "tesse_cd_occlusion_v1_checkpoint_frames",
        "evaluation_checkpoint_frames": {
            "apartment": [0, 1],
            "office": [0, 1],
        },
    }
    config["evaluation_checkpoint_frames_sha256"] = hashlib.sha256(
        json.dumps(
            evaluation_plan,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    config["algorithm_hash"] = production_runner.algorithm_hash(config)
    return _write(baseline_config, config)


def _two_frame_dependencies(config: dict[str, object]):
    baseline_non_temporal = non_temporal_config_sha256(config)
    provenance = _production_receipt_provenance(
        config["algorithm_hash"], "a" * 40
    )
    dependencies, _ = runner_fixture._dependencies(
        production_runner, provenance=provenance
    )

    def load_caches(runtime_config: object, dataset: object):
        caches = dependencies.cache_loader_factory(runtime_config, dataset)
        caches.temporal_config = temporal_config_from_json(
            {"temporal_readout": config["temporal_readout"]}
        )
        return caches

    return (
        production_runner.RunnerDependencies(
            dataset_factory=lambda unused: _TwoFrameDataset(),
            cache_loader_factory=load_caches,
            runtime_factory=dependencies.runtime_factory,
            provenance_factory=dependencies.provenance_factory,
            environment_factory=dependencies.environment_factory,
        ),
        baseline_non_temporal,
    )


def _compare_cumulative_with_run_end_map(left: Path, right: Path) -> dict[str, object]:
    saved: dict[Path, bytes] = {}
    moved: dict[Path, Path] = {}
    try:
        for root in {left, right}:
            manifest_path = root / "run_manifest.json"
            saved[manifest_path] = manifest_path.read_bytes()
            manifest = json.loads(saved[manifest_path])
            manifest.pop("final_current_map")
            manifest["artifact_inventory"] = [
                item for item in manifest["artifact_inventory"]
                if not item.startswith("final_current_map/")
            ]
            _write(manifest_path, manifest)
            final_root = root / "final_current_map"
            hidden = root.parent / f".{root.name}-final-current-map"
            final_root.rename(hidden)
            moved[final_root] = hidden
        return compare_cumulative_artifacts(left, right)
    finally:
        for original, hidden in moved.items():
            hidden.rename(original)
        for path, data in saved.items():
            path.write_bytes(data)


@pytest.fixture(scope="module")
def _temporal_mutation_baseline(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path, dict[str, object]]:
    root = tmp_path_factory.mktemp("temporal-mutation-baseline")
    baseline_config = _two_frame_config(root / "input")
    config = json.loads(baseline_config.read_text())
    baseline_dependencies, baseline_non_temporal = _two_frame_dependencies(config)
    baseline = root / "baseline"
    production_runner.run(
        baseline_config, baseline, dependencies=baseline_dependencies
    )
    baseline_audit = _compare_cumulative_with_run_end_map(baseline, baseline)
    assert json.loads((baseline / "run_manifest.json").read_text())["processed_frame_count"] == 2
    return config, baseline, {
        "non_temporal": baseline_non_temporal,
        "audit": baseline_audit,
    }


@pytest.mark.parametrize(
    ("leaf_path", "replacement"),
    TEMPORAL_MUTATION_CASES,
    ids=lambda value: ".".join(value) if isinstance(value, tuple) and all(isinstance(item, str) for item in value) else None,
)
def test_every_independently_configurable_temporal_leaf_runs_exact_cumulative_transaction(
    tmp_path: Path,
    _temporal_mutation_baseline: tuple[dict[str, object], Path, dict[str, object]],
    leaf_path: tuple[str, ...],
    replacement: object,
) -> None:
    config, baseline, expected = _temporal_mutation_baseline
    mutated = json.loads(json.dumps(config))
    if leaf_path == ("execution_profile", "components"):
        profile, components = replacement
        mutated["temporal_readout"]["execution_profile"] = profile
        mutated["temporal_readout"]["components"] = components
    else:
        section, leaf = leaf_path
        mutated["temporal_readout"][section][leaf] = replacement
    mutated["algorithm_hash"] = production_runner.algorithm_hash(mutated)
    mutation_config = _write(tmp_path / "mutation-input/config.json", mutated)
    mutation_dependencies, _ = _two_frame_dependencies(mutated)
    mutation = tmp_path / "mutation"
    production_runner.run(
        mutation_config, mutation, dependencies=mutation_dependencies
    )

    assert mutated["algorithm_hash"] != config["algorithm_hash"]
    assert non_temporal_config_sha256(mutated) == expected["non_temporal"]
    assert _compare_cumulative_with_run_end_map(baseline, mutation) == expected["audit"]
    assert json.loads((mutation / "run_manifest.json").read_text())["processed_frame_count"] == 2


@pytest.mark.parametrize("leaf", ["voxel_size_m", "depth_max_m"])
def test_temporal_cumulative_geometry_identity_leaf_rejects_independent_mutation(
    tmp_path: Path, leaf: str
) -> None:
    config_path = _two_frame_config(tmp_path / "input")
    config = json.loads(config_path.read_text())
    config["temporal_readout"]["geometry"][leaf] += 0.000001
    config["algorithm_hash"] = production_runner.algorithm_hash(config)
    dependencies, _ = _two_frame_dependencies(config)

    with pytest.raises(ValueError, match=f"temporal {leaf} must match cumulative geometry"):
        production_runner.run(
            _write(tmp_path / "mutated.json", config),
            tmp_path / "output",
            dependencies=dependencies,
        )


def test_rejects_invalid_execution_profile_component_combination(tmp_path: Path) -> None:
    config_path = _two_frame_config(tmp_path / "input")
    config = json.loads(config_path.read_text())
    config["temporal_readout"]["execution_profile"] = "a3"
    config["algorithm_hash"] = production_runner.algorithm_hash(config)
    dependencies, _ = _two_frame_dependencies(config)

    with pytest.raises(ValueError, match="components do not match execution_profile"):
        production_runner.run(
            _write(tmp_path / "invalid.json", config),
            tmp_path / "output",
            dependencies=dependencies,
        )


def test_temporal_fixed_rejection_leaf_rejects_mutation(tmp_path: Path) -> None:
    config_path = _two_frame_config(tmp_path / "input")
    config = json.loads(config_path.read_text())
    original_algorithm = config["algorithm_hash"]
    original_non_temporal = non_temporal_config_sha256(config)
    config["temporal_readout"]["motion"]["require_explicit_rejection"] = False
    config["algorithm_hash"] = production_runner.algorithm_hash(config)
    dependencies, _ = _two_frame_dependencies(config)

    assert config["algorithm_hash"] != original_algorithm
    assert non_temporal_config_sha256(config) == original_non_temporal
    with pytest.raises(ValueError, match="require_explicit_rejection must be true"):
        production_runner.run(
            _write(tmp_path / "invalid.json", config),
            tmp_path / "output",
            dependencies=dependencies,
        )


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


def test_publication_revalidates_all_exact_roots_after_link_and_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path / "evidence")
    output = tmp_path / "publication" / "result.json"
    evidence = json.loads(paths["t1_exact_evidence"].read_text())
    ninth_root = Path(
        evidence["deterministic_evidence"]["cumulative_exact"]["executions"][-1][
            "output_root"
        ]
    )
    source = ninth_root / "normalized_run_config.json"
    original_data = source.read_bytes()
    original_link = package_module.os.link
    changed = False

    def mutate_after_link(*args: object, **kwargs: object) -> None:
        nonlocal changed
        original_link(*args, **kwargs)
        if not changed:
            changed = True
            source.write_bytes(original_data + b"changed-after-link")

    monkeypatch.setattr(package_module.os, "link", mutate_after_link)

    with pytest.raises(ValueError, match="changed before publication|exact execution"):
        _package(paths, output)

    assert not output.exists()
    assert not list(output.parent.glob(".result.json.*"))
    source.write_bytes(original_data)
    monkeypatch.setattr(package_module.os, "link", original_link)
    assert _package(paths, output)["status"] == "PASS"


def test_publication_rolls_back_parent_fsync_failure_and_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path / "evidence")
    output = tmp_path / "publication" / "result.json"
    original_fsync = package_module.os.fsync
    failed = False

    def fail_first_parent_fsync(descriptor: int) -> None:
        nonlocal failed
        mode = os.fstat(descriptor).st_mode
        if stat.S_ISDIR(mode) and not failed:
            failed = True
            raise OSError("injected parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(package_module.os, "fsync", fail_first_parent_fsync)

    with pytest.raises(OSError, match="injected parent fsync failure"):
        _package(paths, output)

    assert not output.exists()
    assert not list(output.parent.glob(".result.json.*"))
    monkeypatch.setattr(package_module.os, "fsync", original_fsync)
    assert _package(paths, output)["status"] == "PASS"


def test_publication_name_swap_is_uncertain_and_does_not_delete_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path / "evidence")
    output = tmp_path / "publication" / "result.json"
    replacement = b"replacement-owned-by-another-writer\n"
    original_link = package_module.os.link

    def swap_after_link(
        source: object,
        destination: object,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        os.unlink(destination, dir_fd=dst_dir_fd)
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
            dir_fd=dst_dir_fd,
        )
        try:
            os.write(descriptor, replacement)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(package_module.os, "link", swap_after_link)

    with pytest.raises(RuntimeError, match="publication state is uncertain") as caught:
        _package(paths, output)

    assert type(caught.value) is package_module.PublicationUncertainError
    assert output.read_bytes() == replacement
    assert not list(output.parent.glob(".result.json.*"))
