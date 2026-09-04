from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.evaluation.run_ovi_rescene_two_visit_matrix import (
    BLOCKED_STATUS,
    MatrixError,
    VariantExecutionArtifacts,
    execute_required_matrix,
    load_and_validate_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "configs/evaluation/ovi_rescene_two_visit_matrix.json"


def _head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _successful_artifacts(
    matrix: dict[str, object], row, run_id: str, run_root: Path
) -> VariantExecutionArtifacts:
    variant_id = row["id"]
    stdout = run_root / Path(row["artifacts"]["stdout_log"]).name
    stderr = run_root / Path(row["artifacts"]["stderr_log"]).name
    output = run_root / Path(row["artifacts"]["output_artifact"]).name
    metrics = run_root / Path(row["artifacts"]["metric_receipt"]).name
    stdout.write_text(f"{variant_id} pass\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    output.write_bytes(variant_id.encode("ascii"))
    groups = {
        group: {name: 0.0 for name in names}
        for group, names in matrix["metric_contract"]["groups"].items()
    }
    metrics.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "variant_id": variant_id,
                "run_id": run_id,
                "metric_groups": groups,
                "unavailable": {},
                "method_input_bindings": {"fixture": {"sha256": "a" * 64}},
                "evaluation_only_bindings": {"target": {"sha256": "b" * 64}},
            }
        ),
        encoding="utf-8",
    )
    return VariantExecutionArtifacts(
        command_line=("fake-evaluator", "--variant", variant_id),
        stdout_log=stdout,
        stderr_log=stderr,
        output_artifact=output,
        metric_receipt=metrics,
    )


def test_checked_in_matrix_freezes_exact_b0_b6_contract() -> None:
    matrix = load_and_validate_matrix(MATRIX_PATH)
    rows = matrix["rows"]

    assert [row["id"] for row in rows] == [f"B{index}" for index in range(7)]
    assert [row["name"] for row in rows] == [
        "OVI_T0_ONLY",
        "OVI_UNION",
        "OVI_T1_ONLY",
        "OVI_VISIBILITY_COMPOSE",
        "OVI_GEOMETRIC_PAIRING_VISIBILITY",
        "OVI_RESCENE_VISIBILITY",
        "OVI_RESCENE_NO_VISIBILITY",
    ]
    assert [(row["uses_ovi_t0"], row["uses_ovi_t1"]) for row in rows] == [
        (True, False),
        (True, True),
        (False, True),
        (True, True),
        (True, True),
        (True, True),
        (True, True),
    ]
    assert [row["pair_reasoner"] for row in rows] == [
        "none",
        "none",
        "none",
        "none",
        "geometric_semantic",
        "rescene_concerto",
        "rescene_concerto",
    ]
    assert [row["signed_visibility"] for row in rows] == [
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    assert all(
        set(row["final_geometry_sources"]) <= {"ovi_t0", "ovi_t1"}
        for row in rows
    )
    assert all("command" in row["artifacts"] for row in rows)
    assert all("metric_receipt" in row["artifacts"] for row in rows)
    assert all(row["availability"] == "REQUIRED" for row in rows[:5])
    assert all(row["availability"] == BLOCKED_STATUS for row in rows[5:])
    assert matrix["metric_contract"]["distance_threshold_m"] == 0.05
    assert matrix["office_policy"]["status"] == "OFFICE_NOT_RUN_HELD_OUT"
    assert matrix["method_config"]["ovi_semantics"] == {
        "classifier": "siglip_l_16_384_canonical_relative",
        "vocabulary_path": (
            "configs/evaluation/vocabularies/tesse_cd_apartment.json"
        ),
        "canonical_prompts": ["object", "things", "stuff", "texture"],
        "maximum_text_length": 64,
        "minimum_observation_count": 2,
    }
    assert matrix["method_config"]["signed_visibility"] == {
        "voxel_size_m": 0.05,
        "depth_tolerance_m": 0.1,
        "depth_max_m": 10.0,
        "minimum_absent_fraction": 0.8,
        "minimum_absent_observations": 6,
        "minimum_distinct_viewpoints": 3,
        "minimum_viewpoint_baseline_m": 0.25,
    }
    assert matrix["preregistration_amendment"]["status"] == (
        "FROZEN_BEFORE_FIRST_METHOD_SCORE"
    )
    assert matrix["preregistration_amendment"]["method_results_inspected"] is False
    assert {
        "deterministic_executor",
        "native_ovi_runner",
        "snapshot_metrics",
        "visibility_execution",
        "visit_loader",
    } <= set(matrix["source_bindings"])


def test_matrix_rejects_changed_visibility_or_geometry_authority(tmp_path: Path) -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    matrix["rows"][3]["signed_visibility"] = False
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix), encoding="utf-8")

    with pytest.raises(MatrixError, match="B3 contract"):
        load_and_validate_matrix(path, verify_source_bindings=False)

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    matrix["rows"][5]["final_geometry_sources"] = ["rescene_tokens"]
    path.write_text(json.dumps(matrix), encoding="utf-8")
    with pytest.raises(MatrixError, match="B5 contract"):
        load_and_validate_matrix(path, verify_source_bindings=False)


def test_matrix_rejects_changed_method_threshold_or_amendment(tmp_path: Path) -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    matrix["method_config"]["signed_visibility"]["minimum_absent_observations"] = 5
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix), encoding="utf-8")

    with pytest.raises(MatrixError, match="method configuration"):
        load_and_validate_matrix(path, verify_source_bindings=False)

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    matrix["preregistration_amendment"]["method_results_inspected"] = True
    path.write_text(json.dumps(matrix), encoding="utf-8")
    with pytest.raises(MatrixError, match="preregistration amendment"):
        load_and_validate_matrix(path, verify_source_bindings=False)


def test_matrix_rejects_changed_source_binding(tmp_path: Path) -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    matrix["source_bindings"]["composer"]["sha256"] = "0" * 64
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix), encoding="utf-8")

    with pytest.raises(MatrixError, match="source binding mismatch"):
        load_and_validate_matrix(path)


def test_orchestrator_runs_b0_b4_and_emits_explicit_b5_b6_blockers(
    tmp_path: Path,
) -> None:
    matrix = load_and_validate_matrix(MATRIX_PATH)
    calls: list[str] = []

    def execute(row, run_id: str, run_root: Path) -> VariantExecutionArtifacts:
        variant_id = row["id"]
        calls.append(variant_id)
        return _successful_artifacts(matrix, row, run_id, run_root)

    summary_path = execute_required_matrix(
        matrix_path=MATRIX_PATH,
        scene="apartment",
        output_root=tmp_path / "matrix-run",
        source_commit=_head(),
        executor=execute,
    )

    assert calls == ["B0", "B1", "B2", "B3", "B4"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "MATRIX_EXECUTION_COMPLETE_WITH_BLOCKED_VARIANTS"
    assert [row["status"] for row in summary["variants"][:5]] == ["PASS"] * 5
    assert [row["status"] for row in summary["variants"][5:]] == [
        BLOCKED_STATUS,
        BLOCKED_STATUS,
    ]
    for variant_id in ("B5", "B6"):
        receipt = json.loads(
            (tmp_path / "matrix-run" / variant_id / "blocked-receipt.json").read_text()
        )
        assert receipt["status"] == BLOCKED_STATUS
        assert receipt["ranking_eligible"] is False
        assert receipt["metrics"] is None


def test_orchestrator_never_runs_office_before_release(tmp_path: Path) -> None:
    with pytest.raises(MatrixError, match="OFFICE_NOT_RUN_HELD_OUT"):
        execute_required_matrix(
            matrix_path=MATRIX_PATH,
            scene="office",
            output_root=tmp_path / "office",
            source_commit="a" * 40,
            executor=lambda *_: pytest.fail("Office executor must not run"),
        )


def test_orchestrator_rejects_unexplained_na_without_publishing(tmp_path: Path) -> None:
    matrix = load_and_validate_matrix(MATRIX_PATH)
    output_root = tmp_path / "matrix-run"

    def execute(row, run_id: str, run_root: Path) -> VariantExecutionArtifacts:
        variant_id = row["id"]
        stdout = run_root / Path(row["artifacts"]["stdout_log"]).name
        stderr = run_root / Path(row["artifacts"]["stderr_log"]).name
        output = run_root / Path(row["artifacts"]["output_artifact"]).name
        metrics = run_root / Path(row["artifacts"]["metric_receipt"]).name
        stdout.write_text("pass\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        output.write_bytes(b"output")
        groups = {
            group: {name: 0.0 for name in names}
            for group, names in matrix["metric_contract"]["groups"].items()
        }
        groups["identity"]["moved_association_accuracy"] = None
        metrics.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "variant_id": variant_id,
                    "run_id": run_id,
                    "metric_groups": groups,
                    "unavailable": {},
                    "method_input_bindings": {"fixture": {"sha256": "a" * 64}},
                    "evaluation_only_bindings": {"target": {"sha256": "b" * 64}},
                }
            ),
            encoding="utf-8",
        )
        return VariantExecutionArtifacts(
            command_line=("fake",),
            stdout_log=stdout,
            stderr_log=stderr,
            output_artifact=output,
            metric_receipt=metrics,
        )

    with pytest.raises(MatrixError, match="lack exact reasons"):
        execute_required_matrix(
            matrix_path=MATRIX_PATH,
            scene="apartment",
            output_root=output_root,
            source_commit=_head(),
            executor=execute,
        )
    assert not output_root.exists()


def test_orchestrator_rejects_artifacts_at_unfrozen_paths(tmp_path: Path) -> None:
    matrix = load_and_validate_matrix(MATRIX_PATH)
    output_root = tmp_path / "matrix-run"

    def execute(row, run_id: str, run_root: Path) -> VariantExecutionArtifacts:
        variant_id = row["id"]
        stdout = run_root / "different-stdout.log"
        stderr = run_root / Path(row["artifacts"]["stderr_log"]).name
        output = run_root / Path(row["artifacts"]["output_artifact"]).name
        metrics = run_root / Path(row["artifacts"]["metric_receipt"]).name
        stdout.write_text("pass\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        output.write_bytes(b"output")
        groups = {
            group: {name: 0.0 for name in names}
            for group, names in matrix["metric_contract"]["groups"].items()
        }
        metrics.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "variant_id": variant_id,
                    "run_id": run_id,
                    "metric_groups": groups,
                    "unavailable": {},
                    "method_input_bindings": {"fixture": {"sha256": "a" * 64}},
                    "evaluation_only_bindings": {"target": {"sha256": "b" * 64}},
                }
            ),
            encoding="utf-8",
        )
        return VariantExecutionArtifacts(
            command_line=("fake",),
            stdout_log=stdout,
            stderr_log=stderr,
            output_artifact=output,
            metric_receipt=metrics,
        )

    with pytest.raises(MatrixError, match="frozen artifact path"):
        execute_required_matrix(
            matrix_path=MATRIX_PATH,
            scene="apartment",
            output_root=output_root,
            source_commit=_head(),
            executor=execute,
        )
    assert not output_root.exists()


def test_orchestrator_reopens_all_artifacts_before_publish(tmp_path: Path) -> None:
    matrix = load_and_validate_matrix(MATRIX_PATH)
    first_output: Path | None = None
    output_root = tmp_path / "matrix-run"

    def execute(row, run_id: str, run_root: Path) -> VariantExecutionArtifacts:
        nonlocal first_output
        if row["id"] == "B1":
            assert first_output is not None
            first_output.write_bytes(b"changed-after-recording")
        artifacts = _successful_artifacts(matrix, row, run_id, run_root)
        if row["id"] == "B0":
            first_output = artifacts.output_artifact
        return artifacts

    with pytest.raises(MatrixError, match="changed before matrix publish"):
        execute_required_matrix(
            matrix_path=MATRIX_PATH,
            scene="apartment",
            output_root=output_root,
            source_commit=_head(),
            executor=execute,
        )
    assert not output_root.exists()
