from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.evaluation.run_ovi_rescene_two_visit_matrix import (
    VariantExecutionArtifacts,
    execute_required_matrix,
    load_and_validate_matrix,
)
from scripts.evaluation.summarize_ovi_rescene_two_visit_matrix import (
    SummaryError,
    summarize_matrix,
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


def _execute_fixture(tmp_path: Path) -> Path:
    matrix = load_and_validate_matrix(MATRIX_PATH)

    def execute(row, run_id: str, root: Path) -> VariantExecutionArtifacts:
        variant_id = row["id"]
        stdout = root / "stdout.log"
        stderr = root / "stderr.log"
        output = root / "current-map-manifest.json"
        metrics = root / "metrics.json"
        stdout.write_text("pass\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        output.write_text("{}\n", encoding="utf-8")
        groups = {
            group: {name: 0.0 for name in names}
            for group, names in matrix["metric_contract"]["groups"].items()
        }
        ghost = {"B0": 0.8, "B1": 0.7, "B2": 0.1, "B3": 0.11, "B4": 0.12}[
            variant_id
        ]
        recall = {"B0": 0.4, "B1": 0.8, "B2": 0.2, "B3": 0.27, "B4": 0.3}[
            variant_id
        ]
        groups["current_state"]["ghost"] = ghost
        groups["geometry"]["t1_unobserved_region_recall"] = recall
        metrics.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "variant_id": variant_id,
                    "run_id": run_id,
                    "metric_groups": groups,
                    "unavailable": {},
                    "method_input_bindings": {"input": {"sha256": "a" * 64}},
                    "evaluation_only_bindings": {
                        "target": {"sha256": "b" * 64}
                    },
                }
            ),
            encoding="utf-8",
        )
        return VariantExecutionArtifacts(
            command_line=("fixture", variant_id),
            stdout_log=stdout,
            stderr_log=stderr,
            output_artifact=output,
            metric_receipt=metrics,
        )

    return execute_required_matrix(
        matrix_path=MATRIX_PATH,
        scene="apartment",
        output_root=tmp_path / "matrix-run",
        source_commit=_head(),
        executor=execute,
    )


def test_summarizes_bound_metrics_and_preregistered_gates(tmp_path: Path) -> None:
    matrix_summary = _execute_fixture(tmp_path)

    output = summarize_matrix(
        matrix_path=MATRIX_PATH,
        matrix_summary_path=matrix_summary,
        output_path=tmp_path / "apartment-summary.json",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "DETERMINISTIC_INFRASTRUCTURE_COMPLETE"
    assert payload["rescene_verdict"] == "RESCENE_BLOCKED_EXTERNAL_ASSET"
    assert payload["gates"]["b2_practical_ghost"]["passed"] is True
    assert payload["gates"]["b3_ghost_relative_to_b2"]["passed"] is True
    assert payload["gates"]["b4_ghost_relative_to_b2"]["passed"] is True
    assert payload["gates"]["b3_unobserved_recall_gain_over_b2"]["passed"] is True
    assert payload["gates"]["b4_unobserved_recall_gain_over_b2"]["passed"] is True
    assert [row["status"] for row in payload["variants"]] == [
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
        "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
    ]


def test_rejects_metric_changed_after_matrix_publication(tmp_path: Path) -> None:
    matrix_summary = _execute_fixture(tmp_path)
    metrics = matrix_summary.parent / "B2" / "metrics.json"
    metrics.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SummaryError, match="artifact binding mismatch"):
        summarize_matrix(
            matrix_path=MATRIX_PATH,
            matrix_summary_path=matrix_summary,
            output_path=tmp_path / "apartment-summary.json",
        )
