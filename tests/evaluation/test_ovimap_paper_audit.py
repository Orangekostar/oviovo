from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path

import pytest

from src.evaluation.baselines.ovimap_paper_audit import (
    PAPER_PROTOCOL,
    audit_paper_parity,
    summarize_feature_file,
)


def _paper_result() -> dict:
    return {
        "status": "VERIFIED",
        "method": {"key": "OVIMAP"},
        "protocol": dict(PAPER_PROTOCOL),
        "metrics": {
            "replica_8_compat": {
                "scene_ids": list(PAPER_PROTOCOL["scene_ids"]),
                "scene_count": len(PAPER_PROTOCOL["scene_ids"]),
            }
        },
    }


def _scene_summaries() -> dict[str, dict]:
    return {
        scene_id: {
            "feature_instance_count": 2,
            "eligible_feature_instance_count": 1,
            "query_count": 3,
            "average_queries_per_feature_instance": 1.5,
            "full_frame_bbox_count": 0,
            "full_frame_bbox_ratio": 0.0,
        }
        for scene_id in PAPER_PROTOCOL["scene_ids"]
    }


def test_paper_parity_passes_only_for_exact_protocol_and_replica8() -> None:
    audit = audit_paper_parity(_paper_result(), _scene_summaries())

    assert audit["status"] == "PASS"
    assert audit["protocol_name"] == "ovimap_cvpr2026_replica"
    assert audit["failures"] == []
    assert audit["aggregate_artifacts"]["feature_instance_count"] == 16
    assert audit["aggregate_artifacts"]["eligible_feature_instance_count"] == 8


def test_paper_parity_rejects_current_mixed_protocol() -> None:
    result = _paper_result()
    result["protocol"] = {
        **PAPER_PROTOCOL,
        "semantic_vocabulary": "Replica-41",
        "instance_metrics": "neutral_equal_score_ap25_ap50",
    }
    summaries = _scene_summaries()
    summaries.pop("room2")

    audit = audit_paper_parity(result, summaries)

    assert audit["status"] == "FAIL"
    assert any("semantic_vocabulary" in failure for failure in audit["failures"])
    assert any("instance_metrics" in failure for failure in audit["failures"])
    assert any("scene feature summaries" in failure for failure in audit["failures"])


def test_feature_summary_records_semantic_coverage_and_full_frame_boxes(tmp_path: Path) -> None:
    feature_file = tmp_path / "features.pkl"
    with feature_file.open("wb") as handle:
        pickle.dump(
            {
                1: {
                    "frame_id": [0],
                    "box_2d": [(0, 0, 1199, 679)],
                },
                2: {
                    "frame_id": [10, 20],
                    "box_2d": [(1, 2, 20, 30), (0, 0, 1199, 679)],
                },
            },
            handle,
        )

    summary = summarize_feature_file(feature_file)

    assert summary["feature_instance_count"] == 2
    assert summary["eligible_feature_instance_count"] == 1
    assert summary["query_count"] == 3
    assert summary["average_queries_per_feature_instance"] == pytest.approx(1.5)
    assert summary["full_frame_bbox_count"] == 2
    assert summary["full_frame_bbox_ratio"] == pytest.approx(2.0 / 3.0)


def test_audit_cli_hashes_inputs_and_writes_failure_report(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result = _paper_result()
    result["protocol"]["semantic_vocabulary"] = "Replica-41"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    arguments = []
    for scene_id in PAPER_PROTOCOL["scene_ids"]:
        feature_path = tmp_path / f"{scene_id}.pkl"
        with feature_path.open("wb") as handle:
            pickle.dump(
                {1: {"frame_id": [0, 10], "box_2d": [(0, 0, 10, 10), (0, 0, 10, 10)]}},
                handle,
            )
        arguments.extend(("--scene-feature", f"{scene_id}={feature_path}"))
    output = tmp_path / "audit.json"

    completed = subprocess.run(
        [
            "/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python",
            "scripts/evaluation/audit_ovimap_paper_parity.py",
            "--result",
            str(result_path),
            *arguments,
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["result_source"]["sha256"]
    assert set(payload["feature_sources"]) == set(PAPER_PROTOCOL["scene_ids"])
