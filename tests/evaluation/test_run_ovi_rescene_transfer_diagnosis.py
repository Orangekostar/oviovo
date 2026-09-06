from __future__ import annotations

import pytest

from scripts.evaluation.run_ovi_rescene_transfer_diagnosis import (
    TransferDiagnosisError,
    build_experiment_rows,
    build_per_pair_rows,
)


def _domains() -> tuple[dict[str, object], ...]:
    return (
        {
            "domain_id": "D0_NATIVE_PROCESSED",
            "execution_mode": "REUSED_NATIVE_FORWARD",
            "input_point_count": 100,
            "sensor_supported_point_count": 100,
            "raw_query_count": 10,
            "raw_nonempty_query_count": 8,
            "confident_query_count": 4,
            "t0_candidate_count": 3,
            "t1_candidate_count": 4,
            "persistent_gt_count": 5,
            "represented_persistent_gt_count": 4,
            "forward_runtime_s": 1.0,
            "peak_memory_bytes": 1000,
            "input_sha256": "a" * 64,
            "methods": {
                "R_legacy": {
                    "prediction_count": 3,
                    "true_positive_count": 2,
                    "false_positive_count": 1,
                    "precision": 2 / 3,
                    "recall": 0.4,
                    "postprocess_runtime_s": 0.1,
                }
            },
        },
        {
            "domain_id": "D1_NATIVE_SENSOR_SUPPORT",
            "execution_mode": "CACHED_D0_SENSOR_SUPPORT_PROJECTION",
            "input_point_count": 100,
            "sensor_supported_point_count": 60,
            "raw_query_count": 10,
            "raw_nonempty_query_count": 7,
            "confident_query_count": 3,
            "t0_candidate_count": 2,
            "t1_candidate_count": 3,
            "persistent_gt_count": 5,
            "represented_persistent_gt_count": 3,
            "forward_runtime_s": None,
            "peak_memory_bytes": None,
            "input_sha256": "b" * 64,
            "methods": {
                "R_legacy": {
                    "prediction_count": 2,
                    "true_positive_count": 1,
                    "false_positive_count": 1,
                    "precision": 0.5,
                    "recall": 0.2,
                    "postprocess_runtime_s": 0.2,
                }
            },
        },
        {
            "domain_id": "D2_OVI_RECONSTRUCTION",
            "execution_mode": "REUSED_REAL_OVI_FORWARD",
            "input_point_count": 80,
            "sensor_supported_point_count": 40,
            "raw_query_count": 10,
            "raw_nonempty_query_count": 6,
            "confident_query_count": 2,
            "t0_candidate_count": 2,
            "t1_candidate_count": 2,
            "persistent_gt_count": 5,
            "represented_persistent_gt_count": 2,
            "forward_runtime_s": 0.5,
            "peak_memory_bytes": 800,
            "input_sha256": "c" * 64,
            "methods": {
                "R_obj": {
                    "prediction_count": 2,
                    "true_positive_count": 1,
                    "false_positive_count": 1,
                    "precision": 0.5,
                    "recall": 0.2,
                    "postprocess_runtime_s": None,
                }
            },
        },
    )


def test_builds_null_preserving_identical_uuid_domain_rows() -> None:
    rows = build_per_pair_rows(
        pair_id="pair-a",
        evaluated_commit="d" * 40,
        config_sha256="e" * 64,
        checkpoint_sha256="f" * 64,
        iou_threshold=0.25,
        domains=_domains(),
    )

    assert [row["domain_id"] for row in rows] == [
        "D0_NATIVE_PROCESSED",
        "D1_NATIVE_SENSOR_SUPPORT",
        "D2_OVI_RECONSTRUCTION",
    ]
    assert all(row["pair_id"] == "pair-a" for row in rows)
    assert rows[0]["sensor_support_fraction"] == 1.0
    assert rows[1]["sensor_support_fraction"] == 0.6
    assert rows[1]["forward_runtime_s"] is None
    assert rows[2]["proposal_representation_coverage"] == 0.4
    assert rows[0]["association_f1"] == pytest.approx(0.5)


def test_experiment_ledger_records_that_training_was_not_authorized() -> None:
    rows = build_experiment_rows(
        pair_id="pair-a",
        evaluated_commit="d" * 40,
        config_sha256="e" * 64,
        decision={
            "action": "NO_ADAPTATION",
            "selected_rule": "d1_sensor_support_degradation",
        },
        domain_rows=build_per_pair_rows(
            pair_id="pair-a",
            evaluated_commit="d" * 40,
            config_sha256="e" * 64,
            checkpoint_sha256="f" * 64,
            iou_threshold=0.25,
            domains=_domains(),
        ),
        output_root="/tmp/evidence",
    )

    assert rows[-1]["run_id"] == "decoder_mask_head_adaptation"
    assert rows[-1]["status"] == "NOT_RUN_GATE_SELECTED_NO_ADAPTATION"
    assert rows[-1]["runtime_s"] is None
    assert rows[-1]["peak_memory_bytes"] is None


def test_per_pair_builder_rejects_missing_domain() -> None:
    with pytest.raises(TransferDiagnosisError, match="D0/D1/D2"):
        build_per_pair_rows(
            pair_id="pair-a",
            evaluated_commit="d" * 40,
            config_sha256="e" * 64,
            checkpoint_sha256="f" * 64,
            iou_threshold=0.25,
            domains=_domains()[:2],
        )
