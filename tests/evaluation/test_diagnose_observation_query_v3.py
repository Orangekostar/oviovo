from __future__ import annotations

import numpy as np

from scripts.evaluation.diagnose_observation_query_v3 import (
    _collect_intervention_rows,
    _file_record,
    _write_csv,
    diagnose_visit_stages,
)
from src.evaluation.rscan_gt_instances import GroundTruthInstance, voxelize_points


def _target(instance_id: int, points: np.ndarray) -> GroundTruthInstance:
    return GroundTruthInstance(
        instance_id=instance_id,
        semantic_label=f"object-{instance_id}",
        voxels=voxelize_points(points, voxel_size_m=0.05),
    )


def test_diagnosis_separates_filter_competition_and_residual_stages() -> None:
    points = np.asarray(
        [
            [0.001, 0.0, 0.0],
            [0.051, 0.0, 0.0],
            [0.101, 0.0, 0.0],
            [1.001, 0.0, 0.0],
            [1.051, 0.0, 0.0],
            [2.001, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    targets = (
        _target(1, points[:3]),
        _target(2, points[3:5]),
        _target(3, points[5:]),
    )
    masks = np.asarray(
        [
            [4.0, -4.0, 2.0],
            [4.0, -4.0, 5.0],
            [4.0, -4.0, -4.0],
            [-4.0, 4.0, -4.0],
            [-4.0, 4.0, -4.0],
        ],
        dtype=np.float32,
    )
    logits = np.asarray(
        [
            [4.0, 0.0],
            [0.0, 4.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )

    stage_rows, object_rows = diagnose_visit_stages(
        visit_id=0,
        points_xyz=points,
        entity_owner_indices=np.asarray([0, 0, 0, 1, 1, 2]),
        dense_to_model_indices=np.asarray([0, 1, 2, 3, 4, -1]),
        ground_truth=targets,
        pred_masks_mq=masks,
        pred_logits_qc=logits,
        minimum_query_score=0.3,
        voxel_size_m=0.05,
        evaluation_domain="FULL_GT_V2",
    )

    by_stage = {row["stage"]: row for row in stage_rows}
    assert tuple(by_stage) == (
        "R0_RAW_ALL",
        "R1_ELIGIBLE",
        "R2_EXCLUSIVE_QUERY",
        "R3_FINAL_MAP",
    )
    assert by_stage["R0_RAW_ALL"]["prediction_count"] == 3
    assert by_stage["R1_ELIGIBLE"]["prediction_count"] == 2
    assert by_stage["R2_EXCLUSIVE_QUERY"]["tp50"] < by_stage["R3_FINAL_MAP"]["tp50"]

    by_target = {row["gt_instance_id"]: row for row in object_rows}
    filtered = by_target[2]
    assert filtered["raw_best_query_id"] == 1
    assert filtered["raw_query_survives_threshold"] is False
    assert filtered["eligible_best_query_id"] is None
    assert filtered["residual_involvement"] is True
    assert "filtered" in filtered["classification"].split("|")
    assert by_target[3]["common_support_fraction"] == 0.0
    assert "support_limited" in by_target[3]["classification"].split("|")


def test_diagnosis_common_domain_excludes_unsupported_target() -> None:
    points = np.asarray(
        [[0.001, 0.0, 0.0], [1.001, 0.0, 0.0]], dtype=np.float32
    )
    targets = (_target(1, points[:1]), _target(2, points[1:]))
    masks = np.asarray([[4.0]], dtype=np.float32)
    logits = np.asarray([[4.0, 0.0]], dtype=np.float32)

    stage_rows, object_rows = diagnose_visit_stages(
        visit_id=1,
        points_xyz=points,
        entity_owner_indices=np.asarray([0, 1]),
        dense_to_model_indices=np.asarray([0, -1]),
        ground_truth=targets,
        pred_masks_mq=masks,
        pred_logits_qc=logits,
        minimum_query_score=0.3,
        voxel_size_m=0.05,
        evaluation_domain="COMMON_INPUT_SUPPORT_V2",
    )

    assert all(row["target_count"] == 1 for row in stage_rows)
    assert [row["gt_instance_id"] for row in object_rows] == [1, 2]
    assert object_rows[1]["included_in_evaluation_domain"] is False


def test_published_file_record_uses_relocated_path(tmp_path) -> None:
    artifact = tmp_path / "staging" / "readout_stages.csv"
    artifact.parent.mkdir()
    artifact.write_text("stage\nR0\n", encoding="utf-8")

    record = _file_record(artifact, recorded_path="readout_stages.csv")

    assert record["path"] == "readout_stages.csv"
    assert record["byte_count"] == len(b"stage\nR0\n")


def test_diagnostic_csv_uses_repository_lf_line_endings(tmp_path) -> None:
    output = tmp_path / "diagnostics.csv"

    _write_csv(output, ("left", "right"), ({"left": 1, "right": 2},))

    assert output.read_bytes() == b"left,right\n1,2\n"


def test_intervention_summary_compares_same_checkpoint_and_reuses_real_gates(
    tmp_path,
) -> None:
    import csv
    import json

    specs = (
        (200, "I_FULL", "full"),
        (200, "I_BETA0", "beta0"),
        (200, "I_ALPHA0_BETA0", "both0"),
    )
    values = {"full": 0.1, "beta0": 0.2, "both0": 0.3}
    fields = (
        "visit_id",
        "evaluation_domain",
        "f1_50",
        "tp50",
        "fp50",
        "fn50",
        "raw_best_iou_mean",
        "raw_ar50",
        "raw_ar25",
    )
    for _updates, intervention, run_id in specs:
        root = tmp_path / run_id
        root.mkdir()
        metadata = None
        if intervention != "I_FULL":
            metadata = {
                "intervention_id": intervention,
                "applied_observation_mode": (
                    "no_feedback" if intervention == "I_BETA0" else "base_tuned"
                ),
                "alpha_by_layer": [0.1],
                "beta_by_layer": [0.2],
                "observation_stats": {"region_count": 2},
            }
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "method_id": "OBS_FULL",
                    "checkpoint_id": "a" * 64,
                    "training_updates": 200,
                    "eval_environment_id": "environment",
                    "eval_pair_id": "pair",
                    "inference_intervention": metadata,
                }
            ),
            encoding="utf-8",
        )
        with (root / "results.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for domain in (
                "FULL_GT_V2",
                "COMMON_INPUT_SUPPORT_V2",
                "RAW_COMMON_INPUT_SUPPORT_V2",
            ):
                writer.writerow(
                    {
                        "visit_id": 1,
                        "evaluation_domain": domain,
                        "f1_50": values[run_id],
                        "tp50": 1,
                        "fp50": 2,
                        "fn50": 3,
                        "raw_best_iou_mean": values[run_id],
                        "raw_ar50": 0.1,
                        "raw_ar25": 0.2,
                    }
                )

    rows = _collect_intervention_rows(tmp_path, specs)

    assert [row["intervention_id"] for row in rows] == [
        "I_FULL",
        "I_BETA0",
        "I_ALPHA0_BETA0",
    ]
    assert rows[1]["common_f1_50_delta_vs_full"] == 0.1
    assert rows[0]["alpha_by_layer_json"] == "[0.1]"
    assert rows[0]["applied_observation_mode"] == "full"
