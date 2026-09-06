from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.rscan_association_metrics import evaluate_pair_relations
from src.evaluation.rscan_gt_instances import (
    GroundTruthInstance,
    GroundTruthPair,
    IdentityRules,
    voxelize_points,
)
from src.evaluation.rscan_method_views import build_method_pair_view
from src.oviv2.two_visit_contracts import PairRelation


def _processed(*, offset: float = 0.0) -> np.ndarray:
    return np.asarray(
        [
            [0.00 + offset, 0.0, 0.0, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 4, 9, 101],
            [0.01 + offset, 0.0, 0.0, 0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 4, 9, 101],
            [1.00 + offset, 0.0, 0.0, 0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 8, 7, 202],
        ],
        dtype=np.float32,
    )


def _pair():
    return build_method_pair_view(
        pair_id="scene0001_00-scene0001_01",
        scan_ids=("scan-a", "scan-b"),
        processed_visits=(_processed(), _processed(offset=0.01)),
        source_manifest_sha256="a" * 64,
        domain_id="D0_NATIVE_PROCESSED",
    )


def _identity_rules() -> IdentityRules:
    return IdentityRules.from_official_records(
        changes={
            "rigid": [
                {
                    "instance_reference": 10,
                    "instance_rescan": 10,
                    "symmetry": 0,
                    "transform": np.eye(4).tolist(),
                }
            ],
            "nonrigid": [],
            "removed": [],
        },
        ambiguity=[],
    )


def _ground_truth(*, include_unobserved: bool = False) -> GroundTruthPair:
    pair = _pair()
    visits: list[list[GroundTruthInstance]] = [[], []]
    labels = {"segment:000004": (10, "chair"), "segment:000008": (20, "chair")}
    for visit in pair.visits:
        for candidate_id, (instance_id, label) in labels.items():
            segment_id = int(candidate_id.split(":")[1])
            visits[visit.visit_id].append(
                GroundTruthInstance(
                    instance_id,
                    label,
                    voxelize_points(
                        visit.points_xyz[visit.segment_ids == segment_id],
                        voxel_size_m=0.05,
                    ),
                )
            )
        if include_unobserved:
            visits[visit.visit_id].append(
                GroundTruthInstance(
                    30,
                    "table",
                    frozenset({(200 + visit.visit_id, 200, 200)}),
                )
            )
    return GroundTruthPair(
        pair_id=pair.pair_id,
        voxel_size_m=0.05,
        visits=(tuple(visits[0]), tuple(visits[1])),
        identity_rules=_identity_rules(),
    )


def _relation(query_id: str, t0: str, t1: str, confidence: float = 0.9) -> PairRelation:
    return PairRelation(
        temporal_query_id=query_id,
        t0_entity_ids=(t0,),
        t1_entity_ids=(t1,),
        state="persistent_static",
        query_confidence=confidence,
        evidence={"query_score": confidence},
        identity_source="geometric_baseline",
    )


def test_correct_relations_score_identity_and_geometry_separately() -> None:
    result = evaluate_pair_relations(
        _pair(),
        (
            _relation("q0", "segment:000004", "segment:000004"),
            _relation("q1", "segment:000008", "segment:000008"),
        ),
        _ground_truth(),
    )

    primary = result["thresholds"]["iou_0_50"]
    assert primary["paired_precision"] == pytest.approx(1.0)
    assert primary["end_to_end_persistence_recall"] == pytest.approx(1.0)
    assert primary["conditional_association_recall"] == pytest.approx(1.0)
    assert primary["rigid_recall"] == pytest.approx(1.0)
    assert primary["geometry"]["t0"]["matched_count"] == 2
    assert primary["geometry"]["t1"]["matched_count"] == 2
    assert result["reactivation_recall"] is None


def test_same_class_cross_identity_edges_are_false_reidentifications() -> None:
    result = evaluate_pair_relations(
        _pair(),
        (
            _relation("q0", "segment:000004", "segment:000008"),
            _relation("q1", "segment:000008", "segment:000004"),
        ),
        _ground_truth(),
    )

    primary = result["thresholds"]["iou_0_50"]
    assert primary["paired_precision"] == 0.0
    assert primary["false_reid_count"] == 2
    assert primary["same_class_mismatch_count"] == 2


def test_relation_outcome_rows_expose_endpoint_assignments_and_error_type() -> None:
    from src.evaluation.rscan_association_metrics import relation_outcome_rows

    rows = relation_outcome_rows(
        _pair(),
        (
            _relation("q0", "segment:000004", "segment:000008"),
            _relation("q1", "segment:000008", "segment:000004"),
        ),
        _ground_truth(),
        iou_threshold=0.50,
    )

    assert [row["query_id"] for row in rows] == ["q0", "q1"]
    assert [row["outcome"] for row in rows] == ["false_reid", "false_reid"]
    assert rows[0]["t0_assigned_gt_id"] == 10
    assert rows[0]["t1_assigned_gt_id"] == 20
    assert rows[0]["t0_best_iou"] == pytest.approx(1.0)
    assert rows[0]["t1_best_iou"] == pytest.approx(1.0)
    assert rows[0]["same_class_mismatch"] is True
    assert rows[0]["ambiguity_aware_correct"] is False


def test_duplicate_geometry_keeps_one_true_positive_and_counts_the_rest() -> None:
    repeated = _relation("q1", "segment:000004", "segment:000004", 0.8)
    result = evaluate_pair_relations(
        _pair(),
        (_relation("q0", "segment:000004", "segment:000004"), repeated),
        _ground_truth(),
    )

    primary = result["thresholds"]["iou_0_50"]
    assert primary["true_positive_edges"] == 1
    assert primary["false_positive_edges"] == 1
    assert primary["paired_precision"] == pytest.approx(0.5)
    assert primary["geometry"]["duplicate_prediction_count"] == 2


def test_conditional_recall_excludes_gt_without_method_input_support() -> None:
    result = evaluate_pair_relations(
        _pair(),
        (
            _relation("q0", "segment:000004", "segment:000004"),
            _relation("q1", "segment:000008", "segment:000008"),
        ),
        _ground_truth(include_unobserved=True),
    )

    primary = result["thresholds"]["iou_0_50"]
    assert primary["ground_truth_persistent_edges"] == 3
    assert primary["conditional_ground_truth_edges"] == 2
    assert primary["end_to_end_persistence_recall"] == pytest.approx(2 / 3)
    assert primary["conditional_association_recall"] == pytest.approx(1.0)
