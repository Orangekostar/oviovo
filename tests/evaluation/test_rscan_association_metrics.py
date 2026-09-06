from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.object_pair_association import ObjectPairPrediction
from src.evaluation.ovi_pair_views import (
    OviObjectEntityView,
    OviObjectPairView,
    OviObjectVisitView,
)
from src.evaluation.rscan_association_metrics import (
    build_fixed_endpoint_bindings,
    evaluate_fixed_object_predictions,
    evaluate_pair_relations,
)
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


def _fixed_metric_visit(visit_id: int) -> OviObjectVisitView:
    points = np.asarray(
        [
            [0.00, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [1.00, 0.0, 0.0],
            [2.00, 0.0, 0.0],
            [9.00, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    entities = tuple(
        OviObjectEntityView(
            visit_id=visit_id,
            entity_id=f"ovimap:{index + 1}",
            source_instance_id=index + 1,
            point_indices=np.asarray([index], dtype=np.int64),
            palette_rgb=(10 + index, 20 + index, 30 + index),
            semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            semantic_label="object",
            semantic_score=1.0,
            observation_frame_ids=(0,),
            observation_boxes_xyxy=((index, 0, index, 0),),
        )
        for index in range(5)
    )
    return OviObjectVisitView(
        visit_id=visit_id,
        scan_id=f"fixed-scan-{visit_id}",
        frame_count=1,
        points_xyz=points,
        palette_rgb_uint8=np.asarray(
            [[10 + i, 20 + i, 30 + i] for i in range(5)], dtype=np.uint8
        ),
        normals_xyz=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (5, 1)),
        normal_valid=np.ones(5, dtype=bool),
        source_vertex_indices=np.arange(5, dtype=np.int64),
        source_frame_ids_by_target=np.asarray([10], dtype=np.int64),
        entity_owner_indices=np.arange(5, dtype=np.int64),
        entities=entities,
        camera_rgb_uint8=np.full((5, 3), 128, dtype=np.uint8),
        appearance_valid=np.ones(5, dtype=bool),
        appearance_frame_ids=np.zeros(5, dtype=np.int64),
        appearance_source_frame_ids=np.full(5, 10, dtype=np.int64),
        appearance_rows=np.zeros(5, dtype=np.int64),
        appearance_columns=np.arange(5, dtype=np.int64),
        appearance_camera_depth_m=np.ones(5, dtype=np.float32),
        appearance_observed_depth_m=np.full(5, 1.01, dtype=np.float32),
        appearance_depth_residual_m=np.full(5, 0.01, dtype=np.float32),
        native_manifest_sha256=str(visit_id + 1) * 64,
        materialized_manifest_sha256=str(visit_id + 3) * 64,
        source_artifact_sha256={
            "instance_color_log": "5" * 64,
            "instance_mesh": "6" * 64,
            "semantic_features": "7" * 64,
        },
        global_alignment_application=(
            "identity_reference" if visit_id == 0 else "rescan_to_reference_once"
        ),
    )


def _fixed_metric_pair() -> OviObjectPairView:
    return OviObjectPairView(
        pair_id="scene0099_00-scene0099_01",
        visits=(_fixed_metric_visit(0), _fixed_metric_visit(1)),
        source_manifest_sha256="a" * 64,
        global_alignment=np.eye(4),
    )


def _fixed_metric_ground_truth(pair: OviObjectPairView) -> GroundTruthPair:
    visits = []
    for visit in pair.visits:
        visits.append(
            (
                GroundTruthInstance(
                    10,
                    "chair",
                    voxelize_points(visit.points_xyz[[0, 1]], voxel_size_m=0.05),
                ),
                GroundTruthInstance(
                    20,
                    "table",
                    voxelize_points(visit.points_xyz[[2]], voxel_size_m=0.05),
                ),
                GroundTruthInstance(
                    30,
                    "cabinet",
                    voxelize_points(visit.points_xyz[[3]], voxel_size_m=0.05),
                ),
                GroundTruthInstance(40, "lamp", frozenset({(400, 400, 400)})),
            )
        )
    rules = IdentityRules.from_official_records(
        changes={
            "rigid": [
                {
                    "instance_reference": 20,
                    "instance_rescan": 20,
                    "symmetry": 0,
                    "transform": np.eye(4).tolist(),
                }
            ],
            "nonrigid": [],
            "removed": [],
        },
        ambiguity=[],
    )
    return GroundTruthPair(
        pair_id=pair.pair_id,
        voxel_size_m=0.05,
        visits=(visits[0], visits[1]),
        identity_rules=rules,
    )


def _fixed_prediction(
    pair: OviObjectPairView,
    prediction_id: str,
    t0: str,
    t1: str,
    score: float,
) -> ObjectPairPrediction:
    return ObjectPairPrediction(
        prediction_id=prediction_id,
        pair_id=pair.pair_id,
        method_id="R_obj",
        t0_entity_id=t0,
        t1_entity_id=t1,
        score=score,
        state="persistent_static",
    )


def test_fixed_endpoint_bindings_preserve_duplicate_candidates_without_forcing_gt() -> (
    None
):
    pair = _fixed_metric_pair()
    ground_truth = _fixed_metric_ground_truth(pair)

    bindings = build_fixed_endpoint_bindings(
        pair, ground_truth, support_domain="supported"
    )
    primary = bindings.for_threshold(0.50)

    assert [value.assigned_gt_instance_id for value in primary[0]] == [
        10,
        10,
        20,
        30,
        None,
    ]
    assert primary[0][4].best_gt_instance_id == 10
    assert primary[0][4].best_iou == 0.0
    assert bindings.content_sha256() == bindings.content_sha256()


def test_fixed_object_metrics_separate_endpoint_reid_duplicate_and_denominators() -> (
    None
):
    pair = _fixed_metric_pair()
    ground_truth = _fixed_metric_ground_truth(pair)
    bindings = build_fixed_endpoint_bindings(
        pair, ground_truth, support_domain="supported"
    )
    predictions = (
        _fixed_prediction(pair, "p0", "ovimap:1", "ovimap:1", 0.95),
        _fixed_prediction(pair, "p1", "ovimap:2", "ovimap:2", 0.90),
        _fixed_prediction(pair, "p2", "ovimap:3", "ovimap:4", 0.85),
        _fixed_prediction(pair, "p3", "ovimap:4", "ovimap:3", 0.80),
        _fixed_prediction(pair, "p4", "ovimap:5", "ovimap:5", 0.75),
    )

    result = evaluate_fixed_object_predictions(
        pair, predictions, ground_truth, bindings, iou_threshold=0.50
    )

    assert result["binding_sha256"] == bindings.content_sha256()
    assert result["paired_prediction_count"] == 5
    assert result["true_positive_count"] == 1
    assert result["false_positive_count"] == 4
    assert result["endpoint_failure_count"] == 1
    assert result["false_reid_count"] == 2
    assert result["duplicate_count"] == 1
    assert result["persistent_gt_count"] == 4
    assert result["representation_conditional_gt_count"] == 3
    assert result["end_to_end_recall"] == pytest.approx(0.25)
    assert result["representation_conditional_recall"] == pytest.approx(1 / 3)
    assert result["support_conditional_recall"] == pytest.approx(1 / 3)
    assert result["rigid_true_positive_count"] == 0
    assert result["rigid_gt_count"] == 1
    assert [row["outcome"] for row in result["outcomes"]] == [
        "true_positive",
        "duplicate",
        "false_reid",
        "false_reid",
        "endpoint_failure",
    ]


def test_fixed_bindings_are_reused_when_prediction_subset_changes() -> None:
    pair = _fixed_metric_pair()
    ground_truth = _fixed_metric_ground_truth(pair)
    bindings = build_fixed_endpoint_bindings(pair, ground_truth, support_domain="full")
    digest = bindings.content_sha256()

    complete = tuple(
        _fixed_prediction(
            pair,
            f"p{index}",
            f"ovimap:{index + 1}",
            f"ovimap:{index + 1}",
            0.9 - index * 0.05,
        )
        for index in range(5)
    )
    complete_result = evaluate_fixed_object_predictions(
        pair, complete, ground_truth, bindings
    )
    changed = (
        ObjectPairPrediction(
            prediction_id="unmatched-t0",
            pair_id=pair.pair_id,
            method_id="R_obj",
            t0_entity_id="ovimap:1",
            t1_entity_id=None,
            score=None,
            state="unmatched_t0",
        ),
        ObjectPairPrediction(
            prediction_id="unmatched-t1",
            pair_id=pair.pair_id,
            method_id="R_obj",
            t0_entity_id=None,
            t1_entity_id="ovimap:1",
            score=None,
            state="unmatched_t1",
        ),
        *complete[1:],
    )
    changed_result = evaluate_fixed_object_predictions(
        pair, changed, ground_truth, bindings
    )

    assert bindings.content_sha256() == digest
    assert complete_result["binding_sha256"] == digest
    assert changed_result["binding_sha256"] == digest
