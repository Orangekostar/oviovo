from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.evaluation.dense_instance_repair_metrics import (
    DenseEndpointBindingError,
    build_dense_endpoint_bindings,
    build_p0_method_view,
    build_p1_method_view,
    build_p2_method_view,
    evaluate_dense_identity,
    evaluate_dense_instance_method,
)
from src.evaluation.ovi_pair_views import (
    OviObjectEntityView,
    OviObjectPairView,
    OviObjectVisitView,
)
from src.evaluation.rscan_gt_instances import (
    GroundTruthInstance,
    GroundTruthPair,
    IdentityRules,
    voxelize_points,
)
from src.evaluation.temporal_object_groups import build_temporal_object_groups
from src.oviv2.rescene_dense_instance_readout import (
    OWNER_BACKGROUND,
    OWNER_OVI_RESIDUAL,
    OWNER_QUERY,
    OWNER_UNKNOWN,
    DenseInstance,
    DensePairReadout,
    DenseQueryProposal,
    DenseVisitReadout,
)
from src.oviv2.two_visit_contracts import PairRelation


def _visit(visit_id: int) -> OviObjectVisitView:
    points = np.asarray(
        [
            [0.00, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [1.00, 0.0, 0.0],
            [1.01, 0.0, 0.0],
            [2.00, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    entities = tuple(
        OviObjectEntityView(
            visit_id=visit_id,
            entity_id=f"ovimap:{entity_index + 1}",
            source_instance_id=entity_index + 1,
            point_indices=np.asarray(rows, dtype=np.int64),
            palette_rgb=(50 + 40 * entity_index, 80, 120),
            semantic_embedding=np.asarray(
                [1.0 - entity_index, float(entity_index)], dtype=np.float32
            ),
            semantic_label=("chair", "table")[entity_index],
            semantic_score=0.9,
            observation_frame_ids=(0,),
            observation_boxes_xyxy=((0, 0, 1, 1),),
        )
        for entity_index, rows in enumerate(((0, 1), (2, 3)))
    )
    return OviObjectVisitView(
        visit_id=visit_id,
        scan_id=f"scan-{visit_id}",
        frame_count=1,
        points_xyz=points,
        palette_rgb_uint8=np.asarray(
            [[50, 80, 120], [50, 80, 120], [90, 80, 120], [90, 80, 120], [96, 96, 96]],
            dtype=np.uint8,
        ),
        normals_xyz=np.tile([0.0, 0.0, 1.0], (len(points), 1)).astype(np.float32),
        normal_valid=np.ones(len(points), dtype=np.bool_),
        source_vertex_indices=np.arange(len(points), dtype=np.int64),
        source_frame_ids_by_target=np.asarray([visit_id], dtype=np.int64),
        entity_owner_indices=np.asarray([0, 0, 1, 1, -1], dtype=np.int64),
        entities=entities,
        camera_rgb_uint8=np.full((len(points), 3), 128, dtype=np.uint8),
        appearance_valid=np.asarray([True, True, True, True, False]),
        appearance_frame_ids=np.asarray([0, 0, 0, 0, -1], dtype=np.int64),
        appearance_source_frame_ids=np.asarray(
            [visit_id, visit_id, visit_id, visit_id, -1], dtype=np.int64
        ),
        appearance_rows=np.asarray([0, 0, 0, 0, -1], dtype=np.int64),
        appearance_columns=np.asarray([0, 1, 2, 3, -1], dtype=np.int64),
        appearance_camera_depth_m=np.asarray([1.0, 1.0, 1.0, 1.0, np.nan], dtype=np.float32),
        appearance_observed_depth_m=np.asarray([1.0, 1.0, 1.0, 1.0, np.nan], dtype=np.float32),
        appearance_depth_residual_m=np.asarray([0.0, 0.0, 0.0, 0.0, np.nan], dtype=np.float32),
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


@pytest.fixture
def pair() -> OviObjectPairView:
    return OviObjectPairView(
        pair_id="dense-pair",
        visits=(_visit(0), _visit(1)),
        source_manifest_sha256="a" * 64,
        global_alignment=np.eye(4),
    )


def _ground_truth(pair: OviObjectPairView) -> GroundTruthPair:
    visits = tuple(
        tuple(
            GroundTruthInstance(
                instance_id=10 * (entity_index + 1) + visit_id,
                semantic_label=("chair", "table")[entity_index],
                voxels=voxelize_points(
                    visit.points_xyz[entity.point_indices], voxel_size_m=0.05
                ),
            )
            for entity_index, entity in enumerate(visit.entities)
        )
        for visit_id, visit in enumerate(pair.visits)
    )
    rules = IdentityRules.from_official_records(
        changes={
            "rigid": [
                {
                    "instance_reference": 10,
                    "instance_rescan": 11,
                    "symmetry": 0,
                    "transform": np.eye(4).tolist(),
                }
            ],
            "nonrigid": [
                {"instance_reference": 20, "instance_rescan": 21}
            ],
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


def _p1_grouping(pair: OviObjectPairView):
    relation = PairRelation(
        temporal_query_id="group-all",
        t0_entity_ids=("ovimap:1", "ovimap:2"),
        t1_entity_ids=("ovimap:1", "ovimap:2"),
        state="uncertain",
        query_confidence=0.9,
        evidence={"query_score": 0.9},
        identity_source="rescene",
    )
    return build_temporal_object_groups(pair, (relation,), variant_id="U3")


def _dense_instance(
    visit_id: int,
    instance_index: int,
    instance_id: str,
    points: tuple[int, ...],
    raw_query: int | None,
    temporal_id: str | None,
    parents: tuple[tuple[str, int], ...],
) -> DenseInstance:
    return DenseInstance(
        visit_id=visit_id,
        instance_index=instance_index,
        instance_id=instance_id,
        owner_source="query" if raw_query is not None else "ovi_residual",
        point_indices=np.asarray(points, dtype=np.int64),
        raw_query_index=raw_query,
        temporal_identity_id=temporal_id,
        parent_ovi_entity_point_counts=parents,
        semantic_embedding=np.asarray([0.5, 0.5], dtype=np.float32),
        semantic_labels=("chair", "table") if len(parents) > 1 else ("chair",),
        semantic_provenance="test",
        component_count=2 if len(parents) > 1 else 1,
        multi_object_conflict=len(parents) > 1,
    )


def _p2_readout(pair: OviObjectPairView) -> DensePairReadout:
    visits = []
    proposals = []
    for visit_id in (0, 1):
        instances = (
            _dense_instance(
                visit_id,
                0,
                f"P2:t{visit_id}:query_0000",
                (0, 2),
                0,
                None,
                (("ovimap:1", 1), ("ovimap:2", 1)),
            ),
            _dense_instance(
                visit_id,
                1,
                f"P2:t{visit_id}:query_0001",
                (1,),
                1,
                "query_0001",
                (("ovimap:1", 1),),
            ),
            _dense_instance(
                visit_id,
                2,
                f"P2:t{visit_id}:residual:ovimap:2",
                (3,),
                None,
                None,
                (("ovimap:2", 1),),
            ),
        )
        source_codes = np.asarray(
            [
                OWNER_QUERY,
                OWNER_QUERY,
                OWNER_QUERY,
                OWNER_OVI_RESIDUAL,
                OWNER_BACKGROUND if visit_id == 0 else OWNER_UNKNOWN,
            ],
            dtype=np.uint8,
        )
        visits.append(
            DenseVisitReadout(
                visit_id=visit_id,
                scan_id=pair.visits[visit_id].scan_id,
                owner_instance_indices=np.asarray([0, 1, 0, 2, -1], dtype=np.int64),
                owner_source_codes=source_codes,
                neural_valid=np.asarray([True, True, True, True, False]),
                dense_to_model_indices=np.asarray([0, 1, 2, 3, -1], dtype=np.int64),
                instances=instances,
            )
        )
        proposals.append(
            (
                DenseQueryProposal(
                    visit_id=visit_id,
                    raw_query_index=0,
                    query_id="query_0000",
                    query_score=float(np.float32(0.8)),
                    point_indices=np.asarray([0, 2], dtype=np.int64),
                ),
                DenseQueryProposal(
                    visit_id=visit_id,
                    raw_query_index=1,
                    query_id="query_0001",
                    query_score=float(np.float32(0.7)),
                    point_indices=np.asarray([1], dtype=np.int64),
                ),
            )
        )
    return DensePairReadout(
        pair_content_sha256=pair.content_sha256(),
        minimum_query_score=0.3,
        raw_query_indices=np.asarray([0, 1], dtype=np.int64),
        query_scores=np.asarray([0.8, 0.7], dtype=np.float32),
        visits=(visits[0], visits[1]),
        raw_proposals=(proposals[0], proposals[1]),
    )


def test_p0_p1_p2_share_dense_geometry_while_p2_splits_and_merges(
    pair: OviObjectPairView,
) -> None:
    p0 = build_p0_method_view(pair)
    p1 = build_p1_method_view(pair, _p1_grouping(pair))
    p2 = build_p2_method_view(pair, _p2_readout(pair))

    assert {view.source_xyz_sha256 for view in (p0, p1, p2)} == {
        p0.source_xyz_sha256
    }
    assert all(view.output_xyz_sha256 == view.source_xyz_sha256 for view in (p0, p1, p2))
    owner_shapes = [
        tuple(len(owner) for owner in view.owner_instance_indices)
        for view in (p0, p1, p2)
    ]
    assert owner_shapes == [
        (5, 5),
        (5, 5),
        (5, 5),
    ]
    assert p2.candidates[0][0].parent_ovi_entity_point_counts == (
        ("ovimap:1", 1),
        ("ovimap:2", 1),
    )
    assert {
        candidate.candidate_id
        for candidate in p2.candidates[0]
        if any(parent == "ovimap:1" for parent, _count in candidate.parent_ovi_entity_point_counts)
    } == {"P2:t0:query_0000", "P2:t0:query_0001"}

    p0_rows = evaluate_dense_instance_method(pair, p0, _ground_truth(pair))
    p1_rows = evaluate_dense_instance_method(pair, p1, _ground_truth(pair))
    assert all(row["f1_at_050"] == pytest.approx(1.0) for row in p0_rows)
    assert all(row["tp_at_050"] == 1 for row in p1_rows)
    assert all(row["fn_at_050"] == 1 for row in p1_rows)
    assert all(row["merge_prediction_count_at_050"] == 1 for row in p1_rows)


def test_p1_parent_counts_use_canonical_entity_id_order(
    pair: OviObjectPairView,
) -> None:
    visits = []
    for visit in pair.visits:
        entities = (
            replace(
                visit.entities[0], source_instance_id=2, entity_id="ovimap:2"
            ),
            replace(
                visit.entities[1], source_instance_id=10, entity_id="ovimap:10"
            ),
        )
        visits.append(replace(visit, entities=entities))
    pair = replace(pair, visits=(visits[0], visits[1]))
    relation = PairRelation(
        temporal_query_id="group-all",
        t0_entity_ids=("ovimap:2", "ovimap:10"),
        t1_entity_ids=("ovimap:2", "ovimap:10"),
        state="uncertain",
        query_confidence=0.9,
        evidence={"query_score": 0.9},
        identity_source="rescene",
    )
    grouping = build_temporal_object_groups(pair, (relation,), variant_id="U3")

    view = build_p1_method_view(pair, grouping)

    assert view.candidates[0][0].parent_ovi_entity_point_counts == (
        ("ovimap:10", 2),
        ("ovimap:2", 2),
    )


def test_instance_rows_report_exact_counts_structure_and_owner_fractions(
    pair: OviObjectPairView,
) -> None:
    p2 = build_p2_method_view(pair, _p2_readout(pair))

    rows = evaluate_dense_instance_method(pair, p2, _ground_truth(pair))

    assert len(rows) == 2
    first = rows[0]
    assert first["tp_at_050"] == 2
    assert first["fp_at_050"] == 1
    assert first["fn_at_050"] == 0
    assert first["precision_at_050"] == pytest.approx(2 / 3)
    assert first["recall_at_050"] == pytest.approx(1.0)
    assert first["f1_at_050"] == pytest.approx(0.8)
    assert first["tp_at_025"] == 2
    assert first["f1_at_025"] == pytest.approx(0.8)
    assert first["raw_candidate_count"] == 2
    assert first["final_instance_count"] == 3
    assert first["fragment_gt_count_at_050"] == 2
    assert first["merge_prediction_count_at_050"] == 1
    assert first["duplicate_prediction_count_at_050"] == 2
    assert first["split_parent_count"] == 2
    assert first["merged_candidate_count"] == 1
    assert first["query_owned_fraction"] == pytest.approx(3 / 5)
    assert first["residual_fraction"] == pytest.approx(1 / 5)
    assert first["background_fraction"] == pytest.approx(1 / 5)
    assert first["unknown_fraction"] == 0.0
    assert rows[1]["unknown_fraction"] == pytest.approx(1 / 5)
    assert first["source_point_count"] == first["final_point_count"] == 5
    assert first["geometric_change_count"] == 0


def test_candidate_bindings_are_pool_specific_and_precede_identity_scoring(
    pair: OviObjectPairView,
) -> None:
    ground_truth = _ground_truth(pair)
    p0 = build_p0_method_view(pair)
    p2 = build_p2_method_view(pair, _p2_readout(pair))
    p0_bindings = build_dense_endpoint_bindings(
        pair, p0, ground_truth, support_domain="full"
    )
    p2_bindings = build_dense_endpoint_bindings(
        pair, p2, ground_truth, support_domain="full"
    )

    assert p0_bindings.content_sha256() != p2_bindings.content_sha256()
    assert p0_bindings.method_view_sha256 == p0.content_sha256()
    assert p2_bindings.method_view_sha256 == p2.content_sha256()
    with pytest.raises(DenseEndpointBindingError, match="method view"):
        evaluate_dense_identity(p2, ground_truth, p0_bindings, iou_threshold=0.50)

    result = evaluate_dense_identity(
        p2, ground_truth, p2_bindings, iou_threshold=0.50
    )
    assert result["binding_sha256"] == p2_bindings.content_sha256()
    assert result["paired_prediction_count"] == 1
    assert result["true_positive_count"] == 1
    assert result["precision"] == pytest.approx(1.0)
    assert result["end_to_end_recall"] == pytest.approx(0.5)
    assert result["conditional_recall"] == pytest.approx(0.5)
    assert result["rigid_recall"] == pytest.approx(1.0)


def test_identity_conditional_recall_is_null_without_two_endpoint_gt_support(
    pair: OviObjectPairView,
) -> None:
    p2 = build_p2_method_view(pair, _p2_readout(pair))
    ground_truth = _ground_truth(pair)
    rules = IdentityRules.from_official_records(
        changes={"rigid": [], "nonrigid": [], "removed": []}, ambiguity=[]
    )
    disconnected = replace(ground_truth, visits=(ground_truth.visits[0], ()), identity_rules=rules)
    bindings = build_dense_endpoint_bindings(
        pair, p2, disconnected, support_domain="full"
    )

    result = evaluate_dense_identity(
        p2, disconnected, bindings, iou_threshold=0.50
    )

    assert result["conditional_gt_count"] == 0
    assert result["conditional_recall"] is None
    assert result["conditional_status"] == "NOT_COMPUTED_NO_FIXED_ENDPOINT_SUPPORT"
