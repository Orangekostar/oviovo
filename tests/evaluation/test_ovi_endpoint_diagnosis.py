from __future__ import annotations

import numpy as np

from src.evaluation.ovi_endpoint_diagnosis import (
    RawDenseProposal,
    diagnose_visit_endpoints,
)
from src.evaluation.ovi_pair_views import OviObjectEntityView, OviObjectVisitView
from src.evaluation.rscan_gt_instances import (
    GroundTruthInstance,
    PredictedInstance,
    voxelize_points,
)


def _visit(
    points: list[list[float]],
    entity_points: list[list[int]],
    *,
    supported: list[bool] | None = None,
) -> OviObjectVisitView:
    xyz = np.asarray(points, dtype=np.float32)
    count = len(xyz)
    valid = np.ones(count, dtype=np.bool_) if supported is None else np.asarray(supported)
    owners = np.full(count, -1, dtype=np.int64)
    entities = []
    for entity_index, indices in enumerate(entity_points):
        owner_rows = np.asarray(indices, dtype=np.int64)
        owners[owner_rows] = entity_index
        entities.append(
            OviObjectEntityView(
                visit_id=0,
                entity_id=f"ovimap:{entity_index + 1}",
                source_instance_id=entity_index + 1,
                point_indices=owner_rows,
                palette_rgb=(20 + entity_index, 40, 60),
                semantic_embedding=np.asarray([1.0, float(entity_index)], dtype=np.float32),
                semantic_label="object",
                semantic_score=1.0,
                observation_frame_ids=(0,),
                observation_boxes_xyxy=((0, 0, 1, 1),),
            )
        )
    invalid = ~valid
    return OviObjectVisitView(
        visit_id=0,
        scan_id="scan-0",
        frame_count=1,
        points_xyz=xyz,
        palette_rgb_uint8=np.full((count, 3), 80, dtype=np.uint8),
        normals_xyz=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (count, 1)),
        normal_valid=np.ones(count, dtype=np.bool_),
        source_vertex_indices=np.arange(count, dtype=np.int64),
        source_frame_ids_by_target=np.asarray([0], dtype=np.int64),
        entity_owner_indices=owners,
        entities=tuple(entities),
        camera_rgb_uint8=np.full((count, 3), 100, dtype=np.uint8),
        appearance_valid=valid,
        appearance_frame_ids=np.where(valid, 0, -1),
        appearance_source_frame_ids=np.where(valid, 0, -1),
        appearance_rows=np.where(valid, 1, -1),
        appearance_columns=np.where(valid, np.arange(count), -1),
        appearance_camera_depth_m=np.where(valid, 1.0, np.nan).astype(np.float32),
        appearance_observed_depth_m=np.where(valid, 1.0, np.nan).astype(np.float32),
        appearance_depth_residual_m=np.where(valid, 0.0, np.nan).astype(np.float32),
        native_manifest_sha256="1" * 64,
        materialized_manifest_sha256="2" * 64,
        source_artifact_sha256={
            "instance_color_log": "3" * 64,
            "instance_mesh": "4" * 64,
            "semantic_features": "5" * 64,
        },
        global_alignment_application="identity_reference",
    )


def test_diagnosis_separates_atomic_union_split_raw_and_final_limits() -> None:
    visit = _visit(
        [[0.01, 0.0, 0.0], [0.06, 0.0, 0.0], [0.11, 0.0, 0.0],
         [0.16, 0.0, 0.0], [0.21, 0.0, 0.0]],
        [[0, 4], [1], [2]],
        supported=[True, True, False, False, True],
    )
    target = GroundTruthInstance(
        instance_id=7,
        semantic_label="chair",
        voxels=frozenset({(0, 0, 0), (1, 0, 0), (2, 0, 0), (10, 0, 0)}),
    )
    raw = (
        RawDenseProposal("query_0000", 0.8, np.asarray([0, 1, 2], dtype=np.int64)),
        RawDenseProposal("query_0001", 0.2, np.asarray([4], dtype=np.int64)),
    )
    final = (
        PredictedInstance("P2:query_0000", frozenset({(0, 0, 0), (1, 0, 0)})),
    )

    row = diagnose_visit_endpoints(
        pair_id="pair",
        visit=visit,
        ground_truth=(target,),
        raw_proposals=raw,
        confident_query_ids=frozenset({"query_0000"}),
        final_predictions=final,
        change_type_by_gt={7: "rigid"},
        voxel_size_m=0.05,
    )[0]

    assert row.full_gt_voxels == 4
    assert row.d_intersection_voxels == 3
    assert row.d_coverage == 0.75
    assert row.d_owned_coverage == 0.75
    assert row.d_supported_coverage == 0.5
    assert row.best_atomic_iou == 0.25
    assert row.best_atomic_precision == 1.0
    assert row.best_atomic_recall == 0.25
    assert row.second_atomic_iou == 0.25
    assert row.union_oracle_iou == 0.6
    assert row.union_search_exact is True
    assert row.split_oracle_iou == 0.75
    assert row.split_oracle_voxels == frozenset({(0, 0, 0), (1, 0, 0), (2, 0, 0)})
    assert row.best_raw_mask_iou == 0.75
    assert row.best_raw_query_id == "query_0000"
    assert row.best_raw_query_score == 0.8
    assert row.best_confident_mask_iou == 0.75
    assert row.final_p2_iou == 0.5
    assert row.change_type == "rigid"


def test_large_union_search_is_labeled_approximate_and_keeps_best_single() -> None:
    points = [[0.01 + 0.05 * index, 0.0, 0.0] for index in range(13)]
    visit = _visit(points, [[index] for index in range(13)])
    target = GroundTruthInstance(
        instance_id=1,
        semantic_label="object",
        voxels=voxelize_points(visit.points_xyz, voxel_size_m=0.05),
    )

    row = diagnose_visit_endpoints(
        pair_id="pair",
        visit=visit,
        ground_truth=(target,),
        raw_proposals=(),
        confident_query_ids=frozenset(),
        final_predictions=(),
        change_type_by_gt={},
        voxel_size_m=0.05,
        exact_union_candidate_limit=12,
    )[0]

    assert row.union_search_exact is False
    assert row.union_oracle_iou >= row.best_atomic_iou
    assert row.union_candidate_count == 13


def test_missing_sensor_visibility_remains_null_and_diagnosis_can_be_mixed() -> None:
    visit = _visit([[0.01, 0.0, 0.0]], [[0]])
    target = GroundTruthInstance(
        instance_id=1,
        semantic_label="object",
        voxels=voxelize_points(visit.points_xyz, voxel_size_m=0.05),
    )

    row = diagnose_visit_endpoints(
        pair_id="pair",
        visit=visit,
        ground_truth=(target,),
        raw_proposals=(),
        confident_query_ids=frozenset(),
        final_predictions=(),
        change_type_by_gt={},
        voxel_size_m=0.05,
    )[0]
    payload = row.to_dict()

    assert payload["sensor_visible_gt_voxels"] is None
    assert payload["sensor_visibility_status"] == "NOT_COMPUTED"
    assert payload["best_raw_mask_iou"] is None
    assert payload["best_confident_mask_iou"] is None
    assert payload["final_P2_iou"] is None
    assert payload["provisional_failure_type"] == "mixed"
