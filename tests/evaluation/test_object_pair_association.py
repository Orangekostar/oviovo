from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.evaluation.object_pair_association import (
    AssociationConfig,
    IndependentFeatureBank,
    ObjectPairScoreMatrix,
    build_feature_score_matrix,
    build_geometric_score_matrix,
    build_rescene_score_matrix,
    solve_object_pair_assignment,
)
from src.evaluation.ovi_pair_views import (
    OviObjectEntityView,
    OviObjectPairView,
    OviObjectVisitView,
)
from src.oviv2.two_visit_contracts import TemporalQueryEvidence


def _visit(visit_id: int, *, displacement_m: float) -> OviObjectVisitView:
    points = np.asarray(
        [
            [displacement_m + 0.00, 0.0, 1.0],
            [displacement_m + 1.00, 0.0, 1.0],
            [displacement_m + 2.00, 0.0, 1.0],
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
            semantic_embedding=np.asarray(
                [1.0, 0.0] if index != 1 else [0.0, 1.0], dtype=np.float32
            ),
            semantic_label="chair" if index != 1 else "table",
            semantic_score=0.8,
            observation_frame_ids=(0,),
            observation_boxes_xyxy=((index, 0, index, 0),),
        )
        for index in range(3)
    )
    return OviObjectVisitView(
        visit_id=visit_id,
        scan_id=f"scan-{visit_id}",
        frame_count=1,
        points_xyz=points,
        palette_rgb_uint8=np.asarray(
            [[10, 20, 30], [11, 21, 31], [12, 22, 32]], dtype=np.uint8
        ),
        normals_xyz=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1)),
        normal_valid=np.ones(3, dtype=bool),
        source_vertex_indices=np.arange(3, dtype=np.int64),
        source_frame_ids_by_target=np.asarray([10], dtype=np.int64),
        entity_owner_indices=np.arange(3, dtype=np.int64),
        entities=entities,
        camera_rgb_uint8=np.asarray(
            [[100, 110, 120], [130, 140, 150], [160, 170, 180]], dtype=np.uint8
        ),
        appearance_valid=np.ones(3, dtype=bool),
        appearance_frame_ids=np.zeros(3, dtype=np.int64),
        appearance_source_frame_ids=np.full(3, 10, dtype=np.int64),
        appearance_rows=np.zeros(3, dtype=np.int64),
        appearance_columns=np.arange(3, dtype=np.int64),
        appearance_camera_depth_m=np.ones(3, dtype=np.float32),
        appearance_observed_depth_m=np.asarray(
            [1.01, 1.02, 1.03], dtype=np.float32
        ),
        appearance_depth_residual_m=np.asarray([0.01, 0.02, 0.03], dtype=np.float32),
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
        pair_id="scene0001_00-scene0001_01",
        visits=(_visit(0, displacement_m=0.0), _visit(1, displacement_m=100.0)),
        source_manifest_sha256="a" * 64,
        global_alignment=np.eye(4),
    )


def _feature_bank(pair: OviObjectPairView) -> IndependentFeatureBank:
    return IndependentFeatureBank(
        pair_content_sha256=pair.content_sha256(),
        candidate_ids=pair.candidate_ids,
        features=(
            np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        ),
        valid=(np.ones(3, dtype=bool), np.ones(3, dtype=bool)),
        extraction_mode="separate_visit_forward_before_temporal_overlay",
        visit_forward_sha256=("b" * 64, "c" * 64),
    )


def _evidence(pair: OviObjectPairView):
    sample = pair.geometric_sample(neural_voxel_size_m=0.02)
    affinities = {
        (0, "ovimap:1"): (0.50, 0.90),
        (0, "ovimap:2"): (0.20, 0.10),
        (0, "ovimap:3"): (0.05, 0.20),
        (1, "ovimap:1"): (0.10, 0.30),
        (1, "ovimap:2"): (0.25, 0.80),
        (1, "ovimap:3"): (0.20, 0.10),
    }
    token_scores = np.asarray(
        [
            [
                affinities[key][query]
                for key in zip(sample.visit_ids, sample.token_entity_ids, strict=True)
            ]
            for query in range(2)
        ],
        dtype=np.float32,
    )
    evidence = TemporalQueryEvidence(
        status="PASS",
        backend_name="rescene:fixture",
        backend_config_sha256="d" * 64,
        pair_sha256=sample.content_sha256(),
        temporal_query_ids=("q0", "q1"),
        query_masks=token_scores >= 0.50,
        token_scores=token_scores,
        query_scores=np.asarray([0.80, 0.50], dtype=np.float32),
        checkpoint_sha256="e" * 64,
        ranking_eligible=True,
        runtime_s=0.1,
        peak_memory_bytes=1024,
    )
    return sample, evidence


def test_g_f_r_share_candidates_and_cannot_mutate_pair(
    pair: OviObjectPairView,
) -> None:
    before = pair.content_sha256()
    sample, evidence = _evidence(pair)

    matrices = (
        build_geometric_score_matrix(pair, supported_only=True),
        build_feature_score_matrix(pair, _feature_bank(pair)),
        build_rescene_score_matrix(pair, sample, evidence),
    )

    assert all(matrix.candidate_ids == pair.candidate_ids for matrix in matrices)
    assert all(matrix.pair_content_sha256 == before for matrix in matrices)
    assert pair.content_sha256() == before
    with pytest.raises(ValueError, match="read-only"):
        matrices[0].scores[0, 0] = 0.0


def test_assignment_maximizes_valid_cardinality_then_score_and_uses_dummies(
    pair: OviObjectPairView,
) -> None:
    matrix = ObjectPairScoreMatrix(
        method_id="G_supported",
        pair_content_sha256=pair.content_sha256(),
        candidate_ids=pair.candidate_ids,
        scores=np.asarray(
            [
                [0.90, 0.80, 0.00],
                [0.85, 0.10, 0.00],
                [0.00, 0.00, 0.00],
            ],
            dtype=np.float64,
        ),
        eligible=np.ones((3, 3), dtype=bool),
    )

    predictions = solve_object_pair_assignment(
        pair, matrix, AssociationConfig(minimum_match_score=0.50)
    )

    matched = {
        (row.t0_entity_id, row.t1_entity_id, row.score)
        for row in predictions
        if row.is_matched
    }
    assert matched == {
        ("ovimap:1", "ovimap:2", 0.80),
        ("ovimap:2", "ovimap:1", 0.85),
    }
    assert sum(row.t0_entity_id == "ovimap:3" for row in predictions) == 1
    assert sum(row.t1_entity_id == "ovimap:3" for row in predictions) == 1
    assert all(
        sum(row.t0_entity_id == entity_id for row in predictions) == 1
        for _visit_id, entity_id in pair.candidate_ids[0]
    )
    assert all(
        sum(row.t1_entity_id == entity_id for row in predictions) == 1
        for _visit_id, entity_id in pair.candidate_ids[1]
    )


def test_assignment_can_leave_every_candidate_unmatched(
    pair: OviObjectPairView,
) -> None:
    matrix = ObjectPairScoreMatrix(
        method_id="G_supported",
        pair_content_sha256=pair.content_sha256(),
        candidate_ids=pair.candidate_ids,
        scores=np.full((3, 3), 0.90, dtype=np.float64),
        eligible=np.ones((3, 3), dtype=bool),
    )

    predictions = solve_object_pair_assignment(
        pair, matrix, AssociationConfig(minimum_match_score=0.95)
    )

    assert len(predictions) == 6
    assert all(not row.is_matched and row.score is None for row in predictions)


def test_geometric_scores_keep_large_displacement_edges_eligible(
    pair: OviObjectPairView,
) -> None:
    matrix = build_geometric_score_matrix(pair, supported_only=True)

    assert matrix.eligible[0, 0]
    assert matrix.scores[0, 0] > 0.0
    assert matrix.edge_evidence[0][0]["centroid_distance_m"] == pytest.approx(100.0)


def test_feature_scores_require_independent_pre_temporal_provenance(
    pair: OviObjectPairView,
) -> None:
    bank = _feature_bank(pair)
    matrix = build_feature_score_matrix(pair, bank)

    assert matrix.scores[0, 0] == pytest.approx(1.0)
    assert matrix.scores[0, 2] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="separate visit"):
        replace(bank, extraction_mode="joint_temporal_backbone")


def test_feature_scores_retain_unsupported_candidates_as_ineligible(
    pair: OviObjectPairView,
) -> None:
    bank = replace(
        _feature_bank(pair),
        features=(
            np.asarray([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        ),
        valid=(
            np.asarray([True, False, True]),
            np.asarray([True, True, False]),
        ),
    )

    matrix = build_feature_score_matrix(pair, bank)

    assert matrix.candidate_ids == pair.candidate_ids
    assert matrix.eligible.tolist() == [
        [True, True, False],
        [False, False, False],
        [True, True, False],
    ]
    assert np.all(matrix.scores[~matrix.eligible] == 0.0)


def test_rescene_score_is_exact_maximum_joint_query_support(
    pair: OviObjectPairView,
) -> None:
    sample, evidence = _evidence(pair)

    matrix = build_rescene_score_matrix(pair, sample, evidence)

    assert matrix.scores[0, 1] == pytest.approx(0.50 * 0.90 * 0.80)
    assert matrix.edge_evidence[0][1]["winning_query_id"] == "q1"
    assert matrix.edge_evidence[0][1]["t0_affinity"] == pytest.approx(0.90)
    assert matrix.edge_evidence[0][1]["t1_affinity"] == pytest.approx(0.80)
    assert 0.0 <= matrix.edge_evidence[0][1]["query_purity"] <= 1.0


def test_solver_rejects_candidate_pool_substitution(
    pair: OviObjectPairView,
) -> None:
    matrix = ObjectPairScoreMatrix(
        method_id="G_supported",
        pair_content_sha256=pair.content_sha256(),
        candidate_ids=(pair.candidate_ids[0][:-1], pair.candidate_ids[1]),
        scores=np.ones((2, 3), dtype=np.float64),
        eligible=np.ones((2, 3), dtype=bool),
    )

    with pytest.raises(ValueError, match="candidate pool"):
        solve_object_pair_assignment(pair, matrix, AssociationConfig())
