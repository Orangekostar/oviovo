from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np

from src.oviv2.entity_epoch_update import RelationInferenceState, RelationSupport
from src.oviv2.query_instance_projection import ProjectionConfig
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    OviEntitySemanticEvidence,
    TemporalQueryEvidence,
)
from src.oviv2.two_visit_registration import RegistrationConfig

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _shape() -> np.ndarray:
    return np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.12, 0.00, 0.00],
            [0.00, 0.08, 0.00],
            [0.00, 0.00, 0.05],
            [0.12, 0.08, 0.05],
            [0.04, 0.02, 0.05],
            [0.09, 0.07, 0.01],
            [0.02, 0.06, 0.04],
            [0.11, 0.03, 0.04],
            [0.06, 0.08, 0.02],
            [0.03, 0.01, 0.02],
            [0.08, 0.04, 0.05],
        ],
        dtype=np.float64,
    )


def _pair(
    entities: tuple[tuple[int, int, np.ndarray, str, np.ndarray | None], ...],
) -> NeuralSampleMap:
    ordered = tuple(sorted(entities, key=lambda item: (item[0], item[1])))
    coordinates: list[np.ndarray] = []
    features: list[np.ndarray] = []
    visits: list[int] = []
    entity_ids: list[str] = []
    semantics: list[OviEntitySemanticEvidence] = []
    for visit_id, owner_id, points, label, descriptor in ordered:
        entity_id = f"ovimap:{owner_id}"
        coordinates.extend(
            np.column_stack((points, np.full(len(points), visit_id, dtype=np.float64)))
        )
        feature = (
            np.ones(3, dtype=np.float32)
            if descriptor is None
            else np.asarray(descriptor, dtype=np.float32)
        )
        features.extend([feature] * len(points))
        visits.extend([visit_id] * len(points))
        entity_ids.extend([entity_id] * len(points))
        semantics.append(
            OviEntitySemanticEvidence(
                visit_id=visit_id,
                entity_id=entity_id,
                semantic_label=label,
                semantic_score=0.9,
                semantic_embedding=descriptor,
            )
        )
    count = len(coordinates)
    visit_array = np.asarray(visits, dtype=np.int8)
    return NeuralSampleMap(
        coordinates_xyzt=np.asarray(coordinates, dtype=np.float64),
        features=np.asarray(features, dtype=np.float32),
        visit_ids=visit_array,
        source_visit_ids=visit_array,
        source_entity_ids=tuple(entity_ids),
        source_point_indices=np.arange(count, dtype=np.int64),
        source_to_token_offsets=np.arange(count + 1, dtype=np.int64),
        neural_voxel_size_m=0.01,
        feature_schema="ovi_descriptor",
        coordinate_frame_id="world",
        source_manifest_sha256=SHA_A,
        source_visit_map_sha256=(SHA_B, SHA_C),
        entity_semantics=tuple(semantics),
    )


def _registration_config() -> RegistrationConfig:
    return RegistrationConfig(
        registration_voxel_size_m=0.005,
        maximum_sample_points=1_000,
        maximum_iterations=30,
        trimmed_correspondence_fraction=0.9,
        maximum_correspondence_distance_m=0.12,
        inlier_distance_m=0.025,
        minimum_sample_points=8,
        minimum_inlier_count=8,
        minimum_directional_overlap=0.7,
        maximum_directional_median_m=0.02,
        maximum_directional_p90_m=0.03,
        minimum_second_singular_ratio=0.02,
        maximum_centroid_residual_m=0.02,
        minimum_principal_extent_ratio=0.7,
        maximum_principal_extent_ratio=1.4,
        maximum_static_centroid_displacement_m=0.12,
        maximum_moved_centroid_displacement_m=2.0,
    )


def _config() -> object:
    relations = importlib.import_module("src.oviv2.entity_epoch_relations")
    return relations.StrongGeometricRelationConfig(
        registration_config=_registration_config(),
        maximum_candidate_centroid_distance_m=2.0,
        minimum_relation_score=0.55,
        null_score=0.50,
        minimum_assignment_margin=0.05,
        maximum_registration_candidates_per_entity=4,
        minimum_separated_patches=3,
        patch_voxel_size_m=0.04,
        minimum_motion_residual_improvement_m=0.05,
        maximum_static_identity_residual_m=0.025,
    )


def test_zero_iou_moved_pair_with_soft_label_mismatch_can_be_accepted() -> None:
    relations = importlib.import_module("src.oviv2.entity_epoch_relations")
    shape = _shape()
    pair = _pair(
        (
            (0, 1, shape, "chair", np.asarray([1.0, 0.0, 0.0])),
            (
                1,
                10,
                shape + np.asarray([0.7, 0.0, 0.0]),
                "stool",
                np.asarray([1.0, 0.0, 0.0]),
            ),
        )
    )

    result = relations.build_g1_relation_support(pair, _config())

    assert len(result) == 1
    relation = result[0]
    assert relation.accepted
    assert not relation.assignment_is_null
    assert relation.t0_owner_entity_id == 1
    assert relation.t1_owner_entity_id == 10
    assert relation.t0_source_vertex_indices.tolist() == list(range(len(shape)))
    assert relation.t1_source_vertex_indices.tolist() == list(range(len(shape)))
    assert relation.original_location_iou == 0.0
    assert relation.semantic_compatibility < 1.0
    assert relation.appearance_cosine == 1.0
    assert relation.inference_state is RelationInferenceState.MOVED
    assert relation.motion_verified
    assert relation.transform_world_from_t0 is not None


def test_hungarian_assignment_uses_explicit_null_and_never_reuses_t1() -> None:
    relations = importlib.import_module("src.oviv2.entity_epoch_relations")
    shape = _shape()
    descriptor = np.asarray([1.0, 0.0, 0.0])
    pair = _pair(
        (
            (0, 1, shape, "chair", descriptor),
            (0, 2, shape + np.asarray([0.03, 0.0, 0.0]), "chair", descriptor),
            (1, 10, shape + np.asarray([0.01, 0.0, 0.0]), "chair", descriptor),
        )
    )

    result = relations.build_g1_relation_support(pair, _config())

    accepted = tuple(item for item in result if item.accepted)
    nulls = tuple(item for item in result if item.assignment_is_null)
    assert len(accepted) == 1
    assert [item.t1_owner_entity_id for item in accepted] == [10]
    assert len(nulls) == 1
    assert {item.t0_owner_entity_id for item in (*accepted, *nulls)} == {1, 2}


def test_motion_verification_rejects_low_spatial_patch_support() -> None:
    relations = importlib.import_module("src.oviv2.entity_epoch_relations")
    relation = RelationSupport(
        relation_id="geometry-g1:1:10",
        relation_source="geometry-g1",
        stable_entity_id="stable:g1:1",
        t0_owner_entity_id=1,
        t1_owner_entity_id=10,
        t0_source_surface_id=f"ovi-map:{SHA_B}",
        t1_source_surface_id=f"ovi-map:{SHA_C}",
        t0_source_vertex_indices=np.asarray([0, 1], dtype=np.int64),
        t1_source_vertex_indices=np.asarray([0, 1], dtype=np.int64),
        confidence=0.9,
        inference_state=RelationInferenceState.UNRESOLVED,
        accepted=True,
    )
    source = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    target = source + np.asarray([0.5, 0.0, 0.0])

    motion = relations.verify_relation_motion(
        relation=relation,
        source_points_xyz=source,
        target_points_xyz=target,
        registration_config=_registration_config(),
        minimum_separated_patches=3,
        patch_voxel_size_m=0.04,
        minimum_residual_improvement_m=0.05,
        maximum_static_identity_residual_m=0.025,
    )

    assert motion.state is RelationInferenceState.UNRESOLVED
    assert not motion.motion_verified
    assert "insufficient_spatial_support" in motion.rejection_reasons
    assert motion.transform_world_from_t0 is None


def test_g1_low_patch_support_falls_back_to_explicit_null() -> None:
    relations = importlib.import_module("src.oviv2.entity_epoch_relations")
    source = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    descriptor = np.asarray([1.0, 0.0, 0.0])
    pair = _pair(
        (
            (0, 1, source, "chair", descriptor),
            (1, 10, source + np.asarray([0.5, 0.0, 0.0]), "chair", descriptor),
        )
    )

    result = relations.build_g1_relation_support(pair, _config())

    assert len(result) == 1
    assert not result[0].accepted
    assert result[0].assignment_is_null
    assert "insufficient_spatial_support" in result[0].rejection_reasons


def test_motion_verification_keeps_translated_planar_shape_unresolved() -> None:
    relations = importlib.import_module("src.oviv2.entity_epoch_relations")
    source = np.asarray(
        [[0.04 * x, 0.04 * y, 0.0] for x in range(4) for y in range(4)],
        dtype=np.float64,
    )
    target = source + np.asarray([0.5, 0.0, 0.0])
    relation = RelationSupport(
        relation_id="geometry-g1:planar",
        relation_source="geometry-g1",
        stable_entity_id="stable:g1:planar",
        t0_owner_entity_id=1,
        t1_owner_entity_id=10,
        t0_source_surface_id=f"ovi-map:{SHA_B}",
        t1_source_surface_id=f"ovi-map:{SHA_C}",
        t0_source_vertex_indices=np.arange(len(source), dtype=np.int64),
        t1_source_vertex_indices=np.arange(len(target), dtype=np.int64),
        confidence=0.9,
        inference_state=RelationInferenceState.UNRESOLVED,
        accepted=True,
    )

    motion = relations.verify_relation_motion(
        relation=relation,
        source_points_xyz=source,
        target_points_xyz=target,
        registration_config=_registration_config(),
        minimum_separated_patches=3,
        patch_voxel_size_m=0.04,
        minimum_residual_improvement_m=0.05,
        maximum_static_identity_residual_m=0.025,
    )

    assert motion.state is RelationInferenceState.UNRESOLVED
    assert not motion.motion_verified
    assert motion.transform_world_from_t0 is None


def test_below_null_score_and_ambiguous_assignments_are_rejected() -> None:
    relations = importlib.import_module("src.oviv2.entity_epoch_relations")
    shape = _shape()
    descriptor = np.asarray([1.0, 0.0, 0.0])
    low_confidence_pair = _pair(
        (
            (0, 1, shape, "chair", descriptor),
            (
                1,
                10,
                shape + np.asarray([0.7, 0.0, 0.0]),
                "lamp",
                -descriptor,
            ),
        )
    )
    strict_null = replace(_config(), null_score=0.95)

    low_confidence = relations.build_g1_relation_support(
        low_confidence_pair, strict_null
    )

    assert len(low_confidence) == 1
    assert not low_confidence[0].accepted
    assert low_confidence[0].assignment_is_null
    assert "assigned_null" in low_confidence[0].rejection_reasons

    ambiguous_pair = _pair(
        (
            (0, 1, shape, "chair", descriptor),
            (1, 10, shape, "chair", descriptor),
            (1, 11, shape, "chair", descriptor),
        )
    )
    ambiguous = relations.build_g1_relation_support(ambiguous_pair, _config())

    assert len(ambiguous) == 1
    assert not ambiguous[0].accepted
    assert ambiguous[0].assignment_is_null
    assert "ambiguous_assignment" in ambiguous[0].rejection_reasons


def test_query_relation_projection_never_falls_back_below_coverage_threshold() -> None:
    relations = importlib.import_module("src.oviv2.entity_epoch_relations")
    shape = _shape()
    descriptor = np.asarray([1.0, 0.0, 0.0])
    pair = _pair(
        (
            (0, 1, shape, "chair", descriptor),
            (1, 10, shape, "chair", descriptor),
        )
    )
    mask = np.zeros((1, len(pair.visit_ids)), dtype=np.bool_)
    mask[0, 0] = True
    mask[0, len(shape)] = True
    evidence = TemporalQueryEvidence(
        status="PASS",
        backend_name="rescene:source-bound-repro",
        backend_config_sha256="d" * 64,
        pair_sha256=pair.content_sha256(),
        temporal_query_ids=("query:0001",),
        query_masks=mask,
        token_scores=mask.astype(np.float32),
        query_scores=np.asarray([0.99], dtype=np.float32),
        checkpoint_sha256="e" * 64,
        ranking_eligible=True,
        runtime_s=1.0,
        peak_memory_bytes=1,
    )

    result = relations.relation_support_from_queries(
        pair,
        evidence,
        relation_source="frozen-rescene",
        projection_config=ProjectionConfig(
            minimum_entity_token_coverage=0.5,
            minimum_source_point_coverage=0.5,
        ),
        minimum_query_score=0.5,
        minimum_competition_margin=0.05,
    )

    assert len(result) == 1
    relation = result[0]
    assert not relation.accepted
    assert relation.assignment_is_null
    assert relation.t0_mask_coverage == 1 / len(shape)
    assert relation.t1_mask_coverage == 1 / len(shape)
    assert relation.t0_mask_purity == 0.5
    assert relation.t1_mask_purity == 0.5
    assert relation.rejection_reasons == (
        "insufficient_t0_coverage",
        "insufficient_t1_coverage",
    )


def test_query_relations_enforce_one_to_one_across_duplicate_queries() -> None:
    relations = importlib.import_module("src.oviv2.entity_epoch_relations")
    shape = _shape()
    descriptor = np.asarray([1.0, 0.0, 0.0])
    pair = _pair(
        (
            (0, 1, shape, "chair", descriptor),
            (1, 10, shape, "chair", descriptor),
        )
    )
    masks = np.ones((2, len(pair.visit_ids)), dtype=np.bool_)
    evidence = TemporalQueryEvidence(
        status="PASS",
        backend_name="rescene:source-bound-repro",
        backend_config_sha256="d" * 64,
        pair_sha256=pair.content_sha256(),
        temporal_query_ids=("query:best", "query:duplicate"),
        query_masks=masks,
        token_scores=masks.astype(np.float32),
        query_scores=np.asarray([0.9, 0.8], dtype=np.float32),
        checkpoint_sha256="e" * 64,
        ranking_eligible=True,
        runtime_s=1.0,
        peak_memory_bytes=1,
    )

    result = relations.relation_support_from_queries(
        pair,
        evidence,
        relation_source="frozen-rescene",
        projection_config=ProjectionConfig(
            minimum_entity_token_coverage=0.5,
            minimum_source_point_coverage=0.5,
        ),
        minimum_query_score=0.5,
        minimum_competition_margin=0.05,
    )

    assert len(tuple(item for item in result if item.accepted)) == 1
    rejected = tuple(item for item in result if not item.accepted)
    assert len(rejected) == 1
    assert rejected[0].assignment_is_null
    assert rejected[0].rejection_reasons == ("one_to_one_conflict",)
