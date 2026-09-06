from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.oviv2.two_visit_contracts import PairRelation
from src.oviv2.two_visit_registration import (
    ORACLE_REGISTRATION_METHOD_ID,
    OracleTransformSource,
    RegistrationConfig,
    apply_rigid_transform,
    apply_rigid_transform_normals,
    official_row_vector_to_internal_transform,
    register_composite_relation,
    register_oracle_pair_relation,
    register_pair_relation,
    validate_rigid_transform,
)


def _relation(
    *,
    state: str = "persistent_moved",
    t0_ids: tuple[str, ...] = ("t0:chair",),
    t1_ids: tuple[str, ...] = ("t1:chair",),
) -> PairRelation:
    return PairRelation(
        temporal_query_id="geom:000001",
        t0_entity_ids=t0_ids,
        t1_entity_ids=t1_ids,
        state=state,
        query_confidence=0.9,
        evidence={"query_score": 0.9},
        identity_source="geometric_baseline",
    )


def _anisotropic_cloud() -> np.ndarray:
    grid = np.stack(
        np.meshgrid(
            np.linspace(-0.18, 0.18, 7),
            np.linspace(-0.10, 0.10, 6),
            np.linspace(-0.055, 0.055, 5),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    grid[:, 0] += 0.08 * grid[:, 1] ** 2 + 0.03 * grid[:, 2]
    grid[:, 1] += 0.05 * grid[:, 0] * grid[:, 2]
    grid[:, 2] += 0.02 * np.sin(7.0 * grid[:, 0]) + 0.04 * grid[:, 1]
    asymmetric = grid[~((grid[:, 0] > 0.08) & (grid[:, 1] < 0.0) & (grid[:, 2] > 0.0))]
    return asymmetric.astype(np.float64)


def _transform() -> np.ndarray:
    axis = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.63
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = [0.42, -0.24, 0.17]
    return result


def test_known_rotation_and_translation_are_recovered_deterministically() -> None:
    source = _anisotropic_cloud()
    expected = _transform()
    target = apply_rigid_transform(source, expected)
    source_before = source.copy()
    target_before = target.copy()

    first = register_pair_relation(
        _relation(),
        source,
        target,
        source_semantic_label="Chair",
        target_semantic_label="chair",
        config=RegistrationConfig(),
    )
    second = register_pair_relation(
        _relation(),
        source,
        target,
        source_semantic_label="Chair",
        target_semantic_label="chair",
        config=RegistrationConfig(),
    )

    assert first.accepted
    assert first.rejection_reasons == ()
    assert first.transform_world_from_t0 is not None
    assert np.allclose(first.transform_world_from_t0, expected, atol=2e-3)
    assert first.source_to_target_overlap == pytest.approx(1.0)
    assert first.target_to_source_overlap == pytest.approx(1.0)
    assert first.content_sha256() == second.content_sha256()
    assert first.to_json_record() == second.to_json_record()
    assert not first.transform_world_from_t0.flags.writeable
    assert np.array_equal(source, source_before)
    assert np.array_equal(target, target_before)


def test_official_row_transform_converts_once_and_normals_ignore_translation() -> None:
    internal = _transform()
    official_row = internal.T
    points = np.asarray([[0.2, -0.1, 0.3], [0.4, 0.5, -0.2]])
    normals = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    converted = official_row_vector_to_internal_transform(official_row)
    moved = apply_rigid_transform(points, converted)
    moved_normals = apply_rigid_transform_normals(normals, converted)

    homogeneous = np.column_stack((points, np.ones(len(points))))
    np.testing.assert_allclose(moved, (homogeneous @ official_row)[:, :3])
    np.testing.assert_allclose(moved_normals, normals @ official_row[:3, :3])
    np.testing.assert_allclose(np.linalg.norm(moved_normals, axis=1), 1.0)
    translated_again = internal.copy()
    translated_again[:3, 3] += 100.0
    np.testing.assert_allclose(
        apply_rigid_transform_normals(normals, translated_again), moved_normals
    )


def test_composite_registration_accepts_unlabeled_one_to_one_objects() -> None:
    source = _anisotropic_cloud()
    target = apply_rigid_transform(source, _transform())

    evidence = register_composite_relation(
        _relation(),
        source,
        target,
        source_semantic_label=None,
        target_semantic_label=None,
        config=RegistrationConfig(),
    )

    assert evidence.accepted
    assert evidence.semantic_label is None
    np.testing.assert_allclose(evidence.transform_world_from_t0, _transform(), atol=2e-3)


def test_oracle_source_composes_official_object_then_global_row_transforms() -> None:
    source = _anisotropic_cloud()
    object_from_reference = _transform()
    rescan_to_reference = np.eye(4, dtype=np.float64)
    rescan_to_reference[:3, 3] = [-0.1, 0.3, 0.2]
    expected = rescan_to_reference @ object_from_reference
    target = apply_rigid_transform(source, expected)
    relation = PairRelation(
        temporal_query_id="oracle:2",
        t0_entity_ids=("t0:object",),
        t1_entity_ids=("t1:object",),
        state="persistent_moved",
        query_confidence=1.0,
        evidence={"reference_instance_id": 2},
        identity_source="geometric_baseline",
    )
    oracle = OracleTransformSource(
        reference_instance_id=2,
        official_object_reference_to_rescan_row=object_from_reference.T,
        official_rescan_to_reference_row=rescan_to_reference.T,
    )

    evidence = register_oracle_pair_relation(
        relation,
        source,
        target,
        transform_source=oracle,
        source_semantic_label=None,
        target_semantic_label=None,
        config=RegistrationConfig(),
    )

    assert evidence.accepted
    assert evidence.method_id == ORACLE_REGISTRATION_METHOD_ID
    assert evidence.selected_initialization == oracle.source_type
    np.testing.assert_allclose(evidence.transform_world_from_t0, expected)


def test_oracle_registration_rejects_empty_support_without_crashing() -> None:
    relation = PairRelation(
        temporal_query_id="oracle:2",
        t0_entity_ids=("t0:object",),
        t1_entity_ids=("t1:object",),
        state="persistent_moved",
        query_confidence=1.0,
        evidence={"reference_instance_id": 2},
        identity_source="geometric_baseline",
    )
    oracle = OracleTransformSource(
        reference_instance_id=2,
        official_object_reference_to_rescan_row=np.eye(4),
        official_rescan_to_reference_row=np.eye(4),
    )

    evidence = register_oracle_pair_relation(
        relation,
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
        transform_source=oracle,
        source_semantic_label=None,
        target_semantic_label=None,
        config=RegistrationConfig(),
    )

    assert not evidence.accepted
    assert "low_support" in evidence.rejection_reasons


def test_static_relation_uses_identity_or_fails_closed() -> None:
    points = _anisotropic_cloud()

    accepted = register_pair_relation(
        _relation(state="persistent_static"),
        points,
        points.copy(),
        source_semantic_label="Chair",
        target_semantic_label="Chair",
        config=RegistrationConfig(),
    )
    shifted = register_pair_relation(
        _relation(state="persistent_static"),
        points,
        points + np.asarray([0.20, 0.0, 0.0]),
        source_semantic_label="Chair",
        target_semantic_label="Chair",
        config=RegistrationConfig(),
    )

    assert accepted.accepted
    assert accepted.selected_initialization == "static_identity"
    assert np.array_equal(accepted.transform_world_from_t0, np.eye(4))
    assert not shifted.accepted
    assert shifted.transform_world_from_t0 is None


def test_rigid_transform_validation_rejects_reflection() -> None:
    reflection = np.eye(4, dtype=np.float64)
    reflection[0, 0] = -1.0

    with pytest.raises(ValueError, match="determinant one"):
        validate_rigid_transform(reflection)


def test_low_support_and_linear_geometry_fail_closed() -> None:
    low_support = register_pair_relation(
        _relation(),
        np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        np.asarray([[0.3, 0.0, 0.0], [0.4, 0.0, 0.0]]),
        source_semantic_label="Chair",
        target_semantic_label="Chair",
        config=RegistrationConfig(),
    )
    line = np.column_stack(
        (
            np.linspace(0.0, 2.0, 100),
            np.zeros(100),
            np.zeros(100),
        )
    )
    degenerate = register_pair_relation(
        _relation(),
        line,
        line + np.asarray([0.4, 0.0, 0.0]),
        source_semantic_label="Chair",
        target_semantic_label="Chair",
        config=RegistrationConfig(),
    )

    assert not low_support.accepted
    assert "low_support" in low_support.rejection_reasons
    assert not degenerate.accepted
    assert "degenerate_geometry" in degenerate.rejection_reasons


def test_asymmetric_planar_geometry_can_constrain_a_rigid_transform() -> None:
    xy = np.stack(
        np.meshgrid(
            np.linspace(-0.24, 0.24, 13),
            np.linspace(-0.16, 0.16, 11),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    source = np.column_stack((xy, np.zeros(len(xy))))
    source = source[~((source[:, 0] > 0.08) & (source[:, 1] < -0.02))]
    angle = 0.4
    transform = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 0.3],
            [np.sin(angle), np.cos(angle), 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    target = apply_rigid_transform(source, transform)

    evidence = register_pair_relation(
        _relation(),
        source,
        target,
        source_semantic_label="Chair",
        target_semantic_label="Chair",
        config=RegistrationConfig(),
    )

    assert evidence.accepted
    assert np.allclose(evidence.transform_world_from_t0, transform, atol=2e-3)


def test_excessive_motion_and_mismatched_extent_fail_closed() -> None:
    source = _anisotropic_cloud()
    excessive = register_pair_relation(
        _relation(),
        source,
        source + np.asarray([2.20, 0.0, 0.0]),
        source_semantic_label="Chair",
        target_semantic_label="Chair",
        config=RegistrationConfig(),
    )
    scaled = source.copy()
    scaled[:, 0] *= 3.0
    extent_mismatch = register_pair_relation(
        _relation(),
        source,
        scaled + np.asarray([0.4, 0.0, 0.0]),
        source_semantic_label="Chair",
        target_semantic_label="Chair",
        config=RegistrationConfig(),
    )

    assert not excessive.accepted
    assert "excessive_centroid_displacement" in excessive.rejection_reasons
    assert not extent_mismatch.accepted
    assert "extent_ratio_out_of_range" in extent_mismatch.rejection_reasons


@pytest.mark.parametrize(
    ("relation", "source_label", "target_label", "reason"),
    [
        (
            _relation(
                state="split",
                t0_ids=("t0:chair",),
                t1_ids=("t1:a", "t1:b"),
            ),
            "Chair",
            "Chair",
            "ineligible_relation_state",
        ),
        (_relation(), "Wall", "Wall", "ineligible_semantic_label"),
        (_relation(), "Chair", "Table", "semantic_label_mismatch"),
    ],
)
def test_ineligible_relation_or_semantics_fail_closed(
    relation: PairRelation,
    source_label: str,
    target_label: str,
    reason: str,
) -> None:
    evidence = register_pair_relation(
        relation,
        _anisotropic_cloud(),
        _anisotropic_cloud(),
        source_semantic_label=source_label,
        target_semantic_label=target_label,
        config=RegistrationConfig(),
    )

    assert not evidence.accepted
    assert evidence.transform_world_from_t0 is None
    assert reason in evidence.rejection_reasons


def test_config_hash_and_json_are_stable() -> None:
    first = RegistrationConfig()
    second = RegistrationConfig(
        recoverable_semantic_labels=frozenset(first.recoverable_semantic_labels)
    )

    assert first.content_sha256() == second.content_sha256()
    assert first.to_json_record() == second.to_json_record()


def test_accepted_evidence_requires_complete_quality_measurements() -> None:
    source = _anisotropic_cloud()
    accepted = register_pair_relation(
        _relation(),
        source,
        source + np.asarray([0.3, 0.0, 0.0]),
        source_semantic_label="Chair",
        target_semantic_label="Chair",
        config=RegistrationConfig(),
    )
    assert accepted.accepted

    with pytest.raises(ValueError, match="quality measurements"):
        replace(accepted, source_to_target_overlap=None)
