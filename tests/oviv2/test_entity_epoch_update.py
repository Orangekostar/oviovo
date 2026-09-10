from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
import pytest

from src.oviv2.current_surface import CurrentEvidenceState
from src.oviv2.fine_dynamic_policy import PriorSurfaceState, SurfaceRetirementReason
from src.oviv2.fine_surface_validity import FineSurfaceEvidence


def _prior(
    current: list[bool],
    reasons: list[SurfaceRetirementReason],
    states: list[CurrentEvidenceState],
) -> PriorSurfaceState:
    return PriorSurfaceState(
        current_valid=np.asarray(current, dtype=np.bool_),
        retirement_reason_codes=np.asarray(reasons, dtype=np.uint8),
        evidence_state_codes=np.asarray(states, dtype=np.uint8),
    )


def test_no_new_evidence_inherits_b3_validity_and_tombstones() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")
    prior = PriorSurfaceState(
        current_valid=np.asarray([False, True, False], dtype=np.bool_),
        retirement_reason_codes=np.asarray(
            [
                SurfaceRetirementReason.DIRECT_FREE,
                SurfaceRetirementReason.NONE,
                SurfaceRetirementReason.REPLACED,
            ],
            dtype=np.uint8,
        ),
        evidence_state_codes=np.asarray(
            [
                CurrentEvidenceState.REVOKED_VISIBLE_FREE,
                CurrentEvidenceState.HISTORICAL_UNOBSERVED,
                CurrentEvidenceState.REPLACED_BY_CURRENT,
            ],
            dtype=np.uint8,
        ),
    )

    result = entity_epoch_update.resolve_entity_epoch_update(
        prior=prior,
        source_surface_id="ovi-t0",
        source_vertex_indices=np.asarray([10, 20, 30], dtype=np.int64),
        t0_owner_entity_ids=np.asarray([7, 7, 9], dtype=np.int64),
        evidence=FineSurfaceEvidence.empty(3),
        t1_replacement_mask=np.zeros(3, dtype=np.bool_),
        relation_support=(),
        config=entity_epoch_update.EntityEpochUpdateConfig(),
    )

    assert result.current_valid.tolist() == [False, True, False]
    assert result.retirement_reason_codes.tolist() == [
        SurfaceRetirementReason.DIRECT_FREE,
        SurfaceRetirementReason.NONE,
        SurfaceRetirementReason.REPLACED,
    ]
    assert result.evidence_state_codes.tolist() == [
        CurrentEvidenceState.REVOKED_VISIBLE_FREE,
        CurrentEvidenceState.HISTORICAL_UNOBSERVED,
        CurrentEvidenceState.REPLACED_BY_CURRENT,
    ]
    assert result.surface_deltas == ()
    assert not result.current_valid.flags.writeable
    assert not result.retirement_reason_codes.flags.writeable
    assert not result.evidence_state_codes.flags.writeable


def test_replacement_precedes_local_positive_tombstone_correction() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")
    prior = _prior(
        [True, False, False],
        [
            SurfaceRetirementReason.NONE,
            SurfaceRetirementReason.DIRECT_FREE,
            SurfaceRetirementReason.ENTITY_LIFT,
        ],
        [
            CurrentEvidenceState.HISTORICAL_UNOBSERVED,
            CurrentEvidenceState.REVOKED_VISIBLE_FREE,
            CurrentEvidenceState.REVOKED_VISIBLE_FREE,
        ],
    )
    evidence = FineSurfaceEvidence(
        present_observations=np.asarray([1, 2, 1], dtype=np.uint16),
        visible_absent_observations=np.asarray([0, 1, 1], dtype=np.uint16),
        occluded_observations=np.zeros(3, dtype=np.uint16),
        distinct_absent_viewpoints=np.asarray([0, 1, 1], dtype=np.uint8),
        last_supported_frames=np.asarray([12, 12, 10], dtype=np.int32),
        last_absent_frames=np.asarray([-1, 10, 11], dtype=np.int32),
    )

    result = entity_epoch_update.resolve_entity_epoch_update(
        prior=prior,
        source_surface_id="ovi-t0",
        source_vertex_indices=np.asarray([10, 20, 30], dtype=np.int64),
        t0_owner_entity_ids=np.asarray([7, 7, 7], dtype=np.int64),
        evidence=evidence,
        t1_replacement_mask=np.asarray([True, False, False], dtype=np.bool_),
        relation_support=(),
        config=entity_epoch_update.EntityEpochUpdateConfig(),
    )

    assert result.current_valid.tolist() == [False, True, False]
    assert result.retirement_reason_codes.tolist() == [
        SurfaceRetirementReason.REPLACED,
        SurfaceRetirementReason.NONE,
        SurfaceRetirementReason.ENTITY_LIFT,
    ]
    assert result.evidence_state_codes.tolist() == [
        CurrentEvidenceState.REPLACED_BY_CURRENT,
        CurrentEvidenceState.CURRENT_OBSERVED,
        CurrentEvidenceState.REVOKED_VISIBLE_FREE,
    ]
    assert [delta.action_reason.value for delta in result.surface_deltas] == [
        "local_positive_correction",
        "t1_replaced",
    ]
    assert result.surface_deltas[0].source_vertex_indices.tolist() == [20]
    assert result.surface_deltas[0].evidence_frame_ids == (10, 12)
    assert not result.surface_deltas[0].used_new_measurement


def test_only_later_reliable_negative_evidence_revokes_retained_surface() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")
    prior = _prior(
        [True, True],
        [SurfaceRetirementReason.NONE, SurfaceRetirementReason.NONE],
        [
            CurrentEvidenceState.HISTORICAL_UNOBSERVED,
            CurrentEvidenceState.HISTORICAL_UNOBSERVED,
        ],
    )
    evidence = FineSurfaceEvidence(
        present_observations=np.asarray([0, 1], dtype=np.uint16),
        visible_absent_observations=np.asarray([3, 3], dtype=np.uint16),
        occluded_observations=np.zeros(2, dtype=np.uint16),
        distinct_absent_viewpoints=np.asarray([2, 2], dtype=np.uint8),
        last_supported_frames=np.asarray([10, 13], dtype=np.int32),
        last_absent_frames=np.asarray([12, 12], dtype=np.int32),
    )

    result = entity_epoch_update.resolve_entity_epoch_update(
        prior=prior,
        source_surface_id="ovi-t0",
        source_vertex_indices=np.asarray([4, 5], dtype=np.int64),
        t0_owner_entity_ids=np.asarray([2, 2], dtype=np.int64),
        evidence=evidence,
        t1_replacement_mask=np.zeros(2, dtype=np.bool_),
        relation_support=(),
        config=entity_epoch_update.EntityEpochUpdateConfig(),
    )

    assert result.current_valid.tolist() == [False, True]
    assert result.retirement_reason_codes.tolist() == [
        SurfaceRetirementReason.DIRECT_FREE,
        SurfaceRetirementReason.NONE,
    ]
    assert len(result.surface_deltas) == 1
    assert result.surface_deltas[0].action_reason.value == "reliable_visible_free"
    assert result.surface_deltas[0].source_vertex_indices.tolist() == [4]
    assert result.surface_deltas[0].evidence_frame_ids == (10, 12)


def test_identity_reactivation_does_not_revive_retired_location() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")
    relation = entity_epoch_update.RelationSupport(
        relation_id="rescene:0007",
        relation_source="frozen-rescene",
        stable_entity_id="stable:chair-7",
        t0_owner_entity_id=7,
        t1_owner_entity_id=11,
        t0_source_surface_id="ovi-t0",
        t1_source_surface_id="ovi-t1",
        t0_source_vertex_indices=np.asarray([10], dtype=np.int64),
        t1_source_vertex_indices=np.asarray([40, 41], dtype=np.int64),
        confidence=0.94,
        inference_state=entity_epoch_update.RelationInferenceState.STATIC,
        accepted=True,
    )
    prior = _prior(
        [False],
        [SurfaceRetirementReason.DIRECT_FREE],
        [CurrentEvidenceState.REVOKED_VISIBLE_FREE],
    )

    result = entity_epoch_update.resolve_entity_epoch_update(
        prior=prior,
        source_surface_id="ovi-t0",
        source_vertex_indices=np.asarray([10], dtype=np.int64),
        t0_owner_entity_ids=np.asarray([7], dtype=np.int64),
        evidence=FineSurfaceEvidence.empty(1),
        t1_replacement_mask=np.asarray([False], dtype=np.bool_),
        relation_support=(relation,),
        config=entity_epoch_update.EntityEpochUpdateConfig(),
    )

    assert result.current_valid.tolist() == [False]
    assert result.surface_deltas == ()
    assert [
        (alias.visit_id, alias.owner_entity_id, alias.stable_entity_id)
        for alias in result.identity_aliases
    ] == [
        (0, 7, "stable:chair-7"),
        (1, 11, "stable:chair-7"),
    ]


def test_verified_motion_and_local_free_evidence_retire_only_old_episode() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = 0.5
    relation = entity_epoch_update.RelationSupport(
        relation_id="geom:0003",
        relation_source="geometry",
        stable_entity_id="stable:table-3",
        t0_owner_entity_id=3,
        t1_owner_entity_id=8,
        t0_source_surface_id="ovi-t0",
        t1_source_surface_id="ovi-t1",
        t0_source_vertex_indices=np.asarray([4], dtype=np.int64),
        t1_source_vertex_indices=np.asarray([14, 15], dtype=np.int64),
        confidence=0.91,
        inference_state=entity_epoch_update.RelationInferenceState.MOVED,
        accepted=True,
        motion_verified=True,
        transform_world_from_t0=transform,
        t0_evidence_frame_ids=(1, 2),
        t1_evidence_frame_ids=(11, 12),
    )
    prior = _prior(
        [True],
        [SurfaceRetirementReason.NONE],
        [CurrentEvidenceState.HISTORICAL_UNOBSERVED],
    )
    evidence = FineSurfaceEvidence(
        present_observations=np.asarray([0], dtype=np.uint16),
        visible_absent_observations=np.asarray([3], dtype=np.uint16),
        occluded_observations=np.asarray([0], dtype=np.uint16),
        distinct_absent_viewpoints=np.asarray([2], dtype=np.uint8),
        last_supported_frames=np.asarray([10], dtype=np.int32),
        last_absent_frames=np.asarray([12], dtype=np.int32),
    )

    result = entity_epoch_update.resolve_entity_epoch_update(
        prior=prior,
        source_surface_id="ovi-t0",
        source_vertex_indices=np.asarray([4], dtype=np.int64),
        t0_owner_entity_ids=np.asarray([3], dtype=np.int64),
        evidence=evidence,
        t1_replacement_mask=np.asarray([False], dtype=np.bool_),
        relation_support=(relation,),
        config=entity_epoch_update.EntityEpochUpdateConfig(),
    )

    assert result.current_valid.tolist() == [False]
    assert len(result.surface_deltas) == 1
    assert result.surface_deltas[0].action_reason.value == "moved_location_retired"
    assert result.surface_deltas[0].relation_id == "geom:0003"
    assert result.surface_deltas[0].stable_entity_id_before == "ovi-t0:3"
    assert result.surface_deltas[0].stable_entity_id_after == "stable:table-3"
    assert result.surface_deltas[0].episode_id_before == "ovi-t0:3/location:t0:3"
    assert result.surface_deltas[0].episode_id_after == ("stable:table-3/location:t0:3")
    assert [record.lifecycle_state.value for record in result.episode_records] == [
        "location_retired",
        "active",
    ]
    assert (
        result.episode_records[0].first_observation_frame,
        result.episode_records[0].last_observation_frame,
    ) == (1, 10)
    assert result.episode_records[0].stable_entity_id == "stable:table-3"
    assert result.episode_records[1].stable_entity_id == "stable:table-3"
    assert result.episode_records[0].episode_id != result.episode_records[1].episode_id
    np.testing.assert_array_equal(
        result.episode_records[1].transform_world_from_t0,
        transform,
    )


def test_unverified_centroid_motion_never_retires_surface() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")
    relation = entity_epoch_update.RelationSupport(
        relation_id="rescene:0011",
        relation_source="frozen-rescene",
        stable_entity_id="stable:cabinet-2",
        t0_owner_entity_id=2,
        t1_owner_entity_id=6,
        t0_source_surface_id="ovi-t0",
        t1_source_surface_id="ovi-t1",
        t0_source_vertex_indices=np.asarray([5], dtype=np.int64),
        t1_source_vertex_indices=np.asarray([25], dtype=np.int64),
        confidence=0.99,
        inference_state=entity_epoch_update.RelationInferenceState.MOVED,
        accepted=True,
    )
    prior = _prior(
        [True],
        [SurfaceRetirementReason.NONE],
        [CurrentEvidenceState.HISTORICAL_UNOBSERVED],
    )

    result = entity_epoch_update.resolve_entity_epoch_update(
        prior=prior,
        source_surface_id="ovi-t0",
        source_vertex_indices=np.asarray([5], dtype=np.int64),
        t0_owner_entity_ids=np.asarray([2], dtype=np.int64),
        evidence=FineSurfaceEvidence.empty(1),
        t1_replacement_mask=np.asarray([False], dtype=np.bool_),
        relation_support=(relation,),
        config=entity_epoch_update.EntityEpochUpdateConfig(),
    )

    assert result.current_valid.tolist() == [True]
    assert result.surface_deltas == ()
    assert result.episode_records[0].lifecycle_state.value == "uncertain"


def test_verified_motion_without_local_negative_keeps_old_episode_uncertain() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")
    transform = np.eye(4, dtype=np.float64)
    transform[1, 3] = 0.4
    relation = entity_epoch_update.RelationSupport(
        relation_id="geom:0009",
        relation_source="geometry",
        stable_entity_id="stable:chair-9",
        t0_owner_entity_id=9,
        t1_owner_entity_id=19,
        t0_source_surface_id="ovi-t0",
        t1_source_surface_id="ovi-t1",
        t0_source_vertex_indices=np.asarray([9], dtype=np.int64),
        t1_source_vertex_indices=np.asarray([19], dtype=np.int64),
        confidence=0.93,
        inference_state=entity_epoch_update.RelationInferenceState.MOVED,
        accepted=True,
        motion_verified=True,
        transform_world_from_t0=transform,
    )

    result = entity_epoch_update.resolve_entity_epoch_update(
        prior=_prior(
            [True],
            [SurfaceRetirementReason.NONE],
            [CurrentEvidenceState.HISTORICAL_UNOBSERVED],
        ),
        source_surface_id="ovi-t0",
        source_vertex_indices=np.asarray([9], dtype=np.int64),
        t0_owner_entity_ids=np.asarray([9], dtype=np.int64),
        evidence=FineSurfaceEvidence.empty(1),
        t1_replacement_mask=np.zeros(1, dtype=np.bool_),
        relation_support=(relation,),
        config=entity_epoch_update.EntityEpochUpdateConfig(),
    )

    assert result.current_valid.tolist() == [True]
    assert result.surface_deltas == ()
    assert result.episode_records[0].lifecycle_state.value == "uncertain"


def test_replaying_identical_positive_evidence_is_idempotent() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")
    evidence = FineSurfaceEvidence(
        present_observations=np.asarray([2], dtype=np.uint16),
        visible_absent_observations=np.asarray([1], dtype=np.uint16),
        occluded_observations=np.asarray([0], dtype=np.uint16),
        distinct_absent_viewpoints=np.asarray([1], dtype=np.uint8),
        last_supported_frames=np.asarray([12], dtype=np.int32),
        last_absent_frames=np.asarray([10], dtype=np.int32),
    )
    arguments = {
        "source_surface_id": "ovi-t0",
        "source_vertex_indices": np.asarray([20], dtype=np.int64),
        "t0_owner_entity_ids": np.asarray([7], dtype=np.int64),
        "evidence": evidence,
        "t1_replacement_mask": np.asarray([False], dtype=np.bool_),
        "relation_support": (),
        "config": entity_epoch_update.EntityEpochUpdateConfig(),
    }
    first = entity_epoch_update.resolve_entity_epoch_update(
        prior=_prior(
            [False],
            [SurfaceRetirementReason.ENTITY_LIFT],
            [CurrentEvidenceState.REVOKED_VISIBLE_FREE],
        ),
        **arguments,
    )
    second = entity_epoch_update.resolve_entity_epoch_update(
        prior=PriorSurfaceState(
            current_valid=first.current_valid,
            retirement_reason_codes=first.retirement_reason_codes,
            evidence_state_codes=first.evidence_state_codes,
        ),
        **arguments,
    )

    assert first.current_valid.tolist() == [True]
    assert len(first.surface_deltas) == 1
    assert second.current_valid.tolist() == [True]
    assert second.surface_deltas == ()


def test_accepted_relation_requires_local_support_at_both_visits() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")

    with pytest.raises(ValueError, match="local support at both visits"):
        entity_epoch_update.RelationSupport(
            relation_id="rescene:empty-t1",
            relation_source="frozen-rescene",
            stable_entity_id="stable:chair-7",
            t0_owner_entity_id=7,
            t1_owner_entity_id=11,
            t0_source_surface_id="ovi-t0",
            t1_source_surface_id="ovi-t1",
            t0_source_vertex_indices=np.asarray([10], dtype=np.int64),
            t1_source_vertex_indices=np.empty(0, dtype=np.int64),
            confidence=0.94,
            inference_state=entity_epoch_update.RelationInferenceState.STATIC,
            accepted=True,
        )


def test_motion_transform_must_be_proper_se3() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = 2.0

    with pytest.raises(ValueError, match="rotation must be orthogonal"):
        entity_epoch_update.RelationSupport(
            relation_id="geom:not-rigid",
            relation_source="geometry",
            stable_entity_id="stable:table-3",
            t0_owner_entity_id=3,
            t1_owner_entity_id=8,
            t0_source_surface_id="ovi-t0",
            t1_source_surface_id="ovi-t1",
            t0_source_vertex_indices=np.asarray([4], dtype=np.int64),
            t1_source_vertex_indices=np.asarray([14], dtype=np.int64),
            confidence=0.91,
            inference_state=entity_epoch_update.RelationInferenceState.MOVED,
            accepted=True,
            motion_verified=True,
            transform_world_from_t0=transform,
        )


def test_duplicate_accepted_relation_id_is_rejected() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")
    first = entity_epoch_update.RelationSupport(
        relation_id="geometry:duplicate",
        relation_source="geometry",
        stable_entity_id="stable:first",
        t0_owner_entity_id=1,
        t1_owner_entity_id=11,
        t0_source_surface_id="ovi-t0",
        t1_source_surface_id="ovi-t1",
        t0_source_vertex_indices=np.asarray([10], dtype=np.int64),
        t1_source_vertex_indices=np.asarray([110], dtype=np.int64),
        confidence=0.9,
        inference_state=entity_epoch_update.RelationInferenceState.STATIC,
        accepted=True,
    )
    second = replace(
        first,
        stable_entity_id="stable:second",
        t0_owner_entity_id=2,
        t1_owner_entity_id=12,
        t0_source_vertex_indices=np.asarray([20], dtype=np.int64),
        t1_source_vertex_indices=np.asarray([120], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="relation IDs must be unique"):
        entity_epoch_update.resolve_entity_epoch_update(
            prior=_prior(
                [True, True],
                [SurfaceRetirementReason.NONE, SurfaceRetirementReason.NONE],
                [
                    CurrentEvidenceState.HISTORICAL_UNOBSERVED,
                    CurrentEvidenceState.HISTORICAL_UNOBSERVED,
                ],
            ),
            source_surface_id="ovi-t0",
            source_vertex_indices=np.asarray([10, 20], dtype=np.int64),
            t0_owner_entity_ids=np.asarray([1, 2], dtype=np.int64),
            evidence=FineSurfaceEvidence.empty(2),
            t1_replacement_mask=np.zeros(2, dtype=np.bool_),
            relation_support=(first, second),
            config=entity_epoch_update.EntityEpochUpdateConfig(),
        )


def test_no_match_with_only_occlusion_keeps_surface_and_marks_episode_dormant() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")
    evidence = FineSurfaceEvidence(
        present_observations=np.asarray([0], dtype=np.uint16),
        visible_absent_observations=np.asarray([0], dtype=np.uint16),
        occluded_observations=np.asarray([3], dtype=np.uint16),
        distinct_absent_viewpoints=np.asarray([0], dtype=np.uint8),
        last_supported_frames=np.asarray([-1], dtype=np.int32),
        last_occluded_frames=np.asarray([12], dtype=np.int32),
    )

    result = entity_epoch_update.resolve_entity_epoch_update(
        prior=_prior(
            [True],
            [SurfaceRetirementReason.NONE],
            [CurrentEvidenceState.HISTORICAL_UNOBSERVED],
        ),
        source_surface_id="ovi-t0",
        source_vertex_indices=np.asarray([10], dtype=np.int64),
        t0_owner_entity_ids=np.asarray([1], dtype=np.int64),
        evidence=evidence,
        t1_replacement_mask=np.zeros(1, dtype=np.bool_),
        relation_support=(),
        config=entity_epoch_update.EntityEpochUpdateConfig(),
    )

    assert result.current_valid.tolist() == [True]
    assert result.surface_deltas == ()
    assert result.episode_records[0].lifecycle_state.value == "dormant"


def test_update_result_rejects_inconsistent_validity_and_reason() -> None:
    entity_epoch_update = importlib.import_module("src.oviv2.entity_epoch_update")

    with pytest.raises(ValueError, match="validity and retirement reasons disagree"):
        entity_epoch_update.EntityEpochUpdateResult(
            current_valid=np.asarray([True], dtype=np.bool_),
            retirement_reason_codes=np.asarray(
                [SurfaceRetirementReason.DIRECT_FREE], dtype=np.uint8
            ),
            evidence_state_codes=np.asarray(
                [CurrentEvidenceState.REVOKED_VISIBLE_FREE], dtype=np.uint8
            ),
        )
