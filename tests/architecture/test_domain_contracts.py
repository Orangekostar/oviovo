from __future__ import annotations

import numpy as np
import pytest

from src.domain.observations import FrameObservation, ObservationBatch, ObservationQuality


def make_observation(observation_id: str = "7:11") -> FrameObservation:
    return FrameObservation(
        observation_id=observation_id,
        frame_id=7,
        timestamp=1.25,
        source_proposal_id=11,
        source_backend="sam2",
        mask=np.array([[False, True], [False, False]], dtype=bool),
        bbox_xyxy=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        points_world=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
        voxel_keys=np.array([[0, 0, 20], [2, 0, 20]], dtype=np.int64),
        detector_label="book",
        detector_confidence=0.8,
        visual_embedding=np.array([1.0, 0.0], dtype=np.float32),
        quality=ObservationQuality(
            mask_confidence=0.9,
            valid_depth_ratio=1.0,
            visible_point_ratio=0.75,
            view_quality=0.8,
        ),
        refinement_key="7:11",
        parent_observation_ids=("7:3",),
    )


def test_frame_observation_defensively_freezes_arrays() -> None:
    source_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    observation = FrameObservation(
        observation_id="0:1",
        frame_id=0,
        timestamp=0.0,
        source_proposal_id=1,
        source_backend="sam2",
        mask=None,
        bbox_xyxy=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        points_world=source_points,
        voxel_keys=np.array([[0, 0, 20]], dtype=np.int64),
        quality=ObservationQuality(),
    )

    source_points[:] = 9.0
    assert np.array_equal(observation.points_world, [[0.0, 0.0, 1.0]])
    assert observation.points_world.flags.writeable is False
    with pytest.raises(ValueError):
        observation.points_world[0, 0] = 2.0


def test_observation_batch_rejects_mixed_frames_and_duplicate_ids() -> None:
    first = make_observation("7:11")
    duplicate = make_observation("7:11")
    with pytest.raises(ValueError, match="unique"):
        ObservationBatch(frame_id=7, observations=(first, duplicate))

    other_frame = make_observation("7:12")
    object.__setattr__(other_frame, "frame_id", 8)
    with pytest.raises(ValueError, match="frame_id"):
        ObservationBatch(frame_id=7, observations=(first, other_frame))


def test_observation_quality_is_bounded() -> None:
    with pytest.raises(ValueError, match="view_quality"):
        ObservationQuality(view_quality=1.1)


from src.domain.tracking import LocalTrack, LocalTrackBatch, LocalTrackState
from src.domain.visibility import VisibilityEvidence, VisibilityEvidenceBatch, VisibilityKind


def test_local_track_has_no_persistent_entity_id() -> None:
    track = LocalTrack(
        local_track_id="track-1",
        state=LocalTrackState.STABLE,
        observation_ids=("7:11",),
        first_frame_id=7,
        last_frame_id=7,
        first_timestamp=1.25,
        last_timestamp=1.25,
        hit_count=1,
        miss_count=0,
        centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([-0.1, -0.1, 0.9], dtype=np.float32),
        bbox_max=np.array([0.1, 0.1, 1.1], dtype=np.float32),
    )
    assert not hasattr(track, "entity_id")
    assert LocalTrackBatch(frame_id=7, tracks=(track,)).stable_tracks == (track,)


def test_visibility_batch_keeps_occlusion_separate_from_absence() -> None:
    occluded = VisibilityEvidence(
        entity_id="entity-1",
        frame_id=7,
        kind=VisibilityKind.OCCLUDED,
        projected_count=20,
        valid_depth_count=20,
        present_count=0,
        absent_count=0,
        occluded_count=18,
        unobserved_count=2,
    )
    batch = VisibilityEvidenceBatch(frame_id=7, evidence=(occluded,))
    assert batch.by_entity("entity-1").kind is VisibilityKind.OCCLUDED
    assert batch.by_entity("missing") is None


from src.domain.entities import (
    EntityLifecycleState,
    LifecycleDelta,
    LifecycleInterval,
    LifecycleTransition,
    PersistentEntity,
)


def test_persistent_entity_keeps_closed_lifecycle_intervals() -> None:
    entity = PersistentEntity(
        entity_id="entity-1",
        state=EntityLifecycleState.DORMANT,
        first_seen=1.0,
        last_seen=4.0,
        semantic_label="book",
        semantic_confidence=0.9,
        semantic_embedding=np.array([1.0, 0.0], dtype=np.float32),
        identity_embedding=np.array([0.0, 1.0], dtype=np.float32),
        geometry_handle="geometry/entity-1",
        ownership_revision=5,
        lifecycle_intervals=(
            LifecycleInterval(EntityLifecycleState.ACTIVE, 1.0, 4.0),
            LifecycleInterval(EntityLifecycleState.DORMANT, 4.0, None),
        ),
    )
    assert entity.lifecycle_intervals[-1].end_timestamp is None
    assert entity.semantic_embedding.flags.writeable is False


def test_lifecycle_delta_contains_commands_not_mutable_entities() -> None:
    transition = LifecycleTransition(
        entity_id="entity-1",
        from_state=EntityLifecycleState.ACTIVE,
        to_state=EntityLifecycleState.DORMANT,
        timestamp=4.0,
        reason="signed_absence",
        release_ownership=True,
    )
    delta = LifecycleDelta(base_revision=3, transitions=(transition,))
    assert delta.transitions == (transition,)


from src.domain.association import (
    AssociationDecision,
    AssociationDecisionBatch,
    AssociationKind,
    EntityBinding,
    EntityResolutionBatch,
)


def test_dormant_reid_is_distinct_from_active_match() -> None:
    match = AssociationDecision("track-1", AssociationKind.ACTIVE_MATCH, "entity-1", 0.8)
    reid = AssociationDecision("track-2", AssociationKind.DORMANT_REID, "entity-2", 0.9)
    batch = AssociationDecisionBatch(frame_id=7, decisions=(match, reid))
    assert batch.decisions[0].kind is AssociationKind.ACTIVE_MATCH
    assert batch.decisions[1].kind is AssociationKind.DORMANT_REID


def test_new_entity_id_appears_only_after_registry_resolution() -> None:
    decision = AssociationDecision("track-3", AssociationKind.CREATE, None, 0.7)
    assert decision.entity_id is None
    binding = EntityBinding("track-3", "entity-3", ("7:11",), AssociationKind.CREATE)
    resolution = EntityResolutionBatch(frame_id=7, bindings=(binding,))
    assert resolution.bindings[0].entity_id == "entity-3"
