from __future__ import annotations

import numpy as np
import pytest

from src.oviv2.current_surface import CurrentEvidenceState
from src.oviv2.entity_epoch_memory import (
    OFFLINE_REPLAY_ON_FINAL_OVI_SUPPORT,
    CroveMemoryIndex,
    MemoryCandidateObservation,
    MemoryEntityObservation,
    MemoryRetrievalMode,
    build_memory_relation_support,
)
from src.oviv2.entity_epoch_update import (
    EntityEpochUpdateConfig,
    EpisodeLifecycle,
    resolve_entity_epoch_update,
)
from src.oviv2.fine_dynamic_policy import PriorSurfaceState, SurfaceRetirementReason
from src.oviv2.fine_surface_validity import FineSurfaceEvidence


def _observation(
    *,
    stable_id: str,
    owner_id: int,
    descriptor: tuple[float, ...],
    frame_id: int,
    observation_id: int,
    lifecycle: EpisodeLifecycle = EpisodeLifecycle.ACTIVE,
    feature_space: str = "siglip-so400m-crop",
    projection_version: str = "crop-v1",
) -> MemoryEntityObservation:
    return MemoryEntityObservation(
        stable_entity_id=stable_id,
        owner_entity_id=owner_id,
        source_surface_id="ovi-t0",
        source_vertex_indices=np.asarray([owner_id - 1], dtype=np.int64),
        descriptor=np.asarray(descriptor, dtype=np.float32),
        feature_space_id=feature_space,
        projection_version=projection_version,
        observation_id=observation_id,
        frame_id=frame_id,
        visible_pixel_count=100,
        quality=0.9,
        view_direction_xyz=(1.0, 0.0, 0.0),
        lifecycle_state=lifecycle,
        observation_provenance=OFFLINE_REPLAY_ON_FINAL_OVI_SUPPORT,
    )


def _candidate(
    descriptor: tuple[float, ...],
    *,
    owner_id: int = 10,
    feature_space: str = "siglip-so400m-crop",
    projection_version: str = "crop-v1",
) -> MemoryCandidateObservation:
    return MemoryCandidateObservation(
        entity_id=f"ovimap:{owner_id}",
        source_surface_id="ovi-t1",
        source_vertex_indices=np.asarray([owner_id - 10], dtype=np.int64),
        descriptor=np.asarray(descriptor, dtype=np.float32),
        feature_space_id=feature_space,
        projection_version=projection_version,
        observation_id=100 + owner_id,
        frame_id=200 + owner_id,
        quality=0.8,
    )


def _supports(
    index: CroveMemoryIndex,
    candidate: MemoryCandidateObservation,
    *,
    mode: MemoryRetrievalMode = MemoryRetrievalMode.MULTIVIEW_BANK,
    minimum_cosine: float = 0.75,
    minimum_margin: float = 0.08,
):
    return build_memory_relation_support(
        index,
        (candidate,),
        retrieval_mode=mode,
        minimum_cosine=minimum_cosine,
        minimum_margin=minimum_margin,
    )


def test_memory_search_prefers_qualified_active_identity_over_dormant() -> None:
    index = (
        CroveMemoryIndex()
        .observe(
            _observation(
                stable_id="stable:active",
                owner_id=1,
                descriptor=(0.8, 0.6),
                frame_id=1,
                observation_id=1,
            )
        )
        .observe(
            _observation(
                stable_id="stable:dormant",
                owner_id=2,
                descriptor=(1.0, 0.0),
                frame_id=2,
                observation_id=2,
                lifecycle=EpisodeLifecycle.DORMANT,
            )
        )
    )

    (relation,) = _supports(index, _candidate((1.0, 0.0)))

    assert relation.stable_entity_id == "stable:active"
    assert relation.memory_identity_lifecycle == "active"
    assert relation.appearance_cosine == pytest.approx(0.8)
    assert relation.appearance_projection_version == "crop-v1"
    assert relation.observation_provenance == OFFLINE_REPLAY_ON_FINAL_OVI_SUPPORT


def test_memory_search_falls_back_to_dormant_when_active_is_below_gate() -> None:
    index = (
        CroveMemoryIndex()
        .observe(
            _observation(
                stable_id="stable:active",
                owner_id=1,
                descriptor=(0.0, 1.0),
                frame_id=1,
                observation_id=1,
            )
        )
        .observe(
            _observation(
                stable_id="stable:dormant",
                owner_id=2,
                descriptor=(1.0, 0.0),
                frame_id=2,
                observation_id=2,
                lifecycle=EpisodeLifecycle.DORMANT,
            )
        )
    )

    (relation,) = _supports(index, _candidate((1.0, 0.0)))

    assert relation.stable_entity_id == "stable:dormant"
    assert relation.memory_identity_lifecycle == "dormant"


@pytest.mark.parametrize(
    ("feature_space", "projection_version"),
    [
        ("rescene-query-128", "crop-v1"),
        ("siglip-so400m-crop", "crop-v2"),
    ],
)
def test_memory_never_compares_different_feature_or_projection_spaces(
    feature_space: str,
    projection_version: str,
) -> None:
    index = CroveMemoryIndex().observe(
        _observation(
            stable_id="stable:one",
            owner_id=1,
            descriptor=(1.0, 0.0),
            frame_id=1,
            observation_id=1,
        )
    )

    assert (
        _supports(
            index,
            _candidate(
                (1.0, 0.0),
                feature_space=feature_space,
                projection_version=projection_version,
            ),
        )
        == ()
    )


def test_multiview_bank_can_match_when_last_prototype_cannot() -> None:
    index = (
        CroveMemoryIndex()
        .observe(
            _observation(
                stable_id="stable:one",
                owner_id=1,
                descriptor=(1.0, 0.0),
                frame_id=1,
                observation_id=1,
            )
        )
        .observe(
            _observation(
                stable_id="stable:one",
                owner_id=1,
                descriptor=(0.0, 1.0),
                frame_id=2,
                observation_id=2,
            )
        )
    )
    candidate = _candidate((1.0, 0.0))

    assert (
        _supports(
            index,
            candidate,
            mode=MemoryRetrievalMode.LAST_PROTOTYPE,
            minimum_cosine=0.9,
        )
        == ()
    )
    (relation,) = _supports(
        index,
        candidate,
        mode=MemoryRetrievalMode.MULTIVIEW_BANK,
        minimum_cosine=0.9,
    )
    assert relation.appearance_cosine == pytest.approx(1.0)
    assert relation.memory_retrieval_mode == "multiview_bank"


def test_memory_duplicate_frame_does_not_increase_support() -> None:
    first = _observation(
        stable_id="stable:one",
        owner_id=1,
        descriptor=(1.0, 0.0),
        frame_id=7,
        observation_id=1,
    )
    duplicate = _observation(
        stable_id="stable:one",
        owner_id=1,
        descriptor=(0.0, 1.0),
        frame_id=7,
        observation_id=2,
    )
    once = CroveMemoryIndex().observe(first)

    twice = once.observe(duplicate)

    assert twice is once
    (entry,) = twice.entries
    assert entry.accepted_frame_ids == (7,)
    assert len(entry.feature_bank.prototypes) == 1
    assert entry.feature_bank.prototypes[0].observation_count == 1
    assert entry.last_descriptor == pytest.approx((1.0, 0.0))


def test_memory_identity_reactivation_does_not_revive_prior_location() -> None:
    index = CroveMemoryIndex().observe(
        _observation(
            stable_id="stable:dormant",
            owner_id=1,
            descriptor=(1.0, 0.0),
            frame_id=1,
            observation_id=1,
            lifecycle=EpisodeLifecycle.DORMANT,
        )
    )
    (relation,) = _supports(index, _candidate((1.0, 0.0)))
    prior = PriorSurfaceState(
        current_valid=np.asarray([False], dtype=np.bool_),
        retirement_reason_codes=np.asarray(
            [SurfaceRetirementReason.DIRECT_FREE], dtype=np.uint8
        ),
        evidence_state_codes=np.asarray(
            [CurrentEvidenceState.REVOKED_VISIBLE_FREE], dtype=np.uint8
        ),
    )

    update = resolve_entity_epoch_update(
        prior=prior,
        source_surface_id="ovi-t0",
        source_vertex_indices=np.asarray([0], dtype=np.int64),
        t0_owner_entity_ids=np.asarray([1], dtype=np.int64),
        evidence=FineSurfaceEvidence.empty(1),
        t1_replacement_mask=np.asarray([False], dtype=np.bool_),
        relation_support=(relation,),
        config=EntityEpochUpdateConfig(),
    )

    assert relation.inference_state.value == "unresolved"
    assert relation.transform_world_from_t0 is None
    assert update.current_valid.tolist() == [False]
    assert update.retirement_reason_codes.tolist() == [
        SurfaceRetirementReason.DIRECT_FREE
    ]
    assert update.surface_deltas == ()
