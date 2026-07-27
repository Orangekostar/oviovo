from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import numpy as np
import pytest

from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.temporal_association import (
    TemporalAssignmentDiagnostic,
    TemporalAssociationResult,
    TemporalAssociationTarget,
    associate_temporal_observations,
)
from src.oviv2.temporal_config import TemporalAssociationConfig, TemporalIdentityConfig
from src.oviv2.temporal_lifecycle import TemporalLifecycle


def config(**changes: object) -> TemporalAssociationConfig:
    values: dict[str, object] = {
        "visual_weight": 0.3,
        "semantic_weight": 0.2,
        "size_weight": 0.1,
        "motion_weight": 0.2,
        "geometry_weight": 0.2,
        "minimum_score": 0.5,
        "maximum_centroid_distance_m": 2.0,
        "semantic_conflict_probability": 0.8,
        "conflict_override_visual": 0.9,
        "conflict_override_geometry": 0.8,
    }
    values.update(changes)
    return TemporalAssociationConfig(**values)  # type: ignore[arg-type]


def reid_config(**changes: object) -> TemporalIdentityConfig:
    values: dict[str, object] = {
        "maximum_identities": 100,
        "maximum_dormant_frames": 30,
        "minimum_reid_similarity": 0.8,
        "maximum_reid_distance_m": 8.0,
    }
    values.update(changes)
    return TemporalIdentityConfig(**values)  # type: ignore[arg-type]


def observation(
    observation_id: int,
    *,
    centroid: tuple[float, float, float] = (0.0, 0.0, 0.0),
    extent: tuple[float, float, float] = (1.0, 1.0, 1.0),
    semantic_id: int = 1,
    confidence: float = 0.9,
    image_feature: np.ndarray | None = None,
    feature_model_id: str = "clip",
    kind: ObservationKind = ObservationKind.OBJECT,
) -> FrameObservation:
    lower = tuple(center - size / 2.0 for center, size in zip(centroid, extent))
    upper = tuple(center + size / 2.0 for center, size in zip(centroid, extent))
    return FrameObservation(
        observation_id=observation_id,
        frame_id=3,
        timestamp=3.0,
        kind=kind,
        label="object",
        semantic_id=semantic_id,
        confidence=confidence,
        mask=np.ones((1, 1), dtype=bool),
        bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        voxel_keys=frozenset({(observation_id, 0, 0)}),
        centroid_xyz=centroid,
        bounds_min_xyz=lower,
        bounds_max_xyz=upper,
        image_feature=image_feature,
        feature_model_id=feature_model_id if image_feature is not None else None,
    )


def target(
    entity_id: int,
    *,
    lifecycle: TemporalLifecycle = TemporalLifecycle.ACTIVE,
    centroid: tuple[float, float, float] = (0.0, 0.0, 0.0),
    predicted: tuple[float, float, float] | None = None,
    extent: tuple[float, float, float] = (1.0, 1.0, 1.0),
    prototype: np.ndarray | None = None,
    semantics: tuple[tuple[int, float], ...] = ((1, 1.0),),
    feature_model_id: str = "clip",
) -> TemporalAssociationTarget:
    return TemporalAssociationTarget(
        entity_id=entity_id,
        lifecycle=lifecycle,
        centroid_xyz=centroid,
        extent_xyz=extent,
        image_prototype=prototype,
        semantic_probabilities=semantics,
        predicted_centroid_xyz=centroid if predicted is None else predicted,
        feature_model_id=feature_model_id if prototype is not None else None,
    )


def test_visual_requires_matching_feature_model_provenance() -> None:
    obs = observation(1, image_feature=np.array([1.0, 0.0]), feature_model_id="new")
    wrong = target(2, prototype=np.array([1.0, 0.0]), feature_model_id="old")
    right = target(3, prototype=np.array([1.0, 0.0]), feature_model_id="new")
    visual_only = config(
        visual_weight=1.0,
        semantic_weight=0.0,
        size_weight=0.0,
        motion_weight=0.0,
        geometry_weight=0.0,
        minimum_score=0.5,
    )
    assert associate_temporal_observations((obs,), (wrong,), visual_only).assignments == ()
    assert associate_temporal_observations((obs,), (right,), visual_only).assignments == ((1, 3),)


def test_legacy_prototype_without_model_is_valid_but_visual_is_unavailable() -> None:
    legacy = TemporalAssociationTarget(
        entity_id=2,
        lifecycle=TemporalLifecycle.ACTIVE,
        centroid_xyz=(0.0, 0.0, 0.0),
        extent_xyz=(1.0, 1.0, 1.0),
        image_prototype=np.array([1.0, 0.0]),
        semantic_probabilities=(),
        predicted_centroid_xyz=(0.0, 0.0, 0.0),
    )
    obs = observation(1, semantic_id=0, image_feature=np.array([1.0, 0.0]))
    visual_only = config(
        visual_weight=1.0, semantic_weight=0.0, size_weight=0.0,
        motion_weight=0.0, geometry_weight=0.0, minimum_score=0.0,
    )
    assert associate_temporal_observations((obs,), (legacy,), visual_only).assignments == ()
    with pytest.raises(ValueError, match="prototype"):
        replace(legacy, image_prototype=None, feature_model_id="clip")


def test_a2_a3_exclude_dormant_candidates() -> None:
    obs = observation(1, image_feature=np.array([1.0, 0.0]))
    active = target(20, centroid=(0.6, 0.0, 0.0), prototype=np.array([1.0, 0.0]))
    dormant = target(
        10,
        lifecycle=TemporalLifecycle.DORMANT,
        prototype=np.array([1.0, 0.0]),
    )

    result = associate_temporal_observations((obs,), (dormant, active), config(), dormant_reid=None)

    assert result.assignments == ((1, 20),)
    assert result.unmatched_entity_ids == (10,)


def test_a4_active_and_dormant_compete_in_one_global_assignment() -> None:
    obs = observation(1, image_feature=np.array([1.0, 0.0]))
    active = target(20, centroid=(0.8, 0.0, 0.0), prototype=np.array([1.0, 0.0]))
    dormant = target(10, lifecycle=TemporalLifecycle.DORMANT, prototype=np.array([1.0, 0.0]))
    result = associate_temporal_observations(
        (obs,), (active, dormant), config(), dormant_reid=reid_config()
    )
    assert result.reid_opportunity_count == len(result.reid_opportunity_pairs)
    assert result.reid_trigger_count == len(result.reid_trigger_pairs)
    assert set(result.reid_trigger_pairs) <= set(result.reid_opportunity_pairs)
    assert result.assignments == ((1, 10),)
    assert (result.reid_opportunity_count, result.reid_trigger_count) == (1, 1)
    assert result.assignment_diagnostics == (
        TemporalAssignmentDiagnostic(
            observation_id=1,
            entity_id=10,
            score=result.assignment_diagnostics[0].score,
            target_lifecycle=TemporalLifecycle.DORMANT,
            appearance_similarity=1.0,
            feature_model_id="clip",
            feature_model_match=True,
            semantic_qualified=True,
            high_confidence_identity_match=True,
        ),
    )


def test_active_dormant_global_identity_selection_ignores_all_predictions_and_motion_weight() -> None:
    obs = observation(1, centroid=(0.0, 0.0, 0.0), image_feature=np.array([1.0, 0.0]))
    for active_prediction in ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)):
        for dormant_prediction in ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)):
            for motion_weight in (0.0, 1000.0):
                active = target(
                    20, centroid=(0.0, 0.0, 0.0), predicted=active_prediction,
                    prototype=np.array([1.0, 0.0]),
                )
                dormant = target(
                    10, lifecycle=TemporalLifecycle.DORMANT,
                    centroid=(0.8, 0.0, 0.0), predicted=dormant_prediction,
                    prototype=np.array([1.0, 0.0]),
                )
                result = associate_temporal_observations(
                    (obs,), (active, dormant), config(motion_weight=motion_weight),
                    dormant_reid=reid_config(),
                )
                assert result.assignments == ((1, 20),)


def test_a2_a3_active_gate_uses_current_not_predicted_centroid() -> None:
    obs = observation(1, centroid=(0.0, 0.0, 0.0))
    currently_near = target(1, centroid=(0.5, 0.0, 0.0), predicted=(100.0, 0.0, 0.0))
    currently_far = target(2, centroid=(100.0, 0.0, 0.0), predicted=(0.0, 0.0, 0.0))
    assert associate_temporal_observations(
        (obs,), (currently_near, currently_far), config(), dormant_reid=None
    ).assignments == ((1, 1),)


def test_assignment_diagnostic_is_frozen_sorted_and_fails_closed() -> None:
    result = associate_temporal_observations(
        (observation(1, image_feature=np.array([1.0, 0.0])),),
        (target(2, prototype=np.array([1.0, 0.0])),),
        config(),
        dormant_reid=reid_config(),
    )
    diagnostic = result.assignment_diagnostics[0]
    assert diagnostic.target_lifecycle is TemporalLifecycle.ACTIVE
    assert diagnostic.high_confidence_identity_match is True
    with pytest.raises(FrozenInstanceError):
        diagnostic.score = 0.0  # type: ignore[misc]
    with pytest.raises((TypeError, ValueError)):
        replace(diagnostic, semantic_qualified=np.bool_(True))
    with pytest.raises((TypeError, ValueError)):
        replace(diagnostic, observation_id=np.int64(1))
    with pytest.raises(ValueError, match="provenance"):
        replace(diagnostic, feature_model_id=None)
    with pytest.raises(TypeError, match="assignments"):
        replace(result, assignments=list(result.assignments))
    with pytest.raises(ValueError, match="sorted"):
        replace(result, assignment_diagnostics=(replace(diagnostic, observation_id=2), diagnostic))


def test_active_diagnostic_extracts_current_evidence_without_reid_mode() -> None:
    result = associate_temporal_observations(
        (observation(1, image_feature=np.array([1.0, 0.0])),),
        (target(2, prototype=np.array([1.0, 0.0])),),
        config(),
        dormant_reid=None,
    )
    diagnostic = result.assignment_diagnostics[0]
    assert diagnostic.appearance_similarity == pytest.approx(1.0)
    assert diagnostic.feature_model_id == "clip"
    assert diagnostic.feature_model_match is True
    assert diagnostic.semantic_qualified is True
    assert diagnostic.target_lifecycle is TemporalLifecycle.ACTIVE
    assert 0.0 <= diagnostic.score <= 1.0
    assert diagnostic.high_confidence_identity_match is False


def test_moved_dormant_reuses_original_entity_id() -> None:
    obs = observation(2, centroid=(5.0, 0.0, 0.0), image_feature=np.array([1.0, 0.0]))
    dormant = target(
        7,
        lifecycle=TemporalLifecycle.DORMANT,
        centroid=(0.0, 0.0, 0.0),
        predicted=(5.0, 0.0, 0.0),
        prototype=np.array([1.0, 0.0]),
    )

    result = associate_temporal_observations(
        (obs,), (dormant,), config(minimum_score=0.6), dormant_reid=reid_config()
    )

    assert result.assignments == ((2, 7),)
    assert result.unmatched_observation_ids == ()
    assert result.unmatched_entity_ids == ()
    assert (result.reid_opportunity_count, result.reid_trigger_count) == (1, 1)
    assert result.assignment_diagnostics[0].high_confidence_identity_match is True


def test_dormant_requires_appearance_semantic_qualification_before_wide_gate() -> None:
    far = observation(1, centroid=(5.0, 0.0, 0.0), semantic_id=2, image_feature=np.array([1.0, 0.0]))
    weak_similarity = target(
        1, lifecycle=TemporalLifecycle.DORMANT, prototype=np.array([0.0, 1.0]), semantics=((1, 1.0),)
    )
    wrong_model = target(
        2, lifecycle=TemporalLifecycle.DORMANT, centroid=(5.0, 0.0, 0.0),
        prototype=np.array([1.0, 0.0]), feature_model_id="other", semantics=((2, 1.0),)
    )
    result = associate_temporal_observations(
        (far,), (weak_similarity, wrong_model), config(maximum_centroid_distance_m=1.0),
        dormant_reid=reid_config(maximum_reid_distance_m=10.0),
    )
    assert result.assignments == ()
    assert (result.reid_opportunity_count, result.reid_trigger_count) == (0, 0)


def test_dormant_semantic_conflict_uses_strong_appearance_before_wide_gate() -> None:
    obs = observation(
        1, centroid=(5.0, 0.0, 0.0), semantic_id=2,
        image_feature=np.array([1.0, 0.0]),
    )
    dormant = target(
        7, lifecycle=TemporalLifecycle.DORMANT, centroid=(0.0, 0.0, 0.0),
        predicted=(100.0, 0.0, 0.0), prototype=np.array([1.0, 0.0]),
        semantics=((1, 0.9), (2, 0.1)),
    )
    result = associate_temporal_observations(
        (obs,), (dormant,),
        config(maximum_centroid_distance_m=1.0, conflict_override_visual=0.9),
        dormant_reid=reid_config(maximum_reid_distance_m=6.0),
    )
    assert result.assignments == ((1, 7),)
    assert result.assignment_diagnostics[0].semantic_qualified is True


def test_dormant_identity_selection_ignores_motion_weight_and_predicted_centroid() -> None:
    obs = observation(1, centroid=(1.0, 0.0, 0.0), image_feature=np.array([1.0, 0.0]))
    left = target(
        7, lifecycle=TemporalLifecycle.DORMANT, centroid=(0.9, 0.0, 0.0),
        predicted=(100.0, 0.0, 0.0), prototype=np.array([1.0, 0.0]),
    )
    right = target(
        8, lifecycle=TemporalLifecycle.DORMANT, centroid=(1.5, 0.0, 0.0),
        predicted=(1.0, 0.0, 0.0), prototype=np.array([1.0, 0.0]),
    )
    for motion_weight in (0.0, 1000.0):
        result = associate_temporal_observations(
            (obs,), (left, right), config(motion_weight=motion_weight),
            dormant_reid=reid_config(),
        )
        assert result.assignments == ((1, 7),)
    motion_only = config(
        visual_weight=0.0,
        semantic_weight=0.0,
        size_weight=0.0,
        geometry_weight=0.0,
        motion_weight=1.0,
    )
    assert associate_temporal_observations(
        (obs,), (left, right), motion_only, dormant_reid=reid_config()
    ).assignments == ((1, 7),)


def test_a4_dormant_ties_and_permutations_are_canonical() -> None:
    observations = (
        observation(2, image_feature=np.array([1.0, 0.0])),
        observation(1, image_feature=np.array([1.0, 0.0])),
    )
    targets = (
        target(8, lifecycle=TemporalLifecycle.DORMANT, prototype=np.array([1.0, 0.0])),
        target(7, lifecycle=TemporalLifecycle.DORMANT, prototype=np.array([1.0, 0.0])),
    )
    first = associate_temporal_observations(
        observations, targets, config(), dormant_reid=reid_config()
    )
    expected = first
    for observation_order in permutations(observations):
        for target_order in permutations(targets):
            assert associate_temporal_observations(
                observation_order, target_order, config(), dormant_reid=reid_config()
            ) == expected


@pytest.mark.parametrize(
    ("prototype", "centroid", "matched"),
    [
        (np.array([1.0, 0.0]), (0.2, 0.0, 0.0), True),
        (None, (0.2, 0.0, 0.0), False),
        (np.array([1.0, 0.0]), (0.5, 0.0, 0.0), False),
    ],
)
def test_semantic_conflict_requires_both_visual_and_geometry_override(
    prototype: np.ndarray | None,
    centroid: tuple[float, float, float],
    matched: bool,
) -> None:
    obs = observation(1, semantic_id=2, image_feature=np.array([1.0, 0.0]))
    item = target(
        5,
        centroid=centroid,
        predicted=(0.0, 0.0, 0.0),
        prototype=prototype,
        semantics=((1, 0.9), (2, 0.1)),
    )

    result = associate_temporal_observations(
        (obs,),
        (item,),
        config(
            minimum_score=0.0,
            maximum_centroid_distance_m=1.0,
            conflict_override_visual=0.9,
            conflict_override_geometry=0.8,
        ),
    )

    assert bool(result.assignments) is matched


def test_one_entity_cannot_be_claimed_by_two_observations() -> None:
    result = associate_temporal_observations(
        (observation(2), observation(1)),
        (target(8),),
        config(),
    )

    assert result.assignments == ((1, 8),)
    assert result.unmatched_observation_ids == (2,)


def test_rejected_candidate_stays_unmatched_without_mutating_target() -> None:
    prototype = np.array([3.0, 4.0])
    item = target(9, prototype=prototype)
    original = item.image_prototype.copy()

    result = associate_temporal_observations(
        (observation(4, centroid=(10.0, 0.0, 0.0)),),
        (item,),
        config(),
    )

    assert result == TemporalAssociationResult((), (4,), (9,))
    assert np.array_equal(item.image_prototype, original)
    assert prototype.tolist() == [3.0, 4.0]


def test_exact_score_ties_use_entity_id_then_observation_id() -> None:
    result = associate_temporal_observations(
        (observation(12), observation(10), observation(11)),
        (target(21), target(20)),
        config(),
    )

    assert result.assignments == ((10, 20), (11, 21))
    assert result.unmatched_observation_ids == (12,)


def test_input_permutations_produce_identical_result() -> None:
    observations = (observation(2, centroid=(1.0, 0.0, 0.0)), observation(1))
    targets = (target(11, centroid=(1.0, 0.0, 0.0)), target(10))
    expected = associate_temporal_observations(observations, targets, config())

    for observation_order in permutations(observations):
        for target_order in permutations(targets):
            assert associate_temporal_observations(
                observation_order, target_order, config()
            ) == expected


def test_unavailable_visual_and_semantic_terms_are_removed_not_scored_zero() -> None:
    obs = observation(1, semantic_id=0, image_feature=np.array([1.0, 0.0]))
    item = target(2, prototype=np.array([1.0, 0.0, 0.0]), semantics=())

    result = associate_temporal_observations(
        (obs,),
        (item,),
        config(
            visual_weight=100.0,
            semantic_weight=100.0,
            size_weight=1.0,
            motion_weight=1.0,
            geometry_weight=1.0,
            minimum_score=0.9,
        ),
    )

    assert result.assignments == ((1, 2),)


def test_target_normalizes_copies_and_freezes_image_prototype() -> None:
    source = np.array([3.0, 4.0])
    item = target(1, prototype=source)
    source[0] = 100.0

    assert item.image_prototype is not None
    assert item.image_prototype == pytest.approx(np.array([0.6, 0.8]))
    assert not item.image_prototype.flags.writeable
    with pytest.raises(ValueError):
        item.image_prototype[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        item.entity_id = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"entity_id": -1},
        {"entity_id": True},
        {"lifecycle": "active"},
        {"centroid_xyz": (0.0, np.nan, 0.0)},
        {"predicted_centroid_xyz": (0.0, np.inf, 0.0)},
        {"extent_xyz": (1.0, 0.0, 1.0)},
        {"image_prototype": np.array([0.0, 0.0])},
        {"image_prototype": np.array([[1.0]])},
        {"semantic_probabilities": ((2, 0.5), (1, 0.5))},
        {"semantic_probabilities": ((1, 0.5), (1, 0.5))},
        {"semantic_probabilities": ((0, 1.0),)},
        {"semantic_probabilities": ((1, np.nan),)},
        {"semantic_probabilities": ((1, 0.4),)},
    ],
)
def test_target_rejects_invalid_fields(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(target(1), **changes)


@pytest.mark.parametrize(
    "bad_config",
    [
        object(),
        config(visual_weight=True),
        config(size_weight=-1.0),
        config(motion_weight=np.inf),
        config(minimum_score=1.1),
        config(maximum_centroid_distance_m=0.0),
        config(semantic_conflict_probability=np.nan),
        config(conflict_override_visual=-0.1),
        config(conflict_override_geometry=1.1),
        config(
            visual_weight=0.0,
            semantic_weight=0.0,
            size_weight=0.0,
            motion_weight=0.0,
            geometry_weight=0.0,
        ),
    ],
)
def test_invalid_manually_constructed_config_fails_closed(bad_config: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        associate_temporal_observations(
            (observation(1),),
            (target(1),),
            bad_config,  # type: ignore[arg-type]
        )


def test_inputs_require_exact_tuples_unique_ids_and_object_observations() -> None:
    with pytest.raises(TypeError, match="tuple"):
        associate_temporal_observations([observation(1)], (), config())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple"):
        associate_temporal_observations((), [target(1)], config())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FrameObservation"):
        associate_temporal_observations((object(),), (), config())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TemporalAssociationTarget"):
        associate_temporal_observations((), (object(),), config())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        associate_temporal_observations((observation(1), observation(1)), (), config())
    with pytest.raises(ValueError, match="unique"):
        associate_temporal_observations((), (target(1), target(1)), config())
    with pytest.raises(ValueError, match="OBJECT"):
        associate_temporal_observations(
            (observation(1, kind=ObservationKind.STRUCTURE),), (), config()
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"centroid_xyz": (np.nan, 0.0, 0.0)},
        {"bounds_max_xyz": (np.inf, 0.5, 0.5)},
        {"bounds_max_xyz": (-0.5, 0.5, 0.5)},
    ],
)
def test_invalid_observation_geometry_fails_closed(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        associate_temporal_observations(
            (replace(observation(1), **changes),),
            (target(1),),
            config(),
        )


def test_empty_inputs_return_complete_sorted_unmatched_sets() -> None:
    assert associate_temporal_observations((), (), config()) == TemporalAssociationResult(
        (), (), ()
    )
    assert associate_temporal_observations(
        (), (target(3), target(1)), config()
    ) == TemporalAssociationResult((), (), (1, 3))
    assert associate_temporal_observations(
        (observation(3), observation(1)), (), config()
    ) == TemporalAssociationResult((), (1, 3), ())


def test_nonempty_result_requires_complete_diagnostics_and_signed_int64_ids() -> None:
    with pytest.raises(ValueError, match="diagnostics"):
        TemporalAssociationResult(((1, 2),), (), ())
    with pytest.raises((TypeError, ValueError)):
        TemporalAssociationResult(((1, 2**63),), (), (), assignment_diagnostics=())
