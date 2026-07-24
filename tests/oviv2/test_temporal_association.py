from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import numpy as np
import pytest

from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.temporal_association import (
    TemporalAssociationResult,
    TemporalAssociationTarget,
    associate_temporal_observations,
)
from src.oviv2.temporal_config import TemporalAssociationConfig
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


def test_active_and_uncertain_stage_precedes_better_dormant_candidate() -> None:
    obs = observation(1, image_feature=np.array([1.0, 0.0]))
    active = target(20, centroid=(0.6, 0.0, 0.0), prototype=np.array([1.0, 0.0]))
    dormant = target(
        10,
        lifecycle=TemporalLifecycle.DORMANT,
        prototype=np.array([1.0, 0.0]),
    )

    result = associate_temporal_observations((obs,), (dormant, active), config())

    assert result.assignments == ((1, 20),)
    assert result.unmatched_entity_ids == (10,)


def test_moved_dormant_reuses_original_entity_id() -> None:
    obs = observation(2, centroid=(5.0, 0.0, 0.0), image_feature=np.array([1.0, 0.0]))
    dormant = target(
        7,
        lifecycle=TemporalLifecycle.DORMANT,
        centroid=(0.0, 0.0, 0.0),
        predicted=(5.0, 0.0, 0.0),
        prototype=np.array([1.0, 0.0]),
    )

    result = associate_temporal_observations((obs,), (dormant,), config(minimum_score=0.6))

    assert result == TemporalAssociationResult(((2, 7),), (), ())


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
    expected = TemporalAssociationResult(((1, 10), (2, 11)), (), ())

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
