from __future__ import annotations

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from src.oviv2.association import (
    Assignment,
    AssociationConfig,
    AssociationTarget,
    CandidateScore,
    directed_voxel_overlap,
    expanded_bounds_iou,
    score_candidate,
    solve_assignment,
)


def target(
    target_id: int,
    keys: set[tuple[int, int, int]],
    *,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    centroid: tuple[float, float, float] | None = None,
    semantic_id: int = 2,
    semantic_confidence: float = 0.9,
    visual_feature: tuple[float, ...] | None = None,
    feature_model_id: str | None = None,
    free_space_conflict: bool = False,
) -> AssociationTarget:
    points = np.asarray(sorted(keys), dtype=np.float64) * 0.05
    bounds_min, bounds_max = (
        (tuple(points.min(axis=0)), tuple(points.max(axis=0) + 0.05))
        if bounds is None
        else bounds
    )
    return AssociationTarget(
        target_id=target_id,
        voxel_keys=frozenset(keys),
        centroid_xyz=tuple(points.mean(axis=0)) if centroid is None else centroid,
        bounds_min_xyz=tuple(float(value) for value in bounds_min),
        bounds_max_xyz=tuple(float(value) for value in bounds_max),
        semantic_id=semantic_id,
        semantic_confidence=semantic_confidence,
        visual_feature=visual_feature,
        feature_model_id=feature_model_id,
        free_space_conflict=free_space_conflict,
    )


def test_centroid_distance_without_spatial_support_cannot_match() -> None:
    config = AssociationConfig(bounds_expansion_m=0.0, max_centroid_distance_m=1.0)
    left = target(1, {(0, 0, 0)}, bounds=((0, 0, 0), (0.1, 0.1, 0.1)))
    right = target(2, {(2, 0, 0)}, bounds=((0.2, 0, 0), (0.3, 0.1, 0.1)))

    assert score_candidate(left, right, config) is None


def test_high_confidence_semantic_conflict_blocks_adjacent_merge() -> None:
    config = AssociationConfig(semantic_conflict_confidence=0.75)
    chair = target(2, {(0, 0, 0)}, semantic_id=2, semantic_confidence=0.9)
    table = target(3, {(0, 0, 0)}, semantic_id=3, semantic_confidence=0.9)

    result = score_candidate(chair, table, config)

    assert result is not None
    assert result.conflict is True
    assert result.accepted is False


def test_hungarian_assignment_is_one_to_one_and_deterministic() -> None:
    assignments = solve_assignment(
        (target(10, {(0, 0, 0)}), target(11, {(5, 0, 0)})),
        (target(20, {(0, 0, 0)}), target(21, {(5, 0, 0)})),
        AssociationConfig(),
    )

    assert [(item.left_id, item.right_id) for item in assignments] == [
        (10, 20),
        (11, 21),
    ]


def test_directed_voxel_overlap_preserves_both_directions() -> None:
    larger = frozenset({(0, 0, 0), (1, 0, 0)})
    smaller = frozenset({(0, 0, 0)})

    assert directed_voxel_overlap(larger, smaller) == 0.5
    assert directed_voxel_overlap(smaller, larger) == 1.0
    assert directed_voxel_overlap(frozenset(), smaller) == 0.0


def test_expanded_bounds_create_spatial_support() -> None:
    left = target(1, {(0, 0, 0)})
    right = target(2, {(2, 0, 0)})

    assert expanded_bounds_iou(left, right, 0.0) == 0.0
    assert expanded_bounds_iou(left, right, 0.03) > 0.0
    assert score_candidate(
        left,
        right,
        AssociationConfig(bounds_expansion_m=0.03),
    ) is not None


@pytest.mark.parametrize(
    ("left_bounds", "right_bounds", "expected"),
    [
        (
            ((0.0, 0.0, 0.0), (1e308, 1e308, 1e308)),
            ((5e307, 5e307, 5e307), (1e308, 1e308, 1e308)),
            0.125,
        ),
        (
            ((-1e308, -1e308, -1e308), (-5e307, -5e307, -5e307)),
            ((5e307, 5e307, 5e307), (1e308, 1e308, 1e308)),
            0.0,
        ),
    ],
)
def test_expanded_bounds_iou_is_stable_for_extreme_finite_coordinates(
    left_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    right_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    expected: float,
) -> None:
    left = target(1, {(0, 0, 0)}, bounds=left_bounds)
    right = target(2, {(1, 0, 0)}, bounds=right_bounds)

    with np.errstate(over="raise", invalid="raise"):
        result = expanded_bounds_iou(left, right, 0.0)

    assert result == pytest.approx(expected)


def test_distance_gate_rejects_bounds_only_candidate_beyond_limit() -> None:
    shared_bounds = ((0.0, 0.0, 0.0), (0.1, 0.1, 0.1))
    left = target(1, {(0, 0, 0)}, bounds=shared_bounds, centroid=(0.0, 0.0, 0.0))
    right = target(2, {(1, 0, 0)}, bounds=shared_bounds, centroid=(2.0, 0.0, 0.0))

    assert score_candidate(
        left,
        right,
        AssociationConfig(max_centroid_distance_m=0.5),
    ) is None


def test_overlap_candidate_survives_distance_gate_with_zero_geometry() -> None:
    left = target(1, {(0, 0, 0)}, centroid=(0.0, 0.0, 0.0))
    right = target(2, {(0, 0, 0)}, centroid=(2.0, 0.0, 0.0))

    result = score_candidate(
        left,
        right,
        AssociationConfig(max_centroid_distance_m=0.5),
    )

    assert result is not None
    assert result.accepted is True


def test_extreme_centroid_distance_is_infinite_without_overflow() -> None:
    shared_bounds = ((0.0, 0.0, 0.0), (0.05, 0.05, 0.05))
    overlap_left = target(1, {(0, 0, 0)}, centroid=(1e308, 0.0, 0.0))
    overlap_right = target(2, {(0, 0, 0)}, centroid=(-1e308, 0.0, 0.0))
    bounds_left = target(
        3,
        {(0, 0, 0)},
        bounds=shared_bounds,
        centroid=(1e308, 0.0, 0.0),
    )
    bounds_right = target(
        4,
        {(1, 0, 0)},
        bounds=shared_bounds,
        centroid=(-1e308, 0.0, 0.0),
    )

    with np.errstate(over="raise", invalid="raise"):
        overlap_result = score_candidate(overlap_left, overlap_right, AssociationConfig())
        bounds_result = score_candidate(bounds_left, bounds_right, AssociationConfig())

    assert overlap_result is not None
    assert overlap_result.accepted is True
    assert bounds_result is None


@pytest.mark.parametrize(
    ("right_feature", "right_model"),
    [
        ((1.0, 0.0), "other"),
        ((1.0, 0.0, 0.0), "clip"),
    ],
)
def test_incompatible_visual_features_are_unavailable_and_weights_renormalize(
    right_feature: tuple[float, ...],
    right_model: str,
) -> None:
    left = target(
        1,
        {(0, 0, 0)},
        visual_feature=(1.0, 0.0),
        feature_model_id="clip",
    )
    right = target(
        2,
        {(0, 0, 0)},
        visual_feature=right_feature,
        feature_model_id=right_model,
    )

    result = score_candidate(left, right, AssociationConfig())

    assert result is not None
    assert result.visual_cosine is None
    assert result.score == pytest.approx(1.0)


def test_compatible_visual_cosine_contributes_to_score() -> None:
    config = AssociationConfig()
    left = target(
        1,
        {(0, 0, 0)},
        semantic_id=2,
        semantic_confidence=0.1,
        visual_feature=(1.0, 0.0),
        feature_model_id="clip",
    )
    aligned = target(
        2,
        {(0, 0, 0)},
        semantic_id=3,
        semantic_confidence=0.1,
        visual_feature=(2.0, 0.0),
        feature_model_id="clip",
    )
    orthogonal = replace(aligned, target_id=3, visual_feature=(0.0, 3.0))

    aligned_score = score_candidate(left, aligned, config)
    orthogonal_score = score_candidate(left, orthogonal, config)

    assert aligned_score is not None and orthogonal_score is not None
    assert aligned_score.visual_cosine == pytest.approx(1.0)
    assert orthogonal_score.visual_cosine == pytest.approx(0.0)
    assert aligned_score.score > orthogonal_score.score


def test_compatible_visual_feature_can_override_semantic_conflict() -> None:
    left = target(
        1,
        {(0, 0, 0)},
        semantic_id=2,
        visual_feature=(1.0, 0.0),
        feature_model_id="clip",
    )
    right = target(
        2,
        {(0, 0, 0)},
        semantic_id=3,
        visual_feature=(0.9, np.sqrt(0.19)),
        feature_model_id="clip",
    )

    result = score_candidate(left, right, AssociationConfig())

    assert result is not None
    assert result.visual_cosine == pytest.approx(0.9)
    assert result.conflict is False
    assert result.accepted is True


def test_visual_feature_below_override_does_not_clear_semantic_conflict() -> None:
    left = target(
        1,
        {(0, 0, 0)},
        semantic_id=2,
        visual_feature=(1.0, 0.0),
        feature_model_id="clip",
    )
    right = target(
        2,
        {(0, 0, 0)},
        semantic_id=3,
        visual_feature=(0.79, np.sqrt(1.0 - 0.79**2)),
        feature_model_id="clip",
    )

    result = score_candidate(left, right, AssociationConfig())

    assert result is not None
    assert result.conflict is True
    assert result.accepted is False


def test_semantic_conflict_threshold_is_inclusive() -> None:
    config = AssociationConfig(semantic_conflict_confidence=0.75)
    left = target(1, {(0, 0, 0)}, semantic_id=2, semantic_confidence=0.75)
    right = target(2, {(0, 0, 0)}, semantic_id=3, semantic_confidence=0.75)

    result = score_candidate(left, right, config)

    assert result is not None
    assert result.conflict is True
    assert result.accepted is False


def test_visual_override_threshold_is_inclusive() -> None:
    config = AssociationConfig(semantic_conflict_visual_override=0.8)
    left = target(
        1,
        {(0, 0, 0)},
        semantic_id=2,
        visual_feature=(1.0, 0.0),
        feature_model_id="clip",
    )
    right = target(
        2,
        {(0, 0, 0)},
        semantic_id=3,
        visual_feature=(0.8, 0.6),
        feature_model_id="clip",
    )

    result = score_candidate(left, right, config)

    assert result is not None
    assert result.visual_cosine == pytest.approx(0.8)
    assert result.conflict is False
    assert result.accepted is True


def test_free_space_conflict_is_always_rejected() -> None:
    result = score_candidate(
        target(1, {(0, 0, 0)}, free_space_conflict=True),
        target(2, {(0, 0, 0)}),
        AssociationConfig(),
    )

    assert result is not None
    assert result.conflict is True
    assert result.accepted is False


def test_minimum_score_rejects_weak_candidate() -> None:
    config = AssociationConfig(minimum_score=0.9)
    left = target(1, {(0, 0, 0)}, semantic_id=2, semantic_confidence=0.1)
    right = target(2, {(0, 0, 0)}, semantic_id=3, semantic_confidence=0.1)

    result = score_candidate(left, right, config)

    assert result is not None
    assert result.conflict is False
    assert result.score < config.minimum_score
    assert result.accepted is False
    assert solve_assignment((left,), (right,), config) == ()


def test_extreme_component_weights_produce_finite_renormalized_score() -> None:
    config = AssociationConfig(
        geometry_weight=1e308,
        overlap_weight=1e308,
        visual_weight=1e308,
        semantic_weight=1e308,
        temporal_weight=1e308,
    )
    left = target(1, {(0, 0, 0)}, semantic_id=2, semantic_confidence=0.1)
    right = target(2, {(0, 0, 0)}, semantic_id=3, semantic_confidence=0.1)

    result = score_candidate(left, right, config)

    assert result is not None
    assert np.isfinite(result.score)
    assert result.score == pytest.approx(0.75)


def test_assignment_handles_empty_inputs() -> None:
    item = target(1, {(0, 0, 0)})

    assert solve_assignment((), (item,), AssociationConfig()) == ()
    assert solve_assignment((item,), (), AssociationConfig()) == ()


def test_invalid_hungarian_pairs_are_not_returned() -> None:
    assignments = solve_assignment(
        (target(10, {(0, 0, 0)}), target(11, {(10, 0, 0)})),
        (target(20, {(0, 0, 0)}), target(21, {(20, 0, 0)})),
        AssociationConfig(bounds_expansion_m=0.0),
    )

    assert [(item.left_id, item.right_id) for item in assignments] == [(10, 20)]


def test_assignment_preserves_tiny_real_score_advantage_for_all_permutations() -> None:
    angle = 8.0e-7
    axis = (1.0, 0.0)
    rotated = (float(np.cos(angle)), float(np.sin(angle)))
    left = (
        target(11, {(0, 0, 0)}, visual_feature=rotated, feature_model_id="clip"),
        target(10, {(0, 0, 0)}, visual_feature=axis, feature_model_id="clip"),
    )
    right = (
        target(20, {(0, 0, 0)}, visual_feature=rotated, feature_model_id="clip"),
        target(21, {(0, 0, 0)}, visual_feature=axis, feature_model_id="clip"),
    )
    expected = [(10, 21), (11, 20)]

    for left_values in permutations(left):
        for right_values in permutations(right):
            assignments = solve_assignment(
                left_values,
                right_values,
                AssociationConfig(),
            )
            assert [(item.left_id, item.right_id) for item in assignments] == expected


def test_local_equal_score_tie_uses_sorted_ids() -> None:
    assignments = solve_assignment(
        (target(11, {(10, 0, 0)}), target(10, {(0, 0, 0)})),
        (
            target(21, {(0, 0, 0)}),
            target(22, {(10, 0, 0)}),
            target(20, {(0, 0, 0)}),
        ),
        AssociationConfig(bounds_expansion_m=0.0),
    )

    assert [(item.left_id, item.right_id) for item in assignments] == [
        (10, 20),
        (11, 22),
    ]


def test_fully_tied_rectangular_assignments_prefer_nearby_sorted_ranks() -> None:
    left = tuple(target(target_id, {(0, 0, 0)}) for target_id in (12, 10, 11))
    right = tuple(target(target_id, {(0, 0, 0)}) for target_id in (21, 22, 20))

    wide = solve_assignment(left[:2], right, AssociationConfig())
    tall = solve_assignment(left, right[:2], AssociationConfig())

    assert [(item.left_id, item.right_id) for item in wide] == [(10, 20), (12, 21)]
    assert [(item.left_id, item.right_id) for item in tall] == [(10, 21), (11, 22)]


def test_fully_invalid_assignment_returns_empty() -> None:
    assignments = solve_assignment(
        (target(10, {(0, 0, 0)}), target(11, {(1, 0, 0)})),
        (target(20, {(10, 0, 0)}), target(21, {(11, 0, 0)})),
        AssociationConfig(bounds_expansion_m=0.0),
    )

    assert assignments == ()


def test_equal_score_assignment_is_stable_for_all_input_permutations() -> None:
    left = (
        target(12, {(0, 0, 0)}),
        target(10, {(0, 0, 0)}),
        target(11, {(0, 0, 0)}),
    )
    right = (
        target(21, {(0, 0, 0)}),
        target(22, {(0, 0, 0)}),
        target(20, {(0, 0, 0)}),
    )
    expected = [(10, 20), (11, 21), (12, 22)]

    for left_values in permutations(left):
        for right_values in permutations(right):
            assignments = solve_assignment(
                left_values,
                right_values,
                AssociationConfig(),
            )
            assert [(item.left_id, item.right_id) for item in assignments] == expected


@pytest.mark.parametrize("side", ["left", "right"])
def test_assignment_rejects_duplicate_target_ids(side: str) -> None:
    duplicate = (target(1, {(0, 0, 0)}), target(1, {(1, 0, 0)}))
    single = (target(2, {(0, 0, 0)}),)

    with pytest.raises(ValueError, match="unique"):
        solve_assignment(
            duplicate if side == "left" else single,
            single if side == "left" else duplicate,
            AssociationConfig(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_id", -1),
        ("target_id", True),
        ("semantic_id", -1),
        ("semantic_id", True),
        ("semantic_confidence", -0.1),
        ("semantic_confidence", 1.1),
        ("semantic_confidence", np.nan),
        ("free_space_conflict", 1),
    ],
)
def test_target_rejects_invalid_scalar_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        replace(target(1, {(0, 0, 0)}), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("centroid_xyz", (0.0, 0.0)),
        ("centroid_xyz", (0.0, np.inf, 0.0)),
        ("bounds_min_xyz", (0.0, 0.0)),
        ("bounds_max_xyz", (0.0, 0.0, np.nan)),
        ("bounds_min_xyz", (0.1, 0.0, 0.0)),
    ],
)
def test_target_rejects_invalid_geometry(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field if field != "bounds_min_xyz" else "bounds"):
        replace(target(1, {(0, 0, 0)}), **{field: value})


@pytest.mark.parametrize(
    "voxel_keys",
    [
        frozenset({(0, 0)}),
        frozenset({(0.5, 0, 0)}),
        frozenset({(True, 0, 0)}),
    ],
)
def test_target_rejects_invalid_voxel_keys(voxel_keys: frozenset[tuple]) -> None:
    with pytest.raises(ValueError, match="voxel_keys"):
        replace(target(1, {(0, 0, 0)}), voxel_keys=voxel_keys)


@pytest.mark.parametrize(
    ("feature", "model_id"),
    [
        ((), "clip"),
        ((0.0, 0.0), "clip"),
        ((1.0, np.nan), "clip"),
        ((1.0, 0.0), None),
        ((1.0, 0.0), ""),
        ((1.0, 0.0), "   "),
    ],
)
def test_target_rejects_invalid_visual_features(
    feature: tuple[float, ...],
    model_id: str | None,
) -> None:
    with pytest.raises(ValueError, match="visual_feature|feature_model_id"):
        replace(
            target(1, {(0, 0, 0)}),
            visual_feature=feature,
            feature_model_id=model_id,
        )


def test_target_normalizes_visual_feature_to_stable_tuple() -> None:
    item = target(
        1,
        {(0, 0, 0)},
        visual_feature=(3.0, 4.0),
        feature_model_id="clip",
    )

    assert item.visual_feature == pytest.approx((0.6, 0.8))
    assert isinstance(item.visual_feature, tuple)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_directed_overlap", -0.1),
        ("min_directed_overlap", 1.1),
        ("minimum_score", np.nan),
        ("semantic_conflict_confidence", 1.1),
        ("semantic_conflict_visual_override", -0.1),
        ("bounds_expansion_m", -0.1),
        ("bounds_expansion_m", np.inf),
        ("max_centroid_distance_m", 0.0),
        ("max_centroid_distance_m", np.inf),
        ("geometry_weight", -0.1),
        ("overlap_weight", np.nan),
        ("visual_weight", np.inf),
        ("semantic_weight", -0.1),
        ("temporal_weight", np.nan),
    ],
)
def test_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        replace(AssociationConfig(), **{field: value})


def test_config_requires_a_non_visual_component_weight() -> None:
    with pytest.raises(ValueError, match="non-visual"):
        AssociationConfig(
            geometry_weight=0.0,
            overlap_weight=0.0,
            visual_weight=1.0,
            semantic_weight=0.0,
            temporal_weight=0.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("left_id", -1),
        ("left_id", True),
        ("right_id", -1),
        ("score", np.nan),
        ("score", 1.1),
        ("directed_overlap", -0.1),
        ("bounds_iou", 1.1),
        ("visual_cosine", -1.1),
        ("conflict", 1),
        ("accepted", 0),
    ],
)
def test_candidate_score_rejects_invalid_fields(field: str, value: object) -> None:
    valid = CandidateScore(1, 2, 0.5, 0.25, 0.1, None, False, True)

    with pytest.raises(ValueError, match=field):
        replace(valid, **{field: value})


def test_candidate_score_cannot_accept_a_conflict() -> None:
    with pytest.raises(ValueError, match="accepted"):
        CandidateScore(1, 2, 0.5, 0.25, 0.1, None, True, True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("left_id", -1),
        ("left_id", True),
        ("right_id", -1),
        ("score", np.inf),
        ("score", -0.1),
        ("score", 1.1),
    ],
)
def test_assignment_rejects_invalid_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        replace(Assignment(1, 2, 0.5), **{field: value})
