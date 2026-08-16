from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from src.oviv2.observations import FrameObservation, ObservationKind
import src.oviv2.temporal_observation_merge as observation_merge
from src.oviv2.temporal_observation_merge import (
    TemporalObservationMergeConfig,
    merge_temporal_object_observations,
    regularize_temporal_object_extents,
    suppress_near_duplicate_primary_observations,
)
from src.oviv2.tracking import LocalTracker, LocalTrackerConfig


def _mask(*pixels: tuple[int, int], shape: tuple[int, int] = (6, 8)) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for row, column in pixels:
        result[row, column] = True
    return result


def _observation(
    observation_id: int,
    mask: np.ndarray,
    *,
    semantic_id: int = 2,
    kind: ObservationKind = ObservationKind.OBJECT,
    frame_id: int = 7,
    timestamp: float = 1.25,
) -> FrameObservation:
    rows, columns = np.nonzero(mask)
    return FrameObservation(
        observation_id=observation_id,
        frame_id=frame_id,
        timestamp=timestamp,
        kind=kind,
        label="chair",
        semantic_id=semantic_id,
        confidence=0.9,
        mask=mask,
        bbox_xyxy=(
            float(columns.min()),
            float(rows.min()),
            float(columns.max() + 1),
            float(rows.max() + 1),
        ),
        voxel_keys=frozenset({(observation_id + 1, 0, 20)}),
        centroid_xyz=(0.0, 0.0, 1.0),
        bounds_min_xyz=(0.0, 0.0, 1.0),
        bounds_max_xyz=(0.0, 0.0, 1.0),
    )


def _ids(observations: tuple[FrameObservation, ...]) -> tuple[int, ...]:
    return tuple(item.observation_id for item in observations)


def test_regularizes_degenerate_object_bounds_to_occupied_voxel_cells() -> None:
    feature = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
    observation = replace(
        _observation(1, _mask((1, 1))),
        semantic_id=0,
        label="unknown",
        voxel_keys=frozenset({(-2, 4, 20), (-2, 4, 40)}),
        bounds_min_xyz=(-0.1, 0.2, 1.0),
        bounds_max_xyz=(-0.1, 0.2, 2.0),
        image_feature=feature,
        text_feature=feature,
        feature_model_id="test-features",
        view_direction_xyz=(0.0, 0.0, 1.0),
    )

    (regularized,) = regularize_temporal_object_extents((observation,), 0.05)

    assert regularized is not observation
    assert regularized.bounds_min_xyz == pytest.approx((-0.1, 0.2, 1.0))
    assert regularized.bounds_max_xyz == pytest.approx((-0.05, 0.25, 2.05))
    assert regularized.observation_id == observation.observation_id
    assert regularized.mask is observation.mask
    assert regularized.voxel_keys is observation.voxel_keys
    assert regularized.image_feature is observation.image_feature
    assert regularized.text_feature is observation.text_feature
    assert regularized.view_direction_xyz is observation.view_direction_xyz
    assert observation.bounds_max_xyz == (-0.1, 0.2, 2.0)


@pytest.mark.parametrize("coordinate", (1.0e15, -1.0e15))
def test_regularizer_uses_representable_bounds_for_large_voxel_coordinates(
    coordinate: float,
) -> None:
    key = int(math.floor(coordinate / 0.1))
    observation = replace(
        _observation(1, _mask((1, 1))),
        voxel_keys=frozenset({(key, key, key)}),
        centroid_xyz=(coordinate, coordinate, coordinate),
        bounds_min_xyz=(coordinate, coordinate, coordinate),
        bounds_max_xyz=(coordinate, coordinate, coordinate),
    )

    (regularized,) = regularize_temporal_object_extents((observation,), 0.1)

    assert all(
        math.isfinite(lower)
        and math.isfinite(upper)
        and lower < upper
        and lower <= coordinate <= upper
        for lower, upper in zip(
            regularized.bounds_min_xyz, regularized.bounds_max_xyz
        )
    )


def test_regularizer_keeps_common_voxel_cell_bounds_bit_exact() -> None:
    observation = replace(
        _observation(1, _mask((1, 1))),
        voxel_keys=frozenset({(1, 2, 4)}),
        centroid_xyz=(0.25, 0.5, 1.0),
        bounds_min_xyz=(0.25, 0.5, 1.0),
        bounds_max_xyz=(0.25, 0.5, 1.0),
    )

    (regularized,) = regularize_temporal_object_extents((observation,), 0.25)

    assert regularized.bounds_min_xyz == (0.25, 0.5, 1.0)
    assert regularized.bounds_max_xyz == (0.5, 0.75, 1.25)


def test_regularizer_preserves_positive_objects_and_nonobjects_exactly() -> None:
    positive = replace(
        _observation(1, _mask((1, 1))),
        bounds_max_xyz=(0.1, 0.1, 1.1),
    )
    structure = replace(
        _observation(2, _mask((2, 2))),
        kind=ObservationKind.STRUCTURE,
    )

    result = regularize_temporal_object_extents((positive, structure), 0.05)

    assert result[0] is positive
    assert result[1] is structure


@pytest.mark.parametrize(
    ("bounds_min", "bounds_max"),
    (
        ((float("nan"), 0.0, 1.0), (0.1, 0.1, 1.1)),
        ((0.2, 0.0, 1.0), (0.1, 0.1, 1.1)),
    ),
)
def test_regularizer_rejects_corrupt_object_bounds(
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
) -> None:
    observation = replace(
        _observation(1, _mask((1, 1))),
        bounds_min_xyz=bounds_min,
        bounds_max_xyz=bounds_max,
    )

    with pytest.raises(ValueError, match="finite and ordered"):
        regularize_temporal_object_extents((observation,), 0.05)


@pytest.mark.parametrize("value", (True, 0.0, float("inf")))
def test_regularizer_rejects_invalid_voxel_size(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="voxel_size_m"):
        regularize_temporal_object_extents((), value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field,value,error",
    (
        ("same_semantic_iou_threshold", True, TypeError),
        ("same_semantic_iou_threshold", "0.5", TypeError),
        ("same_semantic_iou_threshold", float("nan"), ValueError),
        ("same_semantic_iou_threshold", -0.01, ValueError),
        ("same_semantic_iou_threshold", 1.01, ValueError),
        ("supplement_containment_threshold", False, TypeError),
        ("supplement_containment_threshold", float("inf"), ValueError),
        ("supplement_containment_threshold", -0.01, ValueError),
        ("supplement_containment_threshold", 1.01, ValueError),
        ("primary_duplicate_iou_threshold", True, TypeError),
        ("primary_duplicate_iou_threshold", 0.0, ValueError),
        ("primary_duplicate_iou_threshold", float("inf"), ValueError),
        ("primary_duplicate_iou_threshold", 1.01, ValueError),
    ),
)
def test_config_strictly_validates_unit_interval(
    field: str, value: object, error: type[Exception]
) -> None:
    values: dict[str, object] = {
        "same_semantic_iou_threshold": 0.5,
        "supplement_containment_threshold": 0.8,
        "primary_duplicate_iou_threshold": 0.9,
    }
    values[field] = value

    with pytest.raises(error):
        TemporalObservationMergeConfig(**values)  # type: ignore[arg-type]


def test_config_accepts_supported_interval_boundaries() -> None:
    low = TemporalObservationMergeConfig(0, 0, 0.1)
    high = TemporalObservationMergeConfig(1, 1, 1)

    assert (
        low.same_semantic_iou_threshold,
        low.supplement_containment_threshold,
        low.primary_duplicate_iou_threshold,
    ) == (
        0.0,
        0.0,
        0.1,
    )
    assert (
        high.same_semantic_iou_threshold,
        high.supplement_containment_threshold,
        high.primary_duplicate_iou_threshold,
    ) == (
        1.0,
        1.0,
        1.0,
    )


def test_primary_duplicate_suppression_keeps_highest_confidence_across_semantics() -> None:
    overlap = _mask((1, 1), (1, 2), (2, 1), (2, 2))
    couch = replace(
        _observation(1, overlap, semantic_id=2),
        label="couch",
        confidence=0.8,
    )
    chair = replace(
        _observation(2, overlap, semantic_id=3),
        label="chair",
        confidence=0.9,
    )
    unique = _observation(3, _mask((5, 7)), semantic_id=4)

    result = suppress_near_duplicate_primary_observations(
        (couch, chair, unique),
        iou_threshold=0.9,
    )

    assert result == (chair, unique)


def test_primary_duplicate_suppression_is_stable_and_preserves_nonobjects() -> None:
    overlap = _mask((1, 1), (1, 2), (2, 1), (2, 2))
    lower_id = _observation(1, overlap, semantic_id=2)
    higher_id = _observation(2, overlap, semantic_id=3)
    structure = replace(
        _observation(3, overlap, semantic_id=2),
        kind=ObservationKind.STRUCTURE,
    )

    forward = suppress_near_duplicate_primary_observations(
        (higher_id, structure, lower_id),
        iou_threshold=1.0,
    )
    reverse = suppress_near_duplicate_primary_observations(
        (lower_id, structure, higher_id),
        iou_threshold=1.0,
    )

    assert forward == (structure, lower_id)
    assert reverse == (lower_id, structure)


def test_primary_has_priority_and_does_not_suppress_itself() -> None:
    overlap = _mask((1, 1), (1, 2), (2, 1), (2, 2))
    primary_first = _observation(1, overlap)
    primary_second = _observation(2, overlap)
    higher_confidence_supplement = replace(
        _observation(3, overlap), confidence=1.0
    )

    merged = merge_temporal_object_observations(
        (primary_first, primary_second), ((higher_confidence_supplement,),)
    )

    assert _ids(merged) == (1, 2)


def test_primary_authority_absorbs_matches_without_promoting_supplements() -> None:
    primary = _observation(1, _mask((1, 1), (1, 2)))
    matched = _observation(2, _mask((1, 1), (1, 2), (2, 1)))
    unmatched = _observation(3, _mask((4, 6)), semantic_id=3)

    result = observation_merge.merge_primary_authoritative_observations(
        (primary,), ((matched, unmatched),)
    )

    assert _ids(result) == (1,)
    assert np.array_equal(result[0].mask, primary.mask | matched.mask)
    assert result[0].voxel_keys == primary.voxel_keys | matched.voxel_keys


def test_iou_threshold_is_inclusive() -> None:
    primary = _observation(1, _mask((1, 1), (1, 2), (2, 1)))
    exactly_half_iou = _observation(2, _mask((1, 1), (1, 2), (2, 2)))

    merged = merge_temporal_object_observations(
        (primary,),
        ((exactly_half_iou,),),
        TemporalObservationMergeConfig(
            same_semantic_iou_threshold=0.5,
            supplement_containment_threshold=1.0,
        ),
    )

    assert _ids(merged) == (1,)
    assert np.array_equal(merged[0].mask, primary.mask | exactly_half_iou.mask)
    assert merged[0].voxel_keys == primary.voxel_keys | exactly_half_iou.voxel_keys


def test_candidate_containment_threshold_is_inclusive() -> None:
    primary_pixels = tuple((row, column) for row in range(4) for column in range(5))
    primary = _observation(1, _mask(*primary_pixels))
    candidate = _observation(
        2,
        _mask((0, 0), (0, 1), (1, 0), (1, 1), (5, 7)),
    )

    merged = merge_temporal_object_observations(
        (primary,),
        ((candidate,),),
        TemporalObservationMergeConfig(
            same_semantic_iou_threshold=0.5,
            supplement_containment_threshold=0.8,
        ),
    )

    assert _ids(merged) == (1,)
    assert np.array_equal(merged[0].mask, primary.mask | candidate.mask)
    assert merged[0].voxel_keys == primary.voxel_keys | candidate.voxel_keys


def test_larger_supplement_coalesces_when_it_matches_one_carrier() -> None:
    primary = _observation(1, _mask((1, 1), (1, 2)))
    candidate = _observation(
        2,
        _mask(
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
        ),
    )

    merged = merge_temporal_object_observations(
        (primary,),
        ((candidate,),),
        TemporalObservationMergeConfig(
            same_semantic_iou_threshold=0.5,
            supplement_containment_threshold=0.8,
        ),
    )

    assert _ids(merged) == (1,)
    assert np.array_equal(merged[0].mask, candidate.mask)
    assert merged[0].voxel_keys == primary.voxel_keys | candidate.voxel_keys


def test_multi_carrier_supplement_is_dropped_without_merging_instances() -> None:
    first = _observation(1, _mask((1, 1), (1, 2)))
    second = _observation(2, _mask((3, 1), (3, 2)))
    spanning = _observation(
        3,
        _mask((1, 1), (1, 2), (2, 1), (3, 1), (3, 2)),
    )

    merged = merge_temporal_object_observations(
        (first,),
        ((second,), (spanning,)),
        TemporalObservationMergeConfig(
            same_semantic_iou_threshold=0.5,
            supplement_containment_threshold=0.8,
        ),
    )

    assert merged == (first, second)


def test_unique_match_preserves_carrier_identity_features_and_motion_anchor() -> None:
    feature = np.array(
        [
            1.8959767689579035,
            0.3864994937442246,
            0.3107125446594851,
            1.8607524641720066,
            -0.022671444019016034,
            -0.30910095833473333,
            -1.4132487822916682,
        ]
    )
    carrier = replace(
        _observation(1, _mask((1, 1), (1, 2), (2, 1), (2, 2))),
        centroid_xyz=(1.0, 2.0, 3.0),
        view_direction_xyz=(-0.9217253762584194, -0.45772582566733916, 0.2201951234700494),
        image_feature=feature,
        text_feature=feature[::-1],
        feature_model_id="test-features",
    )
    supplement = replace(
        _observation(
            2,
            _mask((1, 1), (1, 2), (2, 1), (2, 2), (3, 1)),
        ),
        label="other-label",
        confidence=0.4,
        centroid_xyz=(4.0, 5.0, 6.0),
        bounds_min_xyz=(-1.0, -2.0, 0.5),
        bounds_max_xyz=(2.0, 3.0, 7.0),
    )

    (merged,) = merge_temporal_object_observations(
        (carrier,), ((supplement,),)
    )

    assert merged.observation_id == carrier.observation_id
    assert merged.label == carrier.label
    assert merged.confidence == carrier.confidence
    assert merged.centroid_xyz == carrier.centroid_xyz
    assert merged.view_direction_xyz == carrier.view_direction_xyz
    np.testing.assert_array_equal(merged.image_feature, carrier.image_feature)
    np.testing.assert_array_equal(merged.text_feature, carrier.text_feature)
    assert merged.feature_model_id == carrier.feature_model_id
    assert merged.bounds_min_xyz == (-1.0, -2.0, 0.5)
    assert merged.bounds_max_xyz == (2.0, 3.0, 7.0)
    assert merged.bbox_xyxy == (1.0, 1.0, 3.0, 4.0)
    assert merged.visible_pixel_count == 5


def test_later_group_matches_original_basis_of_coalesced_carrier() -> None:
    primary = _observation(1, _mask((1, 1), (1, 2), (2, 1), (2, 2)))
    first_supplement = _observation(
        2,
        _mask((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)),
    )
    later_supplement = _observation(
        3,
        _mask((2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)),
    )

    merged = merge_temporal_object_observations(
        (primary,), ((first_supplement,), (later_supplement,))
    )

    assert _ids(merged) == (1,)
    assert np.array_equal(
        merged[0].mask,
        primary.mask | first_supplement.mask | later_supplement.mask,
    )


def test_supplement_group_uses_group_start_match_snapshot() -> None:
    primary = _observation(1, _mask((0, 0)))
    first = _observation(2, _mask((2, 1), (2, 2), (3, 1), (3, 2)))
    second = _observation(
        3,
        _mask((2, 2), (3, 1), (3, 2), (4, 1)),
    )

    merged = merge_temporal_object_observations(
        (primary,), ((first, second),)
    )

    assert _ids(merged) == (1, 2, 3)


def test_union_mask_is_not_reused_as_a_match_basis() -> None:
    primary = _observation(
        1,
        _mask((1, 1), (1, 2), (1, 3), (1, 4), (1, 5)),
    )
    first_supplement = _observation(
        2,
        _mask((1, 2), (1, 3), (1, 4), (1, 5), (1, 6)),
    )
    bridge = _observation(
        3,
        _mask((1, 1), (1, 2), (1, 5), (1, 6)),
    )
    config = TemporalObservationMergeConfig(
        same_semantic_iou_threshold=0.6,
        supplement_containment_threshold=0.8,
    )

    merged = merge_temporal_object_observations(
        (primary,), ((first_supplement,), (bridge,)), config
    )

    assert _ids(merged) == (1, 3)


def test_coalesced_observations_form_one_track_across_frames() -> None:
    primary = replace(
        _observation(1, _mask((1, 1), (1, 2), (2, 1), (2, 2))),
        voxel_keys=frozenset({(0, 0, 20), (1, 0, 20)}),
    )
    supplement = replace(
        _observation(2, _mask((1, 1), (1, 2), (2, 1), (2, 2), (3, 1))),
        voxel_keys=frozenset({(1, 0, 20), (2, 0, 20)}),
    )
    first = merge_temporal_object_observations(
        (primary,), ((supplement,),)
    )
    second = merge_temporal_object_observations(
        (
            replace(
                primary,
                observation_id=101,
                frame_id=8,
                timestamp=1.5,
            ),
        ),
        (
            (
                replace(
                    supplement,
                    observation_id=102,
                    frame_id=8,
                    timestamp=1.5,
                ),
            ),
        ),
    )
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))

    first_batch = tracker.update(first, 7)
    second_batch = tracker.update(second, 8)

    assert len(first_batch.accepted) == 1
    assert len(second_batch.accepted) == 1
    assert second_batch.match_count == 1
    assert second_batch.new_track_count == 0
    assert len(tracker.tracks) == 1


def test_mask_overlap_is_not_lost_when_cached_bboxes_are_stale() -> None:
    overlap = _mask((1, 1), (1, 2), (2, 1), (2, 2))
    primary = replace(
        _observation(1, overlap),
        bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
    )
    candidate = replace(
        _observation(2, overlap),
        bbox_xyxy=(6.0, 4.0, 8.0, 6.0),
    )

    (merged,) = merge_temporal_object_observations((primary,), ((candidate,),))

    assert merged.observation_id == primary.observation_id
    assert merged.bbox_xyxy == (1.0, 1.0, 3.0, 3.0)
    assert merged.voxel_keys == primary.voxel_keys | candidate.voxel_keys


def test_bbox_disjoint_candidate_is_retained() -> None:
    primary = _observation(1, _mask((0, 0), (0, 1)))
    candidate = _observation(2, _mask((5, 6), (5, 7)))

    assert merge_temporal_object_observations((primary,), ((candidate,),)) == (
        primary,
        candidate,
    )


def test_earlier_supplement_suppresses_later_supplement_stably() -> None:
    primary = _observation(1, _mask((0, 0)))
    first = _observation(2, _mask((2, 2), (2, 3)))
    first_peer = _observation(3, _mask((4, 4)))
    later_duplicate = _observation(4, _mask((2, 2), (2, 3)))
    later_unique = _observation(5, _mask((5, 7)))

    merged = merge_temporal_object_observations(
        (primary,), ((first, first_peer), (later_duplicate, later_unique))
    )

    assert _ids(merged) == (1, 2, 3, 5)


def test_different_semantics_unknown_and_structure_do_not_suppress() -> None:
    overlap = _mask((1, 1), (1, 2))
    primary = _observation(1, overlap)
    different_semantic = _observation(2, overlap, semantic_id=3)
    unknown = _observation(
        3, overlap, semantic_id=0, kind=ObservationKind.UNKNOWN
    )
    structure = _observation(
        4, overlap, semantic_id=2, kind=ObservationKind.STRUCTURE
    )
    object_after_non_objects = _observation(5, overlap, semantic_id=0)

    merged = merge_temporal_object_observations(
        (primary,),
        ((different_semantic, unknown, structure), (object_after_non_objects,)),
    )

    assert _ids(merged) == (1, 2, 3, 4, 5)


def test_empty_inputs_are_supported() -> None:
    assert merge_temporal_object_observations((), ()) == ()
    assert merge_temporal_object_observations((), ((), ())) == ()


@pytest.mark.parametrize(
    "primary,supplements",
    (
        ([], ()),
        ((), []),
        ((), ([],)),
    ),
)
def test_inputs_must_be_exact_tuples(primary: object, supplements: object) -> None:
    with pytest.raises(TypeError):
        merge_temporal_object_observations(primary, supplements)  # type: ignore[arg-type]


def test_config_must_have_expected_type() -> None:
    with pytest.raises(TypeError):
        merge_temporal_object_observations((), (), object())  # type: ignore[arg-type]


def test_duplicate_ids_are_rejected_before_suppression() -> None:
    overlap = _mask((1, 1), (1, 2))
    primary = _observation(1, overlap)
    would_be_suppressed = _observation(1, overlap)

    with pytest.raises(ValueError, match="observation_id"):
        merge_temporal_object_observations(
            (primary,), ((would_be_suppressed,),)
        )


@pytest.mark.parametrize(
    "changed",
    (
        {"frame_id": 8},
        {"timestamp": 1.5},
        {"mask": _mask((0, 0), shape=(7, 8))},
    ),
)
def test_observations_must_share_frame_timestamp_and_shape(
    changed: dict[str, object],
) -> None:
    primary = _observation(1, _mask((0, 0)))
    mismatch = replace(_observation(2, _mask((1, 1))), **changed)

    with pytest.raises(ValueError):
        merge_temporal_object_observations((primary,), ((mismatch,),))


def test_observation_inputs_are_type_checked() -> None:
    with pytest.raises(TypeError):
        merge_temporal_object_observations((object(),), ())  # type: ignore[arg-type]


def test_merge_does_not_modify_inputs_or_masks() -> None:
    first = _observation(1, _mask((0, 0)))
    second = _observation(2, _mask((5, 7)))
    primary = (first,)
    supplements = ((second,),)
    before_first = first.mask.tobytes()
    before_second = second.mask.tobytes()

    merged = merge_temporal_object_observations(primary, supplements)

    assert primary == (first,)
    assert supplements == ((second,),)
    assert merged[0] is first
    assert merged[1] is second
    assert first.mask.tobytes() == before_first
    assert second.mask.tobytes() == before_second
    assert not first.mask.flags.writeable
    assert not second.mask.flags.writeable
