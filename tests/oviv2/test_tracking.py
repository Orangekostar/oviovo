from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.oviv2.association import AssociationConfig
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.tracking import LocalTracker, LocalTrackerConfig


def observation(
    frame_id: int,
    keys: set[tuple[int, int, int]],
    *,
    label: str = "chair",
    semantic_id: int = 2,
    observation_id: int | None = None,
    image_feature: np.ndarray | None = None,
    text_feature: np.ndarray | None = None,
    feature_model_id: str | None = None,
    view_direction_xyz: tuple[float, float, float] | None = None,
    visible_pixel_count: int = 0,
    border_contact_fraction: float = 0.0,
    confidence: float = 0.9,
) -> FrameObservation:
    points = np.asarray(sorted(keys), dtype=np.float64) * 0.05
    return FrameObservation(
        observation_id=frame_id * 100 + (0 if observation_id is None else observation_id),
        frame_id=frame_id,
        timestamp=float(frame_id),
        kind=ObservationKind.OBJECT,
        label=label,
        semantic_id=semantic_id,
        confidence=confidence,
        mask=np.ones((2, 2), dtype=bool),
        bbox_xyxy=(0.0, 0.0, 2.0, 2.0),
        voxel_keys=frozenset(keys),
        centroid_xyz=tuple(points.mean(axis=0)),
        bounds_min_xyz=tuple(points.min(axis=0)),
        bounds_max_xyz=tuple(points.max(axis=0)),
        image_feature=image_feature,
        text_feature=text_feature,
        feature_model_id=feature_model_id,
        view_direction_xyz=view_direction_xyz,
        visible_pixel_count=visible_pixel_count,
        border_contact_fraction=border_contact_fraction,
    )


def test_tracker_requires_two_hits_and_keeps_bounded_window() -> None:
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=2, window_size=3))
    keys = {(0, 0, 20), (1, 0, 20)}

    assert tracker.update((observation(0, keys),), 0).accepted == ()
    second = tracker.update((observation(1, keys),), 1)
    assert len(second.accepted) == 1
    track_id = second.accepted[0].track_id
    for frame_id in range(2, 6):
        tracker.update((observation(frame_id, keys),), frame_id)

    track = tracker.tracks[track_id]
    assert track.hit_count == 6
    assert [item.frame_id for item in track.observations] == [3, 4, 5]


def test_tracker_expires_stale_tracks_and_ignores_non_objects() -> None:
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1, max_age_frames=2))
    obj = observation(0, {(0, 0, 20)})
    structure = FrameObservation(
        **{**obj.__dict__, "observation_id": 8, "kind": ObservationKind.STRUCTURE}
    )

    result = tracker.update((obj, structure), 0)
    assert len(result.accepted) == 1
    tracker.update((), 3)

    assert tracker.tracks == {}


def test_tracker_does_not_merge_adjacent_conflicting_objects() -> None:
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
    chair = observation(
        0,
        {(0, 0, 20), (1, 0, 20)},
        semantic_id=2,
        label="chair",
        observation_id=1,
    )
    table = observation(
        0,
        {(1, 0, 20), (2, 0, 20)},
        semantic_id=3,
        label="table",
        observation_id=2,
    )
    first = tracker.update((chair, table), 0)
    next_chair = replace(
        chair,
        frame_id=1,
        timestamp=1.0,
        observation_id=101,
        voxel_keys=table.voxel_keys,
        centroid_xyz=table.centroid_xyz,
        bounds_min_xyz=table.bounds_min_xyz,
        bounds_max_xyz=table.bounds_max_xyz,
    )
    next_table = replace(
        table,
        frame_id=1,
        timestamp=1.0,
        observation_id=102,
        voxel_keys=chair.voxel_keys,
        centroid_xyz=chair.centroid_xyz,
        bounds_min_xyz=chair.bounds_min_xyz,
        bounds_max_xyz=chair.bounds_max_xyz,
    )

    second = tracker.update((next_chair, next_table), 1)

    assert len({track.track_id for track in first.accepted}) == 2
    assert len({track.track_id for track in second.accepted}) == 2
    assert len(tracker.tracks) == 2
    assert tracker.tracks[first.accepted[0].track_id].semantic_id == 2
    assert tracker.tracks[first.accepted[1].track_id].semantic_id == 3
    assert (second.match_count, second.new_track_count) == (2, 0)
    assert (second.conflict_count, second.revoked_edge_count) == (2, 0)


def test_tracker_joint_assignment_is_independent_of_observation_order() -> None:
    def run(reverse_second_frame: bool) -> tuple[tuple[object, ...], ...]:
        tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
        left = observation(0, {(0, 0, 20)}, observation_id=1)
        right = observation(0, {(10, 0, 20)}, observation_id=2)
        tracker.update((left, right), 0)
        second = (
            replace(left, frame_id=1, timestamp=1.0, observation_id=101),
            replace(right, frame_id=1, timestamp=1.0, observation_id=102),
        )
        tracker.update(tuple(reversed(second)) if reverse_second_frame else second, 1)
        return tuple(
            (
                track.track_id,
                track.hit_count,
                track.last_frame_id,
                tuple(item.observation_id for item in track.observations),
                tuple(sorted(track.voxel_keys)),
            )
            for track in sorted(tracker.tracks.values(), key=lambda value: value.track_id)
        )

    assert run(False) == run(True)


def test_tracker_keeps_relabel_on_same_track_only_with_visual_override() -> None:
    tracker = LocalTracker(
        LocalTrackerConfig(confirm_hits=1, min_voxel_overlap=0.25, max_centroid_distance_m=0.3)
    )
    keys = {(0, 0, 20), (1, 0, 20)}
    first = tracker.update(
        (
            observation(
                0,
                keys,
                image_feature=np.asarray((1.0, 0.0)),
                feature_model_id="clip",
            ),
        ),
        0,
    ).accepted[0]
    relabeled = observation(
        1,
        keys,
        label="stool",
        semantic_id=3,
        image_feature=np.asarray((0.9, np.sqrt(0.19))),
        feature_model_id="clip",
    )
    second = tracker.update((relabeled,), 1).accepted[0]

    assert second.track_id == first.track_id
    assert second.semantic_id == 2


def test_tracker_starts_new_track_for_confident_relabel_without_compatible_visual() -> None:
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
    keys = {(0, 0, 20), (1, 0, 20)}
    first = tracker.update((observation(0, keys),), 0).accepted[0]

    second = tracker.update(
        (observation(1, keys, label="stool", semantic_id=3),),
        1,
    ).accepted[0]

    assert second.track_id != first.track_id


def test_tracker_does_not_match_semantically_equal_but_distant_observation() -> None:
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1, max_centroid_distance_m=0.2))
    first = tracker.update((observation(0, {(0, 0, 20)}),), 0).accepted[0]
    second = tracker.update((observation(1, {(100, 0, 20)}),), 1).accepted[0]

    assert second.track_id != first.track_id


def test_tracker_assigns_at_most_one_observation_to_each_track_per_frame() -> None:
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
    keys = {(0, 0, 20), (1, 0, 20)}
    first = tracker.update((observation(0, keys, observation_id=1),), 0).accepted[0]

    result = tracker.update(
        (
            observation(1, keys, observation_id=1),
            observation(1, keys, observation_id=2),
        ),
        1,
    )

    assert (result.match_count, result.new_track_count) == (1, 1)
    assert [track.track_id for track in result.updated] == [first.track_id, 2]
    assert len(tracker.tracks) == 2


def test_tracker_revokes_weak_edge_without_independent_support() -> None:
    tracker = LocalTracker(
        LocalTrackerConfig(
            confirm_hits=1,
            association=AssociationConfig(),
            ambiguous_edge_score=0.8,
        )
    )
    keys = {(0, 0, 20), (1, 0, 20)}
    first = tracker.update(
        (
            observation(
                0,
                keys,
                image_feature=np.asarray((1.0, 0.0)),
                feature_model_id="clip",
            ),
        ),
        0,
    ).accepted[0]

    result = tracker.update(
        (
            observation(
                1,
                keys,
                image_feature=np.asarray((0.0, 1.0)),
                feature_model_id="clip",
            ),
        ),
        1,
    )

    assert result.accepted[0].track_id != first.track_id
    assert (result.match_count, result.new_track_count) == (0, 1)
    assert (result.conflict_count, result.revoked_edge_count) == (0, 1)
    assert tracker.graph.edge_count == 0


def test_tracker_accepts_strong_edge_without_third_view() -> None:
    tracker = LocalTracker(
        LocalTrackerConfig(confirm_hits=1, ambiguous_edge_score=0.8)
    )
    keys = {(0, 0, 20), (1, 0, 20)}
    feature = np.asarray((1.0, 0.0))
    first = tracker.update(
        (observation(0, keys, image_feature=feature, feature_model_id="clip"),),
        0,
    ).accepted[0]

    result = tracker.update(
        (observation(1, keys, image_feature=feature, feature_model_id="clip"),),
        1,
    )

    assert result.accepted[0].track_id == first.track_id
    assert (result.match_count, result.new_track_count) == (1, 0)
    assert result.revoked_edge_count == 0
    assert tracker.graph.edge_count == 1


def test_tracker_accepts_weak_latest_edge_with_independent_frame_zero_support() -> None:
    tracker = LocalTracker(
        LocalTrackerConfig(
            confirm_hits=1,
            association=AssociationConfig(),
            ambiguous_edge_score=0.95,
            third_view_min_score=0.95,
        )
    )
    keys = {(0, 0, 20), (1, 0, 20)}
    sine = np.sqrt(0.19)
    first = tracker.update(
        (
            observation(
                0,
                keys,
                observation_id=1,
                image_feature=np.asarray((1.0, 0.0)),
                feature_model_id="clip",
                border_contact_fraction=0.99,
            ),
        ),
        0,
    ).accepted[0]
    strong = tracker.update(
        (
            observation(
                1,
                keys,
                observation_id=1,
                image_feature=np.asarray((0.9, sine)),
                feature_model_id="clip",
            ),
        ),
        1,
    )
    assert strong.match_count == 1

    supported = tracker.update(
        (
            observation(
                2,
                keys,
                observation_id=1,
                image_feature=np.asarray((0.9, -sine)),
                feature_model_id="clip",
            ),
        ),
        2,
    )

    assert supported.accepted[0].track_id == first.track_id
    assert (supported.match_count, supported.new_track_count) == (1, 0)
    assert supported.revoked_edge_count == 0
    assert tracker.graph.edge_count == 3


def test_tracker_visual_feature_is_quality_weighted_unit_mean() -> None:
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
    keys = {(0, 0, 20), (1, 0, 20)}
    tracker.update(
        (
            observation(
                0,
                keys,
                confidence=0.5,
                image_feature=np.asarray((1.0, 0.0)),
                feature_model_id="clip",
            ),
        ),
        0,
    )

    track = tracker.update(
        (
            observation(
                1,
                keys,
                confidence=1.0,
                image_feature=np.asarray((0.0, 1.0)),
                feature_model_id="clip",
            ),
        ),
        1,
    ).accepted[0]

    assert track.visual_feature == pytest.approx((1.0 / np.sqrt(5.0), 2.0 / np.sqrt(5.0)))
    assert track.feature_model_id == "clip"


def test_tracker_visual_group_selection_and_cancellation_are_deterministic() -> None:
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
    keys = {(0, 0, 20), (1, 0, 20)}
    tracker.update(
        (
            observation(
                0,
                keys,
                image_feature=np.asarray((1.0, 0.0)),
                feature_model_id="z-model",
            ),
        ),
        0,
    )
    tied = tracker.update(
        (
            observation(
                1,
                keys,
                image_feature=np.asarray((1.0, 0.0, 0.0)),
                feature_model_id="a-model",
            ),
        ),
        1,
    ).accepted[0]

    assert tied.feature_model_id == "a-model"
    assert tied.visual_feature == pytest.approx((1.0, 0.0, 0.0))

    cancelled = tracker.update(
        (
            observation(
                2,
                keys,
                image_feature=np.asarray((-1.0, 0.0, 0.0)),
                feature_model_id="a-model",
            ),
        ),
        2,
    ).accepted[0]
    assert cancelled.feature_model_id == "a-model"
    assert cancelled.visual_feature == pytest.approx((1.0, 0.0, 0.0))


def test_tracker_does_not_reuse_track_after_latest_observation_expires_from_graph() -> None:
    tracker = LocalTracker(
        LocalTrackerConfig(confirm_hits=1, window_size=3, max_age_frames=10)
    )
    keys = {(0, 0, 20)}
    first = tracker.update((observation(0, keys),), 0).accepted[0]
    tracker.update((), 1)
    tracker.update((), 2)
    tracker.update((), 3)

    next_track = tracker.update((observation(4, keys),), 4).accepted[0]

    assert next_track.track_id != first.track_id


def test_tracker_invalid_updates_are_atomic() -> None:
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
    keys = {(0, 0, 20)}
    tracker.update((observation(0, keys),), 0)

    def assert_rejected_without_mutation(
        values: object,
        frame_id: object,
        error: type[Exception],
    ) -> None:
        tracks = tracker.tracks
        graph = tracker.graph
        next_track_id = tracker._next_track_id
        last_frame_id = tracker._last_frame_id
        with pytest.raises(error):
            tracker.update(values, frame_id)  # type: ignore[arg-type]
        assert tracker.tracks is tracks
        assert tracker.graph is graph
        assert tracker._next_track_id == next_track_id
        assert tracker._last_frame_id == last_frame_id

    incoming = observation(1, keys, observation_id=1)
    duplicate = replace(incoming, label="table", semantic_id=3)
    assert_rejected_without_mutation((incoming, duplicate), 1, ValueError)
    assert_rejected_without_mutation((observation(2, keys),), 1, ValueError)
    assert_rejected_without_mutation((observation(0, keys),), 0, ValueError)
    assert_rejected_without_mutation([incoming], 1, TypeError)
    assert_rejected_without_mutation((incoming,), True, TypeError)


def test_tracker_config_unifies_legacy_and_explicit_association() -> None:
    legacy = LocalTrackerConfig(min_voxel_overlap=0.2, max_centroid_distance_m=0.3)
    assert legacy.association.min_directed_overlap == pytest.approx(0.2)
    assert legacy.association.max_centroid_distance_m == pytest.approx(0.3)

    association = AssociationConfig(max_centroid_distance_m=0.8)
    explicit = LocalTrackerConfig(association=association)
    assert explicit.association is association

    with pytest.raises(ValueError, match="legacy"):
        LocalTrackerConfig(min_voxel_overlap=0.2, association=association)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("ambiguous_edge_score", np.nan, ValueError),
        ("ambiguous_edge_score", -0.01, ValueError),
        ("ambiguous_edge_score", 1.01, ValueError),
        ("third_view_min_score", np.inf, ValueError),
        ("third_view_min_score", False, TypeError),
    ],
)
def test_tracker_config_rejects_invalid_graph_thresholds(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match=field):
        LocalTrackerConfig(**{field: value})
