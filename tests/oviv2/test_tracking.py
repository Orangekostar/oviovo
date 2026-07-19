from __future__ import annotations

import numpy as np

from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.tracking import LocalTracker, LocalTrackerConfig


def observation(
    frame_id: int,
    keys: set[tuple[int, int, int]],
    *,
    label: str = "chair",
    semantic_id: int = 2,
    observation_id: int | None = None,
) -> FrameObservation:
    points = np.asarray(sorted(keys), dtype=np.float64) * 0.05
    return FrameObservation(
        observation_id=frame_id * 100 + (0 if observation_id is None else observation_id),
        frame_id=frame_id,
        timestamp=float(frame_id),
        kind=ObservationKind.OBJECT,
        label=label,
        semantic_id=semantic_id,
        confidence=0.9,
        mask=np.ones((2, 2), dtype=bool),
        bbox_xyxy=(0.0, 0.0, 2.0, 2.0),
        voxel_keys=frozenset(keys),
        centroid_xyz=tuple(points.mean(axis=0)),
        bounds_min_xyz=tuple(points.min(axis=0)),
        bounds_max_xyz=tuple(points.max(axis=0)),
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


def test_tracker_matches_geometry_before_semantic_label() -> None:
    tracker = LocalTracker(
        LocalTrackerConfig(confirm_hits=1, min_voxel_overlap=0.25, max_centroid_distance_m=0.3)
    )
    first = tracker.update((observation(0, {(0, 0, 20), (1, 0, 20)}),), 0).accepted[0]
    relabeled = observation(
        1,
        {(1, 0, 20), (2, 0, 20)},
        label="stool",
        semantic_id=3,
    )
    second = tracker.update((relabeled,), 1).accepted[0]

    assert second.track_id == first.track_id
    assert second.semantic_id == 2


def test_tracker_does_not_match_semantically_equal_but_distant_observation() -> None:
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1, max_centroid_distance_m=0.2))
    first = tracker.update((observation(0, {(0, 0, 20)}),), 0).accepted[0]
    second = tracker.update((observation(1, {(100, 0, 20)}),), 1).accepted[0]

    assert second.track_id != first.track_id
