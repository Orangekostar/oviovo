from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.oviv2.addressing import VoxelKey
from src.oviv2.observations import FrameObservation, ObservationKind


@dataclass(frozen=True)
class LocalTrackerConfig:
    window_size: int = 5
    confirm_hits: int = 2
    max_age_frames: int = 3
    min_voxel_overlap: float = 0.1
    max_centroid_distance_m: float = 0.5

    def __post_init__(self) -> None:
        if not 3 <= self.window_size <= 5:
            raise ValueError("window_size must lie in [3, 5]")
        if self.confirm_hits <= 0 or self.max_age_frames < 0:
            raise ValueError("confirm_hits must be positive and max_age_frames non-negative")
        if not 0.0 <= self.min_voxel_overlap <= 1.0:
            raise ValueError("min_voxel_overlap must lie in [0, 1]")
        if not np.isfinite(self.max_centroid_distance_m) or self.max_centroid_distance_m <= 0.0:
            raise ValueError("max_centroid_distance_m must be finite and positive")


@dataclass(frozen=True)
class LocalTrack:
    track_id: int
    observations: tuple[FrameObservation, ...]
    hit_count: int
    first_frame_id: int
    last_frame_id: int
    confirmed: bool
    voxel_keys: frozenset[VoxelKey]
    centroid_xyz: tuple[float, float, float]
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    label: str
    semantic_id: int
    semantic_confidence: float


@dataclass(frozen=True)
class LocalTrackBatch:
    updated: tuple[LocalTrack, ...]
    accepted: tuple[LocalTrack, ...]


def _overlap(left: frozenset[VoxelKey], right: frozenset[VoxelKey]) -> float:
    denominator = min(len(left), len(right))
    return float(len(left & right) / denominator) if denominator else 0.0


def _summarize(
    track_id: int,
    observations: tuple[FrameObservation, ...],
    *,
    hit_count: int,
    first_frame_id: int,
    confirm_hits: int,
) -> LocalTrack:
    voxel_keys = frozenset().union(*(item.voxel_keys for item in observations))
    centroids = np.asarray([item.centroid_xyz for item in observations], dtype=np.float64)
    bounds_min = np.asarray([item.bounds_min_xyz for item in observations], dtype=np.float64).min(axis=0)
    bounds_max = np.asarray([item.bounds_max_xyz for item in observations], dtype=np.float64).max(axis=0)
    semantic_votes: dict[tuple[int, str], float] = {}
    for item in observations:
        if item.semantic_id > 0:
            key = (item.semantic_id, item.label)
            semantic_votes[key] = semantic_votes.get(key, 0.0) + item.confidence
    if semantic_votes:
        (semantic_id, label), winning = sorted(
            semantic_votes.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[0]
        total = sum(semantic_votes.values())
        semantic_confidence = float(winning / total)
    else:
        semantic_id, label, semantic_confidence = 0, "", 0.0
    return LocalTrack(
        track_id=track_id,
        observations=observations,
        hit_count=hit_count,
        first_frame_id=first_frame_id,
        last_frame_id=observations[-1].frame_id,
        confirmed=hit_count >= confirm_hits,
        voxel_keys=voxel_keys,
        centroid_xyz=tuple(float(value) for value in centroids.mean(axis=0)),
        bounds_min_xyz=tuple(float(value) for value in bounds_min),
        bounds_max_xyz=tuple(float(value) for value in bounds_max),
        label=label,
        semantic_id=semantic_id,
        semantic_confidence=semantic_confidence,
    )


class LocalTracker:
    def __init__(self, config: LocalTrackerConfig = LocalTrackerConfig()) -> None:
        if not isinstance(config, LocalTrackerConfig):
            raise TypeError("config must be LocalTrackerConfig")
        self.config = config
        self.tracks: dict[int, LocalTrack] = {}
        self._next_track_id = 1

    def update(
        self,
        observations: tuple[FrameObservation, ...],
        frame_id: int,
    ) -> LocalTrackBatch:
        current_frame = int(frame_id)
        if current_frame < 0:
            raise ValueError("frame_id must be non-negative")
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if current_frame - track.last_frame_id <= self.config.max_age_frames
        }
        available = set(self.tracks)
        updated: list[LocalTrack] = []
        objects = [item for item in observations if item.kind is ObservationKind.OBJECT]
        for observation in sorted(objects, key=lambda item: item.observation_id):
            candidate = self._match(observation, available)
            if candidate is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                track = _summarize(
                    track_id,
                    (observation,),
                    hit_count=1,
                    first_frame_id=current_frame,
                    confirm_hits=self.config.confirm_hits,
                )
            else:
                previous = self.tracks[candidate]
                track_id = previous.track_id
                history = (*previous.observations, observation)[-self.config.window_size :]
                track = _summarize(
                    track_id,
                    history,
                    hit_count=previous.hit_count + 1,
                    first_frame_id=previous.first_frame_id,
                    confirm_hits=self.config.confirm_hits,
                )
                available.remove(candidate)
            self.tracks[track_id] = track
            updated.append(track)
        accepted = tuple(track for track in updated if track.confirmed)
        return LocalTrackBatch(tuple(updated), accepted)

    def _match(self, observation: FrameObservation, available: set[int]) -> int | None:
        ranked: list[tuple[float, float, int]] = []
        observed_centroid = np.asarray(observation.centroid_xyz)
        for track_id in available:
            track = self.tracks[track_id]
            overlap = _overlap(track.voxel_keys, observation.voxel_keys)
            distance = float(np.linalg.norm(np.asarray(track.centroid_xyz) - observed_centroid))
            if overlap < self.config.min_voxel_overlap and distance > self.config.max_centroid_distance_m:
                continue
            distance_score = max(0.0, 1.0 - distance / self.config.max_centroid_distance_m)
            geometry_score = overlap + distance_score
            semantic_score = float(
                observation.semantic_id > 0 and observation.semantic_id == track.semantic_id
            )
            ranked.append((geometry_score, semantic_score, track_id))
        if not ranked:
            return None
        return sorted(ranked, key=lambda item: (-item[0], -item[1], item[2]))[0][2]
