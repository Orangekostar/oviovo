from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.domain.arrays import readonly_array


class LocalTrackState(str, Enum):
    TENTATIVE = "tentative"
    STABLE = "stable"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class LocalTrack:
    local_track_id: str
    state: LocalTrackState
    observation_ids: tuple[str, ...]
    first_frame_id: int
    last_frame_id: int
    first_timestamp: float
    last_timestamp: float
    hit_count: int
    miss_count: int
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    semantic_label: str = ""
    semantic_confidence: float = 0.0
    appearance_embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not str(self.local_track_id):
            raise ValueError("local_track_id must be non-empty")
        observation_ids = tuple(str(value) for value in self.observation_ids)
        if not observation_ids or len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation_ids must be non-empty and unique")
        first_frame_id = int(self.first_frame_id)
        last_frame_id = int(self.last_frame_id)
        if first_frame_id < 0 or last_frame_id < first_frame_id:
            raise ValueError("track frame interval is invalid")
        first_timestamp = float(self.first_timestamp)
        last_timestamp = float(self.last_timestamp)
        if not np.isfinite([first_timestamp, last_timestamp]).all() or last_timestamp < first_timestamp:
            raise ValueError("track timestamp interval is invalid")
        if int(self.hit_count) < 0 or int(self.miss_count) < 0:
            raise ValueError("track counters must be non-negative")
        semantic_confidence = float(self.semantic_confidence)
        if not 0.0 <= semantic_confidence <= 1.0:
            raise ValueError("semantic_confidence must be in [0, 1]")

        object.__setattr__(self, "observation_ids", observation_ids)
        object.__setattr__(self, "first_frame_id", first_frame_id)
        object.__setattr__(self, "last_frame_id", last_frame_id)
        object.__setattr__(self, "first_timestamp", first_timestamp)
        object.__setattr__(self, "last_timestamp", last_timestamp)
        object.__setattr__(self, "hit_count", int(self.hit_count))
        object.__setattr__(self, "miss_count", int(self.miss_count))
        object.__setattr__(self, "centroid", readonly_array(self.centroid, dtype=np.float32, ndim=1))
        object.__setattr__(self, "bbox_min", readonly_array(self.bbox_min, dtype=np.float32, ndim=1))
        object.__setattr__(self, "bbox_max", readonly_array(self.bbox_max, dtype=np.float32, ndim=1))
        if self.centroid.shape != (3,) or self.bbox_min.shape != (3,) or self.bbox_max.shape != (3,):
            raise ValueError("track centroid and bounds must have shape (3,)")
        if np.any(self.bbox_max < self.bbox_min):
            raise ValueError("track bbox_max must be >= bbox_min")
        object.__setattr__(self, "semantic_label", str(self.semantic_label).strip())
        object.__setattr__(self, "semantic_confidence", semantic_confidence)
        if self.appearance_embedding is not None:
            object.__setattr__(
                self,
                "appearance_embedding",
                readonly_array(self.appearance_embedding, dtype=np.float32, ndim=1),
            )


@dataclass(frozen=True)
class LocalTrackBatch:
    frame_id: int
    tracks: tuple[LocalTrack, ...] = ()

    def __post_init__(self) -> None:
        tracks = tuple(self.tracks)
        track_ids = [track.local_track_id for track in tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("local_track_id values must be unique within a batch")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "tracks", tracks)

    @property
    def stable_tracks(self) -> tuple[LocalTrack, ...]:
        return tuple(track for track in self.tracks if track.state is LocalTrackState.STABLE)
