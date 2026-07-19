from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from src.oviv2.addressing import VoxelKey
from src.oviv2.association import (
    AssociationConfig,
    AssociationTarget,
    Assignment,
    score_candidate,
    solve_assignment,
)
from src.oviv2.observation_graph import (
    CausalObservationGraph,
    ObservationEdge,
    ObservationNode,
)
from src.oviv2.observations import FrameObservation, ObservationKind


_LEGACY_MIN_VOXEL_OVERLAP = 0.1
_LEGACY_MAX_CENTROID_DISTANCE_M = 0.5


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return normalized


@dataclass(frozen=True)
class LocalTrackerConfig:
    window_size: int = 5
    confirm_hits: int = 2
    max_age_frames: int = 3
    min_voxel_overlap: float = _LEGACY_MIN_VOXEL_OVERLAP
    max_centroid_distance_m: float = _LEGACY_MAX_CENTROID_DISTANCE_M
    association: AssociationConfig | None = None
    ambiguous_edge_score: float = 0.70
    third_view_min_score: float = 0.75

    def __post_init__(self) -> None:
        window_size = _positive_integer(self.window_size, "window_size")
        if not 3 <= window_size <= 5:
            raise ValueError("window_size must lie in [3, 5]")
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(
            self,
            "confirm_hits",
            _positive_integer(self.confirm_hits, "confirm_hits"),
        )
        object.__setattr__(
            self,
            "max_age_frames",
            _non_negative_integer(self.max_age_frames, "max_age_frames"),
        )
        min_overlap = _unit_interval(self.min_voxel_overlap, "min_voxel_overlap")
        object.__setattr__(self, "min_voxel_overlap", min_overlap)
        if isinstance(self.max_centroid_distance_m, (bool, np.bool_)) or not isinstance(
            self.max_centroid_distance_m,
            Real,
        ):
            raise TypeError("max_centroid_distance_m must be a real number")
        max_distance = float(self.max_centroid_distance_m)
        if not np.isfinite(max_distance) or max_distance <= 0.0:
            raise ValueError("max_centroid_distance_m must be finite and positive")
        object.__setattr__(self, "max_centroid_distance_m", max_distance)
        object.__setattr__(
            self,
            "ambiguous_edge_score",
            _unit_interval(self.ambiguous_edge_score, "ambiguous_edge_score"),
        )
        object.__setattr__(
            self,
            "third_view_min_score",
            _unit_interval(self.third_view_min_score, "third_view_min_score"),
        )

        association = self.association
        if association is None:
            association = AssociationConfig(
                min_directed_overlap=min_overlap,
                max_centroid_distance_m=max_distance,
            )
        else:
            if not isinstance(association, AssociationConfig):
                raise TypeError("association must be an AssociationConfig")
            if (
                min_overlap != _LEGACY_MIN_VOXEL_OVERLAP
                or max_distance != _LEGACY_MAX_CENTROID_DISTANCE_M
            ):
                raise ValueError(
                    "explicit association cannot be combined with non-default legacy "
                    "association thresholds"
                )
        object.__setattr__(self, "association", association)


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
    visual_feature: tuple[float, ...] | None = None
    feature_model_id: str | None = None


@dataclass(frozen=True)
class LocalTrackBatch:
    updated: tuple[LocalTrack, ...]
    accepted: tuple[LocalTrack, ...]
    match_count: int = 0
    new_track_count: int = 0
    conflict_count: int = 0
    revoked_edge_count: int = 0


def _visual_summary(
    observations: tuple[FrameObservation, ...],
) -> tuple[tuple[float, ...] | None, str | None]:
    groups: dict[tuple[str, int], list[tuple[float, int, np.ndarray]]] = {}
    for item in observations:
        if item.image_feature is None or item.feature_model_id is None:
            continue
        feature = np.asarray(item.image_feature, dtype=np.float64)
        quality = float(item.confidence * (1.0 - item.border_contact_fraction))
        groups.setdefault((item.feature_model_id, int(feature.size)), []).append(
            (quality, item.observation_id, feature)
        )
    if not groups:
        return None, None

    totals = {
        key: float(sum(value[0] for value in values))
        for key, values in groups.items()
    }
    selected = sorted(groups, key=lambda key: (-totals[key], key))[0]
    values = groups[selected]
    weighted_sum = np.zeros(selected[1], dtype=np.float64)
    for quality, _, feature in values:
        weighted_sum += quality * feature
    norm = float(np.linalg.norm(weighted_sum))
    cancellation_limit = 1e-12 * max(1.0, totals[selected])
    if not np.isfinite(norm) or norm <= cancellation_limit:
        _, _, fallback = sorted(values, key=lambda value: (-value[0], value[1]))[0]
        normalized = fallback
    else:
        scale = float(np.max(np.abs(weighted_sum)))
        scaled = weighted_sum / scale
        normalized = scaled / np.linalg.norm(scaled)
    return tuple(float(value) for value in normalized), selected[0]


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
    visual_feature, feature_model_id = _visual_summary(observations)
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
        visual_feature=visual_feature,
        feature_model_id=feature_model_id,
    )


def _track_target(track: LocalTrack) -> AssociationTarget:
    return AssociationTarget(
        target_id=track.track_id,
        voxel_keys=track.voxel_keys,
        centroid_xyz=track.centroid_xyz,
        bounds_min_xyz=track.bounds_min_xyz,
        bounds_max_xyz=track.bounds_max_xyz,
        semantic_id=track.semantic_id,
        semantic_confidence=track.semantic_confidence,
        visual_feature=track.visual_feature,
        feature_model_id=track.feature_model_id,
    )


def _observation_target(observation: FrameObservation) -> AssociationTarget:
    visual_feature = (
        None
        if observation.image_feature is None
        else tuple(float(value) for value in observation.image_feature)
    )
    return AssociationTarget(
        target_id=observation.observation_id,
        voxel_keys=observation.voxel_keys,
        centroid_xyz=observation.centroid_xyz,
        bounds_min_xyz=observation.bounds_min_xyz,
        bounds_max_xyz=observation.bounds_max_xyz,
        semantic_id=observation.semantic_id,
        semantic_confidence=observation.confidence,
        visual_feature=visual_feature,
        feature_model_id=(observation.feature_model_id if visual_feature is not None else None),
    )


class LocalTracker:
    def __init__(self, config: LocalTrackerConfig = LocalTrackerConfig()) -> None:
        if not isinstance(config, LocalTrackerConfig):
            raise TypeError("config must be LocalTrackerConfig")
        self.config = config
        self.tracks: dict[int, LocalTrack] = {}
        self.graph = CausalObservationGraph(config.window_size)
        self._next_track_id = 1
        self._last_frame_id: int | None = None

    def update(
        self,
        observations: tuple[FrameObservation, ...],
        frame_id: int,
    ) -> LocalTrackBatch:
        current_frame = _non_negative_integer(frame_id, "frame_id")
        if not isinstance(observations, tuple):
            raise TypeError("observations must be a tuple")
        if any(not isinstance(item, FrameObservation) for item in observations):
            raise TypeError("observations must contain only FrameObservation values")
        if self._last_frame_id is not None and current_frame <= self._last_frame_id:
            raise ValueError("frame_id must be strictly increasing")
        if any(item.frame_id != current_frame for item in observations):
            raise ValueError("observation frame_id must match frame_id")

        objects = tuple(
            sorted(
                (item for item in observations if item.kind is ObservationKind.OBJECT),
                key=lambda item: item.observation_id,
            )
        )
        object_ids = [item.observation_id for item in objects]
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0
            for value in object_ids
        ):
            raise ValueError("object observation IDs must be non-negative integers")
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("object observation IDs must be unique within a frame")

        nodes = tuple(ObservationNode(item.observation_id, current_frame) for item in objects)
        observation_targets = tuple(_observation_target(item) for item in objects)
        observation_by_id = {item.observation_id: item for item in objects}

        trial_graph = copy.deepcopy(self.graph)
        trial_tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if current_frame - track.last_frame_id <= self.config.max_age_frames
        }
        next_track_id = self._next_track_id
        trial_graph.add_frame(current_frame, nodes)
        active_observation_ids = set(trial_graph.observation_ids)
        candidate_tracks = tuple(
            track
            for track in sorted(trial_tracks.values(), key=lambda item: item.track_id)
            if track.observations[-1].observation_id in active_observation_ids
        )
        track_targets = tuple(_track_target(track) for track in candidate_tracks)

        conflict_count = 0
        for track_target in track_targets:
            for observation_target in observation_targets:
                candidate = score_candidate(
                    track_target,
                    observation_target,
                    self.config.association,
                )
                if candidate is not None and candidate.conflict:
                    conflict_count += 1
        assignments = solve_assignment(
            track_targets,
            observation_targets,
            self.config.association,
        )

        track_by_id = {track.track_id: track for track in candidate_tracks}
        accepted_assignments: dict[int, Assignment] = {}
        revoked_edge_count = 0
        for assignment in sorted(assignments, key=lambda item: item.right_id):
            track = track_by_id[assignment.left_id]
            current = observation_by_id[assignment.right_id]
            latest = track.observations[-1]
            history_edges: list[tuple[int, int]] = []
            for historical in track.observations[:-1]:
                if (
                    historical.observation_id not in active_observation_ids
                    or historical.frame_id >= current_frame
                ):
                    continue
                candidate = score_candidate(
                    _observation_target(historical),
                    _observation_target(current),
                    self.config.association,
                )
                if candidate is None or not candidate.accepted:
                    continue
                trial_graph.add_edge(
                    ObservationEdge(
                        historical.observation_id,
                        current.observation_id,
                        historical.frame_id,
                        current_frame,
                        candidate.score,
                    )
                )
                history_edges.append((historical.observation_id, current.observation_id))

            trial_graph.add_edge(
                ObservationEdge(
                    latest.observation_id,
                    current.observation_id,
                    latest.frame_id,
                    current_frame,
                    assignment.score,
                )
            )
            accepted = assignment.score >= self.config.ambiguous_edge_score
            if not accepted:
                accepted = trial_graph.edge_is_supported(
                    latest.observation_id,
                    current.observation_id,
                    ambiguous_below=self.config.ambiguous_edge_score,
                    third_view_min_score=self.config.third_view_min_score,
                )
            if accepted:
                accepted_assignments[current.observation_id] = assignment
                continue

            trial_graph.remove_edge(latest.observation_id, current.observation_id)
            for left_id, right_id in history_edges:
                trial_graph.remove_edge(left_id, right_id)
            revoked_edge_count += 1

        updated: list[LocalTrack] = []
        for observation in objects:
            assignment = accepted_assignments.get(observation.observation_id)
            if assignment is None:
                track_id = next_track_id
                next_track_id += 1
                track = _summarize(
                    track_id,
                    (observation,),
                    hit_count=1,
                    first_frame_id=current_frame,
                    confirm_hits=self.config.confirm_hits,
                )
            else:
                previous = trial_tracks[assignment.left_id]
                history = (*previous.observations, observation)[-self.config.window_size :]
                track = _summarize(
                    previous.track_id,
                    history,
                    hit_count=previous.hit_count + 1,
                    first_frame_id=previous.first_frame_id,
                    confirm_hits=self.config.confirm_hits,
                )
                track_id = previous.track_id
            trial_tracks[track_id] = track
            updated.append(track)

        accepted = tuple(track for track in updated if track.confirmed)
        batch = LocalTrackBatch(
            updated=tuple(updated),
            accepted=accepted,
            match_count=len(accepted_assignments),
            new_track_count=len(objects) - len(accepted_assignments),
            conflict_count=conflict_count,
            revoked_edge_count=revoked_edge_count,
        )
        self.graph = trial_graph
        self.tracks = trial_tracks
        self._next_track_id = next_track_id
        self._last_frame_id = current_frame
        return batch
