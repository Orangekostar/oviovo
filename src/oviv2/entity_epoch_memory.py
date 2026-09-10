"""Deterministic identity memory that never mutates current geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from src.oviv2.entity_epoch_update import (
    EpisodeLifecycle,
    RelationInferenceState,
    RelationSupport,
)
from src.oviv2.semantic_memory import (
    FeaturePrototypeBank,
    InformativeView,
    InformativeViewBank,
)

OFFLINE_REPLAY_ON_FINAL_OVI_SUPPORT = "OFFLINE_REPLAY_ON_FINAL_OVI_SUPPORT"
ONLINE_CAUSAL_OBSERVATION = "ONLINE_CAUSAL_OBSERVATION"
_OBSERVATION_PROVENANCE = frozenset(
    {OFFLINE_REPLAY_ON_FINAL_OVI_SUPPORT, ONLINE_CAUSAL_OBSERVATION}
)


class MemoryRetrievalMode(str, Enum):
    LAST_PROTOTYPE = "last_prototype"
    MULTIVIEW_BANK = "multiview_bank"


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _unit_descriptor(value: object) -> tuple[float, ...]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("descriptor must be numeric") from error
    if array.ndim != 1 or not len(array):
        raise ValueError("descriptor must be a non-empty vector")
    if not np.isfinite(array).all():
        raise ValueError("descriptor must be finite")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError("descriptor must be nonzero")
    return tuple(float(value) for value in array / norm)


def _rows(value: object) -> np.ndarray:
    array = np.array(value, dtype=np.int64, copy=True, order="C")
    if (
        array.ndim != 1
        or not len(array)
        or np.any(array < 0)
        or (len(array) > 1 and np.any(array[1:] <= array[:-1]))
    ):
        raise ValueError(
            "source_vertex_indices must be ordered unique nonnegative rows"
        )
    array.setflags(write=False)
    return array


def _quality(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("quality must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError("quality must be finite and in (0, 1]")
    return result


def _owner_id(entity_id: str) -> int:
    prefix, separator, suffix = entity_id.partition(":")
    if prefix != "ovimap" or separator != ":":
        raise ValueError("memory candidates require ovimap:<positive-int> IDs")
    try:
        owner_id = int(suffix)
    except ValueError as error:
        raise ValueError(
            "memory candidates require ovimap:<positive-int> IDs"
        ) from error
    if owner_id <= 0:
        raise ValueError("memory candidate owner IDs must be positive")
    return owner_id


@dataclass(frozen=True, slots=True)
class MemoryEntityObservation:
    stable_entity_id: str
    owner_entity_id: int
    source_surface_id: str
    source_vertex_indices: np.ndarray
    descriptor: tuple[float, ...] | np.ndarray
    feature_space_id: str
    projection_version: str
    observation_id: int
    frame_id: int
    visible_pixel_count: int
    quality: float
    view_direction_xyz: tuple[float, float, float]
    lifecycle_state: EpisodeLifecycle
    observation_provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stable_entity_id",
            _nonempty(self.stable_entity_id, "stable_entity_id"),
        )
        object.__setattr__(
            self,
            "owner_entity_id",
            _integer(self.owner_entity_id, "owner_entity_id", minimum=1),
        )
        object.__setattr__(
            self,
            "source_surface_id",
            _nonempty(self.source_surface_id, "source_surface_id"),
        )
        object.__setattr__(
            self, "source_vertex_indices", _rows(self.source_vertex_indices)
        )
        object.__setattr__(self, "descriptor", _unit_descriptor(self.descriptor))
        object.__setattr__(
            self,
            "feature_space_id",
            _nonempty(self.feature_space_id, "feature_space_id"),
        )
        object.__setattr__(
            self,
            "projection_version",
            _nonempty(self.projection_version, "projection_version"),
        )
        object.__setattr__(
            self,
            "observation_id",
            _integer(self.observation_id, "observation_id", minimum=0),
        )
        object.__setattr__(
            self, "frame_id", _integer(self.frame_id, "frame_id", minimum=0)
        )
        object.__setattr__(
            self,
            "visible_pixel_count",
            _integer(self.visible_pixel_count, "visible_pixel_count", minimum=0),
        )
        object.__setattr__(self, "quality", _quality(self.quality))
        view = InformativeView(
            observation_id=self.observation_id,
            frame_id=self.frame_id,
            visible_pixel_count=self.visible_pixel_count,
            quality=self.quality,
            view_direction_xyz=self.view_direction_xyz,
        )
        object.__setattr__(self, "view_direction_xyz", view.view_direction_xyz)
        if self.lifecycle_state not in {
            EpisodeLifecycle.ACTIVE,
            EpisodeLifecycle.DORMANT,
        }:
            raise ValueError("memory identity lifecycle must be active or dormant")
        provenance = _nonempty(self.observation_provenance, "observation_provenance")
        if provenance not in _OBSERVATION_PROVENANCE:
            raise ValueError("unsupported memory observation provenance")
        object.__setattr__(self, "observation_provenance", provenance)


@dataclass(frozen=True, slots=True)
class MemoryCandidateObservation:
    entity_id: str
    source_surface_id: str
    source_vertex_indices: np.ndarray
    descriptor: tuple[float, ...] | np.ndarray
    feature_space_id: str
    projection_version: str
    observation_id: int
    frame_id: int
    quality: float

    def __post_init__(self) -> None:
        entity_id = _nonempty(self.entity_id, "entity_id")
        _owner_id(entity_id)
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(
            self,
            "source_surface_id",
            _nonempty(self.source_surface_id, "source_surface_id"),
        )
        object.__setattr__(
            self, "source_vertex_indices", _rows(self.source_vertex_indices)
        )
        object.__setattr__(self, "descriptor", _unit_descriptor(self.descriptor))
        object.__setattr__(
            self,
            "feature_space_id",
            _nonempty(self.feature_space_id, "feature_space_id"),
        )
        object.__setattr__(
            self,
            "projection_version",
            _nonempty(self.projection_version, "projection_version"),
        )
        object.__setattr__(
            self,
            "observation_id",
            _integer(self.observation_id, "observation_id", minimum=0),
        )
        object.__setattr__(
            self, "frame_id", _integer(self.frame_id, "frame_id", minimum=0)
        )
        object.__setattr__(self, "quality", _quality(self.quality))

    @property
    def owner_entity_id(self) -> int:
        return _owner_id(self.entity_id)


MemoryKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class CroveMemoryEntry:
    stable_entity_id: str
    owner_entity_id: int
    source_surface_id: str
    source_vertex_indices: np.ndarray
    feature_space_id: str
    projection_version: str
    lifecycle_state: EpisodeLifecycle
    feature_bank: FeaturePrototypeBank
    view_bank: InformativeViewBank
    last_descriptor: tuple[float, ...]
    last_observation_frame: int
    accepted_frame_ids: tuple[int, ...]
    observation_provenance: tuple[str, ...]

    @property
    def key(self) -> MemoryKey:
        return (
            self.stable_entity_id,
            self.feature_space_id,
            self.projection_version,
        )


@dataclass(frozen=True, slots=True)
class CroveMemoryIndex:
    entries: tuple[CroveMemoryEntry, ...] = ()
    max_prototypes: int = 3
    prototype_merge_cosine: float = 0.90
    max_views: int = 10
    minimum_view_novelty_cosine: float = 0.10

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(not isinstance(entry, CroveMemoryEntry) for entry in entries):
            raise TypeError("entries must contain CroveMemoryEntry values")
        entries = tuple(sorted(entries, key=lambda item: item.key))
        if len({entry.key for entry in entries}) != len(entries):
            raise ValueError("memory entry keys must be unique")
        FeaturePrototypeBank(
            max_prototypes=self.max_prototypes,
            merge_cosine=self.prototype_merge_cosine,
        )
        InformativeViewBank(
            max_views=self.max_views,
            minimum_novelty_cosine=self.minimum_view_novelty_cosine,
        )
        object.__setattr__(self, "entries", entries)

    @staticmethod
    def _model_id(feature_space_id: str, projection_version: str) -> str:
        return f"{feature_space_id}@{projection_version}"

    def observe(self, observation: MemoryEntityObservation) -> CroveMemoryIndex:
        if not isinstance(observation, MemoryEntityObservation):
            raise TypeError("observation must be MemoryEntityObservation")
        key = (
            observation.stable_entity_id,
            observation.feature_space_id,
            observation.projection_version,
        )
        positions = {entry.key: index for index, entry in enumerate(self.entries)}
        model_id = self._model_id(
            observation.feature_space_id, observation.projection_version
        )
        view = InformativeView(
            observation_id=observation.observation_id,
            frame_id=observation.frame_id,
            visible_pixel_count=observation.visible_pixel_count,
            quality=observation.quality,
            view_direction_xyz=observation.view_direction_xyz,
        )
        values = list(self.entries)
        if key not in positions:
            feature_bank = FeaturePrototypeBank(
                max_prototypes=self.max_prototypes,
                merge_cosine=self.prototype_merge_cosine,
            ).update(observation.descriptor, model_id, observation.quality)
            view_bank = InformativeViewBank(
                max_views=self.max_views,
                minimum_novelty_cosine=self.minimum_view_novelty_cosine,
            ).update(view)
            values.append(
                CroveMemoryEntry(
                    stable_entity_id=observation.stable_entity_id,
                    owner_entity_id=observation.owner_entity_id,
                    source_surface_id=observation.source_surface_id,
                    source_vertex_indices=observation.source_vertex_indices,
                    feature_space_id=observation.feature_space_id,
                    projection_version=observation.projection_version,
                    lifecycle_state=observation.lifecycle_state,
                    feature_bank=feature_bank,
                    view_bank=view_bank,
                    last_descriptor=tuple(observation.descriptor),
                    last_observation_frame=observation.frame_id,
                    accepted_frame_ids=(observation.frame_id,),
                    observation_provenance=(observation.observation_provenance,),
                )
            )
            return replace(self, entries=tuple(values))

        position = positions[key]
        previous = values[position]
        if observation.frame_id in previous.accepted_frame_ids:
            return self
        if (
            previous.owner_entity_id != observation.owner_entity_id
            or previous.source_surface_id != observation.source_surface_id
            or not np.array_equal(
                previous.source_vertex_indices, observation.source_vertex_indices
            )
        ):
            raise ValueError("one memory key cannot mix source OVI identities")
        later = observation.frame_id > previous.last_observation_frame
        values[position] = replace(
            previous,
            lifecycle_state=(
                observation.lifecycle_state if later else previous.lifecycle_state
            ),
            feature_bank=previous.feature_bank.update(
                observation.descriptor, model_id, observation.quality
            ),
            view_bank=previous.view_bank.update(view),
            last_descriptor=(
                tuple(observation.descriptor) if later else previous.last_descriptor
            ),
            last_observation_frame=(
                observation.frame_id if later else previous.last_observation_frame
            ),
            accepted_frame_ids=tuple(
                sorted((*previous.accepted_frame_ids, observation.frame_id))
            ),
            observation_provenance=tuple(
                sorted(
                    {
                        *previous.observation_provenance,
                        observation.observation_provenance,
                    }
                )
            ),
        )
        return replace(self, entries=tuple(values))


@dataclass(frozen=True, slots=True)
class _MemoryMatch:
    entry: CroveMemoryEntry
    candidate: MemoryCandidateObservation
    cosine: float
    competing_cosine: float
    margin: float


def _entry_cosine(
    entry: CroveMemoryEntry,
    candidate: MemoryCandidateObservation,
    mode: MemoryRetrievalMode,
) -> float | None:
    if mode is MemoryRetrievalMode.LAST_PROTOTYPE:
        if len(entry.last_descriptor) != len(candidate.descriptor):
            return None
        return float(np.dot(entry.last_descriptor, candidate.descriptor))
    return entry.feature_bank.maximum_cosine(
        candidate.descriptor,
        CroveMemoryIndex._model_id(
            candidate.feature_space_id, candidate.projection_version
        ),
    )


def _tier_match(
    entries: tuple[CroveMemoryEntry, ...],
    candidate: MemoryCandidateObservation,
    *,
    lifecycle: EpisodeLifecycle,
    mode: MemoryRetrievalMode,
    minimum_cosine: float,
    minimum_margin: float,
) -> tuple[_MemoryMatch | None, bool]:
    scored = []
    for entry in entries:
        if (
            entry.lifecycle_state is not lifecycle
            or entry.feature_space_id != candidate.feature_space_id
            or entry.projection_version != candidate.projection_version
        ):
            continue
        cosine = _entry_cosine(entry, candidate, mode)
        if cosine is not None:
            scored.append((cosine, entry))
    scored.sort(key=lambda item: (-item[0], item[1].stable_entity_id))
    if not scored or scored[0][0] < minimum_cosine:
        return None, False
    best_cosine, best_entry = scored[0]
    competing = scored[1][0] if len(scored) > 1 else 0.0
    margin = best_cosine - competing
    if margin < minimum_margin:
        return None, True
    return (
        _MemoryMatch(
            entry=best_entry,
            candidate=candidate,
            cosine=best_cosine,
            competing_cosine=competing,
            margin=margin,
        ),
        True,
    )


def _candidate_match(
    index: CroveMemoryIndex,
    candidate: MemoryCandidateObservation,
    *,
    mode: MemoryRetrievalMode,
    minimum_cosine: float,
    minimum_margin: float,
) -> _MemoryMatch | None:
    active, active_reached_gate = _tier_match(
        index.entries,
        candidate,
        lifecycle=EpisodeLifecycle.ACTIVE,
        mode=mode,
        minimum_cosine=minimum_cosine,
        minimum_margin=minimum_margin,
    )
    if active is not None or active_reached_gate:
        return active
    dormant, _ = _tier_match(
        index.entries,
        candidate,
        lifecycle=EpisodeLifecycle.DORMANT,
        mode=mode,
        minimum_cosine=minimum_cosine,
        minimum_margin=minimum_margin,
    )
    return dormant


def build_memory_relation_support(
    index: CroveMemoryIndex,
    candidates: tuple[MemoryCandidateObservation, ...],
    *,
    retrieval_mode: MemoryRetrievalMode,
    minimum_cosine: float,
    minimum_margin: float,
) -> tuple[RelationSupport, ...]:
    """Retrieve identity aliases only; location state remains updater-owned."""

    if not isinstance(index, CroveMemoryIndex):
        raise TypeError("index must be CroveMemoryIndex")
    if not isinstance(candidates, tuple) or any(
        not isinstance(candidate, MemoryCandidateObservation)
        for candidate in candidates
    ):
        raise TypeError("candidates must contain MemoryCandidateObservation values")
    if not isinstance(retrieval_mode, MemoryRetrievalMode):
        raise TypeError("retrieval_mode must be MemoryRetrievalMode")
    for value, name, lower in (
        (minimum_cosine, "minimum_cosine", -1.0),
        (minimum_margin, "minimum_margin", 0.0),
    ):
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or not lower <= float(value) <= 1.0:
            raise ValueError(f"{name} is outside its valid interval")
    owner_ids = tuple(candidate.owner_entity_id for candidate in candidates)
    if len(owner_ids) != len(set(owner_ids)):
        raise ValueError("memory candidates must have unique OVI owner IDs")
    matches = tuple(
        match
        for candidate in candidates
        if (
            match := _candidate_match(
                index,
                candidate,
                mode=retrieval_mode,
                minimum_cosine=float(minimum_cosine),
                minimum_margin=float(minimum_margin),
            )
        )
        is not None
    )
    selected: list[_MemoryMatch] = []
    used_stable_ids: set[str] = set()
    for match in sorted(
        matches,
        key=lambda item: (
            -item.cosine,
            -item.margin,
            item.entry.stable_entity_id,
            item.candidate.owner_entity_id,
        ),
    ):
        if match.entry.stable_entity_id in used_stable_ids:
            continue
        used_stable_ids.add(match.entry.stable_entity_id)
        selected.append(match)
    relations = []
    for match in sorted(selected, key=lambda item: item.candidate.owner_entity_id):
        entry = match.entry
        candidate = match.candidate
        relations.append(
            RelationSupport(
                relation_id=(
                    f"crove-memory:{candidate.projection_version}:"
                    f"{entry.stable_entity_id}:{candidate.owner_entity_id}"
                ),
                relation_source="crove-memory",
                stable_entity_id=entry.stable_entity_id,
                t0_owner_entity_id=entry.owner_entity_id,
                t1_owner_entity_id=candidate.owner_entity_id,
                t0_source_surface_id=entry.source_surface_id,
                t1_source_surface_id=candidate.source_surface_id,
                t0_source_vertex_indices=entry.source_vertex_indices,
                t1_source_vertex_indices=candidate.source_vertex_indices,
                confidence=float(np.clip((match.cosine + 1.0) / 2.0, 0.0, 1.0)),
                inference_state=RelationInferenceState.UNRESOLVED,
                accepted=True,
                t0_evidence_frame_ids=entry.accepted_frame_ids,
                t1_evidence_frame_ids=(candidate.frame_id,),
                t0_entity_id=f"ovimap:{entry.owner_entity_id}",
                t1_entity_id=candidate.entity_id,
                assignment_margin=match.margin,
                appearance_cosine=match.cosine,
                appearance_feature_space=entry.feature_space_id,
                appearance_projection_version=entry.projection_version,
                competition_margin=match.margin,
                memory_retrieval_mode=retrieval_mode.value,
                memory_identity_lifecycle=entry.lifecycle_state.value,
                observation_provenance=",".join(entry.observation_provenance),
            )
        )
    return tuple(relations)


__all__ = [
    "OFFLINE_REPLAY_ON_FINAL_OVI_SUPPORT",
    "ONLINE_CAUSAL_OBSERVATION",
    "CroveMemoryEntry",
    "CroveMemoryIndex",
    "MemoryCandidateObservation",
    "MemoryEntityObservation",
    "MemoryRetrievalMode",
    "build_memory_relation_support",
]
