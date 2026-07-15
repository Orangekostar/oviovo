from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.domain.arrays import readonly_array


class EntityLifecycleState(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    REMOVED = "removed"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class LifecycleInterval:
    state: EntityLifecycleState
    start_timestamp: float
    end_timestamp: float | None

    def __post_init__(self) -> None:
        start = float(self.start_timestamp)
        end = None if self.end_timestamp is None else float(self.end_timestamp)
        if not np.isfinite(start) or (end is not None and (not np.isfinite(end) or end < start)):
            raise ValueError("invalid lifecycle interval")
        object.__setattr__(self, "start_timestamp", start)
        object.__setattr__(self, "end_timestamp", end)


@dataclass(frozen=True)
class PersistentEntity:
    entity_id: str
    state: EntityLifecycleState
    first_seen: float
    last_seen: float
    semantic_label: str
    semantic_confidence: float
    semantic_embedding: np.ndarray | None
    identity_embedding: np.ndarray | None
    geometry_handle: str
    ownership_revision: int
    lifecycle_intervals: tuple[LifecycleInterval, ...]

    def __post_init__(self) -> None:
        if not str(self.entity_id) or not str(self.geometry_handle):
            raise ValueError("entity_id and geometry_handle must be non-empty")
        first_seen = float(self.first_seen)
        last_seen = float(self.last_seen)
        if not np.isfinite([first_seen, last_seen]).all() or last_seen < first_seen:
            raise ValueError("invalid entity observation interval")
        confidence = float(self.semantic_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("semantic_confidence must be in [0, 1]")
        intervals = tuple(self.lifecycle_intervals)
        if not intervals or intervals[-1].state is not self.state:
            raise ValueError("last lifecycle interval must represent current entity state")
        if intervals[-1].end_timestamp is not None:
            raise ValueError("current lifecycle interval must be open")
        object.__setattr__(self, "first_seen", first_seen)
        object.__setattr__(self, "last_seen", last_seen)
        object.__setattr__(self, "semantic_label", str(self.semantic_label).strip())
        object.__setattr__(self, "semantic_confidence", confidence)
        object.__setattr__(self, "ownership_revision", int(self.ownership_revision))
        object.__setattr__(self, "lifecycle_intervals", intervals)
        if self.semantic_embedding is not None:
            object.__setattr__(
                self,
                "semantic_embedding",
                readonly_array(self.semantic_embedding, dtype=np.float32, ndim=1),
            )
        if self.identity_embedding is not None:
            object.__setattr__(
                self,
                "identity_embedding",
                readonly_array(self.identity_embedding, dtype=np.float32, ndim=1),
            )


@dataclass(frozen=True)
class LifecycleTransition:
    entity_id: str
    from_state: EntityLifecycleState
    to_state: EntityLifecycleState
    timestamp: float
    reason: str
    release_ownership: bool = False

    def __post_init__(self) -> None:
        if not self.entity_id or self.from_state is self.to_state or not self.reason:
            raise ValueError("lifecycle transition must change one identified entity for a reason")
        if not np.isfinite(float(self.timestamp)):
            raise ValueError("transition timestamp must be finite")


@dataclass(frozen=True)
class LifecycleDelta:
    base_revision: int
    transitions: tuple[LifecycleTransition, ...] = ()

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        entity_ids = [item.entity_id for item in transitions]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("one lifecycle delta may transition each entity at most once")
        object.__setattr__(self, "base_revision", int(self.base_revision))
        object.__setattr__(self, "transitions", transitions)
