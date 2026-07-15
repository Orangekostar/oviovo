from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssociationKind(str, Enum):
    ACTIVE_MATCH = "active_match"
    DORMANT_REID = "dormant_reid"
    CREATE = "create"
    DEFER = "defer"
    REJECT = "reject"


@dataclass(frozen=True)
class AssociationDecision:
    local_track_id: str
    kind: AssociationKind
    entity_id: str | None
    score: float
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.local_track_id:
            raise ValueError("local_track_id must be non-empty")
        score = float(self.score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("association score must be in [0, 1]")
        requires_entity = self.kind in {AssociationKind.ACTIVE_MATCH, AssociationKind.DORMANT_REID}
        forbids_entity = self.kind in {AssociationKind.CREATE, AssociationKind.DEFER, AssociationKind.REJECT}
        if requires_entity and not self.entity_id:
            raise ValueError("match and re-id decisions require entity_id")
        if forbids_entity and self.entity_id is not None:
            raise ValueError("create, defer, and reject decisions cannot bind entity_id")
        object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class AssociationDecisionBatch:
    frame_id: int
    decisions: tuple[AssociationDecision, ...] = ()

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        track_ids = [item.local_track_id for item in decisions]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("one association decision is allowed per local track")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "decisions", decisions)


@dataclass(frozen=True)
class EntityBinding:
    local_track_id: str
    entity_id: str
    observation_ids: tuple[str, ...]
    association_kind: AssociationKind

    def __post_init__(self) -> None:
        observation_ids = tuple(str(value) for value in self.observation_ids)
        if not self.local_track_id or not self.entity_id or not observation_ids:
            raise ValueError("entity binding requires track, entity, and observations")
        if self.association_kind in {AssociationKind.DEFER, AssociationKind.REJECT}:
            raise ValueError("deferred or rejected tracks cannot produce entity bindings")
        object.__setattr__(self, "observation_ids", observation_ids)


@dataclass(frozen=True)
class EntityResolutionBatch:
    frame_id: int
    bindings: tuple[EntityBinding, ...] = ()

    def __post_init__(self) -> None:
        bindings = tuple(self.bindings)
        track_ids = [item.local_track_id for item in bindings]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("one resolved binding is allowed per local track")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "bindings", bindings)
