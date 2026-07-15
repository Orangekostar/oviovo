from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VisibilityKind(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    OCCLUDED = "occluded"
    UNOBSERVED = "unobserved"


@dataclass(frozen=True)
class VisibilityEvidence:
    entity_id: str
    frame_id: int
    kind: VisibilityKind
    projected_count: int
    valid_depth_count: int
    present_count: int
    absent_count: int
    occluded_count: int
    unobserved_count: int
    reason: str = ""

    def __post_init__(self) -> None:
        if not str(self.entity_id):
            raise ValueError("entity_id must be non-empty")
        counts = (
            self.projected_count,
            self.valid_depth_count,
            self.present_count,
            self.absent_count,
            self.occluded_count,
            self.unobserved_count,
        )
        if any(int(value) < 0 for value in counts):
            raise ValueError("visibility counts must be non-negative")
        if int(self.valid_depth_count) > int(self.projected_count):
            raise ValueError("valid_depth_count cannot exceed projected_count")


@dataclass(frozen=True)
class VisibilityEvidenceBatch:
    frame_id: int
    evidence: tuple[VisibilityEvidence, ...] = ()

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        if any(item.frame_id != int(self.frame_id) for item in evidence):
            raise ValueError("all visibility evidence must match batch frame_id")
        entity_ids = [item.entity_id for item in evidence]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("visibility batch must contain at most one record per entity")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "evidence", evidence)

    def by_entity(self, entity_id: str) -> VisibilityEvidence | None:
        return next((item for item in self.evidence if item.entity_id == entity_id), None)
