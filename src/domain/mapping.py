from __future__ import annotations

from dataclasses import dataclass


VoxelKey = tuple[int, int, int]


def _voxel_key(value: VoxelKey) -> VoxelKey:
    if len(value) != 3:
        raise ValueError("voxel key must contain three integers")
    return tuple(int(item) for item in value)


@dataclass(frozen=True)
class EntityVoxelEvidence:
    entity_id: str
    positive_support: float
    negative_support: float
    last_timestamp: float

    def __post_init__(self) -> None:
        if not self.entity_id or self.positive_support < 0.0 or self.negative_support < 0.0:
            raise ValueError("entity voxel evidence is invalid")


@dataclass(frozen=True)
class VoxelEvidence:
    voxel_key: VoxelKey
    entity_evidence: tuple[EntityVoxelEvidence, ...]
    background_support: float
    source_observation_ids: tuple[str, ...]
    revision: int

    def __post_init__(self) -> None:
        evidence = tuple(self.entity_evidence)
        entity_ids = [item.entity_id for item in evidence]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("voxel evidence may contain one record per entity")
        if float(self.background_support) < 0.0:
            raise ValueError("background_support must be non-negative")
        object.__setattr__(self, "voxel_key", _voxel_key(self.voxel_key))
        object.__setattr__(self, "entity_evidence", evidence)
        object.__setattr__(self, "source_observation_ids", tuple(self.source_observation_ids))
        object.__setattr__(self, "revision", int(self.revision))


@dataclass(frozen=True)
class VoxelEvidenceUpdate:
    voxel_key: VoxelKey
    entity_id: str
    positive_delta: float
    negative_delta: float
    background_delta: float
    observation_id: str
    timestamp: float

    def __post_init__(self) -> None:
        if not self.entity_id or not self.observation_id:
            raise ValueError("voxel evidence update requires entity and observation")
        if min(self.positive_delta, self.negative_delta, self.background_delta) < 0.0:
            raise ValueError("voxel evidence deltas must be non-negative")
        object.__setattr__(self, "voxel_key", _voxel_key(self.voxel_key))


@dataclass(frozen=True)
class GeometryDelta:
    base_revision: int
    touched_voxel_keys: tuple[VoxelKey, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "touched_voxel_keys", tuple(_voxel_key(key) for key in self.touched_voxel_keys))


@dataclass(frozen=True)
class VoxelEvidenceDelta:
    base_revision: int
    updates: tuple[VoxelEvidenceUpdate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "updates", tuple(self.updates))


@dataclass(frozen=True)
class FusionDelta:
    geometry: GeometryDelta
    voxel_evidence: VoxelEvidenceDelta


@dataclass(frozen=True)
class VoxelOwnership:
    voxel_key: VoxelKey
    owner_entity_id: str | None
    confidence: float
    epoch: int
    evidence_revision: int

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("ownership confidence must be in [0, 1]")
        object.__setattr__(self, "voxel_key", _voxel_key(self.voxel_key))


@dataclass(frozen=True)
class OwnershipLayer:
    revision: int
    assignments: tuple[VoxelOwnership, ...]

    def __post_init__(self) -> None:
        assignments = tuple(self.assignments)
        keys = [item.voxel_key for item in assignments]
        if len(keys) != len(set(keys)):
            raise ValueError("ownership layer must contain one assignment per voxel")
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "assignments", assignments)

    def owner_of(self, voxel_key: VoxelKey) -> str | None:
        key = _voxel_key(voxel_key)
        item = next((value for value in self.assignments if value.voxel_key == key), None)
        return None if item is None else item.owner_entity_id


@dataclass(frozen=True)
class OwnershipTransition:
    voxel_key: VoxelKey
    previous_owner_entity_id: str | None
    new_owner_entity_id: str | None
    confidence: float
    evidence_revision: int
    reason: str

    def __post_init__(self) -> None:
        if not self.reason or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("ownership transition requires a reason and bounded confidence")
        object.__setattr__(self, "voxel_key", _voxel_key(self.voxel_key))


@dataclass(frozen=True)
class OwnershipDelta:
    base_revision: int
    transitions: tuple[OwnershipTransition, ...] = ()

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        keys = [item.voxel_key for item in transitions]
        if len(keys) != len(set(keys)):
            raise ValueError("one ownership transition is allowed per voxel in a delta")
        object.__setattr__(self, "base_revision", int(self.base_revision))
        object.__setattr__(self, "transitions", transitions)
