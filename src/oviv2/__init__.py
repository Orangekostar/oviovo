"""Voxel-first OVIV2 mapping primitives."""

from src.oviv2.addressing import BlockKey, VoxelKey, join_voxel_key, point_to_voxel, split_voxel_key
from src.oviv2.evidence import (
    EntityCandidate,
    EvidenceConfig,
    SemanticCandidate,
    SparseEvidenceStore,
)
from src.oviv2.entities import (
    EntityRegistry,
    EntityRegistryConfig,
    PersistentEntity,
)
from src.oviv2.geometry import SparseTsdfVolume, TsdfConfig
from src.oviv2.meshing import LabeledMesh, derive_labeled_mesh, write_labeled_mesh
from src.oviv2.observations import (
    CachedFrontendAdapter,
    FrameObservation,
    ObservationKind,
    ReplicaVocabulary,
)
from src.oviv2.ownership import OwnershipRecord, ReversibleOwnershipStore
from src.oviv2.runtime import Oviv2Runtime, Oviv2RuntimeConfig, RuntimeFrameResult
from src.oviv2.snapshot import VoxelMapSnapshot, VoxelSnapshotMetadata
from src.oviv2.tracking import LocalTrack, LocalTrackBatch, LocalTracker, LocalTrackerConfig

__all__ = [
    "BlockKey",
    "CachedFrontendAdapter",
    "EntityCandidate",
    "EntityRegistry",
    "EntityRegistryConfig",
    "EvidenceConfig",
    "FrameObservation",
    "LabeledMesh",
    "LocalTrack",
    "LocalTrackBatch",
    "LocalTracker",
    "LocalTrackerConfig",
    "OwnershipRecord",
    "ObservationKind",
    "Oviv2Runtime",
    "Oviv2RuntimeConfig",
    "ReplicaVocabulary",
    "RuntimeFrameResult",
    "PersistentEntity",
    "ReversibleOwnershipStore",
    "SemanticCandidate",
    "SparseEvidenceStore",
    "SparseTsdfVolume",
    "TsdfConfig",
    "VoxelKey",
    "VoxelMapSnapshot",
    "VoxelSnapshotMetadata",
    "derive_labeled_mesh",
    "join_voxel_key",
    "point_to_voxel",
    "split_voxel_key",
    "write_labeled_mesh",
]
