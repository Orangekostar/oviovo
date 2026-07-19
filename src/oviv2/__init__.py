"""Voxel-first OVIV2 mapping primitives."""

from src.oviv2.addressing import BlockKey, VoxelKey, join_voxel_key, point_to_voxel, split_voxel_key
from src.oviv2.evidence import (
    EntityCandidate,
    EvidenceConfig,
    SemanticCandidate,
    SparseEvidenceStore,
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
from src.oviv2.snapshot import VoxelMapSnapshot, VoxelSnapshotMetadata

__all__ = [
    "BlockKey",
    "CachedFrontendAdapter",
    "EntityCandidate",
    "EvidenceConfig",
    "FrameObservation",
    "LabeledMesh",
    "OwnershipRecord",
    "ObservationKind",
    "ReplicaVocabulary",
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
