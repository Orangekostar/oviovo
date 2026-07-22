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
from src.oviv2.meshing import (
    LabeledMesh,
    canonicalize_labeled_mesh,
    derive_labeled_mesh,
    write_labeled_mesh,
)
from src.oviv2.observations import (
    CachedFrontendAdapter,
    FrameObservation,
    ObservationKind,
    ReplicaVocabulary,
    lift_mask_to_voxels,
)
from src.oviv2.ownership import OwnershipRecord, ReversibleOwnershipStore
from src.oviv2.runtime import Oviv2Runtime, Oviv2RuntimeConfig, RuntimeFrameResult
from src.oviv2.semantic_memory import (
    FeaturePrototype,
    FeaturePrototypeBank,
    InformativeView,
    InformativeViewBank,
    SparseClassPosterior,
)
from src.oviv2.snapshot import (
    SnapshotPublicationUncertainError,
    VoxelMapSnapshot,
    VoxelSnapshotMetadata,
)
from src.oviv2.structure import DepthStructureConfig, DepthStructureFrontend
from src.oviv2.tracking import LocalTrack, LocalTrackBatch, LocalTracker, LocalTrackerConfig
from src.oviv2.visibility import VisibilityConfig, VisibilityStatus, VoxelVisibilityProjector

__all__ = [
    "BlockKey",
    "CachedFrontendAdapter",
    "DepthStructureConfig",
    "DepthStructureFrontend",
    "EntityCandidate",
    "EntityRegistry",
    "EntityRegistryConfig",
    "EvidenceConfig",
    "FrameObservation",
    "FeaturePrototype",
    "FeaturePrototypeBank",
    "InformativeView",
    "InformativeViewBank",
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
    "SparseClassPosterior",
    "SparseTsdfVolume",
    "SnapshotPublicationUncertainError",
    "TsdfConfig",
    "VoxelKey",
    "VoxelMapSnapshot",
    "VisibilityConfig",
    "VisibilityStatus",
    "VoxelVisibilityProjector",
    "VoxelSnapshotMetadata",
    "canonicalize_labeled_mesh",
    "derive_labeled_mesh",
    "join_voxel_key",
    "lift_mask_to_voxels",
    "point_to_voxel",
    "split_voxel_key",
    "write_labeled_mesh",
]
