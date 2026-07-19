"""Neutral evaluation contracts and metrics for OVIOVO and external baselines."""

from src.evaluation.contracts import (
    BaselineRunner,
    DatasetAdapter,
    EntityPrediction,
    FramePacket,
    GroundTruthSequence,
    GroundTruthSnapshot,
    MapSnapshot,
    QueryRequest,
    QueryResult,
    QueryTarget,
    RunMetadata,
    UnifiedEvaluator,
)
from src.evaluation.oviv2_replica import (
    EntityEvaluationInfo,
    ProjectedLabels,
    ReplicaGroundTruth,
    evaluate_replica_voxel_map,
    project_mesh_to_gt,
)

__all__ = [
    "BaselineRunner",
    "DatasetAdapter",
    "EntityPrediction",
    "EntityEvaluationInfo",
    "FramePacket",
    "GroundTruthSequence",
    "GroundTruthSnapshot",
    "MapSnapshot",
    "ProjectedLabels",
    "QueryRequest",
    "QueryResult",
    "QueryTarget",
    "ReplicaGroundTruth",
    "RunMetadata",
    "UnifiedEvaluator",
    "evaluate_replica_voxel_map",
    "project_mesh_to_gt",
]
