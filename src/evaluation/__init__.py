"""Neutral evaluation contracts and metrics for OVIOVO and external baselines."""

from importlib import import_module
from typing import Any

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

_OVIV2_REPLICA_EXPORTS = {
    "EntityEvaluationInfo",
    "ProjectedLabels",
    "ReplicaGroundTruth",
    "evaluate_replica_voxel_map",
    "project_mesh_to_gt",
}


def __getattr__(name: str) -> Any:
    if name not in _OVIV2_REPLICA_EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("src.evaluation.oviv2_replica"), name)
    globals()[name] = value
    return value

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
