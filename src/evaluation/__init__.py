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

__all__ = [
    "BaselineRunner",
    "DatasetAdapter",
    "EntityPrediction",
    "FramePacket",
    "GroundTruthSequence",
    "GroundTruthSnapshot",
    "MapSnapshot",
    "QueryRequest",
    "QueryResult",
    "QueryTarget",
    "RunMetadata",
    "UnifiedEvaluator",
]
