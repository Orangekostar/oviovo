"""Neutral contracts and adapters for externally maintained baselines."""

from src.evaluation.baselines.contracts import (
    BaselineArtifact,
    BaselineMetadata,
    FrozenUpdateAudit,
    RuntimeBreakdown,
)
from src.evaluation.baselines.aggregation import aggregate_replica_static
from src.evaluation.baselines.result_manifest import finalize_static_result
from src.evaluation.baselines.static_metrics import evaluate_static_snapshot

__all__ = [
    "BaselineArtifact",
    "BaselineMetadata",
    "FrozenUpdateAudit",
    "RuntimeBreakdown",
    "aggregate_replica_static",
    "evaluate_static_snapshot",
    "finalize_static_result",
]
