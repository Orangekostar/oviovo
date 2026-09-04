"""Backend-neutral interface for two-visit temporal query reasoning."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.oviv2.two_visit_contracts import NeuralSampleMap, TemporalQueryEvidence


@runtime_checkable
class TemporalPairReasoner(Protocol):
    def infer(self, pair: NeuralSampleMap) -> TemporalQueryEvidence:
        """Infer source-bound temporal query evidence for one visit pair."""


def validate_query_evidence(
    pair: NeuralSampleMap, evidence: TemporalQueryEvidence
) -> None:
    if not isinstance(pair, NeuralSampleMap):
        raise TypeError("pair must be a NeuralSampleMap")
    if not isinstance(evidence, TemporalQueryEvidence):
        raise TypeError("evidence must be TemporalQueryEvidence")
    if evidence.pair_sha256 != pair.content_sha256():
        raise ValueError("temporal evidence pair SHA-256 mismatch")
    if evidence.query_masks is not None and evidence.query_masks.shape[1] != len(
        pair.visit_ids
    ):
        raise ValueError("temporal query token count does not match pair")
