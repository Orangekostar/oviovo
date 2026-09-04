from __future__ import annotations

import numpy as np
import pytest

from src.oviv2.temporal_pair_reasoner import (
    TemporalPairReasoner,
    validate_query_evidence,
)
from src.oviv2.two_visit_contracts import NeuralSampleMap, TemporalQueryEvidence


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _pair() -> NeuralSampleMap:
    return NeuralSampleMap(
        coordinates_xyzt=np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        features=np.ones((2, 3), dtype=np.float32),
        visit_ids=np.array([0, 1], dtype=np.int8),
        source_visit_ids=np.array([0, 1], dtype=np.int8),
        source_entity_ids=("t0:a", "t1:a"),
        source_point_indices=np.array([0, 1], dtype=np.int64),
        source_to_token_offsets=np.array([0, 1, 2], dtype=np.int64),
        neural_voxel_size_m=0.02,
        feature_schema="rgb",
        coordinate_frame_id="world",
        source_manifest_sha256=SHA_A,
        source_visit_map_sha256=(SHA_B, SHA_C),
    )


class _Reasoner:
    def infer(self, pair: NeuralSampleMap) -> TemporalQueryEvidence:
        return TemporalQueryEvidence(
            status="PASS",
            backend_name="fixture",
            backend_config_sha256=SHA_A,
            pair_sha256=pair.content_sha256(),
            temporal_query_ids=("q0",),
            query_masks=np.array([[True, True]]),
            token_scores=np.array([[1.0, 1.0]], dtype=np.float32),
            query_scores=np.array([1.0], dtype=np.float32),
            checkpoint_sha256=None,
            ranking_eligible=True,
            runtime_s=0.0,
            peak_memory_bytes=0,
        )


def test_reasoner_protocol_and_pair_bound_evidence() -> None:
    reasoner: TemporalPairReasoner = _Reasoner()
    pair = _pair()

    evidence = reasoner.infer(pair)
    validate_query_evidence(pair, evidence)

    assert isinstance(reasoner, TemporalPairReasoner)


def test_evidence_rejects_wrong_pair_hash_or_token_count() -> None:
    pair = _pair()
    evidence = _Reasoner().infer(pair)
    wrong_hash = TemporalQueryEvidence(
        status="PASS",
        backend_name=evidence.backend_name,
        backend_config_sha256=evidence.backend_config_sha256,
        pair_sha256="f" * 64,
        temporal_query_ids=evidence.temporal_query_ids,
        query_masks=evidence.query_masks,
        token_scores=evidence.token_scores,
        query_scores=evidence.query_scores,
        checkpoint_sha256=None,
        ranking_eligible=True,
        runtime_s=0.0,
        peak_memory_bytes=0,
    )
    with pytest.raises(ValueError, match="pair SHA-256"):
        validate_query_evidence(pair, wrong_hash)

    wrong_width = TemporalQueryEvidence(
        status="PASS",
        backend_name=evidence.backend_name,
        backend_config_sha256=evidence.backend_config_sha256,
        pair_sha256=pair.content_sha256(),
        temporal_query_ids=("q0",),
        query_masks=np.ones((1, 3), dtype=bool),
        token_scores=np.ones((1, 3), dtype=np.float32),
        query_scores=np.ones(1, dtype=np.float32),
        checkpoint_sha256=None,
        ranking_eligible=True,
        runtime_s=0.0,
        peak_memory_bytes=0,
    )
    with pytest.raises(ValueError, match="token count"):
        validate_query_evidence(pair, wrong_width)
