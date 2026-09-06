from __future__ import annotations

import numpy as np
import pytest

from src.oviv2.query_entity_resolver import (
    ResolverConfig,
    ResolverDomainCoverage,
    ResolverError,
    resolve_supported_queries,
)
from src.oviv2.rescene_supported_view import build_supported_inference_view
from src.oviv2.two_visit_contracts import TemporalQueryEvidence
from tests.oviv2.test_rescene_supported_view import _prepared, _support


def _evidence(pair_sha256: str) -> TemporalQueryEvidence:
    masks = np.asarray(
        [
            [True, True, False, False],
            [True, False, True, True],
            [False, False, True, True],
            [False, False, True, False],
        ],
        dtype=np.bool_,
    )
    scores = np.where(masks, 0.9, 0.1).astype(np.float32)
    return TemporalQueryEvidence(
        status="PASS",
        backend_name="rescene:concerto",
        backend_config_sha256="e" * 64,
        pair_sha256=pair_sha256,
        temporal_query_ids=("q0", "q1", "q2", "q3"),
        query_masks=masks,
        token_scores=scores,
        query_scores=np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        checkpoint_sha256="f" * 64,
        ranking_eligible=True,
        runtime_s=1.0,
        peak_memory_bytes=1024,
    )


def test_supported_resolver_uses_full_denominators_and_mutual_dominance() -> None:
    view = build_supported_inference_view(_prepared(), _support())
    result = resolve_supported_queries(
        pair=view.pair,
        evidence=_evidence(view.pair.content_sha256()),
        entity_coverage=view.entity_coverage,
        adapter_to_model=np.asarray([0, 1, 1, 2], dtype=np.int64),
        domain_coverage=ResolverDomainCoverage(
            full_source_count=8,
            supported_source_count=6,
            full_adapter_count=6,
            supported_adapter_count=4,
            full_model_count=4,
            supported_model_count=3,
        ),
        config=ResolverConfig(
            minimum_supported_entity_coverage=0.75,
            minimum_query_precision=0.5,
            minimum_query_confidence=0.5,
            static_centroid_tolerance_m=0.1,
        ),
    )

    assert len(result.relations) == 1
    relation = result.relations[0]
    assert relation.temporal_query_id == "q0"
    assert relation.t0_entity_ids == ("t0-chair",)
    assert relation.t1_entity_ids == ("t1-chair",)
    assert relation.state == "persistent_static"
    record = result.evidence[("q0", 0, "t0-chair")]
    assert record.supported_entity_token_coverage == 1.0
    assert record.full_entity_token_evidence == 0.5
    assert record.query_to_entity_precision == 1.0
    assert record.supported_source_point_coverage == 1.0
    assert record.full_source_point_evidence == pytest.approx(2 / 3)
    assert result.conflict_query_ids == ("q1",)
    assert result.one_sided_query_ids == ("q2",)
    assert result.fallback_query_ids == ("q3",)
    assert result.collision_model_count == 1
    assert result.duplicate_topology_count == 0
    assert result.input_coverage == {
        "source_fraction": 0.75,
        "adapter_fraction": 2 / 3,
        "model_fraction": 0.75,
        "entity_fraction": 0.75,
    }
    assert result.conditional_query_coverage == 0.25


def test_supported_resolver_rejects_missing_full_entity_denominator() -> None:
    view = build_supported_inference_view(_prepared(), _support())

    with pytest.raises(ResolverError, match="entity coverage"):
        resolve_supported_queries(
            pair=view.pair,
            evidence=_evidence(view.pair.content_sha256()),
            entity_coverage=view.entity_coverage[:-1],
            adapter_to_model=view.model_input.adapter_to_model,
            domain_coverage=ResolverDomainCoverage(
                full_source_count=8,
                supported_source_count=6,
                full_adapter_count=6,
                supported_adapter_count=4,
                full_model_count=4,
                supported_model_count=3,
            ),
            config=ResolverConfig(),
        )


def test_supported_resolver_breaks_equal_scores_by_stable_query_id() -> None:
    view = build_supported_inference_view(_prepared(), _support())
    evidence = _evidence(view.pair.content_sha256())
    tied = TemporalQueryEvidence(
        status=evidence.status,
        backend_name=evidence.backend_name,
        backend_config_sha256=evidence.backend_config_sha256,
        pair_sha256=evidence.pair_sha256,
        temporal_query_ids=evidence.temporal_query_ids,
        query_masks=evidence.query_masks,
        token_scores=evidence.token_scores,
        query_scores=np.asarray([0.8, 0.8, 0.7, 0.6], dtype=np.float32),
        checkpoint_sha256=evidence.checkpoint_sha256,
        ranking_eligible=True,
        runtime_s=evidence.runtime_s,
        peak_memory_bytes=evidence.peak_memory_bytes,
    )

    result = resolve_supported_queries(
        pair=view.pair,
        evidence=tied,
        entity_coverage=view.entity_coverage,
        adapter_to_model=view.model_input.adapter_to_model,
        domain_coverage=ResolverDomainCoverage(8, 6, 6, 4, 4, 3),
        config=ResolverConfig(
            minimum_supported_entity_coverage=0.75,
            minimum_query_precision=0.5,
            minimum_query_confidence=0.5,
        ),
    )

    assert [relation.temporal_query_id for relation in result.relations] == ["q0"]
