from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.ovi_ownership_completion import (
    OwnershipCompletionError,
    build_ownership_readout,
    evaluate_completion_surface,
)
from src.evaluation.rscan_method_views import build_method_pair_view
from src.oviv2.two_visit_contracts import PairRelation


def _sample():
    def processed(offset: float) -> np.ndarray:
        return np.asarray(
            [
                [0.00 + offset, 0.0, 0.0, 0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 4, 1, 10],
                [0.01 + offset, 0.0, 0.0, 0.3, 0.4, 0.5, 1.0, 0.0, 0.0, 4, 1, 10],
                [1.00 + offset, 0.0, 0.0, 0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 8, 1, 20],
            ],
            dtype=np.float32,
        )

    return build_method_pair_view(
        pair_id="pair",
        scan_ids=("a", "b"),
        processed_visits=(processed(0.0), processed(0.01)),
        source_manifest_sha256="a" * 64,
        domain_id="D0_NATIVE_PROCESSED",
    ).geometric_sample(neural_voxel_size_m=0.02)


def _relation() -> PairRelation:
    return PairRelation(
        temporal_query_id="q0",
        t0_entity_ids=("segment:000004",),
        t1_entity_ids=("segment:000004",),
        state="persistent_static",
        query_confidence=0.9,
        evidence={"query_score": 0.9},
        identity_source="rescene",
    )


def test_identity_readout_groups_entities_without_relocating_geometry() -> None:
    sample = _sample()
    baseline = build_ownership_readout(sample, (), variant_id="A0")
    temporal = build_ownership_readout(sample, (_relation(),), variant_id="A2")

    assert baseline["geometry_sha256"] == temporal["geometry_sha256"]
    assert baseline["entity_geometry_sha256"] == temporal["entity_geometry_sha256"]
    assert baseline["identity_group_count"] == 4
    assert temporal["cross_visit_identity_group_count"] == 1
    assert temporal["identity_group_count"] == 3
    assert temporal["geometry_mutation_count"] == 0


def test_identity_readout_rejects_unknown_or_multiply_owned_entities() -> None:
    sample = _sample()
    unknown = PairRelation(
        temporal_query_id="bad",
        t0_entity_ids=("missing",),
        t1_entity_ids=("segment:000004",),
        state="persistent_static",
        query_confidence=0.8,
        evidence={"query_score": 0.8},
        identity_source="rescene",
    )
    with pytest.raises(OwnershipCompletionError, match="unknown"):
        build_ownership_readout(sample, (unknown,), variant_id="A2")
    with pytest.raises(OwnershipCompletionError, match="multiple"):
        build_ownership_readout(sample, (_relation(), _relation()), variant_id="A2")


def test_completion_surface_uses_only_actual_historical_and_target_shapes() -> None:
    baseline = np.asarray([[0.01, 0.0, 0.0]], dtype=np.float32)
    historical = np.asarray(
        [[0.01, 0.0, 0.0], [0.11, 0.0, 0.0], [0.21, 0.0, 0.0]],
        dtype=np.float32,
    )
    target = np.asarray(
        [[0.01, 0.0, 0.0], [0.11, 0.0, 0.0]], dtype=np.float32
    )
    recovered = np.asarray(
        [[0.01, 0.0, 0.0], [0.11, 0.0, 0.0], [0.31, 0.0, 0.0]],
        dtype=np.float32,
    )

    result = evaluate_completion_surface(
        baseline_xyz=baseline,
        recovered_xyz=recovered,
        target_xyz=target,
        historical_candidate_xyz=historical,
        voxel_size_m=0.05,
    )

    assert result["opportunity_voxel_count"] == 1
    assert result["recoverable_voxel_count"] == 1
    assert result["recovered_new_voxel_count"] == 2
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f_score"] == pytest.approx(2 / 3)
