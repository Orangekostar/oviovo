from __future__ import annotations

import numpy as np
import pytest

from src.oviv2.semantic_fusion import SemanticFusionConfig, fuse_semantics


def test_dense_only_normalizes_support_and_uses_lowest_id_for_ties() -> None:
    fused = fuse_semantics(((3, 2.0), (1, 2.0)), (), 0.0)

    assert fused.semantic_id == 1
    assert fused.confidence == pytest.approx(0.5)
    assert fused.entity_weight == pytest.approx(0.0)


def test_entity_only_uses_posterior_when_owned_voxel_has_no_dense_evidence() -> None:
    fused = fuse_semantics((), ((2, 0.8), (4, 0.2)), 0.75)

    assert fused.semantic_id == 2
    assert fused.confidence == pytest.approx(0.8)
    assert fused.entity_weight == pytest.approx(0.3)


def test_uncertain_dense_distribution_can_be_corrected_by_entity_posterior() -> None:
    fused = fuse_semantics(
        ((1, 0.51), (2, 0.49)),
        ((2, 1.0),),
        1.0,
    )

    assert fused.semantic_id == 2
    assert fused.confidence == pytest.approx(0.61495)
    assert fused.entity_weight == pytest.approx(0.245)


def test_confident_dense_distribution_resists_disagreeing_entity() -> None:
    fused = fuse_semantics(
        ((1, 3.0), (2, 1.0)),
        ((2, 0.8), (3, 0.2)),
        1.0,
    )

    assert fused.semantic_id == 1
    assert fused.confidence == pytest.approx(0.675)
    assert fused.entity_weight == pytest.approx(0.1)


def test_agreeing_distributions_increase_winner_support() -> None:
    fused = fuse_semantics(
        ((2, 0.6), (1, 0.4)),
        ((2, 0.9), (3, 0.1)),
        1.0,
    )

    assert fused.semantic_id == 2
    assert fused.confidence == pytest.approx(0.654)
    assert fused.entity_weight == pytest.approx(0.18)


def test_empty_distributions_remain_unknown() -> None:
    fused = fuse_semantics((), (), 1.0)

    assert fused.semantic_id == 0
    assert fused.confidence == pytest.approx(0.0)
    assert fused.entity_weight == pytest.approx(0.0)


@pytest.mark.parametrize("scale", [False, np.nan, -0.01, 1.01])
def test_config_rejects_invalid_entity_weight_scale(scale: object) -> None:
    with pytest.raises((TypeError, ValueError), match="entity_weight_scale"):
        SemanticFusionConfig(entity_weight_scale=scale)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("dense_support", "entity_probabilities", "ownership_confidence", "message"),
    [
        (((0, 1.0),), (), 0.0, "semantic ID"),
        (((1, 1.0), (1, 2.0)), (), 0.0, "unique"),
        (((1, -1.0),), (), 0.0, "support"),
        ((), ((1, 0.6), (2, 0.3)), 1.0, "sum to one"),
        ((), ((1, np.nan),), 1.0, "probability"),
        ((), ((1, 0.6), (np.iinfo(np.int64).max + 1, 0.4)), 1.0, "int64"),
        ((), ((1, 1.0),), 1.01, "ownership_confidence"),
    ],
)
def test_fusion_inputs_are_strictly_validated(
    dense_support: object,
    entity_probabilities: object,
    ownership_confidence: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        fuse_semantics(
            dense_support,  # type: ignore[arg-type]
            entity_probabilities,  # type: ignore[arg-type]
            ownership_confidence,  # type: ignore[arg-type]
        )
