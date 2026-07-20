from __future__ import annotations

import pytest

from scripts.evaluation.finalize_oviv2_replica_result import _evaluation_contract


def test_legacy_batch_defaults_to_owner_evaluation_directory() -> None:
    assert _evaluation_contract({}, {}) == ("owner_authoritative", "evaluation", 0.5)


def test_stage3_batch_selects_fused_directory_and_frozen_weight() -> None:
    assert _evaluation_contract(
        {"semantic_head": "fused_uncertainty"},
        {
            "dense_semantic_mode": "cached_probabilities",
            "fusion_semantic_mode": "uncertainty_linear",
            "fusion_entity_weight_scale": 0.5,
        },
    ) == ("fused_uncertainty", "evaluation_fused", 0.5)


@pytest.mark.parametrize("head", ["dense_only", "fused_uncertainty", "unknown"])
def test_nonlegacy_head_requires_matching_scene_config(head: str) -> None:
    with pytest.raises(ValueError, match="semantic_head|dense|fusion"):
        _evaluation_contract({"semantic_head": head}, {})
