from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect

import numpy as np
import pytest


def _api():
    from src.evaluation.oviv2_instance_ensemble import (
        AuxiliaryInstanceConfig,
        build_auxiliary_entity_hypotheses,
    )

    return AuxiliaryInstanceConfig, build_auxiliary_entity_hypotheses


def _mesh():
    from src.oviv2.meshing import LabeledMesh

    return LabeledMesh(
        vertices_xyz=np.asarray(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            dtype=np.float32,
        ),
        triangles=np.empty((0, 3), dtype=np.int64),
        colors_rgb=np.zeros((5, 3), dtype=np.float32),
        semantic_ids=np.ones(5, dtype=np.int64),
        entity_ids=np.asarray([0, 9, 9, 9, 7], dtype=np.int64),
        semantic_confidence=np.ones(5, dtype=np.float32),
        ownership_confidence=np.ones(5, dtype=np.float32),
    )


def test_auxiliary_config_is_frozen_and_rejects_invalid_scores() -> None:
    config_type, _ = _api()
    config = config_type(score_weight=6.0, support_exponent=1.5, score_bias=0.05)
    assert config.score_weight == 6.0
    assert config.support_exponent == 1.5
    assert config.score_bias == 0.05
    with pytest.raises(FrozenInstanceError):
        config.score_weight = 1.0

    for values in (
        {"score_weight": 0.0},
        {"score_weight": np.inf},
        {"support_exponent": -0.1},
        {"support_exponent": np.nan},
        {"score_bias": -0.1},
        {"score_bias": 1.1},
    ):
        with pytest.raises((TypeError, ValueError)):
            config_type(**values)


def test_auxiliary_hypotheses_use_only_normalized_map_support() -> None:
    config_type, build = _api()
    config = config_type(score_weight=2.0, support_exponent=1.0, score_bias=0.1)

    hypotheses = build(_mesh(), {7: 4, 9: 5, 11: 6}, config)

    assert [item.hypothesis_id for item in hypotheses] == [
        "aux:e7:parent",
        "aux:e9:parent",
    ]
    assert [item.entity_id for item in hypotheses] == [7, 9]
    assert [item.semantic_id for item in hypotheses] == [4, 5]
    np.testing.assert_array_equal(hypotheses[0].vertex_indices, [4])
    np.testing.assert_array_equal(hypotheses[1].vertex_indices, [1, 2, 3])
    assert hypotheses[0].score == pytest.approx(2.0 / 3.0 + 0.1)
    assert hypotheses[1].score == pytest.approx(1.0)
    assert all(not item.vertex_indices.flags.writeable for item in hypotheses)


def test_auxiliary_hypotheses_require_semantics_for_every_mesh_entity() -> None:
    config_type, build = _api()
    with pytest.raises(ValueError, match="missing"):
        build(_mesh(), {7: 4}, config_type())
    with pytest.raises(TypeError, match="integer"):
        build(_mesh(), {7: 4, 9: 5.5}, config_type())


def test_auxiliary_generation_signature_has_no_evaluation_inputs() -> None:
    _, build = _api()
    parameters = set(inspect.signature(build).parameters)
    assert parameters == {"mesh", "entity_semantic_ids", "config"}
    assert not parameters & {"ground_truth", "target_vertices", "projected_masks"}
