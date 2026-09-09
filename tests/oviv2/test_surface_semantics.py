from __future__ import annotations

import numpy as np

from src.oviv2.current_surface import SemanticSource
from src.oviv2.surface_semantics import (
    SurfaceSemanticConfig,
    SurfaceSemanticStrategy,
    read_surface_semantics,
    transfer_surface_semantics,
)


def _inputs() -> dict[str, np.ndarray]:
    return {
        "local_semantic_ids": np.asarray(
            [[18, 0, 0], [18, 18, 18], [18, 18, 18], [0, 0, 0]],
            dtype=np.int32,
        ),
        "local_supports": np.asarray(
            [[0.1, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "local_distances_m": np.asarray(
            [[0.01, np.inf, np.inf], [0.01, 0.012, 0.015], [0.2, 0.3, 0.4], [np.inf, np.inf, np.inf]],
            dtype=np.float32,
        ),
        "entity_semantic_ids": np.asarray([10, 10, 10, 0], dtype=np.int32),
        "entity_confidences": np.asarray([0.9, 0.9, 0.9, 0.0], dtype=np.float32),
    }


def _config(strategy: SurfaceSemanticStrategy) -> SurfaceSemanticConfig:
    return SurfaceSemanticConfig(
        strategy=strategy,
        maximum_correspondence_distance_m=0.05,
        distance_sigma_m=0.025,
        minimum_local_support=2.0,
        entity_weight_scale=0.75,
        minimum_entity_confidence=0.2,
    )


def test_s1_normalized_singleton_exposes_false_high_confidence_control() -> None:
    result = read_surface_semantics(**_inputs(), config=_config(SurfaceSemanticStrategy.S1))

    assert result.semantic_ids[0] == 18
    assert result.semantic_confidences[0] == 1.0
    assert result.support_reliabilities[0] < 0.1


def test_s2_keeps_reliability_separate_and_falls_back_to_owner() -> None:
    result = read_surface_semantics(**_inputs(), config=_config(SurfaceSemanticStrategy.S2))

    assert result.semantic_ids.tolist() == [10, 18, 10, 0]
    assert result.semantic_source_codes.tolist() == [
        SemanticSource.OWNER_ENTITY,
        SemanticSource.LOCAL_SURFACE,
        SemanticSource.OWNER_ENTITY,
        SemanticSource.UNKNOWN,
    ]
    assert result.support_reliabilities[0] < 0.1
    assert result.semantic_confidences[0] == np.float32(0.9)
    assert result.support_reliabilities[1] > 0.7
    assert result.semantic_confidences[3] == 0.0


def test_s0_is_bounded_nearest_correspondence() -> None:
    result = read_surface_semantics(**_inputs(), config=_config(SurfaceSemanticStrategy.S0))

    assert result.semantic_ids.tolist() == [18, 18, 0, 0]
    assert result.semantic_source_codes.tolist() == [
        SemanticSource.LOCAL_SURFACE,
        SemanticSource.LOCAL_SURFACE,
        SemanticSource.UNKNOWN,
        SemanticSource.UNKNOWN,
    ]


def test_reference_transfer_is_chunk_invariant_for_all_strategies() -> None:
    fine = np.asarray(
        [[0.00, 0.0, 0.0], [0.04, 0.0, 0.0], [1.00, 0.0, 0.0]],
        dtype=np.float32,
    )
    reference = np.asarray(
        [[0.01, 0.0, 0.0], [0.03, 0.0, 0.0], [0.05, 0.0, 0.0]],
        dtype=np.float32,
    )
    configs = tuple(
        SurfaceSemanticConfig(
            strategy=strategy,
            maximum_correspondence_distance_m=0.06,
            minimum_local_support=1.0,
        )
        for strategy in SurfaceSemanticStrategy
    )
    kwargs = {
        "fine_vertices_xyz": fine,
        "reference_vertices_xyz": reference,
        "reference_semantic_ids": np.asarray([2, 2, 3], dtype=np.int32),
        "reference_supports": np.asarray([0.6, 0.6, 0.9], dtype=np.float32),
        "entity_semantic_ids": np.asarray([4, 4, 4], dtype=np.int32),
        "entity_confidences": np.asarray([0.8, 0.8, 0.8], dtype=np.float32),
        "configs": configs,
        "neighbor_count": 3,
    }

    scalar = transfer_surface_semantics(**kwargs, point_batch_size=1)
    vector = transfer_surface_semantics(**kwargs, point_batch_size=99)

    assert set(scalar) == set(SurfaceSemanticStrategy)
    for strategy in SurfaceSemanticStrategy:
        for name in (
            "semantic_ids",
            "semantic_confidences",
            "support_reliabilities",
            "semantic_source_codes",
        ):
            np.testing.assert_array_equal(
                getattr(scalar[strategy], name), getattr(vector[strategy], name)
            )
    assert scalar[SurfaceSemanticStrategy.S0].semantic_ids.tolist() == [2, 2, 0]
    assert scalar[SurfaceSemanticStrategy.S2].semantic_ids.tolist() == [2, 4, 4]
