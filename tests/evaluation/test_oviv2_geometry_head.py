from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.evaluation.oviv2_geometry_head import (
    GeometrySemanticStabilizationConfig,
    stabilize_low_support_semantics,
)
from src.oviv2.meshing import LabeledMesh


def _mesh(vertices: np.ndarray, semantic_ids: np.ndarray) -> LabeledMesh:
    count = len(vertices)
    return LabeledMesh(
        vertices_xyz=vertices,
        triangles=np.empty((0, 3), dtype=np.int64),
        colors_rgb=np.zeros((count, 3), dtype=np.float32),
        semantic_ids=semantic_ids,
        entity_ids=np.arange(1, count + 1, dtype=np.int64),
        semantic_confidence=np.full(count, 0.5, dtype=np.float32),
        ownership_confidence=np.full(count, 0.75, dtype=np.float32),
    )


def test_stabilizer_transfers_only_novel_nearby_vertex_semantics() -> None:
    low_support = _mesh(
        np.asarray(((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (1.0, 0.0, 0.0))),
        np.asarray((1, 1, 1)),
    )
    reference = _mesh(
        np.asarray(((0.0, 0.0, 0.0), (0.06, 0.0, 0.0))),
        np.asarray((2, 3)),
    )

    result = stabilize_low_support_semantics(
        low_support,
        reference,
        GeometrySemanticStabilizationConfig(maximum_transfer_distance_m=0.075),
    )

    np.testing.assert_array_equal(result.mesh.semantic_ids, (1, 3, 1))
    np.testing.assert_array_equal(result.mesh.entity_ids, low_support.entity_ids)
    np.testing.assert_array_equal(result.mesh.vertices_xyz, low_support.vertices_xyz)
    assert result.mesh.semantic_confidence[1] == reference.semantic_confidence[1]
    assert result.transferred_vertex_count == 1


def test_stabilization_contracts_are_frozen_and_validate_distances() -> None:
    config = GeometrySemanticStabilizationConfig(maximum_transfer_distance_m=0.075)
    with pytest.raises(FrozenInstanceError):
        config.maximum_transfer_distance_m = 0.1
    with pytest.raises((TypeError, ValueError), match="distance|tolerance"):
        GeometrySemanticStabilizationConfig(maximum_transfer_distance_m=-1.0)
