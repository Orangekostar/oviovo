from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.evaluation.oviv2_semantic_replay import (
    SemanticReplayInput,
    replay_semantic_evidence,
    semantic_evidence_equal,
)
from src.oviv2.dense_projection import DenseSemanticConfig
from src.oviv2.evidence import EvidenceConfig, SparseEvidenceStore
from src.oviv2.observations import ObservationKind
from src.oviv2.runtime import Oviv2Runtime, Oviv2RuntimeConfig
from tests.oviv2.test_dense_projection import make_dense_frame
from tests.oviv2.test_runtime import dense_provenance, frame, observation


def _semantic_arrays(store) -> dict[tuple[int, int, int], tuple[np.ndarray, ...]]:
    return {
        key: (
            block.semantic_ids.copy(),
            block.semantic_support.copy(),
            block.semantic_revisions.copy(),
        )
        for key, block in sorted(store._blocks.items())
    }


def test_replay_matches_runtime_dense_then_structure_semantics() -> None:
    current_frame = frame(0)
    dense = make_dense_frame(
        image_shape=(32, 32),
        stride=16,
        source_frame_id=0,
        class_count=4,
        class_ids=np.full((2, 2, 1), 4, dtype=np.int64),
        probabilities=np.full((2, 2, 1), 0.8, dtype=np.float32),
    )
    structure = observation(
        0,
        ObservationKind.STRUCTURE,
        1,
        {(0, 0, 20)},
    )
    dense_config = DenseSemanticConfig(
        voxel_size_m=0.05,
        integration_radius_m=6.0,
        minimum_probability=0.01,
        minimum_quality=0.01,
        entropy_power=1.0,
        view_angle_power=0.0,
    )
    evidence_config = EvidenceConfig(block_resolution=8, semantic_top_k=4)
    runtime = Oviv2Runtime(
        "room0",
        Oviv2RuntimeConfig(
            evidence=evidence_config,
            dense_semantics=dense_config,
        ),
        dense_semantic_provenance=dense_provenance(),
    )
    runtime.process_frame(current_frame, (structure,), dense)

    replay = replay_semantic_evidence(
        (
            SemanticReplayInput(
                frame=current_frame,
                dense_semantics=dense,
                structure_observations=(structure,),
            ),
        ),
        evidence_config=evidence_config,
        dense_config=dense_config,
    )

    expected = _semantic_arrays(runtime.evidence)
    actual = _semantic_arrays(replay.evidence)
    assert actual.keys() == expected.keys()
    for key in actual:
        for actual_array, expected_array in zip(actual[key], expected[key], strict=True):
            np.testing.assert_array_equal(actual_array, expected_array)
    assert replay.structure_replayed is True
    assert replay.frame_results[0].structure_observation_count == 1
    assert replay.frame_results[0].dense_updated_voxel_count > 0


def test_replay_contracts_are_frozen_and_reject_object_observations() -> None:
    current_frame = frame(0)
    dense = make_dense_frame(
        image_shape=(32, 32),
        stride=16,
        source_frame_id=0,
    )
    item = SemanticReplayInput(current_frame, dense, ())
    with pytest.raises(FrozenInstanceError):
        item.structure_observations = ()

    object_value = observation(0, ObservationKind.OBJECT, 4, {(0, 0, 20)})
    with pytest.raises(ValueError, match="structure"):
        SemanticReplayInput(current_frame, dense, (object_value,))


def test_replay_rejects_non_increasing_frame_ids_without_partial_result() -> None:
    first = frame(1)
    second = frame(1)
    dense = make_dense_frame(
        image_shape=(32, 32),
        stride=16,
        source_frame_id=1,
    )
    item = SemanticReplayInput(first, dense, ())

    with pytest.raises(ValueError, match="increase"):
        replay_semantic_evidence(
            (item, SemanticReplayInput(second, dense, ())),
            evidence_config=EvidenceConfig(),
            dense_config=DenseSemanticConfig(voxel_size_m=0.05),
        )


def test_semantic_evidence_equal_compares_ids_support_and_revisions() -> None:
    left = SparseEvidenceStore(EvidenceConfig())
    right = SparseEvidenceStore(EvidenceConfig())
    for store in (left, right):
        store.update_semantic((1, 2, 3), 4, 0.5, 7)

    assert semantic_evidence_equal(left, right) is True

    right.update_semantic((1, 2, 3), 4, 0.25, 8)
    assert semantic_evidence_equal(left, right) is False
