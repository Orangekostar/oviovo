from __future__ import annotations

from dataclasses import fields

import numpy as np

from src.oviv2.geometric_pair_reasoner import (
    GeometricPairReasoner,
    GeometricReasonerConfig,
)
from src.oviv2.temporal_pair_reasoner import TemporalPairReasoner, validate_query_evidence
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    OviEntitySemanticEvidence,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _pair() -> NeuralSampleMap:
    entity_ids = (
        "t0:chair:left",
        "t0:chair:right",
        "t0:table",
        "t1:chair:left",
        "t1:chair:right",
        "t1:lamp",
    )
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 1.0],
            [2.1, 0.0, 0.0, 1.0],
            [7.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = ("chair", "chair", "table", "chair", "chair", "lamp")
    embeddings = {
        "chair": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "table": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "lamp": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    semantics = tuple(
        OviEntitySemanticEvidence(
            visit_id=int(coordinates[index, 3]),
            entity_id=entity_id,
            semantic_label=labels[index],
            semantic_score=0.9,
            semantic_embedding=embeddings[labels[index]],
        )
        for index, entity_id in enumerate(entity_ids)
    )
    return NeuralSampleMap(
        coordinates_xyzt=coordinates,
        features=np.ones((6, 3), dtype=np.float32),
        visit_ids=coordinates[:, 3].astype(np.int8),
        source_visit_ids=coordinates[:, 3].astype(np.int8),
        source_entity_ids=entity_ids,
        source_point_indices=np.arange(6, dtype=np.int64),
        source_to_token_offsets=np.arange(7, dtype=np.int64),
        neural_voxel_size_m=0.02,
        feature_schema="rgb",
        coordinate_frame_id="world",
        source_manifest_sha256=SHA_A,
        source_visit_map_sha256=(SHA_B, SHA_C),
        entity_semantics=semantics,
    )


def _query_entity_keys(pair: NeuralSampleMap, masks: np.ndarray) -> tuple[tuple[str, ...], ...]:
    token_keys = tuple(
        f"{int(pair.visit_ids[index])}:{pair.token_entity_ids[index]}"
        for index in range(len(pair.visit_ids))
    )
    return tuple(
        tuple(token_keys[index] for index in np.flatnonzero(mask)) for mask in masks
    )


def test_geometric_reasoner_is_protocol_compatible_repeatable_and_pair_bound() -> None:
    pair = _pair()
    reasoner: TemporalPairReasoner = GeometricPairReasoner(
        GeometricReasonerConfig(maximum_centroid_distance_m=0.5)
    )

    first = reasoner.infer(pair)
    second = reasoner.infer(pair)
    validate_query_evidence(pair, first)

    assert first == second
    assert _query_entity_keys(pair, first.query_masks) == (
        ("0:t0:chair:left", "1:t1:chair:left"),
        ("0:t0:chair:right", "1:t1:chair:right"),
        ("0:t0:table",),
        ("1:t1:lamp",),
    )


def test_repeated_same_category_entities_are_not_cross_matched() -> None:
    pair = _pair()
    evidence = GeometricPairReasoner(
        GeometricReasonerConfig(maximum_centroid_distance_m=0.5)
    ).infer(pair)

    groups = _query_entity_keys(pair, evidence.query_masks)
    assert ("0:t0:chair:left", "1:t1:chair:right") not in groups
    assert ("0:t0:chair:right", "1:t1:chair:left") not in groups


def test_geometric_reasoner_contract_has_no_ground_truth_inputs() -> None:
    names = {item.name.lower() for item in fields(GeometricReasonerConfig)}
    forbidden = ("ground_truth", "gt_", "change_label", "instance_target")
    assert not any(marker in name for name in names for marker in forbidden)
