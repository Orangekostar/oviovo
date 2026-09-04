from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.two_visit_contracts import (
    CurrentCompositionDecision,
    NeuralSampleMap,
    OviEntitySemanticEvidence,
    PairRelation,
    TemporalQueryEvidence,
    VisitMap,
    validate_visit_pair,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _snapshot(*, timestamp: float, suffix: str) -> MapSnapshot:
    return MapSnapshot(
        method="OVI-MAP",
        scene_id="apartment",
        timestamp=timestamp,
        entities=[
            EntityPrediction(
                entity_id=f"ovi:{suffix}:chair",
                points_xyz=np.array(
                    [[0.001, 0.002, 0.003], [0.011, 0.002, 0.003]],
                    dtype=np.float32,
                ),
                semantic_embedding=np.array([0.2, 0.8], dtype=np.float32),
                semantic_label="chair",
                semantic_score=0.9,
                lifecycle_state="observed",
                first_seen=timestamp,
                last_seen=timestamp,
            )
        ],
        background_xyz=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        scope="current",
    )


def _visit(visit_id: int) -> VisitMap:
    start = 0 if visit_id == 0 else 10
    return VisitMap(
        visit_id=visit_id,
        snapshot=_snapshot(timestamp=float(start), suffix=f"t{visit_id}"),
        coordinate_frame_id="tesse:apartment:world",
        source_manifest_sha256=SHA_A,
        map_voxel_size_m=0.01,
        observed_frame_start=start,
        observed_frame_end=start + 4,
    )


def _pair() -> NeuralSampleMap:
    return NeuralSampleMap(
        coordinates_xyzt=np.array(
            [[0.005, 0.0, 0.0, 0.0], [0.105, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        features=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        visit_ids=np.array([0, 1], dtype=np.int8),
        source_visit_ids=np.array([0, 0, 1], dtype=np.int8),
        source_entity_ids=("ovi:t0:chair", "ovi:t0:chair", "ovi:t1:chair"),
        source_point_indices=np.array([0, 1, 2], dtype=np.int64),
        source_to_token_offsets=np.array([0, 2, 3], dtype=np.int64),
        neural_voxel_size_m=0.02,
        feature_schema="rgb",
        coordinate_frame_id="tesse:apartment:world",
        source_manifest_sha256=SHA_A,
        source_visit_map_sha256=(SHA_B, SHA_C),
    )


def test_visit_pair_requires_independent_non_overlapping_maps_in_one_frame() -> None:
    t0 = _visit(0)
    t1 = _visit(1)

    validate_visit_pair(t0, t1)

    assert t0.snapshot_sha256 != t1.snapshot_sha256
    assert t0.map_voxel_size_m == t1.map_voxel_size_m == 0.01


def test_visit_pair_rejects_mixed_coordinate_frame() -> None:
    t0 = _visit(0)
    source = _visit(1)
    t1 = VisitMap(
        visit_id=1,
        snapshot=source.snapshot,
        coordinate_frame_id="other-frame",
        source_manifest_sha256=SHA_A,
        map_voxel_size_m=0.01,
        observed_frame_start=10,
        observed_frame_end=14,
    )

    with pytest.raises(ValueError, match="coordinate frame"):
        validate_visit_pair(t0, t1)


def test_visit_pair_rejects_overlapping_windows_and_shared_snapshot() -> None:
    t0 = _visit(0)
    with pytest.raises(ValueError, match="independent"):
        validate_visit_pair(
            t0,
            VisitMap(
                visit_id=1,
                snapshot=t0.snapshot,
                coordinate_frame_id=t0.coordinate_frame_id,
                source_manifest_sha256=SHA_A,
                map_voxel_size_m=0.01,
                observed_frame_start=10,
                observed_frame_end=14,
            ),
        )

    t1 = _visit(1)
    overlapping = VisitMap(
        visit_id=1,
        snapshot=t1.snapshot,
        coordinate_frame_id=t1.coordinate_frame_id,
        source_manifest_sha256=SHA_A,
        map_voxel_size_m=0.01,
        observed_frame_start=4,
        observed_frame_end=8,
    )
    with pytest.raises(ValueError, match="non-overlapping"):
        validate_visit_pair(t0, overlapping)


def test_neural_sample_map_requires_exact_time_and_conserving_csr() -> None:
    pair = _pair()

    assert np.array_equal(pair.coordinates_xyzt[:, 3], pair.visit_ids)
    assert pair.source_to_token_offsets.tolist() == [0, 2, 3]
    assert pair.source_point_count == 3
    assert pair.token_entity_ids == ("ovi:t0:chair", "ovi:t1:chair")
    assert all(not array.flags.writeable for array in pair.arrays())


def test_neural_sample_map_rejects_scaled_time_coordinate() -> None:
    values = _pair_kwargs()
    values["coordinates_xyzt"] = np.array(
        [[0.005, 0.0, 0.0, 0.0], [0.105, 0.0, 0.0, 0.02]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="exact visit"):
        NeuralSampleMap(**values)


def test_neural_sample_map_rejects_empty_or_cross_visit_csr_spans() -> None:
    values = _pair_kwargs()
    values["source_to_token_offsets"] = np.array([0, 3, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="non-empty"):
        NeuralSampleMap(**values)

    values = _pair_kwargs()
    values["source_visit_ids"] = np.array([0, 1, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="mixes visits"):
        NeuralSampleMap(**values)


def test_neural_sample_map_rejects_dropped_or_duplicate_source_points() -> None:
    values = _pair_kwargs()
    values["source_point_indices"] = np.array([0, 0, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="permutation"):
        NeuralSampleMap(**values)


def test_ovi_semantic_evidence_is_read_only_and_separate_from_token_features() -> None:
    embedding = np.array([0.25, 0.75], dtype=np.float32)
    semantic = OviEntitySemanticEvidence(
        visit_id=0,
        entity_id="ovi:t0:chair",
        semantic_label="chair",
        semantic_score=0.9,
        semantic_embedding=embedding,
    )
    embedding[:] = 0.0

    assert semantic.semantic_embedding is not None
    assert np.allclose(semantic.semantic_embedding, [0.25, 0.75])
    assert not semantic.semantic_embedding.flags.writeable


def test_temporal_evidence_copies_arrays_and_has_value_equality() -> None:
    masks = np.array([[True, False], [False, True]])
    scores = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    first = TemporalQueryEvidence(
        status="PASS",
        backend_name="geometric_semantic",
        backend_config_sha256=SHA_A,
        pair_sha256=SHA_B,
        temporal_query_ids=("q0", "q1"),
        query_masks=masks,
        token_scores=scores,
        query_scores=np.array([0.9, 0.8], dtype=np.float32),
        checkpoint_sha256=None,
        ranking_eligible=True,
        runtime_s=0.01,
        peak_memory_bytes=1024,
    )
    second = TemporalQueryEvidence(
        status="PASS",
        backend_name="geometric_semantic",
        backend_config_sha256=SHA_A,
        pair_sha256=SHA_B,
        temporal_query_ids=("q0", "q1"),
        query_masks=masks,
        token_scores=scores,
        query_scores=np.array([0.9, 0.8], dtype=np.float32),
        checkpoint_sha256=None,
        ranking_eligible=True,
        runtime_s=0.01,
        peak_memory_bytes=1024,
    )
    masks[:] = False

    assert first == second
    assert first.query_masks is not None and first.query_masks[0, 0]
    assert not first.query_masks.flags.writeable


def test_blocked_temporal_evidence_cannot_carry_predictions() -> None:
    blocked = TemporalQueryEvidence.blocked(
        status="BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
        backend_name="rescene:concerto",
        backend_config_sha256=SHA_A,
        pair_sha256=SHA_B,
    )
    assert blocked.query_masks is None
    assert blocked.ranking_eligible is False

    with pytest.raises(ValueError, match="blocked evidence cannot carry"):
        TemporalQueryEvidence(
            status="BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
            backend_name="rescene:concerto",
            backend_config_sha256=SHA_A,
            pair_sha256=SHA_B,
            temporal_query_ids=("q0",),
            query_masks=np.ones((1, 2), dtype=bool),
            token_scores=np.ones((1, 2), dtype=np.float32),
            query_scores=np.ones(1, dtype=np.float32),
            checkpoint_sha256=None,
            ranking_eligible=False,
            runtime_s=0.0,
            peak_memory_bytes=0,
        )


@pytest.mark.parametrize(
    ("state", "t0_ids", "t1_ids"),
    [
        ("persistent_static", ("a",), ("b",)),
        ("persistent_moved", ("a",), ("b",)),
        ("appeared", (), ("b",)),
        ("removed_candidate", ("a",), ()),
        ("split", ("a",), ("b", "c")),
        ("merge", ("a", "b"), ("c",)),
        ("uncertain", ("a",), ("b", "c")),
    ],
)
def test_pair_relation_accepts_explicit_topologies(
    state: str,
    t0_ids: tuple[str, ...],
    t1_ids: tuple[str, ...],
) -> None:
    relation = PairRelation(
        temporal_query_id="q0",
        t0_entity_ids=t0_ids,
        t1_entity_ids=t1_ids,
        state=state,
        query_confidence=0.8,
        evidence={"overlap": 0.5},
        identity_source="geometric_baseline",
    )
    assert relation.state == state


def test_composition_decision_enforces_visibility_and_ovi_authority() -> None:
    suppressed = CurrentCompositionDecision(
        source_entity_id="ovi:t0:chair",
        source_visit=0,
        decision="suppress_t0_visible_free",
        visibility_status="visible_free",
        visibility_score=0.95,
        relation_id="q0",
        geometry_source=None,
        identity_source="geometric_baseline",
        state_source="t1_visibility",
        semantic_source=None,
    )
    assert suppressed.geometry_source is None

    with pytest.raises(ValueError, match="visible-free"):
        CurrentCompositionDecision(
            source_entity_id="ovi:t0:chair",
            source_visit=0,
            decision="suppress_t0_visible_free",
            visibility_status="unobserved",
            visibility_score=0.0,
            relation_id="q0",
            geometry_source=None,
            identity_source="geometric_baseline",
            state_source="pair_reasoner",
            semantic_source=None,
        )


def test_t0_geometry_replaced_by_t1_occupied_has_explicit_provenance() -> None:
    decision = CurrentCompositionDecision(
        source_entity_id="ovi:t0:chair",
        source_visit=0,
        decision="suppress_t0_occupied_by_t1",
        visibility_status="occupied",
        visibility_score=1.0,
        relation_id="q0",
        geometry_source=None,
        identity_source="geometric_baseline",
        state_source="t1_visibility",
        semantic_source=None,
    )

    assert decision.decision == "suppress_t0_occupied_by_t1"


def _pair_kwargs() -> dict[str, object]:
    pair = _pair()
    return {
        "coordinates_xyzt": pair.coordinates_xyzt,
        "features": pair.features,
        "visit_ids": pair.visit_ids,
        "source_visit_ids": pair.source_visit_ids,
        "source_entity_ids": pair.source_entity_ids,
        "source_point_indices": pair.source_point_indices,
        "source_to_token_offsets": pair.source_to_token_offsets,
        "neural_voxel_size_m": pair.neural_voxel_size_m,
        "feature_schema": pair.feature_schema,
        "coordinate_frame_id": pair.coordinate_frame_id,
        "source_manifest_sha256": pair.source_manifest_sha256,
        "source_visit_map_sha256": pair.source_visit_map_sha256,
    }
