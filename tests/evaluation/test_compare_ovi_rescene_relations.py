from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluation.compare_ovi_rescene_relations import (
    RelationComparisonError,
    classify_b5_relations,
    compare_relation_topologies,
    select_registration_candidates,
    topology_key,
)
from src.oviv2.query_instance_projection import ProjectionConfig
from src.oviv2.rescene_input_bridge import NativeSamplingMap
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    OviEntitySemanticEvidence,
    PairRelation,
    TemporalQueryEvidence,
)


def _relation(
    query: str,
    state: str,
    t0: tuple[str, ...],
    t1: tuple[str, ...],
    score: float,
    *,
    source: str = "rescene",
) -> PairRelation:
    return PairRelation(
        temporal_query_id=query,
        t0_entity_ids=t0,
        t1_entity_ids=t1,
        state=state,
        query_confidence=score,
        evidence={"query_score": score},
        identity_source=source,
    )


def _pair_and_sampling() -> tuple[NeuralSampleMap, NativeSamplingMap]:
    entity_ids = ("a", "c", "c", "b", "b", "d", "d")
    visits = np.asarray([0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    coordinates = np.column_stack(
        (
            np.arange(7, dtype=np.float64) * 0.02,
            np.zeros((7, 2), dtype=np.float64),
            visits.astype(np.float64),
        )
    )
    pair = NeuralSampleMap(
        coordinates_xyzt=coordinates,
        features=np.ones((7, 6), dtype=np.float32),
        visit_ids=visits,
        source_visit_ids=visits,
        source_entity_ids=entity_ids,
        source_point_indices=np.arange(7, dtype=np.int64),
        source_to_token_offsets=np.arange(8, dtype=np.int64),
        neural_voxel_size_m=0.02,
        feature_schema="rgb_normals",
        coordinate_frame_id="world",
        source_manifest_sha256="a" * 64,
        source_visit_map_sha256=("b" * 64, "c" * 64),
        entity_semantics=tuple(
            OviEntitySemanticEvidence(
                visit,
                entity,
                entity,
                1.0,
                None,
            )
            for visit, entity in ((0, "a"), (0, "c"), (1, "b"), (1, "d"))
        ),
    )
    sampling = NativeSamplingMap(
        adapter_to_model=np.asarray([0, 0, 1, 2, 2, 3, 3]),
        selected_adapter_indices=np.asarray([0, 2, 3, 5]),
        model_grid_coordinates=np.asarray([[0, 0, 0], [2, 0, 0], [0, 0, 0], [2, 0, 0]]),
        model_visit_ids=np.asarray([0, 0, 1, 1]),
        visit_model_offsets=np.asarray([0, 2, 4]),
        visit_grid_origins=np.zeros((2, 3), dtype=np.int64),
        voxel_size_m=0.02,
        sampler_seed=45,
        sampler_mode="train",
        sampler_source_sha256="d" * 64,
    )
    return pair, sampling


def _evidence(pair: NeuralSampleMap) -> TemporalQueryEvidence:
    masks = np.asarray(
        [
            [True, True, False, True, True, False, False],
            [False, False, True, True, False, False, False],
        ],
        dtype=np.bool_,
    )
    return TemporalQueryEvidence(
        status="PASS",
        backend_name="rescene:concerto",
        backend_config_sha256="e" * 64,
        pair_sha256=pair.content_sha256(),
        temporal_query_ids=("q0", "q1"),
        query_masks=masks,
        token_scores=masks.astype(np.float32),
        query_scores=np.asarray([0.8, 0.9], dtype=np.float32),
        checkpoint_sha256="f" * 64,
        ranking_eligible=True,
        runtime_s=1.0,
        peak_memory_bytes=1024,
    )


def test_topology_comparison_deduplicates_queries_and_counts_states() -> None:
    shared = _relation("b4:0", "persistent_static", ("a",), ("b",), 0.7)
    b4 = (
        shared,
        _relation("b4:duplicate", "persistent_static", ("a",), ("b",), 0.6),
        _relation("b4:appeared", "appeared", (), ("e",), 1.0),
    )
    b5 = (
        _relation("q0", "persistent_static", ("a",), ("b",), 0.8),
        _relation("q1", "persistent_moved", ("c",), ("d",), 0.9),
    )

    delta = compare_relation_topologies(b4, b5)

    assert delta.b4_raw_count == 3
    assert delta.b4_unique_count == 2
    assert delta.b5_raw_count == 2
    assert delta.b5_unique_count == 2
    assert delta.intersection == (topology_key(shared),)
    assert delta.b4_only == (("appeared", (), ("e",)),)
    assert delta.b5_only == (("persistent_moved", ("c",), ("d",)),)
    assert delta.b4_state_counts == {"appeared": 1, "persistent_static": 1}
    assert delta.b5_state_counts == {
        "persistent_moved": 1,
        "persistent_static": 1,
    }
    assert delta.b4_unique_persistent_one_to_one_count == 1
    assert delta.b5_unique_persistent_one_to_one_count == 2


def test_b5_classification_records_threshold_fallback_collision_and_conflicts() -> None:
    pair, sampling = _pair_and_sampling()
    evidence = _evidence(pair)
    relations = (
        _relation("q0", "merge", ("a", "c"), ("b",), 0.8),
        _relation("q1", "persistent_moved", ("c",), ("b",), 0.9),
    )

    diagnostics = classify_b5_relations(
        pair=pair,
        evidence=evidence,
        relations=relations,
        sampling=sampling,
        projection_config=ProjectionConfig(
            minimum_entity_token_coverage=0.75,
            minimum_source_point_coverage=0.75,
        ),
    )

    assert diagnostics["q0"].threshold_qualified is True
    assert diagnostics["q0"].fallback_only is False
    assert diagnostics["q0"].collision_fraction == pytest.approx(0.5)
    assert diagnostics["q0"].collision_dominated is True
    assert diagnostics["q1"].threshold_qualified is False
    assert diagnostics["q1"].fallback_only is True
    assert diagnostics["q1"].contradictory is False

    contradictory = (
        _relation("q0", "persistent_static", ("a",), ("b",), 0.8),
        _relation("q1", "persistent_moved", ("c",), ("b",), 0.9),
    )
    conflict_diagnostics = classify_b5_relations(
        pair=pair,
        evidence=evidence,
        relations=contradictory,
        sampling=sampling,
        projection_config=ProjectionConfig(
            minimum_entity_token_coverage=0.75,
            minimum_source_point_coverage=0.75,
        ),
    )
    assert conflict_diagnostics["q0"].contradictory is True
    assert conflict_diagnostics["q1"].contradictory is True


def test_registration_selection_is_deterministic_and_excludes_unsafe_relations() -> (
    None
):
    pair, sampling = _pair_and_sampling()
    evidence = _evidence(pair)
    relations = (
        _relation("q0", "persistent_static", ("a",), ("b",), 0.8),
        _relation("q1", "persistent_moved", ("c",), ("b",), 0.9),
    )
    diagnostics = classify_b5_relations(
        pair=pair,
        evidence=evidence,
        relations=relations,
        sampling=sampling,
        projection_config=ProjectionConfig(),
    )

    assert (
        select_registration_candidates(
            relations,
            diagnostics,
            b5_only={topology_key(value) for value in relations},
            maximum_count=20,
        )
        == ()
    )

    independent = (
        _relation("q0", "persistent_static", ("a",), ("b",), 0.8),
        _relation("q1", "persistent_moved", ("c",), ("d",), 0.9),
    )
    independent_diagnostics = classify_b5_relations(
        pair=pair,
        evidence=evidence,
        relations=independent,
        sampling=sampling,
        projection_config=ProjectionConfig(),
    )
    assert tuple(
        relation.temporal_query_id
        for relation in select_registration_candidates(
            independent,
            independent_diagnostics,
            b5_only={topology_key(value) for value in independent},
            maximum_count=20,
        )
    ) == ("q1", "q0")


def test_classification_rejects_mismatched_pair_or_sampling() -> None:
    pair, sampling = _pair_and_sampling()
    evidence = _evidence(pair)
    bad_evidence = TemporalQueryEvidence(
        status="PASS",
        backend_name=evidence.backend_name,
        backend_config_sha256=evidence.backend_config_sha256,
        pair_sha256="0" * 64,
        temporal_query_ids=evidence.temporal_query_ids,
        query_masks=evidence.query_masks,
        token_scores=evidence.token_scores,
        query_scores=evidence.query_scores,
        checkpoint_sha256=evidence.checkpoint_sha256,
        ranking_eligible=True,
        runtime_s=1.0,
        peak_memory_bytes=1,
    )
    relation = _relation("q0", "persistent_static", ("a",), ("b",), 0.8)

    with pytest.raises((RelationComparisonError, ValueError), match="pair"):
        classify_b5_relations(
            pair=pair,
            evidence=bad_evidence,
            relations=(relation,),
            sampling=sampling,
            projection_config=ProjectionConfig(),
        )
