from __future__ import annotations

import numpy as np
import pytest

from src.oviv2.query_instance_projection import (
    ProjectionConfig,
    project_queries_to_instances,
    project_queries_to_relation_support,
    project_query_evidence,
)
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    OviEntitySemanticEvidence,
    TemporalQueryEvidence,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _pair(
    token_entities: tuple[str, ...],
    visits: tuple[int, ...],
    x_positions: tuple[float, ...],
    contributor_counts: tuple[int, ...] | None = None,
    include_semantics: bool = False,
) -> NeuralSampleMap:
    counts = contributor_counts or (1,) * len(token_entities)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    contributor_entities = tuple(
        entity_id
        for entity_id, count in zip(token_entities, counts, strict=True)
        for _ in range(count)
    )
    contributor_visits = np.asarray(
        [
            visit
            for visit, count in zip(visits, counts, strict=True)
            for _ in range(count)
        ],
        dtype=np.int8,
    )
    coordinates = np.asarray(
        [[x, 0.0, 0.0, float(visit)] for x, visit in zip(x_positions, visits, strict=True)],
        dtype=np.float64,
    )
    semantic_keys = sorted(set(zip(visits, token_entities, strict=True)))
    semantics = (
        tuple(
            OviEntitySemanticEvidence(
                visit_id=visit,
                entity_id=entity_id,
                semantic_label="chair",
                semantic_score=0.9,
                semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            )
            for visit, entity_id in semantic_keys
        )
        if include_semantics
        else ()
    )
    return NeuralSampleMap(
        coordinates_xyzt=coordinates,
        features=np.ones((len(token_entities), 3), dtype=np.float32),
        visit_ids=np.asarray(visits, dtype=np.int8),
        source_visit_ids=contributor_visits,
        source_entity_ids=contributor_entities,
        source_point_indices=np.arange(sum(counts), dtype=np.int64),
        source_to_token_offsets=offsets,
        neural_voxel_size_m=0.02,
        feature_schema="rgb",
        coordinate_frame_id="world",
        source_manifest_sha256=SHA_A,
        source_visit_map_sha256=(SHA_B, SHA_C),
        entity_semantics=semantics,
    )


def _evidence(
    pair: NeuralSampleMap,
    masks: list[list[bool]],
    scores: list[list[float]] | None = None,
    *,
    backend_name: str = "geometric_semantic",
) -> TemporalQueryEvidence:
    mask_array = np.asarray(masks, dtype=bool)
    score_array = (
        mask_array.astype(np.float32)
        if scores is None
        else np.asarray(scores, dtype=np.float32)
    )
    return TemporalQueryEvidence(
        status="PASS",
        backend_name=backend_name,
        backend_config_sha256=SHA_A,
        pair_sha256=pair.content_sha256(),
        temporal_query_ids=tuple(f"q{index}" for index in range(len(masks))),
        query_masks=mask_array,
        token_scores=score_array,
        query_scores=np.full(len(masks), 0.8, dtype=np.float32),
        checkpoint_sha256=None,
        ranking_eligible=True,
        runtime_s=0.0,
        peak_memory_bytes=0,
    )


def test_projection_counts_all_source_contributors_not_only_token_representative() -> None:
    pair = _pair(
        ("ovi:chair:0", "ovi:chair:1"),
        (0, 1),
        (0.0, 0.1),
        contributor_counts=(3, 1),
    )
    matrix = project_query_evidence(pair, _evidence(pair, [[True, True]]))

    record = matrix["q0", "ovi:chair:0", 0]
    assert record.token_intersection_count == 1
    assert record.source_point_count == 3
    assert record.entity_source_point_count == 3
    assert record.source_point_coverage == 1.0


def test_projection_reports_literal_soft_mass_and_query_fraction() -> None:
    pair = _pair(
        ("t0:a", "t0:a", "t1:b"),
        (0, 0, 1),
        (0.0, 0.02, 0.0),
        contributor_counts=(2, 1, 1),
        include_semantics=True,
    )
    evidence = _evidence(pair, [[True, False, True]], [[0.75, 0.25, 1.0]])

    record = project_query_evidence(pair, evidence)["q0", "t0:a", 0]

    assert record.token_intersection_count == 1
    assert record.entity_token_coverage == 0.5
    assert record.query_mask_fraction == 0.5
    assert record.soft_mass == pytest.approx(1.0)
    assert record.source_point_count == 2
    assert record.entity_source_point_count == 3
    assert record.source_point_coverage == pytest.approx(2.0 / 3.0)
    assert np.allclose(record.centroid_xyz, [0.0, 0.0, 0.0])
    assert record.semantic_compatibility is None


@pytest.mark.parametrize(
    ("case", "entities", "visits", "positions", "mask", "expected"),
    [
        ("one_to_one", ("a", "b"), (0, 1), (0.0, 0.02), (True, True), "persistent_static"),
        ("moved", ("a", "b"), (0, 1), (0.0, 0.8), (True, True), "persistent_moved"),
        ("appeared", ("a", "b"), (0, 1), (0.0, 0.8), (False, True), "appeared"),
        ("removed", ("a", "b"), (0, 1), (0.0, 0.8), (True, False), "removed_candidate"),
        ("split", ("a", "b", "c"), (0, 1, 1), (0.0, 0.1, 0.2), (True, True, True), "split"),
        ("merge", ("a", "b", "c"), (0, 0, 1), (0.0, 0.1, 0.2), (True, True, True), "merge"),
        (
            "uncertain",
            ("a", "b", "c", "d"),
            (0, 0, 1, 1),
            (0.0, 0.1, 0.2, 0.3),
            (True, True, True, True),
            "uncertain",
        ),
    ],
)
def test_projection_preserves_relation_topology(
    case: str,
    entities: tuple[str, ...],
    visits: tuple[int, ...],
    positions: tuple[float, ...],
    mask: tuple[bool, ...],
    expected: str,
) -> None:
    pair = _pair(entities, visits, positions)
    result = project_queries_to_instances(
        pair,
        _evidence(pair, [list(mask)]),
        ProjectionConfig(static_centroid_tolerance_m=0.10),
    )

    assert len(result.relations) == 1, case
    assert result.relations[0].state == expected


def test_projection_keeps_repeated_entity_labels_distinct_by_id() -> None:
    pair = _pair(
        ("t0:chair:left", "t0:chair:right", "t1:chair:left", "t1:chair:right"),
        (0, 0, 1, 1),
        (0.0, 2.0, 0.1, 2.1),
    )
    evidence = _evidence(
        pair,
        [[True, False, True, False], [False, True, False, True]],
    )

    result = project_queries_to_instances(pair, evidence, ProjectionConfig())

    assert result.relations[0].t0_entity_ids == ("t0:chair:left",)
    assert result.relations[0].t1_entity_ids == ("t1:chair:left",)
    assert result.relations[1].t0_entity_ids == ("t0:chair:right",)
    assert result.relations[1].t1_entity_ids == ("t1:chair:right",)


def test_projection_rejects_blocked_evidence() -> None:
    pair = _pair(("a", "b"), (0, 1), (0.0, 0.0))
    evidence = TemporalQueryEvidence.blocked(
        status="BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
        backend_name="rescene:concerto",
        backend_config_sha256=SHA_A,
        pair_sha256=pair.content_sha256(),
    )

    with pytest.raises(ValueError, match="PASS"):
        project_queries_to_instances(pair, evidence, ProjectionConfig())


def test_strict_rescene_projection_records_fixed_ovi_support() -> None:
    pair = _pair(
        (
            "ovimap:1",
            "ovimap:1",
            "ovimap:2",
            "ovimap:10",
            "ovimap:10",
            "ovimap:11",
        ),
        (0, 0, 0, 1, 1, 1),
        (0.0, 0.02, 1.0, 0.1, 0.12, 1.1),
        contributor_counts=(2, 1, 1, 1, 2, 1),
    )
    evidence = _evidence(
        pair,
        [[True, True, False, True, True, False]],
        [[0.9, 0.8, 0.1, 0.85, 0.75, 0.1]],
        backend_name="rescene:concerto",
    )

    result = project_queries_to_relation_support(
        pair,
        evidence,
        minimum_source_coverage=0.25,
        minimum_competition_margin=0.08,
    )

    assert len(result) == 1
    relation = result[0]
    assert relation.relation_source == "frozen-rescene"
    assert relation.accepted
    assert not relation.assignment_is_null
    assert relation.t0_owner_entity_id == 1
    assert relation.t1_owner_entity_id == 10
    assert np.array_equal(relation.t0_source_vertex_indices, [0, 1, 2])
    assert np.array_equal(relation.t1_source_vertex_indices, [0, 1, 2])
    assert relation.t0_mask_coverage == 1.0
    assert relation.t1_mask_coverage == 1.0
    assert relation.t0_mask_purity == 0.5
    assert relation.t1_mask_purity == 0.5
    assert relation.t0_competing_score == 0.0
    assert relation.t1_competing_score == 0.0
    assert relation.competition_margin == 1.0
    assert relation.query_confidence == pytest.approx(0.8)


def test_strict_rescene_projection_requires_source_coverage_not_token_fallback() -> None:
    pair = _pair(
        ("ovimap:1", "ovimap:1", "ovimap:10", "ovimap:10"),
        (0, 0, 1, 1),
        (0.0, 0.02, 0.1, 0.12),
        contributor_counts=(1, 9, 1, 9),
    )
    evidence = _evidence(
        pair,
        [[True, False, True, False]],
        backend_name="rescene:concerto",
    )

    (relation,) = project_queries_to_relation_support(
        pair,
        evidence,
        minimum_source_coverage=0.25,
        minimum_competition_margin=0.08,
    )

    assert not relation.accepted
    assert relation.assignment_is_null
    assert relation.t0_mask_coverage == pytest.approx(0.1)
    assert relation.t1_mask_coverage == pytest.approx(0.1)
    assert relation.rejection_reasons == (
        "insufficient_t0_coverage",
        "insufficient_t1_coverage",
    )


def test_strict_rescene_projection_returns_null_for_competing_entities() -> None:
    pair = _pair(
        ("ovimap:1", "ovimap:2", "ovimap:10", "ovimap:11"),
        (0, 0, 1, 1),
        (0.0, 0.1, 0.0, 0.1),
    )
    evidence = _evidence(
        pair,
        [[True, True, True, True]],
        backend_name="rescene:concerto",
    )

    (relation,) = project_queries_to_relation_support(
        pair,
        evidence,
        minimum_source_coverage=0.25,
        minimum_competition_margin=0.08,
    )

    assert not relation.accepted
    assert relation.assignment_is_null
    assert relation.t0_competing_score == 1.0
    assert relation.t1_competing_score == 1.0
    assert relation.competition_margin == 0.0
    assert relation.rejection_reasons == ("ambiguous_query_competition",)


def test_strict_rescene_projection_rejects_non_rescene_evidence() -> None:
    pair = _pair(("ovimap:1", "ovimap:10"), (0, 1), (0.0, 0.1))

    with pytest.raises(ValueError, match="ReScene"):
        project_queries_to_relation_support(
            pair,
            _evidence(pair, [[True, True]]),
            minimum_source_coverage=0.25,
            minimum_competition_margin=0.08,
        )
