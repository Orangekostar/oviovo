from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.two_visit_contracts import PairRelation, VisitMap
from src.oviv2.two_visit_current_map import (
    CompositionConfig,
    SignedVisibilityGrid,
    compose_current_map,
    write_two_visit_current_map,
)

SHA_A = "a" * 64
SHA_V = "f" * 64


def _entity(
    entity_id: str, points: list[list[float]], label: str = "chair"
) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(points, dtype=np.float32),
        semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        semantic_label=label,
        semantic_score=0.9,
        lifecycle_state="observed",
        first_seen=0.0,
        last_seen=1.0,
        metadata={"semantic_authority": "ovi"},
    )


def _visit(
    visit_id: int,
    entities: list[EntityPrediction],
    *,
    background: list[list[float]] | None = None,
) -> VisitMap:
    start = 0 if visit_id == 0 else 10
    return VisitMap(
        visit_id=visit_id,
        snapshot=MapSnapshot(
            method="OVI-MAP",
            scene_id="apartment",
            timestamp=float(start),
            entities=entities,
            background_xyz=(
                None if background is None else np.asarray(background, dtype=np.float32)
            ),
            scope="current",
        ),
        coordinate_frame_id="world",
        source_manifest_sha256=SHA_A,
        map_voxel_size_m=0.01,
        observed_frame_start=start,
        observed_frame_end=start + 4,
    )


def _visibility(
    status: str, point: tuple[float, float, float] = (0.01, 0.0, 0.0)
) -> SignedVisibilityGrid:
    key = np.floor(np.asarray(point) / 0.05).astype(np.int64)
    return SignedVisibilityGrid(
        voxel_size_m=0.05,
        voxel_keys=key[None, :],
        statuses=(status,),
        source_sha256=SHA_V,
    )


def _visibility_many(
    points: list[list[float]], statuses: tuple[str, ...]
) -> SignedVisibilityGrid:
    keys = np.floor(np.asarray(points) / 0.05).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    return SignedVisibilityGrid(
        voxel_size_m=0.05,
        voxel_keys=keys[order],
        statuses=tuple(statuses[index] for index in order),
        source_sha256=SHA_V,
    )


@pytest.mark.parametrize(
    ("visibility", "expected"),
    [
        ("occupied", "suppress_t0_occupied_by_t1"),
        ("visible_free", "suppress_t0_visible_free"),
        ("occluded", "retain_t0_occluded"),
        ("unobserved", "retain_t0_unobserved"),
    ],
)
def test_t1_first_composition_has_explicit_four_branch_decision(
    visibility: str, expected: str
) -> None:
    t0 = _visit(0, [_entity("t0:chair", [[0.01, 0.0, 0.0]])])
    t1 = _visit(1, [_entity("t1:lamp", [[2.0, 0.0, 0.0]], "lamp")])

    result = compose_current_map(
        t0,
        t1,
        (),
        _visibility(visibility),
        CompositionConfig(),
    )

    decisions = [
        item.decision
        for item in result.provenance
        if item.decision.source_visit == 0
        and item.decision.source_entity_id == "t0:chair"
    ]
    assert [item.decision for item in decisions] == [expected]


def test_missing_t1_query_without_visible_free_never_deletes_t0() -> None:
    t0 = _visit(0, [_entity("t0:chair", [[0.01, 0.0, 0.0]])])
    t1 = _visit(1, [_entity("t1:lamp", [[2.0, 0.0, 0.0]], "lamp")])
    relation = PairRelation(
        temporal_query_id="q0",
        t0_entity_ids=("t0:chair",),
        t1_entity_ids=(),
        state="removed_candidate",
        query_confidence=0.95,
        evidence={"query_score": 0.95},
        identity_source="geometric_baseline",
    )

    result = compose_current_map(
        t0,
        t1,
        (relation,),
        SignedVisibilityGrid.empty(0.05, SHA_V),
        CompositionConfig(),
    )

    fallback = next(
        entity
        for entity in result.snapshot.entities
        if entity.entity_id.startswith("t0-fallback:")
    )
    assert np.array_equal(fallback.points_xyz, t0.snapshot.entities[0].points_xyz)
    assert any(
        item.decision.decision == "retain_t0_unobserved" for item in result.provenance
    )


def test_object_visible_free_majority_suppresses_occluded_entity_residue() -> None:
    points = [[0.01 + 0.10 * index, 0.0, 0.0] for index in range(5)]
    t0 = _visit(0, [_entity("t0:couch", points, "Couch")])
    t1 = _visit(1, [_entity("t1:lamp", [[2.0, 0.0, 0.0]], "Lamp")])
    visibility = _visibility_many(
        points,
        ("visible_free", "visible_free", "visible_free", "visible_free", "occluded"),
    )

    result = compose_current_map(
        t0,
        t1,
        (),
        visibility,
        CompositionConfig(
            object_semantic_labels=frozenset({"Couch"}),
            minimum_entity_visible_free_fraction=0.8,
            minimum_entity_visible_free_voxels=3,
        ),
    )

    assert [entity.entity_id for entity in result.snapshot.entities] == ["t1:lamp"]
    residue = next(
        item
        for item in result.provenance
        if item.decision.source_entity_id == "t0:couch"
        and item.decision.visibility_status == "occluded"
    )
    assert residue.decision.decision == "suppress_t0_entity_visible_free"
    assert residue.decision.visibility_score == pytest.approx(1.0)
    assert residue.output_point_count == 0


@pytest.mark.parametrize(
    ("label", "statuses"),
    [
        ("Floor", ("visible_free",) * 4 + ("occluded",)),
        ("Couch", ("visible_free",) * 2 + ("occluded",) * 3),
        (
            "Couch",
            ("visible_free",) * 4 + ("occupied",) * 2 + ("occluded",),
        ),
    ],
)
def test_entity_lift_preserves_nonobjects_and_insufficient_evidence(
    label: str,
    statuses: tuple[str, ...],
) -> None:
    points = [[0.01 + 0.10 * index, 0.0, 0.0] for index in range(len(statuses))]
    t0 = _visit(0, [_entity("t0:entity", points, label)])
    t1 = _visit(1, [_entity("t1:lamp", [[2.0, 0.0, 0.0]], "Lamp")])

    result = compose_current_map(
        t0,
        t1,
        (),
        _visibility_many(points, statuses),
        CompositionConfig(
            object_semantic_labels=frozenset({"Couch"}),
            minimum_entity_visible_free_fraction=0.8,
            minimum_entity_visible_free_voxels=3,
        ),
    )

    fallback = next(
        entity
        for entity in result.snapshot.entities
        if entity.entity_id.startswith("t0-fallback:")
    )
    assert len(fallback.points_xyz) >= 1
    assert all(
        item.decision.decision != "suppress_t0_entity_visible_free"
        for item in result.provenance
    )


def test_moved_entity_uses_dense_ovi_t1_points_and_ovi_semantics() -> None:
    t0 = _visit(0, [_entity("t0:chair", [[0.001, 0.0, 0.0], [0.011, 0.0, 0.0]])])
    t1_points = np.asarray(
        [[1.001, 0.0, 0.0], [1.011, 0.0, 0.0], [1.019, 0.0, 0.0]],
        dtype=np.float32,
    )
    t1 = _visit(1, [_entity("t1:chair", t1_points.tolist())])
    relation = PairRelation(
        temporal_query_id="q0",
        t0_entity_ids=("t0:chair",),
        t1_entity_ids=("t1:chair",),
        state="persistent_moved",
        query_confidence=0.9,
        evidence={"centroid_distance_m": 1.0},
        identity_source="geometric_baseline",
    )

    result = compose_current_map(
        t0,
        t1,
        (relation,),
        _visibility("visible_free", point=(0.001, 0.0, 0.0)),
        CompositionConfig(),
    )

    assert len(result.snapshot.entities) == 1
    assert np.array_equal(result.snapshot.entities[0].points_xyz, t1_points)
    assert result.snapshot.entities[0].semantic_label == "chair"
    assert result.snapshot.entities[0].metadata["geometry_authority"] == "ovi_t1"
    emitted = next(
        item for item in result.provenance if item.decision.decision == "emit_t1"
    )
    assert emitted.decision.geometry_source == "ovi_t1"
    assert emitted.decision.semantic_source == "ovi_t1"


def test_t1_occupied_cells_replace_t0_but_unobserved_background_is_retained() -> None:
    t0 = _visit(
        0,
        [],
        background=[[0.001, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    t1 = _visit(1, [], background=[[0.011, 0.0, 0.0]])

    result = compose_current_map(
        t0,
        t1,
        (),
        SignedVisibilityGrid.empty(0.05, SHA_V),
        CompositionConfig(),
    )

    assert np.array_equal(
        result.snapshot.background_xyz,
        np.asarray([[0.011, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    )
    background_actions = [
        item.decision.decision
        for item in result.provenance
        if item.decision.source_entity_id == "__background__"
    ]
    assert background_actions == [
        "emit_t1",
        "suppress_t0_occupied_by_t1",
        "retain_t0_unobserved",
    ]


def test_multiple_retained_status_groups_reuse_one_fallback_entity() -> None:
    t0 = _visit(
        0,
        [_entity("t0:chair", [[0.001, 0.0, 0.0], [0.101, 0.0, 0.0]])],
    )
    t1 = _visit(1, [_entity("t1:lamp", [[2.0, 0.0, 0.0]], "lamp")])

    result = compose_current_map(
        t0,
        t1,
        (),
        _visibility("occluded", point=(0.001, 0.0, 0.0)),
        CompositionConfig(),
    )

    fallbacks = [
        entity
        for entity in result.snapshot.entities
        if entity.entity_id.startswith("t0-fallback:")
    ]
    assert len(fallbacks) == 1
    assert np.allclose(
        np.sort(fallbacks[0].points_xyz[:, 0]),
        [0.001, 0.101],
    )


def test_every_output_group_traces_only_to_ovi_source_points() -> None:
    t0 = _visit(0, [_entity("t0:chair", [[0.0, 0.0, 0.0]])])
    t1 = _visit(1, [_entity("t1:chair", [[1.0, 0.0, 0.0]])])

    result = compose_current_map(
        t0,
        t1,
        (),
        SignedVisibilityGrid.empty(0.05, SHA_V),
        CompositionConfig(),
    )

    emitted = [item for item in result.provenance if item.output_point_count > 0]
    assert {item.decision.geometry_source for item in emitted} == {"ovi_t0", "ovi_t1"}
    assert all(
        item.decision.semantic_source in {"ovi_t0", "ovi_t1", "fused_ovi"}
        for item in emitted
    )


def test_current_map_export_is_atomic_and_source_bound(tmp_path: Path) -> None:
    t0 = _visit(0, [_entity("t0:chair", [[0.0, 0.0, 0.0]])])
    t1 = _visit(1, [_entity("t1:chair", [[1.0, 0.0, 0.0]])])
    result = compose_current_map(
        t0,
        t1,
        (),
        SignedVisibilityGrid.empty(0.05, SHA_V),
        CompositionConfig(),
    )
    output = tmp_path / "current-map"

    manifest_path = write_two_visit_current_map(result, output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "PASS"
    assert manifest["source_visit_map_sha256"] == list(result.source_visit_map_sha256)
    assert manifest["visibility_source_sha256"] == SHA_V
    assert manifest["geometry_sources"] == ["ovi_t0", "ovi_t1"]
    with pytest.raises(ValueError, match="already exists"):
        write_two_visit_current_map(result, output)
