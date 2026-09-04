from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.attribute_two_visit_failures import run_two_visit_attribution
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.two_visit_attribution import (
    GHOST_CATEGORIES,
    EvaluatorFailure,
    attribute_two_visit_failures,
    write_two_visit_attribution,
)
from src.oviv2.two_visit_contracts import PairRelation, VisitMap
from src.oviv2.two_visit_current_map import (
    CompositionConfig,
    SignedVisibilityGrid,
    TwoVisitCurrentMap,
    compose_current_map,
    write_two_visit_current_map,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _entity(entity_id: str, x: float) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray([[x, 0.0, 0.0]], dtype=np.float32),
        semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        semantic_label="chair",
        semantic_score=0.9,
        lifecycle_state="current",
        first_seen=0.0,
        last_seen=1.0,
        metadata={"point_rgb_source": "camera_rgb"},
    )


def _current_map() -> tuple[TwoVisitCurrentMap, tuple[PairRelation, ...]]:
    t0_snapshot = MapSnapshot(
        method="OVI-MAP",
        scene_id="Apartment",
        timestamp=1.0,
        entities=[
            _entity("unsafe", 0.0),
            _entity("depth", 1.0),
            _entity("unobserved", 2.0),
            _entity("occluded", 3.0),
            _entity("paired0", 4.0),
        ],
        background_xyz=None,
        scope="current",
    )
    t1_snapshot = MapSnapshot(
        method="OVI-MAP",
        scene_id="Apartment",
        timestamp=2.0,
        entities=[_entity("paired1", 5.0), _entity("false-positive", 6.0)],
        background_xyz=np.asarray([[7.0, 0.0, 0.0]], dtype=np.float32),
        scope="current",
    )
    visits = (
        VisitMap(0, t0_snapshot, "world", SHA_A, 0.01, 0, 9),
        VisitMap(1, t1_snapshot, "world", SHA_A, 0.01, 10, 19),
    )
    relation = PairRelation(
        temporal_query_id="q-paired",
        t0_entity_ids=("paired0",),
        t1_entity_ids=("paired1",),
        state="persistent_moved",
        query_confidence=0.8,
        evidence={"overlap": 0.7},
        identity_source="geometric_baseline",
    )
    visibility = SignedVisibilityGrid(
        voxel_size_m=0.05,
        voxel_keys=np.asarray([[60, 0, 0]], dtype=np.int64),
        statuses=("occluded",),
        source_sha256=SHA_D,
    )
    return (
        compose_current_map(
            visits[0],
            visits[1],
            (relation,),
            visibility,
            CompositionConfig(0.05),
        ),
        (relation,),
    )


def _all_category_failures() -> tuple[EvaluatorFailure, ...]:
    return (
        EvaluatorFailure(
            "g-unsafe",
            "ghost_fp",
            predicted_entity_id="t0-fallback:unsafe",
            predicted_point_index=0,
            evaluator_visibility="visible_free",
            visibility_depth_correct=True,
        ),
        EvaluatorFailure(
            "g-depth",
            "ghost_fp",
            predicted_entity_id="t0-fallback:depth",
            predicted_point_index=0,
            evaluator_visibility="visible_free",
            visibility_depth_correct=False,
        ),
        EvaluatorFailure(
            "g-unobserved",
            "ghost_fp",
            predicted_entity_id="t0-fallback:unobserved",
            predicted_point_index=0,
        ),
        EvaluatorFailure(
            "g-occluded",
            "ghost_fp",
            predicted_entity_id="t0-fallback:occluded",
            predicted_point_index=0,
        ),
        EvaluatorFailure(
            "g-t1",
            "ghost_fp",
            predicted_entity_id="false-positive",
            predicted_point_index=0,
        ),
        EvaluatorFailure(
            "g-pair",
            "ghost_fp",
            predicted_entity_id="paired1",
            predicted_point_index=1,
            expected_relation_id="gt-paired",
            relation_correct=False,
        ),
        EvaluatorFailure(
            "g-background",
            "ghost_fp",
            predicted_entity_id="__background__",
            predicted_point_index=0,
        ),
    )


def test_every_ghost_is_assigned_once_and_mass_is_conserved() -> None:
    current, relations = _current_map()

    result = attribute_two_visit_failures(
        current,
        _all_category_failures(),
        relations,
        evaluator_source_sha256=SHA_E,
        relation_source_sha256=SHA_C,
    )

    assert result.total_ghost_count == 7
    assert sum(result.ghost_counts.values()) == result.total_ghost_count
    assert set(result.ghost_counts) == set(GHOST_CATEGORIES)
    assert result.unattributed_count == 0
    for category in GHOST_CATEGORIES[:-1]:
        assert result.ghost_counts[category] == 1
    assert result.ghost_counts["UNATTRIBUTED"] == 0


def test_each_ghost_row_binds_visit_entity_relation_visibility_and_sources() -> None:
    current, relations = _current_map()

    result = attribute_two_visit_failures(
        current,
        _all_category_failures(),
        relations,
        evaluator_source_sha256=SHA_E,
        relation_source_sha256=SHA_C,
    )

    assert [row.failure_id for row in result.rows] == sorted(
        row.failure_id for row in result.rows
    )
    assert all(row.source_visit in {0, 1} for row in result.rows)
    assert all(row.geometry_source in {"ovi_t0", "ovi_t1"} for row in result.rows)
    assert all(row.semantic_source in {"ovi_t0", "ovi_t1"} for row in result.rows)
    assert all(
        row.source_snapshot_sha256 in set(current.source_visit_map_sha256)
        for row in result.rows
    )
    paired = next(row for row in result.rows if row.failure_id == "g-pair")
    assert paired.relation_id == "q-paired"
    assert paired.expected_relation_id == "gt-paired"
    assert paired.pair_relation_state == "persistent_moved"
    assert paired.t0_ovi_entity_id == "paired0"
    assert paired.t1_ovi_entity_id == "paired1"
    assert paired.identity_source == "geometric_baseline"


def test_unattributed_remains_visible_and_duplicate_events_are_rejected() -> None:
    current, relations = _current_map()
    failure = EvaluatorFailure(
        "missing-object",
        "object_fn",
        target_entity_id="gt:missing",
    )

    result = attribute_two_visit_failures(
        current,
        (failure,),
        relations,
        evaluator_source_sha256=SHA_E,
        relation_source_sha256=SHA_C,
    )

    assert result.total_failure_count == 1
    assert result.unattributed_count == 1
    assert result.unattributed_share == 1.0
    assert result.rows[0].category == "UNATTRIBUTED"
    with pytest.raises(ValueError, match="failure IDs must be unique"):
        attribute_two_visit_failures(
            current,
            (failure, failure),
            relations,
            evaluator_source_sha256=SHA_E,
            relation_source_sha256=SHA_C,
        )


def test_relations_must_exactly_match_composer_identity() -> None:
    current, relations = _current_map()

    with pytest.raises(ValueError, match="relation identities"):
        attribute_two_visit_failures(
            current,
            _all_category_failures(),
            (),
            evaluator_source_sha256=SHA_E,
            relation_source_sha256=SHA_C,
        )
    wrong_members = PairRelation(
        temporal_query_id="q-paired",
        t0_entity_ids=("unsafe",),
        t1_entity_ids=("paired1",),
        state="persistent_moved",
        query_confidence=0.8,
        evidence={"overlap": 0.7},
        identity_source="geometric_baseline",
    )
    with pytest.raises(ValueError, match="not a member"):
        attribute_two_visit_failures(
            current,
            _all_category_failures(),
            (wrong_members,),
            evaluator_source_sha256=SHA_E,
            relation_source_sha256=SHA_C,
        )
    assert relations


def test_object_and_change_errors_remain_one_row_per_evaluator_failure() -> None:
    current, relations = _current_map()
    failures = (
        EvaluatorFailure(
            "object-fp",
            "object_fp",
            predicted_entity_id="false-positive",
        ),
        EvaluatorFailure(
            "object-fn",
            "object_fn",
            target_entity_id="gt:missing",
        ),
        EvaluatorFailure(
            "change-error",
            "change_error",
            predicted_entity_id="paired1",
            predicted_point_index=1,
            expected_relation_id="gt-paired",
            relation_correct=False,
        ),
    )

    result = attribute_two_visit_failures(
        current,
        failures,
        relations,
        evaluator_source_sha256=SHA_E,
        relation_source_sha256=SHA_C,
    )

    assert result.total_failure_count == 3
    assert {row.failure_type for row in result.rows} == {
        "object_fp",
        "object_fn",
        "change_error",
    }
    assert sum(result.category_counts.values()) == 3
    assert next(
        row for row in result.rows if row.failure_id == "change-error"
    ).category == "PAIR_REASONER_WRONG_ID"


def test_attribution_sidecars_are_atomic_hash_bound_and_non_overwriting(
    tmp_path: Path,
) -> None:
    current, relations = _current_map()
    result = attribute_two_visit_failures(
        current,
        _all_category_failures(),
        relations,
        evaluator_source_sha256=SHA_E,
        relation_source_sha256=SHA_C,
    )

    manifest_path = write_two_visit_attribution(result, tmp_path / "attribution")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_path = manifest_path.parent / manifest["artifacts"]["rows"]["path"]
    assert manifest["status"] == "PASS"
    assert manifest["attribution_sha256"] == result.content_sha256()
    assert manifest["artifacts"]["rows"]["sha256"] == hashlib.sha256(
        rows_path.read_bytes()
    ).hexdigest()
    assert len(rows_path.read_text(encoding="utf-8").splitlines()) == 7
    with pytest.raises(ValueError, match="already exists"):
        write_two_visit_attribution(result, tmp_path / "attribution")


def test_runner_loads_hash_bound_current_map_relations_and_evaluator_failures(
    tmp_path: Path,
) -> None:
    current, relations = _current_map()
    current_root = tmp_path / "current"
    write_two_visit_current_map(current, current_root)
    relation_path = tmp_path / "relations.json"
    relation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "relations": [
                    {
                        "temporal_query_id": item.temporal_query_id,
                        "t0_entity_ids": list(item.t0_entity_ids),
                        "t1_entity_ids": list(item.t1_entity_ids),
                        "state": item.state,
                        "query_confidence": item.query_confidence,
                        "evidence": dict(item.evidence),
                        "identity_source": item.identity_source,
                    }
                    for item in relations
                ],
            }
        ),
        encoding="utf-8",
    )
    failure_path = tmp_path / "failures.json"
    failure_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "failures": [asdict(item) for item in _all_category_failures()],
            }
        ),
        encoding="utf-8",
    )

    manifest_path = run_two_visit_attribution(
        current_root=current_root,
        failures_path=failure_path,
        relations_path=relation_path,
        output_root=tmp_path / "attribution",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["current_map_sha256"] == current.content_sha256()
    assert manifest["evaluator_source_sha256"] == hashlib.sha256(
        failure_path.read_bytes()
    ).hexdigest()
    assert manifest["relation_source_sha256"] == hashlib.sha256(
        relation_path.read_bytes()
    ).hexdigest()

    provenance = current_root / "provenance.jsonl"
    provenance.write_text(
        provenance.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        run_two_visit_attribution(
            current_root=current_root,
            failures_path=failure_path,
            relations_path=relation_path,
            output_root=tmp_path / "tampered",
        )
