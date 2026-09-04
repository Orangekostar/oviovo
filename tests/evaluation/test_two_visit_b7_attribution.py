from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.two_visit_b7_attribution import (
    attribute_b7_recovery,
    write_b7_attribution,
)
from src.evaluation.two_visit_snapshot_metrics import TwoVisitEvaluationContext
from src.oviv2.two_visit_contracts import (
    PairRelation,
    VisitMap,
    snapshot_content_sha256,
)
from src.oviv2.two_visit_current_map import (
    CompositionConfig,
    SignedVisibilityGrid,
    compose_current_map,
)
from src.oviv2.two_visit_dense_recovery import (
    DenseRecoveryConfig,
    recover_dense_history,
)
from src.oviv2.two_visit_registration import (
    REGISTRATION_METHOD_ID,
    RegistrationEvidence,
    point_cloud_sha256,
)


def _entity(entity_id: str, points: list[list[float]]) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(points, dtype=np.float32),
        semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        semantic_label="Chair",
        semantic_score=0.9,
        lifecycle_state="current",
        first_seen=0.0,
        last_seen=1.0,
        metadata={"semantic_authority": "ovi"},
    )


def _visit(visit_id: int, entity: EntityPrediction) -> VisitMap:
    start = 0 if visit_id == 0 else 10
    return VisitMap(
        visit_id=visit_id,
        snapshot=MapSnapshot(
            method=f"OVI t{visit_id}",
            scene_id="apartment",
            timestamp=float(start + 4),
            entities=[entity],
            background_xyz=None,
            scope="current",
        ),
        coordinate_frame_id="world",
        source_manifest_sha256="a" * 64,
        map_voxel_size_m=0.01,
        observed_frame_start=start,
        observed_frame_end=start + 4,
    )


def _relation() -> PairRelation:
    return PairRelation(
        temporal_query_id="geom:000000",
        t0_entity_ids=("t0:chair",),
        t1_entity_ids=("t1:chair",),
        state="persistent_moved",
        query_confidence=0.9,
        evidence={"query_score": 0.9},
        identity_source="geometric_baseline",
    )


def _registration(
    relation: PairRelation,
    source: np.ndarray,
    target: np.ndarray,
    *,
    accepted: bool,
) -> RegistrationEvidence:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = 1.0
    return RegistrationEvidence(
        method_id=REGISTRATION_METHOD_ID,
        config_sha256="c" * 64,
        relation_id=str(relation.temporal_query_id),
        relation_state=relation.state,
        identity_source=relation.identity_source,
        t0_entity_ids=relation.t0_entity_ids,
        t1_entity_ids=relation.t1_entity_ids,
        semantic_label="Chair",
        source_points_sha256=point_cloud_sha256(source),
        target_points_sha256=point_cloud_sha256(target),
        source_point_count=len(source),
        target_point_count=len(target),
        source_sample_count=len(source),
        target_sample_count=len(target),
        transform_world_from_t0=transform if accepted else None,
        selected_initialization="synthetic" if accepted else None,
        iteration_count=1 if accepted else 0,
        source_to_target_inlier_count=len(source) if accepted else 0,
        target_to_source_inlier_count=len(target) if accepted else 0,
        source_to_target_overlap=1.0 if accepted else None,
        target_to_source_overlap=1.0 if accepted else None,
        source_to_target_median_m=0.01 if accepted else None,
        target_to_source_median_m=0.01 if accepted else None,
        source_to_target_p90_m=0.02 if accepted else None,
        target_to_source_p90_m=0.02 if accepted else None,
        centroid_residual_m=0.01 if accepted else None,
        centroid_displacement_m=1.0 if accepted else None,
        principal_extent_ratios=(1.0, 1.0, 1.0) if accepted else None,
        accepted=accepted,
        rejection_reasons=() if accepted else ("low_support",),
    )


def _fixture(statuses: tuple[str, str] = ("unobserved", "unobserved")):
    source = _entity("t0:chair", [[0.025, 0.025, 0.025], [0.125, 0.025, 0.025]])
    target = _entity("t1:chair", [[1.225, 0.025, 0.025], [1.325, 0.025, 0.025]])
    t0 = _visit(0, source)
    t1 = _visit(1, target)
    relation = _relation()
    baseline = compose_current_map(
        t0,
        t1,
        (),
        SignedVisibilityGrid.empty(0.05, "b" * 64),
        CompositionConfig(),
    )
    candidates = source.points_xyz + np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    keys = np.floor(candidates / 0.05).astype(np.int64)
    visibility = SignedVisibilityGrid(
        voxel_size_m=0.05,
        voxel_keys=keys,
        statuses=statuses,
        source_sha256="d" * 64,
    )
    registration = _registration(
        relation,
        source.points_xyz,
        target.points_xyz,
        accepted=True,
    )
    result = recover_dense_history(
        baseline,
        t0,
        t1,
        (relation,),
        (registration,),
        visibility,
        DenseRecoveryConfig(maximum_target_surface_distance_m=0.21),
    )
    context = TwoVisitEvaluationContext(
        event_id="event",
        frame_id=14,
        intervention_frame_id=9,
        current_semantic_voxels=np.asarray(
            [[20, 0, 0, 1], [22, 0, 0, 1]], dtype=np.int64
        ),
        changed_region_voxels=np.asarray([[20, 0, 0], [22, 0, 0]], dtype=np.int64),
        confirmed_free_voxels=np.asarray([[22, 0, 0]], dtype=np.int64),
        revealed_background_voxels=np.asarray([[22, 0, 0]], dtype=np.int64),
        current_target_t1_unobserved_mask=np.asarray([True, True], dtype=np.bool_),
    )
    return t0, t1, relation, baseline, result, context


def test_attributes_new_surface_free_space_and_background_without_mutation() -> None:
    t0, t1, relation, baseline, result, context = _fixture()
    before = snapshot_content_sha256(result.snapshot)

    attribution = attribute_b7_recovery(
        result=result,
        baseline=baseline,
        t0=t0,
        t1=t1,
        relations=(relation,),
        evaluation=context,
        evaluation_source_sha256="e" * 64,
    )

    assert attribution.summary["recovered_point_count"] == 2
    assert attribution.summary["newly_covered_gt_surface_count"] == 2
    assert attribution.summary["introduced_confirmed_free_match_count"] == 1
    assert attribution.summary["revealed_background_conflict_count"] == 1
    assert snapshot_content_sha256(result.snapshot) == before
    assert attribution.dense_recovery_sha256 == result.content_sha256()
    assert attribution.baseline_current_map_sha256 == baseline.content_sha256()
    assert attribution.source_t0_snapshot_sha256 == t0.snapshot_sha256
    assert attribution.source_t1_snapshot_sha256 == t1.snapshot_sha256
    row = next(item for item in attribution.rows if item.recovered_point_count)
    assert row.relation_id == "geom:000000"
    assert row.identity_source == "geometric_baseline"
    assert row.registration_quality_bin == "high"
    assert row.visibility_state in {"occluded", "unobserved"}
    assert row.source_visit == 0
    assert row.semantic_label == "Chair"


def test_attributes_rejected_candidates_and_registration_failures() -> None:
    t0, t1, relation, baseline, rejected, context = _fixture(
        ("visible_free", "visible_free")
    )
    attribution = attribute_b7_recovery(
        result=rejected,
        baseline=baseline,
        t0=t0,
        t1=t1,
        relations=(relation,),
        evaluation=context,
        evaluation_source_sha256="e" * 64,
    )
    assert attribution.summary["rejected_visible_free_point_count"] == 2
    assert {row.decision for row in attribution.rows} == {"reject_visible_free"}

    failed_registration = _registration(
        relation,
        t0.snapshot.entities[0].points_xyz,
        t1.snapshot.entities[0].points_xyz,
        accepted=False,
    )
    failed_result = replace(
        rejected, registrations=(failed_registration,), provenance=()
    )
    failed = attribute_b7_recovery(
        result=failed_result,
        baseline=baseline,
        t0=t0,
        t1=t1,
        relations=(relation,),
        evaluation=context,
        evaluation_source_sha256="e" * 64,
    )
    assert failed.summary["registration_rejected_count"] == 1
    assert failed.rows[0].decision == "registration_rejected"
    assert failed.rows[0].registration_rejection_reason == "low_support"

    forged = replace(
        failed_result,
        registrations=(replace(failed_registration, target_points_sha256="f" * 64),),
    )
    with np.testing.assert_raises_regex(ValueError, "registration point binding"):
        attribute_b7_recovery(
            result=forged,
            baseline=baseline,
            t0=t0,
            t1=t1,
            relations=(relation,),
            evaluation=context,
            evaluation_source_sha256="e" * 64,
        )


def test_attribution_writer_is_atomic_and_deterministic(tmp_path: Path) -> None:
    t0, t1, relation, baseline, result, context = _fixture()
    attribution = attribute_b7_recovery(
        result=result,
        baseline=baseline,
        t0=t0,
        t1=t1,
        relations=(relation,),
        evaluation=context,
        evaluation_source_sha256="e" * 64,
    )

    manifest = write_b7_attribution(attribution, tmp_path / "attribution")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["attribution_sha256"] == attribution.content_sha256()
    assert set(payload["artifacts"]) == {"summary", "rows"}
    assert (manifest.parent / payload["artifacts"]["rows"]["path"]).is_file()
