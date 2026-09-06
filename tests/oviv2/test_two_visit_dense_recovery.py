from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.two_visit_contracts import PairRelation, VisitMap
from src.oviv2.two_visit_current_map import (
    CompositionConfig,
    SignedVisibilityGrid,
    compose_current_map,
)
from src.oviv2.two_visit_dense_recovery import (
    DenseRecoveryConfig,
    historical_output_points,
    read_dense_recovery,
    recover_dense_history,
    write_dense_recovery,
)
from src.oviv2.two_visit_registration import (
    REGISTRATION_METHOD_ID,
    RegistrationEvidence,
    point_cloud_sha256,
)

SOURCE_SHA = "a" * 64
VISIBILITY_SHA = "b" * 64
CONFIG_SHA = "c" * 64


def _entity(
    entity_id: str,
    points: list[list[float]],
    *,
    label: str | None = "Chair",
    score: float = 0.9,
) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(points, dtype=np.float32),
        semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        semantic_label=label,
        semantic_score=score,
        lifecycle_state="current",
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
            method=f"OVI t{visit_id}",
            scene_id="apartment",
            timestamp=float(start + 4),
            entities=entities,
            background_xyz=(
                None if background is None else np.asarray(background, dtype=np.float32)
            ),
            scope="current",
        ),
        coordinate_frame_id="world",
        source_manifest_sha256=SOURCE_SHA,
        map_voxel_size_m=0.01,
        observed_frame_start=start,
        observed_frame_end=start + 4,
    )


def _relation(state: str = "persistent_moved") -> PairRelation:
    return PairRelation(
        temporal_query_id="geom:000001",
        t0_entity_ids=("t0:chair",),
        t1_entity_ids=("t1:chair",),
        state=state,
        query_confidence=0.9,
        evidence={"query_score": 0.9},
        identity_source="geometric_baseline",
    )


def _baseline(t0: VisitMap, t1: VisitMap):
    return compose_current_map(
        t0,
        t1,
        (),
        SignedVisibilityGrid.empty(0.05, VISIBILITY_SHA),
        CompositionConfig(),
    )


def _registration(
    relation: PairRelation,
    source: np.ndarray,
    target: np.ndarray,
    *,
    transform: np.ndarray | None,
    accepted: bool = True,
    semantic_label: str | None = "Chair",
) -> RegistrationEvidence:
    return RegistrationEvidence(
        method_id=REGISTRATION_METHOD_ID,
        config_sha256=CONFIG_SHA,
        relation_id=str(relation.temporal_query_id),
        relation_state=relation.state,
        identity_source=relation.identity_source,
        t0_entity_ids=relation.t0_entity_ids,
        t1_entity_ids=relation.t1_entity_ids,
        semantic_label=semantic_label,
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
        source_to_target_median_m=0.0 if accepted else None,
        target_to_source_median_m=0.0 if accepted else None,
        source_to_target_p90_m=0.0 if accepted else None,
        target_to_source_p90_m=0.0 if accepted else None,
        centroid_residual_m=0.0 if accepted else None,
        centroid_displacement_m=1.0 if accepted else None,
        principal_extent_ratios=(1.0, 1.0, 1.0) if accepted else None,
        accepted=accepted,
        rejection_reasons=() if accepted else ("synthetic_rejection",),
    )


def _transform_x(distance: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = distance
    return transform


def _visibility(
    points: np.ndarray,
    statuses: tuple[str, ...],
) -> SignedVisibilityGrid:
    keys = np.floor(np.asarray(points) / 0.05).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    return SignedVisibilityGrid(
        voxel_size_m=0.05,
        voxel_keys=keys[order],
        statuses=tuple(statuses[index] for index in order),
        source_sha256=VISIBILITY_SHA,
    )


def _moved_fixture():
    source = np.asarray([[0.01, 0.01, 0.01], [0.11, 0.01, 0.01]], dtype=np.float32)
    target = np.asarray([[1.01, 0.11, 0.01], [1.11, 0.11, 0.01]], dtype=np.float32)
    t0 = _visit(0, [_entity("t0:chair", source.tolist())])
    t1 = _visit(1, [_entity("t1:chair", target.tolist(), score=0.97)])
    relation = _relation()
    transform = _transform_x(1.0)
    candidates = source + np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    registration = _registration(
        relation,
        source,
        target,
        transform=transform,
    )
    return t0, t1, relation, candidates, registration


def test_no_registration_returns_the_exact_b3_snapshot() -> None:
    t0, t1, relation, candidates, _registration_value = _moved_fixture()
    baseline = _baseline(t0, t1)

    result = recover_dense_history(
        baseline,
        t0,
        t1,
        (relation,),
        (),
        _visibility(candidates, ("unobserved", "unobserved")),
        DenseRecoveryConfig(),
    )

    assert result.snapshot is baseline.snapshot
    assert result.recovered_point_count == 0
    assert result.removed_baseline_point_count == 0
    assert result.provenance == ()


@pytest.mark.parametrize(
    ("status", "decision"),
    [
        ("visible_free", "reject_visible_free"),
        ("occupied", "reject_occupied"),
    ],
)
def test_current_evidence_never_revives_a_candidate(
    status: str,
    decision: str,
) -> None:
    t0, t1, relation, candidates, registration = _moved_fixture()
    if status == "occupied":
        t1 = _visit(
            1,
            [_entity("t1:chair", [[1.011, 0.011, 0.011], [1.111, 0.011, 0.011]])],
        )
        registration = _registration(
            relation,
            t0.snapshot.entities[0].points_xyz,
            t1.snapshot.entities[0].points_xyz,
            transform=_transform_x(1.0),
        )
    baseline = _baseline(t0, t1)

    result = recover_dense_history(
        baseline,
        t0,
        t1,
        (relation,),
        (registration,),
        _visibility(candidates, (status, status)),
        DenseRecoveryConfig(),
    )

    assert result.snapshot is baseline.snapshot
    assert result.recovered_point_count == 0
    assert {group.decision for group in result.provenance} == {decision}
    assert all(group.output_point_count == 0 for group in result.provenance)


def test_occluded_and_unobserved_points_replace_unwarped_b3_history() -> None:
    t0, t1, relation, candidates, registration = _moved_fixture()
    baseline = _baseline(t0, t1)
    source_before = t0.snapshot.entities[0].points_xyz.copy()
    target_before = t1.snapshot.entities[0].points_xyz.copy()

    result = recover_dense_history(
        baseline,
        t0,
        t1,
        (relation,),
        (registration,),
        _visibility(candidates, ("occluded", "unobserved")),
        DenseRecoveryConfig(),
    )

    assert result.snapshot is not baseline.snapshot
    assert result.recovered_point_count == 2
    assert result.removed_baseline_point_count == 2
    assert [entity.entity_id for entity in result.snapshot.entities] == ["t1:chair"]
    output = result.snapshot.entities[0]
    assert output.semantic_label == "Chair"
    assert output.semantic_score == pytest.approx(0.97)
    assert output.metadata["semantic_authority"] == "ovi_t1"
    assert np.array_equal(output.points_xyz[-2:], candidates)
    assert not np.any(np.all(output.points_xyz[:, None] == source_before[None], axis=2))
    assert np.array_equal(t0.snapshot.entities[0].points_xyz, source_before)
    assert np.array_equal(t1.snapshot.entities[0].points_xyz, target_before)
    assert {group.decision for group in result.provenance} == {
        "recover_occluded",
        "recover_unobserved",
    }
    assert all(
        group.baseline_current_map_sha256 == baseline.content_sha256()
        for group in result.provenance
    )
    assert all(
        group.recovery_config_sha256 == DenseRecoveryConfig().content_sha256()
        for group in result.provenance
    )
    with pytest.raises(ValueError, match="provenance result binding"):
        replace(
            result,
            provenance=(
                replace(
                    result.provenance[0],
                    baseline_current_map_sha256="f" * 64,
                ),
                *result.provenance[1:],
            ),
        )
    assert sum(group.output_point_count for group in result.provenance) == 2
    assert np.array_equal(historical_output_points(result, baseline), candidates)


def test_unlabeled_composite_registration_recovers_nonobserved_history() -> None:
    t0, t1, relation, candidates, _ = _moved_fixture()
    source = t0.snapshot.entities[0].points_xyz
    target = t1.snapshot.entities[0].points_xyz
    t0 = _visit(0, [_entity("t0:chair", source.tolist(), label=None)])
    t1 = _visit(1, [_entity("t1:chair", target.tolist(), label=None)])
    registration = _registration(
        relation,
        source,
        target,
        transform=_transform_x(1.0),
        semantic_label=None,
    )
    baseline = _baseline(t0, t1)

    result = recover_dense_history(
        baseline,
        t0,
        t1,
        (relation,),
        (registration,),
        _visibility(candidates, ("occluded", "unobserved")),
        DenseRecoveryConfig(),
    )

    assert registration.accepted
    assert registration.semantic_label is None
    assert result.recovered_point_count == 2


def test_incompatible_candidate_preserves_b3_fallback() -> None:
    t0, _t1, relation, candidates, _registration_value = _moved_fixture()
    far_target = _visit(1, [_entity("t1:chair", [[5.0, 5.0, 5.0]])])
    registration = _registration(
        relation,
        t0.snapshot.entities[0].points_xyz,
        far_target.snapshot.entities[0].points_xyz,
        transform=_transform_x(1.0),
    )
    baseline = _baseline(t0, far_target)

    result = recover_dense_history(
        baseline,
        t0,
        far_target,
        (relation,),
        (registration,),
        _visibility(candidates, ("unobserved", "unobserved")),
        DenseRecoveryConfig(),
    )

    assert result.snapshot is baseline.snapshot
    assert {group.decision for group in result.provenance} == {"reject_incompatible"}


@pytest.mark.parametrize(
    "state",
    ["appeared", "removed_candidate", "uncertain", "split", "merge"],
)
def test_nonpersistent_relations_leave_b3_unchanged(state: str) -> None:
    t0, t1, _relation_value, candidates, _registration_value = _moved_fixture()
    if state == "split":
        t1 = _visit(
            1,
            [
                t1.snapshot.entities[0],
                _entity("t1:other", [[2.0, 0.0, 0.0]], label="Table"),
            ],
        )
    if state == "merge":
        t0 = _visit(
            0,
            [
                t0.snapshot.entities[0],
                _entity("t0:other", [[-2.0, 0.0, 0.0]], label="Table"),
            ],
        )
    ids = {
        "appeared": ((), ("t1:chair",)),
        "removed_candidate": (("t0:chair",), ()),
        "uncertain": (("t0:chair",), ("t1:chair",)),
        "split": (("t0:chair",), ("t1:chair", "t1:other")),
        "merge": (("t0:chair", "t0:other"), ("t1:chair",)),
    }[state]
    relation = PairRelation(
        temporal_query_id=f"q:{state}",
        t0_entity_ids=ids[0],
        t1_entity_ids=ids[1],
        state=state,
        query_confidence=0.8,
        evidence={"query_score": 0.8},
        identity_source="geometric_baseline",
    )
    baseline = _baseline(t0, t1)

    result = recover_dense_history(
        baseline,
        t0,
        t1,
        (relation,),
        (),
        _visibility(candidates, ("unobserved", "unobserved")),
        DenseRecoveryConfig(),
    )

    assert result.snapshot is baseline.snapshot
    assert result.provenance == ()


def test_registration_must_bind_the_exact_source_and_target_points() -> None:
    t0, t1, relation, candidates, registration = _moved_fixture()
    baseline = _baseline(t0, t1)
    wrong_source = t0.snapshot.entities[0].points_xyz.copy()
    wrong_source[0, 0] += 0.001
    registration = replace(
        registration,
        source_points_sha256=point_cloud_sha256(wrong_source),
    )

    with pytest.raises(ValueError, match="registration point binding"):
        recover_dense_history(
            baseline,
            t0,
            t1,
            (relation,),
            (registration,),
            _visibility(candidates, ("unobserved", "unobserved")),
            DenseRecoveryConfig(),
        )


def test_rejected_registration_still_requires_exact_input_bindings() -> None:
    t0, t1, relation, candidates, _registration_value = _moved_fixture()
    baseline = _baseline(t0, t1)
    rejected = _registration(
        relation,
        t0.snapshot.entities[0].points_xyz,
        t1.snapshot.entities[0].points_xyz,
        transform=None,
        accepted=False,
    )
    rejected = replace(rejected, source_points_sha256="f" * 64)

    with pytest.raises(ValueError, match="registration point binding"):
        recover_dense_history(
            baseline,
            t0,
            t1,
            (relation,),
            (rejected,),
            _visibility(candidates, ("unobserved", "unobserved")),
            DenseRecoveryConfig(),
        )


def test_provenance_reject_incompatible_requires_nonobserved_visibility() -> None:
    t0, t1, relation, candidates, registration = _moved_fixture()
    baseline = _baseline(t0, t1)
    result = recover_dense_history(
        baseline,
        t0,
        t1,
        (relation,),
        (registration,),
        _visibility(candidates, ("visible_free", "visible_free")),
        DenseRecoveryConfig(),
    )

    with pytest.raises(ValueError, match="decision, visibility"):
        replace(
            result.provenance[0],
            decision="reject_incompatible",
            compatible=False,
        )


def test_candidate_visibility_must_cover_every_transformed_voxel() -> None:
    t0, t1, relation, _candidates, registration = _moved_fixture()
    baseline = _baseline(t0, t1)

    with pytest.raises(ValueError, match="candidate visibility is incomplete"):
        recover_dense_history(
            baseline,
            t0,
            t1,
            (relation,),
            (registration,),
            SignedVisibilityGrid.empty(0.05, VISIBILITY_SHA),
            DenseRecoveryConfig(),
        )


def test_result_hash_and_atomic_artifact_are_deterministic(tmp_path: Path) -> None:
    t0, t1, relation, candidates, registration = _moved_fixture()
    baseline = _baseline(t0, t1)
    visibility = _visibility(candidates, ("occluded", "unobserved"))
    first = recover_dense_history(
        baseline,
        t0,
        t1,
        (relation,),
        (registration,),
        visibility,
        DenseRecoveryConfig(),
    )
    second = recover_dense_history(
        baseline,
        t0,
        t1,
        (relation,),
        (registration,),
        visibility,
        DenseRecoveryConfig(),
    )

    assert first.content_sha256() == second.content_sha256()
    manifest_path = write_dense_recovery(first, tmp_path / "b7")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_id"] == "OVI_TWO_VISIT_B7_DENSE_RECOVERY_V1"
    assert manifest["dense_recovery_sha256"] == first.content_sha256()
    assert manifest["recovered_point_count"] == 2
    assert (manifest_path.parent / "registration.json").is_file()
    assert (manifest_path.parent / "recovery-provenance.jsonl").is_file()
    loaded = read_dense_recovery(manifest_path)
    assert loaded.content_sha256() == first.content_sha256()
    with pytest.raises(ValueError, match="already exists"):
        write_dense_recovery(first, tmp_path / "b7")
