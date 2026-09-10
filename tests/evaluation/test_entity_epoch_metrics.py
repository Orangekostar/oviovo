from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.evaluation.entity_epoch_metrics import (
    EntityEpochEvaluationSupport,
    RelationEvaluationLabel,
    action_attribution_rows,
    evaluate_entity_epoch_actions,
    relation_diagnostic_rows,
    verify_d4_identity_only_invariance,
)
from src.oviv2.current_surface import CurrentEvidenceState, CurrentSurfaceView
from src.oviv2.entity_epoch_update import (
    EntityEpochUpdateResult,
    RelationInferenceState,
    RelationSupport,
    SurfaceActionReason,
    SurfaceStateDelta,
)
from src.oviv2.fine_current_composer import (
    FineVisitSurface,
    compose_entity_epoch_fine_surface,
)
from src.oviv2.fine_dynamic_policy import SurfaceRetirementReason


def _visit(
    visit_id: int,
    source_surface_index: int,
    points: list[list[float]],
    owners: list[int],
    semantics: list[int],
) -> FineVisitSurface:
    count = len(points)
    return FineVisitSurface(
        visit_id=visit_id,
        source_surface_index=source_surface_index,
        geometry_epoch=visit_id,
        vertices_xyz=np.asarray(points, dtype=np.float32),
        normals_xyz=np.tile(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (count, 1)
        ),
        triangles=np.empty((0, 3), dtype=np.int64),
        source_vertex_indices=np.arange(count, dtype=np.int64),
        observed_rgb_uint8=np.full((count, 3), 100, dtype=np.uint8),
        rgb_valid=np.ones(count, dtype=np.bool_),
        last_supported_frames=np.arange(10, 10 + count, dtype=np.int32),
        owner_entity_ids=np.asarray(owners, dtype=np.int64),
        owner_confidences=np.ones(count, dtype=np.float32),
        semantic_ids=np.asarray(semantics, dtype=np.int32),
        semantic_confidences=np.ones(count, dtype=np.float32),
        semantic_support_reliabilities=np.ones(count, dtype=np.float32),
        semantic_source_codes=np.where(np.asarray(semantics) > 0, 1, 0).astype(
            np.uint8
        ),
    )


def _delta(
    rows: list[int],
    *,
    before_valid: bool,
    after_valid: bool,
    before_reason: SurfaceRetirementReason,
    after_reason: SurfaceRetirementReason,
    action: SurfaceActionReason,
) -> SurfaceStateDelta:
    return SurfaceStateDelta(
        source_surface_id="ovi-t0",
        source_vertex_indices=np.asarray(rows, dtype=np.int64),
        owner_entity_id=1,
        stable_entity_id_before="ovi-t0:1",
        stable_entity_id_after="ovi-t0:1",
        episode_id_before="ovi-t0:1/location:t0:1",
        episode_id_after="ovi-t0:1/location:t0:1",
        current_valid_before=before_valid,
        current_valid_after=after_valid,
        retirement_reason_before=before_reason,
        retirement_reason_after=after_reason,
        action_reason=action,
        evidence_frame_ids=(20, 21),
        used_new_measurement=True,
    )


def _composition_and_update():
    t0 = _visit(
        0,
        4,
        [
            [0.01, 0.01, 0.01],
            [0.11, 0.01, 0.01],
            [0.21, 0.01, 0.01],
            [0.31, 0.01, 0.01],
        ],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    )
    t1 = _visit(
        1,
        9,
        [[0.41, 0.01, 0.01], [0.51, 0.01, 0.01]],
        [0, 2],
        [0, 2],
    )
    update = EntityEpochUpdateResult(
        current_valid=np.asarray([False, True, True, True], dtype=np.bool_),
        retirement_reason_codes=np.asarray(
            [
                SurfaceRetirementReason.DIRECT_FREE,
                SurfaceRetirementReason.NONE,
                SurfaceRetirementReason.NONE,
                SurfaceRetirementReason.NONE,
            ],
            dtype=np.uint8,
        ),
        evidence_state_codes=np.asarray(
            [
                CurrentEvidenceState.REVOKED_VISIBLE_FREE,
                CurrentEvidenceState.CURRENT_OBSERVED,
                CurrentEvidenceState.CURRENT_OBSERVED,
                CurrentEvidenceState.HISTORICAL_UNOBSERVED,
            ],
            dtype=np.uint8,
        ),
        surface_deltas=(
            _delta(
                [0],
                before_valid=True,
                after_valid=False,
                before_reason=SurfaceRetirementReason.NONE,
                after_reason=SurfaceRetirementReason.DIRECT_FREE,
                action=SurfaceActionReason.RELIABLE_VISIBLE_FREE,
            ),
            _delta(
                [1, 2],
                before_valid=False,
                after_valid=True,
                before_reason=SurfaceRetirementReason.ENTITY_LIFT,
                after_reason=SurfaceRetirementReason.NONE,
                action=SurfaceActionReason.LOCAL_POSITIVE_CORRECTION,
            ),
        ),
    )
    composition = compose_entity_epoch_fine_surface(
        t0=t0,
        t1=t1,
        update=update,
        surface_id="metrics-test",
    )
    return composition, update


def _relation(
    relation_id: str,
    t0_owner: int,
    t1_owner: int,
    *,
    accepted: bool,
) -> RelationSupport:
    return RelationSupport(
        relation_id=relation_id,
        relation_source="geometry-g1",
        stable_entity_id=f"stable:{t0_owner}",
        t0_owner_entity_id=t0_owner,
        t1_owner_entity_id=t1_owner,
        t0_source_surface_id="ovi-t0",
        t1_source_surface_id="ovi-t1",
        t0_source_vertex_indices=(
            np.asarray([t0_owner - 1], dtype=np.int64)
            if accepted
            else np.empty(0, dtype=np.int64)
        ),
        t1_source_vertex_indices=(
            np.asarray([t1_owner - 10], dtype=np.int64)
            if accepted
            else np.empty(0, dtype=np.int64)
        ),
        confidence=0.9,
        inference_state=RelationInferenceState.UNRESOLVED,
        accepted=accepted,
        assignment_is_null=not accepted,
        rejection_reasons=() if accepted else ("assigned_null",),
    )


def test_entity_epoch_action_metrics_report_rows_voxels_and_denominators() -> None:
    composition, update = _composition_and_update()
    support = EntityEpochEvaluationSupport(
        current_gt_supported_mask=np.asarray(
            [True, False, True, True, False, True], dtype=np.bool_
        ),
        confirmed_free_mask=np.asarray(
            [False, True, False, False, True, False], dtype=np.bool_
        ),
        t1_surface_covered_mask=np.asarray(
            [False, False, False, True, False, False], dtype=np.bool_
        ),
        relation_labels=(),
        voxel_size_m=0.05,
    )

    metrics = evaluate_entity_epoch_actions(
        composition,
        update,
        support,
        relation_support=(),
    )

    assert metrics.deleted_supported_source_rows == 1
    assert metrics.deleted_supported_unique_voxels == 1
    assert metrics.bad_recovery_source_rows == 1
    assert metrics.bad_recovery_unique_voxels == 1
    assert metrics.correct_new_coverage_source_rows == 1
    assert metrics.correct_new_coverage_unique_voxels == 1
    assert metrics.free_conflict_source_rows == 2
    assert metrics.free_conflict_unique_voxels == 2
    assert metrics.deleted_supported_rate.numerator == 1
    assert metrics.deleted_supported_rate.denominator == 2
    assert metrics.bad_recovery_rate.numerator == 1
    assert metrics.bad_recovery_rate.denominator == 2
    assert metrics.free_conflict_rate.numerator == 2
    assert metrics.free_conflict_rate.denominator == 5
    assert composition.surface.semantic_ids[4] == 0
    action_rows = action_attribution_rows(composition, update, support)
    recovery = next(
        row for row in action_rows if row.action_reason == "local_positive_correction"
    )
    assert recovery.source_row_count == 2
    assert recovery.current_gt_supported_rows == 1
    assert recovery.confirmed_free_rows == 1
    assert recovery.used_new_measurement


def test_relation_metrics_include_rejections_and_large_motion_ambiguity() -> None:
    composition, update = _composition_and_update()
    labels = (
        RelationEvaluationLabel(1, 10, is_same_identity=True, large_motion=True),
        RelationEvaluationLabel(2, 20, is_same_identity=False, large_motion=False),
        RelationEvaluationLabel(3, 30, is_same_identity=True, large_motion=False),
        RelationEvaluationLabel(4, 40, is_same_identity=True, large_motion=True),
    )
    support = EntityEpochEvaluationSupport(
        current_gt_supported_mask=np.zeros(6, dtype=np.bool_),
        confirmed_free_mask=np.zeros(6, dtype=np.bool_),
        t1_surface_covered_mask=np.zeros(6, dtype=np.bool_),
        relation_labels=labels,
        voxel_size_m=0.05,
    )

    metrics = evaluate_entity_epoch_actions(
        composition,
        update,
        support,
        relation_support=(
            _relation("r1", 1, 10, accepted=True),
            _relation("r2", 2, 20, accepted=True),
            _relation("r3", 3, 30, accepted=False),
        ),
    )

    assert metrics.relation_precision.numerator == 1
    assert metrics.relation_precision.denominator == 2
    assert metrics.relation_precision.value == pytest.approx(0.5)
    assert metrics.relation_recall.numerator == 1
    assert metrics.relation_recall.denominator == 3
    assert metrics.relation_rejection_rate.numerator == 2
    assert metrics.relation_rejection_rate.denominator == 4
    assert metrics.large_motion_ambiguity_rate.numerator == 2
    assert metrics.large_motion_ambiguity_rate.denominator == 2
    rows = relation_diagnostic_rows(
        (
            _relation("r1", 1, 10, accepted=True),
            _relation("r2", 2, 20, accepted=True),
            _relation("r3", 3, 30, accepted=False),
        ),
        labels,
    )
    assert len(rows) == 4
    false_positive = next(row for row in rows if row.relation_id == "r2")
    assert false_positive.gt_same_identity is False
    missing = next(row for row in rows if row.t0_owner_entity_id == 4)
    assert not missing.prediction_present
    assert missing.gt_large_motion is True


def test_d4_identity_only_requires_exact_headlines_mask_and_semantics() -> None:
    composition, _ = _composition_and_update()
    baseline = composition.surface
    identity_only = replace(baseline, surface_id="d4")
    headline = {
        "current_miou": 0.5,
        "ghost": 0.02,
        "background_f1_at_5cm": 0.4,
        "surface_f1_at_5cm": 0.7,
    }

    result = verify_d4_identity_only_invariance(
        baseline_surface=baseline,
        identity_only_surface=identity_only,
        baseline_headline=headline,
        identity_only_headline=dict(headline),
    )

    assert result.headline_metrics_equal
    assert result.current_mask_equal
    assert result.point_semantics_equal
    changed_semantics = np.array(identity_only.semantic_ids, copy=True)
    changed_semantics[0] += 1
    invalid = CurrentSurfaceView(
        **{
            field: (
                changed_semantics
                if field == "semantic_ids"
                else getattr(identity_only, field)
            )
            for field in identity_only.__dataclass_fields__
        }
    )
    with pytest.raises(ValueError, match="semantic"):
        verify_d4_identity_only_invariance(
            baseline_surface=baseline,
            identity_only_surface=invalid,
            baseline_headline=headline,
            identity_only_headline=dict(headline),
        )
