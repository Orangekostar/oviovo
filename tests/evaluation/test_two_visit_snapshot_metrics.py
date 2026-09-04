from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.baselines.tesse_semantics import (
    SemanticLookup,
    TesseSemanticCrosswalk,
)
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.two_visit_snapshot_metrics import (
    TwoVisitEvaluationContext,
    evaluate_two_visit_snapshot,
)


def _crosswalk() -> TesseSemanticCrosswalk:
    unknown = SemanticLookup(semantic_id=0, native_name="Unknown", matched=False)
    chair = SemanticLookup(semantic_id=1, native_name="Chair", matched=True)
    return TesseSemanticCrosswalk(
        scene="apartment",
        unknown=unknown,
        aliases={"chair": chair},
        valid_semantic_ids=frozenset({1}),
        label_space_sha256="a" * 64,
        alias_config_sha256="b" * 64,
    )


def _snapshot() -> MapSnapshot:
    return MapSnapshot(
        method="OVI_T1_ONLY",
        scene_id="apartment",
        timestamp=1472.0,
        entities=[
            EntityPrediction(
                entity_id="ovimap:1",
                points_xyz=np.asarray([[0.025, 0.025, 1.025]], dtype=np.float32),
                semantic_embedding=None,
                semantic_label="chair",
                semantic_score=0.9,
                lifecycle_state="current",
                first_seen=1217.0,
                last_seen=1472.0,
            )
        ],
        background_xyz=np.asarray([[1.025, 0.025, 1.025]], dtype=np.float32),
        scope="current",
    )


def test_evaluates_common_v2_and_completeness_without_hidden_scalar() -> None:
    context = TwoVisitEvaluationContext(
        event_id="apartment_event_04",
        frame_id=1472,
        intervention_frame_id=1022,
        current_semantic_voxels=np.asarray([[0, 0, 20, 1]], dtype=np.int64),
        changed_region_voxels=np.asarray([[0, 0, 20], [20, 0, 20]], dtype=np.int64),
        confirmed_free_voxels=np.asarray([[5, 0, 20]], dtype=np.int64),
        revealed_background_voxels=np.asarray([[20, 0, 20]], dtype=np.int64),
        current_target_t1_unobserved_mask=np.asarray([False], dtype=np.bool_),
    )

    metrics = evaluate_two_visit_snapshot(
        _snapshot(),
        context,
        _crosswalk(),
        retained_t0_xyz=np.empty((0, 3), dtype=np.float32),
        retained_t0_t1_observed_mask=np.empty((0,), dtype=np.bool_),
    )

    assert metrics["current_miou"] == pytest.approx(1.0)
    assert metrics["ghost"] == pytest.approx(0.0)
    assert metrics["background_f1_at_5cm"] == pytest.approx(1.0)
    assert metrics["surface_precision_at_5cm"] == pytest.approx(1.0)
    assert metrics["surface_recall_at_5cm"] == pytest.approx(1.0)
    assert metrics["surface_f1_at_5cm"] == pytest.approx(1.0)
    assert metrics["total_current_surface_coverage"] == pytest.approx(1.0)
    assert metrics["t1_unobserved_region_recall"] == pytest.approx(1.0)
    assert metrics["t1_observed_region_stale_precision"] == pytest.approx(1.0)


def test_known_nonobject_entity_is_background_not_ghost() -> None:
    snapshot = _snapshot()
    snapshot.entities.append(
        EntityPrediction(
            entity_id="ovimap:2",
            points_xyz=np.asarray([[0.275, 0.025, 1.025]], dtype=np.float32),
            semantic_embedding=None,
            semantic_label="Floor",
            semantic_score=0.9,
            lifecycle_state="current",
            first_seen=1217.0,
            last_seen=1472.0,
        )
    )
    snapshot.background_xyz = None
    crosswalk = _crosswalk()
    crosswalk.aliases["floor"] = SemanticLookup(
        semantic_id=3,
        native_name="Floor",
        matched=True,
    )
    context = TwoVisitEvaluationContext(
        event_id="apartment_event_04",
        frame_id=1472,
        intervention_frame_id=1022,
        current_semantic_voxels=np.asarray([[0, 0, 20, 1]], dtype=np.int64),
        changed_region_voxels=np.asarray([[0, 0, 20], [5, 0, 20]], dtype=np.int64),
        confirmed_free_voxels=np.asarray([[5, 0, 20]], dtype=np.int64),
        revealed_background_voxels=np.asarray([[5, 0, 20]], dtype=np.int64),
        current_target_t1_unobserved_mask=np.asarray([False], dtype=np.bool_),
    )

    metrics = evaluate_two_visit_snapshot(
        snapshot,
        context,
        crosswalk,
        retained_t0_xyz=np.empty((0, 3), dtype=np.float32),
        retained_t0_t1_observed_mask=np.empty((0,), dtype=np.bool_),
    )

    assert metrics["ghost"] == pytest.approx(0.0)
    assert metrics["background_f1_at_5cm"] == pytest.approx(1.0)
    assert metrics["surface_f1_at_5cm"] == pytest.approx(1.0)


def test_unmatched_entity_remains_conservative_object_for_ghost() -> None:
    snapshot = _snapshot()
    snapshot.entities[0].points_xyz = np.asarray(
        [[0.275, 0.025, 1.025]], dtype=np.float32
    )
    snapshot.entities[0].semantic_label = "unmatched open vocabulary label"
    snapshot.background_xyz = None
    context = TwoVisitEvaluationContext(
        event_id="apartment_event_04",
        frame_id=1472,
        intervention_frame_id=1022,
        current_semantic_voxels=np.asarray([[5, 0, 20, 1]], dtype=np.int64),
        changed_region_voxels=np.asarray([[5, 0, 20]], dtype=np.int64),
        confirmed_free_voxels=np.asarray([[5, 0, 20]], dtype=np.int64),
        revealed_background_voxels=np.empty((0, 3), dtype=np.int64),
        current_target_t1_unobserved_mask=np.asarray([False], dtype=np.bool_),
    )

    metrics = evaluate_two_visit_snapshot(
        snapshot,
        context,
        _crosswalk(),
        retained_t0_xyz=np.empty((0, 3), dtype=np.float32),
        retained_t0_t1_observed_mask=np.empty((0,), dtype=np.bool_),
    )

    assert metrics["ghost"] == pytest.approx(1.0)


def test_rejects_nonmatching_scene_or_final_frame() -> None:
    context = TwoVisitEvaluationContext(
        event_id="apartment_event_04",
        frame_id=1472,
        intervention_frame_id=1022,
        current_semantic_voxels=np.asarray([[0, 0, 20, 1]], dtype=np.int64),
        changed_region_voxels=np.asarray([[0, 0, 20]], dtype=np.int64),
        confirmed_free_voxels=np.empty((0, 3), dtype=np.int64),
        revealed_background_voxels=np.empty((0, 3), dtype=np.int64),
        current_target_t1_unobserved_mask=np.asarray([False], dtype=np.bool_),
    )
    snapshot = _snapshot()
    snapshot.scene_id = "office"

    with pytest.raises(ValueError, match="scene"):
        evaluate_two_visit_snapshot(
            snapshot,
            context,
            _crosswalk(),
            retained_t0_xyz=np.empty((0, 3), dtype=np.float32),
            retained_t0_t1_observed_mask=np.empty((0,), dtype=np.bool_),
        )
