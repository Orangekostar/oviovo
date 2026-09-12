"""The bridge must not conflate unknown entities and explicit background."""

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


def fixture():
    crosswalk = TesseSemanticCrosswalk(
        scene="apartment",
        unknown=SemanticLookup(0, "Unknown", False),
        aliases={
            "chair": SemanticLookup(1, "Chair", True),
            "floor": SemanticLookup(3, "Floor", True),
        },
        valid_semantic_ids=frozenset({1}),
        label_space_sha256="a" * 64,
        alias_config_sha256="b" * 64,
    )
    xyz = np.array(
        [
            [0.025, 0.025, 1.025],
            [1.025, 0.025, 1.025],
            [2.025, 0.025, 1.025],
            [3.025, 0.025, 1.025],
        ],
        np.float32,
    )
    entities = [
        EntityPrediction(
            entity_id=str(i),
            points_xyz=xyz[i : i + 1],
            semantic_embedding=None,
            semantic_label=label,
            semantic_score=0.9,
            lifecycle_state="current",
            first_seen=1.0,
            last_seen=2.0,
        )
        for i, label in enumerate(["chair", "floor", None])
    ]
    snapshot = MapSnapshot(
        method="test",
        scene_id="apartment",
        timestamp=2.0,
        entities=entities,
        background_xyz=xyz[3:],
        scope="current",
    )
    context = TwoVisitEvaluationContext(
        event_id="test",
        frame_id=2,
        intervention_frame_id=1,
        current_semantic_voxels=np.array([[0, 0, 20, 1], [40, 0, 20, 1]]),
        changed_region_voxels=np.array(
            [[0, 0, 20], [20, 0, 20], [40, 0, 20], [60, 0, 20]]
        ),
        confirmed_free_voxels=np.array([[40, 0, 20]]),
        revealed_background_voxels=np.array([[20, 0, 20], [60, 0, 20]]),
        current_target_t1_unobserved_mask=np.array([False, False]),
    )
    return crosswalk, xyz, snapshot, context


def test_legacy_roles_and_supported_updates_are_separate_from_owner():
    from src.oviv2.surface_readout import legacy_roles, resolve_semantic_update

    crosswalk, _, _, _ = fixture()
    ids = np.array([1, 3, 0, 0])
    sources, roles = legacy_roles(ids, np.array([False, False, False, True]), crosswalk)
    assert sources.tolist() == [2, 1, 3, 0]
    assert roles.tolist() == [1, 0, 2, 0]
    new_ids, new_roles = resolve_semantic_update(
        ids, roles, np.array([3, 0, 99, 1]), np.array([1, 2, 0, 1]), crosswalk
    )
    assert new_ids.tolist() == [3, 0, 0, 1]
    assert new_roles.tolist() == [0, 2, 2, 1]
    assert ids.tolist() == [1, 3, 0, 0]


def test_pointwise_scores_match_legacy_and_consume_new_prediction():
    from src.evaluation.two_visit_snapshot_metrics import evaluate_two_visit_points

    crosswalk, xyz, snapshot, context = fixture()
    extra = {
        "retained_t0_xyz": np.empty((0, 3), np.float32),
        "retained_t0_t1_observed_mask": np.empty(0, bool),
    }
    old = evaluate_two_visit_snapshot(snapshot, context, crosswalk, **extra)
    kwargs = dict(
        points_xyz=xyz,
        semantic_ids=np.array([1, 3, 0, 0]),
        eval_role=np.array([1, 0, 2, 0]),
        owner_ids=np.array([1, 2, 3, 0]),
        scene_id="apartment",
        timestamp=2.0,
        context=context,
        crosswalk=crosswalk,
        **extra,
    )
    actual = evaluate_two_visit_points(**kwargs)
    assert actual == pytest.approx(old, abs=1e-6)
    assert actual["current_miou"] == pytest.approx(0.5)
    assert actual["ghost"] > 0
    kwargs["owner_ids"] = np.array([0, 99, 55, 2])
    assert evaluate_two_visit_points(**kwargs) == pytest.approx(old, abs=1e-6)
    kwargs["semantic_ids"] = np.array([1, 3, 1, 0])
    kwargs["eval_role"] = np.array([1, 0, 1, 0])
    assert evaluate_two_visit_points(**kwargs)["current_miou"] == pytest.approx(1.0)


def test_invalid_new_known_class_cannot_enter_evaluation():
    from src.oviv2.surface_readout import resolve_semantic_update

    crosswalk, _, _, _ = fixture()
    with pytest.raises(ValueError, match="known"):
        resolve_semantic_update(
            np.array([1]), np.array([1]), np.array([99]), np.array([1]), crosswalk
        )
