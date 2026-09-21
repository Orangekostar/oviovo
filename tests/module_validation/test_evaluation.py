"""Prediction isolation, payload identities, invariants, and evaluator reuse."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.static_ovmap.module_validation.evaluation import (
    EvaluationMetrics,
    GeometryIdentity,
    PinnedReleasedEvaluator,
    PredictionPayload,
    ReleasedEvaluationAdapter,
    aggregate_scene_metrics,
    validate_prediction_invariants,
)


def _payload(method: str = "N0", branch: str = "N0") -> PredictionPayload:
    return PredictionPayload(
        method_id=method,
        branch=branch,
        scene_id="scene-a",
        geometry=GeometryIdentity("a" * 64, "b" * 64, "c" * 64, "projection-v1", 4),
        owner_ids=np.array([1, 1, 2, 0]),
        semantic_labels=np.array([3, 3, 4, 0]),
        instance_ranks=((1, 0.9), (2, 0.8)),
        logical_cost={"attempts": 2, "crop_inputs": 12},
        metadata={"model_revision": "frozen"},
    )


def test_full_prediction_identity_ignores_method_name_but_not_effective_payload() -> None:
    baseline = _payload()
    copied = _payload("S_COPY", "S")
    assert baseline.prediction_key == copied.prediction_key
    assert baseline.record_key != copied.record_key
    changed = replace(copied, semantic_labels=np.array([3, 3, 5, 0]))
    assert changed.prediction_key != baseline.prediction_key
    with pytest.raises(ValueError, match="forbidden"):
        replace(copied, metadata={"ground_truth_path": "/tmp/gt.npy"})


def test_branch_invariants_fix_s_q_geometry_and_ranks_and_complete_g_partition() -> None:
    native = _payload()
    semantic = replace(native, method_id="S_PAIRED", branch="S")
    validate_prediction_invariants(semantic, native)
    with pytest.raises(ValueError, match="owner"):
        validate_prediction_invariants(
            replace(semantic, owner_ids=np.array([1, 2, 2, 0])), native
        )
    with pytest.raises(ValueError, match="rank"):
        validate_prediction_invariants(
            replace(semantic, instance_ranks=((1, 0.8), (2, 0.9))), native
        )
    geometry = replace(
        native,
        method_id="G_QUALITY",
        branch="G",
        owner_ids=np.array([1, 3, 2, 0]),
        semantic_labels=np.array([3, 3, 4, 0]),
        instance_ranks=((1, 0.25), (2, 0.25), (3, 0.25)),
    )
    validate_prediction_invariants(geometry, native)
    with pytest.raises(ValueError, match="source geometry"):
        validate_prediction_invariants(
            replace(geometry, geometry=replace(geometry.geometry, xyz_sha256="d" * 64)),
            native,
        )


def test_adapter_reuses_identical_prediction_only_after_lock_and_preserves_rows() -> None:
    calls = []

    def evaluator(payload, ground_truth):
        calls.append((payload.method_id, ground_truth["scene_id"]))
        return EvaluationMetrics(
            uap=0.4,
            ap50=0.5,
            ap25=0.6,
            miou=0.7,
            macc=0.8,
            canonical_ap50=None,
            canonical_ap75=None,
            classes_observed=3,
            masks_observed=2,
            trace_parity=True,
        )

    adapter = ReleasedEvaluationAdapter(evaluator)
    first = _payload()
    copied = _payload("S_COPY", "S")
    with pytest.raises(ValueError, match="locked"):
        adapter.evaluate_many((first,), {"scene-a": {"scene_id": "scene-a"}})
    first.lock()
    copied.lock()
    rows = adapter.evaluate_many(
        (first, copied), {"scene-a": {"scene_id": "scene-a"}}
    )
    assert len(calls) == 1
    assert [row.method_id for row in rows] == ["N0", "S_COPY"]
    assert [row.reused_evaluation for row in rows] == [False, True]


def test_prediction_lock_is_irreversible_and_prevents_field_rebinding() -> None:
    payload = _payload()
    payload.lock()

    with pytest.raises(AttributeError, match="locked"):
        payload.semantic_labels = np.array([3, 3, 5, 0])
    with pytest.raises(AttributeError, match="locked"):
        payload._locked = False
    with pytest.raises(ValueError, match="read-only"):
        payload.owner_ids[0] = 2


def test_undefined_scene_metric_makes_required_comparison_inconclusive() -> None:
    rows = (
        EvaluationMetrics(0.4, 0.5, 0.6, 0.7, 0.8, None, None, 3, 2, True),
        EvaluationMetrics(None, 0.4, 0.5, 0.6, 0.7, None, None, 2, 1, True),
    )
    aggregate = aggregate_scene_metrics(rows, required=("uap", "miou"))
    assert aggregate.status == "INCONCLUSIVE_UNDEFINED_METRIC"
    assert aggregate.means["uap"] == pytest.approx(0.4)
    assert aggregate.denominators["uap"] == 1
    assert aggregate.means["miou"] == pytest.approx(0.65)


def test_pinned_adapter_projects_locked_owner_payload_into_released_inputs(tmp_path) -> None:
    payload = _payload()
    payload.lock()
    observed = {}

    def evaluate_set(
        evaluator,
        destination,
        mask_root,
        mask_paths,
        labels,
        serialized,
        kept,
        gt_path,
        cache,
        cache_key,
    ):
        observed["labels"] = labels.tolist()
        observed["serialized"] = serialized
        observed["kept"] = kept.tolist()
        assert evaluator == {"released": True}
        assert mask_root == destination.parent / "prediction"
        assert str(gt_path).endswith("gt.npy")
        assert all(np.load(mask_paths[index]).dtype == bool for index in kept)
        return {
            "released": {"all_ap": 0.4, "all_ap_50%": 0.5, "all_ap_25%": 0.6},
            "trace_parity": {"ap": True, "matches": True},
        }

    def semantic_metrics(gt, prediction, valid_ids):
        np.testing.assert_array_equal(gt, np.array([3, 3, 4, 0]))
        np.testing.assert_array_equal(prediction, np.array([3, 3, 4, 0]))
        assert valid_ids == (3, 4)
        return {"semantic_miou": 1.0, "semantic_macc": 1.0}

    adapter = PinnedReleasedEvaluator(
        {"released": True},
        tmp_path,
        evaluate_set_fn=evaluate_set,
        semantic_metrics_fn=semantic_metrics,
    )
    metrics = adapter(
        payload,
        {
            "nearest": np.arange(4),
            "matched": np.ones(4, dtype=bool),
            "gt_instance_path": tmp_path / "gt.npy",
            "gt_semantic": np.array([3, 3, 4, 0]),
            "valid_ids": (3, 4),
        },
    )
    assert metrics.uap == pytest.approx(0.4)
    assert metrics.trace_parity is True
    assert observed == {
        "labels": [3, 4],
        "serialized": ["0.900000", "0.800000"],
        "kept": [0, 1],
    }
