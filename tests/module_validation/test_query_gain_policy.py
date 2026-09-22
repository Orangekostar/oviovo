"""Query ranking, 20+20 utility features, labels, and prefix causality."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.static_ovmap.module_validation.query_gain_policy import (
    QueryFeatureStandardizer,
    QueryTrainingExample,
    build_query_features,
    identifiable_target,
    nll_gain_label,
    predict_query_gain,
    query_target_support_status,
    rank_query_candidates,
    run_query_policy,
    train_query_gain_head,
)
from src.static_ovmap.module_validation.query_state import (
    AcquiredFeature,
    AcquisitionPayload,
    FeatureStore,
    QueryCandidate,
    QueryPolicyState,
)


def _candidate(owner: int, request: str, overlap: int) -> QueryCandidate:
    global_mask = np.zeros((8, 8), dtype=bool)
    global_mask[:4, :4] = True
    local_mask = np.zeros_like(global_mask)
    global_rows = np.flatnonzero(global_mask)
    outside_rows = np.flatnonzero(~global_mask)
    local_mask.reshape(-1)[global_rows[:overlap]] = True
    local_mask.reshape(-1)[outside_rows[: 16 - overlap]] = True
    pose = np.eye(4)
    pose[0, 3] = 1.0
    return QueryCandidate(
        scene_id="scene-a",
        frame_index=2,
        frame_id=20,
        request_id=request,
        owner_id=owner,
        local_entity_id=3,
        majority_entity_zero=False,
        global_mask=global_mask,
        local_mask=local_mask,
        union_mask=global_mask | local_mask,
        bbox_xyxy=(0, 0, 3, 3),
        global_pixels=16,
        local_pixels=16,
        overlap_pixels=overlap,
        valid_depth_fraction=0.75,
        median_depth=2.0,
        depth_iqr=0.5,
        spherical_cells=frozenset((1, 2, 3, 4)),
        camera_pose=pose,
    )


def test_literal_policy_rankings_include_negative_gain_and_deterministic_ties() -> None:
    candidates = (_candidate(2, "b", 10), _candidate(1, "a", 12))
    state = QueryPolicyState("Q_GAIN", class_count=2)
    state.object_state(2).cached_scores = np.array([0.5, 0.5])
    state.object_state(1).cached_scores = np.array([0.9, 0.1])

    assert [row.request_id for row in rank_query_candidates("Q_AREA", candidates, state)] == ["a", "b"]
    assert [row.request_id for row in rank_query_candidates("Q_UNCERTAINTY", candidates, state)] == ["b", "a"]
    gain = rank_query_candidates(
        "Q_GAIN", candidates, state, gain_predictor=lambda _rows: np.array([-2.0, -1.0])
    )
    assert [row.request_id for row in gain] == ["a", "b"]


def test_hand_computed_twenty_features_and_missing_history_bits() -> None:
    candidate = _candidate(1, "a", 8)
    state = QueryPolicyState("Q_GAIN", class_count=2)
    row = build_query_features(candidate, state, frame_count=4, budget=200, spent=20)

    assert row.values.shape == (20,)
    assert row.available.shape == (20,)
    assert row.values[0] == pytest.approx(math.log1p(16))
    assert row.values[1] == pytest.approx(math.log1p(16))
    assert row.values[2:6].tolist() == pytest.approx([0.5, 0.5, 1.0, 0.75])
    assert row.values[8:12].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert not row.available[12:16].any()
    assert row.values[16] == pytest.approx(8.0)
    assert not row.available[17:19].any()
    assert row.values[19] == pytest.approx(0.9)

    object_state = state.object_state(1)
    object_state.attempts = 2
    object_state.last_success_frame = 1
    object_state.best_paid_overlap = 4
    object_state.last_paid_pose = np.eye(4)
    object_state.cached_scores = np.array([0.8, 0.2])
    object_state.features.extend(
        (
            AcquiredFeature("r1", 1, 0, 10, np.array([1.0, 0.0]), frozenset((1,))),
            AcquiredFeature("r2", 1, 1, 9, np.array([0.0, 1.0]), frozenset((2,))),
        )
    )
    observed = build_query_features(candidate, state, frame_count=4, budget=200, spent=20)
    assert observed.available[12:19].all()
    assert observed.values[12] == pytest.approx(0.25)
    assert observed.values[15] == pytest.approx(1.0)
    assert observed.values[16] == pytest.approx(2.0)
    assert observed.values[17] == pytest.approx(1.0)
    assert observed.values[18] == pytest.approx(0.0)


def test_target_rule_gain_label_support_and_fit_only_standardization() -> None:
    annotations = np.array([3] * 80 + [2] * 20 + [0] * 10)
    assert identifiable_target(annotations, allowed_class_ids=(1, 2, 3)) == 3
    assert identifiable_target(np.array([3] * 63), allowed_class_ids=(3,)) is None
    gain = nll_gain_label(None, np.array([0.1, 0.9]), target_index=1)
    assert gain > 0.0
    assert -5.0 <= nll_gain_label(np.array([0.9, 0.1]), np.array([0.1, 0.9]), target_index=0) <= 5.0
    events = [
        {"scene_id": f"s{index % 4}", "target": 1.0 if index < 10 else 0.0}
        for index in range(100)
    ]
    assert query_target_support_status(events) == "SUPPORTED"
    assert query_target_support_status(events[:-1]) == "BLOCKED_QUERY_TARGET_SUPPORT"

    candidate = _candidate(1, "a", 8)
    state = QueryPolicyState("Q_GAIN", class_count=2)
    first = build_query_features(candidate, state, frame_count=4, budget=200, spent=0)
    standardizer = QueryFeatureStandardizer.fit((first,))
    assert standardizer.transform(first).shape == (40,)


def test_fixed_gain_head_and_prefix_sentinel_do_not_read_future_features() -> None:
    pytest.importorskip("torch", exc_type=OSError)
    rng = np.random.default_rng(17)
    fit = tuple(
        QueryTrainingExample(
            scene_id=f"s{index % 4}",
            features=rng.normal(size=40).astype(np.float32),
            target=float((index % 5) - 2),
        )
        for index in range(32)
    )
    cal = fit[:8]
    first = train_query_gain_head(fit, cal_examples=cal, max_epochs=10)
    second = train_query_gain_head(fit, cal_examples=cal, max_epochs=10)
    assert first.status == "COMPLETE"
    assert first.parameter_count == 1345
    for name in first.state_dict:
        np.testing.assert_array_equal(
            first.state_dict[name].numpy(), second.state_dict[name].numpy()
        )
    predictions = predict_query_gain(first.state_dict, np.stack([row.features for row in cal]))
    assert predictions.shape == (8,)
    assert np.all((predictions >= -5.0) & (predictions <= 5.0))

    candidates = (
        replace(_candidate(1, "a", 10), frame_index=0),
        replace(_candidate(2, "future", 9), frame_index=0),
    )
    common_cache = {
        "a": AcquisitionPayload(np.array([1.0, 0.0]), 6, 0.0),
    }
    normal_store = FeatureStore(
        lambda _candidate: (_ for _ in ()).throw(AssertionError("cache miss")),
        initial_cache={
            **common_cache,
            "future": AcquisitionPayload(np.array([0.0, 1.0]), 6, 0.0),
        },
    )
    sentinel_store = FeatureStore(
        lambda _candidate: (_ for _ in ()).throw(AssertionError("cache miss")),
        initial_cache={
            **common_cache,
            "future": AcquisitionPayload(np.array([1e30, -1e30]), 6, 0.0),
        },
    )
    normal = run_query_policy("Q_AREA", (candidates, ()), normal_store, np.eye(2), budget=2)
    sentinel = run_query_policy(
        "Q_AREA", (candidates, ()), sentinel_store, np.eye(2), budget=2
    )
    assert normal.frames[0].ranked_request_ids == sentinel.frames[0].ranked_request_ids
    assert normal.frames[0].attempted_request_ids == sentinel.frames[0].attempted_request_ids
    assert normal.frames[0].attempted_request_ids == ("a",)


def test_replay_advances_empty_frames_and_reconciles_logical_costs() -> None:
    frames = (
        (),
        (replace(_candidate(1, "first", 12), frame_index=1),),
        (replace(_candidate(2, "second", 10), frame_index=2),),
    )
    store = FeatureStore(
        lambda candidate: AcquisitionPayload(
            np.array([1.0, float(candidate.owner_id == 2)]),
            attempted_crop_inputs=6,
            inference_seconds=0.01,
        )
    )
    replay = run_query_policy("Q_AREA", frames, store, np.eye(2), budget=2)

    assert [frame.quota for frame in replay.frames] == [0, 1, 1]
    assert replay.state.logical_ledger.attempts == 2
    assert replay.state.logical_ledger.successes == 2
    assert replay.state.logical_ledger.crop_inputs == 12
    assert [frame.attempted_request_ids for frame in replay.frames] == [
        (),
        ("first",),
        ("second",),
    ]


def test_query_checkpoint_assessment_weights_cal_scenes_equally() -> None:
    rng = np.random.default_rng(17)
    fit = [QueryTrainingExample(f"f{i % 4}", rng.normal(size=40).astype(np.float32), float(i % 3 - 1)) for i in range(16)]
    cal = [QueryTrainingExample("many", np.zeros(40, np.float32), 0.) for _ in range(5)]
    cal.append(QueryTrainingExample("one", np.zeros(40, np.float32), 5.))
    result = train_query_gain_head(fit, cal_examples=cal, max_epochs=5)
    predictions = predict_query_gain(result.state_dict, np.stack([row.features for row in cal]))
    expected = .5 * (np.mean(predictions[:5] ** 2) + (predictions[5] - 5.) ** 2)
    assert result.calibration_mse == pytest.approx(expected)
