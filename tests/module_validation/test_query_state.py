"""Causal raw candidates, isolated acquisition state, and cost accounting."""

from __future__ import annotations

import numpy as np

from src.static_ovmap.module_validation.query_state import (
    AcquisitionPayload,
    FeatureStore,
    NativeCombineState,
    QueryCandidate,
    QueryPolicyState,
    apply_alias_merge,
    build_query_candidates,
    cumulative_quota,
    dispatch_frame_batch,
    export_query_labels,
    native_combine_candidates,
)


def _candidate(owner: int, request: str, overlap: int, cells=(1, 2)) -> QueryCandidate:
    global_mask = np.zeros((50, 50), dtype=bool)
    global_mask.reshape(-1)[:600] = True
    local_mask = np.zeros_like(global_mask)
    local_mask.reshape(-1)[:overlap] = True
    local_mask.reshape(-1)[600 : 1800 - overlap] = True
    pose = np.eye(4)
    pose[0, 3] = owner
    return QueryCandidate(
        scene_id="scene-a",
        frame_index=0,
        frame_id=10,
        request_id=request,
        owner_id=owner,
        local_entity_id=0,
        majority_entity_zero=True,
        global_mask=global_mask,
        local_mask=local_mask,
        union_mask=global_mask | local_mask,
        bbox_xyxy=(0, 0, 29, 19),
        global_pixels=600,
        local_pixels=1200,
        overlap_pixels=overlap,
        valid_depth_fraction=0.75,
        median_depth=2.0,
        depth_iqr=0.4,
        spherical_cells=frozenset(cells),
        camera_pose=pose,
    )


def test_pure_candidates_keep_majority_zero_and_enforce_native_technical_rules() -> None:
    height = width = 60
    owners = np.zeros((height, width), dtype=np.int64)
    owners[:20, :30] = 1  # 600, accepted with majority local entity 0.
    owners[20:40, :20] = 2  # 400, rejected by global area.
    owners[40:60, :30] = 3  # 600, local entity exists but has area <1000.
    local = np.zeros_like(owners)
    local[35:60, :40] = 7  # 1000 total; all 600 owner3 pixels belong to entity 7.
    depth = np.ones_like(owners, dtype=np.float32)
    valid = np.ones_like(owners, dtype=bool)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    points = np.dstack((grid_x, grid_y, depth)).astype(np.float32)

    candidates = build_query_candidates(
        "scene-a",
        0,
        10,
        owners,
        local,
        depth,
        valid,
        np.eye(4),
        points,
        visibility_area_threshold=1000,
    )

    assert tuple(row.owner_id for row in candidates) == (1, 3)
    assert candidates[0].local_entity_id == 0
    assert candidates[0].majority_entity_zero is True
    assert candidates[0].global_pixels == 600
    assert candidates[1].local_pixels == 1000


def test_native_combine_preserves_coverage_mutation_before_visibility_reject() -> None:
    state = NativeCombineState()
    state.coverage_by_owner[1] = {0}
    state.successful_overlaps_by_owner[1] = list(range(100, 110))
    candidate = _candidate(1, "request-low", 10, cells=(1, 2))

    selected = native_combine_candidates((candidate,), state)

    assert selected == ()
    assert state.coverage_by_owner[1] == {0, 1, 2}


def test_allowance_carries_forward_failed_attempts_debit_and_repeats_skip() -> None:
    assert cumulative_quota(200, frame_index=0, frame_count=3, spent=0) == 66
    assert cumulative_quota(200, frame_index=1, frame_count=3, spent=60) == 73
    state = QueryPolicyState("Q_AREA", class_count=3)
    calls = []

    def loader(candidate):
        calls.append(candidate.request_id)
        if candidate.request_id == "failure":
            return AcquisitionPayload(None, attempted_crop_inputs=4, inference_seconds=0.2)
        return AcquisitionPayload(np.array([1.0, 0.0]), attempted_crop_inputs=6, inference_seconds=0.3)

    store = FeatureStore(loader)
    candidates = (_candidate(1, "success", 20), _candidate(2, "failure", 10))
    results = dispatch_frame_batch(state, store, candidates, quota=2)

    assert [row.success for row in results] == [True, False]
    assert state.logical_ledger.attempts == 2
    assert state.logical_ledger.successes == 1
    assert state.logical_ledger.crop_inputs == 10
    assert store.physical_ledger.model_forwards == 2
    assert dispatch_frame_batch(state, store, candidates, quota=2) == ()
    assert calls == ["success", "failure"]


def test_zero_quota_never_debits_or_retrieves_a_candidate() -> None:
    state = QueryPolicyState("Q_AREA", class_count=2)
    calls = []
    store = FeatureStore(lambda candidate: calls.append(candidate.request_id))
    assert dispatch_frame_batch(state, store, [_candidate(1, "unpaid", 20)], quota=0) == ()
    assert calls == []
    assert state.logical_ledger.attempts == 0


def test_common_query_fusion_preserves_raw_six_crop_mean_magnitudes() -> None:
    state = QueryPolicyState("Q_AREA", class_count=2)
    features = {"weak": np.array([.1, 0.]), "strong": np.array([0., 1.])}
    store = FeatureStore(lambda candidate: AcquisitionPayload(features[candidate.request_id], 6, .1))
    dispatch_frame_batch(state, store, [_candidate(1, "weak", 20), _candidate(1, "strong", 10)], quota=2)
    labels = export_query_labels(state, [1], np.eye(2), valid_class_ids=(4, 7))
    assert labels[1] == 7
    np.testing.assert_allclose(state.successful_features(1)[0].feature, [.1, 0.])


def test_frame_barrier_retention_alias_merge_and_shared_cache_isolation() -> None:
    features = {f"r{index}": np.array([float(index + 1), 1.0]) for index in range(12)}
    store = FeatureStore(
        lambda candidate: AcquisitionPayload(
            features[candidate.request_id], attempted_crop_inputs=6, inference_seconds=0.1
        )
    )
    left = QueryPolicyState("Q_AREA", class_count=2)
    right = QueryPolicyState("Q_UNCERTAINTY", class_count=2)
    candidates = tuple(
        _candidate(1, f"r{index}", overlap=index + 1) for index in range(12)
    )

    dispatch_frame_batch(left, store, candidates, quota=12)
    assert len(left.successful_features(1)) == 10
    assert [row.request_id for row in left.successful_features(1)] == [
        f"r{index}" for index in range(11, 1, -1)
    ]
    assert right.successful_features(1) == ()
    dispatch_frame_batch(right, store, candidates[:1], quota=1)
    assert store.physical_ledger.cache_hits == 1
    assert right.logical_ledger.attempts == 1
    assert right.successful_features(1)[0].request_id == "r0"

    left_state = left.object_state(1)
    left.object_state(2).geometric_cells.add(99)
    apply_alias_merge(left, old_owner=2, new_owner=1)
    assert left.object_state(1) is left_state
    assert 99 in left.object_state(1).geometric_cells

    labels = export_query_labels(
        right,
        (1, 2),
        np.eye(2),
        valid_class_ids=(4, 7),
    )
    assert labels[1] == 4
    assert labels[2] == 0
