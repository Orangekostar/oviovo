"""Temporal replay, shared-budget lane selection and model-state separation."""

from dataclasses import replace

import numpy as np
import pytest

from src.static_ovmap.module_validation.query_state import (
    AcquisitionPayload,
    FeatureStore,
    export_query_labels,
)
from tests.module_validation.test_query_gain_policy import _candidate as small_candidate
from tests.module_validation.test_query_lineage import _snapshot
from tests.module_validation.test_query_state import _candidate


def test_fixed_replay_pays_failed_and_later_dropped_attempts_without_resurrection():
    from src.static_ovmap.composition_study.trajectory_reread import replay_fixed

    class Frames:
        scene_id, schedule = "scene-a", (0, 1, 2)

        def load(self, index):
            mapping = {11: 1, 22: 2} if index == 1 else {11: 1, 22: 1}
            candidates = (
                []
                if index == 1
                else [
                    replace(
                        small_candidate(1, f"r{index}", 12),
                        frame_index=index,
                        frame_id=index,
                    )
                ]
            )
            return {"snapshot": _snapshot(mapping), "candidates": candidates}

    trace = [
        {
            "frame_index": i,
            "frame_id": i,
            "results": [] if i == 1 else [{"request_id": f"r{i}"}],
        }
        for i in range(3)
    ]
    calls = []

    def acquire(candidate):
        calls.append(candidate.request_id)
        return AcquisitionPayload(
            np.array([1.0, 0.0]) if candidate.request_id == "r0" else None, 6, 0.1
        )

    result = replay_fixed(Frames(), trace, FeatureStore(acquire), np.eye(2), budget=3)
    assert calls == ["r0", "r2"]
    assert result["state"].logical_ledger.attempts == 2
    assert result["state"].logical_ledger.crop_inputs == 12
    assert result["state"].logical_ledger.failures == 1
    assert result["lineage"].dropped_feature_ids == {"r0"}
    assert export_query_labels(
        result["state"], [1], np.eye(2), valid_class_ids=[4, 7]
    ) == {1: 0}
    trace[2]["results"] = [{"request_id": "r0"}]
    with pytest.raises(ValueError, match="trajectory"):
        replay_fixed(Frames(), trace, FeatureStore(acquire), np.eye(2), budget=3)


def test_fixed_streams_use_the_same_attempts_but_separate_feature_spaces():
    from src.static_ovmap.composition_study.trajectory_reread import replay_fixed

    class Frames:
        scene_id, schedule = "scene-a", (0,)

        def load(self, index):
            return {
                "snapshot": _snapshot({11: 1}),
                "candidates": [
                    replace(small_candidate(1, "r", 12), frame_index=0, frame_id=0)
                ],
            }

    trace = [{"frame_index": 0, "frame_id": 0, "results": [{"request_id": "r"}]}]
    native = replay_fixed(
        Frames(),
        trace,
        FeatureStore(lambda _: AcquisitionPayload(np.array([1.0, 0.0]), 6, 0)),
        np.eye(2),
    )
    other = replay_fixed(
        Frames(),
        trace,
        FeatureStore(lambda _: AcquisitionPayload(np.array([0.0, 1.0]), 6, 0)),
        np.eye(2),
    )
    assert export_query_labels(
        native["state"], [1], np.eye(2), valid_class_ids=[4, 7]
    ) == {1: 4}
    assert export_query_labels(
        other["state"], [1], np.eye(2), valid_class_ids=[4, 7]
    ) == {1: 7}
    assert native["decisions"] == other["decisions"]


def test_mixed_zero_quota_mutates_combine_and_global_slots_fall_back():
    from src.static_ovmap.composition_study.mixed_query import replay_mixed

    class Frames:
        scene_id, schedule = "scene-a", (0, 1, 2, 3)

        def load(self, index):
            candidate = replace(
                _candidate(1, f"r{index}", 20, cells=(1,) if index < 3 else (2,)),
                valid_depth_fraction=1.0,
                frame_index=index,
                frame_id=index,
            )
            return {"snapshot": _snapshot({11: 1}), "candidates": [candidate]}

    result = replay_mixed(
        Frames(),
        FeatureStore(lambda _: AcquisitionPayload(np.array([1.0, 0.0]), 6, 0)),
        np.eye(2),
        predictor=lambda values: np.zeros(len(values)),
        standardizer=None,
        budget=2,
    )
    assert [row["request_id"] for row in result["lane_ledger"]] == ["r1", "r3"]
    assert [row["preferred_lane"] for row in result["lane_ledger"]] == [
        "COMBINE",
        "GAIN",
    ]
    assert [row["winning_lane"] for row in result["lane_ledger"]] == ["GAIN", "GAIN"]
    assert result["lane_ledger"][0]["fallback_reason"] == "PREFERRED_LIST_EXHAUSTED"
    assert result["state"].logical_ledger.attempts == 2
    assert result["combine"].successful_overlaps_by_owner == {1: [20, 20]}
    assert result["decisions"][0]["quota"] == 0
    assert result["decisions"][1]["combine_admitted_request_ids"] == []


def test_mixed_batch_is_ranked_once_and_debited_before_feature_access():
    from src.static_ovmap.composition_study.mixed_query import replay_mixed

    class Frames:
        scene_id, schedule = "scene-a", (0,)

        def load(self, index):
            return {
                "snapshot": _snapshot({11: 1, 22: 2}),
                "candidates": [
                    replace(
                        _candidate(owner, request, overlap),
                        valid_depth_fraction=1.0,
                        frame_id=0,
                    )
                    for owner, request, overlap in ((1, "a", 20), (2, "b", 10))
                ],
            }

    loaded = []

    def predictor(values):
        assert loaded == []  # Same-frame visual results cannot affect ranking.
        return np.array([0.0, 1.0])

    result = replay_mixed(
        Frames(),
        FeatureStore(
            lambda c: (
                loaded.append(c.request_id)
                or AcquisitionPayload(np.array([1.0, 0.0]), 6, 0)
            )
        ),
        np.eye(2),
        predictor=predictor,
        standardizer=None,
        budget=2,
    )
    assert loaded == ["a", "b"]
    assert [row["winning_lane"] for row in result["lane_ledger"]] == ["COMBINE", "GAIN"]
    assert result["decisions"][0]["pre_acquisition_spent"] == 0
    assert result["decisions"][0]["batch_debited"] == 2
