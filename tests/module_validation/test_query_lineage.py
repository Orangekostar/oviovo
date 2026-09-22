"""Current native memberships must route paid evidence without future aliases."""

from dataclasses import replace

import numpy as np
import pytest

from src.static_ovmap.module_validation.query_lineage import CurrentLineage
from src.static_ovmap.module_validation.query_state import (
    AcquisitionPayload,
    FeatureStore,
    NativeCombineState,
    QueryPolicyState,
    dispatch_frame_batch,
)
from tests.module_validation.test_query_gain_policy import _candidate


def _snapshot(mapping, aliases=()):
    return {"label_instances_scope": "all_known_labels", "label_instances": [
        {"segment_label": segment, "instance_label": owner} for segment, owner in mapping.items()],
        "aliases": [{"old_label": a, "resolved_label": b} for a, b in aliases]}


def _pay(lineage, state, combine, candidates, mapping):
    lineage.advance(_snapshot(mapping), state, combine, np.eye(2))
    lineage.register_candidates(candidates)
    results = dispatch_frame_batch(state, FeatureStore(lambda _: AcquisitionPayload(np.array([.2, 1.]), 6, 0)),
                                   candidates, quota=len(candidates))
    lineage.record(candidates, results)


def test_current_merge_then_owner_reuse_does_not_create_permanent_owner_alias():
    lineage, state, combine = CurrentLineage(), QueryPolicyState("Q_AREA", 2), NativeCombineState()
    a, b = replace(_candidate(1, "a", 10), frame_index=0), replace(_candidate(2, "b", 12), frame_index=0)
    _pay(lineage, state, combine, (a, b), {11: 1, 22: 2})
    lineage.advance(_snapshot({11: 2, 22: 2}), state, combine, np.eye(2))
    assert state.object_state(2).attempts == 2
    assert {x.request_id for x in state.object_state(2).features} == {"a", "b"}
    lineage.advance(_snapshot({11: 1, 22: 2}), state, combine, np.eye(2))
    assert [x.request_id for x in state.object_state(1).features] == ["a"]
    assert [x.request_id for x in state.object_state(2).features] == ["b"]
    assert state.logical_ledger.attempts == 2
    assert state.aliases == {}


def test_split_discards_ambiguous_feature_without_refund_or_resurrection():
    lineage, state, combine = CurrentLineage(), QueryPolicyState("Q_AREA", 2), NativeCombineState()
    a = replace(_candidate(1, "mixed", 10), frame_index=0)
    _pay(lineage, state, combine, (a,), {11: 1, 22: 1})
    lineage.advance(_snapshot({11: 1, 22: 2}), state, combine, np.eye(2))
    assert not state.object_state(1).features and not state.object_state(2).features
    assert not state.object_state(1).lineage_available and not state.object_state(2).lineage_available
    assert state.logical_ledger.attempts == 1
    lineage.advance(_snapshot({11: 1, 22: 1}), state, combine, np.eye(2))
    assert not state.object_state(1).features
    assert lineage.dropped_feature_ids == {"mixed"}


def test_segment_alias_is_applied_only_when_captured_and_sparse_capture_blocks():
    lineage, state, combine = CurrentLineage(), QueryPolicyState("Q_AREA", 2), NativeCombineState()
    a = replace(_candidate(1, "a", 10), frame_index=0)
    _pay(lineage, state, combine, (a,), {11: 1})
    assert [x.request_id for x in state.object_state(1).features] == ["a"]
    lineage.advance(_snapshot({22: 2}, aliases=((11, 22),)), state, combine, np.eye(2))
    assert [x.request_id for x in state.object_state(2).features] == ["a"]
    with pytest.raises(ValueError, match="BLOCKED_CAUSAL_LINEAGE"):
        lineage.advance({"label_instances": []}, state, combine, np.eye(2))
