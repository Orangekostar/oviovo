"""Production checks for missing-source weights and append-only recovery."""

import numpy as np
import pytest

from static_ovmap.module_validation.evaluation import GeometryIdentity, PredictionPayload
from static_ovmap.module_validation.native_capture import RegionRequest


def test_weight_endpoints_and_partial_sources_keep_inherited_weights():
    from static_ovmap.recovery_wave2.weight_readout import interpolate_probabilities

    equal = np.array([.43333333333333335, .5666666666666667])
    grouped = np.array([.35, .65])
    np.testing.assert_array_equal(interpolate_probabilities(equal, grouped, 1 / 3), equal)
    np.testing.assert_array_equal(interpolate_probabilities(equal, grouped, .5), grouped)
    np.testing.assert_allclose(interpolate_probabilities(equal, grouped, .4), [.4, .6], atol=1e-15)
    np.testing.assert_allclose(interpolate_probabilities(equal, grouped, .45), [.375, .625], atol=1e-15)
    for partial in ([.45, .55], [.25, .75], [.6, .4], [.8, .2], [.4, .6], [.1, .9]):
        for gamma in (.4, .45):
            np.testing.assert_array_equal(interpolate_probabilities(partial, partial, gamma), partial)
    assert interpolate_probabilities(None, None, .4) is None
    with pytest.raises(ValueError, match="availability"):
        interpolate_probabilities(None, grouped, .4)


def test_weight_decisions_preserve_source_free_label_and_class_order_ties():
    from static_ovmap.recovery_wave2.weight_readout import weight_decisions

    endpoints = {
        "2": {"probabilities": None, "available_sources": [], "label": 17},
        "9": {"probabilities": [.5, .5], "available_sources": ["N", "F"], "label": 17},
    }
    labels, audit = weight_decisions(endpoints, endpoints, .4, [17, 3], {2: 17, 9: 3})
    assert labels == {2: 17, 9: 17}
    assert audit["2"]["all_unavailable_fallback"]
    assert audit["2"]["probabilities"] is None
    assert audit["9"]["available_sources"] == ["N", "F"]


def test_registry_excludes_active_owner_and_caps_residual_support_before_views():
    from static_ovmap.recovery_wave2.recovery_registry import build_registry

    raw = np.array([5] * 5 + [7] * 3 + [9] * 4 + [12] * 2)
    painted = np.array([5, 5] + [0] * 12)
    xyz = np.column_stack((np.arange(len(raw)), np.zeros((len(raw), 2)))).astype(np.float32)
    registry = build_registry(xyz, raw, painted, minimum_rows=3, candidate_cap=1)
    assert [c["raw_owner"] for c in registry["candidates"]] == [9]
    assert registry["candidates"][0]["residual_source_rows"] == 4
    assert {(e["raw_owner"], e["reason"]) for e in registry["excluded"]} == {
        (5, "ALREADY_ACTIVE_NATIVE_PAINTED"), (7, "CANDIDATE_CAP_EXCLUDED"),
        (12, "RESIDUAL_SOURCE_ROWS_BELOW_MINIMUM"),
    }


def test_registry_tied_priority_uses_coordinates_not_owner_ids_or_row_order():
    from static_ovmap.recovery_wave2.recovery_registry import build_registry

    xyz = np.array([[0., 0., 0.], [0., 0., 0.], [1., 2., 3.],
                    [10., 0., 0.], [10., 0., 0.], [11., 2., 3.]], np.float32)
    raw = np.array([7, 7, 7, 9, 9, 9])
    first = build_registry(xyz, raw, np.zeros(6, int), minimum_rows=3, candidate_cap=1)
    order = np.array([5, 2, 4, 0, 3, 1])
    renamed = np.where(raw == 7, 999, 1)[order]
    second = build_registry(xyz[order], renamed, np.zeros(6, int), minimum_rows=3, candidate_cap=1)
    assert first["candidates"][0]["support_sha256"] == second["candidates"][0]["support_sha256"]
    assert first["candidates"][0]["residual_source_rows"] == second["candidates"][0]["residual_source_rows"] == 3


def test_request_reconciliation_uses_raw_owners_and_rejects_ambiguous_lineage():
    from static_ovmap.recovery_wave2.recovery_registry import build_registry, reconcile_requests

    raw = np.array([9, 9, 10, 10])
    xyz = np.column_stack((np.arange(4), np.zeros((4, 2)))).astype(np.float32)
    registry = build_registry(xyz, raw, np.zeros(4, int), minimum_rows=1)
    def request(segment, area):
        return RegionRequest("scene", 1, "owner:99", (f"segment:{segment}", "owner:99"),
                             "snapshot", "1" * 64, (0, 0, 2, 2), "2" * 64,
                             area, "native", 0, "3" * 64).to_dict()
    legal, ambiguous = request(1, 2), request(3, 1)
    capture = {"scene_id": "scene", "alias_table": [[1, 2]], "frames": [{
        "frame_id": 1, "map_state_id": "snapshot", "requests": [legal, ambiguous],
        "native_selected_request_ids": [legal["request_id"]],
    }]}
    result = reconcile_requests(capture, np.array([2, 3, 3, 3]), raw, registry)
    assert result["views"]["owner:9"] == [legal["request_id"]]
    assert result["views"]["owner:10"] == []
    assert result["lineage_proofs"][legal["request_id"]]["target_id"] == "owner:9"
    assert result["lineage_proofs"][ambiguous["request_id"]]["reason"] == "AMBIGUOUS_FINAL_OWNER"
    assert result["native_selected_request_ids"] == [legal["request_id"]]


def test_expansion_preserves_old_masks_and_classes_and_recalculates_old_rank():
    from static_ovmap.recovery_wave2.export import expanded_prediction
    from static_ovmap.recovery_wave2.recovery_registry import build_registry

    geometry = GeometryIdentity("0" * 64, "1" * 64, "2" * 64, "projection", 6)
    baseline = PredictionPayload("D2", "COMBO", "scene", geometry,
        np.array([7, 7, 0, 0, 0, 0]), np.array([1, 1, 0, 0, 0, 0]), ((7, 1.),), {})
    baseline.lock()
    raw = np.array([8, 7, 9, 9, 10, 0])
    xyz = np.column_stack((np.arange(6), np.zeros((6, 2)))).astype(np.float32)
    registry = build_registry(xyz, raw, baseline.owner_ids, minimum_rows=1)
    nearest = np.array([0] * 100 + [2] * 200 + [3] * 40 + [4] * 150)
    payload = expanded_prediction(baseline, raw, registry, {9: 1}, [1, 2],
                                  nearest, np.ones(len(nearest), bool), "U", {})
    assert payload.locked
    np.testing.assert_array_equal(payload.owner_ids, [7, 7, 9, 9, 0, 0])
    np.testing.assert_array_equal(payload.semantic_labels, [1, 1, 1, 1, 0, 0])
    assert dict(payload.instance_ranks) == {7: .416667, 9: 1.}
    assert payload.geometry == baseline.geometry
    assert payload.prediction_key != baseline.prediction_key
    with pytest.raises(ValueError, match="candidate"):
        expanded_prediction(baseline, raw, registry, {8: 1}, [1, 2],
                            nearest, np.ones(len(nearest), bool), "U", {})
