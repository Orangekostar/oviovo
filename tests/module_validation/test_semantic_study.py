"""Captured request lineage remains distinct from final semantic target IDs."""

import numpy as np

from src.static_ovmap.module_validation import semantic_study as study
from src.static_ovmap.module_validation.native_capture import RegionRequest


def request(segments, owner=3):
    return RegionRequest("sceneA", 0, f"owner:{owner}", tuple([f"segment:{x}" for x in segments] + [f"owner:{owner}"]),
        "map:0", "a" * 64, (0, 0, 4, 4), "b" * 64, 4,
        "native_global_bbox_union_exclusive_upper_v1", 0, "c" * 64)


def test_request_reconciliation_follows_aliases_without_changing_native_request():
    original = request([10])
    result = study.reconcile_semantic_request(original, {30: {7}, 40: {8}}, [(10, 20), (20, 30)], {7, 8})
    assert result["target_id"] == "owner:7"
    assert result["request_id"] == original.request_id
    assert result["segment_paths"] == [[10, 20, 30]]
    assert original.target_id == "owner:3"
    ambiguous = study.reconcile_semantic_request(request([10, 40]), {30: {7}, 40: {8}}, [(10, 30)], {7, 8})
    assert ambiguous["target_id"] is None
    assert ambiguous["reason"] == "AMBIGUOUS_FINAL_OWNER"
    missing = study.reconcile_semantic_request(request([]), {30: {3}}, [], {3})
    assert missing["target_id"] is None  # Numeric owner reuse alone is not lineage proof.


def test_direct_semantic_fallback_is_not_counted_as_teacher_success():
    manifest = {"selected_targets": ["owner:7", "owner:8"],
        "views": {"owner:7": ["r1", "r2"], "owner:8": ["r3"]},
        "requests": {"r1": {"visible_target_pixels": 10}, "r2": {"visible_target_pixels": 20},
                     "r3": {"visible_target_pixels": 30}}}
    records = {"r1": {"status": "COMPLETE", "feature": np.array([0., 1.])},
               "r2": {"status": "UNAVAILABLE_TECHNICAL_FAILURE"},
               "r3": {"status": "UNAVAILABLE_TECHNICAL_FAILURE"}}
    result = study.direct_readouts(manifest, records, np.eye(2), (2, 5), {7: 2, 8: 2}, model="native")
    assert result["S_NATIVE_AREA"][7]["label_id"] == 5
    assert result["S_NATIVE_VOTE"][7]["successful_views"] == 1
    assert result["S_NATIVE_AREA"][8]["label_id"] == 2
    assert result["S_NATIVE_AREA"][8]["technical_fallback"] is True
    assert result["S_NATIVE_AREA"][8]["successful_views"] == 0


def test_area_fusion_preserves_six_crop_mean_magnitude_before_final_normalization():
    from src.static_ovmap.module_validation.semantic_selector import area_readout

    value = area_readout([[0.1, 0], [0, 1]], [2, 1], np.eye(2), valid_ids=(2, 5))
    assert value.label_id == 5
    np.testing.assert_allclose(value.aggregate_feature, np.array([0.2, 1]) / np.sqrt(1.04))
