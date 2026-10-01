"""Native/Q provenance and recovery-only frame allowance contracts."""

import copy

import numpy as np
import pytest

from static_ovmap.module_validation.contracts import canonical_digest
from static_ovmap.module_validation.native_capture import RegionRequest


def source_fixture():
    request = RegionRequest("scene", 5, "owner:9", ("segment:2", "owner:9"),
        "snapshot", "1" * 64, (0, 0, 2, 2), "2" * 64, 4, "native", 0, "3" * 64).to_dict()
    rid = request["request_id"]
    registry = {"candidates": [{"raw_owner": 9}]}
    manifest = {"requests": {rid: request}, "native_selected_request_ids": [rid],
        "lineage_proofs": {rid: {"target_id": "owner:9", "reason": "VERIFIED_NATIVE_SEGMENT_LINEAGE"}}}
    saved = {9: {"frame_id": [5], "feat": np.array([[1., 0.]], np.float32),
        "box_2d": [(0, 0, 2, 2)], "pose": [np.eye(4, dtype=np.float32)], "vis_area": [4]}}
    receipt = {"status": "COMPLETE", "native_requests": [{"status": "COMPLETE",
        "request_id": rid, "frame_id": 5, "owner": 9,
        "lineage": ["segment:2", "owner:9"], "source_map_version": "snapshot",
        "encoder_content": {"content_identity": "content", "required_crop_encodings": 6}}]}
    frames = {5: {"pose_c2w": np.eye(4).tolist()}}
    return registry, manifest, saved, receipt, frames, rid


def test_single_native_classifier_preserves_original_canonical_fp32_rule():
    from static_ovmap.module_validation.scannet_study import native_readout
    from static_ovmap.recovery_wave2.recovery_sources import single_native_classifier

    text = np.eye(2, dtype=np.float32)
    canonical = np.array([[1., 1.], [-1., -1.]], np.float32)
    row = single_native_classifier(np.array([1., 0.], np.float32), text, canonical, [9, 17])
    original = native_readout({9: {"frame_id": [5, 5], "feat": np.array([[1., 0.], [1., 0.]], np.float32),
        "vis_area": np.array([4, 4])}}, text, canonical, (9, 17))[9]
    assert row["label"] == original["class_id"] == 9
    assert row["scores"] == pytest.approx([.57270, .33024], abs=1e-4)
    assert row["precision"] == "float32"


def test_native_single_requires_actual_selected_request_and_final_owner_proof():
    from static_ovmap.recovery_wave2.recovery_sources import native_single_sources

    registry, manifest, saved, receipt, frames, rid = source_fixture()
    args = (np.eye(2, dtype=np.float32), np.ones((1, 2), np.float32), [9, 17])
    row = native_single_sources(registry, manifest, saved, receipt, frames, *args)["9"]
    assert row["available"] and row["used_request_ids"] == [rid]
    manifest["lineage_proofs"][rid]["target_id"] = "owner:10"
    row = native_single_sources(registry, manifest, saved, receipt, frames, *args)["9"]
    assert not row["available"]
    assert row["scores"] is None


def test_paid_query_recovery_requires_retained_final_reconciled_lineage():
    from static_ovmap.recovery_wave2.recovery_sources import paid_query_sources

    registry, manifest, _, _, _, rid = source_fixture()
    arrays = {"owner_ids": np.array([9]), "available": np.array([True]),
              "scores": np.array([[.2, .8]]), "valid_ids": np.array([9, 17])}
    trace = {"final_surface_read_after_last_barrier": True, "retained_feature_requests": {"9": [rid]},
             "aliases": {rid: {"frame_id": 5, "lineage": ["segment:2", "owner:9"],
                               "encoder_content": {"content_identity": "content"}}}}
    row = paid_query_sources(registry, manifest, arrays, trace, [9, 17])["9"]
    assert row["available"] and row["label"] == 17
    bad_trace = copy.deepcopy(trace)
    bad_trace["aliases"][rid]["lineage"] = ["segment:3", "owner:9"]
    assert not paid_query_sources(registry, manifest, arrays, bad_trace, [9, 17])["9"]["available"]


def test_fc_allowance_deduplicates_union_and_never_admits_a_33rd_missing_frame():
    from static_ovmap.recovery_wave2.recovery_sources import allocate_fc_requests

    requests = {}
    for i in range(35):
        requests[str(i)] = {"frame_id": i, "image_sha256": f"{i:064x}",
            "target_mask_sha256": "1" * 64, "native_union_mask_sha256": "2" * 64,
            "bbox_xyxy": [0, 0, 2, 2], "crop_convention": "native"}
    requests["alias"] = dict(requests["0"])
    cached = canonical_digest({"model": "model", "rgb": requests["34"]["image_sha256"],
                               "preprocess": "original_800_1333_bilinear_32pad_FP32"})
    plan = allocate_fc_requests([str(i) for i in range(35)], ["alias", "34"], requests,
                               "model", {cached}, max_new_frames=32)
    assert len(plan["authorized_new_image_contents"]) == 32
    assert plan["requests"]["32"]["reason"] == "NEW_FC_FRAME_BUDGET_EXCLUDED"
    assert plan["requests"]["33"]["reason"] == "NEW_FC_FRAME_BUDGET_EXCLUDED"
    assert plan["requests"]["34"]["authorized"]
    assert plan["requests"]["alias"]["authorized"]
    assert len(plan["required_image_contents"]) == 35
