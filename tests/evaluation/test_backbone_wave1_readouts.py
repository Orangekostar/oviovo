from types import SimpleNamespace

import numpy as np
import pytest

from static_ovmap.backbone_wave1.diagnostics import canonical_partition, instance_diagnostics
from static_ovmap.backbone_wave1.features import TensorEncoderCache
from static_ovmap.backbone_wave1.readouts import fuse_readout
from static_ovmap.backbone_wave1.selection import banded_rank, freeze_candidates
from static_ovmap.backbone_wave1.semantic_readout import reconstruct_native
from static_ovmap.module_validation.query_state import (
    AcquisitionPayload, FeatureStore, QueryPolicyState, dispatch_frame_batch,
)


def test_deferred_native_success_filter_saved_order_and_integer_precision():
    areas = [10, 90, 20, 100, 30, 80, 40, 70, 50, 60, 110, 120]
    metadata = {"view_select_strategy": "combine", "max_top_views": 10, "rows": [{
        "owner": 7, "frame_id": list(range(12)), "vis_area": areas,
        "pose": [np.eye(4).tolist()] * 12, "box_2d": [[0, 0, 4, 4]] * 12, "color": [4, 8, 12]}]}
    records = {(i, 7): np.array([i, 1], np.float32) for i in range(12) if i != 10}
    saved = reconstruct_native(metadata, records)[7]
    valid = [i for i in range(12) if i != 10]
    selected = np.argsort(np.asarray(areas)[valid])[-10:]
    assert saved["frame_id"] == [valid[i] for i in selected]
    assert saved["vis_area"].dtype == np.int64 and saved["feat"].dtype == np.float32
    assert saved["frame_id"][-1] == 11 and 0 not in saved["frame_id"]


def test_encoder_cache_exact_tensor_content_and_separate_model_identity(tmp_path):
    import torch
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = torch.nn.Identity()
        def get_image_features(self, pixel_values):
            return self.vision_model(pixel_values).mean((-1, -2))
    model = Model()
    cache = TensorEncoderCache(model, "weights-A", tmp_path)
    pixels = torch.ones((6, 3, 2, 2), dtype=torch.float32)
    first = model.get_image_features(pixel_values=pixels)
    second = model.get_image_features(pixel_values=pixels.clone())
    torch.testing.assert_close(first, second)
    assert cache.physical_calls == 1 and cache.physical_encodings == 6
    assert cache.last_call["physical_cache_hit"]
    model.get_image_features(pixel_values=pixels + 1)
    assert cache.physical_calls == 2 and len(cache.required_content) == 2
    cache.close()
    other = TensorEncoderCache(model, "weights-B", tmp_path)
    model.get_image_features(pixel_values=pixels)
    assert other.physical_calls == 1
    other.close()


def test_frozen_readouts_renormalize_available_groups_without_fake_sources():
    sources = {name: {"objects": {"7": {"available": True, "scores": scores}}}
               for name, scores in {"N": [1., 0.], "Q": [1., 0.], "F": [0., 1.]}.items()}
    ts = dict.fromkeys(sources, 1.)
    _, audit = fuse_readout(sources, ts, "D2", [3, 9], {7: 3})
    np.testing.assert_allclose(audit["7"]["probabilities"], [.5, .5])
    sources["Q"]["objects"]["7"] = {"available": False, "scores": None}
    _, audit = fuse_readout(sources, ts, "D2", [3, 9], {7: 3})
    np.testing.assert_allclose(audit["7"]["probabilities"], [.5, .5])
    sources["N"]["objects"]["7"] = {"available": False, "scores": None}
    labels, audit = fuse_readout(sources, ts, "D2", [3, 9], {7: 3})
    assert labels[7] == 9 and audit["7"]["available_sources"] == ["F"]


def test_raw_diagnostics_strict_threshold_all_gt_denominator_and_id_renaming():
    gt = np.repeat([1001, 1002, 1003], 200)
    prediction = np.concatenate([np.full(100, 7), np.zeros(100), np.full(200, 8), np.zeros(200)])
    result = instance_diagnostics(prediction, gt, [1])
    assert result["eligible_gt"] == 3
    assert result["matches"]["0.5"]["matched_gt"] == 1
    assert result["matches"]["0.25"]["matched_gt"] == 2
    assert result["matches"]["0.5"]["recall"] == 1 / 3
    assert result["per_gt"][-1]["best_iou"] == 0
    np.testing.assert_array_equal(canonical_partition(prediction),
        canonical_partition(np.where(prediction == 7, 800, np.where(prediction == 8, 300, 0))))


def test_selection_uses_bands_before_standalone_cost_and_keeps_strict_ranking(tmp_path, monkeypatch):
    rows = [{"id": "expensive", "metrics": {"apall": .1004, "miou": .3, "ap50": .4},
             "required_image_encodings": 100, "median_standalone_seconds": 1., "changed_blocks": 1},
            {"id": "cheap", "metrics": {"apall": .1, "miou": .3, "ap50": .4},
             "required_image_encodings": 10, "median_standalone_seconds": 2., "changed_blocks": 1}]
    result = banded_rank(rows)
    assert result["strict_metric_ranking"] == ["expensive", "cheap"]
    assert result["banded_preference"] == ["cheap", "expensive"]
    recipes = [("BB00_NATIVE", "control"), ("BB01_SYNC", "structural"),
               ("BB05_RATIO_GATE", "structural"), ("BB05_FORWARD", "structural"),
               ("BB05_BIDIR", "structural"), ("BB03_SAM2_RAW", "frontend"),
               ("BB03_SAM2_GEOM", "frontend")]
    inputs = [{"id": name, "recipe": {"id": name, "family": family}, "metrics": {
        "apall": .9 if name == "BB05_RATIO_GATE" else .1, "ap50": .4, "miou": .3},
        "required_image_encodings": 10, "median_standalone_seconds": 1.,
        "changed_blocks": 1, "real_intervention": False} for name, family in recipes]
    monkeypatch.setattr("static_ovmap.backbone_wave1.selection.selection_inputs", lambda *args: inputs)
    freeze = freeze_candidates({"output_root": str(tmp_path), "identity": "bound-inputs"},
        {"map_variants": [r["recipe"] for r in inputs]}, "a" * 40)
    assert len(freeze["inputs"]) == 7
    assert freeze["families"]["structural"]["chosen"]["id"] == "BB01_SYNC"
    assert "BB05_RATIO_GATE" not in freeze["families"]["structural"]["ranking"]["banded_preference"]
    assert freeze["composition_recipe"] is None


def test_query_debits_full_batch_before_cache_and_failure_without_refund():
    state = QueryPolicyState("Q_GAIN", 2)
    candidates = [SimpleNamespace(request_id=str(i), owner_id=i + 1, overlap_pixels=100,
        camera_pose=np.eye(4), frame_index=0, spherical_cells=frozenset([1])) for i in range(2)]
    def loader(candidate):
        assert state.logical_ledger.attempts == 2
        return AcquisitionPayload(np.ones(2) if candidate.request_id == "0" else None,
                                  6, 0., "failed" if candidate.request_id == "1" else None,
                                  physical_cache_hit=True)
    store = FeatureStore(loader)
    with pytest.raises(PermissionError):
        store.acquire(candidates[0], state._mint_token("wrong"))
    dispatch_frame_batch(state, store, candidates, quota=2)
    assert state.logical_ledger.attempts == 2 and state.logical_ledger.failures == 1
    assert state.logical_ledger.crop_inputs == 12 and store.physical_ledger.model_forwards == 0
    assert not dispatch_frame_batch(state, store, candidates, quota=2)
