"""Disk reuse stays behind a real paid token and preserves six-crop magnitudes."""

from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.static_ovmap.module_validation.assets import sha256_file
from src.static_ovmap.module_validation.native_capture import (
    RegionRequest,
    _array_digest,
)
from src.static_ovmap.module_validation.query_models import NativeQueryLoader
from src.static_ovmap.module_validation.query_state import (
    FeatureStore,
    QueryPolicyState,
    dispatch_frame_batch,
)
from tests.module_validation.test_query_gain_policy import _candidate


def test_query_disk_store_never_encodes_without_debit_and_reuses_exact_request(tmp_path, monkeypatch):
    from src.static_ovmap.module_validation import query_models

    candidate = _candidate(1, "placeholder", 12)
    Image.fromarray(np.full((8, 8, 3), 100, np.uint8)).save(tmp_path / "rgb.png")
    request = RegionRequest("scene-a", 20, "owner:1", ("segment:1", "owner:1"), "map",
        _array_digest(candidate.global_mask), candidate.bbox_xyxy, _array_digest(candidate.union_mask), 12,
        "native_global_bbox_union_exclusive_upper_v1", 0, sha256_file(tmp_path / "rgb.png"))
    from dataclasses import replace

    candidate = replace(candidate, request_id=request.request_id)
    frames = SimpleNamespace(path=tmp_path / "manifest.json", current={"frame": {"frame_id": 20,
        "rgb_path": "rgb.png"}, "requests": {request.request_id: request}})
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    calls = []

    class Backend:
        model = SimpleNamespace(parameters=lambda: ())

        def encode_images(self, images):
            calls.append(len(images))
            return np.array([[1., 0.], [-1., 0.], [0., 1.], [0., 1.], [0., 1.], [0., 1.]])

    monkeypatch.setattr(query_models.FrozenSiglipBackend, "from_local", lambda *args, **kwargs: Backend())
    loader = NativeQueryLoader(frames, {"native_model": str(model)}, tmp_path / "cache", device="cpu")
    store = FeatureStore(loader)
    state = QueryPolicyState("Q_AREA", 2)
    assert dispatch_frame_batch(state, store, [candidate], quota=0) == ()
    assert calls == [] and loader.model_loads == 0
    dispatch_frame_batch(state, store, [candidate], quota=1)
    assert calls == [6]
    np.testing.assert_allclose(state.successful_features(1)[0].feature, [0., 2 / 3])
    second = NativeQueryLoader(frames, {"native_model": str(model)}, tmp_path / "cache", device="cpu")
    second_store, second_state = FeatureStore(second), QueryPolicyState("Q_UNCERTAINTY", 2)
    result = dispatch_frame_batch(second_state, second_store, [candidate], quota=1)
    assert result[0].physical_cache_hit
    assert calls == [6] and second.model_loads == 0
    assert second_store.physical_ledger.model_forwards == 0
    assert second_state.logical_ledger.attempts == 1 and second_state.logical_ledger.crop_inputs == 6
