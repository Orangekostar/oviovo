"""Read-only physical native cache reuse and honest query exhaustion accounting."""

from types import SimpleNamespace

import pytest


def test_budget_exhaustion_requires_every_technical_request_to_be_paid():
    from static_ovmap.recovery_wave2.native_worker import validate_query_budget

    frames = SimpleNamespace(schedule=(0, 1), frames={0: {"requests": [{"request_id": "a"}]},
                                                     1: {"requests": [{"request_id": "b"}]}})
    result = {"state": SimpleNamespace(logical_ledger=SimpleNamespace(attempts=2)),
              "decisions": [{"frame_id": 0, "results": [{"request_id": "a"}]},
                            {"frame_id": 1, "results": [{"request_id": "b"}]}]}
    proof = validate_query_budget(result, frames)
    assert proof["reason"] == "TRUE_TECHNICAL_UNIVERSE_EXHAUSTION" and proof["attempts"] == 2
    result["state"].logical_ledger.attempts = 1
    result["decisions"][1]["results"] = []
    with pytest.raises(ValueError, match="exhaust"):
        validate_query_budget(result, frames)


def test_native_parent_cache_is_read_only_and_keyed_by_actual_tensor(tmp_path):
    import numpy as np
    import torch
    from static_ovmap.backbone_wave1.features import TensorEncoderCache
    from static_ovmap.recovery_wave2.native_worker import ReadOnlyNativeCache

    class Model:
        def __init__(self):
            self.vision_model = torch.nn.Identity()
            self.calls = 0

        def get_image_features(self, *, pixel_values):
            self.calls += 1
            return self.vision_model(pixel_values).reshape(len(pixel_values), -1)[:, :3]

    model = Model()
    tensor = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    parent = tmp_path / "parent"
    cache = TensorEncoderCache(model, "model", parent)
    expected = model.get_image_features(pixel_values=tensor).clone()
    cache.close()
    before = {p.relative_to(parent): (p.stat().st_mtime_ns, p.read_bytes()) for p in parent.rglob("*") if p.is_file()}
    cache = ReadOnlyNativeCache(model, "model", tmp_path / "task", [parent])
    actual = model.get_image_features(pixel_values=tensor)
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    assert model.calls == 1 and cache.last_call["physical_cache_hit"]
    model.get_image_features(pixel_values=tensor + 1)
    assert model.calls == 2 and not cache.last_call["physical_cache_hit"]
    cache.close()
    after = {p.relative_to(parent): (p.stat().st_mtime_ns, p.read_bytes()) for p in parent.rglob("*") if p.is_file()}
    assert before == after
    assert len(list((tmp_path / "task").rglob("*.npz"))) == 1
