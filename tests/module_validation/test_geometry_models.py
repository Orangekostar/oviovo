"""G semantic readout never inherits parent labels for changed components."""

import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.static_ovmap.module_validation.geometry_models import final_mask_labels


def test_unavailable_unchanged_mask_keeps_native_but_new_mask_is_unknown():
    native = SimpleNamespace(locked=True, owner_ids=np.array([7, 7, 8, 8, 0]),
                             semantic_labels=np.array([2, 2, 5, 5, 0]))
    owners = np.array([7, 7, 9, 10, 0])
    manifest = {"views": {"owner:7": [], "owner:9": [], "owner:10": ["r"]},
                "requests": {"r": {"request": {"visible_target_pixels": 10}}}}
    labels, ledger = final_mask_labels(native, owners, manifest,
        {"r": {"status": "COMPLETE", "feature": np.array([1., 0.])}}, np.eye(2), (2, 5))
    assert labels == {7: 2, 9: 0, 10: 2}
    assert ledger[9]["status"] == "UNAVAILABLE_NEW_COMPONENT_UNKNOWN"
    assert ledger[7]["status"] == "UNAVAILABLE_UNCHANGED_KEEP_NATIVE"


def test_exact_static_request_reuses_one_physical_forward_across_conditions(tmp_path, monkeypatch):
    from src.static_ovmap.module_validation import geometry_models as models
    from src.static_ovmap.module_validation.boundary_jobs import file_identity
    from src.static_ovmap.module_validation.contracts import canonical_digest
    from src.static_ovmap.module_validation.static_regions import static_region

    image_path = tmp_path / "rgb.png"
    Image.fromarray(np.full((4, 4, 3), 100, np.uint8)).save(image_path)
    raster = np.zeros((4, 4), np.int64)
    raster[1:3, 1:3] = 7
    frame = {"scene_id": "sceneA", "frame_id": 0, "rgb_path": "rgb.png", "rgb_sha256": file_identity(image_path)["sha256"]}
    request, target, union = static_region(raster, (raster > 0).astype(int), 7, frame, "map", (7,), 0)
    mask_path = tmp_path / "masks.npz"
    np.savez(mask_path, target=target, union=union)
    capture_path = tmp_path / "capture.json"
    capture_path.write_text("{}")
    monkeypatch.setattr(models, "verify_capture", lambda *args, **kwargs: {"frames": [frame]})
    manifest = {"capture": file_identity(capture_path), "requests": {request.request_id: {
        "request": request.to_dict(), "masks": file_identity(mask_path)}}}
    manifest["identity"] = canonical_digest(manifest)
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(manifest))
    model_path = tmp_path / "model"
    model_path.mkdir()
    calls = []

    class Backend:
        model = SimpleNamespace(parameters=list)

        def encode_images(self, images):
            calls.append(len(images))
            return np.tile([1., 0.], (6, 1))

    monkeypatch.setattr(models.FrozenSiglipBackend, "from_local", lambda *args, **kwargs: Backend())
    first = models.encode_static_requests(path, {"native_model": str(model_path)}, tmp_path / "cache", tmp_path / "first", device="cpu")
    second = models.encode_static_requests(path, {"native_model": str(model_path)}, tmp_path / "cache", tmp_path / "second", device="cpu")
    assert calls == [6]
    assert first["physical_attempts_this_invocation"] == 1
    assert second["physical_attempts_this_invocation"] == 0
    assert second["logical_crop_inputs"] == 6
    np.testing.assert_array_equal(models.load_static_records(tmp_path / "second")[request.request_id]["feature"], [1., 0.])
