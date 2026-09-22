"""Failed visual calls stay unavailable when their transport receipts are reused."""

import json

import numpy as np
from PIL import Image

from src.static_ovmap.module_validation import semantic_models as models
from src.static_ovmap.module_validation.boundary_jobs import file_identity
from src.static_ovmap.module_validation.contracts import (
    atomic_write_json,
    canonical_digest,
)
from src.static_ovmap.module_validation.native_capture import (
    RegionRequest,
    _array_digest,
    _write_npz,
)


def test_exact_wow_name_keeps_exact_label_but_features_use_real_cosines():
    def embed(values):
        vectors = {"chair": [1., 0.], "table": [.8, .6]}
        return np.array([vectors[value] for value in values])

    mapping = models.wow_mapping_with_strengths("chair", ("chair", "table"), embed)
    assert mapping["method"] == "EXACT"
    assert mapping["class_index"] == 0
    np.testing.assert_allclose(mapping["similarities"], [1, .8])
    np.testing.assert_allclose(mapping["top1_top2_gap"], .2)


def make_inputs(tmp_path, *, empty=False):
    root = tmp_path / "capture"
    root.mkdir()
    rgb = np.full((6, 6, 3), [30, 60, 90], np.uint8)
    mask = np.zeros((6, 6), bool)
    mask[1:4, 1:4] = True
    Image.fromarray(rgb).save(root / "rgb.png")
    Image.fromarray(mask.astype(np.uint8)).save(root / "panoptic.png")
    _write_npz(root / "depth.npz", {"depth_m": np.ones((6, 6), np.float32)})
    _write_npz(root / "surface.npz", {"fixture": np.zeros(1)})
    request = RegionRequest("sceneA", 0, "owner:7", ("segment:1", "owner:7"), "map:0",
        _array_digest(mask), (1, 1, 3, 3), _array_digest(mask), 9,
        "native_global_bbox_union_exclusive_upper_v1", 0, file_identity(root / "rgb.png")["sha256"])
    _write_npz(root / "requests.npz", {request.request_id + "_target": mask, request.request_id + "_union": mask})
    frame = {"frame_id": 0, "requests": [] if empty else [request.to_dict()],
        "request_arrays": {"path": "requests.npz", "sha256": file_identity(root / "requests.npz")["sha256"]}}
    for name, suffix in (("rgb", "png"), ("panoptic", "png"), ("depth", "npz")):
        frame[name + "_path"] = f"{name}.{suffix}"
        frame[name + "_sha256"] = file_identity(root / f"{name}.{suffix}")["sha256"]
    capture = {"artifact_type": "OVIMAP_NATIVE_CAPTURE", "scheduled_frame_ids": [0], "completed_frame_ids": [0],
        "frames": [frame], "surface": {"path": "surface.npz", "sha256": file_identity(root / "surface.npz")["sha256"]}}
    capture["identity"] = canonical_digest(capture)
    atomic_write_json(root / "manifest.json", capture)
    manifest = {"capture": file_identity(root / "manifest.json"),
        "requests": {} if empty else {request.request_id: request.to_dict()}}
    manifest["identity"] = canonical_digest(manifest)
    atomic_write_json(tmp_path / "requests.json", manifest)
    model_root = tmp_path / "model"
    model_root.mkdir()
    _write_npz(tmp_path / "text.npz", {"class_names": np.array(["chair", "table"])})
    return tmp_path / "requests.json", {"native_model": str(model_root), "native_text_cache": str(tmp_path / "text.npz")}


def test_failed_request_is_reused_as_failure_not_success(tmp_path, monkeypatch):
    manifest, config = make_inputs(tmp_path)

    class FailingEncoder:
        def encode_images(self, images):
            raise RuntimeError("recorded visual failure")

    monkeypatch.setattr(models.FrozenSiglipBackend, "from_local", lambda *args, **kwargs: FailingEncoder())
    result = models.encode_semantic_requests(manifest, "native", config, tmp_path / "output", device="cpu")
    assert result["request_count"] == 1
    assert result["successful_requests"] == 0
    assert result["crop_inputs"] == 6
    records = models.load_semantic_records(tmp_path / "output")
    assert next(iter(records.values()))["status"] == "UNAVAILABLE_TECHNICAL_FAILURE"
    assert "feature" not in next(iter(records.values()))
    assert models.encode_semantic_requests(manifest, "native", config, tmp_path / "output", device="cpu") == result


def test_empty_manifest_does_not_load_visual_model_and_is_resumable(tmp_path, monkeypatch):
    manifest, config = make_inputs(tmp_path, empty=True)

    def unexpected(*args, **kwargs):
        raise AssertionError("empty request manifest must not load a visual model")

    monkeypatch.setattr(models.FrozenSiglipBackend, "from_local", unexpected)
    result = models.encode_semantic_requests(manifest, "native", config, tmp_path / "output", device="cpu")
    assert result["request_count"] == 0
    assert result["model_load_seconds"] == 0
    assert json.loads((tmp_path / "output/receipt.json").read_text())["outputs"]
    assert models.encode_semantic_requests(manifest, "native", config, tmp_path / "output", device="cpu") == result


def test_static_semantic_request_reads_its_fresh_mask_not_the_captured_parent(tmp_path, monkeypatch):
    from dataclasses import replace

    manifest_path, config = make_inputs(tmp_path)
    old = json.loads(manifest_path.read_text())
    request = RegionRequest.from_dict(next(iter(old["requests"].values())))
    parent = np.zeros((6, 6), bool)
    parent[1:4, 1:4] = True
    target = np.zeros((6, 6), bool)
    target[1:4, 2:5] = True
    union = parent | target
    request = replace(request, target_mask_sha256=_array_digest(target), native_union_mask_sha256=_array_digest(union),
                      bbox_xyxy=(2, 1, 4, 3), visible_target_pixels=6, source_map_version="new_final_mask")
    mask_path = tmp_path / "fresh.npz"
    _write_npz(mask_path, {"target": target, "union": union})
    manifest = {"artifact_type": "OVIMAP_STATIC_FINAL_MASK_REQUESTS", "capture": old["capture"],
                "requests": {request.request_id: {"request": request.to_dict(), "masks": file_identity(mask_path)}}}
    manifest["identity"] = canonical_digest(manifest)
    atomic_write_json(manifest_path, manifest)

    class Encoder:
        def encode_images(self, images):
            return np.tile([1., 0.], (len(images), 1))

    monkeypatch.setattr(models.FrozenSiglipBackend, "from_local", lambda *args, **kwargs: Encoder())
    monkeypatch.setattr(models, "_mask_support", lambda *args: [4, 4, 4])
    result = models.encode_semantic_requests(manifest_path, "native", config, tmp_path / "output", device="cpu")
    row = next(iter(models.load_semantic_records(tmp_path / "output").values()))
    assert row["stats"]["local_global_iou"] == 6 / 12
    assert result["physical_attempts_this_invocation"] == 1
    assert result["physical_crop_inputs_this_invocation"] == 9


def test_context_encoder_failure_preserves_original_six_crop_control(tmp_path, monkeypatch):
    manifest, config = make_inputs(tmp_path)

    class ContextFailure:
        def encode_images(self, images):
            if len(images) == 3:
                raise RuntimeError("background-only failure")
            return np.tile([1., 0.], (6, 1))

    monkeypatch.setattr(models.FrozenSiglipBackend, "from_local", lambda *args, **kwargs: ContextFailure())
    monkeypatch.setattr(models, "_mask_support", lambda *args: [4, 4, 4])
    result = models.encode_semantic_requests(manifest, "native", config, tmp_path / "output", device="cpu")
    assert result["successful_requests"] == 1
    row = next(iter(models.load_semantic_records(tmp_path / "output").values()))
    np.testing.assert_array_equal(row["feature"], [1., 0.])
    assert row["background_scale_usable"] == [False, False, False]
    assert row["background_error"] == "background-only failure"
    assert result["background_crop_inputs"] == 3  # Attempted work is still paid.
