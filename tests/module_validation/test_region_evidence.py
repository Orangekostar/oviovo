"""Region-manifest, crop, and frozen-adapter contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.static_ovmap.module_validation.native_capture import (
    FrameObservation,
    RegionRequest,
)
from src.static_ovmap.module_validation.region_evidence import (
    WOW_PROMPT,
    NativeRegionEncoder,
    OfficialWowRunner,
    WowRegionAdapter,
    build_target_view_manifest,
    clean_wow_category_response,
    map_generated_name,
    native_crops,
    stable_target_subset,
)
from src.static_ovmap.module_validation.rgb_siglip import FrozenSiglipBackend

SHA = "a" * 64


def _request(
    *,
    frame_id: int,
    owner: int,
    visible: int,
    rank: int = 0,
    lineage: tuple[str, ...] | None = None,
) -> RegionRequest:
    target = f"owner:{owner}"
    return RegionRequest(
        scene_id="sceneA",
        frame_id=frame_id,
        target_id=target,
        lineage=lineage or (f"segment:{owner + 10}", target),
        source_map_version=f"map:{frame_id}",
        target_mask_sha256=SHA,
        bbox_xyxy=(2, 2, 12, 12),
        native_union_mask_sha256="b" * 64,
        visible_target_pixels=visible,
        crop_convention="native_minmax_exclusive_slice_v1",
        requested_view_rank=rank,
        image_sha256="c" * 64,
    )


def _frame(frame_id: int, requests: tuple[RegionRequest, ...]) -> FrameObservation:
    return FrameObservation(
        scene_id="sceneA",
        frame_id=frame_id,
        pose_c2w=np.eye(4),
        image_size_hw=(16, 16),
        intrinsics=np.eye(3),
        rgb_path=f"frames/{frame_id}/rgb.png",
        rgb_sha256=SHA,
        depth_path=f"frames/{frame_id}/depth.npz",
        depth_sha256="b" * 64,
        panoptic_path=f"frames/{frame_id}/panoptic.png",
        panoptic_sha256="c" * 64,
        global_owner_path=f"frames/{frame_id}/owner.png",
        global_owner_sha256="d" * 64,
        map_state_id=f"map:{frame_id}",
        refined_segment_ids=np.arange(len(requests), dtype=np.int64),
        registered_labels=np.arange(len(requests), dtype=np.int64),
        requests=requests,
        native_selected_request_ids=(),
        request_completion_boundary=0,
    )


def test_target_hash_cap_and_three_view_order_are_label_free() -> None:
    selected, excluded = stable_target_subset(
        "sceneA", ["owner:1", "owner:2", "owner:3", "owner:4"], limit=3
    )
    assert selected == ("owner:1", "owner:3", "owner:4")
    assert excluded == ("owner:2",)

    first = _request(frame_id=10, owner=1, visible=20)
    duplicate = replace(first)
    second = _request(frame_id=20, owner=1, visible=40)
    third = _request(frame_id=30, owner=1, visible=40)
    fourth = _request(frame_id=40, owner=1, visible=10)
    frames = (
        _frame(10, (first,)),
        _frame(10, (duplicate,)),
        _frame(20, (second,)),
        _frame(30, (third,)),
        _frame(40, (fourth,)),
    )
    reconciliation = {
        request.request_id: "owner:1" for request in (first, second, third, fourth)
    }

    manifest = build_target_view_manifest(
        "sceneA", ["owner:1"], frames, reconciliation, max_views=3
    )

    assert tuple(view.frame_id for view in manifest.views["owner:1"]) == (20, 30, 10)
    assert manifest.duplicate_request_ids == (first.request_id,)


def test_target_manifest_rejects_unproven_lineage_and_positional_assignment() -> None:
    request = _request(frame_id=10, owner=1, visible=20)
    frame = _frame(10, (request,))

    manifest = build_target_view_manifest(
        "sceneA", ["owner:1", "owner:2"], (frame,), {request.request_id: "owner:2"}
    )

    assert manifest.views["owner:1"] == ()
    assert manifest.views["owner:2"] == ()
    assert manifest.unavailable[0].reason == "UNPROVEN_TARGET_LINEAGE"


def test_native_crops_preserve_exclusive_upper_bounds_and_nine_identities() -> None:
    rgb = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    target = np.zeros((16, 16), dtype=bool)
    target[3:11, 3:11] = True
    union = target.copy()
    union[2, 2] = True

    crops = native_crops(rgb, target, union, (2, 2, 12, 12))

    assert crops.geometries == ((2, 2, 12, 12), (1, 1, 13, 13), (0, 0, 14, 14))
    assert tuple(image.shape[:2] for image in crops.raw) == ((10, 10), (12, 12), (14, 14))
    assert len(crops.raw) == len(crops.foreground) == len(crops.background) == 3
    assert crops.identities == (
        "raw:scale0",
        "foreground:scale0",
        "raw:scale1",
        "foreground:scale1",
        "raw:scale2",
        "foreground:scale2",
        "background:scale0",
        "background:scale1",
        "background:scale2",
    )
    assert np.count_nonzero(crops.foreground[0][~union[2:12, 2:12]]) == 0
    assert np.count_nonzero(crops.background[0][target[2:12, 2:12]]) == 0


def test_native_encoder_returns_unit_nine_vectors_and_legacy_first_six_mean() -> None:
    class Backend:
        def encode_images(self, images):
            return np.array(
                [[float(image.shape[0]), float(image.shape[1]), index + 1.0]
                 for index, image in enumerate(images)],
                dtype=np.float64,
            )

    rgb = np.full((16, 16, 3), 127, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:12, 2:12] = True
    crops = native_crops(rgb, mask, mask, (2, 2, 12, 12))

    encoded = NativeRegionEncoder(Backend()).encode(crops)
    raw = Backend().encode_images(crops.legacy_six)
    legacy = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).mean(axis=0)

    np.testing.assert_allclose(np.linalg.norm(encoded.vectors, axis=1), 1.0)
    np.testing.assert_allclose(encoded.legacy_mean, legacy, atol=1e-12)
    assert encoded.vectors.shape == (9, 3)


@pytest.mark.parametrize("shape", [(1, 2, 3), (3, 5, 3), (8, 9, 3)])
def test_siglip_backend_preserves_rgb_hwc_for_narrow_crops(shape) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    processor = transformers.SiglipImageProcessor(size={"height": 8, "width": 8})

    class Model(torch.nn.Module):
        def get_image_features(self, pixel_values):
            return pixel_values.mean(dim=(-1, -2))

    rgb = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    expected = processor(images=[rgb], input_data_format="channels_last", return_tensors="pt")
    backend = FrozenSiglipBackend(model=Model(), processor=processor, tokenizer=None, device="cpu")

    np.testing.assert_array_equal(
        backend.encode_images([rgb]), expected["pixel_values"].mean(dim=(-1, -2)).numpy()
    )


class _WowRunner:
    def __init__(self, mask16: np.ndarray, *, consumed: bool = True) -> None:
        self.mask16 = mask16
        self.consumed = consumed
        self.calls = []

    def prepare_region(self, image, region_mask, *, scale, image_size):
        assert scale == 2.5
        assert image_size == 448
        return image.copy(), self.mask16.copy()

    def generate(self, *, images, region_mask_16, prompt, generation):
        self.calls.append((images, region_mask_16.copy(), prompt, dict(generation)))
        return {
            "raw_generation": "Chair",
            "trace": {
                "boundary": "visual_token_mask_multiply",
                "mask_consumed": self.consumed,
                "mask_shape": list(region_mask_16.shape),
                "mask_support": int(np.count_nonzero(region_mask_16)),
            },
        }


def test_wow_adapter_requires_nonempty_16x16_mask_and_consumption_trace() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    region = np.ones((20, 20), dtype=bool)

    empty = WowRegionAdapter(_WowRunner(np.zeros((16, 16), dtype=bool)))
    unavailable = empty.classify(image, region)
    assert unavailable.status == "UNAVAILABLE_EMPTY_WOW_MASK"

    runner = _WowRunner(np.eye(16, dtype=bool))
    result = WowRegionAdapter(runner).classify(image, region)
    assert result.status == "COMPLETE"
    assert result.raw_generation == "Chair"
    assert runner.calls[0][2] == WOW_PROMPT
    assert runner.calls[0][3] == {
        "do_sample": False,
        "max_dynamic_patches": 12,
        "max_new_tokens": 32,
        "num_beams": 1,
        "thumbnail": True,
    }

    with pytest.raises(RuntimeError, match="did not consume"):
        WowRegionAdapter(_WowRunner(np.eye(16, dtype=bool), consumed=False)).classify(
            image, region
        )


def test_official_wow_runner_uses_combined_tensor_and_region_tile_index() -> None:
    import torch
    from PIL import Image

    class Model:
        dtype = torch.float32

        def eval(self):
            return self

        def parameters(self):
            return iter((torch.nn.Parameter(torch.zeros(1)),))

        def generate(self, *, pixel_masks, vaild_region_idx, **_kwargs):
            mask_token_idx = torch.ones((3, 256), dtype=torch.bool)
            mask_token_idx[vaild_region_idx] = pixel_masks.reshape(-1, 256)
            vit_embeds_with_mask_token = mask_token_idx.reshape(-1)
            return vit_embeds_with_mask_token

        def chat(
            self,
            *,
            tokenizer,
            pixel_values,
            question,
            generation_config,
            vaild_region_idx,
            pixel_masks,
        ):
            assert tokenizer is not None
            assert question == WOW_PROMPT
            assert generation_config["max_new_tokens"] == 32
            assert pixel_values.shape == (3, 3, 448, 448)
            assert int(vaild_region_idx) == 2
            self.generate(
                pixel_masks=pixel_masks,
                vaild_region_idx=vaild_region_idx,
            )
            return "chair"

    def dynamic(image, **kwargs):
        assert kwargs == {
            "min_num": 1,
            "max_num": 12,
            "image_size": 448,
            "use_thumbnail": True,
        }
        resized = image.resize((32, 16))
        return [resized, resized], resized

    def transform(_image):
        return torch.zeros((3, 448, 448), dtype=torch.float32)

    runner = OfficialWowRunner(
        model=Model(),
        tokenizer=object(),
        transform=transform,
        dynamic_preprocess=dynamic,
        device="cpu",
    )
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    mask = np.zeros((16, 32), dtype=bool)
    mask[4:12, 10:22] = True
    crop, mask16 = runner.prepare_region(image, mask, scale=2.5, image_size=448)

    assert isinstance(Image.fromarray(crop), Image.Image)
    assert mask16.shape == (16, 16)
    response = runner.generate(
        images=(image, crop),
        region_mask_16=mask16,
        prompt=WOW_PROMPT,
        generation={
            "do_sample": False,
            "max_dynamic_patches": 12,
            "max_new_tokens": 32,
            "num_beams": 1,
            "thumbnail": True,
        },
    )

    assert response["raw_generation"] == "chair"
    assert response["trace"]["mask_consumed"] is True
    assert response["trace"]["region_tile_index"] == 2


def test_generated_name_mapping_uses_exact_then_embedding_and_original_ties() -> None:
    assert clean_wow_category_response("Category: Office chair. More detail\nignored") == "Office chair"
    classes = ("office chair", "table", "wall")
    exact = map_generated_name(
        "  Office-chair ", classes, clean_response=lambda value: value
    )
    assert exact.class_index == 0
    assert exact.method == "EXACT"

    vectors = {
        "seat": np.array([1.0, 0.0]),
        "office chair": np.array([1.0, 0.0]),
        "table": np.array([1.0, 0.0]),
        "wall": np.array([0.0, 1.0]),
    }
    embedded = map_generated_name(
        "seat",
        classes,
        clean_response=lambda value: value,
        embed_text=lambda values: np.stack([vectors[value] for value in values]),
    )
    assert embedded.class_index == 0
    assert embedded.method == "MINILM_COSINE"
    assert embedded.similarities == (1.0, 1.0, 0.0)
    assert embedded.top1_top2_gap == 0.0
