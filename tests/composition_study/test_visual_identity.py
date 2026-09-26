"""Cache equivalence requires the complete model and actual crop inputs."""

import pytest


def test_visual_cache_key_separates_models_masks_rgb_and_bbox():
    from src.static_ovmap.composition_study.visual_requests import operation_identity

    request = {
        "request_id": "r",
        "scene_id": "scene-a",
        "frame_id": 1,
        "image_sha256": "a" * 64,
        "target_mask_sha256": "b" * 64,
        "native_union_mask_sha256": "c" * 64,
        "bbox_xyxy": [0, 0, 5, 7],
        "crop_convention": "native_global_bbox_union_exclusive_upper_v1",
    }
    original = operation_identity("native-model-and-processor", request)
    assert operation_identity("siglip2-model-and-processor", request) != original
    for field, value in (
        ("image_sha256", "d" * 64),
        ("target_mask_sha256", "d" * 64),
        ("native_union_mask_sha256", "d" * 64),
        ("bbox_xyxy", [0, 0, 6, 7]),
    ):
        assert (
            operation_identity("native-model-and-processor", {**request, field: value})
            != original
        )
    with pytest.raises(ValueError, match="identity"):
        operation_identity(
            "native-model-and-processor",
            {k: v for k, v in request.items() if k != "image_sha256"},
        )
