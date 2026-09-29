def test_raw_crops_share_across_masks_but_not_models_and_mapping_is_required():
    from src.static_ovmap.a7_evidence_upgrade.operations import (
        crop_operations,
        method_operations,
    )

    request = {"bbox_xyxy": [20, 20, 80, 80], "image_sha256": "rgb"}
    first = crop_operations("s2", request, (100, 100), "union")
    changed = crop_operations("s2", request, (100, 100), "global")
    other_model = crop_operations("so400m", request, (100, 100), "union")
    assert len(first) == len(changed) == 6
    assert len(set(first) & set(changed)) == 3
    assert not set(first) & set(other_model)
    sources = {"N0": first, "Q_GAIN": first, "S_SIGLIP2_AREA": other_model}
    assert len(method_operations(sources, "RV_A7_COS_REFIT")) == 12
    assert len(method_operations(sources, "S_SIGLIP2_AREA")) == 12
    assert len(method_operations(sources, "N0")) == 6
