from src.static_ovmap.m2_reviewer_study.costs import operation_key
from src.static_ovmap.m2_reviewer_study.selection import select_simple


def test_exact_operation_union_ignores_metadata_not_pixels_or_model():
    request = {"image_sha256": "a", "target_mask_sha256": "b", "native_union_mask_sha256": "c", "bbox_xyxy": [0, 0, 3, 4], "crop_convention": "six", "request_id": "1"}
    assert operation_key("model1", request) == operation_key("model1", {**request, "request_id": "2"})
    assert operation_key("model1", request) != operation_key("model2", request)
    assert operation_key("model1", request) != operation_key("model1", {**request, "bbox_xyxy": [0, 0, 2, 4]})


def test_nomination_uses_cal_priority_and_intervention_not_replica():
    rows = [{"method": "a", "uap": .1, "miou": .2, "unique_operations": 20, "changed_owners": 1},
            {"method": "b", "uap": .1, "miou": .2, "unique_operations": 10, "changed_owners": 1},
            {"method": "c", "uap": .9, "miou": .9, "unique_operations": 0, "changed_owners": 0}]
    assert select_simple(rows, ["a", "b", "c"])["selected"] == "b"
    assert select_simple([{**r, "changed_owners": 0} for r in rows], ["a", "b", "c"])["selected"] == "RV_A5_T001"
