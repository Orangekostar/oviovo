"""Four-scene map-first screening must use pooled GT object counts."""


def test_catastrophe_gate_requires_full_cohort_and_combines_iou_with_r50_loss():
    import pytest
    from static_ovmap.recovery_wave2.geometry import screen_cohort

    def row(scene, iou, fragments, matched):
        return {"scene": scene, "status": "COMPLETE", "per_gt": [
            {"gt_id": 1001, "best_iou": iou, "substantial_fragments": fragments}],
            "matches": {"0.5": {"matched_gt": matched, "eligible_gt": 1}}}

    scenes = ["a", "b", "c", "d"]
    base = [row(scene, .5, 2, 1) for scene in scenes]
    variant = [row(scene, .46, 2, int(i > 1)) for i, scene in enumerate(scenes)]
    result = screen_cohort(base, variant, scenes)
    assert result["status"] == "SCREENED_OUT_GEOMETRY" and result["R50_count_loss"] == 2
    assert result["semantic_status"] == "NOT_RUN_RESOURCE_SCREEN"
    variant[0]["per_gt"][0]["best_iou"] = .7
    assert screen_cohort(base, variant, scenes)["status"] == "GEOMETRY_PASS"
    for item in variant:
        item["per_gt"][0]["substantial_fragments"] = 4
    assert screen_cohort(base, variant, scenes)["fragment_rule_triggered"]
    with pytest.raises(ValueError, match="complete"):
        screen_cohort(base, variant[:-1], scenes)
