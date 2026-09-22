"""Receipt selection respects frozen teachers, controls, and missing heads."""

from copy import deepcopy

import pytest

from src.static_ovmap.module_validation.selection_pipeline import standalone_selection


def _fixtures():
    def rows(method, ap, *, changes=0):
        return [{"method_id": method, "scene_id": scene, "prediction_key": method + scene,
            "metrics": {"uap": ap, "miou": .6, "canonical_ap50": .4, "canonical_ap75": .3},
            "effective_changes": changes, "logical_cost": {"attempts": 10}, "added_seconds": float(changes)}
            for scene in ("s1", "s2")]

    semantic = {"teacher_id": "S_SIGLIP2_AREA", "learned_status": "BLOCKED_EVENT_SUPPORT",
        "rows": rows("N0", .5) + rows("S_SIGLIP2_AREA", .52, changes=2) + rows("S_WOW_VOTE", .9, changes=2)
        + rows("S_NATIVE_AREA", .5) + rows("S_NATIVE_VOTE", .5) + rows("S_SIGLIP2_VOTE", .51)}
    geometry = {"learned_status": "BLOCKED_GEOMETRY_TARGET_SUPPORT", "rows": rows("N0", .5)
                + rows("G_ORIGINAL", .5) + rows("G_AGREEMENT", .5)}
    controls = [{"role": "cal", "scene_id": scene, "method_id": method, "budget": 200,
        "row": {"method_id": method, "scene_id": scene, "metrics": {"uap": .4, "miou": .5}},
        "logical_cost": {"attempts": 10, "crop_inputs": 60}}
        for scene in ("c1", "c2") for method in ("Q_COMBINE", "Q_AREA", "Q_UNCERTAINTY")]
    query_cal = {"control_rows": controls, "comparator_id": "Q_COMBINE"}
    query = {"learned_status": "BLOCKED_QUERY_TARGET_SUPPORT", "comparator_id": "Q_COMBINE", "results": [
        {"row": row, "logical_cost": {"attempts": 10}, "budget": 200}
        for method in ("Q_COMBINE", "Q_AREA", "Q_UNCERTAINTY") for row in rows(method, .45)]}
    return semantic, geometry, query_cal, query


def test_module_selection_does_not_reselect_teacher_or_fabricate_blocked_heads():
    decision = standalone_selection(*_fixtures(), select_scenes=("s1", "s2"), cal_scenes=("c1", "c2"))
    assert decision["semantic"]["selected_method"] == "S_SIGLIP2_AREA"
    assert decision["geometry"]["selected_method"] == "G_ORIGINAL"
    assert decision["query"]["selected_method"] == "Q_COMBINE"
    assert decision["independent_final"]["selected_method"] == "S_SIGLIP2_AREA"
    assert decision["combination_plan"]["required"] == ()
    assert decision["budget_curves"]["status"] == "NOT_REQUIRED_QUERY_GAIN_NOT_ELIGIBLE"
    assert decision["unavailable_methods"]["S_PAIRED"] == "BLOCKED_EVENT_SUPPORT"
    assert decision["bootstrap"]["S_SIGLIP2_AREA"]["resamples"] == 2000


def test_selection_rejects_wrong_select_scene_or_inconsistent_native_bridge():
    values = list(_fixtures())
    wrong = deepcopy(values)
    wrong[0]["rows"][0]["scene_id"] = "c1"
    with pytest.raises(ValueError, match="SELECT"):
        standalone_selection(*wrong, select_scenes=("s1", "s2"), cal_scenes=("c1", "c2"))
    wrong = deepcopy(values)
    wrong[1]["rows"][0]["metrics"]["uap"] = .8
    with pytest.raises(ValueError, match="N0"):
        standalone_selection(*wrong, select_scenes=("s1", "s2"), cal_scenes=("c1", "c2"))
