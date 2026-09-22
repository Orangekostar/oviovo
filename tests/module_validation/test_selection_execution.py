"""Selection finalization cannot skip its conditional evidence obligations."""

import json
from copy import deepcopy

import pytest

from src.static_ovmap.module_validation.selection_execution import finalize_decision


def _decision():
    return {"status": "STANDALONE_MODULES_SELECTED", "SELECT_scenes": ["s1", "s2"],
        "semantic": {"selected_method": "N0", "frozen_teacher_id": "S_SIGLIP2_AREA"},
        "geometry": {"selected_method": "G_ORIGINAL"},
        "query": {"selected_method": "Q_COMBINE", "locked_comparator": "Q_COMBINE"},
        "independent_final": {"selected_method": "N0", "science_status": "NO_NET_GAIN", "eligible_methods": []},
        "combination_plan": {"required": [], "blocked": {}, "not_required": {"COMBO_GS": "NO_STATIC_GAIN"}},
        "budget_curves": {"status": "NOT_REQUIRED_QUERY_GAIN_NOT_ELIGIBLE", "comparator_id": "Q_COMBINE"},
        "bootstrap": {}}


def test_no_gain_is_complete_without_opening_confirmation():
    result = finalize_decision(_decision(), {}, {}, combinations=[], curves=None)
    assert result["final_candidate"] == "N0"
    assert result["science_status"] == "NO_NET_GAIN"
    assert result["confirmation"]["status"] == "NOT_REQUIRED_NO_RETAINED_CANDIDATE"
    assert not result["confirmation_authorized"]


def test_pending_combinations_and_curves_cannot_be_silently_skipped():
    decision = _decision()
    decision["combination_plan"]["required"] = ["COMBO_GS"]
    with pytest.raises(ValueError, match="COMBO_GS"):
        finalize_decision(decision, {}, {}, combinations=[], curves=None)
    decision = _decision()
    decision["budget_curves"]["status"] = "REQUIRED"
    with pytest.raises(ValueError, match="budget"):
        finalize_decision(decision, {}, {}, combinations=[], curves=None)


def test_negative_combination_preserved_and_simpler_supported_candidate_kept():
    decision = _decision()
    decision["semantic"]["selected_method"] = "S_SIGLIP2_AREA"
    decision["geometry"]["selected_method"] = "G_AGREEMENT"
    decision["combination_plan"]["required"] = ["COMBO_GS"]

    def rows(method, ap):
        return [{"method_id": method, "scene_id": scene, "metrics": {"uap": ap, "miou": .5},
                 "logical_cost": {}, "added_seconds": 1.} for scene in ("s1", "s2")]

    candidates = {"S_SIGLIP2_AREA": rows("S_SIGLIP2_AREA", .6)}
    baselines = {"S_SIGLIP2_AREA": rows("N0", .5)}
    combo = {"method_id": "COMBO_GS", "status": "MEASURED", "rows": rows("COMBO_GS", .4),
             "matched_rows": rows("G_ORIGINAL", .5), "evidence_path": "combo_receipt.json"}
    result = finalize_decision(decision, candidates, baselines, combinations=[combo], curves=None)
    assert result["final_candidate"] == "S_SIGLIP2_AREA"
    assert result["combinations"][0]["status"] == "MEASURED"
    assert result["confirmation"]["rows"] == ("N0", "S_SIGLIP2_AREA", "S_NATIVE_AREA")
    assert result["confirmation_authorized"]
    wrong = deepcopy(combo)
    wrong["status"] = "NOT_IMPLEMENTED"
    with pytest.raises(ValueError, match="disposition"):
        finalize_decision(decision, candidates, baselines, combinations=[wrong], curves=None)


def test_not_required_budget_curve_still_has_a_verifiable_gate_receipt(tmp_path, monkeypatch):
    from src.static_ovmap.module_validation import selection_execution as execution
    from src.static_ovmap.module_validation.boundary_jobs import file_identity
    from src.static_ovmap.module_validation.study_execution import verify_receipt

    for name, extra in (("selection/module_receipt.json", {"decision": _decision()}),
                        ("query/calibration_receipt.json", {})):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = path.with_suffix(".txt")
        data.write_text("recorded evidence")
        path.write_text(json.dumps({"status": "COMPLETE", "input_identity": name,
                                   "outputs": [file_identity(data)], **extra}))
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    config = {"study_root": str(tmp_path), "runtime_config": str(config_path)}
    monkeypatch.setattr(execution, "roles", lambda runtime: ({"select": ("s1", "s2")}, config_path))
    result = execution.run_budget_curves({}, config, config_path)
    assert result["rows"] == []
    assert verify_receipt(tmp_path / "selection/budget_curve_receipt.json") == result
