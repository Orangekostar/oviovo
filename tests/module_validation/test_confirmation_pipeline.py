"""Confirmation reports the preselected candidate even when a control wins."""

import json

import pytest

from src.static_ovmap.module_validation.confirmation_pipeline import assess_confirmation


def test_confirmation_failure_cannot_reselect_a_better_control():
    selection = {"final_candidate": "S_PAIRED", "confirmation": {"rows": ["N0", "S_PAIRED", "S_SIMPLE"]}}
    rows = [{"method_id": method, "scene_id": scene, "metrics": {"uap": score, "miou": .5}}
            for scene in ("h1", "h2") for method, score in (("N0", .4), ("S_PAIRED", .3), ("S_SIMPLE", .8))]
    result = assess_confirmation(selection, rows, ("h1", "h2"))
    assert result["status"] == "NOT_CONFIRMED"
    assert result["final_candidate"] == "S_PAIRED"
    assert result["baseline"] == "N0"
    with pytest.raises(ValueError, match="exact"):
        assess_confirmation(selection, rows[:-1], ("h1", "h2"))


def test_confirmed_query_uses_frozen_comparator_not_n0():
    selection = {"final_candidate": "Q_GAIN", "confirmation": {"rows": ["N0", "Q_GAIN", "Q_AREA"]},
                 "module_selection": {"query": {"locked_comparator": "Q_AREA"}}}
    rows = [{"method_id": method, "scene_id": scene, "metrics": {"uap": score, "miou": .5}}
            for scene in ("h1", "h2") for method, score in (("N0", .9), ("Q_GAIN", .5), ("Q_AREA", .4))]
    result = assess_confirmation(selection, rows, ("h1", "h2"))
    assert result["status"] == "CONFIRMED"
    assert result["baseline"] == "Q_AREA"
    assert result["bootstrap"]["resamples"] == 2000
    rows[1]["metrics"]["miou"] = None
    assert assess_confirmation(selection, rows, ("h1", "h2"))["status"] == "INCONCLUSIVE_UNDEFINED_CONFIRMATION_METRIC"


def test_no_retained_candidate_never_opens_confirmation_assets(tmp_path, monkeypatch):
    from src.static_ovmap.module_validation import confirmation_pipeline as pipeline

    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"status": "FROZEN", "final_candidate": "N0"}))
    receipt = tmp_path / "selection_receipt.json"
    receipt.write_text("{}")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    monkeypatch.setattr(pipeline, "verify_receipt", lambda path: {"selection_path": str(selection)})
    monkeypatch.setattr(pipeline, "roles", lambda runtime: ({"confirm": ("h1", "h2")}, config_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("no-gain confirmation must not open any scene asset")

    monkeypatch.setattr(pipeline, "export_sensor", forbidden)
    monkeypatch.setattr(pipeline, "prepare_ground_truth", forbidden)
    result = pipeline.run_confirmation({}, {"study_root": str(tmp_path), "runtime_config": str(config_path)}, config_path, receipt)
    assert result["confirmation_status"] == "NOT_REQUIRED_NO_RETAINED_CANDIDATE"
    assert result["rows"] == []
