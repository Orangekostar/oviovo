"""Hybrid refinements consume paid causal incumbents and preserve negative gates."""

import json

import numpy as np
import pytest

from src.static_ovmap.module_validation import combination_pipeline as pipeline
from src.static_ovmap.module_validation.evaluation import (
    GeometryIdentity,
    PredictionPayload,
)
from src.static_ovmap.module_validation.scannet_study import (
    load_prediction,
    relabel_prediction,
    save_prediction,
)


def test_static_noninferiority_preserves_negative_and_undefined_comparisons():
    baseline = [{"scene_id": scene, "metrics": {"uap": .2, "miou": .4}} for scene in ("s1", "s2")]
    negative = [{"scene_id": scene, "metrics": {"uap": .1, "miou": .4}} for scene in ("s1", "s2")]
    assert pipeline.static_noninferiority(baseline, baseline) == "NONINFERIOR"
    assert pipeline.static_noninferiority(negative, baseline) == "NOT_REQUIRED_STATIC_NONINFERIORITY_FAILED"
    negative[0]["metrics"]["uap"] = None
    assert pipeline.static_noninferiority(negative, baseline) == "INCONCLUSIVE_UNDEFINED_STATIC_COMPARISON"


@pytest.mark.parametrize("method", ["COMBO_Q_REFINEMENT", "COMBO_Q_REFINEMENT_CONTROL"])
def test_hybrid_starts_from_paid_query_before_binding_final_map(tmp_path, monkeypatch, method):
    scene = "s1"
    native = PredictionPayload("N0", "N0", scene, GeometryIdentity("a" * 64, "b" * 64, "c" * 64, "projection", 3),
        np.array([1, 1, 0]), np.array([2, 2, 0]), ((1, .6),), {"attempts": 1000})
    native.lock()
    native_path = save_prediction(native, tmp_path / "native")
    query = relabel_prediction(native, "Q_GAIN" if method == "COMBO_Q_REFINEMENT" else "Q_AREA", "Q", {1: 0},
                               {"attempts": 200, "crop_inputs": 1200}, {})
    query_path = save_prediction(query, tmp_path / "query/prediction")
    query_receipt = query_path.parents[1] / "receipt.json"
    query_receipt.write_text("{}")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    for path in ("selection/module_receipt.json", "semantic/calibration_receipt.json", "geometry/calibration_receipt.json",
                 "query/calibration_receipt.json"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    modules = {"semantic": {"selected_method": "S_SIGLIP2_AREA"}, "geometry": {"selected_method": "G_ORIGINAL"},
               "query": {"selected_method": "Q_GAIN", "locked_comparator": "Q_AREA"},
               "combination_plan": {"required": ["COMBO_Q_REFINEMENT"]}}
    monkeypatch.setattr(pipeline, "verify_receipt", lambda path: {"decision": modules})
    monkeypatch.setattr(pipeline, "roles", lambda runtime: ({"select": ("s1", "s2")}, config_path))
    events = []

    def acquire(*args):
        assert not events
        events.append("paid_causal_readout")
        return query, query_receipt, {}

    def bind(*args):
        assert events == ["paid_causal_readout"]
        events.append("final_map")
        return {"native": native, "native_manifest_path": native_path, "output": tmp_path}, {}

    def refine(data, reference, *args):
        assert data["native"].prediction_key == query.prediction_key
        assert data["native"].semantic_labels.tolist() == [0, 0, 0]
        assert reference["native"].prediction_key == native.prediction_key
        return relabel_prediction(data["native"], "S_SIGLIP2_AREA", "S", {1: 1},
            {"attempts": 200, "crop_inputs": 1200, "added_attempts": 3, "added_crop_inputs": 18}, {}), []

    def evaluate(values, *args):
        assert values[0].locked
        return [{"method_id": method, "scene_id": scene, "prediction_key": values[0].prediction_key,
                 "metrics": {"uap": .2, "miou": .3}}]

    monkeypatch.setattr(pipeline, "_query_readout", acquire)
    monkeypatch.setattr(pipeline, "bind_scene", bind)
    monkeypatch.setattr(pipeline, "_static_refinement", refine)
    monkeypatch.setattr(pipeline, "evaluate_predictions", evaluate)
    config = {"study_root": str(tmp_path), "runtime_config": str(config_path)}
    result = pipeline.run_combination_scene(scene, "select", method, {"upstream": str(tmp_path)}, config, config_path)
    assert result["logical_cost"]["attempts"] == 200
    assert result["logical_cost"]["added_crop_inputs"] == 18
    payload = load_prediction(result["prediction_manifest"])
    np.testing.assert_array_equal(payload.owner_ids, native.owner_ids)
    assert payload.instance_ranks == native.instance_ranks
    assert "separately" in payload.metadata["cost_scope"]
    assert json.loads((tmp_path / "selection/combinations" / method / "scenes/s1/receipt.json").read_text())["status"] == "COMPLETE"
