"""Scientific driver gates remain independent of SELECT outcomes."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.static_ovmap.module_validation.evaluation import (
    GeometryIdentity,
    PredictionPayload,
)
from src.static_ovmap.module_validation.semantic_pipeline import (
    calibrated_teacher,
    selector_prediction,
)

UPSTREAM = Path(os.environ.get("OVIMAP_NATIVE_UPSTREAM", "/home/ww/crove/ovimap-module-validation-upstream"))


def test_teacher_choice_requires_exact_cal_scenes_and_actual_alternative_success():
    rows = []
    for scene in ("cal1", "cal2"):
        rows.append({"scene_id": scene, "role": "cal", "rows": [
            {"method_id": method, "metrics": {"uap": score, "miou": .2},
             "successful_requests": successful, "added_seconds": cost}
            for method, score, successful, cost in (
                ("S_SIGLIP2_AREA", .1, 3, 4.), ("S_SIGLIP2_VOTE", .1, 3, 4.),
                ("S_WOW_VOTE", .9, 0, 10.))]})
    choice = calibrated_teacher(rows, ("cal1", "cal2"))
    assert choice["teacher_id"] == "S_SIGLIP2_AREA"
    assert "S_WOW_VOTE" not in [row["method"] for row in choice["candidates"]]
    rows[0]["role"] = "select"
    with pytest.raises(ValueError, match="CAL"):
        calibrated_teacher(rows, ("cal1", "cal2"))


def test_selector_keeps_threshold_ties_fallback_and_masks_ranks_unchanged():
    native = PredictionPayload("N0", "N0", "sceneA", GeometryIdentity("a" * 64, "b" * 64,
        "c" * 64, "projection", 5), np.array([7, 7, 8, 9, 0]), np.array([5, 5, 5, 5, 0]),
        ((7, .5), (8, .2), (9, .1)), {"attempts": 5})
    native.lock()
    suggestions = {7: {"label_id": 42, "technical_fallback": False},
                   8: {"label_id": 42, "technical_fallback": False},
                   9: {"label_id": 42, "technical_fallback": True}}
    payload, decisions = selector_prediction(native, "S_PAIRED", suggestions, {7: .2, 8: .3, 9: .9}, .2, {})
    np.testing.assert_array_equal(payload.semantic_labels, [5, 5, 42, 5, 0])
    np.testing.assert_array_equal(payload.owner_ids, native.owner_ids)
    assert payload.instance_ranks == native.instance_ranks
    assert payload.locked
    assert [row["accepted"] for row in decisions] == [False, True, False]
    keep, _ = selector_prediction(native, "S_PAIRED", suggestions, {7: .2, 8: .3}, "KEEP_ALL", {})
    assert keep.prediction_key == native.prediction_key


def test_successful_teacher_with_undefined_cal_metric_makes_comparison_inconclusive():
    rows = [{"scene_id": scene, "role": "cal", "rows": [
        {"method_id": method, "successful_requests": 2, "added_seconds": 1.,
         "metrics": {"uap": None if method == "S_WOW_VOTE" and scene == "c1" else .2, "miou": .3}}
        for method in ("S_SIGLIP2_AREA", "S_SIGLIP2_VOTE", "S_WOW_VOTE")]}
        for scene in ("c1", "c2")]
    result = calibrated_teacher(rows, ("c1", "c2"))
    assert result["status"] == "INCONCLUSIVE_UNDEFINED_CAL_METRIC"
    assert result["teacher_id"] is None
    assert result["undefined_candidates"] == ["S_WOW_VOTE"]


@pytest.mark.skipif(not (UPSTREAM / "scripts/eval_sem_seg.py").is_file(), reason="pinned native checkout required")
def test_direct_scene_driver_locks_five_rows_then_uses_real_released_evaluator(tmp_path, monkeypatch):
    from src.static_ovmap.module_validation import semantic_pipeline as pipeline
    from src.static_ovmap.module_validation.boundary_jobs import file_identity
    from src.static_ovmap.module_validation.scannet_ground_truth import (
        canonical_metrics,
    )
    from src.static_ovmap.module_validation.scannet_study import save_prediction

    scene = "sceneA"
    output = tmp_path / "scenes" / scene
    owners = np.repeat([1, 2], 120)
    native = PredictionPayload("N0", "N0", scene, GeometryIdentity("a" * 64, "b" * 64,
        "c" * 64, "projection", 240), owners, np.ones(240, np.int64), ((1, 1.), (2, 1.)), {"attempts": 2})
    native.lock()
    native_path = save_prediction(native, output / "baseline/N0")
    (output / "baseline/receipt.json").write_text(json.dumps({"status": "COMPLETE", "input_identity": "N0",
        "outputs": [file_identity(native_path)], "inputs": []}))
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, owners * 1000)
    targets = {"nearest": np.arange(240), "matched": np.ones(240, bool), "gt_semantic": owners,
        "gt_instance": owners, "gt_instance_path": gt_path, "valid_ids": np.array([1, 2]),
        "canonical_metrics": canonical_metrics}
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    data = {"native": native, "output": output, "native_manifest_path": native_path, "capture_path": config_path}
    monkeypatch.setattr(pipeline, "study_roles", lambda runtime: ({"fit": (scene,), "cal": (), "select": ()}, config_path))
    monkeypatch.setattr(pipeline, "bind_scene", lambda *args: (data, targets))
    manifest = {"identity": "captured-request-identity"}

    def prepare(capture, reference, path):
        path.write_text(json.dumps(manifest))
        return manifest

    monkeypatch.setattr(pipeline, "prepare_semantic_manifest", prepare)
    records = {}
    for model in ("native", "siglip2", "wow"):
        model_output = output / "semantic_models" / model
        model_output.mkdir(parents=True)
        stored = model_output / "stored.json"
        stored.write_text("{}")
        (model_output / "receipt.json").write_text(json.dumps({"status": "COMPLETE", "input_identity": model,
            "inputs": [], "outputs": [file_identity(stored)]}))
        records[model] = {"r1": {"status": "COMPLETE", "elapsed_seconds": 2., "six_crop_seconds": 1.,
            "crop_inputs": 6, "background_crop_inputs": 3 if model == "native" else 0,
            "generations": 1 if model == "wow" else 0}}
    suggestions = {owner: {"label_id": owner, "technical_fallback": False, "successful_views": 1} for owner in (1, 2)}
    readouts = {method: suggestions for method in pipeline.DIRECT_IDS}
    monkeypatch.setattr(pipeline, "_records_and_readouts", lambda *args: (manifest, records, readouts, np.eye(2), np.eye(2), (1, 2)))
    config, runtime = {"study_root": str(tmp_path), "runtime_config": str(config_path)}, {"upstream": str(UPSTREAM)}
    result = pipeline.prepare_semantic_scene(scene, "fit", runtime, config, config_path, run_models=False)
    assert len(result["rows"]) == 6
    assert result["rows"][0]["metrics"]["miou"] == .25
    assert all(row["metrics"]["uap"] == 1. for row in result["rows"][1:])
    assert all(row["effective_changes"] == 1 for row in result["rows"][1:])
    events = json.loads((output / "semantic/direct_evaluation/object_events.json").read_text())
    assert events["events"]["S_SIGLIP2_AREA"][1]["event"] == "GAIN"
    assert any(row["path"].endswith("released_trace.json.gz") for row in result["outputs"])
    assert pipeline.prepare_semantic_scene(scene, "fit", runtime, config, config_path, run_models=False) == result
