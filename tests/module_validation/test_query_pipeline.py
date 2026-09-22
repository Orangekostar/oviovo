"""Q calibration and final export preserve held-out decisions and native masks."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.static_ovmap.module_validation.query_pipeline import (
    choose_comparator,
    reconcile_export,
)
from src.static_ovmap.module_validation.query_state import (
    AcquisitionPayload,
    FeatureStore,
)
from src.static_ovmap.module_validation.query_study import replay_captured
from tests.module_validation.test_query_gain_policy import _candidate
from tests.module_validation.test_query_lineage import _snapshot


def test_comparator_uses_all_cal_scenes_and_rejects_partial_or_undefined():
    rows = [{"scene_id": scene, "role": "cal", "method_id": method, "budget": 200,
             "row": {"metrics": {"uap": .1, "miou": .2}}, "logical_cost": {"crop_inputs": cost, "attempts": cost // 6}}
            for scene in ("c1", "c2") for method, cost in (("Q_AREA", 12), ("Q_COMBINE", 6), ("Q_UNCERTAINTY", 18))]
    assert choose_comparator(rows, ("c1", "c2")) == "Q_COMBINE"
    with pytest.raises(ValueError, match="exact CAL"):
        choose_comparator(rows[:-1], ("c1", "c2"))
    rows[0]["row"]["metrics"]["uap"] = None
    assert choose_comparator(rows, ("c1", "c2")) is None


def test_comparator_respects_metric_tolerance_and_protocol_id_order():
    rows = [{"scene_id": scene, "role": "cal", "method_id": method, "budget": 200,
        "row": {"metrics": {"uap": .1 + (5e-11 if method == "Q_AREA" else 0.), "miou": .2}},
        "logical_cost": {"attempts": 10, "crop_inputs": 30 if method == "Q_AREA" else 60}}
        for scene in ("c1", "c2") for method in ("Q_COMBINE", "Q_AREA", "Q_UNCERTAINTY")]
    assert choose_comparator(rows, ("c1", "c2")) == "Q_COMBINE"


def test_confirmation_queries_require_frozen_access_and_b200_before_opening_data(tmp_path, monkeypatch):
    from src.static_ovmap.module_validation import query_pipeline as pipeline

    monkeypatch.setattr(pipeline, "roles", lambda _: ({"confirm": ("held-out",)}, tmp_path / "split.json"))
    with pytest.raises(ValueError, match="confirmation.*lock"):
        pipeline.run_query_scene("held-out", "confirm", "Q_AREA", {}, {}, tmp_path / "config.json")
    with pytest.raises(ValueError, match="B200"):
        pipeline.run_query_scene("held-out", "confirm", "Q_AREA", {}, {}, tmp_path / "config.json",
                                 budget=100, confirmation_lock=tmp_path / "selection.json")


def test_final_registry_reconciliation_occurs_after_all_acquisition():
    from dataclasses import replace

    class Frames:
        scene_id = "scene-a"
        schedule = (0,)

        def load(self, index):
            return {"candidates": (replace(_candidate(1, "a", 10), frame_index=0),),
                    "snapshot": _snapshot({11: 1})}

    result = replay_captured(Frames(), "Q_AREA", FeatureStore(lambda _: AcquisitionPayload(np.array([1., 0.]), 6, 0.)),
                             np.eye(2), budget=1)
    decisions = result["decisions"].copy()
    reconcile_export(result, {"segment_labels": np.array([11, 11]), "original_owner": np.array([2, 2])}, np.eye(2))
    assert result["decisions"] == decisions
    assert [row.request_id for row in result["state"].object_state(2).features] == ["a"]
    assert result["state"].logical_ledger.attempts == 1


@pytest.mark.skipif(not Path(os.environ.get("OVIMAP_NATIVE_UPSTREAM", "/home/ww/crove/ovimap-module-validation-upstream"),
                            "scripts/eval_sem_seg.py").is_file(), reason="pinned native checkout required")
def test_query_scene_driver_uses_real_released_evaluation_and_zero_unobserved_label(tmp_path, monkeypatch):
    from dataclasses import replace

    from src.static_ovmap.module_validation import query_pipeline as pipeline
    from src.static_ovmap.module_validation.boundary_jobs import file_identity
    from src.static_ovmap.module_validation.evaluation import (
        GeometryIdentity,
        PredictionPayload,
    )
    from src.static_ovmap.module_validation.native_capture import _array_digest
    from src.static_ovmap.module_validation.scannet_ground_truth import (
        canonical_metrics,
    )
    from src.static_ovmap.module_validation.scannet_study import (
        load_prediction,
        save_prediction,
    )

    scene = "scene-a"
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    mapping_path = tmp_path / "runtime" / scene / "mapping_job/receipt.json"
    mapping_path.parent.mkdir(parents=True)
    mapping_path.write_text(json.dumps({"status": "COMPLETE", "input_identity": "fixture",
        "capture_manifest": str(config_path), "outputs": [file_identity(config_path)]}))
    text_path = tmp_path / "text.npz"
    np.savez(text_path, text_embeddings=np.eye(2), valid_ids=[1, 2])
    owners = np.repeat([1, 2], 120)
    xyz = np.column_stack((np.arange(240) * .01, np.zeros(240), np.ones(240))).astype(np.float32)
    native = PredictionPayload("N0", "N0", scene, GeometryIdentity(_array_digest(xyz), "b" * 64, "c" * 64, "projection", 240),
        owners, np.full(240, 2), ((1, 1.), (2, 1.)), {"attempts": 2})
    native.lock()
    native_path = save_prediction(native, tmp_path / "native")
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, owners * 1000)
    targets = {"nearest": np.arange(240), "matched": np.ones(240, bool), "gt_semantic": owners,
        "gt_instance": owners, "gt_instance_path": gt_path, "valid_ids": np.array([1, 2]), "canonical_metrics": canonical_metrics}
    data = {"native": native, "native_manifest_path": native_path,
        "surface": {"segment_labels": np.repeat([11, 22], 120), "original_owner": owners}}
    monkeypatch.setattr(pipeline, "roles", lambda _: ({"fit": (), "cal": (scene,), "select": (), "confirm": ()}, config_path))
    monkeypatch.setattr(pipeline, "bind_scene", lambda *args: (data, targets))
    calls = []

    class Frames:
        scene_id = scene
        schedule = tuple(range(200))

        def __init__(self, path):
            pass

        def load(self, index):
            calls.append(index)
            return {"candidates": (replace(_candidate(1, "a", 10), frame_index=0),) if index == 0 else (),
                    "snapshot": _snapshot({11: 1, 22: 2})}

    class Loader:
        inputs, used_receipts = [], {}
        model_loads, model_load_seconds, model_identity = 0, 0., "fixture"

        def __init__(self, *args):
            pass

        def __call__(self, candidate):
            assert calls == [0]
            return AcquisitionPayload(np.array([1., 0.]), 6, 0.)

    monkeypatch.setattr(pipeline, "CapturedFrames", Frames)
    monkeypatch.setattr(pipeline, "NativeQueryLoader", Loader)
    config = {"study_root": str(tmp_path), "runtime_config": str(config_path), "native_text_cache": str(text_path)}
    runtime = {"output_root": str(tmp_path / "runtime"), "upstream": os.environ.get("OVIMAP_NATIVE_UPSTREAM",
                                                                                  "/home/ww/crove/ovimap-module-validation-upstream")}
    result = pipeline.run_query_scene(scene, "cal", "Q_AREA", runtime, config, config_path)
    predicted = load_prediction(tmp_path / "scenes" / scene / "query/B200/Q_AREA/prediction/manifest.json")
    assert predicted.instance_ranks == native.instance_ranks
    np.testing.assert_array_equal(predicted.owner_ids, owners)
    np.testing.assert_array_equal(predicted.semantic_labels, np.repeat([1, 0], 120))
    assert result["logical_cost"]["attempts"] == 1 and result["logical_cost"]["crop_inputs"] == 6
    assert result["row"]["metrics"]["miou"] == .5
    assert pipeline.run_query_scene(scene, "cal", "Q_AREA", runtime, config, config_path) == result
