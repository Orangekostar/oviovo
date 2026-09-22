"""Full geometry driver uses stored complete pools and unchanged evaluation."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.static_ovmap.module_validation import geometry_pipeline as pipeline
from src.static_ovmap.module_validation.boundary_jobs import file_identity
from src.static_ovmap.module_validation.evaluation import (
    GeometryIdentity,
    PredictionPayload,
)
from src.static_ovmap.module_validation.native_capture import _array_digest
from src.static_ovmap.module_validation.scannet_ground_truth import canonical_metrics
from src.static_ovmap.module_validation.scannet_study import save_prediction

UPSTREAM = Path(os.environ.get("OVIMAP_NATIVE_UPSTREAM", "/home/ww/crove/ovimap-module-validation-upstream"))


@pytest.mark.skipif(not (UPSTREAM / "scripts/eval_sem_seg.py").is_file(), reason="pinned native checkout required")
def test_direct_geometry_scene_driver_keeps_rank_bridge_and_separate_targets(tmp_path, monkeypatch):
    scene = "sceneA"
    output = tmp_path / "scenes" / scene
    owners = np.repeat([1, 2], 120)
    xyz = np.column_stack((np.arange(240) * .01, np.zeros(240), np.ones(240))).astype(np.float32)
    faces = np.array([[index, index + 1, index + 2] for index in range(238)])
    native = PredictionPayload("N0", "N0", scene, GeometryIdentity(_array_digest(xyz), _array_digest(faces),
        "c" * 64, "projection", 240), owners, np.ones(240, np.int64), ((1, 1.), (2, 1.)), {"attempts": 2})
    native.lock()
    native_path = save_prediction(native, output / "baseline/N0")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, owners * 1000)
    text_path = tmp_path / "text.npz"
    np.savez(text_path, text_embeddings=np.eye(2), valid_ids=[1, 2])
    targets = {"nearest": np.arange(240), "matched": np.ones(240, bool), "gt_semantic": owners,
        "gt_instance": owners, "gt_instance_path": gt_path, "valid_ids": np.array([1, 2]),
        "canonical_metrics": canonical_metrics}
    data = {"native": native, "scene_id": scene, "output": output, "native_manifest_path": native_path,
        "capture_path": config_path, "capture": {"frames": [], "alias_table": []}, "surface": {
            "surface_xyz": xyz, "surface_faces": faces, "segment_labels": owners,
            "surface_normals": np.tile([0., 0., 1.], (240, 1)), "normal_valid": np.ones(240, bool)}}
    monkeypatch.setattr(pipeline, "roles", lambda runtime: ({"fit": (scene,), "cal": (), "select": ()}, config_path))
    monkeypatch.setattr(pipeline, "bind_scene", lambda *args: (data, targets))

    def static_requests(data, owner_ids, ancestry, destination):
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "manifest.json"
        path.write_text(json.dumps({"identity": "same-static-input", "views": {}, "requests": {}}))
        (destination / "receipt.json").write_text(json.dumps({"status": "COMPLETE", "input_identity": "requests",
            "outputs": [file_identity(path)], "inputs": []}))
        return path

    monkeypatch.setattr(pipeline, "prepare_static_manifest", static_requests)
    for method in ("G_ORIGINAL", "G_AGREEMENT"):
        inference = output / "geometry/direct" / method / "inference"
        inference.mkdir(parents=True)
        index = inference / "request_index.json"
        index.write_text(json.dumps({"requests": {}}))
        (inference / "receipt.json").write_text(json.dumps({"status": "COMPLETE", "input_identity": method,
            "outputs": [file_identity(index)], "inputs": [], "physical_attempts_this_invocation": 0}))
    config = {"study_root": str(tmp_path), "runtime_config": str(config_path), "native_text_cache": str(text_path)}
    result = pipeline.prepare_geometry_scene(scene, "fit", {"upstream": str(UPSTREAM)}, config, config_path, run_models=False)
    assert len(result["rows"]) == 3
    assert all(row["metrics"]["miou"] == .25 for row in result["rows"])
    assert all(row["changed_points"] == 0 for row in result["rows"])
    manifest = json.loads((output / "geometry/direct/G_ORIGINAL/prediction/manifest.json").read_text())
    assert manifest["instance_ranks"] == [[1, 120.], [2, 120.]]
    gt_targets = json.loads((output / "geometry/targets/targets.json").read_text())
    values = next(iter(gt_targets["targets"].values()))
    assert [row["quality"] for row in values] == [1., 0.]
    assert pipeline.prepare_geometry_scene(scene, "fit", {"upstream": str(UPSTREAM)}, config, config_path, run_models=False) == result
