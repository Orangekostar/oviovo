"""The original exporter retains its object-zero and raw-category semantics."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from src.static_ovmap.module_validation import scannet_ground_truth as gt
from src.static_ovmap.module_validation import study_scene

UPSTREAM = Path(os.environ.get("OVIMAP_NATIVE_UPSTREAM", "/home/ww/crove/ovimap-module-validation-upstream"))


@pytest.mark.skipif(not (UPSTREAM / "scripts/eval_sem_seg.py").is_file(), reason="pinned native checkout required")
def test_released_annotation_converter_preserves_object_zero_and_full_rows(tmp_path):
    scene = "scene0000_00"
    raw = tmp_path / "raw" / scene
    raw.mkdir(parents=True)
    vertices = np.zeros(6, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("label", "u2")])
    vertices["x"] = np.arange(6)
    vertices["label"] = 39  # Existing NYU40 values must not become ScanNet200 IDs.
    PlyData([PlyElement.describe(vertices, "vertex")]).write(raw / f"{scene}_vh_clean_2.labels.ply")
    (raw / f"{scene}_vh_clean_2.0.010000.segs.json").write_text(json.dumps({"segIndices": [4, 4, 6, 6, 8, 8]}))
    (raw / f"{scene}.aggregation.json").write_text(json.dumps({"segGroups": [
        {"objectId": 0, "label": "wall", "segments": [4]},
        {"objectId": 1, "label": "chair", "segments": [6]},
        {"objectId": 2, "label": "wall", "segments": [8]}]}))
    table = tmp_path / "labels.tsv"
    table.write_text("raw_category\tid\tnyu40id\nwall\t1\t1\nchair\t2\t5\n")
    receipt = gt.prepare_ground_truth(UPSTREAM, raw, scene, table, tmp_path / "converted")
    values = gt.load_ground_truth(receipt)
    np.testing.assert_array_equal(values["gt_semantic"], [1, 1, 2, 2, 1, 1])
    np.testing.assert_array_equal(values["gt_instance"], [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(np.load(values["gt_instance_path"]), [0, 0, 2000, 2000, 1000, 1000])
    np.testing.assert_array_equal(values["xyz"][:, 0], np.arange(6))
    assert gt.prepare_ground_truth(UPSTREAM, raw, scene, table, tmp_path / "converted") == receipt


@pytest.mark.skipif(not (UPSTREAM / "scripts/eval_sem_seg.py").is_file(), reason="pinned native checkout required")
def test_scannet200_released_evaluation_and_native_export_keep_trace_parity(tmp_path):
    from src.static_ovmap.module_validation.evaluation import (
        GeometryIdentity,
        PredictionPayload,
    )
    from src.static_ovmap.module_validation.scannet_ground_truth import (
        canonical_metrics,
    )
    from src.static_ovmap.module_validation.scannet_study import relabel_prediction

    count = 240
    vertices = np.zeros(count, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("label", "u2")])
    owners = np.repeat([1, 2], 120)
    vertices["label"] = owners
    PlyData([PlyElement.describe(vertices, "vertex")]).write(tmp_path / "gt_instance_mesh.ply")
    np.save(tmp_path / "gt_sem_inst_id.npy", owners * 1000)
    target = {"nearest": np.arange(count), "matched": np.ones(count, bool), "gt_semantic": owners,
        "gt_instance": owners, "gt_instance_path": tmp_path / "gt_sem_inst_id.npy",
        "valid_ids": np.array([1, 2]), "canonical_metrics": canonical_metrics}
    native = PredictionPayload("N0", "N0", "sceneA", GeometryIdentity("a" * 64, "b" * 64, "c" * 64, "projection", count),
        owners, np.ones(count, np.int64), ((1, 1.0), (2, 1.0)), {"attempts": 2})
    native.lock()
    assert study_scene.audit_native_export(native, target, UPSTREAM, tmp_path / "parity")["exact_mask_and_serialized_rank_parity"]
    corrected = relabel_prediction(native, "S_NATIVE_AREA", "S", {2: 2}, {"attempts": 3})
    result = study_scene.evaluate_predictions([native, corrected], {"sceneA": target}, UPSTREAM, tmp_path / "evaluated")
    assert all(row["metrics"]["trace_parity"] for row in result)
    assert result[0]["metrics"]["miou"] == 0.25
    assert result[1]["metrics"]["miou"] == 1.0
    assert result[1]["metrics"]["uap"] == 1.0
    assert result[0]["metrics"]["canonical_ap50"] == result[1]["metrics"]["canonical_ap50"] == 1.0
