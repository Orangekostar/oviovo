"""The original exporter retains its object-zero and raw-category semantics."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from src.static_ovmap.module_validation import scannet_ground_truth as gt

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
