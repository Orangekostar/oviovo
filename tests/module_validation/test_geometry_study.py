"""Geometry pool serialization preserves the same complete hypothesis universe."""

import json

import numpy as np

from src.static_ovmap.module_validation.evaluation import (
    GeometryIdentity,
    PredictionPayload,
)
from src.static_ovmap.module_validation.geometry_study import (
    load_geometry_pool,
    prepare_geometry_pool,
)
from src.static_ovmap.module_validation.native_capture import _array_digest


def test_geometry_pool_roundtrip_preserves_native_original_and_complete_hypotheses(tmp_path):
    xyz = np.array([[0, 0, 1], [.01, 0, 1], [.02, 0, 1], [.03, 0, 1]], np.float32)
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    owners = np.array([7, 7, 8, 8])
    native = PredictionPayload("N0", "N0", "sceneA", GeometryIdentity(_array_digest(xyz), _array_digest(faces),
        "c" * 64, "projection", 4), owners, np.ones(4, np.int64), ((7, 1.), (8, 1.)), {})
    native.lock()
    capture_path, native_path = tmp_path / "capture.json", tmp_path / "native.json"
    capture_path.write_text("{}")
    native_path.write_text("{}")
    data = {"native": native, "scene_id": "sceneA", "capture_path": capture_path,
        "native_manifest_path": native_path, "output": tmp_path / "scene", "capture": {"frames": [], "alias_table": []},
        "surface": {"surface_xyz": xyz, "surface_faces": faces, "segment_labels": np.array([10, 10, 20, 20]),
                    "surface_normals": np.tile([0., 0., 1.], (4, 1)), "normal_valid": np.ones(4, bool)}}
    path = prepare_geometry_pool(data, tmp_path / "pool")
    pool = load_geometry_pool(path)
    assert pool["receipt"]["original_owner_parity"]
    assert len(pool["groups"]) == 1
    group = pool["groups"][0]
    hypotheses = pool["hypotheses"][group.group_id]
    assert [row.kind for row in hypotheses] == ["ORIGINAL", "MERGE"]
    assert all(row.leaf_ids == group.leaf_ids for row in hypotheses)
    np.testing.assert_array_equal(pool["leaves"].row_leaf_ids, [0, 0, 1, 1])
    assert pool["features"][hypotheses[0].hypothesis_id].values.shape == (20,)
    assert prepare_geometry_pool(data, tmp_path / "pool") == path
    receipt = json.loads(path.read_text())
    assert receipt["frame_ids"] == []
