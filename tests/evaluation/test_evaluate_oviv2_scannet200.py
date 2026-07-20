from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement
import pytest

from scripts.evaluation.evaluate_oviv2_scannet200 import (
    evaluate_scannet_mesh,
    load_scannet_ground_truth,
)
from src.evaluation.contracts import GroundTruthSnapshot
from src.oviv2.meshing import LabeledMesh


def _mesh() -> LabeledMesh:
    return LabeledMesh(
        vertices_xyz=np.asarray(
            [
                [0.00, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [1.00, 0.0, 0.0],
                [1.02, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        triangles=np.empty((0, 3), dtype=np.int64),
        colors_rgb=np.zeros((4, 3), dtype=np.float32),
        semantic_ids=np.asarray([1, 1, 2, 2], dtype=np.int64),
        entity_ids=np.asarray([10, 10, 10, 10], dtype=np.int64),
        semantic_confidence=np.asarray([0.9, 0.9, 0.8, 0.8], dtype=np.float32),
        ownership_confidence=np.asarray([0.7, 0.7, 0.7, 0.7], dtype=np.float32),
    )


def test_evaluate_scannet_mesh_aligns_semantics_without_splitting_instances() -> None:
    alignment = np.eye(4)
    alignment[0, 3] = 2.0
    aligned_points = _mesh().vertices_xyz.copy()
    aligned_points[:, 0] += 2.0
    ground_truth = GroundTruthSnapshot(
        scene_id="scene0011_00",
        timestamp=1.0,
        points_xyz=aligned_points,
        semantic_labels=np.asarray(["chair", "chair", "table", "table"], dtype=object),
        instance_ids=np.asarray([100, 100, 200, 200], dtype=np.int64),
    )

    metrics = evaluate_scannet_mesh(
        scene_id="scene0011_00",
        mesh=_mesh(),
        axis_alignment=alignment,
        ground_truth=ground_truth,
        classes=("chair", "table"),
        entity_scores={10: 0.7},
        distance_threshold_m=0.05,
        min_instance_points=1,
    )

    assert metrics["semantic"]["miou"] == pytest.approx(1.0)
    assert metrics["semantic"]["macc"] == pytest.approx(1.0)
    assert metrics["instance"]["predicted_instance_count"] == 1
    assert metrics["geometry"]["f5"] == pytest.approx(1.0)
    assert metrics["protocol"]["coordinate_frame"] == "official axis-aligned world"


def test_load_scannet_ground_truth_maps_noncontiguous_official_ids(tmp_path) -> None:
    vertices = np.zeros(
        3,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("label", "i4"),
            ("instance_id", "i4"),
        ],
    )
    vertices["x"] = [0.0, 1.0, 2.0]
    vertices["label"] = [5, 7, 999]
    vertices["instance_id"] = [100, 200, -1]
    path = tmp_path / "gt.ply"
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)

    ground_truth = load_scannet_ground_truth(
        path,
        scene_id="scene0011_00",
        class_ids=(5, 7),
        classes=("chair", "table"),
    )

    assert ground_truth.semantic_labels.tolist() == [
        "chair",
        "table",
        "__ignored__",
    ]
    assert ground_truth.instance_ids.tolist() == [100, 200, -1]
