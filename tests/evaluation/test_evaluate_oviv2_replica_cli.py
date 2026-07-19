from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from plyfile import PlyData, PlyElement
import pytest

from scripts.evaluation.evaluate_oviv2_replica import _majority_object_per_vertex
from src.oviv2.addressing import point_to_voxel
from src.oviv2.evidence import SparseEvidenceStore
from src.oviv2.geometry import SparseTsdfVolume
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.snapshot import VoxelMapSnapshot, VoxelSnapshotMetadata

from tests.oviv2.test_geometry import _integrate_twice, _plane_frame


SCRIPT = Path("scripts/evaluation/evaluate_oviv2_replica.py")


def test_majority_object_assignment_accepts_replica_quad_faces() -> None:
    faces = np.empty(2, dtype=object)
    faces[0] = np.asarray([0, 1, 2, 3], dtype=np.int64)
    faces[1] = np.asarray([1, 2, 4, 5], dtype=np.int64)

    assigned = _majority_object_per_vertex(
        6,
        faces,
        np.asarray([9, 4], dtype=np.int64),
    )

    np.testing.assert_array_equal(assigned, np.asarray([9, 4, 4, 9, 4, 4]))


def _write_gt_mesh(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    vertex_data = np.empty(len(vertices), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex_data["x"] = vertices[:, 0]
    vertex_data["y"] = vertices[:, 1]
    vertex_data["z"] = vertices[:, 2]
    face_data = np.empty(
        len(triangles),
        dtype=[("vertex_indices", "i4", (3,)), ("object_id", "i4")],
    )
    face_data["vertex_indices"] = triangles.astype(np.int32)
    face_data["object_id"] = 0
    PlyData(
        [PlyElement.describe(vertex_data, "vertex"), PlyElement.describe(face_data, "face")],
        text=False,
    ).write(path)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    geometry = SparseTsdfVolume()
    depth, rgb, intrinsics = _plane_frame()
    _integrate_twice(geometry, depth, rgb, intrinsics, np.eye(4))
    raw_mesh = geometry.extract_mesh()
    vertices = raw_mesh.vertex.positions.numpy()
    triangles = raw_mesh.triangle.indices.numpy()
    evidence = SparseEvidenceStore()
    ownership = ReversibleOwnershipStore()
    keys = {
        point_to_voxel(vertex, geometry.config.voxel_size_m)
        for vertex in vertices
    }
    for key in sorted(keys):
        evidence.update_semantic(key, label_id=2, support_delta=1.0, revision=1)
        evidence.update_entity(key, entity_id=11, positive_delta=1.0, negative_delta=0.0, timestamp=1.0, revision=1)
        ownership.assign(key, entity_id=11, confidence=1.0, evidence_revision=1)

    snapshot = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(
        snapshot,
        VoxelSnapshotMetadata("fixture", 1, 1.0, 1, 0.05, 8),
        geometry,
        evidence,
        ownership,
    )
    entities = tmp_path / "entities.jsonl"
    entities.write_text(
        json.dumps({"entity_id": 11, "semantic_id": 2, "accepted_view_count": 2}) + "\n",
        encoding="utf-8",
    )
    gt_mesh = tmp_path / "mesh_semantic.ply"
    _write_gt_mesh(gt_mesh, vertices, triangles)
    gt_info = tmp_path / "info_semantic.json"
    gt_info.write_text(
        json.dumps(
            {
                "classes": [{"id": 80, "name": "chair"}],
                "objects": [{"id": 0, "class_id": 80, "class_name": "chair"}],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "replica.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "synthetic_replica",
                "dataset": "Replica",
                "vocabulary": {"classes": ["wall", "chair"]},
                "aliases": {},
                "scenes": [{"scene": "fixture", "ground_truth_scene": "fixture_gt"}],
                "protocol": {"geometry_primary_threshold_m": 0.05},
            }
        ),
        encoding="utf-8",
    )
    return {
        "snapshot": snapshot,
        "entities": entities,
        "gt_mesh": gt_mesh,
        "gt_info": gt_info,
        "manifest": manifest,
    }


def _run(paths: dict[str, Path], output: Path, *, scene: str = "fixture") -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--snapshot",
            str(paths["snapshot"]),
            "--entity-info",
            str(paths["entities"]),
            "--gt-mesh",
            str(paths["gt_mesh"]),
            "--gt-info",
            str(paths["gt_info"]),
            "--manifest",
            str(paths["manifest"]),
            "--scene",
            scene,
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_writes_complete_deterministic_synthetic_evaluation(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = tmp_path / "evaluation-a"
    second = tmp_path / "evaluation-b"

    first_run = _run(paths, first)
    second_run = _run(paths, second)

    assert first_run.returncode == 0, first_run.stderr
    assert second_run.returncode == 0, second_run.stderr
    assert {path.name for path in first.iterdir()} == {
        "metrics.json",
        "per_class_semantic.json",
        "per_class_instance_ap.json",
        "gt_aligned_semantic_ids.npy",
        "gt_aligned_instance_ids.npy",
        "oviv2_instance_mesh.ply",
    }
    metrics = json.loads((first / "metrics.json").read_text())
    for name in ("miou", "macc", "f_miou", "ap25", "ap50", "f5"):
        assert metrics[name] == pytest.approx(1.0)
        assert np.isfinite(metrics[name])
    assert metrics["protocol"]["projection_comparator"] == "strict_less_than"
    assert metrics["protocol"]["snapshot_revision"] == 1
    assert metrics["protocol"]["vocabulary_hash"]
    np.testing.assert_array_equal(
        np.load(first / "gt_aligned_semantic_ids.npy"),
        np.load(second / "gt_aligned_semantic_ids.npy"),
    )
    assert (first / "metrics.json").read_bytes() == (second / "metrics.json").read_bytes()


@pytest.mark.parametrize("failure", ["scene", "vocabulary", "missing_gt"])
def test_cli_preflight_failure_leaves_no_partial_output(tmp_path: Path, failure: str) -> None:
    paths = _fixture(tmp_path)
    scene = "fixture"
    if failure == "scene":
        scene = "not-in-manifest"
    elif failure == "vocabulary":
        paths["entities"].write_text(
            json.dumps({"entity_id": 11, "semantic_id": 99, "accepted_view_count": 2}) + "\n"
        )
    else:
        paths["gt_mesh"] = tmp_path / "missing.ply"
    output = tmp_path / "evaluation"

    result = _run(paths, output, scene=scene)

    assert result.returncode != 0
    assert not output.exists()
