from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from plyfile import PlyData, PlyElement
import pytest

from scripts.evaluation.evaluate_oviv2_replica import (
    _load_entity_info,
    _majority_object_per_vertex,
)
from src.evaluation.oviv2_replica import EntityEvaluationInfo
from src.oviv2.addressing import point_to_voxel
from src.oviv2.evidence import SparseEvidenceStore
from src.oviv2.entities import EntityRegistry
from src.oviv2.geometry import SparseTsdfVolume
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.snapshot import VoxelMapSnapshot, VoxelSnapshotMetadata

from tests.oviv2.test_geometry import _integrate_twice, _plane_frame
from tests.oviv2.test_entities import _track


SCRIPT = Path("scripts/evaluation/evaluate_oviv2_replica.py")


def test_entity_info_loader_skips_v2_registry_metadata(tmp_path: Path) -> None:
    path = tmp_path / "entities.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "record_type": "registry",
                        "schema_version": 2,
                        "config": {},
                        "next_entity_id": 12,
                    }
                ),
                json.dumps(
                    {
                        "record_type": "entity",
                        "entity_id": 11,
                        "semantic_id": 2,
                        "accepted_view_count": 3,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = _load_entity_info(path)

    assert [
        (
            item.entity_id,
            item.semantic_id,
            item.accepted_view_count,
            item.semantic_confidence,
        )
        for item in loaded
    ] == [(11, 2, 3, 1.0)]


def test_entity_info_loader_derives_stable_v2_posterior_confidence(tmp_path: Path) -> None:
    path = tmp_path / "entities.jsonl"
    path.write_text(
        json.dumps(
            {
                "record_type": "entity",
                "entity_id": 11,
                "semantic_id": 2,
                "accepted_view_count": 3,
                "semantic_posterior": {
                    "log_evidence": [[1, 1000.0], [2, 1000.0 + np.log(4.0)]],
                    "effective_support": 5.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = _load_entity_info(path)

    assert loaded[0].semantic_confidence == pytest.approx(0.8)


@pytest.mark.parametrize("confidence", [False, np.nan, -0.01, 1.01])
def test_entity_evaluation_info_rejects_invalid_semantic_confidence(
    confidence: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="semantic_confidence"):
        EntityEvaluationInfo(1, 2, 1, semantic_confidence=confidence)


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


def _fixture(
    tmp_path: Path,
    *,
    semantic_evidence: bool = True,
    schema_version: int = 1,
) -> dict[str, Path]:
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
        if semantic_evidence:
            evidence.update_semantic(key, label_id=2, support_delta=1.0, revision=1)
        evidence.update_entity(key, entity_id=11, positive_delta=1.0, negative_delta=0.0, timestamp=1.0, revision=1)
        ownership.assign(key, entity_id=11, confidence=1.0, evidence_revision=1)

    snapshot = tmp_path / "snapshot"
    registry = None
    if schema_version == 2:
        registry = EntityRegistry()
        entity = registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
        registry.entities = {11: replace(entity, entity_id=11)}
        registry._next_entity_id = 12
    VoxelMapSnapshot.commit(
        snapshot,
        VoxelSnapshotMetadata("fixture", 1, 1.0, 1, 0.05, 8, schema_version),
        geometry,
        evidence,
        ownership,
        registry=registry,
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


def _run(
    paths: dict[str, Path],
    output: Path,
    *,
    scene: str = "fixture",
    include_entity_info: bool = True,
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(SCRIPT),
        "--snapshot",
        str(paths["snapshot"]),
    ]
    if include_entity_info:
        command.extend(("--entity-info", str(paths["entities"])))
    command.extend(
        [
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
        ]
    )
    return subprocess.run(
        command,
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
        "class_agnostic_instance_ap.json",
        "semantic_class_instance_ap.json",
        "gt_aligned_semantic_ids.npy",
        "gt_aligned_instance_ids.npy",
        "oviv2_instance_mesh.ply",
        "semantic_map_gt.ply",
        "instance_map_gt.ply",
    }
    metrics = json.loads((first / "metrics.json").read_text())
    for name in ("miou", "macc", "f_miou", "ap25", "ap50", "f5"):
        assert metrics[name] == pytest.approx(1.0)
        assert np.isfinite(metrics[name])
    class_agnostic = json.loads(
        (first / "class_agnostic_instance_ap.json").read_text(encoding="utf-8")
    )
    semantic_diagnostic = json.loads(
        (first / "semantic_class_instance_ap.json").read_text(encoding="utf-8")
    )
    assert metrics["ap25"] == class_agnostic["ap25"]
    assert metrics["ap50"] == class_agnostic["ap50"]
    assert semantic_diagnostic == metrics["instance"]["semantic_class_constrained"]
    assert metrics["protocol"]["headline_instance_protocol"] == "class_agnostic"
    assert metrics["protocol"]["semantic_instance_protocol"] == "diagnostic_only"
    assert metrics["protocol"]["projection_comparator"] == "strict_less_than"
    assert metrics["protocol"]["snapshot_revision"] == 1
    assert metrics["protocol"]["vocabulary_hash"]
    np.testing.assert_array_equal(
        np.load(first / "gt_aligned_semantic_ids.npy"),
        np.load(second / "gt_aligned_semantic_ids.npy"),
    )
    semantic_audit = PlyData.read(first / "semantic_map_gt.ply")
    instance_audit = PlyData.read(first / "instance_map_gt.ply")
    np.testing.assert_array_equal(
        semantic_audit["vertex"]["semantic_id"],
        np.load(first / "gt_aligned_semantic_ids.npy"),
    )
    np.testing.assert_array_equal(
        instance_audit["vertex"]["entity_id"],
        np.load(first / "gt_aligned_instance_ids.npy"),
    )
    assert len(semantic_audit["face"]) == len(PlyData.read(paths["gt_mesh"])["face"])
    assert (first / "metrics.json").read_bytes() == (second / "metrics.json").read_bytes()


def test_evaluator_exports_owner_label_without_semantic_voxel_evidence(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, semantic_evidence=False)
    output = tmp_path / "evaluation"

    result = _run(paths, output)

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["miou"] == pytest.approx(1.0)
    assert metrics["macc"] == pytest.approx(1.0)


def test_v2_evaluator_uses_embedded_registry_without_external_entity_info(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, semantic_evidence=False, schema_version=2)
    output = tmp_path / "evaluation"

    result = _run(paths, output, include_entity_info=False)

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["miou"] == pytest.approx(1.0)


def test_v2_evaluator_ignores_invalid_external_entity_info(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, semantic_evidence=False, schema_version=2)
    paths["entities"].write_text("not-json\n", encoding="utf-8")
    output = tmp_path / "evaluation"

    result = _run(paths, output)

    assert result.returncode == 0, result.stderr
    assert json.loads((output / "metrics.json").read_text())["miou"] == pytest.approx(1.0)


def test_v1_evaluator_requires_external_entity_info(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "evaluation"

    result = _run(paths, output, include_entity_info=False)

    assert result.returncode != 0
    assert "schema v1 requires --entity-info" in result.stderr
    assert not output.exists()


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
