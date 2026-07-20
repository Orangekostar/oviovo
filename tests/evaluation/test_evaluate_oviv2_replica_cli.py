from __future__ import annotations

import argparse
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
    evaluate,
)
from src.evaluation.oviv2_replica import EntityEvaluationInfo
from src.oviv2.addressing import point_to_voxel
from src.oviv2.dense_semantics import DenseSemanticProvenance
from src.oviv2.evidence import SparseEvidenceStore
from src.oviv2.entities import EntityRegistry
from src.oviv2.geometry import SparseTsdfVolume
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.semantic_memory import SparseClassPosterior
from src.oviv2.snapshot import VoxelMapSnapshot, VoxelSnapshotMetadata

from tests.oviv2.test_geometry import _integrate_twice, _plane_frame
from tests.oviv2.test_entities import _track


SCRIPT = Path("scripts/evaluation/evaluate_oviv2_replica.py")


def _dense_provenance() -> DenseSemanticProvenance:
    return DenseSemanticProvenance(
        backend="radseg",
        source_commit="a" * 40,
        radio_commit="b" * 40,
        model_id="nvidia/C-RADIOv3-B",
        model_sha256="c" * 64,
        auxiliary_model_sha256="",
        vocabulary_sha256="d" * 64,
        prompt_sha256="e" * 64,
        inference_config_sha256="f" * 64,
        cache_prefix_sha256="1" * 64,
        language_model_id="google/siglip2-so400m-patch16-naflex",
        language_model_revision="2" * 40,
        language_model_sha256="3" * 64,
    )


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
    semantic_label_id: int = 2,
    semantic_supports: tuple[tuple[int, float], ...] | None = None,
    entity_probabilities: tuple[tuple[int, float], ...] | None = None,
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
            supports = semantic_supports or ((semantic_label_id, 1.0),)
            for label_id, support in supports:
                evidence.update_semantic(
                    key,
                    label_id=label_id,
                    support_delta=support,
                    revision=1,
                )
        evidence.update_entity(key, entity_id=11, positive_delta=1.0, negative_delta=0.0, timestamp=1.0, revision=1)
        ownership.assign(key, entity_id=11, confidence=1.0, evidence_revision=1)

    snapshot = tmp_path / "snapshot"
    registry = None
    if schema_version in {2, 3}:
        registry = EntityRegistry()
        entity = registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
        if entity_probabilities is not None:
            posterior = SparseClassPosterior(
                tuple(
                    (semantic_id, float(np.log(probability)))
                    for semantic_id, probability in entity_probabilities
                ),
                effective_support=1.0,
            )
            entity = replace(
                entity,
                semantic_id=posterior.best_semantic_id,
                semantic_posterior=posterior,
                semantic_entropy=posterior.entropy,
                semantic_margin=posterior.margin,
            )
        registry.entities = {11: replace(entity, entity_id=11)}
        registry._next_entity_id = 12
    metadata = VoxelSnapshotMetadata(
        scene_id="fixture",
        frame_id=1,
        timestamp=1.0,
        revision=1,
        voxel_size_m=0.05,
        block_resolution=8,
        schema_version=schema_version,
        dense_semantic_provenance=(
            _dense_provenance() if schema_version == 3 else None
        ),
    )
    VoxelMapSnapshot.commit(
        snapshot,
        metadata,
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
    semantic_head: str | None = None,
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(SCRIPT),
        "--snapshot",
        str(paths["snapshot"]),
    ]
    if include_entity_info:
        command.extend(("--entity-info", str(paths["entities"])))
    if semantic_head is not None:
        command.extend(("--semantic-head", semantic_head))
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
    assert metrics["protocol"]["semantic_head"] == "owner_authoritative"
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


def test_schema3_dense_only_uses_voxel_labels_and_preserves_instance_ids(
    tmp_path: Path,
) -> None:
    paths = _fixture(
        tmp_path,
        semantic_label_id=1,
        schema_version=3,
    )
    owner_output = tmp_path / "owner-evaluation"
    dense_output = tmp_path / "dense-evaluation"

    owner = _run(
        paths,
        owner_output,
        include_entity_info=False,
        semantic_head="owner_authoritative",
    )
    dense = _run(
        paths,
        dense_output,
        include_entity_info=False,
        semantic_head="dense_only",
    )

    assert owner.returncode == 0, owner.stderr
    assert dense.returncode == 0, dense.stderr
    owner_metrics = json.loads((owner_output / "metrics.json").read_text())
    dense_metrics = json.loads((dense_output / "metrics.json").read_text())
    assert owner_metrics["protocol"]["semantic_head"] == "owner_authoritative"
    assert dense_metrics["protocol"]["semantic_head"] == "dense_only"
    owner_semantics = np.load(owner_output / "gt_aligned_semantic_ids.npy")
    dense_semantics = np.load(dense_output / "gt_aligned_semantic_ids.npy")
    owner_instances = np.load(owner_output / "gt_aligned_instance_ids.npy")
    dense_instances = np.load(dense_output / "gt_aligned_instance_ids.npy")
    assert np.any(owner_semantics == 2)
    assert np.any(dense_semantics == 1)
    assert not np.array_equal(owner_semantics, dense_semantics)
    assert np.any(dense_instances > 0)
    np.testing.assert_array_equal(dense_instances, owner_instances)
    assert dense_metrics["ap25"] == owner_metrics["ap25"] == pytest.approx(1.0)


def test_schema3_fused_head_uses_full_entity_posterior_and_records_policy(
    tmp_path: Path,
) -> None:
    paths = _fixture(
        tmp_path,
        semantic_supports=((1, 0.51), (2, 0.49)),
        schema_version=3,
    )
    dense_output = tmp_path / "dense-evaluation"
    fused_output = tmp_path / "fused-evaluation"

    dense = _run(
        paths,
        dense_output,
        include_entity_info=False,
        semantic_head="dense_only",
    )
    fused = _run(
        paths,
        fused_output,
        include_entity_info=False,
        semantic_head="fused_uncertainty",
    )

    assert dense.returncode == 0, dense.stderr
    assert fused.returncode == 0, fused.stderr
    dense_semantics = np.load(dense_output / "gt_aligned_semantic_ids.npy")
    fused_semantics = np.load(fused_output / "gt_aligned_semantic_ids.npy")
    assert np.any(dense_semantics == 1)
    assert np.any(fused_semantics == 2)
    np.testing.assert_array_equal(
        np.load(fused_output / "gt_aligned_instance_ids.npy"),
        np.load(dense_output / "gt_aligned_instance_ids.npy"),
    )
    metrics = json.loads((fused_output / "metrics.json").read_text())
    assert metrics["miou"] == pytest.approx(1.0)
    assert metrics["protocol"]["semantic_head"] == "fused_uncertainty"
    assert metrics["protocol"]["semantic_fusion"] == {
        "entity_weight_scale": 0.5,
        "mode": "uncertainty_linear",
    }


def test_fused_head_rejects_snapshot_without_embedded_registry(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, schema_version=1)
    output = tmp_path / "evaluation"

    result = _run(paths, output, semantic_head="fused_uncertainty")

    assert result.returncode != 0
    assert "embedded registry" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("oov_source", ["dense", "entity"])
def test_fused_head_rejects_any_distribution_class_outside_frozen_vocabulary(
    tmp_path: Path,
    oov_source: str,
) -> None:
    paths = _fixture(
        tmp_path,
        semantic_supports=(
            ((1, 0.55), (999, 0.45))
            if oov_source == "dense"
            else ((1, 0.55), (2, 0.45))
        ),
        entity_probabilities=(
            ((2, 0.6), (999, 0.4))
            if oov_source == "entity"
            else ((2, 1.0),)
        ),
        schema_version=3,
    )
    output = tmp_path / "evaluation"

    result = _run(
        paths,
        output,
        include_entity_info=False,
        semantic_head="fused_uncertainty",
    )

    assert result.returncode != 0
    assert "outside frozen vocabulary" in result.stderr
    assert not output.exists()


def test_unknown_semantic_head_is_rejected_before_output_creation(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "evaluation"
    args = argparse.Namespace(
        snapshot=paths["snapshot"],
        entity_info=paths["entities"],
        gt_mesh=paths["gt_mesh"],
        gt_info=paths["gt_info"],
        manifest=paths["manifest"],
        scene="fixture",
        output=output,
        min_instance_vertices=100,
        semantic_head="fused",
    )

    with pytest.raises(ValueError, match="semantic_head"):
        evaluate(args)

    assert not output.exists()


def test_cli_rejects_unknown_semantic_head_before_output_creation(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "evaluation"

    result = _run(paths, output, semantic_head="fused")

    assert result.returncode != 0
    assert "--semantic-head" in result.stderr
    assert not output.exists()


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
