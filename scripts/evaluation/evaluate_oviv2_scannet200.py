#!/usr/bin/env python3
"""Evaluate an OVIV2 map on the frozen ScanNet200 five-scene split."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.baselines.static_metrics import evaluate_static_predictions  # noqa: E402
from src.evaluation.contracts import (  # noqa: E402
    EntityPrediction,
    GroundTruthSnapshot,
    MapSnapshot,
)
from src.oviv2.meshing import (  # noqa: E402
    LabeledMesh,
    derive_labeled_mesh,
    write_labeled_mesh,
)
from src.oviv2.semantic_fusion import SemanticFusionConfig  # noqa: E402
from src.oviv2.snapshot import VoxelMapSnapshot  # noqa: E402


SEMANTIC_HEADS = ("owner_authoritative", "dense_only", "fused_uncertainty")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_axis_alignment(path: str | Path) -> np.ndarray:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, separator, raw_value = line.partition(" = ")
        if separator and key == "axisAlignment":
            values = np.fromstring(raw_value, sep=" ", dtype=np.float64)
            if values.size != 16 or not np.all(np.isfinite(values)):
                raise ValueError("axisAlignment must contain 16 finite values")
            return values.reshape(4, 4)
    raise ValueError("metadata does not contain axisAlignment")


def load_scannet_ground_truth(
    path: str | Path,
    *,
    scene_id: str,
    class_ids: Sequence[int],
    classes: Sequence[str],
) -> GroundTruthSnapshot:
    """Load an official axis-aligned ScanNet200 vertex PLY."""
    normalized_ids = tuple(int(value) for value in class_ids)
    normalized_classes = tuple(str(value) for value in classes)
    if (
        not normalized_ids
        or len(normalized_ids) != len(normalized_classes)
        or len(set(normalized_ids)) != len(normalized_ids)
        or len(set(normalized_classes)) != len(normalized_classes)
    ):
        raise ValueError("ScanNet class IDs and names must be unique aligned sequences")
    vertices = PlyData.read(Path(path))["vertex"]
    fields = set(vertices.data.dtype.names or ())
    required = {"x", "y", "z", "label", "instance_id"}
    if not required <= fields:
        raise ValueError("official ScanNet PLY is missing required vertex fields")
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    id_to_class = dict(zip(normalized_ids, normalized_classes, strict=True))
    labels = np.asarray(
        [id_to_class.get(int(value), "__ignored__") for value in vertices["label"]],
        dtype=object,
    )
    return GroundTruthSnapshot(
        scene_id=scene_id,
        timestamp=1.0,
        points_xyz=np.asarray(points, dtype=np.float32),
        semantic_labels=labels,
        instance_ids=np.asarray(vertices["instance_id"], dtype=np.int64),
    )


def _entity(
    entity_id: str,
    points: np.ndarray,
    *,
    label: str | None,
    score: float,
) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=points,
        semantic_embedding=None,
        semantic_label=label,
        semantic_score=score,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=1.0,
    )


def evaluate_scannet_mesh(
    *,
    scene_id: str,
    mesh: LabeledMesh,
    axis_alignment: np.ndarray,
    ground_truth: GroundTruthSnapshot,
    classes: Sequence[str],
    entity_scores: Mapping[int, float],
    distance_threshold_m: float = 0.05,
    min_instance_points: int = 100,
) -> dict:
    """Evaluate fused per-vertex semantics and ownership entities independently."""
    transform = np.asarray(axis_alignment, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("axis_alignment must be a finite 4x4 matrix")
    points = np.asarray(
        mesh.vertices_xyz @ transform[:3, :3].T + transform[:3, 3],
        dtype=np.float32,
    )
    vocabulary = tuple(str(value) for value in classes)
    semantic_entities = []
    for semantic_id in sorted(int(value) for value in np.unique(mesh.semantic_ids) if value > 0):
        if semantic_id > len(vocabulary):
            raise ValueError("mesh semantic ID is outside the frozen vocabulary")
        keep = mesh.semantic_ids == semantic_id
        semantic_entities.append(
            _entity(
                f"semantic:{semantic_id:03d}",
                points[keep],
                label=vocabulary[semantic_id - 1],
                score=float(np.mean(mesh.semantic_confidence[keep])),
            )
        )
    instance_entities = []
    for entity_id in sorted(int(value) for value in np.unique(mesh.entity_ids) if value > 0):
        keep = mesh.entity_ids == entity_id
        score = float(
            entity_scores.get(entity_id, float(np.mean(mesh.ownership_confidence[keep])))
        )
        instance_entities.append(
            _entity(
                f"entity:{entity_id}",
                points[keep],
                label=None,
                score=score,
            )
        )
    semantic_snapshot = MapSnapshot(
        method="OVIV2",
        scene_id=scene_id,
        timestamp=1.0,
        entities=semantic_entities,
        background_xyz=None,
        scope="current",
    )
    instance_snapshot = MapSnapshot(
        method="OVIV2",
        scene_id=scene_id,
        timestamp=1.0,
        entities=instance_entities,
        background_xyz=None,
        scope="current",
    )
    geometry_snapshot = MapSnapshot(
        method="OVIV2",
        scene_id=scene_id,
        timestamp=1.0,
        entities=[],
        background_xyz=points,
        scope="current",
    )
    metrics = evaluate_static_predictions(
        semantic_prediction=semantic_snapshot,
        instance_prediction=instance_snapshot,
        geometry_prediction=geometry_snapshot,
        ground_truth=ground_truth,
        semantic_vocabulary=vocabulary,
        instance_vocabulary=vocabulary,
        distance_threshold_m=distance_threshold_m,
        min_instance_points=min_instance_points,
    )
    metrics["protocol"].update(
        {
            "dataset": "ScanNet200",
            "coordinate_frame": "official axis-aligned world",
            "semantic_head": "fused per-vertex semantic ID",
            "instance_head": "OVIV2 ownership entity ID",
        }
    )
    return metrics


def _load_manifest(path: Path, scene_id: str) -> tuple[dict, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != "ScanNet200":
        raise ValueError("manifest must be ScanNet200 schema_version 1")
    matches = [value for value in manifest.get("scenes", ()) if value.get("scene") == scene_id]
    if len(matches) != 1:
        raise ValueError(f"scene {scene_id!r} is not uniquely defined")
    vocabulary = manifest.get("vocabulary", {})
    classes = vocabulary.get("classes")
    class_ids = vocabulary.get("class_ids")
    if (
        not isinstance(classes, list)
        or not isinstance(class_ids, list)
        or len(classes) != len(class_ids)
        or not classes
    ):
        raise ValueError("manifest vocabulary classes/class_ids are invalid")
    return manifest, matches[0]


def _verify_scene_inputs(scene: dict, gt_path: Path, metadata_path: Path) -> None:
    expected = {
        "official GT": (scene.get("official_gt_path"), scene.get("official_gt_sha256"), gt_path),
        "metadata": (scene.get("metadata_path"), scene.get("metadata_sha256"), metadata_path),
    }
    for name, (recorded_path, recorded_hash, actual_path) in expected.items():
        if Path(str(recorded_path)).resolve() != actual_path.resolve():
            raise ValueError(f"{name} path does not match manifest")
        if not actual_path.is_file() or _sha256(actual_path) != recorded_hash:
            raise ValueError(f"{name} hash does not match manifest")


def _entity_score(entity) -> float:
    semantic_confidence = 0.0
    probabilities = tuple(entity.semantic_posterior.probabilities)
    if probabilities:
        semantic_confidence = max(float(value) for _, value in probabilities)
    return float(entity.accepted_view_count) + semantic_confidence


def _derive_snapshot_mesh(
    snapshot: VoxelMapSnapshot,
    *,
    semantic_head: str,
    valid_semantic_ids: set[int],
    fusion_entity_weight_scale: float,
) -> tuple[LabeledMesh, dict[int, float]]:
    if snapshot.registry is None:
        raise ValueError("ScanNet evaluation requires an embedded entity registry")
    entities = [
        entity
        for entity in snapshot.registry.entities.values()
        if entity.lifecycle_state in {"active", "dormant"}
    ]
    entity_scores = {entity.entity_id: _entity_score(entity) for entity in entities}
    entity_semantics = {
        entity.entity_id: (
            entity.semantic_id,
            dict(entity.semantic_posterior.probabilities)[entity.semantic_id],
        )
        for entity in entities
        if entity.semantic_id > 0
    }
    entity_posteriors = {
        entity.entity_id: entity.semantic_posterior.probabilities
        for entity in entities
        if entity.semantic_id > 0
    }
    mesh = derive_labeled_mesh(
        snapshot.geometry,
        snapshot.evidence,
        snapshot.ownership,
        entity_semantics=(
            entity_semantics if semantic_head == "owner_authoritative" else None
        ),
        entity_posteriors=(
            entity_posteriors if semantic_head == "fused_uncertainty" else None
        ),
        semantic_fusion=(
            SemanticFusionConfig(entity_weight_scale=fusion_entity_weight_scale)
            if semantic_head == "fused_uncertainty"
            else None
        ),
        valid_semantic_ids=valid_semantic_ids,
    )
    return mesh, entity_scores


def _aligned_mesh(mesh: LabeledMesh, transform: np.ndarray) -> LabeledMesh:
    return LabeledMesh(
        vertices_xyz=np.asarray(
            mesh.vertices_xyz @ transform[:3, :3].T + transform[:3, 3],
            dtype=np.float32,
        ),
        triangles=mesh.triangles,
        colors_rgb=mesh.colors_rgb,
        semantic_ids=mesh.semantic_ids,
        entity_ids=mesh.entity_ids,
        semantic_confidence=mesh.semantic_confidence,
        ownership_confidence=mesh.ownership_confidence,
    )


def _project_to_gt(
    mesh: LabeledMesh,
    ground_truth: GroundTruthSnapshot,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    semantic_ids = np.zeros(len(ground_truth.points_xyz), dtype=np.int64)
    entity_ids = np.zeros(len(ground_truth.points_xyz), dtype=np.int64)
    if len(mesh.vertices_xyz) and len(ground_truth.points_xyz):
        distances, nearest = cKDTree(mesh.vertices_xyz).query(
            ground_truth.points_xyz,
            k=1,
            workers=-1,
        )
        matched = np.asarray(distances <= threshold, dtype=bool)
        indices = np.asarray(nearest[matched], dtype=np.int64)
        semantic_ids[matched] = mesh.semantic_ids[indices]
        entity_ids[matched] = mesh.entity_ids[indices]
    return semantic_ids, entity_ids


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _publish_output(target: Path, writer) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not target.is_dir():
        raise ValueError("evaluation output exists and is not a directory")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    backup: Path | None = None
    try:
        writer(temporary)
        if target.exists():
            backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except BaseException:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def evaluate(args: argparse.Namespace) -> dict:
    if args.semantic_head not in SEMANTIC_HEADS:
        raise ValueError("unsupported semantic head")
    manifest_path = args.manifest.resolve()
    manifest, scene = _load_manifest(manifest_path, args.scene)
    _verify_scene_inputs(scene, args.gt_ply, args.metadata)
    classes = tuple(str(value) for value in manifest["vocabulary"]["classes"])
    class_ids = tuple(int(value) for value in manifest["vocabulary"]["class_ids"])
    valid_semantic_ids = set(range(1, len(classes) + 1))
    snapshot = VoxelMapSnapshot.load(args.snapshot)
    if snapshot.metadata.scene_id != args.scene:
        raise ValueError("snapshot scene does not match requested scene")
    mesh, entity_scores = _derive_snapshot_mesh(
        snapshot,
        semantic_head=args.semantic_head,
        valid_semantic_ids=valid_semantic_ids,
        fusion_entity_weight_scale=args.fusion_entity_weight_scale,
    )
    transform = load_axis_alignment(args.metadata)
    ground_truth = load_scannet_ground_truth(
        args.gt_ply,
        scene_id=args.scene,
        class_ids=class_ids,
        classes=classes,
    )
    metrics = evaluate_scannet_mesh(
        scene_id=args.scene,
        mesh=mesh,
        axis_alignment=transform,
        ground_truth=ground_truth,
        classes=classes,
        entity_scores=entity_scores,
        distance_threshold_m=args.distance_threshold_m,
        min_instance_points=args.min_instance_points,
    )
    metrics.update(
        {
            "miou": metrics["semantic"]["miou"],
            "macc": metrics["semantic"]["macc"],
            "f_miou": metrics["semantic"]["f_miou"],
            "ap25": metrics["instance"]["ap25"],
            "ap50": metrics["instance"]["ap50"],
            "f5": metrics["geometry"]["f5"],
        }
    )
    metrics["protocol"].update(
        {
            "manifest_id": manifest.get("manifest_id"),
            "manifest_sha256": _sha256(manifest_path),
            "scene_id": args.scene,
            "semantic_head": args.semantic_head,
            "fusion_entity_weight_scale": args.fusion_entity_weight_scale,
            "snapshot_revision": snapshot.metadata.revision,
            "snapshot_checksums": dict(sorted(snapshot.checksums.items())),
            "axis_alignment": transform.tolist(),
        }
    )
    aligned = _aligned_mesh(mesh, transform)
    projected_semantic, projected_instance = _project_to_gt(
        aligned,
        ground_truth,
        args.distance_threshold_m,
    )

    def write(directory: Path) -> None:
        _write_json(directory / "metrics.json", metrics)
        _write_json(directory / "per_class_semantic.json", metrics["semantic"]["per_class"])
        _write_json(directory / "class_agnostic_instance_ap.json", metrics["instance"])
        np.save(directory / "gt_aligned_semantic_ids.npy", projected_semantic)
        np.save(directory / "gt_aligned_instance_ids.npy", projected_instance)
        write_labeled_mesh(directory / "oviv2_instance_mesh_aligned.ply", aligned)

    _publish_output(args.output, write)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--entity-info", type=Path)
    parser.add_argument("--gt-ply", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-instance-points", type=int, default=100)
    parser.add_argument("--distance-threshold-m", type=float, default=0.05)
    parser.add_argument("--semantic-head", choices=SEMANTIC_HEADS, default="fused_uncertainty")
    parser.add_argument("--fusion-entity-weight-scale", type=float, default=0.49)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    metrics = evaluate(parse_args(argv))
    print(
        json.dumps(
            {name: metrics[name] for name in ("miou", "macc", "f_miou", "ap25", "ap50", "f5")},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
