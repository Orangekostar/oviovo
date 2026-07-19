#!/usr/bin/env python3
"""Evaluate one immutable OVIV2 voxel snapshot on frozen Replica ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid
from typing import Any

import numpy as np
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.oviv2_replica import (  # noqa: E402
    EntityEvaluationInfo,
    ReplicaGroundTruth,
    evaluate_replica_voxel_map,
    project_mesh_to_gt,
)
from src.oviv2.meshing import derive_labeled_mesh, write_labeled_mesh  # noqa: E402
from src.oviv2.snapshot import VoxelMapSnapshot  # noqa: E402


NON_INSTANCE_CLASSES = {"ceiling", "floor", "wall"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _basic_label(label: str) -> str:
    return str(label).strip().lower().replace("_", "-")


def _normalized_aliases(values: dict[str, str]) -> dict[str, str]:
    return {_basic_label(source): _basic_label(target) for source, target in values.items()}


def _normalize_label(label: str, aliases: dict[str, str]) -> str:
    normalized = _basic_label(label)
    return aliases.get(normalized, normalized)


def _load_manifest(path: Path, scene_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != "Replica":
        raise ValueError("manifest must use Replica schema_version 1")
    matches = [item for item in manifest.get("scenes", ()) if item.get("scene") == scene_id]
    if len(matches) != 1:
        raise ValueError(f"scene {scene_id!r} is not uniquely defined by the manifest")
    vocabulary = manifest.get("vocabulary", {})
    classes = vocabulary.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("manifest vocabulary must contain a non-empty classes list")
    normalized_classes = [_basic_label(value) for value in classes]
    if len(set(normalized_classes)) != len(normalized_classes):
        raise ValueError("manifest vocabulary contains duplicate normalized classes")
    source_path = vocabulary.get("source_path")
    source_sha256 = vocabulary.get("source_sha256")
    if source_path is not None or source_sha256 is not None:
        if not isinstance(source_path, str) or not isinstance(source_sha256, str):
            raise ValueError("manifest vocabulary source path/hash must be specified together")
        source = REPO_ROOT / source_path
        if not source.is_file() or _sha256(source) != source_sha256:
            raise ValueError("manifest vocabulary source hash mismatch")
    return manifest, matches[0]


def _verify_scene_inputs(scene: dict[str, Any], mesh_path: Path, info_path: Path) -> None:
    for path in (mesh_path, info_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = {
        mesh_path: scene.get("semantic_mesh_sha256"),
        info_path: scene.get("semantic_info_sha256"),
    }
    for path, digest in expected.items():
        if digest is not None and _sha256(path) != digest:
            raise ValueError(f"manifest hash mismatch for {path.name}")


def _majority_object_per_vertex(
    vertex_count: int,
    triangles: np.ndarray,
    face_object_ids: np.ndarray,
) -> np.ndarray:
    if triangles.shape != (len(face_object_ids), 3):
        raise ValueError("Replica GT faces must be triangles with one object_id each")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= vertex_count):
        raise ValueError("Replica GT face contains an out-of-range vertex index")
    assigned = np.full(vertex_count, -1, dtype=np.int64)
    if not len(triangles):
        return assigned
    pairs = np.column_stack((triangles.reshape(-1), np.repeat(face_object_ids, 3)))
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    order = np.lexsort((unique_pairs[:, 1], -counts, unique_pairs[:, 0]))
    ranked = unique_pairs[order]
    first = np.concatenate(([True], ranked[1:, 0] != ranked[:-1, 0]))
    selected = ranked[first]
    assigned[selected[:, 0]] = selected[:, 1]
    return assigned


def load_replica_ground_truth(
    mesh_path: Path,
    info_path: Path,
    *,
    class_to_id: dict[str, int],
    aliases: dict[str, str],
) -> ReplicaGroundTruth:
    ply = PlyData.read(mesh_path)
    if "vertex" not in ply or "face" not in ply:
        raise ValueError("Replica GT PLY must contain vertex and face elements")
    vertices_element = ply["vertex"]
    vertices = np.column_stack(
        (vertices_element["x"], vertices_element["y"], vertices_element["z"])
    ).astype(np.float32)
    faces = ply["face"]
    if "object_id" not in faces.data.dtype.names:
        raise ValueError("Replica GT PLY faces must contain object_id")
    triangles = np.asarray([np.asarray(value, dtype=np.int64) for value in faces["vertex_indices"]])
    if len(triangles) == 0:
        triangles = np.empty((0, 3), dtype=np.int64)
    face_object_ids = np.asarray(faces["object_id"], dtype=np.int64)
    vertex_objects = _majority_object_per_vertex(len(vertices), triangles, face_object_ids)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    object_labels = {
        int(item["id"]): _normalize_label(
            str(item.get("class_name", item.get("name", ""))),
            aliases,
        )
        for item in info.get("objects", ())
    }
    semantic_ids = np.zeros(len(vertices), dtype=np.int64)
    instance_ids = np.zeros(len(vertices), dtype=np.int64)
    for object_id in sorted(int(value) for value in np.unique(vertex_objects) if value >= 0):
        mask = vertex_objects == object_id
        semantic_ids[mask] = class_to_id.get(object_labels.get(object_id, ""), 0)
        instance_ids[mask] = object_id + 1
    return ReplicaGroundTruth(vertices, semantic_ids, instance_ids)


def _load_entity_info(path: Path) -> list[EntityEvaluationInfo]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[EntityEvaluationInfo] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            records.append(
                EntityEvaluationInfo(
                    entity_id=payload["entity_id"],
                    semantic_id=payload["semantic_id"],
                    accepted_view_count=payload["accepted_view_count"],
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid entity info at line {line_number}: {exc}") from exc
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest, scene = _load_manifest(args.manifest, args.scene)
    _verify_scene_inputs(scene, args.gt_mesh, args.gt_info)
    snapshot = VoxelMapSnapshot.load(args.snapshot)
    if snapshot.metadata.scene_id != args.scene:
        raise ValueError("snapshot scene does not match requested manifest scene")

    aliases = _normalized_aliases(manifest.get("aliases", {}))
    classes = [_normalize_label(value, aliases) for value in manifest["vocabulary"]["classes"]]
    class_to_id = {label: index + 1 for index, label in enumerate(classes)}
    valid_semantic_ids = set(class_to_id.values())
    instance_semantic_ids = {
        semantic_id
        for label, semantic_id in class_to_id.items()
        if label not in NON_INSTANCE_CLASSES
    }
    entity_info = _load_entity_info(args.entity_info)
    for info in entity_info:
        if info.semantic_id not in valid_semantic_ids:
            raise ValueError(f"entity {info.entity_id} semantic ID is outside frozen vocabulary")

    mesh = derive_labeled_mesh(snapshot.geometry, snapshot.evidence, snapshot.ownership)
    mesh_semantic_ids = {int(value) for value in np.unique(mesh.semantic_ids) if value > 0}
    if not mesh_semantic_ids.issubset(valid_semantic_ids):
        raise ValueError("snapshot semantic evidence is outside frozen vocabulary")
    ground_truth = load_replica_ground_truth(
        args.gt_mesh,
        args.gt_info,
        class_to_id=class_to_id,
        aliases=aliases,
    )
    threshold = float(
        manifest.get("protocol", {}).get("geometry_primary_threshold_m", 0.05)
    )
    projected = project_mesh_to_gt(mesh, ground_truth.vertices_xyz, threshold)
    metrics = evaluate_replica_voxel_map(
        mesh,
        ground_truth,
        entity_info,
        valid_semantic_ids=valid_semantic_ids,
        instance_semantic_ids=instance_semantic_ids,
        min_instance_vertices=args.min_instance_vertices,
        distance_threshold_m=threshold,
    )
    vocabulary_payload = {
        "classes": classes,
        "aliases": dict(sorted(aliases.items())),
    }
    vocabulary_hash = hashlib.sha256(
        json.dumps(vocabulary_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    metrics["protocol"].update(
        {
            "manifest_id": manifest.get("manifest_id"),
            "scene_id": args.scene,
            "vocabulary_hash": vocabulary_hash,
            "snapshot_revision": snapshot.metadata.revision,
            "snapshot_checksums": dict(sorted(snapshot.checksums.items())),
        }
    )

    def write_output(directory: Path) -> None:
        _write_json(directory / "metrics.json", metrics)
        _write_json(directory / "per_class_semantic.json", metrics["semantic"]["per_class"])
        _write_json(directory / "per_class_instance_ap.json", metrics["instance"]["per_class"])
        np.save(directory / "gt_aligned_semantic_ids.npy", projected.semantic_ids)
        np.save(directory / "gt_aligned_instance_ids.npy", projected.entity_ids)
        write_labeled_mesh(directory / "oviv2_instance_mesh.ply", mesh)

    _publish_output(args.output, write_output)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--entity-info", type=Path, required=True)
    parser.add_argument("--gt-mesh", type=Path, required=True)
    parser.add_argument("--gt-info", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-instance-vertices", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    metrics = evaluate(parse_args(argv))
    print(json.dumps({name: metrics[name] for name in ("miou", "macc", "f_miou", "ap25", "ap50", "f5")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
