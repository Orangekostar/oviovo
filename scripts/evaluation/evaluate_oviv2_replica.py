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
from plyfile import PlyData, PlyElement

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
from src.oviv2.semantic_fusion import SemanticFusionConfig  # noqa: E402
from src.oviv2.snapshot import VoxelMapSnapshot  # noqa: E402


NON_INSTANCE_CLASSES = {"ceiling", "floor", "wall"}
SEMANTIC_HEADS = ("owner_authoritative", "dense_only", "fused_uncertainty")


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
    face_vertex_indices: Any,
    face_object_ids: np.ndarray,
) -> np.ndarray:
    if len(face_vertex_indices) != len(face_object_ids):
        raise ValueError("Replica GT faces must have one object_id each")
    assigned = np.full(vertex_count, -1, dtype=np.int64)
    if not len(face_vertex_indices):
        return assigned
    faces = [np.asarray(value, dtype=np.int64).reshape(-1) for value in face_vertex_indices]
    if any(len(face) < 3 for face in faces):
        raise ValueError("Replica GT faces must contain at least three vertices")
    flat_vertices = np.concatenate(faces)
    if flat_vertices.size and (
        flat_vertices.min() < 0 or flat_vertices.max() >= vertex_count
    ):
        raise ValueError("Replica GT face contains an out-of-range vertex index")
    repeated_object_ids = np.repeat(
        np.asarray(face_object_ids, dtype=np.int64),
        [len(face) for face in faces],
    )
    pairs = np.column_stack((flat_vertices, repeated_object_ids))
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
    face_vertex_indices = [
        np.asarray(value, dtype=np.int64) for value in faces["vertex_indices"]
    ]
    face_object_ids = np.asarray(faces["object_id"], dtype=np.int64)
    vertex_objects = _majority_object_per_vertex(
        len(vertices), face_vertex_indices, face_object_ids
    )

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


def _entity_semantic_confidence(payload: dict[str, Any], semantic_id: int) -> float:
    if "semantic_posterior" not in payload:
        return 1.0
    posterior = payload["semantic_posterior"]
    if not isinstance(posterior, dict):
        raise TypeError("semantic_posterior must be an object")
    entries = posterior.get("log_evidence")
    if not isinstance(entries, (list, tuple)):
        raise TypeError("semantic_posterior.log_evidence must be a sequence")
    log_values: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("semantic_posterior.log_evidence entries must be pairs")
        entry_id, log_value = entry
        if not isinstance(entry_id, int) or isinstance(entry_id, bool) or entry_id <= 0:
            raise ValueError("semantic_posterior semantic IDs must be positive integers")
        if entry_id in log_values:
            raise ValueError("semantic_posterior semantic IDs must be unique")
        if isinstance(log_value, bool):
            raise TypeError("semantic_posterior log evidence must be numeric")
        normalized_log = float(log_value)
        if not np.isfinite(normalized_log):
            raise ValueError("semantic_posterior log evidence must be finite")
        log_values[entry_id] = normalized_log
    if not log_values or semantic_id not in log_values:
        return 0.0
    maximum = max(log_values.values())
    denominator = sum(np.exp(value - maximum) for value in log_values.values())
    return float(np.exp(log_values[semantic_id] - maximum) / denominator)


def _load_entity_info(path: Path) -> list[EntityEvaluationInfo]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[EntityEvaluationInfo] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("entity info records must be JSON objects")
            record_type = payload.get("record_type")
            if record_type == "registry":
                continue
            if record_type not in {None, "entity"}:
                raise ValueError(f"unknown record_type {record_type!r}")
            semantic_id = payload["semantic_id"]
            semantic_confidence = _entity_semantic_confidence(payload, semantic_id)
            records.append(
                EntityEvaluationInfo(
                    entity_id=payload["entity_id"],
                    semantic_id=semantic_id,
                    accepted_view_count=payload["accepted_view_count"],
                    semantic_confidence=semantic_confidence,
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid entity info at line {line_number}: {exc}") from exc
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_gt_aligned_audit_ply(
    path: Path,
    source_mesh_path: Path,
    ids: np.ndarray,
    *,
    field_name: str,
) -> None:
    source = PlyData.read(source_mesh_path)
    vertices = source["vertex"]
    values = np.asarray(ids, dtype=np.int64)
    if values.shape != (len(vertices),):
        raise ValueError("GT-aligned audit IDs must match GT vertex count")
    vertex_data = np.empty(
        len(vertices),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            (field_name, "i4"),
        ],
    )
    for coordinate in ("x", "y", "z"):
        vertex_data[coordinate] = vertices[coordinate]
    colors = np.column_stack(
        (
            (values * 53 + 31) % 256,
            (values * 97 + 67) % 256,
            (values * 193 + 101) % 256,
        )
    ).astype(np.uint8)
    colors[values == 0] = 64
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    vertex_data[field_name] = values.astype(np.int32)
    elements = [PlyElement.describe(vertex_data, "vertex")]
    if "face" in source:
        elements.append(PlyElement.describe(source["face"].data.copy(), "face"))
    PlyData(elements, text=False).write(path)


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
    semantic_head = getattr(args, "semantic_head", "owner_authoritative")
    if semantic_head not in SEMANTIC_HEADS:
        raise ValueError(
            "semantic_head must be one of " + ", ".join(SEMANTIC_HEADS)
        )
    manifest, scene = _load_manifest(args.manifest, args.scene)
    _verify_scene_inputs(scene, args.gt_mesh, args.gt_info)
    snapshot = VoxelMapSnapshot.load(args.snapshot)
    if snapshot.metadata.scene_id != args.scene:
        raise ValueError("snapshot scene does not match requested manifest scene")
    if semantic_head == "fused_uncertainty" and snapshot.registry is None:
        raise ValueError("fused_uncertainty requires an embedded registry")
    fusion_config = SemanticFusionConfig(
        entity_weight_scale=getattr(args, "fusion_entity_weight_scale", 0.5)
    )

    aliases = _normalized_aliases(manifest.get("aliases", {}))
    classes = [_normalize_label(value, aliases) for value in manifest["vocabulary"]["classes"]]
    class_to_id = {label: index + 1 for index, label in enumerate(classes)}
    valid_semantic_ids = set(class_to_id.values())
    instance_semantic_ids = {
        semantic_id
        for label, semantic_id in class_to_id.items()
        if label not in NON_INSTANCE_CLASSES
    }
    if snapshot.metadata.schema_version in {2, 3}:
        if snapshot.registry is None:
            raise ValueError(
                f"schema v{snapshot.metadata.schema_version} snapshot is missing "
                "its entity registry"
            )
        entity_info = [
            EntityEvaluationInfo(
                entity_id=entity.entity_id,
                semantic_id=entity.semantic_id,
                accepted_view_count=entity.accepted_view_count,
                semantic_confidence=dict(entity.semantic_posterior.probabilities)[
                    entity.semantic_id
                ],
            )
            for entity in sorted(
                snapshot.registry.entities.values(),
                key=lambda value: value.entity_id,
            )
            if entity.lifecycle_state in {"active", "dormant"} and entity.semantic_id > 0
        ]
        entity_semantics = snapshot.registry.semantic_labels()
        entity_posteriors = {
            entity.entity_id: entity.semantic_posterior.probabilities
            for entity in sorted(
                snapshot.registry.entities.values(),
                key=lambda value: value.entity_id,
            )
            if entity.lifecycle_state in {"active", "dormant"}
            and entity.semantic_id > 0
        }
    else:
        entity_info_path = getattr(args, "entity_info", None)
        if entity_info_path is None:
            raise ValueError("schema v1 requires --entity-info")
        entity_info = _load_entity_info(entity_info_path)
        entity_semantics = {
            info.entity_id: (info.semantic_id, info.semantic_confidence)
            for info in entity_info
        }
        entity_posteriors = None
    for info in entity_info:
        if info.semantic_id not in valid_semantic_ids:
            raise ValueError(f"entity {info.entity_id} semantic ID is outside frozen vocabulary")

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
            fusion_config if semantic_head == "fused_uncertainty" else None
        ),
        valid_semantic_ids=valid_semantic_ids,
    )
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
            "semantic_head": semantic_head,
            "vocabulary_hash": vocabulary_hash,
            "snapshot_revision": snapshot.metadata.revision,
            "snapshot_checksums": dict(sorted(snapshot.checksums.items())),
        }
    )
    if semantic_head == "fused_uncertainty":
        metrics["protocol"]["semantic_fusion"] = {
            "mode": "uncertainty_linear",
            "entity_weight_scale": fusion_config.entity_weight_scale,
        }

    def write_output(directory: Path) -> None:
        _write_json(directory / "metrics.json", metrics)
        _write_json(directory / "per_class_semantic.json", metrics["semantic"]["per_class"])
        _write_json(
            directory / "class_agnostic_instance_ap.json",
            metrics["instance"]["class_agnostic"],
        )
        _write_json(
            directory / "semantic_class_instance_ap.json",
            metrics["instance"]["semantic_class_constrained"],
        )
        np.save(directory / "gt_aligned_semantic_ids.npy", projected.semantic_ids)
        np.save(directory / "gt_aligned_instance_ids.npy", projected.entity_ids)
        write_labeled_mesh(directory / "oviv2_instance_mesh.ply", mesh)
        _write_gt_aligned_audit_ply(
            directory / "semantic_map_gt.ply",
            args.gt_mesh,
            projected.semantic_ids,
            field_name="semantic_id",
        )
        _write_gt_aligned_audit_ply(
            directory / "instance_map_gt.ply",
            args.gt_mesh,
            projected.entity_ids,
            field_name="entity_id",
        )

    _publish_output(args.output, write_output)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--entity-info", type=Path)
    parser.add_argument("--gt-mesh", type=Path, required=True)
    parser.add_argument("--gt-info", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-instance-vertices", type=int, default=100)
    parser.add_argument(
        "--semantic-head",
        choices=SEMANTIC_HEADS,
        default="owner_authoritative",
    )
    parser.add_argument(
        "--fusion-entity-weight-scale",
        type=float,
        default=0.5,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    metrics = evaluate(parse_args(argv))
    print(json.dumps({name: metrics[name] for name in ("miou", "macc", "f_miou", "ap25", "ap50", "f5")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
