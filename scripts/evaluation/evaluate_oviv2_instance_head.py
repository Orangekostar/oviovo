#!/usr/bin/env python3
"""Evaluate GT-free OVIV2 instance hypotheses on a frozen Replica snapshot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_replica import (  # noqa: E402
    NON_INSTANCE_CLASSES,
    _load_manifest,
    _normalize_label,
    _normalized_aliases,
    _load_semantic_replay,
    _verify_scene_inputs,
    _write_json,
    load_replica_ground_truth,
)
from src.evaluation.oviv2_instance_head import (  # noqa: E402
    InstanceHeadConfig,
    build_instance_hypotheses,
    evaluate_instance_hypotheses,
    evaluate_projected_instance_hypotheses,
    project_instance_hypotheses,
)
from src.evaluation.oviv2_instance_ensemble import (  # noqa: E402
    AuxiliaryInstanceConfig,
    build_auxiliary_entity_hypotheses,
)
from src.evaluation.oviv2_geometry_head import (  # noqa: E402
    GeometrySemanticStabilizationConfig,
    stabilize_low_support_semantics,
)
from src.evaluation.oviv2_replica import (  # noqa: E402
    EntityEvaluationInfo,
    evaluate_replica_voxel_map,
)
from src.oviv2.meshing import derive_labeled_mesh  # noqa: E402
from src.oviv2.semantic_fusion import SemanticFusionConfig  # noqa: E402
from src.oviv2.snapshot import VoxelMapSnapshot  # noqa: E402


def _normalized_config(config: InstanceHeadConfig) -> dict[str, Any]:
    return asdict(config)


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_auxiliary_run_manifest(
    path: Path,
    *,
    snapshot: VoxelMapSnapshot,
    scene_id: str,
    benchmark_manifest_hash: str,
    vocabulary_hash: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("auxiliary instance run manifest must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("auxiliary instance run manifest must be an object")
    if payload.get("method") != "OVIV2" or payload.get("scene") != scene_id:
        raise ValueError("auxiliary instance run manifest method/scene mismatch")
    if payload.get("final_revision") != snapshot.metadata.revision:
        raise ValueError("auxiliary instance run manifest revision mismatch")
    if payload.get("benchmark_manifest_hash") != benchmark_manifest_hash:
        raise ValueError("auxiliary instance benchmark manifest hash mismatch")
    if payload.get("vocabulary_hash") != vocabulary_hash:
        raise ValueError("auxiliary instance vocabulary hash mismatch")
    if not _valid_sha256(payload.get("algorithm_hash")):
        raise ValueError("auxiliary instance algorithm hash must be SHA-256")
    artifact_checksums = payload.get("artifact_checksums")
    if not isinstance(artifact_checksums, dict):
        raise ValueError("auxiliary instance artifact checksums are missing")
    for name, digest in snapshot.checksums.items():
        key = f"final/oviv2_voxel_snapshot.npz/{name}"
        if artifact_checksums.get(key) != digest:
            raise ValueError(f"auxiliary instance snapshot checksum mismatch: {name}")
    return payload


def _publish_fresh_output(target: Path, writer) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"output already exists: {target}")
    lock = target.parent / f".{target.name}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"output publication already in progress: {target}") from error
    os.close(descriptor)
    temporary: Path | None = None
    try:
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        )
        writer(temporary)
        if target.exists():
            raise FileExistsError(f"output already exists: {target}")
        os.rename(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    auxiliary_snapshot_path = getattr(args, "auxiliary_instance_snapshot", None)
    auxiliary_run_manifest_path = getattr(
        args,
        "auxiliary_instance_run_manifest",
        None,
    )
    if (auxiliary_snapshot_path is None) != (auxiliary_run_manifest_path is None):
        raise ValueError(
            "auxiliary instance snapshot and run manifest must be supplied together"
        )
    manifest, scene = _load_manifest(args.manifest, args.scene)
    _verify_scene_inputs(scene, args.gt_mesh, args.gt_info)
    snapshot = VoxelMapSnapshot.load(args.snapshot)
    if snapshot.metadata.scene_id != args.scene:
        raise ValueError("snapshot scene does not match requested manifest scene")
    if snapshot.metadata.schema_version not in {2, 3} or snapshot.registry is None:
        raise ValueError("instance head requires a schema v2/v3 registry snapshot")

    aliases = _normalized_aliases(manifest.get("aliases", {}))
    classes = [
        _normalize_label(value, aliases)
        for value in manifest["vocabulary"]["classes"]
    ]
    class_to_id = {label: index + 1 for index, label in enumerate(classes)}
    valid_semantic_ids = set(class_to_id.values())
    vocabulary_payload = {
        "classes": classes,
        "aliases": dict(sorted(aliases.items())),
    }
    vocabulary_hash = _json_hash(vocabulary_payload)
    semantic_evidence_path = getattr(args, "semantic_evidence", None)
    semantic_replay_manifest_path = getattr(args, "semantic_replay_manifest", None)
    if (semantic_evidence_path is None) != (semantic_replay_manifest_path is None):
        raise ValueError(
            "--semantic-evidence and --semantic-replay-manifest must be supplied together"
        )
    if semantic_evidence_path is None:
        evaluation_evidence = snapshot.evidence
        semantic_replay_protocol = None
    else:
        evaluation_evidence, semantic_replay_protocol = _load_semantic_replay(
            snapshot=snapshot,
            evidence_path=semantic_evidence_path,
            manifest_path=semantic_replay_manifest_path,
            scene_id=args.scene,
            benchmark_manifest_path=args.manifest,
            vocabulary_hash=vocabulary_hash,
        )
    structural_semantic_ids = frozenset(
        semantic_id
        for label, semantic_id in class_to_id.items()
        if label in NON_INSTANCE_CLASSES
    )
    instance_semantic_ids = valid_semantic_ids - set(structural_semantic_ids)

    auxiliary_snapshot = None
    auxiliary_run_manifest = None
    if auxiliary_snapshot_path is not None:
        auxiliary_snapshot = VoxelMapSnapshot.load(auxiliary_snapshot_path)
        if (
            auxiliary_snapshot.metadata.scene_id != args.scene
            or auxiliary_snapshot.metadata.revision != snapshot.metadata.revision
        ):
            raise ValueError("auxiliary instance snapshot scene/revision mismatch")
        if (
            auxiliary_snapshot.metadata.schema_version not in {2, 3}
            or auxiliary_snapshot.registry is None
        ):
            raise ValueError(
                "auxiliary instance snapshot requires a schema v2/v3 registry"
            )
        auxiliary_run_manifest = _load_auxiliary_run_manifest(
            auxiliary_run_manifest_path,
            snapshot=auxiliary_snapshot,
            scene_id=args.scene,
            benchmark_manifest_hash=_sha256_file(args.manifest),
            vocabulary_hash=vocabulary_hash,
        )

    entities = [
        entity
        for entity in sorted(
            snapshot.registry.entities.values(),
            key=lambda value: value.entity_id,
        )
        if entity.lifecycle_state in {"active", "dormant"} and entity.semantic_id > 0
    ]
    legacy_entity_info = [
        EntityEvaluationInfo(
            entity_id=entity.entity_id,
            semantic_id=entity.semantic_id,
            accepted_view_count=entity.accepted_view_count,
            semantic_confidence=dict(entity.semantic_posterior.probabilities)[
                entity.semantic_id
            ],
        )
        for entity in entities
    ]
    head_entity_info = [
        EntityEvaluationInfo(
            entity_id=entity.entity_id,
            semantic_id=entity.semantic_id,
            accepted_view_count=entity.accepted_view_count,
            semantic_confidence=entity.semantic_margin,
        )
        for entity in entities
    ]
    entity_posteriors = {
        entity.entity_id: entity.semantic_posterior.probabilities for entity in entities
    }
    fusion_scale = float(args.fusion_entity_weight_scale)
    mesh_weight_threshold = float(getattr(args, "mesh_weight_threshold", 1.0))
    if not np.isfinite(mesh_weight_threshold) or mesh_weight_threshold < 0.0:
        raise ValueError("mesh_weight_threshold must be finite and non-negative")
    semantic_reference_threshold = getattr(
        args,
        "semantic_reference_weight_threshold",
        None,
    )
    semantic_transfer_distance = getattr(args, "semantic_transfer_distance_m", None)
    if (semantic_reference_threshold is None) != (semantic_transfer_distance is None):
        raise ValueError(
            "semantic reference threshold and transfer distance must be supplied together"
        )
    mesh = derive_labeled_mesh(
        snapshot.geometry,
        evaluation_evidence,
        snapshot.ownership,
        weight_threshold=mesh_weight_threshold,
        entity_posteriors=entity_posteriors,
        semantic_fusion=SemanticFusionConfig(entity_weight_scale=fusion_scale),
        valid_semantic_ids=valid_semantic_ids,
    )
    geometry_stabilization_protocol = None
    if semantic_reference_threshold is not None:
        reference_threshold = float(semantic_reference_threshold)
        if (
            not np.isfinite(reference_threshold)
            or reference_threshold <= mesh_weight_threshold
        ):
            raise ValueError(
                "semantic reference weight threshold must be finite and above the mesh threshold"
            )
        stabilization_config = GeometrySemanticStabilizationConfig(
            maximum_transfer_distance_m=semantic_transfer_distance,
        )
        reference_mesh = derive_labeled_mesh(
            snapshot.geometry,
            evaluation_evidence,
            snapshot.ownership,
            weight_threshold=reference_threshold,
            entity_posteriors=entity_posteriors,
            semantic_fusion=SemanticFusionConfig(entity_weight_scale=fusion_scale),
            valid_semantic_ids=valid_semantic_ids,
        )
        stabilization = stabilize_low_support_semantics(
            mesh,
            reference_mesh,
            stabilization_config,
        )
        mesh = stabilization.mesh
        stabilization_payload = {
            **asdict(stabilization_config),
            "reference_weight_threshold": reference_threshold,
        }
        stabilization_sources = {
            "cli": _sha256_file(Path(__file__)),
            "geometry_head": _sha256_file(
                REPO_ROOT / "src/evaluation/oviv2_geometry_head.py"
            ),
        }
        geometry_stabilization_protocol = {
            "config": stabilization_payload,
            "transferred_vertex_count": stabilization.transferred_vertex_count,
            "source_hashes": stabilization_sources,
            "algorithm_hash": _json_hash(
                {
                    "config": stabilization_payload,
                    "source_hashes": stabilization_sources,
                }
            ),
        }
    ground_truth = load_replica_ground_truth(
        args.gt_mesh,
        args.gt_info,
        class_to_id=class_to_id,
        aliases=aliases,
    )
    threshold = float(
        manifest.get("protocol", {}).get("geometry_primary_threshold_m", 0.05)
    )
    metrics = evaluate_replica_voxel_map(
        mesh,
        ground_truth,
        legacy_entity_info,
        valid_semantic_ids=valid_semantic_ids,
        instance_semantic_ids=instance_semantic_ids,
        min_instance_vertices=args.min_instance_vertices,
        distance_threshold_m=threshold,
    )
    legacy_instance = metrics["instance"]["class_agnostic"]

    config = InstanceHeadConfig(
        minimum_component_vertices=args.minimum_component_vertices,
        child_score_multiplier=args.child_score_multiplier,
        maximum_projection_distance_m=threshold,
        deduplication_iou_threshold=args.deduplication_iou_threshold,
        view_count_exponent=args.view_count_exponent,
        semantic_evidence_exponent=args.semantic_evidence_exponent,
    )
    hypotheses = build_instance_hypotheses(mesh, head_entity_info, config)
    auxiliary_config = None
    auxiliary_hypotheses = ()
    if auxiliary_snapshot is None:
        instance_head = evaluate_instance_hypotheses(
            mesh,
            hypotheses,
            ground_truth,
            instance_semantic_ids=instance_semantic_ids,
            min_instance_vertices=args.min_instance_vertices,
            config=config,
        )
    else:
        auxiliary_entities = [
            entity
            for entity in sorted(
                auxiliary_snapshot.registry.entities.values(),
                key=lambda value: value.entity_id,
            )
            if entity.lifecycle_state in {"active", "dormant"}
            and entity.semantic_id > 0
        ]
        auxiliary_posteriors = {
            entity.entity_id: entity.semantic_posterior.probabilities
            for entity in auxiliary_entities
        }
        auxiliary_mesh = derive_labeled_mesh(
            auxiliary_snapshot.geometry,
            auxiliary_snapshot.evidence,
            auxiliary_snapshot.ownership,
            weight_threshold=mesh_weight_threshold,
            entity_posteriors=auxiliary_posteriors,
            semantic_fusion=SemanticFusionConfig(entity_weight_scale=fusion_scale),
            valid_semantic_ids=valid_semantic_ids,
        )
        auxiliary_config = AuxiliaryInstanceConfig(
            score_weight=float(getattr(args, "auxiliary_score_weight", 1.0)),
            support_exponent=float(
                getattr(args, "auxiliary_support_exponent", 1.0)
            ),
            score_bias=float(getattr(args, "auxiliary_score_bias", 0.0)),
        )
        auxiliary_hypotheses = build_auxiliary_entity_hypotheses(
            auxiliary_mesh,
            {
                entity.entity_id: entity.semantic_id
                for entity in auxiliary_entities
            },
            auxiliary_config,
        )
        projected = project_instance_hypotheses(
            mesh,
            hypotheses,
            ground_truth.vertices_xyz,
            threshold,
        ) + project_instance_hypotheses(
            auxiliary_mesh,
            auxiliary_hypotheses,
            ground_truth.vertices_xyz,
            threshold,
        )
        instance_head = evaluate_projected_instance_hypotheses(
            projected,
            ground_truth,
            instance_semantic_ids=instance_semantic_ids,
            min_instance_vertices=args.min_instance_vertices,
            deduplication_iou_threshold=config.deduplication_iou_threshold,
        )
    metrics["ap25"] = instance_head["ap25"]
    metrics["ap50"] = instance_head["ap50"]
    metrics["instance"]["legacy_class_agnostic"] = legacy_instance
    metrics["instance"]["class_agnostic"] = instance_head

    config_payload = _normalized_config(config)
    source_hashes = {
        "cli": _sha256_file(Path(__file__)),
        "geometry_head": _sha256_file(
            REPO_ROOT / "src/evaluation/oviv2_geometry_head.py"
        ),
        "instance_head": _sha256_file(
            REPO_ROOT / "src" / "evaluation" / "oviv2_instance_head.py"
        ),
        "replica_evaluator": _sha256_file(
            REPO_ROOT / "scripts/evaluation/evaluate_oviv2_replica.py"
        ),
    }
    auxiliary_protocol = None
    algorithm_payload: dict[str, Any] = {
        "config": config_payload,
        "source_hashes": source_hashes,
    }
    if auxiliary_snapshot is not None:
        auxiliary_source_hashes = {
            "cli": source_hashes["cli"],
            "instance_ensemble": _sha256_file(
                REPO_ROOT / "src/evaluation/oviv2_instance_ensemble.py"
            ),
        }
        auxiliary_payload = {
            "config": asdict(auxiliary_config),
            "run_manifest_sha256": _sha256_file(auxiliary_run_manifest_path),
            "snapshot_checksums": dict(sorted(auxiliary_snapshot.checksums.items())),
            "source_hashes": auxiliary_source_hashes,
            "upstream_algorithm_hash": auxiliary_run_manifest["algorithm_hash"],
        }
        auxiliary_protocol = {
            **auxiliary_payload,
            "algorithm_hash": _json_hash(auxiliary_payload),
        }
        algorithm_payload["auxiliary_instance_ensemble_algorithm_hash"] = (
            auxiliary_protocol["algorithm_hash"]
        )
    algorithm_hash = _json_hash(algorithm_payload)
    metrics["protocol"].update(
        {
            "manifest_id": manifest.get("manifest_id"),
            "scene_id": args.scene,
            "semantic_head": "fused_uncertainty",
            "semantic_fusion": {
                "mode": "uncertainty_linear",
                "entity_weight_scale": fusion_scale,
            },
            "vocabulary_hash": vocabulary_hash,
            "snapshot_revision": snapshot.metadata.revision,
            "snapshot_checksums": dict(sorted(snapshot.checksums.items())),
            "mesh_weight_threshold": mesh_weight_threshold,
            "headline_instance_protocol": "independent_gt_projection_hypotheses",
            "instance_head_config": config_payload,
            "instance_head_config_hash": _json_hash(config_payload),
            "instance_head_source_hashes": source_hashes,
            "instance_head_algorithm_hash": algorithm_hash,
        }
    )
    if semantic_replay_protocol is not None:
        metrics["protocol"]["semantic_replay"] = semantic_replay_protocol
    if geometry_stabilization_protocol is not None:
        metrics["protocol"]["geometry_semantic_stabilization"] = (
            geometry_stabilization_protocol
        )
    if auxiliary_protocol is not None:
        metrics["protocol"]["auxiliary_instance_ensemble"] = auxiliary_protocol
    audit = {
        "config": config_payload,
        "config_hash": metrics["protocol"]["instance_head_config_hash"],
        "algorithm_hash": algorithm_hash,
        "source_hashes": source_hashes,
        "snapshot_checksums": metrics["protocol"]["snapshot_checksums"],
        "hypothesis_counts": {
            "parent": sum(item.kind == "parent" for item in hypotheses),
            "child": sum(item.kind == "child" for item in hypotheses),
            "total": len(hypotheses),
        },
        "legacy": legacy_instance,
        "instance_head": instance_head,
    }
    if auxiliary_protocol is not None:
        audit["auxiliary_hypothesis_count"] = len(auxiliary_hypotheses)
        audit["auxiliary_instance_ensemble"] = auxiliary_protocol

    def write_output(directory: Path) -> None:
        _write_json(directory / "metrics.json", metrics)
        _write_json(directory / "instance_head_audit.json", audit)

    _publish_fresh_output(args.output, write_output)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--auxiliary-instance-snapshot", type=Path)
    parser.add_argument("--auxiliary-instance-run-manifest", type=Path)
    parser.add_argument("--auxiliary-score-weight", type=float, default=1.0)
    parser.add_argument("--auxiliary-support-exponent", type=float, default=1.0)
    parser.add_argument("--auxiliary-score-bias", type=float, default=0.0)
    parser.add_argument("--semantic-evidence", type=Path)
    parser.add_argument("--semantic-replay-manifest", type=Path)
    parser.add_argument("--gt-mesh", type=Path, required=True)
    parser.add_argument("--gt-info", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-instance-vertices", type=int, default=100)
    parser.add_argument("--minimum-component-vertices", type=int, default=200)
    parser.add_argument("--child-score-multiplier", type=float, default=2.0)
    parser.add_argument("--deduplication-iou-threshold", type=float, default=0.95)
    parser.add_argument("--view-count-exponent", type=float, default=1.0)
    parser.add_argument("--semantic-evidence-exponent", type=float, default=1.0)
    parser.add_argument("--fusion-entity-weight-scale", type=float, default=0.49)
    parser.add_argument("--mesh-weight-threshold", type=float, default=1.0)
    parser.add_argument("--semantic-reference-weight-threshold", type=float)
    parser.add_argument("--semantic-transfer-distance-m", type=float)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    metrics = evaluate(parse_args(argv))
    print(
        json.dumps(
            {
                name: metrics[name]
                for name in ("miou", "macc", "f_miou", "ap25", "ap50", "f5")
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
