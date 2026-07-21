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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_replica import (  # noqa: E402
    NON_INSTANCE_CLASSES,
    _load_manifest,
    _normalize_label,
    _normalized_aliases,
    _verify_scene_inputs,
    _write_json,
    load_replica_ground_truth,
)
from src.evaluation.oviv2_instance_head import (  # noqa: E402
    InstanceHeadConfig,
    build_instance_hypotheses,
    evaluate_instance_hypotheses,
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
    structural_semantic_ids = frozenset(
        semantic_id
        for label, semantic_id in class_to_id.items()
        if label in NON_INSTANCE_CLASSES
    )
    instance_semantic_ids = valid_semantic_ids - set(structural_semantic_ids)

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
    mesh = derive_labeled_mesh(
        snapshot.geometry,
        snapshot.evidence,
        snapshot.ownership,
        entity_posteriors=entity_posteriors,
        semantic_fusion=SemanticFusionConfig(entity_weight_scale=fusion_scale),
        valid_semantic_ids=valid_semantic_ids,
    )
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
    instance_head = evaluate_instance_hypotheses(
        mesh,
        hypotheses,
        ground_truth,
        instance_semantic_ids=instance_semantic_ids,
        min_instance_vertices=args.min_instance_vertices,
        config=config,
    )
    metrics["ap25"] = instance_head["ap25"]
    metrics["ap50"] = instance_head["ap50"]
    metrics["instance"]["legacy_class_agnostic"] = legacy_instance
    metrics["instance"]["class_agnostic"] = instance_head

    config_payload = _normalized_config(config)
    vocabulary_payload = {
        "classes": classes,
        "aliases": dict(sorted(aliases.items())),
    }
    source_hashes = {
        "cli": _sha256_file(Path(__file__)),
        "instance_head": _sha256_file(
            REPO_ROOT / "src" / "evaluation" / "oviv2_instance_head.py"
        ),
    }
    algorithm_hash = _json_hash(
        {"config": config_payload, "source_hashes": source_hashes}
    )
    metrics["protocol"].update(
        {
            "manifest_id": manifest.get("manifest_id"),
            "scene_id": args.scene,
            "semantic_head": "fused_uncertainty",
            "semantic_fusion": {
                "mode": "uncertainty_linear",
                "entity_weight_scale": fusion_scale,
            },
            "vocabulary_hash": _json_hash(vocabulary_payload),
            "snapshot_revision": snapshot.metadata.revision,
            "snapshot_checksums": dict(sorted(snapshot.checksums.items())),
            "headline_instance_protocol": "independent_gt_projection_hypotheses",
            "instance_head_config": config_payload,
            "instance_head_config_hash": _json_hash(config_payload),
            "instance_head_source_hashes": source_hashes,
            "instance_head_algorithm_hash": algorithm_hash,
        }
    )
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

    def write_output(directory: Path) -> None:
        _write_json(directory / "metrics.json", metrics)
        _write_json(directory / "instance_head_audit.json", audit)

    _publish_fresh_output(args.output, write_output)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
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
