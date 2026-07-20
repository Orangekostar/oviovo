#!/usr/bin/env python3
"""Verify OVIV2 ScanNet200-5 runs, repeat evaluation, and publish a result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_scannet200 import evaluate  # noqa: E402
from scripts.evaluation.finalize_oviv2_replica_result import (  # noqa: E402
    DENSE_PROVENANCE_FIELDS,
    _atomic_json,
    _dense_cache_prefix,
    _evaluation_contract,
    _json_hash,
    _load_json,
    _load_verified_batch_config,
    _resolve_repo_path,
    _sha256,
    _verify_run_artifacts,
    _verify_scene_config_binding,
)
from src.evaluation.oviv2_scannet200_result import (  # noqa: E402
    SCANNET200_5_SCENES,
    aggregate_oviv2_scannet200,
    validate_scene_run_manifests,
    verify_byte_identical_evaluation_dirs,
)


_CLEAN_DIRTY_DIGEST = hashlib.sha256(b"").hexdigest()


def _token_bindings() -> list[dict[str, Any]]:
    values = (
        (
            "T1_OVIV2_SCANNET5_MIOU",
            "/metrics/scannet200_5_heldout/semantic/miou",
        ),
        (
            "T1_OVIV2_SCANNET5_AP25",
            "/metrics/scannet200_5_heldout/instance/ap25",
        ),
        (
            "T1_OVIV2_SCANNET5_AP50",
            "/metrics/scannet200_5_heldout/instance/ap50",
        ),
        (
            "T1_OVIV2_SCANNET5_F5",
            "/metrics/scannet200_5_heldout/geometry/f5",
        ),
    )
    return [
        {"token": token, "json_pointer": pointer, "precision": 3}
        for token, pointer in values
    ]


def _validate_dense_provenance(
    scene_runs: dict[str, dict[str, Any]],
    expected: dict[str, Any],
) -> str:
    if not isinstance(expected, dict) or set(expected) != set(DENSE_PROVENANCE_FIELDS):
        raise ValueError("batch dense_provenance does not match the frozen field contract")
    by_scene: dict[str, dict[str, Any]] = {}
    for scene in SCANNET200_5_SCENES:
        dense = scene_runs[scene].get("dense_semantics", {})
        producer = dense.get("producer_provenance", {})
        consumed = dense.get("provenance", {})
        if dense.get("mode") != "cached_probabilities":
            raise ValueError(f"missing dense producer provenance: {scene}")
        if any(
            field not in producer or consumed.get(field) != producer[field]
            for field in DENSE_PROVENANCE_FIELDS
        ):
            raise ValueError(f"incomplete dense producer provenance: {scene}")
        consumed_prefix = _dense_cache_prefix(dense.get("cache_files_sha256", {}))
        producer_prefix = _dense_cache_prefix(
            dense.get("producer_cache_files_sha256", {})
        )
        if (
            dense.get("cache_prefix_sha256") != consumed_prefix
            or consumed.get("cache_prefix_sha256") != consumed_prefix
            or dense.get("producer_cache_prefix_sha256") != producer_prefix
            or producer.get("cache_prefix_sha256") != producer_prefix
        ):
            raise ValueError(f"dense cache prefix mismatch: {scene}")
        by_scene[scene] = {
            field: producer[field] for field in DENSE_PROVENANCE_FIELDS
        }
    encoded = {
        json.dumps(value, separators=(",", ":"), sort_keys=True)
        for value in by_scene.values()
    }
    if len(encoded) != 1:
        raise ValueError("dense producer provenance differs across scenes")
    common = by_scene[SCANNET200_5_SCENES[0]]
    if common != expected:
        raise ValueError("dense producer provenance does not match frozen batch config")
    return _json_hash(common)


def _repeat_scene_evaluation(
    scene: str,
    scene_root: Path,
    scene_config: dict[str, Any],
    repeat_root: Path,
    semantic_head: str,
    fusion_entity_weight_scale: float,
) -> dict[str, str]:
    repeat_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{scene}-fresh-",
        dir=repeat_root.parent,
    ) as temporary:
        repeated = Path(temporary) / "evaluation"
        evaluate(
            argparse.Namespace(
                snapshot=scene_root / "final" / "oviv2_voxel_snapshot.npz",
                entity_info=scene_root / "final" / "oviv2_entities.jsonl",
                gt_ply=_resolve_repo_path(scene_config["gt_mesh"]),
                metadata=_resolve_repo_path(scene_config["gt_info"]),
                manifest=_resolve_repo_path(scene_config["manifest"]),
                scene=scene,
                output=repeated,
                min_instance_points=int(
                    scene_config.get("min_instance_vertices", 100)
                ),
                distance_threshold_m=0.05,
                semantic_head=semantic_head,
                fusion_entity_weight_scale=fusion_entity_weight_scale,
            )
        )
        evaluation_dir = {
            "owner_authoritative": (
                "evaluation_owner"
                if scene_config.get("dense_semantic_mode") == "cached_probabilities"
                else "evaluation"
            ),
            "dense_only": "evaluation_dense",
            "fused_uncertainty": "evaluation_fused",
        }[semantic_head]
        return verify_byte_identical_evaluation_dirs(
            scene_root / evaluation_dir,
            repeated,
        )


def finalize(batch_root: str | Path, output: str | Path) -> dict[str, Any]:
    batch_root = Path(batch_root).resolve()
    batch = _load_json(batch_root / "batch_manifest.json")
    frame_counts = batch.get("frame_counts")
    if (
        batch.get("method") != "OVIV2"
        or batch.get("dataset") != "ScanNet200"
        or tuple(batch.get("scene_ids", ())) != SCANNET200_5_SCENES
        or not isinstance(frame_counts, dict)
    ):
        raise ValueError("batch manifest is not the exact OVIV2 ScanNet200-5 contract")
    batch_config = _load_verified_batch_config(batch)
    if batch.get("semantic_head") != batch_config.get("semantic_head"):
        raise ValueError("batch semantic_head does not match frozen config")

    records = batch.get("scenes", ())
    if tuple(record.get("scene") for record in records) != SCANNET200_5_SCENES:
        raise ValueError("batch scene records do not match frozen ScanNet200 order")
    scene_runs: dict[str, dict[str, Any]] = {}
    scene_metrics: dict[str, dict[str, Any]] = {}
    scene_configs: dict[str, dict[str, Any]] = {}
    repeat_hashes: dict[str, dict[str, str]] = {}
    frontend_provenance: dict[str, dict[str, str]] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    hardware: dict[str, dict[str, Any]] = {}
    raw_outputs: list[dict[str, str]] = []
    repository_commits: set[str] = set()

    for record in records:
        scene = str(record["scene"])
        scene_root = batch_root / scene
        run_path = scene_root / "run_manifest.json"
        if _sha256(run_path) != record.get("run_manifest_sha256"):
            raise ValueError(f"batch run manifest hash mismatch: {scene}")
        run = _load_json(run_path)
        _verify_run_artifacts(scene_root, run)
        if run.get("dirty_state_digest") != _CLEAN_DIRTY_DIGEST:
            raise ValueError(f"scene run repository was dirty: {scene}")
        scene_runs[scene] = run
        repository_commits.add(str(run.get("repository_commit", "")))
        hardware[scene] = dict(run.get("hardware", {}))
        timing_path = scene_root / "timing.json"
        runtimes[scene] = _load_json(timing_path)
        scene_config = _verify_scene_config_binding(
            batch_root / "scene_configs" / f"{scene}.json",
            record,
            run,
        )
        scene_configs[scene] = scene_config
        semantic_head, evaluation_dir, fusion_scale = _evaluation_contract(
            batch_config,
            scene_config,
        )
        metrics_path = scene_root / evaluation_dir / "metrics.json"
        scene_metrics[scene] = _load_json(metrics_path)
        frontend_path = (
            Path(scene_config["frontend_cache_dir"]) / "frontend_manifest.json"
        )
        if _sha256(frontend_path) != run.get("frontend_manifest_hash"):
            raise ValueError(f"frontend manifest hash mismatch: {scene}")
        frontend = _load_json(frontend_path)
        frontend_provenance[scene] = dict(frontend.get("provenance_sha256", {}))
        repeat_hashes[scene] = _repeat_scene_evaluation(
            scene,
            scene_root,
            scene_config,
            batch_root / "repeated_evaluation" / scene,
            semantic_head,
            fusion_scale,
        )
        raw_outputs.extend(
            (
                {
                    "kind": "run_manifest",
                    "scene": scene,
                    "path": str(run_path),
                    "sha256": _sha256(run_path),
                },
                {
                    "kind": "metrics",
                    "scene": scene,
                    "path": str(metrics_path),
                    "sha256": _sha256(metrics_path),
                },
            )
        )

    normalized_counts = {scene: int(frame_counts[scene]) for scene in SCANNET200_5_SCENES}
    contract = validate_scene_run_manifests(scene_runs, normalized_counts)
    if contract["algorithm_hash"] != batch.get("algorithm_hash"):
        raise ValueError("batch algorithm hash does not match scene runs")
    if contract["frontend_algorithm_hash"] != batch.get("frontend_algorithm_hash"):
        raise ValueError("batch frontend algorithm hash does not match scene runs")
    if len(repository_commits) != 1 or "" in repository_commits:
        raise ValueError("all ScanNet scene runs must use one repository commit")
    if len({json.dumps(value, sort_keys=True) for value in frontend_provenance.values()}) != 1:
        raise ValueError("frontend model provenance differs across scenes")
    contract["dense_provenance_hash"] = _validate_dense_provenance(
        scene_runs,
        batch_config.get("dense_provenance", {}),
    )

    metrics = aggregate_oviv2_scannet200(scene_metrics)
    first_scene = SCANNET200_5_SCENES[0]
    config_path = _resolve_repo_path(batch["config_path"])
    benchmark_path = _resolve_repo_path(scene_configs[first_scene]["manifest"])
    frontend_config = batch_config["frontend"]
    provenance = frontend_provenance[first_scene]
    weight_fields = (
        ("yolo_world", "yolo_model_path", "yolo_model"),
        ("yolo_clip", "yolo_clip_model_path", "yolo_clip_model"),
        ("mobile_sam", "mobile_sam_model_path", "mobile_sam_model"),
        ("open_clip", "clip_pretrained_path", "clip_model"),
    )
    weights = []
    for name, path_field, hash_field in weight_fields:
        path = Path(frontend_config[path_field]).resolve()
        if _sha256(path) != provenance.get(hash_field):
            raise ValueError(f"frontend weight provenance mismatch: {name}")
        weights.append(
            {"name": name, "path": str(path), "sha256": provenance[hash_field]}
        )

    repository_commit = repository_commits.pop()
    result = {
        "status": "VERIFIED",
        "run_id": "oviv2-scannet200-5-20260720-stage4-s10",
        "method": {"key": "OVIV2", "display_label": "OVIV2", "mode": "online"},
        "upstream_commit": repository_commit,
        "adapter_commit": repository_commit,
        "environment": hardware[first_scene],
        "command": [
            "python",
            "scripts/run_oviv2_scannet200_5.py",
            "--config",
            str(config_path),
            "--output",
            str(batch_root),
        ],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "weights": weights,
        "dataset": {
            "name": "ScanNet200",
            "splits": ["scannet200_5_heldout"],
            "manifest": {
                "path": str(benchmark_path),
                "sha256": _sha256(benchmark_path),
            },
        },
        "hardware": hardware,
        "seed": "deterministic-no-rng",
        "runtime": {
            "per_scene": {
                scene: runtimes[scene]["total_elapsed_sec"]
                for scene in SCANNET200_5_SCENES
            },
            "total_elapsed_sec": sum(
                float(runtimes[scene]["total_elapsed_sec"])
                for scene in SCANNET200_5_SCENES
            ),
        },
        "raw_outputs": raw_outputs,
        "repeated_evaluation_sha256": repeat_hashes,
        "protocol_deviations": [],
        "semantic_head": batch_config["semantic_head"],
        "provenance_contract": contract,
        "metrics": metrics,
        "token_bindings": _token_bindings(),
    }
    _atomic_json(Path(output).resolve(), result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = finalize(args.batch_root, args.output)
    print(json.dumps({"output": str(args.output), "tokens": len(result["token_bindings"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
