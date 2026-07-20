#!/usr/bin/env python3
"""Verify OVIV2 Replica-8 runs, repeat evaluation, and publish a result manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_replica import evaluate
from src.oviv2.semantic_fusion import SemanticFusionConfig
from src.evaluation.oviv2_result import (
    REPLICA8_SCENES,
    aggregate_oviv2_replica,
    validate_scene_run_manifests,
    verify_byte_identical_evaluation_dirs,
)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".json",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _verify_run_artifacts(scene_root: Path, run: dict[str, Any]) -> None:
    for relative, expected in run.get("artifact_checksums", {}).items():
        path = scene_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"run artifact checksum mismatch: {scene_root.name}/{relative}")


def _repeat_scene_evaluation(
    scene: str,
    scene_root: Path,
    scene_config: dict[str, Any],
    repeat_root: Path,
    semantic_head: str,
    fusion_entity_weight_scale: float,
) -> dict[str, str]:
    if not repeat_root.exists():
        evaluate(
            argparse.Namespace(
                snapshot=scene_root / "final" / "oviv2_voxel_snapshot.npz",
                entity_info=scene_root / "final" / "oviv2_entities.jsonl",
                gt_mesh=_resolve_repo_path(scene_config["gt_mesh"]),
                gt_info=_resolve_repo_path(scene_config["gt_info"]),
                manifest=_resolve_repo_path(scene_config["manifest"]),
                scene=scene,
                output=repeat_root,
                min_instance_vertices=int(scene_config.get("min_instance_vertices", 100)),
                semantic_head=semantic_head,
                fusion_entity_weight_scale=fusion_entity_weight_scale,
            )
        )
    _, evaluation_dir, _ = _evaluation_contract(
        {"semantic_head": semantic_head},
        scene_config,
    )
    return verify_byte_identical_evaluation_dirs(scene_root / evaluation_dir, repeat_root)


def _evaluation_contract(
    batch_config: dict[str, Any],
    scene_config: dict[str, Any],
) -> tuple[str, str, float]:
    semantic_head = batch_config.get("semantic_head", "owner_authoritative")
    dense_enabled = (
        scene_config.get("dense_semantic_mode", "disabled")
        == "cached_probabilities"
    )
    if semantic_head == "owner_authoritative":
        return semantic_head, "evaluation_owner" if dense_enabled else "evaluation", 0.5
    if semantic_head == "dense_only":
        if not dense_enabled:
            raise ValueError("dense_only semantic_head requires cached dense semantics")
        return semantic_head, "evaluation_dense", 0.5
    if semantic_head == "fused_uncertainty":
        if not dense_enabled:
            raise ValueError("fused semantic_head requires cached dense semantics")
        if scene_config.get("fusion_semantic_mode") != "uncertainty_linear":
            raise ValueError("fused semantic_head requires uncertainty_linear fusion")
        fusion = SemanticFusionConfig(
            entity_weight_scale=scene_config.get("fusion_entity_weight_scale", 0.5)
        )
        return semantic_head, "evaluation_fused", fusion.entity_weight_scale
    raise ValueError("semantic_head is unsupported")


def _token_bindings() -> list[dict[str, Any]]:
    values = (
        ("T1_OVIV2_REPLICA8_MIOU", "/metrics/replica_8_compat/semantic/miou"),
        ("T1_OVIV2_REPLICA8_MACC", "/metrics/replica_8_compat/semantic/macc"),
        ("T1_OVIV2_REPLICA8_FMIOU", "/metrics/replica_8_compat/semantic/f_miou"),
        ("T1_OVIV2_REPLICA7_MIOU", "/metrics/replica_7_heldout/semantic/miou"),
        ("T1_OVIV2_REPLICA8_AP25", "/metrics/replica_8_compat/instance/ap25"),
        ("T1_OVIV2_REPLICA8_AP50", "/metrics/replica_8_compat/instance/ap50"),
        ("T1_OVIV2_REPLICA8_F5", "/metrics/replica_8_compat/geometry/f5"),
        ("T1_OVIV2_REPLICA7_AP50", "/metrics/replica_7_heldout/instance/ap50"),
    )
    return [
        {"token": token, "json_pointer": pointer, "precision": 3}
        for token, pointer in values
    ]


def finalize(batch_root: str | Path, output: str | Path) -> dict[str, Any]:
    batch_root = Path(batch_root).resolve()
    batch = _load_json(batch_root / "batch_manifest.json")
    if batch.get("method") != "OVIV2" or tuple(batch.get("scene_ids", ())) != REPLICA8_SCENES:
        raise ValueError("batch manifest is not the exact OVIV2 Replica-8 contract")
    batch_config = _load_json(Path(batch["config_path"]))
    if batch.get("semantic_head", "owner_authoritative") != batch_config.get(
        "semantic_head",
        "owner_authoritative",
    ):
        raise ValueError("batch semantic_head does not match its frozen config")

    scene_runs: dict[str, dict[str, Any]] = {}
    scene_metrics: dict[str, dict[str, Any]] = {}
    scene_configs: dict[str, dict[str, Any]] = {}
    raw_outputs: list[dict[str, str]] = []
    repeat_hashes: dict[str, dict[str, str]] = {}
    frontend_provenance: dict[str, dict[str, str]] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    hardware: dict[str, dict[str, Any]] = {}
    repository_commits: set[str] = set()

    records = batch.get("scenes", ())
    if tuple(item.get("scene") for item in records) != REPLICA8_SCENES:
        raise ValueError("batch scene records do not match frozen Replica-8 order")
    for record in records:
        scene = str(record["scene"])
        scene_root = batch_root / scene
        run_path = scene_root / "run_manifest.json"
        if _sha256(run_path) != record.get("run_manifest_sha256"):
            raise ValueError(f"batch run manifest hash mismatch: {scene}")
        run = _load_json(run_path)
        _verify_run_artifacts(scene_root, run)
        scene_runs[scene] = run
        repository_commits.add(str(run.get("repository_commit", "")))
        hardware[scene] = dict(run.get("hardware", {}))
        timing_path = scene_root / "timing.json"
        runtimes[scene] = _load_json(timing_path)
        scene_config = _load_json(batch_root / "scene_configs" / f"{scene}.json")
        scene_configs[scene] = scene_config
        semantic_head, evaluation_dir, fusion_scale = _evaluation_contract(
            batch_config,
            scene_config,
        )
        metrics_path = scene_root / evaluation_dir / "metrics.json"
        scene_metrics[scene] = _load_json(metrics_path)
        frontend_manifest_path = Path(scene_config["frontend_cache_dir"]) / "frontend_manifest.json"
        if _sha256(frontend_manifest_path) != run.get("frontend_manifest_hash"):
            raise ValueError(f"frontend manifest hash mismatch: {scene}")
        frontend = _load_json(frontend_manifest_path)
        frontend_provenance[scene] = dict(frontend.get("provenance_sha256", {}))
        repeat_root = batch_root / "repeated_evaluation" / scene
        repeat_hashes[scene] = _repeat_scene_evaluation(
            scene,
            scene_root,
            scene_config,
            repeat_root,
            semantic_head,
            fusion_scale,
        )
        raw_outputs.extend(
            (
                {"kind": "run_manifest", "scene": scene, "path": str(run_path), "sha256": _sha256(run_path)},
                {"kind": "metrics", "scene": scene, "path": str(metrics_path), "sha256": _sha256(metrics_path)},
            )
        )

    contract = validate_scene_run_manifests(scene_runs)
    if contract["algorithm_hash"] != batch.get("algorithm_hash"):
        raise ValueError("batch mapping algorithm hash does not match scene runs")
    if contract["frontend_algorithm_hash"] != batch.get("frontend_algorithm_hash"):
        raise ValueError("batch frontend algorithm hash does not match scene runs")
    if len(repository_commits) != 1 or "" in repository_commits:
        raise ValueError("all OVIV2 scene runs must use one repository commit")
    if len({json.dumps(value, sort_keys=True) for value in frontend_provenance.values()}) != 1:
        raise ValueError("frontend model provenance differs across scenes")

    metrics = aggregate_oviv2_replica(scene_metrics)
    config_path = Path(batch["config_path"])
    benchmark_path = _resolve_repo_path(scene_configs["room0"]["manifest"])
    frontend_config = batch_config["frontend"]
    provenance = frontend_provenance["room0"]
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
        weights.append({"name": name, "path": str(path), "sha256": provenance[hash_field]})

    result = {
        "status": "VERIFIED",
        "run_id": (
            "oviv2-replica8-20260720-stage3-s10-200f"
            if batch_config.get("semantic_head") == "fused_uncertainty"
            else "oviv2-replica8-20260719-s10-200f"
        ),
        "method": {"key": "OVIV2", "display_label": "OVIV2", "mode": "online"},
        "upstream_commit": repository_commits.pop(),
        "adapter_commit": scene_runs["room0"]["repository_commit"],
        "environment": hardware["room0"],
        "command": ["python", "scripts/run_oviv2_replica8.py", "--config", str(config_path), "--output", str(batch_root)],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "weights": weights,
        "dataset": {
            "name": "Replica",
            "splits": ["replica_8_compat", "replica_7_heldout"],
            "manifest": {"path": str(benchmark_path), "sha256": _sha256(benchmark_path)},
        },
        "hardware": hardware,
        "seed": "deterministic-no-rng",
        "runtime": {
            "per_scene": {scene: runtimes[scene]["total_elapsed_sec"] for scene in REPLICA8_SCENES},
            "total_elapsed_sec": sum(float(runtimes[scene]["total_elapsed_sec"]) for scene in REPLICA8_SCENES),
        },
        "raw_outputs": raw_outputs,
        "repeated_evaluation_sha256": repeat_hashes,
        "protocol_deviations": [],
        "semantic_head": batch_config.get("semantic_head", "owner_authoritative"),
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
