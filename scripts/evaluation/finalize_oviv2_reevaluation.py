#!/usr/bin/env python3
"""Finalize OVIV2 metrics re-evaluated from immutable Replica snapshots."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.oviv2_result import (  # noqa: E402
    REPLICA8_SCENES,
    aggregate_oviv2_replica,
    evaluation_file_hashes,
    validate_scene_run_manifests,
    verify_byte_identical_evaluation_dirs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _assert_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"metric must be finite: {path}")


def validate_metric_transition(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    scene: str,
) -> dict[str, dict[str, float]]:
    _assert_finite(new)
    for name in ("semantic", "geometry", "projection", "miou", "macc", "f_miou", "f5"):
        if old.get(name) != new.get(name):
            raise ValueError(f"{scene}: non-instance metric changed: {name}")
    protocol = new.get("protocol", {})
    if protocol.get("headline_instance_protocol") != "class_agnostic":
        raise ValueError(f"{scene}: headline protocol must be class_agnostic")
    if protocol.get("semantic_instance_protocol") != "diagnostic_only":
        raise ValueError(f"{scene}: semantic instance metrics must be diagnostic_only")
    class_agnostic = new.get("instance", {}).get("class_agnostic", {})
    for name in ("ap25", "ap50"):
        if new.get(name) != class_agnostic.get(name):
            raise ValueError(f"{scene}: headline {name} is not bound to class_agnostic")
    old_values = {name: float(old[name]) for name in ("ap25", "ap50")}
    new_values = {name: float(new[name]) for name in ("ap25", "ap50")}
    return {
        "old": old_values,
        "new": new_values,
        "delta": {name: new_values[name] - old_values[name] for name in old_values},
    }


def _verify_source_artifacts(scene_root: Path, run: dict[str, Any]) -> None:
    for relative, expected in run.get("artifact_checksums", {}).items():
        path = scene_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source run artifact checksum mismatch: {scene_root.name}/{relative}")


def _repeat_evaluation(
    scene: str,
    source_scene: Path,
    scene_config: dict[str, Any],
    evaluation_root: Path,
) -> dict[str, str]:
    from scripts.evaluation.evaluate_oviv2_replica import evaluate

    repeat = evaluation_root / "repeated_evaluation" / scene
    if not repeat.exists():
        evaluate(
            argparse.Namespace(
                snapshot=source_scene / "final/oviv2_voxel_snapshot.npz",
                entity_info=source_scene / "final/oviv2_entities.jsonl",
                gt_mesh=_resolve(scene_config["gt_mesh"]),
                gt_info=_resolve(scene_config["gt_info"]),
                manifest=_resolve(scene_config["manifest"]),
                scene=scene,
                output=repeat,
                min_instance_vertices=int(scene_config.get("min_instance_vertices", 100)),
            )
        )
    return verify_byte_identical_evaluation_dirs(
        evaluation_root / scene / "evaluation",
        repeat,
    )


def finalize(
    *,
    source_batch_root: Path,
    source_result_path: Path,
    evaluation_root: Path,
    run_id: str,
) -> dict[str, Any]:
    source_batch_root = source_batch_root.resolve()
    source_result_path = source_result_path.resolve()
    evaluation_root = evaluation_root.resolve()
    source_result = _read_object(source_result_path, "source result")
    if source_result.get("status") != "VERIFIED" or source_result.get("method", {}).get("key") != "OVIV2":
        raise ValueError("source result must be a VERIFIED OVIV2 result")
    for record in source_result.get("raw_outputs", ()):
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(f"source result raw output hash mismatch: {path}")

    batch = _read_object(source_batch_root / "batch_manifest.json", "source batch")
    records = tuple(batch.get("scenes", ()))
    if tuple(record.get("scene") for record in records) != REPLICA8_SCENES:
        raise ValueError("source batch scene order does not match Replica-8")
    scene_runs = {}
    scene_metrics = {}
    transitions = {}
    repeat_hashes = {}
    raw_outputs = [
        {
            "kind": "source_result",
            "path": str(source_result_path),
            "sha256": _sha256(source_result_path),
        }
    ]
    for record in records:
        scene = str(record["scene"])
        source_scene = source_batch_root / scene
        run_path = source_scene / "run_manifest.json"
        if _sha256(run_path) != record.get("run_manifest_sha256"):
            raise ValueError(f"source batch run manifest hash mismatch: {scene}")
        run = _read_object(run_path, f"{scene} run manifest")
        _verify_source_artifacts(source_scene, run)
        scene_runs[scene] = run

        new_directory = evaluation_root / scene / "evaluation"
        evaluation_file_hashes(new_directory)
        new_metrics_path = new_directory / "metrics.json"
        old_metrics_path = source_scene / "evaluation/metrics.json"
        new_metrics = _read_object(new_metrics_path, f"{scene} new metrics")
        old_metrics = _read_object(old_metrics_path, f"{scene} old metrics")
        transitions[scene] = validate_metric_transition(old_metrics, new_metrics, scene=scene)
        scene_metrics[scene] = new_metrics

        status_path = evaluation_root / scene / "status.json"
        status = _read_object(status_path, f"{scene} reevaluation status")
        if status.get("exit_status") != 0 or not status.get("formal_result_eligible"):
            raise ValueError(f"reevaluation status is not eligible: {scene}")
        if status.get("output_metrics", {}).get("sha256") != _sha256(new_metrics_path):
            raise ValueError(f"reevaluation status metrics hash mismatch: {scene}")
        audit_path = evaluation_root / scene / "protocol_audit.json"
        audit = _read_object(audit_path, f"{scene} protocol audit")
        if audit.get("headline_compatible") is not True or audit.get("reasons"):
            raise ValueError(f"protocol audit is incompatible: {scene}")

        scene_config = _read_object(
            source_batch_root / "scene_configs" / f"{scene}.json",
            f"{scene} config",
        )
        repeat_hashes[scene] = _repeat_evaluation(
            scene,
            source_scene,
            scene_config,
            evaluation_root,
        )
        raw_outputs.extend(
            (
                {"kind": "source_run_manifest", "scene": scene, "path": str(run_path), "sha256": _sha256(run_path)},
                {"kind": "metrics", "scene": scene, "path": str(new_metrics_path), "sha256": _sha256(new_metrics_path)},
                {"kind": "protocol_audit", "scene": scene, "path": str(audit_path), "sha256": _sha256(audit_path)},
                {"kind": "reevaluation_status", "scene": scene, "path": str(status_path), "sha256": _sha256(status_path)},
            )
        )

    validate_scene_run_manifests(scene_runs)
    result = copy.deepcopy(source_result)
    result.update(
        {
            "status": "VERIFIED",
            "run_id": run_id,
            "adapter_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPOSITORY_ROOT,
                text=True,
            ).strip(),
            "command": [
                "python",
                "scripts/evaluation/finalize_oviv2_reevaluation.py",
                "--source-batch-root",
                str(source_batch_root),
                "--source-result",
                str(source_result_path),
                "--evaluation-root",
                str(evaluation_root),
                "--run-id",
                run_id,
            ],
            "raw_outputs": raw_outputs,
            "repeated_evaluation_sha256": repeat_hashes,
            "metrics": aggregate_oviv2_replica(scene_metrics),
            "protocol_transition": transitions,
            "protocol_corrections": [
                "Table 1 instance AP is class-agnostic and independent of semantic labels and accepted-view filtering."
            ],
        }
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-batch-root", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = finalize(
        source_batch_root=args.source_batch_root,
        source_result_path=args.source_result,
        evaluation_root=args.evaluation_root,
        run_id=args.run_id,
    )
    _write_json_atomic(args.output.resolve(), result)
    print(json.dumps({"output": str(args.output.resolve()), "run_id": result["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
