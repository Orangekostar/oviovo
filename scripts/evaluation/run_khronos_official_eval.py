#!/usr/bin/env python3
"""Run and repeat-summarize the official Khronos TESSE-CD evaluator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.baselines.tesse_cd import (
    summarize_khronos_official_metrics,
    summarize_khronos_official_metrics_partial,
)
from src.evaluation.json_contracts import loads_strict


CANONICAL_TESSE_MANIFEST = (
    REPO_ROOT / "configs/evaluation/manifests/tesse_cd.json"
)
CANONICAL_TESSE_MANIFEST_SHA256 = (
    "be63826267109a67fe4109c1419eb02dbad99a9e3856d44ac8f848e32e00b907"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tesse_manifest(path: Path, *, scene: str) -> dict[str, Any]:
    if not path.is_file() or _sha256(path) != CANONICAL_TESSE_MANIFEST_SHA256:
        raise ValueError("TESSE manifest does not match the canonical checked bytes")
    payload = loads_strict(path.read_text(encoding="utf-8"), label="TESSE manifest")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("manifest_id") != "tesse_cd_dynamic_v1"
        or payload.get("dataset") != "TESSE-CD"
        or scene not in payload.get("sequences", {})
    ):
        raise ValueError("invalid canonical TESSE manifest identity")
    return payload


def _validate_source_entry(entry: Mapping[str, Any], *, label: str) -> Path:
    if set(entry) != {"path", "sha256", "byte_count"}:
        raise ValueError(f"{label} source record fields are invalid")
    path = Path(str(entry.get("path", "")))
    byte_count = entry.get("byte_count")
    if (
        type(byte_count) is not int
        or byte_count < 0
        or not path.is_file()
        or path.stat().st_size != byte_count
        or _sha256(path) != entry.get("sha256")
    ):
        raise ValueError(f"{label} source provenance mismatch")
    return path


def validate_khronos_run_status(path: Path, *, scene: str) -> dict[str, Any]:
    payload = loads_strict(path.read_text(encoding="utf-8"), label="Khronos run status")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("status") != "PASS"
        or payload.get("dataset") != "TESSE-CD"
        or payload.get("scene") != scene
        or payload.get("method") != "OVIV2"
        or payload.get("mode") != "causal_checkpoints"
        or payload.get("bridge_mode") != "temporal_checkpoints"
    ):
        raise ValueError("invalid OVIV2 Khronos run identity")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Khronos run status requires hashed sources")
    for index, entry in enumerate(sources):
        if not isinstance(entry, Mapping):
            raise ValueError("Khronos run source record is invalid")
        _validate_source_entry(entry, label=f"Khronos run source {index}")
    return payload


def patch_evaluation_config(
    config: Mapping[str, Any], files: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    patched = copy.deepcopy(dict(config))
    patched["ground_truth_dsg_file"] = str(files["dsg"]["path"])
    patched["ground_truth_background_file"] = str(files["background_mesh"]["path"])
    patched["gt_changes_file"] = str(files["changes"]["path"])
    patched["evaluation"]["object_evaluation"]["changes_file"] = str(
        files["changes"]["path"]
    )
    return patched


def build_evaluation_command(
    *, workspace: Path, map_dir: Path, config: Path
) -> list[str]:
    return [
        str(workspace / "install/khronos_eval/lib/khronos_eval/evaluate_pipeline.sh"),
        str(map_dir),
        str(config),
        "true",
        "false",
    ]


def _overlay_command(
    *,
    workspace: Path,
    conda_executable: Path,
    conda_environment: str,
    command: list[str],
) -> list[str]:
    return [
        str(conda_executable),
        "run",
        "--no-capture-output",
        "-n",
        conda_environment,
        "bash",
        "-lc",
        'source "$1/install/setup.bash"; shift; exec "$@"',
        "bash",
        str(workspace),
        *command,
    ]


def _write_metrics(
    *, results_dir: Path, scene: str, method: str, mode: str, output: Path
) -> None:
    sources = []
    for name in ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv"):
        path = results_dir / name
        sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    payload = {
        "dataset": "TESSE-CD",
        "scene": scene,
        "split": f"{scene}_test",
        "method": method,
        "mode": mode,
        "display_mode": "online",
        "aggregation": "upstream online 4D plotting aggregation",
        "metrics": summarize_khronos_official_metrics(results_dir),
        "sources": sources,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_partial_metrics(
    *, results_dir: Path, scene: str, method: str, mode: str, output: Path
) -> None:
    partial = summarize_khronos_official_metrics_partial(results_dir)
    payload = {
        "status": "PARTIAL",
        "dataset": "TESSE-CD",
        "scene": scene,
        "split": f"{scene}_test",
        "method": method,
        "mode": mode,
        "display_mode": "online",
        "aggregation": "upstream online 4D plotting aggregation",
        "metrics": {
            "state_count": partial["state_count"],
            **partial["metrics"],
        },
        "unavailable": partial["unavailable"],
        "sources": _metric_sources(results_dir),
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _metric_sources(results_dir: Path) -> list[dict[str, Any]]:
    sources = []
    for name in ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv"):
        path = results_dir / name
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "byte_count": path.stat().st_size,
            }
        )
    return sources


def write_repeated_metrics(
    *,
    results_dir: Path,
    scene: str,
    method: str,
    mode: str,
    metrics_path: Path,
    repeat_path: Path,
) -> dict[str, Any]:
    if method != "OVIV2" or mode != "causal_checkpoints":
        raise ValueError("official metrics require OVIV2 causal identity")
    for output in (metrics_path, repeat_path):
        if output.exists():
            raise FileExistsError(f"official metric output already exists: {output}")
    errors: list[str] = []
    for output in (metrics_path, repeat_path):
        try:
            _write_metrics(
                results_dir=results_dir,
                scene=scene,
                method=method,
                mode=mode,
                output=output,
            )
        except ValueError as error:
            errors.append(str(error))
    if errors:
        if len(errors) != 2 or errors[0] != errors[1]:
            raise RuntimeError("official Khronos metric repeat failed inconsistently")
        _write_partial_metrics(
            results_dir=results_dir,
            scene=scene,
            method=method,
            mode=mode,
            output=metrics_path,
        )
        _write_partial_metrics(
            results_dir=results_dir,
            scene=scene,
            method=method,
            mode=mode,
            output=repeat_path,
        )
        if metrics_path.read_bytes() != repeat_path.read_bytes():
            raise RuntimeError("partial Khronos metric repeat is not byte-identical")
        return {
            "status": "UNAVAILABLE",
            "reason": errors[0],
            "repeat_reason": errors[1],
            "sources": _metric_sources(results_dir),
            "official_metrics_partial": {
                "path": str(metrics_path.resolve()),
                "sha256": _sha256(metrics_path),
            },
            "official_metrics_partial_repeat": {
                "path": str(repeat_path.resolve()),
                "sha256": _sha256(repeat_path),
            },
        }
    if metrics_path.read_bytes() != repeat_path.read_bytes():
        raise RuntimeError("official Khronos metric repeat is not byte-identical")
    return {
        "status": "PASS",
        "official_metrics": {
            "path": str(metrics_path.resolve()),
            "sha256": _sha256(metrics_path),
        },
        "official_metrics_repeat": {
            "path": str(repeat_path.resolve()),
            "sha256": _sha256(repeat_path),
        },
    }


def run(args: argparse.Namespace) -> Path:
    evaluation_dir = args.run_root / "evaluation"
    if os.path.lexists(evaluation_dir):
        raise FileExistsError(
            f"evaluation output already exists: {evaluation_dir}"
        )
    if args.method != "OVIV2" or args.mode != "causal_checkpoints":
        raise ValueError("official evaluation requires OVIV2 causal identity")
    status_path = args.run_root / "run_status.json"
    validate_khronos_run_status(status_path, scene=args.scene)

    manifest = validate_tesse_manifest(args.manifest, scene=args.scene)
    sequence = manifest["sequences"][args.scene]
    files = sequence["ground_truth"]["files"]
    for entry in files.values():
        path = Path(entry["path"])
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise ValueError(f"ground-truth provenance mismatch: {path}")

    base_path = (
        args.workspace
        / "src/khronos/khronos_eval/config/pipeline"
        / f"{args.scene}.yaml"
    )
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config = patch_evaluation_config(base, files)
    evaluation_dir.mkdir(parents=True)
    config_path = evaluation_dir / f"{args.scene}.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    inner = build_evaluation_command(
        workspace=args.workspace,
        map_dir=args.run_root / "map",
        config=config_path,
    )
    process_time_path = evaluation_dir / "evaluate.time.log"
    command = _overlay_command(
        workspace=args.workspace,
        conda_executable=args.conda_executable,
        conda_environment=args.conda_environment,
        command=["/usr/bin/time", "-v", "-o", str(process_time_path), *inner],
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=args.workspace,
        env=environment,
        capture_output=True,
        text=True,
        timeout=args.timeout_s,
        check=False,
    )
    wall_s = time.perf_counter() - started
    log_path = evaluation_dir / "evaluate.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    evaluation_status: dict[str, Any] = {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "scene": args.scene,
        "method": args.method,
        "mode": args.mode,
        "display_mode": "online",
        "command": command,
        "exit_status": completed.returncode,
        "wall_s": wall_s,
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _sha256(config_path),
        },
        "log": {"path": str(log_path.resolve()), "sha256": _sha256(log_path)},
        "process_time": {
            "path": str(process_time_path.resolve()),
            "sha256": _sha256(process_time_path),
        },
    }
    output = evaluation_dir / "evaluation_status.json"
    if completed.returncode != 0:
        output.write_text(
            json.dumps(evaluation_status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"official Khronos evaluator failed: {completed.returncode}")

    metrics_path = evaluation_dir / "official_metrics.json"
    repeat_path = evaluation_dir / "official_metrics.repeat.json"
    results_dir = args.run_root / "map/results"
    metric_summary = write_repeated_metrics(
        results_dir=results_dir,
        scene=args.scene,
        method=args.method,
        mode=args.mode,
        metrics_path=metrics_path,
        repeat_path=repeat_path,
    )
    if metric_summary["status"] == "PASS":
        evaluation_status.update(
            official_metrics=metric_summary["official_metrics"],
            official_metrics_repeat=metric_summary["official_metrics_repeat"],
        )
    else:
        evaluation_status["status"] = "UNAVAILABLE"
        evaluation_status["official_metrics_unavailable"] = metric_summary
    output.write_text(
        json.dumps(evaluation_status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scene", choices=("apartment", "office"), required=True)
    parser.add_argument("--method", choices=("OVIV2",), default="OVIV2")
    parser.add_argument(
        "--mode",
        choices=("causal_checkpoints",),
        default="causal_checkpoints",
    )
    parser.add_argument(
        "--conda-executable", type=Path, default=Path("/home/ww/miniconda3/bin/conda")
    )
    parser.add_argument("--conda-environment", default="oviovo-khronos-jazzy")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
