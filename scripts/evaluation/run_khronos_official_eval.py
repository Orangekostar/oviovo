#!/usr/bin/env python3
"""Run and repeat-summarize the official Khronos TESSE-CD evaluator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
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
from scripts.evaluation.prepare_temporal_khronos_bridge import (
    validate_temporal_bridge_manifest,
)
from scripts.evaluation.run_temporal_khronos_bridge import validate_build_manifest


CANONICAL_TESSE_MANIFEST = (
    REPO_ROOT / "configs/evaluation/manifests/tesse_cd.json"
)
CANONICAL_TESSE_MANIFEST_SHA256 = (
    "be63826267109a67fe4109c1419eb02dbad99a9e3856d44ac8f848e32e00b907"
)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_identity(payload: Mapping[str, Any]) -> dict[str, str]:
    raw = payload.get("run_identity")
    if not isinstance(raw, Mapping):
        raise ValueError("Khronos run identity is required")
    run_id = raw.get("run_id")
    if type(run_id) is not str or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("Khronos run identity run_id is not canonical")
    config_sha256 = raw.get("config_sha256")
    if (
        type(config_sha256) is not str
        or len(config_sha256) != 64
        or any(char not in "0123456789abcdef" for char in config_sha256)
    ):
        raise ValueError("Khronos run identity config_sha256 is invalid")
    return {"run_id": run_id, "config_sha256": config_sha256}


def validate_tesse_manifest(path: Path, *, scene: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("canonical TESSE manifest must not be a symlink")
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
    if path.is_symlink():
        raise ValueError(f"{label} source must not be a symlink")
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


def validate_ground_truth_files(
    files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for name in ("dsg", "background_mesh", "changes"):
        entry = files.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"canonical TESSE manifest lacks ground truth {name}")
        path = Path(str(entry.get("path", "")))
        size = entry.get("size_bytes")
        if path.is_symlink():
            raise ValueError(f"ground-truth provenance rejects symlink: {path}")
        if (
            type(size) is not int
            or size < 0
            or not path.is_file()
            or path.stat().st_size != size
            or _sha256(path) != entry.get("sha256")
        ):
            raise ValueError(f"ground-truth provenance mismatch: {path}")
        selected[name] = entry
    return selected


def validate_khronos_run_status(
    path: Path, *, scene: str
) -> dict[str, Any]:
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
    source_paths: set[Path] = set()
    source_records: dict[Path, dict[str, Any]] = {}
    for index, entry in enumerate(sources):
        if not isinstance(entry, Mapping):
            raise ValueError("Khronos run source record is invalid")
        source_path = _validate_source_entry(
            entry, label=f"Khronos run source {index}"
        ).resolve()
        if source_path in source_paths:
            raise ValueError("Khronos run source paths must be unique")
        source_paths.add(source_path)
        source_records[source_path] = dict(entry)
    identity = _run_identity(payload)
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Khronos run config source is required")
    config_path = _validate_source_entry(config, label="Khronos run config").resolve()
    if source_records.get(config_path) != dict(config):
        raise ValueError("Khronos run config must be a declared hashed source")
    if config.get("sha256") != identity["config_sha256"]:
        raise ValueError("Khronos run identity config binding mismatch")
    run_root = path.parent.resolve()
    required = {
        (run_root / "map/final.4dmap").resolve(),
        (run_root / "map/map_timestamps.json").resolve(),
        (run_root / "map/experiment_log.txt").resolve(),
        (run_root / "bridge_input/bridge_manifest.json").resolve(),
        (run_root / "build_manifest.json").resolve(),
    }
    missing = required - source_paths
    if missing:
        raise ValueError(
            f"Khronos run required artifact is missing: {sorted(map(str, missing))}"
        )
    validate_temporal_bridge_manifest(run_root / "bridge_input/bridge_manifest.json")
    validate_build_manifest(run_root / "build_manifest.json")
    normalized = dict(payload)
    normalized["run_identity"] = identity
    return normalized


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
    *,
    results_dir: Path,
    scene: str,
    method: str,
    mode: str,
    output: Path,
    run_identity: Mapping[str, str],
) -> None:
    payload = {
        "status": "PASS",
        "dataset": "TESSE-CD",
        "scene": scene,
        "split": f"{scene}_test",
        "method": method,
        "mode": mode,
        "display_mode": "online",
        "aggregation": "upstream online 4D plotting aggregation",
        "run_identity": dict(run_identity),
        "metrics": summarize_khronos_official_metrics(results_dir),
        "sources": _metric_sources(results_dir, relative_to=output.parent),
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_partial_metrics(
    *,
    results_dir: Path,
    scene: str,
    method: str,
    mode: str,
    output: Path,
    run_identity: Mapping[str, str],
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
        "run_identity": dict(run_identity),
        "metrics": {
            "state_count": partial["state_count"],
            **partial["metrics"],
        },
        "unavailable": partial["unavailable"],
        "sources": _metric_sources(
            results_dir, relative_to=output.parent, allow_missing=True
        ),
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _metric_sources(
    results_dir: Path, *, relative_to: Path, allow_missing: bool = False
) -> list[dict[str, Any]]:
    sources = []
    for name in ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv"):
        path = results_dir / name
        logical_path = os.path.relpath(path, start=relative_to)
        if not path.is_file():
            if allow_missing:
                sources.append({"path": logical_path, "status": "MISSING"})
                continue
            raise ValueError(f"Khronos result file is missing: {path}")
        sources.append(
            {
                "path": logical_path,
                "sha256": _sha256(path),
                "byte_count": path.stat().st_size,
            }
        )
    return sources


def _validate_metric_run_layout(
    *,
    results_dir: Path,
    metrics_path: Path,
    repeat_path: Path,
    run_status_path: Path,
) -> Path:
    status_path = Path(os.path.abspath(run_status_path))
    if (
        status_path.name != "run_status.json"
        or status_path.is_symlink()
        or not status_path.is_file()
        or status_path.resolve(strict=True) != status_path
    ):
        raise ValueError(
            "official metrics inputs and outputs must share the same run root"
        )
    run_root = status_path.parent
    expected = {
        Path(os.path.abspath(results_dir)): run_root / "map/results",
        Path(os.path.abspath(metrics_path)): run_root
        / "evaluation/official_metrics.json",
        Path(os.path.abspath(repeat_path)): run_root
        / "evaluation/official_metrics.repeat.json",
    }
    if any(actual != required for actual, required in expected.items()):
        raise ValueError(
            "official metrics inputs and outputs must share the same run root"
        )
    results = run_root / "map/results"
    evaluation = run_root / "evaluation"
    if (
        results.is_symlink()
        or not results.is_dir()
        or results.resolve(strict=True) != results
        or evaluation.is_symlink()
        or not evaluation.is_dir()
        or evaluation.resolve(strict=True) != evaluation
    ):
        raise ValueError(
            "official metrics inputs and outputs must share the same run root"
        )
    return status_path


def write_repeated_metrics(
    *,
    results_dir: Path,
    scene: str,
    method: str,
    mode: str,
    metrics_path: Path,
    repeat_path: Path,
    run_status_path: Path,
) -> dict[str, Any]:
    if method != "OVIV2" or mode != "causal_checkpoints":
        raise ValueError("official metrics require OVIV2 causal identity")
    status_path = _validate_metric_run_layout(
        results_dir=results_dir,
        metrics_path=metrics_path,
        repeat_path=repeat_path,
        run_status_path=run_status_path,
    )
    run_status = validate_khronos_run_status(status_path, scene=scene)
    run_identity = dict(run_status["run_identity"])
    if (
        run_status.get("method") != method
        or run_status.get("mode") != mode
    ):
        raise ValueError("official metrics do not match validated run status")
    for output in (metrics_path, repeat_path):
        if os.path.lexists(output):
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
                run_identity=run_identity,
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
            run_identity=run_identity,
        )
        _write_partial_metrics(
            results_dir=results_dir,
            scene=scene,
            method=method,
            mode=mode,
            output=repeat_path,
            run_identity=run_identity,
        )
        if metrics_path.read_bytes() != repeat_path.read_bytes():
            raise RuntimeError("partial Khronos metric repeat is not byte-identical")
        return {
            "status": "UNAVAILABLE",
            "reason": errors[0],
            "repeat_reason": errors[1],
            "sources": _metric_sources(
                results_dir,
                relative_to=metrics_path.parent,
                allow_missing=True,
            ),
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
    files = validate_ground_truth_files(sequence["ground_truth"]["files"])

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

    validate_tesse_manifest(args.manifest, scene=args.scene)
    validate_ground_truth_files(sequence["ground_truth"]["files"])

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
        run_status_path=status_path,
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
