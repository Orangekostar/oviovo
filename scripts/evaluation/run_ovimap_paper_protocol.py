#!/usr/bin/env python3
"""Stage Replica-8 and run OVI-MAP's released evaluation commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.baselines.ovimap import (
    parse_instance_color_log,
    remap_instance_colors,
)


REPLICA8_SCENES = (
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
)
FEATURE_NAME = "inst_sem_siglip-l-16-384_200_incre_combine.pkl"
MESH_NAME = "instance_mesh_200.ply"
RELEASED_SCRIPTS = (
    "scripts/datasets/preprocess_gt_mesh.py",
    "scripts/utils/mesh_postprocess_utils.py",
    "scripts/eval_inst_seg.py",
    "scripts/eval_sem_seg.py",
)


@dataclass(frozen=True)
class RunnerConfig:
    evaluator_root: Path
    python: Path
    scene_sources: dict[str, Path]
    scene_color_logs: dict[str, Path]
    gt_meshes: dict[str, Path]
    gt_semantic_folder: Path
    gt_instance_folder: Path
    siglip_model: Path
    output: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} missing: {path}")


def _validate_scene_mapping(values: dict[str, Path], label: str) -> None:
    observed = set(values)
    expected = set(REPLICA8_SCENES)
    if observed != expected:
        raise ValueError(
            f"{label} must contain Replica-8; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def preflight(config: RunnerConfig) -> None:
    _require_file(config.python, "evaluation python")
    _require_directory(config.evaluator_root, "OVI-MAP evaluator root")
    _require_directory(config.siglip_model, "local SigLIP model")
    _require_directory(config.gt_semantic_folder, "Replica semantic-label folder")
    _require_directory(config.gt_instance_folder, "Replica instance-label folder")
    for relative in RELEASED_SCRIPTS:
        _require_file(config.evaluator_root / relative, f"released evaluator {relative}")
    _validate_scene_mapping(config.scene_sources, "scene_sources")
    _validate_scene_mapping(config.scene_color_logs, "scene_color_logs")
    _validate_scene_mapping(config.gt_meshes, "gt_meshes")
    for scene in REPLICA8_SCENES:
        source = config.scene_sources[scene]
        _require_directory(source, f"{scene} source folder")
        _require_file(source / MESH_NAME, f"{scene} {MESH_NAME}")
        _require_file(source / FEATURE_NAME, f"{scene} {FEATURE_NAME}")
        _require_file(config.scene_color_logs[scene], f"{scene} instance color log")
        _require_file(config.gt_meshes[scene], f"{scene} GT mesh")
        _require_file(
            config.gt_semantic_folder / f"semantic_labels_{scene}.txt",
            f"{scene} semantic labels",
        )
        _require_file(
            config.gt_instance_folder / f"instance_labels_{scene}.txt",
            f"{scene} instance labels",
        )


def build_command_plan(config: RunnerConfig) -> list[dict[str, Any]]:
    layout = config.output / "layout"
    dataset = config.output / "dataset" / "Replica"
    python = str(config.python)
    commands: list[dict[str, Any]] = []
    for scene in REPLICA8_SCENES:
        commands.append(
            {
                "name": f"preprocess_gt_{scene}",
                "argv": [
                    python,
                    "-m",
                    "scripts.datasets.preprocess_gt_mesh",
                    "--scene_num",
                    scene,
                    "--data_folder",
                    str(dataset),
                    "--result_folder",
                    str(layout),
                    "--gt_sem_folder",
                    str(config.gt_semantic_folder),
                    "--gt_inst_folder",
                    str(config.gt_instance_folder),
                ],
            }
        )
        commands.append(
            {
                "name": f"mesh_postprocess_{scene}",
                "argv": [
                    python,
                    "-m",
                    "scripts.utils.mesh_postprocess_utils",
                    "--scene_num",
                    scene,
                    "--result_folder",
                    str(layout),
                ],
            }
        )
    commands.extend(
        (
            {
                "name": "released_instance_diagnostics",
                "argv": [
                    python,
                    "-m",
                    "scripts.eval_inst_seg",
                    "--scene_num",
                    "all",
                    "--result_folder",
                    str(layout),
                ],
            },
            {
                "name": "released_semantic_evaluation",
                "argv": [
                    python,
                    "-m",
                    "scripts.eval_sem_seg",
                    "--result_folder",
                    str(layout),
                ],
            },
        )
    )
    return commands


def _symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve())


def _stage(config: RunnerConfig) -> dict[str, Any]:
    dataset_root = config.output / "dataset"
    (dataset_root / "Replica").mkdir(parents=True)
    sources: dict[str, Any] = {}
    for scene in REPLICA8_SCENES:
        cropformer = config.output / "layout" / scene / "cropformer_inst"
        cropformer.mkdir(parents=True)
        source = config.scene_sources[scene]
        _symlink(source / MESH_NAME, cropformer / MESH_NAME)
        with (source / FEATURE_NAME).open("rb") as handle:
            semantic_features = pickle.load(handle)
        colors = parse_instance_color_log(config.scene_color_logs[scene])
        stale_ids = sorted({int(value) for value in semantic_features} - set(colors))
        final_features = {
            int(instance_id): record
            for instance_id, record in semantic_features.items()
            if int(instance_id) in colors
        }
        remapped_features = remap_instance_colors(final_features, colors)
        staged_feature_path = cropformer / FEATURE_NAME
        with staged_feature_path.open("wb") as handle:
            pickle.dump(remapped_features, handle)
        (dataset_root / "Replica" / scene).mkdir(parents=True)
        _symlink(
            config.gt_meshes[scene],
            dataset_root / "Replica" / f"{scene}_mesh.ply",
        )
        sources[scene] = {
            "instance_mesh": {
                "path": str((source / MESH_NAME).resolve()),
                "sha256": _sha256(source / MESH_NAME),
            },
            "semantic_features": {
                "path": str((source / FEATURE_NAME).resolve()),
                "sha256": _sha256(source / FEATURE_NAME),
                "remapped_path": str(staged_feature_path.resolve()),
                "remapped_sha256": _sha256(staged_feature_path),
                "color_source": "method_output.LogInstanceColor",
                "dropped_stale_ids": stale_ids,
            },
            "instance_color_log": {
                "path": str(config.scene_color_logs[scene].resolve()),
                "sha256": _sha256(config.scene_color_logs[scene]),
            },
            "gt_mesh": {
                "path": str(config.gt_meshes[scene].resolve()),
                "sha256": _sha256(config.gt_meshes[scene]),
            },
        }
    return sources


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run(config: RunnerConfig, *, dry_run: bool = False) -> dict[str, Any]:
    preflight(config)
    if config.output.exists():
        raise FileExistsError(f"output already exists: {config.output}")
    config.output.mkdir(parents=True)
    sources = _stage(config)
    commands = build_command_plan(config)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "DRY_RUN" if dry_run else "RUNNING",
        "source_protocol": "released_ovimap_replica51",
        "scene_ids": list(REPLICA8_SCENES),
        "frame_count_per_scene": 200,
        "semantic_vocabulary": "Replica-51",
        "released_metric_contract": {
            "instance": [
                "mIoU",
                "wIoU",
                "mP@75",
                "mR@75",
                "mP@50",
                "mR@50",
                "mP@25",
                "mR@25",
            ],
            "semantic": ["mIoU", "mAcc"],
        },
        "paper_metric_availability": {
            "table_2_class_agnostic_ap": False,
            "table_3_semantic": True,
            "reason": (
                "The released eval_inst_seg reports mean precision/recall, not the "
                "paper's class-agnostic AP25/AP50/AP75."
            ),
        },
        "inputs": sources,
        "commands": [{**command, "status": "PLANNED"} for command in commands],
    }
    manifest_path = config.output / "run_manifest.json"
    _atomic_json(manifest_path, manifest)
    if dry_run:
        return manifest

    environment = os.environ.copy()
    environment.update(
        {
            "OVIMAP_SIGLIP_MODEL": str(config.siglip_model.resolve()),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": os.pathsep.join(
                value
                for value in (
                    str(config.evaluator_root.resolve()),
                    environment.get("PYTHONPATH", ""),
                )
                if value
            ),
        }
    )
    logs = config.output / "logs"
    logs.mkdir()
    records: list[dict[str, Any]] = []
    for command in commands:
        stdout_path = logs / f"{command['name']}.stdout.txt"
        stderr_path = logs / f"{command['name']}.stderr.txt"
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                command["argv"],
                cwd=config.evaluator_root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
                text=True,
            )
        record = {
            **command,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "exit_status": completed.returncode,
            "elapsed_s": time.perf_counter() - started,
            "stdout": str(stdout_path.resolve()),
            "stderr": str(stderr_path.resolve()),
        }
        records.append(record)
        manifest["commands"] = records + [
            {**pending, "status": "PLANNED"} for pending in commands[len(records) :]
        ]
        if completed.returncode != 0:
            manifest["status"] = "FAILED"
            _atomic_json(manifest_path, manifest)
            return manifest
        _atomic_json(manifest_path, manifest)
    manifest["status"] = "COMPLETE_RELEASED_EVALUATION"
    manifest["commands"] = records
    _atomic_json(manifest_path, manifest)
    return manifest


def _parse_paths(values: list[str], label: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid {label}: {value}")
        scene, raw_path = value.split("=", 1)
        if not scene or scene in parsed:
            raise ValueError(f"duplicate or empty scene in {label}: {scene}")
        parsed[scene] = Path(raw_path)
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--scene-source", action="append", default=[], required=True)
    parser.add_argument("--scene-color-log", action="append", default=[], required=True)
    parser.add_argument("--gt-mesh", action="append", default=[], required=True)
    parser.add_argument("--gt-semantic-folder", type=Path, required=True)
    parser.add_argument("--gt-instance-folder", type=Path, required=True)
    parser.add_argument("--siglip-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        config = RunnerConfig(
            evaluator_root=args.evaluator_root,
            python=args.python,
            scene_sources=_parse_paths(args.scene_source, "--scene-source"),
            scene_color_logs=_parse_paths(args.scene_color_log, "--scene-color-log"),
            gt_meshes=_parse_paths(args.gt_mesh, "--gt-mesh"),
            gt_semantic_folder=args.gt_semantic_folder,
            gt_instance_folder=args.gt_instance_folder,
            siglip_model=args.siglip_model,
            output=args.output,
        )
        manifest = run(config, dry_run=args.dry_run)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    return 0 if manifest["status"] in {"DRY_RUN", "COMPLETE_RELEASED_EVALUATION"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
