#!/usr/bin/env python3
"""Generate and verify frozen YOLO-World+MobileSAM caches for Replica-8."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
STRUCTURE_CLASSES = {"wall", "floor", "ceiling"}


@dataclass(frozen=True)
class FrontendCommand:
    scene: str
    gpu_id: int
    argv: tuple[str, ...]
    cache_dir: Path
    log_dir: Path
    algorithm_hash: str


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _algorithm_descriptor(frontend: dict[str, Any]) -> dict[str, Any]:
    return {
        key: frontend[key]
        for key in sorted(frontend)
        if key not in {"gsa_variant_template", "exp_suffix_template"}
    }


def build_commands(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    gpu_ids: tuple[int, ...],
    scenes: Iterable[str] | None = None,
) -> list[FrontendCommand]:
    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("at least one non-negative GPU ID is required")
    manifest_scenes = [str(item["scene"]) for item in manifest.get("scenes", ())]
    expected_scenes = ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")
    if tuple(manifest_scenes) != expected_scenes:
        raise ValueError("Replica manifest must contain the frozen eight scenes in order")
    requested = set(scenes or manifest_scenes)
    if not requested <= set(manifest_scenes):
        raise ValueError(f"unknown Replica scenes: {', '.join(sorted(requested - set(manifest_scenes)))}")
    frontend = dict(config["frontend"])
    algorithm_hash = _json_hash(_algorithm_descriptor(frontend))
    view_root = Path(config["view_root"]).expanduser().resolve()
    logs_root = Path(config["frontend_logs_root"])
    if not logs_root.is_absolute():
        logs_root = (REPO_ROOT / logs_root).resolve()
    frame_count = int(manifest["frame_selection"]["sampled_frames_per_scene"])

    commands: list[FrontendCommand] = []
    for index, scene in enumerate(manifest_scenes):
        if scene not in requested:
            continue
        view_scene = f"{scene}{config['view_suffix']}"
        variant = str(frontend["gsa_variant_template"]).format(scene=scene)
        suffix = str(frontend["exp_suffix_template"]).format(scene=scene)
        log_dir = logs_root / scene
        argv = (
            str(frontend["python"]),
            str(frontend["script"]),
            f"--config-name={frontend['hydra_config_name']}",
            f"dataset_root={view_root}",
            f"dataset_config={frontend['dataset_config']}",
            f"scene_id={view_scene}",
            "start=0",
            f"end={frame_count}",
            "stride=1",
            f"desired_height={int(frontend['desired_height'])}",
            f"desired_width={int(frontend['desired_width'])}",
            f"classes_file={frontend['classes_file']}",
            f"gsa_variant={variant}",
            f"exp_suffix={suffix}",
            f"device={frontend['device']}",
            "save_video=false",
            f"yolo_model_path={frontend['yolo_model_path']}",
            f"mobile_sam_model_path={frontend['mobile_sam_model_path']}",
            f"clip_model_card={frontend['clip_model_card']}",
            f"clip_pretrained_path={frontend['clip_pretrained_path']}",
            f"hydra.run.dir={log_dir / 'hydra'}",
            "hydra.job.chdir=false",
        )
        commands.append(
            FrontendCommand(
                scene=scene,
                gpu_id=gpu_ids[index % len(gpu_ids)],
                argv=argv,
                cache_dir=view_root / view_scene / f"gsa_detections_{variant}",
                log_dir=log_dir,
                algorithm_hash=algorithm_hash,
            )
        )
    return commands


def validate_frontend_cache(
    command: FrontendCommand,
    config: dict[str, Any],
    *,
    frame_count: int,
) -> dict[str, Any]:
    frontend = config["frontend"]
    classes_file = Path(frontend["classes_file"])
    expected_classes = [
        line.strip()
        for line in classes_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.strip().lower() not in STRUCTURE_CLASSES
    ]
    view_scene = command.cache_dir.parent
    with Image.open(view_scene / "results" / "frame000000.jpg") as image:
        width, height = image.size
    file_hashes: dict[str, str] = {}
    for frame_id in range(frame_count):
        path = command.cache_dir / f"frame{frame_id:06d}.pkl.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"frontend cache payload must be a dictionary: {path}")
        masks = np.asarray(payload.get("mask"))
        boxes = np.asarray(payload.get("xyxy"))
        confidences = np.asarray(payload.get("confidence"))
        class_ids = np.asarray(payload.get("class_id"))
        classes = [str(value) for value in payload.get("classes", ())]
        count = masks.shape[0] if masks.ndim == 3 else -1
        if masks.shape != (count, height, width) or boxes.shape != (count, 4):
            raise ValueError(f"frontend cache mask/box shape mismatch: {path}")
        if confidences.shape != (count,) or class_ids.shape != (count,):
            raise ValueError(f"frontend cache vector length mismatch: {path}")
        if classes != expected_classes:
            raise ValueError(f"frontend cache class list mismatch: {path}")
        if count and (class_ids.min() < 0 or class_ids.max() >= len(classes)):
            raise ValueError(f"frontend cache class ID out of range: {path}")
        file_hashes[path.name] = _sha256(path)

    provenance_paths = {
        "script": Path(frontend["script"]),
        "hydra_config": Path(frontend["hydra_config_path"]),
        "dataset_config": Path(frontend["dataset_config"]),
        "classes_file": classes_file,
        "yolo_model": Path(frontend["yolo_model_path"]),
        "mobile_sam_model": Path(frontend["mobile_sam_model_path"]),
        "clip_model": Path(frontend["clip_pretrained_path"]),
    }
    provenance = {name: _sha256(path) for name, path in provenance_paths.items()}
    manifest = {
        "schema_version": 1,
        "method": "OVIV2",
        "scene": command.scene,
        "frame_count": frame_count,
        "mask_shape": [height, width],
        "classes": expected_classes,
        "classes_hash": _json_hash(expected_classes),
        "algorithm_hash": command.algorithm_hash,
        "provenance_sha256": provenance,
        "cache_files_sha256": file_hashes,
    }
    _atomic_json(command.cache_dir / "frontend_manifest.json", manifest)
    return manifest


def _run_queue(
    commands: list[FrontendCommand],
    config: dict[str, Any],
    frame_count: int,
) -> None:
    cwd = Path(config["frontend"]["script"]).resolve().parents[2]
    for command in commands:
        try:
            validate_frontend_cache(command, config, frame_count=frame_count)
            continue
        except (FileNotFoundError, ValueError, EOFError, OSError, pickle.UnpicklingError):
            pass
        command.log_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(command.gpu_id)
        with (command.log_dir / "frontend.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command.argv,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        validate_frontend_cache(command, config, frame_count=frame_count)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/oviv2_replica8.json")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--gpu", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = _load_json(args.config.resolve())
    manifest_path = Path(config["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()
    manifest = _load_json(manifest_path)
    gpu_ids = tuple(args.gpu or [0, 1])
    commands = build_commands(config, manifest, gpu_ids=gpu_ids, scenes=args.scene)
    if args.dry_run:
        for command in commands:
            print(json.dumps({"scene": command.scene, "gpu": command.gpu_id, "argv": command.argv}))
        return 0
    frame_count = int(manifest["frame_selection"]["sampled_frames_per_scene"])
    queues = [[command for command in commands if command.gpu_id == gpu_id] for gpu_id in gpu_ids]
    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = [executor.submit(_run_queue, queue, config, frame_count) for queue in queues]
        for future in futures:
            future.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
