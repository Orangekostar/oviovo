#!/usr/bin/env python3
"""Generate and verify independent OVIV2 frontend caches for ScanNet200-5."""

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
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.materialize_oviv2_scannet200_view import materialize_scene
from scripts.precompute_oviv2_replica_frontend import (
    build_environment,
    warm_shared_clip_cache,
)


SCENES = (
    "scene0011_00",
    "scene0050_00",
    "scene0231_00",
    "scene0378_00",
    "scene0518_00",
)
STRUCTURE_CLASSES = {"wall", "floor", "ceiling"}


@dataclass(frozen=True)
class FrontendCommand:
    scene: str
    gpu_id: int
    argv: tuple[str, ...]
    cache_dir: Path
    log_dir: Path
    algorithm_hash: str
    frame_count: int
    source_frame_ids: tuple[int, ...]


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
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
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


def build_commands(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    gpu_ids: tuple[int, ...],
    scenes: Iterable[str] | None = None,
) -> list[FrontendCommand]:
    if not gpu_ids or any(value < 0 for value in gpu_ids):
        raise ValueError("at least one non-negative GPU ID is required")
    if manifest.get("dataset") != "ScanNet200":
        raise ValueError("manifest dataset must be ScanNet200")
    records = list(manifest.get("scenes", ()))
    if tuple(value.get("scene") for value in records) != SCENES:
        raise ValueError("manifest must contain the frozen ScanNet200 five-scene order")
    requested = set(SCENES if scenes is None else scenes)
    if not requested <= set(SCENES):
        raise ValueError("requested scene is outside the frozen ScanNet200 split")
    frontend = dict(config["frontend"])
    descriptor = {
        key: frontend[key]
        for key in sorted(frontend)
        if key not in {"gsa_variant_template", "exp_suffix_template"}
    }
    algorithm_hash = _json_hash(descriptor)
    view_root = Path(config["view_root"]).expanduser().resolve()
    logs_root = Path(config["frontend_logs_root"]).expanduser().resolve()
    commands: list[FrontendCommand] = []
    for index, record in enumerate(records):
        scene = str(record["scene"])
        if scene not in requested:
            continue
        frame_count = int(record["frame_count"])
        source_frame_ids = tuple(int(value) for value in record["source_frame_ids"])
        if frame_count != len(source_frame_ids):
            raise ValueError(f"scene {scene} frame_count mismatch")
        variant = str(frontend["gsa_variant_template"]).format(scene=scene)
        suffix = str(frontend["exp_suffix_template"]).format(scene=scene)
        log_dir = logs_root / scene
        argv = (
            str(frontend["python"]),
            str(frontend["script"]),
            f"--config-name={frontend['hydra_config_name']}",
            f"dataset_root={view_root}",
            f"dataset_config={frontend['dataset_config']}",
            f"scene_id={scene}",
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
                cache_dir=view_root / scene / f"gsa_detections_{variant}",
                log_dir=log_dir,
                algorithm_hash=algorithm_hash,
                frame_count=frame_count,
                source_frame_ids=source_frame_ids,
            )
        )
    return commands


def validate_frontend_cache(
    command: FrontendCommand,
    config: dict[str, Any],
    *,
    input_manifest_sha256: str,
) -> dict[str, Any]:
    frontend = config["frontend"]
    classes_path = Path(frontend["classes_file"])
    classes = [
        value.strip()
        for value in classes_path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if len(classes) != 200 or len(set(classes)) != 200:
        raise ValueError("ScanNet frontend vocabulary must contain 200 unique classes")
    object_classes = [value for value in classes if value.lower() not in STRUCTURE_CLASSES]
    view_manifest_path = command.cache_dir.parent / "view_manifest.json"
    view_manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
    if (
        view_manifest.get("scene") != command.scene
        or view_manifest.get("frame_count") != command.frame_count
        or view_manifest.get("source_frame_ids") != list(command.source_frame_ids)
        or view_manifest.get("source_manifest_sha256") != input_manifest_sha256
    ):
        raise ValueError("view manifest does not match frontend command")
    with Image.open(command.cache_dir.parent / "results" / "frame000000.jpg") as image:
        width, height = image.size
    expected_cache_names = {
        f"frame{cache_index:06d}.pkl.gz" for cache_index in range(command.frame_count)
    }
    actual_cache_names = {path.name for path in command.cache_dir.glob("*.pkl.gz")}
    if actual_cache_names != expected_cache_names:
        raise ValueError("frontend directory contains a missing or unexpected cache file")
    file_hashes: dict[str, str] = {}
    for cache_index in range(command.frame_count):
        path = command.cache_dir / f"frame{cache_index:06d}.pkl.gz"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"frontend cache must be a regular non-symlink file: {path}")
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"frontend cache payload must be a dictionary: {path}")
        masks = np.asarray(payload.get("mask"))
        boxes = np.asarray(payload.get("xyxy"))
        confidences = np.asarray(payload.get("confidence"))
        class_ids = np.asarray(payload.get("class_id"))
        cached_classes = [str(value) for value in payload.get("classes", ())]
        count = masks.shape[0] if masks.ndim == 3 else -1
        if masks.shape != (count, height, width) or boxes.shape != (count, 4):
            raise ValueError(f"frontend cache mask/box shape mismatch: {path}")
        if confidences.shape != (count,) or class_ids.shape != (count,):
            raise ValueError(f"frontend cache vector length mismatch: {path}")
        if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(confidences)):
            raise ValueError(f"frontend cache boxes/confidences must be finite: {path}")
        if cached_classes != object_classes:
            raise ValueError(f"frontend cache class order mismatch: {path}")
        if count and (class_ids.min() < 0 or class_ids.max() >= len(object_classes)):
            raise ValueError(f"frontend cache class ID out of range: {path}")
        for name in ("image_feats", "text_feats"):
            features = np.asarray(payload.get(name), dtype=np.float64)
            if (
                features.ndim != 2
                or features.shape[0] != count
                or features.shape[1] <= 0
                or not np.all(np.isfinite(features))
            ):
                raise ValueError(f"frontend cache {name} is invalid: {path}")
            if count and np.any(np.linalg.norm(features, axis=1) == 0.0):
                raise ValueError(f"frontend cache {name} contains a zero row: {path}")
        file_hashes[path.name] = _sha256(path)

    provenance_paths = {
        "script": Path(frontend["script"]),
        "hydra_config": Path(frontend["hydra_config_path"]),
        "dataset_config": Path(frontend["dataset_config"]),
        "classes_file": classes_path,
        "yolo_model": Path(frontend["yolo_model_path"]),
        "yolo_clip_model": Path(frontend["yolo_clip_model_path"]),
        "mobile_sam_model": Path(frontend["mobile_sam_model_path"]),
        "clip_model": Path(frontend["clip_pretrained_path"]),
    }
    provenance = {name: _sha256(path) for name, path in provenance_paths.items()}
    result = {
        "schema_version": 1,
        "method": "OVIV2",
        "dataset": "ScanNet200",
        "scene": command.scene,
        "frame_count": command.frame_count,
        "source_frame_ids": list(command.source_frame_ids),
        "mask_shape": [height, width],
        "class_count": len(object_classes),
        "classes": object_classes,
        "classes_hash": _json_hash(object_classes),
        "algorithm_hash": command.algorithm_hash,
        "feature_model_id": f"clip-sha256:{provenance['clip_model']}",
        "input_manifest_sha256": input_manifest_sha256,
        "view_manifest_sha256": _sha256(view_manifest_path),
        "provenance_sha256": provenance,
        "cache_files_sha256": file_hashes,
    }
    _atomic_json(command.cache_dir / "frontend_manifest.json", result)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _run_queue(
    commands: list[FrontendCommand],
    config: dict[str, Any],
    input_manifest_sha256: str,
) -> None:
    frontend = config["frontend"]
    cwd = Path(frontend["script"]).resolve().parents[2]
    for command in commands:
        try:
            validate_frontend_cache(
                command,
                config,
                input_manifest_sha256=input_manifest_sha256,
            )
            continue
        except (FileNotFoundError, ValueError, EOFError, OSError, pickle.UnpicklingError):
            pass
        command.log_dir.mkdir(parents=True, exist_ok=True)
        environment = build_environment(command.gpu_id, cwd)
        with (command.log_dir / "frontend.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command.argv,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=input_manifest_sha256,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--gpu", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _load_json(config_path)
    manifest_path = Path(config["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()
    manifest = _load_json(manifest_path)
    gpu_ids = tuple(args.gpu or (0, 1, 2))
    commands = build_commands(
        config,
        manifest,
        gpu_ids=gpu_ids,
        scenes=args.scene,
    )
    if args.dry_run:
        for command in commands:
            print(
                json.dumps(
                    {
                        "scene": command.scene,
                        "gpu": command.gpu_id,
                        "frame_count": command.frame_count,
                        "argv": command.argv,
                    }
                )
            )
        return 0

    for command in commands:
        view_root = command.cache_dir.parent
        if not view_root.exists():
            materialize_scene(manifest_path, command.scene, view_root)
    warm_shared_clip_cache(config["frontend"])
    manifest_sha256 = _sha256(manifest_path)
    queues = [
        [command for command in commands if command.gpu_id == gpu_id]
        for gpu_id in gpu_ids
    ]
    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = [
            executor.submit(_run_queue, queue, config, manifest_sha256)
            for queue in queues
        ]
        for future in futures:
            future.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
