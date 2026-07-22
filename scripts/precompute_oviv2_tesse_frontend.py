#!/usr/bin/env python3
"""Generate and verify frozen OVIV2 object-frontend caches for TESSE-CD."""

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
import re
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENES = ("apartment", "office")
SCENE_GPUS = {"apartment": 0, "office": 1}
SCENE_FRAME_COUNTS = {"apartment": 1745, "office": 4346}
IMAGE_SHAPE = (480, 720)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REQUIRED_CACHE_FIELDS = {
    "mask",
    "xyxy",
    "confidence",
    "class_id",
    "classes",
    "image_feats",
    "text_feats",
}
PROVENANCE_PATH_KEYS = {
    "script": "script",
    "hydra_config": "hydra_config_path",
    "dataset_config": "dataset_config",
    "yolo_model": "yolo_model_path",
    "yolo_clip_model": "yolo_clip_model_path",
    "mobile_sam_model": "mobile_sam_model_path",
    "clip_model": "clip_pretrained_path",
}


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
    image_shape: tuple[int, int]
    classes_file: Path
    classes_sha256: str


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return Path(os.path.abspath(path))


def _strict_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _scene_records(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    if manifest.get("dataset") != "TESSE-CD":
        raise ValueError("manifest dataset must be TESSE-CD")
    camera = manifest.get("camera")
    if not isinstance(camera, Mapping) or (
        camera.get("height"), camera.get("width")
    ) != IMAGE_SHAPE:
        raise ValueError("TESSE-CD camera must be frozen at 480x720")
    records = manifest.get("scenes")
    if not isinstance(records, Mapping) or tuple(records) != SCENES:
        raise ValueError("manifest must contain the frozen TESSE-CD scene order")
    return records


def build_commands(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    scenes: Iterable[str] | None = None,
) -> list[FrontendCommand]:
    records = _scene_records(manifest)
    requested_values = SCENES if scenes is None else tuple(scenes)
    requested = set(requested_values)
    unknown = requested - set(SCENES)
    if unknown:
        raise ValueError(f"unknown TESSE-CD scene: {', '.join(sorted(unknown))}")
    frontend = config.get("frontend")
    if not isinstance(frontend, Mapping):
        raise ValueError("frontend config must be an object")
    dimensions = (
        _strict_int(frontend.get("desired_height"), name="desired_height"),
        _strict_int(frontend.get("desired_width"), name="desired_width"),
    )
    if dimensions != IMAGE_SHAPE:
        raise ValueError("TESSE-CD frontend image shape must be 480x720")
    logs_root = _resolve_path(config.get("frontend_logs_root"))

    commands: list[FrontendCommand] = []
    for scene in SCENES:
        if scene not in requested:
            continue
        record = records[scene]
        if not isinstance(record, Mapping):
            raise ValueError(f"scene {scene} record must be an object")
        frame_count = _strict_int(record.get("frame_count"), name=f"{scene} frame_count")
        if frame_count != SCENE_FRAME_COUNTS[scene]:
            raise ValueError(f"scene {scene} frame_count is not frozen")
        if tuple(record.get("image_shape", ())) != IMAGE_SHAPE:
            raise ValueError(f"scene {scene} image shape is not 480x720")
        selection = record.get("source_frame_ids")
        if not isinstance(selection, Mapping) or selection != {
            "start": 0,
            "stop_exclusive": frame_count,
            "stride": 1,
        }:
            raise ValueError(f"scene {scene} source frame IDs must equal range(frame_count)")
        root = _resolve_path(record.get("root"))
        if root.is_symlink() or not root.is_dir() or root.name != scene:
            raise ValueError(f"scene {scene} root path must end with the scene name")
        vocabulary = record.get("vocabulary")
        if not isinstance(vocabulary, Mapping):
            raise ValueError(f"scene {scene} vocabulary must be an object")
        classes_file = _resolve_path(vocabulary.get("txt_path"))
        classes_sha256 = str(vocabulary.get("txt_sha256", ""))
        if classes_file.is_symlink() or not classes_file.is_file():
            raise ValueError(
                f"scene {scene} vocabulary must be a regular non-symlink file"
            )
        if _sha256(classes_file) != classes_sha256:
            raise ValueError(f"scene {scene} vocabulary hash mismatch")
        variant = str(frontend["gsa_variant_template"]).format(scene=scene)
        suffix = str(frontend["exp_suffix_template"]).format(scene=scene)
        if not variant or Path(variant).name != variant:
            raise ValueError(f"scene {scene} gsa variant is not a safe path component")
        if not suffix or Path(suffix).name != suffix:
            raise ValueError(f"scene {scene} experiment suffix is not a safe path component")
        log_dir = logs_root / scene
        descriptor = {
            "scene": scene,
            "frame_count": frame_count,
            "image_shape": list(IMAGE_SHAPE),
            "classes_sha256": classes_sha256,
            "frontend": {
                key: frontend[key]
                for key in sorted(frontend)
                if key not in {"gsa_variant_template", "exp_suffix_template"}
            },
            "gsa_variant": variant,
            "exp_suffix": suffix,
        }
        argv = (
            str(frontend["python"]),
            str(frontend["script"]),
            f"--config-name={frontend['hydra_config_name']}",
            f"dataset_root={root.parent}",
            f"dataset_config={frontend['dataset_config']}",
            f"scene_id={scene}",
            "start=0",
            f"end={frame_count}",
            "stride=1",
            "desired_height=480",
            "desired_width=720",
            f"classes_file={classes_file}",
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
                gpu_id=SCENE_GPUS[scene],
                argv=argv,
                cache_dir=root / f"gsa_detections_{variant}",
                log_dir=log_dir,
                algorithm_hash=_json_hash(descriptor),
                frame_count=frame_count,
                source_frame_ids=tuple(range(frame_count)),
                image_shape=IMAGE_SHAPE,
                classes_file=classes_file,
                classes_sha256=classes_sha256,
            )
        )
    return commands


def build_environment(
    gpu_id: int,
    conceptgraphs_root: Path,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if type(gpu_id) is not int or gpu_id < 0:
        raise ValueError("GPU ID must be a non-negative integer")
    environment = dict(os.environ if base is None else base)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{conceptgraphs_root}{os.pathsep}{existing}"
        if existing
        else str(conceptgraphs_root)
    )
    return environment


def warm_shared_clip_cache(frontend: Mapping[str, Any]) -> str:
    model_path = _resolve_path(frontend.get("yolo_clip_model_path"))
    provenance = frontend.get("provenance_sha256")
    if not isinstance(provenance, Mapping):
        raise ValueError("frontend provenance_sha256 must be an object")
    expected = str(provenance.get("yolo_clip_model", ""))
    if model_path.is_file():
        if _sha256(model_path) != expected:
            raise ValueError("shared YOLO-World CLIP model checksum mismatch")
        return expected
    code = (
        "from clip.clip import _MODELS, _download; "
        "_download(_MODELS['ViT-B/32'], download_root=r'{}')"
    ).format(model_path.parent)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(frontend["python"]), "-c", code], check=True)
    if not model_path.is_file() or _sha256(model_path) != expected:
        raise ValueError("shared YOLO-World CLIP model checksum mismatch after warmup")
    return expected


def _regular_file(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")


def _validated_provenance(frontend: Mapping[str, Any]) -> dict[str, str]:
    expected = frontend.get("provenance_sha256")
    if not isinstance(expected, Mapping) or set(expected) != set(PROVENANCE_PATH_KEYS):
        raise ValueError("frontend provenance_sha256 keys are not frozen")
    result: dict[str, str] = {}
    for name, config_key in PROVENANCE_PATH_KEYS.items():
        path = _resolve_path(frontend.get(config_key))
        _regular_file(path, name=name)
        digest = _sha256(path)
        recorded = expected[name]
        if not isinstance(recorded, str) or SHA256_PATTERN.fullmatch(recorded) is None:
            raise ValueError(f"{name} hash is not a canonical SHA-256")
        if digest != recorded:
            raise ValueError(f"{name} hash mismatch")
        result[name] = digest
    return result


def _cache_prefix_sha256(cache_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for cache_index, checksum in enumerate(cache_hashes.values()):
        digest.update(cache_index.to_bytes(8, "little", signed=False))
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def _publish_json_new(path: Path, payload: Mapping[str, Any]) -> None:
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
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def validate_frontend_cache(
    command: FrontendCommand,
    config: Mapping[str, Any],
    *,
    input_manifest_sha256: str,
) -> dict[str, Any]:
    if command.scene not in SCENES:
        raise ValueError("unknown TESSE-CD scene in frontend command")
    if command.source_frame_ids != tuple(range(command.frame_count)):
        raise ValueError("frontend source frame IDs must equal range(frame_count)")
    if command.image_shape != IMAGE_SHAPE:
        raise ValueError("frontend command image shape must be 480x720")
    recorded_manifest_sha256 = config.get("manifest_sha256")
    if (
        not isinstance(input_manifest_sha256, str)
        or SHA256_PATTERN.fullmatch(input_manifest_sha256) is None
        or input_manifest_sha256 != recorded_manifest_sha256
    ):
        raise ValueError("input manifest hash mismatch")

    classes = [
        value.strip()
        for value in command.classes_file.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("frontend vocabulary must contain unique non-empty classes")
    if _sha256(command.classes_file) != command.classes_sha256:
        raise ValueError("frontend vocabulary hash mismatch")

    expected_names = [
        f"frame{cache_index:06d}.pkl.gz" for cache_index in range(command.frame_count)
    ]
    actual_names = sorted(path.name for path in command.cache_dir.glob("*.pkl.gz"))
    if actual_names != expected_names:
        raise ValueError("frontend directory contains a missing or unexpected cache file")

    cache_hashes: dict[str, str] = {}
    height, width = command.image_shape
    for cache_index, name in enumerate(expected_names):
        path = command.cache_dir / name
        _regular_file(path, name=f"frontend cache frame {cache_index}")
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"frontend cache payload must be a dictionary: {path}")
        if not REQUIRED_CACHE_FIELDS <= set(payload):
            raise ValueError(f"frontend cache payload is missing required fields: {path}")
        masks = np.asarray(payload["mask"])
        boxes = np.asarray(payload["xyxy"])
        confidences = np.asarray(payload["confidence"])
        class_ids = np.asarray(payload["class_id"])
        cached_classes = [str(value) for value in payload["classes"]]
        count = masks.shape[0] if masks.ndim == 3 else -1
        if masks.shape != (count, height, width) or boxes.shape != (count, 4):
            raise ValueError(f"frontend cache mask/box shape mismatch: {path}")
        if confidences.shape != (count,) or class_ids.shape != (count,):
            raise ValueError(f"frontend cache vector length mismatch: {path}")
        if class_ids.dtype.kind not in {"i", "u"}:
            raise ValueError(f"frontend cache class IDs must be integers: {path}")
        if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(confidences)):
            raise ValueError(f"frontend cache boxes/confidences must be finite: {path}")
        if cached_classes != classes:
            raise ValueError(f"frontend cache class order mismatch: {path}")
        if count and (class_ids.min() < 0 or class_ids.max() >= len(classes)):
            raise ValueError(f"frontend cache class ID out of range: {path}")
        for field in ("image_feats", "text_feats"):
            features = np.asarray(payload[field], dtype=np.float64)
            if (
                features.ndim != 2
                or features.shape[0] != count
                or features.shape[1] <= 0
                or not np.all(np.isfinite(features))
            ):
                raise ValueError(f"frontend cache {field} is invalid: {path}")
            if count and np.any(np.linalg.norm(features, axis=1) == 0.0):
                raise ValueError(f"frontend cache {field} contains a zero row: {path}")
        cache_hashes[name] = _sha256(path)

    frontend = config.get("frontend")
    if not isinstance(frontend, Mapping):
        raise ValueError("frontend config must be an object")
    provenance = _validated_provenance(frontend)
    expected_provenance = frontend.get("provenance_sha256")
    if provenance != expected_provenance:
        raise ValueError("verified frontend provenance does not match config")

    result = {
        "schema_version": 1,
        "method": "OVIV2",
        "dataset": "TESSE-CD",
        "scene": command.scene,
        "frame_count": command.frame_count,
        "source_frame_ids": list(command.source_frame_ids),
        "source_frame_ids_hash": _json_hash(list(command.source_frame_ids)),
        "image_shape": list(command.image_shape),
        "class_count": len(classes),
        "classes": classes,
        "vocabulary_sha256": command.classes_sha256,
        "algorithm_hash": command.algorithm_hash,
        "feature_model_id": f"clip-sha256:{provenance['clip_model']}",
        "input_manifest_sha256": input_manifest_sha256,
        "provenance_sha256": provenance,
        "cache_files_sha256": cache_hashes,
        "cache_prefix_sha256": _cache_prefix_sha256(cache_hashes),
    }
    manifest_path = command.cache_dir / "frontend_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        _regular_file(manifest_path, name="frontend manifest")
        if _load_json_object(manifest_path) != result:
            raise ValueError("existing frontend manifest does not match validated cache")
        return result
    try:
        _publish_json_new(manifest_path, result)
    except FileExistsError:
        _regular_file(manifest_path, name="frontend manifest")
        if _load_json_object(manifest_path) != result:
            raise ValueError("existing frontend manifest does not match validated cache")
    return result


def _run_queue(
    commands: list[FrontendCommand],
    config: Mapping[str, Any],
    input_manifest_sha256: str,
) -> None:
    frontend = config["frontend"]
    cwd = _resolve_path(frontend["script"]).parents[2]
    for command in commands:
        manifest_path = command.cache_dir / "frontend_manifest.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            validate_frontend_cache(
                command,
                config,
                input_manifest_sha256=input_manifest_sha256,
            )
            continue
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
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/oviv2_tesse_cd_frontend_stage3.json",
    )
    parser.add_argument("--scene", action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    _regular_file(config_path, name="frontend config")
    config = _load_json_object(config_path)
    manifest_path = _resolve_path(config.get("manifest"))
    _regular_file(manifest_path, name="TESSE-CD input manifest")
    input_manifest_sha256 = _sha256(manifest_path)
    recorded_manifest_sha256 = config.get("manifest_sha256")
    if (
        not isinstance(recorded_manifest_sha256, str)
        or SHA256_PATTERN.fullmatch(recorded_manifest_sha256) is None
        or input_manifest_sha256 != recorded_manifest_sha256
    ):
        raise ValueError("input manifest hash mismatch")
    manifest = _load_json_object(manifest_path)
    commands = build_commands(config, manifest, scenes=args.scene)
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

    frontend = config["frontend"]
    warm_shared_clip_cache(frontend)
    _validated_provenance(frontend)
    queues = [
        [command for command in commands if command.gpu_id == gpu_id]
        for gpu_id in (0, 1)
    ]
    queues = [queue for queue in queues if queue]
    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = [
            executor.submit(
                _run_queue,
                queue,
                config,
                input_manifest_sha256,
            )
            for queue in queues
        ]
        for future in futures:
            future.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
