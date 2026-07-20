#!/usr/bin/env python3
"""Precompute immutable dense semantic probability caches for OVIV2 Replica."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import base64
import binascii
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.replica import ReplicaRoom0Dataset  # noqa: E402
from src.models.json_line_worker_client import JsonLineWorkerClient  # noqa: E402
from src.oviv2.dense_semantics import (  # noqa: E402
    DenseSemanticFrame,
    DenseSemanticProvenance,
    load_dense_frame,
    sha256_file,
    write_dense_frame,
)


CLASS_COUNT = 41
_METHOD = "OVIV2-dense-semantic-cache"
_MANIFEST_NAME = "dense_manifest.json"
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_ARRAY_BYTES = 512 * 1024 * 1024
_MAX_WORKER_RESPONSE_CHARS = 16 * 1024 * 1024 + 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ARRAY_BLOCK_KEYS = frozenset({"encoding", "dtype", "shape", "data"})
_METADATA_KEYS = frozenset(
    {"id", "ok", "class_count", "classes", "sample_stride", "top_k", "provenance"}
)
_INFERENCE_KEYS = frozenset(
    {
        "id",
        "ok",
        "image_shape",
        "class_count",
        "sample_stride",
        "class_ids",
        "probabilities",
        "entropy",
        "margin",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "backend",
        "source_commit",
        "radio_commit",
        "model_id",
        "model_sha256",
        "auxiliary_model_sha256",
        "language_model_id",
        "language_model_revision",
        "language_model_sha256",
        "vocabulary_sha256",
        "prompt_sha256",
        "inference_config_sha256",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "method",
        "scene",
        "frame_count",
        "source_frame_ids",
        "image_shape",
        "sample_stride",
        "top_k",
        "class_count",
        "vocabulary_sha256",
        "provenance",
        "cache_files_sha256",
    }
)


@dataclass(frozen=True)
class _WorkerConfig:
    command: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Path
    request_timeout_sec: float
    classes_json: Path


@dataclass(frozen=True)
class _Preflight:
    scene: str
    rgb_frames: tuple[np.ndarray, ...]
    source_frame_ids: tuple[int, ...]
    image_shape: tuple[int, int]
    classes: tuple[str, ...]
    vocabulary_sha256: str
    worker: _WorkerConfig


@dataclass(frozen=True)
class _WorkerMetadata:
    class_count: int
    classes: tuple[str, ...]
    sample_stride: int
    top_k: int
    provenance: Mapping[str, str]


def _positive_int(value: str) -> int:
    try:
        normalized = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if normalized <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return normalized


def _strict_int(value: object, name: str, *, positive: bool) -> int:
    if type(value) is not int:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    normalized = int(value)
    if (positive and normalized <= 0) or (not positive and normalized < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    if normalized > int(np.iinfo(np.int64).max):
        raise ValueError(f"{name} must fit in a signed int64")
    return normalized


def _strict_image_shape(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element integer list")
    return tuple(
        _strict_int(dimension, f"{name}[{index}]", positive=True)
        for index, dimension in enumerate(value)
    )


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        initial = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{name} must be a regular file: {path}") from exc
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError(f"{name} must be a regular file: {path}")
    if initial.st_size > _MAX_JSON_BYTES:
        raise ValueError(f"{name} exceeds the JSON size limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{name} could not be opened as a regular file: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (initial.st_dev, initial.st_ino)
        ):
            raise ValueError(f"{name} changed before it could be opened safely")
        if opened.st_size > _MAX_JSON_BYTES:
            raise ValueError(f"{name} exceeds the JSON size limit")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            raw = stream.read(_MAX_JSON_BYTES + 1)
        after = os.fstat(descriptor)
        if len(raw) > _MAX_JSON_BYTES or after.st_size > _MAX_JSON_BYTES:
            raise ValueError(f"{name} exceeds the JSON size limit")
        if len(raw) != after.st_size or opened.st_size != after.st_size:
            raise ValueError(f"{name} changed while it was being read")
    finally:
        os.close(descriptor)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} root must be an object")
    return payload


def _config_path(value: object, name: str, *, base: Path = REPO_ROOT) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return path.absolute() if path.is_absolute() else (base / path).absolute()


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    parent = path.parent
    try:
        expected_parent = os.stat(parent, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"manifest parent directory is unavailable: {parent}") from exc
    if not stat.S_ISDIR(expected_parent.st_mode):
        raise ValueError(f"manifest parent must be a directory: {parent}")
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(parent, parent_flags)
    temporary_name: str | None = None
    temporary_descriptor: int | None = None
    destination_linked = False
    directory_changed = False

    def require_parent_identity() -> None:
        opened = os.fstat(parent_descriptor)
        try:
            current = os.stat(parent, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError("manifest parent directory changed during publication") from exc
        identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or identity != (expected_parent.st_dev, expected_parent.st_ino)
            or identity != (current.st_dev, current.st_ino)
        ):
            raise RuntimeError("manifest parent directory changed during publication")

    try:
        require_parent_identity()
        for _ in range(128):
            candidate = f".{path.name}.{secrets.token_hex(16)}.tmp"
            try:
                temporary_descriptor = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            directory_changed = True
            break
        if temporary_descriptor is None or temporary_name is None:
            raise FileExistsError("could not allocate a unique manifest temporary file")
        owned_descriptor = temporary_descriptor
        temporary_descriptor = None
        with os.fdopen(owned_descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        require_parent_identity()
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        destination_linked = True
        try:
            require_parent_identity()
        except RuntimeError:
            os.unlink(path.name, dir_fd=parent_descriptor)
            destination_linked = False
            raise
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_name = None
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        if directory_changed or destination_linked:
            os.fsync(parent_descriptor)
        os.close(parent_descriptor)


def _classes_argument(command: Sequence[str]) -> str | None:
    found: list[str] = []
    for index, argument in enumerate(command):
        if argument == "--classes-json":
            if index + 1 >= len(command):
                raise ValueError("worker_command --classes-json requires a value")
            found.append(command[index + 1])
        elif argument.startswith("--classes-json="):
            found.append(argument.split("=", 1)[1])
    if len(found) > 1:
        raise ValueError("worker_command must contain at most one --classes-json")
    return found[0] if found else None


def _replace_classes_argument(command: Sequence[str], classes_json: Path) -> tuple[str, ...]:
    updated: list[str] = []
    replaced = False
    index = 0
    while index < len(command):
        argument = command[index]
        if argument == "--classes-json":
            if index + 1 >= len(command):
                raise ValueError("worker_command --classes-json requires a value")
            updated.extend((argument, str(classes_json)))
            replaced = True
            index += 2
            continue
        if argument.startswith("--classes-json="):
            updated.append(f"--classes-json={classes_json}")
            replaced = True
        else:
            updated.append(argument)
        index += 1
    if not replaced:
        updated.extend(("--classes-json", str(classes_json)))
    return tuple(updated)


def _explicit_worker_requested(args: argparse.Namespace) -> bool:
    names = (
        "worker_python",
        "worker_script",
        "backend",
        "source_root",
        "radio_root",
        "model_version",
        "lang_model",
        "language_model_root",
        "language_model_id",
        "language_model_revision",
        "language_model_sha256",
        "device",
        "sample_stride",
        "top_k",
        "amp",
        "sam_refinement",
        "sam_checkpoint",
    )
    return any(getattr(args, name, None) is not None for name in names)


def _explicit_worker_config(
    args: argparse.Namespace,
    classes_json: Path,
) -> _WorkerConfig:
    required = (
        "backend",
        "source_root",
        "radio_root",
        "model_version",
        "lang_model",
        "language_model_root",
        "language_model_id",
        "language_model_revision",
        "language_model_sha256",
    )
    missing = [name.replace("_", "-") for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(
            "explicit worker arguments require: "
            + ", ".join(f"--{name}" for name in missing)
        )
    worker_python = args.worker_python or Path(sys.executable)
    worker_script = args.worker_script or REPO_ROOT / "scripts/radseg_dense_worker.py"
    command = [
        str(worker_python),
        str(worker_script),
        "--backend",
        args.backend,
        "--source-root",
        str(args.source_root),
        "--radio-root",
        str(args.radio_root),
        "--model-version",
        args.model_version,
        "--lang-model",
        args.lang_model,
        "--language-model-root",
        str(args.language_model_root),
        "--language-model-id",
        args.language_model_id,
        "--language-model-revision",
        args.language_model_revision,
        "--language-model-sha256",
        args.language_model_sha256,
        "--classes-json",
        str(classes_json),
        "--device",
        args.device or "cuda",
        "--sample-stride",
        str(args.sample_stride or 4),
        "--top-k",
        str(args.top_k or 4),
    ]
    if args.amp:
        command.append("--amp")
    if args.sam_refinement:
        command.append("--sam-refinement")
    if args.sam_checkpoint is not None:
        command.extend(("--sam-checkpoint", str(args.sam_checkpoint)))
    return _WorkerConfig(
        command=tuple(command),
        env=os.environ.copy(),
        cwd=REPO_ROOT,
        request_timeout_sec=300.0,
        classes_json=classes_json,
    )


def _worker_config(
    config: Mapping[str, Any],
    args: argparse.Namespace,
    manifest_classes_json: Path | None,
) -> _WorkerConfig:
    cli_classes = getattr(args, "classes_json", None)
    cli_classes_path = (
        _absolute_without_resolving(Path(cli_classes))
        if cli_classes is not None
        else None
    )
    if _explicit_worker_requested(args):
        selected_classes = cli_classes_path or manifest_classes_json
        if selected_classes is None:
            raise ValueError(
                "--classes-json or benchmark vocabulary.source_path is required "
                "with explicit worker arguments"
            )
        return _explicit_worker_config(
            args,
            selected_classes,
        )

    dense = config.get("dense_semantics")
    if not isinstance(dense, dict):
        raise ValueError(
            "dense_semantics must be an object when explicit worker arguments are absent"
        )
    raw_command = dense.get("worker_command")
    if (
        not isinstance(raw_command, list)
        or not raw_command
        or any(not isinstance(value, str) or not value.strip() for value in raw_command)
        or any("\0" in value for value in raw_command)
    ):
        raise ValueError("dense_semantics.worker_command must be a non-empty string list")
    command = tuple(raw_command)

    raw_cwd = dense.get("worker_cwd", str(REPO_ROOT))
    if not isinstance(raw_cwd, str) or not raw_cwd.strip():
        raise ValueError("dense_semantics.worker_cwd must be a non-empty string")
    cwd = _config_path(raw_cwd, "dense_semantics.worker_cwd")
    if cwd.is_symlink() or not cwd.is_dir():
        raise ValueError("dense_semantics.worker_cwd must be a regular directory")

    raw_timeout = dense.get("request_timeout_sec", 300.0)
    if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
        raise ValueError("dense_semantics.request_timeout_sec must be finite and positive")
    try:
        timeout = float(raw_timeout)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "dense_semantics.request_timeout_sec must be finite and positive"
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("dense_semantics.request_timeout_sec must be finite and positive")

    raw_env = dense.get("worker_env", {})
    if not isinstance(raw_env, dict) or any(
        not isinstance(key, str)
        or not key
        or "\0" in key
        or not isinstance(value, str)
        or "\0" in value
        for key, value in raw_env.items()
    ):
        raise ValueError("dense_semantics.worker_env must be a string mapping")
    environment = os.environ.copy()
    environment.update(raw_env)

    explicit_classes = dense.get("classes_json")
    command_classes = _classes_argument(command)
    explicit_path = (
        _config_path(explicit_classes, "dense_semantics.classes_json")
        if explicit_classes is not None
        else None
    )
    command_path = (
        _config_path(command_classes, "worker_command --classes-json", base=cwd)
        if command_classes is not None
        else None
    )
    if cli_classes_path is None and explicit_path is not None and command_path is not None:
        if explicit_path.resolve(strict=False) != command_path.resolve(strict=False):
            raise ValueError("config and worker_command classes JSON paths do not match")
    classes_json = cli_classes_path or explicit_path or command_path or manifest_classes_json
    if classes_json is None:
        raise ValueError(
            "classes JSON is required from CLI, config, worker command, or benchmark vocabulary"
        )
    if cli_classes_path is not None:
        command = _replace_classes_argument(command, classes_json)
    elif explicit_path is None and command_path is None:
        command = _replace_classes_argument(command, classes_json)
    return _WorkerConfig(
        command=command,
        env=environment,
        cwd=cwd,
        request_timeout_sec=timeout,
        classes_json=classes_json,
    )


def _load_classes(path: Path, benchmark_classes: Sequence[str]) -> tuple[tuple[str, ...], str]:
    payload = _load_json(path, "classes JSON")
    if set(payload) - {"classes", "aliases"}:
        raise ValueError("classes JSON contains unsupported keys")
    classes = payload.get("classes")
    if (
        not isinstance(classes, list)
        or len(classes) != CLASS_COUNT
        or any(not isinstance(value, str) or not value for value in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError(f"classes JSON must contain {CLASS_COUNT} unique classes")
    if classes != list(benchmark_classes):
        raise ValueError("classes JSON order does not match benchmark vocabulary")
    return tuple(classes), sha256_file(path)


def _indexed_replica_paths(directory: Path, pattern: str) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in sorted(directory.glob(pattern)):
        match = re.search(r"(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Replica frame path has no numeric index: {path.name}")
        frame_index = int(match.group(1))
        if frame_index in indexed:
            raise ValueError(f"Replica frame index is duplicated: {frame_index}")
        indexed[frame_index] = path
    return indexed


def _load_requested_rgb_frames(
    dataset: ReplicaRoom0Dataset,
    requested_frames: int,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, ...]:
    rgb_paths = _indexed_replica_paths(dataset.rgb_dir, "frame*.jpg")
    depth_paths = _indexed_replica_paths(dataset.depth_dir, "depth*.png")
    frames: list[np.ndarray] = []
    expected_size = (image_shape[1], image_shape[0])
    for cache_index, frame_index in enumerate(dataset.frame_indices[:requested_frames]):
        rgb_path = rgb_paths.get(frame_index)
        depth_path = depth_paths.get(frame_index)
        if rgb_path is None or depth_path is None:
            raise ValueError(f"Replica frame {frame_index} is missing RGB or depth data")
        with Image.open(rgb_path) as image:
            if image.size != expected_size:
                raise ValueError(f"dataset frame {cache_index} RGB image shape is inconsistent")
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(depth_path) as image:
            if image.size != expected_size:
                raise ValueError(f"dataset frame {cache_index} depth image shape is inconsistent")
        if rgb.shape != (*image_shape, 3) or rgb.dtype != np.dtype(np.uint8):
            raise ValueError(f"dataset frame {cache_index} has an invalid RGB image")
        frames.append(np.ascontiguousarray(rgb))
    if len(frames) != requested_frames:
        raise ValueError("Replica dataset does not cover the requested frame prefix")
    return tuple(frames)


def _preflight(
    config_path: Path,
    requested_frames: int,
    args: argparse.Namespace,
) -> _Preflight:
    config = _load_json(config_path, "config")
    scene = config.get("scene")
    if not isinstance(scene, str) or not scene.strip():
        raise ValueError("config scene must be a non-empty string")
    scene = scene.strip()
    dataset_root = _config_path(config.get("dataset_root"), "dataset_root")
    manifest_path = _config_path(config.get("manifest"), "manifest")
    config_frames = _strict_int(config.get("num_frames"), "config num_frames", positive=True)
    source_start = _strict_int(
        config.get("source_start"),
        "source_start",
        positive=False,
    )
    source_stride = _strict_int(
        config.get("source_stride"),
        "source_stride",
        positive=True,
    )
    if requested_frames > config_frames:
        raise ValueError("requested num_frames exceeds config num_frames")

    benchmark = _load_json(manifest_path, "benchmark manifest")
    if type(benchmark.get("schema_version")) is not int or benchmark["schema_version"] != 1:
        raise ValueError("benchmark manifest must use schema_version 1")
    if benchmark.get("dataset") != "Replica":
        raise ValueError("benchmark manifest dataset must be Replica")
    scenes = benchmark.get("scenes")
    if not isinstance(scenes, list) or scene not in {
        item.get("scene") for item in scenes if isinstance(item, dict)
    }:
        raise ValueError("configured scene is absent from benchmark manifest")
    vocabulary = benchmark.get("vocabulary")
    if not isinstance(vocabulary, dict):
        raise ValueError("benchmark vocabulary must be an object")
    benchmark_classes = vocabulary.get("classes")
    if (
        not isinstance(benchmark_classes, list)
        or len(benchmark_classes) != CLASS_COUNT
        or any(not isinstance(value, str) or not value for value in benchmark_classes)
        or len(set(benchmark_classes)) != CLASS_COUNT
    ):
        raise ValueError(f"benchmark vocabulary must contain exactly {CLASS_COUNT} classes")
    source_path = vocabulary.get("source_path")
    manifest_classes_json = (
        _config_path(source_path, "benchmark vocabulary.source_path")
        if source_path is not None
        else None
    )
    selection = benchmark.get("frame_selection")
    if not isinstance(selection, dict):
        raise ValueError("benchmark frame_selection must be an object")
    selection_start = _strict_int(
        selection.get("start"),
        "manifest frame_selection.start",
        positive=False,
    )
    selection_stride = _strict_int(
        selection.get("stride"),
        "manifest frame_selection.stride",
        positive=True,
    )
    selection_stop = _strict_int(
        selection.get("stop_exclusive"),
        "manifest frame_selection.stop_exclusive",
        positive=True,
    )
    if source_start != selection_start or source_stride != selection_stride:
        raise ValueError("source_start/source_stride must match manifest frame_selection")
    expected_frames = (selection_stop - selection_start + selection_stride - 1) // selection_stride
    recorded_frames = _strict_int(
        selection.get("sampled_frames_per_scene"),
        "manifest frame_selection.sampled_frames_per_scene",
        positive=True,
    )
    if (
        selection_stop <= selection_start
        or config_frames != expected_frames
        or recorded_frames != expected_frames
    ):
        raise ValueError("config num_frames must match frozen manifest frame_selection")

    worker = _worker_config(config, args, manifest_classes_json)
    classes, vocabulary_sha256 = _load_classes(worker.classes_json, benchmark_classes)
    dataset = ReplicaRoom0Dataset(dataset_root)
    if len(dataset) != config_frames:
        raise ValueError(
            f"Replica dataset length {len(dataset)} does not match config num_frames {config_frames}"
        )
    image_shape = (
        _strict_int(dataset.intrinsics.height, "dataset image height", positive=True),
        _strict_int(dataset.intrinsics.width, "dataset image width", positive=True),
    )
    rgb_frames = (
        ()
        if args.resume
        else _load_requested_rgb_frames(dataset, requested_frames, image_shape)
    )
    source_ids = tuple(source_start + index * source_stride for index in range(config_frames))
    if source_ids[-1] >= selection_stop:
        raise ValueError("source frame IDs exceed manifest frame_selection")
    return _Preflight(
        scene=scene,
        rgb_frames=rgb_frames,
        source_frame_ids=source_ids,
        image_shape=image_shape,
        classes=classes,
        vocabulary_sha256=vocabulary_sha256,
        worker=worker,
    )


def _request(
    client: JsonLineWorkerClient,
    payload: dict[str, Any],
    context: str,
) -> dict[str, Any]:
    response = client.request(payload)
    if response.get("ok") is not True:
        error = response.get("error")
        message = error.strip() if isinstance(error, str) and error.strip() else "unknown error"
        raise RuntimeError(f"dense worker {context} failed: {message}")
    return response


def _metadata(response: dict[str, Any], preflight: _Preflight) -> _WorkerMetadata:
    if set(response) != _METADATA_KEYS:
        raise ValueError("worker metadata response keys do not match the contract")
    class_count = _strict_int(response.get("class_count"), "metadata class_count", positive=True)
    classes = response.get("classes")
    if class_count != CLASS_COUNT or classes != list(preflight.classes):
        raise ValueError("worker metadata classes do not match the frozen 41-class vocabulary")
    sample_stride = _strict_int(
        response.get("sample_stride"),
        "metadata sample_stride",
        positive=True,
    )
    top_k = _strict_int(response.get("top_k"), "metadata top_k", positive=True)
    if top_k > class_count:
        raise ValueError("metadata top_k cannot exceed class_count")
    provenance = response.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_KEYS:
        raise ValueError("worker metadata provenance keys do not match the contract")
    if provenance.get("vocabulary_sha256") != preflight.vocabulary_sha256:
        raise ValueError("worker vocabulary_sha256 does not match config classes JSON")
    typed = DenseSemanticProvenance(
        **provenance,
        cache_prefix_sha256="0" * 64,
    )
    normalized = asdict(typed)
    normalized.pop("cache_prefix_sha256")
    return _WorkerMetadata(
        class_count=class_count,
        classes=tuple(classes),
        sample_stride=sample_stride,
        top_k=top_k,
        provenance=normalized,
    )


def _encode_rgb(rgb: np.ndarray) -> dict[str, Any]:
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.dtype(np.uint8)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.shape[0] <= 0
        or rgb.shape[1] <= 0
    ):
        raise ValueError("RGB frame must have uint8 shape [H, W, 3]")
    contiguous = np.ascontiguousarray(rgb)
    return {
        "encoding": "base64",
        "dtype": "uint8",
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
    }


def _decode_array(
    block: object,
    field_name: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(block, dict) or set(block) != _ARRAY_BLOCK_KEYS:
        raise ValueError(f"{field_name} array block keys do not match the contract")
    if block.get("encoding") != "base64":
        raise ValueError(f"{field_name} encoding must be base64")
    if block.get("dtype") != dtype.name:
        raise ValueError(f"{field_name} dtype must be {dtype.name}")
    raw_shape = block.get("shape")
    if (
        not isinstance(raw_shape, list)
        or any(type(value) is not int for value in raw_shape)
        or tuple(raw_shape) != shape
    ):
        raise ValueError(f"{field_name} shape must be {shape}")
    byte_count = math.prod(shape) * dtype.itemsize
    if byte_count > _MAX_ARRAY_BYTES:
        raise ValueError(f"{field_name} exceeds the decoded resource limit")
    encoded = block.get("data")
    if not isinstance(encoded, str):
        raise ValueError(f"{field_name} base64 data must be a string")
    expected_encoded = 4 * ((byte_count + 2) // 3)
    if len(encoded) != expected_encoded:
        raise ValueError(f"{field_name} base64 length does not match its shape")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise ValueError(f"{field_name} data is not valid base64") from exc
    if len(raw) != byte_count:
        raise ValueError(f"{field_name} byte count does not match its shape")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy(order="C")


def _decode_inference(
    response: dict[str, Any],
    metadata: _WorkerMetadata,
    image_shape: tuple[int, int],
    cache_index: int,
    source_frame_id: int,
) -> DenseSemanticFrame:
    if set(response) != _INFERENCE_KEYS:
        raise ValueError("worker inference response keys do not match the contract")
    response_image_shape = _strict_image_shape(
        response.get("image_shape"),
        "worker response image_shape",
    )
    if response_image_shape != image_shape:
        raise ValueError("worker image_shape does not match the input RGB frame")
    response_class_count = _strict_int(
        response.get("class_count"),
        "worker response class_count",
        positive=True,
    )
    if response_class_count != metadata.class_count:
        raise ValueError("worker response class_count does not match metadata")
    response_stride = _strict_int(
        response.get("sample_stride"),
        "worker response sample_stride",
        positive=True,
    )
    if response_stride != metadata.sample_stride:
        raise ValueError("worker response sample_stride does not match metadata")
    sampled_shape = (
        (image_shape[0] + metadata.sample_stride - 1) // metadata.sample_stride,
        (image_shape[1] + metadata.sample_stride - 1) // metadata.sample_stride,
    )
    top_k_shape = (*sampled_shape, metadata.top_k)
    return DenseSemanticFrame(
        cache_frame_id=cache_index,
        source_frame_id=source_frame_id,
        image_shape=image_shape,
        sample_stride=metadata.sample_stride,
        class_count=metadata.class_count,
        class_ids=_decode_array(
            response.get("class_ids"),
            "class_ids",
            np.dtype(np.int64),
            top_k_shape,
        ),
        probabilities=_decode_array(
            response.get("probabilities"),
            "probabilities",
            np.dtype(np.float32),
            top_k_shape,
        ),
        entropy=_decode_array(
            response.get("entropy"),
            "entropy",
            np.dtype(np.float32),
            sampled_shape,
        ),
        margin=_decode_array(
            response.get("margin"),
            "margin",
            np.dtype(np.float32),
            sampled_shape,
        ),
    )


def _verify_frame(
    dense: DenseSemanticFrame,
    cache_index: int,
    source_frame_id: int,
    image_shape: tuple[int, int],
    metadata: _WorkerMetadata,
) -> None:
    if (
        dense.cache_frame_id != cache_index
        or dense.source_frame_id != source_frame_id
        or dense.image_shape != image_shape
        or dense.sample_stride != metadata.sample_stride
        or dense.class_count != metadata.class_count
        or dense.class_ids.shape[2] != metadata.top_k
    ):
        raise ValueError(f"dense frame {cache_index} metadata does not match the cache contract")


def _cache_prefix_sha256(
    cache_files_sha256: Mapping[str, str],
) -> str:
    digest = hashlib.sha256()
    expected_names = [f"frame{index:06d}.npz" for index in range(len(cache_files_sha256))]
    if list(cache_files_sha256) != expected_names:
        raise ValueError("cache checksum keys are not the canonical frame prefix")
    for cache_index, name in enumerate(expected_names):
        checksum = cache_files_sha256[name]
        if not isinstance(checksum, str) or _SHA256_PATTERN.fullmatch(checksum) is None:
            raise ValueError(f"cache checksum is invalid: {name}")
        digest.update(cache_index.to_bytes(8, "little", signed=False))
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def _completed_manifest(
    preflight: _Preflight,
    metadata: _WorkerMetadata,
    frame_count: int,
    cache_hashes: Mapping[str, str],
) -> dict[str, Any]:
    source_ids = list(preflight.source_frame_ids[:frame_count])
    prefix_hash = _cache_prefix_sha256(cache_hashes)
    provenance = DenseSemanticProvenance(
        **metadata.provenance,
        cache_prefix_sha256=prefix_hash,
    )
    return {
        "schema_version": 1,
        "method": _METHOD,
        "scene": preflight.scene,
        "frame_count": frame_count,
        "source_frame_ids": source_ids,
        "image_shape": list(preflight.image_shape),
        "sample_stride": metadata.sample_stride,
        "top_k": metadata.top_k,
        "class_count": metadata.class_count,
        "vocabulary_sha256": preflight.vocabulary_sha256,
        "provenance": asdict(provenance),
        "cache_files_sha256": dict(cache_hashes),
    }


def _regular_output_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"resume {name} must be a regular non-symlink file")


def _reject_symlink_components(path: Path, name: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if _lexists(current) and current.is_symlink():
            raise ValueError(f"{name} cannot contain symlink path components")


def _verify_resume(
    output: Path,
    requested_frames: int,
    preflight: _Preflight,
    metadata: _WorkerMetadata,
) -> dict[str, Any]:
    if output.is_symlink() or not output.is_dir():
        raise ValueError("resume output must be a regular directory")
    manifest_path = output / _MANIFEST_NAME
    _regular_output_file(manifest_path, "manifest")
    manifest = _load_json(manifest_path, "dense manifest")
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError("dense manifest keys contain unsupported or missing entries")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("dense manifest schema_version mismatch")
    if manifest.get("method") != _METHOD or manifest.get("scene") != preflight.scene:
        raise ValueError("dense manifest method or scene mismatch")
    frame_count = _strict_int(manifest.get("frame_count"), "manifest frame_count", positive=True)
    if frame_count < requested_frames or frame_count > len(preflight.source_frame_ids):
        raise ValueError("completed dense manifest does not cover the requested prefix")
    source_ids = manifest.get("source_frame_ids")
    if source_ids != list(preflight.source_frame_ids[:frame_count]):
        raise ValueError("dense manifest source frame prefix mismatch")
    manifest_image_shape = _strict_image_shape(
        manifest.get("image_shape"),
        "manifest image_shape",
    )
    manifest_stride = _strict_int(
        manifest.get("sample_stride"),
        "manifest sample_stride",
        positive=True,
    )
    manifest_top_k = _strict_int(
        manifest.get("top_k"),
        "manifest top_k",
        positive=True,
    )
    manifest_class_count = _strict_int(
        manifest.get("class_count"),
        "manifest class_count",
        positive=True,
    )
    if (
        manifest_image_shape != preflight.image_shape
        or manifest_stride != metadata.sample_stride
        or manifest_top_k != metadata.top_k
        or manifest_class_count != metadata.class_count
        or manifest.get("vocabulary_sha256") != preflight.vocabulary_sha256
    ):
        raise ValueError("dense manifest image, vocabulary, or worker metadata mismatch")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("dense manifest provenance must be an object")
    if set(provenance) != _PROVENANCE_KEYS | {"cache_prefix_sha256"}:
        raise ValueError("dense manifest provenance keys contain unsupported entries")
    typed_provenance = DenseSemanticProvenance(**provenance)
    normalized_provenance = asdict(typed_provenance)
    for name, value in metadata.provenance.items():
        if normalized_provenance.get(name) != value:
            raise ValueError(f"dense manifest provenance mismatch: {name}")
    cache_hashes = manifest.get("cache_files_sha256")
    if not isinstance(cache_hashes, dict):
        raise ValueError("dense manifest cache_files_sha256 must be an object")
    expected_names = [f"frame{index:06d}.npz" for index in range(frame_count)]
    if list(cache_hashes) != expected_names:
        raise ValueError("dense manifest cache checksum keys are not canonical")
    expected_entries = set(expected_names) | {_MANIFEST_NAME}
    actual_entries = {path.name for path in output.iterdir()}
    if actual_entries != expected_entries:
        raise ValueError("resume output contains missing or unexpected files")
    for cache_index, (source_id, name) in enumerate(zip(source_ids, expected_names)):
        path = output / name
        _regular_output_file(path, name)
        expected_hash = cache_hashes[name]
        if not isinstance(expected_hash, str) or _SHA256_PATTERN.fullmatch(expected_hash) is None:
            raise ValueError(f"dense manifest checksum is invalid: {name}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"dense cache checksum mismatch: {name}")
        dense = load_dense_frame(path, expected_sha256=expected_hash)
        _verify_frame(
            dense,
            cache_index,
            source_id,
            preflight.image_shape,
            metadata,
        )
    prefix_hash = _cache_prefix_sha256(cache_hashes)
    if typed_provenance.cache_prefix_sha256 != prefix_hash:
        raise ValueError("dense manifest cache_prefix_sha256 mismatch")
    return manifest


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=_positive_int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--worker-script", type=Path)
    parser.add_argument("--backend", choices=("radseg", "naradio"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--radio-root", type=Path)
    parser.add_argument("--model-version")
    parser.add_argument("--lang-model")
    parser.add_argument("--language-model-root", type=Path)
    parser.add_argument("--language-model-id")
    parser.add_argument("--language-model-revision")
    parser.add_argument("--language-model-sha256")
    parser.add_argument("--classes-json", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--sample-stride", type=_positive_int)
    parser.add_argument("--top-k", type=_positive_int)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--sam-refinement", action="store_true", default=None)
    parser.add_argument("--sam-checkpoint", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested_frames = _strict_int(args.num_frames, "num_frames", positive=True)
    config_path = _absolute_without_resolving(Path(args.config))
    output = _absolute_without_resolving(Path(args.output))
    preflight = _preflight(config_path, requested_frames, args)

    if _lexists(output) and not args.resume:
        raise FileExistsError(output)
    if not _lexists(output) and args.resume:
        raise FileNotFoundError(output)
    if output.is_symlink():
        if args.resume:
            raise ValueError("resume output cannot be a symlink")
        raise FileExistsError(output)
    _reject_symlink_components(
        output if args.resume else output.parent,
        "output path",
    )

    client = JsonLineWorkerClient(
        command=list(preflight.worker.command),
        env=dict(preflight.worker.env),
        cwd=str(preflight.worker.cwd),
        request_timeout_sec=preflight.worker.request_timeout_sec,
        max_response_chars=_MAX_WORKER_RESPONSE_CHARS,
    )
    try:
        metadata = _metadata(
            _request(client, {"operation": "metadata"}, "metadata request"),
            preflight,
        )
        if args.resume:
            return _verify_resume(
                output,
                requested_frames,
                preflight,
                metadata,
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.parent.is_symlink():
            raise ValueError("output parent cannot be a symlink")
        output.mkdir(exist_ok=False)
        _fsync_directory(output.parent)
        cache_hashes: dict[str, str] = {}
        for cache_index in range(requested_frames):
            source_id = preflight.source_frame_ids[cache_index]
            rgb = preflight.rgb_frames[cache_index]
            if tuple(rgb.shape[:2]) != preflight.image_shape:
                raise ValueError(f"dataset frame {cache_index} image shape changed after preflight")
            response = _request(
                client,
                {
                    "operation": "infer",
                    "cache_frame_id": cache_index,
                    "source_frame_id": source_id,
                    "rgb": _encode_rgb(rgb),
                },
                f"frame {cache_index}",
            )
            dense = _decode_inference(
                response,
                metadata,
                preflight.image_shape,
                cache_index,
                source_id,
            )
            name = f"frame{cache_index:06d}.npz"
            path = output / name
            write_dense_frame(path, dense)
            digest = sha256_file(path)
            restored = load_dense_frame(path, expected_sha256=digest)
            _verify_frame(
                restored,
                cache_index,
                source_id,
                preflight.image_shape,
                metadata,
            )
            cache_hashes[name] = digest

        manifest = _completed_manifest(
            preflight,
            metadata,
            requested_frames,
            cache_hashes,
        )
        _atomic_json(output / _MANIFEST_NAME, manifest)
        return manifest
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    manifest = run(parse_args(argv))
    print(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
