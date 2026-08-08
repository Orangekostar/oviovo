#!/usr/bin/env python3
"""Build a temporal-only TESSE-CD frontend cache from frozen sources."""

from __future__ import annotations

import argparse
import ctypes
import errno
import gzip
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_FRAME_SUFFIX = ".pkl.gz"
_MANIFEST_NAME = "frontend_manifest.json"
_MAX_ALIAS_PER_FRAME = 32
_IOU_THRESHOLD = 0.5
_CONTAINMENT_THRESHOLD = 0.8
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_SHA256_CHARS = frozenset("0123456789abcdef")
_FRAME_RANGE_PATTERN = re.compile(r"(0|[1-9][0-9]*):(0|[1-9][0-9]*)")
_MODEL_CONFIG_KEYS = (
    "yolo_model_path",
    "mobile_sam_model_path",
    "clip_pretrained_path",
)
_CANONICAL_CLASSES_BY_SCENE = {
    "apartment": (
        "Fridge",
        "Books",
        "Chair",
        "Vase",
        "Couch",
        "Drawer",
        "Objects",
        "Table",
        "Bin",
        "Humans",
    ),
    "office": (
        "Small office objects",
        "Large static wall furniture",
        "Large office objects",
        "Bathroom",
        "Bedroom",
        "Chairs",
        "Signs",
    ),
}
_REQUIRED_FRAME_KEYS = frozenset(
    {
        "xyxy",
        "confidence",
        "class_id",
        "mask",
        "classes",
        "image_feats",
        "text_feats",
    }
)
_ALLOWED_FRAME_KEYS = _REQUIRED_FRAME_KEYS | {"image_crops"}
_MERGE_ALGORITHM = {
    "schema_version": 1,
    "name": "oviv2-tesse-temporal-alias-merge",
    "canonical_schema": "frozen-source",
    "alias_order": "confidence-descending-original-index-ascending",
    "selection_order": "confidence-descending-canonical-first-original-index",
    "same_class_iou_threshold": _IOU_THRESHOLD,
    "same_class_containment_threshold": _CONTAINMENT_THRESHOLD,
    "maximum_alias_per_frame": _MAX_ALIAS_PER_FRAME,
}


@dataclass(frozen=True)
class _SourceManifest:
    directory: Path
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    scene: str
    frame_count: int
    source_frame_ids: tuple[int, ...]
    image_shape: tuple[int, int]
    classes: tuple[str, ...]
    vocabulary_sha256: str
    feature_model_id: str
    algorithm_hash: str
    cache_files_sha256: Mapping[str, str]


@dataclass(frozen=True)
class _AliasShard:
    directory: Path
    start: int
    end: int
    classes_path: Path
    classes_sha256: str
    config_path: Path
    config_sha256: str
    config: Mapping[str, Any]
    classes: tuple[str, ...]
    cache_files_sha256: Mapping[str, str]
    models: Mapping[str, tuple[Path, str]]
    generation_settings_hash: str


@dataclass(frozen=True)
class _Frame:
    masks: np.ndarray
    boxes: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray
    classes: tuple[str, ...]
    image_features: np.ndarray
    text_features: np.ndarray


@dataclass(frozen=True)
class _Entry:
    source: str
    original_index: int
    class_id: int
    mask: np.ndarray
    box: np.ndarray
    confidence: float
    image_feature: np.ndarray
    text_feature: np.ndarray


class _IgnoredImageCrop:
    def __new__(  # noqa: PYI034 -- typing.Self requires Python 3.11.
        cls, *_args: object, **_kwargs: object
    ) -> _IgnoredImageCrop:
        return super().__new__(cls)

    def __setstate__(self, _state: object) -> None:
        return None


class _RestrictedUnpickler(pickle.Unpickler):
    _ALLOWED_GLOBALS = frozenset(
        {
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"),
        }
    )

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("PIL.Image", "Image"):
            return _IgnoredImageCrop
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"unsafe pickle global is not permitted: {module}.{name}"
            )
        return super().find_class(module, name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge canonical and aliased OVIV2 TESSE-CD caches for temporal use."
        )
    )
    parser.add_argument("--canonical-cache-dir", type=Path, required=True)
    parser.add_argument("--alias-cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--alias-frame-range", action="append", required=True)
    parser.add_argument("--alias-classes", type=Path, action="append", required=True)
    parser.add_argument(
        "--alias-config-params", type=Path, action="append", required=True
    )
    parser.add_argument("--alias-classes-txt", type=Path, required=True)
    parser.add_argument("--alias-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: object, field: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer")  # noqa: TRY004
    result = int(value)
    if result < (1 if positive else 0):
        raise ValueError(f"{field} has an invalid value")
    return result


def _classes(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result = tuple(_string(item, f"{field} item") for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique names")
    return result


def _shape(value: object, field: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be [height, width]")
    return (
        _integer(value[0], f"{field}[0]", positive=True),
        _integer(value[1], f"{field}[1]", positive=True),
    )


def _path_has_symlink(path: Path) -> bool:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            metadata = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            return True
    return False


def _read_regular(path: Path, field: str) -> tuple[bytes, str]:
    if _path_has_symlink(path):
        raise ValueError(f"{field} must not be a symlink or reside below one")
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(path) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{field} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            snapshot = stream.read()
        after = os.stat(path, follow_symlinks=False)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if not stat.S_ISREG(after.st_mode) or identity_before != identity_after:
        raise ValueError(f"{field} changed while it was read")
    return snapshot, hashlib.sha256(snapshot).hexdigest()


def _hash_regular(path: Path, field: str) -> str:
    if _path_has_symlink(path):
        raise ValueError(f"{field} must not be a symlink or reside below one")
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(path) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{field} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.stat(path, follow_symlinks=False)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if not stat.S_ISREG(after.st_mode) or identity_before != identity_after:
        raise ValueError(f"{field} changed while it was read")
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _decode_json(snapshot: bytes, field: str) -> Any:
    try:
        return json.loads(snapshot.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must be valid UTF-8 JSON") from exc


def _decode_json_object(snapshot: bytes, field: str) -> dict[str, Any]:
    value = _decode_json(snapshot, field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} root must be an object")  # noqa: TRY004
    return value


def _cache_hashes(value: object, frame_count: int, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")  # noqa: TRY004
    expected = [f"frame{index:06d}{_FRAME_SUFFIX}" for index in range(frame_count)]
    if set(value) != set(expected):
        raise ValueError(f"{field} must exactly cover every frame")
    return {
        name: _require_sha256(value[name], f"{field}[{name!r}]") for name in expected
    }


def _cache_prefix(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for cache_index, checksum in enumerate(hashes.values()):
        digest.update(cache_index.to_bytes(8, "little", signed=False))
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def _load_source_manifest(directory: Path, role: str) -> _SourceManifest:
    directory = Path(os.path.abspath(directory.expanduser()))
    if _path_has_symlink(directory):
        raise ValueError(f"{role} cache directory must not contain symlinks")
    if not directory.is_dir():
        raise ValueError(f"{role} cache directory must be a directory")
    manifest_path = directory / _MANIFEST_NAME
    snapshot, manifest_sha256 = _read_regular(manifest_path, f"{role} manifest")
    manifest = _decode_json_object(snapshot, f"{role} manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{role} manifest schema_version must be 1")
    if manifest.get("method") != "OVIV2":
        raise ValueError(f"{role} manifest method must be OVIV2")
    if manifest.get("dataset") != "TESSE-CD":
        raise ValueError(f"{role} manifest dataset must be TESSE-CD")
    scene = _string(manifest.get("scene"), f"{role} manifest scene")
    frame_count = _integer(
        manifest.get("frame_count"), f"{role} manifest frame_count", positive=True
    )
    source_ids_value = manifest.get("source_frame_ids")
    if not isinstance(source_ids_value, list) or len(source_ids_value) != frame_count:
        raise ValueError(f"{role} manifest source_frame_ids length mismatch")
    source_frame_ids = tuple(
        _integer(value, f"{role} source_frame_id") for value in source_ids_value
    )
    if len(source_frame_ids) != len(set(source_frame_ids)):
        raise ValueError(f"{role} source_frame_ids must be unique")
    source_hash = manifest.get("source_frame_ids_hash")
    if source_hash is not None and source_hash != _json_hash(list(source_frame_ids)):
        raise ValueError(f"{role} source_frame_ids_hash mismatch")
    image_shape = _shape(manifest.get("image_shape"), f"{role} image_shape")
    classes = _classes(manifest.get("classes"), f"{role} classes")
    if manifest.get("class_count") != len(classes):
        raise ValueError(f"{role} class_count disagrees with classes")
    vocabulary_sha256 = _require_sha256(
        manifest.get("vocabulary_sha256"), f"{role} vocabulary_sha256"
    )
    feature_model_id = _string(
        manifest.get("feature_model_id"), f"{role} feature_model_id"
    )
    algorithm_hash = _require_sha256(
        manifest.get("algorithm_hash"), f"{role} algorithm_hash"
    )
    hashes = _cache_hashes(
        manifest.get("cache_files_sha256"),
        frame_count,
        f"{role} cache_files_sha256",
    )
    declared_prefix = manifest.get("cache_prefix_sha256")
    if declared_prefix is not None and declared_prefix != _cache_prefix(hashes):
        raise ValueError(f"{role} cache_prefix_sha256 mismatch")
    actual_names = sorted(path.name for path in directory.glob(f"*{_FRAME_SUFFIX}"))
    if actual_names != list(hashes):
        raise ValueError(f"{role} cache contains missing or unexpected frame files")
    return _SourceManifest(
        directory=directory,
        path=manifest_path,
        sha256=manifest_sha256,
        payload=manifest,
        scene=scene,
        frame_count=frame_count,
        source_frame_ids=source_frame_ids,
        image_shape=image_shape,
        classes=classes,
        vocabulary_sha256=vocabulary_sha256,
        feature_model_id=feature_model_id,
        algorithm_hash=algorithm_hash,
        cache_files_sha256=hashes,
    )


def _parse_frame_range(value: object, field: str) -> tuple[int, int]:
    if not isinstance(value, str):
        raise ValueError(f"{field} must use START:END")  # noqa: TRY004
    match = _FRAME_RANGE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"{field} must use START:END with non-negative integers")
    start, end = (int(item) for item in match.groups())
    if end < start:
        raise ValueError(f"{field} end must not precede start")
    return start, end


def _repeated_argument(value: object, field: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be repeated for both alias shards")
    return tuple(value)


def _alias_argument_groups(
    args: argparse.Namespace, frame_count: int
) -> tuple[tuple[Path, int, int, Path, Path], ...]:
    directories = _repeated_argument(args.alias_cache_dir, "--alias-cache-dir")
    raw_ranges = _repeated_argument(args.alias_frame_range, "--alias-frame-range")
    classes_paths = _repeated_argument(args.alias_classes, "--alias-classes")
    config_paths = _repeated_argument(args.alias_config_params, "--alias-config-params")
    counts = {len(directories), len(raw_ranges), len(classes_paths), len(config_paths)}
    if counts != {2}:
        raise ValueError(
            "exactly two complete alias shard argument groups are required"
        )
    groups = []
    for index, (directory, raw_range, classes_path, config_path) in enumerate(
        zip(directories, raw_ranges, classes_paths, config_paths, strict=True)
    ):
        start, end = _parse_frame_range(raw_range, f"alias shard {index} frame range")
        groups.append(
            (Path(directory), start, end, Path(classes_path), Path(config_path))
        )
    groups.sort(key=lambda item: (item[1], item[2]))
    expected_start = 0
    for _, start, end, _, _ in groups:
        if start != expected_start:
            raise ValueError(
                "alias shard ranges must exactly partition canonical cache indexes"
            )
        expected_start = end + 1
    if expected_start != frame_count:
        raise ValueError(
            "alias shard ranges must exactly partition canonical cache indexes"
        )
    absolute_directories = [
        Path(os.path.abspath(directory.expanduser()))
        for directory, _, _, _, _ in groups
    ]
    if len(set(absolute_directories)) != len(absolute_directories):
        raise ValueError("alias shard cache directories must be distinct")
    return tuple(groups)


def _load_classes_txt(path: Path) -> tuple[tuple[str, ...], str, Path]:
    absolute = Path(os.path.abspath(path.expanduser()))
    snapshot, digest = _read_regular(absolute, "alias classes TXT")
    try:
        text = snapshot.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("alias classes TXT must be valid UTF-8") from exc
    raw_lines = text.splitlines()
    if not raw_lines or any(not line.strip() for line in raw_lines):
        raise ValueError("alias classes TXT must contain only non-empty class lines")
    classes = tuple(line.strip() for line in raw_lines)
    if len(classes) != len(set(classes)):
        raise ValueError("alias classes TXT must contain unique names")
    return classes, digest, absolute


def _load_classes_json(path: Path, role: str) -> tuple[tuple[str, ...], str, Path]:
    absolute = Path(os.path.abspath(path.expanduser()))
    snapshot, digest = _read_regular(absolute, f"{role} classes JSON")
    value = _decode_json(snapshot, f"{role} classes JSON")
    return _classes(value, f"{role} classes JSON"), digest, absolute


def _validate_alias_config(
    path: Path,
    *,
    role: str,
    canonical: _SourceManifest,
    start: int,
    end: int,
    classes_txt_path: Path,
    model_hash_cache: dict[Path, str],
) -> tuple[dict[str, Any], str, Path, dict[str, tuple[Path, str]], str]:
    absolute = Path(os.path.abspath(path.expanduser()))
    snapshot, digest = _read_regular(absolute, f"{role} config params")
    config = _decode_json_object(snapshot, f"{role} config params")
    if _string(config.get("scene_id"), f"{role} scene_id") != canonical.scene:
        raise ValueError(f"{role} scene_id disagrees with canonical scene")
    config_start = _integer(config.get("start"), f"{role} start")
    config_end = _integer(config.get("end"), f"{role} end", positive=True)
    if config_start != start or config_end != end + 1:
        raise ValueError(
            f"{role} config frame range disagrees with declared frame range"
        )
    if _integer(config.get("stride"), f"{role} stride", positive=True) != 1:
        raise ValueError(f"{role} config stride must be 1")
    config_shape = (
        _integer(config.get("desired_height"), f"{role} desired_height", positive=True),
        _integer(config.get("desired_width"), f"{role} desired_width", positive=True),
    )
    if config_shape != canonical.image_shape:
        raise ValueError(
            f"{role} config image shape disagrees with canonical image shape"
        )
    raw_classes_file = _string(config.get("classes_file"), f"{role} classes_file")
    config_classes_file = Path(os.path.abspath(Path(raw_classes_file).expanduser()))
    if config_classes_file != classes_txt_path:
        raise ValueError(f"{role} config classes_file disagrees with alias classes TXT")
    for field, expected in (
        ("class_set", None),
        ("add_bg_classes", False),
        ("accumu_classes", False),
    ):
        if config.get(field) is not expected:
            raise ValueError(f"{role} config {field} must be {expected!r}")
    variant = _string(config.get("gsa_variant"), f"{role} gsa_variant")
    if _string(config.get("exp_suffix"), f"{role} exp_suffix") != variant:
        raise ValueError(f"{role} config gsa_variant and exp_suffix must agree")
    models: dict[str, tuple[Path, str]] = {}
    for key in _MODEL_CONFIG_KEYS:
        model_path = Path(
            os.path.abspath(
                Path(_string(config.get(key), f"{role} {key}")).expanduser()
            )
        )
        model_hash = model_hash_cache.get(model_path)
        if model_hash is None:
            model_hash = _hash_regular(model_path, f"{role} {key}")
            model_hash_cache[model_path] = model_hash
        models[key] = (model_path, model_hash)
    normalized_settings = {
        key: value
        for key, value in config.items()
        if key not in {"start", "end", "gsa_variant", "exp_suffix"}
    }
    return config, digest, absolute, models, _json_hash(normalized_settings)


def _load_alias_shards(
    args: argparse.Namespace, canonical: _SourceManifest
) -> tuple[tuple[_AliasShard, ...], tuple[str, ...], str, Path]:
    groups = _alias_argument_groups(args, canonical.frame_count)
    classes_txt, classes_txt_sha256, classes_txt_path = _load_classes_txt(
        Path(args.alias_classes_txt)
    )
    model_hash_cache: dict[Path, str] = {}
    shards: list[_AliasShard] = []
    for index, (directory, start, end, classes_path, config_path) in enumerate(groups):
        role = f"alias shard {index}"
        absolute_directory = Path(os.path.abspath(directory.expanduser()))
        if _path_has_symlink(absolute_directory):
            raise ValueError(f"{role} cache directory must not contain symlinks")
        if not absolute_directory.is_dir():
            raise ValueError(f"{role} cache directory must be a directory")
        expected_names = [
            f"frame{cache_index:06d}{_FRAME_SUFFIX}"
            for cache_index in range(start, end + 1)
        ]
        actual_names = sorted(path.name for path in absolute_directory.iterdir())
        if actual_names != expected_names:
            raise ValueError(
                f"{role} cache must exactly cover its declared frame range"
            )
        classes, classes_sha256, absolute_classes_path = _load_classes_json(
            classes_path, role
        )
        if classes != classes_txt:
            raise ValueError(f"{role} classes JSON disagrees with alias classes TXT")
        config, config_sha256, absolute_config_path, models, settings_hash = (
            _validate_alias_config(
                config_path,
                role=role,
                canonical=canonical,
                start=start,
                end=end,
                classes_txt_path=classes_txt_path,
                model_hash_cache=model_hash_cache,
            )
        )
        frame_hashes = {
            name: _hash_regular(absolute_directory / name, f"{role} frame {name}")
            for name in expected_names
        }
        shards.append(
            _AliasShard(
                directory=absolute_directory,
                start=start,
                end=end,
                classes_path=absolute_classes_path,
                classes_sha256=classes_sha256,
                config_path=absolute_config_path,
                config_sha256=config_sha256,
                config=config,
                classes=classes,
                cache_files_sha256=frame_hashes,
                models=models,
                generation_settings_hash=settings_hash,
            )
        )
    first = shards[0]
    for shard in shards[1:]:
        if shard.classes != first.classes:
            raise ValueError("alias shard classes disagree")
        if shard.generation_settings_hash != first.generation_settings_hash:
            raise ValueError("alias shard generation settings disagree")
        if {key: value[1] for key, value in shard.models.items()} != {
            key: value[1] for key, value in first.models.items()
        }:
            raise ValueError("alias shard model artifacts disagree")
    clip_hash = first.models["clip_pretrained_path"][1]
    if canonical.feature_model_id != f"clip-sha256:{clip_hash}":
        raise ValueError("alias CLIP model disagrees with canonical feature_model_id")
    return tuple(shards), classes_txt, classes_txt_sha256, classes_txt_path


def _numeric_array(value: object, field: str, *, integer: bool = False) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{field} must be a NumPy array")  # noqa: TRY004
    kinds = "iu" if integer else "iufb"
    if value.dtype.kind not in kinds:
        raise ValueError(f"{field} has an unsafe dtype")
    return value


def _validate_frame(
    payload: object,
    *,
    expected_classes: tuple[str, ...],
    image_shape: tuple[int, int],
    role: str,
) -> _Frame:
    if not isinstance(payload, dict):
        raise ValueError(f"{role} payload must be a dictionary")  # noqa: TRY004
    keys = set(payload)
    if not _REQUIRED_FRAME_KEYS <= keys or not keys <= _ALLOWED_FRAME_KEYS:
        raise ValueError(f"{role} payload has invalid fields")
    classes = _classes(payload["classes"], f"{role} classes")
    if classes != expected_classes:
        raise ValueError(f"{role} classes disagree with manifest classes")
    masks = _numeric_array(payload["mask"], f"{role} mask")
    boxes = _numeric_array(payload["xyxy"], f"{role} xyxy")
    confidences = _numeric_array(payload["confidence"], f"{role} confidence")
    class_ids = _numeric_array(payload["class_id"], f"{role} class_id", integer=True)
    image_features = _numeric_array(payload["image_feats"], f"{role} image_feats")
    text_features = _numeric_array(payload["text_feats"], f"{role} text_feats")
    if masks.ndim != 3 or masks.shape[1:] != image_shape:
        raise ValueError(f"{role} mask shape disagrees with manifest image_shape")
    count = masks.shape[0]
    if boxes.shape != (count, 4):
        raise ValueError(f"{role} box rows must align with masks")
    if confidences.shape != (count,) or class_ids.shape != (count,):
        raise ValueError(f"{role} vector rows must align with masks")
    if count and (np.any(class_ids < 0) or np.any(class_ids >= len(classes))):
        raise ValueError(f"{role} class_id lies outside classes")
    if (
        image_features.ndim != 2
        or text_features.ndim != 2
        or image_features.shape[0] != count
        or text_features.shape[0] != count
        or image_features.shape[1] <= 0
        or text_features.shape != image_features.shape
    ):
        raise ValueError(f"{role} feature rows must align and share one dimension")
    if image_features.dtype != np.float32 or text_features.dtype != np.float32:
        raise ValueError(f"{role} CLIP feature rows must use float32")
    for name, array in (
        ("mask", masks),
        ("xyxy", boxes),
        ("confidence", confidences),
        ("image_feats", image_features),
        ("text_feats", text_features),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{role} {name} must be finite")
    for name, array in (("xyxy", boxes), ("confidence", confidences)):
        if np.any(array > _FLOAT32_MAX) or np.any(array < -_FLOAT32_MAX):
            raise ValueError(f"{role} {name} must be representable as float32")
    if np.any(confidences < 0.0):
        raise ValueError(f"{role} confidence must be non-negative")
    if count and np.any(masks.reshape(count, -1).sum(axis=1) == 0):
        raise ValueError(f"{role} masks must not contain empty rows")
    for name, features in (
        ("image_feats", image_features),
        ("text_feats", text_features),
    ):
        if count and np.any(np.linalg.norm(features.astype(np.float64), axis=1) == 0):
            raise ValueError(f"{role} {name} must not contain zero rows")
    crops = payload.get("image_crops")
    if crops is not None and (
        not isinstance(crops, (list, tuple)) or len(crops) != count
    ):
        raise ValueError(f"{role} image_crops must align with masks")
    return _Frame(
        masks=np.asarray(masks, dtype=np.bool_),
        boxes=boxes,
        confidences=confidences,
        class_ids=class_ids,
        classes=classes,
        image_features=image_features,
        text_features=text_features,
    )


def _load_frame_from_cache(
    *,
    directory: Path,
    cache_files_sha256: Mapping[str, str],
    cache_index: int,
    expected_classes: tuple[str, ...],
    image_shape: tuple[int, int],
    role: str,
) -> _Frame:
    name = f"frame{cache_index:06d}{_FRAME_SUFFIX}"
    snapshot, actual = _read_regular(directory / name, f"{role} frame {name}")
    if actual != cache_files_sha256[name]:
        raise ValueError(f"{role} frame checksum mismatch: {name}")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(snapshot), mode="rb") as compressed:
            payload = _RestrictedUnpickler(compressed).load()
            if compressed.read(1):
                raise ValueError(f"{role} frame payload has trailing data")
    except ValueError:
        raise
    except (
        EOFError,
        OSError,
        pickle.PickleError,
        AttributeError,
        ImportError,
        IndexError,
        TypeError,
    ) as exc:
        raise ValueError(f"{role} frame is invalid gzip or pickle data") from exc
    return _validate_frame(
        payload,
        expected_classes=expected_classes,
        image_shape=image_shape,
        role=f"{role} frame {name}",
    )


def _load_frame(source: _SourceManifest, cache_index: int, role: str) -> _Frame:
    return _load_frame_from_cache(
        directory=source.directory,
        cache_files_sha256=source.cache_files_sha256,
        cache_index=cache_index,
        expected_classes=source.classes,
        image_shape=source.image_shape,
        role=role,
    )


def _load_alias_frame(
    shard: _AliasShard,
    cache_index: int,
    *,
    image_shape: tuple[int, int],
) -> _Frame:
    return _load_frame_from_cache(
        directory=shard.directory,
        cache_files_sha256=shard.cache_files_sha256,
        cache_index=cache_index,
        expected_classes=shard.classes,
        image_shape=image_shape,
        role=f"alias shard {shard.start}:{shard.end}",
    )


def _load_alias_map(
    path: Path,
    *,
    alias_classes: tuple[str, ...],
    canonical_classes: tuple[str, ...],
) -> tuple[dict[str, str], str, Path]:
    absolute = Path(os.path.abspath(path.expanduser()))
    snapshot, digest = _read_regular(absolute, "alias map")
    value = _decode_json_object(snapshot, "alias map")
    normalized: dict[str, str] = {}
    for raw_alias, raw_canonical in value.items():
        alias = _string(raw_alias, "alias map key")
        canonical = _string(raw_canonical, f"alias map value for {alias!r}")
        if canonical not in canonical_classes:
            raise ValueError(f"alias map value is not a canonical class: {canonical}")
        normalized[alias] = canonical
    missing = [name for name in alias_classes if name not in normalized]
    extra = [name for name in normalized if name not in alias_classes]
    if missing or extra:
        raise ValueError(
            "alias map must explicitly map every alias class and contain no extra keys"
        )
    return dict(sorted(normalized.items())), digest, absolute


def _entry(
    frame: _Frame,
    index: int,
    *,
    source: str,
    class_id: int,
) -> _Entry:
    return _Entry(
        source=source,
        original_index=index,
        class_id=class_id,
        mask=frame.masks[index],
        box=frame.boxes[index],
        confidence=float(frame.confidences[index]),
        image_feature=frame.image_features[index],
        text_feature=frame.text_features[index],
    )


def _is_duplicate(first: _Entry, second: _Entry) -> bool:
    if first.class_id != second.class_id:
        return False
    intersection = int(np.count_nonzero(first.mask & second.mask))
    if intersection == 0:
        return False
    first_area = int(np.count_nonzero(first.mask))
    second_area = int(np.count_nonzero(second.mask))
    union = first_area + second_area - intersection
    iou = intersection / union
    containment = intersection / min(first_area, second_area)
    return iou >= _IOU_THRESHOLD or containment >= _CONTAINMENT_THRESHOLD


def _merge_frame(
    canonical: _Frame,
    alias: _Frame,
    *,
    canonical_classes: tuple[str, ...],
    alias_mapping: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, int]]:
    if canonical.image_features.shape[1] != alias.image_features.shape[1]:
        raise ValueError("canonical and alias feature rows use different dimensions")
    canonical_indices = {name: index for index, name in enumerate(canonical_classes)}
    canonical_entries = [
        _entry(canonical, index, source="canonical", class_id=int(class_id))
        for index, class_id in enumerate(canonical.class_ids)
    ]
    alias_entries = [
        _entry(
            alias,
            index,
            source="alias",
            class_id=canonical_indices[alias_mapping[alias.classes[int(class_id)]]],
        )
        for index, class_id in enumerate(alias.class_ids)
    ]
    alias_entries.sort(key=lambda item: (-item.confidence, item.original_index))
    ordered = sorted(
        (*canonical_entries, *alias_entries),
        key=lambda item: (
            -item.confidence,
            0 if item.source == "canonical" else 1,
            item.original_index,
        ),
    )
    selected: list[_Entry] = []
    alias_selected = 0
    deduplicated = 0
    cap_dropped = 0
    for candidate in ordered:
        if any(_is_duplicate(candidate, retained) for retained in selected):
            deduplicated += 1
            continue
        if candidate.source == "alias" and alias_selected >= _MAX_ALIAS_PER_FRAME:
            cap_dropped += 1
            continue
        selected.append(candidate)
        alias_selected += candidate.source == "alias"
    canonical_output = sorted(
        (item for item in selected if item.source == "canonical"),
        key=lambda item: item.original_index,
    )
    alias_output = sorted(
        (item for item in selected if item.source == "alias"),
        key=lambda item: (-item.confidence, item.original_index),
    )
    output = [*canonical_output, *alias_output]
    feature_dimension = canonical.image_features.shape[1]
    if output:
        masks = np.stack([item.mask for item in output]).astype(np.bool_, copy=False)
        boxes = np.stack([item.box for item in output]).astype(np.float32, copy=False)
        confidences = np.asarray([item.confidence for item in output], dtype=np.float32)
        class_ids = np.asarray([item.class_id for item in output], dtype=np.int64)
        image_features = np.stack([item.image_feature for item in output]).astype(
            np.float32, copy=False
        )
        text_features = np.stack([item.text_feature for item in output]).astype(
            np.float32, copy=False
        )
    else:
        masks = np.empty((0, *canonical.masks.shape[1:]), dtype=np.bool_)
        boxes = np.empty((0, 4), dtype=np.float32)
        confidences = np.empty((0,), dtype=np.float32)
        class_ids = np.empty((0,), dtype=np.int64)
        image_features = np.empty((0, feature_dimension), dtype=np.float32)
        text_features = np.empty((0, feature_dimension), dtype=np.float32)
    payload: dict[str, object] = {
        "xyxy": boxes,
        "confidence": confidences,
        "class_id": class_ids,
        "mask": masks,
        "classes": list(canonical_classes),
        "image_feats": image_features,
        "text_feats": text_features,
    }
    return payload, {
        "canonical_input": len(canonical_entries),
        "alias_input": len(alias_entries),
        "canonical_selected": len(canonical_output),
        "alias_selected": len(alias_output),
        "deduplicated": deduplicated,
        "alias_cap_dropped": cap_dropped,
        "output": len(output),
    }


def _write_frame(path: Path, payload: Mapping[str, object]) -> str:
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            pickle.dump(dict(payload), compressed, protocol=4, fix_imports=False)
        raw.flush()
        os.fsync(raw.fileno())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_file(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_no_replace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-clobber directory publication is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(target)
        raise OSError(error_number, os.strerror(error_number), target)
    _fsync_directory(target.parent)


def _source_record(source: _SourceManifest) -> dict[str, object]:
    return {
        "directory_name": source.directory.name,
        "manifest_name": source.path.name,
        "manifest_sha256": source.sha256,
        "algorithm_hash": source.algorithm_hash,
        "cache_files_sha256": dict(source.cache_files_sha256),
        "cache_prefix_sha256": _cache_prefix(source.cache_files_sha256),
    }


def _alias_source_record(shard: _AliasShard) -> dict[str, object]:
    return {
        "directory_name": shard.directory.name,
        "frame_range": {
            "start": shard.start,
            "end": shard.end,
            "convention": "closed",
        },
        "cache_files_sha256": dict(shard.cache_files_sha256),
        "cache_prefix_sha256": _cache_prefix(shard.cache_files_sha256),
        "classes": {
            "file_name": shard.classes_path.name,
            "sha256": shard.classes_sha256,
        },
        "config_params": {
            "file_name": shard.config_path.name,
            "sha256": shard.config_sha256,
            "generation_settings_sha256": shard.generation_settings_hash,
        },
        "models": {
            key: {"file_name": path.name, "sha256": digest}
            for key, (path, digest) in sorted(shard.models.items())
        },
    }


def _validate_canonical(canonical: _SourceManifest) -> None:
    expected_classes = _CANONICAL_CLASSES_BY_SCENE.get(canonical.scene)
    if expected_classes is None:
        raise ValueError(f"canonical cache scene is unsupported: {canonical.scene}")
    if canonical.classes != expected_classes:
        raise ValueError(
            "canonical cache must provide the exact "
            f"{len(expected_classes)}-class {canonical.scene} schema"
        )
    if canonical.source_frame_ids != tuple(range(canonical.frame_count)):
        raise ValueError("canonical cache must use the identity frame schedule")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not isinstance(args, argparse.Namespace):
        raise TypeError("args must be an argparse.Namespace")
    output = Path(os.path.abspath(Path(args.output).expanduser()))
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if _path_has_symlink(output.parent):
        raise ValueError("output parent must not be a symlink or reside below one")

    canonical = _load_source_manifest(Path(args.canonical_cache_dir), "canonical")
    _validate_canonical(canonical)
    alias_shards, alias_classes, alias_classes_txt_sha256, alias_classes_txt_path = (
        _load_alias_shards(args, canonical)
    )
    alias_mapping, alias_map_sha256, alias_map_path = _load_alias_map(
        Path(args.alias_map),
        alias_classes=alias_classes,
        canonical_classes=canonical.classes,
    )
    algorithm_hash = _json_hash(_MERGE_ALGORITHM)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    )
    cache_hashes: dict[str, str] = {}
    diagnostics = {
        "canonical_input": 0,
        "alias_input": 0,
        "canonical_selected": 0,
        "alias_selected": 0,
        "deduplicated": 0,
        "alias_cap_dropped": 0,
        "output": 0,
    }
    published = False
    try:
        shard_index = 0
        for cache_index in range(canonical.frame_count):
            while cache_index > alias_shards[shard_index].end:
                shard_index += 1
            alias_shard = alias_shards[shard_index]
            canonical_frame = _load_frame(canonical, cache_index, "canonical")
            alias_frame = _load_alias_frame(
                alias_shard, cache_index, image_shape=canonical.image_shape
            )
            payload, frame_diagnostics = _merge_frame(
                canonical_frame,
                alias_frame,
                canonical_classes=canonical.classes,
                alias_mapping=alias_mapping,
            )
            name = f"frame{cache_index:06d}{_FRAME_SUFFIX}"
            cache_hashes[name] = _write_frame(staging / name, payload)
            for key, value in frame_diagnostics.items():
                diagnostics[key] += value

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "method": "OVIV2",
            "dataset": "TESSE-CD",
            "scene": canonical.scene,
            "frame_count": canonical.frame_count,
            "source_frame_ids": list(canonical.source_frame_ids),
            "source_frame_ids_hash": _json_hash(list(canonical.source_frame_ids)),
            "image_shape": list(canonical.image_shape),
            "class_count": len(canonical.classes),
            "classes": list(canonical.classes),
            "vocabulary_sha256": canonical.vocabulary_sha256,
            "algorithm_hash": algorithm_hash,
            "feature_model_id": canonical.feature_model_id,
            "cache_files_sha256": cache_hashes,
            "cache_prefix_sha256": _cache_prefix(cache_hashes),
            "temporal_only": True,
            "alias_map_sha256": alias_map_sha256,
            "merge_algorithm": {**_MERGE_ALGORITHM, "sha256": algorithm_hash},
            "merge_policy": {
                "same_canonical_iou_threshold": _IOU_THRESHOLD,
                "same_canonical_containment_threshold": _CONTAINMENT_THRESHOLD,
                "maximum_alias_per_frame": _MAX_ALIAS_PER_FRAME,
            },
            "diagnostics": diagnostics,
            "temporal_frontend_sources": {
                "canonical": _source_record(canonical),
                "alias_shards": [_alias_source_record(shard) for shard in alias_shards],
                "alias_cache_prefix_sha256": _cache_prefix(
                    {
                        name: digest
                        for shard in alias_shards
                        for name, digest in shard.cache_files_sha256.items()
                    }
                ),
                "alias_classes_txt": {
                    "file_name": alias_classes_txt_path.name,
                    "sha256": alias_classes_txt_sha256,
                },
                "alias_map": {
                    "file_name": alias_map_path.name,
                    "sha256": alias_map_sha256,
                    "mapping": alias_mapping,
                },
            },
        }
        for field in ("input_manifest_sha256", "input_witness", "provenance_sha256"):
            if field in canonical.payload:
                manifest[field] = canonical.payload[field]
        _write_file(staging / _MANIFEST_NAME, _json_bytes(manifest))
        _fsync_directory(staging)
        _publish_directory_no_replace(staging, output)
        published = True
        return manifest
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run(parse_args(argv))
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
