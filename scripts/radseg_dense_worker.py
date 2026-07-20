#!/usr/bin/env python3
"""Persistent JSONL worker for pinned RADSeg and NARADIO dense semantics."""

from __future__ import annotations

import argparse
import ast
import base64
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence, TextIO
from urllib.parse import unquote, urlparse

import numpy as np


RADSEG_COMMIT = "3fe8789a3c1b11e41688f7deadc8b8db088ef1a7"
RAYFRONTS_COMMIT = "031262a9ed4d0ea456a1d5605df835ba9e53b027"
RADIO_COMMIT = "c0f37017930e9dda53f93424cf4bf39fc51f287e"

MAX_RGB_BYTES = 128 * 1024 * 1024
MAX_CLASSES_JSON_BYTES = 1024 * 1024
MAX_JSONL_LINE_CHARS = 16 * 1024 * 1024
MAX_JSONL_RESPONSE_CHARS = 16 * 1024 * 1024
MAX_PROBABILITY_TENSOR_BYTES = 256 * 1024 * 1024
CLASS_COUNT = 41
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_OFFLINE_ENVIRONMENT = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
)
_ARRAY_KEYS = frozenset({"encoding", "dtype", "shape", "data"})
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


@dataclass(frozen=True)
class SourceSpec:
    name: str
    commit: str
    official_origins: tuple[str, ...]


@dataclass(frozen=True)
class LanguageModelSpec:
    revision: str
    backends: frozenset[str]


@dataclass(frozen=True)
class LanguageModelAssets:
    backend: str
    root: Path
    model_id: str
    revision: str
    sha256: str


@dataclass(frozen=True)
class InferenceBudget:
    probability_tensor_bytes: int
    sampled_shape: tuple[int, int]
    encoded_response_chars: int


LANGUAGE_MODEL_SPECS = {
    "google/siglip2-giant-opt-patch16-384": LanguageModelSpec(
        revision="a713301b217d38485fb2204c808367d10bc3cc40",
        backends=frozenset({"radseg"}),
    ),
    "google/siglip2-so400m-patch16-naflex": LanguageModelSpec(
        revision="cc24074f717b612951c2dead130904ab9b65a81e",
        backends=frozenset({"radseg"}),
    ),
    "timm/ViT-SO400M-14-SigLIP-384": LanguageModelSpec(
        revision="ac16108d567c4389e6cd2b11c9b8585f7474435b",
        backends=frozenset({"naradio"}),
    ),
}


RADSEG_SOURCE = SourceSpec(
    name="RADSeg",
    commit=RADSEG_COMMIT,
    official_origins=(
        "https://github.com/RADSeg-OVSS/RADSeg.git",
        "git@github.com:RADSeg-OVSS/RADSeg.git",
    ),
)
RAYFRONTS_SOURCE = SourceSpec(
    name="RayFronts",
    commit=RAYFRONTS_COMMIT,
    official_origins=(
        "https://github.com/RayFronts/RayFronts.git",
        "git@github.com:RayFronts/RayFronts.git",
    ),
)
RADIO_SOURCE = SourceSpec(
    name="RADIO",
    commit=RADIO_COMMIT,
    official_origins=(
        "https://github.com/NVlabs/RADIO.git",
        "git@github.com:NVlabs/RADIO.git",
    ),
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _sha256_argument(value: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a 64-character hexadecimal SHA-256")
    return value.lower()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("radseg", "naradio"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--radio-root", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--lang-model", required=True)
    parser.add_argument("--language-model-root", type=Path, required=True)
    parser.add_argument("--language-model-id", required=True)
    parser.add_argument("--language-model-revision", required=True)
    parser.add_argument(
        "--language-model-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument("--classes-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-stride", type=_positive_int, default=4)
    parser.add_argument("--top-k", type=_positive_int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--sam-refinement", action="store_true")
    parser.add_argument("--sam-checkpoint", type=Path)
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace, classes: Sequence[str]) -> None:
    if args.top_k > len(classes):
        raise ValueError("--top-k cannot exceed the frozen class count")
    if args.sam_refinement and args.backend != "radseg":
        raise ValueError("SAM refinement is supported only by RADSeg")
    if args.sam_refinement and args.sam_checkpoint is None:
        raise ValueError("--sam-checkpoint is required with --sam-refinement")
    if not args.sam_refinement and args.sam_checkpoint is not None:
        raise ValueError("--sam-checkpoint requires --sam-refinement")
    if args.sam_checkpoint is not None:
        checkpoint = args.sam_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError(f"SAM checkpoint is missing: {checkpoint}")
    _validate_language_model_binding(args)


def _validate_language_model_binding(args: argparse.Namespace) -> LanguageModelSpec:
    spec = LANGUAGE_MODEL_SPECS.get(args.language_model_id)
    if spec is None:
        raise ValueError("--language-model-id is not in the pinned official mapping")
    if args.backend not in spec.backends:
        raise ValueError("language model ID is not valid for the selected backend")
    if args.language_model_revision != spec.revision:
        raise ValueError("language model revision does not match the pinned mapping")
    expected_adaptor = "siglip2" if args.backend == "radseg" else "siglip"
    if args.lang_model != expected_adaptor:
        raise ValueError(
            f"--lang-model must be {expected_adaptor!r} for backend {args.backend!r}"
        )
    if _SHA256_PATTERN.fullmatch(args.language_model_sha256) is None:
        raise ValueError("language model SHA-256 must be 64 hexadecimal characters")
    return spec


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tree_record(digest: Any, kind: bytes, relative_path: str, size: int) -> None:
    encoded_path = relative_path.encode("utf-8")
    digest.update(kind)
    digest.update(len(encoded_path).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded_path)
    digest.update(size.to_bytes(8, byteorder="big", signed=False))


def language_model_tree_sha256(path: str | Path) -> str:
    """Hash every directory and regular file; no path is excluded from v1."""

    requested_root = Path(path).expanduser()
    if requested_root.is_symlink():
        raise RuntimeError("language model root must not be a symlink")
    root = requested_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"language model root is missing: {root}")

    entries: list[tuple[str, Path, os.stat_result]] = []
    directories: dict[Path, os.stat_result] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        directory_stat = directory.lstat()
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise RuntimeError(f"language model tree contains a non-directory: {directory}")
        if stat.S_IMODE(directory_stat.st_mode) & 0o222:
            raise RuntimeError(f"language model tree must be recursively read-only: {directory}")
        directories[directory] = directory_stat
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise RuntimeError(f"cannot enumerate language model directory: {directory}") from exc
        for child in children:
            child_stat = child.lstat()
            relative = child.relative_to(root).as_posix()
            if stat.S_ISLNK(child_stat.st_mode):
                raise RuntimeError(f"language model tree must not contain symlinks: {relative}")
            if stat.S_IMODE(child_stat.st_mode) & 0o222:
                raise RuntimeError(
                    f"language model tree must be recursively read-only: {relative}"
                )
            if stat.S_ISDIR(child_stat.st_mode):
                entries.append((relative, child, child_stat))
                pending.append(child)
            elif stat.S_ISREG(child_stat.st_mode):
                entries.append((relative, child, child_stat))
            else:
                raise RuntimeError(
                    "language model tree may contain only directories and regular files; "
                    f"found non-regular entry: {relative}"
                )

    digest = hashlib.sha256()
    digest.update(b"OVIV2-language-model-tree-v1\0")
    for relative, entry, before in sorted(entries, key=lambda item: item[0]):
        if stat.S_ISDIR(before.st_mode):
            _tree_record(digest, b"D", relative, 0)
            continue
        _tree_record(digest, b"F", relative, before.st_size)
        bytes_read = 0
        with entry.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                bytes_read += len(chunk)
                digest.update(chunk)
        after = entry.lstat()
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if bytes_read != before.st_size or any(
            getattr(before, field) != getattr(after, field) for field in stable_fields
        ):
            raise RuntimeError(f"language model file changed while hashing: {relative}")

    for directory, before in directories.items():
        after = directory.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"language model directory changed while hashing: {directory}")
    return digest.hexdigest()


def validate_language_model_assets(args: argparse.Namespace) -> LanguageModelAssets:
    _validate_language_model_binding(args)
    root = args.language_model_root.expanduser().resolve()
    actual_hash = language_model_tree_sha256(args.language_model_root)
    expected_hash = args.language_model_sha256.lower()
    if actual_hash != expected_hash:
        raise RuntimeError(
            "language model tree SHA-256 mismatch: "
            f"expected {expected_hash}, received {actual_hash}"
        )
    return LanguageModelAssets(
        backend=args.backend,
        root=root,
        model_id=args.language_model_id,
        revision=args.language_model_revision,
        sha256=actual_hash,
    )


_MISSING = object()


@dataclass(frozen=True)
class _AttributePatch:
    owner: Any
    name: str
    previous: Any


def _patch_attribute(owner: Any, name: str, replacement: Any) -> _AttributePatch:
    namespace = vars(owner)
    previous = namespace[name] if name in namespace else _MISSING
    setattr(owner, name, replacement)
    return _AttributePatch(owner=owner, name=name, previous=previous)


def _restore_attribute(patch: _AttributePatch) -> None:
    if patch.previous is _MISSING:
        delattr(patch.owner, patch.name)
    else:
        setattr(patch.owner, patch.name, patch.previous)


def _read_call_argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    index: int,
    name: str,
    *,
    default: Any = _MISSING,
) -> Any:
    positional = len(args) > index
    keyword = name in kwargs
    if positional and keyword:
        raise RuntimeError(f"language loader received duplicate argument {name}")
    if positional:
        return args[index]
    if keyword:
        return kwargs[name]
    if default is _MISSING:
        raise RuntimeError(f"language loader omitted required argument {name}")
    return default


def _replace_call_argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    index: int,
    name: str,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    updated_args = list(args)
    updated_kwargs = dict(kwargs)
    if len(updated_args) > index:
        if name in updated_kwargs:
            raise RuntimeError(f"language loader received duplicate argument {name}")
        updated_args[index] = value
    else:
        updated_kwargs[name] = value
    return tuple(updated_args), updated_kwargs


@contextmanager
def pinned_language_model_load(
    assets: LanguageModelAssets,
    *,
    transformers_module: Any | None = None,
    open_clip_module: Any | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> Iterator[None]:
    spec = LANGUAGE_MODEL_SPECS.get(assets.model_id)
    if (
        spec is None
        or assets.backend not in spec.backends
        or assets.revision != spec.revision
    ):
        raise RuntimeError("language model assets do not match the pinned mapping")
    before_hash = language_model_tree_sha256(assets.root)
    if before_hash != assets.sha256:
        raise RuntimeError("language model tree hash changed before model loading")

    if assets.backend == "radseg":
        if transformers_module is None:
            transformers_module = importlib.import_module("transformers")
    elif assets.backend == "naradio":
        if open_clip_module is None:
            open_clip_module = importlib.import_module("open_clip")
    else:
        raise RuntimeError(f"unsupported language-model backend: {assets.backend}")

    if assets.backend == "radseg":
        getattr(transformers_module.AutoModel, "from_pretrained")
        getattr(transformers_module.AutoProcessor, "from_pretrained")
    else:
        getattr(open_clip_module, "create_model_from_pretrained")
        getattr(open_clip_module, "get_tokenizer")

    environment = os.environ if environ is None else environ
    saved_environment = {
        name: environment[name] if name in environment else _MISSING
        for name in _OFFLINE_ENVIRONMENT
    }
    for name in _OFFLINE_ENVIRONMENT:
        environment[name] = "1"

    patches: list[_AttributePatch] = []
    call_counts: dict[str, int] = {}
    expected_counts: dict[str, int]

    def restore_setup() -> None:
        for patch in reversed(patches):
            _restore_attribute(patch)
        for name, previous in saved_environment.items():
            if previous is _MISSING:
                environment.pop(name, None)
            else:
                environment[name] = previous

    if assets.backend == "radseg":
        assert transformers_module is not None

        def transformer_wrapper(owner: Any, kind: str) -> Any:
            original = getattr(owner, "from_pretrained")

            def load(*args: Any, **kwargs: Any) -> Any:
                if call_counts.get(kind, 0) != 0:
                    raise RuntimeError(f"unexpected extra {kind} from_pretrained request")
                requested_id = _read_call_argument(
                    args,
                    kwargs,
                    0,
                    "pretrained_model_name_or_path",
                )
                if requested_id != assets.model_id:
                    raise RuntimeError(
                        f"{kind} language model ID does not match the pinned CLI ID"
                    )
                requested_revision = kwargs.get("revision")
                if requested_revision is not None and requested_revision != assets.revision:
                    raise RuntimeError(f"{kind} language model revision is not pinned")
                local_args, local_kwargs = _replace_call_argument(
                    args,
                    kwargs,
                    0,
                    "pretrained_model_name_or_path",
                    str(assets.root),
                )
                local_kwargs.pop("revision", None)
                local_kwargs["local_files_only"] = True
                local_kwargs["trust_remote_code"] = False
                call_counts[kind] = 1
                return original(*local_args, **local_kwargs)

            return load

        try:
            for owner, kind in (
                (transformers_module.AutoModel, "AutoModel"),
                (transformers_module.AutoProcessor, "AutoProcessor"),
            ):
                patches.append(
                    _patch_attribute(
                        owner,
                        "from_pretrained",
                        transformer_wrapper(owner, kind),
                    )
                )
        except BaseException:
            restore_setup()
            raise
        expected_counts = {"AutoModel": 1, "AutoProcessor": 1}
    else:
        assert open_clip_module is not None
        original_create = open_clip_module.create_model_from_pretrained
        original_tokenizer = open_clip_module.get_tokenizer
        expected_open_clip_name = "ViT-SO400M-14-SigLIP-384"

        def create_model(*args: Any, **kwargs: Any) -> Any:
            if call_counts.get("create_model_from_pretrained", 0) != 0:
                raise RuntimeError("unexpected extra OpenCLIP model request")
            requested_name = _read_call_argument(args, kwargs, 0, "model_name")
            requested_pretrained = _read_call_argument(
                args,
                kwargs,
                1,
                "pretrained",
                default=None,
            )
            if requested_name != expected_open_clip_name:
                raise RuntimeError("OpenCLIP model_name does not match the pinned request")
            if requested_pretrained != "webli":
                raise RuntimeError("OpenCLIP pretrained tag does not match pinned webli")
            local_args, local_kwargs = _replace_call_argument(
                args,
                kwargs,
                0,
                "model_name",
                f"local-dir:{assets.root}",
            )
            call_counts["create_model_from_pretrained"] = 1
            return original_create(*local_args, **local_kwargs)

        def get_tokenizer(*args: Any, **kwargs: Any) -> Any:
            if call_counts.get("get_tokenizer", 0) != 0:
                raise RuntimeError("unexpected extra OpenCLIP tokenizer request")
            requested_name = _read_call_argument(args, kwargs, 0, "model_name")
            if requested_name != expected_open_clip_name:
                raise RuntimeError("OpenCLIP tokenizer model_name is not pinned")
            local_args, local_kwargs = _replace_call_argument(
                args,
                kwargs,
                0,
                "model_name",
                f"local-dir:{assets.root}",
            )
            call_counts["get_tokenizer"] = 1
            return original_tokenizer(*local_args, **local_kwargs)

        try:
            patches.append(
                _patch_attribute(
                    open_clip_module,
                    "create_model_from_pretrained",
                    create_model,
                )
            )
            patches.append(
                _patch_attribute(open_clip_module, "get_tokenizer", get_tokenizer)
            )
        except BaseException:
            restore_setup()
            raise
        expected_counts = {"create_model_from_pretrained": 1, "get_tokenizer": 1}

    failure: tuple[Any, Any, Any] | None = None
    try:
        yield
    except BaseException:
        failure = sys.exc_info()
    finally:
        restore_setup()

    try:
        after_hash = language_model_tree_sha256(assets.root)
    except Exception as exc:
        if failure is not None:
            raise RuntimeError(
                f"language model tree changed during model loading: {exc}"
            ) from failure[1]
        raise RuntimeError(
            f"language model tree changed during model loading: {exc}"
        ) from exc
    if after_hash != before_hash:
        error = RuntimeError("language model tree changed during model loading")
        if failure is not None:
            raise error from failure[1]
        raise error
    if failure is not None:
        raise failure[1].with_traceback(failure[2])
    if call_counts != expected_counts:
        raise RuntimeError(
            "language adaptor did not make exactly the pinned model and tokenizer requests"
        )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_frozen_classes(path: str | Path) -> tuple[list[str], str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"classes JSON is missing: {source}")
    with source.open("rb") as stream:
        raw = stream.read(MAX_CLASSES_JSON_BYTES + 1)
    if len(raw) > MAX_CLASSES_JSON_BYTES:
        raise ValueError("classes JSON exceeds the size limit")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("classes JSON is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("classes JSON root must be an object")
    if set(payload) - {"classes", "aliases"}:
        raise ValueError("classes JSON contains unsupported keys")
    classes = payload.get("classes")
    if not isinstance(classes, list):
        raise ValueError("classes must be a list")
    if len(classes) != CLASS_COUNT:
        raise ValueError(f"classes must contain exactly {CLASS_COUNT} entries")
    normalized: list[str] = []
    for value in classes:
        if not isinstance(value, str) or not value:
            raise ValueError("classes must contain non-empty strings")
        if value != value.strip():
            raise ValueError("class names must not contain surrounding whitespace")
        normalized.append(value)
    if len({value.casefold() for value in normalized}) != len(normalized):
        raise ValueError("class names must be unique ignoring case")
    aliases = payload.get("aliases", {})
    if not isinstance(aliases, dict):
        raise ValueError("aliases must be an object when present")
    for alias, target in aliases.items():
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("alias names must be non-empty strings")
        if not isinstance(target, str) or target not in normalized:
            raise ValueError("alias targets must name a frozen class")
    return normalized, hashlib.sha256(raw).hexdigest()


def decode_rgb(block: Mapping[str, Any]) -> np.ndarray:
    if not isinstance(block, dict):
        raise ValueError("RGB block must be an object")
    if set(block) != _ARRAY_KEYS:
        raise ValueError("RGB block keys must be encoding, dtype, shape, and data")
    if block["encoding"] != "base64":
        raise ValueError("RGB encoding must be base64")
    if block["dtype"] != "uint8":
        raise ValueError("RGB dtype must be uint8")
    shape = block["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(value) is not int for value in shape)
        or any(value <= 0 for value in shape)
        or shape[2] != 3
    ):
        raise ValueError("RGB shape must be [positive H, positive W, 3]")
    expected_bytes = shape[0] * shape[1] * shape[2]
    if expected_bytes > MAX_RGB_BYTES:
        raise ValueError("RGB payload exceeds the size limit")
    encoded = block["data"]
    if not isinstance(encoded, str):
        raise ValueError("RGB base64 data must be a string")
    maximum_encoded = 4 * ((MAX_RGB_BYTES + 2) // 3)
    if len(encoded) > maximum_encoded:
        raise ValueError("RGB base64 payload exceeds the size limit")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("RGB data is not valid base64") from exc
    if len(raw) != expected_bytes:
        raise ValueError(
            f"RGB byte count mismatch: expected {expected_bytes}, received {len(raw)}"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape)).copy(order="C")


def encode_array(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.int64)):
        raise ValueError("worker output arrays must use float32 or int64")
    contiguous = np.ascontiguousarray(values)
    return {
        "encoding": "base64",
        "dtype": contiguous.dtype.name,
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
    }


def _base64_character_count(byte_count: int) -> int:
    return 4 * ((byte_count + 2) // 3)


def validate_inference_budget(
    *,
    height: int,
    width: int,
    sample_stride: int,
    top_k: int,
) -> InferenceBudget:
    normalized_height = _require_positive_integer(height, "height")
    normalized_width = _require_positive_integer(width, "width")
    stride = _require_positive_integer(sample_stride, "sample_stride")
    candidates = _require_positive_integer(top_k, "top_k")
    if candidates > CLASS_COUNT:
        raise ValueError("top_k cannot exceed the class count")

    pixels = normalized_height * normalized_width
    probability_bytes = pixels * CLASS_COUNT * np.dtype(np.float32).itemsize
    if probability_bytes > MAX_PROBABILITY_TENSOR_BYTES:
        raise ValueError(
            "dense probability tensor exceeds the inference memory budget"
        )
    sampled_height = (normalized_height + stride - 1) // stride
    sampled_width = (normalized_width + stride - 1) // stride
    sampled_pixels = sampled_height * sampled_width
    array_byte_counts = (
        sampled_pixels * candidates * np.dtype(np.int64).itemsize,
        sampled_pixels * candidates * np.dtype(np.float32).itemsize,
        sampled_pixels * np.dtype(np.float32).itemsize,
        sampled_pixels * np.dtype(np.float32).itemsize,
    )
    encoded_response_chars = sum(
        _base64_character_count(byte_count) for byte_count in array_byte_counts
    ) + 8192
    if encoded_response_chars > MAX_JSONL_RESPONSE_CHARS:
        raise ValueError("dense inference response exceeds the JSONL response budget")
    return InferenceBudget(
        probability_tensor_bytes=probability_bytes,
        sampled_shape=(sampled_height, sampled_width),
        encoded_response_chars=encoded_response_chars,
    )


def _require_positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def reduce_probabilities(
    probabilities: np.ndarray,
    sample_stride: int,
    top_k: int,
) -> dict[str, np.ndarray]:
    stride = _require_positive_integer(sample_stride, "sample_stride")
    requested_top_k = _require_positive_integer(top_k, "top_k")
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("probabilities must have shape [1,C,H,W]")
    if any(dimension <= 0 for dimension in values.shape[1:]):
        raise ValueError("probabilities must have non-empty class and spatial dimensions")
    class_count = values.shape[1]
    if requested_top_k > class_count:
        raise ValueError("top_k cannot exceed the class count")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("probabilities must be non-negative")

    sampled = values[0, :, ::stride, ::stride].transpose(1, 2, 0)
    mass = sampled.sum(axis=-1, keepdims=True, dtype=np.float64)
    if np.any(mass <= 0.0):
        raise ValueError("probabilities must have positive probability mass per pixel")
    if np.any(sampled > 1.0) or np.any(mass > 1.0 + 1e-6):
        raise ValueError("probabilities must represent a distribution with mass at most one")
    normalized = np.divide(
        sampled.astype(np.float64),
        mass,
        out=np.zeros_like(sampled, dtype=np.float64),
        where=mass > 0.0,
    )
    order = np.argsort(-sampled, axis=-1, kind="stable")[..., :requested_top_k]
    top = np.take_along_axis(sampled, order, axis=-1)

    entropy_terms = np.zeros_like(normalized, dtype=np.float64)
    positive = normalized > 0.0
    entropy_terms[positive] = normalized[positive] * np.log(normalized[positive])
    entropy = -entropy_terms.sum(axis=-1, dtype=np.float64)
    runner_up = top[..., 1] if requested_top_k > 1 else 0.0
    margin = top[..., 0] - runner_up
    class_ids = order.astype(np.int64) + 1
    class_ids[top <= 0.0] = 0
    return {
        "class_ids": np.ascontiguousarray(class_ids),
        "probabilities": np.ascontiguousarray(top, dtype=np.float32),
        "entropy": np.ascontiguousarray(entropy, dtype=np.float32),
        "margin": np.ascontiguousarray(margin, dtype=np.float32),
    }


@contextmanager
def local_radio_hub(torch_module: ModuleType | Any, radio_root: str | Path) -> Iterator[None]:
    root = Path(radio_root).expanduser().resolve()
    original_load = torch_module.hub.load

    def pinned_load(repo_or_dir: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
        if repo_or_dir != "NVlabs/RADIO":
            raise RuntimeError("all non-pinned torch hub repositories are prohibited")
        local_kwargs = dict(kwargs)
        local_kwargs["source"] = "local"
        return original_load(str(root), model, *args, **local_kwargs)

    torch_module.hub.load = pinned_load
    try:
        yield
    finally:
        torch_module.hub.load = original_load


@dataclass(frozen=True)
class FileFingerprint:
    path: Path
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _fingerprint_file(
    path: str | Path,
    *,
    require_read_only: bool,
) -> FileFingerprint:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise RuntimeError(f"model file must not be a symlink: {requested}")
    resolved = requested.resolve()
    try:
        before = resolved.lstat()
    except OSError as exc:
        raise RuntimeError(f"model file is missing: {resolved}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"model file must be regular: {resolved}")
    if require_read_only and stat.S_IMODE(before.st_mode) & 0o222:
        raise RuntimeError(f"explicit model file must be read-only: {resolved}")
    digest = hashlib.sha256()
    bytes_read = 0
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            bytes_read += len(chunk)
            digest.update(chunk)
    after = resolved.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if bytes_read != before.st_size or any(
        getattr(before, field) != getattr(after, field) for field in stable_fields
    ):
        raise RuntimeError(f"model file changed while hashing: {resolved}")
    return FileFingerprint(
        path=resolved,
        sha256=digest.hexdigest(),
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )


class CheckpointTracker:
    """Capture checkpoint paths requested by one model construction."""

    def __init__(self, hub: Any, *, expected_url: Any = _MISSING) -> None:
        self._hub = hub
        self._expected_url = expected_url
        self._original: Any | None = None
        self._paths: list[Path] = []
        self._fingerprints: dict[Path, FileFingerprint] = {}

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(self._paths))

    @property
    def fingerprints(self) -> Mapping[Path, FileFingerprint]:
        return dict(self._fingerprints)

    def __enter__(self) -> "CheckpointTracker":
        self._original = self._hub.load_state_dict_from_url

        def tracked(url: str, *args: Any, **kwargs: Any) -> Any:
            assert self._original is not None
            if self._expected_url is not _MISSING and url != self._expected_url:
                raise RuntimeError(
                    "model construction requested an unexpected RADIO checkpoint URL"
                )
            filename = kwargs.get("file_name")
            if filename is None:
                filename = Path(unquote(urlparse(str(url)).path)).name
            if not isinstance(filename, str) or not filename or Path(filename).name != filename:
                raise RuntimeError("checkpoint loader returned an invalid cache filename")
            model_dir = kwargs.get("model_dir")
            directory = (
                Path(model_dir).expanduser()
                if model_dir is not None
                else Path(self._hub.get_dir()) / "checkpoints"
            )
            checkpoint_path = (directory / filename).resolve()
            before = (
                _fingerprint_file(checkpoint_path, require_read_only=False)
                if checkpoint_path.is_file()
                else None
            )
            result = self._original(url, *args, **kwargs)
            after = _fingerprint_file(checkpoint_path, require_read_only=False)
            if before is not None and before != after:
                raise RuntimeError("cached RADIO weight changed during checkpoint loading")
            self._paths.append(checkpoint_path)
            self._fingerprints[checkpoint_path] = after
            return result

        self._hub.load_state_dict_from_url = tracked
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original is not None:
            self._hub.load_state_dict_from_url = self._original


def _expected_radio_checkpoint_url(
    radio_root: str | Path,
    model_version: str,
) -> str | None:
    if Path(model_version).expanduser().is_file():
        return None
    common_path = Path(radio_root).expanduser().resolve() / "radio" / "common.py"
    try:
        source = common_path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(common_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise RuntimeError("cannot parse pinned RADIO resource map") from exc
    resource_dict: ast.Dict | None = None
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "RESOURCE_MAP" for target in node.targets):
            if not isinstance(node.value, ast.Dict):
                raise RuntimeError("pinned RADIO RESOURCE_MAP is not a literal mapping")
            resource_dict = node.value
            break
    if resource_dict is None:
        raise RuntimeError("pinned RADIO source does not define RESOURCE_MAP")
    urls: dict[str, str] = {}
    for key_node, value_node in zip(resource_dict.keys, resource_dict.values):
        if (
            not isinstance(key_node, ast.Constant)
            or not isinstance(key_node.value, str)
            or not isinstance(value_node, ast.Call)
            or not value_node.args
            or not isinstance(value_node.args[0], ast.Constant)
            or not isinstance(value_node.args[0].value, str)
        ):
            raise RuntimeError("pinned RADIO RESOURCE_MAP contains a dynamic entry")
        if key_node.value in urls:
            raise RuntimeError("pinned RADIO RESOURCE_MAP contains a duplicate version")
        urls[key_node.value] = value_node.args[0].value
    try:
        return urls[model_version]
    except KeyError as exc:
        raise RuntimeError(f"model version is absent from pinned RADIO source: {model_version}") from exc


def resolve_model_checkpoint(
    captured_paths: Iterable[str | Path], model_version: str
) -> Path:
    candidates = [Path(path).expanduser().resolve() for path in captured_paths]
    explicit = Path(model_version).expanduser()
    if explicit.is_file():
        candidates.append(explicit.resolve())
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            "model construction must identify exactly one actual checkpoint; "
            f"identified {len(candidates)}"
        )
    checkpoint = candidates[0]
    if not checkpoint.is_file():
        raise RuntimeError(f"identified model checkpoint is missing: {checkpoint}")
    return checkpoint


def _run_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _assert_tree_read_only(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            target = path.resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"source tree contains an external symlink: {path}") from exc
            continue
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError as exc:
            raise RuntimeError(f"cannot inspect source-tree permissions: {path}") from exc
        if mode & 0o222:
            raise RuntimeError(f"source checkout must be recursively read-only: {path}")


def validate_source_checkout(path: str | Path, spec: SourceSpec) -> str:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"{spec.name} source root is missing: {root}")
    try:
        head = _run_git(root, "rev-parse", "HEAD").stdout.strip().lower()
        origin = _run_git(root, "config", "--get", "remote.origin.url").stdout.strip()
        status_output = _run_git(
            root, "status", "--porcelain=v1", "--untracked-files=all"
        ).stdout
        symbolic = _run_git(root, "symbolic-ref", "-q", "HEAD", check=False)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{spec.name} source root is not a valid Git checkout") from exc
    if head != spec.commit.lower():
        raise RuntimeError(
            f"{spec.name} commit mismatch: expected {spec.commit}, received {head}"
        )
    if symbolic.returncode == 0:
        raise RuntimeError(f"{spec.name} checkout must use detached HEAD")
    if symbolic.returncode not in (0, 1):
        raise RuntimeError(f"cannot verify detached HEAD for {spec.name}")
    if status_output:
        raise RuntimeError(f"{spec.name} checkout must be clean")
    if origin not in spec.official_origins:
        raise RuntimeError(f"{spec.name} checkout does not use an official origin")
    _assert_tree_read_only(root)
    return head


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


@dataclass
class RadsegRuntime:
    encoder: Any
    device: str
    amp: bool

    def infer_probabilities(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        height, width = rgb.shape[:2]
        image = (
            torch.from_numpy(rgb)
            .to(device=self.device)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .div_(255.0)
            .unsqueeze(0)
        )
        with torch.inference_mode():
            probabilities = self.encoder.encode_image_to_feat_map(
                image,
                orig_img_size=(height, width),
                return_preds=False,
                ignore_label=False,
            )
        values = np.asarray(_to_numpy(probabilities), dtype=np.float32)
        if values.shape != (1, CLASS_COUNT, height, width):
            raise RuntimeError(
                "RADSeg must return probabilities for exactly 41 classes at original size"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RuntimeError("RADSeg returned invalid probabilities")
        mass = values.sum(axis=1, keepdims=True, dtype=np.float64)
        if np.any(mass <= 0.0):
            raise RuntimeError("RADSeg returned zero probability mass")
        if np.any(values > 1.0) or np.any(mass > 1.0 + 1e-6):
            raise RuntimeError("RADSeg returned probability mass above one")
        return np.ascontiguousarray(values, dtype=np.float32)


@dataclass
class NARadioRuntime:
    encoder: Any
    text_embeddings: Any
    device: str
    amp: bool

    def __post_init__(self) -> None:
        shape = tuple(self.text_embeddings.shape)
        if len(shape) != 2 or shape[0] != CLASS_COUNT or shape[1] <= 0:
            raise RuntimeError("NARADIO text embeddings must have shape [41,D]")

    def infer_probabilities(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        from torch.nn import functional as functional

        height, width = rgb.shape[:2]
        nearest = self.encoder.get_nearest_size(height, width)
        if (
            not isinstance(nearest, (tuple, list))
            or len(nearest) != 2
            or any(type(value) is not int or value <= 0 for value in nearest)
        ):
            raise RuntimeError("NARADIO returned an invalid nearest-supported resolution")
        supported_size = (nearest[0], nearest[1])
        image = (
            torch.from_numpy(rgb)
            .to(device=self.device)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .div_(255.0)
            .unsqueeze(0)
        )
        if supported_size != (height, width):
            image = functional.interpolate(
                image,
                size=supported_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        self.encoder.input_resolution = supported_size
        with torch.inference_mode():
            features = self.encoder.encode_image_to_feat_map(image)
            aligned = self.encoder.align_spatial_features_with_language(features)
            aligned = functional.normalize(aligned.float(), dim=1)
            text = functional.normalize(self.text_embeddings.float(), dim=1)
            cosine = torch.einsum("bdhw,cd->bchw", aligned, text)
            probabilities = torch.softmax(100.0 * cosine, dim=1)
            if probabilities.shape[-2:] != (height, width):
                probabilities = functional.interpolate(
                    probabilities,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            probabilities = probabilities.clamp_min(0.0)
            mass = probabilities.sum(dim=1, keepdim=True)
            if not bool(torch.all(torch.isfinite(mass) & (mass > 0.0))):
                raise RuntimeError("NARADIO produced invalid probability mass")
            probabilities = probabilities / mass
        values = np.asarray(_to_numpy(probabilities), dtype=np.float32)
        if values.shape != (1, CLASS_COUNT, height, width):
            raise RuntimeError("NARADIO must return 41-class probabilities at original size")
        return np.ascontiguousarray(values)


def _instantiate_radseg(
    encoder_class: Any,
    args: argparse.Namespace,
    classes: Sequence[str],
) -> Any:
    kwargs: dict[str, Any] = {
        "device": args.device,
        "model_version": args.model_version,
        "lang_model": args.lang_model,
        "predict": True,
        "classes": list(classes),
        "amp": args.amp,
        "sam_refinement": args.sam_refinement,
        "prompt_denoising_thresh": 0.5,
        "scra_scaling": 10.0,
        "scga_scaling": 10.0,
        "slide_crop": 336,
        "slide_stride": 112,
    }
    if args.sam_refinement:
        assert args.sam_checkpoint is not None
        kwargs["sam_ckpt"] = str(args.sam_checkpoint.expanduser().resolve())
    return encoder_class(**kwargs)


def _instantiate_naradio(encoder_class: Any, args: argparse.Namespace) -> Any:
    return encoder_class(
        device=args.device,
        model_version=args.model_version,
        lang_model=args.lang_model,
        input_resolution=(224, 224),
        return_radio_features=True,
        compile=False,
        amp=args.amp,
    )


def _import_pinned_module(root: Path, module_name: str) -> ModuleType:
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(module_name)
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
    module_file = Path(str(getattr(module, "__file__", ""))).resolve()
    try:
        module_file.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"imported {module_name} outside pinned source root") from exc
    return module


def _prompt_descriptor(encoder: Any, classes: Sequence[str]) -> dict[str, Any]:
    method = getattr(encoder, "insert_labels_into_templates", None)
    if method is None:
        raise RuntimeError("encoder does not expose its label prompt templates")
    prompts = method(list(classes))
    if (
        not isinstance(prompts, list)
        or len(prompts) != len(classes)
        or any(not isinstance(group, list) or not group for group in prompts)
        or any(
            not isinstance(prompt, str) or not prompt
            for group in prompts
            for prompt in group
        )
    ):
        raise RuntimeError("encoder returned an invalid prompt-template ensemble")
    return {"mode": "labels", "classes": list(classes), "prompts": prompts}


def _detach_text_embeddings(value: Any, name: str) -> Any:
    if not hasattr(value, "detach") or not hasattr(value, "shape"):
        raise RuntimeError(f"{name} text embeddings must be tensor-like")
    shape = tuple(value.shape)
    if len(shape) != 2 or shape[0] != CLASS_COUNT or shape[1] <= 0:
        raise RuntimeError(f"{name} text embeddings must have shape [41,D]")
    detached = value.detach()
    if bool(getattr(detached, "requires_grad", True)):
        raise RuntimeError(f"{name} text embeddings still require gradients")
    if getattr(detached, "grad_fn", None) is not None:
        raise RuntimeError(f"{name} text embeddings retain an autograd graph")
    return detached


@dataclass(frozen=True)
class DenseWorker:
    runtime: Any
    classes: tuple[str, ...]
    sample_stride: int
    top_k: int
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if len(self.classes) != CLASS_COUNT:
            raise ValueError("DenseWorker requires exactly 41 frozen classes")
        _require_positive_integer(self.sample_stride, "sample_stride")
        _require_positive_integer(self.top_k, "top_k")
        if self.top_k > len(self.classes):
            raise ValueError("top_k cannot exceed the class count")
        if set(self.provenance) != _PROVENANCE_KEYS:
            raise ValueError("DenseWorker provenance keys do not match the frozen contract")


def build_worker(args: argparse.Namespace) -> DenseWorker:
    classes, vocabulary_sha256 = load_frozen_classes(args.classes_json)
    validate_cli_args(args, classes)
    language_assets = validate_language_model_assets(args)
    source_root = args.source_root.expanduser().resolve()
    radio_root = args.radio_root.expanduser().resolve()
    source_spec = RADSEG_SOURCE if args.backend == "radseg" else RAYFRONTS_SOURCE
    source_commit = validate_source_checkout(source_root, source_spec)
    radio_commit = validate_source_checkout(radio_root, RADIO_SOURCE)
    expected_radio_url = _expected_radio_checkpoint_url(radio_root, args.model_version)
    explicit_checkpoint_before = (
        _fingerprint_file(args.model_version, require_read_only=True)
        if Path(args.model_version).expanduser().is_file()
        else None
    )
    sam_before = (
        _fingerprint_file(args.sam_checkpoint, require_read_only=True)
        if args.sam_checkpoint is not None
        else None
    )

    import torch

    text_embeddings: Any | None = None
    with CheckpointTracker(torch.hub, expected_url=expected_radio_url) as tracker:
        with local_radio_hub(torch, radio_root):
            with pinned_language_model_load(language_assets):
                with torch.inference_mode():
                    if args.backend == "radseg":
                        module = _import_pinned_module(source_root, "radseg.radseg")
                        encoder = _instantiate_radseg(module.RADSegEncoder, args, classes)
                        encoder.text_embeds = _detach_text_embeddings(
                            getattr(encoder, "text_embeds", None),
                            "RADSeg",
                        )
                    else:
                        module = _import_pinned_module(
                            source_root, "rayfronts.image_encoders.naradio"
                        )
                        encoder = _instantiate_naradio(module.NARadioEncoder, args)
                        text_embeddings = _detach_text_embeddings(
                            encoder.encode_labels(classes),
                            "NARADIO",
                        )
    source_commit_after = validate_source_checkout(source_root, source_spec)
    radio_commit_after = validate_source_checkout(radio_root, RADIO_SOURCE)
    if source_commit_after != source_commit or radio_commit_after != radio_commit:
        raise RuntimeError("pinned source checkout changed during model loading")
    model_checkpoint = resolve_model_checkpoint(tracker.paths, args.model_version)
    model_after = _fingerprint_file(
        model_checkpoint,
        require_read_only=explicit_checkpoint_before is not None,
    )
    expected_model_fingerprint = (
        explicit_checkpoint_before
        if explicit_checkpoint_before is not None
        else tracker.fingerprints.get(model_checkpoint)
    )
    if expected_model_fingerprint is None or model_after != expected_model_fingerprint:
        raise RuntimeError("RADIO checkpoint changed during model construction")
    model_sha256 = model_after.sha256

    if args.sam_refinement:
        assert args.sam_checkpoint is not None
        assert sam_before is not None
        sam_after = _fingerprint_file(args.sam_checkpoint, require_read_only=True)
        if sam_after != sam_before:
            raise RuntimeError("SAM checkpoint changed during model construction")
        auxiliary_model_sha256 = sam_after.sha256
    else:
        auxiliary_model_sha256 = ""

    if args.backend == "radseg":
        runtime: Any = RadsegRuntime(encoder=encoder, device=args.device, amp=args.amp)
        model_id = (
            f"radseg:{args.model_version}:{args.lang_model}:"
            f"sam={int(args.sam_refinement)}:sha256={model_sha256}"
        )
        model_config = {
            "predict": True,
            "class_count": CLASS_COUNT,
            "scra_scaling": 10.0,
            "scga_scaling": 10.0,
            "slide_crop": 336,
            "slide_stride": 112,
            "prompt_denoising_thresh": 0.5,
            "sam_refinement": args.sam_refinement,
        }
    else:
        assert text_embeddings is not None
        runtime = NARadioRuntime(
            encoder=encoder,
            text_embeddings=text_embeddings,
            device=args.device,
            amp=args.amp,
        )
        model_id = (
            f"naradio:{args.model_version}:{args.lang_model}:sha256={model_sha256}"
        )
        model_config = {
            "class_count": CLASS_COUNT,
            "return_radio_features": True,
            "compile": False,
            "cosine_temperature": 100.0,
            "resolution_policy": "nearest-supported-then-resize-probabilities",
        }

    prompt_sha256 = canonical_sha256(_prompt_descriptor(encoder, classes))
    inference_config_sha256 = canonical_sha256(
        {
            "inference_config_version": 2,
            "backend": args.backend,
            "model_version": args.model_version,
            "lang_model": args.lang_model,
            "device": args.device,
            "amp": args.amp,
            "sample_stride": args.sample_stride,
            "top_k": args.top_k,
            "model": model_config,
            "model_sha256": model_sha256,
            "auxiliary_model_sha256": auxiliary_model_sha256,
            "language_model_id": language_assets.model_id,
            "language_model_revision": language_assets.revision,
            "language_model_sha256": language_assets.sha256,
        }
    )
    provenance = {
        "backend": args.backend,
        "source_commit": source_commit,
        "radio_commit": radio_commit,
        "model_id": model_id,
        "model_sha256": model_sha256,
        "auxiliary_model_sha256": auxiliary_model_sha256,
        "language_model_id": language_assets.model_id,
        "language_model_revision": language_assets.revision,
        "language_model_sha256": language_assets.sha256,
        "vocabulary_sha256": vocabulary_sha256,
        "prompt_sha256": prompt_sha256,
        "inference_config_sha256": inference_config_sha256,
    }
    return DenseWorker(
        runtime=runtime,
        classes=tuple(classes),
        sample_stride=args.sample_stride,
        top_k=args.top_k,
        provenance=provenance,
    )


def run_request(worker: DenseWorker, request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    request_id = request.get("id")
    operation = request.get("operation")
    if operation == "metadata":
        return {
            "id": request_id,
            "ok": True,
            "class_count": len(worker.classes),
            "classes": list(worker.classes),
            "sample_stride": worker.sample_stride,
            "top_k": worker.top_k,
            "provenance": dict(worker.provenance),
        }
    if operation != "infer":
        raise ValueError("operation must be 'metadata' or 'infer'")
    rgb = decode_rgb(request.get("rgb"))
    validate_inference_budget(
        height=int(rgb.shape[0]),
        width=int(rgb.shape[1]),
        sample_stride=worker.sample_stride,
        top_k=worker.top_k,
    )
    probabilities = worker.runtime.infer_probabilities(rgb)
    reduced = reduce_probabilities(
        probabilities,
        sample_stride=worker.sample_stride,
        top_k=worker.top_k,
    )
    return {
        "id": request_id,
        "ok": True,
        "image_shape": list(rgb.shape[:2]),
        "class_count": len(worker.classes),
        "sample_stride": worker.sample_stride,
        "class_ids": encode_array(reduced["class_ids"]),
        "probabilities": encode_array(reduced["probabilities"]),
        "entropy": encode_array(reduced["entropy"]),
        "margin": encode_array(reduced["margin"]),
    }


def _error_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _bounded_jsonl_lines(input_stream: TextIO) -> Iterator[str | None]:
    while True:
        line = input_stream.readline(MAX_JSONL_LINE_CHARS + 1)
        if line == "":
            return
        if len(line) <= MAX_JSONL_LINE_CHARS:
            yield line
            continue
        complete = line.endswith("\n")
        while not complete:
            remainder = input_stream.readline(MAX_JSONL_LINE_CHARS + 1)
            if remainder == "":
                complete = True
            elif remainder.endswith("\n"):
                complete = True
        yield None


def serve_jsonl(
    worker: DenseWorker,
    input_stream: TextIO,
    output_stream: TextIO,
    error_stream: TextIO,
) -> None:
    for line in _bounded_jsonl_lines(input_stream):
        request_id: Any = None
        if line is None:
            response = {
                "id": None,
                "ok": False,
                "error": "request line exceeds the JSONL size limit",
            }
        else:
            try:
                request = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"invalid JSON constant: {value}")
                    ),
                )
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                request_id = request.get("id")
                with redirect_stdout(error_stream):
                    response = run_request(worker, request)
            except Exception as exc:
                response = {"id": request_id, "ok": False, "error": _error_message(exc)}
        encoded = json.dumps(
            response,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        if len(encoded) > MAX_JSONL_RESPONSE_CHARS:
            encoded = json.dumps(
                {
                    "id": None,
                    "ok": False,
                    "error": "response exceeds the JSONL size limit",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        output_stream.write(encoded + "\n")
        output_stream.flush()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            worker = build_worker(args)
    except Exception as exc:
        print(f"worker startup failed: {_error_message(exc)}", file=sys.stderr)
        return 2
    serve_jsonl(worker, sys.stdin, sys.stdout, sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
