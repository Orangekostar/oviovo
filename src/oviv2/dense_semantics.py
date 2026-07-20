from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
from typing import BinaryIO
import zipfile
import zlib

import numpy as np


_SCHEMA_VERSION = 1
_INT64_MAX = int(np.iinfo(np.int64).max)
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
_ARCHIVE_KEYS = frozenset(
    {
        "schema_version",
        "cache_frame_id",
        "source_frame_id",
        "image_shape",
        "sample_stride",
        "class_count",
        "class_ids",
        "probabilities",
        "entropy",
        "margin",
    }
)


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return _sha256_stream(stream)


@dataclass(frozen=True)
class DenseSemanticProvenance:
    backend: str
    source_commit: str
    radio_commit: str
    model_id: str
    model_sha256: str
    auxiliary_model_sha256: str
    vocabulary_sha256: str
    prompt_sha256: str
    inference_config_sha256: str
    cache_prefix_sha256: str

    def __post_init__(self) -> None:
        for name in ("backend", "model_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())

        for name in ("source_commit", "radio_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a 40-character hexadecimal commit")
            object.__setattr__(self, name, value.lower())

        for name in (
            "model_sha256",
            "vocabulary_sha256",
            "prompt_sha256",
            "inference_config_sha256",
            "cache_prefix_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
            object.__setattr__(self, name, value.lower())

        auxiliary = self.auxiliary_model_sha256
        if not isinstance(auxiliary, str) or (
            auxiliary and _SHA256_PATTERN.fullmatch(auxiliary) is None
        ):
            raise ValueError(
                "auxiliary_model_sha256 must be empty or a 64-character "
                "hexadecimal SHA-256"
            )
        object.__setattr__(self, "auxiliary_model_sha256", auxiliary.lower())


def _normalize_integer(value: object, name: str, *, positive: bool) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    normalized = int(value)
    if (positive and normalized <= 0) or (not positive and normalized < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    if normalized > _INT64_MAX:
        raise ValueError(f"{name} must fit in a signed int64")
    return normalized


def _immutable_array(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class DenseSemanticFrame:
    cache_frame_id: int
    source_frame_id: int
    image_shape: tuple[int, int]
    sample_stride: int
    class_count: int
    class_ids: np.ndarray
    probabilities: np.ndarray
    entropy: np.ndarray
    margin: np.ndarray

    def __post_init__(self) -> None:
        cache_frame_id = _normalize_integer(
            self.cache_frame_id,
            "cache_frame_id",
            positive=False,
        )
        source_frame_id = _normalize_integer(
            self.source_frame_id,
            "source_frame_id",
            positive=False,
        )
        sample_stride = _normalize_integer(
            self.sample_stride,
            "sample_stride",
            positive=True,
        )
        class_count = _normalize_integer(
            self.class_count,
            "class_count",
            positive=True,
        )
        if not isinstance(self.image_shape, tuple) or len(self.image_shape) != 2:
            raise ValueError("image_shape must be a (height, width) tuple")
        image_shape = tuple(
            _normalize_integer(value, "image_shape", positive=True)
            for value in self.image_shape
        )

        array_dtypes = {
            "class_ids": np.dtype(np.int64),
            "probabilities": np.dtype(np.float32),
            "entropy": np.dtype(np.float32),
            "margin": np.dtype(np.float32),
        }
        for name, required_dtype in array_dtypes.items():
            array = getattr(self, name)
            if not isinstance(array, np.ndarray):
                raise ValueError(f"{name} must be a numpy ndarray")
            if array.dtype != required_dtype:
                raise ValueError(f"{name} must have dtype {required_dtype}")

        sampled_shape = (
            (image_shape[0] + sample_stride - 1) // sample_stride,
            (image_shape[1] + sample_stride - 1) // sample_stride,
        )
        if self.class_ids.ndim != 3 or self.class_ids.shape[:2] != sampled_shape:
            raise ValueError(
                f"class_ids shape must start with sampled shape {sampled_shape}"
            )
        top_k = self.class_ids.shape[2]
        if not 1 <= top_k <= class_count:
            raise ValueError("class_ids top-k must be between 1 and class_count")
        if self.probabilities.shape != self.class_ids.shape:
            raise ValueError("probabilities shape must match class_ids")
        if self.entropy.shape != sampled_shape:
            raise ValueError(f"entropy shape must be sampled shape {sampled_shape}")
        if self.margin.shape != sampled_shape:
            raise ValueError(f"margin shape must be sampled shape {sampled_shape}")

        if np.any(self.class_ids < 0) or np.any(self.class_ids > class_count):
            raise ValueError("class_ids values must be in [0, class_count]")
        if not np.all(np.isfinite(self.probabilities)):
            raise ValueError("probabilities must contain only finite values")
        if np.any(self.probabilities < 0.0) or np.any(self.probabilities > 1.0):
            raise ValueError("probabilities values must be in [0, 1]")
        probability_sum = np.sum(self.probabilities, axis=-1, dtype=np.float64)
        if np.any(probability_sum > 1.0 + 1e-6):
            raise ValueError("probabilities sum must not exceed 1 + 1e-6")
        if top_k > 1 and np.any(
            self.probabilities[..., :-1] < self.probabilities[..., 1:]
        ):
            raise ValueError("probabilities must be ordered from highest to lowest")
        if np.any((self.class_ids == 0) & (self.probabilities > 0.0)):
            raise ValueError("class_ids equal to zero must have zero probabilities")
        if np.any((self.class_ids > 0) & (self.probabilities <= 0.0)):
            raise ValueError("positive class_ids must have positive probabilities")
        for candidates in self.class_ids.reshape(-1, top_k):
            positive_candidates = candidates[candidates > 0]
            if np.unique(positive_candidates).size != positive_candidates.size:
                raise ValueError("class_ids contain duplicate positive IDs at a pixel")

        if not np.all(np.isfinite(self.entropy)):
            raise ValueError("entropy must contain only finite values")
        maximum_entropy = math.log(class_count)
        entropy_tolerance = 0.0 if class_count == 1 else 1e-6
        if np.any(self.entropy < 0.0) or np.any(
            self.entropy > maximum_entropy + entropy_tolerance
        ):
            raise ValueError("entropy values must be in [0, log(class_count)]")
        if not np.all(np.isfinite(self.margin)):
            raise ValueError("margin must contain only finite values")
        if np.any(self.margin < 0.0) or np.any(self.margin > 1.0):
            raise ValueError("margin values must be in [0, 1]")
        second_probability = (
            self.probabilities[..., 1]
            if top_k > 1
            else np.zeros(sampled_shape, dtype=np.float32)
        )
        expected_margin = self.probabilities[..., 0] - second_probability
        if not np.allclose(
            self.margin,
            expected_margin,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("margin must equal the top-1 minus top-2 probability")

        object.__setattr__(self, "cache_frame_id", cache_frame_id)
        object.__setattr__(self, "source_frame_id", source_frame_id)
        object.__setattr__(self, "image_shape", image_shape)
        object.__setattr__(self, "sample_stride", sample_stride)
        object.__setattr__(self, "class_count", class_count)
        for name in array_dtypes:
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))


def _archive_payload(frame: DenseSemanticFrame) -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(_SCHEMA_VERSION, dtype=np.int64),
        "cache_frame_id": np.asarray(frame.cache_frame_id, dtype=np.int64),
        "source_frame_id": np.asarray(frame.source_frame_id, dtype=np.int64),
        "image_shape": np.asarray(frame.image_shape, dtype=np.int64),
        "sample_stride": np.asarray(frame.sample_stride, dtype=np.int64),
        "class_count": np.asarray(frame.class_count, dtype=np.int64),
        "class_ids": frame.class_ids,
        "probabilities": frame.probabilities,
        "entropy": frame.entropy,
        "margin": frame.margin,
    }


def write_dense_frame(path: str | Path, frame: DenseSemanticFrame) -> None:
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("dense frame path must have a .npz suffix")
    if not isinstance(frame, DenseSemanticFrame):
        raise TypeError("frame must be a DenseSemanticFrame")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **_archive_payload(frame))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_int_scalar(payload: dict[str, np.ndarray], name: str) -> int:
    value = payload[name]
    if value.shape != () or value.dtype != np.dtype(np.int64):
        raise ValueError(f"{name} must be a scalar int64")
    return int(value.item())


def load_dense_frame(
    path: str | Path,
    expected_sha256: str | None = None,
) -> DenseSemanticFrame:
    source = Path(path)
    if source.suffix != ".npz":
        raise ValueError("dense frame path must have a .npz suffix")
    if not source.is_file():
        raise ValueError(f"dense frame file does not exist: {source}")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise ValueError("expected checksum must be a 64-character hexadecimal SHA-256")

    with source.open("rb") as stream:
        if expected_sha256 is not None:
            actual_sha256 = _sha256_stream(stream)
            if actual_sha256 != expected_sha256.lower():
                raise ValueError(
                    f"dense frame checksum mismatch: expected {expected_sha256.lower()}, "
                    f"got {actual_sha256}"
                )
            stream.seek(0)
        try:
            with np.load(stream, allow_pickle=False) as archive:
                archive_keys = archive.files
                keys = set(archive_keys)
                if len(archive_keys) != len(_ARCHIVE_KEYS) or keys != _ARCHIVE_KEYS:
                    missing = sorted(_ARCHIVE_KEYS - keys)
                    extra = sorted(keys - _ARCHIVE_KEYS)
                    duplicates = sorted(
                        name for name in keys if archive_keys.count(name) > 1
                    )
                    raise ValueError(
                        "dense frame NPZ keys mismatch: "
                        f"missing={missing}, extra={extra}, duplicates={duplicates}"
                    )
                payload = {
                    name: np.array(archive[name], copy=True)
                    for name in _ARCHIVE_KEYS
                }
        except (OSError, EOFError, ValueError, zipfile.BadZipFile, zlib.error) as exc:
            raise ValueError(f"invalid or corrupt dense frame NPZ: {exc}") from exc

    schema_version = _load_int_scalar(payload, "schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported dense frame schema_version {schema_version}; "
            f"expected {_SCHEMA_VERSION}"
        )
    image_shape = payload["image_shape"]
    if image_shape.shape != (2,) or image_shape.dtype != np.dtype(np.int64):
        raise ValueError("image_shape must be an int64 array with shape (2,)")

    try:
        return DenseSemanticFrame(
            cache_frame_id=_load_int_scalar(payload, "cache_frame_id"),
            source_frame_id=_load_int_scalar(payload, "source_frame_id"),
            image_shape=(int(image_shape[0]), int(image_shape[1])),
            sample_stride=_load_int_scalar(payload, "sample_stride"),
            class_count=_load_int_scalar(payload, "class_count"),
            class_ids=payload["class_ids"],
            probabilities=payload["probabilities"],
            entropy=payload["entropy"],
            margin=payload["margin"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid dense frame field data: {exc}") from exc
