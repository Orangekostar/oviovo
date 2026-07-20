from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import BinaryIO
import zipfile
import zlib

import numpy as np
from numpy.lib import format as npy_format


_SCHEMA_VERSION = 1
_INT64_MAX = int(np.iinfo(np.int64).max)
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_MEMBER_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
_MAX_METADATA_MEMBER_BYTES = 64 * 1024
_MAX_NPY_HEADER_BYTES = 16 * 1024
_MAX_CENTRAL_DIRECTORY_BYTES = 1024 * 1024
_MAX_COMPRESSION_RATIO = 4096.0
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
_ARCHIVE_MEMBERS = frozenset(f"{name}.npy" for name in _ARCHIVE_KEYS)
_ZIP_EOCD_SIGNATURE = b"PK\x05\x06"


@dataclass(frozen=True)
class _NpyHeader:
    shape: tuple[int, ...]
    dtype: np.dtype
    header_bytes: int


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return _sha256_stream(stream)


def _read_archive_snapshot(path: Path) -> bytes:
    if path.stat().st_size > _MAX_ARCHIVE_BYTES:
        raise ValueError(
            f"dense frame archive exceeds {_MAX_ARCHIVE_BYTES} byte size limit"
        )
    with path.open("rb") as stream:
        snapshot = stream.read(_MAX_ARCHIVE_BYTES + 1)
    if len(snapshot) > _MAX_ARCHIVE_BYTES:
        raise ValueError(
            f"dense frame archive exceeds {_MAX_ARCHIVE_BYTES} byte size limit"
        )
    return snapshot


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
        array_contracts = {
            "class_ids": (np.dtype(np.int64), 3),
            "probabilities": (np.dtype(np.float32), 3),
            "entropy": (np.dtype(np.float32), 2),
            "margin": (np.dtype(np.float32), 2),
        }
        arrays: dict[str, np.ndarray] = {}
        for name, (required_dtype, required_ndim) in array_contracts.items():
            raw_array = getattr(self, name)
            if not isinstance(raw_array, np.ndarray):
                raise ValueError(f"{name} must be a numpy ndarray")
            if raw_array.dtype != required_dtype:
                raise ValueError(f"{name} must have dtype {required_dtype}")
            if raw_array.ndim != required_ndim:
                raise ValueError(f"{name} must have {required_ndim} dimensions")
            arrays[name] = np.array(
                raw_array,
                dtype=required_dtype,
                copy=True,
                order="C",
            )
        class_ids = arrays["class_ids"]
        probabilities = arrays["probabilities"]
        entropy = arrays["entropy"]
        margin = arrays["margin"]

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

        sampled_shape = (
            (image_shape[0] + sample_stride - 1) // sample_stride,
            (image_shape[1] + sample_stride - 1) // sample_stride,
        )
        if class_ids.shape[:2] != sampled_shape:
            raise ValueError(
                f"class_ids shape must start with sampled shape {sampled_shape}"
            )
        top_k = class_ids.shape[2]
        if not 1 <= top_k <= class_count:
            raise ValueError("class_ids top-k must be between 1 and class_count")
        if probabilities.shape != class_ids.shape:
            raise ValueError("probabilities shape must match class_ids")
        if entropy.shape != sampled_shape:
            raise ValueError(f"entropy shape must be sampled shape {sampled_shape}")
        if margin.shape != sampled_shape:
            raise ValueError(f"margin shape must be sampled shape {sampled_shape}")

        if np.any(class_ids < 0) or np.any(class_ids > class_count):
            raise ValueError("class_ids values must be in [0, class_count]")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities must contain only finite values")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("probabilities values must be in [0, 1]")
        probability_sum = np.sum(probabilities, axis=-1, dtype=np.float64)
        if np.any(probability_sum > 1.0 + 1e-6):
            raise ValueError("probabilities sum must not exceed 1 + 1e-6")
        if top_k > 1 and np.any(
            probabilities[..., :-1] < probabilities[..., 1:]
        ):
            raise ValueError("probabilities must be ordered from highest to lowest")
        if np.any((class_ids == 0) & (probabilities > 0.0)):
            raise ValueError("class_ids equal to zero must have zero probabilities")
        if np.any((class_ids > 0) & (probabilities <= 0.0)):
            raise ValueError("positive class_ids must have positive probabilities")
        for candidates in class_ids.reshape(-1, top_k):
            positive_candidates = candidates[candidates > 0]
            if np.unique(positive_candidates).size != positive_candidates.size:
                raise ValueError("class_ids contain duplicate positive IDs at a pixel")

        if not np.all(np.isfinite(entropy)):
            raise ValueError("entropy must contain only finite values")
        maximum_entropy = math.log(class_count)
        entropy_tolerance = 0.0 if class_count == 1 else 1e-6
        if np.any(entropy < 0.0) or np.any(
            entropy > maximum_entropy + entropy_tolerance
        ):
            raise ValueError("entropy values must be in [0, log(class_count)]")
        if not np.all(np.isfinite(margin)):
            raise ValueError("margin must contain only finite values")
        if np.any(margin < 0.0) or np.any(margin > 1.0):
            raise ValueError("margin values must be in [0, 1]")
        second_probability = (
            probabilities[..., 1]
            if top_k > 1
            else np.zeros(sampled_shape, dtype=np.float32)
        )
        expected_margin = probabilities[..., 0] - second_probability
        if not np.allclose(
            margin,
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
        for name, array in arrays.items():
            object.__setattr__(self, name, _immutable_array(array))


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


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


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
        _preflight_archive(_read_archive_snapshot(temporary_path))
        os.replace(temporary_path, destination)
        temporary_path = None
        try:
            _fsync_directory(destination.parent)
        except OSError as exc:
            raise RuntimeError(
                "dense frame publication may already be visible, but parent "
                "directory fsync failed"
            ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_int_scalar(payload: dict[str, np.ndarray], name: str) -> int:
    value = payload[name]
    if value.shape != () or value.dtype != np.dtype(np.int64):
        raise ValueError(f"{name} must be a scalar int64")
    return int(value.item())


def _read_npy_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    field_name: str,
) -> _NpyHeader:
    with archive.open(info, "r") as member:
        version = npy_format.read_magic(member)
        if version == (1, 0):
            shape, _fortran_order, dtype = npy_format.read_array_header_1_0(
                member,
                max_header_size=_MAX_NPY_HEADER_BYTES,
            )
        elif version == (2, 0):
            shape, _fortran_order, dtype = npy_format.read_array_header_2_0(
                member,
                max_header_size=_MAX_NPY_HEADER_BYTES,
            )
        else:
            raise ValueError(f"{field_name} has unsupported NPY version {version}")
        header_bytes = member.tell()
    normalized_dtype = np.dtype(dtype)
    if normalized_dtype.hasobject:
        raise ValueError(f"{field_name} object dtype is forbidden")
    element_count = 1
    for dimension in shape:
        if not isinstance(dimension, int) or dimension < 0:
            raise ValueError(f"{field_name} has invalid NPY shape {shape}")
        element_count *= dimension
        if element_count * normalized_dtype.itemsize > _MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise ValueError(f"{field_name} NPY shape exceeds member resource limit")
    expected_file_size = header_bytes + element_count * normalized_dtype.itemsize
    if expected_file_size != info.file_size:
        raise ValueError(
            f"{field_name} NPY header/data size mismatch: "
            f"expected {expected_file_size}, got {info.file_size}"
        )
    return _NpyHeader(tuple(shape), normalized_dtype, header_bytes)


def _require_npy_contract(
    headers: dict[str, _NpyHeader],
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> None:
    header = headers[name]
    if header.shape != shape:
        raise ValueError(f"{name} NPY shape must be {shape}, got {header.shape}")
    if header.dtype != dtype:
        raise ValueError(f"{name} NPY dtype must be {dtype}, got {header.dtype}")


def _read_metadata_array(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    name: str,
) -> np.ndarray:
    if info.file_size > _MAX_METADATA_MEMBER_BYTES:
        raise ValueError(f"{name} metadata member exceeds safe size limit")
    with archive.open(info, "r") as member:
        return npy_format.read_array(
            member,
            allow_pickle=False,
            max_header_size=_MAX_NPY_HEADER_BYTES,
        )


def _precheck_central_directory(snapshot: bytes) -> None:
    search_start = max(0, len(snapshot) - (65_535 + 22))
    search_end = len(snapshot)
    eocd_offset = -1
    eocd: tuple[bytes, int, int, int, int, int, int, int] | None = None
    while search_end > search_start:
        candidate = snapshot.rfind(
            _ZIP_EOCD_SIGNATURE,
            search_start,
            search_end,
        )
        if candidate < 0:
            break
        if candidate + 22 <= len(snapshot):
            parsed = struct.unpack_from("<4s4H2IH", snapshot, candidate)
            if candidate + 22 + parsed[-1] == len(snapshot):
                eocd_offset = candidate
                eocd = parsed
                break
        search_end = candidate
    if eocd is None:
        raise ValueError("dense frame NPZ has no valid ZIP central directory")
    (
        _signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        _comment_size,
    ) = eocd
    expected_entries = len(_ARCHIVE_MEMBERS)
    if disk_number != 0 or central_disk != 0 or disk_entries != total_entries:
        raise ValueError("dense frame NPZ multi-disk ZIP archives are forbidden")
    if total_entries != expected_entries:
        raise ValueError(
            "dense frame NPZ keys/member count mismatch: "
            f"expected {expected_entries}, got {total_entries}"
        )
    if central_size > _MAX_CENTRAL_DIRECTORY_BYTES:
        raise ValueError("dense frame ZIP central directory exceeds resource limit")
    if central_offset + central_size != eocd_offset:
        raise ValueError("dense frame ZIP central directory offsets are inconsistent")


def _preflight_archive(snapshot: bytes) -> dict[str, object]:
    _precheck_central_directory(snapshot)
    with zipfile.ZipFile(io.BytesIO(snapshot), mode="r") as archive:
        infos = archive.infolist()
        name_counts = Counter(info.filename for info in infos)
        names = set(name_counts)
        if len(infos) != len(_ARCHIVE_MEMBERS) or names != _ARCHIVE_MEMBERS:
            missing = sorted(_ARCHIVE_MEMBERS - names)
            extra = sorted(names - _ARCHIVE_MEMBERS)
            duplicates = sorted(
                name for name, count in name_counts.items() if count > 1
            )
            raise ValueError(
                "dense frame NPZ keys mismatch: "
                f"missing={missing}, extra={extra}, duplicates={duplicates}"
            )

        info_by_field: dict[str, zipfile.ZipInfo] = {}
        total_uncompressed = 0
        for info in infos:
            field_name = info.filename.removesuffix(".npy")
            if (
                info.is_dir()
                or "/" in info.filename
                or "\\" in info.filename
                or info.flag_bits & 0x1
            ):
                raise ValueError(f"{field_name} has unsafe encrypted or nested ZIP path")
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError(f"{field_name} uses unsupported ZIP compression")
            if info.file_size > _MAX_MEMBER_UNCOMPRESSED_BYTES:
                raise ValueError(f"{field_name} member exceeds safe size limit")
            total_uncompressed += info.file_size
            if total_uncompressed > _MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError("dense frame total uncompressed size exceeds resource limit")
            if info.file_size > 0 and (
                info.compress_size <= 0
                or info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO
            ):
                raise ValueError(f"{field_name} compression ratio exceeds resource limit")
            info_by_field[field_name] = info

        headers = {
            name: _read_npy_header(archive, info_by_field[name], name)
            for name in _ARCHIVE_KEYS
        }
        int64_dtype = np.dtype(np.int64)
        for name in (
            "schema_version",
            "cache_frame_id",
            "source_frame_id",
            "sample_stride",
            "class_count",
        ):
            _require_npy_contract(headers, name, (), int64_dtype)
        _require_npy_contract(headers, "image_shape", (2,), int64_dtype)

        metadata_names = (
            "schema_version",
            "cache_frame_id",
            "source_frame_id",
            "image_shape",
            "sample_stride",
            "class_count",
        )
        metadata_arrays = {
            name: _read_metadata_array(archive, info_by_field[name], name)
            for name in metadata_names
        }
        schema_version = _load_int_scalar(metadata_arrays, "schema_version")
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported dense frame schema_version {schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        cache_frame_id = _normalize_integer(
            _load_int_scalar(metadata_arrays, "cache_frame_id"),
            "cache_frame_id",
            positive=False,
        )
        source_frame_id = _normalize_integer(
            _load_int_scalar(metadata_arrays, "source_frame_id"),
            "source_frame_id",
            positive=False,
        )
        sample_stride = _normalize_integer(
            _load_int_scalar(metadata_arrays, "sample_stride"),
            "sample_stride",
            positive=True,
        )
        class_count = _normalize_integer(
            _load_int_scalar(metadata_arrays, "class_count"),
            "class_count",
            positive=True,
        )
        image_shape_array = metadata_arrays["image_shape"]
        image_shape = tuple(
            _normalize_integer(value, "image_shape", positive=True)
            for value in image_shape_array
        )
        sampled_shape = (
            (image_shape[0] + sample_stride - 1) // sample_stride,
            (image_shape[1] + sample_stride - 1) // sample_stride,
        )
        class_ids_header = headers["class_ids"]
        if len(class_ids_header.shape) != 3:
            raise ValueError("class_ids NPY shape must have three dimensions")
        top_k = class_ids_header.shape[2]
        if not 1 <= top_k <= class_count:
            raise ValueError("class_ids top-k must be between 1 and class_count")
        top_k_shape = (*sampled_shape, top_k)
        _require_npy_contract(headers, "class_ids", top_k_shape, int64_dtype)
        _require_npy_contract(
            headers,
            "probabilities",
            top_k_shape,
            np.dtype(np.float32),
        )
        for name in ("entropy", "margin"):
            _require_npy_contract(
                headers,
                name,
                sampled_shape,
                np.dtype(np.float32),
            )

    return {
        "cache_frame_id": cache_frame_id,
        "source_frame_id": source_frame_id,
        "image_shape": image_shape,
        "sample_stride": sample_stride,
        "class_count": class_count,
    }


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

    snapshot = _read_archive_snapshot(source)
    if expected_sha256 is not None:
        actual_sha256 = hashlib.sha256(snapshot).hexdigest()
        if actual_sha256 != expected_sha256.lower():
            raise ValueError(
                f"dense frame checksum mismatch: expected {expected_sha256.lower()}, "
                f"got {actual_sha256}"
            )
    try:
        metadata = _preflight_archive(snapshot)
        with np.load(io.BytesIO(snapshot), allow_pickle=False) as archive:
            payload = {
                name: np.array(archive[name], copy=True)
                for name in ("class_ids", "probabilities", "entropy", "margin")
            }
    except (OSError, EOFError, ValueError, zipfile.BadZipFile, zlib.error) as exc:
        raise ValueError(f"invalid or corrupt dense frame NPZ: {exc}") from exc

    try:
        return DenseSemanticFrame(
            cache_frame_id=metadata["cache_frame_id"],
            source_frame_id=metadata["source_frame_id"],
            image_shape=metadata["image_shape"],
            sample_stride=metadata["sample_stride"],
            class_count=metadata["class_count"],
            class_ids=payload["class_ids"],
            probabilities=payload["probabilities"],
            entropy=payload["entropy"],
            margin=payload["margin"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid dense frame field data: {exc}") from exc
