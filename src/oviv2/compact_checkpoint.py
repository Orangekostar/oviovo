from __future__ import annotations

from collections import Counter
import ctypes
from dataclasses import asdict, dataclass, fields
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import struct
from typing import Any
import zipfile
import zlib

import numpy as np
from numpy.lib import format as npy_format

from src.oviv2.dense_semantics import DenseSemanticProvenance
from src.oviv2.ownership import ReversibleOwnershipStore


COMPACT_OWNERSHIP_FORMAT = "oviv2_compact_ownership_checkpoint"
_INVENTORY = frozenset({"metadata.json", "ownership.npz", "checksums.json"})
_CHECKSUM_FILES = frozenset({"metadata.json", "ownership.npz"})
_OWNERSHIP_ARRAYS = (
    "schema_version",
    "block_resolution",
    "voxel_keys",
    "entity_ids",
    "confidence",
    "epochs",
    "evidence_revisions",
)
_OWNERSHIP_MEMBERS = frozenset(f"{name}.npy" for name in _OWNERSHIP_ARRAYS)
_ZIP_EOCD_SIGNATURE = b"PK\x05\x06"

# These limits match the dense-cache resource scale while remaining finite. A
# 1M-record compact archive has about 56 MB of raw payload; its conservative
# ZIP/deflate bound is checked by _canonical_archive_budget before writing.
_MAX_OWNERSHIP_RECORDS = 1_000_000
_MAX_OWNERSHIP_BLOCKS = 32_768
_MAX_DENSE_OWNERSHIP_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_MEMBER_COMPRESSED_BYTES = 32 * 1024 * 1024
_MAX_MEMBER_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024
_MAX_NPY_HEADER_BYTES = 16 * 1024
_MAX_JSON_BYTES = 64 * 1024
_OWNERSHIP_COMPRESSION_LEVEL = 6
_NPY_MEMBER_OVERHEAD_BOUND = _MAX_NPY_HEADER_BYTES + 16
_ZIP64_LOCAL_EXTRA_BOUND = 20
_ZIP64_CENTRAL_EXTRA_BOUND = 28
_ZIP_DATA_DESCRIPTOR_BOUND = 24
_ZIP_END_RECORDS_BOUND = 22 + 56 + 20
_RENAME_NOREPLACE = 1
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
)


class CompactCheckpointPublicationUncertainError(RuntimeError):
    def __init__(self, target: Path, publication_error: Exception) -> None:
        self.target = target
        self.publication_error = publication_error
        self.published = True
        super().__init__(
            f"compact checkpoint was published in its anchored parent for {target}, "
            "but the user path or durability could not be verified; publication "
            "state is uncertain"
        )


def _publication_test_hook(_stage: str) -> None:
    pass


@dataclass(frozen=True)
class _NpyHeader:
    shape: tuple[int, ...]
    dtype: np.dtype
    header_bytes: int


@dataclass(frozen=True)
class _OwnershipArchiveBudget:
    max_member_uncompressed_bytes: int
    max_member_compressed_bytes: int
    total_uncompressed_bytes: int
    central_directory_bytes: int
    archive_bytes: int


@dataclass(frozen=True)
class _ValidatedOwnershipArrays:
    arrays: dict[str, np.ndarray]
    block_keys: np.ndarray
    local_keys: np.ndarray
    by_block: np.ndarray
    sorted_block_keys: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


def _zlib_compress_bound(source_bytes: int) -> int:
    return (
        source_bytes
        + (source_bytes >> 12)
        + (source_bytes >> 14)
        + (source_bytes >> 25)
        + 13
    )


def _canonical_archive_budget(record_count: int) -> _OwnershipArchiveBudget:
    payload_bytes = {
        "schema_version": 8,
        "block_resolution": 8,
        "voxel_keys": record_count * 3 * 8,
        "entity_ids": record_count * 8,
        "confidence": record_count * 8,
        "epochs": record_count * 8,
        "evidence_revisions": record_count * 8,
    }
    member_bytes = {
        name: payload_bytes[name] + _NPY_MEMBER_OVERHEAD_BOUND
        for name in _OWNERSHIP_ARRAYS
    }
    compressed_bounds = {
        name: _zlib_compress_bound(size)
        for name, size in member_bytes.items()
    }
    central_directory_bytes = sum(
        46
        + len(f"{name}.npy".encode("utf-8"))
        + _ZIP64_CENTRAL_EXTRA_BOUND
        for name in _OWNERSHIP_ARRAYS
    )
    archive_bytes = _ZIP_END_RECORDS_BOUND + central_directory_bytes
    for name in _OWNERSHIP_ARRAYS:
        filename_bytes = len(f"{name}.npy".encode("utf-8"))
        archive_bytes += (
            30
            + filename_bytes
            + _ZIP64_LOCAL_EXTRA_BOUND
            + compressed_bounds[name]
            + _ZIP_DATA_DESCRIPTOR_BOUND
        )
    return _OwnershipArchiveBudget(
        max_member_uncompressed_bytes=max(member_bytes.values()),
        max_member_compressed_bytes=max(compressed_bounds.values()),
        total_uncompressed_bytes=sum(member_bytes.values()),
        central_directory_bytes=central_directory_bytes,
        archive_bytes=archive_bytes,
    )


def _validate_canonical_archive_budget(
    budget: _OwnershipArchiveBudget,
) -> None:
    if budget.max_member_uncompressed_bytes > _MAX_MEMBER_UNCOMPRESSED_BYTES:
        raise ValueError("ownership member uncompressed resource budget exceeds limit")
    if budget.max_member_compressed_bytes > _MAX_MEMBER_COMPRESSED_BYTES:
        raise ValueError("ownership member compressed resource budget exceeds limit")
    if budget.total_uncompressed_bytes > _MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError("ownership total uncompressed resource budget exceeds limit")
    if budget.central_directory_bytes > _MAX_CENTRAL_DIRECTORY_BYTES:
        raise ValueError("ownership central directory resource budget exceeds limit")
    if budget.archive_bytes > _MAX_ARCHIVE_BYTES:
        raise ValueError("ownership compressed archive resource budget exceeds limit")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _fingerprint(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _identity(status: os.stat_result) -> tuple[int, int]:
    return status.st_dev, status.st_ino


def _require_basename(name: str, *, label: str) -> str:
    if (
        not name
        or name in {".", ".."}
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
        or "\x00" in name
    ):
        raise ValueError(f"{label} must be a single safe basename")
    return name


def _open_directory_without_symlinks(path: Path) -> int:
    if not path.is_absolute():
        raise ValueError("directory path must be absolute")
    descriptor = os.open(os.sep, _DIRECTORY_OPEN_FLAGS)
    try:
        for component in path.parts[1:]:
            component = _require_basename(component, label="directory component")
            next_descriptor = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_parent_path_identity(
    parent: Path,
    expected_identity: tuple[int, int],
) -> None:
    try:
        current_fd = _open_directory_without_symlinks(parent)
    except (OSError, ValueError) as error:
        raise ValueError("compact checkpoint parent identity changed") from error
    try:
        if _identity(os.fstat(current_fd)) != expected_identity:
            raise ValueError("compact checkpoint parent identity changed")
    finally:
        os.close(current_fd)


def _create_temporary_directory_at(
    parent_fd: int,
    target_name: str,
) -> tuple[str, int, tuple[int, int]]:
    prefix = f".{_require_basename(target_name, label='target name')}.tmp-"
    for _ in range(128):
        temporary_name = prefix + secrets.token_hex(12)
        temporary_fd: int | None = None
        try:
            os.mkdir(temporary_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            created = os.stat(
                temporary_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            created_identity = _identity(created)
            temporary_fd = os.open(
                temporary_name,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=parent_fd,
            )
            opened = os.fstat(temporary_fd)
            named = os.stat(
                temporary_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(named.st_mode)
                or _identity(opened) != created_identity
                or _identity(named) != created_identity
            ):
                raise ValueError(
                    "compact checkpoint temporary directory identity changed"
                )
            return temporary_name, temporary_fd, _identity(opened)
        except BaseException:
            if temporary_fd is not None:
                os.close(temporary_fd)
            try:
                os.rmdir(temporary_name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
    raise FileExistsError("could not reserve a unique compact checkpoint temp name")


def _write_regular_at(
    directory_fd: int,
    name: str,
    content: bytes,
) -> tuple[int, int]:
    descriptor = os.open(
        _require_basename(name, label="checkpoint member"),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    identity = _identity(os.fstat(descriptor))
    succeeded = False
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError(errno.EIO, f"failed writing compact checkpoint {name}")
            remaining = remaining[written:]
        os.fsync(descriptor)
        if _identity(os.fstat(descriptor)) != identity:
            raise ValueError(f"compact checkpoint file identity changed: {name}")
        succeeded = True
        return identity
    finally:
        os.close(descriptor)
        if not succeeded:
            try:
                current = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if _identity(current) == identity:
                    os.unlink(name, dir_fd=directory_fd)
            except OSError:
                pass


def _rename_directory_no_replace_at(
    parent_fd: int,
    source_name: str,
    target_name: str,
    *,
    parent_path: Path,
    expected_parent_identity: tuple[int, int],
) -> None:
    source_name = _require_basename(source_name, label="temporary name")
    target_name = _require_basename(target_name, label="target name")
    _assert_parent_path_identity(parent_path, expected_parent_identity)
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise OSError(
            errno.ENOSYS,
            "dirfd-relative RENAME_NOREPLACE is unavailable",
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(target_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            errno.EEXIST,
            "immutable compact checkpoint target already exists",
            target_name,
        )
    raise OSError(
        error_number,
        "dirfd-relative compact checkpoint publication failed: "
        f"{os.strerror(error_number)}",
        f"{source_name} -> {target_name}",
    )


def _cleanup_owned_temporary(
    parent_fd: int,
    temporary_fd: int,
    temporary_name: str,
    temporary_identity: tuple[int, int],
    owned_files: dict[str, tuple[int, int]],
) -> None:
    for name, expected_identity in owned_files.items():
        try:
            current = os.stat(name, dir_fd=temporary_fd, follow_symlinks=False)
        except OSError:
            continue
        if _identity(current) != expected_identity:
            continue
        try:
            os.unlink(name, dir_fd=temporary_fd)
        except OSError:
            pass
    try:
        named = os.stat(
            temporary_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError:
        return
    if not stat.S_ISDIR(named.st_mode) or _identity(named) != temporary_identity:
        return
    try:
        os.rmdir(temporary_name, dir_fd=parent_fd)
    except OSError:
        pass


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            metadata = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink path component")


@dataclass(frozen=True)
class CompactOwnershipMetadata:
    scene_id: str
    frame_id: int
    timestamp: float
    revision: int
    voxel_size_m: float
    block_resolution: int
    dense_semantic_provenance: DenseSemanticProvenance
    format: str = COMPACT_OWNERSHIP_FORMAT
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.format != COMPACT_OWNERSHIP_FORMAT:
            raise ValueError("compact ownership format is invalid")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("compact ownership schema_version must be 1")
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        object.__setattr__(self, "scene_id", self.scene_id.strip())
        for name in ("frame_id", "revision"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not np.isfinite(self.timestamp)
        ):
            raise ValueError("timestamp must be finite numeric")
        if (
            isinstance(self.voxel_size_m, bool)
            or not isinstance(self.voxel_size_m, (int, float))
            or not np.isfinite(self.voxel_size_m)
            or self.voxel_size_m <= 0.0
        ):
            raise ValueError("voxel_size_m must be finite and positive")
        if type(self.block_resolution) is not int or self.block_resolution <= 0:
            raise ValueError("block_resolution must be a positive integer")
        provenance = self.dense_semantic_provenance
        if isinstance(provenance, dict):
            expected = {field.name for field in fields(DenseSemanticProvenance)}
            if set(provenance) != expected:
                raise ValueError("dense semantic provenance fields are not exact")
            provenance = DenseSemanticProvenance(**provenance)
            object.__setattr__(self, "dense_semantic_provenance", provenance)
        if not isinstance(provenance, DenseSemanticProvenance):
            raise TypeError(
                "dense_semantic_provenance must be DenseSemanticProvenance"
            )


@dataclass(frozen=True)
class CompactOwnershipSourceWitness:
    path: Path
    directory_fingerprint: tuple[int, int, int, int, int]
    file_fingerprints: dict[str, tuple[int, int, int, int, int]]

    @classmethod
    def capture_published(
        cls,
        path: Path,
        *,
        expected_file_fingerprints: dict[
            str, tuple[int, int, int, int, int]
        ],
    ) -> "CompactOwnershipSourceWitness":
        _reject_symlink_components(path, label="compact checkpoint source")
        try:
            path_before = os.lstat(path)
        except OSError as error:
            raise ValueError("compact checkpoint source changed") from error
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            directory_fd = os.open(path, flags)
        except OSError as error:
            raise ValueError("compact checkpoint source changed") from error
        try:
            directory_before = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(directory_before.st_mode)
                or _fingerprint(directory_before) != _fingerprint(path_before)
            ):
                raise ValueError("compact checkpoint source identity changed")
            if set(os.listdir(directory_fd)) != _INVENTORY:
                raise ValueError("compact checkpoint physical inventory is invalid")
            file_fingerprints: dict[
                str, tuple[int, int, int, int, int]
            ] = {}
            for name in sorted(_INVENTORY):
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if not stat.S_ISREG(current.st_mode):
                    raise ValueError(
                        f"compact checkpoint file is not regular: {name}"
                    )
                file_fingerprints[name] = _fingerprint(current)
            if file_fingerprints != expected_file_fingerprints:
                raise ValueError("compact checkpoint files changed during publication")
            directory_after = os.fstat(directory_fd)
            if _fingerprint(directory_before) != _fingerprint(directory_after):
                raise ValueError("compact checkpoint source identity changed")
        finally:
            os.close(directory_fd)
        try:
            path_after = os.lstat(path)
        except OSError as error:
            raise ValueError("compact checkpoint source changed") from error
        if _fingerprint(path_after) != _fingerprint(directory_after):
            raise ValueError("compact checkpoint source identity changed")
        witness = cls(
            path=path,
            directory_fingerprint=_fingerprint(directory_after),
            file_fingerprints=file_fingerprints,
        )
        witness.revalidate()
        return witness

    def revalidate(self) -> None:
        try:
            path_before = os.lstat(self.path)
        except OSError as error:
            raise ValueError("compact checkpoint source changed") from error
        if not stat.S_ISDIR(path_before.st_mode) or (
            _fingerprint(path_before) != self.directory_fingerprint
        ):
            raise ValueError("compact checkpoint source identity changed")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            directory_fd = os.open(self.path, flags)
        except OSError as error:
            raise ValueError("compact checkpoint source changed") from error
        try:
            directory = os.fstat(directory_fd)
            if _fingerprint(directory) != self.directory_fingerprint:
                raise ValueError("compact checkpoint source identity changed")
            try:
                inventory = set(os.listdir(directory_fd))
            except OSError as error:
                raise ValueError("compact checkpoint source changed") from error
            if inventory != set(self.file_fingerprints):
                raise ValueError("compact checkpoint inventory changed")
            for name, expected in self.file_fingerprints.items():
                try:
                    current = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise ValueError("compact checkpoint file changed") from error
                if not stat.S_ISREG(current.st_mode) or (
                    _fingerprint(current) != expected
                ):
                    raise ValueError("compact checkpoint file identity changed")
            if _fingerprint(os.fstat(directory_fd)) != self.directory_fingerprint:
                raise ValueError("compact checkpoint source identity changed")
        finally:
            os.close(directory_fd)
        try:
            path_after = os.lstat(self.path)
        except OSError as error:
            raise ValueError("compact checkpoint source changed") from error
        if _fingerprint(path_after) != self.directory_fingerprint:
            raise ValueError("compact checkpoint source identity changed")


@dataclass(frozen=True)
class CompactOwnershipCommitReceipt:
    path: Path
    metadata: CompactOwnershipMetadata
    checksums: dict[str, str]
    source_witness: CompactOwnershipSourceWitness

    def revalidate_source(self) -> None:
        self.source_witness.revalidate()


def _ownership_arrays(
    ownership: ReversibleOwnershipStore,
) -> dict[str, np.ndarray]:
    key_chunks: list[np.ndarray] = []
    entity_chunks: list[np.ndarray] = []
    confidence_chunks: list[np.ndarray] = []
    epoch_chunks: list[np.ndarray] = []
    revision_chunks: list[np.ndarray] = []
    resolution = ownership.block_resolution
    for block_key in sorted(ownership._blocks):
        block = ownership._blocks[block_key]
        local_keys = np.argwhere(block.entity_ids > 0)
        if local_keys.size == 0:
            continue
        positions = (local_keys[:, 0], local_keys[:, 1], local_keys[:, 2])
        key_chunks.append(
            local_keys + np.asarray(block_key, dtype=np.int64) * resolution
        )
        entity_chunks.append(block.entity_ids[positions])
        confidence_chunks.append(block.confidence[positions])
        epoch_chunks.append(block.epochs[positions])
        revision_chunks.append(block.evidence_revisions[positions])

    count = sum(chunk.shape[0] for chunk in key_chunks)
    if count > _MAX_OWNERSHIP_RECORDS:
        raise ValueError("ownership record count exceeds resource limit")
    if len(key_chunks) > _MAX_OWNERSHIP_BLOCKS:
        raise ValueError("ownership block count exceeds resource limit")
    dense_storage_bytes = len(key_chunks) * resolution**3 * 4 * 8
    if dense_storage_bytes > _MAX_DENSE_OWNERSHIP_BYTES:
        raise ValueError("ownership dense block storage exceeds resource limit")
    if key_chunks:
        voxel_keys = np.concatenate(key_chunks)
        entity_ids = np.concatenate(entity_chunks)
        confidence = np.concatenate(confidence_chunks)
        epochs = np.concatenate(epoch_chunks)
        evidence_revisions = np.concatenate(revision_chunks)
        order = np.lexsort(
            (voxel_keys[:, 2], voxel_keys[:, 1], voxel_keys[:, 0])
        )
        voxel_keys = voxel_keys[order]
        entity_ids = entity_ids[order]
        confidence = confidence[order]
        epochs = epochs[order]
        evidence_revisions = evidence_revisions[order]
    else:
        voxel_keys = np.empty((0, 3), dtype=np.int64)
        entity_ids = np.empty((0,), dtype=np.int64)
        confidence = np.empty((0,), dtype=np.float64)
        epochs = np.empty((0,), dtype=np.int64)
        evidence_revisions = np.empty((0,), dtype=np.int64)
    return {
        "schema_version": np.asarray([1], dtype=np.int64),
        "block_resolution": np.asarray(
            [ownership.block_resolution], dtype=np.int64
        ),
        "voxel_keys": voxel_keys,
        "entity_ids": entity_ids,
        "confidence": confidence,
        "epochs": epochs,
        "evidence_revisions": evidence_revisions,
    }


def _canonical_ownership_npz(ownership: ReversibleOwnershipStore) -> bytes:
    arrays = _ownership_arrays(ownership)
    _validate_canonical_archive_budget(
        _canonical_archive_budget(arrays["voxel_keys"].shape[0])
    )
    destination = io.BytesIO()
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=_OWNERSHIP_COMPRESSION_LEVEL,
        allowZip64=True,
    ) as archive:
        for name in _OWNERSHIP_ARRAYS:
            payload = io.BytesIO()
            npy_format.write_array(payload, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                payload.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
            )
    return destination.getvalue()


def _precheck_central_directory(content: bytes) -> None:
    search_start = max(0, len(content) - (65_535 + 22))
    search_end = len(content)
    eocd_offset = -1
    eocd: tuple[bytes, int, int, int, int, int, int, int] | None = None
    while search_end > search_start:
        candidate = content.rfind(_ZIP_EOCD_SIGNATURE, search_start, search_end)
        if candidate < 0:
            break
        if candidate + 22 <= len(content):
            parsed = struct.unpack_from("<4s4H2IH", content, candidate)
            if candidate + 22 + parsed[-1] == len(content):
                eocd_offset = candidate
                eocd = parsed
                break
        search_end = candidate
    if eocd is None:
        raise ValueError("ownership archive has no valid ZIP central directory")
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
    if disk_number != 0 or central_disk != 0 or disk_entries != total_entries:
        raise ValueError("ownership archive multi-disk ZIP is forbidden")
    if total_entries != len(_OWNERSHIP_MEMBERS):
        raise ValueError("ownership archive member count is invalid")
    if central_size > _MAX_CENTRAL_DIRECTORY_BYTES:
        raise ValueError("ownership archive central directory exceeds resource limit")
    if central_offset + central_size != eocd_offset:
        raise ValueError("ownership archive central directory offsets are invalid")


def _read_npy_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    name: str,
) -> _NpyHeader:
    try:
        with archive.open(info, mode="r") as member:
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
                raise ValueError(
                    f"ownership {name} has unsupported NPY version {version}"
                )
            header_bytes = member.tell()
    except (EOFError, OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("ownership"):
            raise
        raise ValueError(
            f"ownership {name} NPY header exceeds resource limit or is invalid"
        ) from error
    normalized_dtype = np.dtype(dtype)
    if normalized_dtype.hasobject:
        raise ValueError(f"ownership {name} object dtype is forbidden")
    element_count = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise ValueError(f"ownership {name} has an invalid NPY shape")
        element_count *= dimension
        if (
            element_count * normalized_dtype.itemsize
            > _MAX_MEMBER_UNCOMPRESSED_BYTES
        ):
            raise ValueError(f"ownership {name} NPY shape exceeds resource limit")
    if (
        name == "voxel_keys"
        and len(shape) == 2
        and shape[1:] == (3,)
        and shape[0] > _MAX_OWNERSHIP_RECORDS
    ):
        raise ValueError("ownership voxel_keys record count exceeds resource limit")
    expected_file_size = header_bytes + element_count * normalized_dtype.itemsize
    if expected_file_size != info.file_size:
        raise ValueError(f"ownership {name} NPY header/data size mismatch")
    return _NpyHeader(tuple(shape), normalized_dtype, header_bytes)


def _require_npy_contract(
    headers: dict[str, _NpyHeader],
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> None:
    header = headers[name]
    if header.dtype != dtype:
        raise ValueError(f"ownership {name} has an invalid NPY dtype")
    if header.shape != shape:
        raise ValueError(f"ownership {name} has an invalid NPY shape")


def _preflight_ownership_archive(content: bytes) -> None:
    if len(content) > _MAX_ARCHIVE_BYTES:
        raise ValueError("ownership archive exceeds compressed size limit")
    _precheck_central_directory(content)
    with zipfile.ZipFile(io.BytesIO(content), mode="r") as archive:
        infos = archive.infolist()
        counts = Counter(info.filename for info in infos)
        names = set(counts)
        if (
            len(infos) != len(_OWNERSHIP_MEMBERS)
            or names != _OWNERSHIP_MEMBERS
            or any(count != 1 for count in counts.values())
        ):
            raise ValueError("ownership archive inventory is invalid")
        info_by_name: dict[str, zipfile.ZipInfo] = {}
        total_uncompressed = 0
        for info in infos:
            name = info.filename.removesuffix(".npy")
            if (
                info.is_dir()
                or "/" in info.filename
                or "\\" in info.filename
                or info.flag_bits & 0x1
            ):
                raise ValueError(f"ownership {name} has an unsafe ZIP member")
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError(f"ownership {name} uses unsupported ZIP compression")
            if info.compress_size > _MAX_MEMBER_COMPRESSED_BYTES:
                raise ValueError(
                    f"ownership {name} compressed member exceeds resource limit"
                )
            if info.file_size > _MAX_MEMBER_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"ownership {name} uncompressed member exceeds resource limit"
                )
            if info.file_size > 0 and info.compress_size <= 0:
                raise ValueError(f"ownership {name} has invalid compressed size")
            total_uncompressed += info.file_size
            if total_uncompressed > _MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "ownership archive total uncompressed size exceeds resource limit"
                )
            info_by_name[name] = info

        headers = {
            name: _read_npy_header(archive, info_by_name[name], name)
            for name in _OWNERSHIP_ARRAYS
        }
        int64 = np.dtype(np.int64)
        _require_npy_contract(headers, "schema_version", (1,), int64)
        _require_npy_contract(headers, "block_resolution", (1,), int64)
        voxel_keys = headers["voxel_keys"]
        if voxel_keys.dtype != int64:
            raise ValueError("ownership voxel_keys have an invalid NPY dtype")
        if len(voxel_keys.shape) != 2 or voxel_keys.shape[1:] != (3,):
            raise ValueError("ownership voxel_keys have an invalid NPY shape")
        count = voxel_keys.shape[0]
        if count > _MAX_OWNERSHIP_RECORDS:
            raise ValueError("ownership voxel_keys record count exceeds resource limit")
        for name, dtype in {
            "entity_ids": int64,
            "confidence": np.dtype(np.float64),
            "epochs": int64,
            "evidence_revisions": int64,
        }.items():
            _require_npy_contract(headers, name, (count,), dtype)


def _validated_ownership_arrays(
    content: bytes,
    *,
    block_resolution: int,
    revision: int,
) -> _ValidatedOwnershipArrays:
    try:
        _preflight_ownership_archive(content)
        with np.load(io.BytesIO(content), allow_pickle=False) as payload:
            if set(payload.files) != set(_OWNERSHIP_ARRAYS):
                raise ValueError("ownership array inventory is invalid")
            arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    except (EOFError, OSError, ValueError, zipfile.BadZipFile, zlib.error) as error:
        if isinstance(error, ValueError) and str(error).startswith("ownership"):
            raise
        raise ValueError("ownership checkpoint is not a valid safe NPZ") from error

    exact = {
        "schema_version": (np.dtype(np.int64), (1,)),
        "block_resolution": (np.dtype(np.int64), (1,)),
    }
    for name, (dtype, shape) in exact.items():
        if arrays[name].dtype != dtype or arrays[name].shape != shape:
            raise ValueError(f"ownership {name} has an invalid dtype or shape")
    if int(arrays["schema_version"][0]) != 1:
        raise ValueError("ownership schema_version is invalid")
    if int(arrays["block_resolution"][0]) != block_resolution:
        raise ValueError("ownership block_resolution does not match metadata")

    voxel_keys = arrays["voxel_keys"]
    if (
        voxel_keys.dtype != np.int64
        or voxel_keys.ndim != 2
        or voxel_keys.shape[1:] != (3,)
    ):
        raise ValueError("ownership voxel_keys have an invalid dtype or shape")
    count = voxel_keys.shape[0]
    contracts = {
        "entity_ids": np.dtype(np.int64),
        "confidence": np.dtype(np.float64),
        "epochs": np.dtype(np.int64),
        "evidence_revisions": np.dtype(np.int64),
    }
    for name, dtype in contracts.items():
        if arrays[name].dtype != dtype or arrays[name].shape != (count,):
            raise ValueError(f"ownership {name} has an invalid dtype or shape")
    if count > 1:
        previous = voxel_keys[:-1]
        current = voxel_keys[1:]
        strictly_greater = (current[:, 0] > previous[:, 0]) | (
            (current[:, 0] == previous[:, 0])
            & (
                (current[:, 1] > previous[:, 1])
                | (
                    (current[:, 1] == previous[:, 1])
                    & (current[:, 2] > previous[:, 2])
                )
            )
        )
        if not np.all(strictly_greater):
            raise ValueError("ownership voxel_keys must be sorted and unique")
    if np.any(arrays["entity_ids"] <= 0):
        raise ValueError("ownership entity_ids must be positive")
    confidence = arrays["confidence"]
    if not np.all(np.isfinite(confidence)):
        raise ValueError("ownership confidence must be finite")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("ownership confidence must lie in [0, 1]")
    if np.any(arrays["epochs"] <= 0):
        raise ValueError("ownership epochs must be positive")
    if np.any(arrays["evidence_revisions"] < 0):
        raise ValueError("ownership evidence_revisions must be non-negative")
    if np.any(arrays["evidence_revisions"] > revision):
        raise ValueError(
            "ownership evidence_revisions cannot exceed checkpoint revision"
        )

    if count:
        dense_block_bytes = block_resolution**3 * 4 * 8
        if dense_block_bytes > _MAX_DENSE_OWNERSHIP_BYTES:
            raise ValueError(
                "ownership dense block storage exceeds resource limit"
            )
        block_keys = np.floor_divide(voxel_keys, block_resolution)
        local_keys = voxel_keys - block_keys * block_resolution
        by_block = np.lexsort(
            (block_keys[:, 2], block_keys[:, 1], block_keys[:, 0])
        )
        sorted_block_keys = block_keys[by_block]
        starts = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(
                    np.any(sorted_block_keys[1:] != sorted_block_keys[:-1], axis=1)
                )
                + 1,
            )
        )
        ends = np.concatenate((starts[1:], np.asarray([count], dtype=np.int64)))
        if len(starts) > _MAX_OWNERSHIP_BLOCKS:
            raise ValueError("ownership block count exceeds resource limit")
        dense_storage_bytes = len(starts) * dense_block_bytes
        if dense_storage_bytes > _MAX_DENSE_OWNERSHIP_BYTES:
            raise ValueError(
                "ownership dense block storage exceeds resource limit"
            )
    else:
        block_keys = np.empty((0, 3), dtype=np.int64)
        local_keys = np.empty((0, 3), dtype=np.int64)
        by_block = np.empty((0,), dtype=np.int64)
        sorted_block_keys = np.empty((0, 3), dtype=np.int64)
        starts = np.empty((0,), dtype=np.int64)
        ends = np.empty((0,), dtype=np.int64)
    return _ValidatedOwnershipArrays(
        arrays=arrays,
        block_keys=block_keys,
        local_keys=local_keys,
        by_block=by_block,
        sorted_block_keys=sorted_block_keys,
        starts=starts,
        ends=ends,
    )


def _materialize_ownership(
    validated: _ValidatedOwnershipArrays,
    *,
    block_resolution: int,
) -> ReversibleOwnershipStore:
    arrays = validated.arrays
    voxel_keys = arrays["voxel_keys"]
    confidence = arrays["confidence"]
    store = ReversibleOwnershipStore(block_resolution=block_resolution)
    for start, end in zip(validated.starts, validated.ends, strict=True):
        indices = validated.by_block[int(start) : int(end)]
        block_key = tuple(
            int(value) for value in validated.sorted_block_keys[int(start)]
        )
        block = store._new_block()
        local = validated.local_keys[indices]
        positions = (local[:, 0], local[:, 1], local[:, 2])
        block.entity_ids[positions] = arrays["entity_ids"][indices]
        block.confidence[positions] = confidence[indices]
        block.epochs[positions] = arrays["epochs"][indices]
        block.evidence_revisions[positions] = arrays[
            "evidence_revisions"
        ][indices]
        store._blocks[block_key] = block
    for row, entity_id in zip(voxel_keys, arrays["entity_ids"], strict=True):
        key = tuple(int(value) for value in row)
        store._entity_voxels.setdefault(int(entity_id), set()).add(key)
    return store


def _validated_ownership(
    content: bytes,
    *,
    block_resolution: int,
    revision: int,
) -> ReversibleOwnershipStore:
    return _materialize_ownership(
        _validated_ownership_arrays(
            content,
            block_resolution=block_resolution,
            revision=revision,
        ),
        block_resolution=block_resolution,
    )


def _read_regular_at(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"compact checkpoint file is not regular: {name}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"compact checkpoint file is not regular: {name}")
        if before.st_size > max_bytes:
            raise ValueError(f"compact checkpoint {name} exceeds size limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"compact checkpoint {name} exceeds size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _fingerprint(before) != _fingerprint(after):
            raise ValueError(f"compact checkpoint file changed while reading: {name}")
        return b"".join(chunks), _fingerprint(after)
    finally:
        os.close(descriptor)


def _capture_published_directory_at(
    parent_fd: int,
    target_name: str,
    expected_file_fingerprints: dict[
        str, tuple[int, int, int, int, int]
    ],
) -> tuple[int, int, int, int, int]:
    target_name = _require_basename(target_name, label="target name")
    directory_fd = os.open(
        target_name,
        _DIRECTORY_OPEN_FLAGS,
        dir_fd=parent_fd,
    )
    try:
        directory_before = os.fstat(directory_fd)
        named = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(named.st_mode)
            or _identity(named) != _identity(directory_before)
        ):
            raise ValueError("published compact checkpoint identity changed")
        if set(os.listdir(directory_fd)) != _INVENTORY:
            raise ValueError("published compact checkpoint inventory changed")
        actual_files: dict[str, tuple[int, int, int, int, int]] = {}
        for name in sorted(_INVENTORY):
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(current.st_mode):
                raise ValueError(
                    f"published compact checkpoint file is not regular: {name}"
                )
            actual_files[name] = _fingerprint(current)
        if actual_files != expected_file_fingerprints:
            raise ValueError("published compact checkpoint files changed")
        directory_after = os.fstat(directory_fd)
        if _fingerprint(directory_before) != _fingerprint(directory_after):
            raise ValueError("published compact checkpoint identity changed")
        return _fingerprint(directory_after)
    finally:
        os.close(directory_fd)


@dataclass(frozen=True)
class CompactOwnershipCheckpoint:
    path: Path
    metadata: CompactOwnershipMetadata
    ownership: ReversibleOwnershipStore
    checksums: dict[str, str]
    _source_witness: CompactOwnershipSourceWitness

    def revalidate_source(self) -> None:
        self._source_witness.revalidate()

    @classmethod
    def commit_new(
        cls,
        target_dir: str | Path,
        metadata: CompactOwnershipMetadata,
        ownership: ReversibleOwnershipStore,
    ) -> "CompactOwnershipCheckpoint":
        committed = cls._commit_new(
            target_dir,
            metadata,
            ownership,
            materialize=True,
        )
        assert isinstance(committed, cls)
        return committed

    @classmethod
    def commit_receipt_new(
        cls,
        target_dir: str | Path,
        metadata: CompactOwnershipMetadata,
        ownership: ReversibleOwnershipStore,
    ) -> CompactOwnershipCommitReceipt:
        committed = cls._commit_new(
            target_dir,
            metadata,
            ownership,
            materialize=False,
        )
        assert isinstance(committed, CompactOwnershipCommitReceipt)
        return committed

    @classmethod
    def _commit_new(
        cls,
        target_dir: str | Path,
        metadata: CompactOwnershipMetadata,
        ownership: ReversibleOwnershipStore,
        *,
        materialize: bool,
    ) -> "CompactOwnershipCheckpoint | CompactOwnershipCommitReceipt":
        if not isinstance(metadata, CompactOwnershipMetadata):
            raise TypeError("metadata must be CompactOwnershipMetadata")
        if not isinstance(ownership, ReversibleOwnershipStore):
            raise TypeError("ownership must be ReversibleOwnershipStore")
        if ownership.block_resolution != metadata.block_resolution:
            raise ValueError("ownership block_resolution does not match metadata")
        raw_target = Path(target_dir)
        _require_basename(raw_target.name, label="compact checkpoint target")
        target = Path(os.path.abspath(os.fspath(raw_target)))
        target_name = _require_basename(target.name, label="compact checkpoint target")
        _reject_symlink_components(target.parent, label="compact checkpoint parent")
        try:
            parent_fd = _open_directory_without_symlinks(target.parent)
        except FileNotFoundError as error:
            raise FileNotFoundError(target.parent) from error
        except OSError as error:
            raise ValueError(
                "compact checkpoint parent is not a real symlink-free directory"
            ) from error
        parent_status = os.fstat(parent_fd)
        parent_fingerprint = _fingerprint(parent_status)
        parent_identity = parent_fingerprint[:2]
        temporary_name: str | None = None
        temporary_fd: int | None = None
        temporary_identity: tuple[int, int] | None = None
        owned_files: dict[str, tuple[int, int]] = {}
        published = False
        try:
            _assert_parent_path_identity(target.parent, parent_identity)
            _publication_test_hook("after_parent_check")
            _assert_parent_path_identity(target.parent, parent_identity)
            try:
                os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(target)

            temporary_name, temporary_fd, temporary_identity = (
                _create_temporary_directory_at(parent_fd, target_name)
            )
            _publication_test_hook("after_temp_create")
            _assert_parent_path_identity(target.parent, parent_identity)

            metadata_bytes = _canonical_json(asdict(metadata))
            ownership_bytes = _canonical_ownership_npz(ownership)
            checksums = {
                "metadata.json": _sha256(metadata_bytes),
                "ownership.npz": _sha256(ownership_bytes),
            }
            files = {
                "metadata.json": metadata_bytes,
                "ownership.npz": ownership_bytes,
                "checksums.json": _canonical_json(checksums),
            }
            validated_arrays = _validated_ownership_arrays(
                ownership_bytes,
                block_resolution=metadata.block_resolution,
                revision=metadata.revision,
            )
            validated_ownership = (
                _materialize_ownership(
                    validated_arrays,
                    block_resolution=metadata.block_resolution,
                )
                if materialize
                else None
            )
            for name, content in files.items():
                owned_files[name] = _write_regular_at(
                    temporary_fd,
                    name,
                    content,
                )
            os.fsync(temporary_fd)
            named_temporary = os.stat(
                temporary_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(named_temporary.st_mode)
                or _identity(named_temporary) != temporary_identity
                or _identity(os.fstat(temporary_fd)) != temporary_identity
            ):
                raise ValueError(
                    "compact checkpoint temporary directory identity changed"
                )
            if set(os.listdir(temporary_fd)) != _INVENTORY:
                raise ValueError("compact checkpoint temporary inventory changed")
            file_fingerprints: dict[
                str, tuple[int, int, int, int, int]
            ] = {}
            for name in sorted(_INVENTORY):
                max_bytes = (
                    _MAX_ARCHIVE_BYTES
                    if name == "ownership.npz"
                    else _MAX_JSON_BYTES
                )
                persisted, file_fingerprints[name] = _read_regular_at(
                    temporary_fd,
                    name,
                    max_bytes=max_bytes,
                )
                if persisted != files[name]:
                    raise ValueError(
                        f"compact checkpoint file changed after write: {name}"
                    )

            _publication_test_hook("before_rename")
            _assert_parent_path_identity(target.parent, parent_identity)
            _rename_directory_no_replace_at(
                parent_fd,
                temporary_name,
                target_name,
                parent_path=target.parent,
                expected_parent_identity=parent_identity,
            )
            published = True
            os.fsync(parent_fd)
            _publication_test_hook("after_rename")
            _assert_parent_path_identity(target.parent, parent_identity)
            if _identity(os.fstat(parent_fd)) != parent_identity:
                raise ValueError("anchored compact checkpoint parent changed")
            anchored_directory_fingerprint = _capture_published_directory_at(
                parent_fd,
                target_name,
                file_fingerprints,
            )
            witness = CompactOwnershipSourceWitness.capture_published(
                target,
                expected_file_fingerprints=file_fingerprints,
            )
            if witness.directory_fingerprint != anchored_directory_fingerprint:
                raise ValueError("published compact checkpoint path identity changed")
            _assert_parent_path_identity(target.parent, parent_identity)
            if materialize:
                assert validated_ownership is not None
                return cls(
                    target,
                    metadata,
                    validated_ownership,
                    dict(checksums),
                    witness,
                )
            return CompactOwnershipCommitReceipt(
                path=target,
                metadata=metadata,
                checksums=dict(checksums),
                source_witness=witness,
            )
        except CompactCheckpointPublicationUncertainError:
            raise
        except Exception as error:
            if published:
                raise CompactCheckpointPublicationUncertainError(
                    target,
                    error,
                ) from error
            raise
        finally:
            if (
                not published
                and temporary_name is not None
                and temporary_fd is not None
                and temporary_identity is not None
            ):
                _cleanup_owned_temporary(
                    parent_fd,
                    temporary_fd,
                    temporary_name,
                    temporary_identity,
                    owned_files,
                )
            if temporary_fd is not None:
                os.close(temporary_fd)
            os.close(parent_fd)

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> "CompactOwnershipCheckpoint":
        source = Path(checkpoint_dir).absolute()
        _reject_symlink_components(source, label="compact checkpoint source")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            directory_fd = os.open(source, flags)
        except OSError as error:
            raise ValueError(f"compact checkpoint source is not a real directory: {source}") from error
        try:
            directory_before = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_before.st_mode):
                raise ValueError("compact checkpoint source is not a directory")
            inventory = set(os.listdir(directory_fd))
            if inventory != _INVENTORY:
                raise ValueError("compact checkpoint physical inventory is invalid")
            contents: dict[str, bytes] = {}
            fingerprints: dict[str, tuple[int, int, int, int, int]] = {}
            for name in sorted(_INVENTORY):
                max_bytes = (
                    _MAX_ARCHIVE_BYTES
                    if name == "ownership.npz"
                    else _MAX_JSON_BYTES
                )
                contents[name], fingerprints[name] = _read_regular_at(
                    directory_fd,
                    name,
                    max_bytes=max_bytes,
                )
            directory_after = os.fstat(directory_fd)
            if _fingerprint(directory_before) != _fingerprint(directory_after):
                raise ValueError("compact checkpoint directory changed while loading")
        finally:
            os.close(directory_fd)

        checksums = _strict_json(contents["checksums.json"], label="checksums")
        if set(checksums) != _CHECKSUM_FILES:
            raise ValueError("compact checkpoint checksum inventory is invalid")
        for name in sorted(_CHECKSUM_FILES):
            checksum = checksums[name]
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
            ):
                raise ValueError("compact checkpoint checksum is invalid")
            if _sha256(contents[name]) != checksum:
                raise ValueError(f"compact checkpoint checksum mismatch for {name}")

        metadata_payload = _strict_json(contents["metadata.json"], label="metadata")
        expected_metadata = {
            "format",
            "schema_version",
            "scene_id",
            "frame_id",
            "timestamp",
            "revision",
            "voxel_size_m",
            "block_resolution",
            "dense_semantic_provenance",
        }
        if set(metadata_payload) != expected_metadata:
            raise ValueError("compact checkpoint metadata fields are not exact")
        metadata = CompactOwnershipMetadata(**metadata_payload)
        ownership = _validated_ownership(
            contents["ownership.npz"],
            block_resolution=metadata.block_resolution,
            revision=metadata.revision,
        )
        witness = CompactOwnershipSourceWitness(
            path=source,
            directory_fingerprint=_fingerprint(directory_after),
            file_fingerprints=fingerprints,
        )
        result = cls(source, metadata, ownership, dict(checksums), witness)
        result.revalidate_source()
        return result
