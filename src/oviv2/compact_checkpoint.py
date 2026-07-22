from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any
import zipfile

import numpy as np
from numpy.lib import format as npy_format

from src.oviv2.dense_semantics import DenseSemanticProvenance
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.snapshot import VoxelMapSnapshot


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
class _SourceWitness:
    path: Path
    directory_fingerprint: tuple[int, int, int, int, int]
    file_fingerprints: dict[str, tuple[int, int, int, int, int]]

    def revalidate(self) -> None:
        try:
            directory = os.lstat(self.path)
        except OSError as error:
            raise ValueError("compact checkpoint source changed") from error
        if not stat.S_ISDIR(directory.st_mode) or (
            _fingerprint(directory) != self.directory_fingerprint
        ):
            raise ValueError("compact checkpoint source identity changed")
        try:
            inventory = {entry.name for entry in os.scandir(self.path)}
        except OSError as error:
            raise ValueError("compact checkpoint source changed") from error
        if inventory != set(self.file_fingerprints):
            raise ValueError("compact checkpoint inventory changed")
        for name, expected in self.file_fingerprints.items():
            try:
                current = os.lstat(self.path / name)
            except OSError as error:
                raise ValueError("compact checkpoint file changed") from error
            if not stat.S_ISREG(current.st_mode) or _fingerprint(current) != expected:
                raise ValueError("compact checkpoint file identity changed")


def _ownership_arrays(
    ownership: ReversibleOwnershipStore,
) -> dict[str, np.ndarray]:
    records = ownership.records()
    return {
        "schema_version": np.asarray([1], dtype=np.int64),
        "block_resolution": np.asarray(
            [ownership.block_resolution], dtype=np.int64
        ),
        "voxel_keys": np.asarray(
            [key for key, _ in records], dtype=np.int64
        ).reshape(-1, 3),
        "entity_ids": np.asarray(
            [record.entity_id for _, record in records], dtype=np.int64
        ),
        "confidence": np.asarray(
            [record.confidence for _, record in records], dtype=np.float64
        ),
        "epochs": np.asarray(
            [record.epoch for _, record in records], dtype=np.int64
        ),
        "evidence_revisions": np.asarray(
            [record.evidence_revision for _, record in records], dtype=np.int64
        ),
    }


def _canonical_ownership_npz(ownership: ReversibleOwnershipStore) -> bytes:
    arrays = _ownership_arrays(ownership)
    destination = io.BytesIO()
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for name in _OWNERSHIP_ARRAYS:
            payload = io.BytesIO()
            npy_format.write_array(payload, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return destination.getvalue()


def _validated_ownership(content: bytes, *, block_resolution: int) -> ReversibleOwnershipStore:
    try:
        with zipfile.ZipFile(io.BytesIO(content), mode="r") as archive:
            names = [entry.filename for entry in archive.infolist()]
            if len(names) != len(set(names)):
                raise ValueError("ownership archive contains duplicate members")
            if set(names) != _OWNERSHIP_MEMBERS:
                raise ValueError("ownership archive inventory is invalid")
            if any(entry.is_dir() for entry in archive.infolist()):
                raise ValueError("ownership archive contains a directory")
        with np.load(io.BytesIO(content), allow_pickle=False) as payload:
            if set(payload.files) != set(_OWNERSHIP_ARRAYS):
                raise ValueError("ownership array inventory is invalid")
            arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
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
    if voxel_keys.dtype != np.int64 or voxel_keys.ndim != 2 or voxel_keys.shape[1:] != (3,):
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
    keys = [tuple(int(value) for value in row) for row in voxel_keys]
    if keys != sorted(set(keys)):
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

    store = ReversibleOwnershipStore(block_resolution=block_resolution)
    for index, key in enumerate(keys):
        store.assign(
            key,
            int(arrays["entity_ids"][index]),
            float(confidence[index]),
            int(arrays["evidence_revisions"][index]),
        )
        block, local, _ = store._location(key, allocate=False)
        assert block is not None
        block.epochs[local] = arrays["epochs"][index]
    return store


def _read_regular_at(
    directory_fd: int,
    name: str,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"compact checkpoint file is not regular: {name}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"compact checkpoint file is not regular: {name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _fingerprint(before) != _fingerprint(after):
            raise ValueError(f"compact checkpoint file changed while reading: {name}")
        return b"".join(chunks), _fingerprint(after)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class CompactOwnershipCheckpoint:
    path: Path
    metadata: CompactOwnershipMetadata
    ownership: ReversibleOwnershipStore
    checksums: dict[str, str]
    _source_witness: _SourceWitness

    def revalidate_source(self) -> None:
        self._source_witness.revalidate()

    @classmethod
    def commit_new(
        cls,
        target_dir: str | Path,
        metadata: CompactOwnershipMetadata,
        ownership: ReversibleOwnershipStore,
    ) -> "CompactOwnershipCheckpoint":
        if not isinstance(metadata, CompactOwnershipMetadata):
            raise TypeError("metadata must be CompactOwnershipMetadata")
        if not isinstance(ownership, ReversibleOwnershipStore):
            raise TypeError("ownership must be ReversibleOwnershipStore")
        if ownership.block_resolution != metadata.block_resolution:
            raise ValueError("ownership block_resolution does not match metadata")
        target = Path(target_dir).absolute()
        _reject_symlink_components(target.parent, label="compact checkpoint parent")
        try:
            parent_status = os.lstat(target.parent)
        except FileNotFoundError as error:
            raise FileNotFoundError(target.parent) from error
        if not stat.S_ISDIR(parent_status.st_mode):
            raise NotADirectoryError(target.parent)
        if os.path.lexists(target):
            raise FileExistsError(target)

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        )
        published = False
        try:
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
            for name, content in files.items():
                path = temporary / name
                descriptor = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                )
                try:
                    with os.fdopen(descriptor, "wb", closefd=False) as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                finally:
                    os.close(descriptor)
            VoxelMapSnapshot._fsync_directory(temporary)
            cls.load(temporary)
            VoxelMapSnapshot._publish_directory_no_replace(temporary, target)
            published = True
            return cls.load(target)
        finally:
            if not published and temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

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
                contents[name], fingerprints[name] = _read_regular_at(
                    directory_fd, name
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
        )
        witness = _SourceWitness(
            path=source,
            directory_fingerprint=_fingerprint(directory_after),
            file_fingerprints=fingerprints,
        )
        result = cls(source, metadata, ownership, dict(checksums), witness)
        result.revalidate_source()
        return result
