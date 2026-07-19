from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

import numpy as np

from src.oviv2.evidence import EvidenceConfig, SparseEvidenceStore
from src.oviv2.entities import EntityRegistry
from src.oviv2.geometry import SparseTsdfVolume, TsdfConfig
from src.oviv2.ownership import ReversibleOwnershipStore


class _ConcurrentSnapshotChange(RuntimeError):
    pass


@dataclass(frozen=True)
class VoxelSnapshotMetadata:
    scene_id: str
    frame_id: int
    timestamp: float
    revision: int
    voxel_size_m: float
    block_resolution: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        for name in ("frame_id", "revision"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not np.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        if not np.isfinite(float(self.voxel_size_m)) or self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive")
        if (
            not isinstance(self.block_resolution, int)
            or isinstance(self.block_resolution, bool)
            or self.block_resolution <= 0
        ):
            raise ValueError("block_resolution must be a positive integer")
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version not in {1, 2}
        ):
            raise ValueError("unsupported schema_version")


@dataclass(frozen=True)
class VoxelMapSnapshot:
    path: Path
    metadata: VoxelSnapshotMetadata
    geometry: SparseTsdfVolume
    evidence: SparseEvidenceStore
    ownership: ReversibleOwnershipStore
    checksums: dict[str, str]
    registry: EntityRegistry | None = None

    _DATA_FILES_V1 = ("metadata.json", "geometry.npz", "evidence.npz", "ownership.npz")
    _DATA_FILES_V2 = (*_DATA_FILES_V1, "entities.jsonl")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    @classmethod
    def _validate_components(
        cls,
        metadata: VoxelSnapshotMetadata,
        geometry: SparseTsdfVolume,
        evidence: SparseEvidenceStore,
        ownership: ReversibleOwnershipStore,
        registry: EntityRegistry | None,
    ) -> None:
        if not np.isclose(
            geometry.config.voxel_size_m,
            metadata.voxel_size_m,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError("geometry voxel_size_m does not match snapshot metadata")
        resolutions = {
            metadata.block_resolution,
            geometry.config.block_resolution,
            evidence.config.block_resolution,
            ownership.block_resolution,
        }
        if len(resolutions) != 1:
            raise ValueError("block_resolution mismatch across snapshot components")
        for voxel_key, owner in ownership.records():
            candidate_ids = {item.entity_id for item in evidence.entity_candidates(voxel_key)}
            if owner.entity_id not in candidate_ids:
                raise ValueError("ownership record has no matching entity evidence")
        if metadata.schema_version == 1:
            if registry is not None:
                raise ValueError("schema v1 snapshot forbids a registry")
            return
        if not isinstance(registry, EntityRegistry):
            raise ValueError("schema v2 snapshot requires a registry")
        referenced_entity_ids = {
            int(entity_id)
            for block in evidence._blocks.values()
            for entity_id in np.unique(block.entity_ids)
            if entity_id > 0
        }
        referenced_entity_ids.update(
            owner.entity_id for _, owner in ownership.records()
        )
        missing = referenced_entity_ids.difference(registry.entities)
        if missing:
            raise ValueError(f"registry is missing referenced entity IDs {sorted(missing)}")
        if registry._last_revision > metadata.revision:
            raise ValueError("registry watermark revision exceeds snapshot metadata revision")
        if any(
            entity.last_revision > metadata.revision
            for entity in registry.entities.values()
        ):
            raise ValueError("registry entity revision exceeds snapshot metadata revision")

    @classmethod
    def _data_files(cls, schema_version: int) -> tuple[str, ...]:
        return cls._DATA_FILES_V1 if schema_version == 1 else cls._DATA_FILES_V2

    @classmethod
    def _exchange_directories(cls, left: Path, right: Path) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as exc:
            raise OSError(
                errno.ENOSYS,
                "renameat2 is unavailable; atomic snapshot overwrite is unsupported",
            ) from exc
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
            -100,
            os.fsencode(left),
            -100,
            os.fsencode(right),
            2,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                f"atomic directory exchange failed: {os.strerror(error_number)}",
                f"{left} <-> {right}",
            )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @classmethod
    def commit(
        cls,
        target_dir: str | Path,
        metadata: VoxelSnapshotMetadata,
        geometry: SparseTsdfVolume,
        evidence: SparseEvidenceStore,
        ownership: ReversibleOwnershipStore,
        *,
        registry: EntityRegistry | None = None,
    ) -> "VoxelMapSnapshot":
        if not isinstance(metadata, VoxelSnapshotMetadata):
            raise TypeError("metadata must be VoxelSnapshotMetadata")
        cls._validate_components(metadata, geometry, evidence, ownership, registry)
        target = Path(target_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not target.is_dir():
            raise ValueError("snapshot target exists and is not a directory")

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        )
        try:
            cls._write_json(temporary / "metadata.json", asdict(metadata))
            geometry.save(temporary / "geometry.npz")
            evidence.save(temporary / "evidence.npz")
            ownership.save(temporary / "ownership.npz")
            if registry is not None:
                registry.save(temporary / "entities.jsonl")
            data_files = cls._data_files(metadata.schema_version)
            checksums = {
                name: cls._sha256(temporary / name)
                for name in data_files
            }
            cls._write_json(temporary / "checksums.json", checksums)
            cls._fsync_directory(temporary)
            cls.load(temporary)

            if target.exists():
                cls._exchange_directories(target, temporary)
            else:
                os.replace(temporary, target)
            cls._fsync_directory(target.parent)
            return cls.load(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    @classmethod
    def load(cls, snapshot_dir: str | Path) -> "VoxelMapSnapshot":
        source = Path(snapshot_dir)
        if not source.is_dir():
            raise FileNotFoundError(source)
        last_concurrent_change: _ConcurrentSnapshotChange | None = None
        for _ in range(3):
            try:
                return cls._load_once(source)
            except _ConcurrentSnapshotChange as exc:
                last_concurrent_change = exc
        raise ValueError("snapshot changed concurrently during load") from last_concurrent_change

    @staticmethod
    def _directory_identity(source: Path) -> tuple[int, int]:
        source_stat = source.stat()
        if not stat.S_ISDIR(source_stat.st_mode):
            raise FileNotFoundError(source)
        return source_stat.st_dev, source_stat.st_ino

    @classmethod
    def _source_changed(
        cls,
        source: Path,
        identity: tuple[int, int],
        manifest_bytes: bytes | None,
    ) -> bool:
        try:
            if cls._directory_identity(source) != identity:
                return True
            if manifest_bytes is not None:
                return (source / "checksums.json").read_bytes() != manifest_bytes
        except OSError:
            return True
        return False

    @classmethod
    def _checksums_match(cls, source: Path, checksums: dict[str, str]) -> bool:
        try:
            return all(
                (source / name).is_file()
                and cls._sha256(source / name) == expected
                for name, expected in sorted(checksums.items())
            )
        except OSError:
            return False

    @classmethod
    def _load_once(cls, source: Path) -> "VoxelMapSnapshot":
        try:
            identity = cls._directory_identity(source)
        except OSError as exc:
            raise _ConcurrentSnapshotChange("snapshot directory changed") from exc
        checksums_path = source / "checksums.json"
        if not checksums_path.is_file():
            if cls._source_changed(source, identity, None):
                raise _ConcurrentSnapshotChange("snapshot directory changed")
            raise ValueError("snapshot is missing checksums.json")
        try:
            manifest_bytes = checksums_path.read_bytes()
            checksums = json.loads(manifest_bytes.decode("utf-8"))
        except OSError as exc:
            raise _ConcurrentSnapshotChange("snapshot checksum manifest changed") from exc
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if cls._source_changed(source, identity, None):
                raise _ConcurrentSnapshotChange(
                    "snapshot checksum manifest changed"
                ) from exc
            raise ValueError(f"invalid snapshot checksum manifest: {exc}") from exc
        if not isinstance(checksums, dict):
            raise ValueError("snapshot checksum manifest must be an object")
        checksum_files = set(checksums)
        if (
            checksum_files != set(cls._DATA_FILES_V1)
            and checksum_files != set(cls._DATA_FILES_V2)
        ):
            raise ValueError("snapshot checksum manifest has unexpected files")
        for name in sorted(checksum_files):
            data_path = source / name
            try:
                checksum_matches = (
                    data_path.is_file()
                    and cls._sha256(data_path) == checksums[name]
                )
            except OSError:
                checksum_matches = False
            if not checksum_matches:
                if cls._source_changed(source, identity, manifest_bytes):
                    raise _ConcurrentSnapshotChange("snapshot directory changed")
                raise ValueError(f"snapshot checksum mismatch for {name}")

        try:
            with (source / "metadata.json").open("r", encoding="utf-8") as stream:
                metadata_payload = json.load(stream)
            metadata = VoxelSnapshotMetadata(**metadata_payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if cls._source_changed(source, identity, manifest_bytes) or not cls._checksums_match(
                source, checksums
            ):
                raise _ConcurrentSnapshotChange(
                    "snapshot data changed during load"
                ) from exc
            raise ValueError(f"invalid snapshot metadata: {exc}") from exc
        data_files = cls._data_files(metadata.schema_version)
        if checksum_files != set(data_files):
            raise ValueError("snapshot schema and checksum files do not match")
        try:
            physical_files = {path.name for path in source.iterdir()}
        except OSError as exc:
            raise _ConcurrentSnapshotChange("snapshot directory changed") from exc
        if physical_files != {*data_files, "checksums.json"}:
            raise ValueError("snapshot physical files do not match schema")

        try:
            tsdf_config = TsdfConfig(
                voxel_size_m=metadata.voxel_size_m,
                block_resolution=metadata.block_resolution,
            )
            geometry = SparseTsdfVolume.load(source / "geometry.npz", tsdf_config)
            with np.load(source / "evidence.npz", allow_pickle=False) as payload:
                evidence_config = EvidenceConfig(
                    block_resolution=int(payload["block_resolution"][0]),
                    semantic_top_k=int(payload["semantic_top_k"][0]),
                    entity_top_k=int(payload["entity_top_k"][0]),
                )
            evidence = SparseEvidenceStore.load(
                source / "evidence.npz", evidence_config
            )
            ownership = ReversibleOwnershipStore.load(
                source / "ownership.npz",
                block_resolution=metadata.block_resolution,
            )
            registry = (
                EntityRegistry.load(source / "entities.jsonl")
                if metadata.schema_version == 2
                else None
            )
            cls._validate_components(
                metadata, geometry, evidence, ownership, registry
            )
        except Exception as exc:
            if cls._source_changed(source, identity, manifest_bytes) or not cls._checksums_match(
                source, checksums
            ):
                raise _ConcurrentSnapshotChange(
                    "snapshot data changed during load"
                ) from exc
            raise
        result = cls(
            source,
            metadata,
            geometry,
            evidence,
            ownership,
            dict(checksums),
            registry,
        )
        if not cls._checksums_match(source, checksums):
            raise _ConcurrentSnapshotChange("snapshot data changed during load")
        if cls._source_changed(source, identity, manifest_bytes):
            raise _ConcurrentSnapshotChange("snapshot directory changed during load")
        return result
