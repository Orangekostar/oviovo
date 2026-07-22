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

from src.oviv2.dense_semantics import DenseSemanticProvenance
from src.oviv2.evidence import EvidenceConfig, SparseEvidenceStore
from src.oviv2.entities import EntityRegistry
from src.oviv2.geometry import SparseTsdfVolume, TsdfConfig
from src.oviv2.ownership import ReversibleOwnershipStore


class _ConcurrentSnapshotChange(RuntimeError):
    pass


class _SnapshotRollbackError(RuntimeError):
    def __init__(
        self,
        publication_error: Exception,
        rollback_error: Exception,
    ) -> None:
        self.publication_error = publication_error
        self.rollback_error = rollback_error
        super().__init__(
            "snapshot publication failed "
            f"({publication_error}); rollback failed ({rollback_error}); "
            "target state is uncertain"
        )


@dataclass(frozen=True)
class VoxelSnapshotMetadata:
    scene_id: str
    frame_id: int
    timestamp: float
    revision: int
    voxel_size_m: float
    block_resolution: int
    schema_version: int = 1
    dense_semantic_provenance: DenseSemanticProvenance | None = None

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
            or self.schema_version not in {1, 2, 3}
        ):
            raise ValueError("unsupported schema_version")

        provenance = self.dense_semantic_provenance
        if isinstance(provenance, dict):
            provenance = DenseSemanticProvenance(**provenance)
            object.__setattr__(self, "dense_semantic_provenance", provenance)
        if provenance is not None and not isinstance(
            provenance,
            DenseSemanticProvenance,
        ):
            raise TypeError(
                "dense_semantic_provenance must be DenseSemanticProvenance or None"
            )
        if self.schema_version in {1, 2} and provenance is not None:
            raise ValueError(
                "schema v1/v2 metadata forbids dense semantic provenance"
            )
        if self.schema_version == 3 and provenance is None:
            raise ValueError("schema v3 metadata requires dense semantic provenance")


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
            raise ValueError(
                f"schema v{metadata.schema_version} snapshot requires a registry"
            )
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

    @classmethod
    def _rename_directory_no_replace(cls, source: Path, target: Path) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as exc:
            raise NotImplementedError(
                "renameat2 is unavailable; immutable snapshot publication is unsupported"
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
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                errno.EEXIST,
                "immutable snapshot target already exists",
                target,
            )
        if error_number in {
            errno.ENOSYS,
            errno.EINVAL,
            getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
        }:
            raise NotImplementedError(
                "RENAME_NOREPLACE is unavailable; immutable snapshot publication is unsupported"
            )
        raise OSError(
            error_number,
            f"immutable snapshot publication failed: {os.strerror(error_number)}",
            f"{source} -> {target}",
        )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @classmethod
    def _write_snapshot_files(
        cls,
        destination: Path,
        metadata: VoxelSnapshotMetadata,
        geometry: SparseTsdfVolume,
        evidence: SparseEvidenceStore,
        ownership: ReversibleOwnershipStore,
        registry: EntityRegistry | None,
    ) -> tuple[str, ...]:
        metadata_payload = asdict(metadata)
        if metadata.schema_version in {1, 2}:
            metadata_payload.pop("dense_semantic_provenance")
        cls._write_json(destination / "metadata.json", metadata_payload)
        geometry.save(destination / "geometry.npz")
        evidence.save(destination / "evidence.npz")
        ownership.save(destination / "ownership.npz")
        if registry is not None:
            registry.save(destination / "entities.jsonl")
        data_files = cls._data_files(metadata.schema_version)
        checksums = {
            name: cls._sha256(destination / name)
            for name in data_files
        }
        cls._write_json(destination / "checksums.json", checksums)
        return data_files

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
        exchanged = False
        published = False
        try:
            cls._write_snapshot_files(
                temporary,
                metadata,
                geometry,
                evidence,
                ownership,
                registry,
            )
            cls._fsync_directory(temporary)
            cls.load(temporary)

            if target.exists():
                cls._exchange_directories(target, temporary)
                exchanged = True
            else:
                os.replace(temporary, target)
            try:
                cls._fsync_directory(target.parent)
                restored = cls.load(target)
            except Exception as publication_error:
                if exchanged:
                    try:
                        cls._exchange_directories(target, temporary)
                        exchanged = False
                        cls._fsync_directory(target.parent)
                    except Exception as rollback_error:
                        raise _SnapshotRollbackError(
                            publication_error,
                            rollback_error,
                        ) from publication_error
                else:
                    try:
                        if target.exists():
                            shutil.rmtree(target)
                        cls._fsync_directory(target.parent)
                    except Exception as cleanup_error:
                        raise _SnapshotRollbackError(
                            publication_error,
                            cleanup_error,
                        ) from publication_error
                raise
            published = True
            if temporary.exists():
                try:
                    shutil.rmtree(temporary)
                except OSError:
                    pass
            return restored
        finally:
            if not published and not exchanged and temporary.exists():
                try:
                    shutil.rmtree(temporary)
                except OSError:
                    pass

    @classmethod
    def commit_new(
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
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        )
        published = False
        published_identity = cls._directory_identity(temporary)
        try:
            data_files = cls._write_snapshot_files(
                temporary,
                metadata,
                geometry,
                evidence,
                ownership,
                registry,
            )
            for name in (*data_files, "checksums.json"):
                file_descriptor = os.open(temporary / name, os.O_RDONLY)
                try:
                    os.fsync(file_descriptor)
                finally:
                    os.close(file_descriptor)
            cls._fsync_directory(temporary)
            cls.load(temporary)

            cls._rename_directory_no_replace(temporary, target)
            published = True
            cls._fsync_directory(target.parent)
            return cls.load(target)
        except Exception as publication_error:
            if published:
                try:
                    if cls._directory_identity(target) == published_identity:
                        shutil.rmtree(target)
                except FileNotFoundError:
                    pass
                except Exception as cleanup_error:
                    raise _SnapshotRollbackError(
                        publication_error,
                        cleanup_error,
                    ) from publication_error
                finally:
                    try:
                        cls._fsync_directory(target.parent)
                    except OSError:
                        pass
            raise
        finally:
            if temporary.exists():
                try:
                    shutil.rmtree(temporary)
                except OSError:
                    pass

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
                if metadata.schema_version in {2, 3}
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
