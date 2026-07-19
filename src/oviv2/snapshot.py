from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np

from src.oviv2.evidence import EvidenceConfig, SparseEvidenceStore
from src.oviv2.geometry import SparseTsdfVolume, TsdfConfig
from src.oviv2.ownership import ReversibleOwnershipStore


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
        if self.schema_version != 1:
            raise ValueError("unsupported schema_version")


@dataclass(frozen=True)
class VoxelMapSnapshot:
    path: Path
    metadata: VoxelSnapshotMetadata
    geometry: SparseTsdfVolume
    evidence: SparseEvidenceStore
    ownership: ReversibleOwnershipStore
    checksums: dict[str, str]

    _DATA_FILES = ("metadata.json", "geometry.npz", "evidence.npz", "ownership.npz")

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

    @classmethod
    def commit(
        cls,
        target_dir: str | Path,
        metadata: VoxelSnapshotMetadata,
        geometry: SparseTsdfVolume,
        evidence: SparseEvidenceStore,
        ownership: ReversibleOwnershipStore,
    ) -> "VoxelMapSnapshot":
        if not isinstance(metadata, VoxelSnapshotMetadata):
            raise TypeError("metadata must be VoxelSnapshotMetadata")
        cls._validate_components(metadata, geometry, evidence, ownership)
        target = Path(target_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not target.is_dir():
            raise ValueError("snapshot target exists and is not a directory")

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        )
        backup: Path | None = None
        try:
            cls._write_json(temporary / "metadata.json", asdict(metadata))
            geometry.save(temporary / "geometry.npz")
            evidence.save(temporary / "evidence.npz")
            ownership.save(temporary / "ownership.npz")
            checksums = {
                name: cls._sha256(temporary / name)
                for name in cls._DATA_FILES
            }
            cls._write_json(temporary / "checksums.json", checksums)
            directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            cls.load(temporary)

            if target.exists():
                backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
                os.replace(target, backup)
            try:
                os.replace(temporary, target)
            except BaseException:
                if backup is not None and backup.exists() and not target.exists():
                    os.replace(backup, target)
                raise
            if backup is not None:
                shutil.rmtree(backup)
                backup = None
            parent_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            return cls.load(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
            if backup is not None and backup.exists():
                shutil.rmtree(backup)

    @classmethod
    def load(cls, snapshot_dir: str | Path) -> "VoxelMapSnapshot":
        source = Path(snapshot_dir)
        if not source.is_dir():
            raise FileNotFoundError(source)
        checksums_path = source / "checksums.json"
        if not checksums_path.is_file():
            raise ValueError("snapshot is missing checksums.json")
        with checksums_path.open("r", encoding="utf-8") as stream:
            checksums = json.load(stream)
        if set(checksums) != set(cls._DATA_FILES):
            raise ValueError("snapshot checksum manifest has unexpected files")
        for name in cls._DATA_FILES:
            data_path = source / name
            if not data_path.is_file() or cls._sha256(data_path) != checksums[name]:
                raise ValueError(f"snapshot checksum mismatch for {name}")

        with (source / "metadata.json").open("r", encoding="utf-8") as stream:
            metadata_payload = json.load(stream)
        try:
            metadata = VoxelSnapshotMetadata(**metadata_payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid snapshot metadata: {exc}") from exc

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
        evidence = SparseEvidenceStore.load(source / "evidence.npz", evidence_config)
        ownership = ReversibleOwnershipStore.load(
            source / "ownership.npz",
            block_resolution=metadata.block_resolution,
        )
        cls._validate_components(metadata, geometry, evidence, ownership)
        return cls(source, metadata, geometry, evidence, ownership, dict(checksums))
