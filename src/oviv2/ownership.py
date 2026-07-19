from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np

from src.oviv2.addressing import BlockKey, VoxelKey, join_voxel_key, split_voxel_key


@dataclass(frozen=True)
class OwnershipRecord:
    entity_id: int
    confidence: float
    epoch: int
    evidence_revision: int


@dataclass
class _OwnershipBlock:
    entity_ids: np.ndarray
    confidence: np.ndarray
    epochs: np.ndarray
    evidence_revisions: np.ndarray


class ReversibleOwnershipStore:
    _SCHEMA_VERSION = 1

    def __init__(self, block_resolution: int = 8) -> None:
        if (
            not isinstance(block_resolution, int)
            or isinstance(block_resolution, bool)
            or block_resolution <= 0
        ):
            raise ValueError("block_resolution must be a positive integer")
        self._block_resolution = block_resolution
        self._blocks: dict[BlockKey, _OwnershipBlock] = {}
        self._entity_voxels: dict[int, set[VoxelKey]] = {}

    @property
    def block_resolution(self) -> int:
        return self._block_resolution

    @property
    def allocated_block_count(self) -> int:
        return len(self._blocks)

    def _new_block(self) -> _OwnershipBlock:
        shape = (self._block_resolution,) * 3
        return _OwnershipBlock(
            entity_ids=np.zeros(shape, dtype=np.int64),
            confidence=np.zeros(shape, dtype=np.float64),
            epochs=np.zeros(shape, dtype=np.int64),
            evidence_revisions=np.zeros(shape, dtype=np.int64),
        )

    def _location(
        self,
        voxel_key: VoxelKey,
        *,
        allocate: bool,
    ) -> tuple[_OwnershipBlock | None, tuple[int, int, int], VoxelKey]:
        block_key, local_key = split_voxel_key(voxel_key, self._block_resolution)
        normalized_key = join_voxel_key(block_key, local_key, self._block_resolution)
        block = self._blocks.get(block_key)
        if block is None and allocate:
            block = self._new_block()
            self._blocks[block_key] = block
        return block, local_key, normalized_key

    @staticmethod
    def _entity_id(value: int) -> int:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
            raise ValueError("entity_id must be a positive integer")
        return int(value)

    @staticmethod
    def _revision(value: int) -> int:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
            raise ValueError("evidence_revision must be a non-negative integer")
        return int(value)

    def assign(
        self,
        voxel_key: VoxelKey,
        entity_id: int,
        confidence: float,
        evidence_revision: int,
    ) -> None:
        normalized_entity = self._entity_id(entity_id)
        normalized_confidence = float(confidence)
        revision = self._revision(evidence_revision)
        if (
            not np.isfinite(normalized_confidence)
            or normalized_confidence < 0.0
            or normalized_confidence > 1.0
        ):
            raise ValueError("confidence must be finite and lie in [0, 1]")

        block, local, normalized_key = self._location(voxel_key, allocate=True)
        assert block is not None
        if revision < int(block.evidence_revisions[local]):
            raise ValueError("evidence revision cannot move backwards")
        previous_entity = int(block.entity_ids[local])
        if previous_entity > 0:
            previous_voxels = self._entity_voxels.get(previous_entity)
            if previous_voxels is not None:
                previous_voxels.discard(normalized_key)
                if not previous_voxels:
                    self._entity_voxels.pop(previous_entity, None)

        block.entity_ids[local] = normalized_entity
        block.confidence[local] = normalized_confidence
        block.epochs[local] += 1
        block.evidence_revisions[local] = revision
        self._entity_voxels.setdefault(normalized_entity, set()).add(normalized_key)

    def release(
        self,
        voxel_key: VoxelKey,
        expected_entity_id: int,
        evidence_revision: int,
    ) -> bool:
        expected = self._entity_id(expected_entity_id)
        revision = self._revision(evidence_revision)
        block, local, normalized_key = self._location(voxel_key, allocate=False)
        if block is None or int(block.entity_ids[local]) != expected:
            return False
        if revision < int(block.evidence_revisions[local]):
            raise ValueError("evidence revision cannot move backwards")

        block.entity_ids[local] = 0
        block.confidence[local] = 0.0
        block.epochs[local] += 1
        block.evidence_revisions[local] = revision
        entity_voxels = self._entity_voxels.get(expected)
        if entity_voxels is not None:
            entity_voxels.discard(normalized_key)
            if not entity_voxels:
                self._entity_voxels.pop(expected, None)
        return True

    def owner_of(self, voxel_key: VoxelKey) -> OwnershipRecord | None:
        block, local, _ = self._location(voxel_key, allocate=False)
        if block is None:
            return None
        entity_id = int(block.entity_ids[local])
        if entity_id == 0:
            return None
        return OwnershipRecord(
            entity_id=entity_id,
            confidence=float(block.confidence[local]),
            epoch=int(block.epochs[local]),
            evidence_revision=int(block.evidence_revisions[local]),
        )

    def voxels_for_entity(self, entity_id: int) -> frozenset[VoxelKey]:
        normalized_entity = self._entity_id(entity_id)
        return frozenset(self._entity_voxels.get(normalized_entity, ()))

    def records(self) -> tuple[tuple[VoxelKey, OwnershipRecord], ...]:
        records: list[tuple[VoxelKey, OwnershipRecord]] = []
        for block_key in sorted(self._blocks):
            block = self._blocks[block_key]
            for local_array in np.argwhere(block.entity_ids > 0):
                local = tuple(int(value) for value in local_array)
                voxel_key = join_voxel_key(block_key, local, self._block_resolution)
                record = self.owner_of(voxel_key)
                assert record is not None
                records.append((voxel_key, record))
        return tuple(records)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("ownership path must end in .npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._blocks)
        empty_shape = (0,) + (self._block_resolution,) * 3

        def stacked(name: str, dtype) -> np.ndarray:
            if not keys:
                return np.empty(empty_shape, dtype=dtype)
            return np.stack([getattr(self._blocks[key], name) for key in keys])

        payload = {
            "schema_version": np.asarray([self._SCHEMA_VERSION], dtype=np.int64),
            "block_resolution": np.asarray([self._block_resolution], dtype=np.int64),
            "block_keys": np.asarray(keys, dtype=np.int64).reshape(-1, 3),
            "entity_ids": stacked("entity_ids", np.int64),
            "confidence": stacked("confidence", np.float64),
            "epochs": stacked("epochs", np.int64),
            "evidence_revisions": stacked("evidence_revisions", np.int64),
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.stem}.",
                suffix=".npz",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            np.savez_compressed(temporary_path, **payload)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(cls, path: str | Path, block_resolution: int) -> "ReversibleOwnershipStore":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        store = cls(block_resolution=block_resolution)
        with np.load(source, allow_pickle=False) as payload:
            schema_version = int(np.asarray(payload["schema_version"]).reshape(-1)[0])
            stored_resolution = int(np.asarray(payload["block_resolution"]).reshape(-1)[0])
            if schema_version != cls._SCHEMA_VERSION:
                raise ValueError("unsupported ownership schema_version")
            if stored_resolution != block_resolution:
                raise ValueError("stored block_resolution does not match config")
            keys_array = np.asarray(payload["block_keys"], dtype=np.int64)
            arrays = {
                name: np.asarray(payload[name]).copy()
                for name in ("entity_ids", "confidence", "epochs", "evidence_revisions")
            }
        if keys_array.ndim != 2 or keys_array.shape[1] != 3:
            raise ValueError("stored ownership block keys have an invalid shape")
        keys = [tuple(int(component) for component in row) for row in keys_array]
        if len(set(keys)) != len(keys):
            raise ValueError("stored ownership block keys contain duplicates")
        if any(array.shape[0] != len(keys) for array in arrays.values()):
            raise ValueError("stored ownership arrays do not match block keys")

        for index, block_key in enumerate(keys):
            block = _OwnershipBlock(
                entity_ids=arrays["entity_ids"][index],
                confidence=arrays["confidence"][index],
                epochs=arrays["epochs"][index],
                evidence_revisions=arrays["evidence_revisions"][index],
            )
            store._blocks[block_key] = block
            for local_array in np.argwhere(block.entity_ids > 0):
                local = tuple(int(value) for value in local_array)
                entity_id = int(block.entity_ids[local])
                voxel_key = join_voxel_key(block_key, local, block_resolution)
                store._entity_voxels.setdefault(entity_id, set()).add(voxel_key)
        return store
