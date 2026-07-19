from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np

from src.oviv2.addressing import BlockKey, VoxelKey, split_voxel_key


@dataclass(frozen=True)
class EvidenceConfig:
    block_resolution: int = 8
    semantic_top_k: int = 4
    entity_top_k: int = 4

    def __post_init__(self) -> None:
        for name in ("block_resolution", "semantic_top_k", "entity_top_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SemanticCandidate:
    label_id: int
    support: float
    revision: int


@dataclass(frozen=True)
class EntityCandidate:
    entity_id: int
    positive_support: float
    negative_support: float
    timestamp: float
    revision: int


@dataclass
class _EvidenceBlock:
    semantic_ids: np.ndarray
    semantic_support: np.ndarray
    semantic_revisions: np.ndarray
    entity_ids: np.ndarray
    entity_positive: np.ndarray
    entity_negative: np.ndarray
    entity_timestamps: np.ndarray
    entity_revisions: np.ndarray


class SparseEvidenceStore:
    _SCHEMA_VERSION = 1

    def __init__(self, config: EvidenceConfig = EvidenceConfig()) -> None:
        if not isinstance(config, EvidenceConfig):
            raise TypeError("config must be an EvidenceConfig")
        self._config = config
        self._blocks: dict[BlockKey, _EvidenceBlock] = {}

    @property
    def config(self) -> EvidenceConfig:
        return self._config

    @property
    def allocated_block_count(self) -> int:
        return len(self._blocks)

    def _new_block(self) -> _EvidenceBlock:
        r = self._config.block_resolution
        semantic_shape = (r, r, r, self._config.semantic_top_k)
        entity_shape = (r, r, r, self._config.entity_top_k)
        return _EvidenceBlock(
            semantic_ids=np.zeros(semantic_shape, dtype=np.int64),
            semantic_support=np.zeros(semantic_shape, dtype=np.float64),
            semantic_revisions=np.zeros(semantic_shape, dtype=np.int64),
            entity_ids=np.zeros(entity_shape, dtype=np.int64),
            entity_positive=np.zeros(entity_shape, dtype=np.float64),
            entity_negative=np.zeros(entity_shape, dtype=np.float64),
            entity_timestamps=np.zeros(entity_shape, dtype=np.float64),
            entity_revisions=np.zeros(entity_shape, dtype=np.int64),
        )

    def _location(
        self,
        voxel_key: VoxelKey,
        *,
        allocate: bool,
    ) -> tuple[_EvidenceBlock | None, tuple[int, int, int]]:
        block_key, local_key = split_voxel_key(voxel_key, self._config.block_resolution)
        block = self._blocks.get(block_key)
        if block is None and allocate:
            block = self._new_block()
            self._blocks[block_key] = block
        return block, local_key

    @staticmethod
    def _validate_revision(revision: int) -> int:
        if not isinstance(revision, (int, np.integer)) or isinstance(revision, bool) or revision < 0:
            raise ValueError("revision must be a non-negative integer")
        return int(revision)

    @staticmethod
    def _validate_id(value: int, name: str) -> int:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def update_semantic(
        self,
        voxel_key: VoxelKey,
        label_id: int,
        support_delta: float,
        revision: int,
    ) -> None:
        normalized_id = self._validate_id(label_id, "label_id")
        delta = float(support_delta)
        normalized_revision = self._validate_revision(revision)
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("support_delta must be finite and positive")
        block, local = self._location(voxel_key, allocate=True)
        assert block is not None
        ids = block.semantic_ids[local]
        supports = block.semantic_support[local]
        revisions = block.semantic_revisions[local]
        if ids.any() and normalized_revision < int(revisions[ids > 0].max()):
            raise ValueError("revision cannot move backwards")

        candidates = {
            int(candidate_id): [float(support), int(candidate_revision)]
            for candidate_id, support, candidate_revision in zip(ids, supports, revisions)
            if candidate_id > 0
        }
        current = candidates.setdefault(normalized_id, [0.0, normalized_revision])
        current[0] += delta
        current[1] = normalized_revision
        ranked = sorted(candidates.items(), key=lambda item: (-item[1][0], item[0]))
        ranked = ranked[: self._config.semantic_top_k]

        ids.fill(0)
        supports.fill(0.0)
        revisions.fill(0)
        for slot, (candidate_id, (support, candidate_revision)) in enumerate(ranked):
            ids[slot] = candidate_id
            supports[slot] = support
            revisions[slot] = candidate_revision

    def update_entity(
        self,
        voxel_key: VoxelKey,
        entity_id: int,
        positive_delta: float,
        negative_delta: float,
        timestamp: float,
        revision: int,
    ) -> None:
        normalized_id = self._validate_id(entity_id, "entity_id")
        positive = float(positive_delta)
        negative = float(negative_delta)
        normalized_timestamp = float(timestamp)
        normalized_revision = self._validate_revision(revision)
        if not np.isfinite(positive) or not np.isfinite(negative):
            raise ValueError("entity support delta values must be finite")
        if positive < 0.0 or negative < 0.0 or (positive == 0.0 and negative == 0.0):
            raise ValueError("at least one entity support delta must be positive")
        if not np.isfinite(normalized_timestamp):
            raise ValueError("timestamp must be finite")

        block, local = self._location(voxel_key, allocate=True)
        assert block is not None
        ids = block.entity_ids[local]
        positives = block.entity_positive[local]
        negatives = block.entity_negative[local]
        timestamps = block.entity_timestamps[local]
        revisions = block.entity_revisions[local]
        if ids.any() and normalized_revision < int(revisions[ids > 0].max()):
            raise ValueError("revision cannot move backwards")

        candidates = {
            int(candidate_id): [
                float(candidate_positive),
                float(candidate_negative),
                float(candidate_timestamp),
                int(candidate_revision),
            ]
            for candidate_id, candidate_positive, candidate_negative, candidate_timestamp, candidate_revision in zip(
                ids,
                positives,
                negatives,
                timestamps,
                revisions,
            )
            if candidate_id > 0
        }
        current = candidates.setdefault(
            normalized_id,
            [0.0, 0.0, normalized_timestamp, normalized_revision],
        )
        current[0] += positive
        current[1] += negative
        current[2] = normalized_timestamp
        current[3] = normalized_revision
        ranked = sorted(
            candidates.items(),
            key=lambda item: (-(item[1][0] - item[1][1]), item[0]),
        )[: self._config.entity_top_k]

        ids.fill(0)
        positives.fill(0.0)
        negatives.fill(0.0)
        timestamps.fill(0.0)
        revisions.fill(0)
        for slot, (candidate_id, values) in enumerate(ranked):
            ids[slot] = candidate_id
            positives[slot] = values[0]
            negatives[slot] = values[1]
            timestamps[slot] = values[2]
            revisions[slot] = values[3]

    def semantic_candidates(self, voxel_key: VoxelKey) -> tuple[SemanticCandidate, ...]:
        block, local = self._location(voxel_key, allocate=False)
        if block is None:
            return ()
        return tuple(
            SemanticCandidate(int(label_id), float(support), int(revision))
            for label_id, support, revision in zip(
                block.semantic_ids[local],
                block.semantic_support[local],
                block.semantic_revisions[local],
            )
            if label_id > 0
        )

    def entity_candidates(self, voxel_key: VoxelKey) -> tuple[EntityCandidate, ...]:
        block, local = self._location(voxel_key, allocate=False)
        if block is None:
            return ()
        return tuple(
            EntityCandidate(
                int(entity_id),
                float(positive),
                float(negative),
                float(timestamp),
                int(revision),
            )
            for entity_id, positive, negative, timestamp, revision in zip(
                block.entity_ids[local],
                block.entity_positive[local],
                block.entity_negative[local],
                block.entity_timestamps[local],
                block.entity_revisions[local],
            )
            if entity_id > 0
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("evidence path must end in .npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._blocks)
        r = self._config.block_resolution
        semantic_shape = (0, r, r, r, self._config.semantic_top_k)
        entity_shape = (0, r, r, r, self._config.entity_top_k)

        def stacked(name: str, empty_shape: tuple[int, ...], dtype) -> np.ndarray:
            if not keys:
                return np.empty(empty_shape, dtype=dtype)
            return np.stack([getattr(self._blocks[key], name) for key in keys])

        payload = {
            "schema_version": np.asarray([self._SCHEMA_VERSION], dtype=np.int64),
            "block_resolution": np.asarray([r], dtype=np.int64),
            "semantic_top_k": np.asarray([self._config.semantic_top_k], dtype=np.int64),
            "entity_top_k": np.asarray([self._config.entity_top_k], dtype=np.int64),
            "block_keys": np.asarray(keys, dtype=np.int64).reshape(-1, 3),
            "semantic_ids": stacked("semantic_ids", semantic_shape, np.int64),
            "semantic_support": stacked("semantic_support", semantic_shape, np.float64),
            "semantic_revisions": stacked("semantic_revisions", semantic_shape, np.int64),
            "entity_ids": stacked("entity_ids", entity_shape, np.int64),
            "entity_positive": stacked("entity_positive", entity_shape, np.float64),
            "entity_negative": stacked("entity_negative", entity_shape, np.float64),
            "entity_timestamps": stacked("entity_timestamps", entity_shape, np.float64),
            "entity_revisions": stacked("entity_revisions", entity_shape, np.int64),
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
    def load(cls, path: str | Path, config: EvidenceConfig) -> "SparseEvidenceStore":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        if not isinstance(config, EvidenceConfig):
            raise TypeError("config must be an EvidenceConfig")
        with np.load(source, allow_pickle=False) as payload:
            scalar_fields = {
                name: int(np.asarray(payload[name]).reshape(-1)[0])
                for name in ("schema_version", "block_resolution", "semantic_top_k", "entity_top_k")
            }
            if scalar_fields["schema_version"] != cls._SCHEMA_VERSION:
                raise ValueError("unsupported evidence schema_version")
            for name in ("block_resolution", "semantic_top_k", "entity_top_k"):
                if scalar_fields[name] != getattr(config, name):
                    raise ValueError(f"stored {name} does not match config")
            keys_array = np.asarray(payload["block_keys"], dtype=np.int64)
            if keys_array.ndim != 2 or keys_array.shape[1] != 3:
                raise ValueError("stored block keys have an invalid shape")
            names = (
                "semantic_ids",
                "semantic_support",
                "semantic_revisions",
                "entity_ids",
                "entity_positive",
                "entity_negative",
                "entity_timestamps",
                "entity_revisions",
            )
            arrays = {name: np.asarray(payload[name]).copy() for name in names}

        keys = [tuple(int(component) for component in row) for row in keys_array]
        if len(set(keys)) != len(keys):
            raise ValueError("stored block keys contain duplicates")
        if any(array.shape[0] != len(keys) for array in arrays.values()):
            raise ValueError("stored evidence arrays do not match block keys")

        store = cls(config)
        for index, key in enumerate(keys):
            store._blocks[key] = _EvidenceBlock(
                semantic_ids=arrays["semantic_ids"][index],
                semantic_support=arrays["semantic_support"][index],
                semantic_revisions=arrays["semantic_revisions"][index],
                entity_ids=arrays["entity_ids"][index],
                entity_positive=arrays["entity_positive"][index],
                entity_negative=arrays["entity_negative"][index],
                entity_timestamps=arrays["entity_timestamps"][index],
                entity_revisions=arrays["entity_revisions"][index],
            )
        return store
