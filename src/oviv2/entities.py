from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from src.oviv2.addressing import VoxelKey
from src.oviv2.tracking import LocalTrack


@dataclass(frozen=True)
class EntityRegistryConfig:
    min_voxel_overlap: float = 0.1
    max_centroid_distance_m: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_voxel_overlap <= 1.0:
            raise ValueError("min_voxel_overlap must lie in [0, 1]")
        if not np.isfinite(self.max_centroid_distance_m) or self.max_centroid_distance_m <= 0.0:
            raise ValueError("max_centroid_distance_m must be finite and positive")


@dataclass(frozen=True)
class PersistentEntity:
    entity_id: int
    semantic_id: int
    label: str
    semantic_support: float
    accepted_view_count: int
    accepted_frame_ids: frozenset[int]
    first_frame_id: int
    last_frame_id: int
    last_revision: int
    lifecycle_state: str
    voxel_keys: frozenset[VoxelKey]
    centroid_xyz: tuple[float, float, float]
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]


class EntityRegistry:
    def __init__(self, config: EntityRegistryConfig = EntityRegistryConfig()) -> None:
        if not isinstance(config, EntityRegistryConfig):
            raise TypeError("config must be EntityRegistryConfig")
        self.config = config
        self.entities: dict[int, PersistentEntity] = {}
        self._next_entity_id = 1

    def resolve(self, track: LocalTrack, revision: int) -> PersistentEntity:
        if not isinstance(track, LocalTrack) or not track.confirmed:
            raise ValueError("only confirmed LocalTrack values can resolve entities")
        normalized_revision = int(revision)
        if normalized_revision < 0:
            raise ValueError("revision must be non-negative")
        entity_id = self._match(track)
        if entity_id is None:
            entity_id = self._next_entity_id
            self._next_entity_id += 1
            accepted_frames = frozenset(item.frame_id for item in track.observations)
            entity = PersistentEntity(
                entity_id=entity_id,
                semantic_id=track.semantic_id,
                label=track.label,
                semantic_support=track.semantic_confidence,
                accepted_view_count=len(accepted_frames),
                accepted_frame_ids=accepted_frames,
                first_frame_id=min(accepted_frames),
                last_frame_id=track.last_frame_id,
                last_revision=normalized_revision,
                lifecycle_state="active",
                voxel_keys=track.voxel_keys,
                centroid_xyz=track.centroid_xyz,
                bounds_min_xyz=track.bounds_min_xyz,
                bounds_max_xyz=track.bounds_max_xyz,
            )
        else:
            previous = self.entities[entity_id]
            previous_weight = previous.semantic_support
            current_weight = track.semantic_confidence
            if track.semantic_id == previous.semantic_id or current_weight <= previous_weight:
                semantic_id, label = previous.semantic_id, previous.label
            else:
                semantic_id, label = track.semantic_id, track.label
            track_frames = frozenset(item.frame_id for item in track.observations)
            accepted_frames = previous.accepted_frame_ids | track_frames
            new_frame_count = len(accepted_frames - previous.accepted_frame_ids)
            current_centroid = np.asarray(track.observations[-1].centroid_xyz)
            if new_frame_count:
                centroid = (
                    np.asarray(previous.centroid_xyz) * previous.accepted_view_count
                    + current_centroid * new_frame_count
                ) / len(accepted_frames)
            else:
                centroid = (np.asarray(previous.centroid_xyz) + current_centroid) / 2.0
            entity = replace(
                previous,
                semantic_id=semantic_id,
                label=label,
                semantic_support=previous_weight + current_weight,
                accepted_view_count=len(accepted_frames),
                accepted_frame_ids=accepted_frames,
                first_frame_id=min(accepted_frames),
                last_frame_id=track.last_frame_id,
                last_revision=normalized_revision,
                lifecycle_state="active",
                voxel_keys=previous.voxel_keys | track.voxel_keys,
                centroid_xyz=tuple(float(value) for value in centroid),
                bounds_min_xyz=tuple(
                    float(value)
                    for value in np.minimum(previous.bounds_min_xyz, track.bounds_min_xyz)
                ),
                bounds_max_xyz=tuple(
                    float(value)
                    for value in np.maximum(previous.bounds_max_xyz, track.bounds_max_xyz)
                ),
            )
        self.entities[entity_id] = entity
        return entity

    def _match(self, track: LocalTrack) -> int | None:
        ranked: list[tuple[float, float, int]] = []
        for entity_id, entity in self.entities.items():
            denominator = min(len(track.voxel_keys), len(entity.voxel_keys))
            overlap = len(track.voxel_keys & entity.voxel_keys) / denominator if denominator else 0.0
            distance = float(np.linalg.norm(np.asarray(track.centroid_xyz) - entity.centroid_xyz))
            if overlap < self.config.min_voxel_overlap and distance > self.config.max_centroid_distance_m:
                continue
            distance_score = max(0.0, 1.0 - distance / self.config.max_centroid_distance_m)
            geometry_score = float(overlap + distance_score)
            semantic_score = float(track.semantic_id > 0 and track.semantic_id == entity.semantic_id)
            ranked.append((geometry_score, semantic_score, entity_id))
        if not ranked:
            return None
        return sorted(ranked, key=lambda item: (-item[0], -item[1], item[2]))[0][2]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.suffix != ".jsonl":
            raise ValueError("entity registry path must end in .jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.stem}.",
                suffix=".jsonl",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                for entity_id in sorted(self.entities):
                    payload = asdict(self.entities[entity_id])
                    payload["voxel_keys"] = [list(key) for key in sorted(self.entities[entity_id].voxel_keys)]
                    payload["accepted_frame_ids"] = sorted(
                        self.entities[entity_id].accepted_frame_ids
                    )
                    stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: EntityRegistryConfig = EntityRegistryConfig(),
    ) -> "EntityRegistry":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        registry = cls(config)
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                payload["voxel_keys"] = frozenset(
                    tuple(int(value) for value in key) for key in payload["voxel_keys"]
                )
                payload["accepted_frame_ids"] = frozenset(
                    int(value)
                    for value in payload.get("accepted_frame_ids", [payload["last_frame_id"]])
                )
                payload["accepted_view_count"] = len(payload["accepted_frame_ids"])
                for name in ("centroid_xyz", "bounds_min_xyz", "bounds_max_xyz"):
                    payload[name] = tuple(float(value) for value in payload[name])
                entity = PersistentEntity(**payload)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid entity registry line {line_number}: {exc}") from exc
            if entity.entity_id <= 0 or entity.entity_id in registry.entities:
                raise ValueError("entity IDs must be unique and positive")
            registry.entities[entity.entity_id] = entity
        registry._next_entity_id = max(registry.entities, default=0) + 1
        return registry
