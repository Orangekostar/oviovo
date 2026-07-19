from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from itertools import chain
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, TextIO

import numpy as np

from src.oviv2.addressing import VoxelKey
from src.oviv2.association import AssociationConfig, AssociationTarget, solve_assignment
from src.oviv2.semantic_memory import (
    FeaturePrototype,
    FeaturePrototypeBank,
    InformativeView,
    InformativeViewBank,
    SparseClassPosterior,
)
from src.oviv2.tracking import LocalTrack


_LEGACY_MIN_VOXEL_OVERLAP = 0.1
_LEGACY_MAX_CENTROID_DISTANCE_M = 0.6


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


def _number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _bounded_number(value: object, name: str, lower: float, upper: float) -> float:
    normalized = _number(value, name)
    if not lower <= normalized <= upper:
        raise ValueError(f"{name} must lie in [{lower:g}, {upper:g}]")
    return normalized


def _point3(value: object, name: str) -> tuple[float, float, float]:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain three numeric coordinates") from exc
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain three finite coordinates")
    return tuple(float(item) for item in point)  # type: ignore[return-value]


def _voxel_keys(value: object) -> frozenset[VoxelKey]:
    if not isinstance(value, frozenset):
        raise TypeError("voxel_keys must be a frozenset")
    normalized: set[VoxelKey] = set()
    for key in value:
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError("voxel_keys must contain integer 3-tuples")
        if any(
            isinstance(component, (bool, np.bool_))
            or not isinstance(component, Integral)
            for component in key
        ):
            raise TypeError("voxel_keys must contain integer 3-tuples")
        normalized.add(tuple(int(component) for component in key))  # type: ignore[arg-type]
    return frozenset(normalized)


@dataclass(frozen=True)
class EntityRegistryConfig:
    min_voxel_overlap: float = _LEGACY_MIN_VOXEL_OVERLAP
    max_centroid_distance_m: float = _LEGACY_MAX_CENTROID_DISTANCE_M
    association: AssociationConfig | None = None
    prototype_top_k: int = 3
    prototype_merge_cosine: float = 0.90
    view_top_k: int = 10
    view_minimum_novelty_cosine: float = 0.10

    def __post_init__(self) -> None:
        min_overlap = _bounded_number(
            self.min_voxel_overlap,
            "min_voxel_overlap",
            0.0,
            1.0,
        )
        max_distance = _number(
            self.max_centroid_distance_m,
            "max_centroid_distance_m",
        )
        if max_distance <= 0.0:
            raise ValueError("max_centroid_distance_m must be positive")
        prototype_top_k = _integer(self.prototype_top_k, "prototype_top_k", minimum=1)
        prototype_merge_cosine = _bounded_number(
            self.prototype_merge_cosine,
            "prototype_merge_cosine",
            -1.0,
            1.0,
        )
        view_top_k = _integer(self.view_top_k, "view_top_k", minimum=1)
        view_minimum_novelty_cosine = _bounded_number(
            self.view_minimum_novelty_cosine,
            "view_minimum_novelty_cosine",
            0.0,
            2.0,
        )
        association = self.association
        if association is None:
            association = AssociationConfig(
                min_directed_overlap=min_overlap,
                max_centroid_distance_m=max_distance,
            )
        else:
            if not isinstance(association, AssociationConfig):
                raise TypeError("association must be an AssociationConfig")
            if (
                min_overlap != _LEGACY_MIN_VOXEL_OVERLAP
                or max_distance != _LEGACY_MAX_CENTROID_DISTANCE_M
            ):
                raise ValueError(
                    "explicit association cannot be combined with non-default legacy "
                    "association thresholds"
                )
        object.__setattr__(self, "min_voxel_overlap", min_overlap)
        object.__setattr__(self, "max_centroid_distance_m", max_distance)
        object.__setattr__(self, "association", association)
        object.__setattr__(self, "prototype_top_k", prototype_top_k)
        object.__setattr__(self, "prototype_merge_cosine", prototype_merge_cosine)
        object.__setattr__(self, "view_top_k", view_top_k)
        object.__setattr__(
            self,
            "view_minimum_novelty_cosine",
            view_minimum_novelty_cosine,
        )


@dataclass(frozen=True)
class PersistentEntity:
    entity_id: int
    semantic_id: int
    label: str
    semantic_posterior: SparseClassPosterior
    feature_bank: FeaturePrototypeBank
    view_bank: InformativeViewBank
    accepted_observation_ids: frozenset[int]
    semantic_entropy: float
    semantic_margin: float
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

    def __post_init__(self) -> None:
        entity_id = _integer(self.entity_id, "entity_id", minimum=1)
        semantic_id = _integer(self.semantic_id, "semantic_id", minimum=0)
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")
        if semantic_id > 0 and not self.label.strip():
            raise ValueError("label must be non-empty for a positive semantic winner")
        if semantic_id == 0 and self.label != "":
            raise ValueError("label must be empty without a semantic winner")
        if not isinstance(self.semantic_posterior, SparseClassPosterior):
            raise TypeError("semantic_posterior must be a SparseClassPosterior")
        if not isinstance(self.feature_bank, FeaturePrototypeBank):
            raise TypeError("feature_bank must be a FeaturePrototypeBank")
        if not isinstance(self.view_bank, InformativeViewBank):
            raise TypeError("view_bank must be an InformativeViewBank")
        if semantic_id != self.semantic_posterior.best_semantic_id:
            raise ValueError("semantic_id must equal the posterior winner")

        entropy = _number(self.semantic_entropy, "semantic_entropy")
        margin = _number(self.semantic_margin, "semantic_margin")
        if not math.isclose(entropy, self.semantic_posterior.entropy, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("semantic_entropy must equal the posterior entropy")
        if not math.isclose(margin, self.semantic_posterior.margin, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("semantic_margin must equal the posterior margin")

        observation_ids = self._id_set(
            self.accepted_observation_ids,
            "accepted_observation_ids",
        )
        frame_ids = self._id_set(self.accepted_frame_ids, "accepted_frame_ids")
        if any(
            view.observation_id not in observation_ids
            or view.frame_id not in frame_ids
            for view in self.view_bank.views
        ):
            raise ValueError("view_bank IDs must belong to accepted IDs")
        accepted_view_count = _integer(
            self.accepted_view_count,
            "accepted_view_count",
            minimum=0,
        )
        if accepted_view_count != len(frame_ids):
            raise ValueError("accepted_view_count must equal distinct accepted frame count")
        first_frame_id = _integer(self.first_frame_id, "first_frame_id", minimum=0)
        last_frame_id = _integer(self.last_frame_id, "last_frame_id", minimum=0)
        if first_frame_id > last_frame_id:
            raise ValueError("first_frame_id must not exceed last_frame_id")
        if frame_ids and (first_frame_id != min(frame_ids) or last_frame_id < max(frame_ids)):
            raise ValueError("first/last frame IDs must cover accepted_frame_ids")
        last_revision = _integer(self.last_revision, "last_revision", minimum=0)
        if self.lifecycle_state not in {"active", "dormant"}:
            raise ValueError("lifecycle_state must be active or dormant")

        voxel_keys = _voxel_keys(self.voxel_keys)
        centroid = _point3(self.centroid_xyz, "centroid_xyz")
        bounds_min = _point3(self.bounds_min_xyz, "bounds_min_xyz")
        bounds_max = _point3(self.bounds_max_xyz, "bounds_max_xyz")
        if any(lower > upper for lower, upper in zip(bounds_min, bounds_max)):
            raise ValueError("bounds_min_xyz must not exceed bounds_max_xyz")

        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "semantic_id", semantic_id)
        object.__setattr__(self, "accepted_observation_ids", observation_ids)
        object.__setattr__(self, "semantic_entropy", entropy)
        object.__setattr__(self, "semantic_margin", margin)
        object.__setattr__(self, "accepted_view_count", accepted_view_count)
        object.__setattr__(self, "accepted_frame_ids", frame_ids)
        object.__setattr__(self, "first_frame_id", first_frame_id)
        object.__setattr__(self, "last_frame_id", last_frame_id)
        object.__setattr__(self, "last_revision", last_revision)
        object.__setattr__(self, "voxel_keys", voxel_keys)
        object.__setattr__(self, "centroid_xyz", centroid)
        object.__setattr__(self, "bounds_min_xyz", bounds_min)
        object.__setattr__(self, "bounds_max_xyz", bounds_max)

    @staticmethod
    def _id_set(value: object, name: str) -> frozenset[int]:
        if not isinstance(value, frozenset):
            raise TypeError(f"{name} must be a frozenset")
        return frozenset(_integer(item, name, minimum=0) for item in value)

    @property
    def semantic_support(self) -> float:
        return self.semantic_posterior.effective_support


def _observation_quality(observation: Any) -> float:
    quality = (
        float(observation.confidence)
        * (1.0 - float(observation.border_contact_fraction))
        * min(1.0, float(observation.visible_pixel_count) / 4096.0)
        * min(1.0, len(observation.voxel_keys) / 256.0)
    )
    return float(np.clip(quality, 0.0, 1.0))


class EntityRegistry:
    def __init__(self, config: EntityRegistryConfig = EntityRegistryConfig()) -> None:
        if not isinstance(config, EntityRegistryConfig):
            raise TypeError("config must be EntityRegistryConfig")
        self.config = config
        self.entities: dict[int, PersistentEntity] = {}
        self._next_entity_id = 1

    def resolve(self, track: LocalTrack, revision: int) -> PersistentEntity:
        return self.resolve_batch((track,), revision)[0]

    def resolve_batch(
        self,
        tracks: tuple[LocalTrack, ...],
        revision: int,
    ) -> tuple[PersistentEntity, ...]:
        normalized_revision = _integer(revision, "revision", minimum=0)
        minimum_revision = max(
            (entity.last_revision for entity in self.entities.values()),
            default=0,
        )
        if normalized_revision < minimum_revision:
            raise ValueError(
                f"revision must be at least the current registry revision {minimum_revision}"
            )
        if not isinstance(tracks, tuple):
            raise TypeError("tracks must be a tuple")
        if any(not isinstance(track, LocalTrack) for track in tracks):
            raise TypeError("tracks must contain only LocalTrack values")
        if any(track.confirmed is not True for track in tracks):
            raise ValueError("only confirmed LocalTrack values can resolve entities")
        track_ids = [track.track_id for track in tracks]
        if any(
            isinstance(track_id, (bool, np.bool_))
            or not isinstance(track_id, Integral)
            or int(track_id) < 0
            for track_id in track_ids
        ):
            raise ValueError("track IDs must be non-negative integers")
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("track IDs must be unique")
        observation_ids = [
            observation.observation_id
            for track in tracks
            for observation in track.observations
        ]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("observation IDs must be unique within a resolve batch")

        sorted_tracks = tuple(sorted(tracks, key=lambda track: track.track_id))
        track_targets = tuple(self._track_target(track) for track in sorted_tracks)
        entity_targets = tuple(
            self._entity_target(entity)
            for entity in sorted(self.entities.values(), key=lambda value: value.entity_id)
            if entity.lifecycle_state in {"active", "dormant"}
        )
        assignments = solve_assignment(
            track_targets,
            entity_targets,
            self.config.association,
        )
        matched_entity_ids = {
            assignment.left_id: assignment.right_id for assignment in assignments
        }

        updated_entities = dict(self.entities)
        next_entity_id = self._next_entity_id
        resolved: list[PersistentEntity] = []
        for track in sorted_tracks:
            entity_id = matched_entity_ids.get(track.track_id)
            if entity_id is None:
                entity = self._create_entity(track, next_entity_id, normalized_revision)
                next_entity_id += 1
            else:
                entity = self._update_entity(
                    updated_entities[entity_id],
                    track,
                    normalized_revision,
                )
            updated_entities[entity.entity_id] = entity
            resolved.append(entity)

        self.entities = updated_entities
        self._next_entity_id = next_entity_id
        return tuple(resolved)

    @staticmethod
    def _track_target(track: LocalTrack) -> AssociationTarget:
        return AssociationTarget(
            target_id=track.track_id,
            voxel_keys=track.voxel_keys,
            centroid_xyz=track.centroid_xyz,
            bounds_min_xyz=track.bounds_min_xyz,
            bounds_max_xyz=track.bounds_max_xyz,
            semantic_id=track.semantic_id,
            semantic_confidence=track.semantic_confidence,
            visual_feature=track.visual_feature,
            feature_model_id=track.feature_model_id,
        )

    @staticmethod
    def _entity_target(entity: PersistentEntity) -> AssociationTarget:
        probabilities = dict(entity.semantic_posterior.probabilities)
        prototype = entity.feature_bank.prototypes[0] if entity.feature_bank.prototypes else None
        return AssociationTarget(
            target_id=entity.entity_id,
            voxel_keys=entity.voxel_keys,
            centroid_xyz=entity.centroid_xyz,
            bounds_min_xyz=entity.bounds_min_xyz,
            bounds_max_xyz=entity.bounds_max_xyz,
            semantic_id=entity.semantic_id,
            semantic_confidence=probabilities.get(entity.semantic_id, 0.0),
            visual_feature=None if prototype is None else prototype.vector,
            feature_model_id=None if prototype is None else prototype.model_id,
        )

    def _empty_memory(
        self,
    ) -> tuple[SparseClassPosterior, FeaturePrototypeBank, InformativeViewBank]:
        return (
            SparseClassPosterior.empty(),
            FeaturePrototypeBank(
                max_prototypes=self.config.prototype_top_k,
                merge_cosine=self.config.prototype_merge_cosine,
            ),
            InformativeViewBank(
                max_views=self.config.view_top_k,
                minimum_novelty_cosine=self.config.view_minimum_novelty_cosine,
            ),
        )

    def _create_entity(
        self,
        track: LocalTrack,
        entity_id: int,
        revision: int,
    ) -> PersistentEntity:
        posterior, feature_bank, view_bank = self._empty_memory()
        observations = tuple(sorted(track.observations, key=lambda item: (item.frame_id, item.observation_id)))
        if not observations:
            raise ValueError("confirmed tracks must contain observations")
        first = observations[0]
        seed = PersistentEntity(
            entity_id=entity_id,
            semantic_id=0,
            label="",
            semantic_posterior=posterior,
            feature_bank=feature_bank,
            view_bank=view_bank,
            accepted_observation_ids=frozenset(),
            semantic_entropy=0.0,
            semantic_margin=0.0,
            accepted_view_count=0,
            accepted_frame_ids=frozenset(),
            first_frame_id=first.frame_id,
            last_frame_id=first.frame_id,
            last_revision=revision,
            lifecycle_state="active",
            voxel_keys=frozenset(),
            centroid_xyz=first.centroid_xyz,
            bounds_min_xyz=first.bounds_min_xyz,
            bounds_max_xyz=first.bounds_max_xyz,
        )
        return self._update_entity(seed, track, revision)

    def _update_entity(
        self,
        previous: PersistentEntity,
        track: LocalTrack,
        revision: int,
    ) -> PersistentEntity:
        observations = tuple(sorted(track.observations, key=lambda item: (item.frame_id, item.observation_id)))
        seen = set(previous.accepted_observation_ids)
        unseen = []
        for observation in observations:
            if observation.observation_id not in seen:
                unseen.append(observation)
                seen.add(observation.observation_id)

        posterior = previous.semantic_posterior
        feature_bank = previous.feature_bank
        view_bank = previous.view_bank
        qualities: dict[int, float] = {}
        for observation in unseen:
            quality = _observation_quality(observation)
            qualities[observation.observation_id] = quality
            if observation.semantic_id > 0:
                posterior = posterior.update_label(
                    observation.semantic_id,
                    observation.confidence,
                    quality,
                )
            if observation.image_feature is not None and quality > 0.0:
                feature_bank = feature_bank.update(
                    observation.image_feature,
                    observation.feature_model_id,
                    quality,
                )
            if observation.view_direction_xyz is not None:
                view_bank = view_bank.update(
                    InformativeView(
                        observation.observation_id,
                        observation.frame_id,
                        observation.visible_pixel_count,
                        quality,
                        observation.view_direction_xyz,
                    )
                )

        winner = posterior.best_semantic_id
        label = previous.label
        winner_observations = [
            observation
            for observation in unseen
            if winner > 0
            and observation.semantic_id == winner
            and qualities[observation.observation_id] > 0.0
        ]
        if winner_observations:
            label = min(
                winner_observations,
                key=lambda item: (
                    -qualities[item.observation_id],
                    item.frame_id,
                    item.observation_id,
                ),
            ).label
        elif winner != previous.semantic_id:
            raise ValueError("posterior winner change requires new evidence for the winner")

        accepted_frame_ids = previous.accepted_frame_ids | frozenset(
            observation.frame_id for observation in unseen
        )
        accepted_observation_ids = frozenset(seen)
        voxel_keys = previous.voxel_keys
        bounds_min = np.asarray(previous.bounds_min_xyz, dtype=np.float64)
        bounds_max = np.asarray(previous.bounds_max_xyz, dtype=np.float64)
        centroid = np.asarray(previous.centroid_xyz, dtype=np.float64)
        observation_count = max(
            len(previous.accepted_observation_ids),
            previous.accepted_view_count,
        )
        for observation in unseen:
            voxel_keys |= observation.voxel_keys
            bounds_min = np.minimum(bounds_min, observation.bounds_min_xyz)
            bounds_max = np.maximum(bounds_max, observation.bounds_max_xyz)
            centroid = (
                centroid * observation_count + np.asarray(observation.centroid_xyz, dtype=np.float64)
            ) / (observation_count + 1)
            observation_count += 1

        first_frame_id = min(accepted_frame_ids) if accepted_frame_ids else previous.first_frame_id
        last_frame_id = max(
            previous.last_frame_id,
            max((observation.frame_id for observation in unseen), default=previous.last_frame_id),
        )
        return replace(
            previous,
            semantic_id=winner,
            label=label,
            semantic_posterior=posterior,
            feature_bank=feature_bank,
            view_bank=view_bank,
            accepted_observation_ids=accepted_observation_ids,
            semantic_entropy=posterior.entropy,
            semantic_margin=posterior.margin,
            accepted_view_count=len(accepted_frame_ids),
            accepted_frame_ids=accepted_frame_ids,
            first_frame_id=first_frame_id,
            last_frame_id=last_frame_id,
            last_revision=revision,
            lifecycle_state="active",
            voxel_keys=voxel_keys,
            centroid_xyz=tuple(float(value) for value in centroid),
            bounds_min_xyz=tuple(float(value) for value in bounds_min),
            bounds_max_xyz=tuple(float(value) for value in bounds_max),
        )

    def semantic_labels(self) -> dict[int, tuple[int, float]]:
        return {
            entity.entity_id: (
                entity.semantic_id,
                dict(entity.semantic_posterior.probabilities)[entity.semantic_id],
            )
            for entity in sorted(self.entities.values(), key=lambda value: value.entity_id)
            if entity.lifecycle_state in {"active", "dormant"} and entity.semantic_id > 0
        }

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
                metadata = {
                    "record_type": "registry",
                    "schema_version": 2,
                    "config": asdict(self.config),
                    "next_entity_id": self._next_entity_id,
                }
                stream.write(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n")
                for entity_id in sorted(self.entities):
                    payload = asdict(self.entities[entity_id])
                    payload["record_type"] = "entity"
                    payload["voxel_keys"] = [list(key) for key in sorted(self.entities[entity_id].voxel_keys)]
                    payload["accepted_frame_ids"] = sorted(self.entities[entity_id].accepted_frame_ids)
                    payload["accepted_observation_ids"] = sorted(
                        self.entities[entity_id].accepted_observation_ids
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
        config: EntityRegistryConfig | None = None,
    ) -> "EntityRegistry":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("r", encoding="utf-8") as stream:
            records = cls._stream_records(stream)
            first = next(records, None)
            if first is None:
                return cls._load_v1((), config)
            if first[1].get("record_type") == "registry":
                return cls._load_v2(first, records, config)
            return cls._load_v1(chain((first,), records), config)

    @staticmethod
    def _stream_records(stream: TextIO) -> Iterator[tuple[int, dict[str, Any]]]:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("records must be JSON objects")
                yield line_number, payload
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid entity registry line {line_number}: {exc}") from exc

    @classmethod
    def _load_v1(
        cls,
        records: Iterable[tuple[int, dict[str, Any]]],
        config: EntityRegistryConfig | None,
    ) -> "EntityRegistry":
        registry = cls(EntityRegistryConfig() if config is None else config)
        for line_number, payload in records:
            try:
                if "record_type" in payload:
                    raise ValueError("v1 records cannot contain record_type")
                entity = registry._load_v1_entity(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid entity registry line {line_number}: {exc}") from exc
            if entity.entity_id in registry.entities:
                raise ValueError("entity IDs must be unique and positive")
            registry.entities[entity.entity_id] = entity
        registry._next_entity_id = max(registry.entities, default=0) + 1
        return registry

    @classmethod
    def _load_v2(
        cls,
        metadata_record: tuple[int, dict[str, Any]],
        records: Iterable[tuple[int, dict[str, Any]]],
        explicit_config: EntityRegistryConfig | None,
    ) -> "EntityRegistry":
        metadata_line, metadata = metadata_record
        expected_metadata_keys = {
            "record_type",
            "schema_version",
            "config",
            "next_entity_id",
        }
        try:
            if set(metadata) != expected_metadata_keys:
                raise ValueError("registry metadata fields are invalid")
            schema_version = _integer(
                metadata["schema_version"],
                "schema_version",
                minimum=0,
            )
            if metadata["record_type"] != "registry" or schema_version != 2:
                raise ValueError("registry metadata must use schema_version 2")
            stored_config = cls._config_from_payload(metadata["config"])
            next_entity_id = _integer(
                metadata["next_entity_id"],
                "next_entity_id",
                minimum=1,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid entity registry line {metadata_line}: {exc}") from exc
        if explicit_config is not None and explicit_config != stored_config:
            raise ValueError("explicit config does not match v2 registry metadata")

        registry = cls(stored_config)
        for line_number, payload in records:
            try:
                if payload.get("record_type") != "entity":
                    raise ValueError("v2 records after metadata must have record_type entity")
                entity = registry._load_v2_entity(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid entity registry line {line_number}: {exc}") from exc
            if entity.entity_id in registry.entities:
                raise ValueError("entity IDs must be unique and positive")
            registry.entities[entity.entity_id] = entity
        if registry.entities and next_entity_id <= max(registry.entities):
            raise ValueError("next_entity_id must exceed all entity IDs")
        registry._next_entity_id = next_entity_id
        return registry

    @staticmethod
    def _config_from_payload(payload: Any) -> EntityRegistryConfig:
        if not isinstance(payload, dict):
            raise TypeError("config must be an object")
        expected = {field.name for field in fields(EntityRegistryConfig)}
        if set(payload) != expected:
            raise ValueError("config fields are invalid")
        association_payload = payload["association"]
        if not isinstance(association_payload, dict):
            raise TypeError("config association must be an object")
        association_fields = {field.name for field in fields(AssociationConfig)}
        if set(association_payload) != association_fields:
            raise ValueError("association config fields are invalid")
        association = AssociationConfig(**association_payload)
        values = dict(payload)
        min_overlap = values["min_voxel_overlap"]
        max_distance = values["max_centroid_distance_m"]
        if (
            min_overlap != _LEGACY_MIN_VOXEL_OVERLAP
            or max_distance != _LEGACY_MAX_CENTROID_DISTANCE_M
        ):
            expected_association = AssociationConfig(
                min_directed_overlap=min_overlap,
                max_centroid_distance_m=max_distance,
            )
            if association != expected_association:
                raise ValueError("canonical association conflicts with legacy thresholds")
            values["association"] = None
        else:
            values["association"] = association
        return EntityRegistryConfig(**values)

    def _load_v2_entity(self, payload: dict[str, Any]) -> PersistentEntity:
        expected = {field.name for field in fields(PersistentEntity)} | {"record_type"}
        if set(payload) != expected:
            raise ValueError("entity record fields are invalid")

        posterior_payload = payload["semantic_posterior"]
        if not isinstance(posterior_payload, dict) or set(posterior_payload) != {
            "log_evidence",
            "effective_support",
        }:
            raise ValueError("semantic_posterior fields are invalid")
        posterior = SparseClassPosterior(**posterior_payload)

        feature_payload = payload["feature_bank"]
        if not isinstance(feature_payload, dict) or set(feature_payload) != {
            "max_prototypes",
            "merge_cosine",
            "prototypes",
        }:
            raise ValueError("feature_bank fields are invalid")
        prototypes = []
        for prototype_payload in feature_payload["prototypes"]:
            if not isinstance(prototype_payload, dict) or set(prototype_payload) != {
                "vector",
                "model_id",
                "support",
                "observation_count",
            }:
                raise ValueError("feature prototype fields are invalid")
            prototypes.append(FeaturePrototype(**prototype_payload))
        feature_bank = FeaturePrototypeBank(
            feature_payload["max_prototypes"],
            feature_payload["merge_cosine"],
            tuple(prototypes),
        )
        if (
            feature_bank.max_prototypes != self.config.prototype_top_k
            or feature_bank.merge_cosine != self.config.prototype_merge_cosine
        ):
            raise ValueError("feature_bank config does not match registry config")

        view_payload = payload["view_bank"]
        if not isinstance(view_payload, dict) or set(view_payload) != {
            "max_views",
            "minimum_novelty_cosine",
            "views",
        }:
            raise ValueError("view_bank fields are invalid")
        views = []
        for view_payload_item in view_payload["views"]:
            if not isinstance(view_payload_item, dict) or set(view_payload_item) != {
                "observation_id",
                "frame_id",
                "visible_pixel_count",
                "quality",
                "view_direction_xyz",
            }:
                raise ValueError("informative view fields are invalid")
            views.append(InformativeView(**view_payload_item))
        view_bank = InformativeViewBank(
            view_payload["max_views"],
            view_payload["minimum_novelty_cosine"],
            tuple(views),
        )
        if (
            view_bank.max_views != self.config.view_top_k
            or view_bank.minimum_novelty_cosine
            != self.config.view_minimum_novelty_cosine
        ):
            raise ValueError("view_bank config does not match registry config")

        voxel_keys = frozenset(
            tuple(component for component in key) for key in payload["voxel_keys"]
        )
        return PersistentEntity(
            entity_id=payload["entity_id"],
            semantic_id=payload["semantic_id"],
            label=payload["label"],
            semantic_posterior=posterior,
            feature_bank=feature_bank,
            view_bank=view_bank,
            accepted_observation_ids=frozenset(payload["accepted_observation_ids"]),
            semantic_entropy=payload["semantic_entropy"],
            semantic_margin=payload["semantic_margin"],
            accepted_view_count=payload["accepted_view_count"],
            accepted_frame_ids=frozenset(payload["accepted_frame_ids"]),
            first_frame_id=payload["first_frame_id"],
            last_frame_id=payload["last_frame_id"],
            last_revision=payload["last_revision"],
            lifecycle_state=payload["lifecycle_state"],
            voxel_keys=voxel_keys,
            centroid_xyz=payload["centroid_xyz"],
            bounds_min_xyz=payload["bounds_min_xyz"],
            bounds_max_xyz=payload["bounds_max_xyz"],
        )

    def _load_v1_entity(self, payload: dict[str, Any]) -> PersistentEntity:
        semantic_id = _integer(payload["semantic_id"], "semantic_id", minimum=0)
        support = _number(payload["semantic_support"], "semantic_support")
        if semantic_id > 0 and support > 0.0:
            posterior = SparseClassPosterior(((semantic_id, math.log(support)),), support)
        elif semantic_id == 0 and support == 0.0:
            posterior = SparseClassPosterior.empty()
        else:
            raise ValueError("v1 semantic_id and semantic_support must both be positive or zero")
        accepted_frame_ids = frozenset(
            _integer(value, "accepted_frame_ids", minimum=0)
            for value in payload.get("accepted_frame_ids", [payload["last_frame_id"]])
        )
        voxel_keys = frozenset(
            tuple(int(component) for component in key) for key in payload["voxel_keys"]
        )
        return PersistentEntity(
            entity_id=payload["entity_id"],
            semantic_id=semantic_id,
            label=payload["label"],
            semantic_posterior=posterior,
            feature_bank=FeaturePrototypeBank(
                self.config.prototype_top_k,
                self.config.prototype_merge_cosine,
            ),
            view_bank=InformativeViewBank(
                self.config.view_top_k,
                self.config.view_minimum_novelty_cosine,
            ),
            accepted_observation_ids=frozenset(),
            semantic_entropy=posterior.entropy,
            semantic_margin=posterior.margin,
            accepted_view_count=len(accepted_frame_ids),
            accepted_frame_ids=accepted_frame_ids,
            first_frame_id=payload["first_frame_id"],
            last_frame_id=payload["last_frame_id"],
            last_revision=payload["last_revision"],
            lifecycle_state=payload["lifecycle_state"],
            voxel_keys=voxel_keys,
            centroid_xyz=payload["centroid_xyz"],
            bounds_min_xyz=payload["bounds_min_xyz"],
            bounds_max_xyz=payload["bounds_max_xyz"],
        )
