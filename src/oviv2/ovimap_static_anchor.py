"""Causal composition of immutable OVI-MAP anchors and CROVE temporal state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.temporal_export import (
    DynamicState,
    TemporalExportBatch,
)
from src.oviv2.temporal_lifecycle import (
    TemporalEvidenceKind,
    TemporalLifecycle,
)
from src.oviv2.temporal_snapshot import (
    TemporalCurrentSnapshot,
    TemporalSnapshotEntity,
    build_temporal_map_snapshot,
)


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _point(value: object, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite coordinates")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def _points(value: object, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float32, copy=True)
    if array.ndim != 2 or array.shape[1:] != (3,) or not len(array):
        raise ValueError(f"{name} must have shape (N, 3) with N > 0")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    array.setflags(write=False)
    return array


def _embedding(value: object | None, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.array(value, dtype=np.float32, copy=True)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    if float(np.linalg.norm(array)) == 0.0:
        raise ValueError(f"{name} must not be zero")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class StaticAnchorConfig:
    minimum_spatial_iou: float
    maximum_centroid_distance_m: float
    minimum_semantic_cosine: float
    moved_displacement_m: float
    background_voxel_size_m: float

    def __post_init__(self) -> None:
        minimum_iou = _finite(self.minimum_spatial_iou, "minimum_spatial_iou")
        maximum_distance = _finite(
            self.maximum_centroid_distance_m, "maximum_centroid_distance_m"
        )
        minimum_cosine = _finite(
            self.minimum_semantic_cosine, "minimum_semantic_cosine"
        )
        moved = _finite(self.moved_displacement_m, "moved_displacement_m")
        voxel = _finite(self.background_voxel_size_m, "background_voxel_size_m")
        if not 0.0 <= minimum_iou <= 1.0:
            raise ValueError("minimum_spatial_iou must be in [0, 1]")
        if maximum_distance <= 0.0:
            raise ValueError("maximum_centroid_distance_m must be positive")
        if not -1.0 <= minimum_cosine <= 1.0:
            raise ValueError("minimum_semantic_cosine must be in [-1, 1]")
        if moved <= 0.0 or voxel <= 0.0:
            raise ValueError("movement and voxel thresholds must be positive")
        object.__setattr__(self, "minimum_spatial_iou", minimum_iou)
        object.__setattr__(self, "maximum_centroid_distance_m", maximum_distance)
        object.__setattr__(self, "minimum_semantic_cosine", minimum_cosine)
        object.__setattr__(self, "moved_displacement_m", moved)
        object.__setattr__(self, "background_voxel_size_m", voxel)


@dataclass(frozen=True)
class PrefixIdentitySample:
    temporal_entity_id: int
    frame_index: int
    centroid_xyz: tuple[float, float, float]
    points_xyz: np.ndarray
    semantic_label: str | None
    semantic_embedding: np.ndarray | None
    geometry_epoch: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temporal_entity_id",
            _integer(self.temporal_entity_id, "temporal_entity_id", minimum=1),
        )
        object.__setattr__(
            self, "frame_index", _integer(self.frame_index, "frame_index", minimum=0)
        )
        object.__setattr__(
            self, "centroid_xyz", _point(self.centroid_xyz, "centroid_xyz")
        )
        object.__setattr__(self, "points_xyz", _points(self.points_xyz, "points_xyz"))
        label = self.semantic_label
        if label is not None:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("semantic_label must be None or non-empty")
            label = label.strip()
        object.__setattr__(self, "semantic_label", label)
        object.__setattr__(
            self,
            "semantic_embedding",
            _embedding(self.semantic_embedding, "semantic_embedding"),
        )
        object.__setattr__(
            self,
            "geometry_epoch",
            _integer(self.geometry_epoch, "geometry_epoch", minimum=0),
        )


@dataclass(frozen=True)
class AnchorOverlayState:
    cutoff_frame: int
    last_frame_index: int
    bindings: tuple[tuple[str, int], ...]
    initial_geometry_epochs: tuple[tuple[int, int], ...]
    moved_anchor_ids: frozenset[str]
    removed_anchor_ids: frozenset[str]

    def __post_init__(self) -> None:
        cutoff = _integer(self.cutoff_frame, "cutoff_frame", minimum=0)
        last = _integer(self.last_frame_index, "last_frame_index", minimum=cutoff)
        bindings = tuple(sorted(self.bindings))
        if bindings != self.bindings:
            raise ValueError("bindings must be sorted")
        if any(
            not isinstance(anchor_id, str)
            or not anchor_id
            or _integer(temporal_id, "binding temporal ID", minimum=1) != temporal_id
            for anchor_id, temporal_id in bindings
        ):
            raise ValueError("bindings are invalid")
        if len({item[0] for item in bindings}) != len(bindings) or len(
            {item[1] for item in bindings}
        ) != len(bindings):
            raise ValueError("bindings must be one-to-one")
        epochs = tuple(sorted(self.initial_geometry_epochs))
        if epochs != self.initial_geometry_epochs:
            raise ValueError("initial_geometry_epochs must be sorted")
        if {item[0] for item in epochs} != {item[1] for item in bindings}:
            raise ValueError("initial epochs must cover every bound temporal identity")
        for temporal_id, epoch in epochs:
            _integer(temporal_id, "initial epoch temporal ID", minimum=1)
            _integer(epoch, "initial geometry epoch", minimum=0)
        bound_anchors = {item[0] for item in bindings}
        for name in ("moved_anchor_ids", "removed_anchor_ids"):
            values = getattr(self, name)
            if not isinstance(values, frozenset) or any(
                not isinstance(item, str) or not item for item in values
            ):
                raise TypeError(f"{name} must be a frozenset of strings")
            if not values.issubset(bound_anchors):
                raise ValueError("only bound anchors may change overlay state")
        object.__setattr__(self, "cutoff_frame", cutoff)
        object.__setattr__(self, "last_frame_index", last)


def _voxel_keys(points: np.ndarray, voxel_size_m: float) -> set[tuple[int, int, int]]:
    indices = np.floor(np.asarray(points, dtype=np.float64) / voxel_size_m).astype(
        np.int64
    )
    return {tuple(int(value) for value in row) for row in indices}


def _spatial_iou(
    left: np.ndarray, right: np.ndarray, voxel_size_m: float
) -> float:
    left_keys = _voxel_keys(left, voxel_size_m)
    right_keys = _voxel_keys(right, voxel_size_m)
    union = left_keys | right_keys
    return 0.0 if not union else len(left_keys & right_keys) / len(union)


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def _normalized_label(value: str | None) -> str | None:
    return None if value is None else value.casefold().replace("_", "-")


def _pair_score(
    anchor: EntityPrediction,
    sample: PrefixIdentitySample,
    config: StaticAnchorConfig,
) -> float | None:
    anchor_centroid = np.asarray(anchor.points_xyz, dtype=np.float64).mean(axis=0)
    distance = float(
        np.linalg.norm(anchor_centroid - np.asarray(sample.centroid_xyz))
    )
    iou = _spatial_iou(
        anchor.points_xyz, sample.points_xyz, config.background_voxel_size_m
    )
    if (
        iou < config.minimum_spatial_iou
        and distance > config.maximum_centroid_distance_m
    ):
        return None

    anchor_label = _normalized_label(anchor.semantic_label)
    sample_label = _normalized_label(sample.semantic_label)
    labels_equal = (
        anchor_label is not None
        and sample_label is not None
        and anchor_label == sample_label
    )
    cosine = _cosine(anchor.semantic_embedding, sample.semantic_embedding)
    if (
        anchor_label is not None
        and sample_label is not None
        and not labels_equal
        and (cosine is None or cosine < config.minimum_semantic_cosine)
    ):
        return None
    if cosine is not None and cosine < config.minimum_semantic_cosine and not labels_equal:
        return None

    centroid_score = max(
        0.0, 1.0 - distance / config.maximum_centroid_distance_m
    )
    semantic_score = 1.0 if labels_equal else (0.5 if cosine is None else (cosine + 1.0) / 2.0)
    return 0.5 * iou + 0.3 * centroid_score + 0.2 * semantic_score


def bind_anchor_identities(
    anchor: MapSnapshot,
    samples: tuple[PrefixIdentitySample, ...],
    config: StaticAnchorConfig,
    *,
    cutoff_frame: int,
) -> AnchorOverlayState:
    """Bind each static anchor to at most one causal-prefix temporal identity."""

    if not isinstance(anchor, MapSnapshot):
        raise TypeError("anchor must be a MapSnapshot")
    if not isinstance(config, StaticAnchorConfig):
        raise TypeError("config must be StaticAnchorConfig")
    if not isinstance(samples, tuple) or any(
        not isinstance(item, PrefixIdentitySample) for item in samples
    ):
        raise TypeError("samples must be a tuple of PrefixIdentitySample values")
    cutoff = _integer(cutoff_frame, "cutoff_frame", minimum=0)
    if any(item.frame_index > cutoff for item in samples):
        raise ValueError("prefix identity sample occurs after causal cutoff")
    temporal_ids = tuple(item.temporal_entity_id for item in samples)
    if len(set(temporal_ids)) != len(temporal_ids):
        raise ValueError("prefix temporal entity IDs must be unique")
    anchors = tuple(sorted(anchor.entities, key=lambda item: item.entity_id))
    ordered_samples = tuple(
        sorted(samples, key=lambda item: item.temporal_entity_id)
    )
    if not anchors or not ordered_samples:
        return AnchorOverlayState(
            cutoff, cutoff, (), (), frozenset(), frozenset()
        )

    sentinel = 1_000_000.0
    costs = np.full((len(anchors), len(ordered_samples)), sentinel, dtype=np.float64)
    valid = np.zeros_like(costs, dtype=np.bool_)
    for row, anchor_entity in enumerate(anchors):
        for column, sample in enumerate(ordered_samples):
            score = _pair_score(anchor_entity, sample, config)
            if score is None:
                continue
            valid[row, column] = True
            costs[row, column] = (
                -score + row * 1e-9 + column * 1e-12
            )
    row_indices, column_indices = linear_sum_assignment(costs)
    pairs = tuple(
        sorted(
            (
                anchors[int(row)].entity_id,
                ordered_samples[int(column)].temporal_entity_id,
            )
            for row, column in zip(row_indices, column_indices, strict=True)
            if valid[int(row), int(column)]
        )
    )
    sample_by_id = {
        item.temporal_entity_id: item for item in ordered_samples
    }
    epochs = tuple(
        sorted(
            (temporal_id, sample_by_id[temporal_id].geometry_epoch)
            for _, temporal_id in pairs
        )
    )
    return AnchorOverlayState(
        cutoff_frame=cutoff,
        last_frame_index=cutoff,
        bindings=pairs,
        initial_geometry_epochs=epochs,
        moved_anchor_ids=frozenset(),
        removed_anchor_ids=frozenset(),
    )


@dataclass(frozen=True)
class CheckpointOverlayDiagnostics:
    frame_index: int
    unchanged_anchor_ids: tuple[str, ...]
    moved_anchor_ids: tuple[str, ...]
    removed_anchor_ids: tuple[str, ...]
    new_temporal_ids: tuple[int, ...]
    occluded_anchor_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_index",
            _integer(self.frame_index, "frame_index", minimum=0),
        )
        for name in (
            "unchanged_anchor_ids",
            "moved_anchor_ids",
            "removed_anchor_ids",
            "occluded_anchor_ids",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be a sorted unique tuple")
        if (
            not isinstance(self.new_temporal_ids, tuple)
            or self.new_temporal_ids != tuple(sorted(set(self.new_temporal_ids)))
            or any(
                _integer(value, "new temporal ID", minimum=1) != value
                for value in self.new_temporal_ids
            )
        ):
            raise ValueError("new_temporal_ids must be a sorted unique tuple")


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 value")
    return value


def _class_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("class_names must be a sequence")
    names = tuple(str(item).strip() for item in values)
    if not names or any(not item for item in names):
        raise ValueError("class_names must contain non-empty values")
    return names


def _copy_prediction(
    entity: EntityPrediction,
    *,
    metadata: dict[str, object],
    lifecycle_state: str | None = None,
) -> EntityPrediction:
    merged = dict(entity.metadata)
    merged.update(metadata)
    return EntityPrediction(
        entity_id=entity.entity_id,
        points_xyz=entity.points_xyz,
        semantic_embedding=entity.semantic_embedding,
        semantic_label=entity.semantic_label,
        semantic_score=entity.semantic_score,
        lifecycle_state=(
            entity.lifecycle_state if lifecycle_state is None else lifecycle_state
        ),
        first_seen=entity.first_seen,
        last_seen=entity.last_seen,
        metadata=merged,
    )


def _prediction_from_wrapper(
    wrapper: TemporalSnapshotEntity, class_names: tuple[str, ...]
) -> EntityPrediction:
    probabilities = wrapper.semantic_probabilities
    class_id, score = (
        max(probabilities, key=lambda item: (item[1], -item[0]))
        if probabilities
        else (0, 0.0)
    )
    label = class_names[class_id] if 0 <= class_id < len(class_names) else None
    points = wrapper.submap.world_points(wrapper.object_to_world).astype(
        np.float32, copy=False
    )
    return EntityPrediction(
        entity_id=f"temporal:{wrapper.lifecycle.entity_id}",
        points_xyz=points,
        semantic_embedding=wrapper.image_prototype,
        semantic_label=label,
        semantic_score=score,
        lifecycle_state=wrapper.lifecycle.lifecycle.value,
        first_seen=float(wrapper.first_seen_frame_id),
        last_seen=float(wrapper.last_seen_frame_id),
        metadata={
            "temporal_entity_id": wrapper.lifecycle.entity_id,
            "semantic_id": class_id,
            "geometry_epoch": wrapper.geometry_epoch,
            "readout_valid": wrapper.readout_valid,
        },
    )


def _merged_background(
    anchor: np.ndarray | None,
    temporal: np.ndarray | None,
    voxel_size_m: float,
) -> np.ndarray | None:
    selected: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for source in (anchor, temporal):
        if source is None:
            continue
        for point in np.asarray(source, dtype=np.float32):
            key = tuple(
                int(value)
                for value in np.floor(
                    np.asarray(point, dtype=np.float64) / voxel_size_m
                ).astype(np.int64)
            )
            candidate = tuple(float(value) for value in point)
            previous = selected.get(key)
            if previous is None or candidate < previous:
                selected[key] = candidate
    if not selected:
        return None
    return np.asarray([selected[key] for key in sorted(selected)], dtype=np.float32)


def compose_anchor_checkpoint(
    *,
    anchor: MapSnapshot,
    temporal: TemporalCurrentSnapshot | MapSnapshot,
    frame_index: int | None = None,
    exports: tuple[TemporalExportBatch, ...],
    state: AnchorOverlayState,
    config: StaticAnchorConfig,
    anchor_manifest_sha256: str,
    class_names: Sequence[str],
) -> tuple[MapSnapshot, AnchorOverlayState, CheckpointOverlayDiagnostics]:
    """Compose one causal current readout without mutating either authority."""

    if not isinstance(anchor, MapSnapshot):
        raise TypeError("anchor must be a MapSnapshot")
    if not isinstance(state, AnchorOverlayState):
        raise TypeError("state must be an AnchorOverlayState")
    if not isinstance(config, StaticAnchorConfig):
        raise TypeError("config must be a StaticAnchorConfig")
    manifest_hash = _sha256(anchor_manifest_sha256, "anchor_manifest_sha256")
    names = _class_names(class_names)
    wrappers: dict[int, TemporalSnapshotEntity]
    if isinstance(temporal, TemporalCurrentSnapshot):
        observed_frame_index = temporal.metadata.frame_id
        if frame_index is not None and (
            _integer(frame_index, "frame_index", minimum=0) != observed_frame_index
        ):
            raise ValueError("explicit frame_index does not match temporal checkpoint")
        temporal_scene_id = temporal.metadata.scene_id
        temporal_neutral = build_temporal_map_snapshot(temporal, names)
        wrappers = {
            item.lifecycle.entity_id: item for item in temporal.entities
        }
    elif isinstance(temporal, MapSnapshot):
        if frame_index is None:
            raise ValueError("frame_index is required for persisted temporal maps")
        observed_frame_index = _integer(frame_index, "frame_index", minimum=0)
        if temporal.scope != "current":
            raise ValueError("persisted temporal map must have current scope")
        temporal_scene_id = temporal.scene_id
        temporal_neutral = temporal
        wrappers = {}
    else:
        raise TypeError("temporal must be a TemporalCurrentSnapshot or MapSnapshot")
    if observed_frame_index <= state.last_frame_index:
        raise ValueError("checkpoint frame must increase strictly")
    if temporal_scene_id != anchor.scene_id:
        raise ValueError("anchor and temporal checkpoint scenes must match")
    if (
        not isinstance(exports, tuple)
        or not exports
        or any(not isinstance(item, TemporalExportBatch) for item in exports)
    ):
        raise TypeError("exports must be a non-empty tuple of TemporalExportBatch")
    export_frames = tuple(item.frame_index for item in exports)
    if (
        export_frames != tuple(sorted(set(export_frames)))
        or export_frames[0] <= state.last_frame_index
        or export_frames[-1] != observed_frame_index
        or any(item > observed_frame_index for item in export_frames)
    ):
        raise ValueError("exports do not cover the next checkpoint interval")

    bindings = dict(state.bindings)
    anchor_by_temporal = {
        temporal_id: anchor_id for anchor_id, temporal_id in state.bindings
    }
    initial_epochs = dict(state.initial_geometry_epochs)
    anchor_entities = {
        entity.entity_id: entity for entity in anchor.entities
    }
    moved = set(state.moved_anchor_ids)
    removed = set(state.removed_anchor_ids)
    latest_samples = {}
    latest_events = {}
    for batch in exports:
        for sample in batch.samples:
            latest_samples[sample.entity_id] = sample
        for event in batch.events:
            latest_events[event.entity_id] = event
            anchor_id = anchor_by_temporal.get(event.entity_id)
            if anchor_id is None:
                continue
            if (
                event.evidence is TemporalEvidenceKind.VISIBLE_ABSENT
                and event.after is TemporalLifecycle.DORMANT
            ):
                removed.add(anchor_id)
            elif (
                event.evidence is TemporalEvidenceKind.PRESENT
                and event.after
                in (TemporalLifecycle.ACTIVE, TemporalLifecycle.UNCERTAIN)
            ):
                removed.discard(anchor_id)

    for temporal_id, sample in latest_samples.items():
        anchor_id = anchor_by_temporal.get(temporal_id)
        if anchor_id is None:
            continue
        anchor_entity = anchor_entities[anchor_id]
        centroid = np.asarray(anchor_entity.points_xyz, dtype=np.float64).mean(axis=0)
        displacement = float(
            np.linalg.norm(centroid - np.asarray(sample.centroid_xyz))
        )
        if (
            sample.dynamic_state is DynamicState.DYNAMIC
            and sample.geometry_epoch > initial_epochs[temporal_id]
            and displacement >= config.moved_displacement_m
        ):
            moved.add(anchor_id)
        elif (
            sample.dynamic_state is DynamicState.STATIC
            and displacement < config.moved_displacement_m
        ):
            moved.discard(anchor_id)

    active_predictions = {
        int(entity.metadata["temporal_entity_id"]): entity
        for entity in temporal_neutral.entities
    }
    if len(active_predictions) != len(temporal_neutral.entities):
        raise ValueError("temporal checkpoint entity IDs must be unique")
    predictions: list[EntityPrediction] = []
    unchanged_ids: list[str] = []
    moved_ids: list[str] = []
    removed_ids: list[str] = []
    occluded_ids: list[str] = []
    for anchor_id in sorted(anchor_entities):
        anchor_entity = anchor_entities[anchor_id]
        temporal_id = bindings.get(anchor_id)
        if anchor_id in removed:
            removed_ids.append(anchor_id)
            continue
        if anchor_id in moved and temporal_id is not None:
            prediction = active_predictions.get(temporal_id)
            wrapper = wrappers.get(temporal_id)
            if prediction is None and wrapper is not None:
                prediction = _prediction_from_wrapper(wrapper, names)
            if prediction is not None:
                predictions.append(
                    _copy_prediction(
                        prediction,
                        metadata={
                            "authority": "crove_temporal",
                            "anchor_entity_id": anchor_id,
                            "anchor_manifest_sha256": manifest_hash,
                            "overlay_state": "moved",
                        },
                    )
                )
                moved_ids.append(anchor_id)
                continue
        overlay_state = "unchanged"
        if temporal_id is not None:
            wrapper = wrappers.get(temporal_id)
            event = latest_events.get(temporal_id)
            if wrapper is not None and wrapper.lifecycle.lifecycle is TemporalLifecycle.DORMANT:
                overlay_state = "occluded"
                occluded_ids.append(anchor_id)
            elif event is not None and event.evidence in {
                TemporalEvidenceKind.OCCLUDED,
                TemporalEvidenceKind.OUT_OF_VIEW,
                TemporalEvidenceKind.DEPTH_UNKNOWN,
            }:
                overlay_state = "occluded"
                occluded_ids.append(anchor_id)
        if overlay_state == "unchanged":
            unchanged_ids.append(anchor_id)
        predictions.append(
            _copy_prediction(
                anchor_entity,
                metadata={
                    "authority": "ovimap_anchor",
                    "anchor_entity_id": anchor_id,
                    "temporal_entity_id": temporal_id,
                    "anchor_manifest_sha256": manifest_hash,
                    "overlay_state": overlay_state,
                },
            )
        )

    bound_temporal_ids = set(anchor_by_temporal)
    new_temporal_ids = tuple(
        sorted(set(active_predictions) - bound_temporal_ids)
    )
    for temporal_id in new_temporal_ids:
        predictions.append(
            _copy_prediction(
                active_predictions[temporal_id],
                metadata={
                    "authority": "crove_temporal",
                    "anchor_entity_id": None,
                    "anchor_manifest_sha256": manifest_hash,
                    "overlay_state": "new",
                },
            )
        )
    predictions.sort(key=lambda item: item.entity_id)
    composed = MapSnapshot(
        method="CROVE + OVI-MAP static anchor (composed)",
        scene_id=anchor.scene_id,
        timestamp=temporal_neutral.timestamp,
        entities=predictions,
        background_xyz=_merged_background(
            anchor.background_xyz,
            temporal_neutral.background_xyz,
            config.background_voxel_size_m,
        ),
        scope="current",
        runtime=dict(temporal_neutral.runtime),
    )
    next_state = AnchorOverlayState(
        cutoff_frame=state.cutoff_frame,
        last_frame_index=observed_frame_index,
        bindings=state.bindings,
        initial_geometry_epochs=state.initial_geometry_epochs,
        moved_anchor_ids=frozenset(moved),
        removed_anchor_ids=frozenset(removed),
    )
    diagnostics = CheckpointOverlayDiagnostics(
        frame_index=observed_frame_index,
        unchanged_anchor_ids=tuple(sorted(unchanged_ids)),
        moved_anchor_ids=tuple(sorted(moved_ids)),
        removed_anchor_ids=tuple(sorted(removed_ids)),
        new_temporal_ids=new_temporal_ids,
        occluded_anchor_ids=tuple(sorted(occluded_ids)),
    )
    return composed, next_state, diagnostics


__all__ = [
    "AnchorOverlayState",
    "CheckpointOverlayDiagnostics",
    "PrefixIdentitySample",
    "StaticAnchorConfig",
    "bind_anchor_identities",
    "compose_anchor_checkpoint",
]
