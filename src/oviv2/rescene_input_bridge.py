"""Typed D-to-A-to-M contracts for source-bound OVI to ReScene inference."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.oviv2.ovi_surface_attributes import SurfaceGroup, validate_surface_group
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    OviEntitySemanticEvidence,
    VisitMap,
    snapshot_content_sha256,
    validate_visit_pair,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MODEL_ARRAY_KEYS = frozenset(
    {
        "coordinates_bxyzt",
        "grid_coordinates_xyz",
        "features",
        "sparse_batch_offsets",
        "point2segment",
        "adapter_to_model",
        "model_visit_ids",
        "representative_source_point_indices",
        "local_frame_indices",
        "global_frame_indices",
        "rows",
        "columns",
        "camera_depth_m",
        "observed_depth_m",
        "depth_residual_m",
    }
)
_MODEL_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "status",
        "content_sha256",
        "arrays",
        "neural_voxel_size_m",
        "adapter_geometry_sha256",
        "native_sampling_sha256",
        "surface_attributes_sha256",
        "model_candidates_sha256",
        "sampler_source_sha256",
    }
)


class BridgeError(ValueError):
    """Raised when one provenance or index domain cannot be closed exactly."""


def _readonly(values: object, dtype: np.dtype | type, ndim: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != ndim:
        raise BridgeError(f"array must have {ndim} dimensions")
    result = np.array(raw, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _integer_vector(values: object, dtype: np.dtype | type, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(
        raw.dtype, np.bool_
    ):
        raise BridgeError(f"{name} must be a one-dimensional integer array")
    return _readonly(raw, dtype, 1)


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BridgeError(f"{name} must be a lowercase SHA-256")
    return value


def _array_record(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _content_digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AdapterGeometry:
    """Entity-aware adapter geometry with compact source membership."""

    coordinates_xyzt: np.ndarray
    visit_ids: np.ndarray
    source_visit_ids: np.ndarray
    source_entity_indices: np.ndarray
    source_entity_local_indices: np.ndarray
    source_point_indices: np.ndarray
    source_to_adapter_offsets: np.ndarray
    entity_keys: tuple[tuple[int, str], ...]
    entity_semantics: tuple[OviEntitySemanticEvidence, ...]
    neural_voxel_size_m: float
    coordinate_frame_id: str
    source_manifest_sha256: str
    source_visit_map_sha256: tuple[str, str]
    shared_center_xyz: np.ndarray
    _content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        coordinates = _readonly(self.coordinates_xyzt, np.float64, 2)
        if coordinates.shape[1:] != (4,) or not np.all(np.isfinite(coordinates)):
            raise BridgeError("adapter coordinates must have finite shape (N, 4)")
        if len(coordinates) == 0:
            raise BridgeError("adapter geometry cannot be empty")
        visits = _integer_vector(self.visit_ids, np.int8, "adapter visit IDs")
        if visits.shape != (len(coordinates),) or set(visits.tolist()) != {0, 1}:
            raise BridgeError("adapter visit IDs must align and contain visits 0 and 1")
        if not np.array_equal(coordinates[:, 3], visits.astype(np.float64)):
            raise BridgeError("adapter time must equal visit ID")

        source_visits = _integer_vector(
            self.source_visit_ids, np.int8, "source visit IDs"
        )
        entity_indices = _integer_vector(
            self.source_entity_indices, np.int32, "source entity indices"
        )
        local_indices = _integer_vector(
            self.source_entity_local_indices,
            np.int64,
            "source entity-local indices",
        )
        point_indices = _integer_vector(
            self.source_point_indices, np.int64, "source point indices"
        )
        offsets = _integer_vector(
            self.source_to_adapter_offsets,
            np.int64,
            "source-to-adapter offsets",
        )
        source_count = len(point_indices)
        if any(len(value) != source_count for value in (source_visits, entity_indices, local_indices)):
            raise BridgeError("source membership arrays must have equal lengths")
        if offsets.shape != (len(coordinates) + 1,) or offsets[0] != 0 or offsets[-1] != source_count:
            raise BridgeError("source-to-adapter offsets do not span every source row")
        counts = np.diff(offsets)
        if np.any(counts <= 0):
            raise BridgeError("every adapter token must have source contributors")
        if not np.array_equal(
            np.sort(point_indices), np.arange(source_count, dtype=np.int64)
        ):
            raise BridgeError("source point indices must be a conserving permutation")
        if np.any(local_indices < 0):
            raise BridgeError("source entity-local indices must be nonnegative")

        keys = tuple(self.entity_keys)
        if not keys or any(
            type(visit) is not int
            or visit not in (0, 1)
            or not isinstance(entity_id, str)
            or not entity_id
            for visit, entity_id in keys
        ):
            raise BridgeError("entity key table is invalid")
        if len(keys) != len(set(keys)) or tuple(sorted(keys)) != keys:
            raise BridgeError("entity keys must be unique and sorted")
        if np.any(entity_indices < 0) or np.any(entity_indices >= len(keys)):
            raise BridgeError("source entity index is outside the entity table")
        contributor_min_visit = np.minimum.reduceat(source_visits, offsets[:-1])
        contributor_max_visit = np.maximum.reduceat(source_visits, offsets[:-1])
        contributor_min_entity = np.minimum.reduceat(entity_indices, offsets[:-1])
        contributor_max_entity = np.maximum.reduceat(entity_indices, offsets[:-1])
        if not np.array_equal(contributor_min_visit, visits) or not np.array_equal(
            contributor_max_visit, visits
        ):
            raise BridgeError("an adapter contributor span crosses visits")
        if not np.array_equal(contributor_min_entity, contributor_max_entity):
            raise BridgeError("an adapter contributor span crosses entities")
        key_visits = np.asarray([item[0] for item in keys], dtype=np.int8)
        if not np.array_equal(key_visits[entity_indices], source_visits):
            raise BridgeError("source entity and visit indices disagree")

        semantics = tuple(self.entity_semantics)
        if any(not isinstance(item, OviEntitySemanticEvidence) for item in semantics):
            raise BridgeError("entity semantics contain an invalid record")
        if tuple((item.visit_id, item.entity_id) for item in semantics) != keys:
            raise BridgeError("entity semantics must exactly cover the entity table")
        voxel = float(self.neural_voxel_size_m)
        if not math.isfinite(voxel) or voxel <= 0.0:
            raise BridgeError("neural voxel size must be finite and positive")
        if not isinstance(self.coordinate_frame_id, str) or not self.coordinate_frame_id:
            raise BridgeError("coordinate frame ID must be non-empty")
        manifest = _sha256(self.source_manifest_sha256, "source manifest SHA-256")
        if not isinstance(self.source_visit_map_sha256, tuple) or len(self.source_visit_map_sha256) != 2:
            raise BridgeError("source VisitMap hashes must contain t0 and t1")
        visit_hashes = (
            _sha256(self.source_visit_map_sha256[0], "t0 VisitMap SHA-256"),
            _sha256(self.source_visit_map_sha256[1], "t1 VisitMap SHA-256"),
        )
        center = _readonly(self.shared_center_xyz, np.float64, 1)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise BridgeError("shared center must contain three finite values")

        object.__setattr__(self, "coordinates_xyzt", coordinates)
        object.__setattr__(self, "visit_ids", visits)
        object.__setattr__(self, "source_visit_ids", source_visits)
        object.__setattr__(self, "source_entity_indices", entity_indices)
        object.__setattr__(self, "source_entity_local_indices", local_indices)
        object.__setattr__(self, "source_point_indices", point_indices)
        object.__setattr__(self, "source_to_adapter_offsets", offsets)
        object.__setattr__(self, "entity_keys", keys)
        object.__setattr__(self, "entity_semantics", semantics)
        object.__setattr__(self, "neural_voxel_size_m", voxel)
        object.__setattr__(self, "source_manifest_sha256", manifest)
        object.__setattr__(self, "source_visit_map_sha256", visit_hashes)
        object.__setattr__(self, "shared_center_xyz", center)
        object.__setattr__(
            self,
            "_content_sha256",
            _content_digest(
                {
                    "arrays": {
                        name: _array_record(getattr(self, name))
                        for name in (
                            "coordinates_xyzt",
                            "visit_ids",
                            "source_visit_ids",
                            "source_entity_indices",
                            "source_entity_local_indices",
                            "source_point_indices",
                            "source_to_adapter_offsets",
                            "shared_center_xyz",
                        )
                    },
                    "entity_keys": self.entity_keys,
                    "neural_voxel_size_m": self.neural_voxel_size_m,
                    "coordinate_frame_id": self.coordinate_frame_id,
                    "source_manifest_sha256": self.source_manifest_sha256,
                    "source_visit_map_sha256": self.source_visit_map_sha256,
                }
            ),
        )

    @property
    def adapter_count(self) -> int:
        return len(self.coordinates_xyzt)

    @property
    def source_point_count(self) -> int:
        return len(self.source_point_indices)

    @property
    def source_entity_ids(self) -> tuple[str, ...]:
        return tuple(self.entity_keys[int(index)][1] for index in self.source_entity_indices)

    @property
    def token_entity_ids(self) -> tuple[str, ...]:
        return tuple(
            self.entity_keys[int(self.source_entity_indices[int(offset)])][1]
            for offset in self.source_to_adapter_offsets[:-1]
        )

    @property
    def token_entity_indices(self) -> np.ndarray:
        result = self.source_entity_indices[self.source_to_adapter_offsets[:-1]]
        result.setflags(write=False)
        return result

    def content_sha256(self) -> str:
        return self._content_sha256


@dataclass(frozen=True, slots=True)
class SurfaceAttributeBundle:
    """Source arrays indexed by the global D index."""

    points_xyz: np.ndarray
    normals_xyz: np.ndarray
    normal_valid: np.ndarray
    original_vertex_indices: np.ndarray
    source_visit_ids: np.ndarray
    source_entity_indices: np.ndarray
    adapter_geometry_sha256: str
    rgb_source: str = "not_materialized"
    _content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        points = _readonly(self.points_xyz, np.float32, 2)
        normals = _readonly(self.normals_xyz, np.float32, 2)
        valid = _readonly(self.normal_valid, np.bool_, 1)
        original = _integer_vector(
            self.original_vertex_indices, np.int64, "original PLY vertex indices"
        )
        visits = _integer_vector(self.source_visit_ids, np.int8, "surface visit IDs")
        entities = _integer_vector(
            self.source_entity_indices, np.int32, "surface entity indices"
        )
        count = len(points)
        if points.shape[1:] != (3,) or normals.shape != points.shape:
            raise BridgeError("surface point and normal arrays must have shape (D, 3)")
        if any(len(value) != count for value in (valid, original, visits, entities)):
            raise BridgeError("surface arrays must have equal D rows")
        if not np.all(np.isfinite(points)):
            raise BridgeError("surface points must be finite")
        for visit_id in (0, 1):
            visit_original = original[visits == visit_id]
            if len(np.unique(visit_original)) != len(visit_original):
                raise BridgeError("original PLY rows must be unique within each visit")
        if self.rgb_source != "not_materialized":
            raise BridgeError("surface bundle cannot claim camera RGB before recovery")
        object.__setattr__(self, "points_xyz", points)
        object.__setattr__(self, "normals_xyz", normals)
        object.__setattr__(self, "normal_valid", valid)
        object.__setattr__(self, "original_vertex_indices", original)
        object.__setattr__(self, "source_visit_ids", visits)
        object.__setattr__(self, "source_entity_indices", entities)
        object.__setattr__(
            self,
            "adapter_geometry_sha256",
            _sha256(self.adapter_geometry_sha256, "adapter geometry SHA-256"),
        )
        object.__setattr__(
            self,
            "_content_sha256",
            _content_digest(
                {
                    "arrays": {
                        name: _array_record(getattr(self, name))
                        for name in (
                            "points_xyz",
                            "normals_xyz",
                            "normal_valid",
                            "original_vertex_indices",
                            "source_visit_ids",
                            "source_entity_indices",
                        )
                    },
                    "adapter_geometry_sha256": self.adapter_geometry_sha256,
                    "rgb_source": self.rgb_source,
                }
            ),
        )

    def content_sha256(self) -> str:
        return self._content_sha256


@dataclass(frozen=True, slots=True)
class NativeSamplingMap:
    """Actual native GridSample inverse from A rows to M rows."""

    adapter_to_model: np.ndarray
    selected_adapter_indices: np.ndarray
    model_grid_coordinates: np.ndarray
    model_visit_ids: np.ndarray
    visit_model_offsets: np.ndarray
    visit_grid_origins: np.ndarray
    voxel_size_m: float
    sampler_seed: int
    sampler_mode: str
    sampler_source_sha256: str
    _content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mapping = _integer_vector(
            self.adapter_to_model, np.int64, "adapter-to-model mapping"
        )
        selected = _integer_vector(
            self.selected_adapter_indices, np.int64, "selected adapter indices"
        )
        grid = _readonly(self.model_grid_coordinates, np.int64, 2)
        visits = _integer_vector(self.model_visit_ids, np.int8, "model visit IDs")
        offsets = _integer_vector(
            self.visit_model_offsets, np.int64, "visit model offsets"
        )
        origins = _readonly(self.visit_grid_origins, np.int64, 2)
        model_count = len(selected)
        if model_count == 0 or len(mapping) == 0:
            raise BridgeError("native sampling cannot be empty")
        if grid.shape != (model_count, 3) or visits.shape != (model_count,):
            raise BridgeError("native model arrays must have M aligned rows")
        if origins.shape != (2, 3):
            raise BridgeError("visit grid origins must have shape (2, 3)")
        if np.any(mapping < 0) or np.any(mapping >= model_count):
            raise BridgeError("adapter-to-model mapping values are outside model range")
        if not np.array_equal(np.unique(mapping), np.arange(model_count)):
            raise BridgeError("model indices must form one contiguous covered range")
        if np.any(selected < 0) or np.any(selected >= len(mapping)) or len(np.unique(selected)) != model_count:
            raise BridgeError("selected adapter indices must be unique and in range")
        if not np.array_equal(mapping[selected], np.arange(model_count)):
            raise BridgeError("each selected adapter must represent its model row")
        if offsets.shape != (3,) or not np.array_equal(
            offsets, [0, np.count_nonzero(visits == 0), model_count]
        ):
            raise BridgeError("visit model offsets must exactly split t0 and t1")
        if not np.array_equal(visits[: offsets[1]], np.zeros(offsets[1], dtype=np.int8)) or not np.array_equal(
            visits[offsets[1] :], np.ones(model_count - offsets[1], dtype=np.int8)
        ):
            raise BridgeError("model visit rows must be ordered t0 then t1")
        if offsets[1] in (0, model_count):
            raise BridgeError("native sampling must retain both visits")
        for visit_id in (0, 1):
            visit_grid = grid[visits == visit_id]
            if not np.array_equal(visit_grid.min(axis=0), np.zeros(3, dtype=np.int64)):
                raise BridgeError("each native visit grid must start at zero origin")
        voxel = float(self.voxel_size_m)
        if not math.isfinite(voxel) or voxel <= 0.0:
            raise BridgeError("native voxel size must be finite and positive")
        if type(self.sampler_seed) is not int or self.sampler_seed < 0:
            raise BridgeError("sampler seed must be a nonnegative integer")
        if self.sampler_mode != "train":
            raise BridgeError("native sampler mode must be train")
        object.__setattr__(self, "adapter_to_model", mapping)
        object.__setattr__(self, "selected_adapter_indices", selected)
        object.__setattr__(self, "model_grid_coordinates", grid)
        object.__setattr__(self, "model_visit_ids", visits)
        object.__setattr__(self, "visit_model_offsets", offsets)
        object.__setattr__(self, "visit_grid_origins", origins)
        object.__setattr__(self, "voxel_size_m", voxel)
        object.__setattr__(
            self,
            "sampler_source_sha256",
            _sha256(self.sampler_source_sha256, "sampler source SHA-256"),
        )
        object.__setattr__(
            self,
            "_content_sha256",
            _content_digest(
                {
                    "arrays": {
                        name: _array_record(getattr(self, name))
                        for name in (
                            "adapter_to_model",
                            "selected_adapter_indices",
                            "model_grid_coordinates",
                            "model_visit_ids",
                            "visit_model_offsets",
                            "visit_grid_origins",
                        )
                    },
                    "voxel_size_m": self.voxel_size_m,
                    "sampler_seed": self.sampler_seed,
                    "sampler_mode": self.sampler_mode,
                    "sampler_source_sha256": self.sampler_source_sha256,
                }
            ),
        )

    @property
    def adapter_count(self) -> int:
        return len(self.adapter_to_model)

    @property
    def model_count(self) -> int:
        return len(self.selected_adapter_indices)

    @property
    def merge_count(self) -> int:
        return self.adapter_count - self.model_count

    def cross_entity_model_token_count(self, geometry: AdapterGeometry) -> int:
        validate_native_sampling(geometry, self)
        entities = geometry.token_entity_indices
        order = np.argsort(self.adapter_to_model, kind="stable")
        counts = np.bincount(self.adapter_to_model, minlength=self.model_count)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        minimum = np.minimum.reduceat(entities[order], offsets[:-1])
        maximum = np.maximum.reduceat(entities[order], offsets[:-1])
        return int(np.count_nonzero(minimum != maximum))

    def content_sha256(self) -> str:
        return self._content_sha256


@dataclass(frozen=True, slots=True)
class ModelCandidateMap:
    source_point_indices: np.ndarray
    candidate_counts: np.ndarray
    maximum_candidates: int
    _content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        source = _readonly(self.source_point_indices, np.int64, 2)
        counts = _integer_vector(self.candidate_counts, np.int16, "candidate counts")
        if type(self.maximum_candidates) is not int or self.maximum_candidates <= 0:
            raise BridgeError("maximum candidates must be a positive integer")
        if source.shape != (len(counts), self.maximum_candidates):
            raise BridgeError("candidate array shape is invalid")
        if np.any(counts < 0) or np.any(counts > self.maximum_candidates):
            raise BridgeError("candidate count is outside the fixed width")
        for row, count in zip(source, counts, strict=True):
            if np.any(row[:count] < 0) or np.any(row[count:] != -1):
                raise BridgeError("candidate padding or source index is invalid")
            if len(np.unique(row[:count])) != count:
                raise BridgeError("model candidates must be unique")
        object.__setattr__(self, "source_point_indices", source)
        object.__setattr__(self, "candidate_counts", counts)
        object.__setattr__(
            self,
            "_content_sha256",
            _content_digest(
                {
                    "source_point_indices": _array_record(source),
                    "candidate_counts": _array_record(counts),
                    "maximum_candidates": self.maximum_candidates,
                }
            ),
        )

    def content_sha256(self) -> str:
        return self._content_sha256


@dataclass(frozen=True, slots=True)
class RecoveredModelSupport:
    """Same-visit RGB-D support, including explicit unsupported M rows."""

    support_valid: np.ndarray
    representative_source_point_indices: np.ndarray
    rgb_uint8: np.ndarray
    local_frame_indices: np.ndarray
    global_frame_indices: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    camera_depth_m: np.ndarray
    observed_depth_m: np.ndarray
    depth_residual_m: np.ndarray

    def __post_init__(self) -> None:
        valid = _readonly(self.support_valid, np.bool_, 1)
        representatives = _integer_vector(
            self.representative_source_point_indices,
            np.int64,
            "representative source point indices",
        )
        rgb = _readonly(self.rgb_uint8, np.uint8, 2)
        local_frames = _integer_vector(
            self.local_frame_indices, np.int64, "local frame indices"
        )
        global_frames = _integer_vector(
            self.global_frame_indices, np.int64, "global frame indices"
        )
        rows = _integer_vector(self.rows, np.int64, "support rows")
        columns = _integer_vector(self.columns, np.int64, "support columns")
        camera = _readonly(self.camera_depth_m, np.float32, 1)
        observed = _readonly(self.observed_depth_m, np.float32, 1)
        residual = _readonly(self.depth_residual_m, np.float32, 1)
        count = len(valid)
        if rgb.shape != (count, 3) or any(
            len(value) != count
            for value in (
                representatives,
                local_frames,
                global_frames,
                rows,
                columns,
                camera,
                observed,
                residual,
            )
        ):
            raise BridgeError("recovered support arrays must have M aligned rows")
        integer_support = (representatives, local_frames, global_frames, rows, columns)
        if any(np.any(value[valid] < 0) or np.any(value[~valid] != -1) for value in integer_support):
            raise BridgeError("support indices must use -1 only for unsupported rows")
        if any(
            np.any(~np.isfinite(value[valid])) or np.any(~np.isnan(value[~valid]))
            for value in (camera, observed, residual)
        ):
            raise BridgeError("support depth values must be finite or explicit NaN")
        if np.any(camera[valid] <= 0.0) or np.any(observed[valid] <= 0.0) or np.any(residual[valid] < 0.0):
            raise BridgeError("support depth values are outside their legal range")
        object.__setattr__(self, "support_valid", valid)
        object.__setattr__(self, "representative_source_point_indices", representatives)
        object.__setattr__(self, "rgb_uint8", rgb)
        object.__setattr__(self, "local_frame_indices", local_frames)
        object.__setattr__(self, "global_frame_indices", global_frames)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "camera_depth_m", camera)
        object.__setattr__(self, "observed_depth_m", observed)
        object.__setattr__(self, "depth_residual_m", residual)


@dataclass(frozen=True, slots=True)
class ReSceneModelInput:
    """Complete M-domain tensors and support provenance for the executor."""

    coordinates_bxyzt: np.ndarray
    grid_coordinates_xyz: np.ndarray
    features: np.ndarray
    sparse_batch_offsets: np.ndarray
    point2segment: np.ndarray
    adapter_to_model: np.ndarray
    model_visit_ids: np.ndarray
    representative_source_point_indices: np.ndarray
    local_frame_indices: np.ndarray
    global_frame_indices: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    camera_depth_m: np.ndarray
    observed_depth_m: np.ndarray
    depth_residual_m: np.ndarray
    neural_voxel_size_m: float
    adapter_geometry_sha256: str
    native_sampling_sha256: str
    surface_attributes_sha256: str
    model_candidates_sha256: str
    sampler_source_sha256: str
    _content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        coordinates = _readonly(self.coordinates_bxyzt, np.float32, 2)
        grid = _readonly(self.grid_coordinates_xyz, np.int32, 2)
        features = _readonly(self.features, np.float32, 2)
        offsets = _integer_vector(
            self.sparse_batch_offsets, np.int64, "sparse batch offsets"
        )
        segments = _integer_vector(self.point2segment, np.int64, "point2segment")
        mapping = _integer_vector(
            self.adapter_to_model, np.int64, "adapter-to-model mapping"
        )
        visits = _integer_vector(self.model_visit_ids, np.int8, "model visit IDs")
        representatives = _integer_vector(
            self.representative_source_point_indices,
            np.int64,
            "model representative source indices",
        )
        local_frames = _integer_vector(self.local_frame_indices, np.int64, "local frames")
        global_frames = _integer_vector(self.global_frame_indices, np.int64, "global frames")
        rows = _integer_vector(self.rows, np.int64, "support rows")
        columns = _integer_vector(self.columns, np.int64, "support columns")
        camera = _readonly(self.camera_depth_m, np.float32, 1)
        observed = _readonly(self.observed_depth_m, np.float32, 1)
        residual = _readonly(self.depth_residual_m, np.float32, 1)
        model_count = len(visits)
        if coordinates.shape != (model_count, 5) or grid.shape != (model_count, 3) or features.shape != (model_count, 9):
            raise BridgeError("model coordinates, grid, and features have invalid shapes")
        if any(
            len(value) != model_count
            for value in (
                segments,
                representatives,
                local_frames,
                global_frames,
                rows,
                columns,
                camera,
                observed,
                residual,
            )
        ):
            raise BridgeError("model support arrays must have M rows")
        if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(features)):
            raise BridgeError("model input must be finite")
        if not np.array_equal(coordinates[:, 0], np.zeros(model_count, dtype=np.float32)):
            raise BridgeError("true sequence batch must remain zero")
        if not np.array_equal(coordinates[:, 4], visits.astype(np.float32)):
            raise BridgeError("model temporal coordinate must equal visit ID")
        if not np.allclose(features[:, :3], coordinates[:, 1:4], atol=0.0, rtol=0.0):
            raise BridgeError("feature XYZ must equal shared-centered source XYZ")
        if np.any(features[:, 3:6] < 0.0) or np.any(features[:, 3:6] > 1.0):
            raise BridgeError("model RGB must remain in [0, 1]")
        if not np.allclose(np.linalg.norm(features[:, 6:9], axis=1), 1.0, atol=1e-5):
            raise BridgeError("every model normal must be unit length")
        if offsets.shape != (2,) or offsets[1] != model_count or not 0 < offsets[0] < model_count:
            raise BridgeError("sparse batch offsets must split both temporal stages")
        if not np.array_equal(segments, np.arange(model_count)):
            raise BridgeError("point2segment must be one global identity mapping")
        if np.any(mapping < 0) or np.any(mapping >= model_count):
            raise BridgeError("adapter-to-model mapping is outside model range")
        if not np.array_equal(np.unique(mapping), np.arange(model_count)):
            raise BridgeError("adapter-to-model mapping must cover every model row")
        if np.any(representatives < 0) or any(
            np.any(value < 0)
            for value in (local_frames, global_frames, rows, columns)
        ):
            raise BridgeError("complete model input cannot contain unsupported indices")
        if any(not np.all(np.isfinite(value)) for value in (camera, observed, residual)):
            raise BridgeError("complete model support depths must be finite")
        if np.any(camera <= 0.0) or np.any(observed <= 0.0) or np.any(residual < 0.0):
            raise BridgeError("model support depth values are invalid")
        voxel = float(self.neural_voxel_size_m)
        if not math.isfinite(voxel) or voxel <= 0.0:
            raise BridgeError("model voxel size must be finite and positive")
        object.__setattr__(self, "coordinates_bxyzt", coordinates)
        object.__setattr__(self, "grid_coordinates_xyz", grid)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "sparse_batch_offsets", offsets)
        object.__setattr__(self, "point2segment", segments)
        object.__setattr__(self, "adapter_to_model", mapping)
        object.__setattr__(self, "model_visit_ids", visits)
        object.__setattr__(self, "representative_source_point_indices", representatives)
        object.__setattr__(self, "local_frame_indices", local_frames)
        object.__setattr__(self, "global_frame_indices", global_frames)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "camera_depth_m", camera)
        object.__setattr__(self, "observed_depth_m", observed)
        object.__setattr__(self, "depth_residual_m", residual)
        object.__setattr__(self, "neural_voxel_size_m", voxel)
        for name in (
            "adapter_geometry_sha256",
            "native_sampling_sha256",
            "surface_attributes_sha256",
            "model_candidates_sha256",
            "sampler_source_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "_content_sha256",
            _content_digest(
                {
                    "arrays": {
                        name: _array_record(getattr(self, name))
                        for name in _MODEL_ARRAY_KEYS
                    },
                    "neural_voxel_size_m": self.neural_voxel_size_m,
                    "adapter_geometry_sha256": self.adapter_geometry_sha256,
                    "native_sampling_sha256": self.native_sampling_sha256,
                    "surface_attributes_sha256": self.surface_attributes_sha256,
                    "model_candidates_sha256": self.model_candidates_sha256,
                    "sampler_source_sha256": self.sampler_source_sha256,
                }
            ),
        )

    def content_sha256(self) -> str:
        return self._content_sha256


@dataclass(frozen=True, slots=True)
class ModelInputArtifactPaths:
    root: Path
    manifest: Path
    arrays: Path


def _quantized(points: np.ndarray, voxel_size_m: float, entity_id: str) -> np.ndarray:
    values = np.floor(np.asarray(points, dtype=np.float64) / voxel_size_m)
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > np.iinfo(np.int64).max):
        raise BridgeError(f"OVI entity {entity_id} points exceed quantization range")
    return values.astype(np.int64)


def build_adapter_geometry(t0: VisitMap, t1: VisitMap, voxel_size_m: float) -> AdapterGeometry:
    """Build V1-equivalent A geometry without requiring neural attributes."""

    voxel = float(voxel_size_m)
    if not math.isfinite(voxel) or voxel <= 0.0:
        raise BridgeError("neural voxel size must be finite and positive")
    if t0.map_voxel_size_m != 0.01 or t1.map_voxel_size_m != 0.01:
        raise BridgeError("OVI mapping voxel must remain frozen at 0.01 m")
    validate_visit_pair(t0, t1)
    before = (snapshot_content_sha256(t0.snapshot), snapshot_content_sha256(t1.snapshot))

    entity_keys: list[tuple[int, str]] = []
    semantics: list[OviEntitySemanticEvidence] = []
    token_grids: list[np.ndarray] = []
    token_centroids: list[np.ndarray] = []
    token_visits: list[np.ndarray] = []
    token_entities: list[np.ndarray] = []
    token_counts: list[np.ndarray] = []
    contributor_globals: list[np.ndarray] = []
    contributor_locals: list[np.ndarray] = []
    global_start = 0
    for visit in (t0, t1):
        for entity in sorted(visit.snapshot.entities, key=lambda item: item.entity_id):
            points = np.asarray(entity.points_xyz, dtype=np.float64)
            if not len(points):
                continue
            entity_index = len(entity_keys)
            entity_keys.append((visit.visit_id, entity.entity_id))
            semantics.append(
                OviEntitySemanticEvidence(
                    visit_id=visit.visit_id,
                    entity_id=entity.entity_id,
                    semantic_label=entity.semantic_label,
                    semantic_score=entity.semantic_score,
                    semantic_embedding=entity.semantic_embedding,
                )
            )
            grid = _quantized(points, voxel, entity.entity_id)
            local = np.arange(len(points), dtype=np.int64)
            order = np.lexsort((local, grid[:, 2], grid[:, 1], grid[:, 0]))
            sorted_grid = grid[order]
            starts = np.concatenate(
                (
                    np.asarray([0], dtype=np.int64),
                    np.flatnonzero(np.any(np.diff(sorted_grid, axis=0) != 0, axis=1)) + 1,
                )
            )
            counts = np.diff(np.concatenate((starts, [len(order)]))).astype(np.int64)
            centroids = np.add.reduceat(points[order], starts, axis=0) / counts[:, None]
            token_grids.append(sorted_grid[starts])
            token_centroids.append(centroids)
            token_visits.append(np.full(len(starts), visit.visit_id, dtype=np.int8))
            token_entities.append(np.full(len(starts), entity_index, dtype=np.int32))
            token_counts.append(counts)
            contributor_globals.append(global_start + order)
            contributor_locals.append(order)
            global_start += len(points)
    if not token_grids:
        raise BridgeError("OVI pair contains no entity surface points")

    grids = np.concatenate(token_grids, axis=0)
    centroids = np.concatenate(token_centroids, axis=0)
    visits = np.concatenate(token_visits)
    entities = np.concatenate(token_entities)
    counts = np.concatenate(token_counts)
    temp_source = np.concatenate(contributor_globals)
    temp_local = np.concatenate(contributor_locals)
    temp_offsets = np.concatenate(([0], np.cumsum(counts)))
    token_order = np.lexsort((entities, grids[:, 2], grids[:, 1], grids[:, 0], visits))
    ordered_counts = counts[token_order]
    source_count = int(ordered_counts.sum())
    source_points = np.empty(source_count, dtype=np.int64)
    source_local = np.empty(source_count, dtype=np.int64)
    source_visits = np.empty(source_count, dtype=np.int8)
    source_entities = np.empty(source_count, dtype=np.int32)
    output_offsets = np.concatenate(([0], np.cumsum(ordered_counts)))
    for output_index, temp_index in enumerate(token_order):
        temp_start = int(temp_offsets[temp_index])
        temp_end = int(temp_offsets[temp_index + 1])
        output_start = int(output_offsets[output_index])
        output_end = int(output_offsets[output_index + 1])
        source_points[output_start:output_end] = temp_source[temp_start:temp_end]
        source_local[output_start:output_end] = temp_local[temp_start:temp_end]
        source_visits[output_start:output_end] = visits[temp_index]
        source_entities[output_start:output_end] = entities[temp_index]
    ordered_visits = visits[token_order]
    ordered_centroids = centroids[token_order]
    coordinates = np.column_stack((ordered_centroids, ordered_visits)).astype(np.float64)
    minimum = ordered_centroids.min(axis=0)
    maximum = ordered_centroids.max(axis=0)
    center = np.asarray(
        [(minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, minimum[2]],
        dtype=np.float64,
    )
    result = AdapterGeometry(
        coordinates_xyzt=coordinates,
        visit_ids=ordered_visits,
        source_visit_ids=source_visits,
        source_entity_indices=source_entities,
        source_entity_local_indices=source_local,
        source_point_indices=source_points,
        source_to_adapter_offsets=output_offsets,
        entity_keys=tuple(entity_keys),
        entity_semantics=tuple(semantics),
        neural_voxel_size_m=voxel,
        coordinate_frame_id=t0.coordinate_frame_id,
        source_manifest_sha256=t0.source_manifest_sha256,
        source_visit_map_sha256=(t0.snapshot_sha256, t1.snapshot_sha256),
        shared_center_xyz=center,
    )
    after = (snapshot_content_sha256(t0.snapshot), snapshot_content_sha256(t1.snapshot))
    if before != after or before != (t0.snapshot_sha256, t1.snapshot_sha256):
        raise BridgeError("OVI input snapshot changed during geometry adaptation")
    return result


def bind_surface_attributes(
    geometry: AdapterGeometry,
    t0: VisitMap,
    t1: VisitMap,
    groups: Mapping[tuple[int, str], SurfaceGroup],
) -> SurfaceAttributeBundle:
    """Bind exact PLY rows to the global D indexing used by adapter geometry."""

    if not isinstance(geometry, AdapterGeometry):
        raise TypeError("geometry must be AdapterGeometry")
    validate_visit_pair(t0, t1)
    entities = {
        (visit.visit_id, entity.entity_id): entity
        for visit in (t0, t1)
        for entity in visit.snapshot.entities
        if len(entity.points_xyz)
    }
    if set(groups) != set(geometry.entity_keys) or set(entities) != set(geometry.entity_keys):
        raise BridgeError("surface groups must exactly cover adapter entities")
    count = geometry.source_point_count
    points = np.empty((count, 3), dtype=np.float32)
    normals = np.empty((count, 3), dtype=np.float32)
    valid = np.empty(count, dtype=np.bool_)
    original = np.empty(count, dtype=np.int64)
    visits_by_d = np.empty(count, dtype=np.int8)
    entities_by_d = np.empty(count, dtype=np.int32)
    start = 0
    for entity_index, key in enumerate(geometry.entity_keys):
        entity = entities[key]
        group = groups[key]
        try:
            validate_surface_group(entity.points_xyz, group)
        except ValueError as error:
            raise BridgeError(str(error)) from error
        end = start + len(group.points_xyz)
        points[start:end] = group.points_xyz
        normals[start:end] = group.normals_xyz
        valid[start:end] = group.normal_valid
        original[start:end] = group.original_vertex_indices
        visits_by_d[start:end] = key[0]
        entities_by_d[start:end] = entity_index
        start = end
    if start != count:
        raise BridgeError("surface group point count differs from adapter source count")
    observed_visits = np.empty(count, dtype=np.int8)
    observed_entities = np.empty(count, dtype=np.int32)
    observed_visits[geometry.source_point_indices] = geometry.source_visit_ids
    observed_entities[geometry.source_point_indices] = geometry.source_entity_indices
    if not np.array_equal(observed_visits, visits_by_d) or not np.array_equal(
        observed_entities, entities_by_d
    ):
        raise BridgeError("surface rows disagree with adapter D indexing")
    return SurfaceAttributeBundle(
        points_xyz=points,
        normals_xyz=normals,
        normal_valid=valid,
        original_vertex_indices=original,
        source_visit_ids=visits_by_d,
        source_entity_indices=entities_by_d,
        adapter_geometry_sha256=geometry.content_sha256(),
    )


def _adapter_native_grid(geometry: AdapterGeometry, sampling: NativeSamplingMap) -> np.ndarray:
    centered = geometry.coordinates_xyzt[:, :3] - geometry.shared_center_xyz
    grid = np.floor(centered / sampling.voxel_size_m).astype(np.int64)
    return grid - sampling.visit_grid_origins[geometry.visit_ids]


def validate_native_sampling(
    geometry: AdapterGeometry, sampling: NativeSamplingMap
) -> NativeSamplingMap:
    if not isinstance(geometry, AdapterGeometry) or not isinstance(sampling, NativeSamplingMap):
        raise TypeError("geometry and sampling contracts are required")
    if sampling.adapter_count != geometry.adapter_count:
        raise BridgeError("native mapping adapter count does not match A")
    if sampling.voxel_size_m != geometry.neural_voxel_size_m:
        raise BridgeError("native and adapter voxel sizes differ")
    if not np.array_equal(
        geometry.visit_ids,
        sampling.model_visit_ids[sampling.adapter_to_model],
    ):
        raise BridgeError("adapter-to-model mapping crosses visits")
    if not np.array_equal(
        geometry.visit_ids[sampling.selected_adapter_indices], sampling.model_visit_ids
    ):
        raise BridgeError("selected adapter visit does not match model row")
    grid = _adapter_native_grid(geometry, sampling)
    if not np.array_equal(
        grid,
        sampling.model_grid_coordinates[sampling.adapter_to_model],
    ):
        raise BridgeError("native grid coordinates disagree with adapter inverse")
    keys = np.column_stack((sampling.model_visit_ids, sampling.model_grid_coordinates))
    if len(np.unique(keys, axis=0)) != sampling.model_count:
        raise BridgeError("native model grid rows must be unique within each visit")
    return sampling


def select_model_candidates(
    geometry: AdapterGeometry,
    sampling: NativeSamplingMap,
    surface: SurfaceAttributeBundle,
    *,
    maximum_candidates: int = 8,
) -> ModelCandidateMap:
    """Select bounded real D candidates inside each measured native M voxel."""

    validate_native_sampling(geometry, sampling)
    if surface.adapter_geometry_sha256 != geometry.content_sha256():
        raise BridgeError("surface attributes bind a different adapter geometry")
    if type(maximum_candidates) is not int or maximum_candidates <= 0:
        raise BridgeError("maximum candidates must be a positive integer")
    if len(surface.points_xyz) != geometry.source_point_count:
        raise BridgeError("surface and adapter source domains differ")
    model_count = sampling.model_count
    output = np.full((model_count, maximum_candidates), -1, dtype=np.int64)
    output_counts = np.zeros(model_count, dtype=np.int16)
    adapter_order = np.argsort(sampling.adapter_to_model, kind="stable")
    adapter_counts = np.bincount(sampling.adapter_to_model, minlength=model_count)
    adapter_offsets = np.concatenate(([0], np.cumsum(adapter_counts)))
    for model_index in range(model_count):
        contributors = adapter_order[
            adapter_offsets[model_index] : adapter_offsets[model_index + 1]
        ]
        selected = int(sampling.selected_adapter_indices[model_index])
        contributors = np.concatenate(
            (
                np.asarray([selected], dtype=np.int64),
                contributors[contributors != selected],
            )
        )
        chosen: list[int] = []
        chosen_set: set[int] = set()
        visit = int(sampling.model_visit_ids[model_index])
        target_grid = sampling.model_grid_coordinates[model_index]
        for adapter_index in contributors:
            start = int(geometry.source_to_adapter_offsets[adapter_index])
            end = int(geometry.source_to_adapter_offsets[adapter_index + 1])
            source = geometry.source_point_indices[start:end]
            source = source[surface.normal_valid[source]]
            if len(source) == 0:
                continue
            source_grid = np.floor(
                (surface.points_xyz[source] - geometry.shared_center_xyz)
                / sampling.voxel_size_m
            ).astype(np.int64) - sampling.visit_grid_origins[visit]
            source = source[np.all(source_grid == target_grid, axis=1)]
            if len(source) == 0:
                continue
            distance = np.sum(
                (
                    surface.points_xyz[source].astype(np.float64)
                    - geometry.coordinates_xyzt[adapter_index, :3]
                )
                ** 2,
                axis=1,
            )
            order = np.lexsort(
                (source, surface.original_vertex_indices[source], distance)
            )
            for source_index in source[order]:
                value = int(source_index)
                if value in chosen_set:
                    continue
                chosen.append(value)
                chosen_set.add(value)
                if len(chosen) == maximum_candidates:
                    break
            if len(chosen) == maximum_candidates:
                break
        output_counts[model_index] = len(chosen)
        output[model_index, : len(chosen)] = chosen
    return ModelCandidateMap(output, output_counts, maximum_candidates)


def _source_to_model(
    geometry: AdapterGeometry, sampling: NativeSamplingMap
) -> np.ndarray:
    counts = np.diff(geometry.source_to_adapter_offsets)
    model_by_contributor = np.repeat(sampling.adapter_to_model, counts)
    result = np.empty(geometry.source_point_count, dtype=np.int64)
    result[geometry.source_point_indices] = model_by_contributor
    return result


def build_model_input(
    geometry: AdapterGeometry,
    sampling: NativeSamplingMap,
    surface: SurfaceAttributeBundle,
    candidates: ModelCandidateMap,
    recovered: RecoveredModelSupport,
) -> ReSceneModelInput:
    """Build complete M tensors only when every row has legal source support."""

    validate_native_sampling(geometry, sampling)
    if surface.adapter_geometry_sha256 != geometry.content_sha256():
        raise BridgeError("surface attributes bind a different adapter geometry")
    if len(recovered.support_valid) != sampling.model_count:
        raise BridgeError("recovered support model count differs from native sampling")
    if len(candidates.candidate_counts) != sampling.model_count:
        raise BridgeError("frozen candidate map model count differs from native sampling")
    if not np.all(recovered.support_valid):
        raise BridgeError("every model token must have source RGB and normal support")
    representatives = recovered.representative_source_point_indices
    if np.any(representatives >= geometry.source_point_count):
        raise BridgeError("representative source point is outside D")
    source_to_model = _source_to_model(geometry, sampling)
    if not np.array_equal(source_to_model[representatives], np.arange(sampling.model_count)):
        raise BridgeError("representative source point does not contribute to its model row")
    for model_index, representative in enumerate(representatives):
        count = int(candidates.candidate_counts[model_index])
        if representative not in candidates.source_point_indices[model_index, :count]:
            raise BridgeError("model representative is outside its frozen candidate list")
    if not np.all(surface.normal_valid[representatives]):
        raise BridgeError("model representative lacks a valid source normal")
    centered = (
        surface.points_xyz[representatives].astype(np.float64)
        - geometry.shared_center_xyz
    )
    representative_grid = np.floor(centered / sampling.voxel_size_m).astype(
        np.int64
    ) - sampling.visit_grid_origins[sampling.model_visit_ids]
    if not np.array_equal(representative_grid, sampling.model_grid_coordinates):
        raise BridgeError("model representative does not occupy its native grid")
    rgb = recovered.rgb_uint8.astype(np.float32) / np.float32(255.0)
    normals = surface.normals_xyz[representatives]
    features = np.concatenate((centered.astype(np.float32), rgb, normals), axis=1)
    coordinates = np.column_stack(
        (
            np.zeros(sampling.model_count, dtype=np.float32),
            centered.astype(np.float32),
            sampling.model_visit_ids.astype(np.float32),
        )
    )
    return ReSceneModelInput(
        coordinates_bxyzt=coordinates,
        grid_coordinates_xyz=sampling.model_grid_coordinates,
        features=features,
        sparse_batch_offsets=sampling.visit_model_offsets[1:],
        point2segment=np.arange(sampling.model_count, dtype=np.int64),
        adapter_to_model=sampling.adapter_to_model,
        model_visit_ids=sampling.model_visit_ids,
        representative_source_point_indices=representatives,
        local_frame_indices=recovered.local_frame_indices,
        global_frame_indices=recovered.global_frame_indices,
        rows=recovered.rows,
        columns=recovered.columns,
        camera_depth_m=recovered.camera_depth_m,
        observed_depth_m=recovered.observed_depth_m,
        depth_residual_m=recovered.depth_residual_m,
        neural_voxel_size_m=sampling.voxel_size_m,
        adapter_geometry_sha256=geometry.content_sha256(),
        native_sampling_sha256=sampling.content_sha256(),
        surface_attributes_sha256=surface.content_sha256(),
        model_candidates_sha256=candidates.content_sha256(),
        sampler_source_sha256=sampling.sampler_source_sha256,
    )


def materialize_neural_sample_map(
    geometry: AdapterGeometry, model_input: ReSceneModelInput
) -> NeuralSampleMap:
    """Materialize the existing A-domain public contract from M support."""

    if model_input.adapter_geometry_sha256 != geometry.content_sha256():
        raise BridgeError("model input binds a different adapter geometry")
    features = model_input.features[:, 3:9][model_input.adapter_to_model]
    return NeuralSampleMap(
        coordinates_xyzt=geometry.coordinates_xyzt,
        features=features,
        visit_ids=geometry.visit_ids,
        source_visit_ids=geometry.source_visit_ids,
        source_entity_ids=geometry.source_entity_ids,
        source_point_indices=geometry.source_point_indices,
        source_to_token_offsets=geometry.source_to_adapter_offsets,
        neural_voxel_size_m=geometry.neural_voxel_size_m,
        feature_schema="rgb_normals",
        coordinate_frame_id=geometry.coordinate_frame_id,
        source_manifest_sha256=geometry.source_manifest_sha256,
        source_visit_map_sha256=geometry.source_visit_map_sha256,
        entity_semantics=geometry.entity_semantics,
    )


def expand_model_predictions(
    values_qm: np.ndarray, adapter_to_model: np.ndarray
) -> np.ndarray:
    """Expand Q x M values to exact Q x A order through the measured inverse."""

    values = np.asarray(values_qm)
    mapping_raw = np.asarray(adapter_to_model)
    if mapping_raw.ndim != 1 or not np.issubdtype(mapping_raw.dtype, np.integer) or np.issubdtype(
        mapping_raw.dtype, np.bool_
    ):
        raise BridgeError("adapter-to-model mapping must contain integer values")
    if values.ndim != 2 or values.shape[1] == 0:
        raise BridgeError("model predictions must have shape (Q, M)")
    mapping = mapping_raw.astype(np.int64, copy=False)
    if np.any(mapping < 0) or np.any(mapping >= values.shape[1]):
        raise BridgeError("adapter-to-model mapping is outside prediction range")
    return np.ascontiguousarray(values[:, mapping])


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _file_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def write_model_input_artifact(
    model_input: ReSceneModelInput, output_root: str | Path
) -> ModelInputArtifactPaths:
    """Atomically publish a strict M-domain model-input artifact."""

    if not isinstance(model_input, ReSceneModelInput):
        raise TypeError("model_input must be ReSceneModelInput")
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise BridgeError(f"model input artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arrays_path = staging / "arrays.npz"
        with arrays_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                **{
                    name: getattr(model_input, name)
                    for name in sorted(_MODEL_ARRAY_KEYS)
                },
            )
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "schema_version": 2,
            "artifact_id": "OVI_RESCENE_MODEL_INPUT_V2",
            "status": "PASS",
            "content_sha256": model_input.content_sha256(),
            "arrays": _file_record(arrays_path),
            "neural_voxel_size_m": model_input.neural_voxel_size_m,
            "adapter_geometry_sha256": model_input.adapter_geometry_sha256,
            "native_sampling_sha256": model_input.native_sampling_sha256,
            "surface_attributes_sha256": model_input.surface_attributes_sha256,
            "model_candidates_sha256": model_input.model_candidates_sha256,
            "sampler_source_sha256": model_input.sampler_source_sha256,
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ModelInputArtifactPaths(output, output / "manifest.json", output / "arrays.npz")


def load_model_input_artifact(output_root: str | Path) -> ReSceneModelInput:
    """Load and verify a strict M-domain model-input artifact."""

    root = Path(output_root).absolute()
    manifest_path = root / "manifest.json"
    arrays_path = root / "arrays.npz"
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, arrays_path)):
        raise BridgeError("model input artifact files are missing or symlinks")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BridgeError("model input manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != _MODEL_MANIFEST_KEYS:
        raise BridgeError("model input manifest schema is invalid")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("artifact_id") != "OVI_RESCENE_MODEL_INPUT_V2"
        or manifest.get("status") != "PASS"
    ):
        raise BridgeError("model input manifest identity is invalid")
    record = manifest.get("arrays")
    observed = _file_record(arrays_path)
    if record != observed:
        raise BridgeError("model input arrays SHA-256 or byte count mismatch")
    try:
        with np.load(io.BytesIO(arrays_path.read_bytes()), allow_pickle=False) as source:
            if set(source.files) != _MODEL_ARRAY_KEYS:
                raise BridgeError("model input array schema is invalid")
            arrays = {name: source[name] for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, BridgeError):
            raise
        raise BridgeError("model input arrays are invalid") from error
    result = ReSceneModelInput(
        **arrays,
        neural_voxel_size_m=manifest["neural_voxel_size_m"],
        adapter_geometry_sha256=manifest["adapter_geometry_sha256"],
        native_sampling_sha256=manifest["native_sampling_sha256"],
        surface_attributes_sha256=manifest["surface_attributes_sha256"],
        model_candidates_sha256=manifest["model_candidates_sha256"],
        sampler_source_sha256=manifest["sampler_source_sha256"],
    )
    if result.content_sha256() != manifest.get("content_sha256"):
        raise BridgeError("model input content SHA-256 mismatch")
    return result
