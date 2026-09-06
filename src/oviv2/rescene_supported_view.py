"""Compact a partially supported OVI/ReScene input into one legal model view."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.oviv2.rescene_input_bridge import (
    AdapterGeometry,
    ModelCandidateMap,
    NativeSamplingMap,
    RecoveredModelSupport,
    ReSceneModelInput,
    SurfaceAttributeBundle,
    validate_native_sampling,
)
from src.oviv2.two_visit_contracts import NeuralSampleMap


class SupportedViewError(ValueError):
    """Raised when supported rows cannot form a legal two-visit inference view."""


class PreparedInferenceInput(Protocol):
    geometry: AdapterGeometry
    surface: SurfaceAttributeBundle
    sampling: NativeSamplingMap
    candidates: ModelCandidateMap


def _readonly_integer(
    values: object, *, dtype: np.dtype | type, name: str
) -> np.ndarray:
    raw = np.asarray(values)
    if (
        raw.ndim != 1
        or not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise SupportedViewError(f"{name} must be a one-dimensional integer array")
    result = np.array(raw, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class EntityCoverage:
    """Full-domain denominators and supported-domain numerators for one entity."""

    visit_id: int
    entity_id: str
    full_adapter_token_count: int
    supported_adapter_token_count: int
    full_source_point_count: int
    supported_source_point_count: int
    full_model_token_count: int
    supported_model_token_count: int

    def __post_init__(self) -> None:
        if type(self.visit_id) is not int or self.visit_id not in (0, 1):
            raise SupportedViewError("entity coverage visit ID must be 0 or 1")
        if not isinstance(self.entity_id, str) or not self.entity_id:
            raise SupportedViewError("entity coverage entity ID must be non-empty")
        for full_name, supported_name in (
            ("full_adapter_token_count", "supported_adapter_token_count"),
            ("full_source_point_count", "supported_source_point_count"),
            ("full_model_token_count", "supported_model_token_count"),
        ):
            full = getattr(self, full_name)
            supported = getattr(self, supported_name)
            if (
                type(full) is not int
                or type(supported) is not int
                or full < 0
                or not 0 <= supported <= full
            ):
                raise SupportedViewError("entity coverage counts are invalid")


@dataclass(frozen=True, slots=True)
class SupportedInferenceView:
    """Executable compact view plus exact mappings back to frozen D/A/M domains."""

    model_input: ReSceneModelInput
    pair: NeuralSampleMap
    new_to_old_model_indices: np.ndarray
    old_to_new_model_indices: np.ndarray
    new_to_old_adapter_indices: np.ndarray
    new_to_old_source_point_indices: np.ndarray
    entity_coverage: tuple[EntityCoverage, ...]
    full_source_point_count: int
    full_adapter_count: int
    full_model_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_input, ReSceneModelInput):
            raise TypeError("model_input must be ReSceneModelInput")
        if not isinstance(self.pair, NeuralSampleMap):
            raise TypeError("pair must be NeuralSampleMap")
        new_m = _readonly_integer(
            self.new_to_old_model_indices,
            dtype=np.int64,
            name="new-to-old model indices",
        )
        old_m = _readonly_integer(
            self.old_to_new_model_indices,
            dtype=np.int64,
            name="old-to-new model indices",
        )
        new_a = _readonly_integer(
            self.new_to_old_adapter_indices,
            dtype=np.int64,
            name="new-to-old adapter indices",
        )
        new_d = _readonly_integer(
            self.new_to_old_source_point_indices,
            dtype=np.int64,
            name="new-to-old source indices",
        )
        for name in ("full_source_point_count", "full_adapter_count", "full_model_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise SupportedViewError(f"{name} must be a positive integer")
        if (
            len(old_m) != self.full_model_count
            or len(new_m) != len(self.model_input.model_visit_ids)
            or len(new_a) != len(self.pair.visit_ids)
            or len(new_d) != self.pair.source_point_count
        ):
            raise SupportedViewError("supported view domain counts are inconsistent")
        if (
            np.any(new_m < 0)
            or np.any(new_m >= self.full_model_count)
            or not np.all(np.diff(new_m) > 0)
            or np.any(new_a < 0)
            or np.any(new_a >= self.full_adapter_count)
            or not np.all(np.diff(new_a) > 0)
            or np.any(new_d < 0)
            or np.any(new_d >= self.full_source_point_count)
            or len(np.unique(new_d)) != len(new_d)
        ):
            raise SupportedViewError("supported view reverse mappings are invalid")
        expected_old_m = np.full(self.full_model_count, -1, dtype=np.int64)
        expected_old_m[new_m] = np.arange(len(new_m), dtype=np.int64)
        if not np.array_equal(old_m, expected_old_m):
            raise SupportedViewError("old-to-new model mapping is not the exact inverse")
        coverage = tuple(self.entity_coverage)
        if not coverage or any(not isinstance(row, EntityCoverage) for row in coverage):
            raise SupportedViewError("entity coverage is missing or invalid")
        coverage_keys = tuple((row.visit_id, row.entity_id) for row in coverage)
        if coverage_keys != tuple(sorted(set(coverage_keys))):
            raise SupportedViewError("entity coverage keys must be unique and sorted")
        object.__setattr__(self, "new_to_old_model_indices", new_m)
        object.__setattr__(self, "old_to_new_model_indices", old_m)
        object.__setattr__(self, "new_to_old_adapter_indices", new_a)
        object.__setattr__(self, "new_to_old_source_point_indices", new_d)
        object.__setattr__(self, "entity_coverage", coverage)

    @property
    def unsupported_entity_keys(self) -> tuple[tuple[int, str], ...]:
        return tuple(
            (row.visit_id, row.entity_id)
            for row in self.entity_coverage
            if row.supported_adapter_token_count == 0
        )


def _prepared_components(
    prepared: PreparedInferenceInput,
) -> tuple[AdapterGeometry, SurfaceAttributeBundle, NativeSamplingMap, ModelCandidateMap]:
    try:
        geometry = prepared.geometry
        surface = prepared.surface
        sampling = prepared.sampling
        candidates = prepared.candidates
    except AttributeError as error:
        raise TypeError("prepared input lacks a required component") from error
    if (
        not isinstance(geometry, AdapterGeometry)
        or not isinstance(surface, SurfaceAttributeBundle)
        or not isinstance(sampling, NativeSamplingMap)
        or not isinstance(candidates, ModelCandidateMap)
    ):
        raise TypeError("prepared input contains an invalid component")
    return geometry, surface, sampling, candidates


def _entity_counts(
    geometry: AdapterGeometry,
    sampling: NativeSamplingMap,
    *,
    kept_adapters: np.ndarray,
    kept_contributors: np.ndarray,
    supported_models: np.ndarray,
) -> tuple[EntityCoverage, ...]:
    entity_count = len(geometry.entity_keys)
    token_entities = geometry.token_entity_indices
    full_adapter = np.bincount(token_entities, minlength=entity_count)
    supported_adapter = np.bincount(
        token_entities[kept_adapters], minlength=entity_count
    )
    full_source = np.bincount(
        geometry.source_entity_indices, minlength=entity_count
    )
    supported_source = np.bincount(
        geometry.source_entity_indices[kept_contributors], minlength=entity_count
    )
    entity_model_codes = np.unique(
        token_entities.astype(np.int64) * sampling.model_count
        + sampling.adapter_to_model
    )
    model_entities = entity_model_codes // sampling.model_count
    model_indices = entity_model_codes % sampling.model_count
    full_model = np.bincount(model_entities, minlength=entity_count)
    model_supported = np.zeros(sampling.model_count, dtype=np.bool_)
    model_supported[supported_models] = True
    supported_model = np.bincount(
        model_entities[model_supported[model_indices]],
        minlength=entity_count,
    )
    return tuple(
        EntityCoverage(
            visit_id=visit_id,
            entity_id=entity_id,
            full_adapter_token_count=int(full_adapter[index]),
            supported_adapter_token_count=int(supported_adapter[index]),
            full_source_point_count=int(full_source[index]),
            supported_source_point_count=int(supported_source[index]),
            full_model_token_count=int(full_model[index]),
            supported_model_token_count=int(supported_model[index]),
        )
        for index, (visit_id, entity_id) in enumerate(geometry.entity_keys)
    )


def build_supported_inference_view(
    prepared: PreparedInferenceInput,
    support: RecoveredModelSupport,
) -> SupportedInferenceView:
    """Remove unsupported M rows and compact every dependent domain exactly once."""

    geometry, surface, sampling, candidates = _prepared_components(prepared)
    if not isinstance(support, RecoveredModelSupport):
        raise TypeError("support must be RecoveredModelSupport")
    validate_native_sampling(geometry, sampling)
    if surface.adapter_geometry_sha256 != geometry.content_sha256():
        raise SupportedViewError("surface attributes bind a different adapter geometry")
    if len(support.support_valid) != sampling.model_count:
        raise SupportedViewError("support and sampling model counts differ")
    if len(candidates.candidate_counts) != sampling.model_count:
        raise SupportedViewError("candidate and sampling model counts differ")

    supported_models = np.flatnonzero(support.support_valid).astype(np.int64)
    supported_visits = sampling.model_visit_ids[supported_models]
    if set(supported_visits.tolist()) != {0, 1}:
        raise SupportedViewError("supported model rows must retain both visits")
    old_to_new_model = np.full(sampling.model_count, -1, dtype=np.int64)
    old_to_new_model[supported_models] = np.arange(
        len(supported_models), dtype=np.int64
    )
    valid_adapters = support.support_valid[sampling.adapter_to_model]
    kept_adapters = np.flatnonzero(valid_adapters).astype(np.int64)
    compact_adapter_to_model = old_to_new_model[
        sampling.adapter_to_model[kept_adapters]
    ]
    if np.any(compact_adapter_to_model < 0):
        raise SupportedViewError("adapter compaction retained an unsupported model row")

    contributor_counts = np.diff(geometry.source_to_adapter_offsets)
    kept_contributors = np.repeat(valid_adapters, contributor_counts)
    new_to_old_source = geometry.source_point_indices[kept_contributors]
    compact_counts = contributor_counts[kept_adapters]
    compact_offsets = np.concatenate(([0], np.cumsum(compact_counts))).astype(
        np.int64
    )

    representatives = support.representative_source_point_indices[supported_models]
    if np.any(representatives < 0) or np.any(representatives >= geometry.source_point_count):
        raise SupportedViewError("supported representative is outside the source domain")
    ranks = np.arange(candidates.maximum_candidates, dtype=np.int64)
    rows = candidates.source_point_indices[supported_models]
    valid_ranks = ranks[None, :] < candidates.candidate_counts[supported_models, None]
    if not np.all(np.any((rows == representatives[:, None]) & valid_ranks, axis=1)):
        raise SupportedViewError("supported representative is outside its candidate row")
    if not np.all(surface.normal_valid[representatives]):
        raise SupportedViewError("supported representative lacks a source normal")

    centered = (
        surface.points_xyz[representatives].astype(np.float64)
        - geometry.shared_center_xyz
    )
    representative_grid = (
        np.floor(centered / sampling.voxel_size_m).astype(np.int64)
        - sampling.visit_grid_origins[supported_visits]
    )
    if not np.array_equal(
        representative_grid, sampling.model_grid_coordinates[supported_models]
    ):
        raise SupportedViewError("supported representative occupies the wrong native grid")
    rgb = support.rgb_uint8[supported_models].astype(np.float32) / np.float32(255.0)
    features = np.concatenate(
        (centered.astype(np.float32), rgb, surface.normals_xyz[representatives]),
        axis=1,
    )
    coordinates = np.column_stack(
        (
            np.zeros(len(supported_models), dtype=np.float32),
            centered.astype(np.float32),
            supported_visits.astype(np.float32),
        )
    )
    first_visit_count = int(np.count_nonzero(supported_visits == 0))
    model_input = ReSceneModelInput(
        coordinates_bxyzt=coordinates,
        grid_coordinates_xyz=sampling.model_grid_coordinates[supported_models],
        features=features,
        sparse_batch_offsets=np.asarray(
            [first_visit_count, len(supported_models)], dtype=np.int64
        ),
        point2segment=np.arange(len(supported_models), dtype=np.int64),
        adapter_to_model=compact_adapter_to_model,
        model_visit_ids=supported_visits,
        representative_source_point_indices=representatives,
        local_frame_indices=support.local_frame_indices[supported_models],
        global_frame_indices=support.global_frame_indices[supported_models],
        rows=support.rows[supported_models],
        columns=support.columns[supported_models],
        camera_depth_m=support.camera_depth_m[supported_models],
        observed_depth_m=support.observed_depth_m[supported_models],
        depth_residual_m=support.depth_residual_m[supported_models],
        neural_voxel_size_m=sampling.voxel_size_m,
        adapter_geometry_sha256=geometry.content_sha256(),
        native_sampling_sha256=sampling.content_sha256(),
        surface_attributes_sha256=surface.content_sha256(),
        model_candidates_sha256=candidates.content_sha256(),
        sampler_source_sha256=sampling.sampler_source_sha256,
    )

    contributor_entities = geometry.source_entity_indices[kept_contributors]
    source_entity_ids = tuple(
        geometry.entity_keys[int(index)][1] for index in contributor_entities
    )
    supported_entity_indices = set(
        int(index) for index in geometry.token_entity_indices[kept_adapters]
    )
    pair = NeuralSampleMap(
        coordinates_xyzt=geometry.coordinates_xyzt[kept_adapters],
        features=model_input.features[:, 3:9][compact_adapter_to_model],
        visit_ids=geometry.visit_ids[kept_adapters],
        source_visit_ids=geometry.source_visit_ids[kept_contributors],
        source_entity_ids=source_entity_ids,
        source_point_indices=np.arange(len(new_to_old_source), dtype=np.int64),
        source_to_token_offsets=compact_offsets,
        neural_voxel_size_m=geometry.neural_voxel_size_m,
        feature_schema="rgb_normals",
        coordinate_frame_id=geometry.coordinate_frame_id,
        source_manifest_sha256=geometry.source_manifest_sha256,
        source_visit_map_sha256=geometry.source_visit_map_sha256,
        entity_semantics=tuple(
            item
            for index, item in enumerate(geometry.entity_semantics)
            if index in supported_entity_indices
        ),
    )
    coverage = _entity_counts(
        geometry,
        sampling,
        kept_adapters=kept_adapters,
        kept_contributors=kept_contributors,
        supported_models=supported_models,
    )
    return SupportedInferenceView(
        model_input=model_input,
        pair=pair,
        new_to_old_model_indices=supported_models,
        old_to_new_model_indices=old_to_new_model,
        new_to_old_adapter_indices=kept_adapters,
        new_to_old_source_point_indices=new_to_old_source,
        entity_coverage=coverage,
        full_source_point_count=geometry.source_point_count,
        full_adapter_count=geometry.adapter_count,
        full_model_count=sampling.model_count,
    )


__all__ = [
    "EntityCoverage",
    "PreparedInferenceInput",
    "SupportedInferenceView",
    "SupportedViewError",
    "build_supported_inference_view",
]
