from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.oviv2.rescene_input_bridge import (
    AdapterGeometry,
    ModelCandidateMap,
    NativeSamplingMap,
    RecoveredModelSupport,
    SurfaceAttributeBundle,
)
from src.oviv2.rescene_supported_view import (
    SupportedViewError,
    build_supported_inference_view,
)
from src.oviv2.two_visit_contracts import OviEntitySemanticEvidence


def _prepared() -> SimpleNamespace:
    entity_keys = (
        (0, "t0-chair"),
        (0, "t0-table"),
        (1, "t1-chair"),
        (1, "t1-table"),
    )
    semantics = tuple(
        OviEntitySemanticEvidence(
            visit_id=visit_id,
            entity_id=entity_id,
            semantic_label="chair" if "chair" in entity_id else "table",
            semantic_score=0.9,
            semantic_embedding=np.asarray([0.25, 0.75], dtype=np.float32),
        )
        for visit_id, entity_id in entity_keys
    )
    geometry = AdapterGeometry(
        coordinates_xyzt=np.asarray(
            [
                [0.003, 0.0, 0.0, 0.0],
                [0.021, 0.0, 0.0, 0.0],
                [0.025, 0.0, 0.0, 0.0],
                [0.003, 0.0, 0.0, 1.0],
                [0.021, 0.0, 0.0, 1.0],
                [0.025, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        visit_ids=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8),
        source_visit_ids=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8),
        source_entity_indices=np.asarray([0, 0, 0, 1, 2, 2, 3, 3], dtype=np.int32),
        source_entity_local_indices=np.asarray([0, 1, 2, 0, 0, 1, 0, 1]),
        source_point_indices=np.arange(8, dtype=np.int64),
        source_to_adapter_offsets=np.asarray([0, 2, 3, 4, 6, 7, 8]),
        entity_keys=entity_keys,
        entity_semantics=semantics,
        neural_voxel_size_m=0.02,
        coordinate_frame_id="world",
        source_manifest_sha256="a" * 64,
        source_visit_map_sha256=("b" * 64, "c" * 64),
        shared_center_xyz=np.zeros(3, dtype=np.float64),
    )
    surface = SurfaceAttributeBundle(
        points_xyz=np.asarray(
            [
                [0.001, 0.0, 0.0],
                [0.005, 0.0, 0.0],
                [0.021, 0.0, 0.0],
                [0.025, 0.0, 0.0],
                [0.001, 0.0, 0.0],
                [0.005, 0.0, 0.0],
                [0.021, 0.0, 0.0],
                [0.025, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        normals_xyz=np.asarray([[0.0, 0.0, 1.0]] * 8, dtype=np.float32),
        normal_valid=np.ones(8, dtype=np.bool_),
        original_vertex_indices=np.arange(8, dtype=np.int64),
        source_visit_ids=geometry.source_visit_ids,
        source_entity_indices=geometry.source_entity_indices,
        adapter_geometry_sha256=geometry.content_sha256(),
    )
    sampling = NativeSamplingMap(
        adapter_to_model=np.asarray([0, 1, 1, 2, 3, 3], dtype=np.int64),
        selected_adapter_indices=np.asarray([0, 1, 3, 4], dtype=np.int64),
        model_grid_coordinates=np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]],
            dtype=np.int64,
        ),
        model_visit_ids=np.asarray([0, 0, 1, 1], dtype=np.int8),
        visit_model_offsets=np.asarray([0, 2, 4], dtype=np.int64),
        visit_grid_origins=np.zeros((2, 3), dtype=np.int64),
        voxel_size_m=0.02,
        sampler_seed=45,
        sampler_mode="train",
        sampler_source_sha256="d" * 64,
    )
    candidates = ModelCandidateMap(
        source_point_indices=np.asarray(
            [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64
        ),
        candidate_counts=np.asarray([2, 2, 2, 2], dtype=np.int16),
        maximum_candidates=2,
    )
    return SimpleNamespace(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
    )


def _support() -> RecoveredModelSupport:
    valid = np.asarray([True, False, True, True], dtype=np.bool_)
    return RecoveredModelSupport(
        support_valid=valid,
        representative_source_point_indices=np.asarray([0, -1, 4, 6]),
        rgb_uint8=np.asarray(
            [[10, 20, 30], [0, 0, 0], [40, 50, 60], [70, 80, 90]],
            dtype=np.uint8,
        ),
        local_frame_indices=np.asarray([2, -1, 3, 4]),
        global_frame_indices=np.asarray([12, -1, 23, 24]),
        rows=np.asarray([100, -1, 101, 102]),
        columns=np.asarray([200, -1, 201, 202]),
        camera_depth_m=np.asarray([1.0, np.nan, 2.0, 3.0], dtype=np.float32),
        observed_depth_m=np.asarray([1.01, np.nan, 2.01, 3.01], dtype=np.float32),
        depth_residual_m=np.asarray([0.01, np.nan, 0.01, 0.01], dtype=np.float32),
    )


def test_supported_view_compacts_domains_without_dereferencing_negative_rows() -> None:
    prepared = _prepared()

    view = build_supported_inference_view(prepared, _support())

    assert view.new_to_old_model_indices.tolist() == [0, 2, 3]
    assert view.old_to_new_model_indices.tolist() == [0, -1, 1, 2]
    assert view.new_to_old_adapter_indices.tolist() == [0, 3, 4, 5]
    assert view.new_to_old_source_point_indices.tolist() == [0, 1, 4, 5, 6, 7]
    assert view.model_input.adapter_to_model.tolist() == [0, 1, 2, 2]
    assert view.model_input.model_visit_ids.tolist() == [0, 1, 1]
    assert view.model_input.sparse_batch_offsets.tolist() == [1, 3]
    assert view.model_input.point2segment.tolist() == [0, 1, 2]
    assert view.model_input.features[:, 3:6].tolist() == pytest.approx(
        np.asarray([[10, 20, 30], [40, 50, 60], [70, 80, 90]]) / 255.0
    )
    assert view.pair.source_point_indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert view.pair.token_entity_ids == (
        "t0-chair",
        "t1-chair",
        "t1-table",
        "t1-table",
    )


def test_supported_view_retains_full_entity_denominators_and_zero_support() -> None:
    view = build_supported_inference_view(_prepared(), _support())
    coverage = {
        (row.visit_id, row.entity_id): row
        for row in view.entity_coverage
    }

    chair0 = coverage[(0, "t0-chair")]
    assert (
        chair0.full_adapter_token_count,
        chair0.supported_adapter_token_count,
        chair0.full_source_point_count,
        chair0.supported_source_point_count,
        chair0.full_model_token_count,
        chair0.supported_model_token_count,
    ) == (2, 1, 3, 2, 2, 1)
    table0 = coverage[(0, "t0-table")]
    assert (
        table0.full_adapter_token_count,
        table0.supported_adapter_token_count,
        table0.full_source_point_count,
        table0.supported_source_point_count,
        table0.full_model_token_count,
        table0.supported_model_token_count,
    ) == (1, 0, 1, 0, 1, 0)
    assert view.unsupported_entity_keys == ((0, "t0-table"),)


def test_supported_view_rejects_support_that_drops_one_visit() -> None:
    support = _support()
    only_t0 = np.asarray([True, False, False, False], dtype=np.bool_)
    invalid = replace(
        support,
        support_valid=only_t0,
        representative_source_point_indices=np.asarray([0, -1, -1, -1]),
        rgb_uint8=np.asarray([[10, 20, 30], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        local_frame_indices=np.asarray([2, -1, -1, -1]),
        global_frame_indices=np.asarray([12, -1, -1, -1]),
        rows=np.asarray([100, -1, -1, -1]),
        columns=np.asarray([200, -1, -1, -1]),
        camera_depth_m=np.asarray([1.0, np.nan, np.nan, np.nan], dtype=np.float32),
        observed_depth_m=np.asarray([1.01, np.nan, np.nan, np.nan], dtype=np.float32),
        depth_residual_m=np.asarray([0.01, np.nan, np.nan, np.nan], dtype=np.float32),
    )

    with pytest.raises(SupportedViewError, match="both visits"):
        build_supported_inference_view(_prepared(), invalid)


def test_supported_view_does_not_mutate_frozen_inputs() -> None:
    prepared = _prepared()
    support = _support()
    before = {
        "geometry": prepared.geometry.content_sha256(),
        "sampling": prepared.sampling.content_sha256(),
        "surface": prepared.surface.content_sha256(),
        "candidates": prepared.candidates.content_sha256(),
        "valid": support.support_valid.copy(),
    }

    build_supported_inference_view(prepared, support)

    assert prepared.geometry.content_sha256() == before["geometry"]
    assert prepared.sampling.content_sha256() == before["sampling"]
    assert prepared.surface.content_sha256() == before["surface"]
    assert prepared.candidates.content_sha256() == before["candidates"]
    assert np.array_equal(support.support_valid, before["valid"])
