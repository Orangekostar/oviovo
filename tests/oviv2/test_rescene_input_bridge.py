from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.ovi_rescene_adapter import AdapterConfig, adapt_visit_pair
from src.oviv2.ovi_surface_attributes import SurfaceGroup
from src.oviv2.rescene_input_bridge import (
    BridgeError,
    ModelCandidateMap,
    NativeSamplingMap,
    RecoveredModelSupport,
    bind_surface_attributes,
    build_adapter_geometry,
    build_model_input,
    expand_model_predictions,
    load_model_input_artifact,
    materialize_neural_sample_map,
    select_model_candidates,
    validate_native_sampling,
    write_model_input_artifact,
)
from src.oviv2.two_visit_contracts import VisitMap, snapshot_content_sha256

SOURCE_SHA256 = "a" * 64


def _entity(
    entity_id: str,
    points: list[list[float]],
    colors: list[list[int]],
    normals: list[list[float]] | None = None,
) -> EntityPrediction:
    if normals is None:
        normals = [[0.0, 0.0, 1.0] for _ in points]
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(points, dtype=np.float32),
        semantic_embedding=np.asarray([0.25, 0.75], dtype=np.float32),
        semantic_label="chair",
        semantic_score=0.9,
        lifecycle_state="observed",
        first_seen=0.0,
        last_seen=1.0,
        metadata={
            "point_rgb": np.asarray(colors, dtype=np.uint8),
            "point_rgb_source": "camera_rgb",
            "point_normals": np.asarray(normals, dtype=np.float32),
            "point_normals_source": "source_geometry",
        },
    )


def _visit(visit_id: int) -> VisitMap:
    if visit_id == 0:
        entities = [
            _entity(
                "ovi:t0:chair",
                [[0.001, 0.001, 0.001], [0.019, 0.001, 0.001]],
                [[255, 0, 0], [127, 0, 0]],
            ),
            _entity(
                "ovi:t0:table",
                [[0.001, 0.001, 0.001]],
                [[0, 0, 255]],
                [[1.0, 0.0, 0.0]],
            ),
        ]
        start = 0
    else:
        entities = [
            _entity(
                "ovi:t1:chair",
                [[0.021, 0.001, 0.001], [0.039, 0.001, 0.001]],
                [[0, 255, 0], [0, 127, 0]],
                [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            )
        ]
        start = 10
    return VisitMap(
        visit_id=visit_id,
        snapshot=MapSnapshot(
            method="OVI-MAP",
            scene_id="apartment",
            timestamp=float(start),
            entities=entities,
            background_xyz=None,
            scope="current",
        ),
        coordinate_frame_id="world",
        source_manifest_sha256=SOURCE_SHA256,
        map_voxel_size_m=0.01,
        observed_frame_start=start,
        observed_frame_end=start + 4,
    )


def _surface_groups() -> dict[tuple[int, str], SurfaceGroup]:
    return {
        (0, "ovi:t0:chair"): SurfaceGroup(
            points_xyz=np.asarray(
                [[0.001, 0.001, 0.001], [0.019, 0.001, 0.001]], dtype=np.float32
            ),
            normals_xyz=np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.float32),
            normal_valid=np.asarray([True, True]),
            original_vertex_indices=np.asarray([4, 8]),
            palette_rgb=(1, 2, 3),
        ),
        (0, "ovi:t0:table"): SurfaceGroup(
            points_xyz=np.asarray([[0.001, 0.001, 0.001]], dtype=np.float32),
            normals_xyz=np.asarray([[1, 0, 0]], dtype=np.float32),
            normal_valid=np.asarray([True]),
            original_vertex_indices=np.asarray([11]),
            palette_rgb=(4, 5, 6),
        ),
        (1, "ovi:t1:chair"): SurfaceGroup(
            points_xyz=np.asarray(
                [[0.021, 0.001, 0.001], [0.039, 0.001, 0.001]], dtype=np.float32
            ),
            normals_xyz=np.asarray([[0, 1, 0], [0, 1, 0]], dtype=np.float32),
            normal_valid=np.asarray([True, True]),
            original_vertex_indices=np.asarray([3, 9]),
            palette_rgb=(7, 8, 9),
        ),
    }


def _geometry():
    return build_adapter_geometry(_visit(0), _visit(1), 0.02)


def _one_entity_visit(visit_id: int, points: list[list[float]]) -> VisitMap:
    start = 0 if visit_id == 0 else 10
    return VisitMap(
        visit_id=visit_id,
        snapshot=MapSnapshot(
            method="OVI-MAP",
            scene_id="apartment",
            timestamp=float(start),
            entities=[
                _entity(
                    f"ovi:t{visit_id}:chair",
                    points,
                    [[1, 2, 3] for _ in points],
                )
            ],
            background_xyz=None,
            scope="current",
        ),
        coordinate_frame_id="world",
        source_manifest_sha256=SOURCE_SHA256,
        map_voxel_size_m=0.01,
        observed_frame_start=start,
        observed_frame_end=start + 4,
    )


def _sampling() -> NativeSamplingMap:
    return NativeSamplingMap(
        adapter_to_model=np.asarray([0, 0, 1], dtype=np.int64),
        selected_adapter_indices=np.asarray([0, 2], dtype=np.int64),
        model_grid_coordinates=np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.int64),
        model_visit_ids=np.asarray([0, 1], dtype=np.int8),
        visit_model_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        visit_grid_origins=np.asarray([[-1, 0, 0], [0, 0, 0]], dtype=np.int64),
        voxel_size_m=0.02,
        sampler_seed=45,
        sampler_mode="train",
        sampler_source_sha256="b" * 64,
    )


def _recovered() -> RecoveredModelSupport:
    return RecoveredModelSupport(
        support_valid=np.asarray([True, True]),
        representative_source_point_indices=np.asarray([0, 3], dtype=np.int64),
        rgb_uint8=np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        local_frame_indices=np.asarray([2, 7], dtype=np.int64),
        global_frame_indices=np.asarray([2, 17], dtype=np.int64),
        rows=np.asarray([3, 4], dtype=np.int64),
        columns=np.asarray([5, 6], dtype=np.int64),
        camera_depth_m=np.asarray([1.0, 2.0], dtype=np.float32),
        observed_depth_m=np.asarray([1.001, 2.002], dtype=np.float32),
        depth_residual_m=np.asarray([0.001, 0.002], dtype=np.float32),
    )


def _candidates(geometry=None, surface=None):
    if geometry is None:
        geometry = _geometry()
    if surface is None:
        surface = bind_surface_attributes(
            geometry, _visit(0), _visit(1), _surface_groups()
        )
    return select_model_candidates(
        geometry,
        _sampling(),
        surface,
        maximum_candidates=8,
    )


def test_vectorized_geometry_matches_v1_grouping_csr_and_semantics() -> None:
    t0, t1 = _visit(0), _visit(1)
    before = (snapshot_content_sha256(t0.snapshot), snapshot_content_sha256(t1.snapshot))
    expected = adapt_visit_pair(t0, t1, AdapterConfig(0.02, "rgb_normals"))

    geometry = build_adapter_geometry(t0, t1, 0.02)

    assert np.array_equal(geometry.coordinates_xyzt, expected.coordinates_xyzt)
    assert np.array_equal(geometry.visit_ids, expected.visit_ids)
    assert np.array_equal(geometry.source_visit_ids, expected.source_visit_ids)
    assert np.array_equal(geometry.source_point_indices, expected.source_point_indices)
    assert np.array_equal(
        geometry.source_to_adapter_offsets, expected.source_to_token_offsets
    )
    assert geometry.source_entity_ids == expected.source_entity_ids
    assert geometry.token_entity_ids == expected.token_entity_ids
    assert [
        (
            item.visit_id,
            item.entity_id,
            item.semantic_label,
            item.semantic_score,
            item.semantic_embedding.tolist(),
        )
        for item in geometry.entity_semantics
    ] == [
        (
            item.visit_id,
            item.entity_id,
            item.semantic_label,
            item.semantic_score,
            item.semantic_embedding.tolist(),
        )
        for item in expected.entity_semantics
    ]
    assert before == (
        snapshot_content_sha256(t0.snapshot),
        snapshot_content_sha256(t1.snapshot),
    ) == (t0.snapshot_sha256, t1.snapshot_sha256)


def test_adapter_centroid_is_bit_exact_to_v1_float64_mean() -> None:
    rng = np.random.default_rng(7)
    points = (rng.random((17, 3)) * 0.005).astype(np.float32)
    t0 = _one_entity_visit(0, points.tolist())
    t1 = _one_entity_visit(1, [[0.1, 0.0, 0.0]])
    expected = adapt_visit_pair(t0, t1, AdapterConfig(0.02, "rgb_normals"))

    geometry = build_adapter_geometry(t0, t1, 0.02)

    assert np.array_equal(geometry.coordinates_xyzt, expected.coordinates_xyzt)


def test_surface_bundle_binds_exact_entity_rows_without_using_metadata_rgb() -> None:
    geometry = _geometry()

    surface = bind_surface_attributes(
        geometry,
        _visit(0),
        _visit(1),
        _surface_groups(),
    )

    assert np.array_equal(surface.original_vertex_indices, [4, 8, 11, 3, 9])
    assert np.array_equal(surface.normal_valid, np.ones(5, dtype=np.bool_))
    assert np.array_equal(surface.normals_xyz[2], [1.0, 0.0, 0.0])
    assert surface.rgb_source == "not_materialized"

    bad = _surface_groups()
    chair = bad[(0, "ovi:t0:chair")]
    bad[(0, "ovi:t0:chair")] = SurfaceGroup(
        points_xyz=chair.points_xyz[::-1],
        normals_xyz=chair.normals_xyz,
        normal_valid=chair.normal_valid,
        original_vertex_indices=chair.original_vertex_indices,
        palette_rgb=chair.palette_rgb,
    )
    with pytest.raises(BridgeError, match="exactly match"):
        bind_surface_attributes(geometry, _visit(0), _visit(1), bad)

    duplicate_source_row = _surface_groups()
    table = duplicate_source_row[(0, "ovi:t0:table")]
    duplicate_source_row[(0, "ovi:t0:table")] = SurfaceGroup(
        points_xyz=table.points_xyz,
        normals_xyz=table.normals_xyz,
        normal_valid=table.normal_valid,
        original_vertex_indices=np.asarray([4]),
        palette_rgb=table.palette_rgb,
    )
    with pytest.raises(BridgeError, match="original PLY rows"):
        bind_surface_attributes(
            geometry, _visit(0), _visit(1), duplicate_source_row
        )


def test_native_mapping_allows_cross_entity_many_to_one_and_is_complete() -> None:
    geometry = _geometry()
    sampling = _sampling()

    validated = validate_native_sampling(geometry, sampling)

    assert validated is sampling
    assert sampling.adapter_count == 3
    assert sampling.model_count == 2
    assert sampling.merge_count == 1
    assert sampling.cross_entity_model_token_count(geometry) == 1


def test_native_mapping_accepts_identity_and_same_entity_many_to_one() -> None:
    geometry = build_adapter_geometry(
        _one_entity_visit(0, [[0.001, 0.0, 0.0]]),
        _one_entity_visit(1, [[0.1, 0.0, 0.0]]),
        0.02,
    )
    identity = NativeSamplingMap(
        adapter_to_model=np.arange(2, dtype=np.int64),
        selected_adapter_indices=np.arange(2, dtype=np.int64),
        model_grid_coordinates=np.zeros((2, 3), dtype=np.int64),
        model_visit_ids=np.asarray([0, 1], dtype=np.int8),
        visit_model_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        visit_grid_origins=np.asarray([[-3, 0, 0], [2, 0, 0]], dtype=np.int64),
        voxel_size_m=0.02,
        sampler_seed=45,
        sampler_mode="train",
        sampler_source_sha256="b" * 64,
    )
    assert validate_native_sampling(geometry, identity).merge_count == 0

    same_entity_geometry = build_adapter_geometry(
        _one_entity_visit(0, [[0.019, 0.0, 0.0], [0.021, 0.0, 0.0]]),
        _one_entity_visit(1, [[0.12, 0.0, 0.0]]),
        0.02,
    )
    merged = NativeSamplingMap(
        adapter_to_model=np.asarray([0, 0, 1], dtype=np.int64),
        selected_adapter_indices=np.asarray([0, 2], dtype=np.int64),
        model_grid_coordinates=np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.int64),
        model_visit_ids=np.asarray([0, 1], dtype=np.int8),
        visit_model_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        visit_grid_origins=np.asarray([[-3, 0, 0], [2, 0, 0]], dtype=np.int64),
        voxel_size_m=0.02,
        sampler_seed=45,
        sampler_mode="train",
        sampler_source_sha256="b" * 64,
    )
    assert validate_native_sampling(same_entity_geometry, merged).merge_count == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"adapter_to_model": [0, 2, 1]}, "range"),
        (
            {
                "adapter_to_model": [0, 1],
                "selected_adapter_indices": [0, 1],
            },
            "adapter count",
        ),
        (
            {
                "adapter_to_model": [0, 1, 0],
                "selected_adapter_indices": [0, 1],
            },
            "crosses visits",
        ),
        ({"model_visit_ids": [1, 0]}, "ordered"),
        ({"selected_adapter_indices": [2, 0]}, "represent"),
        (
            {
                "model_grid_coordinates": [[1, 0, 0], [0, 0, 0]],
                "visit_grid_origins": [[-2, 0, 0], [0, 0, 0]],
            },
            "zero origin",
        ),
    ],
)
def test_invalid_native_mapping_is_rejected(changes: dict[str, list[int]], message: str) -> None:
    geometry = _geometry()
    values = {
        "adapter_to_model": [0, 0, 1],
        "selected_adapter_indices": [0, 2],
        "model_grid_coordinates": [[0, 0, 0], [0, 0, 0]],
        "model_visit_ids": [0, 1],
        "visit_model_offsets": [0, 1, 2],
        "visit_grid_origins": [[-1, 0, 0], [0, 0, 0]],
    }
    values.update(changes)
    with pytest.raises(BridgeError, match=message):
        sampling = NativeSamplingMap(
            **{name: np.asarray(value) for name, value in values.items()},
            voxel_size_m=0.02,
            sampler_seed=45,
            sampler_mode="train",
            sampler_source_sha256="b" * 64,
        )
        validate_native_sampling(geometry, sampling)


def test_candidate_order_prefers_native_adapter_and_requires_same_native_grid() -> None:
    geometry = _geometry()
    surface = bind_surface_attributes(
        geometry, _visit(0), _visit(1), _surface_groups()
    )

    candidates = select_model_candidates(
        geometry,
        _sampling(),
        surface,
        maximum_candidates=8,
    )

    assert candidates.candidate_counts.tolist() == [2, 1]
    assert candidates.source_point_indices[0, :2].tolist() == [0, 2]
    assert candidates.source_point_indices[1, :1].tolist() == [3]
    assert np.all(candidates.source_point_indices[:, 2:] == -1)


def test_model_input_has_real_nine_channels_and_global_identity_segments() -> None:
    geometry = _geometry()
    sampling = _sampling()
    surface = bind_surface_attributes(
        geometry, _visit(0), _visit(1), _surface_groups()
    )

    model_input = build_model_input(
        geometry,
        sampling,
        surface,
        _candidates(geometry, surface),
        _recovered(),
    )

    assert model_input.features.shape == (2, 9)
    assert np.allclose(
        model_input.features[:, :3],
        surface.points_xyz[[0, 3]] - geometry.shared_center_xyz,
    )
    assert np.allclose(model_input.features[:, 3:6], [[10, 20, 30], [40, 50, 60]] / np.float32(255))
    assert np.array_equal(model_input.features[:, 6:], [[0, 0, 1], [0, 1, 0]])
    assert np.array_equal(model_input.coordinates_bxyzt[:, 0], [0.0, 0.0])
    assert np.array_equal(model_input.coordinates_bxyzt[:, -1], [0.0, 1.0])
    assert np.array_equal(model_input.sparse_batch_offsets, [1, 2])
    assert np.array_equal(model_input.point2segment, [0, 1])

    pair = materialize_neural_sample_map(geometry, model_input)
    assert pair.feature_schema == "rgb_normals"
    assert np.array_equal(pair.coordinates_xyzt, geometry.coordinates_xyzt)
    assert np.allclose(pair.features, model_input.features[:, 3:][sampling.adapter_to_model])


def test_model_input_rejects_missing_support_and_wrong_grid_representative() -> None:
    geometry = _geometry()
    sampling = _sampling()
    surface = bind_surface_attributes(
        geometry, _visit(0), _visit(1), _surface_groups()
    )
    recovered = _recovered()
    missing = RecoveredModelSupport(
        support_valid=np.asarray([False, True]),
        representative_source_point_indices=np.asarray([-1, 3]),
        rgb_uint8=recovered.rgb_uint8,
        local_frame_indices=np.asarray([-1, 7]),
        global_frame_indices=np.asarray([-1, 17]),
        rows=np.asarray([-1, 4]),
        columns=np.asarray([-1, 6]),
        camera_depth_m=np.asarray([np.nan, 2.0]),
        observed_depth_m=np.asarray([np.nan, 2.002]),
        depth_residual_m=np.asarray([np.nan, 0.002]),
    )
    with pytest.raises(BridgeError, match="every model token"):
        build_model_input(
            geometry,
            sampling,
            surface,
            _candidates(geometry, surface),
            missing,
        )

    wrong_grid = RecoveredModelSupport(
        support_valid=recovered.support_valid,
        representative_source_point_indices=np.asarray([1, 3]),
        rgb_uint8=recovered.rgb_uint8,
        local_frame_indices=recovered.local_frame_indices,
        global_frame_indices=recovered.global_frame_indices,
        rows=recovered.rows,
        columns=recovered.columns,
        camera_depth_m=recovered.camera_depth_m,
        observed_depth_m=recovered.observed_depth_m,
        depth_residual_m=recovered.depth_residual_m,
    )
    with pytest.raises(BridgeError, match="native grid"):
        build_model_input(
            geometry,
            sampling,
            surface,
            ModelCandidateMap(
                source_point_indices=np.asarray([[1], [3]], dtype=np.int64),
                candidate_counts=np.asarray([1, 1], dtype=np.int16),
                maximum_candidates=1,
            ),
            wrong_grid,
        )

    one_candidate = select_model_candidates(
        geometry, sampling, surface, maximum_candidates=1
    )
    outside_frozen_candidates = RecoveredModelSupport(
        support_valid=recovered.support_valid,
        representative_source_point_indices=np.asarray([2, 3]),
        rgb_uint8=recovered.rgb_uint8,
        local_frame_indices=recovered.local_frame_indices,
        global_frame_indices=recovered.global_frame_indices,
        rows=recovered.rows,
        columns=recovered.columns,
        camera_depth_m=recovered.camera_depth_m,
        observed_depth_m=recovered.observed_depth_m,
        depth_residual_m=recovered.depth_residual_m,
    )
    with pytest.raises(BridgeError, match="frozen candidate"):
        build_model_input(
            geometry,
            sampling,
            surface,
            one_candidate,
            outside_frozen_candidates,
        )


def test_prediction_expansion_is_exact_and_rejects_bad_mapping() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mapping = np.asarray([0, 0, 1], dtype=np.int64)

    expanded = expand_model_predictions(values, mapping)

    assert np.array_equal(expanded, values[:, mapping])
    with pytest.raises(BridgeError, match="range"):
        expand_model_predictions(values, np.asarray([0, 2]))
    with pytest.raises(BridgeError, match="integer"):
        expand_model_predictions(values, np.asarray([0.0, 1.0]))


def test_model_input_artifact_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    geometry = _geometry()
    sampling = _sampling()
    surface = bind_surface_attributes(
        geometry, _visit(0), _visit(1), _surface_groups()
    )
    model_input = build_model_input(
        geometry,
        sampling,
        surface,
        _candidates(geometry, surface),
        _recovered(),
    )
    output = tmp_path / "model-input"

    paths = write_model_input_artifact(model_input, output)
    restored = load_model_input_artifact(output)

    assert restored.content_sha256() == model_input.content_sha256()
    assert paths.manifest == output / "manifest.json"
    assert paths.arrays == output / "arrays.npz"
    with pytest.raises(BridgeError, match="already exists"):
        write_model_input_artifact(model_input, output)
    paths.arrays.write_bytes(paths.arrays.read_bytes() + b"tamper")
    with pytest.raises(BridgeError, match="SHA-256"):
        load_model_input_artifact(output)
