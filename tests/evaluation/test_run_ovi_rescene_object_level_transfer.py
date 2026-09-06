from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from scripts.evaluation.rescene_pair_executor import NativeForwardResult
from scripts.evaluation.run_ovi_rescene_object_level_transfer import (
    build_d1_sensor_support_view,
    build_d2_inference_bundle,
    build_shared_candidate_rows,
    build_source_bound_model_support,
    build_temporal_query_evidence,
    evaluate_shared_candidate_methods,
    execute_association_forwards,
    load_object_transfer_config,
    measure_native_candidate_representation,
    pool_independent_candidate_features,
    project_cached_relations_to_sensor_support,
    write_shared_candidate_csv,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from src.evaluation.ovi_pair_views import (
    OviObjectEntityView,
    OviObjectPairView,
    OviObjectVisitView,
)
from src.evaluation.rscan_gt_instances import (
    GroundTruthInstance,
    GroundTruthPair,
    IdentityRules,
    voxelize_points,
)
from src.evaluation.rscan_method_views import (
    ProcessedVisitView,
    RScanMethodPairView,
)
from src.oviv2.rescene_input_bridge import ModelCandidateMap


class _IdentityGridSample:
    def __init__(self, *, grid_size: float, **_options: object) -> None:
        self._grid_size = grid_size

    def __call__(self, data: dict[str, object]) -> dict[str, np.ndarray]:
        coordinates = np.asarray(data["coord"], dtype=np.float64)
        grid = np.floor(coordinates / self._grid_size).astype(np.int64)
        grid -= grid.min(axis=0)
        return {
            "adapter_index": np.asarray(data["adapter_index"], dtype=np.int64),
            "inverse": np.arange(len(coordinates), dtype=np.int64),
            "grid_coord": grid,
        }


def _visit(visit_id: int) -> OviObjectVisitView:
    offset = 2.0 * visit_id
    points = np.asarray(
        [[offset, 0.0, 1.0], [offset + 0.5, 0.0, 1.0]], dtype=np.float32
    )
    supported = np.asarray([True, visit_id == 0], dtype=bool)
    entities = tuple(
        OviObjectEntityView(
            visit_id=visit_id,
            entity_id=f"ovimap:{index + 1}",
            source_instance_id=index + 1,
            point_indices=np.asarray([index], dtype=np.int64),
            palette_rgb=(10 + index, 20 + index, 30 + index),
            semantic_embedding=np.asarray([1.0, float(index)], dtype=np.float32),
            semantic_label="object",
            semantic_score=1.0,
            observation_frame_ids=(0,),
            observation_boxes_xyxy=((index, 0, index, 0),),
        )
        for index in range(2)
    )
    residual = np.asarray([0.01, 0.02], dtype=np.float32)
    return OviObjectVisitView(
        visit_id=visit_id,
        scan_id=f"scan-{visit_id}",
        frame_count=1,
        points_xyz=points,
        palette_rgb_uint8=np.asarray([[10, 20, 30], [11, 21, 31]], dtype=np.uint8),
        normals_xyz=np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        normal_valid=np.ones(2, dtype=bool),
        source_vertex_indices=np.arange(2, dtype=np.int64),
        source_frame_ids_by_target=np.asarray([10 + visit_id], dtype=np.int64),
        entity_owner_indices=np.arange(2, dtype=np.int64),
        entities=entities,
        camera_rgb_uint8=np.asarray([[100, 110, 120], [130, 140, 150]], dtype=np.uint8),
        appearance_valid=supported,
        appearance_frame_ids=np.where(supported, 0, -1),
        appearance_source_frame_ids=np.where(supported, 10 + visit_id, -1),
        appearance_rows=np.where(supported, 4, -1),
        appearance_columns=np.where(supported, np.arange(2), -1),
        appearance_camera_depth_m=np.where(supported, 1.0, np.nan).astype(np.float32),
        appearance_observed_depth_m=np.where(
            supported, 1.0 + residual, np.nan
        ).astype(np.float32),
        appearance_depth_residual_m=np.where(supported, residual, np.nan).astype(
            np.float32
        ),
        native_manifest_sha256=str(visit_id + 1) * 64,
        materialized_manifest_sha256=str(visit_id + 3) * 64,
        source_artifact_sha256={
            "instance_color_log": "5" * 64,
            "instance_mesh": "6" * 64,
            "semantic_features": "7" * 64,
        },
        global_alignment_application=(
            "identity_reference" if visit_id == 0 else "rescan_to_reference_once"
        ),
    )


def _pair() -> OviObjectPairView:
    return OviObjectPairView(
        pair_id="scene0001_00-scene0001_01",
        visits=(_visit(0), _visit(1)),
        source_manifest_sha256="a" * 64,
        global_alignment=np.eye(4),
    )


def _native_pair() -> RScanMethodPairView:
    visits = tuple(
        ProcessedVisitView(
            visit_id=visit_id,
            scan_id=f"scan-{visit_id}",
            points_xyz=np.asarray(
                [
                    [2.0 * visit_id + 0.01, 0.0, 1.0],
                    [2.0 * visit_id + 0.51, 0.0, 1.0],
                    [4.0 + visit_id, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            rgb=np.full((3, 3), 0.5, dtype=np.float32),
            normals_xyz=np.tile(
                np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1)
            ),
            segment_ids=np.asarray([1, 2, 3], dtype=np.int64),
            source_point_indices=np.asarray([10, 20, 30], dtype=np.int64),
        )
        for visit_id in (0, 1)
    )
    return RScanMethodPairView(
        pair_id="scene0001_00-scene0001_01",
        domain_id="D0_NATIVE_PROCESSED",
        visits=visits,
        source_manifest_sha256="a" * 64,
    )


def _bundle(tmp_path: Path):
    source = tmp_path / "transform.py"
    source.write_bytes(b"pinned sampler\n")
    return build_d2_inference_bundle(
        _pair(),
        sampler_source_path=source,
        sampler_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        grid_sample_factory=lambda **options: _IdentityGridSample(**options),
    )


def _outputs(bundle):
    return execute_association_forwards(
        bundle.model_input,
        independent_forward=lambda _input, visit_id, indices: np.asarray(
            [[1.0 + visit_id, float(index + 1)] for index in range(len(indices))],
            dtype=np.float32,
        ),
        joint_forward=lambda model_input: NativeForwardResult(
            pred_masks_mq=np.ones((len(model_input.model_visit_ids), 1), dtype=np.float32),
            pred_logits_qc=np.asarray([[1.0, 0.0]], dtype=np.float32),
            runtime_s=0.1,
            peak_memory_bytes=1,
            peak_reserved_memory_bytes=1,
            rss_peak_bytes=1,
            device_name="fixture",
            model_tensor_count=796,
        ),
    )


def _ground_truth(pair: OviObjectPairView) -> GroundTruthPair:
    visits = tuple(
        tuple(
            GroundTruthInstance(
                instance_id=index + 1,
                semantic_label="object",
                voxels=voxelize_points(
                    visit.points_xyz[entity.point_indices], voxel_size_m=0.05
                ),
            )
            for index, entity in enumerate(visit.entities)
        )
        for visit in pair.visits
    )
    return GroundTruthPair(
        pair_id=pair.pair_id,
        voxel_size_m=0.05,
        visits=(visits[0], visits[1]),
        identity_rules=IdentityRules.from_official_records(
            changes={"rigid": [], "nonrigid": [], "removed": []}, ambiguity=[]
        ),
    )


def test_builds_source_bound_model_input_without_dropping_candidates(
    tmp_path: Path,
) -> None:
    pair = _pair()
    before = pair.content_sha256()

    bundle = _bundle(tmp_path)

    assert bundle.pair_content_sha256 == before
    assert bundle.candidate_ids == pair.candidate_ids
    assert bundle.sample.feature_schema == "rgb_normals"
    assert bundle.sample.source_entity_ids == (
        "ovimap:1",
        "ovimap:2",
        "ovimap:1",
    )
    np.testing.assert_allclose(
        bundle.model_input.camera_depth_m, [1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(
        bundle.model_input.observed_depth_m, [1.01, 1.02, 1.01]
    )
    assert pair.content_sha256() == before


def test_empty_model_candidate_row_becomes_explicit_unsupported_support(
    tmp_path: Path,
) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)
    candidates = ModelCandidateMap(
        source_point_indices=np.asarray([[0], [-1], [2]], dtype=np.int64),
        candidate_counts=np.asarray([1, 0, 1], dtype=np.int16),
        maximum_candidates=1,
    )

    support = build_source_bound_model_support(pair, bundle.surface, candidates)

    assert support.support_valid.tolist() == [True, False, True]
    assert support.representative_source_point_indices.tolist() == [0, -1, 2]
    assert support.local_frame_indices.tolist() == [0, -1, 0]
    assert np.isnan(support.camera_depth_m[1])
    assert np.isnan(support.observed_depth_m[1])


def test_executes_two_separate_visit_features_before_one_joint_forward(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    calls: list[object] = []

    def independent(model_input, visit_id, model_indices):
        assert model_input is bundle.model_input
        calls.append((visit_id, model_indices.tolist()))
        return np.column_stack(
            (
                np.full(len(model_indices), visit_id + 1, dtype=np.float32),
                np.arange(len(model_indices), dtype=np.float32) + 1,
            )
        )

    def joint(model_input):
        assert model_input is bundle.model_input
        calls.append("joint")
        return NativeForwardResult(
            pred_masks_mq=np.ones((len(model_input.model_visit_ids), 2), dtype=np.float32),
            pred_logits_qc=np.asarray([[2.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            runtime_s=0.5,
            peak_memory_bytes=100,
            peak_reserved_memory_bytes=200,
            rss_peak_bytes=300,
            device_name="fixture",
            model_tensor_count=796,
        )

    result = execute_association_forwards(
        bundle.model_input,
        independent_forward=independent,
        joint_forward=joint,
    )

    assert calls == [(0, [0, 1]), (1, [2]), "joint"]
    assert [value.shape for value in result.independent_model_features] == [
        (2, 2),
        (1, 2),
    ]
    assert result.visit_forward_sha256[0] != result.visit_forward_sha256[1]
    assert result.joint_forward is not None

    evidence = build_temporal_query_evidence(
        bundle,
        result,
        backend_config_sha256="d" * 64,
        checkpoint_sha256="e" * 64,
    )
    assert evidence.pair_sha256 == bundle.sample.content_sha256()
    assert evidence.temporal_query_ids == ("query_0000", "query_0001")
    assert evidence.query_masks.shape == (2, bundle.geometry.adapter_count)
    assert evidence.token_scores.shape == (2, bundle.geometry.adapter_count)
    assert evidence.query_scores.shape == (2,)


def test_pools_independent_features_and_marks_missing_entity_unsupported(
    tmp_path: Path,
) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)

    outputs = _outputs(bundle)

    bank = pool_independent_candidate_features(pair, bundle, outputs)

    assert bank.candidate_ids == pair.candidate_ids
    assert bank.valid[0].tolist() == [True, True]
    assert bank.valid[1].tolist() == [True, False]
    np.testing.assert_array_equal(bank.features[1][1], [0.0, 0.0])


def test_evaluates_all_methods_with_one_shared_supported_endpoint_binding(
    tmp_path: Path,
) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)

    evaluation = evaluate_shared_candidate_methods(
        pair,
        bundle,
        _outputs(bundle),
        _ground_truth(pair),
        minimum_match_scores={
            "G_full": 0.5,
            "G_supported": 0.5,
            "F_obj": 0.5,
            "R_obj": 0.5,
        },
        centroid_scale_m=2.0,
        static_centroid_tolerance_m=0.2,
        backend_config_sha256="d" * 64,
        checkpoint_sha256="e" * 64,
    )

    assert set(evaluation.metrics) == {
        "G_full",
        "G_supported",
        "F_obj",
        "R_obj",
    }
    supported_hashes = {
        evaluation.metrics[method]["0.50"]["binding_sha256"]
        for method in ("G_supported", "F_obj", "R_obj")
    }
    assert len(supported_hashes) == 1
    assert evaluation.metrics["G_full"]["0.50"]["support_domain"] == "full"
    for method in ("G_full", "G_supported", "F_obj", "R_obj"):
        predictions = evaluation.predictions[method]
        assert sorted(
            value.t0_entity_id
            for value in predictions
            if value.t0_entity_id is not None
        ) == ["ovimap:1", "ovimap:2"]
        assert sorted(
            value.t1_entity_id
            for value in predictions
            if value.t1_entity_id is not None
        ) == ["ovimap:1", "ovimap:2"]


def test_builds_two_threshold_rows_per_method_without_hiding_null_rates(
    tmp_path: Path,
) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)
    outputs = _outputs(bundle)
    evaluation = evaluate_shared_candidate_methods(
        pair,
        bundle,
        outputs,
        _ground_truth(pair),
        minimum_match_scores={method: 0.5 for method in (
            "G_full", "G_supported", "F_obj", "R_obj"
        )},
        centroid_scale_m=2.0,
        static_centroid_tolerance_m=0.2,
        backend_config_sha256="d" * 64,
        checkpoint_sha256="e" * 64,
    )

    rows = build_shared_candidate_rows(
        pair,
        bundle,
        outputs,
        evaluation,
        evaluated_commit="f" * 40,
        config_sha256="d" * 64,
        checkpoint_sha256="e" * 64,
    )

    assert [(row["method_id"], row["endpoint_iou_threshold"]) for row in rows] == [
        (method, threshold)
        for method in ("G_full", "G_supported", "F_obj", "R_obj")
        for threshold in (0.5, 0.25)
    ]
    for row in rows:
        assert row["pair_id"] == pair.pair_id
        assert row["t0_candidate_count"] == 2
        assert row["t1_candidate_count"] == 2
        assert row["supported_model_count"] == len(bundle.model_input.model_visit_ids)
        assert row["query_count"] == 1
        assert row["evaluated_commit"] == "f" * 40
        assert row["config_sha256"] == "d" * 64
        assert row["checkpoint_sha256"] == "e" * 64
    unsupported_rate = next(
        row
        for row in rows
        if row["method_id"] == "G_full"
        and row["endpoint_iou_threshold"] == 0.5
    )["support_conditional_recall"]
    assert unsupported_rate is None


def test_writes_measured_rows_as_lf_csv_with_blank_nulls(tmp_path: Path) -> None:
    rows = (
        {
            "pair_id": "pair-a",
            "method_id": "G_full",
            "precision": None,
            "true_positive_count": 0,
        },
    )
    output = tmp_path / "shared_candidate_gfr.csv"

    write_shared_candidate_csv(output, rows)

    assert b"\r\n" not in output.read_bytes()
    with output.open(newline="", encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))
    assert written == [
        {
            "pair_id": "pair-a",
            "method_id": "G_full",
            "precision": "",
            "true_positive_count": "0",
        }
    ]


def test_builds_d1_sensor_support_from_geometry_without_ground_truth() -> None:
    d0 = _native_pair()

    d1, audit = build_d1_sensor_support_view(
        d0,
        _pair(),
        maximum_distance_m=0.05,
    )

    assert d1.domain_id == "D1_NATIVE_SENSOR_SUPPORT"
    assert d1.parent_method_tensor_sha256 == d0.method_tensor_sha256()
    assert [visit.point_count for visit in d1.visits] == [2, 2]
    assert [visit.source_point_indices.tolist() for visit in d1.visits] == [
        [10, 20],
        [10, 20],
    ]
    assert audit == {
        "definition": "nearest_d2_dense_surface_within_distance",
        "maximum_distance_m": 0.05,
        "visit_point_counts": [3, 3],
        "visit_support_counts": [2, 2],
        "visit_support_fractions": [2 / 3, 2 / 3],
        "overall_support_fraction": 2 / 3,
        "ground_truth_used": False,
    }


def test_cached_d0_relations_are_rebased_to_d1_candidates() -> None:
    d1, _audit = build_d1_sensor_support_view(
        _native_pair(),
        _pair(),
        maximum_distance_m=0.05,
    )
    records = (
        {
            "temporal_query_id": "query_0001",
            "t0_entity_ids": ["segment:000001", "segment:000003"],
            "t1_entity_ids": ["segment:000001"],
            "state": "merge",
            "query_confidence": 0.8,
            "identity_source": "rescene",
            "evidence": {"query_score": 0.8},
        },
        {
            "temporal_query_id": "query_0002",
            "t0_entity_ids": ["segment:000003"],
            "t1_entity_ids": ["segment:000003"],
            "state": "persistent_static",
            "query_confidence": 0.7,
            "identity_source": "rescene",
            "evidence": {"query_score": 0.7},
        },
    )

    projected = project_cached_relations_to_sensor_support(
        d1,
        records,
        static_centroid_tolerance_m=0.2,
    )

    assert len(projected) == 1
    assert projected[0].temporal_query_id == "query_0001"
    assert projected[0].t0_entity_ids == ("segment:000001",)
    assert projected[0].t1_entity_ids == ("segment:000001",)
    assert projected[0].state == "persistent_moved"
    assert projected[0].evidence["cached_d0_projection"] == 1.0


def test_native_proposal_coverage_is_measured_before_association() -> None:
    d1, _audit = build_d1_sensor_support_view(
        _native_pair(),
        _pair(),
        maximum_distance_m=0.05,
    )

    measured = measure_native_candidate_representation(
        d1,
        _ground_truth(_pair()),
        iou_threshold=0.5,
    )

    assert measured == {
        "persistent_gt_count": 2,
        "represented_persistent_gt_count": 2,
        "proposal_representation_coverage": 1.0,
    }


def test_checked_in_config_freezes_runtime_and_uniform_association_thresholds() -> (
    None
):
    config = load_object_transfer_config(
        REPOSITORY_ROOT
        / "configs/evaluation/ovi_rescene_object_level_transfer_v1.json"
    )

    assert config.minimum_match_scores == {
        "G_full": 0.5,
        "G_supported": 0.5,
        "F_obj": 0.5,
        "R_obj": 0.5,
    }
    assert config.centroid_scale_m == 2.0
    assert config.static_centroid_tolerance_m == 0.2
    assert config.sampler_seed == 45
    assert config.maximum_candidates == 8
    assert config.model_python == Path(
        "/home/ww/miniconda3/envs/persist4d/bin/python"
    )
    assert config.rescene_checkout == Path(
        "/home/ww/oviovo_references/evaluation/rescene4d"
    )
    assert config.concerto_checkpoint.sha256 == (
        "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"
    )
    assert config.native_sampler_source.sha256 == (
        "cf799a0c2d4d835ee722ef6300b5c1ffc401b12a129bed783c081a6cce23c615"
    )
