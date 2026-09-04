from __future__ import annotations

import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.geometric_pair_reasoner import GeometricReasonerConfig
from src.oviv2.two_visit_b7_visibility import derive_signed_visibility_for_points
from src.oviv2.two_visit_contracts import VisitMap
from src.oviv2.two_visit_execution import (
    SignedVisibilityConfig,
    build_geometric_pair_sample,
    build_static_baseline_snapshot,
    build_visibility_baseline,
    derive_observed_point_mask,
    derive_signed_visibility,
)

SOURCE_SHA256 = "a" * 64


def _entity(entity_id: str, points: list[list[float]], label: str) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(points, dtype=np.float32),
        semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        semantic_label=label,
        semantic_score=0.9,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=1.0,
        metadata={"semantic_label_source": "method_output.feat"},
    )


def _visit(visit_id: int, points: list[list[float]]) -> VisitMap:
    start = 0 if visit_id == 0 else 10
    snapshot = MapSnapshot(
        method="OVI-MAP",
        scene_id="apartment",
        timestamp=float(start + 4),
        entities=[_entity(f"ovimap:{visit_id}", points, "Chair")],
        background_xyz=np.asarray([[4.0 + visit_id, 0.0, 1.0]], dtype=np.float32),
        scope="current",
    )
    return VisitMap(
        visit_id=visit_id,
        snapshot=snapshot,
        coordinate_frame_id="tesse_world",
        source_manifest_sha256=SOURCE_SHA256,
        map_voxel_size_m=0.01,
        observed_frame_start=start,
        observed_frame_end=start + 4,
    )


def _frame(
    frame_id: int,
    *,
    depth_m: float,
    camera_x: float = 0.0,
    camera_z: float = 0.0,
) -> Frame:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [camera_x, 0.0, camera_z]
    return Frame(
        frame_id=frame_id,
        source_frame_id=frame_id,
        rgb=np.zeros((40, 40, 3), dtype=np.uint8),
        depth=np.full((40, 40), depth_m, dtype=np.float32),
        pose=pose,
        intrinsics=CameraIntrinsics(
            fx=20.0,
            fy=20.0,
            cx=20.0,
            cy=20.0,
            width=40,
            height=40,
        ),
        timestamp=float(frame_id + 1),
    )


def test_signed_visibility_requires_repeated_distinct_visible_free_evidence() -> None:
    t0 = _visit(0, [[0.025, 0.025, 1.025]])
    frames = tuple(
        _frame(index, depth_m=2.0, camera_x=(index % 3) * 0.30) for index in range(6)
    )

    result = derive_signed_visibility(
        t0,
        frames,
        SignedVisibilityConfig(),
        source_sha256="b" * 64,
    )

    assert result.as_mapping()[(0, 0, 20)] == "visible_free"


def test_present_evidence_prevents_visible_free_suppression() -> None:
    t0 = _visit(0, [[0.025, 0.025, 1.025]])
    frames = [
        _frame(index, depth_m=2.0, camera_x=(index % 3) * 0.30) for index in range(6)
    ]
    frames.append(_frame(6, depth_m=1.025))

    result = derive_signed_visibility(
        t0,
        tuple(frames),
        SignedVisibilityConfig(),
        source_sha256="b" * 64,
    )

    assert result.as_mapping()[(0, 0, 20)] == "occupied"


def test_occluded_and_unobserved_states_remain_non_deleting() -> None:
    t0 = _visit(0, [[0.025, 0.025, 1.025]])

    occluded = derive_signed_visibility(
        t0,
        (_frame(0, depth_m=0.5),),
        SignedVisibilityConfig(),
        source_sha256="b" * 64,
    )
    unobserved = derive_signed_visibility(
        t0,
        (_frame(0, depth_m=1.0, camera_z=2.0),),
        SignedVisibilityConfig(),
        source_sha256="b" * 64,
    )

    assert occluded.as_mapping()[(0, 0, 20)] == "occluded"
    assert unobserved.as_mapping()[(0, 0, 20)] == "unobserved"


def test_transformed_points_receive_signed_visibility_at_new_voxels() -> None:
    points = np.asarray([[0.525, 0.025, 1.025]], dtype=np.float32)
    before = points.copy()
    frames = tuple(
        _frame(index, depth_m=2.0, camera_x=(index % 3) * 0.30) for index in range(6)
    )

    result = derive_signed_visibility_for_points(
        points,
        frames,
        SignedVisibilityConfig(),
        source_sha256="c" * 64,
    )

    assert result.as_mapping()[(10, 0, 20)] == "visible_free"
    assert np.array_equal(points, before)


def test_transformed_point_visibility_preserves_float64_voxel_boundary() -> None:
    points = np.asarray([[np.nextafter(0.05, 0.0), 0.025, 1.025]], dtype=np.float64)

    result = derive_signed_visibility_for_points(
        points,
        (_frame(0, depth_m=2.0),),
        SignedVisibilityConfig(),
        source_sha256="c" * 64,
    )

    assert tuple(result.voxel_keys[0]) == (0, 0, 20)


def test_point_visibility_accepts_an_empty_candidate_set() -> None:
    result = derive_signed_visibility_for_points(
        np.empty((0, 3), dtype=np.float32),
        (_frame(0, depth_m=1.0),),
        SignedVisibilityConfig(),
        source_sha256="c" * 64,
    )

    assert result.voxel_keys.shape == (0, 3)
    assert result.statuses == ()


def test_point_visibility_is_equivalent_to_the_legacy_visit_entry_point() -> None:
    t0 = _visit(0, [[0.025, 0.025, 1.025], [0.025, 0.025, 2.025]])
    frames = tuple(
        _frame(index, depth_m=1.025, camera_x=(index % 3) * 0.30) for index in range(6)
    )
    all_points = np.concatenate(
        (
            t0.snapshot.entities[0].points_xyz,
            t0.snapshot.background_xyz,
        ),
        axis=0,
    )

    legacy = derive_signed_visibility(
        t0,
        frames,
        SignedVisibilityConfig(),
        source_sha256="d" * 64,
    )
    arbitrary = derive_signed_visibility_for_points(
        all_points,
        frames,
        SignedVisibilityConfig(),
        source_sha256="d" * 64,
    )

    assert np.array_equal(arbitrary.voxel_keys, legacy.voxel_keys)
    assert arbitrary.statuses == legacy.statuses
    assert arbitrary.source_sha256 == legacy.source_sha256


def test_evaluator_observed_mask_counts_present_or_free_rays_only() -> None:
    points = np.asarray(
        [
            [0.025, 0.025, 1.025],
            [0.025, 0.025, 2.025],
            [5.025, 0.025, 1.025],
        ],
        dtype=np.float32,
    )

    observed = derive_observed_point_mask(
        points,
        (_frame(0, depth_m=1.025),),
        SignedVisibilityConfig(),
    )

    assert observed.tolist() == [True, False, False]


def test_geometric_sample_needs_no_palette_or_camera_rgb() -> None:
    t0 = _visit(0, [[0.01, 0.0, 1.0], [0.019, 0.0, 1.0]])
    t1 = _visit(1, [[0.02, 0.0, 1.0]])

    pair = build_geometric_pair_sample(t0, t1, neural_voxel_size_m=0.02)

    assert pair.feature_schema == "geometric_only"
    assert pair.features.shape[1] == 1
    assert pair.source_point_count == 3
    assert sorted(pair.source_point_indices.tolist()) == [0, 1, 2]


def test_static_baselines_preserve_exact_geometry_authority() -> None:
    t0 = _visit(0, [[0.01, 0.0, 1.0]])
    t1 = _visit(1, [[2.01, 0.0, 1.0]])

    b0 = build_static_baseline_snapshot("B0", t0, t1)
    b1 = build_static_baseline_snapshot("B1", t0, t1)
    b2 = build_static_baseline_snapshot("B2", t0, t1)

    assert sum(len(entity.points_xyz) for entity in b0.entities) == 1
    assert sum(len(entity.points_xyz) for entity in b1.entities) == 2
    assert sum(len(entity.points_xyz) for entity in b2.entities) == 1
    assert b0.entities[0].metadata["geometry_authority"] == "ovi_t0"
    assert {entity.metadata["geometry_authority"] for entity in b1.entities} == {
        "ovi_t0",
        "ovi_t1",
    }
    assert b2.entities[0].metadata["geometry_authority"] == "ovi_t1"


def test_visibility_baselines_share_geometry_and_b4_adds_relations() -> None:
    t0 = _visit(0, [[0.025, 0.025, 1.025]])
    t1 = _visit(1, [[2.025, 0.025, 1.025]])
    visibility = derive_signed_visibility(
        t0,
        tuple(
            _frame(index, depth_m=2.0, camera_x=(index % 3) * 0.30)
            for index in range(6)
        ),
        SignedVisibilityConfig(),
        source_sha256="b" * 64,
    )

    b3, b3_relations = build_visibility_baseline(
        t0,
        t1,
        visibility,
        use_geometric_pairing=False,
        geometric_config=GeometricReasonerConfig(),
    )
    b4, b4_relations = build_visibility_baseline(
        t0,
        t1,
        visibility,
        use_geometric_pairing=True,
        geometric_config=GeometricReasonerConfig(),
    )

    assert b3_relations == ()
    assert b4_relations
    assert sum(len(entity.points_xyz) for entity in b3.snapshot.entities) == 1
    assert sum(len(entity.points_xyz) for entity in b4.snapshot.entities) == 1
    assert np.array_equal(
        b3.snapshot.entities[0].points_xyz,
        b4.snapshot.entities[0].points_xyz,
    )
