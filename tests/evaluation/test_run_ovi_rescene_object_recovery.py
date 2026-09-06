from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.evaluation.run_ovi_rescene_object_recovery import (
    build_gt_oracle_relations,
    execute_recovery_variant,
    load_materialized_visibility_frames,
    relations_from_object_predictions,
    relations_from_temporal_grouping,
    visit_maps_from_grouping,
)
from src.core.data_structures import CameraIntrinsics, Frame
from src.evaluation.object_pair_association import ObjectPairPrediction
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
from src.evaluation.temporal_object_groups import build_temporal_object_groups
from src.oviv2.two_visit_contracts import PairRelation
from src.oviv2.two_visit_current_map import CompositionConfig
from src.oviv2.two_visit_dense_recovery import DenseRecoveryConfig
from src.oviv2.two_visit_execution import (
    SignedVisibilityConfig,
    derive_signed_visibility,
)
from src.oviv2.two_visit_registration import RegistrationConfig


def _entity(visit_id: int, point_index: int) -> OviObjectEntityView:
    return OviObjectEntityView(
        visit_id=visit_id,
        entity_id=f"ovimap:{point_index + 1}",
        source_instance_id=point_index + 1,
        point_indices=np.asarray([point_index], dtype=np.int64),
        palette_rgb=(10, 20, 30),
        semantic_embedding=None,
        semantic_label=None,
        semantic_score=0.0,
        observation_frame_ids=(0,),
        observation_boxes_xyxy=((0, 0, 0, 0),),
    )


def _visit(visit_id: int) -> OviObjectVisitView:
    points = np.asarray([[0.01, 0.01, 0.01], [1.01, 0.01, 0.01]], dtype=np.float32)
    return OviObjectVisitView(
        visit_id=visit_id,
        scan_id=f"scan-{visit_id}",
        frame_count=1,
        points_xyz=points,
        palette_rgb_uint8=np.asarray([[10, 20, 30], [30, 20, 10]], dtype=np.uint8),
        normals_xyz=np.asarray([[0.0, 0.0, 1.0]] * 2, dtype=np.float32),
        normal_valid=np.ones(2, dtype=bool),
        source_vertex_indices=np.arange(2, dtype=np.int64),
        source_frame_ids_by_target=np.asarray([0], dtype=np.int64),
        entity_owner_indices=np.arange(2, dtype=np.int64),
        entities=(_entity(visit_id, 0), _entity(visit_id, 1)),
        camera_rgb_uint8=np.asarray([[100, 110, 120]] * 2, dtype=np.uint8),
        appearance_valid=np.ones(2, dtype=bool),
        appearance_frame_ids=np.zeros(2, dtype=np.int64),
        appearance_source_frame_ids=np.zeros(2, dtype=np.int64),
        appearance_rows=np.zeros(2, dtype=np.int64),
        appearance_columns=np.zeros(2, dtype=np.int64),
        appearance_camera_depth_m=np.ones(2, dtype=np.float32),
        appearance_observed_depth_m=np.ones(2, dtype=np.float32),
        appearance_depth_residual_m=np.zeros(2, dtype=np.float32),
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
        pair_id="pair",
        visits=(_visit(0), _visit(1)),
        source_manifest_sha256="a" * 64,
        global_alignment=np.eye(4),
    )


def test_only_matched_object_predictions_become_pair_relations() -> None:
    predictions = (
        ObjectPairPrediction(
            prediction_id="g:match",
            pair_id="pair",
            method_id="G_full",
            t0_entity_id="ovimap:1",
            t1_entity_id="ovimap:2",
            score=0.8,
            state="persistent_moved",
        ),
        ObjectPairPrediction(
            prediction_id="g:unmatched",
            pair_id="pair",
            method_id="G_full",
            t0_entity_id="ovimap:2",
            t1_entity_id=None,
            score=None,
            state="unmatched_t0",
        ),
    )

    relations = relations_from_object_predictions(predictions)

    assert len(relations) == 1
    assert relations[0].t0_entity_ids == ("ovimap:1",)
    assert relations[0].t1_entity_ids == ("ovimap:2",)
    assert relations[0].identity_source == "geometric_baseline"


def test_temporal_grouping_relations_reference_composite_object_ids() -> None:
    pair = _pair()
    source = PairRelation(
        temporal_query_id="query:1",
        t0_entity_ids=("ovimap:1", "ovimap:2"),
        t1_entity_ids=("ovimap:1", "ovimap:2"),
        state="uncertain",
        query_confidence=0.9,
        evidence={"soft_mass": 3.0},
        identity_source="rescene",
    )
    grouping = build_temporal_object_groups(pair, (source,), variant_id="U3")

    relations = relations_from_temporal_grouping(
        grouping,
        identity_source="rescene",
        static_centroid_tolerance_m=0.2,
    )

    assert len(relations) == 1
    assert relations[0].t0_entity_ids == ("U3:query:query:1:t0",)
    assert relations[0].t1_entity_ids == ("U3:query:query:1:t1",)
    assert relations[0].identity_source == "rescene"


def test_gt_oracle_pairs_predicted_objects_without_substituting_gt_geometry() -> None:
    pair = _pair()
    grouping = build_temporal_object_groups(pair, (), variant_id="U0")
    gt = GroundTruthPair(
        pair_id="pair",
        voxel_size_m=0.05,
        visits=(
            (
                GroundTruthInstance(
                    1,
                    "object",
                    voxelize_points(pair.visits[0].points_xyz[:1], voxel_size_m=0.05),
                ),
            ),
            (
                GroundTruthInstance(
                    1,
                    "object",
                    voxelize_points(pair.visits[1].points_xyz[:1], voxel_size_m=0.05),
                ),
            ),
        ),
        identity_rules=IdentityRules.from_official_records(
            changes={
                "rigid": [
                    {
                        "instance_reference": 1,
                        "instance_rescan": 1,
                        "symmetry": 0,
                        "transform": np.eye(4).reshape(-1).tolist(),
                    }
                ],
                "nonrigid": [],
                "removed": [],
            },
            ambiguity=[],
        ),
    )

    relations = build_gt_oracle_relations(grouping, gt, iou_threshold=0.25)

    assert len(relations) == 1
    relation = relations[0]
    assert relation.identity_source == "ground_truth_oracle"
    assert relation.evidence["reference_instance_id"] == 1
    assert relation.t0_entity_ids == ("U0:atomic:t0:ovimap:1",)
    assert relation.t1_entity_ids == ("U0:atomic:t1:ovimap:1",)
    assert np.array_equal(
        grouping.snapshots[0].entities[0].points_xyz,
        pair.visits[0].points_xyz[:1],
    )


def _record(path: Path, root: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def test_materialized_visibility_frames_apply_global_alignment_once(
    tmp_path: Path,
) -> None:
    (tmp_path / "color").mkdir()
    (tmp_path / "depth").mkdir()
    (tmp_path / "pose").mkdir()
    (tmp_path / "intrinsic").mkdir()
    Image.fromarray(np.asarray([[[10, 20, 30]]], dtype=np.uint8)).save(
        tmp_path / "color/0.jpg"
    )
    Image.fromarray(np.asarray([[1000]], dtype=np.uint16)).save(
        tmp_path / "depth/0.png"
    )
    pose = np.eye(4)
    pose[0, 3] = 1.0
    np.savetxt(tmp_path / "pose/0.txt", pose)
    np.savetxt(tmp_path / "intrinsic/intrinsic_depth.txt", np.eye(4))
    files = tuple(
        _record(tmp_path / relative, tmp_path)
        for relative in (
            "color/0.jpg",
            "depth/0.png",
            "pose/0.txt",
            "intrinsic/intrinsic_depth.txt",
        )
    )
    manifest = {
        "artifact_id": "RSCAN_OVI_VISIT_V1",
        "frame_count": 1,
        "depth_shift": 1000.0,
        "dimensions": {
            "color": {"height": 1, "width": 1},
            "depth": {"height": 1, "width": 1},
        },
        "frame_map": [{"source_frame_id": 7, "target_frame_id": 0}],
        "output_tree": {"files": list(files)},
    }
    manifest_path = tmp_path / "materialized_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    world_from_visit = np.eye(4)
    world_from_visit[1, 3] = 2.0

    frames = load_materialized_visibility_frames(
        manifest_path,
        world_from_visit=world_from_visit,
    )

    assert len(frames) == 1
    assert frames[0].source_frame_id == 7
    np.testing.assert_allclose(frames[0].pose[:3, 3], [1.0, 2.0, 0.0])


def test_c0_recovery_variant_returns_the_visibility_baseline_without_registration() -> None:
    grouping = build_temporal_object_groups(_pair(), (), variant_id="U0")
    t0, _t1 = visit_maps_from_grouping(grouping, source_manifest_sha256="a" * 64)
    frames = (
        Frame(
            frame_id=0,
            source_frame_id=0,
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=np.zeros((4, 4), dtype=np.float32),
            pose=np.eye(4),
            intrinsics=CameraIntrinsics(1.0, 1.0, 1.5, 1.5, 4, 4),
        ),
    )
    visibility_config = SignedVisibilityConfig(
        minimum_absent_observations=1,
        minimum_distinct_viewpoints=1,
    )
    visibility = derive_signed_visibility(
        t0,
        frames,
        visibility_config,
        source_sha256="b" * 64,
    )

    result = execute_recovery_variant(
        "C0",
        grouping,
        (),
        frames=frames,
        baseline_visibility=visibility,
        source_manifest_sha256="a" * 64,
        visibility_config=visibility_config,
        registration_config=RegistrationConfig(),
        composition_config=CompositionConfig(),
        recovery_config=DenseRecoveryConfig(),
        oracle_sources={},
    )

    assert result.variant_id == "C0"
    assert result.registrations == ()
    assert result.recovery.snapshot is result.baseline.snapshot
    assert result.recovery.recovered_point_count == 0
