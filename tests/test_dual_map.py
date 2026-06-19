"""Tests for dense/coarse dual-map functionality."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_room0_full_eval as room0_eval
from run_room0_full_eval import (
    apply_conservative_structural_overlay,
    build_output_profile,
    build_dense_surface_records,
    build_pool_debug_records,
    build_pool_semantic_records,
    build_stage_timing_summary,
    build_structural_overlay_records,
    build_tsdf_backbone_records,
    colors_for_object_labels,
    evaluate_semantics,
    labels_to_class_colors,
    project_instances_to_dense_points,
    project_structural_overlay_to_dense_points,
    render_final_object_semantic_audit,
    semantic_surface_color,
    instance_color,
    structural_overlay_direct_label_maps,
    structural_overlay_object_id,
    write_eval_artifacts,
    write_run_report,
)
from scripts.run_room0_checkpointed_eval import _update_report_sidecar
from src.core.data_structures import (
    ActiveSet,
    AssociationResult,
    AssociationScore,
    CameraIntrinsics,
    DenseSurfaceEntry,
    ObjectMap,
    ObjectState,
    ObservationRecord,
    Patch3D,
    SemanticMemory,
    SurfaceTier,
    StructuralOverlayVoxel,
    SystemState,
    TSDFInstanceVolume,
    VoxelOwnerSupport,
)
from src.modules.dense_surface import DenseSurfaceModule
from src.modules.map_tiering import MapTieringModule
from src.modules.object_update import ObjectUpdateModule
from src.modules.semantic_memory import object_export_semantic_label, preferred_object_semantic_label
from src.modules.tsdf_instance_map import TSDFInstanceMapModule


def _make_patch(points: np.ndarray, frame_id: int = 0) -> Patch3D:
    return Patch3D(
        patch_id=frame_id,
        points=points.astype(np.float32),
        centroid=points.mean(axis=0).astype(np.float32),
        bbox_min=points.min(axis=0).astype(np.float32),
        bbox_max=points.max(axis=0).astype(np.float32),
        source_frame_id=frame_id,
    )


def test_dense_surface_module_builds_capped_resident_surface() -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.01, 0.0, 1.0],
            [0.02, 0.0, 1.0],
            [0.03, 0.0, 1.0],
            [0.04, 0.0, 1.0],
            [0.05, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    patch = _make_patch(points, frame_id=3)
    obj = ObjectMap(
        object_id=1,
        local_pcd=points.copy(),
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        observations=[ObservationRecord(frame_id=3, patch=patch)],
        last_seen_frame=3,
        creation_frame=3,
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.9)]),
    )
    obj.debug["global_instance_substrate"] = {"owned_voxel_count": 0, "stability_score": 0.4}

    state = SystemState(objects={1: obj})
    owner_support_before = dict(state.tsdf_volume.owner_support)

    module = DenseSurfaceModule({"dense_surface_voxel": 0.02, "dense_surface_cap_per_object": 4})
    state = module.process(state, frame_id=3)

    entry = state.dense_surface_map.entries[1]
    assert entry.object_id == 1
    assert entry.semantic_label == "chair"
    assert entry.resident is True
    assert len(entry.points) <= 4
    assert obj.dense_surface_resident is True
    assert state.tsdf_volume.owner_support == owner_support_before


def test_map_tiering_transitions_and_cold_objects_keep_coarse_visibility() -> None:
    obj = ObjectMap(
        object_id=7,
        centroid=np.array([0.0, 0.0, 0.5], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 0.5], dtype=np.float32),
        bbox_max=np.array([0.1, 0.1, 0.6], dtype=np.float32),
        last_seen_frame=100,
        creation_frame=100,
    )
    state = SystemState(
        objects={7: obj},
        active_set=ActiveSet(visible_ids={7}),
        tsdf_volume=TSDFInstanceVolume(voxel_size=0.05),
    )
    state.tsdf_volume.owner_support[(0, 0, 10)] = VoxelOwnerSupport(support={7: 1.0})
    state.dense_surface_map.entries[7] = DenseSurfaceEntry(
        points=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
        object_id=7,
        semantic_label="lamp",
        last_refresh_frame=100,
        resident=True,
    )

    module = MapTieringModule(
        {
            "enabled": True,
            "active_radius": 0.1,
            "warm_ttl_frames": 20,
            "cold_ttl_frames": 100,
        }
    )
    camera = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    state = module.process(state, current_frame=100, camera_position=camera)
    assert obj.surface_tier == SurfaceTier.ACTIVE
    assert obj.dense_surface_resident is True

    state.active_set = ActiveSet()
    state = module.process(state, current_frame=110, camera_position=np.array([5.0, 0.0, 0.0], dtype=np.float32))
    assert obj.surface_tier == SurfaceTier.WARM
    assert obj.dense_surface_resident is True

    state = module.process(state, current_frame=130, camera_position=np.array([5.0, 0.0, 0.0], dtype=np.float32))
    assert obj.surface_tier == SurfaceTier.COLD
    assert obj.dense_surface_resident is False
    assert len(state.dense_surface_map.entries[7].points) == 0

    visible = TSDFInstanceMapModule({}).query_visible_instances(
        state.tsdf_volume,
        np.eye(4, dtype=np.float64),
        CameraIntrinsics(fx=10.0, fy=10.0, cx=5.0, cy=5.0, width=20, height=20),
    )
    assert 7 in visible


def test_dense_projection_records_use_pool_instance_map_not_dense_surface() -> None:
    obj = ObjectMap(
        object_id=4,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[0.02, 0.02, 0.52]], dtype=np.float32),
        centroid=np.array([0.02, 0.02, 0.52], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 0.5], dtype=np.float32),
        bbox_max=np.array([0.05, 0.05, 0.55], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.9)]),
    )
    obj.debug["global_instance_substrate"] = {"stability_score": 0.8}
    state = SystemState(objects={4: obj})
    state.dense_surface_map.entries[4] = DenseSurfaceEntry(
        points=np.array([[10.0, 10.0, 10.0]], dtype=np.float32),
        object_id=4,
        semantic_label="lamp",
        last_refresh_frame=1,
        resident=True,
    )

    pool_records = build_pool_semantic_records(state)
    dense_records = build_dense_surface_records(state)
    dense_points = np.array([[0.02, 0.02, 0.52]], dtype=np.float32)
    labels, _, _ = project_instances_to_dense_points(
        dense_points=dense_points,
        instance_records=pool_records,
        instance_voxel_size=0.05,
        neighbor_radius=0,
    )

    assert len(dense_records) == 1
    assert np.allclose([dense_records[0]["x"], dense_records[0]["y"], dense_records[0]["z"]], [10.0, 10.0, 10.0])
    assert np.allclose([pool_records[0]["x"], pool_records[0]["y"], pool_records[0]["z"]], [0.02, 0.02, 0.52])
    assert int(labels[0]) == 4


def test_pool_semantic_and_debug_records_use_local_pcd_not_dense_surface() -> None:
    local_points = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float32)
    dense_points = np.array([[9.0, 9.0, 9.0]], dtype=np.float32)
    obj = ObjectMap(
        object_id=5,
        state=ObjectState.DORMANT,
        local_pcd=local_points.copy(),
        centroid=local_points.mean(axis=0),
        bbox_min=local_points.min(axis=0),
        bbox_max=local_points.max(axis=0),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.95)]),
    )
    obj.debug["global_instance_substrate"] = {"stability_score": 0.6}
    state = SystemState(objects={5: obj})
    state.dense_surface_map.entries[5] = DenseSurfaceEntry(
        points=dense_points.copy(),
        object_id=5,
        semantic_label="lamp",
        last_refresh_frame=1,
        resident=True,
    )

    semantic_records = build_pool_semantic_records(state)
    debug_records = build_pool_debug_records(state)
    resident_dense_records = build_dense_surface_records(state)

    assert len(semantic_records) == len(local_points)
    assert len(debug_records) == len(local_points)
    assert len(resident_dense_records) == len(dense_points)
    assert np.allclose(
        np.stack([semantic_records["x"], semantic_records["y"], semantic_records["z"]], axis=1),
        local_points,
    )
    assert np.allclose(
        np.stack([debug_records["x"], debug_records["y"], debug_records["z"]], axis=1),
        local_points,
    )
    assert np.allclose(
        np.stack([resident_dense_records["x"], resident_dense_records["y"], resident_dense_records["z"]], axis=1),
        dense_points,
    )
    expected_color = semantic_surface_color("chair", 5)
    assert tuple(int(value) for value in semantic_records[0][["red", "green", "blue"]]) == expected_color
    assert tuple(int(value) for value in debug_records[0][["red", "green", "blue"]]) == expected_color
    assert float(semantic_records[0]["support"]) == pytest.approx(0.6)
    assert float(debug_records[0]["support"]) == pytest.approx(0.6)


def test_dense_projection_colors_use_object_semantic_labels() -> None:
    chair_a = ObjectMap(
        object_id=2,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.90)]),
    )
    chair_b = ObjectMap(
        object_id=9,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.85)]),
    )
    table = ObjectMap(
        object_id=4,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[2.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("table", 0.95)]),
    )
    state = SystemState(objects={2: chair_a, 9: chair_b, 4: table})

    colors = colors_for_object_labels(np.array([2, 9, 4, -1], dtype=np.int32), state)

    chair_color = semantic_surface_color("chair", 2)
    assert tuple(int(value) for value in colors[0]) == chair_color
    assert tuple(int(value) for value in colors[1]) == chair_color
    assert tuple(int(value) for value in colors[2]) == semantic_surface_color("table", 4)
    assert tuple(int(value) for value in colors[3]) == (180, 180, 180)
    assert tuple(int(value) for value in colors[2]) != chair_color


def test_eval_class_colors_share_semantic_palette_with_exports() -> None:
    colors = labels_to_class_colors(
        np.array([80, 42, -1], dtype=np.int32),
        class_names={80: "chair", 42: "table"},
    )

    assert tuple(int(value) for value in colors[0]) == semantic_surface_color("chair", 80)
    assert tuple(int(value) for value in colors[1]) == semantic_surface_color("table", 42)
    assert tuple(int(value) for value in colors[2]) == (80, 80, 80)


def test_structural_overlay_object_id_is_stable_negative_id() -> None:
    assert structural_overlay_object_id(17) == -10017
    assert structural_overlay_object_id(0) == -10000


def test_conservative_structural_overlay_does_not_overwrite_protected_object() -> None:
    sofa = ObjectMap(
        object_id=4,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("sofa", 0.9)]),
    )
    state = SystemState(objects={4: sofa})
    labels = np.array([4, -1], dtype=np.int32)
    state_ids = np.array([1, 0], dtype=np.uint8)
    supports = np.array([0.8, 0.0], dtype=np.float32)
    overlay_ids = np.array([-10003, -10003], dtype=np.int32)
    overlay_supports = np.array([2.0, 2.0], dtype=np.float32)

    fused_labels, fused_state_ids, fused_supports, summary = apply_conservative_structural_overlay(
        labels=labels,
        state_ids=state_ids,
        supports=supports,
        overlay_ids=overlay_ids,
        overlay_supports=overlay_supports,
        state=state,
        structure_labels={"wall"},
        protected_labels={"sofa"},
    )

    assert fused_labels.tolist() == [4, -10003]
    assert fused_state_ids.tolist() == [1, 0]
    assert fused_supports.tolist() == pytest.approx([0.8, 2.0])
    assert summary["overlay_candidate_point_count"] == 2
    assert summary["overlay_replaced_point_count"] == 1
    assert summary["overlay_protected_point_count"] == 1


def test_conservative_structural_overlay_may_replace_existing_structure_object() -> None:
    wall = ObjectMap(
        object_id=8,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("wall", 0.9)]),
    )
    state = SystemState(objects={8: wall})
    labels = np.array([8], dtype=np.int32)
    state_ids = np.array([1], dtype=np.uint8)
    supports = np.array([0.5], dtype=np.float32)
    overlay_ids = np.array([-10005], dtype=np.int32)
    overlay_supports = np.array([3.0], dtype=np.float32)

    fused_labels, _state_ids, fused_supports, summary = apply_conservative_structural_overlay(
        labels=labels,
        state_ids=state_ids,
        supports=supports,
        overlay_ids=overlay_ids,
        overlay_supports=overlay_supports,
        state=state,
        structure_labels={"wall", "window"},
        protected_labels={"chair", "sofa", "rug"},
    )

    assert fused_labels.tolist() == [-10005]
    assert fused_supports.tolist() == pytest.approx([3.0])
    assert summary["overlay_replaced_point_count"] == 1


def test_conservative_structural_overlay_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same length|shape"):
        apply_conservative_structural_overlay(
            labels=np.array([1, -1], dtype=np.int32),
            state_ids=np.array([1, 0], dtype=np.uint8),
            supports=np.array([0.5, 0.0], dtype=np.float32),
            overlay_ids=np.array([-10003], dtype=np.int32),
            overlay_supports=np.array([2.0, 2.0], dtype=np.float32),
            state=SystemState(),
            structure_labels={"wall"},
            protected_labels={"chair"},
        )


def test_colors_for_object_labels_supports_direct_negative_structure_labels() -> None:
    object_ids = np.array([-10003, -1], dtype=np.int32)
    colors = colors_for_object_labels(
        object_ids,
        SystemState(),
        direct_object_label_names={-10003: "wall"},
    )

    assert tuple(int(value) for value in colors[0]) == semantic_surface_color("wall", -10003)
    assert tuple(int(value) for value in colors[1]) == (180, 180, 180)


def test_colors_for_object_labels_supports_direct_negative_structure_labels_and_positive_objects() -> None:
    chair = ObjectMap(
        object_id=7,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.9)]),
    )
    state = SystemState(objects={7: chair})
    object_ids = np.array([-10003, 7, -1], dtype=np.int32)

    colors = colors_for_object_labels(
        object_ids,
        state,
        direct_object_label_names={-10003: "wall"},
    )

    assert tuple(int(value) for value in colors[0]) == semantic_surface_color("wall", -10003)
    assert tuple(int(value) for value in colors[1]) == semantic_surface_color("chair", 7)
    assert tuple(int(value) for value in colors[2]) == (180, 180, 180)


def test_structural_overlay_direct_label_maps_use_class_names() -> None:
    direct_ids_to_class, direct_ids_to_name = structural_overlay_direct_label_maps(
        class_names={3: "wall", 5: "window", 9: "chair"},
        structure_labels={"wall", "window"},
    )

    assert direct_ids_to_class == {-10003: 3, -10005: 5}
    assert direct_ids_to_name == {-10003: "wall", -10005: "window"}


def test_project_structural_overlay_to_dense_points_returns_synthetic_ids() -> None:
    state = SystemState()
    state.structural_overlay_map.voxel_size = 0.1
    voxel = StructuralOverlayVoxel()
    voxel.add_vote("wall", 2.5, frame_id=1)
    state.structural_overlay_map.voxels[(0, 0, 10)] = voxel
    class_names = {3: "wall"}
    direct_ids_to_class, _direct_ids_to_name = structural_overlay_direct_label_maps(class_names, {"wall"})

    overlay_ids, overlay_supports = project_structural_overlay_to_dense_points(
        dense_points=np.array([[0.02, 0.02, 1.02], [5.0, 5.0, 5.0]], dtype=np.float32),
        structural_overlay_map=state.structural_overlay_map,
        direct_ids_to_class=direct_ids_to_class,
        class_names=class_names,
        neighbor_radius=0,
    )

    assert overlay_ids.tolist() == [-10003, -1]
    assert overlay_supports.tolist() == [2.5, 0.0]


def test_project_structural_overlay_to_dense_points_rejects_invalid_voxel_size() -> None:
    state = SystemState()
    state.structural_overlay_map.voxel_size = 0.0
    voxel = StructuralOverlayVoxel()
    voxel.add_vote("wall", 2.5, frame_id=1)
    state.structural_overlay_map.voxels[(0, 0, 10)] = voxel
    direct_ids_to_class, _direct_ids_to_name = structural_overlay_direct_label_maps({3: "wall"}, {"wall"})

    with pytest.raises(ValueError, match="voxel_size"):
        project_structural_overlay_to_dense_points(
            dense_points=np.array([[0.02, 0.02, 1.02]], dtype=np.float32),
            structural_overlay_map=state.structural_overlay_map,
            direct_ids_to_class=direct_ids_to_class,
            class_names={3: "wall"},
            neighbor_radius=0,
        )


def test_project_structural_overlay_to_dense_points_returns_defaults_without_direct_mapping() -> None:
    state = SystemState()
    state.structural_overlay_map.voxel_size = 0.1
    voxel = StructuralOverlayVoxel()
    voxel.add_vote("wall", 2.5, frame_id=1)
    state.structural_overlay_map.voxels[(0, 0, 10)] = voxel

    overlay_ids, overlay_supports = project_structural_overlay_to_dense_points(
        dense_points=np.array([[0.02, 0.02, 1.02]], dtype=np.float32),
        structural_overlay_map=state.structural_overlay_map,
        direct_ids_to_class={},
        class_names={3: "wall"},
        neighbor_radius=0,
    )

    assert overlay_ids.tolist() == [-1]
    assert overlay_supports.tolist() == [0.0]


def test_evaluate_semantics_uses_direct_labels_for_negative_structure_ids() -> None:
    gt_vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    gt_labels = np.array([3, 4], dtype=np.int32)
    dense_points = gt_vertices.copy()
    object_ids = np.array([-10003, 7], dtype=np.int32)

    evaluation = evaluate_semantics(
        gt_vertices=gt_vertices,
        gt_labels=gt_labels,
        dense_points=dense_points,
        object_ids=object_ids,
        class_names={3: "wall", 4: "chair"},
        direct_object_label_ids={-10003: 3},
    )

    assert evaluation["pred_labels"].tolist() == [3, 4]
    assert evaluation["object_to_class"][-10003] == 3
    assert evaluation["object_to_class"][7] == 4


def test_build_structural_overlay_records_exports_negative_ids_with_semantic_colors() -> None:
    state = SystemState()
    state.structural_overlay_map.voxel_size = 0.1
    voxel = StructuralOverlayVoxel()
    voxel.add_vote("window", 1.25, frame_id=2)
    state.structural_overlay_map.voxels[(1, 2, 3)] = voxel
    direct_ids_to_class, direct_ids_to_name = structural_overlay_direct_label_maps(
        {5: "window"},
        {"window"},
    )

    records = build_structural_overlay_records(
        state.structural_overlay_map,
        direct_ids_to_class=direct_ids_to_class,
        direct_ids_to_name=direct_ids_to_name,
    )

    assert len(records) == 1
    assert int(records[0]["object_id"]) == -10005
    assert float(records[0]["support"]) == 1.25
    assert tuple(int(value) for value in records[0][["red", "green", "blue"]]) == semantic_surface_color("window", -10005)
    assert np.allclose([records[0]["x"], records[0]["y"], records[0]["z"]], [0.15, 0.25, 0.35])


@pytest.mark.parametrize("voxel_size", [0.0, -0.1, float("nan"), float("inf")])
def test_build_structural_overlay_records_rejects_invalid_voxel_size(voxel_size: float) -> None:
    state = SystemState()
    state.structural_overlay_map.voxel_size = voxel_size

    with pytest.raises(ValueError, match="voxel_size"):
        build_structural_overlay_records(
            state.structural_overlay_map,
            direct_ids_to_class={},
            direct_ids_to_name={},
        )


def test_fast_eval_output_profile_disables_heavy_debug_artifacts() -> None:
    args = SimpleNamespace(
        fast_eval=True,
        full_debug=False,
        lightweight_benchmark=False,
        skip_debug_ply=False,
        skip_eval_ply=False,
        skip_preview_render=False,
    )

    profile = build_output_profile(args)

    assert profile.name == "fast_eval"
    assert profile.benchmark_audit_enabled is False
    assert profile.write_debug_ply is False
    assert profile.write_eval_mesh_ply is False
    assert profile.write_preview_render is False
    assert profile.write_primary_exports is True
    assert profile.write_final_audit is True
    assert profile.write_reports is True
    assert profile.benchmark_profile == "fast"
    assert profile.headline_excludes_runtime_vis_debug_artifacts is True
    assert profile.runtime_vis_debug_images_enabled is False


def test_full_debug_output_profile_keeps_existing_artifacts() -> None:
    args = SimpleNamespace(
        fast_eval=True,
        full_debug=True,
        lightweight_benchmark=True,
        skip_debug_ply=True,
        skip_eval_ply=True,
        skip_preview_render=True,
    )

    profile = build_output_profile(args)

    assert profile.name == "full_debug"
    assert profile.benchmark_audit_enabled is True
    assert profile.write_debug_ply is True
    assert profile.write_eval_mesh_ply is True
    assert profile.write_preview_render is True
    assert profile.write_primary_exports is True
    assert profile.write_final_audit is True
    assert profile.write_reports is True
    assert profile.benchmark_profile == "debug_overlay"
    assert profile.headline_excludes_runtime_vis_debug_artifacts is False
    assert profile.runtime_vis_debug_images_enabled is True


def test_selective_skip_flags_disable_individual_artifacts() -> None:
    args = SimpleNamespace(
        fast_eval=False,
        full_debug=False,
        lightweight_benchmark=False,
        skip_debug_ply=True,
        skip_eval_ply=True,
        skip_preview_render=True,
    )

    profile = build_output_profile(args)

    assert profile.name == "custom"
    assert profile.benchmark_audit_enabled is True
    assert profile.write_debug_ply is False
    assert profile.write_eval_mesh_ply is False
    assert profile.write_preview_render is False


def test_apply_runtime_profile_to_pipeline_marks_fast_runtime_vis() -> None:
    pipeline = SimpleNamespace(
        config={"runtime_vis": {"enabled": True, "debug_images_enabled": True}},
        runtime_vis=SimpleNamespace(config={}),
    )
    profile = build_output_profile(
        SimpleNamespace(
            fast_eval=True,
            full_debug=False,
            lightweight_benchmark=False,
            skip_debug_ply=False,
            skip_eval_ply=False,
            skip_preview_render=False,
        )
    )

    room0_eval.apply_runtime_profile_to_pipeline(pipeline, profile)

    assert pipeline.config["runtime_vis"]["benchmark_profile"] == "fast"
    assert pipeline.config["runtime_vis"]["debug_images_enabled"] is False
    assert pipeline.config["runtime_vis"]["debug_every"] == 0
    assert pipeline.runtime_vis.config == pipeline.config["runtime_vis"]


def test_write_eval_artifacts_can_skip_mesh_plys(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    gt_vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    gt_labels = np.array([1, 2], dtype=np.int32)
    evaluation = {
        "miou": 0.25,
        "macc": 0.50,
        "fmiou": 0.75,
        "fmacc": 1.0,
        "class_ious": {"chair": 0.25},
        "class_accs": {"chair": 0.50},
        "per_class": [{"class_name": "chair", "acc": 0.50, "iou": 0.25}],
    }

    write_eval_artifacts(
        eval_dir,
        gt_vertices,
        gt_labels,
        evaluation,
        {1: "chair", 2: "table"},
        write_mesh_ply=False,
    )

    assert (eval_dir / "results.json").exists()
    assert (eval_dir / "classes_iou.json").exists()
    assert (eval_dir / "classes_acc.json").exists()
    assert (eval_dir / "statistics.txt").exists()
    assert not (eval_dir / "room0_gtmesh_gt_semantic.ply").exists()
    assert not (eval_dir / "room0_gtmesh_pred_semantic.ply").exists()
    assert not (eval_dir / "room0_gtmesh_semantic_correctness.ply").exists()


def test_semantic_surface_color_is_stable_and_vivid() -> None:
    chair_a = semantic_surface_color("chair", 1)
    chair_b = semantic_surface_color("chair", 99)

    assert chair_a == chair_b
    assert max(chair_a) >= 220
    assert min(chair_a) <= 80


def test_anchor_voted_labels_drive_dense_surface_and_pool_exports() -> None:
    patch_points = np.array(
        [[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0], [0.05, 0.05, 1.0]],
        dtype=np.float32,
    )
    patches = [
        Patch3D(
            patch_id=patch_id,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
            source_frame_id=frame_id,
            metadata={
                "anchor_class_name": "rug",
                "anchor_confidence": 0.91,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
                "source_bbox_xyxy": np.array([2, 2, 10, 10], dtype=np.float32),
            },
        )
        for patch_id, frame_id in [(7, 3), (8, 4)]
    ]
    updater = ObjectUpdateModule({"provisional_pool": {"enabled": False}, "tsdf": {"voxel_size": 0.05}})
    state = updater.process(AssociationResult(new_object_patches=[7]), [patches[0]], SystemState())

    obj = next(iter(state.objects.values()))
    obj.semantic_memory = SemanticMemory(label_hypotheses=[("table", 0.99)])
    obj.debug["global_instance_substrate"]["owned_voxel_count"] = 0
    obj.debug["global_instance_substrate"]["stability_score"] = 0.7
    state = updater.process(
        AssociationResult(matched=[(8, obj.object_id, AssociationScore(total_score=0.95))]),
        [patches[1]],
        state,
    )

    state = DenseSurfaceModule({"dense_surface_voxel": 0.02, "dense_surface_cap_per_object": 8}).process(
        state,
        frame_id=4,
    )

    assert preferred_object_semantic_label(obj) == "rug"
    assert state.dense_surface_map.entries[obj.object_id].semantic_label == "rug"

    pool_records = build_pool_semantic_records(state)
    dense_records = build_dense_surface_records(state)
    expected_color = semantic_surface_color("rug", int(obj.object_id))

    assert tuple(int(value) for value in pool_records[0][["red", "green", "blue"]]) == expected_color
    assert tuple(int(value) for value in dense_records[0][["red", "green", "blue"]]) == expected_color


def test_safe_provisional_labels_drive_dense_surface_and_pool_exports() -> None:
    from src.modules.semantic_memory import accumulate_anchor_semantic_vote, set_anchor_commit_policy, set_anchor_export_policy

    patch_points = np.repeat(
        np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0], [0.05, 0.05, 1.0]], dtype=np.float32),
        64,
        axis=0,
    )
    patch = Patch3D(
        patch_id=77,
        points=patch_points,
        centroid=patch_points.mean(axis=0),
        bbox_min=patch_points.min(axis=0),
        bbox_max=patch_points.max(axis=0),
        source_frame_id=11,
        metadata={
            "anchor_class_name": "rug",
            "anchor_confidence": 0.94,
            "anchor_view_quality": 0.90,
            "anchor_label_strength": "strong",
        },
    )
    obj = ObjectMap(
        object_id=77,
        state=ObjectState.ACTIVE,
        local_pcd=patch_points.copy(),
        centroid=patch_points.mean(axis=0),
        bbox_min=patch_points.min(axis=0),
        bbox_max=patch_points.max(axis=0),
        observations=[ObservationRecord(frame_id=11, patch=patch)],
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.99)]),
    )
    obj.debug["global_instance_substrate"] = {"stability_score": 0.8}

    try:
        set_anchor_commit_policy(
            {
                "min_frame_hits": 2,
                "min_high_quality_hits": 1,
                "min_weighted_score": 1.0,
                "min_score_margin": 0.15,
                "provisional_export_fallback": False,
            }
        )
        set_anchor_export_policy(
            {
                "enabled": True,
                "min_frame_hits": 1,
                "min_high_quality_hits": 1,
                "min_weighted_score": 0.75,
                "min_score_margin": 0.05,
                "min_confidence": 0.75,
                "min_view_quality": 0.25,
                "allow_single_frame_high_confidence": True,
                "single_frame_min_confidence": 0.90,
                "single_frame_min_view_quality": 0.50,
            }
        )
        accumulate_anchor_semantic_vote(obj, patch)
        state = SystemState(objects={77: obj})
        state = DenseSurfaceModule({"dense_surface_voxel": 0.02, "dense_surface_cap_per_object": 8}).process(
            state,
            frame_id=11,
        )

        assert preferred_object_semantic_label(obj) == ""
        assert object_export_semantic_label(obj) == "rug"
        assert state.dense_surface_map.entries[77].semantic_label == "rug"

        pool_records = build_pool_semantic_records(state)
        debug_records = build_pool_debug_records(state)
        dense_records = build_dense_surface_records(state)
        expected_color = semantic_surface_color("rug", 77)
        assert tuple(int(value) for value in pool_records[0][["red", "green", "blue"]]) == expected_color
        assert tuple(int(value) for value in debug_records[0][["red", "green", "blue"]]) == expected_color
        assert tuple(int(value) for value in dense_records[0][["red", "green", "blue"]]) == expected_color
    finally:
        set_anchor_commit_policy(None)
        set_anchor_export_policy(None)


def test_final_semantic_audit_reports_export_state_counts() -> None:
    from run_room0_full_eval import build_final_object_semantic_audit
    from src.modules.semantic_memory import accumulate_anchor_semantic_vote, set_anchor_commit_policy, set_anchor_export_policy

    points = np.repeat(
        np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], dtype=np.float32),
        64,
        axis=0,
    )
    patch = Patch3D(
        patch_id=1,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        source_frame_id=1,
        metadata={
            "anchor_class_name": "rug",
            "anchor_confidence": 0.93,
            "anchor_view_quality": 0.90,
            "anchor_label_strength": "strong",
        },
    )
    obj = ObjectMap(
        object_id=5,
        state=ObjectState.ACTIVE,
        local_pcd=points.copy(),
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        observations=[ObservationRecord(frame_id=1, patch=patch)],
    )

    try:
        set_anchor_commit_policy(
            {
                "min_frame_hits": 2,
                "min_high_quality_hits": 1,
                "min_weighted_score": 1.0,
                "min_score_margin": 0.15,
                "provisional_export_fallback": False,
            }
        )
        set_anchor_export_policy(
            {
                "enabled": True,
                "min_frame_hits": 1,
                "min_high_quality_hits": 1,
                "min_weighted_score": 0.75,
                "min_score_margin": 0.05,
                "min_confidence": 0.75,
                "min_view_quality": 0.25,
                "allow_single_frame_high_confidence": True,
                "single_frame_min_confidence": 0.90,
                "single_frame_min_view_quality": 0.50,
            }
        )
        accumulate_anchor_semantic_vote(obj, patch)
        audit = build_final_object_semantic_audit(
            SystemState(objects={5: obj}),
            {
                "gt_labels": np.array([], dtype=np.int32),
                "per_class": [],
                "object_to_class": {},
            },
            {},
        )

        assert audit["semantic_state_counts"] == {"provisional": 1}
        assert audit["export_state_counts"] == {"safe_provisional": 1}
        assert audit["objects"][0]["exported_label"] == "rug"
        assert audit["objects"][0]["export_state"] == "safe_provisional"
        assert audit["objects"][0]["export_source"] == "anchor_safe_provisional"
        assert audit["objects"][0]["export_reason"].startswith("safe_provisional_anchor_evidence:")
    finally:
        set_anchor_commit_policy(None)
        set_anchor_export_policy(None)


def test_final_semantic_audit_markdown_escapes_table_cells() -> None:
    audit = {
        "object_count": 1,
        "provisional_object_count": 0,
        "semantic_state_counts": {},
        "export_state_counts": {},
        "classes": [
            {
                "class_id": 1,
                "class_name": "rug|mat",
                "gt_count": 8,
                "covered_gt_count": 8,
                "uncovered_or_unassigned_gt_count": 0,
                "matching_majority_gt_count": 8,
                "absorbed_by_other_majority_gt_count": 0,
                "absorbed_by_other_majority_fraction": 0.0,
                "iou": 0.25,
                "acc": 0.5,
                "top_covering_objects": [
                    {
                        "object_id": 5,
                        "gt_vertex_count": 8,
                        "share_of_class": 1.0,
                        "absorbed_by_other_majority": False,
                        "object_gt_majority_class": "label|x",
                        "object_gt_majority_count": 8,
                        "object_exported_label": "label|x",
                        "object_source_anchor_top_class": "source\nlabel",
                    }
                ],
            }
        ],
        "objects": [
            {
                "object_id": 5,
                "exported_label": "export|label",
                "source_anchor_top_class": "anchor\nlabel",
                "gt_majority": {"class_name": "gt|label", "count": 8},
                "gt_top_classes": [{"class_name": "top\nlabel", "count": 8}],
                "gt_total_votes": 8,
                "point_count": 16,
                "observation_count": 2,
            }
        ],
    }

    rendered = render_final_object_semantic_audit(audit)

    assert "rug\\|mat" in rendered
    assert "source label" in rendered
    assert "export\\|label" in rendered
    assert "anchor label" in rendered
    assert "gt\\|label:8" in rendered
    assert "top label:8" in rendered
    assert "source\nlabel" not in rendered
    assert "anchor\nlabel" not in rendered
    assert "top\nlabel" not in rendered


def test_tsdf_backbone_records_export_voxel_centers_and_owner_support() -> None:
    obj = ObjectMap(
        object_id=7,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[10.0, 10.0, 10.0]], dtype=np.float32),
    )
    state = SystemState(
        objects={7: obj},
        tsdf_volume=TSDFInstanceVolume(voxel_size=0.5),
    )
    state.tsdf_volume.owner_support[(2, 4, 6)] = VoxelOwnerSupport(support={7: 3.5})

    records = build_tsdf_backbone_records(state)

    assert len(records) == 1
    assert int(records[0]["object_id"]) == 7
    assert float(records[0]["support"]) == pytest.approx(3.5)
    assert np.allclose(
        [records[0]["x"], records[0]["y"], records[0]["z"]],
        [(2.0 + 0.5) * 0.5, (4.0 + 0.5) * 0.5, (6.0 + 0.5) * 0.5],
    )


def test_room0_checkpoint_roundtrip_preserves_mapping_state(tmp_path: Path) -> None:
    obj = ObjectMap(
        object_id=1,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.9)]),
    )
    state = SystemState(objects={1: obj})
    layout = room0_eval.build_room0_run_layout(tmp_path / "runs", "checkpoint_roundtrip")
    args = SimpleNamespace(
        output_root=tmp_path / "runs",
        dataset_root=tmp_path / "dataset",
        gt_labels=tmp_path / "gt.txt",
        gt_mesh_ply=tmp_path / "mesh.ply",
        gt_info_json=tmp_path / "info.json",
        proposal_device="cuda",
        proposal_backend="precomputed",
    )
    pipeline = SimpleNamespace(
        config={"structural_overlay": {"classes": ["wall"]}},
        proposal=SimpleNamespace(active_backend_name="precomputed"),
    )
    output_profile = room0_eval.OutputProfile(
        name="fast_eval",
        benchmark_audit_enabled=False,
        write_debug_ply=False,
        write_eval_mesh_ply=False,
        write_preview_render=False,
    )
    payload = room0_eval.build_room0_checkpoint_payload(
        experiment_name="checkpoint_roundtrip",
        layout=layout,
        args=args,
        pipeline=pipeline,
        state=state,
        geometry_accum={
            (0, 0, 20): [
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                np.array([10.0, 20.0, 30.0], dtype=np.float64),
                2,
            ]
        },
        frame_metrics=[{"frame_id": 0, "raw_proposal_count": 1}],
        frame_audits=[],
        frame_limit=1,
        selected_dataset_indices=[0],
        output_profile=output_profile,
        build_timing={"mapping_loop_sec_excluding_init_and_final_outputs": 1.25},
    )

    checkpoint_path = layout.scene_dir / "mapping_state.pkl"
    room0_eval.save_room0_checkpoint(checkpoint_path, payload)
    loaded = room0_eval.load_room0_checkpoint(checkpoint_path)

    loaded_args = room0_eval.args_from_room0_checkpoint(loaded)
    loaded_pipeline = room0_eval.pipeline_from_room0_checkpoint(loaded)
    loaded_profile = room0_eval.output_profile_from_checkpoint(loaded["output_profile"])
    assert loaded["state"].objects[1].semantic_memory.label_hypotheses[0][0] == "chair"
    assert loaded["geometry_accum"][(0, 0, 20)][2] == 2
    assert loaded_args.dataset_root == tmp_path / "dataset"
    assert loaded_pipeline.proposal.active_backend_name == "precomputed"
    assert loaded_profile.name == "fast_eval"
    assert loaded_profile.write_debug_ply is False
    assert loaded_profile.benchmark_profile == "fast"

    size_audit = room0_eval.audit_checkpoint_payload_sizes(payload)
    assert size_audit["key_count"] == len(payload)
    assert "state" in size_audit["top_level_key_sizes"]
    assert size_audit["top_level_key_sizes"]["state"]["size_bytes"] > 0
    assert size_audit["largest_keys"]


def test_export_room0_outputs_reuses_checkpoint_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = room0_eval.build_room0_run_layout(tmp_path / "runs", "export_from_state")
    gt_labels_path = tmp_path / "gt.txt"
    gt_info_path = tmp_path / "info.json"
    gt_mesh_path = tmp_path / "mesh.ply"
    gt_labels_path.write_text("1\n", encoding="utf-8")
    gt_info_path.write_text(json.dumps({"classes": [{"id": 1, "name": "chair"}]}), encoding="utf-8")
    gt_mesh_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        room0_eval,
        "read_gt_vertices",
        lambda _path: np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
    )

    obj = ObjectMap(
        object_id=1,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.9)]),
    )
    obj.debug["global_instance_substrate"] = {"stability_score": 0.8}
    state = SystemState(objects={1: obj})
    args = SimpleNamespace(
        gt_mesh_ply=gt_mesh_path,
        gt_labels=gt_labels_path,
        gt_info_json=gt_info_path,
        instance_voxel_size=0.05,
        projection_neighbor_radius=1,
        proposal_device="cuda",
        sam_version="2",
        sam_encoder="hiera_l",
        points_per_side=16,
        max_proposals=64,
    )
    output_profile = room0_eval.OutputProfile(
        name="fast_eval",
        benchmark_audit_enabled=False,
        write_debug_ply=False,
        write_eval_mesh_ply=False,
        write_preview_render=False,
    )
    frame_metrics = [
        {
            "frame_id": 0,
            "raw_proposal_count": 1,
            "matched_patch_count": 1,
            "ambiguous_patch_count": 0,
            "ambiguous_matched_count": 0,
            "ambiguous_new_patch_count": 0,
            "local_memory_point_count_total": 1,
            "current_frame_visibility_gate": {},
        }
    ]

    result = room0_eval.export_room0_outputs(
        layout=layout,
        experiment_name="export_from_state",
        dataset=SimpleNamespace(),
        args=args,
        frame_limit=1,
        pipeline=SimpleNamespace(
            config={"structural_overlay": {}},
            proposal=SimpleNamespace(active_backend_name="precomputed"),
        ),
        state=state,
        geometry_accum={
            (0, 0, 20): [
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                np.array([255.0, 0.0, 0.0], dtype=np.float64),
                1,
            ]
        },
        frame_metrics=frame_metrics,
        output_profile=output_profile,
    )

    results = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert results["miou"] == pytest.approx(1.0)
    assert result.report_path.exists()
    assert result.instance_map_path.exists()
    assert result.dense_instance_path.exists()
    assert not (layout.exports_dir / "room0_instance_map_local_memory.ply").exists()
    assert "checkpoint" not in result.report_path.read_text(encoding="utf-8").lower()


def test_write_run_report_describes_pool_and_tsdf_exports(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    eval_dir = tmp_path / "eval"
    scene_dir.mkdir()
    eval_dir.mkdir()
    obj = ObjectMap(
        object_id=1,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.9)]),
    )
    obj.debug["global_instance_substrate"] = {"stability_score": 0.8}
    state = SystemState(objects={1: obj})
    tsdf_records = np.zeros(1, dtype=build_tsdf_backbone_records(SystemState(objects={}, tsdf_volume=TSDFInstanceVolume(voxel_size=0.05))).dtype)
    pool_records = build_pool_semantic_records(state)
    dense_surface_records = np.zeros(0, dtype=pool_records.dtype)
    dense_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    evaluation = {
        "miou": 0.1,
        "macc": 0.2,
        "fmiou": 0.3,
        "fmacc": 0.4,
        "per_class": [{"class_name": "chair", "iou": 0.5, "acc": 0.6}],
    }
    frame_metrics = [
        {
            "raw_proposal_count": 1,
            "matched_patch_count": 1,
            "ambiguous_patch_count": 0,
            "ambiguous_matched_count": 0,
            "ambiguous_new_patch_count": 0,
            "local_memory_point_count_total": 1,
            "current_frame_visibility_gate": {
                "rejected_patch_count": 2,
                "depth_rejected_point_count": 7,
            },
        }
    ]
    args = SimpleNamespace(
        proposal_device="cuda",
        sam_version="2",
        sam_encoder="hiera_l",
        points_per_side=16,
        max_proposals=64,
    )
    pipeline = SimpleNamespace(proposal=SimpleNamespace(active_backend_name="precomputed"))

    write_run_report(
        scene_dir=scene_dir,
        eval_dir=eval_dir,
        experiment_name="test_export_contract",
        dataset=SimpleNamespace(),
        args=args,
        frame_limit=1,
        pipeline=pipeline,
        state=state,
        tsdf_records=tsdf_records,
        pool_semantic_records=pool_records,
        dense_surface_records=dense_surface_records,
        dense_points=dense_points,
        labels=labels,
        evaluation=evaluation,
        frame_metrics=frame_metrics,
        audit_dir=None,
        output_profile=build_output_profile(
            SimpleNamespace(
                fast_eval=False,
                full_debug=False,
                lightweight_benchmark=True,
                skip_debug_ply=False,
                skip_eval_ply=False,
                skip_preview_render=False,
            )
        ),
    )

    report_md = (scene_dir / "run_report.md").read_text(encoding="utf-8")
    report_json = (scene_dir / "run_report.json").read_text(encoding="utf-8")

    assert "`output_profile`: `custom`" in report_md
    assert "room0_instance_map_local_memory.ply" in report_md
    assert "room0_gtmesh_pred_semantic.ply" in report_md
    assert '"output_profile": "custom"' in report_json
    assert "pool-based semantic object map" in report_md
    assert "resident dense surface export" in report_md
    assert "coarse TSDF owner/support/stability backbone" in report_md
    assert "projected from the pool-based instance map" in report_md
    assert "room0_dense_geometry_fused_rgb.ply" in report_md
    assert "current_frame_visibility_rejected_patch_total" in report_md
    assert "current_frame_visibility_depth_rejected_point_total" in report_md
    assert '"coarse_point_count": 1' in report_json
    assert '"pool_point_count": 1' in report_json
    assert '"dense_surface_point_count": 0' in report_json
    assert '"projected_dense_labeled_point_count": 1' in report_json
    assert '"benchmark_audit_enabled": false' in report_json
    assert '"write_primary_exports": true' in report_json
    assert '"write_debug_ply": true' in report_json
    assert '"write_eval_mesh_ply": true' in report_json
    assert '"write_preview_render": true' in report_json
    assert '"write_final_audit": true' in report_json
    assert '"write_reports": true' in report_json
    assert '"current_frame_visibility_rejected_patch_total": 2' in report_json
    assert '"current_frame_visibility_depth_rejected_point_total": 7' in report_json


def test_write_run_report_omits_fast_eval_debug_artifacts(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    eval_dir = tmp_path / "eval"
    scene_dir.mkdir()
    eval_dir.mkdir()
    obj = ObjectMap(
        object_id=1,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.9)]),
    )
    obj.debug["global_instance_substrate"] = {"stability_score": 0.8}
    obj.debug["_local_pcd_chunk_pool"] = {
        "pending_point_peak": 11,
        "compaction_count": 2,
        "compaction_sec": 0.75,
    }
    obj.debug["association_geometry"] = {
        "sketch_pending_update_count": 3,
        "sketch_new_key_count": 5,
        "sketch_dropped_key_count": 1,
        "materialized_from_sketch": True,
        "association_geometry_sketch_update_sec": 0.2,
        "association_geometry_sketch_materialize_sec": 0.4,
    }
    state = SystemState(objects={1: obj})
    tsdf_records = np.zeros(0, dtype=build_tsdf_backbone_records(SystemState(objects={}, tsdf_volume=TSDFInstanceVolume(voxel_size=0.05))).dtype)
    pool_records = build_pool_semantic_records(state)
    dense_surface_records = np.zeros(0, dtype=pool_records.dtype)
    dense_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    evaluation = {
        "miou": 0.1,
        "macc": 0.2,
        "fmiou": 0.3,
        "fmacc": 0.4,
        "per_class": [{"class_name": "chair", "iou": 0.5, "acc": 0.6}],
    }
    frame_metrics = [
        {
            "raw_proposal_count": 1,
            "matched_patch_count": 1,
            "ambiguous_patch_count": 0,
            "ambiguous_matched_count": 0,
            "ambiguous_new_patch_count": 0,
            "local_memory_point_count_total": 1,
            "current_frame_visibility_gate": {
                "rejected_patch_count": 0,
                "depth_rejected_point_count": 0,
            },
        }
    ]
    args = SimpleNamespace(
        proposal_device="cuda",
        sam_version="2",
        sam_encoder="hiera_l",
        points_per_side=16,
        max_proposals=64,
    )
    pipeline = SimpleNamespace(proposal=SimpleNamespace(active_backend_name="precomputed"))
    output_profile = build_output_profile(
        SimpleNamespace(
            fast_eval=True,
            full_debug=False,
            lightweight_benchmark=False,
            skip_debug_ply=False,
            skip_eval_ply=False,
            skip_preview_render=False,
        )
    )

    write_run_report(
        scene_dir=scene_dir,
        eval_dir=eval_dir,
        experiment_name="test_fast_eval_report",
        dataset=SimpleNamespace(),
        args=args,
        frame_limit=1,
        pipeline=pipeline,
        state=state,
        tsdf_records=tsdf_records,
        pool_semantic_records=pool_records,
        dense_surface_records=dense_surface_records,
        dense_points=dense_points,
        labels=labels,
        evaluation=evaluation,
        frame_metrics=frame_metrics,
        audit_dir=None,
        output_profile=output_profile,
        benchmark_timing={
            "online_mapping_sec": 12.5,
            "headline_online_mapping_sec": 10.5,
            "finalization_sec": 1.25,
            "eval_io_sec": 3.5,
            "mapping_state_size_bytes": 4096,
            "largest_export_size_bytes": 8192,
            "benchmark_profile": "fast",
            "headline_excludes_runtime_vis_debug_artifacts": True,
            "runtime_vis_debug_overlay_sec_excluded_from_headline": 2.0,
            "checkpoint_size_audit": {
                "largest_keys": [{"key": "state", "size_bytes": 123, "method": "pickle"}],
                "top_level_key_sizes": {"state": {"size_bytes": 123, "method": "pickle"}},
            },
            "checkpoint_largest_payload_keys": [{"key": "state", "size_bytes": 123, "method": "pickle"}],
        },
    )

    report_md = (scene_dir / "run_report.md").read_text(encoding="utf-8")
    report_json = (scene_dir / "run_report.json").read_text(encoding="utf-8")

    assert "`output_profile`: `fast_eval`" in report_md
    assert "room0_instance_map.ply" in report_md
    assert "room0_instance_map_dense_surface.ply" in report_md
    assert "room0_dense_geometry_fused_rgb.ply" in report_md
    assert "room0_dense_geometry_instance_projected.ply" in report_md
    assert "conservative structural overlay fusion" in report_md
    assert "synthetic negative structural overlay ids" in report_md
    assert "room0_instance_map_local_memory.ply" not in report_md
    assert "room0_gtmesh_pred_semantic.ply" not in report_md
    assert "local_memory_audit.jsonl" not in report_md
    assert '"output_profile": "fast_eval"' in report_json
    assert '"benchmark_audit_enabled": false' in report_json
    assert '"write_primary_exports": true' in report_json
    assert '"write_debug_ply": false' in report_json
    assert '"write_eval_mesh_ply": false' in report_json
    assert '"write_preview_render": false' in report_json
    assert '"write_final_audit": true' in report_json
    assert '"write_reports": true' in report_json
    report_payload = json.loads(report_json)
    assert report_payload["online_mapping_sec"] == pytest.approx(12.5)
    assert report_payload["headline_online_mapping_sec"] == pytest.approx(10.5)
    assert report_payload["raw_online_mapping_sec"] == pytest.approx(12.5)
    assert report_payload["runtime_vis_debug_overlay_sec_excluded_from_headline"] == pytest.approx(2.0)
    assert report_payload["benchmark_profile"] == "fast"
    assert report_payload["checkpoint_largest_payload_keys"][0]["key"] == "state"
    assert "Checkpoint Size Audit" in report_md
    assert report_payload["finalization_sec"] == pytest.approx(1.25)
    assert report_payload["eval_io_sec"] == pytest.approx(3.5)
    assert report_payload["end_to_end_sec"] == pytest.approx(17.25)
    assert report_payload["final_local_memory_point_count_total"] == 1
    assert report_payload["max_local_memory_point_count_total"] == 1
    assert report_payload["local_pcd_pending_point_peak"] == 11
    assert report_payload["local_pcd_compaction_count"] == 2
    assert report_payload["local_pcd_compaction_sec"] == pytest.approx(0.75)
    assert report_payload["association_geometry_sketch_update_count"] == 3
    assert report_payload["association_geometry_sketch_new_key_count"] == 5
    assert report_payload["association_geometry_sketch_dropped_key_count"] == 1
    assert report_payload["association_geometry_sketch_materialize_count"] == 1
    assert report_payload["association_geometry_sketch_update_sec"] == pytest.approx(0.2)
    assert report_payload["association_geometry_sketch_materialize_sec"] == pytest.approx(0.4)
    assert report_payload["mapping_state_size_bytes"] == 4096
    assert report_payload["largest_export_size_bytes"] == 8192


def test_benchmark_contract_reads_flushed_local_pool_debug() -> None:
    obj = ObjectMap(
        object_id=1,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.9)]),
    )
    obj.debug["local_geometry_pool"] = {
        "local_pcd_pending_point_peak": 17,
        "local_pcd_compaction_count": 3,
        "local_pcd_compaction_sec": 0.5,
    }
    obj.debug["association_geometry"] = {
        "sketch_update_count": 4,
        "sketch_new_key_count": 0,
        "sketch_dropped_key_count": 0,
        "sketch_new_key_count_total": 7,
        "sketch_dropped_key_count_total": 2,
        "sketch_materialize_count": 3,
    }
    state = SystemState(objects={1: obj})

    payload = room0_eval.build_benchmark_contract_payload(
        state=state,
        frame_metrics=[],
        online_mapping_sec=1.0,
        finalization_sec=2.0,
        eval_io_sec=3.0,
    )

    assert payload["end_to_end_sec"] == pytest.approx(6.0)
    assert payload["local_pcd_pending_point_peak"] == 17
    assert payload["local_pcd_compaction_count"] == 3
    assert payload["local_pcd_compaction_sec"] == pytest.approx(0.5)
    assert payload["association_geometry_sketch_update_count"] == 4
    assert payload["association_geometry_sketch_new_key_count"] == 7
    assert payload["association_geometry_sketch_dropped_key_count"] == 2
    assert payload["association_geometry_sketch_materialize_count"] == 3


def test_checkpointed_export_updates_report_sidecar_final_timing(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "run_report.json"
    sidecar_path.write_text(
        json.dumps({"online_mapping_sec": 1.0, "eval_io_sec": 0.0, "miou": 0.1}),
        encoding="utf-8",
    )

    _update_report_sidecar(sidecar_path, {"eval_io_sec": 3.0, "end_to_end_sec": 6.0})

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["online_mapping_sec"] == pytest.approx(1.0)
    assert payload["eval_io_sec"] == pytest.approx(3.0)
    assert payload["end_to_end_sec"] == pytest.approx(6.0)
    assert payload["miou"] == pytest.approx(0.1)


def test_write_run_report_includes_stage_timing_summary(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    eval_dir = tmp_path / "eval"
    scene_dir.mkdir()
    eval_dir.mkdir()
    obj = ObjectMap(
        object_id=1,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.9)]),
    )
    obj.debug["global_instance_substrate"] = {"stability_score": 0.8}
    state = SystemState(objects={1: obj})
    tsdf_records = np.zeros(0, dtype=build_tsdf_backbone_records(SystemState(objects={}, tsdf_volume=TSDFInstanceVolume(voxel_size=0.05))).dtype)
    pool_records = build_pool_semantic_records(state)
    dense_surface_records = np.zeros(0, dtype=pool_records.dtype)
    dense_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    evaluation = {
        "miou": 0.1,
        "macc": 0.2,
        "fmiou": 0.3,
        "fmacc": 0.4,
        "per_class": [{"class_name": "chair", "iou": 0.5, "acc": 0.6}],
    }
    output_profile = build_output_profile(
        SimpleNamespace(
            fast_eval=True,
            full_debug=False,
            lightweight_benchmark=False,
            skip_debug_ply=False,
            skip_eval_ply=False,
            skip_preview_render=False,
        )
    )
    frame_metrics = [
        {
            "raw_proposal_count": 2,
            "matched_patch_count": 1,
            "ambiguous_patch_count": 0,
            "ambiguous_matched_count": 0,
            "ambiguous_new_patch_count": 0,
            "local_memory_point_count_total": 10,
            "frontend_stage": {
                "source_sam_proposal_count": 0,
                "anchor_voted_proposal_count": 2,
                "runtime_merged_proposal_count": 2,
                "runtime_merge_count": 0,
                "anchor_voted_unanchored_count": 0,
                "runtime_anchor_label_missing_edge_count": 0,
                "runtime_anchor_label_mismatch_edge_count": 0,
                "runtime_anchor_identity_mismatch_edge_count": 0,
            },
            "async_refinement": {
                "sam_proposal_count": 10,
                "fine_proposal_count": 3,
                "fine_patch_count": 2,
                "replaced_observation_count": 2,
                "inserted_observation_count": 2,
                "updated_object_count": 1,
            },
            "surface_owner_gate": {},
            "current_frame_visibility_gate": {},
            "stage_timings": {"proposal_generation": 0.2, "patch_lifting": 0.4},
        },
        {
            "raw_proposal_count": 4,
            "matched_patch_count": 2,
            "ambiguous_patch_count": 0,
            "ambiguous_matched_count": 0,
            "ambiguous_new_patch_count": 0,
            "local_memory_point_count_total": 20,
            "frontend_stage": {
                "source_sam_proposal_count": 0,
                "anchor_voted_proposal_count": 4,
                "runtime_merged_proposal_count": 4,
                "runtime_merge_count": 0,
                "anchor_voted_unanchored_count": 0,
                "runtime_anchor_label_missing_edge_count": 0,
                "runtime_anchor_label_mismatch_edge_count": 0,
                "runtime_anchor_identity_mismatch_edge_count": 0,
            },
            "async_refinement": {
                "sam_proposal_count": 8,
                "fine_proposal_count": 1,
                "fine_patch_count": 1,
                "replaced_observation_count": 1,
                "inserted_observation_count": 1,
                "updated_object_count": 1,
            },
            "surface_owner_gate": {},
            "current_frame_visibility_gate": {},
            "stage_timings": {"proposal_generation": 0.4, "patch_lifting": 0.8},
        },
    ]
    args = SimpleNamespace(
        proposal_device="cuda",
        sam_version="2",
        sam_encoder="hiera_l",
        points_per_side=16,
        max_proposals=64,
    )
    pipeline = SimpleNamespace(proposal=SimpleNamespace(active_backend_name="precomputed"))

    write_run_report(
        scene_dir=scene_dir,
        eval_dir=eval_dir,
        experiment_name="timing_report",
        dataset=SimpleNamespace(),
        args=args,
        frame_limit=2,
        pipeline=pipeline,
        state=state,
        tsdf_records=tsdf_records,
        pool_semantic_records=pool_records,
        dense_surface_records=dense_surface_records,
        dense_points=dense_points,
        labels=labels,
        evaluation=evaluation,
        frame_metrics=frame_metrics,
        audit_dir=None,
        output_profile=output_profile,
    )

    report_md = (scene_dir / "run_report.md").read_text(encoding="utf-8")
    report_json = json.loads((scene_dir / "run_report.json").read_text(encoding="utf-8"))
    assert "Stage Timing Summary" in report_md
    assert (
        "- `proposal_generation`: mean `0.3000s`, median `0.3000s`, "
        "p95 `0.4000s`, max `0.4000s` at frame `-1`, total `0.6000s`"
    ) in report_md
    assert (
        "- `patch_lifting`: mean `0.6000s`, median `0.6000s`, "
        "p95 `0.8000s`, max `0.8000s` at frame `-1`, total `1.2000s`"
    ) in report_md
    assert "- `async_refinement_fine_proposal_count_total`: `4`" in report_md
    assert "- `async_refinement_replaced_observation_count_total`: `3`" in report_md
    assert report_json["stage_timing_summary"] == {
        "patch_lifting": {
            "mean_sec": 0.6000000000000001,
            "median_sec": 0.6000000000000001,
            "p95_sec": 0.8,
            "max_sec": 0.8,
            "total_sec": 1.2000000000000002,
            "max_frame_id": -1,
            "outlier_frame_ids": [-1],
        },
        "proposal_generation": {
            "mean_sec": 0.30000000000000004,
            "median_sec": 0.30000000000000004,
            "p95_sec": 0.4,
            "max_sec": 0.4,
            "total_sec": 0.6000000000000001,
            "max_frame_id": -1,
            "outlier_frame_ids": [-1],
        },
    }
    assert report_json["async_refinement_fine_patch_count_total"] == 3
    assert report_json["async_refinement_updated_object_count_total"] == 2


def test_build_stage_timing_summary_skips_bad_values() -> None:
    frame_metrics = [
        {},
        {"stage_timings": {}},
        {"stage_timings": None},
        {"stage_timings": {"proposal_generation": 0.2, "patch_lifting": "0.4"}},
        {"stage_timings": {"proposal_generation": "0.4", "patch_lifting": 0.8}},
        {"stage_timings": {"proposal_generation": "bad", "patch_lifting": float("nan")}},
        {"stage_timings": {"proposal_generation": float("inf"), "invalid_only": "bad"}},
    ]

    assert build_stage_timing_summary(frame_metrics) == {
        "patch_lifting": {
            "mean_sec": 0.6000000000000001,
            "median_sec": 0.6000000000000001,
            "p95_sec": 0.8,
            "max_sec": 0.8,
            "total_sec": 1.2000000000000002,
            "max_frame_id": -1,
            "outlier_frame_ids": [-1],
        },
        "proposal_generation": {
            "mean_sec": 0.30000000000000004,
            "median_sec": 0.30000000000000004,
            "p95_sec": 0.4,
            "max_sec": 0.4,
            "total_sec": 0.6000000000000001,
            "max_frame_id": -1,
            "outlier_frame_ids": [-1],
        },
    }


def test_build_async_refinement_summary_totals_counts() -> None:
    from run_room0_full_eval import build_async_refinement_summary

    frame_metrics = [
        {},
        {"async_refinement": None},
        {"async_refinement": "bad"},
        {
            "async_refinement": {
                "sam_proposal_count": 10,
                "fine_proposal_count": 3,
                "fine_patch_count": 2,
                "replaced_observation_count": 2,
                "inserted_observation_count": 2,
                "updated_object_count": 1,
            }
        },
        {
            "async_refinement": {
                "sam_proposal_count": 8,
                "fine_proposal_count": 1,
                "fine_patch_count": 1,
                "replaced_observation_count": 1,
                "inserted_observation_count": 1,
                "updated_object_count": 1,
            }
        },
        {
            "async_refinement": {
                "sam_proposal_count": "bad",
                "fine_proposal_count": float("nan"),
                "fine_patch_count": float("inf"),
                "replaced_observation_count": object(),
                "inserted_observation_count": "",
                "updated_object_count": None,
            }
        },
    ]

    assert build_async_refinement_summary(frame_metrics) == {
        "async_refinement_sam_proposal_count_total": 18,
        "async_refinement_fine_proposal_count_total": 4,
        "async_refinement_fine_patch_count_total": 3,
        "async_refinement_replaced_observation_count_total": 3,
        "async_refinement_inserted_observation_count_total": 3,
        "async_refinement_updated_object_count_total": 2,
    }


def test_room0_fast_hybrid_config_uses_anchor_primary_fast_path() -> None:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "room0_fast_hybrid_4090.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["anchor_frontend"]["anchor_primary_mode"] is True
    assert config["anchor_frontend"]["use_sam_intersection_proposals"] is False
    assert config["patch_lifting"]["point_sample_ratio"] == 0.08
    assert config["patch_lifting"]["max_points_per_patch"] == 2048
    assert config["object_update"]["surface_owner_gate"]["representative_voxel_mode"] is True
    assert config["pipeline"]["collect_stage_timings"] is True
    assert config["anchor_frontend"]["supplemental"]["worker_enabled"] is True
    assert config["anchor_frontend"]["supplemental"]["worker_fallback_on_error"] is True
    assert config["anchor_frontend"]["supplemental"]["worker_request_timeout_sec"] == 120.0
    assert config["depth_refinement"]["connected_components_backend"] == "opencv"
    assert config["depth_refinement"]["use_bbox_crop"] is True


def test_room0_fast_structural_overlay_config_enables_conservative_overlay() -> None:
    config_path = Path("configs/room0_fast_structural_overlay_4090.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["proposal"]["backend"] == "precomputed"
    assert config["anchor_frontend"]["anchor_primary_mode"] is True
    assert config["anchor_frontend"]["use_sam_intersection_proposals"] is False
    assert config["structural_overlay"]["enabled"] is True
    assert config["structural_overlay"]["conservative_fusion"] is True
    assert config["structural_overlay"]["classes"] == ["wall", "window", "blinds", "ceiling", "floor", "door"]
    assert "sofa" in config["structural_overlay"]["protected_labels"]
    assert "rug" in config["structural_overlay"]["protected_labels"]
    assert "chair" in config["structural_overlay"]["protected_labels"]
    assert "table" in config["structural_overlay"]["protected_labels"]


def test_room0_coarse_to_fine_async_config_enables_fast_coarse_and_refinement() -> None:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "room0_coarse_to_fine_async_4090.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["anchor_frontend"]["anchor_primary_mode"] is True
    assert config["anchor_frontend"]["use_sam_intersection_proposals"] is False
    assert config["async_refinement"]["enabled"] is True
    assert config["async_refinement"]["delay_frames"] == 0
    assert config["async_refinement"]["clip_to_anchor_box"] is True
    assert config["async_refinement"]["replace_coarse_observations"] is True
    assert config["async_refinement"]["rebuild_tsdf_support"] is False
    assert config["async_refinement"]["max_components_per_replacement"] == 1
    assert config["pipeline"]["collect_stage_timings"] is True
    assert config["patch_lifting"]["point_sample_ratio"] == 0.08
    assert config["depth_refinement"]["connected_components_backend"] == "opencv"
    assert config["depth_refinement"]["use_bbox_crop"] is True
