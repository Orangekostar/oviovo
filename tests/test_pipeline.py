"""Unit test stubs for the OVIOVO backbone."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import (
    ActiveSet,
    AssociationResult,
    AssociationScore,
    BackgroundMap,
    CameraIntrinsics,
    Frame,
    ObjectMap,
    ObjectState,
    ObservationRecord,
    Patch3D,
    Proposal2D,
    SemanticMemory,
    SystemState,
    TSDFInstanceVolume,
    VoxelOwnerSupport,
)
from src.modules.association import AssociationModule
from src.modules.depth_refinement import DepthRefinementModule
from src.modules.object_update import ObjectUpdateModule
from src.modules.semantic_memory import SemanticMemoryModule
from src.modules.tsdf_instance_map import TSDFInstanceMapModule


class TestDataStructures:
    """Tests for core data structures."""

    def test_camera_intrinsics_matrix(self):
        intr = CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480)
        K = intr.to_matrix()
        assert K.shape == (3, 3)
        assert K[0, 0] == 525
        assert K[1, 1] == 525
        assert K[0, 2] == 320
        assert K[1, 2] == 240

    def test_frame_creation(self):
        intr = CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480)
        frame = Frame(
            frame_id=0,
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            depth=np.ones((480, 640), dtype=np.float32),
            pose=np.eye(4),
            intrinsics=intr,
        )
        assert frame.frame_id == 0
        assert frame.rgb.shape == (480, 640, 3)

    def test_proposal2d(self):
        mask = np.ones((480, 640), dtype=bool)
        p = Proposal2D(proposal_id=0, mask=mask, bbox_xyxy=np.array([0, 0, 640, 480]), area=480 * 640)
        assert p.area == 480 * 640

    def test_patch3d(self):
        points = np.random.rand(100, 3).astype(np.float32)
        patch = Patch3D(
            patch_id=0,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
        )
        assert patch.points.shape == (100, 3)

    def test_object_state_enum(self):
        assert ObjectState.ACTIVE.value == "active"
        assert ObjectState.GHOST.value == "ghost"

    def test_object_map_defaults(self):
        obj = ObjectMap(object_id=0)
        assert obj.state == ObjectState.ACTIVE
        assert len(obj.observations) == 0
        assert obj.confidence == 1.0

    def test_semantic_memory_defaults(self):
        mem = SemanticMemory()
        assert len(mem.feature_bank) == 0
        assert mem.aggregated_feature is None

    def test_background_map_defaults(self):
        bg = BackgroundMap()
        assert bg.voxel_size == 0.05
        assert len(bg.point_cloud) == 0

    def test_association_result(self):
        result = AssociationResult()
        assert len(result.matched) == 0
        assert len(result.new_object_patches) == 0

    def test_system_state(self):
        state = SystemState()
        assert len(state.objects) == 0
        assert state.next_object_id == 0


class TestGeometryUtils:
    """Tests for geometry utilities."""

    def test_depth_to_points(self):
        from src.utils.geometry import depth_to_points
        depth = np.full((4, 4), 2.0, dtype=np.float32)
        intr = CameraIntrinsics(fx=1, fy=1, cx=2, cy=2, width=4, height=4)
        pose = np.eye(4)
        pts = depth_to_points(depth, intr, pose)
        assert pts.shape[1] == 3
        assert len(pts) == 16

    def test_bbox_iou_3d(self):
        from src.utils.geometry import bbox_iou_3d
        # Identical boxes -> IoU = 1
        a_min = np.array([0, 0, 0], dtype=np.float32)
        a_max = np.array([1, 1, 1], dtype=np.float32)
        iou = bbox_iou_3d(a_min, a_max, a_min, a_max)
        assert abs(iou - 1.0) < 1e-6

        # Non-overlapping -> IoU = 0
        b_min = np.array([2, 2, 2], dtype=np.float32)
        b_max = np.array([3, 3, 3], dtype=np.float32)
        iou = bbox_iou_3d(a_min, a_max, b_min, b_max)
        assert iou == 0.0

    def test_voxel_downsample(self):
        from src.utils.geometry import voxel_downsample
        pts = np.array([[0, 0, 0], [0.001, 0, 0], [1, 1, 1]], dtype=np.float32)
        ds = voxel_downsample(pts, voxel_size=0.1)
        assert len(ds) <= len(pts)


class TestPipeline:
    """Integration tests for the full pipeline."""

    def test_pipeline_runs(self, tmp_path):
        """Verify the pipeline can process a frame without errors."""
        pipeline_mod = __import__("src.pipelines.main_pipeline", fromlist=["Pipeline"])
        Pipeline = pipeline_mod.Pipeline
        config_path = tmp_path / "test_pipeline_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "proposal:",
                    "  backend: placeholder",
                    "  min_mask_area: 10",
                    "  max_proposals: 3",
                    "pipeline:",
                    "  verbose: false",
                    "logging:",
                    "  level: ERROR",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=config_path)

        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        pose = np.eye(4)
        intr = CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480)

        state = pipe.process_frame(rgb, depth, pose, intr)
        assert isinstance(state, SystemState)
        assert state.frame_count == 1


class TestRoadmapRefactor:
    """Regression tests for the overviewpro roadmap constraints."""

    def test_depth_refinement_splits_disconnected_components_and_exposes_scores(self):
        depth = np.full((8, 8), 2.0, dtype=np.float32)
        depth[:, 4] = 4.0

        mask = np.zeros((8, 8), dtype=bool)
        mask[1:7, 1:7] = True
        proposal = Proposal2D(
            proposal_id=7,
            mask=mask,
            bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.9,
        )

        module = DepthRefinementModule({"depth_edge_threshold": 0.5, "min_mask_area_after_refine": 4})
        refined = module.process(depth, [proposal])

        assert len(refined) == 2
        assert all(r.metadata["source_raw_proposal_id"] == 7 for r in refined)
        assert all("geometric_features" in r.metadata for r in refined)
        assert all("soft_scores" in r.metadata for r in refined)
        assert all(r.geometric_features.depth_valid_ratio > 0.0 for r in refined)

    def test_association_restricts_candidates_to_active_set(self):
        module = AssociationModule({"match_threshold": 0.3})

        near_points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
        far_points = np.array([[4.0, 0.0, 1.0], [4.1, 0.0, 1.0], [4.0, 0.1, 1.0]], dtype=np.float32)
        patch = Patch3D(
            patch_id=0,
            points=near_points,
            centroid=near_points.mean(axis=0),
            bbox_min=near_points.min(axis=0),
            bbox_max=near_points.max(axis=0),
        )

        objects = {
            1: ObjectMap(
                object_id=1,
                centroid=far_points.mean(axis=0),
                bbox_min=far_points.min(axis=0),
                bbox_max=far_points.max(axis=0),
                local_pcd=far_points.copy(),
            ),
            2: ObjectMap(
                object_id=2,
                centroid=near_points.mean(axis=0),
                bbox_min=near_points.min(axis=0),
                bbox_max=near_points.max(axis=0),
                local_pcd=near_points.copy(),
            ),
        }

        result = module.process(
            [patch],
            objects,
            SystemState().tsdf_volume,
            active_set=ActiveSet(visible_ids={1}),
        )

        assert result.new_object_patches == [0]
        assert (0, 2) not in result.scores
        assert result.debug["used_active_set"] is True
        assert result.debug["per_patch"][0]["candidate_object_ids"] == [1]

    def test_association_with_empty_active_set_does_not_fallback_to_global_objects(self):
        module = AssociationModule({"match_threshold": 0.1})

        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
        patch = Patch3D(
            patch_id=5,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
        )
        objects = {
            9: ObjectMap(
                object_id=9,
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                local_pcd=points.copy(),
            )
        }

        result = module.process([patch], objects, SystemState().tsdf_volume, active_set=ActiveSet())

        assert result.new_object_patches == [5]
        assert result.scores == {}
        assert result.debug["used_active_set"] is True
        assert result.debug["fallback_to_all_objects"] is False
        assert result.debug["per_patch"][5]["candidate_object_ids"] == []

    def test_tsdf_visible_instances_respects_camera_frustum(self):
        module = TSDFInstanceMapModule({"voxel_size": 1.0})
        volume = TSDFInstanceVolume(voxel_size=1.0)
        volume.owner_support[(0, 0, 2)] = VoxelOwnerSupport({1: 2.0})
        volume.owner_support[(20, 0, 2)] = VoxelOwnerSupport({2: 2.0})
        volume.owner_support[(0, 0, -2)] = VoxelOwnerSupport({3: 2.0})

        intr = CameraIntrinsics(fx=10.0, fy=10.0, cx=10.0, cy=10.0, width=20, height=20)
        visible = module.query_visible_instances(volume, np.eye(4), intr)

        assert visible == {1}

    def test_association_uses_supported_voxel_normalization_consistently(self):
        module = AssociationModule({"match_threshold": 0.0})
        patch_points = np.array(
            [[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [2.1, 0.1, 0.1]],
            dtype=np.float32,
        )
        patch = Patch3D(
            patch_id=0,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )
        obj = ObjectMap(
            object_id=7,
            centroid=patch_points[:2].mean(axis=0),
            bbox_min=patch_points[:2].min(axis=0),
            bbox_max=patch_points[:2].max(axis=0),
            local_pcd=patch_points[:2].copy(),
        )
        volume = TSDFInstanceVolume(voxel_size=1.0)
        volume.owner_support[(0, 0, 0)] = VoxelOwnerSupport({7: 1.0})
        volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({7: 1.0})

        result = module.process([patch], {7: obj}, volume)
        score = result.scores[(0, 7)]

        assert score.voxel_vote_score == pytest.approx(1.0)
        assert score.vote_result is not None
        assert score.vote_result.touched_voxel_count == 3
        assert score.vote_result.supported_voxel_count == 2
        assert score.vote_result.normalized_vote_score == pytest.approx(1.0)
        assert result.debug["per_patch"][0]["vote"]["normalized_vote_score"] == pytest.approx(1.0)

    def test_object_update_marks_tsdf_backbone_as_global_and_local_pcd_as_maintenance(self):
        patch_points = np.array(
            [[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0], [0.05, 0.05, 1.0]],
            dtype=np.float32,
        )
        patch = Patch3D(
            patch_id=0,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
            source_frame_id=3,
            metadata={"source_bbox_xyxy": np.array([1, 1, 4, 4], dtype=np.float32)},
        )
        state = SystemState(
            objects={
                0: ObjectMap(
                    object_id=0,
                    local_pcd=patch_points[:2].copy(),
                    centroid=patch_points[:2].mean(axis=0),
                    bbox_min=patch_points[:2].min(axis=0),
                    bbox_max=patch_points[:2].max(axis=0),
                )
            }
        )
        association = AssociationResult(matched=[(0, 0, AssociationScore(total_score=0.9))])

        updated = ObjectUpdateModule({}).process(association, [patch], state)
        obj = updated.objects[0]

        assert obj.debug["global_instance_substrate"]["role"] == "true_low_level_backbone"
        assert obj.debug["global_instance_substrate"]["stability_score"] > 0.0
        assert obj.debug["local_geometry_memory"]["role"] == "maintenance_only"
        assert obj.observations[-1].crop_bbox.tolist() == [1.0, 1.0, 4.0, 4.0]

    def test_semantic_memory_waits_for_stabilized_instances_then_updates_explicit_matches(self):
        rgb = np.full((32, 32, 3), 120, dtype=np.uint8)
        patch_points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
        patch = Patch3D(
            patch_id=0,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )
        obj = ObjectMap(
            object_id=0,
            update_count=1,
            observations=[ObservationRecord(frame_id=0, patch=patch, crop_bbox=np.array([4, 4, 20, 20], dtype=np.float32))],
            debug={"global_instance_substrate": {"stability_score": 0.05}},
        )
        state = SystemState(objects={0: obj})
        module = SemanticMemoryModule(
            {
                "backend": "placeholder",
                "feature_dim": 32,
                "top_k_labels": 2,
                "min_observations_for_label": 1,
                "min_stable_observations": 2,
                "min_stability_score": 0.2,
                "label_candidates": ["chair", "table", "cabinet"],
            }
        )

        module.process(state, rgb, [0])
        assert state.objects[0].semantic_memory.observation_count == 0
        assert state.objects[0].semantic_memory.debug["stability_gate"]["passed"] is False

        state.objects[0].update_count = 3
        state.objects[0].debug["global_instance_substrate"]["stability_score"] = 0.8
        module.process(state, rgb, [0])

        mem = state.objects[0].semantic_memory
        assert mem.observation_count == 1
        assert len(mem.label_hypotheses) == 2
        assert mem.debug["stability_gate"]["passed"] is True
        assert len(mem.debug["similarity_scores"]) == 3
        assert mem.debug["selected_bbox_xyxy"] == [4, 4, 20, 20]

    def test_association_does_not_match_ghost_objects(self):
        module = AssociationModule({"match_threshold": 0.1})
        points = np.array(
            [[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]],
            dtype=np.float32,
        )
        patch = Patch3D(
            patch_id=8,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
        )
        ghost = ObjectMap(
            object_id=2,
            state=ObjectState.GHOST,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            local_pcd=points.copy(),
        )

        result = module.process([patch], {2: ghost}, SystemState().tsdf_volume)

        assert result.matched == []
        assert result.new_object_patches == [8]
        assert result.debug["per_patch"][8]["candidate_object_ids"] == []
