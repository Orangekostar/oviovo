"""Unit test stubs for the OVIOVO backbone."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import (
    ActiveSet,
    Anchor2D,
    AnchorAssignment,
    AssociationResult,
    AssociationScore,
    BackgroundMap,
    CameraIntrinsics,
    ContestedAssociation,
    Frame,
    ObjectMap,
    ObjectState,
    ObservationRecord,
    Patch3D,
    Proposal2D,
    ProposalSoftScores,
    RefinedProposal2D,
    SemanticMemory,
    SystemState,
    TSDFInstanceVolume,
    VoxelVoteResult,
    VoxelOwnerSupport,
    WholeEvidenceScores,
)
from src.modules.active_set import ActiveSetModule
from src.modules.association import AssociationModule
from src.modules.anchor_guided_sam import AnchorGuidedSAMModule
from src.modules.bg_obj_split import BgObjSplitModule
from src.modules.depth_refinement import DepthRefinementModule
from src.modules.dense_surface import DenseSurfaceModule
from src.modules.object_update import ObjectUpdateModule
from src.modules.patch_lifting import PatchLiftingModule
from src.modules.runtime_vis import RuntimeVisModule, RuntimeVisOutput
from src.modules.semantic_memory import (
    SemanticMemoryModule,
    accumulate_anchor_semantic_vote,
    object_export_semantic_label,
    object_export_semantic_state,
    object_association_identity_semantic_label,
    object_identity_semantic_label,
    object_semantic_commit_state,
    preferred_object_semantic_label,
)
from src.modules.tsdf_instance_map import TSDFInstanceMapModule
from frontend.proposal_cache import save_proposals, write_manifest
from run_room0_full_eval import (
    build_frontend_stage_report_payload,
    build_final_object_semantic_audit,
    build_local_memory_frame_audit,
    build_prefetch_pipeline,
    build_scheduling_report_payload,
    build_stage_timing_summary,
    class_color,
    labels_to_class_colors,
    save_local_memory_audit_overlay,
)
from src.pipelines.main_pipeline import Pipeline
from src.pipelines.proposal_bundle import FrameProposalBundle


class TestObjectUpdateModule:
    def test_object_update_records_observation_provenance(self):
        module = ObjectUpdateModule(
            {
                "surface_owner_gate": {"enabled": False},
                "provisional": {"enabled": False},
            }
        )
        state = SystemState()
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        patch = Patch3D(
            patch_id=7,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=12,
            metadata={
                "source_proposal_id": 7,
                "source_backend_name": "anchor_box_primary",
                "anchor_id": 3,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.91,
                "anchor_label_strength": "strong",
                "observation_layer": "coarse",
                "refinement_key": "12:3",
            },
        )
        association = AssociationResult(new_object_patches=[7])

        state = module.process(association, [patch], state)

        obs = state.objects[0].observations[0]
        assert obs.source_frame_id == 12
        assert obs.source_proposal_id == 7
        assert obs.anchor_id == 3
        assert obs.observation_layer == "coarse"
        assert obs.refinement_key == "12:3"
        assert obs.replaced_by_refinement is False

    def test_replace_observations_removes_coarse_points_and_rebuilds_anchor_votes(self):
        module = ObjectUpdateModule(
            {
                "surface_owner_gate": {"enabled": False},
                "provisional": {"enabled": False},
            }
        )
        state = SystemState()
        coarse_points = np.array(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
            dtype=np.float32,
        )
        coarse_patch = Patch3D(
            patch_id=1,
            points=coarse_points,
            centroid=coarse_points.mean(axis=0),
            bbox_min=coarse_points.min(axis=0),
            bbox_max=coarse_points.max(axis=0),
            source_frame_id=5,
            metadata={
                "source_proposal_id": 1,
                "source_backend_name": "anchor_box_primary",
                "anchor_id": 9,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.9,
                "anchor_label_strength": "strong",
                "observation_layer": "coarse",
                "refinement_key": "5:9",
            },
        )
        state = module.process(AssociationResult(new_object_patches=[1]), [coarse_patch], state)

        fine_points = np.array([[0.0, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
        fine_patch = Patch3D(
            patch_id=2,
            points=fine_points,
            centroid=fine_points.mean(axis=0),
            bbox_min=fine_points.min(axis=0),
            bbox_max=fine_points.max(axis=0),
            source_frame_id=5,
            metadata={
                "source_proposal_id": 2,
                "source_backend_name": "async_sam_refinement",
                "anchor_id": 9,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.95,
                "anchor_label_strength": "strong",
                "observation_layer": "fine",
                "refinement_key": "5:9",
            },
        )

        summary = module.replace_observations(state, [fine_patch])

        obj = state.objects[0]
        assert summary["replaced_observation_count"] == 1
        assert summary["inserted_observation_count"] == 1
        assert len(obj.observations) == 1
        assert obj.observations[0].observation_layer == "fine"
        assert obj.update_count == 1
        assert obj.last_seen_frame == 5
        assert len(obj.local_pcd) == 2
        np.testing.assert_allclose(obj.centroid, fine_points.mean(axis=0))
        np.testing.assert_allclose(obj.bbox_min, fine_points.min(axis=0))
        np.testing.assert_allclose(obj.bbox_max, fine_points.max(axis=0))
        assert obj.debug["anchor_semantics"]["label_frame_hits"] == {"sofa": 1}
        assert obj.debug["local_geometry_memory"]["point_count"] == 2
        assert obj.debug["last_async_refinement"]["replaced_observation_count"] == 1
        assert summary["updated_object_ids"] == [0]

    def test_object_update_updates_association_geometry_incrementally_from_patch(self, monkeypatch):
        import src.modules.object_update as object_update_module

        module = ObjectUpdateModule(
            {
                "downsample_interval": 100,
                "max_points_per_object": 4096,
                "association_geometry": {
                    "enabled": True,
                    "sketch_enabled": True,
                    "voxel_size": 0.03,
                    "max_points_per_object": 8,
                },
            }
        )
        existing_points = np.array(
            [[float(i) * 0.01, 0.0, 0.0] for i in range(1000)],
            dtype=np.float32,
        )
        existing_association_points = np.array(
            [[float(i), 1.0, 0.0] for i in range(6)],
            dtype=np.float32,
        )
        patch_points = np.array(
            [[10.0 + float(i) * 0.01, 0.0, 0.0] for i in range(4)],
            dtype=np.float32,
        )
        obj = ObjectMap(
            object_id=3,
            local_pcd=existing_points.copy(),
            association_pcd=existing_association_points.copy(),
            centroid=existing_points.mean(axis=0),
            bbox_min=existing_points.min(axis=0),
            bbox_max=existing_points.max(axis=0),
            update_count=1,
        )
        patch = Patch3D(
            patch_id=44,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
            source_frame_id=2,
        )
        downsample_input_lengths = []

        def recording_voxel_downsample(points, voxel_size):
            downsample_input_lengths.append(len(points))
            return np.asarray(points, dtype=np.float32).copy()

        monkeypatch.setattr(object_update_module, "voxel_downsample", recording_voxel_downsample)

        module._update_object(obj, patch)

        assert downsample_input_lengths == [len(patch_points)]
        assert len(obj.local_pcd) == len(existing_points) + len(patch_points)
        assert len(obj.association_pcd) <= 8
        assert obj.debug["association_geometry"]["point_count"] == len(obj.association_pcd)
        assert obj.debug["association_geometry"]["source_point_count"] == len(obj.local_pcd)
        assert obj.debug["association_geometry"]["role"] == "bounded_association_geometry"
        assert obj.debug["association_geometry"]["max_points_per_object"] == 8
        assert obj.debug["association_geometry"]["voxel_size"] == pytest.approx(0.03)
        assert obj.debug["association_geometry"]["sketch_enabled"] is True
        assert obj.debug["association_geometry"]["sketch_key_count"] >= len(obj.association_pcd)
        assert obj.debug["association_geometry"]["association_geometry_revision"] >= 1
        assert "sketch_point_by_key" not in obj.debug["association_geometry"]
        assert obj.debug["local_geometry_pool"]["incremental_bounds_used"] is True
        np.testing.assert_allclose(obj.centroid, obj.local_pcd.mean(axis=0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(obj.bbox_min, obj.local_pcd.min(axis=0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(obj.bbox_max, obj.local_pcd.max(axis=0), rtol=1e-6, atol=1e-6)

    def test_object_update_chunk_pool_defers_local_pcd_concat_until_flush(self):
        module = ObjectUpdateModule(
            {
                "downsample_interval": 100,
                "max_points_per_object": 4096,
                "local_pcd_chunk_pool": {
                    "enabled": True,
                    "bounded_compaction_enabled": True,
                    "materialize_interval": 100,
                    "max_pending_points": 1000,
                    "compact_to_max_points": True,
                },
                "association_geometry": {
                    "enabled": True,
                    "voxel_size": 0.03,
                    "max_points_per_object": 32,
                },
            }
        )
        existing_points = np.array(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
            dtype=np.float32,
        )
        patch_points = np.array(
            [[1.0, 0.0, 1.0], [1.1, 0.0, 1.0], [1.2, 0.0, 1.0]],
            dtype=np.float32,
        )
        obj = ObjectMap(
            object_id=3,
            local_pcd=existing_points.copy(),
            centroid=existing_points.mean(axis=0),
            bbox_min=existing_points.min(axis=0),
            bbox_max=existing_points.max(axis=0),
            update_count=1,
        )
        patch = Patch3D(
            patch_id=44,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
            source_frame_id=2,
        )

        module._update_object(obj, patch)

        assert len(obj.local_pcd) == len(existing_points)
        assert module._local_pcd_total_count(obj) == len(existing_points) + len(patch_points)
        assert obj.debug["local_geometry_pool"]["append_only_update"] is True
        assert obj.debug["local_geometry_pool"]["materialized_this_update"] is False
        assert obj.debug["local_geometry_pool"]["pending_point_count"] == len(patch_points)
        assert obj.debug["local_geometry_pool"]["local_pcd_pending_point_peak"] == len(patch_points)
        np.testing.assert_allclose(
            obj.centroid,
            np.concatenate([existing_points, patch_points], axis=0).mean(axis=0),
            rtol=1e-6,
            atol=1e-6,
        )
        state = SystemState(objects={3: obj})

        flushed = module.flush_deferred_geometry(state)

        assert flushed == 1
        assert len(obj.local_pcd) == len(existing_points) + len(patch_points)
        assert len(obj.local_pcd) <= module.max_points
        assert obj.debug["local_geometry_pool"]["pending_point_count"] == 0
        assert obj.debug["local_geometry_pool"]["flushed_for_export"] is True
        assert "_local_pcd_chunk_pool" not in obj.debug
        assert obj.debug["association_geometry"]["source_point_count"] == len(obj.local_pcd)

    def test_object_update_chunk_pool_compacts_and_caps_when_pending_threshold_exceeded(self):
        module = ObjectUpdateModule(
            {
                "downsample_interval": 100,
                "max_points_per_object": 3,
                "local_pcd_chunk_pool": {
                    "enabled": True,
                    "bounded_compaction_enabled": True,
                    "materialize_interval": 100,
                    "max_pending_points": 2,
                    "compact_to_max_points": True,
                },
                "association_geometry": {"enabled": False},
            }
        )
        existing_points = np.array(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
            dtype=np.float32,
        )
        patch_points = np.array([[1.0, 0.0, 1.0], [1.1, 0.0, 1.0]], dtype=np.float32)
        obj = ObjectMap(
            object_id=3,
            local_pcd=existing_points.copy(),
            centroid=existing_points.mean(axis=0),
            bbox_min=existing_points.min(axis=0),
            bbox_max=existing_points.max(axis=0),
            update_count=1,
        )
        patch = Patch3D(
            patch_id=44,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
            source_frame_id=2,
        )

        module._update_object(obj, patch)

        assert len(obj.local_pcd) <= 3
        assert module._local_pcd_total_count(obj) <= 3
        assert obj.debug["local_geometry_pool"]["append_only_update"] is False
        assert obj.debug["local_geometry_pool"]["materialized_this_update"] is True
        assert obj.debug["local_geometry_pool"]["pending_point_count"] == 0
        assert obj.debug["local_geometry_pool"]["local_pcd_compaction_count"] == 1
        assert obj.debug["local_geometry_pool"]["local_pcd_compacted_point_count_before"] == 5
        assert obj.debug["local_geometry_pool"]["local_pcd_compacted_point_count_after"] <= 3
        assert obj.debug["local_geometry_pool"]["local_pcd_cap_applied_count"] == 1

    def test_object_update_chunk_pool_flush_caps_existing_materialized_local_pcd(self):
        module = ObjectUpdateModule(
            {
                "downsample_interval": 100,
                "max_points_per_object": 4,
                "local_pcd_chunk_pool": {
                    "enabled": True,
                    "bounded_compaction_enabled": True,
                    "materialize_interval": 100,
                    "max_pending_points": 1000,
                    "compact_to_max_points": True,
                },
                "association_geometry": {"enabled": False},
            }
        )
        points = np.array([[float(i), 0.0, 1.0] for i in range(8)], dtype=np.float32)
        obj = ObjectMap(
            object_id=3,
            local_pcd=points.copy(),
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            update_count=1,
        )

        module.flush_deferred_geometry(SystemState(objects={3: obj}))

        assert len(obj.local_pcd) <= 4
        assert obj.debug["local_geometry_pool"]["pending_point_count"] == 0
        assert obj.debug["local_geometry_pool"]["local_pcd_cap_applied_count"] == 1

    def test_object_update_voxel_pool_updates_index_without_online_materialize(self):
        module = ObjectUpdateModule(
            {
                "downsample_interval": 100,
                "max_points_per_object": 4,
                "local_pcd_voxel_pool": {
                    "enabled": True,
                    "voxel_size": 0.1,
                    "max_keys_per_object": 4,
                    "materialize_on_interval": False,
                },
                "association_geometry": {"enabled": False},
            }
        )
        existing_points = np.array(
            [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]],
            dtype=np.float32,
        )
        patch_points = np.array(
            [[0.01, 0.0, 1.0], [0.3, 0.0, 1.0], [0.4, 0.0, 1.0]],
            dtype=np.float32,
        )
        obj = ObjectMap(
            object_id=3,
            local_pcd=existing_points.copy(),
            centroid=existing_points.mean(axis=0),
            bbox_min=existing_points.min(axis=0),
            bbox_max=existing_points.max(axis=0),
            update_count=1,
        )
        patch = Patch3D(
            patch_id=44,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
            source_frame_id=2,
        )

        module._update_object(obj, patch)

        assert len(obj.local_pcd) == len(existing_points)
        assert module._local_pcd_total_count(obj) == 4
        assert obj.debug["local_geometry_pool"]["local_pcd_voxel_pool_enabled"] is True
        assert obj.debug["local_geometry_pool"]["local_pcd_voxel_key_count"] == 4
        assert obj.debug["local_geometry_pool"]["local_pcd_voxel_insert_count"] == 2
        assert obj.debug["local_geometry_pool"]["local_pcd_voxel_update_count"] == 1
        assert obj.debug["local_geometry_pool"]["local_pcd_snapshot_stale"] is True
        assert obj.debug["local_geometry_pool"]["local_pcd_compaction_count"] == 0

        flushed = module.flush_deferred_geometry(SystemState(objects={3: obj}))

        assert flushed == 1
        assert len(obj.local_pcd) == 4
        assert obj.debug["local_geometry_pool"]["flushed_for_export"] is True
        assert obj.debug["local_geometry_pool"]["local_pcd_snapshot_stale"] is False
        assert "_local_pcd_voxel_pool" not in obj.debug

    def test_association_geometry_sketch_materializes_on_interval_and_flush(self):
        module = ObjectUpdateModule(
            {
                "downsample_interval": 100,
                "max_points_per_object": 4096,
                "local_pcd_chunk_pool": {
                    "enabled": True,
                    "bounded_compaction_enabled": True,
                    "materialize_interval": 100,
                    "max_pending_points": 1000,
                    "compact_to_max_points": True,
                },
                "association_geometry": {
                    "enabled": True,
                    "sketch_enabled": True,
                    "materialize_interval": 20,
                    "patch_budget_points": 1,
                    "update_budget_new_keys": 1,
                    "voxel_size": 0.03,
                    "max_points_per_object": 2,
                },
            }
        )
        existing_points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        existing_association_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        patch_points = np.array([[1.0, 0.0, 1.0], [1.1, 0.0, 1.0]], dtype=np.float32)
        obj = ObjectMap(
            object_id=3,
            local_pcd=existing_points.copy(),
            association_pcd=existing_association_points.copy(),
            centroid=existing_points.mean(axis=0),
            bbox_min=existing_points.min(axis=0),
            bbox_max=existing_points.max(axis=0),
            update_count=1,
        )
        patch = Patch3D(
            patch_id=44,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
            source_frame_id=2,
        )

        module._update_object(obj, patch)

        np.testing.assert_allclose(obj.association_pcd, existing_association_points)
        assert obj.debug["association_geometry"]["materialized_from_sketch"] is False
        assert obj.debug["association_geometry"]["sketch_pending_update_count"] == 1
        assert obj.debug["association_geometry"]["sketch_key_count"] == 2
        assert obj.debug["association_geometry"]["association_geometry_sketch_update_count"] == 1
        assert obj.debug["association_geometry"]["association_geometry_sketch_new_key_count"] == 1

        state = SystemState(objects={3: obj})
        flushed = module.flush_deferred_geometry(state)

        assert flushed == 1
        assert len(obj.local_pcd) == len(existing_points) + len(patch_points)
        assert len(obj.association_pcd) == obj.debug["association_geometry"]["sketch_key_count"]
        assert len(obj.association_pcd) <= 2
        assert obj.debug["association_geometry"]["association_geometry_sketch_materialize_count"] == 1
        assert obj.debug["association_geometry"]["materialized_from_sketch"] is True
        assert obj.debug["association_geometry"]["sketch_pending_update_count"] == 0
        assert obj.debug["association_geometry"]["source_point_count"] == len(obj.local_pcd)

    def test_object_debug_refresh_preserves_association_sketch_counters(self):
        module = ObjectUpdateModule(
            {
                "association_geometry": {
                    "enabled": True,
                    "sketch_enabled": True,
                    "voxel_size": 0.03,
                    "max_points_per_object": 2,
                },
            }
        )
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        obj = ObjectMap(
            object_id=3,
            local_pcd=points.copy(),
            association_pcd=points[:1].copy(),
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            update_count=1,
        )
        obj.debug["association_geometry"] = {
            "association_geometry_revision": 4,
            "sketch_key_count": 2,
            "sketch_pending_update_count": 1,
            "sketch_update_count": 5,
            "sketch_new_key_count_total": 7,
            "sketch_dropped_key_count_total": 2,
            "sketch_materialize_count": 3,
            "sketch_update_sec": 0.4,
            "sketch_materialize_sec": 0.5,
            "association_geometry_sketch_update_count": 5,
            "association_geometry_sketch_new_key_count": 7,
            "association_geometry_sketch_dropped_key_count": 2,
            "association_geometry_sketch_materialize_count": 3,
            "association_geometry_sketch_update_sec": 0.4,
            "association_geometry_sketch_materialize_sec": 0.5,
            "materialized_from_sketch": False,
        }
        state = SystemState(objects={3: obj}, tsdf_volume=TSDFInstanceVolume(voxel_size=0.05))

        module._refresh_object_debug(obj, state.tsdf_volume, current_frame=10)

        debug = obj.debug["association_geometry"]
        assert debug["source_point_count"] == len(points)
        assert debug["association_geometry_revision"] == 4
        assert debug["association_geometry_sketch_update_count"] == 5
        assert debug["association_geometry_sketch_new_key_count"] == 7
        assert debug["association_geometry_sketch_dropped_key_count"] == 2
        assert debug["association_geometry_sketch_materialize_count"] == 3
        assert debug["association_geometry_sketch_update_sec"] == pytest.approx(0.4)
        assert debug["association_geometry_sketch_materialize_sec"] == pytest.approx(0.5)
        assert debug["materialized_from_sketch"] is False

    def test_replace_observations_can_be_disabled_by_planned_config_key(self):
        module = ObjectUpdateModule(
            {
                "surface_owner_gate": {"enabled": False},
                "provisional": {"enabled": False},
                "async_refinement": {"replace_coarse_observations": False},
            }
        )
        state = SystemState()
        coarse_points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        coarse_patch = Patch3D(
            patch_id=1,
            points=coarse_points,
            centroid=coarse_points.mean(axis=0),
            bbox_min=coarse_points.min(axis=0),
            bbox_max=coarse_points.max(axis=0),
            source_frame_id=5,
            metadata={
                "source_proposal_id": 1,
                "source_backend_name": "anchor_box_primary",
                "anchor_id": 9,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.9,
                "anchor_label_strength": "strong",
                "observation_layer": "coarse",
                "refinement_key": "5:9",
            },
        )
        state = module.process(AssociationResult(new_object_patches=[1]), [coarse_patch], state)

        fine_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        fine_patch = Patch3D(
            patch_id=2,
            points=fine_points,
            centroid=fine_points.mean(axis=0),
            bbox_min=fine_points.min(axis=0),
            bbox_max=fine_points.max(axis=0),
            source_frame_id=5,
            metadata={
                "source_proposal_id": 2,
                "source_backend_name": "async_sam_refinement",
                "anchor_id": 9,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.95,
                "anchor_label_strength": "strong",
                "observation_layer": "fine",
                "refinement_key": "5:9",
            },
        )

        summary = module.replace_observations(state, [fine_patch])

        obj = state.objects[0]
        assert summary["enabled"] is False
        assert summary["replaced_observation_count"] == 0
        assert summary["inserted_observation_count"] == 0
        assert summary["updated_object_ids"] == []
        assert len(obj.observations) == 1
        assert obj.observations[0].observation_layer == "coarse"
        np.testing.assert_allclose(obj.local_pcd, coarse_points)

    def test_replace_observations_counts_unique_updated_objects(self):
        module = ObjectUpdateModule(
            {
                "surface_owner_gate": {"enabled": False},
                "provisional": {"enabled": False},
            }
        )
        state = SystemState()
        patch1_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        patch2_points = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
        coarse_patch1 = Patch3D(
            patch_id=1,
            points=patch1_points,
            centroid=patch1_points.mean(axis=0),
            bbox_min=patch1_points.min(axis=0),
            bbox_max=patch1_points.max(axis=0),
            source_frame_id=5,
            metadata={
                "source_proposal_id": 1,
                "anchor_id": 9,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.9,
                "anchor_label_strength": "strong",
                "observation_layer": "coarse",
                "refinement_key": "5:9",
            },
        )
        coarse_patch2 = Patch3D(
            patch_id=2,
            points=patch2_points,
            centroid=patch2_points.mean(axis=0),
            bbox_min=patch2_points.min(axis=0),
            bbox_max=patch2_points.max(axis=0),
            source_frame_id=6,
            metadata={
                "source_proposal_id": 2,
                "anchor_id": 10,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.88,
                "anchor_label_strength": "strong",
                "observation_layer": "coarse",
                "refinement_key": "6:10",
            },
        )
        state = module.process(AssociationResult(new_object_patches=[1]), [coarse_patch1], state)
        state = module.process(
            AssociationResult(matched=[(2, 0, AssociationScore(total_score=0.9))]),
            [coarse_patch2],
            state,
        )

        fine_patch1 = Patch3D(
            patch_id=3,
            points=patch1_points + np.array([[0.0, 0.1, 0.0]], dtype=np.float32),
            centroid=(patch1_points + np.array([[0.0, 0.1, 0.0]], dtype=np.float32)).mean(axis=0),
            bbox_min=(patch1_points + np.array([[0.0, 0.1, 0.0]], dtype=np.float32)).min(axis=0),
            bbox_max=(patch1_points + np.array([[0.0, 0.1, 0.0]], dtype=np.float32)).max(axis=0),
            source_frame_id=5,
            metadata={
                "source_proposal_id": 3,
                "anchor_id": 9,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.95,
                "anchor_label_strength": "strong",
                "observation_layer": "fine",
                "refinement_key": "5:9",
            },
        )
        fine_patch2 = Patch3D(
            patch_id=4,
            points=patch2_points + np.array([[0.0, 0.1, 0.0]], dtype=np.float32),
            centroid=(patch2_points + np.array([[0.0, 0.1, 0.0]], dtype=np.float32)).mean(axis=0),
            bbox_min=(patch2_points + np.array([[0.0, 0.1, 0.0]], dtype=np.float32)).min(axis=0),
            bbox_max=(patch2_points + np.array([[0.0, 0.1, 0.0]], dtype=np.float32)).max(axis=0),
            source_frame_id=6,
            metadata={
                "source_proposal_id": 4,
                "anchor_id": 10,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.96,
                "anchor_label_strength": "strong",
                "observation_layer": "fine",
                "refinement_key": "6:10",
            },
        )

        summary = module.replace_observations(state, [fine_patch1, fine_patch2])

        assert summary["replaced_observation_count"] == 2
        assert summary["inserted_observation_count"] == 2
        assert summary["updated_object_count"] == 1
        assert summary["updated_object_ids"] == [0]

    def test_replace_observations_skips_multi_component_refinement_by_default(self):
        module = ObjectUpdateModule(
            {
                "surface_owner_gate": {"enabled": False},
                "provisional": {"enabled": False},
            }
        )
        state = SystemState()
        coarse_points = np.array(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
            dtype=np.float32,
        )
        coarse_patch = Patch3D(
            patch_id=1,
            points=coarse_points,
            centroid=coarse_points.mean(axis=0),
            bbox_min=coarse_points.min(axis=0),
            bbox_max=coarse_points.max(axis=0),
            source_frame_id=5,
            metadata={
                "source_proposal_id": 1,
                "anchor_id": 9,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.9,
                "anchor_label_strength": "strong",
                "observation_layer": "coarse",
                "refinement_key": "5:9",
            },
        )
        state = module.process(AssociationResult(new_object_patches=[1]), [coarse_patch], state)

        fine_a = np.array([[0.0, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
        fine_b = np.array([[0.2, 0.2, 1.0]], dtype=np.float32)
        fine_patches = []
        for patch_id, points in [(2, fine_a), (3, fine_b)]:
            fine_patches.append(
                Patch3D(
                    patch_id=patch_id,
                    points=points,
                    centroid=points.mean(axis=0),
                    bbox_min=points.min(axis=0),
                    bbox_max=points.max(axis=0),
                    source_frame_id=5,
                    metadata={
                        "source_proposal_id": patch_id,
                        "anchor_id": 9,
                        "anchor_class_name": "sofa",
                        "anchor_confidence": 0.95,
                        "anchor_label_strength": "strong",
                        "observation_layer": "fine",
                        "refinement_key": "5:9",
                    },
                )
            )

        summary = module.replace_observations(state, fine_patches)

        obj = state.objects[0]
        assert summary["fine_patch_count"] == 2
        assert summary["skipped_fine_patch_count"] == 2
        assert summary["replaced_observation_count"] == 0
        assert summary["inserted_observation_count"] == 0
        assert len(obj.observations) == 1
        assert obj.observations[0].observation_layer == "coarse"


def test_surface_owner_gate_representative_mode_uses_unique_voxel_counts():
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "representative_voxel_mode": True,
                "min_accept_points": 1,
                "min_update_accept_ratio": 0.5,
                "max_foreign_owner_ratio": 0.5,
                "max_background_owner_ratio": 1.0,
            },
            "tsdf": {"voxel_size": 1.0},
        }
    )
    state = SystemState(tsdf_volume=TSDFInstanceVolume(voxel_size=1.0))
    state.tsdf_volume.owner_support[(0, 0, 0)] = VoxelOwnerSupport({7: 1.0})
    state.tsdf_volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({3: 1.0})
    points = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.3, 0.1, 0.1],
            [1.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    patch = Patch3D(
        patch_id=1,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
    )

    filtered, structural, debug, voxel_view = module._filter_patch_by_surface_owner(
        patch,
        target_object_id=7,
        state=state,
        background_support=set(),
        new_object=False,
    )

    assert filtered is not None
    assert structural is None
    assert debug["representative_voxel_mode"] is True
    assert debug["source_point_count"] == 4
    assert debug["decision_voxel_count"] == 2
    assert debug["foreign_owner_point_count"] == 1
    assert debug["foreign_owner_decision_voxel_count"] == 1
    assert debug["accepted_decision_voxel_count"] == 1
    assert debug["accepted_point_count"] == 3
    assert debug["foreign_owner_ratio"] == pytest.approx(0.5)
    assert debug["accepted_ratio"] == pytest.approx(0.5)
    assert debug["min_accept_count_unit"] == "decision_voxel"


def test_surface_owner_gate_representative_mode_min_accept_uses_unique_voxels():
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "representative_voxel_mode": True,
                "min_accept_points": 2,
                "min_update_accept_ratio": 1.0,
                "max_foreign_owner_ratio": 1.0,
                "max_background_owner_ratio": 1.0,
            },
            "tsdf": {"voxel_size": 1.0},
        }
    )
    state = SystemState(tsdf_volume=TSDFInstanceVolume(voxel_size=1.0))
    points = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.3, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    patch = Patch3D(
        patch_id=2,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
    )

    filtered, structural, debug, voxel_view = module._filter_patch_by_surface_owner(
        patch,
        target_object_id=7,
        state=state,
        background_support=set(),
        new_object=False,
    )

    assert filtered is None
    assert structural is None
    assert debug["accepted_point_count"] == 3
    assert debug["accepted_decision_voxel_count"] == 1
    assert debug["min_accept_points"] == 2
    assert debug["min_accept_count_unit"] == "decision_voxel"
    assert "insufficient_accepted_points" in debug["rejection_reasons"]


def test_surface_owner_gate_point_mode_min_accept_uses_expanded_points():
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "representative_voxel_mode": False,
                "min_accept_points": 2,
                "min_update_accept_ratio": 1.0,
                "max_foreign_owner_ratio": 1.0,
                "max_background_owner_ratio": 1.0,
            },
            "tsdf": {"voxel_size": 1.0},
        }
    )
    state = SystemState(tsdf_volume=TSDFInstanceVolume(voxel_size=1.0))
    points = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.3, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    patch = Patch3D(
        patch_id=3,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
    )

    filtered, structural, debug, voxel_view = module._filter_patch_by_surface_owner(
        patch,
        target_object_id=7,
        state=state,
        background_support=set(),
        new_object=False,
    )

    assert filtered is not None
    assert structural is None
    assert debug["accepted_point_count"] == 3
    assert debug["accepted_decision_voxel_count"] == 3
    assert debug["min_accept_points"] == 2
    assert debug["min_accept_count_unit"] == "point"
    assert debug["rejection_reasons"] == []


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

    def test_semantic_memory_module_applies_anchor_commit_config(self):
        from src.modules.semantic_memory import set_anchor_commit_policy, set_anchor_export_policy

        try:
            module = SemanticMemoryModule(
                {
                    "backend": "placeholder",
                    "anchor_commit": {
                        "min_frame_hits": 3,
                        "min_high_quality_hits": 2,
                        "min_weighted_score": 2.5,
                        "min_score_margin": 0.4,
                        "provisional_export_fallback": True,
                    },
                    "anchor_export": {
                        "enabled": True,
                        "min_frame_hits": 1,
                        "min_high_quality_hits": 1,
                        "min_weighted_score": 0.8,
                        "min_score_margin": 0.06,
                        "min_confidence": 0.76,
                        "min_view_quality": 0.30,
                        "allow_single_frame_high_confidence": True,
                        "single_frame_min_confidence": 0.91,
                        "single_frame_min_view_quality": 0.55,
                    },
                }
            )

            assert module.anchor_commit_min_frame_hits == 3
            assert module.anchor_commit_min_high_quality_hits == 2
            assert module.anchor_commit_min_weighted_score == 2.5
            assert module.anchor_commit_min_score_margin == 0.4
            assert module.anchor_provisional_export_fallback is True
            assert module.anchor_export_enabled is True
            assert module.anchor_export_min_frame_hits == 1
            assert module.anchor_export_min_high_quality_hits == 1
            assert module.anchor_export_min_weighted_score == 0.8
            assert module.anchor_export_min_score_margin == 0.06
            assert module.anchor_export_min_confidence == 0.76
            assert module.anchor_export_min_view_quality == 0.30
            assert module.anchor_export_allow_single_frame_high_confidence is True
            assert module.anchor_export_single_frame_min_confidence == 0.91
            assert module.anchor_export_single_frame_min_view_quality == 0.55
        finally:
            set_anchor_commit_policy(None)
            set_anchor_export_policy(None)

    def test_semantic_memory_anchor_commit_config_controls_runtime_policy(self):
        from src.modules.semantic_memory import object_semantic_commit_state, set_anchor_commit_policy

        try:
            SemanticMemoryModule(
                {
                    "backend": "placeholder",
                    "anchor_commit": {
                        "min_frame_hits": 3,
                        "min_high_quality_hits": 1,
                        "min_weighted_score": 1.0,
                        "min_score_margin": 0.15,
                        "provisional_export_fallback": False,
                    },
                }
            )
            obj = ObjectMap(object_id=50)
            points = np.repeat(
                np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
                128,
                axis=0,
            )
            for frame_id in [1, 2]:
                patch = Patch3D(
                    patch_id=frame_id,
                    points=points,
                    centroid=points.mean(axis=0),
                    bbox_min=points.min(axis=0),
                    bbox_max=points.max(axis=0),
                    source_frame_id=frame_id,
                    metadata={
                        "anchor_class_name": "rug",
                        "anchor_confidence": 0.90,
                        "anchor_view_quality": 0.90,
                        "anchor_label_strength": "strong",
                    },
                )
                accumulate_anchor_semantic_vote(obj, patch)

            assert object_semantic_commit_state(obj)["semantic_state"] == "provisional"
            assert preferred_object_semantic_label(obj) == ""
        finally:
            set_anchor_commit_policy(None)

    def test_background_map_defaults(self):
        bg = BackgroundMap()
        assert bg.voxel_size == 0.05
        assert len(bg.point_cloud) == 0

    def test_association_result(self):
        result = AssociationResult()
        assert len(result.matched) == 0
        assert len(result.new_object_patches) == 0

    def test_association_result_records_contested_matches(self):
        from src.core.data_structures import AssociationScore, ContestedAssociation

        score = AssociationScore(total_score=0.91)
        contested = ContestedAssociation(
            patch_id=8,
            blocked_object_id=3,
            patch_label="cushion",
            object_label="sofa",
            reason="cross_label_observation_identity",
            score=score,
        )
        result = AssociationResult(contested_matches=[contested], contested_object_patches=[8])

        assert result.matched == []
        assert result.new_object_patches == []
        assert result.contested_object_patches == [8]
        assert result.contested_matches[0].blocked_object_id == 3
        assert result.contested_matches[0].patch_label == "cushion"
        assert result.contested_matches[0].object_label == "sofa"
        assert result.contested_matches[0].score.total_score == 0.91

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

    def test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend(self):
        path = Path("configs/room0_surface_gate_fast_high_iou_4090.yaml")
        assert path.exists()

        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert config["anchor_frontend"]["enabled"] is True
        assert config["anchor_frontend"]["use_sam_intersection_proposals"] is True
        assert config["anchor_frontend"]["anchor_primary_mode"] is False
        assert config["anchor_guided_sam"]["enabled"] is False
        assert config["pipeline"]["proposal_prefetch_enabled"] is True
        assert config["pipeline"]["proposal_prefetch_wait_timeout_sec"] == -1.0
        assert config["depth_refinement"]["proposal_parallel_enabled"] is True
        assert config["depth_refinement"]["proposal_parallel_workers"] == 4
        assert config["depth_refinement"]["proposal_parallel_min_tasks"] == 16
        assert config["association"]["max_geometry_candidates"] == 12
        assert config["association"]["geometry_candidate_min_cheap_score"] == 0.02
        assert config["association"]["geometry_vote_owner_priority"] is True
        assert config["association"]["geometry_nn_patch_sample"] == 96
        assert config["association"]["geometry_nn_object_sample"] == 96
        assert config["association"]["score_parallel_enabled"] is True
        assert config["association"]["score_parallel_workers"] == 4
        assert config["association"]["score_parallel_min_candidates"] == 16
        assert config["object_update"]["association_geometry"]["enabled"] is True
        assert config["object_update"]["association_geometry"]["voxel_size"] == 0.03
        assert config["object_update"]["association_geometry"]["max_points_per_object"] == 2048
        assert config["semantic_memory"]["anchor_export"]["enabled"] is True
        anchor_export = config["semantic_memory"]["anchor_export"]
        assert anchor_export["repeated_evidence_enabled"] is True
        assert anchor_export["repeated_min_frame_hits"] == 3
        assert anchor_export["repeated_min_weighted_score"] == 1.0
        assert anchor_export["repeated_min_score_margin"] == 0.25
        assert anchor_export["repeated_min_view_quality"] == 0.50
        assert anchor_export["repeated_min_confidence"] == 0.25
        supplemental = config["anchor_frontend"]["supplemental"]
        assert supplemental["worker_enabled"] is True
        assert supplemental["worker_fallback_on_error"] is True
        assert supplemental["worker_request_timeout_sec"] == 120.0
        assert config["patch_lifting"]["point_sample_ratio"] == 1.0
        assert config["patch_lifting"]["max_points_per_patch"] == 0
        assert config["object_update"]["downsample_voxel_size"] == 0.01
        assert config["object_update"]["downsample_interval"] == 20
        assert config["object_update"]["max_points_per_object"] == 40000
        assert config["pipeline"]["verbose"] is False
        assert config["pipeline"]["collect_stage_timings"] is True

    def test_scheduling_report_payload_sums_prefetch_metrics(self):
        payload = build_scheduling_report_payload(
            [
                {
                    "scheduling": {
                        "prefetch_enabled": True,
                        "frontend_source": "prefetch",
                        "prefetch_submitted": True,
                        "prefetch_hit": True,
                        "prefetch_miss": False,
                        "prefetch_wait_sec": 0.25,
                        "prefetch_exception": "",
                    }
                },
                {
                    "scheduling": {
                        "prefetch_enabled": True,
                        "frontend_source": "inline",
                        "prefetch_submitted": True,
                        "prefetch_hit": False,
                        "prefetch_miss": True,
                        "prefetch_wait_sec": 0.50,
                        "prefetch_exception": "RuntimeError: bad frame",
                    }
                },
                {
                    "association_summary": {
                        "score_parallel_used_count": 2,
                        "score_parallel_candidate_count_total": 20,
                    },
                    "depth_refinement": {
                        "parallel_used": True,
                        "parallel_worker_count": 4,
                    },
                },
                {
                    "association_summary": {
                        "score_parallel_used_count": 3,
                        "score_parallel_candidate_count_total": 30,
                    },
                    "depth_refinement": {
                        "parallel_used": False,
                        "parallel_worker_count": 1,
                    },
                },
            ]
        )

        assert payload["prefetch_enabled"] is True
        assert payload["prefetch_submitted_count"] == 2
        assert payload["prefetch_hit_count"] == 1
        assert payload["prefetch_miss_count"] == 1
        assert payload["prefetch_wait_sec_total"] == pytest.approx(0.75)
        assert payload["frontend_prefetch_frame_count"] == 1
        assert payload["frontend_inline_frame_count"] == 1
        assert payload["frontend_exception_count"] == 1
        assert payload["frontend_last_exception"] == "RuntimeError: bad frame"
        assert payload["association_score_parallel_used_count"] == 5
        assert payload["association_score_parallel_candidate_count_total"] == 50
        assert payload["depth_refinement_parallel_used_count"] == 1
        assert payload["depth_refinement_parallel_worker_count_max"] == 4

    def test_stage_timing_summary_includes_percentiles_and_outliers(self):
        summary = build_stage_timing_summary(
            [
                {"frame_id": 0, "stage_timings": {"association": 1.0}},
                {"frame_id": None, "stage_timings": {"bad_frame_id_stage": 0.5}},
                {"frame_id": 10, "stage_timings": {"association": 2.0}},
                {"frame_id": 20, "stage_timings": {"association": 10.0}},
            ]
        )

        assoc = summary["association"]
        assert assoc["mean_sec"] == pytest.approx(13.0 / 3.0)
        assert assoc["median_sec"] == pytest.approx(2.0)
        assert assoc["p95_sec"] == pytest.approx(10.0)
        assert assoc["max_sec"] == pytest.approx(10.0)
        assert assoc["max_frame_id"] == 20
        assert assoc["outlier_frame_ids"] == [20]
        assert summary["bad_frame_id_stage"]["max_frame_id"] == -1

    def test_stage_timing_summary_includes_prefetched_proposal_generation(self):
        summary = build_stage_timing_summary(
            [
                {
                    "frame_id": 5,
                    "stage_timings": {
                        "proposal_generation": 0.75,
                        "yoloe_supplemental": 0.25,
                    },
                }
            ]
        )

        assert summary["proposal_generation"]["total_sec"] == pytest.approx(0.75)
        assert summary["proposal_generation"]["max_frame_id"] == 5

    def test_build_prefetch_pipeline_uses_isolated_pipeline_and_runner_backend(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "pipeline:",
                    "  verbose: true",
                    "proposal:",
                    "  backend: placeholder",
                    "  min_mask_area: 5",
                ]
            ),
            encoding="utf-8",
        )
        main_pipe = Pipeline(config_path=str(config_path))
        manifest_path = write_manifest(
            tmp_path / "cache",
            dataset_summary={"frame_count": 0},
            backend_config={"backend": "precomputed"},
            frame_entries=[],
        )
        args = SimpleNamespace(
            proposal_backend="precomputed",
            min_mask_area=17,
            proposal_cache_dir=tmp_path / "cache",
            proposal_cache_manifest=manifest_path,
        )

        prefetch_pipe = build_prefetch_pipeline(config_path, args)

        assert prefetch_pipe is not main_pipe
        assert prefetch_pipe.verbose is False
        assert prefetch_pipe._build_proposal_bundle.__self__ is prefetch_pipe
        assert prefetch_pipe.proposal.active_backend_name == "precomputed"
        assert prefetch_pipe.proposal.config["min_mask_area"] == 17
        assert prefetch_pipe.proposal.config["precomputed"]["cache_dir"] == str(tmp_path / "cache")
        assert prefetch_pipe.proposal.config["precomputed"]["manifest_path"] == str(manifest_path)

    def test_process_frame_consumes_proposal_bundle_without_calling_frontend(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "pipeline:",
                    "  verbose: false",
                    "  collect_stage_timings: true",
                    "anchor_frontend:",
                    "  enabled: false",
                    "proposal:",
                    "  backend: placeholder",
                ]
            ),
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))

        class FailingProposal:
            config = {}
            active_backend_name = "failing"

            def process(self, *args, **kwargs):
                raise AssertionError("proposal frontend should not run when bundle is supplied")

        pipe.proposal = FailingProposal()

        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0, width=4, height=4)
        bundle = FrameProposalBundle(
            frame_id=0,
            source_frame_id=123,
            source_proposals=[],
            raw_proposals=[],
            anchors=[],
            anchor_assignments=[],
            proposal_source="prefetched_empty",
            anchor_guided_sam_summary={"enabled": False, "assignment_count": 0},
            generation_timings={"proposal_generation": 0.75, "yoloe_supplemental": 0.25},
            actual_backend="precomputed",
        )

        state = pipe.process_frame(
            rgb,
            depth,
            np.eye(4, dtype=np.float64),
            intrinsics,
            timestamp=1.5,
            source_frame_id=123,
            proposal_bundle=bundle,
        )

        assert isinstance(state, SystemState)
        assert pipe.last_raw_proposals == []
        assert pipe.last_source_proposals == []
        assert pipe.last_anchors == []
        assert pipe.last_anchor_assignments == []
        assert pipe.last_frame_debug["proposal_source"] == "prefetched_empty"
        assert pipe.last_frame_debug["frontend_source"] == "prefetch"
        assert pipe.last_frame_debug["frontend_actual_backend"] == "precomputed"
        assert pipe.last_frame_debug["depth_refinement"]["parallel_used"] is False
        assert pipe.last_frame_debug["depth_refinement"]["parallel_worker_count"] == 1
        assert pipe.last_frame_debug["stage_timings"]["proposal_generation"] == pytest.approx(0.75)
        assert pipe.last_frame_debug["stage_timings"]["yoloe_supplemental"] == pytest.approx(0.25)

    def test_build_proposal_bundle_records_top_level_proposal_generation_timing(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "pipeline:",
                    "  verbose: false",
                    "  collect_stage_timings: true",
                    "anchor_frontend:",
                    "  enabled: true",
                    "  anchor_primary_mode: false",
                    "  use_sam_intersection_proposals: false",
                    "proposal:",
                    "  backend: placeholder",
                ]
            ),
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))

        class EmptyProposal:
            config = {}
            active_backend_name = "precomputed"

            def process(self, *args, **kwargs):
                return []

        pipe.proposal = EmptyProposal()

        def _fake_anchor_process(rgb, proposals):
            pipe.object_anchor.last_generation_timings = {"yoloe_supplemental": 0.25}
            return [], []

        pipe.object_anchor.process = _fake_anchor_process

        intrinsics = CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0, width=4, height=4)
        frame = Frame(
            frame_id=0,
            source_frame_id=123,
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=np.ones((4, 4), dtype=np.float32),
            pose=np.eye(4, dtype=np.float64),
            intrinsics=intrinsics,
        )

        bundle = pipe._build_proposal_bundle(frame)

        assert "proposal_generation" in bundle.generation_timings
        assert bundle.generation_timings["proposal_generation"] >= 0.0
        assert bundle.generation_timings["yoloe_supplemental"] == pytest.approx(0.25)

    def test_prefetched_bundle_blocks_structural_overlay_frontend_fallback(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "pipeline:",
                    "  verbose: false",
                    "anchor_frontend:",
                    "  enabled: false",
                    "proposal:",
                    "  backend: placeholder",
                    "structural_overlay:",
                    "  enabled: true",
                    "  classes: [wall]",
                    "depth_refinement:",
                    "  min_mask_area_after_refine: 1",
                    "patch_lifting:",
                    "  min_points: 1",
                ]
            ),
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))

        class FailingProposal:
            config = {}
            active_backend_name = "precomputed"

            def process(self, *args, **kwargs):
                raise AssertionError("structural overlay must not call proposal frontend for a bundle")

        pipe.proposal = FailingProposal()

        anchor = Anchor2D(4, np.array([1, 1, 3, 3], dtype=np.float32), "wall", 0.9)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        proposal = Proposal2D(
            proposal_id=4,
            mask=mask,
            bbox_xyxy=np.array([1, 1, 3, 3], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.9,
            metadata={"anchor_id": 4, "anchor_class_name": "wall"},
        )
        bundle = FrameProposalBundle(
            frame_id=0,
            source_frame_id=123,
            source_proposals=[],
            raw_proposals=[proposal],
            anchors=[anchor],
            anchor_assignments=[AnchorAssignment(4, 4, "wall", 0.9)],
            proposal_source="anchor_box_primary",
            actual_backend="precomputed",
        )

        pipe.process_frame(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4), dtype=np.float32),
            np.eye(4, dtype=np.float64),
            CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0, width=4, height=4),
            source_frame_id=123,
            proposal_bundle=bundle,
        )

        debug = pipe.last_frame_debug["structural_overlay"]
        assert debug["proposal_source"] == "unavailable"
        assert debug["skip_reason"] == "prefetched_bundle_without_source_proposals"

    def test_prefetched_bundle_blocks_async_refinement_frontend_fallback(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "pipeline:",
                    "  verbose: false",
                    "anchor_frontend:",
                    "  enabled: false",
                    "proposal:",
                    "  backend: placeholder",
                    "async_refinement:",
                    "  enabled: true",
                    "  delay_frames: 0",
                    "  proposal_min_area: 1",
                    "depth_refinement:",
                    "  min_mask_area_after_refine: 1",
                    "patch_lifting:",
                    "  min_points: 1",
                ]
            ),
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))

        class FailingProposal:
            config = {}
            active_backend_name = "precomputed"

            def process(self, *args, **kwargs):
                raise AssertionError("async refinement must not call proposal frontend for a bundle")

        pipe.proposal = FailingProposal()

        bundle = FrameProposalBundle(
            frame_id=0,
            source_frame_id=123,
            source_proposals=[],
            raw_proposals=[],
            anchors=[],
            anchor_assignments=[],
            proposal_source="prefetched_empty",
            actual_backend="precomputed",
        )

        pipe.process_frame(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4), dtype=np.float32),
            np.eye(4, dtype=np.float64),
            CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0, width=4, height=4),
            source_frame_id=123,
            proposal_bundle=bundle,
        )

        debug = pipe.last_frame_debug["async_refinement"]
        assert debug["proposal_source"] == "prefetched_source_proposals"
        assert debug["skip_reason"] == "prefetched_bundle_without_source_proposals"

    def test_process_frame_rejects_proposal_bundle_source_frame_mismatch(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "pipeline:",
                    "  verbose: false",
                    "anchor_frontend:",
                    "  enabled: false",
                    "proposal:",
                    "  backend: placeholder",
                ]
            ),
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))

        class FailingProposal:
            config = {}
            active_backend_name = "failing"

            def process(self, *args, **kwargs):
                raise AssertionError("proposal frontend should not run for mismatched bundle")

        pipe.proposal = FailingProposal()

        bundle = FrameProposalBundle(
            frame_id=0,
            source_frame_id=124,
            source_proposals=[],
            raw_proposals=[],
            anchors=[],
            anchor_assignments=[],
            proposal_source="prefetched_empty",
        )

        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0, width=4, height=4)

        with pytest.raises(ValueError, match="source_frame_id mismatch"):
            pipe.process_frame(
                rgb,
                depth,
                np.eye(4, dtype=np.float64),
                intrinsics,
                timestamp=1.5,
                source_frame_id=123,
                proposal_bundle=bundle,
            )

        assert pipe.last_raw_proposals == []

    def test_process_frame_rejects_proposal_bundle_missing_source_frame_id(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "pipeline:",
                    "  verbose: false",
                    "anchor_frontend:",
                    "  enabled: false",
                    "proposal:",
                    "  backend: placeholder",
                ]
            ),
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))

        class FailingProposal:
            config = {}
            active_backend_name = "failing"

            def process(self, *args, **kwargs):
                raise AssertionError("proposal frontend should not run for missing bundle source id")

        pipe.proposal = FailingProposal()

        bundle = FrameProposalBundle(
            frame_id=0,
            source_frame_id=None,
            source_proposals=[],
            raw_proposals=[],
            anchors=[],
            anchor_assignments=[],
            proposal_source="prefetched_empty",
        )

        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0, width=4, height=4)

        with pytest.raises(ValueError, match="source_frame_id mismatch"):
            pipe.process_frame(
                rgb,
                depth,
                np.eye(4, dtype=np.float64),
                intrinsics,
                timestamp=1.5,
                source_frame_id=123,
                proposal_bundle=bundle,
            )

        assert pipe.last_raw_proposals == []

    def test_frame_proposal_bundle_normalizes_mutable_inputs(self):
        proposal = Proposal2D(
            proposal_id=7,
            mask=np.ones((2, 2), dtype=bool),
            bbox_xyxy=np.array([0, 0, 2, 2], dtype=np.float32),
            area=4,
        )
        anchor = Anchor2D(
            anchor_id=3,
            bbox_xyxy=np.array([0, 0, 2, 2], dtype=np.float32),
            class_name="chair",
            confidence=0.9,
        )
        assignment = AnchorAssignment(proposal_id=7, anchor_id=3)
        source_proposals = [proposal]
        raw_proposals = [proposal]
        anchors = [anchor]
        assignments = [assignment]
        sam_summary = {"enabled": True, "assignment_count": 1}
        timings = {"proposal_generation": 0.75, "yoloe_supplemental": 0.25}

        bundle = FrameProposalBundle(
            frame_id=0,
            source_frame_id=123,
            source_proposals=source_proposals,
            raw_proposals=raw_proposals,
            anchors=anchors,
            anchor_assignments=assignments,
            proposal_source="prefetched",
            anchor_guided_sam_summary=sam_summary,
            generation_timings=timings,
            actual_backend="precomputed",
        )

        source_proposals.clear()
        raw_proposals.clear()
        anchors.clear()
        assignments.clear()
        sam_summary["assignment_count"] = 99
        timings["proposal_generation"] = 10.0
        timings["yoloe_supplemental"] = 20.0

        assert bundle.source_proposals == (proposal,)
        assert bundle.raw_proposals == (proposal,)
        assert bundle.anchors == (anchor,)
        assert bundle.anchor_assignments == (assignment,)
        assert bundle.anchor_guided_sam_summary == {"enabled": True, "assignment_count": 1}
        assert bundle.generation_timings == {"proposal_generation": 0.75, "yoloe_supplemental": 0.25}

    def test_yoloworld_sam_frontend_runs_sam_after_yolo_without_yolo_waiting(self, tmp_path):
        from src.pipelines.yoloworld_sam_frontend import build_yoloworld_sam_bundle

        events: list[str] = []
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        frame = Frame(
            frame_id=0,
            source_frame_id=100,
            rgb=rgb,
            depth=depth,
            pose=np.eye(4, dtype=np.float64),
            intrinsics=intrinsics,
        )
        anchor = Anchor2D(3, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.9)
        anchor_mask = np.zeros((8, 8), dtype=bool)
        anchor_mask[1:7, 1:7] = True
        anchor_proposal = Proposal2D(
            proposal_id=0,
            mask=anchor_mask,
            bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
            area=int(anchor_mask.sum()),
            confidence=0.9,
            backend_name="anchor_box_primary",
            metadata={"anchor_id": 3, "anchor_class_name": "sofa"},
        )
        anchor_assignment = AnchorAssignment(0, 3, "sofa", 0.9, keepalive=True)
        sam_proposal = Proposal2D(
            proposal_id=11,
            mask=anchor_mask.copy(),
            bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
            area=int(anchor_mask.sum()),
            confidence=1.0,
            backend_name="sam2",
        )

        class FakeAnchorFrontend:
            active_backend_name = "yoloworld"
            last_generation_timings = {}

            def generate_anchor_box_proposals(self, input_rgb, *, depth=None, intrinsics=None):
                events.append("yolo_start")
                self.last_generation_timings = {"yoloworld_primary": 0.12, "anchor_merge": 0.0}
                events.append("yolo_end")
                return [anchor], [anchor_proposal], [anchor_assignment]

        class FakeProposal:
            active_backend_name = "sam2"

            def process_for_anchors(self, input_rgb, input_depth, anchors, frame=None):
                events.append("sam_start")
                assert events[:2] == ["yolo_start", "yolo_end"]
                assert [int(item.anchor_id) for item in anchors] == [3]
                events.append("sam_end")
                return [sam_proposal]

        anchor_guided_sam = AnchorGuidedSAMModule(
            {
                "enabled": True,
                "proposal_min_area": 1,
                "include_anchor_box_fallbacks": True,
                "include_unknown_residuals": True,
            }
        )

        bundle = build_yoloworld_sam_bundle(
            frame=frame,
            object_anchor=FakeAnchorFrontend(),
            proposal=FakeProposal(),
            anchor_guided_sam=anchor_guided_sam,
            collect_stage_timings=True,
        )

        assert events == ["yolo_start", "yolo_end", "sam_start", "sam_end"]
        assert bundle.frame_id == 0
        assert bundle.source_frame_id == 100
        assert bundle.proposal_source == "anchor_guided_sam"
        assert bundle.actual_backend == "sam2"
        assert len(bundle.anchors) == 1
        assert len(bundle.source_proposals) == 1
        assert len(bundle.raw_proposals) == 1
        assert bundle.generation_timings["yoloworld_primary"] == pytest.approx(0.12)
        assert bundle.generation_timings["sam2_proposals"] >= 0.0
        assert bundle.generation_timings["anchor_guided_sam_fusion"] >= 0.0
        assert bundle.generation_timings["proposal_generation"] >= 0.0

    def test_yoloworld_sam_frontend_preserves_yolo_assignments_for_anchor_box_fallback(self):
        from src.pipelines.yoloworld_sam_frontend import build_yoloworld_sam_bundle

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        frame = Frame(
            frame_id=0,
            source_frame_id=100,
            rgb=rgb,
            depth=depth,
            pose=np.eye(4, dtype=np.float64),
            intrinsics=intrinsics,
        )
        anchor = Anchor2D(3, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.9)
        anchor_mask = np.zeros((8, 8), dtype=bool)
        anchor_mask[1:7, 1:7] = True
        anchor_proposal = Proposal2D(
            proposal_id=0,
            mask=anchor_mask,
            bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
            area=int(anchor_mask.sum()),
            confidence=0.9,
            backend_name="anchor_box_primary",
            metadata={"anchor_id": 3, "anchor_class_name": "sofa"},
        )
        anchor_assignment = AnchorAssignment(0, 3, "sofa", 0.9, keepalive=True)

        class FakeAnchorFrontend:
            last_generation_timings = {}

            def generate_anchor_box_proposals(self, input_rgb, *, depth=None, intrinsics=None):
                return [anchor], [anchor_proposal], [anchor_assignment]

        class FakeProposal:
            active_backend_name = "sam2"

            def process_for_anchors(self, input_rgb, input_depth, anchors, frame=None):
                return []

        class EmptyFusion:
            def build_proposals(self, *, frame, anchors, sam_proposals):
                return [], [], {"enabled": True, "assignment_count": 0}

        bundle = build_yoloworld_sam_bundle(
            frame=frame,
            object_anchor=FakeAnchorFrontend(),
            proposal=FakeProposal(),
            anchor_guided_sam=EmptyFusion(),
            collect_stage_timings=True,
        )

        assert bundle.raw_proposals == (anchor_proposal,)
        assert bundle.anchor_assignments == (anchor_assignment,)
        assert bundle.source_proposals == ()
        assert bundle.anchor_guided_sam_summary == {"enabled": True, "assignment_count": 0}
        assert bundle.raw_proposals[0].metadata["observation_layer"] == "coarse"
        assert bundle.raw_proposals[0].metadata["refinement_key"] == "100:3"

    def test_yoloworld_sam_frontend_anchor_only_frame_skips_sam_and_uses_anchor_box_primary(self):
        from src.pipelines.yoloworld_sam_frontend import build_yoloworld_sam_bundle

        events: list[str] = []
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        frame = Frame(
            frame_id=1,
            source_frame_id=101,
            rgb=rgb,
            depth=depth,
            pose=np.eye(4, dtype=np.float64),
            intrinsics=intrinsics,
        )
        anchor = Anchor2D(3, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.9)
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:7, 1:7] = True
        anchor_proposal = Proposal2D(
            proposal_id=0,
            mask=mask,
            bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.9,
            backend_name="anchor_box_primary",
            metadata={"anchor_id": 3, "anchor_class_name": "sofa", "anchor_keepalive": True},
        )
        anchor_assignment = AnchorAssignment(0, 3, "sofa", 0.9, keepalive=True)

        class FakeAnchorFrontend:
            active_backend_name = "yoloworld"
            last_generation_timings = {}

            def generate_anchor_box_proposals(self, input_rgb, *, depth=None, intrinsics=None):
                events.append("yolo")
                self.last_generation_timings = {"yoloworld_primary": 0.05}
                return [anchor], [anchor_proposal], [anchor_assignment]

        class FakeProposal:
            active_backend_name = "sam2"

            def process_for_anchors(self, input_rgb, input_depth, anchors, frame=None):
                raise AssertionError("SAM should not run on anchor-only Balanced frames")

        bundle = build_yoloworld_sam_bundle(
            frame=frame,
            object_anchor=FakeAnchorFrontend(),
            proposal=FakeProposal(),
            anchor_guided_sam=AnchorGuidedSAMModule({"enabled": True}),
            collect_stage_timings=True,
            run_sam=False,
        )

        assert events == ["yolo"]
        assert bundle.proposal_source == "anchor_box_primary"
        assert bundle.actual_backend == "sam2"
        assert bundle.source_proposals == ()
        assert bundle.raw_proposals == (anchor_proposal,)
        assert bundle.anchor_assignments == (anchor_assignment,)
        assert bundle.generation_timings["yoloworld_primary"] == pytest.approx(0.05)
        assert bundle.generation_timings["sam2_full_frame_ran"] == 0.0
        assert bundle.generation_timings["proposal_generation"] >= 0.0
        assert bundle.anchor_guided_sam_summary["skip_reason"] == "balanced_anchor_only_frame"
        assert bundle.raw_proposals[0].metadata["observation_layer"] == "coarse"
        assert bundle.raw_proposals[0].metadata["refinement_key"] == "101:3"

    def test_association_restricts_candidates_to_active_set(self):
        TestRoadmapRefactor().test_association_restricts_candidates_to_active_set()

    def test_association_limits_geometry_consistency_to_top_k_candidates(self):
        TestRoadmapRefactor().test_association_limits_geometry_consistency_to_top_k_candidates()

    def test_association_geometry_top_k_prioritizes_tsdf_vote_owner(self):
        TestRoadmapRefactor().test_association_geometry_top_k_prioritizes_tsdf_vote_owner()

    def test_association_geometry_uses_association_pcd_when_present(self):
        TestRoadmapRefactor().test_association_geometry_uses_association_pcd_when_present()

    def test_object_update_refreshes_bounded_association_geometry_memory(self):
        TestRoadmapRefactor().test_object_update_refreshes_bounded_association_geometry_memory()

    def test_association_routes_confident_cross_label_overlap_to_contested_not_matched(self):
        TestRoadmapRefactor().test_association_routes_confident_cross_label_overlap_to_contested_not_matched()

    def test_repeated_provisional_anchor_label_becomes_association_identity_guard_only(self):
        TestRoadmapRefactor().test_repeated_provisional_anchor_label_becomes_association_identity_guard_only()

    def test_association_blocks_confident_cross_label_patch_against_repeated_provisional_identity(self):
        TestRoadmapRefactor().test_association_blocks_confident_cross_label_patch_against_repeated_provisional_identity()

    def test_safe_provisional_export_label_does_not_create_association_conflict(self):
        TestRoadmapRefactor().test_safe_provisional_export_label_does_not_create_association_conflict()

    def test_pipeline_uses_anchor_guided_sam_in_anchor_primary_mode(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
  min_mask_area: 1
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
anchor_guided_sam:
  enabled: true
  proposal_min_area: 1
  min_proposal_anchor_coverage_for_label: 0.5
  min_anchor_proposal_coverage_for_label: 0.5
  contained_min_proposal_coverage: 0.85
  contained_max_anchor_coverage: 0.35
pipeline:
  verbose: false
  collect_stage_timings: true
runtime_vis:
  enabled: false
depth_refinement:
  min_mask_area_after_refine: 1
bg_obj_split:
  background_threshold: 0.95
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
  provisional:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))

        class AnchorPrimary:
            enabled = True
            anchor_primary_mode = True
            use_sam_intersection_proposals = False
            depth_structure_enabled = False
            last_generation_timings = {}
            collect_generation_timings = False

            def generate_anchor_box_proposals(self, rgb, *, depth=None, intrinsics=None):
                anchor = Anchor2D(4, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.9)
                mask = np.zeros((8, 8), dtype=bool)
                mask[1:7, 1:7] = True
                proposal = Proposal2D(
                    proposal_id=4,
                    mask=mask,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=0.9,
                    backend_name="anchor_box_primary",
                    metadata={"anchor_id": 4, "anchor_class_name": "sofa", "anchor_keepalive": True},
                )
                assignment = AnchorAssignment(4, 4, "sofa", 0.9, keepalive=True)
                return [anchor], [proposal], [assignment]

        class PrecomputedProposal:
            active_backend_name = "precomputed"

            def process_for_anchors(self, rgb, depth, anchors, frame=None):
                mask = np.zeros((8, 8), dtype=bool)
                mask[1:7, 1:7] = True
                return [
                    Proposal2D(
                        proposal_id=90,
                        mask=mask,
                        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                        area=int(mask.sum()),
                        confidence=1.0,
                        backend_name="precomputed",
                    )
                ]

        pipe.object_anchor = AnchorPrimary()
        pipe.proposal = PrecomputedProposal()

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        assert pipe.last_frame_debug["proposal_source"] == "anchor_guided_sam"
        assert pipe.last_frame_debug["raw_proposal_count"] == 1
        anchor_guided_debug = pipe.last_frame_debug["anchor_guided_sam"]
        required_debug_keys = {
            "source_sam_proposal_count",
            "anchored_proposal_count",
            "unknown_residual_count",
            "dropped_proposal_count",
            "mean_sam_candidates_per_anchor",
            "semantic_blocked_residual_count",
        }
        assert required_debug_keys <= set(anchor_guided_debug)
        assert anchor_guided_debug["source_sam_proposal_count"] == 1
        assert pipe.last_raw_proposals[0].metadata["source"] == "anchor_guided_sam"
        assert pipe.last_raw_proposals[0].metadata["source_raw_proposal_ids"] == [90]

    def test_pipeline_uses_parallel_yoloworld_sam_frontend_when_enabled(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
  min_mask_area: 1
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
anchor_guided_sam:
  enabled: true
  proposal_min_area: 1
pipeline:
  verbose: false
  collect_stage_timings: true
  yoloworld_sam_parallel_frontend_enabled: true
runtime_vis:
  enabled: false
depth_refinement:
  min_mask_area_after_refine: 1
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
  provisional:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))
        events: list[str] = []
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:7, 1:7] = True

        class FakeAnchor:
            enabled = True
            anchor_primary_mode = True
            use_sam_intersection_proposals = False
            last_generation_timings = {}
            active_backend_name = "yoloworld"

            def generate_anchor_box_proposals(self, rgb, *, depth=None, intrinsics=None):
                events.append("yolo")
                self.last_generation_timings = {"yoloworld_primary": 0.01, "anchor_merge": 0.0}
                anchor = Anchor2D(1, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.95)
                proposal = Proposal2D(
                    proposal_id=1,
                    mask=mask,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=0.95,
                    backend_name="anchor_box_primary",
                    metadata={"anchor_id": 1, "anchor_class_name": "sofa"},
                )
                assignment = AnchorAssignment(1, 1, "sofa", 0.95, keepalive=True)
                return [anchor], [proposal], [assignment]

        class FakeSAM:
            active_backend_name = "sam2"

            def process_for_anchors(self, rgb, depth, anchors, frame=None):
                events.append("sam")
                assert [int(anchor.anchor_id) for anchor in anchors] == [1]
                return [
                    Proposal2D(
                        proposal_id=5,
                        mask=mask.copy(),
                        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                        area=int(mask.sum()),
                        confidence=1.0,
                        backend_name="sam2",
                    )
                ]

        pipe.object_anchor = FakeAnchor()
        pipe.proposal = FakeSAM()

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics, source_frame_id=42)

        assert events == ["yolo", "sam"]
        assert pipe.last_frame_debug["proposal_source"] == "anchor_guided_sam"
        assert pipe.last_frame_debug["frontend_actual_backend"] == "sam2"
        assert pipe.last_frame_debug["stage_timings"]["yoloworld_primary"] == pytest.approx(0.01)
        assert "sam2_proposals" in pipe.last_frame_debug["stage_timings"]
        assert "anchor_guided_sam_fusion" in pipe.last_frame_debug["stage_timings"]
        assert "proposal_generation" in pipe.last_frame_debug["stage_timings"]

    def test_pipeline_uses_anchor_first_sam_without_anchor_guided_fusion(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: sam2
  min_mask_area: 1
  sam2:
    anchor_prompt:
      enabled: true
anchor_frontend:
  enabled: true
  backend: yoloworld
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
anchor_guided_sam:
  enabled: false
pipeline:
  verbose: false
  collect_stage_timings: true
  yoloworld_sam_parallel_frontend_enabled: true
  yoloworld_sam_mode: anchor_first_sam
runtime_vis:
  enabled: false
depth_refinement:
  min_mask_area_after_refine: 1
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
  provisional:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))
        events: list[str] = []
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:7, 1:7] = True

        class FakeAnchor:
            enabled = True
            anchor_primary_mode = True
            use_sam_intersection_proposals = False
            last_generation_timings = {}
            active_backend_name = "yoloworld"

            def generate_anchor_box_proposals(self, rgb, *, depth=None, intrinsics=None):
                events.append("yolo")
                self.last_generation_timings = {"yoloworld_primary": 0.01}
                anchor = Anchor2D(1, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.95)
                proposal = Proposal2D(
                    proposal_id=1,
                    mask=mask.copy(),
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=0.95,
                    backend_name="anchor_box_primary",
                    metadata={"anchor_id": 1, "anchor_class_name": "sofa"},
                )
                assignment = AnchorAssignment(1, 1, "sofa", 0.95, keepalive=True)
                return [anchor], [proposal], [assignment]

        class FakeSAM:
            active_backend_name = "sam2"
            backend = type("Backend", (), {"anchor_prompt_enabled": True})()

            def process_for_anchors(self, rgb, depth, anchors, frame=None):
                events.append("sam2_anchor_prompts")
                assert [int(anchor.anchor_id) for anchor in anchors] == [1]
                return [
                    Proposal2D(
                        proposal_id=5,
                        mask=mask.copy(),
                        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                        area=int(mask.sum()),
                        confidence=1.0,
                        backend_name="sam2",
                        metadata={"source": "anchor_prompted_sam", "anchor_id": 1},
                    )
                ]

        class FailIfCalledAnchorGuidedSAM:
            enabled = False

            def build_proposals(self, **kwargs):
                raise AssertionError("anchor_first_sam must not call anchor_guided_sam.build_proposals")

        pipe.object_anchor = FakeAnchor()
        pipe.proposal = FakeSAM()
        pipe.anchor_guided_sam = FailIfCalledAnchorGuidedSAM()

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics, source_frame_id=42)

        assert events == ["yolo", "sam2_anchor_prompts"]
        assert pipe.last_frame_debug["proposal_source"] == "anchor_prompted_sam"
        assert pipe.last_frame_debug["frontend_actual_backend"] == "sam2"
        assert pipe.last_frame_debug["stage_timings"]["yoloworld_primary"] == pytest.approx(0.01)
        assert "sam2_anchor_prompts" in pipe.last_frame_debug["stage_timings"]
        assert "anchor_guided_sam_fusion" not in pipe.last_frame_debug["stage_timings"]

    def test_pipeline_anchor_first_sam_requires_anchor_prompt_enabled(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
  min_mask_area: 1
anchor_frontend:
  enabled: true
  backend: yoloworld
  anchor_primary_mode: true
anchor_guided_sam:
  enabled: false
pipeline:
  verbose: false
  collect_stage_timings: true
  yoloworld_sam_parallel_frontend_enabled: true
  yoloworld_sam_mode: anchor_first_sam
runtime_vis:
  enabled: false
depth_refinement:
  min_mask_area_after_refine: 1
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
  provisional:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))

        class FakeAnchor:
            enabled = True
            anchor_primary_mode = True
            active_backend_name = "yoloworld"

        class FakeSAMWithoutAnchorPrompts:
            active_backend_name = "sam2"
            backend = type("Backend", (), {"anchor_prompt_enabled": False})()

        pipe.object_anchor = FakeAnchor()
        pipe.proposal = FakeSAMWithoutAnchorPrompts()

        assert pipe._should_use_parallel_yoloworld_sam_frontend() is False

    def test_anchor_first_sam_bundle_ignores_unanchored_sam_outputs(self):
        from src.pipelines.yoloworld_sam_frontend import build_anchor_first_sam_bundle

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        frame = Frame(
            frame_id=0,
            source_frame_id=42,
            rgb=rgb,
            depth=depth,
            pose=np.eye(4, dtype=np.float64),
            intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8),
        )
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:7, 1:7] = True
        anchor = Anchor2D(1, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.95)
        anchor_proposal = Proposal2D(
            proposal_id=0,
            mask=mask.copy(),
            bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.95,
            backend_name="anchor_box_primary",
            metadata={"anchor_id": 1, "anchor_class_name": "sofa"},
        )
        anchor_assignment = AnchorAssignment(0, 1, "sofa", 0.95, keepalive=True)

        class FakeAnchor:
            active_backend_name = "yoloworld"
            last_generation_timings = {"yoloworld_primary": 0.01}

            def generate_anchor_box_proposals(self, rgb, *, depth=None, intrinsics=None):
                return [anchor], [anchor_proposal], [anchor_assignment]

        class FakeSAM:
            active_backend_name = "sam2"
            backend = type("Backend", (), {"last_generation_info": {"skipped_anchor_count": 1}})()
            last_generation_timings = {"sam2_anchor_prompts": 0.02}

            def process_for_anchors(self, rgb, depth, anchors, frame=None):
                return [
                    Proposal2D(
                        proposal_id=3,
                        mask=mask.copy(),
                        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                        area=int(mask.sum()),
                        confidence=0.7,
                        backend_name="sam2",
                        metadata={"source": "sam2"},
                    )
                ]

        bundle = build_anchor_first_sam_bundle(
            frame=frame,
            object_anchor=FakeAnchor(),
            proposal=FakeSAM(),
            collect_stage_timings=True,
        )

        assert bundle.proposal_source == "anchor_box_primary"
        assert bundle.raw_proposals == (anchor_proposal,)
        assert bundle.source_proposals[0].metadata["source"] == "sam2"
        assert bundle.anchor_guided_sam_summary["skip_reason"] == "anchor_first_sam_empty_prompted_masks"

    def test_pipeline_skips_yoloworld_sam_frontend_for_sam3_concept_backend(self, tmp_path):
        config_path = tmp_path / "sam3_pipeline.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: sam3_concept
  min_mask_area: 1
  sam3_concept:
    mode: mock
    classes: [chair]
    max_proposals: 1
anchor_frontend:
  enabled: false
  backend: placeholder
runtime_vis:
  enabled: false
depth_refinement:
  enabled: false
patch_lifting:
  min_points: 1
association:
  match_threshold: 0.5
object_update:
  provisional_pool:
    enabled: false
pipeline:
  collect_stage_timings: true
  yoloworld_sam_parallel_frontend_enabled: true
""",
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))

        assert pipe.proposal.active_backend_name == "sam3_concept"
        assert pipe._should_use_parallel_yoloworld_sam_frontend() is False

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=5.0, fy=5.0, cx=4.0, cy=4.0, width=8, height=8)
        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        assert pipe.last_frame_debug["proposal_source"] == "proposal_backend"
        assert pipe.last_frame_debug["frontend_actual_backend"] == "sam3_concept"
        assert "sam2_proposals" not in pipe.last_frame_debug["stage_timings"]
        assert "anchor_guided_sam_fusion" not in pipe.last_frame_debug["stage_timings"]
        assert pipe.last_raw_proposals[0].metadata["anchor_class_name"] == "chair"
        assert pipe.last_raw_proposals[0].metadata["source"] == "sam3_concept"
        assert pipe.last_raw_proposals[0].metadata["actual_backend"] == "sam3_concept"

    def test_pipeline_balanced_yoloworld_sam_runs_sam_only_on_interval_frames(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
  min_mask_area: 1
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
anchor_guided_sam:
  enabled: true
  proposal_min_area: 1
pipeline:
  verbose: false
  collect_stage_timings: true
  yoloworld_sam_parallel_frontend_enabled: true
  yoloworld_sam_balanced_frontend_enabled: true
  yoloworld_sam_full_frame_interval: 10
runtime_vis:
  enabled: false
depth_refinement:
  min_mask_area_after_refine: 1
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
  provisional:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:7, 1:7] = True
        sam_calls: list[int] = []

        class FakeAnchor:
            enabled = True
            anchor_primary_mode = True
            use_sam_intersection_proposals = False
            last_generation_timings = {}
            active_backend_name = "yoloworld"

            def generate_anchor_box_proposals(self, rgb, *, depth=None, intrinsics=None):
                self.last_generation_timings = {"yoloworld_primary": 0.01}
                anchor = Anchor2D(1, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.95)
                proposal = Proposal2D(
                    proposal_id=1,
                    mask=mask,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=0.95,
                    backend_name="anchor_box_primary",
                    metadata={"anchor_id": 1, "anchor_class_name": "sofa", "anchor_keepalive": True},
                )
                assignment = AnchorAssignment(1, 1, "sofa", 0.95, keepalive=True)
                return [anchor], [proposal], [assignment]

        class FakeSAM:
            active_backend_name = "sam2"

            def process_for_anchors(self, rgb, depth, anchors, frame=None):
                sam_calls.append(int(frame.frame_id))
                return [
                    Proposal2D(
                        proposal_id=5,
                        mask=mask.copy(),
                        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                        area=int(mask.sum()),
                        confidence=1.0,
                        backend_name="sam2",
                    )
                ]

        pipe.object_anchor = FakeAnchor()
        pipe.proposal = FakeSAM()

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics, source_frame_id=100)
        first_debug = dict(pipe.last_frame_debug)
        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics, source_frame_id=110)
        second_debug = dict(pipe.last_frame_debug)

        assert sam_calls == [0]
        assert first_debug["proposal_source"] == "anchor_guided_sam"
        assert first_debug["stage_timings"]["sam2_full_frame_ran"] == pytest.approx(1.0)
        assert second_debug["proposal_source"] == "anchor_box_primary"
        assert second_debug["stage_timings"]["sam2_full_frame_ran"] == pytest.approx(0.0)
        assert "sam2_proposals" not in second_debug["stage_timings"]
        assert second_debug["anchor_guided_sam"]["skip_reason"] == "balanced_anchor_only_frame"

    @pytest.mark.parametrize(
        ("anchor_backend", "proposal_backend"),
        [
            ("precomputed", "sam2"),
            ("yoloworld", "precomputed"),
        ],
    )
    def test_pipeline_skips_parallel_yoloworld_sam_frontend_for_non_baseline_backends(
        self,
        tmp_path,
        anchor_backend,
        proposal_backend,
    ):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
  min_mask_area: 1
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
anchor_guided_sam:
  enabled: true
  proposal_min_area: 1
pipeline:
  verbose: false
  collect_stage_timings: true
  yoloworld_sam_parallel_frontend_enabled: true
runtime_vis:
  enabled: false
depth_refinement:
  min_mask_area_after_refine: 1
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
  provisional:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))
        events: list[str] = []
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:7, 1:7] = True

        class FakeAnchor:
            enabled = True
            anchor_primary_mode = True
            use_sam_intersection_proposals = False
            last_generation_timings = {}
            active_backend_name = anchor_backend

            def generate_anchor_box_proposals(self, rgb, *, depth=None, intrinsics=None):
                events.append("serial_anchor")
                anchor = Anchor2D(1, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.95)
                proposal = Proposal2D(
                    proposal_id=1,
                    mask=mask,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=0.95,
                    backend_name="anchor_box_primary",
                    metadata={"anchor_id": 1, "anchor_class_name": "sofa"},
                )
                assignment = AnchorAssignment(1, 1, "sofa", 0.95, keepalive=True)
                return [anchor], [proposal], [assignment]

        class FakeProposal:
            active_backend_name = proposal_backend

            def process_for_anchors(self, rgb, depth, anchors, frame=None):
                events.append("serial_sam")
                return [
                    Proposal2D(
                        proposal_id=5,
                        mask=mask.copy(),
                        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                        area=int(mask.sum()),
                        confidence=1.0,
                        backend_name=proposal_backend,
                    )
                ]

        pipe.object_anchor = FakeAnchor()
        pipe.proposal = FakeProposal()

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        assert events == ["serial_anchor", "serial_sam"]
        assert "sam2_proposals" not in pipe.last_frame_debug["stage_timings"]
        assert pipe.last_frame_debug["proposal_source"] == "anchor_guided_sam"
        assert pipe.last_frame_debug["frontend_actual_backend"] == proposal_backend

    def test_anchor_guided_fallback_preserves_force_object_candidate_after_depth_refinement(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
  min_mask_area: 1
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
anchor_guided_sam:
  enabled: true
  proposal_min_area: 1
pipeline:
  verbose: false
runtime_vis:
  enabled: false
depth_refinement:
  min_mask_area_after_refine: 1
bg_obj_split:
  background_threshold: 0.95
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
  provisional:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))

        class AnchorPrimary:
            enabled = True
            anchor_primary_mode = True
            use_sam_intersection_proposals = False
            last_generation_timings = {}
            collect_generation_timings = False

            def generate_anchor_box_proposals(self, rgb, *, depth=None, intrinsics=None):
                anchor = Anchor2D(4, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.9)
                mask = np.zeros((8, 8), dtype=bool)
                mask[1:7, 1:7] = True
                proposal = Proposal2D(
                    proposal_id=4,
                    mask=mask,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=0.9,
                    backend_name="anchor_box_primary",
                    metadata={"anchor_id": 4, "anchor_class_name": "sofa", "anchor_keepalive": True},
                )
                assignment = AnchorAssignment(4, 4, "sofa", 0.9, keepalive=True)
                return [anchor], [proposal], [assignment]

        class EmptyAnchorProposal:
            active_backend_name = "precomputed"

            def process_for_anchors(self, rgb, depth, anchors, frame=None):
                return []

        pipe.object_anchor = AnchorPrimary()
        pipe.proposal = EmptyAnchorProposal()

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        assert pipe.last_frame_debug["proposal_source"] == "anchor_guided_sam"
        assert len(pipe.last_raw_proposals) >= 1
        assert pipe.last_raw_proposals[0].metadata["mask_anchor_relation"] == "anchor_box_fallback"
        assert pipe.last_raw_proposals[0].metadata["force_object_candidate"] is True
        assert len(pipe.last_refined_proposals) >= 1
        assert pipe.last_refined_proposals[0].metadata["force_object_candidate"] is True

    def test_structural_overlay_uses_precomputed_sam_without_changing_object_proposals(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        (tmp_path / "precomputed_cache").mkdir()
        config_path.write_text(
            f"""
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: precomputed
  cache_dir: {tmp_path / "precomputed_cache"}
  strict: false
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
pipeline:
  verbose: false
  collect_stage_timings: true
structural_overlay:
  enabled: true
  classes: [wall]
  voxel_size: 0.01
  min_overlap_area: 1
  min_proposal_anchor_coverage: 0.01
  min_anchor_proposal_coverage: 0.01
depth_refinement:
  min_mask_area_after_refine: 1
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))

        class AnchorPrimary:
            enabled = True
            anchor_primary_mode = True
            use_sam_intersection_proposals = False
            last_generation_timings = {}
            collect_generation_timings = False

            def generate_anchor_box_proposals(self, rgb, *, depth=None, intrinsics=None):
                anchor = Anchor2D(4, np.array([1, 1, 7, 7], dtype=np.float32), "wall", 0.9)
                mask = np.zeros((8, 8), dtype=bool)
                mask[1:7, 1:7] = True
                proposal = Proposal2D(
                    proposal_id=4,
                    mask=mask,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=0.9,
                    metadata={"anchor_id": 4, "anchor_class_name": "wall", "anchor_keepalive": True},
                )
                assignment = AnchorAssignment(4, 4, "wall", 0.9, keepalive=True)
                return [anchor], [proposal], [assignment]

        class PrecomputedProposal:
            active_backend_name = "precomputed"

            def __init__(self):
                self.calls = 0

            def process(self, rgb, depth, frame=None):
                self.calls += 1
                mask = np.zeros((8, 8), dtype=bool)
                mask[2:6, 2:6] = True
                return [
                    Proposal2D(
                        proposal_id=90,
                        mask=mask,
                        bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
                        area=int(mask.sum()),
                        confidence=1.0,
                        backend_name="precomputed",
                    )
                ]

        precomputed = PrecomputedProposal()
        pipe.object_anchor = AnchorPrimary()
        pipe.proposal = precomputed
        runtime_vis_inputs = []

        class CaptureRuntimeVis:
            def process(self, frame, proposals, state):
                runtime_vis_inputs.append(
                    [
                        (
                            int(proposal.proposal_id),
                            proposal.metadata.get("anchor_class_name"),
                        )
                        for proposal in proposals
                    ]
                )
                return RuntimeVisOutput(
                    raw_proposals=list(proposals),
                    merged_proposals=list(proposals),
                    groups=[],
                    raw_to_group={},
                    group_to_linked_object={},
                    proposal_profiles=[],
                    merge_decisions=[],
                    group_stats={},
                )

        pipe.runtime_vis = CaptureRuntimeVis()

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics, source_frame_id=12)

        assert precomputed.calls == 1
        assert len(pipe.last_raw_proposals) == 1
        assert pipe.last_raw_proposals[0].metadata["anchor_class_name"] == "wall"
        assert len(pipe.last_source_proposals) == 1
        assert pipe.last_source_proposals[0].proposal_id == 90
        assert runtime_vis_inputs == [[(4, "wall")]]
        assert pipe.last_frame_debug["structural_overlay"]["accepted_pair_count"] == 1
        assert pipe.last_frame_debug["stage_timings"]["structural_overlay"] >= 0.0
        assert len(pipe.state.structural_overlay_map.voxels) > 0

    def test_structural_overlay_preserves_module_skip_reason_when_loader_succeeds(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        (tmp_path / "precomputed_cache").mkdir()
        config_path.write_text(
            f"""
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: precomputed
  cache_dir: {tmp_path / "precomputed_cache"}
  strict: false
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
pipeline:
  verbose: false
structural_overlay:
  enabled: true
  classes: [wall]
  min_overlap_area: 1
  min_proposal_anchor_coverage: 0.01
  min_anchor_proposal_coverage: 0.01
depth_refinement:
  min_mask_area_after_refine: 1
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))

        class AnchorPrimary:
            enabled = True
            anchor_primary_mode = True
            use_sam_intersection_proposals = False
            last_generation_timings = {}
            collect_generation_timings = False

            def generate_anchor_box_proposals(self, rgb, *, depth=None, intrinsics=None):
                anchor = Anchor2D(4, np.array([1, 1, 7, 7], dtype=np.float32), "chair", 0.9)
                mask = np.zeros((8, 8), dtype=bool)
                mask[1:7, 1:7] = True
                proposal = Proposal2D(
                    proposal_id=4,
                    mask=mask,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=0.9,
                    metadata={"anchor_id": 4, "anchor_class_name": "chair", "anchor_keepalive": True},
                )
                assignment = AnchorAssignment(4, 4, "chair", 0.9, keepalive=True)
                return [anchor], [proposal], [assignment]

        class PrecomputedProposal:
            active_backend_name = "precomputed"

            def process(self, rgb, depth, frame=None):
                mask = np.zeros((8, 8), dtype=bool)
                mask[2:6, 2:6] = True
                return [
                    Proposal2D(
                        proposal_id=90,
                        mask=mask,
                        bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
                        area=int(mask.sum()),
                        confidence=1.0,
                        backend_name="precomputed",
                    )
                ]

        pipe.object_anchor = AnchorPrimary()
        pipe.proposal = PrecomputedProposal()

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        debug = pipe.last_frame_debug["structural_overlay"]
        assert debug["proposal_source"] == "precomputed_overlay_only"
        assert debug["skip_reason"] == "no_structure_anchors"
        assert debug["sam_proposal_count"] == 1

    def test_pipeline_applies_ready_async_refinement_after_coarse_update(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
pipeline:
  verbose: false
  collect_stage_timings: true
async_refinement:
  enabled: true
  delay_frames: 0
  proposal_min_area: 1
  replace_coarse_observations: true
depth_refinement:
  min_mask_area_after_refine: 1
bg_obj_split:
  background_threshold: 0.95
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
  provisional:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))
        pipe.object_anchor.generate_anchor_box_proposals = lambda rgb, *, depth=None, intrinsics=None: (
            [
                Anchor2D(
                    anchor_id=4,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.9,
                )
            ],
            [
                Proposal2D(
                    proposal_id=0,
                    mask=np.pad(np.ones((6, 6), dtype=bool), ((1, 1), (1, 1))),
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=36,
                    confidence=0.9,
                    backend_name="anchor_box_primary",
                    metadata={
                        "source": "anchor_box_primary",
                        "anchor_id": 4,
                        "anchor_class_name": "sofa",
                        "anchor_confidence": 0.9,
                        "anchor_label_strength": "strong",
                        "anchor_keepalive": True,
                    },
                )
            ],
            [AnchorAssignment(proposal_id=0, anchor_id=4, class_name="sofa", confidence=0.9, keepalive=True)],
        )

        sam_mask = np.zeros((8, 8), dtype=bool)
        sam_mask[2:6, 2:6] = True
        pipe.async_refinement_backend = lambda frame: [
            Proposal2D(
                proposal_id=20,
                mask=sam_mask,
                bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
                area=16,
                confidence=0.8,
                backend_name="sam2",
            )
        ]

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics, source_frame_id=100)

        obj = pipe.state.objects[0]
        assert pipe.last_frame_debug["async_refinement"]["fine_proposal_count"] == 1
        assert pipe.last_frame_debug["async_refinement"]["replaced_observation_count"] == 1
        assert len(obj.observations) == 1
        assert obj.observations[0].observation_layer == "fine"
        assert obj.observations[0].frame_id == 0
        assert len(obj.local_pcd) == 16
        assert pipe.last_frame_debug["dense_surface_resident_point_count"] > 0
        assert pipe.last_frame_debug["stage_timings"]["async_refinement"] >= 0.0

    def test_pipeline_skips_async_refinement_fallback_for_non_sam_backend(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
pipeline:
  verbose: false
async_refinement:
  enabled: true
  delay_frames: 0
  proposal_min_area: 1
  replace_coarse_observations: true
depth_refinement:
  min_mask_area_after_refine: 1
bg_obj_split:
  background_threshold: 0.95
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
  provisional:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        debug = pipe.last_frame_debug["async_refinement"]
        assert debug["fine_proposal_count"] == 0
        assert debug["replaced_observation_count"] == 0
        assert debug["proposal_source"] == "disabled_non_sam_backend"
        assert debug["skip_reason"] == "async_refinement_requires_sam_backend"
        assert debug["active_backend"] == "placeholder"

    def test_pipeline_skips_cropformer_as_async_refinement_fallback(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  proposal_min_area: 1
pipeline:
  verbose: false
async_refinement:
  enabled: true
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))
        pipe.object_anchor.generate_anchor_box_proposals = (
            lambda rgb, *, depth=None, intrinsics=None: ([], [], [])
        )
        pipe.proposal.active_backend_name = "cropformer"

        def _unexpected_proposals(*args, **kwargs):
            raise AssertionError("cropformer fallback should not be used for async SAM refinement")

        pipe.proposal.process = _unexpected_proposals
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        debug = pipe.last_frame_debug["async_refinement"]
        assert debug["proposal_source"] == "disabled_non_sam_backend"
        assert debug["skip_reason"] == "async_refinement_requires_sam_backend"
        assert debug["active_backend"] == "cropformer"

    def test_pipeline_async_refinement_disabled_summary_has_updated_ids(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 4
  image_width: 4
proposal:
  backend: placeholder
anchor_frontend:
  enabled: false
pipeline:
  verbose: false
async_refinement:
  enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=4, height=4)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        assert pipe.last_frame_debug["async_refinement"]["updated_object_ids"] == []

    def test_pipeline_records_stage_timings_when_enabled(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 4
  image_width: 4
proposal:
  backend: placeholder
anchor_frontend:
  enabled: false
pipeline:
  verbose: false
  collect_stage_timings: true
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=4, height=4)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        timings = pipe.last_frame_debug["stage_timings"]
        expected_stages = {
            "frame_input",
            "proposal_generation",
            "runtime_vis",
            "depth_refinement",
            "patch_lifting",
            "active_set",
            "bg_obj_split",
            "association",
            "object_update",
            "semantic_memory",
            "dense_surface",
            "map_tiering",
            "background_update",
            "dynamic_maintenance",
        }
        assert expected_stages.issubset(set(timings))
        assert all(value >= 0.0 for value in timings.values())

    def test_pipeline_merges_anchor_frontend_stage_timings(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 4
  image_width: 4
proposal:
  backend: placeholder
anchor_frontend:
  enabled: false
pipeline:
  verbose: false
  collect_stage_timings: true
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))
        pipe.object_anchor.enabled = True
        pipe.object_anchor.anchor_primary_mode = True

        def _fake_generate_anchor_box_proposals(rgb, *, depth=None, intrinsics=None):
            pipe.object_anchor.last_generation_timings = {
                "yoloworld_primary": 0.12,
                "yoloe_supplemental": 0.34,
                "anchor_merge": 0.05,
            }
            return [], [], []

        pipe.object_anchor.generate_anchor_box_proposals = _fake_generate_anchor_box_proposals
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=4, height=4)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        timings = pipe.last_frame_debug["stage_timings"]
        assert timings["yoloworld_primary"] == pytest.approx(0.12)
        assert timings["yoloe_supplemental"] == pytest.approx(0.34)
        assert timings["anchor_merge"] == pytest.approx(0.05)
        assert timings["proposal_generation"] >= 0.0

    def test_pipeline_rejects_nonfinite_nested_stage_timings_and_renames_collisions(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
pipeline:
  collect_stage_timings: true
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))
        pipe._merge_stage_timings(
            {
                "proposal_generation": 1.0,
                "bad_nan": float("nan"),
                "bad_inf": float("inf"),
                "valid": 0.5,
            }
        )

        assert pipe._stage_timings["nested_proposal_generation"] == pytest.approx(1.0)
        assert pipe._stage_timings["valid"] == pytest.approx(0.5)
        assert "proposal_generation" not in pipe._stage_timings
        assert "bad_nan" not in pipe._stage_timings
        assert "bad_inf" not in pipe._stage_timings

    def test_pipeline_stage_timings_empty_when_disabled(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 4
  image_width: 4
proposal:
  backend: placeholder
anchor_frontend:
  enabled: false
pipeline:
  verbose: false
  collect_stage_timings: false
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        def _raise_if_called():
            raise AssertionError("perf_counter should not be called when stage timings are disabled")

        monkeypatch.setattr("src.pipelines.main_pipeline.time.perf_counter", _raise_if_called)
        pipe = Pipeline(config_path=str(config_path))
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=4, height=4)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        assert pipe.last_frame_debug["stage_timings"] == {}

    def test_final_object_semantic_audit_reports_absorbed_classes(self):
        sofa_patch = Patch3D(
            patch_id=1,
            points=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            centroid=np.array([0.05, 0.0, 1.0], dtype=np.float32),
            bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            bbox_max=np.array([0.1, 0.0, 1.0], dtype=np.float32),
            source_frame_id=10,
            metadata={"anchor_class_name": "sofa", "anchor_confidence": 0.9},
        )
        obj = ObjectMap(
            object_id=7,
            local_pcd=sofa_patch.points.copy(),
            centroid=sofa_patch.centroid.copy(),
            bbox_min=sofa_patch.bbox_min.copy(),
            bbox_max=sofa_patch.bbox_max.copy(),
            observations=[
                ObservationRecord(
                    frame_id=10,
                    patch=sofa_patch,
                    crop_bbox=None,
                    timestamp=0.0,
                )
            ],
            creation_frame=10,
            last_seen_frame=10,
            update_count=1,
        )
        obj.debug["anchor_semantics"] = {
            "canonical_label": "sofa",
            "canonical_score": 0.9,
            "canonical_frame_hits": 1,
            "label_score_sum": {"sofa": 0.9},
            "label_max_confidence": {"sofa": 0.9},
            "label_seen_frames": {"sofa": [10]},
            "label_frame_hits": {"sofa": 1},
            "source": "anchor_vote",
        }
        state = SystemState(objects={7: obj})
        evaluation = {
            "nearest_object_ids": np.array([7, 7, 7, 7, -1], dtype=np.int32),
            "gt_labels": np.array([1, 1, 1, 2, 2], dtype=np.int32),
            "object_to_class": {7: 1},
            "per_class": [
                {"class_id": 1, "class_name": "sofa", "iou": 0.75, "acc": 1.0},
                {"class_id": 2, "class_name": "cushion", "iou": 0.0, "acc": 0.0},
            ],
        }

        audit = build_final_object_semantic_audit(
            state,
            evaluation,
            {1: "sofa", 2: "cushion"},
        )

        object_record = audit["objects"][0]
        assert object_record["source_anchor_top_class"] == "sofa"
        assert object_record["gt_majority"]["class_name"] == "sofa"
        cushion_record = next(item for item in audit["classes"] if item["class_name"] == "cushion")
        assert cushion_record["absorbed_by_other_majority_gt_count"] == 1
        assert cushion_record["top_covering_objects"][0]["object_id"] == 7
        assert cushion_record["top_covering_objects"][0]["object_gt_majority_class"] == "sofa"

    def test_final_object_semantic_audit_includes_semantic_commit_state(self):
        obj = ObjectMap(object_id=5)
        obj.local_pcd = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        obj.debug["anchor_semantics"] = {
            "canonical_label": "rug",
            "canonical_score": 1.25,
            "semantic_state": "committed",
            "committed_label": "rug",
            "commit_reason": "multiframe_strong_anchor_evidence",
            "label_weighted_score": {"rug": 1.25},
            "contextual_label_score_sum": {"sofa": 0.8},
        }
        state = SystemState(objects={5: obj})

        audit = build_final_object_semantic_audit(
            state,
            {
                "nearest_object_ids": np.array([], dtype=np.int64),
                "gt_labels": np.array([], dtype=np.int64),
                "object_to_class": {},
                "per_class": [],
            },
            {98: "rug"},
        )

        record = audit["objects"][0]
        assert record["semantic_state"] == "committed"
        assert record["committed_label"] == "rug"
        assert record["posterior_label"] == "rug"
        assert record["commit_reason"] == "multiframe_strong_anchor_evidence"
        assert record["contextual_label_score_sum"] == {"sofa": 0.8}
        assert audit["semantic_state_counts"] == {"committed": 1}

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

    def test_pipeline_anchor_primary_mode_skips_sam_source_proposals(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
anchor_frontend:
  enabled: false
  anchor_primary_mode: true
  proposal_min_area: 1
pipeline:
  verbose: false
patch_lifting:
  min_points: 1
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))
        pipe.object_anchor.enabled = True
        pipe.object_anchor.anchor_primary_mode = True

        class _FakeAnchorBackend:
            def generate_anchors(self, rgb):
                return [
                    Anchor2D(
                        anchor_id=2,
                        bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
                        class_name="chair",
                        confidence=0.9,
                    )
                ]

        class _ExplodingProposalBackend:
            def generate_proposals(self, rgb, depth, frame=None):
                raise AssertionError("SAM proposal backend should not run in anchor-primary mode")

        pipe.object_anchor.backend = _FakeAnchorBackend()
        pipe.proposal.backend = _ExplodingProposalBackend()
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        assert pipe.last_frame_debug["proposal_source"] == "anchor_box_primary"
        assert len(pipe.last_source_proposals) == 0
        assert len(pipe.last_raw_proposals) == 1
        assert pipe.last_raw_proposals[0].metadata["anchor_class_name"] == "chair"

    def test_process_frame_passes_current_frame_geometry_to_object_update(self, tmp_path):
        """Object update receives the current frame geometry needed by visibility gates."""
        pipeline_mod = __import__("src.pipelines.main_pipeline", fromlist=["Pipeline"])
        Pipeline = pipeline_mod.Pipeline
        config_path = tmp_path / "test_pipeline_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "proposal:",
                    "  backend: placeholder",
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

        class EmptyProposal:
            def process(self, *_args, **_kwargs):
                return []

        class DisabledAnchor:
            enabled = False
            use_sam_intersection_proposals = False

            def process(self, *_args, **_kwargs):
                return [], []

        class EmptyRuntimeVis:
            def process(self, _frame, proposals, _state):
                return SimpleNamespace(merged_proposals=list(proposals), group_stats={})

        class EmptyDepthRefinement:
            def process(self, *_args, **_kwargs):
                return []

        class EmptyPatchLifting:
            def process(self, *_args, **_kwargs):
                return []

        class PassThroughActiveSet:
            def process(self, state, *_args, **_kwargs):
                return state.active_set

            def cap_active_set(self, active_set):
                return active_set

        class EmptyBgObjSplit:
            def process(self, *_args, **_kwargs):
                return [], [], []

        class EmptyAssociation:
            def process(self, *_args, **_kwargs):
                return AssociationResult()

        class CapturingObjectUpdate:
            last_structural_reject_patches = []
            last_updated_object_ids = []
            last_surface_gate_stats = {}
            last_surface_gate_records = []
            last_current_frame_visibility_gate_stats = {"accepted": 0}
            last_current_frame_visibility_gate_records = [{"patch_id": 1}]

            def __init__(self):
                self.kwargs = None

            def process(self, _association, _patches, state, **kwargs):
                self.kwargs = kwargs
                return state

        class PassThroughSemanticMemory:
            last_updated_object_ids = []

            def process(self, state, *_args, **_kwargs):
                return state

        class PassThroughDenseSurface:
            def process(self, state, *_args, **_kwargs):
                return state

            def resident_point_count(self, _dense_surface_map):
                return 0

        class PassThroughMapTiering:
            def process(self, state, *_args, **_kwargs):
                return state

        class PassThroughBackgroundUpdate:
            def process(self, _patches, background, _objects):
                return background

        class PassThroughDynamicMaintenance:
            def process(self, state):
                return state

        object_update = CapturingObjectUpdate()
        pipe.proposal = EmptyProposal()
        pipe.object_anchor = DisabledAnchor()
        pipe.runtime_vis = EmptyRuntimeVis()
        pipe.depth_refinement = EmptyDepthRefinement()
        pipe.patch_lifting = EmptyPatchLifting()
        pipe.active_set = PassThroughActiveSet()
        pipe.bg_obj_split = EmptyBgObjSplit()
        pipe.association = EmptyAssociation()
        pipe.object_update = object_update
        pipe.semantic_memory = PassThroughSemanticMemory()
        pipe.dense_surface = PassThroughDenseSurface()
        pipe.map_tiering = PassThroughMapTiering()
        pipe.background_update = PassThroughBackgroundUpdate()
        pipe.dynamic_maintenance = PassThroughDynamicMaintenance()

        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        depth = np.full((4, 5), 2.0, dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        intr = CameraIntrinsics(fx=10, fy=11, cx=2, cy=3, width=5, height=4)

        pipe.process_frame(rgb, depth, pose, intr)

        assert object_update.kwargs["current_depth"] is depth
        assert object_update.kwargs["current_pose"] is pose
        assert object_update.kwargs["current_intrinsics"] is intr

    def test_pipeline_runs_with_precomputed_backend(self, tmp_path):
        pipeline_mod = __import__("src.pipelines.main_pipeline", fromlist=["Pipeline"])
        Pipeline = pipeline_mod.Pipeline
        cache_dir = tmp_path / "cache"
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:4, 1:5] = True
        proposals = [
            Proposal2D(
                proposal_id=0,
                mask=mask,
                bbox_xyxy=np.array([1, 1, 5, 4], dtype=np.float32),
                area=int(mask.sum()),
                confidence=0.9,
                backend_name="sam2",
            )
        ]
        save_proposals(
            cache_dir,
            frame_id=10,
            image_shape=(8, 8),
            proposals=proposals,
            source_backend="sam2",
        )
        write_manifest(
            cache_dir,
            dataset_summary={"frame_count": 1},
            backend_config={"backend": "sam2"},
            frame_entries=[{"frame_id": 10, "file": "frames/frame000010_proposals.npz", "proposal_count": 1}],
        )

        config_path = tmp_path / "test_pipeline_precomputed.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "proposal:",
                    "  backend: precomputed",
                    "  min_mask_area: 1",
                    "  precomputed:",
                    f"    cache_dir: {cache_dir}",
                    "    strict: true",
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

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        pose = np.eye(4)
        intr = CameraIntrinsics(fx=5, fy=5, cx=4, cy=4, width=8, height=8)

        state = pipe.process_frame(rgb, depth, pose, intr, source_frame_id=10)
        assert isinstance(state, SystemState)
        assert state.frame_count == 1
        assert pipe.proposal.active_backend_name == "precomputed"
        assert len(pipe.last_raw_proposals) == 1
        assert pipe.last_raw_proposals[0].metadata["cache_frame_id"] == 10

    def test_pipeline_uses_anchor_primary_proposals_when_enabled(self, tmp_path):
        pipeline_mod = __import__("src.pipelines.main_pipeline", fromlist=["Pipeline"])
        Pipeline = pipeline_mod.Pipeline
        config_path = tmp_path / "test_pipeline_anchor_primary.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "proposal:",
                    "  backend: placeholder",
                    "anchor_frontend:",
                    "  enabled: false",
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
        pipe.object_anchor.enabled = True
        pipe.object_anchor.use_sam_intersection_proposals = True
        pipe.object_anchor.prefer_largest_covering_box = True
        pipe.object_anchor.proposal_min_area = 1

        class _FakeAnchorBackend:
            def generate_anchors(self, rgb):
                return [
                    Anchor2D(
                        anchor_id=0,
                        bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
                        class_name="chair",
                        confidence=0.9,
                    )
                ]

        pipe.object_anchor.backend = _FakeAnchorBackend()

        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        sam_proposal = Proposal2D(
            proposal_id=9,
            mask=mask,
            bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.9,
            backend_name="sam2",
        )
        pipe.proposal.process = lambda *args, **kwargs: [sam_proposal]

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.full((8, 8), 2.0, dtype=np.float32)
        pose = np.eye(4)
        intr = CameraIntrinsics(fx=5, fy=5, cx=4, cy=4, width=8, height=8)

        state = pipe.process_frame(rgb, depth, pose, intr)
        assert isinstance(state, SystemState)
        assert state.frame_count == 1
        assert pipe.last_frame_debug["proposal_source"] == "anchor_sam_union"
        assert len(pipe.last_raw_proposals) == 1
        assert pipe.last_raw_proposals[0].metadata["anchor_id"] == 0
        assert pipe.last_raw_proposals[0].backend_name == "anchor_sam_union"

    def test_label_color_mapping_is_stable_and_vectorized(self):
        labels = np.array([1, -1, 2, 1, 2, -1], dtype=np.int32)
        colors = labels_to_class_colors(labels)
        assert colors.shape == (6, 3)
        assert tuple(colors[0]) == class_color(1)
        assert tuple(colors[2]) == class_color(2)
        assert tuple(colors[1]) == (80, 80, 80)
        assert tuple(colors[3]) == class_color(1)

    def test_build_local_memory_frame_audit_includes_anchor_metadata(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:5, 2:6] = True
        proposal = Proposal2D(
            proposal_id=5,
            mask=mask,
            bbox_xyxy=np.array([2, 1, 6, 5], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.9,
        )
        anchor = Anchor2D(
            anchor_id=3,
            bbox_xyxy=np.array([1, 1, 6, 6], dtype=np.float32),
            class_name="chair",
            confidence=0.88,
        )
        assignment = AnchorAssignment(
            proposal_id=5,
            anchor_id=3,
            class_name="chair",
            confidence=0.88,
            bbox_iou=0.52,
            center_inside=True,
            keepalive=True,
        )
        refined = RefinedProposal2D(
            proposal_id=50,
            mask=mask,
            bbox_xyxy=np.array([2, 1, 6, 5], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.9,
            metadata={"source_raw_proposal_id": 5},
        )
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        patch = Patch3D(
            patch_id=50,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            metadata={
                "source_proposal_id": 50,
                "foreground_depth_filter": {"enabled": True, "removed_point_count": 4},
                "surface_owner_gate": {"rejected_point_count": 3},
                "current_frame_visibility_gate": {
                    "rejected_patch_count": 1,
                    "depth_rejected_point_count": 2,
                },
            },
        )
        frame_visibility_stats = {
            "rejected_patch_count": 2,
            "depth_rejected_point_count": 5,
        }

        pipeline = SimpleNamespace(
            last_runtime_vis_output=SimpleNamespace(groups=[], proposal_profiles=[], raw_to_group={}),
            last_association=AssociationResult(),
            state=SystemState(),
            last_raw_proposals=[proposal],
            last_refined_proposals=[refined],
            last_patches=[patch],
            last_anchors=[anchor],
            last_anchor_assignments=[assignment],
            object_update=SimpleNamespace(last_current_frame_visibility_gate_stats=frame_visibility_stats),
            last_frame_debug={"bg_patch_ids": [], "obj_patch_ids": [50], "amb_patch_ids": []},
            association=SimpleNamespace(match_threshold=0.45),
            patch_lifting=SimpleNamespace(min_points=12),
            depth_refinement=SimpleNamespace(min_area=16),
        )
        frame = SimpleNamespace(frame_id=0, depth=np.ones((8, 8), dtype=np.float32))

        audit = build_local_memory_frame_audit(
            frame=frame,
            pipeline=pipeline,
            prev_object_ids=set(),
            prev_next_object_id=0,
        )

        assert audit["anchor_count"] == 1
        assert audit["anchored_proposal_count"] == 1
        assert audit["current_frame_visibility_gate"] == frame_visibility_stats
        assert audit["anchors"][0]["class_name"] == "chair"
        assert audit["raw_proposals"][0]["anchor_assignment"]["anchor_id"] == 3
        patch_record = audit["raw_proposals"][0]["patches"][0]
        assert patch_record["foreground_depth_filter"] == {"enabled": True, "removed_point_count": 4}
        assert patch_record["surface_owner_gate"] == {"rejected_point_count": 3}
        assert patch_record["current_frame_visibility_gate"] == {
            "rejected_patch_count": 1,
            "depth_rejected_point_count": 2,
        }

    def test_local_memory_audit_exposes_contested_residual_outcome(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:5, 2:6] = True
        proposal = Proposal2D(proposal_id=22, mask=mask, bbox_xyxy=np.array([2, 1, 6, 5], dtype=np.float32), area=int(mask.sum()), confidence=0.9)
        refined = RefinedProposal2D(proposal_id=22, mask=mask, bbox_xyxy=np.array([2, 1, 6, 5], dtype=np.float32), area=int(mask.sum()), confidence=0.9, metadata={"source_raw_proposal_id": 22})
        points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], dtype=np.float32)
        patch = Patch3D(
            patch_id=22,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            metadata={
                "source_proposal_id": 22,
                "anchor_class_name": "blanket",
                "anchor_confidence": 0.88,
                "contested_parent_object_id": 31,
                "contested_parent_label": "sofa",
                "contested_patch_label": "blanket",
                "contested_reason": "cross_label_observation_identity",
            },
        )
        association = AssociationResult(
            contested_object_patches=[22],
            contested_matches=[ContestedAssociation(patch_id=22, blocked_object_id=31, patch_label="blanket", object_label="sofa", reason="cross_label_observation_identity", score=AssociationScore(total_score=1.06))],
        )
        pipeline = SimpleNamespace(
            last_runtime_vis_output=SimpleNamespace(groups=[], proposal_profiles=[], raw_to_group={}),
            last_association=association,
            state=SystemState(),
            last_raw_proposals=[proposal],
            last_refined_proposals=[refined],
            last_patches=[patch],
            last_anchors=[],
            last_anchor_assignments=[],
            object_update=SimpleNamespace(last_current_frame_visibility_gate_stats={}, last_contested_residual_patch_ids=[22], last_contested_residual_promoted_object_ids=[40]),
            last_frame_debug={"bg_patch_ids": [], "obj_patch_ids": [22], "amb_patch_ids": []},
            association=SimpleNamespace(match_threshold=0.45),
            patch_lifting=SimpleNamespace(min_points=12),
            depth_refinement=SimpleNamespace(min_area=16),
        )
        frame = SimpleNamespace(frame_id=1800, depth=np.ones((8, 8), dtype=np.float32))

        audit = build_local_memory_frame_audit(frame=frame, pipeline=pipeline, prev_object_ids=set(), prev_next_object_id=0)

        assert audit["contested_residual_count"] == 1
        assert audit["contested_residual_events"] == [{"patch_id": 22, "blocked_object_id": 31, "patch_label": "blanket", "object_label": "sofa", "reason": "cross_label_observation_identity", "score": 1.06}]
        assert audit["contested_residual_patch_ids"] == [22]
        assert audit["contested_residual_promoted_object_ids"] == [40]
        patch_record = audit["raw_proposals"][0]["patches"][0]
        assert audit["raw_proposals"][0]["final_outcome"] == "contested_residual"
        assert patch_record["association_outcome"] == "contested_residual"
        assert patch_record["contested_parent_object_id"] == 31
        assert patch_record["contested_parent_label"] == "sofa"
        assert patch_record["contested_patch_label"] == "blanket"
        assert patch_record["contested_reason"] == "cross_label_observation_identity"

    def test_save_local_memory_audit_overlay_draws_anchor_boxes(self, tmp_path):
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:10, 4:10] = True
        proposal = Proposal2D(
            proposal_id=1,
            mask=mask,
            bbox_xyxy=np.array([4, 4, 10, 10], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.8,
        )
        raw_records = [
            {
                "proposal_id": 1,
                "final_outcome": "matched_existing_instance",
                "matched_existing_ids": [7],
                "anchor_assignment": {"anchor_id": 3},
            }
        ]
        no_anchor_path = tmp_path / "no_anchor.png"
        with_anchor_path = tmp_path / "with_anchor.png"
        positional_call_path = tmp_path / "positional_call.png"

        save_local_memory_audit_overlay(
            frame=SimpleNamespace(rgb=rgb),
            source_proposals=[],
            raw_proposals=[proposal],
            raw_records=raw_records,
            anchors=[],
            output_path=no_anchor_path,
        )
        save_local_memory_audit_overlay(
            frame=SimpleNamespace(rgb=rgb),
            source_proposals=[proposal],
            raw_proposals=[proposal],
            raw_records=raw_records,
            anchors=[
                {
                    "anchor_id": 3,
                    "bbox_xyxy": [1.0, 1.0, 14.0, 14.0],
                    "class_name": "chair",
                    "confidence": 0.88,
                }
            ],
            output_path=with_anchor_path,
        )
        save_local_memory_audit_overlay(
            SimpleNamespace(rgb=rgb),
            [proposal],
            [proposal],
            raw_records,
            [
                {
                    "anchor_id": 3,
                    "bbox_xyxy": [1.0, 1.0, 14.0, 14.0],
                    "class_name": "chair",
                    "confidence": 0.88,
                }
            ],
            positional_call_path,
        )

        without_anchor = np.asarray(Image.open(no_anchor_path))
        with_anchor = np.asarray(Image.open(with_anchor_path))

        assert without_anchor.shape == (40, 80, 3)
        assert with_anchor.shape == (40, 80, 3)
        assert np.any(with_anchor != without_anchor)
        assert positional_call_path.exists()

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

    def test_depth_refinement_keepalive_forces_protected_small_anchor_object_candidate(self):
        depth = np.full((8, 8), 1.0, dtype=np.float32)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        proposal = Proposal2D(
            proposal_id=4,
            mask=mask,
            bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
            area=int(mask.sum()),
            metadata={
                "anchor_keepalive": True,
                "anchor_class_name": "cushion",
                "anchor_confidence": 0.8,
                "protected_small_anchor": True,
                "force_object_candidate": True,
            },
        )
        module = DepthRefinementModule(
            {
                "depth_edge_threshold": 0.0,
                "min_mask_area_after_refine": 1000,
                "keepalive_objectness_floor": 0.58,
            }
        )

        refined = module.process(depth, [proposal])
        assert len(refined) == 1
        assert refined[0].metadata["depth_keepalive_reason"] == "small_anchor_protected"
        assert refined[0].metadata["force_object_candidate"] is True
        assert refined[0].soft_scores.objectness_score >= 0.58

        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        patch = Patch3D(
            patch_id=4,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            soft_scores=refined[0].soft_scores,
            metadata={"force_object_candidate": True},
        )
        _bg, obj, _amb = BgObjSplitModule({}).process([patch], BackgroundMap(), {})
        assert obj == [patch]

    def test_depth_refinement_opencv_components_match_python_backend(self):
        cv2 = pytest.importorskip("cv2")
        if not hasattr(cv2, "connectedComponentsWithStats"):
            pytest.skip("OpenCV connected-components API is unavailable")
        depth = np.full((12, 12), 2.0, dtype=np.float32)
        mask = np.zeros((12, 12), dtype=bool)
        mask[1:4, 1:4] = True
        mask[7:11, 7:11] = True
        proposal = Proposal2D(
            proposal_id=21,
            mask=mask,
            bbox_xyxy=np.array([1, 1, 11, 11], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.9,
        )

        python_module = DepthRefinementModule(
            {
                "depth_edge_threshold": 10.0,
                "min_mask_area_after_refine": 1,
                "connected_components_backend": "python",
                "use_bbox_crop": False,
            }
        )
        opencv_module = DepthRefinementModule(
            {
                "depth_edge_threshold": 10.0,
                "min_mask_area_after_refine": 1,
                "connected_components_backend": "opencv",
                "use_bbox_crop": False,
            }
        )

        python_refined = python_module.process(depth, [proposal])
        opencv_refined = opencv_module.process(depth, [proposal])

        assert [item.area for item in opencv_refined] == [item.area for item in python_refined]
        assert [item.mask.tolist() for item in opencv_refined] == [item.mask.tolist() for item in python_refined]

    def test_depth_refinement_uses_bbox_crop_for_component_split(self):
        depth = np.full((32, 32), 2.0, dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:15, 20:27] = True
        proposal = Proposal2D(
            proposal_id=22,
            mask=mask,
            bbox_xyxy=np.array([20, 10, 27, 15], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.9,
        )
        module = DepthRefinementModule(
            {
                "depth_edge_threshold": 10.0,
                "min_mask_area_after_refine": 1,
                "connected_components_backend": "python",
                "use_bbox_crop": True,
            }
        )
        seen_shapes = []
        original_split = module._split_connected_components

        def _recording_split(component_mask):
            seen_shapes.append(component_mask.shape)
            return original_split(component_mask)

        module._split_connected_components = _recording_split

        refined = module.process(depth, [proposal])

        assert len(refined) == 1
        assert refined[0].mask.shape == (32, 32)
        assert refined[0].area == int(mask.sum())
        assert seen_shapes == [(5, 7)]

    def test_depth_refinement_parallel_matches_serial_output(self):
        depth = np.full((48, 32), 2.0, dtype=np.float32)
        proposals = []
        proposal_id = 30
        for y in (0, 16, 32):
            for x in (0, 16):
                mask = np.zeros(depth.shape, dtype=bool)
                mask[y : y + 16, x : x + 16] = True
                proposals.append(
                    Proposal2D(
                        proposal_id=proposal_id,
                        mask=mask,
                        bbox_xyxy=np.array([x, y, x + 16, y + 16], dtype=np.float32),
                        area=int(mask.sum()),
                        confidence=0.9,
                    )
                )
                proposal_id += 1

        base_config = {
            "connected_components_backend": "python",
            "use_bbox_crop": True,
            "depth_edge_threshold": 10.0,
            "min_mask_area_after_refine": 1,
        }
        serial = DepthRefinementModule(base_config)
        parallel = DepthRefinementModule(
            {
                **base_config,
                "proposal_parallel_enabled": True,
                "proposal_parallel_workers": 2,
                "proposal_parallel_min_tasks": 2,
            }
        )

        serial_refined = serial.process(depth, proposals)
        parallel_refined = parallel.process(depth, proposals)

        assert parallel.last_parallel_used is True
        assert parallel.last_parallel_worker_count == 2
        assert [item.proposal_id for item in parallel_refined] == [
            item.proposal_id for item in serial_refined
        ]
        assert [item.area for item in parallel_refined] == [item.area for item in serial_refined]
        assert [item.mask.tolist() for item in parallel_refined] == [
            item.mask.tolist() for item in serial_refined
        ]

    def test_depth_refinement_rejects_unknown_connected_components_backend(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        module = DepthRefinementModule({"connected_components_backend": "bad"})

        with pytest.raises(ValueError, match="Unknown connected_components_backend"):
            module._split_connected_components(mask)


def test_active_set_caps_candidates_with_visible_priority():
    TestRoadmapRefactor().test_active_set_caps_candidates_with_visible_priority()


def test_active_set_cap_prioritizes_new_objects_after_visible_ids():
    TestRoadmapRefactor().test_active_set_cap_prioritizes_new_objects_after_visible_ids()


def test_association_caps_scored_candidates_before_expensive_scoring():
    TestRoadmapRefactor().test_association_caps_scored_candidates_before_expensive_scoring()


def test_association_geometry_score_count_respects_scored_cap():
    TestRoadmapRefactor().test_association_geometry_score_count_respects_scored_cap()


def test_association_scored_candidate_cap_stays_hard_for_many_vote_owners():
    TestRoadmapRefactor().test_association_scored_candidate_cap_stays_hard_for_many_vote_owners()


def test_dense_surface_skips_point_refresh_on_non_interval_frames():
    module = DenseSurfaceModule({"dense_surface_voxel": 0.02, "dense_surface_cap_per_object": 4000, "update_interval": 5})
    state = SystemState()
    obj = ObjectMap(
        object_id=7,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.1, 0.1, 1.1], dtype=np.float32),
        last_seen_frame=1,
    )
    patch = Patch3D(
        patch_id=3,
        points=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.05, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.1, 0.0, 1.0], dtype=np.float32),
        source_frame_id=1,
    )
    obj.observations.append(ObservationRecord(frame_id=1, patch=patch))
    state.objects[7] = obj

    module.process(state, frame_id=1)
    assert 7 not in state.dense_surface_map.entries

    module.process(state, frame_id=5)
    assert 7 in state.dense_surface_map.entries


def test_dense_surface_default_interval_preserves_current_frame_gate():
    module = DenseSurfaceModule({})
    state = SystemState()
    obj = ObjectMap(
        object_id=8,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.1, 0.1, 1.1], dtype=np.float32),
        last_seen_frame=1,
    )
    patch = Patch3D(
        patch_id=4,
        points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        source_frame_id=1,
    )
    obj.observations.append(ObservationRecord(frame_id=1, patch=patch))
    state.objects[8] = obj

    module.process(state, frame_id=5)

    assert 8 not in state.dense_surface_map.entries


def test_dense_surface_interval_does_not_sweep_stale_objects():
    module = DenseSurfaceModule({"update_interval": 5})
    state = SystemState()
    obj = ObjectMap(
        object_id=9,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.1, 0.1, 1.1], dtype=np.float32),
        last_seen_frame=4,
        last_dense_refresh_frame=4,
    )
    patch = Patch3D(
        patch_id=5,
        points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        source_frame_id=4,
    )
    obj.observations.append(ObservationRecord(frame_id=4, patch=patch))
    state.objects[9] = obj
    surface_calls = []

    def track_surface_points(obj_arg, frame_id_arg, **_kwargs):
        surface_calls.append((int(obj_arg.object_id), int(frame_id_arg)))
        return np.asarray(obj_arg.local_pcd, dtype=np.float32)

    module._surface_points_for_object = track_surface_points

    module.process(state, frame_id=5)

    assert surface_calls == []
    assert 9 not in state.dense_surface_map.entries


def test_dense_surface_interval_refreshes_pending_observation_points_without_local_supplement():
    module = DenseSurfaceModule({"update_interval": 5})
    state = SystemState()
    obj = ObjectMap(
        object_id=10,
        local_pcd=np.array([[9.0, 9.0, 9.0]], dtype=np.float32),
        centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.1, 0.1, 1.1], dtype=np.float32),
        last_seen_frame=1,
    )
    obj.debug["global_instance_substrate"] = {"owned_voxel_count": 10}
    patch = Patch3D(
        patch_id=6,
        points=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.05, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.1, 0.0, 1.0], dtype=np.float32),
        source_frame_id=1,
    )
    obj.observations.append(ObservationRecord(frame_id=1, patch=patch))
    state.objects[10] = obj

    module.process(state, frame_id=1)
    module.process(state, frame_id=5)

    entry = state.dense_surface_map.entries[10]
    assert obj.last_dense_refresh_frame == 5
    assert entry.points.tolist() == patch.points.tolist()


def test_dense_surface_interval_refresh_includes_pending_and_current_observations():
    module = DenseSurfaceModule({"update_interval": 5})
    state = SystemState()
    obj = ObjectMap(
        object_id=11,
        local_pcd=np.array([[9.0, 9.0, 9.0]], dtype=np.float32),
        centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.1, 0.1, 1.1], dtype=np.float32),
        last_seen_frame=1,
    )
    obj.debug["global_instance_substrate"] = {"owned_voxel_count": 10}
    pending_patch = Patch3D(
        patch_id=7,
        points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        source_frame_id=1,
    )
    current_patch = Patch3D(
        patch_id=8,
        points=np.array([[0.2, 0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.2, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.2, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.2, 0.0, 1.0], dtype=np.float32),
        source_frame_id=5,
    )
    obj.observations.append(ObservationRecord(frame_id=1, patch=pending_patch))
    state.objects[11] = obj

    module.process(state, frame_id=1)
    obj.last_seen_frame = 5
    obj.observations.append(ObservationRecord(frame_id=5, patch=current_patch))
    module.process(state, frame_id=5)

    entry = state.dense_surface_map.entries[11]
    np.testing.assert_allclose(
        entry.points,
        np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=np.float32),
    )


class TestRoadmapRefactor:
    """Regression tests for the overviewpro roadmap constraints."""

    def test_observation_identity_allows_same_label_update(self):
        from src.modules.observation_identity import classify_observation_identity

        patch = Patch3D(
            patch_id=1,
            points=np.zeros((1, 3), dtype=np.float32),
            centroid=np.zeros(3, dtype=np.float32),
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.zeros(3, dtype=np.float32),
            metadata={"anchor_class_name": " Cushion ", "anchor_confidence": 0.80},
        )
        obj = ObjectMap(object_id=2)
        obj.debug["anchor_semantics"] = {
            "canonical_label": "cushion",
            "canonical_score": 0.90,
            "semantic_state": "committed",
            "committed_label": "cushion",
            "label_max_confidence": {"cushion": 0.90},
        }
        decision = classify_observation_identity(patch, obj, min_patch_confidence=0.35)
        assert decision.can_update is True
        assert decision.relation == "same_label"
        assert decision.patch_label == "cushion"
        assert decision.object_label == "cushion"

    def test_observation_identity_blocks_confident_cross_label_update(self):
        from src.modules.observation_identity import classify_observation_identity

        patch = Patch3D(
            patch_id=8,
            points=np.zeros((1, 3), dtype=np.float32),
            centroid=np.zeros(3, dtype=np.float32),
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.zeros(3, dtype=np.float32),
            metadata={"anchor_class_name": "blanket", "anchor_confidence": 0.82},
        )
        obj = ObjectMap(object_id=31)
        obj.debug["anchor_semantics"] = {
            "canonical_label": "sofa",
            "canonical_score": 0.95,
            "semantic_state": "committed",
            "committed_label": "sofa",
            "label_max_confidence": {"sofa": 0.95},
        }
        decision = classify_observation_identity(patch, obj, min_patch_confidence=0.35)
        assert decision.can_update is False
        assert decision.relation == "cross_label_contested"
        assert decision.reason == "cross_label_observation_identity"
        assert decision.patch_label == "blanket"
        assert decision.object_label == "sofa"

    def test_observation_identity_does_not_block_against_uncommitted_parent_label(self):
        from src.modules.observation_identity import classify_observation_identity

        patch = Patch3D(
            patch_id=7,
            points=np.zeros((4, 3), dtype=np.float32),
            centroid=np.zeros(3, dtype=np.float32),
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.ones(3, dtype=np.float32),
            metadata={"anchor_class_name": "blanket", "anchor_confidence": 0.90},
        )
        obj = ObjectMap(object_id=2)
        obj.debug["anchor_semantics"] = {
            "canonical_label": "sofa",
            "semantic_state": "provisional",
            "committed_label": "",
            "label_max_confidence": {"sofa": 0.95},
        }

        decision = classify_observation_identity(patch, obj, min_patch_confidence=0.35)

        assert decision.can_update is True
        assert decision.relation == "uncommitted_object"
        assert decision.patch_label == "blanket"
        assert decision.object_label == ""

    def test_observation_identity_blocks_against_committed_parent_label(self):
        from src.modules.observation_identity import classify_observation_identity

        patch = Patch3D(
            patch_id=8,
            points=np.zeros((4, 3), dtype=np.float32),
            centroid=np.zeros(3, dtype=np.float32),
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.ones(3, dtype=np.float32),
            metadata={"anchor_class_name": "blanket", "anchor_confidence": 0.90},
        )
        obj = ObjectMap(object_id=3)
        obj.debug["anchor_semantics"] = {
            "canonical_label": "sofa",
            "semantic_state": "committed",
            "committed_label": "sofa",
            "label_max_confidence": {"sofa": 0.95},
        }

        decision = classify_observation_identity(patch, obj, min_patch_confidence=0.35)

        assert decision.can_update is False
        assert decision.relation == "cross_label_contested"
        assert decision.patch_label == "blanket"
        assert decision.object_label == "sofa"
        assert decision.reason == "cross_label_observation_identity"

    def test_observation_identity_treats_nonfinite_confidence_as_low_confidence(self):
        from src.modules.observation_identity import classify_observation_identity

        patch = Patch3D(
            patch_id=10,
            points=np.zeros((1, 3), dtype=np.float32),
            centroid=np.zeros(3, dtype=np.float32),
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.zeros(3, dtype=np.float32),
            metadata={
                "anchor_class_name": "blanket",
                "anchor_confidence": float("nan"),
            },
        )
        obj = ObjectMap(object_id=31)
        obj.debug["anchor_semantics"] = {
            "canonical_label": "sofa",
            "canonical_score": 0.95,
            "label_max_confidence": {"sofa": 0.95},
        }

        decision = classify_observation_identity(patch, obj, min_patch_confidence=0.35)

        assert decision.can_update is True
        assert decision.relation == "low_confidence_patch"
        assert decision.patch_confidence == 0.0

    def test_observation_identity_allows_unlabeled_patch_legacy_update(self):
        from src.modules.observation_identity import classify_observation_identity

        patch = Patch3D(
            patch_id=9,
            points=np.zeros((1, 3), dtype=np.float32),
            centroid=np.zeros(3, dtype=np.float32),
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.zeros(3, dtype=np.float32),
            metadata={},
        )
        obj = ObjectMap(object_id=31)
        obj.debug["anchor_semantics"] = {
            "canonical_label": "sofa",
            "canonical_score": 0.95,
            "label_max_confidence": {"sofa": 0.95},
        }
        decision = classify_observation_identity(patch, obj, min_patch_confidence=0.35)
        assert decision.can_update is True
        assert decision.relation == "unlabeled_patch"

    def test_run_report_includes_frontend_stage_diagnostics(self):
        frame_metrics = [
            {
                "frontend_stage": {
                    "source_sam_proposal_count": 4,
                    "anchor_voted_proposal_count": 4,
                    "runtime_merged_proposal_count": 3,
                    "runtime_merge_count": 1,
                    "anchor_voted_unanchored_count": 1,
                    "runtime_anchor_label_missing_edge_count": 2,
                    "runtime_anchor_label_mismatch_edge_count": 3,
                    "runtime_anchor_identity_mismatch_edge_count": 4,
                    "anchor_guided_sam": {
                        "source_sam_proposal_count": 4,
                        "anchored_proposal_count": 3,
                        "unknown_residual_count": 1,
                        "dropped_proposal_count": 2,
                        "mean_sam_candidates_per_anchor": 1.5,
                        "semantic_blocked_residual_count": 1,
                    },
                }
            }
        ]

        payload = build_frontend_stage_report_payload(frame_metrics)

        assert payload["source_sam_proposal_count_total"] == 4
        assert payload["anchor_voted_proposal_count_total"] == 4
        assert payload["runtime_merged_proposal_count_total"] == 3
        assert payload["runtime_merge_count_total"] == 1
        assert payload["anchor_voted_unanchored_count_total"] == 1
        assert payload["runtime_anchor_label_missing_edge_count_total"] == 2
        assert payload["runtime_anchor_label_mismatch_edge_count_total"] == 3
        assert payload["runtime_anchor_identity_mismatch_edge_count_total"] == 4
        assert payload["anchor_guided_sam_source_sam_proposal_count_total"] == 4
        assert payload["anchor_guided_sam_anchored_proposal_count_total"] == 3
        assert payload["anchor_guided_sam_unknown_residual_count_total"] == 1
        assert payload["anchor_guided_sam_dropped_proposal_count_total"] == 2
        assert payload["anchor_guided_sam_mean_sam_candidates_per_anchor_total"] == 1.5
        assert payload["anchor_guided_sam_semantic_blocked_residual_count_total"] == 1

    def test_anchor_frontend_uses_sam_masks_as_geometry_and_anchor_as_vote(self, tmp_path):
        pipeline_mod = __import__("src.pipelines.main_pipeline", fromlist=["Pipeline"])
        Pipeline = pipeline_mod.Pipeline
        config_path = tmp_path / "test_pipeline_anchor_vote.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "proposal:",
                    "  backend: placeholder",
                    "anchor_frontend:",
                    "  enabled: false",
                    "pipeline:",
                    "  verbose: false",
                    "logging:",
                    "  level: ERROR",
                    "runtime_vis:",
                    "  enabled: false",
                    "depth_refinement:",
                    "  min_mask_area_after_refine: 1",
                    "patch_lifting:",
                    "  min_points: 1",
                    "association:",
                    "  match_threshold: 999.0",
                    "object_update:",
                    "  surface_owner_gate:",
                    "    enabled: false",
                    "  current_frame_visibility_gate:",
                    "    enabled: false",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=config_path)
        pipe.object_anchor.enabled = True
        pipe.object_anchor.use_sam_intersection_proposals = True
        pipe.object_anchor.assignment_policy = "semantic_vote"

        mask = np.array([[True, False], [True, True]], dtype=bool)
        sam_proposal = Proposal2D(
            proposal_id=7,
            mask=mask,
            bbox_xyxy=np.array([0, 0, 2, 2], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.9,
            backend_name="sam2",
        )
        pipe.proposal.process = lambda *args, **kwargs: [sam_proposal]

        class _FakeAnchor:
            enabled = True
            use_sam_intersection_proposals = True
            assignment_policy = "semantic_vote"

            def generate_proposals(self, rgb, sam_proposals):
                proposal = sam_proposals[0]
                anchored = Proposal2D(
                    proposal_id=proposal.proposal_id,
                    mask=proposal.mask.copy(),
                    bbox_xyxy=proposal.bbox_xyxy.copy(),
                    area=proposal.area,
                    confidence=proposal.confidence,
                    backend_name="sam2_anchor_vote",
                    metadata={
                        "source": "sam2_anchor_vote",
                        "geometry_source": "sam2",
                        "anchor_id": 3,
                        "anchor_class_name": "chair",
                        "anchor_confidence": 0.8,
                        "anchor_vote_score": 0.8,
                    },
                )
                anchor = Anchor2D(
                    anchor_id=3,
                    bbox_xyxy=np.array([0, 0, 2, 2], dtype=np.float32),
                    class_name="chair",
                    confidence=0.8,
                )
                assignment = AnchorAssignment(
                    proposal_id=proposal.proposal_id,
                    anchor_id=3,
                    class_name="chair",
                    confidence=0.8,
                )
                return [anchor], [anchored], [assignment]

            def process(self, rgb, proposals):
                return [], []

        pipe.object_anchor = _FakeAnchor()

        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        depth = np.ones((2, 2), dtype=np.float32)
        pose = np.eye(4)
        intr = CameraIntrinsics(fx=1, fy=1, cx=0, cy=0, width=2, height=2)

        state = pipe.process_frame(rgb, depth, pose, intr)
        assert isinstance(state, SystemState)
        assert pipe.last_frame_debug["proposal_source"] == "sam2_anchor_vote"
        assert len(pipe.last_source_proposals) == 1
        assert len(pipe.last_raw_proposals) == 1
        assert np.array_equal(pipe.last_source_proposals[0].mask, mask)
        assert np.array_equal(pipe.last_raw_proposals[0].mask, mask)
        assert pipe.last_raw_proposals[0].metadata["anchor_class_name"] == "chair"

    def test_local_memory_audit_exposes_frontend_stage_records(self):
        source_mask = np.zeros((8, 8), dtype=bool)
        source_mask[1:4, 1:4] = True
        voted_mask = source_mask.copy()
        merged_mask = np.zeros((8, 8), dtype=bool)
        merged_mask[1:5, 1:4] = True

        source_proposal = Proposal2D(
            proposal_id=5,
            mask=source_mask,
            bbox_xyxy=np.array([1, 1, 4, 4], dtype=np.float32),
            area=int(source_mask.sum()),
            confidence=0.7,
            backend_name="sam2",
        )
        voted_proposal = Proposal2D(
            proposal_id=5,
            mask=voted_mask,
            bbox_xyxy=np.array([1, 1, 4, 4], dtype=np.float32),
            area=int(voted_mask.sum()),
            confidence=0.8,
            backend_name="sam2_anchor_vote",
            metadata={
                "anchor_id": 3,
                "anchor_class_name": "chair",
                "anchor_confidence": 0.9,
                "anchor_vote_score": 0.85,
                "source_raw_proposal_ids": [5],
                "geometry_source": "sam2",
                "anchor_label_strength": "weak_overlap",
                "mask_source": "weak_structure_anchor_box_clip",
                "anchor_proposal_coverage": 0.25,
                "anchor_anchor_coverage": 1.0,
                "anchor_blocked_candidates": [
                    {
                        "anchor_id": 7,
                        "class_name": "sofa",
                        "reason": "contained_subproposal_without_child_anchor",
                    }
                ],
            },
        )
        runtime_group = SimpleNamespace(
            group_id=9,
            member_mask_ids=[5],
            merged_mask=merged_mask,
            merged_bbox_xyxy=np.array([1, 1, 4, 5], dtype=np.float32),
            area=int(merged_mask.sum()),
            confidence=0.75,
            linked_object_id=None,
            whole_prior_used=False,
            merge_reason="connected_component_merge",
        )

        pipeline = SimpleNamespace(
            last_runtime_vis_output=SimpleNamespace(
                groups=[runtime_group],
                proposal_profiles=[],
                raw_to_group={5: 9},
            ),
            last_association=AssociationResult(),
            state=SystemState(),
            last_source_proposals=[source_proposal],
            last_raw_proposals=[voted_proposal],
            last_refined_proposals=[],
            last_patches=[],
            last_anchors=[],
            last_anchor_assignments=[],
            object_update=SimpleNamespace(last_current_frame_visibility_gate_stats={}),
            last_frame_debug={"bg_patch_ids": [], "obj_patch_ids": [], "amb_patch_ids": []},
            association=SimpleNamespace(match_threshold=0.45),
            patch_lifting=SimpleNamespace(min_points=12),
            depth_refinement=SimpleNamespace(min_area=16),
        )
        frame = SimpleNamespace(frame_id=0, depth=np.ones((8, 8), dtype=np.float32))

        audit = build_local_memory_frame_audit(
            frame=frame,
            pipeline=pipeline,
            prev_object_ids=set(),
            prev_next_object_id=0,
        )

        assert audit["source_sam_proposal_count"] == 1
        assert audit["anchor_voted_proposal_count"] == 1
        assert audit["runtime_merged_proposal_count"] == 1
        assert audit["source_sam_proposals"][0]["proposal_id"] == 5
        assert audit["anchor_voted_proposals"][0]["anchor_class_name"] == "chair"
        assert audit["anchor_voted_proposals"][0]["anchor_label_strength"] == "weak_overlap"
        assert audit["anchor_voted_proposals"][0]["mask_source"] == "weak_structure_anchor_box_clip"
        assert audit["anchor_voted_proposals"][0]["anchor_proposal_coverage"] == 0.25
        assert audit["anchor_voted_proposals"][0]["anchor_anchor_coverage"] == 1.0
        assert audit["anchor_voted_proposals"][0]["anchor_blocked_candidates"] == [
            {
                "anchor_id": 7,
                "class_name": "sofa",
                "reason": "contained_subproposal_without_child_anchor",
            }
        ]
        assert audit["runtime_merged_proposals"][0]["group_id"] == 9
        assert audit["runtime_merged_proposals"][0]["area"] == int(merged_mask.sum())

    def test_local_memory_audit_attributes_runtime_group_patches_to_all_members(self):
        left_mask = np.zeros((8, 8), dtype=bool)
        right_mask = np.zeros((8, 8), dtype=bool)
        merged_mask = np.zeros((8, 8), dtype=bool)
        left_mask[1:5, 1:3] = True
        right_mask[1:5, 4:6] = True
        merged_mask |= left_mask
        merged_mask |= right_mask

        raw_proposals = [
            Proposal2D(
                proposal_id=10,
                mask=left_mask,
                bbox_xyxy=np.array([1, 1, 3, 5], dtype=np.float32),
                area=int(left_mask.sum()),
                confidence=0.8,
                metadata={"anchor_id": 7, "anchor_class_name": "chair", "anchor_confidence": 0.9},
            ),
            Proposal2D(
                proposal_id=11,
                mask=right_mask,
                bbox_xyxy=np.array([4, 1, 6, 5], dtype=np.float32),
                area=int(right_mask.sum()),
                confidence=0.8,
                metadata={"anchor_id": 7, "anchor_class_name": "chair", "anchor_confidence": 0.9},
            ),
        ]
        runtime_group = SimpleNamespace(
            group_id=3,
            member_mask_ids=[10, 11],
            merged_mask=merged_mask,
            merged_bbox_xyxy=np.array([1, 1, 6, 5], dtype=np.float32),
            area=int(merged_mask.sum()),
            confidence=0.8,
            linked_object_id=None,
            whole_prior_used=False,
            merge_reason="connected_component_merge",
        )
        refined = RefinedProposal2D(
            proposal_id=30,
            mask=merged_mask,
            bbox_xyxy=np.array([1, 1, 6, 5], dtype=np.float32),
            area=int(merged_mask.sum()),
            confidence=0.8,
            metadata={"source_raw_proposal_id": 3},
        )
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        patch = Patch3D(
            patch_id=30,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            metadata={"source_proposal_id": 30},
        )

        state = SystemState(objects={0: ObjectMap(object_id=0)}, next_object_id=1)
        association = AssociationResult(new_object_patches=[30])

        pipeline = SimpleNamespace(
            last_runtime_vis_output=SimpleNamespace(
                groups=[runtime_group],
                proposal_profiles=[],
                raw_to_group={10: 3, 11: 3},
            ),
            last_association=association,
            state=state,
            last_source_proposals=raw_proposals,
            last_raw_proposals=raw_proposals,
            last_refined_proposals=[refined],
            last_patches=[patch],
            last_anchors=[],
            last_anchor_assignments=[],
            object_update=SimpleNamespace(last_current_frame_visibility_gate_stats={}),
            last_frame_debug={"bg_patch_ids": [], "obj_patch_ids": [30], "amb_patch_ids": []},
            association=SimpleNamespace(match_threshold=0.45),
            patch_lifting=SimpleNamespace(min_points=12),
            depth_refinement=SimpleNamespace(min_area=16),
        )
        frame = SimpleNamespace(frame_id=0, depth=np.ones((8, 8), dtype=np.float32))

        audit = build_local_memory_frame_audit(
            frame=frame,
            pipeline=pipeline,
            prev_object_ids=set(),
            prev_next_object_id=0,
        )

        by_id = {record["proposal_id"]: record for record in audit["raw_proposals"]}
        assert [record["refined_proposal_id"] for record in by_id[10]["refined_components"]] == [30]
        assert [record["refined_proposal_id"] for record in by_id[11]["refined_components"]] == [30]
        assert [record["patch_id"] for record in by_id[10]["patches"]] == [30]
        assert [record["patch_id"] for record in by_id[11]["patches"]] == [30]
        assert by_id[10]["final_outcome"] == "new_instance_created"
        assert by_id[11]["final_outcome"] == "new_instance_created"
        assert audit["new_instance_count"] == 1
        assert audit["new_instance_events"] == [
            {
                "object_id": 0,
                "source_proposal_id": 10,
                "source_patch_id": 30,
                "anchor_id": -1,
                "anchor_class_name": "",
                "anchor_confidence": 0.0,
                "reason": by_id[10]["reason"],
                "local_point_count": 0,
                "update_count": 0,
                "state": "active",
                "stability_score": 0.0,
            }
        ]

    def test_runtime_vis_blocks_cross_class_anchor_merges(self):
        depth = np.full((8, 8), 1.0, dtype=np.float32)
        frame = Frame(
            frame_id=0,
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            depth=depth,
            pose=np.eye(4),
            intrinsics=CameraIntrinsics(fx=5, fy=5, cx=4, cy=4, width=8, height=8),
        )
        mask_a = np.zeros((8, 8), dtype=bool)
        mask_b = np.zeros((8, 8), dtype=bool)
        mask_a[2:6, 2:4] = True
        mask_b[2:6, 4:6] = True
        proposals = [
            Proposal2D(
                proposal_id=0,
                mask=mask_a,
                bbox_xyxy=np.array([2, 2, 4, 6], dtype=np.float32),
                area=int(mask_a.sum()),
                confidence=0.9,
                metadata={"anchor_class_name": "sofa", "anchor_confidence": 0.9, "anchor_id": 0},
            ),
            Proposal2D(
                proposal_id=1,
                mask=mask_b,
                bbox_xyxy=np.array([4, 2, 6, 6], dtype=np.float32),
                area=int(mask_b.sum()),
                confidence=0.8,
                metadata={"anchor_class_name": "cushion", "anchor_confidence": 0.8, "anchor_id": 1},
            ),
        ]
        module = RuntimeVisModule(
            {
                "semantic_class_merge_gate_enabled": True,
                "merge_score_threshold": 0.1,
                "base_score_floor": 0.0,
                "edge_admission_min_adjacency": 0.0,
                "edge_admission_min_depth_continuity": 0.0,
                "edge_admission_min_plane_similarity": 0.0,
            }
        )

        output = module.process(frame, proposals, SystemState())

        assert len(output.merged_proposals) == 2
        assert output.merge_decisions[0].accepted is False
        assert output.merge_decisions[0].accepted_reason == "cross_class_anchor_conflict"

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

    def test_active_set_caps_candidates_with_visible_priority(self):
        module = ActiveSetModule(
            {
                "nearby_radius": 100.0,
                "whole_prior_threshold": 0.1,
                "max_candidate_ids": 5,
                "max_nearby_ids": 3,
                "max_whole_prior_ids": 2,
            }
        )
        state = SystemState()
        for object_id in range(10):
            obj = ObjectMap(
                object_id=object_id,
                local_pcd=np.array([[float(object_id), 0.0, 1.0]], dtype=np.float32),
                centroid=np.array([float(object_id), 0.0, 1.0], dtype=np.float32),
                bbox_min=np.array([float(object_id), 0.0, 1.0], dtype=np.float32),
                bbox_max=np.array([float(object_id) + 0.1, 0.1, 1.1], dtype=np.float32),
                whole_evidence=WholeEvidenceScores(whole_evidence_score=0.9),
            )
            state.objects[object_id] = obj

        module.tsdf_module.query_visible_instances = lambda volume, pose, intrinsics: {8, 9}

        active = module.process(
            state,
            np.eye(4, dtype=np.float64),
            CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8),
        )

        assert active.visible_ids == {8, 9}
        assert len(active.nearby_ids) <= 3
        assert len(active.whole_prior_ids) <= 2
        assert len(active.all_candidate_ids) <= 5
        assert {8, 9}.issubset(active.all_candidate_ids)

    def test_active_set_cap_prioritizes_new_objects_after_visible_ids(self):
        module = ActiveSetModule({"max_candidate_ids": 4})
        active = ActiveSet(
            visible_ids={8, 9},
            new_object_candidate_ids={5, 6},
            nearby_ids={0, 1},
            whole_prior_ids={2, 3},
        )

        capped = module.cap_active_set(active)

        assert capped.visible_ids == {8, 9}
        assert capped.new_object_candidate_ids == {5, 6}
        assert capped.nearby_ids == set()
        assert capped.whole_prior_ids == set()
        assert capped.all_candidate_ids == {5, 6, 8, 9}

    def test_association_limits_geometry_consistency_to_top_k_candidates(self):
        module = AssociationModule(
            {
                "match_threshold": 0.1,
                "voxel_vote_weight": 0.0,
                "centroid_distance_weight": 0.7,
                "bbox_overlap_weight": 0.1,
                "geometry_overlap_weight": 0.2,
                "max_geometry_candidates": 2,
            }
        )
        patch_points = np.array(
            [[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]],
            dtype=np.float32,
        )
        patch = Patch3D(
            patch_id=10,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )
        objects = {}
        for obj_id, offset in [(1, 0.01), (2, 0.20), (3, 1.20), (4, 1.60), (5, 2.00)]:
            pts = patch_points + np.array([offset, 0.0, 0.0], dtype=np.float32)
            objects[obj_id] = ObjectMap(
                object_id=obj_id,
                local_pcd=pts.copy(),
                centroid=pts.mean(axis=0),
                bbox_min=pts.min(axis=0),
                bbox_max=pts.max(axis=0),
            )

        called_object_ids: list[int] = []

        def fake_geometry(_patch, obj):
            called_object_ids.append(int(obj.object_id))
            return 1.0 if int(obj.object_id) == 1 else 0.25

        module._geometry_consistency = fake_geometry

        result = module.process([patch], objects, SystemState().tsdf_volume)

        assert called_object_ids == [1, 2]
        assert len(result.matched) == 1
        assert result.matched[0][1] == 1
        patch_debug = result.debug["per_patch"][10]
        assert patch_debug["candidate_count_before_geometry"] == 5
        assert patch_debug["geometry_candidate_object_ids"] == [1, 2]
        assert patch_debug["geometry_candidate_count"] == 2
        assert result.debug["summary"]["candidate_score_count"] == 5
        assert result.debug["summary"]["geometry_score_count"] == 2
        assert result.debug["summary"]["geometry_pruned_candidate_count"] == 3

    def test_association_caps_scored_candidates_before_expensive_scoring(self):
        module = AssociationModule(
            {
                "match_threshold": 0.1,
                "max_scored_candidates": 4,
                "max_geometry_candidates": 2,
                "geometry_candidate_min_cheap_score": 0.0,
                "score_parallel_enabled": False,
            }
        )
        patch = Patch3D(
            patch_id=1,
            points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            bbox_max=np.array([0.1, 0.1, 1.1], dtype=np.float32),
        )
        objects = {}
        for object_id in range(20):
            objects[object_id] = ObjectMap(
                object_id=object_id,
                local_pcd=np.array([[float(object_id), 0.0, 1.0]], dtype=np.float32),
                association_pcd=np.array([[float(object_id), 0.0, 1.0]], dtype=np.float32),
                centroid=np.array([float(object_id), 0.0, 1.0], dtype=np.float32),
                bbox_min=np.array([float(object_id), 0.0, 1.0], dtype=np.float32),
                bbox_max=np.array([float(object_id) + 0.1, 0.1, 1.1], dtype=np.float32),
            )
        active = ActiveSet(nearby_ids=set(objects))

        result = module.process([patch], objects, TSDFInstanceVolume(), active_set=active)
        debug = result.debug["per_patch"][1]

        assert debug["candidate_count_before_scored_cap"] == 20
        assert debug["candidate_count_after_scored_cap"] == 4
        assert len(debug["candidate_object_ids"]) == 4
        assert len(result.scores) == 4

    def test_association_geometry_score_count_respects_scored_cap(self):
        module = AssociationModule(
            {
                "match_threshold": 0.1,
                "voxel_vote_weight": 0.0,
                "centroid_distance_weight": 0.7,
                "bbox_overlap_weight": 0.1,
                "geometry_overlap_weight": 0.2,
                "max_scored_candidates": 2,
                "max_geometry_candidates": 4,
                "geometry_candidate_min_cheap_score": 0.0,
                "score_parallel_enabled": False,
            }
        )
        patch_points = np.array(
            [[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]],
            dtype=np.float32,
        )
        patch = Patch3D(
            patch_id=11,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )
        objects = {}
        for obj_id, offset in [(1, 0.01), (2, 0.20), (3, 0.40), (4, 0.60), (5, 0.80)]:
            pts = patch_points + np.array([offset, 0.0, 0.0], dtype=np.float32)
            objects[obj_id] = ObjectMap(
                object_id=obj_id,
                local_pcd=pts.copy(),
                centroid=pts.mean(axis=0),
                bbox_min=pts.min(axis=0),
                bbox_max=pts.max(axis=0),
            )

        called_object_ids: list[int] = []

        def fake_geometry(_patch, obj):
            called_object_ids.append(int(obj.object_id))
            return 0.25

        module._geometry_consistency = fake_geometry

        result = module.process([patch], objects, SystemState().tsdf_volume)
        patch_debug = result.debug["per_patch"][11]
        summary = result.debug["summary"]

        assert len(called_object_ids) == 2
        assert patch_debug["geometry_scored_count"] == 2
        assert summary["geometry_score_count"] == 2
        assert summary["geometry_pruned_candidate_count"] == (
            summary["candidate_score_count"] - summary["geometry_score_count"]
        )

    def test_association_scored_candidate_cap_stays_hard_for_many_vote_owners(self):
        module = AssociationModule({"max_scored_candidates": 3})
        cheap_by_id = {
            1: (0.10, 0.10, 0.0, 0.10),
            2: (0.30, 0.30, 0.0, 0.30),
            3: (0.20, 0.20, 0.0, 0.20),
            4: (0.90, 0.90, 0.0, 0.90),
            5: (0.80, 0.80, 0.0, 0.80),
        }
        vote = VoxelVoteResult(owner_votes={1: 1, 2: 1, 3: 1, 4: 1, 5: 1})

        selected = module._select_scored_candidate_ids([1, 2, 3, 4, 5], cheap_by_id, vote)

        assert selected == [4, 5, 2]

    def test_association_parallel_scoring_matches_serial_result(self):
        patch_points = np.array(
            [[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0], [0.05, 0.05, 1.0]],
            dtype=np.float32,
        )
        patch = Patch3D(
            patch_id=10,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )
        objects = {}
        for obj_id, offset in [(1, 0.02), (2, 0.35), (3, 0.70), (4, 1.10)]:
            pts = patch_points + np.array([offset, 0.0, 0.0], dtype=np.float32)
            objects[obj_id] = ObjectMap(
                object_id=obj_id,
                centroid=pts.mean(axis=0),
                bbox_min=pts.min(axis=0),
                bbox_max=pts.max(axis=0),
                local_pcd=pts.copy(),
            )

        base_config = {
            "match_threshold": 0.1,
            "voxel_vote_weight": 0.0,
            "centroid_distance_weight": 0.7,
            "bbox_overlap_weight": 0.1,
            "geometry_overlap_weight": 0.2,
            "max_geometry_candidates": 4,
            "score_parallel_min_candidates": 2,
        }
        serial = AssociationModule({**base_config, "score_parallel_enabled": False})
        parallel = AssociationModule(
            {
                **base_config,
                "score_parallel_enabled": True,
                "score_parallel_workers": 2,
            }
        )

        serial_result = serial.process([patch], objects, SystemState().tsdf_volume)
        parallel_result = parallel.process([patch], objects, SystemState().tsdf_volume)

        assert parallel_result.matched[0][0:2] == serial_result.matched[0][0:2]
        assert parallel_result.new_object_patches == serial_result.new_object_patches
        assert parallel_result.contested_object_patches == serial_result.contested_object_patches
        assert sorted(parallel_result.scores) == sorted(serial_result.scores)
        for key in sorted(serial_result.scores):
            assert parallel_result.scores[key].total_score == pytest.approx(serial_result.scores[key].total_score)
            assert parallel_result.scores[key].geometry_overlap == pytest.approx(serial_result.scores[key].geometry_overlap)
        assert parallel_result.debug["summary"]["score_parallel_used_count"] == 1
        assert parallel_result.debug["summary"]["score_parallel_candidate_count_total"] == 4

    def test_association_score_parallel_defaults_and_worker_caps(self, monkeypatch):
        import src.modules.association as association_module

        module = AssociationModule({})
        assert module.score_parallel_enabled is False
        assert module.score_parallel_workers == 0
        assert module.score_parallel_min_candidates == 16
        assert module._resolve_worker_count(configured_workers=8, task_count=3) == 3
        assert module._resolve_worker_count(configured_workers=0, task_count=0) == 1

        monkeypatch.setattr(association_module.os, "cpu_count", lambda: 64)
        assert module._resolve_worker_count(configured_workers=0, task_count=32) == 4

    def test_association_geometry_top_k_prioritizes_tsdf_vote_owner(self):
        module = AssociationModule(
            {
                "match_threshold": 0.1,
                "voxel_vote_weight": 0.7,
                "centroid_distance_weight": 0.3,
                "bbox_overlap_weight": 0.0,
                "geometry_overlap_weight": 0.0,
                "max_geometry_candidates": 1,
            }
        )
        patch_points = np.array(
            [[0.05, 0.0, 0.0], [0.06, 0.0, 0.0], [0.10, 0.0, 0.0], [0.11, 0.0, 0.0]],
            dtype=np.float32,
        )
        patch = Patch3D(
            patch_id=20,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )
        near_points = patch_points.copy()
        far_owner_points = patch_points + np.array([2.0, 0.0, 0.0], dtype=np.float32)
        objects = {
            1: ObjectMap(
                object_id=1,
                local_pcd=near_points.copy(),
                centroid=near_points.mean(axis=0),
                bbox_min=near_points.min(axis=0),
                bbox_max=near_points.max(axis=0),
            ),
            9: ObjectMap(
                object_id=9,
                local_pcd=far_owner_points.copy(),
                centroid=far_owner_points.mean(axis=0),
                bbox_min=far_owner_points.min(axis=0),
                bbox_max=far_owner_points.max(axis=0),
            ),
        }
        volume = TSDFInstanceVolume(voxel_size=0.05)
        volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({9: 2.0})
        volume.owner_support[(2, 0, 0)] = VoxelOwnerSupport({9: 2.0})
        called_object_ids: list[int] = []

        def fake_geometry(_patch, obj):
            called_object_ids.append(int(obj.object_id))
            return 0.0

        module._geometry_consistency = fake_geometry

        result = module.process([patch], objects, volume)

        assert called_object_ids == [9]
        assert result.debug["per_patch"][20]["geometry_candidate_object_ids"] == [9]

    def test_association_geometry_uses_association_pcd_when_present(self):
        module = AssociationModule(
            {
                "geometry_nn_patch_sample": 3,
                "geometry_nn_object_sample": 2,
            }
        )
        patch_points = np.array(
            [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.10, 0.0, 0.0]],
            dtype=np.float32,
        )
        far_local_points = patch_points + np.array([5.0, 0.0, 0.0], dtype=np.float32)
        association_points = patch_points.copy()
        patch = Patch3D(
            patch_id=30,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )
        obj = ObjectMap(
            object_id=8,
            local_pcd=far_local_points.copy(),
            association_pcd=association_points.copy(),
            centroid=far_local_points.mean(axis=0),
            bbox_min=far_local_points.min(axis=0),
            bbox_max=far_local_points.max(axis=0),
        )

        assert module._geometry_consistency(patch, obj) > 0.95
        np.testing.assert_allclose(module._object_geometry_points(obj), association_points)

    def test_object_update_refreshes_bounded_association_geometry_memory(self):
        module = ObjectUpdateModule(
            {"association_geometry": {"enabled": True, "voxel_size": 0.03, "max_points_per_object": 128}}
        )
        state = SystemState()
        first_points = np.array(
            [[float(i) * 0.04, 0.0, 0.0] for i in range(256)],
            dtype=np.float32,
        )
        first_patch = Patch3D(
            patch_id=40,
            points=first_points,
            centroid=first_points.mean(axis=0),
            bbox_min=first_points.min(axis=0),
            bbox_max=first_points.max(axis=0),
            source_frame_id=1,
        )

        created = module.process(AssociationResult(new_object_patches=[40]), [first_patch], state)
        obj = created.objects[0]
        assert len(obj.local_pcd) == 256
        assert len(obj.association_pcd) <= 128
        assert obj.debug["association_geometry"]["point_count"] == len(obj.association_pcd)
        assert obj.debug["association_geometry"]["max_points_per_object"] == 128
        assert obj.debug["association_geometry"]["voxel_size"] == pytest.approx(0.03)
        created_association_pcd = obj.association_pcd.copy()

        update_points = np.array(
            [[float(i) * 0.04, 1.0, 0.0] for i in range(256)],
            dtype=np.float32,
        )
        update_patch = Patch3D(
            patch_id=41,
            points=update_points,
            centroid=update_points.mean(axis=0),
            bbox_min=update_points.min(axis=0),
            bbox_max=update_points.max(axis=0),
            source_frame_id=2,
        )
        association = AssociationResult(matched=[(41, 0, AssociationScore(total_score=1.0))])

        updated = module.process(association, [update_patch], created)
        obj = updated.objects[0]

        assert len(obj.local_pcd) == 512
        assert len(obj.association_pcd) <= 128
        assert obj.debug["association_geometry"]["point_count"] == len(obj.association_pcd)
        assert obj.debug["association_geometry"]["source_point_count"] == 512
        assert obj.debug["association_geometry"]["role"] == "bounded_association_geometry"
        assert obj.debug["association_geometry"]["max_points_per_object"] == 128
        assert obj.debug["association_geometry"]["voxel_size"] == pytest.approx(0.03)
        assert not np.array_equal(obj.association_pcd, created_association_pcd)

    def test_object_update_disables_association_geometry_memory(self):
        module = ObjectUpdateModule({"association_geometry": {"enabled": False}})
        state = SystemState()
        points = np.array(
            [[float(i), 0.0, 0.0] for i in range(6)],
            dtype=np.float32,
        )
        patch = Patch3D(
            patch_id=42,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=1,
        )

        updated = module.process(AssociationResult(new_object_patches=[42]), [patch], state)
        obj = updated.objects[0]

        assert len(obj.local_pcd) == 6
        assert len(obj.association_pcd) == 0
        assert obj.debug["association_geometry"]["enabled"] is False
        assert obj.debug["association_geometry"]["point_count"] == 0

    def test_object_update_disabled_association_geometry_clears_existing_memory(self):
        module = ObjectUpdateModule({"association_geometry": {"enabled": False}})
        existing_points = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=np.float32,
        )
        stale_association_points = np.array(
            [[9.0, 9.0, 9.0], [9.1, 9.0, 9.0]],
            dtype=np.float32,
        )
        state = SystemState(
            objects={
                0: ObjectMap(
                    object_id=0,
                    local_pcd=existing_points.copy(),
                    association_pcd=stale_association_points.copy(),
                    centroid=existing_points.mean(axis=0),
                    bbox_min=existing_points.min(axis=0),
                    bbox_max=existing_points.max(axis=0),
                )
            }
        )
        update_points = np.array(
            [[0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
            dtype=np.float32,
        )
        patch = Patch3D(
            patch_id=43,
            points=update_points,
            centroid=update_points.mean(axis=0),
            bbox_min=update_points.min(axis=0),
            bbox_max=update_points.max(axis=0),
            source_frame_id=2,
        )

        updated = module.process(
            AssociationResult(matched=[(43, 0, AssociationScore(total_score=1.0))]),
            [patch],
            state,
        )
        obj = updated.objects[0]

        assert len(obj.local_pcd) == 5
        assert len(obj.association_pcd) == 0
        assert obj.debug["association_geometry"]["enabled"] is False
        assert obj.debug["association_geometry"]["point_count"] == 0
        assert obj.debug["association_geometry"]["source_point_count"] == 5

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

    def test_association_routes_confident_cross_label_overlap_to_contested_not_matched(self):
        module = AssociationModule(
            {
                "match_threshold": 0.1,
                "observation_identity_gate_enabled": True,
                "observation_identity_min_patch_confidence": 0.35,
            }
        )
        sofa_points = np.array(
            [[x * 0.05, y * 0.05, 1.0] for x in range(4) for y in range(4)],
            dtype=np.float32,
        )
        blanket_patch = Patch3D(
            patch_id=12,
            points=sofa_points.copy(),
            centroid=sofa_points.mean(axis=0),
            bbox_min=sofa_points.min(axis=0),
            bbox_max=sofa_points.max(axis=0),
            metadata={"anchor_class_name": "blanket", "anchor_confidence": 0.92},
        )
        sofa = ObjectMap(
            object_id=3,
            local_pcd=sofa_points.copy(),
            centroid=sofa_points.mean(axis=0),
            bbox_min=sofa_points.min(axis=0),
            bbox_max=sofa_points.max(axis=0),
        )
        sofa.debug["anchor_semantics"] = {
            "canonical_label": "sofa",
            "canonical_score": 0.95,
            "semantic_state": "committed",
            "committed_label": "sofa",
            "label_max_confidence": {"sofa": 0.95},
        }

        result = module.process([blanket_patch], {3: sofa}, SystemState().tsdf_volume)

        assert result.matched == []
        assert result.new_object_patches == []
        assert result.contested_object_patches == [12]
        assert len(result.contested_matches) == 1
        contested = result.contested_matches[0]
        assert contested.patch_id == 12
        assert contested.blocked_object_id == 3
        assert contested.patch_label == "blanket"
        assert contested.object_label == "sofa"
        assert contested.reason == "cross_label_observation_identity"
        assert contested.score is result.scores[(12, 3)]
        assert contested.debug["relation"] == "cross_label_contested"
        assert contested.debug["patch_confidence"] == pytest.approx(0.92)
        assert blanket_patch.metadata["contested_parent_object_id"] == 3
        assert blanket_patch.metadata["contested_parent_label"] == "sofa"
        assert blanket_patch.metadata["contested_patch_label"] == "blanket"
        assert blanket_patch.metadata["contested_reason"] == "cross_label_observation_identity"
        assert blanket_patch.metadata["semantic_split_candidate_from_object_id"] == 3
        assert blanket_patch.metadata["semantic_split_candidate_parent_label"] == "sofa"
        assert blanket_patch.metadata["semantic_split_candidate_new_label"] == "blanket"
        assert blanket_patch.metadata["semantic_split_candidate_reason"] == "cross_label_observation_identity"
        patch_debug = result.debug["per_patch"][12]
        assert patch_debug["final_outcome"] == "contested_residual"
        assert patch_debug["blocked_object_id"] == 3
        assert patch_debug["final_association_score"] == pytest.approx(contested.score.total_score)
        assert patch_debug["new_instance_created"] is False
        identity_blocked = patch_debug["observation_identity_blocked_candidates"]
        assert identity_blocked[0]["reason"] == "cross_label_observation_identity"
        assert patch_debug["semantic_conflict_blocked_candidates"] is identity_blocked
        assert patch_debug["blocked_candidate_source"] == "observation_identity"

    def test_association_blocks_semantic_conflict_subregion_matches(self):
        module = AssociationModule(
            {
                "match_threshold": 0.1,
                "observation_identity_gate_enabled": True,
                "observation_identity_min_patch_confidence": 0.35,
            }
        )
        obj_points = np.array(
            [[x * 0.05, y * 0.05, 1.0] for x in range(5) for y in range(5)],
            dtype=np.float32,
        )
        patch_points = obj_points[:4].copy()
        patch = Patch3D(
            patch_id=8,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
            metadata={"anchor_class_name": "cushion", "anchor_confidence": 0.8},
        )
        obj = ObjectMap(
            object_id=3,
            local_pcd=obj_points,
            centroid=obj_points.mean(axis=0),
            bbox_min=obj_points.min(axis=0),
            bbox_max=obj_points.max(axis=0),
        )
        obj.debug["anchor_semantics"] = {
            "canonical_label": "sofa",
            "canonical_score": 0.9,
            "semantic_state": "committed",
            "committed_label": "sofa",
            "label_max_confidence": {"sofa": 0.9},
        }

        result = module.process([patch], {3: obj}, SystemState().tsdf_volume)

        assert result.matched == []
        assert result.new_object_patches == []
        assert result.contested_object_patches == [8]
        assert len(result.contested_matches) == 1
        contested = result.contested_matches[0]
        assert contested.blocked_object_id == 3
        assert contested.patch_label == "cushion"
        assert contested.object_label == "sofa"
        assert contested.reason == "cross_label_observation_identity"
        assert patch.metadata["semantic_split_candidate_from_object_id"] == 3
        assert patch.metadata["semantic_split_candidate_parent_label"] == "sofa"
        assert patch.metadata["semantic_split_candidate_new_label"] == "cushion"
        assert patch.metadata["semantic_split_candidate_reason"] == "cross_label_observation_identity"
        assert patch.metadata["contested_parent_object_id"] == 3
        assert patch.metadata["contested_parent_label"] == "sofa"
        assert patch.metadata["contested_patch_label"] == "cushion"
        assert patch.metadata["contested_reason"] == "cross_label_observation_identity"
        blocked = result.debug["per_patch"][8]["semantic_conflict_blocked_candidates"]
        assert blocked[0]["reason"] == "cross_label_observation_identity"
        assert result.debug["per_patch"][8]["final_outcome"] == "contested_residual"

    def test_association_routes_contained_residual_to_contested_pool_before_parent_update(self):
        module = AssociationModule(
            {
                "match_threshold": 0.1,
                "observation_identity_gate_enabled": True,
                "observation_identity_min_patch_confidence": 0.35,
            }
        )
        parent_points = np.array(
            [[x * 0.05, y * 0.05, 1.0] for x in range(5) for y in range(5)],
            dtype=np.float32,
        )
        residual_points = parent_points[:6].copy()
        residual = Patch3D(
            patch_id=41,
            points=residual_points,
            centroid=residual_points.mean(axis=0),
            bbox_min=residual_points.min(axis=0),
            bbox_max=residual_points.max(axis=0),
            metadata={
                "anchor_class_name": "",
                "anchor_confidence": 0.0,
                "anchor_label_strength": "none",
                "semantic_commit_allowed": False,
                "residual_semantic_policy": "unknown",
                "mask_anchor_relation": "contained_residual",
                "anchor_blocked_candidates": [
                    {
                        "anchor_id": 7,
                        "class_name": "sofa",
                        "reason": "contained_subproposal_without_child_anchor",
                    }
                ],
            },
        )
        parent = ObjectMap(
            object_id=7,
            local_pcd=parent_points,
            centroid=parent_points.mean(axis=0),
            bbox_min=parent_points.min(axis=0),
            bbox_max=parent_points.max(axis=0),
        )
        parent.debug["anchor_semantics"] = {
            "canonical_label": "sofa",
            "canonical_score": 0.9,
            "semantic_state": "committed",
            "committed_label": "sofa",
            "label_max_confidence": {"sofa": 0.9},
        }

        result = module.process([residual], {7: parent}, SystemState().tsdf_volume)

        assert result.matched == []
        assert result.new_object_patches == []
        assert result.contested_object_patches == [41]
        contested = result.contested_matches[0]
        assert contested.blocked_object_id == 7
        assert contested.patch_label == ""
        assert contested.object_label == "sofa"
        assert contested.reason == "contained_residual_identity_guard"
        assert residual.metadata["contested_parent_object_id"] == 7
        assert residual.metadata["contested_parent_label"] == "sofa"
        assert residual.metadata["contested_patch_label"] == ""
        assert residual.metadata["contested_reason"] == "contained_residual_identity_guard"
        patch_debug = result.debug["per_patch"][41]
        assert patch_debug["final_outcome"] == "contested_residual"
        assert patch_debug["blocked_candidate_source"] == "observation_identity"

    def test_committed_parent_does_not_absorb_confident_child_patch(self):
        patch_points = np.array(
            [[0.05, 0.0, 0.0], [0.06, 0.0, 0.0], [0.10, 0.0, 0.0], [0.11, 0.0, 0.0]],
            dtype=np.float32,
        )
        sofa = ObjectMap(
            object_id=15,
            local_pcd=patch_points.copy(),
            centroid=np.array([0.1, 0.0, 0.0], dtype=np.float32),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )
        for frame_id in [18, 19]:
            sofa_observation = Patch3D(
                patch_id=frame_id,
                points=patch_points.copy(),
                centroid=patch_points.mean(axis=0),
                bbox_min=patch_points.min(axis=0),
                bbox_max=patch_points.max(axis=0),
                source_frame_id=frame_id,
                metadata={
                    "anchor_class_name": "sofa",
                    "anchor_confidence": 0.95,
                    "anchor_view_quality": 0.90,
                    "anchor_label_strength": "strong",
                },
            )
            accumulate_anchor_semantic_vote(sofa, sofa_observation)
        sofa_commit_state = object_semantic_commit_state(sofa)
        assert sofa_commit_state["semantic_state"] == "committed"
        assert sofa_commit_state["committed_label"] == "sofa"

        patch = Patch3D(
            patch_id=8,
            points=patch_points.copy(),
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
            source_frame_id=20,
            metadata={"anchor_class_name": "blanket", "anchor_confidence": 0.91},
        )
        assoc = AssociationModule(
            {
                "match_threshold": 0.5,
                "voxel_vote_weight": 1.0,
                "centroid_distance_weight": 0.0,
                "bbox_overlap_weight": 0.0,
                "geometry_overlap_weight": 0.0,
                "observation_identity_gate_enabled": True,
                "observation_identity_min_patch_confidence": 0.35,
            }
        )
        volume = TSDFInstanceVolume(voxel_size=0.05)
        volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({15: 2.0})
        volume.owner_support[(2, 0, 0)] = VoxelOwnerSupport({15: 2.0})

        result = assoc.process([patch], {15: sofa}, volume)

        assert result.matched == []
        assert result.contested_object_patches == [8]
        score = result.scores[(8, 15)]
        assert score.voxel_vote_score > 0.9
        assert score.total_score > assoc.match_threshold
        assert score.vote_result is not None
        assert score.vote_result.owner_votes == {15: 2}
        assert score.vote_result.supported_voxel_count == 2
        assert result.contested_matches[0].blocked_object_id == 15
        assert result.contested_matches[0].patch_label == "blanket"
        assert result.contested_matches[0].object_label == "sofa"

    def test_uncommitted_floor_like_parent_does_not_block_rug_geometry_recall(self):
        patch_points = np.array(
            [[0.05, 0.0, 0.0], [0.06, 0.0, 0.0], [0.10, 0.0, 0.0], [0.11, 0.0, 0.0]],
            dtype=np.float32,
        )
        floor_like = ObjectMap(
            object_id=41,
            local_pcd=patch_points.copy(),
            centroid=np.array([0.1, 0.0, 0.0], dtype=np.float32),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )
        floor_observation = Patch3D(
            patch_id=40,
            points=patch_points.copy(),
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
            source_frame_id=20,
            metadata={
                "anchor_class_name": "floor",
                "anchor_confidence": 0.88,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
            },
        )
        accumulate_anchor_semantic_vote(floor_like, floor_observation)
        floor_commit_state = object_semantic_commit_state(floor_like)
        assert floor_commit_state["semantic_state"] == "provisional"
        assert floor_commit_state["posterior_label"] == "floor"
        assert floor_commit_state["committed_label"] == ""

        patch = Patch3D(
            patch_id=9,
            points=patch_points.copy(),
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
            source_frame_id=21,
            metadata={"anchor_class_name": "rug", "anchor_confidence": 0.88},
        )
        assoc = AssociationModule(
            {
                "match_threshold": 0.5,
                "voxel_vote_weight": 1.0,
                "centroid_distance_weight": 0.0,
                "bbox_overlap_weight": 0.0,
                "geometry_overlap_weight": 0.0,
                "observation_identity_gate_enabled": True,
                "observation_identity_min_patch_confidence": 0.35,
            }
        )
        volume = TSDFInstanceVolume(voxel_size=0.05)
        volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({41: 2.0})
        volume.owner_support[(2, 0, 0)] = VoxelOwnerSupport({41: 2.0})

        result = assoc.process([patch], {41: floor_like}, volume)

        assert len(result.matched) == 1
        assert result.matched[0][1] == 41
        score = result.scores[(9, 41)]
        assert score.voxel_vote_score > 0.9
        assert score.total_score > assoc.match_threshold
        assert score.vote_result is not None
        assert score.vote_result.owner_votes == {41: 2}
        assert score.vote_result.supported_voxel_count == 2
        assert result.contested_object_patches == []

    def test_safe_provisional_export_label_does_not_create_association_conflict(self):
        from src.modules.semantic_memory import set_anchor_commit_policy, set_anchor_export_policy

        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        obj = ObjectMap(
            object_id=71,
            local_pcd=points.copy(),
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
        )
        source_patch = Patch3D(
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
        incoming_patch = Patch3D(
            patch_id=2,
            points=points.copy(),
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=2,
            metadata={
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.91,
                "anchor_view_quality": 0.85,
                "anchor_label_strength": "strong",
            },
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
            accumulate_anchor_semantic_vote(obj, source_patch)

            module = AssociationModule(
                {
                    "match_threshold": 0.1,
                    "semantic_conflict_gate_enabled": True,
                    "semantic_conflict_min_patch_confidence": 0.5,
                    "semantic_conflict_min_object_confidence": 0.5,
                    "observation_identity_gate_enabled": True,
                    "observation_identity_min_patch_confidence": 0.5,
                }
            )
            result = module.process([incoming_patch], {71: obj}, SystemState().tsdf_volume)

            assert object_export_semantic_label(obj) == "rug"
            assert object_identity_semantic_label(obj) == ""
            assert result.contested_matches == []
            assert result.matched[0][0] == 2
            assert result.matched[0][1] == 71
        finally:
            set_anchor_commit_policy(None)
            set_anchor_export_policy(None)

    def test_association_blocks_confident_cross_label_patch_against_repeated_provisional_identity(self):
        from src.modules.semantic_memory import set_anchor_commit_policy, set_anchor_export_policy

        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        obj = ObjectMap(
            object_id=231,
            local_pcd=points.copy(),
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
        )

        def rug_patch(frame_id: int) -> Patch3D:
            return Patch3D(
                patch_id=frame_id,
                points=points.copy(),
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=frame_id,
                metadata={
                    "anchor_class_name": "rug",
                    "anchor_confidence": 0.55,
                    "anchor_view_quality": 0.90,
                    "anchor_label_strength": "strong",
                },
            )

        incoming_patch = Patch3D(
            patch_id=99,
            points=points.copy(),
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=99,
            metadata={
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.91,
                "anchor_view_quality": 0.85,
                "anchor_label_strength": "strong",
            },
        )

        try:
            set_anchor_commit_policy(
                {
                    "min_frame_hits": 100,
                    "min_high_quality_hits": 100,
                    "min_weighted_score": 100.0,
                    "min_score_margin": 100.0,
                    "provisional_export_fallback": False,
                }
            )
            set_anchor_export_policy(
                {
                    "enabled": True,
                    "min_frame_hits": 100,
                    "min_high_quality_hits": 100,
                    "min_weighted_score": 100.0,
                    "min_score_margin": 100.0,
                    "min_confidence": 0.95,
                    "min_view_quality": 0.95,
                    "allow_single_frame_high_confidence": False,
                    "repeated_evidence_enabled": True,
                    "repeated_min_frame_hits": 3,
                    "repeated_min_weighted_score": 1.0,
                    "repeated_min_score_margin": 0.25,
                    "repeated_min_view_quality": 0.50,
                    "repeated_min_confidence": 0.25,
                }
            )
            accumulate_anchor_semantic_vote(obj, rug_patch(1))
            accumulate_anchor_semantic_vote(obj, rug_patch(2))
            accumulate_anchor_semantic_vote(obj, rug_patch(3))

            module = AssociationModule(
                {
                    "match_threshold": 0.1,
                    "observation_identity_gate_enabled": True,
                    "observation_identity_min_patch_confidence": 0.5,
                    "semantic_conflict_gate_enabled": True,
                    "semantic_conflict_min_patch_confidence": 0.5,
                    "semantic_conflict_min_object_confidence": 0.5,
                }
            )
            result = module.process([incoming_patch], {231: obj}, SystemState().tsdf_volume)

            assert result.matched == []
            assert result.contested_object_patches == [99]
            assert result.contested_matches[0].patch_label == "sofa"
            assert result.contested_matches[0].object_label == "rug"
            assert result.contested_matches[0].reason == "cross_label_observation_identity"
        finally:
            set_anchor_commit_policy(None)
            set_anchor_export_policy(None)

    def test_bg_obj_split_records_split_origin_metadata(self):
        module = BgObjSplitModule({})
        object_patch = Patch3D(
            patch_id=1,
            points=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            centroid=np.array([0.05, 0.0, 1.0], dtype=np.float32),
            bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            bbox_max=np.array([0.1, 0.0, 1.0], dtype=np.float32),
            soft_scores=ProposalSoftScores(objectness_score=0.8, backgroundness_score=0.1),
        )
        background_patch = Patch3D(
            patch_id=2,
            points=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32),
            centroid=np.array([0.05, 0.0, 0.0], dtype=np.float32),
            bbox_min=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            bbox_max=np.array([0.1, 0.0, 0.0], dtype=np.float32),
            soft_scores=ProposalSoftScores(objectness_score=0.1, backgroundness_score=0.8),
        )
        ambiguous_patch = Patch3D(
            patch_id=3,
            points=np.array([[0.0, 0.0, 1.2], [0.1, 0.0, 1.2]], dtype=np.float32),
            centroid=np.array([0.05, 0.0, 1.2], dtype=np.float32),
            bbox_min=np.array([0.0, 0.0, 1.2], dtype=np.float32),
            bbox_max=np.array([0.1, 0.0, 1.2], dtype=np.float32),
            soft_scores=ProposalSoftScores(
                objectness_score=0.45,
                backgroundness_score=0.45,
                attachedness_score=0.7,
            ),
        )

        bg_patches, obj_patches, amb_patches = module.process(
            [object_patch, background_patch, ambiguous_patch],
            BackgroundMap(),
            {},
        )

        assert [patch.metadata["split_origin"] for patch in obj_patches] == ["object"]
        assert [patch.metadata["split_origin"] for patch in bg_patches] == ["background"]
        assert [patch.metadata["split_origin"] for patch in amb_patches] == ["ambiguous"]

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

    def test_object_update_marks_tsdf_backbone_as_global_and_local_pcd_as_pool_geometry(self):
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
        assert obj.debug["local_geometry_memory"]["role"] == "object_pool_geometry"
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
        assert len(mem.debug["effective_feature_weights"]) == 1

    def test_object_debug_refresh_preserves_anchor_semantic_evidence_ledger(self):
        obj = ObjectMap(object_id=14)
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        patch1 = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=1,
            metadata={
                "anchor_class_name": "chair",
                "anchor_confidence": 0.65,
                "anchor_view_quality": 0.50,
                "anchor_label_strength": "strong",
            },
        )
        patch2 = Patch3D(
            patch_id=2,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=2,
            metadata={
                "anchor_class_name": "chair",
                "anchor_confidence": 0.70,
                "anchor_view_quality": 0.60,
                "anchor_label_strength": "strong",
            },
        )

        accumulate_anchor_semantic_vote(obj, patch1)
        ObjectUpdateModule({})._refresh_object_debug(obj, TSDFInstanceVolume(voxel_size=1.0))

        anchor_state = obj.debug["anchor_semantics"]
        assert anchor_state["evidence"][0]["label"] == "chair"
        assert anchor_state["evidence"][0]["frame_id"] == 1

        accumulate_anchor_semantic_vote(obj, patch2)

        anchor_state = obj.debug["anchor_semantics"]
        assert anchor_state["label_frame_hits"] == {"chair": 2}
        assert anchor_state["label_seen_frames"] == {"chair": [1, 2]}

    def test_object_debug_refresh_preserves_committed_anchor_semantic_state(self):
        from src.modules.semantic_memory import object_semantic_commit_state

        obj = ObjectMap(object_id=15)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            128,
            axis=0,
        )

        for frame_id in [1, 2]:
            patch = Patch3D(
                patch_id=frame_id,
                points=points,
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=frame_id,
                metadata={
                    "anchor_class_name": "chair",
                    "anchor_confidence": 0.85,
                    "anchor_view_quality": 0.8,
                    "anchor_label_strength": "strong",
                },
            )
            accumulate_anchor_semantic_vote(obj, patch)

        ObjectUpdateModule({})._refresh_object_debug(obj, TSDFInstanceVolume(voxel_size=1.0))

        assert object_semantic_commit_state(obj)["committed_label"] == "chair"
        assert preferred_object_semantic_label(obj) == "chair"

    def test_anchor_semantic_vote_allows_later_high_confidence_relabel(self):
        obj = ObjectMap(object_id=11)
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        for frame_id in range(2):
            patch = Patch3D(
                patch_id=frame_id,
                points=np.repeat(points, 128, axis=0),
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=frame_id,
                metadata={
                    "anchor_class_name": "chair",
                    "anchor_confidence": 0.85,
                    "anchor_view_quality": 0.8,
                },
            )
            accumulate_anchor_semantic_vote(obj, patch)

        assert preferred_object_semantic_label(obj) == "chair"

        for frame_id in [10, 11]:
            better_patch = Patch3D(
                patch_id=frame_id,
                points=np.repeat(points, 128, axis=0),
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=frame_id,
                metadata={
                    "anchor_class_name": "cabinet",
                    "anchor_confidence": 0.95,
                    "anchor_view_quality": 0.9,
                },
            )
            accumulate_anchor_semantic_vote(obj, better_patch)

        anchor_state = obj.debug["anchor_semantics"]
        assert preferred_object_semantic_label(obj) == "cabinet"
        assert anchor_state["label_high_quality_hits"]["cabinet"] == 2
        assert anchor_state["relabel_events"][-1]["old_label"] == "chair"
        assert anchor_state["relabel_events"][-1]["new_label"] == "cabinet"

    def test_anchor_semantic_drops_committed_label_during_provisional_relabel(self):
        from src.modules.semantic_memory import object_semantic_commit_state

        obj = ObjectMap(object_id=16)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            128,
            axis=0,
        )
        obj.debug["anchor_semantics"] = {
            "semantic_state": "committed",
            "committed_label": "chair",
            "canonical_label": "chair",
            "canonical_score": 1.0,
            "evidence": [],
            "relabel_events": [],
            "posterior_relabel_events": [],
        }

        cabinet_patch = Patch3D(
            patch_id=10,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=10,
            metadata={
                "anchor_class_name": "cabinet",
                "anchor_confidence": 0.98,
                "anchor_view_quality": 0.95,
                "anchor_label_strength": "strong",
            },
        )
        accumulate_anchor_semantic_vote(obj, cabinet_patch)

        state = object_semantic_commit_state(obj)
        assert state["posterior_label"] == "cabinet"
        assert state["semantic_state"] == "provisional"
        assert state["committed_label"] == ""
        assert preferred_object_semantic_label(obj) == ""

    def test_preferred_label_does_not_fallback_to_semantic_memory_for_uncommitted_anchor(self):
        from src.modules.semantic_memory import set_anchor_commit_policy

        obj = ObjectMap(object_id=17)
        obj.debug["anchor_semantics"] = {
            "canonical_label": "rug",
            "semantic_state": "provisional",
            "committed_label": "",
        }
        obj.semantic_memory = SemanticMemory(label_hypotheses=[("chair", 0.99)])

        try:
            set_anchor_commit_policy({"provisional_export_fallback": False})
            assert preferred_object_semantic_label(obj) == ""

            set_anchor_commit_policy({"provisional_export_fallback": True})
            assert preferred_object_semantic_label(obj) == ""
            assert object_export_semantic_label(obj) == ""
        finally:
            set_anchor_commit_policy(None)

    def test_anchor_semantic_commit_drops_stale_commit_when_final_margin_is_insufficient(self):
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            128,
            axis=0,
        )

        def make_patch(patch_id: int, label: str) -> Patch3D:
            return Patch3D(
                patch_id=patch_id,
                points=points,
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=patch_id,
                metadata={
                    "anchor_class_name": label,
                    "anchor_confidence": 0.90,
                    "anchor_view_quality": 0.90,
                    "anchor_label_strength": "strong",
                },
            )

        def accumulate(labels: list[str]) -> ObjectMap:
            obj = ObjectMap(object_id=len(labels))
            for patch_id, label in enumerate(labels, start=1):
                accumulate_anchor_semantic_vote(obj, make_patch(patch_id, label))
            return obj

        forward = accumulate(["sofa", "sofa", "blanket", "blanket"])
        reverse = accumulate(["blanket", "blanket", "sofa", "sofa"])
        forward_state = object_semantic_commit_state(forward)
        reverse_state = object_semantic_commit_state(reverse)

        assert forward_state["committed_label"] == reverse_state["committed_label"] == ""
        assert forward_state["semantic_state"] == reverse_state["semantic_state"] == "provisional"
        assert preferred_object_semantic_label(forward) == ""
        assert preferred_object_semantic_label(reverse) == ""

    def test_anchor_semantic_vote_is_order_invariant_for_same_evidence(self):
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            128,
            axis=0,
        )

        def make_patch(
            patch_id: int,
            label: str,
            confidence: float,
            view_quality: float,
        ) -> Patch3D:
            return Patch3D(
                patch_id=patch_id,
                points=points,
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=patch_id,
                metadata={
                    "anchor_class_name": label,
                    "anchor_confidence": confidence,
                    "anchor_view_quality": view_quality,
                    "anchor_label_strength": "strong",
                },
            )

        evidence = [
            make_patch(1, "sofa", 0.60, 0.70),
            make_patch(2, "sofa", 0.60, 0.70),
            make_patch(3, "blanket", 0.95, 0.90),
            make_patch(4, "blanket", 0.95, 0.90),
        ]

        forward = ObjectMap(object_id=21)
        reverse = ObjectMap(object_id=22)
        for patch in evidence:
            accumulate_anchor_semantic_vote(forward, patch)
        for patch in reversed(evidence):
            accumulate_anchor_semantic_vote(reverse, patch)

        assert (
            forward.debug["anchor_semantics"]["label_weighted_score"]
            == reverse.debug["anchor_semantics"]["label_weighted_score"]
        )
        assert (
            forward.debug["anchor_semantics"]["label_recent_score"]
            == reverse.debug["anchor_semantics"]["label_recent_score"]
        )
        assert (
            forward.debug["anchor_semantics"]["canonical_label"]
            == reverse.debug["anchor_semantics"]["canonical_label"]
        )
        assert preferred_object_semantic_label(forward) == "blanket"
        assert preferred_object_semantic_label(reverse) == "blanket"

    def test_anchor_semantic_posterior_requires_multiframe_commit(self):
        from src.modules.semantic_memory import object_semantic_commit_state

        obj = ObjectMap(object_id=31)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            128,
            axis=0,
        )

        first_patch = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=10,
            metadata={
                "anchor_class_name": "blanket",
                "anchor_confidence": 0.92,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
            },
        )
        accumulate_anchor_semantic_vote(obj, first_patch)

        state = object_semantic_commit_state(obj)
        assert state["semantic_state"] == "provisional"
        assert state["posterior_label"] == "blanket"
        assert state["committed_label"] == ""
        assert preferred_object_semantic_label(obj) == ""

        second_patch = Patch3D(
            patch_id=2,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=12,
            metadata={
                "anchor_class_name": "blanket",
                "anchor_confidence": 0.88,
                "anchor_view_quality": 0.85,
                "anchor_label_strength": "strong",
            },
        )
        accumulate_anchor_semantic_vote(obj, second_patch)

        state = object_semantic_commit_state(obj)
        assert state["semantic_state"] == "committed"
        assert state["posterior_label"] == "blanket"
        assert state["committed_label"] == "blanket"
        assert preferred_object_semantic_label(obj) == "blanket"

    def test_malformed_anchor_semantic_scores_do_not_break_export_state(self):
        obj = ObjectMap(object_id=135)
        obj.debug["anchor_semantics"] = {
            "canonical_label": "rug",
            "canonical_score": "bad",
            "semantic_state": "provisional",
            "committed_label": "",
            "label_weighted_score": {"rug": "bad", "sofa": "also_bad"},
            "label_frame_hits": {"rug": "bad"},
            "label_high_quality_hits": {"rug": "bad"},
            "label_max_confidence": {"rug": "bad"},
            "label_best_view_quality": {"rug": "bad"},
            "evidence": [{"label": "rug"}],
        }

        assert object_semantic_commit_state(obj)["posterior_score"] == 0.0

        export_state = object_export_semantic_state(obj)
        assert object_export_semantic_label(obj) == ""
        assert export_state["export_state"] == "unlabeled"
        assert export_state["export_source"] == "none"
        assert export_state["export_reason"].startswith("waiting_for_export:")

    def test_safe_provisional_anchor_exports_without_identity_commit(self):
        from src.modules.semantic_memory import set_anchor_commit_policy, set_anchor_export_policy

        obj = ObjectMap(object_id=131)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            128,
            axis=0,
        )
        patch = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=10,
            metadata={
                "anchor_class_name": "rug",
                "anchor_confidence": 0.92,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
            },
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

            commit_state = object_semantic_commit_state(obj)
            export_state = object_export_semantic_state(obj)

            assert commit_state["semantic_state"] == "provisional"
            assert commit_state["committed_label"] == ""
            assert object_identity_semantic_label(obj) == ""
            assert preferred_object_semantic_label(obj) == ""
            assert object_export_semantic_label(obj) == "rug"
            assert export_state["export_label"] == "rug"
            assert export_state["export_state"] == "safe_provisional"
            assert export_state["export_source"] == "anchor_safe_provisional"
            assert export_state["semantic_state"] == "provisional"
        finally:
            set_anchor_commit_policy(None)
            set_anchor_export_policy(None)

    def test_repeated_provisional_anchor_label_becomes_association_identity_guard_only(self):
        from src.modules.semantic_memory import set_anchor_commit_policy, set_anchor_export_policy

        obj = ObjectMap(object_id=230)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )

        def make_patch(frame_id: int) -> Patch3D:
            return Patch3D(
                patch_id=frame_id,
                points=points.copy(),
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=frame_id,
                metadata={
                    "anchor_class_name": "rug",
                    "anchor_confidence": 0.55,
                    "anchor_view_quality": 0.90,
                    "anchor_label_strength": "strong",
                },
            )

        try:
            set_anchor_commit_policy(
                {
                    "min_frame_hits": 100,
                    "min_high_quality_hits": 100,
                    "min_weighted_score": 100.0,
                    "min_score_margin": 100.0,
                    "provisional_export_fallback": False,
                }
            )
            set_anchor_export_policy(
                {
                    "enabled": True,
                    "min_frame_hits": 100,
                    "min_high_quality_hits": 100,
                    "min_weighted_score": 100.0,
                    "min_score_margin": 100.0,
                    "min_confidence": 0.95,
                    "min_view_quality": 0.95,
                    "allow_single_frame_high_confidence": False,
                    "repeated_evidence_enabled": True,
                    "repeated_min_frame_hits": 3,
                    "repeated_min_weighted_score": 1.0,
                    "repeated_min_score_margin": 0.25,
                    "repeated_min_view_quality": 0.50,
                    "repeated_min_confidence": 0.25,
                }
            )

            accumulate_anchor_semantic_vote(obj, make_patch(1))
            accumulate_anchor_semantic_vote(obj, make_patch(2))
            accumulate_anchor_semantic_vote(obj, make_patch(3))

            assert object_semantic_commit_state(obj)["semantic_state"] == "provisional"
            assert object_identity_semantic_label(obj) == ""
            assert preferred_object_semantic_label(obj) == ""
            assert object_association_identity_semantic_label(obj) == "rug"
            assert object_export_semantic_label(obj) == "rug"
            assert object_export_semantic_state(obj)["export_state"] == "safe_provisional"
        finally:
            set_anchor_commit_policy(None)
            set_anchor_export_policy(None)

    def test_contextual_overlap_evidence_never_exports_as_safe_provisional(self):
        from src.modules.semantic_memory import set_anchor_export_policy

        obj = ObjectMap(object_id=132)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        weak_patch = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=1,
            metadata={
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.99,
                "anchor_label_strength": "weak_overlap",
            },
        )

        try:
            set_anchor_export_policy(
                {
                    "enabled": True,
                    "min_frame_hits": 1,
                    "min_high_quality_hits": 1,
                    "min_weighted_score": 0.50,
                    "min_score_margin": 0.0,
                    "min_confidence": 0.50,
                    "min_view_quality": 0.0,
                    "allow_single_frame_high_confidence": True,
                    "single_frame_min_confidence": 0.50,
                    "single_frame_min_view_quality": 0.0,
                }
            )

            accumulate_anchor_semantic_vote(obj, weak_patch)

            export_state = object_export_semantic_state(obj)
            assert obj.debug["anchor_semantics"]["contextual_label_score_sum"] == {"sofa": 0.99}
            assert object_export_semantic_label(obj) == ""
            assert export_state["export_state"] == "unlabeled"
            assert export_state["export_source"] == "none"
            assert export_state["export_reason"] == "no_direct_strong_anchor_evidence"
        finally:
            set_anchor_export_policy(None)

    def test_close_provisional_posterior_margin_is_not_exported(self):
        from src.modules.semantic_memory import set_anchor_commit_policy, set_anchor_export_policy

        obj = ObjectMap(object_id=133)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )

        def make_patch(patch_id: int, label: str, confidence: float) -> Patch3D:
            return Patch3D(
                patch_id=patch_id,
                points=points,
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=patch_id,
                metadata={
                    "anchor_class_name": label,
                    "anchor_confidence": confidence,
                    "anchor_view_quality": 0.80,
                    "anchor_label_strength": "strong",
                },
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
                    "min_weighted_score": 0.50,
                    "min_score_margin": 0.20,
                    "min_confidence": 0.50,
                    "min_view_quality": 0.25,
                    "allow_single_frame_high_confidence": True,
                    "single_frame_min_confidence": 0.50,
                    "single_frame_min_view_quality": 0.25,
                }
            )

            accumulate_anchor_semantic_vote(obj, make_patch(1, "sofa", 0.90))
            accumulate_anchor_semantic_vote(obj, make_patch(2, "blanket", 0.86))

            export_state = object_export_semantic_state(obj)
            assert object_semantic_commit_state(obj)["semantic_state"] == "provisional"
            assert export_state["posterior_label"] == "sofa"
            assert object_export_semantic_label(obj) == ""
            assert export_state["export_state"] == "unlabeled"
            assert export_state["export_reason"].startswith("waiting_for_export:")
            assert "margin=" in export_state["export_reason"]
        finally:
            set_anchor_commit_policy(None)
            set_anchor_export_policy(None)

    def test_committed_anchor_export_still_wins_over_safe_provisional_policy(self):
        from src.modules.semantic_memory import set_anchor_export_policy

        obj = ObjectMap(object_id=134)
        obj.debug["anchor_semantics"] = {
            "semantic_state": "committed",
            "committed_label": "chair",
            "canonical_label": "chair",
            "canonical_score": 1.25,
            "canonical_frame_hits": 2,
            "canonical_confidence": 0.88,
            "canonical_best_view_quality": 0.80,
            "commit_reason": "multiframe_strong_anchor_evidence",
            "label_weighted_score": {"chair": 1.25},
            "label_frame_hits": {"chair": 2},
            "label_high_quality_hits": {"chair": 2},
            "label_max_confidence": {"chair": 0.88},
            "label_best_view_quality": {"chair": 0.80},
            "evidence": [],
        }

        try:
            set_anchor_export_policy({"enabled": False})

            export_state = object_export_semantic_state(obj)
            assert object_identity_semantic_label(obj) == "chair"
            assert preferred_object_semantic_label(obj) == "chair"
            assert object_export_semantic_label(obj) == "chair"
            assert export_state["export_label"] == "chair"
            assert export_state["export_state"] == "committed"
            assert export_state["export_source"] == "anchor_committed"
            assert export_state["export_reason"] == "committed_anchor_semantic"
        finally:
            set_anchor_export_policy(None)

    def test_promoted_provisional_requires_semantic_commit_before_export_label(self):
        updater = ObjectUpdateModule(
            {
                "provisional_pool": {
                    "enabled": True,
                    "promotion_hits": 2,
                    "match_distance": 1.0,
                    "downsample_voxel_size": 0.01,
                    "max_points_per_object": 1000,
                }
            }
        )
        state = SystemState()
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        patch1 = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=1,
            metadata={
                "anchor_class_name": "rug",
                "anchor_confidence": 0.90,
                "anchor_view_quality": 0.80,
                "anchor_label_strength": "strong",
            },
        )
        shifted = points + np.array([0.02, 0.0, 0.0], dtype=np.float32)
        patch2 = Patch3D(
            patch_id=2,
            points=shifted,
            centroid=shifted.mean(axis=0),
            bbox_min=shifted.min(axis=0),
            bbox_max=shifted.max(axis=0),
            source_frame_id=2,
            metadata={
                "anchor_class_name": "rug",
                "anchor_confidence": 0.91,
                "anchor_view_quality": 0.85,
                "anchor_label_strength": "strong",
            },
        )

        updater._upsert_provisional_object(state, patch1)
        updater._upsert_provisional_object(state, patch2)
        updater._promote_stable_provisionals(state)

        assert len(state.objects) == 1
        obj = next(iter(state.objects.values()))
        assert obj.debug["anchor_semantics"]["semantic_state"] == "committed"
        assert obj.debug["anchor_semantics"]["committed_label"] == "rug"
        assert preferred_object_semantic_label(obj) == "rug"

    def test_promoted_contested_residual_keeps_parent_context_out_of_commit(self):
        updater = ObjectUpdateModule(
            {
                "provisional_pool": {
                    "enabled": True,
                    "promotion_hits": 2,
                    "match_distance": 1.0,
                    "downsample_voxel_size": 0.01,
                    "max_points_per_object": 1000,
                },
                "contested_residual_pool": {"enabled": True},
            }
        )
        state = SystemState()
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        for frame_id in [1, 2]:
            shifted = points + np.array([0.01 * frame_id, 0.0, 0.0], dtype=np.float32)
            patch = Patch3D(
                patch_id=frame_id,
                points=shifted,
                centroid=shifted.mean(axis=0),
                bbox_min=shifted.min(axis=0),
                bbox_max=shifted.max(axis=0),
                source_frame_id=frame_id,
                metadata={
                    "anchor_class_name": "blanket",
                    "anchor_confidence": 0.92,
                    "anchor_view_quality": 0.90,
                    "anchor_label_strength": "strong",
                    "contested_parent_object_id": 15,
                    "contested_parent_label": "sofa",
                    "contested_patch_label": "blanket",
                    "contested_reason": "cross_label_observation_identity",
                },
            )
            updater._upsert_provisional_object(state, patch, contested=True)

        updater._promote_stable_provisionals(state)

        obj = next(iter(state.objects.values()))
        assert obj.debug["promoted_contested_residual"]["parent_label"] == "sofa"
        assert obj.debug["anchor_semantics"]["contextual_label_score_sum"] == {}
        assert obj.debug["anchor_semantics"]["committed_label"] == "blanket"
        assert preferred_object_semantic_label(obj) == "blanket"

    def test_object_semantic_commit_state_reports_missing_anchor_semantics(self):
        from src.modules.semantic_memory import object_semantic_commit_state

        obj = ObjectMap(object_id=99)

        assert object_semantic_commit_state(obj) == {
            "semantic_state": "unlabeled",
            "posterior_label": "",
            "committed_label": "",
            "posterior_score": 0.0,
            "commit_reason": "no_anchor_semantics",
        }

    def test_anchor_semantic_vote_tie_breaks_by_label_deterministically(self):
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        obj = ObjectMap(object_id=23)

        for patch_id, label in [(1, "blanket"), (2, "sofa")]:
            patch = Patch3D(
                patch_id=patch_id,
                points=points,
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=patch_id,
                metadata={
                    "anchor_class_name": label,
                    "anchor_confidence": 0.80,
                    "anchor_view_quality": 0.75,
                    "anchor_label_strength": "strong",
                },
            )
            accumulate_anchor_semantic_vote(obj, patch)

        assert obj.debug["anchor_semantics"]["label_frame_hits"] == {
            "blanket": 1,
            "sofa": 1,
        }
        assert obj.debug["anchor_semantics"]["canonical_label"] == "sofa"
        assert obj.debug["anchor_semantics"]["semantic_state"] == "provisional"
        assert preferred_object_semantic_label(obj) == ""

    def test_anchor_semantic_vote_ignores_blocked_and_weak_overlap_labels(self):
        obj = ObjectMap(object_id=12)
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        weak_patch = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=1,
            metadata={
                "anchor_class_name": "wall",
                "anchor_confidence": 0.9,
                "anchor_label_strength": "weak_overlap",
            },
        )
        blocked_patch = Patch3D(
            patch_id=2,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=2,
            metadata={
                "anchor_class_name": "",
                "anchor_label_strength": "none",
                "anchor_blocked_candidates": [
                    {
                        "anchor_id": 0,
                        "class_name": "sofa",
                        "reason": "contained_subproposal_without_child_anchor",
                    }
                ],
            },
        )
        strong_patch = Patch3D(
            patch_id=3,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=3,
            metadata={
                "anchor_class_name": "blanket",
                "anchor_confidence": 0.85,
                "anchor_view_quality": 0.8,
                "anchor_label_strength": "strong",
            },
        )

        for patch in [weak_patch, blocked_patch, strong_patch]:
            accumulate_anchor_semantic_vote(obj, patch)

        assert preferred_object_semantic_label(obj) == ""
        anchor_state = obj.debug["anchor_semantics"]
        assert anchor_state["label_weighted_score"] == {"blanket": 0.7225}
        assert anchor_state["semantic_state"] == "provisional"
        assert anchor_state["ignored_observation_count"] == 2
        assert anchor_state["ignored_observation_reasons"] == {
            "non_strong_anchor_label": 1,
            "missing_anchor_label": 1,
        }

    def test_weak_contextual_anchor_evidence_is_recorded_but_not_committed(self):
        from src.modules.semantic_memory import object_semantic_commit_state

        obj = ObjectMap(object_id=40)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        weak_patch = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=1,
            metadata={
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.99,
                "anchor_label_strength": "weak_overlap",
            },
        )
        strong_patch = Patch3D(
            patch_id=2,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=2,
            metadata={
                "anchor_class_name": "blanket",
                "anchor_confidence": 0.91,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
            },
        )

        accumulate_anchor_semantic_vote(obj, weak_patch)
        accumulate_anchor_semantic_vote(obj, strong_patch)

        anchor_state = obj.debug["anchor_semantics"]
        assert anchor_state["contextual_label_score_sum"] == {"sofa": 0.99}
        assert anchor_state["ignored_observation_reasons"] == {"non_strong_anchor_label": 1}
        assert anchor_state["label_weighted_score"] == {"blanket": pytest.approx(0.84175)}
        assert object_semantic_commit_state(obj)["semantic_state"] == "provisional"
        assert preferred_object_semantic_label(obj) == ""

    def test_semantic_commit_blocked_anchor_observation_is_delayed_not_exported(self):
        obj = ObjectMap(object_id=41)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        blocked_patch = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=1,
            metadata={
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.96,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "weak_overlap",
                "semantic_commit_allowed": False,
                "residual_semantic_policy": "unknown",
                "mask_anchor_relation": "contained_residual",
                "semantic_commit_blocked_reason": "contained_residual_identity_guard",
            },
        )

        accumulate_anchor_semantic_vote(obj, blocked_patch)
        ObjectUpdateModule({})._refresh_object_debug(obj, TSDFInstanceVolume(voxel_size=1.0))

        anchor_state = obj.debug["anchor_semantics"]
        assert anchor_state["evidence"] == []
        assert anchor_state["delayed_evidence"] == [
            {
                "label": "sofa",
                "confidence": pytest.approx(0.96),
                "frame_id": 1,
                "view_quality": pytest.approx(0.90),
                "anchor_label_strength": "weak_overlap",
                "semantic_commit_allowed": False,
                "residual_semantic_policy": "unknown",
                "mask_anchor_relation": "contained_residual",
                "reason": "contained_residual_identity_guard",
                "commit_eligible": False,
            }
        ]
        assert anchor_state["contextual_evidence"] == []
        assert anchor_state["contextual_label_score_sum"] == {}
        assert anchor_state["ignored_observation_reasons"] == {"semantic_commit_blocked": 1}
        assert object_semantic_commit_state(obj)["semantic_state"] == "unlabeled"
        assert object_export_semantic_label(obj) == ""

    def test_delayed_residual_can_later_export_from_direct_strong_anchor(self):
        from src.modules.semantic_memory import set_anchor_export_policy

        obj = ObjectMap(object_id=42)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )

        def make_patch(patch_id: int, blocked: bool) -> Patch3D:
            metadata = {
                "anchor_class_name": "blanket",
                "anchor_confidence": 0.94,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
            }
            if blocked:
                metadata.update(
                    {
                        "semantic_commit_allowed": False,
                        "residual_semantic_policy": "unknown",
                        "mask_anchor_relation": "contained_residual",
                    }
                )
            return Patch3D(
                patch_id=patch_id,
                points=points,
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=patch_id,
                metadata=metadata,
            )

        try:
            set_anchor_export_policy(
                {
                    "enabled": True,
                    "min_frame_hits": 1,
                    "min_high_quality_hits": 1,
                    "min_weighted_score": 0.50,
                    "min_score_margin": 0.0,
                    "min_confidence": 0.50,
                    "min_view_quality": 0.25,
                    "allow_single_frame_high_confidence": True,
                    "single_frame_min_confidence": 0.50,
                    "single_frame_min_view_quality": 0.25,
                }
            )

            accumulate_anchor_semantic_vote(obj, make_patch(1, blocked=True))
            assert object_export_semantic_label(obj) == ""

            accumulate_anchor_semantic_vote(obj, make_patch(2, blocked=False))

            anchor_state = obj.debug["anchor_semantics"]
            assert [item["frame_id"] for item in anchor_state["delayed_evidence"]] == [1]
            assert [item["frame_id"] for item in anchor_state["evidence"]] == [2]
            assert object_export_semantic_state(obj)["export_state"] == "safe_provisional"
            assert object_export_semantic_label(obj) == "blanket"
        finally:
            set_anchor_export_policy(None)

    def test_patch_lifting_preserves_anchor_label_strength_and_blocked_candidates(self):
        depth = np.ones((3, 3), dtype=np.float32)
        mask = np.ones((3, 3), dtype=bool)
        blocked_candidates = [
            {
                "anchor_id": 0,
                "class_name": "sofa",
                "reason": "contained_subproposal_without_child_anchor",
            }
        ]
        proposal = RefinedProposal2D(
            proposal_id=16,
            mask=mask,
            bbox_xyxy=np.array([0, 0, 3, 3], dtype=np.float32),
            area=int(mask.sum()),
            metadata={
                "anchor_class_name": "wall",
                "anchor_confidence": 0.8,
                "anchor_label_strength": "weak_overlap",
                "anchor_blocked_candidates": blocked_candidates,
                "semantic_commit_allowed": False,
                "residual_semantic_policy": "unknown",
                "mask_anchor_relation": "contained_residual",
            },
        )
        module = PatchLiftingModule({"min_points": 1})

        patches = module.process(
            [proposal],
            depth,
            np.eye(4, dtype=np.float64),
            CameraIntrinsics(fx=10.0, fy=10.0, cx=1.0, cy=1.0, width=3, height=3),
            frame_id=0,
        )

        assert len(patches) == 1
        assert patches[0].metadata["anchor_label_strength"] == "weak_overlap"
        assert patches[0].metadata["anchor_blocked_candidates"] == blocked_candidates
        assert patches[0].metadata["semantic_commit_allowed"] is False
        assert patches[0].metadata["residual_semantic_policy"] == "unknown"
        assert patches[0].metadata["mask_anchor_relation"] == "contained_residual"

    def test_patch_lifting_deterministically_samples_points_after_depth_filter(self):
        depth = np.ones((10, 10), dtype=np.float32)
        mask = np.ones((10, 10), dtype=bool)
        proposal = RefinedProposal2D(
            proposal_id=21,
            mask=mask,
            bbox_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
            area=int(mask.sum()),
        )
        module = PatchLiftingModule(
            {
                "min_points": 1,
                "point_sample_ratio": 0.25,
                "max_points_per_patch": 20,
            }
        )
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=5.0, cy=5.0, width=10, height=10)

        first = module.process([proposal], depth, np.eye(4, dtype=np.float64), intrinsics, frame_id=0)
        second = module.process([proposal], depth, np.eye(4, dtype=np.float64), intrinsics, frame_id=0)

        assert len(first) == 1
        assert len(second) == 1
        assert len(first[0].points) == 20
        np.testing.assert_allclose(first[0].points, second[0].points)
        sampling = first[0].metadata["point_sampling"]
        assert sampling["enabled"] is True
        assert sampling["input_point_count"] == 100
        assert sampling["sampled_point_count"] == 20
        assert sampling["point_sample_ratio"] == 0.25
        assert sampling["max_points_per_patch"] == 20

    def test_patch_lifting_samples_after_foreground_depth_filter(self):
        depth = np.full((6, 6), 1.0, dtype=np.float32)
        depth[0, 0:4] = 2.0
        mask = np.ones((6, 6), dtype=bool)
        proposal = RefinedProposal2D(
            proposal_id=22,
            mask=mask,
            bbox_xyxy=np.array([0, 0, 6, 6], dtype=np.float32),
            area=int(mask.sum()),
        )
        module = PatchLiftingModule(
            {
                "min_points": 1,
                "point_sample_ratio": 0.5,
                "max_points_per_patch": 100,
                "foreground_depth_filter": {
                    "enabled": True,
                    "front_quantile": 0.05,
                    "depth_band": 0.08,
                    "min_component_points": 1,
                    "min_depth_gap": 0.15,
                    "max_removed_ratio": 0.50,
                },
            }
        )

        patches = module.process(
            [proposal],
            depth,
            np.eye(4, dtype=np.float64),
            CameraIntrinsics(fx=10.0, fy=10.0, cx=3.0, cy=3.0, width=6, height=6),
            frame_id=0,
        )

        assert len(patches) == 1
        patch = patches[0]
        assert patch.metadata["foreground_depth_filter"]["kept_point_count"] == 32
        assert patch.metadata["point_sampling"]["input_point_count"] == 32
        assert patch.metadata["point_sampling"]["sampled_point_count"] == 16
        assert len(patch.points) == 16

    @pytest.mark.parametrize(
        ("point_sample_ratio", "max_points_per_patch", "min_points", "expected_points", "expected_enabled"),
        [
            (0.0, 0, 1, 25, False),
            (1.5, 0, 1, 25, False),
            (1.0, 3, 10, 10, True),
        ],
    )
    def test_patch_lifting_sampling_edge_configs(
        self,
        point_sample_ratio,
        max_points_per_patch,
        min_points,
        expected_points,
        expected_enabled,
    ):
        depth = np.ones((5, 5), dtype=np.float32)
        mask = np.ones((5, 5), dtype=bool)
        proposal = RefinedProposal2D(
            proposal_id=23,
            mask=mask,
            bbox_xyxy=np.array([0, 0, 5, 5], dtype=np.float32),
            area=int(mask.sum()),
        )
        module = PatchLiftingModule(
            {
                "min_points": min_points,
                "point_sample_ratio": point_sample_ratio,
                "max_points_per_patch": max_points_per_patch,
            }
        )

        patches = module.process(
            [proposal],
            depth,
            np.eye(4, dtype=np.float64),
            CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=5, height=5),
            frame_id=0,
        )

        assert len(patches) == 1
        sampling = patches[0].metadata["point_sampling"]
        assert len(patches[0].points) == expected_points
        assert sampling["enabled"] is expected_enabled
        assert sampling["input_point_count"] == 25
        assert sampling["sampled_point_count"] == expected_points

    def test_patch_lifting_foreground_depth_core_discards_background_tail(self):
        depth = np.full((6, 6), 1.0, dtype=np.float32)
        depth[0, 0:4] = 2.0
        mask = np.ones((6, 6), dtype=bool)
        proposal = RefinedProposal2D(
            proposal_id=12,
            mask=mask,
            bbox_xyxy=np.array([0, 0, 6, 6], dtype=np.float32),
            area=int(mask.sum()),
            metadata={"anchor_class_name": "lamp", "anchor_confidence": 0.9},
        )
        module = PatchLiftingModule(
            {
                "min_points": 1,
                "foreground_depth_filter": {
                    "enabled": True,
                    "front_quantile": 0.05,
                    "depth_band": 0.08,
                    "min_component_points": 1,
                    "min_depth_gap": 0.15,
                    "max_removed_ratio": 0.50,
                },
            }
        )

        patches = module.process(
            [proposal],
            depth,
            np.eye(4, dtype=np.float64),
            CameraIntrinsics(fx=10.0, fy=10.0, cx=3.0, cy=3.0, width=6, height=6),
            frame_id=0,
        )

        assert len(patches) == 1
        np.testing.assert_allclose(patches[0].points[:, 2], np.ones(32, dtype=np.float32), atol=1e-6)
        assert patches[0].metadata["foreground_depth_filter"]["enabled"] is True
        assert patches[0].metadata["foreground_depth_filter"]["removed_point_count"] == 4
        assert patches[0].metadata["foreground_depth_filter"]["removed_ratio"] == pytest.approx(4 / 36)
        assert patches[0].metadata["lifted_point_count"] == 32

    def test_patch_lifting_foreground_depth_core_falls_back_when_too_few_points_remain(self):
        depth = np.full((4, 4), 2.0, dtype=np.float32)
        depth[1, 1] = 1.0
        mask = np.ones((4, 4), dtype=bool)
        proposal = RefinedProposal2D(
            proposal_id=13,
            mask=mask,
            bbox_xyxy=np.array([0, 0, 4, 4], dtype=np.float32),
            area=int(mask.sum()),
        )
        module = PatchLiftingModule(
            {
                "min_points": 8,
                "foreground_depth_filter": {
                    "enabled": True,
                    "front_quantile": 0.05,
                    "depth_band": 0.05,
                    "min_component_points": 8,
                },
            }
        )

        patches = module.process(
            [proposal],
            depth,
            np.eye(4, dtype=np.float64),
            CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=4, height=4),
            frame_id=0,
        )

        assert len(patches) == 1
        assert len(patches[0].points) == 16
        assert patches[0].metadata["foreground_depth_filter"]["fallback_reason"] == "insufficient_foreground_points"

    def test_patch_lifting_foreground_depth_core_falls_back_when_removed_ratio_too_large(self):
        depth = np.full((4, 4), 2.0, dtype=np.float32)
        depth[1:3, 1:3] = 1.0
        mask = np.ones((4, 4), dtype=bool)
        proposal = RefinedProposal2D(
            proposal_id=15,
            mask=mask,
            bbox_xyxy=np.array([0, 0, 4, 4], dtype=np.float32),
            area=int(mask.sum()),
        )
        module = PatchLiftingModule(
            {
                "min_points": 1,
                "foreground_depth_filter": {
                    "enabled": True,
                    "front_quantile": 0.05,
                    "depth_band": 0.08,
                    "min_component_points": 1,
                    "min_depth_gap": 0.15,
                    "max_removed_ratio": 0.50,
                },
            }
        )

        patches = module.process(
            [proposal],
            depth,
            np.eye(4, dtype=np.float64),
            CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=4, height=4),
            frame_id=0,
        )

        assert len(patches) == 1
        assert len(patches[0].points) == 16
        fg_debug = patches[0].metadata["foreground_depth_filter"]
        assert fg_debug["removed_point_count"] == 0
        assert fg_debug["removed_ratio"] == pytest.approx(12 / 16)
        assert fg_debug["fallback_reason"] == "removed_ratio_too_large"

    def test_patch_lifting_foreground_depth_core_keeps_continuous_depth_ramp(self):
        depth = np.linspace(1.0, 1.5, num=36, dtype=np.float32).reshape(6, 6)
        mask = np.ones((6, 6), dtype=bool)
        proposal = RefinedProposal2D(
            proposal_id=14,
            mask=mask,
            bbox_xyxy=np.array([0, 0, 6, 6], dtype=np.float32),
            area=int(mask.sum()),
        )
        module = PatchLiftingModule(
            {
                "min_points": 1,
                "foreground_depth_filter": {
                    "enabled": True,
                    "front_quantile": 0.05,
                    "depth_band": 0.08,
                    "min_component_points": 1,
                    "min_depth_gap": 0.15,
                    "max_removed_ratio": 0.50,
                },
            }
        )

        patches = module.process(
            [proposal],
            depth,
            np.eye(4, dtype=np.float64),
            CameraIntrinsics(fx=10.0, fy=10.0, cx=3.0, cy=3.0, width=6, height=6),
            frame_id=0,
        )

        assert len(patches) == 1
        assert len(patches[0].points) == 36
        fg_debug = patches[0].metadata["foreground_depth_filter"]
        assert fg_debug["removed_point_count"] == 0
        assert fg_debug["fallback_reason"] == "no_depth_gap"
