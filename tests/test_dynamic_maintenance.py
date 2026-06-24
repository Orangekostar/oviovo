"""Tests for TSDF-driven dynamic maintenance."""

import numpy as np
import pytest
from src.core.data_structures import (
    CandidateEvidenceEntry,
    CandidateEvidenceGrid,
    ObjectMap,
    ObjectState,
    TSDFInstanceVolume,
    VoxelOwnerSupport,
)
from src.modules.dynamic_maintenance import DynamicMaintenanceModule


def _make_volume_with_owner(
    voxel_size: float, owned_positions: list, owner_id: int
) -> TSDFInstanceVolume:
    """Helper: create a TSDF volume with voxels owned by owner_id."""
    volume = TSDFInstanceVolume(voxel_size=voxel_size)
    for pos in owned_positions:
        vk = tuple(np.floor(np.array(pos) / voxel_size).astype(np.int64).tolist())
        if vk not in volume.owner_support:
            volume.owner_support[vk] = VoxelOwnerSupport()
        volume.owner_support[vk].support[owner_id] = 3.0
    return volume


class TestSpatialFootprint:
    def test_all_points_owned(self):
        """When all object points land on owned voxels, current_voxels > 0."""
        obj = ObjectMap(object_id=1, state=ObjectState.ACTIVE)
        obj.local_pcd = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)

        volume = _make_volume_with_owner(0.05, [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], owner_id=1)

        module = DynamicMaintenanceModule({})
        result = module._compute_spatial_footprint(obj, volume)

        assert result["current_voxels"] >= 1
        assert result["total_queried"] >= 1

    def test_no_points_owned(self):
        """When object points land on unowned voxels, current_voxels == 0."""
        obj = ObjectMap(object_id=1, state=ObjectState.ACTIVE)
        obj.local_pcd = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)

        volume = TSDFInstanceVolume(voxel_size=0.05)

        module = DynamicMaintenanceModule({})
        result = module._compute_spatial_footprint(obj, volume)

        assert result["current_voxels"] == 0

    def test_empty_local_pcd(self):
        """Empty local_pcd returns zero current_voxels."""
        obj = ObjectMap(object_id=1, state=ObjectState.ACTIVE)
        obj.local_pcd = np.empty((0, 3), dtype=np.float32)

        volume = TSDFInstanceVolume(voxel_size=0.05)

        module = DynamicMaintenanceModule({})
        result = module._compute_spatial_footprint(obj, volume)

        assert result["current_voxels"] == 0
        assert result["total_queried"] == 0

    def test_foreign_owner_not_counted(self):
        """Voxels owned by a different object_id are not counted."""
        obj = ObjectMap(object_id=1, state=ObjectState.ACTIVE)
        obj.local_pcd = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        # voxel owned by object 2, not object 1
        volume = _make_volume_with_owner(0.05, [[0.0, 0.0, 0.0]], owner_id=2)

        module = DynamicMaintenanceModule({})
        result = module._compute_spatial_footprint(obj, volume)

        assert result["current_voxels"] == 0


class TestCheckDisappeared:
    def test_high_ownership_ratio_not_disappeared(self):
        """ownership_ratio close to 1.0 -> NOT disappeared."""
        obj = ObjectMap(object_id=1, state=ObjectState.ACTIVE)
        obj.peak_voxel_count = 10
        # 10 distinct points mapping to 10 distinct voxels
        obj.local_pcd = np.array(
            [[i * 0.05, 0.0, 0.0] for i in range(10)], dtype=np.float32
        )

        volume = TSDFInstanceVolume(voxel_size=0.05)
        for i in range(10):
            vk = (i, 0, 0)
            volume.owner_support[vk] = VoxelOwnerSupport()
            volume.owner_support[vk].support[1] = 3.0

        module = DynamicMaintenanceModule({"disappear_threshold": 0.2, "creation_grace_frames": 5})
        # Give the object some history
        import src.core.data_structures as d
        from dataclasses import replace
        state = d.SystemState()
        state.frame_count = 10

        result = module._check_disappeared(obj, volume, state, module.config)
        assert result is False

    def test_low_ownership_ratio_disappeared(self):
        """ownership_ratio below threshold -> disappeared (when no matching proposals)."""
        obj = ObjectMap(object_id=1, state=ObjectState.ACTIVE)
        obj.peak_voxel_count = 100
        obj.local_pcd = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        volume = TSDFInstanceVolume(voxel_size=0.05)

        module = DynamicMaintenanceModule({"disappear_threshold": 0.2, "creation_grace_frames": 5})
        import src.core.data_structures as d
        state = d.SystemState()
        state.frame_count = 20

        result = module._check_disappeared(obj, volume, state, module.config)
        assert result is True

    def test_zero_peak_voxel_count_not_disappeared(self):
        """New object with peak_voxel_count=0 -> not disappeared."""
        obj = ObjectMap(object_id=1, state=ObjectState.ACTIVE)
        obj.peak_voxel_count = 0
        obj.local_pcd = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        volume = TSDFInstanceVolume(voxel_size=0.05)

        module = DynamicMaintenanceModule({"disappear_threshold": 0.2, "creation_grace_frames": 5})
        import src.core.data_structures as d
        state = d.SystemState()
        state.frame_count = 10

        result = module._check_disappeared(obj, volume, state, module.config)
        assert result is False

    def test_grace_period_skips_judgment(self):
        """Object younger than creation_grace_frames -> always False."""
        obj = ObjectMap(object_id=1, state=ObjectState.ACTIVE)
        obj.peak_voxel_count = 100
        obj.creation_frame = 8  # frame_count=10, diff=2 < grace=5
        obj.local_pcd = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        volume = TSDFInstanceVolume(voxel_size=0.05)

        module = DynamicMaintenanceModule({"disappear_threshold": 0.2, "creation_grace_frames": 5})
        import src.core.data_structures as d
        state = d.SystemState()
        state.frame_count = 10

        result = module._check_disappeared(obj, volume, state, module.config)
        assert result is False


class TestCheckMoved:
    def test_small_centroid_shift_not_moved(self):
        """Centroid shift below threshold -> not moved."""
        obj = ObjectMap(object_id=1, state=ObjectState.ACTIVE)
        obj.creation_centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obj.centroid = np.array([0.1, 0.0, 0.0], dtype=np.float32)  # 0.1m shift
        obj.local_pcd = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        volume = TSDFInstanceVolume(voxel_size=0.05)

        module = DynamicMaintenanceModule({
            "moved_centroid_threshold": 0.3,
            "moved_search_steps": 10,
            "moved_ownership_skew": 0.6,
        })
        result = module._check_moved(obj, volume, module.config)
        assert result is False

    def test_large_shift_but_no_ownership_migration_not_moved(self):
        """Large centroid shift but no ownership at new location -> not moved."""
        obj = ObjectMap(object_id=1, state=ObjectState.ACTIVE)
        obj.creation_centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obj.centroid = np.array([2.0, 0.0, 0.0], dtype=np.float32)  # 2m shift
        obj.local_pcd = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        volume = TSDFInstanceVolume(voxel_size=0.05)

        module = DynamicMaintenanceModule({
            "moved_centroid_threshold": 0.3,
            "moved_search_steps": 10,
            "moved_ownership_skew": 0.6,
        })
        result = module._check_moved(obj, volume, module.config)
        assert result is False

    def test_ownership_migrated_to_new_half_confirms_moved(self):
        """Ownership predominantly in new half -> moved confirmed."""
        obj = ObjectMap(object_id=1, state=ObjectState.ACTIVE)
        obj.creation_centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obj.centroid = np.array([2.0, 0.0, 0.0], dtype=np.float32)  # 2m shift
        obj.local_pcd = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        volume = TSDFInstanceVolume(voxel_size=0.05)

        # Place owner voxels along the new half of the shift path
        # Direction: (2,0,0), sampling from old to new.
        # Steps 0-4 = old half, 5-9 = new half (approx).
        # Place ownership in new half:
        for i in range(6, 10):
            t = i / 9.0
            pos = np.array([2.0 * t, 0.0, 0.0])
            vk = tuple(np.floor(pos / 0.05).astype(np.int64).tolist())
            volume.owner_support[vk] = VoxelOwnerSupport()
            volume.owner_support[vk].support[1] = 3.0

        module = DynamicMaintenanceModule({
            "moved_centroid_threshold": 0.3,
            "moved_search_steps": 10,
            "moved_ownership_skew": 0.6,
        })
        result = module._check_moved(obj, volume, module.config)
        assert result is True


class TestCandidateEvidenceGrid:
    def test_write_and_cluster(self):
        """Entries written to grid can be clustered spatially."""
        grid = CandidateEvidenceGrid(voxel_size=0.05)
        # Two entries close together
        vk1 = (0, 0, 0)
        vk2 = (1, 0, 0)  # adjacent at 0.05m voxel size
        grid.entries[vk1] = CandidateEvidenceEntry(
            label_votes={"chair": 2.0}, hit_count=2, last_seen_frame=10
        )
        grid.entries[vk2] = CandidateEvidenceEntry(
            label_votes={"chair": 1.5}, hit_count=1, last_seen_frame=10
        )

        module = DynamicMaintenanceModule({})
        clusters = module._cluster_candidate_entries(
            grid.entries, voxel_size=0.05, cluster_radius=0.3
        )
        assert len(clusters) == 1
        # The cluster should contain both entries
        assert len(clusters[0]) == 2

    def test_distant_entries_separate_clusters(self):
        """Entries far apart form separate clusters."""
        grid = CandidateEvidenceGrid(voxel_size=0.05)
        vk1 = (0, 0, 0)
        vk2 = (100, 0, 0)  # 5m away at 0.05m voxel size
        grid.entries[vk1] = CandidateEvidenceEntry(
            label_votes={"chair": 2.0}, hit_count=2, last_seen_frame=10
        )
        grid.entries[vk2] = CandidateEvidenceEntry(
            label_votes={"table": 2.0}, hit_count=2, last_seen_frame=10
        )

        module = DynamicMaintenanceModule({})
        clusters = module._cluster_candidate_entries(
            grid.entries, voxel_size=0.05, cluster_radius=0.3
        )
        assert len(clusters) == 2

    def test_prune_expired_entries(self):
        """Entries beyond max_idle_frames are removed."""
        import src.core.data_structures as d
        state = d.SystemState()
        vk = (0, 0, 0)
        state.candidate_evidence.entries[vk] = CandidateEvidenceEntry(
            label_votes={"chair": 1.0}, hit_count=1, last_seen_frame=5
        )
        state.frame_count = 100

        module = DynamicMaintenanceModule({"candidate_max_idle_frames": 60})
        module._prune_expired_candidates(state, module.config)

        # Entry from frame 5, current frame 100, idle 95 > 60
        assert vk not in state.candidate_evidence.entries

    def test_recent_entry_not_pruned(self):
        """Recently updated entry is not pruned."""
        import src.core.data_structures as d
        state = d.SystemState()
        vk = (0, 0, 0)
        state.candidate_evidence.entries[vk] = CandidateEvidenceEntry(
            label_votes={"chair": 1.0}, hit_count=1, last_seen_frame=90
        )
        state.frame_count = 100

        module = DynamicMaintenanceModule({"candidate_max_idle_frames": 60})
        module._prune_expired_candidates(state, module.config)

        # Entry from frame 90, current 100, idle 10 < 60
        assert vk in state.candidate_evidence.entries

    def test_promotion_candidate_detected(self):
        """When cluster total score >= threshold, promotion is triggered."""
        import src.core.data_structures as d
        state = d.SystemState()
        grid = state.candidate_evidence
        # Create a tight cluster with high cumulative confidence
        for i in range(3):
            vk = (i, 0, 0)
            grid.entries[vk] = CandidateEvidenceEntry(
                label_votes={"chair": 1.5}, hit_count=2, last_seen_frame=10
            )
        state.frame_count = 10

        volume = TSDFInstanceVolume(voxel_size=0.05)
        # All voxels are background (no owner)
        # We need to verify background: owner_support is empty for these keys

        module = DynamicMaintenanceModule({
            "candidate_cluster_radius": 0.3,
            "candidate_promotion_threshold": 3.0,
        })
        results = module._cluster_and_check_new_objects(state, volume, module.config)
        # Cluster total = 3 entries * 1.5 = 4.5 >= 3.0
        assert len(results) >= 1
        assert results[0]["label"] == "chair"


class TestProcessIntegration:
    """Integration tests for DynamicMaintenanceModule.process()."""

    def test_static_object_stays_active(self):
        """Object with stable TSDF ownership stays ACTIVE."""
        import src.core.data_structures as d

        state = d.SystemState()
        state.frame_count = 50

        obj = ObjectMap(
            object_id=1,
            state=ObjectState.ACTIVE,
            creation_frame=10,
        )
        # Points spread across 20 distinct voxels
        obj.local_pcd = np.array(
            [[i * 0.05, 0.0, 0.0] for i in range(20)], dtype=np.float32
        )
        obj.centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obj.creation_centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obj.peak_voxel_count = 20
        state.objects[1] = obj

        # Populate TSDF with ownership at matching voxels
        for i in range(20):
            vk = (i, 0, 0)
            state.tsdf_volume.owner_support[vk] = VoxelOwnerSupport()
            state.tsdf_volume.owner_support[vk].support[1] = 3.0

        config = {
            "lifecycle_check_interval": 1,
            "disappear_threshold": 0.2,
            "creation_grace_frames": 5,
            "moved_centroid_threshold": 0.3,
            "moved_search_steps": 10,
            "moved_ownership_skew": 0.6,
            "candidate_cluster_radius": 0.3,
            "candidate_promotion_threshold": 3.0,
            "candidate_max_idle_frames": 60,
        }
        module = DynamicMaintenanceModule(config)
        result = module.process(state)

        assert result.objects[1].state == ObjectState.ACTIVE

    def test_disappeared_object_becomes_ghost(self):
        """Object with zero current ownership becomes GHOST."""
        import src.core.data_structures as d

        state = d.SystemState()
        state.frame_count = 50

        obj = ObjectMap(
            object_id=1,
            state=ObjectState.ACTIVE,
            creation_frame=10,
        )
        obj.local_pcd = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        obj.centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obj.creation_centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obj.peak_voxel_count = 100  # was large, now zero
        state.objects[1] = obj
        # TSDF: no ownership for obj 1

        config = {
            "lifecycle_check_interval": 1,
            "disappear_threshold": 0.2,
            "creation_grace_frames": 5,
            "moved_centroid_threshold": 0.3,
            "moved_search_steps": 10,
            "moved_ownership_skew": 0.6,
            "candidate_cluster_radius": 0.3,
            "candidate_promotion_threshold": 3.0,
            "candidate_max_idle_frames": 60,
        }
        module = DynamicMaintenanceModule(config)
        result = module.process(state)

        assert result.objects[1].state == ObjectState.GHOST

    def test_check_interval_respected(self):
        """Maintenance only runs at check_interval frequency."""
        import src.core.data_structures as d

        state = d.SystemState()
        state.frame_count = 3  # Not divisible by check_interval=10

        obj = ObjectMap(
            object_id=1,
            state=ObjectState.ACTIVE,
            creation_frame=0,
        )
        obj.local_pcd = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        obj.centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obj.creation_centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obj.peak_voxel_count = 100
        state.objects[1] = obj

        config = {
            "lifecycle_check_interval": 10,
            "disappear_threshold": 0.2,
            "creation_grace_frames": 5,
            "moved_centroid_threshold": 0.3,
        }
        module = DynamicMaintenanceModule(config)
        result = module.process(state)

        # Object should be unchanged (maintenance skipped)
        assert result.objects[1].state == ObjectState.ACTIVE

    def test_ghost_cleanup(self):
        """Ghost objects have TSDF support removed."""
        import src.core.data_structures as d

        state = d.SystemState()
        state.frame_count = 20

        obj = ObjectMap(
            object_id=1,
            state=ObjectState.GHOST,
            creation_frame=0,
        )
        obj.local_pcd = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        obj.centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obj.creation_centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obj.peak_voxel_count = 10
        state.objects[1] = obj

        # Add some TSDF ownership that should be cleaned
        vk = (0, 0, 0)
        state.tsdf_volume.owner_support[vk] = VoxelOwnerSupport()
        state.tsdf_volume.owner_support[vk].support[1] = 5.0

        config = {
            "lifecycle_check_interval": 1,
            "disappear_threshold": 0.2,
            "creation_grace_frames": 5,
            "moved_centroid_threshold": 0.3,
        }
        module = DynamicMaintenanceModule(config)
        result = module.process(state)

        # Object removed
        assert 1 not in result.objects
        # TSDF ownership cleaned
        support = state.tsdf_volume.owner_support.get(vk)
        if support is not None:
            assert support.owner_id != 1
