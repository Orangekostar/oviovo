"""Integration tests: simulate multi-frame RGB-D scenario for TSDF-driven maintenance."""

import numpy as np
from src.core.data_structures import (
    ObjectMap,
    ObjectState,
    SystemState,
    TSDFInstanceVolume,
    VoxelOwnerSupport,
)
from src.modules.dynamic_maintenance import DynamicMaintenanceModule


def _create_static_object(obj_id: int, voxel_count: int = 20) -> ObjectMap:
    """Create a stable object with TSDF ownership at origin."""
    obj = ObjectMap(
        object_id=obj_id,
        state=ObjectState.ACTIVE,
        creation_frame=5,
    )
    obj.local_pcd = np.array(
        [[i * 0.05, 0.0, 0.0] for i in range(voxel_count)], dtype=np.float32
    )
    obj.centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    obj.creation_centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    obj.peak_voxel_count = voxel_count
    return obj


def _populate_tsdf_ownership(volume: TSDFInstanceVolume, obj_id: int, count: int):
    """Add voxels owned by obj_id to the TSDF volume."""
    for i in range(count):
        vk = (i, 0, 0)
        volume.owner_support[vk] = VoxelOwnerSupport()
        volume.owner_support[vk].support[obj_id] = 3.0


class TestStaticSceneNoFalseGhost:
    def test_200_frames_static_no_ghost(self):
        """200-frame static scene should produce 0 GHOST transitions."""
        config = {
            "lifecycle_check_interval": 10,
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

        state = SystemState()
        obj = _create_static_object(1, voxel_count=20)
        state.objects[1] = obj
        _populate_tsdf_ownership(state.tsdf_volume, 1, 20)

        ghost_count = 0
        for frame in range(5, 205):
            state.frame_count = frame
            state = module.process(state)

            if 1 in state.objects:
                current_state = state.objects[1].state
                if current_state == ObjectState.GHOST:
                    ghost_count += 1
            else:
                ghost_count += 1  # object removed = also a fail

            if ghost_count > 0:
                break

        assert ghost_count == 0, (
            f"Static object incorrectly ghosted at frame {state.frame_count}"
        )

    def test_multiple_static_objects_no_false_ghost(self):
        """Multiple static objects in different locations — no GHOST."""
        config = {
            "lifecycle_check_interval": 10,
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

        state = SystemState()
        # Object 1 at origin, Object 2 at (2, 0, 0)
        for obj_id, offset in [(1, 0), (2, 50)]:  # offset in voxel index
            obj = ObjectMap(
                object_id=obj_id,
                state=ObjectState.ACTIVE,
                creation_frame=5,
            )
            obj.local_pcd = np.array(
                [[(offset + i) * 0.05, 0.0, 0.0] for i in range(15)], dtype=np.float32
            )
            obj.centroid = np.array([offset * 0.05, 0.0, 0.0], dtype=np.float32)
            obj.creation_centroid = np.array([offset * 0.05, 0.0, 0.0], dtype=np.float32)
            obj.peak_voxel_count = 15
            state.objects[obj_id] = obj
            for i in range(15):
                vk = (offset + i, 0, 0)
                state.tsdf_volume.owner_support[vk] = VoxelOwnerSupport()
                state.tsdf_volume.owner_support[vk].support[obj_id] = 3.0

        ghost_count = 0
        for frame in range(5, 205):
            state.frame_count = frame
            state = module.process(state)

            for obj_id in [1, 2]:
                if obj_id in state.objects:
                    if state.objects[obj_id].state == ObjectState.GHOST:
                        ghost_count += 1
                else:
                    ghost_count += 1

            if ghost_count > 0:
                break

        assert ghost_count == 0, (
            f"Static objects incorrectly ghosted at frame {state.frame_count}"
        )
