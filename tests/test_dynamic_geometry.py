import numpy as np

from src.core.data_structures import (
    DenseSurfaceEntry,
    ObjectMap,
    ObjectState,
    ObservationRecord,
    Patch3D,
    SystemState,
    VoxelOwnerSupport,
)
from src.modules.dynamic_geometry import DynamicGeometryModule, is_exportable_object


def _patch(points):
    points = np.asarray(points, dtype=np.float32)
    return Patch3D(
        patch_id=7,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
    )


def test_purge_object_geometry_clears_all_export_sources():
    state = SystemState()
    patch = _patch([[0.01, 0.0, 0.0], [0.06, 0.0, 0.0]])
    obj = ObjectMap(
        object_id=3,
        state=ObjectState.ACTIVE,
        local_pcd=patch.points.copy(),
        association_pcd=patch.points.copy(),
        observations=[ObservationRecord(frame_id=10, patch=patch)],
        centroid=patch.centroid.copy(),
        bbox_min=patch.bbox_min.copy(),
        bbox_max=patch.bbox_max.copy(),
    )
    state.objects[3] = obj
    state.dense_surface_map.entries[3] = DenseSurfaceEntry(
        points=patch.points.copy(),
        object_id=3,
        semantic_label="box",
        resident=True,
    )
    for key in [(0, 0, 0), (1, 0, 0)]:
        state.tsdf_volume.owner_support[key] = VoxelOwnerSupport()
        state.tsdf_volume.owner_support[key].support[3] = 2.0

    DynamicGeometryModule({}).purge_object_geometry(
        state,
        obj,
        frame_id=42,
        reason="disappeared",
    )

    assert obj.state == ObjectState.GHOST
    assert obj.removed_frame == 42
    assert obj.purged_frame == 42
    assert obj.dynamic_state_reason == "disappeared"
    assert obj.local_pcd.shape == (0, 3)
    assert obj.association_pcd.shape == (0, 3)
    assert obj.observations == []
    assert state.dense_surface_map.entries[3].resident is False
    assert len(state.dense_surface_map.entries[3].points) == 0
    assert all(3 not in support.support for support in state.tsdf_volume.owner_support.values())
    assert is_exportable_object(obj) is False


def test_active_object_with_points_is_exportable():
    obj = ObjectMap(
        object_id=4,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    assert is_exportable_object(obj) is True


from run_room0_full_eval import build_pool_semantic_records, project_instances_to_dense_points


def test_pool_export_skips_ghost_objects():
    state = SystemState()
    state.objects[1] = ObjectMap(
        object_id=1,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    )
    state.objects[2] = ObjectMap(
        object_id=2,
        state=ObjectState.GHOST,
        local_pcd=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    records = build_pool_semantic_records(state)

    assert set(records["object_id"].tolist()) == {1}


def test_projection_requires_voxel_majority():
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("object_id", "<i4"),
            ("state_id", "u1"),
            ("support", "<f4"),
        ]
    )
    records = np.zeros(3, dtype=dtype)
    records["x"] = [0.01, 0.02, 0.03]
    records["object_id"] = [1, 2, 2]
    records["state_id"] = [1, 1, 1]
    records["support"] = [0.5, 0.5, 0.5]
    dense = np.array([[0.02, 0.0, 0.0]], dtype=np.float32)

    labels, _, _ = project_instances_to_dense_points(
        dense,
        records,
        instance_voxel_size=0.1,
        neighbor_radius=0,
        min_voxel_votes=2,
        min_vote_ratio=0.6,
    )

    assert labels.tolist() == [2]
