import numpy as np
import pytest

from src.static_ovmap.native_export import remap_partition, compact_export, restore_partition, partition_change_ledger


def test_wide_ids_remap_without_overflow_and_restore_original_partition():
    original = np.array([0, 70000, 70001, 9, 70000], dtype=np.int64)
    changed, ledger = remap_partition(original, [[70000, 70001], [9]])
    np.testing.assert_array_equal(changed, [0, 70000, 70000, 9, 70000])
    encoded, mapping = compact_export(changed, np.uint8)
    np.testing.assert_array_equal(encoded, [0, 2, 2, 1, 2])
    assert mapping == {0: 0, 1: 9, 2: 70000}
    np.testing.assert_array_equal(restore_partition(changed, ledger), original)


def test_incomplete_components_and_insufficient_output_range_are_rejected():
    with pytest.raises(ValueError):
        remap_partition(np.array([1, 2]), [[1]])
    with pytest.raises(ValueError):
        compact_export(np.arange(257), np.uint8)
    with pytest.raises(ValueError):
        remap_partition(np.array([1, 2]), [[1, 2], [2]])


def test_split_ledger_restores_native_partition_and_rejects_background_reassignment():
    original = np.array([0, 1, 1, 1, 2])
    changed = np.array([0, 70000, 70001, 1, 2])
    ledger = partition_change_ledger(original, changed)
    np.testing.assert_array_equal(restore_partition(changed, ledger), original)
    with pytest.raises(ValueError):
        partition_change_ledger(original, np.array([3, 70000, 70001, 1, 2]))
    with pytest.raises(ValueError):
        restore_partition(np.array([0, 70001, 70000, 1, 2]), ledger)


def test_native_mesh_export_preserves_coordinates_faces_and_wide_owner_identity(tmp_path):
    from plyfile import PlyData
    from src.static_ovmap.native_export import write_partition_mesh
    points = np.array([[.1, .2, .3], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    labels = np.array([70000, 70001, 70000])
    path = tmp_path/'native.ply'
    receipt = write_partition_mesh(points, faces, labels, path)
    mesh = PlyData.read(path)
    np.testing.assert_array_equal(np.column_stack([mesh['vertex'][c] for c in ('x', 'y', 'z')]), points)
    np.testing.assert_array_equal(np.stack(mesh['face']['vertex_indices']), faces)
    np.testing.assert_array_equal(mesh['vertex']['instance_id'], labels)
    assert receipt['rgb_to_native_id'] == {'1,0,0': 70000, '2,0,0': 70001}
