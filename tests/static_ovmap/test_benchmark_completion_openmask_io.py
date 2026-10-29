import os
from pathlib import Path

import numpy as np
import pytest

from test_benchmark_completion_external_io import replica_rgbd


def binding_for_openmask(tmp_path):
    from static_ovmap.benchmark_completion_v2.common import write
    checkout = os.environ.get('OVMAP_TEST_OPENMASK3D_CHECKOUT')
    if not checkout:
        pytest.skip('Set OVMAP_TEST_OPENMASK3D_CHECKOUT to the pinned checkout')
    binding, _ = replica_rgbd(tmp_path)
    binding['asset_config'] = dict(external=dict(OpenMask3D=dict(checkout=checkout)))
    names = [f'class{i}' for i in range(1, 49)]
    write(Path(binding['output_root']) / 'protocol_lock.json', dict(scopes=dict(replica48=dict(
        scenes=['room0'], text_ids=list(range(1, 49)), text_names=names,
        instance_ids=list(range(1, 49)), instance_names=names))))
    return binding, names


def test_original_openmask_camera_keeps_depth_scale_pose_and_frame_count(tmp_path):
    from static_ovmap.benchmark_completion_v2.openmask_io import camera
    binding, _ = binding_for_openmask(tmp_path)
    value, frames = camera(binding, 'room0')
    assert frames == [0, 10, 20]
    assert value.depth_scale == 6553.5
    assert np.array_equal(value.get_adapted_intrinsic((3, 4)), [[2, 0, 1.5], [0, 2, 1], [0, 0, 1]])
    assert value.load_poses([10])[0, 0, 3] == -.12345678901234567
    assert np.array_equal(value.load_depth(10, value.depth_scale), np.full((3, 4), 10))


def test_original_openmask_query_keeps_overlaps_zero_features_and_native_confidence(tmp_path):
    import torch
    from static_ovmap.benchmark_completion_v2.openmask_io import query
    binding, names = binding_for_openmask(tmp_path)
    masks = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0]], bool)
    masks_path, feature_path, text_path = (tmp_path / name for name in ('masks.pt', 'features.npy', 'text.npy'))
    # The original proposal writer stores a NumPy float16 array via torch.save.
    torch.save(masks.astype(np.float16), masks_path)
    features = np.zeros((3, 768), np.float32)
    features[0, 0], features[1, 47] = .4, .3
    np.save(feature_path, features)
    np.save(text_path, np.eye(48, 768, dtype=np.float32))
    actual = query(binding, masks_path=masks_path, features_path=feature_path, text_path=text_path)
    assert np.array_equal(actual['pred_masks'], masks)
    assert actual['pred_classes'].tolist() == [1, 48, 1]
    assert actual['pred_scores'].tolist() == [1, 1, 1]
    assert names[47] == 'class48'


def test_openmask_requires_real_rgb_instead_of_coordinate_only_sidecar(tmp_path):
    from plyfile import PlyData, PlyElement
    from static_ovmap.benchmark_completion_v2.common import StageFailure, digest_file
    from static_ovmap.benchmark_completion_v2.openmask_io import geometry
    binding, _ = binding_for_openmask(tmp_path)
    folder = tmp_path / 'geometry'
    folder.mkdir()
    binding['asset_config']['datasets'] = dict(replica48=dict(evaluation_mesh_root=str(folder)))
    path = folder / 'room0.ply'
    xyz = [(2, 0, 1), (-1, 3, 4), (0, -2, 3)]
    points = np.array(xyz, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    PlyData([PlyElement.describe(points, 'vertex')], text=False).write(str(path))
    with pytest.raises(StageFailure, match='color'):
        geometry(binding, 'room0')
    vertices = np.empty(3, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, name in enumerate(('x', 'y', 'z')):
        vertices[name] = np.asarray(xyz)[:, i]
    vertices['red'], vertices['green'], vertices['blue'] = [255, 0, 128], [0, 255, 64], [32, 16, 255]
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(str(path))
    before = digest_file(path)
    record = geometry(binding, 'room0')
    assert record['point_count'] == 3
    assert record['geometry']['path'] == str(path)
    assert record['geometry']['sha256'] == before == digest_file(path)
    assert record['color_values_synthesized'] is False
    assert record['GT_labels_read'] is False
