import json
from pathlib import Path

import numpy as np
import pytest


def replica_rgbd(tmp_path):
    import cv2
    from static_ovmap.benchmark_completion_v2.common import write
    raw = tmp_path / 'raw'
    images = raw / 'room0/results'
    images.mkdir(parents=True)
    camera = dict(w=4, h=3, fx=2.0, fy=2.0, cx=1.5, cy=1.0, scale=6553.5)
    (raw / 'cam_params.json').write_text(json.dumps(dict(camera=camera)))
    poses = np.tile(np.eye(4), (21, 1, 1))
    poses[10, 0, 3] = 0.12345678901234567
    np.savetxt(raw / 'room0/traj.txt', poses.reshape(21, 16), fmt='%.17g')
    for frame in range(21):
        cv2.imwrite(str(images / f'frame{frame:06d}.jpg'), np.full((3, 4, 3), 42, np.uint8))
        cv2.imwrite(str(images / f'depth{frame:06d}.png'), np.full((3, 4), 65535, np.uint16))
    out = tmp_path / 'out'
    write(out / 'protocol_lock.json', dict(scopes=dict(replica48=dict(scenes=['room0']))))
    return dict(output_root=str(out), asset_config={}, inherited=dict(data_roots=[str(raw)]), path_map={}), raw


def test_replica_layout_preserves_all_poses_camera_and_original_image_bytes(tmp_path):
    from static_ovmap.benchmark_completion_v2.replica_layout import prepare
    binding, raw = replica_rgbd(tmp_path)
    result = prepare(binding, 'room0')
    layout = Path(result['layout_root'])
    assert len(list((layout / 'pose').glob('*.txt'))) == 21
    assert result['selected_frame_ids'] == [0, 10, 20]
    assert result['camera']['scale'] == 6553.5
    assert result['camera']['w'] == 4 and result['camera']['h'] == 3
    assert np.loadtxt(layout / 'pose/10.txt')[0, 3] == 0.12345678901234567
    assert np.array_equal(np.loadtxt(layout / 'intrinsic/intrinsic_depth.txt'),
                          [[2, 0, 1.5, 0], [0, 2, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert (layout / 'color/10.jpg').resolve() == raw / 'room0/results/frame000010.jpg'
    assert (layout / 'depth/10.png').resolve() == raw / 'room0/results/depth000010.png'
    assert result['evaluation_point_domain_ready'] is False
    receipt = layout / 'layout.json'
    before = receipt.stat().st_mtime_ns
    assert prepare(binding, 'room0')['identity'] == result['identity']
    assert receipt.stat().st_mtime_ns == before


def test_replica_layout_rejects_a_missing_original_frame_before_staging(tmp_path):
    from static_ovmap.benchmark_completion_v2.replica_layout import prepare
    from static_ovmap.benchmark_completion_v2.common import StageFailure
    binding, raw = replica_rgbd(tmp_path)
    (raw / 'room0/results/depth000011.png').unlink()
    with pytest.raises(StageFailure, match='frame'):
        prepare(binding, 'room0')
    assert not (Path(binding['output_root']) / 'external/replica_rgbd/room0/pose').exists()


def test_maskclustering_native_npz_retains_overlap_empty_slots_and_confidence(tmp_path):
    from static_ovmap.benchmark_completion_v2.baseline_io import decode_maskclustering
    masks = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], bool)
    source = tmp_path / 'room0.npz'
    np.savez(source, pred_masks=masks, pred_score=[0.25, 0.875, 1.0],
             pred_classes=np.array([1, 48, 0], np.int32))
    actual = decode_maskclustering(source, target_points=4, class_ids=list(range(1, 49)))
    assert np.array_equal(actual['pred_masks'], masks)
    assert actual['pred_classes'].tolist() == [1, 48, 0]
    assert actual['pred_scores'].tolist() == [0.25, 0.875, 1.0]
    with pytest.raises(ValueError, match='point'):
        decode_maskclustering(source, target_points=5, class_ids=list(range(1, 49)))
    np.savez(source, pred_masks=masks, pred_score=[1, 1, 1], pred_classes=[0, 48, 0])
    with pytest.raises(ValueError, match='class'):
        decode_maskclustering(source, target_points=4, class_ids=list(range(1, 49)))


def test_maskclustering_replica_reader_uses_original_depth_pose_and_point_order(tmp_path):
    import os
    checkout = os.environ.get('OVMAP_TEST_MASKCLUSTERING_CHECKOUT')
    if not checkout:
        pytest.skip('Set OVMAP_TEST_MASKCLUSTERING_CHECKOUT to the pinned author checkout')
    from plyfile import PlyData, PlyElement
    from static_ovmap.benchmark_completion_v2.mask_clustering_io import make_dataset
    from static_ovmap.benchmark_completion_v2.common import write
    binding, _ = replica_rgbd(tmp_path)
    names = [f'label{i}' for i in range(1, 49)]
    write(Path(binding['output_root']) / 'protocol_lock.json', dict(scopes=dict(replica48=dict(
        scenes=['room0'], text_ids=list(range(1, 49)), text_names=names))))
    geometry = tmp_path / 'geometry'
    geometry.mkdir()
    points = np.array([(3, 1, 2), (0, 1, 0), (-2, -1, 4)],
                      dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    PlyData([PlyElement.describe(points, 'vertex')], text=False).write(str(geometry / 'room0.ply'))
    binding['asset_config'] = dict(datasets=dict(replica48=dict(evaluation_mesh_root=str(geometry))),
                                  external=dict(MaskClustering=dict(checkout=checkout)))
    text = tmp_path / 'text.npy'
    np.save(text, dict(zip(names, np.eye(48, 1024), strict=True)))
    dataset = make_dataset(binding, 'room0', text_features_path=text)
    assert dataset.get_frame_list(10) == [0, 10, 20]
    assert dataset.get_depth(10).dtype == np.float32
    assert np.array_equal(dataset.get_depth(10), np.full((3, 4), 10, np.float32))
    assert dataset.get_extrinsic(10)[0, 3] == 0.12345678901234567
    assert np.array_equal(dataset.get_scene_points(), [[3, 1, 2], [0, 1, 0], [-2, -1, 4]])
    assert dataset.get_label_id()[0]['label48'] == 48
    assert np.array_equal(dataset.get_label_features()['label48'], np.eye(48, 1024)[47])
