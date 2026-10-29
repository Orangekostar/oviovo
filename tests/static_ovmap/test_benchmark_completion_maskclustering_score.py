import os
from pathlib import Path

import numpy as np
import pytest


def replica_score_fixture(tmp_path, method='MaskClustering'):
    import torch
    from plyfile import PlyData, PlyElement
    from static_ovmap.benchmark_completion_v2.common import write, digest_file
    from static_ovmap.benchmark_completion_v2.external import save_prediction
    repositories = {key: os.environ.get('OVMAP_TEST_' + key + '_CHECKOUT')
                    for key in ('OPENMASK3D', 'OPEN3DIS')}
    if not all(repositories.values()):
        pytest.skip('Set the pinned OPENMASK3D and OPEN3DIS test checkout paths')
    root = tmp_path / 'out'
    author = Path(repositories['OPEN3DIS'])
    (root / 'assets/external').mkdir(parents=True)
    (root / 'assets/external/Open3DIS').symlink_to(author, target_is_directory=True)
    sources = dict(evaluator_scan=Path(repositories['OPENMASK3D']) /
                   'openmask3d/evaluation/eval_semantic_instance.py',
                   evaluator_replica=author / 'open3dis/evaluation/scannetv2_inst_eval.py')
    write(root / 'protocol_lock.json', dict(official_sources={key: dict(
        path=str(path), sha256=digest_file(path)) for key, path in sources.items()},
        scopes=dict(replica48=dict(scenes=['room0'], instance_ids=list(range(1, 49)),
            instance_names=[f'class{i}' for i in range(1, 49)],
            IoU_thresholds=np.r_[np.arange(.5, .95, .05), .25].tolist(), minimum_region_vertices=100))))
    xyz = np.arange(600, dtype=np.float32).reshape(200, 3) / 100
    vertices = np.empty(200, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    for i, key in enumerate(('x', 'y', 'z')):
        vertices[key] = xyz[:, i]
    geometry = tmp_path / 'room0.ply'
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(str(geometry))
    gt_root = tmp_path / 'tuples'
    gt_root.mkdir()
    source = gt_root / 'room0.pth'
    torch.save((xyz, np.zeros_like(xyz), np.repeat([47, 48], 100),
                np.repeat([0, 1], 100)), source)
    masks = np.zeros((200, 2), bool)
    masks[100:, 0] = True
    folder = root / 'external' / method / 'predictions/replica48/room0'
    labels, scores = ([48, 0], [.875, .25]) if method == 'MaskClustering' else ([48, 1], [1., 1.])
    manifest = save_prediction(folder, dict(pred_masks=masks, pred_classes=np.array(labels),
        pred_scores=np.array(scores)), method=method, benchmark='replica48', scene='room0',
        provenance=dict(synthetic=True, geometry=dict(path=str(geometry), sha256=digest_file(geometry))))
    binding = dict(output_root=str(root), asset_config=dict(datasets=dict(replica48=dict(gt_root=str(gt_root)))))
    return binding, folder / 'manifest.json', source, manifest


def test_incomplete_original_pipeline_cannot_freeze_or_read_gt(tmp_path):
    from static_ovmap.benchmark_completion_v2.mask_clustering_score import freeze
    from static_ovmap.benchmark_completion_v2.common import write
    binding = dict(output_root=str(tmp_path))
    write(tmp_path / 'external/MaskClustering/pipeline/replica48/room0/receipt.json',
          dict(status='DONE', scene='room0', full_original_inference_stages=True, actual_jobs={}))
    with pytest.raises(ValueError, match='five original'):
        freeze(binding, 'room0')
    assert not (tmp_path / 'external/MaskClustering/GT').exists()


def test_replica_common_score_retains_full_gt_and_agrees_with_author_without_ours(tmp_path):
    from static_ovmap.benchmark_completion_v2.mask_clustering_score import score_prediction, pool
    binding, path, _, _ = replica_score_fixture(tmp_path)
    score = score_prediction(binding, 'room0', path)
    assert score['metrics'] == dict(apall=.5, ap50=.5, ap25=.5)
    assert score['original_GT_instances'] == 2
    assert score['full_GT_denominator'] is True
    assert score['synthetic'] is True and score['formal_G3_verified'] is False
    assert score['author_agreement']['all_class_IoU_AP_arrays_equal'] is True
    assert score['author_agreement']['maximum_absolute_error'] == 0
    assert not (Path(binding['output_root']) / 'benchmark/bundles').exists()
    before = (path.parent / 'masks.npz').read_bytes()
    assert score_prediction(binding, 'room0', path)['identity'] == score['identity']
    assert (path.parent / 'masks.npz').read_bytes() == before
    # A synthetic connection test can never populate a real standard pool.
    result = pool(binding)
    assert result['completed_scenes'] == 0 and result['metrics'] is None
    assert result['full_scene_coverage'] is False


def test_changed_prediction_is_rejected_before_any_gt_export(tmp_path):
    from static_ovmap.benchmark_completion_v2.mask_clustering_score import score_prediction
    binding, path, _, manifest = replica_score_fixture(tmp_path)
    with Path(manifest['arrays']['path']).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='arrays changed'):
        score_prediction(binding, 'room0', path)
    assert not (Path(binding['output_root']) / 'external/MaskClustering/GT').exists()


def test_reordered_author_gt_points_are_rejected(tmp_path):
    import torch
    from static_ovmap.benchmark_completion_v2.mask_clustering_score import score_prediction
    binding, path, source, _ = replica_score_fixture(tmp_path)
    value = list(torch.load(source, weights_only=False))
    value[0] = value[0][::-1].copy()
    torch.save(tuple(value), source)
    with pytest.raises(ValueError, match='original point order'):
        score_prediction(binding, 'room0', path)
    assert not list((Path(binding['output_root']) / 'external/MaskClustering').glob('GT/**/*.txt'))
