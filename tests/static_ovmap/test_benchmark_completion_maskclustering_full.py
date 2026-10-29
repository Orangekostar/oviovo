import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_benchmark_completion_external_io import replica_rgbd


def author_checkout():
    checkout = os.environ.get('OVMAP_TEST_MASKCLUSTERING_CHECKOUT')
    if not checkout:
        pytest.skip('Set OVMAP_TEST_MASKCLUSTERING_CHECKOUT to the pinned checkout')
    return Path(checkout)


@pytest.mark.parametrize('preloaded', [False, True])
def test_author_utilities_import_without_host_package_shadowing(tmp_path, preloaded):
    import importlib
    import sys
    import types
    from static_ovmap.benchmark_completion_v2.mask_clustering_full import _author_context
    checkout = author_checkout()
    original = {name: value for name, value in sys.modules.items()
                if name == 'utils' or name.startswith('utils.')}
    try:
        for name in original:
            sys.modules.pop(name)
        if preloaded:
            importlib.import_module('utils')
            sys.modules['utils.geometry'] = types.ModuleType('utils.geometry')
        previous = {name: value for name, value in sys.modules.items()
                    if name == 'utils' or name.startswith('utils.')}
        with _author_context(checkout, tmp_path / 'work', object(), lambda _: None):
            geometry = importlib.import_module('utils.geometry')
            assert Path(geometry.__file__).resolve() == (checkout / 'utils/geometry.py').resolve()
            assert callable(geometry.denoise)
        assert {name: value for name, value in sys.modules.items()
                if name == 'utils' or name.startswith('utils.')} == previous
    finally:
        for name in list(sys.modules):
            if name == 'utils' or name.startswith('utils.'):
                sys.modules.pop(name)
        sys.modules.update(original)


def test_changed_clustering_threshold_is_rejected_before_execution(tmp_path):
    from static_ovmap.benchmark_completion_v2.mask_clustering_full import locked_args
    checkout = author_checkout()
    binding, _ = replica_rgbd(tmp_path)
    config = json.loads((checkout / 'configs/scannet.json').read_text())
    config['view_consensus_threshold'] = .1
    override = tmp_path / 'changed.json'
    override.write_text(json.dumps(config))
    binding['asset_config'] = dict(external=dict(MaskClustering=dict(
        checkout=str(checkout), original_config=str(override))))
    with pytest.raises(ValueError, match='numeric'):
        locked_args(binding, 'room0')
    assert not (Path(binding['output_root']) / 'jobs').exists()


def test_missing_assets_have_concrete_paths_and_start_no_worker(tmp_path):
    from static_ovmap.benchmark_completion_v2.mask_clustering_full import dependencies
    checkout = author_checkout()
    binding, _ = replica_rgbd(tmp_path)
    binding['asset_config'] = dict(external=dict(MaskClustering=dict(checkout=str(checkout))))
    result = dependencies(binding, 'room0')
    absent = {row['asset']: row for row in result['missing']}
    assert 'cropformer_checkpoint' in absent and 'clip_assets' in absent
    assert absent['cropformer_checkpoint']['expected'].endswith('Mask2Former_hornet_3x_576d0b.pth')
    assert absent['clip_assets']['expected'].endswith('open_clip_pytorch_model.bin')
    assert 'author_evaluation_point_domain' in absent
    assert not (Path(binding['output_root']) / 'jobs').exists()


def test_external_pipeline_checks_disk_even_when_ours_visual_cache_is_complete(tmp_path, monkeypatch):
    import shutil
    from static_ovmap.benchmark_completion_v2 import capacity
    from static_ovmap.benchmark_completion_v2.common import StageFailure, write
    from static_ovmap.benchmark_completion_v2.mask_clustering_full import check_capacity
    binding, _ = replica_rgbd(tmp_path)
    binding['specification'] = dict(runtime=dict(minimal_disk_reserve_GiB=20))
    root = Path(binding['output_root'])
    for name in ('bundles/replica48/room0/bundle.json', 'F/replica48/room0/receipt.json',
                 'NQ/replica48/room0/sources.json'):
        write(root / 'benchmark' / name, dict(status='DONE'))
    gib = 2**30
    monkeypatch.setattr(capacity, 'estimate_scene', lambda *args: dict(estimated_bytes=4*gib))
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: shutil._ntuple_diskusage(100*gib, 0, 28*gib-1))
    with pytest.raises(StageFailure) as error:
        check_capacity(binding, 'room0')
    assert error.value.status == 'BLOCKED_RESOURCE'
    assert not (root / 'jobs').exists()


def test_original_semantic_query_preserves_overlap_and_empty_slot(tmp_path):
    from plyfile import PlyData, PlyElement
    from static_ovmap.benchmark_completion_v2.common import write
    from static_ovmap.benchmark_completion_v2.mask_clustering_full import run_worker, workspace
    from static_ovmap.benchmark_completion_v2.replica_layout import prepare
    checkout = author_checkout()
    binding, _ = replica_rgbd(tmp_path)
    names = [f'label{i}' for i in range(1, 49)]
    write(Path(binding['output_root']) / 'protocol_lock.json', dict(scopes=dict(replica48=dict(
        scenes=['room0'], text_names=names, text_ids=list(range(1, 49))))))
    geometry = tmp_path / 'geometry'
    geometry.mkdir()
    points = np.array([(3, 1, 2), (0, 1, 0), (-2, -1, 4), (0, 0, 3)],
                      dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    PlyData([PlyElement.describe(points, 'vertex')], text=False).write(str(geometry / 'room0.ply'))
    binding['asset_config'] = dict(datasets=dict(replica48=dict(evaluation_mesh_root=str(geometry))),
                                  external=dict(MaskClustering=dict(checkout=str(checkout))))
    layout = prepare(binding, 'room0')
    objects = Path(layout['layout_root']) / 'output/object/replica48'
    objects.mkdir(parents=True)
    np.save(objects / 'object_dict.npy', {
        12: dict(point_ids=[0, 1, 2], repre_mask_list=[(0, 1, .9)]),
        3: dict(point_ids=[1, 2, 3], repre_mask_list=[(10, 2, .8)]),
        21: dict(point_ids=[0], repre_mask_list=[])})
    first = np.zeros(1024, np.float32)
    first[0] = .4
    second = np.zeros(1024, np.float32)
    second[47] = .4
    np.save(objects / 'open-vocabulary_features.npy', {'0_1': first, '10_2': second})
    text = workspace(binding) / 'data/text_features/replica48.npy'
    text.parent.mkdir(parents=True)
    np.save(text, dict(zip(names, np.eye(48, 1024, dtype=np.float32), strict=True)))
    before = Path.cwd()
    run_worker(binding, 'room0', 'query')
    assert Path.cwd() == before
    output = workspace(binding) / 'data/prediction/replica48/room0.npz'
    with np.load(output) as prediction:
        assert prediction['pred_classes'].tolist() == [1, 48, 0]
        assert prediction['pred_score'].tolist() == [1, 1, 1]
        assert prediction['pred_masks'].tolist() == [
            [True, False, False], [True, True, False],
            [True, True, False], [False, True, False]]
