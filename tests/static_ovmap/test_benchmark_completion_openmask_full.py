from pathlib import Path

import numpy as np
import pytest

from test_benchmark_completion_openmask_io import binding_for_openmask


@pytest.mark.parametrize('preexisting', [False, True])
def test_author_context_supports_legacy_sequence_and_restores_host_on_error(tmp_path, monkeypatch, preexisting):
    import collections
    import collections.abc
    import sys
    from static_ovmap.benchmark_completion_v2.openmask_full import _author_context
    repo = tmp_path / 'author'
    (repo / 'openmask3d').mkdir(parents=True)
    sentinel = object()
    if preexisting:
        monkeypatch.setattr(collections, 'Sequence', sentinel, raising=False)
    else:
        monkeypatch.delattr(collections, 'Sequence', raising=False)
    cwd, path, argv = Path.cwd(), list(sys.path), list(sys.argv)
    with pytest.raises(RuntimeError, match='exit probe'):
        with _author_context(repo, ['original_entry.py']):
            namespace = {}
            exec('from collections import Sequence', namespace)
            assert namespace['Sequence'] is (sentinel if preexisting else collections.abc.Sequence)
            raise RuntimeError('exit probe')
    assert hasattr(collections, 'Sequence') is preexisting
    if preexisting:
        assert collections.Sequence is sentinel
    assert Path.cwd() == cwd and sys.path == path and sys.argv == argv


@pytest.mark.parametrize('preloaded', [False, True])
def test_author_dataset_import_uses_pinned_namespace_and_restores_host(tmp_path, preloaded):
    import contextlib
    import importlib
    import sys
    import types
    from static_ovmap.benchmark_completion_v2.openmask_full import _author_context
    binding, _ = binding_for_openmask(tmp_path)
    repo = Path(binding['asset_config']['external']['OpenMask3D']['checkout'])
    original = {name: module for name, module in sys.modules.items()
                if name == 'datasets' or name.startswith('datasets.')}
    try:
        for name in original:
            sys.modules.pop(name)
        if preloaded:
            host = types.ModuleType('datasets')
            host.__path__ = [str(tmp_path / 'host')]
            sys.modules['datasets'] = host
            sys.modules['datasets.scannet200'] = types.ModuleType('datasets.scannet200')
        previous = {name: module for name, module in sys.modules.items()
                    if name == 'datasets' or name.startswith('datasets.')}
        expectation = pytest.raises(RuntimeError, match='exit probe') if preloaded else contextlib.nullcontext()
        with expectation:
            with _author_context(repo, ['original_entry.py']):
                module = importlib.import_module('datasets.scannet200.scannet200_splits')
                assert Path(module.__file__).resolve() == (
                    repo / 'openmask3d/class_agnostic_mask_computation/datasets/scannet200/scannet200_splits.py').resolve()
                if preloaded:
                    raise RuntimeError('exit probe')
        assert {name: module for name, module in sys.modules.items()
                if name == 'datasets' or name.startswith('datasets.')} == previous
    finally:
        for name in list(sys.modules):
            if name == 'datasets' or name.startswith('datasets.'):
                sys.modules.pop(name)
        sys.modules.update(original)


def test_changed_openmask_view_frequency_is_rejected_before_workers(tmp_path):
    import yaml
    from static_ovmap.benchmark_completion_v2.openmask_full import locked_configuration
    binding, _ = binding_for_openmask(tmp_path)
    assets = binding['asset_config']['external']['OpenMask3D']
    config = yaml.safe_load((Path(assets['checkout']) /
                            'openmask3d/configs/openmask3d_inference.yaml').read_text())
    config['openmask3d']['frequency'] = 20
    changed = tmp_path / 'changed.yaml'
    changed.write_text(yaml.safe_dump(config))
    assets['original_config'] = str(changed)
    with pytest.raises(ValueError, match='numeric'):
        locked_configuration(binding)
    assert not (Path(binding['output_root']) / 'jobs').exists()


def test_openmask_missing_original_assets_stop_before_any_child(tmp_path):
    from static_ovmap.benchmark_completion_v2.common import StageFailure
    from static_ovmap.benchmark_completion_v2.openmask_full import dependencies, run_scene
    binding, _ = binding_for_openmask(tmp_path)
    report = dependencies(binding, 'room0')
    absent = {row['asset']: row for row in report['missing']}
    assert {'python', 'mask_ckpt_replica', 'sam_checkpoint', 'clip_assets',
            'author_colored_evaluation_PLY', 'author_instance_GT'} <= set(absent)
    assert absent['mask_ckpt_replica']['expected'].endswith('arbitrary_scene_mask3d.ckpt')
    assert absent['sam_checkpoint']['expected'].endswith('sam_vit_h_4b8939.pth')
    assert absent['clip_assets']['expected'].endswith('ViT-L-14-336px.pt')
    with pytest.raises(StageFailure) as error:
        run_scene(binding, 'room0')
    assert error.value.status == 'BLOCKED_DEPENDENCY'
    assert not (Path(binding['output_root']) / 'jobs').exists()


def test_openmask_cached_query_uses_original_confidence_and_empty_feature_tie(tmp_path):
    import torch
    from static_ovmap.benchmark_completion_v2.openmask_full import paths, run_worker
    binding, _ = binding_for_openmask(tmp_path)
    files = paths(binding, 'room0')
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    masks = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float16)
    torch.save(masks, files['masks'])
    features = np.zeros((3, 768), np.float32)
    features[0, 47], features[1, 0] = .4, .2
    np.save(files['features'], features)
    np.save(files['text'], np.eye(48, 768, dtype=np.float32))
    run_worker(binding, 'room0', 'query')
    with np.load(files['query'], allow_pickle=False) as prediction:
        assert np.array_equal(prediction['pred_masks'], masks)
        assert prediction['pred_classes'].tolist() == [48, 1, 1]
        assert prediction['pred_scores'].tolist() == [1, 1, 1]
    assert not (Path(binding['output_root']) / 'external/OpenMask3D/GT').exists()


def test_openmask_owns_disk_reserve_despite_complete_ours_cache(tmp_path, monkeypatch):
    import shutil
    from static_ovmap.benchmark_completion_v2 import capacity
    from static_ovmap.benchmark_completion_v2.common import StageFailure, write
    from static_ovmap.benchmark_completion_v2.openmask_full import check_capacity
    binding, _ = binding_for_openmask(tmp_path)
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


def test_openmask_baseline_reports_concrete_dependencies_and_empty_formal_pool(tmp_path):
    from static_ovmap.benchmark_completion_v2.baselines import openmask_replica
    from static_ovmap.benchmark_completion_v2.common import verified
    from static_ovmap.benchmark_completion_v2.work_state import openmask_state
    binding, _ = binding_for_openmask(tmp_path)
    result = openmask_replica(binding)
    assert result['status'] == 'BLOCKED_DEPENDENCY'
    assert result['pool']['metrics'] is None
    assert result['pool']['completed_scenes'] == 0
    root = Path(binding['output_root'])
    report = verified(root / 'external/OpenMask3D/pipeline/replica48/room0/availability.json')
    assert any(row['asset'] == 'mask_ckpt_replica' for row in report['missing'])
    state = openmask_state(binding, 'room0')
    assert state['status'] == 'BLOCKED_DEPENDENCY' and state['runnable'] is False
    assert state['reason'] == 'FULL_OPENMASK3D_ASSETS_MISSING'
    assert not (root / 'jobs').exists()
