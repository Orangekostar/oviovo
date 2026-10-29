from pathlib import Path

import pytest

from test_benchmark_completion_maskclustering_score import replica_score_fixture


def test_openmask_incomplete_stages_cannot_freeze_or_read_gt(tmp_path):
    from static_ovmap.benchmark_completion_v2.common import write
    from static_ovmap.benchmark_completion_v2.openmask_score import freeze
    binding = dict(output_root=str(tmp_path))
    write(tmp_path / 'external/OpenMask3D/pipeline/replica48/room0/receipt.json',
          dict(status='DONE', scene='room0', full_original_inference_stages=True, actual_jobs={}))
    with pytest.raises(ValueError, match='four original'):
        freeze(binding, 'room0')
    assert not (tmp_path / 'external/OpenMask3D/GT').exists()


def test_openmask_full_gt_author_agreement_and_synthetic_pool_exclusion(tmp_path):
    from static_ovmap.benchmark_completion_v2.openmask_score import score_prediction, pool
    binding, path, _, _ = replica_score_fixture(tmp_path, 'OpenMask3D')
    before = (path.parent / 'masks.npz').read_bytes()
    score = score_prediction(binding, 'room0', path)
    assert score['metrics'] == dict(apall=.5, ap50=.5, ap25=.5)
    assert score['method'] == 'OpenMask3D' and score['original_GT_instances'] == 2
    assert score['author_agreement']['all_class_IoU_AP_arrays_equal'] is True
    assert score['author_agreement']['maximum_absolute_error'] == 0
    assert score['synthetic'] is True and score['formal_G3_verified'] is False
    assert score_prediction(binding, 'room0', path)['identity'] == score['identity']
    assert (path.parent / 'masks.npz').read_bytes() == before
    assert not (Path(binding['output_root']) / 'benchmark/bundles').exists()
    result = pool(binding)
    assert result['metrics'] is None and result['completed_scenes'] == 0
    assert result['full_scene_coverage'] is False


def test_openmask_changed_prediction_fails_before_any_gt_read(tmp_path):
    from static_ovmap.benchmark_completion_v2.openmask_score import score_prediction
    binding, path, _, manifest = replica_score_fixture(tmp_path, 'OpenMask3D')
    with Path(manifest['arrays']['path']).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='arrays changed'):
        score_prediction(binding, 'room0', path)
    assert not (Path(binding['output_root']) / 'external/OpenMask3D/GT').exists()


def test_openmask_reordered_gt_point_domain_cannot_be_scored(tmp_path):
    import torch
    from static_ovmap.benchmark_completion_v2.openmask_score import score_prediction
    binding, path, source, _ = replica_score_fixture(tmp_path, 'OpenMask3D')
    value = list(torch.load(source, weights_only=False))
    value[0] = value[0][::-1].copy()
    torch.save(tuple(value), source)
    with pytest.raises(ValueError, match='original point order'):
        score_prediction(binding, 'room0', path)
    assert not list((Path(binding['output_root']) / 'external/OpenMask3D').glob('GT/**/*.txt'))


def test_unverified_first_scene_stops_later_full_method_execution(tmp_path):
    from static_ovmap.benchmark_completion_v2.baselines import _replica_method
    from static_ovmap.benchmark_completion_v2.common import verified, write
    from static_ovmap.benchmark_completion_v2.openmask_score import score_prediction, pool
    binding, manifest, _, _ = replica_score_fixture(tmp_path, 'OpenMask3D')
    path = Path(binding['output_root']) / 'protocol_lock.json'
    protocol = verified(path)
    protocol['scopes']['replica48']['scenes'] = ['room0', 'office0']
    write(path, protocol)
    # This real CPU author-agreement receipt is synthetic and is insufficient
    # to authorize a later full GPU scene, despite status=DONE and equal AP.
    def connection_score(current, scene):
        return score_prediction(current, scene, manifest)
    result = _replica_method(binding, 'OpenMask3D', lambda *args: dict(missing=[]),
                             connection_score, pool, ('openmask_score',))
    assert result['scenes']['room0']['status'] == 'FAILED_RUNTIME'
    assert result['scenes']['office0']['status'] == 'BLOCKED_DEPENDENCY'
    assert result['pool']['completed_scenes'] == 0
