"""Pending scene dependencies must not hide real failures or stop later scores."""
from pathlib import Path

import pytest

from static_ovmap.benchmark_completion_v2.common import verified, write
from static_ovmap.benchmark_completion_v2.runner import _scene_matrix


def protocol(root, benchmark, scenes):
    write(root/'protocol_lock.json', dict(scopes={benchmark: dict(status='DONE', scenes=scenes)}))
    return dict(output_root=str(root))


@pytest.mark.parametrize('benchmark', ['scannet200_val', 'replica48'])
def test_pending_prediction_is_dependency_without_runtime_traceback(tmp_path, benchmark):
    # Calling semantic score preparation before the freeze exists breaks this.
    binding = protocol(tmp_path, benchmark, ['pending'])
    result = _scene_matrix(binding, score=True)
    assert result['status'] == 'BLOCKED_DEPENDENCY'
    assert result['completed'] == []
    assert len(result['blocked']) == 1
    pending = result['blocked'][0]
    assert pending['status'] == 'BLOCKED_DEPENDENCY' and pending['scene'] == 'pending'
    assert pending['missing'][0]['path'] == str(tmp_path/'benchmark/predictions'/benchmark/'pending/receipt.json')
    assert not (tmp_path/'failures').exists()
    assert not (tmp_path/'benchmark/GT').exists()
    progress = verified(tmp_path/'progress/benchmark_score.json')
    assert progress['expected_scenes'] == 1 and progress['blocked'] == result['blocked']


def test_pending_prediction_does_not_stop_later_existing_scene(tmp_path, monkeypatch):
    from static_ovmap.benchmark_completion_v2 import standard_score, capacity
    binding = protocol(tmp_path, 'scannet200_val', ['pending', 'ready'])
    write(tmp_path/'benchmark/predictions/scannet200_val/ready/receipt.json',
          dict(status='DONE', methods={}, synthetic=True))
    cached = write(tmp_path/'benchmark/evaluation/scannet200_val/ready/receipt.json',
                   dict(status='DONE', synthetic=True))
    calls = []
    def cached_score(actual_binding, benchmark, scene, *, formal_connection):
        calls.append((benchmark, scene, formal_connection))
        assert actual_binding == binding
        return verified(tmp_path/'benchmark/evaluation'/benchmark/scene/'receipt.json')
    monkeypatch.setattr(standard_score, 'score_scene', cached_score)
    # Resource observation is outside score dispatch and needs real model caches.
    monkeypatch.setattr(capacity, 'observe_scene', lambda *args: None)
    result = _scene_matrix(binding, score=True)
    assert result['status'] == 'BLOCKED_DEPENDENCY'
    assert result['completed'] == [dict(benchmark='scannet200_val', scene='ready',
                                       identity=cached['identity'], status='DONE')]
    assert calls == [('scannet200_val', 'ready', False)]
    assert [row['scene'] for row in result['blocked']] == ['pending']
    assert not (tmp_path/'failures').exists()


@pytest.mark.parametrize('damage', ['invalid_json', 'missing_archive', 'broken_link'])
def test_existing_bad_freeze_or_unexpected_missing_input_remains_runtime_failure(tmp_path, damage):
    # Broad FileNotFoundError handling would incorrectly swallow these failures.
    binding = protocol(tmp_path, 'scannet200_val', ['broken'])
    receipt = tmp_path/'benchmark/predictions/scannet200_val/broken/receipt.json'
    receipt.parent.mkdir(parents=True)
    if damage == 'invalid_json':
        receipt.write_text('{broken')
    elif damage == 'broken_link':
        receipt.symlink_to(tmp_path/'missing-target.json')
    else:
        write(receipt, dict(status='DONE', semantic_refresh=dict(previous_prediction=dict(
            path=str(tmp_path/'missing-archive.json'), sha256='0'*64))))
    result = _scene_matrix(binding, score=True)
    assert result['status'] == 'FAILED_RUNTIME' and result['completed'] == []
    assert len(result['blocked']) == 1
    error = result['blocked'][0]
    assert error['status'] == 'FAILED_RUNTIME'
    trace = Path(error['traceback']).read_text()
    assert ('JSONDecodeError' if damage == 'invalid_json' else 'FileNotFoundError') in trace
    assert 'missing' not in error
