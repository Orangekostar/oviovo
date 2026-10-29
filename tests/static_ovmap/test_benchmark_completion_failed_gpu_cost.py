"""Synthetic launcher accounting: CPU-only children never initialize a GPU."""
import contextlib
import os
import sys

import pytest


def test_failed_external_gpu_attempts_count_toward_quota_and_retry_limit(tmp_path, monkeypatch):
    from static_ovmap.benchmark_completion_v2 import runtime
    from static_ovmap.benchmark_completion_v2.common import StageFailure, digest_file, verified, write

    monkeypatch.setattr(runtime, 'gpu_lock', lambda _: contextlib.nullcontext())
    binding = dict(output_root=str(tmp_path))
    name = 'external/Example/replica48/room0/proposals'
    command = [sys.executable, '-c', 'raise SystemExit(7)']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    receipt = tmp_path/'jobs'/name/'receipt.json'
    for attempt in (1, 2):
        with pytest.raises(StageFailure) as error:
            runtime.execute(binding, name, command, cwd=tmp_path, env=env, gpu=True)
        assert error.value.status == 'FAILED_RUNTIME'
        row = verified(receipt)
        assert row['actual_child_exit'] == 7 and len(row['attempts']) == attempt
        expected = sum(item['elapsed_seconds'] for item in row['attempts'])
        assert expected > 0
        assert runtime.gpu_seconds(tmp_path) == expected
        assert row['gpu_worker_launches'] == 1
    before = digest_file(receipt), receipt.stat().st_mtime_ns
    with pytest.raises(StageFailure, match='retry budget exhausted'):
        runtime.execute(binding, name, command, cwd=tmp_path, env=env, gpu=True)
    assert (digest_file(receipt), receipt.stat().st_mtime_ns) == before
    write(tmp_path/'runtime_constraints.json', dict(max_gpu_hours=expected/7200))
    with pytest.raises(StageFailure) as error:
        runtime.check_gpu_budget(binding)
    assert error.value.status == 'BLOCKED_RESOURCE'


def test_missing_outputs_bill_gpu_worker_but_not_cpu_worker(tmp_path, monkeypatch):
    from static_ovmap.benchmark_completion_v2 import runtime
    from static_ovmap.benchmark_completion_v2.common import StageFailure, verified

    monkeypatch.setattr(runtime, 'gpu_lock', lambda _: contextlib.nullcontext())
    binding = dict(output_root=str(tmp_path))
    command = [sys.executable, '-c', 'pass']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    for gpu, name in ((True, 'external/Example/replica48/room0/features'),
                      (False, 'external/Example/replica48/room0/export')):
        absent = tmp_path/(str(gpu)+'_missing.npz')
        with pytest.raises(StageFailure) as error:
            runtime.execute(binding, name, command, cwd=tmp_path, env=env, outputs=[absent], gpu=gpu)
        assert error.value.status == 'FAILED_RUNTIME'
        row = verified(tmp_path/'jobs'/name/'receipt.json')
        assert row['actual_child_exit'] == 0 and row['missing'] == [str(absent)]
        if gpu:
            expected = row['attempts'][0]['elapsed_seconds']
            assert expected > 0
        assert runtime.gpu_seconds(tmp_path) == expected
        assert row['gpu_worker_launches'] == int(gpu)
