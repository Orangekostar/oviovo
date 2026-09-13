import pickle
from pathlib import Path

import numpy as np
import pytest

from src.static_ovmap.query_history import capture_history, instrument_mapper


def records():
    full = {5: {'frame_id': list(range(12)), 'feat': np.arange(36).reshape(12, 3),
                'pose': np.repeat(np.eye(4)[None], 12, axis=0),
                'box_2d': [(0, 1, 2, 3)] * 12,
                'vis_area': np.array([10, 1, 9, 2, 8, 3, 7, 4, 6, 5, 11, 12])}}
    selected = np.argsort(full[5]['vis_area'])[-10:]
    native = {5: {k: np.asarray(v)[selected] for k, v in full[5].items()}}
    native[5]['color'] = [20, 30, 40]
    return full, native, selected


def test_complete_history_preserves_discarded_queries_and_native_order(tmp_path):
    full, native, selected = records()
    original = pickle.dumps((full, native))
    result = capture_history(full, native, tmp_path/'history', max_top_vis=10)
    saved = pickle.loads((tmp_path/'history/full_query_cache.pkl').read_bytes())
    assert len(saved[5]['frame_id']) == 12
    assert result['native_to_full_indices']['5'] == selected.tolist()
    assert np.array_equal(saved[5]['feat'], full[5]['feat'])
    assert pickle.dumps((full, native)) == original
    native[5]['feat'][0, 0] = -999
    with pytest.raises(ValueError, match='native retained'):
        capture_history(full, native, tmp_path/'bad', max_top_vis=10)


def test_instrumentation_only_adds_normal_completion_capture():
    source = 'def main(args):\n    value = args + 2\n    events.append(value)\n'
    captured = []
    namespace = {'events': [], '_static_capture': lambda scope: captured.append(scope['value'])}
    exec(instrument_mapper(source, 'toy.py'), namespace)
    namespace['main'](3)
    assert namespace['events'] == [5]
    assert captured == [5]
    with pytest.raises(ValueError, match='return'):
        instrument_mapper('def main(args):\n    return args\n', 'bad.py')
