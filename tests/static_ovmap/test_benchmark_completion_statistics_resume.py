"""Interrupted pooled bootstrap must retain complete draws without changing the plan."""
import json
from pathlib import Path

import numpy as np
import pytest

from static_ovmap.benchmark_completion_v2.common import REPO, verified, write
from static_ovmap.benchmark_completion_v2 import standard_score
from static_ovmap.benchmark_completion_v2.statistics import paired_bootstrap


def fixture(tmp_path, monkeypatch):
    spec = json.loads((REPO / 'configs/static_ovmap/benchmark_completion_v2.json').read_text())
    scenes = ['sceneA_00', 'sceneB_00']
    write(tmp_path / 'protocol_lock.json', dict(scopes=dict(tiny=dict(
        scenes=scenes, families=['sceneA', 'sceneB']))))
    for scene in scenes:
        rows = {}
        for method in ('U11', 'U10', 'U01'):
            for seed in (17, 29):
                key = method + '_seed' + str(seed)
                path = tmp_path / 'matches' / scene / (key + '.json')
                record = write(path, dict(matches={scene: dict(gt={}, pred={}, method=key)}))
                rows[key] = dict(matches=dict(path=str(path), identity=record['identity']))
        write(tmp_path / 'benchmark/evaluation/tiny' / scene / 'receipt.json', dict(rows=rows))
    monkeypatch.setattr(standard_score, 'evaluator', lambda *args: {})
    return dict(output_root=str(tmp_path), specification=spec)


def metric(matches):
    # A metric boundary double: real match loading, namespacing, cache and statistics run.
    # A draw containing only B has undefined AP and must never be resampled.
    values = {'U11_seed17': .8, 'U11_seed29': .4, 'U10_seed17': .5,
              'U10_seed29': .2, 'U01_seed17': .9, 'U01_seed29': .5}
    key = next(iter(matches.values()))['method']
    value = None if all(tag.endswith('sceneB_00') for tag in matches) else values[key]
    return dict(apall=value), {}, None


def interrupt_after_two_draws(binding, monkeypatch):
    completed = []
    def interrupted(_, matches):
        if len(completed) == 12:
            raise KeyboardInterrupt('controlled interruption after two complete shared draws')
        completed.append(tuple(matches))
        return metric(matches)
    monkeypatch.setattr(standard_score, 'ap_metrics', interrupted)
    with pytest.raises(KeyboardInterrupt):
        paired_bootstrap(binding, 'tiny')
    assert len(completed) == 12
    assert not (Path(binding['output_root']) / 'statistics/tiny.json').exists()
    return set(completed)


def test_interrupted_draws_resume_without_recomputing_or_resampling(tmp_path, monkeypatch):
    binding = fixture(tmp_path, monkeypatch)
    completed = interrupt_after_two_draws(binding, monkeypatch)
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in (tmp_path / 'statistics/checkpoints/tiny').glob('*.json')}
    called = []
    def resumed(_, matches):
        assert tuple(matches) not in completed, 'A fully completed draw was recomputed after interruption'
        called.append(tuple(matches))
        return metric(matches)
    monkeypatch.setattr(standard_score, 'ap_metrics', resumed)
    result = paired_bootstrap(binding, 'tiny')
    assert len(called) == 998 * 6
    assert all((Path(p).read_bytes(), Path(p).stat().st_mtime_ns) == value for p, value in before.items())
    expected_plan = np.random.default_rng(6173).integers(0, 2, size=(1000, 2))
    assert np.array_equal(result['shared_family_draws'], expected_plan)
    undefined = int(np.all(expected_plan == 1, axis=1).sum())
    for comparison, per_seed, delta in [('U11-U10', [.3, .2], .25), ('U11-U01', [-.1, -.1], -.1)]:
        row = result['comparisons'][comparison]
        assert (row['draws'], row['undefined_draws'], row['valid_draws']) == (1000, undefined, 1000 - undefined)
        assert sum(value is None for value in row['mean_delta_draws']) == undefined
        assert np.allclose([value for value in row['mean_delta_draws'] if value is not None], delta)
        assert np.allclose([value for value in row['per_seed_deltas'] if value[0] is not None], per_seed)
        assert np.allclose(row['percentile_95'], [delta, delta])
    saved = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / 'statistics').rglob('*.json')}
    monkeypatch.setattr(standard_score, 'ap_metrics', lambda *args: pytest.fail('Completed statistics ran an evaluator'))
    assert paired_bootstrap(binding, 'tiny') == result
    assert saved == {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / 'statistics').rglob('*.json')}


@pytest.mark.parametrize('changed', ['family_order', 'match_record'])
def test_pending_bootstrap_rejects_changed_scientific_inputs_before_new_scores(tmp_path, monkeypatch, changed):
    binding = fixture(tmp_path, monkeypatch)
    interrupt_after_two_draws(binding, monkeypatch)
    if changed == 'family_order':
        protocol = verified(tmp_path / 'protocol_lock.json')
        protocol['scopes']['tiny']['families'].reverse()
        write(tmp_path / 'protocol_lock.json', protocol)
    else:
        path = tmp_path / 'matches/sceneA_00/U11_seed17.json'
        record = verified(path)
        record['matches']['sceneA_00']['changed_input_marker'] = True
        new = write(path, record)
        scene_path = tmp_path / 'benchmark/evaluation/tiny/sceneA_00/receipt.json'
        scene = verified(scene_path)
        scene['rows']['U11_seed17']['matches']['identity'] = new['identity']
        write(scene_path, scene)
    saved = {str(p): p.read_bytes() for p in (tmp_path / 'statistics').rglob('*.json')}
    monkeypatch.setattr(standard_score, 'ap_metrics', lambda *args: pytest.fail('Changed cached inputs reached scoring'))
    with pytest.raises(ValueError, match='[Ii]nput|[Pp]rotocol|[Ii]dentity'):
        paired_bootstrap(binding, 'tiny')
    assert saved == {str(p): p.read_bytes() for p in (tmp_path / 'statistics').rglob('*.json')}
