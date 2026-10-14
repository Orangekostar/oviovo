"""Four bounded contract groups; real FC/scorer parity is recorded at runtime."""
import importlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest

PACKAGE = 'static_ovmap.contribution_attribution_v1'

def modules(*names):
    assert importlib.util.find_spec(PACKAGE), 'contribution attribution implementation is missing'
    return [importlib.import_module(PACKAGE + '.' + name) for name in names]

def test_T1_probability_missing_sources_and_dev_selection():
    s, selection = modules('sources', 'selection')
    probabilities = {'N': np.array([.8, .2]), 'Q': np.array([.2, .8]), 'F': np.array([.1, .9])}
    full = s.combine(probabilities, ('N', 'Q', 'F'), 'PARENT_D2')
    weighted = s.combine(probabilities, ('N', 'Q', 'F'), 'DEV_SELECTED_WEIGHT', (.25, .25, .5))
    np.testing.assert_allclose(full, [.3, .7], rtol=0, atol=1e-15)
    np.testing.assert_allclose(full, weighted, rtol=0, atol=1e-15)
    missing = {**probabilities, 'Q': None}
    np.testing.assert_allclose(s.combine(missing, ('N', 'Q', 'F'), 'PARENT_D2'), [.45, .55])
    np.testing.assert_allclose(s.combine(missing, ('N', 'Q', 'F'), 'DEV_SELECTED_WEIGHT', (.25, .25, .5)), [.3333333333333333, .6666666666666667])
    empty = dict.fromkeys(('N', 'Q', 'F'))
    assert s.classify(s.combine(empty, ('N',), 'EQUAL_AVAILABLE'), [4, 9], 7, 'EQUAL_AVAILABLE') == 0
    assert s.classify(s.combine(empty, ('N', 'Q', 'F'), 'PARENT_D2'), [4, 9], 7, 'PARENT_D2') == 7
    sources = {n: {'valid_ids': [4, 9], 'objects': {'1': {'available': True, 'scores': [0., 2.]}}} for n in empty}
    p = s.source_probabilities(sources, dict.fromkeys(empty, 1.), [4, 9], 1)
    assert p['N'][1] > .8
    sources['N']['objects']['1']['scores'] = [float('nan'), 2.]
    with pytest.raises(ValueError, match='available'):
        s.source_probabilities(sources, dict.fromkeys(empty, 1.), [4, 9], 1)
    grid = [(i/4, j/4, (4-i-j)/4) for i in range(5) for j in range(5-i)]
    assert len(grid) == 15
    candidates = [dict(weight=list(w), metrics=dict(apall=.2, miou=.3)) for w in grid]
    assert selection.select_weight(candidates)['weight'] == [0., 0., 1.]
    assert selection.select_weight([dict(weight=[1., 0., 0.], metrics=dict(apall=None, miou=.3))]) is None
    assert selection.no_dev_lock()['status'] == 'NOT_SELECTED_NO_FUSION_DEV'

def test_T2_fixed_support_unknown_and_joint_token_shuffle():
    p, s, recovery = modules('payloads', 'surface', 'recovery')
    from static_ovmap.module_validation.evaluation import PredictionPayload, GeometryIdentity
    geometry = GeometryIdentity('1'*64, '2'*64, '3'*64, 'fixed', 4)
    native = PredictionPayload('N0', 'N0', 'test', geometry, np.array([0, 1, 1, 2]), np.array([0, 4, 4, 9]), ((1, 1.), (2, 1.)), {}, {})
    native.lock()
    new = p.fixed_owner_output(native, {1: 0, 2: 4}, [4, 9], np.arange(4), np.ones(4, bool), 'E2')
    np.testing.assert_array_equal(new.owner_ids, native.owner_ids)
    np.testing.assert_array_equal(new.semantic_labels, [0, 0, 0, 4])
    assert new.locked and 1 in dict(new.instance_ranks)
    p.validate_eligible_replacement({1: 4, 2: 9}, {1: 0, 2: 9}, [1])
    with pytest.raises(ValueError):
        p.validate_eligible_replacement({1: 4, 2: 9}, {1: 0, 2: 4}, [1])
    recovery.validate_residual_support(np.array([0, 1, 1, 0]), np.array([3, 1, 2, 3]), np.array([3, 1, 1, 3]))
    with pytest.raises(ValueError):
        recovery.validate_residual_support(np.array([0, 1, 1, 0]), np.array([3, 1, 2, 3]), np.array([3, 1, 2, 3]))
    import torch
    raw = torch.arange(2*5*3, dtype=torch.float32).reshape(2, 5, 3)
    x = dict(local_raw=raw, local_unit=raw+100, metadata=raw+200, local_valid=torch.tensor([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.bool), view_valid=torch.ones(2, dtype=torch.bool), fc_vectors=torch.ones(2, 3), frame_ids=torch.tensor([7, 11]))
    before = {k: v.clone() for k, v in x.items()}
    shuffled = s.permute_valid(x, 'object', 'clean', 101)
    again = s.permute_valid(x, 'object', 'clean', 101)
    for key in x:
        assert torch.equal(x[key], before[key]) and torch.equal(shuffled[key], again[key])
    for view in range(2):
        valid = x['local_valid'][view]
        assert torch.equal(shuffled['local_raw'][view, ~valid], raw[view, ~valid])
        assert sorted(shuffled['local_raw'][view, valid, 0].tolist()) == sorted(raw[view, valid, 0].tolist())
        assert torch.equal(shuffled['local_unit'][view, valid]-shuffled['local_raw'][view, valid], torch.full((3, 3), 100.))
        assert torch.equal(shuffled['metadata'][view, valid]-shuffled['local_raw'][view, valid], torch.full((3, 3), 200.))
    permutation = torch.tensor([4, 0, 3, 1, 2])
    common = s.permute_slots(x, permutation)
    assert torch.equal(common['local_valid'], x['local_valid'][:, permutation])
    assert torch.equal(common['local_raw'], raw[:, permutation])

def test_T3_factorial_matching_rank_and_repeated_family_namespace():
    h, r, stability, rank = modules('history', 'recovery', 'stability', 'rank')
    effects = h.factorial(.1, .2, .15, .31)
    assert effects['interaction'] == pytest.approx(.06)
    assert effects['mean_marginal_E']+effects['mean_marginal_R'] == pytest.approx(.21)
    owners = np.repeat([1, 2], 100)
    gt = np.repeat([1001, 1002], 100)
    result = r.one_to_one(owners, gt, [1], 100, .5)
    assert result['matching_count'] == 2 and result['recall'] == 1.
    assert r.one_to_one(owners, gt, [1], 100, 1.)['matching_count'] == 0
    assert result['measurement_kind'] == 'DIAGNOSTIC_ONE_TO_ONE_GEOMETRY'
    view = rank.explicit_view(owners, {1: 1, 2: 0}, {1: .3123459, 2: 0.}, 100)
    assert view == {1: dict(label=1, rank='0.312346', area=100)}
    with pytest.raises(ValueError, match='rank'):
        rank.explicit_view(owners, {1: 1, 2: 1}, {1: .3}, 100)
    matches = {'scene': {'gt': {'a': [dict(instance_id=1001, matched_pred=[dict(filename='mask/1.npy', confidence=.2, intersection=7)])]}, 'pred': {'a': [dict(filename='mask/1.npy', matched_gt=[dict(instance_id=1001)])]}}}
    a, b = stability.namespace_matches(matches, 'copy0'), stability.namespace_matches(matches, 'copy1')
    assert set(a).isdisjoint(b)
    ma = next(iter(a.values()))
    assert ma['gt']['a'][0]['matched_pred'][0]['filename'] == ma['pred']['a'][0]['filename']
    assert ma['pred']['a'][0]['filename'] != next(iter(b.values()))['pred']['a'][0]['filename']
    assert matches['scene']['pred']['a'][0]['filename'] == 'mask/1.npy'
    assert ma['gt']['a'][0]['matched_pred'][0]['intersection'] == 7

def test_T4_bounds_immutable_resume_prelock_and_cli(tmp_path,monkeypatch):
    c, selection, publication, resume = modules('common', 'selection', 'publication', 'resume')
    nomination, = modules('nomination')
    candidate=dict(specific_positive_cells=2,primary_positive_cells=2,lfo_positive_fraction=.5,mean_primary_delta_apall=.01,missing=[])
    tied=nomination.shortlist([dict(candidate,module='S',lfo_positive_fraction=.5+5e-11),dict(candidate,module='E_GROUP')])
    assert [r['module'] for r in tied]==['E_GROUP','S']
    negative=dict(candidate,module='R',specific_positive_cells=0,primary_positive_cells=0)
    assert nomination.shortlist([negative])==[]
    paths, = modules('paths')
    from types import SimpleNamespace
    fallback=tmp_path/'fg_fallback';area=tmp_path/'area';out=tmp_path/'resolved';out.mkdir()
    c.write(area/'experiment.json',dict(parent=str(tmp_path/'compact')))
    c.write(fallback/'source_binding.json',dict(specification=dict(task_id='ovimap-frozen-fc-routing-v1'),cohorts={'replica8':['r']},reference=dict(v2_results_root=str(area))))
    monkeypatch.setattr(paths,'FG_CANDIDATES',(str(tmp_path/'missing'),str(fallback)))
    args=SimpleNamespace(output_root=out,parent_root=None,area_root=None,compact_root=None,backbone_root=None,wave2_root=None,path_map=None,gpu=None)
    paths.resolve_defaults(args,dict(cohorts={'replica8':['r']}))
    assert args.parent_root==fallback and args.area_root==area and args.compact_root==tmp_path/'compact'
    c.write(out/'source_binding.json',dict(path_map={},gpu='0',**{field:str(getattr(args,option)) for option,field in paths.ROOT_FIELDS.items()}))
    args.gpu='1'
    with pytest.raises(ValueError,match='Resume GPU'):paths.resolve_defaults(args,dict(cohorts={'replica8':['r']}))
    path = tmp_path/'lock.json'
    c.immutable_record(path, dict(status='COMPLETE', input_identity='input1', weight=[.25, .25, .5]))
    assert c.cached_record(path, 'input1')['weight'] == [.25, .25, .5]
    with pytest.raises(ValueError):
        c.cached_record(path, 'input2')
    with pytest.raises(ValueError):
        c.immutable_record(path, dict(status='COMPLETE', input_identity='input1', weight=[1., 0., 0.]))
    selection.validate_prediction_lock(dict(role='FUSION_DEV', status='ALL_PREDICTIONS_LOCKED', scenes=['a', 'b'], methods=['w0', 'w1'], records=[dict(scene=s, method=m) for s in ['a', 'b'] for m in ['w0', 'w1']]), ['a', 'b'], ['w0', 'w1'])
    with pytest.raises(ValueError):
        selection.validate_prediction_lock(dict(role='REGRESSION', status='ALL_PREDICTIONS_LOCKED', scenes=['a'], methods=['w0'], records=[]), ['a'], ['w0'])
    consumed=tmp_path/'consumed';consumed.mkdir();small=consumed/'small';small.write_bytes(b'123')
    assert publication.check_sizes([small], file_limit=3, total_limit=3) == 3
    with pytest.raises(ValueError):
        publication.check_sizes([small], file_limit=2, total_limit=3)
    index=c.ConsumptionIndex();index.identity(small);index.write_memo(consumed/'input_verifications.json')
    assert resume.check_consumed(consumed)['consumed_files']==1
    small.write_bytes(b'124')
    with pytest.raises(ValueError):resume.check_consumed(consumed)
    spec = Path(__file__).resolve().parents[2]/'configs/static_ovmap/contribution_attribution_v1.json'
    import json
    config = json.loads(spec.read_text())
    assert len(config['new_main_methods']) == 12 and config['limits']['new_optimizer_updates'] == 0
    assert config['confirmation']['execute'] is False and config['limits']['focused_test_groups'] == 4
    import subprocess, sys
    git_repo=tmp_path/'git_paths';git_repo.mkdir()
    subprocess.run(['git','init','-q',str(git_repo)],check=True)
    staged_names=['中文提示词.md','line\nbreak.md']
    for name in staged_names:(git_repo/name).write_text('task-owned content\n')
    subprocess.run(['git','add','--',*staged_names],cwd=git_repo,check=True)
    with monkeypatch.context() as patch:
        patch.setattr(publication,'REPO',git_repo)
        assert set(publication._staged_paths())==set(staged_names)
    command=[sys.executable,str(spec.parents[2]/'scripts/evaluation/run_ovmap_contribution_attribution_v1.py'),'--help']
    help_text=subprocess.check_output(command,text=True)
    assert '--parent-root' in help_text and 'main_score' in help_text and 'publish' in help_text
