def test_refresh_external_asset_preserves_calibration_native_and_bootstrap_branches():
    from static_ovmap.benchmark_completion_v2.refresh import affected_phases
    phases = affected_phases({}, dict(external={'Open3DIS': {'author_predictions': '/released/predictions'}}))
    assert phases == ['baselines','tables','review','publish']


def test_refresh_replica_inputs_unlocks_only_input_and_dependent_branches():
    from static_ovmap.benchmark_completion_v2.refresh import affected_phases
    phases = affected_phases({}, dict(datasets={'replica48': {'gt_root': '/author/replica_3d'}}))
    assert set(phases) == {'benchmark_inputs','baselines','benchmark_predict','benchmark_score',
                           'statistics','tables','review','publish'}
    assert 'cache_post' not in phases and 'legacy_predict' not in phases


def test_bootstrap_duplicate_occurrences_keep_all_internal_prediction_references():
    from static_ovmap.benchmark_completion_v2.statistics import namespaced
    original = dict(gt={'chair':[dict(matched_pred=[dict(uuid='u',filename='f')])]},
                    pred={'chair':[dict(uuid='u',filename='f',matched_gt=[])]})
    first,second = namespaced(original,'first'),namespaced(original,'second')
    assert first['gt']['chair'][0]['matched_pred'][0]['uuid'] == first['pred']['chair'][0]['uuid']
    assert first['pred']['chair'][0]['uuid'] != second['pred']['chair'][0]['uuid']
    assert first['pred']['chair'][0]['filename'] != second['pred']['chair'][0]['filename']
    assert original['pred']['chair'][0]['uuid'] == 'u'


def test_bootstrap_uses_frozen_random_seed_and_shared_family_draws(tmp_path,monkeypatch):
    import json
    import numpy as np
    from static_ovmap.benchmark_completion_v2.common import REPO,write
    from static_ovmap.benchmark_completion_v2 import standard_score
    from static_ovmap.benchmark_completion_v2.statistics import paired_bootstrap
    spec=json.loads((REPO/'configs/static_ovmap/benchmark_completion_v2.json').read_text())
    scenes=['sceneA_00','sceneB_00']
    write(tmp_path/'protocol_lock.json',dict(scopes=dict(tiny=dict(scenes=scenes,families=['sceneA','sceneB']))))
    for scene in scenes:
        rows={}
        for method in ['U11','U10','U01']:
            for seed in [17,29]:
                name=method+'_seed'+str(seed)
                path=tmp_path/'matches'/scene/(name+'.json')
                matches=write(path,dict(matches={scene:dict(gt={},pred={})}))
                rows[name]=dict(matches=dict(path=str(path),identity=matches['identity']))
        write(tmp_path/'benchmark/evaluation/tiny'/scene/'receipt.json',dict(rows=rows))
    called=[]
    def score(_,matches):
        called.append(tuple(matches))
        return dict(apall=0.),{},None
    monkeypatch.setattr(standard_score,'evaluator',lambda *args: {})
    monkeypatch.setattr(standard_score,'ap_metrics',score)
    result=paired_bootstrap(dict(output_root=str(tmp_path),specification=spec),'tiny')
    assert result['random_seed']==spec['statistics']['seed']==6173
    expected=np.random.default_rng(6173).integers(0,2,size=(1000,2))
    assert np.array_equal(result['shared_family_draws'],expected)
    assert len(called)==6000
    assert all(len(set(called[i:i+6]))==1 for i in range(0,6000,6))
    assert all(row['valid_draws']==1000 for row in result['comparisons'].values())


def test_complete_negative_experiment_closes_and_terminal_cli_only_reads(tmp_path):
    import subprocess
    import sys
    from static_ovmap.benchmark_completion_v2.common import REPO, write, digest_file
    from static_ovmap.benchmark_completion_v2.runner import parser
    from static_ovmap.benchmark_completion_v2.lifecycle import close
    args = parser().parse_args(['--output-root', str(tmp_path), '--resume'])
    receipt = tmp_path/'scientific.json'
    write(receipt, dict(status='DONE',synthetic=True,positive_gain=False))
    flags = {key: True for key in ('software_complete','calibration_complete','legacy_complete',
        'standard_coverage_complete','external_comparisons_complete','publication_complete')}
    flags['claim_supported'] = False
    terminal = close(args, flags=flags,receipts=[receipt],remaining_jobs=[])
    assert terminal['lifecycle'] == 'CLOSED'
    before = {str(p): (digest_file(p),p.stat().st_mtime_ns) for p in tmp_path.iterdir() if p.is_file()}
    command = [sys.executable,str(REPO/'scripts/evaluation/run_ovmap_benchmark_completion_v2.py'),
               '--output-root',str(tmp_path),'--resume']
    actual = subprocess.run(command,cwd='/tmp',capture_output=True,text=True)
    assert actual.returncode == 0, actual.stderr
    assert 'new_workers=0 scientific_imports=0 writes=0' in actual.stdout
    assert before == {str(p): (digest_file(p),p.stat().st_mtime_ns) for p in tmp_path.iterdir() if p.is_file()}
