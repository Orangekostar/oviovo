import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path
from static_ovmap.semantic_update_evidence_v1.publication import allowed,check_files
from static_ovmap.semantic_update_evidence_v1.calibration import gate
from static_ovmap.semantic_update_evidence_v1.stages import inactive
from static_ovmap.structure_confirmation_v1.evaluation import proposal_manifest_entries
from static_ovmap.contribution_attribution_v1.stability import namespace_matches
from static_ovmap.semantic_update_evidence_v1.common import save_memo,ConsumptionIndex,read

def test_no_activation_on_missing_or_insufficient_signal(tmp_path):
    rows={m:[dict(delta_H_macro=.02,positive_families=8),dict(delta_H_macro=.009,positive_families=12)] for m in ('S_LL','G_LL')}
    decision=gate(rows);assert not decision['open']
    records=inactive(tmp_path,dict(phases=['B_train','C_acquire']),decision)
    assert all(r['actual_new_workers']==0 for r in records.values())
    rows['S_LL'][1]['delta_H_macro']=None;assert not gate(rows)['open']

def test_namespace_unknown_denominator_and_scope():
    source={'scene':{'gt':{'chair':[{'filename':'p.npy'}]}}}
    a=namespace_matches(source,'draw0');b=namespace_matches(source,'draw1')
    assert not set(a)&set(b)
    assert a['draw0|scene']['gt']['chair'][0]['filename']!=b['draw1|scene']['gt']['chair'][0]['filename']
    assert source['scene']['gt']['chair'][0]['filename']=='p.npy'
    masks=[np.ones(110,bool),np.ones(110,bool)]
    entries=proposal_manifest_entries(masks,[7,0],[7])
    assert len(entries)==1 and entries[0]['rank']=='1.000000'
    assert len(masks)==2 # abstention was not removed from original coverage
    assert allowed('src/static_ovmap/semantic_update_evidence_v1/metrics.py')
    assert not allowed('src/static_ovmap/structure_confirmation_v1/metrics.py')
    with pytest.raises(ValueError):check_files(['README.md'])

def test_cached_bind_cannot_drop_prior_consumed_identities(tmp_path):
    a=tmp_path/'first';b=tmp_path/'second';memo=tmp_path/'memo.json'
    a.write_bytes(b'first');b.write_bytes(b'second')
    first=ConsumptionIndex();desc=first.identity(a);save_memo(first,memo)
    second=ConsumptionIndex(memo);second.identity(b);save_memo(second,memo)
    assert {r['identity']['path'] for r in read(memo)['entries']}=={str(a),str(b)}
    a.write_bytes(b'changed')
    with pytest.raises(ValueError):ConsumptionIndex(memo).identity(a,desc)

def test_missing_cache_binding_is_nonzero_partial_not_negative_result(tmp_path):
    repo=Path(__file__).resolve().parents[2]
    out=tmp_path/'attempt'
    result=subprocess.run([sys.executable,str(repo/'scripts/evaluation/run_semantic_update_evidence_v1.py'),
         '--parent-root',str(tmp_path/'missing_parent'),'--output-root',str(out),'--phase','science'],capture_output=True,text=True)
    assert result.returncode==2
    record=read(out/'science.json')
    assert record['status']=='PARTIAL_DEPENDENCY' and 'source_binding.json' in record['error']
    assert record['new_head_calls']==0
