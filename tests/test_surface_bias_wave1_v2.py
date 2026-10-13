"""Eight bounded protocol test groups; real FC checks require original caches."""
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest

def test_01_relative_bias_and_missing_support():
    from static_ovmap.surface_bias_wave1_v2.solver import estimate_view_bias, apply_bias
    rng=np.random.default_rng(17)
    mu=rng.normal(size=(32,12)); b=rng.normal(size=(4,12)); b-=b.mean(0)
    x=mu[None]+b[:,None]; valid=np.ones((4,32),bool)
    est=estimate_view_bias(x,valid)
    np.testing.assert_allclose(est.bias,b,atol=1e-10,rtol=0)
    missing=valid.copy(); missing[:2,16:]=False; missing[2:,:16]=False
    failed=estimate_view_bias(x,missing)
    assert failed.status=='FALLBACK_DISCONNECTED_GRAPH'
    np.testing.assert_array_equal(apply_bias(x,missing,failed,.5),x)
    bad=x.copy(); bad[0,0,0]=np.nan
    with pytest.raises(ValueError): estimate_view_bias(bad,valid)

def test_02_expected_mathematical_degeneracies():
    from static_ovmap.surface_bias_wave1_v2.solver import estimate_view_bias, apply_bias
    rng=np.random.default_rng(18); x=rng.normal(size=(4,32,12))
    valid=np.ones(x.shape[:2],bool); out=apply_bias(x,valid,estimate_view_bias(x,valid),.5)
    np.testing.assert_allclose(x.mean(0),out.mean(0),atol=1e-14,rtol=0)
    np.testing.assert_allclose((x[0]-x[1]).mean(0),(x[0]-x[1,rng.permutation(32)]).mean(0),atol=1e-14)

def test_03_parent_head_value_only_and_padding():
    import torch
    from static_ovmap.frozen_fc_routing.models import FGHead
    from static_ovmap.surface_bias_wave1_v2.adapter import adapt
    torch.manual_seed(17); head=FGHead(dim=12,hidden=8).eval()
    x=dict(fc_vectors=torch.randn(4,12),view_valid=torch.ones(4,dtype=torch.bool),frame_ids=torch.arange(4),local_raw=torch.randn(4,32,12),local_unit=torch.randn(4,32,12),metadata=torch.randn(4,32,8),local_valid=torch.ones(4,32,dtype=torch.bool))
    phi=lambda t:t
    with torch.no_grad():
        base=head(x,phi); zero,audit=adapt(head,x,phi,'M1_CORRESPONDENCE_GRAPH',0)
        assert torch.equal(base['embedding'],zero['embedding'])
        before={k:v.clone() for k,v in x.items()}
        result,audit=adapt(head,x,phi,'M1_CORRESPONDENCE_GRAPH',.25)
        for key in ('quality_logits','quality_weights','alpha','null','energies'): assert torch.equal(result[key],base[key])
        for k in before: assert torch.equal(before[k],x[k])
        x['local_valid'][0,0]=False; x['local_raw'][0,0]=float('nan')
        result,audit=adapt(head,x,phi,'M1_CORRESPONDENCE_GRAPH',.25)
        assert torch.isfinite(result['embedding']).all()
        assert np.isfinite(audit['group_raw_difference_norm'])

def test_04_real_frozen_fc_parity_is_never_simulated():
    from static_ovmap.surface_bias_wave1_v2.common import REPO, verified
    path=REPO.parent/'attempt_001/parity.json'
    if not path.exists(): pytest.skip('Original FG/LR cache dependencies unavailable; real FC parity not run')
    proof=verified(path)
    if proof['status']!='COMPLETE': pytest.skip('Real FC parity blocked; not replaced by synthetic phi')
    assert proof['real_FC_used'] and proof['sample_objects']==2 and proof['bitwise_value_only']

def test_05_spatial_folds_and_donor_isolation():
    from static_ovmap.surface_bias_wave1_v2.diagnostics import spatial_folds, purge_fit_sites
    xyz=np.stack([np.arange(32),np.zeros(32),np.zeros(32)],1)
    blocks=spatial_folds(xyz,np.ones(32,bool))
    assert [list(b) for b in blocks]==[list(range(i,i+8)) for i in range(0,32,8)]
    donors=[{i:{(i//2,0)} for i in range(32)} for _ in range(2)]
    fit=np.ones(32,bool); fit[[1,2]]=False
    out=purge_fit_sites(fit,np.array([1,2]),donors)
    assert not out[0] and not out[3] and out[4]

def test_06_selection_and_activation_state_machine():
    from static_ovmap.surface_bias_wave1_v2.selection import choose, activate
    a=dict(A=.4,M=.5,C=.6,net_corrections=1,lambda_=0,numeric_fallback_rate=0)
    b={**a,'lambda_':.25}; assert choose([b,a])['lambda_']==0
    good={**a,'A':.41,'lambda_':.25}; base={**a,'net_corrections':0}
    lock=dict(selected={'M1_CORRESPONDENCE_GRAPH':good,'C1_FULL_VIEW_CENTER':a,'C2_SITE_HUBER_CONSENSUS':b},C_STAR='C1_FULL_VIEW_CENTER')
    result=activate(lock,lock['selected'],base,base)
    assert result['expanded']==['M1_CORRESPONDENCE_GRAPH','C1_FULL_VIEW_CENTER']
    failed=activate({**lock,'selected':{k:a for k in lock['selected']}},{k:a for k in lock['selected']},base,base)
    assert failed['expanded']==[]

def test_07_resume_identity_and_dynamic_coverage(tmp_path):
    from static_ovmap.surface_bias_wave1_v2.common import cached_unit, write
    path=tmp_path/'unit.json'; write(path,dict(status='COMPLETE',input_identity='a',value=7))
    assert cached_unit(path,'a')['value']==7
    with pytest.raises(ValueError): cached_unit(path,'b')
    from static_ovmap.surface_bias_wave1_v2.evaluation import required_coverage
    rows=[dict(scene='s',method='m',status='COMPLETE',evaluation_identity='x')]
    pools={'c':{'m':dict(ordered_scenes=['s'],ordered_inputs=['x'])}}
    assert required_coverage({'c':['s']},['m'],rows,pools)==(1,1)
    with pytest.raises(ValueError): required_coverage({'c':['s','t']},['m'],rows,pools)

def test_08_two_commit_publication_local_remote(tmp_path):
    from static_ovmap.surface_bias_wave1_v2.publication import publish_local_protocol
    repo=tmp_path/'repo'; remote=tmp_path/'remote.git'; repo.mkdir()
    subprocess.run(['git','init','--bare',str(remote)],check=True,capture_output=True)
    subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
    def git(*args): return subprocess.run(['git',*args],cwd=repo,check=True,capture_output=True,text=True).stdout.strip()
    git('config','user.name','Protocol test'); git('config','user.email','protocol@example.invalid'); git('remote','add','origin',str(remote))
    (repo/'code.py').write_text('x=1\n'); (repo/'HANDOFF.md').write_text('Results\n'); (repo/'说明.md').write_text('Protocol\n')
    files=['code.py','HANDOFF.md','说明.md']
    hook=remote/'hooks/pre-receive'; hook.write_text('#!/bin/sh\nexit 1\n'); hook.chmod(0o755)
    with pytest.raises(RuntimeError): publish_local_protocol(repo,tmp_path/'out',files,'test-wave',receipt='PUBLICATION_RECEIPT.json',handoff='HANDOFF.md')
    state=json.loads((tmp_path/'out/publication/transaction.json').read_text())
    assert state['stage']=='C1_LOCAL'
    hook.unlink()
    result=publish_local_protocol(repo,tmp_path/'out',files,'test-wave',receipt='PUBLICATION_RECEIPT.json',handoff='HANDOFF.md')
    assert result['C1']==state['C1']
    assert result['C1']!=result['C2'] and result['remote_HEAD']==result['C2']
    receipt=json.loads((repo/'PUBLICATION_RECEIPT.json').read_text()); assert receipt['commit']==result['C1'] and 'C2' not in receipt
    assert set(git('diff-tree','--no-commit-id','--name-only','-r',result['C2']).splitlines())=={'PUBLICATION_RECEIPT.json','HANDOFF.md'}
