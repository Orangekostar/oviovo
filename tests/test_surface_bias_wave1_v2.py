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

def test_05_spatial_folds_and_donor_isolation(tmp_path,monkeypatch):
    from static_ovmap.surface_bias_wave1_v2.diagnostics import spatial_folds, purge_fit_sites, diagnostic_metadata
    xyz=np.stack([np.arange(32),np.zeros(32),np.zeros(32)],1)
    blocks=spatial_folds(xyz,np.ones(32,bool))
    assert [list(b) for b in blocks]==[list(range(i,i+8)) for i in range(0,32,8)]
    donors=[{i:{(i//2,0)} for i in range(32)} for _ in range(2)]
    fit=np.ones(32,bool); fit[[1,2]]=False
    out=purge_fit_sites(fit,np.array([1,2]),donors)
    assert not out[0] and not out[3] and out[4]
    record=dict(original_hw=[8,8],resized_hw=[8,8],padded_hw=[8,8],raw_hw=[8,8])
    arrays=dict(site_xyz=np.stack([xyz,xyz+100]),site_available=np.ones((2,32),bool),uv=np.zeros((2,2,32,2)))
    metadata,reason=diagnostic_metadata(arrays,[record,record],1,2)
    np.testing.assert_array_equal(metadata['xyz'],xyz+100)
    assert reason is None
    assert diagnostic_metadata({k:v for k,v in arrays.items() if k!='site_xyz'},[record,record],0,2)==(None,'MISSING_DIAGNOSTIC_METADATA')
    assert diagnostic_metadata(arrays,[{},record],0,2)==(None,'MISSING_DIAGNOSTIC_METADATA')
    from static_ovmap.surface_bias_wave1_v2.data import forward_array_schema
    assert forward_array_schema({'masks','uv','metadata','local_valid'})=='DIAGNOSTIC_FIELDS_ABSENT'
    assert forward_array_schema({'masks','uv','metadata','local_valid','site_xyz','site_available'})=='PARENT_EXACT'
    with pytest.raises(ValueError): forward_array_schema({'masks','uv','metadata','local_valid','class_id'})
    # The thin compatibility path agrees with the actual parent loader on
    # synthetic cached tensors. Only the optional diagnostic fields differ.
    import torch
    from types import SimpleNamespace
    from static_ovmap.surface_bias_wave1_v2.common import ConsumptionIndex, arrays_record, FORWARD_KEYS
    from static_ovmap.surface_bias_wave1_v2.data import ObjectLoader
    import static_ovmap.cvpr_compact.area_fallback as area
    from static_ovmap.surface_bias_wave1_v2.diagnostics import bilinear_donors
    from static_ovmap.learned_object_readout.observations import sample_raw
    pixels=torch.eye(64).reshape(64,8,8)[None]
    uv=np.array([[1.5,1.5],[1.75,1.75],[7.5,7.5]],np.float32)
    sampled,_=sample_raw(pixels,uv,[8,8],[8,8],[8,8])
    actual_donors=bilinear_donors(uv,record,np.ones(3,bool))
    for site in range(3):
        positive=torch.nonzero(sampled[site]>0,as_tuple=True)[0].tolist()
        assert actual_donors[site]=={(i%8,i//8) for i in positive}
    monkeypatch.setattr(area,'region_vector',lambda model,operators,raw,signed:(raw.mean((0,2,3)),dict(synthetic_test=True)))
    index=ConsumptionIndex(); raw=torch.arange(12*4*4,dtype=torch.float32).reshape(1,12,4,4)+1
    grid=arrays_record(tmp_path/'grid.npz',dict(raw=raw.numpy(),projected=raw.numpy()),index)
    frames={f's:{i}':dict(record,raw_hw=[4,4],arrays=grid) for i in (0,1)}
    fc=SimpleNamespace(binding=dict(path_map={}),index=index,device='cpu',phi=lambda x:x,model=None,operators=None)
    observation=dict(masks=np.ones((4,8,8,8),bool),uv=np.full((4,8,32,2),2.,np.float32),metadata=np.zeros((4,8,32,8),np.float32),local_valid=np.ones((4,8,32),bool))
    complete=arrays_record(tmp_path/'complete.npz',dict(observation,site_xyz=np.zeros((4,32,3)),site_available=np.ones((4,32),bool)),index)
    partial=arrays_record(tmp_path/'partial.npz',observation,index)
    obj=dict(scene='s',views=[dict(frame_id=i) for i in range(8)],inputs=complete,targets=dict(path='MUST_NOT_READ_LABELS'))
    loader=ObjectLoader(fc,dict(frames=frames))
    expected,_,_=loader.load(obj,2,2)
    actual,_,_=loader.load(dict(obj,inputs=partial,targets=dict(path='DIFFERENT_LABEL_SIDECAR')),2,2)
    assert set(actual)==FORWARD_KEYS
    for key in FORWARD_KEYS: assert torch.equal(actual[key],expected[key])
    assert actual['frame_ids'].tolist()==[0,1]  # Frames2..7 do not exist in the bank.
    from static_ovmap.surface_bias_wave1_v2.binding import minimal_fc_dependencies
    from static_ovmap.surface_bias_wave1_v2.common import write,PathResolver
    context=tmp_path/'context.json'; write(context,dict(config=str(tmp_path/'FC_config.json'),FC_text=dict(path=str(tmp_path/'text.npz')),FC_model_reference_root=str(tmp_path/'FC')))
    source=dict(FC_python='test-python',cohorts=dict(scannet_cf18=['s','map_only']),scenes=dict(s=dict(context=dict(path=str(context))),map_only=dict(context=dict(path=str(tmp_path/'P3_missing_context.json')))),assets=dict(checkpoint=dict(path='UNUSED_ANYUP_WEIGHT'),execution_config=dict(assets_root=str(tmp_path/'assets'),fc=dict(operator_root=str(tmp_path/'operators')))))
    dependencies=[str(p) for p in minimal_fc_dependencies(source,PathResolver({}))]
    assert 'UNUSED_ANYUP_WEIGHT' not in dependencies
    assert str(tmp_path/'P3_missing_context.json') not in dependencies
    assert str(tmp_path/'assets/ovrcoat/ovrcoat.pth') in dependencies

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
    simple_lock={**lock,'selected':{**lock['selected'],'M1_CORRESPONDENCE_GRAPH':a,'C1_FULL_VIEW_CENTER':good}}
    assert activate(simple_lock,simple_lock['selected'],base,base)['expanded']==['C1_FULL_VIEW_CENTER']
    # A better non-champion seed29 control cannot replace the locked C_STAR.
    repeat={**lock['selected'],'C2_SITE_HUBER_CONSENSUS':{**good,'A':.99}}
    assert activate(lock,repeat,base,base)['C_STAR']=='C1_FULL_VIEW_CENTER'

def test_07_resume_identity_and_dynamic_coverage(tmp_path,monkeypatch):
    from static_ovmap.surface_bias_wave1_v2.common import cached_unit, write
    path=tmp_path/'unit.json'; write(path,dict(status='COMPLETE',input_identity='a',value=7))
    assert cached_unit(path,'a')['value']==7
    assert cached_unit(path,'b') is None
    assert list((tmp_path/'superseded').glob('*.json'))
    arrays=tmp_path/'unit.npz'; np.savez(arrays,value=np.array([7]))
    from static_ovmap.surface_bias_wave1_v2.common import digest_file
    write(path,dict(status='COMPLETE',input_identity='b',arrays=dict(path=str(arrays),bytes=arrays.stat().st_size,sha256=digest_file(arrays))))
    assert cached_unit(path,'b') is not None
    arrays.unlink()
    assert cached_unit(path,'b') is None
    from static_ovmap.surface_bias_wave1_v2.reporting import execution_complete
    completed={name:dict(status='COMPLETE') for name in ('parity','diagnostic','selection','repeat','prefix','activation')}
    completed['activation']['expanded']=[]
    assert execution_complete(**completed,evaluation=dict(status='NOT_ACTIVATED_BY_FIXED_RULE'))
    completed['activation']['expanded']=['M1']
    assert not execution_complete(**completed,evaluation=dict(status='NOT_ACTIVATED_BY_FIXED_RULE'))
    completed['parity']['status']='NOT_RUN_DEPENDENCY'
    assert not execution_complete(**completed,evaluation=dict(status='EVALUATION_COMPLETE'))
    from static_ovmap.surface_bias_wave1_v2.evaluation import required_coverage
    rows=[dict(scene='s',method='m',status='COMPLETE',evaluation_identity='x')]
    pools={'c':{'m':dict(ordered_scenes=['s'],ordered_inputs=['x'])}}
    assert required_coverage({'c':['s']},['m'],rows,pools)==(1,1)
    with pytest.raises(ValueError): required_coverage({'c':['s','t']},['m'],rows,pools)
    # Interrupt the actual readout loop after one completed condition. Resume
    # must compute the remaining conditions without repeating the first head.
    import torch
    from types import SimpleNamespace
    from static_ovmap.surface_bias_wave1_v2 import prediction,runner,publication
    from static_ovmap.surface_bias_wave1_v2.common import ConsumptionIndex
    binding=dict(identity='synthetic_binding',output_root=str(tmp_path/'readout'))
    obj=dict(key='s:1',family='s',class_id=1,base=True,inputs=dict(sha256='synthetic'))
    monkeypatch.setattr(prediction,'items',lambda binding,role='dev':[obj])
    fc=SimpleNamespace(model_key='synthetic_FC',device='cpu',ids=[1,2],index=ConsumptionIndex(),text_tensor=torch.eye(2),base_text=torch.eye(2),base_targets={1:0},phi=lambda t:t)
    state=dict(interrupt=True,head_calls=0)
    class Loader:
        features='synthetic_features'
        def load(self,obj,condition,prefix):
            if condition==1 and state['interrupt']: raise RuntimeError('controlled interruption')
            return dict(view_valid=torch.ones(2,dtype=torch.bool)),None,None
    def head(x,phi):
        state['head_calls']+=1; return dict(embedding=torch.tensor([1.,0.]))
    with pytest.raises(RuntimeError,match='controlled interruption'): prediction.recognition(binding,fc,Loader(),head,17,'FG2')
    assert state['head_calls']==1
    state['interrupt']=False
    with pytest.raises(ValueError,match='Fixed69'): prediction.recognition(binding,fc,Loader(),head,17,'FG2')
    assert state['head_calls']==4
    terminal=tmp_path/'terminal'; terminal.mkdir(); missing=tmp_path/'missing_parent'
    write(terminal/'dependency_manifest.json',dict(missing=[str(missing/'source_binding.json')],checked_parent=str(missing)))
    write(terminal/'run_summary.json',dict(execution_status='BLOCKED_DEPENDENCY',science_status='UNASSESSED_DEPENDENCY'))
    monkeypatch.setattr(publication,'unchanged_release',lambda root:dict(C2='verified_fixture'))
    monkeypatch.setattr(runner,'observed_phase',lambda *args:pytest.fail('Terminal resume repeated a worker'))
    args=SimpleNamespace(phase='all',resume=True,output_root=terminal,parent_root=missing,path_map=None,gpu=None)
    assert runner.controller(args,{})==1

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
    reused=publish_local_protocol(repo,tmp_path/'out',files,'test-wave',receipt='PUBLICATION_RECEIPT.json',handoff='HANDOFF.md')
    assert reused['C2']==result['C2']
    (repo/'code.py').write_text('x=2\n')
    revised=publish_local_protocol(repo,tmp_path/'out',files,'test-wave',receipt='PUBLICATION_RECEIPT.json',handoff='HANDOFF.md')
    assert revised['C1']!=result['C1'] and revised['C2']!=result['C2']
    assert (tmp_path/'out/publication/history'/(result['C2']+'_final.json')).exists()
    assert git('merge-base','--is-ancestor',result['C2'],revised['C1'])==''
    assert set(git('diff-tree','--no-commit-id','--name-only','-r',revised['C2']).splitlines())=={'PUBLICATION_RECEIPT.json','HANDOFF.md'}
