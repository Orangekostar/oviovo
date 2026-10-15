"""Four focused production contracts; real FC/scorer checks run in the task."""
import copy
import importlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest

PACKAGE='static_ovmap.structure_confirmation_v1'

def modules(*names):
    assert importlib.util.find_spec(PACKAGE), 'structure confirmation production package is missing'
    return [importlib.import_module(PACKAGE+'.'+name) for name in names]

def inputs():
    import torch
    gen=torch.Generator().manual_seed(71)
    return dict(fc_vectors=torch.randn(3,768,generator=gen),view_valid=torch.ones(3,dtype=torch.bool),frame_ids=torch.tensor([4,9,13]),local_raw=torch.randn(3,5,1536,generator=gen),local_unit=torch.nn.functional.normalize(torch.randn(3,5,768,generator=gen),dim=-1),metadata=torch.randn(3,5,8,generator=gen),local_valid=torch.tensor([[1,0,1,1,0],[1,1,1,0,0],[0,1,1,1,1]],dtype=torch.bool))

def projection():
    import torch
    with torch.random.fork_rng():
        torch.manual_seed(73);phi=torch.nn.Linear(1536,768)
    return phi.requires_grad_(False)

def test_T1_seven_tensor_grouping_prefix_and_parent_parity(tmp_path,monkeypatch):
    models,grouping=modules('models','grouping')
    import torch
    from static_ovmap.frozen_fc_routing.models import initialize
    torch.set_num_threads(1)
    x=inputs();before={k:v.clone() for k,v in x.items()};y,audit=grouping.permute_inputs(x,'object:a','clean')
    yy,_=grouping.permute_inputs({k:v[:2] for k,v in x.items()},'object:a','clean')
    for k in x:
        assert torch.equal(x[k],before[k]) and torch.equal(y[k][:2],yy[k])
    for v in range(3):
        valid=x['local_valid'][v]
        assert torch.equal(y['local_valid'][v],valid)
        assert sorted(y['local_raw'][v,valid,0].tolist())==sorted(x['local_raw'][v,valid,0].tolist())
        for dest,source in audit['source_slots'][v]:
            for key in ['local_raw','local_unit','metadata']:assert torch.equal(y[key][v,dest],x[key][v,source])
    heads=models.make_models(17);phi=projection()
    for name,group in [('S_LL','SURFACE'),('V_LL','VIEW')]:
        parent=initialize(17,dict(grouping=group,direct_routing=False));parent.load_state_dict(heads[name].state_dict())
        expected=parent(x,phi);actual=heads[name](x,phi)
        torch.testing.assert_close(actual['embedding'],expected['embedding'],rtol=0,atol=0)
    s=heads['S_LL'](x,phi);g=heads['G_LL'](y,phi)
    assert s['group_count']==g['group_count'] and torch.equal(s['members'],g['members'])
    with pytest.raises(ValueError,match='seven'):heads['S_LL'](dict(x,targets=torch.ones(1)),phi)
    no_view=dict(x,view_valid=torch.zeros(3,dtype=torch.bool));assert heads['S_LL'](no_view,phi)['embedding'] is None
    recognition,common=modules('recognition','common');seen=[]
    class Reader(torch.nn.Module):
        def forward(self,inputs,phi,**kwargs):return {}
    def parent_fixture(adapter,loader,selected,fc,**kwargs):
        for obj in selected:
            for condition in range(4):adapter({},None)
        return dict(status='COMPLETE')
    def transform(method,x,key,condition):seen.append((key,condition));return x,None
    with monkeypatch.context() as scope:
        scope.setattr(recognition,'parent_evaluate',parent_fixture);scope.setattr(recognition,'method_inputs',transform)
        items=[dict(key='base-a',base=True),dict(key='heldout-b',base=False),dict(key='base-c',base=True)]
        recognition.evaluate(Reader(),'G_LL',None,items,None,common.Budget(tmp_path/'dev_budget.json',{'main_head':8}),save_path=tmp_path/'dev.json')
    assert [key for key,condition in seen]==['base-a']*4+['base-c']*4

def test_T2_differentiable_mass_raw_projection_and_static_boundaries():
    models,=modules('models')
    import torch
    x=inputs();phi=projection();heads=models.make_models(17)
    for name in ['S_LL','S_UL','S_LU','S_UU','S_PRE']:
        h=heads[name];r=h(x,phi);r['embedding'].sum().backward()
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in h.named_parameters() if 'membership' not in n)
        assert all(p.grad is None and not p.requires_grad for p in h.membership.parameters())
        if name in ['S_UL','S_UU']:assert all(p.grad is None for p in h.quality.parameters())
        if name in ['S_LU','S_UU']:
            torch.testing.assert_close(r['alpha'],r['routing_mass'].expand_as(r['alpha'])/r['group_count'])
            assert r['routing_mass'].requires_grad and h.queries.grad is not None and bool(h.queries.grad.abs().sum()>0)
    assert all(p.grad is None for p in phi.parameters())
    pre=heads['S_PRE'](x,phi);raw=torch.where(x['local_valid'][...,None],x['local_raw'],0).reshape(-1,1536)
    projected=phi(raw);valid=x['local_valid'].flatten();projected=torch.where(valid[:,None],projected,0)
    expected=torch.nn.functional.normalize(pre['quality_weights']@projected,dim=-1)
    torch.testing.assert_close(pre['group_vectors'],expected,rtol=1e-6,atol=1e-7)
    empty=dict(x,local_valid=torch.zeros_like(x['local_valid']))
    for name in ['B_FC','B_SURFACE_MEAN','B_PFE_ADAPTED']:
        r=models.static_model(name)(empty,phi);b=torch.nn.functional.normalize(x['fc_vectors'].mean(0),dim=0)
        torch.testing.assert_close(r['embedding'],b)
    with torch.no_grad():
        r=heads['MV_ATTN'](x,phi);torch.testing.assert_close(r['embedding'],r['base'],rtol=1e-6,atol=1e-7)
    assert all(h.grouping=='SURFACE' for n,h in heads.items() if n not in ['V_LL','MV_ATTN'])

def test_T3_updated_resume_draw_order_and_seed17_selection(tmp_path):
    checkpoints,selection=modules('checkpoints','selection')
    import torch,random
    from static_ovmap.learned_object_readout.training import optimizer_for,draw_schedule
    torch.manual_seed(91);model=torch.nn.Linear(3,2);opt=optimizer_for(model)
    def step(m,o):
        o.zero_grad();loss=m(torch.randn(4,3)).square().mean();loss.backward();o.step()
    step(model,opt);checkpoints.save_resume(tmp_path,model,opt,seed=17,method='test',completed=1,input_identity='fixed',validations=[])
    step(model,opt);expected=copy.deepcopy(model.state_dict())
    restored=torch.nn.Linear(3,2);ropt=optimizer_for(restored);state=checkpoints.restore_resume(tmp_path,restored,ropt,input_identity='fixed');assert state['step']==1
    step(restored,ropt)
    for k,v in expected.items():assert torch.equal(v,restored.state_dict()[k])
    rows=[dict(step=500,A=.4,M=.3,C=.2),dict(step=250,A=.4+5e-11,M=.3,C=.2),dict(step=1000,A=.3,M=.9,C=.9)]
    assert selection.choose_step(rows)['step']==250
    objects=[dict(base=False,class_id=4),dict(base=True,class_id=8),dict(base=True,class_id=9)]
    schedule=draw_schedule(objects,17,updates=1);assert set(schedule[:,0])=={1,2}
    assert np.array_equal(schedule,draw_schedule(objects,17,updates=1))
    with pytest.raises(ValueError):checkpoints.restore_resume(tmp_path,restored,ropt,input_identity='changed')

def test_T4_overlapping_proposals_resources_namespace_and_cli(tmp_path,monkeypatch):
    evaluation,common=modules('evaluation','common')
    a=np.zeros(300,bool);a[:200]=True;b=np.zeros(300,bool);b[100:]=True;c=np.ones(300,bool)
    rows=evaluation.proposal_manifest_entries([a,b,c],[4,9,0],[4,9],minimum=100)
    assert [r['proposal_index'] for r in rows]==[0,1] and all(r['rank']=='1.000000' for r in rows)
    assert int((a&b).sum())==100 and c.sum()==300
    budget=common.Budget(tmp_path/'budget.json',{'head':2});budget.consume('head',2)
    with pytest.raises(ValueError,match='budget'):budget.consume('head',1)
    assert common.verified(tmp_path/'budget.json')['counts']['head']==2
    from static_ovmap.contribution_attribution_v1.stability import namespace_matches
    matches={'scene':{'pred':{'class':[dict(filename='mask0',matched_gt=[])]},'gt':{'class':[]}}}
    first=namespace_matches(matches,'copy0');second=namespace_matches(matches,'copy1');assert set(first).isdisjoint(second)
    repo=Path(__file__).resolve().parents[2]
    import subprocess,sys
    help_text=subprocess.check_output([sys.executable,str(repo/'scripts/evaluation/run_fg2_structure_confirmation_v1.py'),'--help'],text=True)
    for flag in ['--attribution-root','--scannet-root','--sam2-python','confirm_real_predict','map_score']:assert flag in help_text
    codec,=modules('prediction_codec')
    scores=np.random.default_rng(101).normal(size=(7,4,200)).astype(np.float32);scores[0,0,:]=np.nan;scores[1,0,0]=np.float32(-0.)
    codec.encode(tmp_path/'full200.scores.gz',scores);decoded=codec.decode(tmp_path/'full200.scores.gz')
    assert decoded.dtype==scores.dtype and decoded.shape==scores.shape and decoded.tobytes()==scores.tobytes()
    acquisition,=modules('acquisition');remote=dict(url='https://example.invalid/frozen',size_bytes=10,etag='fixed',last_modified='fixed');calls=[]
    with pytest.raises(FileNotFoundError):acquisition.run(dict(output_root=str(tmp_path/'unlocked')))
    def failed_curl(command,**kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command,28)
    with monkeypatch.context() as scope:
        scope.setattr(subprocess,'run',failed_curl);scope.setattr(acquisition,'remote_identity',lambda url:remote);scope.setattr(acquisition.time,'sleep',lambda seconds:None)
        with pytest.raises(subprocess.CalledProcessError):acquisition.bounded_transfer(remote['url'],tmp_path/'bounded.raw',remote)
        with pytest.raises(common.DependencyError,match='initial GET and one retry'):acquisition.bounded_transfer(remote['url'],tmp_path/'bounded.raw',remote)
    assert len(calls)==2
    # A successful resumed attempt must replace the current error summary.
    # Historical failures stay in the archived attempt, not the current status.
    runner,=modules('runner');attempt=tmp_path/'resumed_attempt'
    common.write(attempt/'execution/errors.json',dict(status='PARTIAL',errors=[dict(phase='old_attempt')]))
    monkeypatch.setattr(runner,'phase',lambda *args:None)
    assert runner.main(['--output-root',str(attempt),'--phase','all','--science-worker','--resume'])==0
    errors=common.verified(attempt/'execution/errors.json')
    assert errors['status']=='COMPLETE' and errors['errors']==[]
