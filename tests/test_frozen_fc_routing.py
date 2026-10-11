"""Twelve bounded production validation groups; actual CUDA/scorer pilots run in engineer."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from static_ovmap.frozen_fc_routing.models import (
    FGHead, FrozenFCBase, initialize, unit, route, correct, matched_mass, match_delta_norm, interventions,
)
from static_ovmap.frozen_fc_routing.losses import item_loss
from static_ovmap.frozen_fc_routing.selection import choose, qualifies, path_counts
from static_ovmap.frozen_fc_routing.binding import validate_lineage
from static_ovmap.frozen_fc_routing.export import load_export
from static_ovmap.learned_object_readout.training import draw_schedule, optimizer_for, save_checkpoint, restore_checkpoint, model_digest

torch.set_num_threads(1)


def sample():
    torch.manual_seed(123)
    phi=nn.Sequential(nn.Linear(5,6),nn.Tanh()).requires_grad_(False)
    raw=torch.randn(3,4,5)
    x=dict(fc_vectors=unit(torch.randn(3,6)),view_valid=torch.ones(3,dtype=torch.bool),frame_ids=torch.tensor([10,20,30]),
           local_raw=raw,local_unit=unit(phi(raw)).detach(),metadata=torch.randn(3,4,8),
           local_valid=torch.tensor([[1,1,0,1],[1,0,1,1],[1,1,1,0]],dtype=torch.bool))
    return x,phi


def head(group='SURFACE',routing=True):
    torch.manual_seed(1026)
    return FGHead(group,routing,dim=6,hidden=12)


def test_01_lineage_rejects_parent_result_mismatch_not_failed_R():
    spec=dict(lineage=dict(pr_commit='prsha',lr_commit='lrsha',pr_terminal='COMPLETE_NO_2D_FOUNDATION',parent_split_identity='split'))
    pr=dict(status='COMPLETE_NO_2D_FOUNDATION',identity='pr',scientific_optimizer_updates=8000,scene_method_coverage=0,full_cohort_pool_coverage=0)
    lr=dict(identity='lr',scene_method_coverage=286,full_cohort_pool_coverage=22,scene_metrics=[{}]*286,pooled_metrics={'a':{str(i):{} for i in range(22)}})
    pub=lambda sha,result:dict(status='PUSH_VERIFIED',local_HEAD=sha,remote_HEAD=sha,result_store_identity=result,full_resume_exit_code=0)
    args=[spec,pr,pub('prsha','pr'),lr,pub('lrsha','lr'),{'identity':'split'}]
    args[2]['all_exit_code']=args[2].pop('full_resume_exit_code')
    validate_lineage(*args)
    bad=copy.deepcopy(args);bad[2]['result_store_identity']='different'
    with pytest.raises(ValueError):validate_lineage(*bad)
    bad=copy.deepcopy(args);bad[3]['scene_method_coverage']=285
    with pytest.raises(ValueError):validate_lineage(*bad)


def test_02_base_uses_valid_ordinary_FC_and_rejects_missing_inputs():
    x,phi=sample();x['view_valid'][1]=False;x['fc_vectors'][1].fill_(float('nan'))
    expected=unit(x['fc_vectors'][[0,2]].mean(0));base=FrozenFCBase()
    assert sum(p.numel() for p in base.parameters())==0
    assert torch.equal(base(x),expected)
    with pytest.raises(ValueError):head()({**x,'class_id':42},phi)
    with pytest.raises(ValueError):base({k:v for k,v in x.items() if k!='fc_vectors'})


def test_03_pool_then_phi_keeps_quality_input_derivative():
    x,phi=sample();m=head();r=m(x,phi)
    (r['embedding']*torch.arange(1.,7.)).sum().backward()
    assert m.quality[0].weight.grad.norm()>0 and m.queries.grad.norm()>0
    assert m.membership[-1].weight.grad.norm()>0
    assert all(p.grad is None for p in phi.parameters())
    raw=x['local_raw'][0,:2]
    assert not torch.allclose(unit(phi(raw.mean(0))),unit(unit(phi(raw)).mean(0)),atol=1e-5)


def test_04_disabled_routes_and_missing_support_are_exact_FC():
    x,phi=sample();m=head();b=FrozenFCBase()(x)
    for kwargs in ({'eta_override':0.},{'rho_override':0.},{'training_forward':True,'completed_steps':0}):
        r=m(x,phi,**kwargs);assert torch.equal(r['embedding'],b)
        r['embedding'].sum().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    x['local_valid'].zero_()
    for k in ('local_raw','local_unit','metadata'):x[k].fill_(float('nan'))
    r=m(x,phi);assert r['fallback']=='NO_LOCAL_SUPPORT_FC_FALLBACK' and r['group_count']==0
    assert torch.equal(r['embedding'],b)


def test_05_masked_grouping_and_permutation_preserve_only_real_evidence():
    x,phi=sample();m=head();r=m(x,phi)
    vp=torch.tensor([2,0,1]);sp=torch.tensor([3,1,0,2])
    y={k:v[vp][:,sp] if k in ('local_raw','local_unit','metadata','local_valid') else v[vp] for k,v in x.items()}
    torch.testing.assert_close(r['embedding'],m(y,phi)['embedding'],atol=2e-6,rtol=2e-6)
    for k in ('local_raw','local_unit','metadata'):x[k][~x['local_valid']]=float('nan')
    torch.testing.assert_close(r['embedding'],m(x,phi)['embedding'])
    assert head('VIEW',False)(x,phi)['group_count']==3 and head('SURFACE',False)(x,phi)['group_count']==4
    assert not torch.allclose(head('VIEW',False)(x,phi)['embedding'],head('SURFACE',False)(x,phi)['embedding'])


def test_06_initialization_and_draws_are_matched_across_arms():
    cfgs=[dict(grouping=g,direct_routing=r) for g,r in [('VIEW',False),('SURFACE',False),('SURFACE',False),('SURFACE',True)]]
    models=[initialize(17,c) for c in cfgs]
    assert all(sum(p.numel() for p in m.parameters())==216194 for m in models)
    assert len({model_digest(m) for m in models})==1
    assert all(torch.count_nonzero(m.membership[-1].weight)==0 for m in models)
    items=[dict(base=True,class_id=1),dict(base=False,class_id=2),dict(base=True,class_id=3)]
    draws=draw_schedule(items,17)
    assert np.array_equal(draws,draw_schedule(items,17)) and not np.any(draws[:,0]==1)
    assert draws[:6,1:].tolist()==[[2,1],[4,1],[8,1],[2,2],[4,2],[8,2]]


def test_07_BCE_pools_known_tokens_and_disabled_head_gets_no_supervision():
    x,phi=sample();m=head(routing=False);a=m(x,phi);b=m(x,phi)
    ys=[torch.full((3,4),-1),torch.full((3,4),-1)];ys[0][0,0]=1;ys[1][0,:2]=0
    text=unit(torch.randn(4,6))
    loss,parts=item_loss(a,b,0,text,memberships=ys,membership_enabled=False)
    loss.backward();assert all(p.grad is None for p in m.membership.parameters())
    assert parts['known_tokens']==3 and parts['positive_tokens']==1 and parts['negative_tokens']==2
    m.zero_grad();a=m(x,phi);b=m(x,phi)
    loss,parts=item_loss(a,b,0,text,memberships=ys,membership_enabled=True);loss.backward()
    assert m.membership[-1].bias.grad.abs()>0
    assert parts['membership_loss']==pytest.approx(np.log(2),abs=1e-6)
    unknown=[torch.full((3,4),-1)]*2
    _,parts=item_loss(m(x,phi),m(x,phi),0,text,memberships=unknown,membership_enabled=True)
    assert parts['known_tokens']==0 and parts['membership_loss']==0


def test_08_mass_and_norm_interventions_keep_their_declared_invariants():
    e=torch.zeros(1,2);a,n=route(e,torch.tensor([.1,.9]));am=matched_mass(e,a)
    torch.testing.assert_close(a.sum(-1),am.sum(-1))
    base=torch.tensor([1.,0.]);v=torch.tensor([[-1.,1.],[-1.,-1.]])
    _,d=correct(base,v,a,.5);_,dm=correct(base,v,am,.5)
    assert abs(float(d.norm()-dm.norm()))>1e-3
    z,audit=match_delta_norm(base,d,dm);assert audit['status']=='MATCHED'
    torch.testing.assert_close(audit['delta'].norm(),d.norm())
    z,audit=match_delta_norm(base,d,torch.zeros_like(d));assert z is None and audit['status']=='UNDEFINED_ZERO_CONTROL'
    aa,_=route(torch.zeros(4,3),torch.ones(3));bb,_=route(torch.zeros(4,30),torch.ones(30))
    torch.testing.assert_close(aa.sum(-1),bb.sum(-1))
    reduced,_=route(torch.zeros(4,3),torch.tensor([.5,1.,1.]));assert (reduced[:,0]<aa[:,0]).all()


def test_09_selection_excludes_step0_and_never_switches_repeat_architecture():
    ref=dict(A=.4,M=.5,C=.6)
    row=dict(A=.405,M=.5,C=.6,CE=1.,step=250,net_corrections=1)
    assert qualifies(row,ref) and not qualifies({**row,'step':0},ref)
    assert not qualifies({**row,'net_corrections':0},ref)
    assert choose([{**row,'step':500},row],ref)['step']==250
    assert path_counts(None,False)==(8000,0,0)
    assert path_counts('FG4_ROUTE',False)==(16000,0,0)
    assert path_counts('FG4_ROUTE',True)==(16000,286,22)


def test_10_intervention_uses_same_forward_values_and_missing_support():
    x,phi=sample();r=head()(x,phi);out=interventions(r)
    assert set(out)=={'MATCHED_MASS','MATCHED_MASS_AND_DELTA_NORM','ALL_RHO_ONE','ALL_RHO_ZERO'}
    torch.testing.assert_close(out['MATCHED_MASS']['alpha'].sum(-1),r['alpha'].sum(-1))
    assert torch.equal(out['ALL_RHO_ZERO']['embedding'],r['base'])
    assert out['MATCHED_MASS_AND_DELTA_NORM']['paired_defined']
    x['local_valid'].zero_();r=head()(x,phi)
    assert all(torch.equal(v['embedding'],r['base']) for v in interventions(r).values())


def test_11_checkpoint_resume_reproduces_optimizer_and_next_update(tmp_path):
    from static_ovmap.frozen_fc_routing.training import pending_validation
    assert pending_validation(250,[])
    assert pending_validation(2000,[{'step':250}])
    assert not pending_validation(100,[])
    assert not pending_validation(500,[{'step':500}])
    x,phi=sample();m=head();op=optimizer_for(m)
    def step(model,opt):
        opt.zero_grad(set_to_none=True);model(x,phi)['embedding'][0].backward();opt.step()
    step(m,op);p=tmp_path/'last.pt'
    save_checkpoint(p,m,op,seed=17,branch='FG4_ROUTE',step=1,input_identity='same')
    step(m,op);expected=model_digest(m)
    other=head();oo=optimizer_for(other);restore_checkpoint(p,other,oo,input_identity='same');step(other,oo)
    assert model_digest(other)==expected
    with pytest.raises(ValueError):restore_checkpoint(p,other,oo,input_identity='different')


def test_12_export_strictly_loads_architecture_without_MA_dependency(tmp_path):
    cfg=dict(grouping='SURFACE',direct_routing=True);m=initialize(17,cfg);p=tmp_path/'head.pt'
    torch.save(dict(model=m.state_dict(),config=cfg,seed=17,step=250),p)
    restored,_=load_export(p);assert model_digest(restored)==model_digest(m)
    state=torch.load(p,weights_only=False);state['model'].pop('queries');torch.save(state,p)
    with pytest.raises(RuntimeError):load_export(p)
