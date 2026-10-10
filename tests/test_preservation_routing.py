"""Focused production-kernel groups; real CUDA/integration receipts supplement these."""
import copy
import numpy as np
import torch
from torch.nn import functional as F


def inputs():
    torch.manual_seed(7)
    return dict(raw=torch.randn(2,1536,3,4),projected=torch.randn(2,768,3,4),
                masks=torch.rand(2,1,3,4),fc_vectors=F.normalize(torch.randn(2,768),dim=-1),
                view_valid=torch.ones(2,dtype=torch.bool),frame_ids=torch.tensor([3,8]),
                local_raw=torch.randn(2,4,1536),local_unit=F.normalize(torch.randn(2,4,768),dim=-1),
                metadata=torch.randn(2,4,8),local_valid=torch.ones(2,4,dtype=torch.bool))


def test_residual_identity_and_first_gradient():
    from static_ovmap.preservation_routing.models import RHead,teacher
    x=inputs(); h=RHead(True); z=h(x,lambda v:v[...,:768])['embedding']
    torch.testing.assert_close(z,teacher(x),atol=1e-6,rtol=1e-5)
    z[0].backward()
    assert sum(float(p.grad.abs().sum()) for p in h.ma.parameters() if p.grad is not None)>0
    assert all(p.grad is None and not p.requires_grad for p in h.initial.parameters())
    h.train(); assert not h.initial.training


def test_preservation_wrong_has_no_gradient():
    from static_ovmap.preservation_routing.losses import preservation
    z=F.normalize(torch.randn(768),dim=0).requires_grad_(); t=F.normalize(torch.randn(768),dim=0)
    loss=preservation(z,z,t,False); loss.backward(); assert torch.count_nonzero(z.grad)==0
    z.grad=None; preservation(z,z,t,True).backward(); assert z.grad.abs().sum()>0


def test_absolute_null_and_group_count():
    from static_ovmap.preservation_routing.models import absolute_route
    for j in (1,8,64):
        a,n=absolute_route(torch.zeros(4,j),torch.ones(j))
        torch.testing.assert_close(a.sum(-1),torch.full((4,),.5))
        az,nz=absolute_route(torch.zeros(4,j),torch.zeros(j))
        assert torch.count_nonzero(az)==0; torch.testing.assert_close(nz,torch.ones(4))
    e=torch.tensor([[1.,-1.]])
    a,n=absolute_route(e,torch.ones(2)); b,nb=absolute_route(e,torch.tensor([.1,1.]))
    assert b[0,0]<a[0,0] and nb[0]>n[0] and b.sum()<a.sum()


def test_local_initialization_masking_and_permutation():
    from static_ovmap.preservation_routing.models import GHead,RHead
    x=inputs(); base=RHead(True); h=GHead(base,'SURFACE',True)
    phi=lambda v:v[...,:768]
    z=h(x,phi)['embedding']; bz=base(x,phi)['embedding']
    torch.testing.assert_close(z,bz); z[0].backward()
    assert h.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in h.base.parameters())
    with torch.no_grad():h.output.weight.normal_(0,.001)
    actual=h(x,phi);z=actual['embedding']; xp=copy.deepcopy(x)
    with torch.no_grad():h.quality[-1].bias.add_(3.)
    torch.testing.assert_close(h(x,phi)['quality_weights'],actual['quality_weights'],atol=1e-6,rtol=1e-5)
    with torch.no_grad():h.quality[-1].bias.sub_(3.)
    for k in ('local_raw','local_unit','metadata','local_valid'):xp[k]=xp[k][:,[2,0,3,1]]
    torch.testing.assert_close(h(xp,phi)['embedding'],z,atol=1e-6,rtol=1e-5)
    xp['local_valid'][:]=False; xp['local_raw'][:]=float('nan');xp['metadata'][:]=float('nan')
    xp['local_unit'][:]=float('nan'); torch.testing.assert_close(h(xp,phi)['embedding'],bz)
    assert h.quality[-1].weight.data_ptr()!=h.membership[-1].weight.data_ptr()


def test_membership_unknown_and_correspondence():
    from static_ovmap.preservation_routing.losses import auxiliary
    h=torch.randn(2,3,128,requires_grad=True);m=torch.randn(2,3,requires_grad=True)
    r=dict(membership_logits=m,local_hidden=h,local_valid=torch.ones(2,3,dtype=torch.bool),
           view_indices=torch.arange(2),frame_ids=torch.tensor([1,2]))
    loss,c=auxiliary([r],[torch.tensor([[1,0,-1],[1,0,-1]])],True,True)
    assert c['positive_tokens']==2 and c['negative_tokens']==2 and c['unknown_tokens']==2
    assert c['correspondence_pairs']==1
    loss.backward(); assert torch.count_nonzero(m.grad[:,2])==0 and m.grad[:,:2].abs().sum()>0


def test_full200_gate_and_tolerant_order():
    from static_ovmap.preservation_routing.selection import qualifies,choose
    fc=dict(A=.5,M=.5,C=.6)
    rows=[dict(A=.505,M=.5,C=.6,CE=3.,step=250,net_corrections=1),
          dict(A=.505+1e-12,M=.5,C=.6,CE=2.,step=500,net_corrections=1)]
    assert qualifies(rows[0],fc);assert choose(rows,fc)['step']==500
    assert not qualifies(dict(rows[0],step=0),fc)
    assert not qualifies(dict(rows[0],C=.599),fc)
    assert not qualifies(dict(rows[0],net_corrections=0),fc)


def test_equal_original_family_and_class_metrics():
    from static_ovmap.preservation_routing.selection import summarize
    def row(key,fam,cls,p):
        return dict(key=key,family=fam,class_id=cls,predictions=p,CE=[1.,2.,3.,4.],NLL=[1.]*4)
    rows=[row('a','x',1,[1,1,1,1]),row('b','x',2,[0,0,0,0]),row('c','y',1,[0,0,0,0])]
    s=summarize(rows)
    assert s['A']==.25 and s['C']==.25 and s['M']==.25
    assert s['original_objects']==3 and s['variant_records']==12


def test_fixed_draws_and_optimizer_resume(tmp_path):
    from static_ovmap.learned_object_readout.training import draw_schedule,optimizer_for,save_checkpoint,restore_checkpoint
    from static_ovmap.preservation_routing.models import RHead
    from static_ovmap.preservation_routing.training import initialize
    objects=[dict(base=True,class_id=1),dict(base=False,class_id=2)]
    d=draw_schedule(objects,17);assert np.all(d[:,0]==0)
    a=initialize(17,False);b=initialize(17,True)
    assert all(torch.equal(v,b.ma.state_dict()[k]) for k,v in a.ma.state_dict().items())
    opt=optimizer_for(a);p=tmp_path/'last.pt'
    save_checkpoint(p,a,opt,seed=17,branch='R0',step=100,input_identity='fixed')
    expected=torch.rand(3);restore_checkpoint(p,a,opt,input_identity='fixed');torch.testing.assert_close(torch.rand(3),expected)

def test_real_registry_requires_ordered_pool_inputs():
    from static_ovmap.preservation_routing.evaluation import required_coverage
    import pytest
    cohorts={'replica8':['a','b']};methods=['m1','m2']
    rows=[dict(scene=s,method=m,status='COMPLETE',evaluation_identity=s+m) for s in ('a','b') for m in methods]
    pools={'replica8':{m:dict(ordered_scenes=['a','b'],ordered_inputs=['a'+m,'b'+m]) for m in methods}}
    assert required_coverage(cohorts,methods,rows,pools)==(4,2)
    pools['replica8']['m1']['ordered_inputs'].reverse()
    with pytest.raises(ValueError):required_coverage(cohorts,methods,rows,pools)
    from static_ovmap.preservation_routing.scorer_compat import ReleasedNumpy
    values=np.array([[1001,2001,2002,0]],np.int32)
    np.testing.assert_array_equal(ReleasedNumpy.in1d(values//1000,[1,2]),[True,True,True,False])
    np.testing.assert_array_equal(ReleasedNumpy.in1d(values//1000,[1,2],invert=True),[False,False,False,True])

def test_confirmation_never_accepts_negative_foundation(tmp_path,monkeypatch):
    import pytest
    from static_ovmap.preservation_routing.holdout2 import require_transfer
    from static_ovmap.preservation_routing.common import write
    b={'output_root':str(tmp_path)}
    with pytest.raises(FileNotFoundError):require_transfer(b)
    write(tmp_path/'predictor_lock.json',dict(status='COMPLETE_NO_2D_FOUNDATION',activated_full_path=False))
    with pytest.raises(ValueError):require_transfer(b)
    from static_ovmap.preservation_routing.runner import all_status
    for seed in (17,29):write(tmp_path/f'R_nomination_seed{seed}.json',dict(Rstar='PR_R3_RESIDUAL_KEEP'))
    write(tmp_path/'result_store.json',dict(status='PARTIAL_CONFIRMATION_BLOCKED'))
    assert all_status(tmp_path,0)==('PARTIAL_CONFIRMATION_BLOCKED',1)
    write(tmp_path/'result_store.json',dict(status='SCIENCE_COMPLETE'))
    assert all_status(tmp_path,0)==('SCIENCE_COMPLETE',0)
    assert all_status(tmp_path,1)==('PARTIAL_CONFIRMATION_BLOCKED',1)
    write(tmp_path/'R_nomination_seed29.json',dict(Rstar=None))
    write(tmp_path/'result_store.json',dict(status='COMPLETE_2D_NOT_REPEATED'))
    assert all_status(tmp_path,0)==('COMPLETE_2D_NOT_REPEATED',0)
    write(tmp_path/'R_nomination_seed17.json',dict(Rstar=None))
    write(tmp_path/'result_store.json',dict(status='COMPLETE_NO_2D_FOUNDATION'))
    assert all_status(tmp_path,0)==('COMPLETE_NO_2D_FOUNDATION',0)
    from static_ovmap.preservation_routing import runner,reporting
    from static_ovmap.preservation_routing.common import verified
    monkeypatch.setattr(runner,'fixed_spec',lambda path:{'phases':['report','all']})
    monkeypatch.setattr(runner,'bind',lambda *args,**kwargs:dict(output_root=str(tmp_path),identity='bound'))
    monkeypatch.setattr(reporting,'report',lambda binding:dict(status='COMPLETE_NO_2D_FOUNDATION',source_binding='bound'))
    assert runner.main(['--spec','unused.json','--parent-root','parent','--output-root',str(tmp_path),'--phase','report','--resume'])==0
    phase=verified(tmp_path/'phases/report.json')
    assert phase['source_binding']=='bound' and phase['status']=='COMPLETE_NO_2D_FOUNDATION'
    assert phase['stage_elapsed_seconds']>=0
    assert verified(tmp_path/'execution_last.json')['exit_code']==0
