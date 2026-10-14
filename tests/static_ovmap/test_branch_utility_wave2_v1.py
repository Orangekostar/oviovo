"""Six bounded contract groups; engineering fixtures are not real FC parity."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

P='static_ovmap.branch_utility_wave2_v1'

def modules(*names):
    assert importlib.util.find_spec(P), 'branch utility implementation is missing'
    return [__import__(P+'.'+name,fromlist=['*']) for name in names]

def inputs(n=4):
    torch.manual_seed(50)
    raw=torch.randn(n,5,12)
    return dict(fc_vectors=torch.randn(n,12),view_valid=torch.ones(n,dtype=torch.bool),frame_ids=torch.arange(n)*3+7,local_raw=raw,local_unit=torch.nn.functional.normalize(raw,dim=-1),metadata=torch.ones(n,5,8),local_valid=torch.ones(n,5,dtype=torch.bool))

def head():
    from static_ovmap.frozen_fc_routing.models import FGHead
    torch.manual_seed(10)
    return FGHead(dim=12,hidden=8).eval()

def test_1_seven_tensor_subset_prefix_condition_history():
    s,c=modules('subsets','capture')
    x=inputs(8); y=s.subset_inputs(x,[0,2])
    assert set(y)==set(x)
    for k in x: assert torch.equal(y[k],x[k][[0,2]])
    assert y['frame_ids'].tolist()==[7,13]
    assert s.state_key(y,0)!=s.state_key(y,1)
    with pytest.raises(ValueError): s.validate_inputs(dict(x,target=torch.tensor(1)))
    with pytest.raises(ValueError): s.subset_inputs(x,[0,0])
    calls=[]
    def producer(obj,condition,prefix):
        calls.append((condition,prefix)); return s.subset_inputs(x,list(range(prefix))),None,[]
    z=c.load_prefix(producer,{},2,2)
    assert calls==[(2,2)] and z['frame_ids'].tolist()==[7,10]
    c.assert_history(np.ones((2,12),np.float32),np.ones((2,12),np.float32),[1,2],[1,2],{'A':.5,'M':.2,'C':.1},{'A':.5,'M':.2,'C':.1})
    with pytest.raises(ValueError): c.assert_history(np.ones((2,12)),np.ones((2,12)),[1,2],[2,1],{'A':.5},{'A':.5})

def test_2_compose_crossbase_empty_keep():
    d,=modules('decomposition')
    x=inputs(); h=head(); r=h(x,lambda z:z)
    assert torch.equal(d.compose(r['base'],r),r['embedding'])
    b=torch.nn.functional.normalize(torch.arange(1,13,dtype=torch.float32),dim=0)
    from static_ovmap.frozen_fc_routing.models import correct,unit
    expected=correct(b,r['group_vectors']-b,r['alpha'],.5)[0]
    assert torch.equal(d.compose(b,r),expected)
    assert not torch.allclose(unit(b+r['delta']),expected)
    empty=h(dict(x,local_valid=torch.zeros_like(x['local_valid'])),lambda z:z)
    assert torch.equal(d.compose(b,empty),b)
    m,g,null,j=d.local_statistics(empty)
    assert m==0 and torch.equal(g,torch.zeros_like(b)) and null==1 and j==0
    e=d.effects(1,2,4,7)
    assert e['base']+e['local']==e['total']==6 and e['interaction']==2

def test_3_actions_subset_dedup_short_support_noop():
    a,=modules('action_bank')
    x=inputs(); h=head(); r=h(x,lambda z:z); seen=[]
    def evaluate(z): seen.append(z['frame_ids'].tolist()); return h(z,lambda q:q)
    actions=a.build_actions(x,r,evaluate)
    assert len(actions)==13 and len(seen)==4
    assert [v['kind'] for v in actions]==['KEEP']+['G','L','B']*4
    for i in range(4):
        g,l,b=actions[1+3*i:4+3*i]
        assert l['local'] is b['local'] and g['local'] is r
        assert g['frame_id']==l['frame_id']==b['frame_id']==int(x['frame_ids'][i])
    short=inputs(2); assert len(a.build_actions(short,h(short,lambda z:z),evaluate))==1
    x['local_valid'][0]=False; r=h(x,lambda z:z); seen.clear()
    actions=a.build_actions(x,r,evaluate)
    assert len(seen)==3 and not actions[2]['edited'] and actions[2]['noop_proof']
    assert torch.equal(actions[2]['embedding'],r['embedding'])
    assert actions[1]['edited'] and actions[3]['edited']

def test_4_features_targets_norm_paired_init_and_draws():
    a,f,m,t=modules('action_bank','features','utility_model','training')
    x=inputs();h=head();r=h(x,lambda z:z)
    actions=a.build_actions(x,r,lambda z:h(z,lambda q:q))
    v=f.action_features(x,r,actions[1],0)
    assert v.shape==(24,) and np.isfinite(v).all() and np.array_equal(v[:3],[1,0,0]) and v[3]==.5
    torch.testing.assert_close(x['fc_vectors'],inputs()['fc_vectors'])
    # Label/family sidecars cannot alter target-free feature values.
    np.testing.assert_array_equal(v,f.action_features(dict(x,class_id=7,family='unused'),r,actions[1],0))
    same={k:value.clone() for k,value in x.items()}
    same['local_raw']=same['local_raw'][:1].expand(4,-1,-1).clone()
    same['local_unit']=same['local_unit'][:1].expand(4,-1,-1).clone()
    sr=h(same,lambda z:z);sa=a.build_actions(same,sr,lambda z:h(z,lambda q:q))
    np.testing.assert_allclose(f.action_features(same,sr,sa[1],0)[16:],[1,1,0,1,1,0,0,1],atol=1e-6)
    invalid=dict(x,fc_vectors=x['fc_vectors'].clone());invalid['fc_vectors'][0]=0
    with pytest.raises(ValueError):f.action_features(invalid,r,actions[1],0)
    no=dict(x,local_valid=torch.zeros_like(x['local_valid']))
    rr=h(no,lambda z:z);aa=a.build_actions(no,rr,lambda z:h(z,lambda q:q))
    assert np.array_equal(f.action_features(no,rr,aa[1],0)[16:],np.zeros(8))
    norm=m.Normalizer.fit(np.array([v,v+2]))
    np.testing.assert_allclose(norm.mean,v+1,atol=1e-6)
    assert np.array_equal(norm.transform(v,without_new=True)[16:],np.zeros(8))
    model=m.UtilityRegressor(17); model2=m.UtilityRegressor(17)
    assert sum(p.numel() for p in model.parameters())==1345
    for p,q in zip(model.parameters(),model2.parameters()): assert torch.equal(p,q)
    assert torch.equal(model(torch.ones(3,24)),torch.zeros(3))
    objs=[{'key':'a','class_id':1},{'key':'b','class_id':1},{'key':'c','class_id':2}]
    draws=t.make_draws(objs,17,steps=2,batch=16)
    assert np.array_equal(draws,t.make_draws(objs,17,steps=2,batch=16)) and draws.shape==(2,16,3)
    assert set(draws[:,:,1].flatten())<=set([4,8])
    y=torch.tensor([.1,-.2]); p=torch.tensor([0.,0.])
    loss=t.group_loss(p,y,np.array([0,2,2]))
    assert torch.isclose(loss,torch.nn.functional.smooth_l1_loss(p,y,beta=.1)/2)

def test_5_selection_lock_alias_resume(tmp_path):
    s,t=modules('selection','training')
    actions=[dict(kind='KEEP',frame_id=-1,edited=False),dict(kind='G',frame_id=9,edited=True),dict(kind='L',frame_id=1,edited=True),dict(kind='B',frame_id=0,edited=True)]
    assert s.select_action(actions,[0,.03,.03,.03],0)==1
    assert s.select_action(actions,[0,.03,.03,.03],.03)==0
    assert s.select_action(actions,[0,.04,.05,.01],0,coupled=True)==3
    assert s.select_action(actions,[0,-1,-1,-1],0)==0
    candidate=dict(method='STATIC',base='MEAN',gamma=1,step=0,tau=0,mean_edits=0,A8=.5,M8=.4,C8=.3,A4=.2,net_corrections8=1)
    assert s.compare_candidates(candidate,dict(candidate,gamma=.5))<0
    assert s.promotion(candidate,dict(A8=.495,M8=.405,C8=.305,A4=.205))
    capture,=modules('capture')
    torch.manual_seed(50);vectors=torch.randn(2,12)
    x=inputs(2);x['fc_vectors']=vectors;x['frame_ids']=torch.tensor([7,10])
    from static_ovmap.frozen_fc_routing.models import unit
    # Two-view mean-other cosine is identical; lowest physical frame ID wins.
    assert torch.equal(capture.static_bases(x,[1,1])['COSINE_MEDOID'],unit(vectors[0]))
    locked=s.lock_profiles(tmp_path,{'STATIC':candidate},'STATIC')
    assert s.lock_profiles(tmp_path,{'STATIC':candidate},'STATIC')['identity']==locked['identity']
    with pytest.raises(ValueError): s.lock_profiles(tmp_path,{'STATIC':dict(candidate,tau=.1)},'STATIC')
    model=torch.nn.Linear(2,1);optim=torch.optim.AdamW(model.parameters(),lr=.001)
    file=tmp_path/'resume.pt';t.save_resume(file,model,optim,100,{'data':'fixed','draw_index':100})
    old={k:v.clone() for k,v in model.state_dict().items()}
    model.weight.data.add_(1)
    step,extra=t.load_resume(file,model,optim,{'data':'fixed'})
    assert step==100 and extra['draw_index']==100
    for k,v in model.state_dict().items(): assert torch.equal(v,old[k])
    with pytest.raises(ValueError): t.load_resume(file,model,optim,{'data':'wrong'})

def test_6_native_pool_rank_dynamic_coverage_publication(tmp_path):
    p,e,pub=modules('prediction','evaluation','publication')
    from static_ovmap.paired_evidence_study.residuals import pool,simple_dependence
    source={'N':np.array([.2,.1]),'Q':np.array([.1,.3]),'F':np.array([.3,.1])}
    temperatures={'N':1.,'Q':1.,'F':1.};new=np.array([.1,.4]);old={k:v.copy() for k,v in source.items()}
    mixed=dict(source,F=new);expected=simple_dependence(mixed,temperatures,pool(mixed,temperatures,('N','Q','F')),'PE_D2_GROUPED')
    np.testing.assert_array_equal(p.replace_f(source,new,temperatures),expected)
    for k in source: np.testing.assert_array_equal(source[k],old[k])
    cohorts={'r':['room0'],'s':['scene1','scene2']}; methods=['m17','m29']
    rows=[{'scene':scene,'method':m} for scenes in cohorts.values() for scene in scenes for m in methods]
    assert e.validate_coverage(cohorts,methods,rows)==6
    with pytest.raises(ValueError):e.validate_coverage(cohorts,methods,rows[:-1])
    conveyor,=modules('conveyor')
    owners=np.array([1,2,1,2]);nearest=np.tile([0,2,1,3],100);matched=np.tile([True,True,False,True],100);gt=np.tile([1001,1001,0,2001],100)
    matches=conveyor.scorer_geometry_matches(owners,gt,nearest,matched,[1,2])
    assert matches[1]['GT_class']==1 and matches[1]['IoU']==1 and matches[2]['IoU']==1
    from static_ovmap.m2_reviewer_study.evaluation import official_view
    projected=np.tile([1,1,0,2],100)
    separate=official_view(projected,{1:1,2:2},100);merged=official_view(projected,{1:1,2:1},100)
    assert separate[2]['rank']=='1.000000' and merged[2]['rank']=='0.500000'
    file=tmp_path/'a';file.write_bytes(b'abc')
    assert pub.check_sizes([file],file_limit=3,total_limit=3)==3
    with pytest.raises(ValueError):pub.check_sizes([file],file_limit=2,total_limit=3)
