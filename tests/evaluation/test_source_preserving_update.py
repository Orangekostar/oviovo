"""Protocol behaviors: each assertion catches a semantic change, not source text."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import softmax


def math_module():
    from static_ovmap.source_preserving_update import decisions
    return decisions


def sources():
    return {k:np.log(v) for k,v in {'N':[.1,.2,.7], 'Q':[.3,.4,.3], 'F':[.6,.3,.1]}.items()}


def test_d2_grouping_missing_source_normalization():
    d=math_module()
    p,q,f,w=d.source_components(sources(),dict.fromkeys('NQF',1.))
    np.testing.assert_allclose(p,[.4,.3,.3],atol=1e-15)
    assert w==.5
    p,q,f,w=d.source_components({'N':None,'Q':None,'F':np.log([.6,.3,.1])},{'F':1.})
    np.testing.assert_allclose(p,[.6,.3,.1],atol=1e-15)
    assert q is None and w==1.
    assert d.source_components(dict.fromkeys('NQF'),{})[0] is None


def test_same_view_pair_identity_and_cosine_then_softmax():
    from static_ovmap.source_preserving_update.evidence import paired_means
    rows=[{'frame_id':10,'mask_digest':'a','scores':[0.,4.]},
          {'frame_id':20,'mask_digest':'b','scores':[1.,0.]}]
    coarse=[{**r,'scores':[2.,1.]} for r in rows]
    a,c=paired_means(rows,coarse,[2,11],[2,11])
    np.testing.assert_array_equal(a,[.5,2.]);np.testing.assert_array_equal(c,[2.,1.])
    assert not np.allclose(softmax(a),np.mean([softmax(r['scores']) for r in rows],axis=0))
    with pytest.raises(ValueError):paired_means(rows,coarse,[2,11],[11,2])
    with pytest.raises(ValueError):paired_means(rows,[coarse[0],{**coarse[1],'mask_digest':'c'}],[2,11],[2,11])


def test_f_replace_blend_and_equal_evidence_identity():
    d=math_module();s=sources();t=dict.fromkeys('NQF',1.);a=np.log([.2,.5,.3]);c=np.log([.3,.3,.4])
    np.testing.assert_allclose(d.update_probability('SU04_F_REPLACE',s,t,a,c),[.2,.4,.4],atol=1e-15)
    np.testing.assert_allclose(d.update_probability('SU05_F_BLEND',s,t,a,c),[.3,.35,.35],atol=1e-15)
    np.testing.assert_allclose(d.update_probability('SU06_F_COARSE',s,t,a,c),[.325,.3,.375],atol=1e-15)
    np.testing.assert_allclose(d.update_probability('SU05_F_BLEND',s,t,s['F'],c),[.4,.3,.3],atol=1e-15)


def test_equal_a_mass_global_and_f_only_equality():
    d=math_module();s=sources();t=dict.fromkeys('NQF',1.);a=np.log([.2,.5,.3]);c=np.log([.3,.3,.4])
    np.testing.assert_allclose(d.update_probability('SU07_GLOBAL_BLEND',s,t,a,c),[.35,.35,.3],atol=1e-15)
    f={'F':s['F']}
    np.testing.assert_array_equal(d.update_probability('SU05_F_BLEND',f,t,a,c),d.update_probability('SU07_GLOBAL_BLEND',f,t,a,c))


def test_residual_identity_saturation_and_same_view_coarse():
    d=math_module();s=sources();t=dict.fromkeys('NQF',1.);a=np.array([1.,2.,3.])
    np.testing.assert_array_equal(d.update_probability('SU08_PAIRED_DELTA',s,t,a,a),d.source_components(s,t)[0])
    np.testing.assert_array_equal(d.update_probability('SU08_PAIRED_DELTA',s,t,a,a-2.),d.source_components(s,t)[0])
    f={'F':np.array([-2000.,0.,-4000.])};p=d.update_probability('SU08_PAIRED_DELTA',f,{'F':1.},[5000.,0.,0.],[0.,0.,0.])
    assert np.isfinite(p).all() and int(p.argmax())==0


def test_common_domain_keeps_every_arm_if_any_pair_unavailable():
    from static_ovmap.source_preserving_update.evidence import common_domain
    row={'selected':True,'p0':[.4,.3,.3],'scores':sources(),'protected_raw_zero_count':0,
         'full_views':[{'region_id':'a'},{'region_id':'b'}]}
    complete={'a':{'available':True},'b':{'available':True}}
    assert common_domain(row,complete)==(True,[])
    assert common_domain(row,{'a':{'available':True},'b':{'available':False,'reason':'EMPTY_AREA_MASK_SUPPORT'}})[0] is False
    assert common_domain({**row,'protected_raw_zero_count':1},complete)[0] is False
    assert common_domain({**row,'scores':{'N':row['scores']['N'],'F':None}},complete)[0] is False
    assert common_domain({**row,'full_views':[]},complete)[0] is False
    assert common_domain({**row,'full_views':row['full_views'][:1]},complete)[0] is True
    with pytest.raises(ValueError):common_domain(row,{'a':{'available':True}})


def test_replay_real_parent_decisions_and_matched_keep():
    from static_ovmap.source_preserving_update.evidence import replay_decision
    from static_ovmap.source_preserving_update.decisions import decide
    root=Path(__file__).resolve().parents[2]
    cfg=json.loads((root/'configs/static_ovmap/minimal_instance_repair_v1.json').read_text())['reread']
    parent=json.loads((root/'artifacts/static_ovmap/minimal_instance_repair_v1/scenes/office1/semantic_decisions.json').read_text())
    # Actual saved region scores, rather than a synthetic stability success.
    ids=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]
    for old in parent['incumbents'].values():
        actual=replay_decision(old['old_class'],ids,old['observations'],cfg)
        assert actual['IR06_class']==old['IR06_class'] and actual['IR07_class']==old['IR07_class']
        if 'aggregate_scores' in old:np.testing.assert_array_equal(actual['aggregate_scores'],old['aggregate_scores'])
    item=decide('SU02_HARD_MATCHED',[2,11,27],2,sources(),dict.fromkeys('NQF',1.),common=False,hard_class=11)
    assert item['applied_class']==2 and not item['changed']


def toy_inputs():
    from static_ovmap.module_validation.evaluation import GeometryIdentity,PredictionPayload
    from static_ovmap.module_validation.scannet_study import native_ranks
    owners=np.repeat([1,2,3],[200,120,280]);labels={1:1,2:2,3:2};sem=np.repeat([1,2,2],[200,120,280])
    d2=owners.copy();d2[320:]=0;d2sem=sem.copy();d2sem[320:]=0
    geom=GeometryIdentity('a'*64,'b'*64,'c'*64,'d'*64,600);nearest=np.arange(600);matched=np.ones(600,bool)
    g1=PredictionPayload('G1','COMBO','toy',geom,owners,sem,native_ranks(owners,labels,nearest,matched),{},{});g1.lock()
    base=PredictionPayload('D2','COMBO','toy',geom,d2,d2sem,native_ranks(d2,{1:1,2:2},nearest,matched),{},{});base.lock()
    raw=owners.copy();raw[:10]=0
    units=SimpleNamespace(units={f'I:{o}':SimpleNamespace(support_hash=str(o)) for o in [1,2]},vertex_weights=np.ones(600))
    return SimpleNamespace(g1=g1,d2=base,raw=raw,valid_ids=[1,2],units=units,nearest=nearest,matched=matched)


def test_whole_owner_raw_zero_cancellation_and_recovered_parity():
    from static_ovmap.source_preserving_update.outputs import relabel_g1
    inputs=toy_inputs();p,audit=relabel_g1(inputs,'SU04_F_REPLACE',{1:2,2:1},eligible={2})
    np.testing.assert_array_equal(p.owner_ids,inputs.g1.owner_ids)
    np.testing.assert_array_equal(p.semantic_labels[:200],1)
    np.testing.assert_array_equal(p.semantic_labels[200:320],1)
    np.testing.assert_array_equal(p.semantic_labels[320:],2)
    assert len(audit['canceled_class_transitions'])==1


def test_current_class_ranks_and_content_alias():
    from static_ovmap.source_preserving_update.outputs import relabel_g1,same_output
    inputs=toy_inputs();p,audit=relabel_g1(inputs,'SU04_F_REPLACE',{2:1},eligible={2})
    assert dict(p.instance_ranks)[2]==.6 and dict(p.instance_ranks)[3]==1.
    assert not same_output(p,inputs.g1)
    keep,_=relabel_g1(inputs,'SU05_F_BLEND',{},eligible=set())
    assert same_output(keep,inputs.g1)


def test_unrounded_gate_fallback_and_fixed_ties():
    from static_ovmap.source_preserving_update.selection import select
    root=Path(__file__).resolve().parents[2];cfg=json.loads((root/'configs/static_ovmap/source_preserving_update_v1.json').read_text())['selection']
    methods=['SU00_D2','SU01_G1',*cfg['candidates']]
    pools={cohort:{m:{'metrics':dict.fromkeys(['apall','ap50','ap25','miou','macc'],.2)} for m in methods} for cohort in ['replica8','scannet_cf18']}
    assert select(pools,cfg)['selected']=='SU01_G1'
    for m in ['SU05_F_BLEND','SU06_F_COARSE']:pools['scannet_cf18'][m]['metrics']['apall']=.201
    assert select(pools,cfg)['selected']=='SU06_F_COARSE'
    pools['replica8']['SU06_F_COARSE']['metrics']['macc']-=2e-10
    assert select(pools,cfg)['selected']=='SU05_F_BLEND'
