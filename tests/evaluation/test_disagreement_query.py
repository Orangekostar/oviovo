"""Twelve directed production checks for the frozen 1009 experiment."""
from types import SimpleNamespace

import numpy as np
import pytest


def test_exact_duplicate_triangle_ambiguity_signed_zero_and_area():
    from static_ovmap.disagreement_query.physical_support import canonical_sites
    xyz=np.array([[0.,0,0],[1,0,0],[0,1,0],[-0.,0,0],[1,0,0],[0,1,0]])
    faces=np.array([[0,1,2],[5,4,3],[0,0,0]])
    s=canonical_sites(xyz,faces,np.ones(6,int))
    assert s['audit']['duplicates']==1 and s['audit']['degenerates']==1
    assert len(s['area'])==3 and s['area'].sum()==pytest.approx(.5)
    assert not np.signbit(s['xyz'][s['xyz']==0]).any()
    s=canonical_sites(xyz,faces,np.array([1,1,1,2,1,1]))
    assert s['audit']['ambiguous']==1 and s['audit']['ambiguous_area']==pytest.approx(.5)
    assert len(s['area'])==0


def test_sites_do_not_weld_nearby_or_cross_owner_coordinates():
    from static_ovmap.disagreement_query.physical_support import canonical_sites
    xyz=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,0],[1,0,1e-9],[0,1,1e-9]])
    s=canonical_sites(xyz,np.array([[0,1,2],[3,4,5]]),np.array([1,1,1,2,2,2]))
    assert len(s['area'])==6 and len(s['coordinates'])==5
    assert len(s['owners'][np.all(s['xyz']==[0,0,0],axis=1)])==2
    assert s['audit']['duplicates']==0


def test_quantile_quadrature_conserves_mass_and_input_order():
    from static_ovmap.disagreement_query.physical_support import quadrature
    x=np.column_stack((np.arange(5000),np.zeros(5000),np.zeros(5000)))
    a=np.ones(5000);a[9]=9000
    q,w=quadrature(x,a)
    assert len(q)<=4096 and w.sum()==pytest.approx(a.sum())
    qr,wr=quadrature(x[::-1],a[::-1]);np.testing.assert_array_equal(q,qr);np.testing.assert_array_equal(w,wr)
    small,weights=quadrature(x[:3],a[:3]);np.testing.assert_array_equal(small,x[:3]);np.testing.assert_array_equal(weights,a[:3])


def test_direct_visibility_real_full_bvh_no_clamp_and_full_alignment():
    import open3d as o3d
    from static_ovmap.disagreement_query.physical_support import direct_visibility
    xyz=np.array([[-2,-2,2],[2,-2,2],[2,2,2],[-2,2,2],[-.5,-.5,1],[0,-.5,1],[0,.5,1],[-.5,.5,1]],np.float32)
    faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]],np.uint32)
    mesh=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(xyz),o3d.core.Tensor(faces))
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(mesh)
    frame={'pose_c2w':np.eye(4).tolist(),'intrinsics':[[2,0,2],[0,2,2],[0,0,1]],'image_size_hw':[5,5]}
    points=np.array([[.5,0,2],[-.5,0,2],[2.8,0,2],[0,0,-1]])
    full=np.ones((5,5),bool);full[2,3]=False
    v,o,uv,audit=direct_visibility(points,frame,np.full((5,5),2.),full,scene)
    np.testing.assert_array_equal(v,[True,False,False,False]);assert not o.any()
    assert uv[2,0]==5 and audit['geometric_visible']==1 and audit['mask_supported']==0


def test_predictor_namespace_has_no_evaluation_projection_or_target_labels():
    from static_ovmap.disagreement_query.binding import predictor_namespace
    src=SimpleNamespace(scene='s',xyz=np.zeros((3,3)),faces=np.array([[0,1,2]]),raw=np.ones(3),
        g1=SimpleNamespace(owner_ids=np.ones(3),semantic_labels=np.ones(3),geometry='digest'),
        d2=SimpleNamespace(owner_ids=np.ones(3)),sources={},probabilities={},valid_ids=[1,2],
        data={'temperatures':{'F':1},'FC_physical_model_identity':'m','gt_labels':'forbidden'},
        capture={'frames':[],'identity':'capture'},capture_path='p',index=None,
        nearest=np.arange(3),matched=np.ones(3,bool),targets='forbidden',units='legacy')
    p=predictor_namespace(src)
    assert not {'nearest','matched','targets','units','data'}.intersection(vars(p))
    assert p.temperatures=={'F':1} and p.owner_ids.shape==(3,)


def test_target_inventory_excludes_raw_zero_and_uses_margin_js_order():
    from static_ovmap.disagreement_query.query_plan import target_inventory,select_targets
    p=SimpleNamespace(d2_owner_ids=np.array([1,1,2,2,3,3]),owner_ids=np.array([1,1,2,2,3,3]),
        semantic_labels=np.array([1,1,1,1,1,1]),raw=np.array([1,1,2,0,3,3]),valid_ids=[1,2],
        probabilities={str(i):{'probabilities':[.51,.49]} for i in (1,2,3)},
        temperatures={'N':1.,'Q':1.,'F':1.},sources={n:{'objects':{str(i):{'available':True,'scores':([4.,0.] if n=='N' and i==3 else [0.,0.])} for i in (1,2,3)}} for n in ('N','Q','F')})
    views={i:{10:{'pixels':100,'bbox':[0,0,10,10],'qualified':True},20:{'pixels':100,'bbox':[0,0,10,10],'qualified':True}} for i in (1,2,3)}
    inventory=target_inventory(p,views);chosen=select_targets(inventory)
    assert [r['owner'] for r in chosen]==[3,1]
    assert next(r for r in inventory if r['owner']==2)['reason']=='EXCLUDE_RAW_ZERO_SUPPORT'


def test_candidate_farthest_bank_only_qualified_and_chronological():
    from static_ovmap.disagreement_query.query_plan import candidate_bank
    frames=[]
    for fid,x in [(10,0),(20,.2),(30,.4),(40,50)]:
        pose=np.eye(4);pose[0,3]=x;frames.append({'frame_id':fid,'pose_c2w':pose.tolist()})
    qualified={10:{'pixels':200},20:{'pixels':100},30:{'pixels':150}}
    anchor,bank=candidate_bank(frames,qualified,limit=2)
    assert anchor==10 and bank==[10,30] and 40 not in bank


def test_surface_direction_coverage_and_area_id_ties():
    from static_ovmap.disagreement_query.policies import score_bank
    x=np.array([[1,0,0],[0,1,0],[-1,0,0.]])
    footprint=np.array([[1,0,0],[1,0,0],[0,1,1]],bool)
    args=dict(xyz=x,area=np.ones(3),footprints=footprint,center=np.zeros(3),
        cameras=np.array([[0,0,2],[2,0,0],[0,0,2.]]),pixels=[100,200,150],frame_ids=[10,20,30])
    second,audit=score_bank('COVERAGE',**args);assert second==30
    assert audit['candidates']['20']['score']==0 and audit['candidates']['30']['score']==1
    args['pixels']=[100,150,150];assert score_bank('AREA',**args)[0]==20


def test_quadrants_interest_fallback_and_actual_margin_signs():
    from static_ovmap.disagreement_query.policies import quadrants,anchor_interest
    mask=np.ones((12,12),bool);regions,bbox=quadrants(mask)
    assert len(regions)==4 and all(r['qualified'] for r in regions)
    uv=np.array([[1,1],[8,1],[1,8],[8,8]])
    cos={0:np.array([1.,0.]),1:np.array([0.,1.])}
    interest,audit=anchor_interest(np.ones(4),np.ones(4,bool),uv,bbox,np.array([1.,0.]),cos,0,1,1.)
    np.testing.assert_array_equal(interest,[0,2,0,0]);assert audit['tile_sign_conflict']
    interest,audit=anchor_interest(np.ones(4),np.ones(4,bool),uv,bbox,np.array([1.,0.]),{0:cos[0]},0,1,1.)
    assert not interest.any() and audit['fallback']=='VERIFY_FEWER_THAN_TWO_SUCCESSFUL_TILES'


def test_evidence_whitelist_denies_future_cached_candidate():
    from static_ovmap.disagreement_query.acquisition import EvidenceAccess
    a=EvidenceAccess({'anchor':{'scores':[0,1]},'future':{'scores':[1,0]}},['anchor'])
    assert a.get('anchor')['scores']==[0,1]
    with pytest.raises(ValueError,match='not visible'):a.get('future')


def test_updates_dedup_withdrawal_mass_degeneracies_and_unavailable_keep():
    from static_ovmap.disagreement_query.updates import Evidence,update
    p=np.array([.5,.5]);a=np.array([1.,3.,2.]);ids=[1,2]
    r=Evidence('a',np.array([3.,0]),np.array([1,1,0],bool));s=Evidence('b',np.array([0.,3]),np.array([0,1,1],bool))
    f=lambda rows,kind:update(p,rows,a,kind,1.,ids,1)
    base=f([r,s],'SUPPORT');np.testing.assert_allclose(base['weights'],[2.5,3.5]);assert sum(base['weights'])==6
    np.testing.assert_array_equal(base['probability'],f([s,r,r],'SUPPORT')['probability'])
    np.testing.assert_array_equal(f([r,s][:1],'SUPPORT')['probability'],f([r],'AREA')['probability'])
    same=Evidence('c',s.cosine,r.footprint)
    np.testing.assert_array_equal(f([r,same],'SUPPORT')['probability'],f([r,same],'MEAN')['probability'])
    disjoint=Evidence('d',s.cosine,np.array([0,0,1],bool))
    np.testing.assert_array_equal(f([r,disjoint],'SUPPORT')['probability'],f([r,disjoint],'AREA')['probability'])
    assert f([],'SUPPORT')['label']==1;np.testing.assert_array_equal(f([],'SUPPORT')['probability'],p)
    assert f([r,Evidence('missing',None,None,False,'INVALID_REGION_FEATURE')],'SUPPORT')['label']==1
    with pytest.raises(ValueError,match='contradictory'):f([r,Evidence('a',s.cosine,r.footprint)],'MEAN')


def test_payload_registry_ranks_and_exact_screen_final_gates():
    from static_ovmap.disagreement_query.outputs import build_payload
    from static_ovmap.disagreement_query.selection import screen_gate,final_gate
    from static_ovmap.module_validation.evaluation import GeometryIdentity,PredictionPayload
    from static_ovmap.module_validation.scannet_study import native_ranks
    owners=np.repeat([1,2,3,0],[100,200,400,10]);labels=np.repeat([1,1,2,0],[100,200,400,10]);nearest=np.arange(len(owners));matched=np.ones(len(owners),bool)
    geom=GeometryIdentity(*(['a'*64]*3),'proj',len(owners))
    g1=PredictionPayload('G1','N0','s',geom,owners,labels,native_ranks(owners,{1:1,2:1,3:2},nearest,matched),{});g1.lock()
    d2=SimpleNamespace(owner_ids=np.where(owners==3,0,owners))
    out=build_payload(g1,d2,np.where(owners==3,0,owners),{1:2},{1},nearest,matched,'NEW',{})
    assert dict(out.instance_ranks)=={1:.25,2:1.,3:1.};assert out.locked
    np.testing.assert_array_equal(out.semantic_labels[owners!=1],g1.semantic_labels[owners!=1]);np.testing.assert_array_equal(out.owner_ids,owners)
    with pytest.raises(ValueError):build_payload(g1,d2,owners,{3:1},{3},nearest,matched,'BAD',{})
    metrics=lambda v:{k:v for k in ('apall','ap50','ap25','miou','macc')}
    c={k:metrics(.201) for k in ('replica_probe2','cf_probe2')};b={k:metrics(.2) for k in c}
    assert screen_gate(c,b,b,changed=True,corrections_net=1,gt50_net=0)
    assert not screen_gate(c,b,b,changed=False,corrections_net=1,gt50_net=0)
    c={k:metrics(.201) for k in ('replica8','scannet_cf18')};b={k:metrics(.2) for k in c}
    assert final_gate(c,b,b);c['replica8']['macc']=None;assert not final_gate(c,b,b)
