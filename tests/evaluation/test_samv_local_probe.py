"""Twelve production boundaries for the fixed four-map segmentation probe."""

from types import SimpleNamespace

import numpy as np
import pytest


def test_auto_prompts_largest_component_padded_interior_distinct():
    from static_ovmap.samv_local_probe.geometry import interior_points
    mask=np.zeros((14,22),bool);mask[1:8,1:8]=True;mask[1:8,12:19]=True
    points=interior_points(mask)
    assert points[0]==(4,4) and len(points)==3 and len(set(points))==3
    assert all(x<8 for x,y in points)
    assert interior_points(np.ones((1,20),bool))==[]
    assert interior_points(np.ones((3,3),bool))==[(1,1)]


def test_fixed_view_window_keeps_absent_context_and_sorts_true_anchor():
    from static_ovmap.samv_local_probe.geometry import choose_window,transform_points
    frames=[]
    for fid,x in [(50,0.),(10,.2),(40,.4),(20,.6),(60,.8),(30,1.)]:
        pose=np.eye(4);pose[0,3]=x;frames.append({'frame_id':fid,'pose_c2w':pose.tolist()})
    ids,anchor,second=choose_window(frames,[50,10],50,6)
    assert ids==[10,20,30,40,50,60] and ids[anchor]==50 and ids[second]==10
    np.testing.assert_allclose(transform_points([[300,170]],(680,1200)),[[256,256]])
    with pytest.raises(ValueError):choose_window(frames[:3],[50,10],50,6)


def test_target_quotas_uncertainty_shortage_and_no_gt_inputs():
    from static_ovmap.samv_local_probe.geometry import select_targets
    rows=[{'owner':i,'pool':'INCUMBENT','eligible':True,'margin':i/100,'maximum_area':500} for i in range(1,8)]
    rows += [{'owner':10,'pool':'RECOVERED','eligible':True,'margin':None,'maximum_area':300}]
    rows += [{'owner':11,'pool':'INCUMBENT','eligible':False,'margin':0.,'maximum_area':1000}]
    selected=select_targets(rows)
    assert [r['owner'] for r in selected]==[1,2,3,4,5,6,7,10]
    changed=[{**r,'unrelated_diagnostic_GT':-999} for r in rows]
    assert [r['owner'] for r in select_targets(changed)]==[r['owner'] for r in selected]


def test_panorama_split_before_resize_avoids_cross_view_seam():
    from static_ovmap.samv_local_probe.geometry import split_panorama
    panorama=np.concatenate([np.full((2,3),-7.),np.full((2,3),9.)],axis=1)
    tiles=split_panorama(panorama,2)
    assert tiles.shape==(2,2,3) and np.all(tiles[0]==-7) and np.all(tiles[1]==9)
    with pytest.raises(ValueError):split_panorama(np.zeros((2,7)),2)


def test_visibility_normalization_one_vote_per_view_invalid_minus_one():
    from static_ovmap.samv_local_probe.lifting import count_view_votes
    source=[np.array([[0,0,0,0,1,-1]]),np.array([[0,1,1,1,1,-1]])]
    valid=[s>=0 for s in source]
    masks=[np.array([[1,1,0,0,0,1]],bool),np.array([[0,1,1,1,1,1]],bool)]
    n,k=count_view_votes(source,valid,masks,3)
    np.testing.assert_array_equal(n,[2,2,0]);np.testing.assert_array_equal(k,[1,1,0])


def test_simultaneous_conflict_ties_keep_parent_unknown_kept_prompt_forced():
    from static_ovmap.samv_local_probe.lifting import arbitrate
    base=np.array([9,1,1,0,0,9]);editable=np.array([1,1,1,1,1,0],bool)
    counts={1:(np.array([3,3,3,3,0,3]),np.array([2,0,0,3,0,3])),
            2:(np.array([3,3,0,3,0,3]),np.array([2,0,0,2,0,3]))}
    domains={i:np.ones(6,bool) for i in counts}
    out,audit=arbitrate(base,[2,1],counts,domains,editable,{2:1})
    np.testing.assert_array_equal(out,[9,0,1,1,0,9])
    assert audit['tie_rows']==1
    reversed_out,_=arbitrate(base,[1,2],counts,domains,editable,{2:1})
    np.testing.assert_array_equal(out,reversed_out)


def test_mesh_boundary_hops_protect_unselected_interiors_allow_unowned():
    from static_ovmap.samv_local_probe.geometry import boundary_band,edit_domain
    owners=np.array([1,2,2,2,2,2,0]);faces=np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
    band=boundary_band(owners,faces,hops=0)
    assert band[0] and band[1] and band[2] and not band[3]
    xyz=np.column_stack([np.arange(7)*.04,np.zeros((7,2))])
    domains,editable,protected=edit_domain(xyz,faces,owners,[1],padding=.3,hops=0)
    assert domains[1].all() and editable[6] and protected[3] and not editable[3]
    with pytest.raises(ValueError):boundary_band(owners,np.empty((0,3),int))


def test_local_output_uniform_classes_raw_zero_edits_no_new_ids():
    from static_ovmap.samv_local_probe.outputs import build
    from static_ovmap.module_validation.evaluation import GeometryIdentity,PredictionPayload
    from static_ovmap.module_validation.scannet_study import native_ranks
    owners=np.array([1,1,2,2,0,0]);labels={1:7,2:9};sem=np.array([7,7,9,9,0,0])
    nearest=np.arange(6);matched=np.ones(6,bool)
    base=PredictionPayload('G1','COMBO','fixture',GeometryIdentity('a'*64,'b'*64,'c'*64,'fixed',6),owners,sem,
                           native_ranks(owners,labels,nearest,matched),{},{});base.lock()
    inputs=SimpleNamespace(g1=base,nearest=nearest,matched=matched,valid_ids=[7,9],raw=np.zeros(6,int))
    new=np.array([1,1,2,2,1,0]);editable=np.array([0,0,0,0,1,0],bool)
    result,audit=build(inputs,'SV02_SAMV_GEOM',new,{},editable,{0:1})
    assert result.locked and result.owner_ids[4]==1 and result.semantic_labels[4]==7 and result.semantic_labels[5]==0
    assert audit['old_raw_zero_ownership_edits']==1
    with pytest.raises(ValueError):build(inputs,'SV02_SAMV_GEOM',np.array([1,1,2,2,3,0]),{},editable,{0:1})


def test_paired_semantics_requires_both_sources_and_both_views():
    from static_ovmap.samv_local_probe.readout import paired_labels
    old=[np.array([.2,.3]),np.array([.3,.5])];new=[np.array([.8,.1]),None]
    assert paired_labels(7,[7,9],old,new)==(7,7,False)
    new[1]=np.array([.8,.1])
    assert paired_labels(7,[7,9],old,new)==(9,7,True)
    assert paired_labels(7,[7,9],[np.array([.5,.509])]*2,new)==(7,7,True)


def test_registry_key_uses_entire_actual_partition_same_owner_ids():
    from static_ovmap.samv_local_probe.evaluation import partition_registry_key
    assert partition_registry_key(np.array([1,1,2,2]))!=partition_registry_key(np.array([1,2,1,2]))


def test_complete_subset_pooling_rejects_missing_or_duplicate_scenes():
    from static_ovmap.samv_local_probe.evaluation import ordered_rows
    rows=[{'scene':'b','method':'SV00_G1','status':'COMPLETE'},{'scene':'a','method':'SV00_G1','status':'COMPLETE'}]
    assert [r['scene'] for r in ordered_rows(rows,['a','b'],'SV00_G1')]==['a','b']
    with pytest.raises(ValueError):ordered_rows(rows[:1],['a','b'],'SV00_G1')
    with pytest.raises(ValueError):ordered_rows(rows+[rows[0]],['a','b'],'SV00_G1')


def test_resume_whole_window_identity_and_selection_tolerant_ties():
    from static_ovmap.samv_local_probe.common import request_key
    from static_ovmap.samv_local_probe.selection import select
    query={'frames':[{'frame_id':1,'RGB_sha256':'a'},{'frame_id':2,'RGB_sha256':'b'}],'points':[[2,3]],'profile':6}
    assert request_key(query,'SAMV','weights-a')!=request_key({**query,'frames':list(reversed(query['frames']))},'SAMV','weights-a')
    assert request_key(query,'SAMV','weights-a')!=request_key(query,'SAMV','weights-b')
    candidates=['SV02_SAMV_GEOM','SV04_SAMV_FC','SV05_COMBINED']
    methods=['SV00_G1','REF_D2',*candidates]
    pools={c:{m:{'metrics':dict.fromkeys(['apall','ap50','ap25','miou','macc'],.2)} for m in methods} for c in ['replica_probe2','cf_probe2']}
    assert select(pools)['selected']=='SV00_G1'
    pools['cf_probe2']['REF_D2']['metrics']['apall']=.19
    pools['cf_probe2']['SV05_COMBINED']['metrics']['apall']+=5e-11
    assert select(pools)['selected']=='SV02_SAMV_GEOM'
    pools['replica_probe2']['SV02_SAMV_GEOM']['metrics']['macc']-=2e-10
    assert select(pools)['selected']=='SV04_SAMV_FC'
