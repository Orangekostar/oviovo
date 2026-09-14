import numpy as np
import pytest
from scipy.sparse import csr_matrix


def test_shared_zbuffer_camera_z_ties_unknown_and_depth_abstention():
    from src.static_ovmap.ownership_evidence import project_shared_visibility
    xyz=np.array([[0,0,1],[0,0,1],[0,0,2],[1,0,1],[2,0,1],[3,0,1],[0,0,-1]],float)
    camera={'w':4,'h':1,'fx':1.,'fy':1.,'cx':0.,'cy':0.}
    depth=np.array([[1,1,0,1.06]]);masks=np.array([[7,0,8,9]])
    r=project_shared_visibility(xyz,np.eye(4),camera,depth,masks,chunk=2)
    assert r['point_ids'].tolist()==[0]
    assert r['mask_ids'].tolist()==[7]
    assert r['zbuffer_points'].tolist()==[0,3,4,5]
    assert r['depth_consistent_pixels']==2


def test_pose_selection_uses_every_previous_representative_and_original_schedule():
    from src.static_ovmap.ownership_evidence import select_observation_frames
    poses=np.repeat(np.eye(4)[None],5,axis=0);poses[:,0,3]=[0,.06,.001,.12,.18]
    selected,record=select_observation_frames([0,1,2,3,4],poses,max_frames=3)
    assert selected==[0,3,4]
    assert record['pose_redundant']==[2]


def test_atoms_refine_membership_and_disconnect_without_parent_lock():
    from src.static_ovmap.local_ownership import build_membership_atoms,solve_local_unaries,validate_source_owners
    xyz=np.array([[.001,0,0],[.002,0,0],[.019,.019,0],[.003,0,0],[.05,0,0]])
    masks=np.array([[1,1,1,1,0],[1,1,1,0,0]],bool)
    a=build_membership_atoms(xyz,masks,np.array([0,1]),np.array([1,1,1,1,0]))
    assert a['point_to_atom'][0]==a['point_to_atom'][1]
    assert len(set(a['point_to_atom'][[0,2,3]]))==3
    assert a['point_to_atom'][4]>=0
    n=len(a['candidates']);views=np.full(n,2);scores=np.where(a['candidates']==1,.9,.5)
    u=solve_local_unaries(a,views,scores)
    owners=u['labels'][a['point_to_atom']]+1
    assert owners.tolist()==[2,2,2,1,0]
    assert validate_source_owners(masks,np.array([0,1]),owners)['coverage_exact']
    with pytest.raises(ValueError):validate_source_owners(masks,np.array([0,1]),np.ones(5,dtype=int))


def test_leave_atom_out_removes_self_evidence_and_frame_local_ids():
    from src.static_ovmap.ownership_evidence import frame_local_evidence
    # Candidate0: atom0(2 px mask7)+atom1(20 px mask7); candidate1 only atom0.
    a={'indptr':np.array([0,2,3]),'candidates':np.array([0,1,0]),'counts':np.array([2,20])}
    hist=csr_matrix(np.array([[2,0],[20,0]],dtype=np.int64))
    r=frame_local_evidence(a,hist,np.array([7,99]),candidate_count=2)
    assert r['usable'].tolist()==[True,False,False]
    assert r['r'][0]==1 and r['p'][0]==1 and r['matched_mask'][0]==7
    assert r['reason'][1]!=0
    r2=frame_local_evidence(a,hist,np.array([91,7]),candidate_count=2)
    assert r2['matched_mask'][0]==91


def test_unary_feasibility_fallback_ties_and_missing_state():
    from src.static_ovmap.local_ownership import solve_local_unaries
    a={'indptr':np.array([0,2,4,6]),'candidates':np.array([0,1,0,1,0,1]),'fallback':np.array([0,0,1]),'counts':np.ones(3,dtype=int)}
    u=solve_local_unaries(a,np.array([0,1,2,2,2,2]),np.array([np.nan,.99,.5,.9,.9,.5]))
    assert u['labels'].tolist()==[0,1,0]
    assert u['feasible'].tolist()==[True,False,True,True,True,True]
    # Equal actual cost: fallback2 is retained even though candidate0 sorts first.
    b={'indptr':np.array([0,2]),'candidates':np.array([0,2]),'fallback':np.array([2]),'counts':np.array([1])}
    assert solve_local_unaries(b,np.array([2,2]),np.array([.5,.5]),beta=0)['labels'].tolist()==[2]


def test_spatial_graph_abstains_at_boundaries_missing_neighbors_and_descends():
    from src.static_ovmap.local_ownership import build_evidence_graph,solve_spatial_ownership
    a={'xyz':np.array([[0,0,0],[.01,0,0],[1,0,0]]),'counts':np.ones(3,dtype=int),
       'indptr':np.array([0,2,4,5]),'candidates':np.array([0,1,0,1,2]),'fallback':np.array([0,0,2])}
    same=csr_matrix(np.array([[1,0],[1,0],[0,1]]))
    different=csr_matrix(np.array([[1,0],[0,1],[0,1]]))
    g=build_evidence_graph(a,[same,same]);assert g['edges'].tolist()==[[0,1]] and g['weights'][0]==pytest.approx(1,abs=1e-15)
    assert len(build_evidence_graph(a,[same,different])['edges'])==0
    assert len(build_evidence_graph(a,[same])['edges'])==0
    u={'labels':np.array([0,1,2]),'costs':np.array([.5,.51,.51,.5,.5]),'feasible':np.ones(5,bool)}
    r=solve_spatial_ownership(a,u,g)
    assert r['labels'].tolist()==[1,1,2]
    assert all(x>=y for x,y in zip(r['energy'],r['energy'][1:]))
    assert u['labels'].tolist()==[0,1,2]


def test_family_validation_accepts_local_but_keeps_global_parity_and_rejects_unknown():
    from src.static_ovmap.local_ownership import validate_assignment,method_family,cache_identity
    masks=np.ones((2,3),bool);kept=np.array([0,1]);local=np.array([1,2,1]);near=np.array([1,2,0]);matched=np.array([1,1,0],bool)
    result=validate_assignment('local_evidence',masks,kept,local,near,matched)
    assert result['projected_owners'].tolist()==[2,1,0]
    with pytest.raises(ValueError):validate_assignment('global_priority',masks,kept,local,near,matched,priorities=np.array([2.,1.]),groups=np.array([0,0]))
    with pytest.raises(ValueError):method_family('LO_FAKE')
    assert cache_identity({'evidence':'a','method':'LO_U00_LOCAL'})!=cache_identity({'evidence':'b','method':'LO_U00_LOCAL'})
    assert cache_identity({'evidence':'a','method':'LO_U00_LOCAL'})!=cache_identity({'evidence':'a','method':'LO_U00_SPATIAL'})


def test_candidate_manifest_parity_checks_masks_classes_and_serialized_scores(tmp_path):
    from scripts.evaluation.evaluate_static_local_ownership import candidate_manifest_parity
    mask=tmp_path/'candidate_0000.npy';np.save(mask,np.array([True,False]))
    manifest=tmp_path/'old.txt';manifest.write_text('candidate_0000.npy 4 2.000000\n')
    result=candidate_manifest_parity(manifest,{0:mask},[4],['2.000000'],[0])
    assert result['exact']
    with pytest.raises(ValueError):candidate_manifest_parity(manifest,{0:mask},[4],['1.000000'],[0])
    other=tmp_path/'other.npy';np.save(other,np.array([False,True]))
    with pytest.raises(ValueError):candidate_manifest_parity(manifest,{0:other},[4],['2.000000'],[0])


def test_fixed_regions_keep_unmatched_fn_and_same_class_owner_changes():
    from src.static_ovmap.attribution_regions import regional_confusions,region_transitions
    gt=np.array([1,1,2]);before=np.array([1,1,0]);after=np.array([1,2,0]);regions=np.array([34,35,-1])
    c=regional_confusions(gt,after,regions,[1,2]);assert np.array(c['global_confusion']).tolist()==[[0,0,0],[0,1,1],[1,0,0]]
    assert np.array_equal(sum(np.array(x['confusion']) for x in c['regions'].values()),c['global_confusion'])
    t=region_transitions(gt,before,after,np.array([1,1,0]),np.array([2,2,0]),regions,[1,2])
    assert t['34']['same_class_owner_changes']==1 and t['35']['correct_to_wrong']==1


def test_coincident_centroid_missing_self_still_caps_neighbors():
    from src.static_ovmap.local_ownership import build_evidence_graph
    a={'xyz':np.zeros((3,3)),'counts':np.ones(3,dtype=int)}
    hist=csr_matrix(np.ones((3,1),dtype=int))
    graph=build_evidence_graph(a,[hist,hist],neighbors=1)
    # The query can return two other zero-distance points when self is absent.
    # One directed choice per row yields two undirected edges for this fixture.
    assert len(graph['edges'])==2
