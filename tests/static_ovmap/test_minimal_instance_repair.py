"""New production boundaries; prescribed cases, no historical repository sweep."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


def module(name):
    return importlib.import_module('static_ovmap.minimal_instance_repair.' + name)


def frame(fid, translation=0., degrees=0.):
    a = np.deg2rad(degrees)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    pose[0, 3] = translation
    return {'frame_id': fid, 'pose_c2w': pose.tolist()}


def test_pose_bins_use_first_representative_and_floor_subsampling():
    obs = module('observations')
    selected, bins = obs.pose_banks([frame(2, .30), frame(0), frame(1, .15), frame(3, .30, 16)])
    assert [x['frame_id'] for x in selected] == [0, 2, 3]
    assert [x['bank'] for x in selected] == ['proposal', 'verification', 'proposal']
    assert bins[1]['representative_frame_id'] == 0
    selected, _ = obs.pose_banks([frame(i, i) for i in range(34)])
    assert [x['frame_id'] for x in selected] == [int(np.floor(k*33/31)) for k in range(32)]


def test_barycentric_ties_preserve_original_source_rows():
    obs = module('observations')
    faces = np.array([[7, 3, 9], [4, 6, 2]], np.int64)
    uv = np.array([[.5, 0.], [0., 1.]])
    np.testing.assert_array_equal(obs.representative_rows(faces, uv), [3, 2])
    # Tiny valid negative values stay unmodified: clipping could introduce a tie.
    assert obs.representative_rows(np.array([[7, 3, 9]]), np.array([[.50000001, -.00000003]]))[0] == 7
    with pytest.raises(ValueError, match='barycentric'):
        obs.representative_rows(faces[:1], np.array([[np.nan, 0.]]))


def test_observer_uses_nearest_full_mesh_hit_and_original_camera_depth():
    obs = module('observations')
    xyz = np.array([[-2,-2,1], [2,-2,1], [0,2,1], [-2,-2,2], [2,-2,2], [0,2,2]], np.float64)
    projector = obs.SourceRowProjector(xyz, np.array([[0,1,2], [3,4,5]], np.int64))
    K = np.eye(3)
    rows, valid = projector.project(K, np.eye(4), np.array([[2]], np.float32))
    assert not valid[0,0] and rows[0,0] == -1  # Front surface occludes the candidate behind it.
    rows, valid = projector.project(K, np.eye(4), np.array([[1]], np.float32))
    assert valid[0,0] and rows[0,0] == 2


def test_frame_local_votes_exclude_unknown_but_include_neutral():
    obs = module('observations')
    strong = obs.unit_evidence(np.ones(30, bool), np.full(30, 7, np.int32))
    weak = obs.unit_evidence(np.ones(30, bool), np.r_[np.full(20, 9), np.full(10, 10)])
    unknown = obs.unit_evidence(np.ones(30, bool), np.zeros(30, np.int32))
    assert obs.pair_vote(strong, strong) == (1, 0)
    assert obs.pair_vote(strong, weak) == (0, 0)
    assert obs.pair_vote(strong, unknown) is None
    # Local IDs can change between frames; within-frame equal IDs still vote SAME.
    other = obs.unit_evidence(np.ones(30, bool), np.full(30, 99))
    rates = obs.pair_rates([(strong,strong), (other,other), (strong,weak), (strong,unknown)])
    assert rates == {'n': 3, 'same': 2/3, 'separate': 0.}
    assert obs.pair_rates([(strong,unknown)]) == {'n': 0, 'same': None, 'separate': None}


def test_vertex_area_is_physical_and_isolated_vertices_are_zero():
    support = module('support_units')
    xyz = np.array([[0,0,0], [2,0,0], [0,3,0], [9,9,9]], np.float64)
    np.testing.assert_allclose(support.vertex_areas(xyz, np.array([[0,1,2], [0,0,1]])), [1,1,1,0])
    with pytest.raises(ValueError, match='area'):
        support.vertex_areas(xyz, np.array([[0,0,1]]))


def test_missing_panoptic_is_a_block_and_present_unknown_is_valid(tmp_path):
    obs = module('observations')
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex
    from static_ovmap.module_validation.assets import sha256_file
    import cv2
    missing = tmp_path / 'missing.png'
    frame_info = {'panoptic_path':missing.name,'panoptic_sha256':'0'*64,'source_paths':{}}
    with pytest.raises(FileNotFoundError, match='panoptic'):
        obs.load_panoptic(tmp_path, frame_info, ConsumptionIndex(), {})
    present = tmp_path / 'panoptic.png'
    cv2.imwrite(str(present), np.zeros((3,4),np.uint16))
    frame_info.update(panoptic_path=present.name,panoptic_sha256=sha256_file(present))
    raster, identity = obs.load_panoptic(tmp_path, frame_info, ConsumptionIndex(), {})
    np.testing.assert_array_equal(raster,np.zeros((3,4),np.uint16))
    assert identity['sha256'] == frame_info['panoptic_sha256']


def test_group_library_requires_clique_but_does_not_filter_separation():
    proposals = module('proposals')
    support = module('support_units')
    from static_ovmap.module_validation.contracts import canonical_digest
    units = {name:support.SupportUnit(name,name[0],int(name[2:]),np.array([i]),canonical_digest(name),1.,1)
             for i,name in enumerate(['C:1','C:2','I:3'])}
    neighbors = {'C:1':[{'unit':'C:2','distance':.01},{'unit':'I:3','distance':.02}]}
    pairs = {('C:1','C:2'):{'n':2,'same':.6,'separate':.4},
             ('C:1','I:3'):{'n':2,'same':.6,'separate':0.}}
    library = proposals.build_library(units,['C:1'],neighbors,pairs,max_parent_id=10)
    assert {tuple(x['units']) for x in library} == {('C:1','C:2'),('C:1','I:3')}
    weak = next(x for x in library if x['host'] is None)
    assert weak['proposal_score'] < 0 and weak['fresh_owner'] >= 11
    pairs[('C:2','I:3')] = {'n':2,'same':1.,'separate':0.}
    library = proposals.build_library(units,['C:1'],neighbors,pairs,max_parent_id=10)
    assert ('C:1','C:2','I:3') in {tuple(x['units']) for x in library}


def test_verified_repair_keeps_shared_priority_and_abstains_on_missing_views():
    verify = module('verification')
    groups = [
        {'units':['C:1','I:3'],'digest':'a','edit_cost':.5,'proposal_score':.9,'host':3},
        {'units':['C:2','I:3'],'digest':'b','edit_cost':.5,'proposal_score':.8,'host':3},
        {'units':['C:4','C:5'],'digest':'c','edit_cost':.5,'proposal_score':.7,'host':None},
    ]
    rates = {('C:1','I:3'):{'n':2,'same':.7,'separate':0.},
             ('C:2','I:3'):{'n':2,'same':1.,'separate':0.},
             ('C:4','C:5'):{'n':1,'same':1.,'separate':0.}}
    selected, ledger = verify.select_groups(groups,rates,verified=True)
    assert [x['digest'] for x in selected] == ['a']
    assert ledger[1]['status'] == 'KEEP_CONFLICT'
    assert ledger[2]['status'] == 'KEEP_INSUFFICIENT_VERIFICATION'
    direct,_ = verify.select_groups(groups,rates,verified=False)
    assert [x['digest'] for x in direct] == ['a','c']


def test_boundary_reread_deduplicates_regions_and_keeps_disagreement():
    reread = module('reread')
    mask = np.ones((20,20),bool)
    regions,missing = reread.recognition_masks(mask,np.ones(mask.shape,np.uint16))
    assert [x['type'] for x in regions] == ['FULL','CORE']
    assert any(x['reason']=='DUPLICATE_MASK' for x in missing)
    full = [
        {'frame_id':1,'bank':'proposal','type':'FULL','mask_digest':'a','scores':[.30,.40]},
        {'frame_id':2,'bank':'verification','type':'FULL','mask_digest':'b','scores':[.50,.48]},
        {'frame_id':1,'bank':'proposal','type':'CORE','mask_digest':'c','scores':[.30,.42]},
    ]
    decision = reread.reread_decision(1,[1,2],full)
    assert decision['IR06_class'] == 2 and decision['IR07_class'] == 1
    assert not decision['tests']['both_full_argmax_agree']
    assert reread.reread_decision(1,[1,2],[])['IR06_class'] == 1


def test_all_five_gate_rejects_single_decrease_and_strict_ap_equality():
    select = module('selection')
    from pathlib import Path
    import json
    spec = json.loads((Path(__file__).resolve().parents[2]/'configs/static_ovmap/minimal_instance_repair_v1.json').read_text())
    values = {m:.20 for m in ['apall','ap50','ap25','miou','macc']}
    pools = {cohort:{'IR00_D2':values.copy(),'IR01_G1':values.copy(),'IR07_BOUNDARY_STABLE':values.copy()}
             for cohort in spec['cohorts']}
    result = select.select_methods(pools,spec)
    assert result['selected'] == 'IR01_G1'
    pools['scannet_cf18']['IR07_BOUNDARY_STABLE']['apall'] = .202
    assert select.select_methods(pools,spec)['selected'] == 'IR07_BOUNDARY_STABLE'
    pools['replica8']['IR07_BOUNDARY_STABLE']['macc'] -= .0001
    assert select.select_methods(pools,spec)['selected'] == 'IR01_G1'


def partition_fixture():
    from types import SimpleNamespace
    from static_ovmap.module_validation.evaluation import GeometryIdentity,PredictionPayload
    from static_ovmap.module_validation.scannet_study import native_ranks
    from static_ovmap.composition_study.object_evidence import owner_labels
    support = module('support_units')
    xyz = np.column_stack((np.arange(8),np.zeros(8),np.zeros(8)))
    faces = np.array([[0,1,2]])
    geometry = GeometryIdentity('1'*64,'2'*64,'3'*64,'projection',8)
    b = np.array([1,1,0,0,0,0,0,0])
    g = np.array([1,1,2,2,3,3,0,0])
    nearest,matched = np.arange(8),np.ones(8,bool)
    def payload(name,owners,semantics):
        labels = {int(o):int(semantics[np.flatnonzero(owners==o)[0]]) for o in np.unique(owners) if o>0}
        p = PredictionPayload(name,'COMBO','fixture',geometry,owners,semantics,native_ranks(owners,labels,nearest,matched),{})
        p.lock()
        return p
    d2,g1 = payload('D2',b,np.where(b>0,1,0)),payload('G1',g,np.where(g>0,1,0))
    units = {}
    for name,rows in [('I:1',[0,1]),('C:2',[2,3]),('C:3',[4,5])]:
        units[name] = support.SupportUnit(name,name[0],int(name[2:]),np.array(rows,np.int64),name,1.,2)
    unit_set = support.UnitSet(units,['C:2','C:3'],{},np.ones(8),'mesh')
    return SimpleNamespace(scene='fixture',xyz=xyz,faces=faces,raw=np.array([1,1,2,2,3,3,0,0]),
                           d2=d2,g1=g1,units=unit_set,nearest=nearest,matched=matched,valid_ids=[1,2])


def test_partition_keep_attachment_union_and_original_incumbent_anchor():
    outputs = module('outputs')
    inputs = partition_fixture()
    keep,ledger = outputs.construct_partition(inputs,'IR05_VERIFIED_REPAIR',[])
    assert keep.prediction_key == inputs.g1.prediction_key
    attached = {'units':['C:2','I:1'],'host':1,'output_owner':1,'fresh_owner':None,'digest':'attach'}
    payload,ledger = outputs.construct_partition(inputs,'IR05_VERIFIED_REPAIR',[attached])
    np.testing.assert_array_equal(payload.owner_ids,[1,1,1,1,3,3,0,0])
    assert ledger['moved_incumbent_rows'] == 0
    combined,_ = outputs.construct_partition(inputs,'IR08_COMBINATION',[attached],relabels={1:2})
    np.testing.assert_array_equal(combined.semantic_labels,[2,2,2,2,1,1,0,0])
    union = {'units':['C:2','C:3'],'host':None,'output_owner':10,'fresh_owner':10,'digest':'union'}
    unknown,_ = outputs.construct_partition(inputs,'IR04_DIRECT_GROUP',[union],union_classes={})
    assert unknown.prediction_key == inputs.g1.prediction_key
    payload,_ = outputs.construct_partition(inputs,'IR04_DIRECT_GROUP',[union],union_classes={'union':2})
    np.testing.assert_array_equal(payload.owner_ids,[1,1,10,10,10,10,0,0])
    assert all(len(np.unique(payload.semantic_labels[payload.owner_ids==o]))==1 for o in [1,10])
    with pytest.raises(ValueError,match='disjoint'):
        outputs.construct_partition(inputs,'IR05_VERIFIED_REPAIR',[attached,union],union_classes={'union':2})


def test_diagnostic_keeps_full_raw_clipping_and_eligible_matching_separate():
    diag = module('diagnostics')
    # Same geometry/projection: full raw can recover an object that its clipped
    # residual cannot; unavailable/ignored GT does not enter either matching set.
    gt = np.array([1001]*6+[2001]*4+[3001]*2)
    eligible = {1001:{'label_id':1},2001:{'label_id':2}}
    full = diag.support_overlap(np.arange(6),np.arange(12),np.ones(12,bool),gt,eligible,minimum=1)
    residual = diag.support_overlap(np.arange(4),np.arange(12),np.ones(12,bool),gt,eligible,minimum=1)
    assert full['best']['iou'] == 1 and residual['best']['iou'] == 2/3
    rows = [{'name':'a',**residual},{'name':'b',**residual},
            {'name':'ignored',**diag.support_overlap(np.arange(10,12),np.arange(12),np.ones(12,bool),gt,eligible,minimum=1)}]
    assert diag.maximum_matches(rows,eligible,.5)['count'] == 1
    # Released matches use strictly greater IoU, rather than >= threshold.
    assert diag.maximum_matches(rows,eligible,2/3)['count'] == 0
    assert diag.maximum_matches([{'name':'full',**full}],eligible,.9)['count'] == 1


def test_partition_evaluator_keys_actual_membership_and_all_positive_owners(tmp_path,monkeypatch):
    evaluation = module('evaluation')
    outputs = module('outputs')
    inputs = partition_fixture()
    inputs.data = {'dataset':'Replica','config':'bound_config'}
    inputs.sources = {'N':{'objects':{'1':{'available':True}}}}
    inputs.index = object()
    captured = []
    def factory(evidence,root,index):
        captured.append((evidence,root))
        return evidence
    monkeypatch.setattr(evaluation,'SceneEvaluator',factory)
    monkeypatch.setattr(evaluation,'read',lambda path:{'original_target':True})
    binding = {'path_map':{},'reference':{}}
    first = evaluation.partition_evaluator(inputs,inputs.g1,binding,tmp_path)
    union = {'units':['C:2','C:3'],'host':None,'output_owner':10,'fresh_owner':10,'digest':'union'}
    changed,_ = outputs.construct_partition(inputs,'IR04_DIRECT_GROUP',[union],union_classes={'union':2})
    second = evaluation.partition_evaluator(inputs,changed,binding,tmp_path)
    assert captured[0][1] != captured[1][1]
    assert set(first.sources['N0']['objects']) == {'1','2','3'}
    assert set(second.sources['N0']['objects']) == {'1','10'}
    assert not second.sources['N0']['objects']['10']['available']


def test_recognition_view_choice_is_mask_area_then_frame_id_and_bank():
    plan = module('recognition_plan')
    masks = [np.ones((1,120),bool),np.ones((10,12),bool),np.ones((12,10),bool)]
    choices = [{'frame_id':3,'bank':'proposal','mask':masks[0]},
               {'frame_id':2,'bank':'proposal','mask':masks[1]},
               {'frame_id':1,'bank':'verification','mask':masks[2]}]
    chosen = plan.choose_view(choices)
    assert chosen['frame_id'] == 1  # Thin mask has enough pixels but invalid bbox.
    assert plan.choose_view(choices,bank='proposal')['frame_id'] == 2
    assert plan.choose_view(choices,bank='missing') is None
    a = plan.region_identity('support-a',chosen,'FULL','model','text',[24,42],[192,336],'ANYUP_FULL_EQUAL_COSINE')
    b = plan.region_identity('support-b',chosen,'FULL','model','text',[24,42],[192,336],'ANYUP_FULL_EQUAL_COSINE')
    assert a != b  # Same numeric owner/view can never alias a different support.
    descriptor = {'frame_id':chosen['frame_id'],'mask_digest':chosen['mask_digest'],
                  'mask':{'path':'a-real-packed-mask.npz','sha256':'verified'}}
    # Acquisition receives a file descriptor, not an ndarray; an existing mask
    # digest must avoid evaluating the array fallback eagerly.
    assert plan.region_identity('support-a',descriptor,'FULL','model','text',[24,42],[192,336],
                                 'ANYUP_FULL_EQUAL_COSINE') == a


def test_cold_partition_builder_does_not_hash_parent_outputs(monkeypatch):
    outputs = module('outputs')
    inputs = partition_fixture()
    from static_ovmap.module_validation.evaluation import PredictionPayload
    def forbidden(self):
        raise AssertionError('parent prediction hashing entered the cold timer')
    monkeypatch.setattr(PredictionPayload,'prediction_key',property(forbidden))
    payload,_ = outputs.construct_partition(inputs,'IR07_BOUNDARY_STABLE',[],lock=False,
                                            parent_prediction_keys={'G1':'prebound-g1','D2':'prebound-d2'})
    assert not payload.locked
    assert payload.metadata['parent_prediction_key']=='prebound-g1'


def test_resume_invalidates_only_changed_scene_descendants_and_preserves_history(tmp_path):
    resume = module('resume')
    files = ['recognition/office1/plan.json','recognition/office1/decisions.json',
             'recognition/office1/frames/0.json','recognition/room0/plan.json',
             'observations/office1/receipt.json','proposals/office1/receipt.json']
    for name in files:
        p = tmp_path/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(name)
    binding = {'output_root':str(tmp_path),'scenes':{'office1':{'cohort':'replica8'}}}
    receipt = resume.invalidate_descendants(binding,'office1','recognition_plan','changed input')
    assert not (tmp_path/'recognition/office1/plan.json').exists()
    assert not (tmp_path/'recognition/office1/decisions.json').exists()
    assert (tmp_path/'recognition/office1/frames/0.json').exists()  # Content leaves recheck their own identities.
    assert (tmp_path/'recognition/room0/plan.json').exists()
    assert (tmp_path/'observations/office1/receipt.json').exists()
    assert (tmp_path/'proposals/office1/receipt.json').exists()
    assert all(Path(item['archived']).read_text()==item['relative_path'] for item in receipt['archived'])


def test_new_timing_series_has_four_unique_arms_and_reverses_both_orders():
    timing = module('timing')
    import json
    spec = json.loads((Path(__file__).resolve().parents[2]/'configs/static_ovmap/minimal_instance_repair_v1.json').read_text())
    plan = timing.make_plan(spec,'IR03_EVIDENCE_ATTACH')
    assert plan['methods']==['IR01_G1','IR03_EVIDENCE_ATTACH','IR08_COMBINATION','IR05_VERIFIED_REPAIR']
    assert len(plan['calls'])==64
    assert len({(r['repeat'],r['scene'],r['method']) for r in plan['calls']})==64
    assert plan['calls'][0]=={'repeat':1,'scene':'office0','method':'IR01_G1'}
    assert plan['calls'][32]=={'repeat':2,'scene':'room2','method':'IR05_VERIFIED_REPAIR'}
    assert timing.make_plan(spec,'IR01_G1')['methods']==['IR01_G1','IR08_COMBINATION','IR05_VERIFIED_REPAIR','IR07_BOUNDARY_STABLE']


def test_postlock_partition_iou_counts_all_support_points_and_original_projection():
    diag = module('output_diagnostics')
    from types import SimpleNamespace
    payload = SimpleNamespace(owner_ids=np.array([1,1,2,2]))
    inputs = SimpleNamespace(nearest=np.array([0,1,1,2,3,0]),matched=np.array([True]*5+[False]))
    gt = np.array([1001,1001,0,2001,2001,1001])
    rows = diag.describe_partition(payload,inputs,gt,{1001:{'label_id':1},2001:{'label_id':2}},1)
    assert rows[0]['evaluation_points']==3  # Void overlap stays in the union.
    assert rows[0]['best']['iou']==.5  # Unmatched GT point stays in GT size.
    assert rows[1]['best']['iou']==1
    assert module('diagnostics').maximum_matches(rows,{1001:{},2001:{}},.5)['count']==1


def test_postlock_semantic_errors_separate_matchable_from_insufficient_geometry():
    diag = module('output_diagnostics')
    original = [{'owner':1,'geometrically_matchable50':True,'best':{'class':2}},
                {'owner':2,'geometrically_matchable50':False,'best':{'class':2}}]
    counts,ledger = diag.semantic_transition_counts({1:2,2:1},{1:1,2:2},original)
    assert counts['matchable_wrong_to_right']==1
    assert counts['geometry_insufficient_right_to_wrong']==1
    assert counts.get('matchable_right_to_wrong',0)==0
    assert len(ledger)==2
