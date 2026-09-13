import numpy as np


def test_released_object_outcomes_preserve_duplicate_score_owner_and_exact_threshold():
    from src.static_ovmap.attribution_objects import released_object_outcomes
    trace={'states':[{'overlap_threshold':.7500000000000002}], 'events':[
        {'event':'first_match','overlap_threshold':.7500000000000002,'gt_id':1000,'candidate_file':'a','confidence':.6,'overlap':.8},
        {'event':'duplicate','overlap_threshold':.7500000000000002,'gt_id':1000,'candidate_file':'b','tp_score_owner':'b','fp_score_owner':'a','tp_score':.9,'fp_score':.6,'overlap':.9},
        {'event':'hard_fn','overlap_threshold':.7500000000000002,'gt_id':1001}]}
    out=released_object_outcomes(trace,.75)
    assert out['threshold']==.7500000000000002
    assert out['matched']['1000']['first_match_candidate']=='a'
    assert out['matched']['1000']['tp_score_owner']=='b'
    assert out['hard_fn_gt_ids']==[1001]


def test_borrowing_requires_unique_geometry_correspondence_and_records_invalid_class_exit():
    from src.static_ovmap.attribution_objects import borrowing_diagnosis
    # First candidate covers two equally large objects at IoU 0.5 each: ambiguous at 0.25.
    intersections=np.array([[100,100],[100,0],[50,0]])
    areas=np.array([200,100,50]); gt_sizes=np.array([100,100]); gt_ids=np.array([1000,2000])
    records=[{'borrowed_from':'ovi','original_class_id':1,'final_class_id':2,'kept':True},
             {'borrowed_from':'ovi','original_class_id':1,'final_class_id':9,'kept':False},
             {'borrowed_from':'ovi','original_class_id':2,'final_class_id':1,'kept':True}]
    result=borrowing_diagnosis(records,intersections,areas,gt_ids,gt_sizes,[1,2],[.25,.5])
    assert result[0]['thresholds']['0.25']['label_outcome']=='ambiguous'
    assert result[1]['instance_class_transition']=='exit'
    assert result[1]['thresholds']['0.5']['label_outcome']=='right_to_wrong'
    assert result[2]['thresholds']['0.5']['label_outcome']=='unmatched'  # strict > .5


def test_candidate_error_flags_are_nonexclusive_and_use_actual_fp_events():
    from src.static_ovmap.attribution_objects import candidate_event_diagnostics
    events={'events':[
        {'event':'ignore_test','candidate_file':'/x/candidate_0000.npy','counted_fp':True,'overlap_threshold':.5},
        {'event':'ignore_test','candidate_file':'/x/candidate_0001.npy','counted_fp':True,'overlap_threshold':.5},
        {'event':'ignore_test','candidate_file':'/x/candidate_0002.npy','counted_fp':False,'overlap_threshold':.5}]}
    rows=candidate_event_diagnostics(events,np.array([[40,40],[0,0],[0,0]]),np.array([100,100,100]),
        np.array([1,1,1]),np.array([1000,1001]),np.array([100,100]),[1])
    assert 'poor_localization_diagnostic' in rows[0]['flags']
    assert 'multiple_evaluable_GT_positive_intersections' in rows[0]['flags']
    assert 'background_diagnostic' in rows[1]['flags']
    assert 'evaluator_ignored' in rows[2]['flags']
    assert 'counted_unmatched_fp' not in rows[2]['flags']
