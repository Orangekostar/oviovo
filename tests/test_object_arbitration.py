import numpy as np
from src.static_ovmap.object_arbitration import group_agreement, tentative_action, propose_actions


def test_group_unmatched_penalty_and_unknown():
    assert group_agreement(np.array([1,1,2,2]), np.array([7,7,7,7]), 1) == 0.5
    assert group_agreement(np.zeros(4,int), np.ones(4,int), 1) == 0
    assert group_agreement(np.ones(4,int), np.zeros(4,int), 1) is None


def test_atomic_rollback_and_protection():
    masks = np.array([[1,1,1,1,0,0], [0,0,0,0,1,1], [1,1,0,0,0,0], [0,0,1,1,1,1]],bool)
    before = np.array([1,1,1,1,2,2])
    action = {'parent':0,'children':[2,3]}
    owner, reason, retention = tentative_action(before,masks,action,{2:1.,3:.9},.85)
    assert owner is None and reason == 'support_retention'
    assert np.array_equal(before,[1,1,1,1,2,2])
    masks[3,4:] = False
    owner, reason, _ = tentative_action(before,masks,action,{2:1.,3:.9},.85)
    assert reason == 'valid' and np.array_equal(owner,[3,3,4,4,2,2])


def test_relations_distinguish_children_and_alternative():
    masks=np.array([[1,1,1,1], [1,1,1,1], [1,1,0,0], [0,0,1,1]],bool)
    params={'replace_bidirectional_coverage_min':.7,'child_inside_parent_min':.9,
            'child_union_parent_coverage_min':.8,'child_pair_overlap_over_smaller_max':.1,
            'shortlist_children_per_parent_max':6,'children_per_parent':[2,3],
            'add_outside_qualified_ovi_fraction_min':.8,'shortlisted_actions_per_run_max':128}
    actions,_=propose_actions(masks,[0],[1,2,3],params)
    assert any(a['type']=='REPLACE_1_TO_1' and a['children']==[1] for a in actions)
    assert any(a['type']=='REPLACE_1_TO_GROUP' and a['children']==[2,3] for a in actions)
    assert all(a['parent'] in (None,0) for a in actions)


def test_query_rows_are_not_canonical_rows_and_unknown_is_retained():
    from src.static_ovmap.object_arbitration import query_quality,validate_final
    records=[{'source':'SpaCeFormer','canonical_index':70,'spaceformer_query_id':8,'original_t0_row':1},
             {'source':'SpaCeFormer','canonical_index':64,'spaceformer_query_id':19,'original_t0_row':0}]
    assert query_quality(records,[19,8],[.4,.8])=={70:.8,64:.4}
    masks=np.array([[1,1,1,1],[1,1,1,0]],bool)
    after,reason,_=tentative_action(np.ones(4,int),masks,{'parent':0,'children':[1]},{1:1.},.85)
    assert reason=='valid' and after.tolist()==[2,2,2,0]
    validate_final(after,masks,[1])
    assert len(after)==masks.shape[1]
