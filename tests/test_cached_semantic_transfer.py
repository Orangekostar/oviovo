import numpy as np
import pytest
from src.static_ovmap.cached_semantic_transfer import (
    aligned_quality,
    digest,
    donor_choice,
    native_suggestions,
    pair_intersections,
    payload_key,
    support_gate,
)


def test_cached_pre_gate_and_target_filter():
    owners=np.array([1,1,0]);ids=[7,3]
    row={'candidate':0,'methods':['OD_E1_SEMANTIC'],'mask_hash':digest(owners==1),
         'budget_included':True,'aggregate_cosine':[.1,.2],'per_view_cosine':[[.1,.2]],
         'proposed_class':3,'class_id':7,'reason':'insufficient_views'}
    result=native_suggestions([row],owners,[0],ids)
    assert result[0]['proposed_label']==3 and result[0]['encoded_views']==1
    wrong={**row,'methods':['OD_E3_OBJECT_SEMANTIC']}
    assert native_suggestions([wrong],owners,[0],ids)=={}
    with pytest.raises(ValueError):native_suggestions([{**row,'mask_hash':'wrong'}],owners,[0],ids)
    assert native_suggestions([{**row,'aggregate_cosine':[.2,.2],'proposed_class':7}],owners,[0],ids)[0]['tie']
    with pytest.raises(ValueError):native_suggestions([row,{**row,'class_id':3}],owners,[0],ids)


def test_directional_not_iou_and_query_tie_without_class_filter():
    donors=[{'canonical_index':3,'query_id':9,'class_id':4,'intersection':2,'area':4},
            {'canonical_index':4,'query_id':2,'class_id':7,'intersection':2,'area':4}]
    selected,eligible,ties=donor_choice(4,donors)
    assert selected['canonical_index']==4 and selected['iou']==1/3
    assert len(eligible)==2 and ties==[4,3]
    owners=np.array([1,1,2,2,0]);masks=np.array([[1,0,1,0,1]],bool)
    assert pair_intersections(owners,masks,[0,1],[0],2).tolist()==[[1],[1]]


def test_scores_alignment_and_no_double_probability():
    rows=[{'source':'SpaCeFormer','canonical_index':9,'spaceformer_query_id':12,'original_t0_row':1,'original_class_id':4}]
    t={'query_ids':np.array([5,12]),'class_ids':np.array([2,4]),'scores':np.array([.1,.4]),
       'objectness_mask_scores':np.array([.5,.8]),'class_probabilities':np.array([.2,.5])}
    assert aligned_quality(rows,t)=={9:.4}
    with pytest.raises(ValueError):aligned_quality(rows,{**t,'scores':np.array([.1,.2])})


def test_single_gate_duplicate_same_label_zero_and_rejection():
    a={'canonical_index':1,'class_id':4,'iou':.8,'quality':1.,'mask_hash':'a'}
    b={'canonical_index':2,'class_id':5,'iou':.7,'quality':1.,'mask_hash':'b'}
    assert support_gate(3,a,[a])['accepted']
    assert not support_gate(3,a,[a,b])['accepted']
    duplicate={**a,'canonical_index':3,'quality':.5}
    result=support_gate(3,a,[a,b,duplicate])
    assert result['support']==.8/1.5 and len(result['groups'])==2
    assert support_gate(4,a,[a,b])['reason']=='same_label'
    assert support_gate(3,{**a,'quality':0},[{**a,'quality':0}])['reason']=='zero_weight'
    assert payload_key({'run':'a','labels':[4]})!=payload_key({'run':'a','labels':[5]})
    assert payload_key({'run':'a','labels':[4]})!=payload_key({'run':'b','labels':[4]})


def test_fixed_contract_and_released_subset():
    from src.static_ovmap.cached_semantic_transfer import assert_fixed
    base={'kept':[0],'rank_scores':[2.,1.],'rank_scores_serialized':['2.000000','1.000000'],
          'owner_path':'frozen.npy','labels':[4,5]}
    assert_fixed(base,{**base,'labels':[1,5]})
    with pytest.raises(ValueError):assert_fixed(base,{**base,'labels':[1,4]})
    with pytest.raises(ValueError):assert_fixed(base,{**base,'rank_scores_serialized':['1.000000','1.000000']})
    from src.static_ovmap.candidate_semantics import subset_transition
    assert subset_transition(1,4,[4,5])=='entry'
    assert subset_transition(4,1,[4,5])=='exit'


def test_nonfinite_cached_aggregate_abstains_without_json_failure():
    owners=np.array([1])
    row={'candidate':0,'methods':['OD_E1_SEMANTIC'],'mask_hash':digest(owners==1),
         'budget_included':True,'aggregate_cosine':[float('nan'),.2],'per_view_cosine':[[.1,.2]]}
    result=native_suggestions([row],owners,[0],[1,2])[0]
    assert result['proposed_label'] is None and result['reason']=='invalid_aggregate'
    import json
    json.dumps(result,allow_nan=False)


def test_condition_manifest_distinguishes_proposed_accepted_and_changed_receivers():
    from src.static_ovmap import cached_semantic_transfer

    condition_receiver_sets = getattr(cached_semantic_transfer, 'condition_receiver_sets', None)
    assert condition_receiver_sets is not None, 'condition manifests need explicit receiver sets'
    rows = [
        {
            'receiver': 0,
            'old_class': 1,
            'native': {'proposed_label': 2},
            'selected_donor': {'class_id': 3},
            'gate': {'accepted': True},
            'final_labels': {
                'ST_A_NATIVE_TOP1': 2,
                'ST_B_SF_TRANSFER': 3,
                'ST_C_SF_SUPPORT_GATE': 3,
            },
        },
        {
            'receiver': 1,
            'old_class': 4,
            'native': {'proposed_label': 4},
            'selected_donor': {'class_id': 4},
            'gate': {'accepted': True},
            'final_labels': {
                'ST_A_NATIVE_TOP1': 4,
                'ST_B_SF_TRANSFER': 4,
                'ST_C_SF_SUPPORT_GATE': 4,
            },
        },
        {
            'receiver': 2,
            'old_class': 5,
            'native': {'proposed_label': None},
            'selected_donor': {'class_id': 6},
            'gate': {'accepted': False},
            'final_labels': {
                'ST_A_NATIVE_TOP1': 5,
                'ST_B_SF_TRANSFER': 6,
                'ST_C_SF_SUPPORT_GATE': 5,
            },
        },
    ]

    assert condition_receiver_sets('ST_A_NATIVE_TOP1', rows) == {
        'proposed_receivers': [0, 1],
        'proposed_change_receivers': [0],
        'accepted_receivers': [0, 1],
        'accepted_change_receivers': [0],
        'changed_receivers': [0],
    }
    assert condition_receiver_sets('ST_B_SF_TRANSFER', rows) == {
        'proposed_receivers': [0, 1, 2],
        'proposed_change_receivers': [0, 2],
        'accepted_receivers': [0, 1, 2],
        'accepted_change_receivers': [0, 2],
        'changed_receivers': [0, 2],
    }
    assert condition_receiver_sets('ST_C_SF_SUPPORT_GATE', rows) == {
        'proposed_receivers': [0, 1, 2],
        'proposed_change_receivers': [0, 2],
        'accepted_receivers': [0, 1],
        'accepted_change_receivers': [0],
        'changed_receivers': [0],
    }
