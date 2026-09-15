import numpy as np
import pytest
from src.static_ovmap.candidate_semantics import native_crops, score_views, semantic_cache_key
from src.static_ovmap.final_instance_ranking import final_source_ranks


def test_native_six_crops_and_mask_cache():
    rgb = np.full((10, 12, 3), 120, np.uint8)
    a = np.zeros((10, 12), bool)
    a[2:8, 2:9] = True
    b = a.copy()
    b[4, 4] = False
    crops = native_crops(rgb, a, (2, 2, 8, 7))
    assert len(crops) == 6
    assert crops[0].size == (6, 5)
    assert semantic_cache_key({'frame': 1}, a) != semantic_cache_key({'frame': 1}, b)
    assert np.asarray(native_crops(rgb, b, (2, 2, 8, 7))[1])[2, 2].sum() == 0


def test_full_cosine_acceptance_and_space_guard():
    text = np.eye(3)
    features = np.array([[0., 1., 0.], [0., 1., 0.]])
    result = score_views(features, [10, 20], text, [1, 2, 3], 1, 'native', 'native')
    assert result['class_id'] == 2
    assert np.asarray(result['per_view_cosine']).shape == (2, 3)
    assert score_views(features[:1], [10], text, [1, 2, 3], 1, 'native', 'native')['class_id'] == 1
    with pytest.raises(ValueError):
        score_views(features, [10, 20], text, [1, 2, 3], 1, 'other', 'native')
    tied = np.array([[1., 1., 0.], [1., 1., 0.]])
    assert score_views(tied, [1, 1], text, [1, 2, 3], 2, 'native', 'native')['class_id'] == 2


def test_rank_uses_source_owner_and_preserves_it():
    owners = np.array([1, 1, 0, 3, 3, 3])
    old = owners.copy()
    assert final_source_ranks(owners, [0, 2, 4]) == [2, 3, 0]
    assert np.array_equal(old, owners)


def test_evaluation_cache_separates_all_interventions():
    import copy
    from scripts.evaluation.evaluate_static_object_decoupling import evaluation_cache_key
    doc={'kept':[0], 'labels':[4], 'rank_scores_serialized':['2.000000'],
         'owner_sha256':'a', 'final_mask_hashes':{'0':'b'}}
    key=evaluation_cache_key('run1','E1','unique',doc,'bank',{})
    assert key!=evaluation_cache_key('run2','E1','unique',doc,'bank',{})
    assert key!=evaluation_cache_key('run1','E2','unique',doc,'bank',{})
    for field,value in [('kept',[]),('labels',[5]),('rank_scores_serialized',['1.000000']),
                        ('owner_sha256','c'),('final_mask_hashes',{'0':'d'})]:
        other=copy.deepcopy(doc);other[field]=value
        assert key!=evaluation_cache_key('run1','E1','unique',other,'bank',{})


def test_semantic_instance_subset_is_not_forced():
    from src.static_ovmap.candidate_semantics import subset_transition
    assert subset_transition(1,4,[4,5])=='entry'
    assert subset_transition(4,1,[4,5])=='exit'
    assert subset_transition(4,5,[4,5])=='unchanged'
