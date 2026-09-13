import numpy as np
import pytest


def pool_fixture():
    from src.static_ovmap.fusion_attribution import build_frozen_candidates
    return build_frozen_candidates(
        np.array([1, 1, 1, 1, 2, 2, 0, 0]),
        {'1': {'class_id': 4, 'selected_query_ids': ['a', 'b']}, '2': None},
        np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]], bool),
        np.array([7, 8]), np.array([.8, .7]), np.array([11, 12]),
        scene_id='room0', prediction_run='first', ovi_readout_id='sha256:readout')


def test_frozen_pool_identity_and_label_nms_isolation():
    from src.static_ovmap.fusion_attribution import apply_label_reuse, apply_fusion_nms
    p = pool_fixture()
    with pytest.raises(ValueError):
        p.masks[0, 0] = False
    original = p.original_labels.copy()
    off, _ = apply_label_reuse(p, enabled=False)
    on, events = apply_label_reuse(p, enabled=True)
    assert off.tolist() == [4, 7, 8]
    assert on.tolist() == [4, 4, 8]
    assert events[1]['borrowed_from'] == p.records[0]['candidate_id']
    assert events[1]['label_actually_changed'] is True
    kept, edges = apply_fusion_nms(p)
    assert kept.tolist() == [0, 2] and edges[1]['suppressed_by'] == 0
    assert np.array_equal(p.original_labels, original)
    assert p.records[0]['source'] == 'OVI'
    assert p.records[0]['ovi_readout_id'] == 'sha256:readout'
    assert len({r['candidate_id'] for r in p.records}) == 3


def test_sf_first_changes_visit_not_export_order():
    from src.static_ovmap.fusion_attribution import apply_fusion_nms
    p = pool_fixture()
    kept, edges = apply_fusion_nms(p, sf_first=True)
    assert kept.tolist() == [1, 2]
    assert edges[0]['suppressed_by'] == 1
    assert edges[1]['nms_visit_rank'] == 0
    all_ids, _ = apply_fusion_nms(p, enabled=False)
    assert all_ids.tolist() == [0, 1, 2]


def test_exact_reuse_and_suppression_boundaries():
    from src.static_ovmap.fusion_attribution import build_frozen_candidates, apply_label_reuse, apply_fusion_nms
    p = build_frozen_candidates(np.ones(10, dtype=int), {'1': {'class_id': 4, 'selected_query_ids': []}},
        np.array([[1]*5+[0]*5, [1]*7+[0]*3], bool), np.array([8, 9]),
        np.array([.9, .8]), np.array([0, 1]))
    labels, _ = apply_label_reuse(p)
    assert labels.tolist() == [4, 4, 4]
    kept, edges = apply_fusion_nms(p)
    assert kept.tolist() == [0, 1]
    assert edges[2]['suppression_iou'] == .7


def test_same_class_competition_positive_ids_ties_and_projection():
    from src.static_ovmap.fusion_attribution import resolve_native_owners
    masks = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], bool)
    kept = np.array([0, 1])
    owners = resolve_native_owners(masks, kept, np.array([2., 2.]))
    assert owners.tolist() == [1, 1, 2, 0]
    nearest = np.array([2, 1, 3, 0, 0]); matched = np.array([1, 1, 1, 1, 0], bool)
    source_then_project = np.where(matched, owners[nearest], 0)
    legacy = resolve_native_owners(masks[:, nearest] & matched, kept, np.array([2., 2.]))
    assert np.array_equal(source_then_project, legacy)
    reverse = resolve_native_owners(masks, kept, np.array([1., 2.]))
    assert reverse.tolist() == [1, 2, 2, 0]
    assert np.all(masks[(reverse[reverse > 0]-1), np.flatnonzero(reverse)])


def test_assignment_score_control_never_changes_rank_input():
    from src.static_ovmap.fusion_attribution import assignment_scores, serialize_scores
    p = pool_fixture()
    rank = p.areas.astype(float)
    norm = assignment_scores(p, np.arange(3), np.array([4, 4, 8]), 'class_norm')
    assert norm.tolist() == [1., 1., 1.]
    assert rank.tolist() == [4., 4., 4.]
    strings, parsed = serialize_scores(np.array([.12345641, .12345649]))
    assert strings == ['0.123456', '0.123456']
    assert parsed[0] == parsed[1]


def test_regions_exhaustive_original_labels_multiple_sf_and_unknown_owner():
    from src.static_ovmap.fusion_attribution import fixed_regions
    p = pool_fixture()
    regions, registry, multiplicity = fixed_regions(p)
    assert len(regions) == 8 and multiplicity.tolist() == [2, 2, 2, 2, 1, 1, 1, 1]
    assert registry[int(regions[0])]['source_support'] == 'both'
    assert registry[int(regions[0])]['class_agreement'] == 'conflicting'
    assert registry[int(regions[4])]['source_support'] == 'SpaCeFormer-only'
    assert np.array_equal(p.native_owners, [1, 1, 1, 1, 2, 2, 0, 0])
    assert sum(np.count_nonzero(regions == key) for key in registry) == len(regions)


def test_regional_confusion_keeps_unmatched_false_negatives_and_sums():
    from src.static_ovmap.attribution_regions import regional_confusions, region_transitions
    gt=np.array([1,1,2,0]); before=np.array([1,0,1,2]); after=np.array([2,0,2,1])
    regions=np.array([14,-1,32,14]); old_owner=np.array([1,0,2,3]); new_owner=np.array([2,0,3,4])
    result=regional_confusions(gt,before,regions,[1,2])
    assert result['global_confusion']==[[0,0,0],[1,1,0],[0,1,0]]
    assert np.array_equal(np.sum([r['confusion'] for r in result['regions'].values()],axis=0),result['global_confusion'])
    assert result['metrics']['semantic_miou']==pytest.approx(1/6)
    t=region_transitions(gt,before,after,old_owner,new_owner,regions,[1,2])
    assert t['14']['correct_to_wrong']==1 and t['32']['wrong_to_correct']==1
    assert t['-1']['wrong_to_wrong']==1 and t['-1']['owner_changes']==0
