import numpy as np

from src.static_ovmap.contracts import Observation
from src.static_ovmap.hierarchy import component_readout


def test_singletons_keep_original_readout_and_featureless_components_remain_unknown():
    baseline = {'1': {'class_id': 4}, '2': None}
    result = component_readout([[1], [2, 3]], {}, baseline, np.eye(2), [1, 2], 'x', np.eye(2))
    assert result['observations'] == {'1': {'class_id': 4}, '2': None}
    assert baseline == {'1': {'class_id': 4}, '2': None}


def test_merge_fuses_only_original_eligible_selected_queries():
    def obs(owner, query, feat):
        return Observation(scene_id='s', instance_id=owner, frame_id=query,
            source_query_id=f'{owner}:{query}', feature_space_id='x', feature=np.array(feat),
            visible_area_px=1., crop_bbox_xyxy=(0, 0, 1, 1), source_config_hash='test')
    bank = {1: [obs(1, 0, [1., 0.]), obs(1, 1, [1., 0.])], 2: [obs(2, 0, [0., 1.])]}
    baseline = {'1': {'selected_query_ids': ['1:0', '1:1']}, '2': None}
    result = component_readout([[1, 2]], bank, baseline, np.eye(2), [1, 2], 'x', np.eye(2))
    assert result['observations']['1']['class_id'] == 1
    assert result['observations']['1']['selected_query_ids'] == ['1:0', '1:1']
    assert result['pooled_query_counts'] == {'1': 2}


def test_two_eligible_native_owners_can_fuse_only_through_declared_component():
    bank = {}
    for owner, feature in [(1, [1., 0.]), (2, [0., 1.])]:
        bank[owner] = [Observation(scene_id='s', instance_id=owner, frame_id=i,
            source_query_id=f'{owner}:{i}', feature_space_id='x', feature=np.array(feature),
            visible_area_px=float(owner), crop_bbox_xyxy=(0, 0, 1, 1), source_config_hash='test') for i in (0, 1)]
    baseline = {str(o): {'selected_query_ids': [f'{o}:0', f'{o}:1']} for o in (1, 2)}
    result = component_readout([[1, 2]], bank, baseline, np.eye(2), [1, 2], 'x', np.eye(2))
    assert result['observations']['1']['class_id'] == 2
    assert result['observations']['1']['selected_query_ids'] == ['1:0', '1:1', '2:0', '2:1']
    assert bank[2][0].instance_id == 2
