from dataclasses import replace
import numpy as np

from test_static_ovmap_readout import observations
from src.static_ovmap.fallback import fallback_decision, apply_fallback


def test_single_semantic_query_needs_independent_geometry_and_measured_quality():
    obs = replace(observations(1)[0], visible_area_px=1000,
                  depth_valid_ratio=.99, sharpness_proxy=.8)
    assert fallback_decision([obs], {'independent_frame_ids': [0]})['accepted'] is False
    assert fallback_decision([obs], {'independent_frame_ids': [0, 20]})['accepted'] is True
    assert fallback_decision([replace(obs, sharpness_proxy=None)],
                             {'independent_frame_ids': [0, 20]})['accepted'] is False


def test_fallback_does_not_duplicate_query_or_change_existing_instance():
    obs = replace(observations(1)[0], visible_area_px=1000,
                  depth_valid_ratio=.99, sharpness_proxy=.8)
    existing = {'1': None, '2': {'class_id': 7, 'selected_count': 2}}
    result, ledger = apply_fallback({1: [obs]}, existing,
        {'objects': {'1': {'independent_frame_ids': [0,20]}}},
        np.eye(2), [3,7], 'siglip:a')
    assert result['1']['selected_count'] == 1
    assert result['2'] == existing['2']
    assert existing['1'] is None
    assert ledger[0]['accepted'] is True
