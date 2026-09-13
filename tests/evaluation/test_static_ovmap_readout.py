import numpy as np
import pytest

from src.static_ovmap.contracts import Observation
from src.static_ovmap.observation_bank import select_observations
from src.static_ovmap.readout import fuse_features, classify


def observations(n=10):
    return [Observation(scene_id='room0', instance_id=1, frame_id=100-i,
                        source_query_id=str(i), feature_space_id='siglip:a',
                        feature=np.array([i+1., 1.], dtype=np.float32),
                        visible_area_px=float(i+1), crop_bbox_xyxy=(0, 0, 9, 9),
                        source_config_hash='config-a') for i in range(n)]


def test_last8_preserves_cache_order_not_frame_order_and_original_formula():
    obs = observations()
    chosen = select_observations(obs, 'last8')
    assert [o.source_query_id for o in chosen] == list(map(str, range(2, 10)))
    area = np.array([o.visible_area_px for o in chosen])
    expected = (np.stack([o.feature for o in chosen]) *
                (area / (area.sum() + 1e-6))[:, None]).sum(axis=0)
    np.testing.assert_array_equal(fuse_features(chosen, 'vis_area'), expected)


def test_equal_dimension_different_spaces_are_rejected():
    obs = observations(2)
    from dataclasses import replace
    obs[1] = replace(obs[1], feature_space_id='siglip:other-crop')
    with pytest.raises(ValueError, match='feature space'):
        fuse_features(obs, 'vis_area')


def test_fixed_random_selection_and_budget():
    obs = observations(20)
    a = select_observations(obs, 'random8', seed=7)
    b = select_observations(obs, 'random8', seed=7)
    assert len(a) == 8
    assert [o.source_query_id for o in a] == [o.source_query_id for o in b]
    assert len(select_observations(obs, 'all_views')) == 20


def test_quality_coverage_rewards_distinct_observed_direction():
    from dataclasses import replace
    obs = [replace(o, depth_valid_ratio=1., camera_direction=(1., 0., 0.))
           for o in observations(10)]
    obs[0] = replace(obs[0], camera_direction=(-1., 0., 0.), visible_area_px=9.)
    chosen = select_observations(obs, 'quality_coverage8')
    assert '0' in [o.source_query_id for o in chosen]
    assert len(chosen) == 8


def test_unknown_quality_is_explicit_and_area_only_is_supported():
    obs = observations(10)
    assert obs[0].depth_valid_ratio is None
    assert 'depth_valid_ratio' in obs[0].missing_fields
    assert [o.source_query_id for o in select_observations(obs, 'quality8')] == list(map(str, range(2, 10)))
    from dataclasses import replace
    tied = [replace(o, visible_area_px=1.) for o in obs]
    assert [o.source_query_id for o in select_observations(tied, 'quality_coverage8')] == [
        o.source_query_id for o in select_observations(tied, 'quality8')]


def test_min_two_and_text_space_binding():
    text = np.eye(2)
    assert classify(observations(1), text, [3, 7], 'siglip:a') is None
    result = classify(observations(2), text, [3, 7], 'siglip:a')
    assert result['class_id'] == 3
    with pytest.raises(ValueError, match='feature space'):
        classify(observations(2), text, [3, 7], 'other')


def test_s1_weight_and_selection_are_independent():
    from dataclasses import replace
    obs = [replace(o, depth_valid_ratio=(.01 if i > 5 else 1.))
           for i, o in enumerate(observations())]
    last = select_observations(obs, 'last8')
    quality = select_observations(obs, 'quality8')
    assert [o.source_query_id for o in last] != [o.source_query_id for o in quality]
    assert not np.allclose(fuse_features(last, 'quality'), fuse_features(last, 'vis_area'))


def test_native_cache_preserves_source_order_and_does_not_invent_geometry(tmp_path):
    import pickle
    from src.static_ovmap.cache_io import load_native_cache
    raw = {70000: {'feat': np.eye(2), 'vis_area': np.array([5, 8]),
                   'frame_id': [90, 10], 'box_2d': [(0, 0, 4, 4)]*2,
                   'pose': [np.eye(4)]*2, 'color': np.array([1, 2, 3])}}
    path = tmp_path / 'native.pkl'
    path.write_bytes(pickle.dumps(raw))
    bank, metadata = load_native_cache(path, scene_id='room0',
        feature_space_id='siglip:a', source_config_hash='a',
        history_scope='retained_native_top10')
    assert [o.frame_id for o in bank[70000]] == [90, 10]
    assert bank[70000][0].camera_direction is None
    assert bank[70000][0].geometry_support_frames is None
    assert metadata['history_scope'] == 'retained_native_top10'
    assert metadata['instance_colors']['70000'] == [1, 2, 3]
