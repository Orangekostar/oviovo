import numpy as np
import pytest

from src.static_ovmap.enrichment import crop_quality, view_direction, apply_enrichment
from src.static_ovmap.cache_io import validate_native_binding
from test_static_ovmap_readout import observations


def test_crop_quality_uses_legacy_slice_and_measured_depth():
    rgb = np.full((6, 6, 3), 100, dtype=np.uint8)
    depth = np.ones((6, 6)); depth[0, 0] = 0
    depth[5, :] = 0
    result = crop_quality(rgb, depth, (0, 0, 5, 5))
    assert result['depth_valid_ratio'] == 24 / 25
    assert result['sharpness_proxy'] == 0
    assert result['mask_quality_proxy'] is None
    assert crop_quality(rgb, depth, (2, 2, 2, 2))['sharpness_proxy'] is None


def test_sharpness_distinguishes_texture_without_semantics():
    rgb = np.repeat(((np.indices((12, 12)).sum(0) % 2) * 255)[:, :, None], 3, 2)
    result = crop_quality(rgb.astype(np.uint8), np.ones((12, 12)), (0, 0, 11, 11))
    assert .9 < result['sharpness_proxy'] < 1


def test_view_is_object_relative_and_translation_invariant():
    np.testing.assert_allclose(view_direction([2, 0, 0], [1, 0, 0]), [1, 0, 0])
    np.testing.assert_allclose(view_direction([12, 5, 3], [11, 5, 3]), [1, 0, 0])
    assert view_direction([1, 0, 0], [1, 0, 0]) is None


def test_enrichment_binds_exact_cache_and_query_identity():
    obs = observations(2); bank = {1: obs}
    data = {'native_cache_sha256': 'abc', 'observations': {
        '0': {'frame_id': 100, 'instance_id': 1, 'sharpness_proxy': .8}}}
    with pytest.raises(ValueError, match='cache'):
        apply_enrichment(bank, data, 'wrong')
    enriched = apply_enrichment(bank, data, 'abc')
    assert enriched[1][0].sharpness_proxy == .8
    assert enriched[1][1].sharpness_proxy is None
    assert enriched[1][0].geometry_support_frames is None
    data['observations']['0']['frame_id'] = 99
    with pytest.raises(ValueError, match='identity'):
        apply_enrichment(bank, data, 'abc')


def test_native_binding_rejects_relabelled_text_space():
    binding = {'native_cache_sha256': 'cache', 'feature_space_id': 'original',
               'source_config_sha256': 'source', 'feature_dim': 2}
    validate_native_binding(binding, 'cache', 'source', 'original', 2)
    with pytest.raises(ValueError, match='feature space'):
        validate_native_binding(binding, 'cache', 'source', 'new_encoder_same_dim', 2)
    with pytest.raises(ValueError, match='cache'):
        validate_native_binding(binding, 'different', 'source', 'original', 2)
