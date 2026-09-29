import numpy as np
import pytest


def test_original_union_pixel_parity_and_independent_shrinking_mask():
    from src.static_ovmap.a7_evidence_upgrade.crops import recognition_crops
    from src.static_ovmap.module_validation.region_evidence import native_crops

    rgb = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    target = np.zeros((48, 64), bool)
    target[12:35, 15:45] = True
    union = target.copy()
    union[8:38, 10:50] = True
    bbox = (15, 12, 44, 34)
    old = native_crops(rgb, target, union, bbox)
    new = recognition_crops(rgb, bbox, union)
    assert new["geometries"] == old.geometries
    for actual, expected in zip(new["six"], old.legacy_six, strict=True):
        np.testing.assert_array_equal(actual, expected)
    shrunk = np.zeros_like(target)
    shrunk[20:23, 25:29] = True
    smaller = recognition_crops(rgb, bbox, shrunk)
    for i in (0, 2, 4):
        np.testing.assert_array_equal(smaller["six"][i], new["six"][i])
    assert smaller["foreground_pixels"] == [12, 12, 12]
    with pytest.raises(ValueError, match="foreground"):
        recognition_crops(rgb, bbox, np.zeros_like(target))
    with pytest.raises(ValueError, match="contain"):
        native_crops(rgb, target, shrunk, bbox)


def test_view_aggregation_preserves_raw_mean_norm_and_original_area():
    from src.static_ovmap.a7_evidence_upgrade.recognition_worker import aggregate_views

    actual = aggregate_views([[.2, 0.], [0., .8]], [2., 1.])
    np.testing.assert_allclose(actual, np.array([1., 2.]) / np.sqrt(5), atol=1e-15)
    assert not np.allclose(actual, np.array([2., 1.]) / np.sqrt(5))
    with pytest.raises(ValueError, match="zero"):
        aggregate_views([[1., 0.], [-1., 0.]], [1., 1.])
