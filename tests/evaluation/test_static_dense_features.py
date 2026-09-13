import numpy as np

from src.static_ovmap.dense_features import window_boxes, pool_owner_features


def test_windows_cover_non_stride_aligned_frame_without_changing_crop_size():
    boxes = window_boxes(336, 592)
    coverage = np.zeros((336, 592), dtype=np.int16)
    for x1, y1, x2, y2 in boxes:
        assert x2-x1 == y2-y1 == 224
        coverage[y1:y2, x1:x2] += 1
    assert len(boxes) == 10
    assert boxes[-1] == (368, 112, 592, 336)
    assert coverage.min() > 0


def test_owner_pool_excludes_unknown_and_reports_no_support():
    features = np.array([[[1., 0.], [1., 0.]], [[0., 1.], [100., 100.]]])
    owners = np.array([[1, 1], [2, 0]])
    pooled = pool_owner_features(features, owners, [1, 2, 3])
    np.testing.assert_array_equal(pooled[1]['feature'], [1., 0.])
    np.testing.assert_array_equal(pooled[2]['feature'], [0., 1.])
    assert pooled[3] is None
    assert 0 not in pooled
