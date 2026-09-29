import numpy as np
import pytest


def test_signed_support_has_no_positive_halo_or_padding_and_resize_is_fixed():
    from src.static_ovmap.a7_evidence_upgrade.region_adapter import (
        image_tensor,
        resized_shape,
        signed_mask,
    )

    assert resized_shape((480, 640)) == (800, 1067)
    assert resized_shape((100, 1000)) == (133, 1333)
    mask = np.array([[False, True], [False, False]])
    signed, support, resized = signed_mask(mask, (2, 2), (4, 4), (8, 8), "cpu")
    np.testing.assert_array_equal(resized, mask)
    assert (signed[0, 0, 2:] == -1).all()
    assert (signed[0, 0, :, 2:] == -1).all()
    assert int(support.sum()) == 4
    image, shape = image_tensor(np.zeros((3, 4, 3), np.uint8), "cpu")
    assert shape == (800, 1067) and tuple(image.shape) == (1, 3, 800, 1088)
    assert (image[:, :, :, 1067:] == 0).all()
    with pytest.raises(ValueError, match="boolean"):
        signed_mask(mask.astype(np.uint8), (2, 2), (4, 4), (8, 8), "cpu")
