import numpy as np
import pytest
import torch

from static_ovmap.a7_evidence_upgrade.region_adapter import signed_mask


def original_pool(dense, signed):
    mask = torch.nn.functional.interpolate(
        signed, size=dense.shape[-2:], mode="bilinear", align_corners=False
    )
    mask = (mask > 0).to(mask.dtype)
    return torch.einsum("bchw,bqhw->bqc", dense, mask / (mask.sum((-1, -2), keepdim=True) + 1e-8))


def test_tiny_mask_survives_with_unequal_area_weights():
    from static_ovmap.cvpr_compact.area_fallback import pool_region

    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 0] = True
    mask[0, 4:6] = True
    signed, support, _ = signed_mask(mask, (8, 8), (8, 8), (2, 2), "cpu")
    assert not support.any()
    dense = torch.tensor([[[[3., 9.], [50., 100.]]]])
    pooled, audit = pool_region(dense, signed, original_pool)
    assert audit["fallback"] is True
    assert audit["original_support"] == 0
    assert audit["area_support"] == 2
    assert audit["area_mass"] == pytest.approx(3 / 16)
    torch.testing.assert_close(pooled, torch.tensor([[[7.]]]))


def test_padding_has_no_foreground_contribution():
    from static_ovmap.cvpr_compact.area_fallback import pool_region

    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    signed, support, _ = signed_mask(mask, (4, 4), (8, 8), (2, 2), "cpu")
    assert not support.any()
    dense = torch.tensor([[[[5., 100.], [200., 300.]]]])
    pooled, audit = pool_region(dense, signed, original_pool)
    assert audit["area_mass"] == pytest.approx(1 / 16)
    torch.testing.assert_close(pooled, torch.tensor([[[5.]]]))


def test_nonempty_original_path_is_bit_exact():
    from static_ovmap.cvpr_compact.area_fallback import pool_region

    mask = np.zeros((8, 8), dtype=bool)
    mask[:4, :4] = True
    signed, _, _ = signed_mask(mask, (8, 8), (8, 8), (2, 2), "cpu")
    dense = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    expected = original_pool(dense, signed)
    actual, audit = pool_region(dense, signed, original_pool)
    assert torch.equal(actual, expected)
    assert audit["fallback"] is False
    assert audit["original_support"] == 1


def test_empty_mask_remains_an_explicit_failure():
    from static_ovmap.cvpr_compact.area_fallback import pool_region

    with pytest.raises(ValueError, match="EMPTY_AREA_MASK_SUPPORT"):
        pool_region(torch.ones(1, 2, 2, 2), -torch.ones(1, 1, 8, 8), original_pool)


@pytest.mark.parametrize("bad_value", [0., float("nan")])
def test_invalid_signed_mask_is_rejected(bad_value):
    from static_ovmap.cvpr_compact.area_fallback import pool_region

    mask = -torch.ones(1, 1, 8, 8)
    mask[0, 0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="signed"):
        pool_region(torch.ones(1, 2, 2, 2), mask, original_pool)
