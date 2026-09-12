import math

import pytest
import torch


def test_learned_pool_uses_raw_and_projects_each_map_before_mean():
    from src.oviv2.mask_adapter_readout import learned_pool

    raw = torch.tensor([[[[0.0, 10.0]]]])
    activation = torch.tensor([[[[0.0, math.log(3.0)]], [[math.log(3.0), 0.0]]]])
    value = learned_pool(raw, activation, 2, lambda x: x.square())
    assert value.shape == (1, 1, 1)
    assert value.item() == pytest.approx(26.0)


def test_mean_pool_marks_empty_masks_invalid():
    from src.oviv2.mask_adapter_readout import mean_pool

    raw = torch.tensor([[[[2.0, 10.0]]]])
    masks = torch.tensor([[[[1.0, 0.0]], [[0.0, 0.0]], [[1.0, 1.0]]]])
    value, valid = mean_pool(raw, masks, lambda x: x.square())
    assert valid.tolist() == [[True, False, True]]
    assert value[0, 0, 0].item() == 4.0
    assert value[0, 2, 0].item() == 36.0


def test_activation_resolution_is_resized_to_raw_grid():
    from src.oviv2.mask_adapter_readout import learned_pool

    raw = torch.tensor([[[[2.0, 10.0], [2.0, 10.0]]]])
    activation = torch.zeros(1, 2, 1, 1)
    assert learned_pool(raw, activation, 2, lambda x: x).item() == 6.0
