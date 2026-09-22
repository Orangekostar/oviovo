"""Geometry supervision keeps full GT denominators and invisible components."""

import numpy as np

from src.static_ovmap.module_validation.geometry_targets import projected_local_quality


def test_half_object_has_no_strict_match_and_unprojected_component_is_false_positive():
    half = projected_local_quality(np.array([0, 0, -1, -1]), 1, np.array([1, 1, 1, 1]))
    assert half.quality == 0
    assert half.false_negatives == 1
    assert half.false_positives == 1
    extra = projected_local_quality(np.array([0, 0, 0]), 2, np.array([1, 1, 1]))
    assert extra.true_positives == 1
    assert extra.false_positives == 1
    assert extra.quality == 2 / 3
    empty = projected_local_quality(np.array([-1, -1]), 2, np.array([1, 2]))
    assert empty.included_gt_count == 0
    assert empty.false_positives == 2
    assert empty.quality == 0
