import numpy as np
import pytest


def test_geometric_matching_retains_unmatched_gt_and_requires_strict_half_iou():
    from static_ovmap.benchmark_completion_v2.diagnostics import geometry_matches
    # Three full GT instances. The second owner covers exactly half of GT2;
    # GT3 is never observed. Neither is removed from the recall denominator.
    gt = np.array([2001, 2001, 2001, 2001, 4001, 4001, 4001, 4001, 6001, 6001])
    result = geometry_matches(np.array([7, 9]), np.array([0, 4, 6]),
                              np.array([0, 1, 2, 3, 4, 5]), gt, [2, 4, 6], minimum=1)
    assert result['valid_GT_instances'] == 3
    assert result['matched_GT_instances'] == 1
    assert result['pairs'] == [dict(owner_index=0, owner=7, GT_instance=2001,
                                    class_id=2, IoU=1.)]
    assert result['proposal_recall'] == 1 / 3


def test_native_geometry_diagnostic_rejects_overlapping_owner_partition():
    from static_ovmap.benchmark_completion_v2.diagnostics import geometry_matches
    with pytest.raises(ValueError, match='partition'):
        geometry_matches(np.array([7, 9]), np.array([0, 2, 4]),
                         np.array([0, 1, 1, 2]), np.array([2001, 2001, 2001]), [2], minimum=1)
