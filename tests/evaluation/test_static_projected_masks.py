import numpy as np
import pytest

from src.evaluation.static_projected_instances import projected_mask_metrics


def test_overlapping_duplicate_masks_cannot_match_gt_twice():
    gt = np.array([1, 1, 2, 2])
    masks = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    result = projected_mask_metrics(masks, gt, [3., 2., 1.], min_region=1)
    assert result['predicted_instance_count'] == 3
    assert result['recall50'] == 1
    assert result['ap50'] == pytest.approx(5/6)


def test_void_points_remain_in_proposal_iou_denominator():
    result = projected_mask_metrics(np.ones((1, 4), dtype=bool), np.array([1, 0, 0, 0]), [1.], min_region=1)
    assert result['ap25'] == 1
    assert result['ap50'] == 0
