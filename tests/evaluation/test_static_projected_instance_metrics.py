import numpy as np

from src.evaluation.static_projected_instances import projected_instance_metrics


def test_class_agnostic_ap_integrates_confidence_ranked_precision_and_counts_void_fp():
    # Two perfect objects and one high-confidence prediction entirely on void.
    gt = np.array([1, 1, 2, 2, 0, 0])
    pred = np.array([10, 10, 20, 20, 30, 30])
    result = projected_instance_metrics(pred, gt, {30: .9, 10: .8, 20: .7}, min_region=1)
    assert result['ap50'] == 2/3
    assert result['recall50'] == 1.
    assert result['predicted_instance_count'] == 3


def test_duplicate_fragments_do_not_match_same_gt_twice_and_high_iou_is_distinct():
    gt = np.array([1, 1, 1, 1])
    pred = np.array([10, 10, 20, 20])
    result = projected_instance_metrics(pred, gt, {10: .8, 20: .7}, min_region=1)
    assert result['ap50'] == 1.
    assert result['ap75'] == 0.
    assert result['recall50'] == 1.


def test_no_gt_is_undefined_and_missing_prediction_confidence_is_rejected():
    import pytest
    assert projected_instance_metrics(np.array([1]), np.array([0]), {1: .5}, min_region=1)['ap50'] is None
    with pytest.raises(ValueError):
        projected_instance_metrics(np.array([1]), np.array([1]), {}, min_region=1)
