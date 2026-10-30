"""Full-point semantic metrics must preserve unknown errors and frozen mask order."""
import importlib

import numpy as np
import pytest


def implementation():
    try:
        return importlib.import_module('static_ovmap.benchmark_point_metrics')
    except ModuleNotFoundError:
        pytest.fail('The common full-point semantic metric implementation is missing')


def test_overlap_uses_frozen_confidence_then_original_order_and_keeps_uncovered_points():
    m=implementation()
    # The second and third proposals tie; the earlier second column owns point2.
    result=m.project_masks(indptr=[0,3,6,8,9],indices=[0,1,2,1,2,3,2,4,5],
        labels=[10,20,30,999],confidence=[.5,.9,.9,1.],target_points=7,class_ids=[10,20,30])
    assert result.tolist()==[10,20,20,20,30,0,0]


def test_full_gt_confusion_keeps_unknown_false_negatives_and_ignores_only_invalid_gt():
    m=implementation()
    matrix=m.confusion_counts([10,10,20,20,0,999],[10,0,10,20,10,10],[10,20,30])
    assert matrix.tolist()==[[0,0,0,0],[1,1,0,0],[0,1,1,0],[0,0,0,0]]
    x=m.summarize(matrix,[10,20,30])
    assert x['valid_gt_points']==4 and x['correct_points']==2 and x['defined_classes']==2
    assert x['miou']==pytest.approx(5/12) and x['macc']==.5 and x['oacc']==.5
    assert x['per_class'][2]['iou'] is None


def test_pool_confusions_before_macro_metrics_instead_of_averaging_scene_scores():
    m=implementation()
    a=np.array([[0,0,0],[10,90,0],[0,0,0]])
    b=np.array([[0,0,0],[1,0,0],[0,0,1]])
    x=m.summarize(a+b,[10,20])
    assert x['valid_gt_points']==102 and x['correct_points']==91
    assert x['miou']==pytest.approx((90/101+1)/2)
    assert x['macc']==pytest.approx((90/101+1)/2)
    assert x['oacc']==pytest.approx(91/102)


@pytest.mark.parametrize('kwargs',[
    dict(indptr=[0,1],indices=[7],labels=[10],confidence=[1.],target_points=7,class_ids=[10]),
    dict(indptr=[0,1],indices=[0],labels=[10],confidence=[float('nan')],target_points=7,class_ids=[10]),
    dict(indptr=[0,2],indices=[0],labels=[10],confidence=[1.],target_points=7,class_ids=[10]),
])
def test_malformed_mask_domains_and_nonfinite_confidence_fail_before_scoring(kwargs):
    with pytest.raises(ValueError):implementation().project_masks(**kwargs)
