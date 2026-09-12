import numpy as np
import pytest

from scripts.evaluation.audit_crove_dynamic_per_class import per_class_counts
from src.evaluation.baselines.dynamic_metrics import _semantic_miou
from scripts.evaluation.summarize_crove_multimethod_readouts import per_class_delta


def test_counts_preserve_original_gt_domain_and_absent_class_policy():
    gt = np.array([1, 1, 2, 2, 99])
    pred = np.array([1, 0, 1, 3, 2])
    result = per_class_counts(gt, pred, {1, 2, 3})
    assert set(result) == {"1", "2"}
    assert result["1"] == {"iou": 1 / 3, "support": 2, "intersection": 1, "union": 3}
    assert result["2"] == {"iou": 0.0, "support": 2, "intersection": 0, "union": 2}
    assert np.mean([v["iou"] for v in result.values()]) == pytest.approx(
        _semantic_miou(gt, pred, {1, 2, 3}), abs=1e-12
    )


def test_empty_valid_gt_has_no_invented_per_class_values():
    assert per_class_counts(np.array([99]), np.array([1]), {1}) == {}


def test_summary_exports_supported_class_deltas_from_reconstructed_counts():
    gt = np.array([1, 1, 2, 2])
    current = per_class_counts(gt, np.array([1, 0, 2, 2]), {1, 2})
    reference = per_class_counts(gt, np.array([1, 1, 1, 2]), {1, 2})
    delta = per_class_delta(
        {"semantic": {"per_class": current}},
        {"semantic": {"per_class": reference}},
    )
    assert delta["1"]["delta"] == pytest.approx(-1 / 6)
    assert delta["2"]["delta"] == pytest.approx(0.5)
    assert delta["1"]["GT_support"] == delta["2"]["GT_support"] == 2
