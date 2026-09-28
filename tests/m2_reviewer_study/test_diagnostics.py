import numpy as np
import pytest


def test_probability_scores_use_sum_brier_clip_only_nll_and_full_confidence_bin():
    from src.static_ovmap.m2_reviewer_study.diagnostics import probability_metrics

    result = probability_metrics([[1., 0.], [.25, .75]], [1, 1])
    assert result["objects"] == 2
    assert result["accuracy"] == .5
    assert result["brier"] == pytest.approx((2 + .125) / 2)
    assert result["nll"] == pytest.approx((-np.log(1e-12) - np.log(.75)) / 2)
    assert result["ece15"] == pytest.approx(.625)
    assert sum(row["count"] for row in result["bins"]) == 2
    tied = probability_metrics([[.5, .5]], [1], predictions=[1])
    assert tied["accuracy"] == 1
    assert tied["ece15"] == .5


def test_bootstrap_resamples_scenes_not_vertices_and_singleton_softmax():
    from scipy.special import softmax

    from src.static_ovmap.m2_reviewer_study.diagnostics import bootstrap_delta

    result = bootstrap_delta([.2] * 8)
    assert result["ci95"] == pytest.approx([.2, .2])
    assert result["scenes"] == 8 and result["draws"] == 2000 and result["seed"] == 17
    np.testing.assert_array_equal(softmax(np.array([[-100.], [0.], [100.]]), axis=1), np.ones((3, 1)))
