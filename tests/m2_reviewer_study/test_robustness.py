import numpy as np
import pytest


def test_distractor_winner_is_an_error_not_an_ignored_label():
    from src.static_ovmap.m2_reviewer_study.robustness import stress_metrics

    result = stress_metrics([[.1, .2, .7]], [1], [1], 2)
    assert result["objects"] == 1 and result["accuracy"] == 0
    assert result["distractor_selection_rate"] == 1
    assert result["nll"] == pytest.approx(-np.log(.2))
    assert result["original_to_original_denominator"] == 0
    assert result["restricted_original_argmax_changes"] == 0
