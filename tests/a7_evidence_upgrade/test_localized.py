import numpy as np


def test_common_view_failure_zero_detection_and_tie_rules():
    from src.static_ovmap.a7_evidence_upgrade.localized import decide

    # Failure for class2 removes its entire high-area view, including class1.
    r = decide([1, 2], 1, [.8, .2], [[1., None], [.1, .9]], [100., 1.])
    assert r["common_view_indices"] == [1]
    assert (r["SHORTLIST"], r["SPATIAL"], r["MIX50"]) == (1, 2, 2)
    np.testing.assert_allclose(r["q"], [.1, .9])
    r = decide([1, 2], 2, [.5, .5], [[.3, .3]], [1.])
    assert (r["SHORTLIST"], r["SPATIAL"], r["MIX50"]) == (2, 2, 2)
    r = decide([1, 2], 1, [.8, .2], [[0., 0.]], [1.])
    assert r["status"] == "NO_LOCALIZED_EVIDENCE" and r["SPATIAL"] == 1
    assert r["common_view_indices"] == [0]
