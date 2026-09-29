import numpy as np
import pytest


def test_raw_unit_and_median_preserve_defined_small_sample_behavior():
    from src.static_ovmap.a7_evidence_upgrade.readouts import query_readout

    features = np.array([[.2, 0.], [0., .8]])
    areas = np.array([4., 1.])
    raw, _ = query_readout(features, areas, "RAW_EQ")
    unit, _ = query_readout(features, areas, "UNIT_EQ")
    median, detail = query_readout(features, areas, "GMED")
    np.testing.assert_allclose(raw, np.array([1., 4.]) / np.sqrt(17))
    np.testing.assert_allclose(unit, np.ones(2) / np.sqrt(2))
    np.testing.assert_array_equal(median, unit)
    assert detail["small_sample"] == "TWO_VIEW_MIDPOINT"
    one, _ = query_readout(features[:1], areas[:1], "GMED")
    np.testing.assert_array_equal(one, [1., 0.])
    opposite, detail = query_readout(np.array([[1., 0.], [-1., 0.]]), [2., 1.], "GMED")
    np.testing.assert_array_equal(opposite, [1., 0.])
    assert detail["numerical_fallback"] == "ORIGINAL_AREA_DIRECTION"
    with pytest.raises(ValueError, match="finite"):
        query_readout([[float("nan"), 0.]], [1.], "GMED")


def test_smoothed_median_uses_euclidean_iterations_without_sphere_projection():
    from src.static_ovmap.a7_evidence_upgrade.readouts import query_readout

    features = np.array([[1., 0.], [1., .1], [0., 1.]])
    vectors = features / np.linalg.norm(features, axis=1, keepdims=True)
    x = vectors.mean(0)
    for _ in range(64):
        weights = 1 / np.sqrt(np.square(vectors - x).sum(1) + 1e-8)
        next_x = np.average(vectors, axis=0, weights=weights)
        distance = np.linalg.norm(next_x - x)
        x = next_x
        if distance <= 1e-8:
            break
    actual, detail = query_readout(features, [1., 2., 3.], "GMED")
    np.testing.assert_allclose(actual, x / np.linalg.norm(x), atol=1e-14)
    np.testing.assert_allclose(sum(detail["influence_weights"]), 1.)
    assert 1 <= detail["effective_sample_size"] <= 3
    assert detail["iterations"] <= 64
