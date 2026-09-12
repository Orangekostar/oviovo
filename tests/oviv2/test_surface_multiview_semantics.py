import numpy as np


def test_diverse_selection_avoids_redundant_camera_direction():
    from src.oviv2.surface_multiview_semantics import diverse_views

    directions = np.array([[1, 0, 0], [1, 0.01, 0], [-1, 0, 0], [0, 1, 0]])
    got = diverse_views(directions, np.array([100, 90, 40, 30]), k=3)
    assert got.tolist() == [0, 2, 3]


def test_zero_quality_is_missing_evidence_and_view_norm_does_not_weight_votes():
    from src.oviv2.surface_multiview_semantics import aggregate_views

    features = np.array([[100, 0], [0, 1]], dtype=float)
    np.testing.assert_allclose(aggregate_views(features, [1, 1]), [2**-0.5] * 2)
    np.testing.assert_allclose(aggregate_views(features, [0, 1]), [0, 1])
    assert aggregate_views(features, [0, 0]) is None


def test_equal_direction_ties_use_visibility_and_do_not_repeat_views():
    from src.oviv2.surface_multiview_semantics import diverse_views

    got = diverse_views(np.ones((3, 3)), np.array([3, 8, 5]), k=4)
    assert got.tolist() == [1, 2, 0]
