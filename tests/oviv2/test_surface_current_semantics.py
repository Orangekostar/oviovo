import numpy as np


def test_ambiguous_current_region_pixels_supply_no_semantic_vote():
    from src.oviv2.surface_current_semantics import unique_region_pixels

    # Duplicate vertices within one region are one vote; overlapping regions
    # must not be arbitrarily resolved by iteration order or owner ID.
    actual = unique_region_pixels([np.array([1, 1, 2]), np.array([2, 3])], 5)
    assert actual.tolist() == [-1, 0, -1, 1, -1]


def test_current_semantics_require_two_positive_quality_distinct_frames():
    from src.oviv2.surface_current_semantics import current_view_embedding

    features = np.eye(2)
    directions = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    assert current_view_embedding(features, [1, 1], [10, 10], directions) is None
    assert current_view_embedding(features, [1, 0], [10, 20], directions) is None
    np.testing.assert_allclose(
        current_view_embedding(features, [1, 1], [10, 20], directions),
        [2**-0.5, 2**-0.5],
    )
