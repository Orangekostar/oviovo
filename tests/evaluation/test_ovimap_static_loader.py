from __future__ import annotations

import numpy as np

from src.evaluation.baselines.ovimap import group_points_by_color, relative_similarity_labels


def test_group_points_by_color_keeps_only_requested_instance_colors() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    colors = np.asarray([[10, 20, 30], [1, 2, 3], [10, 20, 30]], dtype=np.uint8)

    grouped = group_points_by_color(points, colors, {(10, 20, 30)})

    assert set(grouped) == {(10, 20, 30)}
    np.testing.assert_allclose(grouped[(10, 20, 30)], points[[0, 2]])


def test_relative_similarity_labels_matches_ovimap_canonical_rule() -> None:
    entity_features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    text_features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    canonical_features = np.asarray([[-1.0, -1.0]], dtype=np.float32)

    labels, scores = relative_similarity_labels(
        entity_features,
        text_features,
        canonical_features,
        ("chair", "table"),
    )

    assert labels == ("chair", "table")
    assert all(0.5 < score < 1.0 for score in scores)
