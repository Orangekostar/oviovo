from __future__ import annotations

import numpy as np

from src.evaluation.baselines.ovimap import (
    group_points_by_color,
    parse_instance_color_log,
    relative_similarity_labels,
    remap_instance_colors,
)


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


def test_instance_color_log_remaps_mask_colors_to_mesh_colors(tmp_path) -> None:
    log = tmp_path / "mapping.log"
    log.write_text(
        "I global_segment_map_py.cpp:642 Instance: 7 Color: (214,36,176)\n",
        encoding="utf-8",
    )
    instances = {7: {"color": [214, 214, 214], "feat": [[1.0, 0.0]]}}

    colors = parse_instance_color_log(log)
    remapped = remap_instance_colors(instances, colors)

    assert colors == {7: (214, 36, 176)}
    assert remapped[7]["color"] == (214, 36, 176)
    assert instances[7]["color"] == [214, 214, 214]
