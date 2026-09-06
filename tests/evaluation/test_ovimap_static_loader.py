from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.baselines.ovimap import (
    bind_mesh_instances,
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


def test_instance_color_log_accepts_native_audit_tsv(tmp_path) -> None:
    log = tmp_path / "instance_colors_cpp.tsv"
    log.write_text(
        "instance_id\tr\tg\tb\n7\t10\t20\t30\n8\t40\t50\t60\n",
        encoding="utf-8",
    )

    assert parse_instance_color_log(log) == {
        7: (10, 20, 30),
        8: (40, 50, 60),
    }


@pytest.mark.parametrize(
    "contents",
    [
        (
            "Instance: 7 Color: (10,20,30)\n"
            "Instance: 7 Color: (40,50,60)\n"
        ),
        (
            "Instance: 7 Color: (10,20,30)\n"
            "Instance: 8 Color: (10,20,30)\n"
        ),
    ],
)
def test_instance_color_log_rejects_conflicting_identity(contents, tmp_path) -> None:
    log = tmp_path / "mapping.log"
    log.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting|reused"):
        parse_instance_color_log(log)


def test_bind_mesh_instances_keeps_logged_instances_without_features() -> None:
    features = {
        7: {"feat": [[1.0, 0.0]], "frame_id": [0], "color": [7, 7, 7]}
    }
    colors = {7: (10, 20, 30), 8: (40, 50, 60)}
    points = {
        (10, 20, 30): np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        (40, 50, 60): np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    }

    bound = bind_mesh_instances(features, colors, points)

    assert set(bound) == {7, 8}
    assert bound[7]["color"] == (10, 20, 30)
    assert bound[8] == {"color": (40, 50, 60)}


def test_bind_mesh_instances_ignores_stale_features_missing_from_final_mesh() -> None:
    bound = bind_mesh_instances(
        {9: {"feat": [[1.0]]}},
        {7: (10, 20, 30)},
        {(10, 20, 30): np.zeros((1, 3), dtype=np.float32)},
    )

    assert bound == {7: {"color": (10, 20, 30)}}
