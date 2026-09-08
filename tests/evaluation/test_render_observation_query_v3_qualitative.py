from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.evaluation.render_observation_query_v3_qualitative import (
    QualitativeRenderError,
    ground_truth_labels_for_points,
    render_topdown_plate,
)
from src.evaluation.rscan_gt_instances import GroundTruthInstance


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.4],
            [1.0, 1.0, 0.6],
            [0.5, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    rgb = np.asarray(
        [
            [20, 30, 40],
            [50, 60, 70],
            [80, 90, 100],
            [110, 120, 130],
            [140, 150, 160],
        ],
        dtype=np.uint8,
    )
    ground_truth = np.asarray([1, 1, 2, 2, 3], dtype=np.int64)
    predictions = {
        "Frozen": np.asarray([0, 0, 1, 1, -1], dtype=np.int64),
        "BASE": np.asarray([0, 1, 1, 2, -1], dtype=np.int64),
        "FULL": np.asarray([0, 0, 1, 1, 2], dtype=np.int64),
    }
    return points, rgb, ground_truth, predictions


def test_topdown_plate_is_deterministic_and_records_panel_contract(
    tmp_path: Path,
) -> None:
    points, rgb, ground_truth, predictions = _fixture()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    first_record = render_topdown_plate(
        points_xyz=points,
        camera_rgb_uint8=rgb,
        ground_truth_labels=ground_truth,
        prediction_labels=predictions,
        output_path=first,
        title="scene0009 visit 1",
        panel_size=96,
    )
    second_record = render_topdown_plate(
        points_xyz=points,
        camera_rgb_uint8=rgb,
        ground_truth_labels=ground_truth,
        prediction_labels=predictions,
        output_path=second,
        title="scene0009 visit 1",
        panel_size=96,
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_record == second_record
    assert first_record["panels"] == ["RGB support", "GT", "Frozen", "BASE", "FULL"]
    assert first_record["point_count"] == 5
    assert first_record["projection"] == "XY_TOPDOWN_MAX_Z"
    assert first_record["prediction_color_semantics"] == "PANEL_LOCAL_INSTANCE_ID"
    assert first_record["label_text_height_px"] >= 14
    with Image.open(first) as image:
        assert image.format == "PNG"
        assert image.width > image.height


def test_topdown_plate_rejects_prediction_rows_from_another_dense_visit(
    tmp_path: Path,
) -> None:
    points, rgb, ground_truth, predictions = _fixture()
    predictions["FULL"] = predictions["FULL"][:-1]

    with pytest.raises(QualitativeRenderError, match="prediction labels"):
        render_topdown_plate(
            points_xyz=points,
            camera_rgb_uint8=rgb,
            ground_truth_labels=ground_truth,
            prediction_labels=predictions,
            output_path=tmp_path / "bad.png",
            title="bad",
            panel_size=96,
        )


def test_ground_truth_labels_use_the_evaluation_voxel_grid() -> None:
    points = np.asarray(
        [[0.001, 0.001, 0.001], [0.049, 0.049, 0.049], [0.051, 0.0, 0.0]],
        dtype=np.float32,
    )
    targets = (
        GroundTruthInstance(7, "first", frozenset({(0, 0, 0)})),
        GroundTruthInstance(8, "second", frozenset({(1, 0, 0)})),
    )

    labels = ground_truth_labels_for_points(
        points_xyz=points,
        targets=targets,
        voxel_size_m=0.05,
    )

    np.testing.assert_array_equal(labels, [7, 7, 8])


def test_ground_truth_labels_mark_overlapping_voxels_as_ambiguous() -> None:
    points = np.asarray([[0.001, 0.001, 0.001]], dtype=np.float32)
    targets = (
        GroundTruthInstance(7, "first", frozenset({(0, 0, 0)})),
        GroundTruthInstance(8, "second", frozenset({(0, 0, 0)})),
    )

    labels = ground_truth_labels_for_points(
        points_xyz=points,
        targets=targets,
        voxel_size_m=0.05,
    )

    np.testing.assert_array_equal(labels, [-2])
