"""Native readout/projection parity, without visual inference or private inputs."""

from pathlib import Path

import numpy as np
import pytest

from src.static_ovmap.module_validation import scannet_study as study


def test_prediction_reload_keeps_native_masks_ranks_and_checks_array_hash(tmp_path):
    from src.static_ovmap.module_validation.evaluation import (
        GeometryIdentity,
        PredictionPayload,
    )

    native = PredictionPayload("N0", "N0", "scene-a", GeometryIdentity("a" * 64, "b" * 64,
        "c" * 64, "projection", 3), np.array([7, 7, 0]), np.array([2, 2, 0]),
        ((7, 0.123456),), {"attempts": 1})
    native.lock()
    path = study.save_prediction(native, tmp_path / "N0")
    loaded = study.load_prediction(path)
    assert loaded.prediction_key == native.prediction_key
    assert loaded.locked
    changed = study.relabel_prediction(loaded, "S_NATIVE_AREA", "S", {7: 5}, {"attempts": 3})
    assert changed.instance_ranks == loaded.instance_ranks
    np.testing.assert_array_equal(changed.owner_ids, loaded.owner_ids)
    np.testing.assert_array_equal(changed.semantic_labels, [5, 5, 0])
    with (tmp_path / "N0/prediction.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="prediction array"):
        study.load_prediction(path)


def test_saved_native_readout_keeps_last_eight_and_requires_two_observations():
    saved = {
        7: {"feat": np.array([[1000, 0]] * 2 + [[0, 1]] * 8, dtype=np.float32),
            "vis_area": np.arange(1, 11), "frame_id": list(range(10))},
        8: {"feat": np.array([[1, 0]], dtype=np.float32), "vis_area": np.array([100]), "frame_id": [0]},
        9: {"feat": np.array([[0, 1], [1, 0]], dtype=np.float32),
            "vis_area": np.array([1, 10]), "frame_id": [0, 1]},
    }
    # Canonical-relative native classification is monotone in cosine similarity.
    rows = study.native_readout(saved, np.eye(2, dtype=np.float32),
                                np.array([[1, 1]], dtype=np.float32), (5, 42))
    assert rows[7]["class_id"] == 42
    np.testing.assert_allclose(rows[7]["feature"], [0, 1], atol=1e-7)
    assert rows[7]["used_frames"] == list(range(2, 10))
    assert rows[8]["class_id"] == 0
    assert rows[8]["status"] == "INSUFFICIENT_NATIVE_OBSERVATIONS"
    assert rows[9]["class_id"] == 5
    assert all(row["retained_observations"] <= 10 for row in rows.values())


def test_native_ranks_use_projected_area_within_native_class_and_six_decimals():
    owners = np.array([7, 8, 9, 10, 0])
    nearest = np.repeat(np.arange(5), [301, 100, 200, 99, 10])
    matched = np.ones(len(nearest), bool)
    ranks = study.native_ranks(owners, {7: 5, 8: 5, 9: 42, 10: 5}, nearest, matched)
    assert dict(ranks) == {7: 1.0, 8: 0.332226, 9: 1.0, 10: 0.0}
    # Unmatched coordinates must never index source rows, even with sentinel IDs.
    nearest[-10:] = -999
    matched[-10:] = False
    assert study.native_ranks(owners, {7: 5, 8: 5, 9: 42, 10: 5}, nearest, matched) == ranks


def test_projection_uses_float32_strict_distance_and_frozen_identity(tmp_path: Path):
    xyz = np.array([[0, 0, 0], [1, 0, 0]], np.float32)
    target = np.array([[0, 0, 0], [0.049, 0, 0], [0.051, 0, 0], [1, 0, 0]], np.float32)
    first = study.freeze_projection(xyz, target, tmp_path / "projection")
    np.testing.assert_array_equal(first["nearest"], [0, 0, 0, 1])
    np.testing.assert_array_equal(first["matched"], [True, True, False, True])
    assert study.freeze_projection(xyz, target, tmp_path / "projection")["identity"] == first["identity"]
    with pytest.raises(ValueError, match="projection input"):
        study.freeze_projection(xyz + 1, target, tmp_path / "projection")


def test_native_surface_readout_reproduces_triangle_first_color_and_unavailable():
    colors = np.array([[255, 0, 0], [0, 0, 255], [0, 0, 0],
                       [0, 0, 255], [255, 0, 0], [255, 0, 0]], np.uint8)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    saved = {7: {"color": np.array([255, 0, 0])}, 8: {"color": np.array([0, 0, 255])}}
    rows = {7: {"class_id": 5, "status": "AVAILABLE"},
            8: {"class_id": 0, "status": "INSUFFICIENT_NATIVE_OBSERVATIONS"}}
    owners, labels = study.native_surface_readout(colors, faces, saved, rows)
    np.testing.assert_array_equal(owners, [7, 7, 7, 0, 0, 0])
    np.testing.assert_array_equal(labels, [5, 5, 5, 0, 0, 0])
