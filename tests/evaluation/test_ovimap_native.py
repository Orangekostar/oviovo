from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest


def _native():
    return importlib.import_module("src.evaluation.baselines.ovimap_native")


def test_compose_instance_id_mask_uses_score_order() -> None:
    masks = np.array(
        [
            [[1, 1], [0, 0]],
            [[0, 1], [0, 1]],
        ],
        dtype=bool,
    )

    result = _native().compose_instance_id_mask(masks, np.array([0.9, 0.6]))

    assert result.dtype == np.uint8
    assert result.tolist() == [[1, 1], [0, 2]]
    assert np.unique(result).tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("masks", "scores", "message"),
    [
        (np.zeros((2, 3), dtype=bool), np.array([0.5, 0.4]), r"\[N,H,W\]"),
        (np.zeros((2, 3, 4), dtype=np.uint8), np.array([0.5, 0.4]), "boolean"),
        (np.zeros((2, 3, 4), dtype=bool), np.array([0.5]), "one finite"),
        (np.zeros((2, 3, 4), dtype=bool), np.array([0.5, np.nan]), "one finite"),
    ],
)
def test_compose_instance_id_mask_rejects_invalid_predictions(
    masks: np.ndarray,
    scores: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _native().compose_instance_id_mask(masks, scores)


def test_summarize_instance_id_mask_records_coverage() -> None:
    mask = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint8)

    summary = _native().summarize_instance_id_mask(mask)

    assert summary == {
        "shape": [2, 3],
        "dtype": "uint8",
        "instance_ids": [1, 2],
        "instance_count": 2,
        "foreground_ratio": pytest.approx(4 / 6),
        "largest_instance_ratio": pytest.approx(2 / 6),
    }


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.zeros((2, 3, 3), dtype=np.uint8), "two-dimensional"),
        (np.zeros((2, 3), dtype=np.float32), "integer"),
        (np.array([[0, 2]], dtype=np.uint8), "contiguous"),
    ],
)
def test_summarize_instance_id_mask_rejects_non_id_images(
    mask: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _native().summarize_instance_id_mask(mask)


def test_normalize_raycast_ids_reshapes_exact_flat_buffer() -> None:
    raycast = np.arange(6, dtype=np.uint32)

    result = _native().normalize_raycast_ids(raycast, height=2, width=3)

    assert result.shape == (2, 3)
    assert result.dtype == np.uint32
    assert result.tolist() == [[0, 1, 2], [3, 4, 5]]


@pytest.mark.parametrize(
    "raycast",
    [
        np.arange(5, dtype=np.uint32),
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.zeros((2, 3), dtype=np.float32),
    ],
)
def test_normalize_raycast_ids_rejects_malformed_outputs(raycast: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _native().normalize_raycast_ids(raycast, height=2, width=3)


def test_bbox_from_instance_map_preserves_sparse_extent() -> None:
    ids = np.zeros((4, 5), dtype=np.uint32)
    ids[1:3, 2:4] = 7

    assert _native().bbox_from_instance_map(ids, 7) == [2, 1, 3, 2]


def test_bbox_from_instance_map_rejects_absent_instance() -> None:
    with pytest.raises(ValueError, match="instance 4 is absent"):
        _native().bbox_from_instance_map(np.zeros((2, 3), dtype=np.uint8), 4)


def test_validate_color_round_trip_rejects_channel_collapse() -> None:
    with pytest.raises(ValueError, match="differ"):
        _native().validate_color_round_trip((214, 36, 176), (214, 214, 214))


def test_audit_native_frame_passes_sparse_raycast() -> None:
    mask = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint8)
    raycast = np.array([[0, 5, 5], [0, 0, 0]], dtype=np.uint32)

    result = _native().audit_native_frame(
        mask,
        raycast,
        [{"mapper_rgb": [10, 20, 30], "logged_rgb": [10, 20, 30]}],
    )

    assert result["status"] == "PASS"
    assert result["raycast_instance_count"] == 1
    assert result["full_frame_bbox_count"] == 0
    assert result["boxes"] == [[1, 0, 2, 0]]


def test_audit_native_frame_rejects_all_full_frame_boxes() -> None:
    mask = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    raycast = np.array([[8, 8], [8, 8]], dtype=np.uint32)

    with pytest.raises(ValueError, match="all non-empty raycast boxes are full-frame"):
        _native().audit_native_frame(mask, raycast, [])


def test_audit_cli_writes_hash_bound_json(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    raycast_path = tmp_path / "raycast.npy"
    colors_path = tmp_path / "colors.json"
    output_path = tmp_path / "audit.json"
    assert cv2.imwrite(
        str(mask_path), np.array([[0, 1, 1], [0, 1, 1]], dtype=np.uint8)
    )
    np.save(raycast_path, np.array([[0, 3, 3], [0, 0, 0]], dtype=np.uint32))
    colors_path.write_text(
        json.dumps([{"mapper_rgb": [1, 2, 3], "logged_rgb": [1, 2, 3]}]),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python",
            "scripts/evaluation/audit_ovimap_native_frame.py",
            "--mask",
            str(mask_path),
            "--raycast-npy",
            str(raycast_path),
            "--colors-json",
            str(colors_path),
            "--height",
            "2",
            "--width",
            "3",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert set(payload["inputs"]) == {"mask", "raycast_npy", "colors_json"}
    assert all(len(record["sha256"]) == 64 for record in payload["inputs"].values())
