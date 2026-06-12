"""Tests for visualization helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import Proposal2D
from src.utils.visualization import (
    anchor_vote_overlay_image,
    proposal_overlay_image,
    runtime_proposal_class_overlay_image,
    save_multi_panel,
)


def test_proposal_overlay_image_renders_rgb_output() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 1:5] = True
    proposal = Proposal2D(
        proposal_id=0,
        mask=mask,
        bbox_xyxy=np.array([1, 2, 5, 6], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
    )

    image = proposal_overlay_image(rgb, [proposal])
    arr = np.asarray(image)

    assert arr.shape == (8, 8, 3)
    assert arr.dtype == np.uint8
    assert arr.sum() > 0


def test_runtime_proposal_class_overlay_image_renders_rgb_output() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 1:5] = True
    proposal = Proposal2D(
        proposal_id=0,
        mask=mask,
        bbox_xyxy=np.array([1, 2, 5, 6], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
    )
    profiles = [
        {
            "proposal_id": 0,
            "proposal_class": "object_like",
            "objectness_score": 0.8,
            "backgroundness_score": 0.1,
        }
    ]

    image = runtime_proposal_class_overlay_image(rgb, [proposal], profiles)
    arr = np.asarray(image)

    assert arr.shape == (8, 8, 3)
    assert arr.dtype == np.uint8
    assert arr.sum() > 0


def test_anchor_vote_overlay_image_labels_votes_without_changing_shape() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 1:5] = True
    proposal = Proposal2D(
        proposal_id=0,
        mask=mask,
        bbox_xyxy=np.array([1, 2, 5, 6], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
        metadata={
            "anchor_class_name": "chair",
            "anchor_vote_score": 0.88,
            "anchor_id": 3,
        },
    )

    image = anchor_vote_overlay_image(rgb, [proposal])
    arr = np.asarray(image)

    assert arr.shape == (8, 8, 3)
    assert arr.dtype == np.uint8
    assert arr.sum() > 0


def test_save_multi_panel_writes_horizontal_panel(tmp_path) -> None:
    panels = [
        Image.new("RGB", (5, 4), color=(255, 0, 0)),
        Image.new("RGB", (5, 4), color=(0, 255, 0)),
        Image.new("RGB", (5, 4), color=(0, 0, 255)),
        Image.new("RGB", (5, 4), color=(255, 255, 0)),
    ]
    output_path = tmp_path / "panel.png"

    save_multi_panel(panels, output_path)

    with Image.open(output_path) as image:
        assert image.size == (20, 4)
