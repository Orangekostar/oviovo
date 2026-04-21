"""Tests for visualization helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import Proposal2D
from src.utils.visualization import proposal_overlay_image, runtime_proposal_class_overlay_image


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
