"""Tests for the structural dense overlay layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import (
    Anchor2D,
    CameraIntrinsics,
    Frame,
    Proposal2D,
    StructuralOverlayMap,
    StructuralOverlayVoxel,
    SystemState,
)
from src.modules.structural_overlay import StructuralOverlayModule


def test_structural_overlay_voxel_returns_top_label_and_support() -> None:
    voxel = StructuralOverlayVoxel()
    voxel.add_vote("wall", 0.25, frame_id=3)
    voxel.add_vote("window", 0.75, frame_id=4)
    voxel.add_vote("wall", 0.75, frame_id=5)

    assert voxel.top_label == "wall"
    assert voxel.top_support == 1.0
    assert voxel.observation_count == 3
    assert voxel.last_seen_frame == 5


def test_system_state_owns_empty_structural_overlay_map() -> None:
    state = SystemState()

    assert isinstance(state.structural_overlay_map, StructuralOverlayMap)
    assert state.structural_overlay_map.voxel_size == 0.05
    assert state.structural_overlay_map.voxels == {}


def _frame(height: int = 8, width: int = 8) -> Frame:
    return Frame(
        frame_id=7,
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        depth=np.ones((height, width), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=0.0, cy=0.0, width=width, height=height),
        source_frame_id=70,
    )


def test_structural_overlay_clips_proposal_to_anchor_box() -> None:
    frame = _frame()
    anchor = Anchor2D(
        anchor_id=1,
        bbox_xyxy=np.array([2, 2, 5, 5], dtype=np.float32),
        class_name="window",
        confidence=0.8,
    )
    mask = np.ones((8, 8), dtype=bool)
    proposal = Proposal2D(
        proposal_id=10,
        mask=mask,
        bbox_xyxy=np.array([0, 0, 8, 8], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
        backend_name="precomputed",
    )
    overlay = StructuralOverlayMap(voxel_size=0.001)
    module = StructuralOverlayModule(
        {
            "enabled": True,
            "classes": ["window"],
            "voxel_size": 0.001,
            "min_overlap_area": 1,
            "min_proposal_anchor_coverage": 0.01,
            "min_anchor_proposal_coverage": 0.01,
            "clip_to_anchor_box": True,
        }
    )

    summary = module.process(frame, [anchor], [proposal], overlay)

    assert summary["enabled"] is True
    assert summary["structure_anchor_count"] == 1
    assert summary["accepted_pair_count"] == 1
    assert summary["voted_pixel_count"] == 9
    assert len(overlay.voxels) == 9
    assert {voxel.top_label for voxel in overlay.voxels.values()} == {"window"}


def test_structural_overlay_ignores_non_structure_anchors() -> None:
    frame = _frame()
    anchor = Anchor2D(
        anchor_id=2,
        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
        class_name="chair",
        confidence=0.9,
    )
    mask = np.ones((8, 8), dtype=bool)
    proposal = Proposal2D(3, mask, np.array([0, 0, 8, 8], dtype=np.float32), int(mask.sum()))
    overlay = StructuralOverlayMap(voxel_size=0.01)
    module = StructuralOverlayModule({"enabled": True, "classes": ["wall"], "min_overlap_area": 1})

    summary = module.process(frame, [anchor], [proposal], overlay)

    assert summary["structure_anchor_count"] == 0
    assert summary["skip_reason"] == "no_structure_anchors"
    assert overlay.voxels == {}


def test_structural_overlay_skips_mismatched_proposal_masks() -> None:
    frame = _frame(height=8, width=8)
    anchor = Anchor2D(1, np.array([0, 0, 8, 8], dtype=np.float32), "wall", 0.7)
    bad_mask = np.ones((4, 4), dtype=bool)
    proposal = Proposal2D(5, bad_mask, np.array([0, 0, 4, 4], dtype=np.float32), int(bad_mask.sum()))
    overlay = StructuralOverlayMap(voxel_size=0.01)
    module = StructuralOverlayModule({"enabled": True, "classes": ["wall"], "min_overlap_area": 1})

    summary = module.process(frame, [anchor], [proposal], overlay)

    assert summary["mismatched_mask_count"] == 1
    assert summary["accepted_pair_count"] == 0
    assert overlay.voxels == {}


def test_structural_overlay_counts_each_mismatched_proposal_once() -> None:
    frame = _frame(height=8, width=8)
    anchors = [
        Anchor2D(1, np.array([0, 0, 8, 8], dtype=np.float32), "wall", 0.7),
        Anchor2D(2, np.array([0, 0, 8, 8], dtype=np.float32), "floor", 0.8),
    ]
    bad_mask = np.ones((4, 4), dtype=bool)
    proposal = Proposal2D(5, bad_mask, np.array([0, 0, 4, 4], dtype=np.float32), int(bad_mask.sum()))
    overlay = StructuralOverlayMap(voxel_size=0.01)
    module = StructuralOverlayModule({"enabled": True, "classes": ["wall", "floor"], "min_overlap_area": 1})

    summary = module.process(frame, anchors, [proposal], overlay)

    assert summary["mismatched_mask_count"] == 1
    assert summary["accepted_pair_count"] == 0
    assert overlay.voxels == {}


def test_structural_overlay_reports_voxels_touched_in_current_process_call() -> None:
    frame = _frame(height=3, width=3)
    anchor = Anchor2D(1, np.array([0, 0, 2, 2], dtype=np.float32), "wall", 1.0)
    mask = np.zeros((3, 3), dtype=bool)
    mask[0:2, 0:2] = True
    proposal = Proposal2D(6, mask, np.array([0, 0, 2, 2], dtype=np.float32), int(mask.sum()), confidence=1.0)
    overlay = StructuralOverlayMap(voxel_size=0.1)
    overlay.voxels[(0, 0, 10)] = StructuralOverlayVoxel(label_votes={"existing": 2.0})
    overlay.voxels[(99, 99, 99)] = StructuralOverlayVoxel(label_votes={"untouched": 1.0})
    module = StructuralOverlayModule(
        {
            "enabled": True,
            "classes": ["wall"],
            "voxel_size": 0.1,
            "min_overlap_area": 1,
            "min_proposal_anchor_coverage": 0.01,
            "min_anchor_proposal_coverage": 0.01,
        }
    )

    summary = module.process(frame, [anchor], [proposal], overlay)

    assert summary["accepted_pair_count"] == 1
    assert summary["voted_pixel_count"] == 4
    assert summary["updated_voxel_count"] == 4
    assert summary["new_voxel_count"] == 3
    assert len(overlay.voxels) == 5
