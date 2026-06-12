from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import Anchor2D, CameraIntrinsics, Frame, Proposal2D
from src.modules.async_refinement import AsyncRefinementModule
from src.modules.depth_refinement import DepthRefinementModule
from src.modules.patch_lifting import PatchLiftingModule


def test_async_refinement_rejects_positive_delay_frames() -> None:
    with pytest.raises(ValueError, match="delay_frames"):
        AsyncRefinementModule({"enabled": True, "delay_frames": 1})


def test_async_refinement_rejects_fractional_nonzero_delay_frames() -> None:
    with pytest.raises(ValueError, match="delay_frames"):
        AsyncRefinementModule({"enabled": True, "delay_frames": 0.5})


def _frame() -> Frame:
    return Frame(
        frame_id=4,
        source_frame_id=40,
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth=np.ones((10, 10), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=5.0, cy=5.0, width=10, height=10),
    )


def _proposal(proposal_id: int, y1: int, y2: int, x1: int, x2: int) -> Proposal2D:
    mask = np.zeros((10, 10), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return Proposal2D(
        proposal_id=proposal_id,
        mask=mask,
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.8,
        backend_name="sam2",
        metadata={"source": "sam2"},
    )


def test_async_refinement_selects_sam_masks_by_anchor_overlap() -> None:
    module = AsyncRefinementModule(
        {
            "enabled": True,
            "min_proposal_anchor_coverage": 0.20,
            "min_anchor_proposal_coverage": 0.20,
            "proposal_min_area": 1,
        }
    )
    anchor = Anchor2D(
        anchor_id=2,
        bbox_xyxy=np.array([2, 2, 8, 8], dtype=np.float32),
        class_name="sofa",
        confidence=0.9,
    )
    fine, debug = module.build_fine_proposals(
        frame=_frame(),
        anchors=[anchor],
        sam_proposals=[
            _proposal(10, 2, 8, 2, 8),
            _proposal(11, 0, 2, 0, 2),
        ],
    )

    assert len(fine) == 1
    assert fine[0].proposal_id == 2
    assert fine[0].metadata["observation_layer"] == "fine"
    assert fine[0].metadata["refinement_key"] == "40:2"
    assert fine[0].metadata["anchor_class_name"] == "sofa"
    assert fine[0].metadata["source_raw_proposal_ids"] == [10]
    assert debug["matched_anchor_count"] == 1
    assert debug["fine_proposal_count"] == 1


def test_async_refinement_preserves_unanchored_sam_masks_as_context_only() -> None:
    module = AsyncRefinementModule({"enabled": True, "proposal_min_area": 1})
    fine, debug = module.build_fine_proposals(
        frame=_frame(),
        anchors=[],
        sam_proposals=[_proposal(10, 2, 8, 2, 8)],
    )

    assert fine == []
    assert debug["unmatched_sam_proposal_count"] == 1


def test_async_refinement_sorts_source_sam_metadata_by_proposal_id() -> None:
    module = AsyncRefinementModule(
        {
            "enabled": True,
            "min_proposal_anchor_coverage": 0.20,
            "min_anchor_proposal_coverage": 0.20,
            "proposal_min_area": 1,
        }
    )
    anchor = Anchor2D(
        anchor_id=2,
        bbox_xyxy=np.array([2, 2, 8, 8], dtype=np.float32),
        class_name="sofa",
        confidence=0.9,
    )
    later = _proposal(12, 2, 5, 2, 5)
    later.confidence = 0.7
    earlier = _proposal(10, 5, 8, 5, 8)
    earlier.confidence = 0.5

    fine, _debug = module.build_fine_proposals(
        frame=_frame(),
        anchors=[anchor],
        sam_proposals=[later, earlier],
    )

    assert len(fine) == 1
    assert fine[0].metadata["source_raw_proposal_ids"] == [10, 12]
    assert fine[0].metadata["source_raw_proposal_confidences"] == [0.5, 0.7]


def test_async_refinement_fine_provenance_survives_depth_and_patch_lifting() -> None:
    module = AsyncRefinementModule({"enabled": True, "proposal_min_area": 1})
    anchor = Anchor2D(
        anchor_id=2,
        bbox_xyxy=np.array([2, 2, 8, 8], dtype=np.float32),
        class_name="sofa",
        confidence=0.9,
    )
    frame = _frame()
    fine, _debug = module.build_fine_proposals(
        frame=frame,
        anchors=[anchor],
        sam_proposals=[_proposal(10, 2, 8, 2, 8)],
    )
    refined = DepthRefinementModule(
        {"depth_edge_threshold": 10.0, "min_mask_area_after_refine": 1}
    ).process(frame.depth, fine)

    patches = PatchLiftingModule({"min_points": 1}).process(
        refined,
        frame.depth,
        frame.pose,
        frame.intrinsics,
        frame_id=int(frame.source_frame_id),
    )

    assert len(patches) == 1
    assert patches[0].patch_id == 2
    assert patches[0].metadata["observation_layer"] == "fine"
    assert patches[0].metadata["refinement_key"] == "40:2"
    assert patches[0].metadata["mask_source"] == "delayed_sam_union"
    assert patches[0].metadata["source_raw_proposal_ids"] == [10]
