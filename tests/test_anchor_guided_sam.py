"""Tests for pure anchor-guided SAM proposal construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import Anchor2D, CameraIntrinsics, Frame, Proposal2D
from src.modules.anchor_guided_sam import AnchorGuidedSAMModule, MaskAnchorRelation
from src.modules.proposal import ProposalModule


def _frame() -> Frame:
    return Frame(
        frame_id=0,
        rgb=np.zeros((12, 12, 3), dtype=np.uint8),
        depth=np.ones((12, 12), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=6.0, cy=6.0, width=12, height=12),
    )


def _anchor() -> Anchor2D:
    return Anchor2D(
        anchor_id=3,
        bbox_xyxy=np.array([2, 2, 8, 8], dtype=np.float32),
        class_name="sofa",
        confidence=0.87,
    )


def _proposal() -> Proposal2D:
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:8, 2:8] = True
    return Proposal2D(
        proposal_id=40,
        mask=mask,
        bbox_xyxy=np.array([2, 2, 8, 8], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.91,
        backend_name="sam",
    )


def _proposal_from_bbox(
    proposal_id: int,
    bbox_xyxy: list[int],
    confidence: float = 0.91,
    metadata: dict | None = None,
    shape: tuple[int, int] = (12, 12),
) -> Proposal2D:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = bbox_xyxy
    mask[y1:y2, x1:x2] = True
    return Proposal2D(
        proposal_id=proposal_id,
        mask=mask,
        bbox_xyxy=np.array(bbox_xyxy, dtype=np.float32),
        area=int(mask.sum()),
        confidence=confidence,
        backend_name="sam",
        metadata=dict(metadata or {}),
    )


def test_anchor_guided_sam_labels_scale_compatible_mask() -> None:
    module = AnchorGuidedSAMModule({"enabled": True, "proposal_min_area": 1})

    proposals, assignments, debug = module.build_proposals(
        frame=_frame(),
        anchors=[_anchor()],
        sam_proposals=[_proposal()],
    )

    assert len(proposals) == 1
    assert len(assignments) == 1

    proposal = proposals[0]
    assert proposal.proposal_id == 3
    assert proposal.metadata["source"] == "anchor_guided_sam"
    assert proposal.metadata["observation_layer"] == "fine"
    assert proposal.metadata["anchor_id"] == 3
    assert proposal.metadata["anchor_class_name"] == "sofa"
    assert proposal.metadata["anchor_confidence"] == 0.87
    assert proposal.metadata["anchor_label_strength"] == "strong"
    assert proposal.metadata["anchor_keepalive"] is True
    assert proposal.metadata["anchor_label_votes"] == {"sofa": 0.87}
    assert proposal.metadata["semantic_commit_allowed"] is True
    assert proposal.metadata["source_raw_proposal_ids"] == [40]
    assert proposal.metadata["mask_anchor_relation"] == "scale_compatible"
    assert proposal.metadata["proposal_anchor_coverage"] == 1.0
    assert proposal.metadata["anchor_proposal_coverage"] == 1.0
    assert proposal.metadata["mask_source"] == "anchor_guided_sam_selected"

    assignment = assignments[0]
    assert assignment.proposal_id == 3
    assert assignment.anchor_id == 3
    assert assignment.class_name == "sofa"
    assert assignment.confidence == 0.87
    assert assignment.keepalive is True

    assert debug["anchored_proposal_count"] == 1
    assert debug["unknown_residual_count"] == 0


def test_anchor_guided_sam_does_not_label_small_contained_mask_as_parent() -> None:
    module = AnchorGuidedSAMModule(
        {
            "enabled": True,
            "proposal_min_area": 1,
            "min_proposal_anchor_coverage_for_label": 0.70,
            "min_anchor_proposal_coverage_for_label": 0.30,
            "contained_residual_enabled": True,
            "contained_min_proposal_coverage": 0.85,
            "contained_max_anchor_coverage": 0.20,
            "clip_to_anchor_box": True,
            "include_unknown_residuals": True,
            "include_anchor_box_fallbacks": True,
        }
    )
    sofa_anchor = Anchor2D(
        anchor_id=7,
        bbox_xyxy=np.array([1, 1, 11, 11], dtype=np.float32),
        class_name="sofa",
        confidence=0.82,
    )
    contained_proposal = _proposal_from_bbox(99, [4, 4, 7, 7])

    proposals, assignments, debug = module.build_proposals(
        frame=_frame(),
        anchors=[sofa_anchor],
        sam_proposals=[contained_proposal],
    )

    parent_fallbacks = [
        proposal
        for proposal in proposals
        if proposal.metadata.get("mask_anchor_relation") == "anchor_box_fallback"
    ]
    assert len(parent_fallbacks) == 1
    assert parent_fallbacks[0].metadata["anchor_label_strength"] == "strong"
    assert parent_fallbacks[0].metadata["anchor_class_name"] == "sofa"

    residuals = [
        proposal
        for proposal in proposals
        if proposal.metadata.get("residual_semantic_policy") == "unknown"
    ]
    assert len(residuals) == 1
    residual = residuals[0]
    assert residual.metadata["anchor_id"] == -1
    assert residual.metadata["anchor_class_name"] == ""
    assert residual.metadata["anchor_label_strength"] == "none"
    assert residual.metadata["semantic_commit_allowed"] is False
    assert residual.metadata["residual_semantic_policy"] == "unknown"
    assert residual.metadata["nearby_anchor_class_name"] == "sofa"
    assert residual.metadata["source_raw_proposal_ids"] == [99]
    assert residual.metadata["observation_layer"] == "residual"

    residual_assignments = [assignment for assignment in assignments if assignment.proposal_id == residual.proposal_id]
    assert all(assignment.class_name != "sofa" for assignment in residual_assignments)

    assert debug["unknown_residual_count"] == 1
    assert debug["semantic_blocked_residual_count"] == 1


def test_anchor_guided_sam_labels_contained_mask_when_matching_child_anchor_exists() -> None:
    module = AnchorGuidedSAMModule(
        {
            "enabled": True,
            "proposal_min_area": 1,
            "min_proposal_anchor_coverage_for_label": 0.70,
            "min_anchor_proposal_coverage_for_label": 0.30,
            "contained_residual_enabled": True,
            "contained_min_proposal_coverage": 0.85,
            "contained_max_anchor_coverage": 0.20,
            "include_unknown_residuals": True,
            "include_anchor_box_fallbacks": True,
        }
    )
    sofa_anchor = Anchor2D(
        anchor_id=4,
        bbox_xyxy=np.array([1, 1, 11, 11], dtype=np.float32),
        class_name="sofa",
        confidence=0.9,
    )
    blanket_anchor = Anchor2D(
        anchor_id=5,
        bbox_xyxy=np.array([4, 4, 7, 7], dtype=np.float32),
        class_name="blanket",
        confidence=0.8,
    )
    contained_proposal = _proposal_from_bbox(99, [4, 4, 7, 7])

    proposals, assignments, debug = module.build_proposals(
        frame=_frame(),
        anchors=[sofa_anchor, blanket_anchor],
        sam_proposals=[contained_proposal],
    )

    blanket_proposals = [
        proposal
        for proposal in proposals
        if proposal.metadata.get("anchor_class_name") == "blanket"
    ]
    assert len(blanket_proposals) == 1
    blanket_proposal = blanket_proposals[0]
    assert blanket_proposal.metadata["semantic_commit_allowed"] is True
    assert blanket_proposal.metadata["source_raw_proposal_ids"] == [99]
    assert blanket_proposal.metadata["mask_anchor_relation"] == "scale_compatible"

    assert not any(
        proposal.metadata.get("anchor_label_strength") == "none"
        for proposal in proposals
    )
    assert any(
        assignment.anchor_id == 5 and assignment.class_name == "blanket"
        for assignment in assignments
    )
    assert debug["unknown_residual_count"] == 0


def test_anchor_guided_sam_labels_contained_mask_when_child_anchor_comes_first() -> None:
    module = AnchorGuidedSAMModule(
        {
            "enabled": True,
            "proposal_min_area": 1,
            "min_proposal_anchor_coverage_for_label": 0.70,
            "min_anchor_proposal_coverage_for_label": 0.30,
            "contained_residual_enabled": True,
            "contained_min_proposal_coverage": 0.85,
            "contained_max_anchor_coverage": 0.20,
            "include_unknown_residuals": True,
            "include_anchor_box_fallbacks": True,
        }
    )
    sofa_anchor = Anchor2D(
        anchor_id=4,
        bbox_xyxy=np.array([1, 1, 11, 11], dtype=np.float32),
        class_name="sofa",
        confidence=0.9,
    )
    blanket_anchor = Anchor2D(
        anchor_id=5,
        bbox_xyxy=np.array([4, 4, 7, 7], dtype=np.float32),
        class_name="blanket",
        confidence=0.8,
    )
    contained_proposal = _proposal_from_bbox(99, [4, 4, 7, 7])

    proposals, assignments, debug = module.build_proposals(
        frame=_frame(),
        anchors=[blanket_anchor, sofa_anchor],
        sam_proposals=[contained_proposal],
    )

    blanket_proposals = [
        proposal
        for proposal in proposals
        if proposal.metadata.get("anchor_class_name") == "blanket"
    ]
    assert len(blanket_proposals) == 1
    blanket_proposal = blanket_proposals[0]
    assert blanket_proposal.metadata["source_raw_proposal_ids"] == [99]
    assert blanket_proposal.metadata["mask_anchor_relation"] == "scale_compatible"

    assert not any(
        proposal.metadata.get("anchor_label_strength") == "none"
        for proposal in proposals
    )
    assert any(
        assignment.anchor_id == 5 and assignment.class_name == "blanket"
        for assignment in assignments
    )
    assert debug["unknown_residual_count"] == 0


def test_anchor_guided_sam_tie_breaks_equal_strong_anchors_by_lower_anchor_id() -> None:
    module = AnchorGuidedSAMModule(
        {
            "enabled": True,
            "proposal_min_area": 1,
            "include_anchor_box_fallbacks": True,
        }
    )
    chair_anchor = Anchor2D(
        anchor_id=4,
        bbox_xyxy=np.array([2, 2, 8, 8], dtype=np.float32),
        class_name="chair",
        confidence=0.9,
    )
    sofa_anchor = Anchor2D(
        anchor_id=5,
        bbox_xyxy=np.array([2, 2, 8, 8], dtype=np.float32),
        class_name="sofa",
        confidence=0.8,
    )
    proposal = _proposal_from_bbox(99, [2, 2, 8, 8])

    proposals, assignments, _ = module.build_proposals(
        frame=_frame(),
        anchors=[chair_anchor, sofa_anchor],
        sam_proposals=[proposal],
    )

    sam_owned_proposal = next(
        item
        for item in proposals
        if item.metadata.get("source_raw_proposal_ids") == [99]
    )
    assert sam_owned_proposal.metadata["anchor_id"] == 4
    assert sam_owned_proposal.metadata["anchor_class_name"] == "chair"
    assert sam_owned_proposal.metadata["mask_anchor_relation"] == "scale_compatible"

    anchor_five_proposals = [
        item
        for item in proposals
        if item.metadata.get("anchor_id") == 5
    ]
    assert all(
        item.metadata.get("source_raw_proposal_ids") == []
        for item in anchor_five_proposals
    )
    assert any(
        assignment.anchor_id == 4 and assignment.proposal_id == sam_owned_proposal.proposal_id
        for assignment in assignments
    )


def test_anchor_guided_sam_contained_residual_wins_when_default_scale_thresholds_overlap() -> None:
    module = AnchorGuidedSAMModule({"enabled": True, "proposal_min_area": 1})
    sofa_anchor = Anchor2D(
        anchor_id=7,
        bbox_xyxy=np.array([1, 1, 11, 11], dtype=np.float32),
        class_name="sofa",
        confidence=0.82,
    )
    contained_proposal = _proposal_from_bbox(99, [1, 1, 6, 6])

    proposals, assignments, debug = module.build_proposals(
        frame=_frame(),
        anchors=[sofa_anchor],
        sam_proposals=[contained_proposal],
    )

    assert [proposal.metadata.get("mask_anchor_relation") for proposal in proposals] == [
        "anchor_box_fallback",
        "contained_residual",
    ]
    assert not any(
        proposal.metadata.get("observation_layer") == "fine"
        and proposal.metadata.get("anchor_class_name") == "sofa"
        for proposal in proposals
    )
    assert all(assignment.proposal_id != contained_proposal.proposal_id for assignment in assignments)
    assert debug["unknown_residual_count"] == 1
    assert debug["semantic_blocked_residual_count"] == 1


def test_anchor_guided_sam_residual_proposal_id_does_not_collide_with_anchor_id() -> None:
    module = AnchorGuidedSAMModule({"enabled": True, "proposal_min_area": 1})
    sofa_anchor = Anchor2D(
        anchor_id=99,
        bbox_xyxy=np.array([1, 1, 11, 11], dtype=np.float32),
        class_name="sofa",
        confidence=0.82,
    )
    contained_proposal = _proposal_from_bbox(99, [4, 4, 7, 7])

    proposals, _, _ = module.build_proposals(
        frame=_frame(),
        anchors=[sofa_anchor],
        sam_proposals=[contained_proposal],
    )

    proposal_ids = [proposal.proposal_id for proposal in proposals]
    assert len(proposal_ids) == len(set(proposal_ids))
    fallback = next(
        proposal
        for proposal in proposals
        if proposal.metadata.get("mask_anchor_relation") == "anchor_box_fallback"
    )
    residual = next(
        proposal
        for proposal in proposals
        if proposal.metadata.get("mask_anchor_relation") == "contained_residual"
    )
    assert fallback.proposal_id == 99
    assert residual.proposal_id != fallback.proposal_id
    assert residual.proposal_id >= 1_000_000_000_000
    assert residual.metadata["source_raw_proposal_ids"] == [99]


def test_anchor_guided_sam_residual_proposal_id_does_not_collide_across_anchor_proposal_pairs() -> None:
    module = AnchorGuidedSAMModule({"enabled": True, "proposal_min_area": 1})

    first_id = module._residual_proposal_id(
        MaskAnchorRelation(
            proposal=_proposal_from_bbox(10_000, [4, 4, 7, 7]),
            anchor=Anchor2D(
                anchor_id=0,
                bbox_xyxy=np.array([1, 1, 11, 11], dtype=np.float32),
                class_name="sofa",
                confidence=0.82,
            ),
            overlap_area=9,
            proposal_anchor_coverage=1.0,
            anchor_proposal_coverage=0.09,
            bbox_iou=0.09,
            relation="contained_residual",
        )
    )
    second_id = module._residual_proposal_id(
        MaskAnchorRelation(
            proposal=_proposal_from_bbox(0, [4, 4, 7, 7]),
            anchor=Anchor2D(
                anchor_id=1,
                bbox_xyxy=np.array([1, 1, 11, 11], dtype=np.float32),
                class_name="sofa",
                confidence=0.82,
            ),
            overlap_area=9,
            proposal_anchor_coverage=1.0,
            anchor_proposal_coverage=0.09,
            bbox_iou=0.09,
            relation="contained_residual",
        )
    )

    assert first_id != second_id


def test_anchor_guided_sam_residual_proposal_id_does_not_wrap_large_or_negative_ids() -> None:
    module = AnchorGuidedSAMModule({"enabled": True, "proposal_min_area": 1})

    ids = [
        module._residual_proposal_id(
            MaskAnchorRelation(
                proposal=_proposal_from_bbox(proposal_id, [4, 4, 7, 7]),
                anchor=Anchor2D(
                    anchor_id=anchor_id,
                    bbox_xyxy=np.array([1, 1, 11, 11], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.82,
                ),
                overlap_area=9,
                proposal_anchor_coverage=1.0,
                anchor_proposal_coverage=0.09,
                bbox_iou=0.09,
                relation="contained_residual",
            )
        )
        for anchor_id, proposal_id in [
            (0, 0),
            (0, 2**32),
            (-1, 0),
            (2**32 - 1, 0),
        ]
    ]

    assert len(set(ids)) == len(ids)


def test_anchor_guided_sam_residual_neutralizes_stale_anchor_metadata() -> None:
    module = AnchorGuidedSAMModule({"enabled": True, "proposal_min_area": 1})
    sofa_anchor = Anchor2D(
        anchor_id=7,
        bbox_xyxy=np.array([1, 1, 11, 11], dtype=np.float32),
        class_name="sofa",
        confidence=0.82,
    )
    contained_proposal = _proposal_from_bbox(
        99,
        [4, 4, 7, 7],
        metadata={
            "anchor_id": 7,
            "anchor_class_name": "sofa",
            "anchor_confidence": 0.82,
            "anchor_label_strength": "strong",
            "anchor_keepalive": True,
            "anchor_label_votes": {"sofa": 0.82},
            "anchor_center_inside": True,
            "anchor_bbox_iou": 0.5,
            "semantic_commit_allowed": True,
            "raw_sam_trace_id": "sam-99",
        },
    )

    proposals, _, _ = module.build_proposals(
        frame=_frame(),
        anchors=[sofa_anchor],
        sam_proposals=[contained_proposal],
    )

    residual = next(
        proposal
        for proposal in proposals
        if proposal.metadata.get("mask_anchor_relation") == "contained_residual"
    )
    assert residual.metadata["anchor_id"] == -1
    assert residual.metadata["anchor_class_name"] == ""
    assert residual.metadata["anchor_confidence"] == 0.0
    assert residual.metadata["anchor_label_strength"] == "none"
    assert residual.metadata["anchor_keepalive"] is False
    assert residual.metadata["anchor_label_votes"] == {}
    assert residual.metadata["anchor_center_inside"] is False
    assert residual.metadata["anchor_bbox_iou"] == 0.0
    assert residual.metadata["semantic_commit_allowed"] is False
    assert residual.metadata["residual_semantic_policy"] == "unknown"
    assert residual.metadata["raw_sam_trace_id"] == "sam-99"


def test_anchor_guided_sam_skips_shape_mismatched_sam_proposal() -> None:
    module = AnchorGuidedSAMModule({"enabled": True, "proposal_min_area": 1})
    mismatched_proposal = _proposal_from_bbox(99, [1, 1, 4, 4], shape=(10, 10))

    proposals, assignments, debug = module.build_proposals(
        frame=_frame(),
        anchors=[_anchor()],
        sam_proposals=[mismatched_proposal],
    )

    assert len(proposals) == 1
    assert proposals[0].metadata["mask_anchor_relation"] == "anchor_box_fallback"
    assert len(assignments) == 1
    assert debug["dropped_proposal_count"] == 1
    assert debug["unknown_residual_count"] == 0


def test_anchor_guided_sam_fractional_anchor_bbox_fallback_uses_floor_ceil_rasterization() -> None:
    module = AnchorGuidedSAMModule({"enabled": True, "proposal_min_area": 1})
    anchor = Anchor2D(
        anchor_id=7,
        bbox_xyxy=np.array([1.6, 1.6, 3.2, 3.2], dtype=np.float32),
        class_name="sofa",
        confidence=0.82,
    )

    proposals, _, _ = module.build_proposals(
        frame=_frame(),
        anchors=[anchor],
        sam_proposals=[],
    )

    assert len(proposals) == 1
    assert proposals[0].area == 9
    np.testing.assert_array_equal(proposals[0].bbox_xyxy, np.array([1, 1, 4, 4], dtype=np.float32))


def test_proposal_module_process_for_anchors_delegates_to_backend() -> None:
    class Backend:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def generate_proposals_for_anchors(
            self,
            rgb: np.ndarray,
            depth: np.ndarray,
            anchors: list[Anchor2D],
            frame: Frame | None = None,
        ) -> list[Proposal2D]:
            self.calls.append((len(anchors), frame.frame_id if frame else -1))
            return [_proposal_from_bbox(7, [2, 2, 5, 5])]

    frame = _frame()
    backend = Backend()
    module = object.__new__(ProposalModule)
    module.config = {"min_mask_area": 1}
    module.requested_backend_name = "fake"
    module.active_backend_name = "fake"
    module.backend_init_error = None
    module.backend = backend

    proposals = module.process_for_anchors(frame.rgb, frame.depth, anchors=[_anchor()], frame=frame)

    assert backend.calls == [(1, frame.frame_id)]
    assert len(proposals) == 1
    assert proposals[0].metadata["requested_backend"] == "fake"
    assert proposals[0].metadata["actual_backend"] == "fake"
