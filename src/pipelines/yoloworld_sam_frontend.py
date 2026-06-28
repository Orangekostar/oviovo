"""Online YOLOWorld + SAM2 frontend scheduling for anchor-guided Replica runs."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from src.core.data_structures import AnchorAssignment, Frame
from src.pipelines.proposal_bundle import FrameProposalBundle


def _finite_timing(value: Any) -> float:
    try:
        timing = float(value)
    except (TypeError, ValueError):
        return 0.0
    if timing < 0.0 or not np.isfinite(timing):
        return 0.0
    return timing


def _frame_refinement_id(frame: Frame) -> int:
    return int(frame.source_frame_id if frame.source_frame_id is not None else frame.frame_id)


def _stamp_anchor_primary_coarse_metadata(
    frame: Frame,
    proposals: list[Any] | tuple[Any, ...],
    anchor_assignments: list[Any] | tuple[Any, ...],
) -> None:
    anchor_by_proposal_id = {
        int(assignment.proposal_id): int(assignment.anchor_id)
        for assignment in anchor_assignments
        if int(getattr(assignment, "anchor_id", -1)) >= 0
    }
    frame_id = _frame_refinement_id(frame)
    for proposal in proposals:
        proposal.metadata = dict(getattr(proposal, "metadata", {}) or {})
        anchor_id = proposal.metadata.get("anchor_id", anchor_by_proposal_id.get(int(proposal.proposal_id)))
        if anchor_id is None:
            continue
        proposal.metadata["observation_layer"] = "coarse"
        proposal.metadata["refinement_key"] = f"{frame_id}:{int(anchor_id)}"


def _stamp_anchor_prompted_metadata(
    frame: Frame,
    proposals: list[Any] | tuple[Any, ...],
    anchors: list[Any] | tuple[Any, ...],
) -> list[AnchorAssignment]:
    anchors_by_id = {int(anchor.anchor_id): anchor for anchor in anchors}
    frame_id = _frame_refinement_id(frame)
    assignments: list[AnchorAssignment] = []
    for proposal in proposals:
        proposal.metadata = dict(getattr(proposal, "metadata", {}) or {})
        anchor_id = int(proposal.metadata.get("anchor_id", -1))
        anchor = anchors_by_id.get(anchor_id)
        if anchor is None:
            continue
        class_name = str(proposal.metadata.get("anchor_class_name", getattr(anchor, "class_name", "")))
        confidence = float(proposal.metadata.get("anchor_confidence", getattr(anchor, "confidence", 0.0)))
        proposal.metadata.setdefault("source", "anchor_prompted_sam")
        proposal.metadata.setdefault("proposal_source", "anchor_prompted_sam")
        proposal.metadata.setdefault("geometry_source", "sam2_anchor_prompt")
        proposal.metadata.setdefault("mask_source", "sam2_box_prompt")
        proposal.metadata.setdefault("observation_layer", "fine")
        proposal.metadata["refinement_key"] = f"{frame_id}:{anchor_id}"
        proposal.metadata.setdefault("anchor_class", class_name)
        proposal.metadata.setdefault("anchor_class_name", class_name)
        proposal.metadata.setdefault("anchor_confidence", confidence)
        proposal.metadata.setdefault("anchor_label_strength", "strong")
        proposal.metadata.setdefault("anchor_keepalive", True)
        proposal.metadata.setdefault("anchor_center_inside", True)
        proposal.metadata.setdefault("anchor_label_votes", {class_name: confidence} if class_name else {})
        proposal.metadata.setdefault("semantic_commit_allowed", True)
        proposal.metadata.setdefault("source_raw_proposal_id", int(proposal.proposal_id))
        proposal.metadata.setdefault("source_raw_proposal_ids", [int(proposal.proposal_id)])
        assignments.append(
            AnchorAssignment(
                proposal_id=int(proposal.proposal_id),
                anchor_id=anchor_id,
                class_name=class_name,
                confidence=confidence,
                bbox_iou=float(proposal.metadata.get("anchor_bbox_iou", 1.0)),
                center_inside=bool(proposal.metadata.get("anchor_center_inside", True)),
                keepalive=bool(proposal.metadata.get("anchor_keepalive", True)),
            )
        )
    return assignments


def _anchor_prompted_summary(
    *,
    anchor_count: int,
    source_sam_proposal_count: int,
    output_proposal_count: int,
    anchored_proposal_count: int,
    skipped_anchor_count: int = 0,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "anchor_first_sam",
        "anchor_prompt_enabled": True,
        "anchor_count": int(anchor_count),
        "source_sam_proposal_count": int(source_sam_proposal_count),
        "output_proposal_count": int(output_proposal_count),
        "anchored_proposal_count": int(anchored_proposal_count),
        "unknown_residual_count": 0,
        "dropped_proposal_count": int(skipped_anchor_count),
        "matched_sam_proposal_count": int(anchored_proposal_count),
        "mean_sam_candidates_per_anchor": float(output_proposal_count / max(anchor_count, 1)),
        "semantic_blocked_residual_count": 0,
        "assignment_count": int(anchored_proposal_count),
        "skip_reason": "anchor_first_sam",
    }


def build_yoloworld_sam_bundle(
    *,
    frame: Frame,
    object_anchor: Any,
    proposal: Any,
    anchor_guided_sam: Any,
    collect_stage_timings: bool = False,
    run_sam: bool = True,
) -> FrameProposalBundle:
    """Build a frontend bundle with independent YOLOWorld and dependent SAM lines.

    YOLOWorld is submitted first and never waits on SAM. The SAM worker waits for
    YOLO anchors before calling ``process_for_anchors``, then fusion waits for both.
    """
    start = time.perf_counter()

    def frontend_timings() -> dict[str, float]:
        if not collect_stage_timings:
            return {}
        return {
            str(key): _finite_timing(value)
            for key, value in dict(getattr(object_anchor, "last_generation_timings", {}) or {}).items()
        }

    if not run_sam:
        anchors, anchor_box_proposals, anchor_assignments = object_anchor.generate_anchor_box_proposals(frame.rgb, depth=frame.depth, intrinsics=(frame.intrinsics.fx, frame.intrinsics.fy, frame.intrinsics.cx, frame.intrinsics.cy))
        _stamp_anchor_primary_coarse_metadata(frame, anchor_box_proposals, anchor_assignments)
        generation_timings = frontend_timings()
        if collect_stage_timings:
            generation_timings["sam2_full_frame_ran"] = 0.0
            generation_timings["proposal_generation"] = max(0.0, time.perf_counter() - start)
        return FrameProposalBundle(
            frame_id=int(frame.frame_id),
            source_frame_id=frame.source_frame_id,
            source_proposals=tuple(),
            raw_proposals=tuple(anchor_box_proposals),
            anchors=tuple(anchors),
            anchor_assignments=tuple(anchor_assignments),
            proposal_source="anchor_box_primary",
            anchor_guided_sam_summary={
                "enabled": bool(getattr(anchor_guided_sam, "enabled", False)),
                "anchor_count": int(len(anchors)),
                "source_sam_proposal_count": 0,
                "output_proposal_count": int(len(anchor_box_proposals)),
                "anchored_proposal_count": int(len(anchor_assignments)),
                "unknown_residual_count": 0,
                "dropped_proposal_count": 0,
                "matched_sam_proposal_count": 0,
                "mean_sam_candidates_per_anchor": 0.0,
                "semantic_blocked_residual_count": 0,
                "assignment_count": int(len(anchor_assignments)),
                "skip_reason": "balanced_anchor_only_frame",
            },
            generation_timings=generation_timings,
            actual_backend=str(getattr(proposal, "active_backend_name", "")),
        )

    def run_yolo():
        return object_anchor.generate_anchor_box_proposals(frame.rgb, depth=frame.depth, intrinsics=(frame.intrinsics.fx, frame.intrinsics.fy, frame.intrinsics.cx, frame.intrinsics.cy))

    def run_sam_after_yolo(yolo_future):
        anchors, _anchor_box_proposals, _anchor_assignments = yolo_future.result()
        sam_start = time.perf_counter()
        sam_proposals = proposal.process_for_anchors(
            frame.rgb,
            frame.depth,
            list(anchors),
            frame=frame,
        )
        return list(sam_proposals), max(0.0, time.perf_counter() - sam_start)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="yoloworld-sam-front") as executor:
        yolo_future = executor.submit(run_yolo)
        sam_future = executor.submit(run_sam_after_yolo, yolo_future)
        anchors, anchor_box_proposals, anchor_assignments = yolo_future.result()
        sam_proposals, sam_elapsed = sam_future.result()

    fusion_start = time.perf_counter()
    proposals, fusion_anchor_assignments, anchor_guided_sam_summary = anchor_guided_sam.build_proposals(
        frame=frame,
        anchors=list(anchors),
        sam_proposals=list(sam_proposals),
    )
    fusion_elapsed = max(0.0, time.perf_counter() - fusion_start)

    generation_timings = {}
    if collect_stage_timings:
        generation_timings.update(frontend_timings())
        generation_timings["sam2_proposals"] = float(sam_elapsed)
        generation_timings["anchor_guided_sam_fusion"] = float(fusion_elapsed)
        generation_timings["sam2_full_frame_ran"] = 1.0
        generation_timings["proposal_generation"] = max(0.0, time.perf_counter() - start)

    if not proposals and anchor_box_proposals:
        proposals = list(anchor_box_proposals)
        _stamp_anchor_primary_coarse_metadata(frame, proposals, anchor_assignments)
    else:
        anchor_assignments = fusion_anchor_assignments

    return FrameProposalBundle(
        frame_id=int(frame.frame_id),
        source_frame_id=frame.source_frame_id,
        source_proposals=tuple(sam_proposals),
        raw_proposals=tuple(proposals),
        anchors=tuple(anchors),
        anchor_assignments=tuple(anchor_assignments),
        proposal_source="anchor_guided_sam",
        anchor_guided_sam_summary=dict(anchor_guided_sam_summary),
        generation_timings=generation_timings,
        actual_backend=str(getattr(proposal, "active_backend_name", "")),
    )


def build_anchor_first_sam_bundle(
    *,
    frame: Frame,
    object_anchor: Any,
    proposal: Any,
    collect_stage_timings: bool = False,
) -> FrameProposalBundle:
    """Build a frontend bundle from YOLOWorld anchors and SAM2 box prompts."""
    start = time.perf_counter()

    def frontend_timings() -> dict[str, float]:
        if not collect_stage_timings:
            return {}
        return {
            str(key): _finite_timing(value)
            for key, value in dict(getattr(object_anchor, "last_generation_timings", {}) or {}).items()
        }

    anchors, anchor_box_proposals, anchor_box_assignments = object_anchor.generate_anchor_box_proposals(frame.rgb, depth=frame.depth, intrinsics=(frame.intrinsics.fx, frame.intrinsics.fy, frame.intrinsics.cx, frame.intrinsics.cy))

    sam_start = time.perf_counter()
    sam_proposals = list(
        proposal.process_for_anchors(
            frame.rgb,
            frame.depth,
            list(anchors),
            frame=frame,
        )
    )
    sam_elapsed = max(0.0, time.perf_counter() - sam_start)

    anchor_assignments = _stamp_anchor_prompted_metadata(frame, sam_proposals, anchors)
    assigned_proposal_ids = {int(assignment.proposal_id) for assignment in anchor_assignments}
    anchored_sam_proposals = [
        proposal for proposal in sam_proposals if int(proposal.proposal_id) in assigned_proposal_ids
    ]
    if not anchored_sam_proposals and anchor_box_proposals:
        proposals = list(anchor_box_proposals)
        anchor_assignments = list(anchor_box_assignments)
        _stamp_anchor_primary_coarse_metadata(frame, proposals, anchor_assignments)
        proposal_source = "anchor_box_primary"
        skip_reason = "anchor_first_sam_empty_prompted_masks"
    else:
        proposals = anchored_sam_proposals
        proposal_source = "anchor_prompted_sam"
        skip_reason = "anchor_first_sam"

    generation_timings = {}
    if collect_stage_timings:
        generation_timings.update(frontend_timings())
        backend_timings = dict(getattr(proposal, "last_generation_timings", {}) or {})
        if "sam2_anchor_prompts" in backend_timings:
            generation_timings["sam2_anchor_prompts"] = _finite_timing(backend_timings["sam2_anchor_prompts"])
        else:
            generation_timings["sam2_anchor_prompts"] = float(sam_elapsed)
        generation_timings["sam2_full_frame_ran"] = 0.0
        generation_timings["proposal_generation"] = max(0.0, time.perf_counter() - start)

    backend_info = dict(getattr(getattr(proposal, "backend", None), "last_generation_info", {}) or {})
    skipped_anchor_count = int(backend_info.get("skipped_anchor_count", 0) or 0)
    summary = _anchor_prompted_summary(
        anchor_count=len(anchors),
        source_sam_proposal_count=len(sam_proposals),
        output_proposal_count=len(proposals),
        anchored_proposal_count=len(anchor_assignments),
        skipped_anchor_count=skipped_anchor_count,
    )
    summary["skip_reason"] = skip_reason

    return FrameProposalBundle(
        frame_id=int(frame.frame_id),
        source_frame_id=frame.source_frame_id,
        source_proposals=tuple(sam_proposals),
        raw_proposals=tuple(proposals),
        anchors=tuple(anchors),
        anchor_assignments=tuple(anchor_assignments),
        proposal_source=proposal_source,
        anchor_guided_sam_summary=summary,
        generation_timings=generation_timings,
        actual_backend=str(getattr(proposal, "active_backend_name", "")),
    )
