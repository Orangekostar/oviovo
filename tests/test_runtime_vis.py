"""Tests for the depth-aware runtime_vis consolidation stage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import (
    CameraIntrinsics,
    Frame,
    ObjectMap,
    ObservationRecord,
    Patch3D,
    SemanticMemory,
    SystemState,
)
from src.core.data_structures import ObjectState
from src.core.data_structures import Proposal2D
from src.modules.runtime_vis import RuntimeVisModule


def make_frame(depth: np.ndarray) -> Frame:
    h, w = depth.shape
    return Frame(
        frame_id=0,
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth=depth.astype(np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(
            fx=10.0,
            fy=10.0,
            cx=w / 2.0,
            cy=h / 2.0,
            width=w,
            height=h,
        ),
    )


def rect_proposal(proposal_id: int, x1: int, y1: int, x2: int, y2: int, shape: tuple[int, int]) -> Proposal2D:
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return Proposal2D(
        proposal_id=proposal_id,
        mask=mask,
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
    )


def runtime_vis_config() -> dict:
    return {
        "enabled": True,
        "merge_score_threshold": 0.85,
        "base_score_floor": 0.35,
        "lambda_whole_prior": 1.5,
        "mu_background_conflict": 1.35,
        "bbox_gap_px": 3,
        "depth_mean_diff_max": 0.2,
        "prior_depth_diff_max": 0.4,
        "plane_similarity_diff_max": 0.2,
        "bbox_fill_ratio_ref": 0.55,
        "expanded_prior_bbox_scale": 1.15,
        "recent_history_size": 10,
        "edge_admission_min_adjacency": 0.55,
        "edge_admission_min_depth_continuity": 0.55,
        "edge_admission_min_plane_similarity": 0.60,
        "edge_admission_min_whole_prior": 0.65,
        "small_object_area_threshold": 16,
        "small_object_depth_gap_min": 0.12,
        "small_object_bbox_plausibility_margin": 0.08,
        "small_object_protection_threshold": 0.60,
        "small_object_protection_penalty": 0.25,
        "background_conflict_reject_threshold": 0.62,
        "max_plane_fit_points": 64,
    }


def make_state_with_object(object_id: int = 42) -> SystemState:
    bbox_min = np.array([-0.4, -0.4, 2.0], dtype=np.float32)
    bbox_max = np.array([1.2, 0.4, 2.1], dtype=np.float32)
    points = np.array(
        [
            [-0.4, -0.4, 2.0],
            [1.2, -0.4, 2.0],
            [-0.4, 0.4, 2.1],
            [1.2, 0.4, 2.1],
        ],
        dtype=np.float32,
    )
    patch = Patch3D(
        patch_id=0,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        timestamp=0.0,
        source_frame_id=0,
    )
    obj = ObjectMap(
        object_id=object_id,
        state=ObjectState.ACTIVE,
        local_pcd=points.copy(),
        centroid=points.mean(axis=0),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        semantic_memory=SemanticMemory(),
        observations=[
            ObservationRecord(frame_id=0, patch=patch),
            ObservationRecord(frame_id=0, patch=patch),
            ObservationRecord(frame_id=0, patch=patch),
        ],
        confidence=1.0,
        last_seen_frame=0,
        creation_frame=0,
        update_count=4,
    )
    return SystemState(objects={object_id: obj}, next_object_id=object_id + 1, frame_count=1)


def test_runtime_vis_wall_and_cabinet_do_not_merge() -> None:
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    frame = make_frame(depth)
    proposals = [
        rect_proposal(0, 0, 0, 20, 20, depth.shape),
        rect_proposal(1, 10, 6, 14, 10, depth.shape),
    ]

    module = RuntimeVisModule(runtime_vis_config())
    output = module.process(frame, proposals, SystemState())

    assert len(output.groups) == 2
    decision = output.merge_decisions[0]
    assert decision.accepted is False
    assert decision.background_conflict_penalty >= module.background_conflict_reject_threshold
    assert decision.accepted_reason in {"rejected_due_to_background_conflict", "background_mask_excluded"}

    profile_by_id = {profile.proposal_id: profile for profile in output.proposal_profiles}
    assert profile_by_id[0].is_background_like is True
    assert profile_by_id[1].is_object_like is True


def test_runtime_vis_merges_compatible_object_fragments() -> None:
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    frame = make_frame(depth)
    proposals = [
        rect_proposal(0, 2, 2, 6, 6, depth.shape),
        rect_proposal(1, 6, 2, 10, 6, depth.shape),
    ]

    module = RuntimeVisModule(runtime_vis_config())
    output = module.process(frame, proposals, SystemState())

    assert len(output.groups) == 1
    assert output.raw_to_group[0] == output.raw_to_group[1]
    assert output.group_stats["accepted_edge_count"] == 1
    assert output.merge_decisions[0].accepted is True


def test_runtime_vis_small_standalone_object_remains_separate() -> None:
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    depth[2:5, 10:13] = 2.5
    frame = make_frame(depth)
    proposals = [
        rect_proposal(0, 2, 2, 10, 10, depth.shape),
        rect_proposal(1, 10, 2, 13, 5, depth.shape),
    ]

    module = RuntimeVisModule(runtime_vis_config())
    output = module.process(frame, proposals, SystemState())

    assert len(output.groups) == 2
    assert output.raw_to_group[0] != output.raw_to_group[1]
    assert output.merge_decisions[0].accepted is False


def test_runtime_vis_historical_whole_prior_helps_merge_fragments() -> None:
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    frame = make_frame(depth)
    proposals = [
        rect_proposal(0, 8, 8, 11, 12, depth.shape),
        rect_proposal(1, 13, 8, 16, 12, depth.shape),
    ]

    module = RuntimeVisModule(runtime_vis_config())
    without_history = module.process(frame, proposals, SystemState())
    with_history = module.process(frame, proposals, make_state_with_object())

    assert len(without_history.groups) == 2
    assert len(with_history.groups) == 1
    assert list(with_history.group_to_linked_object.values())[0] == 42
    assert any(
        decision.accepted_reason == "accepted_with_whole_prior_boost"
        for decision in with_history.merge_decisions
    )


def test_runtime_vis_whole_prior_cannot_override_background_conflict() -> None:
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    frame = make_frame(depth)
    proposals = [
        rect_proposal(0, 0, 0, 20, 20, depth.shape),
        rect_proposal(1, 9, 7, 12, 10, depth.shape),
    ]

    module = RuntimeVisModule(runtime_vis_config())
    output = module.process(frame, proposals, make_state_with_object())

    assert len(output.groups) == 2
    decision = output.merge_decisions[0]
    assert decision.accepted is False
    assert decision.background_conflict_penalty >= module.background_conflict_reject_threshold
    assert decision.accepted_reason in {"rejected_due_to_background_conflict", "background_mask_excluded"}


def test_runtime_vis_debug_outputs_expose_depth_aware_decisions() -> None:
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    frame = make_frame(depth)
    proposals = [
        rect_proposal(0, 2, 2, 6, 6, depth.shape),
        rect_proposal(1, 6, 2, 10, 6, depth.shape),
        rect_proposal(2, 15, 15, 17, 17, depth.shape),
    ]

    module = RuntimeVisModule(runtime_vis_config())
    output = module.process(frame, proposals, SystemState())

    assert len(output.proposal_profiles) == len(proposals)
    assert all(hasattr(profile, "objectness_score") for profile in output.proposal_profiles)
    assert all(hasattr(profile, "backgroundness_score") for profile in output.proposal_profiles)
    assert all(hasattr(profile, "depth_variance") for profile in output.proposal_profiles)

    assert len(output.merge_decisions) > 0
    assert all(hasattr(decision, "background_conflict_penalty") for decision in output.merge_decisions)
    assert all(hasattr(decision, "median_depth_gap") for decision in output.merge_decisions)
    assert all(hasattr(decision, "boundary_depth_continuity") for decision in output.merge_decisions)
    assert all(hasattr(decision, "rejected_due_to_background_conflict") for decision in output.merge_decisions)

    assert all(hasattr(group, "merge_reason_summary") for group in output.groups)
    assert "background_blocked_edge_count" in output.group_stats
    assert "object_like_count" in output.group_stats
    assert "background_like_count" in output.group_stats
