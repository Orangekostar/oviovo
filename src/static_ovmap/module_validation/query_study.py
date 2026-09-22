"""Streaming causal replay from current captured RGB-D and native membership."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .assets import sha256_file
from .native_capture import RegionRequest, _array_digest
from .query_gain_policy import (
    build_query_features,
    random_exploration_ranking,
    rank_query_candidates,
)
from .query_lineage import CurrentLineage
from .query_state import (
    NativeCombineState,
    QueryPolicyState,
    build_query_candidates,
    cumulative_quota,
    dispatch_frame_batch,
    native_combine_candidates,
    observe_geometric_candidates,
    update_cached_class_scores,
)


class CapturedFrames:
    """Only load current frame assets; final surface/aliases/features stay unread."""

    def __init__(self, capture_path):
        self.path = Path(capture_path)
        manifest = json.loads(self.path.read_text())
        self.scene_id = manifest["scene_id"]
        self.schedule = tuple(manifest["scheduled_frame_ids"])
        self.frames = {row["frame_id"]: row for row in manifest["frames"]}
        self.current = None

    def _path(self, relative, digest):
        path = self.path.parent / relative
        if sha256_file(path) != digest:
            raise ValueError(f"current query input hash changed: {path}")
        return path

    def load(self, index):
        import cv2

        frame = self.frames.get(self.schedule[index])
        if frame is None:
            self.current = None
            return None
        state_entry = frame["native_state"]
        snapshot = json.loads(self._path(state_entry["path"], state_entry["sha256"]).read_text())["native_state"]
        if snapshot.get("label_instances_scope") != "all_known_labels":
            raise ValueError("BLOCKED_CAUSAL_LINEAGE: native frame snapshot has only partial membership")
        owners = np.asarray(Image.open(self._path(frame["global_owner_path"], frame["global_owner_sha256"])))
        local = np.asarray(Image.open(self._path(frame["panoptic_path"], frame["panoptic_sha256"])))
        with np.load(self._path(frame["depth_path"], frame["depth_sha256"]), allow_pickle=False) as arrays:
            depth = np.array(arrays["depth_m"], dtype=np.float32, copy=True)
        valid = np.isfinite(depth) & (depth > 0) & (depth < 50.)
        depth[~valid] = 0.
        points = cv2.rgbd.depthTo3d(depth, np.asarray(frame["intrinsics"], np.float32))
        requests = {request.request_id: request for request in map(RegionRequest.from_dict, frame["requests"])}
        by_owner = {int(request.target_id.removeprefix("owner:")): request.request_id for request in requests.values()}
        candidates = build_query_candidates(self.scene_id, index, frame["frame_id"], owners, local, depth, valid,
            frame["pose_c2w"], points, request_ids_by_owner=by_owner)
        if {candidate.request_id for candidate in candidates} != set(requests):
            raise ValueError("current technical candidate universe differs from native capture")
        for candidate in candidates:
            request = requests[candidate.request_id]
            if (request.target_mask_sha256 != _array_digest(candidate.global_mask)
                    or request.native_union_mask_sha256 != _array_digest(candidate.union_mask)
                    or request.bbox_xyxy != candidate.bbox_xyxy or request.visible_target_pixels != candidate.overlap_pixels):
                raise ValueError("current candidate differs from its actual native request")
        self.current = {"frame": frame, "snapshot": snapshot, "requests": requests,
                        "candidates": candidates, "depth": depth}
        return self.current


def replay_captured(frames, policy_id, store, text, *, budget, seed=None, predictor=None, standardizer=None,
                    acquired_callback=None):
    """One frame barrier, all budget debits before any feature access.

    The optional callback receives only paid acquisition records after the
    barrier; training labels cannot influence this loop's ranking decisions.
    """
    if policy_id == "Q_RANDOM_TRACE":
        if (seed, budget) not in {(17, 512), (23, 256)}:
            raise ValueError("random traces require frozen FIT or CAL seed/budget")
    elif policy_id not in {"Q_COMBINE", "Q_AREA", "Q_UNCERTAINTY", "Q_GAIN"}:
        raise ValueError("unknown causal query policy")
    text = np.asarray(text, np.float64)
    state, combine, lineage = QueryPolicyState(policy_id, len(text)), NativeCombineState(), CurrentLineage()
    decisions, events = [], []
    started = time.monotonic()
    for index in range(len(frames.schedule)):
        current = frames.load(index)
        quota = cumulative_quota(budget, frame_index=index, frame_count=len(frames.schedule), spent=state.logical_ledger.attempts)
        candidates = () if current is None else current["candidates"]
        if current is not None:
            lineage.advance(current["snapshot"], state, combine, text)
            lineage.register_candidates(candidates)
        candidates = tuple(row for row in candidates if row.request_id not in lineage.paid)
        eligible = native_combine_candidates(candidates, combine) if policy_id == "Q_COMBINE" else candidates
        if policy_id == "Q_RANDOM_TRACE":
            ranked = random_exploration_ranking(eligible, seed=seed, scene_id=frames.scene_id, frame_index=index)
        else:
            ranked = rank_query_candidates(policy_id, eligible, state, gain_predictor=predictor,
                standardizer=standardizer, frame_count=len(frames.schedule), budget=budget)
        selected = ranked[:quota]
        prefix = {}
        for candidate in selected:
            scores = state.object_state(candidate.owner_id).cached_scores
            prefix[candidate.request_id] = {"features": build_query_features(candidate, state,
                frame_count=len(frames.schedule), budget=budget, spent=state.logical_ledger.attempts).unstandardized,
                "before_scores": None if scores is None else scores.copy()}
        results = dispatch_frame_batch(state, store, ranked, quota=quota)
        for owner in {row.owner_id for row in results if row.success}:
            update_cached_class_scores(state, owner, text)
        lineage.record(candidates, results)
        by_request = {row.request_id: row for row in selected}
        for result in results:
            candidate = by_request[result.request_id]
            if result.success and policy_id == "Q_COMBINE":
                combine.successful_overlaps_by_owner.setdefault(candidate.owner_id, []).append(candidate.overlap_pixels)
            scores = state.object_state(candidate.owner_id).cached_scores
            event = {"scene_id": frames.scene_id, "frame_index": index, "frame_id": candidate.frame_id,
                "request_id": result.request_id, "owner_id": candidate.owner_id, **prefix[result.request_id],
                "after_scores": None if scores is None else scores.copy(), "success": result.success}
            events.append(event)
            if acquired_callback is not None:
                acquired_callback(event, candidate, current)
        observe_geometric_candidates(state, candidates)
        decisions.append({"frame_index": index, "frame_id": frames.schedule[index], "quota": quota,
            "ranked_request_ids": [row.request_id for row in ranked], "results": [asdict(row) for row in results]})
    if state.logical_ledger.attempts != sum(len(row["results"]) for row in decisions) or state.logical_ledger.attempts > budget:
        raise AssertionError("causal query budget and frame ledger disagree")
    return {"state": state, "lineage": lineage, "decisions": decisions, "events": events,
            "elapsed_seconds": time.monotonic() - started}
