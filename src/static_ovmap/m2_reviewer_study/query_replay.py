"""Study-local causal replay adapter."""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np

from src.static_ovmap.module_validation.query_gain_policy import (
    build_query_features,
    random_exploration_ranking,
    rank_query_candidates,
)
from src.static_ovmap.module_validation.query_lineage import CurrentLineage
from src.static_ovmap.module_validation.query_state import (
    NativeCombineState,
    QueryPolicyState,
    cumulative_quota,
    dispatch_frame_batch,
    native_combine_candidates,
    observe_geometric_candidates,
    update_cached_class_scores,
)


def replay_study(frames, policy_id, store, text, *, budget, seed=None, predictor=None, standardizer=None,
                 acquired_callback=None):
    """One frame barrier, all budget debits before any feature access.

    The optional callback receives only paid acquisition records after the
    barrier; training labels cannot influence this loop's ranking decisions.
    """
    if policy_id == "RV_Q_RANDOM":
        if seed not in (17, 23, 41) or budget != 200:
            raise ValueError("random study policy requires a frozen seed and budget 200")
    elif policy_id not in {"Q_COMBINE", "Q_GAIN"} or budget not in (100, 200, 400):
        raise ValueError("unknown study policy or budget")
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
        if policy_id == "RV_Q_RANDOM":
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
    return {"state": state, "combine": combine, "lineage": lineage, "decisions": decisions, "events": events,
            "elapsed_seconds": time.monotonic() - started}
