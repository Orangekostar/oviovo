"""One B200 native policy with alternating preferred lanes and shared history."""

import time
from dataclasses import asdict

import numpy as np

from src.static_ovmap.module_validation.query_gain_policy import rank_query_candidates
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


def replay_mixed(frames, store, text, *, predictor, standardizer, budget=200):
    if not 0 <= budget <= 200:
        raise ValueError("mixed policy has one total budget of at most 200")
    text = np.asarray(text, np.float64)
    state, combine, lineage = (
        QueryPolicyState("CP_M5_MIX50_NATIVE", len(text)),
        NativeCombineState(),
        CurrentLineage(),
    )
    decisions, lanes, paid_requests = [], [], []
    started = time.monotonic()
    for index in range(len(frames.schedule)):
        current = frames.load(index)
        candidates = () if current is None else current["candidates"]
        if current is not None:
            lineage.advance(current["snapshot"], state, combine, text)
            lineage.register_candidates(candidates)
        candidates = tuple(
            row for row in candidates if row.request_id not in lineage.paid
        )
        admitted = (
            native_combine_candidates(candidates, combine)
            if current is not None
            else ()
        )
        ranking = {
            "COMBINE": rank_query_candidates("Q_COMBINE", admitted, state),
            "GAIN": rank_query_candidates(
                "Q_GAIN",
                candidates,
                state,
                gain_predictor=predictor,
                standardizer=standardizer,
                frame_count=len(frames.schedule),
                budget=budget,
            ),
        }
        spent = state.logical_ledger.attempts
        quota = cumulative_quota(
            budget, frame_index=index, frame_count=len(frames.schedule), spent=spent
        )
        selected, selected_ids, frame_lanes = [], set(), []
        for offset in range(quota):
            slot = spent + offset + 1
            preferred = "COMBINE" if slot % 2 else "GAIN"
            alternative = "GAIN" if preferred == "COMBINE" else "COMBINE"
            found = None
            for lane in (preferred, alternative):
                found = next(
                    (
                        (rank, candidate)
                        for rank, candidate in enumerate(ranking[lane], 1)
                        if candidate.request_id not in selected_ids
                    ),
                    None,
                )
                if found is not None:
                    break
            if found is None:
                break
            rank, candidate = found
            selected.append(candidate)
            selected_ids.add(candidate.request_id)
            row = {
                "frame_index": index,
                "frame_id": frames.schedule[index],
                "request_id": candidate.request_id,
                "owner_id": candidate.owner_id,
                "paid_slot": slot,
                "frame_quota": quota,
                "preferred_lane": preferred,
                "winning_lane": lane,
                "rank_position": rank,
                "fallback_reason": None
                if lane == preferred
                else "PREFERRED_LIST_EXHAUSTED",
            }
            frame_lanes.append(row)
            if "requests" in current:
                paid_requests.append(
                    {
                        "frame_index": index,
                        "frame_id": frames.schedule[index],
                        "request": current["requests"][candidate.request_id].to_dict(),
                    }
                )
        # dispatch_frame_batch debits the complete selection, then retrieves all
        # features, then admits results at one barrier. Neither ranking is rerun.
        results = dispatch_frame_batch(state, store, selected, quota=len(selected))
        if [row.request_id for row in results] != [row.request_id for row in selected]:
            raise ValueError("mixed batch contains an already paid request")
        for owner in {row.owner_id for row in results if row.success}:
            update_cached_class_scores(state, owner, text)
        lineage.record(candidates, results)
        for candidate, result in zip(selected, results, strict=True):
            if result.success:
                combine.successful_overlaps_by_owner.setdefault(
                    candidate.owner_id, []
                ).append(candidate.overlap_pixels)
        observe_geometric_candidates(state, candidates)
        lanes.extend(frame_lanes)
        decisions.append(
            {
                "frame_index": index,
                "frame_id": frames.schedule[index],
                "quota": quota,
                "pre_acquisition_spent": spent,
                "batch_debited": len(results),
                "combine_admitted_request_ids": [row.request_id for row in admitted],
                "ranked_combine_request_ids": [
                    row.request_id for row in ranking["COMBINE"]
                ],
                "ranked_gain_request_ids": [row.request_id for row in ranking["GAIN"]],
                "ranked_request_ids": [row.request_id for row in selected],
                "results": [asdict(row) for row in results],
            }
        )
    if state.logical_ledger.attempts != len(lanes) or len(lanes) > budget:
        raise AssertionError("mixed shared-budget accounting differs")
    return {
        "state": state,
        "combine": combine,
        "lineage": lineage,
        "decisions": decisions,
        "lane_ledger": lanes,
        "paid_requests": paid_requests,
        "events": [],
        "elapsed_seconds": time.monotonic() - started,
    }
