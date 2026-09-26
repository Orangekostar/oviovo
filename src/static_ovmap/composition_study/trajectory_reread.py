"""Forced attempted-trajectory replay, with its own model and causal lineage."""

import time
from dataclasses import asdict

import numpy as np

from src.static_ovmap.module_validation.query_lineage import CurrentLineage
from src.static_ovmap.module_validation.query_state import (
    NativeCombineState,
    QueryPolicyState,
    cumulative_quota,
    dispatch_frame_batch,
    observe_geometric_candidates,
    update_cached_class_scores,
)


def replay_fixed(frames, decisions, store, text, *, budget=200):
    decisions = decisions["frames"] if isinstance(decisions, dict) else decisions
    if len(decisions) != len(frames.schedule) or not 0 <= budget <= 200:
        raise ValueError("trajectory frame schedule or budget differs")
    text = np.asarray(text, np.float64)
    state, combine, lineage = (
        QueryPolicyState("FIXED_TRAJECTORY", len(text)),
        NativeCombineState(),
        CurrentLineage(),
    )
    output, paid_requests = [], []
    started = time.monotonic()
    for index, frozen in enumerate(decisions):
        if (
            frozen["frame_index"] != index
            or frozen["frame_id"] != frames.schedule[index]
        ):
            raise ValueError("trajectory frame order differs")
        current = frames.load(index)
        candidates = () if current is None else current["candidates"]
        if current is not None:
            lineage.advance(current["snapshot"], state, combine, text)
            lineage.register_candidates(candidates)
        candidates = tuple(
            row for row in candidates if row.request_id not in lineage.paid
        )
        by_id = {row.request_id: row for row in candidates}
        requested = [row["request_id"] for row in frozen["results"]]
        quota = cumulative_quota(
            budget,
            frame_index=index,
            frame_count=len(frames.schedule),
            spent=state.logical_ledger.attempts,
        )
        if (
            len(requested) != len(set(requested))
            or not set(requested) <= set(by_id)
            or len(requested) > quota
        ):
            raise ValueError(
                "trajectory requests are duplicated, unavailable or over budget"
            )
        selected = [by_id[request] for request in requested]
        for request in requested:
            if "requests" in current:
                paid_requests.append(
                    {
                        "frame_index": index,
                        "frame_id": frames.schedule[index],
                        "request": current["requests"][request].to_dict(),
                    }
                )
        results = dispatch_frame_batch(state, store, selected, quota=len(selected))
        if [row.request_id for row in results] != requested:
            raise ValueError("trajectory attempted sequence changed at dispatch")
        for owner in {row.owner_id for row in results if row.success}:
            update_cached_class_scores(state, owner, text)
        lineage.record(candidates, results)
        observe_geometric_candidates(state, candidates)
        output.append(
            {
                "frame_index": index,
                "frame_id": frames.schedule[index],
                "quota": quota,
                "ranked_request_ids": requested,
                "results": [asdict(row) for row in results],
            }
        )
    return {
        "state": state,
        "combine": combine,
        "lineage": lineage,
        "decisions": output,
        "paid_requests": paid_requests,
        "events": [],
        "elapsed_seconds": time.monotonic() - started,
    }
