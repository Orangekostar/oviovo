"""Project frozen B3 decisions onto native OVI rows for fine refinement."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import MappingProxyType

import numpy as np

from src.oviv2.fine_surface_validity import (
    FineSurfaceEvidence,
    FineSurfaceObservation,
)


class CoarseSurfaceState(IntEnum):
    RETAINED_OCCLUDED = 1
    RETAINED_UNOBSERVED = 2
    REPLACED_OCCUPIED = 3
    VISIBLE_FREE_CANDIDATE = 4


_DECISION_STATES = {
    ("retain_t0_occluded", "occluded"): CoarseSurfaceState.RETAINED_OCCLUDED,
    ("retain_t0_unobserved", "unobserved"): CoarseSurfaceState.RETAINED_UNOBSERVED,
    (
        "suppress_t0_occupied_by_t1",
        "occupied",
    ): CoarseSurfaceState.REPLACED_OCCUPIED,
    (
        "suppress_t0_visible_free",
        "visible_free",
    ): CoarseSurfaceState.VISIBLE_FREE_CANDIDATE,
}


@dataclass(frozen=True, slots=True)
class FineSurfacePolicy:
    states: np.ndarray
    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        raw = np.asarray(self.states)
        allowed = {int(value) for value in CoarseSurfaceState}
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise ValueError("fine policy states must be a one-dimensional integer array")
        states = np.array(raw, dtype=np.uint8, copy=True)
        if not set(int(value) for value in np.unique(states)) <= allowed:
            raise ValueError("fine policy contains an unknown coarse state")
        states.setflags(write=False)
        counts = {str(key): int(value) for key, value in self.counts.items()}
        expected = {
            "retained_occluded": int(
                np.count_nonzero(states == CoarseSurfaceState.RETAINED_OCCLUDED)
            ),
            "retained_unobserved": int(
                np.count_nonzero(states == CoarseSurfaceState.RETAINED_UNOBSERVED)
            ),
            "replaced_occupied": int(
                np.count_nonzero(states == CoarseSurfaceState.REPLACED_OCCUPIED)
            ),
            "visible_free_candidate": int(
                np.count_nonzero(states == CoarseSurfaceState.VISIBLE_FREE_CANDIDATE)
            ),
        }
        if counts != expected:
            raise ValueError("fine policy counts do not match its states")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "counts", MappingProxyType(counts))


def _source_owner_id(value: object) -> int:
    if value == "__background__":
        return 0
    if not isinstance(value, str) or not value.startswith("ovimap:"):
        raise ValueError("B3 provenance has an invalid source entity ID")
    try:
        owner_id = int(value.split(":", 1)[1])
    except ValueError as error:
        raise ValueError("B3 provenance has an invalid source entity ID") from error
    if owner_id <= 0:
        raise ValueError("B3 provenance owner IDs must be positive")
    return owner_id


def _state(decision: Mapping[str, object]) -> CoarseSurfaceState:
    action = decision.get("decision")
    visibility = decision.get("visibility_status")
    if action == "suppress_t0_entity_visible_free" and visibility in {
        "visible_free",
        "occluded",
        "unobserved",
    }:
        return CoarseSurfaceState.VISIBLE_FREE_CANDIDATE
    try:
        return _DECISION_STATES[(str(action), str(visibility))]
    except KeyError as error:
        raise ValueError(
            f"unsupported B3 t0 decision: {action}/{visibility}"
        ) from error


def load_b3_surface_policy(
    provenance_path: str | Path,
    source_owner_ids: np.ndarray,
) -> FineSurfacePolicy:
    """Map entity-local B3 source indices back to native t0 PLY row order."""

    owners = np.asarray(source_owner_ids)
    if owners.ndim != 1 or owners.dtype.kind not in "iu" or np.any(owners < 0):
        raise ValueError("source_owner_ids must be a nonnegative integer vector")
    if not len(owners):
        raise ValueError("source_owner_ids must not be empty")
    path = Path(provenance_path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)

    order = np.argsort(owners, kind="stable")
    sorted_owners = owners[order]
    unique, starts, owner_counts = np.unique(
        sorted_owners, return_index=True, return_counts=True
    )
    owner_spans = {
        int(owner_id): (int(start), int(count))
        for owner_id, start, count in zip(
            unique, starts, owner_counts, strict=True
        )
    }
    states = np.zeros(len(owners), dtype=np.uint8)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid B3 provenance JSON at line {line_number}"
                ) from error
            if not isinstance(payload, Mapping) or not isinstance(
                payload.get("decision"), Mapping
            ):
                raise ValueError(f"invalid B3 provenance record at line {line_number}")
            decision = payload["decision"]
            source_visit = decision.get("source_visit")
            if source_visit == 1:
                continue
            if source_visit != 0:
                raise ValueError(f"invalid B3 source visit at line {line_number}")
            owner_id = _source_owner_id(decision.get("source_entity_id"))
            if owner_id not in owner_spans:
                raise ValueError(
                    f"B3 provenance references absent owner {owner_id}"
                )
            raw_indices = np.asarray(payload.get("source_point_indices"))
            if (
                raw_indices.ndim != 1
                or not len(raw_indices)
                or raw_indices.dtype.kind not in "iu"
            ):
                raise ValueError(
                    f"B3 source indices are invalid at line {line_number}"
                )
            local = raw_indices.astype(np.int64, copy=False)
            start, count = owner_spans[owner_id]
            if (
                np.any(local < 0)
                or np.any(local >= count)
                or (len(local) > 1 and np.any(local[1:] <= local[:-1]))
            ):
                raise ValueError(
                    f"B3 source indices are out of range or unordered at line {line_number}"
                )
            global_rows = order[start + local]
            if np.any(states[global_rows] != 0):
                raise ValueError("a t0 native row is assigned more than once")
            states[global_rows] = int(_state(decision))
    if np.any(states == 0):
        raise ValueError("B3 provenance must cover every t0 native row")
    counts = {
        "retained_occluded": int(
            np.count_nonzero(states == CoarseSurfaceState.RETAINED_OCCLUDED)
        ),
        "retained_unobserved": int(
            np.count_nonzero(states == CoarseSurfaceState.RETAINED_UNOBSERVED)
        ),
        "replaced_occupied": int(
            np.count_nonzero(states == CoarseSurfaceState.REPLACED_OCCUPIED)
        ),
        "visible_free_candidate": int(
            np.count_nonzero(states == CoarseSurfaceState.VISIBLE_FREE_CANDIDATE)
        ),
    }
    return FineSurfacePolicy(states=states, counts=counts)


def assemble_fine_surface_evidence(
    *,
    states: np.ndarray,
    candidate_rows: np.ndarray,
    candidate_observation: FineSurfaceObservation,
    historical_last_supported_frames: np.ndarray,
) -> FineSurfaceEvidence:
    """Preserve B3 decisions while replacing only its free-space decision evidence."""

    raw_states = np.asarray(states)
    rows = np.asarray(candidate_rows)
    last = np.asarray(historical_last_supported_frames)
    count = len(raw_states)
    if raw_states.shape != (count,) or raw_states.dtype.kind not in "iu":
        raise ValueError("states must be a one-dimensional integer array")
    expected_rows = np.flatnonzero(
        raw_states == int(CoarseSurfaceState.VISIBLE_FREE_CANDIDATE)
    )
    if rows.dtype.kind not in "iu" or not np.array_equal(rows, expected_rows):
        raise ValueError("candidate_rows must exactly identify visible-free candidates")
    if not isinstance(candidate_observation, FineSurfaceObservation) or len(
        candidate_observation.rgb_valid
    ) != len(rows):
        raise ValueError("candidate observation must align with candidate_rows")
    if last.shape != (count,) or last.dtype.kind not in "iu" or np.any(last < -1):
        raise ValueError("historical support frames must align and be at least -1")

    present = np.zeros(count, dtype=np.uint16)
    absent = np.zeros(count, dtype=np.uint16)
    occluded = np.zeros(count, dtype=np.uint16)
    distinct = np.zeros(count, dtype=np.uint8)
    present[raw_states == int(CoarseSurfaceState.REPLACED_OCCUPIED)] = 1
    occluded[raw_states == int(CoarseSurfaceState.RETAINED_OCCLUDED)] = 1
    candidate_evidence = candidate_observation.evidence
    present[rows] = candidate_evidence.present_observations
    absent[rows] = candidate_evidence.visible_absent_observations
    occluded[rows] = candidate_evidence.occluded_observations
    distinct[rows] = candidate_evidence.distinct_absent_viewpoints
    supported = np.array(last, dtype=np.int32, copy=True)
    candidate_supported = candidate_evidence.last_supported_frames
    supported[rows] = np.maximum(supported[rows], candidate_supported)
    return FineSurfaceEvidence(
        present_observations=present,
        visible_absent_observations=absent,
        occluded_observations=occluded,
        distinct_absent_viewpoints=distinct,
        last_supported_frames=supported,
    )


__all__ = [
    "CoarseSurfaceState",
    "FineSurfacePolicy",
    "assemble_fine_surface_evidence",
    "load_b3_surface_policy",
]
