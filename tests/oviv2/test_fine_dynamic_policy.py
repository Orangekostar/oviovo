from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.oviv2 import fine_dynamic_policy
from src.oviv2.current_surface import CurrentEvidenceState
from src.oviv2.fine_dynamic_policy import (
    CoarseSurfaceState,
    assemble_fine_surface_evidence,
    load_b3_surface_policy,
)
from src.oviv2.fine_surface_validity import FineSurfaceEvidence, FineSurfaceObservation


def _record(
    *,
    owner: str,
    source_indices: list[int],
    decision: str,
    visibility: str,
) -> dict[str, object]:
    emitted = decision.startswith("retain_")
    return {
        "decision": {
            "decision": decision,
            "geometry_source": "ovi_t0" if emitted else None,
            "identity_source": "unmatched",
            "relation_id": None,
            "semantic_source": "ovi_t0" if emitted else None,
            "source_entity_id": owner,
            "source_visit": 0,
            "state_source": "fallback" if emitted else "t1_visibility",
            "visibility_score": 0.0 if visibility == "unobserved" else 1.0,
            "visibility_status": visibility,
        },
        "output_entity_id": f"t0-fallback:{owner}" if emitted else None,
        "output_point_count": len(source_indices) if emitted else 0,
        "output_point_start": 0 if emitted else None,
        "source_point_indices": source_indices,
        "source_snapshot_sha256": "a" * 64,
    }


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_b3_policy_maps_entity_local_indices_back_to_native_rows(tmp_path: Path) -> None:
    path = tmp_path / "provenance.jsonl"
    _write_records(
        path,
        [
            _record(
                owner="ovimap:1",
                source_indices=[0],
                decision="retain_t0_occluded",
                visibility="occluded",
            ),
            _record(
                owner="ovimap:1",
                source_indices=[1],
                decision="suppress_t0_visible_free",
                visibility="visible_free",
            ),
            _record(
                owner="ovimap:2",
                source_indices=[0],
                decision="suppress_t0_occupied_by_t1",
                visibility="occupied",
            ),
            _record(
                owner="__background__",
                source_indices=[0],
                decision="retain_t0_unobserved",
                visibility="unobserved",
            ),
        ],
    )

    policy = load_b3_surface_policy(
        path,
        np.asarray([1, 2, 1, 0], dtype=np.int64),
    )

    assert policy.states.tolist() == [
        CoarseSurfaceState.RETAINED_OCCLUDED,
        CoarseSurfaceState.REPLACED_OCCUPIED,
        CoarseSurfaceState.VISIBLE_FREE_CANDIDATE,
        CoarseSurfaceState.RETAINED_UNOBSERVED,
    ]
    assert policy.counts == {
        "replaced_occupied": 1,
        "retained_occluded": 1,
        "retained_unobserved": 1,
        "visible_free_candidate": 1,
    }


def test_b3_prior_preserves_direct_free_entity_lift_and_replaced_reasons(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provenance.jsonl"
    _write_records(
        path,
        [
            _record(
                owner="ovimap:1",
                source_indices=[0],
                decision="suppress_t0_visible_free",
                visibility="visible_free",
            ),
            _record(
                owner="ovimap:1",
                source_indices=[1],
                decision="suppress_t0_entity_visible_free",
                visibility="occluded",
            ),
            _record(
                owner="ovimap:2",
                source_indices=[0],
                decision="suppress_t0_occupied_by_t1",
                visibility="occupied",
            ),
            _record(
                owner="__background__",
                source_indices=[0],
                decision="retain_t0_unobserved",
                visibility="unobserved",
            ),
        ],
    )

    prior = fine_dynamic_policy.load_b3_prior_surface_state(
        path,
        np.asarray([1, 2, 1, 0], dtype=np.int64),
    )

    assert prior.current_valid.tolist() == [False, False, False, True]
    assert prior.retirement_reason_codes.tolist() == [
        fine_dynamic_policy.SurfaceRetirementReason.DIRECT_FREE,
        fine_dynamic_policy.SurfaceRetirementReason.REPLACED,
        fine_dynamic_policy.SurfaceRetirementReason.ENTITY_LIFT,
        fine_dynamic_policy.SurfaceRetirementReason.NONE,
    ]
    assert prior.evidence_state_codes.tolist() == [
        CurrentEvidenceState.REVOKED_VISIBLE_FREE,
        CurrentEvidenceState.REPLACED_BY_CURRENT,
        CurrentEvidenceState.REVOKED_VISIBLE_FREE,
        CurrentEvidenceState.HISTORICAL_UNOBSERVED,
    ]
    assert not prior.current_valid.flags.writeable
    assert not prior.retirement_reason_codes.flags.writeable
    assert not prior.evidence_state_codes.flags.writeable


def test_b3_policy_rejects_incomplete_or_duplicate_native_row_coverage(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.jsonl"
    _write_records(
        missing,
        [
            _record(
                owner="ovimap:1",
                source_indices=[0],
                decision="retain_t0_occluded",
                visibility="occluded",
            )
        ],
    )
    with pytest.raises(ValueError, match="cover every t0 native row"):
        load_b3_surface_policy(missing, np.asarray([1, 1], dtype=np.int64))

    duplicate = tmp_path / "duplicate.jsonl"
    record = _record(
        owner="ovimap:1",
        source_indices=[0],
        decision="retain_t0_occluded",
        visibility="occluded",
    )
    _write_records(duplicate, [record, record])
    with pytest.raises(ValueError, match="assigned more than once"):
        load_b3_surface_policy(duplicate, np.asarray([1], dtype=np.int64))


def test_fine_evidence_preserves_b3_states_and_refines_only_free_candidates() -> None:
    states = np.asarray(
        [
            CoarseSurfaceState.RETAINED_OCCLUDED,
            CoarseSurfaceState.RETAINED_UNOBSERVED,
            CoarseSurfaceState.REPLACED_OCCUPIED,
            CoarseSurfaceState.VISIBLE_FREE_CANDIDATE,
            CoarseSurfaceState.VISIBLE_FREE_CANDIDATE,
        ],
        dtype=np.uint8,
    )
    candidate_rows = np.asarray([3, 4], dtype=np.int64)
    observation = FineSurfaceObservation(
        evidence=FineSurfaceEvidence(
            present_observations=np.asarray([0, 1], dtype=np.uint16),
            visible_absent_observations=np.asarray([3, 0], dtype=np.uint16),
            occluded_observations=np.asarray([0, 2], dtype=np.uint16),
            distinct_absent_viewpoints=np.asarray([2, 0], dtype=np.uint8),
            last_supported_frames=np.asarray([-1, 1400], dtype=np.int32),
            last_absent_frames=np.asarray([1390, -1], dtype=np.int32),
            last_occluded_frames=np.asarray([-1, 1395], dtype=np.int32),
        ),
        observed_rgb_uint8=np.zeros((2, 3), dtype=np.uint8),
        rgb_valid=np.asarray([False, True]),
        best_rgb_depth_residual_m=np.asarray([np.inf, 0.01], dtype=np.float32),
    )

    evidence = assemble_fine_surface_evidence(
        states=states,
        candidate_rows=candidate_rows,
        candidate_observation=observation,
        historical_last_supported_frames=np.asarray(
            [900, 901, 902, 903, 904], dtype=np.int32
        ),
    )

    assert evidence.present_observations.tolist() == [0, 0, 1, 0, 1]
    assert evidence.visible_absent_observations.tolist() == [0, 0, 0, 3, 0]
    assert evidence.occluded_observations.tolist() == [1, 0, 0, 0, 2]
    assert evidence.distinct_absent_viewpoints.tolist() == [0, 0, 0, 2, 0]
    assert evidence.last_supported_frames.tolist() == [900, 901, 902, 903, 1400]
    assert evidence.last_absent_frames.tolist() == [-1, -1, -1, 1390, -1]
    assert evidence.last_occluded_frames.tolist() == [-1, -1, -1, -1, 1395]
