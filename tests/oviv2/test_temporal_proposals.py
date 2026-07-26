from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import numpy as np
import pytest

from src.oviv2.temporal_config import TemporalProposalConfig
from src.oviv2.temporal_proposals import (
    ProjectedIdentitySearchRegion,
    ProposalRecoveryInput,
    recover_temporal_proposals,
)


def config(**changes: object) -> TemporalProposalConfig:
    values: dict[str, object] = {
        "minimum_residual_area_px": 2,
        "maximum_recovered_proposals": 4,
        "search_region_expansion_m": 0.1,
        "minimum_depth_residual_m": 0.5,
    }
    values.update(changes)
    return TemporalProposalConfig(**values)  # type: ignore[arg-type]


def region(identity_id: int, mask: np.ndarray, expected: np.ndarray, source: int = 2):
    return ProjectedIdentitySearchRegion(
        identity_id=identity_id,
        source_frame_id=source,
        mask=mask,
        expected_depth_m=expected,
        projection_provenance_hash=f"projection-{identity_id}",
    )


def recovery_input(*regions: ProjectedIdentitySearchRegion, **changes: object):
    depth = np.full((4, 5), 3.0, dtype=np.float32)
    xyz = np.dstack(np.meshgrid(np.arange(5), np.arange(4))[::-1] + [depth]).astype(np.float64)
    values: dict[str, object] = {
        "frame_id": 3,
        "timestamp": 3.0,
        "depth_m": depth,
        "current_xyz": xyz,
        "segmentation_occupied": np.zeros((4, 5), dtype=bool),
        "semantic_support": np.ones((4, 5), dtype=bool),
        "semantic_source_frame_id": 3,
        "semantic_provenance_hash": "semantic-current-hash",
        "appearance_support": None,
        "appearance_source_frame_id": None,
        "appearance_model_id": None,
        "appearance_provenance_hash": None,
        "search_regions": tuple(regions),
    }
    values.update(changes)
    return ProposalRecoveryInput(**values)


def test_current_depth_residual_recovers_component_without_fake_appearance() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:3] = True
    value = recovery_input(region(7, mask, np.full((4, 5), 4.0)))
    result = recover_temporal_proposals(value, config())
    assert result.opportunity_count == 1
    assert result.trigger_count == 1
    proposal = result.proposals[0]
    assert (proposal.identity_hint, proposal.area_px) == (7, 4)
    assert proposal.semantic_provenance_hash == "semantic-current-hash"
    assert proposal.appearance_available is False
    assert proposal.appearance_model_id is None
    assert proposal.centroid_xyz == pytest.approx((1.5, 1.5, 3.0))
    assert not proposal.mask.flags.writeable


def test_segmentation_occupied_pixels_are_never_recovered_and_zero_opportunity_stays_zero() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:3] = True
    value = recovery_input(
        region(7, mask, np.full((4, 5), 4.0)),
        segmentation_occupied=mask.copy(),
    )
    result = recover_temporal_proposals(value, config())
    assert result.proposals == ()
    assert (result.opportunity_count, result.trigger_count) == (0, 0)


def test_future_sources_are_rejected_and_later_data_does_not_change_current_result() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[0, :2] = True
    current = recovery_input(region(1, mask, np.full((4, 5), 4.0)))
    expected = recover_temporal_proposals(current, config())
    future_mask = np.ones((4, 5), dtype=bool)
    assert recover_temporal_proposals(current, config()) == expected
    with pytest.raises(ValueError, match="future"):
        recovery_input(region(1, future_mask, np.full((4, 5), 4.0), source=4))
    with pytest.raises(ValueError, match="future"):
        replace(current, semantic_source_frame_id=4)


def test_permutation_overlap_connected_components_and_capacity_have_canonical_order() -> None:
    left = np.zeros((4, 5), dtype=bool)
    left[0, :3] = True
    right = np.zeros((4, 5), dtype=bool)
    right[0:2, 2:4] = True
    isolated = np.zeros((4, 5), dtype=bool)
    isolated[3, 3:5] = True
    regions = (
        region(9, left, np.full((4, 5), 4.0)),
        region(3, right, np.full((4, 5), 4.0)),
        region(5, isolated, np.full((4, 5), 4.0)),
    )
    expected = recover_temporal_proposals(recovery_input(*regions), config(maximum_recovered_proposals=2))
    assert expected.trigger_count == 2
    assert expected.opportunity_count == 3
    assert tuple(item.proposal_id for item in expected.proposals) == (0, 1)
    assert expected.proposals[0].identity_hint == 3  # overlapping pixel belongs to lower identity ID
    for order in permutations(regions):
        assert recover_temporal_proposals(recovery_input(*order), config(maximum_recovered_proposals=2)) == expected


def test_inputs_are_owned_readonly_and_invalid_input_does_not_publish_partial_output() -> None:
    mask = np.ones((4, 5), dtype=bool)
    original = mask.copy()
    item = region(1, mask, np.full((4, 5), 4.0))
    value = recovery_input(item)
    mask[:] = False
    assert np.array_equal(item.mask, original)
    assert not item.mask.flags.writeable
    with pytest.raises(FrozenInstanceError):
        item.identity_id = 2  # type: ignore[misc]
    bad_xyz = value.current_xyz.copy()
    bad_xyz[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        replace(value, current_xyz=bad_xyz)
    result = recover_temporal_proposals(value, config())
    assert result.trigger_count == 1
    proposal = result.proposals[0]
    with pytest.raises((TypeError, ValueError)):
        replace(proposal, mask=proposal.mask.astype(np.uint8))
    with pytest.raises((TypeError, ValueError)):
        replace(proposal, area_px=proposal.area_px + 1)
    with pytest.raises((TypeError, ValueError)):
        replace(result, trigger_count=np.int64(1))
    with pytest.raises(ValueError):
        replace(result, opportunity_count=0)


@pytest.mark.parametrize(
    "change",
    [
        {"frame_id": True},
        {"timestamp": np.inf},
        {"depth_m": np.ones((2, 2), dtype=np.int64)},
        {"segmentation_occupied": np.zeros((2, 2), dtype=bool)},
        {"semantic_support": np.ones((4, 5), dtype=np.uint8)},
        {"search_regions": []},
    ],
)
def test_recovery_input_exact_validation(change: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(recovery_input(), **change)
