from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from itertools import permutations
import math
import time

import numpy as np
import pytest

import src.oviv2.temporal_proposals as temporal_proposals
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


SEMANTIC_HASH = "a" * 64


def projection_hash(identity_id: int) -> str:
    return f"{identity_id:064x}"


def region(identity_id: int, mask: np.ndarray, expected: np.ndarray, source: int = 2):
    return ProjectedIdentitySearchRegion(
        identity_id=identity_id,
        source_frame_id=source,
        mask=mask,
        expected_depth_m=expected,
        projection_provenance_hash=projection_hash(identity_id),
    )


def recovery_input(*regions: ProjectedIdentitySearchRegion, **changes: object):
    depth = np.full((4, 5), 3.0, dtype=np.float32)
    rows, columns = np.indices(depth.shape)
    xyz = np.stack((columns, rows, depth), axis=-1).astype(np.float64)
    values: dict[str, object] = {
        "frame_id": 3,
        "timestamp": 3.0,
        "depth_m": depth,
        "current_xyz": xyz,
        "segmentation_occupied": np.zeros((4, 5), dtype=bool),
        "semantic_support": np.ones((4, 5), dtype=bool),
        "semantic_source_frame_id": 3,
        "semantic_provenance_hash": SEMANTIC_HASH,
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
    assert proposal.semantic_provenance_hash == SEMANTIC_HASH
    assert proposal.appearance_available is False
    assert proposal.appearance_model_id is None
    assert proposal.centroid_xyz == pytest.approx((1.5, 1.5, 3.0))
    assert not proposal.mask.flags.writeable
    assert result.opportunity_records == ("proposal:3:7:6",)
    assert result.trigger_records == result.opportunity_records


def test_component_centroid_stays_within_bounds_when_float64_sum_rounds_outward() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[0, :3] = True
    xyz = np.full((4, 5, 3), 0.1, dtype=np.float64)

    result = recover_temporal_proposals(
        recovery_input(
            region(7, mask, np.full((4, 5), 4.0)),
            current_xyz=xyz,
        ),
        config(),
    )

    proposal = result.proposals[0]
    assert proposal.centroid_xyz == (0.1, 0.1, 0.1)
    assert proposal.bounds_min_xyz == (0.1, 0.1, 0.1)
    assert proposal.bounds_max_xyz == (0.1, 0.1, 0.1)


def test_point_set_geometry_centroid_is_permutation_invariant_under_cancellation() -> None:
    centroids = []
    for values in permutations((1.0e16, 1.0, -1.0e16)):
        xyz = np.zeros((3, 3), dtype=np.float64)
        xyz[:, 0] = values
        centroid, _, _ = temporal_proposals._point_set_geometry(xyz)
        centroids.append(centroid)

    assert all(item.dtype == np.float64 for item in centroids)
    assert all(np.array_equal(item, centroids[0]) for item in centroids)
    assert centroids[0][0] == 1.0 / 3.0


def test_point_set_geometry_preserves_tiny_residual_when_extremes_cancel() -> None:
    maximum = np.finfo(np.float64).max
    xyz = np.zeros((3, 3), dtype=np.float64)
    xyz[:, 0] = (maximum, -maximum, 1.0e-100)

    centroid, _, _ = temporal_proposals._point_set_geometry(xyz)

    assert centroid[0] == math.fsum((maximum, -maximum, 1.0e-100)) / 3.0
    assert centroid[0] > 0.0


@pytest.mark.parametrize(
    "value",
    (np.finfo(np.float64).max, -np.finfo(np.float64).max),
)
def test_point_set_geometry_fallback_averages_many_same_sign_maxima(
    value: float,
) -> None:
    xyz = np.full((105, 3), value, dtype=np.float64)

    centroid, bounds_min, bounds_max = temporal_proposals._point_set_geometry(xyz)

    assert np.array_equal(centroid, np.full(3, value, dtype=np.float64))
    assert np.array_equal(bounds_min, centroid)
    assert np.array_equal(bounds_max, centroid)


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        ((0.0, 0.0, 0.0), 0.0),
        ((np.nextafter(0.0, 1.0),), np.nextafter(0.0, 1.0)),
        ((np.finfo(np.float64).max, np.finfo(np.float64).max), np.finfo(np.float64).max),
        ((0.0, 0.0, 8.0), 8.0 / 3.0),
    ),
)
def test_point_set_geometry_handles_finite_boundaries_and_uses_arithmetic_mean(
    values: tuple[float, ...],
    expected: float,
) -> None:
    xyz = np.zeros((len(values), 3), dtype=np.float64)
    xyz[:, 0] = values

    centroid, bounds_min, bounds_max = temporal_proposals._point_set_geometry(xyz)

    assert centroid.dtype == bounds_min.dtype == bounds_max.dtype == np.float64
    assert np.isfinite(centroid).all()
    assert centroid[0] == expected
    assert centroid[1:].tolist() == [0.0, 0.0]


def test_point_set_geometry_only_corrects_one_ulp_of_mean_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xyz = np.ones((1, 3), dtype=np.float64)
    one_ulp_high = np.nextafter(1.0, np.inf)
    monkeypatch.setattr(temporal_proposals.math, "fsum", lambda _values: one_ulp_high)
    centroid, _, _ = temporal_proposals._point_set_geometry(xyz)
    assert np.array_equal(centroid, np.ones(3, dtype=np.float64))

    two_ulps_high = np.nextafter(one_ulp_high, np.inf)
    monkeypatch.setattr(temporal_proposals.math, "fsum", lambda _values: two_ulps_high)
    with pytest.raises(ArithmeticError, match="one ULP"):
        temporal_proposals._point_set_geometry(xyz)


@pytest.mark.parametrize(
    "xyz",
    (
        np.empty((0, 3), dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2, 3, 1), dtype=np.float64),
    ),
)
def test_point_set_geometry_rejects_empty_or_non_nx3_arrays(
    xyz: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="non-empty Nx3"):
        temporal_proposals._point_set_geometry(xyz)


@pytest.mark.parametrize("invalid", (np.nan, np.inf, -np.inf))
def test_point_set_geometry_rejects_nonfinite_points_directly(invalid: float) -> None:
    xyz = np.zeros((2, 3), dtype=np.float64)
    xyz[1, 2] = invalid

    with pytest.raises(ValueError, match="only finite"):
        temporal_proposals._point_set_geometry(xyz)


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


def test_support_sources_enforce_current_and_projected_region_enforces_past() -> None:
    mask = np.ones((4, 5), dtype=bool)
    with pytest.raises(ValueError, match="current frame"):
        recovery_input(region(1, mask, np.full((4, 5), 4.0)), semantic_source_frame_id=2)
    with pytest.raises(ValueError, match="strictly past"):
        recovery_input(region(1, mask, np.full((4, 5), 4.0), source=3))
    with pytest.raises(ValueError, match="current frame"):
        recovery_input(
            region(1, mask, np.full((4, 5), 4.0)),
            appearance_support=mask,
            appearance_source_frame_id=2,
            appearance_model_id="clip",
            appearance_provenance_hash="b" * 64,
        )


def test_only_currently_nearer_semantic_foreground_can_form_proposals() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[0, :2] = True
    behind = recovery_input(region(1, mask, np.full((4, 5), 2.0)))
    assert recover_temporal_proposals(behind, config()).proposals == ()

    appearance_only = recovery_input(
        region(1, mask, np.full((4, 5), 4.0)),
        semantic_support=np.zeros((4, 5), dtype=bool),
        appearance_support=mask,
        appearance_source_frame_id=3,
        appearance_model_id="clip",
        appearance_provenance_hash="b" * 64,
    )
    assert recover_temporal_proposals(appearance_only, config()).opportunity_count == 0


def test_metric_expansion_changes_region_at_configured_3d_boundary() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 1] = True
    semantic = np.zeros((4, 5), dtype=bool)
    semantic[1, 1:3] = True
    value = recovery_input(
        region(1, mask, np.full((4, 5), 4.0)), semantic_support=semantic
    )
    assert recover_temporal_proposals(
        value, config(minimum_residual_area_px=1, search_region_expansion_m=0.0)
    ).proposals[0].area_px == 1
    assert recover_temporal_proposals(
        value, config(minimum_residual_area_px=1, search_region_expansion_m=1.0)
    ).proposals[0].area_px == 2


def test_metric_expansion_matches_exact_fixture_and_uses_one_batched_tree_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, columns = np.indices((120, 160))
    xyz = np.stack((columns * 0.01, rows * 0.01, np.ones_like(rows)), axis=-1)
    mask = np.zeros((120, 160), dtype=bool)
    mask[40:80, 55:105] = True
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original = temporal_proposals._nearest_metric_distance

    def spy(source_xyz: np.ndarray, query_xyz: np.ndarray) -> np.ndarray:
        calls.append((source_xyz.shape, query_xyz.shape))
        return original(source_xyz, query_xyz)

    monkeypatch.setattr(temporal_proposals, "_nearest_metric_distance", spy)
    started = time.perf_counter()
    expanded = temporal_proposals._expand_metric_region(mask, xyz.astype(np.float64), 0.02)
    elapsed = time.perf_counter() - started
    assert calls == [((2000, 3), (19200, 3))]
    assert expanded[39, 55] and expanded[80, 104]
    assert not expanded[37, 55]
    assert elapsed < 1.0


def test_zero_depth_is_invalid_only_for_eligibility_and_expected_depth_is_local() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 1:3] = True
    expected = np.zeros((4, 5), dtype=np.float64)
    expected[mask] = 4.0
    depth = np.full((4, 5), 3.0, dtype=np.float32)
    depth[0, 0] = 0.0
    value = recovery_input(region(1, mask, expected), depth_m=depth)
    assert recover_temporal_proposals(value, config()).proposals[0].area_px == 2

    depth[1, 1] = 0.0
    value = recovery_input(region(1, mask, expected), depth_m=depth)
    assert recover_temporal_proposals(
        value, config(minimum_residual_area_px=1)
    ).proposals[0].area_px == 1

    item = region(1, mask, expected)
    assert item == item
    assert item == replace(item, expected_depth_m=expected.copy())
    for invalid in (np.nan, np.inf):
        invalid_outside = expected.copy()
        invalid_outside[0, 0] = invalid
        with pytest.raises(ValueError, match="finite"):
            region(1, mask, invalid_outside)
        invalid_inside = expected.copy()
        invalid_inside[1, 1] = invalid
        with pytest.raises(ValueError, match="finite"):
            region(1, mask, invalid_inside)


def test_nan_and_inf_current_depth_still_fail_closed() -> None:
    for invalid in (np.nan, np.inf):
        depth = np.full((4, 5), 3.0, dtype=np.float32)
        depth[0, 0] = invalid
        with pytest.raises(ValueError, match="finite"):
            recovery_input(depth_m=depth)


def test_appearance_provenance_is_component_local() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[0, :2] = True
    mask[3, 3:5] = True
    appearance = np.zeros((4, 5), dtype=bool)
    appearance[0, 0] = True
    result = recover_temporal_proposals(
        recovery_input(
            region(1, mask, np.full((4, 5), 4.0)),
            appearance_support=appearance,
            appearance_source_frame_id=3,
            appearance_model_id="clip",
            appearance_provenance_hash="b" * 64,
        ),
        config(),
    )
    assert tuple(item.appearance_available for item in result.proposals) == (True, False)
    assert result.proposals[1].appearance_provenance_hash is None


def test_same_identity_regions_merge_once_and_use_latest_contributing_provenance() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:3] = True
    regions = (
        ProjectedIdentitySearchRegion(7, 1, mask, np.full((4, 5), 4.0), "1" * 64),
        ProjectedIdentitySearchRegion(7, 2, mask, np.full((4, 5), 4.0), "2" * 64),
    )
    expected = recover_temporal_proposals(recovery_input(*regions), config())
    assert (expected.opportunity_count, expected.trigger_count) == (1, 1)
    assert expected.proposals[0].projection_source_frame_id == 2
    assert expected.proposals[0].projection_provenance_hash == "2" * 64
    assert recover_temporal_proposals(
        recovery_input(*reversed(regions)), config()
    ) == expected


def test_same_identity_regions_cannot_cross_join_source_and_residual_evidence() -> None:
    left = np.zeros((4, 5), dtype=bool)
    left[0, 0] = True
    right = np.zeros((4, 5), dtype=bool)
    right[0, 2] = True
    expected_left = np.zeros((4, 5), dtype=np.float64)
    expected_left[0, 0] = 3.1
    expected_left[0, 2] = 4.0
    expected_right = np.zeros((4, 5), dtype=np.float64)
    expected_right[0, 0] = 4.0
    expected_right[0, 2] = 3.1
    first = ProjectedIdentitySearchRegion(7, 1, left, expected_left, "1" * 64)
    second = ProjectedIdentitySearchRegion(7, 2, right, expected_right, "2" * 64)
    strict = config(
        minimum_residual_area_px=1,
        search_region_expansion_m=0.0,
    )
    assert recover_temporal_proposals(recovery_input(first), strict).opportunity_count == 0
    assert recover_temporal_proposals(recovery_input(second), strict).opportunity_count == 0
    assert recover_temporal_proposals(
        recovery_input(first, second), strict
    ).opportunity_count == 0


def test_same_identity_atomic_contributions_merge_and_choose_latest_actual_region() -> None:
    left = np.zeros((4, 5), dtype=bool)
    left[0, 0:2] = True
    right = np.zeros((4, 5), dtype=bool)
    right[0, 1:3] = True
    expected_left = np.zeros((4, 5), dtype=np.float64)
    expected_left[left] = 4.0
    expected_right = np.zeros((4, 5), dtype=np.float64)
    expected_right[right] = 4.0
    first = ProjectedIdentitySearchRegion(7, 1, left, expected_left, "1" * 64)
    second = ProjectedIdentitySearchRegion(7, 2, right, expected_right, "2" * 64)
    result = recover_temporal_proposals(
        recovery_input(second, first),
        config(minimum_residual_area_px=1, search_region_expansion_m=0.0),
    )
    assert (result.opportunity_count, result.proposals[0].area_px) == (1, 3)
    assert result.proposals[0].projection_source_frame_id == 2
    assert result.proposals[0].projection_provenance_hash == "2" * 64


def test_workload_bound_rejects_before_metric_query(monkeypatch: pytest.MonkeyPatch) -> None:
    mask = np.ones((4, 5), dtype=bool)
    regions = tuple(
        ProjectedIdentitySearchRegion(
            identity_id,
            2,
            mask,
            np.full((4, 5), 4.0),
            f"{identity_id:064x}",
        )
        for identity_id in (1, 2, 3)
    )

    def forbidden(*_args: object) -> np.ndarray:
        raise AssertionError("metric query ran before workload rejection")

    monkeypatch.setattr(temporal_proposals, "_nearest_metric_distance", forbidden)
    with pytest.raises(OverflowError, match="workload"):
        recover_temporal_proposals(
            recovery_input(*regions),
            config(
                minimum_residual_area_px=1,
                maximum_recovered_proposals=1,
                search_region_expansion_m=1.0,
            ),
        )


def test_workload_bound_allows_exactly_four_frame_equivalents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = np.ones((4, 5), dtype=bool)
    regions = tuple(
        ProjectedIdentitySearchRegion(
            identity_id, 2, mask, np.full((4, 5), 4.0), f"{identity_id:064x}"
        )
        for identity_id in (1, 2)
    )
    calls = 0
    original = temporal_proposals._nearest_metric_distance

    def spy(source_xyz: np.ndarray, query_xyz: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(source_xyz, query_xyz)

    monkeypatch.setattr(temporal_proposals, "_nearest_metric_distance", spy)
    recover_temporal_proposals(
        recovery_input(*regions),
        config(
            minimum_residual_area_px=1,
            maximum_recovered_proposals=1,
            search_region_expansion_m=1.0,
        ),
    )
    assert calls == 2


def test_region_count_bound_is_capacity_times_four_before_metric_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[0, 0] = True
    regions = tuple(
        ProjectedIdentitySearchRegion(
            identity_id, 2, mask, np.full((4, 5), 4.0), f"{identity_id:064x}"
        )
        for identity_id in range(1, 6)
    )

    def forbidden(*_args: object) -> np.ndarray:
        raise AssertionError("metric query ran before region bound rejection")

    monkeypatch.setattr(temporal_proposals, "_nearest_metric_distance", forbidden)
    with pytest.raises(OverflowError, match="region count"):
        recover_temporal_proposals(
            recovery_input(*regions),
            config(
                minimum_residual_area_px=1,
                maximum_recovered_proposals=1,
                search_region_expansion_m=1.0,
            ),
        )


def test_noisy_components_allocate_full_masks_only_for_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height, width = 64, 64
    rows, columns = np.indices((height, width))
    semantic = (rows + columns) % 2 == 0
    depth = np.full((height, width), 3.0, dtype=np.float32)
    xyz = np.stack((columns, rows, depth), axis=-1).astype(np.float64)
    search = np.ones((height, width), dtype=bool)
    item = ProjectedIdentitySearchRegion(
        1, 1, search, np.full((height, width), 4.0), "1" * 64
    )
    value = ProposalRecoveryInput(
        2, 2.0, depth, xyz, np.zeros_like(search), semantic, 2, "a" * 64,
        None, None, None, None, (item,),
    )
    calls: list[int] = []
    original = temporal_proposals._full_mask_from_indices

    def spy(shape: tuple[int, int], indices: tuple[int, ...]) -> np.ndarray:
        calls.append(len(indices))
        return original(shape, indices)

    monkeypatch.setattr(temporal_proposals, "_full_mask_from_indices", spy)
    result = recover_temporal_proposals(
        value,
        config(
            minimum_residual_area_px=1,
            maximum_recovered_proposals=3,
            search_region_expansion_m=0.0,
        ),
    )
    assert result.opportunity_count == int(semantic.sum())
    assert result.trigger_count == 3
    assert calls == [1, 1, 1]


@pytest.mark.parametrize("region_count", [1, 5, 10])
def test_region_specific_metric_queries_are_prefiltered_and_bounded(
    region_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height, width = 120, 160
    rows, columns = np.indices((height, width))
    depth = np.full((height, width), 3.0, dtype=np.float32)
    xyz = np.stack((columns * 0.01, rows * 0.01, depth), axis=-1)
    mask = np.zeros((height, width), dtype=bool)
    mask[40:80, 55:105] = True
    regions = tuple(
        ProjectedIdentitySearchRegion(
            1, source, mask, np.full((height, width), 4.0), f"{source:064x}"
        )
        for source in range(1, region_count + 1)
    )
    value = ProposalRecoveryInput(
        20, 20.0, depth, xyz, np.zeros_like(mask), mask.copy(),
        20, "a" * 64, None, None, None, None, regions,
    )
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original = temporal_proposals._nearest_metric_distance

    def spy(source_xyz: np.ndarray, query_xyz: np.ndarray) -> np.ndarray:
        calls.append((source_xyz.shape, query_xyz.shape))
        return original(source_xyz, query_xyz)

    monkeypatch.setattr(temporal_proposals, "_nearest_metric_distance", spy)
    result = recover_temporal_proposals(
        value,
        config(
            minimum_residual_area_px=10,
            maximum_recovered_proposals=4,
            search_region_expansion_m=0.02,
        ),
    )
    assert result.opportunity_count == 1
    assert len(calls) == region_count
    assert all(call == ((2000, 3), (2000, 3)) for call in calls)
    assert sum(source[0] + query[0] for source, query in calls) <= 4 * height * width


def test_int64_max_identity_is_owned_and_lower_identity_still_wins_overlap() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:3] = True
    maximum = 2**63 - 1
    max_region = ProjectedIdentitySearchRegion(
        maximum, 2, mask, np.full((4, 5), 4.0), "f" * 64
    )
    only_max = recover_temporal_proposals(recovery_input(max_region), config())
    assert only_max.proposals[0].identity_hint == maximum

    lower = ProjectedIdentitySearchRegion(
        1, 2, mask, np.full((4, 5), 4.0), "1" * 64
    )
    expected = recover_temporal_proposals(recovery_input(max_region, lower), config())
    assert expected.proposals[0].identity_hint == 1
    assert recover_temporal_proposals(
        recovery_input(lower, max_region), config()
    ) == expected


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
    assert replace(
        proposal,
        centroid_xyz=proposal.bounds_max_xyz,
    ).centroid_xyz == proposal.bounds_max_xyz
    with pytest.raises(ValueError, match="finite"):
        replace(proposal, centroid_xyz=(np.nan, 0.0, 0.0))
    with pytest.raises(ValueError, match="ordered bounds"):
        replace(
            proposal,
            centroid_xyz=(
                proposal.bounds_max_xyz[0] + 1.0,
                proposal.centroid_xyz[1],
                proposal.centroid_xyz[2],
            ),
        )
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
        {"semantic_provenance_hash": "../not-a-hash"},
    ],
)
def test_recovery_input_exact_validation(change: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(recovery_input(), **change)
