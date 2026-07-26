from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError

import pytest

from src.oviv2.temporal_config import (
    ExecutionProfile,
    TemporalBackgroundLedgerConfig,
    TemporalDynamicConfig,
    TemporalGeometryEpochConfig,
    TemporalIdentityConfig,
    TemporalMotionConfig,
    TemporalProposalConfig,
    TemporalReadoutConfig,
    temporal_config_from_json,
    temporal_config_to_json,
)


@pytest.fixture
def valid_config() -> dict[str, object]:
    return {
        "temporal_readout": {
            "execution_profile": "a4",
            "components": {
                "lifecycle": "probabilistic_hysteresis",
                "proposal": "temporal_recovery",
                "identity": "dormant_reid",
                "geometry": "object_submap_epoch",
                "background": "reversible_ledger",
                "motion": "gated_icp",
            },
            "lifecycle": {
                "initial_log_odds": 0.0,
                "present_log_likelihood": 1.2,
                "absent_log_likelihood": -0.8,
                "log_odds_limit": 6.0,
                "decay_half_life_seconds": 30.0,
                "active_on_probability": 0.75,
                "dormant_off_probability": 0.35,
                "minimum_absent_streak": 2,
                "minimum_distinct_view_bins": 3,
                "visibility_depth_tolerance_m": 0.1,
                "minimum_visible_pixel_count": 10,
                "minimum_visible_fraction": 0.05,
                "view_bin_azimuth_count": 8,
                "view_bin_elevation_count": 4,
            },
            "association": {
                "visual_weight": 0.35,
                "semantic_weight": 0.25,
                "size_weight": 0.1,
                "motion_weight": 0.1,
                "geometry_weight": 0.2,
                "minimum_score": 0.55,
                "maximum_centroid_distance_m": 1.5,
                "semantic_conflict_probability": 0.9,
                "conflict_override_visual": 0.95,
                "conflict_override_geometry": 0.85,
            },
            "geometry": {
                "voxel_size_m": 0.05,
                "depth_max_m": 10.0,
                "maximum_entities": 500,
                "maximum_object_voxels": 20_000,
                "maximum_visibility_points_per_entity": 10_000,
                "background_block_count": 100_000,
                "background_mask_dilation_px": 2,
                "minimum_icp_points": 20,
                "minimum_icp_fitness": 0.5,
                "maximum_icp_rmse_m": 0.2,
                "maximum_motion_m": 2.0,
            },
            "proposal": {
                "minimum_residual_area_px": 32,
                "maximum_recovered_proposals": 16,
                "search_region_expansion_m": 0.25,
                "minimum_depth_residual_m": 0.1,
            },
            "identity": {
                "maximum_identities": 1000,
                "maximum_dormant_frames": 600,
                "minimum_reid_similarity": 0.8,
                "maximum_reid_distance_m": 3.0,
            },
            "dynamic_state": {
                "minimum_consecutive_motion_frames": 2,
                "displacement_floor_m": 0.1,
                "minimum_motion_confidence": 0.7,
                "static_off_streak_frames": 10,
            },
            "motion": {
                "minimum_translation_confidence": 0.6,
                "maximum_translation_residual_m": 0.25,
                "require_explicit_rejection": True,
            },
            "geometry_epoch": {
                "maximum_epochs_per_identity": 4,
                "maximum_retained_epochs": 1000,
            },
            "background_ledger": {
                "maximum_journal_blocks": 50000,
                "commit_support_frames": 2,
                "commit_distinct_view_bins": 2,
                "minimum_commit_frame_gap": 1,
                "maximum_records_per_block": 8,
            },
        }
    }


def _set(
    config: dict[str, object], group: str, field: str, value: object
) -> dict[str, object]:
    changed = copy.deepcopy(config)
    changed["temporal_readout"][group][field] = value  # type: ignore[index]
    return changed


def test_valid_config_round_trips_and_is_frozen(
    valid_config: dict[str, object],
) -> None:
    parsed = temporal_config_from_json(valid_config)

    assert isinstance(parsed, TemporalReadoutConfig)
    assert temporal_config_from_json(temporal_config_to_json(parsed)) == parsed
    with pytest.raises(FrozenInstanceError):
        parsed.lifecycle.initial_log_odds = 1.0  # type: ignore[misc]
    for group in (
        parsed.proposal,
        parsed.identity,
        parsed.dynamic_state,
        parsed.motion,
        parsed.geometry_epoch,
        parsed.background_ledger,
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(group, next(iter(group.__dict__)), 0)


@pytest.mark.parametrize(
    "profile_id,expected_components",
    [
        (
            "a0",
            {
                "lifecycle": "v1_native",
                "proposal": "v1_native",
                "identity": "v1_native",
                "geometry": "v1_cumulative",
                "background": "v1_cumulative",
                "motion": "none",
            },
        ),
        (
            "a1",
            {
                "lifecycle": "probabilistic_hysteresis",
                "proposal": "v1_native",
                "identity": "v1_identity",
                "geometry": "v1_cumulative",
                "background": "v1_cumulative",
                "motion": "none",
            },
        ),
        (
            "a2",
            {
                "lifecycle": "probabilistic_hysteresis",
                "proposal": "temporal_recovery",
                "identity": "active_uncertain",
                "geometry": "object_submap_epoch",
                "background": "v1_cumulative",
                "motion": "translation",
            },
        ),
        (
            "a3",
            {
                "lifecycle": "probabilistic_hysteresis",
                "proposal": "temporal_recovery",
                "identity": "active_uncertain",
                "geometry": "object_submap_epoch",
                "background": "reversible_ledger",
                "motion": "translation",
            },
        ),
        (
            "a4",
            {
                "lifecycle": "probabilistic_hysteresis",
                "proposal": "temporal_recovery",
                "identity": "dormant_reid",
                "geometry": "object_submap_epoch",
                "background": "reversible_ledger",
                "motion": "gated_icp",
            },
        ),
    ],
)
def test_execution_profiles_have_canonical_derived_axes(
    valid_config: dict[str, object],
    profile_id: str,
    expected_components: dict[str, str],
) -> None:
    temporal = valid_config["temporal_readout"]
    temporal["execution_profile"] = profile_id  # type: ignore[index]
    temporal["components"] = expected_components  # type: ignore[index]

    profile = temporal_config_from_json(valid_config).execution_profile

    assert profile.profile_id == profile_id
    assert profile.components == expected_components
    assert ExecutionProfile.from_id(profile_id) is profile
    with pytest.raises(AttributeError):
        profile.motion_mode = "custom"  # type: ignore[misc]


def test_direct_config_construction_defaults_to_safe_a4(
    valid_config: dict[str, object],
) -> None:
    parsed = temporal_config_from_json(valid_config)

    direct = TemporalReadoutConfig(
        lifecycle=parsed.lifecycle,
        association=parsed.association,
        geometry=parsed.geometry,
        proposal=parsed.proposal,
        identity=parsed.identity,
        dynamic_state=parsed.dynamic_state,
        motion=parsed.motion,
        geometry_epoch=parsed.geometry_epoch,
        background_ledger=parsed.background_ledger,
    )

    assert direct.execution_profile is ExecutionProfile.A4


@pytest.mark.parametrize("value", ["a5", "A4", 4, None, {"id": "a4"}])
def test_execution_profile_rejects_unknown_or_non_string_values(
    valid_config: dict[str, object], value: object
) -> None:
    valid_config["temporal_readout"]["execution_profile"] = value  # type: ignore[index]

    expected_error = TypeError if not isinstance(value, str) else ValueError
    with pytest.raises(expected_error, match="execution_profile"):
        temporal_config_from_json(valid_config)


def test_execution_profile_is_required_in_json(valid_config: dict[str, object]) -> None:
    del valid_config["temporal_readout"]["execution_profile"]  # type: ignore[index]

    with pytest.raises(ValueError, match="missing.*execution_profile"):
        temporal_config_from_json(valid_config)


def test_execution_profile_serializes_as_canonical_id(
    valid_config: dict[str, object],
) -> None:
    valid_config["temporal_readout"]["execution_profile"] = "a2"  # type: ignore[index]
    valid_config["temporal_readout"]["components"] = dict(  # type: ignore[index]
        ExecutionProfile.A2.components
    )

    serialized = temporal_config_to_json(temporal_config_from_json(valid_config))

    assert (
        serialized["temporal_readout"]["execution_profile"]  # type: ignore[index]
        == "a2"
    )


def test_profile_component_override_is_rejected(
    valid_config: dict[str, object],
) -> None:
    valid_config["temporal_readout"]["execution_profile"] = "a2"  # type: ignore[index]
    temporal = valid_config["temporal_readout"]
    temporal["components"] = dict(ExecutionProfile.A2.components)  # type: ignore[index]
    temporal["components"]["identity"] = "dormant_reid"  # type: ignore[index]

    with pytest.raises(ValueError, match="profile components"):
        temporal_config_from_json(valid_config)


def test_new_groups_parse_as_frozen_contract_types(
    valid_config: dict[str, object],
) -> None:
    parsed = temporal_config_from_json(valid_config)

    assert isinstance(parsed.proposal, TemporalProposalConfig)
    assert isinstance(parsed.identity, TemporalIdentityConfig)
    assert isinstance(parsed.dynamic_state, TemporalDynamicConfig)
    assert isinstance(parsed.motion, TemporalMotionConfig)
    assert isinstance(parsed.geometry_epoch, TemporalGeometryEpochConfig)
    assert isinstance(parsed.background_ledger, TemporalBackgroundLedgerConfig)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: config.update({"unexpected": 1}),
        lambda config: config["temporal_readout"].update({"unexpected": 1}),
        lambda config: config["temporal_readout"]["lifecycle"].update(
            {"unexpected": 1}
        ),
        lambda config: config["temporal_readout"]["association"].update(
            {"unexpected": 1}
        ),
        lambda config: config["temporal_readout"]["geometry"].update(
            {"unexpected": 1}
        ),
        lambda config: config["temporal_readout"]["components"].update(
            {"unexpected": "none"}
        ),
        lambda config: config["temporal_readout"]["proposal"].update(
            {"unexpected": 1}
        ),
        lambda config: config["temporal_readout"]["identity"].update(
            {"unexpected": 1}
        ),
        lambda config: config["temporal_readout"]["dynamic_state"].update(
            {"unexpected": 1}
        ),
        lambda config: config["temporal_readout"]["motion"].update(
            {"unexpected": 1}
        ),
        lambda config: config["temporal_readout"]["geometry_epoch"].update(
            {"unexpected": 1}
        ),
        lambda config: config["temporal_readout"]["background_ledger"].update(
            {"unexpected": 1}
        ),
    ],
)
def test_unknown_keys_are_rejected(valid_config: dict[str, object], mutate) -> None:
    mutate(valid_config)
    with pytest.raises(ValueError, match="unknown"):
        temporal_config_from_json(valid_config)


@pytest.mark.parametrize(
    "group,field",
    [
        ("lifecycle", "initial_log_odds"),
        ("association", "visual_weight"),
        ("geometry", "voxel_size_m"),
        ("proposal", "minimum_residual_area_px"),
        ("identity", "maximum_identities"),
        ("dynamic_state", "minimum_consecutive_motion_frames"),
        ("motion", "require_explicit_rejection"),
        ("geometry_epoch", "maximum_epochs_per_identity"),
        ("background_ledger", "commit_support_frames"),
    ],
)
def test_missing_fields_are_rejected(
    valid_config: dict[str, object], group: str, field: str
) -> None:
    del valid_config["temporal_readout"][group][field]  # type: ignore[index]
    with pytest.raises(ValueError, match="missing"):
        temporal_config_from_json(valid_config)


@pytest.mark.parametrize(
    "group,field",
    [
        ("lifecycle", "minimum_absent_streak"),
        ("lifecycle", "active_on_probability"),
        ("association", "visual_weight"),
        ("geometry", "maximum_entities"),
        ("geometry", "voxel_size_m"),
        ("proposal", "maximum_recovered_proposals"),
        ("identity", "minimum_reid_similarity"),
        ("dynamic_state", "minimum_motion_confidence"),
        ("motion", "minimum_translation_confidence"),
        ("geometry_epoch", "maximum_epochs_per_identity"),
        ("background_ledger", "commit_support_frames"),
    ],
)
def test_bool_cannot_masquerade_as_a_number(
    valid_config: dict[str, object], group: str, field: str
) -> None:
    with pytest.raises(TypeError, match=field):
        temporal_config_from_json(_set(valid_config, group, field, True))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_numbers_are_rejected(
    valid_config: dict[str, object], value: float
) -> None:
    with pytest.raises(ValueError, match="finite"):
        temporal_config_from_json(
            _set(valid_config, "association", "minimum_score", value)
        )


def test_boolean_fields_require_exact_bool(valid_config: dict[str, object]) -> None:
    with pytest.raises(TypeError, match="require_explicit_rejection"):
        temporal_config_from_json(
            _set(valid_config, "motion", "require_explicit_rejection", 1)
        )


@pytest.mark.parametrize(
    "field,values",
    [
        ("minimum_consecutive_motion_frames", (2, 3, 4)),
        ("displacement_floor_m", (0.05, 0.10, 0.15)),
        ("minimum_motion_confidence", (0.6, 0.7, 0.8)),
        ("static_off_streak_frames", (5, 10, 20)),
    ],
)
def test_dynamic_search_grid_is_exact(
    valid_config: dict[str, object], field: str, values: tuple[object, ...]
) -> None:
    for value in values:
        temporal_config_from_json(_set(valid_config, "dynamic_state", field, value))
    rejected = (
        5
        if field in {"minimum_consecutive_motion_frames", "static_off_streak_frames"}
        else 0.11
    )
    if field == "static_off_streak_frames":
        rejected = 6
    with pytest.raises(ValueError, match=field):
        temporal_config_from_json(_set(valid_config, "dynamic_state", field, rejected))


@pytest.mark.parametrize(
    "field,values",
    [
        ("commit_support_frames", (2, 3, 4)),
        ("commit_distinct_view_bins", (2, 3)),
    ],
)
def test_background_ledger_search_grid_is_exact(
    valid_config: dict[str, object], field: str, values: tuple[int, ...]
) -> None:
    for value in values:
        temporal_config_from_json(
            _set(valid_config, "background_ledger", field, value)
        )
    with pytest.raises(ValueError, match=field):
        temporal_config_from_json(_set(valid_config, "background_ledger", field, 5))


def test_capacity_relationships_are_enforced(valid_config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="maximum_identities"):
        temporal_config_from_json(
            _set(valid_config, "identity", "maximum_identities", 499)
        )
    with pytest.raises(ValueError, match="maximum_retained_epochs"):
        temporal_config_from_json(
            _set(valid_config, "geometry_epoch", "maximum_retained_epochs", 3)
        )
    with pytest.raises(ValueError, match="maximum_journal_blocks"):
        temporal_config_from_json(
            _set(valid_config, "background_ledger", "maximum_journal_blocks", 100001)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_entities", -1),
        ("maximum_object_voxels", -1),
        ("maximum_visibility_points_per_entity", -1),
        ("background_block_count", -1),
        ("background_mask_dilation_px", -1),
    ],
)
def test_negative_capacities_are_rejected(
    valid_config: dict[str, object], field: str, value: int
) -> None:
    with pytest.raises(ValueError, match=field):
        temporal_config_from_json(_set(valid_config, "geometry", field, value))


@pytest.mark.parametrize(
    "group,field,value",
    [
        ("lifecycle", "active_on_probability", 1.01),
        ("lifecycle", "minimum_visible_fraction", -0.01),
        ("association", "minimum_score", -0.01),
        ("association", "semantic_conflict_probability", 1.01),
        ("geometry", "minimum_icp_fitness", 1.01),
    ],
)
def test_probability_score_and_fraction_ranges_are_enforced(
    valid_config: dict[str, object], group: str, field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        temporal_config_from_json(_set(valid_config, group, field, value))


def test_voxel_size_must_fit_visibility_tolerance(
    valid_config: dict[str, object],
) -> None:
    config = _set(valid_config, "geometry", "voxel_size_m", 0.2)
    with pytest.raises(ValueError, match="visibility_depth_tolerance_m"):
        temporal_config_from_json(config)


def test_dormant_threshold_must_be_below_active_threshold(
    valid_config: dict[str, object],
) -> None:
    config = _set(valid_config, "lifecycle", "dormant_off_probability", 0.75)
    with pytest.raises(ValueError, match="dormant_off_probability"):
        temporal_config_from_json(config)


def test_required_view_bins_cannot_exceed_available_bins(
    valid_config: dict[str, object],
) -> None:
    config = _set(valid_config, "lifecycle", "minimum_distinct_view_bins", 33)
    with pytest.raises(ValueError, match="minimum_distinct_view_bins"):
        temporal_config_from_json(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_entities", 100_001),
        ("maximum_object_voxels", 100_001),
        ("maximum_visibility_points_per_entity", 100_001),
    ],
)
def test_per_entity_and_entity_capacities_cannot_exceed_background_capacity(
    valid_config: dict[str, object], field: str, value: int
) -> None:
    with pytest.raises(ValueError, match=field):
        temporal_config_from_json(_set(valid_config, "geometry", field, value))


def test_at_least_one_association_weight_must_be_positive(
    valid_config: dict[str, object],
) -> None:
    for field in (
        "visual_weight",
        "semantic_weight",
        "size_weight",
        "motion_weight",
        "geometry_weight",
    ):
        valid_config = _set(valid_config, "association", field, 0.0)
    with pytest.raises(ValueError, match="weight"):
        temporal_config_from_json(valid_config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_log_odds", 6.1),
        ("present_log_likelihood", 0.0),
        ("absent_log_likelihood", 0.0),
    ],
)
def test_log_odds_values_have_consistent_sign_and_bounds(
    valid_config: dict[str, object], field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        temporal_config_from_json(_set(valid_config, "lifecycle", field, value))


def test_serialization_is_deterministic(valid_config: dict[str, object]) -> None:
    parsed = temporal_config_from_json(valid_config)
    first = json.dumps(
        temporal_config_to_json(parsed), sort_keys=True, separators=(",", ":")
    )
    second = json.dumps(
        temporal_config_to_json(parsed), sort_keys=True, separators=(",", ":")
    )

    assert first == second
