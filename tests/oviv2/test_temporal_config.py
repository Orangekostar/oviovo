from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError

import pytest

from src.oviv2.temporal_config import (
    TemporalReadoutConfig,
    temporal_config_from_json,
    temporal_config_to_json,
)


@pytest.fixture
def valid_config() -> dict[str, object]:
    return {
        "temporal_readout": {
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
        }
    }


def _set(
    config: dict[str, object], group: str, field: str, value: object
) -> dict[str, object]:
    changed = copy.deepcopy(config)
    changed["temporal_readout"][group][field] = value  # type: ignore[index]
    return changed


def test_valid_config_round_trips_and_is_frozen(valid_config: dict[str, object]) -> None:
    parsed = temporal_config_from_json(valid_config)

    assert isinstance(parsed, TemporalReadoutConfig)
    assert temporal_config_from_json(temporal_config_to_json(parsed)) == parsed
    with pytest.raises(FrozenInstanceError):
        parsed.lifecycle.initial_log_odds = 1.0  # type: ignore[misc]


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
    ],
)
def test_unknown_keys_are_rejected(valid_config: dict[str, object], mutate) -> None:
    mutate(valid_config)
    with pytest.raises(ValueError, match="unknown"):
        temporal_config_from_json(valid_config)


@pytest.mark.parametrize("group,field", [("lifecycle", "initial_log_odds"), ("association", "visual_weight"), ("geometry", "voxel_size_m")])
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
