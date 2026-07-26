from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from numbers import Real


@dataclass(frozen=True)
class TemporalLifecycleConfig:
    initial_log_odds: float
    present_log_likelihood: float
    absent_log_likelihood: float
    log_odds_limit: float
    decay_half_life_seconds: float
    active_on_probability: float
    dormant_off_probability: float
    minimum_absent_streak: int
    minimum_distinct_view_bins: int
    visibility_depth_tolerance_m: float
    minimum_visible_pixel_count: int
    minimum_visible_fraction: float
    view_bin_azimuth_count: int
    view_bin_elevation_count: int


@dataclass(frozen=True)
class TemporalAssociationConfig:
    visual_weight: float
    semantic_weight: float
    size_weight: float
    motion_weight: float
    geometry_weight: float
    minimum_score: float
    maximum_centroid_distance_m: float
    semantic_conflict_probability: float
    conflict_override_visual: float
    conflict_override_geometry: float


@dataclass(frozen=True)
class TemporalGeometryConfig:
    voxel_size_m: float
    depth_max_m: float
    maximum_entities: int
    maximum_object_voxels: int
    maximum_visibility_points_per_entity: int
    background_block_count: int
    background_mask_dilation_px: int
    minimum_icp_points: int
    minimum_icp_fitness: float
    maximum_icp_rmse_m: float
    maximum_motion_m: float


@dataclass(frozen=True)
class TemporalProposalConfig:
    minimum_residual_area_px: int
    maximum_recovered_proposals: int
    search_region_expansion_m: float
    minimum_depth_residual_m: float


@dataclass(frozen=True)
class TemporalIdentityConfig:
    maximum_identities: int
    maximum_dormant_frames: int
    minimum_reid_similarity: float
    maximum_reid_distance_m: float


@dataclass(frozen=True)
class TemporalDynamicConfig:
    minimum_consecutive_motion_frames: int
    displacement_floor_m: float
    minimum_motion_confidence: float
    static_off_streak_frames: int


@dataclass(frozen=True)
class TemporalMotionConfig:
    minimum_translation_confidence: float
    maximum_translation_residual_m: float
    require_explicit_rejection: bool


@dataclass(frozen=True)
class TemporalGeometryEpochConfig:
    maximum_epochs_per_identity: int
    maximum_retained_epochs: int


@dataclass(frozen=True)
class TemporalBackgroundLedgerConfig:
    maximum_journal_blocks: int
    commit_support_frames: int
    commit_distinct_view_bins: int
    minimum_commit_frame_gap: int
    maximum_records_per_block: int


class ExecutionProfile(Enum):
    A0 = (
        "a0",
        "v1_native",
        "v1_native",
        "v1_native",
        "v1_cumulative",
        "v1_cumulative",
        "none",
    )
    A1 = (
        "a1",
        "probabilistic_hysteresis",
        "v1_native",
        "v1_identity",
        "v1_cumulative",
        "v1_cumulative",
        "none",
    )
    A2 = (
        "a2",
        "probabilistic_hysteresis",
        "temporal_recovery",
        "active_uncertain",
        "object_submap_epoch",
        "v1_cumulative",
        "translation",
    )
    A3 = (
        "a3",
        "probabilistic_hysteresis",
        "temporal_recovery",
        "active_uncertain",
        "object_submap_epoch",
        "reversible_ledger",
        "translation",
    )
    A4 = (
        "a4",
        "probabilistic_hysteresis",
        "temporal_recovery",
        "dormant_reid",
        "object_submap_epoch",
        "reversible_ledger",
        "gated_icp",
    )

    @property
    def profile_id(self) -> str:
        return self.value[0]

    @property
    def lifecycle_mode(self) -> str:
        return self.value[1]

    @property
    def proposal_mode(self) -> str:
        return self.value[2]

    @property
    def identity_mode(self) -> str:
        return self.value[3]

    @property
    def geometry_component(self) -> str:
        return self.value[4]

    @property
    def background_component(self) -> str:
        return self.value[5]

    @property
    def motion_mode(self) -> str:
        return self.value[6]

    @property
    def components(self) -> dict[str, str]:
        return {
            "lifecycle": self.lifecycle_mode,
            "proposal": self.proposal_mode,
            "identity": self.identity_mode,
            "geometry": self.geometry_component,
            "background": self.background_component,
            "motion": self.motion_mode,
        }

    @property
    def geometry_mode(self) -> str:
        if self.geometry_component == "object_submap_epoch":
            return "object_submap"
        return self.geometry_component

    @property
    def background_mode(self) -> str:
        if self.background_component == "reversible_ledger":
            return "masked_temporal"
        return self.background_component

    @property
    def association_mode(self) -> str:
        return self.identity_mode

    @classmethod
    def from_id(cls, profile_id: str) -> ExecutionProfile:
        if not isinstance(profile_id, str):
            raise TypeError("execution_profile must be a string")
        for profile in cls:
            if profile.profile_id == profile_id:
                return profile
        raise ValueError(f"execution_profile has unknown value: {profile_id!r}")


@dataclass(frozen=True)
class TemporalReadoutConfig:
    lifecycle: TemporalLifecycleConfig
    association: TemporalAssociationConfig
    geometry: TemporalGeometryConfig
    execution_profile: ExecutionProfile = ExecutionProfile.A4
    proposal: TemporalProposalConfig | None = None
    identity: TemporalIdentityConfig | None = None
    dynamic_state: TemporalDynamicConfig | None = None
    motion: TemporalMotionConfig | None = None
    geometry_epoch: TemporalGeometryEpochConfig | None = None
    background_ledger: TemporalBackgroundLedgerConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution_profile, ExecutionProfile):
            raise TypeError("execution_profile must be an ExecutionProfile")
        if self.proposal is None:
            object.__setattr__(
                self,
                "proposal",
                TemporalProposalConfig(
                    32, min(16, self.geometry.maximum_entities), 0.25, 0.1
                ),
            )
        if self.identity is None:
            object.__setattr__(
                self,
                "identity",
                TemporalIdentityConfig(
                    max(1000, self.geometry.maximum_entities), 600, 0.8, 3.0
                ),
            )
        if self.dynamic_state is None:
            object.__setattr__(
                self, "dynamic_state", TemporalDynamicConfig(2, 0.1, 0.7, 10)
            )
        if self.motion is None:
            object.__setattr__(
                self, "motion", TemporalMotionConfig(0.6, 0.25, True)
            )
        if self.geometry_epoch is None:
            assert self.identity is not None
            object.__setattr__(
                self,
                "geometry_epoch",
                TemporalGeometryEpochConfig(
                    4, min(1000, self.identity.maximum_identities * 4)
                ),
            )
        if self.background_ledger is None:
            object.__setattr__(
                self,
                "background_ledger",
                TemporalBackgroundLedgerConfig(
                    min(50000, self.geometry.background_block_count), 2, 2, 1, 8
                ),
            )


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return value


def _require_exact_keys(
    config: Mapping[str, object], expected: set[str], path: str
) -> None:
    actual = set(config)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ValueError(f"{path} has missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown keys: {sorted(unknown)}")


def _finite_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value


def _positive_float(value: object, path: str) -> float:
    result = _finite_float(value, path)
    if result <= 0.0:
        raise ValueError(f"{path} must be positive")
    return result


def _nonnegative_float(value: object, path: str) -> float:
    result = _finite_float(value, path)
    if result < 0.0:
        raise ValueError(f"{path} must be nonnegative")
    return result


def _probability(value: object, path: str) -> float:
    result = _finite_float(value, path)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must be in [0, 1]")
    return result


def _positive_int(value: object, path: str) -> int:
    result = _integer(value, path)
    if result <= 0:
        raise ValueError(f"{path} must be positive")
    return result


def _nonnegative_int(value: object, path: str) -> int:
    result = _integer(value, path)
    if result < 0:
        raise ValueError(f"{path} must be nonnegative")
    return result


def _allowed_int(value: object, allowed: set[int], path: str) -> int:
    result = _integer(value, path)
    if result not in allowed:
        raise ValueError(f"{path} must be one of {sorted(allowed)}")
    return result


def _allowed_float(value: object, allowed: set[float], path: str) -> float:
    result = _finite_float(value, path)
    if result not in allowed:
        raise ValueError(f"{path} must be one of {sorted(allowed)}")
    return result


def _parse_lifecycle(config: object) -> TemporalLifecycleConfig:
    path = "temporal_readout.lifecycle"
    raw = _require_mapping(config, path)
    expected = {field.name for field in fields(TemporalLifecycleConfig)}
    _require_exact_keys(raw, expected, path)

    log_odds_limit = _positive_float(raw["log_odds_limit"], f"{path}.log_odds_limit")
    initial_log_odds = _finite_float(
        raw["initial_log_odds"], f"{path}.initial_log_odds"
    )
    present_log_likelihood = _finite_float(
        raw["present_log_likelihood"], f"{path}.present_log_likelihood"
    )
    absent_log_likelihood = _finite_float(
        raw["absent_log_likelihood"], f"{path}.absent_log_likelihood"
    )
    if abs(initial_log_odds) > log_odds_limit:
        raise ValueError(f"{path}.initial_log_odds must be within log_odds_limit")
    if not 0.0 < present_log_likelihood <= log_odds_limit:
        raise ValueError(
            f"{path}.present_log_likelihood must be positive and within log_odds_limit"
        )
    if not -log_odds_limit <= absent_log_likelihood < 0.0:
        raise ValueError(
            f"{path}.absent_log_likelihood must be negative and within log_odds_limit"
        )

    result = TemporalLifecycleConfig(
        initial_log_odds=initial_log_odds,
        present_log_likelihood=present_log_likelihood,
        absent_log_likelihood=absent_log_likelihood,
        log_odds_limit=log_odds_limit,
        decay_half_life_seconds=_positive_float(
            raw["decay_half_life_seconds"], f"{path}.decay_half_life_seconds"
        ),
        active_on_probability=_probability(
            raw["active_on_probability"], f"{path}.active_on_probability"
        ),
        dormant_off_probability=_probability(
            raw["dormant_off_probability"], f"{path}.dormant_off_probability"
        ),
        minimum_absent_streak=_positive_int(
            raw["minimum_absent_streak"], f"{path}.minimum_absent_streak"
        ),
        minimum_distinct_view_bins=_positive_int(
            raw["minimum_distinct_view_bins"], f"{path}.minimum_distinct_view_bins"
        ),
        visibility_depth_tolerance_m=_positive_float(
            raw["visibility_depth_tolerance_m"],
            f"{path}.visibility_depth_tolerance_m",
        ),
        minimum_visible_pixel_count=_nonnegative_int(
            raw["minimum_visible_pixel_count"],
            f"{path}.minimum_visible_pixel_count",
        ),
        minimum_visible_fraction=_probability(
            raw["minimum_visible_fraction"], f"{path}.minimum_visible_fraction"
        ),
        view_bin_azimuth_count=_positive_int(
            raw["view_bin_azimuth_count"], f"{path}.view_bin_azimuth_count"
        ),
        view_bin_elevation_count=_positive_int(
            raw["view_bin_elevation_count"], f"{path}.view_bin_elevation_count"
        ),
    )
    if result.dormant_off_probability >= result.active_on_probability:
        raise ValueError(
            f"{path}.dormant_off_probability must be below active_on_probability"
        )
    available_bins = result.view_bin_azimuth_count * result.view_bin_elevation_count
    if result.minimum_distinct_view_bins > available_bins:
        raise ValueError(
            f"{path}.minimum_distinct_view_bins exceeds available view bins"
        )
    return result


def _parse_association(config: object) -> TemporalAssociationConfig:
    path = "temporal_readout.association"
    raw = _require_mapping(config, path)
    expected = {field.name for field in fields(TemporalAssociationConfig)}
    _require_exact_keys(raw, expected, path)
    result = TemporalAssociationConfig(
        visual_weight=_nonnegative_float(raw["visual_weight"], f"{path}.visual_weight"),
        semantic_weight=_nonnegative_float(
            raw["semantic_weight"], f"{path}.semantic_weight"
        ),
        size_weight=_nonnegative_float(raw["size_weight"], f"{path}.size_weight"),
        motion_weight=_nonnegative_float(raw["motion_weight"], f"{path}.motion_weight"),
        geometry_weight=_nonnegative_float(
            raw["geometry_weight"], f"{path}.geometry_weight"
        ),
        minimum_score=_probability(raw["minimum_score"], f"{path}.minimum_score"),
        maximum_centroid_distance_m=_positive_float(
            raw["maximum_centroid_distance_m"],
            f"{path}.maximum_centroid_distance_m",
        ),
        semantic_conflict_probability=_probability(
            raw["semantic_conflict_probability"],
            f"{path}.semantic_conflict_probability",
        ),
        conflict_override_visual=_probability(
            raw["conflict_override_visual"], f"{path}.conflict_override_visual"
        ),
        conflict_override_geometry=_probability(
            raw["conflict_override_geometry"],
            f"{path}.conflict_override_geometry",
        ),
    )
    if not any(
        weight > 0.0
        for weight in (
            result.visual_weight,
            result.semantic_weight,
            result.size_weight,
            result.motion_weight,
            result.geometry_weight,
        )
    ):
        raise ValueError(f"{path} must have at least one positive weight")
    return result


def _parse_geometry(config: object) -> TemporalGeometryConfig:
    path = "temporal_readout.geometry"
    raw = _require_mapping(config, path)
    expected = {field.name for field in fields(TemporalGeometryConfig)}
    _require_exact_keys(raw, expected, path)
    result = TemporalGeometryConfig(
        voxel_size_m=_positive_float(raw["voxel_size_m"], f"{path}.voxel_size_m"),
        depth_max_m=_positive_float(raw["depth_max_m"], f"{path}.depth_max_m"),
        maximum_entities=_positive_int(
            raw["maximum_entities"], f"{path}.maximum_entities"
        ),
        maximum_object_voxels=_positive_int(
            raw["maximum_object_voxels"], f"{path}.maximum_object_voxels"
        ),
        maximum_visibility_points_per_entity=_positive_int(
            raw["maximum_visibility_points_per_entity"],
            f"{path}.maximum_visibility_points_per_entity",
        ),
        background_block_count=_positive_int(
            raw["background_block_count"], f"{path}.background_block_count"
        ),
        background_mask_dilation_px=_nonnegative_int(
            raw["background_mask_dilation_px"],
            f"{path}.background_mask_dilation_px",
        ),
        minimum_icp_points=_positive_int(
            raw["minimum_icp_points"], f"{path}.minimum_icp_points"
        ),
        minimum_icp_fitness=_probability(
            raw["minimum_icp_fitness"], f"{path}.minimum_icp_fitness"
        ),
        maximum_icp_rmse_m=_positive_float(
            raw["maximum_icp_rmse_m"], f"{path}.maximum_icp_rmse_m"
        ),
        maximum_motion_m=_positive_float(
            raw["maximum_motion_m"], f"{path}.maximum_motion_m"
        ),
    )
    if result.voxel_size_m > result.depth_max_m:
        raise ValueError(f"{path}.voxel_size_m cannot exceed depth_max_m")
    for field_name in (
        "maximum_entities",
        "maximum_object_voxels",
        "maximum_visibility_points_per_entity",
    ):
        if getattr(result, field_name) > result.background_block_count:
            raise ValueError(
                f"{path}.{field_name} cannot exceed background_block_count"
            )
    return result


def _parse_proposal(config: object) -> TemporalProposalConfig:
    path = "temporal_readout.proposal"
    raw = _require_mapping(config, path)
    _require_exact_keys(
        raw, {field.name for field in fields(TemporalProposalConfig)}, path
    )
    return TemporalProposalConfig(
        minimum_residual_area_px=_positive_int(
            raw["minimum_residual_area_px"], f"{path}.minimum_residual_area_px"
        ),
        maximum_recovered_proposals=_positive_int(
            raw["maximum_recovered_proposals"],
            f"{path}.maximum_recovered_proposals",
        ),
        search_region_expansion_m=_positive_float(
            raw["search_region_expansion_m"], f"{path}.search_region_expansion_m"
        ),
        minimum_depth_residual_m=_positive_float(
            raw["minimum_depth_residual_m"], f"{path}.minimum_depth_residual_m"
        ),
    )


def _parse_identity(config: object) -> TemporalIdentityConfig:
    path = "temporal_readout.identity"
    raw = _require_mapping(config, path)
    _require_exact_keys(
        raw, {field.name for field in fields(TemporalIdentityConfig)}, path
    )
    return TemporalIdentityConfig(
        maximum_identities=_positive_int(
            raw["maximum_identities"], f"{path}.maximum_identities"
        ),
        maximum_dormant_frames=_positive_int(
            raw["maximum_dormant_frames"], f"{path}.maximum_dormant_frames"
        ),
        minimum_reid_similarity=_probability(
            raw["minimum_reid_similarity"], f"{path}.minimum_reid_similarity"
        ),
        maximum_reid_distance_m=_positive_float(
            raw["maximum_reid_distance_m"], f"{path}.maximum_reid_distance_m"
        ),
    )


def _parse_dynamic(config: object) -> TemporalDynamicConfig:
    path = "temporal_readout.dynamic_state"
    raw = _require_mapping(config, path)
    _require_exact_keys(
        raw, {field.name for field in fields(TemporalDynamicConfig)}, path
    )
    return TemporalDynamicConfig(
        minimum_consecutive_motion_frames=_allowed_int(
            raw["minimum_consecutive_motion_frames"],
            {2, 3, 4},
            f"{path}.minimum_consecutive_motion_frames",
        ),
        displacement_floor_m=_allowed_float(
            raw["displacement_floor_m"],
            {0.05, 0.10, 0.15},
            f"{path}.displacement_floor_m",
        ),
        minimum_motion_confidence=_allowed_float(
            raw["minimum_motion_confidence"],
            {0.6, 0.7, 0.8},
            f"{path}.minimum_motion_confidence",
        ),
        static_off_streak_frames=_allowed_int(
            raw["static_off_streak_frames"],
            {5, 10, 20},
            f"{path}.static_off_streak_frames",
        ),
    )


def _parse_motion(config: object) -> TemporalMotionConfig:
    path = "temporal_readout.motion"
    raw = _require_mapping(config, path)
    _require_exact_keys(
        raw, {field.name for field in fields(TemporalMotionConfig)}, path
    )
    result = TemporalMotionConfig(
        minimum_translation_confidence=_probability(
            raw["minimum_translation_confidence"],
            f"{path}.minimum_translation_confidence",
        ),
        maximum_translation_residual_m=_positive_float(
            raw["maximum_translation_residual_m"],
            f"{path}.maximum_translation_residual_m",
        ),
        require_explicit_rejection=_boolean(
            raw["require_explicit_rejection"], f"{path}.require_explicit_rejection"
        ),
    )
    if not result.require_explicit_rejection:
        raise ValueError(f"{path}.require_explicit_rejection must be true")
    return result


def _parse_geometry_epoch(config: object) -> TemporalGeometryEpochConfig:
    path = "temporal_readout.geometry_epoch"
    raw = _require_mapping(config, path)
    _require_exact_keys(
        raw, {field.name for field in fields(TemporalGeometryEpochConfig)}, path
    )
    return TemporalGeometryEpochConfig(
        maximum_epochs_per_identity=_positive_int(
            raw["maximum_epochs_per_identity"],
            f"{path}.maximum_epochs_per_identity",
        ),
        maximum_retained_epochs=_positive_int(
            raw["maximum_retained_epochs"], f"{path}.maximum_retained_epochs"
        ),
    )


def _parse_background_ledger(config: object) -> TemporalBackgroundLedgerConfig:
    path = "temporal_readout.background_ledger"
    raw = _require_mapping(config, path)
    _require_exact_keys(
        raw, {field.name for field in fields(TemporalBackgroundLedgerConfig)}, path
    )
    result = TemporalBackgroundLedgerConfig(
        maximum_journal_blocks=_positive_int(
            raw["maximum_journal_blocks"], f"{path}.maximum_journal_blocks"
        ),
        commit_support_frames=_allowed_int(
            raw["commit_support_frames"], {2, 3, 4}, f"{path}.commit_support_frames"
        ),
        commit_distinct_view_bins=_allowed_int(
            raw["commit_distinct_view_bins"],
            {2, 3},
            f"{path}.commit_distinct_view_bins",
        ),
        minimum_commit_frame_gap=_positive_int(
            raw["minimum_commit_frame_gap"], f"{path}.minimum_commit_frame_gap"
        ),
        maximum_records_per_block=_positive_int(
            raw["maximum_records_per_block"], f"{path}.maximum_records_per_block"
        ),
    )
    if result.maximum_records_per_block < result.commit_support_frames:
        raise ValueError(
            f"{path}.maximum_records_per_block cannot be below commit_support_frames"
        )
    return result


def _parse_components(config: object, profile: ExecutionProfile) -> None:
    path = "temporal_readout.components"
    raw = _require_mapping(config, path)
    _require_exact_keys(raw, set(profile.components), path)
    if any(not isinstance(value, str) for value in raw.values()):
        raise TypeError(f"{path} values must be strings")
    if dict(raw) != profile.components:
        raise ValueError(
            "temporal_readout profile components do not match execution_profile"
        )


def temporal_config_from_json(config: Mapping[str, object]) -> TemporalReadoutConfig:
    raw = _require_mapping(config, "config")
    _require_exact_keys(raw, {"temporal_readout"}, "config")
    temporal = _require_mapping(raw["temporal_readout"], "temporal_readout")
    _require_exact_keys(
        temporal,
        {
            "execution_profile",
            "components",
            "lifecycle",
            "association",
            "geometry",
            "proposal",
            "identity",
            "dynamic_state",
            "motion",
            "geometry_epoch",
            "background_ledger",
        },
        "temporal_readout",
    )
    profile = ExecutionProfile.from_id(temporal["execution_profile"])
    _parse_components(temporal["components"], profile)
    result = TemporalReadoutConfig(
        lifecycle=_parse_lifecycle(temporal["lifecycle"]),
        association=_parse_association(temporal["association"]),
        geometry=_parse_geometry(temporal["geometry"]),
        proposal=_parse_proposal(temporal["proposal"]),
        identity=_parse_identity(temporal["identity"]),
        dynamic_state=_parse_dynamic(temporal["dynamic_state"]),
        motion=_parse_motion(temporal["motion"]),
        geometry_epoch=_parse_geometry_epoch(temporal["geometry_epoch"]),
        background_ledger=_parse_background_ledger(temporal["background_ledger"]),
        execution_profile=profile,
    )
    if (
        result.lifecycle.visibility_depth_tolerance_m
        < result.geometry.voxel_size_m
    ):
        raise ValueError(
            "temporal_readout.lifecycle.visibility_depth_tolerance_m "
            "must be at least geometry.voxel_size_m"
        )
    if result.proposal.maximum_recovered_proposals > result.geometry.maximum_entities:
        raise ValueError(
            "temporal_readout.proposal.maximum_recovered_proposals cannot exceed "
            "geometry.maximum_entities"
        )
    if result.identity.maximum_identities < result.geometry.maximum_entities:
        raise ValueError(
            "temporal_readout.identity.maximum_identities cannot be below "
            "geometry.maximum_entities"
        )
    if (
        result.geometry_epoch.maximum_retained_epochs
        < result.geometry_epoch.maximum_epochs_per_identity
    ):
        raise ValueError(
            "temporal_readout.geometry_epoch.maximum_retained_epochs cannot be below "
            "maximum_epochs_per_identity"
        )
    if (
        result.geometry_epoch.maximum_retained_epochs
        > result.identity.maximum_identities
        * result.geometry_epoch.maximum_epochs_per_identity
    ):
        raise ValueError(
            "temporal_readout.geometry_epoch.maximum_retained_epochs exceeds identity "
            "epoch capacity"
        )
    if (
        result.background_ledger.maximum_journal_blocks
        > result.geometry.background_block_count
    ):
        raise ValueError(
            "temporal_readout.background_ledger.maximum_journal_blocks cannot exceed "
            "geometry.background_block_count"
        )
    available_bins = (
        result.lifecycle.view_bin_azimuth_count
        * result.lifecycle.view_bin_elevation_count
    )
    if result.background_ledger.commit_distinct_view_bins > available_bins:
        raise ValueError(
            "temporal_readout.background_ledger.commit_distinct_view_bins exceeds "
            "available view bins"
        )
    return result


def _dataclass_mapping(config: object) -> dict[str, object]:
    return {field.name: getattr(config, field.name) for field in fields(config)}


def temporal_config_to_json(config: TemporalReadoutConfig) -> dict[str, object]:
    if not isinstance(config, TemporalReadoutConfig):
        raise TypeError("config must be a TemporalReadoutConfig")
    return {
        "temporal_readout": {
            "execution_profile": config.execution_profile.profile_id,
            "components": config.execution_profile.components,
            "lifecycle": _dataclass_mapping(config.lifecycle),
            "association": _dataclass_mapping(config.association),
            "geometry": _dataclass_mapping(config.geometry),
            "proposal": _dataclass_mapping(config.proposal),
            "identity": _dataclass_mapping(config.identity),
            "dynamic_state": _dataclass_mapping(config.dynamic_state),
            "motion": _dataclass_mapping(config.motion),
            "geometry_epoch": _dataclass_mapping(config.geometry_epoch),
            "background_ledger": _dataclass_mapping(config.background_ledger),
        }
    }
