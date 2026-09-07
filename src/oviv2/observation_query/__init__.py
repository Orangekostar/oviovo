"""Observation-query inputs for OVI-backed ReScene inference."""

from src.oviv2.observation_query.contracts import (
    OBSERVATION_METADATA_COLUMNS,
    ObservationBank,
    ObservationBankArtifactPaths,
    ObservationBankError,
    load_observation_bank,
    save_observation_bank,
)
from src.oviv2.observation_query.observations import (
    DepthRelation,
    RawRegionObservation,
    RegionSupportCandidate,
    build_region_metadata,
    build_region_model_incidence,
    camera_to_reference,
    classify_depth_relations,
    extract_raw_regions,
    neighborhood_depth_reliability,
    prepare_observation_bank,
    register_labels_nearest,
    select_observation_frames,
    stable_region_key,
)

__all__ = [
    "OBSERVATION_METADATA_COLUMNS",
    "DepthRelation",
    "ObservationBank",
    "ObservationBankArtifactPaths",
    "ObservationBankError",
    "RawRegionObservation",
    "RegionSupportCandidate",
    "build_region_metadata",
    "build_region_model_incidence",
    "camera_to_reference",
    "classify_depth_relations",
    "extract_raw_regions",
    "load_observation_bank",
    "neighborhood_depth_reliability",
    "prepare_observation_bank",
    "register_labels_nearest",
    "save_observation_bank",
    "select_observation_frames",
    "stable_region_key",
]
