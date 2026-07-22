from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.oviv2.association import AssociationConfig
from src.oviv2.dense_projection import DenseSemanticConfig
from src.oviv2.entities import EntityRegistryConfig
from src.oviv2.evidence import EvidenceConfig
from src.oviv2.geometry import TsdfConfig
from src.oviv2.runtime import Oviv2RuntimeConfig
from src.oviv2.semantic_fusion import SemanticFusionConfig
from src.oviv2.structure import DepthStructureConfig
from src.oviv2.tracking import LocalTrackerConfig


def runtime_config_from_json(config: dict[str, Any]) -> Oviv2RuntimeConfig:
    if "semantic_mode" in config and config["semantic_mode"] != "owner_authoritative":
        raise ValueError("semantic_mode must be owner_authoritative")
    if "feature_mode" in config and config["feature_mode"] != "cached_image":
        raise ValueError("feature_mode must be cached_image")
    voxel_size = float(config.get("voxel_size_m", 0.05))
    block_resolution = int(config.get("block_resolution", 8))
    source_stride = int(config.get("source_stride", 10))
    dense_mode = config.get("dense_semantic_mode", "disabled")
    if dense_mode == "disabled":
        dense_semantics = None
    elif dense_mode == "cached_probabilities":
        dense_semantics = DenseSemanticConfig(
            voxel_size_m=voxel_size,
            integration_radius_m=float(config.get("dense_integration_radius_m", 6.0)),
            minimum_probability=float(config.get("dense_minimum_probability", 0.01)),
            minimum_quality=float(config.get("dense_minimum_quality", 0.01)),
            entropy_power=float(config.get("dense_entropy_power", 1.0)),
            view_angle_power=float(config.get("dense_view_angle_power", 1.0)),
        )
    else:
        raise ValueError(
            "dense_semantic_mode must be disabled or cached_probabilities"
        )
    association_keys = {
        "min_directed_overlap": "association_min_directed_overlap",
        "bounds_expansion_m": "association_bounds_expansion_m",
        "max_centroid_distance_m": "association_max_centroid_distance_m",
        "minimum_score": "association_minimum_score",
        "geometry_weight": "association_geometry_weight",
        "overlap_weight": "association_overlap_weight",
        "visual_weight": "association_visual_weight",
        "semantic_weight": "association_semantic_weight",
        "temporal_weight": "association_temporal_weight",
        "semantic_conflict_confidence": "semantic_conflict_confidence",
        "semantic_conflict_visual_override": "semantic_conflict_visual_override",
    }
    association = None
    if any(name in config for name in association_keys.values()):
        defaults = asdict(AssociationConfig())
        association = AssociationConfig(
            **{
                field: config.get(config_name, defaults[field])
                for field, config_name in association_keys.items()
            }
        )
    return Oviv2RuntimeConfig(
        tsdf=TsdfConfig(
            voxel_size_m=voxel_size,
            block_resolution=block_resolution,
            block_count=int(config.get("block_count", 100_000)),
            depth_max_m=float(config.get("depth_max_m", 10.0)),
            trunc_voxel_multiplier=float(config.get("trunc_voxel_multiplier", 4.0)),
        ),
        evidence=EvidenceConfig(
            block_resolution=block_resolution,
            semantic_top_k=int(config.get("semantic_top_k", 4)),
            entity_top_k=int(config.get("entity_top_k", 4)),
        ),
        tracker=LocalTrackerConfig(
            window_size=int(config.get("track_window_size", 5)),
            confirm_hits=int(config.get("confirm_hits", 2)),
            max_age_frames=int(config.get("max_age_frames", 3)) * source_stride,
            min_voxel_overlap=float(config.get("track_min_voxel_overlap", 0.1)),
            max_centroid_distance_m=float(config.get("track_max_centroid_distance_m", 0.5)),
            association=association,
            ambiguous_edge_score=float(config.get("ambiguous_edge_score", 0.70)),
            third_view_min_score=float(config.get("third_view_min_score", 0.75)),
        ),
        registry=EntityRegistryConfig(
            min_voxel_overlap=float(config.get("entity_min_voxel_overlap", 0.1)),
            max_centroid_distance_m=float(config.get("entity_max_centroid_distance_m", 0.6)),
            association=association,
            prototype_top_k=int(config.get("prototype_top_k", 3)),
            prototype_merge_cosine=float(config.get("prototype_merge_cosine", 0.90)),
            view_top_k=int(config.get("view_top_k", 10)),
            view_minimum_novelty_cosine=float(
                config.get("view_minimum_novelty_cosine", 0.10)
            ),
        ),
        visibility_depth_tolerance_m=float(
            config.get("visibility_depth_tolerance_m", 0.1)
        ),
        absence_negative_support=float(config.get("absence_negative_support", 1.0)),
        ownership_min_net_support=float(
            config.get("ownership_min_net_support", 1e-6)
        ),
        dense_semantics=dense_semantics,
    )


def semantic_fusion_config_from_json(
    config: dict[str, Any],
) -> SemanticFusionConfig | None:
    mode = config.get("fusion_semantic_mode", "disabled")
    if mode == "disabled":
        if "fusion_entity_weight_scale" in config:
            raise ValueError(
                "fusion_entity_weight_scale requires fusion_semantic_mode"
            )
        return None
    if mode != "uncertainty_linear":
        raise ValueError(
            "fusion_semantic_mode must be disabled or uncertainty_linear"
        )
    if config.get("dense_semantic_mode", "disabled") != "cached_probabilities":
        raise ValueError("uncertainty_linear fusion requires cached dense semantics")
    return SemanticFusionConfig(
        entity_weight_scale=config.get("fusion_entity_weight_scale", 0.5)
    )


def structure_config_from_json(
    config: dict[str, Any],
    *,
    voxel_size_m: float,
) -> DepthStructureConfig:
    return DepthStructureConfig(
        enabled=bool(config.get("structure_enabled", True)),
        voxel_size_m=voxel_size_m,
        pixel_stride=int(config.get("structure_pixel_stride", config.get("pixel_stride", 4))),
        min_valid_points=int(
            config.get("structure_min_valid_points", config.get("min_valid_points", 10))
        ),
        horizontal_threshold=float(config.get("structure_horizontal_threshold", 0.6)),
        wall_vertical_threshold=float(config.get("structure_wall_vertical_threshold", 0.5)),
        min_component_pixels=int(config.get("structure_min_component_pixels", 500)),
        min_component_fraction=float(config.get("structure_min_component_fraction", 0.01)),
        max_components_per_class=int(config.get("structure_max_components_per_class", 5)),
        object_exclusion_dilation=int(config.get("structure_object_exclusion_dilation", 3)),
        wall_confidence=float(config.get("structure_wall_confidence", 0.75)),
        floor_confidence=float(config.get("structure_floor_confidence", 0.85)),
        ceiling_confidence=float(config.get("structure_ceiling_confidence", 0.80)),
    )
