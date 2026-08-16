#!/usr/bin/env python3
"""Run the OVIV2 dual cumulative/current readout on one TESSE-CD scene."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import random
import secrets
import shutil
import stat
import sys
import tempfile
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (  # noqa: E402
    RUNNER_SCENE_CONFIG_FIELDS,
)
from scripts.evaluation.oviv2_tesse_cd_v2_config import (  # noqa: E402
    V2_ONLY_SCENE_CONFIG_FIELDS,
    canonical_algorithm_config,
    canonical_algorithm_hash,
)
from scripts.evaluation.oviv2_tesse_cd_v2_provenance import (  # noqa: E402
    collect_formal_environment as _collect_formal_environment,
    repository_provenance as _repository_provenance,
)
from scripts.evaluation.run_oviv2_tesse_cd import (  # noqa: E402
    RunPublicationUncertainError,
    TesseCausalCheckpoint,
    _absolute_lexical,
    _binding_path,
    _byte_record,
    _cache_prefix_sha256,
    _checkpoint_plan_from_target_manifest,
    _file_record,
    _is_sha256,
    _json_hash,
    _load_causal_checkpoints_bytes,
    _load_json_bytes,
    _production_cache_loader_factory,
    _production_dataset_factory,
    _production_provenance,
    _publish_run,
    _revalidate_frozen_bindings,
    _reject_symlink_components,
    _require_regular_file,
    _resolve_path,
    _run_execution,
    _sha256,
    _sha256_bytes,
    _tree_record,
    _validate_repository_state,
    _verify_frozen_file_binding,
    apply_frozen_visibility_policy,
)
from src.evaluation.oviv2_temporal_tesse import (  # noqa: E402
    TEMPORAL_CURRENT_FORMAT,
    load_temporal_current_checkpoint,
    publish_temporal_current_checkpoint,
)
from src.evaluation.oviv2_runtime_diagnostics import (  # noqa: E402
    RUNTIME_DIAGNOSTIC_KEYS,
    canonical_diagnostic_claim,
    validate_runtime_diagnostics,
)
from src.evaluation.contracts import MapSnapshot  # noqa: E402
from src.evaluation.exporters.oviovo import write_map_snapshot  # noqa: E402
from src.oviv2.temporal_snapshot import (  # noqa: E402
    TEMPORAL_COMPACT_FORMAT,
    TemporalCompactCheckpoint,
    TemporalCurrentSnapshot,
    TemporalSnapshotMetadata,
    build_temporal_map_snapshot,
)
from src.oviv2.temporal_export import (  # noqa: E402
    TemporalExportBatch,
    validate_temporal_export_sequence,
)
from src.oviv2.reference_readout import CumulativeReadoutView  # noqa: E402


PROTOCOL_ID = "oviv2-tessecd-v2"
TEMPORAL_EXPORT_SCHEMA_VERSION = 1
TABLE_ONLY_KEYFRAME_STRIDE = 5
SCENE_CONFIG_FIELDS = RUNNER_SCENE_CONFIG_FIELDS | V2_ONLY_SCENE_CONFIG_FIELDS

_STRING_CONFIG_FIELDS = frozenset(
    {
        "algorithm_hash",
        "dataset",
        "dataset_root",
        "dense_cache_dir",
        "dense_manifest",
        "dense_semantic_mode",
        "evaluation_checkpoint_frames_sha256",
        "export_manifest",
        "feature_mode",
        "frontend_cache_dir",
        "frontend_manifest",
        "temporal_frontend_cache_dir",
        "temporal_frontend_manifest",
        "fusion_semantic_mode",
        "input_manifest",
        "method_id",
        "missing_observation_policy",
        "occlusion_target_manifest",
        "occlusion_target_manifest_sha256",
        "protocol_id",
        "scene",
        "schedule_manifest",
        "semantic_mode",
        "stage3_lineage_commit",
        "vocabulary_json",
        "vocabulary_txt",
    }
)
_INTEGER_CONFIG_FIELDS = frozenset(
    {
        "block_count",
        "block_resolution",
        "confirm_hits",
        "dense_sample_stride",
        "dense_top_k",
        "entity_top_k",
        "frame_count",
        "max_age_frames",
        "min_valid_points",
        "pixel_stride",
        "prototype_top_k",
        "schema_version",
        "semantic_top_k",
        "source_stride",
        "structure_max_components_per_class",
        "structure_min_component_pixels",
        "structure_min_valid_points",
        "structure_object_exclusion_dilation",
        "structure_pixel_stride",
        "track_window_size",
        "temporal_dense_maximum_area_px",
        "temporal_dense_maximum_observations",
        "temporal_dense_minimum_area_px",
        "temporal_depth_maximum_area_px",
        "temporal_depth_maximum_observations",
        "temporal_depth_maximum_unknown_observations",
        "temporal_depth_minimum_area_px",
        "temporal_depth_plane_minimum_area_px",
        "temporal_depth_semantic_minimum_votes",
        "temporal_proposal_min_valid_points",
        "temporal_proposal_pixel_stride",
        "view_top_k",
    }
)
_FLOAT_CONFIG_FIELDS = frozenset(
    {
        "absence_negative_support",
        "ambiguous_edge_score",
        "association_bounds_expansion_m",
        "association_geometry_weight",
        "association_max_centroid_distance_m",
        "association_min_directed_overlap",
        "association_minimum_score",
        "association_overlap_weight",
        "association_semantic_weight",
        "association_temporal_weight",
        "association_visual_weight",
        "dense_entropy_power",
        "dense_integration_radius_m",
        "dense_minimum_probability",
        "dense_minimum_quality",
        "dense_view_angle_power",
        "depth_max_m",
        "entity_max_centroid_distance_m",
        "entity_min_voxel_overlap",
        "fusion_entity_weight_scale",
        "ownership_min_net_support",
        "prototype_merge_cosine",
        "semantic_conflict_confidence",
        "semantic_conflict_visual_override",
        "structure_ceiling_confidence",
        "structure_floor_confidence",
        "structure_horizontal_threshold",
        "structure_min_component_fraction",
        "structure_wall_confidence",
        "structure_wall_vertical_threshold",
        "third_view_min_score",
        "temporal_depth_edge_threshold_m",
        "temporal_depth_planar_rmse_threshold_m",
        "temporal_depth_semantic_minimum_fraction",
        "temporal_depth_semantic_minimum_probability",
        "temporal_merge_same_semantic_iou",
        "temporal_merge_primary_duplicate_iou",
        "temporal_merge_supplement_containment",
        "track_max_centroid_distance_m",
        "track_min_voxel_overlap",
        "trunc_voxel_multiplier",
        "view_minimum_novelty_cosine",
        "visibility_depth_tolerance_m",
        "voxel_size_m",
    }
)
_V2_CONFIG_KEYS = (
    _STRING_CONFIG_FIELDS
    | _INTEGER_CONFIG_FIELDS
    | _FLOAT_CONFIG_FIELDS
    | {"evaluation_checkpoint_frames", "structure_enabled", "temporal_readout"}
)

_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "scene",
        "temporal_readout",
        "semantic_mode",
        "feature_mode",
        "dense_semantic_mode",
        "missing_observation_policy",
        "voxel_size_m",
        "block_resolution",
        "block_count",
        "depth_max_m",
        "trunc_voxel_multiplier",
        "source_stride",
        "semantic_top_k",
        "entity_top_k",
        "track_window_size",
        "confirm_hits",
        "max_age_frames",
        "track_min_voxel_overlap",
        "track_max_centroid_distance_m",
        "entity_min_voxel_overlap",
        "entity_max_centroid_distance_m",
        "prototype_top_k",
        "prototype_merge_cosine",
        "view_top_k",
        "view_minimum_novelty_cosine",
        "visibility_depth_tolerance_m",
        "absence_negative_support",
        "ownership_min_net_support",
        "dense_integration_radius_m",
        "dense_minimum_probability",
        "dense_minimum_quality",
        "dense_entropy_power",
        "dense_view_angle_power",
        "association_min_directed_overlap",
        "association_bounds_expansion_m",
        "association_max_centroid_distance_m",
        "association_minimum_score",
        "association_geometry_weight",
        "association_overlap_weight",
        "association_visual_weight",
        "association_semantic_weight",
        "association_temporal_weight",
        "semantic_conflict_confidence",
        "semantic_conflict_visual_override",
        "ambiguous_edge_score",
        "third_view_min_score",
    }
)
_DATASET_CONFIG_KEYS = frozenset(
    {"dataset_root", "scene", "export_manifest", "schedule_manifest"}
)
_CACHE_CONFIG_KEYS = frozenset(
    {
        "scene",
        "frame_count",
        "input_manifest",
        "vocabulary_json",
        "vocabulary_txt",
        "frontend_cache_dir",
        "frontend_manifest",
        "temporal_frontend_cache_dir",
        "temporal_frontend_manifest",
        "temporal_readout",
        "dense_cache_dir",
        "dense_manifest",
        "stage3_lineage_commit",
        "dense_sample_stride",
        "dense_top_k",
        "dense_semantic_mode",
        "depth_max_m",
        "voxel_size_m",
        "pixel_stride",
        "min_valid_points",
        "structure_enabled",
        "structure_pixel_stride",
        "structure_min_valid_points",
        "structure_horizontal_threshold",
        "structure_wall_vertical_threshold",
        "structure_min_component_pixels",
        "structure_min_component_fraction",
        "structure_max_components_per_class",
        "structure_object_exclusion_dilation",
        "structure_wall_confidence",
        "structure_floor_confidence",
        "structure_ceiling_confidence",
        "fusion_semantic_mode",
        "fusion_entity_weight_scale",
        "temporal_dense_minimum_area_px",
        "temporal_dense_maximum_area_px",
        "temporal_dense_maximum_observations",
        "temporal_depth_edge_threshold_m",
        "temporal_depth_minimum_area_px",
        "temporal_depth_maximum_area_px",
        "temporal_depth_plane_minimum_area_px",
        "temporal_depth_planar_rmse_threshold_m",
        "temporal_depth_semantic_minimum_votes",
        "temporal_depth_semantic_minimum_fraction",
        "temporal_depth_semantic_minimum_probability",
        "temporal_depth_maximum_observations",
        "temporal_depth_maximum_unknown_observations",
        "temporal_proposal_pixel_stride",
        "temporal_proposal_min_valid_points",
        "temporal_merge_same_semantic_iou",
        "temporal_merge_primary_duplicate_iou",
        "temporal_merge_supplement_containment",
    }
)
_ENVIRONMENT_LIBRARY_KEYS = frozenset(
    {"numpy", "open3d", "scipy", "torch", "pillow"}
)
_CHECKPOINT_MEMBERS = {
    TEMPORAL_CURRENT_FORMAT: frozenset(
        {
            "manifest.json",
            "snapshot.npz",
            "entities.jsonl",
            "diagnostics.json",
            "checksums.json",
        }
    ),
    TEMPORAL_COMPACT_FORMAT: frozenset(
        {"manifest.json", "arrays.npz", "checksums.json"}
    ),
}

V2_FREEZE_TOP_KEYS = frozenset(
    {
        "schema_version",
        "freeze_id",
        "status",
        "method",
        "dataset",
        "repository",
        "algorithm",
        "scenes",
        "shared_bindings",
        "output_roots",
        "selection",
        "environment",
        "commands",
        "models",
        "release_bindings",
        "office_pre_freeze_audit",
        "evidence",
        "seed_policy",
        "frozen_hashes",
        "office_authorizations",
    }
)
V2_FREEZE_REPOSITORY_KEYS = frozenset(
    {
        "clean",
        "commit",
        "parents",
        "tree",
        "commit_time_utc",
        "stage3_lineage_commit",
        "stage3_is_ancestor",
    }
)
V2_FREEZE_SCENE_KEYS = frozenset(
    {
        "frozen_config",
        "export_manifest",
        "frontend_manifest",
        "temporal_frontend_manifest",
        "dense_manifest",
        "vocabulary_json",
        "vocabulary_txt",
    }
)
V2_FREEZE_SHARED_KEYS = frozenset(
    {"input_manifest", "schedule", "occlusion_target_manifest"}
)
V2_FREEZE_SELECTION_KEYS = frozenset(
    {
        "artifact",
        "development_scene",
        "selected_config_sha256",
        "selected_algorithm_hash",
    }
)
V2_FREEZE_ENVIRONMENT_KEYS = frozenset(
    {
        "python",
        "python_implementation",
        "platform",
        "machine",
        "host",
        "cuda",
        "cuda_visible_devices",
        "gpu",
        "libraries",
    }
)
V2_FREEZE_COMMAND_KEYS = frozenset({"cwd", "python", "mapping"})
V2_FREEZE_MODEL_KEYS = frozenset(
    {"manifest_sha256", "model_id", "model_sha256"}
)
V2_FREEZE_RELEASE_KEYS = frozenset(
    {"temporal_evaluator", "result_finalizer"}
)
V2_RELEASE_EXPECTED_PATHS = {
    "temporal_evaluator": REPO_ROOT
    / "scripts/evaluation/evaluate_oviv2_tesse_temporal_occlusion.py",
    "result_finalizer": REPO_ROOT / "scripts/evaluation/finalize_tesse_t2.py",
}
V2_FREEZE_OFFICE_AUDIT_KEYS = frozenset(
    {"selection_scene", "metric_sources_found", "office_outputs_read"}
)
V2_T4_METRIC_KEYS = frozenset(
    {
        "total_runtime_s_per_frame",
        "query_mean_ms",
        "query_p95_ms",
        "peak_gpu_gb",
        "peak_ram_gb",
        "final_map_mb",
    }
)
V2_T4_RAW_SOURCE_KEYS = frozenset(
    {
        "config", "run_manifest", "time_log", "gpu_samples", "query_measurements",
        "final_map_inventory", "protocol", "shortlist",
    }
)
V2_T4_BOUNDS = {
    "total_runtime_s_per_frame": 6.42,
    "query_mean_ms": 11.92,
    "query_p95_ms": 12.12,
    "peak_gpu_gb": 12.76,
    "peak_ram_gb": 9.36,
    "final_map_mb": 46.77,
}
V2_SHORTLIST_KEYS = frozenset(
    {
        "schema_version", "manifest_id", "phase", "dataset", "method_id",
        "protocol_id", "status", "development_scene", "transfer_scene",
        "office_results_read", "scenes_read", "result_contract", "results_root",
        "manifest", "result_files", "profile_fallback_order", "floors",
        "shortlisted_candidate_ids", "shortlisted_candidates", "rejection_ledger",
    }
)
V2_FINAL_SELECTION_KEYS = frozenset(
    {
        "schema_version", "manifest_id", "phase", "dataset", "method_id",
        "protocol_id", "status", "development_scene", "transfer_scene",
        "office_results_read", "scenes_read", "manifest", "shortlist",
        "t4_matrix", "t4_protocol", "t4_root_sha256",
        "profile_fallback_order", "selected_candidate_id", "selected_config",
        "selected_config_record", "selected_config_sha256", "algorithm_hash",
        "t4_ledger",
    }
)
V2_RUNTIME_DIAGNOSTIC_KEYS = RUNTIME_DIAGNOSTIC_KEYS

# Backward-compatible private aliases for existing Task 9 tests/importers.
_FREEZE_TOP_KEYS = V2_FREEZE_TOP_KEYS
_FREEZE_REPOSITORY_KEYS = V2_FREEZE_REPOSITORY_KEYS
_FREEZE_SCENE_KEYS = V2_FREEZE_SCENE_KEYS
_FREEZE_SHARED_KEYS = V2_FREEZE_SHARED_KEYS


@dataclass(frozen=True)
class RunnerDependencies:
    dataset_factory: Callable[[Mapping[str, Any]], Any]
    cache_loader_factory: Callable[[Mapping[str, Any], Any], Any]
    runtime_factory: Callable[[Mapping[str, Any], Any], Any]
    provenance_factory: Callable[[], Mapping[str, Any]]
    environment_factory: Callable[[], Mapping[str, Any]]


@dataclass(frozen=True)
class FrozenRunContext:
    manifest_path: Path
    manifest_bytes: bytes
    repository_state: Mapping[str, str]
    input_bindings: Mapping[str, Any]
    frozen_run_identity: Mapping[str, Any]
    run_slot: str


@dataclass(frozen=True)
class _CheckpointArtifactWitness:
    path: Path
    checkpoint_format: str
    tree_record: Mapping[str, Any]
    maximum_entities: int
    maximum_object_voxels: int


@dataclass(frozen=True)
class _RelativeArtifactRecord:
    path: str
    sha256: str
    byte_count: int

    @classmethod
    def bind(
        cls, record: Mapping[str, Any], *, label: str
    ) -> _RelativeArtifactRecord:
        if set(record) != {"path", "sha256", "byte_count"}:
            raise ValueError(f"{label} record fields are invalid")
        path = record.get("path")
        sha256 = record.get("sha256")
        byte_count = record.get("byte_count")
        relative = Path(path) if isinstance(path, str) else Path()
        if (
            not isinstance(path, str)
            or relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or not _is_sha256(sha256)
            or type(byte_count) is not int
            or byte_count < 0
        ):
            raise ValueError(f"{label} record is invalid")
        return cls(path, str(sha256), byte_count)

    def to_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class _CumulativeAuditWitness:
    artifact: _RelativeArtifactRecord
    snapshot: _RelativeArtifactRecord
    entities: _RelativeArtifactRecord
    voxel_snapshot: _RelativeArtifactRecord | None

    @classmethod
    def bind(
        cls, record: Mapping[str, Any]
    ) -> _CumulativeAuditWitness:
        expected = {"format", "artifact", "snapshot", "entities"}
        if "voxel_snapshot" in record:
            expected.add("voxel_snapshot")
        if set(record) != expected or record.get("format") != "oviv2_cumulative_audit_v1":
            raise ValueError("cumulative audit manifest record is invalid")
        voxel = record.get("voxel_snapshot")
        if "voxel_snapshot" in record and not isinstance(voxel, Mapping):
            raise ValueError("cumulative audit voxel snapshot record is invalid")
        return cls(
            artifact=_RelativeArtifactRecord.bind(
                record["artifact"], label="cumulative audit artifact"
            ),
            snapshot=_RelativeArtifactRecord.bind(
                record["snapshot"], label="cumulative audit snapshot"
            ),
            entities=_RelativeArtifactRecord.bind(
                record["entities"], label="cumulative audit entities"
            ),
            voxel_snapshot=(
                _RelativeArtifactRecord.bind(
                    voxel, label="cumulative audit voxel snapshot"
                )
                if isinstance(voxel, Mapping)
                else None
            ),
        )


def _staging_identity(staging: Path) -> tuple[int, int]:
    status = os.stat(staging, follow_symlinks=False)
    if not staging.is_dir():
        raise ValueError("run staging root is not a directory")
    return status.st_dev, status.st_ino


def _assert_staging_identity(staging: Path, expected: tuple[int, int]) -> None:
    try:
        current = _staging_identity(staging)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("run staging root identity changed during run") from exc
    if current != expected:
        raise ValueError("run staging root identity changed during run")


def _entry_inventory(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in root.rglob("*"):
        metadata = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(metadata.st_mode):
            result[relative] = "directory"
        elif stat.S_ISREG(metadata.st_mode):
            result[relative] = "file"
        else:
            raise ValueError("run publication inventory contains a forbidden entry")
    return result


def _revalidate_relative_file_record(
    root: Path, record: Mapping[str, Any], *, label: str
) -> None:
    if set(record) != {"path", "sha256", "byte_count"}:
        raise ValueError(f"{label} record fields are invalid")
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{label} record path is invalid")
    path = root / relative
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file")
    if _file_record(path, relative_to=root) != dict(record):
        raise ValueError(f"{label} content changed")


def _revalidate_relative_tree_record(
    root: Path, record: _RelativeArtifactRecord, *, label: str
) -> None:
    path = root / record.path
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} is not a directory")
    if _tree_record(path, relative_to=root) != record.to_record():
        raise ValueError(f"{label} content changed")


def _revalidate_cumulative_audit_witness(
    root: Path,
    witness: _CumulativeAuditWitness,
    manifest_record: Mapping[str, Any],
) -> None:
    if _CumulativeAuditWitness.bind(manifest_record) != witness:
        raise ValueError("cumulative audit manifest binding changed")
    _revalidate_relative_tree_record(
        root, witness.artifact, label="cumulative audit artifact"
    )
    _revalidate_relative_file_record(
        root, witness.snapshot.to_record(), label="cumulative audit snapshot"
    )
    _revalidate_relative_file_record(
        root, witness.entities.to_record(), label="cumulative audit entities"
    )
    if witness.voxel_snapshot is not None:
        _revalidate_relative_tree_record(
            root,
            witness.voxel_snapshot,
            label="cumulative audit voxel snapshot",
        )


def _revalidate_checkpoint_artifact(
    witness: _CheckpointArtifactWitness, *, run_root: Path
) -> None:
    if witness.checkpoint_format == TEMPORAL_CURRENT_FORMAT:
        loaded = load_temporal_current_checkpoint(witness.path)
    elif witness.checkpoint_format == TEMPORAL_COMPACT_FORMAT:
        loaded = TemporalCompactCheckpoint.load(
            witness.path,
            maximum_entities=witness.maximum_entities,
            maximum_object_voxels=witness.maximum_object_voxels,
        )
    else:
        raise ValueError("checkpoint witness format is invalid")
    loaded.revalidate_source()
    del loaded
    if _tree_record(witness.path, relative_to=run_root) != witness.tree_record:
        raise ValueError("checkpoint artifact tree changed during run")


def _bind_checkpoint_artifact(
    *,
    artifact_root: Path,
    source_witness: Any,
    checkpoint_format: str,
    staging: Path,
    temporal_config: Any,
    witnesses: list[_CheckpointArtifactWitness],
    expected_inventory: set[str],
) -> dict[str, Any]:
    if hasattr(source_witness, "revalidate_source"):
        source_witness.revalidate_source()
    else:
        source_witness.revalidate()
    artifact_tree = _tree_record(artifact_root, relative_to=staging)
    witnesses.append(
        _CheckpointArtifactWitness(
            path=artifact_root,
            checkpoint_format=checkpoint_format,
            tree_record=artifact_tree,
            maximum_entities=temporal_config.geometry.maximum_entities,
            maximum_object_voxels=temporal_config.geometry.maximum_object_voxels,
        )
    )
    artifact_entries = list(artifact_root.iterdir())
    actual_members = {path.name for path in artifact_entries}
    expected_members = _CHECKPOINT_MEMBERS[checkpoint_format]
    if actual_members != expected_members or any(
        not stat.S_ISREG(os.lstat(path).st_mode) for path in artifact_entries
    ):
        raise ValueError("checkpoint artifact inventory is invalid")
    expected_inventory.update(
        (artifact_root / name).relative_to(staging).as_posix()
        for name in expected_members
    )
    return {
        "format": checkpoint_format,
        "artifact": artifact_tree,
        "checksums_sha256": _sha256(artifact_root / "checksums.json"),
    }


def algorithm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return canonical_algorithm_config(config)


def algorithm_hash(config: Mapping[str, Any]) -> str:
    return canonical_algorithm_hash(config)


def _is_git_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_exact_frozen_file_binding(
    value: object,
    *,
    base: Path,
    role: str,
    expected_path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError(f"{role} frozen input binding fields are not exact")
    return _verify_frozen_file_binding(
        value,
        base=base,
        role=role,
        expected_path=expected_path,
    )


def _nonempty_string(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonempty_string_list(value: object, name: str) -> list[str]:
    if (
        type(value) is not list
        or not value
        or any(type(item) is not str or not item.strip() for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty string list")
    return list(value)


def _validate_environment(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != V2_FREEZE_ENVIRONMENT_KEYS:
        raise ValueError("freeze environment schema is invalid")
    result = {
        key: _nonempty_string(value[key], f"environment.{key}")
        for key in (
            "python",
            "python_implementation",
            "platform",
            "machine",
            "host",
        )
    }
    result["cuda"] = _nonempty_string_list(value["cuda"], "environment.cuda")
    result["gpu"] = _nonempty_string_list(value["gpu"], "environment.gpu")
    visible = value["cuda_visible_devices"]
    if visible is not None:
        visible = _nonempty_string(visible, "environment.cuda_visible_devices")
    result["cuda_visible_devices"] = visible
    libraries = value["libraries"]
    if (
        not isinstance(libraries, Mapping)
        or set(libraries) != _ENVIRONMENT_LIBRARY_KEYS
        or any(
            type(key) is not str
            or not key.strip()
            or type(item) is not str
            or not item.strip()
            for key, item in libraries.items()
        )
    ):
        raise ValueError("environment.libraries must be a non-empty string mapping")
    result["libraries"] = dict(sorted(libraries.items()))
    return result


def _formal_environment_compatibility(value: object) -> dict[str, Any]:
    environment = _validate_environment(value)
    visible = environment["cuda_visible_devices"]
    visible_lanes = (
        {item.strip() for item in visible.split(",") if item.strip()}
        if isinstance(visible, str)
        else set()
    )
    gpu_signatures: set[str] = set()
    for item in environment["gpu"]:
        parts = [part.strip() for part in item.split(",")]
        lane_ids: set[str] = set()
        identity = item.strip()
        if len(parts) >= 4 and parts[0].isdigit() and parts[1].startswith("GPU-"):
            lane_ids = {parts[0], parts[1]}
            identity = ", ".join(parts[2:])
        elif len(parts) >= 3 and parts[0].isdigit():
            lane_ids = {parts[0]}
            identity = ", ".join(parts[1:])
        if visible_lanes and lane_ids:
            if lane_ids.isdisjoint(visible_lanes):
                continue
        gpu_signatures.add(identity)
    return {
        key: environment[key]
        for key in (
            "python",
            "python_implementation",
            "platform",
            "machine",
            "cuda",
            "libraries",
        )
    } | {"gpu": sorted(gpu_signatures)}


def _validate_selection(
    value: object,
    *,
    manifest_base: Path,
    apartment_config: Mapping[str, Any],
    algorithm_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != V2_FREEZE_SELECTION_KEYS:
        raise ValueError("freeze selection schema is invalid")
    selected_config_sha256 = _json_hash(apartment_config)
    if not (
        value.get("development_scene") == "apartment"
        and value.get("selected_config_sha256") == selected_config_sha256
        and value.get("selected_algorithm_hash") == algorithm_sha256
    ):
        raise ValueError("freeze selection identity is invalid")
    artifact = value["artifact"]
    if not isinstance(artifact, Mapping):
        raise ValueError("freeze selection artifact binding is invalid")
    artifact_path = _binding_path(
        artifact.get("path"), base=manifest_base, role="selection artifact"
    )
    _verify_exact_frozen_file_binding(
        artifact,
        base=manifest_base,
        role="selection artifact",
        expected_path=artifact_path,
    )
    payload = _load_json_bytes(artifact_path.read_bytes(), artifact_path)
    if not (
        set(payload) == V2_FINAL_SELECTION_KEYS
        and payload.get("schema_version") == 1
        and payload.get("manifest_id") == "oviv2_tesse_dual_readout_selection_v2"
        and payload.get("phase") == "final"
        and payload.get("status") == "PASS"
        and payload.get("development_scene") == "apartment"
        and payload.get("office_results_read") is False
        and payload.get("scenes_read") == ["apartment"]
        and payload.get("selected_config_sha256") == selected_config_sha256
        and payload.get("algorithm_hash") == algorithm_sha256
    ):
        raise ValueError("selection artifact identity differs from the freeze")
    return {
        "artifact": dict(artifact),
        "development_scene": "apartment",
        "selected_config_sha256": selected_config_sha256,
        "selected_algorithm_hash": algorithm_sha256,
    }


def _validate_models(
    value: object,
    *,
    scenes: Mapping[str, Mapping[str, Any]],
    manifest_base: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"frontend", "dense"}:
        raise ValueError("freeze models schema is invalid")
    result: dict[str, Any] = {}
    for branch in ("frontend", "dense"):
        per_scene = value[branch]
        if not isinstance(per_scene, Mapping) or set(per_scene) != {
            "apartment",
            "office",
        }:
            raise ValueError(f"freeze {branch} model scene schema is invalid")
        result[branch] = {}
        for scene in ("apartment", "office"):
            model = per_scene[scene]
            if not isinstance(model, Mapping) or set(model) != V2_FREEZE_MODEL_KEYS:
                raise ValueError(f"freeze {branch} {scene} model schema is invalid")
            expected_manifest = scenes[scene][f"{branch}_manifest"]["sha256"]
            manifest_path = _binding_path(
                scenes[scene][f"{branch}_manifest"]["path"],
                base=manifest_base,
                role=f"{scene} {branch} manifest",
            )
            manifest = _load_json_bytes(manifest_path.read_bytes(), manifest_path)
            if branch == "frontend":
                provenance = manifest.get("provenance_sha256")
                expected_model_id = manifest.get("feature_model_id")
                expected_model_sha256 = (
                    provenance.get("clip_model")
                    if isinstance(provenance, Mapping)
                    else None
                )
                if not (
                    type(expected_model_id) is str
                    and _is_sha256(expected_model_sha256)
                    and expected_model_id
                    == f"clip-sha256:{expected_model_sha256}"
                ):
                    raise ValueError(
                        f"{scene} frontend manifest model identity is invalid"
                    )
            else:
                provenance = manifest.get("provenance")
                expected_model_id = (
                    provenance.get("model_id")
                    if isinstance(provenance, Mapping)
                    else None
                )
                expected_model_sha256 = (
                    provenance.get("model_sha256")
                    if isinstance(provenance, Mapping)
                    else None
                )
                if not (
                    type(expected_model_id) is str
                    and bool(expected_model_id.strip())
                    and _is_sha256(expected_model_sha256)
                ):
                    raise ValueError(f"{scene} dense manifest model identity is invalid")
            if not (
                model.get("manifest_sha256") == expected_manifest
                and model.get("model_id") == expected_model_id
                and model.get("model_sha256") == expected_model_sha256
            ):
                raise ValueError(f"freeze {branch} {scene} model binding is invalid")
            result[branch][scene] = {
                "manifest_sha256": expected_manifest,
                "model_id": expected_model_id,
                "model_sha256": expected_model_sha256,
            }
    return result


def _validate_commands(
    value: object,
    *,
    manifest_path: Path,
    scenes: Mapping[str, Mapping[str, Any]],
    output_roots: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != V2_FREEZE_COMMAND_KEYS:
        raise ValueError("freeze commands schema is invalid")
    cwd = _nonempty_string(value.get("cwd"), "commands.cwd")
    if not Path(cwd).is_absolute() or _absolute_lexical(cwd) != REPO_ROOT:
        raise ValueError("freeze command cwd must equal the repository root")
    python = _nonempty_string(value.get("python"), "commands.python")
    python_path = Path(python)
    expected_python = Path(sys.executable).resolve()
    if not (
        python_path.is_absolute()
        and python == str(python_path.resolve())
        and python_path.resolve() == expected_python
    ):
        raise ValueError("freeze command python must equal the active interpreter")
    _require_regular_file(python_path, "commands.python")
    if not os.access(python_path, os.X_OK):
        raise ValueError("freeze command python must be executable")
    mapping = value.get("mapping")
    expected_slots = ["apartment_run1", "apartment_run2"] + sorted(
        slot for slot in output_roots if slot.startswith("office_seed_")
    )
    if type(mapping) is not list or len(mapping) != len(expected_slots):
        raise ValueError("freeze mapping commands do not match authorized runs")
    runner_path = Path(__file__).resolve()
    normalized: list[dict[str, Any]] = []
    for raw, slot in zip(mapping, expected_slots):
        if not isinstance(raw, Mapping) or set(raw) != {
            "scene",
            "run_slot",
            "output",
            "argv",
        }:
            raise ValueError("freeze mapping command schema is invalid")
        scene = slot.split("_", 1)[0]
        config_path = _binding_path(
            scenes[scene]["frozen_config"]["path"],
            base=manifest_path.parent,
            role=f"{scene} frozen command config",
        )
        expected_argv = [
            python,
            str(runner_path),
            "--config",
            str(config_path),
            "--output",
            output_roots[slot],
            "--freeze-manifest",
            str(manifest_path),
            "--run-slot",
            slot,
        ]
        if not (
            raw.get("scene") == scene
            and raw.get("run_slot") == slot
            and raw.get("output") == output_roots[slot]
            and raw.get("argv") == expected_argv
        ):
            raise ValueError("freeze mapping command identity is invalid")
        normalized.append(dict(raw))
    return {"cwd": cwd, "python": python, "mapping": normalized}


def _validate_release_bindings(
    value: object, *, manifest_base: Path
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != V2_FREEZE_RELEASE_KEYS:
        raise ValueError("freeze release binding schema is invalid")
    result: dict[str, Any] = {}
    for role in sorted(V2_FREEZE_RELEASE_KEYS):
        record = value[role]
        if not isinstance(record, Mapping):
            raise ValueError(f"freeze {role} binding is invalid")
        expected_path = _absolute_lexical(V2_RELEASE_EXPECTED_PATHS[role])
        _verify_exact_frozen_file_binding(
            record,
            base=manifest_base,
            role=f"release {role}",
            expected_path=expected_path,
        )
        result[role] = dict(record)
    return result


def _validate_office_audit(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != V2_FREEZE_OFFICE_AUDIT_KEYS:
        raise ValueError("freeze Office audit schema is invalid")
    if not (
        value.get("selection_scene") == "apartment"
        and value.get("metric_sources_found") == []
        and value.get("office_outputs_read") is False
    ):
        raise ValueError("freeze Office audit is not fail-closed")
    return {
        "selection_scene": "apartment",
        "metric_sources_found": [],
        "office_outputs_read": False,
    }


def _validate_seed_policy(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "behavior",
        "seeds",
        "sha256",
    }:
        raise ValueError("freeze seed policy schema is invalid")
    behavior = value.get("behavior")
    seeds = value.get("seeds")
    expected = [0] if behavior == "deterministic" else [17, 29, 43, 71, 101]
    if behavior not in {"deterministic", "stochastic"} or seeds != expected:
        raise ValueError("freeze seed policy is not pre-registered")
    policy = {"behavior": behavior, "seeds": list(seeds)}
    if value.get("sha256") != _json_hash(policy):
        raise ValueError("freeze seed policy hash is stale")
    return {**policy, "sha256": value["sha256"]}


def _validate_frozen_evidence(
    value: object, *, manifest_base: Path, selection_artifact: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping) or set(value) != {"t1", "t4"}:
        raise ValueError("freeze T1/T4 evidence schema is invalid")
    selection_path = _binding_path(
        selection_artifact.get("path"), base=manifest_base, role="selection artifact"
    )
    selection = _load_json_bytes(selection_path.read_bytes(), selection_path)
    selected_candidate = selection.get("selected_candidate_id")
    if selected_candidate not in {f"a{index}" for index in range(5)}:
        raise ValueError("freeze selected candidate identity is invalid")
    result: dict[str, Any] = {}
    for kind, source_roles in (("t1", ("source_manifest",)), ("t4", ("shortlist", "protocol"))):
        frozen = value.get(kind)
        expected_keys = {"artifact", "root_sha256", *source_roles}
        if not isinstance(frozen, Mapping) or set(frozen) != expected_keys:
            raise ValueError(f"freeze {kind.upper()} evidence binding is invalid")
        artifact_path = _binding_path(
            frozen["artifact"].get("path"), base=manifest_base, role=f"{kind} artifact"
        )
        artifact = _verify_exact_frozen_file_binding(
            frozen["artifact"], base=manifest_base, role=f"{kind} artifact",
            expected_path=artifact_path,
        )
        source_paths: dict[str, Path] = {}
        for role in source_roles:
            source_path = _binding_path(
                frozen[role].get("path"), base=manifest_base, role=f"{kind} {role}"
            )
            _verify_exact_frozen_file_binding(
                frozen[role], base=manifest_base, role=f"{kind} {role}",
                expected_path=source_path,
            )
            source_paths[role] = source_path
        payload = _load_json_bytes(artifact_path.read_bytes(), artifact_path)
        if kind == "t1":
            deterministic = payload.get("deterministic_evidence")
            exact = (
                deterministic.get("cumulative_exact")
                if isinstance(deterministic, Mapping)
                else None
            )
            profiles = exact.get("profiles") if isinstance(exact, Mapping) else None
            gates = deterministic.get("gates") if isinstance(deterministic, Mapping) else None
            roots = {
                profile.get("cumulative_root_sha256")
                for profile in profiles.values()
                if isinstance(profile, Mapping)
            } if isinstance(profiles, Mapping) else set()
            if not (
                payload.get("manifest_id") == "oviv2_dual_readout_development_gates_v1"
                and isinstance(exact, Mapping)
                and exact.get("format") == "oviv2_t1_exact_transaction_v1"
                and set(profiles or {}) == {f"a{index}" for index in range(5)}
                and len(roots) == 1
                and next(iter(roots), None) == frozen["root_sha256"]
                and isinstance(gates, Mapping)
                and set(gates) == {"t1_exact", "determinism"}
                and all(
                    isinstance(gate, Mapping) and gate.get("status") == "PASS"
                    for gate in gates.values()
                )
                and deterministic.get("source_manifest") == frozen["source_manifest"]
            ):
                raise ValueError("frozen T1 evidence no longer proves exactness")
        else:
            claimed = payload.get("root_sha256")
            unhashed = dict(payload)
            unhashed.pop("root_sha256", None)
            shortlist = _load_json_bytes(
                source_paths["shortlist"].read_bytes(), source_paths["shortlist"]
            )
            shortlist_ids = shortlist.get("shortlisted_candidate_ids")
            if not (
                set(shortlist) == V2_SHORTLIST_KEYS
                and shortlist.get("schema_version") == 1
                and shortlist.get("manifest_id") == "oviv2_tesse_dual_readout_shortlist_v1"
                and shortlist.get("phase") == "shortlist"
                and shortlist.get("dataset") == "TESSE-CD"
                and shortlist.get("method_id") == "OVIV2"
                and shortlist.get("protocol_id") == PROTOCOL_ID
                and shortlist.get("status") == "PASS"
                and shortlist.get("development_scene") == "apartment"
                and shortlist.get("transfer_scene") == "office"
                and shortlist.get("office_results_read") is False
                and shortlist.get("scenes_read") == ["apartment"]
                and shortlist.get("profile_fallback_order") == ["a4", "a3", "a2"]
                and isinstance(shortlist_ids, list)
                and bool(shortlist_ids)
                and len(shortlist_ids) == len(set(shortlist_ids))
                and shortlist_ids == [
                    candidate
                    for candidate in ("a4", "a3", "a2")
                    if candidate in shortlist_ids
                ]
            ):
                raise ValueError("frozen T4 shortlist identity is invalid")
            shortlist_candidates = shortlist.get("shortlisted_candidates")
            if not (
                isinstance(shortlist_candidates, list)
                and len(shortlist_candidates) == len(shortlist_ids)
                and all(
                    isinstance(item, Mapping)
                    and set(item) == {
                        "candidate_id", "profile", "config_sha256", "algorithm_hash",
                        "result", "selected_config", "selected_config_record",
                    }
                    and item.get("candidate_id") == candidate
                    and item.get("profile") == candidate
                    and _is_sha256(item.get("config_sha256"))
                    and _is_sha256(item.get("algorithm_hash"))
                    for candidate, item in zip(
                        shortlist_ids, shortlist_candidates, strict=True
                    )
                )
            ):
                raise ValueError("frozen T4 shortlist candidates are invalid")
            candidates = payload.get("candidates")
            if not isinstance(candidates, Mapping) or set(candidates) != set(shortlist_ids):
                raise ValueError("frozen T4 candidate inventory is invalid")
            for candidate, candidate_row in candidates.items():
                if not (
                    isinstance(candidate_row, Mapping)
                    and set(candidate_row) == {
                        "status", "config_sha256", "run_manifest_sha256", "metrics", "gates"
                    }
                    and _is_sha256(candidate_row.get("config_sha256"))
                    and _is_sha256(candidate_row.get("run_manifest_sha256"))
                    and isinstance(candidate_row.get("metrics"), Mapping)
                    and set(candidate_row["metrics"]) == V2_T4_METRIC_KEYS
                    and all(
                        type(metric_value) in {int, float} and math.isfinite(metric_value)
                        for metric_value in candidate_row["metrics"].values()
                    )
                    and isinstance(candidate_row.get("gates"), Mapping)
                    and set(candidate_row["gates"]) == V2_T4_METRIC_KEYS
                ):
                    raise ValueError(f"frozen T4 matrix row {candidate} is invalid")
                expected_gates = {
                    metric: candidate_row["metrics"][metric] <= V2_T4_BOUNDS[metric]
                    for metric in V2_T4_METRIC_KEYS
                }
                expected_status = "PASS" if all(expected_gates.values()) else "FAIL"
                if candidate_row["gates"] != expected_gates or candidate_row["status"] != expected_status:
                    raise ValueError(f"frozen T4 matrix row {candidate} has stale gates or status")
            aggregate_status = (
                "PASS" if any(item["status"] == "PASS" for item in candidates.values()) else "FAIL"
            )
            expected_ledger: list[dict[str, Any]] = []
            first_passed: str | None = None
            shortlisted_by_id = {
                item["candidate_id"]: item for item in shortlist_candidates
            }
            for candidate in shortlist_ids:
                candidate_row = candidates[candidate]
                failures = [
                    metric
                    for metric, bound in V2_T4_BOUNDS.items()
                    if candidate_row["metrics"][metric] > bound
                ]
                if first_passed is None and not failures:
                    first_passed = candidate
                shortlist_item = shortlisted_by_id[candidate]
                expected_ledger.append(
                    {
                        "candidate_id": candidate,
                        "profile": shortlist_item["profile"],
                        "passed": not failures,
                        "selected": candidate == first_passed and not failures,
                        "failed_gates": failures,
                        "config_sha256": candidate_row["config_sha256"],
                        "run_manifest_sha256": candidate_row["run_manifest_sha256"],
                    }
                )
            selected_item = shortlisted_by_id.get(first_passed)
            if not (
                first_passed == selected_candidate
                and selection.get("t4_ledger") == expected_ledger
                and selection.get("shortlist") == frozen["shortlist"]
                and selection.get("t4_matrix") == frozen["artifact"]
                and selection.get("t4_protocol") == frozen["protocol"]
                and selection.get("t4_root_sha256") == frozen["root_sha256"]
                and selected_item is not None
                and selection.get("selected_config") == selected_item["selected_config"]
                and selection.get("selected_config_record")
                == selected_item["selected_config_record"]
                and selection.get("selected_config_sha256")
                == selected_item["config_sha256"]
                and selection.get("algorithm_hash") == selected_item["algorithm_hash"]
            ):
                raise ValueError("frozen final selection is not derived from T4 fallback")
            row = (
                payload.get("candidates", {}).get(selected_candidate)
                if isinstance(payload.get("candidates"), Mapping)
                else None
            )
            if not (
                payload.get("manifest_id") == "oviv2_tesse_t4_matrix_v1"
                and payload.get("status") == aggregate_status == "PASS"
                and claimed == frozen["root_sha256"] == _json_hash(unhashed)
                and payload.get("shortlist") == frozen["shortlist"]
                and payload.get("protocol") == frozen["protocol"]
                and isinstance(row, Mapping)
                and row.get("status") == "PASS"
                and isinstance(row.get("gates"), Mapping)
                and set(row["gates"]) == V2_T4_METRIC_KEYS
                and all(item is True for item in row["gates"].values())
            ):
                raise ValueError("frozen selected candidate no longer has a T4 PASS")
            protocol_path = _binding_path(
                frozen["protocol"].get("path"),
                base=manifest_base,
                role="T4 protocol",
            )
            protocol = _load_json_bytes(protocol_path.read_bytes(), protocol_path)
            if set(protocol) != {
                "schema_version", "manifest_id", "dataset", "method_id",
                "protocol_id", "scene", "bounds", "candidates",
            } or not (
                protocol.get("schema_version") == 1
                and protocol.get("manifest_id") == "oviv2_tesse_t4_protocol_v1"
                and protocol.get("dataset") == "TESSE-CD"
                and protocol.get("method_id") == "OVIV2"
                and protocol.get("protocol_id") == PROTOCOL_ID
                and protocol.get("scene") == "apartment"
                and protocol.get("bounds") == V2_T4_BOUNDS
                and isinstance(protocol.get("candidates"), Mapping)
                and set(protocol["candidates"]) == set(payload.get("candidates", {}))
            ):
                raise ValueError("frozen T4 protocol identity is invalid")
            seen_sources: set[Path] = set()
            for candidate, matrix_row in payload["candidates"].items():
                protocol_row = protocol["candidates"][candidate]
                if not isinstance(matrix_row, Mapping) or not (
                    isinstance(protocol_row, Mapping)
                    and set(protocol_row) == {
                        "config_sha256", "run_manifest", "metric_sources", "raw_sources"
                    }
                    and protocol_row.get("config_sha256")
                    == matrix_row.get("config_sha256")
                ):
                    raise ValueError("frozen T4 candidate source binding is invalid")
                run_path = _binding_path(
                    protocol_row["run_manifest"].get("path"),
                    base=manifest_base,
                    role=f"T4 {candidate} run manifest",
                )
                run_record = _verify_exact_frozen_file_binding(
                    protocol_row["run_manifest"],
                    base=manifest_base,
                    role=f"T4 {candidate} run manifest",
                    expected_path=run_path,
                )
                run_payload = _load_json_bytes(run_path.read_bytes(), run_path)
                if run_path in seen_sources or not (
                    run_record["sha256"] == matrix_row.get("run_manifest_sha256")
                    and run_payload.get("dataset") == "TESSE-CD"
                    and run_payload.get("method_id") == "OVIV2"
                    and run_payload.get("protocol_id") == PROTOCOL_ID
                    and run_payload.get("scene") == "apartment"
                    and run_payload.get("candidate_id") == candidate
                    and run_payload.get("config_sha256")
                    == matrix_row.get("config_sha256")
                ):
                    raise ValueError("frozen T4 whole-profile run binding is invalid")
                seen_sources.add(run_path)
                raw_sources = protocol_row.get("raw_sources")
                if not isinstance(raw_sources, Mapping) or set(raw_sources) != (
                    V2_T4_RAW_SOURCE_KEYS
                    | {f"{name}_sha256" for name in V2_T4_RAW_SOURCE_KEYS}
                ):
                    raise ValueError("frozen T4 raw source inventory is invalid")
                if raw_sources.get("run_manifest") != protocol_row.get("run_manifest"):
                    raise ValueError("frozen T4 raw run manifest binding is invalid")
                if raw_sources.get("shortlist") != payload.get("shortlist"):
                    raise ValueError("frozen T4 raw shortlist binding is invalid")
                for raw_name in sorted(V2_T4_RAW_SOURCE_KEYS):
                    source_record = raw_sources.get(raw_name)
                    if not isinstance(source_record, Mapping):
                        raise ValueError("frozen T4 raw source record is invalid")
                    raw_path = _binding_path(
                        source_record.get("path"),
                        base=manifest_base,
                        role=f"T4 {candidate} raw {raw_name}",
                    )
                    normalized_raw = _verify_exact_frozen_file_binding(
                        source_record,
                        base=manifest_base,
                        role=f"T4 {candidate} raw {raw_name}",
                        expected_path=raw_path,
                    )
                    if raw_sources.get(f"{raw_name}_sha256") != normalized_raw["sha256"]:
                        raise ValueError("frozen T4 raw source flat hash is invalid")
                metric_sources = protocol_row.get("metric_sources")
                if not isinstance(metric_sources, Mapping) or set(metric_sources) != V2_T4_METRIC_KEYS:
                    raise ValueError("frozen T4 metric source inventory is invalid")
                for metric, source_record in metric_sources.items():
                    metric_path = _binding_path(
                        source_record.get("path"),
                        base=manifest_base,
                        role=f"T4 {candidate} {metric}",
                    )
                    normalized_source = _verify_exact_frozen_file_binding(
                        source_record,
                        base=manifest_base,
                        role=f"T4 {candidate} {metric}",
                        expected_path=metric_path,
                    )
                    metric_payload = _load_json_bytes(
                        metric_path.read_bytes(), metric_path
                    )
                    if metric_path in seen_sources or set(metric_payload) != {
                        "schema_version", "manifest_id", "scene", "candidate_id",
                        "config_sha256", "run_manifest_sha256", "metric", "value",
                    } or not (
                        metric_payload.get("schema_version") == 1
                        and metric_payload.get("manifest_id")
                        == "oviv2_tesse_t4_metric_v1"
                        and metric_payload.get("scene") == "apartment"
                        and metric_payload.get("candidate_id") == candidate
                        and metric_payload.get("config_sha256")
                        == matrix_row.get("config_sha256")
                        and metric_payload.get("run_manifest_sha256")
                        == run_record["sha256"]
                        and metric_payload.get("metric") == metric
                        and type(metric_payload.get("value")) in {int, float}
                        and math.isfinite(metric_payload["value"])
                        and metric_payload.get("value")
                        == matrix_row["metrics"][metric]
                    ):
                        raise ValueError("frozen T4 metric source binding is invalid")
                    seen_sources.add(metric_path)
        result[kind] = dict(frozen)
        result[kind]["artifact"] = artifact
        if kind == "t4":
            result[kind]["protocol_payload"] = protocol
    return result, selected_candidate


def _validate_frozen_hashes(
    value: object,
    *,
    repository: Mapping[str, Any],
    scenes: Mapping[str, Mapping[str, Any]],
    shared: Mapping[str, Any],
    release: Mapping[str, Any],
    seed_policy: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "code_sha256": _json_hash({"commit": repository["commit"], "tree": repository["tree"]}),
        "config_sha256": {
            scene: scenes[scene]["frozen_config"]["sha256"]
            for scene in ("apartment", "office")
        },
        "evaluator_sha256": {role: record["sha256"] for role, record in release.items()},
        "ground_truth_sha256": shared["occlusion_target_manifest"]["sha256"],
        "schedule_sha256": shared["schedule"]["sha256"],
        "seed_policy_sha256": seed_policy["sha256"],
    }
    if value != expected:
        raise ValueError("freeze code/config/evaluator/GT/schedule/seed hashes differ")
    return expected


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dataset_timestamp_ns(dataset: Any, frame_index: int) -> int:
    value = dataset.timestamp_ns(frame_index)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("dataset timestamp_ns must be an exact integer")
    result = int(value)
    if result < 0:
        raise ValueError("dataset timestamp_ns must be non-negative")
    return result


def _validate_config(
    config: Mapping[str, Any],
) -> tuple[str, int, Path, tuple[int, ...], Any]:
    missing = _V2_CONFIG_KEYS - set(config)
    unknown = set(config) - _V2_CONFIG_KEYS
    if missing:
        raise ValueError(f"v2 runner config has missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"v2 runner config has unknown keys: {sorted(unknown)}")
    for key in _STRING_CONFIG_FIELDS:
        value = config[key]
        if type(value) is not str or not value.strip():
            raise TypeError(f"{key} must be an exact non-empty string")
    for key in _INTEGER_CONFIG_FIELDS:
        if type(config[key]) is not int:
            raise TypeError(f"{key} must be an exact integer")
    nonnegative_integer_fields = {
        "structure_min_component_pixels",
        "structure_object_exclusion_dilation",
    }
    for key in _INTEGER_CONFIG_FIELDS - {"schema_version"}:
        minimum = 0 if key in nonnegative_integer_fields else 1
        if config[key] < minimum or config[key] > 10_000_000:
            raise ValueError(f"{key} is outside the supported runner range")
    for key in _FLOAT_CONFIG_FIELDS:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{key} must be a finite real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{key} must be finite")
    if type(config["structure_enabled"]) is not bool:
        raise TypeError("structure_enabled must be an exact boolean")
    if config.get("schema_version") != 2:
        raise ValueError("v2 runner schema_version must be 2")
    if (
        config.get("protocol_id") != PROTOCOL_ID
        or config.get("dataset") != "TESSE-CD"
        or config.get("method_id") != "OVIV2"
    ):
        raise ValueError("runner identity must be oviv2-tessecd-v2 on TESSE-CD")
    scene = config.get("scene")
    if scene not in {"apartment", "office"}:
        raise ValueError("scene must be apartment or office")
    frame_count = _positive_integer(config.get("frame_count"), "frame_count")
    if config.get("missing_observation_policy") not in {
        "signed_depth",
        "missing_as_absence",
    }:
        raise ValueError("missing_observation_policy is invalid")

    from src.oviv2.temporal_config import temporal_config_from_json

    temporal_config = temporal_config_from_json(
        {"temporal_readout": config.get("temporal_readout")}
    )
    if temporal_config.geometry.voxel_size_m != config.get("voxel_size_m"):
        raise ValueError("temporal voxel_size_m must match cumulative geometry")
    if temporal_config.geometry.depth_max_m != config.get("depth_max_m"):
        raise ValueError("temporal depth_max_m must match cumulative geometry")
    from src.oviv2.runner_config import (
        runtime_config_from_json,
        semantic_fusion_config_from_json,
        structure_config_from_json,
    )

    runtime_config_from_json(dict(config))
    structure_config_from_json(dict(config), voxel_size_m=float(config["voxel_size_m"]))
    semantic_fusion_config_from_json(dict(config))
    from src.oviv2.temporal_dense_observations import TemporalDenseObservationConfig
    from src.oviv2.temporal_depth_observations import TemporalDepthObservationConfig
    from src.oviv2.temporal_observation_merge import TemporalObservationMergeConfig

    TemporalDenseObservationConfig(
        sample_stride=config["dense_sample_stride"],
        depth_max_m=config["depth_max_m"],
        minimum_area_px=config["temporal_dense_minimum_area_px"],
        maximum_area_px=config["temporal_dense_maximum_area_px"],
        maximum_observations=config["temporal_dense_maximum_observations"],
        voxel_size_m=config["voxel_size_m"],
        pixel_stride=config["temporal_proposal_pixel_stride"],
        min_valid_points=config["temporal_proposal_min_valid_points"],
    )
    TemporalDepthObservationConfig(
        edge_threshold_m=config["temporal_depth_edge_threshold_m"],
        minimum_area_px=config["temporal_depth_minimum_area_px"],
        maximum_area_px=config["temporal_depth_maximum_area_px"],
        plane_minimum_area_px=config["temporal_depth_plane_minimum_area_px"],
        planar_rmse_threshold_m=config["temporal_depth_planar_rmse_threshold_m"],
        semantic_minimum_votes=config["temporal_depth_semantic_minimum_votes"],
        semantic_minimum_fraction=config["temporal_depth_semantic_minimum_fraction"],
        semantic_minimum_probability=config["temporal_depth_semantic_minimum_probability"],
        maximum_observations=config["temporal_depth_maximum_observations"],
        maximum_unknown_observations=config[
            "temporal_depth_maximum_unknown_observations"
        ],
        depth_max_m=config["depth_max_m"],
        voxel_size_m=config["voxel_size_m"],
        pixel_stride=config["temporal_proposal_pixel_stride"],
        min_valid_points=config["temporal_proposal_min_valid_points"],
    )
    TemporalObservationMergeConfig(
        same_semantic_iou_threshold=config["temporal_merge_same_semantic_iou"],
        supplement_containment_threshold=config[
            "temporal_merge_supplement_containment"
        ],
        primary_duplicate_iou_threshold=config[
            "temporal_merge_primary_duplicate_iou"
        ],
    )
    configured_hash = config.get("algorithm_hash")
    if not _is_sha256(configured_hash) or configured_hash != algorithm_hash(config):
        raise ValueError("configured algorithm_hash does not match mapping parameters")
    schedule = _resolve_path(config.get("schedule_manifest", ""))
    raw_frames = config.get("evaluation_checkpoint_frames")
    if not isinstance(raw_frames, list) or any(
        type(frame) is not int or not 0 <= frame < frame_count for frame in raw_frames
    ):
        raise ValueError("evaluation_checkpoint_frames must contain in-range integers")
    evaluation_frames = tuple(raw_frames)
    if list(evaluation_frames) != sorted(set(evaluation_frames)):
        raise ValueError("evaluation_checkpoint_frames must be sorted and unique")
    return str(scene), frame_count, schedule, evaluation_frames, temporal_config


def load_v2_frozen_run_context(
    freeze_manifest: str | Path,
    *,
    run_slot: str,
    source_config: Path,
    source_config_bytes: bytes,
    config: Mapping[str, Any],
    destination: Path,
) -> FrozenRunContext:
    manifest_path = _absolute_lexical(freeze_manifest)
    _require_regular_file(manifest_path, "freeze manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_json_bytes(manifest_bytes, manifest_path)
    if not (
        set(manifest) == _FREEZE_TOP_KEYS
        and manifest.get("schema_version") == 1
        and manifest.get("freeze_id") == PROTOCOL_ID
        and manifest.get("status") == "FROZEN"
        and manifest.get("method") == "OVIV2"
        and manifest.get("dataset") == "TESSE-CD"
    ):
        raise ValueError(f"formal runner requires the FROZEN {PROTOCOL_ID} manifest")
    repository = manifest.get("repository")
    if not (
        isinstance(repository, Mapping)
        and set(repository) == _FREEZE_REPOSITORY_KEYS
        and repository.get("clean") is True
        and repository.get("stage3_lineage_commit")
        == "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5"
        and repository.get("stage3_is_ancestor") is True
        and _is_git_id(repository.get("commit"))
        and _is_git_id(repository.get("tree"))
        and type(repository.get("parents")) is list
        and all(_is_git_id(parent) for parent in repository["parents"])
        and type(repository.get("commit_time_utc")) is str
        and bool(repository["commit_time_utc"].strip())
    ):
        raise ValueError("freeze repository identity is invalid")
    current = _validate_repository_state(_repository_provenance(), repository)

    scene = str(config["scene"])
    scenes = manifest.get("scenes")
    if not isinstance(scenes, Mapping) or set(scenes) != {"apartment", "office"}:
        raise ValueError("freeze scene bindings are incomplete")
    loaded_configs: dict[str, Mapping[str, Any]] = {}
    normalized_scenes: dict[str, dict[str, Any]] = {}
    selected_config_record: dict[str, Any] | None = None
    for bound_scene in ("apartment", "office"):
        selected = scenes.get(bound_scene)
        if not isinstance(selected, Mapping) or set(selected) != _FREEZE_SCENE_KEYS:
            raise ValueError(f"freeze {bound_scene} scene binding schema is invalid")
        frozen_config = selected["frozen_config"]
        if not isinstance(frozen_config, Mapping):
            raise ValueError(f"freeze {bound_scene} config binding is invalid")
        frozen_path = _binding_path(
            frozen_config.get("path"),
            base=manifest_path.parent,
            role=f"{bound_scene} frozen config",
        )
        config_record = _verify_exact_frozen_file_binding(
            frozen_config,
            base=manifest_path.parent,
            role=f"{bound_scene} frozen config",
            expected_path=source_config if bound_scene == scene else frozen_path,
        )
        frozen_bytes = frozen_path.read_bytes()
        frozen_payload = _load_json_bytes(frozen_bytes, frozen_path)
        parsed_scene, _, _, _, temporal_config = _validate_config(frozen_payload)
        if parsed_scene != bound_scene:
            raise ValueError("freeze scene config identity mismatch")
        loaded_configs[bound_scene] = frozen_payload
        normalized_scenes[bound_scene] = {"frozen_config": dict(frozen_config)}
        for role in (
            "export_manifest",
            "frontend_manifest",
            "dense_manifest",
            "vocabulary_json",
            "vocabulary_txt",
        ):
            _verify_exact_frozen_file_binding(
                selected[role],
                base=manifest_path.parent,
                role=f"{bound_scene} {role}",
                expected_path=_resolve_path(frozen_payload[role]),
            )
            normalized_scenes[bound_scene][role] = dict(selected[role])
        from src.oviv2.temporal_config import ExecutionProfile

        temporal_binding = selected["temporal_frontend_manifest"]
        if temporal_config.execution_profile in {
            ExecutionProfile.A0,
            ExecutionProfile.A1,
        }:
            if temporal_binding is not None:
                raise ValueError(
                    f"freeze {bound_scene} reference profile cannot bind temporal frontend"
                )
            normalized_scenes[bound_scene]["temporal_frontend_manifest"] = None
        else:
            _verify_exact_frozen_file_binding(
                temporal_binding,
                base=manifest_path.parent,
                role=f"{bound_scene} temporal_frontend_manifest",
                expected_path=_resolve_path(
                    frozen_payload["temporal_frontend_manifest"]
                ),
            )
            normalized_scenes[bound_scene]["temporal_frontend_manifest"] = dict(
                temporal_binding
            )
        if bound_scene == scene:
            selected_config_record = config_record
    if selected_config_record is None:
        raise ValueError("freeze selected config binding is missing")
    if selected_config_record != _byte_record(source_config_bytes):
        raise ValueError("frozen config content does not match runner config")
    algorithm = manifest.get("algorithm")
    if not (
        isinstance(algorithm, Mapping)
        and set(algorithm) == {"sha256", "normalized_config"}
        and _is_sha256(algorithm.get("sha256"))
        and isinstance(algorithm.get("normalized_config"), Mapping)
        and algorithm.get("sha256") == config["algorithm_hash"]
        and algorithm.get("normalized_config") == algorithm_config(config)
        and all(
            frozen["algorithm_hash"] == algorithm["sha256"]
            and algorithm_config(frozen) == algorithm["normalized_config"]
            for frozen in loaded_configs.values()
        )
    ):
        raise ValueError("runner config algorithm differs from the freeze")
    normalized_selection = _validate_selection(
        manifest["selection"],
        manifest_base=manifest_path.parent,
        apartment_config=loaded_configs["apartment"],
        algorithm_sha256=algorithm["sha256"],
    )
    normalized_environment = _validate_environment(manifest["environment"])
    normalized_models = _validate_models(
        manifest["models"],
        scenes=normalized_scenes,
        manifest_base=manifest_path.parent,
    )
    normalized_release = _validate_release_bindings(
        manifest["release_bindings"], manifest_base=manifest_path.parent
    )
    normalized_office_audit = _validate_office_audit(
        manifest["office_pre_freeze_audit"]
    )
    normalized_seed_policy = _validate_seed_policy(manifest["seed_policy"])
    normalized_evidence, selected_candidate = _validate_frozen_evidence(
        manifest["evidence"],
        manifest_base=manifest_path.parent,
        selection_artifact=normalized_selection["artifact"],
    )

    output_roots = manifest.get("output_roots")
    expected_slots = {"apartment_run1", "apartment_run2"} | {
        f"office_seed_{seed}" for seed in normalized_seed_policy["seeds"]
    }
    if not isinstance(output_roots, Mapping) or set(output_roots) != expected_slots:
        raise ValueError("freeze output slots are incomplete")
    if any(
        type(value) is not str
        or not Path(value).is_absolute()
        or value != os.fspath(_absolute_lexical(value))
        for value in output_roots.values()
    ):
        raise ValueError("frozen run slot output roots are invalid")
    expected_scene_slots = (
        {"apartment_run1", "apartment_run2"}
        if scene == "apartment"
        else {f"office_seed_{seed}" for seed in normalized_seed_policy["seeds"]}
    )
    if run_slot not in expected_scene_slots:
        raise ValueError("run slot does not match the frozen scene")
    raw_output = output_roots.get(run_slot)
    if not isinstance(raw_output, str) or not Path(raw_output).is_absolute():
        raise ValueError("frozen run slot output root is invalid")
    if (
        raw_output != os.fspath(_absolute_lexical(raw_output))
        or _absolute_lexical(raw_output) != destination
    ):
        raise ValueError("run slot output does not match the frozen output root")
    normalized_roots = [_absolute_lexical(str(value)) for value in output_roots.values()]
    if len(set(normalized_roots)) != len(normalized_roots):
        raise ValueError("frozen output roots must be distinct")
    normalized_commands = _validate_commands(
        manifest["commands"],
        manifest_path=manifest_path,
        scenes=normalized_scenes,
        output_roots=output_roots,
    )

    shared = manifest.get("shared_bindings")
    if not isinstance(shared, Mapping) or set(shared) != _FREEZE_SHARED_KEYS:
        raise ValueError("freeze shared input binding schema is invalid")
    shared_config_fields = {
        "input_manifest": "input_manifest",
        "schedule": "schedule_manifest",
        "occlusion_target_manifest": "occlusion_target_manifest",
    }
    normalized_shared: dict[str, Any] = {}
    for role, config_field in shared_config_fields.items():
        expected_paths = {
            _resolve_path(frozen[config_field]) for frozen in loaded_configs.values()
        }
        if len(expected_paths) != 1:
            raise ValueError(f"freeze shared {role} config paths differ across scenes")
        normalized_shared[role] = _verify_exact_frozen_file_binding(
            shared[role],
            base=manifest_path.parent,
            role=f"shared {role}",
            expected_path=next(iter(expected_paths)),
        )
    normalized_hashes = _validate_frozen_hashes(
        manifest["frozen_hashes"],
        repository=repository,
        scenes=normalized_scenes,
        shared=normalized_shared,
        release=normalized_release,
        seed_policy=normalized_seed_policy,
    )
    authorizations = manifest.get("office_authorizations")
    office_slots = {f"office_seed_{seed}" for seed in normalized_seed_policy["seeds"]}
    if not isinstance(authorizations, Mapping) or set(authorizations) != office_slots:
        raise ValueError("freeze Office authorizations are incomplete")
    normalized_authorizations: dict[str, Any] = {}
    for slot in sorted(office_slots):
        authorization = authorizations[slot]
        expected_seed = int(slot.removeprefix("office_seed_"))
        if not isinstance(authorization, Mapping) or set(authorization) != {
            "authorization_id", "scene", "seed", "config_sha256", "algorithm_hash", "output_root",
            "t1_root_sha256", "t4_root_sha256", "frozen_hashes_sha256",
            "authorization_sha256",
        }:
            raise ValueError("freeze Office authorization schema is invalid")
        body = dict(authorization)
        claimed = body.pop("authorization_sha256")
        if not (
            authorization.get("authorization_id") == slot
            and authorization.get("scene") == "office"
            and authorization.get("seed") == expected_seed
            and authorization.get("config_sha256") == _json_hash(loaded_configs["office"])
            and authorization.get("algorithm_hash") == algorithm["sha256"]
            and authorization.get("output_root") == output_roots[slot]
            and authorization.get("t1_root_sha256") == normalized_evidence["t1"]["root_sha256"]
            and authorization.get("t4_root_sha256") == normalized_evidence["t4"]["root_sha256"]
            and authorization.get("frozen_hashes_sha256") == _json_hash(normalized_hashes)
            and claimed == _json_hash(body)
        ):
            raise ValueError("freeze Office authorization hash binding is invalid")
        normalized_authorizations[slot] = dict(authorization)
    if scene == "office" and run_slot not in normalized_authorizations:
        raise ValueError("Office requires frozen authorization")
    input_bindings = {
        "repository": dict(repository),
        "algorithm": dict(algorithm),
        "selection": normalized_selection,
        "environment": normalized_environment,
        "commands": normalized_commands,
        "models": normalized_models,
        "release_bindings": normalized_release,
        "office_pre_freeze_audit": normalized_office_audit,
        "output_roots": dict(output_roots),
        "shared_bindings": dict(shared),
        "scenes": normalized_scenes,
        "evidence": normalized_evidence,
        "seed_policy": normalized_seed_policy,
        "frozen_hashes": normalized_hashes,
        "office_authorizations": normalized_authorizations,
    }
    _revalidate_frozen_bindings(input_bindings, base=manifest_path.parent)
    identity = {
        "schema_version": 1,
        "freeze_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "scene": scene,
        "freeze_manifest": _byte_record(manifest_bytes),
        "repository": {
            "commit": current["repository_commit"],
            "tree": current["repository_tree"],
        },
        "config": selected_config_record,
        "algorithm_hash": algorithm["sha256"],
        "selected_candidate_id": selected_candidate,
        **(
            {"office_authorization": normalized_authorizations[run_slot]}
            if scene == "office"
            else {}
        ),
        "input_bindings_sha256": _json_hash(input_bindings),
        "formal_evidence_sha256": _json_hash(manifest),
    }
    return FrozenRunContext(
        manifest_path,
        manifest_bytes,
        current,
        input_bindings,
        identity,
        run_slot,
    )


def _production_runtime_factory(config: Mapping[str, Any], caches: Any) -> Any:
    from src.oviv2.dual_readout import DualReadoutRuntime
    from src.oviv2.reference_readout import (
        LifecycleOverlayReadout,
        ReferenceCurrentReadout,
    )
    from src.oviv2.runner_config import runtime_config_from_json
    from src.oviv2.runtime import Oviv2Runtime
    from src.oviv2.temporal_config import ExecutionProfile, temporal_config_from_json
    from src.oviv2.temporal_runtime import TemporalCurrentRuntime

    runtime_config = apply_frozen_visibility_policy(
        runtime_config_from_json(dict(config)),
        str(config["missing_observation_policy"]),
    )
    temporal_config = temporal_config_from_json(
        {"temporal_readout": config["temporal_readout"]}
    )
    cumulative = Oviv2Runtime(
        str(config["scene"]),
        runtime_config,
        dense_semantic_provenance=caches.dense_provenance,
    )
    if temporal_config.execution_profile is ExecutionProfile.A0:
        temporal = ReferenceCurrentReadout(str(config["scene"]), temporal_config)
    elif temporal_config.execution_profile is ExecutionProfile.A1:
        temporal = LifecycleOverlayReadout(str(config["scene"]), temporal_config)
    else:
        temporal_tracker_config = replace(
            runtime_config.tracker,
            confirm_hits=temporal_config.confirm_hits,
        )
        temporal = TemporalCurrentRuntime(
            str(config["scene"]),
            temporal_config,
            tracker_config=temporal_tracker_config,
        )
    return DualReadoutRuntime(cumulative, temporal)


_production_environment = _collect_formal_environment


_TEMPORAL_FRONTEND_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "method",
        "dataset",
        "scene",
        "frame_count",
        "source_frame_ids",
        "source_frame_ids_hash",
        "image_shape",
        "class_count",
        "classes",
        "vocabulary_sha256",
        "algorithm_hash",
        "feature_model_id",
        "cache_files_sha256",
        "cache_prefix_sha256",
        "temporal_only",
        "alias_map_sha256",
        "merge_algorithm",
        "merge_policy",
        "diagnostics",
        "temporal_frontend_sources",
    }
)
_TEMPORAL_FRONTEND_OPTIONAL_KEYS = frozenset(
    {"input_manifest_sha256", "input_witness", "provenance_sha256"}
)


def _validate_temporal_frontend_manifest(
    manifest: object,
    *,
    scene: str,
    frame_count: int,
    classes: tuple[str, ...],
    vocabulary_sha256: str,
    feature_model_id: str,
) -> dict[str, str]:
    if not isinstance(manifest, dict):
        raise ValueError("temporal frontend manifest must be an object")
    keys = set(manifest)
    if not (
        _TEMPORAL_FRONTEND_REQUIRED_KEYS <= keys
        and keys <= _TEMPORAL_FRONTEND_REQUIRED_KEYS | _TEMPORAL_FRONTEND_OPTIONAL_KEYS
    ):
        raise ValueError("temporal frontend manifest fields are not exact")
    source_frame_ids = list(range(frame_count))
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("method") == "OVIV2"
        and manifest.get("dataset") == "TESSE-CD"
        and manifest.get("scene") == scene
        and manifest.get("frame_count") == frame_count
        and manifest.get("source_frame_ids") == source_frame_ids
        and manifest.get("source_frame_ids_hash") == _json_hash(source_frame_ids)
        and manifest.get("image_shape") == [480, 720]
        and manifest.get("class_count") == len(classes)
        and manifest.get("classes") == list(classes)
        and manifest.get("vocabulary_sha256") == vocabulary_sha256
        and manifest.get("feature_model_id") == feature_model_id
        and manifest.get("temporal_only") is True
        and _is_sha256(manifest.get("algorithm_hash"))
        and _is_sha256(manifest.get("alias_map_sha256"))
        and isinstance(manifest.get("merge_algorithm"), Mapping)
        and manifest["merge_algorithm"].get("sha256")
        == manifest.get("algorithm_hash")
        and isinstance(manifest.get("merge_policy"), Mapping)
        and isinstance(manifest.get("diagnostics"), Mapping)
        and isinstance(manifest.get("temporal_frontend_sources"), Mapping)
    ):
        raise ValueError("temporal frontend manifest identity mismatch")
    raw_hashes = manifest.get("cache_files_sha256")
    if not isinstance(raw_hashes, dict):
        raise ValueError("temporal frontend cache_files_sha256 must be an object")
    expected_names = [f"frame{index:06d}.pkl.gz" for index in range(frame_count)]
    if list(raw_hashes) != expected_names:
        raise ValueError("temporal frontend checksum keys are not canonical")
    hashes: dict[str, str] = {}
    for name in expected_names:
        checksum = raw_hashes[name]
        if not _is_sha256(checksum):
            raise ValueError(f"temporal frontend checksum is invalid: {name}")
        hashes[name] = checksum
    if manifest.get("cache_prefix_sha256") != _cache_prefix_sha256(hashes):
        raise ValueError("temporal frontend cache prefix checksum mismatch")
    return hashes


@dataclass
class _TemporalProductionCaches:
    base: Any
    temporal_frontend: Any
    temporal_cache_dir: Path
    temporal_manifest_path: Path
    temporal_hashes: dict[str, str]
    temporal_manifest_sha256: str
    dense_config: Any
    depth_config: Any
    merge_config: Any
    bindings: dict[str, Any]
    input_hashes: dict[Path, str]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def load(self, frame_index: int, frame: Any) -> tuple[tuple[Any, ...], Any]:
        return self.base.load(frame_index, frame)

    def load_temporal(
        self, frame_index: int, frame: Any, dense: Any
    ) -> tuple[Any, ...]:
        from src.oviv2.temporal_dense_observations import (
            generate_temporal_dense_observations,
        )
        from src.oviv2.temporal_depth_observations import (
            generate_temporal_depth_observations,
        )
        from src.oviv2.temporal_observation_merge import (
            merge_primary_authoritative_observations,
            regularize_temporal_object_extents,
            suppress_near_duplicate_primary_observations,
        )

        if dense is None:
            raise ValueError("temporal proposal generation requires dense semantics")
        primary = self.temporal_frontend.observe(frame, frame_index)
        primary = suppress_near_duplicate_primary_observations(
            primary,
            iou_threshold=self.merge_config.primary_duplicate_iou_threshold,
        )
        class_names = tuple(self.base.class_names[1 : 1 + dense.class_count])
        dense_observations = generate_temporal_dense_observations(
            frame, dense, class_names, self.dense_config
        )
        depth_observations = generate_temporal_depth_observations(
            frame, dense, class_names, self.depth_config
        )
        merged_objects = merge_primary_authoritative_observations(
            primary,
            (dense_observations, depth_observations),
            self.merge_config,
        )
        merged_objects = regularize_temporal_object_extents(
            merged_objects,
            self.dense_config.voxel_size_m,
        )
        structure = self.base.structure.observe(
            frame, object_observations=merged_objects
        )
        return (*merged_objects, *structure)

    def assert_inputs_unchanged(self) -> None:
        self.base.assert_inputs_unchanged()
        _reject_symlink_components(
            self.temporal_cache_dir, "temporal frontend cache directory"
        )
        expected = set(self.temporal_hashes) | {self.temporal_manifest_path.name}
        entries = list(self.temporal_cache_dir.iterdir())
        if {path.name for path in entries} != expected or any(
            not stat.S_ISREG(os.lstat(path).st_mode) for path in entries
        ):
            raise ValueError("temporal frontend cache directory changed during run")
        _require_regular_file(self.temporal_manifest_path, "temporal frontend manifest")
        if _sha256(self.temporal_manifest_path) != self.temporal_manifest_sha256:
            raise ValueError("temporal frontend manifest changed during run")
        for name, checksum in self.temporal_hashes.items():
            path = self.temporal_cache_dir / name
            _require_regular_file(path, "temporal frontend cache frame")
            if _sha256(path) != checksum:
                raise ValueError(f"temporal frontend cache changed during run: {name}")


def _production_v2_cache_loader_factory(
    config: Mapping[str, Any], dataset: Any
) -> Any:
    from scripts.precompute_oviv2_tesse_frontend import (
        _read_cache_snapshot,
        _validate_cache_payload,
    )
    from src.oviv2.observations import CachedFrontendAdapter
    from src.oviv2.temporal_dense_observations import TemporalDenseObservationConfig
    from src.oviv2.temporal_depth_observations import TemporalDepthObservationConfig
    from src.oviv2.temporal_observation_merge import TemporalObservationMergeConfig
    from src.oviv2.temporal_config import ExecutionProfile, temporal_config_from_json

    base = _production_cache_loader_factory(config, dataset)
    profile = temporal_config_from_json(
        {"temporal_readout": config.get("temporal_readout")}
    ).execution_profile
    if profile in {ExecutionProfile.A0, ExecutionProfile.A1}:
        return base
    cache_dir = _resolve_path(config["temporal_frontend_cache_dir"])
    manifest_path = _resolve_path(config["temporal_frontend_manifest"])
    _reject_symlink_components(cache_dir, "temporal frontend cache directory")
    if not cache_dir.is_dir():
        raise FileNotFoundError(cache_dir)
    if manifest_path.parent.resolve() != cache_dir.resolve() or manifest_path.name != "frontend_manifest.json":
        raise ValueError("temporal frontend manifest must be inside its cache directory")
    _require_regular_file(manifest_path, "temporal frontend manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_json_bytes(manifest_bytes, manifest_path)
    object_count = len(base.object_semantic_ids)
    classes = tuple(base.class_names[1 : 1 + object_count])
    feature_model_id = base.frontend.feature_model_id
    if not isinstance(feature_model_id, str):
        raise ValueError("base frontend feature model identity is missing")
    vocabulary_sha256 = _sha256(_resolve_path(config["vocabulary_txt"]))
    hashes = _validate_temporal_frontend_manifest(
        manifest,
        scene=str(config["scene"]),
        frame_count=int(config["frame_count"]),
        classes=classes,
        vocabulary_sha256=vocabulary_sha256,
        feature_model_id=feature_model_id,
    )
    expected_entries = set(hashes) | {manifest_path.name}
    entries = list(cache_dir.iterdir())
    if {path.name for path in entries} != expected_entries or any(
        not stat.S_ISREG(os.lstat(path).st_mode) for path in entries
    ):
        raise ValueError("temporal frontend cache has missing, extra, or symlink entries")

    canonical_manifest_path = _resolve_path(config["frontend_manifest"])
    canonical_manifest_bytes = canonical_manifest_path.read_bytes()
    canonical_manifest = _load_json_bytes(
        canonical_manifest_bytes, canonical_manifest_path
    )
    for optional in _TEMPORAL_FRONTEND_OPTIONAL_KEYS:
        if optional in manifest and manifest[optional] != canonical_manifest.get(optional):
            raise ValueError(f"temporal frontend {optional} differs from canonical source")
    sources = manifest["temporal_frontend_sources"]
    canonical_source = sources.get("canonical")
    if not (
        isinstance(canonical_source, Mapping)
        and canonical_source.get("manifest_sha256")
        == _sha256_bytes(canonical_manifest_bytes)
        and canonical_source.get("algorithm_hash")
        == canonical_manifest.get("algorithm_hash")
        and canonical_source.get("cache_prefix_sha256")
        == canonical_manifest.get("cache_prefix_sha256")
        and isinstance(sources.get("alias_shards"), list)
        and bool(sources["alias_shards"])
        and sources.get("alias_map", {}).get("sha256")
        == manifest.get("alias_map_sha256")
    ):
        raise ValueError("temporal frontend source bindings are invalid")

    vocabulary = base.frontend.vocabulary

    class BoundTemporalFrontendAdapter(CachedFrontendAdapter):
        def _load(self, cache_frame_id: int) -> tuple[Any, ...]:
            name = f"frame{cache_frame_id:06d}.pkl.gz"
            path = cache_dir / name
            payload, binding = _read_cache_snapshot(path)
            if binding.sha256 != hashes[name]:
                raise ValueError("temporal frontend cache checksum mismatch")
            _validate_cache_payload(
                payload, classes=classes, image_shape=(480, 720), path=path
            )
            assert isinstance(payload, dict)
            masks = np.asarray(payload["mask"], dtype=bool)
            boxes = np.asarray(payload["xyxy"], dtype=np.float32)
            confidences = np.asarray(payload["confidence"], dtype=np.float32)
            class_ids = np.asarray(payload["class_id"], dtype=np.int64)
            image_features = np.asarray(payload["image_feats"], dtype=np.float64)
            text_features = np.asarray(payload["text_feats"], dtype=np.float64)
            return (
                masks,
                boxes,
                confidences,
                [classes[int(index)] for index in class_ids],
                image_features,
                text_features,
            )

    temporal_frontend = BoundTemporalFrontendAdapter(
        cache_dir,
        vocabulary,
        voxel_size_m=float(config["voxel_size_m"]),
        pixel_stride=int(config["temporal_proposal_pixel_stride"]),
        min_valid_points=int(config["temporal_proposal_min_valid_points"]),
        feature_model_id=feature_model_id,
    )
    dense_config = TemporalDenseObservationConfig(
        sample_stride=int(config["dense_sample_stride"]),
        depth_max_m=float(config["depth_max_m"]),
        minimum_area_px=int(config["temporal_dense_minimum_area_px"]),
        maximum_area_px=int(config["temporal_dense_maximum_area_px"]),
        maximum_observations=int(config["temporal_dense_maximum_observations"]),
        voxel_size_m=float(config["voxel_size_m"]),
        pixel_stride=int(config["temporal_proposal_pixel_stride"]),
        min_valid_points=int(config["temporal_proposal_min_valid_points"]),
    )
    depth_config = TemporalDepthObservationConfig(
        edge_threshold_m=float(config["temporal_depth_edge_threshold_m"]),
        minimum_area_px=int(config["temporal_depth_minimum_area_px"]),
        maximum_area_px=int(config["temporal_depth_maximum_area_px"]),
        plane_minimum_area_px=int(config["temporal_depth_plane_minimum_area_px"]),
        planar_rmse_threshold_m=float(config["temporal_depth_planar_rmse_threshold_m"]),
        semantic_minimum_votes=int(config["temporal_depth_semantic_minimum_votes"]),
        semantic_minimum_fraction=float(config["temporal_depth_semantic_minimum_fraction"]),
        semantic_minimum_probability=float(config["temporal_depth_semantic_minimum_probability"]),
        maximum_observations=int(config["temporal_depth_maximum_observations"]),
        maximum_unknown_observations=int(
            config["temporal_depth_maximum_unknown_observations"]
        ),
        depth_max_m=float(config["depth_max_m"]),
        voxel_size_m=float(config["voxel_size_m"]),
        pixel_stride=int(config["temporal_proposal_pixel_stride"]),
        min_valid_points=int(config["temporal_proposal_min_valid_points"]),
    )
    merge_config = TemporalObservationMergeConfig(
        same_semantic_iou_threshold=float(config["temporal_merge_same_semantic_iou"]),
        supplement_containment_threshold=float(config["temporal_merge_supplement_containment"]),
        primary_duplicate_iou_threshold=float(
            config["temporal_merge_primary_duplicate_iou"]
        ),
    )
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    bindings = dict(base.bindings)
    bindings["temporal_frontend_manifest"] = _byte_record(manifest_bytes)
    input_hashes = dict(base.input_hashes)
    input_hashes[manifest_path] = manifest_sha256
    input_hashes.update({cache_dir / name: digest for name, digest in hashes.items()})
    return _TemporalProductionCaches(
        base=base,
        temporal_frontend=temporal_frontend,
        temporal_cache_dir=cache_dir,
        temporal_manifest_path=manifest_path,
        temporal_hashes=hashes,
        temporal_manifest_sha256=manifest_sha256,
        dense_config=dense_config,
        depth_config=depth_config,
        merge_config=merge_config,
        bindings=bindings,
        input_hashes=input_hashes,
    )


def _production_dependencies() -> RunnerDependencies:
    return RunnerDependencies(
        dataset_factory=_production_dataset_factory,
        cache_loader_factory=_production_v2_cache_loader_factory,
        runtime_factory=_production_runtime_factory,
        provenance_factory=_production_provenance,
        environment_factory=_production_environment,
    )


def _load_temporal_observations(
    profile: Any,
    caches: Any,
    frame_index: int,
    frame: Any,
    dense_semantics: Any,
) -> tuple[Any, ...] | None:
    from src.oviv2.temporal_config import ExecutionProfile

    if profile in {ExecutionProfile.A0, ExecutionProfile.A1}:
        return None
    if profile not in {ExecutionProfile.A2, ExecutionProfile.A3, ExecutionProfile.A4}:
        raise TypeError("profile must be an ExecutionProfile")
    loader = getattr(caches, "load_temporal", None)
    if not callable(loader):
        raise ValueError("temporal profile requires a callable temporal cache loader")
    observations = loader(frame_index, frame, dense_semantics)
    if type(observations) is not tuple:
        raise TypeError("temporal cache loader must return an exact tuple")
    return observations


def _snapshot_from_runtime(
    runtime: Any,
    *,
    checkpoint: TesseCausalCheckpoint,
    scene: str,
    config_sha256: str,
    voxel_size_m: float,
) -> TemporalCurrentSnapshot:
    temporal = getattr(runtime, "temporal", None)
    state = getattr(temporal, "state", None)
    if state is None:
        raise ValueError("dual runtime does not expose temporal state")
    expected_timestamp = checkpoint.timestamp_ns / 1_000_000_000
    if not (
        state.scene_id == scene
        and state.last_frame_id == checkpoint.frame_index
        and state.revision == checkpoint.frame_index + 1
        and float(state.last_timestamp) == expected_timestamp
    ):
        raise ValueError("temporal state does not match checkpoint progress")
    metadata = TemporalSnapshotMetadata(
        scene_id=state.scene_id,
        frame_id=state.last_frame_id,
        timestamp=state.last_timestamp,
        revision=state.revision,
        voxel_size_m=voxel_size_m,
        config_sha256=config_sha256,
    )
    return TemporalCurrentSnapshot(metadata, state.entities, state.background)


def _cumulative_neutral_from_runtime(
    runtime: Any,
    *,
    checkpoint: TesseCausalCheckpoint,
    caches: Any,
) -> MapSnapshot:
    from src.evaluation.oviv2_tesse import build_neutral_current_snapshot
    from src.oviv2.runtime import Oviv2Runtime

    injected_checkpoint = getattr(runtime, "cumulative_audit_checkpoint", None)
    if callable(injected_checkpoint):
        injected = injected_checkpoint(checkpoint, caches)
        if not isinstance(injected, MapSnapshot):
            raise TypeError("cumulative audit checkpoint must be a MapSnapshot")
        temporal_state = getattr(getattr(runtime, "temporal", None), "state", None)
        if not (
            temporal_state is not None
            and injected.scene_id == temporal_state.scene_id
            and injected.timestamp == checkpoint.timestamp_ns / 1_000_000_000
        ):
            raise ValueError("cumulative audit checkpoint does not match progress")
        return injected

    cumulative = getattr(runtime, "cumulative", None)
    if not isinstance(cumulative, Oviv2Runtime):
        raise ValueError("dual runtime does not expose cumulative Oviv2Runtime")
    expected_timestamp = checkpoint.timestamp_ns / 1_000_000_000
    if not (
        cumulative.last_frame_id == checkpoint.frame_index
        and cumulative.revision == checkpoint.frame_index + 1
        and float(cumulative.last_timestamp) == expected_timestamp
    ):
        raise ValueError("cumulative state does not match checkpoint progress")
    required = (
        "class_names",
        "object_semantic_ids",
        "semantic_fusion",
        "timestamp_ns_by_frame",
    )
    if any(not hasattr(caches, name) for name in required):
        raise TypeError("cumulative neutral export requires production cache bindings")
    with tempfile.TemporaryDirectory(prefix="oviv2-v1-neutral-") as temporary:
        snapshot = cumulative.commit_new(Path(temporary) / "cumulative")
        neutral = build_neutral_current_snapshot(
            snapshot,
            timestamp_ns=checkpoint.timestamp_ns,
            class_names=caches.class_names,
            object_semantic_ids=caches.object_semantic_ids,
            fusion=caches.semantic_fusion,
            timestamp_ns_by_frame=caches.timestamp_ns_by_frame,
        )
        snapshot.revalidate_source()
    return neutral


def _compose_checkpoint_neutral(
    profile: Any,
    *,
    cumulative: MapSnapshot | None,
    reference_state: Any | None,
    temporal: MapSnapshot | None,
    expected_timestamp_ns: int | None = None,
) -> MapSnapshot:
    from src.oviv2.reference_readout import ReferenceReadoutState
    from src.oviv2.temporal_config import ExecutionProfile

    if not isinstance(profile, ExecutionProfile):
        raise TypeError("profile must be an ExecutionProfile")
    if profile is ExecutionProfile.A0:
        if not isinstance(cumulative, MapSnapshot):
            raise TypeError("A0 neutral composition requires cumulative snapshot")
        return cumulative
    if profile is ExecutionProfile.A1:
        if not isinstance(cumulative, MapSnapshot):
            raise TypeError("A1 neutral composition requires cumulative snapshot")
        if not isinstance(reference_state, ReferenceReadoutState):
            raise TypeError("A1 neutral composition requires reference state")
        lifecycles = dict(reference_state.entity_lifecycles)
        kept = []
        for entity in cumulative.entities:
            owner_id = entity.metadata.get("owner_entity_id")
            if type(owner_id) is not int or owner_id not in lifecycles:
                raise ValueError("cumulative neutral entity has no reference lifecycle")
            if lifecycles[owner_id] != "dormant":
                kept.append(entity)
        return MapSnapshot(
            method=cumulative.method,
            scene_id=cumulative.scene_id,
            timestamp=cumulative.timestamp,
            entities=kept,
            background_xyz=cumulative.background_xyz,
            scope=cumulative.scope,
            runtime=cumulative.runtime,
        )
    if not isinstance(temporal, MapSnapshot):
        raise TypeError("temporal neutral composition requires a temporal snapshot")
    if type(expected_timestamp_ns) is not int or expected_timestamp_ns <= 0:
        raise ValueError("temporal neutral composition requires expected_timestamp_ns")
    if temporal.timestamp != expected_timestamp_ns / 1_000_000_000:
        raise ValueError("temporal neutral snapshot does not match checkpoint timestamp")
    background_xyz = temporal.background_xyz
    if profile is ExecutionProfile.A2:
        if not isinstance(cumulative, MapSnapshot):
            raise TypeError("A2 neutral composition requires cumulative snapshot")
        if (
            temporal.scene_id != cumulative.scene_id
            or cumulative.timestamp != float(expected_timestamp_ns)
            or temporal.scope != cumulative.scope
        ):
            raise ValueError("A2 temporal and cumulative neutral snapshots do not match")
        background_xyz = cumulative.background_xyz
    return MapSnapshot(
        method=temporal.method,
        scene_id=temporal.scene_id,
        timestamp=float(expected_timestamp_ns),
        entities=temporal.entities,
        background_xyz=background_xyz,
        scope=temporal.scope,
        runtime=temporal.runtime,
    )


def _final_current_neutral_from_runtime(
    runtime: Any,
    *,
    checkpoint: TesseCausalCheckpoint,
    caches: Any,
    profile: Any,
    scene: str,
) -> MapSnapshot:
    from src.oviv2.temporal_config import ExecutionProfile

    checkpoint_api = getattr(runtime, "current_checkpoint", None)
    if not callable(checkpoint_api):
        raise TypeError("dual runtime does not expose current_checkpoint")
    current = checkpoint_api()
    expected_timestamp = checkpoint.timestamp_ns / 1_000_000_000
    reference_state = None
    temporal_neutral = None
    if profile in {ExecutionProfile.A0, ExecutionProfile.A1}:
        if type(current) is not CumulativeReadoutView:
            raise TypeError("reference profile final checkpoint must be cumulative")
        reference_state = getattr(getattr(runtime, "temporal", None), "state", None)
        if not (
            reference_state is not None
            and reference_state.scene_id == current.scene_id == scene
            and reference_state.last_frame_id == checkpoint.frame_index
            and reference_state.revision == checkpoint.frame_index + 1
            and float(reference_state.last_timestamp) == expected_timestamp
            and current.last_frame_id == checkpoint.frame_index
            and current.revision == checkpoint.frame_index + 1
            and float(current.last_timestamp) == expected_timestamp
        ):
            raise ValueError("reference final checkpoint does not match run end")
    else:
        if type(current) is not TemporalCurrentSnapshot:
            raise TypeError("temporal profile final checkpoint must be temporal")
        metadata = current.metadata
        if not (
            metadata.scene_id == scene
            and metadata.frame_id == checkpoint.frame_index
            and metadata.revision == checkpoint.frame_index + 1
            and float(metadata.timestamp) == expected_timestamp
        ):
            raise ValueError("temporal final checkpoint does not match run end")
        temporal_neutral = build_temporal_map_snapshot(current, caches.class_names)
    cumulative = (
        _cumulative_neutral_from_runtime(
            runtime, checkpoint=checkpoint, caches=caches
        )
        if profile in {ExecutionProfile.A0, ExecutionProfile.A1, ExecutionProfile.A2}
        else None
    )
    return _compose_checkpoint_neutral(
        profile,
        cumulative=cumulative,
        reference_state=reference_state,
        temporal=temporal_neutral,
        expected_timestamp_ns=checkpoint.timestamp_ns,
    )


def _input_sha256(
    source_config_bytes: bytes,
    schedule_bytes: bytes,
    target_bytes: bytes,
    cache_bindings: Mapping[str, Any],
) -> str:
    return _json_hash(
        {
            "config": _byte_record(source_config_bytes),
            "schedule": _byte_record(schedule_bytes),
            "target": _byte_record(target_bytes),
            "cache_bindings": dict(cache_bindings),
        }
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(
        (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


_PHASE_EVENT_KEYS = {
    "schema_version",
    "sequence",
    "monotonic_ns",
    "phase",
    "frame_index",
    "run_sha256",
    "config_sha256",
    "input_sha256",
    "process_id",
    "process_group_id",
    "run_manifest_sha256",
    "previous_event_sha256",
    "event_sha256",
}


def _validate_phase_events(
    events: Sequence[Mapping[str, Any]],
    *,
    run_sha256: str,
    config_sha256: str,
    input_sha256: str,
    processed_frame_count: int,
    run_manifest_sha256: str,
) -> None:
    expected_phases = [
        "initialization",
        *("frame_processed" for _ in range(processed_frame_count)),
        "finalization",
        "artifact_publication",
    ]
    expected_frames = [None, *range(processed_frame_count), None, None]
    if len(events) != len(expected_phases):
        raise ValueError("phase event coverage is invalid")
    if not all(
        _is_sha256(value)
        for value in (
            run_sha256,
            config_sha256,
            input_sha256,
            run_manifest_sha256,
        )
    ):
        raise ValueError("phase event binding is invalid")
    expected_manifest_hashes = [
        *(None for _ in range(processed_frame_count + 1)),
        run_manifest_sha256,
        run_manifest_sha256,
    ]
    previous_hash: str | None = None
    previous_clock: int | None = None
    process_id = events[0].get("process_id") if events else None
    process_group_id = events[0].get("process_group_id") if events else None
    for sequence, (event, phase, frame_index, manifest_sha256) in enumerate(
        zip(
            events,
            expected_phases,
            expected_frames,
            expected_manifest_hashes,
            strict=True,
        )
    ):
        if not isinstance(event, Mapping) or set(event) != _PHASE_EVENT_KEYS:
            raise ValueError("phase event schema is invalid")
        clock = event.get("monotonic_ns")
        if (
            event.get("schema_version") != 1
            or isinstance(event.get("sequence"), bool)
            or event.get("sequence") != sequence
            or isinstance(clock, bool)
            or not isinstance(clock, Integral)
            or int(clock) < 0
            or (previous_clock is not None and int(clock) <= previous_clock)
            or event.get("phase") != phase
            or event.get("frame_index") != frame_index
        ):
            raise ValueError("phase event order is invalid")
        if not (
            event.get("run_sha256") == run_sha256
            and event.get("config_sha256") == config_sha256
            and event.get("input_sha256") == input_sha256
            and isinstance(process_id, int)
            and not isinstance(process_id, bool)
            and process_id > 0
            and event.get("process_id") == process_id
            and isinstance(process_group_id, int)
            and not isinstance(process_group_id, bool)
            and process_group_id > 0
            and event.get("process_group_id") == process_group_id
            and event.get("run_manifest_sha256") == manifest_sha256
            and event.get("previous_event_sha256") == previous_hash
        ):
            raise ValueError("phase event binding is invalid")
        body = dict(event)
        claimed = body.pop("event_sha256")
        if not _is_sha256(claimed) or claimed != _json_hash(body):
            raise ValueError("phase event hash chain is invalid")
        previous_hash = str(claimed)
        previous_clock = int(clock)


@dataclass
class _PhaseEventJournal:
    run_sha256: str
    config_sha256: str
    input_sha256: str
    process_id: int
    process_group_id: int
    events: list[dict[str, Any]]
    last_monotonic_ns: int | None = None

    def emit(
        self,
        phase: str,
        *,
        frame_index: int | None = None,
        run_manifest_sha256: str | None = None,
    ) -> None:
        current = time.monotonic_ns()
        if self.last_monotonic_ns is not None:
            current = max(current, self.last_monotonic_ns + 1)
        body = {
            "schema_version": 1,
            "sequence": len(self.events),
            "monotonic_ns": current,
            "phase": phase,
            "frame_index": frame_index,
            "run_sha256": self.run_sha256,
            "config_sha256": self.config_sha256,
            "input_sha256": self.input_sha256,
            "process_id": self.process_id,
            "process_group_id": self.process_group_id,
            "run_manifest_sha256": run_manifest_sha256,
            "previous_event_sha256": (
                self.events[-1]["event_sha256"] if self.events else None
            ),
        }
        self.events.append({**body, "event_sha256": _json_hash(body)})
        self.last_monotonic_ns = current


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    content = b"".join(_canonical_json_bytes(dict(row)) for row in rows)
    directory_fd = _open_directory_nofollow(path.parent)
    try:
        _atomic_write_new_at(directory_fd, path.name, content)
    finally:
        os.close(directory_fd)


def _runtime_diagnostics_payload(
    runtime: Any,
    *,
    config: Mapping[str, Any],
    processed_frame_count: int,
    mechanism_records: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    from src.oviv2.temporal_config import temporal_config_from_json

    temporal_readout = config.get("temporal_readout")
    if not isinstance(temporal_readout, Mapping):
        raise ValueError("runtime diagnostics execution profile is invalid")
    parsed = temporal_config_from_json(
        {"temporal_readout": dict(temporal_readout)}
    )
    profile = parsed.execution_profile.profile_id
    temporal = getattr(runtime, "temporal", None)
    state = getattr(temporal, "state", None)
    diagnostics = getattr(state, "diagnostics", None)
    counters: dict[str, int] = {}
    for name in V2_RUNTIME_DIAGNOSTIC_KEYS:
        value = getattr(diagnostics, name, 0) if diagnostics is not None else 0
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"runtime diagnostic {name} must be a nonnegative integer")
        counters[name] = int(value)
    diagnostic_frames = getattr(diagnostics, "processed_frame_count", processed_frame_count)
    if diagnostic_frames != processed_frame_count:
        raise ValueError("runtime diagnostics frame count differs from the run")
    supplied = (
        {name: [] for name in V2_RUNTIME_DIAGNOSTIC_KEYS}
        if mechanism_records is None
        else mechanism_records
    )
    if not isinstance(supplied, Mapping) or set(supplied) != set(V2_RUNTIME_DIAGNOSTIC_KEYS):
        raise ValueError("runtime mechanism records inventory is invalid")
    serialized_records: dict[str, list[str]] = {}
    for name in V2_RUNTIME_DIAGNOSTIC_KEYS:
        values = supplied[name]
        if (
            not isinstance(values, (tuple, list))
            or len(values) != counters[name]
            or len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError(f"runtime mechanism records do not match counter: {name}")
        serialized_records[name] = list(values)
    subset_pairs = (
        ("proposal_trigger_count", "proposal_opportunity_count"),
        ("reid_trigger_count", "reid_opportunity_count"),
        ("epoch_reset_trigger_count", "epoch_reset_opportunity_count"),
        ("icp_accept_count", "icp_opportunity_count"),
        ("icp_reject_count", "icp_opportunity_count"),
        ("ledger_commit_count", "ledger_stage_count"),
        ("ledger_reclaim_count", "ledger_commit_count"),
    )
    if any(
        not set(serialized_records[child]) <= set(serialized_records[parent])
        for child, parent in subset_pairs
    ):
        raise ValueError("runtime mechanism records relation mismatch")
    icp_opportunities = set(serialized_records["icp_opportunity_count"])
    icp_accepts = set(serialized_records["icp_accept_count"])
    icp_rejects = set(serialized_records["icp_reject_count"])
    if icp_accepts & icp_rejects or icp_accepts | icp_rejects != icp_opportunities:
        raise ValueError("runtime mechanism records ICP partition mismatch")
    controls = temporal_readout.get("diagnostic_controls")
    icp_enabled = profile == "a4" and controls != {"icp_enabled": False}
    if icp_enabled and not set(serialized_records["motion_rejection_count"]) <= icp_opportunities:
        raise ValueError("runtime mechanism records relation mismatch")
    payload = {
        "schema_version": 1,
        "execution_profile": profile,
        "processed_frame_count": processed_frame_count,
        "counters": counters,
        "mechanism_records": serialized_records,
        "diagnostic": canonical_diagnostic_claim(parsed),
    }
    validate_runtime_diagnostics(
        payload,
        label="runtime diagnostics",
        expected_temporal_readout=temporal_readout,
        expected_candidate_id=parsed.diagnostic_identity or profile,
        expected_processed_frame_count=processed_frame_count,
    )
    return payload


def _accumulate_runtime_mechanism_records(
    accumulated: dict[str, list[str]],
    frame_result: Any,
    *,
    seen_by_name: dict[str, set[str]],
) -> None:
    names = set(V2_RUNTIME_DIAGNOSTIC_KEYS)
    if (
        set(accumulated) != names
        or set(seen_by_name) != names
        or any(type(seen_by_name[name]) is not set for name in names)
    ):
        raise ValueError("runtime mechanism accumulator inventory is invalid")
    diagnostics = getattr(frame_result, "diagnostics", None)
    if diagnostics is None:
        diagnostics = getattr(getattr(frame_result, "temporal", None), "diagnostics", None)
    raw = getattr(diagnostics, "mechanism_records", None)
    if raw is None:
        return
    try:
        records = dict(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("frame mechanism records are invalid") from error
    if set(records) != names:
        raise ValueError("frame mechanism records inventory is invalid")
    for name in V2_RUNTIME_DIAGNOSTIC_KEYS:
        values = records[name]
        if type(values) is not tuple or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"frame mechanism records are not unique: {name}")
        value_set = set(values)
        if len(values) != len(value_set) or value_set & seen_by_name[name]:
            raise ValueError(f"frame mechanism records are not unique: {name}")
        accumulated[name].extend(values)
        seen_by_name[name].update(value_set)


def _office_attempt_root(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.attempts"


def _office_claim_root(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.claim"


def _phase_events_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.phase_events.jsonl"


def _open_directory_nofollow(path: Path) -> int:
    absolute = _absolute_lexical(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(os.sep, flags)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file_at(directory_fd: int, name: str, *, role: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{role} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


@dataclass
class _OfficeClaim:
    path: Path
    parent_fd: int
    root_fd: int
    content: bytes


def _verify_directory_entry(parent_fd: int, name: str, directory_fd: int) -> None:
    expected = os.fstat(directory_fd)
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        raise ValueError(f"trusted directory entry changed: {name}")


def _verify_directory_path(path: Path, directory_fd: int) -> None:
    current_fd = _open_directory_nofollow(path)
    try:
        expected = os.fstat(directory_fd)
        current = os.fstat(current_fd)
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError(f"trusted directory path changed: {path}")
    finally:
        os.close(current_fd)


def _office_attempt_binding(frozen: FrozenRunContext, destination: Path) -> dict[str, Any]:
    authorization = frozen.frozen_run_identity.get("office_authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("Office requires frozen authorization")
    return {
        "run_slot": frozen.run_slot,
        "frozen_run_identity_sha256": _json_hash(frozen.frozen_run_identity),
        "authorization_sha256": authorization["authorization_sha256"],
        "config_sha256": frozen.frozen_run_identity["config"]["sha256"],
        "output_root": str(destination),
    }


def _next_office_attempt(
    frozen: FrozenRunContext, destination: Path, *, parent_fd: int
) -> int:
    root_fd: int | None = None
    root_name = f".{destination.name}.attempts"
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        try:
            root_fd = os.open(root_name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            return 1
        except OSError as exc:
            raise ValueError("Office attempt ledger must be a real directory") from exc
        expected_binding = _office_attempt_binding(frozen, destination)
        records = sorted(os.listdir(root_fd))
        expected_names = [f"attempt_{index:04d}.json" for index in range(1, len(records) + 1)]
        if records != expected_names:
            raise ValueError("Office attempt ledger inventory is invalid")
        for index, name in enumerate(records, 1):
            receipt = _load_json_bytes(
                _read_regular_file_at(root_fd, name, role="Office failure receipt"),
                _office_attempt_root(destination) / name,
            )
            if not isinstance(receipt, Mapping) or set(receipt) != {
                "schema_version", "status", "attempt", *expected_binding,
                "staging_published", "metric_artifact_present", "failure", "receipt_sha256",
            }:
                raise ValueError("Office failure receipt schema is invalid")
            body = dict(receipt)
            claimed = body.pop("receipt_sha256")
            if not (
                receipt.get("schema_version") == 1
                and receipt.get("status") == "INFRASTRUCTURE_FAILURE"
                and receipt.get("attempt") == index
                and all(receipt.get(key) == value for key, value in expected_binding.items())
                and receipt.get("staging_published") is False
                and receipt.get("metric_artifact_present") is False
                and isinstance(receipt.get("failure"), Mapping)
                and claimed == _json_hash(body)
            ):
                raise ValueError("Office retry receipt does not match frozen hashes")
        _verify_directory_entry(parent_fd, root_name, root_fd)
        return len(records) + 1
    finally:
        if root_fd is not None:
            os.close(root_fd)


def _atomic_write_new_at(directory_fd: int, name: str, content: bytes) -> None:
    temporary = f".{name}.tmp-{secrets.token_hex(12)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write while publishing {name}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _claim_office_authorization(
    frozen: FrozenRunContext, destination: Path, *, attempt: int, parent_fd: int
) -> _OfficeClaim:
    root = _office_claim_root(destination)
    root_name = root.name
    try:
        os.mkdir(root_name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        raise FileExistsError(f"Office authorization is already claimed: {root}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    root_fd: int | None = None
    try:
        root_fd = os.open(root_name, flags, dir_fd=parent_fd)
        body = {
            "schema_version": 1,
            "status": "CLAIMED",
            "attempt": attempt,
            **_office_attempt_binding(frozen, destination),
        }
        content = _canonical_json_bytes({**body, "claim_sha256": _json_hash(body)})
        _atomic_write_new_at(
            root_fd,
            "claim.json",
            content,
        )
        _verify_directory_entry(parent_fd, root_name, root_fd)
        _verify_directory_path(destination.parent, parent_fd)
        return _OfficeClaim(
            path=root, parent_fd=parent_fd, root_fd=root_fd, content=content
        )
    except BaseException:
        if root_fd is not None:
            os.close(root_fd)
        raise


def _release_office_claim(claim: _OfficeClaim) -> None:
    try:
        _verify_directory_path(claim.path.parent, claim.parent_fd)
        _verify_directory_entry(claim.parent_fd, claim.path.name, claim.root_fd)
        if _read_regular_file_at(
            claim.root_fd, "claim.json", role="Office authorization claim"
        ) != claim.content:
            raise ValueError("Office authorization claim changed during execution")
        os.unlink("claim.json", dir_fd=claim.root_fd)
        os.fsync(claim.root_fd)
        _verify_directory_entry(claim.parent_fd, claim.path.name, claim.root_fd)
        os.rmdir(claim.path.name, dir_fd=claim.parent_fd)
        os.fsync(claim.parent_fd)
    finally:
        os.close(claim.root_fd)
        os.close(claim.parent_fd)


def _close_office_claim(claim: _OfficeClaim) -> None:
    try:
        _verify_directory_path(claim.path.parent, claim.parent_fd)
    finally:
        os.close(claim.root_fd)
        os.close(claim.parent_fd)


def _apply_office_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("Office authorization seed is invalid")
    normalized = int(seed)
    random.seed(normalized)
    np.random.seed(normalized)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(normalized)
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "manual_seed_all"):
            cuda.manual_seed_all(normalized)
    return normalized


def _publish_office_failure_receipt(
    frozen: FrozenRunContext,
    destination: Path,
    *,
    attempt: int,
    failure: BaseException,
    staging: Path,
    claim: _OfficeClaim,
) -> bool:
    metric_artifact_present = any(
        any(token in path.name.lower() for token in ("metric", "evaluation", "summary"))
        for path in staging.rglob("*")
        if path.is_file()
    )
    staging_is_empty = not any(staging.iterdir())
    root = _office_attempt_root(destination)
    retryable = (
        isinstance(failure, OSError)
        and staging_is_empty
        and not metric_artifact_present
    )
    body = {
        "schema_version": 1,
        "status": "INFRASTRUCTURE_FAILURE" if retryable else "NON_RETRYABLE_FAILURE",
        "attempt": attempt,
        **_office_attempt_binding(frozen, destination),
        "staging_published": False,
        "metric_artifact_present": metric_artifact_present,
        "failure": {"type": type(failure).__name__, "message": str(failure)},
    }
    parent_fd = claim.parent_fd
    _verify_directory_path(destination.parent, parent_fd)
    root_fd: int | None = None
    root_name = root.name
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        try:
            os.mkdir(root_name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        root_fd = os.open(root_name, flags, dir_fd=parent_fd)
        _atomic_write_new_at(
            root_fd,
            f"attempt_{attempt:04d}.json",
            _canonical_json_bytes({**body, "receipt_sha256": _json_hash(body)}),
        )
        _verify_directory_entry(parent_fd, root_name, root_fd)
    finally:
        if root_fd is not None:
            os.close(root_fd)
    return retryable


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    freeze_manifest: str | Path | None = None,
    run_slot: str | None = None,
    allow_unfrozen_office: bool = False,
    table_only: bool = False,
    temporal_metrics_only: bool = False,
    dependencies: RunnerDependencies | None = None,
) -> dict[str, Any]:
    if (freeze_manifest is None) != (run_slot is None):
        raise ValueError("freeze_manifest and run_slot must be provided together")
    if table_only and temporal_metrics_only:
        raise ValueError("table-only and temporal-metrics-only are mutually exclusive")
    source_config = Path(config_path).absolute()
    _require_regular_file(source_config, "runner config")
    source_config_bytes = source_config.read_bytes()
    config = _load_json_bytes(source_config_bytes, source_config)
    scene, frame_count, schedule_path, evaluation_frames, temporal_config = (
        _validate_config(config)
    )
    if temporal_metrics_only and temporal_config.execution_profile.profile_id not in {
        "a3",
        "a4",
    }:
        raise ValueError("temporal metrics mode requires profile a3 or a4")
    if temporal_metrics_only and scene != "apartment":
        raise ValueError("temporal metrics mode is restricted to Apartment development")
    if allow_unfrozen_office and (scene != "office" or freeze_manifest is not None):
        raise ValueError("unfrozen Office opt-in is only valid for an unfrozen Office run")
    if scene == "office" and freeze_manifest is None and not allow_unfrozen_office:
        raise ValueError("Office requires frozen authorization")
    destination = _absolute_lexical(output)
    frozen = (
        load_v2_frozen_run_context(
            freeze_manifest,
            run_slot=str(run_slot),
            source_config=source_config,
            source_config_bytes=source_config_bytes,
            config=config,
            destination=destination,
        )
        if freeze_manifest is not None
        else None
    )

    _require_regular_file(schedule_path, "causal schedule")
    schedule_bytes = schedule_path.read_bytes()
    official = _load_causal_checkpoints_bytes(
        schedule_bytes, schedule_path, scene=scene, frame_count=frame_count
    )
    target_path = _resolve_path(config.get("occlusion_target_manifest", ""))
    target_hash = config.get("occlusion_target_manifest_sha256")
    plan_hash = config.get("evaluation_checkpoint_frames_sha256")
    if not _is_sha256(target_hash) or not _is_sha256(plan_hash):
        raise ValueError("occlusion target bindings are invalid")
    _require_regular_file(target_path, "occlusion target manifest")
    target_bytes = target_path.read_bytes()
    if _sha256_bytes(target_bytes) != target_hash:
        raise ValueError("occlusion target manifest checksum binding mismatch")
    plan = _checkpoint_plan_from_target_manifest(target_bytes, target_path)
    if (
        list(evaluation_frames) != plan["evaluation_checkpoint_frames"][scene]
        or plan_hash != plan["evaluation_checkpoint_frames_sha256"]
    ):
        raise ValueError("evaluation checkpoint frame binding mismatch")
    capture_evaluation_frames = () if table_only else evaluation_frames

    dependencies = _production_dependencies() if dependencies is None else dependencies
    current_environment = _validate_environment(
        dict(dependencies.environment_factory())
    )
    if (
        frozen is not None
        and _formal_environment_compatibility(current_environment)
        != _formal_environment_compatibility(
            frozen.input_bindings["environment"]
        )
    ):
        raise ValueError("frozen environment is incompatible with the execution environment")

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    phase_events_output = _phase_events_path(destination)
    if phase_events_output.exists() or phase_events_output.is_symlink():
        raise FileExistsError(phase_events_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, "output parent")
    office_parent_fd = (
        _open_directory_nofollow(destination.parent)
        if scene == "office" and frozen is not None
        else None
    )
    try:
        office_attempt = (
            _next_office_attempt(
                frozen, destination, parent_fd=office_parent_fd
            )
            if office_parent_fd is not None and frozen is not None
            else None
        )
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-", dir=destination.parent
            )
        )
    except BaseException:
        if office_parent_fd is not None:
            os.close(office_parent_fd)
        raise
    staging_identity = _staging_identity(staging)
    try:
        office_claim = (
            _claim_office_authorization(
                frozen,
                destination,
                attempt=office_attempt,
                parent_fd=office_parent_fd,
            )
            if office_attempt is not None and frozen is not None
            else None
        )
    except BaseException:
        shutil.rmtree(staging)
        if office_parent_fd is not None:
            os.close(office_parent_fd)
        raise
    published = False
    execution = (
        _run_execution(run_slot=frozen.run_slot, output=destination, staging=staging)
        if frozen is not None
        else None
    )
    try:
        if scene == "office" and frozen is not None:
            authorization = frozen.frozen_run_identity.get("office_authorization")
            if not isinstance(authorization, Mapping):
                raise ValueError("Office requires frozen authorization")
            _apply_office_seed(authorization.get("seed"))
        provenance = dict(dependencies.provenance_factory())
        code_commit = provenance.get("repository_commit")
        if not isinstance(code_commit, str) or len(code_commit) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in code_commit
        ):
            raise ValueError("provenance repository_commit is invalid")
        dataset_config = {key: config[key] for key in _DATASET_CONFIG_KEYS}
        dataset = dependencies.dataset_factory(dataset_config)
        if len(dataset) != frame_count:
            raise ValueError("dataset frame count does not match runner config")
        cache_config = {key: config[key] for key in _CACHE_CONFIG_KEYS}
        caches = dependencies.cache_loader_factory(cache_config, dataset)
        cache_bindings = getattr(caches, "bindings", {})
        if not isinstance(cache_bindings, Mapping):
            raise ValueError("cache bindings must be a mapping")
        try:
            encoded_bindings = json.dumps(
                dict(cache_bindings),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            canonical_source_bindings = json.loads(encoded_bindings)
        except (TypeError, ValueError) as exc:
            raise ValueError("cache bindings must be strictly JSON serializable") from exc
        if not (
            isinstance(canonical_source_bindings, dict)
            and canonical_source_bindings == dict(cache_bindings)
        ):
            raise ValueError("cache bindings must use canonical JSON values")
        input_sha256 = _input_sha256(
            source_config_bytes,
            schedule_bytes,
            target_bytes,
            canonical_source_bindings,
        )
        config_sha256 = _sha256_bytes(source_config_bytes)
        process_id = os.getpid()
        process_group_id = os.getpgid(0)
        run_sha256 = _json_hash(
            {
                "protocol_id": PROTOCOL_ID,
                "dataset": "TESSE-CD",
                "method_id": "OVIV2",
                "scene": scene,
                "config_sha256": config_sha256,
                "input_sha256": input_sha256,
                "code_commit": code_commit,
                "output_root": str(destination),
                "run_slot": run_slot,
                "process_id": process_id,
                "process_group_id": process_group_id,
            }
        )
        phase_events = _PhaseEventJournal(
            run_sha256=run_sha256,
            config_sha256=config_sha256,
            input_sha256=input_sha256,
            process_id=process_id,
            process_group_id=process_group_id,
            events=[],
        )
        runtime_config = {key: config[key] for key in _RUNTIME_CONFIG_KEYS}
        if scene == "office" and frozen is not None:
            _apply_office_seed(
                frozen.frozen_run_identity["office_authorization"]["seed"]
            )
        runtime = dependencies.runtime_factory(runtime_config, caches)
        phase_events.emit("initialization")

        by_frame = {item.frame_index: item for item in official}
        first_timestamp_ns = _dataset_timestamp_ns(dataset, 0)
        for frame_index in capture_evaluation_frames:
            timestamp_ns = _dataset_timestamp_ns(dataset, frame_index)
            existing = by_frame.get(frame_index)
            if existing is None:
                by_frame[frame_index] = TesseCausalCheckpoint(
                    frame_index,
                    timestamp_ns,
                    timestamp_ns - first_timestamp_ns,
                    (),
                    ("occlusion_v1",),
                )
            elif "occlusion_v1" not in existing.roles:
                by_frame[frame_index] = TesseCausalCheckpoint(
                    existing.frame_index,
                    existing.timestamp_ns,
                    existing.relative_timestamp_ns,
                    existing.event_ids,
                    (*existing.roles, "occlusion_v1"),
                )
        checkpoints = tuple(by_frame[index] for index in sorted(by_frame))
        captured: list[int] = []
        records: list[dict[str, Any]] = []
        witnesses: list[_CheckpointArtifactWitness] = []
        cumulative_audit_witnesses: list[_CumulativeAuditWitness | None] = []
        cumulative_audit_directories: set[str] = set()
        expected_checkpoint_inventory: set[str] = set()
        official_frames = {item.frame_index for item in official}
        official_sources: dict[int, dict[str, Any]] = {}
        trajectory_rows: list[dict[str, Any]] = []
        lifecycle_rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        export_batches: list[TemporalExportBatch] = []
        runtime_mechanism_records = {
            name: [] for name in V2_RUNTIME_DIAGNOSTIC_KEYS
        }
        runtime_mechanism_seen = {
            name: set() for name in V2_RUNTIME_DIAGNOSTIC_KEYS
        }
        temporal_table_only = (
            table_only
            and temporal_config.execution_profile.profile_id in {"a3", "a4"}
        )
        for frame_index in range(frame_count):
            frame = dataset[frame_index]
            if int(frame.frame_id) != frame_index:
                raise ValueError("dataset frame IDs must equal zero-based frame indices")
            table_keyframe = (
                not temporal_table_only
                or frame_index % TABLE_ONLY_KEYFRAME_STRIDE == 0
                or frame_index in by_frame
                or frame_index == frame_count - 1
            )
            dataset_timestamp_ns = _dataset_timestamp_ns(dataset, frame_index)
            if temporal_table_only and not table_keyframe:
                advance_api = getattr(
                    runtime, "advance_temporal_only_frame", None
                )
                if not callable(advance_api):
                    raise TypeError(
                        "table-only temporal profile requires temporal advance API"
                    )
                frame_result = advance_api(frame)
            else:
                observations, dense_semantics = caches.load(frame_index, frame)
                temporal_observations = _load_temporal_observations(
                    temporal_config.execution_profile,
                    caches,
                    frame_index,
                    frame,
                    dense_semantics,
                )
            if temporal_table_only and table_keyframe:
                temporal_only_api = getattr(
                    runtime, "process_temporal_only_frame", None
                )
                if not callable(temporal_only_api):
                    raise TypeError(
                        "table-only temporal profile requires temporal-only runtime API"
                    )
                if temporal_observations is None:
                    raise ValueError(
                        "table-only temporal profile requires temporal observations"
                    )
                frame_result = temporal_only_api(
                    frame, temporal_observations, dense_semantics
                )
            elif not temporal_table_only and temporal_observations is None:
                frame_result = runtime.process_frame(
                    frame,
                    observations=observations,
                    dense_semantics=dense_semantics,
                )
            elif not temporal_table_only:
                frame_result = runtime.process_frame(
                    frame,
                    observations=observations,
                    dense_semantics=dense_semantics,
                    temporal_observations=temporal_observations,
                )
            _accumulate_runtime_mechanism_records(
                runtime_mechanism_records,
                frame_result,
                seen_by_name=runtime_mechanism_seen,
            )
            export = getattr(frame_result, "export", None)
            if type(export) is not TemporalExportBatch:
                raise TypeError("dual runtime frame result has no temporal export batch")
            if (
                export.frame_index != frame_index
                or export.timestamp_ns != dataset_timestamp_ns
            ):
                raise ValueError("temporal export batch does not match input frame")
            if export_batches:
                export.validate_after(export_batches[-1])
            export_batches.append(export)
            trajectory_rows.extend(sample.to_json_record() for sample in export.samples)
            lifecycle_rows.extend(event.to_json_record() for event in export.events)
            coverage_rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ns": dataset_timestamp_ns,
                    "record_count": len(export.samples),
                    "event_count": len(export.events),
                }
            )
            checkpoint = by_frame.get(frame_index)
            if checkpoint is None:
                phase_events.emit("frame_processed", frame_index=frame_index)
                continue
            if dataset_timestamp_ns != checkpoint.timestamp_ns:
                raise ValueError("checkpoint timestamp does not match dataset timestamp")
            if checkpoint.relative_timestamp_ns != (
                dataset_timestamp_ns - first_timestamp_ns
            ):
                raise ValueError(
                    "checkpoint relative timestamp does not match dataset origin"
                )
            reference_state = None
            snapshot = None
            checkpoint_api = getattr(runtime, "current_checkpoint", None)
            if not callable(checkpoint_api):
                raise TypeError("dual runtime does not expose current_checkpoint")
            current_checkpoint = checkpoint_api()
            expected_timestamp = checkpoint.timestamp_ns / 1_000_000_000
            if temporal_config.execution_profile.profile_id not in {"a0", "a1"}:
                if type(current_checkpoint) is not TemporalCurrentSnapshot:
                    raise TypeError("temporal profile checkpoint must be native temporal state")
                snapshot = current_checkpoint
                metadata = snapshot.metadata
                if not (
                    metadata.scene_id == scene
                    and metadata.frame_id == checkpoint.frame_index
                    and metadata.revision == checkpoint.frame_index + 1
                    and float(metadata.timestamp) == expected_timestamp
                ):
                    raise ValueError("temporal checkpoint does not match checkpoint progress")
            else:
                if type(current_checkpoint) is not CumulativeReadoutView:
                    raise TypeError("reference profile checkpoint must be native cumulative view")
                reference_state = getattr(getattr(runtime, "temporal", None), "state", None)
                if not (
                    reference_state is not None
                    and reference_state.scene_id == scene
                    and reference_state.last_frame_id == checkpoint.frame_index
                    and reference_state.revision == checkpoint.frame_index + 1
                    and float(reference_state.last_timestamp) == expected_timestamp
                ):
                    raise ValueError("reference state does not match checkpoint progress")
                if not (
                    current_checkpoint.scene_id == scene
                    and current_checkpoint.last_frame_id == checkpoint.frame_index
                    and current_checkpoint.revision == checkpoint.frame_index + 1
                    and current_checkpoint.last_timestamp == expected_timestamp
                ):
                    raise ValueError("reference checkpoint does not match checkpoint progress")
                metadata = TemporalSnapshotMetadata(
                    scene_id=current_checkpoint.scene_id,
                    frame_id=current_checkpoint.last_frame_id,
                    timestamp=current_checkpoint.last_timestamp,
                    revision=current_checkpoint.revision,
                    voxel_size_m=temporal_config.geometry.voxel_size_m,
                    config_sha256=str(config["algorithm_hash"]),
                )
            root = staging / "checkpoints" / f"{frame_index:08d}-{checkpoint.timestamp_ns}"
            _assert_staging_identity(staging, staging_identity)
            root.mkdir(parents=True)
            roles = set(checkpoint.roles)
            needs_full = bool({"official", "common_v2"} & roles)
            needs_compact = "occlusion_v1" in roles
            if not needs_full and not needs_compact:
                raise ValueError("checkpoint role combination is invalid")
            artifacts: dict[str, dict[str, Any]] = {}
            cumulative_neutral = None
            cumulative_audit: dict[str, Any] | None = None
            cumulative_audit_witness: _CumulativeAuditWitness | None = None
            skip_cumulative_audit = temporal_metrics_only or (
                table_only
                and temporal_config.execution_profile.profile_id in {"a3", "a4"}
            )
            if not skip_cumulative_audit:
                audit_root = root / "cumulative_audit"
                audit_root.mkdir()
                from src.oviv2.runtime import Oviv2Runtime

                cumulative_runtime = getattr(runtime, "cumulative", None)
                cumulative_snapshot_record = None
                if isinstance(cumulative_runtime, Oviv2Runtime):
                    from src.evaluation.oviv2_tesse import build_neutral_current_snapshot

                    required = (
                        "class_names",
                        "object_semantic_ids",
                        "semantic_fusion",
                        "timestamp_ns_by_frame",
                    )
                    if any(not hasattr(caches, name) for name in required):
                        raise TypeError("cumulative audit requires production cache bindings")
                    cumulative_snapshot = cumulative_runtime.commit_new(
                        audit_root / "voxel_snapshot"
                    )
                    if not (
                        cumulative_snapshot.metadata.frame_id == checkpoint.frame_index
                        and float(cumulative_snapshot.metadata.timestamp)
                        == checkpoint.timestamp_ns / 1_000_000_000
                    ):
                        raise ValueError("cumulative audit snapshot does not match progress")
                    cumulative_snapshot.revalidate_source()
                    cumulative_neutral = build_neutral_current_snapshot(
                        cumulative_snapshot,
                        timestamp_ns=checkpoint.timestamp_ns,
                        class_names=caches.class_names,
                        object_semantic_ids=caches.object_semantic_ids,
                        fusion=caches.semantic_fusion,
                        timestamp_ns_by_frame=caches.timestamp_ns_by_frame,
                    )
                    cumulative_snapshot_record = _tree_record(
                        audit_root / "voxel_snapshot", relative_to=staging
                    )
                    cumulative_snapshot.revalidate_source()
                    del cumulative_snapshot
                else:
                    cumulative_neutral = _cumulative_neutral_from_runtime(
                        runtime, checkpoint=checkpoint, caches=caches
                    )
                audit_paths = write_map_snapshot(
                    cumulative_neutral, audit_root / "artifact"
                )
                audit_snapshot = Path(audit_paths["snapshot"])
                audit_entities = Path(audit_paths["entities"])
                for path in (audit_snapshot, audit_entities):
                    if not path.is_file() or path.is_symlink():
                        raise ValueError("cumulative audit sidecar is invalid")
                    expected_checkpoint_inventory.add(
                        path.relative_to(staging).as_posix()
                    )
                for path in audit_root.rglob("*"):
                    relative = path.relative_to(staging).as_posix()
                    if path.is_file():
                        expected_checkpoint_inventory.add(relative)
                    elif path.is_dir():
                        cumulative_audit_directories.add(relative)
                cumulative_audit_directories.add(
                    audit_root.relative_to(staging).as_posix()
                )
                cumulative_audit = {
                    "format": "oviv2_cumulative_audit_v1",
                    "artifact": _tree_record(
                        audit_root / "artifact", relative_to=staging
                    ),
                    "snapshot": _file_record(audit_snapshot, relative_to=staging),
                    "entities": _file_record(audit_entities, relative_to=staging),
                }
                if cumulative_snapshot_record is not None:
                    cumulative_audit["voxel_snapshot"] = cumulative_snapshot_record
                cumulative_audit_witness = _CumulativeAuditWitness.bind(
                    cumulative_audit
                )
                _revalidate_cumulative_audit_witness(
                    staging, cumulative_audit_witness, cumulative_audit
                )
            cumulative_audit_witnesses.append(cumulative_audit_witness)
            temporal_neutral = None
            if needs_full:
                if snapshot is not None:
                    if table_only:
                        direct_temporal = build_temporal_map_snapshot(
                            snapshot, caches.class_names
                        )
                        temporal_neutral = MapSnapshot(
                            method="OVIV2",
                            scene_id=direct_temporal.scene_id,
                            timestamp=direct_temporal.timestamp,
                            entities=direct_temporal.entities,
                            background_xyz=direct_temporal.background_xyz,
                            scope=direct_temporal.scope,
                            runtime={},
                        )
                    else:
                        receipt = publish_temporal_current_checkpoint(
                            root / "temporal_current",
                            snapshot,
                            caches.class_names,
                            code_commit=code_commit,
                            input_sha256=input_sha256,
                        )
                        artifacts["temporal_current"] = _bind_checkpoint_artifact(
                            artifact_root=receipt.path,
                            source_witness=receipt.source_witness,
                            checkpoint_format=TEMPORAL_CURRENT_FORMAT,
                            staging=staging,
                            temporal_config=temporal_config,
                            witnesses=witnesses,
                            expected_inventory=expected_checkpoint_inventory,
                        )
                        loaded = load_temporal_current_checkpoint(receipt.path)
                        temporal_neutral = MapSnapshot(
                            method="OVIV2",
                            scene_id=loaded.snapshot.scene_id,
                            timestamp=loaded.snapshot.timestamp,
                            entities=loaded.snapshot.entities,
                            background_xyz=loaded.snapshot.background_xyz,
                            scope=loaded.snapshot.scope,
                            runtime={},
                        )
                        loaded.revalidate_source()
                neutral = _compose_checkpoint_neutral(
                    temporal_config.execution_profile,
                    cumulative=(
                        cumulative_neutral
                        if temporal_config.execution_profile.profile_id
                        in {"a0", "a1", "a2"}
                        else None
                    ),
                    reference_state=reference_state,
                    temporal=temporal_neutral,
                    expected_timestamp_ns=checkpoint.timestamp_ns,
                )
                neutral_paths = write_map_snapshot(neutral, root / "neutral_current")
                neutral_snapshot = Path(neutral_paths["snapshot"])
                neutral_entities = Path(neutral_paths["entities"])
                for path in (neutral_snapshot, neutral_entities):
                    if not path.is_file() or path.is_symlink():
                        raise ValueError("neutral checkpoint sidecar is invalid")
                    expected_checkpoint_inventory.add(
                        path.relative_to(staging).as_posix()
                    )
                if snapshot is None or table_only:
                    neutral_tree = _tree_record(
                        root / "neutral_current", relative_to=staging
                    )
                    artifacts["neutral_current"] = {
                        "format": "oviv2_neutral_current_v1",
                        "artifact": neutral_tree,
                        "checksums_sha256": neutral_tree["sha256"],
                    }
                elif not table_only:
                    del receipt
            if needs_compact:
                compact_checkpoint = (
                    TemporalCompactCheckpoint.from_reference(
                        metadata,
                        reference_state,
                        maximum_entities=temporal_config.geometry.maximum_entities,
                        maximum_object_voxels=temporal_config.geometry.maximum_object_voxels,
                    )
                    if reference_state is not None
                    else TemporalCompactCheckpoint.from_snapshot(
                        snapshot,
                        maximum_entities=temporal_config.geometry.maximum_entities,
                        maximum_object_voxels=temporal_config.geometry.maximum_object_voxels,
                    )
                )
                compact = compact_checkpoint.commit_new(
                    root / "temporal_compact",
                    maximum_entities=temporal_config.geometry.maximum_entities,
                    maximum_object_voxels=temporal_config.geometry.maximum_object_voxels,
                )
                artifacts["temporal_compact"] = _bind_checkpoint_artifact(
                    artifact_root=compact.path,
                    source_witness=compact,
                    checkpoint_format=TEMPORAL_COMPACT_FORMAT,
                    staging=staging,
                    temporal_config=temporal_config,
                    witnesses=witnesses,
                    expected_inventory=expected_checkpoint_inventory,
                )
                del compact
            primary = artifacts[
                (
                    "temporal_current"
                    if "temporal_current" in artifacts
                    else "neutral_current"
                    if needs_full
                    else "temporal_compact"
                )
            ]
            status_path = root / "checkpoint_status.json"
            _write_json(
                status_path,
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "checkpoint_frame": checkpoint.frame_index,
                    "timestamp_ns": checkpoint.timestamp_ns,
                    "event_ids": list(checkpoint.event_ids),
                    "roles": list(checkpoint.roles),
                    "consumed_through_frame": checkpoint.frame_index,
                    "consumed_through_frame_exclusive": checkpoint.frame_index + 1,
                },
            )
            expected_checkpoint_inventory.add(
                status_path.relative_to(staging).as_posix()
            )
            record = {
                "scene": scene,
                "frame_index": checkpoint.frame_index,
                "timestamp_ns": checkpoint.timestamp_ns,
                "relative_timestamp_ns": checkpoint.relative_timestamp_ns,
                "consumed_through_frame": checkpoint.frame_index,
                "consumed_through_frame_exclusive": checkpoint.frame_index + 1,
                "event_ids": list(checkpoint.event_ids),
                "roles": list(checkpoint.roles),
                "format": primary["format"],
                "artifact": primary["artifact"],
                "checksums_sha256": primary["checksums_sha256"],
                "artifacts": artifacts,
                "cumulative_audit": cumulative_audit,
                "checkpoint_status": _file_record(status_path, relative_to=staging),
            }
            if needs_full:
                record.update(
                    {
                        "neutral_snapshot": _file_record(
                            neutral_snapshot, relative_to=staging
                        ),
                        "neutral_entities": _file_record(
                            neutral_entities, relative_to=staging
                        ),
                    }
                )
            records.append(record)
            if frame_index in official_frames:
                official_sources[frame_index] = {
                    "frame_index": checkpoint.frame_index,
                    "timestamp_ns": checkpoint.timestamp_ns,
                    "consumed_through_frame": checkpoint.frame_index,
                    "consumed_through_frame_exclusive": checkpoint.frame_index + 1,
                    "checkpoint_status": record["checkpoint_status"],
                    "snapshot": record["neutral_snapshot"],
                    "entities": record["neutral_entities"],
                }
            captured.append(frame_index)
            _assert_staging_identity(staging, staging_identity)
            phase_events.emit("frame_processed", frame_index=frame_index)

        scheduled = [item.frame_index for item in checkpoints]
        if captured != scheduled:
            raise ValueError("captured checkpoints do not exactly match the schedule")
        validate_temporal_export_sequence(tuple(export_batches))
        if (
            len(export_batches) != frame_count
            or [row["frame_index"] for row in coverage_rows]
            != list(range(frame_count))
            or [row["timestamp_ns"] for row in coverage_rows]
            != [_dataset_timestamp_ns(dataset, index) for index in range(frame_count)]
        ):
            raise ValueError("temporal frame coverage is not exact")
        final_frame_index = coverage_rows[-1]["frame_index"]
        final_timestamp_ns = coverage_rows[-1]["timestamp_ns"]
        final_checkpoint = TesseCausalCheckpoint(
            final_frame_index,
            final_timestamp_ns,
            final_timestamp_ns - first_timestamp_ns,
            (),
            ("run_end",),
        )
        final_neutral = _final_current_neutral_from_runtime(
            runtime,
            checkpoint=final_checkpoint,
            caches=caches,
            profile=temporal_config.execution_profile,
            scene=scene,
        )
        final_paths = write_map_snapshot(
            final_neutral, staging / "final_current_map"
        )
        final_snapshot = Path(final_paths["snapshot"])
        final_entities = Path(final_paths["entities"])
        for path in (final_snapshot, final_entities):
            if not path.is_file() or path.is_symlink():
                raise ValueError("final current map sidecar is invalid")
            expected_checkpoint_inventory.add(path.relative_to(staging).as_posix())
        final_current_map = {
            "frame_index": final_frame_index,
            "timestamp_ns": final_timestamp_ns,
            "scope": final_neutral.scope,
            "snapshot": _file_record(final_snapshot, relative_to=staging),
            "entities": _file_record(final_entities, relative_to=staging),
            "background_storage": "snapshot.npz:background_xyz",
        }
        expected_dirs = {
            f"{item.frame_index:08d}-{item.timestamp_ns}" for item in checkpoints
        }
        checkpoint_parent = staging / "checkpoints"
        actual_dirs = {item.name for item in checkpoint_parent.iterdir()}
        if actual_dirs != expected_dirs:
            raise ValueError("checkpoint directories do not exactly match the schedule")

        _require_regular_file(source_config, "runner config")
        _require_regular_file(schedule_path, "causal schedule")
        _require_regular_file(target_path, "occlusion target manifest")
        if source_config.read_bytes() != source_config_bytes:
            raise ValueError("runner config changed during run")
        if schedule_path.read_bytes() != schedule_bytes:
            raise ValueError("causal schedule changed during run")
        if target_path.read_bytes() != target_bytes:
            raise ValueError("occlusion target manifest changed during run")
        assert_unchanged = getattr(caches, "assert_inputs_unchanged", None)
        if callable(assert_unchanged):
            assert_unchanged()
        for witness in witnesses:
            _assert_staging_identity(staging, staging_identity)
            _revalidate_checkpoint_artifact(witness, run_root=staging)
            _assert_staging_identity(staging, staging_identity)
        for witness, record in zip(cumulative_audit_witnesses, records, strict=True):
            if witness is None:
                continue
            _assert_staging_identity(staging, staging_identity)
            _revalidate_cumulative_audit_witness(
                staging, witness, record["cumulative_audit"]
            )
            _assert_staging_identity(staging, staging_identity)

        normalized_config = staging / "normalized_run_config.json"
        _assert_staging_identity(staging, staging_identity)
        _write_json(normalized_config, config)
        _assert_staging_identity(staging, staging_identity)
        inputs_root = staging / "inputs"
        inputs_root.mkdir()
        schedule_copy = inputs_root / "schedule.json"
        schedule_copy.write_bytes(schedule_bytes)
        schedule_source_record = _file_record(schedule_copy, relative_to=staging)
        trajectory_rows.sort(key=lambda row: (row["frame_index"], row["entity_id"]))
        lifecycle_rows.sort(key=lambda row: (row["frame_index"], row["entity_id"]))
        trajectories_path = staging / "trajectories.jsonl"
        _write_jsonl(trajectories_path, trajectory_rows)
        trajectories_record = _file_record(trajectories_path, relative_to=staging)
        lifecycle_path = staging / "lifecycle_transitions.jsonl"
        _write_jsonl(lifecycle_path, lifecycle_rows)
        lifecycle_record = _file_record(lifecycle_path, relative_to=staging)
        coverage_path = staging / "temporal_frame_coverage.jsonl"
        _write_jsonl(coverage_path, coverage_rows)
        coverage_record = _file_record(coverage_path, relative_to=staging)
        ordered_official_sources = [
            official_sources[item.frame_index] for item in official
        ]
        capture_path = staging / "capture_status.json"
        _write_json(
            capture_path,
            {
                "schema_version": 1,
                "status": "PASS",
                "scene": scene,
                "mode": "causal_checkpoints",
                "scheduled_frame_indices": [item.frame_index for item in official],
                "captured_frame_indices": [item.frame_index for item in official],
                "schedule": schedule_source_record,
                "trajectories": trajectories_record,
                "frame_coverage": coverage_record,
                "lifecycle_transitions": lifecycle_record,
                "checkpoint_statuses": [
                    item["checkpoint_status"] for item in ordered_official_sources
                ],
            },
        )
        capture_record = _file_record(capture_path, relative_to=staging)
        records_by_frame = {item["frame_index"]: item for item in records}
        if len(records_by_frame) != len(records):
            raise ValueError("checkpoint records contain duplicate frames")
        occlusion_records: list[dict[str, Any]] = []
        for frame_index in capture_evaluation_frames:
            record = records_by_frame.get(frame_index)
            compact_artifact = (
                record.get("artifacts", {}).get("temporal_compact")
                if isinstance(record, Mapping)
                else None
            )
            if not (
                isinstance(record, Mapping)
                and "occlusion_v1" in record["roles"]
                and isinstance(compact_artifact, Mapping)
                and compact_artifact.get("format") == TEMPORAL_COMPACT_FORMAT
            ):
                raise ValueError("occlusion checkpoint has no compact artifact")
            occlusion_records.append(
                {
                    "scene": record["scene"],
                    "frame_index": record["frame_index"],
                    "timestamp_ns": record["timestamp_ns"],
                    "relative_timestamp_ns": record["relative_timestamp_ns"],
                    "consumed_through_frame": record["consumed_through_frame"],
                    "consumed_through_frame_exclusive": record[
                        "consumed_through_frame_exclusive"
                    ],
                    "event_ids": record["event_ids"],
                    "roles": record["roles"],
                    "format": TEMPORAL_COMPACT_FORMAT,
                    "maximum_entities": temporal_config.geometry.maximum_entities,
                    "maximum_object_voxels": temporal_config.geometry.maximum_object_voxels,
                    "artifact": compact_artifact["artifact"],
                    "checksums_sha256": compact_artifact["checksums_sha256"],
                }
            )
        if [item["frame_index"] for item in occlusion_records] != list(
            capture_evaluation_frames
        ):
            raise ValueError("occlusion checkpoint index coverage is invalid")
        occlusion_index = {
            "schema_version": 1,
            "format": "oviv2_temporal_compact_v1",
            "protocol_id": PROTOCOL_ID,
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": scene,
            "algorithm_hash": config["algorithm_hash"],
            "schedule": _byte_record(schedule_bytes),
            "target_manifest": _byte_record(target_bytes),
            "input_sha256": input_sha256,
            "code_commit": code_commit,
            "source_bindings": canonical_source_bindings,
            "checkpoints": occlusion_records,
        }
        occlusion_index_path = staging / "occlusion_checkpoint_index.json"
        _write_json(occlusion_index_path, occlusion_index)
        occlusion_index_bytes = occlusion_index_path.read_bytes()
        if (
            _load_json_bytes(occlusion_index_bytes, occlusion_index_path)
            != occlusion_index
        ):
            raise ValueError("occlusion checkpoint index serialization changed")
        occlusion_index_record = _file_record(
            occlusion_index_path, relative_to=staging
        )
        _assert_staging_identity(staging, staging_identity)
        formal_fields = (
            {"frozen_run_identity": dict(frozen.frozen_run_identity)}
            if frozen is not None
            else (
                {
                    "nonformal_authorization": {
                        "status": "unfrozen_office_opt_in",
                        "publication_eligible": False,
                    }
                }
                if allow_unfrozen_office
                else {}
            )
        )
        if temporal_metrics_only:
            formal_fields["capture_mode"] = {
                "mode": "temporal_metrics_only",
                "occlusion_evaluation_available": True,
                "publication_eligible": False,
                "runtime_mode": "full_causal_without_cumulative_audit",
            }
        elif table_only:
            formal_fields["capture_mode"] = {
                "mode": "official_common_only",
                "occlusion_evaluation_available": False,
                "publication_eligible": False,
                "runtime_mode": "temporal_only_unchecked",
                "keyframe_stride": TABLE_ONLY_KEYFRAME_STRIDE,
            }
        runtime_diagnostics_path = staging / "runtime_diagnostics.json"
        _write_json(
            runtime_diagnostics_path,
            _runtime_diagnostics_payload(
                runtime,
                config=config,
                processed_frame_count=frame_count,
                mechanism_records=runtime_mechanism_records,
            ),
        )
        runtime_diagnostics_record = _file_record(
            runtime_diagnostics_path, relative_to=staging
        )
        source_index_path = staging / "source_index.json"
        _write_json(
            source_index_path,
            {
                "schema_version": 1,
                "dataset": "TESSE-CD",
                "mode": "causal_checkpoint_exports",
                "method": "OVIV2",
                "scene": scene,
                "schedule": schedule_source_record,
                "capture_status": capture_record,
                "trajectories": trajectories_record,
                "frame_coverage": coverage_record,
                "lifecycle_transitions": lifecycle_record,
                "runtime_diagnostics": runtime_diagnostics_record,
                "checkpoints": ordered_official_sources,
                **(
                    {
                        "frozen_run_identity": dict(frozen.frozen_run_identity),
                    }
                    if frozen is not None and execution is not None
                    else {}
                ),
            },
        )
        source_index_record = _file_record(source_index_path, relative_to=staging)
        published_sidecar_records = [
            schedule_source_record,
            trajectories_record,
            lifecycle_record,
            coverage_record,
            capture_record,
            runtime_diagnostics_record,
            source_index_record,
            *(
                item[role]
                for item in ordered_official_sources
                for role in ("checkpoint_status", "snapshot", "entities")
            ),
            *(
                item["cumulative_audit"][role]
                for item in records
                if item["cumulative_audit"] is not None
                for role in ("snapshot", "entities")
            ),
            final_current_map["snapshot"],
            final_current_map["entities"],
        ]
        for position, sidecar_record in enumerate(published_sidecar_records):
            _revalidate_relative_file_record(
                staging, sidecar_record, label=f"published sidecar {position}"
            )
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "protocol_id": PROTOCOL_ID,
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": scene,
            "mode": "dual_readout_causal_checkpoints",
            "algorithm_hash": config["algorithm_hash"],
            "processed_frame_count": frame_count,
            "covered_frame_count": len(coverage_rows),
            "trajectory_frame_count": len(
                {row["frame_index"] for row in trajectory_rows}
            ),
            "first_frame_index": coverage_rows[0]["frame_index"],
            "last_frame_index": coverage_rows[-1]["frame_index"],
            "temporal_export_schema_version": TEMPORAL_EXPORT_SCHEMA_VERSION,
            "scheduled_frame_indices": scheduled,
            "captured_frame_indices": captured,
            "config": _byte_record(source_config_bytes),
            "normalized_run_config": _file_record(
                normalized_config, relative_to=staging
            ),
            "schedule": _byte_record(schedule_bytes),
            "target_manifest": _byte_record(target_bytes),
            "source_bindings": canonical_source_bindings,
            "input_sha256": input_sha256,
            "code_commit": code_commit,
            "checkpoints": records,
            "occlusion_checkpoint_index": occlusion_index_record,
            "source_index": source_index_record,
            "final_current_map": final_current_map,
            **formal_fields,
        }
        manifest["artifact_inventory"] = sorted(
            expected_checkpoint_inventory
            | {
                "normalized_run_config.json",
                "occlusion_checkpoint_index.json",
                "inputs/schedule.json",
                "trajectories.jsonl",
                "lifecycle_transitions.jsonl",
                "temporal_frame_coverage.jsonl",
                "capture_status.json",
                "source_index.json",
                "runtime_diagnostics.json",
            }
        )
        run_manifest_path = staging / "run_manifest.json"
        _assert_staging_identity(staging, staging_identity)
        _write_json(run_manifest_path, manifest)
        _assert_staging_identity(staging, staging_identity)
        receipt_path = staging / "execution_receipt.json"
        _write_json(
            receipt_path,
            {
                "schema_version": 1,
                "provenance": provenance,
                "environment": current_environment,
                **(
                    {
                        "frozen_run_identity": dict(frozen.frozen_run_identity),
                        "run_execution": dict(execution),
                    }
                    if frozen is not None and execution is not None
                    else {}
                ),
            },
        )
        _assert_staging_identity(staging, staging_identity)
        deterministic_inventory = sorted(
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
            and path not in {run_manifest_path, receipt_path}
        )
        if deterministic_inventory != manifest["artifact_inventory"]:
            raise ValueError("run artifact inventory changed during manifest publication")
        expected_files = {
            *manifest["artifact_inventory"],
            "run_manifest.json",
            "execution_receipt.json",
        }
        expected_directories = {"checkpoints", "inputs"}
        for witness in witnesses:
            relative = witness.path.relative_to(staging)
            expected_directories.add(relative.as_posix())
            expected_directories.add(relative.parent.as_posix())
        for record in ordered_official_sources:
            for role in ("snapshot", "entities"):
                parent = Path(record[role]["path"]).parent
                expected_directories.add(parent.as_posix())
                expected_directories.add(parent.parent.as_posix())
                expected_directories.add(parent.parent.parent.as_posix())
        for record in records:
            if record["cumulative_audit"] is None:
                continue
            for role in ("snapshot", "entities"):
                parent = Path(record["cumulative_audit"][role]["path"]).parent
                expected_directories.add(parent.as_posix())
                expected_directories.add(parent.parent.as_posix())
        for role in ("snapshot", "entities"):
            parent = Path(final_current_map[role]["path"]).parent
            expected_directories.add(parent.as_posix())
            expected_directories.add(parent.parent.as_posix())
        expected_directories.update(cumulative_audit_directories)
        expected_entries = {
            **{path: "file" for path in expected_files},
            **{path: "directory" for path in expected_directories},
        }
        actual_entries = _entry_inventory(staging)
        if actual_entries != expected_entries:
            missing = sorted(set(expected_entries.items()) - set(actual_entries.items()))
            unexpected = sorted(set(actual_entries.items()) - set(expected_entries.items()))
            raise ValueError(
                f"run publication inventory is invalid (missing={missing}, "
                f"unexpected={unexpected})"
            )
        if frozen is not None:
            if frozen.manifest_path.read_bytes() != frozen.manifest_bytes:
                raise ValueError("freeze manifest changed during run")
            _revalidate_frozen_bindings(
                frozen.input_bindings, base=frozen.manifest_path.parent
            )
            if _validate_repository_state(
                _repository_provenance(),
                {
                    "commit": frozen.repository_state["repository_commit"],
                    "tree": frozen.repository_state["repository_tree"],
                },
            ) != frozen.repository_state:
                raise ValueError("repository identity changed during run")
        if (
            occlusion_index_path.read_bytes() != occlusion_index_bytes
            or _file_record(occlusion_index_path, relative_to=staging)
            != occlusion_index_record
        ):
            raise ValueError("occlusion checkpoint index changed during run")
        for witness, record in zip(
            cumulative_audit_witnesses, manifest["checkpoints"], strict=True
        ):
            if witness is None:
                continue
            _revalidate_cumulative_audit_witness(
                staging, witness, record["cumulative_audit"]
            )
        _assert_staging_identity(staging, staging_identity)
        if _entry_inventory(staging) != expected_entries:
            raise ValueError("run publication inventory changed before publication")
        prepublish_tree = _tree_record(staging, relative_to=staging)
        _assert_staging_identity(staging, staging_identity)
        run_manifest_sha256 = _sha256(run_manifest_path)
        phase_events.emit(
            "finalization", run_manifest_sha256=run_manifest_sha256
        )
        if office_claim is not None:
            _verify_directory_path(destination.parent, office_claim.parent_fd)
        _publish_run(staging, destination)
        try:
            if _staging_identity(destination) != staging_identity:
                raise ValueError("published run root identity changed")
            published = True
            if _entry_inventory(destination) != expected_entries:
                raise ValueError("published run inventory changed")
            if (
                _tree_record(destination, relative_to=destination)
                != prepublish_tree
            ):
                raise ValueError("published run content changed")
            for position, sidecar_record in enumerate(published_sidecar_records):
                _revalidate_relative_file_record(
                    destination,
                    sidecar_record,
                    label=f"published sidecar {position}",
                )
            for witness, record in zip(
                cumulative_audit_witnesses,
                manifest["checkpoints"],
                strict=True,
            ):
                if witness is None:
                    continue
                _revalidate_cumulative_audit_witness(
                    destination, witness, record["cumulative_audit"]
                )
        except RunPublicationUncertainError:
            raise
        except BaseException as exc:
            raise RunPublicationUncertainError(
                f"published run root identity is uncertain: {destination}"
            ) from exc
        phase_events.emit(
            "artifact_publication", run_manifest_sha256=run_manifest_sha256
        )
        _validate_phase_events(
            phase_events.events,
            run_sha256=run_sha256,
            config_sha256=config_sha256,
            input_sha256=input_sha256,
            processed_frame_count=frame_count,
            run_manifest_sha256=run_manifest_sha256,
        )
        if office_claim is not None:
            _verify_directory_path(destination.parent, office_claim.parent_fd)
        try:
            _atomic_write_jsonl(phase_events_output, phase_events.events)
        except BaseException as exc:
            raise RunPublicationUncertainError(
                f"phase event publication is uncertain: {phase_events_output}"
            ) from exc
        if office_claim is not None:
            _verify_directory_path(destination.parent, office_claim.parent_fd)
            _close_office_claim(office_claim)
            office_claim = None
        return manifest
    except BaseException as exc:
        retryable_office_failure = False
        if (
            office_attempt is not None
            and frozen is not None
            and not published
            and staging.exists()
        ):
            try:
                retryable_office_failure = _publish_office_failure_receipt(
                    frozen,
                    destination,
                    attempt=office_attempt,
                    failure=exc,
                    staging=staging,
                    claim=office_claim,
                )
            except BaseException:
                if office_claim is not None:
                    _close_office_claim(office_claim)
                    office_claim = None
                raise
        if not published and staging.exists():
            try:
                unchanged_staging = _staging_identity(staging) == staging_identity
            except OSError:
                unchanged_staging = False
            if unchanged_staging:
                shutil.rmtree(staging)
                if retryable_office_failure and office_claim is not None:
                    _release_office_claim(office_claim)
                    office_claim = None
        if office_claim is not None:
            _close_office_claim(office_claim)
            office_claim = None
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--allow-unfrozen-office", action="store_true")
    capture_mode = parser.add_mutually_exclusive_group()
    capture_mode.add_argument("--table-only", action="store_true")
    capture_mode.add_argument("--temporal-metrics-only", action="store_true")
    parser.add_argument(
        "--run-slot",
        choices=(
            "apartment_run1",
            "apartment_run2",
            "office_seed_0",
            "office_seed_17",
            "office_seed_29",
            "office_seed_43",
            "office_seed_71",
            "office_seed_101",
        ),
    )
    args = parser.parse_args(argv)
    if (args.freeze_manifest is None) != (args.run_slot is None):
        parser.error("--freeze-manifest and --run-slot must be provided together")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(
        args.config,
        args.output,
        freeze_manifest=args.freeze_manifest,
        run_slot=args.run_slot,
        allow_unfrozen_office=args.allow_unfrozen_office,
        table_only=args.table_only,
        temporal_metrics_only=args.temporal_metrics_only,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "processed_frame_count": manifest["processed_frame_count"],
                "scene": manifest["scene"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
