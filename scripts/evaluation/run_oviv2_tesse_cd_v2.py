#!/usr/bin/env python3
"""Run the OVIV2 dual cumulative/current readout on one TESSE-CD scene."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import shutil
import socket
import stat
import sys
import tempfile
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (  # noqa: E402
    RUNNER_SCENE_CONFIG_FIELDS,
    canonical_algorithm_config,
    canonical_algorithm_hash,
)
from scripts.evaluation.run_oviv2_tesse_cd import (  # noqa: E402
    RunPublicationUncertainError,
    TesseCausalCheckpoint,
    _absolute_lexical,
    _binding_path,
    _byte_record,
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
    _repository_provenance,
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
from src.evaluation.contracts import MapSnapshot  # noqa: E402
from src.evaluation.exporters.oviovo import write_map_snapshot  # noqa: E402
from src.oviv2.temporal_snapshot import (  # noqa: E402
    TEMPORAL_COMPACT_FORMAT,
    TemporalCompactCheckpoint,
    TemporalCurrentSnapshot,
    TemporalSnapshotMetadata,
)


PROTOCOL_ID = "oviv2-tessecd-v2"
SCENE_CONFIG_FIELDS = RUNNER_SCENE_CONFIG_FIELDS

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
        "dense_cache_dir",
        "dense_manifest",
        "stage3_lineage_commit",
        "dense_sample_stride",
        "dense_top_k",
        "dense_semantic_mode",
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
        payload.get("schema_version") == 1
        and payload.get("manifest_id") == "oviv2_tesse_cd_v2_selection"
        and payload.get("development_scene") == "apartment"
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
    expected_slots = [
        f"{scene}_run{repeat}"
        for scene in ("apartment", "office")
        for repeat in (1, 2)
    ]
    if type(mapping) is not list or len(mapping) != len(expected_slots):
        raise ValueError("freeze mapping commands must contain four runs")
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
        parsed_scene, *_ = _validate_config(frozen_payload)
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

    output_roots = manifest.get("output_roots")
    expected_slots = {
        f"{slot_scene}_run{repeat}"
        for slot_scene in ("apartment", "office")
        for repeat in (1, 2)
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
    if run_slot not in expected_slots or not run_slot.startswith(f"{scene}_run"):
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
    for role, config_field in shared_config_fields.items():
        expected_paths = {
            _resolve_path(frozen[config_field]) for frozen in loaded_configs.values()
        }
        if len(expected_paths) != 1:
            raise ValueError(f"freeze shared {role} config paths differ across scenes")
        _verify_exact_frozen_file_binding(
            shared[role],
            base=manifest_path.parent,
            role=f"shared {role}",
            expected_path=next(iter(expected_paths)),
        )
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
        temporal = TemporalCurrentRuntime(
            str(config["scene"]),
            temporal_config,
            tracker_config=runtime_config.tracker,
        )
    return DualReadoutRuntime(cumulative, temporal)


def _production_environment() -> Mapping[str, Any]:
    provenance = dict(_production_provenance())
    gpu = provenance.get("gpu_inventory")
    nvcc = provenance.get("nvcc_version")
    libraries = provenance.get("library_versions")
    if type(gpu) is not list or any(type(item) is not str for item in gpu):
        gpu = []
    if type(nvcc) is not list or any(type(item) is not str for item in nvcc):
        nvcc = []
    if not isinstance(libraries, Mapping):
        libraries = {}
    torch_cuda = provenance.get("torch_cuda_version", "unavailable")
    cudnn = provenance.get("cudnn_version")
    cuda = [
        f"torch_cuda={torch_cuda if torch_cuda is not None else 'unavailable'}",
        f"cudnn={cudnn if cudnn is not None else 'unavailable'}",
        *(
            [f"nvcc={line}" for line in nvcc]
            if nvcc
            else ["nvcc=unavailable"]
        ),
    ]

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": str(provenance.get("platform") or platform.platform()),
        "machine": str(provenance.get("machine") or platform.machine() or "unknown"),
        "host": str(provenance.get("hostname") or socket.gethostname() or "unknown"),
        "cuda": cuda,
        "cuda_visible_devices": provenance.get("cuda_visible_devices"),
        "gpu": gpu or ["unavailable"],
        "libraries": {
            key: str(libraries.get(key, "unavailable"))
            for key in sorted(_ENVIRONMENT_LIBRARY_KEYS)
        },
    }


def _production_dependencies() -> RunnerDependencies:
    return RunnerDependencies(
        dataset_factory=_production_dataset_factory,
        cache_loader_factory=_production_cache_loader_factory,
        runtime_factory=_production_runtime_factory,
        provenance_factory=_production_provenance,
        environment_factory=_production_environment,
    )


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
    if profile is ExecutionProfile.A2:
        if not isinstance(cumulative, MapSnapshot):
            raise TypeError("A2 neutral composition requires cumulative snapshot")
        if (
            temporal.scene_id != cumulative.scene_id
            or temporal.timestamp * 1_000_000_000 != cumulative.timestamp
            or temporal.scope != cumulative.scope
        ):
            raise ValueError("A2 temporal and cumulative neutral snapshots do not match")
        return MapSnapshot(
            method=temporal.method,
            scene_id=temporal.scene_id,
            timestamp=temporal.timestamp,
            entities=temporal.entities,
            background_xyz=cumulative.background_xyz,
            scope=temporal.scope,
            runtime=temporal.runtime,
        )
    return temporal


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


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    freeze_manifest: str | Path | None = None,
    run_slot: str | None = None,
    dependencies: RunnerDependencies | None = None,
) -> dict[str, Any]:
    if (freeze_manifest is None) != (run_slot is None):
        raise ValueError("freeze_manifest and run_slot must be provided together")
    source_config = Path(config_path).absolute()
    _require_regular_file(source_config, "runner config")
    source_config_bytes = source_config.read_bytes()
    config = _load_json_bytes(source_config_bytes, source_config)
    scene, frame_count, schedule_path, evaluation_frames, temporal_config = _validate_config(config)
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

    dependencies = _production_dependencies() if dependencies is None else dependencies
    current_environment = _validate_environment(
        dict(dependencies.environment_factory())
    )
    if (
        frozen is not None
        and current_environment != frozen.input_bindings["environment"]
    ):
        raise ValueError("frozen environment differs from the execution environment")

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, "output parent")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    staging_identity = _staging_identity(staging)
    published = False
    execution = (
        _run_execution(run_slot=frozen.run_slot, output=destination, staging=staging)
        if frozen is not None
        else None
    )
    try:
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
        runtime_config = {key: config[key] for key in _RUNTIME_CONFIG_KEYS}
        runtime = dependencies.runtime_factory(runtime_config, caches)

        by_frame = {item.frame_index: item for item in official}
        first_timestamp_ns = _dataset_timestamp_ns(dataset, 0)
        for frame_index in evaluation_frames:
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
        expected_checkpoint_inventory: set[str] = set()
        official_frames = {item.frame_index for item in official}
        official_sources: dict[int, dict[str, Any]] = {}
        trajectory_rows: list[dict[str, Any]] = []
        for frame_index in range(frame_count):
            frame = dataset[frame_index]
            if int(frame.frame_id) != frame_index:
                raise ValueError("dataset frame IDs must equal zero-based frame indices")
            observations, dense_semantics = caches.load(frame_index, frame)
            runtime.process_frame(
                frame,
                observations=observations,
                dense_semantics=dense_semantics,
            )
            checkpoint = by_frame.get(frame_index)
            if checkpoint is None:
                continue
            dataset_timestamp_ns = _dataset_timestamp_ns(dataset, frame_index)
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
            if temporal_config.execution_profile.profile_id not in {"a0", "a1"}:
                snapshot = _snapshot_from_runtime(
                    runtime,
                    checkpoint=checkpoint,
                    scene=scene,
                    config_sha256=str(config["algorithm_hash"]),
                    voxel_size_m=temporal_config.geometry.voxel_size_m,
                )
                metadata = snapshot.metadata
            else:
                reference_state = getattr(getattr(runtime, "temporal", None), "state", None)
                expected_timestamp = checkpoint.timestamp_ns / 1_000_000_000
                if not (
                    reference_state is not None
                    and reference_state.scene_id == scene
                    and reference_state.last_frame_id == checkpoint.frame_index
                    and reference_state.revision == checkpoint.frame_index + 1
                    and float(reference_state.last_timestamp) == expected_timestamp
                ):
                    raise ValueError("reference state does not match checkpoint progress")
                metadata = TemporalSnapshotMetadata(
                    scene_id=reference_state.scene_id,
                    frame_id=reference_state.last_frame_id,
                    timestamp=reference_state.last_timestamp,
                    revision=reference_state.revision,
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
            temporal_neutral = None
            if needs_full:
                if snapshot is not None:
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
                cumulative_neutral = (
                    _cumulative_neutral_from_runtime(
                        runtime, checkpoint=checkpoint, caches=caches
                    )
                    if temporal_config.execution_profile.profile_id in {"a0", "a1", "a2"}
                    else None
                )
                neutral = _compose_checkpoint_neutral(
                    temporal_config.execution_profile,
                    cumulative=cumulative_neutral,
                    reference_state=reference_state,
                    temporal=temporal_neutral,
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
                for entity in neutral.entities:
                    points = np.asarray(entity.points_xyz, dtype=np.float64)
                    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
                        raise ValueError("neutral entity has no trajectory centroid")
                    centroid = points.mean(axis=0)
                    if not np.isfinite(centroid).all():
                        raise ValueError("neutral entity trajectory is not finite")
                    trajectory_rows.append(
                        {
                            "frame_index": checkpoint.frame_index,
                            "timestamp_ns": checkpoint.timestamp_ns,
                            "entity_id": entity.entity_id,
                            "centroid_xyz": [float(value) for value in centroid],
                        }
                    )
                if snapshot is None:
                    neutral_tree = _tree_record(
                        root / "neutral_current", relative_to=staging
                    )
                    artifacts["neutral_current"] = {
                        "format": "oviv2_neutral_current_v1",
                        "artifact": neutral_tree,
                        "checksums_sha256": neutral_tree["sha256"],
                    }
                else:
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
                    if needs_full and snapshot is not None
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

        scheduled = [item.frame_index for item in checkpoints]
        if captured != scheduled:
            raise ValueError("captured checkpoints do not exactly match the schedule")
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
        trajectories_path = staging / "trajectories.jsonl"
        trajectories_path.write_text(
            "".join(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for row in trajectory_rows
            ),
            encoding="utf-8",
        )
        trajectories_record = _file_record(trajectories_path, relative_to=staging)
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
        for frame_index in evaluation_frames:
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
            evaluation_frames
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
            else {}
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
            capture_record,
            source_index_record,
            *(
                item[role]
                for item in ordered_official_sources
                for role in ("checkpoint_status", "snapshot", "entities")
            ),
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
            **formal_fields,
        }
        manifest["artifact_inventory"] = sorted(
            expected_checkpoint_inventory
            | {
                "normalized_run_config.json",
                "occlusion_checkpoint_index.json",
                "inputs/schedule.json",
                "trajectories.jsonl",
                "capture_status.json",
                "source_index.json",
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
        _assert_staging_identity(staging, staging_identity)
        if _entry_inventory(staging) != expected_entries:
            raise ValueError("run publication inventory changed before publication")
        prepublish_tree = _tree_record(staging, relative_to=staging)
        _assert_staging_identity(staging, staging_identity)
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
        except RunPublicationUncertainError:
            raise
        except BaseException as exc:
            raise RunPublicationUncertainError(
                f"published run root identity is uncertain: {destination}"
            ) from exc
        return manifest
    except BaseException:
        if not published and staging.exists():
            try:
                unchanged_staging = _staging_identity(staging) == staging_identity
            except OSError:
                unchanged_staging = False
            if unchanged_staging:
                shutil.rmtree(staging)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument(
        "--run-slot",
        choices=(
            "apartment_run1",
            "apartment_run2",
            "office_run1",
            "office_run2",
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
