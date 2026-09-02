"""Offline causal attribution for CROVE's Apartment current-map failures."""

from __future__ import annotations

import math
import os
import stat
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from src.evaluation.baselines.dynamic_metrics import evaluate_dynamic_frame
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.exporters.oviovo import read_map_snapshot
from src.evaluation.json_contracts import loads_strict

OBJECT_AUTHORITY_BUCKETS = (
    "ovimap_anchor_bound_unchanged",
    "ovimap_anchor_unbound",
    "crove_temporal_moved",
    "crove_temporal_new",
)
ALL_AUTHORITY_BUCKETS = OBJECT_AUTHORITY_BUCKETS + (
    "anchor_background",
    "temporal_background",
)

_MAX_JSON_BYTES = 64 * 1024 * 1024


def _points(value: object, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"{label} must have shape (N, 3)")
    if points.size and not np.isfinite(points).all():
        raise ValueError(f"{label} must contain finite values")
    return points


def _regular_file(path: str | Path, label: str) -> Path:
    value = Path(os.path.abspath(os.fspath(path)))
    try:
        current = Path(value.anchor)
        metadata = current.lstat()
        for component in value.parts[1:]:
            current /= component
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"{label} must be a direct regular file")
    except FileNotFoundError as exc:
        raise ValueError(f"{label} must be a direct regular file") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a direct regular file")
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label)
    content = source.read_bytes()
    if len(content) > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds the size limit")
    payload = loads_strict(content.decode("utf-8"), label=label)
    return source, dict(_mapping(payload, label))


def validate_file_record(
    root: str | Path,
    record: object,
    *,
    label: str,
) -> Path:
    """Resolve and validate one immutable file declaration."""

    declaration = _mapping(record, f"{label} record")
    raw_path = declaration.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} binding path is invalid")
    relative = Path(raw_path)
    if relative.is_absolute():
        candidate = relative
    else:
        if ".." in relative.parts:
            raise ValueError(f"{label} binding escapes its root")
        candidate = Path(root) / relative
    path = _regular_file(candidate, label)
    byte_count = declaration.get("byte_count")
    digest = declaration.get("sha256")
    if (
        type(byte_count) is not int
        or byte_count < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or path.stat().st_size != byte_count
        or _sha256(path) != digest
    ):
        raise ValueError(f"{label} binding does not match the file")
    return path


def _object_authority(entity: EntityPrediction) -> str:
    authority = entity.metadata.get("authority")
    overlay_state = entity.metadata.get("overlay_state")
    temporal_id = entity.metadata.get("temporal_entity_id")
    bound = type(temporal_id) is int and temporal_id > 0
    if authority == "ovimap_anchor" and overlay_state in {"unchanged", "occluded"}:
        return "ovimap_anchor_bound_unchanged" if bound else "ovimap_anchor_unbound"
    if authority == "crove_temporal" and overlay_state == "moved" and bound:
        return "crove_temporal_moved"
    if authority == "crove_temporal" and overlay_state == "new" and bound:
        return "crove_temporal_new"
    raise ValueError(
        f"invalid object authority combination for {entity.entity_id}: "
        f"{authority!r}/{overlay_state!r}"
    )


def partition_object_authorities(snapshot: MapSnapshot) -> dict[str, np.ndarray]:
    """Partition every object point into one mutually exclusive authority."""

    if not isinstance(snapshot, MapSnapshot):
        raise TypeError("snapshot must be a MapSnapshot")
    chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in OBJECT_AUTHORITY_BUCKETS
    }
    for entity in snapshot.entities:
        chunks[_object_authority(entity)].append(
            _points(entity.points_xyz, f"entity {entity.entity_id} points")
        )
    return {
        name: (
            np.concatenate(chunks[name], axis=0)
            if chunks[name]
            else np.empty((0, 3), dtype=np.float32)
        )
        for name in OBJECT_AUTHORITY_BUCKETS
    }


def partition_background_authorities(
    composed: object,
    anchor: object,
    temporal: object,
    *,
    voxel_size_m: float,
) -> dict[str, np.ndarray]:
    """Replay the production anchor-first, lexicographic background merge."""

    voxel = float(voxel_size_m)
    if not math.isfinite(voxel) or voxel <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")
    composed_points = _points(composed, "composed background")
    sources = (
        ("anchor_background", _points(anchor, "anchor background")),
        ("temporal_background", _points(temporal, "temporal background")),
    )
    selected: dict[tuple[int, int, int], tuple[tuple[float, float, float], str]] = {}
    for authority, points in sources:
        for point in points:
            key = tuple(
                int(value)
                for value in np.floor(
                    np.asarray(point, dtype=np.float64) / voxel
                ).astype(np.int64)
            )
            candidate = tuple(float(value) for value in point)
            previous = selected.get(key)
            if previous is None or candidate < previous[0]:
                selected[key] = (candidate, authority)
    replay = np.asarray(
        [selected[key][0] for key in sorted(selected)], dtype=np.float32
    ).reshape((-1, 3))
    if not np.array_equal(replay, composed_points):
        raise ValueError("background composition does not match the production rule")
    return {
        authority: np.asarray(
            [
                selected[key][0]
                for key in sorted(selected)
                if selected[key][1] == authority
            ],
            dtype=np.float32,
        ).reshape((-1, 3))
        for authority in ALL_AUTHORITY_BUCKETS[-2:]
    }


def _crop_to_region(
    points: np.ndarray, region_voxels: np.ndarray, voxel_size_m: float
) -> np.ndarray:
    region = np.asarray(region_voxels, dtype=np.int64)
    if region.ndim != 2 or region.shape[1:] != (3,) or not len(region):
        raise ValueError("region_voxels must be a non-empty (N, 3) array")
    if not len(points):
        return np.empty((0, 3), dtype=np.float32)
    lower = region.min(axis=0)
    upper = region.max(axis=0)
    spans = upper - lower + 1
    if any(int(value) <= 0 for value in spans):
        raise ValueError("region_voxels has invalid bounds")
    region_relative = region - lower
    region_codes = (region_relative[:, 0] * spans[1] + region_relative[:, 1]) * spans[
        2
    ] + region_relative[:, 2]
    minimum = lower.astype(np.float64) * voxel_size_m
    maximum = (upper.astype(np.float64) + 1.0) * voxel_size_m
    bounded = np.all((points >= minimum) & (points < maximum), axis=1)
    candidates = points[bounded]
    if not len(candidates):
        return np.empty((0, 3), dtype=np.float32)
    keys = np.floor(candidates.astype(np.float64) / voxel_size_m).astype(np.int64)
    relative = keys - lower
    codes = (relative[:, 0] * spans[1] + relative[:, 1]) * spans[2] + relative[:, 2]
    return candidates[np.isin(codes, region_codes, assume_unique=False)]


def _voxel_centers(keys: np.ndarray, voxel_size_m: float) -> np.ndarray:
    values = np.asarray(keys, dtype=np.int64)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise ValueError("voxel array must have shape (N, 3)")
    return ((values.astype(np.float64) + 0.5) * voxel_size_m).astype(np.float32)


def _ghost_count(points: np.ndarray, free_points: np.ndarray, threshold: float) -> int:
    if not len(points) or not len(free_points):
        return 0
    distances, _ = cKDTree(free_points).query(points, k=1, workers=-1)
    return int(np.sum(np.asarray(distances, dtype=np.float64) < threshold))


def attribute_authority_buckets(
    buckets: Mapping[str, object],
    *,
    region_voxels: np.ndarray,
    confirmed_free_voxels: np.ndarray,
    voxel_size_m: float,
    distance_threshold_m: float,
) -> dict[str, Any]:
    """Attribute official object ghosts and an explicitly nonofficial extension."""

    if tuple(buckets) != ALL_AUTHORITY_BUCKETS:
        raise ValueError("authority buckets must use the complete canonical order")
    voxel = float(voxel_size_m)
    threshold = float(distance_threshold_m)
    if not math.isfinite(voxel) or voxel <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("distance_threshold_m must be finite and positive")
    free_points = _voxel_centers(confirmed_free_voxels, voxel)
    rows: dict[str, dict[str, float | int]] = {}
    cropped: dict[str, np.ndarray] = {}
    for name in ALL_AUTHORITY_BUCKETS:
        selected = _crop_to_region(_points(buckets[name], name), region_voxels, voxel)
        cropped[name] = selected
        ghosts = _ghost_count(selected, free_points, threshold)
        rows[name] = {
            "predicted_points": len(selected),
            "ghost_matches": ghosts,
            "ghost_rate": ghosts / len(selected) if len(selected) else 0.0,
        }

    def aggregate(names: Sequence[str]) -> dict[str, float | int | bool]:
        predicted = sum(int(rows[name]["predicted_points"]) for name in names)
        ghosts = sum(int(rows[name]["ghost_matches"]) for name in names)
        direct_points = (
            np.concatenate([cropped[name] for name in names], axis=0)
            if names
            else np.empty((0, 3), dtype=np.float32)
        )
        direct_ghosts = _ghost_count(direct_points, free_points, threshold)
        return {
            "predicted_points": predicted,
            "ghost_matches": ghosts,
            "ghost_rate": ghosts / predicted if predicted else 0.0,
            "partition_exact": predicted == len(direct_points)
            and ghosts == direct_ghosts,
        }

    official = aggregate(OBJECT_AUTHORITY_BUCKETS)
    extended = aggregate(ALL_AUTHORITY_BUCKETS)
    official_ghosts = int(official["ghost_matches"])
    extended_ghosts = int(extended["ghost_matches"])
    official_contributions = {
        name: (
            int(rows[name]["ghost_matches"]) / official_ghosts
            if official_ghosts
            else 0.0
        )
        for name in OBJECT_AUTHORITY_BUCKETS
    }
    extended_contributions = {
        name: (
            int(rows[name]["ghost_matches"]) / extended_ghosts
            if extended_ghosts
            else 0.0
        )
        for name in ALL_AUTHORITY_BUCKETS
    }
    return {
        "buckets": rows,
        "official_object_ghost_contribution_fraction": official_contributions,
        "extended_all_authority_ghost_contribution_fraction": extended_contributions,
        "official_object_partition": official,
        "extended_all_authority": extended,
    }


def _latest_trajectory(
    rows: Sequence[Mapping[str, Any]], entity_id: int, frame: int
) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("entity_id") == entity_id
        and type(row.get("frame_index")) is int
        and int(row["frame_index"]) <= frame
    ]
    return (
        max(candidates, key=lambda row: int(row["frame_index"])) if candidates else None
    )


def build_overlay_funnel(
    anchor: MapSnapshot,
    *,
    bindings: Mapping[str, int],
    trajectory_rows: Sequence[Mapping[str, Any]],
    cutoff_frame: int,
    checkpoint_frame: int,
    diagnostics: Mapping[str, Any],
    moved_displacement_m: float,
) -> list[dict[str, Any]]:
    """Explain every anchor's exact overlay decision at one checkpoint."""

    if not isinstance(anchor, MapSnapshot):
        raise TypeError("anchor must be a MapSnapshot")
    threshold = float(moved_displacement_m)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("moved_displacement_m must be finite and positive")
    anchor_ids = {entity.entity_id for entity in anchor.entities}
    state_keys = {
        "unchanged": "unchanged_anchor_ids",
        "moved": "moved_anchor_ids",
        "removed": "removed_anchor_ids",
        "occluded": "occluded_anchor_ids",
    }
    states = {name: set(diagnostics.get(key, ())) for name, key in state_keys.items()}
    flattened = [value for values in states.values() for value in values]
    if len(flattened) != len(set(flattened)) or set(flattened) != anchor_ids:
        raise ValueError("overlay diagnostics must partition every anchor exactly")
    if set(bindings) - anchor_ids or any(
        type(value) is not int or value <= 0 for value in bindings.values()
    ):
        raise ValueError("overlay bindings are invalid")

    records: list[dict[str, Any]] = []
    for entity in sorted(anchor.entities, key=lambda item: item.entity_id):
        anchor_id = entity.entity_id
        temporal_id = bindings.get(anchor_id)
        initial = (
            None
            if temporal_id is None
            else _latest_trajectory(trajectory_rows, temporal_id, cutoff_frame)
        )
        current = (
            None
            if temporal_id is None
            else _latest_trajectory(trajectory_rows, temporal_id, checkpoint_frame)
        )
        initial_epoch = None if initial is None else initial.get("geometry_epoch")
        current_epoch = None if current is None else current.get("geometry_epoch")
        displacement = None
        if current is not None:
            centroid = np.asarray(current.get("centroid_xyz"), dtype=np.float64)
            if centroid.shape != (3,) or not np.isfinite(centroid).all():
                raise ValueError(f"trajectory centroid is invalid for {anchor_id}")
            anchor_centroid = np.asarray(entity.points_xyz, dtype=np.float64).mean(
                axis=0
            )
            displacement = float(np.linalg.norm(anchor_centroid - centroid))
        records.append(
            {
                "anchor_entity_id": anchor_id,
                "bound": temporal_id is not None,
                "temporal_entity_id": temporal_id,
                "dynamic_state": None
                if current is None
                else current.get("dynamic_state"),
                "geometry_epoch": current_epoch,
                "initial_geometry_epoch": initial_epoch,
                "geometry_epoch_advanced": (
                    current_epoch is not None
                    and initial_epoch is not None
                    and int(current_epoch) > int(initial_epoch)
                ),
                "anchor_to_current_displacement_m": displacement,
                "moved_threshold_passed": (
                    displacement is not None and displacement >= threshold
                ),
                "moved": anchor_id in states["moved"],
                "removed": anchor_id in states["removed"],
                "occluded": anchor_id in states["occluded"],
                "new": False,
                "readout_valid": None
                if current is None
                else current.get("readout_valid"),
            }
        )
    return records


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = loads_strict(line, label=f"{label} line {line_number}")
        rows.append(dict(_mapping(row, f"{label} line {line_number}")))
    return rows


def _load_target_arrays(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    schedule_record: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("manifest_id") == "tesse_cd_common_v2_targets"
        and manifest.get("dataset") == "TESSE-CD"
        and manifest.get("status") == "GENERATED"
        and manifest.get("targets_generated") is True
        and manifest.get("prediction_inputs_used") is False
    ):
        raise ValueError("target manifest is not a generated common-v2 package")
    metadata = _mapping(manifest.get("metadata"), "target metadata")
    if metadata.get("voxel_size_m") != 0.05 or set(metadata.get("scenes", ())) != {
        "apartment",
        "office",
    }:
        raise ValueError("target metadata identity mismatch")
    target_schedule = _mapping(metadata.get("schedule"), "target schedule binding")
    if target_schedule.get("sha256") != schedule_record.get(
        "sha256"
    ) or target_schedule.get("byte_count") != schedule_record.get("byte_count"):
        raise ValueError("target schedule binding mismatch")
    declaration = _mapping(manifest.get("target_arrays"), "target arrays")
    arrays_path = validate_file_record(
        manifest_path.parent, declaration, label="target arrays"
    )
    declared = _mapping(declaration.get("arrays"), "target array declarations")
    arrays: dict[str, np.ndarray] = {}
    with np.load(arrays_path, allow_pickle=False) as bundle:
        if set(bundle.files) != set(declared):
            raise ValueError("target array inventory mismatch")
        for name in sorted(
            value for value in bundle.files if value.startswith("apartment")
        ):
            values = np.asarray(bundle[name])
            record = _mapping(declared[name], f"target array {name}")
            if (
                list(values.shape) != record.get("shape")
                or str(values.dtype) != record.get("dtype")
                or values.size != record.get("element_count")
                or not np.issubdtype(values.dtype, np.integer)
            ):
                raise ValueError(f"target array declaration mismatch: {name}")
            arrays[name] = np.array(values, copy=True)
    return arrays, _file_record(arrays_path)


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bucket_totals = {
        name: {"predicted_points": 0, "ghost_matches": 0}
        for name in ALL_AUTHORITY_BUCKETS
    }
    for row in rows:
        for name in ALL_AUTHORITY_BUCKETS:
            metrics = row["attribution"]["buckets"][name]
            bucket_totals[name]["predicted_points"] += metrics["predicted_points"]
            bucket_totals[name]["ghost_matches"] += metrics["ghost_matches"]
    extended_ghosts = sum(item["ghost_matches"] for item in bucket_totals.values())
    official_ghosts = sum(
        bucket_totals[name]["ghost_matches"] for name in OBJECT_AUTHORITY_BUCKETS
    )
    for item in bucket_totals.values():
        predicted = item["predicted_points"]
        item["micro_ghost_rate"] = (
            item["ghost_matches"] / predicted if predicted else 0.0
        )
        item["extended_ghost_contribution_fraction"] = (
            item["ghost_matches"] / extended_ghosts if extended_ghosts else 0.0
        )
    for name in OBJECT_AUTHORITY_BUCKETS:
        bucket_totals[name]["official_ghost_contribution_fraction"] = (
            bucket_totals[name]["ghost_matches"] / official_ghosts
            if official_ghosts
            else 0.0
        )
    official_predicted = sum(
        bucket_totals[name]["predicted_points"] for name in OBJECT_AUTHORITY_BUCKETS
    )
    overlay_rows = [item for row in rows for item in row["overlay"]]
    bound_rows = [item for item in overlay_rows if item["bound"]]
    overlay = {
        "anchor_event_frame_records": len(overlay_rows),
        "bound_records": len(bound_rows),
        "unbound_records": len(overlay_rows) - len(bound_rows),
        "dynamic_records": sum(
            item["dynamic_state"] == "dynamic" for item in bound_rows
        ),
        "geometry_epoch_advanced_records": sum(
            item["geometry_epoch_advanced"] for item in bound_rows
        ),
        "displacement_threshold_passed_records": sum(
            item["moved_threshold_passed"] for item in bound_rows
        ),
        "all_moved_gates_passed_records": sum(
            item["dynamic_state"] == "dynamic"
            and item["geometry_epoch_advanced"]
            and item["moved_threshold_passed"]
            for item in bound_rows
        ),
        "moved_records": sum(item["moved"] for item in overlay_rows),
        "removed_records": sum(item["removed"] for item in overlay_rows),
        "occluded_records": sum(item["occluded"] for item in overlay_rows),
    }
    return {
        "evaluated_event_frames": len(rows),
        "mean_frame_official_ghost_rate": fmean(
            float(row["attribution"]["official_object_partition"]["ghost_rate"])
            for row in rows
        ),
        "official_object_micro": {
            "predicted_points": official_predicted,
            "ghost_matches": official_ghosts,
            "ghost_rate": official_ghosts / official_predicted
            if official_predicted
            else 0.0,
        },
        "buckets": bucket_totals,
        "overlay_funnel": overlay,
    }


def analyze_crove_tesse_failures(
    *,
    composed_run_manifest: str | Path,
    source_run_manifest: str | Path,
    anchor_manifest: str | Path,
    schedule: str | Path,
    target_manifest: str | Path,
) -> dict[str, Any]:
    """Analyze Apartment only; no Office method output is opened."""

    composed_path, composed = _load_json(composed_run_manifest, "composed run manifest")
    source_path, source = _load_json(source_run_manifest, "source run manifest")
    anchor_path, anchor_payload = _load_json(anchor_manifest, "anchor manifest")
    schedule_path, schedule_payload = _load_json(schedule, "causal schedule")
    target_path, target_payload = _load_json(target_manifest, "target manifest")
    if not (
        composed.get("status") == "PASS"
        and composed.get("scene") == "apartment"
        and composed.get("integration") == "composed"
        and source.get("scene") == "apartment"
        and anchor_payload.get("scene") == "apartment"
        and anchor_payload.get("status") == "PASS"
    ):
        raise ValueError("all method artifacts must be passing Apartment artifacts")
    inputs = _mapping(composed.get("inputs"), "composed inputs")
    if (
        validate_file_record(
            composed_path.parent,
            inputs.get("source_run_manifest"),
            label="source manifest input",
        )
        != source_path
    ):
        raise ValueError("source manifest argument does not match composed input")
    if (
        validate_file_record(
            composed_path.parent,
            inputs.get("anchor_manifest"),
            label="anchor manifest input",
        )
        != anchor_path
    ):
        raise ValueError("anchor manifest argument does not match composed input")
    schedule_input_path = validate_file_record(
        composed_path.parent, inputs.get("causal_schedule"), label="schedule input"
    )
    schedule_record = _file_record(schedule_path)
    if {
        key: _file_record(schedule_input_path)[key] for key in ("sha256", "byte_count")
    } != {key: schedule_record[key] for key in ("sha256", "byte_count")}:
        raise ValueError("schedule argument does not match composed input")

    scenes = _mapping(schedule_payload.get("scenes"), "schedule scenes")
    apartment_schedule = _mapping(scenes.get("apartment"), "Apartment schedule")
    events_raw = apartment_schedule.get("events")
    entries_raw = apartment_schedule.get("entries")
    if not (
        schedule_payload.get("schema_version") == 2
        and schedule_payload.get("manifest_id") == "tesse_cd_causal_schedule_v2"
        and schedule_payload.get("method_predictions_used") is False
        and isinstance(events_raw, list)
        and isinstance(entries_raw, list)
    ):
        raise ValueError("causal schedule identity mismatch")
    events = [dict(_mapping(value, "Apartment event")) for value in events_raw]
    common_frames = {
        int(entry["frame_index"])
        for value in entries_raw
        for entry in [_mapping(value, "Apartment schedule entry")]
        if "common_v2" in entry.get("roles", ())
    }
    expected_event_frames = {
        (str(event["event_id"]), int(frame))
        for event in events
        for frame in event["common_checkpoint_frame_indices"]
    }
    if {frame for _, frame in expected_event_frames} != common_frames:
        raise ValueError("event grids do not cover the Apartment common-v2 schedule")

    arrays, target_arrays_record = _load_target_arrays(
        target_path, target_payload, schedule_record
    )
    anchor_snapshot_path = validate_file_record(
        anchor_path.parent,
        _mapping(anchor_payload.get("outputs"), "anchor outputs").get("snapshot"),
        label="anchor snapshot",
    )
    anchor_entities_path = validate_file_record(
        anchor_path.parent,
        _mapping(anchor_payload.get("outputs"), "anchor outputs").get("entities"),
        label="anchor entities",
    )
    anchor_snapshot = read_map_snapshot(anchor_snapshot_path, anchor_entities_path)
    if anchor_snapshot.scene_id != "apartment":
        raise ValueError("anchor snapshot is not Apartment")
    anchor_config_path = validate_file_record(
        composed_path.parent, inputs.get("anchor_config"), label="anchor config"
    )
    _, anchor_config = _load_json(anchor_config_path, "anchor config")
    voxel_size = float(anchor_config.get("background_voxel_size_m"))
    moved_threshold = float(anchor_config.get("moved_displacement_m"))
    cutoff_frame = composed.get("causal_anchor_cutoff_frame")
    if type(cutoff_frame) is not int or cutoff_frame < 0:
        raise ValueError("causal anchor cutoff is invalid")

    trajectory_path = validate_file_record(
        composed_path.parent,
        inputs.get("source_trajectories"),
        label="source trajectories",
    )
    trajectory_rows = _load_jsonl(trajectory_path, "source trajectories")
    bindings_raw = composed.get("identity_bindings")
    if not isinstance(bindings_raw, list):
        raise TypeError("identity_bindings must be a list")
    binding_pairs: list[tuple[str, int]] = []
    for item in bindings_raw:
        binding = _mapping(item, "identity binding")
        anchor_id = binding.get("anchor_entity_id")
        temporal_id = binding.get("temporal_entity_id")
        if (
            not isinstance(anchor_id, str)
            or not anchor_id
            or type(temporal_id) is not int
            or temporal_id <= 0
        ):
            raise ValueError("identity binding values are invalid")
        binding_pairs.append((anchor_id, temporal_id))
    if len({item[0] for item in binding_pairs}) != len(binding_pairs) or len(
        {item[1] for item in binding_pairs}
    ) != len(binding_pairs):
        raise ValueError("identity bindings must be one-to-one")
    bindings = dict(binding_pairs)

    source_checkpoints = {
        int(_mapping(value, "source checkpoint")["frame_index"]): _mapping(
            value, "source checkpoint"
        )
        for value in source.get("checkpoints", ())
    }
    composed_checkpoints = {
        int(_mapping(value, "composed checkpoint")["frame_index"]): _mapping(
            value, "composed checkpoint"
        )
        for value in composed.get("checkpoints", ())
    }
    if not common_frames.issubset(composed_checkpoints) or not common_frames.issubset(
        source_checkpoints
    ):
        raise ValueError("checkpoint manifests do not cover Apartment common-v2")

    event_by_frame: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        for frame in event["common_checkpoint_frame_indices"]:
            event_by_frame.setdefault(int(frame), []).append(event)

    frame_rows: list[dict[str, Any]] = []
    entity_totals: dict[tuple[str, str], dict[str, Any]] = {}
    for frame in sorted(common_frames):
        checkpoint = composed_checkpoints[frame]
        source_checkpoint = source_checkpoints[frame]
        composed_snapshot_path = validate_file_record(
            composed_path.parent,
            checkpoint.get("snapshot"),
            label=f"frame {frame} snapshot",
        )
        composed_entities_path = validate_file_record(
            composed_path.parent,
            checkpoint.get("entities"),
            label=f"frame {frame} entities",
        )
        if checkpoint.get("source_snapshot") != source_checkpoint.get(
            "neutral_snapshot"
        ) or checkpoint.get("source_entities") != source_checkpoint.get(
            "neutral_entities"
        ):
            raise ValueError(f"frame {frame} source checkpoint binding mismatch")
        temporal_snapshot_path = validate_file_record(
            source_path.parent,
            checkpoint.get("source_snapshot"),
            label=f"frame {frame} temporal snapshot",
        )
        temporal_entities_path = validate_file_record(
            source_path.parent,
            checkpoint.get("source_entities"),
            label=f"frame {frame} temporal entities",
        )
        diagnostics_path = validate_file_record(
            composed_path.parent,
            checkpoint.get("diagnostics_file"),
            label=f"frame {frame} diagnostics",
        )
        _, diagnostics = _load_json(diagnostics_path, f"frame {frame} diagnostics")
        if diagnostics != checkpoint.get("diagnostics"):
            raise ValueError(f"frame {frame} inline diagnostics mismatch")
        composed_snapshot = read_map_snapshot(
            composed_snapshot_path, composed_entities_path
        )
        temporal_snapshot = read_map_snapshot(
            temporal_snapshot_path, temporal_entities_path
        )
        if (
            composed_snapshot.scene_id != "apartment"
            or temporal_snapshot.scene_id != "apartment"
        ):
            raise ValueError(f"frame {frame} contains a non-Apartment snapshot")
        object_buckets = partition_object_authorities(composed_snapshot)
        background_buckets = partition_background_authorities(
            np.empty((0, 3), dtype=np.float32)
            if composed_snapshot.background_xyz is None
            else composed_snapshot.background_xyz,
            np.empty((0, 3), dtype=np.float32)
            if anchor_snapshot.background_xyz is None
            else anchor_snapshot.background_xyz,
            np.empty((0, 3), dtype=np.float32)
            if temporal_snapshot.background_xyz is None
            else temporal_snapshot.background_xyz,
            voxel_size_m=voxel_size,
        )
        buckets = {**object_buckets, **background_buckets}
        overlay = build_overlay_funnel(
            anchor_snapshot,
            bindings=bindings,
            trajectory_rows=trajectory_rows,
            cutoff_frame=cutoff_frame,
            checkpoint_frame=frame,
            diagnostics=diagnostics,
            moved_displacement_m=moved_threshold,
        )
        for event in event_by_frame[frame]:
            event_id = str(event["event_id"])
            region_name = f"{event_id}.region"
            free_name = f"{event_id}.confirmed_free.{frame:06d}"
            if region_name not in arrays or free_name not in arrays:
                raise ValueError(f"target arrays missing for {event_id}/{frame}")
            attribution = attribute_authority_buckets(
                buckets,
                region_voxels=arrays[region_name],
                confirmed_free_voxels=arrays[free_name],
                voxel_size_m=voxel_size,
                distance_threshold_m=0.05,
            )
            official = attribution["official_object_partition"]
            object_points = np.concatenate(
                [
                    _crop_to_region(
                        object_buckets[name], arrays[region_name], voxel_size
                    )
                    for name in OBJECT_AUTHORITY_BUCKETS
                ],
                axis=0,
            )
            official_check = evaluate_dynamic_frame(
                event_id=event_id,
                frame_id=frame,
                intervention_frame_id=int(event["intervention_frame_index"]),
                ground_truth_semantic_ids=np.asarray([1], dtype=np.int64),
                predicted_semantic_ids=np.asarray([1], dtype=np.int64),
                valid_semantic_ids={1},
                predicted_object_points_in_changed_region=object_points,
                confirmed_free_space_points=_voxel_centers(
                    arrays[free_name], voxel_size
                ),
                predicted_background_points_in_revealed_region=np.empty(
                    (0, 3), dtype=np.float32
                ),
                ground_truth_revealed_background_points=np.empty(
                    (0, 3), dtype=np.float32
                ),
                distance_threshold_m=0.05,
            )
            if (
                official_check.predicted_changed_object_count
                != official["predicted_points"]
                or official_check.ghost_count != official["ghost_matches"]
                or official_check.ghost_rate != official["ghost_rate"]
            ):
                raise RuntimeError(
                    "authority partition disagrees with official ghost metric"
                )

            entity_rows: list[dict[str, Any]] = []
            free_points = _voxel_centers(arrays[free_name], voxel_size)
            for entity in composed_snapshot.entities:
                selected = _crop_to_region(
                    entity.points_xyz, arrays[region_name], voxel_size
                )
                if not len(selected):
                    continue
                ghosts = _ghost_count(selected, free_points, 0.05)
                authority = _object_authority(entity)
                entity_row = {
                    "entity_id": entity.entity_id,
                    "authority": authority,
                    "predicted_points": len(selected),
                    "ghost_matches": ghosts,
                    "ghost_rate": ghosts / len(selected),
                }
                entity_rows.append(entity_row)
                key = (entity.entity_id, authority)
                total = entity_totals.setdefault(
                    key,
                    {
                        "entity_id": entity.entity_id,
                        "authority": authority,
                        "event_frame_appearances": 0,
                        "predicted_points": 0,
                        "ghost_matches": 0,
                    },
                )
                total["event_frame_appearances"] += 1
                total["predicted_points"] += len(selected)
                total["ghost_matches"] += ghosts
            frame_rows.append(
                {
                    "event_id": event_id,
                    "frame_index": frame,
                    "attribution": attribution,
                    "official_cross_check": "PASS",
                    "entity_attribution": sorted(
                        entity_rows,
                        key=lambda item: (-item["ghost_matches"], item["entity_id"]),
                    ),
                    "overlay": overlay,
                    "new_temporal_ids": diagnostics.get("new_temporal_ids", []),
                }
            )

    aggregate = _aggregate_rows(frame_rows)
    representatives = []
    for row in entity_totals.values():
        row["micro_ghost_rate"] = (
            row["ghost_matches"] / row["predicted_points"]
            if row["predicted_points"]
            else 0.0
        )
        representatives.append(row)
    representatives.sort(key=lambda item: (-item["ghost_matches"], item["entity_id"]))

    source_records = {
        "composed_run_manifest": _file_record(composed_path),
        "source_run_manifest": _file_record(source_path),
        "anchor_manifest": _file_record(anchor_path),
        "schedule": schedule_record,
        "target_manifest": _file_record(target_path),
        "target_arrays": target_arrays_record,
        "anchor_config": _file_record(anchor_config_path),
        "source_trajectories": _file_record(trajectory_path),
        "analyzer": _file_record(Path(__file__).resolve()),
    }
    for label, record in source_records.items():
        path = _regular_file(str(record["path"]), label)
        if _file_record(path) != record:
            raise ValueError(f"{label} changed during attribution")
    return {
        "schema_version": 1,
        "manifest_id": "crove_tesse_failure_attribution_v1",
        "dataset": "TESSE-CD",
        "scene": "apartment",
        "status": "PASS",
        "metric_contract": {
            "official_scope": "object_points_in_changed_region",
            "distance_rule": "nearest_confirmed_free_distance_m < 0.05",
            "background_authorities_are_extended_diagnostics_only": True,
            "voxel_size_m": voxel_size,
        },
        "aggregate": aggregate,
        "representative_entities": representatives[:20],
        "frames": frame_rows,
        "sources": source_records,
    }


def render_attribution_markdown(report: Mapping[str, Any]) -> str:
    aggregate = _mapping(report.get("aggregate"), "report aggregate")
    buckets = _mapping(aggregate.get("buckets"), "aggregate buckets")
    lines = [
        "# CROVE Apartment Failure Attribution",
        "",
        "Official Ghost uses object points only; background rows are diagnostic.",
        "",
        f"- Event-frame evaluations: {aggregate['evaluated_event_frames']}",
        f"- Mean official Ghost: {aggregate['mean_frame_official_ghost_rate']:.9f}",
        "",
        "| Authority | Predicted points | Ghost matches | Micro Ghost |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ALL_AUTHORITY_BUCKETS:
        row = _mapping(buckets[name], name)
        lines.append(
            f"| `{name}` | {row['predicted_points']} | {row['ghost_matches']} | "
            f"{row['micro_ghost_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Leading object contributors",
            "",
            "| Entity | Authority | Ghost matches | Predicted points | Micro Ghost |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in report.get("representative_entities", ()):  # type: ignore[union-attr]
        lines.append(
            f"| `{row['entity_id']}` | `{row['authority']}` | {row['ghost_matches']} | "
            f"{row['predicted_points']} | {row['micro_ghost_rate']:.6f} |"
        )
    return "\n".join(lines) + "\n"
