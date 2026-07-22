#!/usr/bin/env python3
"""Evaluate a frozen TESSE-CD map against a complete common-v2 target package."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.baselines.dynamic_metrics import (  # noqa: E402
    DynamicFrameMetrics,
    aggregate_dynamic_metrics,
    evaluate_dynamic_frame,
)
from src.evaluation.baselines.tesse_semantics import (  # noqa: E402
    TesseSemanticCrosswalk,
    load_tesse_semantic_crosswalk,
)
from src.evaluation.exporters.oviovo import read_map_snapshot  # noqa: E402


KHRONOS_CLIP_WEIGHT_SHA256 = (
    "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
)
CAUSAL_SNAPSHOT_METHOD_LABELS = {
    "DUALMAP": frozenset({"DUALMAP", "DualMap"}),
    "OVIV2": frozenset({"OVIV2"}),
    "PANOPTIC_SHARED": frozenset(
        {"PANOPTIC_SHARED", "Panoptic Mapping + shared masks"}
    ),
}


@dataclass(frozen=True)
class KhronosTextHead:
    semantic_ids: np.ndarray
    names: tuple[str, ...]
    embeddings: np.ndarray
    manifest_record: Mapping[str, object]
    arrays_record: Mapping[str, object]


def _snapshot_method_matches(method: str, snapshot_method: str) -> bool:
    return snapshot_method in CAUSAL_SNAPSHOT_METHOD_LABELS.get(
        method, frozenset({method})
    )


def classify_khronos_open_embedding(
    embedding: np.ndarray | None, head: KhronosTextHead
) -> str | None:
    if embedding is None:
        return None
    feature = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if feature.size == 0:
        return None
    if (
        feature.size != head.embeddings.shape[1]
        or not np.all(np.isfinite(feature))
    ):
        raise ValueError("Khronos open embedding dimension or finite values are invalid")
    norm = float(np.linalg.norm(feature.astype(np.float64)))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Khronos open embedding is zero or non-finite")
    scores = head.embeddings @ np.asarray(feature / norm, dtype=np.float32)
    return head.names[int(np.argmax(scores))]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if resolved != path.absolute() or not resolved.is_file():
        raise ValueError(f"source must be a direct regular file: {path}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "byte_count": resolved.stat().st_size,
    }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    return dict(_mapping(payload, label))


def _declared_file(
    declaration: object,
    *,
    label: str,
    base: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    source = _mapping(declaration, f"{label} declaration")
    raw = Path(str(source.get("path", "")))
    path = raw if raw.is_absolute() else (base / raw if base is not None else raw)
    record = _file_record(path)
    if source.get("sha256") != record["sha256"]:
        raise ValueError(f"{label} SHA256 mismatch")
    if "byte_count" in source and source.get("byte_count") != record["byte_count"]:
        raise ValueError(f"{label} byte count mismatch")
    return path, record


def _validate_index(index: Mapping[str, Any]) -> None:
    if not (
        index.get("schema_version") == 1
        and index.get("manifest_id")
        == "tesse_cd_common_v2_frozen_temporal_artifact"
        and index.get("dataset") == "TESSE-CD"
        and index.get("protocol") == "tesse_cd_common_v2"
        and index.get("status") == "PASS"
        and index.get("mode") == "frozen"
        and index.get("updates_after_freeze") == 0
        and index.get("prediction_semantics")
        == "pre_intervention_snapshot_reused_without_updates"
    ):
        raise ValueError("frozen temporal index identity mismatch")
    if index.get("method") not in {"OVIMAP_FROZEN", "CONCEPTGRAPHS_FROZEN"}:
        raise ValueError("unsupported frozen temporal method")
    if index.get("scene") not in {"apartment", "office"}:
        raise ValueError("unsupported TESSE-CD scene")


def _load_schedule(
    path: Path, *, scene: str
) -> tuple[list[Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    payload = _load_json(path, "schedule")
    if not (
        payload.get("schema_version") == 2
        and payload.get("manifest_id") == "tesse_cd_causal_schedule_v2"
        and payload.get("dataset") == "TESSE-CD"
        and payload.get("method_predictions_used") is False
    ):
        raise ValueError("common-v2 schedule identity mismatch")
    scene_payload = _mapping(
        _mapping(payload.get("scenes"), "schedule scenes").get(scene),
        f"schedule scene {scene}",
    )
    events = scene_payload.get("events")
    entries = scene_payload.get("entries")
    if not isinstance(events, list) or not isinstance(entries, list):
        raise ValueError("schedule scene requires event and checkpoint lists")
    common: dict[int, Mapping[str, Any]] = {}
    for raw_entry in entries:
        entry = _mapping(raw_entry, "schedule entry")
        if "common_v2" not in entry.get("roles", ()):
            continue
        frame = entry.get("frame_index")
        if type(frame) is not int or frame in common:
            raise ValueError("common-v2 schedule frame IDs must be unique integers")
        common[frame] = entry
    return [_mapping(event, "schedule event") for event in events], common


def _validate_checkpoints(
    index: Mapping[str, Any],
    *,
    common: Mapping[int, Mapping[str, Any]],
) -> None:
    checkpoints = index.get("checkpoints")
    freeze_stop = index.get("freeze_stop_exclusive")
    if (
        not isinstance(checkpoints, list)
        or len(checkpoints) != index.get("checkpoint_count")
        or type(freeze_stop) is not int
    ):
        raise ValueError("frozen temporal checkpoint identity mismatch")
    observed: dict[int, Mapping[str, Any]] = {}
    for raw in checkpoints:
        checkpoint = _mapping(raw, "frozen checkpoint")
        frame = checkpoint.get("frame_index")
        if type(frame) is not int or frame in observed or frame not in common:
            raise ValueError("frozen checkpoints do not exactly cover common schedule")
        scheduled = common[frame]
        if (
            checkpoint.get("timestamp_ns") != scheduled.get("timestamp_ns")
            or checkpoint.get("event_ids") != scheduled.get("event_ids")
            or checkpoint.get("snapshot_source") != "frozen_snapshot"
            or checkpoint.get("consumed_through_frame_exclusive") != freeze_stop
        ):
            raise ValueError("frozen checkpoint has future state or freeze boundary mismatch")
        observed[frame] = checkpoint
    if set(observed) != set(common):
        raise ValueError("frozen checkpoints do not exactly cover common schedule")


def _load_targets(
    manifest_path: Path,
    *,
    schedule_record: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, Any]]:
    manifest = _load_json(manifest_path, "target manifest")
    metadata = _mapping(manifest.get("metadata"), "target metadata")
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("manifest_id") == "tesse_cd_common_v2_targets"
        and manifest.get("dataset") == "TESSE-CD"
        and manifest.get("status") == "GENERATED"
        and manifest.get("targets_generated") is True
        and manifest.get("prediction_inputs_used") is False
        and metadata.get("protocol_complete") is True
        and metadata.get("window_frames") == 450
        and metadata.get("voxel_size_m") == 0.05
        and set(metadata.get("scenes", ())) == {"apartment", "office"}
    ):
        raise ValueError("target package must be complete GENERATED common-v2 targets")
    declared_schedule = _mapping(metadata.get("schedule"), "target schedule")
    if (
        declared_schedule.get("sha256") != schedule_record["sha256"]
        or Path(str(declared_schedule.get("path", ""))).resolve()
        != Path(str(schedule_record["path"])).resolve()
    ):
        raise ValueError("target package schedule binding mismatch")

    declaration = _mapping(manifest.get("target_arrays"), "target arrays")
    arrays_path, arrays_record = _declared_file(
        declaration,
        label="target arrays",
        base=manifest_path.parent,
    )
    declared_arrays = _mapping(declaration.get("arrays"), "target array declarations")
    arrays: dict[str, np.ndarray] = {}
    with np.load(arrays_path, allow_pickle=False) as bundle:
        if set(bundle.files) != set(declared_arrays):
            raise ValueError("target array names disagree with manifest")
        for name in sorted(bundle.files):
            value = np.asarray(bundle[name])
            declared = _mapping(declared_arrays[name], f"target array {name}")
            if (
                list(value.shape) != declared.get("shape")
                or str(value.dtype) != declared.get("dtype")
                or int(value.size) != declared.get("element_count")
                or not np.issubdtype(value.dtype, np.integer)
                or (value.size and not np.all(np.isfinite(value)))
            ):
                raise ValueError(f"target array declaration mismatch: {name}")
            arrays[name] = np.array(value, copy=True)
    if len(arrays) != declaration.get("count"):
        raise ValueError("target array count disagrees with manifest")
    return arrays, arrays_record, dict(metadata)


def load_khronos_text_head(
    manifest_path: Path,
    *,
    scene: str,
    label_space_record: Mapping[str, object],
    aliases_record: Mapping[str, object],
    valid_semantic_ids: frozenset[int],
) -> tuple[KhronosTextHead, dict[str, dict[str, object]]]:
    manifest_record = _file_record(manifest_path)
    payload = _load_json(manifest_path, "Khronos text-head manifest")
    if not (
        payload.get("schema_version") == 1
        and payload.get("manifest_id") == "khronos_tesse_clip_text_head_v1"
        and payload.get("status") == "GENERATED"
        and payload.get("dataset") == "TESSE-CD"
        and payload.get("scene") == scene
        and payload.get("method") == "KHRONOS_OPEN"
        and payload.get("model_name") == "ViT-L/14"
        and payload.get("weight_sha256") == KHRONOS_CLIP_WEIGHT_SHA256
        and payload.get("feature_dimension") == 768
        and payload.get("similarity") == "cosine"
        and payload.get("normalization") == "l2"
        and payload.get("prompt_policy")
        == "canonical_native_object_names_verbatim"
        and payload.get("prediction_inputs_used") is False
        and payload.get("official_text_path_consistency") == "byte_identical"
    ):
        raise ValueError("Khronos text head identity mismatch")
    source_declarations = _mapping(payload.get("sources"), "text-head sources")
    required_sources = {
        "label_space",
        "aliases",
        "model_config",
        "run_log",
        "weight",
        "builder",
        "semantic_inference.wrappers",
        "semantic_inference.openset_segmenter",
        "clip_package.__init__.py",
        "clip_package.clip.py",
        "clip_package.model.py",
        "clip_package.simple_tokenizer.py",
        "clip_package.bpe_simple_vocab_16e6.txt.gz",
    }
    if not required_sources <= set(source_declarations):
        raise ValueError("Khronos text head source coverage is incomplete")
    source_records: dict[str, dict[str, object]] = {}
    for label, declaration in source_declarations.items():
        _, record = _declared_file(
            declaration,
            label=f"text-head source {label}",
            base=manifest_path.parent,
        )
        source_records[f"text_head.{label}"] = record
    if (
        source_records["text_head.label_space"] != label_space_record
        or source_records["text_head.aliases"] != aliases_record
        or source_records["text_head.weight"]["sha256"]
        != KHRONOS_CLIP_WEIGHT_SHA256
    ):
        raise ValueError("Khronos text head label or weight binding mismatch")

    arrays_path, arrays_record = _declared_file(
        payload.get("arrays"),
        label="Khronos text-head arrays",
        base=manifest_path.parent,
    )
    with np.load(arrays_path, allow_pickle=False) as bundle:
        if set(bundle.files) != {"semantic_ids", "names", "embeddings"}:
            raise ValueError("Khronos text-head arrays have unexpected keys")
        semantic_ids = np.asarray(bundle["semantic_ids"], dtype=np.int64)
        raw_names = np.asarray(bundle["names"])
        embeddings = np.asarray(bundle["embeddings"], dtype=np.float32)
    names = tuple(str(value) for value in raw_names.tolist())
    if (
        semantic_ids.ndim != 1
        or embeddings.shape != (len(semantic_ids), 768)
        or raw_names.ndim != 1
        or len(names) != len(semantic_ids)
        or len(set(names)) != len(names)
        or not np.all(np.isfinite(embeddings))
        or set(int(value) for value in semantic_ids) != set(valid_semantic_ids)
        or semantic_ids.tolist() != payload.get("candidate_semantic_ids")
        or list(names) != payload.get("prompts")
        or not np.allclose(
            np.linalg.norm(embeddings.astype(np.float64), axis=1),
            1.0,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise ValueError("Khronos text-head arrays or vocabulary mismatch")
    head = KhronosTextHead(
        semantic_ids=semantic_ids,
        names=names,
        embeddings=embeddings,
        manifest_record=manifest_record,
        arrays_record=arrays_record,
    )
    return head, {
        "text_head.manifest": manifest_record,
        "text_head.arrays": arrays_record,
        **source_records,
    }


def _background_unobservable_events(
    events: Iterable[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> set[str]:
    event_ids = {str(event.get("event_id", "")) for event in events}
    if "" in event_ids:
        raise ValueError("event IDs must be non-empty")
    observability = metadata.get("background_observable_by_event")
    if (
        not isinstance(observability, Mapping)
        or "" in observability
        or not event_ids <= set(observability)
    ):
        raise ValueError("target background observability metadata mismatch")
    if any(type(value) is not bool for value in observability.values()):
        raise ValueError("target background observability values must be booleans")
    for event_id in event_ids:
        observable = observability[event_id]
        target_name = f"{event_id}.revealed_background"
        if target_name not in arrays or bool(len(arrays[target_name])) is not observable:
            raise ValueError("target background observability disagrees with arrays")
    global_observable_count = sum(observability.values())
    if (
        metadata.get("background_observable_event_count") != global_observable_count
        or metadata.get("unobservable_revealed_target_event_count")
        != len(observability) - global_observable_count
    ):
        raise ValueError("target background observability counts disagree")
    if not any(observability[event_id] for event_id in event_ids):
        raise ValueError("scene must contain at least one background-observable event")
    return {
        str(event_id)
        for event_id in event_ids
        if not observability[event_id]
    }


def _voxel_centers(keys: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    values = np.asarray(keys, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("voxel target must have shape (N, 3)")
    return ((values.astype(np.float64) + 0.5) * voxel_size).astype(np.float32)


def _crop_points_to_voxels(
    chunks: Iterable[np.ndarray], region_keys: np.ndarray, *, voxel_size: float = 0.05
) -> np.ndarray:
    region = np.asarray(region_keys, dtype=np.int64)
    if region.ndim != 2 or region.shape[1] != 3 or len(region) == 0:
        raise ValueError("event region must be a non-empty (N, 3) voxel array")
    lower = np.min(region, axis=0)
    upper = np.max(region, axis=0)
    spans = upper - lower + 1
    if int(spans[0]) * int(spans[1]) * int(spans[2]) > np.iinfo(np.int64).max:
        raise ValueError("event region is too large to encode")
    region_relative = region - lower
    region_codes = (
        (region_relative[:, 0] * spans[1] + region_relative[:, 1]) * spans[2]
        + region_relative[:, 2]
    )
    minimum_point = lower.astype(np.float64) * voxel_size
    maximum_point = (upper.astype(np.float64) + 1.0) * voxel_size
    selected: list[np.ndarray] = []
    for raw in chunks:
        points = np.asarray(raw, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("prediction point chunk must have shape (N, 3)")
        bounds = np.all(
            (points >= minimum_point) & (points < maximum_point), axis=1
        )
        candidates = points[bounds]
        if len(candidates) == 0:
            continue
        keys = np.floor(candidates.astype(np.float64) / voxel_size).astype(np.int64)
        relative = keys - lower
        codes = (
            (relative[:, 0] * spans[1] + relative[:, 1]) * spans[2]
            + relative[:, 2]
        )
        keep = np.isin(codes, region_codes, assume_unique=False)
        if np.any(keep):
            selected.append(candidates[keep])
    return (
        np.concatenate(selected, axis=0)
        if selected
        else np.empty((0, 3), dtype=np.float32)
    )


def _prediction_semantics(
    object_points: np.ndarray,
    point_semantic_ids: np.ndarray,
    gt_voxels: np.ndarray,
    *,
    tree: cKDTree | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(gt_voxels, dtype=np.int64)
    if targets.ndim != 2 or targets.shape[1] != 4:
        raise ValueError("current semantic target must have shape (N, 4)")
    gt_ids = targets[:, 3].astype(np.int64, copy=False)
    prediction = np.zeros(len(targets), dtype=np.int64)
    if len(object_points) and len(targets):
        semantic_tree = tree if tree is not None else cKDTree(object_points)
        distances, indices = semantic_tree.query(
            _voxel_centers(targets[:, :3]), k=1, workers=-1
        )
        matched = np.asarray(distances, dtype=np.float64) < 0.05
        prediction[matched] = point_semantic_ids[np.asarray(indices)[matched]]
    return gt_ids, prediction


def _write_output(output: Path, payload: Mapping[str, Any]) -> Path:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        summary = temporary / "summary.json"
        data = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        with summary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "summary.json"


def evaluate_frozen_common_v2(
    temporal_index: Path, target_manifest: Path, output: Path
) -> Path:
    index_record = _file_record(temporal_index)
    target_manifest_record = _file_record(target_manifest)
    index = _load_json(temporal_index, "frozen temporal index")
    _validate_index(index)
    scene = str(index["scene"])
    method = str(index["method"])
    sources = _mapping(index.get("sources"), "frozen index sources")
    schedule_path, schedule_record = _declared_file(
        sources.get("schedule"), label="schedule"
    )
    aliases_path, aliases_record = _declared_file(
        sources.get("aliases"), label="aliases"
    )
    label_space_path, label_space_record = _declared_file(
        sources.get("label_space"), label="label space"
    )
    events, common = _load_schedule(schedule_path, scene=scene)
    _validate_checkpoints(index, common=common)
    arrays, target_arrays_record, target_metadata = _load_targets(
        target_manifest, schedule_record=schedule_record
    )
    unobservable_events = _background_unobservable_events(
        events, arrays, target_metadata
    )

    frozen = _mapping(index.get("frozen_snapshot"), "frozen snapshot")
    snapshot_path, snapshot_record = _declared_file(
        frozen.get("snapshot"), label="frozen snapshot"
    )
    entities_path, entities_record = _declared_file(
        frozen.get("entities"), label="frozen entities"
    )
    snapshot = read_map_snapshot(snapshot_path, entities_path)
    if snapshot.scene_id != scene or snapshot.scope != "current":
        raise ValueError("frozen snapshot identity mismatch")
    expected_snapshot_method = {
        "OVIMAP_FROZEN": "OVI-MAP (frozen)",
        "CONCEPTGRAPHS_FROZEN": "ConceptGraphs (frozen)",
    }[method]
    if snapshot.method != expected_snapshot_method:
        raise ValueError("frozen snapshot method mismatch")
    crosswalk: TesseSemanticCrosswalk = load_tesse_semantic_crosswalk(
        aliases_path, scene, label_space_path
    )

    point_chunks: list[np.ndarray] = []
    semantic_chunks: list[np.ndarray] = []
    for entity in snapshot.entities:
        points = np.asarray(entity.points_xyz, dtype=np.float32)
        lookup = (
            crosswalk.lookup(entity.semantic_label)
            if entity.semantic_label is not None
            else crosswalk.unknown
        )
        point_chunks.append(points)
        semantic_chunks.append(
            np.full(len(points), lookup.semantic_id, dtype=np.int64)
        )
    object_points = (
        np.concatenate(point_chunks, axis=0)
        if point_chunks
        else np.empty((0, 3), dtype=np.float32)
    )
    point_semantic_ids = (
        np.concatenate(semantic_chunks, axis=0)
        if semantic_chunks
        else np.empty((0,), dtype=np.int64)
    )
    semantic_tree = cKDTree(object_points) if len(object_points) else None
    background_chunks = (
        [np.asarray(snapshot.background_xyz, dtype=np.float32)]
        if snapshot.background_xyz is not None
        else []
    )

    semantic_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    event_region_predictions: dict[str, np.ndarray] = {}
    event_background_predictions: dict[str, np.ndarray] = {}
    evaluations: list[DynamicFrameMetrics] = []
    intervention_by_event: dict[str, int] = {}
    ordered_events: list[tuple[str, int]] = []
    for event in events:
        event_id = str(event.get("event_id", ""))
        intervention = event.get("intervention_frame_index")
        frame_ids = event.get("common_checkpoint_frame_indices")
        if (
            not event_id
            or type(intervention) is not int
            or not isinstance(frame_ids, list)
            or frame_ids != list(range(intervention, intervention + 451, 50))
        ):
            raise ValueError("event does not use the fixed common-v2 checkpoint grid")
        intervention_by_event[event_id] = intervention
        ordered_events.append((event_id, intervention))
        region_name = f"{event_id}.region"
        background_name = f"{event_id}.revealed_background"
        if region_name not in arrays or background_name not in arrays:
            raise ValueError(f"target package is missing event arrays for {event_id}")
        region = arrays[region_name]
        event_region_predictions[event_id] = _crop_points_to_voxels(
            point_chunks, region
        )
        event_background_predictions[event_id] = _crop_points_to_voxels(
            background_chunks, region
        )
        target_background = _voxel_centers(arrays[background_name])
        for frame in frame_ids:
            if frame not in common:
                raise ValueError(f"event checkpoint is missing from schedule: {frame}")
            semantic_name = f"{scene}.current_semantic.{frame:06d}"
            free_name = f"{event_id}.confirmed_free.{frame:06d}"
            if semantic_name not in arrays or free_name not in arrays:
                raise ValueError(f"target package is missing checkpoint arrays for {frame}")
            if frame not in semantic_cache:
                semantic_cache[frame] = _prediction_semantics(
                    object_points,
                    point_semantic_ids,
                    arrays[semantic_name],
                    tree=semantic_tree,
                )
            gt_ids, predicted_ids = semantic_cache[frame]
            evaluations.append(
                evaluate_dynamic_frame(
                    event_id=event_id,
                    frame_id=frame,
                    intervention_frame_id=intervention,
                    ground_truth_semantic_ids=gt_ids,
                    predicted_semantic_ids=predicted_ids,
                    valid_semantic_ids=crosswalk.valid_semantic_ids,
                    predicted_object_points_in_changed_region=event_region_predictions[
                        event_id
                    ],
                    confirmed_free_space_points=_voxel_centers(arrays[free_name]),
                    predicted_background_points_in_revealed_region=event_background_predictions[
                        event_id
                    ],
                    ground_truth_revealed_background_points=target_background,
                    distance_threshold_m=0.05,
                )
            )

    overlap_censors = {
        event_id: next_intervention
        for (event_id, intervention), (_, next_intervention) in zip(
            ordered_events, ordered_events[1:]
        )
        if next_intervention <= intervention + 450
    }
    metrics = aggregate_dynamic_metrics(
        evaluations,
        recovery_background_f5=0.9,
        recovery_consecutive=3,
        checkpoint_step_frames=50,
        recovery_horizon_frames=450,
        overlap_censor_frames=overlap_censors,
        background_unobservable_events=unobservable_events,
    )
    for name in ("current_miou", "ghost_rate", "background_f5", "recovery_frames"):
        if not math.isfinite(float(metrics[name])):
            raise ValueError(f"non-finite common-v2 metric: {name}")

    source_records = {
        "temporal_index": index_record,
        "target_manifest": target_manifest_record,
        "target_arrays": target_arrays_record,
        "schedule": schedule_record,
        "aliases": aliases_record,
        "label_space": label_space_record,
        "snapshot": snapshot_record,
        "entities": entities_record,
        "evaluator": _file_record(Path(__file__)),
    }
    for label, record in source_records.items():
        if _file_record(Path(str(record["path"]))) != record:
            raise ValueError(f"{label} changed during common-v2 evaluation")
    payload = {
        "schema_version": 1,
        "manifest_id": "tesse_cd_common_v2_scene_summary",
        "dataset": "TESSE-CD",
        "protocol": "tesse_cd_common_v2",
        "status": "PASS",
        "method": method,
        "mode": "frozen",
        "scene": scene,
        "metrics": metrics,
        "frames": [asdict(value) for value in evaluations],
        "event_region_prediction_counts": {
            event_id: len(points)
            for event_id, points in sorted(event_region_predictions.items())
        },
        "event_background_prediction_counts": {
            event_id: len(points)
            for event_id, points in sorted(event_background_predictions.items())
        },
        "sources": source_records,
    }
    return _write_output(output, payload)


def evaluate_common_v2(
    temporal_index: Path,
    target_manifest: Path,
    aliases: Path,
    label_space: Path,
    output: Path,
    semantic_head: Path | None = None,
) -> Path:
    index_record = _file_record(temporal_index)
    target_manifest_record = _file_record(target_manifest)
    aliases_record = _file_record(aliases)
    label_space_record = _file_record(label_space)
    index = _load_json(temporal_index, "causal temporal index")
    scene = str(index.get("scene", ""))
    method = str(index.get("method", "")).strip()
    if not (
        index.get("schema_version") == 1
        and index.get("dataset") == "TESSE-CD"
        and index.get("mode") == "causal_checkpoints"
        and scene in {"apartment", "office"}
        and method
    ):
        raise ValueError("causal temporal index identity mismatch")
    sources = _mapping(index.get("sources"), "causal index sources")
    schedule_path, schedule_record = _declared_file(
        sources.get("schedule"),
        label="schedule",
        base=temporal_index.parent,
    )
    events, common = _load_schedule(schedule_path, scene=scene)
    arrays, target_arrays_record, target_metadata = _load_targets(
        target_manifest, schedule_record=schedule_record
    )
    unobservable_events = _background_unobservable_events(
        events, arrays, target_metadata
    )
    crosswalk = load_tesse_semantic_crosswalk(aliases, scene, label_space)
    text_head: KhronosTextHead | None = None
    text_head_sources: dict[str, dict[str, object]] = {}
    if method == "KHRONOS_OPEN":
        if semantic_head is None:
            raise ValueError("KHRONOS_OPEN requires a frozen hash-bound text head")
        text_head, text_head_sources = load_khronos_text_head(
            semantic_head,
            scene=scene,
            label_space_record=label_space_record,
            aliases_record=aliases_record,
            valid_semantic_ids=crosswalk.valid_semantic_ids,
        )
    elif semantic_head is not None:
        raise ValueError("semantic text head is only valid for KHRONOS_OPEN")

    raw_checkpoints = index.get("checkpoints")
    if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
        raise ValueError("causal temporal checkpoints must be non-empty")
    checkpoints: dict[int, dict[str, Any]] = {}
    checkpoint_sources: dict[str, dict[str, object]] = {}
    previous_frame = -1
    previous_timestamp = -1
    for raw_checkpoint in raw_checkpoints:
        checkpoint = _mapping(raw_checkpoint, "causal checkpoint")
        frame = checkpoint.get("frame_index")
        timestamp = checkpoint.get("timestamp_ns")
        consumed = checkpoint.get("consumed_through_frame")
        consumed_exclusive = checkpoint.get("consumed_through_frame_exclusive")
        if (
            type(frame) is not int
            or type(timestamp) is not int
            or type(consumed) is not int
            or type(consumed_exclusive) is not int
            or frame <= previous_frame
            or timestamp <= previous_timestamp
        ):
            raise ValueError("causal checkpoints must have increasing integer identity")
        if consumed != frame or consumed_exclusive != frame + 1:
            raise ValueError(
                "causal checkpoint future or exclusive freeze boundary mismatch"
            )
        previous_frame = frame
        previous_timestamp = timestamp
        if frame not in common:
            continue
        if timestamp != common[frame].get("timestamp_ns"):
            raise ValueError("causal checkpoint timestamp disagrees with schedule")
        snapshot_path, snapshot_record = _declared_file(
            checkpoint.get("snapshot"),
            label=f"checkpoint {frame} snapshot",
            base=temporal_index.parent,
        )
        entities_path, entities_record = _declared_file(
            checkpoint.get("entities"),
            label=f"checkpoint {frame} entities",
            base=temporal_index.parent,
        )
        snapshot = read_map_snapshot(snapshot_path, entities_path)
        if (
            snapshot.scene_id != scene
            or snapshot.scope != "current"
                or not _snapshot_method_matches(method, snapshot.method)
            or not math.isfinite(float(snapshot.timestamp))
            or not (
                float(snapshot.timestamp) == float(timestamp)
                or math.isclose(
                    float(snapshot.timestamp),
                    timestamp / 1e9,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
        ):
            raise ValueError("causal checkpoint snapshot identity mismatch")
        checkpoints[frame] = {
            "snapshot": snapshot,
            "snapshot_path": snapshot_path,
            "entities_path": entities_path,
        }
        checkpoint_sources[f"snapshot.{frame:06d}"] = snapshot_record
        checkpoint_sources[f"entities.{frame:06d}"] = entities_record
    if set(checkpoints) != set(common):
        raise ValueError("causal checkpoints do not exactly cover common schedule")

    prediction_cache: dict[
        int, tuple[list[np.ndarray], np.ndarray, np.ndarray, list[np.ndarray]]
    ] = {}
    semantic_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    evaluations: list[DynamicFrameMetrics] = []
    region_counts: dict[str, dict[str, int]] = {}
    background_counts: dict[str, dict[str, int]] = {}
    ordered_events: list[tuple[str, int]] = []
    for event in events:
        event_id = str(event.get("event_id", ""))
        intervention = event.get("intervention_frame_index")
        frame_ids = event.get("common_checkpoint_frame_indices")
        if (
            not event_id
            or type(intervention) is not int
            or not isinstance(frame_ids, list)
            or frame_ids != list(range(intervention, intervention + 451, 50))
        ):
            raise ValueError("event does not use the fixed common-v2 checkpoint grid")
        ordered_events.append((event_id, intervention))
        region_name = f"{event_id}.region"
        revealed_name = f"{event_id}.revealed_background"
        if region_name not in arrays or revealed_name not in arrays:
            raise ValueError(f"target package is missing event arrays for {event_id}")
        target_background = _voxel_centers(arrays[revealed_name])
        region_counts[event_id] = {}
        background_counts[event_id] = {}
        for frame in frame_ids:
            if frame not in checkpoints:
                raise ValueError(f"event checkpoint is missing from temporal index: {frame}")
            if frame not in prediction_cache:
                snapshot = checkpoints[frame]["snapshot"]
                point_chunks: list[np.ndarray] = []
                semantic_chunks: list[np.ndarray] = []
                for entity in snapshot.entities:
                    points = np.asarray(entity.points_xyz, dtype=np.float32)
                    if method == "KHRONOS_OPEN":
                        if (
                            entity.semantic_label is not None
                            or entity.metadata.get("semantic_label_source")
                            != "open_set_instance_mask_id_not_class_label"
                            or text_head is None
                        ):
                            raise ValueError(
                                "KHRONOS_OPEN snapshot contains a numeric-derived class label"
                            )
                        predicted_name = classify_khronos_open_embedding(
                            entity.semantic_embedding, text_head
                        )
                        lookup = (
                            crosswalk.lookup(predicted_name)
                            if predicted_name is not None
                            else crosswalk.unknown
                        )
                    else:
                        lookup = (
                            crosswalk.lookup(entity.semantic_label)
                            if entity.semantic_label is not None
                            else crosswalk.unknown
                        )
                    point_chunks.append(points)
                    semantic_chunks.append(
                        np.full(len(points), lookup.semantic_id, dtype=np.int64)
                    )
                object_points = (
                    np.concatenate(point_chunks, axis=0)
                    if point_chunks
                    else np.empty((0, 3), dtype=np.float32)
                )
                semantic_ids = (
                    np.concatenate(semantic_chunks, axis=0)
                    if semantic_chunks
                    else np.empty((0,), dtype=np.int64)
                )
                background_chunks = (
                    [np.asarray(snapshot.background_xyz, dtype=np.float32)]
                    if snapshot.background_xyz is not None
                    else []
                )
                prediction_cache[frame] = (
                    point_chunks,
                    object_points,
                    semantic_ids,
                    background_chunks,
                )
            point_chunks, object_points, semantic_ids, background_chunks = (
                prediction_cache[frame]
            )
            semantic_name = f"{scene}.current_semantic.{frame:06d}"
            free_name = f"{event_id}.confirmed_free.{frame:06d}"
            if semantic_name not in arrays or free_name not in arrays:
                raise ValueError(f"target package is missing checkpoint arrays for {frame}")
            if frame not in semantic_cache:
                semantic_cache[frame] = _prediction_semantics(
                    object_points,
                    semantic_ids,
                    arrays[semantic_name],
                )
            predicted_region = _crop_points_to_voxels(
                point_chunks, arrays[region_name]
            )
            predicted_background = _crop_points_to_voxels(
                background_chunks, arrays[region_name]
            )
            region_counts[event_id][str(frame)] = len(predicted_region)
            background_counts[event_id][str(frame)] = len(predicted_background)
            gt_ids, predicted_ids = semantic_cache[frame]
            evaluations.append(
                evaluate_dynamic_frame(
                    event_id=event_id,
                    frame_id=frame,
                    intervention_frame_id=intervention,
                    ground_truth_semantic_ids=gt_ids,
                    predicted_semantic_ids=predicted_ids,
                    valid_semantic_ids=crosswalk.valid_semantic_ids,
                    predicted_object_points_in_changed_region=predicted_region,
                    confirmed_free_space_points=_voxel_centers(arrays[free_name]),
                    predicted_background_points_in_revealed_region=predicted_background,
                    ground_truth_revealed_background_points=target_background,
                    distance_threshold_m=0.05,
                )
            )

    overlap_censors = {
        event_id: next_intervention
        for (event_id, intervention), (_, next_intervention) in zip(
            ordered_events, ordered_events[1:]
        )
        if next_intervention <= intervention + 450
    }
    metrics = aggregate_dynamic_metrics(
        evaluations,
        recovery_background_f5=0.9,
        recovery_consecutive=3,
        checkpoint_step_frames=50,
        recovery_horizon_frames=450,
        overlap_censor_frames=overlap_censors,
        background_unobservable_events=unobservable_events,
    )
    for name in ("current_miou", "ghost_rate", "background_f5", "recovery_frames"):
        if not math.isfinite(float(metrics[name])):
            raise ValueError(f"non-finite common-v2 metric: {name}")

    source_records = {
        "temporal_index": index_record,
        "target_manifest": target_manifest_record,
        "target_arrays": target_arrays_record,
        "schedule": schedule_record,
        "aliases": aliases_record,
        "label_space": label_space_record,
        "evaluator": _file_record(Path(__file__)),
        **text_head_sources,
        **checkpoint_sources,
    }
    for label, record in source_records.items():
        if _file_record(Path(str(record["path"]))) != record:
            raise ValueError(f"{label} changed during common-v2 evaluation")
    payload = {
        "schema_version": 1,
        "manifest_id": "tesse_cd_common_v2_scene_summary",
        "dataset": "TESSE-CD",
        "protocol": "tesse_cd_common_v2",
        "status": "PASS",
        "method": method,
        "mode": "causal_checkpoints",
        "scene": scene,
        "metrics": metrics,
        "frames": [asdict(value) for value in evaluations],
        "event_region_prediction_counts": region_counts,
        "event_background_prediction_counts": background_counts,
        "sources": source_records,
    }
    return _write_output(output, payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-index", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--label-space", type=Path)
    parser.add_argument("--semantic-head", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    temporal = _load_json(args.temporal_index, "temporal index")
    if temporal.get("mode") == "causal_checkpoints":
        if args.aliases is None or args.label_space is None:
            parser.error("causal checkpoints require --aliases and --label-space")
        evaluate_common_v2(
            args.temporal_index,
            args.target_manifest,
            args.aliases,
            args.label_space,
            args.output,
            args.semantic_head,
        )
    else:
        evaluate_frozen_common_v2(
            args.temporal_index, args.target_manifest, args.output
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
