#!/usr/bin/env python3
"""Compose a frozen OVI-MAP anchor with an existing causal CROVE capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.exporters.oviovo import read_map_snapshot, write_map_snapshot
from src.evaluation.crove_anchor_counterfactual import CounterfactualVariant
from src.oviv2.anchor_visibility import (
    AnchorVisibilityConfig,
    AnchorVisibilityEvidenceKind,
    AnchorVisibilityState,
    advance_anchor_visibility,
    anchor_visibility_config_from_json,
    classify_anchor_visibility,
    sample_anchor_voxels,
)
from src.oviv2.ovimap_static_anchor import (
    PrefixIdentitySample,
    StaticAnchorConfig,
    bind_anchor_identities,
    compose_anchor_checkpoint,
)
from src.oviv2.temporal_export import (
    DynamicState,
    TemporalExportBatch,
    TemporalExportSample,
    TemporalLifecycleEvent,
    validate_temporal_export_sequence,
)
from src.oviv2.temporal_lifecycle import TemporalEvidenceKind, TemporalLifecycle

_RECORD_KEYS = {"path", "sha256", "byte_count"}
_MOVED_GEOMETRY_MODES = (
    "anchor_centroid_translation",
    "temporal_compact",
)
_READOUT_ROLES = (
    "causal_visibility_candidate",
    "counterfactual_diagnostic",
    "evaluation_candidate",
    "formal_baseline",
    "visualization_shadow",
)
_ALLOWED_READOUT_CONTRACTS = frozenset(
    {
        ("temporal_compact", "formal_baseline"),
        ("anchor_centroid_translation", "visualization_shadow"),
        ("anchor_centroid_translation", "evaluation_candidate"),
        ("anchor_centroid_translation", "causal_visibility_candidate"),
        ("temporal_compact", "causal_visibility_candidate"),
        ("temporal_compact", "counterfactual_diagnostic"),
    }
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink path component is not allowed: {current}")


def _regular_bytes(path: Path, *, label: str) -> bytes:
    path = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_components(path)
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    data = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(data) != after.st_size
    ):
        raise RuntimeError(f"{label} changed while reading")
    return data


def _input_record(path: Path, data: bytes | None = None) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    content = _regular_bytes(absolute, label="input") if data is None else data
    return {
        "path": str(absolute),
        "sha256": _sha256(content),
        "byte_count": len(content),
    }


def _relative_record(path: Path, *, root: Path) -> dict[str, object]:
    content = _regular_bytes(path, label="published artifact")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(content),
        "byte_count": len(content),
    }


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _readout_contract(
    mode: object,
    role: object,
    *,
    visibility_enabled: bool = False,
    counterfactual_variant: CounterfactualVariant | None = None,
) -> dict[str, object]:
    if type(visibility_enabled) is not bool:
        raise TypeError("visibility_enabled must be bool")
    if counterfactual_variant is not None and not isinstance(
        counterfactual_variant, CounterfactualVariant
    ):
        raise TypeError("counterfactual_variant must be CounterfactualVariant")
    diagnostic = counterfactual_variant is not None
    if (role == "counterfactual_diagnostic") is not diagnostic:
        raise ValueError(
            "counterfactual diagnostic role and variant must be paired"
        )
    visibility_role = role in {
        "causal_visibility_candidate",
        "counterfactual_diagnostic",
    }
    if visibility_role is not visibility_enabled:
        raise ValueError(
            "causal visibility candidate role and visibility policy must be paired"
        )
    if (
        not isinstance(mode, str)
        or not isinstance(role, str)
        or (mode, role) not in _ALLOWED_READOUT_CONTRACTS
    ):
        raise ValueError("moved geometry mode and readout role are incompatible")
    result = {"moved_geometry_mode": mode, "readout_role": role}
    if role == "counterfactual_diagnostic":
        result.update(
            {
                "unbound_anchor_mode": "causal_visibility_filtered",
                "diagnostic_only": True,
                "promotion_eligible": False,
            }
        )
    elif visibility_enabled:
        result["unbound_anchor_mode"] = "causal_visibility"
    return result


def _bound_path(
    *, root: Path, record: object, label: str, absolute: bool = False
) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise ValueError(f"{label} record schema is invalid")
    raw_path = record.get("path")
    digest = record.get("sha256")
    byte_count = record.get("byte_count")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise ValueError(f"{label} record values are invalid")
    declared = Path(raw_path)
    if absolute:
        if not declared.is_absolute():
            raise ValueError(f"{label} path must be absolute")
        path = declared
    else:
        if declared.is_absolute() or ".." in declared.parts:
            raise ValueError(f"{label} path must stay within its run root")
        path = root / declared
    data = _regular_bytes(path, label=label)
    observed = _input_record(path, data)
    compared = dict(observed)
    if not absolute:
        compared["path"] = declared.as_posix()
    if compared != dict(record):
        raise ValueError(f"{label} binding mismatch")
    return path, observed


def _jsonl(data: bytes, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(data.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{label} line {index} is invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {index} must be an object")
        rows.append(value)
    return rows


def _sample(row: Mapping[str, Any]) -> TemporalExportSample:
    expected = {
        "frame_index",
        "timestamp_ns",
        "entity_id",
        "centroid_xyz",
        "observation_count",
        "dynamic_state",
        "motion_confidence",
        "geometry_epoch",
        "readout_valid",
    }
    if set(row) != expected:
        raise ValueError("trajectory row schema is invalid")
    try:
        dynamic_state = DynamicState(row["dynamic_state"])
    except (TypeError, ValueError) as error:
        raise ValueError("trajectory dynamic state is invalid") from error
    return TemporalExportSample(
        frame_index=row["frame_index"],
        timestamp_ns=row["timestamp_ns"],
        entity_id=row["entity_id"],
        centroid_xyz=row["centroid_xyz"],
        observation_count=row["observation_count"],
        dynamic_state=dynamic_state,
        motion_confidence=row["motion_confidence"],
        geometry_epoch=row["geometry_epoch"],
        readout_valid=row["readout_valid"],
    )


def _event(row: Mapping[str, Any]) -> TemporalLifecycleEvent:
    expected = {
        "frame_index",
        "timestamp_ns",
        "entity_id",
        "before",
        "after",
        "evidence",
        "geometry_epoch",
        "readout_valid",
    }
    if set(row) != expected:
        raise ValueError("lifecycle row schema is invalid")
    try:
        before = TemporalLifecycle(row["before"])
        after = TemporalLifecycle(row["after"])
        evidence = TemporalEvidenceKind(row["evidence"])
    except (TypeError, ValueError) as error:
        raise ValueError("lifecycle enum value is invalid") from error
    return TemporalLifecycleEvent(
        frame_index=row["frame_index"],
        timestamp_ns=row["timestamp_ns"],
        entity_id=row["entity_id"],
        before=before,
        after=after,
        evidence=evidence,
        geometry_epoch=row["geometry_epoch"],
        readout_valid=row["readout_valid"],
    )


def _temporal_batches(
    *,
    manifest: Mapping[str, Any],
    source_root: Path,
    source_index: Mapping[str, Any],
    witnesses: list[tuple[Path, dict[str, object]]],
) -> tuple[tuple[TemporalExportBatch, ...], dict[str, dict[str, object]]]:
    rows_by_frame: dict[int, list[TemporalExportSample]] = {}
    events_by_frame: dict[int, list[TemporalLifecycleEvent]] = {}
    loaded: dict[str, list[dict[str, Any]]] = {}
    source_records: dict[str, dict[str, object]] = {}
    for role, key in (
        ("trajectories", "trajectories"),
        ("lifecycle", "lifecycle_transitions"),
        ("coverage", "frame_coverage"),
    ):
        path, record = _bound_path(
            root=source_root, record=source_index.get(key), label=f"source {key}"
        )
        witnesses.append((path, record))
        source_records[role] = record
        loaded[role] = _jsonl(_regular_bytes(path, label=role), label=role)
    for row in loaded["trajectories"]:
        sample = _sample(row)
        rows_by_frame.setdefault(sample.frame_index, []).append(sample)
    for row in loaded["lifecycle"]:
        event = _event(row)
        events_by_frame.setdefault(event.frame_index, []).append(event)

    coverage = loaded["coverage"]
    expected_count = manifest.get("processed_frame_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ValueError("processed_frame_count is invalid")
    if len(coverage) != expected_count or manifest.get("covered_frame_count") != len(coverage):
        raise ValueError("temporal export coverage count is incomplete")
    batches: list[TemporalExportBatch] = []
    previous_timestamp = -1
    for expected_frame, row in enumerate(coverage):
        if set(row) != {"frame_index", "timestamp_ns", "record_count", "event_count"}:
            raise ValueError("temporal coverage row schema is invalid")
        frame = row["frame_index"]
        timestamp = row["timestamp_ns"]
        if (
            type(frame) is not int
            or type(timestamp) is not int
            or frame != expected_frame
            or timestamp <= previous_timestamp
        ):
            raise ValueError("temporal export coverage is not chronological and complete")
        samples = tuple(rows_by_frame.pop(frame, []))
        events = tuple(events_by_frame.pop(frame, []))
        if row["record_count"] != len(samples) or row["event_count"] != len(events):
            raise ValueError("temporal export coverage counts do not match records")
        batch = TemporalExportBatch(frame, timestamp, samples, events)
        batches.append(batch)
        previous_timestamp = timestamp
    if rows_by_frame or events_by_frame:
        raise ValueError("temporal export records fall outside frame coverage")
    result = tuple(batches)
    validate_temporal_export_sequence(result)
    if (
        manifest.get("first_frame_index") != 0
        or manifest.get("last_frame_index") != len(result) - 1
        or manifest.get("temporal_export_schema_version") not in {1, 2}
    ):
        raise ValueError("run manifest temporal coverage metadata is invalid")
    return result, source_records


@dataclass(frozen=True)
class _VisibilityTimeline:
    suppressed_by_frame: dict[int, frozenset[str]]
    events_by_frame: dict[int, tuple[dict[str, object], ...]]
    diagnostics: dict[str, object]


def _official_visibility_schedule(schedule_manifest: Path) -> Path:
    from src.datasets.tesse_cd import TesseCdRgbdDataset

    cache = _json_object(
        _regular_bytes(
            TesseCdRgbdDataset.CACHE_MANIFEST_PATH,
            label="RGB-D cache input manifest",
        ),
        label="RGB-D cache input manifest",
    )
    binding = cache.get("schedule_manifest")
    if not isinstance(binding, Mapping):
        raise ValueError("official schedule binding is invalid")
    raw_path = binding.get("path")
    expected_sha256 = binding.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("official schedule binding is invalid")
    canonical = Path(raw_path)
    if not canonical.is_absolute():
        canonical = TesseCdRgbdDataset.REPOSITORY_ROOT / canonical
    canonical = Path(os.path.abspath(os.fspath(canonical)))
    canonical_bytes = _regular_bytes(canonical, label="official causal schedule")
    alias_bytes = _regular_bytes(schedule_manifest, label="anchor causal schedule")
    if _sha256(canonical_bytes) != expected_sha256 or alias_bytes != canonical_bytes:
        raise ValueError("official schedule binding does not match anchor schedule")
    return canonical


def _production_visibility_dataset(
    root: Path,
    scene: str,
    export_manifest: Path,
    schedule_manifest: Path,
) -> Any:
    from src.datasets.tesse_cd import TesseCdRgbdDataset

    return TesseCdRgbdDataset(
        root,
        scene,
        export_manifest,
        _official_visibility_schedule(schedule_manifest),
    )


def _visibility_timeline(
    *,
    anchor_entities: Sequence[Any],
    batches: tuple[TemporalExportBatch, ...],
    cutoff_frame: int,
    config: AnchorVisibilityConfig,
    dataset: Any,
    policy_record: Mapping[str, object],
    rgbd_export_record: Mapping[str, object],
    rgbd_combined_output_sha256: str,
    rgbd_file_hash_count: int,
    counterfactual_variant: CounterfactualVariant | None = None,
) -> _VisibilityTimeline:
    if len(dataset) != len(batches):
        raise ValueError("visibility dataset frame count does not match source run")
    for batch in batches:
        timestamp_ns = dataset.timestamp_ns(batch.frame_index)
        if (
            isinstance(timestamp_ns, (bool, np.bool_))
            or not isinstance(timestamp_ns, (int, np.integer))
            or int(timestamp_ns) != batch.timestamp_ns
        ):
            raise ValueError("visibility dataset timestamps do not match source run")

    ordered_entities = tuple(sorted(anchor_entities, key=lambda item: item.entity_id))
    sampled = {
        entity.entity_id: sample_anchor_voxels(entity.points_xyz, config)
        for entity in ordered_entities
    }
    initial_timestamp = batches[cutoff_frame].timestamp_ns / 1_000_000_000
    states = {
        entity.entity_id: AnchorVisibilityState(
            entity_id=entity.entity_id,
            active=True,
            last_frame_id=cutoff_frame,
            last_timestamp=initial_timestamp,
            absence_observation_count=0,
            absence_viewpoints_xyz=(),
            present_streak=0,
        )
        for entity in ordered_entities
    }
    counts = {
        entity.entity_id: {kind.value: 0 for kind in AnchorVisibilityEvidenceKind}
        for entity in ordered_entities
    }
    suppressed_by_frame: dict[int, frozenset[str]] = {}
    events_by_frame: dict[int, tuple[dict[str, object], ...]] = {}
    transition_records: list[dict[str, object]] = []
    reference_transition_records: list[dict[str, object]] = []
    first_absence_frames: dict[str, int | None] = {
        entity.entity_id: None for entity in ordered_entities
    }
    transition_support: dict[str, dict[str, int]] = {}
    selected = (
        None
        if counterfactual_variant is None
        else frozenset(counterfactual_variant.selected_anchor_ids)
    )
    for batch in batches[cutoff_frame + 1 :]:
        frame = dataset[batch.frame_index]
        if (
            frame.frame_id != batch.frame_index
            or frame.source_frame_id != batch.frame_index
            or frame.timestamp != batch.timestamp_ns / 1_000_000_000
        ):
            raise ValueError("visibility frame identity does not match source run")
        frame_events: list[dict[str, object]] = []
        for entity in ordered_entities:
            entity_id = entity.entity_id
            evidence = classify_anchor_visibility(sampled[entity_id], frame, config)
            counts[entity_id][evidence.kind.value] += 1
            if (
                evidence.kind is AnchorVisibilityEvidenceKind.VISIBLE_ABSENT
                and first_absence_frames[entity_id] is None
            ):
                first_absence_frames[entity_id] = batch.frame_index
            before = states[entity_id]
            after = advance_anchor_visibility(before, evidence, config)
            states[entity_id] = after
            if before.active is after.active:
                continue
            event = {
                "frame_index": batch.frame_index,
                "timestamp_ns": batch.timestamp_ns,
                "entity_id": f"anchor:{entity_id}",
                "before": "active" if before.active else "dormant",
                "after": "active" if after.active else "dormant",
                "evidence": evidence.kind.value,
                "geometry_epoch": 0,
                "readout_valid": after.active,
            }
            record = {
                "frame_index": batch.frame_index,
                "timestamp_ns": batch.timestamp_ns,
                "entity_id": f"anchor:{entity_id}",
                "before": event["before"],
                "after": event["after"],
                "evidence": event["evidence"],
                "readout_valid": event["readout_valid"],
            }
            if before.active and not after.active:
                support = {
                    "frame_index": batch.frame_index,
                    "absence_observation_count": after.absence_observation_count,
                    "distinct_absence_viewpoint_count": len(
                        after.absence_viewpoints_xyz
                    ),
                }
                transition_support.setdefault(entity_id, support)
                reference_transition_records.append({**record, **support})
            if selected is None or entity_id in selected:
                frame_events.append(event)
                transition_records.append(record)
        suppressed_by_frame[batch.frame_index] = frozenset(
            entity_id
            for entity_id, state in states.items()
            if not state.active and (selected is None or entity_id in selected)
        )
        if frame_events:
            events_by_frame[batch.frame_index] = tuple(
                sorted(frame_events, key=lambda item: str(item["entity_id"]))
            )

    if selected is not None:
        transitioned = {
            record["entity_id"][len("anchor:") :]
            for record in reference_transition_records
        }
        if not selected.issubset(transitioned):
            raise ValueError(
                "counterfactual selection must contain only transitioned unbound anchors"
            )

    entity_diagnostics = [
        {
            "entity_id": entity.entity_id,
            "sampled_voxel_count": len(sampled[entity.entity_id]),
            "evidence_counts": counts[entity.entity_id],
            "final_active": states[entity.entity_id].active,
            "absence_observation_count": states[
                entity.entity_id
            ].absence_observation_count,
            "distinct_absence_viewpoint_count": len(
                states[entity.entity_id].absence_viewpoints_xyz
            ),
            **(
                {
                    "first_absence_frame": first_absence_frames[entity.entity_id],
                    "transition": transition_support.get(entity.entity_id),
                }
                if counterfactual_variant is not None
                else {}
            ),
        }
        for entity in ordered_entities
    ]
    diagnostics = {
        "schema_version": 1,
        "manifest_id": "crove_ovimap_unbound_visibility_diagnostics_v1",
        "causality": {
            "anchor_cutoff_frame": cutoff_frame,
            "first_visibility_frame": cutoff_frame + 1,
            "last_visibility_frame": len(batches) - 1,
            "current_and_past_rgbd_only": True,
        },
        "policy": dict(policy_record),
        "rgbd_export_manifest": dict(rgbd_export_record),
        "rgbd_combined_output_sha256": rgbd_combined_output_sha256,
        "rgbd_file_hash_count": rgbd_file_hash_count,
        "transition_count": len(transition_records),
        "transitions": transition_records,
        "entities": entity_diagnostics,
        **(
            {
                "counterfactual_variant": counterfactual_variant.to_json_record(),
                "p5_reference_transition_count": len(reference_transition_records),
                "p5_reference_transitions": reference_transition_records,
            }
            if counterfactual_variant is not None
            else {}
        ),
    }
    return _VisibilityTimeline(
        suppressed_by_frame=suppressed_by_frame,
        events_by_frame=events_by_frame,
        diagnostics=diagnostics,
    )


def _anchor_config(payload: Mapping[str, Any]) -> StaticAnchorConfig:
    try:
        return StaticAnchorConfig(
            minimum_spatial_iou=payload["minimum_spatial_iou"],
            maximum_centroid_distance_m=payload["maximum_centroid_distance_m"],
            minimum_semantic_cosine=payload["minimum_semantic_cosine"],
            moved_displacement_m=payload["moved_displacement_m"],
            background_voxel_size_m=payload["background_voxel_size_m"],
        )
    except KeyError as error:
        raise ValueError("anchor configuration is missing an overlay threshold") from error


def _atomic_json(path: Path, value: object) -> None:
    data = _canonical_json(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    data = b"".join(_canonical_json(value) for value in values)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _revalidate(witnesses: Sequence[tuple[Path, Mapping[str, object]]]) -> None:
    for path, record in witnesses:
        observed = _input_record(path)
        if observed != dict(record):
            raise RuntimeError(f"input changed during composition: {path}")


def compose_run(
    *,
    source_run_manifest: Path,
    anchor_manifest: Path,
    output_root: Path,
    moved_geometry_mode: str = "temporal_compact",
    readout_role: str = "formal_baseline",
    visibility_policy: Path | None = None,
    dataset_factory: Any | None = None,
    counterfactual_variant: CounterfactualVariant | None = None,
) -> Path:
    """Publish a source-bound sequential composition without mutating inputs."""

    visibility_enabled = visibility_policy is not None
    readout_contract = _readout_contract(
        moved_geometry_mode,
        readout_role,
        visibility_enabled=visibility_enabled,
        counterfactual_variant=counterfactual_variant,
    )
    if dataset_factory is not None and not visibility_enabled:
        raise ValueError("visibility dataset factory requires a visibility policy")
    if dataset_factory is not None and not callable(dataset_factory):
        raise TypeError("dataset_factory must be callable")
    source_run_manifest = Path(os.path.abspath(os.fspath(source_run_manifest)))
    anchor_manifest = Path(os.path.abspath(os.fspath(anchor_manifest)))
    output_root = Path(os.path.abspath(os.fspath(output_root)))
    if output_root.exists():
        raise ValueError(f"output root already exists: {output_root}")
    source_bytes = _regular_bytes(source_run_manifest, label="source run manifest")
    anchor_bytes = _regular_bytes(anchor_manifest, label="anchor manifest")
    source = _json_object(source_bytes, label="source run manifest")
    anchor_payload = _json_object(anchor_bytes, label="anchor manifest")
    scene = source.get("scene")
    if (
        source.get("schema_version") != 2
        or source.get("dataset") != "TESSE-CD"
        or source.get("method_id") != "OVIV2"
        or source.get("mode") != "dual_readout_causal_checkpoints"
        or not isinstance(scene, str)
        or anchor_payload.get("status") != "PASS"
        or anchor_payload.get("manifest_id") != "crove_ovimap_static_anchor_v1"
        or anchor_payload.get("scene") != scene
    ):
        raise ValueError("source run and anchor manifest identities do not match")
    causality = anchor_payload.get("causality")
    if (
        not isinstance(causality, Mapping)
        or causality.get("strictly_pre_intervention") is not True
        or causality.get("last_source_frame") != causality.get("maximum_source_frame")
        or type(causality.get("maximum_source_frame")) is not int
    ):
        raise ValueError("anchor causality contract is invalid")
    cutoff = int(causality["maximum_source_frame"])
    anchor_root = anchor_manifest.parent
    source_root = source_run_manifest.parent
    witnesses: list[tuple[Path, dict[str, object]]] = [
        (source_run_manifest, _input_record(source_run_manifest, source_bytes)),
        (anchor_manifest, _input_record(anchor_manifest, anchor_bytes)),
    ]

    outputs = anchor_payload.get("outputs")
    sources = anchor_payload.get("sources")
    if not isinstance(outputs, Mapping) or not isinstance(sources, Mapping):
        raise ValueError("anchor manifest sources or outputs are invalid")
    anchor_snapshot_path, anchor_snapshot_record = _bound_path(
        root=anchor_root, record=outputs.get("snapshot"), label="anchor snapshot"
    )
    anchor_entities_path, anchor_entities_record = _bound_path(
        root=anchor_root, record=outputs.get("entities"), label="anchor entities"
    )
    config_path, config_record = _bound_path(
        root=anchor_root,
        record=sources.get("config"),
        label="anchor config",
        absolute=True,
    )
    vocabulary_path, vocabulary_record = _bound_path(
        root=anchor_root,
        record=sources.get("vocabulary"),
        label="anchor vocabulary",
        absolute=True,
    )
    schedule_path, schedule_record = _bound_path(
        root=anchor_root,
        record=sources.get("schedule"),
        label="causal schedule",
        absolute=True,
    )
    witnesses.extend(
        (
            (anchor_snapshot_path, anchor_snapshot_record),
            (anchor_entities_path, anchor_entities_record),
            (config_path, config_record),
            (vocabulary_path, vocabulary_record),
            (schedule_path, schedule_record),
        )
    )
    anchor = read_map_snapshot(anchor_snapshot_path, anchor_entities_path)
    if anchor.scene_id != scene or anchor.scope != "current":
        raise ValueError("anchor snapshot identity is invalid")
    config = _anchor_config(
        _json_object(_regular_bytes(config_path, label="anchor config"), label="anchor config")
    )
    vocabulary = _json_object(
        _regular_bytes(vocabulary_path, label="anchor vocabulary"),
        label="anchor vocabulary",
    )
    class_names = vocabulary.get("classes")
    if (
        vocabulary.get("dataset") != "TESSE-CD"
        or vocabulary.get("scene") != scene
        or not isinstance(class_names, list)
        or not class_names
        or any(not isinstance(item, str) or not item.strip() for item in class_names)
    ):
        raise ValueError("anchor vocabulary identity is invalid")

    source_index_path, source_index_record = _bound_path(
        root=source_root, record=source.get("source_index"), label="source index"
    )
    witnesses.append((source_index_path, source_index_record))
    source_index = _json_object(
        _regular_bytes(source_index_path, label="source index"), label="source index"
    )
    if (
        source_index.get("dataset") != "TESSE-CD"
        or source_index.get("scene") != scene
        or source_index.get("method") != "OVIV2"
    ):
        raise ValueError("source index identity is invalid")
    batches, temporal_input_records = _temporal_batches(
        manifest=source,
        source_root=source_root,
        source_index=source_index,
        witnesses=witnesses,
    )
    batch_by_frame = {item.frame_index: item for item in batches}
    latest_prefix: dict[int, TemporalExportSample] = {}
    for batch in batches:
        if batch.frame_index > cutoff:
            break
        for sample in batch.samples:
            if sample.readout_valid:
                latest_prefix[sample.entity_id] = sample
    prefix_samples = tuple(
        PrefixIdentitySample(
            temporal_entity_id=sample.entity_id,
            frame_index=sample.frame_index,
            centroid_xyz=sample.centroid_xyz,
            points_xyz=np.asarray((sample.centroid_xyz,), dtype=np.float32),
            semantic_label=None,
            semantic_embedding=None,
            geometry_epoch=sample.geometry_epoch,
        )
        for sample in sorted(latest_prefix.values(), key=lambda item: item.entity_id)
    )
    state = bind_anchor_identities(anchor, prefix_samples, config, cutoff_frame=cutoff)
    bound_anchor_ids = {anchor_id for anchor_id, _ in state.bindings}
    unbound_anchors = tuple(
        entity
        for entity in sorted(anchor.entities, key=lambda item: item.entity_id)
        if entity.entity_id not in bound_anchor_ids
    )

    visibility_timeline: _VisibilityTimeline | None = None
    visibility_policy_record: dict[str, object] | None = None
    rgbd_export_record: dict[str, object] | None = None
    if visibility_policy is not None:
        policy_path = Path(os.path.abspath(os.fspath(visibility_policy)))
        policy_bytes = _regular_bytes(policy_path, label="visibility policy")
        policy_payload = _json_object(policy_bytes, label="visibility policy")
        visibility_config = anchor_visibility_config_from_json(policy_payload)
        visibility_policy_record = _input_record(policy_path, policy_bytes)
        rgbd_export_path, rgbd_export_record = _bound_path(
            root=anchor_root,
            record=sources.get("rgbd_export_manifest"),
            label="anchor RGB-D export manifest",
            absolute=True,
        )
        rgbd_payload = _json_object(
            _regular_bytes(rgbd_export_path, label="anchor RGB-D export manifest"),
            label="anchor RGB-D export manifest",
        )
        combined_output = rgbd_payload.get("combined_output_sha256")
        file_hash_count = rgbd_payload.get("file_hash_count")
        if (
            rgbd_payload.get("dataset") != "TESSE-CD"
            or rgbd_payload.get("scene") != scene
            or not isinstance(combined_output, str)
            or len(combined_output) != 64
            or any(character not in "0123456789abcdef" for character in combined_output)
            or isinstance(file_hash_count, bool)
            or not isinstance(file_hash_count, int)
            or file_hash_count <= 0
        ):
            raise ValueError("anchor RGB-D export manifest identity is invalid")
        witnesses.extend(
            (
                (policy_path, visibility_policy_record),
                (rgbd_export_path, rgbd_export_record),
            )
        )
        factory = (
            _production_visibility_dataset
            if dataset_factory is None
            else dataset_factory
        )
        dataset = factory(
            rgbd_export_path.parent,
            scene,
            rgbd_export_path,
            schedule_path,
        )
        visibility_timeline = _visibility_timeline(
            anchor_entities=unbound_anchors,
            batches=batches,
            cutoff_frame=cutoff,
            config=visibility_config,
            dataset=dataset,
            policy_record=visibility_policy_record,
            rgbd_export_record=rgbd_export_record,
            rgbd_combined_output_sha256=combined_output,
            rgbd_file_hash_count=file_hash_count,
            counterfactual_variant=counterfactual_variant,
        )

    checkpoint_values = source.get("checkpoints")
    captured = source.get("captured_frame_indices")
    if not isinstance(checkpoint_values, list) or not checkpoint_values:
        raise ValueError("source run contains no checkpoints")
    frames = [item.get("frame_index") if isinstance(item, Mapping) else None for item in checkpoint_values]
    if (
        any(type(frame) is not int for frame in frames)
        or frames != sorted(set(frames))
        or captured != frames
        or frames[0] <= cutoff
    ):
        raise ValueError("source checkpoint order or cutoff is invalid")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    checkpoint_results: list[dict[str, Any]] = []
    source_checkpoint_records: list[dict[str, Any]] = []
    try:
        trajectory_rows: list[dict[str, object]] = []
        lifecycle_rows: list[dict[str, object]] = []
        coverage_rows: list[dict[str, object]] = []
        for batch in batches:
            rows = [sample.to_json_record() for sample in batch.samples]
            suppressed = (
                visibility_timeline.suppressed_by_frame.get(
                    batch.frame_index, frozenset()
                )
                if visibility_timeline is not None
                else frozenset()
            )
            for entity in unbound_anchors:
                rows.append(
                    {
                        "frame_index": batch.frame_index,
                        "timestamp_ns": batch.timestamp_ns,
                        "entity_id": f"anchor:{entity.entity_id}",
                        "centroid_xyz": np.asarray(
                            entity.points_xyz, dtype=np.float64
                        ).mean(axis=0).tolist(),
                        "observation_count": batch.frame_index + 1,
                        "dynamic_state": "static",
                        "motion_confidence": 0.0,
                        "geometry_epoch": 0,
                        "readout_valid": entity.entity_id not in suppressed,
                    }
                )
            rows.sort(key=lambda item: str(item["entity_id"]))
            events = [event.to_json_record() for event in batch.events]
            if visibility_timeline is not None:
                events.extend(
                    visibility_timeline.events_by_frame.get(batch.frame_index, ())
                )
                events.sort(key=lambda item: str(item["entity_id"]))
            trajectory_rows.extend(rows)
            lifecycle_rows.extend(events)
            coverage_rows.append(
                {
                    "frame_index": batch.frame_index,
                    "timestamp_ns": batch.timestamp_ns,
                    "record_count": len(rows),
                    "event_count": len(events),
                }
            )
        trajectories_path = staging / "trajectories.jsonl"
        lifecycle_path = staging / "lifecycle_transitions.jsonl"
        coverage_path = staging / "temporal_frame_coverage.jsonl"
        _atomic_jsonl(trajectories_path, trajectory_rows)
        _atomic_jsonl(lifecycle_path, lifecycle_rows)
        _atomic_jsonl(coverage_path, coverage_rows)
        trajectories_record = _relative_record(trajectories_path, root=staging)
        lifecycle_record = _relative_record(lifecycle_path, root=staging)
        coverage_record = _relative_record(coverage_path, root=staging)
        runtime_diagnostics_record: dict[str, object] | None = None
        if visibility_timeline is not None:
            runtime_diagnostics_path = staging / "runtime_diagnostics.json"
            _atomic_json(runtime_diagnostics_path, visibility_timeline.diagnostics)
            runtime_diagnostics_record = _relative_record(
                runtime_diagnostics_path,
                root=staging,
            )

        previous_frame = cutoff
        for source_checkpoint in checkpoint_values:
            if not isinstance(source_checkpoint, Mapping):
                raise ValueError("source checkpoint must be an object")
            frame = source_checkpoint["frame_index"]
            timestamp_ns = source_checkpoint.get("timestamp_ns")
            batch = batch_by_frame.get(frame)
            if (
                type(timestamp_ns) is not int
                or batch is None
                or batch.timestamp_ns != timestamp_ns
                or source_checkpoint.get("consumed_through_frame") != frame
                or source_checkpoint.get("consumed_through_frame_exclusive") != frame + 1
            ):
                raise ValueError("checkpoint does not match its temporal export")
            snapshot_path, snapshot_record = _bound_path(
                root=source_root,
                record=source_checkpoint.get("neutral_snapshot"),
                label=f"checkpoint {frame} snapshot",
            )
            entities_path, entities_record = _bound_path(
                root=source_root,
                record=source_checkpoint.get("neutral_entities"),
                label=f"checkpoint {frame} entities",
            )
            witnesses.extend(
                ((snapshot_path, snapshot_record), (entities_path, entities_record))
            )
            temporal = read_map_snapshot(snapshot_path, entities_path)
            if (
                temporal.scene_id != scene
                or temporal.scope != "current"
                or temporal.timestamp != float(timestamp_ns)
            ):
                raise ValueError("checkpoint snapshot identity is invalid")
            interval = tuple(batch_by_frame[index] for index in range(previous_frame + 1, frame + 1))
            composed, state, diagnostics = compose_anchor_checkpoint(
                anchor=anchor,
                temporal=temporal,
                frame_index=frame,
                exports=interval,
                state=state,
                config=config,
                anchor_manifest_sha256=_sha256(anchor_bytes),
                class_names=class_names,
                moved_geometry_mode=moved_geometry_mode,
                suppressed_unbound_anchor_ids=(
                    visibility_timeline.suppressed_by_frame.get(frame, frozenset())
                    if visibility_timeline is not None
                    else frozenset()
                ),
            )
            checkpoint_root = staging / "checkpoints" / f"{frame:08d}-{timestamp_ns}"
            written = write_map_snapshot(composed, checkpoint_root / "current")
            diagnostic_payload = asdict(diagnostics)
            if visibility_timeline is not None:
                diagnostic_payload["visibility_suppressed_anchor_ids"] = sorted(
                    visibility_timeline.suppressed_by_frame.get(frame, frozenset())
                )
            diagnostics_path = checkpoint_root / "diagnostics.json"
            _atomic_json(diagnostics_path, diagnostic_payload)
            status_path = checkpoint_root / "checkpoint_status.json"
            _atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "checkpoint_frame": frame,
                    "timestamp_ns": timestamp_ns,
                    "consumed_through_frame": frame,
                    "consumed_through_frame_exclusive": frame + 1,
                },
            )
            snapshot_output_record = _relative_record(
                written["snapshot"], root=staging
            )
            entities_output_record = _relative_record(
                written["entities"], root=staging
            )
            status_record = _relative_record(status_path, root=staging)
            checkpoint_results.append(
                {
                    "frame_index": frame,
                    "timestamp_ns": timestamp_ns,
                    "consumed_through_frame": frame,
                    "source_snapshot": dict(source_checkpoint["neutral_snapshot"]),
                    "source_entities": dict(source_checkpoint["neutral_entities"]),
                    "snapshot": snapshot_output_record,
                    "entities": entities_output_record,
                    "diagnostics": diagnostic_payload,
                    "diagnostics_file": _relative_record(diagnostics_path, root=staging),
                }
            )
            source_checkpoint_records.append(
                {
                    "frame_index": frame,
                    "timestamp_ns": timestamp_ns,
                    "consumed_through_frame": frame,
                    "consumed_through_frame_exclusive": frame + 1,
                    "checkpoint_status": status_record,
                    "snapshot": snapshot_output_record,
                    "entities": entities_output_record,
                }
            )
            previous_frame = frame

        _revalidate(witnesses)
        capture_path = staging / "capture_status.json"
        _atomic_json(
            capture_path,
            {
                "schema_version": 1,
                "status": "PASS",
                "scene": scene,
                "mode": "causal_checkpoints",
                "scheduled_frame_indices": frames,
                "captured_frame_indices": frames,
                "schedule": schedule_record,
                "trajectories": trajectories_record,
                "frame_coverage": coverage_record,
                "lifecycle_transitions": lifecycle_record,
                "checkpoint_statuses": [
                    item["checkpoint_status"] for item in source_checkpoint_records
                ],
            },
        )
        capture_record = _relative_record(capture_path, root=staging)
        source_index_path = staging / "source_index.json"
        _atomic_json(
            source_index_path,
            {
                "schema_version": 1,
                "dataset": "TESSE-CD",
                "mode": "causal_checkpoint_exports",
                "method": "OVIV2",
                "scene": scene,
                "schedule": schedule_record,
                "capture_status": capture_record,
                "trajectories": trajectories_record,
                "frame_coverage": coverage_record,
                "lifecycle_transitions": lifecycle_record,
                **(
                    {"runtime_diagnostics": runtime_diagnostics_record}
                    if runtime_diagnostics_record is not None
                    else {}
                ),
                "checkpoints": source_checkpoint_records,
            },
        )
        source_index_output_record = _relative_record(
            source_index_path, root=staging
        )
        manifest = {
            "schema_version": 1,
            "manifest_id": "crove_ovimap_static_anchor_composition_v1",
            "status": "PASS",
            "dataset": "TESSE-CD",
            "scene": scene,
            "method": "CROVE + OVI-MAP static anchor (composed)",
            "integration": "composed",
            "execution_mode": "online_after_causal_initialization",
            "causal_anchor_cutoff_frame": cutoff,
            "processed_frame_count": len(batches),
            "official_state_count": len(checkpoint_results),
            "readout_contract": readout_contract,
            **(
                {"counterfactual_variant": counterfactual_variant.to_json_record()}
                if counterfactual_variant is not None
                else {}
            ),
            "inputs": {
                "source_run_manifest": witnesses[0][1],
                "anchor_manifest": witnesses[1][1],
                "anchor_snapshot": anchor_snapshot_record,
                "anchor_entities": anchor_entities_record,
                "anchor_config": config_record,
                "anchor_vocabulary": vocabulary_record,
                "causal_schedule": schedule_record,
                "source_index": source_index_record,
                **(
                    {
                        "visibility_policy": visibility_policy_record,
                        "visibility_rgbd_export_manifest": rgbd_export_record,
                    }
                    if visibility_policy_record is not None
                    and rgbd_export_record is not None
                    else {}
                ),
                **{
                    f"source_{key}": temporal_input_records[key]
                    for key in ("trajectories", "lifecycle", "coverage")
                },
            },
            "identity_bindings": [
                {"anchor_entity_id": anchor_id, "temporal_entity_id": temporal_id}
                for anchor_id, temporal_id in state.bindings
            ],
            "source_index": source_index_output_record,
            "checkpoints": checkpoint_results,
        }
        manifest_path = staging / "run_manifest.json"
        _atomic_json(manifest_path, manifest)
        _revalidate(witnesses)
        os.replace(staging, output_root)
        directory_fd = os.open(output_root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return output_root / "run_manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-manifest", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--moved-geometry-mode",
        choices=_MOVED_GEOMETRY_MODES,
        default="temporal_compact",
    )
    parser.add_argument(
        "--readout-role",
        choices=_READOUT_ROLES,
        default="formal_baseline",
    )
    parser.add_argument("--visibility-policy", type=Path)
    args = parser.parse_args(argv)
    compose_run(
        source_run_manifest=args.source_run_manifest,
        anchor_manifest=args.anchor_manifest,
        output_root=args.output_root,
        moved_geometry_mode=args.moved_geometry_mode,
        readout_role=args.readout_role,
        visibility_policy=args.visibility_policy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
