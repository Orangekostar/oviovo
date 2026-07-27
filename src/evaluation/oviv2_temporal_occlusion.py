from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.evaluation.oviv2_runtime_diagnostics import validate_runtime_diagnostics
from src.oviv2.temporal_snapshot import TemporalCompactCheckpoint


EVALUATION_FORMAT = "oviv2_temporal_compact_v1"
_RETAINED_CODES = frozenset({0, 1})
_OPEN_ENDED_TIMESTAMP_NS = 2**64 - 1
_REQUIRED_ANCHOR_ELIGIBLE_COUNT = 66
_REQUIRED_ANCHOR_MAPPED_COUNT = 53
_TRAJECTORY_FIELDS = frozenset(
    {
        "frame_index", "timestamp_ns", "entity_id", "centroid_xyz",
        "observation_count", "dynamic_state", "motion_confidence",
        "geometry_epoch", "readout_valid",
    }
)
_LIFECYCLE_FIELDS = frozenset(
    {
        "frame_index", "timestamp_ns", "entity_id", "before", "after",
        "evidence", "geometry_epoch", "readout_valid",
    }
)
_COVERAGE_FIELDS = frozenset(
    {"frame_index", "timestamp_ns", "record_count", "event_count"}
)
_MECHANISMS_BY_PROFILE = {
    "a0": (),
    "a1": ("absence", "readout_invalidation"),
    "a2": (
        "absence", "readout_invalidation", "proposal_recovery", "epoch_reset",
        "motion_rejection",
    ),
    "a3": (
        "absence", "readout_invalidation", "proposal_recovery", "epoch_reset",
        "motion_rejection", "background_release", "background_reclaim",
    ),
    "a4": (
        "absence", "readout_invalidation", "proposal_recovery", "epoch_reset",
        "background_release", "background_reclaim", "eligible_reid",
        "icp_attempt", "icp_accept", "motion_rejection",
    ),
}


def _target_keys(array: object) -> set[tuple[int, int, int]]:
    values = np.asarray(array)
    if values.dtype != np.int64 or values.ndim != 2 or values.shape[1:] != (3,):
        raise ValueError("target voxel arrays must have dtype int64 and shape (N, 3)")
    if not np.isfinite(values).all():
        raise ValueError("target voxel arrays must be finite")
    return {tuple(int(value) for value in row) for row in values}


def _entity_world_keys(
    checkpoint: TemporalCompactCheckpoint, index: int, target_voxel_size_m: float
) -> set[tuple[int, int, int]]:
    start = int(checkpoint.voxel_offsets[index])
    stop = int(checkpoint.voxel_offsets[index + 1])
    local = (
        checkpoint.voxel_keys[start:stop].astype(np.float64) + 0.5
    ) * checkpoint.metadata.voxel_size_m
    if not len(local):
        return set()
    homogeneous = np.concatenate([local, np.ones((len(local), 1))], axis=1)
    world = (checkpoint.object_to_world[index] @ homogeneous.T).T[:, :3]
    quantized = np.floor(world / target_voxel_size_m).astype(np.int64)
    return {tuple(int(value) for value in row) for row in quantized}


def _frame_entities(
    checkpoint: TemporalCompactCheckpoint, target_voxel_size_m: float
) -> list[tuple[int, int, set[tuple[int, int, int]]]]:
    return [
        (
            int(entity_id),
            int(checkpoint.lifecycle_codes[index]),
            _entity_world_keys(checkpoint, index, target_voxel_size_m),
        )
        for index, entity_id in enumerate(checkpoint.entity_ids.tolist())
    ]


def _mapped_entity(
    entities: list[tuple[int, int, set[tuple[int, int, int]]]],
    target_keys: set[tuple[int, int, int]],
) -> tuple[int | None, dict[str, Any]]:
    candidates = []
    for entity_id, lifecycle_code, world_keys in entities:
        if lifecycle_code not in _RETAINED_CODES:
            continue
        overlap = len(target_keys & world_keys)
        if overlap:
            candidates.append((overlap, entity_id))
    maximum = max((overlap for overlap, _ in candidates), default=0)
    winners = sorted(
        entity_id for overlap, entity_id in candidates if overlap == maximum
    )
    ambiguous = len(winners) > 1
    mapped_id = winners[0] if len(winners) == 1 else None
    target_count = len(target_keys)
    evidence = {
        "available": True,
        "mapped_temporal_id": mapped_id,
        "overlap_voxel_count": maximum,
        "target_voxel_count": target_count,
        "target_coverage": (
            None if target_count == 0 else float(maximum / target_count)
        ),
        "ambiguous": ambiguous,
        "minimum_overlap_voxel_count": 1,
        "unavailable_reason": None,
    }
    if not candidates:
        return None, evidence
    return mapped_id, evidence


def _unavailable_mapping_evidence(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "mapped_temporal_id": None,
        "overlap_voxel_count": None,
        "target_voxel_count": None,
        "target_coverage": None,
        "ambiguous": None,
        "minimum_overlap_voxel_count": 1,
        "unavailable_reason": reason,
    }


def _required(
    checkpoints: Mapping[tuple[str, int], TemporalCompactCheckpoint],
    scene: str,
    frame: int,
    relative_timestamp_ns: int,
    checkpoint_relative_timestamp_ns: Mapping[tuple[str, int], int] | None,
) -> TemporalCompactCheckpoint:
    value = checkpoints.get((scene, frame))
    if value is None:
        raise ValueError(f"missing checkpoint for {scene} frame {frame}")
    if value.metadata.scene_id != scene or value.metadata.frame_id != frame:
        raise ValueError("checkpoint scene/frame binding mismatch")
    observed_ns = (
        round(value.metadata.timestamp * 1_000_000_000)
        if checkpoint_relative_timestamp_ns is None
        else checkpoint_relative_timestamp_ns.get((scene, frame))
    )
    if observed_ns != relative_timestamp_ns:
        raise ValueError("checkpoint timestamp does not match target event")
    return value


def _rate(numerator: int, denominator: int, unavailable_reason: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": denominator > 0,
        "denominator": denominator,
        "value": None if denominator == 0 else float(numerator / denominator),
    }
    if denominator == 0:
        result["unavailable_reason"] = unavailable_reason
    return result


def _anchor_coverage_gate(
    scene: str, lifecycle_records: Mapping[tuple[Any, int], Mapping[str, Any]]
) -> dict[str, Any]:
    eligible = [record for record in lifecycle_records.values() if record["anchor_keys"]]
    uniquely_mapped_count = sum(
        int(record["temporal_id"] is not None) for record in eligible
    )
    ambiguous_count = sum(
        int(record["mapping_evidence"]["ambiguous"] is True) for record in eligible
    )
    zero_overlap_count = sum(
        int(record["mapping_evidence"]["overlap_voxel_count"] == 0)
        for record in eligible
    )
    eligible_count = len(eligible)
    available = eligible_count > 0
    passed = bool(
        available
        and eligible_count >= _REQUIRED_ANCHOR_ELIGIBLE_COUNT
        and uniquely_mapped_count >= _REQUIRED_ANCHOR_MAPPED_COUNT
    )
    if not available:
        reason = "no_eligible_anchor_mappings"
    elif eligible_count < _REQUIRED_ANCHOR_ELIGIBLE_COUNT:
        reason = "insufficient_eligible_anchor_mappings"
    elif uniquely_mapped_count < _REQUIRED_ANCHOR_MAPPED_COUNT:
        reason = "insufficient_unique_anchor_mappings"
    else:
        reason = None
    return {
        "scene": scene,
        "eligible_count": eligible_count,
        "uniquely_mapped_count": uniquely_mapped_count,
        "zero_overlap_count": zero_overlap_count,
        "ambiguous_count": ambiguous_count,
        "required_eligible_count": _REQUIRED_ANCHOR_ELIGIBLE_COUNT,
        "required_mapped_count": _REQUIRED_ANCHOR_MAPPED_COUNT,
        "available": available,
        "passed": passed,
        "reason": reason,
    }


def _mechanism_record(
    opportunities: int,
    triggers: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if not (0 <= triggers <= opportunities):
        raise ValueError("mechanism triggers must be bounded by opportunities")
    available = opportunities > 0
    passed = bool(available and triggers > 0)
    return {
        "opportunities": opportunities,
        "triggers": triggers,
        "available": available,
        "passed": passed,
        "reason": (
            None
            if passed
            else "no_opportunity"
            if not available
            else "not_triggered"
        ),
        "source": dict(source),
    }


def mechanism_telemetry_from_sources(
    *,
    trajectories: list[Mapping[str, Any]],
    lifecycle_transitions: list[Mapping[str, Any]],
    frame_coverage: list[Mapping[str, Any]],
    runtime_diagnostics: Mapping[str, Any],
    source_records: Mapping[str, Mapping[str, Any]],
    expected_temporal_readout: Mapping[str, Any] | None = None,
    expected_candidate_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    required_sources = {
        "trajectories", "lifecycle_transitions", "frame_coverage",
        "runtime_diagnostics",
    }
    if set(source_records) != required_sources or any(
        set(record) != {"path", "sha256", "byte_count"}
        or not isinstance(record["path"], str)
        or not record["path"]
        or not isinstance(record["sha256"], str)
        or len(record["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in record["sha256"])
        or type(record["byte_count"]) is not int
        or record["byte_count"] < 0
        for record in source_records.values()
    ):
        raise ValueError("mechanism source records are not exact")
    if not frame_coverage:
        raise ValueError("mechanism frame coverage must not be empty")
    for index, row in enumerate(frame_coverage):
        if not (
            set(row) == _COVERAGE_FIELDS
            and type(row["frame_index"]) is int
            and row["frame_index"] == index
            and type(row["timestamp_ns"]) is int
            and row["timestamp_ns"] > 0
            and (index == 0 or row["timestamp_ns"] > frame_coverage[index - 1]["timestamp_ns"])
            and type(row["record_count"]) is int
            and row["record_count"] >= 0
            and type(row["event_count"]) is int
            and row["event_count"] >= 0
        ):
            raise ValueError("mechanism frame coverage is invalid")
    trajectory_counts = [0] * len(frame_coverage)
    seen_trajectories: set[tuple[int, str]] = set()
    for row in trajectories:
        frame = row.get("frame_index")
        identifier = row.get("entity_id")
        centroid = row.get("centroid_xyz")
        if not (
            set(row) == _TRAJECTORY_FIELDS
            and type(frame) is int
            and 0 <= frame < len(frame_coverage)
            and type(row.get("timestamp_ns")) is int
            and row["timestamp_ns"] == frame_coverage[frame]["timestamp_ns"]
            and not isinstance(identifier, bool)
            and isinstance(identifier, (int, str))
            and str(identifier).strip()
            and isinstance(centroid, list)
            and len(centroid) == 3
            and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and np.isfinite(value)
                for value in centroid
            )
            and type(row.get("observation_count")) is int
            and row["observation_count"] > 0
            and row.get("dynamic_state") in {"static", "dynamic", "unknown"}
            and not isinstance(row.get("motion_confidence"), bool)
            and isinstance(row.get("motion_confidence"), (int, float))
            and np.isfinite(row["motion_confidence"])
            and 0 <= row["motion_confidence"] <= 1
            and type(row.get("geometry_epoch")) is int
            and row["geometry_epoch"] >= 0
            and type(row.get("readout_valid")) is bool
        ):
            raise ValueError("mechanism trajectory record is invalid")
        key = (frame, str(identifier))
        if key in seen_trajectories:
            raise ValueError("duplicate mechanism trajectory record")
        seen_trajectories.add(key)
        trajectory_counts[frame] += 1
    lifecycle_counts = [0] * len(frame_coverage)
    seen_lifecycle: set[tuple[int, str]] = set()
    absence_opportunities = 0
    absence_triggers = 0
    for row in lifecycle_transitions:
        frame = row.get("frame_index")
        identifier = row.get("entity_id")
        if not (
            set(row) == _LIFECYCLE_FIELDS
            and type(frame) is int
            and 0 <= frame < len(frame_coverage)
            and type(row.get("timestamp_ns")) is int
            and row["timestamp_ns"] == frame_coverage[frame]["timestamp_ns"]
            and not isinstance(identifier, bool)
            and isinstance(identifier, (int, str))
            and str(identifier).strip()
            and row.get("before") in {"active", "uncertain", "dormant"}
            and row.get("after") in {"active", "uncertain", "dormant"}
            and row.get("evidence")
            in {"present", "visible_absent", "occluded", "out_of_view", "depth_unknown"}
            and type(row.get("geometry_epoch")) is int
            and row["geometry_epoch"] >= 0
            and type(row.get("readout_valid")) is bool
        ):
            raise ValueError("mechanism lifecycle record is invalid")
        key = (frame, str(identifier))
        if key in seen_lifecycle:
            raise ValueError("duplicate mechanism lifecycle record")
        seen_lifecycle.add(key)
        lifecycle_counts[frame] += 1
        if row["evidence"] == "visible_absent":
            absence_opportunities += 1
            absence_triggers += int(row["readout_valid"] is False)
    if any(
        trajectory_counts[index] != row["record_count"]
        or lifecycle_counts[index] != row["event_count"]
        for index, row in enumerate(frame_coverage)
    ):
        raise ValueError("mechanism sources disagree with frame coverage")
    validate_runtime_diagnostics(
        runtime_diagnostics,
        label="runtime diagnostics",
        expected_temporal_readout=expected_temporal_readout,
        expected_candidate_id=expected_candidate_id,
        expected_processed_frame_count=len(frame_coverage),
    )
    profile = runtime_diagnostics["execution_profile"]
    counters = runtime_diagnostics["counters"]
    records = runtime_diagnostics["mechanism_records"]
    diagnostic = runtime_diagnostics["diagnostic"]
    assert isinstance(profile, str)
    assert isinstance(counters, Mapping)
    assert isinstance(records, Mapping)
    icp_enabled = profile == "a4" and not (
        isinstance(diagnostic, Mapping)
        and diagnostic.get("controls") == {"icp_enabled": False}
    )
    lifecycle_source = source_records["lifecycle_transitions"]
    runtime_source = source_records["runtime_diagnostics"]
    values = {
        "absence": _mechanism_record(
            absence_opportunities, absence_opportunities, lifecycle_source
        ),
        "readout_invalidation": _mechanism_record(
            absence_opportunities, absence_triggers, lifecycle_source
        ),
        "proposal_recovery": _mechanism_record(
            counters["proposal_opportunity_count"],
            counters["proposal_trigger_count"], runtime_source,
        ),
        "epoch_reset": _mechanism_record(
            counters["epoch_reset_opportunity_count"],
            counters["epoch_reset_trigger_count"], runtime_source,
        ),
        "background_release": _mechanism_record(
            counters["ledger_stage_count"],
            counters["ledger_commit_count"], runtime_source,
        ),
        "background_reclaim": _mechanism_record(
            counters["ledger_commit_count"],
            counters["ledger_reclaim_count"], runtime_source,
        ),
        "eligible_reid": _mechanism_record(
            counters["reid_opportunity_count"], counters["reid_trigger_count"],
            runtime_source,
        ),
        "icp_attempt": _mechanism_record(
            counters["icp_opportunity_count"], counters["icp_opportunity_count"],
            runtime_source,
        ),
        "icp_accept": _mechanism_record(
            counters["icp_opportunity_count"], counters["icp_accept_count"],
            runtime_source,
        ),
        "motion_rejection": _mechanism_record(
            (
                counters["icp_opportunity_count"]
                if icp_enabled
                else counters["motion_rejection_count"]
            ),
            counters["motion_rejection_count"],
            runtime_source,
        ),
    }
    return {name: values[name] for name in _MECHANISMS_BY_PROFILE[profile]}


def evaluate_temporal_occlusion(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    checkpoints: Mapping[tuple[str, int], TemporalCompactCheckpoint],
    scene: str,
    checkpoint_relative_timestamp_ns: Mapping[tuple[str, int], int] | None = None,
    mechanism_telemetry: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if scene not in {"apartment", "office"}:
        raise ValueError("scene is invalid")
    parameters = metadata.get("parameters")
    episodes = metadata.get("episodes")
    if not isinstance(parameters, Mapping) or not isinstance(episodes, list) or not episodes:
        raise ValueError("target metadata is invalid")
    voxel_size = parameters.get("voxel_size_m")
    if (
        isinstance(voxel_size, bool)
        or not isinstance(voxel_size, (int, float))
        or not np.isfinite(voxel_size)
        or voxel_size <= 0
    ):
        raise ValueError("target voxel_size_m is invalid")
    selected = [item for item in episodes if item.get("scene") == scene]
    if not selected:
        raise ValueError(f"target has no episodes for {scene}")

    lifecycle_records: dict[tuple[Any, int], dict[str, Any]] = {}
    anchors_by_frame: dict[int, list[dict[str, Any]]] = {}
    occluded_by_frame: dict[int, list[dict[str, Any]]] = {}
    timestamps_by_frame: dict[int, int] = {}

    def register_timestamp(frame: int, timestamp_ns: int) -> None:
        previous = timestamps_by_frame.setdefault(frame, timestamp_ns)
        if previous != timestamp_ns:
            raise ValueError("target event timestamp conflict")

    for episode in selected:
        lifecycle = episode["lifecycle"]
        key = (episode["object_id"], int(lifecycle["index"]))
        anchor = episode["anchor"]
        anchor_frame = int(anchor["frame_index"])
        anchor_timestamp = int(anchor["relative_timestamp_ns"])
        anchor_keys = _target_keys(arrays[str(anchor["array"])])
        signature = {
            "key": key,
            "object_id": episode["object_id"],
            "lifecycle": dict(lifecycle),
            "anchor": {
                "frame_index": anchor_frame,
                "relative_timestamp_ns": anchor_timestamp,
            },
            "anchor_keys": anchor_keys,
            "temporal_id": None,
            "mapping_evidence": None,
        }
        previous = lifecycle_records.get(key)
        if previous is not None and any(
            previous[name] != signature[name]
            for name in ("object_id", "lifecycle", "anchor", "anchor_keys")
        ):
            raise ValueError("target lifecycle anchor is inconsistent across episodes")
        if previous is None:
            lifecycle_records[key] = signature
            anchors_by_frame.setdefault(anchor_frame, []).append(signature)
            record = signature
        else:
            record = previous
        register_timestamp(anchor_frame, anchor_timestamp)
        for candidate in episode["checkpoints"]:
            frame = int(candidate["frame_index"])
            timestamp = int(candidate["relative_timestamp_ns"])
            register_timestamp(frame, timestamp)
            occluded_by_frame.setdefault(frame, []).append(
                {
                    "episode_id": str(episode["episode_id"]),
                    "record": record,
                    "frame_index": frame,
                    "relative_timestamp_ns": timestamp,
                    "array": str(candidate["array"]),
                }
            )

    available_checkpoints = sorted(
        (timestamp, frame) for frame, timestamp in timestamps_by_frame.items()
    )
    by_object: dict[Any, list[dict[str, Any]]] = {}
    for record in lifecycle_records.values():
        by_object.setdefault(record["object_id"], []).append(record)
    reactivation_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    next_reappearance: dict[tuple[Any, int], int] = {}
    for records in by_object.values():
        records.sort(
            key=lambda item: (
                int(item["lifecycle"]["first_timestamp_ns"]),
                int(item["lifecycle"]["index"]),
            )
        )
        for old, new in zip(records, records[1:]):
            old_last = int(old["lifecycle"]["last_timestamp_ns"])
            new_first = int(new["lifecycle"]["first_timestamp_ns"])
            if new_first >= old_last:
                next_reappearance[old["key"]] = new_first
                if int(new["lifecycle"]["index"]) == int(old["lifecycle"]["index"]) + 1:
                    reactivation_pairs.append((old, new))

    absent_by_frame: dict[int, list[dict[str, Any]]] = {}
    unavailable_absent_records: list[dict[str, Any]] = []
    for record in lifecycle_records.values():
        last_timestamp = int(record["lifecycle"]["last_timestamp_ns"])
        if last_timestamp >= _OPEN_ENDED_TIMESTAMP_NS:
            continue
        upper_bound = next_reappearance.get(record["key"])
        probe = next(
            (
                (timestamp, frame)
                for timestamp, frame in available_checkpoints
                if timestamp >= last_timestamp
                and (upper_bound is None or timestamp < upper_bound)
            ),
            None,
        )
        if probe is not None:
            timestamp, frame = probe
            absent_by_frame.setdefault(frame, []).append(
                {"record": record, "relative_timestamp_ns": timestamp}
            )
        else:
            unavailable_absent_records.append(record)

    events: list[dict[str, Any]] = []
    evaluation_frames = sorted(
        set(anchors_by_frame) | set(occluded_by_frame) | set(absent_by_frame)
    )
    for frame in evaluation_frames:
        checkpoint = _required(
            checkpoints,
            scene,
            frame,
            timestamps_by_frame[frame],
            checkpoint_relative_timestamp_ns,
        )
        entities = _frame_entities(checkpoint, float(voxel_size))
        for record in anchors_by_frame.get(frame, []):
            record["temporal_id"], record["mapping_evidence"] = _mapped_entity(
                entities, record["anchor_keys"]
            )
        for event in occluded_by_frame.get(frame, []):
            anchor_id = event["record"]["temporal_id"]
            current_id, current_evidence = _mapped_entity(
                entities, _target_keys(arrays[event["array"]])
            )
            counts = {
                "eligible": 1,
                "anchor_mapped": int(anchor_id is not None),
                "retained": 0,
                "false_release": 0,
                "false_reassignment": 0,
            }
            if anchor_id is None or current_id is None:
                counts["false_release"] = 1
            elif current_id == anchor_id:
                counts["retained"] = 1
            else:
                counts["false_reassignment"] = 1
            events.append(
                {
                    "event_id": f"{event['episode_id']}:occluded:{frame}",
                    "kind": "occluded",
                    "scene": scene,
                    "frame_index": frame,
                    "counts": counts,
                    "mapping_evidence": {
                        "anchor": event["record"]["mapping_evidence"],
                        "current": current_evidence,
                    },
                }
            )
        lifecycle_by_id = {entity_id: code for entity_id, code, _ in entities}
        for event in absent_by_frame.get(frame, []):
            record = event["record"]
            anchor_id = record["temporal_id"]
            counts = {
                "eligible": 1,
                "anchor_mapped": int(anchor_id is not None),
                "correct_stale_removal": 0,
                "stale_retention": 0,
                "missing_identity": 0,
            }
            if anchor_id is None or anchor_id not in lifecycle_by_id:
                counts["missing_identity"] = 1
            elif lifecycle_by_id[anchor_id] == 2:
                counts["correct_stale_removal"] = 1
            else:
                counts["stale_retention"] = 1
            lifecycle = record["lifecycle"]
            events.append(
                {
                    "event_id": (
                        f"{record['object_id']}:{lifecycle['index']}:absent:{frame}"
                    ),
                    "kind": "absent",
                    "scene": scene,
                    "frame_index": frame,
                    "counts": counts,
                    "mapping_evidence": {
                        "anchor": record["mapping_evidence"],
                        "current": _unavailable_mapping_evidence(
                            "absent_event_has_no_target_voxels"
                        ),
                    },
                }
            )
        del checkpoint, entities

    for record in unavailable_absent_records:
        lifecycle = record["lifecycle"]
        events.append(
            {
                "event_id": (
                    f"{record['object_id']}:{lifecycle['index']}:absent:unavailable"
                ),
                "kind": "absent",
                "scene": scene,
                "frame_index": None,
                "counts": {
                    "eligible": 0,
                    "anchor_mapped": int(record["temporal_id"] is not None),
                    "correct_stale_removal": 0,
                    "stale_retention": 0,
                    "missing_identity": 0,
                },
                "mapping_evidence": {
                    "anchor": record["mapping_evidence"],
                    "current": _unavailable_mapping_evidence(
                        "no_declared_post_lifecycle_checkpoint"
                    ),
                },
                "unavailable_reason": "no_declared_post_lifecycle_checkpoint",
            }
        )

    for old, new in reactivation_pairs:
        old_id = old["temporal_id"]
        new_id = new["temporal_id"]
        counts = {
            "eligible": 1,
            "anchor_mapped": int(old_id is not None),
            "same_id_reactivation": 0,
            "false_reassignment": 0,
            "missed_reactivation": 0,
        }
        if old_id is None or new_id is None:
            counts["missed_reactivation"] = 1
        elif new_id == old_id:
            counts["same_id_reactivation"] = 1
        else:
            counts["false_reassignment"] = 1
        anchor = new["anchor"]
        events.append(
            {
                "event_id": (
                    f"{old['object_id']}:{new['lifecycle']['index']}:reactivation:"
                    f"{anchor['frame_index']}"
                ),
                "kind": "reactivation",
                "scene": scene,
                "frame_index": int(anchor["frame_index"]),
                "counts": counts,
                "mapping_evidence": {
                    "anchor": old["mapping_evidence"],
                    "current": new["mapping_evidence"],
                },
            }
        )

    events.sort(
        key=lambda item: (
            item["frame_index"] is None,
            -1 if item["frame_index"] is None else item["frame_index"],
            item["kind"],
            item["event_id"],
        )
    )
    occluded_events = [item for item in events if item["kind"] == "occluded"]
    absent_events = [item for item in events if item["kind"] == "absent"]
    reactivation_events = [item for item in events if item["kind"] == "reactivation"]
    mapped_anchors = sum(
        int(record["temporal_id"] is not None) for record in lifecycle_records.values()
    )
    total_anchors = len(lifecycle_records)
    anchor_mappings = [
        {
            "scene": scene,
            "object_id": record["object_id"],
            "lifecycle_index": int(record["lifecycle"]["index"]),
            "anchor_frame_index": int(record["anchor"]["frame_index"]),
            "anchor_relative_timestamp_ns": int(
                record["anchor"]["relative_timestamp_ns"]
            ),
            "eligible": bool(record["anchor_keys"]),
            "target_voxel_count": len(record["anchor_keys"]),
            "mapped_temporal_id": record["temporal_id"],
            "overlap_voxel_count": record["mapping_evidence"][
                "overlap_voxel_count"
            ],
            "ambiguous": record["mapping_evidence"]["ambiguous"],
        }
        for record in lifecycle_records.values()
    ]
    result = {
        "format": EVALUATION_FORMAT,
        "mapping_rule": (
            "maximum_world_voxel_overlap_unique_winner_minimum_one_voxel"
        ),
        "anchor_mappings": anchor_mappings,
        "events": events,
        "macro": {
            "anchor_coverage_gate": _anchor_coverage_gate(
                scene, lifecycle_records
            ),
            "anchor_mapping_coverage": {
                "available": total_anchors > 0,
                "mapped": mapped_anchors,
                "total": total_anchors,
                "value": (
                    None
                    if total_anchors == 0
                    else float(mapped_anchors / total_anchors)
                ),
            },
            "occluded_retention_rate": _rate(
                sum(item["counts"]["retained"] for item in occluded_events),
                sum(item["counts"]["eligible"] for item in occluded_events),
                "no_eligible_occluded_events",
            ),
            "stale_removal_accuracy": _rate(
                sum(item["counts"]["correct_stale_removal"] for item in absent_events),
                sum(item["counts"]["eligible"] for item in absent_events),
                "no_eligible_absent_events",
            ),
            "reactivation_identity_accuracy": _rate(
                sum(
                    item["counts"]["same_id_reactivation"]
                    for item in reactivation_events
                ),
                sum(item["counts"]["eligible"] for item in reactivation_events),
                "no_eligible_reactivation_events",
            ),
        },
    }
    if mechanism_telemetry is not None:
        telemetry = {name: dict(value) for name, value in mechanism_telemetry.items()}
        result["mechanism_telemetry"] = telemetry
        result["macro"]["mechanism_telemetry"] = telemetry
    return result
