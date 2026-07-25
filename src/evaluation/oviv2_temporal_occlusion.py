from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.oviv2.temporal_snapshot import TemporalCompactCheckpoint


EVALUATION_FORMAT = "oviv2_temporal_compact_v1"
_RETAINED_CODES = frozenset({0, 1})
_OPEN_ENDED_TIMESTAMP_NS = 2**64 - 1


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


def evaluate_temporal_occlusion(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    checkpoints: Mapping[tuple[str, int], TemporalCompactCheckpoint],
    scene: str,
    checkpoint_relative_timestamp_ns: Mapping[tuple[str, int], int] | None = None,
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
        signature = {
            "key": key,
            "object_id": episode["object_id"],
            "lifecycle": dict(lifecycle),
            "anchor": {
                "frame_index": anchor_frame,
                "relative_timestamp_ns": anchor_timestamp,
                "array": str(anchor["array"]),
            },
            "temporal_id": None,
            "mapping_evidence": None,
        }
        previous = lifecycle_records.get(key)
        if previous is not None and any(
            previous[name] != signature[name]
            for name in ("object_id", "lifecycle", "anchor")
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
                entities, _target_keys(arrays[record["anchor"]["array"]])
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
    return {
        "format": EVALUATION_FORMAT,
        "mapping_rule": (
            "maximum_world_voxel_overlap_unique_winner_minimum_one_voxel"
        ),
        "events": events,
        "macro": {
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
