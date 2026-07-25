from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.oviv2.temporal_snapshot import TemporalCompactCheckpoint


EVALUATION_FORMAT = "oviv2_temporal_compact_v1"
_RETAINED_CODES = frozenset({0, 1})


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
    local = (checkpoint.voxel_keys[start:stop].astype(np.float64) + 0.5) * checkpoint.metadata.voxel_size_m
    if not len(local):
        return set()
    homogeneous = np.concatenate([local, np.ones((len(local), 1))], axis=1)
    world = (checkpoint.object_to_world[index] @ homogeneous.T).T[:, :3]
    quantized = np.floor(world / target_voxel_size_m).astype(np.int64)
    return {tuple(int(value) for value in row) for row in quantized}


def _mapped_entity(
    checkpoint: TemporalCompactCheckpoint,
    target_keys: set[tuple[int, int, int]],
    target_voxel_size_m: float,
    *,
    retained_only: bool,
) -> int | None:
    candidates: list[tuple[int, int]] = []
    for index, entity_id in enumerate(checkpoint.entity_ids.tolist()):
        if retained_only and int(checkpoint.lifecycle_codes[index]) not in _RETAINED_CODES:
            continue
        overlap = len(target_keys & _entity_world_keys(checkpoint, index, target_voxel_size_m))
        if overlap:
            candidates.append((overlap, int(entity_id)))
    if not candidates:
        return None
    maximum = max(item[0] for item in candidates)
    return min(entity_id for overlap, entity_id in candidates if overlap == maximum)


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


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "available": denominator > 0,
        "denominator": denominator,
        "value": None if denominator == 0 else float(numerator / denominator),
    }


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
    if isinstance(voxel_size, bool) or not isinstance(voxel_size, (int, float)) or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("target voxel_size_m is invalid")
    selected = [item for item in episodes if item.get("scene") == scene]
    if not selected:
        raise ValueError(f"target has no episodes for {scene}")

    lifecycle_records: dict[tuple[str, int], dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    for episode in selected:
        lifecycle = episode["lifecycle"]
        key = (str(episode["object_id"]), int(lifecycle["index"]))
        anchor = episode["anchor"]
        signature = {
            "object_id": episode["object_id"],
            "lifecycle": dict(lifecycle),
            "anchor": {
                "frame_index": anchor["frame_index"],
                "relative_timestamp_ns": anchor["relative_timestamp_ns"],
            },
        }
        previous = lifecycle_records.get(key)
        if previous is not None and any(previous[name] != signature[name] for name in signature):
            raise ValueError("target lifecycle anchor is inconsistent across episodes")
        if previous is None:
            anchor_checkpoint = _required(
                checkpoints, scene, int(anchor["frame_index"]), int(anchor["relative_timestamp_ns"]),
                checkpoint_relative_timestamp_ns,
            )
            anchor_id = _mapped_entity(
                anchor_checkpoint, _target_keys(arrays[str(anchor["array"])]), float(voxel_size),
                retained_only=True,
            )
            del anchor_checkpoint
            signature["temporal_id"] = anchor_id
            lifecycle_records[key] = signature
        else:
            anchor_id = previous["temporal_id"]
        for candidate in episode["checkpoints"]:
            current = _required(
                checkpoints, scene, int(candidate["frame_index"]),
                int(candidate["relative_timestamp_ns"]),
                checkpoint_relative_timestamp_ns,
            )
            current_id = _mapped_entity(
                current, _target_keys(arrays[str(candidate["array"])]), float(voxel_size),
                retained_only=True,
            )
            counts = {"eligible": int(anchor_id is not None), "retained": 0,
                      "false_release": 0, "false_reassignment": 0}
            if anchor_id is not None:
                same = np.flatnonzero(current.entity_ids == anchor_id)
                if len(same) and int(current.lifecycle_codes[int(same[0])]) in _RETAINED_CODES:
                    counts["retained"] = 1
                elif current_id is not None and current_id != anchor_id:
                    counts["false_reassignment"] = 1
                else:
                    counts["false_release"] = 1
            events.append({
                "event_id": f"{episode['episode_id']}:occluded:{candidate['frame_index']}",
                "kind": "occluded", "scene": scene,
                "frame_index": int(candidate["frame_index"]), "counts": counts,
            })
            del current

    by_object: dict[str, list[dict[str, Any]]] = {}
    for record in lifecycle_records.values():
        by_object.setdefault(str(record["object_id"]), []).append(record)
    if checkpoint_relative_timestamp_ns is None:
        available_checkpoints = []
        for candidate_scene, frame in checkpoints:
            if candidate_scene != scene:
                continue
            item = checkpoints[(candidate_scene, frame)]
            available_checkpoints.append(
                (round(item.metadata.timestamp * 1_000_000_000), frame)
            )
            del item
        available_checkpoints.sort()
    else:
        available_checkpoints = sorted(
            (timestamp, frame)
            for (candidate_scene, frame), timestamp in checkpoint_relative_timestamp_ns.items()
            if candidate_scene == scene
        )
    for records in by_object.values():
        records.sort(key=lambda item: int(item["lifecycle"]["index"]))
        for old, new in zip(records, records[1:]):
            if int(new["lifecycle"]["index"]) != int(old["lifecycle"]["index"]) + 1:
                continue
            old_id = old.get("temporal_id")
            old_last = int(old["lifecycle"]["last_timestamp_ns"])
            new_first = int(new["lifecycle"]["first_timestamp_ns"])
            probes = [item for item in available_checkpoints
                      if old_last <= item[0] < new_first]
            if probes:
                _, frame = probes[0]
                probe = checkpoints[(scene, frame)]
                counts = {"eligible": int(old_id is not None), "correct_stale_removal": 0,
                          "stale_retention": 0, "missing_identity": 0}
                if old_id is not None:
                    indices = np.flatnonzero(probe.entity_ids == old_id)
                    if not len(indices): counts["missing_identity"] = 1
                    elif int(probe.lifecycle_codes[int(indices[0])]) == 2: counts["correct_stale_removal"] = 1
                    else: counts["stale_retention"] = 1
                events.append({"event_id": f"{old['object_id']}:{old['lifecycle']['index']}:absent:{frame}",
                               "kind": "absent", "scene": scene, "frame_index": frame, "counts": counts})
                del probe
            anchor = new["anchor"]
            new_id = new.get("temporal_id")
            counts = {"eligible": int(old_id is not None), "same_id_reactivation": 0,
                      "false_reassignment": 0, "missed_reactivation": 0}
            if old_id is not None:
                if new_id == old_id: counts["same_id_reactivation"] = 1
                elif new_id is None: counts["missed_reactivation"] = 1
                else: counts["false_reassignment"] = 1
            events.append({"event_id": f"{old['object_id']}:{new['lifecycle']['index']}:reactivation:{anchor['frame_index']}",
                           "kind": "reactivation", "scene": scene,
                           "frame_index": int(anchor["frame_index"]), "counts": counts})

    events.sort(key=lambda item: (item["frame_index"], item["kind"], item["event_id"]))
    eligible_occluded = sum(item["counts"]["eligible"] for item in events if item["kind"] == "occluded")
    eligible_absent = sum(item["counts"]["eligible"] for item in events if item["kind"] == "absent")
    eligible_reactivation = sum(item["counts"]["eligible"] for item in events if item["kind"] == "reactivation")
    return {
        "format": EVALUATION_FORMAT,
        "mapping_rule": "maximum_world_voxel_overlap_then_lowest_temporal_id",
        "events": events,
        "macro": {
            "occluded_retention_rate": _rate(sum(item["counts"]["retained"] for item in events if item["kind"] == "occluded"), eligible_occluded),
            "stale_removal_accuracy": _rate(sum(item["counts"]["correct_stale_removal"] for item in events if item["kind"] == "absent"), eligible_absent),
            "reactivation_identity_accuracy": _rate(sum(item["counts"]["same_id_reactivation"] for item in events if item["kind"] == "reactivation"), eligible_reactivation),
        },
    }
