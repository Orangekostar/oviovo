from __future__ import annotations

import numpy as np

from src.evaluation.oviv2_temporal_occlusion import evaluate_temporal_occlusion
from src.oviv2.temporal_snapshot import (
    TemporalCompactCheckpoint,
    TemporalSnapshotMetadata,
)


def _checkpoint(frame: int, lifecycle: int, entity_id: int = 7) -> TemporalCompactCheckpoint:
    return TemporalCompactCheckpoint(
        metadata=TemporalSnapshotMetadata(
            "apartment", frame, frame / 10.0, frame + 1, 0.05, "a" * 64
        ),
        entity_ids=np.asarray([entity_id], dtype=np.int64),
        lifecycle_codes=np.asarray([lifecycle], dtype=np.uint8),
        existence_log_odds=np.asarray([1.0], dtype=np.float64),
        absent_streaks=np.asarray([0], dtype=np.int64),
        distinct_view_bin_counts=np.asarray([0], dtype=np.int64),
        object_to_world=np.asarray([np.eye(4)], dtype=np.float64),
        voxel_keys=np.asarray([[0, 0, 0]], dtype=np.int64),
        voxel_offsets=np.asarray([0, 1], dtype=np.int64),
    )


def _target() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays = {
        "life0.anchor": np.asarray([[0, 0, 0]], dtype=np.int64),
        "life0.occluded": np.asarray([[0, 0, 0]], dtype=np.int64),
        "probe.anchor": np.asarray([[10, 10, 10]], dtype=np.int64),
        "probe.occluded": np.asarray([[10, 10, 10]], dtype=np.int64),
        "life1.anchor": np.asarray([[0, 0, 0]], dtype=np.int64),
        "life1.occluded": np.asarray([[0, 0, 0]], dtype=np.int64),
    }
    def episode(name: str, obj: int, life: int, anchor: int, check: int,
                first: int, last: int, array: str) -> dict[str, object]:
        return {
            "episode_id": name,
            "scene": "apartment",
            "object_id": obj,
            "lifecycle": {
                "index": life,
                "first_timestamp_ns": first,
                "last_timestamp_ns": last,
            },
            "anchor": {
                "frame_index": anchor,
                "relative_timestamp_ns": anchor * 100_000_000,
                "array": f"{array}.anchor",
            },
            "checkpoints": [{
                "frame_index": check,
                "relative_timestamp_ns": check * 100_000_000,
                "array": f"{array}.occluded",
            }],
        }
    metadata = {
        "parameters": {"voxel_size_m": 0.05},
        "episodes": [
            episode("life0", 11, 0, 0, 1, 0, 200_000_000, "life0"),
            episode("probe", 99, 0, 2, 2, 0, 2**64 - 1, "probe"),
            episode("life1", 11, 1, 3, 4, 300_000_000, 2**64 - 1, "life1"),
        ],
    }
    return arrays, metadata


def test_scores_retention_absence_and_same_id_reactivation_per_event_before_macro() -> None:
    arrays, metadata = _target()
    result = evaluate_temporal_occlusion(
        arrays=arrays,
        metadata=metadata,
        checkpoints={
            ("apartment", 0): _checkpoint(0, 0),
            ("apartment", 1): _checkpoint(1, 1),
            ("apartment", 2): _checkpoint(2, 2),
            ("apartment", 3): _checkpoint(3, 0),
            ("apartment", 4): _checkpoint(4, 1),
        },
        scene="apartment",
    )
    assert result["format"] == "oviv2_temporal_compact_v1"
    assert [item["kind"] for item in result["events"]].count("occluded") == 3
    life0 = next(item for item in result["events"] if item["event_id"] == "life0:occluded:1")
    assert life0["counts"] == {
        "eligible": 1, "retained": 1, "false_release": 0,
        "false_reassignment": 0,
    }
    absent = next(item for item in result["events"] if item["kind"] == "absent")
    assert absent["counts"]["correct_stale_removal"] == 1
    reactivation = next(item for item in result["events"] if item["kind"] == "reactivation")
    assert reactivation["counts"] == {
        "eligible": 1, "same_id_reactivation": 1,
        "false_reassignment": 0, "missed_reactivation": 0,
    }
    assert result["macro"]["occluded_retention_rate"] == {
        "available": True, "denominator": 2, "value": 1.0
    }


def test_new_id_reactivation_is_reassignment_and_empty_denominator_is_unavailable() -> None:
    arrays, metadata = _target()
    checkpoints = {
        ("apartment", 0): _checkpoint(0, 0),
        ("apartment", 1): _checkpoint(1, 2),
        ("apartment", 2): _checkpoint(2, 2),
        ("apartment", 3): _checkpoint(3, 0, entity_id=9),
        ("apartment", 4): _checkpoint(4, 0, entity_id=9),
    }
    result = evaluate_temporal_occlusion(
        arrays=arrays, metadata=metadata, checkpoints=checkpoints, scene="apartment"
    )
    life0 = next(item for item in result["events"] if item["event_id"] == "life0:occluded:1")
    assert life0["counts"]["false_release"] == 1
    reactivation = next(item for item in result["events"] if item["kind"] == "reactivation")
    assert reactivation["counts"]["false_reassignment"] == 1

    metadata = dict(metadata)
    metadata["episodes"] = [metadata["episodes"][1]]
    unavailable = evaluate_temporal_occlusion(
        arrays=arrays, metadata=metadata,
        checkpoints={("apartment", 2): _checkpoint(2, 2)}, scene="apartment"
    )
    assert unavailable["macro"]["reactivation_identity_accuracy"] == {
        "available": False, "denominator": 0, "value": None
    }


def test_multiple_occlusion_episodes_share_one_lifecycle_anchor() -> None:
    arrays, metadata = _target()
    repeated = dict(metadata["episodes"][0])
    repeated["episode_id"] = "life0-repeat"
    metadata = dict(metadata)
    metadata["episodes"] = [metadata["episodes"][0], repeated]
    result = evaluate_temporal_occlusion(
        arrays=arrays, metadata=metadata,
        checkpoints={("apartment", 0): _checkpoint(0, 0),
                     ("apartment", 1): _checkpoint(1, 1)},
        scene="apartment",
    )
    assert len(result["events"]) == 2
