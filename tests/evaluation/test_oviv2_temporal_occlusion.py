from collections.abc import Iterator, Mapping

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


def _two_entity_checkpoint(
    frame: int, *, move_first_entity: bool = True
) -> TemporalCompactCheckpoint:
    moved = np.eye(4)
    if move_first_entity:
        moved[0, 3] = 0.5
    return TemporalCompactCheckpoint(
        metadata=TemporalSnapshotMetadata(
            "apartment", frame, frame / 10.0, frame + 1, 0.05, "a" * 64
        ),
        entity_ids=np.asarray([7, 9], dtype=np.int64),
        lifecycle_codes=np.asarray([1, 1], dtype=np.uint8),
        existence_log_odds=np.asarray([1.0, 1.0], dtype=np.float64),
        absent_streaks=np.asarray([0, 0], dtype=np.int64),
        distinct_view_bin_counts=np.asarray([0, 0], dtype=np.int64),
        object_to_world=np.asarray([moved, np.eye(4)], dtype=np.float64),
        voxel_keys=np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.int64),
        voxel_offsets=np.asarray([0, 1, 2], dtype=np.int64),
    )


def _tied_checkpoint(frame: int) -> TemporalCompactCheckpoint:
    return _two_entity_checkpoint(frame, move_first_entity=False)


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
        "eligible": 1, "anchor_mapped": 1, "retained": 1, "false_release": 0,
        "false_reassignment": 0,
    }
    absent = next(item for item in result["events"] if item["kind"] == "absent")
    assert absent["counts"]["correct_stale_removal"] == 1
    reactivation = next(item for item in result["events"] if item["kind"] == "reactivation")
    assert reactivation["counts"] == {
        "eligible": 1, "anchor_mapped": 1, "same_id_reactivation": 1,
        "false_reassignment": 0, "missed_reactivation": 0,
    }
    assert result["macro"]["occluded_retention_rate"] == {
        "available": True, "denominator": 3, "value": 2 / 3
    }
    assert result["macro"]["anchor_mapping_coverage"] == {
        "available": True, "mapped": 2, "total": 3, "value": 2 / 3
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
        "available": False, "denominator": 0, "value": None,
        "unavailable_reason": "no_eligible_reactivation_events",
    }


def test_current_target_mapping_takes_priority_over_old_id_active_elsewhere() -> None:
    arrays, metadata = _target()
    metadata = dict(metadata)
    metadata["episodes"] = [metadata["episodes"][0]]
    result = evaluate_temporal_occlusion(
        arrays=arrays,
        metadata=metadata,
        checkpoints={
            ("apartment", 0): _checkpoint(0, 0),
            ("apartment", 1): _two_entity_checkpoint(1),
        },
        scene="apartment",
    )
    event = result["events"][0]
    assert event["counts"] == {
        "eligible": 1,
        "anchor_mapped": 1,
        "retained": 0,
        "false_release": 0,
        "false_reassignment": 1,
    }
    assert event["mapping_evidence"]["anchor"] == {
        "available": True,
        "mapped_temporal_id": 7,
        "overlap_voxel_count": 1,
        "target_voxel_count": 1,
        "target_coverage": 1.0,
        "ambiguous": False,
        "minimum_overlap_voxel_count": 1,
        "unavailable_reason": None,
    }
    assert event["mapping_evidence"]["current"]["mapped_temporal_id"] == 9


def test_exact_top_overlap_tie_is_ambiguous_unmapped_and_counts_as_miss() -> None:
    arrays, metadata = _target()
    metadata = dict(metadata)
    metadata["episodes"] = [metadata["episodes"][0]]
    result = evaluate_temporal_occlusion(
        arrays=arrays,
        metadata=metadata,
        checkpoints={
            ("apartment", 0): _tied_checkpoint(0),
            ("apartment", 1): _checkpoint(1, 1),
        },
        scene="apartment",
    )
    event = result["events"][0]
    assert result["mapping_rule"] == (
        "maximum_world_voxel_overlap_unique_winner_minimum_one_voxel"
    )
    assert event["counts"]["anchor_mapped"] == 0
    assert event["counts"]["false_release"] == 1
    assert event["mapping_evidence"]["anchor"] == {
        "available": True,
        "mapped_temporal_id": None,
        "overlap_voxel_count": 1,
        "target_voxel_count": 1,
        "target_coverage": 1.0,
        "ambiguous": True,
        "minimum_overlap_voxel_count": 1,
        "unavailable_reason": None,
    }


def test_unmapped_anchor_counts_as_miss_and_remains_in_macro_denominator() -> None:
    arrays, metadata = _target()
    metadata = dict(metadata)
    metadata["episodes"] = [metadata["episodes"][1]]
    result = evaluate_temporal_occlusion(
        arrays=arrays,
        metadata=metadata,
        checkpoints={("apartment", 2): _checkpoint(2, 1)},
        scene="apartment",
    )
    assert result["events"][0]["counts"] == {
        "eligible": 1,
        "anchor_mapped": 0,
        "retained": 0,
        "false_release": 1,
        "false_reassignment": 0,
    }
    assert result["macro"]["occluded_retention_rate"] == {
        "available": True, "denominator": 1, "value": 0.0
    }
    assert result["macro"]["anchor_mapping_coverage"] == {
        "available": True, "mapped": 0, "total": 1, "value": 0.0
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
    assert len([item for item in result["events"] if item["kind"] == "occluded"]) == 2
    assert len([item for item in result["events"] if item["kind"] == "absent"]) == 1
    assert result["macro"]["anchor_mapping_coverage"]["total"] == 1


def test_finite_lifecycle_without_reappearance_gets_absent_probe_not_reactivation() -> None:
    arrays, metadata = _target()
    metadata = dict(metadata)
    metadata["episodes"] = metadata["episodes"][:2]
    result = evaluate_temporal_occlusion(
        arrays=arrays, metadata=metadata,
        checkpoints={("apartment", 0): _checkpoint(0, 0),
                     ("apartment", 1): _checkpoint(1, 1),
                     ("apartment", 2): _checkpoint(2, 2)},
        checkpoint_relative_timestamp_ns={
            ("apartment", 0): 0,
            ("apartment", 1): 100_000_000,
            ("apartment", 2): 200_000_000,
        },
        scene="apartment",
    )
    absent = [item for item in result["events"] if item["kind"] == "absent"]
    assert len(absent) == 1
    assert absent[0]["frame_index"] == 2
    assert absent[0]["counts"]["correct_stale_removal"] == 1
    assert not [item for item in result["events"] if item["kind"] == "reactivation"]


def test_finite_lifecycle_at_scene_end_gets_explicit_unavailable_absent_event() -> None:
    arrays, metadata = _target()
    metadata = dict(metadata)
    metadata["episodes"] = [metadata["episodes"][0]]
    result = evaluate_temporal_occlusion(
        arrays=arrays,
        metadata=metadata,
        checkpoints={
            ("apartment", 0): _checkpoint(0, 0),
            ("apartment", 1): _checkpoint(1, 1),
        },
        scene="apartment",
    )
    absent = [item for item in result["events"] if item["kind"] == "absent"]
    assert len(absent) == 1
    assert absent[0]["frame_index"] is None
    assert absent[0]["counts"]["eligible"] == 0
    assert absent[0]["unavailable_reason"] == (
        "no_declared_post_lifecycle_checkpoint"
    )
    assert result["macro"]["stale_removal_accuracy"] == {
        "available": False,
        "denominator": 0,
        "value": None,
        "unavailable_reason": "no_eligible_absent_events",
    }


def test_unavailable_macros_have_stable_reasons() -> None:
    arrays, metadata = _target()
    metadata = dict(metadata)
    metadata["episodes"] = [metadata["episodes"][1]]
    result = evaluate_temporal_occlusion(
        arrays=arrays, metadata=metadata,
        checkpoints={("apartment", 2): _checkpoint(2, 2)}, scene="apartment"
    )
    assert result["macro"]["stale_removal_accuracy"] == {
        "available": False, "denominator": 0, "value": None,
        "unavailable_reason": "no_eligible_absent_events",
    }
    assert result["macro"]["reactivation_identity_accuracy"] == {
        "available": False, "denominator": 0, "value": None,
        "unavailable_reason": "no_eligible_reactivation_events",
    }


class _CountingCheckpoints(Mapping[tuple[str, int], TemporalCompactCheckpoint]):
    def __init__(self, values: dict[tuple[str, int], TemporalCompactCheckpoint]) -> None:
        self.values = values
        self.loads: dict[tuple[str, int], int] = {}

    def __getitem__(self, key: tuple[str, int]) -> TemporalCompactCheckpoint:
        self.loads[key] = self.loads.get(key, 0) + 1
        return self.values[key]

    def __iter__(self) -> Iterator[tuple[str, int]]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def test_repeated_events_are_evaluated_with_one_load_per_unique_frame() -> None:
    arrays, metadata = _target()
    episodes = []
    for index in range(20):
        repeated = dict(metadata["episodes"][0])
        repeated["episode_id"] = f"repeat-{index:02d}"
        episodes.append(repeated)
    metadata = dict(metadata)
    metadata["episodes"] = episodes
    checkpoints = _CountingCheckpoints({
        ("apartment", 0): _checkpoint(0, 0),
        ("apartment", 1): _checkpoint(1, 1),
    })
    result = evaluate_temporal_occlusion(
        arrays=arrays, metadata=metadata, checkpoints=checkpoints,
        checkpoint_relative_timestamp_ns={
            ("apartment", 0): 0, ("apartment", 1): 100_000_000,
        }, scene="apartment",
    )
    assert len([item for item in result["events"] if item["kind"] == "occluded"]) == 20
    assert checkpoints.loads == {("apartment", 0): 1, ("apartment", 1): 1}


def test_repeated_events_precompute_each_entity_world_voxels_once_per_frame(
    monkeypatch: object,
) -> None:
    import src.evaluation.oviv2_temporal_occlusion as module

    arrays, metadata = _target()
    repeated = dict(metadata["episodes"][0])
    metadata = dict(metadata)
    metadata["episodes"] = [dict(repeated, episode_id=f"repeat-{index}") for index in range(20)]
    original = module._entity_world_keys
    calls = 0

    def counted(*args: object, **kwargs: object) -> set[tuple[int, int, int]]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_entity_world_keys", counted)
    evaluate_temporal_occlusion(
        arrays=arrays,
        metadata=metadata,
        checkpoints={
            ("apartment", 0): _checkpoint(0, 0),
            ("apartment", 1): _checkpoint(1, 1),
        },
        scene="apartment",
    )
    assert calls == 2
