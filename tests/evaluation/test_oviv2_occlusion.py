from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (
    evaluate_fixed_anchor_ownership,
)
from src.oviv2.ownership import ReversibleOwnershipStore


def _snapshot(
    scene: str,
    frame_index: int,
    owners: dict[tuple[int, int, int], int],
) -> SimpleNamespace:
    ownership = ReversibleOwnershipStore()
    for key, entity_id in sorted(owners.items()):
        ownership.assign(key, entity_id, 1.0, frame_index + 1)
    return SimpleNamespace(
        metadata=SimpleNamespace(scene_id=scene, frame_id=frame_index),
        ownership=ownership,
    )


def _episode(
    episode_id: str,
    scene: str,
    *,
    anchor_array: str,
    checkpoint_array: str,
    fraction: float,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "scene": scene,
        "object_id": f"gt-{episode_id}",
        "object_name": "chair",
        "semantic_label": 5,
        "lifecycle": {
            "index": 0,
            "first_timestamp_ns": 0,
            "last_timestamp_ns": 10,
        },
        "anchor": {
            "frame_index": 0,
            "relative_timestamp_ns": 0,
            "array": anchor_array,
            "voxel_count": 4,
        },
        "start_frame_index": 1,
        "end_frame_index": 1,
        "checkpoints": [
            {
                "frame_index": 1,
                "relative_timestamp_ns": 1,
                "array": checkpoint_array,
                "occluded_voxel_count": 4,
                "occlusion_fraction": fraction,
            }
        ],
        "occlusion_fraction": fraction,
    }


def _metadata(episodes: list[dict[str, object]]) -> dict[str, object]:
    return {
        "episodes": episodes,
        "stress_layers": {
            "all": {
                "episode_count": len(episodes),
                "episode_ids": [str(item["episode_id"]) for item in episodes],
            },
            **{
                f"{threshold:.2f}": {
                    "episode_count": sum(
                        float(item["occlusion_fraction"]) >= threshold
                        for item in episodes
                    ),
                    "episode_ids": [
                        str(item["episode_id"])
                        for item in episodes
                        if float(item["occlusion_fraction"]) >= threshold
                    ],
                    "minimum_occlusion_fraction": threshold,
                }
                for threshold in (0.50, 0.75, 0.90)
            },
        },
    }


def test_fixed_anchor_majority_tie_uses_lower_owner_and_never_rematches() -> None:
    keys = [(index, 0, 20) for index in range(4)]
    arrays = {
        "apartment.anchor": np.asarray(keys, dtype=np.int64),
        "apartment.occluded": np.asarray(keys, dtype=np.int64),
    }
    episodes = [
        _episode(
            "apartment-e0",
            "apartment",
            anchor_array="apartment.anchor",
            checkpoint_array="apartment.occluded",
            fraction=1.0,
        )
    ]
    snapshots = {
        ("apartment", 0): _snapshot(
            "apartment", 0, {keys[0]: 7, keys[1]: 7, keys[2]: 8, keys[3]: 8}
        ),
        # Owner 8 becomes the current majority, but mapping must remain fixed at 7.
        ("apartment", 1): _snapshot(
            "apartment", 1, {keys[0]: 7, keys[2]: 8, keys[3]: 8}
        ),
    }

    result = evaluate_fixed_anchor_ownership(
        arrays=arrays,
        metadata=_metadata(episodes),
        snapshots=snapshots,
    )

    assert result["fixed_anchor_mappings"] == [
        {
            "episode_id": "apartment-e0",
            "gt_object_id": "gt-apartment-e0",
            "predicted_owner_id": 7,
            "majority_count": 2,
            "tie": True,
        }
    ]
    metrics = result["stress_layers"]["all"]
    assert metrics["all_gt_occluded_target_voxels"] == 4
    assert metrics["anchor_owned_target_voxels"] == 2
    assert metrics["retained_owner_count"] == 1
    assert metrics["false_release_count"] == 1
    assert metrics["false_reassignment_count"] == 0
    assert metrics["false_release_rate"] == 0.5
    assert metrics["false_reassignment_rate"] == 0.0
    assert metrics["retained_ownership_recall"] == 0.5
    assert metrics["gt_retained_object_recall"] == 0.25


def test_reports_finite_voxel_object_episode_metrics_for_all_stress_layers() -> None:
    key_sets = {
        "low": [(0, 0, 20), (1, 0, 20), (2, 0, 20), (3, 0, 20)],
        "headline": [(4, 0, 20), (5, 0, 20), (6, 0, 20), (7, 0, 20)],
    }
    arrays = {
        "low.anchor": np.asarray(key_sets["low"], dtype=np.int64),
        "low.occluded": np.asarray(key_sets["low"], dtype=np.int64),
        "headline.anchor": np.asarray(key_sets["headline"], dtype=np.int64),
        "headline.occluded": np.asarray(key_sets["headline"], dtype=np.int64),
    }
    low = _episode(
        "apartment-low",
        "apartment",
        anchor_array="low.anchor",
        checkpoint_array="low.occluded",
        fraction=0.60,
    )
    headline = _episode(
        "office-headline",
        "office",
        anchor_array="headline.anchor",
        checkpoint_array="headline.occluded",
        fraction=0.95,
    )
    snapshots = {
        ("apartment", 0): _snapshot("apartment", 0, {}),
        ("apartment", 1): _snapshot("apartment", 1, {}),
        ("office", 0): _snapshot(
            "office", 0, {key: 3 for key in key_sets["headline"]}
        ),
        ("office", 1): _snapshot(
            "office",
            1,
            {
                key: 4 if index == 0 else 3
                for index, key in enumerate(key_sets["headline"])
            },
        ),
    }

    result = evaluate_fixed_anchor_ownership(
        arrays=arrays,
        metadata=_metadata([low, headline]),
        snapshots=snapshots,
    )

    assert result["stress_layers"]["0.50"]["episode_count"] == 2
    assert result["stress_layers"]["0.75"]["episode_count"] == 1
    assert result["stress_layers"]["0.90"]["episode_count"] == 1
    assert result["stress_layers"]["0.90"]["false_reassignment_count"] == 1
    assert result["stress_layers"]["0.90"]["zero_release_episode_rate"] == 1.0
    assert result["headline_gate"] == {
        "stress_layer": "0.90",
        "episode_count": 1,
        "anchor_mapped_episode_count": 1,
        "anchor_owned_target_voxels": 4,
        "false_release_count": 0,
        "passed": True,
    }
    for metrics in result["stress_layers"].values():
        for name, value in metrics.items():
            if name.endswith("rate") or name.endswith("recall"):
                assert np.isfinite(value)


def test_headline_gate_fails_closed_for_unmapped_episode() -> None:
    keys = [(0, 0, 20), (1, 0, 20), (2, 0, 20), (3, 0, 20)]
    arrays = {
        "anchor": np.asarray(keys, dtype=np.int64),
        "occluded": np.asarray(keys, dtype=np.int64),
    }
    episode = _episode(
        "apartment-e0",
        "apartment",
        anchor_array="anchor",
        checkpoint_array="occluded",
        fraction=1.0,
    )

    result = evaluate_fixed_anchor_ownership(
        arrays=arrays,
        metadata=_metadata([episode]),
        snapshots={
            ("apartment", 0): _snapshot("apartment", 0, {}),
            ("apartment", 1): _snapshot("apartment", 1, {}),
        },
    )

    headline = result["stress_layers"]["0.90"]
    assert headline["zero_release_episode_count"] == 0
    assert headline["zero_release_episode_rate"] == 0.0
    assert result["headline_gate"]["passed"] is False


def test_rejects_missing_and_future_checkpoints() -> None:
    keys = [(0, 0, 20), (1, 0, 20), (2, 0, 20), (3, 0, 20)]
    arrays = {
        "anchor": np.asarray(keys, dtype=np.int64),
        "occluded": np.asarray(keys, dtype=np.int64),
    }
    episode = _episode(
        "apartment-e0",
        "apartment",
        anchor_array="anchor",
        checkpoint_array="occluded",
        fraction=1.0,
    )
    metadata = _metadata([episode])
    anchor = _snapshot("apartment", 0, {key: 7 for key in keys})

    with pytest.raises(ValueError, match="missing checkpoint"):
        evaluate_fixed_anchor_ownership(
            arrays=arrays,
            metadata=metadata,
            snapshots={("apartment", 0): anchor},
        )

    with pytest.raises(ValueError, match="future snapshot"):
        evaluate_fixed_anchor_ownership(
            arrays=arrays,
            metadata=metadata,
            snapshots={
                ("apartment", 0): anchor,
                ("apartment", 1): _snapshot(
                    "apartment", 2, {key: 7 for key in keys}
                ),
            },
        )
