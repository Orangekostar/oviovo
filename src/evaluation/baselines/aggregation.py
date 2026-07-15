"""Frozen split aggregation for external baseline scene results."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import fmean
from typing import Any

REPLICA8_SCENES = (
    "room0",
    "room1",
    "room2",
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
)
REPLICA7_SCENES = REPLICA8_SCENES[1:]
STATIC_METRICS = {
    "semantic": ("miou", "macc", "f_miou"),
    "instance": ("ap25", "ap50"),
    "geometry": ("f5",),
}


def _macro(
    scene_results: Mapping[str, Mapping[str, Any]],
    scene_ids: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {"scene_ids": list(scene_ids), "scene_count": len(scene_ids)}
    for family, names in STATIC_METRICS.items():
        result[family] = {
            name: fmean(float(scene_results[scene_id]["metrics"][family][name]) for scene_id in scene_ids)
            for name in names
        }
    return result


def aggregate_replica_static(
    scene_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Macro-average exactly the frozen Replica-8 and Replica-7 scene sets."""
    if set(scene_results) != set(REPLICA8_SCENES):
        raise ValueError("Replica scene set must exactly match the frozen Replica-8 manifest")
    for scene_id, result in scene_results.items():
        embedded_scene_id = result.get("scene_id")
        if embedded_scene_id is not None and embedded_scene_id != scene_id:
            raise ValueError(f"scene_id mismatch for {scene_id}: {embedded_scene_id}")
    return {
        "replica_8_compat": _macro(scene_results, REPLICA8_SCENES),
        "replica_7_heldout": _macro(scene_results, REPLICA7_SCENES),
    }
