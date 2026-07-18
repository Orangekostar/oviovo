from __future__ import annotations

import pytest

from src.evaluation.baselines.aggregation import aggregate_replica_static


def _scene(value: float) -> dict:
    return {
        "metrics": {
            "semantic": {"miou": value, "macc": value + 0.1, "f_miou": value + 0.2},
            "instance": {"ap25": value + 0.3, "ap50": value + 0.4},
            "geometry": {"f5": value + 0.5},
        }
    }


def test_aggregate_replica_static_computes_frozen_split_macros() -> None:
    scene_ids = ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")
    scenes = {scene_id: _scene(float(index)) for index, scene_id in enumerate(scene_ids)}

    aggregate = aggregate_replica_static(scenes)

    assert aggregate["replica_8_compat"]["scene_ids"] == list(scene_ids)
    assert aggregate["replica_8_compat"]["semantic"]["miou"] == pytest.approx(3.5)
    assert aggregate["replica_7_heldout"]["semantic"]["miou"] == pytest.approx(4.0)
    assert aggregate["replica_7_heldout"]["instance"]["ap50"] == pytest.approx(4.4)
    assert aggregate["replica_8_compat"]["geometry"]["f5"] == pytest.approx(4.0)


def test_aggregate_replica_static_preserves_unavailable_metric_family() -> None:
    scene_ids = ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")
    scenes = {scene_id: _scene(float(index)) for index, scene_id in enumerate(scene_ids)}
    for scene in scenes.values():
        scene["metrics"]["instance"] = {
            "status": "N/A",
            "reason": "OpenFusion has no native entity instances.",
        }

    aggregate = aggregate_replica_static(scenes)

    assert aggregate["replica_8_compat"]["instance"] == {
        "status": "N/A",
        "reason": "OpenFusion has no native entity instances.",
    }
    assert aggregate["replica_7_heldout"]["semantic"]["miou"] == pytest.approx(4.0)


def test_aggregate_replica_static_rejects_missing_or_mismatched_scene() -> None:
    scenes = {"room0": _scene(0.0)}
    with pytest.raises(ValueError, match="scene set"):
        aggregate_replica_static(scenes)

    scenes = {
        scene_id: _scene(0.0)
        for scene_id in ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")
    }
    scenes["room1"]["scene_id"] = "office4"
    with pytest.raises(ValueError, match="scene_id"):
        aggregate_replica_static(scenes)
