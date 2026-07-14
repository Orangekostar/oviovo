from __future__ import annotations

import numpy as np

from src.core.data_structures import DenseSurfaceEntry, ObjectMap, ObjectState, SystemState
from src.evaluation.exporters.oviovo import (
    export_map_snapshot,
    read_map_snapshot,
    write_map_snapshot,
)


def _object(object_id: int, state: ObjectState, x: float, label: str) -> ObjectMap:
    obj = ObjectMap(
        object_id=object_id,
        state=state,
        local_pcd=np.array([[x, 0.0, 1.0], [x + 0.05, 0.0, 1.0]], dtype=np.float32),
        creation_frame=1,
        last_seen_frame=5,
        update_count=2,
    )
    obj.semantic_memory.aggregated_feature = np.array([1.0, float(object_id)], dtype=np.float32)
    obj.debug["anchor_semantics"] = {
        "semantic_state": "committed",
        "canonical_label": label,
        "committed_label": label,
        "canonical_score": 0.9,
        "commit_reason": "test",
    }
    return obj


def _state() -> SystemState:
    state = SystemState(frame_count=6)
    state.objects = {
        1: _object(1, ObjectState.ACTIVE, 0.0, "chair"),
        2: _object(2, ObjectState.DORMANT, 1.0, "book"),
        3: _object(3, ObjectState.REMOVED, 2.0, "lamp"),
    }
    state.background.point_cloud = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    state.dense_surface_map.entries[1] = DenseSurfaceEntry(
        points=np.array([[99.0, 99.0, 99.0]], dtype=np.float32),
        object_id=1,
        semantic_label="wrong-dense-export",
    )
    return state


def test_current_snapshot_excludes_non_current_entities_but_history_retains_them() -> None:
    state = _state()
    current = export_map_snapshot(state, method="OVIOVO", scene_id="scene-a", timestamp=6.0, scope="current")
    history = export_map_snapshot(state, method="OVIOVO", scene_id="scene-a", timestamp=6.0, scope="history")

    assert [entity.entity_id for entity in current.entities] == ["1"]
    assert [entity.entity_id for entity in history.entities] == ["1", "2", "3"]
    assert current.entities[0].semantic_label == "chair"
    assert current.entities[0].lifecycle_state == "active"
    assert np.array_equal(current.entities[0].points_xyz, state.objects[1].local_pcd)
    assert not np.any(current.entities[0].points_xyz == 99.0)


def test_exported_snapshot_is_immutable_against_future_state_updates() -> None:
    state = _state()
    snapshot_at_t = export_map_snapshot(
        state,
        method="OVIOVO",
        scene_id="scene-a",
        timestamp=6.0,
        scope="current",
    )
    points_at_t = snapshot_at_t.entities[0].points_xyz.copy()

    state.objects[1].local_pcd[:] = 42.0
    state.objects[4] = _object(4, ObjectState.ACTIVE, 4.0, "table")
    state.frame_count = 7

    assert np.array_equal(snapshot_at_t.entities[0].points_xyz, points_at_t)
    assert [entity.entity_id for entity in snapshot_at_t.entities] == ["1"]


def test_dense_export_cache_does_not_change_authoritative_snapshot() -> None:
    state = _state()
    with_dense = export_map_snapshot(state, method="OVIOVO", scene_id="scene-a", timestamp=6.0)
    state.dense_surface_map.entries.clear()
    without_dense = export_map_snapshot(state, method="OVIOVO", scene_id="scene-a", timestamp=6.0)

    assert np.array_equal(with_dense.entities[0].points_xyz, without_dense.entities[0].points_xyz)
    assert with_dense.entities[0].semantic_label == without_dense.entities[0].semantic_label


def test_snapshot_file_round_trip_uses_npz_for_arrays_and_jsonl_for_metadata(tmp_path) -> None:
    snapshot = export_map_snapshot(
        _state(),
        method="OVIOVO",
        scene_id="scene-a",
        timestamp=6.0,
        scope="history",
        runtime={"total_s": 0.25},
    )
    paths = write_map_snapshot(snapshot, tmp_path / "OVIOVO" / "scene-a")

    assert paths["snapshot"].suffix == ".npz"
    assert paths["entities"].suffix == ".jsonl"
    assert paths["snapshot"].parent.name == "snapshots"
    assert paths["entities"].parent.name == "entities"

    loaded = read_map_snapshot(paths["snapshot"], paths["entities"])
    assert loaded.method == snapshot.method
    assert loaded.scene_id == snapshot.scene_id
    assert loaded.scope == "history"
    assert loaded.runtime == {"total_s": 0.25}
    assert [entity.entity_id for entity in loaded.entities] == ["1", "2", "3"]
    for expected, actual in zip(snapshot.entities, loaded.entities, strict=True):
        assert np.array_equal(actual.points_xyz, expected.points_xyz)
        assert np.array_equal(actual.semantic_embedding, expected.semantic_embedding)
        assert actual.metadata == expected.metadata
