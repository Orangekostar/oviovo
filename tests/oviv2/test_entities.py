from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.oviv2.entities import EntityRegistry, EntityRegistryConfig
from src.oviv2.tracking import LocalTracker, LocalTrackerConfig

from tests.oviv2.test_tracking import observation


def _track(frame_id: int, keys: set[tuple[int, int, int]], *, label: str = "chair", semantic_id: int = 2):
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
    return tracker.update((observation(frame_id, keys, label=label, semantic_id=semantic_id),), frame_id).accepted[0]


def test_registry_allocates_stable_ids_by_geometry_and_updates_views() -> None:
    registry = EntityRegistry(EntityRegistryConfig(min_voxel_overlap=0.25))
    first = registry.resolve(_track(0, {(0, 0, 20), (1, 0, 20)}), revision=1)
    second = registry.resolve(_track(1, {(1, 0, 20), (2, 0, 20)}), revision=2)

    assert first.entity_id == second.entity_id == 1
    assert registry.entities[1].accepted_view_count == 2
    assert registry.entities[1].voxel_keys == frozenset({(0, 0, 20), (1, 0, 20), (2, 0, 20)})


def test_registry_rejects_geometry_mismatch_even_when_semantics_agree() -> None:
    registry = EntityRegistry(EntityRegistryConfig(max_centroid_distance_m=0.2))
    first = registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
    second = registry.resolve(_track(1, {(100, 0, 20)}), revision=2)

    assert first.entity_id == 1
    assert second.entity_id == 2


def test_registry_counts_distinct_supporting_frames_not_tracks() -> None:
    registry = EntityRegistry()
    first = registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
    second = registry.resolve(_track(0, {(0, 0, 20), (1, 0, 20)}), revision=2)

    assert first.entity_id == second.entity_id
    assert second.accepted_view_count == 1
    assert second.accepted_frame_ids == frozenset({0})


def test_registry_uses_semantics_only_to_break_equal_geometry_ties() -> None:
    registry = EntityRegistry(EntityRegistryConfig(min_voxel_overlap=0.1, max_centroid_distance_m=0.2))
    chair = registry.resolve(_track(0, {(0, 0, 20), (1, 0, 20)}, label="chair", semantic_id=2), 1)
    stool = registry.resolve(_track(0, {(100, 0, 20)}, label="stool", semantic_id=3), 2)
    registry.entities[stool.entity_id] = replace(
        stool,
        voxel_keys=chair.voxel_keys,
        centroid_xyz=chair.centroid_xyz,
        bounds_min_xyz=chair.bounds_min_xyz,
        bounds_max_xyz=chair.bounds_max_xyz,
    )
    result = registry.resolve(_track(1, {(0, 0, 20)}, label="stool", semantic_id=3), 3)

    assert result.entity_id == stool.entity_id
    assert result.semantic_id == 3


def test_entity_jsonl_round_trip_has_no_dense_point_state(tmp_path: Path) -> None:
    registry = EntityRegistry()
    registry.resolve(_track(0, {(-1, 2, 3), (0, 2, 3)}), revision=1)
    path = tmp_path / "entities.jsonl"

    registry.save(path)
    restored = EntityRegistry.load(path)

    assert restored.entities == registry.entities
    text = path.read_text(encoding="utf-8")
    assert "points" not in text
    assert "point_cloud" not in text
