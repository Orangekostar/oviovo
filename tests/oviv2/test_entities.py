from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.oviv2.association import AssociationConfig
from src.oviv2.entities import EntityRegistry, EntityRegistryConfig
from src.oviv2.semantic_memory import InformativeView
from src.oviv2.tracking import LocalTracker, LocalTrackerConfig

from tests.oviv2.test_tracking import observation


def _track(
    frame_id: int,
    keys: set[tuple[int, int, int]],
    *,
    label: str = "chair",
    semantic_id: int = 2,
    observation_id: int | None = None,
    confidence: float = 0.9,
):
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
    return tracker.update(
        (
            observation(
                frame_id,
                keys,
                label=label,
                semantic_id=semantic_id,
                observation_id=observation_id,
                confidence=confidence,
                image_feature=np.asarray((1.0, 0.0)),
                feature_model_id="clip:test",
                view_direction_xyz=(1.0, 0.0, 0.0),
                visible_pixel_count=4096,
            ),
        ),
        frame_id,
    ).accepted[0]


def test_registry_semantic_posterior_corrects_an_early_label() -> None:
    registry = EntityRegistry()
    keys = {(0, 0, 20), (1, 0, 20)}

    entity = registry.resolve(_track(0, keys), revision=1)
    for frame_id in range(1, 4):
        entity = registry.resolve(
            _track(frame_id, keys, label="stool", semantic_id=3),
            revision=frame_id + 1,
        )

    assert entity.entity_id == 1
    assert entity.semantic_id == 3
    assert entity.label == "stool"
    assert entity.semantic_margin > 0.0


def test_resolve_batch_assigns_at_most_one_track_to_an_existing_entity() -> None:
    registry = EntityRegistry()
    keys = {(0, 0, 20), (1, 0, 20)}
    registry.resolve(_track(0, keys), revision=1)
    first = replace(_track(1, keys, observation_id=1), track_id=10)
    second = replace(_track(1, keys, observation_id=2), track_id=11)

    resolved = registry.resolve_batch((second, first), revision=2)

    assert [entity.entity_id for entity in resolved] == [1, 2]


def test_loads_v1_entity_as_seeded_posterior_without_inventing_memory(
    tmp_path: Path,
) -> None:
    payload = {
        "entity_id": 4,
        "semantic_id": 2,
        "label": "chair",
        "semantic_support": 0.9,
        "accepted_view_count": 3,
        "accepted_frame_ids": [0, 1, 2],
        "first_frame_id": 0,
        "last_frame_id": 2,
        "last_revision": 7,
        "lifecycle_state": "active",
        "voxel_keys": [[0, 0, 20]],
        "centroid_xyz": [0.0, 0.0, 1.0],
        "bounds_min_xyz": [0.0, 0.0, 1.0],
        "bounds_max_xyz": [0.0, 0.0, 1.0],
    }
    path = tmp_path / "v1.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    entity = EntityRegistry.load(path).entities[4]

    assert entity.semantic_posterior.best_semantic_id == entity.semantic_id == 2
    assert entity.semantic_posterior.effective_support == pytest.approx(0.9)
    assert entity.feature_bank.prototypes == ()
    assert entity.view_bank.views == ()
    assert entity.accepted_observation_ids == frozenset()


def test_v1_geometry_uses_historical_frame_count_until_observation_audit_catches_up(
    tmp_path: Path,
) -> None:
    payload = {
        "entity_id": 4,
        "semantic_id": 2,
        "label": "chair",
        "semantic_support": 0.9,
        "accepted_view_count": 3,
        "accepted_frame_ids": [0, 1, 2],
        "first_frame_id": 0,
        "last_frame_id": 2,
        "last_revision": 7,
        "lifecycle_state": "active",
        "voxel_keys": [[0, 0, 20]],
        "centroid_xyz": [0.0, 0.0, 1.0],
        "bounds_min_xyz": [0.0, 0.0, 1.0],
        "bounds_max_xyz": [0.0, 0.0, 1.0],
    }
    path = tmp_path / "v1.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    registry = EntityRegistry.load(path)

    def incoming(frame_id: int, centroid_x: float, observation_id: int):
        previous = registry.entities[4]
        track = _track(
            frame_id,
            {(0, 0, 20)},
            observation_id=observation_id,
        )
        item = replace(
            track.observations[0],
            centroid_xyz=(centroid_x, 0.0, 1.0),
            bounds_min_xyz=(centroid_x, 0.0, 1.0),
            bounds_max_xyz=(centroid_x, 0.0, 1.0),
        )
        return replace(
            track,
            observations=(item,),
            voxel_keys=previous.voxel_keys,
            centroid_xyz=previous.centroid_xyz,
            bounds_min_xyz=previous.bounds_min_xyz,
            bounds_max_xyz=previous.bounds_max_xyz,
        )

    first = registry.resolve(incoming(3, 4.0, 7), revision=8)
    second = registry.resolve(incoming(4, 6.0, 8), revision=9)

    assert first.centroid_xyz == pytest.approx((1.0, 0.0, 1.0))
    assert second.centroid_xyz == pytest.approx((2.0, 0.0, 1.0))
    assert second.accepted_view_count == 5


def test_same_semantic_winner_updates_to_new_canonical_label() -> None:
    registry = EntityRegistry()
    keys = {(0, 0, 20)}
    registry.resolve(_track(0, keys, label="old-chair"), revision=1)

    entity = registry.resolve(_track(1, keys, label="chair"), revision=2)

    assert entity.semantic_id == 2
    assert entity.label == "chair"


def test_persistent_entity_requires_canonical_label_for_semantic_winner() -> None:
    positive = EntityRegistry().resolve(_track(0, {(0, 0, 20)}), revision=1)
    unknown = EntityRegistry().resolve(
        _track(0, {(0, 0, 20)}, label="unknown", semantic_id=0),
        revision=1,
    )

    with pytest.raises(ValueError, match="label"):
        replace(positive, label=" ")
    with pytest.raises(ValueError, match="label"):
        replace(unknown, label="unknown")


def test_v2_load_rejects_empty_label_for_positive_winner(tmp_path: Path) -> None:
    registry = EntityRegistry()
    registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
    path = tmp_path / "entities.jsonl"
    registry.save(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[1]["label"] = ""
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="label"):
        EntityRegistry.load(path)


@pytest.mark.parametrize("empty_batch", [False, True])
def test_resolve_batch_rejects_revision_rollback_atomically(empty_batch: bool) -> None:
    registry = EntityRegistry()
    registry.resolve(_track(0, {(0, 0, 20)}), revision=10)
    before_entities = dict(registry.entities)
    before_next_id = registry._next_entity_id
    tracks = () if empty_batch else (_track(1, {(0, 0, 20)}),)

    with pytest.raises(ValueError, match="revision"):
        registry.resolve_batch(tracks, revision=1)

    assert registry.entities == before_entities
    assert registry._next_entity_id == before_next_id


@pytest.mark.parametrize("across_tracks", [False, True])
def test_resolve_batch_rejects_duplicate_observation_ids_atomically(
    across_tracks: bool,
) -> None:
    first = replace(
        _track(0, {(0, 0, 20)}, observation_id=1),
        track_id=10,
    )
    if across_tracks:
        second = replace(
            _track(0, {(100, 0, 20)}, observation_id=1),
            track_id=11,
        )
        tracks = (first, second)
    else:
        duplicate = replace(
            first.observations[0],
            frame_id=1,
            timestamp=1.0,
        )
        tracks = (
            replace(
                first,
                observations=first.observations + (duplicate,),
                last_frame_id=1,
            ),
        )
    registry = EntityRegistry()

    with pytest.raises(ValueError, match="observation IDs"):
        registry.resolve_batch(tracks, revision=1)

    assert registry.entities == {}
    assert registry._next_entity_id == 1


def test_load_streams_jsonl_without_path_read_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = EntityRegistry()
    registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
    path = tmp_path / "entities.jsonl"
    registry.save(path)

    def fail_read_text(*args: object, **kwargs: object) -> str:
        raise AssertionError("load must stream JSONL")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    assert EntityRegistry.load(path).entities == registry.entities


def test_v2_jsonl_round_trip_preserves_exact_registry_and_nested_memory(
    tmp_path: Path,
) -> None:
    config = EntityRegistryConfig(
        association=AssociationConfig(minimum_score=0.55),
        prototype_top_k=2,
        prototype_merge_cosine=0.8,
        view_top_k=4,
        view_minimum_novelty_cosine=0.2,
    )
    registry = EntityRegistry(config)
    registry.resolve(_track(0, {(-1, 2, 3), (0, 2, 3)}), revision=7)
    path = tmp_path / "entities.jsonl"

    registry.save(path)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0] == {
        "record_type": "registry",
        "schema_version": 2,
        "config": asdict(config),
        "next_entity_id": 2,
    }
    assert records[1]["record_type"] == "entity"
    assert set(records[1]) >= {
        "semantic_posterior",
        "feature_bank",
        "view_bank",
        "accepted_observation_ids",
    }
    assert EntityRegistry.load(path).config == config
    assert EntityRegistry.load(path).entities == registry.entities
    assert "points" not in path.read_text(encoding="utf-8")
    assert "point_cloud" not in path.read_text(encoding="utf-8")


def test_replayed_bounded_track_consumes_only_unseen_observation_ids() -> None:
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
    first = tracker.update(
        (
            observation(
                0,
                {(0, 0, 20)},
                observation_id=1,
                image_feature=np.asarray((1.0, 0.0)),
                feature_model_id="clip:test",
                view_direction_xyz=(1.0, 0.0, 0.0),
                visible_pixel_count=4096,
            ),
        ),
        0,
    ).accepted[0]
    bounded = tracker.update(
        (
            observation(
                1,
                {(1, 0, 20)},
                observation_id=1,
                image_feature=np.asarray((0.0, 1.0)),
                feature_model_id="clip:test",
                view_direction_xyz=(0.0, 1.0, 0.0),
                visible_pixel_count=4096,
            ),
        ),
        1,
    ).accepted[0]
    registry = EntityRegistry()
    registry.resolve(first, revision=1)
    once = registry.resolve(bounded, revision=2)

    replayed = registry.resolve(bounded, revision=3)

    assert replayed.accepted_observation_ids == frozenset({1, 101})
    assert replayed.semantic_posterior == once.semantic_posterior
    assert replayed.feature_bank == once.feature_bank
    assert replayed.view_bank == once.view_bank
    assert replayed.voxel_keys == once.voxel_keys
    assert replayed.centroid_xyz == pytest.approx(once.centroid_xyz)
    assert replayed.bounds_min_xyz == once.bounds_min_xyz
    assert replayed.bounds_max_xyz == once.bounds_max_xyz


def test_semantic_labels_returns_positive_active_and_dormant_winners() -> None:
    registry = EntityRegistry()
    active = registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
    dormant = registry.resolve(
        _track(1, {(100, 0, 20)}, label="stool", semantic_id=3),
        revision=2,
    )
    registry.resolve(
        _track(2, {(200, 0, 20)}, label="unknown", semantic_id=0),
        revision=3,
    )
    registry.entities[dormant.entity_id] = replace(dormant, lifecycle_state="dormant")

    assert registry.semantic_labels() == {
        active.entity_id: (2, pytest.approx(1.0)),
        dormant.entity_id: (3, pytest.approx(1.0)),
    }


def test_observation_quality_updates_posterior_prototype_and_view_exactly() -> None:
    keys = {(index, 0, 20) for index in range(128)}
    track = _track(0, keys, confidence=0.8)
    track = replace(
        track,
        observations=(
            replace(
                track.observations[0],
                border_contact_fraction=0.25,
                visible_pixel_count=2048,
            ),
        ),
    )

    entity = EntityRegistry().resolve(track, revision=1)

    expected_quality = 0.8 * 0.75 * 0.5 * 0.5
    assert entity.semantic_posterior.effective_support == pytest.approx(
        0.8 * expected_quality
    )
    assert entity.feature_bank.prototypes[0].support == pytest.approx(expected_quality)
    assert entity.view_bank.views[0].quality == pytest.approx(expected_quality)
    assert entity.view_bank.views[0].visible_pixel_count == 2048


def test_registry_config_controls_visual_and_view_bank_capacity() -> None:
    config = EntityRegistryConfig(
        prototype_top_k=1,
        prototype_merge_cosine=0.9,
        view_top_k=1,
        view_minimum_novelty_cosine=0.1,
    )
    registry = EntityRegistry(config)
    first = _track(0, {(0, 0, 20)}, confidence=0.4)
    registry.resolve(first, revision=1)
    second_observation = observation(
        1,
        {(0, 0, 20)},
        observation_id=2,
        confidence=0.9,
        image_feature=np.asarray((0.0, 1.0)),
        feature_model_id="clip:test",
        view_direction_xyz=(0.0, 1.0, 0.0),
        visible_pixel_count=4096,
    )
    bounded = replace(
        first,
        observations=first.observations + (second_observation,),
        last_frame_id=1,
    )

    entity = registry.resolve(bounded, revision=2)

    assert entity.feature_bank.max_prototypes == 1
    assert len(entity.feature_bank.prototypes) == 1
    assert entity.feature_bank.prototypes[0].vector == pytest.approx((0.0, 1.0))
    assert entity.view_bank.max_views == 1
    assert [view.observation_id for view in entity.view_bank.views] == [102]


@pytest.mark.parametrize(
    "changes",
    [
        {"min_voxel_overlap": True},
        {"min_voxel_overlap": -0.1},
        {"max_centroid_distance_m": np.nan},
        {"max_centroid_distance_m": 0.0},
        {"prototype_top_k": True},
        {"prototype_top_k": 0},
        {"prototype_merge_cosine": 1.1},
        {"view_top_k": False},
        {"view_top_k": 0},
        {"view_minimum_novelty_cosine": -0.1},
        {"view_minimum_novelty_cosine": 2.1},
    ],
)
def test_registry_config_rejects_invalid_values(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        EntityRegistryConfig(**changes)  # type: ignore[arg-type]


def test_registry_config_rejects_explicit_association_with_legacy_overrides() -> None:
    with pytest.raises(ValueError, match="explicit association"):
        EntityRegistryConfig(
            min_voxel_overlap=0.2,
            association=AssociationConfig(),
        )


def test_resolve_batch_is_independent_of_input_order() -> None:
    tracks = (
        replace(_track(0, {(100, 0, 20)}, observation_id=2), track_id=10),
        replace(_track(0, {(0, 0, 20)}, observation_id=1), track_id=5),
    )

    def resolve(values: tuple) -> tuple[tuple[int, int], ...]:
        registry = EntityRegistry()
        result = registry.resolve_batch(values, revision=1)
        return tuple(
            (track.track_id, entity.entity_id)
            for track, entity in zip(sorted(values, key=lambda item: item.track_id), result)
        )

    assert resolve(tracks) == resolve(tuple(reversed(tracks))) == ((5, 1), (10, 2))


def test_resolve_batch_validation_is_atomic() -> None:
    valid = replace(_track(1, {(0, 0, 20)}, observation_id=1), track_id=10)
    invalid_calls = (
        lambda registry: registry.resolve_batch([valid], revision=2),  # type: ignore[arg-type]
        lambda registry: registry.resolve_batch((replace(valid, confirmed=False),), revision=2),
        lambda registry: registry.resolve_batch((valid, valid), revision=2),
        lambda registry: registry.resolve_batch((valid,), revision=True),  # type: ignore[arg-type]
        lambda registry: registry.resolve_batch((valid,), revision=1.5),  # type: ignore[arg-type]
        lambda registry: registry.resolve_batch(
            (valid, replace(valid, track_id=11, observations=())),
            revision=2,
        ),
    )
    for invalid_call in invalid_calls:
        registry = EntityRegistry()
        registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
        before_entities = dict(registry.entities)
        before_next_id = registry._next_entity_id

        with pytest.raises((TypeError, ValueError)):
            invalid_call(registry)

        assert registry.entities == before_entities
        assert registry._next_entity_id == before_next_id


def test_persistent_entity_rejects_views_outside_accepted_ids() -> None:
    entity = EntityRegistry().resolve(_track(0, {(0, 0, 20)}), revision=1)
    foreign_view = InformativeView(999, 999, 10, 0.5, (0.0, 1.0, 0.0))

    with pytest.raises(ValueError, match="accepted"):
        replace(
            entity,
            view_bank=replace(entity.view_bank, views=(foreign_view,)),
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "schema_type",
        "duplicate_metadata",
        "duplicate_entity",
        "unknown_record",
        "next_entity_id",
        "config",
        "posterior",
        "mixed_v1_v2",
    ],
)
def test_v2_load_rejects_corrupt_or_duplicate_records(
    tmp_path: Path,
    corruption: str,
) -> None:
    registry = EntityRegistry()
    registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
    valid = tmp_path / "valid.jsonl"
    registry.save(valid)
    records = [json.loads(line) for line in valid.read_text(encoding="utf-8").splitlines()]

    if corruption == "schema_type":
        records[0]["schema_version"] = 2.0
    elif corruption == "duplicate_metadata":
        records.insert(1, dict(records[0]))
    elif corruption == "duplicate_entity":
        records.append(dict(records[1]))
    elif corruption == "unknown_record":
        records[1] = {"record_type": "future"}
    elif corruption == "next_entity_id":
        records[0]["next_entity_id"] = 1
    elif corruption == "config":
        del records[0]["config"]["view_top_k"]
    elif corruption == "posterior":
        records[1]["semantic_posterior"]["effective_support"] = -1.0
    elif corruption == "mixed_v1_v2":
        del records[1]["record_type"]
    path = tmp_path / f"{corruption}.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        EntityRegistry.load(path)


def test_v2_load_rejects_an_explicit_config_mismatch(tmp_path: Path) -> None:
    registry = EntityRegistry(EntityRegistryConfig(prototype_top_k=2))
    path = tmp_path / "entities.jsonl"
    registry.save(path)

    with pytest.raises(ValueError, match="config"):
        EntityRegistry.load(path, EntityRegistryConfig(prototype_top_k=3))


def test_v2_round_trip_reconstructs_canonical_legacy_association(tmp_path: Path) -> None:
    config = EntityRegistryConfig(
        min_voxel_overlap=0.25,
        max_centroid_distance_m=0.4,
    )
    registry = EntityRegistry(config)
    path = tmp_path / "entities.jsonl"
    registry.save(path)

    restored = EntityRegistry.load(path)

    assert restored.config == config


def test_save_is_atomic_when_replace_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = EntityRegistry()
    registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
    path = tmp_path / "entities.jsonl"
    path.write_text("previous\n", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("src.oviv2.entities.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        registry.save(path)

    assert path.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.glob(".entities.*.jsonl")) == []


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
