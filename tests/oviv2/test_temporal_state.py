from __future__ import annotations

from tests.oviv2.test_temporal_runtime import _config, _confirm, _runtime


def test_runtime_state_owns_decoupled_temporal_components() -> None:
    runtime = _runtime()
    entity_id = _confirm(runtime)
    state = runtime.state

    assert state.identities.get(entity_id) is not None
    assert state.geometry.current(entity_id).entity_id == entity_id
    assert state.lifecycle_beliefs[0].entity_id == entity_id
    assert state.background_ledger is not None
    assert state.export_tracker.get(entity_id) is not None
    assert state.diagnostics.processed_frame_count == 2


def test_decoupled_component_access_is_defensive_and_canonical() -> None:
    runtime = _runtime()
    entity_id = _confirm(runtime)
    before = runtime.state.canonical_dump()

    identities = runtime.state.identities
    identities.update(
        entity_id,
        frame_id=2,
        timestamp=2.0,
        semantic_probabilities=((1, 1.0),),
        appearance_prototype=None,
        feature_model_id=None,
        lifecycle=identities.get(entity_id).lifecycle,
        centroid_xyz=(9.0, 9.0, 9.0),
        extent_xyz=(1.0, 1.0, 1.0),
        motion_velocity_xyz=(0.0, 0.0, 0.0),
        motion_uncertainty_m=0.0,
    )

    assert runtime.state.canonical_dump() == before
    assert runtime.state.identities.get(entity_id).last_centroid_xyz != (9.0, 9.0, 9.0)


def test_geometry_capacity_never_evicts_identity_memory() -> None:
    config = _config()
    runtime = _runtime(config)
    entity_id = _confirm(runtime)

    assert runtime.state.identities.get(entity_id) is not None
    assert runtime.state.geometry.current(entity_id) is not None


def test_geometry_epoch_eviction_is_canonical_and_keeps_identity() -> None:
    from src.oviv2.temporal_epoch import GeometryEpoch
    from src.oviv2.temporal_state import TemporalGeometryState

    runtime = _runtime()
    entity_id = _confirm(runtime)
    epoch0 = runtime.state.geometry.current(entity_id)
    geometry = TemporalGeometryState((epoch0,), 2, 2)
    epoch1 = GeometryEpoch(
        entity_id, 1, epoch0.object_to_world, epoch0.submap, True,
        epoch0.motion_decision, 2,
    )
    epoch2 = GeometryEpoch(
        entity_id, 2, epoch0.object_to_world, epoch0.submap, True,
        epoch0.motion_decision, 3,
    )

    geometry = geometry.append(epoch1).append(epoch2)

    assert tuple(item.epoch_id for item in geometry.epochs) == (1, 2)
    assert runtime.state.identities.get(entity_id) is not None
    assert geometry.canonical_dump() == geometry.canonical_dump()


def test_legacy_runtime_state_constructor_migrates_components() -> None:
    from src.oviv2.temporal_state import TemporalRuntimeState

    runtime = _runtime()
    entity_id = _confirm(runtime)
    old = runtime.state
    migrated = TemporalRuntimeState(
        old.scene_id, old.revision, old.last_frame_id, old.last_timestamp,
        old.next_entity_id, old.entities, old.background, old.tracker,
    )

    assert migrated.identities.get(entity_id) is not None
    assert migrated.geometry.current(entity_id).epoch_id == 0
    assert migrated.lifecycle_beliefs[0].entity_id == entity_id
