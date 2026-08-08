from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.oviv2.test_temporal_runtime import _config, _confirm, _frame, _runtime


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


def test_pre_normalized_entity_prototype_is_byte_stable() -> None:
    runtime = _runtime()
    _confirm(runtime)
    entity = runtime.state.entities[0]
    raw = np.array([-0.7452943516631282, 0.2677252628411601])
    prototype = raw / np.linalg.norm(raw)
    assert np.linalg.norm(prototype) != 1.0

    updated = replace(
        entity,
        image_prototype=prototype,
        feature_model_id="clip-byte-stable",
    )

    assert np.array_equal(updated.image_prototype, prototype)


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


def test_geometry_state_shares_frozen_epochs_without_deepcopy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.oviv2.temporal_epoch import GeometryEpoch
    from src.oviv2.temporal_state import TemporalGeometryState

    runtime = _runtime()
    entity_id = _confirm(runtime)
    epoch = runtime.state.geometry.current(entity_id)

    def forbidden(*args, **kwargs):
        raise AssertionError("GeometryEpoch must not be deep-copied")

    monkeypatch.setattr(GeometryEpoch, "__deepcopy__", forbidden, raising=False)
    geometry = TemporalGeometryState((epoch,), 3, 8)
    updated = geometry.transaction().finalize()

    assert geometry.epochs[0] is epoch
    assert updated.epochs[0] is epoch


def test_geometry_state_copy_and_deepcopy_return_self() -> None:
    import copy

    runtime = _runtime()
    _confirm(runtime)
    geometry = runtime.state.geometry

    assert copy.copy(geometry) is geometry
    assert copy.deepcopy(geometry) is geometry


def test_geometry_transaction_finalizes_once_and_shares_unchanged_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.oviv2.temporal_epoch import GeometryEpoch
    from src.oviv2.temporal_state import TemporalGeometryState

    runtime = _runtime()
    entity_id = _confirm(runtime)
    first = runtime.state.geometry.current(entity_id)
    second = GeometryEpoch(
        entity_id + 1, 0, first.object_to_world, first.submap, True,
        first.motion_decision, first.last_processed_frame_id,
    )
    geometry = TemporalGeometryState((first, second), 3, 8)
    calls = 0
    original = TemporalGeometryState.__post_init__

    def counted(self):
        nonlocal calls
        calls += 1
        original(self)

    monkeypatch.setattr(TemporalGeometryState, "__post_init__", counted)
    replacement = replace(
        first, last_processed_frame_id=first.last_processed_frame_id + 1
    )
    transaction = geometry.transaction()
    transaction.replace_current(replacement)
    transaction.append(
        GeometryEpoch(
            second.entity_id, 1, second.object_to_world, second.submap, True,
            second.motion_decision, second.last_processed_frame_id + 1,
        )
    )
    updated = transaction.finalize()

    assert calls == 1
    assert updated.current(first.entity_id) is replacement
    assert updated.epochs[1] is second


@pytest.mark.parametrize("operation_count", (64, 128))
def test_geometry_transaction_construction_count_is_constant(
    monkeypatch: pytest.MonkeyPatch, operation_count: int
) -> None:
    from src.oviv2.temporal_state import TemporalGeometryState

    runtime = _runtime()
    entity_id = _confirm(runtime)
    geometry = runtime.state.geometry
    calls = 0
    original = TemporalGeometryState.__post_init__

    def counted(self):
        nonlocal calls
        calls += 1
        original(self)

    monkeypatch.setattr(TemporalGeometryState, "__post_init__", counted)
    transaction = geometry.transaction()
    for _ in range(operation_count):
        transaction.replace_current(transaction.current(entity_id))
    transaction.finalize()

    assert calls == 1


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
    assert migrated.background_ledger is None
    assert migrated.background_mode == "masking_only"


def test_explicit_profile_locked_legacy_state_does_not_silently_migrate() -> None:
    from src.oviv2.temporal_state import TemporalRuntimeState

    runtime = _runtime()
    _confirm(runtime)
    old = runtime.state
    assert old.background.active_block_count > 0

    with pytest.raises(ValueError, match="without.*ledger|empty"):
        TemporalRuntimeState(
            old.scene_id,
            old.revision,
            old.last_frame_id,
            old.last_timestamp,
            old.next_entity_id,
            old.entities,
            old.background,
            old.tracker,
            background_mode="profile_locked",
        )


def test_state_rejects_background_that_disagrees_with_ledger() -> None:
    from src.oviv2.temporal_background import TemporalBackgroundVolume
    from src.oviv2.temporal_runtime import _SparseBackgroundVolume

    runtime = _runtime()
    _confirm(runtime)
    state = runtime.state
    committed = state.background_ledger.combined_volume

    wrong_touch = committed.clone()
    wrong_touch._last_blocks_touched += 1
    with pytest.raises(ValueError, match="background.*ledger"):
        _rebuild_state(state, background=wrong_touch)

    wrong_config = TemporalBackgroundVolume(
        replace(committed.config, depth_max_m=committed.config.depth_max_m / 2.0)
    )
    with pytest.raises(ValueError, match="background.*ledger"):
        _rebuild_state(state, background=wrong_config)

    frame = _frame(2, depth=2.0)
    depth = np.zeros_like(frame.depth)
    depth[2, 2] = frame.depth[2, 2]
    wrong_blocks = _SparseBackgroundVolume(committed.config)
    keys = wrong_blocks.candidate_block_keys(frame, depth)
    wrong_blocks = wrong_blocks.trial_integrate_blocks(frame, depth, keys)
    with pytest.raises(ValueError, match="background.*ledger"):
        _rebuild_state(state, background=wrong_blocks)


def test_state_without_ledger_requires_empty_background() -> None:
    runtime = _runtime()
    state = runtime.state
    background = state.background
    background._last_blocks_touched = 1

    with pytest.raises(ValueError, match="without.*ledger|empty"):
        _rebuild_state(state, background=background, background_ledger=None)


def test_masking_only_state_explicitly_allows_ledgerless_background() -> None:
    runtime = _runtime()
    state = runtime.state
    background = state.background
    background._last_blocks_touched = 1

    diagnostic = _rebuild_state(
        state,
        background=background,
        background_ledger=None,
        background_mode="masking_only",
    )

    assert diagnostic.background_mode == "masking_only"
    assert diagnostic.background_ledger is None
    assert diagnostic.background.last_blocks_touched == 1
    assert diagnostic.canonical_dump()[-1] == ("background_mode", "masking_only")


@pytest.mark.parametrize("mode", ["unknown", "reversible_ledger", None, 1])
def test_state_rejects_invalid_background_mode(mode: object) -> None:
    state = _runtime().state

    with pytest.raises((TypeError, ValueError), match="background_mode"):
        _rebuild_state(state, background_mode=mode)


def _rebuild_state(state, **changes):
    from src.oviv2.temporal_state import TemporalRuntimeState

    values = dict(
        scene_id=state.scene_id,
        revision=state.revision,
        last_frame_id=state.last_frame_id,
        last_timestamp=state.last_timestamp,
        next_entity_id=state.next_entity_id,
        entities=state.entities,
        background=state.background,
        tracker=state.tracker,
        identities=state.identities,
        geometry=state.geometry,
        lifecycle_beliefs=state.lifecycle_beliefs,
        background_ledger=state.background_ledger,
        export_tracker=state.export_tracker,
        diagnostics=state.diagnostics,
        background_mode=state.background_mode,
    )
    values.update(changes)
    return TemporalRuntimeState(**values)


def test_state_rejects_future_identity_without_changing_source_state() -> None:
    runtime = _runtime()
    entity_id = _confirm(runtime)
    state = runtime.state
    before = state.canonical_dump()
    identities = state.identities
    record = identities.get(entity_id)
    identities._records[entity_id] = replace(record, last_frame_id=99, last_timestamp=99.0)

    with pytest.raises(ValueError, match="identity.*runtime|future"):
        _rebuild_state(state, identities=identities)
    assert state.canonical_dump() == before


def test_state_rejects_bank_next_identity_mismatch() -> None:
    runtime = _runtime()
    _confirm(runtime)
    state = runtime.state
    identities = state.identities
    identities._next_identity_id += 1

    with pytest.raises(ValueError, match="next.*identity"):
        _rebuild_state(state, identities=identities)


def test_state_rejects_wrapper_identity_semantic_mismatch() -> None:
    runtime = _runtime()
    entity_id = _confirm(runtime)
    state = runtime.state
    identities = state.identities
    record = identities.get(entity_id)
    identities._records[entity_id] = replace(record, semantic_probabilities=((2, 1.0),))

    with pytest.raises(ValueError, match="semantic"):
        _rebuild_state(state, identities=identities)


def test_state_rejects_wrapper_identity_extent_history_mismatch() -> None:
    runtime = _runtime()
    entity_id = _confirm(runtime)
    state = runtime.state
    identities = state.identities
    record = identities.get(entity_id)
    identities._records[entity_id] = replace(
        record, extent_xyz=(9.0, 9.0, 9.0)
    )

    with pytest.raises(ValueError, match="extent|history"):
        _rebuild_state(state, identities=identities)


def test_state_rejects_current_epoch_wrapper_mismatch() -> None:
    from src.oviv2.temporal_epoch import GeometryEpoch

    runtime = _runtime()
    entity_id = _confirm(runtime)
    state = runtime.state
    geometry = state.geometry
    epoch = geometry.current(entity_id)
    pose = np.array(epoch.object_to_world, copy=True)
    pose[0, 3] += 1.0
    geometry = geometry.replace_current(
        GeometryEpoch(
            epoch.entity_id, epoch.epoch_id, pose, epoch.submap,
            epoch.readout_valid, epoch.motion_decision, epoch.last_processed_frame_id,
        )
    )

    with pytest.raises(ValueError, match="epoch|geometry|pose"):
        _rebuild_state(state, geometry=geometry)
