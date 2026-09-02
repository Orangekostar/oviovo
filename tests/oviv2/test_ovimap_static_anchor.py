from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import src.oviv2.ovimap_static_anchor as anchor_module
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.temporal_background import TemporalBackgroundVolume
from src.oviv2.temporal_config import TemporalGeometryConfig
from src.oviv2.temporal_export import (
    DynamicState,
    TemporalExportBatch,
    TemporalExportSample,
    TemporalLifecycleEvent,
)
from src.oviv2.temporal_geometry import ObjectSubmap
from src.oviv2.temporal_lifecycle import (
    TemporalEvidenceKind,
    TemporalLifecycle,
    TemporalLifecycleState,
)
from src.oviv2.temporal_snapshot import (
    TemporalCurrentSnapshot,
    TemporalSnapshotEntity,
    TemporalSnapshotMetadata,
)
from src.oviv2.temporal_state import TemporalEntityState
from src.oviv2.ovimap_static_anchor import (
    PrefixIdentitySample,
    StaticAnchorConfig,
    bind_anchor_identities,
    compose_anchor_checkpoint,
)


def _entity(
    entity_id: str,
    center_x: float,
    label: str,
    feature: tuple[float, float],
) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(
            (
                (center_x - 0.1, -0.1, -0.1),
                (center_x + 0.1, 0.1, 0.1),
                (center_x, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
        semantic_embedding=np.asarray(feature, dtype=np.float32),
        semantic_label=label,
        semantic_score=0.9,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=262.0,
        metadata={"authority": "ovimap_anchor"},
    )


def _anchor() -> MapSnapshot:
    return MapSnapshot(
        method="OVI-MAP causal static anchor",
        scene_id="apartment",
        timestamp=262.0,
        entities=[
            _entity("ovimap:1", 0.0, "Chair", (1.0, 0.0)),
            _entity("ovimap:2", 2.0, "Table", (0.0, 1.0)),
        ],
        background_xyz=np.asarray(((4.0, 0.0, 0.0),), dtype=np.float32),
        scope="current",
    )


def _sample(
    temporal_id: int,
    center_x: float,
    label: str,
    feature: tuple[float, float],
    *,
    frame_index: int = 262,
) -> PrefixIdentitySample:
    return PrefixIdentitySample(
        temporal_entity_id=temporal_id,
        frame_index=frame_index,
        centroid_xyz=(center_x, 0.0, 0.0),
        points_xyz=np.asarray(
            (
                (center_x - 0.1, -0.1, -0.1),
                (center_x + 0.1, 0.1, 0.1),
                (center_x, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
        semantic_label=label,
        semantic_embedding=np.asarray(feature, dtype=np.float32),
        geometry_epoch=0,
    )


def _config() -> StaticAnchorConfig:
    return StaticAnchorConfig(
        minimum_spatial_iou=0.01,
        maximum_centroid_distance_m=0.75,
        minimum_semantic_cosine=0.65,
        moved_displacement_m=0.2,
        background_voxel_size_m=0.05,
    )


def test_bind_uses_pre_intervention_geometry_and_is_one_to_one() -> None:
    state = bind_anchor_identities(
        _anchor(),
        (
            _sample(8, 2.0, "Table", (0.0, 1.0)),
            _sample(7, 0.0, "Chair", (1.0, 0.0)),
        ),
        _config(),
        cutoff_frame=262,
    )

    assert state.cutoff_frame == 262
    assert state.last_frame_index == 262
    assert state.bindings == (("ovimap:1", 7), ("ovimap:2", 8))
    assert state.initial_geometry_epochs == ((7, 0), (8, 0))
    assert state.removed_anchor_ids == frozenset()


def test_binding_is_invariant_to_anchor_and_sample_order() -> None:
    anchor = _anchor()
    reversed_anchor = replace(anchor, entities=list(reversed(anchor.entities)))
    samples = (
        _sample(7, 0.0, "Chair", (1.0, 0.0)),
        _sample(8, 2.0, "Table", (0.0, 1.0)),
    )

    expected = bind_anchor_identities(
        anchor, samples, _config(), cutoff_frame=262
    )
    observed = bind_anchor_identities(
        reversed_anchor,
        tuple(reversed(samples)),
        _config(),
        cutoff_frame=262,
    )

    assert observed.bindings == expected.bindings
    assert observed.initial_geometry_epochs == expected.initial_geometry_epochs


def test_binding_voxelizes_each_anchor_and_prefix_sample_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = anchor_module._voxel_keys

    def counted(points: np.ndarray, voxel_size_m: float):
        nonlocal calls
        calls += 1
        return original(points, voxel_size_m)

    monkeypatch.setattr(anchor_module, "_voxel_keys", counted)

    bind_anchor_identities(
        _anchor(),
        (
            _sample(7, 0.0, "Chair", (1.0, 0.0)),
            _sample(8, 2.0, "Table", (0.0, 1.0)),
        ),
        _config(),
        cutoff_frame=262,
    )

    assert calls == 4


def test_binding_rejects_future_sample() -> None:
    with pytest.raises(ValueError, match="after causal cutoff"):
        bind_anchor_identities(
            _anchor(),
            (_sample(7, 0.0, "Chair", (1.0, 0.0), frame_index=263),),
            _config(),
            cutoff_frame=262,
        )


def test_binding_rejects_duplicate_temporal_ids() -> None:
    with pytest.raises(ValueError, match="temporal entity IDs must be unique"):
        bind_anchor_identities(
            _anchor(),
            (
                _sample(7, 0.0, "Chair", (1.0, 0.0)),
                _sample(7, 2.0, "Table", (0.0, 1.0)),
            ),
            _config(),
            cutoff_frame=262,
        )


def test_binding_rejects_semantic_conflict_despite_spatial_overlap() -> None:
    state = bind_anchor_identities(
        _anchor(),
        (_sample(7, 0.0, "Table", (0.0, 1.0)),),
        _config(),
        cutoff_frame=262,
    )

    assert state.bindings == ()


def test_binding_deterministically_assigns_single_candidate_once() -> None:
    anchor = _anchor()
    anchor.entities[1].points_xyz = anchor.entities[0].points_xyz
    anchor.entities[1].semantic_label = "Chair"
    anchor.entities[1].semantic_embedding = anchor.entities[0].semantic_embedding

    state = bind_anchor_identities(
        anchor,
        (_sample(7, 0.0, "Chair", (1.0, 0.0)),),
        _config(),
        cutoff_frame=262,
    )

    assert state.bindings == (("ovimap:1", 7),)


def _geometry_config() -> TemporalGeometryConfig:
    return TemporalGeometryConfig(
        voxel_size_m=0.1,
        depth_max_m=4.0,
        maximum_entities=8,
        maximum_object_voxels=16,
        maximum_visibility_points_per_entity=16,
        background_block_count=16,
        background_mask_dilation_px=0,
        minimum_icp_points=3,
        minimum_icp_fitness=0.5,
        maximum_icp_rmse_m=0.1,
        maximum_motion_m=2.0,
    )


def _temporal_entity(
    entity_id: int,
    lifecycle: TemporalLifecycle,
    *,
    x: float,
    frame_index: int,
    geometry_epoch: int,
    first_seen_frame_id: int = 0,
) -> TemporalSnapshotEntity:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    submap = ObjectSubmap(
        reference_centroid_xyz=(0.0, 0.0, 0.0),
        local_voxel_keys=((0, 0, 0), (1, 0, 0)),
        local_points_xyz=np.asarray(
            ((-0.05, 0.0, 0.0), (0.05, 0.0, 0.0)), dtype=np.float32
        ),
        weights=np.ones(2, dtype=np.float32),
        last_seen_frame_ids=np.asarray(
            (frame_index, frame_index), dtype=np.int64
        ),
    )
    absent_streak = 2 if lifecycle is TemporalLifecycle.DORMANT else 0
    entity = TemporalEntityState(
        lifecycle=TemporalLifecycleState(
            entity_id=entity_id,
            lifecycle=lifecycle,
            existence_log_odds=-1.0 if lifecycle is TemporalLifecycle.DORMANT else 1.0,
            last_frame_id=frame_index,
            last_timestamp=float(frame_index),
            absent_streak=absent_streak,
            absence_view_bins=(0, 1) if absent_streak else (),
        ),
        semantic_probabilities=((1, 1.0),),
        image_prototype=np.asarray((1.0, 0.0), dtype=np.float32),
        feature_model_id="fixture",
        extent_xyz=(0.2, 0.2, 0.2),
        object_to_world=pose,
        submap=submap,
        first_seen_frame_id=first_seen_frame_id,
        last_seen_frame_id=frame_index,
    )
    return TemporalSnapshotEntity(entity, geometry_epoch, True)


def _temporal_snapshot(
    *,
    frame_index: int,
    entities: tuple[TemporalSnapshotEntity, ...],
) -> TemporalCurrentSnapshot:
    return TemporalCurrentSnapshot(
        metadata=TemporalSnapshotMetadata(
            "apartment",
            frame_index,
            float(frame_index),
            frame_index + 1,
            0.1,
            "a" * 64,
        ),
        entities=tuple(sorted(entities, key=lambda item: item.lifecycle.entity_id)),
        background=TemporalBackgroundVolume(_geometry_config()),
    )


def _export_batch(
    *,
    frame_index: int,
    entity_id: int,
    x: float,
    dynamic_state: DynamicState,
    geometry_epoch: int,
    evidence: TemporalEvidenceKind | None = None,
    before: TemporalLifecycle = TemporalLifecycle.ACTIVE,
    after: TemporalLifecycle = TemporalLifecycle.ACTIVE,
) -> TemporalExportBatch:
    timestamp_ns = frame_index * 1_000_000_000
    sample = TemporalExportSample(
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        entity_id=entity_id,
        centroid_xyz=(x, 0.0, 0.0),
        observation_count=max(1, frame_index),
        dynamic_state=dynamic_state,
        motion_confidence=0.9 if dynamic_state is DynamicState.DYNAMIC else 0.0,
        geometry_epoch=geometry_epoch,
        readout_valid=True,
    )
    events = ()
    if evidence is not None:
        events = (
            TemporalLifecycleEvent(
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                entity_id=entity_id,
                before=before,
                after=after,
                evidence=evidence,
                geometry_epoch=geometry_epoch,
                readout_valid=True,
            ),
        )
    return TemporalExportBatch(frame_index, timestamp_ns, (sample,), events)


def _single_anchor_and_state():
    anchor = _anchor()
    anchor.entities = anchor.entities[:1]
    state = bind_anchor_identities(
        anchor,
        (_sample(7, 0.0, "Chair", (1.0, 0.0)),),
        _config(),
        cutoff_frame=262,
    )
    return anchor, state


def _compose(
    *,
    anchor: MapSnapshot,
    state,
    temporal: TemporalCurrentSnapshot,
    exports: tuple[TemporalExportBatch, ...],
):
    return compose_anchor_checkpoint(
        anchor=anchor,
        temporal=temporal,
        exports=exports,
        state=state,
        config=_config(),
        anchor_manifest_sha256="b" * 64,
        class_names=("unknown", "Chair", "Table"),
    )


def test_static_active_identity_emits_anchor_without_temporal_duplicate() -> None:
    anchor, state = _single_anchor_and_state()
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=0.0, frame_index=263, geometry_epoch=0
            ),
        ),
    )

    composed, next_state, diagnostics = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=(
            _export_batch(
                frame_index=263,
                entity_id=7,
                x=0.0,
                dynamic_state=DynamicState.STATIC,
                geometry_epoch=0,
            ),
        ),
    )

    assert [item.entity_id for item in composed.entities] == ["ovimap:1"]
    assert composed.entities[0].metadata["overlay_state"] == "unchanged"
    assert next_state.last_frame_index == 263
    assert diagnostics.unchanged_anchor_ids == ("ovimap:1",)


def test_static_bound_identity_keeps_anchor_geometry_and_uses_current_semantics() -> None:
    anchor, state = _single_anchor_and_state()
    anchor.entities[0].points_xyz = np.asarray(
        ((-0.05, 0.0, 0.0), (0.05, 0.0, 0.0)), dtype=np.float32
    )
    anchor.entities[0].semantic_label = "Table"
    anchor.entities[0].semantic_embedding = np.asarray((0.0, 1.0), dtype=np.float32)
    anchor.entities[0].semantic_score = 0.2
    anchor_points = anchor.entities[0].points_xyz.copy()
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=0.0, frame_index=263, geometry_epoch=0
            ),
        ),
    )

    composed, _, _ = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=(
            _export_batch(
                frame_index=263,
                entity_id=7,
                x=0.0,
                dynamic_state=DynamicState.STATIC,
                geometry_epoch=0,
            ),
        ),
    )

    entity = composed.entities[0]
    np.testing.assert_array_equal(entity.points_xyz, anchor_points)
    assert entity.entity_id == "ovimap:1"
    assert entity.semantic_label == "Chair"
    np.testing.assert_array_equal(
        entity.semantic_embedding, np.asarray((1.0, 0.0), dtype=np.float32)
    )
    assert entity.semantic_score == 1.0
    assert entity.metadata["semantic_authority"] == "crove_temporal"
    assert entity.metadata["semantic_temporal_entity_id"] == 7


def test_static_bound_identity_rejects_semantics_without_anchor_geometry_support() -> None:
    anchor, state = _single_anchor_and_state()
    anchor.entities[0].semantic_label = "Table"
    anchor.entities[0].semantic_embedding = np.asarray((0.0, 1.0), dtype=np.float32)
    anchor.entities[0].semantic_score = 0.2
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=0.0, frame_index=263, geometry_epoch=0
            ),
        ),
    )

    composed, _, _ = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=(
            _export_batch(
                frame_index=263,
                entity_id=7,
                x=0.0,
                dynamic_state=DynamicState.STATIC,
                geometry_epoch=0,
            ),
        ),
    )

    entity = composed.entities[0]
    assert entity.semantic_label == "Table"
    np.testing.assert_array_equal(
        entity.semantic_embedding, np.asarray((0.0, 1.0), dtype=np.float32)
    )
    assert entity.semantic_score == 0.2
    assert entity.metadata["semantic_authority"] == "ovimap_anchor"
    assert entity.metadata["semantic_update_rejection"] == (
        "insufficient_current_anchor_coverage"
    )


def test_persisted_temporal_map_snapshot_composes_with_explicit_frame() -> None:
    anchor, state = _single_anchor_and_state()
    temporal = MapSnapshot(
        method="OVIV2-temporal",
        scene_id="apartment",
        timestamp=263.0,
        entities=[
            EntityPrediction(
                entity_id="temporal:7",
                points_xyz=np.asarray(((0.0, 0.0, 0.0),), dtype=np.float32),
                semantic_embedding=np.asarray((1.0, 0.0), dtype=np.float32),
                semantic_label="Chair",
                semantic_score=1.0,
                lifecycle_state="active",
                first_seen=0.0,
                last_seen=263.0,
                metadata={
                    "temporal_entity_id": 7,
                    "semantic_id": 1,
                    "geometry_epoch": 0,
                    "readout_valid": True,
                },
            )
        ],
        background_xyz=None,
        scope="current",
    )

    composed, next_state, diagnostics = compose_anchor_checkpoint(
        anchor=anchor,
        temporal=temporal,
        frame_index=263,
        exports=(
            _export_batch(
                frame_index=263,
                entity_id=7,
                x=0.0,
                dynamic_state=DynamicState.STATIC,
                geometry_epoch=0,
            ),
        ),
        state=state,
        config=_config(),
        anchor_manifest_sha256="b" * 64,
        class_names=("unknown", "Chair", "Table"),
    )

    assert [item.entity_id for item in composed.entities] == ["ovimap:1"]
    assert next_state.last_frame_index == 263
    assert diagnostics.unchanged_anchor_ids == ("ovimap:1",)


def test_confirmed_motion_replaces_anchor_with_current_temporal_geometry() -> None:
    anchor, state = _single_anchor_and_state()
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=1.0, frame_index=263, geometry_epoch=1
            ),
        ),
    )

    composed, next_state, diagnostics = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=(
            _export_batch(
                frame_index=263,
                entity_id=7,
                x=1.0,
                dynamic_state=DynamicState.DYNAMIC,
                geometry_epoch=1,
            ),
        ),
    )

    assert [item.entity_id for item in composed.entities] == ["ovimap:1"]
    assert composed.entities[0].metadata["temporal_entity_id"] == 7
    assert composed.entities[0].metadata["overlay_state"] == "moved"
    assert next_state.moved_anchor_ids == frozenset({"ovimap:1"})
    assert diagnostics.moved_anchor_ids == ("ovimap:1",)


def test_visible_absence_removes_anchor_but_occlusion_keeps_it() -> None:
    anchor, state = _single_anchor_and_state()
    dormant = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.DORMANT, x=0.0, frame_index=263, geometry_epoch=0
            ),
        ),
    )
    occluded, occluded_state, diagnostics = _compose(
        anchor=anchor,
        state=state,
        temporal=dormant,
        exports=(
            _export_batch(
                frame_index=263,
                entity_id=7,
                x=0.0,
                dynamic_state=DynamicState.STATIC,
                geometry_epoch=0,
                evidence=TemporalEvidenceKind.OCCLUDED,
                after=TemporalLifecycle.DORMANT,
            ),
        ),
    )
    assert [item.entity_id for item in occluded.entities] == ["ovimap:1"]
    assert diagnostics.occluded_anchor_ids == ("ovimap:1",)

    removed, removed_state, diagnostics = _compose(
        anchor=anchor,
        state=state,
        temporal=dormant,
        exports=(
            _export_batch(
                frame_index=263,
                entity_id=7,
                x=0.0,
                dynamic_state=DynamicState.STATIC,
                geometry_epoch=0,
                evidence=TemporalEvidenceKind.VISIBLE_ABSENT,
                after=TemporalLifecycle.DORMANT,
            ),
        ),
    )
    assert removed.entities == []
    assert removed_state.removed_anchor_ids == frozenset({"ovimap:1"})
    assert diagnostics.removed_anchor_ids == ("ovimap:1",)
    assert occluded_state.removed_anchor_ids == frozenset()


def test_unbound_post_cutoff_identity_is_emitted_as_new() -> None:
    anchor, state = _single_anchor_and_state()
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                9,
                TemporalLifecycle.ACTIVE,
                x=3.0,
                frame_index=263,
                geometry_epoch=0,
                first_seen_frame_id=263,
            ),
        ),
    )
    composed, _, diagnostics = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=(
            _export_batch(
                frame_index=263,
                entity_id=9,
                x=3.0,
                dynamic_state=DynamicState.STATIC,
                geometry_epoch=0,
            ),
        ),
    )

    assert [item.entity_id for item in composed.entities] == [
        "ovimap:1",
        "temporal:9",
    ]
    assert diagnostics.new_temporal_ids == (9,)


def test_unbound_pre_cutoff_identity_is_suppressed_as_anchor_duplicate() -> None:
    anchor, state = _single_anchor_and_state()
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                9,
                TemporalLifecycle.ACTIVE,
                x=3.0,
                frame_index=263,
                geometry_epoch=0,
                first_seen_frame_id=10,
            ),
        ),
    )

    composed, _, diagnostics = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=(
            _export_batch(
                frame_index=263,
                entity_id=9,
                x=3.0,
                dynamic_state=DynamicState.STATIC,
                geometry_epoch=0,
            ),
        ),
    )

    assert [item.entity_id for item in composed.entities] == ["ovimap:1"]
    assert diagnostics.new_temporal_ids == ()


def test_moved_state_persists_after_dynamic_state_returns_static() -> None:
    anchor, state = _single_anchor_and_state()
    moved_snapshot = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=1.0, frame_index=263, geometry_epoch=1
            ),
        ),
    )
    _, moved_state, _ = _compose(
        anchor=anchor,
        state=state,
        temporal=moved_snapshot,
        exports=(
            _export_batch(
                frame_index=263,
                entity_id=7,
                x=1.0,
                dynamic_state=DynamicState.DYNAMIC,
                geometry_epoch=1,
            ),
        ),
    )
    static_snapshot = _temporal_snapshot(
        frame_index=313,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=1.0, frame_index=313, geometry_epoch=1
            ),
        ),
    )
    composed, persisted, _ = _compose(
        anchor=anchor,
        state=moved_state,
        temporal=static_snapshot,
        exports=(
            _export_batch(
                frame_index=313,
                entity_id=7,
                x=1.0,
                dynamic_state=DynamicState.STATIC,
                geometry_epoch=1,
            ),
        ),
    )

    assert [item.entity_id for item in composed.entities] == ["ovimap:1"]
    assert persisted.moved_anchor_ids == frozenset({"ovimap:1"})


def test_reactivation_restores_bound_anchor_after_visible_absence() -> None:
    anchor, state = _single_anchor_and_state()
    dormant = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.DORMANT, x=0.0, frame_index=263, geometry_epoch=0
            ),
        ),
    )
    _, removed_state, _ = _compose(
        anchor=anchor,
        state=state,
        temporal=dormant,
        exports=(
            _export_batch(
                frame_index=263,
                entity_id=7,
                x=0.0,
                dynamic_state=DynamicState.STATIC,
                geometry_epoch=0,
                evidence=TemporalEvidenceKind.VISIBLE_ABSENT,
                after=TemporalLifecycle.DORMANT,
            ),
        ),
    )
    active = _temporal_snapshot(
        frame_index=313,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=0.0, frame_index=313, geometry_epoch=0
            ),
        ),
    )
    composed, restored, _ = _compose(
        anchor=anchor,
        state=removed_state,
        temporal=active,
        exports=(
            _export_batch(
                frame_index=313,
                entity_id=7,
                x=0.0,
                dynamic_state=DynamicState.STATIC,
                geometry_epoch=0,
                evidence=TemporalEvidenceKind.PRESENT,
                before=TemporalLifecycle.DORMANT,
                after=TemporalLifecycle.ACTIVE,
            ),
        ),
    )

    assert [item.entity_id for item in composed.entities] == ["ovimap:1"]
    assert restored.removed_anchor_ids == frozenset()
