from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import src.oviv2.ovimap_static_anchor as anchor_module
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.anchor_visibility import (
    AnchorCurrentOwnership,
    AnchorVisibilityConfig,
    initialize_anchor_current_ownership,
)
from src.oviv2.dense_moved_readout import DenseMovedReadoutGateConfig
from src.oviv2.ovimap_static_anchor import (
    PrefixIdentitySample,
    StaticAnchorConfig,
    bind_anchor_identities,
    compose_anchor_checkpoint,
)
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
    moved_geometry_mode: str = "temporal_compact",
    suppressed_unbound_anchor_ids: frozenset[str] = frozenset(),
    localized_unbound_ownership: tuple[AnchorCurrentOwnership, ...] | None = None,
    dense_geometry_gate: DenseMovedReadoutGateConfig | None = None,
):
    return compose_anchor_checkpoint(
        anchor=anchor,
        temporal=temporal,
        exports=exports,
        state=state,
        config=_config(),
        anchor_manifest_sha256="b" * 64,
        class_names=("unknown", "Chair", "Table"),
        moved_geometry_mode=moved_geometry_mode,
        suppressed_unbound_anchor_ids=suppressed_unbound_anchor_ids,
        localized_unbound_ownership=localized_unbound_ownership,
        dense_geometry_gate=dense_geometry_gate,
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


def test_dense_moved_geometry_translates_anchor_to_current_export_centroid() -> None:
    anchor, state = _single_anchor_and_state()
    anchor.entities[0].points_xyz = np.asarray(
        (
            (-0.20, -0.10, 0.0),
            (-0.10, 0.10, 0.0),
            (0.00, 0.00, 0.0),
            (0.10, -0.10, 0.0),
            (0.20, 0.10, 0.0),
        ),
        dtype=np.float32,
    )
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=1.0, frame_index=263, geometry_epoch=1
            ),
        ),
    )
    exports = (
        _export_batch(
            frame_index=263,
            entity_id=7,
            x=1.25,
            dynamic_state=DynamicState.DYNAMIC,
            geometry_epoch=1,
        ),
    )

    composed, next_state, diagnostics = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=exports,
        moved_geometry_mode="anchor_centroid_translation",
    )

    entity = composed.entities[0]
    delta = np.asarray(exports[0].samples[0].centroid_xyz) - np.asarray(
        anchor.entities[0].points_xyz, dtype=np.float64
    ).mean(axis=0)
    np.testing.assert_allclose(
        entity.points_xyz,
        np.asarray(anchor.entities[0].points_xyz, dtype=np.float64) + delta,
        rtol=0.0,
        atol=1e-7,
    )
    assert entity.semantic_label == "Chair"
    assert entity.lifecycle_state == "active"
    assert entity.metadata["authority"] == "crove_temporal"
    assert entity.metadata["geometry_authority"] == "ovimap_anchor_template"
    assert entity.metadata["geometry_source"] == "causal_ovimap_anchor"
    assert entity.metadata["state_authority"] == "crove_temporal"
    assert entity.metadata["template_anchor_id"] == "ovimap:1"
    assert (
        entity.metadata["transform_source"]
        == "current_export_centroid_translation"
    )
    assert entity.metadata["readout_resolution_m"] is None
    assert (
        entity.metadata["readout_resolution_source"]
        == "native_ovimap_mesh_not_declared"
    )
    assert next_state.moved_anchor_ids == frozenset({"ovimap:1"})
    assert diagnostics.moved_anchor_ids == ("ovimap:1",)


def test_dense_moved_geometry_uses_temporal_geometry_centroid_without_sample() -> None:
    anchor, state = _single_anchor_and_state()
    state = replace(state, moved_anchor_ids=frozenset({"ovimap:1"}))
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=1.0, frame_index=263, geometry_epoch=1
            ),
        ),
    )

    composed, _, _ = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=(TemporalExportBatch(263, 263_000_000_000, (), ()),),
        moved_geometry_mode="anchor_centroid_translation",
    )

    entity = composed.entities[0]
    np.testing.assert_allclose(
        np.asarray(entity.points_xyz, dtype=np.float64).mean(axis=0),
        (1.0, 0.0, 0.0),
        rtol=0.0,
        atol=1e-7,
    )
    assert (
        entity.metadata["transform_source"]
        == "temporal_geometry_centroid_translation"
    )


def test_hybrid_dense_geometry_accepts_template_only_when_agreement_passes() -> None:
    anchor, state = _single_anchor_and_state()
    anchor.entities[0].points_xyz = np.asarray(
        ((-0.05, 0.0, 0.0), (0.05, 0.0, 0.0)), dtype=np.float32
    )
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=1.0, frame_index=263, geometry_epoch=1
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
                entity_id=7,
                x=1.0,
                dynamic_state=DynamicState.DYNAMIC,
                geometry_epoch=1,
            ),
        ),
        moved_geometry_mode="geometry_gated_anchor_translation",
        dense_geometry_gate=DenseMovedReadoutGateConfig(),
    )

    entity = composed.entities[0]
    assert entity.metadata["geometry_authority"] == "ovimap_anchor_template"
    assert entity.metadata["geometry_gate_accepted"] is True
    assert entity.metadata["geometry_gate_rejection_reasons"] == []
    assert len(diagnostics.dense_geometry_decisions) == 1
    assert diagnostics.dense_geometry_decisions[0].accepted is True


def test_hybrid_dense_geometry_failure_is_exact_compact_fallback() -> None:
    anchor, state = _single_anchor_and_state()
    anchor.entities[0].points_xyz = np.asarray(
        ((-0.50, 0.0, 0.0), (0.50, 0.0, 0.0)), dtype=np.float32
    )
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=1.0, frame_index=263, geometry_epoch=1
            ),
        ),
    )
    exports = (
        _export_batch(
            frame_index=263,
            entity_id=7,
            x=1.0,
            dynamic_state=DynamicState.DYNAMIC,
            geometry_epoch=1,
        ),
    )

    compact, compact_state, _ = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=exports,
    )
    hybrid, hybrid_state, diagnostics = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=exports,
        moved_geometry_mode="geometry_gated_anchor_translation",
        dense_geometry_gate=DenseMovedReadoutGateConfig(),
    )

    assert hybrid_state == compact_state
    np.testing.assert_array_equal(
        hybrid.entities[0].points_xyz,
        compact.entities[0].points_xyz,
    )
    audit_keys = {
        "geometry_gate_id",
        "geometry_gate_accepted",
        "geometry_gate_rejection_reasons",
    }
    assert {
        key: value
        for key, value in hybrid.entities[0].metadata.items()
        if key not in audit_keys
    } == compact.entities[0].metadata
    assert hybrid.entities[0].metadata["geometry_gate_accepted"] is False
    assert diagnostics.dense_geometry_decisions[0].accepted is False


def test_dense_geometry_gate_must_be_paired_with_hybrid_mode() -> None:
    anchor, state = _single_anchor_and_state()
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=1.0, frame_index=263, geometry_epoch=1
            ),
        ),
    )
    exports = (
        _export_batch(
            frame_index=263,
            entity_id=7,
            x=1.0,
            dynamic_state=DynamicState.DYNAMIC,
            geometry_epoch=1,
        ),
    )

    with pytest.raises(ValueError, match="dense geometry gate"):
        _compose(
            anchor=anchor,
            state=state,
            temporal=temporal,
            exports=exports,
            dense_geometry_gate=DenseMovedReadoutGateConfig(),
        )
    with pytest.raises(ValueError, match="dense geometry gate"):
        _compose(
            anchor=anchor,
            state=state,
            temporal=temporal,
            exports=exports,
            moved_geometry_mode="geometry_gated_anchor_translation",
        )


def test_default_and_explicit_compact_moved_geometry_are_identical() -> None:
    anchor, state = _single_anchor_and_state()
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=1.0, frame_index=263, geometry_epoch=1
            ),
        ),
    )
    exports = (
        _export_batch(
            frame_index=263,
            entity_id=7,
            x=1.0,
            dynamic_state=DynamicState.DYNAMIC,
            geometry_epoch=1,
        ),
    )

    default, default_state, default_diagnostics = compose_anchor_checkpoint(
        anchor=anchor,
        temporal=temporal,
        exports=exports,
        state=state,
        config=_config(),
        anchor_manifest_sha256="b" * 64,
        class_names=("unknown", "Chair", "Table"),
    )
    explicit, explicit_state, explicit_diagnostics = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=exports,
        moved_geometry_mode="temporal_compact",
    )

    assert explicit_state == default_state
    assert explicit_diagnostics == default_diagnostics
    assert len(explicit.entities) == len(default.entities) == 1
    np.testing.assert_array_equal(
        explicit.entities[0].points_xyz, default.entities[0].points_xyz
    )
    assert explicit.entities[0].metadata == default.entities[0].metadata
    assert explicit.entities[0].semantic_label == default.entities[0].semantic_label
    assert explicit.entities[0].lifecycle_state == default.entities[0].lifecycle_state


def test_moved_geometry_mode_rejects_unknown_value() -> None:
    anchor, state = _single_anchor_and_state()
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=1.0, frame_index=263, geometry_epoch=1
            ),
        ),
    )

    with pytest.raises(ValueError, match="moved_geometry_mode"):
        _compose(
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
            moved_geometry_mode="unknown",
        )


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


def test_explicit_visibility_suppression_omits_only_unbound_anchor() -> None:
    anchor = _anchor()
    state = bind_anchor_identities(
        anchor,
        (_sample(7, 0.0, "Chair", (1.0, 0.0)),),
        _config(),
        cutoff_frame=262,
    )
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
        suppressed_unbound_anchor_ids=frozenset({"ovimap:2"}),
    )

    assert [item.entity_id for item in composed.entities] == ["ovimap:1"]
    assert diagnostics.unchanged_anchor_ids == ("ovimap:1",)
    assert next_state.bindings == state.bindings
    assert next_state.removed_anchor_ids == state.removed_anchor_ids


@pytest.mark.parametrize("anchor_id", ("ovimap:1", "ovimap:missing"))
def test_visibility_suppression_rejects_bound_or_unknown_anchor(
    anchor_id: str,
) -> None:
    anchor = _anchor()
    state = bind_anchor_identities(
        anchor,
        (_sample(7, 0.0, "Chair", (1.0, 0.0)),),
        _config(),
        cutoff_frame=262,
    )
    temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7, TemporalLifecycle.ACTIVE, x=0.0, frame_index=263, geometry_epoch=0
            ),
        ),
    )

    with pytest.raises(ValueError, match="unbound anchor"):
        _compose(
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
            suppressed_unbound_anchor_ids=frozenset({anchor_id}),
        )


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


def _localized_visibility_config() -> AnchorVisibilityConfig:
    return AnchorVisibilityConfig(
        voxel_size_m=0.05,
        depth_tolerance_m=0.1,
        depth_max_m=10.0,
        maximum_voxels_per_anchor=1_000,
        minimum_tested_voxels=1,
        minimum_absent_fraction=0.8,
        minimum_present_fraction=0.8,
        minimum_absent_observations=6,
        minimum_distinct_viewpoints=3,
        minimum_viewpoint_baseline_m=0.25,
        minimum_present_streak=2,
    )


def _unbound_anchor_fixture():
    anchor = _anchor()
    anchor.entities = anchor.entities[:1]
    state = bind_anchor_identities(anchor, (), _config(), cutoff_frame=262)
    temporal = _temporal_snapshot(frame_index=263, entities=())
    exports = (
        _export_batch(
            frame_index=263,
            entity_id=99,
            x=4.0,
            dynamic_state=DynamicState.STATIC,
            geometry_epoch=0,
        ),
    )
    ownership = initialize_anchor_current_ownership(
        "ovimap:1",
        anchor.entities[0].points_xyz,
        _localized_visibility_config(),
        frame_id=263,
        timestamp=263.0,
    )
    return anchor, state, temporal, exports, ownership


def test_localized_ownership_filters_points_without_downsampling_or_relabeling() -> None:
    anchor, state, temporal, exports, ownership = _unbound_anchor_fixture()
    current = ownership.current_mask.copy()
    zero_voxel = np.flatnonzero(np.all(ownership.voxel_keys == (0, 0, 0), axis=1))
    assert zero_voxel.tolist() == [1]
    current[zero_voxel[0]] = False
    ownership = replace(ownership, current_mask=current)

    composed, _, _ = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=exports,
        localized_unbound_ownership=(ownership,),
    )

    assert [item.entity_id for item in composed.entities] == ["ovimap:1"]
    entity = composed.entities[0]
    assert np.array_equal(
        entity.points_xyz,
        np.asarray(
            ((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)), dtype=np.float32
        ),
    )
    assert entity.semantic_label == anchor.entities[0].semantic_label
    assert np.array_equal(
        entity.semantic_embedding, anchor.entities[0].semantic_embedding
    )
    assert entity.semantic_score == anchor.entities[0].semantic_score
    assert entity.metadata == {
        "authority": "ovimap_anchor",
        "anchor_entity_id": "ovimap:1",
        "anchor_manifest_sha256": "b" * 64,
        "owner_entity_id": "anchor:ovimap:1",
        "semantic_authority": "ovimap_anchor",
        "state_authority": "crove_anchor_visibility",
        "overlay_state": "partially_suppressed",
        "ownership_policy_id": ownership.policy_id,
        "ownership_voxel_size_m": 0.05,
        "active_voxel_count": 2,
        "suppressed_voxel_count": 1,
        "active_point_count": 2,
        "suppressed_point_count": 1,
    }
    assert all("mask" not in key for key in entity.metadata)


def test_localized_ownership_reports_unchanged_and_omits_dormant_anchor() -> None:
    anchor, state, temporal, exports, ownership = _unbound_anchor_fixture()
    unchanged, _, _ = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=exports,
        localized_unbound_ownership=(ownership,),
    )
    assert np.array_equal(
        unchanged.entities[0].points_xyz, anchor.entities[0].points_xyz
    )
    assert unchanged.entities[0].metadata["overlay_state"] == "unchanged"
    assert unchanged.entities[0].metadata["active_point_count"] == 3
    dormant = replace(
        ownership,
        current_mask=np.zeros(len(ownership.current_mask), dtype=np.bool_),
    )
    suppressed, _, _ = _compose(
        anchor=anchor,
        state=state,
        temporal=temporal,
        exports=exports,
        localized_unbound_ownership=(dormant,),
    )
    assert suppressed.entities == []


def test_localized_ownership_rejects_bound_unknown_and_legacy_mix() -> None:
    anchor, state, temporal, exports, ownership = _unbound_anchor_fixture()
    with pytest.raises(ValueError, match="cover every unbound anchor"):
        _compose(
            anchor=anchor,
            state=state,
            temporal=temporal,
            exports=exports,
            localized_unbound_ownership=(),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _compose(
            anchor=anchor,
            state=state,
            temporal=temporal,
            exports=exports,
            suppressed_unbound_anchor_ids=frozenset({"ovimap:1"}),
            localized_unbound_ownership=(ownership,),
        )

    bound_anchor, bound_state = _single_anchor_and_state()
    bound_temporal = _temporal_snapshot(
        frame_index=263,
        entities=(
            _temporal_entity(
                7,
                TemporalLifecycle.ACTIVE,
                x=0.0,
                frame_index=263,
                geometry_epoch=0,
            ),
        ),
    )
    bound_ownership = initialize_anchor_current_ownership(
        "ovimap:1",
        bound_anchor.entities[0].points_xyz,
        _localized_visibility_config(),
        frame_id=263,
        timestamp=263.0,
    )
    with pytest.raises(ValueError, match="unbound anchor"):
        _compose(
            anchor=bound_anchor,
            state=bound_state,
            temporal=bound_temporal,
            exports=(
                _export_batch(
                    frame_index=263,
                    entity_id=7,
                    x=0.0,
                    dynamic_state=DynamicState.STATIC,
                    geometry_epoch=0,
                ),
            ),
            localized_unbound_ownership=(bound_ownership,),
        )
