from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.domain.association import AssociationDecisionBatch, EntityResolutionBatch
from src.domain.entities import LifecycleDelta
from src.domain.mapping import (
    FusionDelta,
    GeometryDelta,
    OwnershipDelta,
    OwnershipLayer,
    VoxelEvidenceDelta,
)
from src.domain.observations import ObservationBatch
from src.domain.snapshots import MapSnapshot, SnapshotScope
from src.domain.tracking import LocalTrackBatch
from src.domain.visibility import VisibilityEvidenceBatch
from src.pipelines.entity_mapping_interfaces import (
    EntityAssociation,
    EntityRegistry,
    LifecycleManager,
    LocalTemporalModule,
    MapCommitter,
    MapQuery,
    ObservationFrontend,
    OwnershipManager,
    VisibilityModule,
    VoxelFusion,
)
from src.pipelines.entity_mapping_pipeline import EntityMappingModules, EntityMappingPipeline


def make_frame(frame_id: int = 0) -> Frame:
    return Frame(
        frame_id=frame_id,
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.float32),
        pose=np.eye(4),
        intrinsics=CameraIntrinsics(1.0, 1.0, 0.0, 0.0, 2, 2),
        timestamp=float(frame_id),
    )


def test_v2_module_protocols_are_importable() -> None:
    protocols = (
        ObservationFrontend,
        LocalTemporalModule,
        VisibilityModule,
        EntityAssociation,
        EntityRegistry,
        VoxelFusion,
        OwnershipManager,
        LifecycleManager,
        MapCommitter,
        MapQuery,
    )
    assert all(getattr(protocol, "_is_protocol", False) for protocol in protocols)


def test_entity_mapping_pipeline_calls_modules_in_frozen_order() -> None:
    calls: list[str] = []
    observations = ObservationBatch(frame_id=0)
    tracks = LocalTrackBatch(frame_id=0)
    visibility = VisibilityEvidenceBatch(frame_id=0)
    decisions = AssociationDecisionBatch(frame_id=0)
    resolutions = EntityResolutionBatch(frame_id=0)
    fusion = FusionDelta(GeometryDelta(0), VoxelEvidenceDelta(0))
    ownership = OwnershipDelta(base_revision=0)
    lifecycle = LifecycleDelta(base_revision=0)
    snapshot = MapSnapshot(
        scene_id="room0",
        timestamp=0.0,
        revision=1,
        scope=SnapshotScope.CURRENT,
        entities=(),
        geometry_revision=1,
        voxel_evidence_revision=1,
        ownership=OwnershipLayer(1, ()),
        lifecycle_revision=1,
    )

    class Frontend:
        def observe(self, frame):
            calls.append("frontend")
            return observations

    class Temporal:
        def update(self, value):
            assert value is observations
            calls.append("temporal")
            return tracks

    class Visibility:
        def evaluate(self, frame, value, previous):
            assert value is tracks and previous is None
            calls.append("visibility")
            return visibility

    class Association:
        def associate(self, value, evidence, previous):
            assert value is tracks and evidence is visibility and previous is None
            calls.append("association")
            return decisions

    class Registry:
        def resolve(self, value, decision_batch, previous):
            assert value is tracks and decision_batch is decisions and previous is None
            calls.append("registry")
            return resolutions

    class Fusion:
        def fuse(self, frame, observation_batch, resolution_batch, previous):
            assert observation_batch is observations and resolution_batch is resolutions and previous is None
            calls.append("fusion")
            return fusion

    class Ownership:
        def update(self, fusion_delta, previous):
            assert fusion_delta is fusion and previous is None
            calls.append("ownership")
            return ownership

    class Lifecycle:
        def update(self, evidence, decision_batch, ownership_delta, previous):
            assert evidence is visibility
            assert decision_batch is decisions
            assert ownership_delta is ownership
            assert previous is None
            calls.append("lifecycle")
            return lifecycle

    class Committer:
        def commit(self, frame, resolution_batch, fusion_delta, ownership_delta, lifecycle_delta, previous):
            assert resolution_batch is resolutions
            assert fusion_delta is fusion
            assert ownership_delta is ownership
            assert lifecycle_delta is lifecycle
            assert previous is None
            calls.append("commit")
            return snapshot

    pipeline = EntityMappingPipeline(
        EntityMappingModules(
            frontend=Frontend(),
            temporal=Temporal(),
            visibility=Visibility(),
            association=Association(),
            registry=Registry(),
            fusion=Fusion(),
            ownership=Ownership(),
            lifecycle=Lifecycle(),
            committer=Committer(),
        )
    )
    result = pipeline.process_frame(make_frame())

    assert result is snapshot
    assert pipeline.current_snapshot is snapshot
    assert calls == [
        "frontend",
        "temporal",
        "visibility",
        "association",
        "registry",
        "fusion",
        "ownership",
        "lifecycle",
        "commit",
    ]


def test_entity_mapping_pipeline_rejects_non_monotonic_snapshot_revision() -> None:
    previous = MapSnapshot(
        scene_id="room0",
        timestamp=0.0,
        revision=2,
        scope=SnapshotScope.CURRENT,
        entities=(),
        geometry_revision=1,
        voxel_evidence_revision=1,
        ownership=OwnershipLayer(1, ()),
        lifecycle_revision=1,
    )
    modules = EntityMappingModules(
        frontend=SimpleNamespace(observe=lambda frame: ObservationBatch(frame.frame_id)),
        temporal=SimpleNamespace(update=lambda batch: LocalTrackBatch(batch.frame_id)),
        visibility=SimpleNamespace(
            evaluate=lambda frame, tracks, snapshot: VisibilityEvidenceBatch(frame.frame_id)
        ),
        association=SimpleNamespace(
            associate=lambda tracks, evidence, snapshot: AssociationDecisionBatch(tracks.frame_id)
        ),
        registry=SimpleNamespace(
            resolve=lambda tracks, decisions, snapshot: EntityResolutionBatch(tracks.frame_id)
        ),
        fusion=SimpleNamespace(
            fuse=lambda frame, observations, resolutions, snapshot: FusionDelta(
                GeometryDelta(previous.geometry_revision),
                VoxelEvidenceDelta(previous.voxel_evidence_revision),
            )
        ),
        ownership=SimpleNamespace(
            update=lambda fusion, snapshot: OwnershipDelta(previous.ownership.revision)
        ),
        lifecycle=SimpleNamespace(
            update=lambda evidence, decisions, ownership, snapshot: LifecycleDelta(
                previous.lifecycle_revision
            )
        ),
        committer=SimpleNamespace(commit=lambda *args: previous),
    )
    pipeline = EntityMappingPipeline(modules)
    pipeline.current_snapshot = previous

    with pytest.raises(ValueError, match="increase monotonically"):
        pipeline.process_frame(make_frame(frame_id=1))
