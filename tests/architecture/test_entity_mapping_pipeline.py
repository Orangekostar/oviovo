from __future__ import annotations

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
