from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.runtime import RuntimeFrameResult
from src.oviv2.temporal_config import (
    ExecutionProfile,
    TemporalAssociationConfig,
    TemporalGeometryConfig,
    TemporalLifecycleConfig,
    TemporalReadoutConfig,
)
from src.oviv2.temporal_lifecycle import (
    TemporalEvidence,
    TemporalEvidenceKind,
    TemporalLifecycle,
)
from src.oviv2.temporal_export import DynamicState
from src.oviv2.reference_readout import (
    CumulativeEntityView,
    CumulativeReadoutView,
    LifecycleOverlayReadout,
    ReferenceCurrentReadout,
)


def config(profile: ExecutionProfile) -> TemporalReadoutConfig:
    return TemporalReadoutConfig(
        lifecycle=TemporalLifecycleConfig(
            initial_log_odds=0.0,
            present_log_likelihood=4.0,
            absent_log_likelihood=-4.0,
            log_odds_limit=8.0,
            decay_half_life_seconds=100.0,
            active_on_probability=0.75,
            dormant_off_probability=0.25,
            minimum_absent_streak=2,
            minimum_distinct_view_bins=1,
            visibility_depth_tolerance_m=0.1,
            minimum_visible_pixel_count=1,
            minimum_visible_fraction=0.5,
            view_bin_azimuth_count=4,
            view_bin_elevation_count=2,
        ),
        association=TemporalAssociationConfig(
            visual_weight=1.0,
            semantic_weight=1.0,
            size_weight=1.0,
            motion_weight=1.0,
            geometry_weight=1.0,
            minimum_score=0.0,
            maximum_centroid_distance_m=1.0,
            semantic_conflict_probability=0.9,
            conflict_override_visual=0.9,
            conflict_override_geometry=0.9,
        ),
        geometry=TemporalGeometryConfig(
            voxel_size_m=0.05,
            depth_max_m=5.0,
            maximum_entities=20,
            maximum_object_voxels=100,
            maximum_visibility_points_per_entity=100,
            background_block_count=100,
            background_mask_dilation_px=0,
            minimum_icp_points=3,
            minimum_icp_fitness=0.0,
            maximum_icp_rmse_m=1.0,
            maximum_motion_m=1.0,
        ),
        execution_profile=profile,
    )


def frame(frame_id: int, depth: float, *, timestamp: float | None = None) -> Frame:
    return Frame(
        frame_id=frame_id,
        rgb=np.zeros((3, 3, 3), dtype=np.uint8),
        depth=np.full((3, 3), depth, dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(20.0, 20.0, 1.0, 1.0, 3, 3),
        timestamp=float(frame_id if timestamp is None else timestamp),
    )


def entity(
    entity_id: int,
    *,
    lifecycle: str = "active",
    voxel_keys: frozenset[tuple[int, int, int]] = frozenset({(0, 0, 20)}),
    centroid_xyz: tuple[float, float, float] = (0.025, 0.025, 1.025),
) -> CumulativeEntityView:
    return CumulativeEntityView(
        entity_id=entity_id,
        lifecycle_state=lifecycle,
        voxel_keys=voxel_keys,
        centroid_xyz=centroid_xyz,
        bounds_min_xyz=(0.0, 0.0, 1.0),
        bounds_max_xyz=(0.05, 0.05, 1.05),
    )


def view(
    revision: int,
    frame_id: int,
    timestamp: float,
    entities: tuple[CumulativeEntityView, ...] = (),
    *,
    scene_id: str = "room0",
) -> CumulativeReadoutView:
    return CumulativeReadoutView(
        scene_id=scene_id,
        revision=revision,
        last_frame_id=frame_id,
        last_timestamp=timestamp,
        voxel_size_m=0.05,
        depth_max_m=5.0,
        visibility_depth_tolerance_m=0.1,
        entities=entities,
    )


def cumulative(frame_id: int, revision: int, accepted: tuple[int, ...]) -> RuntimeFrameResult:
    return RuntimeFrameResult(
        frame_id=frame_id,
        revision=revision,
        geometry_blocks_touched=0,
        observation_count=len(accepted),
        updated_track_count=len(accepted),
        accepted_entity_ids=accepted,
    )


def process(
    readout: ReferenceCurrentReadout | LifecycleOverlayReadout,
    current_frame: Frame,
    before: CumulativeReadoutView,
    after: CumulativeReadoutView,
    accepted: tuple[int, ...],
):
    return readout.process_cumulative_frame(
        current_frame,
        before=before,
        after=after,
        cumulative_result=cumulative(current_frame.frame_id, after.revision, accepted),
    )


def test_module_has_no_heavy_temporal_dependency_or_object_submap() -> None:
    path = Path(__file__).parents[2] / "src" / "oviv2" / "reference_readout.py"
    source = path.read_text(encoding="utf-8")
    imports = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = {
        "src.oviv2.temporal_runtime",
        "src.oviv2.temporal_geometry",
        "src.oviv2.temporal_background",
        "src.oviv2.temporal_state",
    }
    assert imports.isdisjoint(forbidden)
    assert "ObjectSubmap" not in source
    assert "backproject" not in source
    assert "estimate_object_motion" not in source


def test_views_and_state_are_frozen_and_capture_only_registry_readout() -> None:
    from src.oviv2.runtime import Oviv2Runtime

    runtime = Oviv2Runtime("room0")
    captured = CumulativeReadoutView.capture(runtime)

    assert (captured.scene_id, captured.revision, captured.last_frame_id) == (
        "room0",
        0,
        -1,
    )
    assert captured.entities == ()
    with pytest.raises(FrozenInstanceError):
        captured.revision = 1  # type: ignore[misc]

    readout = ReferenceCurrentReadout("room0", config(ExecutionProfile.A0))
    with pytest.raises(FrozenInstanceError):
        readout.state.revision = 1  # type: ignore[misc]


def test_a0_mirrors_exact_v1_ids_native_lifecycle_and_current_geometry(monkeypatch) -> None:
    import src.oviv2.temporal_lifecycle as lifecycle_module

    monkeypatch.setattr(
        lifecycle_module,
        "advance_lifecycle",
        lambda *_args, **_kwargs: pytest.fail("A0 must not advance temporal lifecycle"),
    )
    readout = ReferenceCurrentReadout("room0", config(ExecutionProfile.A0))
    before = view(0, -1, 0.0)
    entities = (
        entity(2, lifecycle="dormant", voxel_keys=frozenset({(1, 0, 20)})),
        entity(7, voxel_keys=frozenset({(0, 0, 20), (0, 0, 21)})),
    )
    after = view(1, 4, 4.5, entities)

    result = process(readout, frame(4, 1.0, timestamp=4.5), before, after, (7,))

    assert result.cumulative_view is after
    assert result.latest_cumulative_view is after
    assert readout.state.latest_cumulative_view is after
    assert result.entity_lifecycles == ((2, "dormant"), (7, "active"))
    assert result.active_entity_ids == (7,)
    assert result.dormant_entity_ids == (2,)
    assert result.new_entity_ids == (2, 7)
    assert readout.state.cumulative_view.entities == entities
    assert readout.state.cumulative_view.entities[1].voxel_keys == frozenset(
        {(0, 0, 20), (0, 0, 21)}
    )


def test_a0_exports_only_accepted_samples_and_does_not_mutate_cumulative_view() -> None:
    readout = ReferenceCurrentReadout("room0", config(ExecutionProfile.A0))
    before = view(0, -1, 0.0)
    accepted_entity = entity(7, centroid_xyz=(1.0, 2.0, 3.0))
    after = view(1, 0, 0.0, (entity(2), accepted_entity))

    result = process(readout, frame(0, 1.0), before, after, (7,))

    assert result.cumulative_view is after
    assert after.entities[1] is accepted_entity
    assert result.export.events == ()
    assert len(result.export.samples) == 1
    assert result.export.samples[0].entity_id == 7
    assert result.export.samples[0].centroid_xyz == (1.0, 2.0, 3.0)
    assert result.export.samples[0].observation_count == 1
    assert result.export.samples[0].dynamic_state is DynamicState.UNKNOWN


def test_a0_exports_each_accepted_observation_and_skips_unobserved_frames() -> None:
    readout = ReferenceCurrentReadout("room0", config(ExecutionProfile.A0))
    empty = view(0, -1, 0.0)
    first = view(1, 0, 0.0, (entity(7, centroid_xyz=(0.0, 0.0, 1.0)),))
    first_result = process(readout, frame(0, 1.0), empty, first, (7,))
    second = view(2, 1, 1.0, (entity(7, centroid_xyz=(10.0, 0.0, 1.0)),))
    missing = process(readout, frame(1, 1.0), first, second, ())
    third = view(3, 2, 2.0, (entity(7, centroid_xyz=(0.2, 0.0, 1.0)),))
    observed = process(readout, frame(2, 1.0), second, third, (7,))

    assert first_result.export.samples[0].observation_count == 1
    assert missing.export.samples == ()
    assert observed.export.samples[0].observation_count == 2
    assert observed.export.samples[0].motion_confidence == 1.0
    assert observed.export.samples[0].dynamic_state is DynamicState.UNKNOWN


def test_a1_occlusion_is_neutral_and_uses_before_geometry() -> None:
    readout = LifecycleOverlayReadout("room0", config(ExecutionProfile.A1))
    empty = view(0, -1, 0.0)
    first = view(1, 0, 0.0, (entity(7),))
    process(readout, frame(0, 1.025), empty, first, (7,))
    old = readout.state.lifecycle_states[0]

    # The old voxel is occluded, while the post-frame geometry was moved out of view.
    moved = entity(7, voxel_keys=frozenset({(100, 100, 20)}))
    second = view(2, 1, 1.0, (moved,))
    result = process(readout, frame(1, 0.5), first, second, ())
    updated = readout.state.lifecycle_states[0]

    assert updated.existence_log_odds == old.existence_log_odds
    assert updated.absent_streak == 0
    assert updated.lifecycle is old.lifecycle
    assert readout.state.export_tracker[0].readout_valid is True
    assert result.export.events == ()


def test_a1_depth_unknown_preserves_validity_without_event() -> None:
    readout = LifecycleOverlayReadout("room0", config(ExecutionProfile.A1))
    empty = view(0, -1, 0.0)
    first = view(1, 0, 0.0, (entity(7),))
    process(readout, frame(0, 1.025), empty, first, (7,))
    second = view(2, 1, 1.0, (entity(7),))

    result = process(readout, frame(1, 0.0), first, second, ())

    assert readout.state.export_tracker[0].readout_valid is True
    assert result.export.events == ()


def test_a1_out_of_view_preserves_validity_without_event(monkeypatch) -> None:
    readout = LifecycleOverlayReadout("room0", config(ExecutionProfile.A1))
    empty = view(0, -1, 0.0)
    first = view(1, 0, 0.0, (entity(7),))
    process(readout, frame(0, 1.025), empty, first, (7,))
    second = view(2, 1, 1.0, (entity(7),))
    monkeypatch.setattr(
        readout,
        "_absence_evidence",
        lambda _entity, current_frame, _before: TemporalEvidence(
            TemporalEvidenceKind.OUT_OF_VIEW,
            0.0,
            current_frame.frame_id,
            current_frame.timestamp,
            None,
        ),
    )

    result = process(readout, frame(1, 0.0), first, second, ())

    assert readout.state.export_tracker[0].readout_valid is True
    assert result.export.events == ()


def test_a1_visible_absence_becomes_dormant_then_same_v1_id_reactivates() -> None:
    readout = LifecycleOverlayReadout("room0", config(ExecutionProfile.A1))
    empty = view(0, -1, 0.0)
    present0 = view(1, 0, 0.0, (entity(7),))
    process(readout, frame(0, 1.025), empty, present0, (7,))

    absent1 = view(2, 1, 1.0, (entity(7),))
    absent_result = process(readout, frame(1, 2.0), present0, absent1, ())
    assert absent_result.export.samples == ()
    assert len(absent_result.export.events) == 1
    assert absent_result.export.events[0].evidence_kind.value == "visible_absent"
    assert absent_result.export.events[0].readout_valid is False
    assert readout.state.export_tracker[0].readout_valid is False
    absent2 = view(3, 2, 2.0, (entity(7),))
    dormant = process(readout, frame(2, 2.0), absent1, absent2, ())
    assert dormant.dormant_entity_ids == (7,)

    present3 = view(4, 3, 3.0, (entity(7),))
    restored = process(readout, frame(3, 1.025), absent2, present3, (7,))
    assert restored.reactivated_entity_ids == (7,)
    assert restored.active_entity_ids == (7,)
    assert tuple(item.entity_id for item in restored.lifecycle_states) == (7,)
    assert readout.state.export_tracker[0].readout_valid is True
    assert len(restored.export.events) == 1
    assert restored.export.events[0].evidence_kind is TemporalEvidenceKind.PRESENT
    assert restored.export.events[0].readout_valid is True


def test_a1_neutral_present_frame_does_not_emit_lifecycle_event() -> None:
    readout = LifecycleOverlayReadout("room0", config(ExecutionProfile.A1))
    empty = view(0, -1, 0.0)
    first = view(1, 0, 0.0, (entity(7),))
    process(readout, frame(0, 1.025), empty, first, (7,))
    second = view(2, 1, 1.0, (entity(7),))

    result = process(readout, frame(1, 1.025), first, second, (7,))

    assert result.export.events == ()
    assert readout.state.export_tracker[0].readout_valid is True


def test_a1_present_authority_is_exactly_cumulative_accepted_ids() -> None:
    readout = LifecycleOverlayReadout("room0", config(ExecutionProfile.A1))
    empty = view(0, -1, 0.0)
    after = view(1, 0, 0.0, (entity(4), entity(9)))

    result = process(readout, frame(0, 2.0), empty, after, (9,))

    states = {item.entity_id: item for item in result.lifecycle_states}
    assert set(states) == {4, 9}
    assert states[9].existence_log_odds > states[4].existence_log_odds
    tracker = {item.entity_id: item for item in readout.state.export_tracker}
    assert set(tracker) == {4, 9}
    assert tracker[4].readout_valid is True
    assert tracker[9].readout_valid is True


def test_transactional_publication_keeps_state_identity_on_failure(monkeypatch) -> None:
    readout = LifecycleOverlayReadout("room0", config(ExecutionProfile.A1))
    old_state = readout.state
    before = view(0, -1, 0.0)
    after = view(1, 0, 0.0, (entity(7),))
    monkeypatch.setattr(
        readout,
        "_before_publish",
        lambda _state: (_ for _ in ()).throw(RuntimeError("injected")),
    )

    with pytest.raises(RuntimeError, match="injected"):
        process(readout, frame(0, 1.025), before, after, (7,))

    assert readout.state is old_state


def test_transaction_retry_is_byte_identical_including_export_tracker(monkeypatch) -> None:
    readout = LifecycleOverlayReadout("room0", config(ExecutionProfile.A1))
    empty = view(0, -1, 0.0)
    present = view(1, 0, 0.0, (entity(7),))
    process(readout, frame(0, 1.025), empty, present, (7,))
    absent = view(2, 1, 1.0, (entity(7),))
    process(readout, frame(1, 2.0), present, absent, ())
    snapshot = readout.transaction_snapshot()
    assert snapshot.export_tracker[0].readout_valid is False
    restored = view(3, 2, 2.0, (entity(7),))
    original_hook = readout._before_publish
    monkeypatch.setattr(
        readout,
        "_before_publish",
        lambda _state: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    with pytest.raises(RuntimeError, match="injected"):
        process(readout, frame(2, 1.025), absent, restored, (7,))
    assert readout.transaction_snapshot() is snapshot
    assert readout.state.export_tracker[0].readout_valid is False

    monkeypatch.setattr(readout, "_before_publish", original_hook)
    retried = process(readout, frame(2, 1.025), absent, restored, (7,))

    clean = LifecycleOverlayReadout("room0", config(ExecutionProfile.A1))
    clean.restore_transaction(snapshot)
    expected = process(clean, frame(2, 1.025), absent, restored, (7,))
    assert retried.export.to_canonical_json() == expected.export.to_canonical_json()
    assert retried.export.samples[0].observation_count == 2
    assert readout.state.export_tracker[0].readout_valid is True


@pytest.mark.parametrize("kind", ["profile", "scene", "progress", "result", "timestamp"])
def test_fail_closed_bindings(kind: str) -> None:
    if kind == "profile":
        with pytest.raises(ValueError, match="A0"):
            ReferenceCurrentReadout("room0", config(ExecutionProfile.A1))
        with pytest.raises(ValueError, match="A1"):
            LifecycleOverlayReadout("room0", config(ExecutionProfile.A0))
        return

    readout = LifecycleOverlayReadout("room0", config(ExecutionProfile.A1))
    before = view(0, -1, 0.0)
    after = view(1, 0, 0.0, (entity(7),))
    current_frame = frame(0, 1.025)
    result = cumulative(0, 1, (7,))
    if kind == "scene":
        after = replace(after, scene_id="other")
    elif kind == "progress":
        before = replace(before, revision=2)
    elif kind == "result":
        result = replace(result, revision=3)
    else:
        current_frame.timestamp = 1.0

    with pytest.raises(ValueError):
        readout.process_cumulative_frame(
            current_frame,
            before=before,
            after=after,
            cumulative_result=result,
        )


def test_a1_is_deterministic_for_identical_inputs() -> None:
    before = view(0, -1, 0.0)
    after = view(1, 0, 0.0, (entity(9), entity(4)))
    outputs = []
    for _ in range(2):
        readout = LifecycleOverlayReadout("room0", config(ExecutionProfile.A1))
        outputs.append(process(readout, frame(0, 1.025), before, after, (9, 4)))

    assert outputs[0] == outputs[1]
    assert outputs[0].entity_lifecycles == ((4, "active"), (9, "active"))
