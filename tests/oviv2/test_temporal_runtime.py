from __future__ import annotations

import copy
import pickle
import warnings
from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.temporal_config import (
    ExecutionProfile,
    TemporalAssociationConfig,
    TemporalGeometryConfig,
    TemporalLifecycleConfig,
    TemporalReadoutConfig,
)
from src.oviv2.tracking import LocalTrackerConfig


def _config(
    execution_profile: ExecutionProfile = ExecutionProfile.A4,
    **geometry_changes: object,
) -> TemporalReadoutConfig:
    geometry = dict(
        voxel_size_m=0.1,
        depth_max_m=4.0,
        maximum_entities=4,
        maximum_object_voxels=32,
        maximum_visibility_points_per_entity=32,
        background_block_count=128,
        background_mask_dilation_px=0,
        minimum_icp_points=100,
        minimum_icp_fitness=0.5,
        maximum_icp_rmse_m=0.1,
        maximum_motion_m=2.0,
    )
    geometry.update(geometry_changes)
    return TemporalReadoutConfig(
        lifecycle=TemporalLifecycleConfig(
            initial_log_odds=0.0,
            present_log_likelihood=4.0,
            absent_log_likelihood=-5.0,
            log_odds_limit=12.0,
            decay_half_life_seconds=100.0,
            active_on_probability=0.7,
            dormant_off_probability=0.3,
            minimum_absent_streak=2,
            minimum_distinct_view_bins=1,
            visibility_depth_tolerance_m=0.1,
            minimum_visible_pixel_count=1,
            minimum_visible_fraction=0.5,
            view_bin_azimuth_count=8,
            view_bin_elevation_count=4,
        ),
        association=TemporalAssociationConfig(
            visual_weight=1.0,
            semantic_weight=1.0,
            size_weight=1.0,
            motion_weight=1.0,
            geometry_weight=1.0,
            minimum_score=0.2,
            maximum_centroid_distance_m=2.0,
            semantic_conflict_probability=0.9,
            conflict_override_visual=0.9,
            conflict_override_geometry=0.9,
        ),
        geometry=TemporalGeometryConfig(**geometry),
        execution_profile=execution_profile,
    )


def _tracker_config() -> LocalTrackerConfig:
    return LocalTrackerConfig(
        confirm_hits=2,
        max_age_frames=20,
        min_voxel_overlap=0.0,
        max_centroid_distance_m=2.0,
    )


def _frame(
    frame_id: int,
    *,
    timestamp: float | None = None,
    depth: float = 1.0,
    camera_x: float = 0.0,
) -> Frame:
    depth_image = np.full((5, 5), depth, dtype=np.float32)
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = camera_x
    return Frame(
        frame_id=frame_id,
        timestamp=float(frame_id if timestamp is None else timestamp),
        source_frame_id=frame_id + 100,
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth=depth_image,
        pose=pose,
        intrinsics=CameraIntrinsics(4.0, 4.0, 2.0, 2.0, 5, 5),
    )


def _observation(
    frame: Frame,
    observation_id: int | None = None,
    *,
    centroid_z: float | None = None,
    semantic_id: int = 1,
    confidence: float = 1.0,
    feature_model_id: str = "test",
    image_feature: np.ndarray | None = None,
) -> FrameObservation:
    observation_id = 10 + frame.frame_id if observation_id is None else observation_id
    z = float(frame.depth[2, 2] if centroid_z is None else centroid_z)
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    return FrameObservation(
        observation_id=observation_id,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        kind=ObservationKind.OBJECT,
        label=f"object-{semantic_id}",
        semantic_id=semantic_id,
        confidence=confidence,
        mask=mask,
        bbox_xyxy=(2.0, 2.0, 3.0, 3.0),
        voxel_keys=frozenset({(0, 0, 10)}),
        centroid_xyz=(0.0, 0.0, z),
        bounds_min_xyz=(-0.05, -0.05, z - 0.05),
        bounds_max_xyz=(0.05, 0.05, z + 0.05),
        image_feature=(
            np.asarray([1.0, 0.0], dtype=np.float32)
            if image_feature is None
            else image_feature
        ),
        feature_model_id=feature_model_id,
        visible_pixel_count=1,
    )


def _pixel_observation(
    frame: Frame,
    observation_id: int,
    row: int,
    column: int,
    *,
    semantic_id: int,
    image_feature: np.ndarray,
) -> FrameObservation:
    depth = float(frame.depth[row, column])
    x = (column - frame.intrinsics.cx) * depth / frame.intrinsics.fx
    y = (row - frame.intrinsics.cy) * depth / frame.intrinsics.fy
    mask = np.zeros(frame.depth.shape, dtype=bool)
    mask[row, column] = True
    return FrameObservation(
        observation_id=observation_id,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        kind=ObservationKind.OBJECT,
        label=f"pixel-{semantic_id}",
        semantic_id=semantic_id,
        confidence=1.0,
        mask=mask,
        bbox_xyxy=(float(column), float(row), float(column + 1), float(row + 1)),
        voxel_keys=frozenset({(int(x * 10), int(y * 10), int(depth * 10))}),
        centroid_xyz=(x, y, depth),
        bounds_min_xyz=(x - 0.05, y - 0.05, depth - 0.05),
        bounds_max_xyz=(x + 0.05, y + 0.05, depth + 0.05),
        image_feature=image_feature,
        feature_model_id="test",
        visible_pixel_count=1,
    )


def _dense_semantics(frame: Frame) -> DenseSemanticFrame:
    source = frame.frame_id if frame.source_frame_id is None else frame.source_frame_id
    return DenseSemanticFrame(
        cache_frame_id=frame.frame_id,
        source_frame_id=source,
        image_shape=frame.depth.shape,
        sample_stride=1,
        class_count=1,
        class_ids=np.ones((*frame.depth.shape, 1), dtype=np.int64),
        probabilities=np.ones((*frame.depth.shape, 1), dtype=np.float32),
        entropy=np.zeros(frame.depth.shape, dtype=np.float32),
        margin=np.ones(frame.depth.shape, dtype=np.float32),
    )


def _proposal_evidence(
    frame: Frame,
    dense: DenseSemanticFrame,
    observations: tuple[FrameObservation, ...] = (),
):
    from src.oviv2.temporal_runtime import _proposal_semantic_provenance_hash
    from src.oviv2.temporal_proposals import ProposalRecoveryInput

    depth = np.asarray(frame.depth, dtype=np.float32)
    rows, columns = np.indices(depth.shape)
    xyz = np.stack(
        (
            (columns - frame.intrinsics.cx) * depth / frame.intrinsics.fx,
            (rows - frame.intrinsics.cy) * depth / frame.intrinsics.fy,
            depth,
        ),
        axis=-1,
    ).astype(np.float32)
    occupied = np.zeros(depth.shape, dtype=bool)
    for observation in observations:
        if observation.kind in (ObservationKind.OBJECT, ObservationKind.UNKNOWN):
            occupied |= observation.mask
    return ProposalRecoveryInput(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        depth_m=depth,
        current_xyz=xyz,
        segmentation_occupied=occupied,
        semantic_support=np.ones(depth.shape, dtype=bool),
        semantic_source_frame_id=frame.frame_id,
        semantic_provenance_hash=_proposal_semantic_provenance_hash(frame, dense),
        appearance_support=None,
        appearance_source_frame_id=None,
        appearance_model_id=None,
        appearance_provenance_hash=None,
        search_regions=(),
    )


def _runtime(config: TemporalReadoutConfig | None = None):
    from src.oviv2.temporal_runtime import TemporalCurrentRuntime

    return TemporalCurrentRuntime("scene", config or _config(), _tracker_config())


def _confirm(runtime, first_id: int = 0) -> int:
    first = _frame(first_id, timestamp=0.0 if first_id == 0 else None)
    assert runtime.process_frame(first, (_observation(first),)).new_entity_ids == ()
    second = _frame(first_id + 1, timestamp=1.0 if first_id == 0 else None)
    result = runtime.process_frame(second, (_observation(second),))
    assert len(result.new_entity_ids) == 1
    return result.new_entity_ids[0]


def test_static_object_has_stable_id_without_duplicate_and_timestamp_zero_is_valid() -> None:
    runtime = _runtime()
    entity_id = _confirm(runtime)
    third = _frame(2, timestamp=2.0)
    result = runtime.process_frame(third, (_observation(third),))
    assert result.active_entity_ids == (entity_id,)
    assert result.new_entity_ids == ()
    assert tuple(item.lifecycle.entity_id for item in runtime.state.entities) == (entity_id,)


def test_one_frame_proposal_remains_only_in_tracker() -> None:
    runtime = _runtime()
    frame = _frame(0, timestamp=0.0)
    result = runtime.process_frame(frame, (_observation(frame),))
    assert result.active_entity_ids == result.new_entity_ids == ()
    assert runtime.state.entities == ()
    assert len(runtime.state.tracker.tracks) == 1


def test_occlusion_is_neutral_but_valid_signed_depth_absence_retires() -> None:
    occluded = _runtime()
    entity_id = _confirm(occluded)
    for frame_id in (2, 3, 4):
        result = occluded.process_frame(_frame(frame_id, depth=0.5), ())
    assert entity_id in result.active_entity_ids
    assert result.dormant_entity_ids == ()

    removed = _runtime()
    entity_id = _confirm(removed)
    assert removed.process_frame(_frame(2, depth=2.0), ()).dormant_entity_ids == ()
    result = removed.process_frame(_frame(3, depth=2.0, camera_x=0.05), ())
    assert result.dormant_entity_ids == (entity_id,)
    assert result.diagnostics.ledger_commit_count == 1
    assert removed.state.background.active_block_count > 0


def test_visible_absent_invalidates_geometry_before_identity_is_dormant() -> None:
    runtime = _runtime()
    entity_id = _confirm(runtime)

    result = runtime.process_frame(_frame(2, depth=2.0), ())

    assert entity_id in result.active_entity_ids
    assert runtime.state.geometry.current(entity_id).readout_valid is False
    assert result.export.samples == ()
    assert result.export.events[0].entity_id == entity_id
    assert result.export.events[0].readout_valid is False


def test_occlusion_does_not_revive_invalid_geometry() -> None:
    runtime = _runtime()
    entity_id = _confirm(runtime)
    runtime.process_frame(_frame(2, depth=2.0), ())

    result = runtime.process_frame(_frame(3, depth=0.5), ())

    assert runtime.state.geometry.current(entity_id).readout_valid is False
    assert runtime.state.background_ledger.provisional_count == 0
    assert result.export.frame_index == 3


def test_every_successful_frame_returns_an_export_batch_and_exact_counters() -> None:
    runtime = _runtime()
    first = runtime.process_frame(_frame(0, timestamp=0.0), ())

    assert first.export.samples == first.export.events == ()
    assert first.proposal_opportunity_count == 0
    assert first.proposal_trigger_count == 0
    assert first.reid_opportunity_count == 0
    assert first.reid_trigger_count == 0


def test_moved_object_keeps_id_and_moves_old_geometry() -> None:
    runtime = _runtime()
    entity_id = _confirm(runtime)
    before = runtime.state.entities[0].submap.world_points(
        runtime.state.entities[0].object_to_world
    )
    frame = _frame(2, depth=1.4)
    result = runtime.process_frame(frame, (_observation(frame, centroid_z=1.4),))
    after = runtime.state.entities[0].submap.world_points(
        runtime.state.entities[0].object_to_world
    )
    assert result.active_entity_ids == (entity_id,)
    assert result.new_entity_ids == ()
    assert float(after[:, 2].min()) > float(before[:, 2].max()) + 0.2


def test_cross_model_geometry_match_atomically_replaces_prototype_provenance() -> None:
    runtime = _runtime()
    entity_id = _confirm(runtime)
    frame = _frame(2)
    result = runtime.process_frame(
        frame,
        (_observation(frame, feature_model_id="new", image_feature=np.array([0.0, 1.0, 0.0])),),
    )
    entity = runtime.state.entities[0]
    assert result.active_entity_ids == (entity_id,)
    assert entity.feature_model_id == "new"
    np.testing.assert_allclose(entity.image_prototype, [0.0, 1.0, 0.0])

    frame = _frame(3)
    runtime.process_frame(frame, (_observation(frame, feature_model_id="test"),))
    entity = runtime.state.entities[0]
    assert entity.feature_model_id == "test"
    np.testing.assert_allclose(entity.image_prototype, [1.0, 0.0])


def test_dormant_object_reappears_with_same_id() -> None:
    runtime = _runtime()
    entity_id = _confirm(runtime)
    runtime.process_frame(_frame(2, depth=2.0), ())
    assert runtime.process_frame(_frame(3, depth=2.0), ()).dormant_entity_ids == (
        entity_id,
    )
    frame = _frame(4)
    result = runtime.process_frame(frame, (_observation(frame),))
    assert result.reactivated_entity_ids == ()
    frame = _frame(5)
    result = runtime.process_frame(frame, (_observation(frame),))
    assert result.reactivated_entity_ids == (entity_id,)
    assert result.new_entity_ids == ()


def test_rejected_high_confidence_reid_starts_epoch_without_old_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_epoch import GeometryEpoch
    from src.oviv2.temporal_geometry import MotionDecision, ObjectMotionEstimate

    runtime = _runtime()
    entity_id = _confirm(runtime)
    runtime.process_frame(_frame(2, depth=2.0), ())
    runtime.process_frame(_frame(3, depth=2.0), ())
    first = _frame(4)
    runtime.process_frame(first, (_observation(first),))

    monkeypatch.setattr(
        module,
        "estimate_object_motion",
        lambda *args, **kwargs: ObjectMotionEstimate(
            kwargs["previous_object_to_world"], MotionDecision.REJECTED, 0.0, 0.1
        ),
    )
    monkeypatch.setattr(
        GeometryEpoch,
        "integrate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("old epoch")),
    )
    second = _frame(5)
    result = runtime.process_frame(second, (_observation(second),))

    assert result.reactivated_entity_ids == (entity_id,)
    assert runtime.state.geometry.current(entity_id).epoch_id == 1


def test_rejected_low_confidence_match_creates_new_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_geometry import MotionDecision, ObjectMotionEstimate

    runtime = _runtime()
    old_id = _confirm(runtime)
    original = module.associate_temporal_observations

    def low_confidence(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(
            result,
            assignment_diagnostics=tuple(
                replace(item, high_confidence_identity_match=False)
                for item in result.assignment_diagnostics
            ),
        )

    monkeypatch.setattr(module, "associate_temporal_observations", low_confidence)
    monkeypatch.setattr(
        module,
        "estimate_object_motion",
        lambda *args, **kwargs: ObjectMotionEstimate(
            kwargs["previous_object_to_world"], MotionDecision.REJECTED, 0.0, 0.1
        ),
    )
    frame = _frame(2)
    result = runtime.process_frame(frame, (_observation(frame),))

    assert result.new_entity_ids == (old_id + 1,)
    assert runtime.state.geometry.current(old_id).epoch_id == 0
    assert runtime.state.geometry.current(old_id + 1).epoch_id == 0


@pytest.mark.parametrize("profile", (ExecutionProfile.A2, ExecutionProfile.A3))
def test_translation_rejection_with_high_identity_confidence_starts_new_epoch(
    monkeypatch: pytest.MonkeyPatch, profile: ExecutionProfile
) -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_geometry import MotionDecision, ObjectMotionEstimate

    runtime = _runtime(_config(profile))
    entity_id = _confirm(runtime)
    monkeypatch.setattr(
        module,
        "estimate_object_translation",
        lambda *args, **kwargs: ObjectMotionEstimate(
            kwargs["previous_object_to_world"],
            MotionDecision.REJECTED,
            0.0,
            0.1,
        ),
    )
    frame = _frame(2)

    result = runtime.process_frame(frame, (_observation(frame),))

    assert result.new_entity_ids == ()
    assert result.reid_opportunity_count == result.reid_trigger_count == 0
    assert runtime.state.geometry.current(entity_id).epoch_id == 1
    assert result.diagnostics.epoch_reset_opportunity_count == 1
    assert result.diagnostics.epoch_reset_trigger_count == 1
    assert result.diagnostics.motion_rejection_count == 1
    assert result.diagnostics.icp_opportunity_count == 0


@pytest.mark.parametrize("profile", (ExecutionProfile.A2, ExecutionProfile.A3))
def test_translation_rejection_with_low_identity_confidence_creates_new_id(
    monkeypatch: pytest.MonkeyPatch, profile: ExecutionProfile
) -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_geometry import MotionDecision, ObjectMotionEstimate

    runtime = _runtime(_config(profile))
    old_id = _confirm(runtime)
    original = module.associate_temporal_observations

    def low_confidence(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(
            result,
            assignment_diagnostics=tuple(
                replace(item, high_confidence_identity_match=False)
                for item in result.assignment_diagnostics
            ),
        )

    monkeypatch.setattr(module, "associate_temporal_observations", low_confidence)
    monkeypatch.setattr(
        module,
        "estimate_object_translation",
        lambda *args, **kwargs: ObjectMotionEstimate(
            kwargs["previous_object_to_world"],
            MotionDecision.REJECTED,
            0.0,
            0.1,
        ),
    )
    frame = _frame(2)

    result = runtime.process_frame(frame, (_observation(frame),))

    assert result.new_entity_ids == (old_id + 1,)
    assert result.reid_opportunity_count == result.reid_trigger_count == 0
    assert runtime.state.geometry.current(old_id).epoch_id == 0


@pytest.mark.parametrize("profile", (ExecutionProfile.A0, ExecutionProfile.A1))
def test_current_runtime_rejects_adapter_profiles(profile: ExecutionProfile) -> None:
    from src.oviv2.temporal_runtime import TemporalCurrentRuntime

    with pytest.raises(ValueError, match="A2.*A3.*A4"):
        TemporalCurrentRuntime("scene", _config(profile), _tracker_config())


@pytest.mark.parametrize("profile", (ExecutionProfile.A2, ExecutionProfile.A3))
def test_translation_profiles_never_call_icp_runtime_path(
    monkeypatch: pytest.MonkeyPatch, profile: ExecutionProfile
) -> None:
    import src.oviv2.temporal_runtime as module

    runtime = _runtime(_config(profile))
    _confirm(runtime)
    weight_before = float(runtime.state.entities[0].submap.weights.sum())
    calls = 0
    original = module.estimate_object_translation

    def capture(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "estimate_object_translation", capture)
    monkeypatch.setattr(
        module,
        "estimate_object_motion",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ICP path")),
    )
    frame = _frame(2, depth=1.2)

    result = runtime.process_frame(frame, (_observation(frame, centroid_z=1.2),))

    assert result.new_entity_ids == ()
    assert calls == 1
    assert float(runtime.state.entities[0].submap.weights.sum()) > weight_before


def test_a2_translation_failure_rolls_back_state_identity_and_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module

    runtime = _runtime(_config(ExecutionProfile.A2))
    _confirm(runtime)
    before = runtime.state
    before_dump = before.canonical_dump()
    monkeypatch.setattr(
        module,
        "estimate_object_translation",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("translation")),
    )
    frame = _frame(2)

    with pytest.raises(RuntimeError, match="translation"):
        runtime.process_frame(frame, (_observation(frame),))

    assert runtime.state is before
    assert runtime.state.canonical_dump() == before_dump


def test_a4_runtime_routes_icp_accepted_and_fallback_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_geometry import ObjectMotionEstimate

    runtime = _runtime(_config(ExecutionProfile.A4))
    entity_id = _confirm(runtime)
    calls = 0

    def gated(*args, **kwargs):
        nonlocal calls
        calls += 1
        pose = np.array(kwargs["previous_object_to_world"], copy=True)
        pose[2, 3] = 1.1 if calls == 1 else 1.2
        return ObjectMotionEstimate(pose, calls == 1, 0.9, 0.05)

    monkeypatch.setattr(module, "estimate_object_motion", gated)
    monkeypatch.setattr(
        module,
        "estimate_object_translation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("translation path")),
    )
    for frame_id, depth in ((2, 1.1), (3, 1.2)):
        frame = _frame(frame_id, depth=depth)
        result = runtime.process_frame(
            frame, (_observation(frame, centroid_z=depth),)
        )

    assert calls == 2
    assert result.active_entity_ids == (entity_id,)
    assert runtime.state.entities[0].object_to_world[2, 3] == 1.2


def test_translation_confidence_uses_verifiable_geometry_residual() -> None:
    from src.oviv2.temporal_geometry import MotionDecision, ObjectMotionEstimate
    from src.oviv2.temporal_runtime import _motion_confidence

    config = _config(ExecutionProfile.A2)
    runtime = _runtime(config)
    _confirm(runtime)
    entity = runtime.state.entities[0]
    pose = np.array(entity.object_to_world, copy=True)
    pose[0, 3] += 0.2
    motion = ObjectMotionEstimate(
        pose, MotionDecision.TRANSLATION_ACCEPTED, 0.0,
        config.geometry.maximum_icp_rmse_m,
    )
    aligned = entity.submap.world_points(pose)
    moderate = np.array(aligned, copy=True)
    moderate[:, 0] += config.motion.maximum_translation_residual_m / 4.0
    poor = np.array(aligned, copy=True)
    poor[:, 0] += config.motion.maximum_translation_residual_m * 2.0

    high = _motion_confidence(motion, config, entity.submap, aligned)
    middle = _motion_confidence(motion, config, entity.submap, moderate)
    low = _motion_confidence(motion, config, entity.submap, poor)

    assert high == 1.0
    assert 0.0 < middle < high
    assert low == 0.0
    below_threshold = np.array(aligned, copy=True)
    below_threshold[:, 0] += config.motion.maximum_translation_residual_m / 2.0
    assert _motion_confidence(
        motion, config, entity.submap, below_threshold
    ) == 0.0
    assert _motion_confidence(
        motion, config, entity.submap, np.empty((0, 3), dtype=np.float64)
    ) == 0.0


def test_chunked_chamfer_matches_exact_and_bounds_query_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module

    rng = np.random.default_rng(7)
    left = rng.normal(size=(19, 3))
    right = rng.normal(size=(23, 3))
    exact_squared = np.sum(
        (left[:, None, :] - right[None, :, :]) ** 2, axis=2
    )
    exact = np.sqrt(
        0.5
        * (
            np.min(exact_squared, axis=1).mean()
            + np.min(exact_squared, axis=0).mean()
        )
    )
    batch_sizes: list[int] = []
    original_tree = module.cKDTree

    class CapturingTree:
        def __init__(self, values):
            self._tree = original_tree(values)

        def query(self, query, *args, **kwargs):
            batch_sizes.append(np.asarray(query).shape[0])
            return self._tree.query(query, *args, **kwargs)

    monkeypatch.setattr(module, "cKDTree", CapturingTree)
    actual = module._symmetric_nearest_residual(left, right)
    large = rng.normal(size=(10_000, 3))
    assert np.isfinite(module._symmetric_nearest_residual(large, large[::-1]))

    assert actual == pytest.approx(exact, rel=1e-12, abs=1e-12)
    assert module._symmetric_nearest_residual(
        np.array([[0.0, 0.0, 0.0]]), np.array([[3.0, 0.0, 0.0]])
    ) == 3.0
    with pytest.raises(ValueError, match="finite nonempty"):
        module._symmetric_nearest_residual(
            np.empty((0, 3)), np.ones((1, 3))
        )
    with pytest.raises(ValueError, match="finite nonempty"):
        module._symmetric_nearest_residual(
            np.array([[np.nan, 0.0, 0.0]]), np.ones((1, 3))
        )
    assert max(batch_sizes) <= module._NEAREST_QUERY_CHUNK_SIZE
    assert module._NEAREST_QUERY_CHUNK_SIZE < 10_000


def test_frame_diagnostics_partition_icp_and_track_ledger_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_geometry import MotionDecision, ObjectMotionEstimate

    runtime = _runtime()
    entity_id = _confirm(runtime)
    staged = runtime.process_frame(_frame(2, depth=2.0), ())
    assert staged.diagnostics.ledger_stage_count == 1
    assert staged.diagnostics.ledger_commit_count == 0
    returned = _frame(3)
    monkeypatch.setattr(
        module,
        "estimate_object_motion",
        lambda *args, **kwargs: ObjectMotionEstimate(
            kwargs["previous_object_to_world"],
            MotionDecision.ICP_ACCEPTED,
            0.8,
            0.01,
        ),
    )

    result = runtime.process_frame(returned, (_observation(returned),))

    assert result.active_entity_ids == (entity_id,)
    assert result.diagnostics.icp_opportunity_count == 1
    assert result.diagnostics.icp_accept_count == 1
    assert result.diagnostics.icp_reject_count == 0
    assert result.diagnostics.ledger_reclaim_count == 1
    assert runtime.state.diagnostics.last_frame == result.diagnostics
    assert runtime.state.diagnostics.ledger_stage_count == 1
    assert runtime.state.diagnostics.ledger_reclaim_count == 1


@pytest.mark.parametrize("profile", (ExecutionProfile.A2, ExecutionProfile.A3))
def test_translation_profiles_exclude_and_retain_dormant_entities(
    monkeypatch: pytest.MonkeyPatch, profile: ExecutionProfile
) -> None:
    import src.oviv2.temporal_runtime as module

    runtime = _runtime(_config(profile))
    old_id = _confirm(runtime)
    runtime.process_frame(_frame(2, depth=2.0), ())
    assert runtime.process_frame(_frame(3, depth=2.0), ()).dormant_entity_ids == (old_id,)
    original = module.associate_temporal_observations
    target_ids: list[tuple[int, ...]] = []

    def capture(observations, targets, config, **kwargs):
        target_ids.append(tuple(target.entity_id for target in targets))
        return original(observations, targets, config, **kwargs)

    monkeypatch.setattr(module, "associate_temporal_observations", capture)
    new_ids: list[int] = []
    for frame_id in (4, 5):
        frame = _frame(frame_id)
        result = runtime.process_frame(frame, (_observation(frame),))
        new_ids.extend(result.new_entity_ids)

    assert all(old_id not in ids for ids in target_ids)
    assert old_id in tuple(entity.lifecycle.entity_id for entity in runtime.state.entities)
    assert result.reactivated_entity_ids == ()
    assert tuple(new_ids) == (old_id + 1,)


@pytest.mark.parametrize("profile", (ExecutionProfile.A2, ExecutionProfile.A3))
def test_translation_profiles_reclaim_geometry_but_keep_dormant_identity_at_capacity(
    profile: ExecutionProfile,
) -> None:
    runtime = _runtime(_config(profile, maximum_entities=1))
    old_id = _confirm(runtime)
    runtime.process_frame(_frame(2, depth=2.0), ())
    assert runtime.process_frame(_frame(3, depth=2.0), ()).dormant_entity_ids == (old_id,)

    new_ids = []
    for frame_id in (4, 5):
        frame = _frame(frame_id, depth=1.8)
        result = runtime.process_frame(
            frame, (_observation(frame, centroid_z=1.8, semantic_id=2),)
        )
        new_ids.extend(result.new_entity_ids)

    assert tuple(new_ids) == (old_id + 1,)
    assert tuple(entity.lifecycle.entity_id for entity in runtime.state.entities) == (old_id + 1,)
    assert runtime.state.identities.get(old_id) is not None


def test_a2_skips_masked_background_and_reports_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_background import TemporalBackgroundVolume

    runtime = _runtime(_config(ExecutionProfile.A2))
    monkeypatch.setattr(
        module,
        "build_background_depth",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("masked background")),
    )
    monkeypatch.setattr(
        TemporalBackgroundVolume,
        "_integrate_owned",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("background integration")),
    )
    frame = _frame(0, timestamp=0.0)

    result = runtime.process_frame(frame, (_observation(frame),))

    assert result.background_blocks_touched == 0
    assert runtime.state.background.active_block_count == 0


def test_a2_background_is_isolated_across_revisions() -> None:
    runtime = _runtime(_config(ExecutionProfile.A2))
    previous = runtime.state
    previous_background = previous._owned_background()
    previous_dump = previous.canonical_dump()
    frame = _frame(0, timestamp=0.0)

    runtime.process_frame(frame, (_observation(frame),))

    current_background = runtime.state._owned_background()
    assert current_background is not previous_background
    current_background._last_blocks_touched = 7
    assert previous.canonical_dump() == previous_dump


def test_runtime_finalizes_geometry_once_per_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.oviv2.temporal_state import TemporalGeometryState

    runtime = _runtime(_config(ExecutionProfile.A2))
    _confirm(runtime)
    calls = 0
    original = TemporalGeometryState.__post_init__

    def counted(self):
        nonlocal calls
        calls += 1
        original(self)

    monkeypatch.setattr(TemporalGeometryState, "__post_init__", counted)
    frame = _frame(2)
    runtime.process_frame(frame, (_observation(frame),))

    assert calls == 1


def test_a2_never_constructs_clones_or_stages_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_background import TemporalBackgroundVolume
    from src.oviv2.temporal_background_ledger import ReversibleBackgroundLedger

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden A2 background path")

    monkeypatch.setattr(ReversibleBackgroundLedger, "__init__", forbidden)
    monkeypatch.setattr(ReversibleBackgroundLedger, "clone", forbidden)
    monkeypatch.setattr(ReversibleBackgroundLedger, "stage", forbidden)
    monkeypatch.setattr(module, "build_background_depth", forbidden)
    monkeypatch.setattr(TemporalBackgroundVolume, "_integrate_owned", forbidden)
    runtime = _runtime(_config(ExecutionProfile.A2))
    frame = _frame(0, timestamp=0.0)

    result = runtime.process_frame(frame, (_observation(frame),))

    assert result.background_blocks_touched == 0
    assert runtime.state.background_ledger is None


def test_a3_ledger_owns_only_each_entities_released_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.oviv2.temporal_background_ledger import (
        ReversibleBackgroundLedger,
    )
    from src.oviv2.temporal_lifecycle import TemporalEvidenceKind

    runtime = _runtime(_config(ExecutionProfile.A3, maximum_entities=2))
    for frame_id in (0, 1):
        frame = _frame(frame_id, timestamp=float(frame_id), depth=1.0)
        observations = (
            _pixel_observation(frame, 10 + frame_id, 1, 1, semantic_id=1, image_feature=np.array([1.0, 0.0])),
            _pixel_observation(frame, 20 + frame_id, 3, 3, semantic_id=2, image_feature=np.array([0.0, 1.0])),
        )
        runtime.process_frame(frame, observations)

    captured = []
    ledger_type = type(runtime.state.background_ledger)
    original = ledger_type.stage

    def capture(self, evidence):
        if evidence.kind is TemporalEvidenceKind.VISIBLE_ABSENT:
            captured.append(evidence)
        return original(self, evidence)

    monkeypatch.setattr(ledger_type, "stage", capture)
    runtime.process_frame(_frame(2, depth=2.0), ())

    assert len(captured) == 2
    positive_masks = tuple(evidence.depth_m > 0.0 for evidence in captured)
    assert all(int(mask.sum()) == 1 for mask in positive_masks)
    assert not np.any(positive_masks[0] & positive_masks[1])
    assert not any(mask[0, 0] for mask in positive_masks)
    first_keys = {item.block_key for item in captured[0].contributions}
    second_keys = {item.block_key for item in captured[1].contributions}
    assert first_keys.isdisjoint(second_keys)


def test_ledger_reuses_lifecycle_view_bins_and_commits_two_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_lifecycle import TemporalEvidenceKind

    base = _config(ExecutionProfile.A3)
    config = replace(
        base,
        background_ledger=replace(
            base.background_ledger,
            commit_support_frames=2,
            commit_distinct_view_bins=2,
            minimum_commit_frame_gap=1,
        ),
    )
    runtime = _runtime(config)
    captured = []
    original = module._stage_ledger_evidence

    def capture(ledger, evidence, block_keys=None):
        captured.append(evidence)
        return original(ledger, evidence, block_keys)

    monkeypatch.setattr(module, "_stage_ledger_evidence", capture)
    entity_id = 1
    for frame_id in (0, 1):
        frame = _frame(frame_id)
        result = runtime.process_frame(
            frame,
            (
                _pixel_observation(
                    frame,
                    10 + frame_id,
                    2,
                    3,
                    semantic_id=1,
                    image_feature=np.array([1.0, 0.0]),
                ),
            ),
        )
    assert result.new_entity_ids == (entity_id,)
    expected_bins = []
    visible_evidence = []
    for frame_id, camera_x in ((2, 0.20), (3, 0.30)):
        frame = _frame(frame_id, depth=2.0, camera_x=camera_x)
        centroid = tuple(runtime.state.entities[0].object_to_world[:3, 3])
        expected = module._view_bin(frame, centroid, config)
        expected_bins.append(expected)
        runtime.process_frame(frame, ())
        visible = [
            item
            for item in captured
            if item.entity_id == entity_id
            and item.kind is TemporalEvidenceKind.VISIBLE_ABSENT
            and item.frame_id == frame_id
        ]
        assert len(visible) == 1
        visible_evidence.append(visible[0])
        assert visible[0].view_bin == expected
        assert expected in runtime.state.lifecycle_beliefs[0].absence_view_bins

    assert len(set(expected_bins)) == 2
    assert tuple(
        item.block_key for item in visible_evidence[0].contributions
    ) == tuple(item.block_key for item in visible_evidence[1].contributions)
    assert runtime.state.background_ledger.committed_record_count > 0
    assert runtime.state.background.active_block_count > 0
    assert all(
        item.view_bin is None
        for item in captured
        if item.kind is not TemporalEvidenceKind.VISIBLE_ABSENT
    )


def test_a3_uses_masked_background_without_icp_or_dormant_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module

    runtime = _runtime(_config(ExecutionProfile.A3))
    calls = 0
    original = module.build_background_depth

    def capture(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "build_background_depth", capture)
    frame = _frame(0, timestamp=0.0)

    runtime.process_frame(frame, (_observation(frame),))

    assert calls == 1


def test_background_reveals_object_pixel_when_entity_retires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module

    original = module.build_background_depth
    center_depths: list[float] = []

    def capture(frame, observations, protected, config):
        result = original(frame, observations, protected, config)
        if frame.frame_id >= 2:
            center_depths.append(float(result.depth_m[2, 2]))
        return result

    runtime = _runtime()
    _confirm(runtime)
    monkeypatch.setattr(module, "build_background_depth", capture)
    runtime.process_frame(_frame(2, depth=2.0), ())
    result = runtime.process_frame(_frame(3, depth=2.0), ())
    assert result.dormant_entity_ids == (1,)
    assert center_depths == [2.0, 2.0]


@pytest.mark.parametrize(
    "symbol",
    [
        "associate_temporal_observations",
        "advance_lifecycle",
        "build_background_depth",
    ],
)
def test_helper_exception_rolls_back_state_identity_and_value(
    monkeypatch: pytest.MonkeyPatch, symbol: str
) -> None:
    import src.oviv2.temporal_runtime as module

    runtime = _runtime()
    _confirm(runtime)
    before = runtime.state
    before_dump = before.canonical_dump()

    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError(symbol)

    monkeypatch.setattr(module, symbol, fail)
    frame = _frame(2)
    with pytest.raises(RuntimeError, match=symbol):
        runtime.process_frame(frame, (_observation(frame),))
    assert runtime.state is before
    assert runtime.state.canonical_dump() == before_dump


def test_publication_exception_rolls_back_state() -> None:
    from src.oviv2.temporal_runtime import TemporalCurrentRuntime

    class FailingRuntime(TemporalCurrentRuntime):
        fail = False

        def _before_publish(self, next_state):
            if self.fail:
                raise RuntimeError("publish")

    runtime = FailingRuntime("scene", _config(), _tracker_config())
    _confirm(runtime)
    before = runtime.state
    before_dump = before.canonical_dump()
    runtime.fail = True
    frame = _frame(2)
    with pytest.raises(RuntimeError, match="publish"):
        runtime.process_frame(frame, (_observation(frame),))
    assert runtime.state is before
    assert runtime.state.canonical_dump() == before_dump


@pytest.mark.parametrize("boundary", ("proposal", "export"))
def test_proposal_and_export_exceptions_deeply_roll_back_state_and_inputs(
    monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    import src.oviv2.temporal_runtime as module

    runtime = _runtime()
    _confirm(runtime)
    frame = _frame(2)
    observation = _observation(frame)
    observations = (observation,)
    dense = _dense_semantics(frame)
    proposal = _proposal_evidence(frame, dense, observations)
    before_state = runtime.state
    before_dump = before_state.canonical_dump()
    frame_fingerprint = (
        frame.rgb.tobytes(), frame.depth.tobytes(), frame.pose.tobytes()
    )
    observation_fingerprint = (
        observation.mask.tobytes(), observation.image_feature.tobytes()
    )
    proposal_fingerprint = (
        proposal.depth_m.tobytes(), proposal.current_xyz.tobytes(),
        proposal.semantic_support.tobytes(),
    )

    def fail(*args, **kwargs):
        raise RuntimeError(boundary)

    monkeypatch.setattr(
        module,
        "recover_temporal_proposals" if boundary == "proposal" else "TemporalExportBatch",
        fail,
    )
    with pytest.raises(RuntimeError, match=boundary):
        runtime.process_frame(
            frame, observations, dense, proposal_evidence=proposal
        )

    assert runtime.state is before_state
    assert runtime.state.canonical_dump() == before_dump
    assert (frame.rgb.tobytes(), frame.depth.tobytes(), frame.pose.tobytes()) == frame_fingerprint
    assert (observation.mask.tobytes(), observation.image_feature.tobytes()) == observation_fingerprint
    assert (
        proposal.depth_m.tobytes(), proposal.current_xyz.tobytes(),
        proposal.semantic_support.tobytes(),
    ) == proposal_fingerprint


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("depth_m", "depth"),
        ("current_xyz", "XYZ"),
        ("semantic_support", "semantic support"),
        ("semantic_provenance_hash", "provenance"),
    ),
)
def test_proposal_evidence_must_bind_validated_current_inputs(
    field: str, message: str
) -> None:
    runtime = _runtime()
    frame = _frame(0, timestamp=0.0)
    dense = _dense_semantics(frame)
    evidence = _proposal_evidence(frame, dense)
    if field == "depth_m":
        value = np.array(evidence.depth_m, copy=True)
        value[0, 0] += 1.0
    elif field == "current_xyz":
        value = np.array(evidence.current_xyz, copy=True)
        value[0, 0, 0] += 1.0
    elif field == "semantic_support":
        value = np.array(evidence.semantic_support, copy=True)
        value[0, 0] = False
    else:
        value = "f" * 64
    tampered = replace(evidence, **{field: value})

    with pytest.raises(ValueError, match=message):
        runtime.process_frame(frame, (), dense, proposal_evidence=tampered)

    assert runtime.state.revision == 0


def test_proposal_evidence_requires_current_dense_semantics() -> None:
    runtime = _runtime()
    frame = _frame(0, timestamp=0.0)
    evidence = _proposal_evidence(frame, _dense_semantics(frame))

    with pytest.raises(ValueError, match="dense semantics"):
        runtime.process_frame(frame, (), proposal_evidence=evidence)

    assert runtime.state.revision == 0


def test_proposal_segmentation_must_cover_current_object_masks() -> None:
    runtime = _runtime()
    frame = _frame(0, timestamp=0.0)
    observation = _observation(frame)
    dense = _dense_semantics(frame)
    evidence = _proposal_evidence(frame, dense)

    with pytest.raises(ValueError, match="segmentation"):
        runtime.process_frame(
            frame, (observation,), dense, proposal_evidence=evidence
        )

    assert runtime.state.revision == 0


def test_proposal_appearance_support_binds_current_features() -> None:
    from src.oviv2.temporal_proposals import ProposalRecoveryInput

    runtime = _runtime()
    frame = _frame(0, timestamp=0.0)
    observation = _observation(frame)
    dense = _dense_semantics(frame)
    base = _proposal_evidence(frame, dense, (observation,))
    evidence = ProposalRecoveryInput(
        **{
            **base.__dict__,
            "appearance_support": np.zeros(frame.depth.shape, dtype=bool),
            "appearance_source_frame_id": frame.frame_id,
            "appearance_model_id": observation.feature_model_id,
            "appearance_provenance_hash": "a" * 64,
        }
    )

    with pytest.raises(ValueError, match="appearance"):
        runtime.process_frame(
            frame, (observation,), dense, proposal_evidence=evidence
        )

    assert runtime.state.revision == 0


def test_proposal_appearance_support_accepts_authoritative_current_features() -> None:
    from src.oviv2.temporal_runtime import _proposal_appearance_provenance_hash

    runtime = _runtime()
    frame = _frame(0, timestamp=0.0)
    observation = _observation(frame)
    observations = (observation,)
    dense = _dense_semantics(frame)
    evidence = replace(
        _proposal_evidence(frame, dense, observations),
        appearance_support=observation.mask,
        appearance_source_frame_id=frame.frame_id,
        appearance_model_id=observation.feature_model_id,
        appearance_provenance_hash=_proposal_appearance_provenance_hash(
            frame, observations, observation.feature_model_id
        ),
    )

    result = runtime.process_frame(
        frame, observations, dense, proposal_evidence=evidence
    )

    assert result.revision == 1


def test_proposal_appearance_provenance_rejects_hash_tampering() -> None:
    from src.oviv2.temporal_runtime import _proposal_appearance_provenance_hash

    runtime = _runtime()
    frame = _frame(0, timestamp=0.0)
    observation = _observation(frame)
    observations = (observation,)
    dense = _dense_semantics(frame)
    evidence = replace(
        _proposal_evidence(frame, dense, observations),
        appearance_support=observation.mask,
        appearance_source_frame_id=frame.frame_id,
        appearance_model_id=observation.feature_model_id,
        appearance_provenance_hash=_proposal_appearance_provenance_hash(
            frame, observations, observation.feature_model_id
        ),
    )
    evidence = replace(evidence, appearance_provenance_hash="d" * 64)

    with pytest.raises(ValueError, match="appearance provenance"):
        runtime.process_frame(
            frame, observations, dense, proposal_evidence=evidence
        )

    assert runtime.state.revision == 0


def test_proposal_search_region_must_bind_retained_identity() -> None:
    from src.oviv2.temporal_proposals import ProjectedIdentitySearchRegion

    runtime = _runtime()
    _confirm(runtime)
    frame = _frame(2)
    dense = _dense_semantics(frame)
    mask = np.zeros(frame.depth.shape, dtype=bool)
    mask[2, 2] = True
    region = ProjectedIdentitySearchRegion(
        999, 1, mask, np.where(mask, 1.0, 0.0).astype(np.float32), "b" * 64
    )
    evidence = replace(
        _proposal_evidence(frame, dense), search_regions=(region,)
    )
    before = runtime.state
    before_dump = before.canonical_dump()

    with pytest.raises(ValueError, match="identity|search region"):
        runtime.process_frame(frame, (), dense, proposal_evidence=evidence)

    assert runtime.state is before
    assert runtime.state.canonical_dump() == before_dump


@pytest.mark.parametrize("field", ("source", "hash", "depth"))
def test_proposal_search_region_rejects_authoritative_field_tampering(
    field: str,
) -> None:
    from src.oviv2.temporal_runtime import _projected_identity_search_region

    runtime = _runtime()
    entity_id = _confirm(runtime)
    frame = _frame(2)
    dense = _dense_semantics(frame)
    region = _projected_identity_search_region(
        frame, runtime.state, entity_id, runtime.config
    )
    if field == "source":
        region = replace(region, source_frame_id=0)
    elif field == "hash":
        region = replace(region, projection_provenance_hash="c" * 64)
    else:
        depth = np.array(region.expected_depth_m, copy=True)
        depth[region.mask] += np.float32(0.01)
        region = replace(region, expected_depth_m=depth)
    evidence = replace(
        _proposal_evidence(frame, dense), search_regions=(region,)
    )
    before = runtime.state
    before_dump = before.canonical_dump()

    with pytest.raises(ValueError, match="search region"):
        runtime.process_frame(frame, (), dense, proposal_evidence=evidence)

    assert runtime.state is before
    assert runtime.state.canonical_dump() == before_dump


def test_proposal_search_region_accepts_authoritative_projection() -> None:
    from src.oviv2.temporal_runtime import _projected_identity_search_region

    runtime = _runtime()
    entity_id = _confirm(runtime)
    frame = _frame(2)
    dense = _dense_semantics(frame)
    region = _projected_identity_search_region(
        frame, runtime.state, entity_id, runtime.config
    )
    evidence = replace(
        _proposal_evidence(frame, dense), search_regions=(region,)
    )

    result = runtime.process_frame(frame, (), dense, proposal_evidence=evidence)

    assert result.revision == 3


def test_proposal_xyz_binding_rejects_sub_tolerance_tampering() -> None:
    runtime = _runtime()
    frame = _frame(0, timestamp=0.0)
    dense = _dense_semantics(frame)
    evidence = _proposal_evidence(frame, dense)
    xyz = np.array(evidence.current_xyz, copy=True)
    xyz[0, 0, 0] += np.float32(5e-7)

    with pytest.raises(ValueError, match="XYZ"):
        runtime.process_frame(
            frame, (), dense, proposal_evidence=replace(evidence, current_xyz=xyz)
        )

    assert runtime.state.revision == 0


def test_ledger_exception_rolls_back_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    _confirm(runtime)
    before = runtime.state
    before_dump = before.canonical_dump()

    def fail(self, evidence):
        raise RuntimeError("ledger")

    monkeypatch.setattr(type(runtime.state.background_ledger), "stage", fail)
    frame = _frame(2)
    with pytest.raises(RuntimeError, match="ledger"):
        runtime.process_frame(frame, (_observation(frame),))
    assert runtime.state is before
    assert runtime.state.canonical_dump() == before_dump


def test_sparse_ledger_rebuild_exception_is_not_downgraded_to_staged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module

    runtime = _runtime()
    _confirm(runtime)
    runtime.process_frame(_frame(2, depth=2.0), ())
    before = runtime.state
    before_dump = before.canonical_dump()
    monkeypatch.setattr(
        module,
        "_rebuild_sparse_background_blocks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rebuild")),
    )

    with pytest.raises(RuntimeError, match="background ledger rejected"):
        runtime.process_frame(_frame(3, depth=2.0, camera_x=0.05), ())

    assert runtime.state is before
    assert runtime.state.canonical_dump() == before_dump


def test_sparse_background_volume_uses_stable_candidate_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect
    import re
    import src.oviv2.temporal_runtime as module

    source = inspect.getsource(module)
    assert re.search(r"candidate_block_keys\s*=", source) is None
    assert re.search(r"del\s+\w+\.candidate_block_keys", source) is None
    frame = _frame(2, depth=2.0)
    depth = np.zeros_like(frame.depth)
    depth[2, 2] = frame.depth[2, 2]
    volume = module._SparseBackgroundVolume(_config().geometry)
    calls = 0
    original = module._sparse_background_block_keys

    def capture(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_sparse_background_block_keys", capture)
    touched = volume.candidate_block_keys(frame, depth)
    trial = volume.trial_integrate_blocks(frame, depth, touched)

    assert calls >= 2
    assert trial.active_block_count > 0
    invalid = ((999, 999, 999),)
    with pytest.raises(ValueError, match="touched"):
        volume.trial_integrate_blocks(frame, depth, invalid)


def test_sparse_background_rebuild_is_deterministic() -> None:
    import src.oviv2.temporal_runtime as module

    frame = _frame(2, depth=2.0)
    depth = np.zeros_like(frame.depth)
    depth[2, 2] = frame.depth[2, 2]
    probe = module._SparseBackgroundVolume(_config().geometry)
    key = probe.candidate_block_keys(frame, depth)[0]
    observations = (((1, 0, 2), key, frame, depth),)

    left = module._rebuild_sparse_background_blocks(
        _config().geometry, observations
    )
    right = module._rebuild_sparse_background_blocks(
        _config().geometry, observations
    )

    assert left.canonical_block_state() == right.canonical_block_state()


def test_sparse_block_key_scan_uses_bounded_pixel_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module

    width = module._SPARSE_PIXEL_CHUNK_SIZE * 3 + 17
    frame = Frame(
        frame_id=0,
        timestamp=0.0,
        source_frame_id=100,
        rgb=np.zeros((1, width, 3), dtype=np.uint8),
        depth=np.full((1, width), 2.0, dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(4.0, 4.0, width / 2.0, 0.0, width, 1),
    )
    maximum_seen = 0
    original = module.np.flatnonzero

    def capture(values):
        nonlocal maximum_seen
        maximum_seen = max(maximum_seen, values.size)
        return original(values)

    monkeypatch.setattr(module.np, "flatnonzero", capture)
    module._sparse_background_block_keys(
        frame,
        frame.depth,
        replace(_config().geometry, background_block_count=100_000),
    )

    assert maximum_seen <= module._SPARSE_PIXEL_CHUNK_SIZE


def test_sparse_block_key_scan_fails_at_configured_capacity() -> None:
    import src.oviv2.temporal_runtime as module

    frame = _frame(0, depth=2.0)
    with pytest.raises(ValueError, match="background_block_count|capacity"):
        module._sparse_background_block_keys(
            frame,
            frame.depth,
            replace(_config().geometry, background_block_count=1),
        )


def test_sparse_rebuild_batches_all_blocks_for_one_native_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module

    frame = _frame(2, depth=2.0)
    depth = np.zeros_like(frame.depth)
    depth[2, 2] = frame.depth[2, 2]
    probe = module._SparseBackgroundVolume(_config().geometry)
    keys = probe.candidate_block_keys(frame, depth)
    expected = probe.trial_integrate_blocks(frame, depth, keys)
    observations = tuple(
        (((frame.frame_id, frame.source_frame_id, key)), key, frame, depth)
        for key in keys
    )
    integrate_calls = 0
    original_integrate = module._SparseBackgroundVolume._integrate_owned_blocks

    def counted(self, *args, **kwargs):
        nonlocal integrate_calls
        integrate_calls += 1
        return original_integrate(self, *args, **kwargs)

    monkeypatch.setattr(
        module._SparseBackgroundVolume, "_integrate_owned_blocks", counted
    )
    monkeypatch.setattr(
        module._SparseBackgroundVolume,
        "_clone",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sparse rebuild must not clone per block")
        ),
    )
    rebuilt = module._rebuild_sparse_background_blocks(
        _config().geometry, observations
    )

    assert integrate_calls == 1
    assert rebuilt.canonical_block_state() == expected.canonical_block_state()


def test_sparse_ledger_aggregates_one_depth_object_for_many_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_background_ledger as ledger_module
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_background_ledger import (
        BackgroundContribution,
        BackgroundLedgerEvidence,
    )
    from src.oviv2.temporal_config import TemporalBackgroundLedgerConfig
    from src.oviv2.temporal_lifecycle import TemporalEvidenceKind

    geometry = replace(_config().geometry, background_block_count=128)
    ledger = module._SparseBackgroundLedger(
        geometry, TemporalBackgroundLedgerConfig(128, 2, 2, 1, 8)
    )
    frame = _frame(2, depth=2.0)
    depth = np.zeros_like(frame.depth)
    depth[2, 2] = frame.depth[2, 2]
    keys = tuple((index, 0, 0) for index in range(100))
    evidence = BackgroundLedgerEvidence(
        entity_id=1,
        geometry_epoch=0,
        frame_id=frame.frame_id,
        timestamp=float(frame.timestamp),
        kind=TemporalEvidenceKind.VISIBLE_ABSENT,
        view_bin=1,
        contributions=tuple(BackgroundContribution(key) for key in keys),
        frame=frame,
        depth_m=depth,
    )
    records = tuple(
        ledger_module._Record(
            evidence.entity_id,
            evidence.geometry_epoch,
            evidence.frame_id,
            evidence.timestamp,
            evidence.view_bin,
            contribution,
            evidence.frame,
            evidence.depth_m,
            evidence._observation_canonical,
        )
        for contribution in evidence.contributions
    )
    allocations = 0
    original = ledger_module.np.zeros_like

    def counted(*args, **kwargs):
        nonlocal allocations
        allocations += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ledger_module.np, "zeros_like", counted)
    observations = ledger._aggregate_observations(
        {record.key: record for record in records}
    )

    assert allocations == 0
    assert len(observations) == 1
    assert observations[0].depth_m is evidence.depth_m
    assert observations[0].block_keys == keys


def test_sparse_ledger_scopes_view_bins_to_entity_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_background import TemporalBackgroundVolume
    from src.oviv2.temporal_background_ledger import (
        BackgroundContribution,
        BackgroundLedgerEvidence,
        LedgerDecision,
    )
    from src.oviv2.temporal_config import TemporalBackgroundLedgerConfig
    from src.oviv2.temporal_lifecycle import TemporalEvidenceKind

    config = _config(ExecutionProfile.A3)
    ledger = module._SparseBackgroundLedger(
        config.geometry, TemporalBackgroundLedgerConfig(128, 2, 2, 1, 8)
    )
    frame = _frame(2, depth=2.0)
    depth = np.zeros_like(frame.depth)
    depth[2, 2] = frame.depth[2, 2]
    keys = module._sparse_background_block_keys(frame, depth, config.geometry)

    def evidence(entity_id: int, view_bin: int):
        return BackgroundLedgerEvidence(
            entity_id=entity_id,
            geometry_epoch=0,
            frame_id=frame.frame_id,
            timestamp=float(frame.timestamp),
            kind=TemporalEvidenceKind.VISIBLE_ABSENT,
            view_bin=view_bin,
            contributions=tuple(BackgroundContribution(key) for key in keys),
            frame=frame,
            depth_m=depth,
        )

    candidate_calls = 0

    def native_candidate(*args, **kwargs):
        nonlocal candidate_calls
        candidate_calls += 1
        return keys

    monkeypatch.setattr(
        TemporalBackgroundVolume,
        "candidate_block_keys",
        native_candidate,
    )
    first = evidence(1, 3)
    assert ledger.stage(first) is LedgerDecision.STAGED
    assert ledger.stage(evidence(2, 7)) is LedgerDecision.STAGED
    assert candidate_calls == 2
    assert ledger.provisional_count == 2 * len(keys)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        ledger.stage(replace(first, view_bin=9))


def test_native_sparse_keys_match_explicit_integrated_blocks() -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_background import TemporalBackgroundVolume

    config = _config(ExecutionProfile.A3)
    frame = _frame(2, depth=2.0)
    native_volume = TemporalBackgroundVolume(config.geometry)
    native_keys = native_volume.candidate_block_keys(frame, frame.depth)
    assert native_keys
    declared = module._candidate_keys_with_sparse_fallback(
        native_volume, frame, frame.depth, config.geometry
    )
    rebuilt = module._SparseBackgroundVolume.preallocated(
        config.geometry, len(declared)
    )
    rebuilt.integrate_blocks_owned(frame, frame.depth, declared)
    active = {
        tuple(int(value) for value in row)
        for row in module._background_module._active_block_keys(rebuilt._volume)
    }

    assert declared == native_keys
    assert active == set(declared)
    assert rebuilt.last_blocks_touched == len(declared)
    assert rebuilt.canonical_block_state()


@pytest.mark.parametrize(
    "kind",
    (
        pytest.param("present", id="present"),
        pytest.param("occluded", id="occluded"),
    ),
)
def test_sparse_nonabsence_cancels_provisional_without_view_index(
    kind: str,
) -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_background_ledger import (
        BackgroundContribution,
        BackgroundLedgerEvidence,
        LedgerDecision,
    )
    from src.oviv2.temporal_config import TemporalBackgroundLedgerConfig
    from src.oviv2.temporal_lifecycle import TemporalEvidenceKind

    config = _config(ExecutionProfile.A3)
    ledger = module._SparseBackgroundLedger(
        config.geometry, TemporalBackgroundLedgerConfig(128, 2, 2, 1, 8)
    )
    frame = _frame(2, depth=2.0)
    depth = np.zeros_like(frame.depth)
    depth[2, 2] = frame.depth[2, 2]
    keys = module._sparse_background_block_keys(frame, depth, config.geometry)
    absent = BackgroundLedgerEvidence(
        1, 0, 2, 2.0, TemporalEvidenceKind.VISIBLE_ABSENT, 3,
        tuple(BackgroundContribution(key) for key in keys), frame, depth,
    )
    assert ledger.stage(absent) is LedgerDecision.STAGED
    decision = ledger.stage(
        BackgroundLedgerEvidence(
            1, 0, 3, 3.0, TemporalEvidenceKind(kind), None, ()
        )
    )

    assert decision is LedgerDecision.CANCELLED
    assert ledger.provisional_count == 0
    assert ledger._native_view_bins == {}


def test_ledger_rejection_is_staged_once_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    _confirm(runtime)
    before = runtime.state
    before_dump = before.canonical_dump()
    calls = 0

    def reject_then_stage(self, evidence):
        nonlocal calls
        calls += 1
        return (
            LedgerDecision.REJECTED_INTEGRATION
            if calls == 1 else LedgerDecision.STAGED
        )

    from src.oviv2.temporal_background_ledger import LedgerDecision

    ledger_type = type(runtime.state.background_ledger)
    monkeypatch.setattr(ledger_type, "stage", reject_then_stage)

    with pytest.raises(RuntimeError, match="background ledger rejected"):
        runtime.process_frame(_frame(2, depth=2.0), ())

    assert calls == 1
    assert runtime.state is before
    assert runtime.state.canonical_dump() == before_dump


def test_ledger_rejection_aborts_the_whole_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.oviv2.temporal_background_ledger import (
        LedgerDecision,
        ReversibleBackgroundLedger,
    )

    runtime = _runtime()
    _confirm(runtime)
    before = runtime.state
    before_dump = before.canonical_dump()
    monkeypatch.setattr(
        ReversibleBackgroundLedger,
        "stage",
        lambda *args, **kwargs: LedgerDecision.REJECTED_CAPACITY,
    )
    frame = _frame(2)

    with pytest.raises(RuntimeError, match="rejected"):
        runtime.process_frame(frame, (_observation(frame),))

    assert runtime.state is before
    assert runtime.state.canonical_dump() == before_dump


def test_runtime_background_uses_one_clone_per_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.oviv2.temporal_background import TemporalBackgroundVolume

    runtime = _runtime()
    _confirm(runtime)
    original = TemporalBackgroundVolume._clone
    calls = 0

    def counted(self, capacity):
        nonlocal calls
        calls += 1
        return original(self, capacity)

    monkeypatch.setattr(TemporalBackgroundVolume, "_clone", counted)
    frame = _frame(2)
    runtime.process_frame(frame, (_observation(frame),))
    assert calls == 2


def test_repeated_runs_have_equal_results_and_complete_canonical_state() -> None:
    left, right = _runtime(), _runtime()
    left_results, right_results = [], []
    for frame_id, depth, present in ((0, 1.0, True), (1, 1.0, True), (2, 1.3, True), (3, 0.7, False)):
        for runtime, results in ((left, left_results), (right, right_results)):
            frame = _frame(frame_id, timestamp=float(frame_id), depth=depth)
            observations = (_observation(frame, centroid_z=depth),) if present else ()
            results.append(runtime.process_frame(frame, observations))
    assert left_results == right_results
    assert left.state == right.state
    assert left.state.canonical_dump() == right.state.canonical_dump()


def test_inputs_are_strict_and_fail_before_mutating_state() -> None:
    runtime = _runtime()
    before = runtime.state
    frame = _frame(0, timestamp=0.0)
    with pytest.raises(TypeError, match="exact tuple"):
        runtime.process_frame(frame, [_observation(frame)])  # type: ignore[arg-type]
    assert runtime.state is before
    duplicated = (_observation(frame, 10), _observation(frame, 10))
    with pytest.raises(ValueError, match="unique"):
        runtime.process_frame(frame, duplicated)
    assert runtime.state is before


def test_dense_semantics_requires_matching_source_frame_and_image_shape() -> None:
    runtime = _runtime()
    frame = _frame(0, timestamp=0.0)
    dense = DenseSemanticFrame(
        cache_frame_id=0,
        source_frame_id=999,
        image_shape=(5, 5),
        sample_stride=1,
        class_count=1,
        class_ids=np.ones((5, 5, 1), dtype=np.int64),
        probabilities=np.ones((5, 5, 1), dtype=np.float32),
        entropy=np.zeros((5, 5), dtype=np.float32),
        margin=np.ones((5, 5), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="source_frame_id"):
        runtime.process_frame(frame, (), dense)
    assert runtime.state.revision == 0


def test_confirmed_observation_without_valid_depth_is_skipped_without_id_use() -> None:
    runtime = _runtime()
    first = _frame(0, timestamp=0.0, depth=0.0)
    runtime.process_frame(first, (_observation(first, centroid_z=1.0),))
    second = _frame(1, timestamp=1.0, depth=0.0)
    result = runtime.process_frame(second, (_observation(second, centroid_z=1.0),))
    assert result.new_entity_ids == ()
    assert runtime.state.entities == ()
    assert runtime.state.next_entity_id == 1


def test_entity_arrays_are_private_readonly_and_value_equal() -> None:
    from src.oviv2.temporal_state import TemporalEntityState

    runtime = _runtime()
    _confirm(runtime)
    entity = runtime.state.entities[0]
    clone = TemporalEntityState(
        lifecycle=entity.lifecycle,
        semantic_probabilities=entity.semantic_probabilities,
        image_prototype=entity.image_prototype,
        extent_xyz=entity.extent_xyz,
        object_to_world=entity.object_to_world,
        submap=entity.submap,
        first_seen_frame_id=entity.first_seen_frame_id,
        last_seen_frame_id=entity.last_seen_frame_id,
        feature_model_id=entity.feature_model_id,
    )
    assert clone == entity
    assert clone is not entity
    assert clone.object_to_world is not entity.object_to_world
    assert clone.image_prototype is not entity.image_prototype
    assert not clone.object_to_world.flags.writeable
    assert clone.image_prototype is not None and not clone.image_prototype.flags.writeable
    with pytest.raises(TypeError):
        hash(clone)


def test_published_tracker_access_is_a_defensive_snapshot() -> None:
    runtime = _runtime()
    entity_id = _confirm(runtime)
    before = runtime.state.canonical_dump()

    exposed = runtime.state.tracker
    exposed.update((), 2)
    exposed.tracks.clear()
    exposed.graph._nodes.clear()
    exposed._next_track_id = 999

    assert runtime.state.canonical_dump() == before
    frame = _frame(2)
    result = runtime.process_frame(frame, (_observation(frame),))
    assert result.active_entity_ids == (entity_id,)


def test_published_background_access_is_a_defensive_snapshot() -> None:
    from src.oviv2.temporal_background import TemporalBackgroundVolume

    runtime = _runtime()
    _confirm(runtime)
    before = runtime.state.canonical_dump()
    expected_touched = runtime.state.background.last_blocks_touched

    exposed = runtime.state.background
    exposed.trial_integrate(_frame(2), np.zeros((5, 5), dtype=np.float32))
    exposed._last_blocks_touched = 999
    exposed._volume = TemporalBackgroundVolume(_config().geometry)._volume

    assert runtime.state.background.last_blocks_touched == expected_touched
    assert runtime.state.canonical_dump() == before


def test_runtime_state_snapshots_constructor_inputs() -> None:
    from src.oviv2.temporal_background import TemporalBackgroundVolume
    from src.oviv2.temporal_state import TemporalRuntimeState
    from src.oviv2.tracking import LocalTracker

    tracker = LocalTracker(_tracker_config())
    background = TemporalBackgroundVolume(_config().geometry)
    state = TemporalRuntimeState(
        scene_id="scene",
        revision=0,
        last_frame_id=-1,
        last_timestamp=-1.0,
        next_entity_id=1,
        entities=(),
        background=background,
        tracker=tracker,
    )
    before = state.canonical_dump()

    tracker.update((), 0)
    tracker._next_track_id = 123
    background._last_blocks_touched = 456

    assert state.canonical_dump() == before


def test_repeated_mutable_state_accesses_do_not_share_snapshots() -> None:
    runtime = _runtime()
    _confirm(runtime)
    before = runtime.state.canonical_dump()

    first_tracker = runtime.state.tracker
    second_tracker = runtime.state.tracker
    first_background = runtime.state.background
    second_background = runtime.state.background
    assert first_tracker is not second_tracker
    assert first_background is not second_background

    first_tracker.tracks.clear()
    first_background._last_blocks_touched = 999
    assert second_tracker.tracks
    assert second_background.last_blocks_touched != 999
    assert runtime.state.canonical_dump() == before


def test_runtime_state_slots_copy_repr_and_pickle_contract() -> None:
    runtime = _runtime()
    _confirm(runtime)
    state = runtime.state
    with pytest.raises(TypeError):
        vars(state)
    assert not hasattr(state, "__dict__")
    assert copy.copy(state) is state
    assert copy.deepcopy(state) is state
    assert "TemporalRuntimeState" in repr(state)
    reflected = asdict(state)
    assert reflected["scene_id"] == "scene"
    reflected["tracker"].tracks.clear()
    assert state.canonical_dump() == runtime.state.canonical_dump()
    with pytest.raises(TypeError, match="pickle"):
        pickle.dumps(state)
    assert tuple(item.name for item in fields(type(state))) == (
        "scene_id", "revision", "last_frame_id", "last_timestamp",
        "next_entity_id", "entities", "background", "tracker",
        "identities", "geometry", "lifecycle_beliefs", "background_ledger",
        "export_tracker", "diagnostics",
    )


def test_legacy_entity_prototype_without_model_remains_constructible() -> None:
    from src.oviv2.temporal_state import TemporalEntityState

    runtime = _runtime()
    _confirm(runtime)
    entity = runtime.state.entities[0]
    legacy = TemporalEntityState(
        lifecycle=entity.lifecycle,
        semantic_probabilities=entity.semantic_probabilities,
        image_prototype=entity.image_prototype,
        extent_xyz=entity.extent_xyz,
        object_to_world=entity.object_to_world,
        submap=entity.submap,
        first_seen_frame_id=entity.first_seen_frame_id,
        last_seen_frame_id=entity.last_seen_frame_id,
    )
    assert legacy.image_prototype is not None
    assert legacy.feature_model_id is None
    with pytest.raises(ValueError, match="prototype"):
        replace(legacy, image_prototype=None, feature_model_id="model")


def test_dense_semantics_cache_axis_and_source_frame_contract() -> None:
    runtime = _runtime()
    frame = _frame(0, timestamp=0.0)
    arrays = dict(
        image_shape=(5, 5), sample_stride=1, class_count=1,
        class_ids=np.ones((5, 5, 1), dtype=np.int64),
        probabilities=np.ones((5, 5, 1), dtype=np.float32),
        entropy=np.zeros((5, 5), dtype=np.float32),
        margin=np.ones((5, 5), dtype=np.float32),
    )
    wrong_cache = DenseSemanticFrame(cache_frame_id=1, source_frame_id=100, **arrays)
    with pytest.raises(ValueError, match="cache_frame_id"):
        runtime.process_frame(frame, (), wrong_cache)
    wrong_source = DenseSemanticFrame(cache_frame_id=0, source_frame_id=101, **arrays)
    with pytest.raises(ValueError, match="source_frame_id"):
        runtime.process_frame(frame, (), wrong_source)


def test_first_frame_confirm_one_handles_timestamp_and_extreme_log_odds() -> None:
    from src.oviv2.temporal_runtime import TemporalCurrentRuntime

    base = _config()
    lifecycle = replace(
        base.lifecycle,
        initial_log_odds=1e308,
        present_log_likelihood=1e308,
        log_odds_limit=1e308,
    )
    runtime = TemporalCurrentRuntime(
        "scene", replace(base, lifecycle=lifecycle),
        LocalTrackerConfig(confirm_hits=1, max_age_frames=20),
    )
    frame = _frame(0, timestamp=0.0)
    result = runtime.process_frame(frame, (_observation(frame),))
    assert result.new_entity_ids == (1,)
    assert runtime.state.entities[0].lifecycle.existence_log_odds == 1e308


def test_frame_result_and_runtime_state_reject_cross_field_invariants() -> None:
    from src.oviv2.temporal_runtime import TemporalFrameResult

    with pytest.raises(ValueError, match="disjoint"):
        TemporalFrameResult(0, 1, (1,), (1,), (), (), 0)
    with pytest.raises(ValueError, match="subset"):
        TemporalFrameResult(0, 1, (), (), (1,), (), 0)
    result = TemporalFrameResult(0, 1, (), (), (), (), 0)
    with pytest.raises(RuntimeError, match="dual.*frame|mismatch"):
        replace(result, frame_id=1)
    runtime = _runtime()
    with pytest.raises(ValueError, match="revision"):
        replace(runtime.state, revision=1)


def test_extreme_longdouble_inputs_fail_without_runtime_warning() -> None:
    runtime = _runtime()
    frame = _frame(0, timestamp=0.0)
    pose = np.asarray(frame.pose, dtype=np.longdouble)
    pose[0, 3] = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)
    frame.pose = pose
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="float64 range"):
            runtime.process_frame(frame, ())


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"active_on_probability": np.nan}, "finite"),
        ({"decay_half_life_seconds": -1.0}, "positive"),
        ({"dormant_off_probability": 2.0}, r"\[0, 1\]"),
        ({"minimum_absent_streak": 0}, "positive"),
        ({"initial_log_odds": np.nan}, "finite"),
        ({"present_log_likelihood": np.nan}, "finite"),
        ({"minimum_absent_streak": True}, "integer"),
        ({"visibility_depth_tolerance_m": 0.05}, "at least"),
    ],
)
def test_runtime_rejects_invalid_temporal_config_eagerly(
    changes: dict[str, object], error: str
) -> None:
    config = _config()
    invalid = replace(config, lifecycle=replace(config.lifecycle, **changes))
    from src.oviv2.temporal_runtime import TemporalCurrentRuntime

    with pytest.raises((TypeError, ValueError), match=error):
        TemporalCurrentRuntime("scene", invalid, _tracker_config())


def test_runtime_normalizes_valid_direct_temporal_config() -> None:
    config = _config()
    runtime = _runtime(config)
    assert runtime.config == config
    assert runtime.config is not config


def test_capacity_rejects_new_entity_without_consuming_id_when_no_dormant() -> None:
    runtime = _runtime(_config(maximum_entities=1))
    first_id = _confirm(runtime)
    frame2 = _frame(2)
    runtime.process_frame(frame2, (_observation(frame2, 20, centroid_z=1.8, semantic_id=2),))
    frame3 = _frame(3)
    result = runtime.process_frame(
        frame3, (_observation(frame3, 21, centroid_z=1.8, semantic_id=2),)
    )
    assert result.new_entity_ids == ()
    assert tuple(item.lifecycle.entity_id for item in runtime.state.entities) == (first_id,)
    assert runtime.state.next_entity_id == first_id + 1


def test_geometry_capacity_reclaims_dormant_wrapper_but_keeps_identity() -> None:
    runtime = _runtime(_config(maximum_entities=1))
    old_id = _confirm(runtime)
    runtime.process_frame(_frame(2, depth=2.0), ())
    assert runtime.process_frame(_frame(3, depth=2.0), ()).dormant_entity_ids == (
        old_id,
    )
    first = _frame(4, depth=1.8)
    first_result = runtime.process_frame(
        first, (_observation(
            first, 40, centroid_z=1.8, semantic_id=2,
            image_feature=np.array([0.0, 1.0]),
        ),)
    )
    second = _frame(5, depth=1.8)
    result = runtime.process_frame(
        second, (_observation(
            second, 41, centroid_z=1.8, semantic_id=2,
            image_feature=np.array([0.0, 1.0]),
        ),)
    )
    assert first_result.new_entity_ids + result.new_entity_ids == (old_id + 1,)
    assert tuple(item.lifecycle.entity_id for item in runtime.state.entities) == (
        old_id + 1,
    )
    assert runtime.state.identities.get(old_id) is not None
    with pytest.raises(KeyError):
        runtime.state.geometry.current(old_id)


def test_full_identity_bank_does_not_reclaim_dormant_wrapper_for_new_id() -> None:
    base = _config(maximum_entities=1)
    runtime = _runtime(replace(
        base,
        identity=replace(base.identity, maximum_identities=1),
        geometry_epoch=replace(base.geometry_epoch, maximum_retained_epochs=4),
    ))
    old_id = _confirm(runtime)
    runtime.process_frame(_frame(2, depth=2.0), ())
    runtime.process_frame(_frame(3, depth=2.0), ())

    for frame_id in (4, 5):
        frame = _frame(frame_id, depth=1.8)
        result = runtime.process_frame(
            frame,
            (_observation(
                frame,
                40 + frame_id,
                centroid_z=1.8,
                semantic_id=2,
                image_feature=np.array([0.0, 1.0]),
            ),),
        )

    assert result.new_entity_ids == ()
    assert tuple(
        item.lifecycle.entity_id for item in runtime.state.entities
    ) == (old_id,)
    assert runtime.state.geometry.current(old_id).epoch_id == 0


def test_a4_reidentifies_bank_only_dormant_identity_with_new_epoch() -> None:
    runtime = _runtime(_config(maximum_entities=1))
    old_id = _confirm(runtime)
    runtime.process_frame(_frame(2, depth=2.0), ())
    runtime.process_frame(_frame(3, depth=2.0), ())
    for frame_id in (4, 5):
        frame = _frame(frame_id, depth=1.8)
        runtime.process_frame(
            frame, (_observation(
                frame, 40 + frame_id, centroid_z=1.8, semantic_id=2,
                image_feature=np.array([0.0, 1.0]),
            ),)
        )
    assert runtime.state.identities.get(old_id) is not None
    with pytest.raises(KeyError):
        runtime.state.geometry.current(old_id)

    runtime.process_frame(_frame(6, depth=2.4), ())
    runtime.process_frame(_frame(7, depth=2.4), ())
    for frame_id in (8, 9):
        frame = _frame(frame_id)
        result = runtime.process_frame(frame, (_observation(frame, 80 + frame_id),))

    assert result.reid_trigger_count == 1
    assert runtime.state.geometry.current(old_id).epoch_id == 1
    assert runtime.state.identities.get(old_id + 1) is not None


def test_dormant_identity_expiry_boundary_is_explicit_and_counted() -> None:
    base = _config()
    config = replace(
        base,
        identity=replace(base.identity, maximum_dormant_frames=1),
    )
    runtime = _runtime(config)
    entity_id = _confirm(runtime)
    runtime.process_frame(_frame(2, depth=2.0), ())
    runtime.process_frame(_frame(3, depth=2.0), ())

    assert runtime.state.identities.get(entity_id) is not None
    result = runtime.process_frame(_frame(4, depth=0.5), ())

    assert runtime.state.identities.get(entity_id) is None
    assert result.expired_identity_ids == (entity_id,)
    assert result.diagnostics.identity_expiry_count == 1
    assert runtime.state.diagnostics.identity_expiry_count == 1


def test_frame_result_tuples_are_sorted_and_revision_advances_on_empty_frame() -> None:
    runtime = _runtime()
    frame = _frame(0, timestamp=0.0, depth=0.0)
    result = runtime.process_frame(frame, ())
    assert result.frame_id == 0
    assert result.revision == 1
    assert result.background_blocks_touched == 0
    assert runtime.state.last_frame_id == 0
    assert runtime.state.revision == 1
    assert all(value == tuple(sorted(value)) for value in (
        result.active_entity_ids,
        result.dormant_entity_ids,
        result.new_entity_ids,
        result.reactivated_entity_ids,
    ))
