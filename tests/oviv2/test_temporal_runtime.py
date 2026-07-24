from __future__ import annotations

import copy
import pickle
import warnings
from dataclasses import asdict, replace

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.temporal_config import (
    TemporalAssociationConfig,
    TemporalGeometryConfig,
    TemporalLifecycleConfig,
    TemporalReadoutConfig,
)
from src.oviv2.tracking import LocalTrackerConfig


def _config(**geometry_changes: object) -> TemporalReadoutConfig:
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
    assert center_depths == [0.0, 2.0]


@pytest.mark.parametrize(
    "symbol",
    [
        "associate_temporal_observations",
        "advance_lifecycle",
        "integrate_object_submap",
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


def test_background_integration_exception_rolls_back_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.oviv2.temporal_background import TemporalBackgroundVolume

    runtime = _runtime()
    _confirm(runtime)
    before = runtime.state
    before_dump = before.canonical_dump()

    def fail(self, depth, rgb, intrinsic, pose):
        self._last_blocks_touched = 999
        raise RuntimeError("background integration")

    monkeypatch.setattr(TemporalBackgroundVolume, "_integrate_owned", fail)
    frame = _frame(2)
    with pytest.raises(RuntimeError, match="background integration"):
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
    assert calls == 1


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
    frame = _frame(0, timestamp=-np.finfo(np.float64).max)
    result = runtime.process_frame(frame, (_observation(frame),))
    assert result.new_entity_ids == (1,)
    assert runtime.state.entities[0].lifecycle.existence_log_odds == 1e308


def test_frame_result_and_runtime_state_reject_cross_field_invariants() -> None:
    from src.oviv2.temporal_runtime import TemporalFrameResult

    with pytest.raises(ValueError, match="disjoint"):
        TemporalFrameResult(0, 1, (1,), (1,), (), (), 0)
    with pytest.raises(ValueError, match="subset"):
        TemporalFrameResult(0, 1, (), (), (1,), (), 0)
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


def test_capacity_evicts_dormant_entity_before_creating_new_one() -> None:
    runtime = _runtime(_config(maximum_entities=1))
    old_id = _confirm(runtime)
    runtime.process_frame(_frame(2, depth=2.0), ())
    assert runtime.process_frame(_frame(3, depth=2.0), ()).dormant_entity_ids == (
        old_id,
    )
    first = _frame(4, depth=1.8)
    first_result = runtime.process_frame(
        first, (_observation(first, 40, centroid_z=1.8, semantic_id=2),)
    )
    second = _frame(5, depth=1.8)
    result = runtime.process_frame(
        second, (_observation(second, 41, centroid_z=1.8, semantic_id=2),)
    )
    assert first_result.new_entity_ids + result.new_entity_ids == (old_id + 1,)
    assert tuple(item.lifecycle.entity_id for item in runtime.state.entities) == (
        old_id + 1,
    )


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
