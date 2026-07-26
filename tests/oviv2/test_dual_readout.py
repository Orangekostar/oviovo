from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.evidence import EvidenceConfig
from src.oviv2.geometry import TsdfConfig
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.runtime import Oviv2Runtime, Oviv2RuntimeConfig, RuntimeFrameResult
from src.oviv2.temporal_config import (
    ExecutionProfile,
    TemporalAssociationConfig,
    TemporalGeometryConfig,
    TemporalLifecycleConfig,
    TemporalReadoutConfig,
)
from src.oviv2.temporal_runtime import TemporalCurrentRuntime, TemporalFrameResult
from src.oviv2.tracking import LocalTrackerConfig
from tests.oviv2.test_dense_projection import make_dense_frame


def _temporal_config() -> TemporalReadoutConfig:
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
        geometry=TemporalGeometryConfig(
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
        ),
    )


def _temporal(scene_id: str = "scene") -> TemporalCurrentRuntime:
    return TemporalCurrentRuntime(
        scene_id,
        _temporal_config(),
        LocalTrackerConfig(confirm_hits=2, min_voxel_overlap=0.0),
    )


def _cumulative(
    scene_id: str = "scene",
    runtime_type: type[Oviv2Runtime] = Oviv2Runtime,
) -> Oviv2Runtime:
    return runtime_type(
        scene_id,
        Oviv2RuntimeConfig(
            tsdf=TsdfConfig(block_count=64),
            evidence=EvidenceConfig(),
        ),
    )


def _frame(frame_id: int = 0, timestamp: float | None = None) -> Frame:
    return Frame(
        frame_id=frame_id,
        timestamp=float(frame_id if timestamp is None else timestamp),
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth=np.ones((5, 5), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(4.0, 4.0, 2.0, 2.0, 5, 5),
    )


def _observation(frame: Frame) -> FrameObservation:
    return FrameObservation(
        observation_id=10,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        kind=ObservationKind.OBJECT,
        label="chair",
        semantic_id=1,
        confidence=0.9,
        mask=np.ones((5, 5), dtype=bool),
        bbox_xyxy=(0.0, 0.0, 5.0, 5.0),
        voxel_keys=frozenset({(0, 0, 10)}),
        centroid_xyz=(0.0, 0.0, 1.0),
        bounds_min_xyz=(-0.1, -0.1, 0.9),
        bounds_max_xyz=(0.1, 0.1, 1.1),
        image_feature=np.asarray((1.0, 0.0), dtype=np.float32),
        feature_model_id="fixture",
        visible_pixel_count=25,
    )


def _result(frame_id: int = 0, revision: int = 1) -> RuntimeFrameResult:
    return RuntimeFrameResult(frame_id, revision, 0, 0, 0, ())


def _temporal_result(frame_id: int = 0, revision: int = 1) -> TemporalFrameResult:
    return TemporalFrameResult(frame_id, revision, (), (), (), (), 0)


def _identity_snapshot(value: object) -> dict[str, object]:
    return dict(value.__dict__)


def _assert_exact_identity_snapshot(value: object, expected: dict[str, object]) -> None:
    assert set(value.__dict__) == set(expected)
    assert all(value.__dict__[name] is item for name, item in expected.items())


def test_result_requires_exact_component_types_and_is_frozen() -> None:
    from src.oviv2.dual_readout import DualFrameResult

    cumulative = _result()
    temporal = _temporal_result()
    result = DualFrameResult(cumulative, temporal)
    assert result.cumulative is cumulative
    assert result.temporal is temporal
    assert result.export is temporal.export
    with pytest.raises(TypeError, match="cumulative"):
        DualFrameResult(object(), temporal)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="temporal"):
        DualFrameResult(cumulative, object())  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        result.cumulative = cumulative  # type: ignore[misc]


@pytest.mark.parametrize("argument", [object(), None, 1])
def test_constructor_rejects_wrong_runtime_types(argument: object) -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    with pytest.raises(TypeError):
        DualReadoutRuntime(argument, _temporal())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DualReadoutRuntime(_cumulative(), argument)  # type: ignore[arg-type]


def test_constructor_rejects_structural_reference_impostor() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    class Impostor:
        state = _temporal().state

        def process_cumulative_frame(self, *args: object, **kwargs: object) -> object:
            return object()

    with pytest.raises(TypeError, match="reference|temporal"):
        DualReadoutRuntime(_cumulative(), Impostor())  # type: ignore[arg-type]


def test_constructor_rejects_builtin_reference_with_extra_mutable_state() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime
    from src.oviv2.reference_readout import ReferenceCurrentReadout

    config = replace(_temporal_config(), execution_profile=ExecutionProfile.A0)
    reference = ReferenceCurrentReadout("scene", config)
    reference.extra = []  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="unexpected mutable state"):
        DualReadoutRuntime(_cumulative(), reference)


def test_constructor_fails_closed_for_scene_progress_and_processed_timestamp_mismatch() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    with pytest.raises(ValueError, match="scene"):
        DualReadoutRuntime(_cumulative("left"), _temporal("right"))

    cumulative = _cumulative()
    temporal = _temporal()
    cumulative.revision = 1
    with pytest.raises(ValueError, match="revision|progress"):
        DualReadoutRuntime(cumulative, temporal)

    cumulative = _cumulative()
    temporal = _temporal()
    cumulative.last_frame_id = 0
    with pytest.raises(ValueError, match="frame|progress"):
        DualReadoutRuntime(cumulative, temporal)

    cumulative = _cumulative()
    temporal = _temporal()
    cumulative.process_frame(_frame(0, 1.0), ())
    temporal.process_frame(_frame(0, 2.0), ())
    with pytest.raises(ValueError, match="timestamp|progress"):
        DualReadoutRuntime(cumulative, temporal)


def test_cumulative_runs_first_and_temporal_is_not_called_when_it_rejects() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    calls: list[str] = []
    error = ValueError("cumulative rejected")

    class RejectingCumulative(Oviv2Runtime):
        def process_frame(self, *args: object, **kwargs: object) -> RuntimeFrameResult:
            calls.append("cumulative")
            raise error

    class TemporalSpy(TemporalCurrentRuntime):
        def process_frame(self, *args: object, **kwargs: object) -> TemporalFrameResult:
            calls.append("temporal")
            return _temporal_result()

    runtime = DualReadoutRuntime(_cumulative(runtime_type=RejectingCumulative), TemporalSpy(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    ))
    with pytest.raises(ValueError) as caught:
        runtime.process_frame(_frame(), ())
    assert caught.value is error
    assert calls == ["cumulative"]


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, KeyboardInterrupt],
)
def test_cumulative_failure_restores_both_shallow_snapshots_and_skips_temporal(
    error_type: type[BaseException],
) -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    temporal_calls = 0
    error = error_type("cumulative mutation failure")

    class MutatingRejectingCumulative(Oviv2Runtime):
        def process_frame(self, *args, **kwargs):
            del self.visibility
            self.geometry = object()
            self.injected = object()
            raise error

    class TemporalSpy(TemporalCurrentRuntime):
        def process_frame(self, *args, **kwargs):
            nonlocal temporal_calls
            temporal_calls += 1
            return super().process_frame(*args, **kwargs)

    cumulative = _cumulative(runtime_type=MutatingRejectingCumulative)
    temporal = TemporalSpy(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    cumulative.marker = object()
    temporal.marker = object()
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)
    with pytest.raises(type(error)) as caught:
        DualReadoutRuntime(cumulative, temporal).process_frame(_frame(), ())
    assert caught.value is error
    assert temporal_calls == 0
    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)


def test_temporal_receives_exact_input_objects_without_content_changes() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    current_frame = _frame()
    observations = (_observation(current_frame),)
    observation_before = (
        observations[0].mask.tobytes(),
        observations[0].image_feature.tobytes(),
        observations[0].voxel_keys,
    )
    dense = make_dense_frame(
        image_shape=(5, 5),
        stride=5,
        source_frame_id=0,
        class_count=1,
        class_ids=[[[1]]],
        probabilities=[[[1.0]]],
    )
    seen: list[tuple[object, object, object]] = []

    class CumulativeSpy(Oviv2Runtime):
        def process_frame(self, frame, observations, dense_semantics=None):
            seen.append((frame, observations, dense_semantics))
            return super().process_frame(frame, observations, None)

    class TemporalSpy(TemporalCurrentRuntime):
        def process_frame(self, frame, observations, dense_semantics=None):
            seen.append((frame, observations, dense_semantics))
            return super().process_frame(frame, observations, None)

    cumulative = _cumulative(runtime_type=CumulativeSpy)
    temporal = TemporalSpy("scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2))
    result = DualReadoutRuntime(cumulative, temporal).process_frame(
        current_frame, observations, dense
    )
    assert seen == [
        (current_frame, observations, dense),
        (current_frame, observations, dense),
    ]
    assert seen[1][0] is current_frame
    assert seen[1][1] is observations
    assert seen[1][1][0] is observations[0]
    assert seen[1][2] is dense
    assert result.cumulative.frame_id == result.temporal.frame_id == 0
    assert current_frame.frame_id == 0
    assert observation_before == (
        observations[0].mask.tobytes(),
        observations[0].image_feature.tobytes(),
        observations[0].voxel_keys,
    )


def test_branch_inputs_are_read_only_and_writeability_is_restored() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    current_frame = _frame()
    observations = (_observation(current_frame),)
    flags = (current_frame.rgb.flags.writeable, current_frame.depth.flags.writeable)

    class ReadonlyCumulative(Oviv2Runtime):
        def process_frame(self, frame, observations, dense_semantics=None):
            assert not frame.rgb.flags.writeable
            assert not frame.depth.flags.writeable
            assert not frame.pose.flags.writeable
            assert not observations[0].mask.flags.writeable
            return super().process_frame(frame, observations, dense_semantics)

    DualReadoutRuntime(
        _cumulative(runtime_type=ReadonlyCumulative), _temporal()
    ).process_frame(current_frame, observations)
    assert (current_frame.rgb.flags.writeable, current_frame.depth.flags.writeable) == flags


def test_temporal_entry_failure_restores_both_shallow_snapshots() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    error = RuntimeError("temporal entry failure")

    class RejectingTemporal(TemporalCurrentRuntime):
        def process_frame(self, *args, **kwargs):
            raise error

    cumulative = _cumulative()
    temporal = RejectingTemporal(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)
    with pytest.raises(RuntimeError) as caught:
        DualReadoutRuntime(cumulative, temporal).process_frame(_frame(), ())
    assert caught.value is error
    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)


def test_temporal_failure_restores_complete_shallow_state_on_both_runtimes() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    error = RuntimeError("temporal failed after mutation")

    class MutatingCumulative(Oviv2Runtime):
        def process_frame(self, *args, **kwargs):
            result = super().process_frame(*args, **kwargs)
            del self.visibility
            self.geometry = object()
            self.injected = object()
            return result

    class MutatingTemporal(TemporalCurrentRuntime):
        def process_frame(self, *args, **kwargs):
            super().process_frame(*args, **kwargs)
            del self.config
            self.state = object()
            self.injected = object()
            raise error

    cumulative = _cumulative(runtime_type=MutatingCumulative)
    temporal = MutatingTemporal("scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2))
    cumulative.marker = object()
    temporal.marker = object()
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)

    with pytest.raises(RuntimeError) as caught:
        DualReadoutRuntime(cumulative, temporal).process_frame(_frame(), ())
    assert caught.value is error
    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)


@pytest.mark.parametrize("field", ["frame_id", "revision"])
def test_mismatched_results_roll_back_both_runtimes(field: str) -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    class MismatchingTemporal(TemporalCurrentRuntime):
        def process_frame(self, *args, **kwargs):
            result = super().process_frame(*args, **kwargs)
            return replace(result, **{field: getattr(result, field) + 1})

    cumulative = _cumulative()
    temporal = MismatchingTemporal("scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2))
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)
    with pytest.raises(RuntimeError, match="dual.*(frame|revision)|mismatch"):
        DualReadoutRuntime(cumulative, temporal).process_frame(_frame(), ())
    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)


def test_temporal_scene_drift_rolls_back_both_runtimes() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    class WrongSceneTemporal(TemporalCurrentRuntime):
        def process_frame(self, *args, **kwargs):
            result = super().process_frame(*args, **kwargs)
            self.state = replace(self.state, scene_id="wrong-scene")
            return result

    cumulative = _cumulative()
    temporal = WrongSceneTemporal(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)
    with pytest.raises(RuntimeError, match="dual.*scene|mismatch"):
        DualReadoutRuntime(cumulative, temporal).process_frame(_frame(), ())
    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)


def test_dual_result_construction_failure_rolls_back_both_runtimes(monkeypatch) -> None:
    import src.oviv2.dual_readout as module

    cumulative = _cumulative()
    temporal = _temporal()
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)
    error = RuntimeError("result construction failed")

    def fail(*args: object, **kwargs: object) -> object:
        raise error

    runtime = module.DualReadoutRuntime(cumulative, temporal)
    monkeypatch.setattr(module, "DualFrameResult", fail)
    with pytest.raises(RuntimeError) as caught:
        runtime.process_frame(_frame(), ())
    assert caught.value is error
    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)


def test_temporal_runtime_has_no_cumulative_component_reference() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    cumulative = _cumulative()
    temporal = _temporal()
    dual = DualReadoutRuntime(cumulative, temporal)
    forbidden = {
        id(cumulative), id(cumulative.geometry), id(cumulative.evidence),
        id(cumulative.ownership), id(cumulative.registry),
    }
    assert id(dual.cumulative) in forbidden
    assert id(dual.temporal) == id(temporal)
    assert all(id(value) not in forbidden for value in temporal.__dict__.values())
    assert all(
        id(getattr(temporal.state, name)) not in forbidden
        for name in temporal.state.__slots__
        if not name.startswith("_")
    )


def test_current_checkpoint_keeps_reference_and_temporal_geometry_native() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime
    from src.oviv2.reference_readout import CumulativeReadoutView, ReferenceCurrentReadout
    from src.oviv2.temporal_snapshot import TemporalCurrentSnapshot

    temporal = DualReadoutRuntime(_cumulative(), _temporal())
    temporal.process_frame(_frame(), ())
    assert isinstance(temporal.current_checkpoint(), TemporalCurrentSnapshot)

    config = replace(_temporal_config(), execution_profile=ExecutionProfile.A0)
    reference = DualReadoutRuntime(_cumulative(), ReferenceCurrentReadout("scene", config))
    assert isinstance(reference.current_checkpoint(), CumulativeReadoutView)


def test_reference_readout_captures_before_and_after_cumulative_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime
    from src.oviv2.reference_readout import ReferenceCurrentReadout

    calls: list[tuple[str, int]] = []

    class CumulativeSpy(Oviv2Runtime):
        def process_frame(self, *args, **kwargs):
            calls.append(("cumulative", self.revision))
            return super().process_frame(*args, **kwargs)

    original = ReferenceCurrentReadout.process_cumulative_frame

    def process(self, frame, *, before, after, cumulative_result):
        calls.extend((("before", before.revision), ("after", after.revision)))
        return original(
            self,
            frame,
            before=before,
            after=after,
            cumulative_result=cumulative_result,
        )

    monkeypatch.setattr(ReferenceCurrentReadout, "process_cumulative_frame", process)

    config = replace(_temporal_config(), execution_profile=ExecutionProfile.A0)
    reference = ReferenceCurrentReadout("scene", config)
    cumulative = _cumulative(runtime_type=CumulativeSpy)
    result = DualReadoutRuntime(cumulative, reference).process_frame(_frame(), ())

    assert calls == [("cumulative", 0), ("before", 0), ("after", 1)]
    assert result.temporal.revision == result.cumulative.revision == 1
    assert reference.state.cumulative_view is not None
    assert reference.state.cumulative_view.revision == 1


def test_reference_readout_captures_cumulative_only_once_per_frame(monkeypatch) -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime
    from src.oviv2.reference_readout import CumulativeReadoutView, ReferenceCurrentReadout

    original = CumulativeReadoutView.capture.__func__
    revisions: list[int] = []

    def capture(cls, cumulative):
        revisions.append(cumulative.revision)
        return original(cls, cumulative)

    monkeypatch.setattr(CumulativeReadoutView, "capture", classmethod(capture))
    config = replace(_temporal_config(), execution_profile=ExecutionProfile.A0)
    runtime = DualReadoutRuntime(_cumulative(), ReferenceCurrentReadout("scene", config))
    revisions.clear()

    runtime.process_frame(_frame(0, 1.0), ())
    runtime.process_frame(_frame(1, 2.0), ())

    assert revisions == [1, 2]


def test_reference_capture_reuses_registry_voxel_frozenset() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime
    from src.oviv2.reference_readout import ReferenceCurrentReadout

    config = replace(_temporal_config(), execution_profile=ExecutionProfile.A0)
    cumulative = _cumulative()
    reference = ReferenceCurrentReadout("scene", config)
    runtime = DualReadoutRuntime(cumulative, reference)
    first = _frame(0, 1.0)
    second = _frame(1, 2.0)
    runtime.process_frame(first, (_observation(first),))
    runtime.process_frame(second, (replace(_observation(second), observation_id=11),))

    entity_id = next(iter(cumulative.registry.entities))
    captured = reference.state.cumulative_view
    assert captured is not None
    assert captured.entities[0].voxel_keys is cumulative.registry.entities[entity_id].voxel_keys


def test_reference_readout_failure_restores_both_complete_shallow_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime
    from src.oviv2.reference_readout import ReferenceCurrentReadout

    error = RuntimeError("reference failed after mutation")

    config = replace(_temporal_config(), execution_profile=ExecutionProfile.A0)
    cumulative = _cumulative()
    reference = ReferenceCurrentReadout("scene", config)
    original = ReferenceCurrentReadout.process_cumulative_frame

    def reject(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.state = replace(self.state, last_timestamp=99.0)
        raise error

    monkeypatch.setattr(ReferenceCurrentReadout, "process_cumulative_frame", reject)
    runtime = DualReadoutRuntime(cumulative, reference)
    before_cumulative = _identity_snapshot(cumulative)
    before_reference = _identity_snapshot(reference)

    with pytest.raises(RuntimeError) as caught:
        runtime.process_frame(_frame(), ())

    assert caught.value is error
    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(reference, before_reference)
