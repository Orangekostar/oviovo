from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.dense_projection import DenseSemanticConfig, DenseSemanticIntegrator
from src.oviv2.t1_exactness import shared_input_sha256
from src.oviv2.t1_exactness import cumulative_state_sha256, temporal_state_sha256
from src.oviv2.dual_readout import DualReadoutRuntime
from src.oviv2.runtime import Oviv2Runtime
from src.oviv2.reference_readout import ReferenceCurrentReadout
from src.oviv2.temporal_config import ExecutionProfile
from src.oviv2.temporal_runtime import TemporalCurrentRuntime
from tests.oviv2.test_dense_projection import make_dense_frame
from tests.oviv2.test_dual_readout import _cumulative, _frame, _observation, _temporal_config
from src.oviv2.tracking import LocalTrackerConfig


def _inputs() -> tuple[Frame, tuple[FrameObservation, ...]]:
    frame = Frame(
        frame_id=0, timestamp=1.0,
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.float32), pose=np.eye(4),
        intrinsics=CameraIntrinsics(1.0, 1.0, 0.0, 0.0, 2, 2),
    )
    observation = FrameObservation(
        1, 0, 1.0, ObservationKind.OBJECT, "chair", 1, 0.9,
        np.ones((2, 2), dtype=bool), (0.0, 0.0, 2.0, 2.0),
        frozenset({(0, 0, 1)}), (0.0, 0.0, 1.0),
        (-0.1, -0.1, 0.9), (0.1, 0.1, 1.1),
        image_feature=np.asarray((1.0, 0.0), dtype=np.float32),
        feature_model_id="fixture", visible_pixel_count=4,
    )
    return frame, (observation,)


def test_shared_input_sha256_covers_arrays_and_rejects_nonfinite() -> None:
    frame, observations = _inputs()
    before = shared_input_sha256(frame, observations, None)
    frame.rgb[0, 0, 0] = 1
    assert shared_input_sha256(frame, observations, None) != before
    frame.depth[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        shared_input_sha256(frame, observations, None)


def test_temporal_failure_restores_nested_state_inputs_and_retry_exactness() -> None:
    fail_next = [True]

    class NestedFailureTemporal(TemporalCurrentRuntime):
        def process_frame(self, frame, observations, dense_semantics=None):
            if fail_next[0]:
                fail_next[0] = False
                nested_tracker = object.__getattribute__(self.state, "_tracker_state")
                nested_tracker._next_track_id = 999
                raise RuntimeError("injected nested failure")
            return super().process_frame(frame, observations, dense_semantics)

    cumulative = _cumulative()
    temporal = NestedFailureTemporal(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    runtime = DualReadoutRuntime(cumulative, temporal)
    frame = _frame()
    observations = ()
    state_identity = temporal.state
    nested_tracker_identity = object.__getattribute__(state_identity, "_tracker_state")
    before = (
        cumulative_state_sha256(cumulative),
        temporal_state_sha256(temporal),
        shared_input_sha256(frame, observations, None),
        frame.rgb.flags.writeable,
        frame.depth.flags.writeable,
        frame.pose.flags.writeable,
    )
    with pytest.raises(RuntimeError, match="injected nested failure"):
        runtime.process_frame(frame, observations)
    assert temporal.state is state_identity
    assert object.__getattribute__(temporal.state, "_tracker_state") is nested_tracker_identity
    assert nested_tracker_identity._next_track_id == 1
    assert before == (
        cumulative_state_sha256(cumulative),
        temporal_state_sha256(temporal),
        shared_input_sha256(frame, observations, None),
        frame.rgb.flags.writeable,
        frame.depth.flags.writeable,
        frame.pose.flags.writeable,
    )

    retry = runtime.process_frame(frame, observations)
    clean_cumulative = _cumulative()
    clean_temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    clean = DualReadoutRuntime(clean_cumulative, clean_temporal).process_frame(
        _frame(), observations
    )
    assert retry == clean
    assert cumulative_state_sha256(cumulative) == cumulative_state_sha256(clean_cumulative)
    assert temporal_state_sha256(temporal) == temporal_state_sha256(clean_temporal)


def test_cumulative_failure_restores_externally_held_nested_objects() -> None:
    class NestedFailureCumulative(Oviv2Runtime):
        def process_frame(self, frame, observations, dense_semantics=None):
            self.geometry.integrate(
                frame.depth, frame.rgb, frame.intrinsics.to_matrix(), frame.pose
            )
            for value in (
                self.evidence, self.ownership, self.tracker, self.registry,
                self.visibility,
            ):
                value.injected_transaction_marker = object()
            raise RuntimeError("injected cumulative failure")

    cumulative = _cumulative(runtime_type=NestedFailureCumulative)
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    external = tuple(
        getattr(cumulative, name)
        for name in ("geometry", "evidence", "ownership", "tracker", "registry", "visibility")
    )
    before = cumulative_state_sha256(cumulative)
    with pytest.raises(RuntimeError, match="injected cumulative failure"):
        DualReadoutRuntime(cumulative, temporal).process_frame(_frame(), ())
    restored = tuple(
        getattr(cumulative, name)
        for name in ("geometry", "evidence", "ownership", "tracker", "registry", "visibility")
    )
    assert all(left is right for left, right in zip(restored, external))
    assert all(not hasattr(value, "injected_transaction_marker") for value in external)
    assert cumulative.geometry.active_block_count == 0
    assert cumulative_state_sha256(cumulative) == before


def test_temporal_failure_restores_externally_held_public_state_objects() -> None:
    fail_next = [False]

    class PublicStateFailure(TemporalCurrentRuntime):
        def process_frame(self, frame, observations, dense_semantics=None):
            if fail_next[0]:
                object.__setattr__(self.state.geometry, "maximum_retained_epochs", 999)
                object.__setattr__(
                    self.state.lifecycle_beliefs[0], "existence_log_odds", 999.0
                )
                object.__setattr__(self.state.export_tracker, "entries", ())
                object.__setattr__(self.state.diagnostics, "processed_frame_count", 999)
                raise RuntimeError("injected public temporal failure")
            return super().process_frame(frame, observations, dense_semantics)

    cumulative = _cumulative()
    temporal = PublicStateFailure(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=1)
    )
    runtime = DualReadoutRuntime(cumulative, temporal)
    first_frame = _frame()
    runtime.process_frame(first_frame, (_observation(first_frame),))
    state = temporal.state
    external = (
        state.geometry,
        state.lifecycle_beliefs,
        state.lifecycle_beliefs[0],
        state.export_tracker,
        state.diagnostics,
    )
    before = temporal_state_sha256(temporal)
    fail_next[0] = True
    second_frame = _frame(1, timestamp=2.0)
    with pytest.raises(RuntimeError, match="injected public temporal failure"):
        runtime.process_frame(second_frame, ())
    assert temporal.state is state
    assert temporal.state.geometry is external[0]
    assert temporal.state.lifecycle_beliefs is external[1]
    assert temporal.state.lifecycle_beliefs[0] is external[2]
    assert temporal.state.export_tracker is external[3]
    assert temporal.state.diagnostics is external[4]
    assert temporal_state_sha256(temporal) == before


def test_failure_restores_all_shared_input_values_and_retry_is_clean() -> None:
    fail_next = [True]

    class InputFailureCumulative(Oviv2Runtime):
        def process_frame(self, frame, observations, dense_semantics=None):
            if fail_next[0]:
                fail_next[0] = False
                frame.frame_id = 99
                frame.timestamp = 99.0
                frame.source_frame_id = 99
                frame.intrinsics.fx = 99.0
                object.__setattr__(observations[0], "confidence", 0.1)
                object.__setattr__(observations[0], "mask", np.zeros((5, 5), dtype=bool))
                object.__setattr__(dense_semantics, "cache_frame_id", 99)
                object.__setattr__(
                    dense_semantics, "class_ids", np.full((5, 5, 1), 2, dtype=np.int64)
                )
                raise RuntimeError("injected shared input failure")
            return super().process_frame(frame, observations, None)

    class IgnoreDenseTemporal(TemporalCurrentRuntime):
        def process_frame(self, frame, observations, dense_semantics=None):
            return super().process_frame(frame, observations, None)

    frame = _frame()
    observations = (_observation(frame),)
    dense = make_dense_frame(image_shape=(5, 5), source_frame_id=0)
    identities = (
        frame,
        frame.intrinsics,
        frame.rgb,
        frame.depth,
        frame.pose,
        observations[0],
        observations[0].mask,
        dense,
        dense.class_ids,
    )
    before = shared_input_sha256(frame, observations, dense)
    cumulative = _cumulative(runtime_type=InputFailureCumulative)
    temporal = IgnoreDenseTemporal(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    runtime = DualReadoutRuntime(cumulative, temporal)
    with pytest.raises(RuntimeError, match="injected shared input failure"):
        runtime.process_frame(frame, observations, dense)
    assert shared_input_sha256(frame, observations, dense) == before
    assert frame.frame_id == 0 and frame.timestamp == 0.0 and frame.source_frame_id is None
    assert frame.intrinsics.fx == 4.0 and observations[0].confidence == 0.9
    assert dense.cache_frame_id == 2
    restored_identities = (
        frame,
        frame.intrinsics,
        frame.rgb,
        frame.depth,
        frame.pose,
        observations[0],
        observations[0].mask,
        dense,
        dense.class_ids,
    )
    assert all(left is right for left, right in zip(restored_identities, identities))

    retry = runtime.process_frame(frame, observations, dense)
    clean_cumulative = _cumulative()
    clean_temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    clean_frame = _frame()
    clean = DualReadoutRuntime(clean_cumulative, clean_temporal).process_frame(
        clean_frame, (_observation(clean_frame),)
    )
    assert retry == clean
    assert cumulative_state_sha256(cumulative) == cumulative_state_sha256(clean_cumulative)
    assert temporal_state_sha256(temporal) == temporal_state_sha256(clean_temporal)


def test_cumulative_hash_covers_dense_semantic_integrator() -> None:
    runtime = _cumulative()
    before = cumulative_state_sha256(runtime)
    runtime.dense_semantic_integrator = DenseSemanticIntegrator(DenseSemanticConfig(0.1))
    assert cumulative_state_sha256(runtime) != before


@pytest.mark.parametrize(
    "value",
    [
        type("Custom", (), {"canonical_dump": lambda self: ()})(),
        type("Mutable", (), {})(),
    ],
)
def test_shared_input_hash_rejects_unwhitelisted_custom_objects(value: object) -> None:
    frame, observations = _inputs()
    frame.source_frame_id = value  # type: ignore[assignment]
    with pytest.raises(TypeError, match="unsupported deterministic hash value"):
        shared_input_sha256(frame, observations, None)


def test_reference_failure_restores_nested_external_refs_and_retry_exactness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_temporal_config(), execution_profile=ExecutionProfile.A0)
    cumulative = _cumulative()
    reference = ReferenceCurrentReadout("scene", config)
    runtime = DualReadoutRuntime(cumulative, reference)
    for frame_id in (0, 1):
        frame = _frame(frame_id, timestamp=float(frame_id + 1))
        runtime.process_frame(
            frame,
            (replace(_observation(frame), observation_id=10 + frame_id),),
        )

    state = reference.state
    view = state.cumulative_view
    tracker = state.export_tracker
    entry = tracker[0]
    before = (
        cumulative_state_sha256(cumulative),
        temporal_state_sha256(reference),
        copy.deepcopy(view),
        copy.deepcopy(entry),
    )
    original = ReferenceCurrentReadout.process_cumulative_frame
    fail_next = [True]

    def fail_after_update(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if fail_next[0]:
            fail_next[0] = False
            object.__setattr__(view, "revision", 999)
            object.__setattr__(entry, "observation_count", 999)
            raise RuntimeError("injected reference nested failure")
        return result

    monkeypatch.setattr(
        ReferenceCurrentReadout, "process_cumulative_frame", fail_after_update
    )
    failed_frame = _frame(2, timestamp=3.0)
    failed_observation = replace(_observation(failed_frame), observation_id=12)
    with pytest.raises(RuntimeError, match="injected reference nested failure"):
        runtime.process_frame(failed_frame, (failed_observation,))

    assert reference.state is state
    assert reference.state.cumulative_view is view
    assert reference.state.export_tracker is tracker
    assert reference.state.export_tracker[0] is entry
    assert reference.state.cumulative_view == before[2]
    assert reference.state.export_tracker[0] == before[3]
    assert cumulative_state_sha256(cumulative) == before[0]
    assert temporal_state_sha256(reference) == before[1]

    retry = runtime.process_frame(failed_frame, (failed_observation,))
    clean_cumulative = _cumulative()
    clean_reference = ReferenceCurrentReadout("scene", config)
    clean_runtime = DualReadoutRuntime(clean_cumulative, clean_reference)
    for frame_id in (0, 1, 2):
        clean_frame = _frame(frame_id, timestamp=float(frame_id + 1))
        clean_result = clean_runtime.process_frame(
            clean_frame,
            (replace(_observation(clean_frame), observation_id=10 + frame_id),),
        )
    assert retry == clean_result
    assert cumulative_state_sha256(cumulative) == cumulative_state_sha256(
        clean_cumulative
    )
    assert temporal_state_sha256(reference) == temporal_state_sha256(clean_reference)


def test_failure_restores_initially_readonly_owning_array_and_retry_exactness() -> None:
    fail_next = [True]

    class ReadonlyInputFailure(Oviv2Runtime):
        def process_frame(self, frame, observations, dense_semantics=None):
            if fail_next[0]:
                fail_next[0] = False
                frame.rgb.flags.writeable = True
                frame.rgb[0, 0, 0] = 255
                raise RuntimeError("injected readonly input failure")
            return super().process_frame(frame, observations, dense_semantics)

    frame = _frame()
    frame.rgb.flags.writeable = False
    original_rgb = frame.rgb.tobytes()
    cumulative = _cumulative(runtime_type=ReadonlyInputFailure)
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    state = temporal.state
    cumulative_refs = tuple(
        getattr(cumulative, name)
        for name in (
            "geometry", "evidence", "ownership", "tracker", "registry", "visibility"
        )
    )
    before = (
        cumulative_state_sha256(cumulative),
        temporal_state_sha256(temporal),
    )
    runtime = DualReadoutRuntime(cumulative, temporal)
    with pytest.raises(RuntimeError, match="injected readonly input failure"):
        runtime.process_frame(frame, ())
    assert frame.rgb.tobytes() == original_rgb
    assert frame.rgb.flags.writeable is False
    assert temporal.state is state
    assert all(
        getattr(cumulative, name) is value
        for name, value in zip(
            ("geometry", "evidence", "ownership", "tracker", "registry", "visibility"),
            cumulative_refs,
            strict=True,
        )
    )
    assert cumulative_state_sha256(cumulative) == before[0]
    assert temporal_state_sha256(temporal) == before[1]

    retry = runtime.process_frame(frame, ())
    clean_cumulative = _cumulative()
    clean_temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    clean_frame = _frame()
    clean_frame.rgb.flags.writeable = False
    clean = DualReadoutRuntime(clean_cumulative, clean_temporal).process_frame(
        clean_frame, ()
    )
    assert retry == clean
    assert cumulative_state_sha256(cumulative) == cumulative_state_sha256(
        clean_cumulative
    )
    assert temporal_state_sha256(temporal) == temporal_state_sha256(clean_temporal)
