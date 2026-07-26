from __future__ import annotations

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.t1_exactness import shared_input_sha256
from src.oviv2.t1_exactness import cumulative_state_sha256, temporal_state_sha256
from src.oviv2.dual_readout import DualReadoutRuntime
from src.oviv2.runtime import Oviv2Runtime
from src.oviv2.temporal_runtime import TemporalCurrentRuntime
from tests.oviv2.test_dual_readout import _cumulative, _frame, _temporal_config
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
