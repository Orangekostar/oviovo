from __future__ import annotations

from types import MethodType

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.dense_projection import DenseSemanticConfig, DenseSemanticIntegrator
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


def test_standard_duplicate_frame_error_rolls_back_and_retry_is_exact() -> None:
    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    runtime = DualReadoutRuntime(cumulative, temporal)
    runtime.process_frame(_frame(0), ())
    state_identity = temporal.state
    nested_tracker_identity = object.__getattribute__(state_identity, "_tracker_state")
    before = (
        cumulative_state_sha256(cumulative),
        temporal_state_sha256(temporal),
    )
    with pytest.raises(ValueError, match="frame IDs must increase monotonically"):
        runtime.process_frame(_frame(0), ())
    assert temporal.state is state_identity
    assert object.__getattribute__(temporal.state, "_tracker_state") is nested_tracker_identity
    assert before == (
        cumulative_state_sha256(cumulative),
        temporal_state_sha256(temporal),
    )

    retry = runtime.process_frame(_frame(1), ())
    clean_cumulative = _cumulative()
    clean_temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    clean_runtime = DualReadoutRuntime(clean_cumulative, clean_temporal)
    clean_runtime.process_frame(_frame(0), ())
    clean = clean_runtime.process_frame(_frame(1), ())
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


def test_transaction_capture_does_not_allocate_geometry_by_capacity(monkeypatch) -> None:
    from src.oviv2.geometry import SparseTsdfVolume
    from src.oviv2.t1_exactness import (
        DualTransactionSnapshot,
        isolated_cumulative_runtime,
    )

    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )

    def reject_allocation(*args, **kwargs):
        raise AssertionError("transaction capture allocated TSDF geometry")

    monkeypatch.setattr(SparseTsdfVolume, "__init__", reject_allocation)
    snapshot = DualTransactionSnapshot.capture(cumulative, temporal)
    assert dict(snapshot.cumulative_attributes)["geometry"] is cumulative.geometry
    trial = isolated_cumulative_runtime(cumulative)
    assert trial.geometry is cumulative.geometry


def test_instance_cumulative_entry_override_fails_before_branch() -> None:
    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    called = False

    def poisoned(self, frame, observations, dense_semantics=None):
        nonlocal called
        called = True

    cumulative.process_frame = MethodType(poisoned, cumulative)
    with pytest.raises(TypeError, match="exact built-in|override"):
        DualReadoutRuntime(cumulative, temporal).process_frame(_frame(), ())
    assert called is False


def test_instance_entry_closure_fails_before_mutating_original_input() -> None:
    frame = _frame()
    called = False
    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )

    def poisoned(self, branch_frame, observations, dense_semantics=None):
        nonlocal called
        called = True
        frame.rgb[0, 0, 0] = 233
        raise RuntimeError("closure input mutation")

    cumulative.process_frame = MethodType(poisoned, cumulative)
    with pytest.raises(TypeError, match="exact built-in|untrusted|override"):
        DualReadoutRuntime(cumulative, temporal).process_frame(frame, ())
    assert called is False
    assert frame.rgb[0, 0, 0] == 0


def test_instance_helper_override_fails_before_branch() -> None:
    called = False
    cumulative = _cumulative()

    def poisoned(self, *args, **kwargs):
        nonlocal called
        called = True

    cumulative._apply_visibility_to = MethodType(poisoned, cumulative)
    with pytest.raises(TypeError, match="exact built-in|override"):
        DualReadoutRuntime(
            cumulative,
            TemporalCurrentRuntime(
                "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
            ),
        ).process_frame(_frame(), ())
    assert called is False


def test_class_helper_override_fails_before_branch(monkeypatch) -> None:
    called = False

    def poisoned(self, *args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(Oviv2Runtime, "_apply_visibility_to", poisoned)
    with pytest.raises(TypeError, match="class namespace.*modified"):
        DualReadoutRuntime(
            _cumulative(),
            TemporalCurrentRuntime(
                "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
            ),
        ).process_frame(_frame(), ())
    assert called is False


def test_class_entry_override_fails_before_branch(monkeypatch) -> None:
    called = False

    def poisoned(self, *args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(Oviv2Runtime, "process_frame", poisoned)
    with pytest.raises(TypeError, match="class namespace.*modified"):
        DualReadoutRuntime(
            _cumulative(),
            TemporalCurrentRuntime(
                "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
            ),
        )
    assert called is False


def test_class_publish_hook_override_fails_before_branch(monkeypatch) -> None:
    called = False

    def poisoned(self, *args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(TemporalCurrentRuntime, "_before_publish", poisoned)
    with pytest.raises(TypeError, match="class namespace.*modified"):
        DualReadoutRuntime(
            _cumulative(),
            TemporalCurrentRuntime(
                "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
            ),
        )
    assert called is False


def test_added_property_fails_before_descriptor_access(monkeypatch) -> None:
    frame = _frame()
    cumulative = _cumulative()
    original_geometry = cumulative.geometry
    called = False

    def get_geometry(self):
        nonlocal called
        called = True
        frame.rgb[0, 0, 0] = 233
        return original_geometry

    monkeypatch.setattr(Oviv2Runtime, "geometry", property(get_geometry), raising=False)
    with pytest.raises(TypeError, match="class namespace.*modified"):
        DualReadoutRuntime(
            cumulative,
            TemporalCurrentRuntime(
                "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
            ),
        )
    assert called is False
    assert frame.rgb[0, 0, 0] == 0


def test_data_descriptor_fails_before_descriptor_access(monkeypatch) -> None:
    frame = _frame()
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    original_state = temporal.state
    called = False

    class PoisonDescriptor:
        def __get__(self, instance, owner):
            nonlocal called
            called = True
            frame.rgb[0, 0, 0] = 233
            return original_state

        def __set__(self, instance, value):
            raise AssertionError("descriptor setter must not run")

    monkeypatch.setattr(
        TemporalCurrentRuntime, "state", PoisonDescriptor(), raising=False
    )
    with pytest.raises(TypeError, match="class namespace.*modified"):
        DualReadoutRuntime(_cumulative(), temporal)
    assert called is False
    assert frame.rgb[0, 0, 0] == 0


def test_added_getattribute_fails_before_instance_access(monkeypatch) -> None:
    frame = _frame()
    cumulative = _cumulative()
    called = False

    def poisoned(self, name):
        nonlocal called
        called = True
        frame.rgb[0, 0, 0] = 233
        return object.__getattribute__(self, name)

    monkeypatch.setattr(
        Oviv2Runtime, "__getattribute__", poisoned, raising=False
    )
    with pytest.raises(TypeError, match="class namespace.*modified"):
        DualReadoutRuntime(
            cumulative,
            TemporalCurrentRuntime(
                "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
            ),
        )
    assert called is False
    assert frame.rgb[0, 0, 0] == 0


def test_in_place_code_replacement_fails_before_branch(monkeypatch) -> None:
    import src.oviv2.runtime as runtime_module

    frame = _frame()
    function = Oviv2Runtime.process_frame
    original_code = function.__code__
    monkeypatch.setattr(runtime_module, "_T1_PROBE_CALLED", False, raising=False)
    monkeypatch.setattr(runtime_module, "_T1_PROBE_FRAME", frame, raising=False)

    def poisoned(self, frame, observations, dense_semantics=None):
        global _T1_PROBE_CALLED
        _T1_PROBE_CALLED = True
        _T1_PROBE_FRAME.rgb[0, 0, 0] = 233
        raise RuntimeError("in-place code replacement")

    try:
        function.__code__ = poisoned.__code__
        with pytest.raises(TypeError, match="function behavior.*modified"):
            DualReadoutRuntime(
                _cumulative(),
                TemporalCurrentRuntime(
                    "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
                ),
            ).process_frame(frame, ())
    finally:
        function.__code__ = original_code
    assert runtime_module._T1_PROBE_CALLED is False
    assert frame.rgb[0, 0, 0] == 0


def test_in_place_defaults_replacement_fails_before_branch() -> None:
    frame = _frame()
    function = Oviv2Runtime.process_frame
    original_defaults = function.__defaults__
    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    try:
        function.__defaults__ = (object(),)
        with pytest.raises(TypeError, match="function behavior.*modified"):
            DualReadoutRuntime(cumulative, temporal)
    finally:
        function.__defaults__ = original_defaults
    assert cumulative.revision == temporal.state.revision == 0
    assert frame.rgb[0, 0, 0] == 0


@pytest.mark.parametrize(
    ("nested", "field", "poison"),
    [
        (False, "missing_observation_policy", "in_place_poison"),
        (True, "block_count", 1),
    ],
)
def test_in_place_default_config_content_mutation_fails_before_branch(
    nested: bool, field: str, poison: object
) -> None:
    frame = _frame()
    defaults = Oviv2Runtime.__init__.__defaults__
    assert defaults is not None
    config = defaults[0]
    target = object.__getattribute__(config, "tsdf") if nested else config
    original = object.__getattribute__(target, field)
    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    try:
        object.__setattr__(target, field, poison)
        with pytest.raises(TypeError, match="function behavior.*modified"):
            DualReadoutRuntime(cumulative, temporal)
    finally:
        object.__setattr__(target, field, original)
    assert cumulative.revision == temporal.state.revision == 0
    assert frame.rgb[0, 0, 0] == 0


def test_in_place_kwdefaults_mutation_fails_before_branch() -> None:
    frame = _frame()
    function = TemporalCurrentRuntime.process_frame
    kwdefaults = function.__kwdefaults__
    assert kwdefaults is not None
    original = dict(kwdefaults)
    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    try:
        kwdefaults["proposal_evidence"] = object()
        with pytest.raises(TypeError, match="function behavior.*modified"):
            DualReadoutRuntime(cumulative, temporal)
    finally:
        kwdefaults.clear()
        kwdefaults.update(original)
    assert cumulative.revision == temporal.state.revision == 0
    assert frame.rgb[0, 0, 0] == 0


def test_reachable_helper_replacement_fails_before_branch(monkeypatch) -> None:
    import src.oviv2.runtime as runtime_module

    called = False
    cumulative = _cumulative()
    geometry = cumulative.geometry

    def poisoned(target, frame):
        nonlocal called
        called = True
        target._t1_poison_marker = object()
        raise RuntimeError("reachable helper replacement")

    monkeypatch.setattr(runtime_module, "_clone_geometry_for_frame", poisoned)
    with pytest.raises(TypeError, match="global binding.*modified"):
        DualReadoutRuntime(
            cumulative,
            TemporalCurrentRuntime(
                "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
            ),
        ).process_frame(_frame(), ())
    assert called is False
    assert not hasattr(geometry, "_t1_poison_marker")


def test_reachable_helper_code_replacement_fails_before_branch(monkeypatch) -> None:
    import src.oviv2.runtime as runtime_module

    cumulative = _cumulative()
    geometry = cumulative.geometry
    function = runtime_module._clone_geometry_for_frame
    original_code = function.__code__
    monkeypatch.setattr(runtime_module, "_T1_HELPER_CALLED", False, raising=False)

    def poisoned(target, frame):
        global _T1_HELPER_CALLED
        _T1_HELPER_CALLED = True
        target._t1_poison_marker = object()
        raise RuntimeError("reachable helper code replacement")

    try:
        function.__code__ = poisoned.__code__
        with pytest.raises(TypeError, match="function behavior.*modified"):
            DualReadoutRuntime(
                cumulative,
                TemporalCurrentRuntime(
                    "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
                ),
            ).process_frame(_frame(), ())
    finally:
        function.__code__ = original_code
    assert runtime_module._T1_HELPER_CALLED is False
    assert not hasattr(geometry, "_t1_poison_marker")


def test_reachable_helper_constant_replacement_fails_before_branch(
    monkeypatch,
) -> None:
    import src.oviv2.temporal_runtime as temporal_module

    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    monkeypatch.setattr(temporal_module, "_SPARSE_PIXEL_CHUNK_SIZE", 1)
    with pytest.raises(TypeError, match="global binding.*modified"):
        DualReadoutRuntime(cumulative, temporal).process_frame(_frame(), ())
    assert cumulative.revision == temporal.state.revision == 0


def test_cumulative_subclass_helper_override_fails_before_branch() -> None:
    called = False

    class InternalOverride(Oviv2Runtime):
        def _apply_visibility_to(self, *args, **kwargs):
            nonlocal called
            called = True

    cumulative = _cumulative(runtime_type=InternalOverride)
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    before = tuple(vars(cumulative).items())
    with pytest.raises(TypeError, match="exact built-in|override"):
        DualReadoutRuntime(cumulative, temporal).process_frame(_frame(), ())
    assert called is False
    assert tuple(vars(cumulative).items()) == before


def test_instance_temporal_entry_override_fails_before_branch() -> None:
    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    called = False

    def poisoned(self, frame, observations, dense_semantics=None):
        nonlocal called
        called = True

    temporal.process_frame = MethodType(poisoned, temporal)
    with pytest.raises(TypeError, match="exact built-in|override"):
        DualReadoutRuntime(cumulative, temporal).process_frame(_frame(), ())
    assert called is False


def test_temporal_publish_hook_subclass_fails_before_branch() -> None:
    called = False

    class PublishOverride(TemporalCurrentRuntime):
        def _before_publish(self, next_state):
            nonlocal called
            called = True

    cumulative = _cumulative()
    temporal = PublishOverride(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    state = temporal.state
    with pytest.raises(TypeError, match="exact built-in|override"):
        DualReadoutRuntime(cumulative, temporal).process_frame(_frame(), ())
    assert temporal.state is state
    assert called is False


def test_untrusted_custom_attribute_fails_before_branch() -> None:
    called = False

    class CustomInputAlias(Oviv2Runtime):
        def process_frame(self, frame, observations, dense_semantics=None):
            nonlocal called
            called = True

    frame = _frame()
    original = (frame.rgb, frame.rgb.tobytes(), frame.rgb.flags.writeable)
    cumulative = _cumulative(runtime_type=CustomInputAlias)
    cumulative.original_frame = frame
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    with pytest.raises(TypeError, match="exact built-in|override"):
        DualReadoutRuntime(cumulative, temporal).process_frame(frame, ())
    assert frame.rgb is original[0]
    assert frame.rgb.tobytes() == original[1]
    assert frame.rgb.flags.writeable is original[2]
    assert cumulative.original_frame is frame
    assert called is False


def test_exact_runtime_extra_attribute_fails_closed_before_branch() -> None:
    cumulative = _cumulative()
    cumulative.custom_extension = object()
    temporal = TemporalCurrentRuntime(
        "scene", _temporal_config(), LocalTrackerConfig(confirm_hits=2)
    )
    frame = _frame()
    before = (frame.rgb, frame.rgb.tobytes(), cumulative.geometry, temporal.state)
    with pytest.raises(TypeError, match="exact built-in|override"):
        DualReadoutRuntime(cumulative, temporal).process_frame(frame, ())
    assert frame.rgb is before[0] and frame.rgb.tobytes() == before[1]
    assert cumulative.geometry is before[2]
    assert temporal.state is before[3]
