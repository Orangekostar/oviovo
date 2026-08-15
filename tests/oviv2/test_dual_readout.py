from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.evidence import EvidenceConfig
from src.oviv2.dense_projection import DenseSemanticConfig
from src.oviv2.geometry import TsdfConfig
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.runtime import Oviv2Runtime, Oviv2RuntimeConfig, RuntimeFrameResult
from src.oviv2.temporal_config import (
    DiagnosticControl,
    ExecutionProfile,
    TemporalAssociationConfig,
    TemporalGeometryConfig,
    TemporalLifecycleConfig,
    TemporalProposalConfig,
    TemporalReadoutConfig,
)
from src.oviv2.dense_semantics import DenseSemanticFrame, DenseSemanticProvenance
from src.oviv2.temporal_runtime import TemporalCurrentRuntime, TemporalFrameResult
from src.oviv2.tracking import LocalTrackerConfig


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
    *,
    dense: bool = False,
) -> Oviv2Runtime:
    dense_config = DenseSemanticConfig(voxel_size_m=0.05) if dense else None
    provenance = (
        DenseSemanticProvenance(
            backend="radseg",
            source_commit="a" * 40,
            radio_commit="b" * 40,
            model_id="fixture/model",
            model_sha256="c" * 64,
            auxiliary_model_sha256="d" * 64,
            vocabulary_sha256="e" * 64,
            prompt_sha256="f" * 64,
            inference_config_sha256="1" * 64,
            cache_prefix_sha256="2" * 64,
            language_model_id="fixture/language-model",
            language_model_revision="3" * 40,
            language_model_sha256="4" * 64,
        )
        if dense
        else None
    )
    return runtime_type(
        scene_id,
        Oviv2RuntimeConfig(
            tsdf=TsdfConfig(block_count=64),
            evidence=EvidenceConfig(),
            dense_semantics=dense_config,
        ),
        dense_semantic_provenance=provenance,
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

    with pytest.raises(TypeError, match="exact built-in|override"):
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


@pytest.mark.parametrize(
    "profile", [ExecutionProfile.A0, ExecutionProfile.A1, ExecutionProfile.A2]
)
def test_second_commit_failure_rolls_back_both_and_retry_is_exact(
    monkeypatch, profile: ExecutionProfile
) -> None:
    import src.oviv2.dual_readout as module
    from src.oviv2.reference_readout import (
        LifecycleOverlayReadout,
        ReferenceCurrentReadout,
    )
    from src.oviv2.t1_exactness import cumulative_state_sha256, temporal_state_sha256

    def make_temporal():
        config = replace(_temporal_config(), execution_profile=profile)
        if profile is ExecutionProfile.A0:
            return ReferenceCurrentReadout("scene", config)
        if profile is ExecutionProfile.A1:
            return LifecycleOverlayReadout("scene", config)
        return TemporalCurrentRuntime(
            "scene", config, LocalTrackerConfig(confirm_hits=2)
        )

    cumulative = _cumulative()
    temporal = make_temporal()
    runtime = module.DualReadoutRuntime(cumulative, temporal)
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)
    original_commit = module.commit_runtime_state
    calls = 0

    def fail_second(target: object, trial: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second commit failed")
        original_commit(target, trial)

    monkeypatch.setattr(module, "commit_runtime_state", fail_second)
    with pytest.raises(RuntimeError, match="second commit failed"):
        runtime.process_frame(_frame(), ())
    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)

    retry = runtime.process_frame(_frame(), ())
    clean_cumulative = _cumulative()
    clean_temporal = make_temporal()
    clean = module.DualReadoutRuntime(clean_cumulative, clean_temporal).process_frame(
        _frame(), ()
    )
    assert retry == clean
    assert cumulative_state_sha256(cumulative) == cumulative_state_sha256(clean_cumulative)
    assert temporal_state_sha256(temporal) == temporal_state_sha256(clean_temporal)


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


@pytest.mark.parametrize(
    "profile", (ExecutionProfile.A2, ExecutionProfile.A3, ExecutionProfile.A4)
)
def test_fresh_temporal_current_checkpoint_fails_closed(profile: ExecutionProfile) -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    config = replace(_temporal_config(), execution_profile=profile)
    runtime = DualReadoutRuntime(
        _cumulative(),
        TemporalCurrentRuntime("scene", config, LocalTrackerConfig(confirm_hits=2)),
    )
    with pytest.raises(ValueError, match="unprocessed|frame|fresh"):
        runtime.current_checkpoint()


def test_reference_readout_captures_cumulative_only_once_per_frame() -> None:
    import sys

    from src.oviv2.dual_readout import DualReadoutRuntime
    from src.oviv2.reference_readout import CumulativeReadoutView, ReferenceCurrentReadout

    capture_code = CumulativeReadoutView.capture.__func__.__code__
    revisions: list[int] = []
    previous_profile = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is capture_code:
            revisions.append(frame.f_locals["runtime"].revision)
        if previous_profile is not None:
            previous_profile(frame, event, arg)

    sys.setprofile(profile)
    try:
        config = replace(_temporal_config(), execution_profile=ExecutionProfile.A0)
        runtime = DualReadoutRuntime(
            _cumulative(), ReferenceCurrentReadout("scene", config)
        )
        revisions.clear()

        runtime.process_frame(_frame(0, 1.0), ())
        runtime.process_frame(_frame(1, 2.0), ())
    finally:
        sys.setprofile(previous_profile)

    assert revisions == [1, 2]
    assert sys.getprofile() is previous_profile


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


@pytest.mark.parametrize("profile", (ExecutionProfile.A0, ExecutionProfile.A1))
def test_reference_readouts_reject_different_temporal_observations(
    profile: ExecutionProfile,
) -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime
    from src.oviv2.reference_readout import (
        LifecycleOverlayReadout,
        ReferenceCurrentReadout,
    )

    config = replace(_temporal_config(), execution_profile=profile)
    temporal = (
        ReferenceCurrentReadout("scene", config)
        if profile is ExecutionProfile.A0
        else LifecycleOverlayReadout("scene", config)
    )
    cumulative = _cumulative()
    runtime = DualReadoutRuntime(cumulative, temporal)
    frame = _frame()
    temporal_observations = (_observation(frame),)
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)
    before_mask = temporal_observations[0].mask.tobytes()

    with pytest.raises(ValueError, match="reference.*temporal observations"):
        runtime.process_frame(
            frame,
            (),
            temporal_observations=temporal_observations,
        )

    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)
    assert temporal_observations[0].mask.tobytes() == before_mask


@pytest.mark.parametrize("profile", (ExecutionProfile.A0, ExecutionProfile.A1))
def test_reference_readouts_accept_content_identical_temporal_observations(
    profile: ExecutionProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.dual_readout as module
    from src.oviv2.reference_readout import (
        LifecycleOverlayReadout,
        ReferenceCurrentReadout,
    )

    config = replace(_temporal_config(), execution_profile=profile)
    temporal = (
        ReferenceCurrentReadout("scene", config)
        if profile is ExecutionProfile.A0
        else LifecycleOverlayReadout("scene", config)
    )
    clone_calls = []
    original_clone = module.clone_shared_inputs

    def recording_clone(*args):
        result = original_clone(*args)
        clone_calls.append(result)
        return result

    monkeypatch.setattr(module, "clone_shared_inputs", recording_clone)
    runtime = module.DualReadoutRuntime(_cumulative(), temporal)
    frame = _frame()
    observations = (_observation(frame),)
    equivalent = (replace(observations[0]),)

    result = runtime.process_frame(
        frame,
        observations,
        temporal_observations=equivalent,
    )

    assert result.cumulative.observation_count == 1
    assert result.temporal.frame_id == frame.frame_id
    assert len(clone_calls) == 1


def test_temporal_branch_independently_clones_aliased_caller_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.dual_readout as module

    clone_calls = []
    original_clone = module.clone_shared_inputs

    def recording_clone(*args):
        result = original_clone(*args)
        clone_calls.append(result)
        return result

    monkeypatch.setattr(module, "clone_shared_inputs", recording_clone)
    frame = _frame()
    observations = (_observation(frame),)
    temporal = TemporalCurrentRuntime(
        "scene",
        replace(_temporal_config(), execution_profile=ExecutionProfile.A2),
        LocalTrackerConfig(confirm_hits=1, min_voxel_overlap=0.0),
    )

    module.DualReadoutRuntime(_cumulative(), temporal).process_frame(
        frame,
        observations,
        temporal_observations=observations,
    )

    assert len(clone_calls) == 2
    cumulative_inputs, temporal_inputs = clone_calls
    assert cumulative_inputs[0] is not temporal_inputs[0]
    assert cumulative_inputs[1] is not temporal_inputs[1]
    assert cumulative_inputs[1][0] is not temporal_inputs[1][0]
    assert not np.shares_memory(
        cumulative_inputs[0].rgb,
        temporal_inputs[0].rgb,
    )
    assert not np.shares_memory(
        cumulative_inputs[1][0].mask,
        temporal_inputs[1][0].mask,
    )


def test_process_frame_rejects_clone_with_modified_observation_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.dual_readout as module

    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene",
        replace(_temporal_config(), execution_profile=ExecutionProfile.A2),
        LocalTrackerConfig(confirm_hits=1, min_voxel_overlap=0.0),
    )
    runtime = module.DualReadoutRuntime(cumulative, temporal)
    frame = _frame()
    observations = (_observation(frame),)
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)
    original_clone = module.clone_shared_inputs

    def modified_clone(*args):
        cloned_frame, cloned_observations, cloned_dense = original_clone(*args)
        object.__setattr__(cloned_observations[0], "label", "tampered")
        return cloned_frame, cloned_observations, cloned_dense

    monkeypatch.setattr(module, "clone_shared_inputs", modified_clone)

    with pytest.raises(ValueError, match="clone|equivalent"):
        runtime.process_frame(frame, observations)

    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)
    assert observations[0].label == "chair"


def test_process_frame_rejects_clone_with_frame_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.dual_readout as module

    class CompatibleFrameSubclass(Frame):
        pass

    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene",
        replace(_temporal_config(), execution_profile=ExecutionProfile.A2),
        LocalTrackerConfig(confirm_hits=1, min_voxel_overlap=0.0),
    )
    runtime = module.DualReadoutRuntime(cumulative, temporal)
    frame = _frame()
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)
    original_clone = module.clone_shared_inputs

    def subclassed_clone(*args):
        cloned_frame, cloned_observations, cloned_dense = original_clone(*args)
        subclassed_frame = CompatibleFrameSubclass(**vars(cloned_frame))
        return subclassed_frame, cloned_observations, cloned_dense

    monkeypatch.setattr(module, "clone_shared_inputs", subclassed_clone)

    with pytest.raises(ValueError, match="clone|equivalent"):
        runtime.process_frame(frame, ())

    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)


def test_process_frame_rejects_clone_with_nested_scalar_type_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.dual_readout as module

    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene",
        replace(_temporal_config(), execution_profile=ExecutionProfile.A2),
        LocalTrackerConfig(confirm_hits=1, min_voxel_overlap=0.0),
    )
    runtime = module.DualReadoutRuntime(cumulative, temporal)
    frame = _frame()
    observation = replace(
        _observation(frame),
        bbox_xyxy=tuple(np.float64(value) for value in (0.0, 0.0, 5.0, 5.0)),
    )
    observations = (observation,)
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)
    original_clone = module.clone_shared_inputs

    def scalar_type_changed_clone(*args):
        cloned_frame, cloned_observations, cloned_dense = original_clone(*args)
        object.__setattr__(
            cloned_observations[0],
            "bbox_xyxy",
            tuple(float(value) for value in cloned_observations[0].bbox_xyxy),
        )
        return cloned_frame, cloned_observations, cloned_dense

    monkeypatch.setattr(module, "clone_shared_inputs", scalar_type_changed_clone)

    with pytest.raises(ValueError, match="clone|equivalent"):
        runtime.process_frame(frame, observations)

    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)


@pytest.mark.parametrize("profile", (ExecutionProfile.A0, ExecutionProfile.A1))
def test_reference_readouts_reject_nested_scalar_type_difference(
    profile: ExecutionProfile,
) -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime
    from src.oviv2.reference_readout import (
        LifecycleOverlayReadout,
        ReferenceCurrentReadout,
    )

    config = replace(_temporal_config(), execution_profile=profile)
    temporal = (
        ReferenceCurrentReadout("scene", config)
        if profile is ExecutionProfile.A0
        else LifecycleOverlayReadout("scene", config)
    )
    cumulative = _cumulative()
    runtime = DualReadoutRuntime(cumulative, temporal)
    frame = _frame()
    observations = (_observation(frame),)
    temporal_observations = (
        replace(
            observations[0],
            bbox_xyxy=tuple(
                np.float64(value) for value in observations[0].bbox_xyxy
            ),
        ),
    )
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)

    with pytest.raises(ValueError, match="reference.*temporal observations"):
        runtime.process_frame(
            frame,
            observations,
            temporal_observations=temporal_observations,
        )

    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)


def test_process_frame_detects_clone_side_effect_on_caller_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.dual_readout as module

    bbox = [0.0, 0.0, 5.0, 5.0]

    class MutatingVoxelKeys(frozenset):
        deepcopy_calls = 0

        def __deepcopy__(self, memo):
            del memo
            type(self).deepcopy_calls += 1
            if type(self).deepcopy_calls == 2:
                bbox.append(99.0)
            return type(self)(self)

    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene",
        replace(_temporal_config(), execution_profile=ExecutionProfile.A2),
        LocalTrackerConfig(confirm_hits=1, min_voxel_overlap=0.0),
    )
    runtime = module.DualReadoutRuntime(cumulative, temporal)
    frame = _frame()
    observation = replace(
        _observation(frame),
        bbox_xyxy=bbox,
        voxel_keys=MutatingVoxelKeys({(0, 0, 10)}),
    )
    observations = (observation,)
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)

    with pytest.raises(RuntimeError, match="original shared inputs were mutated"):
        runtime.process_frame(
            frame,
            observations,
            temporal_observations=observations,
        )

    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)
    assert bbox == [0.0, 0.0, 5.0, 5.0, 99.0]


@pytest.mark.parametrize("separate_temporal_inputs", (False, True))
def test_process_frame_uses_structural_witness_for_exact_builtin_inputs(
    monkeypatch: pytest.MonkeyPatch,
    separate_temporal_inputs: bool,
) -> None:
    import src.oviv2.dual_readout as module

    def reject_recursive_scan(*args: object) -> object:
        del args
        raise AssertionError("exact production inputs used recursive content scan")

    monkeypatch.setattr(module, "shared_input_sha256", reject_recursive_scan)
    monkeypatch.setattr(module, "_shared_input_exact_signature", reject_recursive_scan)
    frame = _frame()
    observations = (_observation(frame),)
    temporal = TemporalCurrentRuntime(
        "scene",
        replace(_temporal_config(), execution_profile=ExecutionProfile.A2),
        LocalTrackerConfig(confirm_hits=1, min_voxel_overlap=0.0),
    )

    module.DualReadoutRuntime(_cumulative(), temporal).process_frame(
        frame,
        observations,
        temporal_observations=(
            (replace(observations[0], observation_id=11),)
            if separate_temporal_inputs
            else None
        ),
    )


def test_structural_witness_rejects_unvalidated_voxel_key_payload() -> None:
    import src.oviv2.dual_readout as module

    frame = _frame()
    observation = replace(
        _observation(frame),
        voxel_keys=frozenset({object()}),  # type: ignore[arg-type]
    )

    assert not module._shared_inputs_support_structural_witness(
        frame,
        (observation,),
        None,
    )


def test_process_frame_rolls_back_without_recursive_digest_when_snapshot_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.dual_readout as module

    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene",
        replace(_temporal_config(), execution_profile=ExecutionProfile.A2),
        LocalTrackerConfig(confirm_hits=1, min_voxel_overlap=0.0),
    )
    runtime = module.DualReadoutRuntime(cumulative, temporal)
    frame = _frame()
    observations = (_observation(frame),)
    temporal_observations = (replace(observations[0], observation_id=11),)
    before_cumulative = _identity_snapshot(cumulative)
    before_temporal = _identity_snapshot(temporal)
    original_hash = module.shared_input_sha256
    original_snapshot = module.shared_input_snapshot
    hash_calls: list[tuple[object, ...]] = []

    def counted_hash(*args: object) -> str:
        hash_calls.append(args)
        return original_hash(*args)  # type: ignore[arg-type]

    class FailingSnapshot:
        def __init__(self, wrapped: object) -> None:
            self.wrapped = wrapped

        def assert_unchanged(self) -> None:
            self.wrapped.assert_unchanged()  # type: ignore[attr-defined]
            raise RuntimeError("forced caller snapshot failure")

    def snapshot_with_failure(*args):
        snapshot = original_snapshot(*args)
        return FailingSnapshot(snapshot) if args[1] is observations else snapshot

    monkeypatch.setattr(module, "shared_input_sha256", counted_hash)
    monkeypatch.setattr(module, "shared_input_snapshot", snapshot_with_failure)

    with pytest.raises(RuntimeError, match="original shared inputs were mutated"):
        runtime.process_frame(
            frame,
            observations,
            temporal_observations=temporal_observations,
        )

    assert hash_calls == []
    _assert_exact_identity_snapshot(cumulative, before_cumulative)
    _assert_exact_identity_snapshot(temporal, before_temporal)


def test_temporal_observation_failure_rolls_back_both_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import MethodType

    import src.oviv2.dual_readout as module
    from src.oviv2.t1_exactness import cumulative_state_sha256, temporal_state_sha256

    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene",
        replace(_temporal_config(), execution_profile=ExecutionProfile.A2),
        LocalTrackerConfig(confirm_hits=1, min_voxel_overlap=0.0),
    )
    runtime = module.DualReadoutRuntime(cumulative, temporal)
    frame = _frame()
    temporal_observations = (_observation(frame),)
    before_cumulative = cumulative_state_sha256(cumulative)
    before_temporal = temporal_state_sha256(temporal)
    before_mask = temporal_observations[0].mask.tobytes()
    original_isolate = module.isolated_temporal_runtime
    error = RuntimeError("temporal trial failed")

    def fail_after_processing(self, *args, **kwargs):
        TemporalCurrentRuntime.process_frame(self, *args, **kwargs)
        raise error

    def isolate_with_failure(value):
        trial = original_isolate(value)
        trial.process_frame = MethodType(fail_after_processing, trial)
        return trial

    monkeypatch.setattr(module, "isolated_temporal_runtime", isolate_with_failure)
    with pytest.raises(RuntimeError) as caught:
        runtime.process_frame(
            frame,
            (),
            temporal_observations=temporal_observations,
        )

    assert caught.value is error
    assert cumulative_state_sha256(cumulative) == before_cumulative
    assert temporal_state_sha256(temporal) == before_temporal
    assert temporal_observations[0].mask.tobytes() == before_mask


@pytest.mark.parametrize("raise_after_mutation", (False, True))
def test_temporal_trial_input_mutation_is_detected_on_success_and_exception(
    monkeypatch: pytest.MonkeyPatch,
    raise_after_mutation: bool,
) -> None:
    from types import MethodType

    import src.oviv2.dual_readout as module
    from src.oviv2.t1_exactness import cumulative_state_sha256, temporal_state_sha256

    cumulative = _cumulative()
    temporal = TemporalCurrentRuntime(
        "scene",
        replace(_temporal_config(), execution_profile=ExecutionProfile.A2),
        LocalTrackerConfig(confirm_hits=1, min_voxel_overlap=0.0),
    )
    runtime = module.DualReadoutRuntime(cumulative, temporal)
    frame = _frame()
    temporal_observations = (_observation(frame),)
    before_cumulative = cumulative_state_sha256(cumulative)
    before_temporal = temporal_state_sha256(temporal)
    original_isolate = module.isolated_temporal_runtime

    def mutate_trial_input(
        self,
        trial_frame,
        trial_observations,
        dense=None,
        *,
        proposal_evidence=None,
    ):
        result = TemporalCurrentRuntime.process_frame(
            self,
            trial_frame,
            trial_observations,
            dense,
            proposal_evidence=proposal_evidence,
        )
        object.__setattr__(trial_observations[0], "label", "tampered")
        if raise_after_mutation:
            raise RuntimeError("failed after mutation")
        return result

    def isolate_with_mutation(value):
        trial = original_isolate(value)
        trial.process_frame = MethodType(mutate_trial_input, trial)
        return trial

    monkeypatch.setattr(module, "isolated_temporal_runtime", isolate_with_mutation)
    with pytest.raises(RuntimeError, match="shared inputs|branch mutated isolated"):
        runtime.process_frame(
            frame,
            (),
            temporal_observations=temporal_observations,
        )

    assert cumulative_state_sha256(cumulative) == before_cumulative
    assert temporal_state_sha256(temporal) == before_temporal
    assert temporal_observations[0].label == "chair"


def test_process_frame_detects_trial_mutation_of_cloned_mutable_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import MethodType

    import src.oviv2.dual_readout as module
    from src.oviv2.t1_exactness import cumulative_state_sha256, temporal_state_sha256

    cumulative = _cumulative()
    temporal = _temporal()
    runtime = module.DualReadoutRuntime(cumulative, temporal)
    frame = _frame()
    bbox = [0.0, 0.0, 5.0, 5.0]
    observations = (replace(_observation(frame), bbox_xyxy=bbox),)
    before_cumulative = cumulative_state_sha256(cumulative)
    before_temporal = temporal_state_sha256(temporal)
    original_isolate = module.isolated_cumulative_runtime

    def mutate_trial_input(self, trial_frame, trial_observations, dense=None):
        result = Oviv2Runtime.process_frame(
            self, trial_frame, trial_observations, dense
        )
        trial_observations[0].bbox_xyxy.append(99.0)
        return result

    def isolate_with_mutation(value):
        trial = original_isolate(value)
        trial.process_frame = MethodType(mutate_trial_input, trial)
        return trial

    monkeypatch.setattr(module, "isolated_cumulative_runtime", isolate_with_mutation)

    with pytest.raises(RuntimeError, match="shared inputs|branch mutated isolated"):
        runtime.process_frame(frame, observations)

    assert cumulative_state_sha256(cumulative) == before_cumulative
    assert temporal_state_sha256(temporal) == before_temporal
    assert bbox == [0.0, 0.0, 5.0, 5.0]


def test_temporal_observations_is_keyword_only() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime

    runtime = DualReadoutRuntime(_cumulative(), _temporal())
    with pytest.raises(TypeError):
        runtime.process_frame(_frame(), (), None, ())  # type: ignore[misc]


def test_temporal_only_frame_matches_temporal_branch_and_leaves_cumulative_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.dual_readout as module
    from src.oviv2.t1_exactness import (
        cumulative_state_sha256,
        temporal_state_sha256,
    )

    standard = module.DualReadoutRuntime(_cumulative(), _temporal())
    accelerated = module.DualReadoutRuntime(_cumulative(), _temporal())
    frame = _frame()
    observations = (_observation(frame),)
    expected = standard.process_frame(
        frame, (), temporal_observations=observations
    ).temporal
    before_cumulative = cumulative_state_sha256(accelerated.cumulative)

    monkeypatch.setattr(
        module,
        "shared_input_sha256",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("temporal-only path hashed dual inputs")
        ),
    )
    observed = accelerated.process_temporal_only_frame(frame, observations)

    assert observed == expected
    assert temporal_state_sha256(accelerated.temporal) == temporal_state_sha256(
        standard.temporal
    )
    assert cumulative_state_sha256(accelerated.cumulative) == before_cumulative


def test_temporal_only_rejects_clone_with_modified_depth_content_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.dual_readout as module
    from src.oviv2.t1_exactness import (
        cumulative_state_sha256,
        temporal_state_sha256,
    )

    runtime = module.DualReadoutRuntime(_cumulative(), _temporal())
    frame = _frame()
    observations = (_observation(frame),)
    original_depth = frame.depth.tobytes()
    before_cumulative = cumulative_state_sha256(runtime.cumulative)
    before_temporal = temporal_state_sha256(runtime.temporal)
    original_clone = module.clone_shared_inputs

    def modified_clone(*args):
        cloned_frame, cloned_observations, cloned_dense = original_clone(*args)
        cloned_frame.depth[0, 0] = np.float32(2.0)
        return cloned_frame, cloned_observations, cloned_dense

    monkeypatch.setattr(module, "clone_shared_inputs", modified_clone)

    with pytest.raises(ValueError, match="clone|equivalent"):
        runtime.process_temporal_only_frame(frame, observations)

    assert cumulative_state_sha256(runtime.cumulative) == before_cumulative
    assert temporal_state_sha256(runtime.temporal) == before_temporal
    assert frame.depth.tobytes() == original_depth


def test_temporal_only_detects_clone_side_effect_on_caller_mutable_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.dual_readout as module
    from src.oviv2.t1_exactness import (
        cumulative_state_sha256,
        temporal_state_sha256,
    )

    bbox = [0.0, 0.0, 5.0, 5.0]

    class MutatingVoxelKeys(frozenset):
        def __deepcopy__(self, memo):
            del memo
            bbox.append(99.0)
            return type(self)(self)

    runtime = module.DualReadoutRuntime(_cumulative(), _temporal())
    frame = _frame()
    observations = (
        replace(
            _observation(frame),
            bbox_xyxy=bbox,
            voxel_keys=MutatingVoxelKeys({(0, 0, 10)}),
        ),
    )
    before_cumulative = cumulative_state_sha256(runtime.cumulative)
    before_temporal = temporal_state_sha256(runtime.temporal)

    with pytest.raises(RuntimeError, match="original.*inputs|caller inputs"):
        runtime.process_temporal_only_frame(frame, observations)

    assert cumulative_state_sha256(runtime.cumulative) == before_cumulative
    assert temporal_state_sha256(runtime.temporal) == before_temporal
    assert bbox == [0.0, 0.0, 5.0, 5.0, 99.0]


@pytest.mark.parametrize("temporal_only", (False, True))
def test_proposal_recovery_evidence_is_built_from_cloned_temporal_inputs_and_passed(
    monkeypatch: pytest.MonkeyPatch,
    temporal_only: bool,
) -> None:
    import src.oviv2.dual_readout as module

    config = replace(
        _temporal_config(), proposal=TemporalProposalConfig(1, 4, 0.25, 0.1)
    )
    temporal = TemporalCurrentRuntime(
        "scene", config, LocalTrackerConfig(confirm_hits=2, min_voxel_overlap=0.0)
    )
    runtime = module.DualReadoutRuntime(_cumulative(dense=True), temporal)
    built_from: list[tuple[Frame, tuple[FrameObservation, ...], DenseSemanticFrame]] = []
    original_observations: list[tuple[FrameObservation, ...]] = []
    original_builder = module.build_proposal_recovery_evidence

    def capture_builder(trial_frame, trial_observations, trial_dense, config, state):
        built_from.append((trial_frame, trial_observations, trial_dense))
        return original_builder(
            trial_frame, trial_observations, trial_dense, config, state
        )

    monkeypatch.setattr(module, "build_proposal_recovery_evidence", capture_builder)
    for frame_id in (0, 1):
        frame = _frame(frame_id)
        observations = (
            replace(_observation(frame), observation_id=10 + frame_id),
        )
        original_observations.append(observations)
        dense = _dense_semantics(frame)
        if temporal_only:
            runtime.process_temporal_only_frame(frame, observations, dense)
        else:
            runtime.process_frame(
                frame, (), dense, temporal_observations=observations
            )
    frame = _frame(2)
    frame = replace(frame, depth=np.full(frame.depth.shape, 0.5, dtype=np.float32))
    dense = _dense_semantics(frame)
    if temporal_only:
        result = runtime.process_temporal_only_frame(frame, (), dense)
    else:
        result = runtime.process_frame(frame, (), dense).temporal

    assert result.proposal_opportunity_count > 0
    assert len(built_from) == 3
    assert built_from[0][1][0] is not original_observations[0][0]
    assert built_from[-1][0] is not frame
    assert built_from[-1][2] is not dense


@pytest.mark.parametrize("temporal_only", (False, True))
def test_proposal_recovery_control_does_not_build_and_passes_none(
    monkeypatch: pytest.MonkeyPatch,
    temporal_only: bool,
) -> None:
    import src.oviv2.dual_readout as module

    config = replace(
        _temporal_config(),
        execution_profile=ExecutionProfile.A2,
        diagnostic_control=DiagnosticControl.A2_NO_PROPOSAL_RECOVERY,
    )
    temporal = TemporalCurrentRuntime(
        "scene", config, LocalTrackerConfig(confirm_hits=2, min_voxel_overlap=0.0)
    )
    runtime = module.DualReadoutRuntime(_cumulative(dense=True), temporal)
    frame = _frame()
    observations = (_observation(frame),)
    dense = _dense_semantics(frame)
    passed_evidence = []
    original_isolate = module.isolated_temporal_runtime
    original_process = TemporalCurrentRuntime.process_frame

    def capture_process(self, *args, **kwargs):
        assert "proposal_evidence" in kwargs
        passed_evidence.append(kwargs["proposal_evidence"])
        monkeypatch.setattr(
            TemporalCurrentRuntime, "process_frame", original_process
        )
        return original_process(self, *args, **kwargs)

    def isolate_with_capture(value):
        trial = original_isolate(value)
        monkeypatch.setattr(
            TemporalCurrentRuntime, "process_frame", capture_process
        )
        return trial

    monkeypatch.setattr(module, "isolated_temporal_runtime", isolate_with_capture)
    monkeypatch.setattr(
        module,
        "build_proposal_recovery_evidence",
        lambda *_: (_ for _ in ()).throw(AssertionError("control built evidence")),
    )

    if temporal_only:
        result = runtime.process_temporal_only_frame(frame, observations, dense)
    else:
        result = runtime.process_frame(frame, observations, dense).temporal

    assert passed_evidence == [None]
    assert result.proposal_opportunity_count == 0
    assert result.proposal_trigger_count == 0


def test_temporal_only_advance_leaves_cumulative_untouched() -> None:
    from src.oviv2.dual_readout import DualReadoutRuntime
    from src.oviv2.t1_exactness import cumulative_state_sha256

    runtime = DualReadoutRuntime(_cumulative(), _temporal())
    before = cumulative_state_sha256(runtime.cumulative)

    result = runtime.advance_temporal_only_frame(_frame())

    assert result.frame_id == 0
    assert result.export is not None and result.export.samples == ()
    assert cumulative_state_sha256(runtime.cumulative) == before
