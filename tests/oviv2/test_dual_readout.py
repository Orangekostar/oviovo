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
