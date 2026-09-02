from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.evaluation.crove_runtime_attribution import (
    CroveRuntimeAttributionProxy,
    RuntimeAttributionCapture,
    publish_runtime_attribution,
)
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.temporal_config import (
    ExecutionProfile,
    TemporalAssociationConfig,
    TemporalGeometryConfig,
    TemporalLifecycleConfig,
    TemporalReadoutConfig,
)
from src.oviv2.temporal_runtime import TemporalCurrentRuntime
from src.oviv2.tracking import LocalTrackerConfig


def _config() -> TemporalReadoutConfig:
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
        execution_profile=ExecutionProfile.A4,
    )


def _runtime() -> TemporalCurrentRuntime:
    return TemporalCurrentRuntime(
        "scene",
        _config(),
        LocalTrackerConfig(
            confirm_hits=1,
            max_age_frames=20,
            min_voxel_overlap=0.0,
            max_centroid_distance_m=2.0,
        ),
    )


def _frame(frame_id: int, depth: float) -> Frame:
    return Frame(
        frame_id=frame_id,
        timestamp=float(frame_id),
        source_frame_id=frame_id,
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth=np.full((5, 5), depth, dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(4.0, 4.0, 2.0, 2.0, 5, 5),
    )


def _observation(frame: Frame, observation_id: int) -> FrameObservation:
    depth = float(frame.depth[2, 2])
    mask = np.zeros(frame.depth.shape, dtype=bool)
    mask[2, 2] = True
    return FrameObservation(
        observation_id=observation_id,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        kind=ObservationKind.OBJECT,
        label="chair",
        semantic_id=1,
        confidence=1.0,
        mask=mask,
        bbox_xyxy=(2.0, 2.0, 3.0, 3.0),
        voxel_keys=frozenset({(0, 0, round(depth * 10))}),
        centroid_xyz=(0.0, 0.0, depth),
        bounds_min_xyz=(-0.05, -0.05, depth - 0.05),
        bounds_max_xyz=(0.05, 0.05, depth + 0.05),
        image_feature=np.asarray([1.0, 0.0], dtype=np.float32),
        feature_model_id="test",
        visible_pixel_count=1,
    )


def test_proxy_records_complete_association_and_motion_without_state_change() -> None:
    direct = _runtime()
    observed = _runtime()
    capture = RuntimeAttributionCapture()
    proxy = CroveRuntimeAttributionProxy(observed, capture)

    for frame_id, depth in ((0, 1.0), (1, 1.2)):
        frame = _frame(frame_id, depth)
        observation = _observation(frame, 10 + frame_id)
        direct_result = direct.process_frame(frame, (observation,))
        proxy_result = proxy.process_frame(frame, (observation,))
        assert proxy_result == direct_result
        assert observed.state.canonical_dump() == direct.state.canonical_dump()

    assert len(capture.records) == 2
    birth = capture.records[0]["association"]
    assert birth["observation_count"] == 1
    assert birth["assignment_count"] == 0
    assert birth["new_identity_births"] == [1]

    continuation = capture.records[1]
    association = continuation["association"]
    assert association["candidate_active_edge_count"] == 1
    assert association["candidate_dormant_edge_count"] == 0
    assert association["assignment_count"] == 1
    assert association["assignments"][0]["observation_id"] == 11
    assert association["assignments"][0]["entity_id"] == 1
    assert association["new_identity_births"] == []
    assert len(continuation["motion"]) == 1
    motion = continuation["motion"][0]
    assert motion["observation_id"] == 11
    assert motion["entity_id"] == 1
    assert motion["geometry_evidence_admitted"] is True
    assert motion["displacement_floor_m"] == 0.1
    assert motion["minimum_motion_confidence"] == 0.7
    assert motion["dynamic_state_before"] == "static"
    assert motion["dynamic_state_after"] in {"static", "dynamic"}
    assert motion["motion_streak_after"] >= 0


def test_proxy_restores_runtime_symbols_when_wrapped_call_fails() -> None:
    import src.oviv2.temporal_association as association_module
    import src.oviv2.temporal_runtime as runtime_module

    originals = {
        "association": runtime_module.associate_temporal_observations,
        "solve": association_module.solve_assignment,
        "backproject": runtime_module.backproject_observation,
        "motion": runtime_module.estimate_object_motion,
        "translation": runtime_module.estimate_object_translation,
        "route": runtime_module.route_active_motion_evidence,
        "dynamic": runtime_module.advance_dynamic_state,
    }

    class FailingRuntime:
        def process_frame(self, frame: Frame) -> None:
            runtime_module.associate_temporal_observations(
                (), (), _config().association
            )
            raise RuntimeError("expected")

    proxy = CroveRuntimeAttributionProxy(FailingRuntime(), RuntimeAttributionCapture())
    with pytest.raises(RuntimeError, match="expected"):
        proxy.process_frame(_frame(0, 1.0))
    assert runtime_module.associate_temporal_observations is originals["association"]
    assert association_module.solve_assignment is originals["solve"]
    assert runtime_module.backproject_observation is originals["backproject"]
    assert runtime_module.estimate_object_motion is originals["motion"]
    assert runtime_module.estimate_object_translation is originals["translation"]
    assert runtime_module.route_active_motion_evidence is originals["route"]
    assert runtime_module.advance_dynamic_state is originals["dynamic"]


def test_publish_runtime_attribution_is_deterministic_and_hash_bound(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    run_manifest = run_root / "run_manifest.json"
    run_manifest.write_text('{"scene":"apartment","status":"PASS"}\n', encoding="utf-8")
    records = [
        {
            "schema_version": 1,
            "frame_index": 0,
            "association": {},
            "motion": [],
        }
    ]

    first = publish_runtime_attribution(
        tmp_path / "first",
        records=records,
        run_manifest=run_manifest,
        instrumentation_sources={"test_source": Path(__file__)},
    )
    second = publish_runtime_attribution(
        tmp_path / "second",
        records=records,
        run_manifest=run_manifest,
        instrumentation_sources={"test_source": Path(__file__)},
    )

    assert (first / "runtime_attribution.jsonl").read_bytes() == (
        second / "runtime_attribution.jsonl"
    ).read_bytes()
    first_manifest = (first / "manifest.json").read_text(encoding="utf-8")
    second_manifest = (second / "manifest.json").read_text(encoding="utf-8")
    assert first_manifest == second_manifest
    assert '"formal_run_manifest"' in first_manifest
    assert '"instrumentation_sources":{"test_source"' in first_manifest
    with pytest.raises(FileExistsError):
        publish_runtime_attribution(first, records=records, run_manifest=run_manifest)
