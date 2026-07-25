from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.dual_readout import DualReadoutRuntime
from src.oviv2.evidence import EvidenceConfig
from src.oviv2.geometry import TsdfConfig
from src.oviv2.meshing import derive_labeled_mesh
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.reference_readout import LifecycleOverlayReadout, ReferenceCurrentReadout
from src.oviv2.runtime import Oviv2Runtime, Oviv2RuntimeConfig
from src.oviv2.temporal_config import (
    ExecutionProfile,
    TemporalAssociationConfig,
    TemporalGeometryConfig,
    TemporalLifecycleConfig,
    TemporalReadoutConfig,
)
from src.oviv2.temporal_runtime import TemporalCurrentRuntime
from src.oviv2.tracking import LocalTrackerConfig


def _temporal_config(profile: ExecutionProfile) -> TemporalReadoutConfig:
    return TemporalReadoutConfig(
        lifecycle=TemporalLifecycleConfig(0.0, 4.0, -5.0, 12.0, 100.0, 0.7, 0.3, 2, 1, 0.1, 1, 0.5, 8, 4),
        association=TemporalAssociationConfig(1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 2.0, 0.9, 0.9, 0.9),
        geometry=TemporalGeometryConfig(0.05, 4.0, 4, 128, 128, 256, 0, 100, 0.5, 0.1, 2.0),
        execution_profile=profile,
    )


def _readout(
    profile: ExecutionProfile, tracker: LocalTrackerConfig
) -> TemporalCurrentRuntime | ReferenceCurrentReadout | LifecycleOverlayReadout:
    config = _temporal_config(profile)
    if profile is ExecutionProfile.A0:
        return ReferenceCurrentReadout("scene", config)
    if profile is ExecutionProfile.A1:
        return LifecycleOverlayReadout("scene", config)
    return TemporalCurrentRuntime("scene", config, tracker)


def _frame(frame_id: int) -> Frame:
    return Frame(
        frame_id=frame_id,
        timestamp=float(frame_id),
        rgb=np.full((16, 16, 3), 80 + frame_id, dtype=np.uint8),
        depth=np.ones((16, 16), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(100.0, 100.0, 0.0, 0.0, 16, 16),
    )


def _observation(frame: Frame, kind: ObservationKind, observation_id: int) -> FrameObservation:
    mask = np.ones((16, 16), dtype=bool)
    label, semantic_id = ("wall", 1) if kind is ObservationKind.STRUCTURE else ("chair", 2)
    return FrameObservation(
        observation_id=observation_id,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        kind=kind,
        label=label,
        semantic_id=semantic_id,
        confidence=0.9,
        mask=mask,
        bbox_xyxy=(0.0, 0.0, 16.0, 16.0),
        voxel_keys=frozenset({(0, 0, 20)}),
        centroid_xyz=(0.0, 0.0, 1.0),
        bounds_min_xyz=(-0.2, -0.2, 0.9),
        bounds_max_xyz=(0.2, 0.2, 1.1),
        image_feature=np.asarray((1.0, 0.0), dtype=np.float32) if kind is ObservationKind.OBJECT else None,
        feature_model_id="fixture" if kind is ObservationKind.OBJECT else None,
        visible_pixel_count=int(mask.sum()),
    )


def _canonical_geometry(runtime: Oviv2Runtime) -> tuple[bytes, ...]:
    grid = runtime.geometry._grid
    active = grid.hashmap().active_buf_indices()
    keys = grid.hashmap().key_tensor()[active].numpy()
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    return (
        keys[order].tobytes(),
        *(grid.attribute(name)[active].numpy()[order].tobytes() for name in runtime.geometry._ATTRIBUTE_NAMES),
    )


def _block_fingerprint(blocks: dict[object, object]) -> tuple[object, ...]:
    return tuple(
        (key, tuple((name, value.dtype.str, value.shape, value.tobytes()) for name, value in sorted(vars(block).items())))
        for key, block in sorted(blocks.items())
    )


def _cumulative_fingerprint(runtime: Oviv2Runtime, registry_path: Path) -> tuple[object, ...]:
    runtime.registry.save(registry_path)
    return (
        runtime.revision,
        runtime.last_frame_id,
        runtime.last_timestamp,
        _canonical_geometry(runtime),
        _block_fingerprint(runtime.evidence._blocks),
        _block_fingerprint(runtime.ownership._blocks),
        runtime.ownership.records(),
        pickle.dumps(runtime.tracker, protocol=5),
        registry_path.read_bytes(),
    )


def _tree_bytes(path: Path) -> dict[str, tuple[str, bytes]]:
    return {
        item.relative_to(path).as_posix(): (hashlib.sha256(item.read_bytes()).hexdigest(), item.read_bytes())
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


@pytest.mark.parametrize("profile", tuple(ExecutionProfile))
def test_dual_readout_is_exactly_noninterfering_for_cumulative_t1(
    tmp_path: Path, profile: ExecutionProfile
) -> None:
    tracker = LocalTrackerConfig(confirm_hits=2, min_voxel_overlap=0.0, max_centroid_distance_m=2.0)
    config = Oviv2RuntimeConfig(
        tsdf=TsdfConfig(
            voxel_size_m=0.05,
            block_resolution=32,
            block_count=64,
        ),
        evidence=EvidenceConfig(block_resolution=32),
        tracker=tracker,
    )
    direct = Oviv2Runtime("scene", config)
    cumulative = Oviv2Runtime("scene", config)
    temporal = _readout(profile, tracker)
    dual = DualReadoutRuntime(cumulative, temporal)

    frames = tuple(_frame(frame_id) for frame_id in range(3))
    batches = (
        (_observation(frames[0], ObservationKind.STRUCTURE, 0), _observation(frames[0], ObservationKind.OBJECT, 1)),
        (_observation(frames[1], ObservationKind.STRUCTURE, 2), _observation(frames[1], ObservationKind.OBJECT, 3)),
        (),
    )
    before_frames = tuple((frame.rgb.tobytes(), frame.depth.tobytes(), frame.pose.tobytes()) for frame in frames)
    before_batches = tuple(tuple(id(item) for item in batch) for batch in batches)
    direct_results = tuple(direct.process_frame(frame, batch) for frame, batch in zip(frames, batches))
    dual_results = tuple(dual.process_frame(frame, batch) for frame, batch in zip(frames, batches))

    assert list(tmp_path.iterdir()) == []
    assert direct_results == tuple(result.cumulative for result in dual_results)
    assert direct_results[1].accepted_entity_ids
    assert dual_results[1].temporal.active_entity_ids
    assert dual_results[2].temporal.active_entity_ids
    assert dual_results[2].temporal.dormant_entity_ids == ()
    assert before_frames == tuple((frame.rgb.tobytes(), frame.depth.tobytes(), frame.pose.tobytes()) for frame in frames)
    assert before_batches == tuple(tuple(id(item) for item in batch) for batch in batches)
    assert _cumulative_fingerprint(direct, tmp_path / "direct-registry.jsonl") == _cumulative_fingerprint(cumulative, tmp_path / "dual-registry.jsonl")

    direct_dir = tmp_path / "direct"
    dual_dir = tmp_path / "dual"
    direct.commit(direct_dir)
    cumulative.commit(dual_dir)
    direct_tree = _tree_bytes(direct_dir)
    dual_tree = _tree_bytes(dual_dir)
    expected = {"metadata.json", "geometry.npz", "evidence.npz", "ownership.npz", "entities.jsonl", "checksums.json"}
    assert set(direct_tree) == set(dual_tree) == expected
    assert direct_tree == dual_tree

    direct_mesh = derive_labeled_mesh(direct.geometry, direct.evidence, direct.ownership, entity_semantics=direct.registry.semantic_labels())
    dual_mesh = derive_labeled_mesh(cumulative.geometry, cumulative.evidence, cumulative.ownership, entity_semantics=cumulative.registry.semantic_labels())
    assert direct.geometry.active_block_count > 0
    assert direct_mesh.vertices_xyz.shape[0] > 0
    assert direct_mesh.triangles.shape[0] > 0
    assert np.any(direct_mesh.semantic_ids != 0)
    assert np.any(direct_mesh.entity_ids != 0)
    for name in ("vertices_xyz", "triangles", "colors_rgb", "semantic_ids", "entity_ids", "semantic_confidence", "ownership_confidence"):
        left = getattr(direct_mesh, name)
        right = getattr(dual_mesh, name)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes() == right.tobytes()

    assert not any(item.is_dir() for item in tmp_path.iterdir() if item.name not in {"direct", "dual"})
