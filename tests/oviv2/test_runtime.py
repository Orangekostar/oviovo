from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.runtime import Oviv2Runtime, Oviv2RuntimeConfig
from src.oviv2.snapshot import VoxelMapSnapshot
from src.oviv2.tracking import LocalTrackerConfig


def frame(frame_id: int = 0) -> Frame:
    depth = np.ones((32, 32), dtype=np.float32)
    return Frame(
        frame_id=frame_id,
        rgb=np.full((32, 32, 3), 128, dtype=np.uint8),
        depth=depth,
        pose=np.eye(4),
        intrinsics=CameraIntrinsics(30.0, 30.0, 15.5, 15.5, 32, 32),
        timestamp=float(frame_id),
    )


def observation(
    frame_id: int,
    kind: ObservationKind,
    semantic_id: int,
    keys: set[tuple[int, int, int]],
    *,
    image_feature: np.ndarray | None = None,
    text_feature: np.ndarray | None = None,
    feature_model_id: str | None = None,
    view_direction_xyz: tuple[float, float, float] | None = None,
    visible_pixel_count: int = 0,
    border_contact_fraction: float = 0.0,
) -> FrameObservation:
    points = np.asarray(sorted(keys), dtype=np.float64) * 0.05
    return FrameObservation(
        observation_id=frame_id * 100,
        frame_id=frame_id,
        timestamp=float(frame_id),
        kind=kind,
        label="wall" if kind is ObservationKind.STRUCTURE else "chair",
        semantic_id=semantic_id,
        confidence=0.9,
        mask=np.ones((2, 2), dtype=bool),
        bbox_xyxy=(0.0, 0.0, 2.0, 2.0),
        voxel_keys=frozenset(keys),
        centroid_xyz=tuple(points.mean(axis=0)),
        bounds_min_xyz=tuple(points.min(axis=0)),
        bounds_max_xyz=tuple(points.max(axis=0)),
        image_feature=image_feature,
        text_feature=text_feature,
        feature_model_id=feature_model_id,
        view_direction_xyz=view_direction_xyz,
        visible_pixel_count=visible_pixel_count,
        border_contact_fraction=border_contact_fraction,
    )


def test_geometry_integrates_even_when_observation_batch_is_empty() -> None:
    runtime = Oviv2Runtime("room0")

    result = runtime.process_frame(frame(), ())

    assert result.geometry_blocks_touched > 0
    assert runtime.geometry.active_block_count > 0
    assert runtime.registry.entities == {}


def test_structure_updates_semantics_but_never_creates_entity() -> None:
    runtime = Oviv2Runtime("room0")
    key = (0, 0, 20)

    runtime.process_frame(frame(), (observation(0, ObservationKind.STRUCTURE, 1, {key}),))

    assert runtime.evidence.semantic_candidates(key)[0].label_id == 1
    assert runtime.registry.entities == {}
    assert runtime.ownership.owner_of(key) is None


def test_tentative_object_cannot_write_entity_evidence() -> None:
    runtime = Oviv2Runtime(
        "room0",
        Oviv2RuntimeConfig(tracker=LocalTrackerConfig(confirm_hits=2)),
    )
    key = (0, 0, 20)

    runtime.process_frame(frame(), (observation(0, ObservationKind.OBJECT, 2, {key}),))

    assert runtime.evidence.entity_candidates(key) == ()
    assert runtime.registry.entities == {}


def test_confirmed_object_writes_evidence_and_reversible_owner() -> None:
    runtime = Oviv2Runtime(
        "room0",
        Oviv2RuntimeConfig(tracker=LocalTrackerConfig(confirm_hits=2)),
    )
    key = (0, 0, 20)
    runtime.process_frame(frame(0), (observation(0, ObservationKind.OBJECT, 2, {key}),))

    result = runtime.process_frame(frame(1), (observation(1, ObservationKind.OBJECT, 2, {key}),))

    assert result.accepted_entity_ids == (1,)
    assert runtime.evidence.semantic_candidates(key)[0].label_id == 2
    assert runtime.evidence.entity_candidates(key)[0].entity_id == 1
    assert runtime.ownership.owner_of(key).entity_id == 1


def test_recomputing_owner_retains_competing_entity_evidence() -> None:
    runtime = Oviv2Runtime("room0")
    key = (0, 0, 20)
    runtime.evidence.update_entity(key, 1, 2.0, 0.0, 0.0, 1)
    runtime.evidence.update_entity(key, 2, 1.0, 0.0, 0.0, 1)
    runtime.recompute_ownership((key,), revision=1)
    runtime.evidence.update_entity(key, 1, 0.0, 2.0, 1.0, 2)
    runtime.evidence.update_entity(key, 2, 2.0, 0.0, 1.0, 2)

    runtime.recompute_ownership((key,), revision=2)

    assert runtime.ownership.owner_of(key).entity_id == 2
    assert {item.entity_id for item in runtime.evidence.entity_candidates(key)} == {1, 2}


def test_only_absent_visibility_adds_negative_entity_evidence() -> None:
    runtime = Oviv2Runtime("room0")
    key = (0, 0, 20)
    runtime.evidence.update_entity(key, 1, 2.0, 0.0, 0.0, 1)
    runtime.ownership.assign(key, 1, 1.0, 1)

    runtime.apply_visibility(frame(), revision=2, entity_voxels={1: frozenset({key})})
    present = runtime.evidence.entity_candidates(key)[0]
    assert present.negative_support == 0.0

    absent_frame = frame(1)
    absent_frame.depth[:] = 2.0
    runtime.apply_visibility(absent_frame, revision=3, entity_voxels={1: frozenset({key})})
    absent = runtime.evidence.entity_candidates(key)[0]
    assert absent.negative_support > 0.0


def test_runtime_commit_round_trip_contains_only_voxel_layers(tmp_path: Path) -> None:
    runtime = Oviv2Runtime("room0")
    runtime.process_frame(frame(), ())

    committed = runtime.commit(tmp_path / "snapshot")
    restored = VoxelMapSnapshot.load(tmp_path / "snapshot")

    assert committed.metadata.revision == 1
    assert restored.metadata.scene_id == "room0"
    assert {path.name for path in (tmp_path / "snapshot").iterdir()} == {
        "metadata.json",
        "geometry.npz",
        "evidence.npz",
        "ownership.npz",
        "checksums.json",
    }
